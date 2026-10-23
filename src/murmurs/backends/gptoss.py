"""GPT-OSS 120B through a local vLLM server, reached only over a Unix socket (D6, D22).

The server runs from its own environment (built by ``murmurs models download``) in its
own process group. It listens on a Unix socket in a private temporary folder and runs
with a network guard that refuses every non-Unix connection, so prompts and responses
never leave the operating system. Its stdout and stderr are discarded, since vLLM can
print request content on errors.

``httpx`` is imported when the client is created, never when this module is imported.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from murmurs.backends import M
from murmurs.config import JobConfig, LLMServerConfig
from murmurs.errors import (
    LLM_INVALID_OUTPUT,
    LLM_OUTPUT_TRUNCATED,
    BackendSetupError,
    PipelineError,
)
from murmurs.models import LLMUsage
from murmurs.offline import OFFLINE_ENV, PreparedModel, WeightSpec, cached_snapshot

MODEL_ID = "openai/gpt-oss-120b"
WEIGHT_PATTERNS = (
    "model-*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
"""The files vLLM loads; the repo's ``original/`` and ``metal/`` checkpoints are skipped."""
SPEC = WeightSpec(MODEL_ID, allow_patterns=WEIGHT_PATTERNS)

VLLM_VERSION = "0.31.0"
"""The vLLM version pinned in ``vllm_server/pyproject.toml`` and its lock."""
SERVER_FILES = Path(__file__).parent / "vllm_server"
"""Ships with murmurs: the server environment's pyproject.toml and uv.lock, and the guard."""
GUARD_DIR = SERVER_FILES / "guard"

TIKTOKEN_FILE = "o200k_base.tiktoken"
TIKTOKEN_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
TIKTOKEN_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
"""GPT-OSS's tokenizer vocabulary. vLLM's harmony encoding would download it on first
use; with ``TIKTOKEN_ENCODINGS_BASE`` set it reads this local copy and checks this hash."""

ENV_READY = ".murmurs_ready"
WARM_READY = ".murmurs_warm"
MAX_SOCKET_PATH = 100
"""Unix socket paths are limited to about 108 bytes; stay below with some margin."""
STOP_GRACE_S = (30.0, 15.0, 5.0)
"""How long to wait after SIGINT, then SIGTERM, then SIGKILL."""

LLM_REQUEST_FAILED = "LLM_REQUEST_FAILED"
"""``generate`` error code: the server could not be reached or rejected the request."""


# --- where things live --------------------------------------------------------------------


def cache_root() -> Path:
    """murmurs' own cache (``$MURMURS_CACHE``, default ``~/.cache/murmurs``)."""
    return Path(os.environ.get("MURMURS_CACHE", Path.home() / ".cache" / "murmurs"))


def server_lock_hash() -> str:
    """Identifies the pinned server environment; a new lock gets a new environment."""
    digest = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        digest.update((SERVER_FILES / name).read_bytes())
    return digest.hexdigest()[:16]


def server_env_dir() -> Path:
    """The built server environment for the shipped lock file."""
    return cache_root() / "vllm-server" / server_lock_hash()


def vllm_executable() -> Path:
    """The ``vllm`` command inside the server environment."""
    return server_env_dir() / ".venv" / "bin" / "vllm"


def tiktoken_dir() -> Path:
    """Folder holding the tokenizer vocabulary (``TIKTOKEN_ENCODINGS_BASE``)."""
    return cache_root() / "tiktoken"


# --- one-time setup, run by `murmurs models download` (D22) ------------------------------


def env_is_ready() -> bool:
    """Whether the server environment for the shipped lock file is built."""
    return (server_env_dir() / ENV_READY).is_file() and vllm_executable().is_file()


def build_env(run: Callable[..., Any] = subprocess.run) -> None:
    """Build the server environment with ``uv sync --frozen`` (needs network access)."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH")
    target = server_env_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(SERVER_FILES / name, target / name)
    env = {
        k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
    }
    run([uv, "sync", "--frozen", "--project", str(target)], check=True, env=env)
    (target / ENV_READY).touch()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tiktoken_is_ready() -> bool:
    """Whether the tokenizer vocabulary is present with the expected hash."""
    path = tiktoken_dir() / TIKTOKEN_FILE
    return path.is_file() and _sha256(path) == TIKTOKEN_SHA256


def fetch_tiktoken(fetch: Callable[[str], bytes] | None = None) -> None:
    """Download the tokenizer vocabulary and check its hash (needs network access)."""
    if fetch is None:
        fetch = _http_get
    data = fetch(TIKTOKEN_URL)
    if hashlib.sha256(data).hexdigest() != TIKTOKEN_SHA256:
        raise RuntimeError("tokenizer vocabulary failed its hash check")
    folder = tiktoken_dir()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{TIKTOKEN_FILE}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, folder / TIKTOKEN_FILE)


def _http_get(url: str) -> bytes:
    import httpx

    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return response.content


def warm_is_ready() -> bool:
    """Whether the server has started once in this environment (kernels compiled)."""
    return (server_env_dir() / WARM_READY).is_file()


def warm_up(settings: LLMServerConfig) -> None:
    """Start the server once and send one request, so GPU kernels compile and are cached
    now rather than during a run. Runs offline; needs the GPU."""
    with start_gpt_oss(settings) as llm:
        llm.client.ping()
    (server_env_dir() / WARM_READY).touch()


def gpt_oss_weights(config: JobConfig) -> Sequence[WeightSpec]:
    """The model weights to download and check."""
    return [SPEC]


def gpt_oss_prepared(config: JobConfig) -> Sequence[PreparedModel]:
    """Server environment, tokenizer vocabulary and warm-up, in the order they must run."""
    settings = config.deid.llm_server
    return [
        PreparedModel(f"vLLM {VLLM_VERSION} server environment", env_is_ready, build_env),
        PreparedModel(f"{MODEL_ID} tokenizer vocabulary", tiktoken_is_ready, fetch_tiktoken),
        PreparedModel(
            f"{MODEL_ID} vLLM warm-up (compiles GPU kernels; needs the GPU)",
            warm_is_ready,
            lambda: warm_up(settings),
        ),
    ]


# --- the server process ----------------------------------------------------------------------


def server_command(
    executable: Path, model_path: Path, socket_path: Path, settings: LLMServerConfig
) -> list[str]:
    """``vllm serve`` on a Unix socket only; host and port are never set."""
    return [
        str(executable),
        "serve",
        str(model_path),
        "--served-model-name",
        MODEL_ID,
        "--uds",
        str(socket_path),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--max-model-len",
        str(settings.max_model_len),
        "--disable-uvicorn-access-log",
    ]


def server_environment(
    base: Mapping[str, str], env_bin: Path, vocab_dir: Path, guard_dir: Path = GUARD_DIR
) -> dict[str, str]:
    """The server's environment variables: offline, no telemetry, no downloads, the
    network guard first on ``PYTHONPATH``, the environment's tools first on ``PATH``
    (FlashInfer needs its ``ninja`` and ``nvcc``), and no Hugging Face token."""
    dropped = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    env = {k: v for k, v in base.items() if k not in dropped}
    env.update(OFFLINE_ENV)
    env.update(
        {
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_DO_NOT_TRACK": "1",
            "DO_NOT_TRACK": "1",
            "FLASHINFER_NO_DOWNLOAD": "1",
            "TIKTOKEN_ENCODINGS_BASE": str(vocab_dir),
            "PYTHONPATH": str(guard_dir),
            "PATH": os.pathsep.join([str(env_bin), base.get("PATH", os.defpath)]),
        }
    )
    return env


_running: weakref.WeakSet[LLMServer] = weakref.WeakSet()


@atexit.register
def _stop_leftover_servers() -> None:
    """Safety net: never leave a server holding the GPU after murmurs exits."""
    for server in list(_running):
        server.stop()


def _socket_dir() -> Path:
    """A private (mode 700) folder for the socket, with a path short enough for AF_UNIX."""
    base = tempfile.gettempdir()
    if len(base) > MAX_SOCKET_PATH - 30:
        base = "/tmp"
    return Path(tempfile.mkdtemp(prefix="murmurs-vllm-", dir=base))


class LLMServer:
    """A vLLM server process in its own process group, listening on a Unix socket."""

    def __init__(self, command: list[str], env: Mapping[str, str], socket_dir: Path) -> None:
        self.socket_dir = socket_dir
        self.socket_path = socket_dir / "vllm.sock"
        if len(str(self.socket_path)) > MAX_SOCKET_PATH:
            raise BackendSetupError("the LLM server's socket path is too long")
        self._process = subprocess.Popen(
            command,
            env=dict(env),
            cwd=socket_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,  # may contain prompts on errors: never kept
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # own process group, so stop() reaches every child
            close_fds=True,
        )
        self._stopped = False
        _running.add(self)

    @classmethod
    def start(cls, command: list[str], env: Mapping[str, str]) -> LLMServer:
        """Start the server. ``command`` must use ``{socket}`` where the socket path goes."""
        socket_dir = _socket_dir()
        socket_path = socket_dir / "vllm.sock"
        filled = [str(socket_path) if part == "{socket}" else part for part in command]
        try:
            return cls(filled, env, socket_dir)
        except BaseException:
            shutil.rmtree(socket_dir, ignore_errors=True)
            raise

    @property
    def pid(self) -> int:
        """Process ID of the server, which is also its process group ID."""
        return self._process.pid

    def wait_ready(self, timeout_s: float, poll_s: float = 1.0) -> None:
        """Wait until ``/health`` answers over the socket. Raises ``BackendSetupError`` if
        the server exits or is not ready in time; the server is stopped either way."""
        deadline = time.monotonic() + timeout_s
        while True:
            code = self._process.poll()
            if code is not None:
                self.stop()
                raise BackendSetupError(f"the LLM server exited during start-up (exit code {code})")
            if self.socket_path.exists() and _health_ok(self.socket_path):
                return
            if time.monotonic() >= deadline:
                self.stop()
                raise BackendSetupError(
                    f"the LLM server was not ready after {timeout_s:.0f} s "
                    "(deid.llm_server.startup_timeout_s)"
                )
            time.sleep(poll_s)

    def stop(self, grace_s: Sequence[float] = STOP_GRACE_S) -> None:
        """Stop the whole process group (SIGINT, then SIGTERM, then SIGKILL) and remove
        the socket folder. Safe to call more than once."""
        if self._stopped:
            return
        self._stopped = True
        _running.discard(self)
        try:
            for sig, wait_s in zip(
                (signal.SIGINT, signal.SIGTERM, signal.SIGKILL), grace_s, strict=False
            ):
                if not self._group_alive():
                    break
                try:
                    os.killpg(self.pid, sig)
                except ProcessLookupError:
                    break
                self._wait_group(wait_s)
        finally:
            if self._process.poll() is None:
                self._process.kill()
            self._process.wait()
            shutil.rmtree(self.socket_dir, ignore_errors=True)

    def _group_alive(self) -> bool:
        self._process.poll()  # reap the leader if it has exited
        try:
            os.killpg(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _wait_group(self, wait_s: float) -> None:
        deadline = time.monotonic() + wait_s
        while self._group_alive() and time.monotonic() < deadline:
            time.sleep(0.1)


def _health_ok(socket_path: Path) -> bool:
    """One ``GET /health`` over the socket, using only the standard library."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        try:
            sock.connect(str(socket_path))
            sock.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            status_line = sock.recv(64).split(b"\r\n", 1)[0].split()
            return (
                len(status_line) >= 2
                and status_line[0].startswith(b"HTTP/")
                and status_line[1] == b"200"
            )
        except OSError:
            return False


# --- the client -------------------------------------------------------------------------------


class UnixSocketChatClient:
    """OpenAI-style chat completions over a Unix domain socket, and nothing else.

    It accepts only an absolute path to an existing Unix socket, never a URL, host or
    port, so it cannot be pointed at a TCP address (hard rule 4). Safe to use from
    several threads at once.
    """

    def __init__(
        self,
        socket_path: Path,
        model: str = MODEL_ID,
        max_attempts: int = 3,
        timeout_s: float = 600.0,
        transport: Any = None,
        max_output_tokens: int = 4096,
        reasoning_effort: str = "medium",
    ) -> None:
        if not isinstance(socket_path, Path):
            raise TypeError("socket_path must be a pathlib.Path to a Unix socket")
        if not socket_path.is_absolute():
            raise ValueError("socket_path must be an absolute path to a Unix socket")
        if transport is None:
            try:
                is_socket = stat.S_ISSOCK(socket_path.stat().st_mode)
            except OSError:
                is_socket = False
            if not is_socket:
                raise ValueError("socket_path is not a Unix socket")
        import httpx

        self.model = model
        self.max_attempts = max_attempts
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.usage = LLMUsage()
        self._lock = threading.Lock()
        self._http = httpx.Client(
            transport=transport or httpx.HTTPTransport(uds=str(socket_path)),
            base_url="http://localhost",  # placeholder host; the transport is the socket
            timeout=timeout_s,
            trust_env=False,  # never pick up proxy settings from the environment
        )

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._http.close()

    def ping(self) -> None:
        """One tiny fixed request with no interview data (used by the warm-up)."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 16,
            "temperature": 0,
        }
        response = self._http.post("/v1/chat/completions", json=body)
        if response.status_code != 200:
            raise RuntimeError(f"warm-up request failed (HTTP {response.status_code})")

    def generate(self, prompt: str, response_model: type[M], system: str | None = None) -> M:
        """Return the response validated as ``response_model``.

        The server constrains output to the model's JSON schema; output that still
        fails validation is retried up to ``max_attempts`` times. A response cut off at
        ``max_output_tokens`` raises ``LLM_OUTPUT_TRUNCATED`` at once: the same request
        would run to the cap again. Failures raise ``PipelineError`` with a code only,
        never request or response content.
        """
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
        }
        import httpx

        for _ in range(self.max_attempts):
            start = time.perf_counter()
            try:
                response = self._http.post("/v1/chat/completions", json=body)
            except httpx.HTTPError:
                raise PipelineError(LLM_REQUEST_FAILED) from None
            finally:
                self._count(requests=1, seconds=time.perf_counter() - start)
            if response.status_code != 200:
                raise PipelineError(LLM_REQUEST_FAILED)
            try:
                payload = response.json()
                choice = payload["choices"][0]
                tokens = payload.get("usage") or {}
                self._count(
                    prompt_tokens=int(tokens.get("prompt_tokens", 0)),
                    completion_tokens=int(tokens.get("completion_tokens", 0)),
                )
                truncated = choice.get("finish_reason") == "length"
                content = choice["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                truncated, content = False, None
            if truncated:
                self._count(truncated=1)
                raise PipelineError(LLM_OUTPUT_TRUNCATED)
            try:
                return response_model.model_validate_json(content or "")
            except (ValueError, ValidationError):
                self._count(invalid_outputs=1)
                continue
        raise PipelineError(LLM_INVALID_OUTPUT)

    def _count(self, seconds: float = 0.0, **counts: int) -> None:
        with self._lock:
            update: dict[str, Any] = {
                name: getattr(self.usage, name) + value for name, value in counts.items()
            }
            update["seconds"] = self.usage.seconds + seconds
            self.usage = self.usage.model_copy(update=update)


# --- the LLMClient backend -------------------------------------------------------------------


class GptOssLLM:
    """``LLMClient`` for GPT-OSS 120B: owns the server and the client; ``close`` stops both."""

    def __init__(self, server: LLMServer, client: UnixSocketChatClient, revision: str) -> None:
        self.server = server
        self.client = client
        self.model_version = f"{MODEL_ID}@{revision[:12]} (vLLM {VLLM_VERSION})"

    @property
    def usage(self) -> LLMUsage:
        """Running totals of requests, tokens and time (counts only)."""
        return self.client.usage

    def generate(
        self, prompt: str, response_model: type[M], file_id: str, system: str | None = None
    ) -> M:
        """Run ``prompt`` and return the response validated as ``response_model``."""
        return self.client.generate(prompt, response_model, system=system)

    def close(self) -> None:
        """Close the client and stop the server, freeing the GPU."""
        try:
            self.client.close()
        finally:
            self.server.stop()

    def __enter__(self) -> GptOssLLM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def start_gpt_oss(settings: LLMServerConfig) -> GptOssLLM:
    """Start the server from the prepared environment and connect to it.

    Raises ``BackendSetupError`` if anything it needs is not prepared, or if the server
    does not come up; a server that was started is stopped first.
    """
    snapshot = cached_snapshot(SPEC)
    if snapshot is None:
        raise BackendSetupError(f"{MODEL_ID} weights are not cached; run `murmurs models download`")
    if not env_is_ready() or not tiktoken_is_ready():
        raise BackendSetupError("the LLM server is not set up; run `murmurs models download`")
    command = server_command(vllm_executable(), snapshot, Path("{socket}"), settings)
    env = server_environment(os.environ, vllm_executable().parent, tiktoken_dir())
    server = LLMServer.start(command, env)
    try:
        server.wait_ready(settings.startup_timeout_s)
        client = UnixSocketChatClient(
            server.socket_path,
            max_output_tokens=settings.max_output_tokens,
            reasoning_effort=settings.reasoning_effort,
        )
    except BaseException:
        server.stop()
        raise
    return GptOssLLM(server, client, snapshot.name)


def create_gpt_oss(config: JobConfig) -> GptOssLLM:
    """Registry factory."""
    return start_gpt_oss(config.deid.llm_server)


if TYPE_CHECKING:
    from murmurs.backends import LLMClient

    def _check(config: JobConfig) -> LLMClient:
        # Static proof (checked by mypy) that the backend satisfies its protocol.
        return create_gpt_oss(config)
