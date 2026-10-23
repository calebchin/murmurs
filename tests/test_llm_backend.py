"""P2-06: GPT-OSS 120B through a local vLLM server over a Unix socket (D6, D22)."""

import fnmatch
import hashlib
import http.server
import json
import logging
import os
import socketserver
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import SENTINEL, StubJob
from pydantic import BaseModel
from typer.testing import CliRunner

from murmurs.backends import LLMClient, gptoss, registry
from murmurs.backends.gptoss import (
    GUARD_DIR,
    LLM_INVALID_OUTPUT,
    LLM_REQUEST_FAILED,
    MODEL_ID,
    SERVER_FILES,
    SPEC,
    VLLM_VERSION,
    GptOssLLM,
    LLMServer,
    UnixSocketChatClient,
    server_command,
    server_environment,
)
from murmurs.backends.registry import BackendKind
from murmurs.backends.stubs import StubLLMClient
from murmurs.cli import app
from murmurs.config import LLMServerConfig, load_config
from murmurs.errors import LLM_OUTPUT_TRUNCATED, BackendSetupError, PipelineError


class Answer(BaseModel):
    speaker: str
    confidence: float


def _completion(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def _mock_client(
    responses: list[Any], seen: list[dict[str, Any]] | None = None
) -> UnixSocketChatClient:
    """A client whose transport answers with ``responses`` in turn (str = content)."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        item = queue.pop(0)
        if isinstance(item, httpx.Response):
            return item
        if isinstance(item, Exception):
            raise item
        return httpx.Response(200, json=_completion(item))

    return UnixSocketChatClient(
        Path("/unused.sock"), transport=httpx.MockTransport(handler), max_attempts=3
    )


GOOD = json.dumps({"speaker": "SPEAKER_00", "confidence": 0.9})


# --- the stub -------------------------------------------------------------------------------


def test_stub_llm_satisfies_protocol_and_is_deterministic() -> None:
    stub = StubLLMClient(response={"speaker": "SPEAKER_01", "confidence": 1.0})
    assert isinstance(stub, LLMClient)
    first = stub.generate("p", Answer, "f1")
    assert (
        first
        == stub.generate("other prompt", Answer, "f2")
        == Answer(speaker="SPEAKER_01", confidence=1.0)
    )


# --- the client -----------------------------------------------------------------------------


def test_structured_request_returns_validated_object() -> None:
    seen: list[dict[str, Any]] = []
    client = _mock_client([GOOD], seen)
    result = client.generate("who speaks?", Answer, system="you label speakers")
    assert result == Answer(speaker="SPEAKER_00", confidence=0.9)
    body = seen[0]
    assert body["model"] == MODEL_ID and body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()
    assert client.usage.requests == 1 and client.usage.prompt_tokens == 11


def test_invalid_output_is_retried_then_succeeds() -> None:
    client = _mock_client(["not json", json.dumps({"speaker": "S"}), GOOD])
    assert client.generate("p", Answer).confidence == 0.9
    assert client.usage.requests == 3 and client.usage.invalid_outputs == 2


def test_request_carries_the_output_cap_and_reasoning_effort() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(GOOD))

    client = UnixSocketChatClient(
        Path("/unused.sock"),
        transport=httpx.MockTransport(handler),
        max_output_tokens=1234,
        reasoning_effort="low",
    )
    client.generate("p", Answer)
    assert seen[0]["max_tokens"] == 1234 and seen[0]["reasoning_effort"] == "low"


def test_truncated_response_is_not_retried() -> None:
    truncated = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": '{"speaker": "SPEAK'}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 4096},
        },
    )
    client = _mock_client([truncated, GOOD])
    with pytest.raises(PipelineError) as info:
        client.generate("p", Answer)
    assert info.value.code == LLM_OUTPUT_TRUNCATED and info.value.__cause__ is None
    assert client.usage.requests == 1 and client.usage.truncated == 1
    assert client.usage.completion_tokens == 4096


def test_usage_counts_stay_exact_under_concurrent_requests() -> None:
    import concurrent.futures

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(GOOD))

    client = UnixSocketChatClient(Path("/unused.sock"), transport=httpx.MockTransport(handler))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: client.generate("p", Answer), range(40)))
    assert client.usage.requests == 40 and client.usage.prompt_tokens == 40 * 11


def test_invalid_output_gives_typed_error_without_content() -> None:
    bad = json.dumps({"speaker": SENTINEL})  # missing a field
    client = _mock_client([bad, f"oops {SENTINEL}", bad])
    with pytest.raises(PipelineError) as info:
        client.generate(f"prompt {SENTINEL}", Answer)
    error = info.value
    assert error.code == LLM_INVALID_OUTPUT and error.detail == ""
    assert error.__cause__ is None and error.__context__ is None
    assert SENTINEL not in str(error) and client.usage.requests == 3


@pytest.mark.parametrize(
    "failure",
    [httpx.Response(500, text=f"server error {SENTINEL}"), httpx.ConnectError(SENTINEL)],
    ids=["http-500", "connect-error"],
)
def test_request_failure_gives_typed_error_without_content(failure: Any) -> None:
    client = _mock_client([failure])
    with pytest.raises(PipelineError) as info:
        client.generate("p", Answer)
    assert info.value.code == LLM_REQUEST_FAILED and SENTINEL not in str(info.value)
    assert info.value.__cause__ is None


def test_sentinel_in_prompt_and_response_never_logged(
    caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]
) -> None:
    caplog.set_level(logging.DEBUG)
    reply = json.dumps({"speaker": SENTINEL, "confidence": 0.5})
    client = _mock_client([reply, "not json", "nope", "still no"])
    assert client.generate(f"my name is {SENTINEL}", Answer, system=SENTINEL).speaker == SENTINEL
    with pytest.raises(PipelineError):
        client.generate(f"again {SENTINEL}", Answer)
    out, err = capfd.readouterr()
    assert SENTINEL not in caplog.text and SENTINEL not in out and SENTINEL not in err


@pytest.mark.parametrize(
    "address",
    ["http://127.0.0.1:8000", "localhost:8000", Path("127.0.0.1:8000"), Path("rel/vllm.sock")],
)
def test_client_refuses_anything_but_a_unix_socket_path(address: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        UnixSocketChatClient(address)


def test_client_refuses_a_path_that_is_not_a_socket(tmp_path: Path) -> None:
    regular = tmp_path / "file.sock"
    regular.write_text("")
    with pytest.raises(ValueError, match="not a Unix socket"):
        UnixSocketChatClient(regular)


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps(_completion(GOOD)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass

    def address_string(self) -> str:  # a Unix socket peer has no host address
        return "unix"


@pytest.fixture
def unix_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    path = Path(os.path.join("/tmp", f"murmurs-test-{os.getpid()}.sock"))
    path.unlink(missing_ok=True)
    server = socketserver.UnixStreamServer(str(path), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield path
    server.shutdown()
    server.server_close()
    path.unlink(missing_ok=True)


def test_client_talks_over_a_real_unix_socket(unix_server: Path) -> None:
    # The autouse network guard blocks every TCP connection, so this can only work
    # over the Unix socket.
    client = UnixSocketChatClient(unix_server)
    assert client.generate("p", Answer).speaker == "SPEAKER_00"
    client.close()


# --- server command, environment and network guard ------------------------------------------


def test_server_command_uses_only_the_unix_socket(tmp_path: Path) -> None:
    settings = LLMServerConfig(gpu_memory_utilization=0.7, max_model_len=8192)
    command = server_command(Path("/env/bin/vllm"), tmp_path, Path("/tmp/x/vllm.sock"), settings)
    assert command[:3] == ["/env/bin/vllm", "serve", str(tmp_path)]
    assert command[command.index("--uds") + 1] == "/tmp/x/vllm.sock"
    assert command[command.index("--served-model-name") + 1] == MODEL_ID
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.7"
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert "--host" not in command and "--port" not in command


def test_server_environment_is_offline_guarded_and_tokenless() -> None:
    base = {"PATH": "/usr/bin", "HF_TOKEN": "hf_secret", "PYTHONPATH": "/x", "HOME": "/h"}
    env = server_environment(base, Path("/env/bin"), Path("/vocab"))
    assert "HF_TOKEN" not in env and env["HOME"] == "/h"
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
        assert env[key] == "1"
    assert env["FLASHINFER_NO_DOWNLOAD"] == "1"
    assert env["TIKTOKEN_ENCODINGS_BASE"] == "/vocab"
    assert env["PYTHONPATH"] == str(GUARD_DIR)
    assert env["PATH"].split(os.pathsep)[0] == "/env/bin"


def _run_guarded(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(GUARD_DIR)}
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )


def test_network_guard_blocks_tcp_and_remote_lookups_in_server_processes() -> None:
    result = _run_guarded(
        "import socket\n"
        "for target in [('127.0.0.1', 9), ('1.1.1.1', 443)]:\n"
        "    try:\n"
        "        socket.create_connection(target, timeout=1)\n"
        "        print('CONNECTED')\n"
        "    except OSError as exc:\n"
        "        print(type(exc).__name__)\n"
        "try:\n"
        "    socket.getaddrinfo('edge.urm.nvidia.com', 443)\n"
        "    print('RESOLVED')\n"
        "except socket.gaierror:\n"
        "    print('gaierror')\n"
        "a, b = socket.socketpair(socket.AF_UNIX)\n"
        "a.sendall(b'ok'); print(b.recv(2).decode())\n"
    )
    # 127.0.0.1 passes the lookup and is refused at connect; a remote address is
    # already refused at the lookup inside create_connection.
    assert result.stdout.split() == ["NetworkBlockedError", "gaierror", "gaierror", "ok"]


# --- the server process, with a fake vLLM ----------------------------------------------------

FAKE_VLLM = r"""
import http.server, json, os, signal, socketserver, subprocess, sys, time
mode = os.environ.get("FAKE_MODE", "ok")
sock = sys.argv[sys.argv.index("--uds") + 1]
print("prompt leak SENTINEL_PHI_7f3a", flush=True)
print("stderr leak SENTINEL_PHI_7f3a", file=sys.stderr, flush=True)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
with open(os.environ["FAKE_CHILD_FILE"], "w") as f:
    f.write(str(child.pid))
if mode == "exit":
    sys.exit(3)
if mode == "ignore_sigint":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
if mode == "hang":
    time.sleep(600)

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"message": {"content": json.dumps(
            {"speaker": "SPEAKER_00", "confidence": 1.0})}}], "usage": {}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
    def address_string(self): return "unix"

socketserver.UnixStreamServer(sock, H).serve_forever()
"""


@pytest.fixture
def fake_vllm(tmp_path: Path) -> Path:
    path = tmp_path / "fake_vllm.py"
    path.write_text(FAKE_VLLM)
    return path


def _start_fake(fake: Path, tmp_path: Path, mode: str) -> tuple[LLMServer, Path]:
    child_file = tmp_path / f"child-{mode}.pid"
    env = {**os.environ, "FAKE_MODE": mode, "FAKE_CHILD_FILE": str(child_file)}
    server = LLMServer.start([sys.executable, str(fake), "serve", "m", "--uds", "{socket}"], env)
    return server, child_file


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _child_pid(child_file: Path) -> int:
    deadline = time.monotonic() + 20
    while not child_file.is_file() or not child_file.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    return int(child_file.read_text())


def test_server_starts_serves_and_stops_its_whole_process_group(
    fake_vllm: Path, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    server, child_file = _start_fake(fake_vllm, tmp_path, "ok")
    server.wait_ready(timeout_s=30, poll_s=0.1)
    assert oct(server.socket_dir.stat().st_mode & 0o777) == "0o700"
    assert len(str(server.socket_path)) <= gptoss.MAX_SOCKET_PATH
    llm = GptOssLLM(server, UnixSocketChatClient(server.socket_path), "0" * 40)
    assert llm.generate("p", Answer, "f1").speaker == "SPEAKER_00"
    child = _child_pid(child_file)

    llm.close()
    llm.close()  # a second close is harmless

    assert not _pid_alive(server.pid) and not _pid_alive(child)
    assert not server.socket_dir.exists()
    out, err = capfd.readouterr()
    assert SENTINEL not in out and SENTINEL not in err  # server output is discarded


def test_server_that_exits_during_start_up_fails_setup(fake_vllm: Path, tmp_path: Path) -> None:
    server, child_file = _start_fake(fake_vllm, tmp_path, "exit")
    with pytest.raises(BackendSetupError, match="exited during start-up \\(exit code 3\\)"):
        server.wait_ready(timeout_s=30, poll_s=0.1)
    assert not server.socket_dir.exists()
    assert not _pid_alive(_child_pid(child_file))  # its child was stopped too


def test_server_not_ready_in_time_fails_setup_and_is_stopped(
    fake_vllm: Path, tmp_path: Path
) -> None:
    server, child_file = _start_fake(fake_vllm, tmp_path, "hang")
    with pytest.raises(BackendSetupError, match="not ready after"):
        server.wait_ready(timeout_s=1.5, poll_s=0.1)
    assert not _pid_alive(server.pid) and not _pid_alive(_child_pid(child_file))


def test_stop_escalates_when_the_server_ignores_sigint(fake_vllm: Path, tmp_path: Path) -> None:
    server, child_file = _start_fake(fake_vllm, tmp_path, "ignore_sigint")
    server.wait_ready(timeout_s=30, poll_s=0.1)
    child = _child_pid(child_file)
    started = time.monotonic()
    server.stop(grace_s=(0.5, 0.5, 2.0))
    assert time.monotonic() - started < 5
    assert not _pid_alive(server.pid) and not _pid_alive(child)


def test_leftover_servers_are_stopped_at_exit(fake_vllm: Path, tmp_path: Path) -> None:
    server, _ = _start_fake(fake_vllm, tmp_path, "ok")
    server.wait_ready(timeout_s=30, poll_s=0.1)
    gptoss._stop_leftover_servers()
    assert not _pid_alive(server.pid)


# --- weights and one-time setup (D22) ---------------------------------------------------------


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import huggingface_hub.constants

    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("MURMURS_CACHE", str(tmp_path / "murmurs-cache"))
    return tmp_path


def test_weights_skip_the_original_and_metal_checkpoints() -> None:
    def wanted(path: str) -> bool:
        return any(fnmatch.fnmatch(path, p) for p in SPEC.allow_patterns or ())

    assert wanted("model-00003-of-00014.safetensors") and wanted("chat_template.jinja")
    assert not wanted("original/model--00001-of-00007.safetensors")
    assert not wanted("metal/model.bin")


def test_registry_declares_weights_and_three_setup_steps(stub_job: StubJob) -> None:
    config = load_config(stub_job(deid={"llm_model": MODEL_ID}))
    plan = registry.required_weights(config, [BackendKind.LLM])
    assert plan.weights == [SPEC]
    labels = [m.label for m in plan.prepared]
    assert labels[0] == f"vLLM {VLLM_VERSION} server environment"
    assert "tokenizer vocabulary" in labels[1] and "warm-up" in labels[2]


def test_shipped_lock_pins_the_declared_vllm_version() -> None:
    assert f'"vllm=={VLLM_VERSION}"' in (SERVER_FILES / "pyproject.toml").read_text()
    lock = (SERVER_FILES / "uv.lock").read_text()
    assert f'name = "vllm"\nversion = "{VLLM_VERSION}"' in lock


def test_build_env_runs_frozen_uv_sync_and_marks_ready(
    caches: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: Any) -> None:
        calls.append(args)
        target = Path(args[args.index("--project") + 1])
        (target / ".venv" / "bin").mkdir(parents=True)
        (target / ".venv" / "bin" / "vllm").write_text("")

    monkeypatch.setattr(gptoss.shutil, "which", lambda name: "/usr/bin/uv")
    assert not gptoss.env_is_ready()
    gptoss.build_env(run=fake_run)
    assert gptoss.env_is_ready()
    target = gptoss.server_env_dir()
    assert calls == [["/usr/bin/uv", "sync", "--frozen", "--project", str(target)]]
    assert (target / "uv.lock").read_bytes() == (SERVER_FILES / "uv.lock").read_bytes()
    assert target.name == gptoss.server_lock_hash()


def test_tiktoken_is_hash_checked(caches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"fake vocabulary"
    with pytest.raises(RuntimeError, match="hash"):
        gptoss.fetch_tiktoken(fetch=lambda url: data)
    assert not gptoss.tiktoken_is_ready()
    monkeypatch.setattr(gptoss, "TIKTOKEN_SHA256", hashlib.sha256(data).hexdigest())
    gptoss.fetch_tiktoken(fetch=lambda url: data)
    assert gptoss.tiktoken_is_ready()
    (gptoss.tiktoken_dir() / gptoss.TIKTOKEN_FILE).write_bytes(b"tampered")
    assert not gptoss.tiktoken_is_ready()


def test_warm_up_pings_once_and_marks_done(caches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pings: list[str] = []

    class FakeLLM:
        client = type("C", (), {"ping": lambda self: pings.append("ping")})()

        def __enter__(self) -> "FakeLLM":
            return self

        def __exit__(self, *exc: object) -> None:
            pings.append("closed")

    monkeypatch.setattr(gptoss, "start_gpt_oss", lambda settings: FakeLLM())
    gptoss.server_env_dir().mkdir(parents=True)
    assert not gptoss.warm_is_ready()
    gptoss.warm_up(LLMServerConfig())
    assert pings == ["ping", "closed"] and gptoss.warm_is_ready()


def test_models_check_reports_missing_llm_setup(stub_job: StubJob, caches: Path) -> None:
    config_path = stub_job(deid={"llm_model": MODEL_ID})
    result = CliRunner().invoke(app, ["models", "check", "-c", str(config_path)])
    assert result.exit_code == 2
    assert f"missing      {MODEL_ID}" in result.stdout
    assert result.stdout.count("unprepared") == 3


def test_starting_without_setup_fails_with_a_setup_error(caches: Path) -> None:
    with pytest.raises(BackendSetupError, match="murmurs models download"):
        gptoss.start_gpt_oss(LLMServerConfig())


# --- real model (only with -m gpu) ----------------------------------------------------------


@pytest.mark.gpu
def test_real_server_answers_a_structured_request_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from murmurs.offline import enforce_offline

    enforce_offline()
    with gptoss.start_gpt_oss(LLMServerConfig()) as llm:
        answer = llm.generate(
            "Segment 1: 'So tell me about your childhood home.' "
            "Segment 2: 'We lived on a farm.' Which segment's speaker is the interviewer? "
            "Use SPEAKER_00 for segment 1 and SPEAKER_01 for segment 2.",
            Answer,
            "f1",
            system="You label speakers in research interviews.",
        )
        assert answer.speaker in ("SPEAKER_00", "SPEAKER_01")
        assert 0.0 <= answer.confidence <= 1.0
        assert llm.client.usage.requests >= 1
        pid = llm.server.pid
    assert not os.path.exists(f"/proc/{pid}")
