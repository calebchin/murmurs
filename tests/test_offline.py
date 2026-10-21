"""P2-00: offline setup."""

import io
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import huggingface_hub
import huggingface_hub.constants
import pytest
from conftest import NetworkBlockedError, StubJob
from typer.testing import CliRunner, Result

from murmurs.backends import registry
from murmurs.backends.registry import BackendKind
from murmurs.backends.stubs import StubTranscriber
from murmurs.cli import app
from murmurs.config import load_config
from murmurs.offline import (
    OFFLINE_ENV,
    MissingWeightsError,
    WeightDownloadError,
    WeightSpec,
    cached_revision,
    check_cached,
    download_weight,
    enforce_offline,
)
from murmurs.runner import run_job
from murmurs.store import ArtifactStore

FAKE_REPO = "murmurs-test/fake-model"
OTHER_REPO = "murmurs-test/other-model"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
WEIGHTED = "stub-weighted"
DUMMY_TOKEN = "hf_DUMMYtokenVALUE9f2c"


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    yield


@pytest.fixture
def hub_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty Hugging Face cache that ``huggingface_hub`` uses for this test."""
    cache = tmp_path / "hf-cache" / "hub"
    cache.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    return cache


def _cache_repo(cache: Path, repo_id: str = FAKE_REPO, files: dict[str, str] | None = None) -> Path:
    """Lay out a cached snapshot the way ``huggingface_hub`` does; returns its folder."""
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(COMMIT)
    snapshot = folder / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    for name, text in (files if files is not None else {"config.json": "{}"}).items():
        (snapshot / name).write_text(text)
    return snapshot


def _register_weighted(*repos: str) -> None:
    """A stub transcriber called ``stub-weighted`` that declares ``repos`` as weights."""
    registry.register(
        BackendKind.TRANSCRIBER,
        WEIGHTED,
        lambda _: StubTranscriber(),
        weights=lambda _: [WeightSpec(repo) for repo in repos],
    )


def _cli(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


# --- forced offline mode ------------------------------------------------------------------


def test_enforce_offline_sets_env_and_hub_flag() -> None:
    assert not huggingface_hub.constants.is_offline_mode()
    enforce_offline()
    for name, value in OFFLINE_ENV.items():
        assert os.environ[name] == value
    assert huggingface_hub.constants.is_offline_mode()


def test_runner_enforces_offline_before_creating_backends(stub_job: StubJob) -> None:
    seen: dict[str, Any] = {}

    def factory(_: object) -> StubTranscriber:
        seen["env"] = os.environ.get("HF_HUB_OFFLINE")
        seen["hub"] = huggingface_hub.constants.is_offline_mode()
        return StubTranscriber()

    registry.register(BackendKind.TRANSCRIBER, "stub-probe", factory)
    config = load_config(stub_job(transcription={"model": "stub-probe"}))
    summary = run_job(config, "job", console=io.StringIO())
    assert summary.exit_code == 0
    assert seen == {"env": "1", "hub": True}


def test_importing_murmurs_does_not_import_model_libraries() -> None:
    watched = ("huggingface_hub", "transformers", "pyannote", "crisperwhisper")
    code = (
        "import sys\n"
        "import murmurs, murmurs.cli, murmurs.runner, murmurs.offline\n"
        "import murmurs.backends.registry\n"
        f"print(','.join(m for m in {watched!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip() == ""


# --- weight declarations ---------------------------------------------------------------------


def test_required_weights_dedups_and_reports_unavailable(stub_job: StubJob) -> None:
    _register_weighted(FAKE_REPO, OTHER_REPO)
    registry.register(
        BackendKind.DIARIZER,
        "stub-weighted-d",
        lambda _: None,
        weights=lambda _: [WeightSpec(FAKE_REPO)],
    )
    config = load_config(
        stub_job(
            transcription={"model": WEIGHTED},
            diarization={"model": "stub-weighted-d"},
            deid={"ner_model": "not-registered"},
        )
    )
    plan = registry.required_weights(config, list(BackendKind))
    assert plan.weights == [WeightSpec(FAKE_REPO), WeightSpec(OTHER_REPO)]
    assert plan.unavailable == ["no ner backend named 'not-registered' is available"]


def test_stubs_declare_no_weights(stub_job: StubJob) -> None:
    plan = registry.required_weights(load_config(stub_job()), list(BackendKind))
    assert plan.weights == [] and plan.unavailable == []


def test_weight_label_includes_pinned_revision() -> None:
    assert WeightSpec(FAKE_REPO).label == FAKE_REPO
    assert WeightSpec(FAKE_REPO, revision="v1").label == f"{FAKE_REPO}@v1"


# --- cache checks (no network: the guard is active) ---------------------------------------------


def test_missing_weights_fail_check_and_run_before_any_file(
    stub_job: StubJob, hub_cache: Path
) -> None:
    _register_weighted(FAKE_REPO)
    config_path = stub_job(transcription={"model": WEIGHTED})

    check = _cli("models", "check", "-c", str(config_path))
    assert check.exit_code == 2
    assert f"missing      {FAKE_REPO}" in check.stdout

    run = _cli("run", "-c", str(config_path))
    assert run.exit_code == 2
    assert FAKE_REPO in run.stderr and "murmurs models download" in run.stderr
    config = load_config(config_path)
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    assert not (store.job_dir / "files").exists()


def test_cached_weights_pass_check_and_run_and_reach_manifest(
    stub_job: StubJob, hub_cache: Path
) -> None:
    _register_weighted(FAKE_REPO)
    _cache_repo(hub_cache)
    config_path = stub_job(transcription={"model": WEIGHTED})

    check = _cli("models", "check", "-c", str(config_path))
    assert check.exit_code == 0, check.output
    assert f"cached       {FAKE_REPO} ({COMMIT})" in check.stdout

    run = _cli("run", "-c", str(config_path))
    assert run.exit_code == 0, run.stderr
    config = load_config(config_path)
    manifest = ArtifactStore(config.data.output_dir, config.data.input_dir, "job").read_manifest()
    assert manifest is not None
    assert manifest.model_versions[FAKE_REPO] == COMMIT
    assert manifest.model_versions["transcriber"] == "stub"


def test_weights_outside_stage_range_are_not_required(stub_job: StubJob, hub_cache: Path) -> None:
    _register_weighted(FAKE_REPO)
    config_path = stub_job(transcription={"model": WEIGHTED}, stages={"end": "diarize"})
    run = _cli("run", "-c", str(config_path))
    assert run.exit_code == 0, run.stderr


def test_empty_snapshot_counts_as_missing(hub_cache: Path) -> None:
    _cache_repo(hub_cache, files={})
    assert cached_revision(WeightSpec(FAKE_REPO)) is None


def test_check_cached_names_every_missing_weight(hub_cache: Path) -> None:
    _cache_repo(hub_cache)
    assert check_cached([WeightSpec(FAKE_REPO)]) == {FAKE_REPO: COMMIT}
    with pytest.raises(MissingWeightsError) as info:
        check_cached([WeightSpec(FAKE_REPO), WeightSpec(OTHER_REPO, revision="v2")])
    assert [s.label for s in info.value.missing] == [f"{OTHER_REPO}@v2"]
    assert f"{OTHER_REPO}@v2" in str(info.value)


def test_check_reports_unregistered_backend(stub_job: StubJob) -> None:
    config_path = stub_job(deid={"ner_model": "not-registered"})
    check = _cli("models", "check", "-c", str(config_path))
    assert check.exit_code == 2
    assert "no ner backend named 'not-registered'" in check.stderr


# --- download (mocked; never touches the network) ------------------------------------------


def _fake_download(tmp_path: Path, calls: list[dict[str, Any]]) -> Any:
    def fake(repo_id: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        snapshot = tmp_path / "downloaded" / COMMIT
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "weights.bin").write_bytes(b"x" * 2048)
        return str(snapshot)

    return fake


def test_download_fetches_declared_weights_without_echoing_token(
    stub_job: StubJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_weighted(FAKE_REPO)
    monkeypatch.setenv("HF_TOKEN", DUMMY_TOKEN)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_download(tmp_path, calls))

    result = _cli("models", "download", "-c", str(stub_job(transcription={"model": WEIGHTED})))
    assert result.exit_code == 0, result.output
    assert f"ok           {FAKE_REPO} (2.0 kB, {COMMIT})" in result.stdout
    assert DUMMY_TOKEN not in result.output
    assert [c["repo_id"] for c in calls] == [FAKE_REPO]
    assert not calls[0].get("local_files_only")


def test_download_failure_never_echoes_library_message(
    stub_job: StubJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_weighted(FAKE_REPO)
    monkeypatch.setenv("HF_TOKEN", DUMMY_TOKEN)

    def failing(repo_id: str, **kwargs: Any) -> str:
        raise OSError(f"401 Unauthorized for token {DUMMY_TOKEN}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", failing)
    result = _cli("models", "download", "-c", str(stub_job(transcription={"model": WEIGHTED})))
    assert result.exit_code == 2
    assert "download failed (OSError)" in result.stderr
    assert DUMMY_TOKEN not in result.output


def test_download_refuses_when_offline_mode_is_on() -> None:
    enforce_offline()
    with pytest.raises(WeightDownloadError) as info:
        download_weight(WeightSpec(FAKE_REPO))
    assert info.value.code == "OFFLINE_MODE_SET"


def test_download_with_unregistered_backend_exits_2(stub_job: StubJob) -> None:
    result = _cli("models", "download", "-c", str(stub_job(deid={"llm_model": "not-registered"})))
    assert result.exit_code == 2
    assert "no llm backend named 'not-registered'" in result.stderr


# --- network guard ---------------------------------------------------------------------------


def test_network_guard_blocks_tcp() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("huggingface.co", 443)


def test_network_guard_allows_unix_sockets() -> None:
    left, right = socket.socketpair()
    with left, right:
        left.sendall(b"ok")
        assert right.recv(2) == b"ok"
