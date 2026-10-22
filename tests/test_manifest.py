"""P1-09: run manifest and sample selection."""

import io
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torchcodec
from conftest import SENTINEL, StubJob
from fixtures.audio import tone_wav_bytes
from typer.testing import CliRunner

from murmurs import __version__
from murmurs.backends.stubs import StubTranscriber
from murmurs.cli import app
from murmurs.config import JobConfig, Stage, load_config
from murmurs.manifest import code_version, dependency_versions, select_sample
from murmurs.models import RunManifest, StageStatus
from murmurs.runner import create_backends, run_job
from murmurs.stages import Backends
from murmurs.store import ArtifactStore

TEN_FILES = {f"f{i}.wav": tone_wav_bytes(frequency=200 + 50 * i) for i in range(10)}


def _manifest(config: JobConfig, job_id: str) -> RunManifest:
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, job_id)
    manifest = store.read_manifest()
    assert manifest is not None
    return manifest


def _with_sample(config: JobConfig, **values: object) -> JobConfig:
    sample = config.sample.model_copy(update={"enabled": True, **values})
    return config.model_copy(update={"sample": sample})


# --- manifest -------------------------------------------------------------------------


def test_manifest_records_resolved_config_with_cli_overrides(stub_job: StubJob) -> None:
    config_path = stub_job()
    result = CliRunner().invoke(
        app,
        [
            "run",
            "-c",
            str(config_path),
            "--end",
            "transcribe",
            "--set",
            "diarization.num_speakers=2",
        ],
    )
    assert result.exit_code == 0, result.stderr
    manifest = _manifest(load_config(config_path), "job")
    assert manifest.config.stages.end.value == "transcribe"
    assert manifest.config.diarization.num_speakers == 2


def test_manifest_records_versions_hashes_and_statuses(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    summary = run_job(config, "job", console=io.StringIO())
    manifest = _manifest(config, "job")

    assert manifest.job_id == "job"
    assert manifest.package_version == __version__
    assert manifest.dependency_versions["pydantic"]
    assert manifest.model_versions == {
        "audio_loader": f"torchcodec {torchcodec.__version__}",
        "diarizer": "stub",
        "transcriber": "stub",
        "ner": "stub",
    }
    assert set(manifest.input_hashes) == set(summary.done)
    assert all(h.startswith(fid) for fid, h in manifest.input_hashes.items())
    assert manifest.finished_at is not None and manifest.finished_at >= manifest.started_at
    assert {s.file_id for s in manifest.file_statuses} == set(summary.done)
    assert all(set(s.stages.values()) == {StageStatus.DONE} for s in manifest.file_statuses)


def test_manifest_only_records_backends_the_range_used(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    stages = config.stages.model_copy(update={"end": Stage.INGEST})
    run_job(config.model_copy(update={"stages": stages}), "job", console=io.StringIO())
    assert _manifest(config, "job").model_versions == {
        "audio_loader": f"torchcodec {torchcodec.__version__}"
    }


def _stub_backends(config: JobConfig) -> Backends:
    return create_backends(config, list(Stage))


def test_interrupted_job_leaves_manifest_unfinished(stub_job: StubJob) -> None:
    class Interrupting(StubTranscriber):
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

    config = load_config(stub_job())
    backends = _stub_backends(config)
    backends.transcriber = Interrupting()
    with pytest.raises(KeyboardInterrupt):
        run_job(config, "job", backends=backends, console=io.StringIO())
    manifest = _manifest(config, "job")
    assert manifest.finished_at is None
    [status] = manifest.file_statuses
    assert status.stages[Stage.LOAD_AUDIO] is StageStatus.DONE
    assert status.stages[Stage.TRANSCRIBE] is StageStatus.PENDING


def test_manifest_never_contains_sentinel(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    backends = _stub_backends(config)
    backends.transcriber = StubTranscriber(text=f"I am {SENTINEL}")
    run_job(config, "job", backends=backends, console=io.StringIO())
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    assert SENTINEL not in store.manifest_path.read_text()


def test_code_version_reads_git_commit(tmp_path: Path) -> None:
    assert code_version(tmp_path) is None  # not a git checkout
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "f.txt").write_text("1")
    subprocess.run([*git, "add", "f.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert code_version(repo) == head
    (repo / "f.txt").write_text("2")
    assert code_version(repo) == f"{head}-dirty"


def test_dependency_versions_skip_missing_packages() -> None:
    versions = dependency_versions()
    assert "numpy" in versions
    assert "crisperwhisper" not in versions  # not installed in phase 1


# --- sample selection -------------------------------------------------------------------


def test_same_seed_selects_same_files(stub_job: StubJob) -> None:
    config = _with_sample(load_config(stub_job(TEN_FILES)), n_files=3, seed=7)
    first = run_job(config, "job", console=io.StringIO())
    no_resume = config.stages.model_copy(update={"resume": False})
    second = run_job(config.model_copy(update={"stages": no_resume}), "job", console=io.StringIO())
    assert sorted(first.done) == sorted(second.done)
    assert len(first.done) == 3


def test_different_seed_can_select_different_files() -> None:
    keys = [f"id{i:02d}" for i in range(20)]
    picks = {tuple(select_sample(keys, keys, 5, seed)) for seed in range(10)}
    assert len(picks) > 1


def test_selection_ignores_input_order() -> None:
    keys = [f"id{i:02d}" for i in range(20)]
    assert select_sample(keys, keys, 5, 3) == select_sample(keys[::-1], keys[::-1], 5, 3)


def test_n_files_larger_than_inputs_selects_all(stub_job: StubJob) -> None:
    config = _with_sample(load_config(stub_job()), n_files=50)
    summary = run_job(config, "job", console=io.StringIO())
    assert len(summary.done) == 3 and summary.exit_code == 0


def test_sample_runs_use_separate_job_id(stub_job: StubJob) -> None:
    config_path = stub_job(TEN_FILES)
    result = CliRunner().invoke(app, ["run", "-c", str(config_path), "--sample", "2"])
    assert result.exit_code == 0, result.stderr
    config = load_config(config_path)
    assert (config.data.output_dir / "job-sample" / "manifest.json").is_file()
    assert not (config.data.output_dir / "job").exists()
    manifest = _manifest(config, "job-sample")
    assert manifest.config.sample.enabled and len(manifest.input_hashes) == 2
