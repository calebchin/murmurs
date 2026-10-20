"""P1-10: phase 1 acceptance test.

A job with stub stages runs end to end through the CLI, skips a deliberately failing
file, and resumes after an interruption without redoing completed stages.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from conftest import SENTINEL
from fixtures.audio import write_tone
from typer.testing import CliRunner, Result

from murmurs.backends import registry
from murmurs.backends.registry import BackendKind
from murmurs.backends.stubs import (
    StubAudioLoader,
    StubDiarizer,
    StubNERDetector,
    StubTranscriber,
)
from murmurs.cli import EXIT_INTERRUPTED, app
from murmurs.config import Stage
from murmurs.models import StageStatus
from murmurs.store import ArtifactStore, file_id_from_hash, hash_file

TRANSCRIPT_TEXT = f"my name is {SENTINEL} and I live nearby"


class StubSet:
    """One run's stub backends, registered under "stub" so the CLI picks them up."""

    def __init__(
        self,
        fail_transcribe_on: set[str] = frozenset(),
        interrupt_ner_on: set[str] = frozenset(),
    ) -> None:
        self.audio = StubAudioLoader()
        self.diarizer = StubDiarizer()
        self.transcriber = StubTranscriber(
            text=TRANSCRIPT_TEXT, fail_on=fail_transcribe_on, fail_detail=f"bad {SENTINEL}"
        )
        self.ner = StubNERDetector({SENTINEL: "first_name"}, interrupt_once=interrupt_ner_on)

    def install(self) -> None:
        registry.register(BackendKind.AUDIO_LOADER, "default", lambda _: self.audio)
        registry.register(BackendKind.DIARIZER, "stub", lambda _: self.diarizer)
        registry.register(BackendKind.TRANSCRIBER, "stub", lambda _: self.transcriber)
        registry.register(BackendKind.NER, "stub", lambda _: self.ner)

    def calls(self) -> dict[str, list[str]]:
        return {
            "audio": self.audio.calls,
            "diarizer": self.diarizer.calls,
            "transcriber": self.transcriber.calls,
            "ner": self.ner.calls,
        }


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    yield


@pytest.fixture
def job(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Three tone files (one in a subfolder) and a stub config. Returns config, in, out."""
    input_dir = tmp_path / "in"
    write_tone(input_dir / "interview_a.wav", seconds=1.5, frequency=440)
    write_tone(input_dir / "interview_b.wav", seconds=2.0, frequency=660)
    write_tone(input_dir / "site2" / "interview_c.wav", seconds=1.0, frequency=0)
    output_dir = tmp_path / "out"
    config = {
        "data": {"input_dir": str(input_dir), "output_dir": str(output_dir)},
        "diarization": {"model": "stub", "llm_correction": False},
        "transcription": {"model": "stub"},
        "deid": {
            "ner_model": "stub",
            "llm_model": "stub",
            "entity_types": [{"name": "PERSON", "tag": "[PERSON_{n}]"}],
        },
    }
    config_path = tmp_path / "job.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, input_dir, output_dir


def _snapshot(input_dir: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(input_dir.rglob("*"))
        if p.is_file()
    }


def _run(config_path: Path, stubs: StubSet) -> Result:
    stubs.install()
    return CliRunner().invoke(app, ["run", "--config", str(config_path)])


def test_phase1_end_to_end(job: tuple[Path, Path, Path]) -> None:
    config_path, input_dir, output_dir = job
    before = _snapshot(input_dir)
    ids = {p.name: file_id_from_hash(hash_file(p)) for p in input_dir.rglob("*.wav")}
    bad = ids["interview_b.wav"]
    good = {ids["interview_a.wav"], ids["interview_c.wav"]}
    store = ArtifactStore(output_dir, input_dir, "job")
    consoles: list[str] = []

    # Run 1: the transcriber fails on b. a and c complete, b is reported, exit 1.
    first = _run(config_path, StubSet(fail_transcribe_on={bad}))
    consoles.append(first.stdout + first.stderr)
    assert first.exit_code == 1, first.stderr
    assert "n_done=2 n_failed=1" in first.stderr
    assert f"file_failed file_id={bad} stage=transcribe error_code=STUB_FAILURE" in first.stderr
    for file_id in good:
        status = store.read_status(file_id)
        assert status is not None and set(status.stages.values()) == {StageStatus.DONE}
    bad_status = store.read_status(bad)
    assert bad_status is not None and bad_status.failed_stage is Stage.TRANSCRIBE

    # Run 2: the fault is gone, but the job is interrupted during b's PHI detection.
    stubs = StubSet(interrupt_ner_on={bad})
    second = _run(config_path, stubs)
    consoles.append(second.stdout + second.stderr)
    assert second.exit_code == EXIT_INTERRUPTED
    assert "rerun the same command to resume" in second.stderr
    assert stubs.calls() == {
        "audio": [bad],  # reloaded only for b's transcription
        "diarizer": [],
        "transcriber": [bad],
        "ner": [bad],
    }
    bad_status = store.read_status(bad)
    assert bad_status is not None
    assert bad_status.stages[Stage.ASSIGN_SPEAKERS] is StageStatus.DONE
    assert bad_status.stages[Stage.DETECT_PHI] is StageStatus.PENDING
    manifest = store.read_manifest()
    assert manifest is not None and manifest.finished_at is None

    # Run 3: completed stages are not redone; only b's PHI detection runs.
    stubs = StubSet()
    third = _run(config_path, stubs)
    consoles.append(third.stdout + third.stderr)
    assert third.exit_code == 0, third.stderr
    assert stubs.calls() == {"audio": [], "diarizer": [], "transcriber": [], "ner": [bad, bad]}
    for file_id in good | {bad}:
        status = store.read_status(file_id)
        assert status is not None and set(status.stages.values()) == {StageStatus.DONE}
    manifest = store.read_manifest()
    assert manifest is not None and manifest.finished_at is not None
    assert len(manifest.file_statuses) == 3

    # Inputs are untouched.
    assert _snapshot(input_dir) == before

    # The sentinel reached the transcripts but nothing else.
    job_dir = store.job_dir
    transcript = store.artifact_path(bad, Stage.TRANSCRIBE, "intended", suffix=".txt")
    assert SENTINEL in transcript.read_text()
    checked = [
        *(job_dir / "logs").iterdir(),
        job_dir / "metrics.json",
        job_dir / "manifest.json",
        *(store.status_path(f) for f in good | {bad}),
    ]
    assert len(list((job_dir / "logs").iterdir())) == 3  # one log file per run
    for path in checked:
        assert SENTINEL not in path.read_text(), path.name
    for console in consoles:
        assert SENTINEL not in console
