"""P1-08: stage runner."""

import io
import json
from pathlib import Path

import pytest
from conftest import SENTINEL, StubJob
from typer.testing import CliRunner

from murmurs.backends.stubs import (
    StubAudioLoader,
    StubDiarizer,
    StubNERDetector,
    StubTranscriber,
)
from murmurs.cli import app
from murmurs.config import JobConfig, Stage, load_config
from murmurs.errors import BackendSetupError
from murmurs.models import FileStatus, StageStatus, Transcript
from murmurs.runner import (
    EXIT_FILES_FAILED,
    EXIT_OK,
    JobSetupError,
    discover_inputs,
    run_job,
    stage_hashes,
)
from murmurs.stages import Backends
from murmurs.store import ArtifactStore, OutputInsideInputError, file_id_from_hash, hash_file


class Stubs:
    """A fresh set of stub backends with call counters."""

    def __init__(self, fail_transcribe_on: set[str] = frozenset(), text: str = "hello there"):
        self.audio = StubAudioLoader()
        self.diarizer = StubDiarizer()
        self.transcriber = StubTranscriber(text=text, fail_on=fail_transcribe_on)
        self.ner = StubNERDetector()

    @property
    def backends(self) -> Backends:
        return Backends(
            audio_loader=self.audio,
            diarizer=self.diarizer,
            transcriber=self.transcriber,
            ner=self.ner,
        )

    @property
    def total_calls(self) -> int:
        stubs = (self.audio, self.diarizer, self.transcriber, self.ner)
        return sum(len(s.calls) for s in stubs)


def _with(config: JobConfig, section: str, **values: object) -> JobConfig:
    updated = getattr(config, section).model_copy(update=values)
    return config.model_copy(update={section: updated})


def _file_id(config: JobConfig, name: str) -> str:
    return file_id_from_hash(hash_file(config.data.input_dir / name))


def _store(config: JobConfig) -> ArtifactStore:
    return ArtifactStore(config.data.output_dir, config.data.input_dir, "job")


def _artifacts(config: JobConfig, file_id: str) -> set[str]:
    file_dir = _store(config).file_dir(file_id)
    return {p.name for p in file_dir.iterdir()} - {"status.json"}


def _run(config: JobConfig, stubs: Stubs, console: io.StringIO | None = None) -> object:
    return run_job(config, "job", backends=stubs.backends, console=console or io.StringIO())


# --- acceptance criteria ----------------------------------------------------------------


def test_stage_range_produces_only_those_artifacts(stub_job: StubJob) -> None:
    config = _with(load_config(stub_job()), "stages", end=Stage.TRANSCRIBE)
    summary = run_job(config, "job", backends=Stubs().backends, console=io.StringIO())
    assert summary.exit_code == EXIT_OK
    assert len(summary.done) == 3
    for file_id in summary.done:
        assert _artifacts(config, file_id) == {
            "ingest.json",
            "load_audio.json",
            "diarize.json",
            "transcribe.verbatim.json",
            "transcribe.verbatim.txt",
            "transcribe.intended.json",
            "transcribe.intended.txt",
        }


def test_failing_file_is_reported_and_others_complete(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    bad = _file_id(config, "b.wav")
    console = io.StringIO()
    summary = run_job(
        config, "job", backends=Stubs(fail_transcribe_on={bad}).backends, console=console
    )

    assert summary.exit_code == EXIT_FILES_FAILED
    assert sorted(summary.done) == sorted([_file_id(config, "a.wav"), _file_id(config, "c.m4a")])
    [failure] = summary.failed
    assert (failure.file_id, failure.stage, failure.error_code) == (
        bad,
        Stage.TRANSCRIBE,
        "STUB_FAILURE",
    )
    assert f"file_failed file_id={bad} stage=transcribe error_code=STUB_FAILURE" in (
        console.getvalue()
    )

    status = _store(config).read_status(bad)
    assert status is not None
    assert status.stages[Stage.TRANSCRIBE] is StageStatus.FAILED
    assert status.stages[Stage.ASSIGN_SPEAKERS] is StageStatus.SKIPPED
    assert status.stages[Stage.DETECT_PHI] is StageStatus.SKIPPED
    assert status.error is not None and status.error.code == "STUB_FAILURE"
    for good in summary.done:
        good_status = _store(config).read_status(good)
        assert good_status is not None
        assert set(good_status.stages.values()) == {StageStatus.DONE}


def test_rerun_after_success_does_no_stage_work(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    _run(config, Stubs())
    stubs = Stubs()
    summary = run_job(config, "job", backends=stubs.backends, console=io.StringIO())
    assert stubs.total_calls == 0
    assert len(summary.skipped) == 3 and summary.done == []


def test_changed_stage_config_reruns_it_and_dependents_only(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    _run(config, Stubs())

    changed = _with(config, "transcription", word_timestamps=False)
    stubs = Stubs()
    _run(changed, stubs)
    assert stubs.audio.calls == stubs.transcriber.calls  # audio reloaded only to transcribe
    assert len(stubs.transcriber.calls) == 3
    assert stubs.diarizer.calls == []
    assert len(set(stubs.ner.calls)) == 3  # detect_phi depends on assign_speakers

    changed = _with(changed, "diarization", num_speakers=2)
    stubs = Stubs()
    _run(changed, stubs)
    assert len(stubs.diarizer.calls) == 3
    assert stubs.transcriber.calls == []


def test_changing_gap_tolerance_reruns_assignment_only(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    _run(config, Stubs())
    stubs = Stubs()
    _run(_with(config, "diarization", speaker_gap_tolerance_s=1.0), stubs)
    assert stubs.audio.calls == stubs.diarizer.calls == stubs.transcriber.calls == []
    assert len(set(stubs.ner.calls)) == 3


# --- resume details --------------------------------------------------------------------


def test_failed_file_is_retried_on_rerun(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    bad = _file_id(config, "b.wav")
    _run(config, Stubs(fail_transcribe_on={bad}))
    stubs = Stubs()
    summary = run_job(config, "job", backends=stubs.backends, console=io.StringIO())
    assert summary.done == [bad] and summary.exit_code == EXIT_OK
    assert stubs.transcriber.calls == [bad]
    assert stubs.diarizer.calls == []
    status = _store(config).read_status(bad)
    assert status is not None and status.error is None and status.failed_stage is None


def test_no_resume_reruns_everything(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    _run(config, Stubs())
    stubs = Stubs()
    _run(_with(config, "stages", resume=False), stubs)
    assert len(stubs.transcriber.calls) == 3 and len(stubs.diarizer.calls) == 3


def test_deleted_artifact_is_rebuilt_with_dependents(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    _run(config, Stubs())
    file_id = _file_id(config, "a.wav")
    _store(config).artifact_path(file_id, Stage.DIARIZE).unlink()
    stubs = Stubs()
    _run(config, stubs)
    assert stubs.diarizer.calls == [file_id]
    assert set(stubs.ner.calls) == {file_id}
    assert stubs.transcriber.calls == []


def test_later_start_needs_upstream_done(stub_job: StubJob) -> None:
    config = _with(load_config(stub_job()), "stages", start=Stage.DIARIZE)
    summary = run_job(config, "job", backends=Stubs().backends, console=io.StringIO())
    assert {f.error_code for f in summary.failed} == {"MISSING_UPSTREAM"}
    assert {f.stage for f in summary.failed} == {Stage.DIARIZE}


def test_later_start_reloads_audio_after_ingest_and_load(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    _run(_with(config, "stages", end=Stage.LOAD_AUDIO), Stubs())
    stubs = Stubs()
    summary = _run(_with(config, "stages", start=Stage.DIARIZE), stubs)
    assert summary.exit_code == EXIT_OK  # type: ignore[attr-defined]
    assert len(stubs.audio.calls) == 3  # one reload per file, shared by both stages


def test_widening_the_range_after_upstream_rerun_reruns_dependents(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    _run(config, Stubs())
    transcribe_only = _with(
        _with(config, "transcription", word_timestamps=False), "stages", end=Stage.TRANSCRIBE
    )
    _run(transcribe_only, Stubs())
    stubs = Stubs()
    _run(_with(config, "transcription", word_timestamps=False), stubs)
    assert stubs.transcriber.calls == []
    assert len(set(stubs.ner.calls)) == 3  # assign/detect saw the new transcripts


def test_stage_hashes_change_only_for_affected_stages(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    before = stage_hashes(config)
    after = stage_hashes(_with(config, "transcription", word_timestamps=False))
    changed = {s for s in Stage if before[s] != after[s]}
    assert changed == {Stage.TRANSCRIBE, Stage.ASSIGN_SPEAKERS, Stage.DETECT_PHI}


def test_assignment_and_detection_artifacts_are_valid(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    _run(config, Stubs())
    file_id = _file_id(config, "a.wav")
    labeled = _store(config).read(file_id, Stage.ASSIGN_SPEAKERS, Transcript, "intended")
    speakers = {w.speaker for s in labeled.segments for w in s.words}
    assert speakers <= {"SPEAKER_00", "SPEAKER_01"} and speakers


# --- discovery ---------------------------------------------------------------------------


def test_discovery_is_recursive_case_insensitive_and_sorted(tmp_path: Path) -> None:
    for name, content in {
        "z.wav": b"1",
        "sub/deeper/a.WAV": b"2",
        "sub/b.m4a": b"3",
        "notes.txt": b"4",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    found = discover_inputs(tmp_path, [".wav", ".m4a"])
    assert [f.path.relative_to(tmp_path).as_posix() for f in found.files] == [
        "sub/b.m4a",
        "sub/deeper/a.WAV",
        "z.wav",
    ]


def test_duplicate_inputs_are_processed_once_and_reported(stub_job: StubJob) -> None:
    config = load_config(stub_job({"a.wav": b"same", "copy/a2.wav": b"same", "b.wav": b"x"}))
    console = io.StringIO()
    stubs = Stubs()
    summary = run_job(config, "job", backends=stubs.backends, console=console)
    assert summary.exit_code == EXIT_OK
    [dup] = summary.duplicates
    assert dup.path.name == "a2.wav"
    assert len(summary.done) == 2
    assert stubs.transcriber.calls.count(dup.file_id) == 1
    assert f"duplicate_input file_id={dup.file_id}" in console.getvalue()


def test_missing_input_dir_is_setup_error(stub_job: StubJob, tmp_path: Path) -> None:
    config = _with(load_config(stub_job()), "data", input_dir=tmp_path / "nope")
    with pytest.raises(JobSetupError, match="not a directory"):
        run_job(config, "job", backends=Stubs().backends, console=io.StringIO())


def test_unknown_mode_is_setup_error(stub_job: StubJob) -> None:
    config = _with(load_config(stub_job()), "transcription", modes=["fancy"])
    with pytest.raises(JobSetupError, match=r"transcription\.modes"):
        run_job(config, "job", backends=Stubs().backends, console=io.StringIO())


def test_parallel_workers_logs_notice(stub_job: StubJob) -> None:
    config = _with(load_config(stub_job()), "compute", parallel_workers=4)
    console = io.StringIO()
    run_job(config, "job", backends=Stubs().backends, console=console)
    assert "parallelism_not_implemented n_workers=4" in console.getvalue()


def test_metrics_and_status_written(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    summary = run_job(config, "job", backends=Stubs().backends, console=io.StringIO())
    metrics = json.loads((_store(config).job_dir / "metrics.json").read_text())
    assert set(metrics["files"]) == set(summary.done)
    status = FileStatus.model_validate_json(_store(config).status_path(summary.done[0]).read_text())
    assert set(status.stage_hashes) == set(Stage)


def test_sentinel_never_leaks_from_runner(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    bad = _file_id(config, "b.wav")
    stubs = Stubs(fail_transcribe_on={bad}, text=f"my name is {SENTINEL}")
    stubs.transcriber.fail_detail = SENTINEL
    console = io.StringIO()
    run_job(config, "job", backends=stubs.backends, console=console)
    job_dir = _store(config).job_dir
    logs = "".join(p.read_text() for p in (job_dir / "logs").iterdir())
    assert SENTINEL not in logs + console.getvalue()
    assert SENTINEL not in (job_dir / "metrics.json").read_text()


# --- CLI ------------------------------------------------------------------------------------


def test_cli_runs_job_named_after_config(stub_job: StubJob) -> None:
    config_path = stub_job()
    result = CliRunner().invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code == EXIT_OK, result.stderr
    config = load_config(config_path)
    assert (config.data.output_dir / "job" / "metrics.json").is_file()
    assert "job_end job_id=job n_done=3 n_failed=0 n_skipped=0" in result.stderr

    rerun = CliRunner().invoke(app, ["run", "--config", str(config_path)])
    assert "n_done=0 n_failed=0 n_skipped=3" in rerun.stderr


def test_cli_job_id_override(stub_job: StubJob) -> None:
    config_path = stub_job()
    result = CliRunner().invoke(app, ["run", "-c", str(config_path), "--job-id", "trial2"])
    assert result.exit_code == EXIT_OK
    assert (load_config(config_path).data.output_dir / "trial2").is_dir()


def test_cli_setup_errors_exit_2(stub_job: StubJob, tmp_path: Path) -> None:
    config_path = stub_job()
    runner = CliRunner()
    inside = runner.invoke(
        app, ["run", "-c", str(config_path), "--output-dir", str(tmp_path / "in" / "out")]
    )
    assert inside.exit_code == 2 and "inside input_dir" in inside.stderr
    unknown = runner.invoke(app, ["run", "-c", str(config_path), "--set", "transcription.model=x"])
    assert unknown.exit_code == 2 and "no transcriber backend named 'x'" in unknown.stderr
    bad_id = runner.invoke(app, ["run", "-c", str(config_path), "--job-id", "a/b"])
    assert bad_id.exit_code == 2 and "job_id" in bad_id.stderr


def test_output_inside_input_raises_before_processing(stub_job: StubJob, tmp_path: Path) -> None:
    config = _with(load_config(stub_job()), "data", output_dir=tmp_path / "in" / "out")
    stubs = Stubs()
    with pytest.raises(OutputInsideInputError):
        run_job(config, "job", backends=stubs.backends, console=io.StringIO())
    assert stubs.total_calls == 0


def test_real_model_names_fail_setup(stub_job: StubJob) -> None:
    config = _with(load_config(stub_job()), "diarization", model="pyannote/speaker-diarization-3.1")
    with pytest.raises(BackendSetupError):
        run_job(config, "job", console=io.StringIO())
