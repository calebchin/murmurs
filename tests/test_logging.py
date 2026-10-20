"""P1-07: PHI-safe logging and timing metrics."""

import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from murmurs.backends import Waveform
from murmurs.backends.stubs import StubDiarizer, StubTranscriber
from murmurs.cli import app
from murmurs.config import Stage
from murmurs.errors import PipelineError
from murmurs.logging import (
    DEBUG_UNSAFE_WARNING,
    UNEXPECTED_ERROR,
    FileOutcome,
    LoggerProgress,
    Metrics,
    MetricsRecorder,
    RunLogger,
    UnsafeLogFieldError,
    install_excepthook,
)
from murmurs.models import StageStatus, TranscriptMode

SENTINEL = "SENTINEL_PHI_7f3a"
EXAMPLE = Path(__file__).parent.parent / "examples" / "job.yaml"


@pytest.fixture
def console() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def logger(tmp_path: Path, console: io.StringIO) -> RunLogger:
    return RunLogger(log_dir=tmp_path / "logs", console=console)


def _records(logger: RunLogger) -> list[dict[str, object]]:
    assert logger.log_path is not None
    return [json.loads(line) for line in logger.log_path.read_text().splitlines()]


def _all_output(tmp_path: Path, console: io.StringIO) -> str:
    files = "".join(p.read_text() for p in tmp_path.rglob("*") if p.is_file())
    return files + console.getvalue()


# --- allowlist --------------------------------------------------------------------


def test_allowed_fields_are_written_to_file_and_console(
    logger: RunLogger, console: io.StringIO
) -> None:
    logger.info("stage_end", file_id="ab12", stage=Stage.TRANSCRIBE, duration_s=1.5, n_words=3)
    [record] = _records(logger)
    assert record["event"] == "stage_end"
    assert record["stage"] == "transcribe"
    assert record["n_words"] == 3
    assert "stage_end file_id=ab12 stage=transcribe duration_s=1.5 n_words=3" in console.getvalue()


@pytest.mark.parametrize("field", ["text", "prompt", "response", "message", "word", "path"])
def test_disallowed_field_raises(logger: RunLogger, field: str) -> None:
    with pytest.raises(UnsafeLogFieldError, match="not allowlisted"):
        logger.info("something", **{field: "x"})
    assert logger.log_path is None or not logger.log_path.exists()


@pytest.mark.parametrize(
    "value", ["two words", "x" * 129, "line\nbreak", ["a"], {"k": "v"}, object()]
)
def test_free_text_values_are_rejected(logger: RunLogger, value: object) -> None:
    with pytest.raises(UnsafeLogFieldError):
        logger.info("something", file_id=value)


def test_event_name_must_be_snake_case(logger: RunLogger) -> None:
    with pytest.raises(UnsafeLogFieldError):
        logger.info("Hello world")


def test_debug_events_go_to_file_only_by_default(logger: RunLogger, console: io.StringIO) -> None:
    logger.event("stage_start", "debug", file_id="ab12")
    assert _records(logger)[0]["event"] == "stage_start"
    assert console.getvalue() == ""


# --- exceptions -----------------------------------------------------------------------


def test_exception_logs_type_code_stage_and_file_only(
    tmp_path: Path, logger: RunLogger, console: io.StringIO
) -> None:
    exc = PipelineError("STUB_FAILURE", detail=f"could not parse '{SENTINEL}'")
    code = logger.exception(exc, stage=Stage.TRANSCRIBE, file_id="ab12")
    assert code == "STUB_FAILURE"
    [record] = _records(logger)
    assert record == {
        "ts": record["ts"],
        "level": "error",
        "event": "exception",
        "exception_type": "PipelineError",
        "error_code": "STUB_FAILURE",
        "stage": "transcribe",
        "file_id": "ab12",
    }
    assert SENTINEL not in _all_output(tmp_path, console)


def test_unexpected_exception_gets_generic_code(logger: RunLogger) -> None:
    assert logger.exception(ValueError(f"bad token {SENTINEL}")) == UNEXPECTED_ERROR
    assert _records(logger)[0]["exception_type"] == "ValueError"


def test_debug_unsafe_warns_and_writes_traceback_to_file_only(tmp_path: Path) -> None:
    console = io.StringIO()
    logger = RunLogger(log_dir=tmp_path / "logs", console=console, debug_unsafe=True)
    assert DEBUG_UNSAFE_WARNING in console.getvalue()
    try:
        raise ValueError(SENTINEL)
    except ValueError as exc:
        logger.exception(exc)
    tracebacks = [r for r in _records(logger) if r["event"] == "unsafe_traceback"]
    assert len(tracebacks) == 1 and SENTINEL in str(tracebacks[0]["traceback"])
    assert SENTINEL not in console.getvalue()


def test_cli_debug_unsafe_flag_prints_warning() -> None:
    result = CliRunner().invoke(app, ["run", "--config", str(EXAMPLE), "--debug-unsafe"])
    assert DEBUG_UNSAFE_WARNING in result.stderr


@pytest.fixture
def restore_excepthook() -> Iterator[None]:
    original = sys.excepthook
    yield
    sys.excepthook = original


@pytest.mark.usefixtures("restore_excepthook")
def test_excepthook_logs_crash_without_message(
    tmp_path: Path, logger: RunLogger, console: io.StringIO
) -> None:
    install_excepthook(logger)
    try:
        raise RuntimeError(SENTINEL)
    except RuntimeError as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)
    assert "murmurs crashed: RuntimeError (UNEXPECTED_ERROR); see logs" in console.getvalue()
    assert _records(logger)[0]["event"] == "crash"
    assert SENTINEL not in _all_output(tmp_path, console)


# --- metrics ------------------------------------------------------------------------


def test_metrics_record_timings_and_write_atomically(tmp_path: Path) -> None:
    ticks = iter([0.0, 1.25, 2.0, 2.5])
    recorder = MetricsRecorder(job_id="job1", clock=lambda: next(ticks))
    with recorder.timed("ab12", Stage.TRANSCRIBE):
        pass
    with pytest.raises(PipelineError), recorder.timed("ab12", Stage.ASSIGN_SPEAKERS):
        raise PipelineError("STUB_FAILURE")
    path = tmp_path / "metrics.json"
    recorder.write(path)

    metrics = Metrics.model_validate_json(path.read_text())
    timings = metrics.files["ab12"]
    assert timings[Stage.TRANSCRIBE].duration_s == 1.25
    assert timings[Stage.TRANSCRIBE].status is StageStatus.DONE
    assert timings[Stage.ASSIGN_SPEAKERS].status is StageStatus.FAILED
    assert [p.name for p in tmp_path.iterdir()] == ["metrics.json"]


# --- end to end with stubs --------------------------------------------------------------


def test_stub_run_with_sentinel_never_leaks(tmp_path: Path, console: io.StringIO) -> None:
    """Drive stub backends the way the runner will; P1-10 repeats this through the CLI."""
    logger = RunLogger(log_dir=tmp_path / "logs", console=console)
    progress = LoggerProgress(logger)
    recorder = MetricsRecorder(job_id="job1")
    transcriber = StubTranscriber(
        text=f"my name is {SENTINEL}", fail_on={"bad"}, fail_detail=SENTINEL
    )
    diarizer = StubDiarizer()
    waveform = Waveform(np.zeros(32_000, dtype=np.float32), 16_000)

    for file_id in ["good", "bad"]:
        outcome = FileOutcome.DONE
        for stage in (Stage.DIARIZE, Stage.TRANSCRIBE):
            progress.on_stage_start(file_id, stage)
            try:
                with recorder.timed(file_id, stage):
                    if stage is Stage.DIARIZE:
                        result = diarizer.diarize(waveform, file_id)
                        logger.info("diarized", file_id=file_id, n_turns=len(result.turns))
                    else:
                        transcripts = transcriber.transcribe(
                            waveform, file_id, list(TranscriptMode)
                        )
                        n_words = sum(
                            len(s.words) for s in transcripts[TranscriptMode.INTENDED].segments
                        )
                        logger.info("transcribed", file_id=file_id, n_words=n_words)
                status = StageStatus.DONE
            except PipelineError as exc:
                logger.exception(exc, stage=stage, file_id=file_id)
                status, outcome = StageStatus.FAILED, FileOutcome.FAILED
            progress.on_stage_end(
                file_id, stage, status, recorder.metrics.files[file_id][stage].duration_s
            )
            if status is StageStatus.FAILED:
                break
        progress.on_file_done(file_id, outcome)
    recorder.write(tmp_path / "metrics.json")

    output = _all_output(tmp_path, console)
    assert "error_code=STUB_FAILURE" in console.getvalue()
    assert "transcribed" in output
    assert SENTINEL not in output
