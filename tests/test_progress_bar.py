"""The terminal progress bar: files done per stage group."""

import io

from conftest import SENTINEL, StubJob

from murmurs.backends.stubs import (
    StubAudioLoader,
    StubDiarizer,
    StubLLMClient,
    StubNERDetector,
    StubTranscriber,
)
from murmurs.config import Stage, load_config
from murmurs.logging import ProgressBar
from murmurs.models import StageStatus
from murmurs.runner import run_job
from murmurs.stages import Backends

CLEAR = "\r\x1b[2K"


def _backends() -> Backends:
    return Backends(
        audio_loader=StubAudioLoader(duration_s=4.0),
        diarizer=StubDiarizer(turn_s=1.0),
        transcriber=StubTranscriber(text=f"my name is {SENTINEL}"),
        ner=StubNERDetector(),
        llm=StubLLMClient(response={"entities": []}),
    )


def _audio_group(bar: ProgressBar, n_files: int) -> None:
    bar.on_group_start("audio", [Stage.LOAD_AUDIO, Stage.TRANSCRIBE], 2, 4, n_files)


def test_bar_counts_a_file_when_it_ends_the_groups_last_stage() -> None:
    bar = ProgressBar(io.StringIO(), width=10)
    _audio_group(bar, 4)
    assert bar.line().startswith("[2/4] audio |░░░░░░░░░░| 0/4 files")
    bar.on_stage_end("f1", Stage.LOAD_AUDIO, StageStatus.DONE, 1.0)
    assert "0/4 files" in bar.line()
    bar.on_stage_end("f1", Stage.TRANSCRIBE, StageStatus.DONE, 1.0)
    assert "|██░░░░░░░░| 1/4 files" in bar.line()
    bar.on_stage_end("f2", Stage.TRANSCRIBE, StageStatus.SKIPPED, 0.0)
    assert "|█████░░░░░| 2/4 files" in bar.line()
    assert "left" in bar.line()


def test_a_failed_file_counts_as_done_for_the_group() -> None:
    bar = ProgressBar(io.StringIO(), width=10)
    _audio_group(bar, 2)
    bar.on_stage_end("f1", Stage.LOAD_AUDIO, StageStatus.FAILED, 1.0)
    bar.on_stage_end("f2", Stage.TRANSCRIBE, StageStatus.DONE, 1.0)
    assert "|██████████| 2/2 files" in bar.line() and "left" not in bar.line()


def test_log_lines_print_above_the_bar_and_close_removes_it() -> None:
    stream = io.StringIO()
    bar = ProgressBar(stream, width=10)
    _audio_group(bar, 2)
    bar.console.write("12:00:00 INFO    stage_end\n")
    out = stream.getvalue()
    assert out.endswith(CLEAR + "12:00:00 INFO    stage_end\n" + bar.line())
    bar.close()
    assert stream.getvalue().endswith(bar.line() + CLEAR)


def test_run_draws_a_bar_per_group_without_transcript_text(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    config = config.model_copy(
        update={
            "stages": config.stages.model_copy(update={"end": Stage.ASSIGN_SPEAKERS}),
            "diarization": config.diarization.model_copy(update={"llm_correction": False}),
        }
    )
    stream = io.StringIO()
    bar = ProgressBar(stream)
    summary = run_job(config, "job", backends=_backends(), progress=bar, console=bar.console)
    bar.close()
    assert len(summary.done) == 3
    out = stream.getvalue()
    for label in ("[1/3] ingest", "[2/3] audio", "[3/3] assign_speakers"):
        assert f"{label} |" in out and "3/3 files" in out.split(label, 1)[1]
    assert "file_done" in out  # log lines still reach the console
    assert SENTINEL not in out
