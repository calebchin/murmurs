"""P2-05b: stage-group runner (D21)."""

import gc
import io
import threading
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import SENTINEL, StubJob
from typer.testing import CliRunner

from murmurs import stages as stages_module
from murmurs.backends import Waveform, registry
from murmurs.backends.crisper import CrisperWhisperTranscriber
from murmurs.backends.registry import DEFAULT_AUDIO_LOADER, BackendKind
from murmurs.backends.stubs import (
    StubAudioLoader,
    StubDiarizer,
    StubNERDetector,
    StubTranscriber,
)
from murmurs.cli import app
from murmurs.config import JobConfig, Stage, load_config
from murmurs.errors import BackendSetupError
from murmurs.models import StageStatus
from murmurs.runner import run_job
from murmurs.stages import Backends, stage_groups
from murmurs.store import ArtifactStore, file_id_from_hash, hash_file

Event = tuple[str, str]


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    yield


class Recorder:
    """Registers recording stubs as the job's backends; logs opens, closes and calls."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.fail_transcribe: set[str] = set()
        self.broken: set[BackendKind] = set()
        self.waveform_refs: list[weakref.ref[Waveform]] = []
        self.alive_at_load: list[int] = []

    def register(self) -> None:
        log = self.events
        recorder = self

        class Audio(StubAudioLoader):
            def probe(self, path: Path, file_id: str) -> Any:
                log.append(("probe", file_id))
                return super().probe(path, file_id)

            def load(self, path: Path, file_id: str) -> Waveform:
                gc.collect()
                recorder.alive_at_load.append(sum(r() is not None for r in recorder.waveform_refs))
                log.append(("load", file_id))
                waveform = super().load(path, file_id)
                recorder.waveform_refs.append(weakref.ref(waveform))
                return waveform

            def close(self) -> None:
                log.append(("close", "audio_loader"))

        class Diar(StubDiarizer):
            def diarize(self, waveform: Waveform, file_id: str) -> Any:
                log.append(("diarize", file_id))
                return super().diarize(waveform, file_id)

            def close(self) -> None:
                log.append(("close", "diarizer"))

        class Trans(StubTranscriber):
            def transcribe(self, waveform: Waveform, file_id: str, modes: Any) -> Any:
                log.append(("transcribe", file_id))
                if file_id in recorder.fail_transcribe:
                    from murmurs.errors import PipelineError

                    raise PipelineError("STUB_FAILURE")
                return super().transcribe(waveform, file_id, modes)

            def close(self) -> None:
                log.append(("close", "transcriber"))

        class Ner(StubNERDetector):
            def detect(self, text: str, file_id: str) -> Any:
                log.append(("detect", file_id))
                return super().detect(text, file_id)

            def close(self) -> None:
                log.append(("close", "ner"))

        def factory(kind: BackendKind, cls: type) -> Callable[[JobConfig], Any]:
            def make(config: JobConfig) -> Any:
                if kind in recorder.broken:
                    raise RuntimeError(f"out of memory near {SENTINEL}")
                log.append(("open", kind.value))
                return cls()

            return make

        registry.register(
            BackendKind.AUDIO_LOADER, DEFAULT_AUDIO_LOADER, factory(BackendKind.AUDIO_LOADER, Audio)
        )
        registry.register(BackendKind.DIARIZER, "rec", factory(BackendKind.DIARIZER, Diar))
        registry.register(BackendKind.TRANSCRIBER, "rec", factory(BackendKind.TRANSCRIBER, Trans))
        registry.register(BackendKind.NER, "rec", factory(BackendKind.NER, Ner))

    def first(self, event: Event) -> int:
        return self.events.index(event)

    def last(self, event: Event) -> int:
        return len(self.events) - 1 - self.events[::-1].index(event)

    def calls(self, name: str) -> list[str]:
        return [fid for kind, fid in self.events if kind == name]

    def opened(self) -> list[str]:
        return self.calls("open")


@pytest.fixture
def rec() -> Recorder:
    recorder = Recorder()
    recorder.register()
    return recorder


def _config(stub_job: StubJob, **sections: dict[str, Any]) -> JobConfig:
    sections.setdefault("diarization", {})
    sections["diarization"] = {"model": "rec", "llm_correction": False, **sections["diarization"]}
    sections["transcription"] = {"model": "rec", **sections.get("transcription", {})}
    sections["deid"] = {"ner_model": "rec", **sections.get("deid", {})}
    return load_config(stub_job(**sections))


def _run(config: JobConfig) -> Any:
    return run_job(config, "job", console=io.StringIO())


def _with_stages(config: JobConfig, **update: Any) -> JobConfig:
    return config.model_copy(update={"stages": config.stages.model_copy(update=update)})


def _store(config: JobConfig) -> ArtifactStore:
    return ArtifactStore(config.data.output_dir, config.data.input_dir, "job")


def _file_id(config: JobConfig, name: str) -> str:
    return file_id_from_hash(hash_file(config.data.input_dir / name))


# --- grouping ---------------------------------------------------------------------------------


def test_stage_groups_follow_d21_and_the_stage_range() -> None:
    assert stage_groups(list(Stage)) == [
        ("ingest", [Stage.INGEST]),
        ("audio", [Stage.LOAD_AUDIO, Stage.DIARIZE, Stage.TRANSCRIBE]),
        ("assign_speakers", [Stage.ASSIGN_SPEAKERS]),
        ("detect_phi", [Stage.DETECT_PHI]),
    ]
    assert stage_groups([Stage.DIARIZE, Stage.TRANSCRIBE, Stage.ASSIGN_SPEAKERS]) == [
        ("audio", [Stage.DIARIZE, Stage.TRANSCRIBE]),
        ("assign_speakers", [Stage.ASSIGN_SPEAKERS]),
    ]


# --- order and backend lifetimes ---------------------------------------------------------------


def test_every_file_finishes_a_group_before_the_next_group(
    stub_job: StubJob, rec: Recorder
) -> None:
    summary = _run(_config(stub_job))
    assert len(summary.done) == 3 and not summary.failed
    files = rec.calls("probe")
    assert len(files) == 3

    # Group boundaries: all ingests, then the audio group, then detect_phi.
    last_probe = max(rec.last(("probe", f)) for f in files)
    first_load = min(rec.first(("load", f)) for f in files)
    last_transcribe = max(rec.last(("transcribe", f)) for f in files)
    first_detect = min(rec.first(("detect", f)) for f in files)
    assert last_probe < first_load and last_transcribe < first_detect

    # Within the audio group, each file runs load, diarize, transcribe back to back.
    audio = [(k, f) for k, f in rec.events if k in ("load", "diarize", "transcribe")]
    for i in range(0, len(audio), 3):
        chunk = audio[i : i + 3]
        assert [k for k, _ in chunk] == ["load", "diarize", "transcribe"]
        assert len({f for _, f in chunk}) == 1


def test_backends_open_just_in_time_and_close_after_their_last_group(
    stub_job: StubJob, rec: Recorder
) -> None:
    _run(_config(stub_job))
    files = rec.calls("probe")
    ev = rec.events
    assert sorted(rec.opened()) == ["audio_loader", "diarizer", "ner", "transcriber"]
    assert sorted(rec.calls("close")) == sorted(rec.opened())  # each closed exactly once

    assert rec.first(("open", "audio_loader")) < rec.first(("probe", files[0]))
    assert rec.first(("open", "diarizer")) < min(rec.first(("diarize", f)) for f in files)
    assert rec.first(("open", "diarizer")) > max(rec.last(("probe", f)) for f in files)
    last_transcribe = max(rec.last(("transcribe", f)) for f in files)
    for kind in ("audio_loader", "diarizer", "transcriber"):
        assert last_transcribe < ev.index(("close", kind)) < ev.index(("open", "ner"))
    assert ev.index(("close", "ner")) > max(rec.last(("detect", f)) for f in files)


def test_stage_range_that_skips_groups_creates_none_of_their_backends(
    stub_job: StubJob, rec: Recorder
) -> None:
    _run(_with_stages(_config(stub_job), end=Stage.INGEST))
    assert rec.opened() == ["audio_loader"]
    assert rec.calls("close") == ["audio_loader"]


def test_resume_with_complete_audio_stages_loads_no_audio_models(
    stub_job: StubJob, rec: Recorder
) -> None:
    config = _config(stub_job)
    _run(config)
    rec.events.clear()
    changed = config.model_copy(
        update={"deid": config.deid.model_copy(update={"redaction_threshold": 0.9})}
    )
    summary = _run(changed)
    assert len(summary.done) == 3
    assert rec.opened() == ["ner"]  # only detect_phi reruns
    assert rec.calls("load") == []


# --- failures -------------------------------------------------------------------------------


def test_failed_files_skip_later_stages_and_others_complete(
    stub_job: StubJob, rec: Recorder
) -> None:
    config = _config(stub_job)
    (config.data.input_dir / "empty.wav").write_bytes(b"")
    bad_transcribe = _file_id(config, "b.wav")
    rec.fail_transcribe.add(bad_transcribe)
    empty = _file_id(config, "empty.wav")

    summary = _run(config)

    assert summary.exit_code == 1
    assert {(f.file_id, f.stage) for f in summary.failed} == {
        (empty, Stage.INGEST),
        (bad_transcribe, Stage.TRANSCRIBE),
    }
    assert len(summary.done) == 2
    assert empty not in rec.calls("load")
    assert empty not in rec.calls("detect") and bad_transcribe not in rec.calls("detect")
    status = _store(config).read_status(bad_transcribe)
    assert status is not None
    assert status.stages[Stage.ASSIGN_SPEAKERS] is StageStatus.SKIPPED
    assert status.stages[Stage.DETECT_PHI] is StageStatus.SKIPPED


def test_backend_that_fails_to_load_stops_the_job_with_exit_2(
    stub_job: StubJob, rec: Recorder
) -> None:
    config = _config(stub_job)
    config_path = config.data.input_dir.parent / "job.yaml"
    rec.broken.add(BackendKind.DIARIZER)

    result = CliRunner().invoke(app, ["run", "-c", str(config_path)])

    assert result.exit_code == 2
    assert "diarizer" in result.stderr and SENTINEL not in result.output
    assert sorted(rec.calls("close")) == ["audio_loader"]  # what was open got closed
    for name in ("a.wav", "b.wav", "c.m4a"):
        status = _store(config).read_status(_file_id(config, name))
        assert status is not None and status.stages[Stage.INGEST] is StageStatus.DONE

    rec.broken.clear()
    rec.events.clear()
    summary = _run(config)
    assert len(summary.done) == 3 and rec.calls("probe") == []  # ingest was kept


def test_setup_error_from_factory_keeps_its_message(stub_job: StubJob, rec: Recorder) -> None:
    def refuse(config: JobConfig) -> Any:
        raise BackendSetupError("no CUDA device is available")

    registry.register(BackendKind.DIARIZER, "rec", refuse)
    with pytest.raises(BackendSetupError, match="no CUDA device"):
        _run(_config(stub_job))


def test_ctrl_c_in_assign_group_then_resume_skips_audio(
    stub_job: StubJob, rec: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(stub_job)
    real_assign = stages_module.STAGE_FUNCTIONS[Stage.ASSIGN_SPEAKERS]
    seen: list[str] = []

    def interrupt_second(ctx: Any) -> None:
        seen.append(ctx.file_id)
        if len(seen) == 2:
            raise KeyboardInterrupt
        real_assign(ctx)

    monkeypatch.setitem(stages_module.STAGE_FUNCTIONS, Stage.ASSIGN_SPEAKERS, interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        _run(config)
    assert sorted(rec.calls("close")) == sorted(rec.opened())  # nothing left open

    monkeypatch.setitem(stages_module.STAGE_FUNCTIONS, Stage.ASSIGN_SPEAKERS, real_assign)
    rec.events.clear()
    summary = _run(config)
    assert len(summary.done) == 3 and not summary.failed
    for name in ("probe", "load", "diarize", "transcribe"):
        assert rec.calls(name) == []
    assert "audio_loader" not in rec.opened()


# --- audio memory ---------------------------------------------------------------------------


def test_waveform_decoded_once_per_file_and_freed_before_the_next(
    stub_job: StubJob, rec: Recorder
) -> None:
    _run(_config(stub_job))
    assert sorted(rec.calls("load")) == sorted(rec.calls("probe"))  # one decode per file
    assert rec.alive_at_load == [0, 0, 0]


def test_group_events_are_logged(stub_job: StubJob, rec: Recorder) -> None:
    console = io.StringIO()
    run_job(_config(stub_job), "job", console=console)
    out = console.getvalue()
    for group in ("ingest", "audio", "assign_speakers", "detect_phi"):
        assert f"group_start group={group} n_files=3" in out
        assert f"group_end group={group} n_files=3" in out


# --- injected backends ------------------------------------------------------------------------


def test_injected_backends_are_never_closed(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    backends = Backends(
        audio_loader=StubAudioLoader(),
        diarizer=StubDiarizer(),
        transcriber=StubTranscriber(),
        ner=StubNERDetector(),
    )
    run_job(config, "job", backends=backends, console=io.StringIO())
    stubs = (backends.audio_loader, backends.diarizer, backends.transcriber, backends.ner)
    assert all(s is not None and s.close_calls == 0 for s in stubs)  # type: ignore[union-attr]


# --- real backends' close ---------------------------------------------------------------------


def test_crisperwhisper_close_drops_model_on_its_thread_and_stops_it(
    stub_job: StubJob, tmp_path: Path
) -> None:
    threads: list[int] = []

    class Model:
        def __del__(self) -> None:
            threads.append(threading.get_ident())

    def load(model_dir: Path, device: str, compute_type: str) -> Model:
        threads.append(threading.get_ident())
        return Model()

    config = load_config(stub_job(compute={"device": "cpu"}))
    transcriber = CrisperWhisperTranscriber(config, tmp_path, "0" * 40, load_model=load)
    transcriber.close()
    transcriber.close()  # a second close is harmless
    assert len(threads) == 2 and threads[0] == threads[1] != threading.get_ident()
    assert transcriber._executor._shutdown  # type: ignore[attr-defined]
