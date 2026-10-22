"""P2-04: CrisperWhisper transcription backend."""

import importlib.metadata
import io
import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import huggingface_hub.constants
import numpy as np
import pytest
import yaml
from conftest import SENTINEL, StubJob
from typer.testing import CliRunner

from murmurs.backends import Waveform, crisper, registry
from murmurs.backends.crisper import (
    MODEL_ID,
    READY_MARKER,
    SPEC,
    TRANSCRIPTION_FAILED,
    UNSUPPORTED_SAMPLE_RATE,
    CrisperWhisperTranscriber,
    check_backend,
    converted_dir,
    create_crisperwhisper,
    words_to_transcript,
)
from murmurs.backends.registry import BackendKind
from murmurs.backends.stubs import StubAudioLoader, StubDiarizer, StubLLMClient, StubNERDetector
from murmurs.cli import app
from murmurs.config import JobConfig, Stage, load_config
from murmurs.errors import BackendSetupError, PipelineError
from murmurs.models import AudioInfo, Transcript, TranscriptMode
from murmurs.offline import PreparedModel, UnpreparedModelError, WeightSpec
from murmurs.runner import run_job
from murmurs.stages import Backends
from murmurs.store import ArtifactStore

VERB, INT = TranscriptMode.VERBATIM, TranscriptMode.INTENDED
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _wave(seconds: float = 4.0, rate: int = 16_000) -> Waveform:
    return Waveform(samples=np.zeros(int(seconds * rate), dtype=np.float32), sample_rate=rate)


# --- words to transcript (D18, D19) -------------------------------------------------------------


def test_segments_split_at_sentence_end_and_long_pauses() -> None:
    words = [
        ("Hello", 0.0, 0.4), ("there.", 0.5, 0.9),  # sentence end
        ("How", 1.0, 1.2), ("are", 1.3, 1.5),       # 1.5 s pause next
        ("you", 3.0, 3.2), ("today?", 3.3, 3.8),
    ]  # fmt: skip
    transcript = words_to_transcript(words, "f1", VERB)
    assert [s.text for s in transcript.segments] == ["Hello there.", "How are", "you today?"]
    for segment in transcript.segments:
        assert segment.start == segment.words[0].start and segment.end == segment.words[-1].end
        assert all(segment.start <= w.start and w.end <= segment.end for w in segment.words)
        assert segment.speaker is None and all(w.speaker is None for w in segment.words)


def test_pause_just_under_threshold_keeps_segment() -> None:
    transcript = words_to_transcript([("a", 0.0, 0.5), ("b", 1.4, 1.6)], "f1", VERB)
    assert len(transcript.segments) == 1


def test_untimed_and_empty_words_are_dropped_and_times_clipped() -> None:
    words = [("", 0.0, 0.1), ("a", None, 0.2), ("b", -0.1, 0.3), (" c ", 0.4, 9.0)]
    transcript = words_to_transcript(words, "f1", INT, duration_s=2.0)
    assert [(w.text, w.start, w.end) for w in transcript.segments[0].words] == [
        ("b", 0.0, 0.3),
        ("c", 0.4, 2.0),
    ]
    assert transcript.mode is INT and transcript.file_id == "f1"


def test_no_words_gives_no_segments() -> None:
    assert words_to_transcript([], "f1", VERB).segments == []


# --- the transcriber, with a fake model ----------------------------------------------------------


class FakeModel:
    """Stands in for ``CrisperWhisperModel``; records the thread of every call."""

    def __init__(self, words: list[tuple[str, float, float]], fail: bool = False) -> None:
        self.words = words
        self.fail = fail
        self.threads: list[int] = []
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def _result(self, mode: str) -> SimpleNamespace:
        words = [SimpleNamespace(word=w, start=s, end=e) for w, s, e in self.words]
        if mode == "verbatim":
            words = [SimpleNamespace(word="um", start=0.0, end=0.05), *words]
        return SimpleNamespace(mode=mode, text="model text is ignored", words=words)

    def transcribe_dual(self, audio: Any, *, modes: tuple[str, ...], **kwargs: Any) -> Any:
        self.threads.append(threading.get_ident())
        self.calls.append(("dual", modes))
        if self.fail:
            raise RuntimeError(f"decoder failed near {SENTINEL}")
        assert kwargs == {"sr": 16_000, "language": "en", "word_timestamps": True}
        return tuple(self._result(m) for m in modes)

    def transcribe(self, audio: Any, *, mode: str, **kwargs: Any) -> Any:
        self.threads.append(threading.get_ident())
        self.calls.append(("single", (mode,)))
        if self.fail:
            raise RuntimeError(f"decoder failed near {SENTINEL}")
        return self._result(mode)


def _transcriber(
    config: JobConfig, model: FakeModel, tmp_path: Path, load_threads: list[int] | None = None
) -> CrisperWhisperTranscriber:
    def load(model_dir: Path, device: str, compute_type: str) -> FakeModel:
        if load_threads is not None:
            load_threads.append(threading.get_ident())
        return model

    return CrisperWhisperTranscriber(config, tmp_path, COMMIT, load_model=load)


@pytest.fixture
def config(stub_job: StubJob) -> JobConfig:
    return load_config(stub_job(compute={"device": "cpu"}))


def test_every_model_call_runs_on_one_dedicated_thread(config: JobConfig, tmp_path: Path) -> None:
    model = FakeModel([("hi.", 0.1, 0.5)])
    load_threads: list[int] = []
    transcriber = _transcriber(config, model, tmp_path, load_threads)
    for _ in range(3):
        transcriber.transcribe(_wave(), "f1", [VERB, INT])
    transcriber.transcribe(_wave(), "f1", [INT])
    threads = set(load_threads + model.threads)
    assert len(threads) == 1 and threading.get_ident() not in threads


def test_both_modes_use_one_dual_pass(config: JobConfig, tmp_path: Path) -> None:
    model = FakeModel([("hello", 0.1, 0.5), ("there.", 0.6, 1.0)])
    result = _transcriber(config, model, tmp_path).transcribe(_wave(), "f1", [VERB, INT])
    assert model.calls == [("dual", ("verbatim", "intended"))]
    assert set(result) == {VERB, INT}
    assert result[VERB].segments[0].text == "um hello there."
    assert result[INT].segments[0].text == "hello there."


def test_single_mode_uses_single_pass_and_returns_only_that_mode(
    config: JobConfig, tmp_path: Path
) -> None:
    model = FakeModel([("hello.", 0.1, 0.5)])
    result = _transcriber(config, model, tmp_path).transcribe(_wave(), "f1", [INT])
    assert model.calls == [("single", ("intended",))]
    assert set(result) == {INT}


def test_model_failure_is_a_coded_error_without_text(config: JobConfig, tmp_path: Path) -> None:
    transcriber = _transcriber(config, FakeModel([], fail=True), tmp_path)
    with pytest.raises(PipelineError) as info:
        transcriber.transcribe(_wave(), "f1", [VERB, INT])
    assert info.value.code == TRANSCRIPTION_FAILED
    assert info.value.detail == "" and info.value.__cause__ is None


def test_wrong_sample_rate_is_rejected(config: JobConfig, tmp_path: Path) -> None:
    transcriber = _transcriber(config, FakeModel([]), tmp_path)
    with pytest.raises(PipelineError) as info:
        transcriber.transcribe(_wave(rate=8_000), "f1", [VERB])
    assert info.value.code == UNSUPPORTED_SAMPLE_RATE


def test_load_failure_is_a_setup_error(config: JobConfig, tmp_path: Path) -> None:
    def load(model_dir: Path, device: str, compute_type: str) -> Any:
        raise RuntimeError(SENTINEL)

    with pytest.raises(BackendSetupError) as info:
        CrisperWhisperTranscriber(config, tmp_path, COMMIT, load_model=load)
    assert SENTINEL not in str(info.value)


def test_cpu_runs_in_float32(config: JobConfig, tmp_path: Path) -> None:
    seen: list[tuple[str, str]] = []

    def load(model_dir: Path, device: str, compute_type: str) -> FakeModel:
        seen.append((device, compute_type))
        return FakeModel([])

    transcriber = CrisperWhisperTranscriber(config, tmp_path, COMMIT, load_model=load)
    assert seen == [("cpu", "float32")]
    assert transcriber.model_version == f"{MODEL_ID}@{COMMIT[:12]} (ct2 float32)"


# --- through the runner -------------------------------------------------------------------------


def _run_with(config: JobConfig, transcriber: CrisperWhisperTranscriber) -> tuple[Any, str]:
    console = io.StringIO()
    backends = Backends(
        audio_loader=StubAudioLoader(),
        diarizer=StubDiarizer(),
        transcriber=transcriber,
        ner=StubNERDetector(),
        llm=StubLLMClient(response={"entities": []}),
    )
    summary = run_job(config, "job", backends=backends, console=console)
    return summary, console.getvalue()


def _until_transcribe(config: JobConfig, **transcription: Any) -> JobConfig:
    return config.model_copy(
        update={
            "stages": config.stages.model_copy(update={"end": Stage.TRANSCRIBE}),
            "transcription": config.transcription.model_copy(update=transcription),
        }
    )


def test_intended_only_writes_only_intended_transcript(config: JobConfig, tmp_path: Path) -> None:
    config = _until_transcribe(config, modes=["intended"])
    summary, _ = _run_with(config, _transcriber(config, FakeModel([("hi.", 0.1, 0.5)]), tmp_path))
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    assert summary.done and not summary.failed
    for file_id in summary.done:
        assert store.exists(file_id, Stage.TRANSCRIBE, INT.value)
        assert not store.exists(file_id, Stage.TRANSCRIBE, VERB.value)
        transcript = store.read(file_id, Stage.TRANSCRIBE, Transcript, INT.value)
        assert transcript.segments[0].text == "hi."
        txt = store.artifact_path(file_id, Stage.TRANSCRIBE, INT.value, suffix=".txt")
        assert txt.is_file()


def _job_outputs(config: JobConfig) -> str:
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    texts = [p.read_text() for p in (store.job_dir / "logs").iterdir()]
    texts.append((store.job_dir / "metrics.json").read_text())
    texts.append(store.manifest_path.read_text())
    return "".join(texts)


def test_sentinel_never_reaches_logs_metrics_or_console(config: JobConfig, tmp_path: Path) -> None:
    config = _until_transcribe(config)
    model = FakeModel(
        [("my", 0.1, 0.2), ("name", 0.3, 0.4), ("is", 0.5, 0.6), (SENTINEL, 0.7, 1.0)]
    )
    summary, console = _run_with(config, _transcriber(config, model, tmp_path))
    assert summary.done and not summary.failed
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    transcript = store.read(summary.done[0], Stage.TRANSCRIBE, Transcript, VERB.value)
    assert SENTINEL in transcript.segments[0].text  # it really went through the stage
    assert SENTINEL not in console and SENTINEL not in _job_outputs(config)


def test_sentinel_in_model_error_never_reaches_logs(config: JobConfig, tmp_path: Path) -> None:
    config = _until_transcribe(config)
    summary, console = _run_with(config, _transcriber(config, FakeModel([], fail=True), tmp_path))
    assert summary.failed and {f.error_code for f in summary.failed} == {TRANSCRIPTION_FAILED}
    assert SENTINEL not in console and SENTINEL not in _job_outputs(config)


# --- backend selection (D4) -------------------------------------------------------------------


def test_transformers_backend_is_not_supported() -> None:
    with pytest.raises(BackendSetupError, match="not supported yet"):
        check_backend("transformers")


def test_auto_without_ct2_is_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crisper, "_ct2_available", lambda: False)
    with pytest.raises(BackendSetupError, match="not supported yet"):
        check_backend("auto")
    with pytest.raises(BackendSetupError, match="not supported yet"):
        check_backend("ct2")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(BackendSetupError, match="must be"):
        check_backend("onnx")


@pytest.mark.parametrize("command", [["run"], ["models", "check"], ["models", "download"]])
def test_cli_exits_2_for_transformers_backend(stub_job: StubJob, command: list[str]) -> None:
    config_path = stub_job(transcription={"model": MODEL_ID, "backend": "transformers"})
    result = CliRunner().invoke(app, [*command, "-c", str(config_path)])
    assert result.exit_code == 2
    assert "not supported yet" in result.stderr


def test_word_timestamps_off_fails_setup(stub_job: StubJob) -> None:
    config = load_config(stub_job(transcription={"model": MODEL_ID, "word_timestamps": False}))
    with pytest.raises(BackendSetupError, match="word_timestamps"):
        create_crisperwhisper(config)


# --- weights and conversion (D17) ---------------------------------------------------------------


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, Path]]:
    """Empty Hugging Face and CrisperWhisper caches for this test."""
    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    ct2 = tmp_path / "crisperwhisper"
    monkeypatch.setenv("CRISPERWHISPER_CACHE", str(ct2))
    yield hub, ct2


def _cache_weights(hub: Path) -> Path:
    folder = hub / ("models--" + MODEL_ID.replace("/", "--"))
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(COMMIT)
    snapshot = folder / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_text("weights")
    return snapshot


def _fake_ensure_ct2(calls: list[str]) -> Any:
    def ensure(model: str, quantization: str, cache_dir: str) -> Path:
        calls.append(quantization)
        out = Path(cache_dir) / "converted"
        out.mkdir()
        (out / "model.bin").write_text("ct2")
        return out

    return ensure


def test_registry_declares_weights_and_conversion(stub_job: StubJob) -> None:
    config = load_config(stub_job(transcription={"model": MODEL_ID}))
    plan = registry.required_weights(config, [BackendKind.TRANSCRIBER])
    assert plan.weights == [
        WeightSpec(MODEL_ID, allow_patterns=("*.json", "*.txt", "*.safetensors"))
    ]
    assert [m.label for m in plan.prepared] == [f"{MODEL_ID} (CTranslate2 float16)"]


def test_convert_writes_a_finished_folder_and_no_staging(
    caches: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import crisperwhisper.converter

    hub, _ = caches
    _cache_weights(hub)
    calls: list[str] = []
    monkeypatch.setattr(crisperwhisper.converter, "ensure_ct2_model", _fake_ensure_ct2(calls))
    assert not crisper.is_converted()

    crisper.convert()
    crisper.convert()  # already done: no second conversion

    final = converted_dir(COMMIT)
    assert calls == ["float16"]
    assert crisper.is_converted()
    assert (final / "model.bin").is_file() and (final / READY_MARKER).is_file()
    assert sorted(p.name for p in final.parent.iterdir()) == [final.name]


def test_unfinished_conversion_does_not_count(caches: tuple[Path, Path]) -> None:
    hub, _ = caches
    _cache_weights(hub)
    final = converted_dir(COMMIT)
    final.mkdir(parents=True)
    (final / "model.bin").write_text("partial")
    assert not crisper.is_converted()


def test_unconverted_model_fails_check_and_run_and_download_converts(
    stub_job: StubJob, caches: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import crisperwhisper.converter

    hub, _ = caches
    _cache_weights(hub)
    config_path = stub_job(transcription={"model": MODEL_ID})
    label = f"{MODEL_ID} (CTranslate2 float16)"

    check = CliRunner().invoke(app, ["models", "check", "-c", str(config_path)])
    assert check.exit_code == 2 and f"unprepared   {label}" in check.stdout
    run = CliRunner().invoke(app, ["run", "-c", str(config_path)])
    assert run.exit_code == 2 and label in run.stderr and "murmurs models download" in run.stderr

    calls: list[str] = []
    monkeypatch.setattr(crisperwhisper.converter, "ensure_ct2_model", _fake_ensure_ct2(calls))
    monkeypatch.setattr(
        "murmurs.cli.download_weight",
        lambda spec: SimpleNamespace(revision=COMMIT, size_bytes=10),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    download = CliRunner().invoke(app, ["models", "download", "-c", str(config_path)])
    assert download.exit_code == 0, download.output
    assert f"preparing    {label}" in download.stdout and calls == ["float16"]

    check = CliRunner().invoke(app, ["models", "check", "-c", str(config_path)])
    assert check.exit_code == 0 and f"prepared     {label}" in check.stdout


def test_prepare_failure_reports_type_only(
    stub_job: StubJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        raise RuntimeError(f"conversion failed at {SENTINEL}")

    registry.register(
        BackendKind.TRANSCRIBER,
        "stub-prepared",
        lambda _: None,
        prepared=lambda _: [PreparedModel("stub-model (converted)", lambda: False, fail)],
    )
    try:
        config_path = stub_job(transcription={"model": "stub-prepared"})
        result = CliRunner().invoke(app, ["models", "download", "-c", str(config_path)])
    finally:
        registry._registry.pop((BackendKind.TRANSCRIBER, "stub-prepared"))
    assert result.exit_code == 2
    assert "failed       stub-model (converted) (RuntimeError)" in result.stderr
    assert SENTINEL not in result.output


def test_unprepared_error_names_models() -> None:
    error = UnpreparedModelError(["a (x)", "b (y)"])
    assert "a (x), b (y)" in str(error) and "murmurs models download" in str(error)


# --- the CTranslate2 fork (CLAUDE.md dependency notes) ----------------------------------------


def test_ctranslate2_comes_from_the_crisperwhisper_fork() -> None:
    owners = importlib.metadata.packages_distributions().get("ctranslate2", [])
    assert owners == ["ctranslate2-crisperwhisper"]
    installed = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
    assert "faster-whisper" not in installed and "ctranslate2" not in installed


# --- real model (only with -m gpu) --------------------------------------------------------------

EXAMPLE_DIR = os.environ.get("MURMURS_EXAMPLE_AUDIO_DIR")


@pytest.mark.gpu
@pytest.mark.skipif(not EXAMPLE_DIR, reason="MURMURS_EXAMPLE_AUDIO_DIR is not set")
def test_example_audio_transcribes_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    raw = {
        "data": {"input_dir": EXAMPLE_DIR, "output_dir": str(tmp_path / "out")},
        "stages": {"end": "transcribe"},
        "diarization": {"model": "stub", "llm_correction": False},
        "transcription": {"model": MODEL_ID},
        "deid": {
            "ner_model": "stub",
            "llm_model": "stub",
            "entity_types": [{"name": "PERSON", "tag": "[PERSON_{n}]"}],
        },
    }
    config_path = tmp_path / "job.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    config = load_config(config_path)

    summary = run_job(config, "job", console=io.StringIO())

    assert summary.done and not summary.failed
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["model_versions"]["transcriber"].startswith(MODEL_ID)
    assert SPEC.label in manifest["model_versions"]
    for file_id in summary.done:
        duration = store.read(file_id, Stage.LOAD_AUDIO, AudioInfo).duration_s
        for mode in (VERB, INT):
            transcript = store.read(file_id, Stage.TRANSCRIBE, Transcript, mode.value)
            words = [w for s in transcript.segments for w in s.words]
            assert words, mode
            starts = [w.start for w in words]
            assert starts == sorted(starts)
            assert words[-1].end <= duration + 1e-6
