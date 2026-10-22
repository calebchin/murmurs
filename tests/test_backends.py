"""P1-06: backend interfaces and stubs."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from murmurs.backends import (
    AudioLoader,
    Diarizer,
    LLMClient,
    NERDetector,
    Transcriber,
    Waveform,
    registry,
)
from murmurs.backends.audio import TorchcodecAudioLoader
from murmurs.backends.registry import BackendKind
from murmurs.backends.stubs import (
    STUB_FAILURE,
    StubAudioLoader,
    StubDiarizer,
    StubLLMClient,
    StubNERDetector,
    StubTranscriber,
)
from murmurs.config import JobConfig, load_config
from murmurs.errors import BackendSetupError, PipelineError
from murmurs.models import TranscriptMode

SENTINEL = "SENTINEL_PHI_7f3a"
EXAMPLE = Path(__file__).parent.parent / "examples" / "job.yaml"
BOTH_MODES = (TranscriptMode.VERBATIM, TranscriptMode.INTENDED)


def _wave(seconds: float = 2.5) -> Waveform:
    return Waveform(samples=np.zeros(int(seconds * 16_000), dtype=np.float32), sample_rate=16_000)


def _stub_config() -> JobConfig:
    config = load_config(EXAMPLE)
    return config.model_copy(
        update={
            "diarization": config.diarization.model_copy(update={"model": "stub"}),
            "transcription": config.transcription.model_copy(update={"model": "stub"}),
            "deid": config.deid.model_copy(update={"ner_model": "stub", "llm_model": "stub"}),
        }
    )


@pytest.mark.parametrize(
    ("stub", "protocol"),
    [
        (StubAudioLoader(), AudioLoader),
        (StubDiarizer(), Diarizer),
        (StubTranscriber(), Transcriber),
        (StubNERDetector(), NERDetector),
        (StubLLMClient(), LLMClient),
    ],
    ids=lambda x: type(x).__name__ if not isinstance(x, type) else x.__name__,
)
def test_stub_satisfies_protocol(stub: object, protocol: type) -> None:
    assert isinstance(stub, protocol)
    assert stub.model_version == "stub"


def test_audio_loader_probe_and_load(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"not really audio")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    loader = StubAudioLoader(duration_s=1.5)

    result = loader.probe(audio, "f1")
    assert result.valid and result.format == "wav" and result.duration_s == 1.5
    bad = loader.probe(empty, "f2")
    assert not bad.valid and bad.error is not None and bad.error.code == "EMPTY"

    waveform = loader.load(audio, "f1")
    assert waveform.samples.dtype == np.float32
    assert waveform.sample_rate == 16_000
    assert waveform.duration_s == pytest.approx(1.5)
    assert audio.read_bytes() == b"not really audio"


def test_diarizer_alternates_speakers_and_covers_audio() -> None:
    turns = StubDiarizer(turn_s=1.0).diarize(_wave(2.5), "f1").turns
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert turns[0].start == 0.0 and turns[-1].end == pytest.approx(2.5)


def test_transcriber_returns_requested_modes_with_sentinel() -> None:
    transcriber = StubTranscriber(text=f"my name is {SENTINEL}")
    result = transcriber.transcribe(_wave(), "f1", BOTH_MODES)
    assert set(result) == set(BOTH_MODES)
    for mode, transcript in result.items():
        assert transcript.mode is mode and transcript.file_id == "f1"
        assert SENTINEL in transcript.segments[0].text
        assert SENTINEL in [w.text for w in transcript.segments[0].words]
    assert result[TranscriptMode.VERBATIM].segments[0].words[0].text == "um"

    only = transcriber.transcribe(_wave(), "f1", [TranscriptMode.INTENDED])
    assert set(only) == {TranscriptMode.INTENDED}


def test_stubs_are_deterministic() -> None:
    a = StubTranscriber().transcribe(_wave(), "f1", BOTH_MODES)
    b = StubTranscriber().transcribe(_wave(), "f1", BOTH_MODES)
    assert a == b
    assert StubDiarizer().diarize(_wave(), "f1") == StubDiarizer().diarize(_wave(), "f1")


def test_ner_finds_configured_targets() -> None:
    text = f"I met {SENTINEL} and Bob, then {SENTINEL} again"
    spans = StubNERDetector({SENTINEL: "first_name", "Bob": "first_name"}).detect(text, "f1")
    assert [text[s.start : s.end] for s in spans] == [SENTINEL, "Bob", SENTINEL]
    assert all(s.entity_type == "first_name" for s in spans)


def test_llm_invalid_response_raises_coded_error() -> None:
    class Answer(BaseModel):
        speaker: str

    with pytest.raises(PipelineError) as info:
        StubLLMClient(response={"wrong": 1}).generate("p", Answer, "f1")
    assert info.value.code == "LLM_INVALID_OUTPUT" and info.value.__cause__ is None


def test_llm_returns_validated_response() -> None:
    class Answer(BaseModel):
        speaker: str
        changed: int = 0

    llm = StubLLMClient(response={"speaker": "SPEAKER_01"})
    assert llm.generate("prompt", Answer, "f1") == Answer(speaker="SPEAKER_01")


@pytest.mark.parametrize(
    ("make", "call"),
    [
        (StubAudioLoader, lambda s: s.load(Path("unused"), "bad")),
        (StubDiarizer, lambda s: s.diarize(_wave(), "bad")),
        (StubTranscriber, lambda s: s.transcribe(_wave(), "bad", BOTH_MODES)),
        (StubNERDetector, lambda s: s.detect("text", "bad")),
        (StubLLMClient, lambda s: s.generate("p", BaseModel, "bad")),
    ],
    ids=["audio", "diarizer", "transcriber", "ner", "llm"],
)
def test_stub_fails_on_configured_file_with_typed_error(
    make: Callable[..., Any], call: Callable[[Any], object]
) -> None:
    stub = make(fail_on={"bad"}, fail_detail=SENTINEL)
    with pytest.raises(PipelineError) as info:
        call(stub)
    assert info.value.code == STUB_FAILURE
    assert SENTINEL not in str(info.value)
    assert info.value.detail == SENTINEL
    assert stub.calls == ["bad"]


def test_stub_succeeds_for_other_files() -> None:
    transcriber = StubTranscriber(fail_on={"bad"})
    transcriber.transcribe(_wave(), "good", BOTH_MODES)
    with pytest.raises(PipelineError):
        transcriber.transcribe(_wave(), "bad", BOTH_MODES)
    assert transcriber.calls == ["good", "bad"]


def test_pipeline_error_requires_code_format() -> None:
    with pytest.raises(ValueError):
        PipelineError("not a code")


def test_registry_creates_stubs_from_config() -> None:
    config = _stub_config()
    # The audio loader has no config field; "default" is the torchcodec loader (P2-01).
    assert isinstance(registry.create_audio_loader(config), TorchcodecAudioLoader)
    assert isinstance(registry.create_diarizer(config), StubDiarizer)
    assert isinstance(registry.create_transcriber(config), StubTranscriber)
    assert isinstance(registry.create_ner_detector(config), StubNERDetector)
    assert isinstance(registry.create_llm_client(config), StubLLMClient)


def test_registry_unknown_name_fails_setup() -> None:
    config = load_config(EXAMPLE)  # real model names; the NER model is not registered yet
    with pytest.raises(BackendSetupError, match="no ner backend named"):
        registry.create_ner_detector(config)


def test_registry_rejects_object_that_breaks_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    registry.register(BackendKind.DIARIZER, "broken", lambda _: object())
    config = _stub_config()
    config = config.model_copy(
        update={"diarization": config.diarization.model_copy(update={"model": "broken"})}
    )
    with pytest.raises(BackendSetupError, match="does not satisfy"):
        registry.create_diarizer(config)
