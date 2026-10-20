"""Choose backend implementations by name from the job config.

The name for each kind comes from an existing config field, so no extra config is needed:

- transcriber: ``transcription.model``
- diarizer: ``diarization.model``
- NER detector: ``deid.ner_model``
- LLM client: ``deid.llm_model``
- audio loader: no config field; always ``"default"``

The name ``"stub"`` selects the stub for any kind. Real backends register under their
model names in later phases. Backends are created on demand, so a run that stops
before PHI detection never loads the NER model or the LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from murmurs.backends import AudioLoader, Diarizer, LLMClient, NERDetector, Transcriber
from murmurs.backends.stubs import (
    StubAudioLoader,
    StubDiarizer,
    StubLLMClient,
    StubNERDetector,
    StubTranscriber,
)
from murmurs.config import JobConfig
from murmurs.errors import BackendSetupError

STUB = "stub"
DEFAULT_AUDIO_LOADER = "default"


class BackendKind(StrEnum):
    """The kinds of backend a job uses."""

    AUDIO_LOADER = "audio_loader"
    DIARIZER = "diarizer"
    TRANSCRIBER = "transcriber"
    NER = "ner"
    LLM = "llm"


Factory = Callable[[JobConfig], Any]

_PROTOCOLS: dict[BackendKind, type] = {
    BackendKind.AUDIO_LOADER: AudioLoader,
    BackendKind.DIARIZER: Diarizer,
    BackendKind.TRANSCRIBER: Transcriber,
    BackendKind.NER: NERDetector,
    BackendKind.LLM: LLMClient,
}

_registry: dict[tuple[BackendKind, str], Factory] = {}


def register(kind: BackendKind, name: str, factory: Factory) -> None:
    """Register ``factory`` as the implementation of ``kind`` called ``name``."""
    _registry[(kind, name)] = factory


def backend_name(kind: BackendKind, config: JobConfig) -> str:
    """The implementation name the config selects for ``kind``."""
    match kind:
        case BackendKind.AUDIO_LOADER:
            return DEFAULT_AUDIO_LOADER
        case BackendKind.DIARIZER:
            return config.diarization.model
        case BackendKind.TRANSCRIBER:
            return config.transcription.model
        case BackendKind.NER:
            return config.deid.ner_model
        case BackendKind.LLM:
            return config.deid.llm_model


def _create(kind: BackendKind, config: JobConfig) -> Any:
    name = backend_name(kind, config)
    factory = _registry.get((kind, name))
    if factory is None:
        raise BackendSetupError(f"no {kind.value} backend named '{name}' is available")
    backend = factory(config)
    if not isinstance(backend, _PROTOCOLS[kind]):
        raise BackendSetupError(f"{kind.value} backend '{name}' does not satisfy its protocol")
    return backend


def create_audio_loader(config: JobConfig) -> AudioLoader:
    """Create the audio loader the config selects."""
    backend: AudioLoader = _create(BackendKind.AUDIO_LOADER, config)
    return backend


def create_diarizer(config: JobConfig) -> Diarizer:
    """Create the diarizer the config selects."""
    backend: Diarizer = _create(BackendKind.DIARIZER, config)
    return backend


def create_transcriber(config: JobConfig) -> Transcriber:
    """Create the transcriber the config selects."""
    backend: Transcriber = _create(BackendKind.TRANSCRIBER, config)
    return backend


def create_ner_detector(config: JobConfig) -> NERDetector:
    """Create the NER detector the config selects."""
    backend: NERDetector = _create(BackendKind.NER, config)
    return backend


def create_llm_client(config: JobConfig) -> LLMClient:
    """Create the LLM client the config selects."""
    backend: LLMClient = _create(BackendKind.LLM, config)
    return backend


register(BackendKind.AUDIO_LOADER, STUB, lambda _: StubAudioLoader())
# Phase 1 has no real audio loader; P2-02 points "default" at the torchcodec loader.
register(BackendKind.AUDIO_LOADER, DEFAULT_AUDIO_LOADER, lambda _: StubAudioLoader())
register(BackendKind.DIARIZER, STUB, lambda _: StubDiarizer())
register(BackendKind.TRANSCRIBER, STUB, lambda _: StubTranscriber())
register(BackendKind.NER, STUB, lambda _: StubNERDetector())
register(BackendKind.LLM, STUB, lambda _: StubLLMClient())
