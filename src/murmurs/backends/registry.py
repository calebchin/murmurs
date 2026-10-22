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

Each registration can declare the model weights its backend needs (see
``murmurs.offline``), so they can be downloaded ahead of time and checked before a run.
Factories must import model libraries inside the factory, never at module level here,
so that offline mode is in force before those libraries are imported.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from murmurs.backends import AudioLoader, Diarizer, LLMClient, NERDetector, Transcriber
from murmurs.backends.audio import TorchcodecAudioLoader
from murmurs.backends.crisper import MODEL_ID as CRISPERWHISPER_ID
from murmurs.backends.crisper import (
    create_crisperwhisper,
    crisperwhisper_prepared,
    crisperwhisper_weights,
)
from murmurs.backends.gptoss import MODEL_ID as GPT_OSS_ID
from murmurs.backends.gptoss import create_gpt_oss, gpt_oss_prepared, gpt_oss_weights
from murmurs.backends.nemotron import MODEL_ID as NEMOTRON_ID
from murmurs.backends.nemotron import NemotronDiarizer, nemotron_weights
from murmurs.backends.openmed_ner import MODEL_ID as OPENMED_ID
from murmurs.backends.openmed_ner import OpenMedNER, openmed_weights
from murmurs.backends.stubs import (
    StubAudioLoader,
    StubDiarizer,
    StubLLMClient,
    StubNERDetector,
    StubTranscriber,
)
from murmurs.config import JobConfig
from murmurs.errors import BackendSetupError
from murmurs.offline import PreparedModel, WeightSpec

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
WeightsFn = Callable[[JobConfig], Sequence[WeightSpec]]
"""Returns the weights a backend needs for a given job config."""
PreparedFn = Callable[[JobConfig], Sequence[PreparedModel]]
"""Returns the models a backend derives from its weights (e.g. format conversions)."""


@dataclass(frozen=True)
class _Entry:
    factory: Factory
    weights: WeightsFn | None = None
    prepared: PreparedFn | None = None


@dataclass(frozen=True)
class WeightsPlan:
    """The weights a set of backends needs, and any backends with no implementation."""

    weights: list[WeightSpec]
    """Deduplicated, in declaration order."""
    unavailable: list[str]
    """One message per selected backend name that is not registered."""
    prepared: list[PreparedModel] = field(default_factory=list)
    """Models built from the weights after download; deduplicated by label."""


_PROTOCOLS: dict[BackendKind, type] = {
    BackendKind.AUDIO_LOADER: AudioLoader,
    BackendKind.DIARIZER: Diarizer,
    BackendKind.TRANSCRIBER: Transcriber,
    BackendKind.NER: NERDetector,
    BackendKind.LLM: LLMClient,
}

_registry: dict[tuple[BackendKind, str], _Entry] = {}


def register(
    kind: BackendKind,
    name: str,
    factory: Factory,
    weights: WeightsFn | None = None,
    prepared: PreparedFn | None = None,
) -> None:
    """Register ``factory`` as the implementation of ``kind`` called ``name``.

    ``weights`` declares the model weights the backend loads; omit it for backends that
    load none (stubs, the audio loader). ``prepared`` declares models built from those
    weights once, after download (see ``PreparedModel``).
    """
    _registry[(kind, name)] = _Entry(factory, weights, prepared)


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


def _unavailable(kind: BackendKind, name: str) -> str:
    return f"no {kind.value} backend named '{name}' is available"


def required_weights(config: JobConfig, kinds: Iterable[BackendKind]) -> WeightsPlan:
    """The weights the backends of ``kinds`` selected by ``config`` declare."""
    weights: dict[WeightSpec, None] = {}
    prepared: dict[str, PreparedModel] = {}
    unavailable: list[str] = []
    for kind in kinds:
        name = backend_name(kind, config)
        entry = _registry.get((kind, name))
        if entry is None:
            unavailable.append(_unavailable(kind, name))
            continue
        if entry.weights is not None:
            weights.update(dict.fromkeys(entry.weights(config)))
        if entry.prepared is not None:
            for model in entry.prepared(config):
                prepared.setdefault(model.label, model)
    return WeightsPlan(
        weights=list(weights), unavailable=unavailable, prepared=list(prepared.values())
    )


def _create(kind: BackendKind, config: JobConfig) -> Any:
    name = backend_name(kind, config)
    entry = _registry.get((kind, name))
    if entry is None:
        raise BackendSetupError(_unavailable(kind, name))
    backend = entry.factory(config)
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
register(
    BackendKind.AUDIO_LOADER,
    DEFAULT_AUDIO_LOADER,
    lambda config: TorchcodecAudioLoader(config.data.audio_extensions),
)
register(BackendKind.DIARIZER, STUB, lambda _: StubDiarizer())
register(BackendKind.DIARIZER, NEMOTRON_ID, NemotronDiarizer, weights=nemotron_weights)
register(BackendKind.TRANSCRIBER, STUB, lambda _: StubTranscriber())
register(
    BackendKind.TRANSCRIBER,
    CRISPERWHISPER_ID,
    create_crisperwhisper,
    weights=crisperwhisper_weights,
    prepared=crisperwhisper_prepared,
)
register(BackendKind.NER, STUB, lambda _: StubNERDetector())
register(BackendKind.NER, OPENMED_ID, OpenMedNER, weights=openmed_weights)
register(BackendKind.LLM, STUB, lambda _: StubLLMClient())
register(
    BackendKind.LLM,
    GPT_OSS_ID,
    create_gpt_oss,
    weights=gpt_oss_weights,
    prepared=gpt_oss_prepared,
)
