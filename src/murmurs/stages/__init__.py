"""Pipeline stages.

Each stage reads its inputs and writes its artifacts through the artifact store, and
calls models only through ``murmurs.backends`` protocols. ``assign_speakers`` and
``detect_phi`` are phase 1 placeholders, replaced in P2-05 and phase 3.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from murmurs.backends import AudioLoader, Diarizer, NERDetector, Transcriber, Waveform
from murmurs.config import JobConfig, Stage
from murmurs.errors import PipelineError
from murmurs.models import (
    UNKNOWN_SPEAKER,
    AudioInfo,
    Diarization,
    RedactionLayer,
    Segment,
    SpeakerTurn,
    Transcript,
    TranscriptMode,
    TranscriptRef,
    Word,
)
from murmurs.store import ArtifactStore

M = TypeVar("M", bound=BaseModel)
B = TypeVar("B")

MISSING_UPSTREAM = "MISSING_UPSTREAM"
INVALID_INPUT = "INVALID_INPUT"

DEPENDS: dict[Stage, tuple[Stage, ...]] = {
    Stage.INGEST: (),
    Stage.LOAD_AUDIO: (Stage.INGEST,),
    Stage.DIARIZE: (Stage.LOAD_AUDIO,),
    Stage.TRANSCRIBE: (Stage.LOAD_AUDIO,),
    Stage.ASSIGN_SPEAKERS: (Stage.DIARIZE, Stage.TRANSCRIBE),
    Stage.DETECT_PHI: (Stage.ASSIGN_SPEAKERS,),
}
"""Which earlier stages' outputs each stage uses. Resume reruns a stage's dependents."""

PER_MODE_STAGES = frozenset({Stage.TRANSCRIBE, Stage.ASSIGN_SPEAKERS, Stage.DETECT_PHI})
"""Stages that write one artifact per transcript mode (as the artifact variant)."""

WAVEFORM_STAGES = frozenset({Stage.LOAD_AUDIO, Stage.DIARIZE, Stage.TRANSCRIBE})
"""Stages that use the decoded waveform. It is freed after the last of them in a run."""


@dataclass
class Backends:
    """The backends a job uses. Only those needed by the stage range are set."""

    audio_loader: AudioLoader | None = None
    diarizer: Diarizer | None = None
    transcriber: Transcriber | None = None
    ner: NERDetector | None = None


@dataclass
class StageContext:
    """Everything a stage needs to process one file."""

    config: JobConfig
    store: ArtifactStore
    backends: Backends
    file_id: str
    path: Path
    input_hash: str
    modes: tuple[TranscriptMode, ...]
    _waveform: Waveform | None = field(default=None, repr=False)

    def waveform(self) -> Waveform:
        """The decoded audio, loaded on first use (e.g. when resuming after load_audio)."""
        if self._waveform is None:
            self._waveform = _need(self.backends.audio_loader).load(self.path, self.file_id)
        return self._waveform

    def release_waveform(self) -> None:
        """Drop this context's reference to the decoded audio so its memory can be freed."""
        self._waveform = None

    def read(self, stage: Stage, model: type[M], variant: str | None = None) -> M:
        """Read an upstream artifact; a missing one is a ``MISSING_UPSTREAM`` failure."""
        if not self.store.exists(self.file_id, stage, variant):
            raise PipelineError(MISSING_UPSTREAM)
        return self.store.read(self.file_id, stage, model, variant)


def stage_variants(stage: Stage, modes: tuple[TranscriptMode, ...]) -> list[str | None]:
    """The artifact variants a stage writes: one per mode, or a single unnamed one."""
    return [mode.value for mode in modes] if stage in PER_MODE_STAGES else [None]


def stage_config(stage: Stage, config: JobConfig) -> dict[str, Any]:
    """The config values that affect a stage's output; changing them reruns the stage."""
    match stage:
        case Stage.INGEST | Stage.LOAD_AUDIO:
            return {}
        case Stage.DIARIZE:
            return config.diarization.model_dump(mode="json", include={"model", "num_speakers"})
        case Stage.TRANSCRIBE:
            return config.transcription.model_dump(mode="json")
        case Stage.ASSIGN_SPEAKERS:
            values = config.diarization.model_dump(
                mode="json", include={"speaker_gap_tolerance_s", "llm_correction"}
            )
            if config.diarization.llm_correction:
                values["llm_model"] = config.deid.llm_model
            return values
        case Stage.DETECT_PHI:
            return config.deid.model_dump(mode="json")


# --- stage implementations ---------------------------------------------------------------


def ingest(ctx: StageContext) -> None:
    """Probe the input file and record a ``ValidationResult``; invalid files fail here."""
    result = _need(ctx.backends.audio_loader).probe(ctx.path, ctx.file_id)
    ctx.store.write(ctx.file_id, Stage.INGEST, result)
    if not result.valid:
        raise PipelineError(result.error.code if result.error else INVALID_INPUT)


def load_audio(ctx: StageContext) -> None:
    """Decode the audio into memory and record its ``AudioInfo``."""
    waveform = _need(ctx.backends.audio_loader).load(ctx.path, ctx.file_id)
    ctx._waveform = waveform
    info = AudioInfo(
        file_id=ctx.file_id,
        input_hash=ctx.input_hash,
        sample_rate=waveform.sample_rate,
        n_samples=len(waveform.samples),
        duration_s=waveform.duration_s,
    )
    ctx.store.write(ctx.file_id, Stage.LOAD_AUDIO, info)


def diarize(ctx: StageContext) -> None:
    """Find speaker turns over the whole recording."""
    result = _need(ctx.backends.diarizer).diarize(ctx.waveform(), ctx.file_id)
    ctx.store.write(ctx.file_id, Stage.DIARIZE, result)


def transcribe(ctx: StageContext) -> None:
    """Transcribe the recording once per configured mode."""
    results = _need(ctx.backends.transcriber).transcribe(ctx.waveform(), ctx.file_id, ctx.modes)
    for mode in ctx.modes:
        if mode not in results:
            raise PipelineError("MISSING_TRANSCRIPT")
        ctx.store.write(ctx.file_id, Stage.TRANSCRIBE, results[mode], variant=mode.value)
        ctx.store.write_transcript_text(
            ctx.file_id, Stage.TRANSCRIBE, results[mode], variant=mode.value
        )


def assign_speakers(ctx: StageContext) -> None:
    """Label words with speakers. Placeholder until P2-05: midpoint lookup, no regrouping."""
    turns = ctx.read(Stage.DIARIZE, Diarization).turns
    for mode in ctx.modes:
        transcript = ctx.read(Stage.TRANSCRIBE, Transcript, variant=mode.value)
        labeled = _label_by_midpoint(transcript, turns)
        ctx.store.write(ctx.file_id, Stage.ASSIGN_SPEAKERS, labeled, variant=mode.value)
        ctx.store.write_transcript_text(
            ctx.file_id, Stage.ASSIGN_SPEAKERS, labeled, variant=mode.value
        )


def detect_phi(ctx: StageContext) -> None:
    """Propose redactions. Placeholder until phase 3: NER over the joined segment text."""
    ner = _need(ctx.backends.ner)
    for mode in ctx.modes:
        transcript = ctx.read(Stage.ASSIGN_SPEAKERS, Transcript, variant=mode.value)
        text = " ".join(segment.text for segment in transcript.segments)
        layer = RedactionLayer(
            transcript=TranscriptRef(file_id=ctx.file_id, mode=mode),
            spans=ner.detect(text, ctx.file_id),
        )
        ctx.store.write(ctx.file_id, Stage.DETECT_PHI, layer, variant=mode.value)


STAGE_FUNCTIONS: dict[Stage, Callable[[StageContext], None]] = {
    Stage.INGEST: ingest,
    Stage.LOAD_AUDIO: load_audio,
    Stage.DIARIZE: diarize,
    Stage.TRANSCRIBE: transcribe,
    Stage.ASSIGN_SPEAKERS: assign_speakers,
    Stage.DETECT_PHI: detect_phi,
}
"""Stage implementations, in pipeline order."""


# --- helpers -------------------------------------------------------------------------------


def _need(backend: B | None) -> B:
    if backend is None:
        raise RuntimeError("backend required by this stage was not created")
    return backend


def _speaker_at(time: float, turns: list[SpeakerTurn]) -> str:
    for turn in turns:
        if turn.start <= time <= turn.end:
            return turn.speaker
    return UNKNOWN_SPEAKER


def _label_by_midpoint(transcript: Transcript, turns: list[SpeakerTurn]) -> Transcript:
    segments = []
    for segment in transcript.segments:
        words = [
            Word(
                text=w.text,
                start=w.start,
                end=w.end,
                speaker=_speaker_at((w.start + w.end) / 2, turns),
            )
            for w in segment.words
        ]
        counts = Counter(w.speaker for w in words)
        speaker = counts.most_common(1)[0][0] if words else UNKNOWN_SPEAKER
        segments.append(
            Segment(
                speaker=speaker,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=words,
            )
        )
    return Transcript(file_id=transcript.file_id, mode=transcript.mode, segments=segments)
