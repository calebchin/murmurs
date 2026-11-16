"""Pipeline stages.

Each stage reads its inputs and writes its artifacts through the artifact store, and
calls models only through ``murmurs.backends`` protocols. ``detect_phi`` is a phase 1
placeholder, replaced in phase 3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from murmurs.backends import AudioLoader, Diarizer, LLMClient, NERDetector, Transcriber, Waveform
from murmurs.config import JobConfig, Stage
from murmurs.errors import PipelineError
from murmurs.models import (
    AudioInfo,
    Diarization,
    SpeakerCorrection,
    Transcript,
    TranscriptMode,
)
from murmurs.stages import correction as _correction
from murmurs.stages import deid as _deid
from murmurs.stages.assign import assign_speakers as assign_words
from murmurs.stages.assign import shift_turns
from murmurs.stages.correction import correct_modes, prompt_hash
from murmurs.stages.deid import deid_mode, deidentify, render_annotated, render_clean
from murmurs.stages.deid import prompt_hash as deid_prompt_hash
from murmurs.store import ArtifactStore

M = TypeVar("M", bound=BaseModel)
B = TypeVar("B")

MISSING_UPSTREAM = "MISSING_UPSTREAM"
RULE_BASED = "rule_based"
"""Variant suffix of the rule-based transcripts ``assign_speakers.<mode>.rule_based``."""
CORRECTION_REPORT = "correction"
"""Variant of the speaker-correction report, ``assign_speakers.correction``."""
DEID_REPORT = "report"
"""Variant of the PHI detection report, ``detect_phi.report``."""
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
"""Stages that use the decoded waveform."""

STAGE_GROUPS: tuple[tuple[str, tuple[Stage, ...]], ...] = (
    ("ingest", (Stage.INGEST,)),
    ("audio", (Stage.LOAD_AUDIO, Stage.DIARIZE, Stage.TRANSCRIBE)),
    ("assign_speakers", (Stage.ASSIGN_SPEAKERS,)),
    ("detect_phi", (Stage.DETECT_PHI,)),
)
"""Named stage groups, in order (D21). The runner finishes a group for every file before
starting the next. Within a group, each file runs all the group's stages before the next
file, so audio is decoded once per file and freed before the next file is decoded."""


def stage_groups(stages: list[Stage]) -> list[tuple[str, list[Stage]]]:
    """The groups with stages in ``stages``, each limited to those stages, in order."""
    groups = []
    for name, members in STAGE_GROUPS:
        selected = [s for s in members if s in stages]
        if selected:
            groups.append((name, selected))
    return groups


@dataclass
class Backends:
    """The backends a job uses. Only those needed by the stage range are set."""

    audio_loader: AudioLoader | None = None
    diarizer: Diarizer | None = None
    transcriber: Transcriber | None = None
    ner: NERDetector | None = None
    llm: LLMClient | None = None


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
    """The artifact variants a stage must have written for it to count as complete: one
    per mode, the deidentified mode only for ``detect_phi`` (D9), or a single unnamed one."""
    if stage is Stage.DETECT_PHI:
        return [deid_mode(modes).value]
    return [mode.value for mode in modes] if stage in PER_MODE_STAGES else [None]


def stage_config(stage: Stage, config: JobConfig) -> dict[str, Any]:
    """The config values that affect a stage's output; changing them reruns the stage."""
    match stage:
        case Stage.INGEST | Stage.LOAD_AUDIO:
            return {}
        case Stage.DIARIZE:
            return config.diarization.model_dump(mode="json", include={"model", "num_speakers"})
        case Stage.TRANSCRIBE:
            values = config.transcription.model_dump(mode="json")
            if values["draft_model"] is None:
                del values["draft_model"]  # keeps the hash of jobs from before the field
            return values
        case Stage.ASSIGN_SPEAKERS:
            values = config.diarization.model_dump(
                mode="json", include={"speaker_gap_tolerance_s", "turn_offset_s", "llm_correction"}
            )
            if config.diarization.llm_correction:
                values["llm_model"] = config.deid.llm_model
                values["prompt"] = prompt_hash()
                values["reasoning_effort"] = config.deid.llm_server.reasoning_effort
                values["chunk_tokens"] = _correction.CORRECTION_CHUNK_TOKENS
            return values
        case Stage.DETECT_PHI:
            # llm_server settings change speed and memory, not output
            values = config.deid.model_dump(mode="json", exclude={"llm_server"})
            values["prompt"] = deid_prompt_hash()
            values["reasoning_effort"] = config.deid.llm_server.reasoning_effort
            values["chunk_tokens"] = _deid.DEID_CHUNK_TOKENS
            values["line_tokens"] = _deid.DEID_LINE_TOKENS
            return values


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
        ctx.store.write_words_tsv(ctx.file_id, Stage.TRANSCRIBE, results[mode], variant=mode.value)


def assign_speakers(ctx: StageContext) -> None:
    """Label every word with a speaker and regroup words into speaker segments.

    Rule-based assignment (P2-05) runs for every mode with the same speaker turns, moved
    ``diarization.turn_offset_s`` earlier (D20). With
    ``diarization.llm_correction`` on, the LLM then fixes label mistakes in one mode and
    the other follows by word time (P2-07, D24). Writes the final transcripts (used
    downstream), the rule-based ones (``<mode>.rule_based``) and a counts-only
    ``correction`` report, whether or not correction ran.
    """
    turns = shift_turns(
        ctx.read(Stage.DIARIZE, Diarization).turns, ctx.config.diarization.turn_offset_s
    )
    tolerance = ctx.config.diarization.speaker_gap_tolerance_s
    rule_based = {
        mode: assign_words(
            ctx.read(Stage.TRANSCRIBE, Transcript, variant=mode.value), turns, tolerance
        )
        for mode in ctx.modes
    }
    final: dict[TranscriptMode, Transcript] = dict(rule_based)
    report = SpeakerCorrection(file_id=ctx.file_id, enabled=False)
    if ctx.config.diarization.llm_correction:
        final, report = correct_modes(
            rule_based,
            _need(ctx.backends.llm),
            ctx.file_id,
            ctx.config.deid.llm_server.max_model_len,
            max_concurrency=ctx.config.deid.llm_server.max_concurrency,
        )
    for mode in ctx.modes:
        variant = f"{mode.value}.{RULE_BASED}"
        ctx.store.write(ctx.file_id, Stage.ASSIGN_SPEAKERS, rule_based[mode], variant=variant)
        ctx.store.write_transcript_text(
            ctx.file_id, Stage.ASSIGN_SPEAKERS, rule_based[mode], variant=variant
        )
        ctx.store.write_words_tsv(
            ctx.file_id, Stage.ASSIGN_SPEAKERS, rule_based[mode], variant=variant
        )
        ctx.store.write(ctx.file_id, Stage.ASSIGN_SPEAKERS, final[mode], variant=mode.value)
        ctx.store.write_transcript_text(
            ctx.file_id, Stage.ASSIGN_SPEAKERS, final[mode], variant=mode.value
        )
        ctx.store.write_words_tsv(
            ctx.file_id, Stage.ASSIGN_SPEAKERS, final[mode], variant=mode.value
        )
    ctx.store.write(ctx.file_id, Stage.ASSIGN_SPEAKERS, report, variant=CORRECTION_REPORT)


def detect_phi(ctx: StageContext) -> None:
    """Propose redactions on the final transcript of one mode (verbatim when transcribed,
    D9) with NER and the LLM.

    Writes the redaction layer (``<mode>.json``), an annotated rendering for reviewers
    (``<mode>.annotated.txt``), the clean rendering (``<mode>.clean.txt``) and a
    counts-only ``report``. Renderings come from the transcript plus the layer; the
    transcript itself is never changed.
    """
    mode = deid_mode(ctx.modes)
    deid = ctx.config.deid
    transcript = ctx.read(Stage.ASSIGN_SPEAKERS, Transcript, variant=mode.value)
    layer, report = deidentify(
        transcript,
        _need(ctx.backends.ner),
        _need(ctx.backends.llm),
        deid.entity_types,
        ctx.file_id,
        deid.llm_server.max_model_len,
        deid.redaction_threshold,
        max_concurrency=deid.llm_server.max_concurrency,
    )
    ctx.store.write(ctx.file_id, Stage.DETECT_PHI, layer, variant=mode.value)
    ctx.store.write_text(
        ctx.file_id,
        Stage.DETECT_PHI,
        f"{mode.value}.annotated",
        render_annotated(transcript, layer),
    )
    ctx.store.write_text(
        ctx.file_id,
        Stage.DETECT_PHI,
        f"{mode.value}.clean",
        render_clean(transcript, layer, deid.entity_types, deid.redaction_threshold),
    )
    ctx.store.write(ctx.file_id, Stage.DETECT_PHI, report, variant=DEID_REPORT)


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
