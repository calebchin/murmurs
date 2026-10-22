"""Data models for JSON artifacts (transcripts, redactions, manifests).

Every model hides input values in validation errors, and validator messages name
indices and fields only, so an error can never quote transcript text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from murmurs.config import JobConfig, Stage

TIME_TOLERANCE_S = 1e-6
"""Slack allowed when checking that a word lies inside its segment (float rounding)."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class _Frozen(BaseModel):
    """Base for transcript models; the original transcript is immutable."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)


# --- Shared -------------------------------------------------------------------


class ErrorInfo(_Model):
    """A PHI-free error: a fixed code plus a short message written by our code.

    Never put exception text here; library messages can quote file or transcript content.
    """

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$", max_length=64)
    message: str = Field("", max_length=200)


def _check_time_range(start: float, end: float) -> None:
    if start < 0:
        raise ValueError("start must be >= 0")
    if end < start:
        raise ValueError("end must be >= start")


# --- Ingest ---------------------------------------------------------------------


class IngestErrorCode(StrEnum):
    """Why ingest rejected a file. The only codes a ``ValidationResult`` error uses."""

    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    """The extension is not in ``data.audio_extensions``."""
    EMPTY = "EMPTY"
    """Zero bytes, or a container with zero duration."""
    UNREADABLE = "UNREADABLE"
    """Not a container the decoder can open (e.g. a text file), or no duration in it."""
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    """A valid container with no audio stream."""
    CORRUPT = "CORRUPT"
    """The start of the audio stream cannot be decoded."""
    TRUNCATED = "TRUNCATED"
    """The file ends before its header says it should, or its end cannot be decoded."""


class ValidationResult(_Model):
    """Stage 1 output: what ingest found out about one input file.

    (Schema 2 added ``codec``.)
    """

    schema_version: Literal[2] = 2
    file_id: str
    path: str
    format: str | None = None
    """Container format, from the file extension (e.g. ``"wav"``, ``"m4a"``)."""
    codec: str | None = None
    """Audio codec name as the decoder reports it (e.g. ``"pcm_s16le"``, ``"aac"``)."""
    duration_s: float | None = Field(None, ge=0)
    sample_rate: int | None = Field(None, gt=0)
    channels: int | None = Field(None, gt=0)
    valid: bool
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _error_matches_valid(self) -> ValidationResult:
        if self.valid and self.error is not None:
            raise ValueError("a valid file must not have an error")
        if not self.valid and self.error is None:
            raise ValueError("an invalid file must have an error")
        return self


# --- Audio loading -----------------------------------------------------------------


class AudioInfo(_Model):
    """Stage 2 output: facts about the decoded waveform. The audio itself is never saved.

    (Schema 2 added ``input_hash``.)
    """

    schema_version: Literal[2] = 2
    file_id: str
    input_hash: str
    """SHA-256 of the input file the waveform was decoded from."""
    sample_rate: int = Field(gt=0)
    n_samples: int = Field(ge=0)
    duration_s: float = Field(ge=0)


# --- Diarization and transcription ------------------------------------------------


UNKNOWN_SPEAKER = "UNKNOWN"
"""Speaker label for speech that cannot be attributed to one of the file's speakers."""


class SpeakerTurn(_Frozen):
    """One stretch of time attributed to one speaker."""

    speaker: str
    start: float
    end: float

    @model_validator(mode="after")
    def _valid_range(self) -> SpeakerTurn:
        _check_time_range(self.start, self.end)
        return self


class Diarization(_Frozen):
    """Stage 3 output: all speaker turns for one file, in time order."""

    schema_version: Literal[1] = 1
    file_id: str
    turns: list[SpeakerTurn]


class Word(_Frozen):
    """One transcribed word with timestamps. ``speaker`` is set by speaker assignment."""

    text: str
    start: float
    end: float
    speaker: str | None = None

    @model_validator(mode="after")
    def _valid_range(self) -> Word:
        _check_time_range(self.start, self.end)
        return self


class Segment(_Frozen):
    """A run of words. ``speaker`` is None until speaker assignment."""

    speaker: str | None = None
    start: float
    end: float
    text: str
    words: list[Word] = []

    @model_validator(mode="after")
    def _words_inside(self) -> Segment:
        _check_time_range(self.start, self.end)
        for i, word in enumerate(self.words):
            if word.start < self.start - TIME_TOLERANCE_S or word.end > self.end + TIME_TOLERANCE_S:
                raise ValueError(f"word {i} lies outside the segment's time range")
        return self


class TranscriptMode(StrEnum):
    """Which CrisperWhisper output a transcript holds."""

    VERBATIM = "verbatim"
    INTENDED = "intended"


class Transcript(_Frozen):
    """A transcript of one file in one mode. Never modified after it is written."""

    schema_version: Literal[1] = 1
    file_id: str
    mode: TranscriptMode
    segments: list[Segment]


class SpeakerCorrection(_Model):
    """Stage 5 report on the LLM speaker-correction pass (P2-07): counts only, no text.

    Written whether or not correction ran (``enabled``), so toggling it never leaves a
    stale report.
    """

    schema_version: Literal[1] = 1
    file_id: str
    enabled: bool
    source_mode: TranscriptMode | None = None
    """The mode the LLM corrected; the other mode was mapped by word time (D24)."""
    n_chunks: int = Field(0, ge=0)
    failed_chunks: int = Field(0, ge=0)
    """Chunks whose LLM output stayed invalid after retries; they keep rule-based labels."""
    proposed: int = Field(0, ge=0)
    """Label changes the LLM proposed."""
    applied: int = Field(0, ge=0)
    rejected: dict[str, int] = {}
    """Rejected changes by reason (e.g. ``"text_mismatch"``)."""
    words_changed: dict[TranscriptMode, int] = {}
    """Words whose speaker differs from the rule-based assignment, per mode."""


class DeidReport(_Model):
    """Stage 6 report on PHI detection (P3-03): counts only, never text."""

    schema_version: Literal[1] = 1
    file_id: str
    mode: TranscriptMode
    n_chunks: int = Field(0, ge=0)
    failed_chunks: int = Field(0, ge=0)
    """Chunks whose LLM output stayed invalid after retries."""
    llm_proposed: int = Field(0, ge=0)
    llm_rejected: dict[str, int] = {}
    """Rejected LLM entities by reason (e.g. ``"not_found"``)."""
    llm_spans: int = Field(0, ge=0)
    """Spans from LLM entities, before merging, including propagated occurrences."""
    propagated: int = Field(0, ge=0)
    """Of ``llm_spans``, occurrences found elsewhere in the text for an accepted entity."""
    ner_spans: int = Field(0, ge=0)
    ner_ignored: int = Field(0, ge=0)
    """NER spans whose label maps to no entity type or to an ignored one."""
    spans: int = Field(0, ge=0)
    """Spans after merging."""
    both_detectors: int = Field(0, ge=0)
    redacted: int = Field(0, ge=0)
    """Spans replaced by a tag in the clean text."""
    flagged: int = Field(0, ge=0)
    by_type: dict[str, int] = {}


# --- Evaluation set ---------------------------------------------------------------------


class InterviewType(StrEnum):
    """The interview styles the evaluation set covers."""

    CECA = "CECA"
    LEDS = "LEDS"


class GroundTruthLabel(_Model):
    """One planted PHI item in an evaluation transcript (P2-09).

    ``start``/``end`` are character offsets into the transcript's canonical text (D9),
    end exclusive; ``text`` must equal that slice.
    """

    id: str
    start: int = Field(ge=0)
    end: int
    entity_type: str
    text: str
    note: str = ""


class GroundTruthSet(_Model):
    """One synthetic evaluation interview with its ground-truth labels."""

    schema_version: Literal[1] = 1
    set_id: str
    interview_type: InterviewType
    transcript: Transcript
    labels: list[GroundTruthLabel]


# --- Redactions and review ------------------------------------------------------------


class DetectorSource(StrEnum):
    """What proposed a redaction."""

    NER = "ner"
    LLM = "llm"
    MANUAL = "manual"


class SpanStatus(StrEnum):
    """Review status of one redaction span."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL = "manual"


class ReviewState(StrEnum):
    """Review state of a whole transcript."""

    UNREVIEWED = "unreviewed"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    SIGNED_OFF = "signed_off"


class AuditAction(StrEnum):
    """Kinds of change recorded in the audit trail."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    ADDED = "added"
    EDITED = "edited"


class RedactionSpan(_Model):
    """One proposed or manual redaction over a transcript's text.

    ``start`` and ``end`` are character offsets, end exclusive.
    """

    id: str
    start: int = Field(ge=0)
    end: int
    entity_type: str
    tag: str
    confidence: float = Field(ge=0, le=1)
    sources: list[DetectorSource] = Field(min_length=1)
    status: SpanStatus = SpanStatus.PROPOSED
    reviewer: str | None = None
    timestamp: AwareDatetime | None = None
    flagged_edge_case: bool = False

    @model_validator(mode="after")
    def _valid_span(self) -> RedactionSpan:
        if self.start >= self.end:
            raise ValueError("start must be < end")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("sources must not repeat")
        return self


class AuditEvent(_Model):
    """One recorded change to a redaction span: who, when, and what changed."""

    span_id: str
    action: AuditAction
    actor: str
    timestamp: AwareDatetime
    before: SpanStatus | None = None
    after: SpanStatus | None = None


class TranscriptRef(_Model):
    """Identifies the immutable transcript a redaction layer applies to."""

    file_id: str
    mode: TranscriptMode


class RedactionLayer(_Model):
    """Redaction decisions kept separately from the original transcript."""

    schema_version: Literal[1] = 1
    transcript: TranscriptRef
    spans: list[RedactionSpan] = []
    review_state: ReviewState = ReviewState.UNREVIEWED
    audit: list[AuditEvent] = []

    @model_validator(mode="after")
    def _consistent_ids(self) -> RedactionLayer:
        ids = [span.id for span in self.spans]
        if len(set(ids)) != len(ids):
            raise ValueError("span ids must be unique")
        known = set(ids)
        for i, event in enumerate(self.audit):
            if event.span_id not in known:
                raise ValueError(f"audit event {i} refers to an unknown span id")
        return self


# --- Run tracking -------------------------------------------------------------------


class StageStatus(StrEnum):
    """Status of one stage for one file."""

    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileStatus(_Model):
    """Per-file progress through the pipeline.

    ``stage_hashes`` records the config hash each done stage ran with, so resume can
    tell when a stage's config has changed. (Schema 2 added ``stage_hashes``.)
    """

    schema_version: Literal[2] = 2
    file_id: str
    input_hash: str
    stages: dict[Stage, StageStatus] = Field(
        default_factory=lambda: dict.fromkeys(Stage, StageStatus.PENDING)
    )
    stage_hashes: dict[Stage, str] = {}
    failed_stage: Stage | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _failure_consistent(self) -> FileStatus:
        failed = [stage for stage, status in self.stages.items() if status is StageStatus.FAILED]
        if len(failed) > 1:
            raise ValueError("at most one stage can be failed")
        if failed != ([] if self.failed_stage is None else [self.failed_stage]):
            raise ValueError("failed_stage must name the failed stage")
        if (self.error is None) != (self.failed_stage is None):
            raise ValueError("error must be set exactly when a stage failed")
        return self


class RunManifest(_Model):
    """A reproducible record of one job."""

    schema_version: Literal[1] = 1
    job_id: str
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    config: JobConfig
    package_version: str
    dependency_versions: dict[str, str] = {}
    model_versions: dict[str, str] = {}
    code_version: str | None = None
    input_hashes: dict[str, str] = {}
    """Content hash per file ID."""
    file_statuses: list[FileStatus] = []


TOP_LEVEL_ARTIFACTS: tuple[type[BaseModel], ...] = (
    ValidationResult,
    AudioInfo,
    Diarization,
    Transcript,
    SpeakerCorrection,
    DeidReport,
    GroundTruthSet,
    RedactionLayer,
    FileStatus,
    RunManifest,
)
"""Models written to disk as standalone JSON files."""


def artifact_json_schemas() -> dict[str, dict[str, Any]]:
    """JSON Schema for every top-level artifact, keyed by model name."""
    return {model.__name__: model.model_json_schema() for model in TOP_LEVEL_ARTIFACTS}
