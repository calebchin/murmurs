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


class ValidationResult(_Model):
    """Stage 1 output: what ingest found out about one input file."""

    schema_version: Literal[1] = 1
    file_id: str
    path: str
    format: str | None = None
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


# --- Diarization and transcription ------------------------------------------------


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
    """Per-file progress through the pipeline."""

    schema_version: Literal[1] = 1
    file_id: str
    input_hash: str
    stages: dict[Stage, StageStatus] = Field(
        default_factory=lambda: dict.fromkeys(Stage, StageStatus.PENDING)
    )
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
    Diarization,
    Transcript,
    RedactionLayer,
    FileStatus,
    RunManifest,
)
"""Models written to disk as standalone JSON files."""


def artifact_json_schemas() -> dict[str, dict[str, Any]]:
    """JSON Schema for every top-level artifact, keyed by model name."""
    return {model.__name__: model.model_json_schema() for model in TOP_LEVEL_ARTIFACTS}
