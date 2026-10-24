"""Basic scoring of PHI detection on the synthetic evaluation set (P3-04, MVP).

A label counts as **found** when a span of the same entity type overlaps it, and as
**covered** when any span overlaps it (what matters for redaction). Precision counts
spans of a type that overlap a label of that type. Types with no labels in a set (for
example the flag-only attribute types) are reported as spans only, without precision.

Scores hold counts and label IDs only, never text.

A run saves each interview's detections as it goes (``SavedDetections``), so a long run
can resume, and scores are always computed from the saved detections against the
current labels: re-importing a set with a changed tag mapping rescores it without
detecting again.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from murmurs.models import GroundTruthSet, RedactionLayer, RedactionSpan

DETECTIONS_SUFFIX = ".detections.json"
"""Suffix of an interview's saved detections in a run folder."""

NAME_TYPES = ("PERSON",)
"""Entity types whose recall is the phase goal's "names recall" (100% target)."""


class TypeScore(BaseModel):
    """Counts for one entity type in one set (or overall)."""

    labels: int = 0
    found: int = 0
    covered: int = 0
    spans: int = 0
    true_spans: int = 0

    @property
    def recall(self) -> float | None:
        """Found / labels, or None without labels."""
        return self.found / self.labels if self.labels else None

    @property
    def precision(self) -> float | None:
        """Spans overlapping a same-type label / spans, or None without labels or spans."""
        return self.true_spans / self.spans if self.labels and self.spans else None

    def add(self, other: TypeScore) -> None:
        """Accumulate another score's counts."""
        self.labels += other.labels
        self.found += other.found
        self.covered += other.covered
        self.spans += other.spans
        self.true_spans += other.true_spans


class SetScore(BaseModel):
    """Scores for one evaluation interview."""

    set_id: str
    interview_type: str
    by_type: dict[str, TypeScore]
    missed: list[str]
    """Label IDs no same-type span overlapped."""
    uncovered: list[str]
    """Label IDs no span overlapped at all."""
    seconds: float = 0.0


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def score_set(
    evalset: GroundTruthSet, spans: Sequence[RedactionSpan], seconds: float = 0.0
) -> SetScore:
    """Score one interview's detected spans against its labels."""
    by_type: dict[str, TypeScore] = {}
    missed, uncovered = [], []
    for label in evalset.labels:
        score = by_type.setdefault(label.entity_type, TypeScore())
        score.labels += 1
        overlapping = [s for s in spans if _overlaps(s.start, s.end, label.start, label.end)]
        if any(s.entity_type == label.entity_type for s in overlapping):
            score.found += 1
        else:
            missed.append(label.id)
        if overlapping:
            score.covered += 1
        else:
            uncovered.append(label.id)
    for span in spans:
        score = by_type.setdefault(span.entity_type, TypeScore())
        score.spans += 1
        if any(
            label.entity_type == span.entity_type
            and _overlaps(span.start, span.end, label.start, label.end)
            for label in evalset.labels
        ):
            score.true_spans += 1
    return SetScore(
        set_id=evalset.set_id,
        interview_type=evalset.interview_type.value,
        by_type=dict(sorted(by_type.items())),
        missed=missed,
        uncovered=uncovered,
        seconds=round(seconds, 2),
    )


def overall(scores: Sequence[SetScore]) -> dict[str, TypeScore]:
    """Counts summed over sets, per entity type."""
    total: dict[str, TypeScore] = {}
    for score in scores:
        for name, type_score in score.by_type.items():
            total.setdefault(name, TypeScore()).add(type_score)
    return dict(sorted(total.items()))


def names_recall(total: dict[str, TypeScore]) -> tuple[int, int]:
    """(found, labels) over the name types."""
    found = sum(total[t].found for t in NAME_TYPES if t in total)
    labels = sum(total[t].labels for t in NAME_TYPES if t in total)
    return found, labels


def format_type_line(name: str, score: TypeScore) -> str:
    """One console line of counts (no text)."""
    parts = [f"{name:<20}"]
    if score.labels:
        parts.append(f"found {score.found}/{score.labels}")
        parts.append(f"covered {score.covered}/{score.labels}")
        parts.append(
            f"precision {score.true_spans}/{score.spans}" if score.spans else "precision -"
        )
    else:
        parts.append(f"spans {score.spans} (no labels of this type)")
    return "  ".join(parts)


def label_breakdown(scores: Sequence[SetScore]) -> Counter[str]:
    """Missed label counts per set, for a one-line summary."""
    return Counter({s.set_id: len(s.missed) for s in scores})


# --- saved detections (resume and rescoring) ---------------------------------------------


class SavedDetections(BaseModel):
    """One interview's detected spans in a run folder, with what they were computed from.

    Spans hold offsets, types and tags only, never text.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    transcript_sha256: str
    """Hash of the transcript detection ran on; a different text means detect again."""
    settings_sha256: str
    """Hash of the detection settings (``settings_digest``)."""
    seconds: float
    layer: RedactionLayer


def transcript_digest(evalset: GroundTruthSet) -> str:
    """Hash of an interview's transcript (not its labels, which only affect scoring)."""
    return hashlib.sha256(evalset.transcript.model_dump_json().encode()).hexdigest()


def settings_digest(settings: dict[str, Any]) -> str:
    """Hash of the settings that affect detection output (the ``detect_phi`` stage config)."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def detections_path(run_dir: Path, set_id: str) -> Path:
    """Where an interview's saved detections live in a run folder."""
    return run_dir / f"{set_id}{DETECTIONS_SUFFIX}"


def load_detections(path: Path, transcript_sha: str, settings_sha: str) -> SavedDetections | None:
    """Saved detections that match this transcript and these settings, or None (missing,
    unreadable or stale: the interview is detected again)."""
    try:
        saved = SavedDetections.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None
    if saved.transcript_sha256 != transcript_sha or saved.settings_sha256 != settings_sha:
        return None
    return saved
