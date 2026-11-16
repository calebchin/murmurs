"""The synthetic evaluation set (P2-09, MVP scope D8): parse, validate and count.

An interview is a Markdown file::

    ---
    interview_type: CECA
    ---
    INTERVIEWER: Where did you grow up?
    PARTICIPANT: In {{LOCATION|Millbrook}}, with my aunt {{PERSON|Rosa}}.

One line per turn; ``INTERVIEWER`` becomes ``SPEAKER_00`` and ``PARTICIPANT``
``SPEAKER_01``. Turns may also start with a speaker label such as ``SPEAKER_02``, kept as
is (``murmurs eval import`` writes these). PHI is marked inline as ``{{TYPE|text}}`` or
``{{TYPE|text|note}}``. The parser builds a transcript (one segment per turn, synthetic
word times) and labels whose offsets point into its canonical text (D9), so offsets are
never counted by hand.

Error messages name the file, line number or label ID, never transcript text.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from murmurs.models import (
    GroundTruthLabel,
    GroundTruthSet,
    InterviewType,
    Segment,
    Transcript,
    TranscriptMode,
    Word,
)
from murmurs.store import canonical_text

SPEAKERS = {"INTERVIEWER": "SPEAKER_00", "PARTICIPANT": "SPEAKER_01"}
WORD_S = 0.3
"""Synthetic duration of each word."""
TURN_GAP_S = 1.0
"""Synthetic pause between turns."""

_MARK = re.compile(r"\{\{([A-Z][A-Z_]*)\|([^{}|]+)(?:\|([^{}]*))?\}\}")
_TURN = re.compile(r"^(INTERVIEWER|PARTICIPANT|SPEAKER_\d{2,}):\s*(.*)$")


class EvalSetError(ValueError):
    """An evaluation file is malformed or its labels are invalid. Messages name files,
    line numbers and label IDs only."""


@dataclass(frozen=True)
class _Turn:
    speaker: str
    text: str
    labels: list[tuple[int, int, str, str]]  # start, end, type, note (offsets in text)


def _parse_turn(body: str, where: str) -> tuple[str, list[tuple[int, int, str, str]]]:
    """Replace inline marks with their text; return the plain text and label offsets."""
    body = " ".join(body.split())
    out: list[str] = []
    labels: list[tuple[int, int, str, str]] = []
    pos = length = 0
    for match in _MARK.finditer(body):
        out.append(body[pos : match.start()])
        length += match.start() - pos
        surface = " ".join(match.group(2).split())
        labels.append((length, length + len(surface), match.group(1), match.group(3) or ""))
        out.append(surface)
        length += len(surface)
        pos = match.end()
    out.append(body[pos:])
    text = "".join(out)
    if "{{" in text or "}}" in text:
        raise EvalSetError(f"{where}: malformed {{{{TYPE|text}}}} mark")
    return text, labels


def parse_interview(source: str, set_id: str) -> GroundTruthSet:
    """Parse one interview file's text into a ``GroundTruthSet`` (not yet validated
    against a tag set; see ``validate``)."""
    lines = source.splitlines()
    meta: dict[str, str] = {}
    start = 0
    if lines and lines[0].strip() == "---":
        try:
            end = lines.index("---", 1)
        except ValueError:
            raise EvalSetError(f"{set_id}: front matter is not closed") from None
        for raw in lines[1:end]:
            key, _, value = raw.partition(":")
            meta[key.strip()] = value.strip()
        start = end + 1
    try:
        interview_type = InterviewType(meta.get("interview_type", ""))
    except ValueError:
        raise EvalSetError(f"{set_id}: interview_type must be CECA or LEDS") from None

    turns: list[_Turn] = []
    for number, raw in enumerate(lines[start:], start=start + 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = _TURN.match(raw.strip())
        if match is None:
            raise EvalSetError(
                f"{set_id} line {number}: must start with INTERVIEWER:, PARTICIPANT: or SPEAKER_NN:"
            )
        text, turn_labels = _parse_turn(match.group(2), f"{set_id} line {number}")
        if not text:
            raise EvalSetError(f"{set_id} line {number}: empty turn")
        speaker = SPEAKERS.get(match.group(1), match.group(1))
        turns.append(_Turn(speaker, text, turn_labels))

    segments: list[Segment] = []
    labels: list[GroundTruthLabel] = []
    t = 0.0
    offset = 0
    for turn in turns:
        words = []
        seg_start = t
        for token in turn.text.split(" "):
            words.append(
                Word(
                    text=token,
                    start=round(t, 3),
                    end=round(t + WORD_S * 0.9, 3),
                    speaker=turn.speaker,
                )
            )
            t += WORD_S
        segments.append(
            Segment(
                speaker=turn.speaker,
                start=round(seg_start, 3),
                end=words[-1].end,
                text=turn.text,
                words=words,
            )
        )
        for begin, finish, entity_type, note in turn.labels:
            labels.append(
                GroundTruthLabel(
                    id=f"{set_id}-{len(labels) + 1:03d}",
                    start=offset + begin,
                    end=offset + finish,
                    entity_type=entity_type,
                    text=turn.text[begin:finish],
                    note=note,
                )
            )
        offset += len(turn.text) + 1  # the newline between segments
        t += TURN_GAP_S
    transcript = Transcript(file_id=set_id, mode=TranscriptMode.INTENDED, segments=segments)
    return GroundTruthSet(
        set_id=set_id, interview_type=interview_type, transcript=transcript, labels=labels
    )


def validate(evalset: GroundTruthSet, entity_types: Iterable[str]) -> None:
    """Check offsets, surface text, entity types and overlaps. Raises ``EvalSetError``
    listing every problem by label ID."""
    text = canonical_text(evalset.transcript)
    known = set(entity_types)
    problems = []
    for label in evalset.labels:
        if not 0 <= label.start < label.end <= len(text):
            problems.append(f"{label.id}: offsets fall outside the text")
        elif text[label.start : label.end] != label.text:
            problems.append(f"{label.id}: text does not match the text at its offsets")
        if label.entity_type not in known:
            problems.append(f"{label.id}: entity type is not in the tag set")
    ordered = sorted(evalset.labels, key=lambda lb: (lb.start, lb.end))
    for before, after in itertools.pairwise(ordered):
        if after.start < before.end:
            problems.append(f"{after.id}: overlaps {before.id}")
    if problems:
        raise EvalSetError(f"{evalset.set_id}: " + "; ".join(problems))


def load_interview(path: Path, entity_types: Iterable[str]) -> GroundTruthSet:
    """Parse and validate one interview file."""
    evalset = parse_interview(path.read_text(encoding="utf-8"), path.stem)
    validate(evalset, entity_types)
    return evalset


def load_dir(directory: Path, entity_types: Iterable[str]) -> list[GroundTruthSet]:
    """Parse and validate every ``*.md`` interview in ``directory``, sorted by name."""
    types = list(entity_types)
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise EvalSetError(f"no *.md interviews in {directory}")
    return [load_interview(path, types) for path in paths]


def label_counts(evalset: GroundTruthSet) -> Counter[str]:
    """Labels per entity type."""
    return Counter(label.entity_type for label in evalset.labels)
