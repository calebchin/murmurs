"""PHI detection (P3-02, P3-03): NER and LLM spans, merged, tagged and rendered.

Works on a transcript's canonical text (D9): segment texts joined with newlines, one
line per speaker segment. The LLM sees numbered lines in chunks and reports entities as
(line, exact text, type); each is checked against the text and becomes spans at every
occurrence of that text. NER spans are mapped to entity types through the tag set
(D10). Overlapping spans merge; confidence is detector agreement (MVP). Tags are
indexed per entity type by surface, so the same name always gets the same tag.

Renderings are built from the original transcript plus the redaction layer and never
change the transcript (hard rule 3). Nothing here logs: prompts, responses and spans
hold transcript text (hard rule 1).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from pydantic import BaseModel, ConfigDict

from murmurs.backends import LLMClient, NERDetector
from murmurs.config import EntityAction, EntityType
from murmurs.models import (
    DeidReport,
    DetectorSource,
    RedactionLayer,
    RedactionSpan,
    Transcript,
    TranscriptMode,
    TranscriptRef,
)
from murmurs.stages.correction import CHARS_PER_TOKEN, CHUNK_SHARE, CONTEXT_SHARE
from murmurs.stages.llm_chunks import halves, run_chunks
from murmurs.store import canonical_text

PROMPT_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = PROMPT_DIR / "deid_system.txt"
USER_PROMPT = PROMPT_DIR / "deid_user.txt"

DEID_CHUNK_TOKENS = 1000.0
"""Transcript tokens a chunk owns. The response lists every entity in the chunk, so its
length grows with the chunk; about 1,000 tokens keeps it well under the output cap and
gives the server many requests to run concurrently (P3-05)."""
DEID_LINE_TOKENS = 250.0
"""Longest line the LLM sees; longer speaker segments are split between words, so a
chunk can always be split into smaller ones (see ``llm_lines``)."""
SPLIT_CONTEXT_LINES = 2
"""Lines of the other half shown as context when a chunk is split."""

LLM_ONLY_CONFIDENCE = 0.7
"""Confidence of a span only the LLM found (MVP; refined after the MVP)."""
BOTH_CONFIDENCE = 1.0
"""Confidence of a span both detectors found."""

REJECT_MISSING_LINE = "missing_line"
REJECT_OUTSIDE_CHUNK = "outside_chunk"
REJECT_UNKNOWN_TYPE = "unknown_type"
REJECT_NOT_FOUND = "not_found"


class FoundEntity(BaseModel):
    """One entity the LLM reports: its line, exact text and entity type."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    line: int
    text: str
    entity_type: str


class FoundEntities(BaseModel):
    """The LLM's structured response for one chunk."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    entities: list[FoundEntity]


def prompt_hash() -> str:
    """Hash of the deid prompt files; part of the stage config hash."""
    digest = hashlib.sha256()
    for path in (SYSTEM_PROMPT, USER_PROMPT):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def deid_mode(modes: Sequence[TranscriptMode]) -> TranscriptMode:
    """The mode PHI detection runs on: verbatim if configured, else the first (D9)."""
    return TranscriptMode.VERBATIM if TranscriptMode.VERBATIM in modes else modes[0]


# --- lines and chunks -------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    """One line of the canonical text (a speaker segment), or, for the LLM, a piece of
    one (see ``llm_lines``)."""

    number: int
    start: int
    """Offset of the line's first character in the canonical text."""
    text: str
    speaker: str


def lines_of(transcript: Transcript) -> list[Line]:
    """The canonical text's lines with their offsets (D9)."""
    lines, offset = [], 0
    for i, segment in enumerate(transcript.segments):
        lines.append(Line(i, offset, segment.text, segment.speaker or "UNKNOWN"))
        offset += len(segment.text) + 1
    return lines


def _pieces(line: Line) -> list[tuple[int, str]]:
    """``(offset, text)`` pieces of a line, split between words so each costs at most
    ``DEID_LINE_TOKENS`` (a single longer word stays whole)."""
    if _cost(line) <= DEID_LINE_TOKENS:
        return [(0, line.text)]
    max_chars = DEID_LINE_TOKENS * CHARS_PER_TOKEN - len(line.speaker) - 10
    pieces: list[tuple[int, str]] = []
    first = last = None
    for word in re.finditer(r"\S+", line.text):
        if first is not None and last is not None and word.end() - first > max_chars:
            pieces.append((first, line.text[first:last]))
            first = None
        if first is None:
            first = word.start()
        last = word.end()
    if first is not None and last is not None:
        pieces.append((first, line.text[first:last]))
    return pieces


def llm_lines(transcript: Transcript) -> list[Line]:
    """The lines the LLM sees: ``lines_of``, with any line longer than
    ``DEID_LINE_TOKENS`` split between words into consecutive pieces, numbered in order.

    Short lines let a chunk that keeps failing be split until every part is answered,
    so no long monologue goes unchecked. Renderings still use ``lines_of``."""
    lines: list[Line] = []
    for line in lines_of(transcript):
        for offset, text in _pieces(line):
            lines.append(Line(len(lines), line.start + offset, text, line.speaker))
    return lines


@dataclass(frozen=True)
class LineChunk:
    """Lines the LLM may report from, with context lines before and after."""

    before: list[Line]
    own: list[Line]
    after: list[Line]

    @property
    def own_numbers(self) -> set[int]:
        """Line numbers the LLM may report entities from."""
        return {line.number for line in self.own}


def _cost(line: Line) -> float:
    return (len(line.text) + len(line.speaker) + 10) / CHARS_PER_TOKEN


def chunk_lines(lines: Sequence[Line], max_model_len: int) -> list[LineChunk]:
    """Group lines into chunks of about ``DEID_CHUNK_TOKENS`` (less if the context window
    is small), each with context lines on both sides. Every line belongs to exactly one
    chunk's own lines. Small chunks keep each response short and give the server
    more requests to run concurrently."""
    budget = max(200.0, min(DEID_CHUNK_TOKENS, max_model_len / CHUNK_SHARE))
    context_budget = budget / CONTEXT_SHARE
    groups: list[tuple[int, int]] = []
    first, size = 0, 0.0
    for i, line in enumerate(lines):
        if i > first and size + _cost(line) > budget:
            groups.append((first, i))
            first, size = i, 0.0
        size += _cost(line)
    if lines:
        groups.append((first, len(lines)))
    chunks = []
    for first, last in groups:
        before: list[Line] = []
        size = 0.0
        for i in range(first - 1, -1, -1):
            if before and size + _cost(lines[i]) > context_budget:
                break
            before.insert(0, lines[i])
            size += _cost(lines[i])
        after: list[Line] = []
        size = 0.0
        for i in range(last, len(lines)):
            if after and size + _cost(lines[i]) > context_budget:
                break
            after.append(lines[i])
            size += _cost(lines[i])
        chunks.append(LineChunk(before, list(lines[first:last]), after))
    return chunks


def split_chunk(chunk: LineChunk) -> tuple[LineChunk, LineChunk] | None:
    """Two chunks, each owning half of ``chunk``'s lines, or None for a one-line chunk.
    Each half shows a couple of the other half's lines as context."""
    parts = halves(chunk.own)
    if parts is None:
        return None
    first, second = parts
    return (
        LineChunk(chunk.before, first, second[:SPLIT_CONTEXT_LINES]),
        LineChunk(first[-SPLIT_CONTEXT_LINES:], second, chunk.after),
    )


def render_chunk(chunk: LineChunk) -> str:
    """The chunk as the LLM sees it: ``[n] SPEAKER: text``, context lines marked."""

    def fmt(line: Line, prefix: str) -> str:
        return f"{prefix}[{line.number}] {line.speaker}: {line.text}"

    return "\n".join(
        [fmt(ln, "CONTEXT ") for ln in chunk.before]
        + [fmt(ln, "") for ln in chunk.own]
        + [fmt(ln, "CONTEXT ") for ln in chunk.after]
    )


def build_prompt(chunk: LineChunk, entity_types: Sequence[EntityType]) -> tuple[str, str]:
    """``(system, user)`` prompts for one chunk."""
    types = "\n".join(f"- {e.name}: {e.description}" for e in entity_types)
    system = SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    user = Template(USER_PROMPT.read_text(encoding="utf-8")).substitute(
        entity_types=types, transcript=render_chunk(chunk)
    )
    return system, user.strip()


# --- spans --------------------------------------------------------------------------------


@dataclass
class Candidate:
    """A span from one detector, before merging."""

    start: int
    end: int
    entity_type: str
    source: DetectorSource
    confidence: float


@dataclass
class _Counts:
    n_chunks: int = 0
    splits: int = 0
    failed_chunks: int = 0
    proposed: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    spans: int = 0
    propagated: int = 0


def _occurrences(text: str, surface: str) -> list[int]:
    """Start offsets of ``surface`` in ``text`` as a whole word or phrase."""
    pattern = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)")
    return [m.start() for m in pattern.finditer(text)]


def llm_candidates(
    transcript: Transcript,
    llm: LLMClient,
    file_id: str,
    entity_types: Sequence[EntityType],
    max_model_len: int,
    *,
    max_concurrency: int = 1,
) -> tuple[list[Candidate], _Counts]:
    """Ask the LLM for entities chunk by chunk and turn the valid ones into spans.

    Chunks go to the LLM up to ``max_concurrency`` at a time; a chunk whose response is
    truncated or invalid is split in half (see ``llm_chunks``). An entity is valid if
    its line is one of the chunk's own lines, its type is a configured (non-ignored)
    type, and its text occurs in that line. Each accepted (text, type) is then marked at
    every whole-word occurrence in the whole text, in chunk order. A chunk that stays
    unusable is skipped and counted; a failed request fails the file.
    """
    lines = llm_lines(transcript)
    text = canonical_text(transcript)
    usable = [e for e in entity_types if e.action is not EntityAction.IGNORE]
    names = {e.name for e in usable}
    counts = _Counts()
    accepted: dict[tuple[str, str], int] = {}  # (surface, type) -> line where first found
    by_line = {line.number: line for line in lines}

    def ask(chunk: LineChunk) -> FoundEntities:
        system, user = build_prompt(chunk, usable)
        return llm.generate(user, FoundEntities, file_id, system=system)

    chunks = chunk_lines(lines, max_model_len)
    outcome = run_chunks(chunks, ask, split_chunk, max_concurrency)
    counts.n_chunks = len(chunks)
    counts.splits = outcome.splits
    counts.failed_chunks = outcome.failed
    for chunk, response in outcome.results:
        for entity in response.entities:
            counts.proposed += 1
            surface = " ".join(entity.text.split())
            if entity.line not in by_line:
                counts.rejected[REJECT_MISSING_LINE] += 1
            elif entity.line not in chunk.own_numbers:
                counts.rejected[REJECT_OUTSIDE_CHUNK] += 1
            elif entity.entity_type not in names:
                counts.rejected[REJECT_UNKNOWN_TYPE] += 1
            elif not surface or not _occurrences(by_line[entity.line].text, surface):
                counts.rejected[REJECT_NOT_FOUND] += 1
            else:
                accepted.setdefault((surface, entity.entity_type), entity.line)

    candidates = []
    for (surface, entity_type), line_number in accepted.items():
        line = by_line[line_number]
        for start in _occurrences(text, surface):
            in_reported_line = line.start <= start < line.start + len(line.text)
            if not in_reported_line:
                counts.propagated += 1
            candidates.append(
                Candidate(
                    start,
                    start + len(surface),
                    entity_type,
                    DetectorSource.LLM,
                    LLM_ONLY_CONFIDENCE,
                )
            )
    counts.spans = len(candidates)
    return candidates, counts


def ner_candidates(
    spans: Sequence[RedactionSpan], entity_types: Sequence[EntityType]
) -> tuple[list[Candidate], int]:
    """Map raw NER labels to entity types (D10). Returns the kept spans and the number
    ignored (no mapping, or an ignored type)."""
    label_to_type = {label: e for e in entity_types for label in e.ner_labels}
    kept, ignored = [], 0
    for span in spans:
        entity = label_to_type.get(span.entity_type)
        if entity is None or entity.action is EntityAction.IGNORE:
            ignored += 1
            continue
        kept.append(
            Candidate(span.start, span.end, entity.name, DetectorSource.NER, span.confidence)
        )
    return kept, ignored


def merge(
    candidates: Sequence[Candidate], entity_types: Sequence[EntityType]
) -> list[tuple[int, int, str, list[DetectorSource], float]]:
    """Merge overlapping candidates into spans covering their union.

    The type comes from the LLM when it found one (it sees context), else from NER; ties
    prefer redact over flag. Confidence: both detectors 1.0, NER alone its probability,
    LLM alone 0.7.
    """
    priority = {e.name: 0 if e.action is EntityAction.REDACT else 1 for e in entity_types}
    ordered = sorted(candidates, key=lambda c: (c.start, -c.end))
    groups: list[list[Candidate]] = []
    end = -1
    for candidate in ordered:
        if groups and candidate.start < end:
            groups[-1].append(candidate)
            end = max(end, candidate.end)
        else:
            groups.append([candidate])
            end = candidate.end
    merged = []
    for group in groups:
        sources = sorted({c.source for c in group}, key=lambda s: s.value)
        llm_types = [c for c in group if c.source is DetectorSource.LLM]
        pool = llm_types or group
        chosen = min(pool, key=lambda c: (priority.get(c.entity_type, 2), -(c.end - c.start)))
        if len(sources) > 1:
            confidence = BOTH_CONFIDENCE
        elif sources[0] is DetectorSource.NER:
            confidence = max(c.confidence for c in group)
        else:
            confidence = LLM_ONLY_CONFIDENCE
        merged.append(
            (
                min(c.start for c in group),
                max(c.end for c in group),
                chosen.entity_type,
                sources,
                confidence,
            )
        )
    return merged


def _normalise(surface: str) -> str:
    return " ".join(surface.casefold().split())


def deidentify(
    transcript: Transcript,
    ner: NERDetector,
    llm: LLMClient,
    entity_types: Sequence[EntityType],
    file_id: str,
    max_model_len: int,
    redaction_threshold: float,
    *,
    max_concurrency: int = 1,
) -> tuple[RedactionLayer, DeidReport]:
    """Detect PHI in a transcript and return the redaction layer and counts."""
    text = canonical_text(transcript)
    by_name = {e.name: e for e in entity_types}
    usage_before = llm.usage.model_copy()
    ner_kept, ner_ignored = ner_candidates(ner.detect(text, file_id), entity_types)
    llm_found, counts = llm_candidates(
        transcript, llm, file_id, entity_types, max_model_len, max_concurrency=max_concurrency
    )
    merged = merge(ner_kept + llm_found, entity_types)

    indexes: dict[str, dict[str, int]] = {}
    spans: list[RedactionSpan] = []
    for i, (start, end, entity_type, sources, confidence) in enumerate(merged):
        per_type = indexes.setdefault(entity_type, {})
        n = per_type.setdefault(_normalise(text[start:end]), len(per_type) + 1)
        entity = by_name[entity_type]
        spans.append(
            RedactionSpan(
                id=f"span-{i + 1:04d}",
                start=start,
                end=end,
                entity_type=entity_type,
                tag=entity.tag.format(n=n),
                confidence=round(confidence, 4),
                sources=sources,
                flagged_edge_case=entity.action is EntityAction.FLAG,
            )
        )
    layer = RedactionLayer(
        transcript=TranscriptRef(file_id=file_id, mode=transcript.mode), spans=spans
    )
    report = DeidReport(
        file_id=file_id,
        mode=transcript.mode,
        n_chunks=counts.n_chunks,
        chunk_splits=counts.splits,
        failed_chunks=counts.failed_chunks,
        llm=llm.usage.minus(usage_before),
        llm_proposed=counts.proposed,
        llm_rejected=dict(counts.rejected),
        llm_spans=counts.spans,
        propagated=counts.propagated,
        ner_spans=len(ner_kept) + ner_ignored,
        ner_ignored=ner_ignored,
        spans=len(spans),
        both_detectors=sum(len(s.sources) > 1 for s in spans),
        redacted=sum(redacts(s, by_name, redaction_threshold) for s in spans),
        flagged=sum(s.flagged_edge_case for s in spans),
        by_type=dict(Counter(s.entity_type for s in spans)),
    )
    return layer, report


def redacts(span: RedactionSpan, by_name: dict[str, EntityType], threshold: float) -> bool:
    """Whether the clean text replaces ``span`` with its tag."""
    entity = by_name.get(span.entity_type)
    return (
        entity is not None and entity.action is EntityAction.REDACT and span.confidence >= threshold
    )


# --- rendering ----------------------------------------------------------------------------------


def _stamp(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"[{int(minutes):02d}:{rest:04.1f}]"


def _render(
    transcript: Transcript,
    layer: RedactionLayer,
    replace: Callable[[RedactionSpan, str], str | None],
) -> str:
    """Render each segment as ``[mm:ss.s] SPEAKER: text`` with spans substituted by
    ``replace(span, original text)`` (None leaves the text as is)."""
    lines = lines_of(transcript)
    spans = sorted(layer.spans, key=lambda s: s.start)
    out = []
    for line, segment in zip(lines, transcript.segments, strict=True):
        line_end = line.start + len(line.text)
        pieces, pos = [], line.start
        for span in spans:
            start, end = max(span.start, line.start), min(span.end, line_end)
            if start >= end or start < pos:
                continue
            original = line.text[start - line.start : end - line.start]
            substitute = replace(span, original)
            if substitute is None:
                continue
            pieces.append(line.text[pos - line.start : start - line.start])
            pieces.append(substitute)
            pos = end
        pieces.append(line.text[pos - line.start :])
        speaker = f" {segment.speaker}:" if segment.speaker else ""
        out.append(f"{_stamp(segment.start)}{speaker} {''.join(pieces)}")
    return "\n".join(out) + ("\n" if out else "")


def render_annotated(transcript: Transcript, layer: RedactionLayer) -> str:
    """Every span marked inline: ``{{PERSON_1|Maria}}``; flagged spans as
    ``{{FLAG DATE_1|March 1994}}``. For reviewers; contains the original text."""

    def mark(span: RedactionSpan, original: str) -> str:
        tag = span.tag.strip("[]")
        return f"{{{{{'FLAG ' if span.flagged_edge_case else ''}{tag}|{original}}}}}"

    return _render(transcript, layer, mark)


def render_clean(
    transcript: Transcript,
    layer: RedactionLayer,
    entity_types: Sequence[EntityType],
    threshold: float,
) -> str:
    """Redacted spans at or above ``threshold`` replaced by their tags; flagged and
    lower-confidence spans left as written (they are for review)."""
    by_name = {e.name: e for e in entity_types}
    return _render(
        transcript,
        layer,
        lambda span, original: span.tag if redacts(span, by_name, threshold) else None,
    )
