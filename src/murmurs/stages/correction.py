"""LLM speaker correction (P2-07): fix diarization label mistakes after rule-based assignment.

The LLM sees the speaker-labeled transcript in chunks, with every word numbered, and
returns label changes only: a word range, the range's text (echoed, as a check) and the
new speaker. Every change is validated before it is applied; the transcript's words and
times never change. One mode is corrected and the other follows by word time (D24).

Nothing here logs: prompts and responses hold transcript text (hard rule 1). Counts go
into the ``SpeakerCorrection`` report.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template

from pydantic import BaseModel, ConfigDict

from murmurs.backends import LLMClient
from murmurs.models import (
    UNKNOWN_SPEAKER,
    SpeakerCorrection,
    SpeakerTurn,
    Transcript,
    TranscriptMode,
    Word,
)
from murmurs.stages.assign import _Turns, relabel, words_of
from murmurs.stages.llm_chunks import halves, run_chunks

PROMPT_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = PROMPT_DIR / "speaker_correction_system.txt"
USER_PROMPT = PROMPT_DIR / "speaker_correction_user.txt"

CHARS_PER_TOKEN = 3.0
"""Conservative estimate for sizing chunks without loading the LLM's tokenizer."""
CHUNK_SHARE = 4
"""A chunk's own words use at most 1/CHUNK_SHARE of the LLM's context, leaving room for
context lines, instructions and the response."""
CONTEXT_SHARE = 8
"""Context shown before and after a chunk: up to chunk budget / CONTEXT_SHARE each side."""
CORRECTION_CHUNK_TOKENS = 1024.0
"""Transcript tokens a chunk owns. The model reasons for a few thousand tokens per
request whatever the chunk size, so many small chunks run concurrently finish sooner
than a few large ones: on a 68-minute podcast, 1,024-token chunks took 95 s against
150 s for a quarter of a 32k context (P3-05)."""

REJECT_INVALID_RANGE = "invalid_range"
REJECT_MISSING_WORD = "missing_word"
REJECT_OUTSIDE_CHUNK = "outside_chunk"
REJECT_TEXT_MISMATCH = "text_mismatch"
REJECT_UNKNOWN_SPEAKER = "unknown_speaker"


class LabelChange(BaseModel):
    """One change the LLM proposes: give words ``first_word``..``last_word`` a speaker."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    first_word: int
    last_word: int
    text: str
    """The range's words as shown, joined by single spaces; must match exactly."""
    speaker: str


class SpeakerCorrections(BaseModel):
    """The LLM's structured response for one chunk."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    changes: list[LabelChange]


def prompt_hash() -> str:
    """Hash of the prompt files; part of the stage config, so editing a prompt reruns it."""
    digest = hashlib.sha256()
    for path in (SYSTEM_PROMPT, USER_PROMPT):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


# --- chunking -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Line:
    """A run of consecutive words from one speaker: word indices ``start``..``end - 1``."""

    start: int
    end: int
    speaker: str


@dataclass(frozen=True)
class Chunk:
    """Lines the LLM may change, with context lines before and after."""

    before: list[_Line]
    own: list[_Line]
    after: list[_Line]

    @property
    def own_words(self) -> range:
        """Word indices the LLM may change in this chunk."""
        return range(self.own[0].start, self.own[-1].end)


def _word_tokens(word: Word, index: int) -> float:
    return (len(word.text) + len(str(index)) + 3) / CHARS_PER_TOKEN


def _lines(words: Sequence[Word], budget: float) -> list[_Line]:
    """Speaker runs, each split so no line exceeds ``budget`` tokens."""
    lines: list[_Line] = []
    start, size = 0, 0.0
    for i, word in enumerate(words):
        cost = _word_tokens(word, i)
        if i > start and (word.speaker != words[start].speaker or size + cost > budget):
            lines.append(_Line(start, i, words[start].speaker or UNKNOWN_SPEAKER))
            start, size = i, 0.0
        size += cost
    if words:
        lines.append(_Line(start, len(words), words[start].speaker or UNKNOWN_SPEAKER))
    return lines


def chunk_transcript(words: Sequence[Word], max_model_len: int) -> list[Chunk]:
    """Split a labeled transcript into chunks of about ``CORRECTION_CHUNK_TOKENS`` (less
    if the context window is small).

    Every word belongs to exactly one chunk's own lines. Each chunk also shows whole
    lines before and after it as context, so turns at its edges are seen whole.
    """
    budget = max(200.0, min(CORRECTION_CHUNK_TOKENS, max_model_len / CHUNK_SHARE))
    context_budget = budget / CONTEXT_SHARE
    lines = _lines(words, budget)
    cost = [sum(_word_tokens(words[i], i) for i in range(ln.start, ln.end)) for ln in lines]

    groups: list[tuple[int, int]] = []
    first, size = 0, 0.0
    for i, line_cost in enumerate(cost):
        if i > first and size + line_cost > budget:
            groups.append((first, i))
            first, size = i, 0.0
        size += line_cost
    if lines:
        groups.append((first, len(lines)))

    chunks = []
    for first, last in groups:
        before: list[_Line] = []
        size = 0.0
        for i in range(first - 1, -1, -1):
            if before and size + cost[i] > context_budget:
                break
            before.insert(0, lines[i])
            size += cost[i]
        after: list[_Line] = []
        size = 0.0
        for i in range(last, len(lines)):
            if after and size + cost[i] > context_budget:
                break
            after.append(lines[i])
            size += cost[i]
        chunks.append(Chunk(before=before, own=lines[first:last], after=after))
    return chunks


def render_chunk(chunk: Chunk, words: Sequence[Word]) -> str:
    """The chunk as the LLM sees it: one line per speaker run, every word numbered."""

    def line(ln: _Line, prefix: str) -> str:
        numbered = " ".join(f"[{i}]{words[i].text}" for i in range(ln.start, ln.end))
        return f"{prefix}{ln.speaker}: {numbered}"

    return "\n".join(
        [line(ln, "CONTEXT ") for ln in chunk.before]
        + [line(ln, "") for ln in chunk.own]
        + [line(ln, "CONTEXT ") for ln in chunk.after]
    )


def build_prompt(chunk: Chunk, words: Sequence[Word], speakers: Sequence[str]) -> tuple[str, str]:
    """``(system, user)`` prompts for one chunk."""
    system = SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    user = Template(USER_PROMPT.read_text(encoding="utf-8")).substitute(
        speakers=", ".join(speakers), transcript=render_chunk(chunk, words)
    )
    return system, user.strip()


# --- validating and applying changes ---------------------------------------------------------


def check_change(
    change: LabelChange, words: Sequence[Word], own: range, speakers: Sequence[str]
) -> str | None:
    """Why ``change`` must be rejected, or None if it is valid."""
    if change.first_word > change.last_word:
        return REJECT_INVALID_RANGE
    if change.first_word < 0 or change.last_word >= len(words):
        return REJECT_MISSING_WORD
    if change.first_word not in own or change.last_word not in own:
        return REJECT_OUTSIDE_CHUNK
    expected = " ".join(words[i].text for i in range(change.first_word, change.last_word + 1))
    if " ".join(change.text.split()) != expected:
        return REJECT_TEXT_MISMATCH
    if change.speaker not in speakers:
        return REJECT_UNKNOWN_SPEAKER
    return None


SPLIT_CONTEXT_LINES = 2
"""Lines of the other half shown as context when a chunk is split."""


def split_chunk(chunk: Chunk) -> tuple[Chunk, Chunk] | None:
    """Two chunks, each owning half of ``chunk``'s lines, or None for a one-line chunk.
    Each half shows a couple of the other half's lines as context."""
    parts = halves(chunk.own)
    if parts is None:
        return None
    first, second = parts
    return (
        Chunk(before=chunk.before, own=first, after=second[:SPLIT_CONTEXT_LINES]),
        Chunk(before=first[-SPLIT_CONTEXT_LINES:], own=second, after=chunk.after),
    )


@dataclass
class _Counts:
    n_chunks: int = 0
    splits: int = 0
    failed_chunks: int = 0
    proposed: int = 0
    applied: int = 0

    def __post_init__(self) -> None:
        self.rejected: Counter[str] = Counter()


def correct_transcript(
    transcript: Transcript,
    llm: LLMClient,
    file_id: str,
    max_model_len: int,
    *,
    max_concurrency: int = 1,
) -> tuple[Transcript, set[int], _Counts]:
    """Run the LLM over ``transcript`` and apply its valid label changes.

    Chunks go to the LLM up to ``max_concurrency`` at a time; a chunk whose response is
    truncated or invalid is split in half (see ``llm_chunks``). Returns the corrected
    transcript, the indices of words whose speaker changed, and counts. A chunk that
    stays unusable keeps its labels; a failed request raises ``PipelineError`` and fails
    the file.
    """
    words = words_of(transcript)
    speakers = sorted({w.speaker for w in words if w.speaker and w.speaker != UNKNOWN_SPEAKER})
    labels = [w.speaker or UNKNOWN_SPEAKER for w in words]
    counts = _Counts()
    if not speakers:
        return transcript, set(), counts

    def ask(chunk: Chunk) -> SpeakerCorrections:
        system, user = build_prompt(chunk, words, speakers)
        return llm.generate(user, SpeakerCorrections, file_id, system=system)

    chunks = chunk_transcript(words, max_model_len)
    outcome = run_chunks(chunks, ask, split_chunk, max_concurrency)
    counts.n_chunks = len(chunks)
    counts.splits = outcome.splits
    counts.failed_chunks = outcome.failed
    for chunk, response in outcome.results:
        for change in response.changes:
            counts.proposed += 1
            reason = check_change(change, words, chunk.own_words, speakers)
            if reason is not None:
                counts.rejected[reason] += 1
                continue
            counts.applied += 1
            for i in range(change.first_word, change.last_word + 1):
                labels[i] = change.speaker

    changed = {i for i, w in enumerate(words) if labels[i] != (w.speaker or UNKNOWN_SPEAKER)}
    return relabel(transcript, labels), changed, counts


def follow_changes(
    other: Transcript, source_words: Sequence[Word], changed: set[int], new_labels: Sequence[str]
) -> Transcript:
    """Carry corrections to another mode of the same file by word time (D24).

    Each word in ``other`` that overlaps a changed source word takes that word's new
    speaker (most overlap; ties go to the earlier word). Every other word keeps its
    speaker.
    """
    changed_turns = [
        SpeakerTurn(speaker=new_labels[i], start=source_words[i].start, end=source_words[i].end)
        for i in sorted(changed)
    ]
    index = _Turns(changed_turns)
    labels = []
    for word in words_of(other):
        speaker = index.speaker_for(word, gap_tolerance_s=-1.0)  # overlap only, no gaps
        labels.append(word.speaker or UNKNOWN_SPEAKER if speaker == UNKNOWN_SPEAKER else speaker)
    return relabel(other, labels)


def correct_modes(
    rule_based: Mapping[TranscriptMode, Transcript],
    llm: LLMClient,
    file_id: str,
    max_model_len: int,
    *,
    max_concurrency: int = 1,
) -> tuple[dict[TranscriptMode, Transcript], SpeakerCorrection]:
    """Correct one mode with the LLM (intended if present) and carry the result to the
    other modes by word time. Returns the corrected transcripts and the report."""
    usage_before = llm.usage.model_copy()
    source = (
        TranscriptMode.INTENDED if TranscriptMode.INTENDED in rule_based else next(iter(rule_based))
    )
    corrected, changed, counts = correct_transcript(
        rule_based[source], llm, file_id, max_model_len, max_concurrency=max_concurrency
    )
    source_words = words_of(rule_based[source])
    new_labels = [w.speaker or UNKNOWN_SPEAKER for w in words_of(corrected)]
    final = {source: corrected}
    for mode, transcript in rule_based.items():
        if mode is not source:
            final[mode] = follow_changes(transcript, source_words, changed, new_labels)

    words_changed = {
        mode: sum(
            a.speaker != b.speaker
            for a, b in zip(words_of(rule_based[mode]), words_of(final[mode]), strict=True)
        )
        for mode in rule_based
    }
    report = SpeakerCorrection(
        file_id=file_id,
        enabled=True,
        source_mode=source,
        n_chunks=counts.n_chunks,
        chunk_splits=counts.splits,
        failed_chunks=counts.failed_chunks,
        llm=llm.usage.minus(usage_before),
        proposed=counts.proposed,
        applied=counts.applied,
        rejected=dict(counts.rejected),
        words_changed=words_changed,
    )
    return {mode: final[mode] for mode in rule_based}, report
