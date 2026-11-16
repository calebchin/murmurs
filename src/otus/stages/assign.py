"""Speaker assignment: label each transcribed word with a diarization speaker (D5).

Pure functions, no model calls. Word text and timestamps are never changed; only word
speakers and the grouping of words into segments.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

from murmurs.models import UNKNOWN_SPEAKER, Segment, SpeakerTurn, Transcript, Word


class _Turns:
    """Turns sorted by start time, indexed for fast lookup around a time range.

    Turns may overlap, so end times are not sorted; ``max_end[i]`` is the largest end
    among turns ``0..i`` and ``max_end_at[i]`` the earliest turn reaching it.
    """

    def __init__(self, turns: Sequence[SpeakerTurn]) -> None:
        self.turns = sorted(turns, key=lambda t: (t.start, t.end))
        self.starts = [t.start for t in self.turns]
        self.max_end: list[float] = []
        self.max_end_at: list[int] = []
        for i, turn in enumerate(self.turns):
            if not self.max_end or turn.end > self.max_end[-1]:
                self.max_end.append(turn.end)
                self.max_end_at.append(i)
            else:
                self.max_end.append(self.max_end[-1])
                self.max_end_at.append(self.max_end_at[-1])

    def speaker_for(self, word: Word, gap_tolerance_s: float) -> str:
        """The speaker of the turn ``word`` overlaps most (ties: earlier turn); else the
        nearest turn within ``gap_tolerance_s`` (ties: earlier turn); else ``UNKNOWN``."""
        if not self.turns:
            return UNKNOWN_SPEAKER
        # Turns starting at or after the word's end cannot overlap it, except that a
        # zero-length word overlaps a turn that starts exactly at it.
        hi = bisect_left(self.starts, word.end)
        if word.start == word.end:
            while hi < len(self.turns) and self.starts[hi] == word.end:
                hi += 1
        # Turns before ``lo`` all end at or before the word's start.
        lo = bisect_left(self.max_end, word.start)

        best, best_overlap = None, -1.0
        for i in range(lo, hi):
            turn = self.turns[i]
            overlap = min(word.end, turn.end) - max(word.start, turn.start)
            point_inside = word.start == word.end and turn.start <= word.start <= turn.end
            if (overlap > 0 or point_inside) and overlap > best_overlap:
                best, best_overlap = turn, overlap
        if best is not None:
            return best.speaker

        # No overlap: every turn before ``hi`` ends at or before the word starts, and the
        # turn at ``hi`` starts at or after it ends.
        candidates: list[tuple[float, int]] = []
        if hi > 0:
            i = self.max_end_at[hi - 1]
            candidates.append((word.start - self.turns[i].end, i))
        if hi < len(self.turns):
            candidates.append((self.turns[hi].start - word.end, hi))
        gap, i = min(candidates)  # equal gaps: lower index, i.e. the earlier turn
        return self.turns[i].speaker if gap <= gap_tolerance_s else UNKNOWN_SPEAKER


def shift_turns(turns: Sequence[SpeakerTurn], offset_s: float) -> list[SpeakerTurn]:
    """Move every turn ``offset_s`` seconds earlier, clamping times at 0 (D20)."""
    if offset_s == 0:
        return list(turns)
    return [
        turn.model_copy(
            update={"start": max(0.0, turn.start - offset_s), "end": max(0.0, turn.end - offset_s)}
        )
        for turn in turns
    ]


def words_of(transcript: Transcript) -> list[Word]:
    """Every word of a transcript, in order."""
    return [word for segment in transcript.segments for word in segment.words]


def assign_speakers(
    transcript: Transcript, turns: Sequence[SpeakerTurn], gap_tolerance_s: float
) -> Transcript:
    """Label every word with a speaker and regroup words into speaker segments.

    Segments become runs of consecutive words with the same speaker, in the original
    word order; a segment spans its words and its text is the words joined with spaces.
    """
    index = _Turns(turns)
    speakers = [index.speaker_for(word, gap_tolerance_s) for word in words_of(transcript)]
    return relabel(transcript, speakers)


def relabel(transcript: Transcript, speakers: Sequence[str]) -> Transcript:
    """Give word ``i`` speaker ``speakers[i]`` and regroup words into speaker segments.

    Word text and times are unchanged.
    """
    words = words_of(transcript)
    if len(speakers) != len(words):
        raise ValueError("one speaker is needed per word")
    labeled = [w.model_copy(update={"speaker": s}) for w, s in zip(words, speakers, strict=True)]
    segments: list[Segment] = []
    run: list[Word] = []
    for word in labeled:
        if run and word.speaker != run[-1].speaker:
            segments.append(_segment(run))
            run = []
        run.append(word)
    if run:
        segments.append(_segment(run))
    return Transcript(file_id=transcript.file_id, mode=transcript.mode, segments=segments)


def _segment(words: list[Word]) -> Segment:
    return Segment(
        speaker=words[0].speaker,
        start=words[0].start,
        end=max(w.end for w in words),
        text=" ".join(w.text for w in words),
        words=words,
    )
