"""P2-05: speaker assignment."""

import io
import random
from pathlib import Path

import pytest
from conftest import SENTINEL, StubJob

from murmurs.backends.stubs import (
    StubAudioLoader,
    StubDiarizer,
    StubLLMClient,
    StubNERDetector,
    StubTranscriber,
)
from murmurs.config import JobConfig, Stage, load_config
from murmurs.models import (
    UNKNOWN_SPEAKER,
    Segment,
    SpeakerTurn,
    StageStatus,
    Transcript,
    TranscriptMode,
    Word,
)
from murmurs.runner import run_job
from murmurs.stages import Backends
from murmurs.stages.assign import assign_speakers
from murmurs.store import ArtifactStore, transcript_to_text

A, B = "SPEAKER_00", "SPEAKER_01"
TOL = 0.5


def _turn(speaker: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(speaker=speaker, start=start, end=end)


def _transcript(*words: tuple[str, float, float], mode: TranscriptMode = TranscriptMode.VERBATIM):
    word_list = [Word(text=t, start=s, end=e) for t, s, e in words]
    segment = Segment(
        start=min(w.start for w in word_list),
        end=max(w.end for w in word_list),
        text=" ".join(w.text for w in word_list),
        words=word_list,
    )
    return Transcript(file_id="f1", mode=mode, segments=[segment])


def _speakers(transcript: Transcript) -> list[str | None]:
    return [w.speaker for s in transcript.segments for w in s.words]


def _speaker_of(start: float, end: float, turns: list[SpeakerTurn], tol: float = TOL) -> str | None:
    return _speakers(assign_speakers(_transcript(("w", start, end)), turns, tol))[0]


# --- the hand-built cases from the acceptance criteria -----------------------------------------


def test_word_fully_inside_one_turn() -> None:
    assert _speaker_of(1.0, 1.5, [_turn(A, 0.0, 2.0), _turn(B, 2.0, 4.0)]) == A


def test_word_straddling_two_turns_goes_to_larger_overlap() -> None:
    turns = [_turn(A, 0.0, 2.0), _turn(B, 2.0, 4.0)]
    assert _speaker_of(1.9, 2.5, turns) == B  # 0.1 s in A, 0.5 s in B
    assert _speaker_of(1.5, 2.1, turns) == A


def test_equal_overlap_goes_to_earlier_turn() -> None:
    assert _speaker_of(1.5, 2.5, [_turn(B, 2.0, 4.0), _turn(A, 0.0, 2.0)]) == A


def test_word_in_gap_within_tolerance_takes_nearest_turn() -> None:
    turns = [_turn(A, 0.0, 1.0), _turn(B, 3.0, 4.0)]
    assert _speaker_of(1.2, 1.4, turns) == A  # 0.2 s after A, 1.6 s before B
    assert _speaker_of(2.6, 2.8, turns) == B  # 0.2 s before B
    assert _speaker_of(1.5, 1.6, turns) == A  # exactly at the tolerance


def test_word_in_gap_beyond_tolerance_is_unknown() -> None:
    turns = [_turn(A, 0.0, 1.0), _turn(B, 3.0, 4.0)]
    assert _speaker_of(1.7, 2.2, turns) == UNKNOWN_SPEAKER
    assert _speaker_of(1.7, 2.2, turns, tol=0.0) == UNKNOWN_SPEAKER
    assert _speaker_of(4.6, 5.0, turns) == UNKNOWN_SPEAKER  # after the last turn


def test_word_equidistant_from_two_turns_goes_to_earlier() -> None:
    assert _speaker_of(1.4, 1.6, [_turn(A, 0.0, 1.0), _turn(B, 2.0, 3.0)]) == A


def test_back_to_back_turns_of_same_speaker_merge_into_one_segment() -> None:
    transcript = _transcript(("a", 0.1, 0.5), ("b", 1.1, 1.5), ("c", 2.1, 2.5), ("d", 3.1, 3.5))
    turns = [_turn(A, 0.0, 1.0), _turn(A, 1.0, 2.0), _turn(B, 2.0, 4.0)]
    labeled = assign_speakers(transcript, turns, TOL)
    assert [(s.speaker, s.text) for s in labeled.segments] == [(A, "a b"), (B, "c d")]


def test_no_turns_makes_every_word_unknown_in_one_segment() -> None:
    labeled = assign_speakers(_transcript(("a", 0.0, 0.5), ("b", 0.6, 1.0)), [], TOL)
    assert [(s.speaker, s.text) for s in labeled.segments] == [(UNKNOWN_SPEAKER, "a b")]


# --- other cases ------------------------------------------------------------------------------


def test_word_in_overlapping_turns_goes_to_larger_overlap() -> None:
    turns = [_turn(A, 0.0, 3.0), _turn(B, 2.0, 5.0)]
    assert _speaker_of(2.2, 3.5, turns) == B
    assert _speaker_of(2.0, 2.4, turns) == A  # inside both, equal overlap: earlier turn


def test_word_matching_a_long_earlier_turn_is_found() -> None:
    # A long turn that starts early still covers a late word, past shorter turns.
    turns = [_turn(A, 0.0, 10.0), _turn(B, 1.0, 2.0), _turn(B, 3.0, 4.0)]
    assert _speaker_of(8.0, 8.5, turns) == A


def test_unknown_turn_from_speaker_cap_labels_words_unknown() -> None:
    assert _speaker_of(1.0, 1.5, [_turn(UNKNOWN_SPEAKER, 0.0, 2.0)]) == UNKNOWN_SPEAKER


def test_zero_length_word_takes_turn_containing_it() -> None:
    turns = [_turn(A, 0.0, 1.0), _turn(B, 1.0, 2.0)]
    assert _speaker_of(0.5, 0.5, turns) == A
    assert _speaker_of(1.0, 1.0, turns) == A  # on the boundary: earlier turn
    assert _speaker_of(1.5, 1.5, turns) == B


def test_word_touching_a_turn_edge_takes_that_turn() -> None:
    assert _speaker_of(2.0, 2.3, [_turn(A, 0.0, 2.0)]) == A


def test_segments_follow_word_order_across_original_segments() -> None:
    first = _transcript(("a", 0.1, 0.4), ("b", 0.5, 0.9)).segments[0]
    second = _transcript(("c", 1.1, 1.4), ("d", 2.1, 2.4)).segments[0]
    transcript = Transcript(file_id="f1", mode=TranscriptMode.INTENDED, segments=[first, second])
    labeled = assign_speakers(transcript, [_turn(A, 0.0, 2.0), _turn(B, 2.0, 3.0)], TOL)
    assert [(s.speaker, s.text, s.start, s.end) for s in labeled.segments] == [
        (A, "a b c", 0.1, 1.4),
        (B, "d", 2.1, 2.4),
    ]
    assert labeled.mode is TranscriptMode.INTENDED and labeled.file_id == "f1"


def test_word_text_and_times_are_unchanged_on_random_input() -> None:
    rng = random.Random(7)
    for _ in range(50):
        words, t = [], 0.0
        for i in range(rng.randint(0, 60)):
            t += rng.uniform(0.0, 1.5)
            length = rng.choice([0.0, rng.uniform(0.05, 0.6)])
            words.append((f"w{i}", round(t, 3), round(t + length, 3)))
            t += length
        turns = []
        for _ in range(rng.randint(0, 12)):
            start = rng.uniform(0, max(t, 1.0))
            turns.append(_turn(rng.choice([A, B]), start, start + rng.uniform(0.0, 8.0)))
        transcript = (
            _transcript(*words)
            if words
            else Transcript(file_id="f1", mode=TranscriptMode.VERBATIM, segments=[])
        )
        labeled = assign_speakers(transcript, turns, rng.choice([0.0, 0.5, 2.0]))
        out = [(w.text, w.start, w.end) for s in labeled.segments for w in s.words]
        assert out == words
        assert all(w.speaker is not None for s in labeled.segments for w in s.words)
        for a, b in zip(labeled.segments, labeled.segments[1:], strict=False):
            assert a.speaker != b.speaker
        for segment in labeled.segments:
            assert {w.speaker for w in segment.words} == {segment.speaker}


def test_modes_with_shared_word_times_get_identical_speakers() -> None:
    words = [("a", 0.1, 0.5), ("b", 0.9, 1.6), ("c", 2.5, 2.7), ("d", 4.0, 4.2)]
    turns = [_turn(A, 0.0, 1.0), _turn(B, 1.2, 3.0)]
    verbatim = assign_speakers(_transcript(("um", 0.0, 0.1), *words), turns, TOL)
    intended = assign_speakers(_transcript(*words, mode=TranscriptMode.INTENDED), turns, TOL)
    by_time_v = {(w.start, w.end): w.speaker for s in verbatim.segments for w in s.words}
    by_time_i = {(w.start, w.end): w.speaker for s in intended.segments for w in s.words}
    assert all(by_time_v[key] == speaker for key, speaker in by_time_i.items())


def test_text_export_has_speaker_and_start_time() -> None:
    labeled = assign_speakers(
        _transcript(("hello", 61.25, 61.5), ("there", 62.0, 62.4)), [_turn(A, 61.0, 63.0)], TOL
    )
    assert transcript_to_text(labeled) == f"[01:01.2] {A}: hello there\n"


# --- through the runner -------------------------------------------------------------------------


def _backends(text: str = "hello there general kenobi") -> Backends:
    return Backends(
        audio_loader=StubAudioLoader(duration_s=4.0),
        diarizer=StubDiarizer(turn_s=1.0),
        transcriber=StubTranscriber(text=text),
        ner=StubNERDetector(),
        llm=StubLLMClient(response={"entities": []}),
    )


def _until_assign(config: JobConfig, **diarization: object) -> JobConfig:
    return config.model_copy(
        update={
            "stages": config.stages.model_copy(update={"end": Stage.ASSIGN_SPEAKERS}),
            "diarization": config.diarization.model_copy(update=diarization),
        }
    )


def _store(config: JobConfig) -> ArtifactStore:
    return ArtifactStore(config.data.output_dir, config.data.input_dir, "job")


def _mtimes(config: JobConfig, file_id: str) -> dict[str, int]:
    folder = _store(config).file_dir(file_id)
    return {p.name: p.stat().st_mtime_ns for p in folder.iterdir() if p.name != "status.json"}


def test_stage_labels_both_modes_and_writes_text(stub_job: StubJob) -> None:
    config = _until_assign(load_config(stub_job()))
    summary = run_job(config, "job", backends=_backends(), console=io.StringIO())
    assert len(summary.done) == 3 and not summary.failed
    store = _store(config)
    for file_id in summary.done:
        by_mode = {}
        for mode in TranscriptMode:
            labeled = store.read(file_id, Stage.ASSIGN_SPEAKERS, Transcript, mode.value)
            source = store.read(file_id, Stage.TRANSCRIBE, Transcript, mode.value)
            assert [(w.text, w.start, w.end) for s in labeled.segments for w in s.words] == [
                (w.text, w.start, w.end) for s in source.segments for w in s.words
            ]
            assert {s.speaker for s in labeled.segments} <= {A, B}
            by_mode[mode] = {(w.start, w.end): w.speaker for s in labeled.segments for w in s.words}
            text = store.artifact_path(file_id, Stage.ASSIGN_SPEAKERS, mode.value, suffix=".txt")
            assert all(": " in line for line in text.read_text().splitlines())
        shared = by_mode[TranscriptMode.VERBATIM].keys() & by_mode[TranscriptMode.INTENDED].keys()
        assert all(
            by_mode[TranscriptMode.VERBATIM][k] == by_mode[TranscriptMode.INTENDED][k]
            for k in shared
        )


def test_changing_tolerance_reruns_only_assignment(stub_job: StubJob) -> None:
    config = _until_assign(load_config(stub_job()))
    first = run_job(config, "job", backends=_backends(), console=io.StringIO())
    file_id = first.done[0]
    before = _mtimes(config, file_id)

    backends = _backends()
    run_job(
        _until_assign(config, speaker_gap_tolerance_s=2.0),
        "job",
        backends=backends,
        console=io.StringIO(),
    )

    assert backends.diarizer.calls == [] and backends.transcriber.calls == []  # type: ignore[union-attr]
    after = _mtimes(config, file_id)
    rewritten = {name for name in after if after[name] != before.get(name)}
    assert rewritten and all(name.startswith("assign_speakers") for name in rewritten)
    status = _store(config).read_status(file_id)
    assert status is not None and status.stages[Stage.ASSIGN_SPEAKERS] is StageStatus.DONE


def test_sentinel_never_reaches_logs(stub_job: StubJob, tmp_path: Path) -> None:
    config = _until_assign(load_config(stub_job()))
    console = io.StringIO()
    summary = run_job(config, "job", backends=_backends(f"my name is {SENTINEL}"), console=console)
    assert summary.done and not summary.failed
    store = _store(config)
    labeled = store.read(summary.done[0], Stage.ASSIGN_SPEAKERS, Transcript, "intended")
    assert any(SENTINEL in s.text for s in labeled.segments)
    logs = "".join(p.read_text() for p in (store.job_dir / "logs").iterdir())
    metrics = (store.job_dir / "metrics.json").read_text()
    for text in (console.getvalue(), logs, metrics, store.manifest_path.read_text()):
        assert SENTINEL not in text


@pytest.mark.parametrize("tolerance", [0.0, 0.5])
def test_stage_uses_configured_tolerance(stub_job: StubJob, tolerance: float) -> None:
    # StubDiarizer covers the whole recording, so no word is ever UNKNOWN either way;
    # this checks the value reaches the stage without errors at both ends of the range.
    config = _until_assign(load_config(stub_job()), speaker_gap_tolerance_s=tolerance)
    summary = run_job(config, "job", backends=_backends(), console=io.StringIO())
    assert len(summary.done) == 3


def _reference_speaker(word: Word, turns: list[SpeakerTurn], tol: float) -> str:
    """Slow, obvious version of the rules, for comparison."""
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    best, best_overlap = None, -1.0
    for turn in ordered:
        overlap = min(word.end, turn.end) - max(word.start, turn.start)
        point_inside = word.start == word.end and turn.start <= word.start <= turn.end
        if (overlap > 0 or point_inside) and overlap > best_overlap:
            best, best_overlap = turn, overlap
    if best is not None:
        return best.speaker
    gaps = [
        (max(turn.start - word.end, word.start - turn.end), i) for i, turn in enumerate(ordered)
    ]
    if not gaps:
        return UNKNOWN_SPEAKER
    gap, i = min(gaps)
    return ordered[i].speaker if gap <= tol else UNKNOWN_SPEAKER


def test_matches_reference_implementation_on_random_input() -> None:
    rng = random.Random(11)
    speakers = [A, B, "SPEAKER_02", UNKNOWN_SPEAKER]
    for _ in range(300):
        grid = [round(rng.uniform(0, 20) * 4) / 4 for _ in range(40)]  # shared edges are common
        turns = []
        for _ in range(rng.randint(0, 15)):
            start, end = sorted(rng.sample(grid, 2))
            turns.append(_turn(rng.choice(speakers), start, end))
        words = []
        for i in range(rng.randint(1, 30)):
            start = rng.choice(grid)
            end = start if rng.random() < 0.2 else start + rng.choice([0.1, 0.25, 0.5, 1.0])
            words.append((f"w{i}", start, end))
        words.sort(key=lambda w: w[1])
        tol = rng.choice([0.0, 0.25, 0.5, 3.0])
        labeled = assign_speakers(_transcript(*words), turns, tol)
        got = [w.speaker for s in labeled.segments for w in s.words]
        want = [_reference_speaker(Word(text=t, start=s, end=e), turns, tol) for t, s, e in words]
        assert got == want
