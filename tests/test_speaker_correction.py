"""P2-07: LLM speaker correction."""

import io
import itertools
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from conftest import SENTINEL, StubJob
from fixtures.audio import tone_wav_bytes
from pydantic import ValidationError

from murmurs.backends import M, registry
from murmurs.backends.registry import BackendKind
from murmurs.config import JobConfig, Stage, load_config
from murmurs.errors import LLM_INVALID_OUTPUT, PipelineError
from murmurs.models import (
    UNKNOWN_SPEAKER,
    LLMUsage,
    Segment,
    SpeakerCorrection,
    StageStatus,
    Transcript,
    TranscriptMode,
    Word,
)
from murmurs.runner import backend_kinds, run_job, stage_hashes
from murmurs.stages import correction
from murmurs.stages.assign import words_of
from murmurs.stages.correction import (
    LabelChange,
    SpeakerCorrections,
    chunk_transcript,
    correct_modes,
    correct_transcript,
    follow_changes,
    render_chunk,
)
from murmurs.store import ArtifactStore

A, B = "SPEAKER_00", "SPEAKER_01"
INT, VERB = TranscriptMode.INTENDED, TranscriptMode.VERBATIM


class ScriptedLLM:
    """An ``LLMClient`` whose responses come from a function of the user prompt."""

    model_version = "scripted"
    usage = LLMUsage()  # counts only; scripted LLMs report no tokens

    def __init__(self, respond: Callable[[str], Any]) -> None:
        self.respond = respond
        self.prompts: list[tuple[str | None, str]] = []
        self.closed = 0

    def generate(
        self, prompt: str, response_model: type[M], file_id: str, system: str | None = None
    ) -> M:
        self.prompts.append((system, prompt))
        reply = self.respond(prompt)
        if isinstance(reply, Exception):
            raise reply
        return response_model.model_validate(reply)

    def close(self) -> None:
        self.closed += 1


def _changes(*changes: tuple[int, int, str, str]) -> dict[str, Any]:
    return {
        "changes": [
            {"first_word": first, "last_word": last, "text": t, "speaker": s}
            for first, last, t, s in changes
        ]
    }


def _labeled(*runs: tuple[str, str], mode: TranscriptMode = INT, t0: float = 0.0) -> Transcript:
    """A labeled transcript from (speaker, "words ...") runs, 0.5 s per word."""
    segments, t = [], t0
    for speaker, text in runs:
        words = []
        for token in text.split():
            words.append(Word(text=token, start=t, end=t + 0.4, speaker=speaker))
            t += 0.5
        segments.append(
            Segment(
                speaker=speaker, start=words[0].start, end=words[-1].end, text=text, words=words
            )
        )
    return Transcript(file_id="f1", mode=mode, segments=segments)


def _texts(transcript: Transcript) -> list[tuple[str, float, float]]:
    return [(w.text, w.start, w.end) for w in words_of(transcript)]


def _labels(transcript: Transcript) -> list[str | None]:
    return [w.speaker for w in words_of(transcript)]


THREE_RUNS = (("A", "so tell me more"), ("B", "we lived there"), ("A", "and then what"))


def _three() -> Transcript:
    return _labeled(*[(A if s == "A" else B, text) for s, text in THREE_RUNS])


# --- the acceptance cases ------------------------------------------------------------------


def test_relabeling_a_known_segment_changes_only_that_segment() -> None:
    transcript = _three()
    llm = ScriptedLLM(lambda prompt: _changes((4, 6, "we lived there", A)))
    corrected, changed, counts = correct_transcript(transcript, llm, "f1", 32768)
    assert _labels(corrected) == [A] * 10
    assert [(s.speaker, s.text) for s in corrected.segments] == [
        (A, "so tell me more we lived there and then what")
    ]
    assert changed == {4, 5, 6} and counts.applied == 1 and not counts.rejected
    assert _texts(corrected) == _texts(transcript)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ((4, 6, "we loved there", A), "text_mismatch"),  # tries to change word text
        ((4, 6, "we lived there", "SPEAKER_07"), "unknown_speaker"),  # invents a speaker
        ((4, 6, "we lived there", UNKNOWN_SPEAKER), "unknown_speaker"),
        ((40, 41, "x y", A), "missing_word"),  # references missing words
        ((-1, 0, "so", B), "missing_word"),
        ((6, 4, "there lived we", A), "invalid_range"),
    ],
)
def test_invalid_changes_are_rejected_and_text_is_unchanged(
    change: tuple[int, int, str, str], reason: str
) -> None:
    transcript = _three()
    llm = ScriptedLLM(lambda prompt: _changes(change))
    corrected, changed, counts = correct_transcript(transcript, llm, "f1", 32768)
    assert counts.rejected == {reason: 1} and counts.applied == 0 and not changed
    assert _texts(corrected) == _texts(transcript)
    assert _labels(corrected) == _labels(transcript)


def test_valid_changes_apply_when_others_in_the_same_response_are_rejected() -> None:
    llm = ScriptedLLM(
        lambda prompt: _changes(
            (4, 4, "we", A),  # valid word-level change
            (5, 6, "lived THERE", A),  # wrong text
            (7, 7, "and", "SPEAKER_09"),  # unknown speaker
        )
    )
    corrected, changed, counts = correct_transcript(_three(), llm, "f1", 32768)
    assert changed == {4} and counts.proposed == 3 and counts.applied == 1
    assert counts.rejected == {"text_mismatch": 1, "unknown_speaker": 1}
    assert _labels(corrected)[3:6] == [A, A, B]


def test_change_text_whitespace_is_normalised() -> None:
    llm = ScriptedLLM(lambda prompt: _changes((4, 6, "  we   lived\nthere ", A)))
    _, changed, _ = correct_transcript(_three(), llm, "f1", 32768)
    assert changed == {4, 5, 6}


def test_chunk_with_persistently_invalid_output_keeps_its_labels() -> None:
    llm = ScriptedLLM(lambda prompt: PipelineError(LLM_INVALID_OUTPUT))
    corrected, changed, counts = correct_transcript(_three(), llm, "f1", 32768)
    # Invalid output splits the 3-line chunk down to single lines, each of which fails.
    assert counts.splits == 2 and counts.failed_chunks == 3 and not changed
    assert _labels(corrected) == _labels(_three())


def test_failed_request_fails_the_file() -> None:
    llm = ScriptedLLM(lambda prompt: PipelineError("LLM_REQUEST_FAILED"))
    with pytest.raises(PipelineError, match="LLM_REQUEST_FAILED"):
        correct_transcript(_three(), llm, "f1", 32768)


def test_no_named_speakers_means_no_llm_call() -> None:
    llm = ScriptedLLM(lambda prompt: _changes())
    transcript = _labeled((UNKNOWN_SPEAKER, "who knows"))
    _, changed, counts = correct_transcript(transcript, llm, "f1", 32768)
    assert llm.prompts == [] and counts.n_chunks == 0 and not changed


# --- prompts and chunking ---------------------------------------------------------------------


def test_prompt_numbers_every_word_and_lists_speakers() -> None:
    llm = ScriptedLLM(lambda prompt: _changes())
    correct_transcript(_three(), llm, "f1", 32768)
    [(system, user)] = llm.prompts
    assert system is not None and "only change which speaker" in system
    assert f"Speakers in this transcript: {A}, {B}" in user
    assert f"{A}: [0]so [1]tell [2]me [3]more" in user
    assert f"{B}: [4]we [5]lived [6]there" in user
    assert "CONTEXT" not in user.split("never change words on CONTEXT lines.")[1].split("List")[0]


def _long(n_runs: int) -> Transcript:
    return _labeled(
        *[(A if i % 2 == 0 else B, " ".join(f"w{i}x{j}" for j in range(12))) for i in range(n_runs)]
    )


def test_chunks_cover_every_word_once_with_context_at_the_edges() -> None:
    transcript = _long(40)
    words = words_of(transcript)
    chunks = chunk_transcript(words, max_model_len=1024)
    assert len(chunks) > 3
    owned = [i for c in chunks for i in c.own_words]
    assert owned == list(range(len(words)))
    assert chunks[0].before == [] and chunks[-1].after == []
    for prev, nxt in itertools.pairwise(chunks):
        assert prev.after and prev.after[0] == nxt.own[0]  # the next chunk's start is context
        assert nxt.before and nxt.before[-1] == prev.own[-1]
    text = render_chunk(chunks[1], words)
    assert text.startswith("CONTEXT ") and text.rstrip().splitlines()[-1].startswith("CONTEXT ")


def test_correction_chunks_are_small_whatever_the_context_window() -> None:
    words = words_of(_long(400))
    chunks = chunk_transcript(words, max_model_len=131072)
    assert len(chunks) > 10  # about 1,000 tokens each, not max_model_len / 4
    assert [i for c in chunks for i in c.own_words] == list(range(len(words)))


def test_a_run_longer_than_a_chunk_is_split() -> None:
    transcript = _labeled((A, " ".join(f"word{j}" for j in range(600))))
    chunks = chunk_transcript(words_of(transcript), max_model_len=1024)
    assert len(chunks) > 1
    assert [i for c in chunks for i in c.own_words] == list(range(600))


def test_changes_to_context_words_are_rejected() -> None:
    transcript = _long(40)
    words = words_of(transcript)
    chunks = chunk_transcript(words, max_model_len=1024)
    target = chunks[1].own_words.start  # owned by chunk 1, shown as context in chunk 0
    calls = {"n": 0}

    def respond(prompt: str) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:  # chunk 0 tries to change a word it only sees as context
            return _changes(
                (target, target, words[target].text, A if words[target].speaker == B else B)
            )
        return _changes()

    _, changed, counts = correct_transcript(transcript, ScriptedLLM(respond), "f1", 1024)
    assert counts.rejected == {"outside_chunk": 1} and not changed
    assert counts.n_chunks == len(chunks)


# --- the other mode follows by time (D24) ----------------------------------------------------


def test_verbatim_follows_corrected_intended_by_time() -> None:
    intended = _three()
    # Verbatim: an extra filler "um" in a gap, and the same words at the same times.
    verb_words = []
    for w in words_of(intended):
        verb_words.append(w.model_copy(update={"text": w.text}))
    filler = Word(text="um", start=4.42, end=4.48, speaker=B)
    verb_words.insert(9, filler)
    verbatim = Transcript(
        file_id="f1",
        mode=VERB,
        segments=[
            Segment(speaker=None, start=w.start, end=w.end, text=w.text, words=[w])
            for w in verb_words
        ],
    )
    llm = ScriptedLLM(lambda prompt: _changes((4, 6, "we lived there", A)))
    final, report = correct_modes({VERB: verbatim, INT: intended}, llm, "f1", 32768)
    assert report.source_mode is INT and len(llm.prompts) == 1
    verb_labels = {(w.text, w.start): w.speaker for w in words_of(final[VERB])}
    assert verb_labels[("we", 2.0)] == A and verb_labels[("there", 3.0)] == A
    assert verb_labels[("um", 4.42)] == B  # overlaps no changed word: unchanged
    assert report.words_changed == {VERB: 3, INT: 3}


def test_follow_changes_ties_go_to_the_earlier_word() -> None:
    source = [
        Word(text="a", start=0.0, end=1.0, speaker=A),
        Word(text="b", start=1.0, end=2.0, speaker=A),
    ]
    other = _labeled((B, "x"), t0=0.5)  # 0.5-0.9, inside "a"
    straddle = Transcript(
        file_id="f1",
        mode=VERB,
        segments=[
            Segment(
                start=0.5, end=1.5, text="y", words=[Word(text="y", start=0.5, end=1.5, speaker=B)]
            )
        ],
    )
    assert _labels(follow_changes(other, source, {0, 1}, ["S2", "S3"])) == ["S2"]
    assert _labels(follow_changes(straddle, source, {0, 1}, ["S2", "S3"])) == ["S2"]


def test_only_verbatim_configured_corrects_verbatim() -> None:
    verbatim = _labeled(*[(A if s == "A" else B, t) for s, t in THREE_RUNS], mode=VERB)
    llm = ScriptedLLM(lambda prompt: _changes((4, 6, "we lived there", A)))
    final, report = correct_modes({VERB: verbatim}, llm, "f1", 32768)
    assert report.source_mode is VERB and _labels(final[VERB]) == [A] * 10


# --- through the runner -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    yield


class ScriptedRegistry:
    """Registers a scripted LLM called "scripted" that moves the first word of each
    SPEAKER_01 line it may edit to SPEAKER_00, echoing the text (optionally a sentinel)."""

    def __init__(self) -> None:
        self.created: list[ScriptedLLM] = []
        registry.register(BackendKind.LLM, "scripted", self._create)

    def _create(self, config: JobConfig) -> ScriptedLLM:
        llm = ScriptedLLM(self._respond)
        self.created.append(llm)
        return llm

    @staticmethod
    def _respond(prompt: str) -> dict[str, Any]:
        body = prompt.split("never change words on CONTEXT lines.", 1)[1]
        for line in body.splitlines():
            if line.startswith(f"{B}: "):
                number, text = re.match(r"\[(\d+)\](\S+)", line[len(B) + 2 :]).groups()  # type: ignore[union-attr]
                return _changes((int(number), int(number), text, A))
        return _changes()


def _job(
    stub_job: StubJob, correction_on: bool, text: str = "one two three four five six"
) -> JobConfig:
    files = {
        "a.wav": tone_wav_bytes(seconds=3.0, frequency=440),
        "b.wav": tone_wav_bytes(seconds=3.0, frequency=660),
    }
    config = load_config(
        stub_job(
            files,
            diarization={"llm_correction": correction_on},
            deid={"llm_model": "scripted"},
            stages={"end": "assign_speakers"},  # these tests are about stage 5 only
        )
    )
    return config


def _run(config: JobConfig, console: io.StringIO | None = None, text: str | None = None) -> Any:
    if text is not None:
        from murmurs.backends.stubs import StubTranscriber

        registry.register(BackendKind.TRANSCRIBER, "stub", lambda _: StubTranscriber(text=text))
    return run_job(config, "job", console=console or io.StringIO())


def _store(config: JobConfig) -> ArtifactStore:
    return ArtifactStore(config.data.output_dir, config.data.input_dir, "job")


def test_stage_writes_corrected_rule_based_and_report(stub_job: StubJob) -> None:
    scripted = ScriptedRegistry()
    config = _job(stub_job, correction_on=True)
    summary = _run(config)
    assert len(summary.done) == 2 and not summary.failed
    store = _store(config)
    for file_id in summary.done:
        report = store.read(file_id, Stage.ASSIGN_SPEAKERS, SpeakerCorrection, "correction")
        assert report.enabled and report.source_mode is INT and report.applied == 1
        for mode in (INT, VERB):
            final = store.read(file_id, Stage.ASSIGN_SPEAKERS, Transcript, mode.value)
            rule = store.read(
                file_id, Stage.ASSIGN_SPEAKERS, Transcript, f"{mode.value}.rule_based"
            )
            assert _texts(final) == _texts(rule)
            assert (
                sum(a != b for a, b in zip(_labels(final), _labels(rule), strict=True))
                == report.words_changed[mode]
            )
            assert store.artifact_path(
                file_id, Stage.ASSIGN_SPEAKERS, f"{mode.value}.rule_based", suffix=".txt"
            ).is_file()
        assert report.words_changed[INT] >= 1
    assert len(scripted.created) == 1 and scripted.created[0].closed == 1  # one LLM per job


def test_correction_off_never_creates_the_llm(stub_job: StubJob) -> None:
    scripted = ScriptedRegistry()
    config = _job(stub_job, correction_on=False)
    summary = _run(config)
    assert scripted.created == []
    store = _store(config)
    for file_id in summary.done:
        report = store.read(file_id, Stage.ASSIGN_SPEAKERS, SpeakerCorrection, "correction")
        assert not report.enabled and report.n_chunks == 0
        final = store.read(file_id, Stage.ASSIGN_SPEAKERS, Transcript, INT.value)
        rule = store.read(file_id, Stage.ASSIGN_SPEAKERS, Transcript, f"{INT.value}.rule_based")
        assert final == rule
    stages_through_assignment = [s for s in Stage if s is not Stage.DETECT_PHI]
    assert BackendKind.LLM not in backend_kinds(stages_through_assignment, config)
    assert BackendKind.LLM in backend_kinds([Stage.DETECT_PHI], config)  # P3-02


def test_correction_on_requires_the_llm_backend(stub_job: StubJob) -> None:
    config = _job(stub_job, correction_on=True)
    assert BackendKind.LLM in backend_kinds([Stage.ASSIGN_SPEAKERS], config)
    assert BackendKind.LLM not in backend_kinds([Stage.DIARIZE, Stage.TRANSCRIBE], config)


def test_toggling_correction_reruns_only_assignment_and_later(stub_job: StubJob) -> None:
    ScriptedRegistry()
    off = _job(stub_job, correction_on=False)
    _run(off)
    store = _store(off)
    file_id = _run(off).skipped[0]
    before = {p.name: p.stat().st_mtime_ns for p in store.file_dir(file_id).iterdir()}

    on = off.model_copy(
        update={"diarization": off.diarization.model_copy(update={"llm_correction": True})}
    )
    summary = _run(on)
    assert len(summary.done) == 2
    after = {p.name: p.stat().st_mtime_ns for p in store.file_dir(file_id).iterdir()}
    rewritten = {name.split(".")[0] for name, t in after.items() if before.get(name) != t}
    assert rewritten <= {"assign_speakers", "status"}  # diarize and transcribe untouched
    assert "assign_speakers" in rewritten
    status = store.read_status(file_id)
    assert status is not None and status.stages[Stage.DIARIZE] is StageStatus.DONE


def test_editing_the_prompt_changes_the_assignment_hash(
    stub_job: StubJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _job(stub_job, correction_on=True)
    first = stage_hashes(config)
    monkeypatch.setattr("murmurs.stages.prompt_hash", lambda: "edited")
    second = stage_hashes(config)
    assert first[Stage.ASSIGN_SPEAKERS] != second[Stage.ASSIGN_SPEAKERS]
    assert first[Stage.TRANSCRIBE] == second[Stage.TRANSCRIBE]
    off = config.model_copy(
        update={"diarization": config.diarization.model_copy(update={"llm_correction": False})}
    )
    assert stage_hashes(off)[Stage.ASSIGN_SPEAKERS] != first[Stage.ASSIGN_SPEAKERS]


def test_sentinel_never_reaches_logs_metrics_or_manifest(stub_job: StubJob) -> None:
    ScriptedRegistry()
    config = _job(stub_job, correction_on=True)
    console = io.StringIO()
    summary = _run(config, console, text=f"hello there {SENTINEL} said the participant")
    assert summary.done and not summary.failed
    store = _store(config)
    report = store.read(summary.done[0], Stage.ASSIGN_SPEAKERS, SpeakerCorrection, "correction")
    assert report.applied >= 1  # the scripted LLM ran over the sentinel text
    final = store.read(summary.done[0], Stage.ASSIGN_SPEAKERS, Transcript, INT.value)
    assert any(SENTINEL in s.text for s in final.segments)
    logs = "".join(p.read_text() for p in (store.job_dir / "logs").iterdir())
    for text in (
        console.getvalue(),
        logs,
        (store.job_dir / "metrics.json").read_text(),
        store.manifest_path.read_text(),
    ):
        assert SENTINEL not in text


def test_label_change_schema_is_strict() -> None:
    with pytest.raises(ValidationError):
        LabelChange.model_validate(
            {"first_word": 1, "last_word": 1, "text": "x", "speaker": A, "new_text": "y"}
        )
    assert SpeakerCorrections.model_validate({"changes": []}).changes == []


def test_prompt_hash_tracks_the_prompt_files() -> None:
    assert correction.prompt_hash() == correction.prompt_hash()
    assert len(correction.prompt_hash()) == 16


# --- concurrency and splitting (LLM speed-ups) ------------------------------------------------


def _own_line_count(prompt: str) -> int:
    body = prompt.split("never change words on CONTEXT lines.", 1)[1].split("List the label", 1)[0]
    return sum(1 for line in body.splitlines() if line.startswith("SPEAKER_"))


def test_truncated_chunks_are_split_and_every_word_is_still_covered() -> None:
    transcript = _long(24)
    owned: list[int] = []

    def respond(prompt: str) -> Any:
        if _own_line_count(prompt) > 2:
            return PipelineError("LLM_OUTPUT_TRUNCATED")
        owned.extend(int(n) for n in re.findall(r"^SPEAKER_\d+: \[(\d+)\]", prompt, re.M))
        return _changes()

    _, _, counts = correct_transcript(transcript, ScriptedLLM(respond), "f1", 1024)
    assert counts.splits > 0 and counts.failed_chunks == 0
    first_words = sorted(owned)
    assert first_words == sorted(set(first_words))  # no line was answered twice


def test_concurrency_does_not_change_the_result() -> None:
    transcript = _long(30)

    def respond(prompt: str) -> Any:
        body = prompt.split("never change words on CONTEXT lines.", 1)[1]
        for line in body.splitlines():
            if line.startswith(f"{B}: "):
                number, text = re.match(r"\[(\d+)\](\S+)", line[len(B) + 2 :]).groups()  # type: ignore[union-attr]
                return _changes((int(number), int(number), text, A))
        return _changes()

    one, changed_one, _ = correct_transcript(
        transcript, ScriptedLLM(respond), "f1", 1024, max_concurrency=1
    )
    many, changed_many, _ = correct_transcript(
        transcript, ScriptedLLM(respond), "f1", 1024, max_concurrency=8
    )
    assert one == many and changed_one == changed_many and changed_one


def test_report_records_splits_and_llm_usage() -> None:
    from murmurs.backends.stubs import StubLLMClient

    llm = StubLLMClient(response={"changes": []})
    _, report = correct_modes({INT: _three()}, llm, "f1", 32768, max_concurrency=4)
    assert report.schema_version == 2 and report.chunk_splits == 0
    assert report.llm.requests == report.n_chunks == 1
