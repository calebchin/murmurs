"""P3-02 and P3-03: LLM detection, NER mapping, merging, tags, the detect_phi stage."""

import io
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from conftest import SENTINEL, StubJob
from fixtures.audio import tone_wav_bytes

from murmurs.backends import M, registry
from murmurs.backends.registry import BackendKind
from murmurs.backends.stubs import StubNERDetector, StubTranscriber
from murmurs.config import (
    PROVISIONAL_ENTITY_TYPES,
    EntityAction,
    EntityType,
    JobConfig,
    Stage,
    load_config,
)
from murmurs.errors import LLM_INVALID_OUTPUT, PipelineError
from murmurs.models import (
    DeidReport,
    DetectorSource,
    RedactionLayer,
    RedactionSpan,
    Segment,
    Transcript,
    TranscriptMode,
    Word,
)
from murmurs.runner import run_job
from murmurs.stages.deid import (
    Candidate,
    chunk_lines,
    deidentify,
    lines_of,
    llm_candidates,
    merge,
    ner_candidates,
    render_annotated,
    render_clean,
)
from murmurs.store import ArtifactStore, canonical_text

TYPES = list(PROVISIONAL_ENTITY_TYPES)
LLM, NER = DetectorSource.LLM, DetectorSource.NER


def _transcript(*lines: tuple[str, str]) -> Transcript:
    segments, t = [], 0.0
    for speaker, text in lines:
        words = []
        for token in text.split(" "):
            words.append(Word(text=token, start=t, end=t + 0.2, speaker=speaker))
            t += 0.3
        segments.append(
            Segment(
                speaker=speaker, start=words[0].start, end=words[-1].end, text=text, words=words
            )
        )
        t += 1.0
    return Transcript(file_id="f1", mode=TranscriptMode.INTENDED, segments=segments)


INTERVIEW = _transcript(
    ("SPEAKER_00", "Who looked after you?"),
    ("SPEAKER_01", "My aunt Maria, at Lakeview Elementary in Millbrook."),
    ("SPEAKER_00", "And Maria was strict?"),
    ("SPEAKER_01", "Yes. In 1999 Maria moved, when I was nine."),
)


class ScriptedLLM:
    """An ``LLMClient`` whose replies come from a function of the user prompt."""

    model_version = "scripted"

    def __init__(self, respond: Callable[[str], Any]) -> None:
        self.respond = respond
        self.prompts: list[str] = []
        self.closed = 0

    def generate(
        self, prompt: str, response_model: type[M], file_id: str, system: str | None = None
    ) -> M:
        self.prompts.append(prompt)
        reply = self.respond(prompt)
        if isinstance(reply, Exception):
            raise reply
        return response_model.model_validate(reply)

    def close(self) -> None:
        self.closed += 1


def _entities(*items: tuple[int, str, str]) -> dict[str, Any]:
    return {"entities": [{"line": n, "text": t, "entity_type": e} for n, t, e in items]}


def _own_lines(prompt: str) -> list[tuple[int, str]]:
    """The editable ``[n] SPEAKER: text`` lines of a deid prompt."""
    body = prompt.split("never report entities from CONTEXT lines.", 1)[1]
    return [
        (int(m.group(1)), m.group(2)) for m in re.finditer(r"^\[(\d+)\] \S+: (.*)$", body, re.M)
    ]


def finder(surfaces: dict[str, str]) -> Callable[[str], dict[str, Any]]:
    """A scripted LLM that reports each configured surface in every editable line."""

    def respond(prompt: str) -> dict[str, Any]:
        found = []
        for number, text in _own_lines(prompt):
            for surface, entity_type in surfaces.items():
                if surface in text:
                    found.append((number, surface, entity_type))
        return _entities(*found)

    return respond


def _texts(text: str, spans: list[Any]) -> list[tuple[str, str]]:
    return [(text[s.start : s.end], s.entity_type) for s in spans]


# --- lines and chunks ------------------------------------------------------------------------


def test_lines_point_into_the_canonical_text() -> None:
    text = canonical_text(INTERVIEW)
    assert [text[ln.start : ln.start + len(ln.text)] for ln in lines_of(INTERVIEW)] == text.split(
        "\n"
    )


def test_chunks_cover_every_line_once() -> None:
    long = _transcript(
        *[("SPEAKER_00" if i % 2 else "SPEAKER_01", f"line {i} " + "word " * 40) for i in range(60)]
    )
    chunks = chunk_lines(lines_of(long), max_model_len=2048)
    assert len(chunks) > 2
    assert [ln.number for c in chunks for ln in c.own] == list(range(60))
    assert chunks[1].before and chunks[1].after


# --- LLM detection ---------------------------------------------------------------------------


def test_llm_entities_become_spans_at_every_occurrence() -> None:
    llm = ScriptedLLM(
        lambda p: _entities((1, "Maria", "PERSON"), (1, "Lakeview Elementary", "ORGANIZATION"))
    )
    spans, counts = llm_candidates(INTERVIEW, llm, "f1", TYPES, 32768)
    text = canonical_text(INTERVIEW)
    assert _texts(text, spans).count(("Maria", "PERSON")) == 3  # reported once, found 3 times
    assert ("Lakeview Elementary", "ORGANIZATION") in _texts(text, spans)
    assert counts.propagated == 2 and counts.proposed == 2 and not counts.rejected
    assert all(s.source is LLM and s.confidence == 0.7 for s in spans)


@pytest.mark.parametrize(
    ("entity", "reason"),
    [
        ((9, "Maria", "PERSON"), "missing_line"),
        ((1, "Maria", "PET"), "unknown_type"),
        ((1, "Rosalind", "PERSON"), "not_found"),
        ((1, "Mari", "PERSON"), "not_found"),  # not a whole word
    ],
)
def test_invalid_llm_entities_are_rejected(entity: tuple[int, str, str], reason: str) -> None:
    spans, counts = llm_candidates(
        INTERVIEW, ScriptedLLM(lambda p: _entities(entity)), "f1", TYPES, 32768
    )
    assert spans == [] and counts.rejected == {reason: 1}


def test_entities_from_context_lines_are_rejected() -> None:
    long = _transcript(
        *[("SPEAKER_01", f"Maria said thing number {i} " + "word " * 40) for i in range(40)]
    )
    calls = {"n": 0}

    def respond(prompt: str) -> dict[str, Any]:
        calls["n"] += 1
        own = {n for n, _ in _own_lines(prompt)}
        context_line = max(own) + 1  # shown as context after this chunk
        return _entities((context_line, "Maria", "PERSON")) if calls["n"] == 1 else _entities()

    _, counts = llm_candidates(long, ScriptedLLM(respond), "f1", TYPES, 2048)
    assert counts.rejected == {"outside_chunk": 1}


def test_ignored_types_are_not_offered_to_the_llm() -> None:
    types = [
        *TYPES,
        EntityType(name="TIME", tag="[TIME_{n}]", description="times", action=EntityAction.IGNORE),
    ]
    llm = ScriptedLLM(lambda p: _entities())
    llm_candidates(INTERVIEW, llm, "f1", types, 32768)
    assert "- PERSON:" in llm.prompts[0] and "- TIME:" not in llm.prompts[0]


def test_invalid_chunk_output_is_skipped_and_failed_requests_fail() -> None:
    _, counts = llm_candidates(
        INTERVIEW, ScriptedLLM(lambda p: PipelineError(LLM_INVALID_OUTPUT)), "f1", TYPES, 32768
    )
    assert counts.failed_chunks == 1
    with pytest.raises(PipelineError, match="LLM_REQUEST_FAILED"):
        llm_candidates(
            INTERVIEW,
            ScriptedLLM(lambda p: PipelineError("LLM_REQUEST_FAILED")),
            "f1",
            TYPES,
            32768,
        )


# --- NER mapping and merging -----------------------------------------------------------------


def _ner_span(start: int, end: int, label: str, confidence: float = 0.9) -> RedactionSpan:
    return RedactionSpan(
        id="n",
        start=start,
        end=end,
        entity_type=label,
        tag="[X]",
        confidence=confidence,
        sources=[NER],
    )


def test_ner_labels_map_through_the_tag_set() -> None:
    spans = [
        _ner_span(0, 4, "first_name"),
        _ner_span(5, 9, "date"),
        _ner_span(10, 14, "language"),
        _ner_span(15, 19, "cvv"),
    ]
    kept, ignored = ner_candidates(spans, TYPES)
    assert [(c.entity_type, c.source) for c in kept] == [("PERSON", NER), ("DATE", NER)]
    assert ignored == 2  # unmapped labels are ignored


def test_merge_unions_overlaps_prefers_llm_type_and_scores_agreement() -> None:
    merged = merge(
        [
            Candidate(0, 9, "LOCATION", NER, 0.6),  # 'Lindqvist' as a city
            Candidate(0, 17, "ORGANIZATION", LLM, 0.7),  # 'Lindqvist Freight'
            Candidate(20, 25, "PERSON", NER, 0.4),
            Candidate(30, 35, "PERSON", LLM, 0.7),
        ],
        TYPES,
    )
    assert merged[0] == (0, 17, "ORGANIZATION", [LLM, NER], 1.0)
    assert merged[1] == (20, 25, "PERSON", [NER], 0.4)
    assert merged[2] == (30, 35, "PERSON", [LLM], 0.7)


# --- the whole detection, tags and renderings ------------------------------------------------


def _deid(
    ner_targets: dict[str, str], llm_surfaces: dict[str, str], threshold: float = 0.5
) -> tuple[RedactionLayer, DeidReport]:
    return deidentify(
        INTERVIEW,
        StubNERDetector(ner_targets),
        ScriptedLLM(finder(llm_surfaces)),
        TYPES,
        "f1",
        32768,
        threshold,
    )


def test_same_surface_gets_the_same_tag_everywhere() -> None:
    layer, report = _deid(
        {"Maria": "first_name"},
        {"Maria": "PERSON", "Millbrook": "LOCATION", "1999": "DATE", "nine": "AGE"},
    )
    tags = {(s.tag, s.entity_type) for s in layer.spans}
    assert ("[PERSON_1]", "PERSON") in tags and ("[LOCATION_1]", "LOCATION") in tags
    maria = [s for s in layer.spans if s.entity_type == "PERSON"]
    assert len(maria) == 3 and {s.tag for s in maria} == {"[PERSON_1]"}
    assert all(s.confidence == 1.0 and s.sources == [LLM, NER] for s in maria)
    assert report.both_detectors == 3 and report.by_type["PERSON"] == 3


def test_flagged_types_are_marked_and_not_redacted() -> None:
    layer, report = _deid({}, {"Maria": "PERSON", "1999": "DATE", "nine": "AGE"})
    clean = render_clean(INTERVIEW, layer, TYPES, 0.5)
    assert "1999" in clean and "nine" in clean  # flagged: kept for review
    assert "Maria" not in clean and "[PERSON_1]" in clean
    annotated = render_annotated(INTERVIEW, layer)
    assert "{{FLAG DATE_1|1999}}" in annotated and "{{PERSON_1|Maria}}" in annotated
    assert report.flagged == 2 and report.redacted == 3


def test_threshold_keeps_low_confidence_spans_in_the_clean_text() -> None:
    ner = StubNERDetector({"Millbrook": "city"})
    layer, _ = deidentify(INTERVIEW, ner, ScriptedLLM(finder({})), TYPES, "f1", 32768, 0.5)
    low = layer.model_copy(
        update={"spans": [s.model_copy(update={"confidence": 0.3}) for s in layer.spans]}
    )
    assert "Millbrook" in render_clean(INTERVIEW, low, TYPES, 0.5)
    assert "Millbrook" not in render_clean(INTERVIEW, layer, TYPES, 0.5)


def test_clean_text_differs_only_inside_redacted_spans() -> None:
    layer, _ = _deid(
        {"Maria": "first_name"}, {"Lakeview Elementary": "ORGANIZATION", "Millbrook": "LOCATION"}
    )
    clean_lines = render_clean(INTERVIEW, layer, TYPES, 0.5).splitlines()
    for line, segment in zip(clean_lines, INTERVIEW.segments, strict=True):
        body = line.split(": ", 1)[1]
        pattern = re.escape(segment.text)
        for surface in ("Maria", "Lakeview Elementary", "Millbrook"):
            pattern = pattern.replace(re.escape(surface), r"\[[A-Z]+_\d+\]")
        assert re.fullmatch(pattern, body), body
    assert canonical_text(INTERVIEW).count("Maria") == 3  # the transcript is unchanged


# --- the stage, through the runner -----------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    yield


def _job(
    stub_job: StubJob, text: str, correction: bool = False
) -> tuple[JobConfig, list[ScriptedLLM]]:
    created: list[ScriptedLLM] = []

    def make_llm(config: JobConfig) -> ScriptedLLM:
        def respond(prompt: str) -> Any:
            if "never report entities from CONTEXT lines." in prompt:
                return finder({SENTINEL: "PERSON", "Millbrook": "LOCATION"})(prompt)
            return {"changes": []}  # speaker correction: no changes

        llm = ScriptedLLM(respond)
        created.append(llm)
        return llm

    registry.register(BackendKind.LLM, "scripted", make_llm)
    registry.register(BackendKind.TRANSCRIBER, "stub", lambda _: StubTranscriber(text=text))
    registry.register(BackendKind.NER, "stub", lambda _: StubNERDetector({SENTINEL: "first_name"}))
    files = {
        "a.wav": tone_wav_bytes(seconds=3.0, frequency=440),
        "b.wav": tone_wav_bytes(seconds=3.0, frequency=660),
    }
    config = load_config(
        stub_job(
            files,
            diarization={"llm_correction": correction},
            deid={
                "llm_model": "scripted",
                "entity_types": [e.model_dump(mode="json") for e in TYPES],
            },
        )
    )
    return config, created


@pytest.mark.parametrize("correction", [False, True])
def test_stage_writes_layer_renderings_and_report(stub_job: StubJob, correction: bool) -> None:
    config, created = _job(stub_job, f"my aunt {SENTINEL} lived in Millbrook", correction)
    console = io.StringIO()
    summary = run_job(config, "job", console=console)
    assert len(summary.done) == 2 and not summary.failed
    assert len(created) == 1 and created[0].closed == 1  # one LLM for the whole job
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    for file_id in summary.done:
        names = {
            p.name for p in store.file_dir(file_id).iterdir() if p.name.startswith("detect_phi")
        }
        assert names == {
            "detect_phi.intended.json",
            "detect_phi.intended.annotated.txt",
            "detect_phi.intended.clean.txt",
            "detect_phi.report.json",
        }
        layer = store.read(file_id, Stage.DETECT_PHI, RedactionLayer, "intended")
        assert layer.transcript.mode is TranscriptMode.INTENDED and layer.spans
        clean = store.artifact_path(file_id, Stage.DETECT_PHI, "intended.clean", ".txt").read_text()
        assert SENTINEL not in clean and "[PERSON_1]" in clean and "[LOCATION_1]" in clean
        annotated = store.artifact_path(
            file_id, Stage.DETECT_PHI, "intended.annotated", ".txt"
        ).read_text()
        assert f"{{{{PERSON_1|{SENTINEL}}}}}" in annotated
        report = store.read(file_id, Stage.DETECT_PHI, DeidReport, "report")
        assert report.both_detectors >= 1 and report.redacted >= 2
    logs = "".join(p.read_text() for p in (store.job_dir / "logs").iterdir())
    for text in (
        console.getvalue(),
        logs,
        (store.job_dir / "metrics.json").read_text(),
        store.manifest_path.read_text(),
    ):
        assert SENTINEL not in text


def test_editing_the_deid_prompt_reruns_only_detect_phi(
    stub_job: StubJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    from murmurs.runner import stage_hashes

    config, _ = _job(stub_job, "hello there")
    before = stage_hashes(config)
    monkeypatch.setattr("murmurs.stages.deid_prompt_hash", lambda: "edited")
    after = stage_hashes(config)
    assert before[Stage.DETECT_PHI] != after[Stage.DETECT_PHI]
    assert before[Stage.ASSIGN_SPEAKERS] == after[Stage.ASSIGN_SPEAKERS]
