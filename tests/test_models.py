"""P1-04: data model."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from murmurs.config import Stage, load_config
from murmurs.models import (
    TOP_LEVEL_ARTIFACTS,
    AuditAction,
    AuditEvent,
    DetectorSource,
    Diarization,
    ErrorInfo,
    FileStatus,
    RedactionLayer,
    RedactionSpan,
    ReviewState,
    RunManifest,
    Segment,
    SpanStatus,
    SpeakerTurn,
    StageStatus,
    Transcript,
    TranscriptMode,
    TranscriptRef,
    ValidationResult,
    Word,
    artifact_json_schemas,
)

SENTINEL = "SENTINEL_PHI_7f3a"
EXAMPLE = Path(__file__).parent.parent / "examples" / "job.yaml"
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def _span(**changes: object) -> RedactionSpan:
    fields: dict[str, object] = {
        "id": "s1",
        "start": 10,
        "end": 15,
        "entity_type": "PERSON",
        "tag": "[PERSON_1]",
        "confidence": 0.9,
        "sources": [DetectorSource.NER, DetectorSource.LLM],
    }
    return RedactionSpan.model_validate(fields | changes)


def _transcript() -> Transcript:
    words = [
        Word(text="hello", start=0.0, end=0.4, speaker="SPEAKER_00"),
        Word(text=SENTINEL, start=0.5, end=1.0, speaker="SPEAKER_00"),
    ]
    segment = Segment(
        speaker="SPEAKER_00", start=0.0, end=1.0, text=f"hello {SENTINEL}", words=words
    )
    return Transcript(file_id="f1", mode=TranscriptMode.INTENDED, segments=[segment])


def _examples() -> list[BaseModel]:
    """One populated instance of every top-level artifact."""
    status = FileStatus(file_id="f1", input_hash="abc")
    failed = FileStatus(
        file_id="f2",
        input_hash="def",
        stages=dict.fromkeys(Stage, StageStatus.PENDING) | {Stage.TRANSCRIBE: StageStatus.FAILED},
        failed_stage=Stage.TRANSCRIBE,
        error=ErrorInfo(code="STUB_FAILURE", message="stub configured to fail"),
    )
    return [
        ValidationResult(
            file_id="f1",
            path="in/a.wav",
            format="wav",
            duration_s=3.5,
            sample_rate=16000,
            channels=1,
            valid=True,
        ),
        ValidationResult(
            file_id="f2", path="in/b.wav", valid=False, error=ErrorInfo(code="UNREADABLE")
        ),
        Diarization(file_id="f1", turns=[SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=1.0)]),
        _transcript(),
        RedactionLayer(
            transcript=TranscriptRef(file_id="f1", mode=TranscriptMode.INTENDED),
            spans=[_span(status=SpanStatus.APPROVED, reviewer="rev1", timestamp=NOW)],
            review_state=ReviewState.IN_PROGRESS,
            audit=[
                AuditEvent(
                    span_id="s1",
                    action=AuditAction.APPROVED,
                    actor="rev1",
                    timestamp=NOW,
                    before=SpanStatus.PROPOSED,
                    after=SpanStatus.APPROVED,
                )
            ],
        ),
        status,
        failed,
        RunManifest(
            job_id="job1",
            started_at=NOW,
            finished_at=NOW,
            config=load_config(EXAMPLE),
            package_version="0.1.0",
            dependency_versions={"pydantic": "2.0"},
            model_versions={"transcriber": "stub"},
            code_version="857cffe",
            input_hashes={"f1": "abc", "f2": "def"},
            file_statuses=[status, failed],
        ),
    ]


@pytest.mark.parametrize("artifact", _examples(), ids=lambda a: type(a).__name__)
def test_artifacts_round_trip_through_json(artifact: BaseModel) -> None:
    restored = type(artifact).model_validate_json(artifact.model_dump_json())
    assert restored == artifact


def test_examples_cover_every_top_level_artifact() -> None:
    assert {type(a) for a in _examples()} == set(TOP_LEVEL_ARTIFACTS)


@pytest.mark.parametrize("model", TOP_LEVEL_ARTIFACTS, ids=lambda m: m.__name__)
def test_top_level_artifacts_carry_schema_version(model: type[BaseModel]) -> None:
    assert model.model_fields["schema_version"].default == 1


def test_json_schemas_export_for_all_artifacts() -> None:
    schemas = artifact_json_schemas()
    assert set(schemas) == {m.__name__ for m in TOP_LEVEL_ARTIFACTS}
    assert all(
        s["type"] == "object" and "schema_version" in s["properties"] for s in schemas.values()
    )


@pytest.mark.parametrize(("start", "end"), [(15, 15), (20, 10)])
def test_span_rejects_start_not_before_end(start: int, end: int) -> None:
    with pytest.raises(ValidationError, match="start must be < end"):
        _span(start=start, end=end)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_span_rejects_confidence_outside_unit_range(confidence: float) -> None:
    with pytest.raises(ValidationError):
        _span(confidence=confidence)


def test_span_rejects_empty_or_repeated_sources() -> None:
    with pytest.raises(ValidationError):
        _span(sources=[])
    with pytest.raises(ValidationError, match="sources must not repeat"):
        _span(sources=["ner", "ner"])


@pytest.mark.parametrize(("start", "end"), [(1.5, 2.2), (0.5, 1.5), (3.0, 3.5)])
def test_segment_rejects_words_outside_its_range(start: float, end: float) -> None:
    word = Word(text="w", start=start, end=end)
    with pytest.raises(ValidationError, match="word 0 lies outside"):
        Segment(start=1.0, end=2.0, text="w", words=[word])


def test_segment_allows_words_on_its_boundaries() -> None:
    words = [Word(text="a", start=0.0, end=0.5), Word(text="b", start=0.5, end=1.0 + 1e-9)]
    assert len(Segment(start=0.0, end=1.0, text="a b", words=words).words) == 2


def test_word_and_turn_reject_reversed_times() -> None:
    with pytest.raises(ValidationError, match="end must be >= start"):
        Word(text="w", start=2.0, end=1.0)
    with pytest.raises(ValidationError, match="start must be >= 0"):
        SpeakerTurn(speaker="S", start=-1.0, end=1.0)


def test_transcript_is_immutable() -> None:
    transcript = _transcript()
    with pytest.raises(ValidationError):
        transcript.file_id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        transcript.segments[0].words[0].text = "changed"  # type: ignore[misc]


def test_validation_result_error_must_match_valid_flag() -> None:
    with pytest.raises(ValidationError, match="invalid file must have an error"):
        ValidationResult(file_id="f", path="p", valid=False)
    with pytest.raises(ValidationError, match="valid file must not have an error"):
        ValidationResult(file_id="f", path="p", valid=True, error=ErrorInfo(code="EMPTY"))


def test_error_info_requires_code_format_and_short_message() -> None:
    with pytest.raises(ValidationError):
        ErrorInfo(code="not a code")
    with pytest.raises(ValidationError):
        ErrorInfo(code="TOO_LONG", message="x" * 201)


def test_file_status_defaults_every_stage_to_pending() -> None:
    status = FileStatus(file_id="f", input_hash="h")
    assert status.stages == dict.fromkeys(Stage, StageStatus.PENDING)


def test_file_status_failure_fields_must_agree() -> None:
    failed_stages = dict.fromkeys(Stage, StageStatus.PENDING) | {Stage.DIARIZE: StageStatus.FAILED}
    with pytest.raises(ValidationError, match="failed_stage must name"):
        FileStatus(file_id="f", input_hash="h", stages=failed_stages)
    with pytest.raises(ValidationError, match="error must be set"):
        FileStatus(file_id="f", input_hash="h", stages=failed_stages, failed_stage=Stage.DIARIZE)


def test_redaction_layer_checks_span_ids() -> None:
    ref = TranscriptRef(file_id="f", mode=TranscriptMode.VERBATIM)
    with pytest.raises(ValidationError, match="span ids must be unique"):
        RedactionLayer(transcript=ref, spans=[_span(), _span()])
    event = AuditEvent(span_id="missing", action=AuditAction.REJECTED, actor="r", timestamp=NOW)
    with pytest.raises(ValidationError, match="audit event 0 refers to an unknown span"):
        RedactionLayer(transcript=ref, spans=[_span()], audit=[event])


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(span_id="s1", action="approved", actor="r", timestamp=datetime(2026, 1, 1))


def test_validation_errors_never_quote_transcript_text() -> None:
    bad_word = Word.model_construct(text=SENTINEL, start=5.0, end=6.0)
    with pytest.raises(ValidationError) as info:
        Segment(start=0.0, end=1.0, text=f"said {SENTINEL}", words=[bad_word])
    assert SENTINEL not in str(info.value)

    with pytest.raises(ValidationError) as info:
        Word.model_validate({"text": SENTINEL, "start": SENTINEL, "end": 1.0})
    assert SENTINEL not in str(info.value)

    with pytest.raises(ValidationError) as info:
        Transcript.model_validate_json(
            f'{{"file_id": "f", "mode": "{SENTINEL}", "segments": [{{"text": "{SENTINEL}"}}]}}'
        )
    assert SENTINEL not in str(info.value)
