"""MVP step 3: the synthetic evaluation set (reduced P2-09, D8, D9, D10)."""

import itertools
from pathlib import Path

import pytest
from conftest import SENTINEL
from typer.testing import CliRunner

from murmurs.cli import app
from murmurs.config import PROVISIONAL_ENTITY_TYPES, load_config
from murmurs.evalset import EvalSetError, label_counts, load_dir, parse_interview, validate
from murmurs.models import GroundTruthLabel, InterviewType
from murmurs.store import canonical_text

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
MVP_SET = Path(__file__).parent.parent / "eval" / "synthetic"
EXAMPLE = Path(__file__).parent.parent / "examples" / "job.yaml"
TYPES = [e.name for e in PROVISIONAL_ENTITY_TYPES]


def test_fixture_set_loads_and_validates() -> None:
    sets = load_dir(FIXTURES, TYPES)
    assert [s.set_id for s in sets] == ["tiny_ceca", "tiny_leds"]
    assert [s.interview_type for s in sets] == [InterviewType.CECA, InterviewType.LEDS]
    assert label_counts(sets[0]) == {
        "PERSON": 1,
        "LOCATION": 1,
        "AGE": 1,
        "ORGANIZATION": 1,
        "DATE": 1,
    }


def test_labels_point_into_the_canonical_text() -> None:
    ceca = load_dir(FIXTURES, TYPES)[0]
    text = canonical_text(ceca.transcript)
    assert [text[lb.start : lb.end] for lb in ceca.labels] == [lb.text for lb in ceca.labels]
    assert ceca.labels[0].text == "Rosa Mendez"  # whitespace inside a mark is normalised
    assert ceca.labels[2].note == "stated age"
    assert [lb.id for lb in ceca.labels] == [f"tiny_ceca-{i:03d}" for i in range(1, 6)]


def test_turns_become_speaker_segments_with_increasing_times() -> None:
    ceca = load_dir(FIXTURES, TYPES)[0]
    segments = ceca.transcript.segments
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
    words = [w for s in segments for w in s.words]
    assert all(a.end <= b.start for a, b in itertools.pairwise(words))
    assert all(" ".join(w.text for w in s.words) == s.text for s in segments)


def _broken(label_update: dict[str, object], index: int = 0):  # type: ignore[no-untyped-def]
    ceca = load_dir(FIXTURES, TYPES)[0]
    labels = list(ceca.labels)
    labels[index] = labels[index].model_copy(update=label_update)
    return ceca.model_copy(update={"labels": labels})


def test_mismatched_text_fails_naming_the_label_not_its_text() -> None:
    with pytest.raises(EvalSetError) as info:
        validate(_broken({"text": SENTINEL}), TYPES)
    assert "tiny_ceca-001" in str(info.value) and "does not match" in str(info.value)
    assert SENTINEL not in str(info.value) and "Rosa" not in str(info.value)


def test_unknown_entity_type_fails_naming_the_label() -> None:
    with pytest.raises(EvalSetError, match="tiny_ceca-002: entity type is not in the tag set"):
        validate(_broken({"entity_type": "PET_NAME"}, index=1), TYPES)


def test_overlapping_labels_fail_naming_both() -> None:
    ceca = load_dir(FIXTURES, TYPES)[0]
    first = ceca.labels[0]
    extra = GroundTruthLabel(
        id="tiny_ceca-099",
        start=first.start,
        end=first.start + 4,
        entity_type="PERSON",
        text="Rosa",
    )
    with pytest.raises(EvalSetError, match="overlaps") as info:
        validate(ceca.model_copy(update={"labels": [*ceca.labels, extra]}), TYPES)
    assert "tiny_ceca-001" in str(info.value) and "tiny_ceca-099" in str(info.value)
    assert "Rosa" not in str(info.value)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("INTERVIEWER: hi\n", "interview_type must be CECA or LEDS"),
        ("---\ninterview_type: XYZ\n---\nINTERVIEWER: hi\n", "interview_type must be CECA or LEDS"),
        ("---\ninterview_type: CECA\n---\nNARRATOR: hi\n", "line 4: must start with"),
        ("---\ninterview_type: CECA\n---\nPARTICIPANT: {{PERSON|Ann\n", "line 4: malformed"),
        ("---\ninterview_type: CECA\nINTERVIEWER: hi\n", "front matter is not closed"),
    ],
)
def test_malformed_files_fail_with_location_only(source: str, message: str) -> None:
    with pytest.raises(EvalSetError, match=message):
        parse_interview(source, "bad")


def test_eval_check_prints_counts_and_no_text() -> None:
    result = CliRunner().invoke(app, ["eval", "check", str(FIXTURES)])
    assert result.exit_code == 0, result.output
    assert "tiny_ceca  CECA  turns=4" in result.stdout and "labels=5" in result.stdout
    assert "total LEDS: labels=4" in result.stdout
    for text in (SENTINEL, "Rosa", "Millbrook", "555-0100"):
        assert text not in result.output


def test_eval_check_exits_2_on_an_invalid_set(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text(
        "---\ninterview_type: CECA\n---\nPARTICIPANT: {{PETNAME|Rex}}\n"
    )
    result = CliRunner().invoke(app, ["eval", "check", str(tmp_path)])
    assert result.exit_code == 2 and "bad-001: entity type is not in the tag set" in result.stderr
    assert "Rex" not in result.output


def test_example_config_holds_the_provisional_tag_set() -> None:
    assert list(load_config(EXAMPLE).deid.entity_types) == list(PROVISIONAL_ENTITY_TYPES)


def test_mvp_set_covers_both_interview_types_and_the_planned_entities() -> None:
    sets = load_dir(MVP_SET, TYPES)
    assert sorted(s.interview_type.value for s in sets) == ["CECA", "LEDS"]
    for evalset in sets:
        counts = label_counts(evalset)
        for entity in ("PERSON", "ORGANIZATION", "LOCATION", "DATE", "AGE", "CONTACT"):
            assert counts[entity] >= 1, (evalset.set_id, entity)
        notes = {lb.note for lb in evalset.labels}
        assert notes - {""}  # edge cases are annotated
