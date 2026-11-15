"""Importing human-deidentified transcripts as an evaluation set (``murmurs eval import``)."""

import hashlib
import json
from pathlib import Path

import pytest
from conftest import SENTINEL
from typer.testing import CliRunner

from murmurs.cli import app
from murmurs.config import PROVISIONAL_ENTITY_TYPES
from murmurs.evalimport import (
    IGNORE,
    EvalImportError,
    Turn,
    align,
    interview_type_of,
    load_mapping,
    normalize_tag,
    pair_files,
    parse_turns,
)
from murmurs.evalset import load_dir, parse_interview

TYPES = [e.name for e in PROVISIONAL_ENTITY_TYPES]

RAW = f"""Interview transcript
Speaker 0: Thanks for coming in. Where did you grow up?
Speaker 1: In Millbrook, with my aunt Rosa Mendez and my brother {SENTINEL}.
Speaker 0: And school?
speaker 1: I went to St. Ansgar's School [inaudible] until I was ten.
It was Rosa's idea.
"""
DEID = """Interview transcript
Speaker 0: Thanks for coming in. Where did you grow up?
Speaker 1: In [LOCATION], with my aunt [first name] [ LAST NAME ] and my brother [NAME OF BROTHER].
Speaker 0: And school?
Speaker 1: I went to [SCHOOL] [inaudible] until I was [AGE].
It was [NAME]'s idea.
"""


def _turns(*texts: str) -> list[Turn]:
    return [Turn(i % 2, text) for i, text in enumerate(texts)]


def _surfaces(raw: list[Turn], deid: list[Turn]) -> list[tuple[str, str]]:
    return [(raw[s.turn].text[s.start : s.end], s.tag) for s in align(raw, deid).spans]


def test_parse_turns_joins_continuations_and_skips_the_preamble() -> None:
    turns, skipped = parse_turns(RAW)
    assert skipped == 1
    assert [t.speaker for t in turns] == [0, 1, 0, 1]
    assert turns[3].text.endswith("until I was ten. It was Rosa's idea.")


def test_tags_normalise_and_file_names_give_the_interview_type() -> None:
    assert normalize_tag("[ participant  name ]") == "PARTICIPANT NAME"
    assert interview_type_of("p01_ceca_raw.txt") is not None
    assert interview_type_of("p01.txt") is None
    assert interview_type_of("ceca_leds.txt") is None


def test_one_tag_covers_every_word_it_replaced() -> None:
    raw = _turns("I lived with Rosa Mendez then.")
    deid = _turns("I lived with [NAME] then.")
    assert _surfaces(raw, deid) == [("Rosa Mendez", "NAME")]


def test_a_tag_before_a_possessive_keeps_the_suffix_out() -> None:
    assert _surfaces(_turns("It was Rosa's idea."), _turns("It was [NAME]'s idea.")) == [
        ("Rosa", "NAME")
    ]


def test_one_tag_per_word_when_the_counts_match() -> None:
    raw = _turns("my aunt Rosa Mendez came")
    deid = _turns("my aunt [FIRST NAME] [LAST NAME] came")
    assert _surfaces(raw, deid) == [("Rosa", "FIRST NAME"), ("Mendez", "LAST NAME")]


def test_shared_bracket_notes_are_text_not_tags() -> None:
    raw = _turns("Rosa [inaudible] said [laughs] yes")
    deid = _turns("[NAME] [inaudible] said [laughs] yes")
    result = align(raw, deid)
    assert set(result.tags) == {"NAME"}
    assert _surfaces(raw, deid) == [("Rosa", "NAME")]


def test_other_edits_are_counted_not_labelled() -> None:
    raw = _turns("I seen Rosa there", "and then some more words here")
    deid = _turns("I saw [NAME] there [PLACE]", "and then here")
    result = align(raw, deid)
    assert [(raw[s.turn].text[s.start : s.end]) for s in result.spans] == ["Rosa"]
    assert result.untagged_edits == 1  # seen -> saw, in the same diff block as the tag
    assert result.tags_without_text == 1  # [PLACE] added
    assert result.deleted == 1  # "some more words" removed


def test_a_tag_that_cannot_be_pinned_down_is_counted_unaligned() -> None:
    raw = _turns("with Rosa Maria Mendez today")
    deid = _turns("with [FIRST] [LAST] today")  # two tags for three words
    result = align(raw, deid)
    assert result.spans == [] and result.unaligned_tags == 2


def test_parser_accepts_speaker_labels_and_keeps_interviewer_mapping() -> None:
    evalset = parse_interview(
        "---\ninterview_type: LEDS\n---\nINTERVIEWER: hi\nSPEAKER_02: I am {{PERSON|Ann}}\n",
        "x",
    )
    assert [s.speaker for s in evalset.transcript.segments] == ["SPEAKER_00", "SPEAKER_02"]
    assert evalset.labels[0].text == "Ann"


# --- end to end through the CLI -----------------------------------------------------------


def _folders(tmp_path: Path) -> tuple[Path, Path]:
    raw, deid = tmp_path / "raw", tmp_path / "deid"
    for folder, text in ((raw, RAW), (deid, DEID)):
        (folder / "site_a").mkdir(parents=True)
        (folder / "site_a" / "p01_CECA.txt").write_text(text)
        (folder / "p02_nolabel.txt").write_text(text)
    (raw / "p03_LEDS.txt").write_text(RAW)  # no deidentified copy
    return raw, deid


def _digest(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        digest.update(str(path).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _fill(mapping: Path, values: dict[str, str]) -> None:
    lines = mapping.read_text().splitlines()
    out = []
    for line in lines:
        for tag, value in values.items():
            if line.startswith(f"{tag}: ''"):
                line = line.replace(f"{tag}: ''", f"{tag}: {value}", 1)
        out.append(line)
    mapping.write_text("\n".join(out) + "\n")


def test_tags_then_import_builds_a_valid_evaluation_set(tmp_path: Path) -> None:
    raw, deid = _folders(tmp_path)
    before = _digest(tmp_path / "raw"), _digest(tmp_path / "deid")
    mapping, out = tmp_path / "tags.yaml", tmp_path / "evalset"
    runner = CliRunner()

    result = runner.invoke(app, ["eval", "tags", str(raw), str(deid), str(mapping)])
    assert result.exit_code == 0, result.output
    assert "pairs=2  turns=8  tags=7" in result.stdout and "still blank=7" in result.stdout
    assert "warning" not in result.output
    assert set(load_mapping(mapping)) == {
        "LOCATION", "FIRST NAME", "LAST NAME", "NAME OF BROTHER", "SCHOOL", "AGE", "NAME",
    }  # fmt: skip

    result = runner.invoke(
        app, ["eval", "import", str(raw), str(deid), str(out), "-t", str(mapping)]
    )
    assert result.exit_code == 2 and "7 tags have no entity type" in result.stderr
    assert not out.exists()

    _fill(
        mapping,
        {
            "LOCATION": "LOCATION",
            "FIRST NAME": "PERSON",
            "LAST NAME": "PERSON",
            "NAME OF BROTHER": "PERSON",
            "SCHOOL": "ORGANIZATION",
            "AGE": IGNORE,
            "NAME": "PERSON",
        },
    )
    result = runner.invoke(
        app, ["eval", "import", str(raw), str(deid), str(out), "-t", str(mapping)]
    )
    assert result.exit_code == 0, result.output
    assert "p02_nolabel.txt  skipped: file name names neither CECA nor LEDS" in result.stdout
    assert "only in the raw folder (1): p03_LEDS.txt" in result.stdout
    assert "imported 1/2 pairs, labels=6" in result.stdout

    (evalset,) = load_dir(out, TYPES)
    assert evalset.set_id == "site_a__p01_CECA"
    assert [(lb.entity_type, lb.text) for lb in evalset.labels] == [
        ("LOCATION", "Millbrook"),
        ("PERSON", "Rosa"),
        ("PERSON", "Mendez"),
        ("PERSON", SENTINEL),
        ("ORGANIZATION", "St. Ansgar's School"),
        ("PERSON", "Rosa"),
    ]
    report = json.loads((out / "import_report.json").read_text())
    assert report["files"][0]["ignored_tags"] == 1  # [AGE]
    assert (_digest(tmp_path / "raw"), _digest(tmp_path / "deid")) == before  # inputs untouched

    for text in (SENTINEL, "Rosa", "Millbrook", "BROTHER"):
        assert text not in result.output
        assert text not in (out / "import_report.json").read_text()


def test_speaker_labels_on_their_own_line_with_text_below() -> None:
    source = "SPEAKER_1:\nWhere did you\ngrow up?\n\nSPEAKER_2:\nIn Millbrook.\n\nAround there.\n"
    turns, skipped = parse_turns(source)
    assert skipped == 0
    assert turns == [Turn(1, "Where did you grow up?"), Turn(2, "In Millbrook. Around there.")]
    deid, _ = parse_turns(source.replace("Millbrook", "[CITY]"))
    assert _surfaces(turns, deid) == [("Millbrook", "CITY")]


def test_tags_warns_about_pairs_without_turns(tmp_path: Path) -> None:
    raw, deid = tmp_path / "raw", tmp_path / "deid"
    raw.mkdir(), deid.mkdir()
    for folder in (raw, deid):
        (folder / "a_CECA.txt").write_text("SPEAKER_1:\nhello there\n")
        (folder / "b_LEDS.txt").write_text("Interviewer - hello there\n")
    result = CliRunner().invoke(
        app, ["eval", "tags", str(raw), str(deid), str(tmp_path / "t.yaml")]
    )
    assert result.exit_code == 0, result.output
    assert "pairs=2  turns=1" in result.stdout
    assert "warning: 1 pairs have a copy with no SPEAKER_N: turns: b_LEDS.txt" in result.stderr
    assert "hello" not in result.output


def test_tags_holding_text_never_reach_the_console(tmp_path: Path) -> None:
    raw, deid = tmp_path / "raw", tmp_path / "deid"
    raw.mkdir(), deid.mkdir()
    (raw / "a_CECA.txt").write_text("Speaker 1: my friend Ann came\n")
    (deid / "a_CECA.txt").write_text(f"Speaker 1: my friend [{SENTINEL}] came\n")
    result = CliRunner().invoke(
        app, ["eval", "tags", str(raw), str(deid), str(tmp_path / "t.yaml")]
    )
    assert result.exit_code == 0, result.output
    assert SENTINEL not in result.output and "Ann" not in result.output


def test_import_refuses_a_used_or_input_output_folder(tmp_path: Path) -> None:
    raw, deid = _folders(tmp_path)
    mapping = tmp_path / "tags.yaml"
    mapping.write_text("NAME: PERSON\n")
    pairing = pair_files(raw, deid)
    from murmurs.evalimport import import_pairs

    with pytest.raises(EvalImportError, match="outside the input folders"):
        import_pairs(pairing, load_mapping(mapping), raw / "out", [raw, deid])
    used = tmp_path / "used"
    used.mkdir()
    (used / "x.md").write_text("")
    with pytest.raises(EvalImportError, match="new or empty folder"):
        import_pairs(pairing, load_mapping(mapping), used, [raw, deid])


def test_mapping_to_an_unknown_type_is_refused(tmp_path: Path) -> None:
    raw, deid = _folders(tmp_path)
    mapping = tmp_path / "tags.yaml"
    mapping.write_text("NAME: PET\n")
    result = CliRunner().invoke(
        app, ["eval", "import", str(raw), str(deid), str(tmp_path / "o"), "-t", str(mapping)]
    )
    assert result.exit_code == 2 and "unknown entity types: PET" in result.stderr


# --- limiting the import to a list of finished transcripts -------------------------------


def _three_pairs(tmp_path: Path) -> tuple[Path, Path]:
    raw, deid = tmp_path / "raw", tmp_path / "deid"
    raw.mkdir(), deid.mkdir()
    for name in ("p01_CECA.txt", "p02_LEDS.txt", "p03_CECA.txt"):
        (raw / name).write_text("SPEAKER_1:\nI lived in Millbrook.\n")
    (deid / "p01_CECA.txt").write_text("SPEAKER_1:\nI lived in [CITY].\n")
    (deid / "p02_LEDS.txt").write_text("SPEAKER_1:\nI lived in [TOWN].\n")
    (deid / "p03_CECA.txt").write_text("SPEAKER_1:\nI lived in Millbrook.\n")  # not reviewed
    return raw, deid


def test_file_list_matches_names_loosely_and_skips_blank_cells(tmp_path: Path) -> None:
    listing = tmp_path / "done.csv"
    listing.write_text("﻿Annotated\n P01_ceca.txt \np02_LEDS\n\n", encoding="utf-8")
    from murmurs.evalimport import load_file_list, select_pairs

    names = load_file_list(listing, "Annotated")
    assert names == {"p01_ceca", "p02_leds"}
    raw, deid = _three_pairs(tmp_path)
    selected, not_found = select_pairs(pair_files(raw, deid), names | {"p09_ceca"})
    assert [str(p.relative) for p in selected.pairs] == ["p01_CECA.txt", "p02_LEDS.txt"]
    assert not_found == ["p09_ceca"]


def test_file_list_without_the_column_is_refused(tmp_path: Path) -> None:
    listing = tmp_path / "done.csv"
    listing.write_text("Done\np01_CECA\n")
    from murmurs.evalimport import load_file_list

    with pytest.raises(EvalImportError, match="no column named 'Annotated'"):
        load_file_list(listing, "Annotated")


def test_only_limits_tags_and_import_to_listed_pairs(tmp_path: Path) -> None:
    raw, deid = _three_pairs(tmp_path)
    listing = tmp_path / "done.csv"
    listing.write_text("Annotated\np01_CECA.txt\nP02_LEDS\np09_CECA\n")
    mapping, out = tmp_path / "tags.yaml", tmp_path / "evalset"
    runner = CliRunner()
    only = ["--only", str(listing)]

    result = runner.invoke(app, ["eval", "tags", str(raw), str(deid), str(mapping), *only])
    assert result.exit_code == 0, result.output
    assert "file list: 2 of 3 pairs selected" in result.stdout
    assert "warning: 1 listed names match no pair: p09_ceca" in result.stderr
    assert "pairs=2" in result.stdout and "tags=2" in result.stdout

    mapping.write_text("CITY: LOCATION\nTOWN: LOCATION\n")
    args = ["eval", "import", str(raw), str(deid), str(out), "-t", str(mapping), *only]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "imported 2/2 pairs, labels=2" in result.stdout
    assert sorted(p.name for p in out.glob("*.md")) == ["p01_CECA.md", "p02_LEDS.md"]
    report = json.loads((out / "import_report.json").read_text())
    assert report["not_found"] == ["p09_ceca"]
    assert "Millbrook" not in result.output
