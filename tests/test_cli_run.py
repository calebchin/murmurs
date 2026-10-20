"""P1-03: CLI run command and overrides."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner, Result

from murmurs.cli import app
from murmurs.config import ConfigError, apply_override, parse_assignment, resolve_config

EXAMPLE = Path(__file__).parent.parent / "examples" / "job.yaml"


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, ["run", "--config", str(EXAMPLE), *args])


def _dry_run(*args: str) -> dict[str, Any]:
    result = _run("--dry-run", *args)
    assert result.exit_code == 0, result.output
    resolved = yaml.safe_load(result.stdout)
    assert isinstance(resolved, dict)
    return resolved


def test_end_flag_overrides_file() -> None:
    assert yaml.safe_load(EXAMPLE.read_text())["stages"]["end"] == "detect_phi"
    assert _dry_run("--end", "transcribe")["stages"]["end"] == "transcribe"


def test_set_parses_int_not_string() -> None:
    job = resolve_config(EXAMPLE, [parse_assignment("diarization.num_speakers=2")])
    assert job.diarization.num_speakers == 2
    assert _dry_run("--set", "diarization.num_speakers=2")["diarization"]["num_speakers"] == 2


def test_set_out_of_range_fails_validation() -> None:
    result = _run("--dry-run", "--set", "deid.redaction_threshold=2")
    assert result.exit_code == 2
    assert "deid.redaction_threshold" in result.stderr


def test_dry_run_reflects_every_override() -> None:
    resolved = _dry_run(
        "--input-dir", "/data/in",
        "--output-dir", "/data/out",
        "--start", "diarize",
        "--end", "assign_speakers",
        "--sample", "3",
        "--no-resume",
        "--device", "cpu",
        "--set", "transcription.modes=[intended]",
        "--set", "deid.entity_types.1.action=flag",
    )  # fmt: skip
    assert resolved["data"]["input_dir"] == "/data/in"
    assert resolved["data"]["output_dir"] == "/data/out"
    assert resolved["stages"] == {"start": "diarize", "end": "assign_speakers", "resume": False}
    assert resolved["sample"]["enabled"] is True
    assert resolved["sample"]["n_files"] == 3
    assert resolved["compute"]["device"] == "cpu"
    assert resolved["transcription"]["modes"] == ["intended"]
    assert resolved["deid"]["entity_types"][1]["action"] == "flag"


def test_named_flag_wins_over_set_for_same_field() -> None:
    resolved = _dry_run("--set", "stages.end=diarize", "--end", "transcribe")
    assert resolved["stages"]["end"] == "transcribe"


def test_overrides_are_revalidated() -> None:
    result = _run("--dry-run", "--start", "detect_phi", "--end", "ingest")
    assert result.exit_code == 2
    assert "stages" in result.stderr
    result = _run("--dry-run", "--device", "tpu")
    assert result.exit_code == 2
    assert "compute.device" in result.stderr


def test_unknown_set_field_fails() -> None:
    result = _run("--dry-run", "--set", "diarization.num_speaker=2")
    assert result.exit_code == 2
    assert "diarization.num_speaker" in result.stderr


def test_malformed_set_fails() -> None:
    result = _run("--dry-run", "--set", "no_equals_sign")
    assert result.exit_code == 2
    assert "key.path=value" in result.stderr


def test_run_with_missing_input_dir_exits_2() -> None:
    result = _run()  # the example config's input_dir does not exist
    assert result.exit_code == 2
    assert "not a directory" in result.stderr


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a.b=2", 2),
        ("a.b=0.25", 0.25),
        ("a.b=true", True),
        ("a.b=null", None),
        ("a.b=[x, y]", ["x", "y"]),
        ("a.b=hello", "hello"),
        ('a.b="1.0"', "1.0"),
        ("a.b=x=y", "x=y"),
        ("a.b=", ""),
    ],
)
def test_parse_assignment_values(text: str, expected: Any) -> None:
    assert parse_assignment(text) == ("a.b", expected)


def test_apply_override_does_not_mutate_input() -> None:
    raw = {"a": {"b": 1}}
    assert apply_override(raw, "a.c", 2) == {"a": {"b": 1, "c": 2}}
    assert raw == {"a": {"b": 1}}


def test_apply_override_rejects_bad_paths() -> None:
    raw: dict[str, Any] = {"a": {"b": 1}, "items": [{"x": 1}]}
    with pytest.raises(ConfigError, match=r"a\.b: is not a section"):
        apply_override(raw, "a.b.c", 2)
    with pytest.raises(ConfigError, match=r"items\.5: no such list item"):
        apply_override(raw, "items.5.x", 2)
    with pytest.raises(ConfigError, match="invalid field path"):
        apply_override(raw, "a..b", 2)
