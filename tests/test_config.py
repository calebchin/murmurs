"""P1-02: config schema and loading."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from murmurs.cli import app
from murmurs.config import (
    CURRENT_SCHEMA_VERSION,
    ConfigError,
    EntityAction,
    JobConfig,
    Stage,
    TranscriptionBackend,
    load_config,
)

EXAMPLE = Path(__file__).parent.parent / "examples" / "job.yaml"


def _example_raw() -> dict[str, Any]:
    raw = yaml.safe_load(EXAMPLE.read_text())
    assert isinstance(raw, dict)
    return raw


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _load_error(tmp_path: Path, raw: dict[str, Any]) -> ConfigError:
    with pytest.raises(ConfigError) as info:
        load_config(_write(tmp_path, raw))
    return info.value


def test_example_config_loads() -> None:
    config = load_config(EXAMPLE)
    assert isinstance(config, JobConfig)
    assert [e.name for e in config.deid.entity_types] == ["PERSON", "LOCATION"]
    assert config.deid.entity_types[0].action is EntityAction.REDACT
    assert config.transcription.backend is TranscriptionBackend.AUTO


def test_minimal_config_takes_defaults(tmp_path: Path) -> None:
    raw = {
        "data": {"input_dir": "in", "output_dir": "out"},
        "deid": {"entity_types": [{"name": "PERSON", "tag": "[PERSON_{n}]"}]},
    }
    config = load_config(_write(tmp_path, raw))
    assert config.schema_version == CURRENT_SCHEMA_VERSION
    assert config.stages.start is Stage.INGEST
    assert config.stages.end is Stage.DETECT_PHI
    assert config.diarization.speaker_gap_tolerance_s == 0.5


def test_missing_input_dir_names_the_field(tmp_path: Path) -> None:
    raw = _example_raw()
    del raw["data"]["input_dir"]
    err = _load_error(tmp_path, raw)
    assert any(p.startswith("data.input_dir:") for p in err.problems)
    assert "data.input_dir" in str(err)


def test_start_after_end_fails(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["stages"] = {"start": "detect_phi", "end": "ingest"}
    err = _load_error(tmp_path, raw)
    assert any(p.startswith("stages:") and "detect_phi" in p for p in err.problems)


def test_start_equal_to_end_is_allowed(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["stages"] = {"start": "transcribe", "end": "transcribe"}
    assert load_config(_write(tmp_path, raw)).stages.start is Stage.TRANSCRIBE


def test_redaction_threshold_out_of_range_fails(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["deid"]["redaction_threshold"] = 1.5
    err = _load_error(tmp_path, raw)
    assert any(p.startswith("deid.redaction_threshold:") for p in err.problems)


def test_newer_schema_version_fails_clearly(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["schema_version"] = 999
    err = _load_error(tmp_path, raw)
    assert err.problems == [
        f"schema_version: version 999 is newer than this version of murmurs supports "
        f"(up to {CURRENT_SCHEMA_VERSION}); upgrade murmurs"
    ]


def test_version_1_config_migrates_with_default_tolerance(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["schema_version"] = 1
    del raw["diarization"]["speaker_gap_tolerance_s"]
    config = load_config(_write(tmp_path, raw))
    assert config.schema_version == CURRENT_SCHEMA_VERSION
    assert config.diarization.speaker_gap_tolerance_s == 0.5


def test_negative_gap_tolerance_fails(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["diarization"]["speaker_gap_tolerance_s"] = -1
    err = _load_error(tmp_path, raw)
    assert any(p.startswith("diarization.speaker_gap_tolerance_s:") for p in err.problems)


def test_enum_fields_reject_unknown_values(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["transcription"]["backend"] = "whisperx"
    raw["deid"]["entity_types"][0]["action"] = "delete"
    err = _load_error(tmp_path, raw)
    paths = {p.split(":")[0] for p in err.problems}
    assert paths == {"transcription.backend", "deid.entity_types.0.action"}


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["deid"]["redaction_treshold"] = 0.7
    err = _load_error(tmp_path, raw)
    assert any(p.startswith("deid.redaction_treshold:") for p in err.problems)


def test_errors_do_not_echo_input_values(tmp_path: Path) -> None:
    raw = _example_raw()
    raw["deid"]["redaction_threshold"] = "SENTINEL_PHI_7f3a"
    raw["deid"]["entity_types"][0]["description"] = "SENTINEL_PHI_7f3a"
    raw["deid"]["entity_types"][0]["action"] = "SENTINEL_PHI_7f3a"
    err = _load_error(tmp_path, raw)
    assert "SENTINEL_PHI_7f3a" not in str(err)


def test_invalid_yaml_reports_location_without_contents(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text("data:\n  input_dir: [SENTINEL_PHI_7f3a\n")
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert "invalid YAML at line" in str(info.value)
    assert "SENTINEL_PHI_7f3a" not in str(info.value)


def test_non_mapping_and_missing_file_fail(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_config(path)
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path / "missing.yaml")


def test_config_schema_command_prints_json_schema() -> None:
    result = CliRunner().invoke(app, ["config", "schema"])
    assert result.exit_code == 0
    schema = json.loads(result.output)
    assert schema["title"] == "JobConfig"
    assert set(schema["required"]) == {"data", "deid"}
