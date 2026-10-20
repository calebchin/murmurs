"""Job configuration schema and loading."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CURRENT_SCHEMA_VERSION = 2
"""Newest config schema version this code reads. Older versions are migrated on load."""


class ConfigError(Exception):
    """A config file could not be read or failed validation.

    ``problems`` holds one ``"field.path: message"`` string per problem. Messages name
    fields and constraints only; they never echo input values or file contents.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid config:\n" + "\n".join(f"  {p}" for p in problems))


class _Model(BaseModel):
    """Base for config sections: unknown keys are errors, so typos are not ignored."""

    model_config = ConfigDict(extra="forbid")


class Stage(StrEnum):
    """Pipeline stages, in run order."""

    INGEST = "ingest"
    LOAD_AUDIO = "load_audio"
    DIARIZE = "diarize"
    TRANSCRIBE = "transcribe"
    ASSIGN_SPEAKERS = "assign_speakers"
    DETECT_PHI = "detect_phi"

    @property
    def order(self) -> int:
        """Position of this stage in the pipeline (0 = first)."""
        return list(Stage).index(self)


class Device(StrEnum):
    """Compute device for model backends."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class TranscriptionBackend(StrEnum):
    """CrisperWhisper inference backend."""

    AUTO = "auto"
    CT2 = "ct2"
    TRANSFORMERS = "transformers"


class EntityAction(StrEnum):
    """What happens to an entity type's detections."""

    REDACT = "redact"
    FLAG = "flag"
    IGNORE = "ignore"


class DataConfig(_Model):
    """Input and output locations."""

    input_dir: Path
    output_dir: Path
    audio_extensions: list[str] = [".wav", ".m4a"]


class SampleConfig(_Model):
    """Sample run on a few files for human quality review."""

    enabled: bool = False
    n_files: int = Field(5, ge=1)
    seed: int = 0


class StagesConfig(_Model):
    """Which stage range to run, and whether to resume."""

    start: Stage = Stage.INGEST
    end: Stage = Stage.DETECT_PHI
    resume: bool = True

    @model_validator(mode="after")
    def _start_not_after_end(self) -> StagesConfig:
        if self.start.order > self.end.order:
            raise ValueError(
                f"start stage '{self.start.value}' comes after end stage '{self.end.value}'"
            )
        return self


class DiarizationConfig(_Model):
    """Speaker diarization and speaker assignment settings."""

    model: str = "pyannote/speaker-diarization-3.1"
    num_speakers: int | None = None
    speaker_gap_tolerance_s: float = Field(0.5, ge=0)
    llm_correction: bool = True


class TranscriptionConfig(_Model):
    """Speech-to-text settings."""

    model: str = "nyralabs/CrisperWhisper2.0_large"
    backend: TranscriptionBackend = TranscriptionBackend.AUTO
    language: str = "en"
    modes: list[str] = ["verbatim", "intended"]
    word_timestamps: bool = True


class EntityType(_Model):
    """One deidentification entity type and its NER label mapping."""

    name: str
    tag: str
    description: str = ""
    ner_labels: list[str] = []
    action: EntityAction = EntityAction.REDACT


class DeidConfig(_Model):
    """PHI detection settings."""

    ner_model: str = "OpenMed/privacy-filter-nemotron-v2"
    llm_model: str = "openai/gpt-oss-120b"
    entity_types: list[EntityType]
    prompt_path: Path | None = None
    redaction_threshold: float = Field(0.5, ge=0, le=1)
    flag_edge_cases: bool = True


class ComputeConfig(_Model):
    """Hardware settings. ``parallel_workers`` counts processes, not threads."""

    device: Device = Device.AUTO
    max_gpu_memory_gb: float | None = None
    parallel_workers: int = 1


class JobConfig(_Model):
    """Complete configuration for one job."""

    schema_version: int = CURRENT_SCHEMA_VERSION
    data: DataConfig
    sample: SampleConfig = Field(default_factory=SampleConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    deid: DeidConfig
    compute: ComputeConfig = Field(default_factory=ComputeConfig)


def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw config mapping to the current schema version.

    Raises ``ConfigError`` if the version is not an integer, is below 1, or is newer
    than this code supports.
    """
    version = raw.get("schema_version", CURRENT_SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigError(["schema_version: must be an integer"])
    if version < 1:
        raise ConfigError(["schema_version: must be at least 1"])
    if version > CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            [
                f"schema_version: version {version} is newer than this version of murmurs "
                f"supports (up to {CURRENT_SCHEMA_VERSION}); upgrade murmurs"
            ]
        )
    # 1 -> 2 added diarization.speaker_gap_tolerance_s, which has a default.
    return {**raw, "schema_version": CURRENT_SCHEMA_VERSION}


def _format_errors(exc: ValidationError) -> list[str]:
    """Turn pydantic errors into ``field.path: message`` lines without input values."""
    problems = []
    for err in exc.errors(include_input=False, include_url=False):
        path = ".".join(str(part) for part in err["loc"]) or "(root)"
        problems.append(f"{path}: {err['msg']}")
    return problems


def validate_config(raw: dict[str, Any]) -> JobConfig:
    """Migrate and validate a raw config mapping.

    Raises ``ConfigError`` listing every problem by field path.
    """
    migrated = _migrate(raw)
    try:
        return JobConfig.model_validate(migrated)
    except ValidationError as exc:
        raise ConfigError(_format_errors(exc)) from None


def load_config(path: Path) -> JobConfig:
    """Load and validate a YAML job config.

    Raises ``ConfigError`` if the file is missing, is not valid YAML, is not a mapping,
    or fails validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError([f"(file): cannot read config file ({type(exc).__name__})"]) from None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigError([f"(file): invalid YAML{where}"]) from None
    if not isinstance(raw, dict):
        raise ConfigError(["(root): config must be a YAML mapping"])
    return validate_config(raw)
