"""Job configuration schema and loading."""

from __future__ import annotations

import copy
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

CURRENT_SCHEMA_VERSION = 5
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


def _absolute(path: Path) -> Path:
    """Expand ``~`` and make a path absolute against the current working directory."""
    return path.expanduser().absolute()


class DataConfig(_Model):
    """Input and output locations. Relative paths resolve against the current directory."""

    input_dir: Path
    output_dir: Path
    audio_extensions: list[str] = [".wav", ".m4a", ".mp3"]

    @field_validator("input_dir", "output_dir")
    @classmethod
    def _make_absolute(cls, path: Path) -> Path:
        return _absolute(path)


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

    model: str = "nvidia/Nemotron-3-Diarization"
    num_speakers: int | None = Field(None, ge=1)
    """Cap on speakers: those beyond the N with the most talk time become ``UNKNOWN``.
    ``None`` keeps every speaker the model finds."""
    speaker_gap_tolerance_s: float = Field(0.5, ge=0)
    llm_correction: bool = True


class TranscriptionConfig(_Model):
    """Speech-to-text settings."""

    model: str = "nyralabs/CrisperWhisper2.0_large"
    backend: TranscriptionBackend = TranscriptionBackend.AUTO
    language: str = "en"
    modes: list[str] = ["verbatim", "intended"]
    word_timestamps: bool = True
    draft_model: str | None = None
    """A smaller CrisperWhisper 2.0 model (e.g. ``nyralabs/CrisperWhisper2.0_turbo``) for
    speculative decoding: each mode is decoded in its own pass, about 1.2 times faster
    than without for one mode, changing about 0.1% of words. ``null`` = off."""


class EntityType(_Model):
    """One deidentification entity type and its NER label mapping."""

    name: str
    tag: str
    description: str = ""
    ner_labels: list[str] = []
    action: EntityAction = EntityAction.REDACT


def _entity(name: str, description: str, ner_labels: list[str], action: EntityAction) -> EntityType:
    return EntityType(
        name=name,
        tag=f"[{name}_{{n}}]",
        description=description,
        ner_labels=ner_labels,
        action=action,
    )


PROVISIONAL_ENTITY_TYPES: tuple[EntityType, ...] = (
    _entity(
        "PERSON",
        "Names of people: full, first or last names, nicknames and partial names "
        "(for example 'Mrs. Ruiz' or 'the Kowalskis').",
        ["first_name", "last_name", "user_name"],
        EntityAction.REDACT,
    ),
    _entity(
        "ORGANIZATION",
        "Named organizations and institutions: companies and employers, schools and "
        "colleges, hospitals and clinics, churches, clubs, care homes and other named "
        "facilities.",
        ["company_name"],
        EntityAction.REDACT,
    ),
    _entity(
        "LOCATION",
        "Named places smaller than a country: street addresses, streets, neighbourhoods, "
        "towns, cities, counties, states and postcodes.",
        ["street_address", "city", "county", "state", "postcode", "coordinate"],
        EntityAction.REDACT,
    ),
    _entity(
        "CONTACT",
        "Phone and fax numbers, email addresses and web addresses.",
        ["phone_number", "email", "fax_number", "url"],
        EntityAction.REDACT,
    ),
    _entity(
        "ID",
        "Identifying numbers: social security, national ID, medical record, health plan, "
        "account, licence, vehicle, device, employee, customer, tax and case numbers.",
        [
            "ssn",
            "national_id",
            "medical_record_number",
            "health_plan_beneficiary_number",
            "account_number",
            "certificate_license_number",
            "license_plate",
            "vehicle_identifier",
            "device_identifier",
            "employee_id",
            "customer_id",
            "tax_id",
            "unique_id",
        ],
        EntityAction.REDACT,
    ),
    _entity("AGE", "A person's age.", ["age"], EntityAction.FLAG),
    _entity(
        "DATE",
        "Specific dates and date references: days, months, years, birthdays.",
        ["date", "date_of_birth", "date_time"],
        EntityAction.FLAG,
    ),
    _entity(
        "SENSITIVE_ATTRIBUTE",
        "Religion, sexuality, race or ethnicity, political views and gender.",
        ["religious_belief", "sexuality", "race_ethnicity", "political_view", "gender"],
        EntityAction.FLAG,
    ),
    _entity(
        "SOCIOECONOMIC",
        "Occupation, education level and employment status.",
        ["occupation", "education_level", "employment_status"],
        EntityAction.FLAG,
    ),
)
"""The provisional tag set (D10), used by ``examples/job.yaml`` and the MVP evaluation set.
The data owners decide the final set. NER labels not mapped to any type are ignored."""


class LLMServerConfig(_Model):
    """Settings for the local vLLM server that runs the LLM, and for requests to it (D22).

    Only ``reasoning_effort`` changes results, so it is the only one in the LLM stages'
    config hashes; the rest affect speed, memory and robustness.
    """

    gpu_memory_utilization: float = Field(0.85, gt=0, le=1)
    """Share of GPU memory vLLM may use for weights, activations and its KV cache."""
    max_model_len: int = Field(32768, ge=1024)
    """Longest prompt plus response, in tokens."""
    startup_timeout_s: float = Field(900, gt=0)
    """How long to wait for the server to become ready before the job stops."""
    max_output_tokens: int = Field(8192, ge=256)
    """Cap on each response, reasoning included. Normal responses take 2,000 to 5,000
    tokens, so hitting the cap means a runaway: it is not retried as is, its chunk is
    split in half and the halves are tried instead."""
    max_concurrency: int = Field(64, ge=1, le=64)
    """Requests sent to the server at the same time; it batches them on the GPU, so more
    requests in flight means more tokens per second overall."""
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    """GPT-OSS reasoning effort for every request. Lower is faster but may lower recall."""


class DeidConfig(_Model):
    """PHI detection settings."""

    ner_model: str = "OpenMed/privacy-filter-nemotron-v2"
    llm_model: str = "openai/gpt-oss-120b"
    llm_server: LLMServerConfig = Field(default_factory=LLMServerConfig)
    entity_types: list[EntityType]
    prompt_path: Path | None = None
    redaction_threshold: float = Field(0.5, ge=0, le=1)
    flag_edge_cases: bool = True

    @field_validator("prompt_path")
    @classmethod
    def _make_absolute(cls, path: Path | None) -> Path | None:
        return None if path is None else _absolute(path)


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
    # 2 -> 3 changed the diarization.model default, made num_speakers a cap and added
    # .mp3 to the default audio_extensions; the fields are unchanged, so there is
    # nothing to rewrite.
    # 3 -> 4 added deid.llm_server, which has defaults.
    # 4 -> 5 added llm_server.max_output_tokens, max_concurrency and reasoning_effort,
    # and transcription.draft_model, which have defaults.
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
    return validate_config(read_config_file(path))


def resolve_config(path: Path, overrides: list[tuple[str, Any]]) -> JobConfig:
    """Load a YAML job config, apply overrides in order, and validate the result.

    Precedence is schema defaults < config file < overrides. Each override is a dotted
    field path and a value, e.g. ``("stages.end", "transcribe")``.
    """
    raw = read_config_file(path)
    for key, value in overrides:
        raw = apply_override(raw, key, value)
    return validate_config(raw)


def parse_assignment(text: str) -> tuple[str, Any]:
    """Parse a ``key.path=value`` override.

    The value is read as a YAML scalar, so ``2`` is an int, ``0.5`` a float,
    ``true`` a bool, ``null`` None and ``[a, b]`` a list. Quote a value to force a
    string, e.g. ``transcription.model="1.0"``.
    """
    key, sep, value_text = text.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(["--set: expected key.path=value"])
    try:
        value = yaml.safe_load(value_text) if value_text.strip() else ""
    except yaml.YAMLError:
        raise ConfigError([f"{key}: value is not valid YAML"]) from None
    return key, value


def apply_override(raw: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``raw`` with the dotted ``key`` set to ``value``.

    Missing intermediate sections are created; numeric parts index into existing lists
    (e.g. ``deid.entity_types.0.action``). Unknown field names are caught later by
    validation.
    """
    parts = key.split(".")
    if any(not part for part in parts):
        raise ConfigError([f"{key}: invalid field path"])
    result = copy.deepcopy(raw)
    node: Any = result
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        where = ".".join(parts[: i + 1])
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                raise ConfigError([f"{where}: no such list item"])
            if last:
                node[int(part)] = value
            else:
                node = node[int(part)]
        elif isinstance(node, dict):
            if last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
        else:
            raise ConfigError([f"{'.'.join(parts[:i])}: is not a section, cannot set {key}"])
    return result


def read_config_file(path: Path) -> dict[str, Any]:
    """Read a YAML config file into a raw mapping, without validating it.

    Raises ``ConfigError`` if the file is missing, is not valid YAML, or is not a mapping.
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
    return raw
