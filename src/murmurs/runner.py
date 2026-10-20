"""Stage runner: stage ranges, resume, and skip-and-report.

Runs ``stages.start`` through ``stages.end`` for every input file, one file at a time.
A stage is skipped on resume when its artifacts exist and it last ran with the same
input and the same stage config hash, and none of the stages it depends on ran again
in this job. A failure marks that stage ``failed``, skips the file's later stages and
moves on to the next file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from murmurs.backends import registry
from murmurs.config import JobConfig, Stage
from murmurs.errors import PipelineError, SetupError
from murmurs.logging import (
    FileOutcome,
    LoggerProgress,
    MetricsRecorder,
    ProgressCallback,
    RunLogger,
    install_excepthook,
)
from murmurs.manifest import finish_manifest, sample_job_id, select_sample, start_manifest
from murmurs.models import ErrorInfo, FileStatus, StageStatus, TranscriptMode
from murmurs.stages import (
    DEPENDS,
    MISSING_UPSTREAM,
    STAGE_FUNCTIONS,
    Backends,
    StageContext,
    stage_config,
    stage_variants,
)
from murmurs.store import ArtifactStore, file_id_from_hash, hash_file

EXIT_OK = 0
EXIT_FILES_FAILED = 1
EXIT_SETUP_ERROR = 2


class JobSetupError(SetupError):
    """The job cannot start because of its inputs or config."""


@dataclass(frozen=True)
class InputFile:
    """One discovered input audio file."""

    path: Path
    content_hash: str

    @property
    def file_id(self) -> str:
        """ID derived from the file's contents."""
        return file_id_from_hash(self.content_hash)


@dataclass(frozen=True)
class Duplicate:
    """An input whose contents match an earlier file's; it is not processed."""

    path: Path
    file_id: str


@dataclass(frozen=True)
class Discovery:
    """The inputs found under ``input_dir``."""

    files: list[InputFile]
    duplicates: list[Duplicate]


@dataclass(frozen=True)
class FileFailure:
    """Which stage failed for a file, and its error code."""

    file_id: str
    stage: Stage
    error_code: str


@dataclass
class JobSummary:
    """What happened to each file in a job.

    ``done`` files ran at least one stage successfully; ``skipped`` files had nothing
    left to do (every stage was already complete).
    """

    job_id: str
    done: list[str] = field(default_factory=list)
    failed: list[FileFailure] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    duplicates: list[Duplicate] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """0 if every file succeeded, 1 if any failed."""
        return EXIT_FILES_FAILED if self.failed else EXIT_OK


def discover_inputs(input_dir: Path, extensions: list[str]) -> Discovery:
    """Find audio files under ``input_dir`` at any depth, sorted by path.

    Extensions match case-insensitively. Files with the same contents as an earlier
    file are returned as duplicates. Raises ``JobSetupError`` if ``input_dir`` is not a
    directory or a file cannot be read.
    """
    if not input_dir.is_dir():
        raise JobSetupError(f"input_dir {input_dir} is not a directory")
    wanted = {ext.lower() for ext in extensions}
    paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in wanted)
    files: list[InputFile] = []
    duplicates: list[Duplicate] = []
    seen: set[str] = set()
    for path in paths:
        try:
            content_hash = hash_file(path)
        except OSError as exc:
            raise JobSetupError(f"cannot read input file {path} ({type(exc).__name__})") from None
        file = InputFile(path=path, content_hash=content_hash)
        if file.file_id in seen:
            duplicates.append(Duplicate(path=path, file_id=file.file_id))
        else:
            seen.add(file.file_id)
            files.append(file)
    return Discovery(files=files, duplicates=duplicates)


def stage_hashes(config: JobConfig) -> dict[Stage, str]:
    """Hash of each stage's config, folded together with the hashes of its dependencies.

    Changing a stage's config changes its hash and the hash of every stage that depends
    on it, but not of earlier or unrelated stages.
    """
    hashes: dict[Stage, str] = {}
    for stage in Stage:
        payload = {
            "stage": stage.value,
            "config": stage_config(stage, config),
            "depends": [hashes[dep] for dep in DEPENDS[stage]],
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        hashes[stage] = hashlib.sha256(encoded).hexdigest()
    return hashes


def stage_range(config: JobConfig) -> list[Stage]:
    """The stages this job runs, in order."""
    return [s for s in Stage if config.stages.start.order <= s.order <= config.stages.end.order]


def parse_modes(config: JobConfig) -> tuple[TranscriptMode, ...]:
    """The configured transcript modes. Raises ``JobSetupError`` for unknown or no modes."""
    try:
        modes = tuple(dict.fromkeys(TranscriptMode(m) for m in config.transcription.modes))
    except ValueError:
        raise JobSetupError(
            "transcription.modes may only contain " + ", ".join(m.value for m in TranscriptMode)
        ) from None
    if not modes:
        raise JobSetupError("transcription.modes must not be empty")
    return modes


def create_backends(config: JobConfig, stages: list[Stage]) -> Backends:
    """Create only the backends the given stages use."""
    audio_stages = {Stage.INGEST, Stage.LOAD_AUDIO, Stage.DIARIZE, Stage.TRANSCRIBE}
    return Backends(
        audio_loader=(
            registry.create_audio_loader(config) if audio_stages.intersection(stages) else None
        ),
        diarizer=registry.create_diarizer(config) if Stage.DIARIZE in stages else None,
        transcriber=registry.create_transcriber(config) if Stage.TRANSCRIBE in stages else None,
        ner=registry.create_ner_detector(config) if Stage.DETECT_PHI in stages else None,
    )


def run_job(
    config: JobConfig,
    job_id: str,
    *,
    backends: Backends | None = None,
    progress: ProgressCallback | None = None,
    console: TextIO | None = None,
    debug_unsafe: bool = False,
    install_crash_hook: bool = False,
) -> JobSummary:
    """Run a job and return its summary.

    Raises ``SetupError`` (or ``StoreError`` for an unsafe output location) if the job
    cannot start; those map to exit code 2. Per-file failures are reported in the
    summary instead. ``KeyboardInterrupt`` stops the job; completed stages are kept.

    A sample run (``sample.enabled``) processes ``sample.n_files`` files chosen with
    ``sample.seed`` and writes to ``<job_id>-sample``.
    """
    modes = parse_modes(config)
    if config.sample.enabled:
        job_id = sample_job_id(job_id)
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, job_id)
    discovery = discover_inputs(config.data.input_dir, config.data.audio_extensions)
    files = discovery.files
    if config.sample.enabled:
        files = select_sample(
            files, [f.file_id for f in files], config.sample.n_files, config.sample.seed
        )
    logger = RunLogger(log_dir=store.job_dir / "logs", console=console, debug_unsafe=debug_unsafe)
    if install_crash_hook:
        install_excepthook(logger)
    progress = progress if progress is not None else LoggerProgress(logger)

    stages = stage_range(config)
    if config.compute.parallel_workers > 1:
        logger.warning("parallelism_not_implemented", n_workers=config.compute.parallel_workers)
    if backends is None:
        backends = create_backends(config, stages)

    summary = JobSummary(job_id=job_id, duplicates=discovery.duplicates)
    for dup in discovery.duplicates:
        logger.warning("duplicate_input", file_id=dup.file_id)
    logger.info(
        "job_start",
        job_id=job_id,
        n_files=len(files),
        count=len(discovery.duplicates),
    )

    manifest = start_manifest(
        job_id,
        config,
        model_versions=_model_versions(backends),
        input_hashes={f.file_id: f.content_hash for f in files},
    )
    store.write_manifest(manifest)
    metrics = MetricsRecorder(job_id)
    hashes = stage_hashes(config)
    completed = False
    try:
        for file in files:
            outcome = _run_file(
                file, config, store, backends, modes, stages, hashes, logger, progress, metrics,
                summary,
            )  # fmt: skip
            progress.on_file_done(file.file_id, outcome)
        completed = True
    finally:
        metrics.write(store.job_dir / "metrics.json")
        statuses = [s for f in files if (s := store.read_status(f.file_id)) is not None]
        store.write_manifest(finish_manifest(manifest, statuses, completed))

    for failure in summary.failed:
        logger.error(
            "file_failed",
            file_id=failure.file_id,
            stage=failure.stage,
            error_code=failure.error_code,
        )
    logger.info(
        "job_end",
        job_id=job_id,
        n_done=len(summary.done),
        n_failed=len(summary.failed),
        n_skipped=len(summary.skipped),
        count=len(summary.duplicates),
    )
    return summary


def _model_versions(backends: Backends) -> dict[str, str]:
    """Model version reported by each backend the job created."""
    created = {
        "audio_loader": backends.audio_loader,
        "diarizer": backends.diarizer,
        "transcriber": backends.transcriber,
        "ner": backends.ner,
    }
    return {kind: b.model_version for kind, b in created.items() if b is not None}


def _run_file(
    file: InputFile,
    config: JobConfig,
    store: ArtifactStore,
    backends: Backends,
    modes: tuple[TranscriptMode, ...],
    stages: list[Stage],
    hashes: dict[Stage, str],
    logger: RunLogger,
    progress: ProgressCallback,
    metrics: MetricsRecorder,
    summary: JobSummary,
) -> FileOutcome:
    file_id = file.file_id
    status = store.read_status(file_id)
    if status is None or status.input_hash != file.content_hash:
        status = FileStatus(file_id=file_id, input_hash=file.content_hash)
    status = _clear_failure(status)
    ctx = StageContext(
        config=config, store=store, backends=backends, file_id=file_id, path=file.path,
        modes=modes,
    )  # fmt: skip
    ran: set[Stage] = set()

    for stage in stages:
        if config.stages.resume and _is_complete(stage, status, hashes, store, modes, ran):
            progress.on_stage_end(file_id, stage, StageStatus.SKIPPED, 0.0)
            continue

        progress.on_stage_start(file_id, stage)
        error_code: str | None = None
        with_timing = metrics.timed(file_id, stage)
        try:
            with with_timing:
                if any(status.stages[dep] is not StageStatus.DONE for dep in DEPENDS[stage]):
                    raise PipelineError(MISSING_UPSTREAM)
                STAGE_FUNCTIONS[stage](ctx)
        except Exception as exc:
            error_code = logger.exception(exc, stage=stage, file_id=file_id)
        duration = metrics.metrics.files[file_id][stage].duration_s

        if error_code is not None:
            status = _mark_failed(status, stage, stages, error_code)
            store.write_status(status)
            progress.on_stage_end(file_id, stage, StageStatus.FAILED, duration)
            summary.failed.append(FileFailure(file_id, stage, error_code))
            return FileOutcome.FAILED

        status = _mark_done(status, stage, hashes[stage])
        store.write_status(status)
        ran.add(stage)
        progress.on_stage_end(file_id, stage, StageStatus.DONE, duration)

    if ran:
        summary.done.append(file_id)
        return FileOutcome.DONE
    summary.skipped.append(file_id)
    return FileOutcome.SKIPPED


def _is_complete(
    stage: Stage,
    status: FileStatus,
    hashes: dict[Stage, str],
    store: ArtifactStore,
    modes: tuple[TranscriptMode, ...],
    ran: set[Stage],
) -> bool:
    """Whether a stage can be skipped: done before with the same config, outputs present,
    its dependencies done, and none of them ran again in this job."""
    return (
        status.stages.get(stage) is StageStatus.DONE
        and status.stage_hashes.get(stage) == hashes[stage]
        and all(status.stages.get(dep) is StageStatus.DONE for dep in DEPENDS[stage])
        and not ran.intersection(DEPENDS[stage])
        and all(store.exists(status.file_id, stage, v) for v in stage_variants(stage, modes))
    )


def _dependents(stage: Stage) -> set[Stage]:
    """Every stage that uses ``stage``'s output, directly or indirectly."""
    found: set[Stage] = set()
    frontier = {stage}
    while frontier:
        frontier = {s for s, deps in DEPENDS.items() if frontier.intersection(deps)} - found
        found |= frontier
    return found


def _rebuild(status: FileStatus, **changes: object) -> FileStatus:
    """A validated copy of ``status`` with ``changes`` applied."""
    return FileStatus.model_validate(status.model_dump() | changes)


def _clear_failure(status: FileStatus) -> FileStatus:
    """Reset a previous run's failed and skipped stages to pending."""
    stages = {
        s: StageStatus.PENDING if st in (StageStatus.FAILED, StageStatus.SKIPPED) else st
        for s, st in status.stages.items()
    }
    return _rebuild(status, stages=stages, failed_stage=None, error=None)


def _mark_done(status: FileStatus, stage: Stage, stage_hash: str) -> FileStatus:
    """Mark a stage done; stages that depend on it go back to pending."""
    stages = dict(status.stages)
    hashes = dict(status.stage_hashes)
    for dependent in _dependents(stage):
        stages[dependent] = StageStatus.PENDING
        hashes.pop(dependent, None)
    stages[stage] = StageStatus.DONE
    hashes[stage] = stage_hash
    return _rebuild(status, stages=stages, stage_hashes=hashes)


def _mark_failed(
    status: FileStatus, stage: Stage, stages_in_run: list[Stage], error_code: str
) -> FileStatus:
    """Mark a stage failed and the rest of this run's stages for the file skipped."""
    stages = dict(status.stages)
    hashes = dict(status.stage_hashes)
    stages[stage] = StageStatus.FAILED
    hashes.pop(stage, None)
    for later in stages_in_run[stages_in_run.index(stage) + 1 :]:
        stages[later] = StageStatus.SKIPPED
        hashes.pop(later, None)
    return _rebuild(
        status,
        stages=stages,
        stage_hashes=hashes,
        failed_stage=stage,
        error=ErrorInfo(code=error_code),
    )
