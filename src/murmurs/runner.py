"""Stage runner: stage groups, resume, and skip-and-report.

Runs ``stages.start`` through ``stages.end`` in stage groups (D21): each group runs for
every input file before the next group starts, and each model backend is open only
while a group that uses it runs. A stage is skipped on resume when its artifacts exist
and it last ran with the same input and the same stage config hash, and none of the
stages it depends on ran again in this job. A failure marks that stage ``failed`` and
the file skips the rest of the job; other files carry on.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from murmurs.backends import AudioLoader, Diarizer, LLMClient, NERDetector, Transcriber, registry
from murmurs.backends.registry import BackendKind
from murmurs.config import JobConfig, Stage
from murmurs.errors import BackendSetupError, PipelineError, SetupError
from murmurs.logging import (
    FileOutcome,
    LoggerProgress,
    MetricsRecorder,
    ProgressCallback,
    RunLogger,
    install_excepthook,
)
from murmurs.manifest import finish_manifest, sample_job_id, select_sample, start_manifest
from murmurs.models import AudioInfo, ErrorInfo, FileStatus, StageStatus, TranscriptMode
from murmurs.offline import check_cached, check_prepared, enforce_offline
from murmurs.stages import (
    DEPENDS,
    MISSING_UPSTREAM,
    RULE_BASED,
    STAGE_FUNCTIONS,
    WAVEFORM_STAGES,
    Backends,
    StageContext,
    stage_config,
    stage_groups,
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


def backend_kinds(stages: list[Stage], config: JobConfig) -> set[BackendKind]:
    """The kinds of backend the given stages use."""
    kinds: set[BackendKind] = set()
    for stage in stages:
        kinds |= stage_backend_kinds(stage, config)
    return kinds


def check_weights(config: JobConfig, stages: list[Stage]) -> dict[str, str]:
    """Check that the weights the given stages' backends need are cached.

    Returns the cached commit hash per weight label. Raises ``BackendSetupError`` if a
    selected backend is not registered, ``MissingWeightsError`` if weights are missing,
    or ``UnpreparedModelError`` if a model built from them has not been prepared.
    """
    plan = registry.required_weights(config, sorted(backend_kinds(stages, config)))
    if plan.unavailable:
        raise BackendSetupError("; ".join(plan.unavailable))
    revisions = check_cached(plan.weights)
    check_prepared(plan.prepared)
    return revisions


def create_backends(config: JobConfig, stages: list[Stage]) -> Backends:
    """Create only the backends the given stages use."""
    kinds = backend_kinds(stages, config)
    return Backends(
        audio_loader=(
            registry.create_audio_loader(config) if BackendKind.AUDIO_LOADER in kinds else None
        ),
        diarizer=registry.create_diarizer(config) if BackendKind.DIARIZER in kinds else None,
        transcriber=(
            registry.create_transcriber(config) if BackendKind.TRANSCRIBER in kinds else None
        ),
        ner=registry.create_ner_detector(config) if BackendKind.NER in kinds else None,
        llm=registry.create_llm_client(config) if BackendKind.LLM in kinds else None,
    )


def stage_backend_kinds(stage: Stage, config: JobConfig) -> set[BackendKind]:
    """The kinds of backend one stage calls when it runs."""
    kinds = set()
    if stage is Stage.INGEST or stage in WAVEFORM_STAGES:
        kinds.add(BackendKind.AUDIO_LOADER)  # diarize and transcribe decode again on resume
    if stage is Stage.DIARIZE:
        kinds.add(BackendKind.DIARIZER)
    if stage is Stage.TRANSCRIBE:
        kinds.add(BackendKind.TRANSCRIBER)
    if stage is Stage.ASSIGN_SPEAKERS and config.diarization.llm_correction:
        kinds.add(BackendKind.LLM)
    if stage is Stage.DETECT_PHI:
        kinds |= {BackendKind.NER, BackendKind.LLM}
    return kinds


AnyBackend = AudioLoader | Diarizer | Transcriber | NERDetector | LLMClient


def _create(kind: BackendKind, config: JobConfig) -> AnyBackend:
    """Create a backend of a kind the runner manages."""
    match kind:
        case BackendKind.AUDIO_LOADER:
            return registry.create_audio_loader(config)
        case BackendKind.DIARIZER:
            return registry.create_diarizer(config)
        case BackendKind.TRANSCRIBER:
            return registry.create_transcriber(config)
        case BackendKind.NER:
            return registry.create_ner_detector(config)
        case BackendKind.LLM:
            return registry.create_llm_client(config)
    raise ValueError(f"the runner does not manage {kind.value} backends")


MANAGED_KINDS = (
    BackendKind.AUDIO_LOADER,
    BackendKind.DIARIZER,
    BackendKind.TRANSCRIBER,
    BackendKind.NER,
    BackendKind.LLM,
)
"""Kinds the runner creates and closes. Their values are attribute names on ``Backends``."""


class _BackendPool:
    """Backends for one job, each open only while a stage group that uses it runs (D21).

    A backend is created just before the first stage that actually runs with it (so a
    resume that skips a group loads nothing for it) and closed after the last group in
    the run that uses it. Backends passed in by the caller are used as given and never
    created or closed here.
    """

    def __init__(
        self,
        config: JobConfig,
        groups: list[tuple[str, list[Stage]]],
        given: Backends | None,
        logger: RunLogger,
    ) -> None:
        self._config = config
        self._logger = logger
        self._owned = given is None
        self.backends = given if given is not None else Backends()
        self._last_group: dict[BackendKind, int] = {}
        for index, (_, stages) in enumerate(groups):
            for stage in stages:
                for kind in stage_backend_kinds(stage, config):
                    self._last_group[kind] = index
        self.model_versions: dict[str, str] = {}
        if not self._owned:
            for kind in MANAGED_KINDS:
                backend = getattr(self.backends, kind.value)
                if backend is not None:
                    self.model_versions[kind.value] = backend.model_version
                    self._log_device(kind, backend)

    def ensure(self, stage: Stage) -> None:
        """Create any backend ``stage`` needs that is not open yet.

        Raises ``SetupError`` if one cannot be created; the job stops (exit code 2).
        """
        if not self._owned:
            return
        for kind in sorted(stage_backend_kinds(stage, self._config)):
            if getattr(self.backends, kind.value) is not None:
                continue
            try:
                backend = _create(kind, self._config)
            except SetupError:
                raise
            except Exception as exc:  # e.g. out of GPU memory; library text is dropped
                raise BackendSetupError(
                    f"could not create the {kind.value} backend ({type(exc).__name__})"
                ) from None
            setattr(self.backends, kind.value, backend)
            self.model_versions[kind.value] = backend.model_version
            self._logger.info("backend_open", backend=kind.value)
            self._log_device(kind, backend)

    def _log_device(self, kind: BackendKind, backend: AnyBackend) -> None:
        if getattr(backend, "device_fallback", False):
            self._logger.warning("device_fallback_cpu", backend=kind.value)

    def close_after(self, group_index: int) -> None:
        """Close the backends no later group in this run uses."""
        for kind, last in self._last_group.items():
            if last == group_index:
                self._close(kind)

    def close_all(self) -> None:
        """Close every backend still open (end of job, Ctrl-C or error)."""
        for kind in MANAGED_KINDS:
            self._close(kind)

    def _close(self, kind: BackendKind) -> None:
        if not self._owned:
            return
        backend = getattr(self.backends, kind.value)
        if backend is None:
            return
        setattr(self.backends, kind.value, None)
        try:
            backend.close()
        except Exception as exc:  # a failed close must not hide the job's own outcome
            self._logger.exception(exc, event="backend_close_failed")
        else:
            self._logger.info("backend_close", backend=kind.value)


@dataclass
class _FileRun:
    """One file's state for the whole job, carried across stage groups."""

    file: InputFile
    status: FileStatus
    ctx: StageContext
    ran: set[Stage] = field(default_factory=set)
    failed: bool = False


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

    Stage groups run in order, each for every file before the next group starts (D21);
    a file that fails a stage skips the rest of the job. Each backend is created just
    before the first stage that runs with it and closed after its last group.

    Raises ``SetupError`` (or ``StoreError`` for an unsafe output location) if the job
    cannot start, or if a backend cannot be created when its group starts; those map to
    exit code 2, and completed stages are kept. Per-file failures are reported in the
    summary instead. ``KeyboardInterrupt`` stops the job; completed stages are kept.

    A sample run (``sample.enabled``) processes ``sample.n_files`` files chosen with
    ``sample.seed`` and writes to ``<job_id>-sample``.

    Hugging Face offline mode is forced for the rest of the process. When ``backends``
    is not given, the weights they need must already be cached
    (``murmurs models download``); otherwise ``MissingWeightsError`` is raised.
    """
    enforce_offline()
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
    groups = stage_groups(stages)
    if config.compute.parallel_workers > 1:
        logger.warning("parallelism_not_implemented", n_workers=config.compute.parallel_workers)
    weight_revisions: dict[str, str] = {}
    if backends is None:
        weight_revisions = check_weights(config, stages)
        logger.info("weights_cached", count=len(weight_revisions))
    pool = _BackendPool(config, groups, backends, logger)

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
        model_versions={**pool.model_versions, **weight_revisions},
        input_hashes={f.file_id: f.content_hash for f in files},
    )
    store.write_manifest(manifest)
    metrics = MetricsRecorder(job_id)
    hashes = stage_hashes(config)
    runs = [_start_file(file, config, store, pool.backends, modes) for file in files]
    step = _Step(config, store, pool, modes, stages, hashes, logger, progress, metrics, summary)
    completed = False
    group_wall: dict[str, float] = {}
    job_started = time.perf_counter()
    try:
        for index, (name, group_stages) in enumerate(groups):
            group_started = time.perf_counter()
            active = [run for run in runs if not run.failed]
            logger.info("group_start", group=name, n_files=len(active))
            for run in active:
                for stage in group_stages:
                    if not step.run(run, stage):
                        break
                run.ctx.release_waveform()
                if run.failed:
                    progress.on_file_done(run.file.file_id, FileOutcome.FAILED)
            pool.close_after(index)
            group_wall[name] = time.perf_counter() - group_started
            logger.info(
                "group_end", group=name, n_files=len(active), duration_s=round(group_wall[name], 3)
            )
        for run in runs:
            export_transcripts(store, run.file, run.status, modes)
        for run in runs:
            if run.failed:
                continue
            outcome = FileOutcome.DONE if run.ran else FileOutcome.SKIPPED
            (summary.done if run.ran else summary.skipped).append(run.file.file_id)
            progress.on_file_done(run.file.file_id, outcome)
        completed = True
    finally:
        pool.close_all()
        processed = [run for run in runs if run.ran]
        throughput = metrics.summarize(
            n_files=len(processed),
            audio_s=sum(_audio_seconds(store, run.file.file_id) for run in processed),
            wall_s=time.perf_counter() - job_started,
            group_wall_s=group_wall,
        )
        metrics.write(store.job_dir / "metrics.json")
        statuses = [s for f in files if (s := store.read_status(f.file_id)) is not None]
        manifest = manifest.model_copy(
            update={"model_versions": {**pool.model_versions, **weight_revisions}}
        )
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
        audio_duration_s=throughput.audio_s,
        duration_s=throughput.wall_s,
    )
    return summary


def _audio_seconds(store: ArtifactStore, file_id: str) -> float:
    """A file's audio duration from its ``load_audio`` artifact, or 0 if it has none."""
    if not store.exists(file_id, Stage.LOAD_AUDIO):
        return 0.0
    return store.read(file_id, Stage.LOAD_AUDIO, AudioInfo).duration_s


def export_folder_name(file: InputFile) -> str:
    """``<input file stem>-<first 8 characters of its file ID>``, safe as a folder name.

    Input file names contain no PHI (design doc), so they can name output folders.
    """
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", file.path.stem).lstrip(".") or "file"
    return f"{stem}-{file.file_id[:8]}"


def export_transcripts(
    store: ArtifactStore, file: InputFile, status: FileStatus, modes: tuple[TranscriptMode, ...]
) -> None:
    """Copy a file's finished transcripts into ``transcripts/<folder>/``.

    For each mode: ``<mode>.txt`` (final, LLM-corrected if correction ran),
    ``<mode>.rule_based.txt`` and ``<mode>.words.tsv`` (final). Only for files whose
    speaker assignment is done.
    """
    if status.stages.get(Stage.ASSIGN_SPEAKERS) is not StageStatus.DONE:
        return
    folder = export_folder_name(file)
    for mode in modes:
        sources = {
            f"{mode.value}.txt": (mode.value, ".txt"),
            f"{mode.value}.rule_based.txt": (f"{mode.value}.{RULE_BASED}", ".txt"),
            f"{mode.value}.words.tsv": (mode.value, ".words.tsv"),
        }
        for name, (variant, suffix) in sources.items():
            source = store.artifact_path(file.file_id, Stage.ASSIGN_SPEAKERS, variant, suffix)
            if source.is_file():
                store.write_export(folder, name, source.read_text(encoding="utf-8"))


def _start_file(
    file: InputFile,
    config: JobConfig,
    store: ArtifactStore,
    backends: Backends,
    modes: tuple[TranscriptMode, ...],
) -> _FileRun:
    """A file's run state: its saved status (failures from earlier runs cleared)."""
    status = store.read_status(file.file_id)
    if status is None or status.input_hash != file.content_hash:
        status = FileStatus(file_id=file.file_id, input_hash=file.content_hash)
    ctx = StageContext(
        config=config, store=store, backends=backends, file_id=file.file_id, path=file.path,
        input_hash=file.content_hash, modes=modes,
    )  # fmt: skip
    return _FileRun(file=file, status=_clear_failure(status), ctx=ctx)


@dataclass
class _Step:
    """Runs one stage for one file: skip if complete, else run and record the result."""

    config: JobConfig
    store: ArtifactStore
    pool: _BackendPool
    modes: tuple[TranscriptMode, ...]
    stages: list[Stage]
    hashes: dict[Stage, str]
    logger: RunLogger
    progress: ProgressCallback
    metrics: MetricsRecorder
    summary: JobSummary

    def run(self, run: _FileRun, stage: Stage) -> bool:
        """Returns False if the stage failed (the file then skips the rest of the job)."""
        file_id = run.file.file_id
        if self.config.stages.resume and _is_complete(
            stage, run.status, self.hashes, self.store, self.modes, run.ran
        ):
            self.progress.on_stage_end(file_id, stage, StageStatus.SKIPPED, 0.0)
            return True

        self.progress.on_stage_start(file_id, stage)
        upstream_done = all(run.status.stages[dep] is StageStatus.DONE for dep in DEPENDS[stage])
        if upstream_done:
            self.pool.ensure(stage)  # outside the timing: model loading is not stage work
        error_code: str | None = None
        try:
            with self.metrics.timed(file_id, stage):
                if not upstream_done:
                    raise PipelineError(MISSING_UPSTREAM)
                STAGE_FUNCTIONS[stage](run.ctx)
        except Exception as exc:
            error_code = self.logger.exception(exc, stage=stage, file_id=file_id)
        duration = self.metrics.metrics.files[file_id][stage].duration_s

        if error_code is not None:
            run.status = _mark_failed(run.status, stage, self.stages, error_code)
            self.store.write_status(run.status)
            self.progress.on_stage_end(file_id, stage, StageStatus.FAILED, duration)
            self.summary.failed.append(FileFailure(file_id, stage, error_code))
            run.failed = True
            return False

        run.status = _mark_done(run.status, stage, self.hashes[stage])
        self.store.write_status(run.status)
        run.ran.add(stage)
        self.progress.on_stage_end(file_id, stage, StageStatus.DONE, duration)
        return True


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
