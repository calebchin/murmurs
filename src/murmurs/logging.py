"""PHI-safe structured logging and timing metrics.

Logs can only carry allowlisted fields (IDs, counts, durations, stage, status, error
code), and string values must be short tokens with no spaces, so free text such as
transcript words, prompts or exception messages cannot be logged by accident.

Events go to a JSON-lines file under ``<output_dir>/<job_id>/logs/`` and, in
human-readable form, to the console (stderr). Log files are append-only and written
line by line, not atomically; a crash can leave a partial last line, which readers
should skip.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field

from murmurs.config import Stage
from murmurs.errors import PipelineError
from murmurs.models import StageStatus
from murmurs.store import atomic_writer

Level = Literal["debug", "info", "warning", "error"]

ALLOWED_FIELDS = frozenset(
    {
        # identifiers
        "job_id",
        "file_id",
        "input_hash",
        "stage",
        "group",
        "status",
        "error_code",
        "exception_type",
        "backend",
        "model_version",
        # counts
        "count",
        "n_files",
        "n_done",
        "n_failed",
        "n_skipped",
        "n_segments",
        "n_words",
        "n_turns",
        "n_spans",
        "n_workers",
        "attempt",
        # durations
        "duration_s",
        "audio_duration_s",
    }
)
"""Field names a log event may carry. Anything else raises ``UnsafeLogFieldError``."""

UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
"""Error code logged for exceptions that are not ``PipelineError``."""

DEBUG_UNSAFE_WARNING = (
    "--debug-unsafe is on: full tracebacks, which may contain transcript content, "
    "are written to the log file. Do not share these logs."
)

_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:/+-]{0,128}$")
_LEVELS: dict[str, int] = {"debug": 10, "info": 20, "warning": 30, "error": 40}

FieldValue = str | int | float | bool | None


class UnsafeLogFieldError(ValueError):
    """A log call used a field or value that could carry free text."""


def _check_fields(event: str, fields: dict[str, object]) -> dict[str, FieldValue]:
    if not _EVENT_NAME.fullmatch(event):
        raise UnsafeLogFieldError("event name must be snake_case")
    clean: dict[str, FieldValue] = {}
    for name, value in fields.items():
        if name not in ALLOWED_FIELDS:
            raise UnsafeLogFieldError(f"log field '{name}' is not allowlisted")
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, str):
            if not _SAFE_TOKEN.fullmatch(value):
                raise UnsafeLogFieldError(f"log field '{name}' must be a short token")
            clean[name] = value
        elif value is None or isinstance(value, int | float | bool):
            clean[name] = value
        else:
            raise UnsafeLogFieldError(f"log field '{name}' must be a string, number or bool")
    return clean


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class RunLogger:
    """Writes allowlisted events to a JSON-lines file and to the console.

    ``log_dir=None`` logs to the console only. With ``debug_unsafe=True``, exception
    tracebacks are added to the log file (never the console) and a warning is shown.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        console: TextIO | None = None,
        debug_unsafe: bool = False,
        console_level: Level = "info",
    ) -> None:
        self.console = console if console is not None else sys.stderr
        self._console_min = _LEVELS[console_level]
        self._debug_unsafe = debug_unsafe
        self._lock = threading.Lock()
        self.log_path: Path | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            self.log_path = log_dir / f"run-{stamp}.jsonl"
        if debug_unsafe:
            self.console.write(f"WARNING: {DEBUG_UNSAFE_WARNING}\n")
            self.warning("debug_unsafe_enabled")

    def event(self, event: str, level: Level = "info", **fields: object) -> None:
        """Log one event. Raises ``UnsafeLogFieldError`` for any unsafe field or value."""
        clean = _check_fields(event, fields)
        record: dict[str, object] = {"ts": _now_iso(), "level": level, "event": event, **clean}
        with self._lock:
            self._write_file(record)
            self._write_console(level, event, clean)

    def info(self, event: str, **fields: object) -> None:
        """Log an info event."""
        self.event(event, "info", **fields)

    def warning(self, event: str, **fields: object) -> None:
        """Log a warning event."""
        self.event(event, "warning", **fields)

    def error(self, event: str, **fields: object) -> None:
        """Log an error event."""
        self.event(event, "error", **fields)

    def exception(
        self,
        exc: BaseException,
        stage: Stage | None = None,
        file_id: str | None = None,
        event: str = "exception",
    ) -> str:
        """Log an exception by type, error code, stage and file ID only. Returns the code.

        The exception message and traceback are never logged, except that with
        ``debug_unsafe`` the traceback goes to the log file.
        """
        code = exc.code if isinstance(exc, PipelineError) else UNEXPECTED_ERROR
        self.error(
            event,
            exception_type=type(exc).__name__,
            error_code=code,
            stage=stage,
            file_id=file_id,
        )
        if self._debug_unsafe:
            record: dict[str, object] = {
                "ts": _now_iso(),
                "level": "debug",
                "event": "unsafe_traceback",
                "traceback": "".join(traceback.format_exception(exc)),
            }
            with self._lock:
                self._write_file(record)
        return code

    def _write_file(self, record: dict[str, object]) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_console(self, level: Level, event: str, fields: dict[str, FieldValue]) -> None:
        if _LEVELS[level] < self._console_min:
            return
        parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        stamp = datetime.now().strftime("%H:%M:%S")
        self.console.write(f"{stamp} {level.upper():7} {event} {parts}".rstrip() + "\n")


def install_excepthook(logger: RunLogger) -> None:
    """Replace ``sys.excepthook`` so an uncaught crash is logged safely instead of
    printing the exception message and traceback to the console."""

    def hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        code = logger.exception(exc.with_traceback(tb), event="crash")
        logger.console.write(f"murmurs crashed: {exc_type.__name__} ({code}); see logs\n")

    sys.excepthook = hook


# --- Metrics --------------------------------------------------------------------------


class StageTiming(BaseModel):
    """Wall time and outcome of one stage for one file."""

    model_config = ConfigDict(extra="forbid")

    duration_s: float = Field(ge=0)
    status: StageStatus


class ThroughputSummary(BaseModel):
    """Whole-job throughput: audio processed against wall time (P3-04)."""

    model_config = ConfigDict(extra="forbid")

    n_files: int = Field(ge=0)
    """Files that ran at least one stage in this job."""
    audio_s: float = Field(ge=0)
    """Total audio duration of those files."""
    wall_s: float = Field(ge=0)
    """Wall time of the job's stage groups, including model loading."""
    group_wall_s: dict[str, float] = {}
    """Wall time per stage group, including model loading."""
    real_time_factor: float | None = None
    """Processing seconds per second of audio (lower is faster)."""
    audio_hours_per_hour: float | None = None
    """Hours of audio processed per hour of wall time (the design's throughput metric)."""


class Metrics(BaseModel):
    """Contents of ``metrics.json``: per-file, per-stage timings and a job summary.

    (Schema 2 added ``summary``.)
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    job_id: str
    files: dict[str, dict[Stage, StageTiming]] = {}
    summary: ThroughputSummary | None = None


class MetricsRecorder:
    """Collects stage timings and writes them atomically to ``metrics.json``."""

    def __init__(self, job_id: str, clock: Callable[[], float] = time.perf_counter) -> None:
        self.metrics = Metrics(job_id=job_id)
        self._clock = clock

    def record(self, file_id: str, stage: Stage, duration_s: float, status: StageStatus) -> None:
        """Record one stage's timing for one file."""
        timing = StageTiming(duration_s=duration_s, status=status)
        self.metrics.files.setdefault(file_id, {})[stage] = timing

    @contextmanager
    def timed(self, file_id: str, stage: Stage) -> Iterator[None]:
        """Time a block; records ``done`` on success and ``failed`` if it raises."""
        start = self._clock()
        status = StageStatus.FAILED
        try:
            yield
            status = StageStatus.DONE
        finally:
            self.record(file_id, stage, self._clock() - start, status)

    def summarize(
        self, n_files: int, audio_s: float, wall_s: float, group_wall_s: dict[str, float]
    ) -> ThroughputSummary:
        """Record and return the job's throughput summary."""
        summary = ThroughputSummary(
            n_files=n_files,
            audio_s=round(audio_s, 3),
            wall_s=round(wall_s, 3),
            group_wall_s={name: round(t, 3) for name, t in group_wall_s.items()},
            real_time_factor=round(wall_s / audio_s, 6) if audio_s > 0 else None,
            audio_hours_per_hour=round(audio_s / wall_s, 4) if wall_s > 0 else None,
        )
        self.metrics.summary = summary
        return summary

    def write(self, path: Path) -> None:
        """Atomically write ``metrics.json``."""
        with atomic_writer(path) as f:
            f.write(self.metrics.model_dump_json(indent=2))


# --- Progress -------------------------------------------------------------------------


class FileOutcome(StrEnum):
    """How a file ended in a run."""

    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProgressCallback(Protocol):
    """Progress hooks the runner calls. The CLI uses them now; the GUI will later."""

    def on_stage_start(self, file_id: str, stage: Stage) -> None:
        """A stage is starting for a file."""
        ...

    def on_stage_end(
        self, file_id: str, stage: Stage, status: StageStatus, duration_s: float
    ) -> None:
        """A stage finished (done, failed or skipped) for a file."""
        ...

    def on_file_done(self, file_id: str, outcome: FileOutcome) -> None:
        """A file has no more stages to run in this job."""
        ...


class NullProgress:
    """Ignores all progress events."""

    def on_stage_start(self, file_id: str, stage: Stage) -> None:
        """Do nothing."""

    def on_stage_end(
        self, file_id: str, stage: Stage, status: StageStatus, duration_s: float
    ) -> None:
        """Do nothing."""

    def on_file_done(self, file_id: str, outcome: FileOutcome) -> None:
        """Do nothing."""


class LoggerProgress:
    """Reports progress as log events, which also reach the console."""

    def __init__(self, logger: RunLogger) -> None:
        self._logger = logger

    def on_stage_start(self, file_id: str, stage: Stage) -> None:
        """Log ``stage_start`` (debug level, so file only by default)."""
        self._logger.event("stage_start", "debug", file_id=file_id, stage=stage)

    def on_stage_end(
        self, file_id: str, stage: Stage, status: StageStatus, duration_s: float
    ) -> None:
        """Log ``stage_end`` with status and duration."""
        self._logger.info(
            "stage_end",
            file_id=file_id,
            stage=stage,
            status=status,
            duration_s=round(duration_s, 3),
        )

    def on_file_done(self, file_id: str, outcome: FileOutcome) -> None:
        """Log ``file_done`` with the outcome."""
        self._logger.info("file_done", file_id=file_id, status=outcome)
