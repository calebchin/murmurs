"""Run manifest contents and sample selection."""

from __future__ import annotations

import random
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypeVar

from murmurs import __version__
from murmurs.config import JobConfig
from murmurs.models import FileStatus, RunManifest

KEY_DEPENDENCIES = (
    "pydantic",
    "pyyaml",
    "numpy",
    "typer",
    "torch",
    "torchcodec",
    "pyannote.audio",
    "crisperwhisper",
    "ctranslate2",
    "transformers",
)
"""Packages whose versions are recorded when installed."""

SAMPLE_SUFFIX = "-sample"

T = TypeVar("T")


def sample_job_id(job_id: str) -> str:
    """Job ID for a sample run, kept apart from the full run's outputs."""
    return f"{job_id}{SAMPLE_SUFFIX}"


def select_sample(items: Sequence[T], keys: Sequence[str], n_files: int, seed: int) -> list[T]:
    """Pick ``n_files`` items reproducibly.

    Items are ordered by ``keys`` (file IDs, which come from file contents) before the
    seeded draw, so the same seed and inputs always give the same picks, whatever the
    file names. Returns all items, in key order, if ``n_files`` covers them.
    """
    ordered = [item for _, item in sorted(zip(keys, items, strict=True), key=lambda p: p[0])]
    if n_files >= len(ordered):
        return ordered
    picked = set(random.Random(seed).sample(range(len(ordered)), n_files))
    return [item for i, item in enumerate(ordered) if i in picked]


def dependency_versions() -> dict[str, str]:
    """Installed versions of ``KEY_DEPENDENCIES``; packages not installed are left out."""
    found = {}
    for name in KEY_DEPENDENCIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def code_version(source_dir: Path | None = None) -> str | None:
    """Git commit of the murmurs source, with ``-dirty`` if it has uncommitted changes.

    Returns None when the code is not in a git checkout or git is unavailable.
    """
    where = source_dir if source_dir is not None else Path(__file__).parent
    try:
        commit = _git(where, "rev-parse", "HEAD")
        dirty = _git(where, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def _git(where: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(where), *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return result.stdout.strip()


def start_manifest(
    job_id: str,
    config: JobConfig,
    model_versions: dict[str, str],
    input_hashes: dict[str, str],
) -> RunManifest:
    """The manifest written when a job starts."""
    return RunManifest(
        job_id=job_id,
        started_at=datetime.now(UTC),
        config=config,
        package_version=__version__,
        dependency_versions=dependency_versions(),
        model_versions=model_versions,
        code_version=code_version(),
        input_hashes=input_hashes,
    )


def finish_manifest(
    manifest: RunManifest, statuses: list[FileStatus], completed: bool
) -> RunManifest:
    """The manifest updated at job end. ``finished_at`` stays empty if interrupted."""
    return manifest.model_copy(
        update={
            "finished_at": datetime.now(UTC) if completed else None,
            "file_statuses": statuses,
        }
    )
