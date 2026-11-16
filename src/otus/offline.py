"""Offline operation: forced Hugging Face offline mode and cached model weights.

Runs never use the network. Model weights are fetched once, while connected, with
``murmurs models download``. At run time the runner calls ``enforce_offline`` and then
checks that every weight the job needs is cached, before any file is processed.

``huggingface_hub`` reads its offline setting when it is first imported, so this module
imports it only inside functions, and real backends must import model libraries only
when they are created (after ``enforce_offline``), never when the registry is imported.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from murmurs.errors import SetupError

OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
"""Environment variables that put Hugging Face libraries in offline mode."""


@dataclass(frozen=True)
class WeightSpec:
    """One Hugging Face model repo a backend needs in the local cache.

    ``revision`` is a branch, tag or commit hash (``None`` means the default branch).
    ``allow_patterns`` limits which files are downloaded (``None`` means all).
    """

    repo_id: str
    revision: str | None = None
    allow_patterns: tuple[str, ...] | None = None

    @property
    def label(self) -> str:
        """``repo_id``, or ``repo_id@revision`` when a revision is pinned."""
        return self.repo_id if self.revision is None else f"{self.repo_id}@{self.revision}"


@dataclass(frozen=True)
class DownloadedWeight:
    """Result of downloading one ``WeightSpec``."""

    revision: str
    """Commit hash of the cached snapshot."""
    size_bytes: int


class MissingWeightsError(SetupError):
    """Weights a job needs are not in the local cache; the job cannot start."""

    def __init__(self, missing: Sequence[WeightSpec]) -> None:
        self.missing = list(missing)
        names = ", ".join(spec.label for spec in self.missing)
        super().__init__(
            f"model weights not in the local cache: {names}. While connected to the "
            "internet, run `murmurs models download --config <this job's config>`."
        )


class WeightDownloadError(Exception):
    """Downloading one weight failed. ``str(exc)`` is a short message written by murmurs;
    library exception text is never included."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def enforce_offline() -> None:
    """Put Hugging Face libraries in offline mode for the rest of this process.

    Sets ``OFFLINE_ENV``. If ``huggingface_hub`` was already imported, its cached
    offline flag is updated too; other libraries only see the setting if they are
    imported after this call.
    """
    os.environ.update(OFFLINE_ENV)
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_OFFLINE = True  # type: ignore[attr-defined]


def cached_snapshot(spec: WeightSpec) -> Path | None:
    """Folder of the complete cached snapshot for ``spec``, or None if it is missing,
    incomplete or empty. Never uses the network."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        folder = snapshot_download(
            spec.repo_id,
            revision=spec.revision,
            allow_patterns=_patterns(spec),
            local_files_only=True,
        )
    except LocalEntryNotFoundError:  # includes IncompleteSnapshotError
        return None
    if not isinstance(folder, str):
        return None
    path = Path(folder)
    if not any(path.iterdir()):
        return None
    return path


def cached_revision(spec: WeightSpec) -> str | None:
    """Commit hash of the complete cached snapshot for ``spec``, or None if it is missing,
    incomplete or empty. Never uses the network."""
    path = cached_snapshot(spec)
    return None if path is None else path.name


@dataclass(frozen=True, eq=False)
class PreparedModel:
    """A model derived from downloaded weights, such as a format conversion.

    ``murmurs models download`` calls ``prepare`` after downloading the weights; it runs
    offline and may be slow. ``is_ready`` is a fast, offline check of its output. Runs
    and ``murmurs models check`` treat a model that is not ready as missing.
    """

    label: str
    """Short name shown to the user; also the deduplication key."""
    is_ready: Callable[[], bool]
    prepare: Callable[[], None]


class UnpreparedModelError(SetupError):
    """Downloaded weights still need their one-time preparation; the job cannot start."""

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        super().__init__(
            f"models not prepared: {', '.join(self.labels)}. While the weights are cached, "
            "run `murmurs models download --config <this job's config>` to prepare them."
        )


def check_prepared(models: Sequence[PreparedModel]) -> None:
    """Raise ``UnpreparedModelError`` naming every model that is not ready."""
    missing = [m.label for m in models if not m.is_ready()]
    if missing:
        raise UnpreparedModelError(missing)


def check_cached(specs: Sequence[WeightSpec]) -> dict[str, str]:
    """Commit hash of each cached weight, keyed by ``WeightSpec.label``.

    Raises ``MissingWeightsError`` naming every weight that is not fully cached.
    """
    found: dict[str, str] = {}
    missing: list[WeightSpec] = []
    for spec in specs:
        revision = cached_revision(spec)
        if revision is None:
            missing.append(spec)
        else:
            found[spec.label] = revision
    if missing:
        raise MissingWeightsError(missing)
    return found


def download_weight(spec: WeightSpec) -> DownloadedWeight:
    """Download ``spec`` into the local Hugging Face cache (needs network access).

    ``huggingface_hub`` reads the token for gated models from ``HF_TOKEN`` or a saved
    ``hf auth login``; murmurs never reads or prints it. Raises ``WeightDownloadError``.
    """
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if constants.is_offline_mode():
        raise WeightDownloadError(
            "OFFLINE_MODE_SET",
            "offline mode is on (HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set); "
            "unset it to download",
        )
    try:
        folder = snapshot_download(
            spec.repo_id, revision=spec.revision, allow_patterns=_patterns(spec)
        )
    except GatedRepoError:
        raise WeightDownloadError(
            "GATED_REPO",
            "access denied: set HF_TOKEN to a token for an account that has accepted "
            "this model's terms on huggingface.co",
        ) from None
    except RepositoryNotFoundError:
        raise WeightDownloadError(
            "REPO_NOT_FOUND", "repository not found, or private and no valid token is set"
        ) from None
    except RevisionNotFoundError:
        raise WeightDownloadError("REVISION_NOT_FOUND", "revision not found") from None
    except OSError as exc:
        raise WeightDownloadError(
            "DOWNLOAD_FAILED", f"download failed ({type(exc).__name__})"
        ) from None
    if not isinstance(folder, str):
        raise WeightDownloadError("DOWNLOAD_FAILED", "download returned no snapshot")
    path = Path(folder)
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return DownloadedWeight(revision=path.name, size_bytes=size)


def _patterns(spec: WeightSpec) -> list[str] | None:
    return None if spec.allow_patterns is None else list(spec.allow_patterns)
