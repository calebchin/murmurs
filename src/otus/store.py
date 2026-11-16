"""Artifact store: output layout and atomic writes.

Layout under ``<output_dir>/<job_id>/``::

    manifest.json
    files/<file_id>/status.json
    files/<file_id>/<stage>[.<variant>].json
    files/<file_id>/<stage>[.<variant>].txt     (plain-text transcript export)
    files/<file_id>/<stage>[.<variant>].words.tsv  (one word per line, with times)
    transcripts/<input name>-<file id>/          (readable copies of finished transcripts)

Every write is atomic and is refused if it would land inside the input directory.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TypeVar

from pydantic import BaseModel

from murmurs.config import Stage
from murmurs.models import FileStatus, RunManifest, Transcript

FILE_ID_LENGTH = 16
"""Hex characters of the content hash used as a file ID."""

_HASH_CHUNK_BYTES = 1 << 20
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")

M = TypeVar("M", bound=BaseModel)


class StoreError(Exception):
    """An artifact could not be written or read safely."""


class OutputInsideInputError(StoreError):
    """The output location is inside the input directory."""


def hash_file(path: Path) -> str:
    """SHA-256 of a file's contents as hex. Opens the file read-only."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def file_id_from_hash(content_hash: str) -> str:
    """File ID for a content hash, so renaming an input does not change its ID."""
    return content_hash[:FILE_ID_LENGTH]


@contextmanager
def atomic_writer(path: Path) -> Iterator[IO[str]]:
    """Open a temp file next to ``path`` for writing; rename it over ``path`` on success.

    If the block raises, the temp file is removed and ``path`` is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def canonical_text(transcript: Transcript) -> str:
    """The transcript's canonical text (D9): segment texts joined with newlines.

    Deidentification spans and evaluation labels are character offsets into this text.
    Speaker labels and timestamps are not part of it; they are added only for display.
    """
    return "\n".join(segment.text for segment in transcript.segments)


def transcript_to_text(transcript: Transcript) -> str:
    """Readable rendering: one line per segment, with start time and speaker if known."""
    lines = []
    for segment in transcript.segments:
        minutes, seconds = divmod(segment.start, 60)
        stamp = f"[{int(minutes):02d}:{seconds:04.1f}]"
        speaker = f" {segment.speaker}:" if segment.speaker else ""
        lines.append(f"{stamp}{speaker} {segment.text}")
    return "\n".join(lines) + ("\n" if lines else "")


WORDS_TSV_HEADER = "start\tend\tspeaker\tword"
_TSV_ESCAPES = str.maketrans({"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r"})


def transcript_to_words_tsv(transcript: Transcript) -> str:
    """One word per line: start and end in seconds (3 decimals), speaker, word.

    Backslashes, tabs and line breaks inside a word are escaped (``\\``, ``\t``,
    ``\n``, ``\r``) so they cannot break the columns or rows.
    """
    lines = [WORDS_TSV_HEADER]
    for segment in transcript.segments:
        for word in segment.words:
            text = word.text.translate(_TSV_ESCAPES)
            lines.append(f"{word.start:.3f}\t{word.end:.3f}\t{word.speaker or ''}\t{text}")
    return "\n".join(lines) + "\n"


class ArtifactStore:
    """Reads and writes one job's artifacts under ``<output_dir>/<job_id>/``.

    Raises ``OutputInsideInputError`` on construction if ``output_dir`` is inside (or
    equal to) ``input_dir``, so the problem surfaces before any processing.
    """

    def __init__(self, output_dir: Path, input_dir: Path, job_id: str) -> None:
        self._input_dir = input_dir.resolve()
        resolved_output = output_dir.resolve()
        if resolved_output.is_relative_to(self._input_dir):
            raise OutputInsideInputError(
                f"output_dir {output_dir} is inside input_dir {input_dir}; "
                "choose an output directory outside the input directory"
            )
        self.job_dir = output_dir / _checked_name(job_id, "job_id")

    @property
    def manifest_path(self) -> Path:
        """Path of this job's run manifest."""
        return self.job_dir / "manifest.json"

    def file_dir(self, file_id: str) -> Path:
        """Directory holding one input file's artifacts."""
        return self.job_dir / "files" / _checked_name(file_id, "file_id")

    def artifact_path(
        self, file_id: str, stage: Stage, variant: str | None = None, suffix: str = ".json"
    ) -> Path:
        """Path of a stage artifact, e.g. ``transcribe.intended.json``."""
        name = (
            stage.value if variant is None else f"{stage.value}.{_checked_name(variant, 'variant')}"
        )
        return self.file_dir(file_id) / f"{name}{suffix}"

    def status_path(self, file_id: str) -> Path:
        """Path of one file's ``status.json``."""
        return self.file_dir(file_id) / "status.json"

    # --- stage artifacts --------------------------------------------------------

    def write(
        self, file_id: str, stage: Stage, artifact: BaseModel, variant: str | None = None
    ) -> Path:
        """Atomically write a stage artifact as JSON and return its path."""
        path = self.artifact_path(file_id, stage, variant)
        self._write_text(path, artifact.model_dump_json(indent=2))
        return path

    def read(self, file_id: str, stage: Stage, model: type[M], variant: str | None = None) -> M:
        """Read a stage artifact as ``model``."""
        return self._read(self.artifact_path(file_id, stage, variant), model)

    def exists(self, file_id: str, stage: Stage, variant: str | None = None) -> bool:
        """Whether a stage artifact has been written."""
        return self.artifact_path(file_id, stage, variant).is_file()

    def write_transcript_text(
        self, file_id: str, stage: Stage, transcript: Transcript, variant: str | None = None
    ) -> Path:
        """Atomically write the plain-text export of a transcript next to its JSON."""
        path = self.artifact_path(file_id, stage, variant, suffix=".txt")
        self._write_text(path, transcript_to_text(transcript))
        return path

    def write_text(
        self, file_id: str, stage: Stage, variant: str, text: str, suffix: str = ".txt"
    ) -> Path:
        """Atomically write a text artifact, e.g. ``detect_phi.intended.clean.txt``."""
        path = self.artifact_path(file_id, stage, variant, suffix=suffix)
        self._write_text(path, text)
        return path

    def write_words_tsv(
        self, file_id: str, stage: Stage, transcript: Transcript, variant: str | None = None
    ) -> Path:
        """Atomically write the word-level TSV of a transcript next to its JSON."""
        path = self.artifact_path(file_id, stage, variant, suffix=".words.tsv")
        self._write_text(path, transcript_to_words_tsv(transcript))
        return path

    # --- job-level exports ----------------------------------------------------------

    @property
    def transcripts_dir(self) -> Path:
        """Folder of readable transcript copies, one folder per input file, nested as
        the inputs are."""
        return self.job_dir / "transcripts"

    def write_export(self, folder: str, name: str, text: str) -> Path:
        """Atomically write ``transcripts/<folder>/<name>``.

        ``folder`` may be nested (parts joined with ``/``); each part must be a safe name,
        so nothing is written outside ``transcripts/``.
        """
        parts = [_checked_name(part, "folder") for part in folder.split("/")]
        path = self.transcripts_dir.joinpath(*parts) / _checked_name(name, "name")
        self._write_text(path, text)
        return path

    # --- status and manifest ------------------------------------------------------

    def write_status(self, status: FileStatus) -> Path:
        """Atomically write a file's ``status.json``."""
        path = self.status_path(status.file_id)
        self._write_text(path, status.model_dump_json(indent=2))
        return path

    def read_status(self, file_id: str) -> FileStatus | None:
        """Read a file's status, or None if it has none yet."""
        path = self.status_path(file_id)
        return self._read(path, FileStatus) if path.is_file() else None

    def write_manifest(self, manifest: RunManifest) -> Path:
        """Atomically write the run manifest."""
        self._write_text(self.manifest_path, manifest.model_dump_json(indent=2))
        return self.manifest_path

    def read_manifest(self) -> RunManifest | None:
        """Read the run manifest, or None if it has not been written."""
        return self._read(self.manifest_path, RunManifest) if self.manifest_path.is_file() else None

    # --- internals ------------------------------------------------------------------

    def _write_text(self, path: Path, text: str) -> None:
        self._refuse_input_dir(path)
        with atomic_writer(path) as f:
            f.write(text)

    def _refuse_input_dir(self, path: Path) -> None:
        if path.resolve().is_relative_to(self._input_dir):
            raise OutputInsideInputError(f"refusing to write {path}: it is inside input_dir")

    @staticmethod
    def _read(path: Path, model: type[M]) -> M:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise StoreError(f"cannot read artifact {path} ({type(exc).__name__})") from None
        return model.model_validate_json(data)


def _checked_name(name: str, what: str) -> str:
    """Reject IDs that are not a single safe path component."""
    if not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise StoreError(f"{what} is not a safe file name")
    return name
