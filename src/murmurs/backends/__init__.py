"""Protocols and implementations for STT, diarization, NER and LLM backends.

Pipeline code depends only on the protocols here. Each method takes and returns types
from ``murmurs.models``, ``Waveform`` or plain Python types, never model-library types.
Backends raise ``murmurs.errors.PipelineError`` for per-file failures. The runner calls
``close`` once a backend is no longer needed in a job (D21).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from murmurs.models import (
    Diarization,
    LLMUsage,
    RedactionSpan,
    Transcript,
    TranscriptMode,
    ValidationResult,
)

M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True, eq=False)
class Waveform:
    """Decoded audio held in memory: mono float32 samples and their sample rate."""

    samples: NDArray[np.float32]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        """Length of the audio in seconds."""
        return len(self.samples) / self.sample_rate


@runtime_checkable
class AudioLoader(Protocol):
    """Probes and decodes input audio files. Never writes audio to disk."""

    model_version: str

    def probe(self, path: Path, file_id: str) -> ValidationResult:
        """Check a file's format and integrity without decoding all of it."""
        ...

    def load(self, path: Path, file_id: str) -> Waveform:
        """Decode a whole file into a 16 kHz mono waveform in memory."""
        ...

    def close(self) -> None:
        """Release the backend's resources (GPU memory, threads, processes). Not used after."""
        ...


@runtime_checkable
class Diarizer(Protocol):
    """Finds who spoke when."""

    model_version: str

    def diarize(self, waveform: Waveform, file_id: str) -> Diarization:
        """Return speaker turns for the whole recording, in time order."""
        ...

    def close(self) -> None:
        """Release the backend's resources (GPU memory, threads, processes). Not used after."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Speech to text with word timestamps."""

    model_version: str

    def transcribe(
        self, waveform: Waveform, file_id: str, modes: Sequence[TranscriptMode]
    ) -> dict[TranscriptMode, Transcript]:
        """Return one transcript per requested mode."""
        ...

    def close(self) -> None:
        """Release the backend's resources (GPU memory, threads, processes). Not used after."""
        ...


@runtime_checkable
class NERDetector(Protocol):
    """Token-classification PHI detector."""

    model_version: str

    def detect(self, text: str, file_id: str) -> list[RedactionSpan]:
        """Return spans over ``text`` with the model's raw label as ``entity_type``."""
        ...

    def close(self) -> None:
        """Release the backend's resources (GPU memory, threads, processes). Not used after."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """A locally run LLM that returns structured output.

    ``generate`` may be called from several threads at once (callers send chunks
    concurrently). It raises ``PipelineError`` with ``LLM_INVALID_OUTPUT`` when output
    stays invalid, ``LLM_OUTPUT_TRUNCATED`` when a response hits the output cap, and
    another code when the request itself fails.
    """

    model_version: str

    @property
    def usage(self) -> LLMUsage:
        """Running totals of requests, tokens and time (counts only)."""
        ...

    def generate(
        self, prompt: str, response_model: type[M], file_id: str, system: str | None = None
    ) -> M:
        """Run ``prompt`` and return the response validated as ``response_model``."""
        ...

    def close(self) -> None:
        """Release the backend's resources (GPU memory, threads, processes). Not used after."""
        ...


__all__ = [
    "AudioLoader",
    "Diarizer",
    "LLMClient",
    "NERDetector",
    "Transcriber",
    "Waveform",
]
