"""The torchcodec audio loader (the ``"default"`` audio loader).

``torchcodec`` (and through it ``torch``) is imported when the loader is created, never
when this module is imported, so importing the registry stays cheap.

The probe reads only container headers plus two short windows of audio at the start
and end of the stream; it never decodes a whole file. Library exceptions are mapped to
fixed error codes and their text is discarded, since it can quote paths or file contents.

``load`` decodes straight from the original file into memory; no audio is ever written
to disk, including the temp directory.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from murmurs.backends import Waveform
from murmurs.errors import PipelineError
from murmurs.models import ErrorInfo, IngestErrorCode, ValidationResult

TARGET_SAMPLE_RATE = 16_000
"""Sample rate of every decoded waveform; both diarization and transcription expect it."""

DECODE_FAILED = "DECODE_FAILED"
"""``load`` error code: the file could not be decoded."""
NO_SAMPLES = "NO_SAMPLES"
"""``load`` error code: the file decoded to zero samples."""

PROBE_WINDOW_S = 0.5
"""Length of audio decoded at each end of the stream to detect corruption."""

_RIFF_UNKNOWN_SIZES = (0, 0xFFFFFFFF)
"""``data`` chunk sizes that streaming writers use when the length is not known."""


class TorchcodecAudioLoader:
    """Probes and decodes audio with ``torchcodec`` and the system ffmpeg."""

    def __init__(self, audio_extensions: Sequence[str]) -> None:
        import torchcodec

        self.model_version = f"torchcodec {torchcodec.__version__}"
        self.audio_extensions = frozenset(ext.lower() for ext in audio_extensions)

    def probe(self, path: Path, file_id: str) -> ValidationResult:
        """Check a file's format and integrity without decoding all of it.

        Opens the file read-only and never modifies it.
        """
        fmt = path.suffix.lower().lstrip(".") or None

        def invalid(code: IngestErrorCode, **found: Any) -> ValidationResult:
            return ValidationResult(
                file_id=file_id,
                path=str(path),
                format=fmt,
                valid=False,
                error=ErrorInfo(code=code),
                **found,
            )

        if path.suffix.lower() not in self.audio_extensions:
            return invalid(IngestErrorCode.UNSUPPORTED_FORMAT)
        if path.stat().st_size == 0:
            return invalid(IngestErrorCode.EMPTY)
        if _wav_data_truncated(path):
            return invalid(IngestErrorCode.TRUNCATED)

        from torchcodec.decoders import (
            AudioDecoder,
            AudioStreamHeaderMetadata,
            get_container_metadata,
        )

        try:
            container = get_container_metadata(path)
        except Exception:
            return invalid(IngestErrorCode.UNREADABLE)
        index = container.best_audio_stream_index
        stream = None if index is None else container.streams[index]
        if index is None or not isinstance(stream, AudioStreamHeaderMetadata):
            return invalid(IngestErrorCode.NO_AUDIO_STREAM)
        found: dict[str, Any] = {
            "codec": stream.codec,
            "sample_rate": stream.sample_rate,
            "channels": stream.num_channels,
        }
        duration = stream.duration_seconds_from_header or container.duration_seconds_from_header
        if duration is None:
            return invalid(IngestErrorCode.UNREADABLE, **found)
        found["duration_s"] = duration
        if duration <= 0:
            return invalid(IngestErrorCode.EMPTY, **found)

        window = min(PROBE_WINDOW_S, duration)
        try:
            decoder = AudioDecoder(path, stream_index=index)
        except Exception:
            return invalid(IngestErrorCode.UNREADABLE, **found)
        if not _decodes(decoder, 0.0, window):
            return invalid(IngestErrorCode.CORRUPT, **found)
        if not _decodes(decoder, duration - window, duration):
            return invalid(IngestErrorCode.TRUNCATED, **found)
        return ValidationResult(file_id=file_id, path=str(path), format=fmt, valid=True, **found)

    def close(self) -> None:
        """Nothing to release: decoders are opened and closed per call."""

    def load(self, path: Path, file_id: str) -> Waveform:
        """Decode a whole file into a 16 kHz mono float32 waveform in memory.

        torchcodec resamples while decoding and keeps the source channels; channels are
        then averaged here (ffmpeg's own downmix scales stereo up by about 1.41).
        """
        from torchcodec.decoders import AudioDecoder

        try:
            decoded = AudioDecoder(path, sample_rate=TARGET_SAMPLE_RATE).get_all_samples()
        except Exception:
            raise PipelineError(DECODE_FAILED) from None
        channels = decoded.data.numpy()
        if channels.ndim != 2 or channels.shape[1] == 0:
            raise PipelineError(NO_SAMPLES)
        mono = channels[0] if channels.shape[0] == 1 else channels.mean(axis=0, dtype=np.float32)
        return Waveform(
            samples=np.ascontiguousarray(mono, dtype=np.float32), sample_rate=TARGET_SAMPLE_RATE
        )


def _decodes(decoder: Any, start_s: float, stop_s: float) -> bool:
    """Whether ``[start_s, stop_s)`` decodes to at least one sample."""
    try:
        samples = decoder.get_samples_played_in_range(start_s, stop_s)
    except Exception:
        return False
    return int(samples.data.shape[-1]) > 0


def _wav_data_truncated(path: Path) -> bool:
    """Whether a RIFF/WAVE file is shorter than its ``data`` chunk header says.

    ffmpeg silently shortens the duration of a truncated WAV file to match its size, so
    this is checked from the header directly. Files that are not RIFF/WAVE, and WAV
    files written with an unknown data size, return ``False`` and are left to the decoder.
    """
    size = path.stat().st_size
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return False
        pos = 12
        while True:
            f.seek(pos)
            chunk = f.read(8)
            if len(chunk) < 8:
                return True  # the file ends before any data chunk
            chunk_id, chunk_size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if chunk_id == b"data":
                if chunk_size in _RIFF_UNKNOWN_SIZES:
                    return False
                return bool(pos + 8 + chunk_size > size)
            pos += 8 + chunk_size + (chunk_size & 1)  # chunks are padded to even sizes


if TYPE_CHECKING:
    from murmurs.backends import AudioLoader

    # Static proof (checked by mypy) that the loader satisfies its protocol.
    _audio: AudioLoader = TorchcodecAudioLoader([])
