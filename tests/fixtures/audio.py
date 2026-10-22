"""Generate tiny audio fixtures at test time. No audio files are committed."""

import io
import math
import struct
import wave
from pathlib import Path


def tone_wav_bytes(
    seconds: float = 1.0,
    frequency: float = 440.0,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> bytes:
    """A 16-bit sine-tone WAV file as bytes. ``frequency=0`` gives silence.

    Different frequencies give different file contents, and so different file IDs.
    """
    n = round(seconds * sample_rate)
    frames = b"".join(
        struct.pack("<h", round(0.3 * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate)))
        * channels
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(frames)
    return buf.getvalue()


def write_tone(
    path: Path,
    seconds: float = 1.0,
    frequency: float = 440.0,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    """Write a 16-bit sine-tone WAV file. ``frequency=0`` writes silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tone_wav_bytes(seconds, frequency, sample_rate, channels))
    return path


def write_tone_m4a(
    path: Path,
    seconds: float = 1.0,
    frequency: float = 440.0,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    """Write a sine-tone AAC .m4a file with torchcodec's encoder."""
    import torch
    from torchcodec.encoders import AudioEncoder

    t = torch.arange(round(seconds * sample_rate)) / sample_rate
    samples = (0.3 * torch.sin(2 * math.pi * frequency * t)).repeat(channels, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    AudioEncoder(samples, sample_rate=sample_rate).to_file(path)
    return path
