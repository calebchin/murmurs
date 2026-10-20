"""Generate tiny audio fixtures at test time. No audio files are committed."""

import math
import struct
import wave
from pathlib import Path


def write_tone(
    path: Path, seconds: float = 1.0, frequency: float = 440.0, sample_rate: int = 16_000
) -> Path:
    """Write a mono 16-bit sine-tone WAV file. ``frequency=0`` writes silence."""
    n = round(seconds * sample_rate)
    samples = (
        round(0.3 * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate)) for i in range(n)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return path
