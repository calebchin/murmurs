"""Deterministic stub backends for tests. No GPU, no downloads, no model libraries.

Every stub records the file IDs it was called with in ``calls`` and can be told to fail
on chosen file IDs, raising ``PipelineError("STUB_FAILURE", detail=fail_detail)``, or to
raise ``KeyboardInterrupt`` once for chosen file IDs.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from murmurs.backends import M, Waveform
from murmurs.errors import PipelineError
from murmurs.models import (
    DetectorSource,
    Diarization,
    ErrorInfo,
    IngestErrorCode,
    RedactionSpan,
    Segment,
    SpeakerTurn,
    Transcript,
    TranscriptMode,
    ValidationResult,
    Word,
)

STUB_VERSION = "stub"
STUB_SAMPLE_RATE = 16_000
STUB_FAILURE = "STUB_FAILURE"


class _Stub:
    """Shared call recording and fail-on-file-ID behavior."""

    model_version = STUB_VERSION

    def __init__(
        self,
        fail_on: Collection[str] = (),
        fail_detail: str = "stub failure",
        interrupt_once: Collection[str] = (),
    ) -> None:
        self.fail_on = frozenset(fail_on)
        self.fail_detail = fail_detail
        self.interrupt_pending = set(interrupt_once)
        """File IDs that raise ``KeyboardInterrupt`` on their first call (simulates Ctrl-C)."""
        self.calls: list[str] = []

    def _record(self, file_id: str) -> None:
        self.calls.append(file_id)
        if file_id in self.interrupt_pending:
            self.interrupt_pending.discard(file_id)
            raise KeyboardInterrupt
        if file_id in self.fail_on:
            raise PipelineError(STUB_FAILURE, detail=self.fail_detail)


class StubAudioLoader(_Stub):
    """Reports every non-empty file as valid and "decodes" it to silence."""

    def __init__(self, duration_s: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.duration_s = duration_s

    def probe(self, path: Path, file_id: str) -> ValidationResult:
        """Stat the file only; never reads or modifies it."""
        self._record(file_id)
        if path.stat().st_size == 0:
            return ValidationResult(
                file_id=file_id,
                path=str(path),
                valid=False,
                error=ErrorInfo(code=IngestErrorCode.EMPTY),
            )
        return ValidationResult(
            file_id=file_id,
            path=str(path),
            format=path.suffix.lstrip(".") or None,
            duration_s=self.duration_s,
            sample_rate=STUB_SAMPLE_RATE,
            channels=1,
            valid=True,
        )

    def load(self, path: Path, file_id: str) -> Waveform:
        """Return ``duration_s`` seconds of silence."""
        self._record(file_id)
        samples = np.zeros(round(self.duration_s * STUB_SAMPLE_RATE), dtype=np.float32)
        return Waveform(samples=samples, sample_rate=STUB_SAMPLE_RATE)


class StubDiarizer(_Stub):
    """Alternates two speakers in fixed-length turns."""

    def __init__(self, turn_s: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.turn_s = turn_s

    def diarize(self, waveform: Waveform, file_id: str) -> Diarization:
        """Turns of ``turn_s`` seconds, SPEAKER_00 and SPEAKER_01 in turn."""
        self._record(file_id)
        turns = []
        start, i = 0.0, 0
        while start < waveform.duration_s:
            end = min(start + self.turn_s, waveform.duration_s)
            turns.append(SpeakerTurn(speaker=f"SPEAKER_{i % 2:02d}", start=start, end=end))
            start, i = end, i + 1
        return Diarization(file_id=file_id, turns=turns)


class StubTranscriber(_Stub):
    """One segment of fixed text, words spread evenly over the audio.

    Put a sentinel string in ``text`` to check that transcript content never leaks.
    The verbatim transcript starts with an extra "um".
    """

    def __init__(self, text: str = "this is a stub transcript", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text = text

    def transcribe(
        self, waveform: Waveform, file_id: str, modes: Sequence[TranscriptMode]
    ) -> dict[TranscriptMode, Transcript]:
        """Return one transcript per requested mode."""
        self._record(file_id)
        return {mode: self._transcript(waveform, file_id, mode) for mode in modes}

    def _transcript(self, waveform: Waveform, file_id: str, mode: TranscriptMode) -> Transcript:
        tokens = self.text.split()
        if mode is TranscriptMode.VERBATIM:
            tokens = ["um", *tokens]
        duration = waveform.duration_s
        step = duration / len(tokens) if tokens else 0.0
        words = [
            Word(text=token, start=i * step, end=min((i + 1) * step, duration))
            for i, token in enumerate(tokens)
        ]
        segment = Segment(start=0.0, end=duration, text=" ".join(tokens), words=words)
        return Transcript(file_id=file_id, mode=mode, segments=[segment] if words else [])


class StubNERDetector(_Stub):
    """Finds exact occurrences of configured strings and labels them."""

    def __init__(self, targets: Mapping[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.targets = dict(targets or {})
        """Surface string -> raw NER label, e.g. ``{"Alice": "first_name"}``."""

    def detect(self, text: str, file_id: str) -> list[RedactionSpan]:
        """One span per occurrence, in order of position, confidence 1.0."""
        self._record(file_id)
        found = sorted(
            (match.start(), match.end(), label)
            for surface, label in self.targets.items()
            for match in re.finditer(re.escape(surface), text)
        )
        return [
            RedactionSpan(
                id=f"ner-{i}",
                start=start,
                end=end,
                entity_type=label,
                tag=f"[{label.upper()}]",
                confidence=1.0,
                sources=[DetectorSource.NER],
            )
            for i, (start, end, label) in enumerate(found)
        ]


class StubLLMClient(_Stub):
    """Returns a fixed response, validated as the requested model."""

    def __init__(self, response: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.response = dict(response or {})

    def generate(
        self, prompt: str, response_model: type[M], file_id: str, system: str | None = None
    ) -> M:
        """Ignore the prompt and validate ``response`` as ``response_model``."""
        self._record(file_id)
        return response_model.model_validate(self.response)


if TYPE_CHECKING:
    from murmurs.backends import AudioLoader, Diarizer, LLMClient, NERDetector, Transcriber

    # Static proof (checked by mypy) that each stub satisfies its protocol.
    _audio: AudioLoader = StubAudioLoader()
    _diarizer: Diarizer = StubDiarizer()
    _transcriber: Transcriber = StubTranscriber()
    _ner: NERDetector = StubNERDetector()
    _llm: LLMClient = StubLLMClient()
