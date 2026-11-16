"""NER with ``OpenMed/privacy-filter-nemotron-v2`` (P3-01), a token classifier with 55 PII
labels in the BIOES scheme.

``transformers`` is imported when the detector is created, never at module import time.

The model accepts very long inputs, but GPU memory grows by about 2 GB per thousand
tokens; next to the vLLM server that is too much for a whole interview, so text runs in
overlapping windows and each token keeps the prediction from the window where it sits
most centrally. Token offsets include leading spaces and tokens can split words, so
spans are trimmed and widened to whole words, then overlapping spans are merged.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from murmurs.backends.device import resolve_device
from murmurs.config import JobConfig
from murmurs.errors import BackendSetupError, PipelineError
from murmurs.models import DetectorSource, RedactionSpan
from murmurs.offline import WeightSpec

MODEL_ID = "OpenMed/privacy-filter-nemotron-v2"
WEIGHT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "label_space_fine_v1.json",
)
SPEC = WeightSpec(MODEL_ID, allow_patterns=WEIGHT_FILES)

WINDOW_TOKENS = 2048
OVERLAP_TOKENS = 256
"""Tokens shared by neighbouring windows; each side keeps the half nearest its centre."""
NER_FAILED = "NER_FAILED"
"""``detect`` error code: the model failed on this file."""

_WORD_CHAR = re.compile(r"\w")


def openmed_weights(config: JobConfig) -> Sequence[WeightSpec]:
    """The model files to download and check."""
    return [SPEC]


@dataclass(frozen=True)
class RawSpan:
    """A decoded span before it becomes a ``RedactionSpan``."""

    start: int
    end: int
    label: str
    confidence: float


@dataclass(frozen=True)
class Window:
    """Tokens ``start``..``end`` go through the model; ``keep_start``..``keep_end`` keep
    this window's predictions."""

    start: int
    end: int
    keep_start: int
    keep_end: int


def windows(
    n_tokens: int, size: int = WINDOW_TOKENS, overlap: int = OVERLAP_TOKENS
) -> list[Window]:
    """Overlapping windows covering ``n_tokens``; every token is kept by exactly one."""
    if n_tokens <= size:
        return [Window(0, n_tokens, 0, n_tokens)]
    step = size - overlap
    starts = list(range(0, n_tokens - overlap, step))
    result = []
    for i, start in enumerate(starts):
        end = min(start + size, n_tokens)
        keep_start = 0 if i == 0 else start + overlap // 2
        keep_end = n_tokens if i == len(starts) - 1 else start + step + overlap // 2
        result.append(Window(start, end, keep_start, keep_end))
    return result


def decode_bioes(
    offsets: Sequence[tuple[int, int]], labels: Sequence[str], confidences: Sequence[float]
) -> list[RawSpan]:
    """Turn per-token BIOES labels into spans. A span starts at B or S (or at any tag
    whose entity differs from the open span) and ends at E or S."""
    spans: list[RawSpan] = []
    open_span: list[Any] | None = None  # start, end, label, confidences
    for (start, end), tagged, confidence in zip(offsets, labels, confidences, strict=True):
        if tagged == "O" or start >= end:
            open_span = None
            continue
        tag, _, label = tagged.partition("-")
        if tag in ("B", "S") or open_span is None or open_span[2] != label:
            open_span = [start, end, label, [confidence]]
            spans.append(RawSpan(start, end, label, confidence))
        else:
            open_span[1] = end
            open_span[3].append(confidence)
            spans[-1] = RawSpan(open_span[0], end, label, sum(open_span[3]) / len(open_span[3]))
        if tag in ("E", "S"):
            open_span = None
    return spans


def snap_to_words(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim whitespace and punctuation-only edges, then widen to whole words."""
    while start < end and not _WORD_CHAR.match(text[start]):
        start += 1
    while end > start and not _WORD_CHAR.match(text[end - 1]):
        end -= 1
    if start >= end:
        return None
    while start > 0 and _WORD_CHAR.match(text[start - 1]):
        start -= 1
    while end < len(text) and _WORD_CHAR.match(text[end]):
        end += 1
    return start, end


def clean_spans(text: str, spans: Sequence[RawSpan]) -> list[RawSpan]:
    """Snap spans to whole words, then merge overlapping ones: same label merges into
    one span; different labels keep the more confident."""
    snapped = []
    for span in spans:
        bounds = snap_to_words(text, span.start, span.end)
        if bounds is not None:
            snapped.append(RawSpan(bounds[0], bounds[1], span.label, span.confidence))
    snapped.sort(key=lambda s: (s.start, s.end))
    merged: list[RawSpan] = []
    for span in snapped:
        if merged and span.start < merged[-1].end:
            last = merged[-1]
            if span.label == last.label:
                merged[-1] = RawSpan(
                    last.start,
                    max(last.end, span.end),
                    last.label,
                    max(last.confidence, span.confidence),
                )
            elif span.confidence > last.confidence:
                merged[-1] = span
            continue
        merged.append(span)
    return merged


class OpenMedNER:
    """``NERDetector`` backed by the OpenMed privacy filter, loaded once from the cache."""

    def __init__(self, config: JobConfig) -> None:
        import torch
        import transformers

        tf: Any = transformers
        tf.utils.logging.set_verbosity_error()
        tf.utils.logging.disable_progress_bar()
        self.device, self.device_fallback = resolve_device(config.compute.device)
        """``device_fallback`` is true when ``auto`` found no CUDA device; the runner logs it."""
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            self._tokenizer: Any = tf.AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            model: Any = tf.AutoModelForTokenClassification.from_pretrained(
                MODEL_ID, local_files_only=True, dtype=dtype
            )
            self._model: Any = model.to(self.device).eval()
        except Exception:
            raise BackendSetupError(
                f"could not load {MODEL_ID} from the local model cache"
            ) from None
        self.model_version = f"{MODEL_ID} (transformers {transformers.__version__})"

    def detect(self, text: str, file_id: str) -> list[RedactionSpan]:
        """Spans over ``text`` with the model's raw label as ``entity_type``."""
        if not text.strip():
            return []
        try:
            raw = self._predict(text)
        except Exception:
            raise PipelineError(NER_FAILED) from None
        return [
            RedactionSpan(
                id=f"ner-{i}",
                start=span.start,
                end=span.end,
                entity_type=span.label,
                tag=f"[{span.label.upper()}]",
                confidence=round(min(1.0, max(0.0, span.confidence)), 4),
                sources=[DetectorSource.NER],
            )
            for i, span in enumerate(clean_spans(text, raw))
        ]

    def _predict(self, text: str) -> list[RawSpan]:
        import torch

        encoded = self._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids: list[int] = encoded["input_ids"]
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        labels: list[str] = ["O"] * len(ids)
        confidences: list[float] = [0.0] * len(ids)
        id2label = self._model.config.id2label
        for window in windows(len(ids)):
            batch = torch.tensor([ids[window.start : window.end]], device=self.device)
            with torch.inference_mode():
                logits = self._model(input_ids=batch).logits[0].float()
            probs = logits.softmax(-1)
            best_prob, best_id = probs.max(-1)
            for i in range(window.keep_start, window.keep_end):
                j = i - window.start
                labels[i] = id2label[int(best_id[j])]
                confidences[i] = float(best_prob[j])
        return decode_bioes(offsets, labels, confidences)

    def close(self) -> None:
        """Drop the model and return its GPU memory to the device."""
        import gc

        import torch

        self._model = None
        self._tokenizer = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


if TYPE_CHECKING:
    from murmurs.backends import NERDetector

    def _check(config: JobConfig) -> NERDetector:
        # Static proof (checked by mypy) that the detector satisfies its protocol.
        return OpenMedNER(config)
