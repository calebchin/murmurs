"""Transcription with CrisperWhisper 2.0 on its CTranslate2 (``ct2``) backend.

Weights are downloaded in Hugging Face format and converted to CTranslate2 format once,
by ``murmurs models download`` (D17). Runs load the converted folder directly, so
CrisperWhisper never looks at the Hugging Face cache or the network.

The ct2 model must be loaded and run on one thread, so the transcriber owns a
single-thread executor for its whole life and sends every model call through it.

Transcripts are built from the model's timed words only (D18): its full ``text`` holds
repetition artifacts at chunk seams that the timed words leave out. Segments end at
sentence punctuation or pauses (D19).

``crisperwhisper`` is imported only when a model is loaded or converted.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from murmurs.backends import Waveform
from murmurs.backends.device import resolve_device
from murmurs.config import JobConfig
from murmurs.errors import BackendSetupError, PipelineError
from murmurs.models import Segment, Transcript, TranscriptMode, Word
from murmurs.offline import PreparedModel, WeightSpec, cached_snapshot

MODEL_ID = "nyralabs/CrisperWhisper2.0_large"
WEIGHT_PATTERNS = ("*.json", "*.txt", "*.safetensors")
"""The Hugging Face-format files CrisperWhisper converts from."""
SPEC = WeightSpec(MODEL_ID, allow_patterns=WEIGHT_PATTERNS)

SAMPLE_RATE = 16_000
STORED_PRECISION = "float16"
"""Precision of the converted model on disk. On CPU it runs as float32 (CTranslate2
converts at load time)."""
READY_MARKER = ".murmurs_ready"
"""Written last into a converted folder, so a half-finished conversion never counts."""

SENTENCE_END = (".", "?", "!")
SEGMENT_GAP_S = 1.0
"""A pause at least this long between two words starts a new segment."""

TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
"""``transcribe`` error code: the model failed on this file."""
UNSUPPORTED_SAMPLE_RATE = "UNSUPPORTED_SAMPLE_RATE"
"""``transcribe`` error code: the waveform is not at 16 kHz."""

RawWord = tuple[str, float | None, float | None]
"""A word as the model returns it: text, start and end (times may be missing)."""
LoadModel = Callable[[Path, str, str], Any]
"""Loads a model from a converted folder, given a device and compute type."""


# --- backend selection (D4) ---------------------------------------------------------------


def _ct2_available() -> bool:
    machine = platform.machine().lower()
    return (
        sys.platform == "linux"
        and machine in ("x86_64", "amd64")
        and importlib.util.find_spec("ctranslate2") is not None
    )


def check_backend(backend: str) -> None:
    """Raise ``BackendSetupError`` unless ``transcription.backend`` can run here."""
    if backend == "transformers":
        raise BackendSetupError("transcription.backend 'transformers' is not supported yet")
    if backend not in ("auto", "ct2"):
        raise BackendSetupError("transcription.backend must be 'auto', 'ct2' or 'transformers'")
    if not _ct2_available():
        raise BackendSetupError(
            "the CrisperWhisper ct2 backend needs Linux x86_64 with crisperwhisper[ct2] "
            "installed; other platforms are not supported yet"
        )


# --- weights and the one-time conversion (D17) ----------------------------------------------


def _cache_root() -> Path:
    default = Path.home() / ".cache" / "crisperwhisper"
    return Path(os.environ.get("CRISPERWHISPER_CACHE", default))


def converted_dir(revision: str) -> Path:
    """Where the converted model for a given weights revision lives."""
    slug = MODEL_ID.replace("/", "--")
    return _cache_root() / "murmurs" / f"{slug}--{revision}--{STORED_PRECISION}"


def _is_ready(folder: Path) -> bool:
    return (folder / "model.bin").is_file() and (folder / READY_MARKER).is_file()


def is_converted() -> bool:
    """Whether the cached weights have a finished conversion. Offline and fast."""
    snapshot = cached_snapshot(SPEC)
    return snapshot is not None and _is_ready(converted_dir(snapshot.name))


def convert() -> None:
    """Convert the cached weights to CTranslate2 format, if not done already.

    Runs offline. Converts into a staging folder next to the target and renames it into
    place, so an interrupted conversion leaves nothing that looks finished.
    """
    snapshot = cached_snapshot(SPEC)
    if snapshot is None:
        raise RuntimeError("weights are not cached")
    final = converted_dir(snapshot.name)
    if _is_ready(final):
        return
    from crisperwhisper.converter import ensure_ct2_model

    final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=final.parent, prefix=".staging-") as staging:
        out = ensure_ct2_model(str(snapshot), quantization=STORED_PRECISION, cache_dir=staging)
        (out / READY_MARKER).touch()
        if final.exists():
            shutil.rmtree(final)
        os.replace(out, final)


def crisperwhisper_weights(config: JobConfig) -> Sequence[WeightSpec]:
    """The Hugging Face weights to download. Fails setup if the backend cannot run here."""
    check_backend(config.transcription.backend)
    return [SPEC]


def crisperwhisper_prepared(config: JobConfig) -> Sequence[PreparedModel]:
    """The CTranslate2 conversion that ``murmurs models download`` runs after downloading."""
    return [
        PreparedModel(
            label=f"{MODEL_ID} (CTranslate2 {STORED_PRECISION})",
            is_ready=is_converted,
            prepare=convert,
        )
    ]


# --- converting model output to transcripts (D18, D19) --------------------------------------


def words_to_transcript(
    words: Iterable[RawWord],
    file_id: str,
    mode: TranscriptMode,
    duration_s: float | None = None,
) -> Transcript:
    """Build a transcript from the model's timed words.

    Words with missing times or no text are dropped; times are clipped to
    ``[0, duration_s]``. A segment ends after a word ending in ``.``, ``?`` or ``!``, or
    when the next word starts at least ``SEGMENT_GAP_S`` later. Segment times span their
    words, and segment text is the words joined with spaces.
    """
    clean: list[Word] = []
    for text, start, end in words:
        text = text.strip()
        if not text or start is None or end is None:
            continue
        start = max(0.0, float(start))
        end = max(start, float(end))
        if duration_s is not None:
            end = min(end, duration_s)
            start = min(start, end)
        clean.append(Word(text=text, start=start, end=end))

    segments: list[Segment] = []
    current: list[Word] = []
    for i, word in enumerate(clean):
        current.append(word)
        nxt = clean[i + 1] if i + 1 < len(clean) else None
        if nxt is None or word.text.endswith(SENTENCE_END) or nxt.start - word.end >= SEGMENT_GAP_S:
            segments.append(
                Segment(
                    start=min(w.start for w in current),
                    end=max(w.end for w in current),
                    text=" ".join(w.text for w in current),
                    words=current,
                )
            )
            current = []
    return Transcript(file_id=file_id, mode=mode, segments=segments)


# --- the transcriber ---------------------------------------------------------------------


def _load_crisperwhisper(model_dir: Path, device: str, compute_type: str) -> Any:
    import logging

    from crisperwhisper import CrisperWhisperModel

    # Its info/debug records carry counts and token IDs only, but keep them off anyway.
    logging.getLogger("crisperwhisper").setLevel(logging.WARNING)

    return CrisperWhisperModel(
        str(model_dir), backend="ct2", compute_type=compute_type, device=device
    )


class CrisperWhisperTranscriber:
    """``Transcriber`` backed by CrisperWhisper on one dedicated thread.

    ``model_dir`` is a converted (CTranslate2) model folder. ``load_model`` exists for
    tests; it runs on the dedicated thread like every other model call.
    """

    def __init__(
        self,
        config: JobConfig,
        model_dir: Path,
        revision: str,
        load_model: LoadModel = _load_crisperwhisper,
    ) -> None:
        self.language = config.transcription.language
        self.device, self.device_fallback = resolve_device(config.compute.device)
        """``device_fallback`` is true when ``auto`` found no CUDA device; the runner logs it."""
        self.compute_type = STORED_PRECISION if self.device == "cuda" else "float32"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crisperwhisper")
        try:
            self._model = self._executor.submit(
                load_model, model_dir, self.device, self.compute_type
            ).result()
        except Exception:
            self._executor.shutdown(wait=False)
            raise BackendSetupError(f"could not load {MODEL_ID} from {model_dir}") from None
        self.model_version = f"{MODEL_ID}@{revision[:12]} (ct2 {self.compute_type})"

    def transcribe(
        self, waveform: Waveform, file_id: str, modes: Sequence[TranscriptMode]
    ) -> dict[TranscriptMode, Transcript]:
        """One transcript per requested mode, from a single model pass when both are asked."""
        if waveform.sample_rate != SAMPLE_RATE:
            raise PipelineError(UNSUPPORTED_SAMPLE_RATE)
        names = tuple(TranscriptMode(m).value for m in modes)
        try:
            raw = self._executor.submit(self._run, waveform.samples, names).result()
        except Exception:
            raise PipelineError(TRANSCRIPTION_FAILED) from None
        return {
            TranscriptMode(name): words_to_transcript(
                words, file_id, TranscriptMode(name), waveform.duration_s
            )
            for name, words in raw.items()
            if name in names
        }

    def _run(self, samples: Any, modes: tuple[str, ...]) -> Mapping[str, list[RawWord]]:
        """Runs on the dedicated thread."""
        common = {"sr": SAMPLE_RATE, "language": self.language, "word_timestamps": True}
        if len(modes) > 1:
            results = self._model.transcribe_dual(samples, modes=modes, **common)
        else:
            results = (self._model.transcribe(samples, mode=modes[0], **common),)
        return {r.mode: [(w.word, w.start, w.end) for w in (r.words or [])] for r in results}


def create_crisperwhisper(config: JobConfig) -> CrisperWhisperTranscriber:
    """Registry factory: check the backend, find the converted model and load it."""
    check_backend(config.transcription.backend)
    if not config.transcription.word_timestamps:
        raise BackendSetupError(
            "transcription.word_timestamps must be true: transcripts are built from timed words"
        )
    snapshot = cached_snapshot(SPEC)
    if snapshot is None or not _is_ready(converted_dir(snapshot.name)):
        raise BackendSetupError(
            f"{MODEL_ID} is not downloaded and converted; run `murmurs models download`"
        )
    return CrisperWhisperTranscriber(config, converted_dir(snapshot.name), snapshot.name)


if TYPE_CHECKING:
    from murmurs.backends import Transcriber

    def _check(config: JobConfig) -> Transcriber:
        # Static proof (checked by mypy) that the transcriber satisfies its protocol.
        return create_crisperwhisper(config)
