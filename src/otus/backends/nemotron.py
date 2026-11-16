"""Speaker diarization with NVIDIA Nemotron-3-Diarization through ``transformers``.

``torch`` and ``transformers`` are imported when the diarizer is created, never when this
module is imported, so offline mode is in force first.

The model finds up to 8 speakers itself and cannot be told how many there are, so
``diarization.num_speakers`` is applied afterwards as a cap (see ``segments_to_turns``).
Its frame-level output is used as the processor returns it, with no smoothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from murmurs.backends import Waveform
from murmurs.backends.device import resolve_device
from murmurs.config import JobConfig
from murmurs.errors import BackendSetupError, PipelineError
from murmurs.models import UNKNOWN_SPEAKER, Diarization, SpeakerTurn
from murmurs.offline import WeightSpec

MODEL_ID = "nvidia/Nemotron-3-Diarization"
WEIGHT_FILES = ("config.json", "processor_config.json", "model.safetensors")
"""The files inference needs; the repo also holds a .nemo checkpoint, a GGUF and demo media."""

SPEECH_THRESHOLD = 0.5
"""Speaker probability above which a 10 ms frame counts as that speaker's speech."""

DIARIZATION_FAILED = "DIARIZATION_FAILED"
"""``diarize`` error code: the model failed on this file."""
UNSUPPORTED_SAMPLE_RATE = "UNSUPPORTED_SAMPLE_RATE"
"""``diarize`` error code: the waveform is not at the model's sample rate."""


def nemotron_weights(config: JobConfig) -> Sequence[WeightSpec]:
    """The model files to download and check for offline runs."""
    return [WeightSpec(MODEL_ID, allow_patterns=WEIGHT_FILES)]


def segments_to_turns(
    segments: Iterable[Mapping[str, Any]], num_speakers: int | None = None
) -> list[SpeakerTurn]:
    """Convert the processor's ``{"Start", "End", "Speaker"}`` segments to speaker turns.

    Speakers are labeled ``SPEAKER_00``, ``SPEAKER_01``, ... in order of first appearance.
    With ``num_speakers`` = N, only the N speakers with the most talk time keep a label
    (ties go to the earlier speaker); the rest become ``UNKNOWN``. Overlapping turns are
    kept. Turns are sorted by start time, then end time, then label.
    """
    raw = [
        (float(s["Start"]), float(s["End"]), int(s["Speaker"]))
        for s in segments
        if float(s["End"]) > float(s["Start"])
    ]
    talk: dict[int, float] = {}
    first: dict[int, float] = {}
    for start, end, speaker in raw:
        talk[speaker] = talk.get(speaker, 0.0) + end - start
        first[speaker] = min(first.get(speaker, start), start)

    kept = list(talk)
    if num_speakers is not None:
        kept = sorted(kept, key=lambda s: (-talk[s], first[s], s))[:num_speakers]
    labels = {
        s: f"SPEAKER_{i:02d}" for i, s in enumerate(sorted(kept, key=lambda s: (first[s], s)))
    }

    turns = [
        SpeakerTurn(speaker=labels.get(speaker, UNKNOWN_SPEAKER), start=start, end=end)
        for start, end, speaker in raw
    ]
    return sorted(turns, key=lambda t: (t.start, t.end, t.speaker))


class NemotronDiarizer:
    """``Diarizer`` backed by Nemotron-3-Diarization, loaded once from the local cache."""

    def __init__(self, config: JobConfig) -> None:
        import torch
        import transformers

        # transformers is only partly typed; go through Any so strict mypy accepts the calls.
        tf: Any = transformers
        tf.utils.logging.set_verbosity_error()
        tf.utils.logging.disable_progress_bar()
        self.device, self.device_fallback = resolve_device(config.compute.device)
        """``device_fallback`` is true when ``auto`` found no CUDA device; the runner logs it."""
        self.num_speakers = config.diarization.num_speakers
        try:
            self._processor: Any = tf.AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            model: Any = tf.AutoModelForAudioFrameClassification.from_pretrained(
                MODEL_ID, local_files_only=True, dtype=torch.float32
            )
            self._model: Any = model.to(self.device).eval()
        except Exception:
            raise BackendSetupError(
                f"could not load {MODEL_ID} from the local model cache"
            ) from None
        self.sample_rate: int = self._processor.feature_extractor.sampling_rate
        self.model_version = f"{MODEL_ID} (transformers {transformers.__version__})"

    def diarize(self, waveform: Waveform, file_id: str) -> Diarization:
        """Speaker turns for the whole recording, in time order."""
        import torch

        if waveform.sample_rate != self.sample_rate:
            raise PipelineError(UNSUPPORTED_SAMPLE_RATE)
        try:
            inputs = self._processor(
                waveform.samples, sampling_rate=self.sample_rate, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**inputs).logits
            segments = self._processor.extract_speaker_dict(
                logits.float(), inputs.get("attention_mask"), threshold=SPEECH_THRESHOLD
            )[0]
        except Exception:
            raise PipelineError(DIARIZATION_FAILED) from None
        return Diarization(file_id=file_id, turns=segments_to_turns(segments, self.num_speakers))

    def close(self) -> None:
        """Drop the model and return its GPU memory to the device."""
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


if TYPE_CHECKING:
    from murmurs.backends import Diarizer

    def _check(config: JobConfig) -> Diarizer:
        # Static proof (checked by mypy) that the diarizer satisfies its protocol.
        return NemotronDiarizer(config)
