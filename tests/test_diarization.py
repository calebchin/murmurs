"""P2-03: Nemotron-3-Diarization backend."""

import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import huggingface_hub.constants
import pytest
import torch
import yaml
from conftest import StubJob
from typer.testing import CliRunner

from murmurs.backends import registry
from murmurs.backends.device import resolve_device
from murmurs.backends.nemotron import MODEL_ID, WEIGHT_FILES, segments_to_turns
from murmurs.backends.registry import BackendKind
from murmurs.backends.stubs import StubAudioLoader, StubDiarizer, StubNERDetector, StubTranscriber
from murmurs.cli import app
from murmurs.config import CURRENT_SCHEMA_VERSION, Device, Stage, load_config, validate_config
from murmurs.errors import BackendSetupError
from murmurs.models import UNKNOWN_SPEAKER, AudioInfo, Diarization, SpeakerTurn
from murmurs.offline import WeightSpec
from murmurs.runner import run_job
from murmurs.stages import Backends
from murmurs.store import ArtifactStore


def _seg(start: float, end: float, speaker: int) -> dict[str, Any]:
    return {"Start": start, "End": end, "Speaker": speaker}


def _labels(turns: list[SpeakerTurn]) -> list[str]:
    return [t.speaker for t in turns]


# --- conversion ------------------------------------------------------------------------------


def test_segments_become_sorted_turns_labeled_by_first_appearance() -> None:
    segments = [_seg(5.0, 7.0, 1), _seg(0.5, 3.0, 4), _seg(3.0, 5.5, 1), _seg(8.0, 9.0, 4)]
    turns = segments_to_turns(segments)
    assert [(t.start, t.end) for t in turns] == [(0.5, 3.0), (3.0, 5.5), (5.0, 7.0), (8.0, 9.0)]
    assert _labels(turns) == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_00"]


def test_overlapping_turns_are_kept() -> None:
    turns = segments_to_turns([_seg(0.0, 4.0, 0), _seg(3.0, 5.0, 1)])
    assert len(turns) == 2 and turns[0].end > turns[1].start


def test_zero_length_segments_are_dropped_and_empty_input_gives_no_turns() -> None:
    assert segments_to_turns([_seg(1.0, 1.0, 0)]) == []
    assert segments_to_turns([]) == []


def test_num_speakers_relabels_least_talkative_as_unknown() -> None:
    segments = [
        _seg(0.0, 1.0, 2),  # first to appear, but only 1 s of talk
        _seg(1.0, 11.0, 0),  # 10 s
        _seg(11.0, 17.0, 1),  # 6 s
        _seg(17.0, 17.5, 3),  # 0.5 s
    ]
    turns = segments_to_turns(segments, num_speakers=2)
    # The two kept speakers are labeled by their own order of first appearance.
    assert _labels(turns) == [UNKNOWN_SPEAKER, "SPEAKER_00", "SPEAKER_01", UNKNOWN_SPEAKER]
    assert [(t.start, t.end) for t in turns] == [(s["Start"], s["End"]) for s in segments]


def test_num_speakers_tie_keeps_earlier_speaker() -> None:
    turns = segments_to_turns([_seg(0.0, 2.0, 5), _seg(2.0, 4.0, 3)], num_speakers=1)
    assert _labels(turns) == ["SPEAKER_00", UNKNOWN_SPEAKER]


def test_num_speakers_at_or_above_count_changes_nothing() -> None:
    segments = [_seg(0.0, 1.0, 0), _seg(1.0, 2.0, 1)]
    assert segments_to_turns(segments, num_speakers=2) == segments_to_turns(segments)
    assert segments_to_turns(segments, num_speakers=8) == segments_to_turns(segments)


# --- config ----------------------------------------------------------------------------------


def _raw_config(**diarization: Any) -> dict[str, Any]:
    return {
        "data": {"input_dir": "/in", "output_dir": "/out"},
        "diarization": diarization,
        "deid": {"entity_types": [{"name": "PERSON", "tag": "[PERSON_{n}]"}]},
    }


def test_default_diarization_model_is_nemotron() -> None:
    assert validate_config(_raw_config()).diarization.model == MODEL_ID


def test_version_2_config_still_loads() -> None:
    config = validate_config({**_raw_config(num_speakers=2), "schema_version": 2})
    assert config.schema_version == CURRENT_SCHEMA_VERSION and config.diarization.num_speakers == 2


def test_num_speakers_must_be_positive() -> None:
    with pytest.raises(Exception, match="num_speakers"):
        validate_config(_raw_config(num_speakers=0))


# --- registry and weights ---------------------------------------------------------------------


def test_registry_declares_only_the_inference_files(stub_job: StubJob) -> None:
    config = load_config(stub_job(diarization={"model": MODEL_ID}))
    plan = registry.required_weights(config, [BackendKind.DIARIZER])
    assert plan.unavailable == []
    assert plan.weights == [WeightSpec(MODEL_ID, allow_patterns=WEIGHT_FILES)]
    assert not any(f.endswith((".nemo", ".gguf", ".mp4")) for f in WEIGHT_FILES)


@pytest.fixture
def empty_hub_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    cache = tmp_path / "hf-cache" / "hub"
    cache.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    yield cache


def test_missing_weights_exit_2_before_any_file(stub_job: StubJob, empty_hub_cache: Path) -> None:
    config_path = stub_job(diarization={"model": MODEL_ID})
    run = CliRunner().invoke(app, ["run", "-c", str(config_path)])
    assert run.exit_code == 2
    assert MODEL_ID in run.stderr and "murmurs models download" in run.stderr
    config = load_config(config_path)
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    assert not (store.job_dir / "files").exists()


# --- device ----------------------------------------------------------------------------------


def test_resolve_device_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(Device.AUTO) == ("cpu", True)
    assert resolve_device(Device.CPU) == ("cpu", False)
    with pytest.raises(BackendSetupError, match="cuda"):
        resolve_device(Device.CUDA)


def test_resolve_device_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(Device.AUTO) == ("cuda", False)
    assert resolve_device(Device.CUDA) == ("cuda", False)
    assert resolve_device(Device.CPU) == ("cpu", False)


class _CpuFallbackDiarizer(StubDiarizer):
    device_fallback = True


def test_runner_logs_cpu_fallback(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    config = config.model_copy(
        update={"stages": config.stages.model_copy(update={"end": Stage.DIARIZE})}
    )
    console = io.StringIO()
    backends = Backends(
        audio_loader=StubAudioLoader(),
        diarizer=_CpuFallbackDiarizer(),
        transcriber=StubTranscriber(),
        ner=StubNERDetector(),
    )
    run_job(config, "job", backends=backends, console=console)
    assert "device_fallback_cpu" in console.getvalue()
    assert "backend=diarizer" in console.getvalue()


# --- real model (only with -m gpu) -------------------------------------------------------------

EXAMPLE_DIR = os.environ.get("MURMURS_EXAMPLE_AUDIO_DIR")


def _covered(turns: list[SpeakerTurn]) -> float:
    """Seconds covered by the union of the turns."""
    total, end = 0.0, 0.0
    for t in sorted(turns, key=lambda t: t.start):
        if t.end > end:
            total += t.end - max(t.start, end)
            end = t.end
    return total


@pytest.mark.gpu
@pytest.mark.skipif(not EXAMPLE_DIR, reason="MURMURS_EXAMPLE_AUDIO_DIR is not set")
def test_example_audio_diarizes_offline_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    raw = {
        "data": {
            "input_dir": EXAMPLE_DIR,
            "output_dir": str(tmp_path / "out"),
        },
        "stages": {"end": "diarize"},
        "diarization": {"model": MODEL_ID, "llm_correction": False},
        "transcription": {"model": "stub"},
        "deid": {
            "ner_model": "stub",
            "llm_model": "stub",
            "entity_types": [{"name": "PERSON", "tag": "[PERSON_{n}]"}],
        },
    }
    config_path = tmp_path / "job.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    config = load_config(config_path)

    summary = run_job(config, "job", console=io.StringIO())

    assert summary.done and not summary.failed
    assert os.environ.get("HF_HUB_OFFLINE") == "1" and "HF_TOKEN" not in os.environ
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    manifest = store.read_manifest()
    assert manifest is not None and manifest.model_versions["diarizer"].startswith(MODEL_ID)
    for file_id in summary.done:
        turns = store.read(file_id, Stage.DIARIZE, Diarization).turns
        duration = store.read(file_id, Stage.LOAD_AUDIO, AudioInfo).duration_s
        assert len({t.speaker for t in turns} - {UNKNOWN_SPEAKER}) >= 2
        assert turns == sorted(turns, key=lambda t: (t.start, t.end, t.speaker))
        assert _covered(turns) >= 0.5 * duration
