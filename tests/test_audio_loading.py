"""P2-02: audio loading with the torchcodec audio loader."""

import gc
import io
import tempfile
import weakref
from pathlib import Path

import numpy as np
import pytest
from conftest import StubJob
from fixtures.audio import tone_wav_bytes, write_tone, write_tone_encoded, write_tone_m4a

from murmurs.backends import Waveform
from murmurs.backends.audio import DECODE_FAILED, TARGET_SAMPLE_RATE, TorchcodecAudioLoader
from murmurs.backends.stubs import StubDiarizer, StubNERDetector, StubTranscriber
from murmurs.config import JobConfig, Stage, load_config
from murmurs.errors import PipelineError
from murmurs.models import AudioInfo, RedactionSpan, StageStatus, ValidationResult
from murmurs.runner import run_job
from murmurs.stages import Backends
from murmurs.store import ArtifactStore, file_id_from_hash, hash_file

EXTENSIONS = [".wav", ".m4a"]
EDGE = int(0.1 * TARGET_SAMPLE_RATE)
"""Samples ignored at each end when comparing codecs (encoder priming and padding)."""


def _loader() -> TorchcodecAudioLoader:
    return TorchcodecAudioLoader(EXTENSIONS)


def _stages(config: JobConfig, **update: object) -> JobConfig:
    return config.model_copy(update={"stages": config.stages.model_copy(update=update)})


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


class SpyLoader(TorchcodecAudioLoader):
    """The real loader, recording calls and keeping weak references to its waveforms."""

    def __init__(self) -> None:
        super().__init__(EXTENSIONS)
        self.probed: list[str] = []
        self.loaded: list[str] = []
        self.waveforms: list[weakref.ref[Waveform]] = []

    def probe(self, path: Path, file_id: str) -> ValidationResult:
        self.probed.append(file_id)
        return super().probe(path, file_id)

    def load(self, path: Path, file_id: str) -> Waveform:
        self.loaded.append(file_id)
        waveform = super().load(path, file_id)
        self.waveforms.append(weakref.ref(waveform))
        return waveform


def _backends(loader: TorchcodecAudioLoader, ner: StubNERDetector | None = None) -> Backends:
    return Backends(
        audio_loader=loader,
        diarizer=StubDiarizer(),
        transcriber=StubTranscriber(),
        ner=ner or StubNERDetector(),
    )


# --- decoding --------------------------------------------------------------------------------


def test_stereo_44k_loads_as_mono_16k(tmp_path: Path) -> None:
    path = write_tone(tmp_path / "a.wav", seconds=2.0, sample_rate=44_100, channels=2)
    waveform = _loader().load(path, "f1")
    assert waveform.sample_rate == TARGET_SAMPLE_RATE
    assert waveform.samples.dtype == np.float32
    assert waveform.samples.ndim == 1 and waveform.samples.flags.c_contiguous
    assert abs(len(waveform.samples) - 2.0 * TARGET_SAMPLE_RATE) <= 1
    assert waveform.duration_s == pytest.approx(2.0, abs=1 / TARGET_SAMPLE_RATE)


def test_downmix_averages_channels(tmp_path: Path) -> None:
    # Identical channels: a mean keeps the 0.3 peak (ffmpeg's own downmix gives ~0.42).
    path = write_tone(tmp_path / "a.wav", seconds=1.0, sample_rate=16_000, channels=2)
    samples = _loader().load(path, "f1").samples
    assert float(np.abs(samples).max()) == pytest.approx(0.3, abs=0.01)


def test_mono_16k_wav_decodes_exactly(tmp_path: Path) -> None:
    path = write_tone(tmp_path / "a.wav", seconds=1.0, sample_rate=16_000)
    samples = _loader().load(path, "f1").samples
    raw = np.frombuffer(tone_wav_bytes(seconds=1.0)[44:], dtype="<i2")
    np.testing.assert_allclose(samples, raw / 32768.0, atol=1e-6)


@pytest.mark.parametrize(("sample_rate", "channels"), [(16_000, 1), (44_100, 2)])
def test_wav_and_m4a_of_same_content_match(tmp_path: Path, sample_rate: int, channels: int) -> None:
    wav_path = write_tone(tmp_path / "a.wav", 2.0, sample_rate=sample_rate, channels=channels)
    m4a_path = write_tone_m4a(tmp_path / "a.m4a", 2.0, sample_rate=sample_rate, channels=channels)
    wav = _loader().load(wav_path, "f1").samples
    m4a = _loader().load(m4a_path, "f2").samples
    assert 0 <= len(m4a) - len(wav) <= 0.1 * TARGET_SAMPLE_RATE  # AAC pads the end
    n = len(wav)
    diff = wav[EDGE : n - EDGE] - m4a[EDGE : n - EDGE]
    assert _rms(diff) < 0.05 * _rms(wav[EDGE : n - EDGE])


def test_undecodable_file_fails_with_code_only(tmp_path: Path) -> None:
    path = tmp_path / "notes.wav"
    path.write_text("not audio\n")
    with pytest.raises(PipelineError) as info:
        _loader().load(path, "f1")
    assert info.value.code == DECODE_FAILED
    assert info.value.detail == "" and info.value.__cause__ is None


# --- no files written ------------------------------------------------------------------------


def _snapshot(*dirs: Path) -> set[Path]:
    return {p for d in dirs for p in d.rglob("*")}


def test_loading_writes_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir, output_dir, temp_dir = tmp_path / "in", tmp_path / "out", tmp_path / "tmp"
    for d in (output_dir, temp_dir):
        d.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp_dir))
    monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR
    assert tempfile.gettempdir() == str(temp_dir)
    paths = [
        write_tone(input_dir / "a.wav", seconds=2.0, sample_rate=44_100, channels=2),
        write_tone_m4a(input_dir / "b.m4a", seconds=2.0),
    ]
    before = _snapshot(input_dir, output_dir, temp_dir)

    loader = _loader()
    for path in paths:
        loader.load(path, path.stem)

    assert _snapshot(input_dir, output_dir, temp_dir) == before


def test_job_through_load_audio_writes_no_audio(
    stub_job: StubJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp_dir))
    monkeypatch.setattr(tempfile, "tempdir", None)
    config = _stages(load_config(stub_job()), end=Stage.LOAD_AUDIO)
    input_before = _snapshot(config.data.input_dir)

    summary = run_job(config, "job", console=io.StringIO())

    assert len(summary.done) == 3 and not summary.failed
    assert _snapshot(config.data.input_dir) == input_before
    assert not any(temp_dir.iterdir())
    written = [p for p in config.data.output_dir.rglob("*") if p.is_file()]
    assert written and all(p.suffix.lower() not in {".wav", ".m4a"} for p in written)


# --- artifact, resume and memory -------------------------------------------------------------


def test_audio_info_records_waveform_and_input_hash(stub_job: StubJob) -> None:
    config = _stages(load_config(stub_job()), end=Stage.LOAD_AUDIO)
    run_job(config, "job", console=io.StringIO())
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    path = config.data.input_dir / "a.wav"
    content_hash = hash_file(path)
    info = store.read(file_id_from_hash(content_hash), Stage.LOAD_AUDIO, AudioInfo)
    assert info.input_hash == content_hash
    assert info.sample_rate == TARGET_SAMPLE_RATE
    assert info.n_samples == TARGET_SAMPLE_RATE  # stub_job writes 1 s tones
    assert info.duration_s == pytest.approx(1.0)


def test_resume_at_diarize_decodes_again_without_rerunning_ingest(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    first = SpyLoader()
    run_job(_stages(config, end=Stage.LOAD_AUDIO), "job", backends=_backends(first),
            console=io.StringIO())  # fmt: skip
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    file_ids = sorted(first.loaded)
    ingest_mtimes = {f: store.artifact_path(f, Stage.INGEST).stat().st_mtime_ns for f in file_ids}

    second = SpyLoader()
    summary = run_job(_stages(config, start=Stage.DIARIZE), "job", backends=_backends(second),
                      console=io.StringIO())  # fmt: skip

    assert sorted(summary.done) == file_ids and not summary.failed
    assert second.probed == []
    assert sorted(second.loaded) == file_ids  # decoded again, once per file
    for file_id in file_ids:
        status = store.read_status(file_id)
        assert status is not None and status.stages[Stage.INGEST] is StageStatus.DONE
        assert (
            store.artifact_path(file_id, Stage.INGEST).stat().st_mtime_ns
            == (ingest_mtimes[file_id])
        )


class _WaveformFreedCheck(StubNERDetector):
    """Records, at detect_phi time, whether every decoded waveform has been freed."""

    def __init__(self, loader: SpyLoader) -> None:
        super().__init__()
        self.loader = loader
        self.alive_at_detect: list[int] = []

    def detect(self, text: str, file_id: str) -> list[RedactionSpan]:
        gc.collect()
        self.alive_at_detect.append(sum(ref() is not None for ref in self.loader.waveforms))
        return super().detect(text, file_id)


def test_waveform_freed_after_last_audio_stage(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    loader = SpyLoader()
    ner = _WaveformFreedCheck(loader)
    summary = run_job(config, "job", backends=_backends(loader, ner), console=io.StringIO())
    assert len(summary.done) == 3
    assert len(loader.loaded) == 3  # decoded once per file, shared by later stages
    assert ner.alive_at_detect and all(n == 0 for n in ner.alive_at_detect)


def test_mp3_loads_as_mono_16k(tmp_path: Path) -> None:
    path = write_tone_encoded(tmp_path / "a.mp3", seconds=2.0, sample_rate=44_100, channels=2)
    waveform = TorchcodecAudioLoader([".mp3"]).load(path, "f1")
    assert waveform.sample_rate == TARGET_SAMPLE_RATE and waveform.samples.ndim == 1
    assert abs(waveform.duration_s - 2.0) < 0.1  # MP3 frames pad the ends
