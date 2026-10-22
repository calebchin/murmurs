"""P2-01: ingest and validate with the torchcodec audio loader."""

import io
import math
import os
import stat
import struct
from pathlib import Path

import pytest
from conftest import SENTINEL, StubJob
from fixtures.audio import tone_wav_bytes, write_tone, write_tone_encoded, write_tone_m4a

from murmurs.backends.audio import TorchcodecAudioLoader, _wav_data_truncated
from murmurs.config import DataConfig, JobConfig, Stage, load_config
from murmurs.models import IngestErrorCode, StageStatus, ValidationResult
from murmurs.runner import JobSummary, run_job
from murmurs.store import ArtifactStore, file_id_from_hash, hash_file

EXTENSIONS = [".wav", ".m4a"]


def _loader() -> TorchcodecAudioLoader:
    return TorchcodecAudioLoader(EXTENSIONS)


def _ingest_only(config_path: Path) -> JobConfig:
    config = load_config(config_path)
    return config.model_copy(
        update={"stages": config.stages.model_copy(update={"end": Stage.INGEST})}
    )


def _store(config: JobConfig) -> ArtifactStore:
    return ArtifactStore(config.data.output_dir, config.data.input_dir, "job")


def _file_id(path: Path) -> str:
    return file_id_from_hash(hash_file(path))


def _code(result: ValidationResult) -> str | None:
    return result.error.code if result.error else None


def _corrupt_start_m4a(path: Path, tmp_path: Path) -> Path:
    """An .m4a whose first audio packets are garbage but whose end decodes."""
    raw = write_tone_m4a(tmp_path / "src" / "clean.m4a", seconds=2.0).read_bytes()
    start = raw.index(b"mdat") + 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw[:start] + b"\xff" * 200 + raw[start + 200 :])
    return path


def _video_only_m4a(path: Path, tmp_path: Path) -> Path:
    import torch
    from torchcodec.encoders import VideoEncoder

    source = tmp_path / "src" / "video.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    VideoEncoder(torch.zeros(10, 3, 32, 32, dtype=torch.uint8), frame_rate=10).to_file(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source.read_bytes())
    return path


# --- valid files -----------------------------------------------------------------------------


def test_stereo_wav_reports_metadata(tmp_path: Path) -> None:
    path = write_tone(tmp_path / "a.wav", seconds=2.0, sample_rate=44_100, channels=2)
    result = _loader().probe(path, "f1")
    assert result.valid and result.error is None
    assert result.format == "wav" and result.codec == "pcm_s16le"
    assert result.sample_rate == 44_100 and result.channels == 2
    assert result.duration_s == pytest.approx(2.0, abs=1e-3)
    assert result.file_id == "f1" and result.path == str(path)


def test_m4a_reports_metadata(tmp_path: Path) -> None:
    path = write_tone_m4a(tmp_path / "a.m4a", seconds=2.0, sample_rate=16_000, channels=1)
    result = _loader().probe(path, "f1")
    assert result.valid
    assert result.format == "m4a" and result.codec == "aac"
    assert result.sample_rate == 16_000 and result.channels == 1
    assert result.duration_s == pytest.approx(2.0, abs=0.1)  # AAC frames pad the end


def test_short_file_shorter_than_probe_window_is_valid(tmp_path: Path) -> None:
    result = _loader().probe(write_tone(tmp_path / "a.wav", seconds=0.1), "f1")
    assert result.valid and result.duration_s == pytest.approx(0.1, abs=1e-3)


def test_extension_matches_case_insensitively(tmp_path: Path) -> None:
    assert _loader().probe(write_tone(tmp_path / "A.WAV"), "f1").valid


# --- invalid files ---------------------------------------------------------------------------


def test_zero_byte_file_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    assert _code(_loader().probe(path, "f1")) == IngestErrorCode.EMPTY


def test_text_file_renamed_to_wav_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "notes.wav"
    path.write_text("meeting notes, not audio\n")
    assert _code(_loader().probe(path, "f1")) == IngestErrorCode.UNREADABLE


def test_truncated_wav_is_truncated(tmp_path: Path) -> None:
    raw = tone_wav_bytes(seconds=2.0)
    path = tmp_path / "cut.wav"
    path.write_bytes(raw[: len(raw) // 2])
    assert _code(_loader().probe(path, "f1")) == IngestErrorCode.TRUNCATED


def test_header_only_wav_is_truncated(tmp_path: Path) -> None:
    path = tmp_path / "hdr.wav"
    path.write_bytes(tone_wav_bytes(seconds=1.0)[:44])
    assert _code(_loader().probe(path, "f1")) == IngestErrorCode.TRUNCATED


def test_truncated_m4a_fails(tmp_path: Path) -> None:
    # The encoder writes the index (moov atom) last, so a cut file cannot be opened.
    raw = write_tone_m4a(tmp_path / "src" / "a.m4a", seconds=2.0).read_bytes()
    path = tmp_path / "cut.m4a"
    path.write_bytes(raw[: len(raw) // 2])
    assert _code(_loader().probe(path, "f1")) == IngestErrorCode.UNREADABLE


def test_undecodable_start_is_corrupt(tmp_path: Path) -> None:
    path = _corrupt_start_m4a(tmp_path / "bad.m4a", tmp_path)
    result = _loader().probe(path, "f1")
    assert _code(result) == IngestErrorCode.CORRUPT
    assert result.codec == "aac" and result.duration_s is not None


def test_container_without_audio_has_no_audio_stream(tmp_path: Path) -> None:
    path = _video_only_m4a(tmp_path / "video.m4a", tmp_path)
    assert _code(_loader().probe(path, "f1")) == IngestErrorCode.NO_AUDIO_STREAM


def test_extension_not_configured_is_unsupported(tmp_path: Path) -> None:
    path = write_tone(tmp_path / "a.wav")
    result = TorchcodecAudioLoader([".m4a"]).probe(path, "f1")
    assert _code(result) == IngestErrorCode.UNSUPPORTED_FORMAT


def test_invalid_results_carry_no_library_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.wav"
    path.write_text(f"{SENTINEL}\n")
    result = _loader().probe(path, "f1")
    assert result.error is not None and result.error.message == ""
    assert SENTINEL not in result.model_dump_json()


# --- WAV header check ------------------------------------------------------------------------


def _riff(*chunks: tuple[bytes, int, bytes]) -> bytes:
    """A RIFF/WAVE file from (id, declared size, payload) chunks."""
    body = b"WAVE" + b"".join(
        cid + struct.pack("<I", size) + payload for cid, size, payload in chunks
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


FMT = (b"fmt ", 16, struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16))


@pytest.mark.parametrize(
    ("content", "truncated"),
    [
        (_riff(FMT, (b"data", 4, b"\0" * 4)), False),
        (_riff(FMT, (b"data", 8, b"\0" * 4)), True),
        (_riff((b"LIST", 3, b"abc\0"), FMT, (b"data", 4, b"\0" * 4)), False),  # odd chunk padded
        (_riff(FMT, (b"data", 0xFFFFFFFF, b"\0" * 4)), False),  # streamed, size unknown
        (_riff(FMT, (b"data", 0, b"\0" * 4)), False),
        (_riff(FMT), True),  # no data chunk at all
        (b"not a riff file at all", False),
        (b"RIFF", False),
    ],
    ids=[
        "complete",
        "short",
        "padded-list",
        "unknown-ff",
        "unknown-zero",
        "no-data",
        "other",
        "tiny",
    ],
)
def test_wav_data_truncated(tmp_path: Path, content: bytes, truncated: bool) -> None:
    path = tmp_path / "x.wav"
    path.write_bytes(content)
    assert _wav_data_truncated(path) is truncated


# --- through the runner ----------------------------------------------------------------------


def _bad_inputs(input_dir: Path) -> dict[str, Path]:
    """One truncated, one zero-byte and one text-as-wav file."""
    raw = tone_wav_bytes(seconds=2.0, frequency=300)
    paths = {
        "truncated": input_dir / "cut.wav",
        "empty": input_dir / "empty.wav",
        "text": input_dir / "notes.wav",
    }
    paths["truncated"].write_bytes(raw[: len(raw) // 2])
    paths["empty"].write_bytes(b"")
    paths["text"].write_text(f"{SENTINEL} is not audio\n")
    return paths


def _run(config: JobConfig) -> tuple[JobSummary, str]:
    console = io.StringIO()
    summary = run_job(config, "job", console=console)
    return summary, console.getvalue()


def test_bad_files_fail_with_distinct_codes_and_job_continues(stub_job: StubJob) -> None:
    config_path = stub_job()
    input_dir = config_path.parent / "in"
    bad = _bad_inputs(input_dir)
    good = [input_dir / "a.wav", input_dir / "b.wav", input_dir / "c.m4a"]
    config = load_config(config_path)  # full range: good files go on to later stages

    summary, console = _run(config)

    store = _store(config)
    codes = {f.file_id: f.error_code for f in summary.failed}
    expected = {
        _file_id(bad["truncated"]): IngestErrorCode.TRUNCATED,
        _file_id(bad["empty"]): IngestErrorCode.EMPTY,
        _file_id(bad["text"]): IngestErrorCode.UNREADABLE,
    }
    assert codes == expected
    assert all(f.stage is Stage.INGEST for f in summary.failed)
    assert sorted(summary.done) == sorted(_file_id(p) for p in good)
    assert summary.exit_code == 1

    for file_id, code in expected.items():
        result = store.read(file_id, Stage.INGEST, ValidationResult)
        assert not result.valid and _code(result) == code
        status = store.read_status(file_id)
        assert status is not None and status.stages[Stage.INGEST] is StageStatus.FAILED
        assert status.stages[Stage.LOAD_AUDIO] is StageStatus.SKIPPED
    for path in good:
        status = store.read_status(_file_id(path))
        assert status is not None and set(status.stages.values()) == {StageStatus.DONE}
        assert store.read(_file_id(path), Stage.INGEST, ValidationResult).valid

    logs = "".join(p.read_text() for p in (store.job_dir / "logs").iterdir())
    for text in (logs, console):
        assert SENTINEL not in text
        assert "Invalid data" not in text and "torchcodec/_core" not in text


def test_unconfigured_extension_is_not_picked_up(stub_job: StubJob) -> None:
    config_path = stub_job()
    input_dir = config_path.parent / "in"
    write_tone(input_dir / "extra.flac", frequency=1000)  # contents unlike a.wav's
    config = _ingest_only(config_path)
    config = config.model_copy(
        update={"data": config.data.model_copy(update={"audio_extensions": [".wav"]})}
    )

    summary, _ = _run(config)

    assert sorted(summary.done) == sorted(_file_id(input_dir / name) for name in ("a.wav", "b.wav"))
    for name in ("c.m4a", "extra.flac"):
        assert not _store(config).exists(_file_id(input_dir / name), Stage.INGEST)


def test_ingest_leaves_inputs_unchanged(stub_job: StubJob) -> None:
    config_path = stub_job()
    input_dir = config_path.parent / "in"
    _bad_inputs(input_dir)
    old = math.floor(os.stat(input_dir).st_mtime) - 3600
    before: dict[Path, tuple[bytes, int]] = {}
    for path in sorted(input_dir.iterdir()):
        os.utime(path, (old, old))
        path.chmod(stat.S_IRUSR)  # read-only: any write attempt would fail
        before[path] = (path.read_bytes(), path.stat().st_mtime_ns)
    listing = sorted(p.name for p in input_dir.iterdir())

    summary, _ = _run(_ingest_only(config_path))

    assert len(summary.done) == 3 and len(summary.failed) == 3
    assert sorted(p.name for p in input_dir.iterdir()) == listing
    for path, (content, mtime_ns) in before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime_ns


# --- MP3 (D16) -------------------------------------------------------------------------------


def test_mp3_is_a_default_extension() -> None:
    assert ".mp3" in DataConfig(input_dir=Path("/in"), output_dir=Path("/out")).audio_extensions


def test_mp3_reports_metadata(tmp_path: Path) -> None:
    path = write_tone_encoded(tmp_path / "a.mp3", seconds=2.0, sample_rate=44_100, channels=2)
    result = TorchcodecAudioLoader([".mp3"]).probe(path, "f1")
    assert result.valid
    assert result.format == "mp3" and result.codec == "mp3"
    assert result.sample_rate == 44_100 and result.channels == 2
    assert result.duration_s == pytest.approx(2.0, abs=0.1)
