"""P1-05: artifact store."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from murmurs.config import Stage, load_config
from murmurs.models import (
    Diarization,
    FileStatus,
    RunManifest,
    Segment,
    SpeakerTurn,
    Transcript,
    TranscriptMode,
    Word,
)
from murmurs.store import (
    FILE_ID_LENGTH,
    ArtifactStore,
    OutputInsideInputError,
    StoreError,
    atomic_writer,
    file_id_from_hash,
    hash_file,
    transcript_to_text,
)

EXAMPLE = Path(__file__).parent.parent / "examples" / "job.yaml"


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    return input_dir, tmp_path / "out"


@pytest.fixture
def store(dirs: tuple[Path, Path]) -> ArtifactStore:
    input_dir, output_dir = dirs
    return ArtifactStore(output_dir, input_dir, job_id="job1")


def _transcript(mode: TranscriptMode = TranscriptMode.INTENDED) -> Transcript:
    words = [Word(text="hi", start=61.0, end=61.4), Word(text="there", start=61.5, end=62.0)]
    segment = Segment(speaker="SPEAKER_00", start=61.0, end=62.0, text="hi there", words=words)
    return Transcript(file_id="f1", mode=mode, segments=[segment])


def test_layout_matches_spec(store: ArtifactStore, dirs: tuple[Path, Path]) -> None:
    job_dir = dirs[1] / "job1"
    assert store.manifest_path == job_dir / "manifest.json"
    assert store.artifact_path("f1", Stage.DIARIZE) == job_dir / "files/f1/diarize.json"
    assert (
        store.artifact_path("f1", Stage.TRANSCRIBE, "intended")
        == job_dir / "files/f1/transcribe.intended.json"
    )
    assert store.status_path("f1") == job_dir / "files/f1/status.json"


def test_write_then_read_returns_equal_object(store: ArtifactStore) -> None:
    diarization = Diarization(
        file_id="f1", turns=[SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=1.5)]
    )
    store.write("f1", Stage.DIARIZE, diarization)
    assert store.read("f1", Stage.DIARIZE, Diarization) == diarization

    for mode in TranscriptMode:
        store.write("f1", Stage.TRANSCRIBE, _transcript(mode), variant=mode.value)
    for mode in TranscriptMode:
        read = store.read("f1", Stage.TRANSCRIBE, Transcript, variant=mode.value)
        assert read == _transcript(mode)


def test_status_and_manifest_round_trip(store: ArtifactStore) -> None:
    assert store.read_status("f1") is None
    assert store.read_manifest() is None
    status = FileStatus(file_id="f1", input_hash="abc")
    store.write_status(status)
    assert store.read_status("f1") == status

    manifest = RunManifest(
        job_id="job1",
        started_at=datetime(2026, 10, 20, tzinfo=UTC),
        config=load_config(EXAMPLE),
        package_version="0.1.0",
    )
    store.write_manifest(manifest)
    assert store.read_manifest() == manifest


def test_exists(store: ArtifactStore) -> None:
    assert not store.exists("f1", Stage.DIARIZE)
    store.write("f1", Stage.DIARIZE, Diarization(file_id="f1", turns=[]))
    assert store.exists("f1", Stage.DIARIZE)


def test_interrupted_write_leaves_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    with pytest.raises(RuntimeError), atomic_writer(target) as f:
        f.write('{"partial": ')
        raise RuntimeError("simulated crash mid-write")
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_interrupted_overwrite_keeps_previous_version(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("old")
    with pytest.raises(KeyboardInterrupt), atomic_writer(target) as f:
        f.write("new but incomplete")
        raise KeyboardInterrupt
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_store_write_interrupted_at_rename_leaves_nothing(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("murmurs.store.os.replace", fail_replace)
    with pytest.raises(OSError):
        store.write("f1", Stage.DIARIZE, Diarization(file_id="f1", turns=[]))
    file_dir = store.file_dir("f1")
    assert not store.exists("f1", Stage.DIARIZE)
    assert list(file_dir.iterdir()) == []


@pytest.mark.parametrize("output", ["in", "in/out", "in/a/b"])
def test_output_inside_input_fails_on_construction(tmp_path: Path, output: str) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    with pytest.raises(OutputInsideInputError, match="inside input_dir"):
        ArtifactStore(tmp_path / output, input_dir, job_id="job1")


def test_output_inside_input_via_symlink_fails(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(input_dir)
    with pytest.raises(OutputInsideInputError):
        ArtifactStore(link / "out", input_dir, job_id="job1")


def test_writes_landing_in_input_are_refused(tmp_path: Path) -> None:
    # Input nested inside output: a job ID equal to the input folder's name would
    # place artifacts inside the input directory.
    output_dir = tmp_path / "out"
    input_dir = output_dir / "audio"
    input_dir.mkdir(parents=True)
    store = ArtifactStore(output_dir, input_dir, job_id="audio")
    with pytest.raises(OutputInsideInputError, match="refusing to write"):
        store.write("f1", Stage.DIARIZE, Diarization(file_id="f1", turns=[]))
    assert list(input_dir.iterdir()) == []


@pytest.mark.parametrize("bad", ["..", ".", "a/b", "", ".hidden", "x\\y"])
def test_unsafe_ids_are_rejected(store: ArtifactStore, bad: str) -> None:
    with pytest.raises(StoreError, match="not a safe file name"):
        store.file_dir(bad)
    with pytest.raises(StoreError):
        store.artifact_path("f1", Stage.TRANSCRIBE, variant=bad)


def test_file_id_comes_from_content_not_name(dirs: tuple[Path, Path]) -> None:
    input_dir, _ = dirs
    original = input_dir / "a.wav"
    original.write_bytes(b"audio bytes")
    before = (original.read_bytes(), original.stat().st_mtime_ns)
    content_hash = hash_file(original)
    renamed = input_dir / "renamed.wav"
    renamed.write_bytes(b"audio bytes")

    assert hash_file(renamed) == content_hash
    assert file_id_from_hash(content_hash) == content_hash[:FILE_ID_LENGTH]
    assert (original.read_bytes(), original.stat().st_mtime_ns) == before


def test_transcript_text_export(store: ArtifactStore) -> None:
    path = store.write_transcript_text("f1", Stage.TRANSCRIBE, _transcript(), variant="intended")
    assert path.name == "transcribe.intended.txt"
    assert path.read_text() == "[01:01.0] SPEAKER_00: hi there\n"


def test_transcript_text_without_speaker() -> None:
    segment = Segment(start=5.0, end=6.0, text="hello")
    transcript = Transcript(file_id="f1", mode=TranscriptMode.VERBATIM, segments=[segment])
    assert transcript_to_text(transcript) == "[00:05.0] hello\n"
