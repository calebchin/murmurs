"""MVP step 2: canonical text (D9), word-level TSV and the job-level transcript export."""

import io
from pathlib import Path

from conftest import StubJob

from murmurs.config import JobConfig, Stage, load_config
from murmurs.models import Segment, Transcript, TranscriptMode, Word
from murmurs.runner import InputFile, export_folder_name, run_job
from murmurs.store import (
    WORDS_TSV_HEADER,
    ArtifactStore,
    canonical_text,
    transcript_to_text,
    transcript_to_words_tsv,
)


def _transcript() -> Transcript:
    first = [Word(text="Hello,", start=0.0, end=0.4, speaker="SPEAKER_00"),
             Word(text="Maria.", start=0.5, end=0.9, speaker="SPEAKER_00")]  # fmt: skip
    second = [Word(text="Hi\tthere\\n", start=61.25, end=61.75, speaker="SPEAKER_01")]
    return Transcript(
        file_id="f1",
        mode=TranscriptMode.INTENDED,
        segments=[
            Segment(speaker="SPEAKER_00", start=0.0, end=0.9, text="Hello, Maria.", words=first),
            Segment(
                speaker="SPEAKER_01", start=61.25, end=61.75, text="Hi\tthere\\n", words=second
            ),
        ],
    )


def test_canonical_text_is_segment_texts_joined_by_newlines() -> None:
    assert canonical_text(_transcript()) == "Hello, Maria.\nHi\tthere\\n"
    assert canonical_text(Transcript(file_id="f", mode=TranscriptMode.VERBATIM, segments=[])) == ""


def test_readable_text_adds_time_and_speaker() -> None:
    assert (
        transcript_to_text(_transcript()).splitlines()[0] == "[00:00.0] SPEAKER_00: Hello, Maria."
    )


def _unescape(field: str) -> str:
    out, i = [], 0
    while i < len(field):
        if field[i] == "\\" and i + 1 < len(field):
            out.append({"\\": "\\", "t": "\t", "n": "\n", "r": "\r"}[field[i + 1]])
            i += 2
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def test_words_tsv_has_one_escaped_row_per_word() -> None:
    tsv = transcript_to_words_tsv(_transcript())
    lines = tsv.splitlines()
    assert lines[0] == WORDS_TSV_HEADER
    assert lines[1] == "0.000\t0.400\tSPEAKER_00\tHello,"
    rows = [line.split("\t") for line in lines[1:]]
    assert all(len(row) == 4 for row in rows)  # tabs inside words did not add columns
    words = [w for s in _transcript().segments for w in s.words]
    assert [_unescape(r[3]) for r in rows] == [w.text for w in words]
    assert [(float(r[0]), float(r[1]), r[2]) for r in rows] == [
        (w.start, w.end, w.speaker) for w in words
    ]
    assert tsv.endswith("\n") and tsv.count("\n") == len(words) + 1


def test_words_tsv_leaves_speaker_empty_before_assignment() -> None:
    t = Transcript(
        file_id="f",
        mode=TranscriptMode.VERBATIM,
        segments=[
            Segment(start=0.0, end=0.5, text="um", words=[Word(text="um", start=0.0, end=0.5)])
        ],
    )
    assert transcript_to_words_tsv(t).splitlines()[1] == "0.000\t0.500\t\tum"


def test_export_folder_name_is_safe_and_unique(tmp_path: Path) -> None:
    file = InputFile(path=tmp_path / "Interview 3 (final).m4a", content_hash="ab" * 32)
    assert export_folder_name(file) == "Interview_3__final_-abababab"
    hidden = InputFile(path=tmp_path / ".hidden.wav", content_hash="cd" * 32)
    assert export_folder_name(hidden) == "hidden-cdcdcdcd"


def _config(stub_job: StubJob, end: Stage) -> JobConfig:
    config = load_config(stub_job())
    return config.model_copy(update={"stages": config.stages.model_copy(update={"end": end})})


def test_job_exports_finished_transcripts(stub_job: StubJob) -> None:
    config = _config(stub_job, Stage.ASSIGN_SPEAKERS)
    summary = run_job(config, "job", console=io.StringIO())
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    folders = sorted(p.name for p in store.transcripts_dir.iterdir())
    assert len(folders) == len(summary.done) == 3
    assert any(name.startswith("a-") for name in folders)
    expected = {
        f"{mode}.{suffix}"
        for mode in ("intended", "verbatim")
        for suffix in ("txt", "rule_based.txt", "words.tsv")
    }
    for folder in store.transcripts_dir.iterdir():
        assert {p.name for p in folder.iterdir()} == expected
        file_id = next(f for f in summary.done if folder.name.endswith(f[:8]))
        exported = (folder / "intended.txt").read_text()
        source = store.artifact_path(file_id, Stage.ASSIGN_SPEAKERS, "intended", ".txt")
        assert exported == source.read_text() and "SPEAKER_" in exported


def test_no_export_before_speaker_assignment(stub_job: StubJob) -> None:
    config = _config(stub_job, Stage.TRANSCRIBE)
    run_job(config, "job", console=io.StringIO())
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    assert not store.transcripts_dir.exists()


def test_deid_outputs_are_exported_next_to_the_original_transcript(stub_job: StubJob) -> None:
    config = _config(stub_job, Stage.DETECT_PHI)
    summary = run_job(config, "job", console=io.StringIO())
    assert summary.done and not summary.failed
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    deid_files = {
        "verbatim.json",
        "verbatim.redactions.json",
        "verbatim.annotated.txt",
        "verbatim.clean.txt",
        "deid_report.json",
    }
    for folder in store.transcripts_dir.iterdir():
        names = {p.name for p in folder.iterdir()}
        assert deid_files <= names and "verbatim.txt" in names
        assert "intended.redactions.json" not in names
        file_id = next(f for f in summary.done if folder.name.endswith(f[:8]))
        original = store.artifact_path(file_id, Stage.ASSIGN_SPEAKERS, "verbatim", ".json")
        assert (folder / "verbatim.json").read_text() == original.read_text()
        layer = store.artifact_path(file_id, Stage.DETECT_PHI, "verbatim", ".json")
        assert (folder / "verbatim.redactions.json").read_text() == layer.read_text()
