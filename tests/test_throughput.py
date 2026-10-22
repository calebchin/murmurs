"""P3-04 (MVP): the job throughput summary in metrics.json."""

import io
import json

import pytest
from conftest import StubJob

from murmurs.config import Stage, load_config
from murmurs.runner import run_job
from murmurs.store import ArtifactStore


def test_summary_reports_audio_wall_time_and_real_time_factor(stub_job: StubJob) -> None:
    config = load_config(stub_job())
    config = config.model_copy(
        update={"stages": config.stages.model_copy(update={"end": Stage.ASSIGN_SPEAKERS})}
    )
    console = io.StringIO()
    run_job(config, "job", console=console)
    store = ArtifactStore(config.data.output_dir, config.data.input_dir, "job")
    metrics = json.loads((store.job_dir / "metrics.json").read_text())
    assert metrics["schema_version"] == 2
    summary = metrics["summary"]
    assert summary["n_files"] == 3
    assert summary["audio_s"] == pytest.approx(
        3.0, abs=0.1
    )  # three 1 s tones (m4a adds AAC padding)
    assert set(summary["group_wall_s"]) == {"ingest", "audio", "assign_speakers"}
    assert summary["wall_s"] >= sum(summary["group_wall_s"].values()) - 1e-3
    # wall_s is stored to the millisecond, so recomputing from it is only approximate
    assert summary["real_time_factor"] == pytest.approx(
        summary["wall_s"] / summary["audio_s"], rel=2e-2
    )
    assert summary["audio_hours_per_hour"] == pytest.approx(
        summary["audio_s"] / summary["wall_s"], rel=1e-2
    )
    assert f"audio_duration_s={summary['audio_s']}" in console.getvalue()

    run_job(config, "job", console=io.StringIO())  # everything is skipped now
    again = json.loads((store.job_dir / "metrics.json").read_text())["summary"]
    assert again["n_files"] == 0 and again["audio_s"] == 0 and again["real_time_factor"] is None
