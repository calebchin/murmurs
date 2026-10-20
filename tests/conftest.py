"""Shared fixtures."""

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

SENTINEL = "SENTINEL_PHI_7f3a"


@pytest.fixture(autouse=True)
def _restore_excepthook() -> Iterator[None]:
    """Runs through the CLI install a crash hook; undo it after each test."""
    original = sys.excepthook
    yield
    sys.excepthook = original


StubJob = Callable[..., Path]


@pytest.fixture
def stub_job(tmp_path: Path) -> StubJob:
    """Create input files and a stub-backend config; returns the config path.

    ``files`` maps relative input paths to contents (fake audio bytes; stubs never
    decode them). Extra keyword arguments are merged into the config sections.
    """

    def make(files: dict[str, bytes] | None = None, **sections: dict[str, Any]) -> Path:
        input_dir = tmp_path / "in"
        input_dir.mkdir(exist_ok=True)
        if files is None:
            files = {"a.wav": b"audio-a", "b.wav": b"audio-b", "c.m4a": b"audio-c"}
        for name, content in files.items():
            path = input_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        raw: dict[str, Any] = {
            "data": {"input_dir": str(input_dir), "output_dir": str(tmp_path / "out")},
            "diarization": {"model": "stub", "llm_correction": False},
            "transcription": {"model": "stub"},
            "deid": {
                "ner_model": "stub",
                "llm_model": "stub",
                "entity_types": [{"name": "PERSON", "tag": "[PERSON_{n}]"}],
            },
        }
        for section, values in sections.items():
            raw[section] = {**raw.get(section, {}), **values}
        config_path = tmp_path / "job.yaml"
        config_path.write_text(yaml.safe_dump(raw))
        return config_path

    return make
