"""Shared fixtures."""

import socket
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import huggingface_hub.constants
import pytest
import yaml

from murmurs.offline import OFFLINE_ENV

SENTINEL = "SENTINEL_PHI_7f3a"

_LOCAL_HOSTS = frozenset({None, "", "localhost", "127.0.0.1", "::1"})


class NetworkBlockedError(RuntimeError):
    """A test tried to use the network. Not an ``OSError``, so libraries that treat
    connection errors as "offline, fall back" cannot swallow it."""


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every non-AF_UNIX socket connection, and name lookups for remote hosts, raise.

    Applies to all tests, including ``@pytest.mark.gpu`` ones, so real backends are
    proven to load from the local cache only.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def connect(self: socket.socket, address: Any) -> None:
        if self.family != socket.AF_UNIX:
            raise NetworkBlockedError("network access is blocked in tests")
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if self.family != socket.AF_UNIX:
            raise NetworkBlockedError("network access is blocked in tests")
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host not in _LOCAL_HOSTS:
            raise NetworkBlockedError("network access is blocked in tests")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture(autouse=True)
def _restore_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs force offline mode for the whole process; undo it after each test."""
    for name in OFFLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)


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
