"""P1-01: package scaffold and CLI entry point."""

import importlib
import subprocess

import pytest
from typer.testing import CliRunner

import otus
from otus.cli import app


def test_version_flag_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == otus.__version__


def test_installed_entry_point_prints_version() -> None:
    result = subprocess.run(["otus", "--version"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == otus.__version__


@pytest.mark.parametrize(
    "module",
    ["config", "models", "backends", "stages", "runner", "store", "logging", "cli"],
)
def test_module_layout_importable(module: str) -> None:
    importlib.import_module(f"otus.{module}")
