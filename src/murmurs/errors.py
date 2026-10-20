"""Typed errors raised by pipeline code and backends."""

from __future__ import annotations

import re

_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class PipelineError(Exception):
    """A failure processing one file, identified by a fixed error code.

    ``str(exc)`` is the code only. ``detail`` is free text for debugging and may quote
    content, so it is never logged or written to artifacts.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        if not _CODE.fullmatch(code):
            raise ValueError("error code must be UPPER_SNAKE_CASE")
        self.code = code
        self.detail = detail
        super().__init__(code)


class BackendSetupError(Exception):
    """A backend could not be chosen or loaded; the job cannot start."""
