"""Typed errors raised by pipeline code and backends."""

from __future__ import annotations

import re

_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")


LLM_INVALID_OUTPUT = "LLM_INVALID_OUTPUT"
"""Error code an ``LLMClient`` raises when its output stays invalid after retries."""
LLM_OUTPUT_TRUNCATED = "LLM_OUTPUT_TRUNCATED"
"""Error code an ``LLMClient`` raises when a response hit its output-token cap. It is not
retried as is; callers split the request into smaller ones instead."""


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


class SetupError(Exception):
    """The job cannot start (bad inputs, config or backends). Maps to exit code 2."""


class BackendSetupError(SetupError):
    """A backend could not be chosen or loaded; the job cannot start."""
