"""Run LLM requests over a file's chunks: concurrently, splitting chunks that fail.

The vLLM server batches concurrent requests on the GPU, so sending a file's chunks at
the same time raises throughput several-fold over one at a time. Results come back in
chunk order, so the outcome does not depend on which request finished first.

A chunk whose response is cut off at the output cap (``LLM_OUTPUT_TRUNCATED``) or stays
invalid (``LLM_INVALID_OUTPUT``) is split in half and the halves are tried instead; a
chunk that cannot be split further is counted as failed. Any other error (the request
itself failed) stops the file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from murmurs.errors import LLM_INVALID_OUTPUT, LLM_OUTPUT_TRUNCATED, PipelineError

C = TypeVar("C")
R = TypeVar("R")

SPLITTABLE = frozenset({LLM_INVALID_OUTPUT, LLM_OUTPUT_TRUNCATED})
"""Error codes that make a chunk worth retrying as two smaller chunks."""


@dataclass
class ChunkResults(Generic[C, R]):
    """Results of a run, in chunk order (a split chunk's halves in place of it)."""

    results: list[tuple[C, R]] = field(default_factory=list)
    splits: int = 0
    """Times a chunk was split in half."""
    failed: int = 0
    """Chunks (after splitting) that produced no usable response."""


def _solve(
    chunk: C, call: Callable[[C], R], split: Callable[[C], tuple[C, C] | None]
) -> ChunkResults[C, R]:
    try:
        return ChunkResults(results=[(chunk, call(chunk))])
    except PipelineError as exc:
        if exc.code not in SPLITTABLE:
            raise
    halves = split(chunk)
    if halves is None:
        return ChunkResults(failed=1)
    combined: ChunkResults[C, R] = ChunkResults(splits=1)
    for half in halves:
        part = _solve(half, call, split)
        combined.results.extend(part.results)
        combined.splits += part.splits
        combined.failed += part.failed
    return combined


def run_chunks(
    chunks: Sequence[C],
    call: Callable[[C], R],
    split: Callable[[C], tuple[C, C] | None],
    max_concurrency: int = 1,
) -> ChunkResults[C, R]:
    """Run ``call`` on every chunk, up to ``max_concurrency`` at a time.

    ``split`` returns two smaller chunks, or None when a chunk cannot be split. Raises
    the first non-splittable ``PipelineError`` (or other exception), after cancelling
    chunks not yet started.
    """
    total: ChunkResults[C, R] = ChunkResults()
    if not chunks:
        return total
    pool = ThreadPoolExecutor(max_workers=max(1, min(max_concurrency, len(chunks))))
    try:
        futures: list[Future[ChunkResults[C, R]]] = [
            pool.submit(_solve, chunk, call, split) for chunk in chunks
        ]
        for future in futures:
            part = future.result()
            total.results.extend(part.results)
            total.splits += part.splits
            total.failed += part.failed
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return total


def halves(items: Sequence[C]) -> tuple[list[C], list[C]] | None:
    """Split a list of lines into two non-empty halves, or None if it has fewer than two."""
    if len(items) < 2:
        return None
    middle = len(items) // 2
    return list(items[:middle]), list(items[middle:])
