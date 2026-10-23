"""LLM chunk runner: concurrency, ordering and splitting on truncated or invalid output."""

import threading
import time

import pytest

from murmurs.errors import LLM_INVALID_OUTPUT, LLM_OUTPUT_TRUNCATED, PipelineError
from murmurs.stages.llm_chunks import halves, run_chunks

Chunk = tuple[int, ...]


def split(chunk: Chunk) -> tuple[Chunk, Chunk] | None:
    parts = halves(chunk)
    return None if parts is None else (tuple(parts[0]), tuple(parts[1]))


def test_results_come_back_in_chunk_order_despite_finish_order() -> None:
    chunks: list[Chunk] = [(i,) for i in range(8)]

    def call(chunk: Chunk) -> int:
        time.sleep(0.05 * (8 - chunk[0]))  # later chunks finish first
        return chunk[0] * 10

    outcome = run_chunks(chunks, call, split, max_concurrency=8)
    assert [result for _, result in outcome.results] == [i * 10 for i in range(8)]
    assert outcome.splits == 0 and outcome.failed == 0


def test_requests_run_concurrently_up_to_the_limit() -> None:
    running = 0
    peak = 0
    lock = threading.Lock()

    def call(chunk: Chunk) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.1)
        with lock:
            running -= 1
        return 0

    run_chunks([(i,) for i in range(12)], call, split, max_concurrency=4)
    assert peak == 4
    peak = 0
    run_chunks([(i,) for i in range(5)], call, split, max_concurrency=1)
    assert peak == 1


@pytest.mark.parametrize("code", [LLM_OUTPUT_TRUNCATED, LLM_INVALID_OUTPUT])
def test_a_failing_chunk_is_split_until_the_parts_succeed(code: str) -> None:
    def call(chunk: Chunk) -> int:
        if len(chunk) > 2:  # only small chunks get a usable response
            raise PipelineError(code)
        return sum(chunk)

    outcome = run_chunks([(1, 2, 3, 4, 5, 6, 7, 8)], call, split, max_concurrency=2)
    assert [chunk for chunk, _ in outcome.results] == [(1, 2), (3, 4), (5, 6), (7, 8)]
    assert [result for _, result in outcome.results] == [3, 7, 11, 15]
    assert outcome.splits == 3 and outcome.failed == 0


def test_a_one_line_chunk_that_keeps_failing_is_counted_as_failed() -> None:
    def call(chunk: Chunk) -> int:
        if 3 in chunk:
            raise PipelineError(LLM_OUTPUT_TRUNCATED)
        return 1

    outcome = run_chunks([(1, 2), (3, 4)], call, split)
    assert [chunk for chunk, _ in outcome.results] == [(1, 2), (4,)]
    assert outcome.splits == 1 and outcome.failed == 1


def test_a_failed_request_stops_the_run_and_cancels_pending_chunks() -> None:
    started: list[int] = []

    def call(chunk: Chunk) -> int:
        started.append(chunk[0])
        if chunk[0] == 0:
            raise PipelineError("LLM_REQUEST_FAILED")
        time.sleep(0.2)
        return 0

    with pytest.raises(PipelineError, match="LLM_REQUEST_FAILED"):
        run_chunks([(i,) for i in range(20)], call, split, max_concurrency=2)
    assert len(started) < 20  # chunks not yet started were cancelled


def test_no_chunks_means_no_requests() -> None:
    outcome = run_chunks([], lambda chunk: 1 / 0, split, max_concurrency=4)
    assert outcome.results == [] and outcome.splits == 0 and outcome.failed == 0


def test_halves() -> None:
    assert halves([1]) is None and halves([]) is None
    assert halves([1, 2, 3]) == ([1], [2, 3])
