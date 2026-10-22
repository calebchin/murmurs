# CLAUDE.md

Instructions for coding agents working in this repository. Read this file, then `docs/design.md`, before starting any task.

## What this project is

A local pipeline that turns research interview audio (CECA and LEDS interviews, 15 min to 1 hour) into a speaker-labeled transcript, then proposes deidentification redactions for human review. The full design, including decisions already made, is in `docs/design.md`. Tasks are in `docs/tasks/`, one file per phase.

The data is sensitive participant interview audio. Treat every transcript, prompt and model output as confidential.

## Hard rules (never break these)

1. **No transcript content in logs.** Logs, exceptions, error messages, progress output and crash reports must never contain transcript text, words, LLM prompts or LLM responses. Log only IDs, counts, durations, stage names and file hashes. Use the logging helpers in `murmurs.logging`; never call `print` or log raw objects that may hold text.
2. **Never modify inputs.** Never write to, move, rename, convert or delete files in the input directory. All outputs go under the configured output directory.
3. **The original transcript is immutable.** Redactions and edits live in a separate layer; the clean transcript is rendered from original + layer.
4. **No network calls with interview data.** Everything runs locally. Network access is allowed only for installing packages and downloading model weights.
5. **No intermediate audio files.** Audio is decoded into memory once; do not write converted or chunked audio to disk.
6. **Stay in scope.** Implement only the current task. Do not add features from "Deferred to later versions" or "Non-goals" in the design doc, and do not refactor code outside the task's listed files without asking.
7. **Stop on open questions.** If a task touches anything in the design doc's "Open questions" list, or the design doc is silent or ambiguous, stop and ask. Do not pick an answer yourself.

## Commands

Use `uv` for everything. Do not use `pip` directly.

```bash
uv sync                          # install dependencies
uv run pytest                    # fast tests (no GPU, no model downloads)
uv run pytest -m gpu             # tests that load real models (only when asked)
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src                  # type check
uv run murmurs --help             # CLI
```

Before finishing any task, run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src` and `uv run pytest`. All must pass.

## Code conventions

- Python 3.11+, `src/` layout, package `murmurs`.
- Pydantic v2 for config and all JSON data models. Data model and config classes are contracts: change them only when a task says so, and bump the relevant `schema_version`.
- The library (`murmurs`) has no imports from the CLI or GUI layers. The CLI is a thin wrapper.
- Each model (STT, diarization, NER, LLM) sits behind a `Protocol` in `murmurs.backends`. Pipeline code depends only on the protocol, never on a model library directly.
- Every backend has a stub implementation used by tests. Stubs are deterministic and need no GPU or downloads.
- Stages read and write artifacts through the artifact store, never through ad hoc paths.
- All file writes are atomic (write to a temp file in the same directory, then rename). Exception: append-only log files are written line by line, not atomically.
- Type hints on all public functions. Docstrings on public classes and functions.
- Config file format is YAML.

## Testing

- Write tests alongside the code for every task. Acceptance criteria in task files are written as tests; make them real tests.
- Default tests must run on a CPU-only machine with no network and no model weights. Mark anything that loads a real model with `@pytest.mark.gpu`.
- All tests, including GPU tests, run with the network blocked: an autouse fixture in `tests/conftest.py` makes non-`AF_UNIX` socket connections and remote name lookups raise `NetworkBlockedError`. Tests that need a Hugging Face cache point `huggingface_hub.constants.HF_HUB_CACHE` at a temporary folder.
- Use fixtures in `tests/fixtures/`: tiny audio clips and short fake transcripts with planted fake PHI. Never commit real interview data.
- Every stage that touches transcript text needs a test that runs it with a sentinel string (e.g. `SENTINEL_PHI_7f3a`) and asserts the sentinel never appears in captured logs.

## Dependency notes

- Transcription uses `crisperwhisper[ct2]` (Linux x86_64 only). Never install `faster-whisper` or upstream `ctranslate2` in the same environment; they overwrite the fork CrisperWhisper depends on.
- The CrisperWhisper `ct2` model must be loaded and run on a single dedicated thread. Parallelism uses separate processes.
- Audio decoding uses `torchcodec` with the system `ffmpeg`. Keep the `torch` and `torchcodec` versions compatible (see torchcodec's compatibility table).
- Diarization uses `nvidia/Nemotron-3-Diarization` through `transformers` (5.18 or later; its feature extractor also needs `librosa`). It is not gated. Gated models (if any are added) need a token: only `murmurs models download` reads it, from the `HF_TOKEN` environment variable; runs load cached weights and never read it. Never hard-code or log the token.
- The pipeline must run with no network connection. Real backends load weights from the local cache only and import model libraries lazily (inside the factory), never at import time of `murmurs.backends.registry`.
- NER uses `OpenMed/privacy-filter-nemotron-v2`; LLM uses GPT-OSS 120B. Both run locally.
- Do not add new dependencies without saying so in your summary and explaining why.

## How to work a task

1. Read the task file and the design doc sections it references.
2. If anything is unclear or touches an open question, ask before writing code.
3. Write tests for the acceptance criteria, then the implementation.
4. Run all checks.
5. Summarize what changed, any new dependencies, and anything you noticed that the design doc does not cover. Do not start the next task unless asked.