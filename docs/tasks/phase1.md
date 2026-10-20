# Phase 1: Foundations — Tasks

Phase goal (from `docs/design.md`): a job with stub stages runs end to end, skips a deliberately failing file, and resumes after an interruption.

No real models are used in this phase. Every backend is a stub. Do the tasks in order; each depends on the ones before it. Finish one task, run all checks, summarize, and stop.

---

## P1-01: Project scaffold

**Goal:** an installable, empty package with tooling in place.

**Scope**
- `uv init` with `src/` layout, package `murmurs`, Python 3.11+.
- Dev dependencies: `pytest`, `ruff`, `mypy`.
- `pyproject.toml` configures ruff, mypy (strict on `src/`) and pytest, including a `gpu` marker excluded by default.
- CLI entry point `murmurs` (Typer or argparse; pick one and say which) with a `--version` flag.
- Empty module layout: `config`, `models` (data model), `backends`, `stages`, `runner`, `store`, `logging`, `cli`.
- `.gitignore` excluding outputs, caches, `.env`, and any audio file types. Do not ignore the `.claude/` directory itself (it holds `.claude/CLAUDE.md`); ignore only `.claude/settings.local.json`.

**Out of scope:** any pipeline logic.

**Acceptance criteria**
- `uv sync` succeeds on a clean checkout.
- `uv run murmurs --version` prints the package version.
- `uv run pytest`, `uv run ruff check .`, `uv run mypy src` all pass.

---

## P1-02: Config schema and loading

**Goal:** the job config from the design doc as validated pydantic models, loaded from YAML.

**Scope**
- Implement the schema in `docs/design.md` → Configuration in `murmurs.config`, exactly as written. `action` on `EntityType` and `backend` on `TranscriptionConfig` become enums.
- `load_config(path) -> JobConfig` reading YAML.
- Validation errors name the field path and the problem, without dumping the whole file.
- Reject configs whose `schema_version` is newer than the code supports.
- Validate that `stages.start` comes before or equals `stages.end`.
- A CLI command `murmurs config schema` that prints the JSON Schema.
- `examples/job.yaml`: a complete example config with two entity types (`PERSON` and `LOCATION`).

**Out of scope:** CLI overrides (P1-03), presets.

**Acceptance criteria**
- `examples/job.yaml` loads without error.
- A config missing `data.input_dir` fails with a message naming `data.input_dir`.
- `stages.start: detect_phi` with `stages.end: ingest` fails validation.
- `redaction_threshold: 1.5` fails validation.
- A config with `schema_version: 999` fails with a clear message.

---

## P1-03: CLI run command and overrides

**Goal:** `murmurs run` loads a config and applies CLI overrides.

**Scope**
- `murmurs run --config job.yaml [overrides]`.
- Named flags for common options: `--input-dir`, `--output-dir`, `--start`, `--end`, `--sample N`, `--no-resume`, `--device`.
- A generic `--set key.path=value` flag (repeatable) for any other field.
- Precedence: schema defaults < config file < CLI flags. Overridden config is re-validated.
- `--dry-run` prints the resolved config and exits without running.
- For now, `run` without `--dry-run` calls the runner from P1-08; until then, it exits with "runner not implemented".

**Acceptance criteria**
- `--end transcribe` overrides the file's `stages.end`.
- `--set diarization.num_speakers=2` sets an int, not a string.
- `--set deid.redaction_threshold=2` fails validation.
- `--dry-run` output reflects every override.

---

## P1-04: Data model

**Goal:** pydantic models for every JSON artifact in the design doc → Data model.

**Scope** (in `murmurs.models`)
- `ValidationResult`: file ID, path, format, duration, sample rate, channels, valid flag, error code.
- `SpeakerTurn`: speaker label, start, end.
- `Word`: text, start, end, speaker (optional).
- `Segment`: speaker, start, end, text, words.
- `Transcript`: file ID, mode (`verbatim` or `intended`), segments, schema version.
- `RedactionSpan`: ID, start and end character offsets, entity type, tag, confidence, sources (`ner`, `llm`, `manual`), status (`proposed`, `approved`, `rejected`, `manual`), reviewer, timestamp, flagged-edge-case flag.
- `RedactionLayer`: transcript reference, spans, review state (`unreviewed`, `in_progress`, `complete`, `signed_off`), audit events.
- `AuditEvent`: span ID, action, actor, timestamp, before and after status.
- `FileStatus`: file ID, per-stage status (`pending`, `done`, `failed`, `skipped`), error code, input hash.
- `RunManifest`: job ID, timestamps, resolved config, package and model versions, code version (git commit if available), input file hashes, file statuses.
- Every top-level artifact carries a `schema_version`.
- Errors in `FileStatus` and `ValidationResult` are codes plus short PHI-free messages, never exception text that might quote content.

**Out of scope:** redaction rendering logic (phase 3), review logic (phase 4).

**Acceptance criteria**
- Every model round-trips through JSON unchanged.
- `RedactionSpan` rejects `start >= end` and confidence outside 0–1.
- `Segment` rejects words outside its time range.
- JSON Schemas for all top-level artifacts can be exported.

---

## P1-05: Artifact store

**Goal:** one place that decides where artifacts live and writes them safely.

**Scope** (in `murmurs.store`)
- Output layout: `<output_dir>/<job_id>/manifest.json`, `<output_dir>/<job_id>/files/<file_id>/<stage>.json`, plus `status.json` per file.
- `file_id` is derived from the input file's content hash, so renaming an input doesn't break resume.
- Atomic writes (temp file in the same directory, then rename).
- Read and write typed artifacts by model class.
- Refuse to write anywhere under the input directory, even if the output directory is configured inside it.
- Plain-text transcript export alongside the JSON (`<stage>.txt`).

**Acceptance criteria**
- Writing then reading an artifact returns an equal object.
- An interrupted write (simulated by raising mid-write) leaves no partial file at the target path.
- Configuring `output_dir` inside `input_dir` fails with a clear error before any processing.
- Input files' bytes and modification times are unchanged after a full stub run (asserted in P1-10).

---

## P1-06: Backend interfaces and stubs

**Goal:** model-agnostic interfaces so pipeline code never imports a model library.

**Scope** (in `murmurs.backends`)
- Protocols: `AudioLoader`, `Diarizer`, `Transcriber`, `NERDetector`, `LLMClient`.
- Each protocol's methods take and return types from `murmurs.models` or numpy arrays, never model-library types.
- Deterministic stubs for each, configurable to fail on a specific file ID (for P1-08 and P1-10 tests).
- A registry that picks the implementation by name from config (`"stub"` for now).

**Out of scope:** real model integrations (phase 2 and 3).

**Acceptance criteria**
- Each stub satisfies its protocol (checked by mypy and a runtime test).
- The stub transcriber returns a transcript containing a configurable sentinel string, for the logging tests.
- A stub configured to fail on file ID X raises a typed pipeline error only for X.

---

## P1-07: PHI-safe logging and timing metrics

**Goal:** structured logs and metrics that cannot leak transcript content.

**Scope** (in `murmurs.logging`)
- JSON-lines logs to `<output_dir>/<job_id>/logs/`, plus human-readable console output.
- A logging helper that accepts only allowlisted field names (IDs, counts, durations, stage, status, error code). Any other field is rejected.
- An exception handler that logs the exception type, error code, stage and file ID only, never the message or traceback locals. Full tracebacks go to the log only when an explicit `--debug-unsafe` flag is set, and that flag prints a warning.
- Per-stage, per-file timing written to `metrics.json`.
- A progress callback interface (`on_stage_start`, `on_stage_end`, `on_file_done`) that the CLI uses now and the GUI will use later.

**Acceptance criteria**
- Logging a disallowed field raises an error in tests.
- A full stub run with the sentinel transcript produces logs, metrics and console output that never contain the sentinel.
- A stub failure whose exception message contains the sentinel never leaks it to logs or console.

---

## P1-08: Stage runner

**Goal:** run a stage range over many files with resume and skip-and-report.

**Scope** (in `murmurs.runner`)
- Stage registry in pipeline order: `ingest`, `load_audio`, `diarize`, `transcribe`, `assign_speakers`, `detect_phi`. In this phase each is a thin stage calling the stub backends.
- Run `stages.start` through `stages.end` for every input file.
- Resume: skip a stage for a file if its artifact exists and its recorded input hash and stage config hash match. `--no-resume` reruns everything.
- Skip and report: a failure marks that file's stage `failed`, skips the file's later stages, and continues with other files.
- End-of-job summary: counts of done, failed and skipped files, with failed file IDs and error codes.
- Process exit code: 0 if all files succeeded, 1 if any failed, 2 for config or setup errors.
- Sequential execution only. `parallel_workers` is accepted but values above 1 log a notice that parallelism is not yet implemented.
- Wire the runner into `murmurs run`.

**Out of scope:** parallel execution, real stage logic.

**Acceptance criteria**
- Running `ingest` through `transcribe` produces artifacts for only those stages.
- With one file set to fail, other files complete and the summary lists the failure.
- Rerunning after success performs no stage work (verified by stub call counts).
- Changing a stage's config reruns that stage and the stages after it, but not earlier ones.

---

## P1-09: Run manifest and sample selection

**Goal:** a reproducible record of every job, and sample runs.

**Scope**
- Write `RunManifest` at job start and update it at job end.
- Record package version, versions of key dependencies, git commit (if in a git repo, else `null`), resolved config, and input file hashes.
- Model versions are recorded by backends; stubs report `"stub"`.
- Sample selection: when `sample.enabled`, pick `sample.n_files` files using `sample.seed`. The same seed and inputs always pick the same files.
- Sample runs write to a job ID marked as a sample, so they don't mix with full runs.

**Acceptance criteria**
- The manifest contains the resolved config including CLI overrides.
- Two sample runs with the same seed select the same files; a different seed can select different files.
- `n_files` larger than the number of inputs selects all inputs without error.

---

## P1-10: Phase 1 acceptance test

**Goal:** prove the phase goal with one end-to-end test.

**Scope**
- An integration test using `tests/fixtures/`: at least three tiny audio files (a few seconds each; generated silence or tones are fine).
- Configure the stub transcriber to fail on one file.
- Run the full stage range through the CLI entry point.
- Simulate an interruption partway through (e.g. a stub that raises `KeyboardInterrupt` once on a chosen stage), then rerun.

**Acceptance criteria**
- First run: two files complete, one is reported failed, exit code 1.
- After interruption and rerun: completed stages are not redone, and the job finishes.
- Input files are byte-identical and have unchanged modification times.
- No log, metric, manifest or console output contains the sentinel string.
- All checks in `.claude/CLAUDE.md` pass.