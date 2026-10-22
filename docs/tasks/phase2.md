# Phase 2: Transcription and Evaluation Set — Tasks

Phase goal (from `docs/design.md`): reviewers judge transcripts of the example audio acceptable, and the synthetic evaluation set is complete.

This phase replaces the stub stages from phase 1 with real logic for stages 1–5 (ingest through speaker assignment) and builds the synthetic evaluation set that phase 3 is measured against. `detect_phi` stays a stub. Do the tasks in order; each depends on the ones before it. Finish one task, run all checks, summarize, and stop.

Every real backend keeps its stub from P1-06. Default tests use stubs and tiny fixtures; tests that load real models are marked `@pytest.mark.gpu`. The example audio from the project owner is never committed: GPU tests read it from the directory in the `MURMURS_EXAMPLE_AUDIO_DIR` environment variable and skip if it is unset.

The pipeline must run on a machine with no network connection (D11). Every real backend loads its weights from the local cache only, and declares what it needs so `murmurs models download` can fetch it ahead of time (P2-00).

---

## Decisions

The design doc was silent or had an open question on each item below. Tasks that depend on an open item must not start until the project owner records a decision here (and, for design open questions, ticks the box in `docs/design.md`). Do not choose an answer yourself.

### Resolved

| ID | Question | Decision | Affects |
| --- | --- | --- | --- |
| D1 | **.m4a decoding** (design open question). | Our loader always decodes; both backends receive a numpy array and never a file path. Whether CrisperWhisper reads .m4a from a path no longer matters. Closed in `docs/design.md`. | P2-02, P2-04 |
| D2 | **Audio decoding library.** | [`torchcodec`](https://github.com/meta-pytorch/torchcodec), using the system `ffmpeg` (assumed installed; ffmpeg 6.1 on the dev machine). New dependency. | P2-01, P2-02 |
| D3 | **Ingest duration limits.** | No duration limits. 15 min to 1 hour is an approximate guide, not a validation rule. Ingest still rejects empty (zero-duration) files. | P2-01 |
| D4 | **Non-Linux transcription backend.** | Out of scope for this phase. `backend: transformers` fails at setup (exit code 2) with "not supported yet", and so does `backend: auto` on a platform where `ct2` is unavailable. Added to "Deferred to later versions" in the design doc. | P2-04 |
| D5 | **Words with no overlapping speaker turn.** | Nearest turn if the gap is within a tolerance, otherwise speaker `UNKNOWN`. The tolerance is a new config field, `diarization.speaker_gap_tolerance_s` (default 0.5 s), and `JobConfig.schema_version` goes to 2. | P2-05, config |
| D11 | **Offline operation.** | The first deployment machine (the current dev machine) runs disconnected. Weights are downloaded on it while connected, then it goes offline. Runs force Hugging Face offline mode, never read `HF_TOKEN`, and fail at setup (exit code 2) if a required model is not cached. `HF_TOKEN` is used only by `murmurs models download`. Recorded in `docs/design.md` → Operations and packaging. | P2-00, P2-03, P2-04, P2-06, P2-10 |
| D12 | **Stub test inputs once the real audio loader exists.** | The `stub_job` fixture in `tests/conftest.py` writes real tone audio (`tests/fixtures/audio.py`) instead of fake bytes, so tests that use the `"default"` audio loader run the real torchcodec loader on CPU. Moved to P2-01 by D13. | P2-01 |
| D13 | **Where the real probe lives, and when `"default"` switches.** | The torchcodec code lives in `murmurs.backends` as a `TorchcodecAudioLoader`; the ingest flow stays in `murmurs.stages`. P2-01 adds the class with a real `probe`, whose `load` delegates to the stub loader until P2-02, and points the `"default"` audio loader at it. Because the real probe rejects fake bytes, the D12 fixture change happens in P2-01, not P2-02. | P2-01, P2-02 |
| D14 | **Downmix to mono.** ffmpeg's built-in downmix (`num_channels=1`) is `(L + R) × 0.7071`, not a mean, so correlated stereo comes out about 1.41× louder and can exceed ±1. | torchcodec resamples to 16 kHz and keeps the source channels; the loader averages channels in numpy. Output stays in [−1, 1]. For multi-channel input, peak memory briefly holds every channel at 16 kHz (about 230 MB per hour per channel) while averaging. | P2-02 |

### Open

| ID | Question | Blocks |
| --- | --- | --- |
| D6 | **How GPT-OSS 120B is served.** In-process (e.g. `transformers`, `vllm` offline mode) or a local server (vLLM, llama.cpp) on `localhost`? Hard rule 4 forbids network calls with interview data; does a loopback-only connection count? Also tied to the design open question on target hardware and a smaller fallback model. Constraint from D11: whichever option is chosen must start and run with no network connection, including any tokenizer vocabulary it fetches on first use. | P2-06, P2-07 |
| D7 | **Default for `diarization.llm_correction`.** The schema defaults it to `true`, so a default phase 2 run needs GPT-OSS 120B loaded. Keep that default, or default to `false` until the LLM is set up? | P2-07 |
| D8 | **Evaluation set form and authorship.** Text transcripts only, or also synthetic audio (e.g. TTS) to test the full pipeline? Who writes the interviews: the project owner, an LLM run locally, or both? How many CECA and LEDS interviews make the set "complete"? | P2-09 |
| D9 | **What ground-truth offsets point into.** Redaction spans use character offsets, but the design does not define which text: the intended or verbatim transcript, and which plain-text rendering (with or without speaker labels and timestamps). Phase 3 will use the same answer. | P2-08, P2-09 |
| D10 | **Evaluation tag set.** The data owners set the deidentification standard. Which entity types (and NER label mappings) do ground-truth labels use? Related to the design open question on NER label actions. | P2-09 |

---

## P2-00: Offline setup

**Goal:** every later task can rely on runs never touching the network, and on one command that fetches all model weights while the machine is still connected.

**Depends on:** D11 (resolved).

**Scope**
- Forced offline mode: a library function (e.g. `murmurs.offline.enforce_offline()`) that sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. The runner calls it before creating any backend. `huggingface_hub` reads these variables when it is imported, so real backends must import model libraries inside their factory or constructor, never at module import time of `murmurs.backends.registry`. Add a test that importing `murmurs` and its registry does not import `huggingface_hub`, `transformers`, `pyannote` or `crisperwhisper`.
- Weight declarations: each registry entry can declare the weights it needs as a list of `WeightSpec` (Hugging Face repo ID, optional revision, optional file patterns), given the job config. Stubs declare none. Real backends add theirs in P2-03, P2-04 and P2-06 (and the NER backend in phase 3). This is registry metadata, not a change to the backend protocols.
- `murmurs models download --config job.yaml`: downloads every declared weight for the backends the config selects (all stages, regardless of `stages.start`/`end`) into the standard Hugging Face cache. Reads `HF_TOKEN` for gated models; never logs or echoes it. Prints repo IDs and sizes only.
- `murmurs models check --config job.yaml`: reports, per declared weight, whether it is fully cached, without network access. Exit code 0 if all are present, 2 otherwise.
- Runner setup check: before any file is processed, verify the declared weights for the job's stage range are cached; if not, exit code 2 with a message naming the missing repo IDs and suggesting `murmurs models download`.
- Record each model's cached revision (commit hash) in the run manifest's `model_versions`.
- Network guard for default tests: an autouse fixture in `tests/conftest.py` that makes any socket connection other than `AF_UNIX` raise. No new dependency. Add a line to `.claude/CLAUDE.md` → Testing saying default tests run with the network blocked.
- New direct dependency: `huggingface_hub` (already a transitive dependency of pyannote and transformers; used here for downloads and cache inspection).

**Out of scope:** the real backends themselves (P2-02 onward), copying caches between machines (not needed per D11).

**Acceptance criteria**
- After `enforce_offline()`, the Hugging Face offline variables are set, and a test confirms the runner calls it before backend creation.
- Importing `murmurs.backends.registry` does not import any model library.
- With a stub registry entry that declares a fake repo, `murmurs models check` and a job run both exit 2 and name the repo; with the repo present in a temporary `HF_HOME` cache, both succeed. No network access is needed for either.
- `murmurs models download` with a dummy `HF_TOKEN` (download call mocked) never writes the token value to logs or console.
- A default test that tries to open a TCP socket fails with the network guard's error.

---

## P2-01: Ingest and validate

**Goal:** stage 1 checks every input file and records why bad files are rejected, without modifying them.

**Depends on:** D2, D3, D12, D13 (resolved).

**Scope** (probe in `murmurs.backends`, ingest flow in `murmurs.stages`; see D13)
- `TorchcodecAudioLoader` in `murmurs.backends` with a real `probe`; its `load` delegates to the stub loader until P2-02. Register it as the `"default"` audio loader.
- Test inputs (D12, moved here by D13): change the `stub_job` fixture in `tests/conftest.py` to write short tone files with `tests/fixtures/audio.py` instead of fake bytes. Tests that pass inputs explicitly with fake bytes and rely on the default loader move to tone files too; tests that inject `StubAudioLoader` directly may keep fake bytes. Generate any .m4a fixtures at test time; never commit audio.
- Read container metadata with `torchcodec` (`AudioDecoder` metadata) without decoding the full file: format, codec, duration, sample rate, channels.
- Checks: extension in `data.audio_extensions`, readable container, at least one audio stream, nonzero duration. No minimum or maximum duration (D3).
- Detect corruption cheaply (header probe plus a short decode at start and end), not by decoding the whole file.
- Write a `ValidationResult` artifact per file. An invalid file is marked `failed` with an error code and skipped for later stages, using the phase 1 skip-and-report path.
- Error codes are a fixed enum (e.g. `UNSUPPORTED_FORMAT`, `UNREADABLE`, `NO_AUDIO_STREAM`, `EMPTY`). Never store library exception text.
- Open input files read-only.

**Out of scope:** decoding the full file or resampling (P2-02).

**Acceptance criteria**
- Fixture .wav and .m4a files pass and report correct duration, sample rate and channels.
- A truncated file, a zero-byte file and a text file renamed to `.wav` each fail with a distinct error code, and the job continues with other files.
- A file with an extension not in `audio_extensions` is not picked up.
- Input bytes and modification times are unchanged after ingest.

---

## P2-02: Audio loading

**Goal:** stage 2 decodes each file once into a 16 kHz mono float32 waveform held in memory.

**Depends on:** D1, D2 (resolved).

**Scope**
- Real `AudioLoader` backend using `torchcodec`: decode, downmix to mono, resample to 16 kHz, return a float32 numpy array plus sample rate. Resample with torchcodec (`AudioDecoder(..., sample_rate=16000)`) and downmix by averaging channels in numpy (D14).
- Decode from the original path straight into memory. No temporary or converted audio files anywhere, including the system temp directory.
- `torchcodec` requires `torch`, and its versions must match torch's per the torchcodec compatibility table. Pin compatible versions; pyannote (P2-03) also depends on torch, so choose a torch version that suits both.
- The waveform is not an artifact. The stage writes a small metadata artifact (sample count, duration, sample rate, input hash) so the runner can record completion.
- The runner keeps the waveform in memory for that file's remaining stages and frees it after the file's last audio stage. On resume, if a later stage needs audio and the waveform is not in memory, it is decoded again.
- Peak memory note in the summary: roughly 230 MB per hour of audio at 16 kHz float32.
- Replace the stub delegation in `TorchcodecAudioLoader.load` (from P2-01, D13) with real decoding. The `stub_job` fixture already writes tone files (D12, done in P2-01).

**Out of scope:** chunking audio (CrisperWhisper and pyannote take the full recording).

**Acceptance criteria**
- A stereo 44.1 kHz fixture loads as mono 16 kHz with the expected sample count (within one sample).
- A .wav and an .m4a fixture with the same content load to waveforms that match within a small tolerance.
- A test that watches the output directory, the input directory and the temp directory finds no new files after loading.
- Resuming a job at `diarize` decodes the audio again and does not rerun `ingest`.

---

## P2-03: Diarization backend

**Goal:** stage 3 produces speaker turns with pyannote 3.1.

**Scope** (in `murmurs.backends`)
- `PyannoteDiarizer` implementing `Diarizer`, loading `diarization.model`.
- Pass the in-memory waveform to pyannote (as a tensor with its sample rate), never a file path.
- Load weights from the local cache only (D11, P2-00); do not read `HF_TOKEN` at run time. Declare the weights for P2-00: the pipeline repo and its sub-models (for 3.1, `pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM`; confirm from the pipeline's `config.yaml`). If any are not cached, setup fails with exit code 2 through the P2-00 check.
- Disable any telemetry or update checks the installed `pyannote.audio` version has, if it has them.
- Honor `diarization.num_speakers` (`None` lets the model estimate) and `compute.device` (`auto` picks CUDA if available, else CPU with a logged notice).
- Convert pyannote output to a list of `SpeakerTurn` with normalized labels (`SPEAKER_00`, `SPEAKER_01`, ...), sorted by start time.
- Report model name and version to the run manifest.
- Load the model once per worker process, not once per file.

**Out of scope:** overlap-aware word assignment (P2-05), LLM correction (P2-07).

**Acceptance criteria**
- Unit test: conversion from a fake pyannote annotation produces sorted, normalized `SpeakerTurn`s.
- With the pyannote weights missing from the cache, a run exits 2 before any file is processed and names the missing repos.
- With `HF_TOKEN` unset, a run with cached weights still succeeds (`@pytest.mark.gpu`).
- `@pytest.mark.gpu`: on a two-speaker example file, diarization returns at least two speakers and turns covering most of the speech, with offline mode forced and the network guard active.

---

## P2-04: Transcription backend

**Goal:** stage 4 produces verbatim and intended transcripts with word timestamps using CrisperWhisper 2.0.

**Depends on:** D1, D4 (resolved).

**Scope** (in `murmurs.backends`)
- First, find out how CrisperWhisper's `ct2` loader locates weights (Hugging Face cache, its own download path, or a local directory) and confirm it loads with no network. Report this in the summary before building on it. Declare its weights for P2-00.
- `CrisperWhisperTranscriber` implementing `Transcriber`, using the `ct2` backend and `transcribe_dual` when both modes are requested, and single-mode `transcribe` when only one is.
- Pass the in-memory waveform and sample rate, never a file path.
- Load and run the `ct2` model on one dedicated thread for the life of the worker process. All calls go through that thread.
- `backend: auto` picks `ct2` on Linux x86_64. `compute.device` decides CUDA or CPU (`auto` picks CUDA if available, else CPU with a logged notice).
- `backend: transformers`, and `backend: auto` on any platform where `ct2` is unavailable, fail at setup with exit code 2 and a "not supported yet" message (D4).
- Output one `Transcript` artifact per mode (`transcribe_verbatim`, `transcribe_intended`, or as the store's naming convention dictates), with word-level timestamps and segments as returned by the model. Speaker fields stay empty until P2-05.
- Plain-text export of each transcript through the artifact store.
- Report model name, backend and version to the run manifest.
- Do not install `faster-whisper` or upstream `ctranslate2`. Add a test or check that fails if either is importable from a different distribution than CrisperWhisper's fork.

**Out of scope:** hotword boosting (Pro models only), languages other than English.

**Acceptance criteria**
- Unit test: conversion from a fake CrisperWhisper result produces valid `Transcript`s whose words fall inside their segments.
- A test asserts every call to the model happens on the same thread.
- `modes: [intended]` produces only the intended transcript.
- Sentinel test: a stub transcriber returning `SENTINEL_PHI_7f3a` runs through the stage and the sentinel never appears in logs, metrics or console output.
- `@pytest.mark.gpu`: on an example file, both transcripts are non-empty, word timestamps increase monotonically, and the last word ends within the audio duration, with offline mode forced and the network guard active.

---

## P2-05: Speaker assignment

**Goal:** stage 5 merges word timestamps and speaker turns into a speaker-labeled transcript.

**Depends on:** D5 (resolved).

**Scope** (in `murmurs.stages`)
- Config change: add `speaker_gap_tolerance_s: float = Field(0.5, ge=0)` to `DiarizationConfig` and bump `JobConfig.schema_version` to 2, as in `docs/design.md` → Configuration. Version 1 configs still load (the field has a default). Update `examples/job.yaml`. If P1-02 was implemented after this change reached the design doc, the field already exists; check it and the version 1 compatibility test, and move on.
- Pure function: `(Transcript, list[SpeakerTurn], gap_tolerance_s: float) -> Transcript`. No model calls.
- Assign each word to the speaker turn it overlaps most. Ties go to the earlier turn.
- A word that overlaps no turn goes to the nearest turn if the gap between them is at most `speaker_gap_tolerance_s`; otherwise its speaker is `UNKNOWN`. Equal distances go to the earlier turn.
- The `assign_speakers` stage config hash includes `speaker_gap_tolerance_s`; the `diarize` stage hash does not, so changing the tolerance reruns only speaker assignment and later stages.
- Rebuild segments as runs of consecutive words from the same speaker. Segment start and end come from their first and last word.
- Apply to every transcript mode produced in P2-04, using the same turns.
- Plain-text export shows one line per segment with speaker label and start time.

**Out of scope:** LLM correction (P2-07).

**Acceptance criteria**
- Hand-built cases: a word fully inside one turn, a word straddling two turns, a word in a gap within the tolerance (nearest speaker), a word in a gap beyond the tolerance (`UNKNOWN`), a word equidistant from two turns, back-to-back turns from the same speaker (merged into one segment), and empty turns.
- A version 1 config without `speaker_gap_tolerance_s` loads with the default of 0.5.
- Changing `speaker_gap_tolerance_s` on resume reruns `assign_speakers` but not `diarize` or `transcribe`.
- Word text and timestamps in the output are identical to the input; only speakers and segment grouping change.
- Verbatim and intended transcripts of the same file get consistent speaker labels for words at the same times.
- Sentinel test: the sentinel never appears in logs.

---

## P2-06: Local LLM backend

**Goal:** a working `LLMClient` for GPT-OSS 120B that runs entirely on the local machine.

**Depends on:** D6.

**Scope** (in `murmurs.backends`)
- `LLMClient` implementation for GPT-OSS 120B, served per D6.
- Structured output: callers pass a pydantic model and get a validated instance back. Invalid output is retried a bounded number of times, then raises a typed pipeline error with a code, never the response text.
- Setup check: if the model cannot be loaded or reached, fail at setup with exit code 2.
- Declare everything the model needs for P2-00, including tokenizer vocabulary files fetched on first use (GPT-OSS's `openai-harmony` encoding is believed to download its vocabulary separately; vLLM documents `TIKTOKEN_ENCODINGS_BASE` for offline use). Confirm the model starts with no network.
- Prompts and responses are never logged, including on error and in `--debug-unsafe` tracebacks' local variables if avoidable. Log only token counts and durations.
- Report model name and version to the run manifest.

**Out of scope:** deidentification prompts (phase 3), fallback models (open question).

**Acceptance criteria**
- The stub `LLMClient` satisfies the protocol and returns deterministic structured output.
- Invalid structured output triggers retries, then a typed error whose message has no response content.
- Sentinel test: a prompt and response containing the sentinel never leak to logs.
- If served over a socket per D6, a test asserts the client refuses any host that is not loopback.
- `@pytest.mark.gpu`: a trivial structured request returns a valid object, with offline mode forced and the network guard active.

---

## P2-07: LLM diarization correction

**Goal:** an optional pass in stage 5 where the LLM fixes speaker label mistakes, switched by `diarization.llm_correction`.

**Depends on:** D6, D7.

**Scope**
- After rule-based assignment, send the speaker-labeled transcript to the LLM in chunks sized to its context window, with overlap so turns at chunk edges are seen whole.
- The LLM may only change speaker labels, at word or segment granularity. It returns label changes as structured output, not rewritten text.
- Validate every change: referenced words exist and speakers are from the existing label set. Invalid changes are dropped and counted, not applied.
- Store both versions: the rule-based assignment and the corrected one, plus a count of changes, so reviewers can compare.
- Prompt template in a file under the package (overridable later), with interview type (CECA or LEDS, interviewer vs participant) as context.
- With `llm_correction: false`, the stage behaves exactly as P2-05 and never loads the LLM.

**Out of scope:** changing transcript text, merging or splitting speakers.

**Acceptance criteria**
- With a stub LLM that relabels a known segment, the corrected transcript has that change and nothing else.
- A stub LLM that tries to change word text, invent a speaker or reference a missing word has those changes rejected, and the output text is identical to the input.
- With `llm_correction: false`, the LLM stub is never called.
- Toggling `llm_correction` reruns only stage 5 and later on resume.
- Sentinel test: the sentinel never appears in logs.

---

## P2-08: Sample runs for reviewers

**Goal:** a sample run produces transcripts a reviewer can read and judge, plus throughput numbers.

**Depends on:** D9 (for the plain-text rendering only).

**Scope**
- A sample run (`sample.enabled`, from P1-09) with real backends ends at `assign_speakers` by default and writes a reviewer folder: `<output_dir>/<job_id>/review/<file_id>.txt` with speaker labels and timestamps, for each mode.
- Define the canonical plain-text rendering of a transcript in one function, used by exports here and by ground-truth offsets in P2-09.
- Record throughput in `metrics.json`: audio duration, wall time per stage, and real-time factor per file and per job.
- Record hardware in the manifest: GPU name and memory, CPU count, backend chosen.
- A short reviewer checklist in `docs/review-checklist.md` (speaker labels, missing or hallucinated speech, timestamps), used for the phase sign-off.

**Out of scope:** the review GUI (phase 4), audio playback (deferred).

**Acceptance criteria**
- A stub sample run writes one review file per selected file and mode, and none for unselected files.
- The rendering function is deterministic and covered by a snapshot test.
- `metrics.json` contains audio duration and real-time factor for every completed file.
- The sentinel never appears in metrics or the manifest (review files are transcripts and may contain it).

---

## P2-09: Synthetic evaluation set

**Goal:** CECA and LEDS synthetic interviews with planted PHI and ground-truth labels, ready for phase 3.

**Depends on:** D8, D9, D10.

**Scope**
- A `GroundTruthLabel` data model in `murmurs.models` (new contract): character offsets per D9, entity type, the exact surface text, and an optional note for edge cases. A `GroundTruthSet` wraps a transcript reference, interview type (CECA or LEDS) and labels, with a `schema_version`.
- A loader and validator: offsets fall within the text, the surface text matches the slice at those offsets, entity types exist in the configured tag set (D10), and labels do not overlap.
- A CLI command `murmurs eval check <dir>` that validates a set and prints per-entity-type counts and per-interview-type totals (counts only, no text).
- Generation or authoring of the interviews per D8. If an LLM generates them, it runs locally, and the prompt and seed are saved alongside the set so it can be regenerated.
- Each interview type covers names (full, first only, nicknames, partial), organizations, schools, hospitals and other facilities, locations and dates, since these are the gaps the design says the LLM must cover.
- A tiny example set (two short interviews) committed in `tests/fixtures/` for tests. Where the full set lives, and whether it is committed, follows D8.

**Out of scope:** scoring detectors and the regression suite (phase 3).

**Acceptance criteria**
- The example set loads and validates.
- A label whose surface text does not match its offsets, a label with an unknown entity type, and overlapping labels each fail validation with a clear message that names the label ID, not its text.
- `murmurs eval check` output on the example set contains counts and no transcript text.
- The full set meets the size and coverage agreed in D8, confirmed by the project owner.

---

## P2-10: Phase 2 acceptance

**Goal:** prove the phase goal: example audio transcribes acceptably and the evaluation set is complete.

**Scope**
- A CPU integration test through the CLI entry point running `ingest` through `assign_speakers` on fixture audio with stub diarization, transcription and LLM backends, one deliberately bad file, and `llm_correction` on.
- A `@pytest.mark.gpu` end-to-end test on the example audio with real backends, run only when asked.
- A sample run on the example audio, with the review folder handed to reviewers using the checklist from P2-08.

**Acceptance criteria**
- CPU test: good files complete through `assign_speakers`, the bad file is reported with its ingest error code, exit code 1.
- No audio files appear anywhere outside the input directory during either test.
- Input files are byte-identical and have unchanged modification times.
- No log, metric, manifest or console output contains the sentinel.
- GPU test: every example file produces speaker-labeled verbatim and intended transcripts, run with offline mode forced, the network guard active and `HF_TOKEN` unset.
- `murmurs models check` exits 0 for the example job config on the deployment machine.
- The sample run on the example audio is done with the machine's network disconnected (a human step; note it in the task summary).
- Reviewers record that the example transcripts are acceptable (a human step; note the date and reviewers in the task summary).
- `murmurs eval check` passes on the full evaluation set.
- All checks in `.claude/CLAUDE.md` pass.
