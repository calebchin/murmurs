# Phase 2: Transcription and Evaluation Set — Tasks

Phase goal (from `docs/design.md`): reviewers judge transcripts of the example audio acceptable, and the synthetic evaluation set is complete.

This phase replaces the stub stages from phase 1 with real logic for stages 1–5 (ingest through speaker assignment) and builds the synthetic evaluation set that phase 3 is measured against. `detect_phi` stays a stub. Do the tasks in order; each depends on the ones before it. Finish one task, run all checks, summarize, and stop.

Every real backend keeps its stub from P1-06. Default tests use stubs and tiny fixtures; tests that load real models are marked `@pytest.mark.gpu`. The example audio from the project owner is never committed: GPU tests read it from the directory in the `MURMURS_EXAMPLE_AUDIO_DIR` environment variable and skip if it is unset.

The pipeline must run on a machine with no network connection (D11). Every real backend loads its weights from the local cache only, and declares what it needs so `murmurs models download` can fetch it ahead of time (P2-00).

---

## Decisions

The design doc was silent or had an open question on each item below. Tasks that depend on an open item must not start until the project owner records a decision here (and, for design open questions, ticks the box in `docs/design.md`). Do not choose an answer yourself.

### MVP reorder (Oct 22, 2026)

To reach a testable MVP sooner, the remaining work runs in this order: P2-08's word-level TSV and a job-level transcript export (no sample runs or reviewer folder yet), a reduced P2-09 (one CECA and one LEDS interview), then a minimal phase 3 (`docs/tasks/phase3.md`), then an offline end-to-end run with throughput. Everything else in P2-08, P2-09 and P2-10 is deferred, not dropped.

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
| D15 | **Diarization model.** The design named pyannote 3.1. | `nvidia/Nemotron-3-Diarization` through `transformers` (≥ 5.18, plus `librosa` for its feature extractor), chosen after a spike on Oct 21, 2026: it loads offline, takes about 2 s and 0.8 GB of GPU memory for an hour of audio, and found the two main speakers of a 53-minute podcast at thresholds 0.3–0.7. `diarization.num_speakers` becomes a cap: speakers beyond the N with the most talk time are labeled `UNKNOWN` (the model cannot be told a speaker count). No smoothing of the model's frame-level output. `JobConfig.schema_version` goes to 3. Recorded in `docs/design.md`. | P2-03, config |
| D16 | **MP3 input.** The design listed .wav and .m4a only. | `.mp3` is added to the default `data.audio_extensions` (the example audio is an MP3; ingest and the torchcodec loader already handle it). Folded into config schema 3. | config |
| D17 | **CrisperWhisper weights and the CTranslate2 conversion.** CrisperWhisper downloads Hugging Face-format weights (`nyralabs/CrisperWhisper2.0_large`, 3.1 GB, not gated) and converts them to CTranslate2 format on first load (about 32 s, offline, about 2.9 GB written to `~/.cache/crisperwhisper` or `$CRISPERWHISPER_CACHE`). Given a folder that already holds a converted model, it loads it directly and never touches the Hugging Face cache. | The conversion happens at download time: a registry entry may declare a prepare step that `murmurs models download` runs after downloading, plus a check of its output. `models check` and the runner's setup check treat an unprepared model as missing (exit code 2). Runs load the converted folder directly. Converted models stay in CrisperWhisper's cache. | P2-00, P2-04 |
| D18 | **Transcript text source.** CrisperWhisper's full `text` contains repetition loops at its 26 s chunk seams that its timed word list leaves out (on a 53-minute podcast: 674 of 10,906 verbatim tokens in 7 blocks; 46 of 9,578 intended). Every timed word appears in the text. | Transcripts are built from the timed words only; segment text is its words joined with spaces. The model's `text` is discarded. | P2-04 |
| D19 | **Segments.** CrisperWhisper returns overlapping 30 s chunks and a flat word list, not segments. | A new segment starts after a word ending in `.`, `?` or `!`, or when the gap to the next word is at least 1 s. P2-05 regroups segments by speaker later. | P2-04 |
| D21 | **Runner order.** Phase 1's runner takes each file through every stage before starting the next file, so GPU models (including the LLM) would be loaded together or reloaded per file. | The runner processes **stage groups**, each across all files before the next group starts: (1) `ingest`; (2) `load_audio`, `diarize`, `transcribe` run per file within the group, so each file is decoded once and its waveform freed before the next (hard rule 5); (3) `assign_speakers`; (4) `detect_phi`. Each backend is created just before the first stage in the run that uses it and closed right after the last, so the audio models and the LLM are never loaded at the same time, and the LLM stays loaded across groups 3 and 4. | P2-05b, P2-06, P2-07 |
| D6 | **How GPT-OSS 120B is served.** | A vLLM OpenAI-compatible server, installed in its own environment (vLLM pins an exact torch version, which must not dictate the pipeline's). murmurs starts it before the first stage in the run that uses the LLM and stops it after the last (D21), including on Ctrl-C and errors. It listens only on a Unix domain socket (`vllm serve --uds <path>`, available in vLLM 0.31.0); the client connects only through that socket and never over TCP, including loopback. A Unix socket counts as local under hard rule 4 because all communication stays within the local operating system. The server runs with usage statistics off, Hugging Face offline mode on, the tokenizer vocabulary pre-fetched, request logging off, and its stdout/stderr discarded so prompts cannot reach any log. GPU memory is not shared with the audio models (D21). The smaller fallback model remains a design open question. | P2-06, P2-07, CLAUDE.md rule 4 |
| D22 | **vLLM server environment and start-up (P2-06 spike, Oct 22, 2026).** vLLM 0.31.0 pins torch 2.13 and transformers <5.18. On this GPU (SM 12.0) it runs GPT-OSS with the Marlin MXFP4 backend; FlashInfer compiles kernels on first start (about 4.5 min; needs `ninja`/`nvcc` from the server environment's `bin` on `PATH`), after which starts take about 1 min. Unix socket paths must be under 108 bytes. Even with offline settings, vLLM's `has_nvidia_artifactory()` makes an HTTPS request to NVIDIA at start-up. | (1) A locked `pyproject.toml` + `uv.lock` (vLLM 0.31.0, Python 3.12) ship inside murmurs; `murmurs models download` builds the environment under `~/.cache/murmurs/vllm-server/<lock hash>/` with `uv sync --frozen`, downloads the hash-checked `o200k_base.tiktoken`, and runs one warm-up start so kernels compile at download time; all three are prepared steps that `models check` and run setup require. (2) Every server process runs with a network guard (`sitecustomize.py` on `PYTHONPATH`) that refuses all non-Unix connections and remote name lookups, plus `FLASHINFER_NO_DOWNLOAD=1`, offline and no-usage-stats variables, and `HF_TOKEN` removed; its stdout/stderr are discarded. (3) Server settings are config fields in a new `deid.llm_server` section (`gpu_memory_utilization` 0.85, `max_model_len` 32768, `startup_timeout_s` 900); `JobConfig.schema_version` goes to 4. (4) The socket lives in a private `/tmp/murmurs-vllm-XXXXXX/` folder (mode 700); the server runs in its own process group and is stopped with SIGINT, then SIGTERM, then SIGKILL. New pipeline dependency: `httpx` (Unix-socket HTTP client). | P2-06, config |
| D7 | **Default for `diarization.llm_correction`.** | Stays `true`. A default run that reaches `assign_speakers` loads GPT-OSS 120B (D6, D22), so its weights and server environment must be set up with `murmurs models download`. | P2-07 |
| D23 | **Interview type in the correction prompt.** P2-07 asked for CECA or LEDS context, but nothing records a file's interview type. | Not yet: the speaker-correction prompt uses generic interviewer/participant context. Interview type is added when prompts are tuned on real interviews. | P2-07 |
| D24 | **Correcting both transcript modes.** | The LLM corrects one transcript per file: intended if configured, else verbatim. In the other mode, each word that overlaps (by time, most overlap, ties to the earlier word) a word whose speaker the LLM changed takes that new speaker; every other word keeps its rule-based speaker. One LLM pass per file, and both modes stay consistent. | P2-07 |
| D8 | **Evaluation set form and authorship.** | MVP (Oct 22, 2026): text transcripts only, written by Claude, one CECA and one LEDS interview (about 2,000–3,000 words each) with fictional PHI marked inline and committed under `eval/synthetic/`. The full set's size, authorship and any synthetic audio are decided after the MVP. | P2-09 |
| D9 | **What ground-truth offsets point into.** | Deidentification and evaluation offsets use the final (LLM-corrected) **intended** transcript (verbatim in a verbatim-only job). Its canonical text is the segment texts joined with `\n`; speaker labels and timestamps are display only (`murmurs.store.canonical_text`). Deidentifying verbatim comes after the MVP. **Changed (Oct 23, 2026):** deidentification now runs on the **verbatim** transcript whenever it is transcribed (intended only in an intended-only job); evaluation offsets are unchanged. | P2-09, phase 3 |
| D10 | **Evaluation tag set and NER label actions.** | Provisional, until the data owners decide. Redact: PERSON (first_name, last_name, user_name), ORGANIZATION (company_name; the LLM covers schools, hospitals and other facilities), LOCATION (street_address, city, county, state, postcode, coordinate), CONTACT (phone_number, email, fax_number, url), ID (ssn, national_id, medical_record_number, health_plan_beneficiary_number, account_number, certificate_license_number, license_plate, vehicle_identifier, device_identifier, employee_id, customer_id, tax_id, unique_id). Flag for review: AGE (age), DATE (date, date_of_birth, date_time), and religious_belief, sexuality, race_ethnicity, political_view, occupation, education_level, employment_status, gender. Ignore: time, country, language and digital-only labels. Lives in `deid.entity_types` (see `examples/job.yaml`). | P2-09, phase 3 |

### Open

| ID | Question | Blocks |
| --- | --- | --- |
| D20 | **Diarization/word-timing lag.** On the 53-minute podcast, Nemotron's speaker changes trail CrisperWhisper's word times by about 0.1 s (median 0.08 s, one Nemotron encoder frame), so 29–34 of 243 speaker changes leave a sentence-opening word with the previous speaker. Shifting turns 0.08–0.12 s earlier before assignment removed about two-thirds of these (34 → 11–14) and halved the opposite error (10 → 4); larger shifts made things worse. Add a `diarization.turn_offset_s` config field (and with what default, tuned on which audio), rely on the P2-07 LLM pass, or both? Needs interview audio to tune. | P2-05 follow-up, P2-07 |

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

**Goal:** stage 3 produces speaker turns with Nemotron-3-Diarization (D15).

**Depends on:** D11, D15 (resolved).

**Scope** (in `murmurs.backends`)
- Config change (D15): `diarization.model` defaults to `nvidia/Nemotron-3-Diarization`; `num_speakers` gets `ge=1` and its meaning becomes a cap; `JobConfig.schema_version` goes to 3 (versions 1 and 2 still load). Update `examples/job.yaml`.
- `NemotronDiarizer` implementing `Diarizer`, registered under `nvidia/Nemotron-3-Diarization`. Load the processor and model with `transformers` inside the factory, never at registry import time.
- Pass the in-memory waveform (numpy array at 16 kHz) to the processor, never a file path. Offline mode: one forward over the whole recording, which the model chunks itself.
- Load weights from the local cache only (D11, P2-00); do not read `HF_TOKEN` at run time. Declare one `WeightSpec` limited to `config.json`, `processor_config.json` and `model.safetensors` (not the `.nemo`, GGUF or demo media). If missing, setup fails with exit code 2 through the P2-00 check.
- `compute.device`: `auto` picks CUDA if available, else CPU with a logged notice. Run in float32.
- Convert the processor's segments (`extract_speaker_dict`, threshold 0.5, no smoothing) to `SpeakerTurn`s sorted by start time, labeled `SPEAKER_00`, `SPEAKER_01`, ... in order of first appearance. With `num_speakers` = N, speakers beyond the N with the most talk time are labeled `UNKNOWN`. Overlapping turns are kept.
- Library exceptions become a `PipelineError` with a fixed code; their text is discarded.
- Report model name and version to the run manifest.
- Load the model once per worker process, not once per file.

**Out of scope:** smoothing, overlap-aware word assignment (P2-05), LLM correction (P2-07), a pyannote backend.

**Acceptance criteria**
- Unit test: conversion from fake processor segments produces sorted, normalized `SpeakerTurn`s, and `num_speakers` relabels the speakers with the least talk time as `UNKNOWN`.
- With the Nemotron weights missing from the cache, a run exits 2 before any file is processed and names the missing repo.
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

## P2-05b: Stage-group runner

**Goal:** the runner completes each stage group across all files before starting the next, and keeps each model in memory only while a stage that uses it runs (D21).

**Depends on:** D21 (resolved).

**Scope** (in `murmurs.runner`, `murmurs.stages`, `murmurs.backends`)
- Stage groups, in order: `ingest`; `load_audio` + `diarize` + `transcribe`; `assign_speakers`; `detect_phi`. Only the stages in `stages.start`..`stages.end` run. Within the audio group, each file runs all its audio stages before the next file, so audio is decoded once per file (hard rule 5) and the waveform is freed before the next file is decoded.
- Per-file run state (status, stages run in this job, stage context) persists across groups. Resume, skip and dependency rules are unchanged: a stage is skipped when complete with the same config hash and none of its dependencies ran again in this job.
- A file that fails a stage is skipped for every later stage and group, with the same status and summary as today.
- Backend lifetime: each backend is created just before the first stage in the run that uses it and closed right after the last such stage. Add a `close()` method to every backend protocol that releases its resources (GPU memory, the CrisperWhisper thread); stubs implement it as a no-op that records the call. The runner never closes backends passed in by the caller.
- Setup checks (config, inputs, cached and prepared weights) still run before any file is processed. A backend that fails to load later, when its group starts, stops the job with exit code 2; completed stages are kept so a rerun resumes.
- Ctrl-C between or within groups keeps completed stages; a rerun resumes.
- Progress: `stage_start`/`stage_end` per file as today, plus `group_start`/`group_end` events with the group's stages and file count. `file_done` fires when a file finishes its last stage in the run or fails.
- Update `docs/design.md` (pipeline and runner description) to match.

**Out of scope:** running LLM requests for several files concurrently, multiple worker processes (`compute.parallel_workers`), the vLLM server (P2-06).

**Acceptance criteria**
- With stub backends and three files, every file finishes a group before any file starts the next group; within the audio group, each file's `load_audio`, `diarize` and `transcribe` run consecutively.
- Each backend is created immediately before its first stage and closed immediately after its last; the audio backends are closed before the NER backend is created; no two groups' backends are open at once except a backend used by both.
- A stage range that skips a group creates none of that group's backends.
- A file that fails ingest, and a file that fails transcription, are skipped for all later stages; other files complete; exit code 1.
- A backend whose factory raises when its group starts stops the job with exit code 2; stages completed before it are kept and a rerun skips them.
- Ctrl-C during the `assign_speakers` group, then a rerun: no audio stage runs again and every file completes.
- The waveform is decoded once per file per run and is freed before the next file is decoded.
- Existing resume, sample-run, duplicate and sentinel tests pass.

---

## P2-06: Local LLM backend

**Goal:** a working `LLMClient` for GPT-OSS 120B that runs entirely on the local machine.

**Depends on:** D6, D21 (resolved), P2-05b.

**Scope** (in `murmurs.backends`)
- `LLMClient` implementation for GPT-OSS 120B, served per D6: a vLLM server in its own environment, reached only over a Unix domain socket. Its `close()` (P2-05b) stops the server.
- Server environment: how it is installed and pinned (a vLLM version at least two weeks old), and how `murmurs models download` prepares it, decided and reported before building on it. The pipeline environment must not depend on vLLM.
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
- The client refuses any address that is not a Unix socket path (no TCP, not even loopback).
- The server is stopped when the job ends, including on Ctrl-C and when a stage fails, and its output never reaches murmurs logs.
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
- Prompt template in a file under the package (overridable later), with generic interviewer/participant context (D23; interview type comes later).
- One LLM pass per file over one mode, mapped to the other by word time (D24).
- Change format: each change names a word range (first and last word number as shown to the LLM), echoes the words' text, and gives the new speaker. A change is rejected if a word number is missing or outside the chunk's own (non-overlap) part, the echoed text does not match, or the speaker is not one of the transcript's speakers (`UNKNOWN` is never a target). A chunk whose output stays invalid after retries keeps its rule-based labels and is counted; a failed request fails the file.
- Outputs: `assign_speakers.<mode>.json`/`.txt` hold the final (corrected) transcript used downstream; `assign_speakers.<mode>.rule_based.json`/`.txt` hold the rule-based one; `assign_speakers.correction.json` (new `SpeakerCorrection` model) holds counts only: chunks, failed chunks, changes proposed, applied and rejected by reason, and words changed per mode. All three are written whether or not correction is on, so nothing stale is left after toggling.
- The `assign_speakers` config hash includes a hash of the prompt files, so editing a prompt reruns the stage.
- Runner: the LLM backend is created for the `assign_speakers` group only when `llm_correction` is on (D21), and closed after its last group.
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

**Status:** deferred after the MVP, except the word-level TSV export, the canonical rendering (`canonical_text`, D9) and a job-level `transcripts/` export of corrected and rule-based transcripts, which are done now.

**Goal:** a sample run produces transcripts a reviewer can read and judge, plus throughput numbers.

**Depends on:** D9 (for the plain-text rendering only).

**Scope**
- A sample run (`sample.enabled`, from P1-09) with real backends ends at `assign_speakers` by default and writes a reviewer folder: `<output_dir>/<job_id>/review/<file_id>.txt` with speaker labels and timestamps, for each mode.
- Define the canonical plain-text rendering of a transcript in one function, used by exports here and by ground-truth offsets in P2-09.
- Word-level export: next to each transcript's `.txt`, write `<stage>.<mode>.words.tsv` through the artifact store, one word per line with a header row: `start`, `end`, `speaker`, `word` (times in seconds, 3 decimals; empty speaker before assignment). Written for both `transcribe` and `assign_speakers` outputs, and copied into the reviewer folder. This is a readable view of the JSON artifacts' word timings, not an SRT/VTT subtitle format (deferred).
- Record throughput in `metrics.json`: audio duration, wall time per stage, and real-time factor per file and per job.
- Record hardware in the manifest: GPU name and memory, CPU count, backend chosen.
- A short reviewer checklist in `docs/review-checklist.md` (speaker labels, missing or hallucinated speech, timestamps), used for the phase sign-off.

**Out of scope:** the review GUI (phase 4), audio playback (deferred).

**Acceptance criteria**
- A stub sample run writes one review file per selected file and mode, and none for unselected files.
- The rendering function is deterministic and covered by a snapshot test.
- The word-level TSV has one row per word in transcript order, its times and speakers match the JSON artifact exactly (to 3 decimals), and words containing tabs or newlines cannot break the format.
- `metrics.json` contains audio duration and real-time factor for every completed file.
- The sentinel never appears in metrics or the manifest (review files are transcripts and may contain it).

---

## P2-09: Synthetic evaluation set

**Status:** reduced for the MVP (D8): two interviews (one CECA, one LEDS) under `eval/synthetic/`, the `GroundTruthLabel`/`GroundTruthSet` models, the loader and validator, and `murmurs eval check`. Generation tooling and the full set come after the MVP.

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
