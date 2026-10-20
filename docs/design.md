# Murmurs — Audio Interview to Cleaned Transcript Design Doc

Last updated: Oct 19, 2026

## Summary and motivation

We are building a local pipeline that turns interview audio into a first-pass deidentified transcript, ready for human review. It combines speech-to-text (STT), speaker diarization and large language model (LLM) processing, and runs from the command line or a local web GUI.

Interview data is hard to use in research and clinical settings for two reasons. Raw audio takes time to scrub through manually, and transcription pipelines are not yet common in many research workflows. Even after transcription, transcripts need a long manual deidentification pass to remove personal information such as names and locations before any downstream analysis.

This tool produces a cleaned, deidentified first draft and a review interface, so the human pass is faster and better prioritized.

## Goals, non-goals, and success metrics

The primary target is 100% recall on names and at least 95% recall on every other entity type, with 100% overall as the long-term goal.

**Goals**

- Transcribe interview audio with speaker labels.
- Propose PHI redactions with type and confidence for human review.
- Support review, correction and approval of redactions without altering the original transcript.

**Metrics**

| Metric | Target | Notes |
| --- | --- | --- |
| Recall on names | 100% | Measured on the synthetic evaluation set; gates the move to the review GUI |
| Recall on other entity types | ≥ 95% each | Measured per entity type; 100% is the long-term goal |
| Transcription quality | Human review | Judged on sample runs, since no ground truth exists |
| Throughput | Tracked | Hours of audio processed per hour, per hardware setup |

**Non-goals (initial version)**

- Real-time transcription
- Non-English audio
- Video
- Summarization

## Deployment model

The core pipeline is a Python library; the first release is a single-user desktop/CLI tool built on it.

- **Library:** all pipeline logic, with no UI or server assumptions.
- **CLI:** runs jobs from a config file.
- **GUI:** a local web app over the same library, possibly a native desktop app later.
- **Where models run:** STT, diarization and LLM models run on the user's machine, or on a local server workstation if installed there. No data leaves that machine.

Starting single-user avoids authentication and multi-user security work in the first version.

## Privacy, logging, and audit

All processing is local, the deidentification standard comes from the data owners, and logs never contain transcript content.

- **Deidentification standard:** set by the data owners in their statement of purpose and governance document. The tool does not hard-code a standard; entity types and tags are supplied through configuration and prompts.
- **Data lifecycle:** original audio is retained; deleting it is the user's decision. No intermediate audio files are written; any temporary files are deleted after processing.
- **Access:** no access restrictions on output files for now, since everything runs locally.
- **Logging:** logs, errors and crash reports must never contain transcript text or LLM prompt contents.
- **Audit trail:** every redaction decision records who made it, when, and what changed.

## User workflows

The CLI and GUI run the same jobs; the review view is where humans approve the deidentification.

### CLI

The user runs a single command with a job config file. The config specifies models, transcription and deidentification parameters, hardware settings, data locations, and which stages to run (full pipeline, STT only from audio, or LLM only from existing transcripts). CLI flags override config defaults. Outputs are written to the configured output folder.

### GUI (local web app)

The user drags and drops audio files and picks task options. Options map one-to-one to the CLI config, with technical settings under an advanced section. A per-stage progress bar runs during processing, and the user is alerted when the job finishes.

### Review

For each transcript, the reviewer sees:

- Statistics on how many potential redactions were proposed.
- Highlighted redactions in the text, each with its type (name, company, etc.) and confidence.
- Flagged edge cases (possible PII) for the reviewer to decide on.

The reviewer can:

- Approve or reject each proposed redaction.
- Add redactions the system missed. These are stored as missed examples for improving the LLM later.
- Approve or reject all occurrences of the same name at once. A bulk action by confidence threshold is optional.
- Edit the text by hand.
- Move between redactions with the keyboard, sorted by confidence.

Each transcript has a review state: unreviewed → in progress → complete → signed off. Changes never touch the original transcript (see Data model).

## Pipeline stages

The pipeline has seven stages with defined inputs and outputs, so any range of stages can be run, cached and resumed. No chunked audio files are written: CrisperWhisper handles long audio internally, and both models read the same in-memory waveform.

| # | Stage | Input | Output | Method |
| --- | --- | --- | --- | --- |
| 1 | Ingest and validate | Raw .wav or .m4a files | Validation metadata (JSON) | Format, sample rate, corruption and duration checks |
| 2 | Audio loading | Original audio file | In-memory 16 kHz mono waveform | Decode once and resample in memory; the original file and format are untouched, and no converted file is written |
| 3 | Diarization | Waveform | Speaker turns (JSON) | Pyannote 3.1 on the full recording |
| 4 | Transcription | Waveform | Verbatim and intended transcripts with word-level timestamps | CrisperWhisper 2.0 `transcribe_dual`, which handles long audio itself with no stitching needed |
| 5 | Speaker assignment | Word timestamps + speaker turns | Full speaker-labeled transcript | Assign each word to the overlapping speaker turn; optional LLM pass to fix diarization mistakes |
| 6 | PHI detection | Full transcript | Annotated transcript for review + clean version (PHI above a set threshold redacted) | NER model + LLM ensemble |
| 7 | Human review | Annotated transcript | Clean, reviewed transcript | Review UI |

Regex-based detection is not used, since the expected identifiers do not follow common patterns.

**Constraints from the chosen models**

- **CrisperWhisper 2.0:** `transcribe` accepts a file path or a numpy array with a sample rate. Word timestamps come at no extra cost, so audio-synced review later needs no separate alignment step.
- **CrisperWhisper backends:** the fast `ct2` backend (about 4–5x faster, and required for `transcribe_dual`) has wheels for Linux x86_64 only. macOS and Windows use the slower `transformers` backend and need two passes for verbatim and intended.
- **CrisperWhisper threading:** the `ct2` model must be loaded and run on one dedicated thread, so parallel transcription means separate processes, not threads.
- **CrisperWhisper license:** inference code is MIT, but the standard model weights are under a non-commercial research license.
- **Hotword boosting** for names and rare terms is available only in the commercial Pro models.

## Data model

Every stage reads and writes a shared JSON format, and the original transcript is never modified.

**Transcript**

- Segments: speaker, start time, end time, text.
- Words: text (verbatim and intended), start and end time.

**Redaction spans** (one per proposed or manual redaction)

- Character offsets (start, end)
- Entity type and replacement tag
- Confidence
- Source detectors (NER, LLM, or manual)
- Status: proposed, approved, rejected, or manually added
- Reviewer and timestamp

**Run manifest** (one per job)

- Config used
- Model names and versions
- Code version
- Input file hashes

**Editing model**

The original transcript is immutable. Redaction decisions and hand edits live in a separate layer on top of it, and the clean transcript is rendered from original + layer. This supports undo, audit history and re-rendering.

## Deidentification design

Redactions use indexed tags, stay consistent across chunks through running metadata, and get confidence from detector agreement.

- **Replacement strategy:** indexed tags such as `[PERSON_1]`. The tag set is specified as part of the prompt and config.
- **Consistency across chunks:** the LLM generates and updates metadata for each chunk, recording which entities it has redacted and the tag it used. Later chunks reuse those tags.
- **Confidence:** based on agreement between the NER model and the LLM to start. To be refined later.
- **Edge cases:** ambiguous items (quasi-identifiers, dates, rare conditions, etc.) are flagged as potential PII in the metadata and surfaced to the reviewer. Decisions on a small sample of transcripts are used to refine the redaction set dynamically.

**Detectors**

- **NER:** [OpenMed/privacy-filter-nemotron-v2](https://huggingface.co/OpenMed/privacy-filter-nemotron-v2), a 1.4B-parameter mixture-of-experts token classifier (about 50M active parameters per token) with 55 fine-grained PII labels. It outputs per-token scores, which can feed confidence later.
- **LLM:** GPT-OSS 120B, run locally.

**Gaps the LLM must cover**

- The NER label set has no organization, company, school, hospital or other facility label. These are common in childhood (CECA) and life-events (LEDS) interviews.
- Names are split into `first_name` and `last_name`, so nicknames and partial references need the LLM.
- The NER model was trained on synthetic PII text, not conversational transcripts, and its model card calls it experimental. It needs validating on the synthetic CECA/LEDS set.

**Label mapping**

The NER labels are mapped to the project's tag set in config. Some labels, such as `religious_belief`, `sexuality`, `race_ethnicity`, `marital_status` and `occupation`, describe content these interviews study. Each mapping decides whether a label is redacted, flagged for review, or ignored.

## Configuration

One validated config per job, with CLI flags overriding its defaults; failed files are skipped and reported, and jobs are resumable.

- Config is defined with pydantic and versioned so older configs keep working.
- Presets (e.g. names only) will be added after the first pipeline stage is implemented.
- Batch behavior: if one file fails, skip it, report it, and continue. A rerun picks up from the last completed stage per file.

**Proposed schema**

```python
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field


class Stage(str, Enum):
    INGEST = "ingest"
    LOAD_AUDIO = "load_audio"
    DIARIZE = "diarize"
    TRANSCRIBE = "transcribe"
    ASSIGN_SPEAKERS = "assign_speakers"
    DETECT_PHI = "detect_phi"


class Device(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class DataConfig(BaseModel):
    input_dir: Path
    output_dir: Path
    audio_extensions: list[str] = [".wav", ".m4a"]


class SampleConfig(BaseModel):
    enabled: bool = False                # sample run for human quality review
    n_files: int = Field(5, ge=1)
    seed: int = 0                        # reproducible file selection


class StagesConfig(BaseModel):
    start: Stage = Stage.INGEST
    end: Stage = Stage.DETECT_PHI
    resume: bool = True


class DiarizationConfig(BaseModel):
    model: str = "pyannote/speaker-diarization-3.1"
    num_speakers: int | None = None      # None = let the model estimate
    llm_correction: bool = True


class TranscriptionConfig(BaseModel):
    model: str = "nyralabs/CrisperWhisper2.0_large"
    backend: str = "auto"                # "auto" | "ct2" | "transformers"
    language: str = "en"
    modes: list[str] = ["verbatim", "intended"]
    word_timestamps: bool = True


class EntityType(BaseModel):
    name: str                            # e.g. "PERSON"
    tag: str                             # e.g. "[PERSON_{n}]"
    description: str = ""                # guidance passed to the LLM
    ner_labels: list[str] = []           # e.g. ["first_name", "last_name"]
    action: str = "redact"               # "redact" | "flag" | "ignore"


class DeidConfig(BaseModel):
    ner_model: str = "OpenMed/privacy-filter-nemotron-v2"
    llm_model: str = "openai/gpt-oss-120b"
    entity_types: list[EntityType]
    prompt_path: Path | None = None
    redaction_threshold: float = Field(0.5, ge=0, le=1)
    flag_edge_cases: bool = True


class ComputeConfig(BaseModel):
    device: Device = Device.AUTO
    max_gpu_memory_gb: float | None = None
    parallel_workers: int = 1            # worker processes, not threads


class JobConfig(BaseModel):
    schema_version: int = 1
    data: DataConfig
    sample: SampleConfig = SampleConfig()
    stages: StagesConfig = StagesConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    deid: DeidConfig
    compute: ComputeConfig = ComputeConfig()
```

`parallel_workers` counts processes because the CrisperWhisper `ct2` model must stay on one thread. `ner_labels` and `action` on each entity type carry the NER label mapping described under Deidentification design.

## Evaluation and testing

Deidentification is evaluated on a synthetic interview set with planted PHI.

- **Transcription quality:** evaluated by human review for now. A sample run transcribes a handful of files from the full dataset so reviewers can check quality before the full job runs.
- **Evaluation set:** synthetic interviews of 15 minutes to 1 hour, in two styles: childhood experiences (CECA interview) and life events and difficulties within the last year (LEDS interview), with planted PHI.
- **Regression suite:** rerun whenever models or prompts change, reporting per-entity precision and recall.
- **Feedback loop:** approved, rejected and manually added redactions from review are stored as labeled data.
- **Tests:** unit and integration tests for each pipeline stage.

## Operations and packaging

The tool picks open-source models that fit available GPU memory, parallelizes where memory allows, and is packaged with uv.

- **Platform:** Linux x86_64 with an NVIDIA GPU for now, which enables the fast CrisperWhisper `ct2` backend.
- **Resources:** a set of open-source models sized to available GPU memory, with CPU as a last-resort fallback.
- **Parallelism:** transcription and deidentification run in parallel when memory allows.
- **Observability:** per-stage progress reporting, PHI-free structured logs, and timing metrics.
- **Packaging:** uv as the package manager.
- **Extensibility:** STT, diarization, NER and LLM backends sit behind interfaces so they can be swapped without touching pipeline code.
- **Output formats:** plain text for transcripts; JSON for metadata and other artifacts.

## Phases and milestones

Version 1 is built in four phases, CLI only until phase 4; each phase ends with a check before the next one starts.

### Phase 1: Foundations

- uv project and library skeleton with backend interfaces for STT, diarization, NER and LLM
- Pydantic config schema, loading and CLI flag overrides
- JSON data model and run manifest
- PHI-free structured logging and timing metrics
- Stage runner: run any stage range, resume, skip and report failed files

**Done when:** a job with stub stages runs end to end, skips a deliberately failing file, and resumes after an interruption.

### Phase 2: Transcription and evaluation set

- Ingest and validate, audio loading, diarization, transcription, speaker assignment
- LLM diarization correction, off or on via config
- Sample runs on a handful of files
- Synthetic CECA and LEDS evaluation set with planted PHI and ground-truth labels

Testing uses example audio provided by the project owner, since the pipeline must exist before real data can be run.

**Done when:** reviewers judge transcripts of the example audio acceptable, and the synthetic evaluation set is complete.

### Phase 3: Deidentification

- NER and LLM ensemble with NER label mapping (redact, flag, ignore)
- Indexed tags kept consistent across chunks
- Confidence from detector agreement; edge-case flagging
- Annotated and clean transcript outputs
- Regression suite reporting per-entity precision and recall

**Done when:** recall on the synthetic set is 100% for names and at least 95% for every other entity type.

### Phase 4: Review GUI

- Local web app: drag-and-drop upload, job options, per-stage progress, completion alert
- Review view: redaction stats, highlighted spans with type and confidence, flagged edge cases
- Approve, reject, add and hand-edit; bulk actions by name; keyboard navigation by confidence
- Review states, audit trail, rendering the clean transcript from original plus decisions
- Storing reviewer corrections as labeled data

**Done when:** a reviewer can take a transcript from unreviewed to signed off, and the exported clean transcript reflects every decision.

## Deferred to later versions

- Audio playback synced to the transcript in review (likely an early version 2 feature; word timestamps are already produced)
- Conflict handling when multiple reviewers open the same file
- Multi-user server deployment with authentication
- Native desktop app
- Liability and sign-off enforcement
- Presets
- SRT/VTT and other audio-linked output formats

## Open questions

- [ ] How confidence scoring evolves beyond detector agreement
- [ ] Confirm CrisperWhisper decodes .m4a from a file path. If not, our loader decodes it and passes a numpy array; either way no converted file is written.
- [ ] Target hardware: GPT-OSS 120B needs roughly one 80 GB GPU; is a smaller fallback model needed for other machines?
- [ ] Which NER labels are redacted, flagged or ignored for CECA and LEDS interviews