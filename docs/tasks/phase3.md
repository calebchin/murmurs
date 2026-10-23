# Phase 3: Deidentification — Tasks (MVP first)

Phase goal (from `docs/design.md`): recall on the synthetic set is 100% for names and at least 95% for every other entity type.

This file starts with an **MVP** subset (Oct 22, 2026): just enough NER + LLM deidentification to run the whole pipeline offline on this machine and measure throughput and a first recall number. Confidence modelling, recall and precision tuning, and the regression suite come after the MVP (see "Deferred"). Do the tasks in order; finish one, run all checks, summarize, and stop.

Decisions this phase builds on are in `docs/tasks/phase2.md`: D6 and D22 (the vLLM server), D9 (deidentification uses the final intended transcript's canonical text), D10 (provisional tag set and NER label actions), D21 (stage groups).

---

## P3-01: NER backend

**Goal:** a `NERDetector` for `OpenMed/privacy-filter-nemotron-v2` that returns character spans over a transcript's canonical text.

**Scope** (in `murmurs.backends`)
- First, a spike, reported before building on it: load offline (`local_files_only`); the label scheme (BIO or BIOES) and how character offsets come back; the maximum sequence length and the windowing an hour of text needs; speed, and GPU memory next to the vLLM server at `gpu_memory_utilization` 0.85.
- The backend imports `transformers` in its factory, uses `resolve_device`, runs one forward pass per overlapping token window, and merges windows into character spans. The raw NER label is the span's `entity_type`; the mean token probability is its `confidence`; `sources` is `[ner]`. `close()` frees the GPU. Library errors become `PipelineError("NER_FAILED")`.
- Registered under the model name, with a `WeightSpec` limited to the files inference needs.

**Acceptance criteria**
- Unit test: decoding spans from fake token predictions handles the tag scheme, overlapping windows and offsets that match the text exactly.
- `@pytest.mark.gpu`: planted fake PHI in a short transcript is found, offline, with the network guard active.

## P3-02: LLM detection and merging

**Goal:** the LLM finds PHI the NER model cannot (organizations, schools, hospitals, facilities, nicknames, partial names), and both detectors' spans merge into one set.

**Scope** (in `murmurs.stages`)
- The canonical text is chunked by segment line with context lines (as in P2-07). The prompt (files under `murmurs/stages/prompts/`) lists `deid.entity_types` with their descriptions. The LLM returns `{entities: [{line, text, entity_type}]}`. Each entity's text must occur exactly in that line, and every occurrence becomes a span. Unknown types and text not found are rejected and counted.
- NER spans are mapped through `EntityType.ner_labels` and `action`: `ignore` drops the span, `flag` marks it as a flagged edge case, `redact` keeps it.
- Overlapping spans are merged (sources unioned). Confidence is simple agreement: both detectors give 1.0, NER alone gives its probability, the LLM alone gives 0.7.
- Tags are indexed per entity type by normalized surface, in order of first appearance (`[PERSON_1]`), so every occurrence of a name gets the same tag.

**Acceptance criteria**
- With a scripted LLM and a stub NER: valid entities become spans at every occurrence; invalid ones are rejected and counted; overlapping NER and LLM spans merge with confidence 1.0; flagged and ignored NER labels behave as configured; the same surface gets the same tag everywhere.
- Sentinel test: the sentinel never appears in logs.

## P3-03: `detect_phi` stage and outputs

**Goal:** stage 6 writes the redaction layer plus annotated and clean transcripts, rendered from the original and the layer.

**Scope**
- `detect_phi` runs on the final intended transcript (D9; verbatim in a verbatim-only job) and replaces the phase 1 placeholder.
- Outputs, in the file's folder: `detect_phi.intended.json` (`RedactionLayer`), `detect_phi.intended.annotated.txt` (inline `{{PERSON_1|Maria}}`, flagged spans marked), `detect_phi.intended.clean.txt` (spans with action redact and confidence at least `deid.redaction_threshold` replaced by tags, with speaker labels and timestamps), and a counts-only `detect_phi.report.json`.
- The runner gives `detect_phi` the NER and LLM backends. The LLM stays open from `assign_speakers` through `detect_phi` (D21). The stage's config hash includes the deid prompt hash.
- `examples/job.yaml` holds the provisional tag set (D10).

**Acceptance criteria**
- Renderings never change the original transcript; the clean text differs from the canonical text only inside redacted spans.
- The LLM backend is created once per job and closed after `detect_phi`.
- Sentinel test: logs, metrics, manifest and console never contain the sentinel.

## P3-04: Basic evaluation and the offline MVP run

**Goal:** a first recall number on the MVP evaluation set, and proof the whole pipeline runs offline with a throughput measure.

**Scope**
- `murmurs eval run <dir> -c job.yaml` runs the P3-02 detection on each evaluation transcript and prints per-type recall and precision as counts, plus names recall (a label counts as found if a span of the same type overlaps it). Never prints text. Writes `scores.json`.
- `metrics.json` gains a job summary: total audio seconds, wall time per stage group (including model loads), and the overall real-time factor.
- An end-to-end job on the example audio through `detect_phi`, audited with strace for non-Unix connections, then rerun with the network disconnected (a human step).

**Acceptance criteria**
- `murmurs eval run` output contains counts only.
- `metrics.json` reports the real-time factor for a completed job.
- The offline rerun produces the same outputs; strace shows Unix-socket connections only.

---

## P3-05: LLM throughput

**Goal:** the LLM passes (speaker correction and deidentification) finish in a few minutes per hour of audio, without runaway responses.

**Why:** in the first end-to-end run (53-minute podcast), deidentification took 626 s of the job's 1,003 s. Requests ran one at a time at about 200 generated tokens/s; responses had no length cap, two ran to the context limit, and a cut-off response was retried as is, three times.

**Scope**
- New `deid.llm_server` settings (config schema 5): `max_output_tokens` (8192), `max_concurrency` (64) and `reasoning_effort` (`low`/`medium`/`high`, default `medium`). Of these, only `reasoning_effort` is part of the LLM stages' config hashes; the chunk sizes below are too, so changing them reruns the stage.
- The client caps every response at `max_output_tokens` and sends `reasoning_effort`. A response cut off at the cap raises `LLM_OUTPUT_TRUNCATED` at once, without retrying.
- Both LLM passes send a file's chunks concurrently (up to `max_concurrency`) and process results in chunk order. A truncated or invalid chunk is split in half and the halves are tried; a one-line chunk that still fails is counted as failed.
- Both passes use small chunks: speaker correction chunks own about 1,024 transcript tokens and deidentification chunks about 1,000 (both were a quarter of the context window).
- `LLMClient` exposes running `usage` (requests, tokens, truncations, invalid outputs, time); the correction and deid reports record per-file usage and chunk splits (both reports go to schema 2).

**Out of scope:** batching requests across files (needs runner changes), multi-process transcription.

**Acceptance criteria**
- Results are in chunk order and identical with and without concurrency.
- A truncated chunk is split until its parts succeed; every line is answered exactly once; a failed request stops the file.
- The client never retries a truncated response, and its usage counts stay exact under concurrent use.
- Changing `reasoning_effort` or a chunk size reruns the LLM stages; changing `max_concurrency` or `max_output_tokens` does not.
- GPU: the podcast job and `murmurs eval run` are rerun to measure the speed-up, comparing `reasoning_effort` low and medium on the evaluation set.

**Measured** (RTX PRO 6000, 96 GB; 68-minute podcast, `reasoning_effort: medium`)
- Output tokens per second rise with concurrent requests: 196 (1), 619 (8), 930 (32), 1,166 (64). The model reasons for 2,000 to 5,000 tokens per request whatever the chunk size.
- With 4,096-token caps and quarter-context chunks, 19 of 45 correction requests were cut off and split; the pass took 398 s, slower than before P3-05 (356 s). Hence the higher cap and small chunks.
- Speaker correction at concurrency 64: 150 s with quarter-context chunks, 125 s with 2,048-token chunks, 95 s with 1,024-token chunks.
- Deidentification at concurrency 64 with 2,000-token chunks: 137 s (712 s per audio hour before P3-05 on the same podcasts, 121 s now).
- With these defaults, the podcast job (3 files, 3.5 hours of audio) took 1,000 s: audio 363 s, speaker correction 345 s, deidentification 264 s, or 98 s and 75 s per audio hour for the two LLM passes (about 250 s and 712 s before P3-05).
- Evaluation set with these defaults: names 52/52 and about 2 times faster than 2,000-token chunks at concurrency 8. Each run leaves 1 or 2 different labels uncovered, with 1,000- and 2,000-token chunks alike: vLLM output varies with how requests are batched, even at temperature 0.
- Speaker correction with 1,024-token chunks relabels 4.6% of a file's words against 3.5% with quarter-context chunks; the two runs' final labels differ on 3.0% of words. Which is more accurate needs speaker ground truth.
- Evaluation set: `low` effort is about 4.5 times faster than `medium` but missed a name (51/52) and left 7 labels uncovered (medium: 52/52, none uncovered), so `medium` stays the default.
- The evaluation set now tags every mention of each name, place and organization, not only the first (32 labels added; pet names are not tagged). Before, a detector that found only first mentions scored 100% on names.

---

## P3-06: Run configuration for a 24-hour budget

**Goal:** about 600 hours of audio through transcription and PHI detection in about 24 hours of compute on one RTX PRO 6000.

**Decisions (Oct 23, 2026):** transcribe verbatim only, with speculative decoding; skip LLM speaker correction for now (`diarization.llm_correction: false`; diarization and rule-based labels stay); deidentify the verbatim transcript (D9 changed); export the original transcript with the deidentification outputs.

**Scope**
- `transcription.draft_model` (config schema 5, default `null`): a smaller CrisperWhisper 2.0 model for speculative decoding. With it, each mode is decoded in its own speculative pass. `murmurs models download` downloads and converts it; it is part of the transcription config hash only when set.
- PHI detection runs on verbatim whenever it is transcribed.
- Lines the LLM sees are capped at about 250 tokens: longer speaker segments are split between words (`llm_lines`), so a chunk that keeps failing can always be split until every part is answered. Renderings still show whole segments.
- `transcripts/<file>/` also gets, once detection is done: the original transcript (`verbatim.json`, which the redaction offsets point into), the redaction layer (`verbatim.redactions.json`, with each span's detectors and confidence), the annotated and clean renderings, and `deid_report.json`.
- `transcripts/` mirrors the input folder tree: a file at `site_a/p01/interview.wav` under `input_dir` exports to `transcripts/site_a/p01/interview-<id>/` (each folder name made safe; the design doc assumes input file and folder names hold no PHI).

**Measured** (same 3 podcasts, 3.5 hours of audio, `draft_model: nyralabs/CrisperWhisper2.0_turbo`)
- Speculative decoding: on a 20-minute slice, verbatim took 48.9 s per audio hour against 60.0 s without it and 72.7 s for the dual pass. It is not bit-identical to the large model alone: 5 of 4,603 words differ (0.11%), and each mode is deterministic on its own.
- Job: 539 s in all (was 1,000 s with correction and both modes): audio group 255 s, speaker assignment 0.3 s, PHI detection 270 s. Per file, transcription takes about 49 s and detection about 63 s per audio hour, plus about 2 minutes of model loading per job. Projected for 600 hours: about 20 hours.
- Detection on the podcasts: no truncated responses, no failed chunks.
- Evaluation set: names 52/52 in every run. Uncovered labels vary from run to run (0 to 6): now and then a valid response for one chunk lists fewer entities, and the labels in its lines are missed.

---

## Deferred after the MVP

- Confidence beyond detector agreement (design open question).
- LLM metadata carried across chunks for tag consistency (the MVP indexes tags after detection instead).
- Deidentifying the verbatim transcript.
- The regression suite with per-entity precision and recall gates, and recall tuning towards the phase goal.
- The full synthetic evaluation set (D8).
