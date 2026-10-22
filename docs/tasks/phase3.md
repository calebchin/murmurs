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

## Deferred after the MVP

- Confidence beyond detector agreement (design open question).
- LLM metadata carried across chunks for tag consistency (the MVP indexes tags after detection instead).
- Deidentifying the verbatim transcript.
- The regression suite with per-entity precision and recall gates, and recall tuning towards the phase goal.
- The full synthetic evaluation set (D8).
