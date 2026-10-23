# murmurs

Speech to text for clinical interviews, with an automated deidentification pass. Everything runs locally; see `docs/design.md` for the full design.

## 1. Install

Linux x86_64 with an NVIDIA GPU, the system `ffmpeg`, and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## 2. Write a job config

Copy the example, which lists every setting with its default:

```bash
cp examples/job.yaml my_job.yaml
```

Then edit `my_job.yaml`. At minimum, set the two paths:

```yaml
data:
  input_dir: /path/to/interview/audio    # read only; never modified
  output_dir: /path/to/murmurs/output    # must not be inside input_dir
```

For the fast configuration (verbatim transcript only, speculative decoding, no LLM speaker correction), also set:

```yaml
diarization:
  llm_correction: false

transcription:
  modes: [verbatim]
  draft_model: nyralabs/CrisperWhisper2.0_turbo
```

Keep the `deid.entity_types` list from the example: it is required, and it sets which kinds of PHI are redacted, flagged or ignored.

The output folder name is the config file name (`my_job` here), unless you pass `--job-id`.

## 3. Prepare the models (once, while online)

```bash
uv run murmurs models download -c my_job.yaml   # weights, CTranslate2 conversion, vLLM server environment
uv run murmurs models check -c my_job.yaml      # confirms everything is cached; never uses the network
```

Rerun `models download` if you change a model in the config, such as adding `draft_model`. Runs never use the network.

## 4. Run the job

```bash
uv run murmurs run -c my_job.yaml
```

- Check the resolved settings without running: add `--dry-run`.
- Try a few files first: add `--sample 3`. The output goes to `<job>-sample`.
- Resume: rerun the same command after an interruption or failure. Finished stages are skipped, and changed settings rerun only the stages they affect.

## 5. Outputs

Reviewer files are in `<output_dir>/<job>/transcripts/<subfolders>/<audio name>-<id>/`, where `<subfolders>` mirrors where the audio file sits under `input_dir` (`site_a/p01/interview.wav` → `transcripts/site_a/p01/interview-5b82188d/`). Characters other than letters, digits, `_`, `.` and `-` become `_`. The ID suffix is the start of the audio's content hash:

| File | Contents |
| --- | --- |
| `verbatim.txt`, `verbatim.words.tsv` | The speaker-labeled transcript, as text and word by word |
| `verbatim.annotated.txt` | The transcript with each detected entity marked inline, e.g. `{{PERSON_1\|Maria}}` |
| `verbatim.clean.txt` | The redacted transcript, e.g. `[PERSON_1]` |
| `verbatim.redactions.json` | The proposed redactions: position, type, tag, confidence, detectors |
| `verbatim.json` | The original transcript before redaction; redaction offsets point into it |
| `deid_report.json` | Counts only |

Logs, timing (`metrics.json`) and the run manifest are in `<output_dir>/<job>/`.
