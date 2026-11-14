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

## 6. Evaluate PHI detection

All evaluation output is counts and label IDs only, never transcript text.

**Synthetic set** (fictional PHI, committed under `eval/synthetic/`):

```bash
uv run murmurs eval check eval/synthetic                     # validate; per-type label counts
uv run murmurs eval run eval/synthetic -c my_job.yaml        # detect, then recall and precision per type
uv run murmurs eval run eval/synthetic -c my_job.yaml --out <folder>   # resume or rescore an earlier run
```

`eval run` writes `scores.json`, plus each interview's detections and annotated rendering, to `<output_dir>/eval/<timestamp>/`. You can also pick a folder with `--out`. Rerun with the same `--out` to resume: interviews already detected with the same settings are skipped, and every interview is scored again.

**Human-deidentified transcripts.** Pairs of raw and hand-deidentified real transcripts can be turned into an evaluation set. These hold real interview text: keep every folder below **outside this repository**, and never commit them.

1. Put the raw `.txt` transcripts in one folder and the deidentified copies in another, at the same relative paths. Each turn starts with `Speaker 1:` or `SPEAKER_1:`. File names must contain `CECA` or `LEDS`.
2. Collect the reviewers' tags (`[NAME]`, `[SCHOOL]`, ...) into a mapping file:
   ```bash
   uv run murmurs eval tags <raw> <deid> tags.yaml -c my_job.yaml
   ```
3. Edit `tags.yaml`: give each tag an entity type from `deid.entity_types`, or `IGNORE`.
4. Import, then check and score as above:
   ```bash
   uv run murmurs eval import <raw> <deid> <new folder> -t tags.yaml -c my_job.yaml
   uv run murmurs eval run <new folder> -c my_job.yaml
   ```

To use only the transcripts a reviewer has finished, add `--only done.csv` to both `eval tags` and `eval import`. The CSV's `Annotated` column (or the one you pick with `--column`) lists the file names. After changing the mapping, re-import into a new folder; `eval run --out` with the earlier results folder rescores without detecting again.
