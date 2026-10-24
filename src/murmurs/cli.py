"""Command-line interface. A thin wrapper over the ``murmurs`` library."""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import yaml

from murmurs import __version__
from murmurs.backends import registry
from murmurs.backends.registry import BackendKind, WeightsPlan
from murmurs.config import (
    PROVISIONAL_ENTITY_TYPES,
    ConfigError,
    JobConfig,
    parse_assignment,
    resolve_config,
)
from murmurs.errors import SetupError
from murmurs.evalimport import (
    EvalImportError,
    Pairing,
    check_mapping,
    collect_tags,
    import_pairs,
    load_file_list,
    load_mapping,
    pair_files,
    select_pairs,
    write_mapping,
)
from murmurs.evalset import EvalSetError, label_counts, load_dir
from murmurs.logging import ProgressBar
from murmurs.offline import WeightDownloadError, cached_revision, download_weight, enforce_offline
from murmurs.runner import EXIT_SETUP_ERROR, check_weights, run_job
from murmurs.store import StoreError, atomic_writer

EXIT_INTERRUPTED = 130
"""Conventional exit code for Ctrl-C; completed stages are kept for resume."""

app = typer.Typer(
    name="murmurs",
    help="Turn interview audio into speaker-labeled transcripts with proposed redactions.",
    no_args_is_help=True,
    add_completion=False,
    # Rich tracebacks print exception messages and local variables, which can hold
    # transcript text. Crashes are logged safely by murmurs.logging instead.
    pretty_exceptions_enable=False,
)

config_app = typer.Typer(help="Inspect the job config format.", no_args_is_help=True)
app.add_typer(config_app, name="config")

models_app = typer.Typer(
    help="Download model weights ahead of time, and check they are cached for offline runs.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")

eval_app = typer.Typer(
    help="Check and score evaluation sets, and import human-deidentified transcripts as one.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="Job config file (YAML).", show_default=False)
]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the package version and exit.",
        ),
    ] = False,
) -> None:
    """Murmurs command-line interface."""


@app.command()
def run(
    config: ConfigOption,
    input_dir: Annotated[
        Path | None, typer.Option(help="Override data.input_dir.", show_default=False)
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Override data.output_dir.", show_default=False)
    ] = None,
    start: Annotated[
        str | None, typer.Option(help="Override stages.start.", show_default=False)
    ] = None,
    end: Annotated[
        str | None, typer.Option(help="Override stages.end.", show_default=False)
    ] = None,
    sample: Annotated[
        int | None,
        typer.Option(
            help="Sample run on N files (sets sample.enabled and sample.n_files).",
            show_default=False,
        ),
    ] = None,
    no_resume: Annotated[
        bool, typer.Option("--no-resume", help="Rerun every stage (stages.resume=false).")
    ] = False,
    device: Annotated[
        str | None, typer.Option(help="Override compute.device.", show_default=False)
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="Override any field: key.path=value (repeatable). Values are YAML scalars.",
            show_default=False,
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the resolved config and exit.")
    ] = False,
    debug_unsafe: Annotated[
        bool,
        typer.Option(
            "--debug-unsafe",
            help="Write full tracebacks to the log file. They may contain transcript content.",
        ),
    ] = False,
    job_id: Annotated[
        str | None,
        typer.Option(
            help="Job ID (output folder name). Defaults to the config file name; reuse it "
            "to resume.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Run a job. Precedence: schema defaults < config file < CLI flags.

    Named flags are applied after --set, so they win if both set the same field.
    """
    overrides: list[tuple[str, Any]] = []
    try:
        overrides.extend(parse_assignment(item) for item in set_ or [])
    except ConfigError as exc:
        _fail_config(exc)
    named: list[tuple[str, Any]] = [
        ("data.input_dir", None if input_dir is None else str(input_dir)),
        ("data.output_dir", None if output_dir is None else str(output_dir)),
        ("stages.start", start),
        ("stages.end", end),
        ("compute.device", device),
    ]
    overrides.extend((key, value) for key, value in named if value is not None)
    if sample is not None:
        overrides += [("sample.enabled", True), ("sample.n_files", sample)]
    if no_resume:
        overrides.append(("stages.resume", False))

    try:
        job = resolve_config(config, overrides)
    except ConfigError as exc:
        _fail_config(exc)

    if dry_run:
        typer.echo(yaml.safe_dump(job.model_dump(mode="json"), sort_keys=False), nl=False)
        return

    bar = ProgressBar(sys.stderr) if sys.stderr.isatty() else None
    try:
        summary = run_job(
            job,
            job_id or config.stem,
            progress=bar,
            console=None if bar is None else bar.console,
            debug_unsafe=debug_unsafe,
            install_crash_hook=True,
        )
    except (SetupError, StoreError) as exc:
        _fail_setup(str(exc))
    except KeyboardInterrupt:
        typer.echo("interrupted; rerun the same command to resume", err=True)
        raise typer.Exit(EXIT_INTERRUPTED) from None
    finally:
        if bar is not None:
            bar.close()
    raise typer.Exit(summary.exit_code)


def _fail_config(exc: ConfigError) -> NoReturn:
    _fail_setup(str(exc))


def _fail_setup(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(EXIT_SETUP_ERROR)


@config_app.command("schema")
def config_schema() -> None:
    """Print the job config JSON Schema."""
    typer.echo(json.dumps(JobConfig.model_json_schema(), indent=2))


def _weights_plan(config: Path) -> WeightsPlan:
    """Weights for every backend the config selects, whatever its stage range."""
    try:
        job = resolve_config(config, [])
    except ConfigError as exc:
        _fail_config(exc)
    try:
        plan = registry.required_weights(job, list(BackendKind))
    except SetupError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_SETUP_ERROR) from None
    for message in plan.unavailable:
        typer.echo(f"unavailable  {message}", err=True)
    return plan


@models_app.command("download")
def models_download(config: ConfigOption) -> None:
    """Download every model the config selects into the local Hugging Face cache.

    Then runs any one-time preparation the models need (e.g. format conversion). Run
    this while connected to the internet. Gated models need HF_TOKEN set (or a saved
    `hf auth login`).
    """
    plan = _weights_plan(config)
    failed = False
    for spec in plan.weights:
        typer.echo(f"downloading  {spec.label}")
        try:
            result = download_weight(spec)
        except WeightDownloadError as exc:
            typer.echo(f"failed       {spec.label}: {exc}", err=True)
            failed = True
            continue
        typer.echo(f"ok           {spec.label} ({_size(result.size_bytes)}, {result.revision})")
    if not failed:
        for model in plan.prepared:
            if model.is_ready():
                typer.echo(f"prepared     {model.label}")
                continue
            typer.echo(f"preparing    {model.label}")
            try:
                model.prepare()
            except Exception as exc:  # library text may quote paths or contents
                typer.echo(f"failed       {model.label} ({type(exc).__name__})", err=True)
                failed = True
                continue
            typer.echo(f"prepared     {model.label}")
    if not plan.weights and not plan.unavailable:
        typer.echo("no model weights to download")
    raise typer.Exit(EXIT_SETUP_ERROR if failed or plan.unavailable else 0)


@models_app.command("check")
def models_check(config: ConfigOption) -> None:
    """Report whether every model the config selects is cached. Never uses the network.

    Exit code 0 if all are cached, 2 otherwise.
    """
    enforce_offline()
    plan = _weights_plan(config)
    missing = False
    for spec in plan.weights:
        revision = cached_revision(spec)
        if revision is None:
            typer.echo(f"missing      {spec.label}")
            missing = True
        else:
            typer.echo(f"cached       {spec.label} ({revision})")
    for model in plan.prepared:
        if model.is_ready():
            typer.echo(f"prepared     {model.label}")
        else:
            typer.echo(f"unprepared   {model.label}")
            missing = True
    if not plan.weights and not plan.unavailable:
        typer.echo("no model weights needed")
    raise typer.Exit(EXIT_SETUP_ERROR if missing or plan.unavailable else 0)


def _size(n_bytes: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n_bytes >= scale:
            return f"{n_bytes / scale:.1f} {unit}"
    return f"{n_bytes} B"


OptionalConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Job config whose deid.entity_types to check against (default: the provisional set).",
        show_default=False,
    ),
]


@eval_app.command("check")
def eval_check(
    directory: Annotated[Path, typer.Argument(help="Folder of *.md evaluation interviews.")],
    config: OptionalConfigOption = None,
) -> None:
    """Validate an evaluation set and print label counts (never text).

    Exit code 0 if every interview is valid, 2 otherwise.
    """
    if config is None:
        types = [entity.name for entity in PROVISIONAL_ENTITY_TYPES]
    else:
        try:
            types = [entity.name for entity in resolve_config(config, []).deid.entity_types]
        except ConfigError as exc:
            _fail_config(exc)
    try:
        sets = load_dir(directory, types)
    except EvalSetError as exc:
        _fail_setup(f"invalid evaluation set: {exc}")
    totals: dict[str, Counter[str]] = {}
    for evalset in sets:
        counts = label_counts(evalset)
        words = sum(len(segment.words) for segment in evalset.transcript.segments)
        detail = ", ".join(f"{name}={counts[name]}" for name in sorted(counts))
        turns = len(evalset.transcript.segments)
        typer.echo(
            f"{evalset.set_id}  {evalset.interview_type.value}  turns={turns}"
            f"  words={words}  labels={sum(counts.values())}  ({detail})"
        )
        totals.setdefault(evalset.interview_type.value, Counter()).update(counts)
    for interview_type, counts in sorted(totals.items()):
        typer.echo(f"total {interview_type}: labels={sum(counts.values())}")


@eval_app.command("run")
def eval_run(
    directory: Annotated[Path, typer.Argument(help="Folder of *.md evaluation interviews.")],
    config: ConfigOption,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Results folder (default: a new <output_dir>/eval/<timestamp>/). Rerun with "
            "the same folder to resume: interviews already detected from the same text with "
            "the same settings are skipped, and every interview is scored again.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Run PHI detection (NER + LLM, offline) on the evaluation set and print recall and
    precision counts per entity type (never text).

    Saves each interview's detections (offsets and types only) and annotated rendering as
    it goes, then writes scores.json. Shows a progress bar on a terminal. Exit code 0
    when the run completes, 2 on setup errors or if detection stops (rerun with --out to
    continue), 130 on Ctrl-C.
    """
    from datetime import UTC, datetime
    from time import perf_counter

    from murmurs.config import Stage
    from murmurs.errors import PipelineError
    from murmurs.evaluate import (
        SavedDetections,
        detections_path,
        format_type_line,
        load_detections,
        names_recall,
        overall,
        score_set,
        settings_digest,
        transcript_digest,
    )
    from murmurs.models import StageStatus
    from murmurs.stages import stage_config
    from murmurs.stages.deid import deidentify, render_annotated

    enforce_offline()
    try:
        job = resolve_config(config, [])
    except ConfigError as exc:
        _fail_config(exc)
    try:
        sets = load_dir(directory, [entity.name for entity in job.deid.entity_types])
    except EvalSetError as exc:
        _fail_setup(f"invalid evaluation set: {exc}")

    if out is not None:
        run_dir = out.expanduser().absolute()
    else:
        run_dir = job.data.output_dir / "eval" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    settings_sha = settings_digest(stage_config(Stage.DETECT_PHI, job))
    saved: dict[str, SavedDetections] = {}
    todo = []
    for evalset in sets:
        path = detections_path(run_dir, evalset.set_id)
        previous = load_detections(path, transcript_digest(evalset), settings_sha)
        if previous is None:
            todo.append(evalset)
        else:
            saved[evalset.set_id] = previous
    typer.echo(f"results folder: {run_dir}")
    typer.echo(f"interviews: {len(sets)}  already detected: {len(saved)}  to detect: {len(todo)}")

    if todo:
        try:
            check_weights(job, [Stage.DETECT_PHI])
        except SetupError as exc:
            _fail_setup(str(exc))
        bar = ProgressBar(sys.stderr) if sys.stderr.isatty() else None
        failure: str | None = None
        interrupted = False
        done = 0
        ner = llm = None
        try:
            ner = registry.create_ner_detector(job)
            llm = registry.create_llm_client(job)
            if bar is not None:
                bar.on_group_start("eval run", [Stage.DETECT_PHI], 1, 1, len(todo))
            for evalset in todo:
                started = perf_counter()
                layer, _ = deidentify(
                    evalset.transcript,
                    ner,
                    llm,
                    job.deid.entity_types,
                    evalset.set_id,
                    job.deid.llm_server.max_model_len,
                    job.deid.redaction_threshold,
                    max_concurrency=job.deid.llm_server.max_concurrency,
                )
                seconds = perf_counter() - started
                with atomic_writer(run_dir / f"{evalset.set_id}.annotated.txt") as f:
                    f.write(render_annotated(evalset.transcript, layer))
                record = SavedDetections(
                    transcript_sha256=transcript_digest(evalset),
                    settings_sha256=settings_sha,
                    seconds=round(seconds, 2),
                    layer=layer,
                )
                # written last: saved detections mean the interview is complete
                with atomic_writer(detections_path(run_dir, evalset.set_id)) as f:
                    f.write(record.model_dump_json())
                saved[evalset.set_id] = record
                done += 1
                if bar is not None:
                    bar.on_stage_end(evalset.set_id, Stage.DETECT_PHI, StageStatus.DONE, seconds)
        except SetupError as exc:
            failure = str(exc)
        except PipelineError as exc:
            failure = f"evaluation stopped ({exc.code})"
        except KeyboardInterrupt:
            interrupted = True
        finally:
            for backend in (llm, ner):
                if backend is not None:
                    backend.close()
            if bar is not None:
                bar.close()
        if failure is not None or interrupted:
            resume = (
                f"{done} of {len(todo)} interviews detected in this run; "
                f"rerun with --out {run_dir} to continue"
            )
            if interrupted:
                typer.echo(f"interrupted; {resume}", err=True)
                raise typer.Exit(EXIT_INTERRUPTED)
            _fail_setup(f"{failure}; {resume}")

    scores = [
        score_set(evalset, saved[evalset.set_id].layer.spans, saved[evalset.set_id].seconds)
        for evalset in sets
    ]
    for score in scores:
        typer.echo(f"{score.set_id}  {score.interview_type}  {score.seconds:.1f}s")
        for name, type_score in score.by_type.items():
            typer.echo(f"  {format_type_line(name, type_score)}")
    total = overall(scores)
    typer.echo("overall")
    for name, type_score in total.items():
        typer.echo(f"  {format_type_line(name, type_score)}")
    found, labels = names_recall(total)
    percent = f"{100 * found / labels:.1f}%" if labels else "-"
    typer.echo(f"names recall: {found}/{labels} ({percent})")
    payload = {
        "sets": [score.model_dump(mode="json") for score in scores],
        "overall": {name: score.model_dump(mode="json") for name, score in total.items()},
        "names_recall": {"found": found, "labels": labels},
    }
    with atomic_writer(run_dir / "scores.json") as f:
        f.write(json.dumps(payload, indent=2))
    typer.echo(f"scores and annotated renderings: {run_dir}")


def _entity_type_names(config: Path | None) -> list[str]:
    """The entity type names of a job config, or of the provisional set without one."""
    if config is None:
        return [entity.name for entity in PROVISIONAL_ENTITY_TYPES]
    try:
        return [entity.name for entity in resolve_config(config, []).deid.entity_types]
    except ConfigError as exc:
        _fail_config(exc)


RawDirArgument = Annotated[
    Path, typer.Argument(help="Folder of raw .txt transcripts (read only).", show_default=False)
]
DeidDirArgument = Annotated[
    Path,
    typer.Argument(
        help="Folder of the human-deidentified copies, same relative paths (read only).",
        show_default=False,
    ),
]


OnlyOption = Annotated[
    Path | None,
    typer.Option(
        "--only",
        help="CSV listing the transcripts to use (e.g. those a reviewer finished); "
        "names match with or without .txt, ignoring case.",
        show_default=False,
    ),
]
ColumnOption = Annotated[
    str, typer.Option("--column", help="CSV column holding the file names for --only.")
]


@eval_app.command("tags")
def eval_tags(
    raw_dir: RawDirArgument,
    deid_dir: DeidDirArgument,
    mapping: Annotated[Path, typer.Argument(help="Tag mapping file to create or update (YAML).")],
    config: OptionalConfigOption = None,
    only: OnlyOption = None,
    column: ColumnOption = "Annotated",
) -> None:
    """Find the redaction tags reviewers used and add them, blank, to a mapping file.

    Fill in each tag's entity type (or IGNORE), then run `murmurs eval import`. Existing
    entries are kept. Prints counts only; the tags are written only to the file.
    """
    types = _entity_type_names(config)
    try:
        pairing, _ = _pair_files_checked(raw_dir, deid_dir, mapping, only, column)
        existing = load_mapping(mapping) if mapping.exists() else {}
        scan = collect_tags(pairing)
        new = write_mapping(mapping, scan.tags, existing, types)
    except EvalImportError as exc:
        _fail_setup(str(exc))
    tags = scan.tags
    blank = sum(1 for key in set(tags) | set(existing) if not existing.get(key))
    typer.echo(
        f"pairs={len(pairing.pairs)}  turns={scan.turns}  tags={len(tags)}"
        f"  occurrences={sum(tags.values())}  new={new}  still blank={blank}"
    )
    if scan.no_turns:
        names = ", ".join(str(p) for p in scan.no_turns[:10])
        more = f" and {len(scan.no_turns) - 10} more" if len(scan.no_turns) > 10 else ""
        typer.echo(
            f"warning: {len(scan.no_turns)} pairs have a copy with no SPEAKER_N: turns: "
            f"{names}{more}",
            err=True,
        )
    typer.echo(f"mapping: {mapping}")


def _pair_files_checked(
    raw_dir: Path, deid_dir: Path, output: Path, only: Path | None, column: str
) -> tuple[Pairing, list[str]]:
    """Pair the transcripts, refusing an output path inside either input folder, and keep
    only those listed in ``only`` when given. Returns the pairing and the listed names
    that match no pair (warned about on stderr)."""
    target = output.expanduser().absolute()
    for folder in (raw_dir, deid_dir):
        if target.is_relative_to(folder.expanduser().absolute()):
            raise EvalImportError("outputs must be outside the input folders")
    pairing = pair_files(raw_dir, deid_dir)
    if only is None:
        return pairing, []
    total = len(pairing.pairs)
    pairing, not_found = select_pairs(pairing, load_file_list(only, column))
    typer.echo(f"file list: {len(pairing.pairs)} of {total} pairs selected")
    if not_found:
        names = ", ".join(not_found[:10])
        more = f" and {len(not_found) - 10} more" if len(not_found) > 10 else ""
        typer.echo(f"warning: {len(not_found)} listed names match no pair: {names}{more}", err=True)
    return pairing, not_found


@eval_app.command("import")
def eval_import(
    raw_dir: RawDirArgument,
    deid_dir: DeidDirArgument,
    out_dir: Annotated[
        Path, typer.Argument(help="New or empty folder for the evaluation interviews.")
    ],
    mapping: Annotated[
        Path,
        typer.Option(
            "--tags", "-t", help="Tag mapping from `murmurs eval tags`.", show_default=False
        ),
    ],
    config: OptionalConfigOption = None,
    only: OnlyOption = None,
    column: ColumnOption = "Annotated",
) -> None:
    """Turn raw and human-deidentified transcripts into an evaluation set.

    Writes one *.md interview per pair plus import_report.json (counts only) to OUT_DIR,
    ready for `murmurs eval check` and `murmurs eval run`. Prints counts and file names
    only. Exit code 0 when the import completes, 2 if it cannot start.
    """
    types = _entity_type_names(config)
    try:
        pairing, not_found = _pair_files_checked(raw_dir, deid_dir, out_dir, only, column)
        tag_map = load_mapping(mapping)
        check_mapping(tag_map, types)
        report = import_pairs(pairing, tag_map, out_dir, [raw_dir, deid_dir], not_found)
    except EvalImportError as exc:
        _fail_setup(str(exc))
    totals: Counter[str] = Counter()
    for item in report.files:
        if not item.imported:
            typer.echo(f"{item.file}  skipped: {item.reason}")
            continue
        totals.update(item.labels)
        labels = ", ".join(f"{name}={n}" for name, n in item.labels.items())
        typer.echo(
            f"{item.file}  {item.interview_type}  turns={item.turns}"
            f"  labels={sum(item.labels.values())} ({labels})  ignored={item.ignored_tags}"
            f"  unaligned={item.unaligned_tags}  untagged_edits={item.untagged_edits}"
            f"  deleted={item.deleted}  tags_without_text={item.tags_without_text}"
        )
    for name, files in (("raw", report.raw_only), ("deidentified", report.deid_only)):
        if files:
            typer.echo(f"only in the {name} folder ({len(files)}): {', '.join(files)}")
    imported = sum(item.imported for item in report.files)
    typer.echo(
        f"imported {imported}/{len(report.files)} pairs, labels={sum(totals.values())}"
        f" ({', '.join(f'{name}={n}' for name, n in sorted(totals.items()))})"
    )
    typer.echo(f"evaluation set and import_report.json: {out_dir}")
