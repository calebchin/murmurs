"""Command-line interface. A thin wrapper over the ``murmurs`` library."""

import json
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import yaml

from murmurs import __version__
from murmurs.backends import registry
from murmurs.backends.registry import BackendKind, WeightsPlan
from murmurs.config import ConfigError, JobConfig, parse_assignment, resolve_config
from murmurs.errors import SetupError
from murmurs.offline import WeightDownloadError, cached_revision, download_weight, enforce_offline
from murmurs.runner import EXIT_SETUP_ERROR, run_job
from murmurs.store import StoreError

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

    try:
        summary = run_job(
            job,
            job_id or config.stem,
            debug_unsafe=debug_unsafe,
            install_crash_hook=True,
        )
    except (SetupError, StoreError) as exc:
        _fail_setup(str(exc))
    except KeyboardInterrupt:
        typer.echo("interrupted; rerun the same command to resume", err=True)
        raise typer.Exit(EXIT_INTERRUPTED) from None
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
