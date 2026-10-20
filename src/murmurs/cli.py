"""Command-line interface. A thin wrapper over the ``murmurs`` library."""

import json
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import yaml

from murmurs import __version__
from murmurs.config import ConfigError, JobConfig, parse_assignment, resolve_config

EXIT_CONFIG_ERROR = 2

app = typer.Typer(
    name="murmurs",
    help="Turn interview audio into speaker-labeled transcripts with proposed redactions.",
    no_args_is_help=True,
    add_completion=False,
)

config_app = typer.Typer(help="Inspect the job config format.", no_args_is_help=True)
app.add_typer(config_app, name="config")


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
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Job config file (YAML).", show_default=False)
    ],
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
    typer.echo("runner not implemented", err=True)
    raise typer.Exit(1)


def _fail_config(exc: ConfigError) -> NoReturn:
    typer.echo(str(exc), err=True)
    raise typer.Exit(EXIT_CONFIG_ERROR)


@config_app.command("schema")
def config_schema() -> None:
    """Print the job config JSON Schema."""
    typer.echo(json.dumps(JobConfig.model_json_schema(), indent=2))
