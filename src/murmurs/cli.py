"""Command-line interface. A thin wrapper over the ``murmurs`` library."""

import json
from typing import Annotated

import typer

from murmurs import __version__
from murmurs.config import JobConfig

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


@config_app.command("schema")
def config_schema() -> None:
    """Print the job config JSON Schema."""
    typer.echo(json.dumps(JobConfig.model_json_schema(), indent=2))
