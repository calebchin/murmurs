"""Command-line interface. A thin wrapper over the ``murmurs`` library."""

from typing import Annotated

import typer

from murmurs import __version__

app = typer.Typer(
    name="murmurs",
    help="Turn interview audio into speaker-labeled transcripts with proposed redactions.",
    no_args_is_help=True,
    add_completion=False,
)


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
