"""Command-line entry point. Translation logic lives in the library, not here."""

from __future__ import annotations

import sys

import click

from perevod.translator import DEFAULT_MAX_TOKENS, Translator


@click.group()
def main() -> None:
    """Offline Russian to Korean translation."""


@main.command()
@click.argument("text", required=False)
@click.option("--model", default=None, help="Model repository id to load instead of the default.")
@click.option(
    "--max-tokens",
    type=int,
    default=DEFAULT_MAX_TOKENS,
    show_default=True,
    help="Upper bound on generated tokens.",
)
def translate(text: str | None, model: str | None, max_tokens: int) -> None:
    """Translate Russian TEXT into Korean.

    Reads standard input when TEXT is omitted or given as "-".
    """
    source = sys.stdin.read() if text is None or text == "-" else text
    source = source.strip()
    if not source:
        raise click.UsageError("No source text. Pass it as an argument or on standard input.")

    click.echo(Translator(model).translate(source, max_tokens=max_tokens))
