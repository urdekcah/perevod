"""Command-line entry point. Translation logic lives in the library, not here."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from perevod.prepare import (
    DEFAULT_MAX_LENGTH_RATIO,
    DEFAULT_OUTPUT_DIR,
    PrepareConfig,
    PrepareError,
    prepare_splits,
    render_report,
)
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


@main.command("prepare-data")
@click.argument("inputs", nargs=-1, type=click.Path(path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory the splits and the manifest are written to.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Fixes the split assignment.")
@click.option("--valid-fraction", type=float, default=0.1, show_default=True, help="Share held out for validation.")
@click.option("--test-fraction", type=float, default=0.1, show_default=True, help="Share held out for testing.")
@click.option(
    "--max-length-ratio",
    type=float,
    default=DEFAULT_MAX_LENGTH_RATIO,
    show_default=True,
    help="Misalignment bound on the two sides' character counts; 0 disables it.",
)
@click.option("--max-chars", type=int, default=None, help="Reject a pair whose longer side exceeds this many characters.")
@click.option("--allow-source-conflicts", is_flag=True, help="Keep one source carrying two different translations.")
@click.option("--overwrite", is_flag=True, help="Replace the split files this command owns.")
def prepare_data(
    inputs: tuple[Path, ...],
    output_dir: Path,
    seed: int,
    valid_fraction: float,
    test_fraction: float,
    max_length_ratio: float,
    max_chars: int | None,
    allow_source_conflicts: bool,
    overwrite: bool,
) -> None:
    """Build training splits from hand-authored pair files under INPUTS.

    Reads data/derived when INPUTS is omitted.
    """
    config = PrepareConfig(
        inputs=inputs,
        output_dir=output_dir,
        seed=seed,
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        max_length_ratio=max_length_ratio,
        max_chars=max_chars,
        allow_source_conflicts=allow_source_conflicts,
        overwrite=overwrite,
    )
    try:
        result = prepare_splits(config)
    except PrepareError as error:
        raise click.ClickException(str(error)) from error

    for note in result.skipped + result.duplicates + result.conflicts:
        click.echo(note, err=True)
    click.echo(render_report(result))
