# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Command-line entry point. Translation logic lives in the library, not here."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import click

from perevod.adapters import AdapterProvenanceError
from perevod.dataset import DEFAULT_SPLIT_DIR, DatasetError, validate_split_dir
from perevod.prepare import (
    DEFAULT_MAX_LENGTH_RATIO,
    DEFAULT_OUTPUT_DIR,
    PrepareConfig,
    PrepareError,
    prepare_splits,
    render_report,
)
from perevod.training import (
    DEFAULT_RUN_NAME,
    TRAINER_FLAGS,
    TrainingConfig,
    TrainingError,
    train_adapter,
)
from perevod.translator import DEFAULT_MAX_TOKENS, Translator

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., None]")


@click.group()
def main() -> None:
    """Offline Russian to Korean translation."""


@main.command()
@click.argument("text", required=False)
@click.option("--model", default=None, help="Model repository id to load instead of the default.")
@click.option(
    "--adapter-path",
    default=None,
    help="Fine-tuned adapter directory to load over the base model.",
)
@click.option(
    "--allow-provenance-mismatch",
    is_flag=True,
    help="Use an adapter trained against another base or prompt shape anyway.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=DEFAULT_MAX_TOKENS,
    show_default=True,
    help="Upper bound on generated tokens.",
)
def translate(
    text: str | None,
    model: str | None,
    adapter_path: str | None,
    max_tokens: int,
    *,
    allow_provenance_mismatch: bool,
) -> None:
    """Translate Russian TEXT into Korean.

    Reads standard input when TEXT is omitted or given as "-".
    """
    source = sys.stdin.read() if text is None or text == "-" else text
    source = source.strip()
    if not source:
        msg = "No source text. Pass it as an argument or on standard input."
        raise click.UsageError(msg)

    try:
        translator = Translator(
            model,
            adapter_path=adapter_path,
            allow_provenance_mismatch=allow_provenance_mismatch,
        )
    except AdapterProvenanceError as error:
        raise click.ClickException(str(error)) from error

    click.echo(translator.translate(source, max_tokens=max_tokens))


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
@click.option(
    "--valid-fraction",
    type=float,
    default=0.1,
    show_default=True,
    help="Share held out for validation.",
)
@click.option(
    "--test-fraction",
    type=float,
    default=0.1,
    show_default=True,
    help="Share held out for testing.",
)
@click.option(
    "--max-length-ratio",
    type=float,
    default=DEFAULT_MAX_LENGTH_RATIO,
    show_default=True,
    help="Misalignment bound on the two sides' character counts; 0 disables it.",
)
@click.option(
    "--max-chars",
    type=int,
    default=None,
    help="Reject a pair whose longer side exceeds this many characters.",
)
@click.option(
    "--allow-source-conflicts",
    is_flag=True,
    help="Keep one source carrying two different translations.",
)
@click.option("--overwrite", is_flag=True, help="Replace the split files this command owns.")
def prepare_data(  # noqa: PLR0913 -- one parameter per click option
    inputs: tuple[Path, ...],
    *,
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


def _trainer_option(name: str, **kwargs: Any) -> Callable[[F], F]:  # noqa: ANN401
    """A CLI option named after the trainer flag it feeds, so the two cannot drift apart."""
    return click.option(TRAINER_FLAGS[name].flag, name, **kwargs)


@main.command("validate-data")
@_trainer_option(
    "data_dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_SPLIT_DIR,
    show_default=True,
    help="Directory holding the splits.",
)
def validate_data(data_dir: Path) -> None:
    """Check the splits the trainer would read, without starting a run."""
    try:
        report = validate_split_dir(data_dir)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error

    for notice in report.notices:
        click.echo(notice, err=True)
    for name, rows in report.rows.items():
        click.echo(f"{name}: {rows} row(s)")


@main.command()
@_trainer_option(
    "data_dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_SPLIT_DIR,
    show_default=True,
    help="Directory holding the splits.",
)
@_trainer_option("model_id", default=None, help="Model repository id to fine-tune.")
@_trainer_option(
    "adapter_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Where the adapter lands; defaults to data/adapters/<run name>.",
)
@_trainer_option("iterations", type=int, default=None, help="Training steps.")
@_trainer_option("batch_size", type=int, default=None, help="Minibatch size.")
@_trainer_option("learning_rate", type=float, default=None, help="Adam learning rate.")
@_trainer_option("num_layers", type=int, default=None, help="Layers to fine-tune; -1 for all.")
@_trainer_option("max_seq_length", type=int, default=None, help="Longest sequence trained on.")
@_trainer_option("seed", type=int, default=None, help="Trainer PRNG seed.")
@_trainer_option("mask_prompt", is_flag=True, help="Fit only the Korean side, not the prompt.")
@_trainer_option("grad_checkpoint", is_flag=True, help="Trade speed for memory.")
@click.option("--run-name", default=DEFAULT_RUN_NAME, show_default=True, help="Names the output.")
@click.option("--overwrite", is_flag=True, help="Replace an adapter already in the output path.")
def train(**options: object) -> None:
    """Fine-tune a LoRA adapter on the splits under DATA.

    Every unset option leaves the trainer's own default in force.
    """
    config = TrainingConfig(**options)  # type: ignore[arg-type]
    try:
        adapter_path, report = train_adapter(config)
    except (DatasetError, TrainingError) as error:
        raise click.ClickException(str(error)) from error

    for notice in report.notices:
        click.echo(notice, err=True)
    click.echo(f"Adapter written to {adapter_path}")
