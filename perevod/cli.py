# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Command-line entry point. Translation logic lives in the library, not here."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import click
from click.core import ParameterSource

from perevod.adapters import AdapterProvenanceError
from perevod.dataset import DEFAULT_SPLIT_DIR, DatasetError, validate_split_dir
from perevod.document import DEFAULT_CHUNK_BUDGET_TOKENS, DocumentResult, Progress
from perevod.evaluation import (
    EvaluationConfig,
    EvaluationError,
    evaluate_pair,
    render_summary,
    write_run,
)
from perevod.files import (
    DEFAULT_ENCODING,
    FileError,
    IncrementalWriter,
    check_destination,
    plan_batch,
    read_source,
)
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


def _render_progress(event: Progress) -> None:
    """Progress goes to stderr so a piped stdout stays byte-clean."""
    if event.phase != "chunk_done":
        return
    where = f"{event.source_name}: " if event.source_name else ""
    batch = f" (file {event.file_index + 1}/{event.file_total})" if event.file_total > 1 else ""
    state = "" if event.ok else " — FAILED"
    click.echo(f"{where}chunk {event.chunk_index + 1}/{event.chunk_total}{batch}{state}", err=True)


def _write_stdout(fragment: str) -> None:
    """Fragments go out raw so stdout stays byte-identical to the assembled document."""
    sys.stdout.write(fragment)


def _report(name: str, result: DocumentResult) -> None:
    if result.aborted:
        click.echo(f"{name}: aborted after too many consecutive failures", err=True)
    if result.chunks_failed:
        failures = ", ".join(str(index) for index in result.chunks_failed)
        click.echo(
            f"{name}: {len(result.chunks_failed)} chunk(s) untranslated: {failures}", err=True
        )


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
    help="Upper bound on generated tokens. Given explicitly, it also fixes the per-chunk ceiling.",
)
@click.option(
    "--input",
    "inputs",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Read the source from this file. Repeatable.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the translation here instead of standard output. Needs exactly one --input.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write one translation per input into this directory. Required for two or more inputs.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing destination.")
@click.option(
    "--encoding",
    default=DEFAULT_ENCODING,
    show_default=True,
    help="Codec the input files are read with.",
)
@click.option(
    "--chunk-budget",
    type=int,
    default=DEFAULT_CHUNK_BUDGET_TOKENS,
    show_default=True,
    help="Chunk size in source tokens.",
)
@click.option(
    "--progress/--no-progress",
    default=None,
    help="Per-chunk progress on stderr. Defaults to on when stderr is a terminal.",
)
@click.pass_context
def translate(  # noqa: PLR0913, PLR0917 -- one parameter per click option
    ctx: click.Context,
    text: str | None,
    model: str | None,
    adapter_path: str | None,
    max_tokens: int,
    inputs: tuple[Path, ...],
    output: Path | None,
    output_dir: Path | None,
    encoding: str,
    chunk_budget: int,
    *,
    progress: bool | None,
    force: bool,
    allow_provenance_mismatch: bool,
) -> None:
    """Translate Russian TEXT into Korean.

    Reads standard input when TEXT is omitted or given as "-". With --input the source is
    read from files instead, split into chunks, and reassembled with its paragraph
    structure intact. Exits 1 when any chunk failed, 2 on a usage or output-path error.
    """
    _reject_conflicting_sources(text, inputs, output, output_dir)

    # Only a value the user typed may override the per-chunk ceiling; the default must not.
    from_command_line = ctx.get_parameter_source("max_tokens") == ParameterSource.COMMANDLINE
    explicit_max_tokens = max_tokens if from_command_line else None
    if progress is None:
        progress = sys.stderr.isatty()

    try:
        pairs = _plan_outputs(inputs, output, output_dir, force=force)
    except FileError as error:
        raise click.UsageError(str(error)) from error

    if not pairs and not inputs:
        _translate_text(text, model, adapter_path, max_tokens, mismatch=allow_provenance_mismatch)
        return

    _translate_files(
        pairs,
        model=model,
        adapter_path=adapter_path,
        allow_provenance_mismatch=allow_provenance_mismatch,
        max_tokens=explicit_max_tokens,
        encoding=encoding,
        chunk_budget=chunk_budget,
        show_progress=progress,
    )


def _reject_conflicting_sources(
    text: str | None,
    inputs: tuple[Path, ...],
    output: Path | None,
    output_dir: Path | None,
) -> None:
    if text is not None and inputs:
        msg = "Pass either TEXT or --input, not both."
        raise click.UsageError(msg)
    if any(str(path) == "-" for path in inputs):
        msg = 'Standard input is the positional form: omit --input or pass "-" as TEXT.'
        raise click.UsageError(msg)
    if output is not None and output_dir is not None:
        msg = "Pass either --output or --output-dir, not both."
        raise click.UsageError(msg)
    if output is not None and len(inputs) != 1:
        msg = "--output needs exactly one --input; use --output-dir for several."
        raise click.UsageError(msg)
    if len(inputs) > 1 and output_dir is None:
        msg = "Two or more --input files need --output-dir."
        raise click.UsageError(msg)
    if (output is not None or output_dir is not None) and not inputs:
        msg = "--output and --output-dir only apply to --input."
        raise click.UsageError(msg)


def _plan_outputs(
    inputs: tuple[Path, ...],
    output: Path | None,
    output_dir: Path | None,
    *,
    force: bool,
) -> list[tuple[Path, Path | None]]:
    """Refuse every unsafe destination before the model is loaded."""
    if not inputs:
        return []
    if output is not None:
        check_destination(output, inputs[0], force=force)
        output.parent.mkdir(parents=True, exist_ok=True)
        return [(inputs[0], output)]
    if output_dir is None:
        return [(inputs[0], None)]
    return list(plan_batch(list(inputs), output_dir, force=force))


def _translate_text(
    text: str | None,
    model: str | None,
    adapter_path: str | None,
    max_tokens: int,
    *,
    mismatch: bool,
) -> None:
    source = sys.stdin.read() if text is None or text == "-" else text
    source = source.strip()
    if not source:
        msg = "No source text. Pass it as an argument or on standard input."
        raise click.UsageError(msg)

    translator = _build_translator(model, adapter_path, mismatch=mismatch)
    click.echo(translator.translate(source, max_tokens=max_tokens))


def _build_translator(model: str | None, adapter_path: str | None, *, mismatch: bool) -> Translator:
    try:
        return Translator(model, adapter_path=adapter_path, allow_provenance_mismatch=mismatch)
    except AdapterProvenanceError as error:
        raise click.ClickException(str(error)) from error


def _translate_files(  # noqa: PLR0913 -- the option surface of one subcommand
    pairs: list[tuple[Path, Path | None]],
    *,
    model: str | None,
    adapter_path: str | None,
    allow_provenance_mismatch: bool,
    max_tokens: int | None,
    encoding: str,
    chunk_budget: int,
    show_progress: bool,
) -> None:
    try:
        sources = [read_source(source, encoding=encoding) for source, _ in pairs]
    except FileError as error:
        raise click.ClickException(str(error)) from error

    # One construction, one model load, however many files follow.
    translator = _build_translator(model, adapter_path, mismatch=allow_provenance_mismatch)
    on_progress = _render_progress if show_progress else None
    failed = False

    for position, ((source, destination), body) in enumerate(zip(pairs, sources, strict=True)):
        writer = IncrementalWriter(destination) if destination is not None else None
        sink = writer.write if writer is not None else _write_stdout
        try:
            result = translator.translate_document(
                body,
                max_tokens=max_tokens,
                budget_tokens=chunk_budget,
                progress=on_progress,
                sink=sink,
                file_index=position,
                file_total=len(pairs),
                source_name=source.name,
            )
        except BaseException:
            if writer is not None:
                writer.abandon()
                click.echo(f"Completed chunks kept in {writer.part_path}", err=True)
            raise
        if writer is not None:
            writer.commit()
        _report(source.name, result)
        failed = failed or bool(result.chunks_failed)

    if failed:
        raise SystemExit(1)


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


@main.command()
@click.option(
    "--test-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Held-out split to score; defaults to data/splits/test.jsonl.",
)
@click.option(
    "--adapter-path",
    default=None,
    help="Adapter directory forming the second arm.",
)
@click.option("--model", "model_id", default=None, help="Base model repository id.")
@click.option("--limit", type=int, default=None, help="Score only the first N rows in file order.")
@click.option(
    "--repeats",
    type=int,
    default=None,
    help="Passes per arm; the spread between them becomes the noise floor.",
)
@click.option("--max-tokens", type=int, default=None, help="Upper bound on generated tokens.")
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory the run record and the index are written to.",
)
@click.option(
    "--bootstrap-seed",
    type=int,
    default=None,
    help="Enable the paired bootstrap with this seed; it can only widen the floor.",
)
@click.option(
    "--allow-provenance-mismatch",
    is_flag=True,
    help="Run against an adapter fitted to another base; the verdict is then withheld.",
)
def evaluate(**options: object) -> None:
    """Score the base model against an adapter on the held-out split.

    Both arms translate the same rows with the same settings. Read the verdict, not the delta.
    """
    supplied = {name: value for name, value in options.items() if value is not None}
    config = EvaluationConfig(**supplied)  # type: ignore[arg-type]

    try:
        run = evaluate_pair(config)
    except (AdapterProvenanceError, DatasetError, EvaluationError) as error:
        raise click.ClickException(str(error)) from error

    target = write_run(run, config.out_dir)
    click.echo(f"Record written to {target}", err=True)
    click.echo(render_summary(run))
