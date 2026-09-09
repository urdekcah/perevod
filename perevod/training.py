# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""LoRA fine-tuning: what to hand the trainer, and what to refuse to start."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from perevod.adapters import AdapterProvenance, write_provenance
from perevod.config import resolve_model_id
from perevod.dataset import (
    DEFAULT_SPLIT_DIR,
    TRAIN_FILENAME,
    VALID_FILENAME,
    validate_split_dir,
)
from perevod.translator import prompt_shape_fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from perevod.dataset import SplitReport

DEFAULT_ADAPTER_ROOT = Path("data/adapters")
DEFAULT_RUN_NAME = "current"

# mlx_lm.lora is deprecated; the console script would resolve through PATH.
TRAINER_MODULE = ("mlx_lm", "lora")

# Validation iterates too; the test split never does — build_argv emits no test mode.
CHECKED_SPLITS = (TRAIN_FILENAME, VALID_FILENAME)

WEIGHTS_SUFFIX = ".safetensors"


class TrainingError(Exception):
    """Base for the errors this module raises."""


class SplitTooSmallError(TrainingError):
    """Refused: a split the trainer iterates holds fewer rows than one batch."""


class AdapterExistsError(TrainingError):
    """Refused: the output path already holds trained weights."""


class AdapterPathError(TrainingError):
    """Refused: the output path is somewhere the trainer must not write."""


class TrainerFailedError(TrainingError):
    """The trainer exited non-zero."""


@dataclass(frozen=True)
class FlagSpec:
    """One trainer flag and its arity."""

    flag: str
    takes_value: bool = True


# The only place a trainer flag name may appear; every entry was read from the tool's own help.
TRAINER_FLAGS: Mapping[str, FlagSpec] = {
    "train": FlagSpec("--train", takes_value=False),
    "model_id": FlagSpec("--model"),
    "data_dir": FlagSpec("--data"),
    "adapter_path": FlagSpec("--adapter-path"),
    "iterations": FlagSpec("--iters"),
    "batch_size": FlagSpec("--batch-size"),
    "learning_rate": FlagSpec("--learning-rate"),
    "num_layers": FlagSpec("--num-layers"),
    "max_seq_length": FlagSpec("--max-seq-length"),
    "seed": FlagSpec("--seed"),
    "mask_prompt": FlagSpec("--mask-prompt", takes_value=False),
    "grad_checkpoint": FlagSpec("--grad-checkpoint", takes_value=False),
}


@dataclass(frozen=True)
class TrainingConfig:
    """A training run's inputs.

    Unset means "emit no flag", so the trainer's own default stands. Rank is absent
    because the pinned trainer takes it from a YAML config, not a flag.
    """

    data_dir: Path = field(default_factory=lambda: DEFAULT_SPLIT_DIR)
    model_id: str | None = None
    adapter_path: Path | None = None
    run_name: str = DEFAULT_RUN_NAME
    iterations: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    num_layers: int | None = None
    max_seq_length: int | None = None
    seed: int | None = None
    mask_prompt: bool = False
    grad_checkpoint: bool = False
    overwrite: bool = False


def resolve_adapter_path(config: TrainingConfig) -> Path:
    """The explicit path if given, else `data/adapters/<run name>` under the current directory.

    Raises:
        AdapterPathError: The path lands inside the splits the run reads from.
    """
    if config.adapter_path is None:
        resolved = (Path.cwd() / DEFAULT_ADAPTER_ROOT / config.run_name).resolve()
    else:
        resolved = config.adapter_path.expanduser().resolve()

    splits = config.data_dir.expanduser().resolve()
    if resolved == splits or splits in resolved.parents:
        msg = f"{resolved}: an adapter may not be written inside the data directory {splits}"
        raise AdapterPathError(msg)
    return resolved


def build_argv(
    config: TrainingConfig,
    adapter_path: Path,
    flags: Mapping[str, FlagSpec] = TRAINER_FLAGS,
) -> list[str]:
    """The trainer invocation, as a list — never a shell string.

    Raises:
        KeyError: `flags` names a field the config lacks, rather than shipping nothing.
    """
    values: dict[str, object | None] = {
        # The trainer's default mode is evaluation; --train is what selects a fit.
        "train": True,
        "model_id": resolve_model_id(config.model_id),
        "data_dir": config.data_dir,
        "adapter_path": adapter_path,
        "iterations": config.iterations,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_layers": config.num_layers,
        "max_seq_length": config.max_seq_length,
        "seed": config.seed,
        "mask_prompt": config.mask_prompt,
        "grad_checkpoint": config.grad_checkpoint,
    }

    argv = [sys.executable, "-m", *TRAINER_MODULE]
    for name, spec in flags.items():
        argv.extend(_render(spec, values[name]))
    return argv


def _render(spec: FlagSpec, value: object | None) -> list[str]:
    if value is None or value is False:
        return []
    if not spec.takes_value:
        return [spec.flag]
    return [spec.flag, str(value)]


def _run(argv: list[str]) -> int:
    """Output is left uncaptured so the trainer's own progress reaches the terminal."""
    return subprocess.run(argv, check=False).returncode  # noqa: S603


def _preflight_batch_size(report: SplitReport, batch_size: int | None) -> SplitReport:
    """Refuse a split too short for one batch, while refusing still costs nothing.

    The trainer guards itself, but only after the base has loaded in the subprocess. An unset
    batch size leaves its own default in force, and the project does not guess at that number.

    Raises:
        SplitTooSmallError: A split the run will iterate is shorter than one batch.
    """
    floor = 1 if batch_size is None else batch_size
    short = [
        f"{report.data_dir / name} holds {rows} row(s)"
        for name in CHECKED_SPLITS
        if (rows := report.rows.get(name)) is not None and rows < floor
    ]
    if short:
        found = "; ".join(short)
        msg = (
            f"{found}. The trainer cannot form a batch from an empty split."
            if batch_size is None
            else f"{found}. Every split the trainer iterates needs at least {batch_size} row(s); "
            f"lower the batch size or add data."
        )
        raise SplitTooSmallError(msg)

    if batch_size is None:
        notice = (
            "batch size is unset, so the trainer's own default stands and row counts were not "
            "compared against it; only empty splits were refused."
        )
        return replace(report, notices=(*report.notices, notice))
    return report


def _is_unfinished_run(adapter_path: Path) -> bool:
    """Whether the directory holds only the debris of a run that died.

    Config lands before the first step and weights only from inside the loop, so files
    without weights hold nothing worth protecting.
    """
    return any(adapter_path.glob("*")) and not any(adapter_path.glob(f"*{WEIGHTS_SUFFIX}"))


def train_adapter(
    config: TrainingConfig,
    *,
    runner: Callable[[list[str]], int] | None = None,
    flags: Mapping[str, FlagSpec] = TRAINER_FLAGS,
) -> tuple[Path, SplitReport]:
    """Fit a LoRA adapter, once the data has been proven worth spending hours on.

    `runner` is the test seam, and keeps mlx_lm out of the import path.

    Raises:
        DatasetError: Bad splits.
        SplitTooSmallError: A checked split is shorter than one batch.
        AdapterPathError: The output path lands inside the splits.
        AdapterExistsError: Output holds weights and `overwrite` is unset.
        TrainerFailedError: The trainer exited non-zero.
    """
    report = _preflight_batch_size(validate_split_dir(config.data_dir), config.batch_size)
    adapter_path = resolve_adapter_path(config)

    if _is_unfinished_run(adapter_path):
        print(  # noqa: T201 -- stdout belongs to the trainer's own progress
            f"{adapter_path}: a previous run left this unfinished, with no weights written. "
            f"Continuing into it. Its files are not removed, so choose another run name if you "
            f"need a clean directory.",
            file=sys.stderr,
        )
    elif not config.overwrite and any(adapter_path.glob("*")):
        msg = (
            f"{adapter_path}: already holds an adapter. Training it again may cost hours, so "
            f"pass overwrite to replace it or choose another run name."
        )
        raise AdapterExistsError(msg)

    argv = build_argv(config, adapter_path, flags)
    adapter_path.mkdir(parents=True, exist_ok=True)

    code = (_run if runner is None else runner)(argv)
    if code != 0:
        invocation = f"{sys.executable} -m {' '.join(TRAINER_MODULE)}"
        msg = f"{invocation} exited {code}; its own output above says why"
        raise TrainerFailedError(msg)

    write_provenance(adapter_path, _provenance_for(config))
    return adapter_path, report


def _provenance_for(config: TrainingConfig) -> AdapterProvenance:
    """Recorded here because nothing downstream can reconstruct it."""
    return AdapterProvenance(
        base_model_id=resolve_model_id(config.model_id),
        prompt_shape_fingerprint=prompt_shape_fingerprint(),
        created_at=datetime.now(tz=UTC).date().isoformat(),
        mlx_lm_version=_trainer_version(),
    )


def _trainer_version() -> str | None:
    try:
        return version("mlx-lm")
    except PackageNotFoundError:
        return None
