"""Turn hand-authored pair files into reproducible, leakage-free training splits."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perevod.dataset import (
    MANIFEST_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALID_FILENAME,
    DatasetError,
    build_training_record,
    prompt_shape_fingerprint,
)

PAIR_DELIMITER = "|||"
INPUT_SUFFIX = ".txt"
DEFAULT_INPUT_DIR = Path("data/derived")
DEFAULT_OUTPUT_DIR = Path("data/splits")

# A guess, not a measurement: far outside any plausible Russian/Korean ratio.
DEFAULT_MAX_LENGTH_RATIO = 8.0

OWNED_FILENAMES = (TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME, MANIFEST_FILENAME)

# Irreplaceable hand-written work; no option, --overwrite included, writes here.
PROTECTED_SUBDIRS = ("data/derived", "data/raw", "data/adapters")


class PrepareError(DatasetError):
    """Base for failures raised while preparing splits."""


class InputError(PrepareError):
    """The corpus holds rows that cannot be turned into records."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = tuple(problems)


class OutputExistsError(PrepareError):
    """The output directory already holds splits an adapter may have been fit to."""


class OutputPathError(PrepareError):
    """The output directory resolves inside one this command never writes to."""


@dataclass(frozen=True)
class PrepareConfig:
    """Same values in, byte-identical splits out.

    Attributes:
        inputs: Empty means `data/derived`.
        max_length_ratio: Misalignment bound; `0` disables the check.
        max_chars: Off by default — a default would drop the longest sentences.
    """

    inputs: tuple[Path, ...] = ()
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 0
    valid_fraction: float = 0.1
    test_fraction: float = 0.1
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO
    max_chars: int | None = None
    allow_source_conflicts: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class PrepareResult:
    """`ratio_stats` and `length_stats` each run (min, median, max); lengths are characters."""

    manifest: dict[str, Any]
    output_dir: Path
    written: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    ratio_stats: tuple[float, float, float] | None = None
    length_stats: tuple[int, int, int] | None = None
    per_input: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class _Pair:
    source: str
    target: str
    source_key: str
    target_key: str
    origin: str
    line: int


def _payload(text: str) -> str:
    """Hangul reaches us both precomposed and decomposed; NFC makes the two comparable."""
    return unicodedata.normalize("NFC", text.strip())


def _key(payload: str) -> str:
    """Comparison only: internal spacing is noise here, but stays in the payload."""
    return " ".join(payload.split())


def _resolve_inputs(inputs: tuple[Path, ...]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    skipped: list[str] = []
    for entry in inputs or (DEFAULT_INPUT_DIR,):
        if entry.is_dir():
            for child in sorted(entry.iterdir(), key=lambda path: path.name):
                if not child.is_file():
                    continue
                if child.suffix == INPUT_SUFFIX:
                    files.append(child)
                else:
                    skipped.append(f"{child}: skipped, not a {INPUT_SUFFIX} file")
        elif entry.is_file():
            files.append(entry)
        else:
            raise InputError([f"{entry}: no such file or directory"])
    return files, skipped


def _read_pairs(
    path: Path, config: PrepareConfig, problems: list[str]
) -> tuple[list[_Pair], int, str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        problems.append(f"{path}: not decodable as UTF-8")
        return [], 0, ""

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    pairs: list[_Pair] = []
    for number, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        # Counted even though skipped: reported line numbers must match the editor's.
        if not stripped or stripped.startswith("#"):
            continue

        found = stripped.count(PAIR_DELIMITER)
        if found != 1:
            problems.append(
                f"{path}:{number}: expected exactly one {PAIR_DELIMITER!r}, found {found}: {stripped}"
            )
            continue

        left, right = stripped.split(PAIR_DELIMITER)
        source, target = _payload(left), _payload(right)
        if not source:
            problems.append(f"{path}:{number}: empty Russian side: {stripped}")
            continue
        if not target:
            problems.append(f"{path}:{number}: empty Korean side: {stripped}")
            continue

        located = f"{path}:{number}"
        if _length_problem(located, source, target, config, problems):
            continue

        pairs.append(
            _Pair(source, target, _key(source), _key(target), str(path), number)
        )

    return pairs, len(lines), text


def _length_problem(
    located: str, source: str, target: str, config: PrepareConfig, problems: list[str]
) -> bool:
    shorter, longer = sorted((len(source), len(target)))
    bound = config.max_length_ratio
    if bound > 0 and longer > bound * max(1, shorter):
        problems.append(
            f"{located}: length ratio {longer / max(1, shorter):.2f} exceeds the bound {bound:.2f} "
            f"({len(source)} against {len(target)} characters); "
            f"raise or disable it with --max-length-ratio"
        )
        return True

    if config.max_chars is not None and longer > config.max_chars:
        problems.append(
            f"{located}: {longer} characters exceeds the --max-chars cap of {config.max_chars}"
        )
        return True

    return False


def _deduplicate(
    pairs: list[_Pair],
    config: PrepareConfig,
    problems: list[str],
    duplicates: list[str],
    conflicts: list[str],
) -> list[_Pair]:
    seen: dict[tuple[str, str], _Pair] = {}
    first_for_source: dict[str, _Pair] = {}
    kept: list[_Pair] = []

    for pair in pairs:
        earlier = seen.get((pair.source_key, pair.target_key))
        if earlier is not None:
            duplicates.append(
                f"{pair.origin}:{pair.line}: duplicate of {earlier.origin}:{earlier.line}, dropped"
            )
            continue

        clash = first_for_source.get(pair.source_key)
        if clash is not None:
            located = (
                f"{pair.origin}:{pair.line}: same source as {clash.origin}:{clash.line} "
                f"with a different translation"
            )
            if not config.allow_source_conflicts:
                problems.append(f"{located}; pass --allow-source-conflicts to keep both")
                continue
            conflicts.append(f"{located}, kept")

        seen[(pair.source_key, pair.target_key)] = pair
        first_for_source.setdefault(pair.source_key, pair)
        kept.append(pair)

    return kept


def _group_key(seed: int, source_key: str) -> str:
    return hashlib.sha256(
        str(seed).encode("utf-8") + b"\x1f" + source_key.encode("utf-8")
    ).hexdigest()


def _assign_splits(pairs: list[_Pair], config: PrepareConfig) -> dict[str, list[_Pair]]:
    """Rows sharing a source stay whole, so no source crosses a split boundary."""
    groups: dict[str, list[_Pair]] = {}
    for pair in pairs:
        groups.setdefault(pair.source_key, []).append(pair)

    total = len(pairs)
    wanted = total * config.valid_fraction, total * config.test_fraction
    plan = [[TRAIN_FILENAME, total - math.floor(wanted[0]) - math.floor(wanted[1])]]
    if config.valid_fraction > 0:
        plan.append([VALID_FILENAME, math.floor(wanted[0])])
    if config.test_fraction > 0:
        plan.append([TEST_FILENAME, math.floor(wanted[1])])

    for (name, size), fraction in zip(plan[1:], (config.valid_fraction, config.test_fraction)):
        if size == 0:
            raise PrepareError(
                f"{name} was requested at {fraction} but {total} pairs yield zero rows; "
                f"at least {math.ceil(1 / fraction)} pairs are needed"
            )

    buckets: dict[str, list[_Pair]] = {name: [] for name, _ in plan}
    index = 0
    for source_key in sorted(groups, key=lambda key: (_group_key(config.seed, key), key)):
        while index < len(plan) - 1 and len(buckets[plan[index][0]]) >= plan[index][1]:
            index += 1
        buckets[plan[index][0]].extend(
            sorted(groups[source_key], key=lambda pair: (pair.target_key, pair.target))
        )

    return buckets


def _render_jsonl(pairs: list[_Pair]) -> str:
    return "".join(
        json.dumps(
            build_training_record(pair.source, pair.target),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for pair in pairs
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_recorded(path: Path, root: Path) -> str:
    """Project-relative where possible, so the manifest stays machine-independent."""
    resolved = path.resolve()
    if resolved == root or root in resolved.parents:
        return resolved.relative_to(root).as_posix()
    return str(resolved)


def _resolve_output_dir(output_dir: Path, root: Path) -> Path:
    resolved = output_dir.resolve()
    for name in PROTECTED_SUBDIRS:
        guarded = (root / name).resolve()
        if resolved == guarded or guarded in resolved.parents:
            raise OutputPathError(f"{resolved} lies inside {guarded}, which is never written to")
    return resolved


def _refuse_to_clobber(resolved: Path) -> None:
    present = sorted(entry.name for entry in resolved.iterdir())
    if not present:
        return

    lines = [
        f"{resolved} already holds: {', '.join(present)}.",
        "An adapter may have been trained against these splits.",
    ]
    manifest_path = resolved / MANIFEST_FILENAME
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict):
            lines.append(
                f"Existing build: seed {existing.get('seed')}, "
                f"fractions {existing.get('valid_fraction')}/{existing.get('test_fraction')}, "
                f"counts {existing.get('counts')}, "
                f"prompt shape {existing.get('prompt_shape_fingerprint')}."
            )
    lines.append(
        "Pass --overwrite, choose another --output-dir, or move the existing directory aside."
    )
    raise OutputExistsError(" ".join(lines))


def prepare_splits(config: PrepareConfig) -> PrepareResult:
    """Nothing is written until the whole corpus validates, so a rejected build creates nothing.

    Raises:
        InputError: Rows that cannot become records, each named by file and line.
        OutputExistsError: The output directory is not empty and `overwrite` is off.
        OutputPathError: The output directory resolves inside a protected one.
        PrepareError: Fractions out of range, or a requested split holds no rows.
    """
    if not 0.0 <= config.valid_fraction < 1.0 or not 0.0 <= config.test_fraction < 1.0:
        raise PrepareError("valid_fraction and test_fraction must each lie in [0.0, 1.0)")
    if config.valid_fraction + config.test_fraction >= 1.0:
        raise PrepareError("valid_fraction and test_fraction must sum to less than 1.0")

    root = Path.cwd().resolve()
    resolved_output = _resolve_output_dir(config.output_dir, root)
    files, skipped = _resolve_inputs(tuple(config.inputs))

    problems: list[str] = []
    duplicates: list[str] = []
    conflicts: list[str] = []
    pairs: list[_Pair] = []
    inputs_record: list[dict[str, Any]] = []
    per_input: list[tuple[str, int]] = []

    for path in files:
        read, lines_read, text = _read_pairs(path, config, problems)
        pairs.extend(read)
        per_input.append((str(path), len(read)))
        inputs_record.append(
            {
                "path": _as_recorded(path, root),
                "sha256": _digest(text),
                "lines_read": lines_read,
                "pairs": len(read),
            }
        )

    kept = _deduplicate(pairs, config, problems, duplicates, conflicts)
    if problems:
        raise InputError(problems)
    if not kept:
        raise InputError([f"no pairs found in {len(files)} input file(s)"])

    buckets = _assign_splits(kept, config)
    rendered = {name: _render_jsonl(rows) for name, rows in buckets.items()}

    if resolved_output.exists() and not config.overwrite:
        _refuse_to_clobber(resolved_output)

    ratios = [
        max(len(pair.source), len(pair.target)) / max(1, min(len(pair.source), len(pair.target)))
        for pair in kept
    ]
    lengths = [len(pair.source) for pair in kept]
    manifest = {
        "seed": config.seed,
        "valid_fraction": config.valid_fraction,
        "test_fraction": config.test_fraction,
        "max_length_ratio": config.max_length_ratio,
        "max_chars": config.max_chars,
        "allow_source_conflicts": config.allow_source_conflicts,
        "prompt_shape_fingerprint": prompt_shape_fingerprint(),
        "inputs": inputs_record,
        "counts": {
            "pairs_read": len(pairs),
            "duplicates_dropped": len(duplicates),
            "unique_sources": len({pair.source_key for pair in kept}),
            **{name.removesuffix(".jsonl"): len(rows) for name, rows in buckets.items()},
        },
        "outputs": [
            {"name": name, "rows": len(buckets[name]), "sha256": _digest(body)}
            for name, body in rendered.items()
        ],
    }
    rendered[MANIFEST_FILENAME] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    resolved_output.mkdir(parents=True, exist_ok=True)
    for name, body in rendered.items():
        (resolved_output / name).write_text(body, encoding="utf-8", newline="\n")
    # A stale split from an earlier run must not survive into training.
    for name in OWNED_FILENAMES:
        if name not in rendered and (resolved_output / name).is_file():
            (resolved_output / name).unlink()

    return PrepareResult(
        manifest=manifest,
        output_dir=resolved_output,
        written=tuple(rendered),
        duplicates=tuple(duplicates),
        conflicts=tuple(conflicts),
        skipped=tuple(skipped),
        ratio_stats=(min(ratios), statistics.median(ratios), max(ratios)),
        length_stats=(min(lengths), int(statistics.median(lengths)), max(lengths)),
        per_input=tuple(per_input),
    )


def render_report(result: PrepareResult) -> str:
    """Kept out of the entry point so the summary is assertable without a terminal."""
    counts = result.manifest["counts"]
    lines = [f"Wrote {len(result.written)} file(s) to {result.output_dir}"]
    lines += [f"  read {pairs} pair(s) from {path}" for path, pairs in result.per_input]
    lines.append(
        f"  {counts['pairs_read']} read, {counts['duplicates_dropped']} duplicate(s) dropped, "
        f"{counts['unique_sources']} unique source(s)"
    )
    lines += [
        f"  {name}: {entry['rows']} row(s)"
        for name, entry in ((item["name"], item) for item in result.manifest["outputs"])
    ]
    if result.ratio_stats:
        low, mid, high = result.ratio_stats
        lines.append(f"  length ratio min {low:.2f} / median {mid:.2f} / max {high:.2f}")
    if result.length_stats:
        low_len, mid_len, high_len = result.length_stats
        lines.append(
            f"  source length min {low_len} / median {mid_len} / max {high_len} characters"
        )
    return "\n".join(lines)
