# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Base against adapter on a held-out split.

A leaked split, a mispaired base, mismatched decoding, or identical arm output each
withholds the verdict rather than being footnoted.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from perevod import metrics
from perevod.adapters import read_provenance
from perevod.config import ADAPTER_ENV_VAR, resolve_adapter_path, resolve_model_id
from perevod.dataset import (
    DEFAULT_SPLIT_DIR,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALID_FILENAME,
    RecordShapeError,
    parse_training_record,
)
from perevod.translator import DEFAULT_MAX_TOKENS, Translator, prompt_shape_fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

SCHEMA_VERSION = 1

DEFAULT_EVAL_DIR = Path("data/eval")
INDEX_FILENAME = "index.jsonl"

# Below this share of differing outputs the arms are the same run and no delta means anything.
MIN_DISTINGUISHABLE_FRACTION = 0.5

# A convention, not a significance threshold: sub-point deltas are not narrated as progress.
MIN_REPORTABLE_DELTA_CHRF = 1.0

DEFAULT_REPEATS = 2
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE = 0.95

BASE_ARM = "base"
ADAPTER_ARM = "adapter"

IMPROVED = "improved"
REGRESSED = "regressed"
INDISTINGUISHABLE = "indistinguishable"
UNAVAILABLE = "unavailable"
VERDICTS = (IMPROVED, REGRESSED, INDISTINGUISHABLE, UNAVAILABLE)

FLOOR_CONVENTION = "convention"
FLOOR_REPEAT_SPREAD = "empirical-repeat-spread"
FLOOR_BOOTSTRAP = "bootstrap-ci"

REQUIRED_RECORD_KEYS = (
    "schema_version",
    "run_id",
    "created_at",
    "test_set",
    "prompt_shape_fingerprint",
    "decoding",
    "arms",
    "metric",
    "scores",
    "comparison",
)

_INDISTINGUISHABLE_NOTE = (
    "The arms produced near-identical text. The adapter may be undertrained, or it may never "
    "have been applied at all; this run does not tell those apart."
)
_CONFOUNDED_NOTE = (
    "The adapter was fitted to a different base than the one the base arm loaded, so the two "
    "arms differ in more than the adapter and no delta is attributable."
)


class EvaluationError(Exception):
    """Base for every typed error raised while evaluating."""


class SplitIntegrityError(EvaluationError):
    """The test split cannot be trusted to measure anything."""


class ConfoundedComparisonError(EvaluationError):
    """The two arms would differ in their base as well as their adapter."""


class DecodingMismatchError(EvaluationError):
    """The arms did not decode alike, so a delta would be measuring the decoder."""


class RecordError(EvaluationError):
    """A run record is missing a required field or carries an unknown verdict."""


@dataclass(frozen=True)
class TestSegment:
    """One scored row with its 1-based line number."""

    line: int
    source: str
    target: str


def canonical_text(text: str) -> str:
    """Comparison form for leak detection. Casefold: a re-cased duplicate is still leakage."""
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def load_split(path: Path) -> tuple[TestSegment, ...]:
    """Read a split through the shipped record inverse, never a second parser.

    Raises:
        EvaluationError: The file is absent or holds no records.
        RecordShapeError: A row is unreadable or was built against another prompt shape.
    """
    if not path.is_file():
        msg = f"{path}: no such file"
        raise EvaluationError(msg)

    segments: list[TestSegment] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            source, target = parse_training_record(json.loads(line))
        except (ValueError, RecordShapeError) as error:
            msg = f"{path}:{number}: {error}"
            raise RecordShapeError(msg) from error
        segments.append(TestSegment(line=number, source=source, target=target))

    if not segments:
        msg = f"{path}: holds no records"
        raise EvaluationError(msg)
    return tuple(segments)


def split_digest(segments: Sequence[TestSegment]) -> str:
    """Identity of the scored data; a silently edited split invalidates earlier runs."""
    body = "\n".join(
        f"{canonical_text(segment.source)}\x1f{canonical_text(segment.target)}"
        for segment in segments
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _first_overlaps(
    test: Sequence[TestSegment],
    other: Sequence[TestSegment],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sources: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    for segment in other:
        key = canonical_text(segment.source)
        sources.setdefault(key, segment.line)
        pairs.setdefault((key, canonical_text(segment.target)), segment.line)

    source_hits: list[tuple[int, int]] = []
    pair_hits: list[tuple[int, int]] = []
    for segment in test:
        key = canonical_text(segment.source)
        if key in sources:
            source_hits.append((segment.line, sources[key]))
        pair = (key, canonical_text(segment.target))
        if pair in pairs:
            pair_hits.append((segment.line, pairs[pair]))
    return source_hits, pair_hits


def check_split_isolation(
    segments: Sequence[TestSegment],
    *,
    test_path: Path,
    split_dir: Path | None = None,
) -> tuple[str, ...]:
    """Refuse to score a split overlapping training data. Returns what was compared.

    A source seen in training is leakage even when the translation differs.

    Raises:
        SplitIntegrityError: Any overlap, named by count and first offending line.
    """
    directory = test_path.parent if split_dir is None else split_dir
    compared: list[str] = []
    for name in (TRAIN_FILENAME, VALID_FILENAME):
        other_path = directory / name
        if not other_path.is_file() or other_path.resolve() == test_path.resolve():
            continue

        compared.append(name)
        source_hits, pair_hits = _first_overlaps(segments, load_split(other_path))
        if not source_hits and not pair_hits:
            continue

        detail = [
            f"{len(hits)} {label}, first at {test_path}:{hits[0][0]} = {other_path}:{hits[0][1]}"
            for label, hits in (("shared source(s)", source_hits), ("shared pair(s)", pair_hits))
            if hits
        ]
        msg = (
            f"{test_path} overlaps {other_path}: {'; '.join(detail)}. A score over leaked rows "
            f"reports memorization, not translation quality; rebuild the splits first."
        )
        raise SplitIntegrityError(msg)
    return tuple(compared)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_of(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecodingSettings:
    """What both arms decode with. One value per run, handed to both.

    No determinism control exists upstream, so `repeats` substitutes for one.
    """

    max_tokens: int = DEFAULT_MAX_TOKENS
    repeats: int = DEFAULT_REPEATS

    def as_dict(self) -> dict[str, Any]:
        """The settings as recorded."""
        return {"max_tokens": self.max_tokens, "repeats": self.repeats}

    def fingerprint(self) -> str:
        """Digest of the settings; arms that disagree on it are never compared."""
        return _digest_of(self.as_dict())


@dataclass(frozen=True)
class ArmResult:
    """One arm's scores; `outputs` is the scored first repeat."""

    name: str
    model_id: str
    adapter_path: str | None
    outputs: tuple[str, ...]
    repeat_scores: tuple[float, ...]
    corpus_chrf: float
    segment_scores: tuple[float, ...]
    seconds: float
    decoding_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        """The arm block, with `n` beside its score."""
        return {
            "model_id": self.model_id,
            "adapter_path": self.adapter_path,
            "corpus_chrf": self.corpus_chrf,
            "n": len(self.segment_scores),
            "repeat_scores": list(self.repeat_scores),
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class Comparison:
    """The verdict and everything a reader needs to disbelieve it."""

    verdict: str
    delta: float
    floor: float
    floor_bases: dict[str, float]
    distinguishable_fraction: float
    wins: int
    losses: int
    ties: int
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """The comparison block."""
        return {
            "verdict": self.verdict,
            "delta": self.delta,
            "floor": self.floor,
            "floor_bases": dict(self.floor_bases),
            "distinguishable_fraction": self.distinguishable_fraction,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "note": self.note,
        }


def distinguishable_fraction(left: Sequence[str], right: Sequence[str]) -> float:
    """Share of positions where the two arms actually produced different text."""
    if not left:
        return 0.0
    differing = sum(
        unicodedata.normalize("NFC", one) != unicodedata.normalize("NFC", other)
        for one, other in zip(left, right, strict=True)
    )
    return differing / len(left)


def _spread(scores: Sequence[float]) -> float | None:
    return max(scores) - min(scores) if len(scores) >= 2 else None  # noqa: PLR2004


def bootstrap_floor(
    base_segments: Sequence[float],
    adapter_segments: Sequence[float],
    *,
    delta: float,
    seed: int,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> float:
    """Paired resampling over the per-segment scores. It can only widen the floor.

    Returns the delta's own magnitude when the interval straddles zero, else zero.
    """
    paired = [adapter - base for base, adapter in zip(base_segments, adapter_segments, strict=True)]
    if len(paired) < 2:  # noqa: PLR2004
        return 0.0

    generator = random.Random(seed)  # noqa: S311 -- resampling, not a security decision
    size = len(paired)
    means = sorted(
        sum(paired[generator.randrange(size)] for _ in range(size)) / size for _ in range(resamples)
    )
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    low = means[int(tail * (resamples - 1))]
    high = means[int((1.0 - tail) * (resamples - 1))]
    return abs(delta) if low <= 0.0 <= high else 0.0


def _tally(
    base_segments: Sequence[float], adapter_segments: Sequence[float]
) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for base, adapter in zip(base_segments, adapter_segments, strict=True):
        if adapter > base:
            wins += 1
        elif adapter < base:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def compare_arms(
    base: ArmResult,
    adapter: ArmResult,
    *,
    confounded: bool = False,
    bootstrap_seed: int | None = None,
) -> Comparison:
    """Turn two arms into one verdict from the closed set.

    Raises:
        DecodingMismatchError: The arms decoded differently, so no delta is attributable.
    """
    if base.decoding_fingerprint != adapter.decoding_fingerprint:
        msg = (
            f"arm {base.name!r} decoded as {base.decoding_fingerprint[:12]} and arm "
            f"{adapter.name!r} as {adapter.decoding_fingerprint[:12]}; a delta between them "
            f"would be measuring the decoder"
        )
        raise DecodingMismatchError(msg)

    delta = adapter.corpus_chrf - base.corpus_chrf
    fraction = distinguishable_fraction(base.outputs, adapter.outputs)
    wins, losses, ties = _tally(base.segment_scores, adapter.segment_scores)

    bases = {FLOOR_CONVENTION: MIN_REPORTABLE_DELTA_CHRF}
    spreads = [
        value
        for value in (_spread(base.repeat_scores), _spread(adapter.repeat_scores))
        if value is not None
    ]
    if spreads:
        bases[FLOOR_REPEAT_SPREAD] = max(spreads)
    if bootstrap_seed is not None:
        bases[FLOOR_BOOTSTRAP] = bootstrap_floor(
            base.segment_scores, adapter.segment_scores, delta=delta, seed=bootstrap_seed
        )
    floor = max(bases.values())

    note: str | None = None
    if confounded:
        verdict, note = UNAVAILABLE, _CONFOUNDED_NOTE
    elif fraction < MIN_DISTINGUISHABLE_FRACTION:
        verdict, note = INDISTINGUISHABLE, _INDISTINGUISHABLE_NOTE
    elif abs(delta) <= floor:
        verdict = INDISTINGUISHABLE
    else:
        verdict = IMPROVED if delta > 0 else REGRESSED

    return Comparison(
        verdict=verdict,
        delta=delta,
        floor=floor,
        floor_bases=bases,
        distinguishable_fraction=fraction,
        wins=wins,
        losses=losses,
        ties=ties,
        note=note,
    )


@dataclass(frozen=True)
class EvaluationRun:
    """One complete measurement."""

    run_id: str
    created_at: str
    test_set: dict[str, Any]
    decoding: dict[str, Any]
    arms: dict[str, ArmResult]
    comparison: Comparison
    prompt_shape_fingerprint: str
    confounded: bool = False

    def as_record(self) -> dict[str, Any]:
        """The full record. Every key in `REQUIRED_RECORD_KEYS` is present by construction."""
        scored = self.test_set["n_scored"]
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "test_set": dict(self.test_set),
            "prompt_shape_fingerprint": self.prompt_shape_fingerprint,
            "decoding": dict(self.decoding),
            "arms": {name: arm.as_dict() for name, arm in self.arms.items()},
            "metric": metrics.metric_parameters(),
            "scores": {
                name: {"corpus_chrf": arm.corpus_chrf, "n": scored}
                for name, arm in self.arms.items()
            }
            | {"delta": self.comparison.delta, "n": scored},
            "comparison": self.comparison.as_dict(),
            "confounded": self.confounded,
        }

    def as_index_line(self) -> dict[str, Any]:
        """The compact summary appended to the index."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "verdict": self.comparison.verdict,
            "delta": self.comparison.delta,
            "n": self.test_set["n_scored"],
            "scores": {name: arm.corpus_chrf for name, arm in self.arms.items()},
            "test_set_sha256": self.test_set["sha256"],
            "limit": self.test_set["limit"],
            "prompt_shape_fingerprint": self.prompt_shape_fingerprint,
            "decoding_fingerprint": self.decoding["fingerprint"],
            "confounded": self.confounded,
        }


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def build_run_id(
    identity: Mapping[str, Any], *, clock: Callable[[], datetime] | None = None
) -> str:
    """Timestamp plus a digest of the run's identity."""
    now = (clock or _utc_now)()
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{_digest_of(dict(identity))[:12]}"


def write_run(run: EvaluationRun, out_dir: Path = DEFAULT_EVAL_DIR) -> Path:
    """Write one record and append exactly one index line. Nothing else is touched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{run.run_id}.json"
    target.write_text(
        json.dumps(run.as_record(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / INDEX_FILENAME).open("a", encoding="utf-8") as index:
        index.write(_canonical_json(run.as_index_line()) + "\n")
    return target


def read_run(path: Path) -> dict[str, Any]:
    """Read a record back, refusing anything incomplete rather than returning silent gaps.

    Raises:
        RecordError: The file is unreadable, a required key is missing, or the verdict is
            outside the closed set.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = f"{path}: unreadable run record ({error})"
        raise RecordError(msg) from error

    if not isinstance(record, dict):
        msg = f"{path}: run record is not a JSON object"
        raise RecordError(msg)

    missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        msg = f"{path}: run record is missing {', '.join(missing)}"
        raise RecordError(msg)

    verdict = record["comparison"].get("verdict")
    if verdict not in VERDICTS:
        msg = f"{path}: verdict {verdict!r} is outside {VERDICTS}"
        raise RecordError(msg)
    return record


@dataclass(frozen=True)
class RunComparison:
    """Whether two records may be compared, and why not."""

    comparable: bool
    reasons: tuple[str, ...]
    delta: float | None = None


_COMPARABILITY_KEYS = (
    ("test_set", "sha256"),
    ("test_set", "limit"),
    ("prompt_shape_fingerprint", None),
)


def compare_runs(first: Mapping[str, Any], second: Mapping[str, Any]) -> RunComparison:
    """Compare two records, or refuse. A refusal never carries a delta."""
    reasons: list[str] = []
    for outer, inner in _COMPARABILITY_KEYS:
        left = first[outer][inner] if inner else first[outer]
        right = second[outer][inner] if inner else second[outer]
        if left != right:
            name = f"{outer}.{inner}" if inner else outer
            reasons.append(f"{name}: {left!r} against {right!r}")

    if reasons:
        return RunComparison(comparable=False, reasons=tuple(reasons))
    return RunComparison(
        comparable=True,
        reasons=(),
        delta=second["scores"]["delta"] - first["scores"]["delta"],
    )


@dataclass(frozen=True)
class EvaluationConfig:
    """One evaluation run's inputs."""

    test_file: Path = DEFAULT_SPLIT_DIR / TEST_FILENAME
    adapter_path: str | None = None
    model_id: str | None = None
    limit: int | None = None
    repeats: int = DEFAULT_REPEATS
    max_tokens: int = DEFAULT_MAX_TOKENS
    out_dir: Path = DEFAULT_EVAL_DIR
    allow_provenance_mismatch: bool = False
    bootstrap_seed: int | None = None


@contextlib.contextmanager
def _without_env_adapter() -> Iterator[None]:
    """`Translator` falls back to the adapter env var, which would arm both sides."""
    saved = os.environ.pop(ADAPTER_ENV_VAR, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[ADAPTER_ENV_VAR] = saved


def _default_translator(
    model_id: str,
    *,
    adapter_path: str | None,
    allow_provenance_mismatch: bool,
) -> Translator:
    if adapter_path is None:
        with _without_env_adapter():
            return Translator(model_id)
    return Translator(
        model_id,
        adapter_path=adapter_path,
        allow_provenance_mismatch=allow_provenance_mismatch,
    )


def _to_stderr(message: str) -> None:
    print(message, file=sys.stderr)  # noqa: T201 -- stdout stays the human summary


def _run_arm(  # noqa: PLR0913 -- one parameter per arm input
    name: str,
    translator: Any,  # noqa: ANN401 -- anything with the translate method
    *,
    model_id: str,
    adapter_path: str | None,
    sources: Sequence[str],
    references: Sequence[str],
    decoding: DecodingSettings,
    scorer: Callable[[Sequence[str], Sequence[str]], metrics.ScoreSet],
) -> ArmResult:
    started = time.monotonic()
    repeat_scores: list[float] = []
    outputs: tuple[str, ...] = ()
    segments: tuple[float, ...] = ()

    for index in range(max(1, decoding.repeats)):
        produced = tuple(
            translator.translate(source, max_tokens=decoding.max_tokens) for source in sources
        )
        scored = scorer(produced, references)
        repeat_scores.append(scored.corpus)
        # The first pass is the scored one; later passes exist only to measure the spread.
        if index == 0:
            outputs, segments = produced, scored.segments

    return ArmResult(
        name=name,
        model_id=model_id,
        adapter_path=adapter_path,
        outputs=outputs,
        repeat_scores=tuple(repeat_scores),
        corpus_chrf=repeat_scores[0],
        segment_scores=segments,
        seconds=time.monotonic() - started,
        decoding_fingerprint=decoding.fingerprint(),
    )


def _check_provenance(adapter_path: str, model_id: str, *, allow_mismatch: bool) -> bool:
    """Whether the pairing is confounded. Raises unless the user has accepted it.

    Raises:
        ConfoundedComparisonError: The adapter was fitted to another base and no override is set.
    """
    provenance = read_provenance(adapter_path)
    if provenance is None or provenance.base_model_id == model_id:
        return False

    if not allow_mismatch:
        msg = (
            f"{adapter_path} was fitted to {provenance.base_model_id!r}, but the base arm loads "
            f"{model_id!r}. The arms would differ in base as well as adapter; pass "
            f"--allow-provenance-mismatch to run anyway, with no comparison verdict."
        )
        raise ConfoundedComparisonError(msg)
    return True


def evaluate_pair(
    config: EvaluationConfig,
    *,
    translator_factory: Callable[..., Any] | None = None,
    scorer: Callable[[Sequence[str], Sequence[str]], metrics.ScoreSet] | None = None,
    clock: Callable[[], datetime] | None = None,
    report: Callable[[str], None] = _to_stderr,
) -> EvaluationRun:
    """Score base and adapter over one split and compare them.

    Raises:
        EvaluationError: No adapter to compare against, or the split is unusable.
        SplitIntegrityError: The test split overlaps training data.
        ConfoundedComparisonError: The adapter belongs to another base and no override is set.
    """
    factory = translator_factory or _default_translator
    score = scorer or metrics.score_corpus

    model_id = resolve_model_id(config.model_id)
    adapter_path = resolve_adapter_path(config.adapter_path)
    if adapter_path is None:
        msg = (
            "no adapter to compare against; pass --adapter-path or set "
            f"${ADAPTER_ENV_VAR}. A single-arm score is not a comparison."
        )
        raise EvaluationError(msg)

    segments = load_split(config.test_file)
    check_split_isolation(segments, test_path=config.test_file)
    digest = split_digest(segments)

    scored = segments if config.limit is None else segments[: config.limit]
    if not scored:
        msg = f"--limit {config.limit} leaves nothing to score"
        raise EvaluationError(msg)

    confounded = _check_provenance(
        adapter_path, model_id, allow_mismatch=config.allow_provenance_mismatch
    )

    decoding = DecodingSettings(max_tokens=config.max_tokens, repeats=config.repeats)
    sources = [segment.source for segment in scored]
    references = [segment.target for segment in scored]
    report(f"Scoring {len(scored)} segment(s) across 2 arm(s), {decoding.repeats} pass(es) each.")

    # One arm at a time: each translator holds gigabytes, and the base is freed as _run_arm returns.
    arms = {
        BASE_ARM: _run_arm(
            BASE_ARM,
            factory(model_id, adapter_path=None, allow_provenance_mismatch=False),
            model_id=model_id,
            adapter_path=None,
            sources=sources,
            references=references,
            decoding=decoding,
            scorer=score,
        ),
        ADAPTER_ARM: _run_arm(
            ADAPTER_ARM,
            factory(
                model_id,
                adapter_path=adapter_path,
                allow_provenance_mismatch=config.allow_provenance_mismatch,
            ),
            model_id=model_id,
            adapter_path=adapter_path,
            sources=sources,
            references=references,
            decoding=decoding,
            scorer=score,
        ),
    }

    comparison = compare_arms(
        arms[BASE_ARM],
        arms[ADAPTER_ARM],
        confounded=confounded,
        bootstrap_seed=config.bootstrap_seed,
    )

    test_set = {
        "path": str(config.test_file),
        "sha256": digest,
        "n_lines": len(segments),
        "n_scored": len(scored),
        "limit": config.limit,
        "subset": config.limit is not None,
    }
    fingerprint = prompt_shape_fingerprint()
    now = (clock or _utc_now)()
    return EvaluationRun(
        run_id=build_run_id(
            {
                "model_id": model_id,
                "adapter_path": adapter_path,
                "test_set_sha256": digest,
                "limit": config.limit,
                "decoding_fingerprint": decoding.fingerprint(),
                "prompt_shape_fingerprint": fingerprint,
            },
            clock=lambda: now,
        ),
        created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        test_set=test_set,
        decoding={"settings": decoding.as_dict(), "fingerprint": decoding.fingerprint()},
        arms=arms,
        comparison=comparison,
        prompt_shape_fingerprint=fingerprint,
        confounded=confounded,
    )


def render_summary(run: EvaluationRun) -> str:
    """The human summary; every score line carries `n`."""
    scored = run.test_set["n_scored"]
    comparison = run.comparison
    lines = [
        f"run {run.run_id}",
        f"test set {run.test_set['path']} — {scored} of {run.test_set['n_lines']} row(s) scored",
        f"base    chrF {run.arms[BASE_ARM].corpus_chrf:6.2f}  (n={scored})",
        f"adapter chrF {run.arms[ADAPTER_ARM].corpus_chrf:6.2f}  (n={scored})",
        (
            f"delta   {comparison.delta:+6.2f}  (n={scored}, floor {comparison.floor:.2f} "
            f"from {', '.join(sorted(comparison.floor_bases))})"
        ),
        f"per segment: {comparison.wins} better, {comparison.losses} worse, {comparison.ties} tied",
        f"arms differed on {comparison.distinguishable_fraction:.0%} of segments",
        f"verdict: {comparison.verdict}",
    ]
    if comparison.note:
        lines.append(comparison.note)
    return "\n".join(lines)
