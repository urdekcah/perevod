# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Self-contained chrF over character n-grams, standard library only.

An empty n-gram bag scores zero for its own side rather than dividing, and corpus scores
pool counts before averaging. Comparable only against other scores from here.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

CHAR_ORDER = 6

# chrF++ sets this to 2, reintroducing the tokenization sensitivity Korean spacing makes acute.
WORD_ORDER = 0

BETA = 2.0

UNICODE_NORMALIZATION: Literal["NFC"] = "NFC"
WHITESPACE_RULE = "stripped-before-ngrams"
SCORE_SCALE = "0-100"
CORPUS_AGGREGATION = "micro"

_ORDERS = tuple(range(1, CHAR_ORDER + 1))
_SCALE = 100.0


def normalize(text: str) -> str:
    """The single normalization path. Decomposed Hangul from macOS would score near zero."""
    return "".join(unicodedata.normalize(UNICODE_NORMALIZATION, text).split())


@dataclass(frozen=True)
class NgramCounts:
    """Matches and bag sizes, one entry per order."""

    matches: tuple[int, ...]
    hypothesis: tuple[int, ...]
    reference: tuple[int, ...]

    def __add__(self, other: NgramCounts) -> NgramCounts:
        """Pool two segments."""
        return NgramCounts(
            matches=tuple(a + b for a, b in zip(self.matches, other.matches, strict=True)),
            hypothesis=tuple(a + b for a, b in zip(self.hypothesis, other.hypothesis, strict=True)),
            reference=tuple(a + b for a, b in zip(self.reference, other.reference, strict=True)),
        )


@dataclass(frozen=True)
class ScoreSet:
    """A corpus score and the per-segment scores behind it — different numbers."""

    corpus: float
    segments: tuple[float, ...]


def _zero_counts() -> NgramCounts:
    empty = (0,) * len(_ORDERS)
    return NgramCounts(matches=empty, hypothesis=empty, reference=empty)


def _bag(text: str, order: int) -> Counter[str]:
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def count_ngrams(hypothesis: str, reference: str) -> NgramCounts:
    """Per-order matches and bag sizes for one pair."""
    left = normalize(hypothesis)
    right = normalize(reference)

    matches: list[int] = []
    hypothesis_totals: list[int] = []
    reference_totals: list[int] = []
    for order in _ORDERS:
        hypothesis_bag = _bag(left, order)
        reference_bag = _bag(right, order)
        matches.append(sum((hypothesis_bag & reference_bag).values()))
        hypothesis_totals.append(sum(hypothesis_bag.values()))
        reference_totals.append(sum(reference_bag.values()))

    return NgramCounts(
        matches=tuple(matches),
        hypothesis=tuple(hypothesis_totals),
        reference=tuple(reference_totals),
    )


def _mean_ratio(numerators: tuple[int, ...], denominators: tuple[int, ...]) -> float:
    ratios = [
        matched / total if total else 0.0
        for matched, total in zip(numerators, denominators, strict=True)
    ]
    return sum(ratios) / len(ratios)


def score(counts: NgramCounts) -> float:
    """Score pooled counts, 0-100."""
    precision = _mean_ratio(counts.matches, counts.hypothesis)
    recall = _mean_ratio(counts.matches, counts.reference)
    if precision == 0.0 and recall == 0.0:
        return 0.0

    weight = BETA**2
    return _SCALE * (1 + weight) * precision * recall / (weight * precision + recall)


def sentence_chrf(hypothesis: str, reference: str) -> float:
    """Score one pair, 0-100."""
    return score(count_ngrams(hypothesis, reference))


def score_corpus(hypotheses: Sequence[str], references: Sequence[str]) -> ScoreSet:
    """Score a whole split.

    Raises:
        ValueError: The two sides are not the same length, so the pairing is unknown.
    """
    if len(hypotheses) != len(references):
        msg = f"{len(hypotheses)} hypotheses against {len(references)} references"
        raise ValueError(msg)

    pooled = _zero_counts()
    segments: list[float] = []
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        counts = count_ngrams(hypothesis, reference)
        pooled += counts
        segments.append(score(counts))

    return ScoreSet(corpus=score(pooled), segments=tuple(segments))


def metric_parameters() -> dict[str, object]:
    """The block that travels beside every score."""
    return {
        "name": "chrF",
        "implementation": "perevod.metrics",
        "char_order": CHAR_ORDER,
        "word_order": WORD_ORDER,
        "beta": BETA,
        "normalization": UNICODE_NORMALIZATION,
        "whitespace": WHITESPACE_RULE,
        "scale": SCORE_SCALE,
        "corpus_aggregation": CORPUS_AGGREGATION,
    }
