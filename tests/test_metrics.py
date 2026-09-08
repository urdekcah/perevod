# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Fixed vectors for chrF, each with the arithmetic a reader can check without running it."""

import unicodedata
import unittest

from perevod import metrics

PLACES = 10

# Expected values, derived by hand. Per order n in 1..6: p_n = matches / hypothesis n-grams,
# r_n = matches / reference n-grams (0 when that side has no n-grams); chrP and chrR are the
# means over the six orders; with BETA = 2, chrF = 5 * chrP * chrR / (4 * chrP + chrR) * 100.
#
# "가나다라마바" against itself: six identical characters, every order matches in full -> 100.00.
# "aaa"/"bbb": nothing shared, orders 4-6 empty on both sides -> chrP = chrR = 0 -> 0.00.
# "abc"/"abd": n=1 2 matches of 3 and 3; n=2 1 of 2 and 2; n=3 0 of 1 and 1; n=4..6 empty.
#   chrP = chrR = (2/3 + 1/2)/6 = 7/36, and with P = R the formula collapses to P -> 19.4444.
# "ab"/"abc": n=1 -> 1, 2/3; n=2 -> 1, 1/2; n=3 hypothesis bag empty -> 0, 0; n=4..6 -> 0, 0.
#   chrP = 1/3, chrR = 7/36 -> (35/108)/(55/36) = 7/33 -> 21.2121.
# "abc"/"ab": the same numbers with P and R swapped, and BETA weights recall, so it scores
#   higher: chrP = 7/36, chrR = 1/3 -> (35/108)/(40/36) = 7/24 -> 29.1666.
# ""/"abc" and "abc"/"": one side has no n-grams at any order -> both means 0 -> 0.00.
VECTORS = (
    ("exact match", "가나다라마바", "가나다라마바", 100.0),
    ("total mismatch", "aaa", "bbb", 0.0),
    ("partial match", "abc", "abd", 100 * 7 / 36),
    ("hypothesis shorter than the order", "ab", "abc", 100 * 7 / 33),
    ("reference shorter than the order", "abc", "ab", 100 * 7 / 24),
    ("empty hypothesis", "", "abc", 0.0),
    ("empty reference", "abc", "", 0.0),
    ("whitespace-only hypothesis", "   ", "abc", 0.0),
)

# Micro-aggregation pools counts before dividing. "ab"/"ab" scores 100/3 = 33.33 (orders 3-6
# empty on both sides), "abcdef"/"abcdef" scores 100.00; pooled, every order has matches equal
# to both totals, so the corpus score is 100.00 — not the 66.67 mean of the two segments.
CORPUS_HYPOTHESES = ("ab", "abcdef")
CORPUS_REFERENCES = ("ab", "abcdef")
CORPUS_EXPECTED = 100.0
SEGMENT_EXPECTED = (100 / 3, 100.0)


class ChrfVectorTest(unittest.TestCase):
    """The arithmetic, provable on a bare interpreter with nothing installed."""

    def test_every_fixed_vector_scores_its_derived_value(self) -> None:
        for name, hypothesis, reference, expected in VECTORS:
            with self.subTest(name):
                self.assertAlmostEqual(
                    metrics.sentence_chrf(hypothesis, reference), expected, places=PLACES
                )

    def test_the_corpus_score_is_not_the_mean_of_its_segments(self) -> None:
        scored = metrics.score_corpus(CORPUS_HYPOTHESES, CORPUS_REFERENCES)

        self.assertAlmostEqual(scored.corpus, CORPUS_EXPECTED, places=PLACES)
        for actual, expected in zip(scored.segments, SEGMENT_EXPECTED, strict=True):
            self.assertAlmostEqual(actual, expected, places=PLACES)
        self.assertNotAlmostEqual(
            scored.corpus, sum(scored.segments) / len(scored.segments), places=PLACES
        )

    def test_mismatched_lengths_are_refused_rather_than_zipped_short(self) -> None:
        with self.assertRaises(ValueError):
            metrics.score_corpus(("a", "b"), ("a",))


class NormalizationTest(unittest.TestCase):
    """Hangul reaching us decomposed must not read as a different translation."""

    TEXT = "한국어 번역"
    COMPOSED = unicodedata.normalize("NFC", TEXT)
    DECOMPOSED = unicodedata.normalize("NFD", TEXT)

    def test_the_two_normal_forms_score_identically_in_both_argument_orders(self) -> None:
        baseline = metrics.sentence_chrf(self.COMPOSED, self.COMPOSED)

        for hypothesis, reference in (
            (self.DECOMPOSED, self.DECOMPOSED),
            (self.DECOMPOSED, self.COMPOSED),
            (self.COMPOSED, self.DECOMPOSED),
        ):
            with self.subTest(hypothesis=hypothesis):
                self.assertAlmostEqual(
                    metrics.sentence_chrf(hypothesis, reference), baseline, places=PLACES
                )

    def test_the_two_forms_really_are_different_strings(self) -> None:
        self.assertNotEqual(self.COMPOSED, self.DECOMPOSED)

    def test_both_sides_pass_through_the_one_normalization_path(self) -> None:
        self.assertEqual(metrics.normalize(self.DECOMPOSED), metrics.normalize(self.COMPOSED))

    def test_whitespace_never_reaches_the_ngrams(self) -> None:
        self.assertEqual(metrics.normalize("한 국 어"), metrics.normalize("한국어"))


class ParameterBlockTest(unittest.TestCase):
    """A score without its parameters cannot be read, so the block travels with it."""

    def test_the_block_names_every_pinned_parameter(self) -> None:
        parameters = metrics.metric_parameters()

        self.assertEqual(parameters["char_order"], metrics.CHAR_ORDER)
        self.assertEqual(parameters["beta"], metrics.BETA)
        self.assertEqual(parameters["normalization"], metrics.UNICODE_NORMALIZATION)
        self.assertEqual(parameters["whitespace"], metrics.WHITESPACE_RULE)
        self.assertEqual(parameters["scale"], metrics.SCORE_SCALE)
        self.assertEqual(parameters["corpus_aggregation"], metrics.CORPUS_AGGREGATION)

    def test_this_is_chrf_and_not_chrf_plus_plus(self) -> None:
        self.assertEqual(metrics.WORD_ORDER, 0)
        self.assertEqual(metrics.metric_parameters()["word_order"], 0)


if __name__ == "__main__":
    unittest.main()
