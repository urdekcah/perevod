# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""The frozen comparison set and the runner that walks it."""

import unittest

from perevod.probes import PROBE_SENTENCES, run_probe_set

MIN_SENTENCES = 6
MAX_SENTENCES = 10
MAX_CHARS = 200


class FakeTranslator:
    """Echoes a prefix so output order is checkable."""

    def translate(self, source_text: str) -> str:
        return f"<{source_text[:4]}>"


class ProbeSetTest(unittest.TestCase):
    """Properties two comparisons depend on to stay comparable."""

    def test_the_set_is_immutable_and_within_its_documented_bounds(self) -> None:
        self.assertIsInstance(PROBE_SENTENCES, tuple)
        self.assertGreaterEqual(len(PROBE_SENTENCES), MIN_SENTENCES)
        self.assertLessEqual(len(PROBE_SENTENCES), MAX_SENTENCES)
        for sentence in PROBE_SENTENCES:
            self.assertLessEqual(len(sentence), MAX_CHARS, sentence)

    def test_no_sentence_repeats(self) -> None:
        self.assertEqual(len(set(PROBE_SENTENCES)), len(PROBE_SENTENCES))

    def test_the_runner_preserves_input_order(self) -> None:
        fake = FakeTranslator()

        self.assertEqual(
            run_probe_set(fake),
            tuple(fake.translate(sentence) for sentence in PROBE_SENTENCES),
        )


if __name__ == "__main__":
    unittest.main()
