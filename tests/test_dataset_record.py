"""The record shape and its inverse, which training and inference both bind to."""

import unittest

from perevod.dataset import (
    ASSISTANT_ROLE,
    CONTENT_KEY,
    MESSAGES_KEY,
    ROLE_KEY,
    SHAPE_SENTINEL,
    RecordShapeError,
    build_training_record,
    parse_training_record,
    prompt_shape_fingerprint,
)
from perevod.translator import build_messages


class BuildTrainingRecordTests(unittest.TestCase):
    def test_prefix_is_the_shipped_prompt_with_the_answer_appended(self):
        record = build_training_record("Привет", "안녕하세요")

        self.assertEqual(record[MESSAGES_KEY][:-1], build_messages("Привет"))
        self.assertEqual(
            record[MESSAGES_KEY][-1], {ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: "안녕하세요"}
        )

    def test_a_source_holding_the_sentinel_is_refused_rather_than_mis_parsed(self):
        with self.assertRaises(RecordShapeError) as caught:
            build_training_record(f"до {SHAPE_SENTINEL} после", "…")

        self.assertIn(SHAPE_SENTINEL, str(caught.exception))


class ParseTrainingRecordTests(unittest.TestCase):
    def test_round_trips_text_that_looks_like_the_scaffolding(self):
        for source, target in (
            ("Мир", "세계"),
            ("«Цитата» — тире", '"인용" — 대시'),
            (build_messages("x")[0][CONTENT_KEY], "약간의 한국어"),
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    parse_training_record(build_training_record(source, target)),
                    (source, target),
                )

    def test_a_diverged_role_sequence_is_named_not_guessed_at(self):
        record = build_training_record("Мир", "세계")
        record[MESSAGES_KEY][-1][ROLE_KEY] = "reviewer"

        with self.assertRaises(RecordShapeError) as caught:
            parse_training_record(record)

        self.assertIn(ASSISTANT_ROLE, str(caught.exception))

    def test_altered_constant_scaffolding_is_rejected(self):
        record = build_training_record("Мир", "세계")
        record[MESSAGES_KEY][0][CONTENT_KEY] += " Also rhyme."

        with self.assertRaises(RecordShapeError):
            parse_training_record(record)

    def test_a_dropped_message_is_rejected(self):
        record = build_training_record("Мир", "세계")
        del record[MESSAGES_KEY][0]

        with self.assertRaises(RecordShapeError):
            parse_training_record(record)


class FingerprintTests(unittest.TestCase):
    def test_is_a_stable_digest_of_the_shipped_shape(self):
        self.assertEqual(prompt_shape_fingerprint(), prompt_shape_fingerprint())
        self.assertEqual(len(prompt_shape_fingerprint()), 64)


if __name__ == "__main__":
    unittest.main()
