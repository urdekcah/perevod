# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""What the trainer will read, and what it must be stopped from reading."""

import json
import tempfile
import unittest
from pathlib import Path

from perevod.dataset import (
    CHAT_SCHEMA,
    COMPLETIONS_SCHEMA,
    TEST_FILENAME,
    TEXT_SCHEMA,
    TRAIN_FILENAME,
    VALID_FILENAME,
    RecordShapeError,
    SchemaError,
    SplitLayoutError,
    build_training_record,
    validate_schema,
    validate_split_dir,
    validate_training_file,
)

PAIRS = (("Здравствуйте", "안녕하세요"), ("Спасибо", "감사합니다"))


def _chat_lines(pairs: tuple[tuple[str, str], ...] = PAIRS) -> str:
    rows = (build_training_record(source, target) for source, target in pairs)
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


class TempTreeTestCase(unittest.TestCase):
    """A scratch directory for one test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path


class SchemaRecognitionTests(TempTreeTestCase):
    """The shapes the trainer reads, and the ones a person writes by mistake."""

    def test_accepts_each_of_the_three_documented_schemas(self) -> None:
        for schema, line in (
            (CHAT_SCHEMA, _chat_lines()),
            (COMPLETIONS_SCHEMA, '{"prompt": "Привет", "completion": "안녕"}\n'),
            (TEXT_SCHEMA, '{"text": "Привет"}\n'),
        ):
            with self.subTest(schema=schema):
                self.assertEqual(validate_schema(self.write(f"{schema}.jsonl", line)), schema)

    def test_every_rejectable_shape_is_named_by_file_and_editor_line(self) -> None:
        cases = {
            "broken.jsonl": ('{"text": "ok"}\n{"text": \n', 2),
            "scalar.jsonl": ('{"text": "ok"}\n"just a string"\n', 2),
            "unknown.jsonl": ('{"input": "Привет", "output": "안녕"}\n', 1),
            "messages.jsonl": ('{"messages": []}\n', 1),
            "keys.jsonl": ('{"messages": [{"role": "user"}]}\n', 1),
            "role.jsonl": ('{"messages": [{"role": "narrator", "content": "x"}]}\n', 1),
        }
        for name, (text, line) in cases.items():
            with self.subTest(case=name):
                path = self.write(name, text)
                with self.assertRaises(SchemaError) as caught:
                    validate_schema(path)
                self.assertIn(f"{path}:{line}:", str(caught.exception))

    def test_an_empty_file_is_an_error_but_a_trailing_newline_is_not(self) -> None:
        with self.assertRaises(SchemaError):
            validate_schema(self.write("empty.jsonl", ""))

        padded = self.write("padded.jsonl", _chat_lines() + "\n")
        self.assertEqual(validate_training_file(padded), 2)

    def test_valid_data_in_the_wrong_schema_says_how_to_convert_it(self) -> None:
        path = self.write(TRAIN_FILENAME, '{"prompt": "Привет", "completion": "안녕"}\n')

        with self.assertRaises(SchemaError) as caught:
            validate_training_file(path)

        self.assertIn("prepare-data", str(caught.exception))

    def test_a_row_from_a_different_prompt_shape_is_refused(self) -> None:
        row = build_training_record(*PAIRS[0])
        row["messages"][0]["content"] = "Переведи как хочешь."
        path = self.write(TRAIN_FILENAME, json.dumps(row, ensure_ascii=False) + "\n")

        with self.assertRaises(RecordShapeError) as caught:
            validate_training_file(path)

        self.assertIn(f"{path}:1:", str(caught.exception))


class SplitDirectoryTests(TempTreeTestCase):
    """One split is required, two are optional, and the difference has to be visible."""

    def test_a_missing_train_split_names_the_path_and_the_documentation(self) -> None:
        with self.assertRaises(SplitLayoutError) as caught:
            validate_split_dir(self.root)

        message = str(caught.exception)
        self.assertIn(str(self.root / TRAIN_FILENAME), message)
        self.assertIn("data/README.md", message)

    def test_a_missing_validation_split_is_a_notice_not_an_error(self) -> None:
        self.write(TRAIN_FILENAME, _chat_lines())

        report = validate_split_dir(self.root)

        self.assertEqual(dict(report.rows), {TRAIN_FILENAME: 2})
        self.assertIn("validation loss", " ".join(report.notices))

    def test_optional_splits_are_counted_when_present(self) -> None:
        self.write(TRAIN_FILENAME, _chat_lines())
        self.write(VALID_FILENAME, _chat_lines(PAIRS[:1]))
        self.write(TEST_FILENAME, _chat_lines(PAIRS[:1]))

        report = validate_split_dir(self.root)

        self.assertEqual(
            dict(report.rows), {TRAIN_FILENAME: 2, VALID_FILENAME: 1, TEST_FILENAME: 1}
        )
        self.assertEqual(report.notices, ())

    def test_an_optional_split_is_validated_as_strictly_as_the_required_one(self) -> None:
        self.write(TRAIN_FILENAME, _chat_lines())
        self.write(TEST_FILENAME, '{"text": "Привет"}\n')

        with self.assertRaises(SchemaError):
            validate_split_dir(self.root)
