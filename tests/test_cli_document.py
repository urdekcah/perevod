# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Flag wiring, exit codes, and stream discipline for the document path."""

import tempfile
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

from click.testing import CliRunner

from perevod.cli import main
from perevod.document import UNTRANSLATED_OPEN, join_translations, split_document
from perevod.translator import Translator

RUSSIAN = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац.\n"
THINK_START = "<|channel>thought"
THINK_END = "<channel|>"


@dataclass(frozen=True)
class _Response:
    text: str
    finish_reason: str | None


class _Tokenizer:
    """Hands the chunk straight back, and advertises the delimiters the guard looks for."""

    think_start = THINK_START
    think_end = THINK_END

    def apply_chat_template(self, messages: list[dict[str, str]], **_: object) -> str:
        return messages[-1]["content"]


class TranslatorFactory:
    """Builds real translators over fake weights and counts how often they load."""

    def __init__(self, *, refuse: str = "", contaminate: str = "", truncate: str = "") -> None:
        self.loads: list[str] = []
        self._refuse = refuse
        self._contaminate = contaminate
        self._truncate = truncate

    def _load(self, model_id: str, **_: object) -> tuple[object, _Tokenizer]:
        self.loads.append(model_id)
        return object(), _Tokenizer()

    def _generate(
        self,
        _model: object,
        _tokenizer: object,
        **kwargs: Any,  # noqa: ANN401
    ) -> Iterator[_Response]:
        prompt = str(kwargs["prompt"])
        if self._refuse and self._refuse in prompt:
            msg = "backend refused this chunk"
            raise RuntimeError(msg)
        if self._contaminate and self._contaminate in prompt:
            yield _Response(f"{THINK_START} рассуждение {THINK_END}[ko]{prompt}", "stop")
        elif self._truncate and self._truncate in prompt:
            yield _Response("[ko]обрыв", "length")
        else:
            yield _Response(f"[ko]{prompt}", "stop")

    def __call__(self, model_id: str | None = None, **kwargs: object) -> Translator:
        return Translator(model_id, loader=self._load, generator=self._generate, **kwargs)  # type: ignore[arg-type]


class CliDocumentTest(unittest.TestCase):
    """The document surface, driven end to end without a model."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.runner = CliRunner()

    def _write(self, name: str, text: str = RUSSIAN) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def _run(self, args: list[str], factory: TranslatorFactory) -> Any:  # noqa: ANN401
        with mock.patch("perevod.cli.Translator", factory):
            return self.runner.invoke(main, ["translate", *args])

    def test_a_batch_loads_the_model_once_and_writes_every_file(self) -> None:
        sources = [self._write(f"doc{n}.txt") for n in range(3)]
        out = self.root / "out"
        factory = TranslatorFactory()

        result = self._run(
            [arg for path in sources for arg in ("--input", str(path))]
            + ["--output-dir", str(out)],
            factory,
        )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(factory.loads), 1)
        self.assertEqual(
            sorted(p.name for p in out.iterdir()), ["doc0.ko.txt", "doc1.ko.txt", "doc2.ko.txt"]
        )

    def test_one_failing_file_leaves_the_others_translated_and_exits_one(self) -> None:
        sources = [self._write("doc0.txt"), self._write("doc1.txt", "Только этот абзац падает.")]
        out = self.root / "out"
        factory = TranslatorFactory(refuse="падает")

        result = self._run(
            [arg for path in sources for arg in ("--input", str(path))]
            + ["--output-dir", str(out)],
            factory,
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("[ko]", (out / "doc0.ko.txt").read_text(encoding="utf-8"))
        self.assertIn(
            UNTRANSLATED_OPEN.format(index=0), (out / "doc1.ko.txt").read_text(encoding="utf-8")
        )

    def test_standard_output_carries_the_document_and_nothing_else(self) -> None:
        source = self._write("doc.txt")
        factory = TranslatorFactory()

        result = self._run(["--input", str(source), "--progress"], factory)

        chunks = split_document(RUSSIAN, budget_chars=2048)
        expected = join_translations(chunks, [f"[ko]{c.text}" for c in chunks])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, expected)
        self.assertIn("chunk 1/3", result.stderr)

    def test_conflicting_source_options_are_usage_errors(self) -> None:
        source = self._write("doc.txt")
        factory = TranslatorFactory()
        cases = [
            ["привет", "--input", str(source)],
            ["--input", "-"],
            ["--input", str(source), "--output", "a.txt", "--output-dir", str(self.root)],
            ["--input", str(source), "--input", str(source), "--output", "a.txt"],
            ["--input", str(source), "--input", str(source)],
        ]
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(self._run(args, factory).exit_code, 2)

    def test_an_output_that_would_destroy_data_is_refused(self) -> None:
        source = self._write("doc.txt")
        existing = self._write("out.txt", "уже здесь")
        factory = TranslatorFactory()

        over_source = self._run(["--input", str(source), "--output", str(source)], factory)
        over_existing = self._run(["--input", str(source), "--output", str(existing)], factory)

        self.assertEqual(over_source.exit_code, 2)
        self.assertEqual(over_existing.exit_code, 2)
        self.assertEqual(existing.read_text(encoding="utf-8"), "уже здесь")
        self.assertEqual(factory.loads, [])

    def test_a_reasoning_trace_is_reported_and_kept_out_of_the_written_file(self) -> None:
        source = self._write("doc.txt")
        out = self.root / "doc.ko.txt"
        factory = TranslatorFactory(contaminate="Второй")

        result = self._run(["--input", str(source), "--output", str(out)], factory)
        written = out.read_text(encoding="utf-8")

        self.assertEqual(result.exit_code, 1)
        self.assertNotIn(THINK_START, written)
        self.assertIn(UNTRANSLATED_OPEN.format(index=1), written)
        self.assertIn("1 chunk(s) untranslated", result.stderr)

    def test_a_chunk_that_ran_out_of_budget_is_reported_not_written(self) -> None:
        source = self._write("doc.txt")
        out = self.root / "doc.ko.txt"
        factory = TranslatorFactory(truncate="Второй")

        result = self._run(["--input", str(source), "--output", str(out)], factory)
        written = out.read_text(encoding="utf-8")

        self.assertEqual(result.exit_code, 1)
        self.assertNotIn("[ko]обрыв", written)
        self.assertIn(UNTRANSLATED_OPEN.format(index=1), written)

    def test_the_bare_string_path_is_unchanged(self) -> None:
        factory = TranslatorFactory()
        result = self._run(["Привет"], factory)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.strip(), "[ko]Привет")


if __name__ == "__main__":
    unittest.main()
