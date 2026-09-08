# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""The partition invariant, the budget postcondition, and Russian sentence boundaries."""

import ast
import pathlib
import re
import unittest

from perevod.document import (
    UNTRANSLATED_OPEN,
    Chunk,
    budget_chars_for,
    join_translations,
    split_document,
    split_sentences,
)

DOCUMENT_PATH = pathlib.Path(__file__).resolve().parent.parent / "perevod" / "document.py"

BUDGETS = (1, 2, 7, 64, 2048)

BLANK_LINE = re.compile(r"\n\s*\n")

HARD_SPLIT_BUDGET = 100

LONG_PARAGRAPH = " ".join(
    f"Это предложение номер {n} в очень длинном абзаце." for n in range(1, 60)
)
LONG_SENTENCE = "и".join(f"слово{n}" for n in range(1, 200)) + " и так далее без точек"
LONG_TOKEN = "ж" * 900
MIXED = (
    "А. С. Пушкин родился 26.05.1799. Он крикнул: «Стой!» Все остановились. "
    "См. рис. 3 на стр. 12. Что это?! Он задумался… Потом ответил."
)

CORPUS = (
    "",
    "   ",
    "\n\n\n",
    "  \n \n  ",
    "Короткий абзац без перевода строки",
    "Короткий абзац с переводом строки\n",
    "Первый абзац.\n\nВторой абзац.\n\nТретий абзац.\n",
    "Первый абзац.\n  \n\n  \nВторой абзац.   \n\n\nТретий абзац.\n",
    "\n\n  Абзац после пустых строк.\n",
    LONG_PARAGRAPH,
    LONG_SENTENCE,
    LONG_TOKEN,
    MIXED,
    f"Текст рядом с маркером.\n\n{UNTRANSLATED_OPEN.format(index=0)}\n\nЕщё текст.",
)

SEGMENTATION_CASES = (
    ("Это первое предложение. Это второе.", 2),
    ("А. С. Пушкин родился в Москве.", 1),
    ("Роман написал А. С. Пушкин. Его читают до сих пор.", 2),
    ("Мы взяли хлеб, молоко и т. д., а потом пошли домой.", 1),
    ("Мы взяли хлеб, молоко и т. д. Потом пошли домой.", 2),
    ("Встреча состоится 01.01.2026. Не опаздывайте.", 2),
    ("Цена — 3,14 руб. за килограмм.", 1),
    ("Он крикнул: «Стой!» Все остановились.", 2),
    ("Что это?! Я не понимаю.", 2),
    ("Он задумался… Потом ответил.", 2),
    ("Он задумался... Потом ответил.", 2),
    ("См. рис. 3 на стр. 12.", 1),
    ("В 1999 г. он уехал в Москву.", 1),
    ("Текст без завершающей точки", 1),
    ("Первое.Второе", 1),
    ("", 0),
)


class PartitionInvariantTest(unittest.TestCase):
    """What must hold for every input at every budget."""

    def test_chunks_reproduce_the_source_exactly(self) -> None:
        for source in CORPUS:
            for budget in BUDGETS:
                with self.subTest(budget=budget, source=source[:24]):
                    chunks = split_document(source, budget_chars=budget)
                    rebuilt = "".join(c.lead + c.text + c.tail for c in chunks)
                    self.assertEqual(rebuilt, source)

    def test_only_the_first_chunk_owns_a_lead(self) -> None:
        for source in CORPUS:
            for budget in BUDGETS:
                with self.subTest(budget=budget, source=source[:24]):
                    chunks = split_document(source, budget_chars=budget)
                    self.assertTrue(all(c.lead == "" for c in chunks[1:]))

    def test_identity_translation_round_trips(self) -> None:
        for source in CORPUS:
            for budget in BUDGETS:
                with self.subTest(budget=budget, source=source[:24]):
                    chunks = split_document(source, budget_chars=budget)
                    self.assertEqual(join_translations(chunks, [c.text for c in chunks]), source)

    def test_every_chunk_fits_the_budget(self) -> None:
        for source in CORPUS:
            for budget in BUDGETS:
                with self.subTest(budget=budget, source=source[:24]):
                    chunks = split_document(source, budget_chars=budget)
                    self.assertTrue(all(len(c.text) <= budget for c in chunks))

    def test_no_chunk_carries_a_blank_line(self) -> None:
        for source in CORPUS:
            for budget in BUDGETS:
                with self.subTest(budget=budget, source=source[:24]):
                    for chunk in split_document(source, budget_chars=budget):
                        self.assertIsNone(BLANK_LINE.search(chunk.text))

    def test_an_unbroken_token_is_hard_split_at_the_budget(self) -> None:
        chunks = split_document(LONG_TOKEN, budget_chars=HARD_SPLIT_BUDGET)
        self.assertEqual(len(chunks), len(LONG_TOKEN) // HARD_SPLIT_BUDGET)
        self.assertTrue(all(len(c.text) == HARD_SPLIT_BUDGET for c in chunks))

    def test_whitespace_only_source_yields_one_empty_chunk(self) -> None:
        chunks = split_document("  \n \n  ", budget_chars=64)
        self.assertEqual(chunks, [Chunk(index=0, lead="", text="", tail="  \n \n  ")])

    def test_a_budget_below_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            split_document("текст", budget_chars=0)
        with self.assertRaises(ValueError):
            budget_chars_for(0)

    def test_assembly_strips_stray_whitespace_from_the_model(self) -> None:
        source = "Первый абзац.\n\nВторой абзац."
        chunks = split_document(source, budget_chars=64)
        noisy = [f"\n  перевод {c.index}  \n" for c in chunks]
        self.assertEqual(join_translations(chunks, noisy), "перевод 0\n\nперевод 1")


class SentenceSegmentationTest(unittest.TestCase):
    """The 16 pinned Russian cases, plus the invariant that makes the heuristic safe."""

    def test_the_pinned_cases_split_as_specified(self) -> None:
        for text, expected in SEGMENTATION_CASES:
            with self.subTest(text=text):
                self.assertEqual(len(split_sentences(text)), expected)

    def test_sentences_reproduce_the_paragraph(self) -> None:
        for source in CORPUS:
            for paragraph in source.split("\n\n"):
                with self.subTest(paragraph=paragraph[:24]):
                    self.assertEqual("".join(split_sentences(paragraph)), paragraph)


class TextLayerPurityTest(unittest.TestCase):
    """The seam that lets eleven core criteria run with nothing installed."""

    def setUp(self) -> None:
        self.source = DOCUMENT_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_imports_nothing_outside_the_standard_library(self) -> None:
        roots: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertEqual(roots & {"mlx", "mlx_lm", "transformers", "click", "perevod"}, set())

    def test_performs_no_input_or_output(self) -> None:
        for banned in ("open(", "print(", "sys.stdout", "sys.stderr", "write_text"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, self.source)

    def test_carries_no_prompt_text(self) -> None:
        self.assertNotIn("Translate the following", self.source)


if __name__ == "__main__":
    unittest.main()
