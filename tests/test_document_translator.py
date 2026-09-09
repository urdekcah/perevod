# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Chunk independence, the failure policy, and progress — all against fakes, no model."""

import os
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from unittest import mock

from perevod.document import (
    UNTRANSLATED_CLOSE,
    UNTRANSLATED_OPEN,
    DocumentResult,
    DocumentTranslator,
    Progress,
    join_translations,
    split_document,
)
from perevod.translator import Translator

BUDGET = 40
TEN_PARAGRAPHS = "\n\n".join(f"Абзац номер {n} для проверки." for n in range(10))

THREE_PARAGRAPHS = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац.\n"
BAD_CHUNK = "Второй абзац."
THINK_START = "<|channel>thought"
THINK_END = "<channel|>"
TRACE = f"{THINK_START} рассуждение {THINK_END}"


class Recorder:
    """A translation function that remembers what it was handed."""

    def __init__(self, *, fail_on: set[int] | None = None, fail_all: bool = False) -> None:
        self.calls: list[str] = []
        self._fail_on = fail_on or set()
        self._fail_all = fail_all

    def __call__(self, text: str) -> str:
        position = len(self.calls)
        self.calls.append(text)
        if self._fail_all or position in self._fail_on:
            msg = f"backend refused chunk {position}"
            raise RuntimeError(msg)
        return f"[ko]{text}"


class ChunkIndependenceTest(unittest.TestCase):
    """Every chunk reaches the model alone, exactly as the splitter produced it."""

    def test_each_non_empty_chunk_is_translated_once_and_verbatim(self) -> None:
        source = "  \n\nПервый абзац.\n\nВторой абзац.\n\n"
        chunks = split_document(source, budget_chars=BUDGET)
        recorder = Recorder()

        result = DocumentTranslator(recorder, budget_chars=BUDGET).translate_document(source)

        self.assertEqual(recorder.calls, [c.text for c in chunks if c.text])
        self.assertEqual(result.text, join_translations(chunks, [f"[ko]{c.text}" for c in chunks]))

    def test_an_empty_chunk_never_reaches_the_model(self) -> None:
        recorder = Recorder()
        result = DocumentTranslator(recorder, budget_chars=BUDGET).translate_document(" \n \n ")
        self.assertEqual(recorder.calls, [])
        self.assertEqual(result.text, " \n \n ")


class FailurePolicyTest(unittest.TestCase):
    """A bad chunk costs one chunk, never the run and never the completed work."""

    def test_a_failed_chunk_becomes_a_marker_holding_its_source(self) -> None:
        chunks = split_document(TEN_PARAGRAPHS, budget_chars=BUDGET)
        recorder = Recorder(fail_on={3})

        result = DocumentTranslator(recorder, budget_chars=BUDGET).translate_document(
            TEN_PARAGRAPHS
        )

        self.assertEqual(result.chunks_failed, (3,))
        self.assertFalse(result.aborted)
        self.assertEqual(len(recorder.calls), len(chunks))
        self.assertIn(UNTRANSLATED_OPEN.format(index=3), result.text)
        self.assertIn(UNTRANSLATED_CLOSE.format(index=3), result.text)
        self.assertIn(chunks[3].text, result.text)
        self.assertIn("[ko]" + chunks[9].text, result.text)

    def test_a_systemic_failure_stops_after_the_circuit_breaker(self) -> None:
        recorder = Recorder(fail_all=True)

        result = DocumentTranslator(recorder, budget_chars=BUDGET).translate_document(
            TEN_PARAGRAPHS
        )

        self.assertTrue(result.aborted)
        self.assertEqual(len(recorder.calls), 3)
        self.assertEqual(len(result.chunks_failed), result.chunks_total)

    def test_a_retry_budget_is_spent_before_the_chunk_is_given_up(self) -> None:
        recorder = Recorder(fail_all=True)
        translator = DocumentTranslator(recorder, budget_chars=BUDGET, retries=2)

        translator.translate_document("Один абзац.")

        self.assertEqual(len(recorder.calls), 3)

    def test_interruption_and_memory_exhaustion_are_not_chunk_failures(self) -> None:
        for failure in (KeyboardInterrupt, MemoryError):
            with self.subTest(failure=failure.__name__):

                def raise_it(_: str, error: type[BaseException] = failure) -> str:
                    raise error

                translator = DocumentTranslator(raise_it, budget_chars=BUDGET)
                with self.assertRaises(failure):
                    translator.translate_document(TEN_PARAGRAPHS)


class ProgressAndSinkTest(unittest.TestCase):
    """The library reports only through what the caller injected."""

    def test_every_chunk_produces_an_event(self) -> None:
        events: list[Progress] = []
        chunks = split_document(TEN_PARAGRAPHS, budget_chars=BUDGET)

        DocumentTranslator(
            Recorder(), budget_chars=BUDGET, progress=events.append
        ).translate_document(TEN_PARAGRAPHS, source_name="doc.txt")

        done = [event for event in events if event.phase == "chunk_done"]
        self.assertEqual(len(done), len(chunks))
        self.assertEqual(done[-1].chars_done, done[-1].chars_total)
        self.assertEqual(done[0].source_name, "doc.txt")

    def test_the_sink_receives_exactly_the_assembled_document(self) -> None:
        written: list[str] = []

        result = DocumentTranslator(Recorder(), budget_chars=BUDGET).translate_document(
            TEN_PARAGRAPHS, sink=written.append
        )

        self.assertEqual("".join(written), result.text)


@dataclass(frozen=True)
class _Response:
    text: str
    finish_reason: str | None


class _Tokenizer:
    """Echoes the chunk back, and advertises the delimiters the guard looks for."""

    think_start = THINK_START
    think_end = THINK_END

    def apply_chat_template(self, messages: list[dict[str, str]], **_: object) -> str:
        return messages[-1]["content"]


class _Stream:
    """One bad chunk out of three: either a reasoning trace or a run out of budget."""

    def __init__(self, *, trace: bool = False, truncate: bool = False) -> None:
        self._trace = trace
        self._truncate = truncate

    def __call__(self, _model: object, _tokenizer: object, **kwargs: object) -> Iterator[_Response]:
        prompt = str(kwargs["prompt"])
        if BAD_CHUNK in prompt:
            if self._trace:
                yield _Response(f"{TRACE}[ko]{prompt}", "stop")
                return
            if self._truncate:
                yield _Response("[ko]Второй аб", "length")
                return
        yield _Response(f"[ko]{prompt}", "stop")


class ContaminationAndTruncationTest(unittest.TestCase):
    """A chunk the model did not translate must be reported, never assembled."""

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def _translate(self, **stream_kwargs: bool) -> DocumentResult:
        translator = Translator(
            loader=lambda _model_id: ("<model>", _Tokenizer()),
            generator=_Stream(**stream_kwargs),
        )
        return translator.translate_document(THREE_PARAGRAPHS, budget_tokens=BUDGET)

    def _returned_not_raised(self, replacement: str) -> DocumentResult:
        def translate_chunk(text: str) -> str:
            return replacement if text == BAD_CHUNK else f"[ko]{text}"

        return DocumentTranslator(translate_chunk, budget_chars=BUDGET).translate_document(
            THREE_PARAGRAPHS
        )

    def test_a_reasoning_trace_fails_its_chunk_instead_of_reaching_the_document(self) -> None:
        result = self._translate(trace=True)

        self.assertEqual(result.chunks_failed, (1,))
        self.assertNotIn(THINK_START, result.text)
        self.assertIn(UNTRANSLATED_OPEN.format(index=1), result.text)

    def test_a_returned_trace_is_assembled_and_the_run_looks_clean(self) -> None:
        result = self._returned_not_raised(f"{TRACE}[ko]{BAD_CHUNK}")

        self.assertEqual(result.chunks_failed, ())
        self.assertIn(THINK_START, result.text)

    def test_running_out_of_budget_fails_its_chunk_instead_of_shortening_the_document(self) -> None:
        result = self._translate(truncate=True)

        self.assertEqual(result.chunks_failed, (1,))
        self.assertNotIn("[ko]Второй аб", result.text)
        self.assertIn(UNTRANSLATED_OPEN.format(index=1), result.text)

    def test_a_returned_shortened_chunk_is_written_as_if_complete(self) -> None:
        result = self._returned_not_raised("[ko]Второй аб")

        self.assertEqual(result.chunks_failed, ())
        self.assertIn("[ko]Второй аб", result.text)


if __name__ == "__main__":
    unittest.main()
