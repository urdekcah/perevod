# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Translation path over injected loader and generator, with MLX absent."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from perevod.adapters import AdapterMismatchError, AdapterProvenance, write_provenance
from perevod.config import ADAPTER_ENV_VAR, DEFAULT_MODEL_ID
from perevod.translator import (
    ADAPTER_LOAD_PARAM,
    ReasoningLeakError,
    Translator,
    TruncatedTranslationError,
    build_messages,
    prompt_shape_fingerprint,
)

SOURCE = "Здравствуйте, как дела?"
TRANSLATION = "안녕하세요, 어떻게 지내세요?"

THINK_START = "<|channel>thought"
THINK_END = "<channel|>"


class FakeTokenizer:
    """Records the render call; advertises reasoning delimiters only when asked to."""

    def __init__(self, *, delimiters: tuple[str, str] | None = None) -> None:
        self.seen_messages: list[dict[str, str]] | None = None
        self.seen_kwargs: dict[str, object] = {}
        if delimiters is not None:
            self.think_start, self.think_end = delimiters

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.seen_messages = messages
        self.seen_kwargs = kwargs
        return "<rendered prompt>"


@dataclass(frozen=True)
class FakeResponse:
    """One yield of the generation seam."""

    text: str
    finish_reason: str | None


class FakeStream:
    """Yields incremental segments; only the terminal one carries a finish reason."""

    def __init__(self, *segments: str, finish_reason: str | None = "stop") -> None:
        self.segments = segments
        self.finish_reason = finish_reason
        self.calls: list[dict[str, object]] = []

    def __call__(
        self, _model: object, _tokenizer: object, **kwargs: object
    ) -> Iterator[FakeResponse]:
        self.calls.append(kwargs)
        last = len(self.segments) - 1
        for index, segment in enumerate(self.segments):
            yield FakeResponse(segment, self.finish_reason if index == last else None)


class TranslateTest(unittest.TestCase):
    """The seams the translator loads and generates through."""

    def setUp(self) -> None:
        # An adapter in the environment would change the loader call shape.
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tokenizer = FakeTokenizer()
        self.load_calls: list[str] = []

    def _loader(self, model_id: str) -> tuple[str, FakeTokenizer]:
        self.load_calls.append(model_id)
        return ("<model>", self.tokenizer)

    def _build(self, stream: FakeStream) -> Translator:
        return Translator(loader=self._loader, generator=stream)

    def test_translate_loads_once_sends_contract_shape_and_returns_output(self) -> None:
        translator = self._build(FakeStream(f"  {TRANSLATION}\n"))

        result = translator.translate(SOURCE)

        self.assertEqual(self.load_calls, [DEFAULT_MODEL_ID])
        self.assertEqual(self.tokenizer.seen_messages, build_messages(SOURCE))
        self.assertEqual(result, TRANSLATION)

    def test_the_render_call_turns_the_reasoning_channel_off(self) -> None:
        self._build(FakeStream(TRANSLATION)).translate(SOURCE)

        self.assertEqual(
            self.tokenizer.seen_kwargs,
            {"add_generation_prompt": True, "tokenize": False, "enable_thinking": False},
        )

    def test_the_generation_call_carries_no_parameter_the_seam_lacks(self) -> None:
        stream = FakeStream(TRANSLATION)

        self._build(stream).translate(SOURCE, max_tokens=64)

        self.assertEqual(stream.calls, [{"prompt": "<rendered prompt>", "max_tokens": 64}])

    def test_import_does_not_pull_in_mlx(self) -> None:
        # A module-scope mlx_lm import would break this; the deferred one must stay.
        probe = "import perevod, sys; assert 'mlx_lm' not in sys.modules"
        completed = subprocess.run(  # noqa: S603 -- the interpreter running this test
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class AccumulationTest(unittest.TestCase):
    """Segments arrive one at a time and the terminal one is part of the answer."""

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tokenizer = FakeTokenizer()

    def _build(self, stream: FakeStream) -> Translator:
        return Translator(loader=lambda _model_id: ("<model>", self.tokenizer), generator=stream)

    def test_every_segment_including_the_terminal_one_reaches_the_result(self) -> None:
        # Returning only the last segment gives "요"; dropping it gives "안녕하세".
        result = self._build(FakeStream("  안녕", "하세", "요\n")).translate(SOURCE)

        self.assertEqual(result, "안녕하세요")

    def test_the_result_is_the_exact_concatenation_of_what_was_yielded(self) -> None:
        segments = ("첫", " 번", "째 ", "문장.")

        result = self._build(FakeStream(*segments)).translate(SOURCE)

        self.assertEqual(result, "".join(segments))


class TruncationTest(unittest.TestCase):
    """A run that stopped without the model saying it was done is a failure, not a result."""

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tokenizer = FakeTokenizer()

    def _build(self, stream: FakeStream) -> Translator:
        return Translator(loader=lambda _model_id: ("<model>", self.tokenizer), generator=stream)

    def test_a_finished_run_returns_its_text(self) -> None:
        result = self._build(FakeStream(TRANSLATION, finish_reason="stop")).translate(SOURCE)

        self.assertEqual(result, TRANSLATION)

    def test_running_out_of_budget_raises(self) -> None:
        translator = self._build(FakeStream("안녕하", finish_reason="length"))

        with self.assertRaises(TruncatedTranslationError):
            translator.translate(SOURCE)

    def test_the_error_carries_the_budget_and_the_text_that_was_produced(self) -> None:
        translator = self._build(FakeStream("안녕", "하", finish_reason="length"))

        with self.assertRaises(TruncatedTranslationError) as caught:
            translator.translate(SOURCE, max_tokens=7)

        self.assertEqual(caught.exception.max_tokens, 7)
        self.assertEqual(caught.exception.partial_text, "안녕하")
        self.assertIn("7", str(caught.exception))

    def test_a_terminal_response_without_a_reason_is_not_a_completion_signal(self) -> None:
        translator = self._build(FakeStream("안녕하", finish_reason=None))

        with self.assertRaises(TruncatedTranslationError):
            translator.translate(SOURCE)


class ReasoningGuardTest(unittest.TestCase):
    """Suppression is asserted at the output, never patched up."""

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def _build(self, tokenizer: FakeTokenizer, stream: FakeStream) -> Translator:
        return Translator(loader=lambda _model_id: ("<model>", tokenizer), generator=stream)

    def test_a_clean_completion_is_returned_unchanged(self) -> None:
        tokenizer = FakeTokenizer(delimiters=(THINK_START, THINK_END))

        result = self._build(tokenizer, FakeStream(TRANSLATION)).translate(SOURCE)

        self.assertEqual(result, TRANSLATION)

    def test_a_surviving_delimiter_raises_and_names_itself(self) -> None:
        tokenizer = FakeTokenizer(delimiters=(THINK_START, THINK_END))
        contaminated = f"{THINK_START} The user wants Korean. {THINK_END}{TRANSLATION}"

        with self.assertRaises(ReasoningLeakError) as caught:
            self._build(tokenizer, FakeStream(contaminated)).translate(SOURCE)

        self.assertEqual(caught.exception.delimiter, THINK_START)
        self.assertIn(THINK_START, str(caught.exception))

    def test_a_tokenizer_with_no_delimiters_degrades_to_no_check(self) -> None:
        result = self._build(FakeTokenizer(), FakeStream(f"{THINK_START} whatever")).translate(
            SOURCE
        )

        self.assertEqual(result, f"{THINK_START} whatever")


class GuardCoverageTest(unittest.TestCase):
    """The fakes must be able to produce what the guards catch, or they prove nothing."""

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tokenizer = FakeTokenizer(delimiters=(THINK_START, THINK_END))

    def _build(self, stream: FakeStream) -> Translator:
        return Translator(loader=lambda _model_id: ("<model>", self.tokenizer), generator=stream)

    def test_the_fake_tokenizer_advertises_delimiters_the_guard_can_find(self) -> None:
        self.assertEqual(self.tokenizer.think_start, THINK_START)
        self.assertEqual(self.tokenizer.think_end, THINK_END)

    def test_without_the_guard_the_reasoning_trace_flows_straight_through(self) -> None:
        trace = f"{THINK_START} рассуждение {THINK_END}{TRANSLATION}"
        translator = self._build(FakeStream(trace))

        with mock.patch.object(Translator, "_leaked_delimiter", return_value=None):
            self.assertEqual(translator.translate(SOURCE), trace)

    def test_truncated_text_is_reachable_only_through_the_error(self) -> None:
        translator = self._build(FakeStream("안녕", "하세", finish_reason="length"))

        with self.assertRaises(TruncatedTranslationError) as caught:
            translator.translate(SOURCE)

        self.assertEqual(caught.exception.partial_text, "안녕하세")


class AdapterLoadTest(unittest.TestCase):
    """Whether an adapter was asked for decides the shape of the loader call."""

    def setUp(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.directory = Path(tempfile.mkdtemp())
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def _loader(self, *args: object, **kwargs: object) -> tuple[str, FakeTokenizer]:
        self.calls.append((args, kwargs))
        return ("<model>", FakeTokenizer())

    def _build(self, **kwargs: object) -> Translator:
        return Translator(loader=self._loader, generator=FakeStream(TRANSLATION), **kwargs)  # type: ignore[arg-type]

    def _annotate(self, **overrides: str) -> None:
        record = AdapterProvenance(
            base_model_id=overrides.get("base_model_id", DEFAULT_MODEL_ID),
            prompt_shape_fingerprint=overrides.get("fingerprint", prompt_shape_fingerprint()),
        )
        write_provenance(self.directory, record)

    def test_without_an_adapter_the_loader_call_is_the_one_shipped_before(self) -> None:
        self._build()

        self.assertEqual(self.calls, [((DEFAULT_MODEL_ID,), {})])

    def test_an_adapter_reaches_the_loader_once_under_the_resolved_parameter(self) -> None:
        self._annotate()

        self._build(adapter_path=str(self.directory))

        self.assertEqual(
            self.calls,
            [((DEFAULT_MODEL_ID,), {ADAPTER_LOAD_PARAM: str(self.directory)})],
        )

    def test_the_environment_selects_an_adapter_when_none_is_passed(self) -> None:
        self._annotate()
        os.environ[ADAPTER_ENV_VAR] = str(self.directory)

        translator = self._build()

        self.assertEqual(translator.adapter_path, str(self.directory))

    def test_an_adapter_from_another_base_fails_before_the_model_loads(self) -> None:
        self._annotate(base_model_id="mlx-community/gemma-4-31b-it-4bit")

        with self.assertRaises(AdapterMismatchError):
            self._build(adapter_path=str(self.directory))
        self.assertEqual(self.calls, [])

    def test_the_override_lets_that_pairing_through(self) -> None:
        self._annotate(base_model_id="mlx-community/gemma-4-31b-it-4bit")

        with contextlib.redirect_stderr(io.StringIO()) as warned:
            self._build(adapter_path=str(self.directory), allow_provenance_mismatch=True)

        self.assertIn("gemma-4-31b-it-4bit", warned.getvalue())

        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
