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
from pathlib import Path
from typing import Any
from unittest import mock

from perevod.adapters import AdapterMismatchError, AdapterProvenance, write_provenance
from perevod.config import ADAPTER_ENV_VAR, DEFAULT_MODEL_ID
from perevod.translator import (
    ADAPTER_LOAD_PARAM,
    Translator,
    build_messages,
    prompt_shape_fingerprint,
)

SOURCE = "Здравствуйте, как дела?"
TRANSLATION = "안녕하세요, 어떻게 지내세요?"


class FakeTokenizer:
    """Records the messages it is handed and renders a fixed prompt."""

    def __init__(self) -> None:
        self.seen_messages: list[dict[str, str]] | None = None

    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
        self.seen_messages = messages
        return "<rendered prompt>"


class TranslateTest(unittest.TestCase):
    """The seams the translator loads and generates through."""

    def setUp(self) -> None:
        # An adapter in the environment would change the loader call shape.
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tokenizer = FakeTokenizer()
        self.load_calls: list[str] = []
        self.generate_calls: list[dict[str, object]] = []

    def _loader(self, model_id: str) -> tuple[str, FakeTokenizer]:
        self.load_calls.append(model_id)
        return ("<model>", self.tokenizer)

    def _generator(self, _model: object, _tokenizer: object, **kwargs: object) -> str:
        self.generate_calls.append(kwargs)
        return f"  {TRANSLATION}\n"

    def test_translate_loads_once_sends_contract_shape_and_returns_output(self) -> None:
        translator = Translator(loader=self._loader, generator=self._generator)
        result = translator.translate(SOURCE)

        self.assertEqual(self.load_calls, [DEFAULT_MODEL_ID])
        self.assertEqual(self.tokenizer.seen_messages, build_messages(SOURCE))
        self.assertEqual(result, TRANSLATION)

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

    def _generator(self, *_args: Any, **_kwargs: Any) -> str:  # noqa: ANN401
        return TRANSLATION

    def _build(self, **kwargs: object) -> Translator:
        return Translator(loader=self._loader, generator=self._generator, **kwargs)  # type: ignore[arg-type]

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
