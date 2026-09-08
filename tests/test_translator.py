# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Translation path over injected loader and generator, with MLX absent."""

import subprocess
import sys
import unittest

from perevod.config import DEFAULT_MODEL_ID
from perevod.translator import Translator, build_messages

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


if __name__ == "__main__":
    unittest.main()
