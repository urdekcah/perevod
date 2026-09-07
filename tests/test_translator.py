"""Translation path over injected loader and generator, with MLX absent."""

import subprocess
import sys
import unittest

from perevod.config import DEFAULT_MODEL_ID
from perevod.translator import Translator, build_messages

SOURCE = "Здравствуйте, как дела?"
TRANSLATION = "안녕하세요, 어떻게 지내세요?"


class FakeTokenizer:
    def __init__(self):
        self.seen_messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.seen_messages = messages
        return "<rendered prompt>"


class TranslateTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.load_calls = []
        self.generate_calls = []

    def _loader(self, model_id):
        self.load_calls.append(model_id)
        return ("<model>", self.tokenizer)

    def _generator(self, model, tokenizer, **kwargs):
        self.generate_calls.append(kwargs)
        return f"  {TRANSLATION}\n"

    def test_translate_loads_once_sends_contract_shape_and_returns_output(self):
        translator = Translator(loader=self._loader, generator=self._generator)
        result = translator.translate(SOURCE)

        self.assertEqual(self.load_calls, [DEFAULT_MODEL_ID])
        self.assertEqual(self.tokenizer.seen_messages, build_messages(SOURCE))
        self.assertEqual(result, TRANSLATION)

    def test_import_does_not_pull_in_mlx(self):
        # A module-scope mlx_lm import would break this; the deferred one must stay.
        probe = "import perevod, sys; assert 'mlx_lm' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
