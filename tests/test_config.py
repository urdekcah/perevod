import os
import unittest
from unittest import mock

from perevod.config import DEFAULT_MODEL_ID, MODEL_ENV_VAR, resolve_model_id

OTHER_MODEL = "mlx-community/gemma-4-31b-it-4bit"
ENV_MODEL = "mlx-community/some-other-model-4bit"


class ResolveModelIdTest(unittest.TestCase):
    def test_explicit_wins_over_env(self):
        with mock.patch.dict(os.environ, {MODEL_ENV_VAR: ENV_MODEL}):
            self.assertEqual(resolve_model_id(OTHER_MODEL), OTHER_MODEL)

    def test_env_wins_over_default(self):
        with mock.patch.dict(os.environ, {MODEL_ENV_VAR: ENV_MODEL}):
            self.assertEqual(resolve_model_id(), ENV_MODEL)

    def test_default_applies_when_nothing_is_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_model_id(), DEFAULT_MODEL_ID)

    def test_blank_env_does_not_shadow_default(self):
        for blank in ("", "   ", "\t\n"):
            with self.subTest(value=repr(blank)):
                with mock.patch.dict(os.environ, {MODEL_ENV_VAR: blank}):
                    self.assertEqual(resolve_model_id(), DEFAULT_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
