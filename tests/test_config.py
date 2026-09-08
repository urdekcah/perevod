# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Precedence for both selections: explicit argument, then environment, then the default."""

import os
import unittest
from unittest import mock

from perevod.config import (
    ADAPTER_ENV_VAR,
    DEFAULT_MODEL_ID,
    MODEL_ENV_VAR,
    resolve_adapter_path,
    resolve_model_id,
)

OTHER_MODEL = "mlx-community/gemma-4-31b-it-4bit"
ENV_MODEL = "mlx-community/some-other-model-4bit"

ENV_ADAPTER = "adapters/from-env"
EXPLICIT_ADAPTER = "adapters/explicit"


class ResolveModelIdTest(unittest.TestCase):
    """Which source wins, and what counts as absent."""

    def test_explicit_wins_over_env(self) -> None:
        with mock.patch.dict(os.environ, {MODEL_ENV_VAR: ENV_MODEL}):
            self.assertEqual(resolve_model_id(OTHER_MODEL), OTHER_MODEL)

    def test_env_wins_over_default(self) -> None:
        with mock.patch.dict(os.environ, {MODEL_ENV_VAR: ENV_MODEL}):
            self.assertEqual(resolve_model_id(), ENV_MODEL)

    def test_default_applies_when_nothing_is_set(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_model_id(), DEFAULT_MODEL_ID)

    def test_blank_env_does_not_shadow_default(self) -> None:
        for blank in ("", "   ", "\t\n"):
            with (
                self.subTest(value=repr(blank)),
                mock.patch.dict(os.environ, {MODEL_ENV_VAR: blank}),
            ):
                self.assertEqual(resolve_model_id(), DEFAULT_MODEL_ID)


class ResolveAdapterPathTest(unittest.TestCase):
    """An adapter is always opted into."""

    def test_explicit_wins_over_env(self) -> None:
        with mock.patch.dict(os.environ, {ADAPTER_ENV_VAR: ENV_ADAPTER}):
            self.assertEqual(resolve_adapter_path(EXPLICIT_ADAPTER), EXPLICIT_ADAPTER)

    def test_env_applies_when_nothing_is_passed(self) -> None:
        with mock.patch.dict(os.environ, {ADAPTER_ENV_VAR: ENV_ADAPTER}):
            self.assertEqual(resolve_adapter_path(), ENV_ADAPTER)

    def test_no_adapter_is_the_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_adapter_path())

    def test_blank_env_does_not_select_an_adapter(self) -> None:
        for blank in ("", "   ", "\t\n"):
            with (
                self.subTest(value=repr(blank)),
                mock.patch.dict(os.environ, {ADAPTER_ENV_VAR: blank}),
            ):
                self.assertIsNone(resolve_adapter_path())


if __name__ == "__main__":
    unittest.main()
