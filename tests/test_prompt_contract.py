# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""The prompt shape every trained adapter is bound to."""

import unittest

from perevod.dataset import prompt_shape_fingerprint
from perevod.translator import TRANSLATION_INSTRUCTION, build_messages

# Changing either pin invalidates every adapter already trained; there is no migration.
SHIPPED_ROLES = ("system", "user")
SHIPPED_FINGERPRINT = "afd13f8761ceb2f9eb1d300b4220a1a5f6e20637a001fdf72dd2c66ae0342228"


class ShippedShapeTests(unittest.TestCase):
    """The shape every adapter is fitted against."""

    def test_the_role_sequence_is_the_one_adapters_were_trained_against(self) -> None:
        roles = tuple(message["role"] for message in build_messages("Привет"))

        self.assertEqual(roles, SHIPPED_ROLES)

    def test_the_instruction_text_has_not_moved(self) -> None:
        self.assertEqual(
            TRANSLATION_INSTRUCTION,
            "Translate the following Russian text into Korean. "
            "Reply with the Korean translation only, with no explanation, "
            "no transliteration, and no restatement of the source.",
        )

    def test_the_whole_shape_still_hashes_to_what_the_splits_recorded(self) -> None:
        self.assertEqual(prompt_shape_fingerprint(), SHIPPED_FINGERPRINT)
