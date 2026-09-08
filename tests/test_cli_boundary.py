# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""CLI-to-library boundary, asserted over parsed source so click need not be installed."""

import ast
import pathlib
import unittest

from perevod.translator import TRANSLATION_INSTRUCTION

CLI_PATH = pathlib.Path(__file__).resolve().parent.parent / "perevod" / "cli.py"


class CliBoundaryTest(unittest.TestCase):
    """What the entry point is allowed to reach for."""

    def setUp(self) -> None:
        self.source = CLI_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def _imported_module_roots(self) -> set[str]:
        roots: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        return roots

    def test_imports_translator_from_the_library(self) -> None:
        imported = {
            alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom) and node.module == "perevod.translator"
            for alias in node.names
        }
        self.assertIn("Translator", imported)

    def test_does_not_import_the_inference_runtime(self) -> None:
        self.assertEqual(self._imported_module_roots() & {"mlx", "mlx_lm"}, set())

    def test_does_not_duplicate_the_prompt(self) -> None:
        self.assertNotIn(TRANSLATION_INSTRUCTION, self.source)

    def test_offers_the_adapter_options_on_translate(self) -> None:
        for option in ("--adapter-path", "--allow-provenance-mismatch"):
            with self.subTest(option=option):
                self.assertIn(f'"{option}"', self.source)

    def test_leaves_the_compatibility_decision_to_the_library(self) -> None:
        self.assertNotIn("check_adapter_compatibility", self.source)
        self.assertNotIn("read_provenance", self.source)


if __name__ == "__main__":
    unittest.main()
