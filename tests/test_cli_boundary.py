"""CLI-to-library boundary, asserted over parsed source so click need not be installed."""

import ast
import pathlib
import unittest

from perevod.translator import TRANSLATION_INSTRUCTION

CLI_PATH = pathlib.Path(__file__).resolve().parent.parent / "perevod" / "cli.py"


class CliBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.source = CLI_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def _imported_module_roots(self):
        roots = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        return roots

    def test_imports_translator_from_the_library(self):
        imported = {
            alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom) and node.module == "perevod.translator"
            for alias in node.names
        }
        self.assertIn("Translator", imported)

    def test_does_not_import_the_inference_runtime(self):
        self.assertEqual(self._imported_module_roots() & {"mlx", "mlx_lm"}, set())

    def test_does_not_duplicate_the_prompt(self):
        self.assertNotIn(TRANSLATION_INSTRUCTION, self.source)


if __name__ == "__main__":
    unittest.main()
