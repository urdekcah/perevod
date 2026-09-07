"""Asserted over parsed source, so this suite runs with click absent."""

import ast
import pathlib
import unittest

import perevod
from perevod.dataset import (
    MANIFEST_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALID_FILENAME,
)

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "perevod"

ENTRY_POINT_CALLS = {
    "PrepareConfig",
    "prepare_splits",
    "render_report",
    "echo",
    "ClickException",
    "option",
    "argument",
    "command",
    "Path",
    "str",
}


def parsed(name):
    return ast.parse((PACKAGE / name).read_text(encoding="utf-8"))


def called_names(node):
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        names.add(target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", ""))
    return names - {""}


class RecordShapeIsDefinedOnceTests(unittest.TestCase):
    def test_the_pipeline_holds_no_second_copy_of_the_shape(self):
        literals = {
            node.value
            for node in ast.walk(parsed("prepare.py"))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for word in ("role", "user", "assistant", "system"):
            with self.subTest(word=word):
                self.assertFalse([text for text in literals if word in text])

    def test_the_pipeline_builds_records_only_through_the_shared_builder(self):
        self.assertIn("build_training_record", called_names(parsed("prepare.py")))

    def test_each_split_file_name_is_written_once_in_the_package(self):
        sources = "".join(path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py"))
        for name in (TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME, MANIFEST_FILENAME):
            with self.subTest(name=name):
                self.assertEqual(sources.count(f'"{name}"'), 1)

    def test_the_pipeline_pulls_in_nothing_third_party(self):
        for module in ("dataset.py", "prepare.py"):
            roots = set()
            for node in ast.walk(parsed(module)):
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots.add(node.module.split(".")[0])
            with self.subTest(module=module):
                self.assertFalse(roots - set(__import__("sys").stdlib_module_names) - {"perevod"})


class EntryPointBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.function = next(
            node
            for node in ast.walk(parsed("cli.py"))
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_data"
        )

    def test_it_is_a_subcommand_on_the_existing_group(self):
        decorators = {
            decorator.func.value.id
            for decorator in self.function.decorator_list
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
        }
        self.assertIn("main", decorators)

    def test_it_delegates_rather_than_reimplementing_the_pipeline(self):
        self.assertFalse(called_names(self.function) - ENTRY_POINT_CALLS)


class PublicSurfaceTests(unittest.TestCase):
    def test_the_prepare_surface_is_exported_without_disturbing_the_earlier_one(self):
        for name in (
            "prepare_splits",
            "PrepareConfig",
            "PrepareResult",
            "PrepareError",
            "InputError",
            "OutputExistsError",
            "OutputPathError",
            "Translator",
            "DEFAULT_MODEL_ID",
            "resolve_model_id",
        ):
            with self.subTest(name=name):
                self.assertIn(name, perevod.__all__)
                self.assertTrue(hasattr(perevod, name))


if __name__ == "__main__":
    unittest.main()
