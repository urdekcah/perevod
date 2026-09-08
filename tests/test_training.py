# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""What gets handed to the trainer, and what stops a run before it starts."""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from perevod.config import DEFAULT_MODEL_ID
from perevod.dataset import TRAIN_FILENAME, DatasetError, build_training_record
from perevod.training import (
    DEFAULT_RUN_NAME,
    TRAINER_FLAGS,
    AdapterExistsError,
    AdapterPathError,
    FlagSpec,
    TrainerFailedError,
    TrainingConfig,
    build_argv,
    resolve_adapter_path,
    train_adapter,
)


class Runner:
    """Stands in for the subprocess, and records whether it was reached."""

    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.argv: list[str] | None = None

    def __call__(self, argv: list[str]) -> int:
        self.argv = argv
        return self.code


class TrainingTestCase(unittest.TestCase):
    """A scratch tree holding a valid `train.jsonl`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.splits = self.root / "splits"
        self.splits.mkdir()
        self.adapter = self.root / "adapter"
        self.write_train(json.dumps(build_training_record("Привет", "안녕"), ensure_ascii=False))

    def write_train(self, text: str) -> None:
        (self.splits / TRAIN_FILENAME).write_text(text + "\n", encoding="utf-8")

    def config(self, **overrides: object) -> TrainingConfig:
        settings: dict[str, object] = {"data_dir": self.splits, "adapter_path": self.adapter}
        settings.update(overrides)
        return TrainingConfig(**settings)  # type: ignore[arg-type]


class ArgvTests(TrainingTestCase):
    """The invocation is a list, built only from flags the trainer itself advertises."""

    def test_it_starts_with_this_interpreter_and_the_trainer_module(self) -> None:
        argv = build_argv(self.config(), self.adapter)

        self.assertEqual(argv[:3], [sys.executable, "-m", "mlx_lm.lora"])
        self.assertIn("--train", argv)
        self.assertEqual(argv[argv.index("--model") + 1], DEFAULT_MODEL_ID)

    def test_an_unset_knob_emits_nothing_and_a_set_one_emits_a_single_pair(self) -> None:
        argv = build_argv(self.config(batch_size=2), self.adapter)

        self.assertNotIn("--iters", argv)
        self.assertNotIn("--learning-rate", argv)
        self.assertEqual(argv.count("--batch-size"), 1)
        self.assertEqual(argv[argv.index("--batch-size") + 1], "2")

    def test_a_boolean_flag_carries_no_value_and_stays_out_when_off(self) -> None:
        off = build_argv(self.config(), self.adapter)
        on = build_argv(self.config(grad_checkpoint=True), self.adapter)

        self.assertNotIn("--grad-checkpoint", off)
        self.assertEqual(on[on.index("--grad-checkpoint") + 1 :], [])

    def test_the_flag_names_come_from_the_injected_mapping_only(self) -> None:
        argv = build_argv(
            self.config(batch_size=2),
            self.adapter,
            flags={
                "train": FlagSpec("--go", takes_value=False),
                "batch_size": FlagSpec("--renamed"),
            },
        )

        self.assertEqual(argv, [sys.executable, "-m", "mlx_lm.lora", "--go", "--renamed", "2"])

    def test_every_advertised_flag_is_a_long_option(self) -> None:
        for name, spec in TRAINER_FLAGS.items():
            with self.subTest(field=name):
                self.assertTrue(spec.flag.startswith("--"))


class GateTests(TrainingTestCase):
    """Nothing reaches the trainer until the data has been read and accepted."""

    def test_malformed_data_stops_the_run_before_the_trainer_is_reached(self) -> None:
        self.write_train("not json at all")
        runner = Runner()

        with self.assertRaises(DatasetError) as caught:
            train_adapter(self.config(), runner=runner)

        self.assertIsNone(runner.argv)
        self.assertIn(str(self.splits / TRAIN_FILENAME), str(caught.exception))

    def test_a_clean_directory_runs_and_reports_what_was_read(self) -> None:
        runner = Runner()

        adapter_path, report = train_adapter(self.config(), runner=runner)

        self.assertEqual(adapter_path, self.adapter.resolve())
        self.assertEqual(dict(report.rows), {TRAIN_FILENAME: 1})
        self.assertIsNotNone(runner.argv)

    def test_a_non_zero_exit_is_raised_rather_than_reported_as_success(self) -> None:
        with self.assertRaises(TrainerFailedError) as caught:
            train_adapter(self.config(), runner=Runner(code=1))

        self.assertIn("exited 1", str(caught.exception))


class AdapterPathTests(TrainingTestCase):
    """Hours of compute live in the output directory; it is not overwritten by accident."""

    def test_the_default_path_is_deterministic_and_under_the_data_tree(self) -> None:
        with contextlib.chdir(self.root):
            resolved = resolve_adapter_path(TrainingConfig(data_dir=self.splits))

        self.assertEqual(resolved, (self.root / "data/adapters" / DEFAULT_RUN_NAME).resolve())

    def test_an_existing_adapter_is_refused_unless_overwrite_is_asked_for(self) -> None:
        self.adapter.mkdir()
        (self.adapter / "adapters.safetensors").write_bytes(b"weights")
        runner = Runner()

        with self.assertRaises(AdapterExistsError):
            train_adapter(self.config(), runner=runner)
        self.assertIsNone(runner.argv)

        train_adapter(self.config(overwrite=True), runner=runner)
        self.assertIsNotNone(runner.argv)

    def test_an_empty_directory_is_not_treated_as_an_existing_adapter(self) -> None:
        self.adapter.mkdir()

        train_adapter(self.config(), runner=Runner())

    def test_writing_the_adapter_into_the_splits_is_refused(self) -> None:
        with self.assertRaises(AdapterPathError):
            resolve_adapter_path(self.config(adapter_path=self.splits / "run"))
