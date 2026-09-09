# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""What gets handed to the trainer, and what stops a run before it starts."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from perevod.config import DEFAULT_MODEL_ID
from perevod.dataset import (
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALID_FILENAME,
    DatasetError,
    SplitReport,
    build_training_record,
)
from perevod.training import (
    DEFAULT_RUN_NAME,
    TRAINER_FLAGS,
    AdapterExistsError,
    AdapterPathError,
    FlagSpec,
    SplitTooSmallError,
    TrainerFailedError,
    TrainingConfig,
    TrainingError,
    _preflight_batch_size,
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

    def write_split(self, name: str, rows: int) -> None:
        record = json.dumps(build_training_record("Привет", "안녕"), ensure_ascii=False)
        (self.splits / name).write_text(f"{record}\n" * rows, encoding="utf-8")

    def config(self, **overrides: object) -> TrainingConfig:
        settings: dict[str, object] = {"data_dir": self.splits, "adapter_path": self.adapter}
        settings.update(overrides)
        return TrainingConfig(**settings)  # type: ignore[arg-type]


class ArgvTests(TrainingTestCase):
    """The invocation is a list, built only from flags the trainer itself advertises."""

    def test_it_starts_with_this_interpreter_and_the_trainer_module(self) -> None:
        argv = build_argv(self.config(), self.adapter)

        self.assertEqual(argv[:4], [sys.executable, "-m", "mlx_lm", "lora"])
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

        self.assertEqual(argv, [sys.executable, "-m", "mlx_lm", "lora", "--go", "--renamed", "2"])

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

        with contextlib.redirect_stderr(io.StringIO()) as captured:
            train_adapter(self.config(), runner=Runner())

        self.assertEqual(captured.getvalue(), "")

    def test_debris_from_a_failed_run_does_not_block_the_retry(self) -> None:
        # A failed run leaves exactly this: config written, no weights.
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        runner = Runner()

        with contextlib.redirect_stderr(io.StringIO()) as captured:
            train_adapter(self.config(), runner=runner)

        self.assertIsNotNone(runner.argv)
        notice = captured.getvalue()
        self.assertIn(str(self.adapter), notice)
        self.assertIn("not removed", notice)
        self.assertTrue((self.adapter / "adapter_config.json").is_file())

    def test_weights_beside_other_files_are_still_protected(self) -> None:
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (self.adapter / "adapters.safetensors").write_bytes(b"weights")
        runner = Runner()

        with self.assertRaises(AdapterExistsError):
            train_adapter(self.config(), runner=runner)
        self.assertIsNone(runner.argv)

    def test_writing_the_adapter_into_the_splits_is_refused(self) -> None:
        with self.assertRaises(AdapterPathError):
            resolve_adapter_path(self.config(adapter_path=self.splits / "run"))


class PreflightTests(TrainingTestCase):
    """A run that cannot form a first batch is refused before it costs anything."""

    def test_every_split_short_of_the_batch_is_named_with_its_count(self) -> None:
        self.write_split(VALID_FILENAME, 2)

        with self.assertRaises(SplitTooSmallError) as caught:
            train_adapter(self.config(batch_size=4), runner=Runner())

        message = str(caught.exception)
        self.assertIn(f"{self.splits / TRAIN_FILENAME} holds 1 row(s)", message)
        self.assertIn(f"{self.splits / VALID_FILENAME} holds 2 row(s)", message)
        self.assertIn("4", message)

    def test_a_split_that_fits_is_left_out_of_the_refusal(self) -> None:
        self.write_split(TRAIN_FILENAME, 8)
        self.write_split(VALID_FILENAME, 2)

        with self.assertRaises(SplitTooSmallError) as caught:
            train_adapter(self.config(batch_size=4), runner=Runner())

        message = str(caught.exception)
        self.assertNotIn(str(self.splits / TRAIN_FILENAME), message)
        self.assertIn(str(self.splits / VALID_FILENAME), message)

    def test_splits_at_the_batch_size_reach_the_trainer(self) -> None:
        self.write_split(TRAIN_FILENAME, 4)
        self.write_split(VALID_FILENAME, 4)
        runner = Runner()

        train_adapter(self.config(batch_size=4), runner=runner)

        self.assertIsNotNone(runner.argv)

    def test_the_test_split_is_not_checked_because_the_run_never_iterates_it(self) -> None:
        self.write_split(TRAIN_FILENAME, 8)
        self.write_split(TEST_FILENAME, 1)
        runner = Runner()

        train_adapter(self.config(batch_size=8), runner=runner)

        self.assertIsNotNone(runner.argv)

    def test_an_empty_split_is_refused_whatever_the_batch_size(self) -> None:
        # Called directly: validate_split_dir refuses an empty file before train_adapter can.
        for name in (TRAIN_FILENAME, VALID_FILENAME):
            with self.subTest(split=name), self.assertRaises(SplitTooSmallError):
                _preflight_batch_size(SplitReport(self.splits, {name: 0}, ()), None)

    def test_an_unset_batch_size_records_why_the_comparison_was_skipped(self) -> None:
        _, unset = train_adapter(self.config(), runner=Runner())
        _, given = train_adapter(
            self.config(adapter_path=self.root / "sized", batch_size=1), runner=Runner()
        )

        self.assertTrue(any("batch size is unset" in notice for notice in unset.notices))
        self.assertFalse(any("batch size is unset" in notice for notice in given.notices))

    def test_a_refusal_creates_no_directory_and_spawns_nothing(self) -> None:
        runner = Runner()

        with self.assertRaises(SplitTooSmallError):
            train_adapter(self.config(batch_size=4), runner=runner)

        self.assertIsNone(runner.argv)
        self.assertFalse(self.adapter.exists())

    def test_the_refusal_surfaces_through_the_cli_handler_already_in_place(self) -> None:
        # The CLI already converts TrainingError, so the subclass is what avoids a CLI edit.
        self.assertTrue(issubclass(SplitTooSmallError, TrainingError))
