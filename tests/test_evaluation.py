# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""The refusals, the floor, and the run record — all provable with no model present."""

import ast
import json
import tempfile
import unittest
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perevod import evaluation, metrics
from perevod.adapters import AdapterProvenance, write_provenance
from perevod.dataset import TRAIN_FILENAME, VALID_FILENAME, RecordShapeError, build_training_record
from perevod.evaluation import (
    ArmResult,
    Comparison,
    ConfoundedComparisonError,
    DecodingMismatchError,
    DecodingSettings,
    EvaluationConfig,
    EvaluationError,
    RecordError,
    SplitIntegrityError,
    check_split_isolation,
    compare_arms,
    compare_runs,
    evaluate_pair,
    load_split,
    read_run,
    write_run,
)

PAIRS = (
    ("Здравствуйте.", "안녕하세요."),
    ("Как дела?", "어떻게 지내세요?"),
    ("Спасибо.", "감사합니다."),
    ("До свидания.", "안녕히 가세요."),
)

MODEL_ID = "test/model"
FROZEN = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)


def write_split(path: Path, pairs: tuple[tuple[str, str], ...]) -> Path:
    """Render pairs through the shipped record builder, the way a real split is written."""
    path.write_text(
        "".join(
            json.dumps(build_training_record(source, target), ensure_ascii=False) + "\n"
            for source, target in pairs
        ),
        encoding="utf-8",
    )
    return path


class FakeTranslator:
    """Returns whatever the arm was told to return, so the harness can be driven without MLX."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs

    def translate(self, source_text: str, *, max_tokens: int) -> str:
        del max_tokens
        return self.outputs.get(source_text, source_text)


def factory_for(base: dict[str, str], adapter: dict[str, str]) -> Callable[..., FakeTranslator]:
    """A translator factory whose two arms answer from fixed tables."""

    def factory(
        model_id: str, *, adapter_path: str | None, allow_provenance_mismatch: bool
    ) -> FakeTranslator:
        del model_id, allow_provenance_mismatch
        return FakeTranslator(adapter if adapter_path else base)

    return factory


def arm(  # noqa: PLR0913 -- one parameter per recorded field
    name: str,
    corpus: float,
    *,
    segments: tuple[float, ...] = (),
    outputs: tuple[str, ...] = (),
    repeats: tuple[float, ...] = (),
    fingerprint: str = "identical",
) -> ArmResult:
    """One arm's result, filled in far enough for the rule under test."""
    return ArmResult(
        name=name,
        model_id=MODEL_ID,
        adapter_path=None,
        outputs=outputs,
        repeat_scores=repeats or (corpus,),
        corpus_chrf=corpus,
        segment_scores=segments,
        seconds=0.0,
        decoding_fingerprint=fingerprint,
    )


class SourceSurfaceTest(unittest.TestCase):
    """What the modules may not contain, asserted over the source with nothing installed."""

    EVALUATION = Path("perevod/evaluation.py").read_text(encoding="utf-8")
    CLI = Path("perevod/cli.py").read_text(encoding="utf-8")

    def _imported_modules(self, source: str) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    def test_evaluation_never_reaches_the_model_except_through_the_translator(self) -> None:
        imported = self._imported_modules(self.EVALUATION)

        self.assertNotIn("mlx", imported)
        self.assertNotIn("mlx_lm", imported)
        for forbidden in ("apply_chat_template", "add_generation_prompt", "Translate the"):
            self.assertNotIn(forbidden, self.EVALUATION)

    def test_evaluation_does_not_destructure_records_itself(self) -> None:
        for forbidden in ('"messages"', '"role"', '"content"'):
            self.assertNotIn(forbidden, self.EVALUATION)

    def test_the_cli_carries_no_metric_integrity_or_artifact_logic(self) -> None:
        for forbidden in ("sha256", "chrF", "ngram", "index.jsonl", "canonical_text"):
            self.assertNotIn(forbidden, self.CLI)

    def test_the_package_contains_no_bleu_implementation(self) -> None:
        for module in Path("perevod").glob("*.py"):
            text = module.read_text(encoding="utf-8").lower()
            with self.subTest(module.name):
                self.assertNotIn("bleu", text)
                self.assertNotIn("brevity", text)


class SplitIntegrityTest(unittest.TestCase):
    """A split that overlaps training data is refused, never repaired and never scored."""

    def test_a_clean_split_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            write_split(directory / TRAIN_FILENAME, PAIRS[:2])
            test = write_split(directory / "test.jsonl", PAIRS[2:])

            self.assertEqual(
                check_split_isolation(load_split(test), test_path=test), (TRAIN_FILENAME,)
            )

    def test_a_shared_source_is_refused_even_with_another_translation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            write_split(directory / TRAIN_FILENAME, (("Спасибо.", "고맙습니다."),))
            test = write_split(directory / "test.jsonl", PAIRS[2:])

            with self.assertRaises(SplitIntegrityError) as caught:
                check_split_isolation(load_split(test), test_path=test)
            self.assertIn("shared source", str(caught.exception))

    def test_a_shared_pair_is_refused_and_names_both_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            write_split(directory / VALID_FILENAME, PAIRS[:1])
            test = write_split(directory / "test.jsonl", PAIRS[:1])

            with self.assertRaises(SplitIntegrityError) as caught:
                check_split_isolation(load_split(test), test_path=test)
            message = str(caught.exception)
            self.assertIn("shared pair", message)
            self.assertIn(":1 =", message)

    def test_the_checker_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            train = write_split(directory / TRAIN_FILENAME, PAIRS[:2])
            test = write_split(directory / "test.jsonl", PAIRS[2:])
            before = {path.name: path.read_bytes() for path in directory.iterdir()}

            check_split_isolation(load_split(test), test_path=test)

            self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, before)
            self.assertEqual(sorted(before), sorted([train.name, test.name]))

    def test_a_row_from_another_prompt_shape_is_rejected_with_its_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            test = Path(raw) / "test.jsonl"
            test.write_text('{"messages": []}\n', encoding="utf-8")

            with self.assertRaises(RecordShapeError) as caught:
                load_split(test)
            self.assertIn(":1:", str(caught.exception))


class DecodingTest(unittest.TestCase):
    """One settings value per run, and arms that disagree on it are never compared."""

    def test_the_fingerprint_moves_with_the_settings(self) -> None:
        self.assertEqual(
            DecodingSettings(max_tokens=64).fingerprint(),
            DecodingSettings(max_tokens=64).fingerprint(),
        )
        self.assertNotEqual(
            DecodingSettings(max_tokens=64).fingerprint(),
            DecodingSettings(max_tokens=128).fingerprint(),
        )

    def test_arms_that_decoded_differently_raise_instead_of_comparing(self) -> None:
        with self.assertRaises(DecodingMismatchError):
            compare_arms(
                arm("base", 40.0, fingerprint="one"),
                arm("adapter", 90.0, fingerprint="two"),
            )


class ComparisonRuleTest(unittest.TestCase):
    """The precondition, the floor, and the closed verdict set."""

    def _compare(
        self,
        base_outputs: tuple[str, ...],
        adapter_outputs: tuple[str, ...],
        base: float = 40.0,
        adapter: float = 90.0,
    ) -> Comparison:
        return compare_arms(
            arm("base", base, outputs=base_outputs, segments=(base,) * len(base_outputs)),
            arm(
                "adapter",
                adapter,
                outputs=adapter_outputs,
                segments=(adapter,) * len(adapter_outputs),
            ),
        )

    def test_identical_arms_are_indistinguishable_despite_a_large_delta(self) -> None:
        outputs = ("가", "나", "다")

        verdict = self._compare(outputs, outputs)

        self.assertEqual(verdict.verdict, evaluation.INDISTINGUISHABLE)
        self.assertEqual(verdict.distinguishable_fraction, 0.0)
        self.assertIn("never have been applied", verdict.note or "")

    def test_one_third_differing_is_still_below_the_precondition(self) -> None:
        verdict = self._compare(("가", "나", "다"), ("가", "나", "라"))

        self.assertLess(verdict.distinguishable_fraction, evaluation.MIN_DISTINGUISHABLE_FRACTION)
        self.assertEqual(verdict.verdict, evaluation.INDISTINGUISHABLE)

    def test_fully_differing_arms_take_the_normal_path(self) -> None:
        verdict = self._compare(("가", "나", "다"), ("라", "마", "바"))

        self.assertEqual(verdict.distinguishable_fraction, 1.0)
        self.assertEqual(verdict.verdict, evaluation.IMPROVED)

    def test_a_regression_is_named_a_regression(self) -> None:
        verdict = self._compare(("가", "나"), ("라", "마"), base=90.0, adapter=40.0)

        self.assertEqual(verdict.verdict, evaluation.REGRESSED)

    def test_a_sub_convention_delta_is_never_an_improvement(self) -> None:
        verdict = self._compare(("가", "나"), ("라", "마"), base=50.0, adapter=50.5)

        self.assertEqual(verdict.floor, evaluation.MIN_REPORTABLE_DELTA_CHRF)
        self.assertEqual(verdict.floor_bases[evaluation.FLOOR_CONVENTION], 1.0)
        self.assertEqual(verdict.verdict, evaluation.INDISTINGUISHABLE)

    def test_the_repeat_spread_can_raise_the_floor_above_the_convention(self) -> None:
        verdict = compare_arms(
            arm("base", 50.0, outputs=("가",), segments=(50.0,), repeats=(50.0, 45.0)),
            arm("adapter", 53.0, outputs=("라",), segments=(53.0,), repeats=(53.0, 53.0)),
        )

        self.assertEqual(verdict.floor_bases[evaluation.FLOOR_REPEAT_SPREAD], 5.0)
        self.assertEqual(verdict.floor, 5.0)
        self.assertEqual(verdict.verdict, evaluation.INDISTINGUISHABLE)

    def test_the_floor_is_the_maximum_over_every_basis_that_produced_one(self) -> None:
        verdict = compare_arms(
            arm("base", 50.0, outputs=("가",), segments=(50.0,), repeats=(50.0, 49.5)),
            arm("adapter", 53.0, outputs=("라",), segments=(53.0,), repeats=(53.0, 53.0)),
        )

        self.assertEqual(verdict.floor_bases[evaluation.FLOOR_REPEAT_SPREAD], 0.5)
        self.assertEqual(verdict.floor, evaluation.MIN_REPORTABLE_DELTA_CHRF)
        self.assertEqual(verdict.verdict, evaluation.IMPROVED)

    def test_the_win_loss_tie_tally_counts_segments_not_the_corpus(self) -> None:
        verdict = compare_arms(
            arm("base", 50.0, outputs=("가", "나", "다"), segments=(10.0, 40.0, 30.0)),
            arm("adapter", 55.0, outputs=("라", "마", "바"), segments=(20.0, 20.0, 30.0)),
        )

        self.assertEqual((verdict.wins, verdict.losses, verdict.ties), (1, 1, 1))


class BootstrapTest(unittest.TestCase):
    """The resampling estimate may widen the floor and may never narrow it."""

    def test_an_interval_straddling_zero_forces_indistinguishable(self) -> None:
        base = (10.0, 90.0, 10.0, 90.0, 50.0, 50.0)
        adapter = (90.0, 10.0, 90.0, 10.0, 52.0, 48.0)

        floor = evaluation.bootstrap_floor(base, adapter, delta=2.0, seed=7)

        self.assertGreaterEqual(floor, 2.0)

    def test_it_never_lowers_the_floor_the_other_bases_already_set(self) -> None:
        base = (10.0,) * 12
        adapter = (90.0,) * 12

        with_bootstrap = compare_arms(
            arm("base", 10.0, outputs=("가",) * 12, segments=base),
            arm("adapter", 90.0, outputs=("라",) * 12, segments=adapter),
            bootstrap_seed=7,
        )

        self.assertGreaterEqual(with_bootstrap.floor, evaluation.MIN_REPORTABLE_DELTA_CHRF)
        self.assertEqual(with_bootstrap.verdict, evaluation.IMPROVED)


def sample_run() -> evaluation.EvaluationRun:
    """A complete run, filled in without touching a model."""
    return evaluation.EvaluationRun(
        run_id="20260908T120000Z-abcdef123456",
        created_at="2026-09-08T12:00:00Z",
        test_set={
            "path": "data/splits/test.jsonl",
            "sha256": "a" * 64,
            "n_lines": 4,
            "n_scored": 2,
            "limit": 2,
            "subset": True,
        },
        decoding={"settings": {"max_tokens": 64, "repeats": 2}, "fingerprint": "d" * 64},
        arms={
            evaluation.BASE_ARM: arm("base", 40.0, outputs=("가",), segments=(40.0, 40.0)),
            evaluation.ADAPTER_ARM: arm("adapter", 45.0, outputs=("라",), segments=(45.0, 45.0)),
        },
        comparison=Comparison(
            verdict=evaluation.IMPROVED,
            delta=5.0,
            floor=1.0,
            floor_bases={evaluation.FLOOR_CONVENTION: 1.0},
            distinguishable_fraction=1.0,
            wins=2,
            losses=0,
            ties=0,
        ),
        prompt_shape_fingerprint="p" * 64,
    )


class RunRecordTest(unittest.TestCase):
    """The artifact is complete on read, or it raises."""

    def test_a_record_carries_every_required_key_and_its_metric_block(self) -> None:
        record = sample_run().as_record()

        for key in evaluation.REQUIRED_RECORD_KEYS:
            self.assertIn(key, record)
        self.assertEqual(record["metric"], metrics.metric_parameters())
        self.assertEqual(record["scores"]["n"], 2)
        self.assertEqual(record["scores"][evaluation.BASE_ARM]["n"], 2)

    def test_write_then_read_round_trips_and_writes_exactly_two_files(self) -> None:
        run = sample_run()
        with tempfile.TemporaryDirectory() as raw:
            out_dir = Path(raw) / "eval"

            target = write_run(run, out_dir)

            self.assertEqual(
                sorted(path.name for path in out_dir.iterdir()),
                sorted([f"{run.run_id}.json", evaluation.INDEX_FILENAME]),
            )
            self.assertEqual(read_run(target), run.as_record())
            index = (out_dir / evaluation.INDEX_FILENAME).read_text(encoding="utf-8")
            self.assertEqual(len(index.strip().splitlines()), 1)
            self.assertEqual(json.loads(index)["verdict"], evaluation.IMPROVED)

    def test_a_second_run_appends_one_line_rather_than_replacing_the_index(self) -> None:
        run = sample_run()
        with tempfile.TemporaryDirectory() as raw:
            out_dir = Path(raw) / "eval"
            write_run(run, out_dir)
            write_run(run, out_dir)

            index = (out_dir / evaluation.INDEX_FILENAME).read_text(encoding="utf-8")
            self.assertEqual(len(index.strip().splitlines()), 2)

    def test_a_missing_required_key_raises_instead_of_reading_as_none(self) -> None:
        record = sample_run().as_record()
        del record["scores"]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "broken.json"
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaises(RecordError) as caught:
                read_run(path)
            self.assertIn("scores", str(caught.exception))

    def test_a_verdict_outside_the_closed_set_raises(self) -> None:
        record = sample_run().as_record()
        record["comparison"]["verdict"] = "better"
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "unknown.json"
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaises(RecordError):
                read_run(path)


class RunComparabilityTest(unittest.TestCase):
    """Two runs are comparable only when they measured the same thing the same way."""

    def _pair(self) -> tuple[dict[str, Any], dict[str, Any]]:
        first = sample_run().as_record()
        return first, json.loads(json.dumps(first))

    def test_identical_identity_fields_compare(self) -> None:
        first, second = self._pair()

        self.assertTrue(compare_runs(first, second).comparable)

    def test_a_changed_test_set_is_refused_without_a_delta(self) -> None:
        first, second = self._pair()
        second["test_set"]["sha256"] = "b" * 64

        result = compare_runs(first, second)

        self.assertFalse(result.comparable)
        self.assertIsNone(result.delta)
        self.assertIn("test_set.sha256", result.reasons[0])

    def test_a_different_limit_or_prompt_shape_is_refused(self) -> None:
        for outer, inner, value in (
            ("test_set", "limit", 9),
            ("prompt_shape_fingerprint", None, "z"),
        ):
            with self.subTest(outer):
                first, second = self._pair()
                if inner:
                    second[outer][inner] = value
                else:
                    second[outer] = value

                self.assertFalse(compare_runs(first, second).comparable)


class EvaluatePairTest(unittest.TestCase):
    """The whole path, driven end to end against fakes."""

    def _config(self, directory: Path, **overrides: Any) -> EvaluationConfig:  # noqa: ANN401
        test = write_split(directory / "test.jsonl", PAIRS)
        return EvaluationConfig(
            test_file=test,
            adapter_path=str(directory / "adapter"),
            model_id=MODEL_ID,
            out_dir=directory / "eval",
            **overrides,
        )

    def _run(
        self,
        config: EvaluationConfig,
        base: dict[str, str] | None = None,
        adapter: dict[str, str] | None = None,
        report: Callable[[str], None] | None = None,
    ) -> evaluation.EvaluationRun:
        return evaluate_pair(
            config,
            translator_factory=factory_for(base or {}, adapter or {}),
            clock=lambda: FROZEN,
            report=report or (lambda _: None),
        )

    def test_a_full_run_writes_one_record_and_one_index_line(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = self._config(directory)
            adapter = dict(PAIRS)

            run = self._run(config, adapter=adapter)
            target = write_run(run, config.out_dir)

            self.assertEqual(run.test_set["n_scored"], len(PAIRS))
            self.assertFalse(run.test_set["subset"])
            self.assertEqual(run.arms[evaluation.ADAPTER_ARM].corpus_chrf, 100.0)
            self.assertGreater(run.comparison.delta, 0)
            self.assertEqual(run.comparison.verdict, evaluation.IMPROVED)
            self.assertEqual(read_run(target)["run_id"], run.run_id)

    def test_the_limit_takes_the_first_rows_in_file_order_and_marks_the_subset(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)

            run = self._run(self._config(directory, limit=2))

            self.assertEqual(run.test_set["n_scored"], 2)
            self.assertEqual(run.test_set["n_lines"], len(PAIRS))
            self.assertTrue(run.test_set["subset"])
            self.assertEqual(run.test_set["limit"], 2)

    def test_the_segment_and_arm_counts_are_reported_before_generating(self) -> None:
        announced: list[str] = []
        with tempfile.TemporaryDirectory() as raw:
            self._run(self._config(Path(raw)), report=announced.append)

        self.assertEqual(len(announced), 1)
        self.assertIn(f"{len(PAIRS)} segment", announced[0])
        self.assertIn("2 arm", announced[0])

    def test_both_arms_share_one_decoding_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = self._run(self._config(Path(raw)))

            self.assertEqual(
                run.arms[evaluation.BASE_ARM].decoding_fingerprint,
                run.arms[evaluation.ADAPTER_ARM].decoding_fingerprint,
            )
            self.assertEqual(
                run.decoding["fingerprint"], run.arms[evaluation.BASE_ARM].decoding_fingerprint
            )

    def test_a_run_without_an_adapter_is_refused_rather_than_scored_single_armed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = self._config(directory, limit=None)
            single = EvaluationConfig(
                test_file=config.test_file, model_id=MODEL_ID, out_dir=config.out_dir
            )

            with self.assertRaises(EvaluationError) as caught:
                self._run(single)
            self.assertIn("not a comparison", str(caught.exception))

    def test_a_leaked_split_is_refused_before_anything_generates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = self._config(directory)
            write_split(directory / TRAIN_FILENAME, PAIRS[:1])

            with self.assertRaises(SplitIntegrityError):
                self._run(config)


class ConfoundingTest(unittest.TestCase):
    """A comparison across two different bases is refused, and the override cannot launder it."""

    def _prepared(self, directory: Path, base_model_id: str) -> EvaluationConfig:
        adapter_dir = directory / "adapter"
        write_provenance(
            adapter_dir,
            AdapterProvenance(base_model_id=base_model_id, prompt_shape_fingerprint="p"),
        )
        return EvaluationConfig(
            test_file=write_split(directory / "test.jsonl", PAIRS),
            adapter_path=str(adapter_dir),
            model_id=MODEL_ID,
            out_dir=directory / "eval",
        )

    def _run(self, config: EvaluationConfig) -> evaluation.EvaluationRun:
        return evaluate_pair(
            config,
            translator_factory=factory_for({}, dict(PAIRS)),
            clock=lambda: FROZEN,
            report=lambda _: None,
        )

    def test_a_mismatched_base_raises_before_generating(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ConfoundedComparisonError) as caught:
                self._run(self._prepared(Path(raw), "another/model"))
            self.assertIn("allow-provenance-mismatch", str(caught.exception))

    def test_the_override_runs_but_withholds_the_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = self._prepared(directory, "another/model")

            run = self._run(
                EvaluationConfig(
                    test_file=config.test_file,
                    adapter_path=config.adapter_path,
                    model_id=MODEL_ID,
                    out_dir=config.out_dir,
                    allow_provenance_mismatch=True,
                )
            )

            self.assertTrue(run.confounded)
            self.assertEqual(run.comparison.verdict, evaluation.UNAVAILABLE)
            self.assertTrue(run.as_record()["confounded"])

    def test_a_matching_base_is_not_confounded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = self._run(self._prepared(Path(raw), MODEL_ID))

            self.assertFalse(run.confounded)
            self.assertNotEqual(run.comparison.verdict, evaluation.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
