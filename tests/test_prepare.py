# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""The preparation pipeline, end to end, on a bare interpreter."""

import json
import shutil
import tempfile
import unicodedata
import unittest
from pathlib import Path
from typing import Any

from perevod.dataset import (
    MANIFEST_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALID_FILENAME,
    parse_training_record,
    prompt_shape_fingerprint,
)
from perevod.prepare import (
    InputError,
    OutputExistsError,
    OutputPathError,
    PrepareConfig,
    PrepareError,
    PrepareResult,
    prepare_splits,
)


def sample_pairs(count: int) -> list[tuple[str, str]]:
    """`count` numbered pairs, distinct on both sides so any row stays traceable."""
    return [(f"Предложение номер {index}.", f"문장 번호 {index}.") for index in range(count)]


class PipelineTestCase(unittest.TestCase):
    """Base for the pipeline suites; each test gets its own corpus directory."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.inbox = self.root / "in"
        self.inbox.mkdir()

    def write(self, lines: list[str], name: str = "batch-01.txt", encoding: str = "utf-8") -> Path:
        path = self.inbox / name
        path.write_text("\n".join(lines) + "\n", encoding=encoding)
        return path

    def write_pairs(self, pairs: list[tuple[str, str]], name: str = "batch-01.txt") -> Path:
        return self.write([f"{ru} ||| {ko}" for ru, ko in pairs], name=name)

    def run_pipeline(self, out: str | Path = "out", **overrides: object) -> PrepareResult:
        options: dict[str, Any] = {"inputs": (self.inbox,), "output_dir": self.root / out}
        options.update(overrides)
        return prepare_splits(PrepareConfig(**options))

    def rows(self, directory: Path | str, name: str) -> list[Any]:
        path = Path(directory) / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ParsingTests(PipelineTestCase):
    """Which rows a corpus file yields, and how the rest are reported."""

    def test_every_malformed_shape_is_named_by_file_and_editor_line_number(self) -> None:
        self.write(
            [
                "# batch header",
                "",
                "Здравствуйте ||| 안녕하세요",
                "нет разделителя",
                "а ||| б ||| в",
                "   ||| 없음",
                "Пусто |||   ",
            ]
        )

        with self.assertRaises(InputError) as caught:
            self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)

        problems = caught.exception.problems
        self.assertEqual(len(problems), 4)
        self.assertFalse((self.root / "out").exists())
        for line, fragment in ((4, "found 0"), (5, "found 2"), (6, "Russian"), (7, "Korean")):
            with self.subTest(line=line):
                self.assertTrue(
                    any(f":{line}:" in problem and fragment in problem for problem in problems),
                    problems,
                )

    def test_comments_blank_lines_a_bom_and_crlf_all_survive(self) -> None:
        path = self.inbox / "batch-01.txt"
        path.write_bytes("﻿# заголовок\r\n\r\nЗдравствуйте ||| 안녕하세요\r\n".encode())

        result = self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)

        self.assertEqual(
            self.rows(result.output_dir, TRAIN_FILENAME)[0]["messages"][1]["content"],
            "Здравствуйте",
        )

    def test_a_file_that_is_not_utf8_is_named_rather_than_crashing(self) -> None:
        (self.inbox / "batch-01.txt").write_bytes(b"\xff\xfe\x00 ||| \x00")

        with self.assertRaises(InputError) as caught:
            self.run_pipeline()

        self.assertIn("UTF-8", caught.exception.problems[0])

    def test_an_input_set_with_no_pairs_is_an_error(self) -> None:
        self.write(["# only a comment"])

        with self.assertRaises(InputError) as caught:
            self.run_pipeline()

        self.assertIn("no pairs", str(caught.exception))

    def test_a_non_txt_file_in_a_scanned_directory_is_reported_not_ignored(self) -> None:
        self.write_pairs(sample_pairs(1))
        (self.inbox / "notes.md").write_text("не пара", encoding="utf-8")

        result = self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)

        self.assertTrue(any("notes.md" in note for note in result.skipped), result.skipped)


class NormalizationTests(PipelineTestCase):
    """What is folded before comparison, and what survives into the payload."""

    def test_decomposed_and_precomposed_hangul_are_the_same_pair(self) -> None:
        target = "안녕하세요"
        self.write([f"Привет ||| {target}", f"Привет ||| {unicodedata.normalize('NFD', target)}"])

        result = self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)

        self.assertEqual(result.manifest["counts"]["duplicates_dropped"], 1)

    def test_the_payload_keeps_the_author_punctuation_and_spacing_verbatim(self) -> None:
        source = "«Цитата»  —  тире"
        self.write([f'{source} ||| "인용"  —  대시'])

        result = self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)

        emitted = parse_training_record(self.rows(result.output_dir, TRAIN_FILENAME)[0])[0]
        self.assertEqual(emitted, source)
        self.assertEqual(unicodedata.normalize("NFC", emitted), emitted)

    def test_russian_case_is_meaningful_so_it_is_never_folded(self) -> None:
        self.write(["Мир ||| 세계", "мир ||| 평화"])

        result = self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)

        self.assertEqual(result.manifest["counts"]["unique_sources"], 2)


class DeduplicationTests(PipelineTestCase):
    """Repeats, conflicting translations, and the split boundary they must not cross."""

    def test_a_repeated_pair_is_dropped_and_both_locations_are_reported(self) -> None:
        self.write(["Привет ||| 안녕", "Привет ||| 안녕"])

        result = self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)

        self.assertEqual(len(result.duplicates), 1)
        self.assertIn(":2:", result.duplicates[0])
        self.assertIn(":1,", result.duplicates[0])

    def test_one_source_with_two_translations_stops_the_run_by_default(self) -> None:
        self.write(["Привет ||| 안녕", "Привет ||| 반갑습니다"])

        with self.assertRaises(InputError) as caught:
            self.run_pipeline()

        self.assertIn("--allow-source-conflicts", caught.exception.problems[0])

    def test_the_opt_in_keeps_conflicting_rows_together_in_one_split(self) -> None:
        pairs = sample_pairs(20)
        self.write_pairs([*pairs, (pairs[0][0], "다른 번역")])

        result = self.run_pipeline(allow_source_conflicts=True)

        located = {
            name: {parse_training_record(row)[0] for row in self.rows(result.output_dir, name)}
            for name in (TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME)
        }
        self.assertEqual(sum(pairs[0][0] in sources for sources in located.values()), 1)
        self.assertEqual(len(result.conflicts), 1)
        self.assertIn("kept", result.conflicts[0])

    def test_no_source_ever_straddles_a_split_boundary(self) -> None:
        self.write_pairs(sample_pairs(40))

        for seed in (0, 7, 99):
            with self.subTest(seed=seed):
                result = self.run_pipeline(out=f"out-{seed}", seed=seed)
                sources = [
                    {parse_training_record(row)[0] for row in self.rows(result.output_dir, name)}
                    for name in (TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME)
                ]
                self.assertFalse(sources[0] & sources[1])
                self.assertFalse(sources[0] & sources[2])
                self.assertFalse(sources[1] & sources[2])

    def test_the_same_target_from_two_sources_is_ordinary_data(self) -> None:
        self.write(["Привет ||| 안녕", "Здравствуйте ||| 안녕"])

        result = self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)

        self.assertEqual(result.manifest["counts"]["unique_sources"], 2)


class AlignmentGuardTests(PipelineTestCase):
    """The length checks that catch a misaligned pair before it reaches training."""

    def test_a_wildly_uneven_pair_is_rejected_with_both_counts_and_the_option(self) -> None:
        self.write([f"{'Слово ' * 40}||| 짧다"])

        with self.assertRaises(InputError) as caught:
            self.run_pipeline()

        problem = caught.exception.problems[0]
        for fragment in ("length ratio", "characters", "--max-length-ratio", "8.00"):
            self.assertIn(fragment, problem)

    def test_the_bound_can_be_raised_or_disabled(self) -> None:
        self.write([f"{'Слово ' * 40}||| 짧다"])

        for bound in (0.0, 200.0):
            with self.subTest(bound=bound):
                result = self.run_pipeline(
                    out=f"out-{bound}",
                    max_length_ratio=bound,
                    valid_fraction=0.0,
                    test_fraction=0.0,
                )
                self.assertEqual(result.manifest["counts"]["pairs_read"], 1)

    def test_the_optional_character_cap_is_off_until_it_is_set(self) -> None:
        self.write(["Очень длинное предложение ||| 아주 긴 문장입니다"])

        self.run_pipeline(valid_fraction=0.0, test_fraction=0.0)
        with self.assertRaises(InputError) as caught:
            self.run_pipeline(out="capped", max_chars=5, valid_fraction=0.0, test_fraction=0.0)

        self.assertIn("--max-chars", caught.exception.problems[0])


class DeterminismTests(PipelineTestCase):
    """Same inputs, same bytes -- whatever order they arrive in."""

    def digests(self, directory: Path | str) -> dict[str, bytes]:
        return {
            name: (Path(directory) / name).read_bytes()
            for name in (TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME)
        }

    def test_two_builds_from_the_same_inputs_are_byte_identical(self) -> None:
        self.write_pairs(sample_pairs(30))

        first = self.run_pipeline(out="a")
        second = self.run_pipeline(out="b")

        self.assertEqual(self.digests(first.output_dir), self.digests(second.output_dir))
        self.assertEqual(
            (Path(first.output_dir) / MANIFEST_FILENAME).read_bytes(),
            (Path(second.output_dir) / MANIFEST_FILENAME).read_bytes(),
        )

    def test_reordering_the_input_leaves_the_splits_untouched(self) -> None:
        pairs = sample_pairs(30)
        self.write_pairs(pairs)
        baseline = self.digests(self.run_pipeline(out="a").output_dir)

        self.write_pairs(list(reversed(pairs)))
        reordered = self.digests(self.run_pipeline(out="b").output_dir)

        self.assertEqual(baseline, reordered)

    def test_two_input_files_give_the_same_splits_in_either_order(self) -> None:
        first = self.write_pairs(sample_pairs(30)[:15], name="a.txt")
        second = self.write_pairs(sample_pairs(30)[15:], name="b.txt")

        forward = prepare_splits(PrepareConfig(inputs=(first, second), output_dir=self.root / "f"))
        backward = prepare_splits(PrepareConfig(inputs=(second, first), output_dir=self.root / "b"))

        self.assertEqual(self.digests(forward.output_dir), self.digests(backward.output_dir))

    def test_changing_the_seed_moves_at_least_one_row(self) -> None:
        self.write_pairs(sample_pairs(30))

        self.assertNotEqual(
            self.digests(self.run_pipeline(out="a", seed=0).output_dir),
            self.digests(self.run_pipeline(out="b", seed=1).output_dir),
        )


class OutputTests(PipelineTestCase):
    """Which files a build writes, and what the manifest records about it."""

    def test_a_split_asked_for_but_unrealizable_is_an_error_not_a_missing_file(self) -> None:
        self.write_pairs(sample_pairs(4))

        with self.assertRaises(PrepareError) as caught:
            self.run_pipeline(valid_fraction=0.1, test_fraction=0.0)

        self.assertIn("at least 10 pairs", str(caught.exception))

    def test_the_unrealizable_split_is_named_even_when_it_is_the_only_one_asked_for(self) -> None:
        self.write_pairs(sample_pairs(10))

        with self.assertRaises(PrepareError) as caught:
            self.run_pipeline(valid_fraction=0.0, test_fraction=0.05)

        message = str(caught.exception)
        self.assertIn(TEST_FILENAME, message)
        self.assertIn("0.05", message)
        self.assertIn("at least 20 pairs", message)

    def test_a_zero_fraction_produces_no_file_for_that_split(self) -> None:
        self.write_pairs(sample_pairs(20))

        result = self.run_pipeline(test_fraction=0.0)

        self.assertNotIn(TEST_FILENAME, result.written)
        self.assertFalse((Path(result.output_dir) / TEST_FILENAME).exists())

    def test_the_manifest_records_the_build_without_volatile_content(self) -> None:
        self.write_pairs(sample_pairs(20))

        manifest = self.run_pipeline().manifest

        self.assertEqual(manifest["prompt_shape_fingerprint"], prompt_shape_fingerprint())
        self.assertEqual(manifest["counts"]["pairs_read"], 20)
        self.assertEqual(
            sorted(entry["name"] for entry in manifest["outputs"]),
            sorted([TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME]),
        )
        self.assertEqual(manifest["inputs"][0]["lines_read"], 20)
        serialized = json.dumps(manifest)
        self.assertNotIn("timestamp", serialized)

    def test_fractions_are_range_checked(self) -> None:
        self.write_pairs(sample_pairs(20))

        for valid, test in ((0.6, 0.5), (-0.1, 0.1), (1.0, 0.0)):
            with self.subTest(valid=valid, test=test), self.assertRaises(PrepareError):
                self.run_pipeline(valid_fraction=valid, test_fraction=test)


class RerunPolicyTests(PipelineTestCase):
    """What a second build is allowed to do to the first one's output."""

    def setUp(self) -> None:
        super().setUp()
        self.write_pairs(sample_pairs(20))
        self.first = self.run_pipeline()

    def snapshot(self, directory: Path | str) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in Path(directory).iterdir()}

    def test_a_non_empty_directory_stops_the_run_before_anything_is_touched(self) -> None:
        before = self.snapshot(self.first.output_dir)

        with self.assertRaises(OutputExistsError) as caught:
            self.run_pipeline()

        self.assertEqual(before, self.snapshot(self.first.output_dir))
        message = str(caught.exception)
        self.assertIn(TRAIN_FILENAME, message)
        self.assertIn("adapter", message)
        self.assertIn(self.first.manifest["prompt_shape_fingerprint"], message)
        self.assertIn("--overwrite", message)

    def test_overwrite_replaces_the_owned_files(self) -> None:
        result = self.run_pipeline(overwrite=True, seed=5)

        self.assertNotEqual(result.manifest["outputs"], self.first.manifest["outputs"])

    def test_overwrite_removes_an_owned_split_the_new_build_does_not_produce(self) -> None:
        self.run_pipeline(overwrite=True, test_fraction=0.0)

        self.assertFalse((Path(self.first.output_dir) / TEST_FILENAME).exists())

    def test_overwrite_never_touches_a_file_this_command_does_not_own(self) -> None:
        stranger = Path(self.first.output_dir) / "notes-from-the-author.txt"
        stranger.write_text("не трогать", encoding="utf-8")

        self.run_pipeline(overwrite=True)

        self.assertEqual(stranger.read_text(encoding="utf-8"), "не трогать")


class ContainmentTests(PipelineTestCase):
    """Where the output directory is allowed to resolve to."""

    def test_a_path_with_parent_segments_writes_only_inside_the_resolved_target(self) -> None:
        self.write_pairs(sample_pairs(20))
        (self.root / "sibling").mkdir()

        result = self.run_pipeline(out=Path("sibling/../target"))

        self.assertEqual(result.output_dir, (self.root / "target").resolve())
        self.assertEqual(list((self.root / "sibling").iterdir()), [])

    def test_a_symlinked_output_directory_resolves_to_its_target(self) -> None:
        self.write_pairs(sample_pairs(20))
        real = self.root / "real"
        real.mkdir()
        link = self.root / "link"
        link.symlink_to(real)

        result = self.run_pipeline(out="link")

        self.assertEqual(result.output_dir, real.resolve())

    def test_writing_into_the_hand_authored_corpus_is_refused(self) -> None:
        self.write_pairs(sample_pairs(20))
        protected = Path("data/derived")
        before = sorted(path.name for path in protected.iterdir())

        for target in (protected, protected / "nested", Path("data/raw"), Path("data/adapters")):
            with self.subTest(target=target), self.assertRaises(OutputPathError):
                self.run_pipeline(output_dir=target)

        self.assertEqual(sorted(path.name for path in protected.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
