# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""The provenance record, and the five states a base/adapter pairing can be in."""

import json
import tempfile
import unittest
from pathlib import Path

from perevod.adapters import (
    PROVENANCE_FILENAME,
    AdapterMismatchError,
    AdapterProvenance,
    AdapterProvenanceError,
    check_adapter_compatibility,
    provenance_path,
    read_provenance,
    write_provenance,
)
from perevod.config import DEFAULT_MODEL_ID

BASE = DEFAULT_MODEL_ID
OTHER_BASE = "mlx-community/gemma-4-31b-it-4bit"
FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64

RECORD = AdapterProvenance(
    base_model_id=BASE,
    prompt_shape_fingerprint=FINGERPRINT,
    created_at="2026-09-08",
    mlx_lm_version="0.31.3",
)


class ProvenanceFileTest(unittest.TestCase):
    """What is written, and what is refused on the way back in."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())

    def test_the_filename_cannot_collide_with_the_trainers_own_output(self) -> None:
        self.assertNotEqual(PROVENANCE_FILENAME, "adapter_config.json")
        self.assertTrue(PROVENANCE_FILENAME.startswith("perevod"))

    def test_a_record_round_trips_unchanged(self) -> None:
        write_provenance(self.directory, RECORD)

        self.assertEqual(read_provenance(self.directory), RECORD)

    def test_an_adapter_without_a_record_reads_as_none(self) -> None:
        self.assertIsNone(read_provenance(self.directory))

    def test_a_missing_required_key_is_refused(self) -> None:
        for dropped in ("schema_version", "base_model_id", "prompt_shape_fingerprint"):
            with self.subTest(key=dropped):
                payload = {
                    "schema_version": 1,
                    "base_model_id": BASE,
                    "prompt_shape_fingerprint": FINGERPRINT,
                }
                del payload[dropped]
                provenance_path(self.directory).write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaises(AdapterProvenanceError) as caught:
                    read_provenance(self.directory)
                self.assertIn(dropped, str(caught.exception))

    def test_a_schema_version_this_release_does_not_know_is_refused(self) -> None:
        write_provenance(self.directory, RECORD)
        path = provenance_path(self.directory)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaises(AdapterProvenanceError):
            read_provenance(self.directory)


class CompatibilityTest(unittest.TestCase):
    """One row per state; absent and malformed must not collapse together."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.warnings: list[str] = []

    def _check(self, **overrides: object) -> object:
        arguments: dict[str, object] = {
            "base_model_id": BASE,
            "prompt_fingerprint": FINGERPRINT,
            "warn": self.warnings.append,
        }
        arguments.update(overrides)
        return check_adapter_compatibility(self.directory, **arguments)  # type: ignore[arg-type]

    def test_an_absent_record_warns_and_proceeds(self) -> None:
        self.assertIsNone(self._check())

        self.assertEqual(len(self.warnings), 1)
        self.assertIn(PROVENANCE_FILENAME, self.warnings[0])

    def test_a_full_match_is_silent(self) -> None:
        write_provenance(self.directory, RECORD)

        self.assertEqual(self._check(), RECORD)
        self.assertEqual(self.warnings, [])

    def test_another_base_is_refused_and_both_ids_are_named(self) -> None:
        write_provenance(self.directory, RECORD)

        with self.assertRaises(AdapterMismatchError) as caught:
            self._check(base_model_id=OTHER_BASE)
        self.assertIn(BASE, str(caught.exception))
        self.assertIn(OTHER_BASE, str(caught.exception))

    def test_another_prompt_shape_is_refused_and_both_digests_are_named(self) -> None:
        write_provenance(self.directory, RECORD)

        with self.assertRaises(AdapterMismatchError) as caught:
            self._check(prompt_fingerprint=OTHER_FINGERPRINT)
        self.assertIn(FINGERPRINT, str(caught.exception))
        self.assertIn(OTHER_FINGERPRINT, str(caught.exception))

    def test_a_malformed_record_is_a_different_failure_from_an_absent_one(self) -> None:
        provenance_path(self.directory).write_text("{not json", encoding="utf-8")

        with self.assertRaises(AdapterProvenanceError) as caught:
            self._check()
        self.assertNotIsInstance(caught.exception, AdapterMismatchError)
        self.assertEqual(self.warnings, [])

    def test_the_override_downgrades_both_mismatches_to_a_warning(self) -> None:
        write_provenance(self.directory, RECORD)

        for overrides in ({"base_model_id": OTHER_BASE}, {"prompt_fingerprint": OTHER_FINGERPRINT}):
            with self.subTest(**overrides):
                self.warnings.clear()
                self._check(allow_mismatch=True, **overrides)

                self.assertEqual(len(self.warnings), 1)

    def test_a_malformed_record_stays_refused_even_with_the_override(self) -> None:
        provenance_path(self.directory).write_text("{not json", encoding="utf-8")

        with self.assertRaises(AdapterProvenanceError):
            self._check(allow_mismatch=True)

    def test_a_path_that_is_not_an_adapter_directory_is_refused(self) -> None:
        with self.assertRaises(AdapterProvenanceError):
            check_adapter_compatibility(
                self.directory / "absent",
                base_model_id=BASE,
                prompt_fingerprint=FINGERPRINT,
                warn=self.warnings.append,
            )


if __name__ == "__main__":
    unittest.main()
