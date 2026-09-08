# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Encoding policy, the output-path guards, and part-file durability."""

import tempfile
import unittest
from pathlib import Path

from perevod.files import (
    PART_SUFFIX,
    DestinationError,
    IncrementalWriter,
    SourceReadError,
    check_destination,
    derive_output_path,
    plan_batch,
    read_source,
    write_output,
)

RUSSIAN = "Первый абзац.\n\nВторой абзац.\n"


class TempDirTest(unittest.TestCase):
    """Nothing here writes outside a temporary directory."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)


class EncodingPolicyTest(TempDirTest):
    """Loud beats mojibake."""

    def test_a_byte_order_mark_never_reaches_the_text(self) -> None:
        path = self.root / "bom.txt"
        path.write_bytes(b"\xef\xbb\xbf" + RUSSIAN.encode())
        self.assertEqual(read_source(path), RUSSIAN)

    def test_a_named_codec_is_honoured(self) -> None:
        path = self.root / "cp1251.txt"
        path.write_bytes(RUSSIAN.encode("cp1251"))
        self.assertEqual(read_source(path, encoding="cp1251"), RUSSIAN)

    def test_an_undecodable_file_names_the_flag_that_fixes_it(self) -> None:
        path = self.root / "wrong.txt"
        path.write_bytes(RUSSIAN.encode("cp1251"))
        with self.assertRaises(SourceReadError) as caught:
            read_source(path)
        self.assertIn("--encoding", str(caught.exception))

    def test_carriage_returns_are_normalised_on_read(self) -> None:
        path = self.root / "crlf.txt"
        path.write_bytes("Первый.\r\n\r\nВторой.\r\n".encode())
        self.assertEqual(read_source(path), "Первый.\n\nВторой.\n")

    def test_output_is_written_utf8_with_line_feeds(self) -> None:
        source = self.root / "in.txt"
        source.write_text(RUSSIAN, encoding="utf-8")
        destination = self.root / "out.txt"

        write_output(destination, RUSSIAN, force=False, source=source)

        self.assertEqual(destination.read_bytes(), RUSSIAN.encode())


class OutputGuardTest(TempDirTest):
    """Three refusals, each before any model call."""

    def test_writing_over_the_source_is_refused(self) -> None:
        source = self.root / "doc.txt"
        source.write_text(RUSSIAN, encoding="utf-8")
        with self.assertRaises(DestinationError):
            check_destination(source, source, force=True)

    def test_an_existing_destination_needs_force(self) -> None:
        source = self.root / "doc.txt"
        source.write_text(RUSSIAN, encoding="utf-8")
        destination = self.root / "doc.ko.txt"
        destination.write_text("уже здесь", encoding="utf-8")

        with self.assertRaises(DestinationError):
            check_destination(destination, source, force=False)
        check_destination(destination, source, force=True)

    def test_colliding_batch_names_are_refused_before_anything_is_written(self) -> None:
        first = self.root / "a" / "doc.txt"
        second = self.root / "b" / "doc.txt"
        for path in (first, second):
            path.parent.mkdir(parents=True)
            path.write_text(RUSSIAN, encoding="utf-8")
        out = self.root / "out"

        with self.assertRaises(DestinationError):
            plan_batch([first, second], out, force=False)
        self.assertFalse(out.exists())

    def test_output_names_keep_the_final_suffix(self) -> None:
        cases = {"doc.txt": "doc.ko.txt", "doc": "doc.ko", "a.tar.gz": "a.tar.ko.gz"}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(derive_output_path(Path(name), self.root).name, expected)


class DurabilityTest(TempDirTest):
    """A crash must cost the failing chunk, not the run."""

    def test_a_crash_leaves_completed_work_in_the_part_file(self) -> None:
        destination = self.root / "out.txt"
        writer = IncrementalWriter(destination)
        writer.write("готово\n")

        with self.assertRaises(RuntimeError), writer:
            msg = "backend died"
            raise RuntimeError(msg)

        self.assertFalse(destination.exists())
        self.assertEqual(writer.part_path.read_text(encoding="utf-8"), "готово\n")

    def test_a_clean_run_leaves_no_part_file_behind(self) -> None:
        destination = self.root / "out.txt"
        with IncrementalWriter(destination) as writer:
            writer.write(RUSSIAN)

        self.assertEqual(destination.read_text(encoding="utf-8"), RUSSIAN)
        self.assertFalse(destination.with_name(destination.name + PART_SUFFIX).exists())


if __name__ == "__main__":
    unittest.main()
