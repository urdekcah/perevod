# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Reading source documents and writing translations without destroying anything."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

DEFAULT_ENCODING = "utf-8-sig"
PART_SUFFIX = ".part"

# doc.txt -> doc.ko.txt, doc -> doc.ko, a.tar.gz -> a.tar.ko.gz
OUTPUT_INFIX = ".ko"


class FileError(Exception):
    """Base for this layer's failures."""


class SourceReadError(FileError):
    """A source file is unreadable under the selected codec."""


class DestinationError(FileError):
    """An output path was refused because writing it would lose data."""


def read_source(path: Path, *, encoding: str = DEFAULT_ENCODING) -> str:
    """Read a document, normalising CRLF and CR to LF.

    The default codec strips a BOM that would otherwise ride into the first prompt. Never
    `errors="replace"`: mojibake in a translation is worse than a refusal.

    Raises:
        SourceReadError: The file is missing, unreadable, or not valid under `encoding`.
    """
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError as error:
        msg = f"{path} is not valid {encoding}; pass the right codec with --encoding"
        raise SourceReadError(msg) from error
    except (OSError, LookupError) as error:
        msg = f"cannot read {path}: {error}"
        raise SourceReadError(msg) from error


def derive_output_path(source: Path, output_dir: Path) -> Path:
    """Name the translation of `source` inside `output_dir`."""
    return output_dir / f"{source.stem}{OUTPUT_INFIX}{source.suffix}"


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def check_destination(destination: Path, source: Path, *, force: bool) -> None:
    """Refuse an output path that would destroy something.

    Raises:
        DestinationError: The destination is the source, or already exists without `force`.
    """
    if _same_file(destination, source):
        msg = f"refusing to write the translation over its own source: {destination}"
        raise DestinationError(msg)
    if destination.exists() and not force:
        msg = f"{destination} already exists; pass --force to overwrite it"
        raise DestinationError(msg)


def plan_batch(sources: list[Path], output_dir: Path, *, force: bool) -> list[tuple[Path, Path]]:
    """Pair every input with its output, refusing the whole batch before any model call.

    `output_dir` is created only once every pair has passed.

    Raises:
        DestinationError: Two inputs collide on one name, or a destination is unsafe.
    """
    pairs: list[tuple[Path, Path]] = []
    claimed: dict[Path, Path] = {}
    for source in sources:
        destination = derive_output_path(source, output_dir)
        key = destination.absolute()
        if key in claimed:
            msg = f"{source} and {claimed[key]} would both be written to {destination}"
            raise DestinationError(msg)
        check_destination(destination, source, force=force)
        claimed[key] = source
        pairs.append((source, destination))
    output_dir.mkdir(parents=True, exist_ok=True)
    return pairs


@dataclass
class IncrementalWriter:
    """Writes to `<destination>.part` and renames only once the run finishes.

    So a destination never exists half written; a crash leaves completed chunks behind.
    """

    destination: Path

    def __post_init__(self) -> None:
        """Opened up front, so even a failure before the first write leaves a part file."""
        self.part_path = self.destination.with_name(self.destination.name + PART_SUFFIX)
        self._handle = self.part_path.open("w", encoding="utf-8", newline="\n")

    def write(self, fragment: str) -> None:
        """Flushed per fragment, so a crash keeps what finished."""
        self._handle.write(fragment)
        self._handle.flush()

    def commit(self) -> None:
        """The rename is atomic: no half-written destination ever exists."""
        self._handle.close()
        self.part_path.replace(self.destination)

    def abandon(self) -> None:
        """Close, leaving the part file for the user."""
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> Self:
        """The part file is open from construction."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Commits only on a clean exit."""
        if exc_type is None:
            self.commit()
        else:
            self.abandon()


def write_output(destination: Path, text: str, *, force: bool, source: Path) -> None:
    """Write a whole translation at once, through the same part-then-rename discipline.

    Raises:
        DestinationError: The destination is unsafe.
    """
    check_destination(destination, source, force=force)
    with IncrementalWriter(destination) as writer:
        writer.write(text)
