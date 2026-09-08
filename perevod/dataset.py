# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""The training-record shape and the split file names, shared with the trainer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from perevod.translator import SHAPE_SENTINEL, build_messages, reference_messages

if TYPE_CHECKING:
    from collections.abc import Mapping

# The names mlx_lm.lora looks up.
TRAIN_FILENAME = "train.jsonl"
VALID_FILENAME = "valid.jsonl"
TEST_FILENAME = "test.jsonl"
MANIFEST_FILENAME = "manifest.json"

DEFAULT_SPLIT_DIR = Path("data/splits")

MESSAGES_KEY = "messages"
ROLE_KEY = "role"
CONTENT_KEY = "content"
ASSISTANT_ROLE = "assistant"

# The three row shapes mlx_lm.lora reads, and the roles it accepts in a chat row.
CHAT_SCHEMA = "chat"
COMPLETIONS_SCHEMA = "completions"
TEXT_SCHEMA = "text"
CHAT_ROLES = ("system", "user", ASSISTANT_ROLE)

PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"
TEXT_KEY = "text"

_REGENERATE_HINT = "regenerate the splits with `perevod prepare-data`"


class DatasetError(Exception):
    """Base for every typed error raised over dataset files."""


class RecordShapeError(DatasetError):
    """A record disagrees with the shape the shipped prompt builder produces."""


class SchemaError(DatasetError):
    """A line is not one of the row shapes the trainer reads."""


class SplitLayoutError(DatasetError):
    """The data directory does not hold the files the trainer requires."""


@dataclass(frozen=True)
class SplitReport:
    """What a data directory holds, once every present file has been accepted."""

    data_dir: Path
    rows: Mapping[str, int]
    notices: tuple[str, ...]


def _affixes() -> tuple[int, str, str]:
    """Which message carries the source, and the constant text either side of it."""
    messages = reference_messages()
    carriers = [i for i, message in enumerate(messages) if SHAPE_SENTINEL in message[CONTENT_KEY]]
    if len(carriers) != 1:
        msg = (
            f"the prompt builder put the source into {len(carriers)} messages; "
            "exactly one is required"
        )
        raise RecordShapeError(msg)

    index = carriers[0]
    prefix, suffix = messages[index][CONTENT_KEY].split(SHAPE_SENTINEL)
    return index, prefix, suffix


def build_training_record(source: str, target: str) -> dict[str, Any]:
    """One pair as an mlx_lm `chat` record; the prompt half comes from the shipped builder.

    Raises:
        RecordShapeError: `source` holds the sentinel, so it could not be parsed back.
    """
    if SHAPE_SENTINEL in source:
        msg = f"source text contains the reserved sentinel {SHAPE_SENTINEL!r}"
        raise RecordShapeError(msg)

    messages = build_messages(source)
    messages.append({ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: target})
    return {MESSAGES_KEY: messages}


def parse_training_record(record: object) -> tuple[str, str]:
    """The exact inverse of `build_training_record`.

    Raises:
        RecordShapeError: The role sequence or scaffolding has drifted.
    """
    index, prefix, suffix = _affixes()
    reference = reference_messages()

    if not isinstance(record, dict) or not isinstance(record.get(MESSAGES_KEY), list):
        msg = f"record is not an object carrying a {MESSAGES_KEY!r} list"
        raise RecordShapeError(msg)

    messages = record[MESSAGES_KEY]
    if len(messages) != len(reference) + 1:
        msg = f"expected {len(reference) + 1} messages, found {len(messages)}"
        raise RecordShapeError(msg)

    for position, expected in enumerate(reference):
        if position != index and messages[position] != expected:
            msg = f"message {position} is {messages[position]!r}, expected {expected!r}"
            raise RecordShapeError(msg)

    carrier = messages[index]
    content = carrier.get(CONTENT_KEY, "")
    if carrier.get(ROLE_KEY) != reference[index][ROLE_KEY]:
        msg = (
            f"message {index} has role {carrier.get(ROLE_KEY)!r}, "
            f"expected {reference[index][ROLE_KEY]!r}"
        )
        raise RecordShapeError(msg)
    if not content.startswith(prefix) or not content.endswith(suffix):
        msg = (
            f"message {index} is not wrapped in the expected "
            f"prefix {prefix!r} and suffix {suffix!r}"
        )
        raise RecordShapeError(msg)

    answer = messages[-1]
    if answer.get(ROLE_KEY) != ASSISTANT_ROLE:
        msg = f"the last message has role {answer.get(ROLE_KEY)!r}, expected {ASSISTANT_ROLE!r}"
        raise RecordShapeError(msg)

    source: str = content[len(prefix) : len(content) - len(suffix)]
    target: str = answer.get(CONTENT_KEY, "")
    return source, target


def _check_chat_messages(messages: object) -> None:
    if not isinstance(messages, list) or not messages:
        msg = f"{MESSAGES_KEY!r} is not a non-empty list"
        raise SchemaError(msg)

    for position, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {ROLE_KEY, CONTENT_KEY}:
            msg = f"message {position} does not carry exactly {ROLE_KEY!r} and {CONTENT_KEY!r}"
            raise SchemaError(msg)
        if not all(isinstance(field, str) for field in message.values()):
            msg = f"message {position} has a non-string {ROLE_KEY} or {CONTENT_KEY}"
            raise SchemaError(msg)
        if message[ROLE_KEY] not in CHAT_ROLES:
            msg = f"message {position} uses role {message[ROLE_KEY]!r}, outside {CHAT_ROLES}"
            raise SchemaError(msg)


def classify_record(value: object) -> str:
    """Which of the trainer's three schemas one decoded line matches."""
    if not isinstance(value, dict):
        msg = f"expected a JSON object, found {type(value).__name__}"
        raise SchemaError(msg)

    if MESSAGES_KEY in value:
        _check_chat_messages(value[MESSAGES_KEY])
        return CHAT_SCHEMA
    if isinstance(value.get(PROMPT_KEY), str) and isinstance(value.get(COMPLETION_KEY), str):
        return COMPLETIONS_SCHEMA
    if isinstance(value.get(TEXT_KEY), str):
        return TEXT_SCHEMA

    msg = f"object matches none of the {CHAT_SCHEMA}, {COMPLETIONS_SCHEMA} or {TEXT_SCHEMA} schemas"
    raise SchemaError(msg)


def _at(path: Path, line: int, detail: str) -> str:
    return f"{path}:{line}: {detail}"


def _load(path: Path) -> list[tuple[int, object]]:
    if not path.is_file():
        msg = f"{path}: no such file"
        raise SchemaError(msg)

    decoded: list[tuple[int, object]] = []
    # Blank lines are the trailing newline every editor writes, not a malformed record.
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded.append((number, json.loads(line)))
        except json.JSONDecodeError as error:
            raise SchemaError(_at(path, number, f"not valid JSON: {error.msg}")) from error

    if not decoded:
        msg = f"{path}: holds no records"
        raise SchemaError(msg)
    return decoded


def _schema_of(path: Path, decoded: list[tuple[int, object]]) -> str:
    seen: dict[str, int] = {}
    for number, value in decoded:
        try:
            schema = classify_record(value)
        except SchemaError as error:
            raise SchemaError(_at(path, number, str(error))) from error
        seen.setdefault(schema, number)

    if len(seen) > 1:
        (first, first_line), (second, second_line) = sorted(seen.items(), key=lambda i: i[1])[:2]
        msg = _at(path, second_line, f"{second} row in a {first} file (line {first_line})")
        raise SchemaError(msg)
    return next(iter(seen))


def validate_schema(path: Path) -> str:
    """The single schema every record in `path` shares.

    Raises:
        SchemaError: The file is unreadable, empty, or not one uniform schema.
    """
    return _schema_of(path, _load(path))


def validate_training_file(path: Path) -> int:
    """Accepted record count, once every row round-trips through the shipped prompt shape.

    Raises:
        SchemaError: Not a `chat` file this project would train on.
        RecordShapeError: A row was built against a different prompt shape.
    """
    decoded = _load(path)
    schema = _schema_of(path, decoded)
    if schema != CHAT_SCHEMA:
        msg = (
            f"{path}: valid {schema} data, but this project trains on {CHAT_SCHEMA} rows carrying "
            f"its own prompt shape; {_REGENERATE_HINT}"
        )
        raise SchemaError(msg)

    for number, value in decoded:
        try:
            parse_training_record(value)
        except RecordShapeError as error:
            detail = (
                f"{error}. Training on this would fit the adapter to a prompt the model "
                f"never sees at translation time; {_REGENERATE_HINT}"
            )
            raise RecordShapeError(_at(path, number, detail)) from error
    return len(decoded)


def validate_split_dir(data_dir: Path | None = None) -> SplitReport:
    """Check every split file before a run that costs hours is worth starting.

    Raises:
        SplitLayoutError: The required `train.jsonl` is missing.
        DatasetError: A present file is unreadable or built against another prompt shape.
    """
    directory = (DEFAULT_SPLIT_DIR if data_dir is None else data_dir).expanduser()

    train = directory / TRAIN_FILENAME
    if not train.is_file():
        msg = (
            f"{train}: required split is missing. {VALID_FILENAME} and {TEST_FILENAME} are "
            f"optional; see data/README.md for how to produce all three"
        )
        raise SplitLayoutError(msg)

    rows = {TRAIN_FILENAME: validate_training_file(train)}
    notices: list[str] = []

    valid = directory / VALID_FILENAME
    if valid.is_file():
        rows[VALID_FILENAME] = validate_training_file(valid)
    else:
        notices.append(f"{valid} is absent; training will report no validation loss.")

    test = directory / TEST_FILENAME
    if test.is_file():
        rows[TEST_FILENAME] = validate_training_file(test)

    return SplitReport(data_dir=directory, rows=rows, notices=tuple(notices))
