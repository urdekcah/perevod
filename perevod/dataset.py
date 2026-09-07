"""The training-record shape and the split file names, shared with the trainer."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from perevod.translator import build_messages

# The names mlx_lm.lora looks up.
TRAIN_FILENAME = "train.jsonl"
VALID_FILENAME = "valid.jsonl"
TEST_FILENAME = "test.jsonl"
MANIFEST_FILENAME = "manifest.json"

MESSAGES_KEY = "messages"
ROLE_KEY = "role"
CONTENT_KEY = "content"
ASSISTANT_ROLE = "assistant"

# Stands in for the source while the scaffolding around it is measured.
SHAPE_SENTINEL = "«PEREVOD-SOURCE»"


class DatasetError(Exception):
    """Base for every typed error raised over dataset files."""


class RecordShapeError(DatasetError):
    """A record disagrees with the shape the shipped prompt builder produces."""


def _reference_messages() -> list[dict[str, str]]:
    return build_messages(SHAPE_SENTINEL)


def _affixes() -> tuple[int, str, str]:
    """Which message carries the source, and the constant text either side of it."""
    messages = _reference_messages()
    carriers = [i for i, message in enumerate(messages) if SHAPE_SENTINEL in message[CONTENT_KEY]]
    if len(carriers) != 1:
        raise RecordShapeError(
            f"the prompt builder put the source into {len(carriers)} messages; exactly one is required"
        )

    index = carriers[0]
    prefix, suffix = messages[index][CONTENT_KEY].split(SHAPE_SENTINEL)
    return index, prefix, suffix


def build_training_record(source: str, target: str) -> dict[str, Any]:
    """One pair as an mlx_lm `chat` record; the prompt half comes from the shipped builder.

    Raises:
        RecordShapeError: `source` holds the sentinel, so it could not be parsed back.
    """
    if SHAPE_SENTINEL in source:
        raise RecordShapeError(f"source text contains the reserved sentinel {SHAPE_SENTINEL!r}")

    messages = build_messages(source)
    messages.append({ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: target})
    return {MESSAGES_KEY: messages}


def parse_training_record(record: Any) -> tuple[str, str]:
    """The exact inverse of `build_training_record`.

    Raises:
        RecordShapeError: The role sequence or scaffolding has drifted.
    """
    index, prefix, suffix = _affixes()
    reference = _reference_messages()

    if not isinstance(record, dict) or not isinstance(record.get(MESSAGES_KEY), list):
        raise RecordShapeError(f"record is not an object carrying a {MESSAGES_KEY!r} list")

    messages = record[MESSAGES_KEY]
    if len(messages) != len(reference) + 1:
        raise RecordShapeError(
            f"expected {len(reference) + 1} messages, found {len(messages)}"
        )

    for position, expected in enumerate(reference):
        if position != index and messages[position] != expected:
            raise RecordShapeError(
                f"message {position} is {messages[position]!r}, expected {expected!r}"
            )

    carrier = messages[index]
    content = carrier.get(CONTENT_KEY, "")
    if carrier.get(ROLE_KEY) != reference[index][ROLE_KEY]:
        raise RecordShapeError(
            f"message {index} has role {carrier.get(ROLE_KEY)!r}, expected {reference[index][ROLE_KEY]!r}"
        )
    if not content.startswith(prefix) or not content.endswith(suffix):
        raise RecordShapeError(
            f"message {index} is not wrapped in the expected prefix {prefix!r} and suffix {suffix!r}"
        )

    answer = messages[-1]
    if answer.get(ROLE_KEY) != ASSISTANT_ROLE:
        raise RecordShapeError(
            f"the last message has role {answer.get(ROLE_KEY)!r}, expected {ASSISTANT_ROLE!r}"
        )

    return content[len(prefix) : len(content) - len(suffix)], answer.get(CONTENT_KEY, "")


def prompt_shape_fingerprint() -> str:
    """Digest of the shipped prompt shape; splits built against an older one stay detectable."""
    canonical = json.dumps(
        _reference_messages(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
