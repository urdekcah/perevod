# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Which base an adapter was fitted to. On the wrong one it degrades quietly."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# Namespaced: the trainer writes into this directory too.
PROVENANCE_FILENAME = "perevod_provenance.json"
PROVENANCE_SCHEMA_VERSION = 1

REQUIRED_KEYS = ("schema_version", "base_model_id", "prompt_shape_fingerprint")

_OVERRIDE_HINT = "pass allow_provenance_mismatch (--allow-provenance-mismatch) to translate anyway"


class AdapterProvenanceError(Exception):
    """The record, or the directory holding it, is unusable."""


class AdapterMismatchError(AdapterProvenanceError):
    """Fitted against a different base or prompt shape."""


@dataclass(frozen=True)
class AdapterProvenance:
    """What an adapter directory records about the run that produced it."""

    base_model_id: str
    prompt_shape_fingerprint: str
    schema_version: int = PROVENANCE_SCHEMA_VERSION
    created_at: str | None = None
    mlx_lm_version: str | None = None


def provenance_path(adapter_dir: str | Path) -> Path:
    """Where the sidecar lives for `adapter_dir`."""
    return Path(adapter_dir).expanduser() / PROVENANCE_FILENAME


def write_provenance(adapter_dir: str | Path, provenance: AdapterProvenance) -> Path:
    """Record `provenance` inside the adapter directory so it travels with the weights."""
    target = provenance_path(adapter_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(asdict(provenance), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_provenance(adapter_dir: str | Path) -> AdapterProvenance | None:
    """Read the sidecar, or `None` when absent. A corrupt one raises: absent only warns."""
    source = provenance_path(adapter_dir)
    if not source.is_file():
        return None

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = f"{source}: unreadable provenance record ({error})"
        raise AdapterProvenanceError(msg) from error

    if not isinstance(payload, dict):
        msg = f"{source}: provenance record is not a JSON object"
        raise AdapterProvenanceError(msg)

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        msg = f"{source}: provenance record is missing {', '.join(missing)}"
        raise AdapterProvenanceError(msg)

    if payload["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        msg = (
            f"{source}: provenance schema version {payload['schema_version']!r} is not "
            f"{PROVENANCE_SCHEMA_VERSION}; this release cannot vouch for the adapter"
        )
        raise AdapterProvenanceError(msg)

    return AdapterProvenance(
        base_model_id=str(payload["base_model_id"]),
        prompt_shape_fingerprint=str(payload["prompt_shape_fingerprint"]),
        schema_version=PROVENANCE_SCHEMA_VERSION,
        created_at=_optional_str(payload.get("created_at")),
        mlx_lm_version=_optional_str(payload.get("mlx_lm_version")),
    )


def _optional_str(value: Any) -> str | None:  # noqa: ANN401 -- whatever the JSON held
    return None if value is None else str(value)


def _warn_to_stderr(message: str) -> None:
    print(message, file=sys.stderr)  # noqa: T201 -- stdout stays translation-only


def check_adapter_compatibility(
    adapter_dir: str | Path,
    *,
    base_model_id: str,
    prompt_fingerprint: str,
    allow_mismatch: bool = False,
    warn: Callable[[str], None] = _warn_to_stderr,
) -> AdapterProvenance | None:
    """Decide whether this adapter may be applied to this base, before anything loads.

    No sidecar still loads, warning: adapters predating the record must keep working.
    An unusable record is not overridable — there is nothing to accept, only to fix.
    """
    directory = Path(adapter_dir).expanduser()
    if not directory.is_dir():
        msg = f"{directory}: no adapter directory at this path"
        raise AdapterProvenanceError(msg)

    provenance = read_provenance(directory)
    if provenance is None:
        warn(
            f"{directory}: no {PROVENANCE_FILENAME}, so the base it was trained against "
            f"cannot be checked. Translations may be silently degraded if the base is wrong."
        )
        return None

    if provenance.base_model_id != base_model_id:
        _report(
            warn,
            allow_mismatch,
            f"adapter {directory} was trained against {provenance.base_model_id!r}, "
            f"not {base_model_id!r}",
        )
    elif provenance.prompt_shape_fingerprint != prompt_fingerprint:
        _report(
            warn,
            allow_mismatch,
            f"adapter {directory} was fitted to prompt shape "
            f"{provenance.prompt_shape_fingerprint}, not {prompt_fingerprint}",
        )
    return provenance


def _report(warn: Callable[[str], None], allow_mismatch: bool, detail: str) -> None:  # noqa: FBT001
    if allow_mismatch:
        warn(f"{detail}; proceeding as asked")
        return
    msg = f"{detail}. To translate anyway, {_OVERRIDE_HINT}"
    raise AdapterMismatchError(msg)
