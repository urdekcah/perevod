# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Model and adapter selection."""

from __future__ import annotations

import os

DEFAULT_MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_ENV_VAR = "PEREVOD_MODEL"
ADAPTER_ENV_VAR = "PEREVOD_ADAPTER"


def resolve_model_id(explicit: str | None = None) -> str:
    """Precedence: `explicit`, `$PEREVOD_MODEL`, default. Blank counts as absent."""
    if explicit and explicit.strip():
        return explicit.strip()

    from_env = os.environ.get(MODEL_ENV_VAR, "")
    if from_env.strip():
        return from_env.strip()

    return DEFAULT_MODEL_ID


def resolve_adapter_path(explicit: str | None = None) -> str | None:
    """Precedence: `explicit`, `$PEREVOD_ADAPTER`, then none. Blank counts as absent."""
    if explicit and explicit.strip():
        return explicit.strip()

    from_env = os.environ.get(ADAPTER_ENV_VAR, "")
    if from_env.strip():
        return from_env.strip()

    return None
