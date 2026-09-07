"""Model selection."""

from __future__ import annotations

import os

DEFAULT_MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_ENV_VAR = "PEREVOD_MODEL"


def resolve_model_id(explicit: str | None = None) -> str:
    """Precedence: `explicit`, `$PEREVOD_MODEL`, default. Blank counts as absent."""
    if explicit and explicit.strip():
        return explicit.strip()

    from_env = os.environ.get(MODEL_ENV_VAR, "")
    if from_env.strip():
        return from_env.strip()

    return DEFAULT_MODEL_ID
