"""Offline Russian to Korean translation for Apple Silicon."""

from perevod.config import DEFAULT_MODEL_ID, MODEL_ENV_VAR, resolve_model_id
from perevod.translator import Translator

__all__ = [
    "DEFAULT_MODEL_ID",
    "MODEL_ENV_VAR",
    "Translator",
    "resolve_model_id",
]
