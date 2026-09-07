"""Offline Russian to Korean translation for Apple Silicon."""

from perevod.config import DEFAULT_MODEL_ID, MODEL_ENV_VAR, resolve_model_id
from perevod.dataset import (
    MANIFEST_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALID_FILENAME,
    DatasetError,
    RecordShapeError,
    build_training_record,
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
    render_report,
)
from perevod.translator import Translator

__all__ = [
    "DEFAULT_MODEL_ID",
    "MANIFEST_FILENAME",
    "MODEL_ENV_VAR",
    "TEST_FILENAME",
    "TRAIN_FILENAME",
    "VALID_FILENAME",
    "DatasetError",
    "InputError",
    "OutputExistsError",
    "OutputPathError",
    "PrepareConfig",
    "PrepareError",
    "PrepareResult",
    "RecordShapeError",
    "Translator",
    "build_training_record",
    "parse_training_record",
    "prepare_splits",
    "prompt_shape_fingerprint",
    "render_report",
    "resolve_model_id",
]
