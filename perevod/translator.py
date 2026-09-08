# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Prompt assembly and model-backed translation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from perevod.config import resolve_model_id

TRANSLATION_INSTRUCTION = (
    "Translate the following Russian text into Korean. "
    "Reply with the Korean translation only, with no explanation, "
    "no transliteration, and no restatement of the source."
)

DEFAULT_MAX_TOKENS = 512

# The two seams tests replace; named so the MLX defaults have something to cast to.
Loader = Callable[[str], tuple[Any, Any]]
Generator = Callable[..., str]


def build_messages(source_text: str) -> list[dict[str, str]]:
    """Fine-tuning data must serialize to this shape; a mismatch degrades quality silently."""
    return [
        {"role": "system", "content": TRANSLATION_INSTRUCTION},
        {"role": "user", "content": source_text},
    ]


class Translator:
    """Russian to Korean translation over a locally held model.

    Weights load in the constructor — a bad repository id fails there, not
    mid-translation — and stay resident at several gigabytes, so build one and
    reuse it. `loader` and `generator` are seams for tests; both default to MLX.

    Raises:
        ImportError: MLX is unavailable and neither seam was supplied.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        loader: Loader | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.model_id = resolve_model_id(model_id)

        if loader is None or generator is None:
            # Deferred so importing this package works where MLX is absent.
            from mlx_lm import generate as mlx_generate  # noqa: PLC0415
            from mlx_lm import load as mlx_load  # noqa: PLC0415

            # MLX's signatures are wider; the extra parameters all carry defaults.
            if loader is None:
                loader = cast("Loader", mlx_load)
            if generator is None:
                generator = cast("Generator", mlx_generate)

        self._generate: Generator = generator
        self.model, self.tokenizer = loader(self.model_id)

    def translate(self, source_text: str, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Truncation at `max_tokens` is silent, not an error."""
        prompt = self.tokenizer.apply_chat_template(
            build_messages(source_text),
            add_generation_prompt=True,
            tokenize=False,
        )
        completion = self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return completion.strip()
