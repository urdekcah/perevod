# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Prompt assembly and model-backed translation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from typing import Any, cast

from perevod.adapters import check_adapter_compatibility
from perevod.config import resolve_adapter_path, resolve_model_id
from perevod.document import (
    CHARS_PER_TOKEN_RU,
    DEFAULT_CHUNK_BUDGET_TOKENS,
    OUTPUT_TOKEN_MULTIPLIER,
    DocumentResult,
    DocumentTranslator,
    Progress,
    budget_chars_for,
)

TRANSLATION_INSTRUCTION = (
    "Translate the following Russian text into Korean. "
    "Reply with the Korean translation only, with no explanation, "
    "no transliteration, and no restatement of the source."
)

DEFAULT_MAX_TOKENS = 512

# Read from the loader's live signature; a rename upstream must raise, not drop it.
ADAPTER_LOAD_PARAM = "adapter_path"

# Stands in for the source while the scaffolding around it is measured.
SHAPE_SENTINEL = "«PEREVOD-SOURCE»"

# The two seams tests replace; named so the MLX defaults have something to cast to.
Loader = Callable[..., tuple[Any, Any]]
Generator = Callable[..., Iterable[Any]]

# A hardcoded delimiter would go quiet the day upstream renames it.
REASONING_DELIMITER_ATTRS = ("think_start", "think_end")


class TranslationError(RuntimeError):
    """Base for the failures `translate` raises instead of returning text."""


class TruncatedTranslationError(TranslationError):
    """Raised when generation stopped before the model signalled it was done."""

    def __init__(self, *, max_tokens: int, partial_text: str, finish_reason: str | None) -> None:
        self.max_tokens = max_tokens
        self.partial_text = partial_text
        self.finish_reason = finish_reason
        super().__init__(
            f"Generation did not finish within {max_tokens} tokens "
            f"(finish_reason={finish_reason!r}). Raise the token ceiling and retry."
        )


class ReasoningLeakError(TranslationError):
    """Raised when the reasoning channel survived suppression at the template."""

    def __init__(self, delimiter: str, partial_text: str) -> None:
        self.delimiter = delimiter
        self.partial_text = partial_text
        super().__init__(
            f"The completion carries the reasoning delimiter {delimiter!r}, so suppression "
            f"did not take effect. The text is not a translation."
        )


def build_messages(source_text: str) -> list[dict[str, str]]:
    """Fine-tuning data must serialize to this shape; a mismatch degrades quality silently."""
    return [
        {"role": "system", "content": TRANSLATION_INSTRUCTION},
        {"role": "user", "content": source_text},
    ]


def reference_messages() -> list[dict[str, str]]:
    """The shipped shape with the source slot marked, for anything that measures it."""
    return build_messages(SHAPE_SENTINEL)


def prompt_shape_fingerprint() -> str:
    """Digest of the shipped prompt shape; work built against an older one stays detectable."""
    canonical = json.dumps(
        reference_messages(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Translator:
    """Russian to Korean translation over a locally held model.

    Weights load in the constructor — a bad repository id or a mispaired adapter fails
    there, not mid-translation — and stay resident at several gigabytes, so build one and
    reuse it. `loader` and `generator` are seams for tests; both default to MLX.

    Raises:
        ImportError: MLX is unavailable and neither seam was supplied.
        AdapterProvenanceError: The adapter is unusable or belongs to another base.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        adapter_path: str | None = None,
        allow_provenance_mismatch: bool = False,
        loader: Loader | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.model_id = resolve_model_id(model_id)
        self.adapter_path = resolve_adapter_path(adapter_path)

        if loader is None or generator is None:
            # Deferred so importing this package works where MLX is absent.
            from mlx_lm import load as mlx_load  # noqa: PLC0415
            from mlx_lm import stream_generate as mlx_stream_generate  # noqa: PLC0415

            # MLX's signatures are wider; the extra parameters all carry defaults.
            if loader is None:
                loader = cast("Loader", mlx_load)
            if generator is None:
                generator = cast("Generator", mlx_stream_generate)

        self._generate: Generator = generator

        if self.adapter_path is None:
            self.model, self.tokenizer = loader(self.model_id)
        else:
            check_adapter_compatibility(
                self.adapter_path,
                base_model_id=self.model_id,
                prompt_fingerprint=prompt_shape_fingerprint(),
                allow_mismatch=allow_provenance_mismatch,
            )
            self.model, self.tokenizer = loader(
                self.model_id, **{ADAPTER_LOAD_PARAM: self.adapter_path}
            )

    def translate(self, source_text: str, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Translate one string; the result is the translation and nothing else.

        Raises:
            ReasoningLeakError: A reasoning delimiter reached the completion.
            TruncatedTranslationError: Generation ran out of budget mid-answer.
        """
        prompt = self.tokenizer.apply_chat_template(
            build_messages(source_text),
            add_generation_prompt=True,
            tokenize=False,
            # Omitting this means "whatever the checkpoint defaults to" — which here is on.
            enable_thinking=False,
        )
        segments: list[str] = []
        finish_reason: str | None = None
        for response in self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            segments.append(response.text)
            finish_reason = response.finish_reason
        completion = "".join(segments).strip()

        leaked = self._leaked_delimiter(completion)
        if leaked is not None:
            raise ReasoningLeakError(leaked, completion)
        if finish_reason != "stop":
            raise TruncatedTranslationError(
                max_tokens=max_tokens, partial_text=completion, finish_reason=finish_reason
            )
        return completion

    def _leaked_delimiter(self, completion: str) -> str | None:
        for attribute in REASONING_DELIMITER_ATTRS:
            delimiter = getattr(self.tokenizer, attribute, None)
            if delimiter and delimiter in completion:
                return cast("str", delimiter)
        return None

    def translate_document(  # noqa: PLR0913 -- progress and batch context are all optional
        self,
        source_text: str,
        *,
        max_tokens: int | None = None,
        budget_tokens: int = DEFAULT_CHUNK_BUDGET_TOKENS,
        progress: Callable[[Progress], None] | None = None,
        sink: Callable[[str], None] | None = None,
        file_index: int = 0,
        file_total: int = 1,
        source_name: str = "",
    ) -> DocumentResult:
        """Translate a document by splitting it into chunks and reassembling the result.

        Every chunk travels the same prompt path as a bare string, so an adapter applies here
        untouched. Left unset, `max_tokens` is sized per chunk from that chunk's length.
        """

        def translate_chunk(text: str) -> str:
            ceiling = max_tokens
            if ceiling is None:
                estimated = math.ceil(len(text) / CHARS_PER_TOKEN_RU * OUTPUT_TOKEN_MULTIPLIER)
                ceiling = max(DEFAULT_MAX_TOKENS, estimated)
            return self.translate(text, max_tokens=ceiling)

        return DocumentTranslator(
            translate_chunk,
            budget_chars=budget_chars_for(budget_tokens),
            progress=progress,
        ).translate_document(
            source_text,
            sink=sink,
            file_index=file_index,
            file_total=file_total,
            source_name=source_name,
        )
