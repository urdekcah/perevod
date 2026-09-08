# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Frozen comparison set: editing it invalidates every comparison recorded before."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

# Conversation, idiom, coreference, technical — a small model degrades unevenly.
PROBE_SENTENCES: tuple[str, ...] = (
    "Здравствуйте, как дела?",
    "Извините, я опоздал — на дорогах пробки.",
    "Не откладывай на завтра то, что можно сделать сегодня.",
    "Он обещал золотые горы, но в итоге всё осталось на словах.",
    "Марина оставила ключи соседке, потому что уезжала на неделю; та потом отдала их брату.",
    "Инженер собрал прототип, показал его команде, и они решили, что он готов к испытаниям.",
    "Модель загружается один раз и остаётся в памяти, поэтому первый запрос заметно медленнее.",
    "Квантование до четырёх бит уменьшает объём весов примерно вчетверо ценой части точности.",
)


class SupportsTranslate(Protocol):
    """So a fake can stand in for the translator."""

    def translate(self, source_text: str) -> str:
        """Translate one sentence."""
        ...


def run_probe_set(
    translator: SupportsTranslate,
    sentences: Iterable[str] = PROBE_SENTENCES,
) -> tuple[str, ...]:
    """In order, so two runs line up row by row."""
    return tuple(translator.translate(sentence) for sentence in sentences)
