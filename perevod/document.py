# © 2026 urdekcah. Все права защищены.
# Лицензировано в соответствии с условиями AGPL-3.0
"""Splitting a document into translatable chunks and putting it back together.

No model, no I/O here, deliberately: keep it that way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

DEFAULT_CHUNK_BUDGET_TOKENS = 1024

# Deliberate under-estimate: too large truncates silently, too small only costs throughput.
CHARS_PER_TOKEN_RU = 2.0

# Over-provisioning costs nothing once the model emits EOS, so err generous here.
OUTPUT_TOKEN_MULTIPLIER = 2.0

DEFAULT_MAX_CONSECUTIVE_FAILURES = 3

# Once a user's script greps for these, the literals are no longer free to change.
UNTRANSLATED_OPEN = "[[PEREVOD:UNTRANSLATED chunk={index}]]"
UNTRANSLATED_CLOSE = "[[/PEREVOD:UNTRANSLATED chunk={index}]]"

# Incomplete by construction; a miss costs a chunk boundary, nothing more.
RU_ABBREVIATIONS = frozenset(
    {
        "др.", "см.", "рис.", "табл.", "стр.", "гг.", "вв.", "руб.", "коп.", "тыс.",
        "млн.", "млрд.", "проф.", "доц.", "акад.", "ул.", "пр.", "пер.", "наб.", "обл.",
        "респ.", "дер.", "пос.", "кв.", "корп.", "им.", "тел.", "ок.", "напр.",
    }
)  # fmt: skip

_TERMINATORS = ".!?…"
_CANDIDATE_RE = re.compile(r"(?P<run>[.!?…]+)(?P<close>[»\"'”’)\]]*)")  # noqa: RUF001
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\S+\s*")

_PARAGRAPH_BREAK_NEWLINES = 2


@dataclass(frozen=True)
class Chunk:
    """Invariant: `lead + text + tail` joined over all chunks is the source, exactly."""

    index: int
    lead: str
    text: str
    tail: str


@dataclass(frozen=True)
class Progress:
    """One step of a run in flight."""

    phase: Literal["start", "chunk_done", "file_done", "done"]
    chunk_index: int
    chunk_total: int
    chars_done: int
    chars_total: int
    file_index: int
    file_total: int
    source_name: str
    ok: bool


@dataclass(frozen=True)
class ChunkResult:
    """When `ok` is false, `text` holds the marker block rather than a translation."""

    index: int
    text: str
    ok: bool
    error: str | None


@dataclass(frozen=True)
class DocumentResult:
    """`text` is complete even when chunks failed — the holes carry markers."""

    text: str
    chunks_total: int
    chunks_failed: tuple[int, ...]
    aborted: bool


def budget_chars_for(budget_tokens: int, *, chars_per_token: float = CHARS_PER_TOKEN_RU) -> int:
    """Convert a token budget to the character budget the splitter works in.

    Raises:
        ValueError: `budget_tokens` is below 1.
    """
    if budget_tokens < 1:
        msg = f"budget_tokens must be at least 1, got {budget_tokens}"
        raise ValueError(msg)
    return max(1, int(budget_tokens * chars_per_token))


def _preceding_token(text: str, dot_index: int) -> str:
    start = dot_index
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in _TERMINATORS:
        start -= 1
    return text[start:dot_index]


def _next_visible(text: str, index: int) -> str:
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index] if index < len(text) else ""


def _is_boundary(paragraph: str, match: re.Match[str]) -> bool:
    """R-c gets no uppercase-next escape hatch: an initial is followed by a capital anyway."""
    after_run = paragraph[match.end("run") : match.end("run") + 1]
    after_all = paragraph[match.end() : match.end() + 1]

    if after_all and not after_all.isspace():  # R-a
        return False
    if match["run"] != ".":
        return True

    dot_index = match.start("run")
    before = paragraph[dot_index - 1] if dot_index else ""

    if before.isdigit() and after_run.isdigit():  # R-b
        return False

    token = _preceding_token(paragraph, dot_index)
    if len(token) == 1 and token.isupper():  # R-c
        return False
    if (len(token) == 1 and token.islower()) or f"{token.lower()}." in RU_ABBREVIATIONS:  # R-d
        return _next_visible(paragraph, match.end()).isupper()
    return True


def split_sentences(paragraph: str) -> list[str]:
    """Split a blank-line-free paragraph, each sentence carrying the whitespace after it.

    Heuristic but lossless, so a wrong boundary only moves a chunk edge.
    """
    pieces: list[str] = []
    start = 0
    for match in _CANDIDATE_RE.finditer(paragraph):
        if not _is_boundary(paragraph, match):
            continue
        end = match.end()
        while end < len(paragraph) and paragraph[end].isspace():
            end += 1
        pieces.append(paragraph[start:end])
        start = end
    if start < len(paragraph):
        pieces.append(paragraph[start:])
    return pieces


def _hard_split(unit: str, budget_chars: int) -> list[str]:
    body = unit.rstrip()
    trailing = unit[len(body) :]
    pieces = [body[at : at + budget_chars] for at in range(0, len(body), budget_chars)]
    pieces[-1] += trailing
    return pieces


def _atoms(paragraph: str, budget_chars: int) -> list[str]:
    atoms: list[str] = []
    for sentence in split_sentences(paragraph):
        if len(sentence.rstrip()) <= budget_chars:
            atoms.append(sentence)
            continue
        for word in _WORD_RE.findall(sentence):
            if len(word.rstrip()) <= budget_chars:
                atoms.append(word)
            else:
                atoms.extend(_hard_split(word, budget_chars))
    return atoms


def _pack(atoms: Iterable[str], budget_chars: int) -> list[str]:
    groups: list[str] = []
    current = ""
    for atom in atoms:
        if current and len((current + atom).rstrip()) > budget_chars:
            groups.append(current)
            current = atom
        else:
            current += atom
    if current:
        groups.append(current)
    return groups


def _paragraphs(body: str, trailing: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    start = 0
    for run in _WHITESPACE_RE.finditer(body):
        if run.group().count("\n") < _PARAGRAPH_BREAK_NEWLINES:
            continue
        pairs.append((body[start : run.start()], run.group()))
        start = run.end()
    pairs.append((body[start:], trailing))
    return pairs


def split_document(source: str, *, budget_chars: int) -> list[Chunk]:
    """Partition a document into chunks that never span a paragraph boundary.

    That is what keeps blank lines out of the model's input.

    Raises:
        ValueError: `budget_chars` is below 1.
    """
    if budget_chars < 1:
        msg = f"budget_chars must be at least 1, got {budget_chars}"
        raise ValueError(msg)
    if not source:
        return []

    body = source.strip()
    lead = source[: len(source) - len(source.lstrip())]
    if not body:
        return [Chunk(index=0, lead="", text="", tail=source)]

    trailing = source[len(source.rstrip()) :]
    chunks: list[Chunk] = []
    pending_lead = lead
    for paragraph, separator in _paragraphs(body, trailing):
        groups = _pack(_atoms(paragraph, budget_chars), budget_chars) or [""]
        for position, group in enumerate(groups):
            text = group.rstrip()
            tail = group[len(text) :]
            if position == len(groups) - 1:
                tail += separator
            chunks.append(Chunk(index=len(chunks), lead=pending_lead, text=text, tail=tail))
            pending_lead = ""
    return chunks


def render_chunk(chunk: Chunk, translation: str) -> str:
    """The strip is load-bearing: a stray newline from the model would double a break."""
    return chunk.lead + translation.strip() + chunk.tail


def join_translations(chunks: Sequence[Chunk], translations: Sequence[str]) -> str:
    """Reassemble a document from its chunks.

    Raises:
        ValueError: The two sequences differ in length.
    """
    return "".join(
        render_chunk(chunk, translation)
        for chunk, translation in zip(chunks, translations, strict=True)
    )


def untranslated_marker(chunk: Chunk) -> str:
    """The block that stands in for a chunk that could not be translated."""
    return (
        f"{UNTRANSLATED_OPEN.format(index=chunk.index)}\n"
        f"{chunk.text}\n"
        f"{UNTRANSLATED_CLOSE.format(index=chunk.index)}"
    )


class DocumentTranslator:
    """Maps chunks through an injected translation function and reassembles the result.

    A failed chunk leaves a marker holding its source, never a gap, so a partial run cannot
    pass for a complete one; `max_consecutive_failures` in a row abandons the rest unattempted.
    `KeyboardInterrupt` and `MemoryError` propagate — they are not chunk failures.
    """

    def __init__(
        self,
        translate_chunk: Callable[[str], str],
        *,
        budget_chars: int = 0,
        progress: Callable[[Progress], None] | None = None,
        retries: int = 0,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ) -> None:
        self._translate_chunk = translate_chunk
        self.budget_chars = budget_chars or budget_chars_for(DEFAULT_CHUNK_BUDGET_TOKENS)
        self._progress = progress
        self.retries = retries
        self.max_consecutive_failures = max_consecutive_failures

    def _attempt(self, chunk: Chunk) -> ChunkResult:
        last_error = ""
        for _ in range(self.retries + 1):
            try:
                translated = self._translate_chunk(chunk.text)
                return ChunkResult(index=chunk.index, text=translated, ok=True, error=None)
            except (KeyboardInterrupt, MemoryError):
                raise
            except Exception as error:  # noqa: BLE001 -- any backend failure is one bad chunk
                last_error = f"{type(error).__name__}: {error}"
        return ChunkResult(
            index=chunk.index, text=untranslated_marker(chunk), ok=False, error=last_error
        )

    def translate_document(
        self,
        source: str,
        *,
        sink: Callable[[str], None] | None = None,
        file_index: int = 0,
        file_total: int = 1,
        source_name: str = "",
    ) -> DocumentResult:
        """`sink` sees each fragment as it completes, so a crash keeps what finished."""
        chunks = split_document(source, budget_chars=self.budget_chars)
        chars_total = sum(len(chunk.text) for chunk in chunks)
        self._emit(
            "start", 0, len(chunks), 0, chars_total, file_index, file_total, source_name, ok=True
        )

        parts: list[str] = []
        failed: list[int] = []
        consecutive = 0
        aborted = False
        chars_done = 0

        for chunk in chunks:
            if not chunk.text:
                result = ChunkResult(index=chunk.index, text="", ok=True, error=None)
            elif aborted:
                result = ChunkResult(
                    index=chunk.index, text=untranslated_marker(chunk), ok=False, error="aborted"
                )
            else:
                result = self._attempt(chunk)

            if not result.ok:
                failed.append(chunk.index)
                consecutive += 1
                if consecutive >= self.max_consecutive_failures:
                    aborted = True
            else:
                consecutive = 0

            fragment = render_chunk(chunk, result.text)
            parts.append(fragment)
            if sink is not None:
                sink(fragment)

            chars_done += len(chunk.text)
            self._emit(
                "chunk_done",
                chunk.index,
                len(chunks),
                chars_done,
                chars_total,
                file_index,
                file_total,
                source_name,
                ok=result.ok,
            )

        self._emit(
            "done",
            len(chunks),
            len(chunks),
            chars_done,
            chars_total,
            file_index,
            file_total,
            source_name,
            ok=not failed,
        )
        return DocumentResult(
            text="".join(parts),
            chunks_total=len(chunks),
            chunks_failed=tuple(failed),
            aborted=aborted,
        )

    def _emit(  # noqa: PLR0913, PLR0917 -- one parameter per Progress field
        self,
        phase: Literal["start", "chunk_done", "file_done", "done"],
        chunk_index: int,
        chunk_total: int,
        chars_done: int,
        chars_total: int,
        file_index: int,
        file_total: int,
        source_name: str,
        *,
        ok: bool,
    ) -> None:
        if self._progress is None:
            return
        self._progress(
            Progress(
                phase=phase,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                chars_done=chars_done,
                chars_total=chars_total,
                file_index=file_index,
                file_total=file_total,
                source_name=source_name,
                ok=ok,
            )
        )
