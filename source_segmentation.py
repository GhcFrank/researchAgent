"""Deterministic source blocks and sequential chunks, using only the standard library.

Blocks retain snapshot-wide B locators. Chunks group whole blocks without
overlap or relevance selection. Character limits include rendered locators
and separators, but not an extractor's surrounding prompt or token overhead.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import re


class SourceSegmentationError(Exception):
    """Base error for source segmentation."""


class SourceSegmentationValidationError(SourceSegmentationError):
    """Content, block collections, or chunk limits are invalid."""


@dataclass(frozen=True)
class SourceBlock:
    block_id: str
    text: str


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    blocks: tuple[SourceBlock, ...]
    char_count: int


_BLOCK_SEPARATOR = "\n\n"


def build_source_blocks(content: str) -> list[SourceBlock]:
    """Split blank-line paragraphs; trim edges while preserving internal text.

    Whitespace-only lines separate blocks. Internal LF/CRLF and indentation
    remain unchanged. Empty or whitespace-only content produces no blocks.
    """
    if not isinstance(content, str):
        raise SourceSegmentationValidationError("content must be a string")
    paragraphs = [block.strip() for block in re.split(r"\r?\n[^\S\r\n]*\r?\n", content) if block.strip()]
    return [SourceBlock(f"B{index:03d}", text) for index, text in enumerate(paragraphs, start=1)]


def _validate_blocks(blocks: Sequence[SourceBlock]) -> None:
    if (
        not isinstance(blocks, Sequence)
        or isinstance(blocks, (str, bytes))
        or any(not isinstance(block, SourceBlock) for block in blocks)
    ):
        raise SourceSegmentationValidationError("blocks must be a sequence of SourceBlock objects")


def render_blocks(blocks: Sequence[SourceBlock]) -> str:
    """Render the existing [Bxxx] newline text format; empty input renders ''."""
    _validate_blocks(blocks)
    return _BLOCK_SEPARATOR.join(f"[{block.block_id}]\n{block.text}" for block in blocks)


def chunk_source_blocks(blocks: Sequence[SourceBlock], max_chars: int = 8000) -> list[SourceChunk]:
    """Pack whole blocks in order using rendered character counts.

    Equality with max_chars fits. An oversized block remains one singleton
    chunk. Every supplied block occurs exactly once with its original locator.
    Empty sequences produce no chunks, and B/C numbering has a minimum width
    of three digits, allowing B1000 and C1000 without truncation.
    """
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise SourceSegmentationValidationError("max_chars must be a positive integer")
    _validate_blocks(blocks)
    chunks = []
    pending = []
    char_count = 0
    for block in blocks:
        block_chars = len(render_blocks((block,)))
        separator_chars = len(_BLOCK_SEPARATOR) if pending else 0
        if pending and char_count + separator_chars + block_chars > max_chars:
            chunks.append(SourceChunk(f"C{len(chunks) + 1:03d}", tuple(pending), char_count))
            pending = []
            char_count = 0
        char_count += block_chars + (len(_BLOCK_SEPARATOR) if pending else 0)
        pending.append(block)
    if pending:
        chunks.append(SourceChunk(f"C{len(chunks) + 1:03d}", tuple(pending), char_count))
    return chunks


def render_chunk(chunk: SourceChunk) -> str:
    """Render original B locators only; the execution-unit C ID is not included."""
    if not isinstance(chunk, SourceChunk):
        raise SourceSegmentationValidationError("chunk must be a SourceChunk")
    return render_blocks(chunk.blocks)
