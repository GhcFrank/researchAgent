"""Pure segmentation and provenance invariants; no files or providers."""

import pytest

from source_segmentation import (
    SourceBlock,
    SourceSegmentationValidationError,
    build_source_blocks,
    chunk_source_blocks,
    render_blocks,
    render_chunk,
)


def test_blank_line_blocks_preserve_internal_whitespace_and_render_deterministically():
    content = "\r\n  first\r\n  indented\t fact  \r\nend \r\n \t\r\n\r\n second\n  line\n\n third \n"
    blocks = build_source_blocks(content)
    assert blocks == [
        SourceBlock("B001", "first\r\n  indented\t fact  \r\nend"),
        SourceBlock("B002", "second\n  line"),
        SourceBlock("B003", "third"),
    ]
    assert build_source_blocks(content) == blocks
    assert render_blocks(blocks) == (
        "[B001]\nfirst\r\n  indented\t fact  \r\nend\n\n[B002]\nsecond\n  line\n\n[B003]\nthird"
    )


@pytest.mark.parametrize(("max_chars", "sizes"), [(22, [2, 2, 1]), (21, [1, 1, 1, 1, 1])])
def test_sequential_packing_boundary_and_exact_provenance(max_chars, sizes):
    blocks = build_source_blocks("one\n\ntwo\n\ntri\n\nfor\n\nfiv")
    chunks = chunk_source_blocks(blocks, max_chars=max_chars)
    assert [len(chunk.blocks) for chunk in chunks] == sizes
    assert [chunk.chunk_id for chunk in chunks] == [f"C{index:03d}" for index in range(1, len(sizes) + 1)]
    assert chunk_source_blocks(blocks, max_chars=max_chars) == chunks
    assert [block for chunk in chunks for block in chunk.blocks] == blocks
    for chunk in chunks:
        assert isinstance(chunk.blocks, tuple)
        assert chunk.char_count == len(render_chunk(chunk)) <= max_chars
        assert chunk.chunk_id not in render_chunk(chunk)
    assert "\n\n".join(render_chunk(chunk) for chunk in chunks) == render_blocks(blocks)
    assert chunks[1].blocks[0].block_id == ("B003" if max_chars == 22 else "B002")


def test_oversized_block_remains_a_single_original_locator():
    blocks = build_source_blocks("small\n\n" + "x" * 50 + "\n\nend")
    chunks = chunk_source_blocks(blocks, max_chars=16)
    assert [chunk.blocks for chunk in chunks] == [(blocks[0],), (blocks[1],), (blocks[2],)]
    assert chunks[1].char_count == 57 > 16
    assert render_chunk(chunks[1]) == "[B002]\n" + "x" * 50
    assert [block for chunk in chunks for block in chunk.blocks] == blocks


def test_default_budget_counts_rendered_locators_and_separators():
    blocks = build_source_blocks("a" * 3993 + "\n\n" + "b" * 3991 + "\n\nc")
    chunks = chunk_source_blocks(blocks)
    assert [chunk.blocks for chunk in chunks] == [tuple(blocks[:2]), (blocks[2],)]
    assert chunks[0].char_count == len(render_chunk(chunks[0])) == 8000
    assert chunks[1].char_count == len(render_chunk(chunks[1])) == 8


def test_blank_content_and_empty_collections_produce_no_blocks_or_chunks():
    for content in ("", " \t\r\n\n"):
        blocks = build_source_blocks(content)
        assert blocks == []
        assert render_blocks(blocks) == ""
        assert chunk_source_blocks(blocks) == []
    assert chunk_source_blocks(()) == []


@pytest.mark.parametrize("max_chars", [0, -1, "8000", True])
def test_invalid_character_limit_is_rejected(max_chars):
    with pytest.raises(SourceSegmentationValidationError, match="max_chars"):
        chunk_source_blocks([], max_chars=max_chars)


def test_ids_keep_full_source_numbering_above_three_digits():
    blocks = build_source_blocks("\n\n".join(["x"] * 1001))
    chunks = chunk_source_blocks(blocks, max_chars=8)
    assert [block.block_id for block in blocks[-2:]] == ["B1000", "B1001"]
    assert [chunk.chunk_id for chunk in chunks[-2:]] == ["C1000", "C1001"]
    assert [block for chunk in chunks for block in chunk.blocks] == blocks
    assert render_chunk(chunks[-1]) == "[B1001]\nx"
