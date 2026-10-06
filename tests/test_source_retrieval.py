"""Small synthetic chunks test ranking explanations and selection boundaries."""

from copy import deepcopy

import pytest

from source_retrieval import SourceRetrievalValidationError, retrieve_candidate_chunks
from source_segmentation import SourceBlock, SourceChunk, render_blocks


def make_chunks(*groups):
    chunks = []
    block_index = 1
    for index, group in enumerate(groups, start=1):
        texts = (group,) if isinstance(group, str) else group
        blocks = tuple(SourceBlock(f"B{block_index + offset:03d}", text) for offset, text in enumerate(texts))
        block_index += len(blocks)
        chunks.append(SourceChunk(f"C{index:03d}", blocks, len(render_blocks(blocks))))
    return chunks


def test_exact_phrase_outweighs_frequent_single_term():
    chunks = make_chunks("revenue " * 40, "Revenue growth slowed.")
    candidates = retrieve_candidate_chunks(chunks, phrases=["revenue growth"], terms=["revenue"], top_k=2, neighbor_radius=0)
    assert [candidate.chunk_id for candidate in candidates] == ["C001", "C002"]
    term, phrase = candidates
    assert (term.score, term.rank, term.matched_phrases, term.matched_terms) == (2, 2, (), ("revenue",))
    assert (phrase.score, phrase.rank, phrase.matched_phrases, phrase.matched_terms) == (9, 1, ("revenue growth",), ("revenue",))


def test_repetition_and_duplicate_concepts_do_not_inflate_score():
    chunks = make_chunks("revenue " * 40, "revenue " * 80)
    candidates = retrieve_candidate_chunks(
        chunks, phrases=[" revenue ", "REVENUE"], terms=["REVENUE", " revenue "], neighbor_radius=0,
    )
    assert [candidate.score for candidate in candidates] == [6, 6]
    assert all(candidate.matched_phrases == candidate.matched_terms == ("revenue",) for candidate in candidates)


def test_distinct_concept_diversity_beats_repeated_single_concept():
    chunks = make_chunks("revenue " * 40, "revenue growth government backlog " + "context " * 40)
    candidates = retrieve_candidate_chunks(chunks, terms=["revenue", "growth", "government", "backlog"], neighbor_radius=0)
    assert [(candidate.score, candidate.rank) for candidate in candidates] == [(2, 2), (8, 1)]
    assert candidates[1].matched_terms == ("revenue", "growth", "government", "backlog")


def test_case_whitespace_word_boundaries_and_block_boundaries():
    chunks = make_chunks(
        " Remaining \n PERFORMANCE\t Obligations; marginal and subscriptional.",
        ("Remaining performance", "obligations"),
        "Marginal gains and subscriptional offerings.",
        "Margin; subscription.",
    )
    candidates = retrieve_candidate_chunks(
        chunks, phrases=["  remaining performance   obligations  ", " "],
        terms=[" MARGIN ", " SUBSCRIPTION ", "\t"], neighbor_radius=0,
    )
    assert [candidate.chunk_id for candidate in candidates] == ["C001", "C004"]
    assert candidates[0].matched_phrases == ("remaining performance obligations",)
    assert candidates[0].matched_terms == ()
    assert candidates[1].matched_terms == ("margin", "subscription")


def test_short_block_bonus_is_small_and_capped_once_per_chunk():
    chunks = make_chunks("Revenue " * 40, ("Revenue", "Revenue"))
    candidates = retrieve_candidate_chunks(chunks, terms=["revenue"], neighbor_radius=0)
    assert [candidate.score for candidate in candidates] == [2, 3]
    assert candidates[1].rank == 1


def test_ties_top_k_and_result_order_follow_source_order_without_mutation():
    original = make_chunks("Revenue", "Revenue", "Revenue")
    chunks = [original[2], original[0], original[1]]
    before = deepcopy(chunks)
    candidates = retrieve_candidate_chunks(chunks, terms=["revenue"], top_k=2, neighbor_radius=0)
    assert [(candidate.chunk_id, candidate.rank) for candidate in candidates] == [("C003", 1), ("C001", 2)]
    assert [candidate.block_ids for candidate in candidates] == [("B003",), ("B001",)]
    assert all(candidate.match_type == "direct_match" for candidate in candidates)
    assert retrieve_candidate_chunks(chunks, terms=["revenue"], top_k=2, neighbor_radius=0) == candidates
    assert chunks == before


@pytest.mark.parametrize(("radius", "indices"), [(0, [0, 4]), (1, [0, 1, 3, 4]), (2, [0, 1, 2, 3, 4])])
def test_neighbor_expansion_clips_deduplicates_and_preserves_real_scores(radius, indices):
    chunks = make_chunks("Revenue growth.", "Growth.", "Unrelated.", "Unrelated.", "Revenue growth.")
    before = deepcopy(chunks)
    candidates = retrieve_candidate_chunks(chunks, phrases=["revenue growth"], terms=["growth"], top_k=2, neighbor_radius=radius)
    assert [candidate.chunk_id for candidate in candidates] == [chunks[index].chunk_id for index in indices]
    assert len({candidate.chunk_id for candidate in candidates}) == len(candidates)
    assert [(candidate.chunk_id, candidate.rank) for candidate in candidates if candidate.match_type == "direct_match"] == [("C001", 1), ("C005", 2)]
    neighbors = [candidate for candidate in candidates if candidate.match_type == "neighbor"]
    assert all(candidate.rank is None for candidate in neighbors)
    if radius:
        scored_neighbor = next(candidate for candidate in neighbors if candidate.chunk_id == "C002")
        assert scored_neighbor.score == 3 and scored_neighbor.matched_terms == ("growth",)
        assert any(candidate.score == 0 and not candidate.matched_terms and not candidate.matched_phrases for candidate in neighbors)
    assert retrieve_candidate_chunks(chunks, phrases=["revenue growth"], terms=["growth"], top_k=2, neighbor_radius=radius) == candidates
    assert chunks == before


def test_zero_matches_blank_concepts_and_empty_chunks_return_empty():
    chunks = make_chunks("Unrelated content.")
    assert retrieve_candidate_chunks(chunks, phrases=["", " \t"], terms=["\n"]) == []
    assert retrieve_candidate_chunks(chunks, terms=["revenue"]) == []
    assert retrieve_candidate_chunks([], terms=["revenue"]) == []


@pytest.mark.parametrize("parameters", [{"top_k": 0}, {"top_k": 1.5}, {"neighbor_radius": -1}, {"neighbor_radius": True}])
def test_invalid_selection_parameters_are_rejected(parameters):
    with pytest.raises(SourceRetrievalValidationError):
        retrieve_candidate_chunks([], **parameters)


@pytest.mark.parametrize("parameters", [{"phrases": "revenue"}, {"terms": [123]}])
def test_invalid_concept_collections_are_rejected(parameters):
    with pytest.raises(SourceRetrievalValidationError, match="sequence of strings"):
        retrieve_candidate_chunks([], **parameters)


def test_ambiguous_duplicate_chunk_ids_are_rejected():
    chunk = make_chunks("Revenue")[0]
    with pytest.raises(SourceRetrievalValidationError, match="unique"):
        retrieve_candidate_chunks([chunk, chunk], terms=["revenue"])
