"""Deterministic local chunk retrieval for caller-supplied search concepts.

No ResearchTask parsing, providers, translation, extraction, or persistence.
Matching happens inside individual blocks and never invents a phrase across
block boundaries. Results contain references and explainable scores, not text.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Literal

from source_segmentation import SourceChunk


class SourceRetrievalError(Exception):
    """Base error for candidate chunk retrieval."""


class SourceRetrievalValidationError(SourceRetrievalError):
    """Concept collections, chunks, or selection parameters are invalid."""


@dataclass(frozen=True)
class CandidateChunk:
    chunk_id: str
    score: int
    matched_phrases: tuple[str, ...]
    matched_terms: tuple[str, ...]
    block_ids: tuple[str, ...]
    match_type: Literal["direct_match", "neighbor"]
    rank: int | None


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _concept_patterns(concepts: Sequence[str]) -> dict[str, re.Pattern]:
    if (
        not isinstance(concepts, Sequence)
        or isinstance(concepts, (str, bytes))
        or any(not isinstance(concept, str) for concept in concepts)
    ):
        raise SourceRetrievalValidationError("Concepts must be a sequence of strings")
    # Blank concepts are ignored; normalized duplicates retain their first order.
    normalized = dict.fromkeys(concept for text in concepts if (concept := _normalize(text)))
    return {concept: re.compile(r"(?<!\w)" + re.escape(concept) + r"(?!\w)") for concept in normalized}


def _score_chunk(chunk: SourceChunk, phrases: dict, terms: dict) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    blocks = [(_normalize(block.text), len(block.text)) for block in chunk.blocks]
    matched_phrases = tuple(concept for concept, pattern in phrases.items()
                            if any(pattern.search(text) for text, _ in blocks))
    matched_terms = tuple(concept for concept, pattern in terms.items()
                          if any(pattern.search(text) for text, _ in blocks))
    phrase_set, term_set = set(matched_phrases), set(matched_terms)
    distinct = phrase_set | term_set
    heading_bonus = int(any(
        length <= 200 and any(pattern.search(text) for pattern in (*phrases.values(), *terms.values()))
        for text, length in blocks
    ))
    # One capped contribution per concept, with phrase precedence for overlap.
    score = 5 * len(phrase_set) + len(term_set - phrase_set) + len(distinct) + heading_bonus
    return score, matched_phrases, matched_terms


def retrieve_candidate_chunks(
    chunks: Sequence[SourceChunk],
    *,
    phrases: Sequence[str] = (),
    terms: Sequence[str] = (),
    top_k: int = 8,
    neighbor_radius: int = 1,
) -> list[CandidateChunk]:
    """Select lexical top_k, expand neighbors, and return candidates in source order.

    P/T are distinct normalized matching phrases/terms. Score is
    5*|P| + |T-P| + |P union T| + H, where H is at most one point for
    a match in a block of <=200 original text characters. Occurrence frequency
    beyond the first match adds nothing. Empty concepts and zero scores select
    nothing; word boundaries and whitespace normalization apply to both groups.

    Ties use input source order. direct_match candidates retain their lexical
    rank; neighbors have rank=None and keep their own real scores and matches,
    even when their lexical score is positive. Expansion uses input indices,
    clips at source ends, and does not recursively expand added neighbors.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise SourceRetrievalValidationError("top_k must be a positive integer")
    if not isinstance(neighbor_radius, int) or isinstance(neighbor_radius, bool) or neighbor_radius < 0:
        raise SourceRetrievalValidationError("neighbor_radius must be a non-negative integer")
    if (
        not isinstance(chunks, Sequence)
        or isinstance(chunks, (str, bytes))
        or any(not isinstance(chunk, SourceChunk) or not isinstance(chunk.chunk_id, str)
               or not chunk.chunk_id.strip() for chunk in chunks)
    ):
        raise SourceRetrievalValidationError("chunks must be a sequence of SourceChunk objects with non-blank IDs")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise SourceRetrievalValidationError("chunk IDs must be unique")
    phrase_patterns, term_patterns = _concept_patterns(phrases), _concept_patterns(terms)
    scores = [_score_chunk(chunk, phrase_patterns, term_patterns) for chunk in chunks]
    ranked = sorted((index for index, score in enumerate(scores) if score[0] > 0),
                    key=lambda index: (-scores[index][0], index))[:top_k]
    ranks = {index: rank for rank, index in enumerate(ranked, start=1)}
    selected = set(ranked)
    for index in ranked:
        selected.update(range(max(0, index - neighbor_radius), min(len(chunks), index + neighbor_radius + 1)))
    return [CandidateChunk(
        chunk_id=chunks[index].chunk_id,
        score=scores[index][0],
        matched_phrases=scores[index][1],
        matched_terms=scores[index][2],
        block_ids=tuple(block.block_id for block in chunks[index].blocks),
        match_type="direct_match" if index in ranks else "neighbor",
        rank=ranks.get(index),
    ) for index in sorted(selected)]
