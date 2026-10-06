"""Manual SEC candidate retrieval observation; no extraction or persistence."""

from pathlib import Path

from dotenv import load_dotenv

from sec_research_tool import SECResearchTool
from source_retrieval import retrieve_candidate_chunks
from source_segmentation import build_source_blocks, chunk_source_blocks, render_chunk


PHRASES = (
    "revenue growth",
    "results of operations",
    "remaining performance obligations",
    "government customers",
    "subscription services",
    "backlog",
)
TERMS = ("revenue", "growth", "backlog", "government", "subscription", "analytics")


def print_candidate(candidate, chunk):
    print(f"rank: {candidate.rank if candidate.rank is not None else 'null'}")
    print(f"match_type: {candidate.match_type}")
    print(f"chunk_id: {candidate.chunk_id}")
    print(f"score: {candidate.score}")
    print(f"matched_phrases: {list(candidate.matched_phrases)}")
    print(f"matched_terms: {list(candidate.matched_terms)}")
    print(f"block_ids: {list(candidate.block_ids)}")
    print("preview:")
    print(render_chunk(chunk)[:1000])
    print()


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    tool = SECResearchTool()
    results = tool.search("PL")
    filings = [result for result in results if result["source_type"] == "10-Q"]
    if not filings:
        raise RuntimeError("SEC search returned no recent 10-Q for PL")
    selected = max(filings, key=lambda filing: filing["filing_date"])
    material = tool.read(selected["source_ref"])
    blocks = build_source_blocks(material["content"])
    chunks = chunk_source_blocks(blocks)
    assert blocks and chunks, "SEC filing has no non-empty normalized source blocks"
    candidates = retrieve_candidate_chunks(chunks, phrases=PHRASES, terms=TERMS, top_k=8, neighbor_radius=1)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    assert len({candidate.chunk_id for candidate in candidates}) == len(candidates), "Duplicate candidates"
    for candidate in candidates:
        assert candidate.block_ids == tuple(block.block_id for block in by_id[candidate.chunk_id].blocks)
    direct = sorted((candidate for candidate in candidates if candidate.match_type == "direct_match"),
                    key=lambda candidate: candidate.rank)
    neighbors = [candidate for candidate in candidates if candidate.match_type == "neighbor"]

    print(f"source_ref: {material['source_ref']}")
    print(f"title: {material['title']}")
    print(f"normalized_content_length: {len(material['content'])}")
    print(f"block_count: {len(blocks)}")
    print(f"chunk_count: {len(chunks)}")
    print("top_k: 8")
    print("neighbor_radius: 1")
    print(f"candidate_count: {len(candidates)}")
    print(f"direct_match_count: {len(direct)}")
    print(f"neighbor_count: {len(neighbors)}")
    print(f"candidate_source_order: {[candidate.chunk_id for candidate in candidates]}")
    print()
    print("=== DIRECT MATCHES (LEXICAL RANK) ===")
    for candidate in direct:
        print_candidate(candidate, by_id[candidate.chunk_id])
    print("=== NEIGHBOR CANDIDATES (SOURCE ORDER) ===")
    for candidate in neighbors:
        print_candidate(candidate, by_id[candidate.chunk_id])


if __name__ == "__main__":
    main()
