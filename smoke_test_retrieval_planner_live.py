"""Single live Planner -> lexical retrieval -> reranker benchmark, no extraction."""

import json
from pathlib import Path

from dotenv import load_dotenv

from chunk_reranker import ChunkRerankerCandidate, ChunkRerankerError, DeepSeekChunkRerankerBackend
from retrieval_planner import DeepSeekRetrievalPlannerBackend, RetrievalPlannerError, SourceContext
from schemas import ResearchTask
from sec_research_tool import SECResearchTool, SECResearchToolError
from source_retrieval import retrieve_candidate_chunks
from source_segmentation import build_source_blocks, chunk_source_blocks, render_chunk


def print_candidate(candidate):
    print(f"chunk_id: {candidate.chunk_id}")
    print(f"lexical_score: {candidate.score}")
    print(f"matched_phrases: {list(candidate.matched_phrases)}")
    print(f"matched_terms: {list(candidate.matched_terms)}")


def main():
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    task = ResearchTask.model_validate_json((project_root / "examples" / "planet_growth.json").read_text(encoding="utf-8"))
    tool = SECResearchTool()
    filings = [result for result in tool.search("PL") if result["source_type"] == "10-Q"]
    if not filings:
        raise RuntimeError("SEC search returned no recent 10-Q for PL")
    filing = max(filings, key=lambda result: result["filing_date"])
    source_context = SourceContext(title=filing["title"], source_type=filing["source_type"], publisher="SEC / Planet Labs")
    planner = DeepSeekRetrievalPlannerBackend()
    plan = planner.plan(task, source_context)
    print("=== GENERATED CONCEPTS ===")
    print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2), flush=True)

    # Only retrieval/reranking receives source text; Planner has already returned.
    material = tool.read(filing["source_ref"])
    blocks = build_source_blocks(material["content"])
    chunks = chunk_source_blocks(blocks)
    direct = retrieve_candidate_chunks(chunks, phrases=plan.phrases, terms=plan.terms, top_k=8, neighbor_radius=0)
    assert all(candidate.match_type == "direct_match" for candidate in direct)
    direct.sort(key=lambda candidate: candidate.rank)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    lexical_by_id = {candidate.chunk_id: candidate for candidate in direct}

    print(f"source_ref: {material['source_ref']}")
    print(f"title: {material['title']}")
    print(f"chunk_count: {len(chunks)}")
    print(f"direct_candidate_count: {len(direct)}")
    print("\n=== LEXICAL DIRECT CANDIDATES ===")
    for candidate in direct:
        print(f"rank: {candidate.rank}")
        print_candidate(candidate)
        print()

    candidates = [ChunkRerankerCandidate(by_id[candidate.chunk_id], candidate) for candidate in direct]
    reranker = DeepSeekChunkRerankerBackend()
    result = reranker.select(task, candidates, max_selected=4)
    print(f"selected_count: {len(result.selected)}")
    print("\n=== SELECTED CHUNKS (MODEL RELEVANCE ORDER) ===")
    for rank, selected in enumerate(result.selected, start=1):
        print(f"rank: {rank}")
        print_candidate(lexical_by_id[selected.chunk_id])
        print(f"LLM reason: {selected.reason}")
        print("preview:")
        print(render_chunk(by_id[selected.chunk_id])[:1200])
        print()
    print("uncovered_topics:")
    print(json.dumps(result.uncovered_topics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RetrievalPlannerError, ChunkRerankerError, SECResearchToolError) as exc:
        print(f"Failure: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
