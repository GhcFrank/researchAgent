"""One combined Top-8 extraction using the production default output budget."""

import json
from pathlib import Path

from dotenv import load_dotenv

from llm_extractor import DeepSeekExtractionBackend, ExtractionError
from schemas import ResearchTask
from sec_research_tool import SECResearchTool, SECResearchToolError
from source_retrieval import retrieve_candidate_chunks
from source_segmentation import build_source_blocks, chunk_source_blocks, render_blocks


# These are the previous smoke benchmark's concepts and input checks.
PHRASES = (
    "revenue growth",
    "results of operations",
    "remaining performance obligations",
    "government customers",
    "subscription services",
    "backlog",
)
TERMS = ("revenue", "growth", "backlog", "government", "subscription", "analytics")
EXPECTED_SOURCE_REF = "sec:0001836833:0001193125-26-382016:pl-20260731.htm"
EXPECTED_CHUNK_IDS = ("C001", "C015", "C018", "C021", "C025", "C030", "C034", "C045")
EXPECTED_BLOCK_COUNT = 225
EXPECTED_RENDERED_CHARS = 62644


def print_response_diagnostics(response):
    """Whitelist response termination and usage fields only."""
    if response is None:
        print("response_diagnostics: unavailable (no SDK response returned)")
        return
    for field in ("status", "incomplete_details"):
        value = getattr(response, field, None)
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        print(f"response.{field}: {json.dumps(value, ensure_ascii=False)}")
    usage = getattr(response, "usage", None)
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        print(f"{field}: {getattr(usage, field, None)}")
    details = getattr(usage, "output_tokens_details", None)
    print(f"reasoning_tokens: {getattr(details, 'reasoning_tokens', None)}")


def print_fields(candidate, fields):
    values = candidate.model_dump(mode="json")
    for field in fields:
        print(f"{field}: {json.dumps(values[field], ensure_ascii=False)}")


def main():
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    task = ResearchTask.model_validate_json(
        (project_root / "examples" / "planet_growth.json").read_text(encoding="utf-8")
    )
    rules = (project_root / "prompts" / "research_agent.md").read_text(encoding="utf-8")
    tool = SECResearchTool()
    filing = next((item for item in tool.search("PL") if item["source_ref"] == EXPECTED_SOURCE_REF), None)
    if filing is None:
        print("input_validation: FAILED; the previous benchmark filing is not in SEC search results")
        return 1
    if (filing["source_type"], filing["filing_date"], filing["report_date"]) != (
        "10-Q", "2026-09-03", "2026-07-31"
    ):
        print("input_validation: FAILED; filing type or dates differ from the previous benchmark")
        return 1
    material = tool.read(filing["source_ref"])
    blocks = build_source_blocks(material["content"])
    chunks = chunk_source_blocks(blocks)
    candidates = retrieve_candidate_chunks(
        chunks, phrases=PHRASES, terms=TERMS, top_k=8, neighbor_radius=0
    )
    actual_ids = tuple(candidate.chunk_id for candidate in candidates)
    print(f"source_ref: {material['source_ref']}")
    print(f"title: {material['title']}")
    print(f"filing_date: {filing['filing_date']}")
    print(f"report_date: {filing['report_date']}")
    print(f"top_8_chunk_ids: {list(actual_ids)}")
    if actual_ids != EXPECTED_CHUNK_IDS:
        print("retrieval_validation: FAILED; Top 8 differs from the previous benchmark")
        return 1
    assert all(candidate.match_type == "direct_match" for candidate in candidates)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for candidate in candidates:
        assert candidate.block_ids == tuple(block.block_id for block in by_id[candidate.chunk_id].blocks)

    wanted_ids = {block.block_id for candidate in candidates for block in by_id[candidate.chunk_id].blocks}
    # Select from the full source to deduplicate and preserve its original order.
    selected_blocks = [block for block in blocks if block.block_id in wanted_ids]
    assert len(selected_blocks) == len(wanted_ids)
    selected_chars = len(render_blocks(selected_blocks))
    print("retrieval_validation: PASS")
    print(f"selected_chunk_count: {len(candidates)}")
    print(f"selected_block_count: {len(selected_blocks)}")
    print(f"selected_rendered_chars: {selected_chars}")
    if len(selected_blocks) != EXPECTED_BLOCK_COUNT or selected_chars != EXPECTED_RENDERED_CHARS:
        print("input_validation: FAILED; selected scope differs from the previous benchmark")
        return 1
    print("input_validation: PASS")

    backend = DeepSeekExtractionBackend()
    assert backend.max_output_tokens == 32768, "Production output budget differs from this benchmark"
    print(f"Model: {backend.model}")
    print(f"max_output_tokens: {backend.max_output_tokens}")
    original_create = backend.client.responses.create
    captured_response = None
    request_count = 0

    def recording_create(*args, **kwargs):
        nonlocal captured_response, request_count
        request_count += 1
        assert request_count == 1, "This smoke test permits exactly one extraction request"
        captured_response = original_create(*args, **kwargs)
        return captured_response

    # Observe the same response before production validation, without changing
    # request arguments, prompt, reasoning, timeout, retries, or return value.
    backend.client.responses.create = recording_create
    print("extraction_calls: 1", flush=True)
    try:
        result = backend.extract(task, material, rules, source_blocks=selected_blocks)
    except ExtractionError as exc:
        print("status: FAILED")
        print(f"exception_type: {type(exc).__name__}")
        print(f"exception_message: {exc}")
        if exc.__cause__ is not None:
            print(f"exception_cause_type: {type(exc.__cause__).__name__}")
        print_response_diagnostics(captured_response)
        return 1

    print("status: SUCCESS")
    print_response_diagnostics(captured_response)
    for field, label in (
        ("evidence", "evidence_count"), ("variables", "variable_count"),
        ("potential_conflicts", "potential_conflicts_count"), ("not_found", "not_found_count"),
        ("candidate_gaps", "candidate_gaps_count"), ("follow_up_candidates", "follow_up_candidates_count"),
    ):
        print(f"{label}: {len(getattr(result, field))}")
    assert all(evidence.source_locator in wanted_ids for evidence in result.evidence)
    print("locator_validation: PASS")
    assert all(
        0 <= index < len(result.evidence)
        for variable in result.variables for index in variable.evidence_indexes
    )
    print("variable_evidence_indexes_validation: PASS")

    print("\n=== EVIDENCE (FIRST 10 AT MOST) ===")
    for index, evidence in enumerate(result.evidence[:10]):
        print(f"[{index}]")
        print_fields(evidence, ("statement", "value", "unit", "period", "scope", "source_locator"))
        print()
    print("\n=== SOURCE SPOT CHECKS (FIRST 5 AT MOST) ===")
    source_by_id = {block.block_id: block for block in selected_blocks}
    for index, evidence in enumerate(result.evidence[:5]):
        print(f"[{index}] {evidence.statement}")
        print(f"-> {evidence.source_locator}")
        print(f"-> {source_by_id[evidence.source_locator].text[:500]}")
        print()
    print("\n=== VARIABLES (FIRST 10 AT MOST) ===")
    for index, variable in enumerate(result.variables[:10]):
        print(f"[{index}]")
        print_fields(variable, (
            "name", "variable_type", "value", "unit", "period", "scope", "input_type", "evidence_indexes"
        ))
        print()
    for field in ("not_found", "candidate_gaps", "follow_up_candidates"):
        print(f"\n=== {field.upper()} ===")
        for candidate in getattr(result, field):
            print(json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExtractionError, SECResearchToolError) as exc:
        print(f"Failure: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
