"""Consume a saved selection through Stage 1, without executing earlier stages.

The caller supplies the already routed tool name and saved candidate metadata.
The returned ResearchResult is an object delta plus extraction diagnostics, not
a new requirement coverage evaluation. Audit persistence belongs to the caller.
"""

from pathlib import Path

from pydantic import ValidationError

from llm_extractor import ExtractionBackend
from next_search_intent import NextSearchIntent
from research_agent import ResearchAgent, ResearchAgentError
from run_source_store import RunSourceStore
from schemas import ResearchResult, ResearchTask
from source_selection import CandidateSourceSelection
from storage import ResearchStorage
from tool_registry import ResearchToolRegistry


def acquire_selected_sources(
    task: ResearchTask,
    intent: NextSearchIntent,
    selection: CandidateSourceSelection,
    registry: ResearchToolRegistry,
    storage: ResearchStorage,
    source_store: RunSourceStore,
    extraction_backend: ExtractionBackend,
    *,
    tool_name: str,
    selected_source_metadata: list[dict],
    prompt_path: str | Path = Path(__file__).resolve().parent / "prompts" / "research_agent.md",
) -> ResearchResult:
    """Directly acquire selected sources in order; never search or re-select.

    Task query construction stays authoritative; intent search terms supplement
    its lexical concepts with the same Top-8/no-neighbor retrieval configuration.
    """
    validated = []
    for value, expected in ((task, ResearchTask), (intent, NextSearchIntent), (selection, CandidateSourceSelection)):
        if type(value) is not expected:
            raise ResearchAgentError(f"Selected-source acquisition requires {expected.__name__}")
        try:
            validated.append(expected.model_validate(value.model_dump(warnings=False)))
        except ValidationError as exc:
            raise ResearchAgentError(f"Invalid {expected.__name__}: {exc}") from exc
    task, intent, selection = validated
    if intent.requirement_id != task.target_requirement.requirement_id:
        raise ResearchAgentError("Intent requirement_id does not match the Target Requirement")
    tool = registry.get(tool_name)
    if "read" not in tool.spec.capabilities:
        raise ResearchAgentError(f"Selected tool {tool_name!r} does not support read")
    agent = ResearchAgent(tool, storage, prompt_path, extraction_backend=extraction_backend, source_store=source_store)
    return agent.run_known_sources(
        task,
        [candidate.source_ref for candidate in selection.selected],
        candidate_metadata=selected_source_metadata,
        retrieval_terms=intent.search_terms,
    )
