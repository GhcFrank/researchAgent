"""Direct acquisition through real Stage 1 components, using offline read/extract fakes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import research_agent as agent_module
from llm_extractor import ExtractionBackend, ExtractionResult, ExtractionValidationError
from next_search_intent import NextSearchIntent, NextSearchIntentGenerator
from requirement_coverage import RequirementCoverageEvaluator
from research_agent import ResearchAgentError
from research_tools import ResearchTool, ResearchToolSpec
from run_source_store import RunSourceStore, RunSourceStoreError
from run_workspace import ResearchRunWorkspace
from schemas import Claim, Entity, Estimate, Event, Evidence, Gap, ResearchTask, Source, SourceOrigin, Variable
from selected_source_acquisition import acquire_selected_sources
from source_segmentation import build_source_blocks, chunk_source_blocks
from source_selection import CandidateSourceSelection, CandidateSourceSelector
from storage import ResearchStorage
from tool_registry import ResearchToolRegistry
from tool_routing import ToolRouter


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "research_agent.md"
OBJECT_TYPES = (Entity, Source, Evidence, Variable, Claim, Gap, Estimate, Event)


class ReadTool(ResearchTool):
    spec = ResearchToolSpec("fixture_reader", "Read supplied offline material.", ("search", "read"), ("8-K",))

    def __init__(self, materials):
        self.materials = {item["source_ref"]: deepcopy(item) for item in materials}
        self.search_calls = []
        self.read_calls = []

    def _search(self, query):
        self.search_calls.append(query)
        pytest.fail("Selected-source acquisition must not search")

    def _read(self, source_ref):
        self.read_calls.append(source_ref)
        return deepcopy(self.materials[source_ref])


class RecordingBackend(ExtractionBackend):
    def __init__(self, *, before_extract=None, failure=None, empty=False):
        self.calls = []
        self.before_extract = before_extract
        self.failure = failure
        self.empty = empty

    def extract(self, task, material, operating_rules, *, source_blocks=None):
        if self.before_extract is not None:
            self.before_extract(material, source_blocks)
        self.calls.append((material["source_ref"], tuple(source_blocks)))
        block = source_blocks[0]
        if self.failure is not None and block.block_id == "B003":
            raise self.failure
        if self.empty:
            return ExtractionResult()
        renewal = "Renewal count" in block.text
        statement = "Renewal count was 65 customers." if renewal else "Data + Analytics revenue was USD 60 million in FY27 Q2."
        value, unit, scope, period = (65, "customers", "Renewals", None) if renewal else (60, "million USD", "Data + Analytics", "FY27 Q2")
        return ExtractionResult.model_validate_json(json.dumps({
            "entities": [{"entity_type": "company", "canonical_name": "Planet Labs PBC", "ticker": "PL"}],
            "evidence": [{
                "statement": statement, "value": value, "unit": unit, "period": period, "scope": scope,
                "entity_names": ["Planet Labs PBC"], "evidence_type": "Reported Fact", "source_locator": block.block_id,
            }],
            "variables": [{
                "name": f"Source-stated measurement in {material['source_ref']} {block.block_id}",
                "variable_type": "customer_count" if renewal else "revenue", "entity_name": "Planet Labs PBC",
                "value": value, "unit": unit, "period": period, "scope": scope,
                "input_type": "Observed", "evidence_indexes": [0],
            }],
        }))


@pytest.fixture
def inputs():
    task = ResearchTask.model_validate_json((ROOT / "examples" / "planet_growth.json").read_text(encoding="utf-8"))
    intent = NextSearchIntent.model_validate_json(json.dumps({
        "requirement_id": task.target_requirement.requirement_id,
        "target_aspects": ["Data + Analytics growth and revenue contribution in FY27 Q2"],
        "search_question": "What were business growth and revenue contribution in FY27 Q2?",
        "search_terms": ["Data + Analytics", "revenue", "Renewal count"],
        "preferred_source_types": task.preferred_source_types, "rationale": "Find direct business disclosure.",
    }))
    return task, intent


@pytest.fixture
def state(tmp_path):
    workspace = ResearchRunWorkspace.create("mock-acquisition", "company", root_dir=tmp_path / "runs")
    return ResearchStorage(workspace.objects_dir), RunSourceStore(workspace)


@pytest.fixture(autouse=True)
def forbid_earlier_stages(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Acquisition must not invoke earlier workflow stages")

    monkeypatch.setattr(RequirementCoverageEvaluator, "evaluate", forbidden)
    monkeypatch.setattr(NextSearchIntentGenerator, "generate", forbidden)
    monkeypatch.setattr(ToolRouter, "route", forbidden)
    monkeypatch.setattr(ToolRouter, "execute_search", forbidden)
    monkeypatch.setattr(CandidateSourceSelector, "select", forbidden)


def material(ref="selected-a", *, long=False):
    statements = ["Data + Analytics revenue was USD 60 million in FY27 Q2."]
    if long:
        statements += ["Renewal count was 65 customers.", statements[0]]
    paragraphs = [statement + (" neutral filler" * 350 if long else "") for statement in statements]
    content = "\n\n".join(paragraphs)
    return {
        "source_ref": ref, "title": f"Synthetic selected filing {ref}", "source_type": "8-K",
        "publisher": "Mock issuer", "published_date": "2026-09-03", "locator": f"mock://{ref}",
        "primary_or_secondary": "Primary", "independence_group": f"mock-origin-{ref}",
        "content": content, "raw_content": f"<html>{content}</html>",
    }


def metadata(items):
    return [{key: value for key, value in item.items() if key not in ("content", "raw_content")} for item in items]


def selected(items):
    return CandidateSourceSelection.model_validate_json(json.dumps({
        "selected": [{"source_ref": item["source_ref"], "rationale": "Read this eligible source."} for item in items],
        "overall_rationale": "Saved selection; do not select again.",
    }))


def acquire(inputs, state, tool, backend, items, *, selected_metadata=None):
    registry = ResearchToolRegistry()
    registry.register(tool)
    return acquire_selected_sources(
        *inputs, selected(items), registry, *state, backend,
        tool_name=tool.spec.name, selected_source_metadata=metadata(items) if selected_metadata is None else selected_metadata,
        prompt_path=PROMPT,
    )


def assert_no_objects(storage):
    for kind in OBJECT_TYPES:
        assert storage.list_objects(kind) == []


def snapshot(storage):
    return {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}


def test_selected_direct_read_reuses_stage1_per_chunk_persistence_lineage_and_rerun(inputs, state, monkeypatch):
    task, intent = inputs
    storage, source_store = state
    item = material(long=True)
    blocks = build_source_blocks(item["content"])
    chunks = chunk_source_blocks(blocks)
    assert len(chunks) == 3
    tool = ReadTool([item])
    backend = RecordingBackend()
    retrieve = agent_module.retrieve_candidate_chunks
    retrieval_calls = []

    def record_retrieve(supplied_chunks, **kwargs):
        assert supplied_chunks == chunks
        assert kwargs["top_k"] == 8 and kwargs["neighbor_radius"] == 0
        assert "Renewal count" in kwargs["terms"]
        result = retrieve(supplied_chunks, **kwargs)
        retrieval_calls.append([candidate.chunk_id for candidate in result])
        return result

    monkeypatch.setattr(agent_module, "retrieve_candidate_chunks", record_retrieve)

    def check_incremental(material, supplied_blocks):
        assert source_store.get(material["source_ref"])["content"] == material["content"]
        previous_calls = len(backend.calls)
        assert len(storage.list_objects(Evidence)) == previous_calls
        assert tuple(supplied_blocks) == chunks[previous_calls].blocks
        if previous_calls:
            assert len(storage.list_objects(Entity)) == len(storage.list_objects(Source)) == 1
            assert len(storage.list_objects(Variable)) == previous_calls

    backend.before_extract = check_incremental
    result = acquire(inputs, state, tool, backend, [item])
    assert tool.search_calls == [] and tool.read_calls == [item["source_ref"]]
    assert retrieval_calls == [["C001", "C002", "C003"]]
    assert [tuple(block.block_id for block in supplied) for _, supplied in backend.calls] == [("B001",), ("B002",), ("B003",)]
    assert len(result.sources_created) == len(result.entities_created) == 1
    assert len(result.evidence_created) == 3 and len(result.variables_created) == 2
    source = result.sources_created[0]
    assert source.locator == item["locator"] and source.primary_or_secondary is SourceOrigin.PRIMARY
    by_locator = {block.block_id: block.text for block in blocks}
    for ev in result.evidence_created:
        assert ev.source_id == source.source_id and ev.statement in by_locator[ev.source_locator]
        assert ev.entity_ids == [result.entities_created[0].entity_id]
    observed = {obj.variable_type: obj for obj in storage.list_objects(Variable)}
    assert observed["revenue"].evidence_ids == [result.evidence_created[0].evidence_id, result.evidence_created[2].evidence_id]
    assert observed["customer_count"].evidence_ids == [result.evidence_created[1].evidence_id]
    for variable in observed.values():
        assert all(storage.get_by_id(Evidence, eid).value == variable.value for eid in variable.evidence_ids)
    for kind in (Claim, Gap, Estimate, Event):
        assert storage.list_objects(kind) == []

    before = snapshot(storage)
    rerun = acquire(inputs, state, tool, backend, [item])
    assert tool.read_calls == [item["source_ref"]] and tool.search_calls == []
    assert len(backend.calls) == 3 and len(retrieval_calls) == 1
    assert rerun.sources_reused == [source.source_id]
    assert not rerun.sources_created and not rerun.entities_created
    assert not rerun.evidence_created and not rerun.variables_created
    assert snapshot(storage) == before


def test_new_source_consolidates_existing_observation_with_independent_lineage(inputs, state):
    storage, _ = state
    first, second = material("first"), material("second")
    tool, backend = ReadTool([first, second]), RecordingBackend()
    initial = acquire(inputs, state, tool, backend, [first])
    original = initial.variables_created[0]
    result = acquire(inputs, state, tool, backend, [second])
    assert len(storage.list_objects(Source)) == 2 and len(storage.list_objects(Entity)) == 1
    assert not result.variables_created and not result.entities_created
    consolidated = storage.list_objects(Variable)
    assert len(consolidated) == 1 and consolidated[0].variable_id == original.variable_id
    assert consolidated[0].evidence_ids == [initial.evidence_created[0].evidence_id, result.evidence_created[0].evidence_id]
    evidence = [storage.get_by_id(Evidence, eid) for eid in consolidated[0].evidence_ids]
    assert len({ev.source_id for ev in evidence}) == 2
    assert tool.search_calls == [] and tool.read_calls == [first["source_ref"], second["source_ref"]]


def test_selected_aliases_for_same_locator_do_not_download_twice(inputs, state):
    first, alias = material("first"), material("alias")
    alias["locator"] = first["locator"]
    tool, backend = ReadTool([first, alias]), RecordingBackend()
    result = acquire(inputs, state, tool, backend, [first, alias])
    assert tool.search_calls == [] and tool.read_calls == [first["source_ref"]]
    assert len(backend.calls) == 1
    assert len(result.sources_created) == len(result.evidence_created) == 1
    assert len(state[0].list_objects(Source)) == 1


def test_snapshot_failure_prevents_extraction_and_all_object_writes(inputs, state, monkeypatch):
    storage, source_store = state
    item = material()
    tool, backend = ReadTool([item]), RecordingBackend()
    failure = RunSourceStoreError("simulated snapshot failure")

    def fail_put(material):
        raise failure

    monkeypatch.setattr(source_store, "put", fail_put)
    with pytest.raises(ResearchAgentError, match="Source material persistence failed") as error:
        acquire(inputs, state, tool, backend, [item])
    assert error.value.__cause__ is failure
    assert tool.read_calls == [item["source_ref"]] and tool.search_calls == []
    assert backend.calls == []
    assert_no_objects(storage)


def test_later_chunk_failure_keeps_prior_objects_and_valid_snapshot(inputs, state):
    storage, source_store = state
    item = material(long=True)
    failure = ExtractionValidationError("simulated third chunk failure")
    backend, tool = RecordingBackend(failure=failure), ReadTool([item])
    with pytest.raises(ResearchAgentError, match="C003") as error:
        acquire(inputs, state, tool, backend, [item])
    assert error.value.__cause__ is failure and len(backend.calls) == 3
    reopened = ResearchStorage(storage.data_dir)
    assert len(reopened.list_objects(Source)) == len(reopened.list_objects(Entity)) == 1
    assert [ev.source_locator for ev in reopened.list_objects(Evidence)] == ["B001", "B002"]
    assert len(reopened.list_objects(Variable)) == 2
    assert source_store.get(item["source_ref"])["content"] == item["content"]
    for variable in reopened.list_objects(Variable):
        assert all(reopened.get_by_id(Evidence, eid) for eid in variable.evidence_ids)


@pytest.mark.parametrize("field", ["source_ref", "locator"])
def test_read_identity_mismatch_is_rejected_before_snapshot_or_extraction(inputs, state, field):
    storage, source_store = state
    item = material()
    tool, backend = ReadTool([item]), RecordingBackend()
    tool.materials[item["source_ref"]][field] = "mock://unexpected" if field == "locator" else "unexpected-ref"
    with pytest.raises(ResearchAgentError, match="source_ref/locator"):
        acquire(inputs, state, tool, backend, [item])
    assert not source_store.contains(item["source_ref"])
    assert backend.calls == []
    assert_no_objects(storage)


def test_metadata_dedup_collision_cannot_attach_blocks_to_another_locator(inputs, state):
    storage, source_store = state
    first, second = material("first"), material("second")
    second["title"] = first["title"]
    tool, backend = ReadTool([first, second]), RecordingBackend()
    with pytest.raises(ResearchAgentError, match="different locator"):
        acquire(inputs, state, tool, backend, [first, second])
    assert backend.calls == []
    assert source_store.contains(first["source_ref"]) and source_store.contains(second["source_ref"])
    assert_no_objects(storage)


@pytest.mark.parametrize("failure", ["intent-mismatch", "missing-candidate-metadata", "no-read-capability", "excluded-source"])
def test_acquisition_input_boundaries_prevent_any_read(inputs, state, failure):
    task, intent = inputs
    item = material()
    tool, backend = ReadTool([item]), RecordingBackend()
    selected_metadata = metadata([item])
    if failure == "intent-mismatch":
        intent.requirement_id = "OTHER-REQUIREMENT"
    elif failure == "missing-candidate-metadata":
        selected_metadata = []
    elif failure == "no-read-capability":
        tool.spec = ResearchToolSpec(tool.spec.name, tool.spec.description, ("search",), ("8-K",))
    else:
        task.constraints.excluded_sources = [item["locator"].upper()]
    with pytest.raises(ResearchAgentError):
        acquire((task, intent), state, tool, backend, [item], selected_metadata=selected_metadata)
    assert tool.search_calls == tool.read_calls == [] and backend.calls == []
    assert_no_objects(state[0])


@pytest.mark.parametrize("already_acquired", ["snapshot-only", "formal-source-only"])
def test_existing_snapshot_or_source_is_skipped_without_resume(inputs, state, already_acquired):
    storage, source_store = state
    item = material()
    existing = None
    if already_acquired == "snapshot-only":
        source_store.put(item)
    else:
        existing = storage.insert(Source(
            source_id="existing-selected-source", title=item["title"], source_type=item["source_type"],
            publisher=item["publisher"], published_date=item["published_date"], accessed_date="2026-10-08",
            locator=item["locator"], primary_or_secondary=SourceOrigin.PRIMARY,
        ))
    before = snapshot(storage)
    tool, backend = ReadTool([item]), RecordingBackend()
    result = acquire(inputs, state, tool, backend, [item])
    assert tool.search_calls == tool.read_calls == [] and backend.calls == []
    assert not result.sources_created and not result.evidence_created and not result.variables_created
    assert result.sources_reused == ([existing.source_id] if existing else [])
    assert "Skipped already acquired" in result.research_notes and "resume" in result.research_notes
    assert snapshot(storage) == before


def test_source_with_no_extracted_facts_is_still_snapshotted_and_persisted(inputs, state):
    storage, source_store = state
    item = material()
    tool, backend = ReadTool([item]), RecordingBackend(empty=True)
    result = acquire(inputs, state, tool, backend, [item])
    assert tool.read_calls == [item["source_ref"]] and tool.search_calls == []
    assert len(backend.calls) == 1 and len(result.sources_created) == 1
    assert source_store.contains(item["source_ref"])
    assert not result.entities_created and not result.evidence_created and not result.variables_created
    assert len(storage.list_objects(Source)) == 1
    for kind in (Entity, Evidence, Variable, Claim, Gap, Estimate, Event):
        assert storage.list_objects(kind) == []
