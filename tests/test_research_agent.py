"""Research Agent business boundaries and the complete offline data flow."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import research_agent as agent_module
from llm_extractor import ExtractionBackend, ExtractionError, ExtractionResult, ExtractionValidationError
from research_agent import AgentPermissionError, ResearchAgent, ResearchAgentError
from research_tools import MockResearchTool, ResearchToolError
from run_source_store import RunSourceStore, RunSourceStoreError
from run_workspace import ResearchRunWorkspace
from schemas import (
    Claim,
    ClaimEvidenceStatus,
    Entity,
    Estimate,
    Event,
    Evidence,
    Gap,
    GapStatus,
    ResearchResult,
    ResearchTask,
    SearchMode,
    Source,
    SourceOrigin,
    Variable,
    VariableInputType,
)
from source_segmentation import build_source_blocks, chunk_source_blocks
from storage import ResearchStorage


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "research_agent.md"


class RecordingTool(MockResearchTool):
    def __init__(self, fixture_path=None):
        if fixture_path is None:
            super().__init__()
        else:
            super().__init__(fixture_path)
        self.queries = []
        self.read_refs = []

    def search(self, query):
        self.queries.append(query)
        return super().search(query)

    def read(self, source_ref):
        self.read_refs.append(source_ref)
        return super().read(source_ref)


class FakeExtractionBackend(ExtractionBackend):
    def __init__(self, result, outcomes=None):
        self.result = result
        self.outcomes = outcomes or {}
        self.calls = []

    def extract(self, task, material, operating_rules, *, source_blocks=None):
        self.calls.append({
            "task": task, "material": deepcopy(material), "operating_rules": operating_rules,
            "source_blocks": source_blocks,
        })
        chunk_key = (material["source_ref"], source_blocks[0].block_id) if source_blocks else None
        outcome = self.outcomes.get(chunk_key, self.outcomes.get(material["source_ref"], self.result))
        if isinstance(outcome, ExtractionError):
            raise outcome
        return outcome.model_copy(deep=True)


@pytest.fixture
def task():
    return ResearchTask.model_validate_json((ROOT / "examples" / "planet_growth.json").read_text(encoding="utf-8"))


@pytest.fixture
def storage(tmp_path):
    return ResearchStorage(tmp_path / "data")


@pytest.fixture
def workspace(tmp_path):
    return ResearchRunWorkspace.create("PL", "company", root_dir=tmp_path / "runs")


@pytest.fixture
def workspace_storage(workspace):
    return ResearchStorage(workspace.objects_dir)


@pytest.fixture
def source_store(workspace):
    return RunSourceStore(workspace)


@pytest.fixture
def tool():
    return RecordingTool()


@pytest.fixture
def agent(tool, storage):
    return ResearchAgent(tool, storage, PROMPT)


@pytest.fixture
def materials():
    return json.loads((ROOT / "fixtures" / "mock_planet_sources.json").read_text(encoding="utf-8"))


@pytest.fixture
def backend_result():
    return ExtractionResult.model_validate_json(json.dumps({
        "entities": [{
            "entity_type": "company", "canonical_name": "Planet Labs PBC",
            "aliases": ["Planet Labs"], "ticker": "PL", "geography": "Global",
        }],
        "evidence": [
            {"statement": "Total revenue was USD 80 million.", "value": 80, "unit": "million USD",
             "entity_names": ["Planet Labs PBC"], "period": "FY27 Q2", "scope": "Total company",
             "evidence_type": "Reported Fact", "source_locator": "B002", "notes": "Source-stated total."},
            {"statement": "Data + Analytics revenue was USD 60 million.", "value": 60, "unit": "million USD",
             "entity_names": ["Planet Labs PBC"], "period": "FY27 Q2", "scope": "Data + Analytics",
             "evidence_type": "Reported Fact", "source_locator": "B002"},
        ],
        "variables": [
            {"name": "Total revenue (FY27 Q2)", "definition": "Source-stated company revenue.",
             "variable_type": "revenue", "entity_name": "Planet Labs PBC", "value": 80,
             "unit": "million USD", "period": "FY27 Q2", "scope": "Total company",
             "input_type": "Observed", "evidence_indexes": [0]},
            {"name": "Data + Analytics revenue (FY27 Q2)", "definition": "Source-stated business revenue.",
             "variable_type": "revenue", "entity_name": "Planet Labs PBC", "value": 60,
             "unit": "million USD", "period": "FY27 Q2", "scope": "Data + Analytics",
             "input_type": "Observed", "evidence_indexes": [1]},
        ],
        "potential_conflicts": [{"description": "An unresolved source qualification.", "evidence_ids": []}],
        "not_found": [{"item": "Customer breakdown", "result": "Not disclosed in this source."}],
        "candidate_gaps": [{"question": "What is the customer breakdown?", "why_it_matters": "Revenue context."}],
        "follow_up_candidates": [{"topic": "Customer breakdown", "reason": "Find a direct disclosure."}],
        "research_notes": "Fake extraction; no inference or calculations.",
    }))


def fixture_tool(tmp_path, records):
    path = tmp_path / "raw_materials.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return RecordingTool(path)


def assert_storage_empty(storage):
    """Storage infrastructure may exist; no Research Objects may be persisted."""
    for object_type in (Entity, Source, Evidence, Variable, Claim, Gap, Estimate, Event):
        assert storage.list_objects(object_type) == [], f"Unexpected persisted {object_type.__name__}"


def test_end_to_end_references_permissions_and_save_order(agent, task, tool, storage, materials, monkeypatch):
    calls = []
    insert = storage.insert

    def record_insert(obj):
        calls.append(type(obj))
        return insert(obj)

    monkeypatch.setattr(storage, "insert", record_insert)
    result = agent.run(task)
    assert isinstance(result, ResearchResult)
    assert agent.prompt == PROMPT.read_text(encoding="utf-8")
    assert result.entities_created and result.sources_created
    assert result.evidence_created and result.variables_created
    assert len(tool.queries) == 1
    assert {"planet", "labs", "data", "analytics", "fy27", "q2"} <= set(tool.queries[0].split())
    assert tool.read_refs == ["mock-src-001", "mock-src-002", "mock-src-004"]
    ranks = {Entity: 0, Source: 1, Evidence: 2, Variable: 3}
    assert [ranks[kind] for kind in calls] == sorted(ranks[kind] for kind in calls)
    assert len({source.independence_group for source in result.sources_created}) == 1
    assert result.not_found == []
    assert result.search_coverage.primary_source_found
    assert result.search_coverage.period_covered and result.search_coverage.scope_covered

    raw_by_locator = {item["locator"]: item for item in materials}
    for item in result.evidence_created:
        source = storage.get_by_id(Source, item.source_id)
        assert item.statement in raw_by_locator[source.locator]["content"]
        assert item.source_locator
        assert item.period == "FY27 Q2" and item.scope == "Data + Analytics"
        for entity_id in item.entity_ids:
            assert storage.get_by_id(Entity, entity_id).canonical_name == "Planet Labs PBC"
    for variable in result.variables_created:
        assert variable.input_type is VariableInputType.OBSERVED
        assert storage.get_by_id(Entity, variable.entity_id)
        for evidence_id in variable.evidence_ids:
            assert storage.get_by_id(Evidence, evidence_id).value == variable.value
    assert len(result.variables_created) == len(storage.list_objects(Variable)) == 3
    assert {item.name: (item.value, item.unit) for item in result.variables_created} == {
        "Data + Analytics revenue": (60, "million USD"),
        "Data + Analytics YoY revenue growth": (24, "%"),
        "Data + Analytics revenue contribution": (75, "%"),
    }
    assert any(item.value is None and "main growth contributor" in item.statement for item in result.evidence_created)
    assert result.follow_up_candidates
    for forbidden in (Claim, Gap, Estimate, Event):
        assert storage.list_objects(forbidden) == []


def test_rerun_reuses_objects_without_changing_storage(agent, task, storage):
    first = agent.run(task)
    before = {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}
    second = agent.run(task)
    assert second.sources_reused == [source.source_id for source in first.sources_created]
    assert not second.entities_created and not second.sources_created
    assert not second.evidence_created and not second.variables_created
    assert second.not_found == []
    assert {path.name: path.read_bytes() for path in storage.data_dir.iterdir()} == before


def test_existing_source_and_canonical_entity_ids_are_reused(agent, task, storage, materials):
    entity = storage.insert(Entity(entity_id="existing-company", entity_type="company", canonical_name="planet labs pbc"))
    raw = materials[0]
    source = storage.insert(Source(
        source_id="existing-release",
        title=raw["title"],
        publisher=raw["publisher"],
        source_type=raw["source_type"],
        published_date=raw["published_date"],
        accessed_date="2026-10-01",
        locator=raw["locator"],
        primary_or_secondary=SourceOrigin.PRIMARY,
    ))
    result = agent.run(task)
    assert result.entities_created == []
    assert source.source_id in result.sources_reused
    assert len(result.sources_created) == 2
    assert storage.list_objects(Entity) == [entity]
    assert all(item.entity_ids == [entity.entity_id] for item in result.evidence_created)
    assert any(item.source_id == source.source_id for item in result.evidence_created)
    assert storage.get_by_id(Source, source.source_id) == source


def forbidden_object(kind):
    timestamp = "2026-10-06T12:00:00Z"
    return {
        Claim: lambda: Claim(claim_id="forbidden", claim="A thesis", claim_type="Thesis", evidence_status=ClaimEvidenceStatus.EVIDENCE_GAP, last_updated=timestamp),
        Gap: lambda: Gap(gap_id="forbidden", question="A formal gap?", why_it_matters="Unknown", status=GapStatus.UNKNOWN, last_updated=timestamp),
        Estimate: lambda: Estimate(estimate_id="forbidden", output_variable_id="missing", formula="x * 2", reason_needed="Model output", calculated_date="2026-10-06"),
        Event: lambda: Event(event_id="forbidden", event_type="Earnings", status="Scheduled", description="A formal event", last_updated=timestamp),
    }[kind]()


@pytest.mark.parametrize("path", ["deterministic", "backend"])
@pytest.mark.parametrize("kind", [Claim, Gap, Estimate, Event], ids=lambda kind: kind.__name__)
def test_forbidden_objects_rejected_before_any_write(agent, task, storage, monkeypatch, kind, path, backend_result):
    if path == "backend":
        backend_result.variables = [forbidden_object(kind)]
        agent.extraction_backend = FakeExtractionBackend(backend_result)
    else:
        monkeypatch.setattr(agent_module, "_extract_variables", lambda evidence: [forbidden_object(kind)])
    with pytest.raises(AgentPermissionError, match=kind.__name__):
        agent.run(task)
    assert_storage_empty(storage)


@pytest.mark.parametrize("path", ["deterministic", "backend"])
@pytest.mark.parametrize("input_type", [VariableInputType.MODEL_ESTIMATE, VariableInputType.DERIVED])
def test_forbidden_variable_input_rejected_before_any_write(agent, task, storage, monkeypatch, input_type, path, backend_result):
    if path == "backend":
        backend_result.variables[0].input_type = input_type
        agent.extraction_backend = FakeExtractionBackend(backend_result)
    else:
        variable = Variable(
            variable_id="forbidden-variable", name="Forbidden model output", variable_type="Financial",
            input_type=input_type, last_updated="2026-10-06T12:00:00Z",
        )
        monkeypatch.setattr(agent_module, "_extract_variables", lambda evidence: [variable])
    with pytest.raises(AgentPermissionError, match="Forbidden Variable input_type"):
        agent.run(task)
    assert_storage_empty(storage)


def test_missing_figures_in_read_material_return_not_found(task, storage, tmp_path, materials):
    raw = deepcopy(materials[0])
    raw["content"] = "MOCK / FIXTURE DATA ONLY.\n\nPlanet Labs FY27 Q2: Data + Analytics revenue and growth figures were not disclosed."
    tool = fixture_tool(tmp_path, [raw])
    result = ResearchAgent(tool, storage, PROMPT).run(task)
    assert tool.read_refs == [raw["source_ref"]]
    assert result.sources_created and not result.evidence_created and not result.variables_created
    assert result.not_found and result.candidate_gaps
    assert all(item.item and item.search_attempted == tool.queries and item.result == "No qualifying evidence found" for item in result.not_found)
    assert not result.search_coverage.period_covered and not result.search_coverage.scope_covered
    assert storage.list_objects(Gap) == []


@pytest.mark.parametrize(
    "source_type, phrase, expected_type, input_type",
    [
        ("Earnings Call Transcript", "revenue guidance is", "Management Guidance", VariableInputType.GUIDANCE),
        ("Business Brief", "revenue estimate is", "Third-party Estimate", VariableInputType.THIRD_PARTY_ESTIMATE),
    ],
)
def test_guidance_and_third_party_estimates_keep_their_type(task, storage, tmp_path, materials, source_type, phrase, expected_type, input_type):
    raw = deepcopy(materials[0])
    raw["source_type"] = source_type
    raw["content"] = f"MOCK / FIXTURE DATA ONLY.\n\nFY27 Q2: Data + Analytics {phrase} USD 65 million."
    result = ResearchAgent(fixture_tool(tmp_path, [raw]), storage, PROMPT).run(task)
    assert len(result.evidence_created) == len(result.variables_created) == 1
    assert result.evidence_created[0].evidence_type == expected_type
    assert result.variables_created[0].input_type is input_type
    assert result.variables_created[0].value == 65


def test_instruction_and_exclusions_bound_one_search(agent, task, tool):
    task.specific_search_instruction = "sovereign"
    task.constraints.excluded_sources = ["mock-src-001"]
    result = agent.run(task)
    assert len(tool.queries) == 1 and "sovereign" in tool.queries[0]
    assert tool.read_refs == ["mock-src-002"]
    assert len(result.sources_created) == 1
    assert result.search_coverage.source_types_checked == ["Earnings Call Transcript"]


def test_unsupported_scope_limit_fails_before_retrieval(agent, task, tool, storage):
    task.constraints.max_search_scope = "Search all related business areas"
    with pytest.raises(ResearchAgentError, match="max_search_scope"):
        agent.run(task)
    assert tool.queries == [] and tool.read_refs == []
    assert_storage_empty(storage)


def test_counter_evidence_reports_conflict_without_formal_claim(task, storage, tmp_path, materials):
    first = deepcopy(materials[0])
    second = deepcopy(first)
    second.update(source_ref="mock-conflicting", title="[MOCK] Another FY27 Q2 release", locator="mock://conflicting/release")
    second["content"] = second["content"].replace("USD 60 million", "USD 61 million")
    tool = fixture_tool(tmp_path, [first, second])
    task.search_mode = SearchMode.COUNTER_EVIDENCE
    result = ResearchAgent(tool, storage, PROMPT).run(task)
    assert len(tool.queries) == 1
    assert result.potential_conflicts
    for conflict in result.potential_conflicts:
        values = {storage.get_by_id(Evidence, evidence_id).value for evidence_id in conflict.evidence_ids}
        assert values == {60, 61}
    assert storage.list_objects(Claim) == []
    assert "Counter-evidence mode" in result.research_notes


def test_other_period_is_not_substituted_for_requested_period(task, storage, tmp_path, materials):
    raw = deepcopy(materials[0])
    # The title still matches search, but the explicit body period is different.
    raw["content"] = raw["content"].replace("FY27 Q2", "FY28 Q2")
    tool = fixture_tool(tmp_path, [raw])
    result = ResearchAgent(tool, storage, PROMPT).run(task)
    assert tool.read_refs
    assert not result.evidence_created and not result.variables_created
    assert result.not_found


def test_backend_conversion_metadata_lineage_and_rerun(task, storage, tmp_path, materials, backend_result, monkeypatch):
    records = materials[:2]
    tool = fixture_tool(tmp_path, records)
    second_extraction = backend_result.model_copy(deep=True)
    for candidate in second_extraction.variables:
        candidate.name = f"Planet Labs PBC {candidate.name}"
        candidate.definition = "Alternate wording for the same observation."
        candidate.variable_type = " revenue "
        candidate.scope = f"  {candidate.scope.replace(' ', '   ')}  "
        candidate.period = "  FY27   Q2  "
        candidate.unit = "  million   USD  "
    backend = FakeExtractionBackend(backend_result, {records[1]["source_ref"]: second_extraction})
    agent = ResearchAgent(tool, storage, PROMPT, extraction_backend=backend)
    writes = []
    insert = storage.insert

    def record_insert(obj):
        writes.append((len(backend.calls), type(obj)))
        return insert(obj)

    monkeypatch.setattr(storage, "insert", record_insert)
    result = agent.run(task)
    assert [call["material"]["source_ref"] for call in backend.calls] == tool.read_refs
    for call, raw in zip(backend.calls, records):
        assert call["task"] == task
        assert call["material"] == raw
        assert call["operating_rules"] == PROMPT.read_text(encoding="utf-8")
        assert call["source_blocks"] == tuple(build_source_blocks(raw["content"]))
    assert len(tool.queries) == 1
    assert len(result.entities_created) == 1
    assert len(result.sources_created) == 2
    assert len(result.evidence_created) == 4
    assert len(result.variables_created) == len(storage.list_objects(Variable)) == 2
    entity = result.entities_created[0]
    assert (entity.canonical_name, entity.ticker, entity.geography) == ("Planet Labs PBC", "PL", "Global")
    ranks = {Entity: 0, Source: 1, Evidence: 2, Variable: 3}
    assert [call_number for call_number, _ in writes] == sorted(call_number for call_number, _ in writes)
    assert {call_number for call_number, _ in writes} == {1, 2}
    for call_number in (1, 2):
        batch_ranks = [ranks[kind] for number, kind in writes if number == call_number]
        assert batch_ranks == sorted(batch_ranks)
    for source, raw in zip(result.sources_created, records):
        for field in ("title", "publisher", "source_type", "published_date", "locator"):
            assert getattr(source, field) == raw[field]
        assert source.primary_or_secondary is SourceOrigin.PRIMARY
        evidence = [item for item in result.evidence_created if item.source_id == source.source_id]
        for item, candidate in zip(evidence, backend_result.evidence):
            assert item.source_locator == "B002" and item.source_locator != source.locator
            assert item.entity_ids == [entity.entity_id]
            assert item.model_dump(exclude={"evidence_id", "source_id", "entity_ids", "collected_at"}) == candidate.model_dump(exclude={"entity_names"})
    for item, candidate in zip(result.variables_created, backend_result.variables):
        assert item.entity_id == entity.entity_id
        expected_ids = []
        for source in result.sources_created:
            evidence = [ev for ev in result.evidence_created if ev.source_id == source.source_id]
            expected_ids.extend(evidence[index].evidence_id for index in candidate.evidence_indexes)
        assert item.evidence_ids == expected_ids
        sources = [storage.get_by_id(Source, storage.get_by_id(Evidence, eid).source_id)
                   for eid in item.evidence_ids]
        assert [source.source_id for source in sources] == [source.source_id for source in result.sources_created]
        assert all(source.independence_group for source in sources)
        assert item.input_type is VariableInputType.OBSERVED
        assert item.model_dump(exclude={"variable_id", "entity_id", "evidence_ids", "last_updated", "input_type"}) == candidate.model_dump(exclude={"entity_name", "evidence_indexes", "input_type"})
    assert result.search_coverage.primary_source_found
    assert result.search_coverage.period_covered and result.search_coverage.scope_covered
    assert result.potential_conflicts == backend_result.potential_conflicts
    assert all(not item.evidence_ids for item in result.potential_conflicts)
    assert all(item.search_attempted[:len(tool.queries)] == tool.queries for item in result.not_found)
    assert len(result.not_found) == 1
    assert result.not_found[0].search_attempted[len(tool.queries):] == [
        f"{record['source_ref']}#C001" for record in records
    ]
    assert "Chunk-local" in result.not_found[0].result
    assert result.candidate_gaps == backend_result.candidate_gaps
    assert result.follow_up_candidates == backend_result.follow_up_candidates
    assert backend_result.research_notes in result.research_notes
    assert result.research_notes.count(backend_result.research_notes) == 1
    for kind in (Claim, Gap, Estimate, Event):
        assert storage.list_objects(kind) == []
    assert ResearchStorage(storage.data_dir).list_objects(Variable) == result.variables_created

    before = {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}
    second = agent.run(task)
    assert second.sources_reused == [item.source_id for item in result.sources_created]
    assert not second.sources_created and not second.entities_created
    assert not second.evidence_created and not second.variables_created
    assert len(backend.calls) == 4
    assert storage.list_objects(Variable) == result.variables_created
    assert {path.name: path.read_bytes() for path in storage.data_dir.iterdir()} == before


@pytest.mark.parametrize(("field", "different"), [
    ("value", 65),
    ("value", 60.5),
    ("value", "60"),
    ("scope", "Total company"),
    ("period", "FY27 Q1"),
])
def test_distinct_observations_are_not_consolidated(task, storage, tmp_path, materials, backend_result, field, different):
    backend_result.variables = [backend_result.variables[1]]
    second_extraction = backend_result.model_copy(deep=True)
    setattr(second_extraction.variables[0], field, different)
    backend = FakeExtractionBackend(backend_result, {materials[1]["source_ref"]: second_extraction})
    agent = ResearchAgent(fixture_tool(tmp_path, materials[:2]), storage, PROMPT, extraction_backend=backend)

    result = agent.run(task)

    variables = storage.list_objects(Variable)
    assert len(variables) == len(result.variables_created) == 2
    assert [getattr(item, field) for item in variables] == [getattr(backend_result.variables[0], field), different]
    assert all(len(item.evidence_ids) == 1 for item in variables)
    assert len({storage.get_by_id(Evidence, item.evidence_ids[0]).source_id for item in variables}) == 2
    assert result.potential_conflicts == backend_result.potential_conflicts


def test_existing_observation_updates_lineage_once_and_keeps_id(task, storage, tmp_path, materials, backend_result, monkeypatch):
    backend_result.variables = [backend_result.variables[1]]
    first = ResearchAgent(
        fixture_tool(tmp_path, [materials[0]]), storage, PROMPT,
        extraction_backend=FakeExtractionBackend(backend_result),
    ).run(task)
    original = first.variables_created[0]
    backend_result.variables[0].name = "Planet Labs PBC Data + Analytics revenue"
    backend_result.variables[0].definition = "Different wording from another source."
    # Repeated support indexes must not produce duplicate persistent IDs.
    backend_result.variables[0].evidence_indexes = [1, 1]
    agent = ResearchAgent(
        fixture_tool(tmp_path, [materials[1]]), storage, PROMPT,
        extraction_backend=FakeExtractionBackend(backend_result),
    )
    updates = []
    update = storage.update

    def record_update(obj):
        assert all(storage.get_by_id(Evidence, eid) for eid in obj.evidence_ids)
        updates.append(obj)
        return update(obj)

    monkeypatch.setattr(storage, "update", record_update)
    second = agent.run(task)
    consolidated = ResearchStorage(storage.data_dir).list_objects(Variable)
    assert len(consolidated) == len(updates) == 1
    variable = consolidated[0]
    assert not second.variables_created
    assert variable.variable_id == original.variable_id
    assert (variable.name, variable.definition) == (original.name, original.definition)
    assert variable.evidence_ids == [*original.evidence_ids, second.evidence_created[1].evidence_id]
    evidence = [storage.get_by_id(Evidence, eid) for eid in variable.evidence_ids]
    assert [item.source_id for item in evidence] == [first.sources_created[0].source_id, second.sources_created[0].source_id]

    before = {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}
    rerun = agent.run(task)
    assert not rerun.variables_created and not rerun.evidence_created
    assert len(updates) == 1
    assert storage.list_objects(Variable) == consolidated
    assert {path.name: path.read_bytes() for path in storage.data_dir.iterdir()} == before


def test_backend_reuses_existing_entity_and_source(task, storage, tmp_path, materials, backend_result):
    entity = storage.insert(Entity(
        entity_id="existing-company", entity_type="company", canonical_name="planet labs pbc",
        aliases=["Planet Labs"],
    ))
    raw = materials[0]
    source = storage.insert(Source(
        source_id="existing-release", title=raw["title"], publisher=raw["publisher"],
        source_type=raw["source_type"], published_date=raw["published_date"],
        accessed_date="2026-10-01", locator=raw["locator"], primary_or_secondary=SourceOrigin.PRIMARY,
    ))
    backend = FakeExtractionBackend(backend_result)
    result = ResearchAgent(fixture_tool(tmp_path, [raw]), storage, PROMPT, extraction_backend=backend).run(task)
    assert not result.entities_created and not result.sources_created
    assert result.sources_reused == [source.source_id]
    assert all(item.source_id == source.source_id and item.entity_ids == [entity.entity_id] for item in result.evidence_created)
    assert all(item.entity_id == entity.entity_id for item in result.variables_created)
    assert storage.list_objects(Entity) == [entity]
    assert storage.list_objects(Source) == [source]


def test_backend_later_extraction_failure_preserves_prior_source_objects(task, storage, tmp_path, materials, backend_result):
    existing = storage.insert(Entity(entity_id="preexisting", entity_type="company", canonical_name="Existing company"))
    failure = ExtractionValidationError("simulated extraction failure")
    backend = FakeExtractionBackend(backend_result, {materials[1]["source_ref"]: failure})
    agent = ResearchAgent(fixture_tool(tmp_path, materials[:2]), storage, PROMPT, extraction_backend=backend)
    with pytest.raises(ResearchAgentError, match="Extraction backend failed") as exc:
        agent.run(task)
    assert exc.value.__cause__ is failure
    assert len(backend.calls) == 2
    assert storage.get_by_id(Entity, existing.entity_id) == existing
    assert len(storage.list_objects(Entity)) == 2
    sources = storage.list_objects(Source)
    assert len(sources) == 1 and sources[0].locator == materials[0]["locator"]
    assert len(storage.list_objects(Evidence)) == len(storage.list_objects(Variable)) == 2
    assert all(item.source_id == sources[0].source_id for item in storage.list_objects(Evidence))


@pytest.mark.parametrize(("failure", "message"), [
    ("evidence-entity", "No EntityCandidate resolves"),
    ("variable-entity", "No EntityCandidate resolves"),
    ("evidence-index", "evidence index"),
    ("locator", "source_locator"),
])
def test_backend_invalid_references_rejected_before_write(task, storage, tmp_path, materials, backend_result, failure, message):
    if failure == "evidence-entity":
        backend_result.evidence[1].entity_names = ["Unresolved company"]
    elif failure == "variable-entity":
        backend_result.variables[1].entity_name = "Unresolved company"
    elif failure == "evidence-index":
        backend_result.variables[1].evidence_indexes = [99]
    else:
        backend_result.evidence[1].source_locator = "B999"
    backend = FakeExtractionBackend(backend_result)
    agent = ResearchAgent(fixture_tool(tmp_path, [materials[0]]), storage, PROMPT, extraction_backend=backend)
    with pytest.raises(ResearchAgentError, match=message):
        agent.run(task)
    assert_storage_empty(storage)


def test_backend_no_material_reports_not_found_without_calling_backend(task, storage, tool, materials, backend_result):
    task.constraints.excluded_sources = [item["source_ref"] for item in materials]
    backend = FakeExtractionBackend(backend_result)
    result = ResearchAgent(tool, storage, PROMPT, extraction_backend=backend).run(task)
    assert backend.calls == [] and tool.read_refs == []
    assert result.not_found[0].search_attempted == tool.queries
    assert result.candidate_gaps
    assert not result.search_coverage.period_covered and not result.search_coverage.scope_covered
    assert_storage_empty(storage)


def test_optional_source_store_preserves_legacy_execution(agent, task, storage, tool):
    assert agent.source_store is None
    assert ResearchAgent(tool, storage, PROMPT, source_store=None).source_store is None
    result = agent.run(task)
    assert result.evidence_created and result.variables_created
    assert set(storage.data_dir.parent.iterdir()) == {storage.data_dir}


@pytest.mark.parametrize("path", ["deterministic", "backend"])
def test_actual_read_materials_saved_before_extraction_and_exclusions_respected(
    task, workspace_storage, source_store, tmp_path, materials, backend_result, monkeypatch, path,
):
    records = deepcopy(materials)
    records[0]["raw_content"] = "<html>Mock raw earnings release</html>"
    tool = fixture_tool(tmp_path, records)
    task.constraints.excluded_sources = ["mock-src-004"]
    events = []
    read, put = tool.read, source_store.put

    def record_read(ref):
        material = read(ref)
        events.append(("read", ref))
        return material

    def record_put(material):
        key = put(material)
        events.append(("put", material["source_ref"]))
        return key

    monkeypatch.setattr(tool, "read", record_read)
    monkeypatch.setattr(source_store, "put", record_put)
    backend = None
    if path == "backend":
        backend = FakeExtractionBackend(backend_result)
        extract = backend.extract

        def record_extract(**kwargs):
            ref = kwargs["material"]["source_ref"]
            assert source_store.contains(ref)
            events.append(("extract", ref))
            return extract(**kwargs)

        monkeypatch.setattr(backend, "extract", record_extract)
    else:
        extract_entities = agent_module._extract_entities

        def record_extract(materials, task):
            assert all(source_store.contains(material["source_ref"]) for material in materials)
            events.append(("extract", None))
            return extract_entities(materials, task)

        monkeypatch.setattr(agent_module, "_extract_entities", record_extract)
    result = ResearchAgent(
        tool, workspace_storage, PROMPT, extraction_backend=backend, source_store=source_store,
    ).run(task)

    assert tool.read_refs == ["mock-src-001", "mock-src-002"]
    assert events == [("read", "mock-src-001"), ("put", "mock-src-001"),
                      ("read", "mock-src-002"), ("put", "mock-src-002")] + (
        [("extract", "mock-src-001"), ("extract", "mock-src-002")] if backend else [("extract", None)]
    )
    assert result.evidence_created and result.variables_created
    for material in records:
        if material["source_ref"] in tool.read_refs:
            saved = source_store.get(material["source_ref"])
            assert saved["content"] == material["content"]
            if "raw_content" in material:
                assert saved["raw_content"] == material["raw_content"]
        else:
            assert not source_store.contains(material["source_ref"])


def test_source_store_stays_empty_when_no_material_is_read(
    task, workspace_storage, source_store, workspace, tool, materials, backend_result,
):
    task.constraints.excluded_sources = [material["source_ref"] for material in materials]
    backend = FakeExtractionBackend(backend_result)
    result = ResearchAgent(
        tool, workspace_storage, PROMPT, extraction_backend=backend, source_store=source_store,
    ).run(task)
    assert result.not_found and not tool.read_refs and not backend.calls
    assert list(workspace.sources_raw_dir.iterdir()) == []
    assert list(workspace.sources_normalized_dir.iterdir()) == []
    assert_storage_empty(workspace_storage)


@pytest.mark.parametrize("path", ["backend-extraction", "deterministic-validation"])
def test_extraction_failure_retains_snapshots_and_prior_successful_objects(
    task, workspace_storage, source_store, tmp_path, materials, backend_result, monkeypatch, path,
):
    tool = fixture_tool(tmp_path, materials[:2])
    backend = None
    if path == "backend-extraction":
        failure = ExtractionValidationError("simulated extraction failure")
        backend = FakeExtractionBackend(backend_result, {materials[1]["source_ref"]: failure})
        expected_error, message = ResearchAgentError, "Extraction backend failed"
    else:
        monkeypatch.setattr(agent_module, "_extract_variables", lambda evidence: [forbidden_object(Claim)])
        expected_error, message = AgentPermissionError, "Claim"
    agent = ResearchAgent(
        tool, workspace_storage, PROMPT, extraction_backend=backend, source_store=source_store,
    )
    with pytest.raises(expected_error, match=message) as error:
        agent.run(task)
    if backend is not None:
        assert error.value.__cause__ is failure and len(backend.calls) == 2
    assert tool.read_refs == [material["source_ref"] for material in materials[:2]]
    for material in materials[:2]:
        assert source_store.get(material["source_ref"])["content"] == material["content"]
    if backend is not None:
        assert len(workspace_storage.list_objects(Source)) == 1
        assert len(workspace_storage.list_objects(Entity)) == 1
        assert len(workspace_storage.list_objects(Evidence)) == 2
        assert len(workspace_storage.list_objects(Variable)) == 2
    else:
        assert_storage_empty(workspace_storage)


@pytest.mark.parametrize("path", ["deterministic", "backend"])
def test_source_store_failure_stops_before_extraction(
    task, workspace_storage, source_store, workspace, tool, backend_result, monkeypatch, path,
):
    failure = RunSourceStoreError("simulated source-store failure")
    puts = []

    def fail_put(material):
        puts.append(material["source_ref"])
        raise failure

    def unexpected_extraction(*args):
        pytest.fail("Extraction must not run after source persistence failure")

    monkeypatch.setattr(source_store, "put", fail_put)
    monkeypatch.setattr(agent_module, "_extract_entities", unexpected_extraction)
    backend = FakeExtractionBackend(backend_result) if path == "backend" else None
    agent = ResearchAgent(
        tool, workspace_storage, PROMPT, extraction_backend=backend, source_store=source_store,
    )
    with pytest.raises(ResearchAgentError, match="Source material persistence failed for mock-src-001") as error:
        agent.run(task)
    assert error.value.__cause__ is failure
    assert tool.read_refs == puts == ["mock-src-001"]
    if backend is not None:
        assert backend.calls == []
    assert list(workspace.sources_raw_dir.iterdir()) == []
    assert list(workspace.sources_normalized_dir.iterdir()) == []
    assert_storage_empty(workspace_storage)


def test_read_failure_preserves_error_and_only_saves_successful_reads(
    task, workspace_storage, source_store, tool, materials, backend_result, monkeypatch,
):
    failure = ResearchToolError("simulated read failure")
    read, put = tool.read, source_store.put
    puts = []

    def fail_second_read(ref):
        if ref == "mock-src-002":
            tool.read_refs.append(ref)
            raise failure
        return read(ref)

    def record_put(material):
        puts.append(material["source_ref"])
        return put(material)

    monkeypatch.setattr(tool, "read", fail_second_read)
    monkeypatch.setattr(source_store, "put", record_put)
    backend = FakeExtractionBackend(backend_result)
    agent = ResearchAgent(
        tool, workspace_storage, PROMPT, extraction_backend=backend, source_store=source_store,
    )
    with pytest.raises(ResearchToolError, match="simulated read failure") as error:
        agent.run(task)
    assert error.value is failure
    assert tool.read_refs == ["mock-src-001", "mock-src-002"] and puts == ["mock-src-001"]
    assert source_store.get("mock-src-001")["content"] == materials[0]["content"]
    assert not source_store.contains("mock-src-002") and not source_store.contains("mock-src-004")
    assert not backend.calls
    assert_storage_empty(workspace_storage)


def long_source_and_chunk_results(task, materials, backend_result, values=(60, 65, 60)):
    """Five real chunks: three lexical matches and two unrelated paragraphs."""
    material = deepcopy(materials[0])
    paragraphs = []
    for index in range(1, 6):
        if index in (1, 3, 5):
            value = values[(index - 1) // 2]
            prefix = f"FY27 Q2: Data + Analytics revenue was USD {value} million. "
        else:
            prefix = "Unrelated archival observations. "
        paragraphs.append(prefix + "neutral filler " * 330)
    material.update(
        source_ref="mock-long-filing", source_type="10-Q", locator="mock://long-filing",
        content="\n\n".join(paragraphs), raw_content="<html>synthetic filing</html>",
        tags=[agent_module._build_query(task)], primary_or_secondary="Primary",
        independence_group="filing-disclosure-group",
    )
    blocks = build_source_blocks(material["content"])
    chunks = chunk_source_blocks(blocks)
    assert [chunk.chunk_id for chunk in chunks] == [f"C{index:03d}" for index in range(1, 6)]
    outcomes = {}
    for index, value in zip((1, 3, 5), values):
        extracted = backend_result.model_copy(deep=True)
        evidence = extracted.evidence[1].model_copy(deep=True)
        evidence.source_locator = f"B{index:03d}"
        evidence.value = value
        evidence.statement = f"Data + Analytics revenue was USD {value} million."
        variable = extracted.variables[1].model_copy(deep=True)
        variable.value = value
        variable.name = f"Business revenue described in block {index}"
        variable.evidence_indexes = [0]
        extracted.evidence = [evidence]
        extracted.variables = [variable]
        outcomes[(material["source_ref"], evidence.source_locator)] = extracted
    return material, chunks, outcomes


def test_stage1_lexical_selection_chunk_local_lineage_incremental_persistence_and_rerun(
    task, workspace_storage, source_store, tmp_path, materials, backend_result, monkeypatch,
):
    material, chunks, outcomes = long_source_and_chunk_results(task, materials, backend_result)
    tool = fixture_tool(tmp_path, [material, deepcopy(materials[1])])
    backend = FakeExtractionBackend(backend_result, outcomes)
    agent = ResearchAgent(
        tool, workspace_storage, PROMPT, extraction_backend=backend, source_store=source_store,
        search_query="PL", source_refs=[material["source_ref"]],
    )
    retrieve = agent_module.retrieve_candidate_chunks
    retrieval_calls = []

    def record_retrieval(supplied_chunks, **kwargs):
        assert supplied_chunks == chunks
        assert kwargs["top_k"] == 8 and kwargs["neighbor_radius"] == 0
        assert set(kwargs["terms"]) == set(agent_module._build_query(task).split())
        assert list(kwargs["phrases"]) == [task.scope.product_or_business]
        selected = retrieve(supplied_chunks, **kwargs)
        retrieval_calls.append([item.chunk_id for item in selected])
        return selected

    monkeypatch.setattr(agent_module, "retrieve_candidate_chunks", record_retrieval)
    extract = backend.extract

    def check_incremental_extract(**kwargs):
        call_number = len(backend.calls)
        assert source_store.get(material["source_ref"])["content"] == material["content"]
        assert len(workspace_storage.list_objects(Evidence)) == call_number
        if call_number:
            assert len(workspace_storage.list_objects(Entity)) == 1
            assert len(workspace_storage.list_objects(Source)) == 1
            assert len(workspace_storage.list_objects(Variable)) == call_number
        return extract(**kwargs)

    monkeypatch.setattr(backend, "extract", check_incremental_extract)
    result = agent.run(task)
    assert tool.queries == ["PL"] and tool.read_refs == [material["source_ref"]]
    assert retrieval_calls == [["C001", "C003", "C005"]]
    assert [tuple(block.block_id for block in call["source_blocks"]) for call in backend.calls] == [
        ("B001",), ("B003",), ("B005",),
    ]
    assert all(len(call["source_blocks"][0].text) < len(material["content"]) for call in backend.calls)
    assert len(result.entities_created) == len(result.sources_created) == 1
    source = result.sources_created[0]
    assert source.primary_or_secondary is SourceOrigin.PRIMARY
    assert source.independence_group == material["independence_group"]
    assert len(result.evidence_created) == 3 and len(result.variables_created) == 2
    assert [item.source_locator for item in result.evidence_created] == ["B001", "B003", "B005"]
    assert all(item.source_id == source.source_id for item in result.evidence_created)
    assert all(item.entity_ids == [result.entities_created[0].entity_id] for item in result.evidence_created)
    observations = {item.value: item for item in workspace_storage.list_objects(Variable)}
    assert observations[60].evidence_ids == [result.evidence_created[0].evidence_id, result.evidence_created[2].evidence_id]
    assert observations[65].evidence_ids == [result.evidence_created[1].evidence_id]
    for variable in observations.values():
        assert all(workspace_storage.get_by_id(Evidence, evidence_id).value == variable.value
                   for evidence_id in variable.evidence_ids)
    assert result.variables_created == workspace_storage.list_objects(Variable)
    assert all("chunk" in item.result.casefold() for item in result.not_found)
    for kind in (Claim, Gap, Estimate, Event):
        assert workspace_storage.list_objects(kind) == []

    monkeypatch.setattr(backend, "extract", extract)
    before = {path.name: path.read_bytes() for path in workspace_storage.data_dir.iterdir()}
    rerun = agent.run(task)
    assert len(backend.calls) == 6
    assert not rerun.entities_created and not rerun.sources_created
    assert not rerun.evidence_created and not rerun.variables_created
    assert rerun.sources_reused == [source.source_id]
    assert {path.name: path.read_bytes() for path in workspace_storage.data_dir.iterdir()} == before


def test_stage1_later_chunk_failure_retains_two_completed_chunks(
    task, workspace_storage, source_store, tmp_path, materials, backend_result,
):
    material, _, outcomes = long_source_and_chunk_results(task, materials, backend_result, values=(60, 60, 60))
    failure = ExtractionValidationError("simulated third chunk failure")
    outcomes[(material["source_ref"], "B005")] = failure
    backend = FakeExtractionBackend(backend_result, outcomes)
    agent = ResearchAgent(
        fixture_tool(tmp_path, [material]), workspace_storage, PROMPT,
        extraction_backend=backend, source_store=source_store,
    )
    with pytest.raises(ResearchAgentError, match="C005") as error:
        agent.run(task)
    assert error.value.__cause__ is failure
    assert len(backend.calls) == 3
    reopened = ResearchStorage(workspace_storage.data_dir)
    assert len(reopened.list_objects(Entity)) == len(reopened.list_objects(Source)) == 1
    evidence = reopened.list_objects(Evidence)
    assert [item.source_locator for item in evidence] == ["B001", "B003"]
    variables = reopened.list_objects(Variable)
    assert len(variables) == 1
    assert variables[0].evidence_ids == [item.evidence_id for item in evidence]
    assert source_store.contains(material["source_ref"])
    for kind in (Claim, Gap, Estimate, Event):
        assert reopened.list_objects(kind) == []


def test_stage1_locator_valid_elsewhere_in_source_is_rejected_for_current_chunk(
    task, workspace_storage, tmp_path, materials, backend_result,
):
    material, _, outcomes = long_source_and_chunk_results(task, materials, backend_result)
    outcomes[(material["source_ref"], "B003")].evidence[0].source_locator = "B001"
    backend = FakeExtractionBackend(backend_result, outcomes)
    agent = ResearchAgent(fixture_tool(tmp_path, [material]), workspace_storage, PROMPT, extraction_backend=backend)
    with pytest.raises(ResearchAgentError, match="source_locator"):
        agent.run(task)
    assert len(backend.calls) == 2
    assert len(workspace_storage.list_objects(Evidence)) == len(workspace_storage.list_objects(Variable)) == 1
    assert workspace_storage.list_objects(Evidence)[0].source_locator == "B001"


def test_stage1_no_matching_chunks_snapshots_source_without_extraction(
    task, workspace_storage, source_store, tmp_path, materials, backend_result,
):
    material = deepcopy(materials[0])
    material["content"] = "Unrelated archival observations. " * 330
    material["tags"] = [agent_module._build_query(task)]
    backend = FakeExtractionBackend(backend_result)
    agent = ResearchAgent(
        fixture_tool(tmp_path, [material]), workspace_storage, PROMPT,
        extraction_backend=backend, source_store=source_store,
    )
    result = agent.run(task)
    assert backend.calls == []
    assert source_store.contains(material["source_ref"])
    assert len(result.sources_created) == 1
    source = result.sources_created[0]
    assert source.locator == material["locator"] and source.title == material["title"]
    assert source.source_type == material["source_type"]
    assert workspace_storage.list_objects(Source) == [source]
    assert source.primary_or_secondary is SourceOrigin.PRIMARY
    assert result.search_coverage.primary_source_found
    assert not result.entities_created and workspace_storage.list_objects(Entity) == []
    assert not result.evidence_created and not result.variables_created
    assert not result.search_coverage.period_covered and not result.search_coverage.scope_covered
    assert workspace_storage.list_objects(Evidence) == workspace_storage.list_objects(Variable) == []


def test_cli_runs_example_with_temporary_data_directory(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "research_agent.py"), "--task", str(ROOT / "examples" / "planet_growth.json"), "--data-dir", str(tmp_path / "cli-data")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Task: TASK-001" in completed.stdout
    assert "Sources created: 3" in completed.stdout and "Entities created: 1" in completed.stdout
    assert len(completed.stdout.splitlines()) == 8
    assert completed.stderr == ""
    assert ResearchStorage(tmp_path / "cli-data").list_objects(Variable)
