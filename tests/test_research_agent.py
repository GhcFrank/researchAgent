"""Research Agent business boundaries and the complete offline data flow."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import research_agent as agent_module
from research_agent import AgentPermissionError, ResearchAgent, ResearchAgentError
from research_tools import MockResearchTool
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


@pytest.fixture
def task():
    return ResearchTask.model_validate_json((ROOT / "examples" / "planet_growth.json").read_text(encoding="utf-8"))


@pytest.fixture
def storage(tmp_path):
    return ResearchStorage(tmp_path / "data")


@pytest.fixture
def tool():
    return RecordingTool()


@pytest.fixture
def agent(tool, storage):
    return ResearchAgent(tool, storage, PROMPT)


@pytest.fixture
def materials():
    return json.loads((ROOT / "fixtures" / "mock_planet_sources.json").read_text(encoding="utf-8"))


def fixture_tool(tmp_path, records):
    path = tmp_path / "raw_materials.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return RecordingTool(path)


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


@pytest.mark.parametrize("kind", [Claim, Gap, Estimate, Event], ids=lambda kind: kind.__name__)
def test_forbidden_objects_rejected_before_any_write(agent, task, storage, monkeypatch, kind):
    monkeypatch.setattr(agent_module, "_extract_variables", lambda evidence: [forbidden_object(kind)])
    with pytest.raises(AgentPermissionError, match=kind.__name__):
        agent.run(task)
    assert not storage.data_dir.exists()


@pytest.mark.parametrize("input_type", [VariableInputType.MODEL_ESTIMATE, VariableInputType.DERIVED])
def test_forbidden_variable_input_rejected_before_any_write(agent, task, storage, monkeypatch, input_type):
    variable = Variable(
        variable_id="forbidden-variable",
        name="Forbidden model output",
        variable_type="Financial",
        input_type=input_type,
        last_updated="2026-10-06T12:00:00Z",
    )
    monkeypatch.setattr(agent_module, "_extract_variables", lambda evidence: [variable])
    with pytest.raises(AgentPermissionError, match="Forbidden Variable input_type"):
        agent.run(task)
    assert not storage.data_dir.exists()


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
    assert not storage.data_dir.exists()


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
