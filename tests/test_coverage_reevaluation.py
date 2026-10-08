"""Re-invoke frozen coverage on fresh task Evidence; semantic decisions are fakes."""

import json
from pathlib import Path

import pytest

import selected_source_acquisition as acquisition_module
from llm_extractor import DeepSeekExtractionBackend
from next_search_intent import NextSearchIntentGenerator
from requirement_coverage import (
    RequirementCoverageBackend,
    RequirementCoverageEvaluation,
    RequirementCoverageEvaluator,
    RequirementCoverageStatus,
)
from research_agent import ResearchAgent
from research_tools import MockResearchTool, ResearchTool
from run_workspace import ResearchRunWorkspace
from schemas import Evidence, ResearchResult, ResearchTask, Source, SourceOrigin
from sec_research_tool import SECResearchTool
from source_selection import CandidateSourceSelector
from storage import ResearchStorage
from tool_routing import ToolRouter


ROOT = Path(__file__).resolve().parents[1]
GROWTH = "Data + Analytics growth in FY27 Q2"
CONTRIBUTION = "Data + Analytics revenue contribution in FY27 Q2"


class BeforeAfterBackend(RequirementCoverageBackend):
    """Supply explicit audit outputs without pretending to test semantic accuracy."""

    def __init__(self, before, after):
        self.responses = [before, after]
        self.calls = []

    def evaluate(self, task, evidence):
        self.calls.append((task, list(evidence)))
        return self.responses[len(self.calls) - 1].model_copy(deep=True)


@pytest.fixture
def task():
    return ResearchTask.model_validate_json((ROOT / "examples" / "planet_growth.json").read_text(encoding="utf-8"))


@pytest.fixture
def storage(tmp_path):
    workspace = ResearchRunWorkspace.create("mock-reevaluation", "company", root_dir=tmp_path / "runs")
    return ResearchStorage(workspace.objects_dir)


@pytest.fixture(autouse=True)
def forbid_acquisition_and_workflow_side_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Coverage re-evaluation must not execute acquisition or subsequent workflow")

    for tool_type in (ResearchTool, SECResearchTool, MockResearchTool):
        monkeypatch.setattr(tool_type, "search", forbidden)
        monkeypatch.setattr(tool_type, "read", forbidden)
    monkeypatch.setattr(DeepSeekExtractionBackend, "extract", forbidden)
    monkeypatch.setattr(ResearchAgent, "run", forbidden)
    monkeypatch.setattr(ResearchAgent, "run_known_sources", forbidden)
    monkeypatch.setattr(NextSearchIntentGenerator, "generate", forbidden)
    monkeypatch.setattr(ToolRouter, "route", forbidden)
    monkeypatch.setattr(ToolRouter, "execute_search", forbidden)
    monkeypatch.setattr(CandidateSourceSelector, "select", forbidden)
    monkeypatch.setattr(acquisition_module, "acquire_selected_sources", forbidden)


def source(source_id):
    return Source(
        source_id=source_id, title=f"Synthetic disclosure {source_id}", publisher="Mock publisher",
        source_type="8-K", published_date="2026-09-03", accessed_date="2026-10-08",
        locator=f"mock://{source_id}", primary_or_secondary=SourceOrigin.PRIMARY,
    )


def evidence(evidence_id, source_id, statement, *, value=None, unit=None, period=None, scope="Data + Analytics"):
    return Evidence(
        evidence_id=evidence_id, source_id=source_id, statement=statement, value=value, unit=unit,
        period=period, scope=scope, evidence_type="Reported Fact", source_locator="B001",
        collected_at="2026-10-08T12:00:00+00:00",
    )


def evaluation(task, status, *, supporting=(), conflicting=(), uncovered=(), rationale):
    return RequirementCoverageEvaluation.model_validate_json(json.dumps({
        "requirement_id": task.target_requirement.requirement_id, "status": status,
        "supporting_evidence_ids": list(supporting), "conflicting_evidence_ids": list(conflicting),
        "uncovered_aspects": list(uncovered), "rationale": rationale,
    }))


def files(storage):
    return {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}


@pytest.mark.parametrize("scenario", ["irrelevant", "partial-improvement", "complete", "conflicted"])
def test_frozen_evaluator_reevaluates_current_task_evidence_without_workflow_side_effects(task, storage, monkeypatch, scenario):
    """A–D canned outcomes verify fresh inputs, supporting lineage and no writes."""
    storage.insert(source("SOURCE-original"))
    storage.insert(source("SOURCE-unrelated"))
    history = storage.insert(evidence(
        "E-UNRELATED-HISTORY", "SOURCE-unrelated", "Another company reported revenue growth of 99%.",
        value=99, unit="%", period="FY26", scope="Another company",
    ))
    has_growth = scenario in ("complete", "conflicted")
    old = storage.insert(evidence(
        "E-ORIGINAL", "SOURCE-original",
        "Data + Analytics revenue grew 14% in FY27 Q2." if has_growth else "The company provides imagery subscriptions.",
        value=14 if has_growth else None, unit="%" if has_growth else None,
        period="FY27 Q2" if has_growth else None,
    ))
    previous = evaluation(
        task, "Uncovered", supporting=[old.evidence_id] if has_growth else [],
        uncovered=[CONTRIBUTION] if has_growth else [GROWTH, CONTRIBUTION],
        rationale="Growth is supported, but revenue contribution is missing." if has_growth else "Neither required metric is answered.",
    )
    if scenario == "irrelevant":
        new = evidence("E-NEW", "SOURCE-new", "The company issued an earnings release.")
        expected = evaluation(task, "Uncovered", uncovered=[GROWTH, CONTRIBUTION], rationale="An earnings release exists, but neither requested metric is answered.")
    elif scenario == "partial-improvement":
        new = evidence("E-NEW", "SOURCE-new", "Data + Analytics represented 75% of FY27 Q2 revenue.", value=75, unit="%", period="FY27 Q2")
        expected = evaluation(task, "Uncovered", supporting=[new.evidence_id], uncovered=[GROWTH], rationale="Revenue contribution is supported; the growth metric remains unanswered.")
    elif scenario == "complete":
        new = evidence("E-NEW", "SOURCE-new", "Data + Analytics represented 75% of FY27 Q2 revenue.", value=75, unit="%", period="FY27 Q2")
        expected = evaluation(task, "Covered", supporting=[old.evidence_id, new.evidence_id], rationale="The Evidence answers growth and revenue contribution for the requested business and period.")
    else:
        new = evidence("E-NEW", "SOURCE-new", "Data + Analytics revenue grew 20% in FY27 Q2.", value=20, unit="%", period="FY27 Q2")
        expected = evaluation(task, "Conflicted", conflicting=[old.evidence_id, new.evidence_id], uncovered=[CONTRIBUTION], rationale="The same business growth metric and period is reported as both 14% and 20%.")

    # Embedded data and extraction diagnostics are deliberately stale/misleading.
    base_result = ResearchResult.model_validate_json(json.dumps({
        "task_id": task.task_id,
        "evidence_created": [old.model_copy(update={"statement": "Untrusted embedded snapshot.", "value": 999}).model_dump(mode="json")],
        "search_coverage": {"primary_source_found": True, "period_covered": True, "scope_covered": True},
        "not_found": [{"item": "All requested metrics", "result": "No answer found in this chunk; do not re-evaluate."}],
        "candidate_gaps": [{"question": "Should the workflow declare that these metrics are undisclosed?"}],
        "research_notes": "Stale extraction diagnostic; none of these fields establishes current Coverage.",
    }))
    base_before = base_result.model_dump(mode="json")
    task.existing_object_ids = [history.evidence_id]
    task_before = task.model_dump(mode="json")
    backend = BeforeAfterBackend(previous, expected)
    evaluator = RequirementCoverageEvaluator(backend)
    reads = []
    get_by_id = storage.get_by_id

    def record_get_by_id(object_type, object_id):
        if object_type is Evidence:
            reads.append(object_id)
        return get_by_id(object_type, object_id)

    monkeypatch.setattr(storage, "get_by_id", record_get_by_id)
    before_files = files(storage)
    before = evaluator.evaluate(task, base_result, storage)
    assert before == previous and before.status is RequirementCoverageStatus.UNCOVERED
    assert reads == [old.evidence_id]
    assert backend.calls[0] == (task, [old])
    assert files(storage) == before_files

    # Simulate Stage 2.5's already-completed persistence, not acquisition itself.
    storage.insert(source("SOURCE-new"))
    new = storage.insert(new)
    latest_old = old
    if scenario == "complete":
        latest_old = storage.update(old.model_copy(update={"collected_at": "2026-10-08T13:00:00+00:00"}))
    reads.clear()
    before_files = files(storage)
    after = evaluator.evaluate(task, base_result, storage, reused_evidence_ids=[new.evidence_id])

    assert after == expected
    assert reads == [old.evidence_id, new.evidence_id]
    assert backend.calls[1] == (task, [latest_old, new])
    assert all(history.evidence_id not in {item.evidence_id for item in items} for _, items in backend.calls)
    assert files(storage) == before_files  # All eight Research Object files remain untouched by evaluation.
    assert base_result.model_dump(mode="json") == base_before and task.model_dump(mode="json") == task_before
    if scenario == "irrelevant":
        assert new.evidence_id not in after.supporting_evidence_ids
        assert after.uncovered_aspects == before.uncovered_aspects
    elif scenario == "partial-improvement":
        assert after.supporting_evidence_ids == [new.evidence_id]
        assert after.uncovered_aspects == [GROWTH]
    elif scenario == "complete":
        assert after.status is RequirementCoverageStatus.COVERED and after.uncovered_aspects == []
    else:
        assert after.status is RequirementCoverageStatus.CONFLICTED
        assert after.conflicting_evidence_ids == [old.evidence_id, new.evidence_id]
