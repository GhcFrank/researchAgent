"""Fake intent-generation contracts; no live provider requests or tool execution."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import next_search_intent as intent_module
from next_search_intent import (
    DeepSeekNextSearchIntentBackend,
    NextSearchIntent,
    NextSearchIntentBackend,
    NextSearchIntentGenerator,
    NextSearchIntentProviderError,
    NextSearchIntentValidationError,
)
from requirement_coverage import RequirementCoverageEvaluation
from run_workspace import ResearchRunWorkspace
from schemas import Evidence, ResearchTask, Source, SourceOrigin
from sec_research_tool import SECResearchTool
from storage import ResearchStorage


ROOT = Path(__file__).resolve().parents[1]
GROWTH = "Data + Analytics growth in FY27 Q2"
CONTRIBUTION = "Data + Analytics revenue contribution in FY27 Q2"
UMBRELLA = "Any FY27 Q2-specific quantitative disclosure for Data + Analytics"


class FakeClient:
    def __init__(self, intents=None, *, output_text=None, error=None):
        self.calls = []
        self.output_text = json.dumps({"intents": intents}) if intents is not None else output_text
        self.error = error
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=self.output_text, status="completed", incomplete_details=None)


class FakeIntentBackend(NextSearchIntentBackend):
    def __init__(self, intents):
        self.intents = intents
        self.calls = []

    def generate(self, task, coverage, *, supporting_evidence=()):
        self.calls.append((task, coverage, list(supporting_evidence)))
        return self.intents


@pytest.fixture(autouse=True)
def prevent_real_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "intent-test-model")
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    def forbidden_sdk_client(**kwargs):
        raise AssertionError("Tests must inject a fake client")

    monkeypatch.setattr(intent_module, "OpenAI", forbidden_sdk_client)


@pytest.fixture
def task():
    return ResearchTask.model_validate_json((ROOT / "examples" / "planet_growth.json").read_text(encoding="utf-8"))


@pytest.fixture
def storage(tmp_path):
    workspace = ResearchRunWorkspace.create("mock-intent-test", "company", root_dir=tmp_path / "runs")
    store = ResearchStorage(workspace.objects_dir)
    store.insert(Source(
        source_id="SOURCE-test", title="Synthetic disclosure", publisher="Mock publisher",
        source_type="Earnings Release", published_date="2026-10-01", accessed_date="2026-10-02",
        locator="mock://intent", primary_or_secondary=SourceOrigin.PRIMARY,
    ))
    store.insert(Evidence(
        evidence_id="E-GROWTH", source_id="SOURCE-test",
        statement="Data + Analytics revenue grew 14% in FY27 Q2.", value=14, unit="%",
        period="FY27 Q2", scope="Data + Analytics", evidence_type="Reported Fact",
        source_locator="B001", collected_at="2026-10-02T12:00:00+00:00",
    ))
    return store


def coverage_for(task, *, status="Uncovered", aspects=(GROWTH,), supporting=(), conflicting=()):
    return RequirementCoverageEvaluation.model_validate_json(json.dumps({
        "requirement_id": task.target_requirement.requirement_id, "status": status,
        "supporting_evidence_ids": list(supporting), "conflicting_evidence_ids": list(conflicting),
        "uncovered_aspects": list(aspects), "rationale": "Synthetic coverage audit context.",
    }))


def intent_payload(task, *, aspects=(GROWTH,)):
    return {
        "requirement_id": task.target_requirement.requirement_id,
        "target_aspects": list(aspects),
        "search_question": "What was Data + Analytics revenue growth in FY27 Q2?",
        "search_terms": ["Data + Analytics", "revenue growth", "FY27 Q2"],
        "preferred_source_types": list(task.preferred_source_types),
        "rationale": "Find direct quarterly disclosure answering the uncovered material components.",
    }


@pytest.mark.parametrize("status", ["Covered", "Not Publicly Observable", "Conflicted"])
def test_non_uncovered_status_returns_no_intents_without_backend(task, status):
    coverage = coverage_for(
        task, status=status, aspects=(),
        supporting=["E-GROWTH"] if status != "Conflicted" else [],
        conflicting=["E-FIRST", "E-SECOND"] if status == "Conflicted" else [],
    )
    backend = FakeIntentBackend([])
    assert NextSearchIntentGenerator(backend).generate(task, coverage) == []
    assert backend.calls == []


@pytest.mark.parametrize("scenario", ["simple", "partial", "overlapping"])
def test_fake_narrow_intents_preserve_scope_and_do_not_search_or_write(
    task, storage, monkeypatch, scenario,
):
    """Canned decisions test workflow contracts, not actual LLM semantic grouping."""
    if scenario == "partial":
        aspects = [CONTRIBUTION]
        supporting = [storage.get_by_id(Evidence, "E-GROWTH")]
        coverage = coverage_for(task, aspects=aspects, supporting=[supporting[0].evidence_id])
        payload = intent_payload(task, aspects=aspects)
        payload["search_question"] = "What was Data + Analytics revenue contribution in FY27 Q2?"
        payload["search_terms"] = ["Data + Analytics", "revenue contribution", "FY27 Q2"]
    elif scenario == "overlapping":
        aspects = [GROWTH, CONTRIBUTION, UMBRELLA]
        supporting = []
        coverage = coverage_for(task, aspects=aspects)
        payload = intent_payload(task, aspects=[GROWTH, CONTRIBUTION])
        payload["search_question"] = "What were Data + Analytics revenue growth and revenue contribution in FY27 Q2?"
        payload["search_terms"] = ["Data + Analytics", "revenue growth", "revenue contribution", "FY27 Q2"]
    else:
        supporting = []
        coverage = coverage_for(task)
        payload = intent_payload(task)

    def forbidden_tool_call(*args, **kwargs):
        raise AssertionError("Intent generation must not execute SEC retrieval")

    monkeypatch.setattr(SECResearchTool, "search", forbidden_tool_call)
    monkeypatch.setattr(SECResearchTool, "read", forbidden_tool_call)
    before = {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}
    task_before = task.model_dump(mode="json")
    client = FakeClient([payload])
    actual = NextSearchIntentGenerator(DeepSeekNextSearchIntentBackend(client=client)).generate(
        task, coverage, supporting_evidence=supporting,
    )
    assert [item.model_dump(mode="json") for item in actual] == [payload]
    assert len(client.calls) == 1
    assert set(actual[0].target_aspects) <= set(coverage.uncovered_aspects)
    if scenario == "partial":
        assert actual[0].target_aspects == [CONTRIBUTION]
        assert "revenue growth" not in actual[0].search_terms
    assert task.model_dump(mode="json") == task_before
    assert {path.name: path.read_bytes() for path in storage.data_dir.iterdir()} == before


@pytest.mark.parametrize("failure", ["wrong-coverage-id", "unrelated-supporting-evidence"])
def test_invalid_input_fails_before_backend(task, storage, failure):
    coverage = coverage_for(task, supporting=["E-GROWTH"])
    supporting = [storage.get_by_id(Evidence, "E-GROWTH")]
    if failure == "wrong-coverage-id":
        coverage.requirement_id = "OTHER-REQUIREMENT"
    else:
        supporting[0] = supporting[0].model_copy(update={"evidence_id": "E-UNRELATED"})
    backend = FakeIntentBackend([])
    with pytest.raises(NextSearchIntentValidationError):
        NextSearchIntentGenerator(backend).generate(task, coverage, supporting_evidence=supporting)
    assert backend.calls == []


@pytest.mark.parametrize("failure", [
    "invented-aspect", "wrong-requirement-id", "empty-aspects", "duplicate-aspects",
    "changed-task-preferences", "tool-name-source-type", "too-many-terms", "duplicate-terms",
    "nonenglish-term", "url-term", "empty-intents", "too-many-intents",
])
def test_deterministic_output_guards_reject_invalid_backend_intents(task, failure):
    payload = intent_payload(task)
    if failure == "invented-aspect":
        payload["target_aspects"] = ["Competitive positioning"]
    elif failure == "wrong-requirement-id":
        payload["requirement_id"] = "OTHER-REQUIREMENT"
    elif failure == "empty-aspects":
        payload["target_aspects"] = []
    elif failure == "duplicate-aspects":
        payload["target_aspects"] *= 2
    elif failure == "changed-task-preferences":
        payload["preferred_source_types"] = ["SEC filing"]
    elif failure == "tool-name-source-type":
        task.preferred_source_types = []
        payload["preferred_source_types"] = ["SECResearchTool"]
    elif failure == "too-many-terms":
        payload["search_terms"] = [f"metric{index}" for index in range(11)]
    elif failure == "duplicate-terms":
        payload["search_terms"] = ["revenue", "revenue"]
    elif failure == "nonenglish-term":
        payload["search_terms"] = ["收入贡献"]
    elif failure == "url-term":
        payload["search_terms"] = ["www.example.com/filing"]
    # Construct bypasses the schema deliberately so the generator must validate injected output.
    intents = [NextSearchIntent.model_construct(**payload)]
    if failure == "empty-intents":
        intents = []
    elif failure == "too-many-intents":
        intents *= 4
    backend = FakeIntentBackend(intents)
    with pytest.raises(NextSearchIntentValidationError):
        NextSearchIntentGenerator(backend).generate(task, coverage_for(task))


@pytest.mark.parametrize("preference", ["https://example.com/filing", "SECResearchTool.search"])
def test_source_preferences_reject_locations_and_tool_methods_before_backend(task, preference):
    task.preferred_source_types = [preference]
    backend = FakeIntentBackend([])
    with pytest.raises(NextSearchIntentValidationError):
        NextSearchIntentGenerator(backend).generate(task, coverage_for(task))
    assert backend.calls == []


@pytest.mark.parametrize("task_preferences", [[], ["Custom issuer briefing", "10-q", "Custom issuer briefing"]])
def test_source_preferences_use_catalog_only_when_task_has_none(task, task_preferences):
    task.preferred_source_types = deepcopy(task_preferences)
    payload = intent_payload(task)
    if not task_preferences:
        payload["preferred_source_types"] = ["10-Q", "earnings release"]
    client = FakeClient([payload])
    actual = NextSearchIntentGenerator(DeepSeekNextSearchIntentBackend(client=client)).generate(task, coverage_for(task))
    assert actual[0].preferred_source_types == payload["preferred_source_types"]
    assert task.preferred_source_types == task_preferences


def test_request_contains_narrow_coverage_context_english_rules_and_schema(task, storage):
    partial = storage.get_by_id(Evidence, "E-GROWTH")
    coverage = coverage_for(task, aspects=[CONTRIBUTION], supporting=[partial.evidence_id])
    payload = intent_payload(task, aspects=[CONTRIBUTION])
    payload["search_question"] = "What was Data + Analytics revenue contribution in FY27 Q2?"
    payload["search_terms"] = ["Data + Analytics", "revenue contribution", "FY27 Q2"]
    task_before = task.model_dump(mode="json")
    client = FakeClient([payload])
    actual = DeepSeekNextSearchIntentBackend(client=client).generate(task, coverage, supporting_evidence=[partial])
    request = client.calls[0]
    assert request["model"] == "intent-test-model"
    assert request["temperature"] == 0 and request["stream"] is False
    assert request["max_output_tokens"] == 32768
    assert [message["role"] for message in request["input"]] == ["system", "user"]
    user = json.loads(request["input"][1]["content"])
    assert user["research_task"] == task_before
    assert user["requirement_coverage"] == coverage.model_dump(mode="json")
    assert user["supporting_evidence"] == [partial.model_dump(mode="json")]
    assert user["allowed_source_types"] == task.preferred_source_types
    system = request["input"][0]["content"]
    for text in (
        "Research Question", "Target Requirement", "Requirement Coverage", "Uncovered Aspect",
        "Next Search Intent", "Target Aspect", "Search Question", "Search Term", "Preferred Source Type",
        "The Target Requirement is authoritative. The Research Question is context only,",
        "Each Target Aspect must be copied exactly from the supplied uncovered_aspects.",
        "Do not search an already covered material component again.",
        "Use the smallest number of search intents needed",
        "An umbrella aspect that adds no independent factual target need not",
        "Search Terms are concise English source-facing terms",
        "unchanged for every intent, preserving order, spelling, casing and duplicates.",
        "Preferred Source Type is a source-category preference, not Tool selection.",
    ):
        assert text.casefold() in system.casefold()
    output_format = request["text"]["format"]
    assert output_format["type"] == "json_schema" and output_format["name"] == "next_search_intents"
    schema = output_format["schema"]
    assert set(schema["$defs"]["NextSearchIntent"]["properties"]) == {
        "requirement_id", "target_aspects", "search_question", "search_terms", "preferred_source_types", "rationale",
    }
    assert [item.model_dump(mode="json") for item in actual] == [payload]
    assert task.model_dump(mode="json") == task_before


@pytest.mark.parametrize("output", ["", "{broken json"])
def test_empty_or_malformed_provider_output_raises_validation_error(task, output):
    backend = DeepSeekNextSearchIntentBackend(client=FakeClient(output_text=output))
    with pytest.raises(NextSearchIntentValidationError):
        backend.generate(task, coverage_for(task))


def test_provider_exception_is_wrapped_without_retry(task):
    failure = RuntimeError("simulated provider failure")
    client = FakeClient(error=failure)
    with pytest.raises(NextSearchIntentProviderError) as error:
        DeepSeekNextSearchIntentBackend(client=client).generate(task, coverage_for(task))
    assert error.value.__cause__ is failure
    assert len(client.calls) == 1


def test_model_configuration_is_required(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    with pytest.raises(NextSearchIntentProviderError):
        DeepSeekNextSearchIntentBackend(client=FakeClient())
