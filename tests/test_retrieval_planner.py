"""Fake Responses tests for concept contracts, not live retrieval quality."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import retrieval_planner as planner_module
from retrieval_planner import (
    DeepSeekRetrievalPlannerBackend,
    RetrievalPlan,
    RetrievalPlannerProviderError,
    RetrievalPlannerValidationError,
    SourceContext,
)
from schemas import ResearchTask


class FakeClient:
    def __init__(self, output_text=None, error=None):
        self.calls = []
        self.error = error
        self.responses = SimpleNamespace(create=self.create)
        self.response = SimpleNamespace(output_text=output_text, status="completed")

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def prevent_real_client(monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def forbidden_client(**kwargs):
        raise AssertionError("Tests must inject a fake client")

    monkeypatch.setattr(planner_module, "OpenAI", forbidden_client)


@pytest.fixture
def inputs():
    task = ResearchTask.model_validate_json(json.dumps({
        "task_id": "TASK-TEST",
        "research_question": "示例公司当前增长主要来自什么业务？",
        "target_requirement": {
            "requirement_id": "REQ-TEST", "question": "订阅业务的增长和收入贡献是多少？", "core_requirement": True,
        },
        "search_mode": "normal",
        "scope": {"entity": "Example Company", "period": "Q2", "product_or_business": "Subscriptions"},
    }))
    return task, SourceContext(title="Example operating update", source_type="business article", publisher="Example Publisher")


def test_valid_english_plan_from_chinese_task_and_metadata(inputs, monkeypatch):
    task, context = inputs
    before = (task.model_dump(), context.model_dump())
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")
    payload = {"phrases": ["subscription revenue", "revenue growth", "revenue mix"], "terms": ["subscription", "growth"]}
    client = FakeClient(json.dumps(payload))
    result = DeepSeekRetrievalPlannerBackend(client=client).plan(task, context)
    assert isinstance(result, RetrievalPlan)
    assert result.model_dump() == payload  # Only concepts, no persistent Research Objects.
    assert (task.model_dump(), context.model_dump()) == before
    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    assert "top_p" not in request
    assert request["text"]["format"] == {
        "type": "json_schema", "name": "retrieval_plan", "schema": RetrievalPlan.model_json_schema(),
    }
    supplied = json.loads(request["input"][1]["content"])
    assert supplied["research_task"]["research_question"] == task.research_question
    assert supplied["research_task"]["target_requirement"] == task.target_requirement.model_dump()
    assert supplied["research_task"]["scope"] == task.scope.model_dump()
    assert supplied["source_context"] == context.model_dump()
    assert set(supplied["source_context"]) == {"title", "source_type", "publisher"}
    instructions = request["input"][0]["content"]
    assert "BOTH research_question and target_requirement" in instructions
    assert "Never leak a research answer" in instructions


def test_normalization_stable_dedup_and_lexical_symbols(inputs):
    payload = {
        "phrases": ["  Revenue   GROWTH  ", "revenue growth", "Data + Analytics", "YEAR-OVER-YEAR", "backlog", "increased capacity"],
        "terms": [" Revenue ", "REVENUE", "R&D", "10-Q", "Analytics"],
    }
    client = FakeClient(json.dumps(payload))
    result = DeepSeekRetrievalPlannerBackend(client=client).plan(*inputs)
    assert result.phrases == ["revenue growth", "data + analytics", "year-over-year", "backlog", "increased capacity"]
    assert result.terms == ["revenue", "r&d", "10-q", "analytics"]


@pytest.mark.parametrize(("field", "concept"), [
    ("phrases", " \t"),
    ("phrases", "收入增长"),
    ("phrases", "revenue\ngrowth"),
    ("phrases", "government is the growth driver"),
    ("phrases", "government has driven growth"),
    ("phrases", "government revenue increased"),
    ("phrases", "government subscriptions generated most of the growth this quarter"),
    ("phrases", "what drives growth?"),
    ("terms", "revenue growth"),
])
def test_invalid_lexical_concepts_rejected(inputs, field, concept):
    payload = {"phrases": [], "terms": [], field: [concept]}
    client = FakeClient(json.dumps(payload))
    with pytest.raises(RetrievalPlannerValidationError):
        DeepSeekRetrievalPlannerBackend(client=client).plan(*inputs)


@pytest.mark.parametrize(("field", "values"), [
    ("phrases", [f"business metric {index}" for index in range(13)]),
    ("terms", [f"metric{index}" for index in range(13)]),
    ("phrases", ["revenue"] * 13),  # Raw cap applies before deduplication.
])
def test_raw_concept_limit_rejected(inputs, field, values):
    payload = {"phrases": [], "terms": [], field: values}
    client = FakeClient(json.dumps(payload))
    with pytest.raises(RetrievalPlannerValidationError):
        DeepSeekRetrievalPlannerBackend(client=client).plan(*inputs)


def test_twelve_concepts_per_list_allowed(inputs):
    payload = {
        "phrases": [f"business metric {index}" for index in range(12)],
        "terms": [f"metric{index}" for index in range(12)],
    }
    client = FakeClient(json.dumps(payload))
    assert DeepSeekRetrievalPlannerBackend(client=client).plan(*inputs).model_dump() == payload


@pytest.mark.parametrize("output_text", [
    "not JSON",
    '{"phrases":["revenue growth"]}',
    '{"phrases":[],"terms":[],"claims":[{"statement":"Unrequested conclusion"}]}',
])
def test_malformed_or_research_object_output_rejected(inputs, output_text):
    client = FakeClient(output_text)
    with pytest.raises(RetrievalPlannerValidationError):
        DeepSeekRetrievalPlannerBackend(client=client).plan(*inputs)


def test_empty_provider_output_rejected(inputs):
    client = FakeClient(" \n")
    with pytest.raises(RetrievalPlannerValidationError):
        DeepSeekRetrievalPlannerBackend(client=client).plan(*inputs)


def test_provider_error_wrapped_without_retry(inputs):
    error = RuntimeError("Provider unavailable")
    client = FakeClient(error=error)
    with pytest.raises(RetrievalPlannerProviderError) as raised:
        DeepSeekRetrievalPlannerBackend(client=client).plan(*inputs)
    assert raised.value.__cause__ is error
    assert len(client.calls) == 1


def test_source_context_rejects_full_text(inputs):
    _, context = inputs
    with pytest.raises(ValidationError):
        SourceContext.model_validate({**context.model_dump(), "content": "Full source body must not reach the Planner."})
