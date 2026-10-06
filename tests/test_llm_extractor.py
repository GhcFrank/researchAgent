"""Fake-client tests only; no provider requests or persistent objects."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import llm_extractor as extractor_module
from llm_extractor import (
    CandidateInputType,
    DeepSeekExtractionBackend,
    EntityCandidate,
    EvidenceCandidate,
    ExtractionResult,
    ExtractionValidationError,
    LLMProviderError,
    VariableCandidate,
)
from schemas import ResearchTask


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, content=None, error=None, finish_reason="stop"):
        self.calls = []
        self.error = error
        self.response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason,
        )])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def prevent_real_client(monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    def forbidden_sdk_client(**kwargs):
        raise AssertionError("Tests must inject a fake client")

    monkeypatch.setattr(extractor_module, "OpenAI", forbidden_sdk_client)


@pytest.fixture
def inputs():
    task = ResearchTask.model_validate_json((ROOT / "examples" / "planet_growth.json").read_text(encoding="utf-8"))
    material = {
        "source_ref": "mock-guidance",
        "title": "[MOCK] Planet Labs guidance",
        "publisher": "Mock Planet Labs",
        "source_type": "Earnings Call Transcript",
        "published_date": "2026-10-01",
        "locator": "mock://guidance",
        "content": "MOCK / FIXTURE DATA. Planet Labs PBC management expects Data revenue to grow 30% in FY27 Q2.",
        "tags": ["mock", "fixture"],
    }
    rules = (ROOT / "prompts" / "research_agent.md").read_text(encoding="utf-8")
    return task, material, rules


@pytest.fixture
def payload():
    return {
        "entities": [{"entity_type": "company", "canonical_name": "Planet Labs PBC", "aliases": ["Planet Labs"]}],
        "evidence": [{
            "statement": "Planet Labs PBC management expects Data revenue to grow 30% in FY27 Q2.",
            "value": 30,
            "unit": "%",
            "entity_names": ["Planet Labs PBC"],
            "period": "FY27 Q2",
            "scope": "Data + Analytics",
            "evidence_type": "Management Guidance",
            "source_locator": "paragraph 1",
        }],
        "variables": [{
            "name": "Data revenue growth",
            "variable_type": "Financial",
            "entity_name": "Planet Labs PBC",
            "period": "FY27 Q2",
            "scope": "Data + Analytics",
            "value": 30,
            "unit": "%",
            "input_type": "Guidance",
            "evidence_indexes": [0],
        }],
        "research_notes": "MOCK data only; management guidance, not an observed result.",
    }


def test_valid_json_candidates_and_prompt_composition(inputs, payload):
    client = FakeClient(json.dumps(payload))
    result = DeepSeekExtractionBackend(client=client).extract(*inputs)
    assert isinstance(result, ExtractionResult)
    assert isinstance(result.entities[0], EntityCandidate)
    assert isinstance(result.evidence[0], EvidenceCandidate)
    assert isinstance(result.variables[0], VariableCandidate)
    assert result.entities[0].canonical_name == "Planet Labs PBC"
    assert result.entities[0].ticker is None
    assert result.evidence[0].statement in inputs[1]["content"]
    assert result.evidence[0].evidence_type == "Management Guidance"
    assert result.variables[0].value == 30
    assert result.variables[0].input_type is CandidateInputType.GUIDANCE
    assert result.variables[0].evidence_indexes == [0]
    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["model"] == "deepseek-flash"
    assert request["response_format"] == {"type": "json_object"}
    assert request["stream"] is False
    system, user = request["messages"]
    assert system["role"] == "system" and inputs[2] in system["content"]
    assert "Do not use outside knowledge" in system["content"]
    assert "CANDIDATE JSON SCHEMA" in system["content"] and "EXAMPLE JSON OUTPUT" in system["content"]
    sent = json.loads(user["content"])
    assert sent["research_task"] == inputs[0].model_dump(mode="json")
    assert sent["raw_source_material"] == {key: value for key, value in inputs[1].items() if key != "tags"}


@pytest.mark.parametrize("input_type", ["MODEL ESTIMATE", "Derived"])
def test_forbidden_variable_input_type_rejected(inputs, payload, input_type):
    payload["variables"][0]["input_type"] = input_type
    with pytest.raises(ExtractionValidationError) as exc:
        DeepSeekExtractionBackend(client=FakeClient(json.dumps(payload))).extract(*inputs)
    assert isinstance(exc.value.__cause__, ValidationError)
    assert exc.value.__cause__.errors()[0]["loc"] == ("variables", 0, "input_type")


@pytest.mark.parametrize("failure", ["missing-required-field", "persistent-id", "source", "claim"])
def test_invalid_candidate_or_forbidden_output_rejected(inputs, payload, failure):
    if failure == "missing-required-field":
        del payload["evidence"][0]["source_locator"]
    elif failure == "persistent-id":
        payload["entities"][0]["entity_id"] = "not-allowed"
    else:
        payload[failure] = {"description": "Forbidden output"}
    with pytest.raises(ExtractionValidationError):
        DeepSeekExtractionBackend(client=FakeClient(json.dumps(payload))).extract(*inputs)


@pytest.mark.parametrize("indexes", [[], [-1], [1]], ids=["missing-support", "negative-index", "out-of-range"])
def test_variable_references_must_resolve_in_same_result(inputs, payload, indexes):
    payload["variables"][0]["evidence_indexes"] = indexes
    with pytest.raises(ExtractionValidationError):
        DeepSeekExtractionBackend(client=FakeClient(json.dumps(payload))).extract(*inputs)


def test_candidate_conflict_cannot_invent_persistent_evidence_ids(inputs, payload):
    payload["potential_conflicts"] = [{"description": "A potential conflict", "evidence_ids": ["invented-id"]}]
    with pytest.raises(ExtractionValidationError):
        DeepSeekExtractionBackend(client=FakeClient(json.dumps(payload))).extract(*inputs)


@pytest.mark.parametrize("fenced", [False, True], ids=["malformed-json", "markdown-not-repaired"])
def test_invalid_json_is_not_repaired(inputs, payload, fenced):
    content = f"```json\n{json.dumps(payload)}\n```" if fenced else "{broken JSON"
    with pytest.raises(ExtractionValidationError):
        DeepSeekExtractionBackend(client=FakeClient(content)).extract(*inputs)


@pytest.mark.parametrize("content", [None, " \n\t"])
def test_empty_response_raises(inputs, content):
    with pytest.raises(ExtractionValidationError, match="empty"):
        DeepSeekExtractionBackend(client=FakeClient(content)).extract(*inputs)


def test_no_choices_raises(inputs):
    client = FakeClient()
    client.response.choices = []
    with pytest.raises(ExtractionValidationError, match="empty response"):
        DeepSeekExtractionBackend(client=client).extract(*inputs)


def test_truncated_response_rejected_even_if_json_is_valid(inputs, payload):
    client = FakeClient(json.dumps(payload), finish_reason="length")
    with pytest.raises(ExtractionValidationError, match="truncated"):
        DeepSeekExtractionBackend(client=client).extract(*inputs)


def test_provider_failure_wrapped_once(inputs):
    failure = RuntimeError("simulated provider failure")
    client = FakeClient(error=failure)
    with pytest.raises(LLMProviderError) as exc:
        DeepSeekExtractionBackend(client=client).extract(*inputs)
    assert exc.value.__cause__ is failure
    assert len(client.calls) == 1


def test_environment_config_constructs_only_fake_client(monkeypatch):
    client = FakeClient()
    constructed = []

    def fake_sdk_client(**kwargs):
        constructed.append(kwargs)
        return client

    monkeypatch.setattr(extractor_module, "OpenAI", fake_sdk_client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "configured-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.test/v1")
    backend = DeepSeekExtractionBackend()
    assert backend.client is client and backend.model == "configured-model"
    assert constructed == [{"api_key": "fake-test-key", "base_url": "https://deepseek.test/v1", "timeout": 60.0, "max_retries": 0}]
    assert client.calls == []


def test_missing_key_rejected_without_falling_back_to_openai():
    with pytest.raises(LLMProviderError, match="DEEPSEEK_API_KEY"):
        DeepSeekExtractionBackend()


def test_invalid_material_fails_before_provider_call(inputs, payload):
    task, material, rules = inputs
    invalid_material = deepcopy(material)
    del invalid_material["content"]
    client = FakeClient(json.dumps(payload))
    with pytest.raises(ExtractionValidationError, match="material"):
        DeepSeekExtractionBackend(client=client).extract(task, invalid_material, rules)
    assert client.calls == []
