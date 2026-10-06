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
    build_locatable_content,
    normalize_variable_candidate,
)
from schemas import ResearchTask


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, output_text=None, error=None, status="completed"):
        self.calls = []
        self.error = error
        self.response = SimpleNamespace(output_text=output_text, status=status)
        self.responses = SimpleNamespace(create=self.create)

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
        "content": "MOCK / FIXTURE DATA.\n\nPlanet Labs PBC management expects Data revenue to grow 30% in FY27 Q2.",
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
            "source_locator": "B002",
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


@pytest.mark.parametrize(("content", "expected"), [
    ("paragraph one\n\nparagraph two", {"B001": "paragraph one", "B002": "paragraph two"}),
    (
        "\r\n  paragraph one\r\n  indented  \r\nend \r\n \t\r\n\r\n paragraph two \r\n",
        {"B001": "paragraph one\r\n  indented  \r\nend", "B002": "paragraph two"},
    ),
])
def test_locatable_blocks_are_deterministic_and_preserve_text(content, expected):
    first = build_locatable_content(content)
    assert first == expected
    assert list(first) == ["B001", "B002"]
    assert build_locatable_content(content) == first


@pytest.mark.parametrize(("variants", "expected"), [
    (("Revenue", "revenue", "Segment Revenue", "segment_revenue", " \tSEGMENT--__REVENUE \n"), "revenue"),
    ((
        "Growth Rate", "Growth rate", "growth_rate",
        "Revenue Growth Rate", "revenue_growth_rate",
        "Segment Revenue Growth Rate", "segment_revenue_growth_rate",
    ), "growth_rate"),
    (("Revenue Mix", "revenue_mix", "Revenue Share", "Revenue Contribution"), "revenue_mix"),
    (("Customer Count", "  CUSTOMER---__ COUNT  "), "customer_count"),
])
def test_variable_type_normalization(payload, variants, expected):
    candidate = VariableCandidate.model_validate_json(json.dumps(payload["variables"][0]))
    for variable_type in variants:
        normalized = normalize_variable_candidate(candidate.model_copy(update={"variable_type": variable_type}))
        assert normalized.variable_type == expected


@pytest.mark.parametrize(("scope", "unit", "expected_scope", "expected_unit"), [
    ("  Total \n company  revenue  ", "  %  of\t total revenue  ", "Total company revenue", "% of total revenue"),
    (None, None, None, None),
])
def test_variable_cleanup_preserves_lineage(scope, unit, expected_scope, expected_unit):
    candidate = VariableCandidate(
        name="  Reported \n revenue   (FY27 Q2)  ",
        definition="  Source-defined metric  ",
        variable_type="Revenue Share",
        value=30.0,
        period=" FY27 Q2 ",
        input_type=CandidateInputType.GUIDANCE,
        entity_name=" Planet Labs PBC ",
        evidence_indexes=[2, 0],
        scope=scope,
        unit=unit,
    )
    original = candidate.model_dump()
    normalized = normalize_variable_candidate(candidate)
    assert normalized.name == "Reported revenue (FY27 Q2)"
    assert normalized.scope == expected_scope
    assert normalized.unit == expected_unit
    assert normalized.variable_type == "revenue_mix"
    changed_fields = {"name", "scope", "unit", "variable_type"}
    assert normalized.model_dump(exclude=changed_fields) == candidate.model_dump(exclude=changed_fields)
    assert candidate.model_dump() == original


def test_valid_responses_candidates_and_prompt_composition(inputs, payload):
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
    assert result.evidence[0].source_locator == "B002"
    assert result.variables[0].value == 30
    assert result.variables[0].variable_type == "financial"
    assert result.variables[0].input_type is CandidateInputType.GUIDANCE
    assert result.variables[0].evidence_indexes == [0]
    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["model"] == "deepseek-flash"
    assert request["text"] == {"format": {
        "type": "json_schema",
        "name": "research_extraction",
        "schema": ExtractionResult.model_json_schema(),
    }}
    assert "response_format" not in request and "messages" not in request
    assert request["max_output_tokens"] == 4096
    assert request["temperature"] == 0
    assert request["stream"] is False
    system, user = request["input"]
    assert system["role"] == "system" and inputs[2] in system["content"]
    assert "Do not use outside knowledge" in system["content"]
    assert "EXTRACTION COMPLETENESS RULES" in system["content"]
    assert "ATOMIC EVIDENCE RULE" in system["content"]
    assert "Use predicate structure, not punctuation alone, to decide whether to split." in system["content"]
    assert "Multiple independent predicates must be split into separate EvidenceCandidates" in system["content"]
    assert "One predicate with multiple objects must remain one EvidenceCandidate." in system["content"]
    assert '"Can part A be true or false independently of part B?"' in system["content"]
    assert "EVIDENCE LOCATOR RULES" in system["content"]
    assert "Do not suppress one variable because it can be mathematically derived from another." in system["content"]
    assert "CANDIDATE JSON SCHEMA" in system["content"] and "EXAMPLE JSON OUTPUT" in system["content"]
    sent = json.loads(user["content"])
    assert sent["research_task"] == inputs[0].model_dump(mode="json")
    expected_material = {key: value for key, value in inputs[1].items() if key != "tags"}
    expected_material["content"] = (
        "SOURCE CONTENT WITH LOCATORS\n\n[B001]\nMOCK / FIXTURE DATA.\n\n[B002]\n"
        "Planet Labs PBC management expects Data revenue to grow 30% in FY27 Q2."
    )
    assert sent["raw_source_material"] == expected_material


@pytest.mark.parametrize("locator", ["B999", "mock://guidance"], ids=["invented-block", "source-level-uri"])
def test_invalid_evidence_locator_rejected(inputs, payload, locator, monkeypatch):
    def forbidden_normalization(candidate):
        raise AssertionError("Locator validation must finish before variable normalization")

    monkeypatch.setattr(extractor_module, "normalize_variable_candidate", forbidden_normalization)
    invalid_evidence = deepcopy(payload["evidence"][0])
    invalid_evidence["source_locator"] = locator
    payload["evidence"].append(invalid_evidence)
    client = FakeClient(json.dumps(payload))
    with pytest.raises(ExtractionValidationError, match="Evidence candidate 1 has invalid source_locator"):
        DeepSeekExtractionBackend(client=client).extract(*inputs)
    assert len(client.calls) == 1


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


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_empty_response_raises(inputs, content):
    with pytest.raises(ExtractionValidationError, match="empty"):
        DeepSeekExtractionBackend(client=FakeClient(content)).extract(*inputs)


def test_missing_output_text_raises(inputs):
    client = FakeClient()
    del client.response.output_text
    with pytest.raises(ExtractionValidationError, match="empty"):
        DeepSeekExtractionBackend(client=client).extract(*inputs)


def test_truncated_response_rejected_even_if_json_is_valid(inputs, payload):
    client = FakeClient(json.dumps(payload), status="incomplete")
    client.response.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    with pytest.raises(ExtractionValidationError, match="truncated"):
        DeepSeekExtractionBackend(client=client).extract(*inputs)


def test_provider_failure_wrapped_once(inputs):
    failure = RuntimeError("simulated provider failure")
    client = FakeClient(error=failure)
    with pytest.raises(LLMProviderError) as exc:
        DeepSeekExtractionBackend(client=client).extract(*inputs)
    assert exc.value.__cause__ is failure
    assert len(client.calls) == 1


def test_failed_response_raises_provider_error(inputs, payload):
    client = FakeClient(json.dumps(payload), status="failed")
    with pytest.raises(LLMProviderError, match="response failed"):
        DeepSeekExtractionBackend(client=client).extract(*inputs)


@pytest.mark.parametrize("refusal", ["content-filter", "refusal-part"])
def test_refused_response_raises_provider_error(inputs, refusal):
    client = FakeClient()
    if refusal == "content-filter":
        client.response.status = "incomplete"
        client.response.incomplete_details = SimpleNamespace(reason="content_filter")
    else:
        client.response.output = [SimpleNamespace(
            type="message", content=[SimpleNamespace(type="refusal")],
        )]
    with pytest.raises(LLMProviderError, match="refused"):
        DeepSeekExtractionBackend(client=client).extract(*inputs)


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
