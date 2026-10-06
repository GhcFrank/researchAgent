"""Fake Responses clients verify selection boundaries, not live semantic quality."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

import chunk_reranker as reranker_module
from chunk_reranker import (
    ChunkRerankerCandidate,
    ChunkRerankerProviderError,
    ChunkRerankerValidationError,
    ChunkSelectionResult,
    DeepSeekChunkRerankerBackend,
    SelectedChunk,
)
from schemas import ResearchTask
from source_retrieval import CandidateChunk
from source_segmentation import SourceBlock, SourceChunk, render_chunk


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
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def forbidden_client(**kwargs):
        raise AssertionError("Tests must inject a fake client")

    monkeypatch.setattr(reranker_module, "OpenAI", forbidden_client)


@pytest.fixture
def inputs():
    task = ResearchTask.model_validate_json(json.dumps({
        "task_id": "TASK-TEST",
        "research_question": "What drives subscription revenue growth?",
        "target_requirement": {
            "requirement_id": "REQ-TEST",
            "question": "What are subscription revenue growth and contribution in Q2?",
            "core_requirement": True,
        },
        "search_mode": "normal",
        "scope": {"entity": "Example Company", "period": "Q2", "product_or_business": "Subscriptions"},
    }))
    candidates = []
    for index, text in enumerate((
        "Subscription revenue growth was 20% in Q2.",
        "Subscription revenue represented 70% of company revenue in Q2.",
    ), start=1):
        block = SourceBlock(f"B{index:03d}", text)
        chunk = SourceChunk(f"C{index:03d}", (block,), len(f"[{block.block_id}]\n{text}"))
        lexical = CandidateChunk(
            chunk_id=chunk.chunk_id, score=12 - index,
            matched_phrases=("subscription revenue",), matched_terms=("revenue",),
            block_ids=(block.block_id,), match_type="direct_match", rank=index,
        )
        candidates.append(ChunkRerankerCandidate(chunk, lexical))
    return task, candidates


def test_structured_selection_prompt_and_model_order(inputs, monkeypatch):
    task, candidates = inputs
    before = deepcopy(candidates)
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")
    payload = {
        "selected": [
            {"chunk_id": "C002", "reason": "Reports subscription revenue contribution."},
            {"chunk_id": "C001", "reason": "Reports subscription revenue growth."},
        ],
        "uncovered_topics": ["Customer growth drivers are not disclosed."],
    }
    client = FakeClient(json.dumps(payload))
    result = DeepSeekChunkRerankerBackend(client=client).select(task, candidates)
    assert isinstance(result, ChunkSelectionResult)
    assert all(isinstance(selected, SelectedChunk) for selected in result.selected)
    assert result.model_dump() == payload  # Only chunk IDs/reasons/topics, no Research Objects.
    assert candidates == before
    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    assert "top_p" not in request
    assert request["text"]["format"] == {
        "type": "json_schema", "name": "chunk_selection", "schema": ChunkSelectionResult.model_json_schema(),
    }
    messages = request["input"]
    prompt = messages[0]["content"]
    assert all(rule in prompt for rule in ("Table of Contents", "Risk Factors", "smallest necessary set"))
    supplied = json.loads(messages[1]["content"])
    assert supplied["research_task"]["research_question"] == task.research_question
    assert supplied["research_task"]["target_requirement"] == task.target_requirement.model_dump()
    assert supplied["max_selected"] == 4
    assert supplied["candidates"] == [{
        "chunk_id": candidate.chunk.chunk_id,
        "lexical_score": candidate.lexical.score,
        "matched_phrases": list(candidate.lexical.matched_phrases),
        "matched_terms": list(candidate.lexical.matched_terms),
        "content": render_chunk(candidate.chunk),
    } for candidate in candidates]


@pytest.mark.parametrize(("selected", "max_selected"), [
    ([{"chunk_id": "C999", "reason": "Not supplied."}], 4),
    ([{"chunk_id": "C001", "reason": "First."}, {"chunk_id": "C001", "reason": "Duplicate."}], 4),
    ([{"chunk_id": "C001", "reason": "First."}, {"chunk_id": "C002", "reason": "Second."}], 1),
    ([{"chunk_id": "C001", "reason": " \t\n"}], 4),
])
def test_invalid_selection_rejected(inputs, selected, max_selected):
    client = FakeClient(json.dumps({"selected": selected}))
    with pytest.raises(ChunkRerankerValidationError):
        DeepSeekChunkRerankerBackend(client=client).select(*inputs, max_selected=max_selected)


def test_zero_selection_preserves_uncovered_topics(inputs):
    payload = {"selected": [], "uncovered_topics": ["No relevant customer explanation.", "Missing target metric."]}
    client = FakeClient(json.dumps(payload))
    result = DeepSeekChunkRerankerBackend(client=client).select(*inputs)
    assert result.model_dump() == payload


@pytest.mark.parametrize("output_text", [None, " \n", "not JSON", '{"selected":"C001"}'])
def test_empty_or_malformed_response_rejected(inputs, output_text):
    client = FakeClient(output_text)
    with pytest.raises(ChunkRerankerValidationError):
        DeepSeekChunkRerankerBackend(client=client).select(*inputs)


def test_research_object_output_rejected(inputs):
    client = FakeClient(json.dumps({"selected": [], "evidence": [{"statement": "Unexpected extraction."}]}))
    with pytest.raises(ChunkRerankerValidationError):
        DeepSeekChunkRerankerBackend(client=client).select(*inputs)


def test_provider_error_wrapped_without_retry(inputs):
    error = RuntimeError("Provider unavailable")
    client = FakeClient(error=error)
    with pytest.raises(ChunkRerankerProviderError) as raised:
        DeepSeekChunkRerankerBackend(client=client).select(*inputs)
    assert raised.value.__cause__ is error
    assert len(client.calls) == 1


@pytest.mark.parametrize("change", ["neighbor", "mismatched_id", "mismatched_blocks", "duplicate"])
def test_invalid_candidate_metadata_rejected_before_api(inputs, change):
    task, candidates = inputs
    first = candidates[0]
    changes = {
        "neighbor": {"match_type": "neighbor", "rank": None},
        "mismatched_id": {"chunk_id": "C999"},
        "mismatched_blocks": {"block_ids": ("B999",)},
        "duplicate": {},
    }
    invalid = ChunkRerankerCandidate(first.chunk, replace(first.lexical, **changes[change]))
    supplied = [first, invalid] if change == "duplicate" else [invalid]
    client = FakeClient('{"selected":[]}')
    with pytest.raises(ChunkRerankerValidationError):
        DeepSeekChunkRerankerBackend(client=client).select(task, supplied)
    assert client.calls == []
