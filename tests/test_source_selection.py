"""Deterministic candidate boundaries and fake metadata-only selection tests."""

from copy import deepcopy
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import source_selection as selection_module
from next_search_intent import NextSearchIntent
from research_tools import MockResearchTool, ResearchTool
from run_workspace import ResearchRunWorkspace
from schemas import ResearchTask
from sec_research_tool import SECResearchTool
from source_selection import (
    CandidateSourceSelection,
    CandidateSourceSelectionBackend,
    CandidateSourceSelector,
    DeepSeekCandidateSourceSelectionBackend,
    SelectedSourceCandidate,
    SourceSelectionProviderError,
    SourceSelectionValidationError,
    prefilter_source_candidates,
)
from storage import ResearchStorage


ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = date(2026, 10, 8)


class FakeBackend(CandidateSourceSelectionBackend):
    def __init__(self, selection):
        self.selection = selection
        self.calls = []

    def select(self, task, intent, eligible_candidates, *, run_date):
        self.calls.append((task, intent, deepcopy(eligible_candidates), run_date))
        return self.selection


class FakeClient:
    def __init__(self, payload=None, *, output_text=None, status="completed", error=None):
        self.calls = []
        self.error = error
        self.response = SimpleNamespace(
            output_text=json.dumps(payload) if payload is not None else output_text,
            status=status,
        )
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def prevent_real_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "selection-test-model")
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    def forbidden_sdk_client(**kwargs):
        raise AssertionError("Tests must inject a fake client")

    monkeypatch.setattr(selection_module, "OpenAI", forbidden_sdk_client)


@pytest.fixture
def inputs():
    task = ResearchTask.model_validate_json((ROOT / "examples" / "planet_growth.json").read_text(encoding="utf-8"))
    intent = NextSearchIntent.model_validate_json(json.dumps({
        "requirement_id": task.target_requirement.requirement_id,
        "target_aspects": ["Data + Analytics growth in FY27 Q2", "Data + Analytics revenue contribution in FY27 Q2"],
        "search_question": "What were Data + Analytics revenue growth and revenue contribution in FY27 Q2?",
        "search_terms": ["Data + Analytics", "revenue growth", "FY27 Q2"],
        "preferred_source_types": task.preferred_source_types,
        "rationale": "Find the missing quarterly financial disclosure.",
    }))
    return task, intent


def candidate(ref, form, filed, *, report="2026-07-31", cik="0000000123"):
    return {
        "source_ref": ref, "title": f"Synthetic {form} filing {filed}", "source_type": form,
        "filing_date": filed, "report_date": report, "publisher": "Mock issuer",
        "primary_or_secondary": "Primary", "locator": f"https://www.sec.gov/Archives/{ref}.htm",
        "accession_number": f"synthetic-{ref}", "primary_document": f"{ref}.htm", "cik": cik,
    }


def selection_payload(refs):
    return {
        "selected": [{"source_ref": ref, "rationale": "Metadata timing suggests relevance; content has not been read."} for ref in refs],
        "overall_rationale": "Choose the smallest complementary unread set likely to answer the target aspects.",
    }


@pytest.mark.parametrize(("form", "boundary", "one_day_old"), [
    ("10-Q", "2025-10-08", "2025-10-07"),
    ("8-K", "2026-04-08", "2026-04-07"),
])
def test_sec_horizon_is_inclusive_and_uses_filing_not_report_date(inputs, form, boundary, one_day_old):
    task, intent = inputs
    records = [
        candidate("at-boundary", form, boundary),
        candidate("old-filing-current-report", form, one_day_old),
        candidate("recent-filing-old-report", form, "2026-10-01", report="2001-01-01"),
    ]
    actual = prefilter_source_candidates(task, intent, records, run_date=RUN_DATE)
    assert [item["source_ref"] for item in actual.retained_candidates] == ["at-boundary", "recent-filing-old-report"]
    assert [item["source_ref"] for item in actual.filtered_by_time] == ["old-filing-current-report"]


def test_latest_ten_k_is_retained_per_issuer_even_outside_year_horizon(inputs):
    task, intent = inputs
    records = [
        candidate("a-old", "10-K", "2024-09-01", report="2024-07-31", cik="0000000001"),
        candidate("a-latest", "10-K", "2025-09-01", report="2025-07-31", cik="0000000001"),
        candidate("b-additional-recent", "10-K", "2025-12-01", cik="0000000002"),
        candidate("b-latest", "10-K", "2026-06-01", cik="0000000002"),
    ]
    actual = prefilter_source_candidates(task, intent, records, run_date=RUN_DATE)
    assert [item["source_ref"] for item in actual.retained_candidates] == ["a-latest", "b-additional-recent", "b-latest"]
    assert [item["source_ref"] for item in actual.filtered_by_time] == ["a-old"]


@pytest.mark.parametrize("requested_period", ["2024年", "比较FY25和FY27", "for 2024-07-31"])
def test_explicit_older_period_override_is_bounded_to_relevant_years(inputs, requested_period):
    task, intent = inputs
    task.target_requirement.question = f"What was business revenue {requested_period}?"
    intent.search_question = task.target_requirement.question
    intent.target_aspects = [f"Business revenue {requested_period}"]
    records = [
        candidate("requested-old", "10-Q", "2024-09-12", report="2024-07-31"),
        candidate("unrelated-old", "10-Q", "2022-09-12", report="2022-07-31"),
    ]
    actual = prefilter_source_candidates(task, intent, records, run_date=RUN_DATE)
    assert [item["source_ref"] for item in actual.retained_candidates] == ["requested-old"]
    assert [item["source_ref"] for item in actual.filtered_by_time] == ["unrelated-old"]


def test_amount_parent_question_and_rationale_do_not_expand_current_fiscal_horizon(inputs):
    task, intent = inputs
    task.research_question = "How did company revenue change during 2024?"
    task.target_requirement.question = "How did the reported value for 2024 units change in FY27 Q2?"
    intent.target_aspects = ["FY27 Q2 revenue associated with 2024 units"]
    intent.search_question = task.target_requirement.question
    intent.rationale = "Historical results for 2024 might be useful context."
    records = [candidate("old", "10-Q", "2024-09-12", report="2024-07-31")]
    actual = prefilter_source_candidates(task, intent, records, run_date=RUN_DATE)
    assert actual.retained_candidates == []
    task.target_requirement.question = "What was revenue for 2024?"
    actual = prefilter_source_candidates(task, intent, records, run_date=RUN_DATE)
    assert [item["source_ref"] for item in actual.retained_candidates] == ["old"]


def test_selection_uses_only_unread_metadata_and_does_not_search_read_or_persist(inputs, tmp_path, monkeypatch):
    """A fake minimal, aligned selection tests plumbing, not semantic LLM ranking."""
    task, intent = inputs
    records = [
        candidate("read-quarter", "10-Q", "2026-09-03"),
        candidate("aligned-8k", "8-K", "2026-09-03"),
        candidate("previous-quarter", "10-Q", "2025-12-10", report="2025-10-31"),
        candidate("too-old-8k", "8-K", "2026-03-27"),
    ]
    records[1]["extra_metadata"] = {"label": "preserve the caller snapshot"}
    originals = deepcopy(records)
    task_before, intent_before = task.model_dump(mode="json"), intent.model_dump(mode="json")
    workspace = ResearchRunWorkspace.create("mock-selection-test", "company", root_dir=tmp_path / "runs")
    storage = ResearchStorage(workspace.objects_dir)
    before = {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}

    def forbidden_tool_call(*args, **kwargs):
        raise AssertionError("Selection must not search or read a Tool")

    for tool_type in (ResearchTool, SECResearchTool, MockResearchTool):
        monkeypatch.setattr(tool_type, "search", forbidden_tool_call)
        monkeypatch.setattr(tool_type, "read", forbidden_tool_call)
    payload = selection_payload(["aligned-8k"])
    client = FakeClient(payload)
    actual = CandidateSourceSelector(DeepSeekCandidateSourceSelectionBackend(client=client)).select(
        task, intent, records, run_date=RUN_DATE, already_read_refs=["read-quarter", "read-quarter"],
    )
    assert actual.selection.model_dump(mode="json") == payload
    assert actual.filtering.already_read_refs == ["read-quarter"]
    retained = {item["source_ref"]: item for item in actual.filtering.retained_candidates}
    assert retained["read-quarter"]["already_read"] and not retained["read-quarter"]["selection_eligible"]
    assert [item["source_ref"] for item in actual.filtering.eligible_candidates] == ["aligned-8k", "previous-quarter"]
    for original in originals[:3]:
        actual_metadata = retained[original["source_ref"]]
        assert {key: actual_metadata[key] for key in original} == original
    assert actual.audit["run_date"] == RUN_DATE.isoformat()
    assert actual.audit["candidate_count_before_filter"] == 4
    assert actual.audit["candidate_count_after_time_filter"] == 3
    assert actual.audit["selected_sources"] == payload["selected"]
    assert actual.audit["selection_rationale"] == payload["overall_rationale"]
    assert records == originals
    retained["aligned-8k"]["extra_metadata"]["label"] = "mutated returned copy"
    assert records == originals
    assert task.model_dump(mode="json") == task_before and intent.model_dump(mode="json") == intent_before
    assert {path.name: path.read_bytes() for path in storage.data_dir.iterdir()} == before

    request = client.calls[0]
    assert len(client.calls) == 1
    assert request["model"] == "selection-test-model"
    assert request["temperature"] == 0 and request["stream"] is False
    assert request["max_output_tokens"] == 32768
    user = json.loads(request["input"][1]["content"])
    assert user["research_task"] == task_before and user["intent"] == intent_before
    assert user["run_date"] == RUN_DATE.isoformat() and user["max_selected_sources"] == 3
    assert [item["source_ref"] for item in user["eligible_candidates"]] == ["aligned-8k", "previous-quarter"]
    assert user["eligible_candidates"][0]["accession_number"] == records[1]["accession_number"]
    assert user["eligible_candidates"][0]["primary_document"] == records[1]["primary_document"]
    system = request["input"][0]["content"]
    for text in (
        "Select only exact source_ref values in eligible_candidates.",
        "Direct relevance: likelihood of resolving", "Period alignment: use report_date, filing timing",
        "Prefer primary, official, qualified sources", "normally choose 1-2 Sources",
        "Selection is based on likelihood of relevance, not", "confirmed content.",
        "exhibits, earnings-release exhibits, or earnings-call transcripts.",
        "claim that it definitely contains an earnings release.",
    ):
        assert text in system
    output_format = request["text"]["format"]
    assert output_format["type"] == "json_schema" and output_format["name"] == "candidate_source_selection"
    assert set(output_format["schema"]["properties"]) == {"selected", "overall_rationale"}


@pytest.mark.parametrize("failure", ["unknown", "filtered", "already-read", "duplicate", "too-many"])
def test_selected_refs_must_be_unique_unread_and_within_filtered_allowlist(inputs, failure):
    task, intent = inputs
    records = [
        candidate("eligible", "10-Q", "2026-09-03"),
        candidate("filtered", "8-K", "2026-03-27"),
        candidate("already-read", "10-Q", "2026-06-05"),
    ]
    refs = [failure] if failure in ("unknown", "filtered", "already-read") else ["eligible"] * (2 if failure == "duplicate" else 4)
    # Bypass initial construction so injected backend output is also validated.
    selected = [SelectedSourceCandidate(source_ref=ref, rationale="Synthetic selection.") for ref in refs]
    output = CandidateSourceSelection.model_construct(selected=selected, overall_rationale="Synthetic selection.")
    with pytest.raises(SourceSelectionValidationError):
        CandidateSourceSelector(FakeBackend(output)).select(
            task, intent, records, run_date=RUN_DATE, already_read_refs=["already-read"],
        )


def test_no_eligible_candidates_returns_empty_selection_without_llm(inputs):
    task, intent = inputs
    backend = FakeBackend(None)
    records = [candidate("read", "10-Q", "2026-09-03"), candidate("old", "8-K", "2026-03-27")]
    actual = CandidateSourceSelector(backend).select(task, intent, records, run_date=RUN_DATE, already_read_refs=["read"])
    assert actual.selection.selected == [] and actual.selection.overall_rationale
    assert actual.filtering.eligible_candidates == []
    assert backend.calls == []


@pytest.mark.parametrize("failure", ["generator-read-refs", "source-body-input"])
def test_wrong_metadata_contract_fails_before_backend(inputs, failure):
    task, intent = inputs
    record = candidate("eligible", "10-Q", "2026-09-03")
    refs = ()
    if failure == "generator-read-refs":
        refs = (ref for ref in ["eligible"])
    else:
        record["content"] = "Already-read source body is outside this metadata-only contract."
    backend = FakeBackend(None)
    with pytest.raises(SourceSelectionValidationError):
        CandidateSourceSelector(backend).select(task, intent, [record], run_date=RUN_DATE, already_read_refs=refs)
    assert backend.calls == []


@pytest.mark.parametrize("case", ["empty", "malformed-json", "incomplete"])
def test_invalid_provider_output_raises_explicit_validation_error(inputs, case):
    task, intent = inputs
    content = "" if case == "empty" else "{broken json" if case == "malformed-json" else json.dumps(selection_payload(["eligible"]))
    client = FakeClient(output_text=content, status="incomplete" if case == "incomplete" else "completed")
    with pytest.raises(SourceSelectionValidationError):
        CandidateSourceSelector(DeepSeekCandidateSourceSelectionBackend(client=client)).select(
            task, intent, [candidate("eligible", "10-Q", "2026-09-03")], run_date=RUN_DATE,
        )
    assert len(client.calls) == 1


def test_provider_exception_is_wrapped_without_retry(inputs):
    task, intent = inputs
    failure = RuntimeError("simulated provider failure")
    client = FakeClient(error=failure)
    with pytest.raises(SourceSelectionProviderError) as error:
        CandidateSourceSelector(DeepSeekCandidateSourceSelectionBackend(client=client)).select(
            task, intent, [candidate("eligible", "10-Q", "2026-09-03")], run_date=RUN_DATE,
        )
    assert error.value.__cause__ is failure and len(client.calls) == 1
