"""Tool-boundary tests with fake semantic decisions; no HTTP or provider calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import sec_research_tool as sec_module
import tool_routing as routing_module
from next_search_intent import NextSearchIntent
from research_tools import MockResearchTool, ResearchTool, ResearchToolSpec
from run_workspace import ResearchRunWorkspace
from schemas import ResearchTask
from sec_research_tool import SECResearchTool
from storage import ResearchStorage
from tool_registry import ResearchToolRegistry
from tool_routing import (
    DeepSeekToolRoutingBackend,
    SearchAction,
    ToolRouter,
    ToolRoutingBackend,
    ToolRoutingProviderError,
    ToolRoutingResult,
    ToolRoutingValidationError,
    ToolSelection,
)


ROOT = Path(__file__).resolve().parents[1]


class RecordingSEC(SECResearchTool):
    """Actual SEC contract identity, with only its HTTP implementation replaced."""

    def __init__(self):
        self.search_calls = []
        self.read_calls = []
        self.records = [{
            "source_ref": "sec:0000000123:0000000123-26-000001:quarter.htm",
            "title": "Synthetic quarterly filing", "source_type": "10-Q",
            "locator": "https://www.sec.gov/Archives/edgar/data/123/000000012326000001/quarter.htm",
        }]

    def _search(self, query):
        self.search_calls.append(query)
        return self.records

    def _read(self, source_ref):
        self.read_calls.append(source_ref)
        pytest.fail("Routing must not read SEC documents")


class RecordingMock(MockResearchTool):
    def __init__(self):
        super().__init__()
        self.search_calls = []
        self.read_calls = []

    def _search(self, query):
        self.search_calls.append(query)
        return super()._search(query)

    def _read(self, source_ref):
        self.read_calls.append(source_ref)
        pytest.fail("Routing must not read mock materials")


class ReadOnlyTool(ResearchTool):
    spec = ResearchToolSpec("document", "Read an existing local document.", ("read",), ("document",))


class UnadaptedTool(ResearchTool):
    spec = ResearchToolSpec("custom_search", "Search with an unspecified request contract.", ("search",), ())


class FakeBackend(ToolRoutingBackend):
    def __init__(self, tool_name, rationale="Synthetic routing rationale."):
        self.selection = ToolSelection(tool_name=tool_name, rationale=rationale)
        self.calls = []

    def select(self, task, intent, specs, *, ticker, search_contracts):
        self.calls.append({"task": task, "intent": intent, "specs": specs, "ticker": ticker, "search_contracts": search_contracts})
        return self.selection


class FakeClient:
    def __init__(self, *, output_text=None, status="completed", error=None):
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
def prevent_real_clients(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "routing-test-model")
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "SEC_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)

    def forbidden_client(*args, **kwargs):
        raise AssertionError("Tests must use fake provider and SEC paths")

    monkeypatch.setattr(routing_module, "OpenAI", forbidden_client)
    monkeypatch.setattr(sec_module, "build_opener", forbidden_client)


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


def registry_for(*tools):
    registry = ResearchToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_sec_routing_request_execution_audit_scope_and_no_persistence(inputs, tmp_path):
    """Fake SEC selection checks plumbing; it does not prove semantic suitability."""
    task, intent = inputs
    sec, mock = RecordingSEC(), RecordingMock()
    registry = registry_for(sec, mock)
    workspace = ResearchRunWorkspace.create("mock-routing-test", "company", root_dir=tmp_path / "runs")
    storage = ResearchStorage(workspace.objects_dir)
    before = {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}
    intent_before, task_before = intent.model_dump(mode="json"), task.model_dump(mode="json")
    rationale = "Primary SEC filings can provide quarterly financial disclosures; the supplied ticker makes search executable."
    client = FakeClient(output_text=json.dumps({"tool_name": "sec", "rationale": rationale}))
    result = ToolRouter(DeepSeekToolRoutingBackend(client=client)).execute_search(task, intent, registry, ticker=" pl ")

    assert result.routing.action.search_query == "PL"
    assert result.routing.action.tool_name == "sec"
    assert result.routing.no_tool_reason is None
    assert sec.search_calls == ["PL"] and sec.read_calls == []
    assert mock.search_calls == [] and mock.read_calls == []
    assert result.results == sec.records
    assert result.audit["intent"] == intent_before
    assert result.audit["selected_tool"] == "sec" and result.audit["routing_rationale"] == rationale
    assert result.audit["actual_search_request"] == {"query": "PL"}
    assert result.audit["search_result_refs"] == [item["source_ref"] for item in sec.records]
    assert [spec["name"] for spec in result.audit["available_tool_specs"]] == ["sec", "mock"]
    assert intent.model_dump(mode="json") == intent_before and task.model_dump(mode="json") == task_before
    assert {path.name: path.read_bytes() for path in storage.data_dir.iterdir()} == before

    request = client.calls[0]
    assert len(client.calls) == 1
    assert request["model"] == "routing-test-model"
    assert request["temperature"] == 0 and request["stream"] is False
    assert request["max_output_tokens"] == 32768
    user = json.loads(request["input"][1]["content"])
    assert user["research_task"] == task_before and user["intent"] == intent_before
    assert user["known_ticker"] == "PL"
    assert [spec["name"] for spec in user["available_tools"]] == ["sec", "mock"]
    assert user["available_tools"][0]["source_types"] == list(sec.spec.source_types)
    sec_contract = user["available_tools"][0]["search_contract"]
    assert "only the supplied public-company ticker" in sec_contract
    assert "No natural-language/full-text query, forms or limit arguments" in sec_contract
    system = request["input"][0]["content"]
    for text in (
        "Select only an exact name in available_tools.",
        "preferred_source_types are preferences, not a hard allowlist.",
        "mean the tool retrieves earnings-call transcripts or 8-K exhibits.",
        "Keep the supplied target_aspects and factual scope unchanged.",
        "Do not infer a ticker from company names or outside",
    ):
        assert text in system
    output_format = request["text"]["format"]
    assert output_format["type"] == "json_schema" and output_format["name"] == "tool_selection"
    assert set(output_format["schema"]["properties"]) == {"tool_name", "rationale"}


def test_registry_allowlist_rejects_invented_tool_before_search(inputs):
    task, intent = inputs
    sec, mock = RecordingSEC(), RecordingMock()
    with pytest.raises(ToolRoutingValidationError, match="not registered"):
        ToolRouter(FakeBackend("google")).execute_search(task, intent, registry_for(sec, mock), ticker="PL")
    assert sec.search_calls == mock.search_calls == []


def test_earnings_call_only_fake_no_tool_runs_no_search(inputs):
    task, intent = inputs
    intent.search_question = "What exact management statement was made on the earnings call?"
    intent.target_aspects = ["Earnings-call-only management commentary"]
    intent.preferred_source_types = ["earnings call"]
    sec = RecordingSEC()
    reason = "No currently registered ResearchTool retrieves earnings-call transcripts."
    backend = FakeBackend(None, reason)
    result = ToolRouter(backend).execute_search(task, intent, registry_for(sec), ticker="PL")
    assert result.routing.action is None and result.routing.no_tool_reason == reason
    assert result.results == [] and sec.search_calls == sec.read_calls == []
    assert len(backend.calls) == 1
    assert result.audit["actual_search_request"] is None and result.audit["selected_tool"] is None


def test_mock_adapter_reuses_existing_keyword_search(inputs):
    task, intent = inputs
    intent.search_terms = ["Data + Analytics", "revenue", "FY27 Q2"]
    mock = RecordingMock()
    result = ToolRouter(FakeBackend("mock")).execute_search(task, intent, registry_for(mock))
    query = " ".join(intent.search_terms)
    assert result.routing.action.search_query == query
    assert mock.search_calls == [query] and mock.read_calls == []
    assert result.results
    assert all(item["source_ref"].startswith("mock-") for item in result.results)


def test_exact_duplicate_action_ignores_intent_index_and_rationale(inputs):
    task, intent = inputs
    sec = RecordingSEC()
    history = [SearchAction(intent_index=99, tool_name="sec", search_query="PL", rationale="Earlier decision.")]
    result = ToolRouter(FakeBackend("sec", "New rationale.")).execute_search(
        task, intent, registry_for(sec), ticker="pl", intent_index=0, executed_actions=history,
    )
    assert result.routing.action is None and result.routing.no_tool_reason
    assert result.results == [] and sec.search_calls == sec.read_calls == []


@pytest.mark.parametrize("reason", ["missing-ticker", "read-only", "unadapted"])
def test_no_executable_contract_short_circuits_without_llm_or_search(inputs, reason):
    task, intent = inputs
    tool = RecordingSEC() if reason == "missing-ticker" else ReadOnlyTool() if reason == "read-only" else UnadaptedTool()
    backend = FakeBackend("sec")
    result = ToolRouter(backend).execute_search(task, intent, registry_for(tool))
    assert result.routing.action is None and result.routing.no_tool_reason
    assert result.results == [] and backend.calls == []
    if reason == "missing-ticker":
        assert tool.search_calls == tool.read_calls == []


def test_selected_read_only_tool_is_rejected_when_other_tool_is_executable(inputs):
    task, intent = inputs
    sec = RecordingSEC()
    backend = FakeBackend("document")
    with pytest.raises(ToolRoutingValidationError, match="no executable search contract"):
        ToolRouter(backend).execute_search(task, intent, registry_for(sec, ReadOnlyTool()), ticker="PL")
    assert len(backend.calls) == 1 and sec.search_calls == sec.read_calls == []


@pytest.mark.parametrize("case", ["empty", "malformed-json", "extra-request-field", "incomplete"])
def test_invalid_structured_provider_output_fails_before_search(inputs, case):
    task, intent = inputs
    content = json.dumps({"tool_name": "sec", "rationale": "Synthetic decision."})
    status = "completed"
    if case == "empty":
        content = ""
    elif case == "malformed-json":
        content = "{broken json"
    elif case == "extra-request-field":
        content = json.dumps({"tool_name": "sec", "rationale": "Synthetic decision.", "search_query": "Invented query"})
    else:
        status = "incomplete"
    client = FakeClient(output_text=content, status=status)
    sec = RecordingSEC()
    with pytest.raises(ToolRoutingValidationError):
        ToolRouter(DeepSeekToolRoutingBackend(client=client)).execute_search(task, intent, registry_for(sec), ticker="PL")
    assert len(client.calls) == 1 and sec.search_calls == sec.read_calls == []


def test_provider_exception_is_wrapped_without_retry(inputs):
    task, intent = inputs
    failure = RuntimeError("simulated provider failure")
    client = FakeClient(error=failure)
    sec = RecordingSEC()
    with pytest.raises(ToolRoutingProviderError) as error:
        ToolRouter(DeepSeekToolRoutingBackend(client=client)).execute_search(task, intent, registry_for(sec), ticker="PL")
    assert error.value.__cause__ is failure
    assert len(client.calls) == 1 and sec.search_calls == sec.read_calls == []


def test_model_configuration_is_required(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    with pytest.raises(ToolRoutingProviderError):
        DeepSeekToolRoutingBackend(client=FakeClient())


@pytest.mark.parametrize("has_action", [False, True])
def test_routing_result_requires_exactly_one_action_or_no_tool_reason(has_action):
    payload = {}
    if has_action:
        payload = {
            "action": SearchAction(intent_index=0, tool_name="mock", search_query="revenue", rationale="Synthetic selection."),
            "no_tool_reason": "Both outcomes are invalid.",
        }
    with pytest.raises(ValidationError, match="Exactly one"):
        ToolRoutingResult(**payload)
