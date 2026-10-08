"""Coverage grounding and boundary tests; semantic responses are supplied fakes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import requirement_coverage as coverage_module
from requirement_coverage import (
    DeepSeekRequirementCoverageBackend,
    RequirementCoverageBackend,
    RequirementCoverageEvaluation,
    RequirementCoverageEvaluator,
    RequirementCoverageProviderError,
    RequirementCoverageStatus,
    RequirementCoverageValidationError,
)
from schemas import Evidence, ResearchResult, ResearchTask, Source, SourceOrigin
from storage import ResearchStorage


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, payload=None, *, output_text=None, error=None):
        self.calls = []
        self.error = error
        self.output_text = json.dumps(payload) if payload is not None else output_text
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=self.output_text, status="completed", incomplete_details=None)


class FakeCoverageBackend(RequirementCoverageBackend):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def evaluate(self, task, evidence):
        self.calls.append((task, list(evidence)))
        return self.result


@pytest.fixture(autouse=True)
def prevent_real_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "coverage-test-model")
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    def forbidden_sdk_client(**kwargs):
        raise AssertionError("Tests must inject a fake client")

    monkeypatch.setattr(coverage_module, "OpenAI", forbidden_sdk_client)


@pytest.fixture
def task():
    return ResearchTask.model_validate_json((ROOT / "examples" / "planet_growth.json").read_text(encoding="utf-8"))


@pytest.fixture
def storage(tmp_path):
    store = ResearchStorage(tmp_path / "objects")
    store.insert(Source(
        source_id="SOURCE-test", title="Synthetic disclosure", publisher="Mock publisher",
        source_type="Earnings Release", published_date="2026-10-01", accessed_date="2026-10-02",
        locator="mock://coverage", primary_or_secondary=SourceOrigin.PRIMARY,
    ))
    return store


def evidence(evidence_id, statement, *, value=None, unit=None, period="FY27 Q2", scope="Data + Analytics"):
    return Evidence(
        evidence_id=evidence_id, source_id="SOURCE-test", statement=statement, value=value, unit=unit,
        period=period, scope=scope, evidence_type="Reported Fact", source_locator="B001",
        collected_at="2026-10-02T12:00:00+00:00",
    )


def stage1_result(task, items=()):
    return ResearchResult.model_validate_json(json.dumps({
        "task_id": task.task_id,
        "evidence_created": [item.model_dump(mode="json") for item in items],
        "search_coverage": {"primary_source_found": True, "period_covered": False, "scope_covered": False},
    }))


def evaluation_payload(task, status, *, supporting=(), conflicting=(), uncovered=(), rationale="Fake audit explanation."):
    return {
        "requirement_id": task.target_requirement.requirement_id, "status": status,
        "supporting_evidence_ids": list(supporting), "conflicting_evidence_ids": list(conflicting),
        "uncovered_aspects": list(uncovered), "rationale": rationale,
    }


def storage_snapshot(storage):
    return {path.name: path.read_bytes() for path in storage.data_dir.iterdir()}


@pytest.mark.parametrize("scenario", [
    "fully-covered", "growth-only", "wrong-period", "real-conflict", "different-periods",
    "different-scopes", "explicit-nondisclosure", "related-background",
])
def test_fake_semantic_outcomes_preserve_grounded_contract(task, storage, scenario):
    """Canned A–E/G decisions test plumbing, not the real model's semantic accuracy."""
    growth = evidence("E-GROWTH", "Data + Analytics revenue grew 20% in FY27 Q2.", value=20, unit="%")
    share = evidence("E-SHARE", "Data + Analytics represented 75% of FY27 Q2 revenue.", value=75, unit="%")
    if scenario == "fully-covered":
        items = [growth, share]
        payload = evaluation_payload(task, "Covered", supporting=[item.evidence_id for item in items])
    elif scenario == "growth-only":
        growth = growth.model_copy(update={"statement": "Data + Analytics revenue grew 14% in FY27 Q2.", "value": 14})
        items = [growth]
        payload = evaluation_payload(task, "Uncovered", supporting=[growth.evidence_id], uncovered=["Revenue contribution"])
    elif scenario == "wrong-period":
        items = [growth.model_copy(update={"period": "FY26 Q2", "statement": "Revenue grew 20% in FY26 Q2."})]
        payload = evaluation_payload(task, "Uncovered", uncovered=["FY27 Q2 growth and revenue contribution"])
    elif scenario == "real-conflict":
        other = evidence("E-OTHER", "Data + Analytics revenue grew 31% in FY27 Q2.", value=31, unit="%")
        items = [growth, other]
        payload = evaluation_payload(task, "Conflicted", conflicting=[item.evidence_id for item in items])
    elif scenario == "different-periods":
        other = evidence("E-OTHER", "Data + Analytics revenue grew 30% in FY26 Q2.", value=30, unit="%", period="FY26 Q2")
        items = [growth, other]
        payload = evaluation_payload(task, "Uncovered", supporting=[growth.evidence_id], uncovered=["Revenue contribution"])
    elif scenario == "different-scopes":
        other = evidence("E-OTHER", "Total company revenue grew 30% in FY27 Q2.", value=30, unit="%", scope="Total company")
        items = [growth, other]
        payload = evaluation_payload(task, "Uncovered", supporting=[growth.evidence_id], uncovered=["Data + Analytics revenue contribution"])
    elif scenario == "related-background":
        items = [evidence("E-BACKGROUND", "The company sells subscription access to imagery through an API.", period=None)]
        payload = evaluation_payload(
            task, "Uncovered", uncovered=["FY27 Q2 revenue growth", "FY27 Q2 revenue contribution"],
            rationale="The subscription and API description does not answer any material factual component of the Target Requirement.",
        )
    else:
        items = [evidence("E-DISCLOSURE", "The company does not separately disclose Data + Analytics growth or revenue contribution.", period=None)]
        payload = evaluation_payload(task, "Not Publicly Observable", supporting=[items[0].evidence_id])
    for item in items:
        storage.insert(item)
    before = storage_snapshot(storage)
    client = FakeClient(payload)
    evaluator = RequirementCoverageEvaluator(DeepSeekRequirementCoverageBackend(client=client))

    actual = evaluator.evaluate(task, stage1_result(task, items), storage)

    assert actual.model_dump(mode="json") == payload
    assert len(client.calls) == 1
    assert actual.requirement_id == task.target_requirement.requirement_id
    assert set(actual.supporting_evidence_ids + actual.conflicting_evidence_ids) <= {item.evidence_id for item in items}
    assert storage_snapshot(storage) == before  # Includes unchanged claims/gaps and all eight object files.


def test_no_evidence_is_uncovered_locally_without_model_or_object_writes(task, storage):
    # Historic storage contents do not become factual input for this task.
    storage.insert(evidence("E-HISTORY", "An unrelated historical metric."))
    backend = FakeCoverageBackend(None)
    before = storage_snapshot(storage)

    actual = RequirementCoverageEvaluator(backend).evaluate(task, stage1_result(task), storage)

    assert actual.status is RequirementCoverageStatus.UNCOVERED
    assert actual.requirement_id == task.target_requirement.requirement_id
    assert actual.uncovered_aspects == ["All material factual components of the Target Requirement"]
    assert not actual.supporting_evidence_ids and not actual.conflicting_evidence_ids
    assert backend.calls == []
    assert storage_snapshot(storage) == before


def test_current_task_ids_load_storage_truth_and_explicit_reuse_without_selecting_history(task, storage, monkeypatch):
    current = storage.insert(evidence("E-CURRENT", "Data + Analytics revenue grew 20%.", value=20, unit="%"))
    reused = storage.insert(evidence("E-REUSED", "Data + Analytics revenue contribution was 75%.", value=75, unit="%"))
    history = storage.insert(evidence("E-HISTORY", "Historical revenue grew 99%.", value=99, unit="%", period="FY26"))
    task.existing_object_ids = [history.evidence_id]
    stale_result = stage1_result(task, [current.model_copy(update={"value": 999, "statement": "Untrusted result copy."})])
    payload = evaluation_payload(task, "Covered", supporting=[current.evidence_id, reused.evidence_id])
    backend = FakeCoverageBackend(RequirementCoverageEvaluation.model_validate_json(json.dumps(payload)))

    reads = []
    get_by_id = storage.get_by_id

    def record_get_by_id(object_type, object_id):
        reads.append((object_type, object_id))
        return get_by_id(object_type, object_id)

    monkeypatch.setattr(storage, "get_by_id", record_get_by_id)
    before = storage_snapshot(storage)
    actual = RequirementCoverageEvaluator(backend).evaluate(
        task, stale_result, storage, reused_evidence_ids=[reused.evidence_id, current.evidence_id, reused.evidence_id],
    )
    assert reads == [(Evidence, current.evidence_id), (Evidence, reused.evidence_id)]
    assert backend.calls == [(task, [current, reused])]
    assert actual.supporting_evidence_ids == [current.evidence_id, reused.evidence_id]
    assert storage_snapshot(storage) == before


@pytest.mark.parametrize("failure", ["wrong-task", "missing-persisted-evidence"])
def test_invalid_stage1_input_fails_before_backend(task, storage, failure):
    item = evidence("E-NOT-PERSISTED", "A synthetic observation.")
    result = stage1_result(task, [item])
    if failure == "wrong-task":
        result.task_id = "OTHER-TASK"
    backend = FakeCoverageBackend(None)
    with pytest.raises(RequirementCoverageValidationError):
        RequirementCoverageEvaluator(backend).evaluate(task, result, storage)
    assert backend.calls == []


@pytest.mark.parametrize("failure", [
    "unknown-support", "unknown-conflict", "wrong-requirement", "covered-without-support",
    "covered-with-missing-part", "conflict-with-one-id", "npo-without-support",
    "uncovered-without-aspects", "nonconflicted-with-conflict-ids",
])
def test_local_output_guards_reject_invalid_backend_evaluation(task, storage, failure):
    item = storage.insert(evidence("E-VALID", "A source-stated metric.", value=20, unit="%"))
    payload = evaluation_payload(task, "Covered", supporting=[item.evidence_id])
    if failure == "unknown-support":
        payload["supporting_evidence_ids"] = ["E-INVENTED"]
    elif failure == "unknown-conflict":
        payload.update(status="Conflicted", conflicting_evidence_ids=[item.evidence_id, "E-INVENTED"])
    elif failure == "wrong-requirement":
        payload["requirement_id"] = "OTHER-REQUIREMENT"
    elif failure == "covered-without-support":
        payload["supporting_evidence_ids"] = []
    elif failure == "covered-with-missing-part":
        payload["uncovered_aspects"] = ["Revenue contribution"]
    elif failure == "conflict-with-one-id":
        payload.update(status="Conflicted", conflicting_evidence_ids=[item.evidence_id])
    elif failure == "npo-without-support":
        payload.update(status="Not Publicly Observable", supporting_evidence_ids=[])
    elif failure == "uncovered-without-aspects":
        payload.update(status="Uncovered")
    else:
        payload.update(status="Uncovered", uncovered_aspects=["Revenue contribution"], conflicting_evidence_ids=[item.evidence_id])
    # Bypass model creation deliberately: the evaluator must validate injected backend output.
    payload["status"] = RequirementCoverageStatus(payload["status"])
    backend = FakeCoverageBackend(RequirementCoverageEvaluation.model_construct(**payload))
    before = storage_snapshot(storage)
    with pytest.raises(RequirementCoverageValidationError):
        RequirementCoverageEvaluator(backend).evaluate(task, stage1_result(task, [item]), storage)
    assert storage_snapshot(storage) == before


def test_request_contains_task_evidence_allowlist_schema_and_coverage_rules(task):
    task_before = task.model_dump(mode="json")
    item = evidence("E-VALID", "Revenue grew 20%.", value=20, unit="%")
    payload = evaluation_payload(
        task, "Uncovered", supporting=[item.evidence_id], uncovered=["Revenue contribution"],
        rationale="The Evidence reports FY27 Q2 revenue growth but does not provide revenue contribution.",
    )
    client = FakeClient(payload)
    actual = DeepSeekRequirementCoverageBackend(client=client).evaluate(task, [item])
    request = client.calls[0]
    assert request["model"] == "coverage-test-model"
    assert request["temperature"] == 0 and request["stream"] is False
    assert request["max_output_tokens"] == 32768
    assert "reasoning" not in request
    assert [message["role"] for message in request["input"]] == ["system", "user"]
    user = json.loads(request["input"][1]["content"])
    assert user["research_question"] == task.research_question
    assert user["target_requirement"] == task.target_requirement.model_dump(mode="json")
    assert user["scope"] == task.scope.model_dump(mode="json")
    assert user["evidence"] == [item.model_dump(mode="json")]
    assert user["evidence_id_allowlist"] == [item.evidence_id]
    system = request["input"][0]["content"]
    for text in (
        "The supplied Evidence objects are the only factual basis",
        "The Target Requirement is the object being evaluated.",
        "provides context only and must not broaden the criteria for Covered.",
        "Covered requires all material components to be answered collectively by Evidence",
        "No relevant Evidence or a partial answer is",
        "Uncovered, never Partially Covered.",
        "with matching entity, period, scope and metric definition.",
        "Differences explained by different",
        "periods, scopes, definitions, or entities are not conflicts;",
        "Not Publicly Observable requires Evidence explicitly establishing",
        "Absence alone means Uncovered, not Not Publicly Observable.",
        "Never use outside knowledge",
        "SUPPORTING EVIDENCE",
        "supporting_evidence_ids must contain only Evidence that directly supports",
        "at least one material factual component of the Target Requirement.",
        "Evidence that is merely related background, context, business description,",
        "or adjacent information must not be included.",
        "supplied Evidence, supporting_evidence_ids must be empty.",
        "For an Uncovered evaluation, supporting_evidence_ids may still be non-empty",
        "only when those Evidence objects genuinely answer one or more material",
        "OUTPUT LANGUAGE",
        "Return all machine-consumed free-text fields in English.",
        "This includes:\n- uncovered_aspects\n- rationale",
    ):
        assert text.casefold() in system.casefold()
    output_format = request["text"]["format"]
    assert output_format["type"] == "json_schema" and output_format["name"] == "requirement_coverage"
    assert set(output_format["schema"]["properties"]) == {
        "requirement_id", "status", "supporting_evidence_ids", "conflicting_evidence_ids", "uncovered_aspects", "rationale",
    }
    assert set(member.value for member in RequirementCoverageStatus) == {
        "Uncovered", "Covered", "Conflicted", "Not Publicly Observable",
    }
    assert actual.status is RequirementCoverageStatus.UNCOVERED
    assert actual.rationale == payload["rationale"]
    assert actual.uncovered_aspects == ["Revenue contribution"]
    assert task.model_dump(mode="json") == task_before


@pytest.mark.parametrize("output", ["", "{broken json"])
def test_empty_or_malformed_output_is_explicit_validation_error(task, output):
    backend = DeepSeekRequirementCoverageBackend(client=FakeClient(output_text=output))
    with pytest.raises(RequirementCoverageValidationError):
        backend.evaluate(task, [evidence("E-VALID", "A synthetic observation.")])


def test_provider_error_is_wrapped_without_retry(task):
    failure = RuntimeError("simulated provider failure")
    client = FakeClient(error=failure)
    with pytest.raises(RequirementCoverageProviderError) as error:
        DeepSeekRequirementCoverageBackend(client=client).evaluate(task, [evidence("E-VALID", "A synthetic observation.")])
    assert error.value.__cause__ is failure
    assert len(client.calls) == 1


def test_model_configuration_is_required(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    with pytest.raises(RequirementCoverageProviderError):
        DeepSeekRequirementCoverageBackend(client=FakeClient())
