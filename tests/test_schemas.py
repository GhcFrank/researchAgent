"""Validation tests for the frozen shared schema, without Agent policy checks."""

from copy import deepcopy
from datetime import date
import json
from typing import get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from schemas import (
    CandidateGap,
    Claim,
    ClaimEvidenceStatus,
    Entity,
    Estimate,
    Event,
    Evidence,
    FollowUpCandidate,
    Gap,
    GapStatus,
    NotFoundItem,
    PotentialConflict,
    ResearchConstraints,
    ResearchResult,
    ResearchScope,
    ResearchTask,
    SearchCoverage,
    SearchMode,
    Source,
    SourceGrade,
    SourceOrigin,
    TargetRequirement,
    Variable,
    VariableInputType,
)


MINIMAL_INPUTS = {
    Entity: {
        "entity_id": "entity-planet",
        "entity_type": "company",
        "canonical_name": "Planet",
    },
    Source: {
        "source_id": "source-report",
        "title": "Mock company report",
        "source_type": "Company Report",
        "accessed_date": "2026-10-06",
        "locator": "fixtures/mock_planet_sources.json",
        "primary_or_secondary": SourceOrigin.PRIMARY,
    },
    Evidence: {
        "evidence_id": "evidence-revenue",
        "source_id": "source-report",
        "statement": "Mock reported revenue is 100 million USD.",
        "evidence_type": "Reported Fact",
        "source_locator": "table 3",
        "collected_at": "2026-10-06T12:00:00Z",
    },
    Claim: {
        "claim_id": "claim-growth",
        "claim": "Revenue is growing.",
        "claim_type": "Growth",
        "evidence_status": ClaimEvidenceStatus.SUPPORTED,
        "last_updated": "2026-10-06T12:00:00Z",
    },
    Gap: {
        "gap_id": "gap-customers",
        "question": "How concentrated are customer revenues?",
        "why_it_matters": "Assess customer dependence.",
        "status": GapStatus.UNKNOWN,
        "last_updated": "2026-10-06T12:00:00Z",
    },
    Variable: {
        "variable_id": "variable-revenue",
        "name": "Revenue",
        "variable_type": "Financial",
        "input_type": VariableInputType.OBSERVED,
        "last_updated": "2026-10-06T12:00:00Z",
    },
    Estimate: {
        "estimate_id": "estimate-growth",
        "output_variable_id": "variable-growth",
        "formula": "current_revenue / prior_revenue - 1",
        "reason_needed": "Compare growth rates.",
        "calculated_date": "2026-10-06",
    },
    Event: {
        "event_id": "event-report",
        "event_type": "Earnings",
        "status": "Scheduled",
        "description": "Next company earnings release.",
        "last_updated": "2026-10-06T12:00:00Z",
    },
    TargetRequirement: {
        "requirement_id": "requirement-growth",
        "question": "What drives growth?",
        "core_requirement": True,
    },
    ResearchScope: {"entity": "Planet"},
    ResearchConstraints: {},
    ResearchTask: {
        "task_id": "task-growth",
        "research_question": "What drives Planet's growth?",
        "target_requirement": {
            "requirement_id": "requirement-growth",
            "question": "What drives growth?",
            "core_requirement": True,
        },
        "search_mode": SearchMode.NORMAL,
        "scope": {"entity": "Planet"},
    },
    SearchCoverage: {
        "primary_source_found": True,
        "period_covered": True,
        "scope_covered": False,
    },
    PotentialConflict: {"description": "Two sources report different values."},
    NotFoundItem: {"item": "Customer count", "result": "Not disclosed"},
    CandidateGap: {"question": "What is the customer count?"},
    FollowUpCandidate: {
        "topic": "Customer concentration",
        "reason": "Explain revenue dependence.",
    },
    ResearchResult: {
        "task_id": "task-growth",
        "search_coverage": {
            "primary_source_found": True,
            "period_covered": True,
            "scope_covered": False,
        },
    },
}

MODELS = tuple(MINIMAL_INPUTS)
ENUM_FIELDS = (
    (Source, "primary_or_secondary", SourceOrigin, "primary"),
    (Source, "source_grade", SourceGrade, "G"),
    (Claim, "evidence_status", ClaimEvidenceStatus, "supported"),
    (Gap, "status", GapStatus, "Pending"),
    (Variable, "input_type", VariableInputType, "Model Estimate"),
    (ResearchTask, "search_mode", SearchMode, "automatic"),
)


def sample(model, **updates):
    return deepcopy(MINIMAL_INPUTS[model]) | updates


def json_payload(model):
    return model.model_validate(sample(model)).model_dump(mode="json")


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.__name__)
def test_valid_models_and_optional_defaults(model):
    # Includes all eight Research Objects, Task, Result, and every helper.
    instance = model.model_validate(sample(model))
    assert isinstance(instance, model)
    assert model.model_config["strict"] is True
    assert model.model_config["extra"] == "forbid"
    for name, field in model.model_fields.items():
        if field.default is None:
            assert getattr(instance, name) is None
            assert model.model_validate(sample(model, **{name: None})) == instance


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.__name__)
def test_extra_field_rejected(model):
    with pytest.raises(ValidationError) as exc:
        model.model_validate(sample(model, unexpected_field="unexpected"))
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("model, field, enum, invalid", ENUM_FIELDS)
def test_invalid_enum_rejected(model, field, enum, invalid):
    payload = json_payload(model)
    payload[field] = invalid
    with pytest.raises(ValidationError) as exc:
        model.model_validate_json(json.dumps(payload))
    assert exc.value.errors()[0]["loc"] == (field,)
    assert exc.value.errors()[0]["type"] == "enum"


@pytest.mark.parametrize("model, field, enum, invalid", ENUM_FIELDS)
def test_all_enum_members_valid_in_python_and_json(model, field, enum, invalid):
    # MODEL ESTIMATE and Derived remain valid in the shared Variable schema.
    for member in enum:
        instance = model.model_validate(sample(model, **{field: member}))
        assert getattr(instance, field) is member
        payload = json_payload(model)
        payload[field] = member.value
        restored = model.model_validate_json(json.dumps(payload))
        assert getattr(restored, field) is member


@pytest.mark.parametrize("model, field, enum, invalid", ENUM_FIELDS)
def test_strict_python_enum_requires_member(model, field, enum, invalid):
    member = next(iter(enum))
    with pytest.raises(ValidationError):
        model.model_validate(sample(model, **{field: member.value}))


@pytest.mark.parametrize(
    "model, updates",
    [
        (Entity, {"entity_id": 123}),
        (Entity, {"aliases": ("Planet Labs",)}),
        (Entity, {"aliases": [123]}),
        (Source, {"accessed_date": date(2026, 10, 6)}),
        (Evidence, {"value": True}),
        (TargetRequirement, {"core_requirement": "true"}),
        (SearchCoverage, {"primary_source_found": 1}),
        (ResearchTask, {"target_requirement": sample(TargetRequirement, core_requirement=1)}),
    ],
)
def test_strict_types_reject_coercion(model, updates):
    with pytest.raises(ValidationError):
        model.model_validate(sample(model, **updates))


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.__name__)
def test_model_dump_validation_round_trip(model):
    instance = model.model_validate(sample(model))
    assert model.model_validate(instance.model_dump()) == instance
    assert model.model_validate_json(instance.model_dump_json()) == instance


LIST_MODELS = tuple(
    model
    for model in MODELS
    if any(get_origin(field.annotation) is list for field in model.model_fields.values())
)


@pytest.mark.parametrize("model", LIST_MODELS, ids=lambda model: model.__name__)
def test_list_defaults_are_independent(model):
    first = model.model_validate(sample(model))
    second = model.model_validate(sample(model))
    for name, field in model.model_fields.items():
        if get_origin(field.annotation) is not list:
            continue
        assert field.default_factory is list
        first_list, second_list = getattr(first, name), getattr(second, name)
        assert first_list == second_list == []
        assert first_list is not second_list
        item_type = get_args(field.annotation)[0]
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            item = item_type.model_validate(sample(item_type))
        else:
            item = "test-item"
        first_list.append(item)
        assert second_list == []


def test_default_constraints_are_independent():
    first = ResearchTask.model_validate(sample(ResearchTask))
    second = ResearchTask.model_validate(sample(ResearchTask))
    assert first.constraints is not second.constraints
    first.constraints.excluded_sources.append("Rumor")
    assert second.constraints.excluded_sources == []


@pytest.mark.parametrize(
    "model, field",
    [
        (Entity, "entity_id"),
        (Source, "source_id"),
        (Evidence, "evidence_id"),
        (Claim, "claim_id"),
        (Gap, "gap_id"),
        (Variable, "variable_id"),
        (Estimate, "estimate_id"),
        (Event, "event_id"),
        (TargetRequirement, "requirement_id"),
        (ResearchTask, "task_id"),
        (ResearchResult, "task_id"),
    ],
)
def test_empty_required_id_rejected(model, field):
    for blank in ("", " \t\n"):
        with pytest.raises(ValidationError) as exc:
            model.model_validate(sample(model, **{field: blank}))
        assert exc.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize(
    "model, field",
    [
        (Entity, "entity_type"),
        (Entity, "canonical_name"),
        (Source, "title"),
        (Evidence, "statement"),
        (Evidence, "source_locator"),
        (Claim, "claim"),
        (Gap, "question"),
        (Variable, "name"),
        (Estimate, "formula"),
        (Event, "description"),
        (TargetRequirement, "question"),
        (ResearchTask, "research_question"),
        (PotentialConflict, "description"),
        (CandidateGap, "question"),
    ],
)
def test_empty_core_text_rejected(model, field):
    for blank in ("", " \t\n"):
        with pytest.raises(ValidationError) as exc:
            model.model_validate(sample(model, **{field: blank}))
        assert exc.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize(
    "model, updates",
    [
        (Entity, {"parent_entity_id": " "}),
        (Evidence, {"source_id": ""}),
        (Evidence, {"entity_ids": [" "]}),
        (Claim, {"supporting_evidence_ids": [""]}),
        (Gap, {"affected_variable_ids": [" "]}),
        (Variable, {"entity_id": ""}),
        (Estimate, {"output_variable_id": " "}),
        (Event, {"related_claim_ids": [""]}),
        (ResearchTask, {"existing_object_ids": [" "]}),
        (ResearchResult, {"sources_reused": [""]}),
    ],
)
def test_empty_reference_ids_rejected(model, updates):
    with pytest.raises(ValidationError):
        model.model_validate(sample(model, **updates))


@pytest.mark.parametrize("value", ["100", 100, 100.5, None])
def test_evidence_value_preserves_type(value):
    instance = Evidence.model_validate(sample(Evidence, value=value))
    assert instance.value == value
    assert type(instance.value) is type(value)


@pytest.mark.parametrize("value", ["100", 100, 100.5, True, False, None])
def test_variable_value_preserves_type(value):
    instance = Variable.model_validate(sample(Variable, value=value))
    assert instance.value == value
    assert type(instance.value) is type(value)


def test_unfrozen_taxonomies_and_period_strings_remain_open():
    entity = Entity.model_validate(sample(Entity, entity_type="custom entity type"))
    evidence = Evidence.model_validate(
        sample(Evidence, evidence_type="custom evidence type", period="FY27 Q2", scope="North America / 800G")
    )
    event = Event.model_validate(sample(Event, event_type="custom event type", status="custom status"))
    claim = Claim.model_validate(sample(Claim, period="2026H2"))
    variable = Variable.model_validate(sample(Variable, period="2027-2030"))
    assert entity.entity_type == "custom entity type"
    assert evidence.period == "FY27 Q2"
    assert evidence.scope == "North America / 800G"
    assert event.status == "custom status"
    assert claim.period == "2026H2"
    assert variable.period == "2027-2030"


def test_populated_research_result_round_trip():
    result = ResearchResult.model_validate(
        sample(
            ResearchResult,
            sources_created=[sample(Source, source_grade=SourceGrade.A)],
            sources_reused=["source-previous"],
            evidence_created=[sample(Evidence, value=100, entity_ids=["entity-planet"])],
            entities_created=[sample(Entity, aliases=["Planet Labs"])],
            variables_created=[sample(Variable, value=100, evidence_ids=["evidence-revenue"])],
            search_coverage=sample(SearchCoverage, source_types_checked=["Company Report"]),
            potential_conflicts=[sample(PotentialConflict, evidence_ids=["evidence-revenue"])],
            not_found=[sample(NotFoundItem, search_attempted=["Company Report"])],
            candidate_gaps=[sample(CandidateGap, why_it_matters="Assess dependence.")],
            follow_up_candidates=[sample(FollowUpCandidate)],
            research_notes="All source material is mock data.",
        )
    )
    assert isinstance(result.sources_created[0], Source)
    assert isinstance(result.evidence_created[0], Evidence)
    assert isinstance(result.entities_created[0], Entity)
    assert isinstance(result.variables_created[0], Variable)
    assert isinstance(result.candidate_gaps[0], CandidateGap)
    assert ResearchResult.model_validate(result.model_dump()) == result
    assert ResearchResult.model_validate_json(result.model_dump_json()) == result


def test_nested_extra_field_rejected():
    payload = sample(ResearchResult, entities_created=[sample(Entity, unexpected_field=True)])
    with pytest.raises(ValidationError) as exc:
        ResearchResult.model_validate(payload)
    assert exc.value.errors()[0]["loc"] == ("entities_created", 0, "unexpected_field")
    assert exc.value.errors()[0]["type"] == "extra_forbidden"
