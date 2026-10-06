"""Shared Research Schema Contract v0.1.

Python validation requires Enum members in strict mode. JSON input uses the
Enum's string values via ``model_validate_json``. Dates remain plain strings.
"""

from enum import Enum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty or whitespace-only")
    return value


NonBlankString = Annotated[str, AfterValidator(_non_blank)]


class SourceOrigin(str, Enum):
    PRIMARY = "Primary"
    SECONDARY = "Secondary"


class SourceGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class ClaimEvidenceStatus(str, Enum):
    SUPPORTED = "Supported"
    CONFLICTED = "Conflicted"
    EVIDENCE_GAP = "Evidence Gap"


class GapStatus(str, Enum):
    UNKNOWN = "Unknown"
    PARTIALLY_KNOWN = "Partially Known"
    CONFLICTED = "Conflicted"
    NOT_PUBLICLY_OBSERVABLE = "Not Publicly Observable"
    RESOLVED = "Resolved"


class VariableInputType(str, Enum):
    OBSERVED = "Observed"
    GUIDANCE = "Guidance"
    THIRD_PARTY_ESTIMATE = "Third-party Estimate"
    MODEL_ESTIMATE = "MODEL ESTIMATE"
    DERIVED = "Derived"


class SearchMode(str, Enum):
    NORMAL = "normal"
    COUNTER_EVIDENCE = "counter_evidence"


class Entity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    entity_id: NonBlankString
    entity_type: NonBlankString
    canonical_name: NonBlankString
    aliases: list[str] = Field(default_factory=list)
    parent_entity_id: NonBlankString | None = None
    ticker: str | None = None
    geography: str | None = None


class Source(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    source_id: NonBlankString
    title: NonBlankString
    publisher: str | None = None
    source_type: str
    source_grade: SourceGrade | None = None
    published_date: str | None = None
    accessed_date: str
    locator: str
    primary_or_secondary: SourceOrigin
    independence_group: str | None = None


class Evidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    evidence_id: NonBlankString
    source_id: NonBlankString
    statement: NonBlankString
    value: str | int | float | None = None
    unit: str | None = None
    entity_ids: list[NonBlankString] = Field(default_factory=list)
    period: str | None = None
    scope: str | None = None
    evidence_type: str
    source_locator: NonBlankString
    collected_at: str
    notes: str | None = None


class Claim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    claim_id: NonBlankString
    claim: NonBlankString
    entity_ids: list[NonBlankString] = Field(default_factory=list)
    claim_type: str
    scope: str | None = None
    period: str | None = None
    supporting_evidence_ids: list[NonBlankString] = Field(default_factory=list)
    counter_evidence_ids: list[NonBlankString] = Field(default_factory=list)
    inference: str | None = None
    evidence_status: ClaimEvidenceStatus
    what_would_change_this_claim: list[str] = Field(default_factory=list)
    next_refresh_trigger: str | None = None
    last_updated: str


class Gap(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    gap_id: NonBlankString
    question: NonBlankString
    entity_ids: list[NonBlankString] = Field(default_factory=list)
    why_it_matters: str
    status: GapStatus
    affected_claim_ids: list[NonBlankString] = Field(default_factory=list)
    affected_variable_ids: list[NonBlankString] = Field(default_factory=list)
    search_path: list[str] = Field(default_factory=list)
    next_trigger: str | None = None
    last_updated: str


class Variable(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    variable_id: NonBlankString
    name: NonBlankString
    definition: str | None = None
    variable_type: str
    entity_id: NonBlankString | None = None
    period: str | None = None
    scope: str | None = None
    value: str | int | float | bool | None = None
    unit: str | None = None
    input_type: VariableInputType
    evidence_ids: list[NonBlankString] = Field(default_factory=list)
    last_updated: str


class Estimate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    estimate_id: NonBlankString
    output_variable_id: NonBlankString
    formula: NonBlankString
    input_variable_ids: list[NonBlankString] = Field(default_factory=list)
    input_evidence_ids: list[NonBlankString] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    range: str | None = None
    sensitivity: str | None = None
    reason_needed: str
    calculated_date: str


class Event(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    event_id: NonBlankString
    event_type: str
    entity_ids: list[NonBlankString] = Field(default_factory=list)
    date: str | None = None
    date_window: str | None = None
    status: str
    description: NonBlankString
    evidence_ids: list[NonBlankString] = Field(default_factory=list)
    related_claim_ids: list[NonBlankString] = Field(default_factory=list)
    related_variable_ids: list[NonBlankString] = Field(default_factory=list)
    expected_outcome: str | None = None
    positive_signal: list[str] = Field(default_factory=list)
    negative_signal: list[str] = Field(default_factory=list)
    last_updated: str


class TargetRequirement(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    requirement_id: NonBlankString
    question: NonBlankString
    core_requirement: bool


class ResearchScope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    entity: str
    period: str | None = None
    geography: str | None = None
    product_or_business: str | None = None


class ResearchConstraints(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    max_search_scope: str | None = None
    excluded_sources: list[str] = Field(default_factory=list)


class ResearchTask(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    task_id: NonBlankString
    research_question: NonBlankString
    target_requirement: TargetRequirement
    search_mode: SearchMode
    scope: ResearchScope
    preferred_source_types: list[str] = Field(default_factory=list)
    existing_object_ids: list[NonBlankString] = Field(default_factory=list)
    specific_search_instruction: str | None = None
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)


class SearchCoverage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    source_types_checked: list[str] = Field(default_factory=list)
    primary_source_found: bool
    period_covered: bool
    scope_covered: bool


class PotentialConflict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    description: NonBlankString
    evidence_ids: list[NonBlankString] = Field(default_factory=list)


class NotFoundItem(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    item: str
    search_attempted: list[str] = Field(default_factory=list)
    result: str


class CandidateGap(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    question: NonBlankString
    why_it_matters: str | None = None


class FollowUpCandidate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    topic: str
    reason: str


class ResearchResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    task_id: NonBlankString
    sources_created: list[Source] = Field(default_factory=list)
    sources_reused: list[NonBlankString] = Field(default_factory=list)
    evidence_created: list[Evidence] = Field(default_factory=list)
    entities_created: list[Entity] = Field(default_factory=list)
    variables_created: list[Variable] = Field(default_factory=list)
    search_coverage: SearchCoverage
    potential_conflicts: list[PotentialConflict] = Field(default_factory=list)
    not_found: list[NotFoundItem] = Field(default_factory=list)
    candidate_gaps: list[CandidateGap] = Field(default_factory=list)
    follow_up_candidates: list[FollowUpCandidate] = Field(default_factory=list)
    research_notes: str | None = None
