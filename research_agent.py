"""Research Agent with offline mock extraction or incremental chunk extraction."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from llm_extractor import (
    EntityCandidate,
    EvidenceCandidate,
    ExtractionBackend,
    ExtractionError,
    ExtractionResult,
    VariableCandidate,
)
from research_tools import MockResearchTool, ResearchMaterial, ResearchTool, ResearchToolError
from run_source_store import RunSourceStore, RunSourceStoreError
from schemas import (
    CandidateGap,
    Entity,
    Evidence,
    FollowUpCandidate,
    NotFoundItem,
    PotentialConflict,
    ResearchResult,
    ResearchTask,
    SearchCoverage,
    SearchMode,
    Source,
    SourceOrigin,
    Variable,
    VariableInputType,
)
from storage import ObjectNotFoundError, ResearchStorage, StorageError
from source_retrieval import SourceRetrievalError, retrieve_candidate_chunks
from source_segmentation import SourceBlock, SourceSegmentationError, build_source_blocks, chunk_source_blocks


class ResearchAgentError(Exception):
    """The agent cannot complete the requested offline research operation."""


class AgentPermissionError(ResearchAgentError):
    """Extraction attempted to produce an object forbidden to this agent."""


_PROJECT_DIR = Path(__file__).resolve().parent
_LOGGER = logging.getLogger(__name__)
_ALLOWED_OBJECTS = {Entity, Source, Evidence, Variable}
_ALLOWED_INPUTS = {
    VariableInputType.OBSERVED,
    VariableInputType.GUIDANCE,
    VariableInputType.THIRD_PARTY_ESTIMATE,
}
_PRIMARY_TYPES = {"earnings release", "earnings call transcript", "contract announcement"}
_METRICS = {
    "revenue": (
        "Data + Analytics revenue",
        "million USD",
        (
            r"Data\s*\+\s*Analytics (?:revenue was|generated|revenue guidance is|revenue estimate is) USD (?P<value>\d+(?:\.\d+)?) million",
            r"reports USD (?P<value>\d+(?:\.\d+)?) million for Data\s*\+\s*Analytics",
        ),
    ),
    "growth": (
        "Data + Analytics YoY revenue growth",
        "%",
        (
            r"up (?P<value>\d+(?:\.\d+)?)% year over year",
            r"(?:Its )?(?P<value>\d+(?:\.\d+)?)% year-over-year growth",
        ),
    ),
    "share": (
        "Data + Analytics revenue contribution",
        "%",
        (
            r"represented (?P<value>\d+(?:\.\d+)?)% of total revenue",
            r"(?P<value>\d+(?:\.\d+)?)% contribution to total revenue",
            r"(?P<value>\d+(?:\.\d+)?)% of total company revenue",
        ),
    ),
}


def _stable_id(prefix: str, *parts) -> str:
    key = json.dumps([prefix, *parts], ensure_ascii=False, sort_keys=True)
    return f"{prefix}-{uuid5(NAMESPACE_URL, key).hex}"


def _observation_key(variable: Variable) -> tuple:
    """Compare measurement context without names or source-specific lineage."""
    def canonical(text: str | None) -> str | None:
        return " ".join(text.split()) if text is not None else None

    return (
        canonical(variable.entity_id),
        canonical(variable.variable_type),
        canonical(variable.period),
        canonical(variable.scope),
        # Values are never cleaned or coerced; bool, number and string differ.
        (type(variable.value).__name__, variable.value),
        canonical(variable.unit),
        canonical(variable.input_type.value),
    )


def _period(text: str) -> str | None:
    match = re.search(r"\bFY\d{2}\s+Q[1-4]\b", text, re.IGNORECASE)
    return " ".join(match.group().upper().split()) if match else None


def _build_query(task: ResearchTask) -> str:
    # Bounded lexical keywords, not query planning or general translation.
    context = "\n".join(filter(None, [
        task.research_question,
        task.target_requirement.question,
        task.scope.entity,
        task.scope.period,
        task.scope.geography,
        task.scope.product_or_business,
        task.specific_search_instruction,
    ]))
    stopwords = {"what", "is", "are", "the", "in", "and", "of", "for", "how", "much", "does", "current", "main", "pbc"}
    keywords = [word for word in re.findall(r"[a-z0-9]+", context.casefold()) if word not in stopwords]
    if "增长" in context:
        keywords.append("growth")
    if "收入" in context:
        keywords.append("revenue")
    return " ".join(dict.fromkeys(keywords))


def _retrieval_concepts(task: ResearchTask) -> tuple[list[str], list[str]]:
    """Reuse task keywords and its literal business scope; no provider planning."""
    phrases = [task.scope.product_or_business] if task.scope.product_or_business else []
    return phrases, _build_query(task).split()


def _validated_task(task: ResearchTask) -> ResearchTask:
    if type(task) is not ResearchTask:
        raise ResearchAgentError("run requires a ResearchTask")
    try:
        task = ResearchTask.model_validate(task.model_dump(warnings=False))
    except ValidationError as exc:
        raise ResearchAgentError(f"Invalid ResearchTask: {exc}") from exc
    if task.constraints.max_search_scope is not None:
        raise ResearchAgentError("v0.1 cannot interpret max_search_scope; use explicit scope and excluded_sources")
    return task


def _requested_metrics(task: ResearchTask) -> set[str]:
    question = task.target_requirement.question.casefold()
    requested = set()
    if "revenue" in question or "收入" in question:
        requested.add("revenue")
    if "growth" in question or "增长" in question:
        requested.add("growth")
    if any(word in question for word in ("share", "contribution", "贡献", "占比")):
        requested.add("share")
    return requested


def _extract_entities(materials: list[ResearchMaterial], task: ResearchTask) -> list[Entity]:
    if task.scope.entity.strip().casefold() not in {"planet labs", "planet labs pbc"}:
        return []
    if not any("planet labs" in (item["title"] + " " + item["content"]).casefold() for item in materials):
        return []
    return [Entity(
        entity_id=_stable_id("ENTITY", "planet labs pbc"),
        entity_type="company",
        canonical_name="Planet Labs PBC",
        aliases=["Planet Labs"],
    )]


def _extract_source(material: ResearchMaterial, timestamp: str) -> Source:
    source_type = material["source_type"].casefold()
    period = _period(material["content"])
    quotes_release = "same fictional release" in material["content"].casefold()
    if period and (source_type in {"earnings release", "earnings call transcript"} or quotes_release):
        independence_group = _stable_id("ORIGIN", "Planet Labs", period, "earnings disclosure")
    else:
        independence_group = _stable_id("ORIGIN", material["locator"])
    origin = material.get("primary_or_secondary")
    try:
        primary_or_secondary = SourceOrigin(origin) if origin is not None else (
            SourceOrigin.PRIMARY if source_type in _PRIMARY_TYPES else SourceOrigin.SECONDARY
        )
    except ValueError as exc:
        raise ResearchAgentError(f"Invalid source primary_or_secondary: {origin!r}") from exc
    return Source(
        source_id=_stable_id("SOURCE", material["locator"]),
        title=material["title"],
        publisher=material.get("publisher"),
        source_type=material["source_type"],
        published_date=material.get("published_date"),
        accessed_date=timestamp[:10],
        locator=material["locator"],
        primary_or_secondary=primary_or_secondary,
        independence_group=material.get("independence_group", independence_group),
    )


def _evidence_type(paragraph: str, source: Source) -> str:
    if re.search(r"\brevenue guidance is\b", paragraph, re.IGNORECASE):
        return "Management Guidance"
    if re.search(r"\brevenue estimate is\b", paragraph, re.IGNORECASE):
        return "Third-party Estimate"
    if source.source_type.casefold() == "earnings call transcript":
        return "Company Statement"
    return "Reported Fact"


def _extract_evidence(
    material: ResearchMaterial,
    source: Source,
    entity_ids: list[str],
    task: ResearchTask,
    timestamp: str,
) -> list[Evidence]:
    # Only the known mock company's Data + Analytics statements are supported.
    if not entity_ids or task.scope.geography is not None:
        return []
    business = task.scope.product_or_business
    if business is not None and re.sub(r"\W", "", business.casefold()) != "dataanalytics":
        return []
    requested = _requested_metrics(task)
    extracted = []
    for index, paragraph in enumerate(re.split(r"\n\s*\n", material["content"]), start=1):
        subject = re.search(r"Data\s*\+\s*Analytics", paragraph, re.IGNORECASE)
        period = _period(paragraph) or _period(material["content"]) or _period(material["title"])
        if not subject or not period:
            continue
        if task.scope.period and period.casefold() != task.scope.period.strip().casefold():
            continue
        evidence_type = _evidence_type(paragraph, source)
        locator = f"paragraph {index}"
        for key, (_, unit, patterns) in _METRICS.items():
            if key not in requested:
                continue
            # Exclude the total-company growth figure before the business name.
            text = paragraph if key == "revenue" else paragraph[subject.start():]
            match = next((found for pattern in patterns if (found := re.search(pattern, text, re.IGNORECASE))), None)
            if match is None:
                continue
            number = match.group("value")
            value = float(number) if "." in number else int(number)
            statement = match.group()
            extracted.append(Evidence(
                evidence_id=_stable_id("EVIDENCE", source.source_id, key, period, locator, statement, entity_ids),
                source_id=source.source_id,
                statement=statement,
                value=value,
                unit=unit,
                entity_ids=entity_ids,
                period=period,
                scope="Data + Analytics",
                evidence_type=evidence_type,
                source_locator=locator,
                collected_at=timestamp,
                notes=f"MOCK / FIXTURE data. Source context: {paragraph}",
            ))

        # Qualitative company statements are quoted, never turned into a thesis.
        growth_statement = re.search(
            r"Data\s*\+\s*Analytics (?:was the main growth contributor|is the main current growth business)[^.]*\.",
            paragraph,
            re.IGNORECASE,
        )
        if growth_statement and ("growth" in task.research_question.casefold() or "增长" in task.research_question):
            extracted.append(Evidence(
                evidence_id=_stable_id("EVIDENCE", source.source_id, "growth commentary", period, locator, growth_statement.group(), entity_ids),
                source_id=source.source_id,
                statement=growth_statement.group(),
                entity_ids=entity_ids,
                period=period,
                scope="Data + Analytics",
                evidence_type="Company Statement",
                source_locator=locator,
                collected_at=timestamp,
                notes="MOCK / FIXTURE data; quoted company statement, not an agent conclusion.",
            ))
    return extracted


def _metric_key(evidence: Evidence) -> str | None:
    if evidence.value is None:
        return None
    if evidence.unit == "million USD":
        return "revenue"
    if evidence.unit == "%" and "year" in evidence.statement.casefold():
        return "growth"
    if evidence.unit == "%" and "total" in evidence.statement.casefold():
        return "share"
    return None


def _input_type(evidence: Evidence) -> VariableInputType:
    return {
        "Management Guidance": VariableInputType.GUIDANCE,
        "Third-party Estimate": VariableInputType.THIRD_PARTY_ESTIMATE,
    }.get(evidence.evidence_type, VariableInputType.OBSERVED)


def _extract_variables(evidence: list[Evidence]) -> list[Variable]:
    variables = []
    for item in evidence:
        key = _metric_key(item)
        if key is None:
            continue
        name, _, _ = _METRICS[key]
        variables.append(Variable(
            variable_id=_stable_id("VARIABLE", item.evidence_id, key),
            name=name,
            definition="Source-stated metric; no calculation or aggregation by the agent.",
            variable_type="Financial",
            entity_id=item.entity_ids[0],
            period=item.period,
            scope=item.scope,
            value=item.value,
            unit=item.unit,
            input_type=_input_type(item),
            evidence_ids=[item.evidence_id],
            last_updated=item.collected_at,
        ))
    return variables


def _validate_batch(objects, expected_type):
    validated = []
    for obj in objects:
        if type(obj) not in _ALLOWED_OBJECTS:
            raise AgentPermissionError(f"Research Agent cannot create or persist {type(obj).__name__}")
        if type(obj) is Variable and obj.input_type not in _ALLOWED_INPUTS:
            raise AgentPermissionError(f"Forbidden Variable input_type: {obj.input_type}")
        if type(obj) is not expected_type:
            raise ResearchAgentError(f"Expected {expected_type.__name__}, received {type(obj).__name__}")
        try:
            validated.append(expected_type.model_validate(obj.model_dump(warnings=False)))
        except ValidationError as exc:
            raise ResearchAgentError(f"Invalid extracted {expected_type.__name__}: {exc}") from exc
    return validated


def _potential_conflicts(evidence: list[Evidence]) -> list[PotentialConflict]:
    groups = {}
    for item in evidence:
        key = _metric_key(item)
        if key:
            group = (key, item.period, item.scope, item.unit, _input_type(item))
            groups.setdefault(group, []).append(item)
    return [
        PotentialConflict(
            description=f"Different source-stated values for {_METRICS[key[0]][0]} in {key[1]}; requires review.",
            evidence_ids=[item.evidence_id for item in items],
        )
        for key, items in groups.items()
        if len({item.value for item in items}) > 1
    ]


def _validate_backend_result(result, source_blocks: tuple[SourceBlock, ...]) -> ExtractionResult:
    if type(result) is not ExtractionResult:
        raise ResearchAgentError("Extraction backend must return an ExtractionResult")
    for candidates, candidate_type in (
        (result.entities, EntityCandidate),
        (result.evidence, EvidenceCandidate),
        (result.variables, VariableCandidate),
    ):
        for candidate in candidates:
            if type(candidate) is not candidate_type:
                raise AgentPermissionError(
                    f"Backend cannot supply {type(candidate).__name__}; expected {candidate_type.__name__}"
                )
    for candidate in result.variables:
        if candidate.input_type not in _ALLOWED_INPUTS:
            raise AgentPermissionError(f"Forbidden Variable input_type: {candidate.input_type}")
    try:
        result = ExtractionResult.model_validate(result.model_dump(warnings=False))
    except ValidationError as exc:
        raise ResearchAgentError(f"Invalid backend ExtractionResult: {exc}") from exc
    block_ids = {block.block_id for block in source_blocks}
    for candidate in result.evidence:
        if candidate.source_locator not in block_ids:
            raise ResearchAgentError(f"Invalid backend source_locator: {candidate.source_locator!r}")
    return result


def _resolve_entity_name(name: str, entity_map: dict[str, str]) -> str:
    try:
        return entity_map[name.strip().casefold()]
    except KeyError as exc:
        raise ResearchAgentError(f"No EntityCandidate resolves entity name {name!r}") from exc


class ResearchAgent:
    def __init__(
        self,
        tool: ResearchTool,
        storage: ResearchStorage,
        prompt_path: Path | str = _PROJECT_DIR / "prompts" / "research_agent.md",
        *,
        extraction_backend: ExtractionBackend | None = None,
        source_store: RunSourceStore | None = None,
        search_query: str | None = None,
        source_refs: list[str] | None = None,
    ):
        self.tool = tool
        self.storage = storage
        self.extraction_backend = extraction_backend
        self.source_store = source_store
        if search_query is not None and (not isinstance(search_query, str) or not search_query.strip()):
            raise ResearchAgentError("search_query must be a non-blank string")
        if source_refs is not None and (
            not isinstance(source_refs, list)
            or any(not isinstance(ref, str) or not ref.strip() for ref in source_refs)
        ):
            raise ResearchAgentError("source_refs must be a list of non-blank references")
        # Retrieval tools define their own query syntax (e.g. SEC accepts tickers).
        # These explicit acquisition bounds do not modify the ResearchTask.
        self.search_query = search_query
        self.source_refs = None if source_refs is None else tuple(dict.fromkeys(source_refs))
        try:
            self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ResearchAgentError(f"Cannot load operating rules from {prompt_path}: {exc}") from exc
        if not self.prompt.strip():
            raise ResearchAgentError("Research Agent operating rules are empty")

    def _prepare_sources(self, materials, timestamp):
        sources_created, sources_reused, sources = [], [], []
        staged_locators, staged_metadata = {}, {}
        for material in materials:
            candidate = _validate_batch([_extract_source(material, timestamp)], Source)[0]
            metadata = (candidate.title, candidate.publisher, candidate.published_date)
            duplicate = self.storage.find_duplicate_source(candidate)
            duplicate = duplicate or staged_locators.get(candidate.locator) or staged_metadata.get(metadata)
            source = duplicate or candidate
            sources.append(source)
            if duplicate:
                if source.source_id not in sources_reused:
                    sources_reused.append(source.source_id)
            else:
                sources_created.append(source)
                staged_locators[source.locator] = source
                staged_metadata[metadata] = source
        return sources_created, sources_reused, sources

    def _new_objects(self, objects, object_type, id_field):
        new_objects = {}
        for obj in objects:
            object_id = getattr(obj, id_field)
            try:
                existing = self.storage.get_by_id(object_type, object_id)
            except ObjectNotFoundError:
                existing = new_objects.get(object_id)
            if existing is None:
                new_objects[object_id] = obj
            elif existing.model_dump(exclude={"collected_at", "last_updated"}) != obj.model_dump(exclude={"collected_at", "last_updated"}):
                raise ResearchAgentError(f"Conflicting content for deterministic {object_type.__name__} ID {object_id}")
        return list(new_objects.values())

    def _consolidate_variables(self, variables: list[Variable]) -> tuple[list[Variable], list[Variable]]:
        """Stage inserts and lineage updates before any persistence starts."""
        observations = {}
        for existing in self.storage.list_objects(Variable):
            # Reuse the first stored observation, including legacy Variable IDs.
            observations.setdefault(_observation_key(existing), existing)
        created, updated = {}, {}
        for candidate in variables:
            key = _observation_key(candidate)
            existing = observations.get(key)
            if existing is None:
                item = candidate.model_copy(update={
                    "variable_id": _stable_id("VARIABLE", key),
                    "evidence_ids": list(dict.fromkeys(candidate.evidence_ids)),
                })
                observations[key] = created[key] = item
                continue
            # Preserve old lineage order; append each new persistent ID once.
            evidence_ids = list(dict.fromkeys([*existing.evidence_ids, *candidate.evidence_ids]))
            if evidence_ids != existing.evidence_ids:
                item = existing.model_copy(update={
                    "evidence_ids": evidence_ids,
                    "last_updated": candidate.last_updated,
                })
                observations[key] = item
                if key in created:
                    created[key] = item
                else:
                    updated[key] = item
        return (
            _validate_batch(list(created.values()), Variable),
            _validate_batch(list(updated.values()), Variable),
        )

    def _persist_extraction_result(self, extracted, material, timestamp):
        """Validate one chunk's plan, resolve local indexes, then persist it."""
        sources_created, sources_reused, sources = self._prepare_sources([material], timestamp)
        source = sources[0]
        existing_entities = {
            item.canonical_name.strip().casefold(): item
            for item in self.storage.list_objects(Entity)
        }
        entities_created = []
        entity_map = {key: item.entity_id for key, item in existing_entities.items()}
        for candidate in extracted.entities:
            key = candidate.canonical_name.strip().casefold()
            entity = existing_entities.get(key)
            if entity is None:
                entity = Entity(entity_id=_stable_id("ENTITY", key), **candidate.model_dump())
                entities_created.append(entity)
                existing_entities[key] = entity
            entity_map[key] = entity.entity_id

        evidence, variables = [], []
        evidence_index_to_id = {}
        for index, candidate in enumerate(extracted.evidence):
            fields = candidate.model_dump(exclude={"entity_names"})
            fields["entity_ids"] = [_resolve_entity_name(name, entity_map) for name in candidate.entity_names]
            item = Evidence(
                evidence_id=_stable_id("EVIDENCE", source.source_id, fields),
                source_id=source.source_id, collected_at=timestamp, **fields,
            )
            evidence.append(item)
            evidence_index_to_id[index] = item.evidence_id
        for candidate in extracted.variables:
            try:
                evidence_ids = [evidence_index_to_id[index] for index in candidate.evidence_indexes]
            except KeyError as exc:
                raise ResearchAgentError(f"Unresolved candidate evidence index: {exc.args[0]}") from exc
            fields = candidate.model_dump(exclude={"entity_name", "evidence_indexes", "input_type"})
            fields.update(
                entity_id=_resolve_entity_name(candidate.entity_name, entity_map) if candidate.entity_name is not None else None,
                evidence_ids=evidence_ids,
                input_type=VariableInputType(candidate.input_type.value),
            )
            variables.append(Variable(
                variable_id=_stable_id("VARIABLE", fields), last_updated=timestamp, **fields,
            ))

        entities_created = _validate_batch(entities_created, Entity)
        sources_created = _validate_batch(sources_created, Source)
        evidence = _validate_batch(evidence, Evidence)
        variables = _validate_batch(variables, Variable)
        evidence_created = self._new_objects(evidence, Evidence, "evidence_id")
        variables_created, variables_updated = self._consolidate_variables(variables)
        for batch in (entities_created, sources_created, evidence_created, variables_created):
            for obj in batch:
                self.storage.insert(obj)
        for obj in variables_updated:
            self.storage.update(obj)
        return entities_created, sources_created, sources_reused, evidence_created, variables_created, evidence

    def _run_backend(self, task, materials, attempts, timestamp, *, retrieval_terms=None) -> ResearchResult:
        result = ResearchResult(
            task_id=task.task_id,
            search_coverage=SearchCoverage(
                source_types_checked=list(dict.fromkeys(item["source_type"] for item in materials)),
                primary_source_found=False, period_covered=False, scope_covered=False,
            ),
        )
        phrases, terms = _retrieval_concepts(task)
        if retrieval_terms is not None:
            terms = list(dict.fromkeys([*terms, *retrieval_terms]))
        diagnostic_keys = {name: set() for name in ("potential_conflicts", "candidate_gaps", "follow_up_candidates")}
        not_found_by_key, notes = {}, []
        selected_count = 0
        for material in materials:
            try:
                blocks = build_source_blocks(material["content"])
                chunks = chunk_source_blocks(blocks)
                candidates = retrieve_candidate_chunks(
                    chunks, phrases=phrases, terms=terms, top_k=8, neighbor_radius=0,
                )
            except (SourceSegmentationError, SourceRetrievalError) as exc:
                raise ResearchAgentError(f"Source retrieval failed for {material['source_ref']}: {exc}") from exc
            if not candidates:
                # A read Source remains an acquired source even when no chunk matches.
                created, reused, sources = self._prepare_sources([material], timestamp)
                for source in created:
                    self.storage.insert(source)
                result.sources_created.extend(created)
                created_source_ids = {source.source_id for source in result.sources_created}
                for source_id in reused:
                    if source_id not in created_source_ids and source_id not in result.sources_reused:
                        result.sources_reused.append(source_id)
                result.search_coverage.primary_source_found |= any(
                    source.primary_or_secondary is SourceOrigin.PRIMARY for source in sources
                )
            by_id = {chunk.chunk_id: chunk for chunk in chunks}
            # The existing retriever returns source order, including lexical ranks.
            for candidate in candidates:
                chunk = by_id[candidate.chunk_id]
                selected_count += 1
                context = f"{material['source_ref']}#{chunk.chunk_id}"
                _LOGGER.info("Extracting %s (%d blocks, %d chars)", context, len(chunk.blocks), chunk.char_count)
                try:
                    extracted = self.extraction_backend.extract(
                        task=task, material=material, operating_rules=self.prompt,
                        source_blocks=chunk.blocks,
                    )
                except ExtractionError as exc:
                    _LOGGER.error("Extraction backend failed for %s: %s", context, type(exc).__name__)
                    raise ResearchAgentError(f"Extraction backend failed for {context}: {exc}") from exc
                try:
                    extracted = _validate_backend_result(extracted, chunk.blocks)
                    batches = self._persist_extraction_result(extracted, material, timestamp)
                except (ResearchAgentError, StorageError, ValidationError) as exc:
                    _LOGGER.error("Chunk validation/persistence failed for %s: %s", context, type(exc).__name__)
                    exc.add_note(f"Source/chunk: {context}; earlier successful chunks remain persisted.")
                    raise
                entities_created, sources_created, sources_reused, evidence_created, variables_created, evidence = batches
                result.entities_created.extend(entities_created)
                result.sources_created.extend(sources_created)
                created_source_ids = {source.source_id for source in result.sources_created}
                for source_id in sources_reused:
                    if source_id not in created_source_ids and source_id not in result.sources_reused:
                        result.sources_reused.append(source_id)
                result.evidence_created.extend(evidence_created)
                result.variables_created.extend(variables_created)
                source_ids = [source.source_id for source in sources_created] + sources_reused
                result.search_coverage.primary_source_found |= any(
                    self.storage.get_by_id(Source, source_id).primary_or_secondary is SourceOrigin.PRIMARY
                    for source_id in source_ids
                )
                for item in evidence:
                    linked_entities = [self.storage.get_by_id(Entity, entity_id) for entity_id in item.entity_ids]
                    entity_covered = not task.scope.entity.strip() or any(
                        task.scope.entity.strip().casefold() in {
                            name.strip().casefold() for name in [entity.canonical_name, *entity.aliases]
                        } for entity in linked_entities
                    )
                    business_covered = task.scope.product_or_business is None or (
                        item.scope is not None and item.scope.strip().casefold() == task.scope.product_or_business.strip().casefold()
                    )
                    geography_covered = task.scope.geography is None or any(
                        entity.geography is not None and entity.geography.strip().casefold() == task.scope.geography.strip().casefold()
                        for entity in linked_entities
                    )
                    result.search_coverage.scope_covered |= entity_covered and business_covered and geography_covered
                    result.search_coverage.period_covered |= task.scope.period is None or (
                        item.period is not None and item.period.strip().casefold() == task.scope.period.strip().casefold()
                    )
                for name in diagnostic_keys:
                    for item in getattr(extracted, name):
                        key = item.model_dump_json()
                        if key not in diagnostic_keys[name]:
                            diagnostic_keys[name].add(key)
                            getattr(result, name).append(item)
                for item in extracted.not_found:
                    # Absence describes only this supplied chunk, never the Source/run.
                    key = (item.item, item.result)
                    search_attempted = list(dict.fromkeys([*attempts, context]))
                    if key in not_found_by_key:
                        saved = not_found_by_key[key]
                        saved.search_attempted = list(dict.fromkeys([*saved.search_attempted, *search_attempted]))
                    else:
                        saved = item.model_copy(update={
                            "search_attempted": search_attempted,
                            "result": "Chunk-local diagnostic only; not a Source/run absence: " + item.result,
                        })
                        not_found_by_key[key] = saved
                        result.not_found.append(saved)
                if extracted.research_notes and extracted.research_notes not in notes:
                    notes.append(extracted.research_notes)
        if not materials:
            result.not_found.append(NotFoundItem(
                item=task.target_requirement.question, search_attempted=attempts,
                result="No qualifying source material found",
            ))
            result.candidate_gaps.append(CandidateGap(question=task.target_requirement.question))
        elif not selected_count:
            result.not_found.append(NotFoundItem(
                item=task.target_requirement.question, search_attempted=attempts,
                result="No lexical candidate chunks selected; requirement coverage has not been established.",
            ))
        # Later chunks may extend a newly created Variable's evidence lineage.
        result.variables_created = [self.storage.get_by_id(Variable, item.variable_id) for item in result.variables_created]
        result.research_notes = "\n\n".join([
            "Incremental per-chunk extraction. Missing-information diagnostics refer only to supplied chunks; they do not establish Source/run absence.",
            *notes,
        ])
        return ResearchResult.model_validate(result.model_dump())

    def _snapshot_material(self, material, source_ref):
        if self.source_store is not None:
            try:
                self.source_store.put(material)
            except RunSourceStoreError as exc:
                raise ResearchAgentError(f"Source material persistence failed for {source_ref}: {exc}") from exc

    def run_known_sources(
        self,
        task: ResearchTask,
        source_refs: list[str],
        *,
        candidate_metadata: list[dict],
        retrieval_terms: list[str] | None = None,
    ) -> ResearchResult:
        """Read selected references directly and reuse incremental Stage 1 processing.

        Snapshotted or persisted sources are skipped; this is not a partial-run
        resume mechanism. Candidate metadata binds each ref to its exact locator.
        No search, routing, selection or coverage evaluation is performed.
        """
        task = _validated_task(task)
        if self.source_store is None or self.extraction_backend is None:
            raise ResearchAgentError("Known-source acquisition requires a source_store and extraction_backend")
        if not isinstance(source_refs, list) or any(not isinstance(ref, str) or not ref.strip() for ref in source_refs):
            raise ResearchAgentError("source_refs must be a list of non-blank references")
        if not isinstance(candidate_metadata, list):
            raise ResearchAgentError("candidate_metadata must be a list of source metadata")
        metadata_by_ref = {}
        for item in candidate_metadata:
            if not isinstance(item, dict) or any(
                not isinstance(item.get(field), str) or not item[field].strip()
                for field in ("source_ref", "locator")
            ):
                raise ResearchAgentError("Candidate metadata requires non-blank source_ref and locator")
            if item["source_ref"] in metadata_by_ref:
                raise ResearchAgentError("Duplicate source_ref in candidate_metadata")
            metadata_by_ref[item["source_ref"]] = item
        refs = list(dict.fromkeys(source_refs))
        if any(ref not in metadata_by_ref for ref in refs):
            raise ResearchAgentError("Selected source_ref is absent from candidate_metadata")
        excluded = {item.casefold() for item in task.constraints.excluded_sources}
        if any(
            str(value).casefold() in excluded
            for ref in refs for value in metadata_by_ref[ref].values()
        ):
            raise ResearchAgentError("Selected source is excluded by ResearchTask constraints")
        if retrieval_terms is not None and (
            not isinstance(retrieval_terms, list)
            or any(not isinstance(term, str) or not term.strip() for term in retrieval_terms)
        ):
            raise ResearchAgentError("retrieval_terms must be a list of non-blank strings")

        timestamp = datetime.now(timezone.utc).isoformat()
        existing_by_locator = {source.locator: source for source in self.storage.list_objects(Source)}
        acquired_locators = set(existing_by_locator)
        materials, skipped, reused = [], [], []
        for ref in refs:
            locator = metadata_by_ref[ref]["locator"]
            try:
                cached = self.source_store.get(ref)
            except RunSourceStoreError as exc:
                raise ResearchAgentError(f"Cannot inspect source snapshot for {ref}: {exc}") from exc
            if cached is not None and cached.get("locator") != locator:
                raise ResearchAgentError(f"Source snapshot locator does not match selected metadata for {ref}")
            existing = existing_by_locator.get(locator)
            if cached is not None or locator in acquired_locators:
                acquired_locators.add(locator)
                skipped.append(ref)
                if existing is not None and existing not in reused:
                    reused.append(existing)
                continue
            material = self.tool.read(ref)
            if not isinstance(material, dict) or material.get("source_ref") != ref or material.get("locator") != locator:
                raise ResearchAgentError(f"Read material source_ref/locator does not match selected metadata for {ref}")
            self._snapshot_material(material, ref)
            acquired_locators.add(locator)
            materials.append(material)

        # Metadata-only dedup must never attach new block IDs to another filing.
        # Validate pending sources together, including collisions within this batch.
        _, _, sources = self._prepare_sources(materials, timestamp)
        if any(source.locator != material["locator"] for source, material in zip(sources, materials)):
            raise ResearchAgentError("Source deduplication would reuse a different locator; selected-source provenance is ambiguous")
        result = self._run_backend(task, materials, [], timestamp, retrieval_terms=retrieval_terms) if materials else ResearchResult(
            task_id=task.task_id,
            search_coverage=SearchCoverage(
                source_types_checked=[], primary_source_found=False, period_covered=False, scope_covered=False,
            ),
        )
        for source in reused:
            if source.source_id not in result.sources_reused:
                result.sources_reused.append(source.source_id)
            if source.source_type not in result.search_coverage.source_types_checked:
                result.search_coverage.source_types_checked.append(source.source_type)
            result.search_coverage.primary_source_found |= source.primary_or_secondary is SourceOrigin.PRIMARY
        if skipped:
            result.research_notes = "\n\n".join(filter(None, [
                result.research_notes,
                "Skipped already acquired selected sources without download or extraction: " + ", ".join(skipped)
                + ". Partial-run resume and requirement coverage evaluation were not performed.",
            ]))
        return ResearchResult.model_validate(result.model_dump())

    def run(self, task: ResearchTask) -> ResearchResult:
        task = _validated_task(task)

        timestamp = datetime.now(timezone.utc).isoformat()
        query = self.search_query if self.search_query is not None else _build_query(task)
        attempts = [query] if query else []
        summaries = self.tool.search(query) if query else []
        if self.source_refs is not None:
            summaries = [item for item in summaries if item["source_ref"] in self.source_refs]
        excluded = {item.casefold() for item in task.constraints.excluded_sources}
        summaries = [item for item in summaries if not any(str(value).casefold() in excluded for value in item.values())]
        preferences = [item.casefold() for item in task.preferred_source_types]
        summaries.sort(key=lambda item: next(
            (index for index, preferred in enumerate(preferences) if preferred in item["source_type"].casefold()),
            len(preferences),
        ))
        materials = []
        for ref in dict.fromkeys(item["source_ref"] for item in summaries):
            material = self.tool.read(ref)
            self._snapshot_material(material, ref)
            materials.append(material)

        if self.extraction_backend is not None:
            return self._run_backend(task, materials, attempts, timestamp)

        entities = _validate_batch(_extract_entities(materials, task), Entity)
        existing_entities = {item.canonical_name.strip().casefold(): item for item in self.storage.list_objects(Entity)}
        entities_created = []
        effective_entities = []
        for candidate in entities:
            existing = existing_entities.get(candidate.canonical_name.strip().casefold())
            effective_entities.append(existing or candidate)
            if existing is None:
                entities_created.append(candidate)
                existing_entities[candidate.canonical_name.strip().casefold()] = candidate

        sources_created, sources_reused, sources = self._prepare_sources(materials, timestamp)

        evidence = []
        entity_ids = [item.entity_id for item in effective_entities]
        for material, source in zip(materials, sources):
            evidence.extend(_validate_batch(_extract_evidence(material, source, entity_ids, task, timestamp), Evidence))
        variables = _validate_batch(_extract_variables(evidence), Variable)

        # Validate the complete plan before the first storage write, including
        # extraction paths injected or replaced by a future model integration.
        entities_created = _validate_batch(entities_created, Entity)
        sources_created = _validate_batch(sources_created, Source)
        evidence = _validate_batch(evidence, Evidence)
        variables = _validate_batch(variables, Variable)
        evidence_created = self._new_objects(evidence, Evidence, "evidence_id")
        variables_created, variables_updated = self._consolidate_variables(variables)

        requested = _requested_metrics(task)
        found = {_metric_key(item) for item in evidence}
        missing = [_METRICS[key][0] for key in _METRICS if key in requested and key not in found]
        if not requested:
            missing = [task.target_requirement.question]
        not_found = [NotFoundItem(item=item, search_attempted=attempts, result="No qualifying evidence found") for item in missing]
        candidate_gaps = [CandidateGap(
            question=f"What source supports {item}?",
            why_it_matters="Required information was not found in the bounded search." if task.target_requirement.core_requirement else None,
        ) for item in missing]
        follow_ups = []
        if any("do not disclose a detailed breakdown" in item["content"].casefold() for item in materials):
            follow_ups.append(FollowUpCandidate(
                topic="Government and commercial revenue breakdown within Data + Analytics",
                reason="The mock business brief explicitly says this breakdown is undisclosed; no further search was started.",
            ))

        notes = "Offline deterministic mock extraction; no general NLP or model inference. Existing Evidence IDs are reused; matching Variable observations aggregate evidence lineage."
        if task.search_mode is SearchMode.COUNTER_EVIDENCE:
            notes += " Counter-evidence mode checks the same bounded sources for divergent reported values; it does not resolve conflicts."
        if task.scope.geography is not None:
            notes += " Geographic extraction is unsupported in v0.1; no broader-scope evidence was substituted."
        result = ResearchResult(
            task_id=task.task_id,
            entities_created=entities_created,
            sources_created=sources_created,
            sources_reused=sources_reused,
            evidence_created=evidence_created,
            variables_created=variables_created,
            search_coverage=SearchCoverage(
                source_types_checked=list(dict.fromkeys(item["source_type"] for item in materials)),
                primary_source_found=any(item.primary_or_secondary is SourceOrigin.PRIMARY for item in sources),
                period_covered=bool(evidence),
                scope_covered=bool(evidence),
            ),
            potential_conflicts=_potential_conflicts(evidence),
            not_found=not_found,
            candidate_gaps=candidate_gaps,
            follow_up_candidates=follow_ups,
            research_notes=notes,
        )
        # Storage owns JSON persistence and reference validation.
        for batch in (entities_created, sources_created, evidence_created, variables_created):
            for obj in batch:
                self.storage.insert(obj)
        for obj in variables_updated:
            self.storage.update(obj)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline Research Agent v0.1")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=_PROJECT_DIR / "data")
    parser.add_argument("--fixture", type=Path, default=_PROJECT_DIR / "fixtures" / "mock_planet_sources.json")
    parser.add_argument("--prompt", type=Path, default=_PROJECT_DIR / "prompts" / "research_agent.md")
    args = parser.parse_args(argv)
    try:
        task = ResearchTask.model_validate_json(args.task.read_text(encoding="utf-8"))
        agent = ResearchAgent(MockResearchTool(args.fixture), ResearchStorage(args.data_dir), args.prompt)
        result = agent.run(task)
    except (ResearchAgentError, ResearchToolError, StorageError, ValidationError, OSError) as exc:
        print(f"Research failed: {exc}", file=sys.stderr)
        return 1
    print(f"Task: {result.task_id}")
    for label, items in (
        ("Sources created", result.sources_created),
        ("Sources reused", result.sources_reused),
        ("Entities created", result.entities_created),
        ("Evidence created", result.evidence_created),
        ("Variables created", result.variables_created),
        ("Candidate gaps", result.candidate_gaps),
        ("Not found", result.not_found),
    ):
        print(f"{label}: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
