"""Storage rules and data integrity, using only pytest temporary directories."""

import json
from pathlib import Path

import pytest

from run_workspace import ResearchRunWorkspace
from schemas import (
    Claim,
    ClaimEvidenceStatus,
    Entity,
    Estimate,
    Event,
    Evidence,
    Gap,
    GapStatus,
    Source,
    SourceOrigin,
    Variable,
    VariableInputType,
)
import storage as storage_module
from storage import (
    DuplicateObjectError,
    ObjectNotFoundError,
    ReferenceValidationError,
    ResearchStorage,
    StorageCorruptionError,
    StorageError,
    StorageValidationError,
)


FILES = {
    Entity: ("entities.json", "entity_id"),
    Source: ("sources.json", "source_id"),
    Evidence: ("evidence.json", "evidence_id"),
    Claim: ("claims.json", "claim_id"),
    Gap: ("gaps.json", "gap_id"),
    Variable: ("variables.json", "variable_id"),
    Estimate: ("estimates.json", "estimate_id"),
    Event: ("events.json", "event_id"),
}


@pytest.fixture
def storage(tmp_path):
    return ResearchStorage(data_dir=tmp_path / "data")


@pytest.fixture
def objects():
    # Dependency order: Entity/Source -> Evidence -> Variable/Claim -> others.
    return {
        Entity: Entity(entity_id="entity-1", entity_type="company", canonical_name="Mock company"),
        Source: Source(
            source_id="source-1",
            title="Mock report",
            publisher="Mock publisher",
            source_type="Company Report",
            published_date="2026-10-01",
            accessed_date="2026-10-06",
            locator="mock://planet/report",
            primary_or_secondary=SourceOrigin.PRIMARY,
        ),
        Evidence: Evidence(
            evidence_id="evidence-1",
            source_id="source-1",
            statement="Mock revenue is 100 million USD.",
            value=100,
            unit="million USD",
            entity_ids=["entity-1"],
            evidence_type="Reported Fact",
            source_locator="table 3",
            collected_at="2026-10-06T12:00:00Z",
        ),
        Variable: Variable(
            variable_id="variable-1",
            name="Projected revenue",
            variable_type="Financial",
            entity_id="entity-1",
            value=120.0,
            input_type=VariableInputType.MODEL_ESTIMATE,
            evidence_ids=["evidence-1"],
            last_updated="2026-10-06T12:00:00Z",
        ),
        Claim: Claim(
            claim_id="claim-1",
            claim="Mock revenue is growing.",
            entity_ids=["entity-1"],
            claim_type="Growth",
            supporting_evidence_ids=["evidence-1"],
            counter_evidence_ids=["evidence-1"],
            evidence_status=ClaimEvidenceStatus.CONFLICTED,
            last_updated="2026-10-06T12:00:00Z",
        ),
        Gap: Gap(
            gap_id="gap-1",
            question="What drives the growth?",
            entity_ids=["entity-1"],
            why_it_matters="Understand revenue drivers.",
            status=GapStatus.UNKNOWN,
            affected_claim_ids=["claim-1"],
            affected_variable_ids=["variable-1"],
            last_updated="2026-10-06T12:00:00Z",
        ),
        Estimate: Estimate(
            estimate_id="estimate-1",
            output_variable_id="variable-1",
            formula="observed_revenue * growth_factor",
            input_variable_ids=["variable-1"],
            input_evidence_ids=["evidence-1"],
            assumptions=["Growth factor is 1.2."],
            range="110-130 million USD",
            reason_needed="Project revenue.",
            calculated_date="2026-10-06",
        ),
        Event: Event(
            event_id="event-1",
            event_type="Earnings",
            entity_ids=["entity-1"],
            status="Scheduled",
            description="Next mock earnings release.",
            evidence_ids=["evidence-1"],
            related_claim_ids=["claim-1"],
            related_variable_ids=["variable-1"],
            last_updated="2026-10-06T12:00:00Z",
        ),
    }


@pytest.fixture
def populated_storage(storage, objects):
    for obj in objects.values():
        storage.insert(obj)
    return storage


def test_workspace_initializes_collections_and_persists_without_repo_pollution(tmp_path, objects):
    repo_data = Path(storage_module.__file__).resolve().parent / "data"
    repo_exists = repo_data.exists()
    repo_before = {path.relative_to(repo_data): path.read_bytes() for path in repo_data.rglob("*") if path.is_file()}
    workspace = ResearchRunWorkspace.create("PL", "company", root_dir=tmp_path / "runs")
    assert list(workspace.objects_dir.iterdir()) == []
    assert not workspace.root.is_relative_to(repo_data.parent)

    storage = ResearchStorage(workspace.objects_dir)
    assert storage.data_dir == workspace.objects_dir
    assert {path.name for path in storage.data_dir.iterdir()} == {filename for filename, _ in FILES.values()}
    for filename, _ in FILES.values():
        assert json.loads((storage.data_dir / filename).read_text(encoding="utf-8")) == []

    entity = objects[Entity].model_copy(update={"canonical_name": "PL"})
    assert storage.insert(entity) == entity
    assert storage.get_by_id(Entity, entity.entity_id) == entity
    assert storage.list_objects(Entity) == [entity]
    reopened = ResearchRunWorkspace.open(workspace.root, root_dir=tmp_path / "runs")
    reloaded = ResearchStorage(reopened.objects_dir)
    assert reloaded.get_by_id(Entity, entity.entity_id) == entity
    assert reloaded.list_objects(Entity) == [entity]
    assert repo_data.exists() == repo_exists
    assert {path.relative_to(repo_data): path.read_bytes() for path in repo_data.rglob("*") if path.is_file()} == repo_before


def test_workspace_storages_are_isolated(tmp_path, objects):
    first = ResearchRunWorkspace.create("PL", "company", root_dir=tmp_path / "runs")
    second = ResearchRunWorkspace.create("PL", "company", root_dir=tmp_path / "runs")
    storage_a, storage_b = ResearchStorage(first.objects_dir), ResearchStorage(second.objects_dir)
    entity_a = storage_a.insert(objects[Entity].model_copy(update={"canonical_name": "PL"}))
    assert storage_b.list_objects(Entity) == []
    with pytest.raises(ObjectNotFoundError, match=entity_a.entity_id):
        storage_b.get_by_id(Entity, entity_a.entity_id)

    # The same ID in another run is independent, not a duplicate in run A.
    entity_b = storage_b.insert(entity_a.model_copy(update={"canonical_name": "Another run"}))
    assert ResearchStorage(first.objects_dir).list_objects(Entity) == [entity_a]
    assert ResearchStorage(second.objects_dir).list_objects(Entity) == [entity_b]


def test_initialization_only_fills_missing_legacy_collections(tmp_path, objects):
    data_dir = tmp_path / "legacy-data"
    data_dir.mkdir()
    entity_path, source_path = data_dir / "entities.json", data_dir / "sources.json"
    entity_path.write_text(json.dumps([objects[Entity].model_dump(mode="json")]), encoding="utf-8")
    source_path.write_text("{broken", encoding="utf-8")
    before = {path: path.read_bytes() for path in (entity_path, source_path)}

    storage = ResearchStorage(data_dir=data_dir)
    assert storage.get_by_id(Entity, objects[Entity].entity_id) == objects[Entity]
    for path, contents in before.items():
        assert path.read_bytes() == contents
    with pytest.raises(StorageCorruptionError, match="sources.json"):
        storage.list_objects(Source)
    assert {path.name for path in data_dir.iterdir()} == {filename for filename, _ in FILES.values()}
    for model, (filename, _) in FILES.items():
        if model not in (Entity, Source):
            assert json.loads((data_dir / filename).read_text(encoding="utf-8")) == []


def test_all_object_types_persist_and_reload(storage, objects):
    assert storage.list_objects(Entity) == []
    for model, obj in objects.items():
        assert storage.insert(obj) == obj
        _, id_field = FILES[model]
        assert storage.get_by_id(model, getattr(obj, id_field)) == obj
        assert storage.list_objects(model) == [obj]

    reloaded = ResearchStorage(data_dir=storage.data_dir)
    for model, obj in objects.items():
        filename, id_field = FILES[model]
        loaded = reloaded.get_by_id(model, getattr(obj, id_field))
        assert type(loaded) is model
        assert loaded == obj
        assert reloaded.list_objects(model) == [obj]
        data = json.loads((storage.data_dir / filename).read_text(encoding="utf-8"))
        assert isinstance(data, list) and all(isinstance(row, dict) for row in data)
        assert data == [obj.model_dump(mode="json")]


def test_duplicate_id_rejected_without_overwrite(storage, objects):
    original = storage.insert(objects[Entity])
    duplicate = original.model_copy(update={"canonical_name": "Replacement"})
    with pytest.raises(DuplicateObjectError, match="entity-1"):
        storage.insert(duplicate)
    assert storage.get_by_id(Entity, original.entity_id) == original


def test_update_replaces_only_matching_id(storage, objects):
    original = storage.insert(objects[Entity])
    untouched = original.model_copy(update={"entity_id": "entity-2", "canonical_name": "Other"})
    storage.insert(untouched)
    changed = original.model_copy(update={"canonical_name": "Updated company"})
    assert storage.update(changed) == changed
    reloaded = ResearchStorage(data_dir=storage.data_dir)
    assert reloaded.list_objects(Entity) == [changed, untouched]


def test_missing_id_rejected_without_writing(storage, objects):
    before = {path: path.read_bytes() for path in storage.data_dir.iterdir()}
    with pytest.raises(ObjectNotFoundError, match="entity-1"):
        storage.update(objects[Entity])
    with pytest.raises(ObjectNotFoundError, match="entity-1"):
        storage.get_by_id(Entity, "entity-1")
    assert {path: path.read_bytes() for path in storage.data_dir.iterdir()} == before


def test_write_rejects_raw_dict_and_revalidates_model(storage, objects):
    original = storage.insert(objects[Entity])
    path = storage.data_dir / "entities.json"
    before = path.read_bytes()
    with pytest.raises(StorageValidationError):
        storage.insert(original.model_dump())
    # model_copy bypasses Pydantic validation; Storage must not trust it.
    invalid = original.model_copy(update={"canonical_name": 123})
    with pytest.raises(StorageValidationError):
        storage.update(invalid)
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", ["json", "root", "row", "schema", "duplicate_ids"])
def test_corrupt_collection_rejected_without_overwrite(storage, objects, failure):
    record = objects[Entity].model_dump(mode="json")
    contents = {
        "json": "[{",
        "root": json.dumps(record),
        "row": "[null]",
        "schema": json.dumps([record, {"entity_id": "entity-invalid"}]),
        "duplicate_ids": json.dumps([record, record]),
    }
    path = storage.data_dir / "entities.json"
    path.write_text(contents[failure], encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(StorageCorruptionError, match="entities.json"):
        storage.list_objects(Entity)
    with pytest.raises(StorageCorruptionError, match="entities.json"):
        storage.insert(objects[Entity].model_copy(update={"entity_id": "entity-new"}))
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "model, fields",
    [
        (Evidence, ("source_id", "entity_ids")),
        (Variable, ("entity_id", "evidence_ids")),
        (Claim, ("supporting_evidence_ids", "counter_evidence_ids", "entity_ids")),
        (Gap, ("entity_ids", "affected_claim_ids", "affected_variable_ids")),
        (Estimate, ("output_variable_id", "input_variable_ids", "input_evidence_ids")),
        (Event, ("entity_ids", "evidence_ids", "related_claim_ids", "related_variable_ids")),
    ],
    ids=["Evidence", "Variable", "Claim", "Gap", "Estimate", "Event"],
)
def test_invalid_direct_references_rejected(populated_storage, objects, model, fields):
    storage = populated_storage
    original = objects[model]
    filename, id_field = FILES[model]
    path = storage.data_dir / filename
    before = path.read_bytes()
    for field in fields:
        missing = ["missing-reference"] if isinstance(getattr(original, field), list) else "missing-reference"
        invalid_update = original.model_copy(update={field: missing})
        invalid_insert = invalid_update.model_copy(update={id_field: "new-object"})
        for operation, candidate in ((storage.insert, invalid_insert), (storage.update, invalid_update)):
            with pytest.raises(ReferenceValidationError, match=rf"{model.__name__}\.{field}"):
                operation(candidate)
            assert path.read_bytes() == before


def test_optional_variable_references_can_be_empty(storage, objects):
    variable = objects[Variable].model_copy(update={"entity_id": None, "evidence_ids": []})
    storage.insert(variable)
    assert storage.get_by_id(Variable, variable.variable_id) == variable


@pytest.mark.parametrize(
    "changes, duplicate",
    [
        ({"locator": "mock://planet/report", "title": "Other", "publisher": "Other", "published_date": None}, True),
        ({}, True),
        ({"title": "Other title"}, False),
        ({"publisher": "Other publisher"}, False),
        ({"published_date": "2026-09-30"}, False),
    ],
    ids=["same-locator", "same-metadata", "different-title", "different-publisher", "different-date"],
)
def test_source_deduplication_is_read_only(storage, objects, changes, duplicate):
    existing = storage.insert(objects[Source])
    candidate = existing.model_copy(
        update={"source_id": "source-new", "locator": "mock://different/report", **changes}
    )
    path = storage.data_dir / "sources.json"
    before = path.read_bytes()
    assert storage.find_duplicate_source(candidate) == (existing if duplicate else None)
    assert path.read_bytes() == before
    assert storage.list_objects(Source) == [existing]


def test_atomic_replace_failure_preserves_existing_file(storage, objects, monkeypatch):
    original = storage.insert(objects[Entity])
    path = storage.data_dir / "entities.json"
    before = path.read_bytes()
    before_files = {entry: entry.read_bytes() for entry in storage.data_dir.iterdir()}
    changed = original.model_copy(update={"canonical_name": "Updated company"})

    def fail_replace(temporary, destination):
        assert Path(temporary).parent == path.parent
        assert Path(destination) == path
        assert json.loads(Path(temporary).read_text(encoding="utf-8")) == [changed.model_dump(mode="json")]
        raise OSError("simulated replace failure")

    monkeypatch.setattr(storage_module.os, "replace", fail_replace)
    with pytest.raises(StorageError, match="simulated replace failure"):
        storage.update(changed)
    assert path.read_bytes() == before
    assert storage.get_by_id(Entity, original.entity_id) == original
    assert {entry: entry.read_bytes() for entry in storage.data_dir.iterdir()} == before_files
