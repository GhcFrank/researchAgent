"""Validated JSON persistence for the eight shared Research Object types.

Object types in the public API are schema classes, e.g. ``list_objects(Source)``.
Missing collection files represent empty collections. Writes use an atomic
replacement of one file; concurrent writers and multi-file transactions are
outside the v0.1 contract.
"""

import json
import os
from pathlib import Path
import tempfile
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from schemas import Claim, Entity, Estimate, Event, Evidence, Gap, Source, Variable


class StorageError(Exception):
    """Base error for storage operations, including filesystem failures."""


class StorageValidationError(StorageError):
    """An input is unsupported, schema-invalid, or cannot be encoded as JSON."""


class DuplicateObjectError(StorageError):
    """An insert would overwrite an ID in the same collection."""


class ObjectNotFoundError(StorageError):
    """The requested object ID does not exist."""


class ReferenceValidationError(StorageError):
    """A direct reference points to an object that does not exist."""


class StorageCorruptionError(StorageError):
    """A collection contains invalid JSON, invalid objects, or duplicate IDs."""


ResearchObject = Entity | Source | Evidence | Claim | Gap | Variable | Estimate | Event
ObjectT = TypeVar("ObjectT", bound=BaseModel)

_COLLECTIONS = {
    Entity: ("entities.json", "entity_id"),
    Source: ("sources.json", "source_id"),
    Evidence: ("evidence.json", "evidence_id"),
    Claim: ("claims.json", "claim_id"),
    Gap: ("gaps.json", "gap_id"),
    Variable: ("variables.json", "variable_id"),
    Estimate: ("estimates.json", "estimate_id"),
    Event: ("events.json", "event_id"),
}

# Only the direct references specified in the Storage v0.1 contract.
_REFERENCES = {
    Evidence: (("source_id", Source), ("entity_ids", Entity)),
    Variable: (("entity_id", Entity), ("evidence_ids", Evidence)),
    Claim: (
        ("supporting_evidence_ids", Evidence),
        ("counter_evidence_ids", Evidence),
        ("entity_ids", Entity),
    ),
    Gap: (
        ("entity_ids", Entity),
        ("affected_claim_ids", Claim),
        ("affected_variable_ids", Variable),
    ),
    Estimate: (
        ("output_variable_id", Variable),
        ("input_variable_ids", Variable),
        ("input_evidence_ids", Evidence),
    ),
    Event: (
        ("entity_ids", Entity),
        ("evidence_ids", Evidence),
        ("related_claim_ids", Claim),
        ("related_variable_ids", Variable),
    ),
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


class ResearchStorage:
    def __init__(self, data_dir: str | Path = Path(__file__).resolve().parent / "data"):
        self.data_dir = Path(data_dir)

    def _collection(self, object_type: type[ObjectT]) -> tuple[Path, str]:
        try:
            filename, id_field = _COLLECTIONS[object_type]
        except (KeyError, TypeError) as exc:
            raise StorageValidationError(f"Unsupported object type: {object_type!r}") from exc
        return self.data_dir / filename, id_field

    def list_objects(self, object_type: type[ObjectT]) -> list[ObjectT]:
        """Read a complete collection, raising on any corrupt record."""
        path, id_field = self._collection(object_type)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except UnicodeDecodeError as exc:
            raise StorageCorruptionError(f"Invalid UTF-8 in {path}") from exc
        except OSError as exc:
            raise StorageError(f"Cannot read {path}: {exc}") from exc

        try:
            records = json.loads(text, parse_constant=_reject_json_constant)
        except ValueError as exc:
            raise StorageCorruptionError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise StorageCorruptionError(f"Expected a JSON list of objects in {path}")

        try:
            # Strict enums are represented by strings on disk, so use JSON-mode
            # schema validation rather than relaxing strict Python validation.
            objects = TypeAdapter(list[object_type]).validate_json(text)
        except ValidationError as exc:
            raise StorageCorruptionError(f"Invalid {object_type.__name__} data in {path}: {exc}") from exc

        seen = set()
        for obj in objects:
            object_id = getattr(obj, id_field)
            if object_id in seen:
                raise StorageCorruptionError(f"Duplicate {object_type.__name__} ID {object_id!r} in {path}")
            seen.add(object_id)
        return objects

    def get_by_id(self, object_type: type[ObjectT], object_id: str) -> ObjectT:
        """Return one object; a missing ID raises ObjectNotFoundError."""
        _, id_field = self._collection(object_type)
        for obj in self.list_objects(object_type):
            if getattr(obj, id_field) == object_id:
                return obj
        raise ObjectNotFoundError(f"{object_type.__name__} ID {object_id!r} not found")

    def _validate_object(self, obj: ObjectT) -> ObjectT:
        object_type = type(obj)
        self._collection(object_type)
        try:
            # Validating an existing model instance alone does not revalidate
            # mutations with the shared schema's configuration.
            return object_type.model_validate(obj.model_dump(warnings=False))
        except ValidationError as exc:
            raise StorageValidationError(f"Invalid {object_type.__name__} object: {exc}") from exc

    def _validate_references(self, obj: ResearchObject) -> None:
        available = {}
        for field, target_type in _REFERENCES.get(type(obj), ()):
            value = getattr(obj, field)
            references = value if isinstance(value, list) else ([] if value is None else [value])
            if not references:
                continue
            if target_type not in available:
                _, target_id_field = self._collection(target_type)
                available[target_type] = {
                    getattr(target, target_id_field) for target in self.list_objects(target_type)
                }
            missing = [reference for reference in references if reference not in available[target_type]]
            if missing:
                raise ReferenceValidationError(
                    f"{type(obj).__name__}.{field} references missing {target_type.__name__} IDs: {missing!r}"
                )

    def insert(self, obj: ObjectT) -> ObjectT:
        """Validate and insert an object without replacing an existing ID."""
        validated = self._validate_object(obj)
        object_type = type(validated)
        _, id_field = self._collection(object_type)
        object_id = getattr(validated, id_field)
        objects = self.list_objects(object_type)
        if any(getattr(existing, id_field) == object_id for existing in objects):
            raise DuplicateObjectError(f"{object_type.__name__} ID {object_id!r} already exists")
        self._validate_references(validated)
        self._write_collection(object_type, [*objects, validated])
        return validated

    def update(self, obj: ObjectT) -> ObjectT:
        """Validate and replace exactly one existing object with the same ID."""
        validated = self._validate_object(obj)
        object_type = type(validated)
        _, id_field = self._collection(object_type)
        object_id = getattr(validated, id_field)
        objects = self.list_objects(object_type)
        for index, existing in enumerate(objects):
            if getattr(existing, id_field) == object_id:
                self._validate_references(validated)
                objects[index] = validated
                self._write_collection(object_type, objects)
                return validated
        raise ObjectNotFoundError(f"{object_type.__name__} ID {object_id!r} not found")

    def find_duplicate_source(self, source: Source) -> Source | None:
        """Return the first exact locator or metadata match, without writing."""
        validated = self._validate_object(source)
        if type(validated) is not Source:
            raise StorageValidationError("find_duplicate_source requires a Source object")
        for existing in self.list_objects(Source):
            if existing.locator == validated.locator or (
                existing.title == validated.title
                and existing.publisher == validated.publisher
                and existing.published_date == validated.published_date
            ):
                return existing
        return None

    def _write_collection(self, object_type: type[ObjectT], objects: list[ObjectT]) -> None:
        path, _ = self._collection(object_type)
        try:
            text = json.dumps(
                [obj.model_dump(mode="json") for obj in objects],
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n"
        except (ValueError, TypeError) as exc:
            raise StorageValidationError(f"Cannot encode {object_type.__name__} objects as JSON: {exc}") from exc

        temporary_path = None
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.data_dir,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(text)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        except OSError as exc:
            raise StorageError(f"Cannot write {path}: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    # Cleanup failure must not mask the write error.
                    pass
