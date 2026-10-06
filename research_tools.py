"""External-source retrieval only, with immutable metadata and plain dict results.

search results require source_ref, title, source_type and locator; read materials
also require content. Optional metadata includes publisher, published_date,
primary_or_secondary, independence_group and content_type. Tools do not create
Research Objects, extract evidence, reason, determine completion, or write Storage.
Mock retrieval preserves its existing fixture fields and behavior.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import NotRequired, TypedDict


class SearchResult(TypedDict):
    source_ref: str
    title: str
    source_type: str
    locator: str
    publisher: NotRequired[str | None]
    published_date: NotRequired[str | None]
    primary_or_secondary: NotRequired[str]
    independence_group: NotRequired[str]
    content_type: NotRequired[str]


class ResearchMaterial(SearchResult):
    content: str
    tags: NotRequired[list[str]]


class ResearchToolError(Exception):
    """Base error for retrieval operations."""


class FixtureLoadError(ResearchToolError):
    """The fixture cannot be read or contains invalid material records."""


class SourceNotFoundError(ResearchToolError):
    """The requested source_ref is absent from the fixture."""


class UnsupportedToolCapabilityError(ResearchToolError):
    """The requested action is absent from the tool's declared capabilities."""


@dataclass(frozen=True)
class ResearchToolSpec:
    """Stable tool identity and guidance for when its retrieval actions are useful."""

    name: str
    description: str
    capabilities: tuple[str, ...]
    source_types: tuple[str, ...] = ()

    def __post_init__(self):
        for field in ("name", "description"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ResearchToolSpec.{field} must be a non-blank string")
        if (
            not isinstance(self.capabilities, tuple)
            or not self.capabilities
            or any(not isinstance(action, str) or action not in ("search", "read") for action in self.capabilities)
            or len(set(self.capabilities)) != len(self.capabilities)
        ):
            raise ValueError("ResearchToolSpec.capabilities must be a non-empty tuple of unique search/read values")
        if not isinstance(self.source_types, tuple) or any(
            not isinstance(source_type, str) or not source_type.strip() for source_type in self.source_types
        ):
            raise ValueError("ResearchToolSpec.source_types must be a tuple of non-blank strings")


class ResearchTool(ABC):
    """Implement spec and the supported _search/_read hooks.

    Public search/read entrypoints enforce capabilities before retrieving any
    material. Overrides that add logging should delegate to these entrypoints.
    """

    @property
    @abstractmethod
    def spec(self) -> ResearchToolSpec:
        """Return tool metadata; a concrete class may supply a class attribute."""

    def search(self, query: str) -> list[SearchResult]:
        """Return lightweight metadata for matching raw materials."""
        self._require_capability("search")
        return self._search(query)

    def read(self, source_ref: str) -> ResearchMaterial:
        """Return a complete raw material or raise SourceNotFoundError."""
        self._require_capability("read")
        return self._read(source_ref)

    def _require_capability(self, action: str) -> None:
        if action not in self.spec.capabilities:
            raise UnsupportedToolCapabilityError(f"Tool {self.spec.name!r} does not support {action!r}")

    def _search(self, query: str) -> list[SearchResult]:
        raise ResearchToolError(f"Tool {self.spec.name!r} declares search but has no _search implementation")

    def _read(self, source_ref: str) -> ResearchMaterial:
        raise ResearchToolError(f"Tool {self.spec.name!r} declares read but has no _read implementation")


_SUMMARY_FIELDS = (
    "source_ref", "title", "publisher", "source_type", "published_date", "locator"
)
_TEXT_FIELDS = (*_SUMMARY_FIELDS, "content")
_DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mock_planet_sources.json"


class MockResearchTool(ResearchTool):
    """Search a validated fixture snapshot; no network or persistence calls.

    All query keywords must occur in the combined title, publisher, source type,
    tags, and content (case-insensitive substring matching). Punctuation splits
    keywords. Results follow fixture order; empty queries return an empty list.
    """

    spec = ResearchToolSpec(
        name="mock",
        description="Offline deterministic source search and retrieval for tests.",
        capabilities=("search", "read"),
        source_types=("mock",),
    )

    def __init__(self, fixture_path: str | Path = _DEFAULT_FIXTURE):
        path = Path(fixture_path)
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FixtureLoadError(f"Cannot load fixture {path}: {exc}") from exc
        if not isinstance(records, list):
            raise FixtureLoadError(f"Fixture {path} must contain a JSON list of materials")

        self._materials: dict[str, ResearchMaterial] = {}
        self._search_text: dict[str, str] = {}
        for index, record in enumerate(records):
            location = f"Fixture {path}, record {index}"
            if not isinstance(record, dict):
                raise FixtureLoadError(f"{location} must be an object")
            if any(
                not isinstance(record.get(field), str) or not record[field].strip()
                for field in _TEXT_FIELDS
            ):
                raise FixtureLoadError(f"{location} requires non-blank string fields: {_TEXT_FIELDS}")
            tags = record.get("tags")
            if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
                raise FixtureLoadError(f"{location} requires tags to be a list of non-blank strings")
            source_ref = record["source_ref"]
            if source_ref in self._materials:
                raise FixtureLoadError(f"{location} has duplicate source_ref {source_ref!r}")
            self._materials[source_ref] = record
            self._search_text[source_ref] = " ".join(
                [record["title"], record["publisher"], record["source_type"], *tags, record["content"]]
            ).casefold()

    def _search(self, query: str) -> list[SearchResult]:
        if not isinstance(query, str):
            raise ResearchToolError("query must be a string")
        keywords = re.findall(r"\w+", query.casefold())
        if not keywords:
            return []
        return [
            {field: material[field] for field in _SUMMARY_FIELDS}
            for source_ref, material in self._materials.items()
            if all(keyword in self._search_text[source_ref] for keyword in keywords)
        ]

    def _read(self, source_ref: str) -> ResearchMaterial:
        if not isinstance(source_ref, str):
            raise ResearchToolError("source_ref must be a string")
        try:
            return deepcopy(self._materials[source_ref])
        except KeyError as exc:
            raise SourceNotFoundError(f"Source reference {source_ref!r} not found") from exc
