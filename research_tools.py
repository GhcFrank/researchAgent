"""Local retrieval of raw mock material, without Research Object creation."""

from abc import ABC, abstractmethod
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import TypedDict


class SearchResult(TypedDict):
    source_ref: str
    title: str
    publisher: str
    source_type: str
    published_date: str
    locator: str


class ResearchMaterial(SearchResult):
    content: str
    tags: list[str]


class ResearchToolError(Exception):
    """Base error for retrieval operations."""


class FixtureLoadError(ResearchToolError):
    """The fixture cannot be read or contains invalid material records."""


class SourceNotFoundError(ResearchToolError):
    """The requested source_ref is absent from the fixture."""


class ResearchTool(ABC):
    @abstractmethod
    def search(self, query: str) -> list[SearchResult]:
        """Return lightweight metadata for matching raw materials."""

    @abstractmethod
    def read(self, source_ref: str) -> ResearchMaterial:
        """Return a complete raw material or raise SourceNotFoundError."""


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

    def search(self, query: str) -> list[SearchResult]:
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

    def read(self, source_ref: str) -> ResearchMaterial:
        if not isinstance(source_ref, str):
            raise ResearchToolError("source_ref must be a string")
        try:
            return deepcopy(self._materials[source_ref])
        except KeyError as exc:
            raise SourceNotFoundError(f"Source reference {source_ref!r} not found") from exc
