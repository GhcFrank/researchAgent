"""Focused tests of local retrieval, fixture validation, and explicit errors."""

from copy import deepcopy
import json

import pytest

from research_tools import FixtureLoadError, MockResearchTool, SourceNotFoundError


@pytest.fixture
def materials():
    return [
        {
            "source_ref": "mock-one",
            "title": "[MOCK] Planet Labs orbit overview",
            "publisher": "Mock Observatory",
            "source_type": "Mock Technical Brief",
            "published_date": "2026-09-01",
            "locator": "mock://one",
            "content": "MOCK / FIXTURE DATA: Data + Analytics FY27 Q2 revenue is synthetic.",
            "tags": ["mock", "fixture", "monitoring"],
        },
        {
            "source_ref": "mock-two",
            "title": "[MOCK] Planet Labs unrelated launch",
            "publisher": "Mock Launch Desk",
            "source_type": "Mock Announcement",
            "published_date": "2026-09-02",
            "locator": "mock://two",
            "content": "MOCK / FIXTURE DATA: A fictional satellite launch.",
            "tags": ["mock", "fixture", "launch"],
        },
    ]


@pytest.fixture
def fixture_path(tmp_path, materials):
    path = tmp_path / "materials.json"
    path.write_text(json.dumps(materials), encoding="utf-8")
    return path


def test_default_planet_fixture_supports_search_and_read():
    tool = MockResearchTool()
    summaries = tool.search("Planet Labs")
    assert [item["source_ref"] for item in summaries] == [
        "mock-src-001", "mock-src-002", "mock-src-003", "mock-src-004"
    ]
    for summary in summaries:
        assert set(summary) == {"source_ref", "title", "publisher", "source_type", "published_date", "locator"}
        assert summary["title"].startswith("[MOCK]")
        material = tool.read(summary["source_ref"])
        assert "MOCK / FIXTURE DATA ONLY" in material["content"]
        assert "mock" in material["tags"]
        assert all(material[field] == value for field, value in summary.items())

    relevant = tool.search("Data + Analytics FY27 Q2")
    assert [item["source_ref"] for item in relevant] == ["mock-src-001", "mock-src-002", "mock-src-004"]
    content = tool.read(relevant[0]["source_ref"])["content"]
    assert "USD 60 million" in content and "24%" in content and "75%" in content


def test_custom_fixture_search_is_case_insensitive_and_deterministic(fixture_path, materials):
    tool = MockResearchTool(fixture_path=fixture_path)
    expected = [record["source_ref"] for record in materials]
    assert [item["source_ref"] for item in tool.search("PLANET LABS")] == expected
    assert tool.search("planet labs") == tool.search("PLANET LABS")
    assert [item["source_ref"] for item in tool.search("data ANALYTICS fy27 q2")] == ["mock-one"]


def test_search_checks_metadata_tags_and_content(fixture_path):
    tool = MockResearchTool(fixture_path)
    # Each unique keyword exercises one searchable field, without exposing body.
    for query in ("orbit", "Observatory", "Technical", "monitoring", "revenue"):
        results = tool.search(query)
        assert [item["source_ref"] for item in results] == ["mock-one"]
        assert "content" not in results[0] and "tags" not in results[0]


def test_unmatched_and_empty_queries_return_empty_list(fixture_path):
    tool = MockResearchTool(fixture_path)
    for query in ("no-such-material-xyz", "", " \t\n"):
        assert tool.search(query) == []


def test_read_returns_complete_independent_material(fixture_path, materials):
    tool = MockResearchTool(fixture_path)
    material = tool.read("mock-one")
    assert material == materials[0]
    material["content"] = "changed by caller"
    material["tags"].append("changed-tag")
    assert tool.read("mock-one") == materials[0]
    assert tool.search("changed-tag") == []


def test_unknown_source_ref_raises(fixture_path):
    with pytest.raises(SourceNotFoundError, match="absent-ref"):
        MockResearchTool(fixture_path).read("absent-ref")


def test_missing_fixture_raises(tmp_path):
    with pytest.raises(FixtureLoadError, match="missing.json"):
        MockResearchTool(tmp_path / "missing.json")


def test_corrupt_json_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("[{", encoding="utf-8")
    with pytest.raises(FixtureLoadError, match="broken.json"):
        MockResearchTool(path)


@pytest.mark.parametrize("failure", ["root", "record", "missing-field", "tags", "duplicate-ref"])
def test_invalid_fixture_structure_raises(tmp_path, materials, failure):
    invalid = deepcopy(materials)
    if failure == "root":
        invalid = {"materials": invalid}
    elif failure == "record":
        invalid.append("not an object")
    elif failure == "missing-field":
        del invalid[1]["content"]
    elif failure == "tags":
        invalid[1]["tags"] = "not a list"
    else:
        invalid[1]["source_ref"] = invalid[0]["source_ref"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(FixtureLoadError, match="invalid.json"):
        MockResearchTool(path)
