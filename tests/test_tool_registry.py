"""Tool catalog tests; no routing, provider access, or persistent objects."""

import pytest

from research_tools import MockResearchTool, ResearchToolSpec
from tool_registry import DuplicateToolError, ResearchToolRegistry, ToolNotFoundError, ToolRegistryError


class SecondMockTool(MockResearchTool):
    spec = ResearchToolSpec(
        name="second_mock",
        description="Use for a second local mock source catalog.",
        capabilities=("search", "read"),
        source_types=("mock",),
    )


def test_register_get_and_specs_preserve_registration_order():
    registry = ResearchToolRegistry()
    assert registry.list_specs() == []
    first, second = MockResearchTool(), SecondMockTool()
    # Register in reverse name order to distinguish registration order from sorting.
    registry.register(second)
    registry.register(first)
    assert registry.get("mock") is first
    assert registry.get("second_mock") is second
    specs = registry.list_specs()
    assert specs == [second.spec, first.spec]
    assert all(isinstance(spec, ResearchToolSpec) for spec in specs)


def test_duplicate_name_rejected_without_replacing_tool():
    registry = ResearchToolRegistry()
    original = MockResearchTool()
    registry.register(original)
    with pytest.raises(DuplicateToolError, match="mock"):
        registry.register(MockResearchTool())
    assert registry.get("mock") is original
    assert registry.list_specs() == [original.spec]


def test_unknown_name_rejected_without_fuzzy_matching():
    registry = ResearchToolRegistry()
    registry.register(MockResearchTool())
    with pytest.raises(ToolNotFoundError, match="Mock"):
        registry.get("Mock")


def test_register_requires_a_research_tool():
    registry = ResearchToolRegistry()
    with pytest.raises(ToolRegistryError, match="ResearchTool"):
        registry.register(object())
    assert registry.list_specs() == []
