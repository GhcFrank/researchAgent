"""Explicit tool registration and metadata discovery; no routing or execution."""

from research_tools import ResearchTool, ResearchToolSpec


class ToolRegistryError(Exception):
    """Base error for tool catalog operations."""


class DuplicateToolError(ToolRegistryError):
    """A tool name is already registered and cannot be overwritten."""


class ToolNotFoundError(ToolRegistryError):
    """No tool is registered under the requested exact name."""


class ResearchToolRegistry:
    def __init__(self):
        self._tools: dict[str, ResearchTool] = {}

    def register(self, tool: ResearchTool) -> None:
        if not isinstance(tool, ResearchTool) or not isinstance(tool.spec, ResearchToolSpec):
            raise ToolRegistryError("register requires a ResearchTool with a ResearchToolSpec")
        name = tool.spec.name
        if name in self._tools:
            raise DuplicateToolError(f"Tool {name!r} is already registered")
        self._tools[name] = tool

    def get(self, name: str) -> ResearchTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool {name!r} is not registered") from exc

    def list_specs(self) -> list[ResearchToolSpec]:
        """Return metadata in registration order, without exposing the catalog list."""
        return [tool.spec for tool in self._tools.values()]
