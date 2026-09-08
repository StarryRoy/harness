"""MCP loading through the official LangChain adapter (never a second runtime)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import MCPError


async def load_mcp_tools(servers: Mapping[str, Mapping[str, Any]]) -> list[Any]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise MCPError(
            "MCP tools require the optional 'langchain-mcp-adapters' package", cause=exc
        ) from exc
    try:
        tools = list(await MultiServerMCPClient(dict(servers)).get_tools())
        for tool in tools:
            tool.metadata = {**(tool.metadata or {}), "harness_mcp": True}
        return tools
    except MCPError:
        raise
    except Exception as exc:
        raise MCPError("Unable to load MCP tools", cause=exc) from exc
