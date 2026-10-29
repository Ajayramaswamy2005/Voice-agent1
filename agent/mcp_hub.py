"""Connects to the MCP servers that hold the agent's tools and adapts them for the LLM."""

import json
import logging
import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from .config import ROOT

log = logging.getLogger("mcp")


def compact_schema(node):
    """Shrink pydantic-generated JSON schemas: every prompt token costs time-to-first-token on each turn.
    Drops auto titles and null defaults, and turns Optional's `anyOf: [X, null]` into plain X."""
    if isinstance(node, list):
        return [compact_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "title" or (k == "default" and v is None):
            continue
        if k == "anyOf" and isinstance(v, list):
            non_null = [x for x in v if x != {"type": "null"}]
            if len(non_null) == 1:
                out.update(compact_schema(non_null[0]))
                continue
        out[k] = compact_schema(v)
    return out


class MCPHub:
    def __init__(self):
        self._stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, list[dict]] = {}  # server -> OpenAI-style tool specs

    async def start(self, servers: dict[str, tuple[str, ...]], reseed: bool = True,
                    env: dict[str, dict[str, str]] | None = None) -> None:
        """servers: name -> stdio command. An empty command runs the built-in module mcp_servers.<name>.

        The MCP client starts servers with a minimal environment (no API keys). Built-in servers are our own
        code and get the full environment; external ones get only what their line lists in `env`."""
        for name, command in servers.items():
            extra = (env or {}).get(name, {})
            if command:
                params = StdioServerParameters(command=command[0], args=list(command[1:]), cwd=str(ROOT),
                                               env={**get_default_environment(), **extra} if extra else None)
            else:
                params = StdioServerParameters(
                    command=sys.executable,
                    args=["-m", f"mcp_servers.{name}"] + (["--reseed"] if reseed else []),
                    cwd=str(ROOT),
                    env={**os.environ, **extra},
                )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
            self.sessions[name] = session
            self.tools[name] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": compact_schema(t.input_schema or {"type": "object", "properties": {}}),
                    },
                }
                for t in listed.tools
            ]
            log.info("MCP server %s: %s", name, [t.name for t in listed.tools])

    async def call(self, server: str, name: str, arguments_json: str) -> str:
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": f"invalid JSON arguments: {arguments_json!r}"})
        try:
            result = await self.sessions[server].call_tool(name, args)
        except Exception as e:
            log.exception("tool %s failed", name)
            return json.dumps({"error": str(e)})
        if result.is_error:
            log.warning("tool %s returned an error", name)
        parts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
        text = "\n".join(parts) if parts else json.dumps(result.structured_content or {})
        return text[:4000]

    async def close(self) -> None:
        await self._stack.aclose()
