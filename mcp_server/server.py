from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

import anyio
import structlog
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, TextContent, Tool

from config import settings
from errors import ToolError
from mcp_server.db import Database
from mcp_server.tools.evaluate_chances import EvaluateChancesInput, evaluate_chances
from mcp_server.tools.get_action_items import GetActionItemsInput, get_action_items


TOOL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "student_gpa": {"type": "number", "minimum": 0, "maximum": 4.0},
        "sat_score": {"type": ["integer", "null"]},
        "gre_verbal": {"type": ["integer", "null"]},
        "gre_quant": {"type": ["integer", "null"]},
        "ap_classes_completed": {"type": "integer", "minimum": 0},
        "lor_count": {"type": "integer", "minimum": 0},
        "target_university_id": {"type": "string"},
    },
    "required": [
        "student_gpa",
        "ap_classes_completed",
        "lor_count",
        "target_university_id",
    ],
}


class UniversityAdmissionMcp:
    def __init__(self) -> None:
        self.name = "university-admission-mcp"
        self.version = "1.0.0"
        self.server = Server(self.name)
        self.db = Database(settings.sqlite_path)
        self._log = structlog.get_logger(__name__)
        self._register_handlers()

    def _register_handlers(self) -> None:
        list_tools_decorator = cast(
            Callable[[Callable[[], Awaitable[list[Tool]]]], Callable[[], Awaitable[list[Tool]]]],
            self.server.list_tools(),  # type: ignore[no-untyped-call]
        )

        @list_tools_decorator
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="evaluate_chances",
                    description="Compute a student's admission probability for a target university.",
                    inputSchema=TOOL_SCHEMA,
                ),
                Tool(
                    name="get_action_items",
                    description="Return specific gaps and upcoming deadlines for a student.",
                    inputSchema=TOOL_SCHEMA,
                ),
            ]

        call_tool_decorator = cast(
            Callable[
                [Callable[[str, dict[str, Any]], Awaitable[Sequence[TextContent]]]],
                Callable[[str, dict[str, Any]], Awaitable[Sequence[TextContent]]],
            ],
            self.server.call_tool(),  # type: ignore[no-untyped-call]
        )

        @call_tool_decorator
        async def call_tool(
            tool_name: str,
            arguments: dict[str, Any],
        ) -> Sequence[TextContent]:
            return await self._dispatch_tool(tool_name, arguments)

    async def _dispatch_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Sequence[TextContent]:
        log = self._log.bind(tool_name=tool_name)
        started_at = time.monotonic()
        try:
            match tool_name:
                case "evaluate_chances":
                    params = EvaluateChancesInput.model_validate(arguments)
                    chances_result = await evaluate_chances(self.db, params)
                    return [TextContent(type="text", text=chances_result.model_dump_json())]
                case "get_action_items":
                    params = GetActionItemsInput.model_validate(arguments)
                    action_result = await get_action_items(self.db, params)
                    return [TextContent(type="text", text=action_result.model_dump_json())]
                case _:
                    raise McpError(
                        ErrorData(code=-32601, message=f"Unknown tool: {tool_name}")
                    )
        except ToolError as exc:
            log.error("tool_error", code=exc.code, msg=exc.message)
            raise McpError(ErrorData(code=-32000, message=exc.message)) from exc
        except McpError:
            raise
        except Exception as exc:
            log.error("tool_unexpected", error=str(exc))
            raise McpError(ErrorData(code=-32000, message="Internal server error")) from exc
        finally:
            log.info(
                "tool_call",
                latency_ms=round((time.monotonic() - started_at) * 1000),
            )

    async def _run_async(self) -> None:
        log = self._log.bind(tool_name="server")
        await self.db.connect()
        try:
            log.info("server_start")
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=self.name,
                        server_version=self.version,
                        capabilities=self.server.get_capabilities(
                            NotificationOptions(),
                            {},
                        ),
                    ),
                )
        finally:
            await self.db.close()
            log.info("server_stop")

    def run(self) -> None:
        anyio.run(self._run_async)


mcp = UniversityAdmissionMcp()


if __name__ == "__main__":
    mcp.run()