"""Example non-socket tool: returns current date and time."""

import json
from datetime import datetime, timezone
from typing import Any

from tiz.tools.base import Tool


class DateTimeTool(Tool):
    """A non-socket tool that returns the current date and time in UTC."""

    @staticmethod
    def prompt() -> str:
        return json.dumps(
            {
                "name": "DateTimeTool",
                "description": "Get the current date and time in UTC.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "Output format: 'iso' (default), 'unix', or 'human'",
                        },
                    },
                    "required": [],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "DateTimeTool"

    def run(self, args: dict[str, Any]) -> str:
        fmt = args.get("format", "iso")
        now = datetime.now(timezone.utc)
        if fmt == "unix":
            return str(int(now.timestamp()))
        if fmt == "human":
            return now.strftime("%Y-%m-%d %H:%M:%S UTC")
        return now.isoformat()
