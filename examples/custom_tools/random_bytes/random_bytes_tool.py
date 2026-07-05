"""Example socket tool: reads /dev/urandom and returns base64-encoded bytes."""

import json
from typing import Any

from tiz.tools.base import SocketTool


class RandomBytesTool(SocketTool):
    """A socket tool that reads /dev/urandom inside the sandbox and returns base64."""

    def __init__(self, socket_path: str) -> None:
        super().__init__(socket_path)

    @staticmethod
    def prompt() -> str:
        return json.dumps(
            {
                "name": "RandomBytesTool",
                "description": "Read random bytes from /dev/urandom and return them base64-encoded.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "Number of random bytes to read (1-8192, default 32)",
                        },
                    },
                    "required": [],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "RandomBytesTool"

    def run(self, args: dict[str, Any]) -> str:
        count = args.get("count", 32)
        if not isinstance(count, int) or count < 1 or count > 8192:
            return "ERROR: count must be an integer between 1 and 8192"
        return self._call(args)
