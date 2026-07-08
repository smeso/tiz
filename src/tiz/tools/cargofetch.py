"""Cargo fetch tool to fetch Rust dependencies."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class CargoFetch(SocketTool):
    """Tool to run cargo fetch via Unix socket."""

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        path = args.get("path", "")
        if markdown:
            return f"Cargo fetch in: `{self._safe_md(path)}`"
        return f"Cargo fetch in: {path}"

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": (
                    "Run 'cargo fetch' to fetch dependencies for a Rust project. "
                    "Useful when regular bash commands don't have internet access "
                    "but the sandbox has a dedicated network path for cargo."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Rust project (directory containing Cargo.toml)",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Command timeout in seconds (1-600, default 300)",
                        },
                    },
                    "required": ["path"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "CargoFetch"

    def run(self, args: dict[str, Any]) -> str:
        if "path" not in args or not isinstance(args["path"], str) or not args["path"]:
            return f"ERROR: {self.fname()} takes a mandatory 'path' argument!"
        if len(args["path"]) > MAX_INPUT_SIZE:
            return "ERROR: path exceeds maximum allowed size"
        timeout = args.get("timeout", 300)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or timeout < 1
            or timeout > 600
        ):
            return "ERROR: timeout must be an integer between 1 and 600"
        call_args = dict(args)
        call_args.setdefault("timeout", 300)
        call_args.pop("name", None)
        call_args.pop("description", None)
        return self._call(call_args)
