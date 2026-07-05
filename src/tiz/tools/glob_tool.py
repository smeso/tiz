"""Glob file search tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class Glob(SocketTool):
    """Tool to search for files matching a pattern via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": "Glob",
                "description": "Search for files matching a glob pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The glob pattern to match",
                        },
                        "path": {
                            "type": "string",
                            "description": "The directory to search in (default '.')",
                        },
                    },
                    "required": ["pattern"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "Glob"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        pattern = args.get("pattern", "")
        search_path = args.get("path", ".")
        if markdown:
            return f"Glob search: `{self._safe_md(pattern)}` in `{self._safe_md(search_path)}`"
        return f"Glob search: '{pattern}' in {search_path}"

    def run(self, args: dict[str, Any]) -> str:
        if (
            "pattern" not in args
            or not isinstance(args["pattern"], str)
            or not args["pattern"]
        ):
            return f"ERROR: {self.fname()} takes a mandatory 'pattern' argument!"
        if len(args["pattern"]) > MAX_INPUT_SIZE:
            return "ERROR: pattern exceeds maximum allowed size"
        if "path" in args and (not isinstance(args["path"], str) or not args["path"]):
            return "ERROR: path must be a non-empty string"
        if "path" in args and len(args["path"]) > MAX_INPUT_SIZE:
            return "ERROR: path exceeds maximum allowed size"
        call_args = dict(args)
        call_args.pop("name", None)
        call_args.pop("description", None)
        return self._call(call_args)
