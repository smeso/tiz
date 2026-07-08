"""Grep text search tool."""

import json
import re
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class Grep(SocketTool):
    """Tool to search for text patterns via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": "Search for text patterns in files using grep or ripgrep.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The search pattern",
                        },
                        "path": {
                            "type": "string",
                            "description": "The directory to search in (default '.')",
                        },
                        "glob": {"type": "string", "description": "File glob filter"},
                        "regex": {
                            "type": "boolean",
                            "description": "Use regex matching (default false for literal matching)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (1-10000, default 100)",
                            "minimum": 1,
                            "maximum": 10000,
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "Case insensitive search",
                        },
                    },
                    "required": ["pattern"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "Grep"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        pattern = args.get("pattern", "")
        search_path = args.get("path", ".")
        if markdown:
            return (
                f"Grep for `{self._safe_md(pattern)}` in `{self._safe_md(search_path)}`"
            )
        return f"Grep for '{pattern}' in {search_path}"

    def run(self, args: dict[str, Any]) -> str:
        if (
            "pattern" not in args
            or not isinstance(args["pattern"], str)
            or not args["pattern"]
        ):
            return f"ERROR: {self.fname()} takes a mandatory 'pattern' argument!"
        if len(args["pattern"]) > MAX_INPUT_SIZE:
            return "ERROR: pattern exceeds maximum allowed size"
        regex = args.get("regex", False)
        if not isinstance(regex, bool):
            return "ERROR: regex must be a boolean"
        if regex:
            try:
                re.compile(args["pattern"])
            except re.error as e:
                return f"ERROR: invalid regex pattern: {e}"
        case_insensitive = args.get("case_insensitive", False)
        if not isinstance(case_insensitive, bool):
            return "ERROR: case_insensitive must be a boolean"
        if "path" in args:
            if not isinstance(args["path"], str):
                return "ERROR: path must be a string"
            if len(args["path"]) > MAX_INPUT_SIZE:
                return "ERROR: path exceeds maximum allowed size"
        if "glob" in args:
            if not isinstance(args["glob"], str):
                return "ERROR: glob must be a string"
            if len(args["glob"]) > MAX_INPUT_SIZE:
                return "ERROR: glob exceeds maximum allowed size"
        call_args = dict(args)
        max_results = call_args.setdefault("max_results", 100)
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or max_results < 1
            or max_results > 10000
        ):
            return "ERROR: max_results must be an integer between 1 and 10000"
        self._pop_name_desc(call_args)
        return self._call(call_args)
