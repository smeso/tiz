"""Read file tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class ReadFile(SocketTool):
    """Tool to read file contents via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": "ReadFile",
                "description": "Read a text file's contents. Each line is prefixed with its line number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The file path"},
                        "view_range": {
                            "type": "array",
                            "description": "Optional parameter to specify first and last line numbers to read (inclusive, 1-indexed)",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "integer", "minimum": 1},
                        },
                    },
                    "required": ["path"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "ReadFile"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        path = args.get("path", "")
        if markdown:
            return f"Read file: `{self._safe_md(path)}`"
        return f"Read file: {path}"

    def run(self, args: dict[str, Any]) -> str:
        if "path" not in args or not isinstance(args["path"], str) or not args["path"]:
            return f"ERROR: {self.fname()} takes a mandatory 'path' argument!"
        if len(args["path"]) > MAX_INPUT_SIZE:
            return "ERROR: path exceeds maximum allowed size"
        view_range = args.get("view_range")
        if view_range is not None:
            if not isinstance(view_range, list) or len(view_range) != 2:
                return "ERROR: view_range must be a list of two integers"
            if not all(
                isinstance(x, int) and not isinstance(x, bool) for x in view_range
            ):
                return "ERROR: view_range must contain integers only"
            if view_range[0] < 1 or view_range[1] < 1:
                return "ERROR: view_range values must be positive"
            if view_range[0] > view_range[1]:
                return "ERROR: view_range[0] must be <= view_range[1]"
        call_args = dict(args)
        call_args.pop("name", None)
        call_args.pop("description", None)
        return self._call(call_args)
