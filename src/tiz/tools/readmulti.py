"""Read multiple files tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool

MAX_PATHS = 50


class ReadMulti(SocketTool):
    """Tool to read multiple files at once via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": "Read multiple files at once.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "paths": {
                            "type": "array",
                            "description": f"List of file paths to read (max {MAX_PATHS})",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["paths"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "ReadMulti"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        paths = args.get("paths") or []
        count = len(paths)
        if markdown:
            return f"Read {count} file(s): {', '.join(f'`{self._safe_md(str(p))}`' for p in paths)}"
        return f"Read {count} file(s): {', '.join(str(p) for p in paths)}"

    def run(self, args: dict[str, Any]) -> str:
        if (
            "paths" not in args
            or not isinstance(args["paths"], list)
            or not args["paths"]
        ):
            return f"ERROR: {self.fname()} takes a mandatory 'paths' argument!"
        if not all(isinstance(p, str) and p for p in args["paths"]):
            return f"ERROR: {self.fname()} 'paths' must contain only non-empty strings!"
        if any(len(p) > MAX_INPUT_SIZE for p in args["paths"]):
            return f"ERROR: {self.fname()} path exceeds maximum allowed size"
        if len(args["paths"]) > MAX_PATHS:
            return f"ERROR: {self.fname()} allows a maximum of {MAX_PATHS} paths"
        call_args = dict(args)
        self._pop_name_desc(call_args)
        return self._call(call_args)
