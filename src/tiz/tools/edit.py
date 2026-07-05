"""Edit file tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class Edit(SocketTool):
    """Tool to edit file contents via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": "Edit",
                "description": "Edit a file by replacing an old string with a new string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The file path"},
                        "old_string": {
                            "type": "string",
                            "description": "The string to replace",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The replacement string",
                        },
                        "expected_replacements": {
                            "type": "integer",
                            "description": "Expected number of replacements (-1 for all, default 1)",
                        },
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "Edit"

    @staticmethod
    def _safe_md(s: str, max_len: int = 80) -> str:
        """Truncate and remove backticks for markdown safety."""
        return s[:max_len].replace("`", "")

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        path = (args.get("path", "") or "")[:80]
        old_preview = (args.get("old_string", "") or "")[:80]
        new_preview = (args.get("new_string", "") or "")[:80]
        if markdown:
            return f"Edit file: `{self._safe_md(path)}` (replace `{self._safe_md(old_preview)}` with `{self._safe_md(new_preview)}`)"
        return f"Edit file: {path} (replace '{old_preview}' with '{new_preview}')"

    def run(self, args: dict[str, Any]) -> str:
        if "path" not in args or not isinstance(args["path"], str) or not args["path"]:
            return f"ERROR: {self.fname()} takes a mandatory 'path' argument!"
        if len(args["path"]) > MAX_INPUT_SIZE:
            return "ERROR: path exceeds maximum allowed size"
        if (
            "old_string" not in args
            or not isinstance(args["old_string"], str)
            or not args["old_string"]
        ):
            return f"ERROR: {self.fname()} takes a mandatory 'old_string' argument!"
        if "new_string" not in args or not isinstance(args["new_string"], str):
            return f"ERROR: {self.fname()} takes a mandatory 'new_string' argument!"
        if len(args["old_string"]) > MAX_INPUT_SIZE:
            return "ERROR: old_string exceeds maximum allowed size"
        if len(args["new_string"]) > MAX_INPUT_SIZE:
            return "ERROR: new_string exceeds maximum allowed size"
        if "expected_replacements" in args:
            er = args["expected_replacements"]
            if isinstance(er, bool) or not isinstance(er, int) or er < -1 or er == 0:
                return "ERROR: expected_replacements must be -1 or an integer greater than 0"
        call_args = dict(args)
        call_args.pop("name", None)
        call_args.pop("description", None)
        return self._call(call_args)
