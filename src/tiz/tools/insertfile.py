"""Insert file content tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class InsertFile(SocketTool):
    """Tool to insert content into a file via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": "InsertFile",
                "description": "Insert content into a file at a specific line number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The file path"},
                        "content": {
                            "type": "string",
                            "description": "The content to insert",
                        },
                        "line_number": {
                            "type": "integer",
                            "description": "Line number to insert at (1-indexed, -1 for end of file)",
                        },
                    },
                    "required": ["path", "content"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "InsertFile"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        path = args.get("path", "")
        ln = args.get("line_number", -1)
        if markdown:
            suffix = (
                " (end of file)" if ln == -1 else f" at line `{self._safe_md(str(ln))}`"
            )
            return f"Insert into: `{self._safe_md(path)}`{suffix}"
        suffix = " (end of file)" if ln == -1 else f" at line {ln}"
        return f"Insert into: {path}{suffix}"

    def run(self, args: dict[str, Any]) -> str:
        if "path" not in args or not isinstance(args["path"], str) or not args["path"]:
            return f"ERROR: {self.fname()} takes a mandatory 'path' argument!"
        if "content" not in args:
            return f"ERROR: {self.fname()} takes a mandatory 'content' argument!"
        if len(args["path"]) > MAX_INPUT_SIZE:
            return "ERROR: path exceeds maximum allowed size"
        if not isinstance(args["content"], str):
            return "ERROR: content must be a string"
        if len(args["content"]) > MAX_INPUT_SIZE:
            return "ERROR: content exceeds maximum allowed size"
        if "line_number" in args:
            ln = args["line_number"]
            if isinstance(ln, bool) or not isinstance(ln, int) or (ln != -1 and ln < 1):
                return (
                    "ERROR: line_number must be -1 (end of file) or a positive integer"
                )
        call_args = dict(args)
        call_args.pop("name", None)
        call_args.pop("description", None)
        return self._call(call_args)
