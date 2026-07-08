"""Write file tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class WriteFile(SocketTool):
    """Tool to write file contents via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": "Write a text file's contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The file path"},
                        "contents": {
                            "type": "string",
                            "description": "The file contents",
                        },
                    },
                    "required": ["path", "contents"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "WriteFile"

    @staticmethod
    def _safe_md(s: str, max_len: int = 80) -> str:
        """Truncate and remove backticks for markdown safety."""
        return s[:max_len].replace("`", "")

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        path = str(args.get("path", "") or "")[:80]
        contents = str(args.get("contents", "") or "")
        preview = contents[:80]
        size = len(contents)
        if markdown:
            return f"Write file: `{self._safe_md(path)}` ({size} bytes): `{self._safe_md(preview)}`"
        return f"Write file: {path} ({size} bytes): {preview!r}"

    def run(self, args: dict[str, Any]) -> str:
        if "path" not in args or not isinstance(args["path"], str) or not args["path"]:
            return f"ERROR: {self.fname()} takes a mandatory 'path' argument!"
        if "contents" not in args:
            return f"ERROR: {self.fname()} takes a mandatory 'contents' argument!"
        if len(args["path"]) > MAX_INPUT_SIZE:
            return "ERROR: path exceeds maximum allowed size"
        if not isinstance(args["contents"], str):
            return "ERROR: contents must be a string"
        if len(args["contents"]) > MAX_INPUT_SIZE:
            return "ERROR: contents exceeds maximum allowed size"
        call_args = dict(args)
        call_args.pop("name", None)
        call_args.pop("description", None)
        return self._call(call_args)
