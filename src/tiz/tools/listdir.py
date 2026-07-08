"""List directory tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class ListDir(SocketTool):
    """Tool to list directory contents via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": "List directory contents with metadata.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path (required)",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "List recursively (default false)",
                        },
                        "show_hidden": {
                            "type": "boolean",
                            "description": "Show hidden files (default false)",
                        },
                    },
                    "required": ["path"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "ListDir"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        path = args.get("path", "")
        recursive = args.get("recursive", False)
        show_hidden = args.get("show_hidden", False)
        details = []
        if recursive:
            details.append("recursive")
        if show_hidden:
            details.append("hidden")
        suffix = f" ({', '.join(details)})" if details else ""
        if markdown:
            return f"List dir: `{self._safe_md(path)}`{suffix}"
        return f"List dir: {path}{suffix}"

    def run(self, args: dict[str, Any]) -> str:
        if "path" not in args or not isinstance(args["path"], str) or not args["path"]:
            return f"ERROR: {self.fname()} takes a mandatory 'path' argument!"
        if len(args["path"]) > MAX_INPUT_SIZE:
            return "ERROR: path exceeds maximum allowed size"
        recursive = args.get("recursive", False)
        if not isinstance(recursive, bool):
            return "ERROR: recursive must be a boolean"
        show_hidden = args.get("show_hidden", False)
        if not isinstance(show_hidden, bool):
            return "ERROR: show_hidden must be a boolean"
        call_args = dict(args)
        self._pop_name_desc(call_args)
        return self._call(call_args)
