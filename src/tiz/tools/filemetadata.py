"""File metadata tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class FileMetadata(SocketTool):
    """Tool to get file metadata via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": "Get metadata for a file or directory (type, size, permissions, timestamps).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file or directory path",
                        }
                    },
                    "required": ["path"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "FileMetadata"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        path = args.get("path", "")
        if markdown:
            return f"Get metadata for: `{self._safe_md(path)}`"
        return f"Get metadata for: {path}"

    def run(self, args: dict[str, Any]) -> str:
        if "path" not in args or not isinstance(args["path"], str) or not args["path"]:
            return f"ERROR: {self.fname()} takes a mandatory 'path' argument!"
        if len(args["path"]) > MAX_INPUT_SIZE:
            return "ERROR: path exceeds maximum allowed size"
        call_args = dict(args)
        call_args.pop("description", None)
        call_args.pop("name", None)
        return self._call(call_args)
