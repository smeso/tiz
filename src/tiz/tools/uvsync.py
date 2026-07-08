"""Uv sync tool to sync Python project dependencies."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


def _normalize_list(value: Any) -> list[str]:
    """Normalize a value into a list of strings.

    Raises ValueError if value is not a string or list of strings,
    or if the string/list is empty.
    """
    if isinstance(value, list):
        if not value:
            msg = "must be a non-empty string or list of non-empty strings"
            raise ValueError(msg)
        if not all(isinstance(v, str) and v.strip() for v in value):
            msg = "must be a non-empty string or list of non-empty strings"
            raise ValueError(msg)
        return list(value)
    if isinstance(value, str):
        if not value.strip():
            msg = "must be a non-empty string or list of non-empty strings"
            raise ValueError(msg)
        return [value]
    msg = "must be a non-empty string or list of non-empty strings"
    raise ValueError(msg)


class UvSync(SocketTool):
    """Tool to run uv sync via Unix socket."""

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        path = args.get("path") or ""
        if not path:
            return "Uv sync"
        if markdown:
            return f"Uv sync in: `{self._safe_md(path)}`"
        return f"Uv sync in: {path}"

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": (
                    "Run 'uv sync' to sync dependencies for a Python project. "
                    "Useful when regular bash commands don't have internet access "
                    "but the sandbox has a dedicated network path for uv."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Python project (directory containing pyproject.toml or requirements)",
                        },
                        "group": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                            "description": "Target a specific dependency group (passed as --group to uv sync). Can be a single string or array of strings.",
                        },
                        "extra": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                            "description": "Target a specific extra (passed as --extra to uv sync). Can be a single string or array of strings.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Command timeout in seconds (1-600, default 300)",
                        },
                    },
                    "required": ["path"],
                },
            },
        )

    @staticmethod
    def fname() -> str:
        return "UvSync"

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
        # Normalize group and extra
        call_args = dict(args)
        call_args.setdefault("timeout", 300)
        if "group" in call_args:
            try:
                call_args["group"] = _normalize_list(call_args["group"])
            except ValueError as e:
                return f"ERROR: group {e}"
        if "extra" in call_args:
            try:
                call_args["extra"] = _normalize_list(call_args["extra"])
            except ValueError as e:
                return f"ERROR: extra {e}"
        call_args.pop("description", None)
        call_args.pop("name", None)
        return self._call(call_args)
