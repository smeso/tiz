"""Uv python install tool to install Python versions."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class UvPythonInstall(SocketTool):
    """Tool to run uv python install via Unix socket."""

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        version = args.get("version", "")
        if markdown:
            return f"Install Python: `{self._safe_md(version)}`"
        return f"Install Python: {version}"

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": (
                    "Run 'uv python install --no-bin -f' to install a Python version. "
                    "Useful when regular bash commands don't have internet access "
                    "but the sandbox has a dedicated network path for uv."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "version": {
                            "type": "string",
                            "description": "Python version to install (e.g. '3.12', '3.10.0')",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Command timeout in seconds (1-600, default 300)",
                        },
                    },
                    "required": ["version"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "UvPythonInstall"

    def run(self, args: dict[str, Any]) -> str:
        args = dict(args)
        if (
            "version" not in args
            or not isinstance(args["version"], str)
            or not args["version"]
        ):
            return f"ERROR: {self.fname()} takes a mandatory 'version' argument!"
        if len(args["version"]) > MAX_INPUT_SIZE:
            return "ERROR: version exceeds maximum allowed size"
        timeout = args.get("timeout", 300)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or timeout < 1
            or timeout > 600
        ):
            return "ERROR: timeout must be an integer between 1 and 600"
        args.pop("description", None)
        args.pop("name", None)
        return self._call(args)
