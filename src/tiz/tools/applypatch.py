"""Apply patch tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool

MAX_STRIP = 20


class ApplyPatch(SocketTool):
    """Tool to apply a patch via Unix socket."""

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": "Apply a patch file using the patch command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string", "description": "The patch content"},
                        "strip": {
                            "type": "integer",
                            "description": "Number of leading path components to strip (default 0)",
                        },
                        "reverse": {
                            "type": "boolean",
                            "description": "Reverse the patch (default false)",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory to apply the patch in",
                        },
                    },
                    "required": ["patch"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "ApplyPatch"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        cwd = args.get("cwd") or "."
        reverse = args.get("reverse", False)
        action = "Reverse patch" if reverse else "Apply patch"
        if markdown:
            return f"{action} in: `{self._safe_md(cwd)}`"
        return f"{action} in: {cwd}"

    def run(self, args: dict[str, Any]) -> str:
        if (
            "patch" not in args
            or not isinstance(args["patch"], str)
            or not args["patch"]
        ):
            return f"ERROR: {self.fname()} takes a mandatory 'patch' argument!"
        patch_content = args["patch"]
        if len(patch_content) > MAX_INPUT_SIZE:
            return "ERROR: patch content exceeds maximum allowed size"
        strip = args.get("strip", 0)
        if (
            isinstance(strip, bool)
            or not isinstance(strip, int)
            or strip < 0
            or strip > MAX_STRIP
        ):
            return "ERROR: strip must be an integer between 0 and 20"
        reverse = args.get("reverse", False)
        if not isinstance(reverse, bool):
            return "ERROR: reverse must be a boolean"
        cwd = args.get("cwd")
        if cwd is not None and (not isinstance(cwd, str) or not cwd):
            return "ERROR: cwd must be a string"
        call_args: dict[str, Any] = {"patch": patch_content}
        if strip != 0:
            call_args["strip"] = strip
        if reverse:
            call_args["reverse"] = reverse
        if cwd is not None:
            call_args["cwd"] = cwd
        return self._call(call_args)
