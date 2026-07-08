"""Bash command execution tool."""

import json
from typing import Any

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool


class Bash(SocketTool):
    """Tool to run bash commands via Unix socket."""

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        cmd_full = args.get("command", "")
        cmd = cmd_full[:120] + ("..." if len(cmd_full) > 120 else "")
        cwd = args.get("cwd")
        timeout = args.get("timeout")
        env = args.get("env")
        if markdown:
            result = f"Run: `{self._safe_md(cmd)}`"
            if cwd:
                result += f" in `{self._safe_md(cwd)}`"
            if timeout is not None:
                result += f" (timeout=`{self._safe_md(str(timeout))}s`)"
            if env:
                env_str = ", ".join(
                    f"`{self._safe_md(k)}`=`{self._safe_md(v)}`" for k, v in env.items()
                )
                result += f" env: {env_str}"
        else:
            result = f"Run: {cmd}"
            if cwd:
                result += f" in {cwd}"
            if timeout is not None:
                result += f" (timeout={timeout}s)"
            if env:
                env_str = ", ".join(
                    f'{k}="{v}"' if (" " in v or "=" in v) else f"{k}={v}"
                    for k, v in env.items()
                )
                result += f" env: {env_str}"
        return result

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": "Run a command in a bash shell and then exit the shell.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to run",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Command timeout in seconds (1-300, default 30)",
                            "minimum": 1,
                            "maximum": 300,
                            "default": 30,
                        },
                        "cwd": {"type": "string", "description": "Working directory"},
                        "env": {
                            "type": "object",
                            "description": "Environment variable overrides",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["command"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "Bash"

    def run(self, args: dict[str, Any]) -> str:
        if (
            "command" not in args
            or not isinstance(args["command"], str)
            or not args["command"]
        ):
            return f"ERROR: {self.fname()} takes a mandatory 'command' argument!"
        if len(args["command"]) > MAX_INPUT_SIZE:
            return "ERROR: command exceeds maximum allowed size"
        timeout = args.get("timeout", 30)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or timeout < 1
            or timeout > 300
        ):
            return "ERROR: timeout must be an integer between 1 and 300"
        cwd = args.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            return "ERROR: cwd must be a string"
        env = args.get("env")
        if env is not None and not isinstance(env, dict):
            return "ERROR: env must be a dict"
        if env is not None and not all(
            isinstance(k, str) and k and isinstance(v, str) for k, v in env.items()
        ):
            return "ERROR: env keys and values must be strings"
        call_args = dict(args)
        call_args.pop("description", None)
        call_args.pop("name", None)
        return self._call(call_args)
