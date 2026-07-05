"""Base class for tools."""

from __future__ import annotations

import json
import socket
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

MAX_INPUT_SIZE = 1024 * 1024  # 1MB


class Tool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: B027
        ...

    @staticmethod
    def _safe_md(s: str) -> str:
        """Remove backticks for markdown safety."""
        return s.replace("`", "")

    @abstractmethod
    def prompt(self) -> str:  # pragma: no cover
        """Return the tool's prompt definition as a JSON string."""
        ...

    @staticmethod
    @abstractmethod
    def fname() -> str:  # pragma: no cover
        """Return the tool's function name."""
        ...

    @abstractmethod
    def run(self, args: dict[str, Any]) -> str:  # pragma: no cover
        """Execute the tool with the given arguments."""
        ...

    @abstractmethod
    def format_confirmation(
        self, args: dict[str, Any], markdown: bool = False
    ) -> str | None:  # pragma: no cover
        """Return a human-readable confirmation string for the given args.

        If markdown is True, the string may use markdown formatting.
        Returns None if no nice formatting is provided.
        """
        ...


class SocketTool(Tool):
    """Base class for tools that communicate via Unix socket."""

    def __init__(self, socket_path: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.socket_path = socket_path

    def _call(self, params: dict[str, Any]) -> str:
        """Send a tool call via Unix socket and return the result."""
        message = json.dumps({**params, "name": self.fname()})
        encoded = (message + "\n").encode("utf-8")
        if len(encoded) > MAX_INPUT_SIZE:
            return "ERROR: input exceeds maximum allowed size"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock_stat = Path(self.socket_path).stat()
            if sock_stat.st_mode & 0o777 != 0o600:
                return "ERROR: tool communication failed: invalid socket permissions"
            sock.settimeout(5)
            sock.connect(self.socket_path)
            sock.settimeout(600)
            length_prefix = struct.pack(">I", len(encoded))
            sock.sendall(length_prefix + encoded)
            length_data = b""
            while len(length_data) < 4:
                chunk = sock.recv(4 - len(length_data))
                if not chunk:
                    return "ERROR: tool communication failed: invalid response"
                length_data += chunk
            payload_length = struct.unpack(">I", length_data)[0]
            if payload_length > MAX_INPUT_SIZE:
                return "ERROR: tool communication failed: response too large"
            data = b""
            while len(data) < payload_length:
                remaining = payload_length - len(data)
                chunk = sock.recv(min(65536, remaining))
                if not chunk:
                    break
                data += chunk
            if len(data) != payload_length:
                return "ERROR: tool communication failed: truncated response"
            response = json.loads(data.decode("utf-8"))
            err = response.get("error", False)
            if err:
                if isinstance(err, str):
                    return f"ERROR: {err}"
                return f"ERROR: {response.get('result', 'unknown error')}"
            result = response.get("result")
            return str(result) if result is not None else ""
        except (
            ConnectionError,
            OSError,
            TimeoutError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return "ERROR: tool communication failed"
        finally:
            sock.close()
