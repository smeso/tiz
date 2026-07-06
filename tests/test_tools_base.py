"""Tests for tool base classes."""

import json
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tiz.tools.base import MAX_INPUT_SIZE, SocketTool, Tool


class ConcreteTool(Tool):
    """Concrete implementation of Tool for testing."""

    @staticmethod
    def prompt() -> str:
        return json.dumps({"name": "ConcreteTool"})

    @staticmethod
    def fname() -> str:
        return "ConcreteTool"

    def run(self, _args: dict[str, Any]) -> str:
        return "ran"

    def format_confirmation(
        self,
        args: dict[str, Any],
        markdown: bool = False,  # noqa: ARG002
    ) -> str | None:
        return f"Run with: {args}"


class ConcreteSocketTool(SocketTool):
    """Concrete implementation of SocketTool for testing."""

    @staticmethod
    def prompt() -> str:
        return json.dumps({"name": "ConcreteSocketTool"})

    @staticmethod
    def fname() -> str:
        return "ConcreteSocketTool"

    def run(self, args: dict[str, Any]) -> str:
        result = self._call(args)
        return str(result) if result is not None else ""

    def format_confirmation(
        self,
        args: dict[str, Any],
        markdown: bool = False,  # noqa: ARG002
    ) -> str | None:
        return f"Socket op with: {args}"


def test_max_input_size() -> None:
    assert MAX_INPUT_SIZE == 1024 * 1024


def test_tool_is_abstract() -> None:
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


def test_concrete_tool_prompt() -> None:
    assert ConcreteTool.prompt() == '{"name": "ConcreteTool"}'


def test_concrete_tool_fname() -> None:
    assert ConcreteTool.fname() == "ConcreteTool"


def test_concrete_tool_run() -> None:
    tool = ConcreteTool()
    assert tool.run({}) == "ran"


def test_socket_tool_init(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)
    assert tool.socket_path == socket_path


def test_safe_md_removes_backticks() -> None:
    assert Tool._safe_md("`hello`") == "hello"


def test_safe_md_no_backticks() -> None:
    assert Tool._safe_md("hello world") == "hello world"


def test_safe_md_empty_string() -> None:
    assert Tool._safe_md("") == ""


def test_safe_md_multiple_backticks() -> None:
    assert Tool._safe_md("`a` `b` `c`") == "a b c"


def test_socket_tool_call_success(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"result": "hello", "error": False})
    resp_data = result_json.encode("utf-8") + b"\n"
    send_data = {"key": "value"}

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run(send_data)

    assert result == "hello"
    assert mock_sock.sendall.called


def test_socket_tool_call_input_too_large(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)
    large_args = {"data": "x" * (MAX_INPUT_SIZE + 1)}
    result = tool.run(large_args)
    assert result == "ERROR: input exceeds maximum allowed size"


def test_socket_tool_call_bad_socket_permissions(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)

    with patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value.st_mode = 0o100644
        result = tool.run({})

    assert result == "ERROR: tool communication failed: invalid socket permissions"


def test_socket_tool_call_connection_error(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)

    with (
        patch("socket.socket") as mock_socket_cls,
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        mock_instance = MagicMock()
        mock_instance.connect.side_effect = ConnectionError("refused")
        mock_socket_cls.return_value = mock_instance
        result = tool.run({})

    assert result == "ERROR: tool communication failed"


def test_socket_tool_call_os_error(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)

    with (
        patch("socket.socket") as mock_socket_cls,
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        mock_instance = MagicMock()
        mock_instance.connect.side_effect = OSError("no such file")
        mock_socket_cls.return_value = mock_instance
        result = tool.run({})

    assert result == "ERROR: tool communication failed"


def test_socket_tool_call_timeout_error(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)

    with (
        patch("socket.socket") as mock_socket_cls,
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        mock_instance = MagicMock()
        mock_instance.connect.side_effect = TimeoutError("timed out")
        mock_socket_cls.return_value = mock_instance
        result = tool.run({})

    assert result == "ERROR: tool communication failed"


def test_socket_tool_call_json_decode_error(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)
    bad_data = b"not json\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(bad_data)),
        bad_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: tool communication failed"
    assert mock_sock.sendall.called


def test_socket_tool_call_unicode_decode_error(socket_path: str) -> None:
    """Non-UTF-8 response triggers UnicodeDecodeError."""
    tool = ConcreteSocketTool(socket_path)
    non_utf8_data = struct.pack(">I", 4) + b"\xff\xfe\xff\xfe"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        non_utf8_data[:4],
        non_utf8_data[4:],
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: tool communication failed"
    assert mock_sock.sendall.called


def test_socket_tool_call_partial_length_header(socket_path: str) -> None:
    """Partial recv calls correctly assemble the 4-byte length header."""
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"result": "hello", "error": False})
    resp_data = result_json.encode("utf-8") + b"\n"
    resp_length = len(resp_data)
    length_prefix = struct.pack(">I", resp_length)

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        length_prefix[:1],  # first recv(4) returns 1 byte
        length_prefix[1:3],  # second recv(3) returns 2 bytes
        length_prefix[3:],  # third recv(1) returns 1 byte -> header complete
        resp_data,  # full payload
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "hello"
    assert mock_sock.sendall.called


def test_socket_tool_call_closed_during_header(socket_path: str) -> None:
    """Connection closed during header read returns appropriate error."""
    tool = ConcreteSocketTool(socket_path)

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [b"", b""]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: tool communication failed: invalid response"
    assert mock_sock.sendall.called


def test_socket_tool_call_truncated_response(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)
    full_data = json.dumps({"result": "ok"}).encode("utf-8") + b"\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(full_data)),
        full_data[:5],
        b"",
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: tool communication failed: truncated response"
    assert mock_sock.sendall.called


def test_socket_tool_call_chunked_response(socket_path: str) -> None:
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"result": "chunked", "error": False})
    resp_data = result_json.encode("utf-8") + b"\n"
    mid = len(resp_data) // 2

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data[:mid],
        resp_data[mid:],
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "chunked"
    assert mock_sock.sendall.called


def test_socket_tool_call_server_error_with_result(socket_path: str) -> None:
    """Server response with error=True returns ERROR: prefixed message."""
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"result": "permission denied", "error": True})
    resp_data = result_json.encode("utf-8") + b"\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: permission denied"
    assert mock_sock.sendall.called


def test_socket_tool_call_server_error_no_result(socket_path: str) -> None:
    """Server response with error=True and no result field."""
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"error": True})
    resp_data = result_json.encode("utf-8") + b"\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: unknown error"
    assert mock_sock.sendall.called


def test_socket_tool_call_server_error_string_field(socket_path: str) -> None:
    """Server response with string error field returns that error message."""
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"error": "Request too large"})
    resp_data = result_json.encode("utf-8") + b"\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: Request too large"
    assert mock_sock.sendall.called


def test_socket_tool_call_server_error_invalid_json_string(socket_path: str) -> None:
    """Server response with error string 'Invalid JSON' returns that message."""
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"error": "Invalid JSON"})
    resp_data = result_json.encode("utf-8") + b"\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: Invalid JSON"
    assert mock_sock.sendall.called


def test_socket_tool_call_response_too_large(socket_path: str) -> None:
    """Server response with payload exceeding MAX_INPUT_SIZE returns error."""
    tool = ConcreteSocketTool(socket_path)
    huge_length = MAX_INPUT_SIZE + 1

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", huge_length),
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "ERROR: tool communication failed: response too large"
    assert mock_sock.sendall.called


def test_socket_tool_call_missing_error_flag(socket_path: str) -> None:
    """Server response without error field defaults to no error."""
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"result": "no error field"})
    resp_data = result_json.encode("utf-8") + b"\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == "no error field"
    assert mock_sock.sendall.called


def test_socket_tool_call_missing_result(socket_path: str) -> None:
    """Server response with no result key returns empty string."""
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"error": False})
    resp_data = result_json.encode("utf-8") + b"\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == ""
    assert mock_sock.sendall.called


def test_socket_tool_call_null_result(socket_path: str) -> None:
    """Server response with result: null returns empty string, not 'None'."""
    tool = ConcreteSocketTool(socket_path)
    result_json = json.dumps({"result": None, "error": False})
    resp_data = result_json.encode("utf-8") + b"\n"

    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [
        struct.pack(">I", len(resp_data)),
        resp_data,
    ]

    with (
        patch("socket.socket", return_value=mock_sock),
        patch.object(Path, "stat") as mock_stat,
    ):
        mock_stat.return_value.st_mode = 0o100600
        result = tool.run({})

    assert result == ""
    assert mock_sock.sendall.called


def test_socket_tool_init_forwards_kwargs(socket_path: str) -> None:
    """SocketTool.__init__ forwards **kwargs to super().__init__()."""
    tool = ConcreteSocketTool(socket_path, extra="value")
    assert tool.socket_path == socket_path


class TestSocketTimeout:
    """Tests for socket timeout calculation in _call."""

    def test_socket_timeout_default(self, socket_path: str) -> None:
        """No timeout in params: default timeout 300 -> socket timeout 600."""
        tool = ConcreteSocketTool(socket_path)
        result_json = json.dumps({"result": "ok", "error": False})
        resp_data = result_json.encode("utf-8") + b"\n"

        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [
            struct.pack(">I", len(resp_data)),
            resp_data,
        ]

        with (
            patch("socket.socket", return_value=mock_sock),
            patch.object(Path, "stat") as mock_stat,
        ):
            mock_stat.return_value.st_mode = 0o100600
            result = tool.run({})

        assert result == "ok"
        # Default timeout 300 -> socket_timeout = max(3, min(600, 300*2)) = 600
        assert mock_sock.settimeout.call_args_list[0][0][0] == 5  # connect timeout
        assert mock_sock.settimeout.call_args_list[1][0][0] == 600  # read timeout

    def test_socket_timeout_from_params(self, socket_path: str) -> None:
        """Timeout from params: 30 -> socket timeout 60."""
        tool = ConcreteSocketTool(socket_path)
        result_json = json.dumps({"result": "ok", "error": False})
        resp_data = result_json.encode("utf-8") + b"\n"

        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [
            struct.pack(">I", len(resp_data)),
            resp_data,
        ]

        with (
            patch("socket.socket", return_value=mock_sock),
            patch.object(Path, "stat") as mock_stat,
        ):
            mock_stat.return_value.st_mode = 0o100600
            result = tool.run({"timeout": 30})

        assert result == "ok"
        # timeout 30 -> socket_timeout = max(3, min(600, 30*2)) = 60
        assert mock_sock.settimeout.call_args_list[1][0][0] == 60

    def test_socket_timeout_minimum(self, socket_path: str) -> None:
        """Very small timeout clamped to minimum of 3."""
        tool = ConcreteSocketTool(socket_path)
        result_json = json.dumps({"result": "ok", "error": False})
        resp_data = result_json.encode("utf-8") + b"\n"

        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [
            struct.pack(">I", len(resp_data)),
            resp_data,
        ]

        with (
            patch("socket.socket", return_value=mock_sock),
            patch.object(Path, "stat") as mock_stat,
        ):
            mock_stat.return_value.st_mode = 0o100600
            result = tool.run({"timeout": 1})

        assert result == "ok"
        # timeout 1 -> socket_timeout = max(3, min(600, 1*2)) = max(3, 2) = 3
        assert mock_sock.settimeout.call_args_list[1][0][0] == 3

    def test_socket_timeout_maximum(self, socket_path: str) -> None:
        """Very large timeout clamped to maximum of 600."""
        tool = ConcreteSocketTool(socket_path)
        result_json = json.dumps({"result": "ok", "error": False})
        resp_data = result_json.encode("utf-8") + b"\n"

        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [
            struct.pack(">I", len(resp_data)),
            resp_data,
        ]

        with (
            patch("socket.socket", return_value=mock_sock),
            patch.object(Path, "stat") as mock_stat,
        ):
            mock_stat.return_value.st_mode = 0o100600
            result = tool.run({"timeout": 600})

        assert result == "ok"
        # timeout 600 -> socket_timeout = max(3, min(600, 600*2)) = max(3, 600) = 600
        assert mock_sock.settimeout.call_args_list[1][0][0] == 600

    def test_socket_timeout_at_minimum_edge(self, socket_path: str) -> None:
        """timeout=2 gives socket_timeout = max(3, min(600, 4)) = 4 (no clamping)."""
        tool = ConcreteSocketTool(socket_path)
        result_json = json.dumps({"result": "ok", "error": False})
        resp_data = result_json.encode("utf-8") + b"\n"

        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [
            struct.pack(">I", len(resp_data)),
            resp_data,
        ]

        with (
            patch("socket.socket", return_value=mock_sock),
            patch.object(Path, "stat") as mock_stat,
        ):
            mock_stat.return_value.st_mode = 0o100600
            result = tool.run({"timeout": 2})

        assert result == "ok"
        # timeout 2 -> socket_timeout = max(3, min(600, 2*2)) = max(3, 4) = 4
        assert mock_sock.settimeout.call_args_list[1][0][0] == 4

    def test_socket_timeout_at_threshold_below_min(self, socket_path: str) -> None:
        """timeout=1 gives socket_timeout = max(3, min(600, 2)) = 3 (clamped minimum)."""
        tool = ConcreteSocketTool(socket_path)
        result_json = json.dumps({"result": "ok", "error": False})
        resp_data = result_json.encode("utf-8") + b"\n"

        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [
            struct.pack(">I", len(resp_data)),
            resp_data,
        ]

        with (
            patch("socket.socket", return_value=mock_sock),
            patch.object(Path, "stat") as mock_stat,
        ):
            mock_stat.return_value.st_mode = 0o100600
            result = tool.run({"timeout": 1})

        assert result == "ok"
        assert mock_sock.settimeout.call_args_list[1][0][0] == 3


def test_concrete_tool_format_confirmation() -> None:
    """Test format_confirmation on a concrete Tool subclass."""
    tool = ConcreteTool()
    result = tool.format_confirmation({"key": "value"}, markdown=False)
    assert result == "Run with: {'key': 'value'}"

    result_md = tool.format_confirmation({"key": "value"}, markdown=True)
    assert result_md == "Run with: {'key': 'value'}"


def test_concrete_socket_tool_format_confirmation() -> None:
    """Test format_confirmation on a concrete SocketTool subclass."""
    tool = ConcreteSocketTool("/tmp/test.sock")
    result = tool.format_confirmation({"path": "/tmp/file.txt"}, markdown=False)
    assert result == "Socket op with: {'path': '/tmp/file.txt'}"

    result_md = tool.format_confirmation({"path": "/tmp/file.txt"}, markdown=True)
    assert result_md == "Socket op with: {'path': '/tmp/file.txt'}"
