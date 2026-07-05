"""Tests for the Glob tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.glob_tool import Glob

_TEST_SOCKET = "/tmp/test.sock"


class TestGlobPrompt:
    def test_prompt(self) -> None:
        tool = Glob(_TEST_SOCKET)
        data = json.loads(tool.prompt())
        assert data == {
            "name": "Glob",
            "description": "Search for files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match",
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in (default '.')",
                    },
                },
                "required": ["pattern"],
            },
        }


class TestGlobFname:
    def test_fname(self) -> None:
        assert Glob.fname() == "Glob"


class TestGlobRunValidation:
    def test_missing_pattern(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        result = tool.run({})
        assert result == "ERROR: Glob takes a mandatory 'pattern' argument!"

    def test_empty_pattern(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        result = tool.run({"pattern": ""})
        assert result == "ERROR: Glob takes a mandatory 'pattern' argument!"

    def test_pattern_too_large(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        result = tool.run({"pattern": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: pattern exceeds maximum allowed size"

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        with patch.object(Glob, "_call", return_value="matches") as mock_call:
            result = tool.run({"pattern": "*.py"})
        assert result == "matches"
        mock_call.assert_called_once_with({"pattern": "*.py"})

    def test_calls_call_with_path(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        with patch.object(Glob, "_call", return_value="matches") as mock_call:
            result = tool.run({"pattern": "*.py", "path": "/some/dir"})
        assert result == "matches"
        mock_call.assert_called_once_with({"pattern": "*.py", "path": "/some/dir"})

    def test_non_string_path_returns_error(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        result = tool.run({"pattern": "*.py", "path": 42})
        assert result == "ERROR: path must be a non-empty string"

    def test_empty_path_returns_error(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        result = tool.run({"pattern": "*.py", "path": ""})
        assert result == "ERROR: path must be a non-empty string"

    def test_path_too_large_returns_error(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        result = tool.run({"pattern": "*.py", "path": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_non_string_pattern(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        result = tool.run({"pattern": 123})
        assert result == "ERROR: Glob takes a mandatory 'pattern' argument!"

    def test_calls_call_strips_description(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        with patch.object(Glob, "_call", return_value="matches") as mock_call:
            result = tool.run({"pattern": "*.py", "description": "find files"})
        assert result == "matches"
        mock_call.assert_called_once_with({"pattern": "*.py"})

    def test_calls_call_strips_name(self, socket_path: str) -> None:
        tool = Glob(socket_path)
        with patch.object(Glob, "_call", return_value="matches") as mock_call:
            result = tool.run({"pattern": "*.py", "name": "Glob"})
        assert result == "matches"
        mock_call.assert_called_once_with({"pattern": "*.py"})

    def test_calls_call_strips_both_name_and_description(
        self, socket_path: str
    ) -> None:
        tool = Glob(socket_path)
        with patch.object(Glob, "_call", return_value="matches") as mock_call:
            result = tool.run(
                {"pattern": "*.py", "name": "Glob", "description": "find files"}
            )
        assert result == "matches"
        mock_call.assert_called_once_with({"pattern": "*.py"})


class TestGlobFormatConfirmation:
    def test_format_confirmation(self) -> None:
        tool = Glob("/tmp/test.sock")
        result = tool.format_confirmation({"pattern": "*.py"}, markdown=False)
        assert result == "Glob search: '*.py' in ."

    def test_format_confirmation_markdown(self) -> None:
        tool = Glob("/tmp/test.sock")
        result = tool.format_confirmation(
            {"pattern": "**/*.rs", "path": "/src"}, markdown=True
        )
        assert result == "Glob search: `**/*.rs` in `/src`"

    def test_format_confirmation_empty_pattern(self) -> None:
        tool = Glob("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Glob search: '' in ."

    def test_format_confirmation_empty_markdown(self) -> None:
        tool = Glob("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=True)
        assert result == "Glob search: `` in `.`"

    def test_format_confirmation_safe_md_strips_backticks(self) -> None:
        tool = Glob("/tmp/test.sock")
        result = tool.format_confirmation(
            {"pattern": "`file`.py", "path": "`dir`"}, markdown=True
        )
        assert result == "Glob search: `file.py` in `dir`"
