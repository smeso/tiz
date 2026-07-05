"""Tests for the WriteFile tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.writefile import WriteFile

_TEST_SOCKET = "/tmp/test.sock"


class TestWriteFilePrompt:
    def test_prompt_returns_valid_json(self) -> None:
        tool = WriteFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "WriteFile"

    def test_prompt_has_required_fields(self) -> None:
        tool = WriteFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert set(data["parameters"]["required"]) == {"path", "contents"}

    def test_prompt_description(self) -> None:
        tool = WriteFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["description"] == "Write a text file's contents."

    def test_prompt_parameters_type(self) -> None:
        tool = WriteFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["parameters"]["type"] == "object"

    def test_prompt_path_type(self) -> None:
        tool = WriteFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["parameters"]["properties"]["path"]["type"] == "string"

    def test_prompt_path_description(self) -> None:
        tool = WriteFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert (
            data["parameters"]["properties"]["path"]["description"] == "The file path"
        )

    def test_prompt_contents_type(self) -> None:
        tool = WriteFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["parameters"]["properties"]["contents"]["type"] == "string"

    def test_prompt_contents_description(self) -> None:
        tool = WriteFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert (
            data["parameters"]["properties"]["contents"]["description"]
            == "The file contents"
        )


class TestWriteFileFname:
    def test_fname(self) -> None:
        assert WriteFile.fname() == "WriteFile"


class TestWriteFileRunValidation:
    def test_missing_path(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        result = tool.run({"contents": "data"})
        assert result == "ERROR: WriteFile takes a mandatory 'path' argument!"

    def test_empty_path(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        result = tool.run({"path": "", "contents": "data"})
        assert result == "ERROR: WriteFile takes a mandatory 'path' argument!"

    def test_path_not_string(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        result = tool.run({"path": 123, "contents": "data"})
        assert result == "ERROR: WriteFile takes a mandatory 'path' argument!"

    def test_path_none(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        result = tool.run({"path": None, "contents": "data"})
        assert result == "ERROR: WriteFile takes a mandatory 'path' argument!"

    def test_missing_contents(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        result = tool.run({"path": "test.txt"})
        assert result == "ERROR: WriteFile takes a mandatory 'contents' argument!"

    def test_contents_not_string(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        result = tool.run({"path": "test.txt", "contents": 42})
        assert result == "ERROR: contents must be a string"

    def test_contents_exceeds_max_size(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        oversized = "x" * (MAX_INPUT_SIZE + 1)
        result = tool.run({"path": "test.txt", "contents": oversized})
        assert result == "ERROR: contents exceeds maximum allowed size"

    def test_contents_exactly_max_size(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        exact = "x" * MAX_INPUT_SIZE
        with patch.object(WriteFile, "_call", return_value="wrote") as mock_call:
            result = tool.run({"path": "test.txt", "contents": exact})
        assert result == "wrote"
        mock_call.assert_called_once_with({"path": "test.txt", "contents": exact})

    def test_path_exceeds_max_size(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        oversized = "x" * (MAX_INPUT_SIZE + 1)
        result = tool.run({"path": oversized, "contents": "data"})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_path_exactly_max_size(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        exact = "x" * MAX_INPUT_SIZE
        with patch.object(WriteFile, "_call", return_value="wrote") as mock_call:
            result = tool.run({"path": exact, "contents": "data"})
        assert result == "wrote"
        mock_call.assert_called_once_with({"path": exact, "contents": "data"})

    def test_path_single_character(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        with patch.object(WriteFile, "_call", return_value="wrote") as mock_call:
            result = tool.run({"path": "a", "contents": "data"})
        assert result == "wrote"
        mock_call.assert_called_once_with({"path": "a", "contents": "data"})

    def test_contents_single_character(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        with patch.object(WriteFile, "_call", return_value="wrote") as mock_call:
            result = tool.run({"path": "test.txt", "contents": "a"})
        assert result == "wrote"
        mock_call.assert_called_once_with({"path": "test.txt", "contents": "a"})

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        with patch.object(WriteFile, "_call", return_value="wrote") as mock_call:
            result = tool.run({"path": "test.txt", "contents": "hello"})
        assert result == "wrote"
        mock_call.assert_called_once_with({"path": "test.txt", "contents": "hello"})

    def test_strips_name_and_description(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        with patch.object(WriteFile, "_call", return_value="wrote") as mock_call:
            result = tool.run(
                {
                    "path": "test.txt",
                    "contents": "hello",
                    "name": "EvilTool",
                    "description": "malicious",
                }
            )
        assert result == "wrote"
        mock_call.assert_called_once_with({"path": "test.txt", "contents": "hello"})

    def test_does_not_mutate_original_args(self, socket_path: str) -> None:
        tool = WriteFile(socket_path)
        args = {"path": "test.txt", "contents": "hello", "description": "test"}
        with patch.object(WriteFile, "_call", return_value="wrote") as mock_call:
            result = tool.run(args)
        assert result == "wrote"
        assert "description" in args
        mock_call.assert_called_once_with({"path": "test.txt", "contents": "hello"})


class TestWriteFileFormatConfirmation:
    def test_format_confirmation(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/home/file.txt", "contents": "hello world"}, markdown=False
        )
        assert result == "Write file: /home/file.txt (11 bytes): 'hello world'"

    def test_format_confirmation_markdown(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/tmp/test.txt", "contents": "hello"}, markdown=True
        )
        assert result == "Write file: `/tmp/test.txt` (5 bytes): `hello`"

    def test_format_confirmation_empty_path(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Write file:  (0 bytes): ''"

    def test_format_confirmation_empty_path_markdown(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=True)
        assert result == "Write file: `` (0 bytes): ``"

    def test_format_confirmation_long_contents(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        long = "a" * 200
        result = tool.format_confirmation(
            {"path": "/tmp/test.txt", "contents": long}, markdown=False
        )
        assert result == "Write file: /tmp/test.txt (200 bytes): '" + ("a" * 80) + "'"

    def test_format_confirmation_long_contents_markdown(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        long = "a" * 200
        result = tool.format_confirmation(
            {"path": "/tmp/test.txt", "contents": long}, markdown=True
        )
        assert result == "Write file: `/tmp/test.txt` (200 bytes): `" + ("a" * 80) + "`"

    def test_format_confirmation_long_path(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        long_path = "a" * 200
        result = tool.format_confirmation(
            {"path": long_path, "contents": "hello"}, markdown=False
        )
        assert result == f"Write file: {'a' * 80} (5 bytes): 'hello'"

    def test_format_confirmation_long_path_markdown(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        long_path = "a" * 200
        result = tool.format_confirmation(
            {"path": long_path, "contents": "hello"}, markdown=True
        )
        assert result == f"Write file: `{'a' * 80}` (5 bytes): `hello`"

    def test_format_confirmation_markdown_safe(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/tmp/`test`.txt", "contents": "hello `world`"}, markdown=True
        )
        assert result == "Write file: `/tmp/test.txt` (13 bytes): `hello world`"

    def test_format_confirmation_non_string_contents(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/tmp/test.txt", "contents": None}, markdown=False
        )
        assert result == "Write file: /tmp/test.txt (0 bytes): ''"

    def test_format_confirmation_non_string_contents_markdown(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/tmp/test.txt", "contents": None}, markdown=True
        )
        assert result == "Write file: `/tmp/test.txt` (0 bytes): ``"

    def test_format_confirmation_non_string_path(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": 123, "contents": "hello"}, markdown=False
        )
        assert result == "Write file: 123 (5 bytes): 'hello'"

    def test_format_confirmation_non_string_path_markdown(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": 123, "contents": "hello"}, markdown=True
        )
        assert result == "Write file: `123` (5 bytes): `hello`"

    def test_format_confirmation_path_none(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": None, "contents": "hello"}, markdown=False
        )
        assert result == "Write file:  (5 bytes): 'hello'"

    def test_format_confirmation_path_none_markdown(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": None, "contents": "hello"}, markdown=True
        )
        assert result == "Write file: `` (5 bytes): `hello`"

    def test_format_confirmation_empty_string_contents(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/tmp/test.txt", "contents": ""}, markdown=False
        )
        assert result == "Write file: /tmp/test.txt (0 bytes): ''"

    def test_format_confirmation_empty_string_contents_markdown(self) -> None:
        tool = WriteFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/tmp/test.txt", "contents": ""}, markdown=True
        )
        assert result == "Write file: `/tmp/test.txt` (0 bytes): ``"
