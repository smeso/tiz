"""Tests for the InsertFile tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.insertfile import InsertFile

_TEST_SOCKET = "/tmp/test.sock"


class TestInsertFilePrompt:
    def test_prompt_name(self) -> None:
        tool = InsertFile(_TEST_SOCKET)
        prompt = json.loads(tool.prompt())
        assert prompt["name"] == "InsertFile"

    def test_prompt_description(self) -> None:
        tool = InsertFile(_TEST_SOCKET)
        prompt = json.loads(tool.prompt())
        assert (
            prompt["description"]
            == "Insert content into a file at a specific line number."
        )

    def test_prompt_parameters_type(self) -> None:
        tool = InsertFile(_TEST_SOCKET)
        prompt = json.loads(tool.prompt())
        assert prompt["parameters"]["type"] == "object"

    def test_prompt_path_properties(self) -> None:
        tool = InsertFile(_TEST_SOCKET)
        prompt = json.loads(tool.prompt())
        path_props = prompt["parameters"]["properties"]["path"]
        assert path_props["type"] == "string"
        assert path_props["description"] == "The file path"

    def test_prompt_content_properties(self) -> None:
        tool = InsertFile(_TEST_SOCKET)
        prompt = json.loads(tool.prompt())
        content_props = prompt["parameters"]["properties"]["content"]
        assert content_props["type"] == "string"
        assert content_props["description"] == "The content to insert"

    def test_prompt_line_number_properties(self) -> None:
        tool = InsertFile(_TEST_SOCKET)
        prompt = json.loads(tool.prompt())
        ln_props = prompt["parameters"]["properties"].get("line_number")
        assert ln_props is not None, "line_number should exist as an optional property"
        assert ln_props["type"] == "integer"
        assert (
            ln_props["description"]
            == "Line number to insert at (1-indexed, -1 for end of file)"
        )

    def test_prompt_has_required_fields(self) -> None:
        tool = InsertFile(_TEST_SOCKET)
        prompt = json.loads(tool.prompt())
        assert set(prompt["parameters"]["required"]) == {"path", "content"}


class TestInsertFileFname:
    def test_fname(self) -> None:
        assert InsertFile.fname() == "InsertFile"


class TestInsertFileRunValidation:
    def test_missing_path(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"content": "data"})
        assert result == "ERROR: InsertFile takes a mandatory 'path' argument!"

    def test_empty_path(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "", "content": "data"})
        assert result == "ERROR: InsertFile takes a mandatory 'path' argument!"

    def test_non_string_path_list(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": ["a"], "content": "data"})
        assert result == "ERROR: InsertFile takes a mandatory 'path' argument!"

    def test_non_string_path_int(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": 42, "content": "data"})
        assert result == "ERROR: InsertFile takes a mandatory 'path' argument!"

    def test_non_string_content(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": 123})
        assert result == "ERROR: content must be a string"

    def test_list_content(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": [1, 2, 3]})
        assert result == "ERROR: content must be a string"

    def test_missing_content(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt"})
        assert result == "ERROR: InsertFile takes a mandatory 'content' argument!"

    def test_path_too_large(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "x" * (MAX_INPUT_SIZE + 1), "content": "data"})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_content_too_large(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: content exceeds maximum allowed size"

    def test_invalid_line_number_type(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": "data", "line_number": "1"})
        assert (
            result
            == "ERROR: line_number must be -1 (end of file) or a positive integer"
        )

    def test_invalid_line_number_zero(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": "data", "line_number": 0})
        assert (
            result
            == "ERROR: line_number must be -1 (end of file) or a positive integer"
        )

    def test_invalid_line_number_negative(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": "data", "line_number": -2})
        assert (
            result
            == "ERROR: line_number must be -1 (end of file) or a positive integer"
        )

    def test_invalid_line_number_boolean_true(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": "data", "line_number": True})
        assert (
            result
            == "ERROR: line_number must be -1 (end of file) or a positive integer"
        )

    def test_invalid_line_number_boolean_false(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": "data", "line_number": False})
        assert (
            result
            == "ERROR: line_number must be -1 (end of file) or a positive integer"
        )

    def test_invalid_line_number_float(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": "data", "line_number": 5.0})
        assert (
            result
            == "ERROR: line_number must be -1 (end of file) or a positive integer"
        )

    def test_invalid_line_number_none(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        result = tool.run({"path": "test.txt", "content": "data", "line_number": None})
        assert (
            result
            == "ERROR: line_number must be -1 (end of file) or a positive integer"
        )

    def test_valid_line_number_negative_one(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        with patch.object(InsertFile, "_call", return_value="inserted") as mock_call:
            result = tool.run(
                {"path": "test.txt", "content": "data", "line_number": -1}
            )
        assert result == "inserted"
        mock_call.assert_called_once_with(
            {"path": "test.txt", "content": "data", "line_number": -1}
        )

    def test_valid_line_number_positive(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        with patch.object(InsertFile, "_call", return_value="inserted") as mock_call:
            result = tool.run({"path": "test.txt", "content": "data", "line_number": 5})
        assert result == "inserted"
        mock_call.assert_called_once_with(
            {"path": "test.txt", "content": "data", "line_number": 5}
        )

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        with patch.object(InsertFile, "_call", return_value="inserted") as mock_call:
            result = tool.run({"path": "test.txt", "content": "new line"})
        assert result == "inserted"
        mock_call.assert_called_once_with({"path": "test.txt", "content": "new line"})

    def test_strips_description_and_name(self, socket_path: str) -> None:
        tool = InsertFile(socket_path)
        with patch.object(InsertFile, "_call", return_value="inserted") as mock_call:
            result = tool.run(
                {
                    "path": "test.txt",
                    "content": "data",
                    "description": "x",
                    "name": "InsertFile",
                }
            )
        assert result == "inserted"
        mock_call.assert_called_once_with({"path": "test.txt", "content": "data"})


class TestInsertFileFormatConfirmation:
    def test_format_confirmation(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/home/file.txt"}, markdown=False)
        assert result == "Insert into: /home/file.txt (end of file)"

    def test_format_confirmation_markdown(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/tmp/test.txt"}, markdown=True)
        assert result == "Insert into: `/tmp/test.txt` (end of file)"

    def test_format_confirmation_empty_path(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Insert into:  (end of file)"

    def test_format_confirmation_empty_path_markdown(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=True)
        assert result == "Insert into: `` (end of file)"

    def test_format_confirmation_markdown_safe(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/tmp/`test`.txt", "content": "hello"}, markdown=True
        )
        assert result == "Insert into: `/tmp/test.txt` (end of file)"

    def test_format_confirmation_markdown_safe_with_line_number(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/tmp/`test`.txt", "content": "hello", "line_number": 3},
            markdown=True,
        )
        assert result == "Insert into: `/tmp/test.txt` at line `3`"

    def test_format_confirmation_with_line_number(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/home/file.txt", "line_number": 5}, markdown=False
        )
        assert result == "Insert into: /home/file.txt at line 5"

    def test_format_confirmation_with_line_number_markdown(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/home/file.txt", "line_number": 5}, markdown=True
        )
        assert result == "Insert into: `/home/file.txt` at line `5`"

    def test_format_confirmation_with_line_number_minus_one(self) -> None:
        tool = InsertFile("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/home/file.txt", "line_number": -1}, markdown=False
        )
        assert result == "Insert into: /home/file.txt (end of file)"
