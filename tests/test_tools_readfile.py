"""Tests for the ReadFile tool."""

import json
from unittest.mock import patch

from tiz.tools.readfile import ReadFile

_TEST_SOCKET = "/tmp/test.sock"


class TestReadFilePrompt:
    def test_prompt_returns_valid_json(self) -> None:
        tool = ReadFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "ReadFile"

    def test_prompt_has_required_fields(self) -> None:
        tool = ReadFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert (
            data["description"]
            == "Read a text file's contents. Each line is prefixed with its line number."
        )
        assert data["parameters"]["type"] == "object"
        assert (
            data["parameters"]["properties"]["path"]["description"] == "The file path"
        )
        assert data["parameters"]["required"] == ["path"]

    def test_prompt_view_range_schema_constraints(self) -> None:
        tool = ReadFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        view_range = data["parameters"]["properties"]["view_range"]
        assert view_range["minItems"] == 2
        assert view_range["maxItems"] == 2
        assert view_range["items"]["type"] == "integer"
        assert view_range["items"]["minimum"] == 1

    def test_prompt_view_range_description_1indexed(self) -> None:
        tool = ReadFile(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        desc = data["parameters"]["properties"]["view_range"]["description"]
        assert "1-indexed" in desc
        assert "inclusive" in desc


class TestReadFileFname:
    def test_fname(self) -> None:
        assert ReadFile.fname() == "ReadFile"


class TestReadFileRunValidation:
    def test_missing_path(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({})
        assert result == "ERROR: ReadFile takes a mandatory 'path' argument!"

    def test_empty_path(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": ""})
        assert result == "ERROR: ReadFile takes a mandatory 'path' argument!"

    def test_path_not_string(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": 123})
        assert result == "ERROR: ReadFile takes a mandatory 'path' argument!"

    def test_path_none(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": None})
        assert result == "ERROR: ReadFile takes a mandatory 'path' argument!"

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        with patch.object(ReadFile, "_call", return_value="content") as mock_call:
            result = tool.run({"path": "test.txt"})
        assert result == "content"
        mock_call.assert_called_once_with({"path": "test.txt"})

    def test_calls_call_with_view_range(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        with patch.object(ReadFile, "_call", return_value="content") as mock_call:
            result = tool.run({"path": "test.txt", "view_range": [1, 5]})
        assert result == "content"
        mock_call.assert_called_once_with({"path": "test.txt", "view_range": [1, 5]})

    def test_path_too_large(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        from tiz.tools.base import MAX_INPUT_SIZE

        result = tool.run({"path": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_view_range_not_a_list(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": "not-a-list"})
        assert result == "ERROR: view_range must be a list of two integers"

    def test_view_range_wrong_length(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": [1, 2, 3]})
        assert result == "ERROR: view_range must be a list of two integers"

    def test_view_range_non_integer_items(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": [1, "two"]})
        assert result == "ERROR: view_range must contain integers only"

    def test_view_range_bool_first_item(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": [True, 5]})
        assert result == "ERROR: view_range must contain integers only"

    def test_view_range_bool_second_item(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": [1, False]})
        assert result == "ERROR: view_range must contain integers only"

    def test_view_range_both_bool(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": [True, False]})
        assert result == "ERROR: view_range must contain integers only"

    def test_view_range_empty_list(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": []})
        assert result == "ERROR: view_range must be a list of two integers"

    def test_strips_description_and_name(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        with patch.object(ReadFile, "_call", return_value="content") as mock_call:
            result = tool.run(
                {"path": "test.txt", "description": "foo", "name": "ReadFile"}
            )
        assert result == "content"
        mock_call.assert_called_once_with({"path": "test.txt"})

    def test_view_range_positive_values(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": [0, 5]})
        assert result == "ERROR: view_range values must be positive"

    def test_view_range_first_gt_second(self, socket_path: str) -> None:
        tool = ReadFile(socket_path)
        result = tool.run({"path": "test.txt", "view_range": [5, 3]})
        assert result == "ERROR: view_range[0] must be <= view_range[1]"


class TestReadFileFormatConfirmation:
    def test_format_confirmation(self) -> None:
        tool = ReadFile(_TEST_SOCKET)
        result = tool.format_confirmation({"path": "/home/file.txt"}, markdown=False)
        assert result == "Read file: /home/file.txt"

    def test_format_confirmation_markdown(self) -> None:
        tool = ReadFile(_TEST_SOCKET)
        result = tool.format_confirmation({"path": "/tmp/test.txt"}, markdown=True)
        assert result == "Read file: `/tmp/test.txt`"

    def test_format_confirmation_markdown_safe_backticks(self) -> None:
        tool = ReadFile(_TEST_SOCKET)
        result = tool.format_confirmation({"path": "/tmp/`bad`.txt"}, markdown=True)
        assert result == "Read file: `/tmp/bad.txt`"

    def test_format_confirmation_empty_path(self) -> None:
        tool = ReadFile(_TEST_SOCKET)
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Read file: "
