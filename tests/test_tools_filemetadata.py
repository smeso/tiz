"""Tests for the FileMetadata tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.filemetadata import FileMetadata

_TEST_SOCKET = "/tmp/test.sock"


class TestFileMetadataPrompt:
    def test_prompt_returns_valid_json(self) -> None:
        tool = FileMetadata(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "FileMetadata"
        assert "description" in data
        assert data["description"]
        assert data["parameters"]["type"] == "object"
        props = data["parameters"]["properties"]
        assert props["path"]["type"] == "string"
        assert "description" in props["path"]
        assert props["path"]["description"]

    def test_prompt_has_required_fields(self) -> None:
        tool = FileMetadata(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["parameters"]["required"] == ["path"]


class TestFileMetadataFname:
    def test_fname(self) -> None:
        assert FileMetadata.fname() == "FileMetadata"


class TestFileMetadataRunValidation:
    def test_missing_path(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        result = tool.run({})
        assert result == "ERROR: FileMetadata takes a mandatory 'path' argument!"

    def test_empty_path(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        result = tool.run({"path": ""})
        assert result == "ERROR: FileMetadata takes a mandatory 'path' argument!"

    def test_path_is_none(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        result = tool.run({"path": None})
        assert result == "ERROR: FileMetadata takes a mandatory 'path' argument!"

    def test_path_is_int(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        result = tool.run({"path": 123})
        assert result == "ERROR: FileMetadata takes a mandatory 'path' argument!"

    def test_path_is_bool(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        result = tool.run({"path": True})
        assert result == "ERROR: FileMetadata takes a mandatory 'path' argument!"

    def test_path_is_list(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        result = tool.run({"path": ["/tmp"]})
        assert result == "ERROR: FileMetadata takes a mandatory 'path' argument!"

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        with patch.object(FileMetadata, "_call", return_value="metadata") as mock_call:
            result = tool.run({"path": "test.txt"})
        assert result == "metadata"
        mock_call.assert_called_once_with({"path": "test.txt"})

    def test_path_exceeds_max_input_size(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        result = tool.run({"path": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_strips_name_from_args(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        with patch.object(FileMetadata, "_call", return_value="metadata") as mock_call:
            result = tool.run({"path": "test.txt", "name": "WrongName"})
        assert result == "metadata"
        call_args = mock_call.call_args[0][0]
        assert "name" not in call_args
        assert call_args["path"] == "test.txt"

    def test_strips_description_from_args(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        with patch.object(FileMetadata, "_call", return_value="metadata") as mock_call:
            result = tool.run({"path": "test.txt", "description": "some description"})
        assert result == "metadata"
        call_args = mock_call.call_args[0][0]
        assert "description" not in call_args
        assert call_args["path"] == "test.txt"

    def test_args_not_mutated(self, socket_path: str) -> None:
        tool = FileMetadata(socket_path)
        original = {"path": "test.txt"}
        with patch.object(FileMetadata, "_call", return_value="metadata"):
            tool.run(original)
        assert original == {"path": "test.txt"}


class TestFileMetadataFormatConfirmation:
    def test_format_confirmation(self) -> None:
        tool = FileMetadata("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/home/file.txt"}, markdown=False)
        assert result == "Get metadata for: /home/file.txt"

    def test_format_confirmation_markdown(self) -> None:
        tool = FileMetadata("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/tmp"}, markdown=True)
        assert result == "Get metadata for: `/tmp`"

    def test_format_confirmation_empty_path(self) -> None:
        tool = FileMetadata("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Get metadata for: "

    def test_format_confirmation_empty_path_markdown(self) -> None:
        tool = FileMetadata("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=True)
        assert result == "Get metadata for: ``"

    def test_format_confirmation_path_is_none(self) -> None:
        tool = FileMetadata("/tmp/test.sock")
        result = tool.format_confirmation({"path": None}, markdown=False)
        assert result == "Get metadata for: None"

    def test_format_confirmation_markdown_backticks_in_path(self) -> None:
        tool = FileMetadata("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/tmp/foo`bar"}, markdown=True)
        assert result == "Get metadata for: `/tmp/foobar`"
        assert "`" not in result.replace("`", "", 2)

    def test_format_confirmation_markdown_backticks_in_path_non_markdown(self) -> None:
        tool = FileMetadata("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/tmp/foo`bar"}, markdown=False)
        assert result == "Get metadata for: /tmp/foo`bar"
        assert "`" in result
