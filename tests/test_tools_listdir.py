"""Tests for the ListDir tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.listdir import ListDir

_TEST_SOCKET = "/tmp/test.sock"


class TestListDirPrompt:
    def test_prompt_returns_valid_json(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "ListDir"

    def test_prompt_has_required_fields(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["parameters"]["required"] == ["path"]

    def test_prompt_has_all_properties(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        data = json.loads(tool.prompt())
        assert data["description"] != ""
        assert data["parameters"]["type"] == "object"
        props = data["parameters"]["properties"]
        assert props["path"]["type"] == "string"
        assert props["path"]["description"] == "The directory path (required)"
        assert props["recursive"]["type"] == "boolean"
        assert props["recursive"]["description"] == "List recursively (default false)"
        assert props["show_hidden"]["type"] == "boolean"
        assert (
            props["show_hidden"]["description"] == "Show hidden files (default false)"
        )


class TestListDirFname:
    def test_fname(self) -> None:
        assert ListDir.fname() == "ListDir"


class TestListDirRunValidation:
    def test_missing_path(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({})
        assert result == "ERROR: ListDir takes a mandatory 'path' argument!"

    def test_empty_path(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": ""})
        assert result == "ERROR: ListDir takes a mandatory 'path' argument!"

    def test_non_string_path(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": 123})
        assert result == "ERROR: ListDir takes a mandatory 'path' argument!"

    def test_path_as_list(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": ["/tmp", "/var"]})
        assert result == "ERROR: ListDir takes a mandatory 'path' argument!"

    def test_path_as_none(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": None})
        assert result == "ERROR: ListDir takes a mandatory 'path' argument!"

    def test_path_too_large(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_invalid_recursive_type_string(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": "/tmp", "recursive": "yes"})
        assert result == "ERROR: recursive must be a boolean"

    def test_invalid_recursive_type_int(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": "/tmp", "recursive": 1})
        assert result == "ERROR: recursive must be a boolean"

    def test_invalid_show_hidden_type_string(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": "/tmp", "show_hidden": "yes"})
        assert result == "ERROR: show_hidden must be a boolean"

    def test_invalid_show_hidden_type_int(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        result = tool.run({"path": "/tmp", "show_hidden": 1})
        assert result == "ERROR: show_hidden must be a boolean"

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        with patch.object(ListDir, "_call", return_value="entries") as mock_call:
            result = tool.run({"path": "/tmp"})
        assert result == "entries"
        mock_call.assert_called_once_with({"path": "/tmp"})

    def test_calls_call_with_recursive(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        with patch.object(ListDir, "_call", return_value="entries") as mock_call:
            result = tool.run({"path": "/tmp", "recursive": True})
        assert result == "entries"
        mock_call.assert_called_once_with({"path": "/tmp", "recursive": True})

    def test_calls_call_with_show_hidden(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        with patch.object(ListDir, "_call", return_value="entries") as mock_call:
            result = tool.run({"path": "/tmp", "show_hidden": True})
        assert result == "entries"
        mock_call.assert_called_once_with({"path": "/tmp", "show_hidden": True})

    def test_recursive_false_is_valid(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        with patch.object(ListDir, "_call", return_value="entries") as mock_call:
            result = tool.run({"path": "/tmp", "recursive": False})
        assert result == "entries"
        mock_call.assert_called_once_with({"path": "/tmp", "recursive": False})

    def test_show_hidden_false_is_valid(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        with patch.object(ListDir, "_call", return_value="entries") as mock_call:
            result = tool.run({"path": "/tmp", "show_hidden": False})
        assert result == "entries"
        mock_call.assert_called_once_with({"path": "/tmp", "show_hidden": False})

    def test_strips_description_key(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        with patch.object(ListDir, "_call", return_value="entries") as mock_call:
            result = tool.run({"path": "/tmp", "description": "some description"})
        assert result == "entries"
        mock_call.assert_called_once_with({"path": "/tmp"})

    def test_strips_name_key(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        with patch.object(ListDir, "_call", return_value="entries") as mock_call:
            result = tool.run({"path": "/tmp", "name": "ListDir"})
        assert result == "entries"
        mock_call.assert_called_once_with({"path": "/tmp"})

    def test_strips_both_description_and_name(self, socket_path: str) -> None:
        tool = ListDir(socket_path)
        with patch.object(ListDir, "_call", return_value="entries") as mock_call:
            result = tool.run(
                {"path": "/tmp", "description": "desc", "name": "n", "recursive": True}
            )
        assert result == "entries"
        mock_call.assert_called_once_with({"path": "/tmp", "recursive": True})


class TestListDirFormatConfirmation:
    def test_format_confirmation_default(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation({"path": "/home/user"}, markdown=False)
        assert result == "List dir: /home/user"

    def test_format_confirmation_recursive(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation(
            {"path": "/tmp", "recursive": True}, markdown=False
        )
        assert result == "List dir: /tmp (recursive)"

    def test_format_confirmation_hidden(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation(
            {"path": "/tmp", "show_hidden": True}, markdown=False
        )
        assert result == "List dir: /tmp (hidden)"

    def test_format_confirmation_recursive_and_hidden(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation(
            {"path": "/tmp", "recursive": True, "show_hidden": True}, markdown=False
        )
        assert result == "List dir: /tmp (recursive, hidden)"

    def test_format_confirmation_markdown(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation({"path": "/tmp"}, markdown=True)
        assert result == "List dir: `/tmp`"

    def test_format_confirmation_empty_path(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation({}, markdown=False)
        assert result == "List dir: "

    def test_format_confirmation_explicit_false(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation(
            {"path": "/tmp", "recursive": False, "show_hidden": False}, markdown=False
        )
        assert result == "List dir: /tmp"

    def test_format_confirmation_markdown_backtick_safe(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation(
            {"path": "/tmp/`backtick`.dir"}, markdown=True
        )
        assert result == "List dir: `/tmp/backtick.dir`"

    def test_format_confirmation_markdown_recursive(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation(
            {"path": "/tmp", "recursive": True}, markdown=True
        )
        assert result == "List dir: `/tmp` (recursive)"

    def test_format_confirmation_markdown_hidden(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation(
            {"path": "/tmp", "show_hidden": True}, markdown=True
        )
        assert result == "List dir: `/tmp` (hidden)"

    def test_format_confirmation_markdown_both(self) -> None:
        tool = ListDir(_TEST_SOCKET)
        result = tool.format_confirmation(
            {"path": "/tmp", "recursive": True, "show_hidden": True},
            markdown=True,
        )
        assert result == "List dir: `/tmp` (recursive, hidden)"
