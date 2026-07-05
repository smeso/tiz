"""Tests for the CargoFetch tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.cargofetch import CargoFetch

_TEST_SOCKET = "/tmp/test.sock"


class TestCargoFetchPrompt:
    def test_prompt_returns_valid_json(self) -> None:
        tool = CargoFetch(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "CargoFetch"
        assert "description" in data
        assert "parameters" in data
        assert data["parameters"]["type"] == "object"

    def test_prompt_has_required_fields(self) -> None:
        tool = CargoFetch(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert "path" in data["parameters"]["required"]


class TestCargoFetchFname:
    def test_fname(self) -> None:
        assert CargoFetch.fname() == "CargoFetch"


class TestCargoFetchRunValidation:
    def test_missing_path(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({})
        assert result == "ERROR: CargoFetch takes a mandatory 'path' argument!"

    def test_empty_path(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": ""})
        assert result == "ERROR: CargoFetch takes a mandatory 'path' argument!"

    def test_path_too_large(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_path_non_string_int(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": 42})
        assert result == "ERROR: CargoFetch takes a mandatory 'path' argument!"

    def test_path_non_string_list(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": ["/project"]})
        assert result == "ERROR: CargoFetch takes a mandatory 'path' argument!"

    def test_path_non_string_dict(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": {"dir": "/project"}})
        assert result == "ERROR: CargoFetch takes a mandatory 'path' argument!"

    def test_timeout_bool_true_rejected(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": "/some/project", "timeout": True})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_timeout_bool_false_rejected(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": "/some/project", "timeout": False})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_invalid_timeout_type(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": "/some/project", "timeout": "300"})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_timeout_too_low(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": "/some/project", "timeout": 0})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_timeout_too_high(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": "/some/project", "timeout": 601})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_negative_timeout(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        result = tool.run({"path": "/some/project", "timeout": -1})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_timeout_minimum_valid(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        with patch.object(CargoFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/p", "timeout": 1})
        assert result == "ok"
        mock_call.assert_called_once_with({"path": "/p", "timeout": 1})

    def test_timeout_maximum_valid(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        with patch.object(CargoFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/p", "timeout": 600})
        assert result == "ok"
        mock_call.assert_called_once_with({"path": "/p", "timeout": 600})

    def test_valid_timeout(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        with patch.object(CargoFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project", "timeout": 300})
        assert result == "ok"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_default_timeout(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        with patch.object(CargoFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project"})
        assert result == "ok"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_description_is_removed_from_call(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        args = {"path": "/some/project", "description": "desc"}
        with patch.object(CargoFetch, "_call", return_value="ok") as mock_call:
            tool.run(args)
        assert "description" in args
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_name_is_removed_from_call(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        args = {"path": "/some/project", "name": "test"}
        with patch.object(CargoFetch, "_call", return_value="ok") as mock_call:
            tool.run(args)
        assert "name" in args
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = CargoFetch(socket_path)
        with patch.object(CargoFetch, "_call", return_value="output") as mock_call:
            result = tool.run({"path": "/some/project"})
        assert result == "output"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})


class TestCargoFetchFormatConfirmation:
    def test_format_confirmation(self) -> None:
        tool = CargoFetch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/home/user/rust-project"}, markdown=False
        )
        assert result == "Cargo fetch in: /home/user/rust-project"

    def test_format_confirmation_markdown(self) -> None:
        tool = CargoFetch("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/tmp/proj"}, markdown=True)
        assert result == "Cargo fetch in: `/tmp/proj`"

    def test_format_confirmation_empty_path(self) -> None:
        tool = CargoFetch("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Cargo fetch in: "

    def test_format_confirmation_markdown_safe_md(self) -> None:
        tool = CargoFetch("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/`evil`/path"}, markdown=True)
        assert result == "Cargo fetch in: `/evil/path`"

    def test_format_confirmation_empty_path_markdown(self) -> None:
        tool = CargoFetch("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=True)
        assert result == "Cargo fetch in: ``"
