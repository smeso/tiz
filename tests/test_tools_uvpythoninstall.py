"""Tests for the UvPythonInstall tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.uvpythoninstall import UvPythonInstall


class TestUvPythonInstallPrompt:
    """Tests for the prompt() method returning JSON tool definition."""

    def test_prompt_returns_valid_json(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "UvPythonInstall"
        assert "description" in data
        assert "parameters" in data

    def test_prompt_has_required_fields(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert "version" in data["parameters"]["required"]

    def test_prompt_has_full_structure(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        data = json.loads(tool.prompt())
        assert data["parameters"]["type"] == "object"
        assert data["parameters"]["properties"]["version"]["type"] == "string"
        assert "timeout" in data["parameters"]["properties"]
        assert data["parameters"]["properties"]["timeout"]["type"] == "integer"
        assert "version" in data["parameters"]["required"]
        assert "timeout" not in data["parameters"]["required"]


class TestUvPythonInstallFname:
    """Tests for the static fname() method."""

    def test_fname(self) -> None:
        assert UvPythonInstall.fname() == "UvPythonInstall"


class TestUvPythonInstallRunValidation:
    """Tests for run() validation of arguments."""

    def test_missing_version(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({})
        assert (
            result
            == f"ERROR: {UvPythonInstall.fname()} takes a mandatory 'version' argument!"
        )
        mock_call.assert_not_called()

    def test_empty_version(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": ""})
        assert result == "ERROR: UvPythonInstall takes a mandatory 'version' argument!"
        mock_call.assert_not_called()

    def test_version_as_none(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": None})
        assert result == "ERROR: UvPythonInstall takes a mandatory 'version' argument!"
        mock_call.assert_not_called()

    def test_version_as_dict(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": {"a": "3.12"}})
        assert result == "ERROR: UvPythonInstall takes a mandatory 'version' argument!"
        mock_call.assert_not_called()

    def test_version_as_int(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": 312})
        assert result == "ERROR: UvPythonInstall takes a mandatory 'version' argument!"
        mock_call.assert_not_called()

    def test_version_as_float(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": 3.12})
        assert result == "ERROR: UvPythonInstall takes a mandatory 'version' argument!"
        mock_call.assert_not_called()

    def test_version_as_list(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": ["3.12"]})
        assert result == "ERROR: UvPythonInstall takes a mandatory 'version' argument!"
        mock_call.assert_not_called()

    def test_version_as_bool_true(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": True})
        assert result == "ERROR: UvPythonInstall takes a mandatory 'version' argument!"
        mock_call.assert_not_called()

    def test_version_as_bool_false(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": False})
        assert result == "ERROR: UvPythonInstall takes a mandatory 'version' argument!"
        mock_call.assert_not_called()

    def test_version_too_large(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: version exceeds maximum allowed size"
        mock_call.assert_not_called()

    def test_invalid_timeout_type(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12", "timeout": "300"})
        assert result == "ERROR: timeout must be an integer between 1 and 600"
        mock_call.assert_not_called()

    def test_timeout_as_float(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12", "timeout": 300.0})
        assert result == "ERROR: timeout must be an integer between 1 and 600"
        mock_call.assert_not_called()

    def test_timeout_too_low(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12", "timeout": 0})
        assert result == "ERROR: timeout must be an integer between 1 and 600"
        mock_call.assert_not_called()

    def test_timeout_too_high(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12", "timeout": 601})
        assert result == "ERROR: timeout must be an integer between 1 and 600"
        mock_call.assert_not_called()

    def test_negative_timeout(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12", "timeout": -1})
        assert result == "ERROR: timeout must be an integer between 1 and 600"
        mock_call.assert_not_called()

    def test_valid_timeout(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12", "timeout": 300})
        assert result == "ok"
        mock_call.assert_called_once_with({"version": "3.12", "timeout": 300})

    def test_default_timeout(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12"})
        assert result == "ok"
        mock_call.assert_called_once_with({"version": "3.12"})

    def test_description_is_removed_from_call(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        args = {"version": "3.12", "description": "desc"}
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run(args)
        assert result == "ok"
        assert "description" in args  # original dict not mutated
        mock_call.assert_called_once_with({"version": "3.12"})

    def test_name_is_removed_from_call(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        args = {"version": "3.12", "name": "test"}
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run(args)
        assert result == "ok"
        assert "name" in args  # original dict not mutated
        mock_call.assert_called_once_with({"version": "3.12"})

    def test_description_and_name_removed_from_call(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        args = {"version": "3.12", "name": "test", "description": "desc", "timeout": 60}
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run(args)
        assert result == "ok"
        assert "name" in args and "description" in args  # original dict not mutated
        mock_call.assert_called_once_with({"version": "3.12", "timeout": 60})

    def test_bool_timeout_true_rejected(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12", "timeout": True})
        assert result == "ERROR: timeout must be an integer between 1 and 600"
        mock_call.assert_not_called()

    def test_bool_timeout_false_rejected(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="ok") as mock_call:
            result = tool.run({"version": "3.12", "timeout": False})
        assert result == "ERROR: timeout must be an integer between 1 and 600"
        mock_call.assert_not_called()

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        with patch.object(UvPythonInstall, "_call", return_value="output") as mock_call:
            result = tool.run({"version": "3.12"})
        assert result == "output"
        mock_call.assert_called_once_with({"version": "3.12"})


class TestUvPythonInstallFormatConfirmation:
    """Tests for format_confirmation() output formatting."""

    def test_format_confirmation(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        result = tool.format_confirmation({"version": "3.12"}, markdown=False)
        assert result == "Install Python: 3.12"

    def test_format_confirmation_markdown(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        result = tool.format_confirmation({"version": "3.10.0"}, markdown=True)
        assert result == "Install Python: `3.10.0`"

    def test_format_confirmation_empty_version(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Install Python: "

    def test_format_confirmation_empty_version_markdown(self, socket_path: str) -> None:
        tool = UvPythonInstall(socket_path)
        result = tool.format_confirmation({}, markdown=True)
        assert result == "Install Python: ``"

    def test_format_confirmation_markdown_with_backticks(
        self, socket_path: str
    ) -> None:
        tool = UvPythonInstall(socket_path)
        result = tool.format_confirmation({"version": "`3.12`"}, markdown=True)
        assert result == "Install Python: `3.12`"
