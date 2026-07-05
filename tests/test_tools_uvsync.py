"""Tests for the UvSync tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.uvsync import UvSync

_TEST_SOCKET = "/tmp/test.sock"


class TestUvSyncPrompt:
    def test_prompt_returns_valid_json(self) -> None:
        tool = UvSync(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "UvSync"
        assert "description" in data
        assert "parameters" in data

    def test_prompt_has_required_fields(self) -> None:
        tool = UvSync(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert "path" in data["parameters"]["required"]

    def test_prompt_schema_has_one_of_for_group(self) -> None:
        """Verify that the schema uses oneOf for group to match _normalize_list."""
        tool = UvSync(_TEST_SOCKET)
        data = json.loads(tool.prompt())
        group_schema = data["parameters"]["properties"]["group"]
        assert "oneOf" in group_schema
        assert {"type": "string"} in group_schema["oneOf"]
        assert {"type": "array", "items": {"type": "string"}} in group_schema["oneOf"]

    def test_prompt_schema_has_one_of_for_extra(self) -> None:
        """Verify that the schema uses oneOf for extra to match _normalize_list."""
        tool = UvSync(_TEST_SOCKET)
        data = json.loads(tool.prompt())
        extra_schema = data["parameters"]["properties"]["extra"]
        assert "oneOf" in extra_schema
        assert {"type": "string"} in extra_schema["oneOf"]
        assert {"type": "array", "items": {"type": "string"}} in extra_schema["oneOf"]


class TestUvSyncFname:
    def test_fname(self) -> None:
        assert UvSync.fname() == "UvSync"


class TestUvSyncRunValidation:
    def test_missing_path(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        result = tool.run({})
        assert result == "ERROR: UvSync takes a mandatory 'path' argument!"

    def test_path_not_a_string(self, socket_path: str) -> None:
        """Non-string path should be rejected before len() call."""
        tool = UvSync(socket_path)
        result = tool.run({"path": 123})
        assert result == "ERROR: UvSync takes a mandatory 'path' argument!"

    def test_path_is_list(self, socket_path: str) -> None:
        """List path should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": ["/some/project"]})
        assert result == "ERROR: UvSync takes a mandatory 'path' argument!"

    def test_empty_path(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        result = tool.run({"path": ""})
        assert result == "ERROR: UvSync takes a mandatory 'path' argument!"

    def test_path_too_large(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        result = tool.run({"path": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_invalid_timeout_type(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "timeout": "300"})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_bool_true_timeout_rejected(self, socket_path: str) -> None:
        """bool is subclass of int; True should be rejected as timeout."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "timeout": True})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_bool_false_timeout_rejected(self, socket_path: str) -> None:
        """bool is subclass of int; False should be rejected as timeout."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "timeout": False})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_timeout_too_low(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "timeout": 0})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_timeout_too_high(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "timeout": 601})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_negative_timeout(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "timeout": -1})
        assert result == "ERROR: timeout must be an integer between 1 and 600"

    def test_valid_timeout(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project", "timeout": 300})
        assert result == "ok"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_group_string(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project", "group": "dev"})
        assert result == "ok"
        mock_call.assert_called_once_with(
            {"path": "/some/project", "group": ["dev"], "timeout": 300}
        )

    def test_group_list(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project", "group": ["dev", "docs"]})
        assert result == "ok"
        mock_call.assert_called_once_with(
            {"path": "/some/project", "group": ["dev", "docs"], "timeout": 300}
        )

    def test_group_empty_string(self, socket_path: str) -> None:
        """Empty string for group should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "group": ""})
        assert (
            result
            == "ERROR: group must be a non-empty string or list of non-empty strings"
        )

    def test_group_whitespace_string(self, socket_path: str) -> None:
        """Whitespace-only string for group should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "group": "   "})
        assert (
            result
            == "ERROR: group must be a non-empty string or list of non-empty strings"
        )

    def test_group_list_with_whitespace_string(self, socket_path: str) -> None:
        """List containing whitespace-only string for group should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "group": ["dev", "   "]})
        assert (
            result
            == "ERROR: group must be a non-empty string or list of non-empty strings"
        )

    def test_group_empty_list(self, socket_path: str) -> None:
        """Empty list for group should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "group": []})
        assert (
            result
            == "ERROR: group must be a non-empty string or list of non-empty strings"
        )

    def test_group_list_with_empty_string(self, socket_path: str) -> None:
        """List containing empty string for group should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "group": ["dev", ""]})
        assert (
            result
            == "ERROR: group must be a non-empty string or list of non-empty strings"
        )

    def test_extra_string(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project", "extra": "web"})
        assert result == "ok"
        mock_call.assert_called_once_with(
            {"path": "/some/project", "extra": ["web"], "timeout": 300}
        )

    def test_extra_list(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project", "extra": ["web", "cli"]})
        assert result == "ok"
        mock_call.assert_called_once_with(
            {"path": "/some/project", "extra": ["web", "cli"], "timeout": 300}
        )

    def test_group_and_extra_together(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            result = tool.run(
                {
                    "path": "/some/project",
                    "group": ["dev"],
                    "extra": ["web"],
                }
            )
        assert result == "ok"
        mock_call.assert_called_once_with(
            {
                "path": "/some/project",
                "group": ["dev"],
                "extra": ["web"],
                "timeout": 300,
            }
        )

    def test_group_not_passed(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project"})
        assert result == "ok"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_group_non_string_non_list(self, socket_path: str) -> None:
        """Non-string/non-list group should return an error."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "group": 42})
        assert (
            result
            == "ERROR: group must be a non-empty string or list of non-empty strings"
        )

    def test_group_list_with_non_strings(self, socket_path: str) -> None:
        """List of non-strings for group should return an error."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "group": [1, 2]})
        assert (
            result
            == "ERROR: group must be a non-empty string or list of non-empty strings"
        )

    def test_group_dict(self, socket_path: str) -> None:
        """Dict for group should return an error."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "group": {"foo": "bar"}})
        assert (
            result
            == "ERROR: group must be a non-empty string or list of non-empty strings"
        )

    def test_extra_empty_string(self, socket_path: str) -> None:
        """Empty string for extra should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "extra": ""})
        assert (
            result
            == "ERROR: extra must be a non-empty string or list of non-empty strings"
        )

    def test_extra_whitespace_string(self, socket_path: str) -> None:
        """Whitespace-only string for extra should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "extra": "   "})
        assert (
            result
            == "ERROR: extra must be a non-empty string or list of non-empty strings"
        )

    def test_extra_list_with_whitespace_string(self, socket_path: str) -> None:
        """List containing whitespace-only string for extra should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "extra": ["web", "   "]})
        assert (
            result
            == "ERROR: extra must be a non-empty string or list of non-empty strings"
        )

    def test_extra_empty_list(self, socket_path: str) -> None:
        """Empty list for extra should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "extra": []})
        assert (
            result
            == "ERROR: extra must be a non-empty string or list of non-empty strings"
        )

    def test_extra_list_with_empty_string(self, socket_path: str) -> None:
        """List containing empty string for extra should be rejected."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "extra": ["web", ""]})
        assert (
            result
            == "ERROR: extra must be a non-empty string or list of non-empty strings"
        )

    def test_extra_non_string_non_list(self, socket_path: str) -> None:
        """Non-string/non-list extra should return an error."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "extra": None})
        assert (
            result
            == "ERROR: extra must be a non-empty string or list of non-empty strings"
        )

    def test_extra_list_with_non_strings(self, socket_path: str) -> None:
        """List of non-strings for extra should return an error."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "extra": [True, False]})
        assert (
            result
            == "ERROR: extra must be a non-empty string or list of non-empty strings"
        )

    def test_extra_dict(self, socket_path: str) -> None:
        """Dict for extra should return an error."""
        tool = UvSync(socket_path)
        result = tool.run({"path": "/some/project", "extra": {"a": "b"}})
        assert (
            result
            == "ERROR: extra must be a non-empty string or list of non-empty strings"
        )

    def test_default_timeout(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            result = tool.run({"path": "/some/project"})
        assert result == "ok"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_description_is_removed(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        args = {"path": "/some/project", "description": "desc"}
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            tool.run(args)
        assert "description" in args, "original args should not be mutated"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_name_is_removed(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        args = {"path": "/some/project", "name": "test"}
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            tool.run(args)
        assert "name" in args, "original args should not be mutated"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})

    def test_original_args_not_mutated(self, socket_path: str) -> None:
        """Verify defensive copy prevents mutation of the original dict."""
        tool = UvSync(socket_path)
        original = {
            "path": "/some/project",
            "group": "dev",
            "extra": "web",
            "description": "desc",
            "name": "UvSync",
        }
        args = dict(original)
        with patch.object(UvSync, "_call", return_value="ok") as mock_call:
            tool.run(args)
        assert args == original, f"original args mutated: {args} != {original}"
        mock_call.assert_called_once_with(
            {
                "path": "/some/project",
                "group": ["dev"],
                "extra": ["web"],
                "timeout": 300,
            }
        )

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = UvSync(socket_path)
        with patch.object(UvSync, "_call", return_value="output") as mock_call:
            result = tool.run({"path": "/some/project"})
        assert result == "output"
        mock_call.assert_called_once_with({"path": "/some/project", "timeout": 300})


class TestUvSyncFormatConfirmation:
    def test_format_confirmation(self) -> None:
        tool = UvSync("/tmp/test.sock")
        result = tool.format_confirmation(
            {"path": "/home/user/project"}, markdown=False
        )
        assert result == "Uv sync in: /home/user/project"

    def test_format_confirmation_markdown(self) -> None:
        tool = UvSync("/tmp/test.sock")
        result = tool.format_confirmation({"path": "/tmp/proj"}, markdown=True)
        assert result == "Uv sync in: `/tmp/proj`"

    def test_format_confirmation_empty_path(self) -> None:
        tool = UvSync("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Uv sync"

    def test_format_confirmation_none_path(self) -> None:
        tool = UvSync("/tmp/test.sock")
        result = tool.format_confirmation({"path": None}, markdown=False)
        assert result == "Uv sync"

    def test_format_confirmation_empty_path_markdown(self) -> None:
        tool = UvSync("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=True)
        assert result == "Uv sync"

    def test_format_confirmation_none_path_markdown(self) -> None:
        tool = UvSync("/tmp/test.sock")
        result = tool.format_confirmation({"path": None}, markdown=True)
        assert result == "Uv sync"
