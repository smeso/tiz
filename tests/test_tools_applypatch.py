"""Tests for the ApplyPatch tool."""

import json
from pathlib import Path
from unittest.mock import patch

from tiz.tools.applypatch import ApplyPatch
from tiz.tools.base import MAX_INPUT_SIZE

_TEST_SOCKET = "/tmp/test.sock"


class TestApplyPatchPrompt:
    """Prompt structure is the tool's interface contract to the LLM — every field matters."""

    def test_prompt_returns_valid_json(self) -> None:
        tool = ApplyPatch(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "ApplyPatch"
        assert data["description"] == "Apply a patch file using the patch command."
        assert data["parameters"]["type"] == "object"
        assert data["parameters"]["required"] == ["patch"]

    def test_prompt_has_all_properties(self) -> None:
        tool = ApplyPatch(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        props = data["parameters"]["properties"]
        assert props == {
            "patch": {"type": "string", "description": "The patch content"},
            "strip": {
                "type": "integer",
                "description": "Number of leading path components to strip (default 0)",
            },
            "reverse": {
                "type": "boolean",
                "description": "Reverse the patch (default false)",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory to apply the patch in",
            },
        }

    def test_prompt_only_requires_patch(self) -> None:
        tool = ApplyPatch(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["parameters"]["required"] == ["patch"]


class TestApplyPatchFname:
    def test_fname(self) -> None:
        assert ApplyPatch.fname() == "ApplyPatch"


class TestApplyPatchRunValidation:
    def test_missing_patch(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({})
        assert result == "ERROR: ApplyPatch takes a mandatory 'patch' argument!"

    def test_empty_patch(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": ""})
        assert result == "ERROR: ApplyPatch takes a mandatory 'patch' argument!"

    def test_patch_too_large(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: patch content exceeds maximum allowed size"

    def test_patch_exactly_max_size(self, socket_path: str) -> None:
        """Boundary test: exactly MAX_INPUT_SIZE bytes should pass validation."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "x" * MAX_INPUT_SIZE})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "x" * MAX_INPUT_SIZE})

    def test_patch_at_max_size_minus_one(self, socket_path: str) -> None:
        """MAX_INPUT_SIZE - 1 should also pass."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "x" * (MAX_INPUT_SIZE - 1)})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "x" * (MAX_INPUT_SIZE - 1)})

    def test_invalid_strip_string(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "strip": "0"})
        assert result == "ERROR: strip must be an integer between 0 and 20"

    def test_invalid_strip_float(self, socket_path: str) -> None:
        """Float values are not integers and must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "strip": 1.5})
        assert result == "ERROR: strip must be an integer between 0 and 20"

    def test_invalid_strip_float_integer_valued(self, socket_path: str) -> None:
        """Integer-valued floats (e.g. 3.0) must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "strip": 3.0})
        assert result == "ERROR: strip must be an integer between 0 and 20"

    def test_invalid_strip_negative(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "strip": -1})
        assert result == "ERROR: strip must be an integer between 0 and 20"

    def test_invalid_strip_too_high(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "strip": 21})
        assert result == "ERROR: strip must be an integer between 0 and 20"

    def test_valid_strip_at_boundary(self, socket_path: str) -> None:
        """Boundary test - strip=20 is accepted."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "strip": 20})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff", "strip": 20})

    def test_valid_strip_zero_omitted(self, socket_path: str) -> None:
        """Default strip=0 is not forwarded to _call."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "strip": 0})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff"})

    def test_valid_strip_positive(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "strip": 5})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff", "strip": 5})

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "--- a/file\n+++ b/file"})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "--- a/file\n+++ b/file"})

    def test_invalid_reverse_type_string(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "reverse": "not_a_bool"})
        assert result == "ERROR: reverse must be a boolean"

    def test_invalid_reverse_type_int_one(self, socket_path: str) -> None:
        """reverse=1 (int) must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "reverse": 1})
        assert result == "ERROR: reverse must be a boolean"

    def test_invalid_reverse_type_int_zero(self, socket_path: str) -> None:
        """reverse=0 (int) must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "reverse": 0})
        assert result == "ERROR: reverse must be a boolean"

    def test_invalid_reverse_type_none(self, socket_path: str) -> None:
        """reverse=None must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "reverse": None})
        assert result == "ERROR: reverse must be a boolean"

    def test_invalid_reverse_type_list(self, socket_path: str) -> None:
        """reverse=[] must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "reverse": []})
        assert result == "ERROR: reverse must be a boolean"

    def test_invalid_reverse_type_dict(self, socket_path: str) -> None:
        """reverse={} must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "reverse": {}})
        assert result == "ERROR: reverse must be a boolean"

    def test_invalid_reverse_type_float(self, socket_path: str) -> None:
        """reverse=1.0 must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "reverse": 1.0})
        assert result == "ERROR: reverse must be a boolean"

    def test_valid_reverse_true(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "reverse": True})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff", "reverse": True})

    def test_valid_reverse_false_omitted(self, socket_path: str) -> None:
        """Default reverse=False is not forwarded to _call."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "reverse": False})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff"})

    def test_valid_reverse_true_with_strip(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "reverse": True, "strip": 2})
        assert result == "patched"
        mock_call.assert_called_once_with(
            {"patch": "diff", "reverse": True, "strip": 2}
        )

    def test_drops_unexpected_keys(self, socket_path: str) -> None:
        """Unexpected keys must be dropped, not forwarded."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run(
                {
                    "patch": "diff",
                    "name": "EvilName",
                    "description": "EvilDescription",
                    "foo": "bar",
                    "extra": 42,
                    "strip": 1,
                    "cwd": "/tmp",
                }
            )
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff", "strip": 1, "cwd": "/tmp"})

    def test_drops_unexpected_keys_no_valid_extra(self, socket_path: str) -> None:
        """All non-valid keys are dropped."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run(
                {
                    "patch": "diff",
                    "name": "EvilName",
                    "description": "EvilDescription",
                    "socket_path": "/tmp/bad.sock",
                    "command": "rm -rf /",
                    "timeout": 999,
                }
            )
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff"})

    def test_strips_name_and_description_from_args(self, socket_path: str) -> None:
        """name and description keys must be removed before _call."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run(
                {
                    "patch": "diff",
                    "name": "EvilName",
                    "description": "EvilDescription",
                    "strip": 1,
                    "cwd": "/tmp",
                }
            )
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff", "strip": 1, "cwd": "/tmp"})

    def test_strips_name_only(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "name": "EvilName"})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff"})

    def test_strips_description_only(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "description": "EvilDescription"})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff"})

    def test_invalid_cwd_type(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "cwd": 123})
        assert result == "ERROR: cwd must be a string"

    def test_invalid_cwd_empty_string(self, socket_path: str) -> None:
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "cwd": ""})
        assert result == "ERROR: cwd must be a string"

    def test_cwd_none_omitted_from_call(self, socket_path: str) -> None:
        """cwd=None is not forwarded to _call."""
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "cwd": None})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff"})

    def test_valid_cwd_calls_call(self, socket_path: str, tmp_path: Path) -> None:
        tool = ApplyPatch(socket_path)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run({"patch": "diff", "cwd": str(tmp_path)})
        assert result == "patched"
        mock_call.assert_called_once_with({"patch": "diff", "cwd": str(tmp_path)})

    def test_cwd_nonexistent_triggers_socket_error(self, socket_path: str) -> None:
        """Non-existent cwd is passed through; 'ERROR: tool communication failed'
        comes from _call() failing to connect to the socket, not from cwd validation.
        The source code itself lacks cwd existence validation."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "cwd": "/nonexistent/dir"})
        assert "ERROR: tool communication failed" in result

    def test_patch_non_string_int(self, socket_path: str) -> None:
        """Non-string patch must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": 123})
        assert result == "ERROR: ApplyPatch takes a mandatory 'patch' argument!"

    def test_patch_non_string_bool(self, socket_path: str) -> None:
        """Bool patch must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": True})
        assert result == "ERROR: ApplyPatch takes a mandatory 'patch' argument!"

    def test_patch_non_string_list(self, socket_path: str) -> None:
        """List patch must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": ["diff"]})
        assert result == "ERROR: ApplyPatch takes a mandatory 'patch' argument!"

    def test_strip_bool_true_is_rejected(self, socket_path: str) -> None:
        """strip=True (bool) must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "strip": True})
        assert result == "ERROR: strip must be an integer between 0 and 20"

    def test_strip_bool_false_is_rejected(self, socket_path: str) -> None:
        """strip=False (bool) must be rejected."""
        tool = ApplyPatch(socket_path)
        result = tool.run({"patch": "diff", "strip": False})
        assert result == "ERROR: strip must be an integer between 0 and 20"

    def test_args_dict_not_mutated(self, socket_path: str) -> None:
        """Original args dict must not be mutated."""
        tool = ApplyPatch(socket_path)
        original = {"patch": "diff", "name": "EvilName", "description": "EvilDesc"}
        args_copy = dict(original)
        with patch.object(ApplyPatch, "_call", return_value="patched") as mock_call:
            result = tool.run(args_copy)
        assert result == "patched"
        assert args_copy == original, f"Expected {original!r}, got {args_copy!r}"
        mock_call.assert_called_once_with({"patch": "diff"})


class TestApplyPatchFormatConfirmation:
    def test_format_confirmation_defaults(self) -> None:
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation({"patch": "diff content"}, markdown=False)
        assert result == "Apply patch in: ."

    def test_format_confirmation_markdown_no_cwd(self) -> None:
        """markdown=True with no cwd."""
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation({"patch": "diff"}, markdown=True)
        assert result == "Apply patch in: `.`"

    def test_format_confirmation_markdown(self) -> None:
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "cwd": "/home/user"}, markdown=True
        )
        assert result == "Apply patch in: `/home/user`"

    def test_format_confirmation_reverse(self) -> None:
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "reverse": True}, markdown=False
        )
        assert result == "Reverse patch in: ."

    def test_format_confirmation_reverse_markdown_no_cwd(self) -> None:
        """reverse=True, markdown=True, no cwd."""
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "reverse": True}, markdown=True
        )
        assert result == "Reverse patch in: `.`"

    def test_format_confirmation_reverse_markdown(self) -> None:
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "reverse": True, "cwd": "/tmp"}, markdown=True
        )
        assert result == "Reverse patch in: `/tmp`"

    def test_format_confirmation_reverse_false_explicit(self) -> None:
        """reverse=False explicitly, with cwd, markdown=False."""
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "reverse": False, "cwd": "/home"}, markdown=False
        )
        assert result == "Apply patch in: /home"

    def test_format_confirmation_reverse_false_markdown(self) -> None:
        """reverse=False explicitly, no cwd, markdown=True."""
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "reverse": False}, markdown=True
        )
        assert result == "Apply patch in: `.`"

    def test_format_confirmation_cwd_with_backticks(self) -> None:
        """cwd with backticks in markdown mode — _safe_md path."""
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "cwd": "`/tmp/dir`"}, markdown=True
        )
        assert result == "Apply patch in: `/tmp/dir`"

    def test_format_confirmation_cwd_none_markdown_false(self) -> None:
        """cwd=None with markdown=False should default to '.'."""
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "cwd": None}, markdown=False
        )
        assert result == "Apply patch in: ."

    def test_format_confirmation_cwd_none_markdown_true(self) -> None:
        """cwd=None with markdown=True should default to '.'."""
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation({"patch": "diff", "cwd": None}, markdown=True)
        assert result == "Apply patch in: `.`"

    def test_format_confirmation_reverse_cwd_none(self) -> None:
        """reverse=True, cwd=None — should not crash."""
        tool = ApplyPatch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"patch": "diff", "reverse": True, "cwd": None}, markdown=True
        )
        assert result == "Reverse patch in: `.`"
