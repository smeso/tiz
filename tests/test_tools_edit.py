"""Tests for the Edit tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.edit import Edit


class TestEditPrompt:
    def test_prompt_structure(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)

        assert data["name"] == "Edit"
        assert (
            data["description"]
            == "Edit a file by replacing an old string with a new string."
        )

        params = data["parameters"]
        assert params["type"] == "object"
        assert set(params["required"]) == {"path", "old_string", "new_string"}

        props = params["properties"]
        assert props["path"] == {"type": "string", "description": "The file path"}
        assert props["old_string"] == {
            "type": "string",
            "description": "The string to replace",
        }
        assert props["new_string"] == {
            "type": "string",
            "description": "The replacement string",
        }
        assert props["expected_replacements"] == {
            "type": "integer",
            "description": "Expected number of replacements (-1 for all, default 1)",
        }

        assert "expected_replacements" not in params["required"]


class TestEditFname:
    def test_fname(self) -> None:
        assert Edit.fname() == "Edit"


class TestEditRunValidation:
    def test_missing_path(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"old_string": "a", "new_string": "b"})
        assert result == "ERROR: Edit takes a mandatory 'path' argument!"

    def test_empty_path(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"path": "", "old_string": "a", "new_string": "b"})
        assert result == "ERROR: Edit takes a mandatory 'path' argument!"

    def test_missing_old_string(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"path": "test.txt", "new_string": "b"})
        assert result == "ERROR: Edit takes a mandatory 'old_string' argument!"

    def test_empty_old_string(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"path": "test.txt", "old_string": "", "new_string": "b"})
        assert result == "ERROR: Edit takes a mandatory 'old_string' argument!"

    def test_missing_new_string(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"path": "test.txt", "old_string": "a"})
        assert result == "ERROR: Edit takes a mandatory 'new_string' argument!"

    def test_empty_new_string_is_valid(self, socket_path: str) -> None:
        """Empty new_string means 'delete the matched string' - should be allowed."""
        tool = Edit(socket_path)
        with patch.object(Edit, "_call", return_value="deleted") as mock_call:
            result = tool.run({"path": "test.txt", "old_string": "a", "new_string": ""})
        assert result == "deleted"
        mock_call.assert_called_once_with(
            {"path": "test.txt", "old_string": "a", "new_string": ""}
        )

    def test_path_not_string(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"path": 123, "old_string": "a", "new_string": "b"})
        assert result == "ERROR: Edit takes a mandatory 'path' argument!"

    def test_path_too_large(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run(
            {
                "path": "x" * (MAX_INPUT_SIZE + 1),
                "old_string": "a",
                "new_string": "b",
            }
        )
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_old_string_not_string(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"path": "test.txt", "old_string": 123, "new_string": "b"})
        assert result == "ERROR: Edit takes a mandatory 'old_string' argument!"

    def test_new_string_not_string(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"path": "test.txt", "old_string": "a", "new_string": 123})
        assert result == "ERROR: Edit takes a mandatory 'new_string' argument!"

    def test_new_string_none(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run({"path": "test.txt", "old_string": "a", "new_string": None})
        assert result == "ERROR: Edit takes a mandatory 'new_string' argument!"

    def test_old_string_too_large(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run(
            {
                "path": "test.txt",
                "old_string": "x" * (MAX_INPUT_SIZE + 1),
                "new_string": "b",
            }
        )
        assert result == "ERROR: old_string exceeds maximum allowed size"

    def test_new_string_too_large(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run(
            {
                "path": "test.txt",
                "old_string": "a",
                "new_string": "x" * (MAX_INPUT_SIZE + 1),
            }
        )
        assert result == "ERROR: new_string exceeds maximum allowed size"

    def test_invalid_expected_replacements_zero(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run(
            {
                "path": "test.txt",
                "old_string": "a",
                "new_string": "b",
                "expected_replacements": 0,
            }
        )
        assert (
            result
            == "ERROR: expected_replacements must be -1 or an integer greater than 0"
        )

    def test_invalid_expected_replacements_negative_two(self, socket_path: str) -> None:
        """-2 is < -1, exercises the er < -1 branch distinct from er == 0."""
        tool = Edit(socket_path)
        result = tool.run(
            {
                "path": "test.txt",
                "old_string": "a",
                "new_string": "b",
                "expected_replacements": -2,
            }
        )
        assert (
            result
            == "ERROR: expected_replacements must be -1 or an integer greater than 0"
        )

    def test_invalid_expected_replacements_true(self, socket_path: str) -> None:
        """True is a bool, not a valid int - must be rejected."""
        tool = Edit(socket_path)
        result = tool.run(
            {
                "path": "test.txt",
                "old_string": "a",
                "new_string": "b",
                "expected_replacements": True,
            }
        )
        assert (
            result
            == "ERROR: expected_replacements must be -1 or an integer greater than 0"
        )

    def test_invalid_expected_replacements_false(self, socket_path: str) -> None:
        """False is a bool, not a valid int - must be rejected."""
        tool = Edit(socket_path)
        result = tool.run(
            {
                "path": "test.txt",
                "old_string": "a",
                "new_string": "b",
                "expected_replacements": False,
            }
        )
        assert (
            result
            == "ERROR: expected_replacements must be -1 or an integer greater than 0"
        )

    def test_invalid_expected_replacements_type(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.run(
            {
                "path": "test.txt",
                "old_string": "a",
                "new_string": "b",
                "expected_replacements": "1",
            }
        )
        assert (
            result
            == "ERROR: expected_replacements must be -1 or an integer greater than 0"
        )

    def test_valid_expected_replacements_negative_one(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        with patch.object(Edit, "_call", return_value="ok") as mock_call:
            result = tool.run(
                {
                    "path": "test.txt",
                    "old_string": "a",
                    "new_string": "b",
                    "expected_replacements": -1,
                }
            )
        assert result == "ok"
        mock_call.assert_called_once_with(
            {
                "path": "test.txt",
                "old_string": "a",
                "new_string": "b",
                "expected_replacements": -1,
            }
        )

    def test_valid_expected_replacements_positive(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        with patch.object(Edit, "_call", return_value="ok") as mock_call:
            result = tool.run(
                {
                    "path": "test.txt",
                    "old_string": "a",
                    "new_string": "b",
                    "expected_replacements": 3,
                }
            )
        assert result == "ok"
        mock_call.assert_called_once_with(
            {
                "path": "test.txt",
                "old_string": "a",
                "new_string": "b",
                "expected_replacements": 3,
            }
        )

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        with patch.object(Edit, "_call", return_value="edited") as mock_call:
            result = tool.run(
                {
                    "path": "test.txt",
                    "old_string": "old",
                    "new_string": "new",
                }
            )
        assert result == "edited"
        mock_call.assert_called_once_with(
            {
                "path": "test.txt",
                "old_string": "old",
                "new_string": "new",
            }
        )

    def test_strips_name_and_description(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        with patch.object(Edit, "_call", return_value="edited") as mock_call:
            result = tool.run(
                {
                    "path": "test.txt",
                    "old_string": "old",
                    "new_string": "new",
                    "name": "EvilEdit",
                    "description": "malicious",
                }
            )
        assert result == "edited"
        mock_call.assert_called_once_with(
            {
                "path": "test.txt",
                "old_string": "old",
                "new_string": "new",
            }
        )

    def test_strips_name_and_description_with_expected_replacements(
        self, socket_path: str
    ) -> None:
        """expected_replacements is preserved even when name/description are stripped."""
        tool = Edit(socket_path)
        with patch.object(Edit, "_call", return_value="edited") as mock_call:
            result = tool.run(
                {
                    "path": "test.txt",
                    "old_string": "old",
                    "new_string": "new",
                    "expected_replacements": 5,
                    "name": "EvilEdit",
                    "description": "malicious",
                }
            )
        assert result == "edited"
        mock_call.assert_called_once_with(
            {
                "path": "test.txt",
                "old_string": "old",
                "new_string": "new",
                "expected_replacements": 5,
            }
        )


class TestEditFormatConfirmation:
    def test_format_confirmation(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {"path": "test.txt", "old_string": "old", "new_string": "replacement text"},
            markdown=False,
        )
        assert result == "Edit file: test.txt (replace 'old' with 'replacement text')"

    def test_format_confirmation_markdown(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {"path": "/home/file.txt", "old_string": "old", "new_string": "new"},
            markdown=True,
        )
        assert result == "Edit file: `/home/file.txt` (replace `old` with `new`)"

    def test_format_confirmation_empty_path(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Edit file:  (replace '' with '')"

    def test_format_confirmation_path_none(self, socket_path: str) -> None:
        """path=None should be treated as empty string."""
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {"path": None, "old_string": "a", "new_string": "b"}, markdown=False
        )
        assert result == "Edit file:  (replace 'a' with 'b')"

    def test_format_confirmation_long_preview_truncated(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        long_old = "y" * 100
        long_new = "x" * 100
        result = tool.format_confirmation(
            {"path": "f.txt", "old_string": long_old, "new_string": long_new},
            markdown=False,
        )
        assert result == f"Edit file: f.txt (replace '{'y' * 80}' with '{'x' * 80}')"

    def test_format_confirmation_long_path_truncated(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        long_path = "p" * 100
        result = tool.format_confirmation(
            {"path": long_path, "old_string": "a", "new_string": "b"},
            markdown=False,
        )
        assert result == f"Edit file: {'p' * 80} (replace 'a' with 'b')"

    def test_format_confirmation_markdown_truncation(self, socket_path: str) -> None:
        """markdown=True with strings exceeding 80 chars: truncation before backtick removal."""
        tool = Edit(socket_path)
        long_old = "x" * 100
        long_new = "y" * 100
        result = tool.format_confirmation(
            {"path": "f.txt", "old_string": long_old, "new_string": long_new},
            markdown=True,
        )
        assert result == (
            "Edit file: `f.txt` (replace `" + "x" * 80 + "` with `" + "y" * 80 + "`)"
        )

    def test_format_confirmation_markdown_backticks_in_strings(
        self, socket_path: str
    ) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {
                "path": "/home/foo`bar.txt",
                "old_string": "foo`bar",
                "new_string": "baz`qux",
            },
            markdown=True,
        )
        assert (
            result == "Edit file: `/home/foobar.txt` (replace `foobar` with `bazqux`)"
        )

    def test_format_confirmation_markdown_backticks_in_path(
        self, socket_path: str
    ) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {
                "path": "/home/foo`bar.txt",
                "old_string": "simple",
                "new_string": "change",
            },
            markdown=True,
        )
        assert (
            result == "Edit file: `/home/foobar.txt` (replace `simple` with `change`)"
        )

    def test_format_confirmation_old_string_none(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {"path": "f.txt", "old_string": None, "new_string": "new"},
            markdown=False,
        )
        assert result == "Edit file: f.txt (replace '' with 'new')"

    def test_format_confirmation_new_string_none(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {"path": "f.txt", "old_string": "old", "new_string": None},
            markdown=False,
        )
        assert result == "Edit file: f.txt (replace 'old' with '')"

    def test_format_confirmation_no_old_string(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {"path": "f.txt", "new_string": "new"}, markdown=False
        )
        assert result == "Edit file: f.txt (replace '' with 'new')"

    def test_format_confirmation_no_new_string(self, socket_path: str) -> None:
        tool = Edit(socket_path)
        result = tool.format_confirmation(
            {"path": "f.txt", "old_string": "old"}, markdown=False
        )
        assert result == "Edit file: f.txt (replace 'old' with '')"


class TestEditSafeMd:
    def test_safe_md_default_max_len(self) -> None:
        result = Edit._safe_md("a" * 100)
        assert result == "a" * 80
        assert len(result) == 80

    def test_safe_md_custom_max_len(self) -> None:
        result = Edit._safe_md("hello world", max_len=5)
        assert result == "hello"

    def test_safe_md_removes_backticks(self) -> None:
        result = Edit._safe_md("foo`bar`baz")
        assert result == "foobarbaz"
        assert "`" not in result

    def test_safe_md_truncates_then_removes_backticks(self) -> None:
        """Truncation happens before backtick removal, so a backtick at the truncation
        boundary is still removed."""
        result = Edit._safe_md("a`" + "b" * 100, max_len=3)
        assert result == "ab"
