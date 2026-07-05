"""Tests for the Bash tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE, Tool
from tiz.tools.bash import Bash


class TestBashPrompt:
    def test_prompt_returns_valid_json(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "Bash"
        assert "description" in data
        assert data["parameters"]["type"] == "object"
        assert data["parameters"]["additionalProperties"] is False
        props = data["parameters"]["properties"]
        assert props["command"]["type"] == "string"
        assert "description" in props["command"]
        assert props["timeout"]["type"] == "integer"
        assert "description" in props["timeout"]
        assert props["cwd"]["type"] == "string"
        assert "description" in props["cwd"]
        assert props["env"]["type"] == "object"
        assert "description" in props["env"]
        assert props["env"]["additionalProperties"]["type"] == "string"
        assert data["parameters"]["required"] == ["command"]

    def test_prompt_has_additional_properties_false(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        data = json.loads(tool.prompt())
        assert data["parameters"].get("additionalProperties") is False

    def test_prompt_has_timeout_constraints(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        data = json.loads(tool.prompt())
        timeout_schema = data["parameters"]["properties"]["timeout"]
        assert timeout_schema["type"] == "integer"
        assert timeout_schema["minimum"] == 1
        assert timeout_schema["maximum"] == 300
        assert timeout_schema["default"] == 30


class TestBashFname:
    def test_fname(self) -> None:
        assert Bash.fname() == "Bash"


class TestBashRunValidation:
    def test_missing_command(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({})
        assert result == "ERROR: Bash takes a mandatory 'command' argument!"

    def test_empty_command(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": ""})
        assert result == "ERROR: Bash takes a mandatory 'command' argument!"

    def test_non_string_command_int(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": 123})
        assert result == "ERROR: Bash takes a mandatory 'command' argument!"

    def test_non_string_command_bool(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": True})
        assert result == "ERROR: Bash takes a mandatory 'command' argument!"

    def test_non_string_command_list(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": ["ls"]})
        assert result == "ERROR: Bash takes a mandatory 'command' argument!"

    def test_command_too_large(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: command exceeds maximum allowed size"

    def test_invalid_timeout_type(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "timeout": "30"})
        assert result == "ERROR: timeout must be an integer between 1 and 300"

    def test_timeout_too_low(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "timeout": 0})
        assert result == "ERROR: timeout must be an integer between 1 and 300"

    def test_timeout_too_high(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "timeout": 301})
        assert result == "ERROR: timeout must be an integer between 1 and 300"

    def test_boolean_timeout_true(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "timeout": True})
        assert result == "ERROR: timeout must be an integer between 1 and 300"

    def test_boolean_timeout_false(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "timeout": False})
        assert result == "ERROR: timeout must be an integer between 1 and 300"

    def test_negative_timeout(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "timeout": -1})
        assert result == "ERROR: timeout must be an integer between 1 and 300"

    def test_valid_timeout(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        with patch.object(Bash, "_call", return_value="ok") as mock_call:
            result = tool.run({"command": "echo hi", "timeout": 30})
        assert result == "ok"
        mock_call.assert_called_once_with({"command": "echo hi", "timeout": 30})

    def test_invalid_cwd_type(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "cwd": 123})
        assert result == "ERROR: cwd must be a string"

    def test_description_is_removed(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        args = {"command": "echo hi", "description": "desc"}
        with patch.object(Bash, "_call", return_value="ok") as mock_call:
            tool.run(args)
        assert "description" in args  # original unchanged (copy is mutated)
        mock_call.assert_called_once_with({"command": "echo hi"})

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        with patch.object(Bash, "_call", return_value="output") as mock_call:
            result = tool.run({"command": "echo hello"})
        assert result == "output"
        mock_call.assert_called_once_with({"command": "echo hello"})

    def test_name_is_removed(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        args = {"command": "echo hi", "name": "test"}
        with patch.object(Bash, "_call", return_value="ok") as mock_call:
            tool.run(args)
        assert "name" in args  # original unchanged (copy is mutated)
        mock_call.assert_called_once_with({"command": "echo hi"})

    def test_env_non_dict(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "env": "bad"})
        assert result == "ERROR: env must be a dict"

    def test_env_int(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "env": 123})
        assert result == "ERROR: env must be a dict"

    def test_env_list(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "env": ["a", "b"]})
        assert result == "ERROR: env must be a dict"

    def test_env_non_string_key(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "env": {123: "value"}})
        assert result == "ERROR: env keys and values must be strings"

    def test_env_non_string_value(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "env": {"key": 456}})
        assert result == "ERROR: env keys and values must be strings"

    def test_env_empty_key(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "env": {"": "val"}})
        assert result == "ERROR: env keys and values must be strings"

    def test_env_empty_value_allowed(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        with patch.object(Bash, "_call", return_value="ok") as mock_call:
            result = tool.run({"command": "echo hi", "env": {"FOO": ""}})
        assert result == "ok"
        mock_call.assert_called_once_with({"command": "echo hi", "env": {"FOO": ""}})

    def test_env_mixed_invalid(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.run({"command": "echo hi", "env": {"good": "val", "bad": 789}})
        assert result == "ERROR: env keys and values must be strings"

    def test_env_valid(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        with patch.object(Bash, "_call", return_value="ok") as mock_call:
            result = tool.run({"command": "echo hi", "env": {"FOO": "bar"}})
        assert result == "ok"
        mock_call.assert_called_once_with({"command": "echo hi", "env": {"FOO": "bar"}})

    def test_env_none_is_valid(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        with patch.object(Bash, "_call", return_value="ok") as mock_call:
            result = tool.run({"command": "echo hi", "env": None})
        assert result == "ok"
        mock_call.assert_called_once_with({"command": "echo hi", "env": None})

    def test_env_empty_dict(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        with patch.object(Bash, "_call", return_value="ok") as mock_call:
            result = tool.run({"command": "echo hi", "env": {}})
        assert result == "ok"
        mock_call.assert_called_once_with({"command": "echo hi", "env": {}})

    def test_cwd_empty_string(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        with patch.object(Bash, "_call", return_value="ok") as mock_call:
            result = tool.run({"command": "echo hi", "cwd": ""})
        assert result == "ok"
        mock_call.assert_called_once_with({"command": "echo hi", "cwd": ""})


class TestBashFormatConfirmation:
    def test_format_confirmation_no_cwd(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation({"command": "echo hello"}, markdown=False)
        assert result == "Run: echo hello"

    def test_format_confirmation_markdown_no_cwd(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation({"command": "ls -la"}, markdown=True)
        assert result == "Run: `ls -la`"

    def test_format_confirmation_with_cwd(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "make", "cwd": "/home/user/project"}, markdown=False
        )
        assert result == "Run: make in /home/user/project"

    def test_format_confirmation_markdown_with_cwd(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "make", "cwd": "/tmp"}, markdown=True
        )
        assert result == "Run: `make` in `/tmp`"

    def test_format_confirmation_long_command_truncated(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        long_cmd = "x" * 200
        result = tool.format_confirmation({"command": long_cmd}, markdown=False)
        assert result == f"Run: {'x' * 120}..."

    def test_format_confirmation_short_command_not_truncated(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation({"command": "short"}, markdown=False)
        assert result == "Run: short"

    def test_format_confirmation_exactly_120_not_truncated(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        cmd = "x" * 120
        result = tool.format_confirmation({"command": cmd}, markdown=False)
        assert result == f"Run: {'x' * 120}"

    def test_format_confirmation_with_timeout_no_markdown(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "timeout": 60}, markdown=False
        )
        assert result == "Run: echo hi (timeout=60s)"

    def test_format_confirmation_with_timeout_markdown(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "timeout": 60}, markdown=True
        )
        assert result == "Run: `echo hi` (timeout=`60s`)"

    def test_format_confirmation_with_env_no_markdown(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": {"FOO": "bar"}}, markdown=False
        )
        assert result == "Run: echo hi env: FOO=bar"

    def test_format_confirmation_with_env_markdown(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": {"FOO": "bar"}}, markdown=True
        )
        assert result == "Run: `echo hi` env: `FOO`=`bar`"

    def test_format_confirmation_with_multiple_env_vars(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "cmd", "env": {"A": "1", "B": "2"}}, markdown=False
        )
        assert result == "Run: cmd env: A=1, B=2"

    def test_format_confirmation_with_timeout_and_env(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "timeout": 30, "env": {"PATH": "/usr/bin"}},
            markdown=False,
        )
        assert result == "Run: echo hi (timeout=30s) env: PATH=/usr/bin"

    def test_format_confirmation_with_all_fields_markdown(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "cwd": "/tmp", "timeout": 30, "env": {"FOO": "bar"}},
            markdown=True,
        )
        assert result == "Run: `echo hi` in `/tmp` (timeout=`30s`) env: `FOO`=`bar`"

    def test_format_confirmation_empty_env_omitted(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": {}}, markdown=False
        )
        assert result == "Run: echo hi"

    def test_format_confirmation_none_timeout_omitted(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "timeout": None}, markdown=False
        )
        assert result == "Run: echo hi"

    def test_format_confirmation_env_value_with_backtick_markdown(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": {"FOO": "`rm -rf /`"}}, markdown=True
        )
        assert result == "Run: `echo hi` env: `FOO`=`rm -rf /`"

    def test_format_confirmation_env_value_with_space_plain(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": {"FOO": "hello world"}}, markdown=False
        )
        assert result == 'Run: echo hi env: FOO="hello world"'

    def test_format_confirmation_env_value_with_equals_plain(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": {"BAR": "a=b=c"}}, markdown=False
        )
        assert result == 'Run: echo hi env: BAR="a=b=c"'

    def test_format_confirmation_env_value_no_quoting_needed_plain(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": {"FOO": "bar"}}, markdown=False
        )
        assert result == "Run: echo hi env: FOO=bar"

    def test_format_confirmation_markdown_multiple_env_with_backticks(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "cmd", "env": {"A": "`a`", "B": "`b`"}}, markdown=True
        )
        assert result == "Run: `cmd` env: `A`=`a`, `B`=`b`"

    def test_format_confirmation_cwd_empty_string(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "cwd": ""}, markdown=False
        )
        assert result == "Run: echo hi"

    def test_format_confirmation_cwd_empty_string_markdown(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "cwd": ""}, markdown=True
        )
        assert result == "Run: `echo hi`"

    def test_format_confirmation_env_none(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": None}, markdown=False
        )
        assert result == "Run: echo hi"

    def test_format_confirmation_env_none_markdown(self, socket_path: str) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "echo hi", "env": None}, markdown=True
        )
        assert result == "Run: `echo hi`"

    def test_format_confirmation_command_with_backtick_markdown(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation({"command": "echo `hi`"}, markdown=True)
        assert result == "Run: `echo hi`"

    def test_format_confirmation_cwd_with_backtick_markdown(
        self, socket_path: str
    ) -> None:
        tool = Bash(socket_path)
        result = tool.format_confirmation(
            {"command": "ls", "cwd": "/tmp/`test`"}, markdown=True
        )
        assert result == "Run: `ls` in `/tmp/test`"


class TestSafeMd:
    def test_safe_md_normal_string(self) -> None:
        assert Tool._safe_md("hello world") == "hello world"

    def test_safe_md_with_backticks(self) -> None:
        assert Tool._safe_md("echo `hi`") == "echo hi"

    def test_safe_md_empty_string(self) -> None:
        assert Tool._safe_md("") == ""

    def test_safe_md_only_backticks(self) -> None:
        assert Tool._safe_md("````") == ""

    def test_safe_md_no_backticks(self) -> None:
        assert (
            Tool._safe_md("plain text without backticks")
            == "plain text without backticks"
        )
