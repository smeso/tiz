"""Tests for the SubAgents tool."""

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from tiz.chat import Chat
from tiz.tools.subagents import SubAgents


def _make_fake_chat(**attrs: Any) -> Chat:
    """Create a MagicMock Chat instance with common defaults."""
    chat = MagicMock(spec=Chat)
    chat.conv = []
    chat.usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
        "tool_calls": [],
    }
    chat.send_message.return_value = {
        "message": "done",
        "reasoning": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
        "tool_calls": [],
    }
    for k, v in attrs.items():
        setattr(chat, k, v)
    return chat


class TestSubAgentsInit:
    def test_init_with_no_subagents(self) -> None:
        tool = SubAgents({})
        assert tool._subagents == {}

    def test_init_with_valid_subagents(self) -> None:
        def make_chat1() -> Chat:
            return _make_fake_chat()

        def make_chat2() -> Chat:
            return _make_fake_chat()

        subagents: dict[str, dict[str, Any]] = {
            "agent1": {"chat": make_chat1, "description": "First agent"},
            "agent2": {"chat": make_chat2, "description": "Second agent"},
        }
        tool = SubAgents(subagents)
        assert "agent1" in tool._subagents
        assert "agent2" in tool._subagents
        assert tool._subagents["agent1"]["chat_factory"] is make_chat1
        assert tool._subagents["agent1"]["description"] == "First agent"
        assert tool._subagents["agent2"]["chat_factory"] is make_chat2
        assert tool._subagents["agent2"]["description"] == "Second agent"

    def test_init_filters_non_dict_values(self) -> None:
        def make_chat1() -> Chat:
            return _make_fake_chat()

        subagents: dict[str, Any] = {
            "agent1": {"chat": make_chat1, "description": "First"},
            "agent2": "not a dict",
            "agent3": 123,
        }
        tool = SubAgents(subagents)  # type: ignore[arg-type]
        assert "agent1" in tool._subagents
        assert "agent2" not in tool._subagents
        assert "agent3" not in tool._subagents

    def test_init_filters_non_callable_chat(self) -> None:
        subagents: dict[str, Any] = {
            "agent1": {"chat": "not a function", "description": "First"},
            "agent2": {"chat": None, "description": "Second"},
        }
        tool = SubAgents(subagents)
        assert "agent1" not in tool._subagents
        assert "agent2" not in tool._subagents

    def test_init_missing_description(self) -> None:
        def make_chat() -> Chat:
            return _make_fake_chat()

        subagents: dict[str, dict[str, Any]] = {
            "agent1": {"chat": make_chat},
        }
        tool = SubAgents(subagents)
        assert tool._subagents["agent1"]["description"] == ""

    def test_init_missing_chat_key(self) -> None:
        subagents: dict[str, dict[str, str]] = {
            "agent1": {"description": "First"},
        }
        tool = SubAgents(subagents)
        assert "agent1" not in tool._subagents

    def test_init_with_subagent_usage_callback(self) -> None:
        def make_chat() -> Chat:
            return _make_fake_chat()

        def sub_callback(usage: dict[str, Any], conv: list[dict[str, Any]]) -> None:
            pass

        subagents: dict[str, dict[str, Any]] = {
            "agent1": {
                "chat": make_chat,
                "description": "Has callback",
                "usage_callback": sub_callback,
            },
        }
        tool = SubAgents(subagents)
        assert tool._subagents["agent1"]["usage_callback"] is sub_callback

    def test_init_filters_non_callable_usage_callback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)

        def make_chat() -> Chat:
            return _make_fake_chat()

        subagents: dict[str, dict[str, Any]] = {
            "agent1": {
                "chat": make_chat,
                "description": "Bad callback",
                "usage_callback": "not callable",
            },
        }
        tool = SubAgents(subagents)
        assert "usage_callback" not in tool._subagents["agent1"]
        assert len(caplog.records) == 1
        assert "usage_callback" in caplog.text
        assert "not callable" in caplog.text

    def test_init_logs_warning_for_non_dict_config(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        subagents: dict[str, Any] = {
            "agent1": {"chat": lambda: _make_fake_chat(), "description": "Good"},
            "agent2": "not a dict",
            "agent3": 123,
        }
        tool = SubAgents(subagents)  # type: ignore[arg-type]
        assert "agent1" in tool._subagents
        assert "agent2" not in tool._subagents
        assert "agent3" not in tool._subagents
        assert len(caplog.records) == 2
        assert "Skipping sub-agent 'agent2': config is not a dict" in caplog.text
        assert "Skipping sub-agent 'agent3': config is not a dict" in caplog.text

    def test_init_logs_warning_for_non_callable_chat(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        tool = SubAgents(
            {
                "agent1": {"chat": "not callable", "description": "Bad"},
                "agent2": {"chat": None, "description": "Also bad"},
            }
        )
        assert "agent1" not in tool._subagents
        assert "agent2" not in tool._subagents
        assert len(caplog.records) == 2
        assert "Skipping sub-agent 'agent1': 'chat' is not callable" in caplog.text
        assert "Skipping sub-agent 'agent2': 'chat' is not callable" in caplog.text


class TestSubAgentsPrompt:
    def test_prompt_returns_valid_json_no_subagents(self) -> None:
        tool = SubAgents({})
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "SubAgents"
        assert "description" in data
        assert "parameters" in data

    def test_prompt_has_required_fields(self) -> None:
        tool = SubAgents({})
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert "name" in data["parameters"]["required"]
        assert "prompt" in data["parameters"]["required"]
        props = data["parameters"]["properties"]
        assert props["name"]["type"] == "string"
        assert props["prompt"]["type"] == "string"

    def test_prompt_description_mentions_subagents(self) -> None:
        tool = SubAgents({})
        prompt = tool.prompt()
        data = json.loads(prompt)
        description = data["description"]
        assert "sub-agent" in description.lower()

    def test_prompt_lists_available_subagents(self) -> None:
        def make_chat1() -> Chat:
            return _make_fake_chat()

        def make_chat2() -> Chat:
            return _make_fake_chat()

        tool = SubAgents(
            {
                "coder": {"chat": make_chat1, "description": "Writes code"},
                "debugger": {"chat": make_chat2, "description": "Finds bugs"},
            }
        )
        prompt = tool.prompt()
        data = json.loads(prompt)
        description = data["description"]
        assert "coder" in description
        assert "debugger" in description
        assert "Writes code" in description
        assert "Finds bugs" in description
        name_desc = data["parameters"]["properties"]["name"]["description"]
        assert "See the tool description" in name_desc
        assert "available sub-agents" in name_desc.lower()

    def test_prompt_with_no_subagents_indicates_none(self) -> None:
        tool = SubAgents({})
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert "No sub-agents are currently available" in data["description"]
        name_desc = data["parameters"]["properties"]["name"]["description"]
        assert "See the tool description" in name_desc

    def test_prompt_with_subagent_missing_description(self) -> None:
        def make_chat() -> Chat:
            return _make_fake_chat()

        tool = SubAgents({"minion": {"chat": make_chat}})
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert "minion" in data["description"]
        name_desc = data["parameters"]["properties"]["name"]["description"]
        assert "See the tool description" in name_desc


class TestSubAgentsFname:
    def test_fname(self) -> None:
        assert SubAgents.fname() == "SubAgents"


class TestSubAgentsFormatConfirmation:
    def test_format_confirmation_default(self) -> None:
        tool = SubAgents({})
        result = tool.format_confirmation(
            {"name": "coder", "prompt": "write hello world"}, markdown=False
        )
        assert result == "Invoke sub-agent coder with: 'write hello world'"

    def test_format_confirmation_markdown(self) -> None:
        tool = SubAgents({})
        result = tool.format_confirmation(
            {"name": "coder", "prompt": "write hello world"}, markdown=True
        )
        assert result == "Invoke sub-agent `coder` with: `write hello world`"

    def test_format_confirmation_long_prompt_truncated(self) -> None:
        tool = SubAgents({})
        long_prompt = "x" * 200
        result = tool.format_confirmation({"name": "agent", "prompt": long_prompt})
        assert isinstance(result, str)
        assert len(result) < 150
        assert result == "Invoke sub-agent agent with: '" + "x" * 80 + "'"

    def test_format_confirmation_empty_args(self) -> None:
        tool = SubAgents({})
        result = tool.format_confirmation({})
        assert isinstance(result, str)
        assert "Invoke sub-agent" in result

    def test_format_confirmation_prompt_is_none(self) -> None:
        """Regression: 'prompt': None from JSON decoder must not crash."""
        tool = SubAgents({})
        result = tool.format_confirmation({"name": "coder", "prompt": None})
        assert result == "Invoke sub-agent coder with: ''"

    def test_format_confirmation_name_is_none(self) -> None:
        tool = SubAgents({})
        result = tool.format_confirmation({"name": None, "prompt": "hello"})
        assert result == "Invoke sub-agent  with: 'hello'"

    def test_format_confirmation_markdown_backtick_sanitization(self) -> None:
        """Backticks in name/prompt are stripped in markdown mode."""
        tool = SubAgents({})
        result = tool.format_confirmation(
            {"name": "a`b", "prompt": "rm ` -rf /"}, markdown=True
        )
        assert result == "Invoke sub-agent `ab` with: `rm  -rf /`"

    def test_format_confirmation_plain_quotes_prompt(self) -> None:
        """Plain mode wraps prompt in single quotes."""
        tool = SubAgents({})
        result = tool.format_confirmation(
            {"name": "coder", "prompt": "do X with: Y"}, markdown=False
        )
        assert result == "Invoke sub-agent coder with: 'do X with: Y'"


class TestSubAgentsUsageCallback:
    def test_usage_callback_invoked_on_successful_call(self) -> None:
        chat = _make_fake_chat()
        chat.conv = [{"role": "user", "content": "old"}]
        chat.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "prompt_time": 0,
            "completion_time": 0,
            "cost": 0,
            "tool_calls": [],
        }
        chat.send_message.return_value = {
            "message": "done",
            "reasoning": "",
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "prompt_time": 0.1,
            "completion_time": 0.2,
            "cost": 0.001,
            "tool_calls": [],
        }

        def make_chat() -> Chat:
            return chat

        callback_usage: dict[str, Any] | None = None
        callback_conv: list[dict[str, Any]] | None = None

        def my_callback(usage: dict[str, Any], conv: list[dict[str, Any]]) -> None:
            nonlocal callback_usage, callback_conv
            callback_usage = usage
            callback_conv = conv

        tool = SubAgents(
            {
                "worker": {
                    "chat": make_chat,
                    "description": "Worker",
                    "usage_callback": my_callback,
                }
            }
        )
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == "done"
        assert callback_usage is not None
        assert callback_usage["prompt_tokens"] == 10
        assert callback_usage["completion_tokens"] == 0
        assert callback_usage["cached_tokens"] == 0
        assert callback_usage["cache_write_tokens"] == 0
        assert callback_usage["prompt_time"] == 0
        assert callback_usage["completion_time"] == 0
        assert callback_usage["cost"] == 0
        assert callback_usage["tool_calls"] == []
        assert callback_conv is not None
        assert callback_conv == [{"role": "user", "content": "old"}]

    def test_usage_callback_not_invoked_when_no_callback(self) -> None:
        """No callback defined at sub-agent level; none is called."""
        chat = _make_fake_chat()
        chat.send_message.return_value = {"message": "done"}

        def make_chat() -> Chat:
            return chat

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})
        result = tool.run({"name": "worker", "prompt": "task"})
        assert result == "done"

    def test_usage_callback_not_invoked_on_unknown_agent(self) -> None:
        callback_called = False

        def my_callback(_usage: Any, _conv: Any) -> None:
            nonlocal callback_called
            callback_called = True

        tool = SubAgents(
            {
                "worker": {
                    "chat": lambda: _make_fake_chat(),
                    "description": "Worker",
                    "usage_callback": my_callback,
                }
            }
        )
        result = tool.run({"name": "unknown", "prompt": "task"})
        assert "ERROR" in result
        assert not callback_called

    def test_usage_callback_not_invoked_on_missing_name(self) -> None:
        callback_called = False

        def my_callback(_usage: Any, _conv: Any) -> None:
            nonlocal callback_called
            callback_called = True

        tool = SubAgents(
            {
                "worker": {
                    "chat": lambda: _make_fake_chat(),
                    "description": "Worker",
                    "usage_callback": my_callback,
                }
            }
        )
        result = tool.run({"prompt": "task"})
        assert "ERROR" in result
        assert not callback_called

    def test_usage_callback_not_invoked_on_missing_prompt(self) -> None:
        callback_called = False

        def my_callback(_usage: Any, _conv: Any) -> None:
            nonlocal callback_called
            callback_called = True

        tool = SubAgents(
            {
                "worker": {
                    "chat": lambda: _make_fake_chat(),
                    "description": "Worker",
                    "usage_callback": my_callback,
                }
            }
        )
        result = tool.run({"name": "agent"})
        assert "ERROR" in result
        assert not callback_called

    def test_usage_callback_receives_usage_from_send_message(self) -> None:
        """Callback receives the usage and conv from the chat after send_message."""
        chat = _make_fake_chat()
        chat.conv = [{"role": "user", "content": "original"}]
        chat.usage = {"prompt_tokens": 999}

        def send_message_side_effect(_msg: Any) -> dict[str, Any]:
            chat.usage["prompt_tokens"] = 42
            return {"message": "done", "prompt_tokens": 42}

        chat.send_message.side_effect = send_message_side_effect

        def make_chat() -> Chat:
            return chat

        call_order: list[str] = []

        def my_callback(usage: dict[str, Any], _conv: Any) -> None:
            call_order.append("callback")
            assert usage["prompt_tokens"] == 42

        tool = SubAgents(
            {
                "worker": {
                    "chat": make_chat,
                    "description": "Worker",
                    "usage_callback": my_callback,
                }
            }
        )
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == "done"
        assert call_order == ["callback"]

    def test_usage_callback_with_real_chat(self) -> None:
        """Integration test: callback receives usage from real Chat with mocked client."""
        mock_client = MagicMock()
        mock_client.get_context_size.return_value = 8192
        mock_client.chat.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 2,
                "cache_write_tokens": 1,
                "cost": 0.002,
            },
            "timings": {
                "prompt_time": 0.01,
                "completion_time": 0.02,
            },
        }
        mock_client.count_tokens.return_value = 10
        mock_client.preserve_thinking = False

        real_chat = Chat(client=mock_client, sys_prompt="You are a sub-agent.")

        def make_chat() -> Chat:
            return real_chat

        callback_usage: dict[str, Any] | None = None
        callback_conv: list[dict[str, Any]] | None = None

        def my_callback(usage: dict[str, Any], conv: list[dict[str, Any]]) -> None:
            nonlocal callback_usage, callback_conv
            callback_usage = usage
            callback_conv = conv

        tool = SubAgents(
            {
                "helper": {
                    "chat": make_chat,
                    "description": "Helper",
                    "usage_callback": my_callback,
                }
            }
        )

        result = tool.run({"name": "helper", "prompt": "say hello"})

        assert result == "Hello!"
        assert callback_usage is not None
        assert callback_usage["prompt_tokens"] == 10
        assert callback_usage["completion_tokens"] == 5
        assert callback_usage["cached_tokens"] == 2
        assert callback_usage["cache_write_tokens"] == 1
        assert callback_usage["prompt_time"] == 0.01
        assert callback_usage["completion_time"] == 0.02
        assert callback_usage["cost"] == 0.002
        assert callback_conv is not None
        assert len(callback_conv) >= 2
        assert callback_conv[-1]["role"] == "assistant"
        assert callback_conv[-1]["content"] == "Hello!"


class TestSubAgentsRun:
    def test_missing_name(self) -> None:
        tool = SubAgents({})
        result = tool.run({})
        assert result == "ERROR: SubAgents takes a mandatory 'name' argument!"

    def test_empty_name(self) -> None:
        tool = SubAgents({})
        result = tool.run({"name": "", "prompt": "do something"})
        assert result == "ERROR: SubAgents takes a mandatory 'name' argument!"

    def test_name_not_string(self) -> None:
        tool = SubAgents({})
        result = tool.run({"name": 123, "prompt": "do something"})
        assert result == "ERROR: SubAgents takes a mandatory 'name' argument!"

    def test_missing_prompt(self) -> None:
        tool = SubAgents({})
        result = tool.run({"name": "agent"})
        assert result == "ERROR: SubAgents takes a mandatory 'prompt' argument!"

    def test_empty_prompt(self) -> None:
        tool = SubAgents({})
        result = tool.run({"name": "agent", "prompt": ""})
        assert result == "ERROR: SubAgents takes a mandatory 'prompt' argument!"

    def test_prompt_not_string(self) -> None:
        tool = SubAgents({})
        result = tool.run({"name": "agent", "prompt": 123})
        assert result == "ERROR: SubAgents takes a mandatory 'prompt' argument!"

    def test_unknown_subagent_no_agents(self) -> None:
        tool = SubAgents({})
        result = tool.run({"name": "unknown", "prompt": "do something"})
        assert result == "ERROR: unknown sub-agent 'unknown'. No sub-agents available."

    def test_unknown_subagent_with_agents(self) -> None:
        def make_chat() -> Chat:
            return _make_fake_chat()

        tool = SubAgents({"coder": {"chat": make_chat, "description": "Coder"}})
        result = tool.run({"name": "unknown", "prompt": "do something"})
        assert result == ("ERROR: unknown sub-agent 'unknown'. Available: coder")

    def test_unknown_subagent_with_multiple_agents(self) -> None:
        def make_chat1() -> Chat:
            return _make_fake_chat()

        def make_chat2() -> Chat:
            return _make_fake_chat()

        tool = SubAgents(
            {
                "coder": {"chat": make_chat1, "description": "Coder"},
                "debugger": {"chat": make_chat2, "description": "Debugger"},
            }
        )
        result = tool.run({"name": "unknown", "prompt": "do something"})
        assert "coder" in result
        assert "debugger" in result

    def test_successful_subagent_call(self) -> None:
        chat = _make_fake_chat()
        chat.send_message.return_value = {
            "message": "Task completed successfully",
            "reasoning": "",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "prompt_time": 0.1,
            "completion_time": 0.2,
            "cost": 0.001,
            "tool_calls": [],
        }

        def make_chat() -> Chat:
            return chat

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker agent"}})
        result = tool.run({"name": "worker", "prompt": "do the task"})

        assert result == "Task completed successfully"
        chat.send_message.assert_called_once_with("do the task")

    def test_factory_creates_new_chat_each_call(self) -> None:
        """Each invocation gets a fresh Chat from the factory."""
        chat1 = _make_fake_chat()
        chat1.send_message.return_value = {
            "message": "first",
            "reasoning": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "prompt_time": 0,
            "completion_time": 0,
            "cost": 0,
            "tool_calls": [],
        }
        chat2 = _make_fake_chat()
        chat2.send_message.return_value = {
            "message": "second",
            "reasoning": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "prompt_time": 0,
            "completion_time": 0,
            "cost": 0,
            "tool_calls": [],
        }

        factory_call_count = 0

        def make_chat() -> Chat:
            nonlocal factory_call_count
            factory_call_count += 1
            if factory_call_count == 1:
                return chat1
            return chat2

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})

        result1 = tool.run({"name": "worker", "prompt": "first task"})
        assert result1 == "first"
        chat1.send_message.assert_called_once_with("first task")

        result2 = tool.run({"name": "worker", "prompt": "second task"})
        assert result2 == "second"
        chat2.send_message.assert_called_once_with("second task")

        assert factory_call_count == 2

    def test_returns_empty_string_when_no_message(self) -> None:
        chat = _make_fake_chat()
        chat.send_message.return_value = {
            "reasoning": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        def make_chat() -> Chat:
            return chat

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == ""

    def test_returns_empty_string_when_message_is_none(self) -> None:
        chat = _make_fake_chat()
        chat.send_message.return_value = {
            "message": None,
            "reasoning": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        def make_chat() -> Chat:
            return chat

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == ""

    def test_returns_error_on_send_message_exception(self) -> None:
        """Exception from chat.send_message is caught and logged, ERROR returned."""
        chat = _make_fake_chat()
        chat.send_message.side_effect = RuntimeError("connection lost")

        def make_chat() -> Chat:
            return chat

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == "ERROR: sub-agent call failed"

    def test_returns_error_on_send_message_oserror(self) -> None:
        """OSError from chat.send_message is caught and logged, ERROR returned."""
        chat = _make_fake_chat()
        chat.send_message.side_effect = OSError("socket error")

        def make_chat() -> Chat:
            return chat

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == "ERROR: sub-agent call failed"

    def test_usage_callback_not_invoked_on_send_message_error(self) -> None:
        """Callback is not called when send_message raises."""
        callback_called = False

        def my_callback(_usage: Any, _conv: Any) -> None:
            nonlocal callback_called
            callback_called = True

        chat = _make_fake_chat()
        chat.send_message.side_effect = RuntimeError("fail")

        def make_chat() -> Chat:
            return chat

        tool = SubAgents(
            {
                "worker": {
                    "chat": make_chat,
                    "description": "Worker",
                    "usage_callback": my_callback,
                }
            }
        )
        result = tool.run({"name": "worker", "prompt": "task"})

        assert "ERROR" in result
        assert not callback_called

    def test_run_with_real_chat_class(self) -> None:
        """Integration-style test using a real Chat with a mocked client."""
        mock_client = MagicMock()
        mock_client.get_context_size.return_value = 8192
        mock_client.chat.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello from sub-agent!",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0,
            },
            "timings": {
                "prompt_time": 0.01,
                "completion_time": 0.02,
            },
        }
        mock_client.count_tokens.return_value = 10
        mock_client.preserve_thinking = False

        real_chat = Chat(client=mock_client, sys_prompt="You are a sub-agent.")

        def make_chat() -> Chat:
            return real_chat

        tool = SubAgents({"helper": {"chat": make_chat, "description": "Helper"}})

        result = tool.run({"name": "helper", "prompt": "say hello"})

        assert result == "Hello from sub-agent!"


class TestSubAgentsRunFactoryException:
    """Tests for Bug 1: chat_factory() call outside try/except."""

    def test_factory_exception_is_caught(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Factory raising propagates as ERROR."""
        caplog.set_level(logging.ERROR)

        def make_chat() -> Chat:
            raise RuntimeError("factory failed")

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == "ERROR: sub-agent call failed"
        assert len(caplog.records) == 1
        assert "sub-agent call failed" in caplog.text

    def test_factory_returns_none_is_detected(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Factory returning None produces a clear error."""
        caplog.set_level(logging.ERROR)

        def make_chat() -> None:
            return None

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == "ERROR: sub-agent call failed"
        assert len(caplog.records) == 1
        assert "did not return a Chat-like object" in caplog.text

    def test_factory_returns_object_without_send_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Factory returning an object without send_message is caught."""
        caplog.set_level(logging.ERROR)

        def make_chat() -> object:
            return object()

        tool = SubAgents({"worker": {"chat": make_chat, "description": "Worker"}})
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == "ERROR: sub-agent call failed"
        assert len(caplog.records) == 1
        assert "did not return a Chat-like object" in caplog.text


class TestSubAgentsRunCallbackException:
    """Tests for Bug 2: usage_callback exception not caught."""

    def test_callback_exception_does_not_lose_result(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Callback raising does not suppress the successful sub-agent result."""
        caplog.set_level(logging.ERROR)
        chat = _make_fake_chat()
        chat.send_message.return_value = {"message": "success"}

        def make_chat() -> Chat:
            return chat

        def bad_callback(_usage: Any, _conv: Any) -> None:
            raise RuntimeError("callback error")

        tool = SubAgents(
            {
                "worker": {
                    "chat": make_chat,
                    "description": "Worker",
                    "usage_callback": bad_callback,
                }
            }
        )
        result = tool.run({"name": "worker", "prompt": "task"})

        assert result == "success"
        assert len(caplog.records) == 1
        assert "usage_callback failed" in caplog.text


class TestSubAgentsPromptSchema:
    """Tests for prompt schema improvements."""

    def test_uses_self_fname_instead_of_hardcoded(self) -> None:
        """Bug 3: prompt uses self.fname() not hardcoded string."""
        tool = SubAgents({})
        data = json.loads(tool.prompt())
        assert data["name"] == tool.fname()

    def test_has_additional_properties_false(self) -> None:
        """Bug 4: schema has additionalProperties: false."""
        tool = SubAgents({})
        data = json.loads(tool.prompt())
        assert data["parameters"].get("additionalProperties") is False


class TestSubAgentsDescriptionValidation:
    """Tests for Bug 6: description not validated as string."""

    def test_non_string_description_is_ignored(self) -> None:
        """Non-string description values are replaced with empty string."""

        def make_chat() -> Chat:
            return _make_fake_chat()

        subagents: dict[str, dict[str, Any]] = {
            "agent1": {"chat": make_chat, "description": 123},
            "agent2": {"chat": make_chat, "description": True},
            "agent3": {"chat": make_chat, "description": ["list"]},
        }
        tool = SubAgents(subagents)
        assert tool._subagents["agent1"]["description"] == ""
        assert tool._subagents["agent2"]["description"] == ""
        assert tool._subagents["agent3"]["description"] == ""


class TestSubAgentsFormatConfirmationEdgeCases:
    """Additional edge-case tests for format_confirmation."""

    def test_markdown_backtick_sanitization_in_name(self) -> None:
        tool = SubAgents({})
        result = tool.format_confirmation(
            {"name": "a`b`c", "prompt": "task"}, markdown=True
        )
        expected = "Invoke sub-agent `abc` with: `task`"
        assert result == expected

    def test_markdown_backtick_sanitization_in_prompt(self) -> None:
        tool = SubAgents({})
        result = tool.format_confirmation(
            {"name": "agent", "prompt": "rm ` -rf /"}, markdown=True
        )
        expected = "Invoke sub-agent `agent` with: `rm  -rf /`"
        assert result == expected
