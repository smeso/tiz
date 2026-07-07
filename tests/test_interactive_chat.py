from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tiz.audio_inference_clients import WhisperCpp
from tiz.base_task_executor import TaskResources
from tiz.cli import _print_usage
from tiz.interactive_chat import InteractiveChat


def _make_feedback_printer() -> Callable[[dict[str, Any], str | None], None]:
    """Return an update_callback that prints interactive_chat_feedback so capsys can capture it."""

    def callback(msg: dict[str, Any], _subtask_name: str | None = None) -> None:
        if "tiz-internal" in msg:
            feedback = msg["tiz-internal"].get("interactive_chat_feedback")
            if feedback is not None:
                print(feedback)
            usage = msg["tiz-internal"].get("interactive_chat_usage")
            if usage is not None and msg["tiz-internal"].get(
                "interactive_chat_usage_accumulated"
            ):
                _print_usage(usage)

    return callback


def _to_inputs(*items: str | dict[str, str] | None) -> list[dict[str, str] | None]:
    """Convert inputs to dict format expected by InteractiveChat.input_callback."""
    result: list[dict[str, str] | None] = []
    for item in items:
        if item is None:
            result.append(None)
        elif isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str):
            stripped = item.strip()
            if not stripped:
                result.append({"command": "", "message": ""})
            elif stripped.startswith("/"):
                parts = stripped.split(maxsplit=1)
                cmd = parts[0]
                arg = parts[1] if len(parts) > 1 else ""
                result.append({"command": cmd, "message": arg})
            else:
                result.append({"command": "", "message": stripped})
    return result


def _make_minimal_manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.meta.verbosity = 0
    manifest.meta.container_engine = None
    manifest.meta.delete_sandbox_on_exit = False
    manifest.meta.save_full_logs = False
    manifest.meta.save_full_toolcalls = False
    manifest.meta.save_full_usage_details = False
    manifest.meta.summarizer_context_ratio = None

    engine = MagicMock()
    engine.name = "test_engine"
    engine.engine_type = "llamacpp"
    engine.host = "http://127.0.0.1:8080"
    engine.model = "test-model"
    engine.timeout = 5
    engine.message_timeout = None
    engine.verify_ssl = True
    engine.ca_cert = None
    engine.sampling_params = None
    engine.preserve_thinking = False
    engine.api_key = None
    manifest.inference_engines = [engine]

    task = MagicMock()
    task.name = "test_task"
    task.sys_prompt = "You are a test assistant."
    task.sys_prompt_custom = None
    task.worker_image = "ubuntu"
    task.project = None
    task.force_copy_files = []
    task.readonly_sandbox = False
    task.inference_engine = None
    task.dedicated_audio_engine = None
    manifest.tasks = [task]

    return manifest


def _mock_manifest(monkeypatch: Any) -> MagicMock:
    mock_client = MagicMock()
    mock_client.get_context_size.return_value = 4096
    mock_client.count_tokens.return_value = 10

    def _build_client(*_args: Any, **_kwargs: Any) -> MagicMock:
        return mock_client

    monkeypatch.setattr(
        "tiz.base_task_executor.BaseTaskExecutor.build_client", _build_client
    )
    monkeypatch.setattr(
        "tiz.base_task_executor.BaseTaskExecutor._create_task_resources",
        MagicMock(
            return_value=TaskResources(
                tool_instances=[],
                tools_confirmations={},
                sandbox=None,
                manager=None,
                sandbox_name=None,
                action_lock=None,
                conversion_sandbox=None,
            )
        ),
    )
    return mock_client


def test_interactive_chat_init_success(monkeypatch: Any, tmp_path: Path) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    chat = InteractiveChat(manifest=manifest, base_path=tmp_path)
    assert chat.manifest is manifest
    assert chat.client is not None
    assert chat.chat is not None


def test_interactive_chat_init_with_conversion_sandbox(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    mock_instance = MagicMock()
    monkeypatch.setattr(
        "tiz.base_task_executor.BaseTaskExecutor._create_task_resources",
        MagicMock(
            return_value=TaskResources(
                tool_instances=[],
                tools_confirmations={},
                sandbox=None,
                manager=None,
                sandbox_name=None,
                action_lock=None,
                conversion_sandbox=mock_instance,
            )
        ),
    )
    chat = InteractiveChat(manifest=manifest, base_path=tmp_path)
    assert chat._conversion_sandbox is mock_instance
    assert chat.chat is not None


def test_interactive_chat_init_no_engines(tmp_path: Path) -> None:
    manifest = _make_minimal_manifest()
    manifest.inference_engines = []
    with pytest.raises(ValueError, match="No inference engines"):
        InteractiveChat(manifest=manifest, base_path=tmp_path)


def test_interactive_chat_init_no_tasks(tmp_path: Path) -> None:
    manifest = _make_minimal_manifest()
    manifest.tasks = []
    with pytest.raises(ValueError, match="No tasks found"):
        InteractiveChat(manifest=manifest, base_path=tmp_path)


def test_interactive_chat_init_with_task_name(monkeypatch: Any, tmp_path: Path) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    task2 = MagicMock()
    task2.name = "second_task"
    task2.sys_prompt = "You are a test assistant."
    task2.sys_prompt_custom = None
    task2.worker_image = "ubuntu"
    task2.project = None
    task2.force_copy_files = []
    task2.readonly_sandbox = False
    task2.inference_engine = None
    task2.dedicated_audio_engine = None
    manifest.tasks = [manifest.tasks[0], task2]

    chat = InteractiveChat(
        manifest=manifest, base_path=tmp_path, task_name="second_task"
    )
    assert chat.task.name == "second_task"


def test_interactive_chat_init_invalid_task_name(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    with pytest.raises(ValueError, match="nonexistent"):
        InteractiveChat(manifest=manifest, base_path=tmp_path, task_name="nonexistent")


def test_interactive_chat_help_command_without_recording(
    monkeypatch: Any, capsys: Any
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/help", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    captured = capsys.readouterr()
    assert "/help" in captured.out
    assert "/quit" in captured.out
    assert "/exit" in captured.out
    assert "/clear" in captured.out
    assert "/replay" in captured.out
    assert "/usage" in captured.out
    assert "/attach" in captured.out
    assert "/record" not in captured.out


def test_interactive_chat_help_command_with_recording(
    monkeypatch: Any, capsys: Any
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/help", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    captured = capsys.readouterr()
    assert "/help" in captured.out
    assert "/quit" in captured.out
    assert "/exit" in captured.out
    assert "/clear" in captured.out
    assert "/replay" in captured.out
    assert "/usage" in captured.out
    assert "/attach" in captured.out
    assert "/record" in captured.out


def test_interactive_chat_quit_command(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.run()


def test_interactive_chat_exit_command(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/exit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.run()


def test_interactive_chat_usage_command_zero(monkeypatch: Any, capsys: Any) -> None:
    """Test /usage command with zero usage."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/usage", "/quit")

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    # Accumulate zero usage
    chat._accumulate_usage(
        "test_task",
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "prompt_time": 0,
            "completion_time": 0,
            "cost": 0,
            "tool_calls": [],
        },
    )
    chat.run()

    captured = capsys.readouterr()
    assert "Usage: input:\t0 (0.0 tk/s)" in captured.out
    assert "      output:\t0 (0.0 tk/s)" in captured.out
    assert "      cached:\t0" in captured.out
    assert "      cwrite:\t0" in captured.out
    assert "Credits spent:\t0.0000000000 $" in captured.out


def test_interactive_chat_usage_command_with_data(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test /usage command after some usage has accumulated."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/usage", "/quit")

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    # Accumulate usage data
    chat._accumulate_usage(
        "test_task",
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 10,
            "cache_write_tokens": 5,
            "prompt_time": 2.5,
            "completion_time": 1.2,
            "cost": 0.001234,
            "tool_calls": [("bash", {"cmd": "echo hi"})],
        },
    )
    chat.run()

    captured = capsys.readouterr()
    assert "Usage: input:\t90 (36.0 tk/s)" in captured.out
    assert "      output:\t50 (41.67 tk/s)" in captured.out
    assert "      cached:\t10" in captured.out
    assert "      cwrite:\t5" in captured.out
    assert "Credits spent:\t0.0012340000 $" in captured.out


def test_interactive_chat_usage_command_no_callback(
    monkeypatch: Any,
) -> None:
    """Test /usage command when no update_callback is set (covers branch where _show_usage does nothing)."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/usage", "/quit")

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat._accumulate_usage(
        "test_task",
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 10,
            "cache_write_tokens": 5,
            "prompt_time": 2.5,
            "completion_time": 1.2,
            "cost": 0.001234,
            "tool_calls": [],
        },
    )
    chat.run()


def test_interactive_chat_clear_command(monkeypatch: Any, capsys: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/clear", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    original_chat = chat.chat
    chat.run()
    assert chat.chat is not original_chat
    captured = capsys.readouterr()
    assert "Conversation cleared" in captured.out


def test_interactive_chat_replay_command(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/replay", "/quit")

    mock_replay_result = MagicMock()
    mock_replay_result.get.side_effect = lambda key, default=None: (
        "replayed message" if key == "message" else default
    )
    chat_obj = MagicMock()
    chat_obj.replay.return_value = mock_replay_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.replay.assert_called_once_with(timeout=None)


def test_interactive_chat_replay_no_message(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/replay", "/quit")

    mock_replay_result = MagicMock()
    mock_replay_result.get.side_effect = lambda key, default=None: (
        "" if key == "message" else default
    )
    chat_obj = MagicMock()
    chat_obj.replay.return_value = mock_replay_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.replay.assert_called_once_with(timeout=None)


def test_interactive_chat_attach_success(monkeypatch: Any, capsys: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/attach /tmp/test.txt", "/quit")

    chat_obj = MagicMock()
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.append_file.assert_called_once_with("/tmp/test.txt")
    captured = capsys.readouterr()
    assert "Attached: /tmp/test.txt" in captured.out


def test_interactive_chat_attach_no_arg(monkeypatch: Any, capsys: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/attach", "/quit")

    chat_obj = MagicMock()
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()
    assert not chat_obj.append_file.called
    captured = capsys.readouterr()
    assert "Usage: /attach <file>" in captured.out


def test_interactive_chat_attach_error(monkeypatch: Any, capsys: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/attach /bad/file", "/quit")

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("file not found")

    chat_obj = MagicMock()
    chat_obj.append_file.side_effect = _raise
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()
    captured = capsys.readouterr()
    assert "Error: file not found" in captured.out


def test_interactive_chat_unknown_command(monkeypatch: Any, capsys: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/foobar", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()
    captured = capsys.readouterr()
    assert "Unknown command: /foobar" in captured.out


def test_interactive_chat_record_unknown_when_disabled(
    monkeypatch: Any, capsys: Any
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()
    captured = capsys.readouterr()
    assert "Unknown command: /record" in captured.out


def test_interactive_chat_normal_message(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("hello world", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result
    chat_obj.usage = {}

    mock_callback = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=mock_callback,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello world", timeout=None)
    # Verify _show_usage was called after send_message
    usage_calls = [
        call
        for call in mock_callback.call_args_list
        if "interactive_chat_usage" in call[0][0].get("tiz-internal", {})
    ]
    assert len(usage_calls) >= 1


def test_interactive_chat_normal_message_empty_reply(
    monkeypatch: Any,
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello", timeout=None)


def test_interactive_chat_empty_input(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("", "  ", "/quit")

    chat_obj = MagicMock()
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.send_message.called


def test_interactive_chat_dict_input_message(monkeypatch: Any) -> None:
    """Test that dict input with a message sends the message."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs: list[dict[str, str] | None] = [
        {"message": "hello world", "command": ""},
        None,
    ]

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello world", timeout=None)


def test_interactive_chat_dict_input_command(monkeypatch: Any, capsys: Any) -> None:
    """Test that dict input with a command executes the command."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs: list[dict[str, str] | None] = [
        {"message": "", "command": "/help"},
        {"message": "", "command": "/quit"},
    ]

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    captured = capsys.readouterr()
    assert "/help" in captured.out


def test_interactive_chat_dict_input_empty(monkeypatch: Any) -> None:
    """Test that dict input with empty message and command is skipped."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs: list[dict[str, str] | None] = [
        {"message": "", "command": ""},
        {"message": "", "command": "/quit"},
    ]

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.send_message.called


def test_interactive_chat_input_callback_none(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs: list[dict[str, str] | None] = [None]
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.run()


def test_interactive_chat_keyboard_interrupt_during_send(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("msg1", "msg2", "/quit")

    call_count = 0
    chat_obj = MagicMock()

    def _send_side_effect(*_args: Any, **_kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise KeyboardInterrupt()
        mock_result = MagicMock()
        mock_result.get.side_effect = lambda key, default=None: (
            "second reply" if key == "message" else default
        )
        return mock_result

    chat_obj.send_message.side_effect = _send_side_effect

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    assert chat_obj.send_message.call_count == 2


def test_interactive_chat_keyboard_interrupt_during_input(
    monkeypatch: Any,
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    call_count = 0

    def _input() -> dict[str, str] | None:
        nonlocal call_count
        call_count += 1
        raise KeyboardInterrupt()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=_input,
    )
    chat.run()

    assert call_count == 1


def test_interactive_chat_no_exit_on_kbdint_init(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Test that no_exit_on_kbdint parameter is properly stored."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    chat = InteractiveChat(
        manifest=manifest, base_path=tmp_path, no_exit_on_kbdint=True
    )
    assert chat.no_exit_on_kbdint is True


def test_interactive_chat_no_exit_on_kbdint_default(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Test that no_exit_on_kbdint defaults to False."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    chat = InteractiveChat(manifest=manifest, base_path=tmp_path)
    assert chat.no_exit_on_kbdint is False


def test_interactive_chat_no_exit_on_kbdint_during_input(
    monkeypatch: Any,
) -> None:
    """Test that with no_exit_on_kbdint=True, KeyboardInterrupt during
    input callback does not break the loop but continues."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/quit")
    call_count = 0

    def _input() -> dict[str, str] | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise KeyboardInterrupt()
        return inputs.pop(0)

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=_input,
        no_exit_on_kbdint=True,
    )
    chat.run()

    assert call_count == 2


def test_interactive_chat_no_exit_on_kbdint_during_send(
    monkeypatch: Any,
) -> None:
    """Test that with no_exit_on_kbdint=True, KeyboardInterrupt during
    send_message does not break the loop but continues to next input."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("msg1", "msg2", "/quit")

    send_count = 0

    def _send(*_args: Any, **_kwargs: Any) -> MagicMock:
        nonlocal send_count
        send_count += 1
        if send_count == 1:
            raise KeyboardInterrupt()
        mock_result = MagicMock()
        mock_result.get.side_effect = lambda key, default=None: (
            "second reply" if key == "message" else default
        )
        return mock_result

    chat_obj = MagicMock()
    chat_obj.send_message.side_effect = _send

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        no_exit_on_kbdint=True,
    )
    chat.chat = chat_obj
    chat.run()

    assert send_count == 2


def test_interactive_chat_with_update_callback(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    callback = MagicMock()
    inputs = _to_inputs("/quit")
    chat = InteractiveChat(
        manifest=manifest,
        update_callback=callback,
        input_callback=lambda: inputs.pop(0),
    )
    chat.run()
    assert chat._update_callback is callback


def test_interactive_chat_clear_creates_new_chat(monkeypatch: Any) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("hello", "/clear", "/quit")

    chat_obj = MagicMock()
    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "reply" if key == "message" else default
    )
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    assert chat.chat is not chat_obj


def test_interactive_chat_replay_keyboard_interrupt(
    monkeypatch: Any,
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/replay", "/quit")

    chat_obj = MagicMock()
    chat_obj.replay.side_effect = KeyboardInterrupt()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.replay.assert_called_once_with(timeout=None)


def test_task_usage_empty(monkeypatch: Any, tmp_path: Path) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    chat = InteractiveChat(manifest=manifest, base_path=tmp_path)
    assert chat.task_usage == {}


def test_task_usage_after_accumulate(monkeypatch: Any, tmp_path: Path) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_logs = False
    chat = InteractiveChat(manifest=manifest, base_path=tmp_path)
    chat._accumulate_usage("test_task", {"prompt_tokens": 10})
    usage = chat.task_usage
    assert "test_task" in usage
    assert usage["test_task"]["prompt_tokens"] == 10


def test_interactive_chat_finally_with_manager_and_sandbox(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/quit")

    mock_manager = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.project_dir = str(tmp_path)

    monkeypatch.setattr(
        "tiz.base_task_executor.BaseTaskExecutor._create_task_resources",
        MagicMock(
            return_value=TaskResources(
                tool_instances=[],
                tools_confirmations={},
                sandbox=mock_sandbox,
                manager=mock_manager,
                sandbox_name="test-sandbox",
                action_lock=None,
                conversion_sandbox=None,
            )
        ),
    )

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat.run()

    mock_manager.kill_all_containers.assert_called_once_with("test-sandbox")
    mock_sandbox.validate_git_project_dir.assert_called_once_with()


def test_interactive_chat_finally_sync_to_original_auto_rebase_exception(
    monkeypatch: Any, tmp_path: Path, caplog: Any
) -> None:
    """Test that an exception in sync_to_original_auto_rebase is caught and logged."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/quit")

    mock_manager = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.project_dir = str(tmp_path)
    mock_sandbox.sync_to_original_auto_rebase.side_effect = RuntimeError("sync failed")

    monkeypatch.setattr(
        "tiz.base_task_executor.BaseTaskExecutor._create_task_resources",
        MagicMock(
            return_value=TaskResources(
                tool_instances=[],
                tools_confirmations={},
                sandbox=mock_sandbox,
                manager=mock_manager,
                sandbox_name="test-sandbox",
                action_lock=None,
                conversion_sandbox=None,
            )
        ),
    )

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat.run()

    assert "Failed to sync sandbox to original" in caplog.text


def test_interactive_chat_finally_save_toolcalls_log_outer_exception(
    monkeypatch: Any, tmp_path: Path, caplog: Any
) -> None:
    """Test that an exception in _save_toolcalls_log is caught by the outer except."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_toolcalls = True
    inputs = _to_inputs("/quit")

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )

    def _raise_exc(_task_name: str) -> None:
        raise RuntimeError("unexpected error")

    monkeypatch.setattr(chat, "_save_toolcalls_log", _raise_exc)
    chat.run()

    assert "Failed to save tool calls log" in caplog.text


def test_interactive_chat_finally_save_chat_log(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_logs = True
    inputs = _to_inputs("hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result
    chat_obj.conv = []

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    log_dir = tmp_path / "logs" / "conversations"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) == 1
    assert "test_task-test_engine" in log_files[0].name
    assert log_files[0].suffix == ".json"


def test_interactive_chat_finally_save_chat_log_exception(
    monkeypatch: Any, tmp_path: Path, caplog: Any
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_logs = True
    inputs = _to_inputs("/quit")

    chat_obj = MagicMock()
    chat_obj.save.side_effect = RuntimeError("save failed")

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    assert "Failed to save chat log" in caplog.text


def test_interactive_chat_finally_save_toolcalls_log(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_toolcalls = True
    inputs = _to_inputs("hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat._accumulate_usage("test_task", {"tool_calls": [{"name": "test_tool"}]})
    chat.chat = chat_obj
    chat.run()

    log_dir = tmp_path / "logs" / "tools"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) == 1
    assert log_files[0].name.endswith(".json")
    assert "test_task-test_engine" in log_files[0].name
    contents = json.loads(log_files[0].read_text())
    assert contents == [{"name": "test_tool"}]


def test_interactive_chat_finally_save_toolcalls_log_exception(
    monkeypatch: Any, tmp_path: Path, caplog: Any
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_toolcalls = True
    inputs = _to_inputs("hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat._accumulate_usage("test_task", {"tool_calls": [{"name": "test_tool"}]})

    def _failing_open(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("write failed")

    monkeypatch.setattr("tiz.base_task_executor.Path.open", _failing_open)

    chat.chat = chat_obj
    chat.run()

    assert "Failed to write toolcalls log for" in caplog.text


def test_interactive_chat_finally_save_full_usage(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_usage_details = True
    inputs = _to_inputs("hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat._accumulate_usage("test_task", {"prompt_tokens": 5})
    chat.chat = chat_obj
    chat.run()

    log_dir = tmp_path / "logs" / "usage"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) == 1
    assert log_files[0].suffix == ".log"
    assert "test_engine" in log_files[0].name
    assert "test_task" in log_files[0].name
    contents = json.loads(log_files[0].read_text())
    assert contents.get("prompt_tokens") == 5
    assert "tool_calls" not in contents


def test_interactive_chat_finally_save_full_usage_exception(
    monkeypatch: Any, tmp_path: Path, caplog: Any
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_usage_details = True
    inputs = _to_inputs("hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )
    chat._accumulate_usage("test_task", {"prompt_tokens": 5})

    def _failing_open(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("usage write failed")

    monkeypatch.setattr("tiz.base_task_executor.Path.open", _failing_open)

    chat.chat = chat_obj
    chat.run()

    assert "Failed to write usage log" in caplog.text


def test_interactive_chat_finally_save_full_usage_outer_exception(
    monkeypatch: Any, tmp_path: Path, caplog: Any
) -> None:
    """Test the outer except handler for _save_full_usage by patching the method directly."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.meta.save_full_usage_details = True
    inputs = _to_inputs("/quit")

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
    )

    def _raise_exc() -> None:
        raise RuntimeError("unexpected error")

    monkeypatch.setattr(chat, "_save_full_usage", _raise_exc)
    chat.run()

    assert "Failed to save full usage" in caplog.text


def test_interactive_chat_init_with_context(monkeypatch: Any, tmp_path: Path) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    ctx = {"extra_var": "hello"}
    chat = InteractiveChat(manifest=manifest, base_path=tmp_path, context=ctx)
    assert chat._context == ctx
    assert chat.manifest is manifest
    assert chat.client is not None
    assert chat.chat is not None


def test_interactive_chat_create_chat_failure_kills_container(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()

    mock_manager = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.project_dir = str(tmp_path)
    monkeypatch.setattr(
        "tiz.base_task_executor.BaseTaskExecutor._create_task_resources",
        MagicMock(
            return_value=TaskResources(
                tool_instances=[],
                tools_confirmations={},
                sandbox=mock_sandbox,
                manager=mock_manager,
                sandbox_name="test-sandbox",
                action_lock=None,
                conversion_sandbox=None,
            )
        ),
    )
    monkeypatch.setattr(
        "tiz.interactive_chat.InteractiveChat._create_chat",
        MagicMock(side_effect=RuntimeError("chat creation failed")),
    )

    with pytest.raises(RuntimeError, match="chat creation failed"):
        InteractiveChat(manifest=manifest, base_path=tmp_path)
    mock_manager.kill_all_containers.assert_called_once_with("test-sandbox")


def test_interactive_chat_create_chat_failure_no_manager(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()

    monkeypatch.setattr(
        "tiz.interactive_chat.InteractiveChat._create_chat",
        MagicMock(side_effect=RuntimeError("chat creation failed")),
    )

    with pytest.raises(RuntimeError, match="chat creation failed"):
        InteractiveChat(manifest=manifest, base_path=tmp_path)


def test_get_audio_client_no_dedicated_engine(monkeypatch: Any, tmp_path: Path) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    chat = InteractiveChat(manifest=manifest, base_path=tmp_path)
    assert chat.audio_inference_client is None


def test_get_audio_client_dedicated_engine_success(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    audio_spec = MagicMock()
    audio_spec.name = "my_audio"
    audio_spec.engine_type = "whispercpp"
    audio_spec.host = "http://127.0.0.1:8080"
    audio_spec.timeout = 10
    audio_spec.inference_timeout = None
    audio_spec.verify_ssl = True
    audio_spec.ca_cert = None
    audio_spec.sampling_params = None
    audio_spec.language = "en"
    audio_spec.prompt = None
    manifest.audio_inference_engines = [audio_spec]
    manifest.tasks[0].dedicated_audio_engine = "my_audio"

    chat = InteractiveChat(manifest=manifest, base_path=tmp_path)
    assert chat.audio_inference_client is not None

    # Verify it was passed to Chat
    assert chat.chat.audio_inference_client is chat.audio_inference_client


def test_get_audio_client_dedicated_engine_not_found(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.tasks[0].dedicated_audio_engine = "nonexistent_audio"
    with pytest.raises(ValueError, match="nonexistent_audio"):
        InteractiveChat(manifest=manifest, base_path=tmp_path)


def test_get_audio_client_no_engines_list(monkeypatch: Any, tmp_path: Path) -> None:
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    manifest.audio_inference_engines = []
    manifest.tasks[0].dedicated_audio_engine = "my_audio"
    with pytest.raises(ValueError, match="my_audio"):
        InteractiveChat(manifest=manifest, base_path=tmp_path)


def test_get_audio_client_engine_not_found_with_nonmatching(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Audio engine exists but has a different name from what task requests."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    audio_spec = MagicMock()
    audio_spec.name = "other_audio"
    audio_spec.engine_type = "whispercpp"
    audio_spec.host = "http://127.0.0.1:8080"
    audio_spec.timeout = 10
    audio_spec.inference_timeout = None
    audio_spec.verify_ssl = True
    audio_spec.ca_cert = None
    audio_spec.sampling_params = None
    audio_spec.language = None
    audio_spec.prompt = None
    manifest.audio_inference_engines = [audio_spec]
    manifest.tasks[0].dedicated_audio_engine = "wanted_audio"
    with pytest.raises(ValueError, match="wanted_audio"):
        InteractiveChat(manifest=manifest, base_path=tmp_path)


def test_build_audio_client_whispercpp() -> None:
    spec = MagicMock()
    spec.engine_type = "whispercpp"
    spec.host = "http://127.0.0.1:8080"
    spec.timeout = 5
    spec.inference_timeout = None
    spec.verify_ssl = True
    spec.ca_cert = None
    spec.sampling_params = None
    spec.language = None
    spec.prompt = None

    client = InteractiveChat._build_audio_client(spec)
    assert isinstance(client, WhisperCpp)
    assert client.host == "http://127.0.0.1:8080"


def test_build_audio_client_unknown_type() -> None:
    spec = MagicMock()
    spec.engine_type = "unknown"
    with pytest.raises(ValueError, match="Unknown audio inference engine type"):
        InteractiveChat._build_audio_client(spec)


def test_interactive_chat_record_command_success(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /record command flow: record, attach, replay."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    record_path = tmp_path / "test_recording.wav"

    mock_file = MagicMock()
    mock_file.name = str(record_path)
    mock_file.__enter__.return_value = mock_file
    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(return_value=mock_file),
    )
    monkeypatch.setattr("tiz.interactive_chat.record_audio", MagicMock())

    mock_replay_result = MagicMock()
    mock_replay_result.get.side_effect = lambda key, default=None: (
        "replayed" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.replay.return_value = mock_replay_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    filepath = str(record_path)
    chat_obj.append_file.assert_called_once_with(filepath)
    chat_obj.replay.assert_called_once_with(timeout=None)

    captured = capsys.readouterr()
    assert "Attached recording" in captured.out


def test_interactive_chat_record_command_record_error(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /record command when recording raises an error."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    record_path = tmp_path / "test_recording.wav"

    mock_file = MagicMock()
    mock_file.name = str(record_path)
    mock_file.__enter__.return_value = mock_file
    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(return_value=mock_file),
    )
    monkeypatch.setattr(
        "tiz.interactive_chat.record_audio",
        MagicMock(side_effect=RuntimeError("no mic")),
    )

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Error: no mic" in captured.out

    assert not chat_obj.append_file.called
    assert not chat_obj.replay.called


def test_interactive_chat_record_command_replay_keyboard_interrupt(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Test /record command when replay is interrupted."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    record_path = tmp_path / "test_recording.wav"

    mock_file = MagicMock()
    mock_file.name = str(record_path)
    mock_file.__enter__.return_value = mock_file
    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(return_value=mock_file),
    )
    monkeypatch.setattr("tiz.interactive_chat.record_audio", MagicMock())

    chat_obj = MagicMock()
    chat_obj.replay.side_effect = KeyboardInterrupt()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.append_file.assert_called_once_with(str(record_path))


def test_interactive_chat_record_tempfile_error(monkeypatch: Any, capsys: Any) -> None:
    """Test /record command when tempfile creation raises an error."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(side_effect=OSError("cannot create temp file")),
    )

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Error: cannot create temp file" in captured.out


def test_interactive_chat_sync_to_with_sandbox(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /sync-to command with sandbox available."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/sync-to", "/quit")

    mock_manager = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.project_dir = str(tmp_path)

    monkeypatch.setattr(
        "tiz.base_task_executor.BaseTaskExecutor._create_task_resources",
        MagicMock(
            return_value=TaskResources(
                tool_instances=[],
                tools_confirmations={},
                sandbox=mock_sandbox,
                manager=mock_manager,
                sandbox_name="test-sandbox",
                action_lock=None,
                conversion_sandbox=None,
            )
        ),
    )

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    mock_sandbox.sync_to_original.assert_called_once_with()
    captured = capsys.readouterr()
    assert "Synced sandbox changes to original project." in captured.out


def test_interactive_chat_sync_to_no_sandbox(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /sync-to command when sandbox is None."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/sync-to", "/quit")

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    captured = capsys.readouterr()
    assert "No sandbox to sync from." in captured.out


def test_interactive_chat_sync_from_with_sandbox(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /sync-from command with sandbox available."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/sync-from", "/quit")

    mock_manager = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.project_dir = str(tmp_path)

    monkeypatch.setattr(
        "tiz.base_task_executor.BaseTaskExecutor._create_task_resources",
        MagicMock(
            return_value=TaskResources(
                tool_instances=[],
                tools_confirmations={},
                sandbox=mock_sandbox,
                manager=mock_manager,
                sandbox_name="test-sandbox",
                action_lock=None,
                conversion_sandbox=None,
            )
        ),
    )

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    mock_sandbox.sync_from_original.assert_called_once_with()
    captured = capsys.readouterr()
    assert "Synced original project changes to sandbox." in captured.out


def test_interactive_chat_sync_from_no_sandbox(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /sync-from command when sandbox is None."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/sync-from", "/quit")

    chat = InteractiveChat(
        manifest=manifest,
        base_path=tmp_path,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    captured = capsys.readouterr()
    assert "No sandbox to sync to." in captured.out


def test_interactive_chat_required_prefix_matching(
    monkeypatch: Any,
) -> None:
    """Test that messages with the required prefix are processed normally."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("bot: hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="bot:",
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello", timeout=None)


def test_interactive_chat_required_prefix_matching_case_insensitive(
    monkeypatch: Any,
) -> None:
    """Test that case-insensitive matching works."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("BOT: hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="bot:",
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello", timeout=None)


def test_interactive_chat_required_prefix_matching_stripped(
    monkeypatch: Any,
) -> None:
    """Test that leading/trailing whitespace on input and prefix is handled."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("  bot: hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="  bot:  ",
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello", timeout=None)


def test_interactive_chat_required_prefix_not_matching(
    monkeypatch: Any, caplog: Any
) -> None:
    """Test that messages without the prefix are logged and skipped."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("hello", "/quit")

    chat_obj = MagicMock()

    caplog.set_level(logging.INFO)

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="bot:",
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.send_message.called
    assert "Input ignored: does not start with required prefix 'bot:'" in caplog.text


def test_interactive_chat_required_prefix_not_matching_multiple(
    monkeypatch: Any, caplog: Any
) -> None:
    """Test multiple non-matching inputs are all skipped."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("hello", "world", "test", "/quit")

    chat_obj = MagicMock()

    caplog.set_level(logging.INFO)

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="bot:",
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.send_message.called
    assert caplog.text.count("Input ignored:") == 3


def test_interactive_chat_required_prefix_empty_after_strip(
    monkeypatch: Any,
) -> None:
    """Test that input that is only the prefix is treated as empty."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("bot:", "/quit")

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="bot:",
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.send_message.called


def test_interactive_chat_required_prefix_with_command(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test that commands still work when prefixed."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs: list[dict[str, str] | None] = [
        {"command": "", "message": "bot: hello"},
        {"command": "/help", "message": ""},
        {"command": "/quit", "message": ""},
    ]

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        required_prefix="bot:",
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "/help" in captured.out


def test_interactive_chat_required_prefix_with_clear_command(
    monkeypatch: Any,
) -> None:
    """Test that /clear command works with prefix."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs: list[dict[str, str] | None] = [
        {"command": "", "message": "bot: hello"},
        {"command": "/clear"},
        {"command": "/quit"},
    ]

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="bot:",
    )
    original_chat = chat.chat
    chat.chat = chat_obj
    chat.run()
    assert chat.chat is not original_chat


def test_interactive_chat_required_prefix_dict_input(
    monkeypatch: Any,
) -> None:
    """Test that dict input still goes through prefix check on the message."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs: list[dict[str, str] | None] = [
        {"message": "bot: hello", "command": ""},
        {"message": "", "command": "/quit"},
    ]

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="bot:",
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello", timeout=None)


def test_interactive_chat_required_prefix_not_set(
    monkeypatch: Any,
) -> None:
    """Test that when required_prefix is None, all strings pass through."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello", timeout=None)


def test_interactive_chat_required_prefix_empty_string(
    monkeypatch: Any,
) -> None:
    """Test that empty string prefix is treated as no prefix."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("hello", "/quit")

    mock_result = MagicMock()
    mock_result.get.side_effect = lambda key, default=None: (
        "assistant reply" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.send_message.return_value = mock_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        required_prefix="",
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.send_message.assert_called_once_with("hello", timeout=None)


def test_interactive_chat_save_command_success(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /save command with a valid file path."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    save_path = str(tmp_path / "conversation.json")
    inputs = _to_inputs(f"/save {save_path}", "/quit")

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.save.assert_called_once_with(save_path)
    captured = capsys.readouterr()
    assert "Saved conversation to:" in captured.out
    assert save_path in captured.out


def test_interactive_chat_save_command_no_arg(monkeypatch: Any, capsys: Any) -> None:
    """Test /save command without a file argument."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/save", "/quit")

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.save.called
    captured = capsys.readouterr()
    assert "Usage: /save <file>" in captured.out


def test_interactive_chat_save_command_error(monkeypatch: Any, capsys: Any) -> None:
    """Test /save command when save raises an error."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/save /bad/path/file.json", "/quit")

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("permission denied")

    chat_obj = MagicMock()
    chat_obj.save.side_effect = _raise

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Error: permission denied" in captured.out


def test_interactive_chat_load_command_success(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /load command with a valid file path."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    load_path = str(tmp_path / "conversation.json")
    inputs = _to_inputs(f"/load {load_path}", "/quit")

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.load.assert_called_once_with(load_path)
    captured = capsys.readouterr()
    assert "Loaded conversation from:" in captured.out
    assert load_path in captured.out


def test_interactive_chat_load_command_no_arg(monkeypatch: Any, capsys: Any) -> None:
    """Test /load command without a file argument."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/load", "/quit")

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.load.called
    captured = capsys.readouterr()
    assert "Usage: /load <file>" in captured.out


def test_interactive_chat_load_command_error(monkeypatch: Any, capsys: Any) -> None:
    """Test /load command when load raises an error."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/load /nonexistent/file.json", "/quit")

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("file not found")

    chat_obj = MagicMock()
    chat_obj.load.side_effect = _raise

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Error: file not found" in captured.out


def test_interactive_chat_help_shows_load_and_save(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test that /help shows /load and /save commands."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/help", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    captured = capsys.readouterr()
    assert "/load" in captured.out
    assert "/save" in captured.out


def test_interactive_chat_in_band_attach_success(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /attach with in_band_files=True reads file content and passes inline."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    encoded = base64.b64encode(b"hello world").decode()
    inputs = [
        {"command": "/attach", "message": str(test_file), "contents": encoded},
        {"command": "/quit", "message": ""},
    ]

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.append_file.assert_called_once()
    call_args = chat_obj.append_file.call_args
    assert call_args[0][0] == str(test_file)
    assert call_args[1]["data"] == b"hello world"
    captured = capsys.readouterr()
    assert "Attached:" in captured.out


def test_interactive_chat_in_band_attach_read_error(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test /attach with in_band_files=True when reading the file fails."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = [
        {"command": "/attach", "message": "/nonexistent/file.txt", "contents": ""},
        {"command": "/quit", "message": ""},
    ]

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    # With in_band_files=True, base64-decode of empty string is b"",
    # so append_file is called with empty data
    chat_obj.append_file.assert_called_once()
    captured = capsys.readouterr()
    assert "Attached:" in captured.out


def test_interactive_chat_in_band_attach_no_arg(monkeypatch: Any, capsys: Any) -> None:
    """Test /attach with in_band_files=True and no argument."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/attach", "/quit")

    chat_obj = MagicMock()
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()
    assert not chat_obj.append_file.called
    captured = capsys.readouterr()
    assert "Usage: /attach <file>" in captured.out


def test_interactive_chat_in_band_save_success(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /save with in_band_files=True."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    save_path = str(tmp_path / "conv.json")
    inputs = _to_inputs(f"/save {save_path}", "/quit")

    conv_data = base64.b64encode(b'[{"role": "user", "content": "hi"}]').decode()

    chat_obj = MagicMock()
    chat_obj.save.return_value = conv_data

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.save.assert_called_once_with()
    written = Path(save_path).read_text()
    # With in_band_files and without "contents" key, save() returns base64 data
    # which is written to the file directly
    decoded = base64.b64decode(written).decode()
    assert json.loads(decoded) == [{"role": "user", "content": "hi"}]
    captured = capsys.readouterr()
    assert "Saved conversation to:" in captured.out


def test_interactive_chat_in_band_save_no_arg(monkeypatch: Any, capsys: Any) -> None:
    """Test /save with in_band_files=True and no argument."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/save", "/quit")

    chat_obj = MagicMock()
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.save.called
    captured = capsys.readouterr()
    assert "Usage: /save <file>" in captured.out


def test_interactive_chat_in_band_save_error(monkeypatch: Any, capsys: Any) -> None:
    """Test /save with in_band_files=True when writing the file fails."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/save /bad/path/file.json", "/quit")

    chat_obj = MagicMock()
    chat_obj.save.return_value = "c29tZSBkYXRh"

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Error" in captured.out


def test_interactive_chat_in_band_load_success(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /load with in_band_files=True."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    load_path = tmp_path / "conv.json"
    load_path.write_text('[{"role": "user", "content": "hello"}]')
    encoded = base64.b64encode(b'[{"role": "user", "content": "hello"}]').decode()
    inputs = [
        {"command": "/load", "message": str(load_path), "contents": encoded},
        {"command": "/quit", "message": ""},
    ]

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.load.assert_called_once()
    call_args = chat_obj.load.call_args
    assert call_args[1]["data"] is not None
    decoded = json.loads(call_args[1]["data"])
    assert decoded == [{"role": "user", "content": "hello"}]
    captured = capsys.readouterr()
    assert "Loaded conversation from:" in captured.out


def test_interactive_chat_in_band_load_no_arg(monkeypatch: Any, capsys: Any) -> None:
    """Test /load with in_band_files=True and no argument."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/load", "/quit")

    chat_obj = MagicMock()
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    assert not chat_obj.load.called
    captured = capsys.readouterr()
    assert "Usage: /load <file>" in captured.out


def test_interactive_chat_in_band_load_error(monkeypatch: Any, capsys: Any) -> None:
    """Test /load with in_band_files=True when reading the file fails."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = [
        {"command": "/load", "message": "/nonexistent/file.json", "contents": ""},
        {"command": "/quit", "message": ""},
    ]

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    # With in_band_files=True and contents="", base64-decode gives b"",
    # so load is called with empty data. Since chat_obj is a mock, it succeeds.
    chat_obj.load.assert_called_once()
    captured = capsys.readouterr()
    assert "Loaded conversation from:" in captured.out


def test_interactive_chat_enable_help_false_show_help_suppressed(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test that when enable_help=False, _show_help does nothing."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/help", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        enable_help=False,
    )
    chat.run()

    captured = capsys.readouterr()
    assert "/help" not in captured.out
    assert "/quit" not in captured.out
    assert "Available commands:" not in captured.out


def test_interactive_chat_enable_help_false_usage_feedback_suppressed(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test that when enable_help=False, 'Usage:' messages are suppressed."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/attach", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        enable_help=False,
    )
    chat.run()

    captured = capsys.readouterr()
    assert "Usage:" not in captured.out
    assert "Error:" not in captured.out  # other messages still work


def test_interactive_chat_initial_help_suppressed_when_disabled(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test that when enable_help=False, the initial help at run start is suppressed."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        enable_help=False,
    )
    chat.run()

    captured = capsys.readouterr()
    # The initial blank line feedback should still be there
    # but no "Available commands" should appear
    assert "Available commands:" not in captured.out


def test_interactive_chat_enable_help_true_still_shows_usage(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test that when enable_help=True (default), 'Usage:' messages are shown."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/attach", "/quit")
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
    )
    chat.run()

    captured = capsys.readouterr()
    assert "Usage: /attach <file>" in captured.out


def test_interactive_chat_in_band_save_with_contents_key(
    monkeypatch: Any,
) -> None:
    """Test /save with in_band_files=True when 'contents' key is present in raw input."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    save_path = "/tmp/test_conv.json"
    inputs = [
        {"command": "/save", "message": save_path, "contents": "existing_data"},
        {"command": "/quit", "message": ""},
    ]

    chat_obj = MagicMock()
    chat_obj.save.return_value = '[{"role": "user", "content": "hi"}]'
    mock_callback = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=mock_callback,
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.save.assert_called_once_with()
    # When "contents" is in raw, save_conv is sent via update_callback
    save_conv_calls = [
        call[0][0]["tiz-internal"]["save_conv"]
        for call in mock_callback.call_args_list
        if "save_conv" in call[0][0].get("tiz-internal", {})
    ]
    assert len(save_conv_calls) == 1


def test_interactive_chat_enable_help_false_initial_feedback_blank_still_works(
    monkeypatch: Any,
) -> None:
    """Test that blank feedback still works when enable_help=False."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/quit")
    mock_callback = MagicMock()
    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=mock_callback,
        enable_help=False,
    )
    chat.run()

    # The blank feedback ("") should still be sent (not starting with "Usage:")
    mock_callback.assert_any_call(
        {"tiz-internal": {"interactive_chat_feedback": ""}},
        None,
    )


def test_interactive_chat_enable_help_false_non_usage_feedback_still_works(
    monkeypatch: Any,
) -> None:
    """Test that non-Usage: feedback still works when enable_help=False."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/attach /tmp/file.txt", "/quit")

    chat_obj = MagicMock()
    mock_callback = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=mock_callback,
        enable_help=False,
    )
    chat.chat = chat_obj
    chat.run()

    # Attached: messages should still go through
    mock_callback.assert_any_call(
        {"tiz-internal": {"interactive_chat_feedback": "Attached: /tmp/file.txt"}},
        None,
    )


def test_interactive_chat_record_command_file_unlink_success(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /record when file deletion in finally block succeeds."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    record_path = tmp_path / "test_recording.wav"
    record_path.write_text("fake audio data")

    mock_file = MagicMock()
    mock_file.name = str(record_path)
    mock_file.__enter__.return_value = mock_file
    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(return_value=mock_file),
    )
    monkeypatch.setattr("tiz.interactive_chat.record_audio", MagicMock())

    mock_replay_result = MagicMock()
    mock_replay_result.get.side_effect = lambda key, default=None: (
        "replayed" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.replay.return_value = mock_replay_result

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    assert not record_path.exists()
    captured = capsys.readouterr()
    assert "Attached recording" in captured.out


def test_interactive_chat_record_feedback_raises_when_filepath_none(
    monkeypatch: Any,
) -> None:
    """Test /record when _feedback raises in the except handler and filepath is None.

    This covers the branch 303->-282: exception propagation through the finally
    block when filepath is None.
    """
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(side_effect=OSError("cannot create temp file")),
    )

    def _raise_on_error(msg: dict[str, Any], _subtask: str | None = None) -> None:
        txt = msg.get("tiz-internal", {}).get("interactive_chat_feedback", "")
        if txt.startswith("Error:"):
            raise RuntimeError("feedback failure")

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_raise_on_error,
    )
    with pytest.raises(RuntimeError, match="feedback failure"):
        chat.run()


def test_interactive_chat_record_command_file_unlink_oserror(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /record when file unlinking in finally block raises OSError (suppressed)."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    record_path = tmp_path / "test_recording.wav"

    mock_file = MagicMock()
    mock_file.name = str(record_path)
    mock_file.__enter__.return_value = mock_file
    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(return_value=mock_file),
    )
    monkeypatch.setattr("tiz.interactive_chat.record_audio", MagicMock())

    mock_replay_result = MagicMock()
    mock_replay_result.get.side_effect = lambda key, default=None: (
        "replayed" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.replay.return_value = mock_replay_result

    def _unlink_oserror(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", _unlink_oserror)

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Attached recording" in captured.out


def test_interactive_chat_in_band_save_with_contents_key_no_callback(
    monkeypatch: Any,
) -> None:
    """Test /save with in_band_files=True, contents key, but no update_callback."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = [
        {"command": "/save", "message": "/tmp/test_conv.json", "contents": "data"},
        {"command": "/quit", "message": ""},
    ]

    chat_obj = MagicMock()
    chat_obj.save.return_value = '[{"role": "user", "content": "hi"}]'

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.save.assert_called_once_with()


def test_interactive_chat_record_send_with_retry_exception(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Test /record when _send_with_retry raises a non-KeyboardInterrupt exception."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    record_path = tmp_path / "test_recording.wav"

    mock_file = MagicMock()
    mock_file.name = str(record_path)
    mock_file.__enter__.return_value = mock_file
    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(return_value=mock_file),
    )
    monkeypatch.setattr("tiz.interactive_chat.record_audio", MagicMock())

    chat_obj = MagicMock()
    chat_obj.replay.side_effect = RuntimeError("api error")

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=_make_feedback_printer(),
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Error: api error" in captured.out


def test_interactive_chat_replay_command_checks_usage(
    monkeypatch: Any,
) -> None:
    """Test that replay calls _show_usage and _accumulate_usage."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/replay", "/quit")

    mock_replay_result = MagicMock()
    mock_replay_result.get.side_effect = lambda key, default=None: (
        "replayed message" if key == "message" else default
    )
    chat_obj = MagicMock()
    chat_obj.replay.return_value = mock_replay_result

    mock_callback = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=mock_callback,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.replay.assert_called_once_with(timeout=None)
    # Verify usage was shown after replay
    usage_calls = [
        call
        for call in mock_callback.call_args_list
        if "interactive_chat_usage" in call[0][0].get("tiz-internal", {})
    ]
    assert len(usage_calls) >= 1


def test_interactive_chat_replay_no_message_checks_usage(
    monkeypatch: Any,
) -> None:
    """Test that replay with no message still calls _show_usage."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/replay", "/quit")

    mock_replay_result = MagicMock()
    mock_replay_result.get.side_effect = lambda key, default=None: (
        "" if key == "message" else default
    )
    chat_obj = MagicMock()
    chat_obj.replay.return_value = mock_replay_result

    mock_callback = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=mock_callback,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.replay.assert_called_once_with(timeout=None)
    usage_calls = [
        call
        for call in mock_callback.call_args_list
        if "interactive_chat_usage" in call[0][0].get("tiz-internal", {})
    ]
    assert len(usage_calls) >= 1


def test_interactive_chat_record_success_checks_usage(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Test that /record success calls _accumulate_usage and _show_usage."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = _to_inputs("/record", "/quit")

    record_path = tmp_path / "test_recording.wav"

    mock_file = MagicMock()
    mock_file.name = str(record_path)
    mock_file.__enter__.return_value = mock_file
    monkeypatch.setattr(
        "tiz.interactive_chat.tempfile.NamedTemporaryFile",
        MagicMock(return_value=mock_file),
    )
    monkeypatch.setattr("tiz.interactive_chat.record_audio", MagicMock())

    mock_replay_result = MagicMock()
    mock_replay_result.get.side_effect = lambda key, default=None: (
        "replayed" if key == "message" else default
    )

    chat_obj = MagicMock()
    chat_obj.replay.return_value = mock_replay_result

    mock_callback = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        enable_recording=True,
        update_callback=mock_callback,
    )
    chat.chat = chat_obj
    chat.run()

    chat_obj.append_file.assert_called_once()
    chat_obj.replay.assert_called_once_with(timeout=None)

    # Verify usage was shown
    usage_calls = [
        call
        for call in mock_callback.call_args_list
        if "interactive_chat_usage" in call[0][0].get("tiz-internal", {})
    ]
    assert len(usage_calls) >= 1

    # Verify "Attached recording" feedback was sent via callback
    feedback_calls = [
        call
        for call in mock_callback.call_args_list
        if call[0][0].get("tiz-internal", {}).get("interactive_chat_feedback")
        == "Attached recording"
    ]
    assert len(feedback_calls) == 1


def test_interactive_chat_in_band_attach_invalid_base64(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test /attach with in_band_files=True and invalid base64 contents."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = [
        {
            "command": "/attach",
            "message": "/tmp/test.txt",
            "contents": "not-valid-base64!!",
        },
        {"command": "/quit", "message": ""},
    ]

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Error" in captured.out


def test_interactive_chat_in_band_load_invalid_base64(
    monkeypatch: Any, capsys: Any
) -> None:
    """Test /load with in_band_files=True and invalid base64 contents."""
    _mock_manifest(monkeypatch)
    manifest = _make_minimal_manifest()
    inputs = [
        {
            "command": "/load",
            "message": "/tmp/test.json",
            "contents": "not-valid-base64!!",
        },
        {"command": "/quit", "message": ""},
    ]

    chat_obj = MagicMock()

    chat = InteractiveChat(
        manifest=manifest,
        input_callback=lambda: inputs.pop(0),
        update_callback=_make_feedback_printer(),
        in_band_files=True,
    )
    chat.chat = chat_obj
    chat.run()

    captured = capsys.readouterr()
    assert "Error" in captured.out
