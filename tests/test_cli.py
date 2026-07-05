# mypy: disable-error-code="no-untyped-def,arg-type,no-any-return,attr-defined"
# ruff: noqa: ARG001, ARG002, ARG005, ANN201, B007, SIM117, SIM222
"""Tests for tiz.cli module."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from tiz.cli import (
    DEFAULT_COLOR_REASONING,
    StreamInput,
    StreamUpdater,
    ToolConfirm,
    _build_completion_parser,
    _build_credits_parser,
    _build_sb_parser,
    _build_stats_parser,
    _build_web_parser,
    _ChatCompleter,
    _confirm,
    _extract_engine_from_filename,
    _extract_task_from_filename,
    _handle_chat,
    _handle_credits,
    _handle_exec,
    _handle_sb,
    _handle_stats,
    _handle_stats_usage,
    _handle_web,
    _is_tty,
    _load_usage_logs,
    _maybe_ring_bell,
    _parse_context,
    _parse_date_arg,
    _parse_date_from_filename,
    _parse_hex_color,
    _PathCompleter,
    _print_usage,
    _round_date,
    _set_terminal_title,
    _setup_logging,
    _strip_ansi,
    get_parser,
    main,
)

# ---------------------------------------------------------------------------
# _parse_hex_color tests
# ---------------------------------------------------------------------------


def test_parse_hex_color_with_hash():
    assert _parse_hex_color("#aabbcc") == "170;187;204"


def test_parse_hex_color_with_hash_2():
    assert _parse_hex_color("#abcdef") == "171;205;239"


def test_parse_hex_color_without_hash():
    assert _parse_hex_color("aabbcc") == "170;187;204"


def test_parse_hex_color_black():
    assert _parse_hex_color("#000000") == "0;0;0"


def test_parse_hex_color_white():
    assert _parse_hex_color("#ffffff") == "255;255;255"


def test_parse_hex_color_invalid_short():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_hex_color("#abc")


def test_parse_hex_color_invalid_non_hex():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_hex_color("#zzzzzz")


def test_parse_hex_color_invalid_empty():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_hex_color("")


def test_parse_hex_color_invalid_no_hash_short():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_hex_color("abc")


def test_parse_hex_color_invalid_no_hash_non_hex():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_hex_color("gggggg")


# ---------------------------------------------------------------------------
# _strip_ansi tests
# ---------------------------------------------------------------------------


def test_strip_ansi_no_ansi():
    assert _strip_ansi("hello world") == "hello world"


def test_strip_ansi_empty():
    assert _strip_ansi("") == ""


def test_strip_ansi_escape_code():
    assert _strip_ansi("hello\x1b[31m world") == "hello world"


def test_strip_ansi_multiple_escapes():
    result = _strip_ansi("\x1b[1m\x1b[38;2;117;129;130mthink step\x1b[0m")
    assert result == "think step"


def test_strip_ansi_partial_escape():
    assert _strip_ansi("foo\x1b[0mbar") == "foobar"


def test_strip_ansi_mixed():
    result = _strip_ansi("normal\x1b[31mred\x1b[0mnormal")
    assert result == "normalrednormal"


def test_strip_ansi_std_escape():
    assert _strip_ansi("foo\x1b[Kbar") == "foobar"


# ---------------------------------------------------------------------------
# StreamUpdater tests
# ---------------------------------------------------------------------------


def test_stream_updater_init():
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    assert updater.reasoning is False
    assert updater._reasoning_token_count == 0
    assert updater._hide_reasoning is False
    assert updater._color is True
    assert updater._color_input == _parse_hex_color("#ff00ff")


def test_stream_updater_reasoning_delta_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"reasoning": "think step"}})
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m" in captured.out
    assert "think step" in captured.out
    assert updater.reasoning is True


def test_stream_updater_reasoning_delta_no_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=False,
    )
    updater({"delta": {"reasoning": "think step"}})
    captured = capsys.readouterr()
    assert "<think>" in captured.out
    assert "\033" not in captured.out
    assert "think step" in captured.out


def test_stream_updater_reasoning_strips_ansi(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"reasoning": "think\x1b[31m step"}})
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m" in captured.out
    assert "\x1b[31m" not in captured.out
    assert "think step" in captured.out


def test_stream_updater_reasoning_hidden_dots(capsys):
    updater = StreamUpdater(
        hide_reasoning=True,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    for i in range(12):
        updater({"delta": {"reasoning": "x"}})
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m" in captured.out
    assert captured.out.count(".") == 1


def test_stream_updater_content_after_reasoning_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"reasoning": "think"}})
    capsys.readouterr()
    updater({"delta": {"content": "answer"}})
    captured = capsys.readouterr()
    assert "\033[0m\n" in captured.out
    assert "answer" in captured.out


def test_stream_updater_content_after_reasoning_no_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=False,
    )
    updater({"delta": {"reasoning": "think"}})
    capsys.readouterr()
    updater({"delta": {"content": "answer"}})
    captured = capsys.readouterr()
    assert "</think>\n" in captured.out
    assert "answer" in captured.out


def test_stream_updater_content_no_reasoning(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"content": "hello"}})
    captured = capsys.readouterr()
    assert captured.out == "hello"


def test_stream_updater_content_strips_ansi(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"content": "hello\x1b[1mworld"}})
    captured = capsys.readouterr()
    assert "\x1b[1m" not in captured.out
    assert "helloworld" in captured.out


def test_stream_updater_prompt_progress(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"prompt_progress": {"processed": 5, "total": 10}})
    captured = capsys.readouterr()
    assert "Progress: 50%" in captured.out


def test_stream_updater_prompt_progress_after_reasoning_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"reasoning": "think"}})
    capsys.readouterr()
    updater({"prompt_progress": {"processed": 5, "total": 10}})
    captured = capsys.readouterr()
    assert "\033[0m\n" in captured.out
    assert "Progress: 50%" in captured.out


def test_stream_updater_prompt_progress_after_reasoning_no_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=False,
    )
    updater({"delta": {"reasoning": "think"}})
    capsys.readouterr()
    updater({"prompt_progress": {"processed": 5, "total": 10}})
    captured = capsys.readouterr()
    assert "</think>\n" in captured.out
    assert "Progress: 50%" in captured.out


def test_stream_updater_init_with_parallelism():
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        parallelism=4,
    )
    assert updater._parallelism == 4


def test_stream_updater_with_subtask_name_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {"delta": {"content": "hello"}}
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    # With subtask_name, content delta enters reasoning mode
    assert "\033[38;2;117;129;130mhello" in captured.out
    assert updater.reasoning is True


def test_stream_updater_content_with_subtask_already_reasoning_hidden(capsys):
    """Content delta with subtask_name and hide_reasoning while already in reasoning mode."""
    updater = StreamUpdater(
        hide_reasoning=True,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    # First call sets reasoning=True with a reasoning delta
    updater({"delta": {"reasoning": "think"}})
    capsys.readouterr()
    # Now send content delta with subtask_name, already reasoning=True, hide_reasoning=True
    updater({"delta": {"content": "answer"}}, subtask_name="test_task")
    captured = capsys.readouterr()
    # _reasoning_token_count was 1 from the reasoning delta (hide_reasoning),
    # now incremented to 2 by content delta. 2 % 10 != 0, no dot.
    assert captured.out == ""
    assert updater._reasoning_token_count == 2
    # Send 8 more to trigger dot at 10
    for _ in range(8):
        updater({"delta": {"content": "x"}}, subtask_name="test_task")
    captured = capsys.readouterr()
    assert captured.out.count(".") == 1
    assert updater._reasoning_token_count == 10


def test_stream_updater_with_subtask_name_no_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=False,
    )
    msg = {"delta": {"content": "hello"}}
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    # With subtask_name, content delta enters reasoning mode
    assert "<think>" in captured.out
    assert "hello" in captured.out
    assert updater.reasoning is True


def test_stream_updater_parallelism_suppresses_all_output(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        parallelism=2,
    )
    # All these should produce no visible output
    updater({"delta": {"reasoning": "think step"}})
    captured = capsys.readouterr()
    assert captured.out == ""

    updater({"delta": {"content": "answer"}})
    captured = capsys.readouterr()
    assert captured.out == ""

    updater({"prompt_progress": {"processed": 5, "total": 10}})
    captured = capsys.readouterr()
    assert captured.out == ""

    updater({"foo": "bar"})
    captured = capsys.readouterr()
    assert captured.out == ""

    updater({"tiz-internal": {"status": "starting_task", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert captured.out == ""


def test_stream_updater_parallelism_updates_title(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
        manifests=["my_manifest"],
        parallelism=2,
    )
    updater({"tiz-internal": {"status": "starting_task", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert "\033]0;tiz: run: my_manifest: Task 1: starting\007" in captured.out
    assert "\033]30;tiz: run: my_manifest: Task 1: starting\007" in captured.out
    assert "\033]31;tiz: run: my_manifest: Task 1: starting\007" in captured.out
    assert "tiz: run: my_manifest: Task 1: starting\n" in captured.err


def test_stream_updater_parallelism_non_tiz_internal_returns_early(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        parallelism=2,
    )
    updater({"delta": {"reasoning": "test"}})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert updater.reasoning is False  # Should not have been set
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"foo": "bar"})
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m" in captured.out
    assert "." in captured.out
    assert updater.reasoning is True


def test_stream_updater_else_already_reasoning(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"reasoning": "think"}})
    capsys.readouterr()
    updater({"foo": "bar"})
    captured = capsys.readouterr()
    assert captured.out == "."
    assert updater.reasoning is True


def test_stream_updater_else_not_reasoning_no_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=False,
    )
    updater({"foo": "bar"})
    captured = capsys.readouterr()
    assert "<think>" in captured.out
    assert "." in captured.out
    assert updater.reasoning is True


def test_stream_updater_reasoning_delta_falsey(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"reasoning": ""}})
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m" in captured.out
    assert "." in captured.out


def test_stream_updater_content_delta_falsey(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"delta": {"content": ""}})
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m" in captured.out
    assert "." in captured.out


def test_stream_updater_messages_assistant_content(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {"messages": [{"role": "assistant", "content": "hello world"}]}
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    # "messages" key not handled by StreamUpdater; falls to else -> reasoning + "."
    assert "\033[38;2;117;129;130m." in captured.out
    assert updater.reasoning is True


def test_stream_updater_messages_non_assistant_reasoning_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {
        "messages": [
            {"role": "user", "content": "think step by step"},
            {"role": "assistant", "content": "done"},
        ]
    }
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m." in captured.out
    assert updater.reasoning is True


def test_stream_updater_messages_non_assistant_reasoning_no_color(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=False,
    )
    msg = {
        "messages": [
            {"role": "user", "content": "think step"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    assert "<think>" in captured.out
    assert "." in captured.out
    assert "\033" not in captured.out


def test_stream_updater_messages_non_assistant_missing_content(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {"messages": [{"role": "user"}]}
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m." in captured.out
    assert updater.reasoning is True


def test_stream_updater_messages_non_assistant_content_none(capsys):
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {"messages": [{"role": "user", "content": None}]}
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m." in captured.out
    assert updater.reasoning is True


def test_stream_updater_messages_hide_reasoning(capsys):
    updater = StreamUpdater(
        hide_reasoning=True,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {
        "messages": [
            {"role": "user", "content": "x"},
        ]
    }
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    # "messages" key falls to else block which writes a dot regardless of hide_reasoning
    assert "\033[38;2;117;129;130m" in captured.out
    assert captured.out.count(".") == 1


def test_stream_updater_messages_hide_reasoning_dots(capsys):
    updater = StreamUpdater(
        hide_reasoning=True,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {"messages": [{"role": "user", "content": "x"}]}
    for i in range(12):
        updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    # "messages" key falls to else block which writes one dot per call
    assert captured.out.count(".") == 12


def test_stream_updater_messages_missing_subtask_name_ignored(capsys):
    """When subtask_name is None, the 'messages' key should be ignored (falls to else)."""
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {"messages": [{"role": "assistant", "content": "hello"}]}
    updater(msg)
    captured = capsys.readouterr()
    # Without subtask_name, messages are not handled; falls to else block
    assert "\033[38;2;117;129;130m" in captured.out
    assert "." in captured.out
    assert updater.reasoning is True


def test_stream_updater_messages_non_assistant_reasoning_after_reasoning(capsys):
    """Messages handling transitions correctly from reasoning state."""
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    msg = {"messages": [{"role": "user", "content": "reason"}]}
    updater(msg, subtask_name="test_task")
    captured = capsys.readouterr()
    assert "\033[38;2;117;129;130m" in captured.out
    assert updater.reasoning is True
    # Now send assistant content - still falls to else block (no "delta" key)
    # so it stays in reasoning mode and writes another dot
    msg2 = {"messages": [{"role": "assistant", "content": "answer"}]}
    updater(msg2, subtask_name="test_task")
    captured = capsys.readouterr()
    assert captured.out == "."
    assert updater.reasoning is True


def test_stream_updater_tiz_internal_with_subtask_name_in_title(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {"tiz-internal": {"status": "starting_task", "task": "Task 1"}},
        subtask_name="subtask_abc",
    )
    captured = capsys.readouterr()
    assert "subtask_abc" in captured.out


def test_stream_updater_parallelism_tiz_internal_with_subtask_name(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
        parallelism=2,
    )
    updater(
        {"tiz-internal": {"status": "starting_task", "task": "Task 1"}},
        subtask_name="subtask_xyz",
    )
    captured = capsys.readouterr()
    assert "subtask_xyz" in captured.out


# ---------------------------------------------------------------------------
# _is_tty tests
# ---------------------------------------------------------------------------


def test_is_tty(capsys):
    result = _is_tty()
    assert result is False  # in pytest, stdout is not a tty


# ---------------------------------------------------------------------------
# _set_terminal_title tests
# ---------------------------------------------------------------------------


def test_set_terminal_title(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    _set_terminal_title("test title")
    captured = capsys.readouterr()
    assert "\033]0;test title\007" in captured.out
    assert "\033]30;test title\007" in captured.out
    assert "\033]31;test title\007" in captured.out


def test_set_terminal_title_no_tty(capsys):
    _set_terminal_title("test title")
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# StreamUpdater tiz-internal title update tests
# ---------------------------------------------------------------------------


def test_stream_updater_tiz_internal_no_tty(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False, color_reasoning="#758182", color_input="#ff00ff"
    )
    updater({"tiz-internal": {"status": "starting_task", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert captured.out == ""


def test_stream_updater_tiz_internal_prompt_color(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color=True,
        color_input="#ff00ff",
    )
    updater(
        {"tiz-internal": {"status": "sending", "task": "Task 1", "prompt": "Hello AI"}}
    )
    captured = capsys.readouterr()
    assert "\033[1m\033[38;2;255;0;255m" in captured.out
    assert "Hello AI" in captured.out
    assert "\033[0m\n" in captured.out


def test_stream_updater_tiz_internal_prompt_no_color(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color=False,
        color_input="#ff00ff",
    )
    updater(
        {"tiz-internal": {"status": "sending", "task": "Task 1", "prompt": "Hello AI"}}
    )
    captured = capsys.readouterr()
    assert "prompt> Hello AI" in captured.out
    assert "\033" not in captured.out


def test_stream_updater_tiz_internal_no_prompt(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color=True,
        color_input="#ff00ff",
    )
    updater({"tiz-internal": {"status": "sending", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert "prompt>" not in captured.out
    assert captured.out == ""


def test_stream_updater_tiz_internal_starting_task(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
        manifests=["my_manifest"],
    )
    updater({"tiz-internal": {"status": "starting_task", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert "\033]0;tiz: run: my_manifest: Task 1: starting\007" in captured.out
    assert "\033]30;tiz: run: my_manifest: Task 1: starting\007" in captured.out
    assert "\033]31;tiz: run: my_manifest: Task 1: starting\007" in captured.out


def test_stream_updater_tiz_internal_completed_task(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater({"tiz-internal": {"status": "completed_task", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert "Task 1: completed" in captured.out


def test_stream_updater_tiz_internal_executing_action(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "status": "executing_action",
                "task": "Task 1",
                "action_idx": 2,
                "total_actions": 5,
                "action_type": "ScoringAction",
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: ScoringAction 2/5" in captured.out


def test_stream_updater_tiz_internal_message_group(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    # message_group is always a sub-action of prompts, so send prompts first
    updater(
        {
            "tiz-internal": {
                "action": "prompts",
                "status": "group",
                "task": "Task 1",
                "group": 1,
                "total_groups": 2,
            }
        }
    )
    capsys.readouterr()
    updater(
        {
            "tiz-internal": {
                "action": "message_group",
                "status": "sending",
                "task": "Task 1",
                "message": 1,
                "total_messages": 3,
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: prompt group 1/2: msg 1/3" in captured.out


def test_stream_updater_tiz_internal_prompts_parallel(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "prompts",
                "status": "running",
                "task": "Task 1",
                "prompts_groups": 2,
                "prompts_parallel": True,
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: prompts (parallel)" in captured.out


def test_stream_updater_tiz_internal_prompts_group(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "prompts",
                "status": "group",
                "task": "Task 1",
                "group": 2,
                "total_groups": 4,
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: prompt group 2/4" in captured.out


def test_stream_updater_tiz_internal_iterator_generating(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "iterator",
                "status": "generating_items",
                "task": "Task 1",
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: generating items" in captured.out


def test_stream_updater_tiz_internal_iterator_line(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "iterator",
                "status": "processing_line",
                "task": "Task 1",
                "current_line": 3,
                "total_lines": 10,
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: line 3/10" in captured.out


def test_stream_updater_tiz_internal_repeater(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "repeater",
                "status": "iteration",
                "task": "Task 1",
                "iteration": 1,
                "total_iterations": 3,
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: iteration 1/3" in captured.out


def test_stream_updater_tiz_internal_scoring_round(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "scoring",
                "status": "round",
                "task": "Task 1",
                "round": 1,
                "total_rounds": 3,
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: scoring round 1/3" in captured.out


def test_stream_updater_tiz_internal_scoring_step(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "scoring",
                "status": "scoring_step",
                "task": "Task 1",
                "scoring_step": 2,
                "total_scoring_steps": 5,
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: scoring step 2/5" in captured.out


def test_stream_updater_tiz_internal_scoring_winner(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "scoring",
                "status": "winner_selected",
                "task": "Task 1",
                "winner": "tiz/scoring_task1_r2",
                "winner_engine": "openai",
                "votes": 3,
                "scoring_engines": ["openai", "openai", "anthropic"],
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: scoring: tiz/scoring_task1_r2 (3 votes)" in captured.out


def test_stream_updater_tiz_internal_scoring_generating(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "scoring",
                "status": "generating_iterator_items",
                "task": "Task 1",
            }
        }
    )
    captured = capsys.readouterr()
    assert "Task 1: generating scoring items" in captured.out


def test_stream_updater_tiz_internal_fallback_task(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater({"tiz-internal": {"foo": "bar", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert "Task 1" in captured.out


def test_stream_updater_tiz_internal_empty_manifests(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="chat",
    )
    updater({"tiz-internal": {"status": "starting_task", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert "tiz: chat: Task 1: starting" in captured.out


def test_stream_updater_tiz_internal_status_used_as_action_when_action_missing(
    capsys, monkeypatch
):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater({"tiz-internal": {"status": "some_status", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert "Task 1" in captured.out


def test_stream_updater_tiz_internal_prompts_unknown_status(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "prompts",
                "status": "something_else",
                "task": "Task 1",
            }
        }
    )
    captured = capsys.readouterr()
    assert "tiz: run" in captured.out


def test_stream_updater_tiz_internal_iterator_unknown_status(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "iterator",
                "status": "something_else",
                "task": "Task 1",
            }
        }
    )
    captured = capsys.readouterr()
    assert "tiz: run" in captured.out


def test_stream_updater_tiz_internal_scoring_unknown_status(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        command="run",
    )
    updater(
        {
            "tiz-internal": {
                "action": "scoring",
                "status": "something_else",
                "task": "Task 1",
            }
        }
    )
    captured = capsys.readouterr()
    assert "tiz: run" in captured.out


def test_stream_updater_tiz_internal_transcribe_color(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater(
        {
            "tiz-internal": {
                "transcribe": "hello world",
            }
        }
    )
    captured = capsys.readouterr()
    assert (
        "\033[1m\033[38;2;255;0;255mTranscription:\033[0m hello world\n" in captured.out
    )


def test_stream_updater_tiz_internal_transcribe_no_color(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=False,
    )
    updater(
        {
            "tiz-internal": {
                "transcribe": "hello world",
            }
        }
    )
    captured = capsys.readouterr()
    assert "Transcription: hello world\n" in captured.out
    assert "\033" not in captured.out


def test_stream_updater_tiz_internal_transcribe_with_prompt(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater(
        {
            "tiz-internal": {
                "prompt": "my prompt",
                "transcribe": "hello world",
            }
        }
    )
    captured = capsys.readouterr()
    assert "\033[1m\033[38;2;255;0;255mmy prompt\033[0m\n" in captured.out
    assert (
        "\033[1m\033[38;2;255;0;255mTranscription:\033[0m hello world\n" in captured.out
    )


def test_stream_updater_tiz_internal_transcribe_parallelism(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
        parallelism=1,
    )
    updater(
        {
            "tiz-internal": {
                "transcribe": "hello world",
            }
        }
    )
    captured = capsys.readouterr()
    assert (
        "\033[1m\033[38;2;255;0;255mTranscription:\033[0m hello world\n" in captured.out
    )


def test_stream_updater_tiz_internal_transcribe_parallelism_no_color(
    capsys, monkeypatch
):
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=False,
        parallelism=1,
    )
    updater(
        {
            "tiz-internal": {
                "transcribe": "hello world",
            }
        }
    )
    captured = capsys.readouterr()
    assert "Transcription: hello world\n" in captured.out
    assert "\033" not in captured.out


def test_stream_updater_tiz_internal_transcribe_no_key_ignored(capsys, monkeypatch):
    """When tiz-internal has no 'transcribe' key, nothing transcription-related is printed."""
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater({"tiz-internal": {"status": "sending", "task": "Task 1"}})
    captured = capsys.readouterr()
    assert "Transcription:" not in captured.out


def test_stream_updater_tiz_internal_feedback_non_parallelism(capsys, monkeypatch):
    """interactive_chat_feedback is printed in non-parallelism mode."""
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater(
        {
            "tiz-internal": {
                "interactive_chat_feedback": "feedback msg",
            }
        }
    )
    captured = capsys.readouterr()
    assert "feedback msg" in captured.out


def test_stream_updater_tiz_internal_feedback_parallelism_suppressed(
    capsys, monkeypatch
):
    """interactive_chat_feedback is suppressed when parallelism > 1."""
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
        parallelism=2,
    )
    updater(
        {
            "tiz-internal": {
                "interactive_chat_feedback": "should be suppressed",
            }
        }
    )
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------


def test_stream_updater_tiz_internal_chat_usage(capsys, monkeypatch):
    """interactive_chat_usage with show=True triggers _print_usage with chat data."""
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater(
        {
            "tiz-internal": {
                "interactive_chat_usage": {
                    "task1": {
                        "prompt_tokens": 100,
                        "prompt_time": 2.0,
                        "completion_tokens": 50,
                        "completion_time": 1.0,
                        "cached_tokens": 10,
                        "cache_write_tokens": 5,
                        "cost": 0.015,
                        "tool_calls": [("bash", {})],
                    }
                },
                "interactive_chat_usage_accumulated": True,
            }
        }
    )
    captured = capsys.readouterr()
    assert "Usage: input:\t90 (45.0 tk/s)" in captured.out
    assert "50 (50.0 tk/s)" in captured.out


def test_stream_updater_tiz_internal_chat_usage_no_show(capsys, monkeypatch):
    """interactive_chat_usage without show=True does NOT print usage."""
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater(
        {
            "tiz-internal": {
                "interactive_chat_usage": {
                    "prompt_tokens": 100,
                    "prompt_time": 2.0,
                    "completion_tokens": 50,
                    "completion_time": 1.0,
                    "cached_tokens": 10,
                    "cache_write_tokens": 5,
                    "cost": 0.015,
                    "tool_calls": [("bash", {})],
                },
                "interactive_chat_usage_accumulated": False,
            }
        }
    )
    captured = capsys.readouterr()
    assert captured.out == ""


def test_stream_updater_tiz_internal_chat_usage_no_show_flag(capsys, monkeypatch):
    """interactive_chat_usage without interactive_chat_usage_accumulated flag does NOT print."""
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    updater = StreamUpdater(
        hide_reasoning=False,
        color_reasoning="#758182",
        color_input="#ff00ff",
        color=True,
    )
    updater(
        {
            "tiz-internal": {
                "interactive_chat_usage": {
                    "prompt_tokens": 100,
                    "prompt_time": 2.0,
                    "completion_tokens": 50,
                    "completion_time": 1.0,
                    "cached_tokens": 10,
                    "cache_write_tokens": 5,
                    "cost": 0.015,
                    "tool_calls": [("bash", {})],
                },
            }
        }
    )
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# StreamInput tests
# ---------------------------------------------------------------------------


def test_stream_input_call_no_readline(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", None)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", return_value="hello"):
        result = inp()
    assert result == {"message": "hello", "command": ""}


def test_stream_input_call_color(monkeypatch, capsys):
    monkeypatch.setattr("tiz.cli.readline", None)
    inp = StreamInput(color_input="#ff00ff", color=True)
    with patch("builtins.input", return_value="hello") as mock_input:
        result = inp()
    captured = capsys.readouterr()
    assert result == {"message": "hello", "command": ""}
    assert "\033[0m" in captured.out
    # readline None path: ANSI codes are passed directly in the prompt
    prompt_arg = mock_input.call_args[0][0]
    assert prompt_arg.startswith("\n\033[1m\033[38;2;")
    # No \001/\002 markers since readline is not available
    assert "\001" not in prompt_arg


def test_stream_input_call_eof(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", None)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", side_effect=EOFError):
        result = inp()
    assert result is None


def test_stream_input_with_readline(monkeypatch):
    mock_readline = MagicMock()
    monkeypatch.setattr("tiz.cli.readline", mock_readline)
    inp = StreamInput(color_input="#ff00ff", color=False)
    mock_readline.set_completer.assert_called_once()
    mock_readline.set_completer_delims.assert_called_once_with(" \t\n")
    mock_readline.parse_and_bind.assert_called_once_with("tab: complete")
    with patch("builtins.input", return_value="hello") as mock_input:
        result = inp()
    assert result == {"message": "hello", "command": ""}
    prompt_arg = mock_input.call_args[0][0]
    assert prompt_arg == "\n> "


def test_stream_input_with_readline_and_color(monkeypatch):
    mock_readline = MagicMock()
    monkeypatch.setattr("tiz.cli.readline", mock_readline)
    inp = StreamInput(color_input="#ff00ff", color=True)
    with patch("builtins.input", return_value="hello") as mock_input:
        result = inp()
    assert result == {"message": "hello", "command": ""}
    prompt_arg = mock_input.call_args[0][0]
    # readline path: ANSI codes wrapped in \001/\002 markers for proper wrapping
    assert "\001\033[1m\033[38;2;255;0;255m\002" in prompt_arg
    # Reset is applied after input() so the typed text also gets colored
    assert "\001\033[0m\002" not in prompt_arg


def test_stream_input_keyboard_interrupt_no_readline(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", None)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            inp()


def test_stream_input_keyboard_interrupt_no_buffer(monkeypatch):
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = ""
    monkeypatch.setattr("tiz.cli.readline", mock_readline)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            inp()


def test_stream_input_keyboard_interrupt_with_buffer(monkeypatch):
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "hello"
    monkeypatch.setattr("tiz.cli.readline", mock_readline)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", side_effect=[KeyboardInterrupt, "world"]):
        result = inp()
    assert result == {"message": "world", "command": ""}


def test_stream_input_command(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", None)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", return_value="/help"):
        result = inp()
    assert result == {"message": "", "command": "/help"}


def test_stream_input_command_with_arg(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", None)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", return_value="/attach /tmp/test.txt"):
        result = inp()
    assert result == {"message": "/tmp/test.txt", "command": "/attach"}


def test_stream_input_empty(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", None)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", return_value=""):
        result = inp()
    assert result == {"message": "", "command": ""}


def test_stream_input_whitespace_only(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", None)
    inp = StreamInput(color_input="#ff00ff", color=False)
    with patch("builtins.input", return_value="  "):
        result = inp()
    assert result == {"message": "", "command": ""}


# ---------------------------------------------------------------------------
# ToolConfirm tests
# ---------------------------------------------------------------------------


def test_tool_confirm_init():
    confirm = ToolConfirm(color_input="#ff00ff", color=True)
    assert confirm._color_input == "255;0;255"
    assert confirm._color is True
    assert confirm._color_reasoning == _parse_hex_color(DEFAULT_COLOR_REASONING)


def test_tool_confirm_init_custom_reasoning():
    confirm = ToolConfirm(color_input="#ff00ff", color=True, color_reasoning="#123456")
    assert confirm._color_input == "255;0;255"
    assert confirm._color is True
    assert confirm._color_reasoning == "18;52;86"


def test_tool_confirm_no_color_init():
    confirm = ToolConfirm(color_input="#00ff00", color=False)
    assert confirm._color_input == "0;255;0"
    assert confirm._color is False
    assert confirm._color_reasoning == _parse_hex_color(DEFAULT_COLOR_REASONING)


def test_tool_confirm_format_confirmation_returns_string_no_color(capsys, monkeypatch):
    responses = {}

    def mock_input(prompt=""):
        responses["prompt"] = prompt
        return "n"

    monkeypatch.setattr("builtins.input", mock_input)
    confirm = ToolConfirm(color_input="#ff00ff", color=False)

    def fmt(args, markdown=False):
        return "Formatted: " + str(args)

    result = confirm({"tool": "bash", "arguments": {"cmd": "echo hi"}}, fmt)
    assert result is False
    captured = capsys.readouterr()
    assert "Formatted: {'cmd': 'echo hi'}" in captured.out
    assert "Execute bash?" in responses["prompt"]
    assert "\033" not in captured.out


def test_tool_confirm_format_confirmation_returns_string_color(capsys, monkeypatch):
    responses = {}

    def mock_input(prompt=""):
        responses["prompt"] = prompt
        return "n"

    monkeypatch.setattr("builtins.input", mock_input)
    confirm = ToolConfirm(color_input="#ff00ff", color=True)

    def fmt(args, markdown=False):
        return "Formatted: " + str(args)

    result = confirm({"tool": "bash", "arguments": {"cmd": "echo hi"}}, fmt)
    assert result is False
    captured = capsys.readouterr()
    assert "Formatted: {'cmd': 'echo hi'}" in captured.out
    # The prompt includes ANSI color codes around "bash" and "YES"
    assert "Execute" in responses["prompt"]
    assert "bash" in responses["prompt"]
    assert "YES" in responses["prompt"]
    assert "\033[1m\033[38;2;255;0;255m" in captured.out
    assert "\033[0m" in captured.out
    # Reasoning color is restored after user input
    assert captured.out.endswith("\033[38;2;117;129;130m")


def test_tool_confirm_format_confirmation_returns_none(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    confirm = ToolConfirm(color_input="#ff00ff", color=False)

    def fmt(args, markdown=False):
        return None

    result = confirm({"tool": "bash", "arguments": {"cmd": "echo hi"}}, fmt)
    assert result is False
    captured = capsys.readouterr()
    assert "{'cmd': 'echo hi'}" in captured.out


def test_tool_confirm_input_yes(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "YES")
    confirm = ToolConfirm(color_input="#ff00ff", color=False)

    def fmt(args, markdown=False):
        return "proceed?"

    result = confirm({"tool": "read", "arguments": {}}, fmt)
    assert result is True


def test_tool_confirm_input_yes_lowercase_loops_then_no(capsys, monkeypatch):
    """Only uppercase YES is accepted, lowercase yes loops and re-asks."""
    responses = iter(["yes", "n"])

    def mock_input(prompt=""):
        return next(responses)

    monkeypatch.setattr("builtins.input", mock_input)
    confirm = ToolConfirm(color_input="#ff00ff", color=False)

    def fmt(args, markdown=False):
        return "proceed?"

    result = confirm({"tool": "read", "arguments": {}}, fmt)
    assert result is False


def test_tool_confirm_input_empty(capsys, monkeypatch):
    responses = iter(["", "n"])

    def mock_input(prompt=""):
        return next(responses)

    monkeypatch.setattr("builtins.input", mock_input)
    confirm = ToolConfirm(color_input="#ff00ff", color=False)

    def fmt(args, markdown=False):
        return "proceed?"

    result = confirm({"tool": "read", "arguments": {}}, fmt)
    assert result is False


def test_tool_confirm_input_eof(capsys, monkeypatch):
    with patch("builtins.input", side_effect=EOFError):
        confirm = ToolConfirm(color_input="#ff00ff", color=False)

        def fmt(args, markdown=False):
            return "proceed?"

        result = confirm({"tool": "read", "arguments": {}}, fmt)
        assert result is False


def test_tool_confirm_input_keyboard_interrupt(capsys, monkeypatch):
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        confirm = ToolConfirm(color_input="#ff00ff", color=False)

        def fmt(args, markdown=False):
            return "proceed?"

        result = confirm({"tool": "read", "arguments": {}}, fmt)
        assert result is False


def test_tool_confirm_no_tool_key(monkeypatch):
    responses = {}

    def mock_input(prompt=""):
        responses["prompt"] = prompt
        return "n"

    monkeypatch.setattr("builtins.input", mock_input)
    confirm = ToolConfirm(color_input="#ff00ff", color=False)

    def fmt(args, markdown=False):
        return "confirming"

    result = confirm({"arguments": {"key": "val"}}, fmt)
    assert result is False
    assert "Execute unknown?" in responses["prompt"]


def test_tool_confirm_with_subtask_name_no_color(capsys, monkeypatch):
    responses = {}

    def mock_input(prompt=""):
        responses["prompt"] = prompt
        return "n"

    monkeypatch.setattr("builtins.input", mock_input)
    confirm = ToolConfirm(color_input="#ff00ff", color=False)

    def fmt(args, markdown=False):
        return "Formatted: " + str(args)

    result = confirm(
        {"tool": "bash", "arguments": {"cmd": "echo hi"}}, fmt, subtask_name="build"
    )
    assert result is False
    captured = capsys.readouterr()
    assert "Formatted: {'cmd': 'echo hi'}" in captured.out
    assert "Execute build (bash)?" in responses["prompt"]
    assert "\033" not in captured.out


def test_tool_confirm_with_subtask_name_color(capsys, monkeypatch):
    responses = {}

    def mock_input(prompt=""):
        responses["prompt"] = prompt
        return "n"

    monkeypatch.setattr("builtins.input", mock_input)
    confirm = ToolConfirm(color_input="#ff00ff", color=True)

    def fmt(args, markdown=False):
        return "Formatted: " + str(args)

    result = confirm(
        {"tool": "bash", "arguments": {"cmd": "echo hi"}}, fmt, subtask_name="build"
    )
    assert result is False
    captured = capsys.readouterr()
    assert "Formatted: {'cmd': 'echo hi'}" in captured.out
    assert "Execute" in responses["prompt"]
    assert "build (bash)" in responses["prompt"]
    assert "YES" in responses["prompt"]
    assert "\033[1m\033[38;2;255;0;255m" in captured.out
    assert "\033[0m" in captured.out
    # Reasoning color is restored after user input
    assert captured.out.endswith("\033[38;2;117;129;130m")


def test_tool_confirm_with_subtask_name_eof(capsys, monkeypatch):
    """ToolConfirm with subtask_name handles EOFError."""
    with patch("builtins.input", side_effect=EOFError):
        confirm = ToolConfirm(color_input="#ff00ff", color=False)

        def fmt(args, markdown=False):
            return "proceed?"

        result = confirm({"tool": "read", "arguments": {}}, fmt, subtask_name="build")
        assert result is False


def test_tool_confirm_with_subtask_name_keyboard_interrupt(capsys, monkeypatch):
    """ToolConfirm with subtask_name handles KeyboardInterrupt."""
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        confirm = ToolConfirm(color_input="#ff00ff", color=False)

        def fmt(args, markdown=False):
            return "proceed?"

        result = confirm({"tool": "read", "arguments": {}}, fmt, subtask_name="build")
        assert result is False


def test_tool_confirm_with_subtask_name_unknown_tool(capsys, monkeypatch):
    responses = {}

    def mock_input(prompt=""):
        responses["prompt"] = prompt
        return "n"

    monkeypatch.setattr("builtins.input", mock_input)
    confirm = ToolConfirm(color_input="#ff00ff", color=False)

    def fmt(args, markdown=False):
        return "confirming"

    result = confirm({"arguments": {"key": "val"}}, fmt, subtask_name="test_task")
    assert result is False
    assert "Execute test_task (unknown)?" in responses["prompt"]


# ---------------------------------------------------------------------------
# _setup_logging tests
# ---------------------------------------------------------------------------


def test_setup_logging_warning():
    with patch("tiz.cli.logging_init") as mock_init:
        _setup_logging(0)
    mock_init.assert_called_once_with(logging.WARNING)


def test_setup_logging_info():
    with patch("tiz.cli.logging_init") as mock_init:
        _setup_logging(1)
    mock_init.assert_called_once_with(logging.INFO)


def test_setup_logging_debug():
    with patch("tiz.cli.logging_init") as mock_init:
        _setup_logging(2)
    mock_init.assert_called_once_with(logging.DEBUG)


def test_setup_logging_debug_high():
    with patch("tiz.cli.logging_init") as mock_init:
        _setup_logging(5)
    mock_init.assert_called_once_with(logging.DEBUG)


# ---------------------------------------------------------------------------
# _build_sb_parser tests
# ---------------------------------------------------------------------------


def test_build_sb_parser():
    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    sb_parser = _build_sb_parser(main_subparsers)
    assert sb_parser is not None

    parsed = main_parser.parse_args(["sb", "list"])
    assert parsed.command == "sb"
    assert parsed.sb_command == "list"

    parsed = main_parser.parse_args(["sb", "ls", "mysandbox"])
    assert parsed.sb_command == "ls"
    assert parsed.sandbox_name == "mysandbox"

    parsed = main_parser.parse_args(["sb", "containers", "mysandbox"])
    assert parsed.sb_command == "containers"
    assert parsed.sandbox_name == "mysandbox"

    parsed = main_parser.parse_args(["sb", "cleanup"])
    assert parsed.sb_command == "cleanup"
    assert parsed.untracked_only is False
    assert parsed.dead_only is False

    parsed = main_parser.parse_args(["sb", "cleanup", "--untracked-only"])
    assert parsed.untracked_only is True

    parsed = main_parser.parse_args(["sb", "cleanup", "--dead-only"])
    assert parsed.dead_only is True

    parsed = main_parser.parse_args(["sb", "kill", "mysandbox", "--do-not-ask"])
    assert parsed.sb_command == "kill"
    assert parsed.sandbox_name == "mysandbox"
    assert parsed.do_not_ask is True

    parsed = main_parser.parse_args(["sb", "rm", "mysandbox"])
    assert parsed.sb_command == "rm"
    assert parsed.sandbox_name == "mysandbox"

    parsed = main_parser.parse_args(
        ["sb", "sync", "from-origin", "mysandbox", "--force"]
    )
    assert parsed.sb_command == "sync"
    assert parsed.sync_direction == "from-origin"
    assert parsed.sandbox_name == "mysandbox"
    assert parsed.force is True

    parsed = main_parser.parse_args(["sb", "sync", "to-origin", "sand2"])
    assert parsed.sync_direction == "to-origin"
    assert parsed.sandbox_name == "sand2"

    parsed = main_parser.parse_args(["sb", "build", "image", "tiz-worker-js:latest"])
    assert parsed.build_command == "image"
    assert parsed.tag == "tiz-worker-js:latest"

    parsed = main_parser.parse_args(["sb", "build", "all"])
    assert parsed.build_command == "all"

    parsed = main_parser.parse_args(["sb", "rm-images", "--do-not-ask"])
    assert parsed.sb_command == "rm-images"
    assert parsed.do_not_ask is True

    parsed = main_parser.parse_args(
        ["sb", "logs", "mysandbox", "--container", "c1", "--separate"]
    )
    assert parsed.sb_command == "logs"
    assert parsed.sandbox_name == "mysandbox"
    assert parsed.container == "c1"
    assert parsed.separate is True

    parsed = main_parser.parse_args(["sb", "rm-all", "--do-not-ask"])
    assert parsed.sb_command == "rm-all"
    assert parsed.do_not_ask is True


# ---------------------------------------------------------------------------
# _confirm tests
# ---------------------------------------------------------------------------


def test_confirm_do_not_ask():
    assert _confirm(True, "msg") is True


def test_confirm_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert _confirm(False, "msg") is True


def test_confirm_yes_upper(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Y")
    assert _confirm(False, "msg") is True


def test_confirm_yes_word(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    assert _confirm(False, "msg") is True


def test_confirm_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert _confirm(False, "msg") is False


def test_confirm_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert _confirm(False, "msg") is False


# ---------------------------------------------------------------------------
# _handle_sb tests
# ---------------------------------------------------------------------------


def _make_sb_args(**kwargs: Any) -> MagicMock:
    args = MagicMock()
    args.verbose = 0
    for k, v in kwargs.items():
        setattr(args, k, v)
    if "sb_command" not in kwargs:
        args.sb_command = None
    return args


def test_handle_sb_no_command(capsys):
    parser = MagicMock()
    args = _make_sb_args(sb_command=None)
    result = _handle_sb(args, Path("/tmp"), parser)
    assert result == 1
    parser.print_help.assert_called_once()


def test_handle_sb_no_engine(monkeypatch):
    monkeypatch.setattr(
        "tiz.cli.SandboxManager.available_engine", MagicMock(return_value=None)
    )
    args = _make_sb_args(sb_command="list")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1


def _patch_sb(monkeypatch) -> tuple[MagicMock, MagicMock]:
    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.cli.SandboxManager", manager_cls)
    return mock_manager, manager_cls


def test_handle_sb_list_all(capsys, monkeypatch):
    mock_manager, manager_cls = _patch_sb(monkeypatch)
    mock_manager.list_sandboxes.return_value = ["sb1", "sb2"]
    args = _make_sb_args(sb_command="list", sandbox_name=None)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "sb1\nsb2\n" in captured.out


def test_handle_sb_list_specific_found(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.list_sandboxes.return_value = ["sb1", "sb2"]
    args = _make_sb_args(sb_command="list", sandbox_name="sb1")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "sb1"


def test_handle_sb_list_specific_not_found(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.list_sandboxes.return_value = ["sb1", "sb2"]
    args = _make_sb_args(sb_command="list", sandbox_name="sb3")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_handle_sb_ls_alias(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.list_sandboxes.return_value = ["sb1"]
    args = _make_sb_args(sb_command="ls", sandbox_name=None)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0


def _make_mock_container(**kwargs) -> MagicMock:
    c = MagicMock()
    for k, v in kwargs.items():
        setattr(c, k, v)
    return c


def test_handle_sb_containers_with_name(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    c1 = _make_mock_container(
        container_id="cid1", container_name="cname1", engine="docker"
    )
    c1.is_running.return_value = True
    mock_manager.get_sandbox_containers.return_value = [c1]
    args = _make_sb_args(sb_command="containers", sandbox_name="mysandbox")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "cid1" in captured.out
    assert "cname1" in captured.out
    assert "running" in captured.out


def test_handle_sb_containers_without_name(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.list_containers.return_value = [
        {
            "sandbox_name": "sb1",
            "container_id": "cid1",
            "container_name": "cname1",
            "engine": "docker",
        }
    ]
    args = _make_sb_args(sb_command="containers", sandbox_name=None)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "sb1\tcid1\tcname1\tdocker" in captured.out


def test_handle_sb_cleanup_both(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.cleanup_untracked_containers.return_value = ["untracked1"]
    mock_manager.cleanup_dead_entries.return_value = ["dead1"]
    args = _make_sb_args(sb_command="cleanup", untracked_only=False, dead_only=False)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "untracked1" in captured.out
    assert "dead1" in captured.out


def test_handle_sb_cleanup_dead_only(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.cleanup_untracked_containers.return_value = ["c1"]
    args = _make_sb_args(sb_command="cleanup", untracked_only=False, dead_only=True)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.cleanup_untracked_containers.assert_not_called()
    mock_manager.cleanup_dead_entries.assert_called_once()


def test_handle_sb_cleanup_untracked_only(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.cleanup_dead_entries.return_value = ["d1"]
    args = _make_sb_args(sb_command="cleanup", untracked_only=True, dead_only=False)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.cleanup_untracked_containers.assert_called_once()
    mock_manager.cleanup_dead_entries.assert_not_called()


def test_handle_sb_kill_with_name_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    args = _make_sb_args(sb_command="kill", sandbox_name="mysandbox", do_not_ask=True)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.kill_all_containers.assert_called_once_with("mysandbox")


def test_handle_sb_kill_with_name_not_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    args = _make_sb_args(sb_command="kill", sandbox_name="mysandbox", do_not_ask=False)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.kill_all_containers.assert_not_called()


def test_handle_sb_kill_all_no_sandboxes(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.list_sandboxes.return_value = []
    args = _make_sb_args(sb_command="kill", sandbox_name=None, do_not_ask=True)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "Error: No sandboxes found" in captured.err


def test_handle_sb_kill_all_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.list_sandboxes.return_value = ["sb1", "sb2"]
    args = _make_sb_args(sb_command="kill", sandbox_name=None, do_not_ask=True)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    assert mock_manager.kill_all_containers.call_args_list == [
        call("sb1"),
        call("sb2"),
    ]


def test_handle_sb_rm_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    args = _make_sb_args(sb_command="rm", sandbox_name="mysandbox", do_not_ask=True)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.kill_and_delete_sandbox.assert_called_once_with("mysandbox")


def test_handle_sb_rm_not_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    args = _make_sb_args(sb_command="rm", sandbox_name="mysandbox", do_not_ask=False)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.kill_and_delete_sandbox.assert_not_called()


def test_handle_sb_sync_from_origin(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    args = _make_sb_args(
        sb_command="sync",
        sandbox_name="mysandbox",
        sync_direction="from-origin",
        force=False,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.sync_from_original.assert_called_once_with("mysandbox", force=False)


def test_handle_sb_sync_to_origin(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    args = _make_sb_args(
        sb_command="sync",
        sandbox_name="mysandbox",
        sync_direction="to-origin",
        force=True,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.sync_to_original.assert_called_once_with("mysandbox", force=True)


def test_handle_sb_sync_no_direction(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    args = _make_sb_args(
        sb_command="sync",
        sandbox_name="mysandbox",
        sync_direction=None,
        force=False,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1
    captured = capsys.readouterr()
    assert "sync direction required" in captured.err


def _mock_get_data_dirs(monkeypatch, containerfiles_dirs=None):
    if containerfiles_dirs is None:
        containerfiles_dirs = [Path("/fake/containerfiles")]
    monkeypatch.setattr(
        "tiz.cli.SandboxManager.get_containerfiles_dirs",
        MagicMock(return_value=containerfiles_dirs),
    )


def test_handle_sb_build_image_bad_tag(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    _mock_get_data_dirs(monkeypatch)
    args = _make_sb_args(
        sb_command="build",
        build_command="image",
        tag="bad-tag",
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1
    captured = capsys.readouterr()
    assert "tag must start with 'tiz-worker'" in captured.err


def test_handle_sb_build_image_not_found(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    _mock_get_data_dirs(monkeypatch)
    args = _make_sb_args(
        sb_command="build",
        build_command="image",
        tag="tiz-worker-js:latest",
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1
    captured = capsys.readouterr()
    assert "no default containerfile" in captured.err


def test_handle_sb_build_image_success(capsys, monkeypatch, tmp_path):
    mock_manager, _ = _patch_sb(monkeypatch)
    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir()
    cf_path = cf_dir / "Containerfile.tiz-worker-js"
    cf_path.write_text("FROM ubuntu")
    _mock_get_data_dirs(monkeypatch, containerfiles_dirs=[cf_dir])
    args = _make_sb_args(
        sb_command="build",
        build_command="image",
        tag="tiz-worker-js:latest",
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "Built image" in captured.out
    mock_manager.build_image.assert_called_once_with(
        containerfile="FROM ubuntu", tag="tiz-worker-js:latest", delete_existing=True
    )


def test_handle_sb_build_all_success(capsys, monkeypatch, tmp_path):
    mock_manager, _ = _patch_sb(monkeypatch)
    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir()
    (cf_dir / "Containerfile.tiz-worker-js").write_text("FROM node")
    (cf_dir / "Containerfile.tiz-worker-py").write_text("FROM python")
    _mock_get_data_dirs(monkeypatch, containerfiles_dirs=[cf_dir])
    args = _make_sb_args(sb_command="build", build_command="all")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "tiz-worker-js" in captured.out
    assert "tiz-worker-py" in captured.out
    assert mock_manager.build_image.call_count == 2


def test_handle_sb_build_all_no_containerfiles(capsys, monkeypatch, tmp_path):
    mock_manager, _ = _patch_sb(monkeypatch)
    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir()
    _mock_get_data_dirs(monkeypatch, containerfiles_dirs=[cf_dir])
    args = _make_sb_args(sb_command="build", build_command="all")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1
    captured = capsys.readouterr()
    assert "no default containerfiles found" in captured.err


def test_handle_sb_build_no_command(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    _mock_get_data_dirs(monkeypatch)
    args = _make_sb_args(sb_command="build", build_command=None)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1
    captured = capsys.readouterr()
    assert "build command required" in captured.err


def test_handle_sb_logs_with_container(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    c1 = _make_mock_container(
        container_id="cid1", container_name="cname1", engine="docker"
    )
    c1.get_container_logs.return_value = "log output"
    mock_manager.get_sandbox_containers.return_value = [c1]
    args = _make_sb_args(
        sb_command="logs",
        sandbox_name="mysandbox",
        container="cname1",
        separate=False,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "log output" in captured.out


def test_handle_sb_logs_container_not_found(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    c1 = _make_mock_container(
        container_id="cid1", container_name="cname1", engine="docker"
    )
    mock_manager.get_sandbox_containers.return_value = [c1]
    args = _make_sb_args(
        sb_command="logs",
        sandbox_name="mysandbox",
        container="nonexistent",
        separate=False,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1
    captured = capsys.readouterr()
    assert "Container 'nonexistent' not found" in captured.err


def test_handle_sb_logs_separate_single(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    c1 = _make_mock_container(
        container_id="cid1", container_name="cname1", engine="docker"
    )
    c1.get_container_logs.return_value = ("stdout output", "stderr output")
    mock_manager.get_sandbox_containers.return_value = [c1]
    args = _make_sb_args(
        sb_command="logs",
        sandbox_name="mysandbox",
        container=None,
        separate=True,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "stdout output" in captured.out


def test_handle_sb_logs_separate_multiple(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    c1 = _make_mock_container(
        container_id="cid1", container_name="cname1", engine="docker"
    )
    c1.get_container_logs.return_value = ("stdout output", None)
    c2 = _make_mock_container(
        container_id="cid2", container_name="cname2", engine="docker"
    )
    c2.get_container_logs.return_value = ("stdout2", "stderr2")
    mock_manager.get_sandbox_containers.return_value = [c1, c2]
    args = _make_sb_args(
        sb_command="logs",
        sandbox_name="mysandbox",
        container=None,
        separate=True,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "=== cname1 STDOUT ===" in captured.out
    assert "=== cname2 STDERR ===" in captured.out


def test_handle_sb_logs_error(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    c1 = _make_mock_container(
        container_id="cid1", container_name="cname1", engine="docker"
    )
    c1.get_container_logs.side_effect = RuntimeError("log error")
    mock_manager.get_sandbox_containers.return_value = [c1]
    args = _make_sb_args(
        sb_command="logs",
        sandbox_name="mysandbox",
        container=None,
        separate=False,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "Error getting logs" in captured.err


def test_handle_sb_logs_not_separate_multiple(capsys, monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    c1 = _make_mock_container(
        container_id="cid1", container_name="cname1", engine="docker"
    )
    c1.get_container_logs.return_value = "combined output"
    c2 = _make_mock_container(
        container_id="cid2", container_name="cname2", engine="docker"
    )
    c2.get_container_logs.return_value = "output 2"
    mock_manager.get_sandbox_containers.return_value = [c1, c2]
    args = _make_sb_args(
        sb_command="logs",
        sandbox_name="mysandbox",
        container=None,
        separate=False,
    )
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "=== cname1 ===" in captured.out


def test_handle_sb_rm_images_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.delete_tiz_worker_images.side_effect = [
        ["img1", "img2"],
        ["img1", "img2"],
    ]
    args = _make_sb_args(sb_command="rm-images", do_not_ask=True)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    assert mock_manager.delete_tiz_worker_images.call_count == 2


def test_handle_sb_rm_images_not_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    args = _make_sb_args(sb_command="rm-images", do_not_ask=False)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.delete_tiz_worker_images.assert_called_once_with(dry_run=True)


def test_handle_sb_rm_all_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    args = _make_sb_args(sb_command="rm-all", do_not_ask=True)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.kill_and_delete_all_sandboxes.assert_called_once()


# ---------------------------------------------------------------------------
# _build_stats_parser tests
# ---------------------------------------------------------------------------


def test_build_stats_parser():
    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    _build_stats_parser(main_subparsers)
    parsed = main_parser.parse_args(["stats", "usage"])
    assert parsed.command == "stats"
    assert parsed.stats_command == "usage"


def test_build_stats_parser_with_args():
    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    _build_stats_parser(main_subparsers)
    parsed = main_parser.parse_args(
        [
            "stats",
            "usage",
            "--period",
            "weekly",
            "--from-date",
            "2024-01-01",
            "--to-date",
            "2024-12-31",
        ]
    )
    assert parsed.period == "weekly"
    assert parsed.from_date == "2024-01-01"
    assert parsed.to_date == "2024-12-31"


# ---------------------------------------------------------------------------
# _build_completion_parser tests
# ---------------------------------------------------------------------------


def test_build_completion_parser_default_shell():
    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    _build_completion_parser(main_subparsers)
    parsed = main_parser.parse_args(["completion"])
    assert parsed.command == "completion"
    assert parsed.shell == "bash"


def test_build_completion_parser_zsh():

    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    _build_completion_parser(main_subparsers)
    parsed = main_parser.parse_args(["completion", "zsh"])
    assert parsed.command == "completion"
    assert parsed.shell == "zsh"


def test_build_completion_parser_fish():

    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    _build_completion_parser(main_subparsers)
    parsed = main_parser.parse_args(["completion", "fish"])
    assert parsed.command == "completion"
    assert parsed.shell == "fish"


def test_build_completion_parser_tcsh():

    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    _build_completion_parser(main_subparsers)
    parsed = main_parser.parse_args(["completion", "tcsh"])
    assert parsed.command == "completion"
    assert parsed.shell == "tcsh"


# ---------------------------------------------------------------------------
# _build_credits_parser tests
# ---------------------------------------------------------------------------


def test_build_credits_parser():

    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    _build_credits_parser(main_subparsers)
    parsed = main_parser.parse_args(["credits", "-m", "manifest.yaml"])
    assert parsed.command == "credits"
    assert parsed.manifest == [Path("manifest.yaml")]


def test_build_credits_parser_no_manifest_fails():

    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    _build_credits_parser(main_subparsers)
    with pytest.raises(SystemExit):
        main_parser.parse_args(["credits"])


# ---------------------------------------------------------------------------
# _handle_credits tests
# ---------------------------------------------------------------------------


def _make_credits_manifest(inference_engines: list | None = None) -> MagicMock:
    manifest = MagicMock()
    if inference_engines is not None:
        manifest.inference_engines = inference_engines
    else:
        manifest.inference_engines = []
    return manifest


def test_handle_credits_no_engines(capsys):
    manifest = _make_credits_manifest(inference_engines=[])
    result = _handle_credits(manifest)
    assert result == 1
    captured = capsys.readouterr()
    assert "No inference engines found" in captured.err


def test_handle_credits_single_engine_success(capsys, monkeypatch):
    mock_engine = MagicMock()
    mock_engine.name = "my-engine"

    manifest = _make_credits_manifest(inference_engines=[mock_engine])

    monkeypatch.setattr(
        "tiz.cli.get_credits",
        MagicMock(
            return_value=[
                {
                    "name": "my-engine",
                    "total_credits": 100.0,
                    "total_usage": 25.0,
                    "remaining": 75.0,
                }
            ]
        ),
    )

    result = _handle_credits(manifest)
    assert result == 0
    captured = capsys.readouterr()
    assert "[my-engine]" in captured.out
    assert "$100.0000" in captured.out
    assert "$25.0000" in captured.out
    assert "$75.0000" in captured.out
    # No TOTAL line for single engine
    assert "[TOTAL]" not in captured.out


def test_handle_credits_multiple_engines(capsys, monkeypatch):
    mock_engine1 = MagicMock()
    mock_engine1.name = "engine1"
    mock_engine2 = MagicMock()
    mock_engine2.name = "engine2"

    manifest = _make_credits_manifest(inference_engines=[mock_engine1, mock_engine2])

    monkeypatch.setattr(
        "tiz.cli.get_credits",
        MagicMock(
            return_value=[
                {
                    "name": "engine1",
                    "total_credits": 50.0,
                    "total_usage": 10.0,
                    "remaining": 40.0,
                },
                {
                    "name": "engine2",
                    "total_credits": 200.0,
                    "total_usage": 50.0,
                    "remaining": 150.0,
                },
            ]
        ),
    )

    result = _handle_credits(manifest)
    assert result == 0
    captured = capsys.readouterr()
    assert "[engine1]" in captured.out
    assert "[engine2]" in captured.out
    assert "$50.0000" in captured.out
    assert "$200.0000" in captured.out
    assert "[TOTAL]" not in captured.out


def test_handle_credits_missing_keys(capsys, monkeypatch):
    """_handle_credits handles missing keys gracefully."""
    mock_engine = MagicMock()
    mock_engine.name = "my-engine"

    manifest = _make_credits_manifest(inference_engines=[mock_engine])

    monkeypatch.setattr(
        "tiz.cli.get_credits",
        MagicMock(
            return_value=[
                {
                    "name": "my-engine",
                }
            ]
        ),
    )

    with pytest.raises(KeyError):
        _handle_credits(manifest)


def test_handle_credits_main_invocation(capsys, monkeypatch, tmp_path):
    """Test the full credits command through main()."""
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(
        "inference_engines:\n"
        "  - name: my-engine\n"
        "    type: llamacpp\n"
        "    host: http://127.0.0.1:8080\n",
        encoding="utf-8",
    )

    mock_client = MagicMock()
    mock_client.get_credits.return_value = {
        "total_credits": 100.0,
        "total_usage": 25.0,
    }
    monkeypatch.setattr(
        "tiz.helpers.BaseTaskExecutor.build_client",
        MagicMock(return_value=mock_client),
    )

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "credits", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "[my-engine]" in captured.out
    assert "$100.0000" in captured.out


# ---------------------------------------------------------------------------
# main() completion tests
# ---------------------------------------------------------------------------


def test_main_completion_default(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.shellcode", lambda shell: f"# completion for {shell}")
    with patch.object(sys, "argv", ["tiz", "completion"]):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "# completion for bash" in captured.out


def test_main_completion_zsh(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.shellcode", lambda shell: f"# completion for {shell}")
    with patch.object(sys, "argv", ["tiz", "completion", "zsh"]):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "# completion for zsh" in captured.out


def test_main_completion_fish(capsys, monkeypatch):
    monkeypatch.setattr("tiz.cli.shellcode", lambda shell: f"# completion for {shell}")
    with patch.object(sys, "argv", ["tiz", "completion", "fish"]):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "# completion for fish" in captured.out


# ---------------------------------------------------------------------------
# _parse_date_arg tests
# ---------------------------------------------------------------------------


def test_parse_date_arg():
    assert _parse_date_arg("2024-01-15") == date(2024, 1, 15)


# ---------------------------------------------------------------------------
# _parse_date_from_filename tests
# ---------------------------------------------------------------------------


def test_parse_date_from_filename():
    assert _parse_date_from_filename("20240115") == date(2024, 1, 15)


# ---------------------------------------------------------------------------
# _load_usage_logs tests
# ---------------------------------------------------------------------------


def test_load_usage_logs_no_dir(tmp_path):
    result = _load_usage_logs(tmp_path)
    assert result == []


def test_load_usage_logs_empty_dir(tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    result = _load_usage_logs(tmp_path)
    assert result == []


def test_load_usage_logs_with_files(tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_test_log.log").write_text(
        json.dumps({"prompt_tokens": 10}), encoding="utf-8"
    )
    result = _load_usage_logs(tmp_path)
    assert len(result) == 1
    assert result[0]["_file"] == "20240115_120000_test_log.log"
    assert result[0]["prompt_tokens"] == 10


def test_load_usage_logs_skips_non_log(tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "data.json").write_text(
        json.dumps({"prompt_tokens": 10}), encoding="utf-8"
    )
    result = _load_usage_logs(tmp_path)
    assert result == []


def test_load_usage_logs_skips_bad_json(tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_test_log.log").write_text(
        "bad json", encoding="utf-8"
    )
    result = _load_usage_logs(tmp_path)
    assert result == []


def test_load_usage_logs_oserror(tmp_path, monkeypatch):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    log_file = usage_dir / "20240115_120000_test_log.log"
    log_file.write_text("{}", encoding="utf-8")

    original_read_text = Path.read_text

    def broken_read(self, *a, **kw):
        if self.name.endswith(".log"):
            raise OSError("permission denied")
        return original_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", broken_read)
    result = _load_usage_logs(tmp_path)
    assert result == []


# ---------------------------------------------------------------------------
# _extract_engine_from_filename tests
# ---------------------------------------------------------------------------


def test_extract_engine_from_filename_matches():
    assert _extract_engine_from_filename("20240115_120000_openai_gpt4.log") == "openai"


def test_extract_engine_from_filename_no_match():
    assert _extract_engine_from_filename("random.log") == "unknown"


def test_extract_engine_from_filename_empty():
    assert _extract_engine_from_filename("") == "unknown"


def test_extract_engine_from_filename_no_underscores():
    assert _extract_engine_from_filename("justafile.log") == "unknown"


def test_extract_engine_from_filename_multiple_underscores():
    assert (
        _extract_engine_from_filename("20240115_120000_openai_gpt4_v2.log") == "openai"
    )


def test_extract_task_from_filename_matches():
    assert _extract_task_from_filename("20240115_120000_openai_mytask.log") == "mytask"


def test_extract_task_from_filename_no_match():
    assert _extract_task_from_filename("random.log") == "unknown"


def test_extract_task_from_filename_empty():
    assert _extract_task_from_filename("") == "unknown"


def test_extract_task_from_filename_no_underscores():
    assert _extract_task_from_filename("justafile.log") == "unknown"


def test_extract_task_from_filename_multiple_underscores():
    assert (
        _extract_task_from_filename("20240115_120000_openai_gpt4_v2.log") == "gpt4_v2"
    )


def test_parse_date_arg_today():
    assert _parse_date_arg("today") == date.today()


def test_parse_date_arg_today_uppercase():
    assert _parse_date_arg("TODAY") == date.today()


# ---------------------------------------------------------------------------
# _handle_stats tests
# ---------------------------------------------------------------------------


def _make_stats_args(**kwargs: Any) -> MagicMock:
    args = MagicMock()
    args.to_date = None
    args.from_date = None
    args.period = "daily"
    args.engine_filter = None
    args.task_filter = None
    args.group_by_task = False
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def test_handle_stats_usage(monkeypatch):
    args = _make_stats_args(stats_command="usage")
    result = _handle_stats(args, Path("/tmp"))
    assert result == 0


def test_handle_stats_no_command(capsys, monkeypatch):
    args = MagicMock()
    args.stats_command = None
    result = _handle_stats(args, Path("/tmp"))
    assert result == 1
    captured = capsys.readouterr()
    assert "stats command required" in captured.err


# ---------------------------------------------------------------------------
# _round_date tests
# ---------------------------------------------------------------------------


def test_round_date_daily():
    d = date(2024, 1, 15)
    assert _round_date(d, "daily") == d


def test_round_date_weekly():
    d = date(2024, 1, 15)
    result = _round_date(d, "weekly")
    assert result.weekday() == 0


def test_round_date_monthly():
    d = date(2024, 1, 15)
    assert _round_date(d, "monthly") == date(2024, 1, 1)


def test_round_date_yearly():
    d = date(2024, 6, 15)
    assert _round_date(d, "yearly") == date(2024, 1, 1)


def test_round_date_unknown():
    d = date(2024, 1, 15)
    assert _round_date(d, "unknown") == d


# ---------------------------------------------------------------------------
# _handle_stats_usage tests
# ---------------------------------------------------------------------------


def test_handle_stats_usage_no_records(capsys, tmp_path):
    args = _make_stats_args()
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "No usage logs found." in captured.out


def test_handle_stats_usage_with_records(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    log_data = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cached_tokens": 10,
        "cache_write_tokens": 5,
        "cost": 0.01,
    }
    (usage_dir / "20240115_120000_openai_test.log").write_text(
        json.dumps(log_data), encoding="utf-8"
    )
    args = _make_stats_args(to_date="2024-01-31", from_date="2024-01-01")
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    expected = (
        " [openai]\n"
        "----------------------------------------------------------------------------------\n"
        "  Period                Input       Output       Cached       CWrite           Cost\n"
        "  ----------------------------------------------------------------------------\n"
        "  2024-01-15               90           50           10            5  0.0100000000$\n"
        "  ----------------------------------------------------------------------------\n"
        "  TOTAL                    90           50           10            5  0.0100000000$\n\n"
    )
    assert captured.out == expected


def test_handle_stats_usage_date_filtered_out(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    log_data = {"prompt_tokens": 100}
    (usage_dir / "20230101_120000_openai_test.log").write_text(
        json.dumps(log_data), encoding="utf-8"
    )
    args = _make_stats_args(to_date="2024-01-31", from_date="2024-01-01")
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "No usage data found." in captured.out


def test_handle_stats_usage_with_from_date_is_none(capsys, tmp_path, monkeypatch):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    log_data = {"prompt_tokens": 100}
    (usage_dir / "20240115_120000_openai_test.log").write_text(
        json.dumps(log_data), encoding="utf-8"
    )
    args = _make_stats_args(to_date="2024-01-31")
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    expected = (
        " [openai]\n"
        "----------------------------------------------------------------------------------\n"
        "  Period                Input       Output       Cached       CWrite           Cost\n"
        "  ----------------------------------------------------------------------------\n"
        "  2024-01-15              100            0            0            0  0.0000000000$\n"
        "  ----------------------------------------------------------------------------\n"
        "  TOTAL                   100            0            0            0  0.0000000000$\n\n"
    )
    assert captured.out == expected


def test_handle_stats_usage_malformed_filename_skipped(capsys, tmp_path):
    """Malformed log filenames (non-YYYYMMDD prefix) should be skipped with a warning."""
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "bad_filename.log").write_text(
        json.dumps({"prompt_tokens": 100}), encoding="utf-8"
    )
    args = _make_stats_args(to_date="2024-01-31", from_date="2024-01-01")
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "No usage data found." in captured.out


def test_load_usage_logs_skips_bad_json_with_warning(tmp_path, caplog):
    """_load_usage_logs logs a warning when a log file has bad JSON."""
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_test_log.log").write_text(
        "bad json", encoding="utf-8"
    )
    result = _load_usage_logs(tmp_path)
    assert result == []
    assert "Skipping malformed/unreadable log file" in caplog.text
    assert str(usage_dir / "20240115_120000_test_log.log") in caplog.text


def test_load_usage_logs_oserror_with_warning(tmp_path, caplog, monkeypatch):
    """_load_usage_logs logs a warning on OSError."""
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    log_file = usage_dir / "20240115_120000_test_log.log"
    log_file.write_text("{}", encoding="utf-8")

    original_read_text = Path.read_text

    def broken_read(self, *a, **kw):
        if self.name.endswith(".log"):
            raise OSError("permission denied")
        return original_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", broken_read)
    result = _load_usage_logs(tmp_path)
    assert result == []
    assert "Skipping malformed/unreadable log file" in caplog.text
    assert str(log_file) in caplog.text


def test_maybe_ring_bell_no_ring(monkeypatch):
    """ring_bell=False should not ring the bell."""
    mock_meta = MagicMock()
    mock_meta.ring_bell = False
    monkeypatch.setattr("tiz.cli._is_tty", lambda: True)
    with patch("tiz.cli.sys.stdout.write") as mock_write:
        with patch("tiz.cli.sys.stdout.flush"):
            _maybe_ring_bell(mock_meta)
    mock_write.assert_not_called()


def test_maybe_ring_bell_not_tty(monkeypatch):
    """_maybe_ring_bell does nothing when not in a TTY."""
    mock_meta = MagicMock()
    mock_meta.ring_bell = True
    monkeypatch.setattr("tiz.cli._is_tty", lambda: False)
    with patch("tiz.cli.sys.stdout.write") as mock_write:
        with patch("tiz.cli.sys.stdout.flush"):
            _maybe_ring_bell(mock_meta)
    mock_write.assert_not_called()


def test_maybe_ring_bell_none_meta(monkeypatch):
    """_maybe_ring_bell does nothing when manifest_meta is None."""
    monkeypatch.setattr("tiz.cli._is_tty", lambda: True)
    with patch("tiz.cli.sys.stdout.write") as mock_write:
        with patch("tiz.cli.sys.stdout.flush"):
            _maybe_ring_bell(None)
    mock_write.assert_not_called()


def test_maybe_ring_bell_tty(monkeypatch):
    """_maybe_ring_bell uses _is_tty() consistently."""
    mock_meta = MagicMock()
    mock_meta.ring_bell = True
    monkeypatch.setattr("tiz.cli._is_tty", lambda: True)
    with patch("tiz.cli.sys.stdout.write") as mock_write:
        with patch("tiz.cli.sys.stdout.flush"):
            _maybe_ring_bell(mock_meta)
    mock_write.assert_called_with("\a")


def test_main_color_detection_is_tty(monkeypatch, capsys, tmp_path):
    """main() uses _is_tty() for color detection."""
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))
    monkeypatch.setattr("tiz.cli._is_tty", lambda: True)
    monkeypatch.setattr("tiz.cli.os.environ", {})

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 0


def test_handle_chat_parallelism_none_guard(monkeypatch):
    """_handle_chat should use parallelism=1 when manifest.meta.parallelism is None."""
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.parallelism = None

    monkeypatch.setattr("tiz.cli.helpers_chat", MagicMock(return_value=({}, None)))

    result = _handle_chat(mock_manifest, Path("/tmp"), task_name="mytask", context={})
    assert result == 0


# ---------------------------------------------------------------------------
# _parse_context tests
# ---------------------------------------------------------------------------


def test_parse_context_empty():
    assert _parse_context([]) == {}


def test_parse_context_single():
    assert _parse_context(["key=value"]) == {"key": "value"}


def test_parse_context_multiple():
    assert _parse_context(["a=1", "b=2"]) == {"a": "1", "b": "2"}


def test_parse_context_with_equals_in_value():
    assert _parse_context(["key=val=ue"]) == {"key": "val=ue"}


def test_parse_context_invalid_format():
    with pytest.raises(ValueError, match="invalid context format"):
        _parse_context(["invalid"])


# ---------------------------------------------------------------------------
# _print_usage tests
# ---------------------------------------------------------------------------


def test_print_usage_empty(capsys):
    _print_usage({})
    captured = capsys.readouterr()
    expected = (
        "\n\n"
        "Usage: input:\t0 (0.0 tk/s)\n"
        "      output:\t0 (0.0 tk/s)\n"
        "      cached:\t0\n"
        "      cwrite:\t0\n"
        "Credits spent:\t0.0000000000 $\n"
        "Tools usage:\n"
    )
    assert captured.out == expected


def test_print_usage_with_data(capsys):
    result = {
        "task1": {
            "prompt_tokens": 100,
            "prompt_time": 2.0,
            "completion_tokens": 50,
            "completion_time": 1.0,
            "cached_tokens": 10,
            "cache_write_tokens": 5,
            "cost": 0.015,
            "tool_calls": [("bash", {}), ("bash", {}), ("read", {})],
        }
    }
    _print_usage(result, running_time="00:00:05")
    captured = capsys.readouterr()
    expected = (
        "\n\n"
        "Running time:\t00:00:05\n"
        "Usage: input:\t90 (45.0 tk/s)\n"
        "      output:\t50 (50.0 tk/s)\n"
        "      cached:\t10\n"
        "      cwrite:\t5\n"
        "Credits spent:\t0.0150000000 $\n"
        "Tools usage:\n"
        "  bash: 2\n"
        "  read: 1\n"
    )
    assert captured.out == expected


def test_print_usage_without_running_time(capsys):
    _print_usage({})
    captured = capsys.readouterr()
    assert "Running time:" not in captured.out


def test_print_usage_zero_times(capsys):
    result = {
        "task1": {
            "prompt_tokens": 100,
            "prompt_time": 0,
            "completion_tokens": 50,
            "completion_time": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "cost": 0.0,
            "tool_calls": [],
        }
    }
    _print_usage(result)
    captured = capsys.readouterr()
    expected = (
        "\n\n"
        "Usage: input:\t100 (0.0 tk/s)\n"
        "      output:\t50 (0.0 tk/s)\n"
        "      cached:\t0\n"
        "      cwrite:\t0\n"
        "Credits spent:\t0.0000000000 $\n"
        "Tools usage:\n"
    )
    assert captured.out == expected


# ---------------------------------------------------------------------------
# _handle_chat tests
# ---------------------------------------------------------------------------


def test_handle_chat_success(monkeypatch):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"

    monkeypatch.setattr("tiz.cli.helpers_chat", MagicMock(return_value=({}, None)))

    result = _handle_chat(mock_manifest, Path("/tmp"), task_name="mytask", context={})
    assert result == 0


def test_handle_chat_error(monkeypatch):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"

    monkeypatch.setattr(
        "tiz.cli.helpers_chat", MagicMock(return_value=({}, "chat error"))
    )

    result = _handle_chat(mock_manifest, Path("/tmp"))
    assert result == 1


# ---------------------------------------------------------------------------
# _handle_exec tests
# ---------------------------------------------------------------------------


def test_handle_exec_success(monkeypatch):
    mock_manifest = MagicMock()
    mock_exec = MagicMock(return_value=None)
    monkeypatch.setattr("tiz.cli.exec_cmd", mock_exec)
    result = _handle_exec(mock_manifest, Path("/tmp"), "mytask", ["cmd"])
    assert result == 0
    mock_exec.assert_called_once_with(
        manifest=mock_manifest,
        base_path=Path("/tmp"),
        task_name="mytask",
        cmd_args=["cmd"],
        extra_run_args=None,
    )


def test_handle_exec_error(monkeypatch):
    mock_manifest = MagicMock()
    mock_exec = MagicMock(return_value="some error")
    monkeypatch.setattr("tiz.cli.exec_cmd", mock_exec)
    result = _handle_exec(mock_manifest, Path("/tmp"), "mytask", ["cmd"])
    assert result == 1
    mock_exec.assert_called_once_with(
        manifest=mock_manifest,
        base_path=Path("/tmp"),
        task_name="mytask",
        cmd_args=["cmd"],
        extra_run_args=None,
    )


def test_handle_exec_with_extra_run_args(monkeypatch):
    mock_manifest = MagicMock()
    mock_exec = MagicMock(return_value=None)
    monkeypatch.setattr("tiz.cli.exec_cmd", mock_exec)
    extra = ["--cpus=2", "--memory=512m"]
    result = _handle_exec(
        mock_manifest, Path("/tmp"), "mytask", ["cmd"], extra_run_args=extra
    )
    assert result == 0
    mock_exec.assert_called_once_with(
        manifest=mock_manifest,
        base_path=Path("/tmp"),
        task_name="mytask",
        cmd_args=["cmd"],
        extra_run_args=extra,
    )


# ---------------------------------------------------------------------------
# _build_web_parser tests
# ---------------------------------------------------------------------------


def test_build_web_parser():

    main_parser = argparse.ArgumentParser()
    main_subparsers = main_parser.add_subparsers(dest="command")
    web_parser = _build_web_parser(main_subparsers)
    assert web_parser is not None

    parsed = main_parser.parse_args(["web", "config.yaml"])
    assert parsed.command == "web"
    assert parsed.config_path == Path("config.yaml")
    assert parsed.host == "localhost"
    assert parsed.port == 8080
    assert parsed.socket_path is None

    parsed = main_parser.parse_args(
        ["web", "config.yaml", "--host", "0.0.0.0", "--port", "9000"]
    )
    assert parsed.command == "web"
    assert parsed.config_path == Path("config.yaml")
    assert parsed.host == "0.0.0.0"
    assert parsed.port == 9000
    assert parsed.socket_path is None

    parsed = main_parser.parse_args(
        ["web", "config.yaml", "--socket-path", "/tmp/tiz.sock"]
    )
    assert parsed.command == "web"
    assert parsed.socket_path == Path("/tmp/tiz.sock")
    assert parsed.host == "localhost"
    assert parsed.port == 8080


def test_handle_web(monkeypatch):
    mock_run_simple = MagicMock()
    monkeypatch.setattr("tiz.cli.web_run_simple", mock_run_simple)

    args = MagicMock()
    args.config_path = Path("config.yaml")
    args.host = "0.0.0.0"
    args.port = 9000
    args.socket_path = None
    base_path = Path("/tmp")

    result = _handle_web(args, base_path)
    assert result == 0
    mock_run_simple.assert_called_once_with(
        base_path=base_path,
        config_path=Path("config.yaml"),
        host="0.0.0.0",
        port=9000,
        path=None,
    )


def test_handle_web_with_socket_path(monkeypatch):
    mock_run_simple = MagicMock()
    monkeypatch.setattr("tiz.cli.web_run_simple", mock_run_simple)

    args = MagicMock()
    args.config_path = Path("config.yaml")
    args.host = "localhost"
    args.port = 8080
    args.socket_path = Path("/tmp/tiz.sock")
    base_path = Path("/tmp")

    result = _handle_web(args, base_path)
    assert result == 0
    mock_run_simple.assert_called_once_with(
        base_path=base_path,
        config_path=Path("config.yaml"),
        host="localhost",
        port=8080,
        path=Path("/tmp/tiz.sock"),
    )


def test_main_web(monkeypatch):
    mock_run_simple = MagicMock()
    monkeypatch.setattr("tiz.cli.web_run_simple", mock_run_simple)

    with patch.object(sys, "argv", ["tiz", "-c", "/tmp", "web", "config.yaml"]):
        result = main()
    assert result == 0
    mock_run_simple.assert_called_once_with(
        base_path=Path("/tmp"),
        config_path=Path("config.yaml"),
        host="localhost",
        port=8080,
        path=None,
    )


def test_main_web_with_host_port(monkeypatch):
    mock_run_simple = MagicMock()
    monkeypatch.setattr("tiz.cli.web_run_simple", mock_run_simple)

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            "/tmp",
            "web",
            "config.yaml",
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
        ],
    ):
        result = main()
    assert result == 0
    mock_run_simple.assert_called_once_with(
        base_path=Path("/tmp"),
        config_path=Path("config.yaml"),
        host="0.0.0.0",
        port=9999,
        path=None,
    )


def test_main_web_with_socket_path(monkeypatch):
    mock_run_simple = MagicMock()
    monkeypatch.setattr("tiz.cli.web_run_simple", mock_run_simple)

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            "/tmp",
            "web",
            "config.yaml",
            "--socket-path",
            "/tmp/tiz.sock",
        ],
    ):
        result = main()
    assert result == 0
    mock_run_simple.assert_called_once_with(
        base_path=Path("/tmp"),
        config_path=Path("config.yaml"),
        host="localhost",
        port=8080,
        path=Path("/tmp/tiz.sock"),
    )


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------
# get_parser tests
# ---------------------------------------------------------------------------


def test_get_parser_returns_parser():

    parser = get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    # Verify subcommands exist
    parsed = parser.parse_args(["chat", "-m", "test.yaml"])
    assert parsed.command == "chat"
    parsed = parser.parse_args(["run", "-m", "test.yaml"])
    assert parsed.command == "run"
    parsed = parser.parse_args(["exec", "-m", "test.yaml"])
    assert parsed.command == "exec"
    parsed = parser.parse_args(["sb", "list"])
    assert parsed.command == "sb"
    assert parsed.sb_command == "list"
    parsed = parser.parse_args(["stats", "usage"])
    assert parsed.command == "stats"
    assert parsed.stats_command == "usage"
    parsed = parser.parse_args(["completion"])
    assert parsed.command == "completion"
    parsed = parser.parse_args(["web", "config.yaml"])
    assert parsed.command == "web"
    assert parsed.config_path == Path("config.yaml")
    assert parsed.host == "localhost"
    assert parsed.port == 8080
    assert parsed.socket_path is None
    parsed = parser.parse_args(
        ["web", "config.yaml", "--host", "0.0.0.0", "--port", "9000"]
    )
    assert parsed.command == "web"
    assert parsed.config_path == Path("config.yaml")
    assert parsed.host == "0.0.0.0"
    assert parsed.port == 9000
    assert parsed.socket_path is None
    parsed = parser.parse_args(["web", "config.yaml", "--socket-path", "/tmp/tiz.sock"])
    assert parsed.command == "web"
    assert parsed.socket_path == Path("/tmp/tiz.sock")
    assert parser._sb_parser is not None


# ---------------------------------------------------------------------------


def test_main_no_command(capsys):
    with patch.object(sys, "argv", ["tiz"]):
        result = main()
    assert result == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_main_sb_list(capsys, monkeypatch):
    mock_manager = MagicMock()
    mock_manager.available_engine.return_value = "docker"
    manager_cls = MagicMock(return_value=mock_manager)
    monkeypatch.setattr("tiz.cli.SandboxManager", manager_cls)
    mock_manager.list_sandboxes.return_value = ["sb1"]

    with patch.object(sys, "argv", ["tiz", "-c", "/tmp", "sb", "list"]):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "sb1"


def test_main_stats(capsys, monkeypatch):
    with patch.object(sys, "argv", ["tiz", "-c", "/tmp", "stats", "usage"]):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "No usage logs found." in captured.out


def test_main_unknown_command(capsys):
    with patch.object(sys, "argv", ["tiz", "unknown"]), pytest.raises(SystemExit):
        main()


def test_main_run_no_manifest(capsys):
    with patch.object(sys, "argv", ["tiz", "run"]), pytest.raises(SystemExit):
        main()


def test_main_run_success(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


def test_main_run_keyboard_interrupt(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(side_effect=KeyboardInterrupt))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 1
    captured = capsys.readouterr()
    assert "Error: interrupted" in captured.err


def test_main_run_manifest_error(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(None, "parse error"))
    )

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 1
    captured = capsys.readouterr()
    assert "parse error" in captured.err


def test_main_chat_success(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.helpers_chat", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "chat", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out
    assert "Credits spent:" in captured.out


def test_main_chat_with_context(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.helpers_chat", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "chat",
            "-m",
            str(manifest_file),
            "--context",
            "key=value",
        ],
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out
    assert "Credits spent:" in captured.out


def test_main_exec_success(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.exec_cmd", MagicMock(return_value=None))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "exec",
            "-m",
            str(manifest_file),
        ],
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_main_exec_with_cmd_args(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.exec_cmd", MagicMock(return_value=None))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "exec",
            "-m",
            str(manifest_file),
            "--",
            "echo",
            "hello",
        ],
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_main_exec_with_extra_run_args(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_exec = MagicMock(return_value=None)
    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.exec_cmd", mock_exec)

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "exec",
            "-m",
            str(manifest_file),
            "--extra-run-args=--cpus=2",
            "--extra-run-args=--memory=512m",
            "--",
            "bash",
        ],
    ):
        result = main()
    assert result == 0
    mock_exec.assert_called_once()
    call_kwargs = mock_exec.call_args[1]
    assert call_kwargs["extra_run_args"] == ["--cpus=2", "--memory=512m"]


def test_parser_exec_extra_run_args():
    parser = get_parser()
    parsed = parser.parse_args(
        [
            "exec",
            "-m",
            "test.yaml",
            "--extra-run-args=--cpus=2",
            "--extra-run-args=--memory=512m",
            "--",
            "bash",
        ]
    )
    assert parsed.command == "exec"
    assert parsed.extra_run_args == ["--cpus=2", "--memory=512m"]
    assert parsed.cmd_args == ["--", "bash"]


def test_parser_exec_extra_run_args_not_set():
    parser = get_parser()
    parsed = parser.parse_args(
        [
            "exec",
            "-m",
            "test.yaml",
        ]
    )
    assert parsed.command == "exec"
    assert parsed.extra_run_args is None
    assert parsed.cmd_args == []


def test_main_exec_with_extra_run_args_shell_split(capsys, monkeypatch, tmp_path):
    """Test that extra-run-args with spaces are shell-split into individual args."""
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_exec = MagicMock(return_value=None)
    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.exec_cmd", mock_exec)

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    # Single --extra-run-args with multiple flags in one string
    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "exec",
            "-m",
            str(manifest_file),
            "--extra-run-args=--cpus=2 --memory=512m",
        ],
    ):
        result = main()
    assert result == 0
    mock_exec.assert_called_once()
    call_kwargs = mock_exec.call_args[1]
    assert call_kwargs["extra_run_args"] == ["--cpus=2", "--memory=512m"]


def test_main_exec_with_extra_run_args_shell_split_quoted(
    capsys, monkeypatch, tmp_path
):
    """Test that quoted args in extra-run-args are handled correctly after shell split."""
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_exec = MagicMock(return_value=None)
    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.exec_cmd", mock_exec)

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    # Shell-split with quoted value containing spaces
    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "exec",
            "-m",
            str(manifest_file),
            '--extra-run-args=--label="my label with spaces"',
            "--extra-run-args=--cpus=2",
        ],
    ):
        result = main()
    assert result == 0
    mock_exec.assert_called_once()
    call_kwargs = mock_exec.call_args[1]
    assert call_kwargs["extra_run_args"] == [
        "--label=my label with spaces",
        "--cpus=2",
    ]


def test_main_exec_with_extra_run_args_none(capsys, monkeypatch, tmp_path):
    """Test that extra_run_args=None is passed through correctly (not split)."""
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_exec = MagicMock(return_value=None)
    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.exec_cmd", mock_exec)

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    # No --extra-run-args provided at all
    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "exec",
            "-m",
            str(manifest_file),
            "--",
            "bash",
        ],
    ):
        result = main()
    assert result == 0
    mock_exec.assert_called_once()
    call_kwargs = mock_exec.call_args[1]
    assert call_kwargs["extra_run_args"] is None


def test_main_run_context_parse_error(capsys, monkeypatch, tmp_path):
    """Context parse error is reported."""
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "run",
            "-m",
            str(manifest_file),
            "--context",
            "invalid",
        ],
    ):
        result = main()
    assert result == 1
    captured = capsys.readouterr()
    assert "invalid context format" in captured.err


def test_main_color_detection_no_color_env(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = False
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))
    monkeypatch.setattr("tiz.cli.os.environ", {"NO_COLOR": "1"})

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


def test_main_verbose_1(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 1
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "-vv", "run", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


# ---------------------------------------------------------------------------
# Additional coverage tests for remaining uncovered lines
# ---------------------------------------------------------------------------


def test_path_completer_complete_no_text():
    c = _PathCompleter()
    result = c.complete("", 0)
    # In the current directory, there will be entries
    assert isinstance(result, str)


def test_path_completer_complete_oserror(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", MagicMock())
    c = _PathCompleter()
    c.complete("", 0)
    # Should not crash - exhausted matches
    result = c.complete("", 99)
    assert result is None


def test_path_completer_oserror(tmp_path):
    c = _PathCompleter()
    result = c.complete("/nonexistent/path", 0)
    assert len(c._matches) == 0
    assert result is None


def test_path_completer_complete_state_out_of_range(tmp_path):
    c = _PathCompleter()
    c._matches = ["a", "b"]
    result = c.complete("", 5)
    assert result is None


def test_chat_completer_no_slash():
    c = _ChatCompleter()
    result = c.complete("hello", 0)
    assert result is None


def test_chat_completer_attach_command():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/attach "
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("", 0)
        assert result is None or isinstance(result, str)


def test_chat_completer_slash_command_state_out_of_range():
    c = _ChatCompleter()
    result = c.complete("/help", 99)
    assert result is None


def test_chat_completer_attach_with_path():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/attach "
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("", 0)
        assert result is None or isinstance(result, str)


def test_chat_completer_dir_with_entry(tmp_path):
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/attach /tmp"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("/tm", 0)
        assert result is None or isinstance(result, str)


def test_main_if_name_main(tmp_path):
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(sys, "argv", ["tiz", "-c", str(tmp_path), "list"]):
        with pytest.raises(SystemExit):
            main()


def test_handle_sb_kill_all_not_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    mock_manager.list_sandboxes.return_value = ["sb1"]
    monkeypatch.setattr("builtins.input", lambda _: "n")
    args = _make_sb_args(sb_command="kill", sandbox_name=None, do_not_ask=False)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.kill_all_containers.assert_not_called()


def test_handle_sb_empty_command_after_none(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    args = _make_sb_args(sb_command="nonexistent")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0


def test_handle_sb_rm_all_not_confirmed(monkeypatch):
    mock_manager, _ = _patch_sb(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    args = _make_sb_args(sb_command="rm-all", do_not_ask=False)
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    mock_manager.kill_and_delete_all_sandboxes.assert_not_called()


def test_handle_sb_exception(capsys, monkeypatch):
    """Test that exceptions inside _handle_sb are caught and formatted."""
    monkeypatch.setattr(
        "tiz.cli.SandboxManager.available_engine",
        MagicMock(side_effect=ValueError("boom")),
    )
    args = _make_sb_args(sb_command="list")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 1
    captured = capsys.readouterr()
    assert "Error: boom" in captured.err


def test_chat_completer_attach_with_path_completion():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/attach /some/path"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("/some/", 0)
        assert result is None


def test_chat_completer_command_matching():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/help"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("/he", 0)
        assert result == "/help"
        result = c.complete("/he", 99)
        assert result is None


def test_chat_completer_record_command():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/help"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("/re", 0)
        assert result is not None
        assert result in ("/record", "/replay")


def test_chat_completer_record_exact():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/help"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("/record", 0)
        assert result == "/record"


def test_chat_completer_record_state_out_of_range():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/help"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("/re", 99)
        assert result is None


def test_chat_completer_try_attach_path_completer_creation():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/attach /tmp"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        c._path_completer = None
        result = c.complete("/tm", 0)
        assert result is None or isinstance(result, str)


def test_chat_completer_attach_path_completer_reuse():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/attach /tmp"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        c._path_completer = _PathCompleter()
        result = c.complete("/tm", 0)
        assert result is None or isinstance(result, str)


def test_path_completer_prefix_no_match(tmp_path):
    c = _PathCompleter()
    (tmp_path / "abc.txt").write_text("")
    (tmp_path / "def.txt").write_text("")
    result = c.complete(str(tmp_path / "xyz"), 0)
    assert result is None
    assert len(c._matches) == 0


def test_path_completer_with_dir_match(tmp_path):
    c = _PathCompleter()
    (tmp_path / "subdir").mkdir()
    result = c.complete(str(tmp_path / "subd"), 0)
    assert result == str(tmp_path / "subdir") + "/"
    assert len(c._matches) == 1


def test_path_completer_with_file_match(tmp_path):
    c = _PathCompleter()
    (tmp_path / "file.txt").write_text("")
    result = c.complete(str(tmp_path / "f"), 0)
    assert result == str(tmp_path / "file.txt")


def test_path_completer_trailing_slash(tmp_path):
    c = _PathCompleter()
    # Pass a root-like path ending with separator
    result = c.complete("/", 0)
    # Should not crash; / exists on all systems and has entries
    assert isinstance(result, str)
    assert result.endswith("/")
    assert len(c._matches) > 0


def test_path_completer_home_expansion(monkeypatch):
    monkeypatch.setattr("tiz.cli.readline", MagicMock())
    c = _PathCompleter()
    result = c.complete("~/nonexistent_path_xyz", 0)
    # Should not crash - ~ is expanded
    assert len(c._matches) == 0
    assert result is None


def test_print_usage_no_tool_calls(capsys):
    result = {
        "task1": {
            "prompt_tokens": 100,
            "prompt_time": 2.0,
            "completion_tokens": 50,
            "completion_time": 1.0,
            "cached_tokens": 10,
            "cache_write_tokens": 5,
            "cost": 0.015,
        }
    }
    _print_usage(result)
    captured = capsys.readouterr()
    assert "Tools usage:" in captured.out


def test_handle_stats_usage_multiple_records_same_engine(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_test1.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    (usage_dir / "20240116_120000_openai_test2.log").write_text(
        json.dumps({"prompt_tokens": 200, "completion_tokens": 100, "cost": 0.02})
    )
    args = _make_stats_args(to_date="2024-01-31", from_date="2024-01-01")
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    expected = (
        " [openai]\n"
        "----------------------------------------------------------------------------------\n"
        "  Period                Input       Output       Cached       CWrite           Cost\n"
        "  ----------------------------------------------------------------------------\n"
        "  2024-01-15              100           50            0            0  0.0100000000$\n"
        "  2024-01-16              200          100            0            0  0.0200000000$\n"
        "  ----------------------------------------------------------------------------\n"
        "  TOTAL                   300          150            0            0  0.0300000000$\n\n"
    )
    assert captured.out == expected


def test_handle_stats_usage_same_date_multiple_records(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_test1.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    (usage_dir / "20240115_140000_openai_test2.log").write_text(
        json.dumps({"prompt_tokens": 200, "completion_tokens": 100, "cost": 0.02})
    )
    args = _make_stats_args(to_date="2024-01-31", from_date="2024-01-01")
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    expected = (
        " [openai]\n"
        "----------------------------------------------------------------------------------\n"
        "  Period                Input       Output       Cached       CWrite           Cost\n"
        "  ----------------------------------------------------------------------------\n"
        "  2024-01-15              300          150            0            0  0.0300000000$\n"
        "  ----------------------------------------------------------------------------\n"
        "  TOTAL                   300          150            0            0  0.0300000000$\n\n"
    )
    assert captured.out == expected


def test_main_color_flag_override(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "--no-color", "run", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


def test_main_no_color_flag_tty_no_env(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("tiz.cli.os.environ", {})

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


def test_chat_completer_non_attach_with_space_in_line():
    mock_readline = MagicMock()
    mock_readline.get_line_buffer.return_value = "/clear somearg"
    with patch("tiz.cli.readline", mock_readline):
        c = _ChatCompleter()
        result = c.complete("/cle", 0)
        assert result == "/clear"


def test_handle_stats_usage_group_by_task(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_mytask.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    args = _make_stats_args(
        to_date="2024-01-31", from_date="2024-01-01", group_by_task=True
    )
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "[mytask]" in captured.out
    assert "100" in captured.out


def test_handle_stats_usage_engine_filter(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_mytask.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    (usage_dir / "20240115_120000_anthropic_mytask.log").write_text(
        json.dumps({"prompt_tokens": 200, "completion_tokens": 100, "cost": 0.02})
    )
    args = _make_stats_args(
        to_date="2024-01-31", from_date="2024-01-01", engine_filter=["openai"]
    )
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "[openai]" in captured.out
    assert "[anthropic]" not in captured.out
    assert "100" in captured.out


def test_handle_stats_usage_task_filter(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_mytask.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    (usage_dir / "20240115_120000_openai_other.log").write_text(
        json.dumps({"prompt_tokens": 200, "completion_tokens": 100, "cost": 0.02})
    )
    args = _make_stats_args(
        to_date="2024-01-31", from_date="2024-01-01", task_filter=["mytask"]
    )
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "100" in captured.out
    assert "200" not in captured.out


def test_handle_stats_usage_engine_filter_regexp(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_mytask.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    (usage_dir / "20240115_120000_anthropic_mytask.log").write_text(
        json.dumps({"prompt_tokens": 200, "completion_tokens": 100, "cost": 0.02})
    )
    (usage_dir / "20240115_120000_oai_mytask.log").write_text(
        json.dumps({"prompt_tokens": 50, "completion_tokens": 25, "cost": 0.005})
    )
    args = _make_stats_args(
        to_date="2024-01-31", from_date="2024-01-01", engine_filter=[".*open.*"]
    )
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "[openai]" in captured.out
    assert "[anthropic]" not in captured.out
    assert "[oai]" not in captured.out


def test_handle_stats_usage_engine_filter_regexp_no_match(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_mytask.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    args = _make_stats_args(
        to_date="2024-01-31", from_date="2024-01-01", engine_filter=["unknown"]
    )
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "No usage data found." in captured.out


def test_handle_stats_usage_task_filter_regexp(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_review_task.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    (usage_dir / "20240115_120000_openai_review.log").write_text(
        json.dumps({"prompt_tokens": 200, "completion_tokens": 60, "cost": 0.02})
    )
    (usage_dir / "20240115_120000_openai_build.log").write_text(
        json.dumps({"prompt_tokens": 900, "completion_tokens": 800, "cost": 0.09})
    )
    args = _make_stats_args(
        to_date="2024-01-31", from_date="2024-01-01", task_filter=["review.*"]
    )
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    # only review_task (100) and review (200) match -> 300 prompt, 110 completion
    assert "300" in captured.out
    assert "110" in captured.out
    assert "900" not in captured.out
    assert "800" not in captured.out


def test_handle_stats_usage_task_filter_regexp_no_match(capsys, tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    (usage_dir / "20240115_120000_openai_mytask.log").write_text(
        json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
    )
    args = _make_stats_args(
        to_date="2024-01-31", from_date="2024-01-01", task_filter=["no_match"]
    )
    result = _handle_stats_usage(args, tmp_path)
    assert result == 0
    captured = capsys.readouterr()
    assert "No usage data found." in captured.out


def test_main_no_color_notty(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("tiz.cli.os.environ", {})

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


def test_handle_sb_build_all_skip_non_containerfile(capsys, monkeypatch, tmp_path):
    mock_manager, _ = _patch_sb(monkeypatch)
    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir()
    (cf_dir / "Containerfile.valid").write_text("FROM base")
    (cf_dir / "not_containerfile.txt").write_text("skip")
    _mock_get_data_dirs(monkeypatch, containerfiles_dirs=[cf_dir])
    args = _make_sb_args(sb_command="build", build_command="all")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "Built image 'valid:latest'" in captured.out


def test_handle_sb_build_all_skip_non_dir(capsys, monkeypatch, tmp_path):
    mock_manager, _ = _patch_sb(monkeypatch)
    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir()
    non_dir = tmp_path / "not_a_dir"
    non_dir.write_text("file")
    _mock_get_data_dirs(monkeypatch, containerfiles_dirs=[cf_dir, non_dir])
    (cf_dir / "Containerfile.foo").write_text("FROM x")
    args = _make_sb_args(sb_command="build", build_command="all")
    result = _handle_sb(args, Path("/tmp"), MagicMock())
    assert result == 0
    captured = capsys.readouterr()
    assert "Built image 'foo:latest'" in captured.out


def test_main_color_detection_no_color_env_false(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = False
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))
    monkeypatch.setattr("tiz.cli.sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("tiz.cli.os.environ", {})

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


def test_main_summarizer_context_ratio_flag(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "--summarizer-context-ratio",
            "0.5",
            "run",
            "-m",
            str(manifest_file),
        ],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs["options"]["meta"]["summarizer_context_ratio"] == 0.5


def test_main_summarizer_context_ratio_unchanged_when_not_set(
    capsys, monkeypatch, tmp_path
):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert "summarizer_context_ratio" not in call_kwargs["options"]["meta"]


def test_main_use_host_timezone_flag(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "--use-host-timezone",
            "run",
            "-m",
            str(manifest_file),
        ],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs["options"]["meta"]["use_host_timezone"] is True


def test_main_no_use_host_timezone_flag(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "--no-use-host-timezone",
            "run",
            "-m",
            str(manifest_file),
        ],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs["options"]["meta"]["use_host_timezone"] is False


def test_main_use_host_timezone_unchanged_when_not_set(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert "use_host_timezone" not in call_kwargs["options"]["meta"]


def test_main_verbose_gt_2(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 2
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "-vvv", "run", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


def test_main_delete_sandbox_on_exit_flag(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "--delete-sandbox-on-exit",
            "run",
            "-m",
            str(manifest_file),
        ],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs["options"]["meta"]["delete_sandbox_on_exit"] is True


def test_main_no_delete_sandbox_on_exit_flag(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "--no-delete-sandbox-on-exit",
            "run",
            "-m",
            str(manifest_file),
        ],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs["options"]["meta"]["delete_sandbox_on_exit"] is False


def test_main_delete_sandbox_on_exit_unchanged_when_not_set(
    capsys, monkeypatch, tmp_path
):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert "delete_sandbox_on_exit" not in call_kwargs["options"]["meta"]


def test_main_ephemeral_sandbox_flag(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "--ephemeral-sandbox",
            "run",
            "-m",
            str(manifest_file),
        ],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs["options"]["meta"]["ephemeral_sandbox"] is True


def test_main_no_ephemeral_sandbox_flag(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "tiz",
            "-c",
            str(tmp_path),
            "--no-ephemeral-sandbox",
            "run",
            "-m",
            str(manifest_file),
        ],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs["options"]["meta"]["ephemeral_sandbox"] is False


def test_main_ephemeral_sandbox_unchanged_when_not_set(capsys, monkeypatch, tmp_path):
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 1
    mock_manifest.tasks = [MagicMock()]

    mock_parse = MagicMock(return_value=(mock_manifest, None))
    monkeypatch.setattr("tiz.cli.parse_manifest", mock_parse)
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)],
    ):
        result = main()
    assert result == 0
    call_kwargs = mock_parse.call_args[1]
    assert "ephemeral_sandbox" not in call_kwargs["options"]["meta"]


def test_readline_import_error_and_module_level_branch():
    import subprocess
    import sys as _sys

    code = """
import sys
sys.modules['readline'] = None
for mod in list(sys.modules.keys()):
    if 'tiz' in mod:
        del sys.modules[mod]
from tiz.cli import _parse_hex_color
assert _parse_hex_color('#ff0000') == '255;0;0'
print('OK')
"""
    p = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
        cwd="/opt/project",
    )
    assert p.returncode == 0, f"stdout: {p.stdout}, stderr: {p.stderr}"


def test_main_parallelism_override_with_confirmations(capsys, monkeypatch, tmp_path):
    """When parallelism > 1 and tools have confirmations, parallelism is forced to 1."""
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 2
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.ask_confirmations", MagicMock(return_value=True))
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 0
    assert mock_manifest.meta.parallelism == 1
    captured = capsys.readouterr()
    assert "Parallelism forced to 1" in captured.err
    assert "Running time:" in captured.out
    assert "Usage:" in captured.out


def test_main_parallelism_gt_1_no_confirmations_preserved(
    capsys, monkeypatch, tmp_path
):
    """When parallelism > 1 but no tools have confirmations, parallelism is preserved."""
    mock_manifest = MagicMock()
    mock_manifest.meta.hide_reasoning = False
    mock_manifest.meta.color_reasoning = "#758182"
    mock_manifest.meta.color = True
    mock_manifest.meta.color_input = "#ff00ff"
    mock_manifest.meta.verbosity = 0
    mock_manifest.meta.parallelism = 2
    mock_manifest.tasks = [MagicMock()]

    monkeypatch.setattr(
        "tiz.cli.parse_manifest", MagicMock(return_value=(mock_manifest, None))
    )
    monkeypatch.setattr("tiz.cli.ask_confirmations", MagicMock(return_value=False))
    monkeypatch.setattr("tiz.cli.run", MagicMock(return_value=({}, None)))

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text("meta:\n  version: '0'", encoding="utf-8")

    with patch.object(
        sys, "argv", ["tiz", "-c", str(tmp_path), "run", "-m", str(manifest_file)]
    ):
        result = main()
    assert result == 0
    assert mock_manifest.meta.parallelism == 2
    captured = capsys.readouterr()
    assert "Warning: parallelism forced to 1" not in captured.err


def test_main_entry_point_block():
    import subprocess
    import sys as _sys

    code = """
import sys
sys.argv = ['tiz']
import tiz.cli
with open('/dev/null', 'w') as f:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = f
    rc = tiz.cli.main()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    sys.exit(rc)
"""
    p = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src", "NO_COLOR": "1"},
        cwd="/opt/project",
    )
    assert p.returncode == 1
