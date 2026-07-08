# mypy: ignore-errors
"""Tests for sandbox_worker tool handlers."""

import contextlib
import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tiz.sandbox_worker

_RealSocket = socket.socket

# ---- run_tool ----


def test_run_tool_unknown_tool() -> None:
    result, is_error = tiz.sandbox_worker.run_tool("NonExistent", {})
    assert "Unknown tool" in result
    assert is_error is True


def test_run_tool_known_tool() -> None:
    result, is_error = tiz.sandbox_worker.run_tool("Bash", {"command": "echo hello"})
    assert "hello" in result
    assert is_error is False


def test_run_tool_handler_exception() -> None:
    original = dict(tiz.sandbox_worker.HANDLERS)  # type: ignore[attr-defined]
    try:
        tiz.sandbox_worker.HANDLERS["Bad"] = lambda _: 1 / 0  # type: ignore
        result, is_error = tiz.sandbox_worker.run_tool("Bad", {})
        assert "Tool execution failed unexpectedly" in result
        assert is_error is True
    finally:
        tiz.sandbox_worker.HANDLERS.clear()
        tiz.sandbox_worker.HANDLERS.update(original)


# ---- _check_has_rg ----


def test_check_has_rg_cached() -> None:
    sw = tiz.sandbox_worker
    original = sw._HAS_RG
    try:
        sw._HAS_RG = True
        assert sw._check_has_rg() is True
        sw._HAS_RG = False
        assert sw._check_has_rg() is False
    finally:
        sw._HAS_RG = original


def test_check_has_rg_file_not_found() -> None:
    sw = tiz.sandbox_worker
    original = sw._HAS_RG
    sw._HAS_RG = None
    try:
        with patch("subprocess.run", side_effect=FileNotFoundError("rg not found")):
            result = sw._check_has_rg()
        assert result is False
    finally:
        sw._HAS_RG = original


# ---- _tool_bash ----


def test_bash_success() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash({"command": "echo hello"})
    assert "hello" in result
    assert is_error is False


def test_bash_failure() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash({"command": "exit 42"})
    assert "exit code: 42" in result
    assert is_error is True


def test_bash_timeout() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash(
        {"command": "sleep 10", "timeout": 1}
    )
    assert "timed out" in result
    assert is_error is True


def test_bash_with_cwd(tmp_path: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_bash(
        {"command": "pwd", "cwd": str(tmp_path)}
    )
    assert str(tmp_path) in result
    assert is_error is False


def test_bash_with_env() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash(
        {"command": "echo $MYVAR", "env": {"MYVAR": "test123"}}
    )
    assert "test123" in result
    assert is_error is False


def test_bash_default_timeout() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash({"command": "echo ok"})
    assert "ok" in result
    assert is_error is False


def test_bash_timeout_clamped() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash(
        {"command": "echo ok", "timeout": 0}
    )
    assert "ok" in result
    assert is_error is False


def test_bash_cwd_not_found() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash(
        {"command": "echo hi", "cwd": "/nonexistent_dir_xyz"}
    )
    assert "ERROR: cwd not found" in result
    assert is_error is True


def test_bash_env_not_dict() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash(
        {"command": "echo hi", "env": "not-a-dict"}
    )
    assert "ERROR: env must be a dict" in result
    assert is_error is True


def test_bash_env_value_not_string() -> None:
    result, is_error = tiz.sandbox_worker._tool_bash(
        {"command": "echo hi", "env": {"MYVAR": 123}}
    )
    assert "ERROR: env value for 'MYVAR' must be a string" in result
    assert is_error is True


# ---- _tool_cargo_fetch ----


def test_cargo_fetch_success(tmp_path: Path) -> None:
    project_dir = tmp_path / "rust_project"
    project_dir.mkdir()
    cargo_toml = project_dir / "Cargo.toml"
    cargo_toml.write_text(
        '[package]\nname = "test_project"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_cargo_fetch(
            {"path": str(project_dir)}
        )
    assert is_error is False
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "cargo"
    assert call_args[1] == "fetch"


def test_cargo_fetch_missing_path() -> None:
    result, is_error = tiz.sandbox_worker._tool_cargo_fetch({})
    assert "ERROR: path is required" in result
    assert is_error is True


def test_cargo_fetch_no_cargo_toml(tmp_path: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_cargo_fetch({"path": str(tmp_path)})
    assert "Cargo.toml not found" in result
    assert is_error is True


def test_cargo_fetch_failure(tmp_path: Path) -> None:
    project_dir = tmp_path / "rust_project"
    project_dir.mkdir()
    cargo_toml = project_dir / "Cargo.toml"
    cargo_toml.write_text(
        '[package]\nname = "test"\nversion = "0.1.0"\ndependencies = { nonexistent = "999" }\n',
        encoding="utf-8",
    )
    mock_result = MagicMock()
    mock_result.stdout = "error: no matching package"
    mock_result.stderr = ""
    mock_result.returncode = 101
    with patch("subprocess.run", return_value=mock_result):
        result, is_error = tiz.sandbox_worker._tool_cargo_fetch(
            {"path": str(project_dir)}
        )
    assert "exit code: 101" in result
    assert is_error is True


def test_cargo_fetch_timeout(tmp_path: Path) -> None:
    project_dir = tmp_path / "rust_project"
    project_dir.mkdir()
    cargo_toml = project_dir / "Cargo.toml"
    cargo_toml.write_text(
        '[package]\nname = "test"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    with patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired("cargo fetch", 1)
    ):
        result, is_error = tiz.sandbox_worker._tool_cargo_fetch(
            {"path": str(project_dir), "timeout": 1}
        )
    assert "Cargo fetch timed out" in result
    assert is_error is True


def test_cargo_fetch_file_not_found(tmp_path: Path) -> None:
    project_dir = tmp_path / "rust_project"
    project_dir.mkdir()
    cargo_toml = project_dir / "Cargo.toml"
    cargo_toml.write_text(
        '[package]\nname = "test"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    with patch("subprocess.run", side_effect=FileNotFoundError("cargo not found")):
        result, is_error = tiz.sandbox_worker._tool_cargo_fetch(
            {"path": str(project_dir)}
        )
    assert "ERROR: cargo not found" in result
    assert is_error is True


def test_cargo_fetch_timeout_clamped(tmp_path: Path) -> None:
    project_dir = tmp_path / "rust_project"
    project_dir.mkdir()
    cargo_toml = project_dir / "Cargo.toml"
    cargo_toml.write_text(
        '[package]\nname = "test"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_cargo_fetch(
            {"path": str(project_dir), "timeout": 0}
        )
    assert is_error is False
    # timeout clamped to 1
    assert mock_run.call_args[1]["timeout"] == 1


# ---- _tool_uv_sync ----


def test_uv_sync_success(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    mock_result = MagicMock()
    mock_result.stdout = "Resolved 10 packages in 1ms"
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_uv_sync({"path": str(project_dir)})
    assert is_error is False
    assert "Resolved" in result
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "uv"
    assert call_args[1] == "sync"


def test_uv_sync_missing_path() -> None:
    result, is_error = tiz.sandbox_worker._tool_uv_sync({})
    assert "ERROR: path is required" in result
    assert is_error is True


def test_uv_sync_directory_not_found(tmp_path: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_uv_sync(
        {"path": str(tmp_path / "nonexistent")}
    )
    assert "ERROR: directory not found" in result
    assert is_error is True


def test_uv_sync_failure(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "error: failed to sync"
    mock_result.returncode = 1
    with patch("subprocess.run", return_value=mock_result):
        result, is_error = tiz.sandbox_worker._tool_uv_sync({"path": str(project_dir)})
    assert "exit code: 1" in result
    assert is_error is True


def test_uv_sync_timeout(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("uv sync", 1)):
        result, is_error = tiz.sandbox_worker._tool_uv_sync(
            {"path": str(project_dir), "timeout": 1}
        )
    assert "uv sync timed out" in result
    assert is_error is True


def test_uv_sync_timeout_clamped(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_uv_sync(
            {"path": str(project_dir), "timeout": 0}
        )
    assert is_error is False
    # timeout clamped to 1
    assert mock_run.call_args[1]["timeout"] == 1


def test_uv_sync_with_group(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_uv_sync(
            {"path": str(project_dir), "group": ["dev", "test"]}
        )
    assert is_error is False
    call_args = mock_run.call_args[0][0]
    assert "--group" in call_args
    group_indices = [i for i, a in enumerate(call_args) if a == "--group"]
    assert len(group_indices) == 2
    assert call_args[group_indices[0] + 1] == "dev"
    assert call_args[group_indices[1] + 1] == "test"


def test_uv_sync_with_extra(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_uv_sync(
            {"path": str(project_dir), "extra": ["web", "cli"]}
        )
    assert is_error is False
    call_args = mock_run.call_args[0][0]
    assert "--extra" in call_args
    extra_indices = [i for i, a in enumerate(call_args) if a == "--extra"]
    assert len(extra_indices) == 2
    assert call_args[extra_indices[0] + 1] == "web"
    assert call_args[extra_indices[1] + 1] == "cli"


def test_uv_sync_with_group_and_extra(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_uv_sync(
            {
                "path": str(project_dir),
                "group": ["dev"],
                "extra": ["web"],
            }
        )
    assert is_error is False
    call_args = mock_run.call_args[0][0]
    assert "--group" in call_args
    assert "--extra" in call_args


def test_uv_sync_group_not_list(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    result, is_error = tiz.sandbox_worker._tool_uv_sync(
        {"path": str(project_dir), "group": "dev"}
    )
    assert "ERROR: group must be a list" in result
    assert is_error is True


def test_uv_sync_extra_not_list(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    result, is_error = tiz.sandbox_worker._tool_uv_sync(
        {"path": str(project_dir), "extra": "web"}
    )
    assert "ERROR: extra must be a list" in result
    assert is_error is True


def test_uv_sync_file_not_found(tmp_path: Path) -> None:
    project_dir = tmp_path / "python_project"
    project_dir.mkdir()
    with patch("subprocess.run", side_effect=FileNotFoundError("uv not found")):
        result, is_error = tiz.sandbox_worker._tool_uv_sync({"path": str(project_dir)})
    assert "ERROR: uv not found" in result
    assert is_error is True


# ---- _tool_uv_python_install ----


def test_uv_python_install_success() -> None:
    mock_result = MagicMock()
    mock_result.stdout = "Installed Python 3.12.0"
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_uv_python_install(
            {"version": "3.12"}
        )
    assert is_error is False
    assert "Installed" in result
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "uv"
    assert call_args[1] == "python"
    assert call_args[2] == "install"
    assert call_args[3] == "--no-bin"
    assert call_args[4] == "-f"
    assert call_args[5] == "3.12"


def test_uv_python_install_missing_version() -> None:
    result, is_error = tiz.sandbox_worker._tool_uv_python_install({})
    assert "ERROR: version is required" in result
    assert is_error is True


def test_uv_python_install_empty_version() -> None:
    result, is_error = tiz.sandbox_worker._tool_uv_python_install({"version": ""})
    assert "ERROR: version is required" in result
    assert is_error is True


def test_uv_python_install_failure() -> None:
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "error: unknown Python version"
    mock_result.returncode = 1
    with patch("subprocess.run", return_value=mock_result):
        result, is_error = tiz.sandbox_worker._tool_uv_python_install(
            {"version": "3.99"}
        )
    assert "exit code: 1" in result
    assert is_error is True


def test_uv_python_install_timeout() -> None:
    with patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired("uv python install", 1)
    ):
        result, is_error = tiz.sandbox_worker._tool_uv_python_install(
            {"version": "3.12", "timeout": 1}
        )
    assert "uv python install timed out" in result
    assert is_error is True


def test_uv_python_install_timeout_clamped() -> None:
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, is_error = tiz.sandbox_worker._tool_uv_python_install(
            {"version": "3.12", "timeout": 0}
        )
    assert is_error is False
    # timeout clamped to 1
    assert mock_run.call_args[1]["timeout"] == 1


def test_uv_python_install_file_not_found() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("uv not found")):
        result, is_error = tiz.sandbox_worker._tool_uv_python_install(
            {"version": "3.12"}
        )
    assert "ERROR: uv not found" in result
    assert is_error is True


# ---- _tool_read ----


def test_read_full_file(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_read({"path": str(tmp_file)})
    assert result == "1\tline1\n2\tline2\n3\tline3"
    assert is_error is False


def test_read_view_range(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_read(
        {"path": str(tmp_file), "view_range": [2, 3]}
    )
    assert result == "2\tline2\n3\tline3"
    assert is_error is False


def test_read_file_not_found() -> None:
    result, is_error = tiz.sandbox_worker._tool_read({"path": "/nonexistent/file.txt"})
    assert "ERROR: file not found" in result
    assert is_error is True


def test_read_view_range_invalid_type(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_read(
        {"path": str(tmp_file), "view_range": "bad"}
    )
    assert "ERROR: view_range must be a two-element array" in result
    assert is_error is True


def test_read_view_range_wrong_length(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_read(
        {"path": str(tmp_file), "view_range": [1]}
    )
    assert "ERROR: view_range must be a two-element array" in result
    assert is_error is True


def test_read_view_range_non_int(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_read(
        {"path": str(tmp_file), "view_range": ["a", "b"]}
    )
    assert "ERROR: view_range values must be integers" in result
    assert is_error is True


def test_read_view_range_negative(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_read(
        {"path": str(tmp_file), "view_range": [-1, 5]}
    )
    assert "ERROR: view_range values must be positive" in result
    assert is_error is True


def test_read_view_range_start_gt_end(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_read(
        {"path": str(tmp_file), "view_range": [5, 2]}
    )
    assert "ERROR: view_range start must be <= end" in result
    assert is_error is True


def test_read_file_too_large(tmp_path: Path) -> None:
    big_file = tmp_path / "big.txt"
    with big_file.open("wb") as f:
        f.write(b"x" * (tiz.sandbox_worker._MAX_BUFFER_SIZE + 1))
    result, is_error = tiz.sandbox_worker._tool_read({"path": str(big_file)})
    assert "ERROR: file too large" in result
    assert is_error is True


def test_read_directory_error(tmp_path: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_read({"path": str(tmp_path)})
    assert "ERROR: file not found" in result
    assert is_error is True


def test_read_oserror(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello", encoding="utf-8")
    with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
        result, is_error = tiz.sandbox_worker._tool_read({"path": str(f)})
    assert "Error reading file" in result
    assert is_error is True


# ---- _tool_write ----


def test_write_success(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "file.txt"
    result, is_error = tiz.sandbox_worker._tool_write(
        {"path": str(target), "contents": "hello"}
    )
    assert "Wrote" in result
    assert is_error is False
    assert target.read_text(encoding="utf-8") == "hello"


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    result, is_error = tiz.sandbox_worker._tool_write(
        {"path": str(target), "contents": "nested"}
    )
    assert "Wrote" in result
    assert is_error is False
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "nested"


def test_write_empty_contents(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    result, is_error = tiz.sandbox_worker._tool_write(
        {"path": str(target), "contents": ""}
    )
    assert "Wrote" in result
    assert is_error is False
    assert target.exists()
    assert target.read_text(encoding="utf-8") == ""


def test_write_oserror() -> None:
    with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
        result, is_error = tiz.sandbox_worker._tool_write(
            {
                "path": "/root/test.txt",
                "contents": "hello",
            }
        )
    assert "Error writing file" in result
    assert is_error is True


# ---- _tool_edit ----


def test_edit_single_replacement(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(tmp_file),
            "old_string": "line2",
            "new_string": "replaced",
        }
    )
    assert "Edited" in result
    assert is_error is False
    assert tmp_file.read_text(encoding="utf-8") == "line1\nreplaced\nline3\n"


def test_edit_multiple_replacements(tmp_path: Path) -> None:
    f = tmp_path / "multi.txt"
    f.write_text("a b a b a", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(f),
            "old_string": "a",
            "new_string": "X",
            "expected_replacements": -1,
        }
    )
    assert "Edited" in result
    assert is_error is False
    assert f.read_text(encoding="utf-8") == "X b X b X"


def test_edit_expected_replacements_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("a a a", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(f),
            "old_string": "a",
            "new_string": "X",
            "expected_replacements": 2,
        }
    )
    assert "expected 2 occurrences but found 3" in result
    assert is_error is True


def test_edit_multiple_locations_error(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("a a", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(f),
            "old_string": "a",
            "new_string": "X",
        }
    )
    assert "ERROR: old_string matches multiple locations" in result
    assert is_error is True


def test_edit_not_found(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(f),
            "old_string": "notthere",
            "new_string": "X",
        }
    )
    assert "ERROR: old_string not found" in result
    assert is_error is True


def test_edit_file_not_found() -> None:
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": "/nonexistent/file.txt",
            "old_string": "a",
            "new_string": "b",
        }
    )
    assert "ERROR: file not found" in result
    assert is_error is True


def test_edit_default_expected_replacements(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("only one", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(f),
            "old_string": "only one",
            "new_string": "replaced",
        }
    )
    assert "Edited" in result
    assert "1 replacement(s)" in result
    assert is_error is False


def test_edit_oserror(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello", encoding="utf-8")
    with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
        result, is_error = tiz.sandbox_worker._tool_edit(
            {
                "path": str(f),
                "old_string": "hello",
                "new_string": "world",
            }
        )
    assert "Error editing file" in result
    assert is_error is True


def test_edit_expected_replacements_explicit(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("a a a", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(f),
            "old_string": "a",
            "new_string": "X",
            "expected_replacements": 2,
        }
    )
    assert "expected 2 occurrences but found 3" in result
    assert is_error is True


def test_edit_expected_replacements_exact_match(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("a a a", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(f),
            "old_string": "a",
            "new_string": "X",
            "expected_replacements": 3,
        }
    )
    assert "Edited" in result
    assert "3 replacement(s)" in result
    content = f.read_text(encoding="utf-8")
    assert content == "X X X"
    assert is_error is False


def test_edit_expected_replacements_non_int(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("a a a", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_edit(
        {
            "path": str(f),
            "old_string": "a",
            "new_string": "X",
            "expected_replacements": "2",
        }
    )
    assert "ERROR: expected_replacements must be an integer" in result
    assert is_error is True


# ---- _tool_glob ----


def test_glob_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_glob(
        {"path": str(tmp_path), "pattern": "*.py"}
    )
    assert str(tmp_path / "b.py") == result
    assert is_error is False


def test_glob_no_matches(tmp_path: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_glob(
        {"path": str(tmp_path), "pattern": "*.xyz"}
    )
    assert result == "No matches"
    assert is_error is False


def test_glob_directory_not_found() -> None:
    result, is_error = tiz.sandbox_worker._tool_glob(
        {"path": "/nonexistent", "pattern": "*"}
    )
    assert "ERROR: directory not found" in result
    assert is_error is True


def test_glob_skips_git(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("", encoding="utf-8")
    (tmp_path / "file.txt").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_glob(
        {"path": str(tmp_path), "pattern": "**/*"}
    )
    assert ".git" not in result
    assert str(tmp_path / "file.txt") in result
    assert is_error is False


def test_glob_limit_100(tmp_path: Path) -> None:
    for i in range(150):
        (tmp_path / f"file_{i:03d}.txt").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_glob(
        {"path": str(tmp_path), "pattern": "*.txt"}
    )
    lines = result.strip().split("\n")
    assert len(lines) <= 100
    assert is_error is False


def test_glob_default_path() -> None:
    result, is_error = tiz.sandbox_worker._tool_glob({"pattern": "*.py"})
    assert is_error is False


def test_glob_recursive_pattern(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("", encoding="utf-8")
    (sub / "c.txt").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_glob(
        {"path": str(tmp_path), "pattern": "**/*.py"}
    )
    assert str(tmp_path / "a.py") in result
    assert str(sub / "b.py") in result
    assert str(sub / "c.txt") not in result
    assert is_error is False


def test_glob_oserror(tmp_path: Path) -> None:
    with patch.object(Path, "glob", side_effect=OSError("permission denied")):
        result, is_error = tiz.sandbox_worker._tool_glob(
            {
                "path": str(tmp_path),
                "pattern": "*",
            }
        )
    assert "Error searching directories" in result
    assert is_error is True


# ---- _tool_grep ----


def test_grep_with_rg(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello world\nfoo bar\n", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_grep(
        {"path": str(tmp_path), "pattern": "hello", "regex": False}
    )
    assert f"{f}:1:hello world" == result
    assert is_error is False


def test_grep_no_matches(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("nothing here", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_grep(
        {"path": str(tmp_path), "pattern": "zzz", "regex": False}
    )
    assert result == "No matches"
    assert is_error is False


def test_grep_case_insensitive(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("HELLO\nworld\n", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_grep(
        {
            "path": str(tmp_path),
            "pattern": "hello",
            "regex": False,
            "case_insensitive": True,
        }
    )
    assert f"{f}:1:HELLO" == result
    assert is_error is False


def test_grep_with_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("found in txt", encoding="utf-8")
    (tmp_path / "test.py").write_text("found in py", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_grep(
        {
            "path": str(tmp_path),
            "pattern": "found",
            "regex": False,
            "glob": "*.txt",
        }
    )
    assert f"{tmp_path / 'test.txt'}:1:found in txt" == result
    assert is_error is False


def test_grep_max_results(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    content = "\n".join(["match"] * 200)
    f.write_text(content, encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_grep(
        {
            "path": str(tmp_path),
            "pattern": "match",
            "regex": False,
            "max_results": 5,
        }
    )
    lines = [line for line in result.split("\n") if line]
    assert len(lines) <= 5
    assert is_error is False


def test_grep_max_results_not_positive(tmp_path: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_grep(
        {
            "path": str(tmp_path),
            "pattern": "test",
            "max_results": 0,
        }
    )
    assert "ERROR: max_results must be a positive integer" in result
    assert is_error is True


def test_grep_max_results_not_int(tmp_path: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_grep(
        {
            "path": str(tmp_path),
            "pattern": "test",
            "max_results": "abc",
        }
    )
    assert "ERROR: max_results must be a positive integer" in result
    assert is_error is True


def test_grep_fallback_to_grep(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        f = tmp_path / "test.txt"
        f.write_text("hello world\n", encoding="utf-8")
        result, is_error = tiz.sandbox_worker._tool_grep(
            {
                "path": str(tmp_path),
                "pattern": "hello",
                "regex": False,
            }
        )
        assert "hello world" in result
        assert is_error is False
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_fallback_case_insensitive(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        f = tmp_path / "test.txt"
        f.write_text("HELLO\n", encoding="utf-8")
        result, is_error = tiz.sandbox_worker._tool_grep(
            {
                "path": str(tmp_path),
                "pattern": "hello",
                "regex": False,
                "case_insensitive": True,
            }
        )
        assert "HELLO" in result
        assert is_error is False
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_subprocess_error(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=subprocess.SubprocessError("fail")):
        result, is_error = tiz.sandbox_worker._tool_grep(
            {
                "path": str(tmp_path),
                "pattern": "test",
            }
        )
    assert "Error running grep" in result
    assert is_error is True


def test_grep_fallback_regex_true(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        f = tmp_path / "test.txt"
        f.write_text("hello world\n", encoding="utf-8")
        result, is_error = tiz.sandbox_worker._tool_grep(
            {
                "path": str(tmp_path),
                "pattern": "hello.*",
                "regex": True,
            }
        )
        assert "hello world" in result
        assert is_error is False
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_fallback_glob_filter(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        (tmp_path / "test.txt").write_text("match", encoding="utf-8")
        (tmp_path / "test.py").write_text("match", encoding="utf-8")
        result, is_error = tiz.sandbox_worker._tool_grep(
            {
                "path": str(tmp_path),
                "pattern": "match",
                "glob": "*.txt",
            }
        )
        assert f"{tmp_path / 'test.txt'}:1:match" == result
        assert is_error is False
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_fallback_glob_filter_no_matches(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        (tmp_path / "test.txt").write_text("match", encoding="utf-8")
        result, is_error = tiz.sandbox_worker._tool_grep(
            {
                "path": str(tmp_path),
                "pattern": "match",
                "glob": "*.xyz",
            }
        )
        assert result == "No matches"
        assert is_error is False
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_fallback_glob_filter_regex_case_insensitive(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        (tmp_path / "test.txt").write_text("HELLO WORLD", encoding="utf-8")
        result, is_error = tiz.sandbox_worker._tool_grep(
            {
                "path": str(tmp_path),
                "pattern": "hello.*",
                "regex": True,
                "case_insensitive": True,
                "glob": "*.txt",
            }
        )
        assert f"{tmp_path / 'test.txt'}:1:HELLO WORLD" == result
        assert is_error is False
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_fallback_glob_filter_subprocess_error(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        with patch("subprocess.run", side_effect=subprocess.SubprocessError("fail")):
            result, is_error = tiz.sandbox_worker._tool_grep(
                {
                    "path": str(tmp_path),
                    "pattern": "test",
                    "glob": "*.txt",
                }
            )
        assert "Error running grep" in result
        assert is_error is True
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_fallback_case_insensitive_regex_true(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        f = tmp_path / "test.txt"
        f.write_text("HELLO\n", encoding="utf-8")
        result, is_error = tiz.sandbox_worker._tool_grep(
            {
                "path": str(tmp_path),
                "pattern": "hello.*",
                "regex": True,
                "case_insensitive": True,
            }
        )
        assert "HELLO" in result
        assert is_error is False
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_rg_regex_true_case_insensitive(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = True
    try:
        f = tmp_path / "test.txt"
        f.write_text("HELLO\n", encoding="utf-8")
        mock_result = MagicMock()
        mock_result.stdout = "HELLO"
        with patch("subprocess.run", return_value=mock_result):
            result, is_error = tiz.sandbox_worker._tool_grep(
                {
                    "path": str(tmp_path),
                    "pattern": "hello.*",
                    "regex": True,
                    "case_insensitive": True,
                }
            )
        assert "HELLO" in result
        assert is_error is False
    finally:
        tiz.sandbox_worker._HAS_RG = original


def test_grep_fallback_subprocess_error(tmp_path: Path) -> None:
    original = tiz.sandbox_worker._HAS_RG
    tiz.sandbox_worker._HAS_RG = False
    try:
        with patch("subprocess.run", side_effect=subprocess.SubprocessError("fail")):
            result, is_error = tiz.sandbox_worker._tool_grep(
                {
                    "path": str(tmp_path),
                    "pattern": "test",
                }
            )
        assert "Error running grep" in result
        assert is_error is True
    finally:
        tiz.sandbox_worker._HAS_RG = original


# ---- _tool_insert ----


def test_insert_at_end(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_insert(
        {
            "path": str(tmp_file),
            "content": "line4",
            "line_number": -1,
        }
    )
    assert "Inserted" in result
    assert is_error is False
    content = tmp_file.read_text(encoding="utf-8")
    assert content == "line1\nline2\nline3\nline4\n"


def test_insert_at_line(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_insert(
        {
            "path": str(tmp_file),
            "content": "new line",
            "line_number": 2,
        }
    )
    assert "Inserted" in result
    assert is_error is False
    content = tmp_file.read_text(encoding="utf-8")
    assert content == "line1\nnew line\nline2\nline3\n"


def test_insert_multiple_lines(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_insert(
        {
            "path": str(tmp_file),
            "content": "a\nb\nc",
            "line_number": 1,
        }
    )
    assert "Inserted 3 line(s)" in result
    assert is_error is False
    content = tmp_file.read_text(encoding="utf-8")
    assert content == "a\nb\nc\nline1\nline2\nline3\n"


def test_insert_line_number_out_of_range(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_insert(
        {
            "path": str(tmp_file),
            "content": "new",
            "line_number": 100,
        }
    )
    assert "ERROR: line_number" in result
    assert is_error is True


def test_insert_file_not_found() -> None:
    result, is_error = tiz.sandbox_worker._tool_insert(
        {
            "path": "/nonexistent/file.txt",
            "content": "new",
        }
    )
    assert "ERROR: file not found" in result
    assert is_error is True


def test_insert_default_line_number(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("existing", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_insert(
        {
            "path": str(f),
            "content": "appended",
        }
    )
    assert "Inserted" in result
    assert is_error is False
    content = f.read_text(encoding="utf-8")
    assert content == "existing\nappended\n"


def test_insert_line_number_not_int(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("existing", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_insert(
        {
            "path": str(f),
            "content": "new",
            "line_number": 1.5,
        }
    )
    assert "ERROR: line_number must be an integer" in result
    assert is_error is True


def test_insert_oserror(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello", encoding="utf-8")
    with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
        result, is_error = tiz.sandbox_worker._tool_insert(
            {
                "path": str(f),
                "content": "new",
            }
        )
    assert "Error inserting content" in result
    assert is_error is True


# ---- _tool_list_dir ----


def test_list_dir_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    result, is_error = tiz.sandbox_worker._tool_list_dir({"path": str(tmp_path)})
    entries = json.loads(result)
    assert len(entries) == 2
    names = {e["name"] for e in entries}
    assert "file.txt" in names
    assert "subdir" in names
    assert is_error is False


def test_list_dir_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_list_dir(
        {"path": str(tmp_path), "recursive": True}
    )
    entries = json.loads(result)
    assert len(entries) == 3
    names = {e["name"] for e in entries}
    assert "a.txt" in names
    assert "b.txt" in names
    assert "sub" in names
    assert is_error is False


def test_list_dir_hidden_files(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_text("", encoding="utf-8")
    (tmp_path / "visible").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_list_dir({"path": str(tmp_path)})
    entries = json.loads(result)
    assert len(entries) == 1
    names = {e["name"] for e in entries}
    assert ".hidden" not in names
    assert "visible" in names
    assert is_error is False


def test_list_dir_show_hidden(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_list_dir(
        {"path": str(tmp_path), "show_hidden": True}
    )
    entries = json.loads(result)
    assert len(entries) == 1
    names = {e["name"] for e in entries}
    assert ".hidden" in names
    assert is_error is False


def test_list_dir_directory_not_found() -> None:
    result, is_error = tiz.sandbox_worker._tool_list_dir({"path": "/nonexistent"})
    assert "ERROR: directory not found" in result
    assert is_error is True


def test_list_dir_entry_types(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("", encoding="utf-8")
    d = tmp_path / "dir"
    d.mkdir()
    link = tmp_path / "link"
    link.symlink_to(f)
    result, is_error = tiz.sandbox_worker._tool_list_dir(
        {"path": str(tmp_path), "show_hidden": True}
    )
    entries = json.loads(result)
    assert len(entries) == 3
    entry_map = {e["name"]: e for e in entries}
    assert entry_map["file.txt"]["type"] == "file"
    assert entry_map["dir"]["type"] == "directory"
    assert entry_map["link"]["type"] == "symlink"
    assert entry_map["file.txt"]["size"] == 0
    assert entry_map["dir"]["size"] is None
    assert entry_map["link"]["size"] is None
    for e in entries:
        assert "modified" in e


def test_list_dir_max_entries(tmp_path: Path) -> None:
    for i in range(tiz.sandbox_worker._MAX_LIST_DIR_ENTRIES + 50):
        (tmp_path / f"file_{i:04d}.txt").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_list_dir(
        {"path": str(tmp_path), "recursive": True}
    )
    entries = json.loads(result)
    assert len(entries) <= tiz.sandbox_worker._MAX_LIST_DIR_ENTRIES
    assert is_error is False


def test_list_dir_default_path() -> None:
    result, is_error = tiz.sandbox_worker._tool_list_dir({})
    assert is_error is False


def test_list_dir_recursive_hidden_git(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("", encoding="utf-8")
    (tmp_path / "file.txt").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_list_dir(
        {
            "path": str(tmp_path),
            "recursive": True,
        }
    )
    data = json.loads(result)
    names = {e["name"] for e in data}
    assert len(data) == 1
    assert "file.txt" in names
    assert "config" not in names
    assert is_error is False


def test_list_dir_recursive_hidden_files(tmp_path: Path) -> None:
    hidden = tmp_path / ".hidden"
    hidden.write_text("", encoding="utf-8")
    visible = tmp_path / "visible"
    visible.write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_list_dir(
        {
            "path": str(tmp_path),
            "recursive": True,
        }
    )
    data = json.loads(result)
    names = {e["name"] for e in data}
    assert len(data) == 1
    assert ".hidden" not in names
    assert "visible" in names
    assert is_error is False


def test_list_dir_recursive_with_symlink(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("content", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(f)
    result, is_error = tiz.sandbox_worker._tool_list_dir(
        {
            "path": str(tmp_path),
            "recursive": True,
            "show_hidden": True,
        }
    )
    data = json.loads(result)
    entry_map = {e["name"]: e for e in data}
    assert len(data) == 2
    assert entry_map["link"]["type"] == "symlink"
    assert entry_map["link"]["size"] is None
    assert entry_map["file.txt"]["type"] == "file"
    assert entry_map["file.txt"]["size"] == 7
    assert is_error is False


def test_list_dir_recursive_max_entries(tmp_path: Path) -> None:
    for i in range(tiz.sandbox_worker._MAX_LIST_DIR_ENTRIES + 50):
        (tmp_path / f"file_{i:04d}.txt").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_list_dir(
        {
            "path": str(tmp_path),
            "recursive": True,
        }
    )
    data = json.loads(result)
    assert len(data) <= tiz.sandbox_worker._MAX_LIST_DIR_ENTRIES
    assert is_error is False


def test_list_dir_oserror(tmp_path: Path) -> None:
    with patch.object(Path, "iterdir", side_effect=OSError("permission denied")):
        result, is_error = tiz.sandbox_worker._tool_list_dir({"path": str(tmp_path)})
    assert "Error listing directory" in result
    assert is_error is True


def test_list_dir_non_recursive_max_entries(tmp_path: Path) -> None:
    for i in range(tiz.sandbox_worker._MAX_LIST_DIR_ENTRIES + 50):
        (tmp_path / f"file_{i:04d}.txt").write_text("", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_list_dir({"path": str(tmp_path)})
    data = json.loads(result)
    assert len(data) <= tiz.sandbox_worker._MAX_LIST_DIR_ENTRIES
    assert is_error is False


# ---- _tool_patch ----


def test_patch_success(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello\n", encoding="utf-8")
    patch_content = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-hello
+world
"""
    original_cwd = str(Path.cwd())
    try:
        os.chdir(tmp_path)
        result, is_error = tiz.sandbox_worker._tool_patch(
            {"patch": patch_content, "strip": 1}
        )
        assert is_error is False
        assert f.read_text(encoding="utf-8") == "world\n"
    finally:
        os.chdir(original_cwd)


def test_patch_failure() -> None:
    result, is_error = tiz.sandbox_worker._tool_patch({"patch": "not a valid patch"})
    assert is_error is True


def test_patch_reverse(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("new\n", encoding="utf-8")
    patch_content = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old
+new
"""
    original_cwd = str(Path.cwd())
    try:
        os.chdir(tmp_path)
        result, is_error = tiz.sandbox_worker._tool_patch(
            {"patch": patch_content, "strip": 1, "reverse": True}
        )
        assert is_error is False
        assert f.read_text(encoding="utf-8") == "old\n"
    finally:
        os.chdir(original_cwd)


def test_patch_oserror() -> None:
    with patch("tempfile.NamedTemporaryFile", side_effect=OSError("no temp")):
        result, is_error = tiz.sandbox_worker._tool_patch({"patch": "diff"})
    assert "Error applying patch" in result
    assert is_error is True


def test_patch_with_cwd(tmp_path: Path) -> None:
    sub = tmp_path / "subdir"
    sub.mkdir()
    f = sub / "test.txt"
    f.write_text("hello\n", encoding="utf-8")
    patch_content = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-hello
+world
"""
    result, is_error = tiz.sandbox_worker._tool_patch(
        {"patch": patch_content, "strip": 1, "cwd": str(sub)}
    )
    assert is_error is False
    assert f.read_text(encoding="utf-8") == "world\n"


def test_patch_with_cwd_failure() -> None:
    result, is_error = tiz.sandbox_worker._tool_patch(
        {"patch": "diff", "cwd": "/nonexistent/"}
    )
    assert is_error is True


def test_patch_strip_not_int() -> None:
    result, is_error = tiz.sandbox_worker._tool_patch({"patch": "diff", "strip": "1"})
    assert "ERROR: strip must be a non-negative integer" in result
    assert is_error is True


def test_patch_strip_negative() -> None:
    result, is_error = tiz.sandbox_worker._tool_patch({"patch": "diff", "strip": -1})
    assert "ERROR: strip must be a non-negative integer" in result
    assert is_error is True


# ---- _tool_read_multi ----


def test_read_multi_success(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content1", encoding="utf-8")
    f2.write_text("content2", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_read_multi(
        {"paths": [str(f1), str(f2)]}
    )
    entries = json.loads(result)
    assert len(entries) == 2
    assert entries[0]["content"] == "content1"
    assert entries[1]["content"] == "content2"
    assert is_error is False


def test_read_multi_not_found(tmp_path: Path) -> None:
    f = tmp_path / "exists.txt"
    f.write_text("data", encoding="utf-8")
    result, is_error = tiz.sandbox_worker._tool_read_multi(
        {"paths": [str(f), "/nonexistent"]}
    )
    entries = json.loads(result)
    assert entries[0]["content"] == "data"
    assert entries[1]["content"] is None
    assert "not found" in entries[1]["error"]
    assert is_error is True


def test_read_multi_invalid_paths_type() -> None:
    result, is_error = tiz.sandbox_worker._tool_read_multi({"paths": "not_a_list"})
    assert "ERROR: paths must be an array" in result
    assert is_error is True


def test_read_multi_empty_list() -> None:
    result, is_error = tiz.sandbox_worker._tool_read_multi({"paths": []})
    entries = json.loads(result)
    assert entries == []
    assert is_error is False


def test_read_multi_file_too_large(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    with big.open("wb") as f:
        f.write(b"x" * (tiz.sandbox_worker._MAX_BUFFER_SIZE + 1))
    result, is_error = tiz.sandbox_worker._tool_read_multi({"paths": [str(big)]})
    entries = json.loads(result)
    assert entries[0]["content"] is None
    assert "too large" in entries[0]["error"]
    assert is_error is True


def test_read_multi_oserror(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello", encoding="utf-8")
    with patch.object(Path, "is_file", side_effect=OSError("permission denied")):
        result, is_error = tiz.sandbox_worker._tool_read_multi({"paths": [str(f)]})
    data = json.loads(result)
    assert data[0]["content"] is None
    assert data[0]["error"] is not None
    assert is_error is True


# ---- _tool_metadata ----


def test_metadata_file(tmp_file: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_metadata({"path": str(tmp_file)})
    data = json.loads(result)
    assert data["exists"] is True
    assert data["type"] == "file"
    assert data["size"] == 18
    assert data["error"] is None
    assert "permissions" in data
    assert is_error is False


def test_metadata_directory(tmp_path: Path) -> None:
    result, is_error = tiz.sandbox_worker._tool_metadata({"path": str(tmp_path)})
    data = json.loads(result)
    assert data["exists"] is True
    assert data["type"] == "directory"
    assert data["error"] is None
    assert "permissions" in data
    assert is_error is False


def test_metadata_symlink(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(f)
    result, is_error = tiz.sandbox_worker._tool_metadata({"path": str(link)})
    data = json.loads(result)
    assert data["exists"] is True
    assert data["type"] == "symlink"
    assert is_error is False


def test_metadata_not_found() -> None:
    result, is_error = tiz.sandbox_worker._tool_metadata({"path": "/nonexistent/path"})
    data = json.loads(result)
    assert data["exists"] is False
    assert "not found" in data["error"]
    assert is_error is False


def test_metadata_symlink_type(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(f)
    result, is_error = tiz.sandbox_worker._tool_metadata({"path": str(link)})
    data = json.loads(result)
    assert data["exists"] is True
    assert data["type"] == "symlink"
    # Symlink size should be None, consistent with list_dir
    assert data["size"] is None
    assert is_error is False


def test_read_view_range_tuple(tmp_file: Path) -> None:
    """view_range as a tuple should be accepted."""
    result, is_error = tiz.sandbox_worker._tool_read(
        {"path": str(tmp_file), "view_range": (2, 3)}
    )
    assert result == "2\tline2\n3\tline3"
    assert is_error is False


def test_metadata_oserror(tmp_path: Path) -> None:
    p = tmp_path / "testfile"
    p.write_text("hello", encoding="utf-8")
    with patch.object(Path, "lstat", side_effect=OSError("permission denied")):
        result, is_error = tiz.sandbox_worker._tool_metadata({"path": str(p)})
    data = json.loads(result)
    assert data["exists"] is False
    assert "error" in data
    assert is_error is True


def test_metadata_special_file(tmp_path: Path) -> None:
    import socket as _socket

    sock_path = str(tmp_path / "mysock")
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.bind(sock_path)
    try:
        result, is_error = tiz.sandbox_worker._tool_metadata({"path": sock_path})
        data = json.loads(result)
        assert data["exists"] is True
        assert data["type"] is None
        assert is_error is False
    finally:
        s.close()


# ---- HANDLERS ----


def test_handlers_has_all_tools() -> None:
    expected = {
        "Bash",
        "CargoFetch",
        "ReadFile",
        "Edit",
        "WriteFile",
        "Glob",
        "Grep",
        "InsertFile",
        "ListDir",
        "ApplyPatch",
        "ReadMulti",
        "FileMetadata",
        "UvSync",
        "UvPythonInstall",
    }
    actual = set(tiz.sandbox_worker.HANDLERS.keys())
    assert expected.issubset(actual)
    assert "Bash" in actual


# ---- _load_custom_tools ----


def test_load_custom_tools_directory_not_exist(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original = sw._CUSTOM_TOOLS_DIR
    sw._CUSTOM_TOOLS_DIR = tmp_path / "nonexistent_tools"
    try:
        result = sw._load_custom_tools()
        assert result == {}
    finally:
        sw._CUSTOM_TOOLS_DIR = original


def test_load_custom_tools_empty_directory(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools"
    custom_dir.mkdir()
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        result = sw._load_custom_tools()
        assert result == {}
    finally:
        sw._CUSTOM_TOOLS_DIR = original


def test_load_custom_tools_with_valid_module(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools"
    custom_dir.mkdir()
    tool_file = custom_dir / "CustomHello.py"
    tool_file.write_text(
        "def handle(params):\n"
        '    name = params.get("name", "world")\n'
        '    return f"Hello, {name}!", False\n',
        encoding="utf-8",
    )
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        result = sw._load_custom_tools()
        assert "CustomHello" in result
        output, is_error = result["CustomHello"]({"name": "test"})
        assert output == "Hello, test!"
        assert is_error is False
    finally:
        sw._CUSTOM_TOOLS_DIR = original


def test_load_custom_tools_module_without_handle(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools"
    custom_dir.mkdir()
    tool_file = custom_dir / "NoHandler.py"
    tool_file.write_text(
        "x = 42\n",
        encoding="utf-8",
    )
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        result = sw._load_custom_tools()
        assert "NoHandler" not in result
    finally:
        sw._CUSTOM_TOOLS_DIR = original


def test_load_custom_tools_skips_non_py_files(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools"
    custom_dir.mkdir()
    (custom_dir / "notes.txt").write_text("not a tool", encoding="utf-8")
    (custom_dir / "data.json").write_text('{"key": "val"}', encoding="utf-8")
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        result = sw._load_custom_tools()
        assert result == {}
    finally:
        sw._CUSTOM_TOOLS_DIR = original


def test_load_custom_tools_multiple_tools(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools"
    custom_dir.mkdir()
    (custom_dir / "Foo.py").write_text(
        "def handle(params):\n    return 'foo', False\n",
        encoding="utf-8",
    )
    (custom_dir / "Bar.py").write_text(
        "def handle(params):\n    return 'bar', False\n",
        encoding="utf-8",
    )
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        result = sw._load_custom_tools()
        assert set(result.keys()) == {"Foo", "Bar"}
        assert result["Foo"]({}) == ("foo", False)
        assert result["Bar"]({}) == ("bar", False)
    finally:
        sw._CUSTOM_TOOLS_DIR = original


def test_load_custom_tools_spec_none(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original_dir = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools"
    custom_dir.mkdir()
    tool_file = custom_dir / "Bogus.py"
    tool_file.write_text("x = 1\n", encoding="utf-8")
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        with patch("importlib.util.spec_from_file_location", return_value=None):
            result = sw._load_custom_tools()
        assert result == {}
    finally:
        sw._CUSTOM_TOOLS_DIR = original_dir


def test_load_custom_tools_module_handler_raises(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools2"
    custom_dir.mkdir()
    (custom_dir / "Broken.py").write_text(
        "def handle(params):\n    raise RuntimeError('broken')\n",
        encoding="utf-8",
    )
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        result = sw._load_custom_tools()
        assert "Broken" in result
    finally:
        sw._CUSTOM_TOOLS_DIR = original


def test_load_custom_tools_import_error(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools3"
    custom_dir.mkdir()
    (custom_dir / "BadImport.py").write_text(
        "raise ImportError('cannot load')\n",
        encoding="utf-8",
    )
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        result = sw._load_custom_tools()
        assert "BadImport" not in result
    finally:
        sw._CUSTOM_TOOLS_DIR = original


def test_custom_tool_loaded_into_handlers(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original_dir = sw._CUSTOM_TOOLS_DIR
    custom_dir = tmp_path / "tools"
    custom_dir.mkdir()
    (custom_dir / "MyTool.py").write_text(
        "def handle(params):\n    return 'mytool result', False\n",
        encoding="utf-8",
    )
    sw._CUSTOM_TOOLS_DIR = custom_dir
    try:
        sw.HANDLERS.update(sw._load_custom_tools())
        assert "MyTool" in sw.HANDLERS
        result, is_error = sw.run_tool("MyTool", {})
        assert result == "mytool result"
        assert is_error is False
    finally:
        sw._CUSTOM_TOOLS_DIR = original_dir


def _send_request(conn: socket.socket, request: dict) -> dict:
    payload = json.dumps(request).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)
    header = b""
    while len(header) < 4:
        chunk = conn.recv(4 - len(header))
        if not chunk:
            break
        header += chunk
    if len(header) < 4:
        return {}
    msg_len = struct.unpack(">I", header)[0]
    body = b""
    while len(body) < msg_len:
        chunk = conn.recv(min(65536, msg_len - len(body)))
        if not chunk:
            break
        body += chunk
    return json.loads(body.decode("utf-8").rstrip("\n"))


def test_handle_connection_valid_request(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(5)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(socket_path)
        conn, _ = server.accept()
        thread = threading.Thread(
            target=sw.handle_connection, args=(conn,), daemon=True
        )
        thread.start()
        resp = _send_request(client, {"name": "Bash", "command": "echo hi"})
        assert "hi" in resp["result"]
        assert resp["error"] is False
        client.close()
    finally:
        server.close()
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_handle_connection_invalid_json(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(5)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(socket_path)
        conn, _ = server.accept()
        thread = threading.Thread(
            target=sw.handle_connection, args=(conn,), daemon=True
        )
        thread.start()
        payload = b"not json"
        client.sendall(struct.pack(">I", len(payload)) + payload)
        header = b""
        while len(header) < 4:
            chunk = client.recv(4 - len(header))
            if not chunk:
                break
            header += chunk
        msg_len = struct.unpack(">I", header)[0]
        body = b""
        while len(body) < msg_len:
            chunk = client.recv(min(65536, msg_len - len(body)))
            if not chunk:
                break
            body += chunk
        resp = json.loads(body.decode("utf-8").rstrip("\n"))
        assert "Invalid JSON" in resp["error"]
        client.close()
    finally:
        server.close()
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_handle_connection_missing_name(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(5)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(socket_path)
        conn, _ = server.accept()
        thread = threading.Thread(
            target=sw.handle_connection, args=(conn,), daemon=True
        )
        thread.start()
        resp = _send_request(client, {"params": {}})
        assert "Missing 'name' field" in resp["error"]
        client.close()
    finally:
        server.close()
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_handle_connection_request_too_large(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(5)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(socket_path)
        conn, _ = server.accept()
        thread = threading.Thread(
            target=sw.handle_connection, args=(conn,), daemon=True
        )
        thread.start()
        large_size = sw._MAX_BUFFER_SIZE + 1
        client.sendall(struct.pack(">I", large_size))
        header = b""
        while len(header) < 4:
            chunk = client.recv(4 - len(header))
            if not chunk:
                break
            header += chunk
        msg_len = struct.unpack(">I", header)[0]
        body = b""
        while len(body) < msg_len:
            chunk = client.recv(min(65536, msg_len - len(body)))
            if not chunk:
                break
            body += chunk
        resp = json.loads(body.decode("utf-8").rstrip("\n"))
        assert "Request too large" in resp["error"]
        client.close()
    finally:
        server.close()
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_handle_connection_disconnect(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(5)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(socket_path)
        conn, _ = server.accept()
        thread = threading.Thread(
            target=sw.handle_connection, args=(conn,), daemon=True
        )
        thread.start()
        client.close()
        time.sleep(0.2)
    finally:
        server.close()
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_handle_connection_disconnect_during_body(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(5)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(socket_path)
        conn, _ = server.accept()
        thread = threading.Thread(
            target=sw.handle_connection, args=(conn,), daemon=True
        )
        thread.start()
        payload = json.dumps({"name": "Bash", "command": "echo hi"}).encode("utf-8")
        client.sendall(struct.pack(">I", len(payload) + 100))
        client.sendall(payload[:10])
        client.close()
        time.sleep(0.3)
    finally:
        server.close()
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_handle_connection_semaphore_release(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    initial = sw._CONNECTION_SEMAPHORE._value
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(5)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(socket_path)
        conn, _ = server.accept()
        thread = threading.Thread(
            target=sw.handle_connection, args=(conn,), daemon=True
        )
        thread.start()
        resp = _send_request(client, {"name": "Bash", "command": "echo ok"})
        assert resp["error"] is False
        client.close()
        time.sleep(0.2)
        assert sw._CONNECTION_SEMAPHORE._value >= initial
    finally:
        server.close()
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_check_has_rg_subprocess_error() -> None:
    sw = tiz.sandbox_worker
    original = sw._HAS_RG
    sw._HAS_RG = None
    try:
        with patch(
            "subprocess.run", side_effect=subprocess.SubprocessError("rg failed")
        ):
            result = sw._check_has_rg()
        assert result is False
    finally:
        sw._HAS_RG = original


def test_main_no_args_in_process() -> None:
    original_argv = sys.argv
    try:
        sys.argv = ["sandbox_worker.py"]
        with patch("sys.exit", side_effect=SystemExit), contextlib.suppress(SystemExit):
            tiz.sandbox_worker.main()
            raise AssertionError()
    finally:
        sys.argv = original_argv


def test_main_too_many_args_in_process() -> None:
    original_argv = sys.argv
    try:
        sys.argv = ["sandbox_worker.py", "a", "b"]
        with patch("sys.exit", side_effect=SystemExit), contextlib.suppress(SystemExit):
            tiz.sandbox_worker.main()
            raise AssertionError()
    finally:
        sys.argv = original_argv


class InterruptibleServer:
    """A socket-like wrapper that accepts connections then raises KeyboardInterrupt after a delay."""

    def __init__(self, accept_once=False):
        self._real = _RealSocket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._accept_once = accept_once
        self._conn = None
        self._interrupt_after = threading.Event()
        self._accept_count = 0
        self._conn_event = threading.Event()

    def bind(self, path):
        self._real.bind(path)
        Path(path).chmod(0o600)
        self._real.listen(5)
        self._real.settimeout(0.05)

        def trigger_interrupt():
            time.sleep(1.0)
            self._interrupt_after.set()

        threading.Thread(target=trigger_interrupt, daemon=True).start()

    def listen(self, n):
        pass

    def accept(self):
        self._accept_count += 1
        while not self._interrupt_after.is_set():
            try:
                conn, addr = self._real.accept()
                if self._accept_once:
                    self._conn = conn
                    self._conn_event.set()
                return conn, addr
            except TimeoutError:
                continue
        raise KeyboardInterrupt

    def close(self):
        self._real.close()

    @property
    def conn(self):
        return self._conn


def test_main_keyboard_interrupt_with_real_socket(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    original_argv = sys.argv
    original_sem = sw._CONNECTION_SEMAPHORE
    sw._CONNECTION_SEMAPHORE = threading.Semaphore(50)
    try:
        sys.argv = ["sandbox_worker.py", socket_path]
        srv = InterruptibleServer()
        with patch("tiz.sandbox_worker.socket.socket", return_value=srv):
            tiz.sandbox_worker.main()
    finally:
        sys.argv = original_argv
        sw._CONNECTION_SEMAPHORE = original_sem
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_main_verbose_flag(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    original_argv = sys.argv
    original_sem = sw._CONNECTION_SEMAPHORE
    sw._CONNECTION_SEMAPHORE = threading.Semaphore(50)
    try:
        sys.argv = ["sandbox_worker.py", "-v", socket_path]
        srv = InterruptibleServer()
        with (
            patch("tiz.sandbox_worker.logging.basicConfig") as mock_bc,
            patch("tiz.sandbox_worker.socket.socket", return_value=srv),
        ):
            tiz.sandbox_worker.main()
            mock_bc.assert_called_once_with(level=logging.INFO)
    finally:
        sys.argv = original_argv
        sw._CONNECTION_SEMAPHORE = original_sem
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_main_debug_flag(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    original_argv = sys.argv
    original_sem = sw._CONNECTION_SEMAPHORE
    sw._CONNECTION_SEMAPHORE = threading.Semaphore(50)
    try:
        sys.argv = ["sandbox_worker.py", "-vv", socket_path]
        srv = InterruptibleServer()
        with (
            patch("tiz.sandbox_worker.logging.basicConfig") as mock_bc,
            patch("tiz.sandbox_worker.socket.socket", return_value=srv),
        ):
            tiz.sandbox_worker.main()
            mock_bc.assert_called_once_with(level=logging.DEBUG)
    finally:
        sys.argv = original_argv
        sw._CONNECTION_SEMAPHORE = original_sem
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_main_unknown_flag() -> None:
    original_argv = sys.argv
    try:
        sys.argv = ["sandbox_worker.py", "--bad"]
        with patch("sys.exit", side_effect=SystemExit), contextlib.suppress(SystemExit):
            tiz.sandbox_worker.main()
            raise AssertionError()
    finally:
        sys.argv = original_argv


def test_main_existing_socket_unlinked(socket_path: str) -> None:
    sw = tiz.sandbox_worker
    original_argv = sys.argv
    original_sem = sw._CONNECTION_SEMAPHORE
    sw._CONNECTION_SEMAPHORE = threading.Semaphore(50)
    try:
        Path(socket_path).touch()
        sys.argv = ["sandbox_worker.py", socket_path]
        srv = InterruptibleServer()
        with patch("tiz.sandbox_worker.socket.socket", return_value=srv):
            tiz.sandbox_worker.main()
        assert not Path(socket_path).exists()
    finally:
        sys.argv = original_argv
        sw._CONNECTION_SEMAPHORE = original_sem
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_if_main_block():
    worker_path = Path(tiz.sandbox_worker.__file__).resolve()
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src"),
    }
    result = subprocess.run(
        [sys.executable, str(worker_path)],
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    assert result.returncode != 0
    assert "Usage:" in result.stderr


def test_main_sigterm_clean_exit(tmp_path: Path) -> None:
    sw = tiz.sandbox_worker
    original_argv = sys.argv
    original_sem = sw._CONNECTION_SEMAPHORE
    sw._CONNECTION_SEMAPHORE = threading.Semaphore(50)
    socket_path = str(tmp_path / "sigterm.sock")
    try:
        sys.argv = ["sandbox_worker.py", socket_path]
        srv = InterruptibleServer()
        with patch("tiz.sandbox_worker.socket.socket", return_value=srv):
            tiz.sandbox_worker.main()
        assert not Path(socket_path).exists()
    finally:
        sys.argv = original_argv
        sw._CONNECTION_SEMAPHORE = original_sem
        if Path(socket_path).exists():
            Path(socket_path).unlink()


def test_signal_handler_registration() -> None:
    original = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        sw = tiz.sandbox_worker
        original_argv = sys.argv
        original_sem = sw._CONNECTION_SEMAPHORE
        sw._CONNECTION_SEMAPHORE = threading.Semaphore(50)
        try:
            sys.argv = ["sandbox_worker.py", "/tmp/test_sigterm_reg.sock"]
            srv = InterruptibleServer()
            with patch("tiz.sandbox_worker.socket.socket", return_value=srv):
                tiz.sandbox_worker.main()
        finally:
            sys.argv = original_argv
            sw._CONNECTION_SEMAPHORE = original_sem
    finally:
        signal.signal(signal.SIGTERM, original)


def test_signal_handler_sys_exit_called() -> None:
    sw = tiz.sandbox_worker
    original_argv = sys.argv
    original_sem = sw._CONNECTION_SEMAPHORE
    sw._CONNECTION_SEMAPHORE = threading.Semaphore(50)
    try:
        sys.argv = ["sandbox_worker.py", "/tmp/test_sigterm_exit.sock"]
        captured = []

        def capture_signal(_sig, handler):
            captured.append(handler)
            return signal.SIG_DFL

        srv = InterruptibleServer()
        with (
            patch("tiz.sandbox_worker.signal.signal", side_effect=capture_signal),
            patch("tiz.sandbox_worker.socket.socket", return_value=srv),
        ):
            sw.main()
        assert len(captured) >= 1
        with pytest.raises(SystemExit) as exc:
            captured[0](signal.SIGTERM, None)
        assert exc.value.code == 0
    finally:
        sys.argv = original_argv
        sw._CONNECTION_SEMAPHORE = original_sem


# ---- _tool_webfetch and related functions ----


def test_rate_limit_domain_first_call() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_LAST_CALL.clear()
    domain = "example.com"
    start = time.monotonic()
    sw._rate_limit_domain(domain)
    elapsed = time.monotonic() - start
    assert elapsed < 0.3
    assert domain in sw._WEBFETCH_LAST_CALL


def test_rate_limit_domain_recent_call() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_LAST_CALL.clear()
    domain = "example.com"
    sw._WEBFETCH_LAST_CALL[domain] = time.monotonic()
    start = time.monotonic()
    sw._rate_limit_domain(domain)
    elapsed = time.monotonic() - start
    assert elapsed > 0.1


def test_cache_get_miss() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    result = sw._cache_get("nonexistent-key")
    assert result is None


def test_cache_get_hit() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    sw._WEBFETCH_CACHE["test-key"] = (200, {"content-type": "text/plain"}, "body")
    result = sw._cache_get("test-key")
    assert result is not None
    assert result[0] == 200
    assert result[1] == {"content-type": "text/plain"}
    assert result[2] == "body"


def test_cache_get_moves_to_end() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    od: OrderedDict = sw._WEBFETCH_CACHE
    od["key1"] = (200, {}, "body1")
    od["key2"] = (200, {}, "body2")
    od["key3"] = (200, {}, "body3")
    sw._cache_get("key1")
    keys = list(od.keys())
    assert keys[-1] == "key1"


def test_cache_set_stores_entry() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    sw._cache_set("test-url", 200, {"content-type": "text"}, "response body")
    assert sw._WEBFETCH_CACHE["test-url"] == (
        200,
        {"content-type": "text"},
        "response body",
    )


def test_cache_set_evicts_oldest() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    max_size = sw._WEBFETCH_CACHE_MAX
    for i in range(max_size + 5):
        sw._cache_set(f"key-{i}", 200, {}, f"body-{i}")
    assert len(sw._WEBFETCH_CACHE) <= max_size
    assert "key-0" not in sw._WEBFETCH_CACHE


def test_webfetch_invalid_url() -> None:
    result, is_error = tiz.sandbox_worker._tool_webfetch({"url": "invalid"})
    assert "url must start with http:// or https://" in result
    assert is_error is True


def test_webfetch_invalid_url_not_string() -> None:
    result, is_error = tiz.sandbox_worker._tool_webfetch({"url": 123})
    assert "url must start with http:// or https://" in result
    assert is_error is True


def test_webfetch_non_int_max_redirects() -> None:
    result, is_error = tiz.sandbox_worker._tool_webfetch(
        {"url": "http://example.com", "max_redirects": "not-an-int"}
    )
    assert "ERROR: max_redirects must be an integer" in result
    assert is_error is True


def test_webfetch_cache_hit() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    cache_key = "GET:http://example.com:"
    cached_entry = (200, {"content-type": "text"}, "cached body")
    sw._WEBFETCH_CACHE[cache_key] = cached_entry
    result, is_error = sw._tool_webfetch({"url": "http://example.com", "raw": False})
    data = json.loads(result)
    assert data["cached"] is True
    assert data["status"] == 200
    assert data["body"] == "cached body"
    assert is_error is False


def test_webfetch_get_request_caches() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text"}
    mock_response.text = "hello"
    with patch.object(sw.requests.Session, "request", return_value=mock_response):
        result, is_error = sw._tool_webfetch(
            {"url": "http://example.com", "method": "GET"}
        )
    data = json.loads(result)
    assert data["url"] == "http://example.com"
    assert data["status"] == 200
    assert "body" in data
    assert is_error is False
    cache_key = "GET:http://example.com:"
    assert cache_key in sw._WEBFETCH_CACHE


def test_webfetch_error_status() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.headers = {}
    mock_response.text = "Not Found"
    with patch.object(sw.requests.Session, "request", return_value=mock_response):
        result, is_error = sw._tool_webfetch({"url": "http://example.com/notfound"})
    data = json.loads(result)
    assert data["status"] == 404
    assert is_error is True


def test_webfetch_post_not_cached() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = "posted ok"
    with patch.object(sw.requests.Session, "request", return_value=mock_response):
        result, is_error = sw._tool_webfetch(
            {"url": "http://example.com", "method": "POST", "body": "data"}
        )
    data = json.loads(result)
    assert data["status"] == 200
    assert data["body"] == "posted ok"
    cache_key = "POST:http://example.com:data"
    assert cache_key not in sw._WEBFETCH_CACHE
    assert is_error is False


def test_webfetch_truncated_body() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = "x" * 200000
    with patch.object(sw.requests.Session, "request", return_value=mock_response):
        result, is_error = sw._tool_webfetch({"url": "http://example.com"})
    data = json.loads(result)
    assert data["truncated"] is True
    assert len(data["body"]) == 100 * 1024
    assert is_error is False


def test_webfetch_raw_truncated_body() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = "x" * (1024 * 1024 + 100)
    with patch.object(sw.requests.Session, "request", return_value=mock_response):
        result, is_error = sw._tool_webfetch({"url": "http://example.com", "raw": True})
    data = json.loads(result)
    assert data["truncated"] is True
    assert len(data["body"]) == 1024 * 1024
    assert "raw=true" not in data["note"]
    assert "1024KB" in data["note"]
    assert is_error is False


def test_webfetch_raw_not_truncated() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = "x" * 5000
    with patch.object(sw.requests.Session, "request", return_value=mock_response):
        result, is_error = sw._tool_webfetch({"url": "http://example.com", "raw": True})
    data = json.loads(result)
    assert data.get("truncated") is None or data["truncated"] is False
    assert len(data["body"]) == 5000
    assert is_error is False


def test_webfetch_timeout() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    with patch.object(
        sw.requests.Session, "request", side_effect=sw.requests.Timeout("timed out")
    ):
        result, is_error = sw._tool_webfetch({"url": "http://example.com"})
    data = json.loads(result)
    assert "timed out" in data["error"]
    assert is_error is True


def test_webfetch_connection_error() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    with patch.object(
        sw.requests.Session,
        "request",
        side_effect=sw.requests.ConnectionError("connection refused"),
    ):
        result, is_error = sw._tool_webfetch({"url": "http://example.com"})
    data = json.loads(result)
    assert "Connection error" in data["error"]
    assert is_error is True


def test_webfetch_request_exception() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    with patch.object(
        sw.requests.Session,
        "request",
        side_effect=sw.requests.RequestException("generic error"),
    ):
        result, is_error = sw._tool_webfetch({"url": "http://example.com"})
    data = json.loads(result)
    assert "Request failed" in data["error"]
    assert is_error is True


def test_webfetch_no_redirects() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = "no redirect"
    mock_session = MagicMock()
    mock_session.request.return_value = mock_response
    with patch.object(sw.requests, "Session", return_value=mock_session):
        result, is_error = sw._tool_webfetch(
            {"url": "http://example.com", "max_redirects": 0}
        )
    data = json.loads(result)
    assert data["status"] == 200
    assert is_error is False


def test_webfetch_non_dict_headers() -> None:
    """When headers param is not a dict, it should return an error."""
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    result, is_error = sw._tool_webfetch(
        {"url": "http://example.com", "headers": "not-a-dict"}
    )
    data = json.loads(result)
    assert "headers must be a dict" in data["error"]
    assert is_error is True


def test_webfetch_invalid_method() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    result, is_error = sw._tool_webfetch(
        {"url": "http://example.com", "method": "TRACE"}
    )
    assert "ERROR: method must be one of" in result
    assert is_error is True


def test_webfetch_raw_not_bool() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    result, is_error = sw._tool_webfetch({"url": "http://example.com", "raw": "yes"})
    assert "ERROR: raw must be a boolean" in result
    assert is_error is True


def test_webfetch_timeout_not_int() -> None:
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    result, is_error = sw._tool_webfetch({"url": "http://example.com", "timeout": "30"})
    assert "ERROR: timeout must be an integer" in result
    assert is_error is True


def test_webfetch_error_status_not_cached() -> None:
    """Error responses should not be cached."""
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {"x-cache": "no"}
    mock_response.text = "Internal Server Error"
    with patch.object(sw.requests.Session, "request", return_value=mock_response):
        result, is_error = sw._tool_webfetch({"url": "http://example.com/error"})
    data = json.loads(result)
    assert data["status"] == 500
    assert is_error is True
    cache_key = "GET:http://example.com/error:"
    assert cache_key not in sw._WEBFETCH_CACHE


def test_webfetch_error_status_cached_for_non_get() -> None:
    """Non-GET requests should never be cached."""
    sw = tiz.sandbox_worker
    sw._WEBFETCH_CACHE.clear()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {}
    mock_response.text = "Error"
    with patch.object(sw.requests.Session, "request", return_value=mock_response):
        result, is_error = sw._tool_webfetch(
            {"url": "http://example.com/error", "method": "POST", "body": "data"}
        )
    data = json.loads(result)
    assert data["status"] == 500
    assert is_error is True
    cache_key = "POST:http://example.com/error:data"
    assert cache_key not in sw._WEBFETCH_CACHE


# ---- _tool_websearch and related functions ----


def test_websearch_rate_limit_first_call() -> None:
    sw = tiz.sandbox_worker
    original = sw._WEBSEARCH_LAST_CALL
    sw._WEBSEARCH_LAST_CALL = 0.0
    start = time.monotonic()
    sw._websearch_rate_limit()
    elapsed = time.monotonic() - start
    assert elapsed < 0.3
    assert sw._WEBSEARCH_LAST_CALL > 0
    sw._WEBSEARCH_LAST_CALL = original


def test_websearch_rate_limit_recent_call() -> None:
    sw = tiz.sandbox_worker
    original = sw._WEBSEARCH_LAST_CALL
    sw._WEBSEARCH_LAST_CALL = time.monotonic()
    start = time.monotonic()
    sw._websearch_rate_limit()
    elapsed = time.monotonic() - start
    assert elapsed > 0.1
    sw._WEBSEARCH_LAST_CALL = original


def test_websearch_cache_get_miss() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    result = sw._websearch_cache_get("nonexistent-key")
    assert result is None


def test_websearch_cache_get_hit() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    sw._WEBSEARCH_CACHE["test-key"] = '{"title": "test"}'
    result = sw._websearch_cache_get("test-key")
    assert result == '{"title": "test"}'


def test_websearch_cache_get_moves_to_end() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    od: OrderedDict = sw._WEBSEARCH_CACHE
    od["key1"] = "v1"
    od["key2"] = "v2"
    od["key3"] = "v3"
    sw._websearch_cache_get("key1")
    keys = list(od.keys())
    assert keys[-1] == "key1"


def test_websearch_cache_set_stores_entry() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    sw._websearch_cache_set("test-key", '{"result": "ok"}')
    assert sw._WEBSEARCH_CACHE["test-key"] == '{"result": "ok"}'


def test_websearch_cache_set_evicts_oldest() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    max_size = sw._WEBSEARCH_CACHE_MAX
    for i in range(max_size + 5):
        sw._websearch_cache_set(f"key-{i}", f"body-{i}")
    assert len(sw._WEBSEARCH_CACHE) <= max_size
    assert "key-0" not in sw._WEBSEARCH_CACHE


def test_parse_ddg_results_empty() -> None:
    sw = tiz.sandbox_worker
    results = sw._parse_ddg_results("")
    assert results == []


def test_parse_ddg_results_no_visible_block() -> None:
    sw = tiz.sandbox_worker
    results = sw._parse_ddg_results("<html><body>no results</body></html>")
    assert results == []


def test_parse_ddg_results_valid_block() -> None:
    sw = tiz.sandbox_worker
    html = (
        "<!-- This is the visible part -->"
        '<div><a class="result__a" href="https://example.com">Example Title</a>'
        '<a class="result__snippet">Example snippet text</a></div></div>'
    )
    results = sw._parse_ddg_results(html)
    assert len(results) == 1
    assert results[0]["title"] == "Example Title"
    assert results[0]["href"] == "https://example.com"
    assert results[0]["body"] == "Example snippet text"


def test_parse_ddg_results_with_uddg() -> None:
    sw = tiz.sandbox_worker
    html = (
        "<!-- This is the visible part -->"
        '<div><a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Freal.example.com">Real Title</a>'
        '<a class="result__snippet">Real snippet</a></div></div>'
    )
    results = sw._parse_ddg_results(html)
    assert len(results) == 1
    assert results[0]["href"] == "https://real.example.com"
    assert results[0]["title"] == "Real Title"
    assert results[0]["body"] == "Real snippet"


def test_parse_ddg_results_multiple_results() -> None:
    sw = tiz.sandbox_worker
    html = (
        "<!-- This is the visible part -->"
        '<div><a class="result__a" href="https://a.com">A</a>'
        '<a class="result__snippet">snippet a</a></div></div>'
        "<!-- This is the visible part -->"
        '<div><a class="result__a" href="https://b.com">B</a>'
        '<a class="result__snippet">snippet b</a></div></div>'
    )
    results = sw._parse_ddg_results(html)
    assert len(results) == 2
    assert results[0]["title"] == "A"
    assert results[1]["title"] == "B"


def test_parse_ddg_results_no_snippet() -> None:
    sw = tiz.sandbox_worker
    html = (
        "<!-- This is the visible part -->"
        '<div><a class="result__a" href="https://example.com">Title</a></div></div>'
    )
    results = sw._parse_ddg_results(html)
    assert len(results) == 1
    assert results[0]["title"] == "Title"
    assert results[0]["body"] == ""


def test_parse_ddg_results_skip_block_without_title() -> None:
    sw = tiz.sandbox_worker
    html = "<!-- This is the visible part --><div><span>no link here</span></div></div>"
    results = sw._parse_ddg_results(html)
    assert results == []


def test_websearch_empty_query() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch({"query": ""})
    assert "ERROR" in result
    assert is_error is True


def test_websearch_missing_query() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch({})
    assert "ERROR" in result
    assert is_error is True


def test_websearch_query_not_string() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch({"query": 123})
    assert "ERROR" in result
    assert is_error is True


def test_websearch_non_int_page() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch(
        {"query": "test", "page": "abc"}
    )
    assert "ERROR" in result
    assert "page" in result
    assert is_error is True


def test_websearch_page_too_low() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch({"query": "test", "page": 0})
    assert "ERROR" in result
    assert "page" in result
    assert is_error is True


def test_websearch_non_int_max_results() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch(
        {"query": "test", "max_results": "abc"}
    )
    assert "ERROR" in result
    assert "max_results" in result
    assert is_error is True


def test_websearch_max_results_too_low() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch(
        {"query": "test", "max_results": 0}
    )
    assert "ERROR" in result
    assert "max_results" in result
    assert is_error is True


def test_websearch_max_results_capped() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp):
        result, is_error = sw._tool_websearch({"query": "test", "max_results": 100})
    data = json.loads(result)
    assert len(data["results"]) == 0
    assert is_error is False


def test_websearch_non_int_timeout() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch(
        {"query": "test", "timeout": "abc"}
    )
    assert "ERROR" in result
    assert "timeout" in result
    assert is_error is True


def test_websearch_timeout_too_low() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch(
        {"query": "test", "timeout": 0}
    )
    assert "ERROR" in result
    assert "timeout" in result
    assert is_error is True


def test_websearch_timeout_too_high() -> None:
    result, is_error = tiz.sandbox_worker._tool_websearch(
        {"query": "test", "timeout": 31}
    )
    assert "ERROR" in result
    assert "timeout" in result
    assert is_error is True


def test_websearch_cache_hit() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    cache_key = "test|wt-wt||1|off"
    cached_data = json.dumps(
        [{"title": "Cached", "href": "https://cached.com", "body": "cached"}]
    )
    sw._WEBSEARCH_CACHE[cache_key] = cached_data
    result, is_error = sw._tool_websearch({"query": "test"})
    data = json.loads(result)
    assert data["cached"] is True
    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Cached"
    assert is_error is False


def test_websearch_cache_hit_limited_by_max_results() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    cache_key = "test|wt-wt||1|off"
    results_list = [
        {"title": f"Result {i}", "href": f"https://x.com/{i}", "body": f"body {i}"}
        for i in range(20)
    ]
    sw._WEBSEARCH_CACHE[cache_key] = json.dumps(results_list)
    result, is_error = sw._tool_websearch({"query": "test", "max_results": 5})
    data = json.loads(result)
    assert data["cached"] is True
    assert len(data["results"]) == 5
    assert is_error is False


def test_websearch_successful_request() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    html = (
        "<!-- This is the visible part -->"
        '<div><a class="result__a" href="https://example.com">Example</a>'
        '<a class="result__snippet">Snippet</a></div></div>'
    )
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp):
        result, is_error = sw._tool_websearch({"query": "test query"})
    data = json.loads(result)
    assert data["query"] == "test query"
    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Example"
    assert data["results"][0]["href"] == "https://example.com"
    assert data["results"][0]["body"] == "Snippet"
    assert is_error is False


def test_websearch_with_region_and_timelimit() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch(
            {"query": "test", "region": "de-de", "timelimit": "w"}
        )
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["data"]["q"] == "test"
    assert call_kwargs["data"]["l"] == "de-de"
    assert call_kwargs["data"]["df"] == "w"


def test_websearch_with_page_2() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch({"query": "test", "page": 2})
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["data"]["s"] == "15"


def test_websearch_with_page_3() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch({"query": "test", "page": 3})
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["data"]["s"] == "30"


def test_websearch_timeout_error() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    with patch.object(
        sw.requests, "post", side_effect=sw.requests.Timeout("timed out")
    ):
        result, is_error = sw._tool_websearch({"query": "test"})
    data = json.loads(result)
    assert "Request timed out" in data["error"]
    assert is_error is True


def test_websearch_connection_error() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    with patch.object(
        sw.requests,
        "post",
        side_effect=sw.requests.ConnectionError("connection refused"),
    ):
        result, is_error = sw._tool_websearch({"query": "test"})
    data = json.loads(result)
    assert "Connection error" in data["error"]
    assert is_error is True


def test_websearch_request_exception() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    with patch.object(
        sw.requests, "post", side_effect=sw.requests.RequestException("generic error")
    ):
        result, is_error = sw._tool_websearch({"query": "test"})
    data = json.loads(result)
    assert "Request failed" in data["error"]
    assert is_error is True


def test_websearch_custom_user_agent() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch(
            {"query": "test", "user_agent": "CustomAgent/1.0"}
        )
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    call_headers = mock_post.call_args[1]["headers"]
    assert call_headers["User-Agent"] == "CustomAgent/1.0"


def test_websearch_default_user_agent() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch({"query": "test"})
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    call_headers = mock_post.call_args[1]["headers"]
    assert "tiz" in call_headers["User-Agent"]


def test_websearch_safe_search_off_sends_kp_1() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch({"query": "test", "safe_search": "off"})
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    assert mock_post.call_args[1]["data"]["kp"] == "1"


def test_websearch_safe_search_on_sends_kp_minus2() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch({"query": "test", "safe_search": "on"})
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    assert mock_post.call_args[1]["data"]["kp"] == "-2"


def test_websearch_safe_search_moderate_sends_kp_minus1() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch(
            {"query": "test", "safe_search": "moderate"}
        )
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    assert mock_post.call_args[1]["data"]["kp"] == "-1"


def test_websearch_default_safe_search_off() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch({"query": "test"})
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    assert mock_post.call_args[1]["data"]["kp"] == "1"


def test_websearch_safe_search_off_in_cache_key() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    cache_key = "test|wt-wt||1|off"
    cached_data = json.dumps([{"title": "hit", "href": "https://x.com", "body": "b"}])
    sw._WEBSEARCH_CACHE[cache_key] = cached_data
    result, is_error = sw._tool_websearch({"query": "test", "safe_search": "off"})
    data = json.loads(result)
    assert data["cached"] is True
    assert is_error is False


def test_websearch_safe_search_on_in_cache_key() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    cache_key = "test|wt-wt||1|on"
    cached_data = json.dumps(
        [{"title": "safe hit", "href": "https://y.com", "body": "b"}]
    )
    sw._WEBSEARCH_CACHE[cache_key] = cached_data
    result, is_error = sw._tool_websearch({"query": "test", "safe_search": "on"})
    data = json.loads(result)
    assert data["cached"] is True
    assert data["results"][0]["title"] == "safe hit"
    assert is_error is False


def test_websearch_safe_search_moderate_in_cache_key() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    cache_key = "test|wt-wt||1|moderate"
    cached_data = json.dumps(
        [{"title": "mod hit", "href": "https://z.com", "body": "b"}]
    )
    sw._WEBSEARCH_CACHE[cache_key] = cached_data
    result, is_error = sw._tool_websearch({"query": "test", "safe_search": "moderate"})
    data = json.loads(result)
    assert data["cached"] is True
    assert data["results"][0]["title"] == "mod hit"
    assert is_error is False


def test_websearch_safe_search_unknown_falls_through() -> None:
    sw = tiz.sandbox_worker
    sw._WEBSEARCH_CACHE.clear()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    with patch.object(sw.requests, "post", return_value=mock_resp) as mock_post:
        result, is_error = sw._tool_websearch(
            {"query": "test", "safe_search": "unknown_value"}
        )
    data = json.loads(result)
    assert "results" in data
    assert is_error is False
    assert "kp" not in mock_post.call_args[1]["data"]
