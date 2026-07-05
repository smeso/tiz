"""Tests for tiz.__main__."""

from __future__ import annotations

import subprocess
import sys as _sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_main_module_help_exit_code() -> None:
    """Running python -m tiz --help should exit with code 0 and show usage: tiz."""
    p = subprocess.run(
        [_sys.executable, "-m", "tiz", "--help"],
        capture_output=True,
        text=True,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "src",
            "NO_COLOR": "1",
        },
        cwd=_PROJECT_ROOT,
    )
    assert p.returncode == 0, f"stdout: {p.stdout}, stderr: {p.stderr}"
    assert "usage: tiz" in p.stdout
    assert p.stderr == ""


def test_main_module_exit_code_non_zero() -> None:
    """Running python -m tiz with missing args should exit with code 2."""
    p = subprocess.run(
        [_sys.executable, "-m", "tiz", "--invalid-flag"],
        capture_output=True,
        text=True,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "src",
            "NO_COLOR": "1",
        },
        cwd=_PROJECT_ROOT,
    )
    assert p.returncode == 2
    assert p.stdout == ""
    assert "usage: tiz" in p.stderr


def test_main_module_import_smoke() -> None:
    """Verify that tiz.__main__ can be imported and has the expected attributes."""
    import tiz.__main__

    assert callable(tiz.__main__.main)
    assert callable(tiz.__main__._main_entry)


def test_main_entry_calls_sys_exit() -> None:
    """_main_entry() should call sys.exit(main())."""
    import tiz.__main__

    with patch("tiz.__main__.main", return_value=0) as mock_main:
        with pytest.raises(SystemExit) as exc_info:
            tiz.__main__._main_entry()
        assert exc_info.value.code == 0
        mock_main.assert_called_once_with()


def test_main_entry_propagates_error_code() -> None:
    """_main_entry() should propagate non-zero exit codes."""
    import tiz.__main__

    with patch("tiz.__main__.main", return_value=42) as mock_main:
        with pytest.raises(SystemExit) as exc_info:
            tiz.__main__._main_entry()
        assert exc_info.value.code == 42
        mock_main.assert_called_once_with()
