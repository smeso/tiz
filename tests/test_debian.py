"""Tests for Debian packaging metadata and build scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Verify man page can be generated
# ---------------------------------------------------------------------------


def test_manpage_generation(tmp_path):
    """Verify the man page can be generated from the CLI module."""
    result = subprocess.run(
        [
            "argparse-manpage",
            "--module",
            "tiz.cli",
            "--function",
            "get_parser",
            "--prog",
            "tiz",
            "--output",
            str(tmp_path / "tiz.1"),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = (tmp_path / "tiz.1").read_text()
    assert ".TH TIZ" in output or ".TH tiz" in output
    assert "tiz" in output


# ---------------------------------------------------------------------------
# Verify shell completions can be generated
# ---------------------------------------------------------------------------


def test_shell_completions_generation():
    """Verify shell completions can be generated from autocomplete module."""
    src_path = str(PROJECT_ROOT / "src")
    for shell in ("bash", "zsh", "fish", "tcsh"):
        script = (
            "import os, sys\n"
            "sys.path.insert(0, os.environ['TIZ_SRC_PATH'])\n"
            "from tiz.autocomplete import shellcode\n"
            "print(shellcode(os.environ['TIZ_SHELL']))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "TIZ_SRC_PATH": src_path, "TIZ_SHELL": shell},
            timeout=30,
        )
        assert result.returncode == 0, f"{shell} failed: {result.stderr}"
        output = result.stdout
        assert len(output) > 0, f"{shell} produced empty output"
        if shell == "bash":
            assert "complete" in output
        elif shell == "zsh":
            assert "compdef" in output or "complete" in output
        elif shell == "fish":
            assert "_ARGCOMPLETE" in output


# ---------------------------------------------------------------------------
# Verify .gitignore includes debian build artifacts
# ---------------------------------------------------------------------------


def test_gitignore_has_debian_artifacts():
    content = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = content.splitlines()
    assert any(line.strip() == "build/" for line in lines)
