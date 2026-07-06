"""Tests for Debian packaging metadata and build scripts."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    full = PROJECT_ROOT / path
    assert full.exists(), f"File not found: {full}"
    return full.read_text(encoding="utf-8")


def _exists(path: str) -> bool:
    return (PROJECT_ROOT / path).exists()


# ---------------------------------------------------------------------------
# Required debian directory structure
# ---------------------------------------------------------------------------


def test_debian_directory_exists():
    assert (PROJECT_ROOT / "debian").is_dir()


def test_debian_changelog_exists():
    assert _exists("debian/changelog")


def test_debian_control_exists():
    assert _exists("debian/control")


def test_debian_copyright_exists():
    assert _exists("debian/copyright")


def test_debian_rules_exists():
    assert _exists("debian/rules")


def test_debian_rules_is_executable():
    st = (PROJECT_ROOT / "debian/rules").stat()
    assert st.st_mode & stat.S_IXUSR


def test_debian_source_format_exists():
    assert _exists("debian/source/format")


def test_debian_watch_exists():
    assert _exists("debian/watch")


def test_debian_tiz_docs_exists():
    assert _exists("debian/tiz.docs")


# ---------------------------------------------------------------------------
# control file fields
# ---------------------------------------------------------------------------


def test_control_source():
    content = _read("debian/control")
    assert "Source: tiz" in content


def test_control_section():
    content = _read("debian/control")
    assert "Section: python" in content


def test_control_maintainer():
    content = _read("debian/control")
    assert "Maintainer: Salvatore Mesoraca <s.mesoraca16@gmail.com>" in content


def test_control_build_depends():
    content = _read("debian/control")
    assert "Build-Depends:" in content
    assert "debhelper-compat" in content
    assert "dh-python" in content
    assert "python3-all" in content
    assert "python3-setuptools" in content
    assert "python3-wheel" in content
    assert "python3-argparse-manpage" in content
    assert "python3-argcomplete" in content
    assert "python3-requests" in content
    assert "python3-jinja2" in content
    assert "python3-yaml" in content
    assert "python3-git" in content


def test_control_package():
    content = _read("debian/control")
    assert "Package: tiz" in content


def test_control_architecture():
    content = _read("debian/control")
    assert "Architecture: all" in content


def test_control_depends():
    content = _read("debian/control")
    assert "Depends:" in content
    assert "${python3:Depends}" in content
    assert "${misc:Depends}" in content
    assert "python3-requests" in content
    assert "python3-jinja2" in content
    assert "python3-yaml" in content
    assert "python3-git" in content
    assert "python3-argcomplete" in content
    assert "podman" in content


def test_control_suggests():
    content = _read("debian/control")
    assert "Suggests:" in content
    assert "python3-pyaudio" in content
    assert "python3-htmlmin" in content
    assert "python3-rjsmin" in content


def test_control_standards_version():
    content = _read("debian/control")
    assert "Standards-Version: 4.6.2" in content


def test_control_homepage():
    content = _read("debian/control")
    assert "Homepage: https://github.com/smeso/tiz" in content


def test_control_rules_requires_root():
    content = _read("debian/control")
    assert "Rules-Requires-Root: no" in content


def test_control_priority():
    content = _read("debian/control")
    assert "Priority: optional" in content


def test_control_description():
    content = _read("debian/control")
    assert "Description:" in content
    assert "Agentic AI chatbot with sandboxed tool execution" in content


# ---------------------------------------------------------------------------
# changelog
# ---------------------------------------------------------------------------


def test_changelog_has_package():
    content = _read("debian/changelog")
    assert content.startswith("tiz ")


def test_changelog_has_version():
    content = _read("debian/changelog")
    assert "0.1.0-1" in content


def test_changelog_has_distribution():
    content = _read("debian/changelog")
    assert "unstable" in content


# ---------------------------------------------------------------------------
# copyright
# ---------------------------------------------------------------------------


def test_copyright_format():
    content = _read("debian/copyright")
    assert "Format:" in content
    assert "Upstream-Name: tiz" in content


def test_copyright_license():
    content = _read("debian/copyright")
    assert "Apache-2.0" in content
    assert "License:" in content


def test_copyright_source():
    content = _read("debian/copyright")
    assert "Source:" in content
    assert "github.com" in content


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


def test_rules_shebang():
    content = _read("debian/rules")
    assert content.startswith("#!/usr/bin/make -f")


def test_rules_uses_dh():
    content = _read("debian/rules")
    assert "%:" in content
    assert "dh $@" in content


def test_rules_generates_manpage():
    content = _read("debian/rules")
    assert "argparse-manpage" in content
    assert "tiz.cli" in content
    assert "tiz.1" in content


def test_rules_generates_completions():
    content = _read("debian/rules")
    for shell_marker in ("completions/bash", "completions/zsh", "completions/fish"):
        assert shell_marker in content


def test_rules_installs_manpage():
    content = _read("debian/rules")
    assert "man/man1" in content


def test_rules_installs_completions():
    content = _read("debian/rules")
    assert "bash-completion/completions" in content
    assert "zsh/vendor-completions" in content
    assert "fish/vendor_completions" in content


# ---------------------------------------------------------------------------
# source/format
# ---------------------------------------------------------------------------


def test_source_format_content():
    content = _read("debian/source/format")
    assert content.strip() == "3.0 (quilt)"


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


def test_watch_content():
    content = _read("debian/watch")
    assert "version=4" in content
    assert "github.com/smeso/tiz" in content


# ---------------------------------------------------------------------------
# tiz.docs
# ---------------------------------------------------------------------------


def test_docs_content():
    content = _read("debian/tiz.docs")
    assert "README.md" in content


# ---------------------------------------------------------------------------
# scripts/mkdeb.sh
# ---------------------------------------------------------------------------


def test_mkdeb_script_exists():
    assert _exists("scripts/mkdeb.sh")


def test_mkdeb_script_is_executable():
    st = (PROJECT_ROOT / "scripts/mkdeb.sh").stat()
    assert st.st_mode & stat.S_IXUSR


def test_mkdeb_script_shebang():
    content = _read("scripts/mkdeb.sh")
    assert content.startswith("#!/bin/bash")


def test_mkdeb_script_uses_buildpackage():
    content = _read("scripts/mkdeb.sh")
    assert "dpkg-buildpackage" in content


def test_mkdeb_script_uses_lintian():
    content = _read("scripts/mkdeb.sh")
    assert "lintian" in content


def test_mkdeb_script_uses_dpkg_deb():
    content = _read("scripts/mkdeb.sh")
    assert "dpkg-deb" in content


def test_mkdeb_script_error_handling():
    content = _read("scripts/mkdeb.sh")
    assert "set -euo pipefail" in content


def test_mkdeb_script_changes_to_project_root():
    content = _read("scripts/mkdeb.sh")
    assert 'realpath "${BASH_SOURCE[0]}"' in content
    assert "projdir=" in content


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
    content = _read(".gitignore")
    lines = content.splitlines()
    assert any(line.strip() == "build/" for line in lines)
