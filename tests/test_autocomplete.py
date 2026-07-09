# mypy: disable-error-code="no-untyped-def,arg-type,no-any-return,attr-defined"
# ruff: noqa: ARG001, ARG002, ARG005, ANN201, B007, SIM117, SIM222
"""Tests for tiz.autocomplete module."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tiz.autocomplete import (
    _image_tag_completer,
    _manifest_path_completer,
    _sandbox_name_completer,
    _set_sandbox_completers,
    autocomplete,
    shellcode,
)

# ---------------------------------------------------------------------------
# _sandbox_name_completer tests
# ---------------------------------------------------------------------------


def test_sandbox_name_completer_returns_matching_names():
    """Completer returns sandbox names matching the prefix."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with (
        patch(
            "tiz.autocomplete.SandboxDirs.list_all",
            return_value=["alpha", "beta", "gamma"],
        ),
        patch(
            "tiz.autocomplete.SandboxManager.available_engine", return_value="docker"
        ),
    ):
        result = _sandbox_name_completer("a", parsed)

    assert result == ["alpha"]


def test_sandbox_name_completer_no_match():
    """Completer returns empty list when no sandbox names match."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with (
        patch("tiz.autocomplete.SandboxDirs.list_all", return_value=["alpha", "beta"]),
        patch(
            "tiz.autocomplete.SandboxManager.available_engine", return_value="docker"
        ),
    ):
        result = _sandbox_name_completer("z", parsed)

    assert result == []


def test_sandbox_name_completer_empty_prefix():
    """Completer returns all sandbox names when prefix is empty."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with (
        patch("tiz.autocomplete.SandboxDirs.list_all", return_value=["alpha", "beta"]),
        patch(
            "tiz.autocomplete.SandboxManager.available_engine", return_value="docker"
        ),
    ):
        result = _sandbox_name_completer("", parsed)

    assert result == ["alpha", "beta"]


def test_sandbox_name_completer_no_config_dir_attr():
    """Completer defaults to ~/.tiz when parsed_args has no config_dir."""
    parsed = argparse.Namespace()

    with (
        patch("tiz.autocomplete.SandboxDirs.list_all", return_value=["alpha"]),
        patch(
            "tiz.autocomplete.SandboxManager.available_engine", return_value="docker"
        ),
        patch("tiz.autocomplete.Path.home", return_value=Path("/fake_home")),
    ):
        result = _sandbox_name_completer("", parsed)

    assert result == ["alpha"]


def test_sandbox_name_completer_config_dir_as_str():
    """Completer handles config_dir being a string."""
    parsed = argparse.Namespace()
    parsed.config_dir = "/tmp/fake_tiz"

    with (
        patch("tiz.autocomplete.SandboxDirs.list_all", return_value=["alpha"]),
        patch(
            "tiz.autocomplete.SandboxManager.available_engine", return_value="docker"
        ),
    ):
        result = _sandbox_name_completer("", parsed)

    assert result == ["alpha"]


def test_sandbox_name_completer_config_dir_none():
    """Completer handles config_dir being None."""
    parsed = argparse.Namespace()
    parsed.config_dir = None

    with (
        patch("tiz.autocomplete.SandboxDirs.list_all", return_value=["alpha"]),
        patch(
            "tiz.autocomplete.SandboxManager.available_engine", return_value="docker"
        ),
        patch("tiz.autocomplete.Path.home", return_value=Path("/fake_home")),
    ):
        result = _sandbox_name_completer("", parsed)

    assert result == ["alpha"]


def test_sandbox_name_completer_no_engine():
    """Completer returns empty list when no container engine is available."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with patch("tiz.autocomplete.SandboxManager.available_engine", return_value=None):
        result = _sandbox_name_completer("", parsed)

    assert result == []


def test_sandbox_name_completer_exception_returns_empty():
    """Completer returns empty list on any exception."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with patch(
        "tiz.autocomplete.SandboxManager.available_engine",
        side_effect=RuntimeError("boom"),
    ):
        result = _sandbox_name_completer("", parsed)

    assert result == []


def test_sandbox_name_completer_preserves_order():
    """Completer preserves the (already sorted) order from SandboxDirs.list_all."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with (
        patch(
            "tiz.autocomplete.SandboxDirs.list_all",
            return_value=["alpha", "beta", "gamma"],
        ),
        patch(
            "tiz.autocomplete.SandboxManager.available_engine", return_value="docker"
        ),
    ):
        result = _sandbox_name_completer("", parsed)

    assert result == ["alpha", "beta", "gamma"]


def test_sandbox_name_completer_passes_sandboxes_dir():
    """Completer passes config_dir/sandboxes to SandboxDirs.list_all."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with (
        patch("tiz.autocomplete.SandboxDirs.list_all", return_value=[]) as mock_list,
        patch(
            "tiz.autocomplete.SandboxManager.available_engine", return_value="docker"
        ),
    ):
        _sandbox_name_completer("", parsed)

    mock_list.assert_called_once_with(Path("/tmp/fake_tiz/sandboxes"))


# ---------------------------------------------------------------------------
# _image_tag_completer tests
# ---------------------------------------------------------------------------


def test_image_tag_completer_returns_matching_tags(tmp_path):
    """Completer returns image tags matching the prefix from containerfiles dirs."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()
    (mock_dir / "Containerfile.tiz-worker-py").touch()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir],
    ):
        result = _image_tag_completer("tiz-worker-js", parsed)

    assert result == ["tiz-worker-js"]


def test_image_tag_completer_no_match(tmp_path):
    """Completer returns empty list when no tags match the prefix."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir],
    ):
        result = _image_tag_completer("tiz-worker-py", parsed)

    assert result == []


def test_image_tag_completer_empty_prefix(tmp_path):
    """Completer returns all tags when prefix is empty."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()
    (mock_dir / "Containerfile.tiz-worker-py").touch()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir],
    ):
        result = _image_tag_completer("", parsed)

    assert result == ["tiz-worker-js", "tiz-worker-py"]


def test_image_tag_completer_no_config_dir_attr(tmp_path):
    """Completer defaults to ~/.tiz when parsed_args has no config_dir."""
    parsed = argparse.Namespace()

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()

    with (
        patch(
            "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
            return_value=[mock_dir],
        ),
        patch("tiz.autocomplete.Path.home", return_value=Path("/fake_home")),
    ):
        result = _image_tag_completer("", parsed)

    assert result == ["tiz-worker-js"]


def test_image_tag_completer_config_dir_as_str(tmp_path):
    """Completer handles config_dir being a string."""
    parsed = argparse.Namespace()
    parsed.config_dir = "/tmp/fake_tiz"

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir],
    ):
        result = _image_tag_completer("", parsed)

    assert result == ["tiz-worker-js"]


def test_image_tag_completer_config_dir_none(tmp_path):
    """Completer handles config_dir being None."""
    parsed = argparse.Namespace()
    parsed.config_dir = None

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()

    with (
        patch(
            "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
            return_value=[mock_dir],
        ),
        patch("tiz.autocomplete.Path.home", return_value=Path("/fake_home")),
    ):
        result = _image_tag_completer("", parsed)

    assert result == ["tiz-worker-js"]


def test_image_tag_completer_empty_dirs():
    """Completer returns empty list when containerfiles dirs are empty."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[],
    ):
        result = _image_tag_completer("", parsed)

    assert result == []


def test_image_tag_completer_non_existent_dir():
    """Completer skips non-existent directories."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    fake_dir = Path("/tmp/nonexistent_dir_for_testing")
    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[fake_dir],
    ):
        result = _image_tag_completer("", parsed)

    assert result == []


def test_image_tag_completer_returns_sorted(tmp_path):
    """Completer returns tags in sorted order and deduplicates."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()
    (mock_dir / "Containerfile.tiz-worker-py").touch()
    # Duplicate via a second dir
    mock_dir2 = tmp_path / "containerfiles2"
    mock_dir2.mkdir(parents=True, exist_ok=True)
    (mock_dir2 / "Containerfile.tiz-worker-js").touch()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir, mock_dir2],
    ):
        result = _image_tag_completer("", parsed)

    assert result == ["tiz-worker-js", "tiz-worker-py"]


def test_image_tag_completer_exception_returns_empty():
    """Completer returns empty list on any exception."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        side_effect=RuntimeError("boom"),
    ):
        result = _image_tag_completer("", parsed)

    assert result == []


def test_image_tag_completer_non_containerfile_skipped(tmp_path):
    """Files not starting with 'Containerfile.' are skipped."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()
    (mock_dir / "README.md").touch()
    (mock_dir / "Dockerfile.other").touch()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir],
    ):
        result = _image_tag_completer("", parsed)

    assert result == ["tiz-worker-js"]


def test_image_tag_completer_skips_directories(tmp_path):
    """Directories named like Containerfile.xxx are skipped."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.tiz-worker-js").touch()
    (mock_dir / "Containerfile.subdir").mkdir()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir],
    ):
        result = _image_tag_completer("", parsed)

    assert result == ["tiz-worker-js"]


def test_image_tag_completer_skips_empty_suffix(tmp_path):
    """File named exactly 'Containerfile.' (no suffix) is skipped."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.").touch()
    (mock_dir / "Containerfile.valid").touch()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir],
    ):
        result = _image_tag_completer("", parsed)

    assert result == ["valid"]


def test_image_tag_completer_calls_get_containerfiles_dirs(tmp_path):
    """Verify get_containerfiles_dirs is called with the resolved config dir."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.test").touch()

    with patch(
        "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
        return_value=[mock_dir],
    ) as mock_get:
        _image_tag_completer("", parsed)

    mock_get.assert_called_once_with(Path("/tmp/fake_tiz"))


def test_image_tag_completer_calls_get_containerfiles_dirs_default(tmp_path):
    """Verify get_containerfiles_dirs is called with default config dir when missing."""
    parsed = argparse.Namespace()

    mock_dir = tmp_path / "containerfiles"
    mock_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "Containerfile.test").touch()

    with (
        patch(
            "tiz.autocomplete.SandboxManager.get_containerfiles_dirs",
            return_value=[mock_dir],
        ) as mock_get,
        patch("tiz.autocomplete.Path.home", return_value=Path("/fake_home")),
    ):
        _image_tag_completer("", parsed)

    mock_get.assert_called_once_with(Path("/fake_home") / ".tiz")


# ---------------------------------------------------------------------------
# _manifest_path_completer tests
# ---------------------------------------------------------------------------


def test_manifest_path_completer_cwd(tmp_path):
    """Completer returns file names from CWD matching the prefix."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    (tmp_path / "my-manifest.yaml").touch()
    (tmp_path / "other.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("my", parsed)

    assert result == ["my-manifest.yaml"]


def test_manifest_path_completer_manifest_dir(tmp_path):
    """Completer returns file names from config_dir/manifests matching the prefix."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (manifest_dir / "workflow.yaml").touch()
    (manifest_dir / "other.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("workflow", parsed)

    assert result == ["workflow.yaml"]


def test_manifest_path_completer_both_dirs(tmp_path):
    """Completer deduplicates results from CWD and manifest dir."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (tmp_path / "common.yaml").touch()
    (manifest_dir / "common.yaml").touch()
    (manifest_dir / "unique.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("", parsed)

    assert result == ["common.yaml", "unique.yaml"]


def test_manifest_path_completer_empty_prefix(tmp_path):
    """Completer returns all files when prefix is empty."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (tmp_path / "a.yaml").touch()
    (tmp_path / "b.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("", parsed)

    assert result == ["a.yaml", "b.yaml"]


def test_manifest_path_completer_no_match(tmp_path):
    """Completer returns empty list when no files match the prefix."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (tmp_path / "a.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("z", parsed)

    assert result == []


def test_manifest_path_completer_no_config_dir_attr(tmp_path):
    """Completer defaults to ~/.tiz when parsed_args has no config_dir."""
    parsed = argparse.Namespace()

    (tmp_path / "my.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
        patch("tiz.autocomplete.Path.home", return_value=Path("/fake_home")),
    ):
        result = _manifest_path_completer("my", parsed)

    assert result == ["my.yaml"]


def test_manifest_path_completer_config_dir_as_str(tmp_path):
    """Completer handles config_dir being a string."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = str(config_dir)

    (manifest_dir / "my.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("my", parsed)

    assert result == ["my.yaml"]


def test_manifest_path_completer_config_dir_none(tmp_path):
    """Completer handles config_dir being None."""
    parsed = argparse.Namespace()
    parsed.config_dir = None

    (tmp_path / "my.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
        patch("tiz.autocomplete.Path.home", return_value=Path("/fake_home")),
    ):
        result = _manifest_path_completer("my", parsed)

    assert result == ["my.yaml"]


def test_manifest_path_completer_exception_returns_empty():
    """Completer returns empty list on any exception."""
    parsed = argparse.Namespace()
    parsed.config_dir = Path("/tmp/fake_tiz")

    with patch("tiz.autocomplete.Path.cwd", side_effect=RuntimeError("boom")):
        result = _manifest_path_completer("", parsed)

    assert result == []


def test_manifest_path_completer_non_existent_manifest_dir(tmp_path):
    """Completer returns only CWD results when manifest dir does not exist."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (tmp_path / "my.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("", parsed)

    assert result == ["my.yaml"]


def test_manifest_path_completer_returns_sorted(tmp_path):
    """Completer returns results in sorted order."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (tmp_path / "z.yaml").touch()
    (tmp_path / "a.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("", parsed)

    assert result == ["a.yaml", "z.yaml"]


def test_manifest_path_completer_directories_skipped(tmp_path):
    """Completer skips directories, only returns files."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (tmp_path / "file.yaml").touch()
    (tmp_path / "subdir").mkdir()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("", parsed)

    assert result == ["file.yaml"]


def test_manifest_path_completer_cwd_is_file(tmp_path):
    """Completer handles case where CWD is a file (unlikely but for coverage)."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    cwd_file = tmp_path / "cwd_file.txt"
    cwd_file.touch()

    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "found.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=cwd_file),
    ):
        result = _manifest_path_completer("", parsed)

    assert result == ["found.yaml"]


def test_manifest_path_completer_with_relative_path_prefix(tmp_path):
    """Completer handles path-like prefix like './my-' preserving the directory portion."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (tmp_path / "my-manifest.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("./my", parsed)

    assert result == ["./my-manifest.yaml"]


def test_manifest_path_completer_manifest_dir_returns_basename(tmp_path):
    """Completer returns only basenames from manifest dir (not full path)."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (manifest_dir / "workflow.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("workflow", parsed)

    assert result == ["workflow.yaml"]


def test_manifest_path_completer_manifest_dir_prefix_mismatch(tmp_path):
    """Manifest dir uses prefix directly (not base_prefix) so './x' doesn't match basename 'x'."""
    config_dir = tmp_path / ".tiz"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    parsed = argparse.Namespace()
    parsed.config_dir = config_dir

    (manifest_dir / "my-manifest.yaml").touch()

    with (
        patch("tiz.autocomplete.Path.cwd", return_value=tmp_path),
    ):
        result = _manifest_path_completer("./my", parsed)

    # Manifest dir uses child.name.startswith(prefix) where prefix is "./my",
    # which won't match "my-manifest.yaml" — so only CWD entry is returned.
    # Since CWD has no files matching, result is empty.
    assert result == []


# ---------------------------------------------------------------------------
# _resolve_config_dir tests
# ---------------------------------------------------------------------------


def test_resolve_config_dir_returns_path():
    """_resolve_config_dir returns a proper Path when given a Path."""
    from tiz.autocomplete import _resolve_config_dir

    parsed = argparse.Namespace()
    parsed.config_dir = Path("/some/path")
    result = _resolve_config_dir(parsed)
    assert result == Path("/some/path")


def test_resolve_config_dir_from_str():
    """_resolve_config_dir converts a string to Path."""
    from tiz.autocomplete import _resolve_config_dir

    parsed = argparse.Namespace()
    parsed.config_dir = "/some/path"
    result = _resolve_config_dir(parsed)
    assert result == Path("/some/path")


def test_resolve_config_dir_default():
    """_resolve_config_dir returns default when attribute is missing."""
    from tiz.autocomplete import _resolve_config_dir

    parsed = argparse.Namespace()
    with patch("tiz.autocomplete.Path.home", return_value=Path("/home/user")):
        result = _resolve_config_dir(parsed)
    assert result == Path("/home/user") / ".tiz"


def test_resolve_config_dir_none():
    """_resolve_config_dir returns default when config_dir is None."""
    from tiz.autocomplete import _resolve_config_dir

    parsed = argparse.Namespace()
    parsed.config_dir = None
    with patch("tiz.autocomplete.Path.home", return_value=Path("/home/user")):
        result = _resolve_config_dir(parsed)
    assert result == Path("/home/user") / ".tiz"


def test_resolve_config_dir_path_subclass():
    """_resolve_config_dir handles a Path subclass without error."""
    from tiz.autocomplete import _resolve_config_dir

    # Python 3.11+ does not allow direct construction of Path subclasses.
    # Test that a duck-typed Path-like object is handled gracefully.
    class MyPath:
        def __str__(self):
            return "/some/path"

        def __fspath__(self):
            return "/some/path"

    parsed = argparse.Namespace()
    parsed.config_dir = MyPath()
    result = _resolve_config_dir(parsed)
    assert result == Path("/some/path")


def test_resolve_config_dir_exact_path_type():
    """_resolve_config_dir handles a plain Path instance gracefully."""
    from tiz.autocomplete import _resolve_config_dir

    parsed = argparse.Namespace()
    parsed.config_dir = Path("/custom/path")
    result = _resolve_config_dir(parsed)
    assert result == Path("/custom/path")


# ---------------------------------------------------------------------------
# _set_sandbox_completers tests
# ---------------------------------------------------------------------------


def _build_parser_with_sandbox_args() -> argparse.ArgumentParser:
    """Build a parser tree similar to cli.py's sb subcommand structure."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    sb_parser = subparsers.add_parser("sb")
    sb_subparsers = sb_parser.add_subparsers(dest="sb_command")

    list_parser = sb_subparsers.add_parser("list")
    list_parser.add_argument("sandbox_name", nargs="?", default=None)

    ls_parser = sb_subparsers.add_parser("ls")
    ls_parser.add_argument("sandbox_name", nargs="?", default=None)

    containers_parser = sb_subparsers.add_parser("containers")
    containers_parser.add_argument("sandbox_name", nargs="?", default=None)

    kill_parser = sb_subparsers.add_parser("kill")
    kill_parser.add_argument("sandbox_name", nargs="?", default=None)

    rm_parser = sb_subparsers.add_parser("rm")
    rm_parser.add_argument("sandbox_name")

    cleanup_parser = sb_subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--untracked-only", action="store_true")
    cleanup_parser.add_argument("--dead-only", action="store_true")

    sync_parser = sb_subparsers.add_parser("sync")
    sync_subparsers = sync_parser.add_subparsers(dest="sync_direction")
    from_parser = sync_subparsers.add_parser("from-origin")
    from_parser.add_argument("sandbox_name")
    to_parser = sync_subparsers.add_parser("to-origin")
    to_parser.add_argument("sandbox_name")

    logs_parser = sb_subparsers.add_parser("logs")
    logs_parser.add_argument("sandbox_name")

    build_parser = sb_subparsers.add_parser("build")
    build_subparsers = build_parser.add_subparsers(dest="build_command")
    image_parser = build_subparsers.add_parser("image")
    image_parser.add_argument("tag")
    build_subparsers.add_parser("all")

    # Add a top-level command with a --manifest argument like run/exec/chat
    manifest_parser = subparsers.add_parser("run")
    manifest_parser.add_argument("-m", "--manifest", action="append", default=[])

    return parser


def test_set_sandbox_completers_attaches_to_subparsers():
    """Completer is attached to all sandbox_name arguments in all subparsers."""
    parser = _build_parser_with_sandbox_args()
    _set_sandbox_completers(parser)

    sb_parser = parser._subparsers._group_actions[0].choices["sb"]  # type: ignore[attr-defined]
    for name in ("list", "ls", "containers", "kill", "rm", "logs"):
        sub = sb_parser._subparsers._group_actions[0].choices[name]  # type: ignore[attr-defined]
        for action in sub._actions:
            if action.dest == "sandbox_name":
                assert action.completer is _sandbox_name_completer

    # sync sub-subparsers also get completers
    sync_parser = sb_parser._subparsers._group_actions[0].choices["sync"]  # type: ignore[attr-defined]
    sync_sub = sync_parser._subparsers._group_actions[0]  # type: ignore[attr-defined]
    for name in ("from-origin", "to-origin"):
        sub = sync_sub.choices[name]
        for action in sub._actions:
            if action.dest == "sandbox_name":
                assert action.completer is _sandbox_name_completer


def test_set_sandbox_completers_does_not_leak():
    """Completer is not attached to non-sandbox_name/non-tag/non-manifest arguments."""
    parser = _build_parser_with_sandbox_args()
    _set_sandbox_completers(parser)

    sb_parser = parser._subparsers._group_actions[0].choices["sb"]  # type: ignore[attr-defined]
    choices: dict[str, argparse.ArgumentParser] = sb_parser._subparsers._group_actions[
        0
    ].choices  # type: ignore[attr-defined]

    # cleanup has no sandbox_name
    cleanup = choices["cleanup"]
    for action in cleanup._actions:
        completer = getattr(action, "completer", None)
        assert completer is not _sandbox_name_completer

    # build has no sandbox_name, but the image subparser has a tag completer
    build_parser = choices["build"]
    # Check image subcommand gets tag completer
    image_parser = build_parser._subparsers._group_actions[0].choices["image"]  # type: ignore[attr-defined]
    for action in image_parser._actions:
        if action.dest == "tag":
            assert action.completer is _image_tag_completer
            break
    else:
        pytest.fail("tag action not found")
    # Build top-level should not have sandbox_name completer
    for action in build_parser._actions:
        completer = getattr(action, "completer", None)
        assert completer is not _sandbox_name_completer


def test_set_sandbox_completers_attaches_manifest_completer():
    """Completer is attached to manifest arguments."""
    parser = _build_parser_with_sandbox_args()
    _set_sandbox_completers(parser)

    # run subcommand has --manifest
    run_parser = parser._subparsers._group_actions[0].choices["run"]  # type: ignore[attr-defined]
    for action in run_parser._actions:
        if action.dest == "manifest":
            assert action.completer is _manifest_path_completer
            break
    else:
        pytest.fail("manifest action not found")


def test_set_sandbox_completers_empty_parser():
    """Calling _set_sandbox_completers on a bare parser does not error."""
    parser = argparse.ArgumentParser()
    _set_sandbox_completers(parser)


def test_set_sandbox_completers_no_sandbox_name_args():
    """Parser with no sandbox_name arguments is unchanged."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("bar")
    _set_sandbox_completers(parser)
    # No error means success


def test_set_sandbox_completers_non_argparse_choices():
    """Completer handles subparser choices that are not ArgumentParser."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    # add_parser always returns ArgumentParser, so manually inject a non-ArgumentParser
    subparsers.choices["not_a_parser"] = "string_value"  # type: ignore[assignment]

    # This should not raise because we guard with isinstance check
    _set_sandbox_completers(parser)


# ---------------------------------------------------------------------------
# autocomplete tests
# ---------------------------------------------------------------------------


def test_autocomplete_no_argcomplete_env_var():
    """When _ARGCOMPLETE is not set, autocomplete returns immediately."""
    parser = MagicMock()
    autocomplete(parser)
    parser.parse_args.assert_not_called()


def test_autocomplete_with_argcomplete_env_var():
    """When _ARGCOMPLETE is set, argcomplete.autocomplete is called."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo")

    with (
        patch("tiz.autocomplete.os.environ", {"_ARGCOMPLETE": "1"}),
        patch("tiz.autocomplete.argcomplete.autocomplete") as mock_ac,
        patch("tiz.autocomplete._set_sandbox_completers") as mock_set,
    ):
        autocomplete(parser)

    mock_set.assert_called_once_with(parser)
    mock_ac.assert_called_once_with(parser)


def test_autocomplete_no_argcomplete_import():
    """When argcomplete is not available, autocomplete returns immediately."""
    parser = MagicMock()
    with patch("tiz.autocomplete._HAS_ARGCOMPLETE", False):
        autocomplete(parser)
    parser.parse_args.assert_not_called()


def test_autocomplete_returns_none():
    """autocomplete returns None."""
    parser = MagicMock()
    result = autocomplete(parser)
    assert result is None


def test_autocomplete_sets_sandbox_completers():
    """autocomplete calls _set_sandbox_completers before argcomplete."""
    parser = MagicMock(spec=argparse.ArgumentParser)

    with (
        patch("tiz.autocomplete.os.environ", {"_ARGCOMPLETE": "1"}),
        patch("tiz.autocomplete.argcomplete.autocomplete") as mock_ac,
        patch("tiz.autocomplete._set_sandbox_completers") as mock_set,
    ):
        autocomplete(parser)

    mock_set.assert_called_once_with(parser)
    mock_ac.assert_called_once_with(parser)


def test_autocomplete_with_argcomplete_env_but_no_import():
    """autocomplete returns early when _ARGCOMPLETE set but argcomplete not installed."""
    parser = MagicMock()
    with (
        patch("tiz.autocomplete._HAS_ARGCOMPLETE", False),
        patch("tiz.autocomplete.os.environ", {"_ARGCOMPLETE": "1"}),
    ):
        autocomplete(parser)
    parser.parse_args.assert_not_called()


# ---------------------------------------------------------------------------
# shellcode tests
# ---------------------------------------------------------------------------


def test_shellcode_bash():
    """shellcode returns bash snippet."""
    code = shellcode("bash")
    assert isinstance(code, str)
    assert len(code) > 0


def test_shellcode_zsh():
    """shellcode returns zsh snippet."""
    code = shellcode("zsh")
    assert isinstance(code, str)
    assert len(code) > 0


def test_shellcode_fish():
    """shellcode returns fish snippet."""
    code = shellcode("fish")
    assert isinstance(code, str)
    assert len(code) > 0


def test_shellcode_tcsh():
    """shellcode returns tcsh snippet."""
    code = shellcode("tcsh")
    assert isinstance(code, str)
    assert len(code) > 0


def test_shellcode_default_bash():
    """shellcode without argument defaults to bash."""
    code = shellcode()
    assert isinstance(code, str)
    assert len(code) > 0


def test_shellcode_no_argcomplete():
    """When argcomplete not available, returns placeholder."""
    with patch("tiz.autocomplete._HAS_ARGCOMPLETE", False):
        result = shellcode("bash")
    assert result == "# argcomplete not installed"


def test_shellcode_contains_expected():
    """shellcode output should contain argcomplete shell integration markers."""
    code = shellcode("bash")
    assert "_ARGCOMPLETE" in code or "argcomplete" in code.lower()
    # Should contain the completion registration mechanism
    assert "complete" in code or "compdef" in code


# ---------------------------------------------------------------------------
# Integration test: autocomplete is called in cli.main()
# ---------------------------------------------------------------------------


def test_cli_main_invokes_autocomplete():
    """Verify cli.main() calls autocomplete before parse_args."""
    from tiz import cli

    with patch.object(cli, "autocomplete") as mock_autocomplete:
        with patch.object(cli.sys, "argv", ["tiz", "unknown-cmd"]):
            with patch.object(cli.sys.stdout, "isatty", return_value=False):
                with pytest.raises(SystemExit):
                    cli.main()
        mock_autocomplete.assert_called_once()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_autocomplete_os_environ_unchanged():
    """autocomplete should not modify os.environ."""
    env_before = dict(os.environ)
    parser = MagicMock()
    autocomplete(parser)
    assert dict(os.environ) == env_before


def test_autocomplete_with_fallback_on_import_error():
    """Check that fallback works correctly if argcomplete is missing."""
    with patch("tiz.autocomplete._HAS_ARGCOMPLETE", False):
        autocomplete(MagicMock())
        result = shellcode()
        assert result == "# argcomplete not installed"


# ---------------------------------------------------------------------------
# Integration: cli.main() sandbox_name completers on sb subcommands
# ---------------------------------------------------------------------------


def test_cli_main_sets_sandbox_completers_on_sb_parser():
    """Ensure the real cli _build_sb_parser gets sandbox completers via autocomplete."""
    from tiz import cli

    with (
        patch.object(cli.sys, "argv", ["tiz", "unknown-cmd"]),
        patch.object(cli.sys.stdout, "isatty", return_value=False),
        patch("tiz.autocomplete._set_sandbox_completers") as mock_set,
        patch("tiz.autocomplete.argcomplete.autocomplete"),
        patch("tiz.autocomplete.os.environ", {"_ARGCOMPLETE": "1"}),
    ):
        with pytest.raises(SystemExit):
            cli.main()

    # _set_sandbox_completers was called on the main parser
    mock_set.assert_called_once()
    parser_arg = mock_set.call_args[0][0]
    assert isinstance(parser_arg, argparse.ArgumentParser)
