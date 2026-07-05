"""Tests for the tiz __init__ module."""

import importlib
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import packaging.version

import tiz


def test_version_is_string() -> None:
    assert isinstance(tiz.__version__, str)
    assert len(tiz.__version__) > 0


def test_version_is_semver() -> None:
    v = packaging.version.Version(tiz.__version__)
    assert v.major >= 0
    assert v.minor >= 0
    assert v.micro >= 0


def test_all_is_list() -> None:
    assert isinstance(tiz.__all__, list)


def test_all_is_empty() -> None:
    assert tiz.__all__ == []


def test_py_typed_exists() -> None:
    py_typed = Path(tiz.__file__).parent / "py.typed"
    assert py_typed.exists()
    assert py_typed.is_file()


def test_version_fallback_on_empty_string() -> None:
    original_version = tiz.__version__
    with patch("importlib.metadata.version", return_value="") as mock_version:
        importlib.reload(tiz)
        assert tiz.__version__ == "0.1.0"
        mock_version.assert_called_once_with("tiz")
    tiz.__version__ = original_version


def test_version_fallback_on_package_not_found() -> None:
    original_version = tiz.__version__
    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError,
    ) as mock_version:
        importlib.reload(tiz)
        assert tiz.__version__ == "0.1.0"
        mock_version.assert_called_once_with("tiz")
    tiz.__version__ = original_version


def test_version_happy_path() -> None:
    original_version = tiz.__version__
    with patch("importlib.metadata.version", return_value="1.2.3") as mock_version:
        importlib.reload(tiz)
        assert tiz.__version__ == "1.2.3"
        mock_version.assert_called_once_with("tiz")
    tiz.__version__ = original_version
