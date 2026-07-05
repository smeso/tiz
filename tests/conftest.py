"""Shared fixtures for tests."""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_drop_reporting_env() -> Generator[None, None, None]:
    """Clear TIZ_DROP_OPENROUTER_REPORTING before each test by default."""
    backup = os.environ.get("TIZ_DROP_OPENROUTER_REPORTING")
    if "TIZ_DROP_OPENROUTER_REPORTING" in os.environ:
        del os.environ["TIZ_DROP_OPENROUTER_REPORTING"]
    yield
    if backup is not None:
        os.environ["TIZ_DROP_OPENROUTER_REPORTING"] = backup
    elif "TIZ_DROP_OPENROUTER_REPORTING" in os.environ:
        del os.environ["TIZ_DROP_OPENROUTER_REPORTING"]


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    """Provide a temporary file with content."""
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")
    return f


@pytest.fixture
def socket_path(tmp_path: Path) -> str:
    """Provide a temporary Unix socket path."""
    return str(tmp_path / "test.sock")


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--accurate-cov",
        action="store_true",
        default=False,
        help="Improve coverage accuracy skipping some tests",
    )


def pytest_runtest_setup(item: Any) -> None:
    if "accurate_cov" in item.keywords and item.config.getoption("--accurate-cov"):
        pytest.skip("coverage accuracy")


@pytest.fixture
def mock_pyaudio():
    """Mock pyaudio for recording tests without a real audio device."""
    mock_p = MagicMock()
    mock_stream = MagicMock()
    mock_stream.read.return_value = (
        b"\x00\x00" * 1024
    )  # 2048 bytes = CHUNK * sample_width * channels (1024 * 2 * 1)
    mock_p.open.return_value = mock_stream
    mock_p.get_sample_size.return_value = 2

    # Create a fake pyaudio module for sys.modules
    mock_pyaudio_mod = MagicMock()
    mock_pyaudio_mod.PyAudio.return_value = mock_p
    mock_pyaudio_mod.paInt16 = 2  # PortAudio constant

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio_mod}):
        yield mock_p, mock_stream
