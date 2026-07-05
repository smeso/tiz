"""Tests for centralized logger."""

import importlib
import logging
import sys
from collections.abc import Iterator
from unittest.mock import patch

import pytest

import tiz.log
from tiz.log import get_logger, logging_init


@pytest.fixture(autouse=True)
def _reset_logger_state() -> Iterator[None]:
    root = logging.getLogger()
    root_handlers_before = list(root.handlers)
    root_level_before = root.level
    tiz_logger = logging.getLogger("tiz")
    tiz_handlers_before = list(tiz_logger.handlers)
    tiz_level_before = tiz_logger.level
    yield
    root.handlers = root_handlers_before
    root.setLevel(root_level_before)
    tiz_logger.handlers = tiz_handlers_before
    tiz_logger.setLevel(tiz_level_before)


class TestLog:
    def test_get_logger_returns_logger(self) -> None:
        logger = get_logger("tiz.test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "tiz.test_module"

    def test_get_logger_is_child_of_tiz(self) -> None:
        logger = get_logger("tiz.test_module")
        assert logger.name.startswith("tiz")
        parent = logging.getLogger("tiz")
        assert logger.parent is parent

    def test_get_logger_called_twice_returns_same_instance(self) -> None:
        logger1 = get_logger("tiz.test_module")
        logger2 = get_logger("tiz.test_module")
        assert logger1 is logger2

    def test_get_logger_prefixes_non_tiz_name(self) -> None:
        logger = get_logger("myapp")
        assert logger.name == "tiz.myapp"

    def test_get_logger_prefixes_bare_tiz(self) -> None:
        logger = get_logger("tiz")
        assert logger.name == "tiz.tiz"

    def test_get_logger_does_not_double_prefix(self) -> None:
        logger = get_logger("tiz.core")
        assert logger.name == "tiz.core"

    def test_init_adds_stream_handler_to_root(self) -> None:
        logging_init()
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        handler = root.handlers[-1]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stderr

    def test_init_called_twice_no_duplicate_handlers(self) -> None:
        root = logging.getLogger()
        logging_init()
        logging_init()
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1

    @staticmethod
    def _format_record(fmt: logging.Formatter) -> str:
        record = logging.LogRecord(
            "tiz.test", logging.INFO, "test.py", 1, "test msg", (), None
        )
        return fmt.format(record)

    def test_tty_format_has_no_date(self) -> None:
        with patch("sys.stderr.isatty", return_value=True):
            logging_init()
            root = logging.getLogger()
            handler = root.handlers[-1]
            fmt = handler.formatter
            assert fmt is not None
            formatted = self._format_record(fmt)
            assert formatted.startswith("[")
            assert "levelname" in formatted or "INFO" in formatted

    def test_non_tty_format_has_date(self) -> None:
        with patch("sys.stderr.isatty", return_value=False):
            logging_init()
            root = logging.getLogger()
            handler = root.handlers[-1]
            fmt = handler.formatter
            assert fmt is not None
            formatted = self._format_record(fmt)
            assert formatted[0].isdigit()

    def test_init_sets_level_on_root_logger(self) -> None:
        logging_init(level=logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG

    def test_logger_propagates_to_root_handler(self) -> None:
        with patch("sys.stderr.isatty", return_value=True):
            logging_init(level=logging.DEBUG)
            caplog_records: list[logging.LogRecord] = []

            class _Handler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    caplog_records.append(record)

            h = _Handler()
            root = logging.getLogger()
            root.addHandler(h)
            logger = get_logger("tiz.logtest")
            logger.info("test message from logger")
            assert len(caplog_records) == 1
            assert caplog_records[0].name == "tiz.logtest"
            assert caplog_records[0].message == "test message from logger"

    def test_default_level_is_warning(self) -> None:
        logging_init()
        assert logging.getLogger().level == logging.WARNING

    def test_child_logger_inherits_tiz_level(self) -> None:
        logging_init(level=logging.INFO)
        child = get_logger("tiz.sub")
        assert child.level == 0
        assert child.getEffectiveLevel() == logging.INFO

    def test_package_logger_has_null_handler(self) -> None:
        pkg_logger = logging.getLogger("tiz")
        assert any(isinstance(h, logging.NullHandler) for h in pkg_logger.handlers)

    def test_no_handler_warnings_for_early_logger(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = get_logger("tiz.early")
        logger.info("no warnings expected")
        assert len(caplog.records) == 0

    def test_init_handles_closed_stderr(self) -> None:
        with patch("sys.stderr.isatty", side_effect=ValueError("closed file")):
            logging_init(level=logging.INFO)
        root = logging.getLogger()
        assert root.level == logging.INFO
        handler = root.handlers[-1]
        fmt = handler.formatter
        assert fmt is not None
        formatted = self._format_record(fmt)
        assert formatted[0].isdigit()

    def test_is_stderr_a_tty_returns_false_when_stderr_is_none(self) -> None:
        with patch.object(sys, "stderr", None):
            from tiz.log import _is_stderr_a_tty

            assert _is_stderr_a_tty() is False

    def test_non_tty_format_includes_timezone(self) -> None:
        with patch("sys.stderr.isatty", return_value=False):
            logging_init()
            root = logging.getLogger()
            handler = root.handlers[-1]
            fmt = handler.formatter
            assert fmt is not None
            assert fmt.datefmt is not None
            assert "%z" in fmt.datefmt

    def test_reload_does_not_add_duplicate_null_handler(self) -> None:
        pkg_logger = logging.getLogger("tiz")
        null_handlers_before = [
            h for h in pkg_logger.handlers if isinstance(h, logging.NullHandler)
        ]

        importlib.reload(tiz.log)

        null_handlers_after = [
            h for h in pkg_logger.handlers if isinstance(h, logging.NullHandler)
        ]
        assert len(null_handlers_after) == len(null_handlers_before)
