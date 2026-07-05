"""Centralized logging utility for tiz."""

import logging
import sys

_logger = logging.getLogger("tiz")
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())


def _is_stderr_a_tty() -> bool:
    """Check if stderr is a TTY, safely handling closed or None stderr."""
    try:
        return sys.stderr.isatty()
    except (ValueError, AttributeError):
        return False


def logging_init(level: int = logging.WARNING) -> None:
    if _is_stderr_a_tty():
        logging.basicConfig(
            level=level,
            format="[%(levelname)s] %(name)s: %(message)s",
            force=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %z",
            force=True,
        )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name, ensuring it is under tiz."""
    if not name.startswith("tiz."):
        name = f"tiz.{name}"
    return logging.getLogger(name)
