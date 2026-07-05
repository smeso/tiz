"""Allow running as python -m tiz."""

from __future__ import annotations

import sys

from tiz.cli import main


def _main_entry() -> None:
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _main_entry()
