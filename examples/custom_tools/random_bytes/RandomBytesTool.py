"""Sandbox-side handler for RandomBytesTool.

Dynamically loaded by sandbox_worker.py's _load_custom_tools()
from /opt/tiz_tools/. Must expose a handle() function.
"""

import base64
import os


def handle(params: dict) -> tuple[str, bool]:
    """Read random bytes from /dev/urandom and return them base64-encoded."""
    count = params.get("count", 32)
    if not isinstance(count, int) or count < 1 or count > 8192:
        return "ERROR: count must be an integer between 1 and 8192", True
    data = os.urandom(count)
    encoded = base64.b64encode(data).decode("ascii")
    return encoded, False
