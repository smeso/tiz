"""tiz - Agentic chatbot using sandboxed tools."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tiz") or "0.1.0"
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

__all__: list[str] = []
