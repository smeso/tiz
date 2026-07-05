# AGENTS.md

tiz - agentic chatbot using sandboxed tools.

Before committing: ruff check, mypy, ruff format. Fix all warnings.

Tests: pytest. Be thorough. Check every return value, every edge case.

Finally commit your changes with a very brief but descriptive message.
Also explain the why in the commit message not just the what, but keep it very short.

src/tiz/ - main package
tests/ - test suite
src/tiz/data/web_static/ - web frontend (validate with scripts/web_checks.sh)
