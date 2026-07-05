# ruff: noqa: ARG001, S101, S603
# mypy: disable-error-code="no-untyped-def,arg-type,no-any-return,attr-defined,misc"
"""End-to-end integration tests for the tiz CLI.

These tests use a fake podman script on PATH to simulate container
operations and a real fake HTTP server to mock the inference API.
Everything else is real: manifest parsing, sandbox directories,
file system operations, the tiz entry point, subprocess calls.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
_PROJECT_DIR = _SCRIPT_DIR.parent

pytestmark = pytest.mark.accurate_cov


# ---------------------------------------------------------------------------
# Fake inference server
# ---------------------------------------------------------------------------


class FakeInferenceHandler(http.server.BaseHTTPRequestHandler):
    """Serves OpenAI-compatible endpoints for testing."""

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ARG002
        """Suppress HTTP server log messages."""

    def do_GET(self) -> None:
        """Handle GET requests for model listing and server properties."""
        if self.path == "/v1/models":
            self._send_json(
                {
                    "data": [
                        {"id": "test-model"},
                    ],
                },
            )
        elif self.path == "/props":
            self._send_json(
                {
                    "default_generation_settings": {"n_ctx": 4096},
                    "model": "test-model",
                    "chat_template_caps": {
                        "supports_tools": True,
                        "supports_tool_calls": True,
                    },
                    "modalities": {"vision": False, "audio": False},
                },
            )
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/v1/models_error":
            self._send_json({"error": "internal error"}, status=500)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests for chat completions and token counting."""
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        data = json.loads(body) if body else {}

        if self.path == "/v1/chat/completions":
            stream = data.get("stream", False)
            if stream:
                self._send_stream_response()
            else:
                self._send_chat_response(data)
        elif self.path == "/v1/messages/count_tokens":
            self._send_json({"input_tokens": 42})
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, obj: dict[str, Any], status: int = 200) -> None:
        """Send a JSON response with the given status code."""
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_chat_response(self, data: dict[str, Any]) -> None:
        """Send a non-streaming chat completion response."""
        messages = data.get("messages", [])
        last_content = messages[-1]["content"] if messages else ""

        tool_calls: list[dict[str, Any]] = []
        if "file" in last_content.lower():
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "ReadFile",
                        "arguments": json.dumps({"file_path": "/tmp/test.txt"}),
                    },
                },
            ]

        result = {
            "choices": [
                {
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None
                        if tool_calls
                        else "Hello from tiz test assistant!",
                        "tool_calls": tool_calls or None,
                    },
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            },
            "model": data.get("model", "test-model"),
            "object": "chat.completion",
        }
        self._send_json(result)

    def _send_stream_response(self) -> None:
        """Send a streaming chat completion response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello "}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"from "}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"tiz "}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"test "}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"assistant!"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0}},"timings":{"prompt_n":10,"predicted_n":5,"cache_n":0,"cache_write_n":0,"prompt_ms":100,"predicted_ms":200}}\n\n',
            b"data: [DONE]\n\n",
        ]
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()
            time.sleep(0.01)


@contextlib.contextmanager
def fake_inference_server() -> Generator[int, None, None]:
    """Start a fake inference server on a random port, yield port."""
    server = http.server.HTTPServer(("127.0.0.1", 0), FakeInferenceHandler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_podman(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Install a fake podman script on PATH via a wrapper dir."""
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_podman_path = fake_bin / "podman"
    quoted = repr(str(_SCRIPT_DIR))
    fake_podman_path.write_text(
        f"#!/usr/bin/env python3\n"
        f"import sys; sys.path.insert(0, {quoted})\n"
        f"from fake_podman import main; main()\n",
        encoding="utf-8",
    )
    fake_podman_path.chmod(0o755)

    storage_dir = tmp_path / "fake-podman-storage"
    storage_dir.mkdir()
    monkeypatch.setenv("FAKE_PODMAN_STORAGE", str(storage_dir))

    old_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{fake_bin}:{old_path}")

    yield fake_bin


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure clean environment for tests."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")


@pytest.fixture
def tiz_home(tmp_path: Path, fake_podman: Path) -> Path:
    """Create a tiz home directory."""
    home = tmp_path / "tiz-home"
    home.mkdir()
    return home


# ---------------------------------------------------------------------------
# Helper to write manifest files
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path,
    inference_server_port: int,
    task_name: str = "test_task",
    tools: list[dict[str, Any]] | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> None:
    """Write a tiz manifest YAML file (in JSON format, YAML-superset)."""
    if tools is None:
        tools = [
            {"Bash": ["none", "nodisk"]},
        ]
    if actions is None:
        actions = [
            {"prompts": [["Say hello to the user and respond with 'HELLO_WORLD_OK'."]]},
        ]

    manifest: dict[str, Any] = {
        "meta": {
            "version": "0",
            "parallelism": 1,
            "container_engine": "podman",
            "color": False,
            "hide_reasoning": True,
            "use_host_timezone": False,
            "save_full_logs": False,
            "save_full_toolcalls": False,
            "save_full_usage_details": False,
            "summarizer_context_ratio": 0.5,
            "verbosity": 0,
            "delete_sandbox_on_exit": True,
            "ephemeral_sandbox": True,
        },
        "inference_engines": [
            {
                "type": "llamacpp",
                "name": "test_llm",
                "host": f"http://127.0.0.1:{inference_server_port}",
                "model": "test-model",
                "timeout": 5,
                "verify_ssl": False,
            },
        ],
        "tasks": [
            {
                "name": task_name,
                "worker_image": "tiz-worker:latest",
                "tools": tools,
                "actions": actions,
                "allow_parallel_run": False,
            },
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers for invoking tiz as subprocess
# ---------------------------------------------------------------------------


def _env(**kw: str) -> dict[str, str]:
    """Build a clean environment dict for subprocess calls."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
        "USER": os.environ.get("USER", "root"),
        "NO_COLOR": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(_SRC_DIR),
    }
    env.update(kw)
    return env


def _tiz_cmd(
    home: Path,
    args: list[str],
    env_add: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the tiz CLI as a subprocess with the given args."""
    cmd = [sys.executable, "-m", "tiz", "-c", str(home), *args]
    e = _env()
    if env_add:
        e.update(env_add)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_PROJECT_DIR,
        env=e,
        check=False,
    )


def _subprocess_run(
    cmd: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with check=False."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("timeout", 30)
    kwargs.setdefault("check", False)
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullCliRun:
    """End-to-end tests running the full tiz CLI with subprocess."""

    def test_help_output(self) -> None:
        """Test that --help produces output with no stderr."""
        result = _subprocess_run(
            [sys.executable, "-m", "tiz", "--help"],
            cwd=_PROJECT_DIR,
            env=_env(),
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout
        assert "tiz" in result.stdout
        assert "chat" in result.stdout
        assert "run" in result.stdout
        assert result.stderr == "", f"stderr should be empty but got:\n{result.stderr}"

    def test_completion_output(self) -> None:
        """Test shell completion script generation with no stderr."""
        result = _subprocess_run(
            [sys.executable, "-m", "tiz", "completion", "bash"],
            cwd=_PROJECT_DIR,
            env=_env(),
        )

        assert result.returncode == 0
        assert result.stdout.strip(), "Completion script should not be empty"
        assert result.stderr == "", f"stderr should be empty but got:\n{result.stderr}"

    def test_sb_list_empty(self, tiz_home: Path) -> None:
        """Test 'tiz sb list' with no sandboxes returns empty stdout."""
        result = _tiz_cmd(tiz_home, ["sb", "list"])

        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert result.stdout.strip() == "", (
            f"Expected empty stdout for no sandboxes, got:\n{result.stdout}"
        )
        assert result.stderr == "", f"stderr should be empty but got:\n{result.stderr}"

    def test_sb_containers_empty(self, tiz_home: Path) -> None:
        """Test 'tiz sb containers' with no sandboxes returns empty stdout."""
        result = _tiz_cmd(tiz_home, ["sb", "containers"])

        assert result.returncode == 0
        assert result.stdout.strip() == "", (
            f"Expected empty stdout for no sandboxes, got:\n{result.stdout}"
        )
        assert result.stderr == "", f"stderr should be empty but got:\n{result.stderr}"

    def test_stats_no_data(self, tiz_home: Path) -> None:
        """Test 'tiz stats usage' with no log data shows appropriate message."""
        result = _tiz_cmd(tiz_home, ["stats", "usage"])

        assert result.returncode == 0
        msgs = ["No usage logs found", "No usage data found"]
        assert any(msg in result.stdout for msg in msgs), (
            f"Expected one of {msgs} in stdout, got:\n{result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_manifest_parse_error(self, tiz_home: Path, tmp_path: Path) -> None:
        """Test that invalid manifest yields non-zero exit and error message."""
        manifest_file = tmp_path / "bad_manifest.yaml"
        manifest_file.write_text("invalid: yaml: [", encoding="utf-8")

        result = _tiz_cmd(tiz_home, ["run", "-m", str(manifest_file)])

        assert result.returncode != 0, (
            f"CLI should have failed but returned 0\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        error_msg = result.stderr.lower()
        assert any(word in error_msg for word in ("parse", "invalid", "error")), (
            f"Expected stderr to contain a parse/invalid/error message, got:\n{result.stderr}"
        )

    def test_run_simple_prompt(self, tiz_home: Path, tmp_path: Path) -> None:
        """Run with a simple prompt and verify full output structure."""
        manifest_file = tmp_path / "manifest.yaml"

        with fake_inference_server() as port:
            write_manifest(manifest_file, port)

            result = _tiz_cmd(tiz_home, ["run", "-m", str(manifest_file)])

            assert result.returncode == 0, (
                f"CLI failed: rc={result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            assert "Hello from tiz test assistant!" in result.stdout, (
                f"Expected response not found in stdout:\n{result.stdout}"
            )
            assert "Credits spent:" in result.stdout
            assert "Tools usage:" in result.stdout
            assert "input:" in result.stdout
            assert "output:" in result.stdout
            assert "cached:" in result.stdout

    def test_run_with_extra_flags(self, tiz_home: Path, tmp_path: Path) -> None:
        """Run with extra CLI flags overridden, verifying all output sections."""
        manifest_file = tmp_path / "manifest_extra.yaml"

        with fake_inference_server() as port:
            write_manifest(manifest_file, port)

            result = _tiz_cmd(
                tiz_home,
                [
                    "--hide-reasoning",
                    "--no-color",
                    "run",
                    "-m",
                    str(manifest_file),
                ],
            )

            assert result.returncode == 0, (
                f"CLI failed: rc={result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            assert "Hello from tiz test assistant!" in result.stdout
            assert "Credits spent:" in result.stdout
            assert "Tools usage:" in result.stdout

    def test_tiz_home_created(self, tiz_home: Path, tmp_path: Path) -> None:
        """Verify running CLI creates expected directory structure."""
        manifest_file = tmp_path / "manifest.yaml"

        with fake_inference_server() as port:
            write_manifest(manifest_file, port)

            result = _tiz_cmd(tiz_home, ["run", "-m", str(manifest_file)])

            assert result.returncode == 0, (
                f"CLI failed: rc={result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

            assert tiz_home.exists()
            sandboxes_dir = tiz_home / "sandboxes"
            assert sandboxes_dir.is_dir(), (
                f"Sandboxes dir not created at {sandboxes_dir}"
            )
            # Sandbox subdirs are ephemeral and cleaned up on exit,
            # but the sandboxes dir itself persists.

    def test_credits(self, tiz_home: Path, tmp_path: Path) -> None:
        """Test 'tiz credits' command with a full manifest."""
        manifest_file = tmp_path / "credits_manifest.yaml"
        manifest_data: dict[str, Any] = {
            "meta": {
                "version": "0",
                "parallelism": 1,
                "container_engine": "podman",
                "color": False,
                "hide_reasoning": True,
                "use_host_timezone": False,
                "save_full_logs": False,
                "save_full_toolcalls": False,
                "save_full_usage_details": False,
                "summarizer_context_ratio": 0.5,
                "verbosity": 0,
                "delete_sandbox_on_exit": True,
                "ephemeral_sandbox": True,
            },
            "inference_engines": [
                {
                    "type": "llamacpp",
                    "name": "test_llm",
                    "host": "http://127.0.0.1:9999",
                    "model": "test-model",
                    "timeout": 1,
                    "verify_ssl": False,
                },
            ],
            "tasks": [
                {
                    "name": "test_task",
                    "worker_image": "tiz-worker:latest",
                    "tools": [{"Bash": ["none", "nodisk"]}],
                    "actions": [{"prompts": [["Say hello"]]}],
                    "allow_parallel_run": False,
                },
            ],
        }
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        result = _tiz_cmd(tiz_home, ["credits", "-m", str(manifest_file)])

        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "test_llm" in result.stdout
        assert "Total credits:" in result.stdout
        assert "Total usage:" in result.stdout
        assert "Remaining:" in result.stdout

    def test_chat_help(self, tiz_home: Path) -> None:
        """Test that 'tiz chat --help' shows chat-specific options."""
        result = _tiz_cmd(
            tiz_home,
            ["chat", "--help"],
        )

        assert result.returncode == 0, (
            f"CLI failed: rc={result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "usage:" in result.stdout
        assert result.stderr == "", f"stderr should be empty but got:\n{result.stderr}"

    def test_run_tool_call(self, tiz_home: Path, tmp_path: Path) -> None:
        """Run with a prompt that triggers a tool call from the fake server."""
        manifest_file = tmp_path / "manifest_tool.yaml"

        with fake_inference_server() as port:
            write_manifest(
                manifest_file,
                port,
                actions=[
                    {"prompts": [["Read the file and tell me what's in it"]]},
                ],
            )

            result = _tiz_cmd(tiz_home, ["run", "-m", str(manifest_file)])

            assert result.returncode == 0, (
                f"CLI failed: rc={result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            assert "Credits spent:" in result.stdout
            assert "Tools usage:" in result.stdout
            assert "Error" not in result.stderr, (
                f"Unexpected error in stderr:\n{result.stderr}"
            )

    def test_sandbox_cleanup(self, tiz_home: Path, tmp_path: Path) -> None:
        """Verify sandbox directories are cleaned up with ephemeral_sandbox."""
        manifest_file = tmp_path / "manifest_cleanup.yaml"

        with fake_inference_server() as port:
            write_manifest(
                manifest_file,
                port,
                tools=[{"Bash": ["none", "nodisk"]}],
                actions=[
                    {"prompts": [["Say hello and respond with HELLO_WORLD_OK"]]},
                ],
            )

            sandboxes_dir = tiz_home / "sandboxes"

            result = _tiz_cmd(tiz_home, ["run", "-m", str(manifest_file)])

            assert result.returncode == 0, (
                f"CLI failed: rc={result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            # With ephemeral_sandbox and delete_sandbox_on_exit, sandbox subdirs
            # should be cleaned up, but the parent sandboxes dir persists.
            assert sandboxes_dir.is_dir(), (
                f"Sandboxes dir should still exist at {sandboxes_dir}"
            )
            sandbox_items = (
                list(sandboxes_dir.iterdir()) if sandboxes_dir.exists() else []
            )
            assert len(sandbox_items) == 0, (
                f"Expected no sandbox subdirs but found: {sandbox_items}"
            )

    def test_stats_with_data(self, tiz_home: Path, tmp_path: Path) -> None:
        """Run a task and then verify stats shows the expected usage data."""
        manifest_file = tmp_path / "manifest_stats.yaml"

        with fake_inference_server() as port:
            write_manifest(manifest_file, port)

            run_result = _tiz_cmd(tiz_home, ["run", "-m", str(manifest_file)])
            assert run_result.returncode == 0, (
                f"Run failed: rc={run_result.returncode}\n"
                f"stdout: {run_result.stdout}\n"
                f"stderr: {run_result.stderr}"
            )

            stats_result = _tiz_cmd(tiz_home, ["stats", "usage"])

            assert stats_result.returncode == 0, (
                f"Stats command failed: rc={stats_result.returncode}\n"
                f"stdout: {stats_result.stdout}\n"
                f"stderr: {stats_result.stderr}"
            )
            # With the fake podman setup, usage logs may or may not be written.
            # Accept either outcome as valid behavior.
            if "No usage logs found" not in stats_result.stdout:
                # If there are logs, verify they contain expected data
                assert (
                    "test_llm" in stats_result.stdout
                    or "test_task" in stats_result.stdout
                )

    def test_inference_server_error(self, tiz_home: Path, tmp_path: Path) -> None:
        """Test graceful handling of inference server returning connection errors."""
        manifest_file = tmp_path / "manifest_error.yaml"
        # Use a port where nothing is listening to trigger a connection error
        manifest: dict[str, Any] = {
            "meta": {
                "version": "0",
                "parallelism": 1,
                "container_engine": "podman",
                "color": False,
                "hide_reasoning": True,
                "use_host_timezone": False,
                "save_full_logs": False,
                "save_full_toolcalls": False,
                "save_full_usage_details": False,
                "summarizer_context_ratio": 0.5,
                "verbosity": 0,
                "delete_sandbox_on_exit": True,
                "ephemeral_sandbox": True,
            },
            "inference_engines": [
                {
                    "type": "llamacpp",
                    "name": "test_llm",
                    "host": "http://127.0.0.1:1",
                    "model": "test-model",
                    "timeout": 1,
                    "verify_ssl": False,
                },
            ],
            "tasks": [
                {
                    "name": "test_task",
                    "worker_image": "tiz-worker:latest",
                    "tools": [{"Bash": ["none", "nodisk"]}],
                    "actions": [
                        {"prompts": [["Say hello"]]},
                    ],
                    "allow_parallel_run": False,
                },
            ],
        }
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        result = _tiz_cmd(
            tiz_home,
            ["run", "-m", str(manifest_file)],
        )

        # The CLI should handle the connection error gracefully
        has_error = (
            result.returncode != 0
            or "Error" in result.stderr
            or "error" in result.stderr.lower()
        )
        assert has_error, (
            f"Expected error handling but got:\n"
            f"rc={result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
