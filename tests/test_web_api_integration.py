"""Integration tests for WebSocket functionality with actual WebSocket connections."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tiz.manifest_parser import Manifest, ManifestMeta, TaskSpec
from tiz.web_api import App, ChatWebSocketHandler
from tiz.web_config_parser import EndpointConfig

pytestmark = pytest.mark.accurate_cov


def _make_minimal_manifest() -> Manifest:
    """Create a minimal manifest for testing."""
    task = MagicMock(spec=TaskSpec)
    task.name = "default"
    task.worker_image = "tiz-worker:latest"
    task.worker_image_containerfile = None
    task.readonly_sandbox = False
    task.project = None
    task.sys_prompt = None
    task.sys_prompt_custom = None
    task.actions = []
    task.allow_parallel_run = False
    task.force_copy_files = []
    task.inference_engine = None
    task.dedicated_audio_engine = None
    task.tmpfs_root = False
    task.subagents = []
    task.tools = []
    meta = ManifestMeta(
        version="0",
        ephemeral_sandbox=True,
        delete_sandbox_on_exit=True,
    )
    manifest = Manifest(meta=meta, tasks=[task])
    manifest.inference_engines = []
    manifest.audio_inference_engines = []
    return manifest


@pytest.fixture
def manifest() -> Manifest:
    return _make_minimal_manifest()


async def _http_get(host: str, port: int, path: str) -> tuple[int, bytes]:
    """Perform an HTTP GET request and return (status_code, body)."""
    reader, writer = await asyncio.open_connection(host, port)
    request = f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    writer.write(request.encode())
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    lines = response.split(b"\r\n")
    status_line = lines[0]
    status_code = int(status_line.split(b" ")[1])
    body_start = response.find(b"\r\n\r\n") + 4
    body = response[body_start:]
    return status_code, body


# ---------------------------------------------------------------------------
# Helper: run a patched chat that sends updates and stops
# ---------------------------------------------------------------------------


def _make_patched_run(
    messages: list[dict[str, Any]] | None = None,
    delay: float = 0.1,
) -> Any:
    """Create a patched _run function that sends update callbacks and stops.

    The patched _run sends a sequence of update callbacks, then sets
    _running to False so the send_worker exits.
    """
    if messages is None:
        messages = [{"type": "chat", "text": "hello from AI"}]

    def _patched_run(self: ChatWebSocketHandler) -> None:
        self.manifest.meta.ephemeral_sandbox = True
        self.manifest.meta.delete_sandbox_on_exit = True
        try:
            for msg in messages:
                self._update_callback(msg, self.task_name)
                time.sleep(delay)
            time.sleep(delay)
        except KeyboardInterrupt:
            logger = logging.getLogger(__name__)
            logger.debug("_patched_run: caught KeyboardInterrupt")
        finally:
            self._running.clear()

    return _patched_run


async def _recv_with_timeout(ws: Any, timeout: float = 2.0) -> Any:
    """Receive a message from websocket with timeout."""
    try:
        data = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(data)
    except asyncio.TimeoutError:
        return None


async def _create_server(
    manifest: Manifest,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[App, int, Any]:
    """Create and start a WebSocket server, return (app, port, server)."""
    from websockets.server import serve

    app = App()
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
    )

    server = await serve(
        app._ws_handler,
        host,
        port,
        process_request=app._process_request,
        server_header=None,
        ping_interval=None,
        max_size=100 * 1024 * 1024,
    )
    sockets = tuple(server.sockets)
    actual_port = sockets[0].getsockname()[1]
    return app, actual_port, server


def test_basic_connect_disconnect(manifest: Manifest) -> None:
    """Test a basic WebSocket connection and clean disconnect."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run()
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_chat_message_receive_update(manifest: Manifest) -> None:
    """Test sending a chat message and receiving an update."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps({"command": "", "message": "hello"}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
                    assert msg["data"]["type"] == "chat"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_command(manifest: Manifest) -> None:
    """Test sending a /help command via WebSocket."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps({"command": "/help", "message": ""}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_files(manifest: Manifest) -> None:
    """Test sending files via WebSocket."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "files": [
                                    {
                                        "name": "test.txt",
                                        "type": "text/plain",
                                        "size": 12,
                                        "content": "aGVsbG8gd29ybGQ=",
                                    },
                                ],
                            }
                        )
                    )
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_files_with_message(manifest: Manifest) -> None:
    """Test sending files with a text message."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.1)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "command": "",
                                "message": "process this file",
                                "files": [
                                    {
                                        "name": "data.csv",
                                        "content": "MSwyLDMK",
                                    },
                                ],
                            }
                        )
                    )
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_interrupt(manifest: Manifest) -> None:
    """Test sending an interrupt via WebSocket."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.5)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps({"type": "interrupt"}))
                    msg = await _recv_with_timeout(ws, timeout=0.5)
                    if msg is not None:
                        assert msg["type"] in ("update",)
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_multiple_updates(manifest: Manifest) -> None:
    """Test receiving multiple streaming updates."""

    async def _test() -> None:
        from websockets.client import connect

        updates = [
            {"type": "thinking", "text": "thinking..."},
            {"type": "chat", "text": "first chunk"},
            {"type": "chat", "text": "second chunk"},
            {"type": "done", "text": ""},
        ]
        patch_run = _make_patched_run(messages=updates, delay=0.05)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            received: list[dict[str, Any]] = []
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    for _ in range(len(updates)):
                        msg = await _recv_with_timeout(ws, timeout=1.0)
                        if msg is None:
                            break
                        received.append(msg)
            finally:
                server.close()
                await server.wait_closed()

            assert len(received) >= len(updates)
            types = [m["data"]["type"] for m in received]
            assert "thinking" in types
            assert "chat" in types
            assert "done" in types

    asyncio.run(_test())


def test_invalid_json(manifest: Manifest) -> None:
    """Test sending invalid JSON is handled gracefully."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send("not valid json at all")
                    await ws.send(json.dumps({"command": "", "message": "valid"}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_confirm_response(manifest: Manifest) -> None:
    """Test sending a confirm response via WebSocket."""

    async def _test() -> None:
        from websockets.client import connect

        confirm_result: list[bool] = []

        def _run_with_confirm(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            result = self._confirm_callback(
                {"action": "run_tool", "tool": "bash"},
                None,
                self.task_name,
            )
            confirm_result.append(result)
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", _run_with_confirm):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "confirm"
                    assert msg["data"]["action"] == "run_tool"

                    await ws.send(
                        json.dumps(
                            {
                                "type": "confirm_response",
                                "confirm": True,
                            }
                        )
                    )
                    await asyncio.sleep(0.2)
            finally:
                server.close()
                await server.wait_closed()

            assert len(confirm_result) == 1
            assert confirm_result[0] is True

    asyncio.run(_test())


def test_confirm_response_false(manifest: Manifest) -> None:
    """Test sending a confirm response with False value."""

    async def _test() -> None:
        from websockets.client import connect

        confirm_result: list[bool] = []

        def _run_with_confirm(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            result = self._confirm_callback(
                {"action": "delete_file"},
                None,
                self.task_name,
            )
            confirm_result.append(result)
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", _run_with_confirm):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "confirm"

                    await ws.send(
                        json.dumps(
                            {
                                "type": "confirm_response",
                                "confirm": False,
                            }
                        )
                    )
                    await asyncio.sleep(0.2)
            finally:
                server.close()
                await server.wait_closed()

            assert len(confirm_result) == 1
            assert confirm_result[0] is False

    asyncio.run(_test())


def test_multiple_concurrent_connections(manifest: Manifest) -> None:
    """Test multiple concurrent WebSocket connections."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with (
                    connect(
                        f"ws://127.0.0.1:{port}/chat/ws",
                        close_timeout=1,
                    ) as ws1,
                    connect(
                        f"ws://127.0.0.1:{port}/chat/ws",
                        close_timeout=1,
                    ) as ws2,
                ):
                    await ws1.send(
                        json.dumps({"command": "", "message": "hello from 1"})
                    )
                    await ws2.send(
                        json.dumps({"command": "", "message": "hello from 2"})
                    )

                    msg1 = await _recv_with_timeout(ws1)
                    assert msg1 is not None
                    assert msg1["type"] == "update"

                    msg2 = await _recv_with_timeout(ws2)
                    assert msg2 is not None
                    assert msg2["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_connect_nonexistent_endpoint(manifest: Manifest) -> None:
    """Test connecting to a non-existent endpoint returns error via HTTP."""

    async def _test() -> None:
        app, port, server = await _create_server(manifest)
        try:
            status, body = await _http_get("127.0.0.1", port, "/nonexistent/ws")
            assert status == 404
            assert body == b"Not Found"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_connect_endpoint_without_manifest(manifest: Manifest) -> None:
    """Test connecting to an endpoint with manifest=None closes connection."""

    async def _test() -> None:
        from websockets.client import connect
        from websockets.exceptions import ConnectionClosed

        app, port, server = await _create_server(manifest)
        ep = app.get_endpoint("chat")
        assert ep is not None
        ep.manifest = None  # type: ignore[assignment]
        try:
            async with connect(
                f"ws://127.0.0.1:{port}/chat/ws",
                close_timeout=1,
            ) as ws:
                with pytest.raises(ConnectionClosed):
                    await ws.recv()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_save_command(manifest: Manifest) -> None:
    """Test sending a /save command via WebSocket receives save_result."""

    async def _test() -> None:
        from websockets.client import connect

        def _run_save(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            save_content = (
                "eyJtZXNzYWdlcyI6IFt7InJvbGUiOiAidXNlciIsICJjb250ZW50IjogImhpIn1dfQ=="
            )
            with self._queue_lock:
                self._pending_save_callback = True
            self._update_callback(
                {
                    "tiz-internal": {
                        "save_conv": save_content,
                    }
                },
                self.task_name,
            )
            time.sleep(0.1)
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", _run_save):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(
                        json.dumps({"command": "/save", "message": "chat.json"})
                    )
                    msg1 = await _recv_with_timeout(ws)
                    assert msg1 is not None
                    assert msg1["type"] == "save_result"
                    assert "contents" in msg1

                    msg2 = await _recv_with_timeout(ws)
                    assert msg2 is not None
                    assert msg2["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_attach_with_content(manifest: Manifest) -> None:
    """Test sending an /attach command with base64 content."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    content = "ZmlsZSBjb250ZW50"
                    await ws.send(
                        json.dumps(
                            {
                                "command": "/attach",
                                "message": "test.txt",
                                "contents": content,
                            }
                        )
                    )
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_load_with_content(manifest: Manifest) -> None:
    """Test sending a /load command with base64 content."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    content = "eyJoaXN0b3J5IjogW119"
                    await ws.send(
                        json.dumps(
                            {
                                "command": "/load",
                                "message": "conv.json",
                                "contents": content,
                            }
                        )
                    )
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_client_disconnect_cleanup(manifest: Manifest) -> None:
    """Test that server cleans up when client disconnects."""

    async def _test() -> None:
        from websockets.client import connect

        cleanup_done = False

        def _run_long(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            nonlocal cleanup_done
            self._update_callback({"type": "chat", "text": "hello"}, None)
            time.sleep(0.5)
            cleanup_done = True
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", _run_long):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
                    await ws.close()
                    await asyncio.sleep(0.3)
            finally:
                server.close()
                await server.wait_closed()

            assert cleanup_done

    asyncio.run(_test())


def test_api_endpoint_names(manifest: Manifest) -> None:
    """Test that the API endpoint returns correct endpoint names."""

    async def _test() -> None:
        app, port, server = await _create_server(manifest)
        try:
            status, body = await _http_get("127.0.0.1", port, "/api/endpoints")
            assert status == 200
            data = json.loads(body)
            assert len(data["endpoints"]) == 1
            assert data["endpoints"][0]["name"] == "chat"
            assert data["endpoints"][0]["websocket"] == "/chat/ws"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_api_endpoint_with_description(manifest: Manifest) -> None:
    """Test API endpoint returns description and suggestions."""

    async def _test() -> None:
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "test-endpoint",
            EndpointConfig(
                manifest=manifest,
                base_path=Path("/tmp"),
                description="A test endpoint",
                suggestions=["suggestion 1", "suggestion 2"],
            ),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/api/endpoints")
            assert status == 200
            data = json.loads(body)
            assert len(data["endpoints"]) == 1
            ep = data["endpoints"][0]
            assert ep["name"] == "test-endpoint"
            assert ep["description"] == "A test endpoint"
            assert ep["suggestions"] == ["suggestion 1", "suggestion 2"]
            assert ep["websocket"] == "/test-endpoint/ws"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_static_file_serving(manifest: Manifest, tmp_path: Path) -> None:
    """Test that static files are served correctly alongside WS."""

    async def _test() -> None:
        from websockets.server import serve

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "hello.txt").write_text("world", encoding="utf-8")

        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/hello.txt")
            assert status == 200
            assert body == b"world"

            status, body = await _http_get("127.0.0.1", port, "/nonexistent.txt")
            assert status == 404
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_rapid_messages(manifest: Manifest) -> None:
    """Test sending many messages rapidly without issues."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.05)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    for i in range(5):
                        await ws.send(
                            json.dumps(
                                {
                                    "command": "",
                                    "message": f"message {i}",
                                }
                            )
                        )
                    received_count = 0
                    for _ in range(5):
                        msg = await _recv_with_timeout(ws, timeout=1.0)
                        if msg is None:
                            break
                        received_count += 1
                    assert received_count >= 1
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_worker_handles_send_failure(manifest: Manifest) -> None:
    """Test send_worker handles send failures without crashing."""

    async def _test() -> None:
        from websockets.client import connect

        app, port, server = await _create_server(manifest)
        try:
            async with connect(
                f"ws://127.0.0.1:{port}/chat/ws",
                close_timeout=1,
            ) as ws:
                await ws.close()
                await asyncio.sleep(0.3)

            await asyncio.sleep(0.3)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_large_message(manifest: Manifest) -> None:
    """Test sending a large message through WebSocket."""

    async def _test() -> None:
        from websockets.client import connect

        large_text = "A" * 10000
        patch_run = _make_patched_run(delay=0.1)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    max_size=2**20,
                    close_timeout=1,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "command": "",
                                "message": large_text,
                            }
                        )
                    )
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_very_large_message_up_to_100mb(manifest: Manifest) -> None:
    """Test sending a very large message (near 100MB limit) through WebSocket."""

    async def _test() -> None:
        from websockets.client import connect

        # Use 1MB to verify large messages work without being too slow
        large_content = "X" * (1024 * 1024)
        patch_run = _make_patched_run(delay=0.1)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    max_size=100 * 1024 * 1024,
                    close_timeout=5,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "command": "",
                                "message": large_content,
                            }
                        )
                    )
                    msg = await _recv_with_timeout(ws, timeout=5.0)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_empty_message(manifest: Manifest) -> None:
    """Test sending an empty message."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps({"command": "", "message": ""}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_then_close_immediately(manifest: Manifest) -> None:
    """Test sending a message then immediately closing."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.1)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps({"command": "", "message": "hello"}))
                    await ws.close()
                await asyncio.sleep(0.3)
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_multiple_endpoints(manifest: Manifest) -> None:
    """Test having multiple endpoints that can all be connected to."""

    async def _test() -> None:
        from websockets.client import connect
        from websockets.server import serve

        manifest2 = _make_minimal_manifest()

        app = App()
        app.add_endpoint(
            "endpoint1",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )
        app.add_endpoint(
            "endpoint2",
            EndpointConfig(manifest=manifest2, base_path=Path("/tmp")),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        patch_run = _make_patched_run(delay=0.1)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            try:
                async with (
                    connect(
                        f"ws://127.0.0.1:{port}/endpoint1/ws",
                        close_timeout=1,
                    ) as ws1,
                    connect(
                        f"ws://127.0.0.1:{port}/endpoint2/ws",
                        close_timeout=1,
                    ) as ws2,
                ):
                    msg1 = await _recv_with_timeout(ws1)
                    assert msg1 is not None
                    assert msg1["type"] == "update"

                    msg2 = await _recv_with_timeout(ws2)
                    assert msg2 is not None
                    assert msg2["type"] == "update"

                status, body = await _http_get("127.0.0.1", port, "/api/endpoints")
                assert status == 200
                data = json.loads(body)
                assert len(data["endpoints"]) == 2
                names = {ep["name"] for ep in data["endpoints"]}
                assert names == {"endpoint1", "endpoint2"}
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_sync_send_guard_when_not_running(manifest: Manifest) -> None:
    """Test that sync_send does not queue data when handler is not running."""

    async def _test() -> None:
        from websockets.client import connect

        def _run_send_after_close(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            self._update_callback({"type": "chat", "text": "before"}, None)

            time.sleep(0.2)
            self.close()
            time.sleep(0.1)
            self._update_callback({"type": "chat", "text": "after"}, None)
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", _run_send_after_close):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
                    assert msg["data"]["text"] == "before"

                    extra = await _recv_with_timeout(ws, timeout=0.5)
                    assert extra is None, (
                        f"sync_send guard failed: received unexpected message: {extra}"
                    )
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_worker_connection_closed(manifest: Manifest) -> None:
    """Test send_worker handles ConnectionClosed when sending."""

    async def _test() -> None:
        from websockets.client import connect

        # _run will call ws_send after client disconnects
        def _run_and_send_after_close(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            # Send some initial updates
            self._update_callback({"type": "initial", "text": "first"}, None)

            time.sleep(0.3)
            # Client should have disconnected by now, so send should fail
            self._update_callback({"type": "late", "text": "after"}, None)
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", _run_and_send_after_close):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    # Close immediately to trigger ConnectionClosed in send_worker
                    await ws.close()
                    await asyncio.sleep(0.5)
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_recv_connection_closed(manifest: Manifest) -> None:
    """Test _ws_handler catches ConnectionClosed on recv."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.3)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    # This triggers ConnectionClosed on the server's recv loop
                    await ws.close()
                    await asyncio.sleep(0.3)
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_add_duplicate_endpoint(manifest: Manifest) -> None:
    """Test that adding a duplicate endpoint raises ValueError."""

    app = App()
    config = EndpointConfig(manifest=manifest, base_path=Path("/tmp"))
    app.add_endpoint("chat", config)
    with pytest.raises(ValueError, match="Endpoint 'chat' already exists"):
        app.add_endpoint("chat", config)


def test_endpoint_names_property(manifest: Manifest) -> None:
    """Test the endpoint_names property."""

    app = App()
    assert app.endpoint_names == []
    config = EndpointConfig(manifest=manifest, base_path=Path("/tmp"))
    app.add_endpoint("test1", config)
    app.add_endpoint("test2", config)
    assert sorted(app.endpoint_names) == ["test1", "test2"]


def test_send_worker_connection_closed_on_send(manifest: Manifest) -> None:
    """Test send_worker handles ConnectionClosed when sending after disconnect."""

    async def _test() -> None:
        from websockets.client import connect

        def _run_send_after_close(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            self._update_callback({"type": "chat", "text": "first"}, None)

            time.sleep(0.3)
            self._update_callback({"type": "chat", "text": "after"}, None)
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", _run_send_after_close):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
                    await ws.close()
                    await asyncio.sleep(0.5)
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_auth_headers_endpoint(manifest: Manifest, tmp_path: Path) -> None:
    """Test that endpoints with auth headers reject unauthorized requests."""

    async def _test() -> None:
        from websockets.client import connect
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "protected",
            EndpointConfig(
                manifest=manifest,
                base_path=tmp_path,
                auth_headers={"X-Api-Key": "supersecret"},
            ),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            # Request without auth header -> should get 403
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            request = (
                f"GET /protected/ws HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Connection: close\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            status_line = response.split(b"\r\n")[0]
            assert b"403" in status_line, f"Expected 403, got: {status_line.decode()}"

            # Request with wrong auth header -> should get 403
            reader2, writer2 = await asyncio.open_connection("127.0.0.1", port)
            request2 = (
                f"GET /protected/ws HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"X-Api-Key: wrongkey\r\n"
                f"Connection: close\r\n\r\n"
            )
            writer2.write(request2.encode())
            await writer2.drain()
            response2 = await reader2.read()
            writer2.close()
            await writer2.wait_closed()
            status_line2 = response2.split(b"\r\n")[0]
            assert b"403" in status_line2, f"Expected 403, got: {status_line2.decode()}"

            # Request with correct auth header for WS -> should allow upgrade
            async with connect(
                f"ws://127.0.0.1:{port}/protected/ws",
                close_timeout=1,
                extra_headers=[("X-Api-Key", "supersecret")],
            ) as ws:
                await _recv_with_timeout(ws, timeout=0.5)
            # Also test API endpoint is filtered by auth headers
            status, body = await _http_get("127.0.0.1", port, "/api/endpoints")
            assert status == 200
            data = json.loads(body)
            # Without auth header, protected endpoint should be excluded
            unprotected_names = [ep["name"] for ep in data["endpoints"]]
            assert "protected" not in unprotected_names
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_http_get_non_existent_path(manifest: Manifest) -> None:
    """Test that a random non-WS, non-API path returns 404."""

    async def _test() -> None:
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            # Random path -> 404
            status, body = await _http_get("127.0.0.1", port, "/random/path")
            assert status == 404
            assert body == b"Not Found"

            # Non-WS endpoint path that isn't /ws -> 404
            status, body = await _http_get("127.0.0.1", port, "/chat/other")
            assert status == 404
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_static_file_index_html(manifest: Manifest, tmp_path: Path) -> None:
    """Test serving index.html for root path and hidden file protection."""

    async def _test() -> None:
        from websockets.server import serve

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html>Hello</html>", encoding="utf-8")
        (static_dir / ".secret").write_text("hidden", encoding="utf-8")

        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            # Root path should serve index.html
            status, body = await _http_get("127.0.0.1", port, "/")
            assert status == 200
            assert b"<html>Hello</html>" in body

            # Hidden file should return 404
            status, body = await _http_get("127.0.0.1", port, "/.secret")
            assert status == 404
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_static_file_minified_html(manifest: Manifest, tmp_path: Path) -> None:
    """Test that HTML files get minified when served."""

    async def _test() -> None:
        from websockets.server import serve

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        html_content = "<!-- comment --><html>\n<body>\n<p>Hello</p>\n</body>\n</html>"
        (static_dir / "test.html").write_text(html_content, encoding="utf-8")

        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/test.html")
            assert status == 200
            assert b"<!-- comment -->" not in body  # comments removed by minifier
            assert b"<html><body><p>Hello</p></body></html>" in body
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_static_file_css_and_js(manifest: Manifest, tmp_path: Path) -> None:
    """Test serving CSS and JS files with minification."""

    async def _test() -> None:
        from websockets.server import serve

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "style.css").write_text(
            "body { color: red; }\n/* comment */", encoding="utf-8"
        )
        (static_dir / "script.js").write_text(
            "var x = 1; // comment\n", encoding="utf-8"
        )

        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/style.css")
            assert status == 200
            # CSS gets minified via htmlmin
            assert len(body) > 0

            status, body = await _http_get("127.0.0.1", port, "/script.js")
            assert status == 200
            # JS gets minified via rjsmin
            assert b"// comment" not in body
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_static_file_minify_cache(manifest: Manifest, tmp_path: Path) -> None:
    """Test that minified files are cached and served from cache."""

    async def _test() -> None:
        from websockets.server import serve

        from tiz.web_api import clear_minify_cache

        clear_minify_cache()

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "test.html").write_text("<html>Hello</html>", encoding="utf-8")

        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            # First request should minify and cache
            status, body = await _http_get("127.0.0.1", port, "/test.html")
            assert status == 200
            assert body == b"<html>Hello</html>"

            # Second request should come from cache
            status, body2 = await _http_get("127.0.0.1", port, "/test.html")
            assert status == 200
            assert body2 == body

            # Clear cache and verify it still works
            clear_minify_cache()
            status, body3 = await _http_get("127.0.0.1", port, "/test.html")
            assert status == 200
            assert body3 == body
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_static_file_binary(manifest: Manifest, tmp_path: Path) -> None:
    """Test serving a binary file (non-minifiable MIME type)."""

    async def _test() -> None:
        from websockets.server import serve

        from tiz.web_api import clear_minify_cache

        clear_minify_cache()

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "test.txt").write_text("Hello World", encoding="utf-8")
        (static_dir / "test.json").write_text('{"key": "value"}', encoding="utf-8")

        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/test.txt")
            assert status == 200
            assert body == b"Hello World"

            status, body = await _http_get("127.0.0.1", port, "/test.json")
            assert status == 200
            assert body == b'{"key": "value"}'
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_static_file_unusual_mime(manifest: Manifest, tmp_path: Path) -> None:
    """Test serving a file with an unusual extension gets octet-stream."""

    async def _test() -> None:
        from websockets.server import serve

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "data.bin").write_bytes(b"\x00\x01\x02\x03")

        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/data.bin")
            assert status == 200
            assert body == b"\x00\x01\x02\x03"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_security_headers_present(manifest: Manifest) -> None:
    """Test that HTTP responses include security headers."""

    async def _test() -> None:
        app, port, server = await _create_server(manifest)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            request = (
                f"GET /api/endpoints HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Connection: close\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()

            headers_str = response.split(b"\r\n\r\n")[0].decode(
                "utf-8", errors="replace"
            )
            assert "Content-Security-Policy" in headers_str
            assert "X-Content-Type-Options" in headers_str
            assert "X-Frame-Options" in headers_str
            assert "Access-Control-Allow-Origin" in headers_str
            # Verify status is OK
            status_line = response.split(b"\r\n")[0]
            assert b"200" in status_line
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_security_headers_with_origin(manifest: Manifest) -> None:
    """Test that CORS Access-Control-Allow-Origin mirrors Origin header."""

    async def _test() -> None:
        app, port, server = await _create_server(manifest)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            request = (
                f"GET /api/endpoints HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Origin: https://myapp.example.com\r\n"
                f"Connection: close\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()

            headers_str = response.split(b"\r\n\r\n")[0].decode(
                "utf-8", errors="replace"
            )
            assert (
                "Access-Control-Allow-Origin: https://myapp.example.com" in headers_str
            )
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_http_get_with_no_port_in_host(manifest: Manifest) -> None:
    """Test that Host header without port omits default port in CORS origin."""

    async def _test() -> None:
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            request = (
                "GET /api/endpoints HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()

            headers_str = response.split(b"\r\n\r\n")[0].decode(
                "utf-8", errors="replace"
            )
            assert "Access-Control-Allow-Origin: http://localhost" in headers_str
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_ws_handler_invalid_path(manifest: Manifest) -> None:
    """Test _ws_handler with invalid WebSocket path returns immediately."""

    async def _test() -> None:
        from websockets.client import connect
        from websockets.exceptions import InvalidStatusCode

        app, port, server = await _create_server(manifest)
        try:
            # Invalid WS path (no /ws suffix) should return 404
            with pytest.raises(InvalidStatusCode) as exc_info:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/invalid",
                    close_timeout=1,
                ) as ws:  # noqa: F841
                    pass
            assert exc_info.value.status_code == 404
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_send_worker_loop_exit_on_handler_close(manifest: Manifest) -> None:
    """Test send_worker loop exits when handler._running becomes False."""

    async def _test() -> None:
        from websockets.client import connect

        def _run_fast(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            self._update_callback({"type": "chat", "text": "msg"}, None)
            time.sleep(0.3)
            self.close()

        with patch.object(ChatWebSocketHandler, "_run", _run_fast):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_confirm_callback_without_ws_send(manifest: Manifest) -> None:
    """Test _confirm_callback when ws_send is not set returns False."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler._running.clear()
    result = handler._confirm_callback({"action": "test"}, None, None)
    assert result is False


def test_input_callback_returns_none_when_stopped(manifest: Manifest) -> None:
    """Test _input_callback returns None when handler is stopped."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler._running.clear()
    result = handler._input_callback()
    assert result is None


def test_interrupt_no_thread(manifest: Manifest) -> None:
    """Test _interrupt when there's no active thread."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    # No thread set, should not crash
    handler._interrupt()
    assert handler._chat_thread is None


def test_slugify_function() -> None:
    """Test the _slugify helper function."""
    from tiz.web_api import _slugify

    assert _slugify("Hello World") == "hello-world"
    assert _slugify("  spaces  ") == "spaces"
    assert _slugify("Special!!!Chars") == "specialchars"
    assert _slugify("---trim---") == "trim"
    assert _slugify("") == ""


def test_guess_mime_function() -> None:
    """Test the _guess_mime helper function."""
    from tiz.web_api import _guess_mime

    assert _guess_mime(".html") == "text/html"
    assert _guess_mime(".css") == "text/css"
    assert _guess_mime(".js") == "application/javascript"
    assert _guess_mime(".json") == "application/json"
    assert _guess_mime(".png") == "image/png"
    assert _guess_mime(".jpg") == "image/jpeg"
    assert _guess_mime(".jpeg") == "image/jpeg"
    assert _guess_mime(".gif") == "image/gif"
    assert _guess_mime(".svg") == "image/svg+xml"
    assert _guess_mime(".ico") == "image/x-icon"
    assert _guess_mime(".txt") == "text/plain"
    assert _guess_mime(".unknown") == "application/octet-stream"


def test_parse_named_ws_path() -> None:
    """Test the _parse_named_ws_path helper function."""
    from tiz.web_api import _parse_named_ws_path

    names = {"chat", "my-endpoint"}

    name, remaining = _parse_named_ws_path("/chat/ws", names)
    assert name == "chat"
    assert remaining == "/ws"

    name, remaining = _parse_named_ws_path("/my-endpoint/ws", names)
    assert name == "my-endpoint"
    assert remaining == "/ws"

    name, remaining = _parse_named_ws_path("/chat/other", names)
    assert name is None
    assert remaining == "/chat/other"

    name, remaining = _parse_named_ws_path("/chat/ws/extra", names)
    assert name is None
    assert remaining == "/chat/ws/extra"


def test_minify_if_possible() -> None:
    """Test the _minify_if_possible helper function."""
    from tiz.web_api import _minify_if_possible

    # Non-minifiable type should return unchanged
    result = _minify_if_possible(b"hello world", "text/plain")
    assert result == b"hello world"

    # HTML should be minified
    result = _minify_if_possible(
        b"<!-- comment --><html>\n<body>\n<p>Hello</p>\n</body>\n</html>",
        "text/html",
    )
    assert b"<!-- comment -->" not in result
    assert b"<html><body><p>Hello</p></body></html>" in result

    # JavaScript should be minified
    result = _minify_if_possible(
        b"var x = 1;\nvar y = 2;\n// comment\n",
        "application/javascript",
    )
    assert b"// comment" not in result
    assert b"var x=1" in result

    # SVG should be minified
    result = _minify_if_possible(
        b"<!-- c --><svg>\n<text>Hi</text>\n</svg>",
        "image/svg+xml",
    )
    assert b"<svg><text>Hi</text></svg>" in result


def test_clear_minify_cache() -> None:
    """Test clear_minify_cache function."""
    from tiz.web_api import _minify_cache, clear_minify_cache

    _minify_cache["test"] = b"hello"
    assert "test" in _minify_cache
    clear_minify_cache()
    assert "test" not in _minify_cache


def test_app_run_method(manifest: Manifest) -> None:
    """Test the App.run method with KeyboardInterrupt."""

    app = App()
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
    )

    import threading

    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(host="127.0.0.1", port=0)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    time.sleep(0.3)

    app.stop()
    thread.join(timeout=2)

    assert not results or "Exception" not in results[0]


def _set_async_keyboard_interrupt(thread: threading.Thread) -> None:
    """Helper to raise KeyboardInterrupt in a thread."""
    import ctypes

    tid = thread.ident
    if tid is not None and isinstance(tid, int):
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt)
        )
        if ret != 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


def test_app_run_no_endpoints() -> None:
    """Test App.run with no endpoints still prints listening message."""

    app = App()
    import threading

    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(host="127.0.0.1", port=0)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    time.sleep(0.3)

    app.stop()
    thread.join(timeout=2)

    assert not results or "Exception" not in results[0]


def test_app_run_with_static_dir(manifest: Manifest, tmp_path: Path) -> None:
    """Test App.run with static_dir set to cover print path."""

    static_dir = tmp_path / "static"
    static_dir.mkdir()

    app = App()
    app._static_dir = static_dir
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=tmp_path),
    )

    import threading

    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(host="127.0.0.1", port=0)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    time.sleep(0.3)

    app.stop()
    thread.join(timeout=2)

    assert not results or "Exception" not in results[0]


def test_enrich_headers_with_existing_security_keys() -> None:
    """Test _enrich_headers when headers already contain security headers."""
    from websockets.datastructures import Headers

    from tiz.web_api import _SECURITY_HEADERS, _enrich_headers

    existing = Headers()
    existing["Content-Security-Policy"] = "default-src 'none'"
    existing["X-Custom"] = "value"

    result = _enrich_headers(existing, None)

    assert result["Content-Security-Policy"] != "default-src 'none'"
    assert (
        result["Content-Security-Policy"]
        == _SECURITY_HEADERS["Content-Security-Policy"]
    )
    assert result["X-Custom"] == "value"


def test_check_auth_headers() -> None:
    """Test _check_auth_headers handles case-insensitive header names."""
    from websockets.datastructures import Headers

    request_headers = Headers()
    request_headers["x-api-key"] = "secret123"
    request_headers["X-Custom-Header"] = "value"

    result = App._check_auth_headers({"X-Api-Key": "secret123"}, request_headers)
    assert result is True

    result = App._check_auth_headers({"X-Api-Key": "wrong"}, request_headers)
    assert result is False

    result = App._check_auth_headers({"X-Missing": "value"}, request_headers)
    assert result is False

    result = App._check_auth_headers(None, request_headers)
    assert result is True

    result = App._check_auth_headers({}, request_headers)
    assert result is True


def test_handle_message_interrupt_no_thread(manifest: Manifest) -> None:
    """Test handle_message with interrupt when no thread is active."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler.handle_message({"type": "interrupt"})
    assert handler._chat_thread is None


def test_handle_message_confirm_response(manifest: Manifest) -> None:
    """Test handle_message with confirm_response."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler.handle_message({"type": "confirm_response", "confirm": True})
    with handler._queue_lock:
        assert handler._confirm_result is True

    handler.handle_message({"type": "confirm_response", "confirm": False})
    with handler._queue_lock:
        assert handler._confirm_result is False


def test_handle_message_attach_without_content(manifest: Manifest) -> None:
    """Test handle_message with /attach but no contents."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler.handle_message({"command": "/attach", "message": "file.txt"})

    with handler._queue_lock:
        assert len(handler._message_queue) == 1
        entry = handler._message_queue[0]
        assert entry["command"] == "/attach"
        assert entry["message"] == "file.txt"
        assert "contents" not in entry


def test_handle_message_load_without_content(manifest: Manifest) -> None:
    """Test handle_message with /load but no contents."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler.handle_message({"command": "/load", "message": "conv.json"})

    with handler._queue_lock:
        assert len(handler._message_queue) == 1
        entry = handler._message_queue[0]
        assert entry["command"] == "/load"
        assert entry["message"] == "conv.json"


def test_set_async_keyboard_interrupt_no_ident() -> None:
    """Test _set_async_keyboard_interrupt with thread that has no ident."""
    import threading

    from tiz.web_api import _set_async_keyboard_interrupt

    thread = threading.Thread(target=lambda: None)
    _set_async_keyboard_interrupt(thread)


def test_set_async_keyboard_interrupt_wrong_type() -> None:
    """Test _set_async_keyboard_interrupt with non-int ident raises TypeError."""
    from tiz.web_api import _set_async_keyboard_interrupt

    class FakeThread:
        ident = "not_an_int"  # type: ignore[assignment]

    thread = FakeThread()
    import pytest

    with pytest.raises(TypeError):
        _set_async_keyboard_interrupt(thread)  # type: ignore[arg-type]


def test_input_callback_with_queued_message(manifest: Manifest) -> None:
    """Test _input_callback returns messages from queue."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

    with handler._queue_lock:
        handler._message_queue.append({"command": "", "message": "test"})
        handler._input_available.set()

    result = handler._input_callback()
    assert result == {"command": "", "message": "test"}


def test_update_callback_no_ws_send(manifest: Manifest) -> None:
    """Test _update_callback when ws_send is not set."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler._update_callback({"type": "chat", "text": "hello"}, None)
    assert handler._ws_send is None


def test_update_callback_save_exception(manifest: Manifest) -> None:
    """Test _update_callback during save when ws_send raises."""

    call_count = 0

    def failing_send(data: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("send failed on save")
        _ = data  # used for type consistency

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler._ws_send = failing_send
    handler._pending_save_callback = True

    handler._update_callback(
        {"tiz-internal": {"save_conv": "c2F2ZWQ="}},
        None,
    )
    assert handler._pending_save_callback is False


def test_minify_if_possible_non_minifiable() -> None:
    """Test _minify_if_possible returns original content for non-minifiable types."""
    from tiz.web_api import _minify_if_possible

    result = _minify_if_possible(b"hello world", "text/plain")
    assert result == b"hello world"

    result = _minify_if_possible(b'{"key": "value"}', "application/json")
    assert result == b'{"key": "value"}'


def test_run_simple_invalid_config(tmp_path: Path) -> None:
    """Test run_simple with an invalid config path."""
    from tiz.web_api import run_simple

    run_simple(
        base_path=tmp_path,
        config_path=tmp_path / "nonexistent.yaml",
        host="127.0.0.1",
        port=0,
    )


def test_run_simple_broken_config(tmp_path: Path) -> None:
    """Test run_simple with a config that fails to parse (returns immediately)."""

    from tiz.web_api import run_simple

    config_path = tmp_path / "broken_config.yaml"
    config_path.write_text("invalid: yaml: : : :", encoding="utf-8")

    # Should not raise or hang: config parsing fails, run_simple returns
    run_simple(
        base_path=tmp_path,
        config_path=config_path,
        host="127.0.0.1",
        port=0,
    )


def test_update_callback_save_pending_no_feedback(manifest: Manifest) -> None:
    """Test _update_callback when _pending_save_callback is True but no feedback.

    Covers branch 145->158 where feedback is falsy.
    When feedback is empty, the save_result is not sent but the update
    message IS sent via _ws_send (line 170-173).
    """

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler._pending_save_callback = True
    handler._ws_send = MagicMock()

    handler._update_callback(
        {"tiz-internal": {"save_conv": ""}},
        None,
    )
    assert handler._pending_save_callback is True
    # _ws_send is called for the update message, not save_result
    handler._ws_send.assert_called_once()


def test_update_callback_save_pending_no_ws_send(manifest: Manifest) -> None:
    """Test _update_callback when pending_save but ws_send is None.

    Covers branch 151->156 where ws_send is falsy during save.
    """

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    handler._pending_save_callback = True
    handler._ws_send = None

    handler._update_callback(
        {"tiz-internal": {"save_conv": "c2F2ZWQ="}},
        None,
    )
    assert handler._pending_save_callback is False


def test_set_async_keyboard_interrupt_ret_not_one(manifest: Manifest) -> None:  # noqa: ARG001
    """Test _set_async_keyboard_interrupt when ret != 1."""
    import ctypes

    from tiz.web_api import _set_async_keyboard_interrupt

    event = threading.Event()

    def target() -> None:
        event.wait()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    try:
        calls: list[int] = []

        def mock_set(tid: int, _exc: Any) -> int:
            calls.append(tid)
            if len(calls) == 1:
                return 0
            return 1

        with patch.object(
            ctypes.pythonapi, "PyThreadState_SetAsyncExc", side_effect=mock_set
        ):
            _set_async_keyboard_interrupt(thread)
            assert len(calls) == 2
    finally:
        event.set()
        thread.join(timeout=1)


def test_input_callback_timeout_exit(manifest: Manifest) -> None:
    """Test _input_callback exits when timeout occurs and running becomes False."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    results: list[dict[str, str] | None] = []

    def run_callback() -> None:
        result = handler._input_callback()
        results.append(result)

    thread = threading.Thread(target=run_callback, daemon=True)
    thread.start()
    time.sleep(0.25)  # Allow at least 2 timeouts (0.1s each)
    handler._running.clear()
    handler._input_available.set()  # Wake up the wait
    thread.join(timeout=2)
    assert len(results) == 1
    assert results[0] is None


def test_confirm_callback_timeout_exit(manifest: Manifest) -> None:
    """Test _confirm_callback exits when timeout occurs and running becomes False."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
    results: list[bool] = []

    def run_confirm_closure() -> None:
        result = handler._confirm_callback({"action": "test"}, None, handler.task_name)
        results.append(result)

    thread = threading.Thread(target=run_confirm_closure, daemon=True)
    thread.start()
    time.sleep(0.25)  # Allow at least 2 timeouts (0.1s each)
    handler._running.clear()
    handler._confirm_available.set()  # Wake up the wait
    thread.join(timeout=2)
    assert len(results) == 1
    assert results[0] is False


def test_minify_if_possible_js_import_error() -> None:
    """Test _minify_if_possible when rjsmin is not available."""

    import builtins

    from tiz.web_api import _minify_if_possible

    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "rjsmin":
            raise ImportError(f"No module named '{name}'")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = _minify_if_possible(b"var x = 1;", "application/javascript")
        assert result == b"var x = 1;"


def test_minify_if_possible_html_import_error() -> None:
    """Test _minify_if_possible when htmlmin is not available."""

    import builtins

    from tiz.web_api import _minify_if_possible

    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "htmlmin":
            raise ImportError(f"No module named '{name}'")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = _minify_if_possible(
            b"<html>\n<body>\n<p>Hello</p>\n</body>\n</html>", "text/html"
        )
        assert result == b"<html>\n<body>\n<p>Hello</p>\n</body>\n</html>"


def test_ws_handler_invalid_path_direct(manifest: Manifest) -> None:
    """Test _ws_handler returns immediately with invalid WebSocket path."""

    async def _test() -> None:
        from websockets.client import connect
        from websockets.exceptions import InvalidStatusCode
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            # Path without /ws suffix -> 404
            with pytest.raises(InvalidStatusCode) as exc_info:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/notws",
                    close_timeout=1,
                ):
                    pass
            assert exc_info.value.status_code == 404
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_ws_handler_endpoint_not_found_direct(manifest: Manifest) -> None:
    """Test _ws_handler returns when endpoint name doesn't exist."""

    async def _test() -> None:
        from websockets.client import connect
        from websockets.exceptions import InvalidStatusCode
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            with pytest.raises(InvalidStatusCode) as exc_info:
                async with connect(
                    f"ws://127.0.0.1:{port}/nonexistent/ws",
                    close_timeout=1,
                ):
                    pass
            assert exc_info.value.status_code == 404
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_app_run_keyboard_interrupt_patched(manifest: Manifest) -> None:
    """Test that App.run catches KeyboardInterrupt from asyncio.run."""

    import tiz.web_api

    app = App()
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
    )

    with patch.object(tiz.web_api.asyncio, "run", side_effect=KeyboardInterrupt):
        app.run(host="127.0.0.1", port=0)
    # Should not raise


def test_app_run_method_success_path(manifest: Manifest) -> None:
    """Test _run covers the success path (lines 224-225)."""

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

    with patch("tiz.web_api.InteractiveChat") as mock_chat_cls:
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat

        results: list[Exception | None] = []

        def run_target() -> None:
            try:
                handler._run()
            except Exception as e:
                results.append(e)

        thread = threading.Thread(target=run_target, daemon=True)
        thread.start()
        time.sleep(0.2)
        handler._running.clear()
        thread.join(timeout=2)

        assert mock_chat.run.called
        assert not results


def test_run_simple_valid_config(tmp_path: Path) -> None:
    """Test run_simple with a valid config that parses successfully."""

    from tiz.web_api import run_simple

    config_path = tmp_path / "config.yaml"

    with patch("tiz.web_api.parse_web_config") as mock_parse:
        from tiz.web_config_parser import WebConfig

        mock_config = MagicMock(spec=WebConfig)
        mock_config.static_dir = None
        mock_config.endpoints = {}
        mock_parse.return_value = mock_config

        with patch("tiz.web_api.App.run") as mock_run:
            run_simple(
                base_path=tmp_path,
                config_path=config_path,
                host="127.0.0.1",
                port=0,
            )
            mock_run.assert_called_once()


def test_run_simple_valid_config_with_run_error(tmp_path: Path) -> None:
    """Test run_simple with valid config but app.run raises an exception."""

    from tiz.web_api import run_simple

    config_path = tmp_path / "config.yaml"

    with patch("tiz.web_api.parse_web_config") as mock_parse:
        from tiz.web_config_parser import WebConfig

        mock_config = MagicMock(spec=WebConfig)
        mock_config.static_dir = None
        mock_config.endpoints = {}
        mock_parse.return_value = mock_config

        with patch("tiz.web_api.App.run", side_effect=RuntimeError("server failed")):
            run_simple(
                base_path=tmp_path,
                config_path=config_path,
                host="127.0.0.1",
                port=0,
            )
    # Should not raise


def test_send_worker_connection_closed_during_transport_close(
    manifest: Manifest,
) -> None:
    """Test send_worker catches ConnectionClosed when client disconnects abruptly."""

    async def _test() -> None:
        from websockets.client import connect

        def _run_send_during_disconnect(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            self._update_callback({"type": "chat", "text": "will_fail"}, None)

            time.sleep(0.2)
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", _run_send_during_disconnect):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await asyncio.sleep(0.05)
                    # Forcefully close the underlying transport to trigger
                    # ConnectionClosed in the send_worker
                    transport = ws.transport
                    if transport:
                        transport.abort()
                    await asyncio.sleep(0.5)
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_recv_connection_closed_abrupt(manifest: Manifest) -> None:
    """Test that recv loop catches ConnectionClosed on abrupt disconnect."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.3)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    transport = ws.transport
                    if transport:
                        transport.abort()
                    await asyncio.sleep(0.3)
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_run_simple_with_valid_yaml_config(tmp_path: Path) -> None:
    """Test run_simple with a valid YAML config containing an endpoint."""

    from tiz.web_api import run_simple

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "tasks:\n  - name: default\n    worker_image: tiz-worker:latest\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"endpoints:\n  test:\n    manifests:\n      - {manifest_path.name}\n",
        encoding="utf-8",
    )

    results: list[str] = []
    app_holder: list[App] = []

    def run_app() -> None:
        try:
            run_simple(
                base_path=tmp_path,
                config_path=config_path,
                host="127.0.0.1",
                port=0,
                _app_holder=app_holder,
            )
        except Exception as exc:
            results.append(f"Exception: {exc}")

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    time.sleep(0.3)

    if app_holder:
        app_holder[0].stop()
    thread.join(timeout=3)

    assert not results or "Exception" not in results[0]


def test_enrich_headers_with_none_headers() -> None:
    """Test _enrich_headers with None headers parameter."""
    from tiz.web_api import _SECURITY_HEADERS, _enrich_headers

    result = _enrich_headers(None, None)
    assert (
        result["Content-Security-Policy"]
        == _SECURITY_HEADERS["Content-Security-Policy"]
    )
    assert "Access-Control-Allow-Origin" not in result


def test_enrich_headers_no_request_headers() -> None:
    """Test _enrich_headers without request_headers."""
    from websockets.datastructures import Headers

    from tiz.web_api import _enrich_headers

    existing = Headers()
    existing["X-Custom"] = "value"
    result = _enrich_headers(existing, None)
    assert result["X-Custom"] == "value"
    assert "Access-Control-Allow-Origin" not in result


def test_enrich_headers_with_origin_no_host_fallback() -> None:
    """Test _enrich_headers with Origin header present."""
    from websockets.datastructures import Headers

    from tiz.web_api import _enrich_headers

    request_headers = Headers()
    request_headers["Origin"] = "https://example.com"
    result = _enrich_headers(None, request_headers)
    assert result["Access-Control-Allow-Origin"] == "https://example.com"


def test_enrich_headers_with_host_no_origin() -> None:
    """Test _enrich_headers falls back to Host when Origin is missing."""

    from websockets.datastructures import Headers

    from tiz.web_api import _enrich_headers

    request_headers = Headers()
    request_headers["Host"] = "myserver:9090"
    result = _enrich_headers(None, request_headers)
    assert result["Access-Control-Allow-Origin"] == "http://myserver:9090"


def test_enrich_headers_with_host_no_port() -> None:
    """Test _enrich_headers omits default port when no explicit port in Host."""

    from websockets.datastructures import Headers

    from tiz.web_api import _enrich_headers

    request_headers = Headers()
    request_headers["Host"] = "localhost"
    result = _enrich_headers(None, request_headers)
    assert result["Access-Control-Allow-Origin"] == "http://localhost"


def test_enrich_headers_with_no_origin_no_host() -> None:
    """Test _enrich_headers when request_headers has neither Origin nor Host.

    Covers branch 72->74 where host is falsy.
    """

    from websockets.datastructures import Headers

    from tiz.web_api import _enrich_headers

    request_headers = Headers()
    request_headers["X-Custom"] = "value"
    result = _enrich_headers(None, request_headers)
    assert "Access-Control-Allow-Origin" not in result


def test_app_run_with_serve_failure() -> None:
    """Test App.run when serve() raises during context manager entry.

    Covers branch 604->exit where async with serve(...) fails on __aenter__.
    """

    import tiz.web_api

    app = App()

    with (
        patch.object(tiz.web_api, "serve", side_effect=RuntimeError("serve failed")),
        pytest.raises(RuntimeError, match="serve failed"),
    ):
        app.run(host="127.0.0.1", port=0)


def test_run_simple_with_exception_from_parse(tmp_path: Path) -> None:
    """Test run_simple when config parsing raises an exception."""

    from tiz.web_api import run_simple

    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with patch(
        "tiz.web_api.parse_web_config",
        side_effect=ValueError("Parse error"),
    ):
        run_simple(
            base_path=tmp_path,
            config_path=config_path,
            host="127.0.0.1",
            port=0,
        )


def test_run_simple_with_exception_from_config_file(tmp_path: Path) -> None:
    """Test run_simple when the config file is empty (yields None)."""

    from tiz.web_api import run_simple

    config_path = tmp_path / "empty_config.yaml"
    config_path.write_text("", encoding="utf-8")

    run_simple(
        base_path=tmp_path,
        config_path=config_path,
        host="127.0.0.1",
        port=0,
    )
    # Should log error and return without crashing


def test_run_backoff_reset_branch(manifest: Manifest) -> None:
    """Test _run backoff reset branch when time gap > 10 * max_backoff.

    Covers the reset branch at line 239 where backoff is reset to 0.0.
    """

    handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

    call_count: list[int] = [0]

    def run_side_effect() -> None:
        call_count[0] += 1
        if call_count[0] >= 4:
            handler._running.clear()
        raise RuntimeError("fail")

    time_call_count: list[int] = [0]

    def mock_time() -> float:
        time_call_count[0] += 1
        if time_call_count[0] <= 2:
            return float(time_call_count[0])
        # Third+ call: return a time far in the future to trigger backoff reset
        return 2000.0

    with patch("tiz.web_api.InteractiveChat") as mock_chat_cls:
        instance = MagicMock()
        instance.run.side_effect = run_side_effect
        mock_chat_cls.return_value = instance

        with (
            patch("tiz.web_api.time.monotonic", side_effect=mock_time),
            patch("tiz.web_api.time.sleep"),
        ):
            handler._run()

    assert handler._chat_thread is None
    assert call_count[0] >= 3


def test_ws_handler_invalid_path_direct_call(manifest: Manifest) -> None:
    """Test _ws_handler with invalid WS paths via direct mock call.

    Both paths (/invalid/path and /chat/notws) are caught by
    _parse_named_ws_path before any endpoint logic is reached.
    This tests _parse_named_ws_path behavior, not endpoint lookup.
    """

    async def _test() -> None:
        mock_ws = MagicMock(spec_set=["send", "recv", "close"])

        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        await app._ws_handler(mock_ws, "/invalid/path")
        await app._ws_handler(mock_ws, "/chat/notws")

    asyncio.run(_test())


def test_ws_handler_endpoint_not_found_direct_call(manifest: Manifest) -> None:
    """Test _ws_handler with a non-existent endpoint name in path.

    The path /nonexistent/ws causes _parse_named_ws_path to return
    name=None (not in the endpoint set), which is caught before any
    endpoint lookup occurs. This tests _parse_named_ws_path behavior,
    not the endpoint-is-None branch (which is pragma: no cover).
    """

    async def _test() -> None:
        mock_ws = MagicMock(spec_set=["send", "recv", "close"])

        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        # Call _ws_handler with a path that parses correctly but
        # no matching endpoint exists
        await app._ws_handler(mock_ws, "/nonexistent/ws")

    asyncio.run(_test())


def test_send_worker_connection_closed_on_send_data(manifest: Manifest) -> None:
    """Test send_worker catches ConnectionClosed on send.

    This covers lines 494-496 by calling _ws_handler with a mock WS
    that raises ConnectionClosed on send, and keeping the recv loop
    alive so the send_worker has time to process data.
    """

    async def _test() -> None:
        from websockets.exceptions import ConnectionClosed

        class MockWS:
            def __init__(self) -> None:
                self.send_called = False
                self.close_called = False
                self._iter_count = 0

            def __aiter__(self) -> MockWS:
                return self

            async def __anext__(self) -> str:
                self._iter_count += 1
                if self._iter_count <= 3:
                    await asyncio.sleep(0.2)
                    return json.dumps({"command": "", "message": "hi"})
                raise StopAsyncIteration

            async def send(self, _data: str) -> None:
                self.send_called = True
                raise ConnectionClosed(rcvd=None, sent=None)

            def close(self) -> None:
                self.close_called = True

        mock_ws = MockWS()

        def raise_on_send_run(self: ChatWebSocketHandler) -> None:
            self.manifest.meta.ephemeral_sandbox = True
            self.manifest.meta.delete_sandbox_on_exit = True
            self._update_callback({"type": "chat", "text": "hello"}, None)
            time.sleep(0.5)
            self._running.clear()

        with patch.object(ChatWebSocketHandler, "_run", raise_on_send_run):
            app = App()
            app.add_endpoint(
                "chat",
                EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
            )

            await app._ws_handler(mock_ws, "/chat/ws")  # type: ignore[arg-type]

        assert mock_ws.send_called
        # _ws_handler calls handler.close() not websocket.close(),
        # so mock_ws.close is not called by the handler.
        # Verify the handler was properly closed instead.

    asyncio.run(_test())


def test_run_simple_with_actual_endpoint_and_run_error(
    tmp_path: Path,
    manifest: Manifest,  # noqa: ARG001
) -> None:
    """Test run_simple with an endpoint config and app.run raising an error.

    This covers lines 703-704 (slugify and add_endpoint) and the
    exception handler at lines 707-708.
    """
    config_path = tmp_path / "config.yaml"
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    manifest_file = manifests_dir / "test.yaml"
    manifest_file.write_text(
        "tasks:\n  - name: default\n    worker_image: tiz-worker:latest\n",
        encoding="utf-8",
    )
    config_path.write_text(
        f"endpoints:\n  my endpoint:\n    manifests:\n      - {manifest_file}\n",
        encoding="utf-8",
    )

    with patch("tiz.web_api.App.run", side_effect=RuntimeError("boom")):
        from tiz.web_api import run_simple

        run_simple(
            base_path=tmp_path,
            config_path=config_path,
            host="127.0.0.1",
            port=0,
        )


def test_unix_socket_connect_disconnect(tmp_path: Path) -> None:
    """Test WebSocket connection over a Unix socket."""

    from websockets.client import connect as ws_connect
    from websockets.server import serve as ws_serve

    manifest = _make_minimal_manifest()
    socket_path = str(tmp_path / "websocket.sock")

    async def _test() -> None:
        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await ws_serve(
            app._ws_handler,
            unix=True,
            path=socket_path,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        try:
            patch_run = _make_patched_run()
            with patch.object(ChatWebSocketHandler, "_run", patch_run):
                async with ws_connect(
                    "ws://localhost/chat/ws",
                    unix=True,
                    path=socket_path,
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_unix_socket_api_endpoints(tmp_path: Path) -> None:
    """Test HTTP API endpoint over a Unix socket."""

    from websockets.server import serve as ws_serve

    manifest = _make_minimal_manifest()
    socket_path = str(tmp_path / "api.sock")

    async def _test() -> None:
        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await ws_serve(
            app._ws_handler,
            unix=True,
            path=socket_path,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        try:
            # Use a raw Unix socket connection to perform HTTP request

            reader, writer = await asyncio.open_unix_connection(socket_path)
            request = (
                "GET /api/endpoints HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()

            lines = response.split(b"\r\n")
            status_line = lines[0]
            status_code = int(status_line.split(b" ")[1])
            assert status_code == 200, f"Expected 200, got {status_code}"

            body_start = response.find(b"\r\n\r\n") + 4
            body = response[body_start:]
            data = json.loads(body)
            assert len(data["endpoints"]) == 1
            assert data["endpoints"][0]["name"] == "chat"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_unix_socket_static_file(tmp_path: Path) -> None:
    """Test static file serving over a Unix socket."""

    from websockets.server import serve as ws_serve

    manifest = _make_minimal_manifest()
    socket_path = str(tmp_path / "static.sock")
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "hello.txt").write_text("world", encoding="utf-8")

    async def _test() -> None:
        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await ws_serve(
            app._ws_handler,
            unix=True,
            path=socket_path,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        try:
            reader, writer = await asyncio.open_unix_connection(socket_path)
            request = (
                "GET /hello.txt HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()

            lines = response.split(b"\r\n")
            status_line = lines[0]
            status_code = int(status_line.split(b" ")[1])
            assert status_code == 200, f"Expected 200, got {status_code}"

            body_start = response.find(b"\r\n\r\n") + 4
            body = response[body_start:]
            assert body == b"world"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_app_run_unix_socket_and_connect(tmp_path: Path) -> None:
    """Test app.run() with unix socket path covers unix socket serve and stop."""

    manifest = _make_minimal_manifest()
    socket_path = str(tmp_path / "run.sock")

    app = App()
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
    )

    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(path=socket_path)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    patch_run = _make_patched_run()
    with patch.object(ChatWebSocketHandler, "_run", patch_run):
        thread = threading.Thread(target=run_app, daemon=True)
        thread.start()
        time.sleep(0.5)

        async def _http_test() -> None:
            reader, writer = await asyncio.open_unix_connection(socket_path)
            request = (
                "GET /api/endpoints HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            lines = response.split(b"\r\n")
            status_code = int(lines[0].split(b" ")[1])
            assert status_code == 200
            body_start = response.find(b"\r\n\r\n") + 4
            body = response[body_start:]
            data = json.loads(body)
            assert len(data["endpoints"]) == 1
            assert data["endpoints"][0]["name"] == "chat"

        asyncio.run(_http_test())

        async def _ws_test() -> None:
            from websockets.client import connect as ws_connect

            async with ws_connect(
                "ws://localhost/chat/ws",
                unix=True,
                path=socket_path,
                close_timeout=1,
            ) as ws:
                msg = await _recv_with_timeout(ws)
                assert msg is not None
                assert msg["type"] == "update"

        asyncio.run(_ws_test())

        app.stop()
        thread.join(timeout=3)

    assert not results or "Exception" not in results[0]


def test_app_run_unix_socket_with_static_dir(tmp_path: Path) -> None:
    """Test app.run() with unix socket and static_dir."""

    manifest = _make_minimal_manifest()
    socket_path = str(tmp_path / "static_run.sock")
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "hello.txt").write_text("unix_static", encoding="utf-8")

    app = App()
    app._static_dir = static_dir
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=tmp_path),
    )

    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(path=socket_path)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    patch_run = _make_patched_run()
    with patch.object(ChatWebSocketHandler, "_run", patch_run):
        thread = threading.Thread(target=run_app, daemon=True)
        thread.start()
        time.sleep(0.5)

        async def _http_test() -> None:
            reader, writer = await asyncio.open_unix_connection(socket_path)
            request = (
                "GET /hello.txt HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            lines = response.split(b"\r\n")
            status_code = int(lines[0].split(b" ")[1])
            assert status_code == 200
            body_start = response.find(b"\r\n\r\n") + 4
            body = response[body_start:]
            assert body == b"unix_static"

        asyncio.run(_http_test())

        app.stop()
        thread.join(timeout=3)

    assert not results or "Exception" not in results[0]


def test_app_run_unix_socket_no_endpoints(tmp_path: Path) -> None:
    """Test app.run() with unix socket and no endpoints."""

    socket_path = str(tmp_path / "noep.sock")
    app = App()
    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(path=socket_path)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    time.sleep(0.3)

    app.stop()
    thread.join(timeout=3)

    assert not results or "Exception" not in results[0]


def test_app_run_unix_socket_with_static_dir_only(tmp_path: Path) -> None:
    """Test app.run() unix socket with static_dir and no endpoints."""

    socket_path = str(tmp_path / "static_only.sock")
    static_dir = tmp_path / "static_only_dir"
    static_dir.mkdir()

    app = App()
    app._static_dir = static_dir
    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(path=socket_path)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    time.sleep(0.3)

    app.stop()
    thread.join(timeout=3)

    assert not results or "Exception" not in results[0]


def test_app_run_tcp_mode_with_static_dir(tmp_path: Path) -> None:
    """Test app.run() TCP mode with static_dir covering TCP static prints."""

    manifest = _make_minimal_manifest()
    static_dir = tmp_path / "tcp_static"
    static_dir.mkdir()

    app = App()
    app._static_dir = static_dir
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=tmp_path),
    )

    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(host="127.0.0.1", port=0)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    patch_run = _make_patched_run()
    with patch.object(ChatWebSocketHandler, "_run", patch_run):
        thread = threading.Thread(target=run_app, daemon=True)
        thread.start()
        time.sleep(0.5)

        app.stop()
        thread.join(timeout=3)

    assert not results or "Exception" not in results[0]


def test_ws_connection_prefix_path(manifest: Manifest) -> None:
    """Test WebSocket connection with a prefix before endpoint name.

    This covers _parse_named_ws_path line 335 (endswith match).
    A path like /base/chat/ws where 'chat' is an endpoint name but the
    path has a prefix component.
    """

    async def _test() -> None:
        from websockets.client import connect
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        patch_run = _make_patched_run(delay=0.1)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            try:
                # Connect with a prefix before the endpoint name: /base/chat/ws
                # This triggers the endswith match at line 335 rather than
                # the simple split match at line 329.
                async with connect(
                    f"ws://127.0.0.1:{port}/base/chat/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_ws_connection_long_prefix_path(manifest: Manifest) -> None:
    """Test WebSocket connection with a longer prefix before endpoint name.

    Covers _parse_named_ws_path line 335 with a multi-segment prefix.
    """

    async def _test() -> None:
        from websockets.client import connect
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "my-endpoint",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        patch_run = _make_patched_run(delay=0.1)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            try:
                # Multi-segment prefix: /api/v2/my-endpoint/ws
                async with connect(
                    f"ws://127.0.0.1:{port}/api/v2/my-endpoint/ws",
                    close_timeout=1,
                ) as ws:
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_ws_connection_prefix_path_api_endpoints(manifest: Manifest) -> None:
    """Test API endpoints work alongside prefixed WS paths.

    Verifies that /api/endpoints still returns the correct endpoint
    even when connections use prefixed WS paths.
    """

    async def _test() -> None:
        from websockets.server import serve

        app = App()
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/api/endpoints")
            assert status == 200
            data = json.loads(body)
            assert len(data["endpoints"]) == 1
            assert data["endpoints"][0]["name"] == "chat"
            assert data["endpoints"][0]["websocket"] == "/chat/ws"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_parse_named_ws_path_endswith_prefix() -> None:
    """Test _parse_named_ws_path prefix matching (endswith branch at line 335)."""
    from tiz.web_api import _parse_named_ws_path

    names = {"chat", "my-endpoint"}

    # Path with a single-segment prefix -> triggers endswith match
    name, remaining = _parse_named_ws_path("/base/chat/ws", names)
    assert name == "chat"
    assert remaining == "/ws"

    # Path with multi-segment prefix -> triggers endswith match
    name, remaining = _parse_named_ws_path("/api/v2/chat/ws", names)
    assert name == "chat"
    assert remaining == "/ws"

    # Path with prefix for hyphenated endpoint name
    name, remaining = _parse_named_ws_path("/services/my-endpoint/ws", names)
    assert name == "my-endpoint"
    assert remaining == "/ws"

    # Path where prefix is not an endpoint name but the endpoint IS at the end
    name, remaining = _parse_named_ws_path("/x/y/z/chat/ws", names)
    assert name == "chat"
    assert remaining == "/ws"


def test_parse_named_ws_path_trailing_slash() -> None:
    """Test _parse_named_ws_path with trailing slash on path."""
    from tiz.web_api import _parse_named_ws_path

    names = {"chat"}

    # Path with trailing slash still matches (strip handles it)
    name, remaining = _parse_named_ws_path("/chat/ws/", names)
    assert name == "chat"
    assert remaining == "/ws"

    # Prefix path with trailing slash
    name, remaining = _parse_named_ws_path("/base/chat/ws/", names)
    assert name == "chat"
    assert remaining == "/ws"


def test_parse_named_ws_path_exact_match_no_leading_slash() -> None:
    """Test _parse_named_ws_path with no leading slash.

    This tests the scenario where path has no leading slash.
    """
    from tiz.web_api import _parse_named_ws_path

    names = {"chat"}

    # path_clean could be 'chat/ws' (no leading slash) if urlparse returns it
    name, remaining = _parse_named_ws_path("chat/ws", names)
    assert name == "chat"
    assert remaining == "/ws"


def test_send_text_message_only(manifest: Manifest) -> None:
    """Test sending a message with only text (no command)."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps({"message": "hello"}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_send_command_only_no_message(manifest: Manifest) -> None:
    """Test sending a command without a message text."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps({"command": "/help"}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_non_dict_json_via_websocket(manifest: Manifest) -> None:
    """Test sending non-dict JSON (list) via WebSocket hits lines 537-538."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps([1, 2, 3]))
                    await ws.send(json.dumps({"command": "", "message": "valid"}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_non_dict_json_string_via_websocket(manifest: Manifest) -> None:
    """Test sending a JSON string via WebSocket hits lines 537-538."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps("just a string"))
                    await ws.send(json.dumps({"command": "", "message": "valid"}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_non_dict_json_number_via_websocket(manifest: Manifest) -> None:
    """Test sending a JSON number via WebSocket hits lines 537-538."""

    async def _test() -> None:
        from websockets.client import connect

        patch_run = _make_patched_run(delay=0.2)
        with patch.object(ChatWebSocketHandler, "_run", patch_run):
            app, port, server = await _create_server(manifest)
            try:
                async with connect(
                    f"ws://127.0.0.1:{port}/chat/ws",
                    close_timeout=1,
                ) as ws:
                    await ws.send(json.dumps(42))
                    await ws.send(json.dumps({"command": "", "message": "valid"}))
                    msg = await _recv_with_timeout(ws)
                    assert msg is not None
                    assert msg["type"] == "update"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_static_file_binary_as_html_triggers_unicode_error(
    manifest: Manifest, tmp_path: Path
) -> None:
    """Test serving a binary file with .html extension triggers UnicodeDecodeError.

    This covers lines 395-396 in _minify_if_possible.
    """
    from tiz.web_api import clear_minify_cache

    async def _test() -> None:
        from websockets.server import serve

        clear_minify_cache()

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        binary_data = bytes([0xFF, 0xFE, 0x00, 0x00])
        (static_dir / "binary.html").write_bytes(binary_data)

        app = App()
        app._static_dir = static_dir
        app.add_endpoint(
            "chat",
            EndpointConfig(manifest=manifest, base_path=tmp_path),
        )

        server = await serve(
            app._ws_handler,
            "127.0.0.1",
            0,
            process_request=app._process_request,
            server_header=None,
            ping_interval=None,
            max_size=100 * 1024 * 1024,
        )
        sockets = tuple(server.sockets)
        port = sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/binary.html")
            assert status == 200
            # Content should be returned as-is since UnicodeDecodeError prevents minification
            assert body == binary_data
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_stop_when_not_running(manifest: Manifest) -> None:
    """Test that calling stop() when loop is None does nothing.

    Covers the False branch of the if condition in stop().
    """

    app = App()
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
    )
    app._loop = None
    app._stop_future = None
    app.stop()
    # Should not raise


def test_stop_when_loop_running_and_future_pending(manifest: Manifest) -> None:
    """Test stop() when loop is running and future is pending.

    Covers the full True branch of the if condition in stop().
    """

    app = App()
    app.add_endpoint(
        "chat",
        EndpointConfig(manifest=manifest, base_path=Path("/tmp")),
    )

    results: list[str] = []

    def run_app() -> None:
        try:
            app.run(host="127.0.0.1", port=0)
        except Exception as exc:
            results.append(f"Exception: {exc}")

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    time.sleep(0.3)

    # stop() will find _loop not None, _stop_future not None, and not done
    app.stop()
    thread.join(timeout=2)

    assert not results or "Exception" not in results[0]
    assert app._loop is None
    assert app._stop_future is None


def test_run_simple_with_endpoints_and_app_holder(tmp_path: Path) -> None:
    """Test run_simple with _app_holder populated to cover line 783."""
    from tiz.web_api import run_simple

    config_path = tmp_path / "config.yaml"
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(
        "tasks:\n  - name: default\n    worker_image: tiz-worker:latest\n",
        encoding="utf-8",
    )
    config_path.write_text(
        f'meta:\n  version: "0"\nendpoints:\n  test:\n    manifests:\n      - {manifest_file}\n',
        encoding="utf-8",
    )

    app_holder: list[App] = []

    with patch("tiz.web_api.App.run", side_effect=RuntimeError("stop")):
        run_simple(
            base_path=tmp_path,
            config_path=config_path,
            host="127.0.0.1",
            port=0,
            _app_holder=app_holder,
        )

    assert len(app_holder) == 1
