"""Web API for tiz chat with WebSocket support via websockets library."""

from __future__ import annotations

import asyncio
import ctypes
import json
import re
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from websockets.datastructures import Headers, SupportsKeysAndGetItem
from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import HTTPResponse
from websockets.server import WebSocketServerProtocol, serve

from tiz.helpers import _format_exc
from tiz.interactive_chat import InteractiveChat
from tiz.log import get_logger
from tiz.manifest_parser import Manifest
from tiz.sandbox_dirs import TIZ_COMMIT_AUTHOR_EMAIL, TIZ_COMMIT_AUTHOR_NAME
from tiz.web_config_parser import EndpointConfig, parse_web_config

logger = get_logger(__name__)

_SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; img-src 'self' blob:; media-src 'self' blob:;",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": (
        "interest-cohort=(), accelerometer=(), autoplay=(), camera=(self), "
        "display-capture=(), document-domain=(), encrypted-media=(), "
        "fullscreen=(self), geolocation=(), gyroscope=(), magnetometer=(), "
        "microphone=(self), midi=(), payment=(), picture-in-picture=(), "
        "publickey-credentials-get=(), screen-wake-lock=(), sync-xhr=(self), "
        "usb=(), web-share=(), xr-spatial-tracking=()"
    ),
    "X-Robots-Tag": "noindex, nofollow",
    "X-DNS-Prefetch-Control": "off",
}


def _enrich_headers(
    headers: Headers
    | Mapping[str, str]
    | SupportsKeysAndGetItem
    | Iterable[tuple[str, str]]
    | None = None,
    request_headers: Headers | None = None,
) -> Headers:
    """Add security headers to the response headers dict."""
    result = Headers()
    if headers is not None:
        result.update(headers)
    for key, value in _SECURITY_HEADERS.items():
        if key in result:
            del result[key]
        result[key] = value
    # Reflect the Origin header for CORS, or use Host with X-Forwarded-Proto
    origin: str | None = None
    if request_headers is not None:
        origin = request_headers.get("Origin")
        if not origin:
            host = request_headers.get("Host")
            if host:
                scheme = request_headers.get("X-Forwarded-Proto", "http")
                origin = f"{scheme}://{host}"
    if origin:
        result["Access-Control-Allow-Origin"] = origin
    return result


def _set_async_keyboard_interrupt(thread: threading.Thread) -> None:
    """Raise KeyboardInterrupt in the given thread asynchronously."""
    tid = thread.ident
    if tid is None:
        return
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt)
    )
    if ret != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


class ChatWebSocketHandler:
    """Handles a single WebSocket connection with an InteractiveChat instance."""

    def __init__(
        self,
        manifest: Manifest,
        base_path: Path,
        context: dict[str, Any] | None = None,
        task_name: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.base_path = base_path
        self.context = context or {}
        self.task_name = task_name
        self._input_available = threading.Event()
        self._confirm_available = threading.Event()
        self._confirm_result: bool | None = None
        self._chat: InteractiveChat | None = None
        self._chat_thread: threading.Thread | None = None
        self._message_queue: deque[dict[str, Any]] = deque()
        self._queue_lock = threading.Lock()
        self._ws_send: Callable[[str], None] | None = None
        self._running = threading.Event()
        self._running.set()
        self._pending_save_callback: bool = False
        self._stop_sleep_event = threading.Event()
        self.manifest.meta.ephemeral_sandbox = True
        self.manifest.meta.delete_sandbox_on_exit = True
        logger.debug(
            "ChatWebSocketHandler created: task_name=%s, base_path=%s",
            task_name,
            base_path,
        )

    def _input_callback(self) -> dict[str, Any] | None:
        logger.debug(
            "_input_callback: waiting for input (running=%s)", self._running.is_set()
        )
        while self._running.is_set():
            with self._queue_lock:
                if self._message_queue:
                    msg = self._message_queue.popleft()
                    logger.debug("_input_callback: popped message: %s", msg)
                    return msg
            if not self._input_available.wait(timeout=0.1):
                continue
            self._input_available.clear()
        logger.debug("_input_callback: stopped, returning None")
        return None

    def _update_callback(self, msg: dict[str, Any], task_name: str | None) -> None:
        logger.debug(
            "_update_callback: pending_save=%s, task_name=%s",
            self._pending_save_callback,
            task_name,
        )
        with self._queue_lock:
            pending = self._pending_save_callback
        if pending:
            feedback = msg.get("tiz-internal", {}).get("save_conv", "")
            if feedback:
                try:
                    save_msg = {
                        "type": "save_result",
                        "contents": feedback,
                    }
                    if self._ws_send:
                        logger.debug("_update_callback: sending save_result")
                        self._ws_send(json.dumps(save_msg))
                except Exception:
                    logger.exception("_update_callback: failed to send save_result")
                with self._queue_lock:
                    self._pending_save_callback = False

        if self._ws_send:
            data = json.dumps({"type": "update", "data": msg, "task_name": task_name})
            logger.debug("_update_callback: sending update to ws")
            self._ws_send(data)

    def _confirm_callback(
        self,
        confirmation: dict[str, Any],
        fmt: Callable[[dict[str, Any], bool], str | None] | None,
        task_name: str | None,
    ) -> bool:
        logger.debug(
            "_confirm_callback: awaiting confirmation, task_name=%s", task_name
        )
        if self._ws_send:
            data = json.dumps(
                {
                    "type": "confirm",
                    "data": confirmation,
                    "fmt_serialized": fmt(confirmation.get("arguments", {}), True)
                    if fmt is not None
                    else None,
                    "task_name": task_name,
                }
            )
            self._ws_send(data)
        while self._running.is_set():
            with self._queue_lock:
                if self._confirm_result is not None:
                    result = self._confirm_result
                    self._confirm_result = None
                    logger.debug("_confirm_callback: got result=%s", result)
                    return result
            if not self._confirm_available.wait(timeout=0.1):
                continue
            self._confirm_available.clear()
        logger.debug("_confirm_callback: stopped, returning False")
        return False

    def run_in_thread(self, ws_send: Callable[[str], None]) -> threading.Thread:
        self._ws_send = ws_send
        thread = threading.Thread(
            target=self._run, daemon=True, name=f"chat-{self.task_name}"
        )
        self._chat_thread = thread
        logger.debug("run_in_thread: starting chat thread")
        thread.start()
        return thread

    def _run(self) -> None:
        logger.debug("_run: starting chat thread (task_name=%s)", self.task_name)
        backoff = 0.0
        max_backoff = 60.0
        last_attempt_time = 0.0
        while self._running.is_set():
            try:
                self._chat = InteractiveChat(
                    manifest=self.manifest,
                    base_path=self.base_path,
                    task_name=self.task_name,
                    update_callback=self._update_callback,
                    input_callback=self._input_callback,
                    context=self.context,
                    confirm_callback=self._confirm_callback,
                    in_band_files=True,
                    enable_help=False,
                    no_exit_on_kbdint=True,
                )
                self._chat.run()
                backoff = 0.0
            except Exception:
                logger.exception("Chat error")
                logger.debug("_run: chat finished, cleaning up")
            finally:
                self._chat_thread = None
                logger.debug("_run: chat thread cleaned up")
            if not self._running.is_set():
                break
            now = time.monotonic()
            if last_attempt_time > 0 and (now - last_attempt_time) > 10 * max_backoff:
                backoff = 0.0
            last_attempt_time = now
            if backoff > 0:
                delay = min(backoff, max_backoff)
                logger.debug("_run: backing off for %.1fs before retry", delay)
                self._stop_sleep_event.wait(timeout=delay)
                backoff = min(backoff * 2, max_backoff)
            else:
                backoff = 1.0
        self._input_available.set()
        self._confirm_available.set()

    def handle_message(self, message: dict[str, Any]) -> None:
        msg_type = message.get("type", "chat")
        logger.debug(
            "handle_message: type=%s, command=%s", msg_type, message.get("command", "")
        )
        if msg_type == "interrupt":
            self._interrupt()
            return
        if msg_type == "confirm_response":
            with self._queue_lock:
                self._confirm_result = message.get("confirm", False)
            self._confirm_available.set()
            return
        command = message.get("command", "")
        msg_text = message.get("message", "")
        contents = message.get("contents")
        files = message.get("files")

        if isinstance(files, list):
            for file_info in files:
                filename = file_info.get("name", "unknown")
                file_content = file_info.get("content", "")
                file_entry: dict[str, Any] = {
                    "command": "/attach",
                    "message": filename,
                    "contents": file_content,
                }
                with self._queue_lock:
                    self._message_queue.append(file_entry)
                    self._input_available.set()
            if command or msg_text:
                msg_entry: dict[str, Any] = {"command": command, "message": msg_text}
                with self._queue_lock:
                    self._message_queue.append(msg_entry)
                    self._input_available.set()
            return

        if contents is not None and command in ("/attach", "/load"):
            filename = Path(msg_text).name
            msg_text = filename

        entry: dict[str, Any] = {"command": command, "message": msg_text}
        if contents is not None:
            entry["contents"] = contents
        if command == "/save":
            filename = Path(msg_text).name
            entry["message"] = filename
            entry["contents"] = "1"
            with self._queue_lock:
                self._pending_save_callback = True

        with self._queue_lock:
            self._message_queue.append(entry)
            self._input_available.set()

    def _interrupt(self) -> None:
        if self._chat_thread is not None and self._chat_thread.is_alive():
            logger.debug("_interrupt: sending async interrupt to chat thread")
            _set_async_keyboard_interrupt(self._chat_thread)
        else:
            logger.debug("_interrupt: no active chat thread to interrupt")

    def close(self) -> None:
        logger.debug("close: closing handler")
        self._running.clear()
        self._stop_sleep_event.set()
        self._input_available.set()
        self._confirm_available.set()


def _parse_named_ws_path(path: str, endpoint_names: set[str]) -> tuple[str | None, str]:
    """Parse path like /{name}/ws or /base/{name}/ws -> (name, /ws) or (None, path).

    Works from any base prefix by matching known endpoint names.
    """
    # Try direct match first (root-level)
    parts = path.strip("/").split("/", 1)
    if len(parts) == 2 and parts[1] == "ws" and parts[0] in endpoint_names:
        return parts[0], "/ws"
    # Try with any prefix: /.../{name}/ws
    stripped = path.strip("/")
    for name in endpoint_names:
        if stripped.endswith(f"/{name}/ws"):
            return name, "/ws"
        if stripped == f"{name}/ws":  # pragma: no cover
            return name, "/ws"
    return None, path


def _guess_mime(suffix: str) -> str:
    mime_map = {
        ".html": "text/html",
        ".css": "text/css",
        ".js": "application/javascript",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".txt": "text/plain",
    }
    return mime_map.get(suffix, "application/octet-stream")


_MINIFIABLE_MIMES = frozenset(
    {
        "text/html",
        "text/css",
        "application/javascript",
        "image/svg+xml",
    }
)

_minify_cache: dict[str, bytes] = {}


def clear_minify_cache() -> None:
    """Clear the in-memory minify cache."""
    _minify_cache.clear()


def _minify_if_possible(content: bytes, content_type: str) -> bytes:
    """Minify HTML/CSS/SVG/JS content.

    Uses rjsmin for JavaScript and htmlmin for HTML/CSS/SVG.
    Falls back silently if the required library is unavailable.
    """
    if content_type not in _MINIFIABLE_MIMES:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    if content_type == "application/javascript":
        try:
            import rjsmin  # noqa: PLC0415

            minified: str = rjsmin.jsmin(text)
        except ImportError:
            return content
    else:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*cgi.*deprecated.*")
                import htmlmin  # noqa: PLC0415

            minified = htmlmin.minify(
                text,
                remove_comments=True,
                remove_empty_space=True,
                remove_all_empty_space=False,
                remove_optional_attribute_quotes=False,
            )
        except ImportError:
            return content
    return minified.encode("utf-8")


def _slugify(text: str) -> str:
    """Convert a string to a URL-friendly slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug.strip("-")


class App:
    """Application with multiple named WebSocket endpoints.

    Each endpoint is accessible via ws://host:port/../<name>/ws.
    Static files are served from a single config-level static_dir.
    """

    def __init__(self) -> None:
        self._endpoints: dict[str, EndpointConfig] = {}
        self._static_dir: Path | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_future: asyncio.Future[None] | None = None

    def add_endpoint(self, name: str, config: EndpointConfig) -> None:
        """Add a named WebSocket endpoint."""
        if name in self._endpoints:
            raise ValueError(f"Endpoint '{name}' already exists")
        self._endpoints[name] = config

    def get_endpoint(self, name: str) -> EndpointConfig | None:
        """Get an endpoint config by name."""
        return self._endpoints.get(name)

    @property
    def endpoint_names(self) -> list[str]:
        return list(self._endpoints)

    async def _handle_api_request(
        self, path: str, request_headers: Headers
    ) -> HTTPResponse | None:
        """Handle API requests. Returns HTTP response or None if not an API path.

        Only returns endpoints that would pass the _check_auth_headers check
        with the given request_headers.
        """
        if path.rstrip("/").endswith("/api/endpoints"):
            logger.debug("_handle_api_request: serving endpoints list for %s", path)
            endpoints_list = []
            for name, config in self._endpoints.items():
                if self._check_auth_headers(config.auth_headers, request_headers):
                    endpoints_list.append(
                        {
                            "name": name,
                            "websocket": f"/{name}/ws",
                            "description": config.description,
                            "suggestions": config.suggestions,
                        }
                    )
            body = json.dumps({"endpoints": endpoints_list}).encode()
            headers = {"Content-Type": "application/json"}
            return HTTPStatus.OK, headers, body

        return None

    async def _ws_handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle a single WebSocket connection."""
        parsed = urlparse(path)
        path_clean = parsed.path
        name, remaining = _parse_named_ws_path(path_clean, set(self._endpoints))
        if name is None or remaining != "/ws":
            logger.debug("_ws_handler: path %s not a valid ws path", path)
            return
        endpoint = self.get_endpoint(name)
        if endpoint is None:  # pragma: no cover (dict consistency guarantee)
            logger.debug("_ws_handler: endpoint '%s' not found", name)
            return
        if endpoint.manifest is None:
            logger.debug("_ws_handler: endpoint '%s' has no manifest", name)
            return

        logger.debug("_ws_handler: new connection for endpoint '%s'", name)
        handler = ChatWebSocketHandler(
            manifest=endpoint.manifest,
            base_path=endpoint.base_path,
            context=endpoint.context,
            task_name=endpoint.task_name,
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()

        def sync_send(data: str) -> None:
            if handler._running.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, data)

        thread = handler.run_in_thread(sync_send)

        async def send_worker() -> None:
            while handler._running.is_set():
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=0.1)
                    try:
                        await websocket.send(data)
                    except ConnectionClosed:
                        logger.debug("_ws_handler: send_worker: connection closed")
                        break
                except asyncio.TimeoutError:
                    continue

        send_task = asyncio.create_task(send_worker())

        try:
            async for message in websocket:
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    logger.debug("_ws_handler: invalid JSON received: %s", message)
                    continue
                if not isinstance(msg, dict):
                    logger.debug("_ws_handler: non-dict JSON received: %s", message)
                    continue
                handler.handle_message(msg)
        except ConnectionClosed:
            logger.debug("_ws_handler: recv connection closed")
        finally:
            logger.debug("_ws_handler: cleaning up connection for '%s'", name)
            handler.close()
            send_task.cancel()
            thread.join(timeout=2)
            logger.debug("_ws_handler: connection cleaned up for '%s'", name)

    @staticmethod
    def _check_auth_headers(
        auth_headers: dict[str, str] | None, request_headers: Headers
    ) -> bool:
        """Check if request headers match all required auth headers.

        Header names (dict keys) are matched case-insensitively.
        Header values are matched case-sensitively.
        Returns True if auth_headers is None/empty or all match.
        """
        if not auth_headers:
            return True
        for key, expected_value in auth_headers.items():
            actual_value: str | None = None
            for header_name, header_value in request_headers.raw_items():
                if header_name.lower() == key.lower():
                    actual_value = header_value
                    break
            if actual_value != expected_value:
                return False
        return True

    async def _process_request(
        self,
        path: str,
        headers: Headers,
    ) -> HTTPResponse | None:
        """Process HTTP requests for WebSocket upgrade, API, and static files."""
        parsed = urlparse(path)
        path_clean = parsed.path
        logger.debug("_process_request: path=%s", path_clean)

        name, remaining = _parse_named_ws_path(path_clean, set(self._endpoints))

        # Check auth headers for any endpoint-specific request
        endpoint: EndpointConfig | None = None
        if name is not None and remaining == "/ws":
            endpoint = self.get_endpoint(name)
        if endpoint is not None and not self._check_auth_headers(
            endpoint.auth_headers, headers
        ):
            logger.debug("_process_request: auth headers mismatch for '%s'", name)
            return HTTPStatus.FORBIDDEN, _enrich_headers({}, headers), b"Forbidden"

        # Check API routes first
        api_response = await self._handle_api_request(path_clean, headers)
        if api_response is not None:
            logger.debug("_process_request: returning API response for %s", path_clean)
            status, resp_headers, body = api_response
            return status, _enrich_headers(resp_headers, headers), body

        if name is not None and remaining == "/ws":
            if endpoint is not None:
                logger.debug("_process_request: allowing WS upgrade for %s", path_clean)
                return None  # Let websockets handle the WebSocket upgrade
            logger.debug(
                "_process_request: endpoint '%s' not found for WS path", name
            )  # pragma: no cover (dict consistency guarantee)
            return (
                HTTPStatus.NOT_FOUND,
                _enrich_headers({}, headers),
                b"Not Found",
            )  # pragma: no cover

        if self._static_dir is not None:
            file_path = path_clean.lstrip("/")
            logger.debug("_process_request: serving static file '%s'", file_path)
            status, static_headers, body = await _serve_static_async(
                self._static_dir, file_path
            )
            return status, _enrich_headers(static_headers, headers), body

        logger.debug("_process_request: 404 for %s", path_clean)
        return HTTPStatus.NOT_FOUND, _enrich_headers({}, headers), b"Not Found"

    def run(
        self,
        host: str = "localhost",
        port: int = 8080,
        path: str | None = None,
    ) -> None:
        """Run the WebSocket server.

        Args:
            host: Host to bind to (TCP mode).
            port: Port to bind to (TCP mode).
            path: Unix socket path (Unix mode). When provided, host and port are ignored.
        """
        if path is not None:
            logger.debug("run: starting server on unix socket %s", path)
            for name in self._endpoints:
                print(f"WebSocket endpoint: ws+unix://{path}/{name}/ws")
            if self._static_dir is not None:
                print(f"Serving static files from: {self._static_dir}")
            print(f"Server listening on unix:{path}")
        else:
            logger.debug("run: starting server on %s:%s", host, port)
            for name in self._endpoints:
                print(f"WebSocket endpoint: ws://{host}:{port}/{name}/ws")
            if self._static_dir is not None:
                print(f"Serving static files from: {self._static_dir}")
            print(f"Server listening on http://{host}:{port}")

        async def _run_server() -> None:
            self._loop = asyncio.get_running_loop()
            self._stop_future = asyncio.Future()
            if path is not None:
                async with serve(
                    self._ws_handler,
                    unix=True,
                    path=path,
                    process_request=self._process_request,
                    server_header=None,
                    max_size=100 * 1024 * 1024,
                ):
                    await self._stop_future
            else:
                async with serve(
                    self._ws_handler,
                    host,
                    port,
                    process_request=self._process_request,
                    server_header=None,
                    max_size=100 * 1024 * 1024,
                ):
                    await self._stop_future

        coro = _run_server()
        try:
            asyncio.run(coro)
        except KeyboardInterrupt:
            print("Shutting down...")
            logger.debug("run: keyboard interrupt received, shutting down")
        finally:
            coro.close()
            self._loop = None
            self._stop_future = None

    def stop(self) -> None:
        """Stop the server by resolving the stop future."""
        if (
            self._loop is not None
            and self._stop_future is not None
            and not self._stop_future.done()
        ):
            self._loop.call_soon_threadsafe(self._stop_future.set_result, None)


async def _serve_static_async(
    static_dir: Path,
    file_path: str,
) -> HTTPResponse:
    """Serve a static file asynchronously. Refuses to serve hidden files (names starting with '.').

    If the relevant minifier library is installed (rjsmin for JS, htmlmin for
    HTML/CSS/SVG), minifies files and caches the result in memory using the
    absolute path as key.
    """
    path = file_path.lstrip("/")
    if not path:
        path = "index.html"
    static_dir_resolved = static_dir.resolve()
    file_path_resolved = (static_dir_resolved / path).resolve()
    if file_path_resolved.is_file() and file_path_resolved.is_relative_to(
        static_dir_resolved
    ):
        rel = file_path_resolved.relative_to(static_dir)
        if any(part.startswith(".") for part in rel.parts):
            return HTTPStatus.NOT_FOUND, {"Content-Type": "text/plain"}, b"Not Found"
        content_type = _guess_mime(file_path_resolved.suffix)
        abs_path = str(file_path_resolved)
        cached = _minify_cache.get(abs_path)
        if cached is not None:
            body = cached
        else:
            with file_path_resolved.open("rb") as f:
                body = f.read()
            body = _minify_if_possible(body, content_type)
            _minify_cache[abs_path] = body
        return HTTPStatus.OK, {"Content-Type": content_type}, body
    return HTTPStatus.NOT_FOUND, {"Content-Type": "text/plain"}, b"Not Found"


def run_simple(
    base_path: str | Path,
    config_path: str | Path,
    host: str = "localhost",
    port: int = 8080,
    path: str | None = None,
    _app_holder: list[App] | None = None,
) -> None:
    """Run a simple HTTP server for testing.

    Args:
        base_path: Base path for resolving manifest paths.
        config_path: Path to the WebConfig YAML file.
        host: Host to bind to (TCP mode).
        port: Port to bind to (TCP mode).
        path: Unix socket path (Unix mode). When provided, host and port are ignored.
    """
    config_path = Path(config_path)
    base_path = Path(base_path)

    default_options: dict[str, Any] = {}
    default_options["meta"] = {
        "version": "0",
        "parallelism": 1,
        "committer_name": TIZ_COMMIT_AUTHOR_NAME,
        "committer_email": TIZ_COMMIT_AUTHOR_EMAIL,
        "use_host_timezone": True,
        "save_full_logs": False,
        "save_full_toolcalls": False,
        "save_full_usage_details": False,
        "summarizer_context_ratio": 0.9,
        "verbosity": 0,
        "ring_bell": False,
        "delete_sandbox_on_exit": True,
        "ephemeral_sandbox": True,
    }

    try:
        config = parse_web_config(
            base_path=base_path,
            path=config_path,
            default_options=default_options,
        )
    except Exception as exc:
        logger.error("Failed to parse web config: %s", _format_exc(exc))
        return

    app = App()
    if _app_holder is not None:
        _app_holder.append(app)
    app._static_dir = config.static_dir
    for endpoint_name, ep_config in config.endpoints.items():
        name = _slugify(endpoint_name)
        app.add_endpoint(name=name, config=ep_config)
    try:
        app.run(host=host, port=port, path=path)
    except Exception as exc:
        logger.error("Failed to run server: %s", _format_exc(exc))
