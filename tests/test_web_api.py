"""Tests for web_api module."""

from __future__ import annotations

import asyncio
import json
import threading
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from websockets.datastructures import Headers

from tiz.manifest_parser import Manifest, ManifestMeta, TaskSpec
from tiz.web_api import (
    _SECURITY_HEADERS,
    App,
    ChatWebSocketHandler,
    _enrich_headers,
    _guess_mime,
    _minify_if_possible,
    _parse_named_ws_path,
    _serve_static_async,
    _set_async_keyboard_interrupt,
    _slugify,
    clear_minify_cache,
)
from tiz.web_config_parser import EndpointConfig


class TestEnrichHeaders:
    """Test the _enrich_headers function for security headers."""

    def test_adds_all_security_headers(self) -> None:
        """Every security header defined in _SECURITY_HEADERS must be present."""
        result = _enrich_headers()
        for key, value in _SECURITY_HEADERS.items():
            assert result.get(key) == value, (
                f"Missing or wrong value for header {key}: "
                f"expected {value!r}, got {result.get(key)!r}"
            )

    def test_no_duplicate_security_headers(self) -> None:
        """Even if input headers contain a security header, it should not be duplicated."""
        input_headers: dict[str, str] = {
            "Content-Security-Policy": "default-src 'self'",
            "X-Custom": "value",
        }
        result = _enrich_headers(input_headers)
        assert (
            result.get("Content-Security-Policy")
            == _SECURITY_HEADERS["Content-Security-Policy"]
        ), "Security header was not overwritten to canonical value"
        assert result.get("X-Custom") == "value"

    def test_preserves_input_headers(self) -> None:
        """Non-security headers passed in should be preserved."""
        input_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }
        result = _enrich_headers(input_headers)
        assert result.get("Content-Type") == "application/json"
        assert result.get("Cache-Control") == "no-cache"

    def test_empty_input_headers(self) -> None:
        """Passing an empty dict should still produce all security headers."""
        result = _enrich_headers({})
        assert len(result) == len(_SECURITY_HEADERS)
        for key in _SECURITY_HEADERS:
            assert key in result

    def test_none_headers(self) -> None:
        """Passing None should produce all security headers."""
        result = _enrich_headers(None)
        for key in _SECURITY_HEADERS:
            assert key in result

    def test_reflects_origin(self) -> None:
        """When request_headers contains an Origin, it should be reflected as CORS."""
        request_headers = Headers()
        request_headers["Origin"] = "https://example.com"
        result = _enrich_headers({}, request_headers)
        assert result.get("Access-Control-Allow-Origin") == "https://example.com"

    def test_origin_fallback_to_host(self) -> None:
        """When no Origin but Host is present, fall back to http://host."""
        request_headers = Headers()
        request_headers["Host"] = "myserver.local"
        result = _enrich_headers({}, request_headers)
        assert result.get("Access-Control-Allow-Origin") == "http://myserver.local"

    def test_origin_fallback_to_host_with_port(self) -> None:
        """Host with port should be used directly."""
        request_headers = Headers()
        request_headers["Host"] = "myserver.local:8080"
        result = _enrich_headers({}, request_headers)
        assert result.get("Access-Control-Allow-Origin") == "http://myserver.local:8080"

    def test_no_origin_no_host(self) -> None:
        """When neither Origin nor Host is present, no CORS header."""
        result = _enrich_headers({})
        assert result.get("Access-Control-Allow-Origin") is None

    def test_request_headers_is_none(self) -> None:
        """When request_headers is None, no CORS header."""
        result = _enrich_headers({}, None)
        assert result.get("Access-Control-Allow-Origin") is None

    def test_x_forwarded_proto_https_with_host(self) -> None:
        """When X-Forwarded-Proto is https and Host is present, use https scheme."""
        request_headers = Headers()
        request_headers["Host"] = "myserver.local:8080"
        request_headers["X-Forwarded-Proto"] = "https"
        result = _enrich_headers({}, request_headers)
        assert (
            result.get("Access-Control-Allow-Origin") == "https://myserver.local:8080"
        )

    def test_x_forwarded_proto_https_with_host_no_port(self) -> None:
        """When X-Forwarded-Proto is https and Host has no port, use https with :443."""
        request_headers = Headers()
        request_headers["Host"] = "myserver.local"
        request_headers["X-Forwarded-Proto"] = "https"
        result = _enrich_headers({}, request_headers)
        assert result.get("Access-Control-Allow-Origin") == "https://myserver.local"

    def test_x_forwarded_proto_overrides_default_http(self) -> None:
        """X-Forwarded-Proto should override the default http scheme."""
        request_headers = Headers()
        request_headers["Host"] = "example.com"
        request_headers["X-Forwarded-Proto"] = "https"
        result = _enrich_headers({}, request_headers)
        assert result.get("Access-Control-Allow-Origin") == "https://example.com"

    def test_correct_header_values(self) -> None:
        """Verify the values of all security headers."""
        result = _enrich_headers()
        for key, expected in _SECURITY_HEADERS.items():
            actual = result.get(key)
            assert actual == expected, (
                f"Header {key}: expected {expected!r}, got {actual!r}"
            )

    def test_result_is_headers_instance(self) -> None:
        """The return type should be websockets Headers."""
        result = _enrich_headers()
        assert isinstance(result, Headers)

    def test_security_headers_case_sensitive(self) -> None:
        """Security headers should be present with proper casing."""
        result = _enrich_headers()
        for key in _SECURITY_HEADERS:
            assert result.get(key) is not None

    def test_enrich_headers_different_casing_in_input(self) -> None:
        """When input headers contain a security key with different casing,
        the enrichment should still replace it with the canonical casing
        without duplicating the header."""
        input_headers: dict[str, str] = {
            "content-security-policy": "default-src 'none'",
            "x-content-type-options": "nosniff",
        }
        result = _enrich_headers(input_headers)
        # Check canonical casing has the correct value
        assert (
            result.get("Content-Security-Policy")
            == _SECURITY_HEADERS["Content-Security-Policy"]
        )
        assert (
            result.get("X-Content-Type-Options")
            == _SECURITY_HEADERS["X-Content-Type-Options"]
        )
        # Verify there's only one key for Content-Security-Policy (no duplicate)
        keys_lower = [k.lower() for k in result]
        assert keys_lower.count("content-security-policy") == 1
        assert keys_lower.count("x-content-type-options") == 1


class TestSecurityHeadersNoDuplication:
    """Test that security headers are NOT added twice in the request flow."""

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        return asyncio.run(coro)

    def test_static_file_headers_not_doubled(self, tmp_path: Path) -> None:
        """Static file should return plain dict headers."""
        clear_minify_cache()
        html_content = "<html><body>Hello</body></html>"
        html_dir = tmp_path / "static"
        html_dir.mkdir()
        html_file = html_dir / "test.html"
        html_file.write_text(html_content)

        status, headers, body = self._run_async(
            _serve_static_async(html_dir, "/test.html")
        )
        assert status == HTTPStatus.OK
        assert isinstance(headers, dict)
        assert headers.get("Content-Type") == "text/html"
        assert "Content-Security-Policy" not in headers

    def test_static_file_headers_are_properly_set(self, tmp_path: Path) -> None:
        """Enriching the result of _serve_static_async should give all security headers."""
        clear_minify_cache()
        html_content = "<html><body>Hello</body></html>"
        html_dir = tmp_path / "static"
        html_dir.mkdir()
        html_file = html_dir / "test.html"
        html_file.write_text(html_content)

        status, headers, body = self._run_async(
            _serve_static_async(html_dir, "/test.html")
        )
        assert status == HTTPStatus.OK
        enriched = _enrich_headers(headers)
        for key, value in _SECURITY_HEADERS.items():
            assert enriched.get(key) == value, (
                f"Missing security header {key} after enrichment"
            )

    def test_static_404_headers_not_doubled(self, tmp_path: Path) -> None:
        """404 from static serving returns plain dict, enrichment happens once."""
        clear_minify_cache()
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        status, headers, body = self._run_async(
            _serve_static_async(static_dir, "/nonexistent.html")
        )
        assert status == HTTPStatus.NOT_FOUND
        enriched = _enrich_headers(headers)
        header_keys = list(enriched)
        for key in _SECURITY_HEADERS:
            occurrences = [k for k in header_keys if k == key.lower()]
            assert len(occurrences) == 1, (
                f"Security header {key} appears {len(occurrences)} times after "
                f"enrichment, expected 1. All keys: {header_keys}"
            )

    def test_enrich_headers_idempotent(self) -> None:
        """Calling _enrich_headers twice should produce the same result."""
        first = _enrich_headers({"Content-Type": "text/html"})
        second = _enrich_headers(dict(first))
        assert dict(first) == dict(second)

    def test_double_enrich_does_not_duplicate(self) -> None:
        """Running _enrich_headers on an already-enriched Headers object
        should not duplicate security headers."""
        first = _enrich_headers({"X-Foo": "bar"})
        second = _enrich_headers(dict(first))
        header_keys_lower = [k.lower() for k in second]
        for key in _SECURITY_HEADERS:
            occurrences = [k for k in header_keys_lower if k == key.lower()]
            assert len(occurrences) == 1, (
                f"Security header {key} appears {len(occurrences)} times "
                f"after double enrichment, expected 1. All keys: {list(second)}"
            )
        for key in _SECURITY_HEADERS:
            value = second.get(key)
            assert value == _SECURITY_HEADERS[key], (
                f"Security header {key} has wrong value after double enrichment: "
                f"{value!r} != {_SECURITY_HEADERS[key]!r}"
            )

    def test_enrich_called_with_request_headers_static_200(
        self, tmp_path: Path
    ) -> None:
        """_process_request passes request headers when serving a static file."""
        clear_minify_cache()
        html_dir = tmp_path / "static"
        html_dir.mkdir()
        html_file = html_dir / "index.html"
        html_file.write_text("<html><body>Hello</body></html>")

        app = App()
        app._static_dir = html_dir

        request_headers = Headers()
        request_headers["Origin"] = "https://example.com"
        request_headers["Host"] = "example.com"

        status, resp_headers, body = self._run_async(
            app._process_request("/index.html", request_headers)
        )
        assert status == HTTPStatus.OK
        assert isinstance(resp_headers, Headers)
        assert resp_headers.get("Access-Control-Allow-Origin") == "https://example.com"

    def test_enrich_called_with_request_headers_static_404(self) -> None:
        """_process_request passes request headers even on static 404."""
        app = App()
        request_headers = Headers()
        request_headers["Origin"] = "https://example.com"

        status, resp_headers, body = self._run_async(
            app._process_request("/nonexistent.html", request_headers)
        )
        assert status == HTTPStatus.NOT_FOUND
        assert isinstance(resp_headers, Headers)
        assert resp_headers.get("Access-Control-Allow-Origin") == "https://example.com"

    def test_enrich_called_with_request_headers_api_endpoints(self) -> None:
        """_process_request passes request headers for the API endpoint."""
        app = App()
        request_headers = Headers()
        request_headers["Origin"] = "https://api-test.com"

        status, resp_headers, body = self._run_async(
            app._process_request("/api/endpoints", request_headers)
        )
        assert status == HTTPStatus.OK
        assert isinstance(resp_headers, Headers)
        assert resp_headers.get("Access-Control-Allow-Origin") == "https://api-test.com"

    def test_enrich_called_with_request_headers_ws_path_not_found(self) -> None:
        """_process_request passes request headers for a WS path with no matching endpoint."""
        app = App()
        request_headers = Headers()
        request_headers["Origin"] = "https://ws-test.com"

        status, resp_headers, body = self._run_async(
            app._process_request("/nonexistent/ws", request_headers)
        )
        assert status == HTTPStatus.NOT_FOUND
        assert isinstance(resp_headers, Headers)
        assert resp_headers.get("Access-Control-Allow-Origin") == "https://ws-test.com"

    def test_enrich_called_with_request_headers_generic_404(self) -> None:
        """_process_request passes request headers for a completely unknown path."""
        app = App()
        request_headers = Headers()
        request_headers["Origin"] = "https://generic-test.com"

        status, resp_headers, body = self._run_async(
            app._process_request("/some/random/path", request_headers)
        )
        assert status == HTTPStatus.NOT_FOUND
        assert isinstance(resp_headers, Headers)
        assert (
            resp_headers.get("Access-Control-Allow-Origin")
            == "https://generic-test.com"
        )

    def test_enrich_called_with_no_request_headers_no_cors(
        self, tmp_path: Path
    ) -> None:
        """When no request headers provided, no CORS header in response."""
        clear_minify_cache()
        html_dir = tmp_path / "static"
        html_dir.mkdir()
        html_file = html_dir / "index.html"
        html_file.write_text("<html><body>Hello</body></html>")

        app = App()
        app._static_dir = html_dir

        status, resp_headers, body = self._run_async(
            app._process_request("/index.html", Headers())
        )
        assert status == HTTPStatus.OK
        assert isinstance(resp_headers, Headers)
        assert resp_headers.get("Access-Control-Allow-Origin") is None


class TestMinify:
    """Tests for the _minify_if_possible function."""

    def test_non_minifiable_mime_passthrough(self) -> None:
        """Content with a non-minifiable MIME type should be returned unchanged."""
        content = b"any content here"
        result = _minify_if_possible(content, "application/json")
        assert result == content

    def test_html_minification_with_htmlmin(self) -> None:
        """HTML content should be minified via htmlmin if available."""
        content = b"<html>\n<head>\n  <title>Test</title>\n</head>\n<body>\n  <p>Hello</p>\n</body>\n</html>"
        result = _minify_if_possible(content, "text/html")
        assert len(result) < len(content)
        assert b"<title>Test</title>" in result
        assert b"Hello" in result

    def test_css_minification(self) -> None:
        """CSS content should be minified via htmlmin if available."""
        content = b"body {\n  color: red;\n  margin: 0;\n}\n"
        result = _minify_if_possible(content, "text/css")
        assert len(result) < len(content)
        assert b"color: red" in result

    def test_js_minification_with_rjsmin(self) -> None:
        """JS content should be minified via rjsmin if available."""
        content = b"function hello() {\n  console.log('hello world');\n}\n"
        result = _minify_if_possible(content, "application/javascript")
        assert len(result) < len(content)
        result_str = result.decode("utf-8")
        assert "function hello()" in result_str
        assert "console.log" in result_str

    def test_js_minification_removes_comments(self) -> None:
        """rjsmin should strip JS comments."""
        content = b"// this is a comment\nvar x = 1;\n/* block comment */\nvar y = 2;\n"
        result = _minify_if_possible(content, "application/javascript")
        result_str = result.decode("utf-8")
        assert "var x=1" in result_str or "var x = 1" in result_str
        assert "var y=2" in result_str or "var y = 2" in result_str

    def test_svg_minification(self) -> None:
        """SVG content should be minified via htmlmin if available."""
        content = b'<svg xmlns="http://www.w3.org/2000/svg">\n  <circle cx="50" cy="50" r="40" />\n</svg>'
        result = _minify_if_possible(content, "image/svg+xml")
        assert len(result) < len(content)
        assert b"<circle" in result

    def test_minify_cache_used(self) -> None:
        """Minified content should be cached in _minify_cache."""
        clear_minify_cache()
        content = b"<p>hello</p>"
        result = _minify_if_possible(content, "text/html")
        assert result is not None

    def test_cache_in_serve_static(self, tmp_path: Path) -> None:
        """_serve_static_async should cache minified results."""
        from tiz.web_api import _minify_cache

        clear_minify_cache()
        static_dir = tmp_path / "static_js"
        static_dir.mkdir()
        js_file = static_dir / "script.js"
        js_file.write_text("function foo() {\n  return 1;\n}\n")

        status, headers, body = asyncio.run(
            _serve_static_async(static_dir, "/script.js")
        )
        assert status == HTTPStatus.OK
        assert isinstance(headers, dict)
        assert headers.get("Content-Type") == "application/javascript"
        assert len(_minify_cache) == 1

        status2, headers2, body2 = asyncio.run(
            _serve_static_async(static_dir, "/script.js")
        )
        assert status2 == HTTPStatus.OK
        assert body2 == body

    def test_clear_cache(self) -> None:
        """clear_minify_cache should empty the cache."""
        from tiz.web_api import _minify_cache

        clear_minify_cache()
        assert len(_minify_cache) == 0
        _minify_cache["/test"] = b"cached"
        assert len(_minify_cache) == 1
        clear_minify_cache()
        assert len(_minify_cache) == 0

    def test_minify_html_preserves_content(self) -> None:
        """Minified HTML should retain all meaningful content."""
        content = b"<html><body><p>Hello World</p><a href='test'>Link</a></body></html>"
        result = _minify_if_possible(content, "text/html")
        assert b"Hello World" in result
        assert b"Link" in result

    def test_minify_js_preserves_content(self) -> None:
        """Minified JS should retain all meaningful code."""
        content = b"var msg = 'hello world'; console.log(msg);"
        result = _minify_if_possible(content, "application/javascript")
        result_str = result.decode("utf-8")
        assert "hello world" in result_str
        assert "console.log" in result_str

    def test_minify_js_empty_string(self) -> None:
        """Empty JS content should be handled."""
        content = b""
        result = _minify_if_possible(content, "application/javascript")
        assert result == b""

    def test_minify_html_empty_string(self) -> None:
        """Empty HTML content should be handled."""
        content = b""
        result = _minify_if_possible(content, "text/html")
        assert result == b""

    def test_minify_invalid_utf8_returns_original(self) -> None:
        """Invalid UTF-8 content with a minifiable MIME type should return the original bytes."""
        content = b"\xff\xfe\x00\xff"
        result = _minify_if_possible(content, "text/html")
        assert result == content

        content2 = b"\x80\x81\x82"
        result2 = _minify_if_possible(content2, "application/javascript")
        assert result2 == content2

        content3 = b"\xff\xfe"
        result3 = _minify_if_possible(content3, "image/svg+xml")
        assert result3 == content3

        content4 = b"\x00\x01\x02"
        result4 = _minify_if_possible(content4, "text/css")
        assert result4 == content4

    def test_minify_js_unicode_content(self) -> None:
        """JS content with Unicode characters should be minified correctly."""
        content = (
            "function greet(name) {\n  console.log('こんにちは ' + name);\n}\n".encode()
        )
        result = _minify_if_possible(content, "application/javascript")
        result_str = result.decode("utf-8")
        assert "greet" in result_str
        assert "こんにちは" in result_str

    def test_minify_preserves_js_semicolons(self) -> None:
        """rjsmin preserves necessary semicolons in JS."""
        content = b"var a = 1; var b = 2; var c = 3;"
        result = _minify_if_possible(content, "application/javascript")
        result_str = result.decode("utf-8")
        assert "var a=1" in result_str or "var a = 1" in result_str
        assert "var c=3" in result_str or "var c = 3" in result_str

    def test_file_not_found_not_cached(self, tmp_path: Path) -> None:
        """Non-existent files should not be cached."""
        from tiz.web_api import _minify_cache

        clear_minify_cache()
        static_dir = tmp_path / "static_empty"
        static_dir.mkdir()

        status, headers, body = asyncio.run(
            _serve_static_async(static_dir, "/nonexistent.js")
        )
        assert status == HTTPStatus.NOT_FOUND
        assert len(_minify_cache) == 0

    def test_js_import_error_passthrough(self) -> None:
        """When rjsmin is not available, JS content passes through unchanged."""
        import builtins
        import sys as _sys

        saved_rjsmin = _sys.modules.pop("rjsmin", None)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rjsmin":
                raise ImportError("no rjsmin")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import
            content = b"var x = 1;\n"
            result = _minify_if_possible(content, "application/javascript")
            assert result == content
        finally:
            builtins.__import__ = original_import
            if saved_rjsmin is not None:
                _sys.modules["rjsmin"] = saved_rjsmin

    def test_html_minify_import_error(self) -> None:
        """When htmlmin is not available, HTML content passes through."""
        import builtins
        import sys as _sys

        saved_htmlmin = _sys.modules.pop("htmlmin", None)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "htmlmin":
                raise ImportError("no htmlmin")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import
            content = b"<p>hello</p>"
            result = _minify_if_possible(content, "text/html")
            assert result == content
        finally:
            builtins.__import__ = original_import
            if saved_htmlmin is not None:
                _sys.modules["htmlmin"] = saved_htmlmin


class TestAuthHeaders:
    """Tests for the auth header checking in _process_request."""

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_check_auth_headers_none(self) -> None:
        """When auth_headers is None, check should pass."""
        headers = Headers({"X-Api-Key": "secret"})
        assert App._check_auth_headers(None, headers) is True

    def test_check_auth_headers_empty_dict(self) -> None:
        """When auth_headers is empty dict, check should pass."""
        headers = Headers({"X-Api-Key": "secret"})
        assert App._check_auth_headers({}, headers) is True

    def test_check_auth_headers_single_match(self) -> None:
        """Single auth header present with correct value should pass."""
        headers = Headers({"X-Api-Key": "my-secret-key"})
        result = App._check_auth_headers({"X-Api-Key": "my-secret-key"}, headers)
        assert result is True

    def test_check_auth_headers_single_mismatch_value(self) -> None:
        """Single auth header present but wrong value should fail."""
        headers = Headers({"X-Api-Key": "wrong-key"})
        result = App._check_auth_headers({"X-Api-Key": "my-secret-key"}, headers)
        assert result is False

    def test_check_auth_headers_missing_header(self) -> None:
        """Required auth header not present should fail."""
        headers = Headers({"Content-Type": "application/json"})
        result = App._check_auth_headers({"X-Api-Key": "my-secret-key"}, headers)
        assert result is False

    def test_check_auth_headers_case_insensitive_key(self) -> None:
        """Header key matching should be case-insensitive."""
        headers = Headers({"x-api-key": "my-secret-key"})
        result = App._check_auth_headers({"X-Api-Key": "my-secret-key"}, headers)
        assert result is True

    def test_check_auth_headers_case_sensitive_value(self) -> None:
        """Header value matching should be case-sensitive."""
        headers = Headers({"X-Api-Key": "My-Secret-Key"})
        result = App._check_auth_headers({"X-Api-Key": "my-secret-key"}, headers)
        assert result is False

    def test_check_auth_headers_multiple_match(self) -> None:
        """Multiple auth headers all matching should pass."""
        headers = Headers(
            {
                "X-Api-Key": "secret123",
                "X-Api-Version": "v2",
                "Authorization": "Bearer token",
            }
        )
        result = App._check_auth_headers(
            {"X-Api-Key": "secret123", "Authorization": "Bearer token"}, headers
        )
        assert result is True

    def test_check_auth_headers_multiple_one_missing(self) -> None:
        """Multiple auth headers with one missing should fail."""
        headers = Headers({"X-Api-Key": "secret123"})
        result = App._check_auth_headers(
            {"X-Api-Key": "secret123", "Authorization": "Bearer token"}, headers
        )
        assert result is False

    def test_check_auth_headers_multiple_one_wrong_value(self) -> None:
        """Multiple auth headers with one wrong value should fail."""
        headers = Headers({"X-Api-Key": "secret123", "Authorization": "wrong-token"})
        result = App._check_auth_headers(
            {"X-Api-Key": "secret123", "Authorization": "Bearer token"}, headers
        )
        assert result is False

    def test_check_auth_headers_case_insensitive_mixed_case(self) -> None:
        """Multiple headers with mixed case keys should all be matched."""
        headers = Headers({"X-API-KEY": "secret", "x-auth-token": "token123"})
        result = App._check_auth_headers(
            {"x-api-key": "secret", "X-Auth-Token": "token123"}, headers
        )
        assert result is True

    def test_auth_ws_upgrade_allowed(self) -> None:
        """WS upgrade should proceed when auth headers match."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({"X-Api-Key": "valid-key"})
        result = self._run_async(app._process_request("/test/ws", headers))
        assert result is None

    def test_auth_ws_upgrade_forbidden(self) -> None:
        """WS upgrade should be 403 when auth headers mismatch."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({"X-Api-Key": "wrong-key"})
        status, resp_headers, body = self._run_async(
            app._process_request("/test/ws", headers)
        )
        assert status == HTTPStatus.FORBIDDEN

    def test_auth_ws_upgrade_missing_header(self) -> None:
        """WS upgrade should be 403 when auth header is missing."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({"Content-Type": "text/plain"})
        status, resp_headers, body = self._run_async(
            app._process_request("/test/ws", headers)
        )
        assert status == HTTPStatus.FORBIDDEN

    def test_auth_ws_upgrade_no_auth_configured(self) -> None:
        """WS upgrade should proceed when no auth_headers configured."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({})
        result = self._run_async(app._process_request("/test/ws", headers))
        assert result is None

    def test_auth_multiple_endpoints_independent(self) -> None:
        """Auth headers for one endpoint should not affect another."""
        app = App()
        ep_with_auth = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        ep_no_auth = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={},
        )
        app.add_endpoint("protected", ep_with_auth)
        app.add_endpoint("public", ep_no_auth)

        headers = Headers({})
        result = self._run_async(app._process_request("/public/ws", headers))
        assert result is None

        status, resp_headers, body = self._run_async(
            app._process_request("/protected/ws", headers)
        )
        assert status == HTTPStatus.FORBIDDEN

    def test_auth_api_endpoints_not_affected(self) -> None:
        """API endpoints are always accessible; protected endpoints are filtered from the listing."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({})
        status, resp_headers, body = self._run_async(
            app._process_request("/api/endpoints", headers)
        )
        assert status == HTTPStatus.OK
        data = json.loads(body)
        assert len(data["endpoints"]) == 0

    def test_auth_api_filters_protected_endpoint_without_auth(self) -> None:
        """Protected endpoints are hidden from API list when auth headers are missing."""
        app = App()
        app.add_endpoint(
            "public",
            EndpointConfig(
                manifest=None,  # type: ignore[arg-type]
                base_path=Path("/tmp"),
                auth_headers={},
            ),
        )
        app.add_endpoint(
            "protected",
            EndpointConfig(
                manifest=None,  # type: ignore[arg-type]
                base_path=Path("/tmp"),
                auth_headers={"X-Api-Key": "secret"},
            ),
        )

        headers = Headers({})
        status, resp_headers, body = self._run_async(
            app._process_request("/api/endpoints", headers)
        )
        assert status == HTTPStatus.OK
        data = json.loads(body)
        names = {ep["name"] for ep in data["endpoints"]}
        assert names == {"public"}

    def test_auth_api_filters_protected_endpoint_with_auth(self) -> None:
        """Protected endpoints are visible when correct auth headers are provided."""
        app = App()
        app.add_endpoint(
            "public",
            EndpointConfig(
                manifest=None,  # type: ignore[arg-type]
                base_path=Path("/tmp"),
                auth_headers={},
            ),
        )
        app.add_endpoint(
            "protected",
            EndpointConfig(
                manifest=None,  # type: ignore[arg-type]
                base_path=Path("/tmp"),
                auth_headers={"X-Api-Key": "secret"},
            ),
        )

        headers = Headers({"X-Api-Key": "secret"})
        status, resp_headers, body = self._run_async(
            app._process_request("/api/endpoints", headers)
        )
        assert status == HTTPStatus.OK
        data = json.loads(body)
        names = {ep["name"] for ep in data["endpoints"]}
        assert names == {"public", "protected"}

    def test_auth_api_no_auth_needed_endpoints_always_visible(self) -> None:
        """Endpoints without auth_headers are always visible in the API listing."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers=None,  # type: ignore[arg-type]
        )
        app.add_endpoint("no_auth", ep_config)
        ep_config2 = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={},
        )
        app.add_endpoint("empty_auth", ep_config2)

        headers = Headers({})
        status, resp_headers, body = self._run_async(
            app._process_request("/api/endpoints", headers)
        )
        assert status == HTTPStatus.OK
        data = json.loads(body)
        names = {ep["name"] for ep in data["endpoints"]}
        assert names == {"no_auth", "empty_auth"}

    def test_auth_api_filters_by_multiple_headers(self) -> None:
        """Multiple auth headers all must match for endpoint to appear."""
        app = App()
        app.add_endpoint(
            "multi_auth",
            EndpointConfig(
                manifest=None,  # type: ignore[arg-type]
                base_path=Path("/tmp"),
                auth_headers={"X-Api-Key": "secret", "X-Api-Version": "v2"},
            ),
        )
        app.add_endpoint(
            "simple",
            EndpointConfig(
                manifest=None,  # type: ignore[arg-type]
                base_path=Path("/tmp"),
                auth_headers={"X-Api-Key": "secret"},
            ),
        )

        headers = Headers({"X-Api-Key": "secret"})
        status, resp_headers, body = self._run_async(
            app._process_request("/api/endpoints", headers)
        )
        assert status == HTTPStatus.OK
        data = json.loads(body)
        names = {ep["name"] for ep in data["endpoints"]}
        assert names == {"simple"}

        headers2 = Headers({"X-Api-Key": "secret", "X-Api-Version": "v2"})
        status2, resp_headers2, body2 = self._run_async(
            app._process_request("/api/endpoints", headers2)
        )
        assert status2 == HTTPStatus.OK
        data2 = json.loads(body2)
        names2 = {ep["name"] for ep in data2["endpoints"]}
        assert names2 == {"multi_auth", "simple"}

    def test_auth_forbidden_response_has_security_headers(self) -> None:
        """403 response should still have security headers."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({"X-Api-Key": "wrong"})
        status, resp_headers, body = self._run_async(
            app._process_request("/test/ws", headers)
        )
        assert status == HTTPStatus.FORBIDDEN
        assert isinstance(resp_headers, Headers)
        for key, value in _SECURITY_HEADERS.items():
            assert resp_headers.get(key) == value

    def test_auth_case_insensitive_header_name_in_request(self) -> None:
        """Auth header name in request should be case-insensitive."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({"x-api-key": "valid-key"})
        result = self._run_async(app._process_request("/test/ws", headers))
        assert result is None

    def test_auth_case_sensitive_header_value(self) -> None:
        """Auth header value should be case-sensitive."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "Valid-Key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({"X-Api-Key": "valid-key"})
        status, resp_headers, body = self._run_async(
            app._process_request("/test/ws", headers)
        )
        assert status == HTTPStatus.FORBIDDEN

    def test_auth_ws_endpoint_not_found(self) -> None:
        """Non-existent WS path should not trigger auth check."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({})
        status, resp_headers, body = self._run_async(
            app._process_request("/nonexistent/ws", headers)
        )
        assert status == HTTPStatus.NOT_FOUND

    def test_auth_static_files_not_affected(self) -> None:
        """Static files should be served without auth check."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
            auth_headers={"X-Api-Key": "valid-key"},
        )
        app.add_endpoint("test", ep_config)

        headers = Headers({})
        status, resp_headers, body = self._run_async(
            app._process_request("/some/random/path", headers)
        )
        assert status == HTTPStatus.NOT_FOUND

    def test_auth_ws_path_endpoint_not_found(self) -> None:
        """_process_request: WS path with valid name but endpoint removed/unavailable (lines 589-590)."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        # Make _parse_named_ws_path find the name but get_endpoint return None
        with patch.object(app, "get_endpoint", side_effect=lambda _: None):
            headers = Headers({})
            status, resp_headers, body = self._run_async(
                app._process_request("/test/ws", headers)
            )
            assert status == HTTPStatus.NOT_FOUND


class TestChatWebSocketHandlerRun:
    """Tests for the ChatWebSocketHandler._run loop and backoff logic."""

    def _make_manifest(self) -> Manifest:
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

    def test_run_success_exits_loop(self) -> None:
        """When _chat.run() succeeds, the loop runs once and exits."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        with patch(
            "tiz.web_api.InteractiveChat",
            autospec=True,
        ) as mock_ic:
            mock_chat_instance = mock_ic.return_value
            mock_chat_instance.run = MagicMock()

            def _run_and_stop() -> None:
                handler._running.clear()

            mock_chat_instance.run.side_effect = _run_and_stop
            handler._run()

        assert handler._chat is not None
        assert not handler._running.is_set()
        assert handler._chat_thread is None
        assert handler._input_available.is_set()
        assert handler._confirm_available.is_set()

    def test_run_exception_triggers_backoff(self) -> None:
        """When _chat.run() raises, the loop should backoff and retry."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        call_count = 0

        with patch(
            "tiz.web_api.InteractiveChat",
            autospec=True,
        ) as mock_ic:
            mock_chat_instance = mock_ic.return_value

            def _failing_run() -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 3:
                    handler._running.clear()
                raise RuntimeError("Test error")

            mock_chat_instance.run.side_effect = _failing_run
            sleeps: list[float] = []

            def _capture_wait(**kwargs) -> bool:
                timeout = kwargs.get("timeout")
                if timeout is not None:
                    sleeps.append(timeout)
                return False

            with patch.object(
                handler._stop_sleep_event, "wait", side_effect=_capture_wait
            ):
                handler._run()

        assert call_count == 3
        assert len(sleeps) == 1
        assert sleeps[0] == 1.0
        assert not handler._running.is_set()

    def test_backoff_exponential_increase(self) -> None:
        """Backoff should double each iteration."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        call_count = 0

        with patch(
            "tiz.web_api.InteractiveChat",
            autospec=True,
        ) as mock_ic:
            mock_chat_instance = mock_ic.return_value

            def _failing_run() -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 5:
                    handler._running.clear()
                raise RuntimeError("Test error")

            mock_chat_instance.run.side_effect = _failing_run
            sleeps: list[float] = []

            def _capture_wait(**kwargs) -> bool:
                timeout = kwargs.get("timeout")
                if timeout is not None:
                    sleeps.append(timeout)
                return False

            with patch.object(
                handler._stop_sleep_event, "wait", side_effect=_capture_wait
            ):
                handler._run()

        assert len(sleeps) == 3
        expected = [1.0, 2.0, 4.0]
        for i, expected_delay in enumerate(expected):
            assert sleeps[i] == expected_delay, (
                f"Sleep {i}: expected {expected_delay}, got {sleeps[i]}"
            )

    def test_backoff_capped_at_max(self) -> None:
        """Backoff should be capped at 60s and reset after reaching max."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        call_count = 0

        with patch(
            "tiz.web_api.InteractiveChat",
            autospec=True,
        ) as mock_ic:
            mock_chat_instance = mock_ic.return_value

            def _failing_run() -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 9:
                    handler._running.clear()
                raise RuntimeError("Test error")

            mock_chat_instance.run.side_effect = _failing_run
            sleeps: list[float] = []

            def _capture_wait(**kwargs) -> bool:
                timeout = kwargs.get("timeout")
                if timeout is not None:
                    sleeps.append(timeout)
                return False

            with patch.object(
                handler._stop_sleep_event, "wait", side_effect=_capture_wait
            ):
                handler._run()

        assert len(sleeps) == 7
        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]
        for i, expected_delay in enumerate(expected):
            assert sleeps[i] == expected_delay, (
                f"Sleep {i}: expected {expected_delay}, got {sleeps[i]}"
            )

    def test_backoff_resets_after_success(self) -> None:
        """Backoff should reset to 0 after a successful run."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        call_count = 0

        with patch(
            "tiz.web_api.InteractiveChat",
            autospec=True,
        ) as mock_ic:
            mock_chat_instance = mock_ic.return_value

            def _run_sequence() -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("First failure")
                handler._running.clear()

            mock_chat_instance.run.side_effect = _run_sequence
            sleeps: list[float] = []

            def _capture_wait(**kwargs) -> bool:
                timeout = kwargs.get("timeout")
                if timeout is not None:
                    sleeps.append(timeout)
                return False

            with patch.object(
                handler._stop_sleep_event, "wait", side_effect=_capture_wait
            ):
                handler._run()

        assert call_count == 2
        assert len(sleeps) == 0
        assert not handler._running.is_set()

    def test_running_false_exits_immediately(self) -> None:
        """When _running is False at loop start, the loop exits."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        handler._running.clear()

        with patch("tiz.web_api.InteractiveChat") as mock_ic:
            handler._run()
            mock_ic.assert_not_called()

        assert handler._input_available.is_set()
        assert handler._confirm_available.is_set()

    def test_loop_exits_when_running_set_false_during_backoff(self) -> None:
        """Even during backoff, the loop should exit if _running becomes False."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

        with patch(
            "tiz.web_api.InteractiveChat",
            autospec=True,
        ) as mock_ic:
            mock_chat_instance = mock_ic.return_value
            mock_chat_instance.run.side_effect = RuntimeError("Error")

            def _stop_after_wait(**_kwargs) -> bool:
                handler._running.clear()
                return False

            with patch.object(
                handler._stop_sleep_event, "wait", side_effect=_stop_after_wait
            ):
                handler._run()

        assert not handler._running.is_set()

    def test_backoff_resets_after_long_gap(self) -> None:
        """When time between attempts exceeds 10x max_backoff, backoff resets to 0."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        call_count = 0
        time_call_count: list[int] = [0]

        with (
            patch("tiz.web_api.InteractiveChat", autospec=True) as mock_ic,
            patch("tiz.web_api.logger"),
        ):
            mock_chat_instance = mock_ic.return_value

            def _failing_run() -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 6:
                    handler._running.clear()
                raise RuntimeError("Test error")

            mock_chat_instance.run.side_effect = _failing_run
            waits: list[float] = []

            def _capture_wait(**kwargs) -> bool:
                timeout = kwargs.get("timeout")
                if timeout is not None:
                    waits.append(timeout)
                return False

            def _time_side_effect() -> float:
                time_call_count[0] += 1
                if time_call_count[0] <= 2:
                    return 100.0
                return 800.0

            with (
                patch.object(
                    handler._stop_sleep_event, "wait", side_effect=_capture_wait
                ),
                patch("tiz.web_api.time.monotonic", side_effect=_time_side_effect),
            ):
                handler._run()

        assert call_count == 6
        assert len(waits) == 3
        assert waits[0] == 1.0
        assert waits[1] == 1.0
        assert waits[2] == 2.0
        assert not handler._running.is_set()


class TestSetAsyncKeyboardInterrupt:
    """Test the _set_async_keyboard_interrupt function."""

    def test_none_tid(self) -> None:
        """When thread.ident is None, function should return early."""
        thread = threading.Thread(target=lambda: None)
        # ident is None before thread.start()
        _set_async_keyboard_interrupt(thread)

    @patch("ctypes.pythonapi.PyThreadState_SetAsyncExc")
    def test_non_int_tid_raises_type_error(self, mock_set_async_exc: MagicMock) -> None:  # noqa: ARG002
        """When thread.ident is not an int, ctypes.c_ulong raises TypeError."""
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.ident = "not_an_int"  # type: ignore[assignment]
        with pytest.raises(TypeError):
            _set_async_keyboard_interrupt(mock_thread)

    @patch("ctypes.pythonapi.PyThreadState_SetAsyncExc")
    def test_ret_not_one(self, mock_set_async_exc: MagicMock) -> None:
        """When ctypes returns != 1, cleanup should be called."""
        mock_set_async_exc.return_value = 0
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.ident = 12345
        _set_async_keyboard_interrupt(mock_thread)
        assert mock_set_async_exc.call_count == 2
        # First call for the actual interrupt, second for cleanup

    @patch("ctypes.pythonapi.PyThreadState_SetAsyncExc")
    def test_ret_one(self, mock_set_async_exc: MagicMock) -> None:
        """When ctypes returns 1, no cleanup should happen."""
        mock_set_async_exc.return_value = 1
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.ident = 12345
        _set_async_keyboard_interrupt(mock_thread)
        mock_set_async_exc.assert_called_once()


class TestInputCallback:
    """Test the _input_callback method."""

    def _make_manifest(self) -> Manifest:
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

    def test_returns_message_when_queued(self) -> None:
        """When a message is in the queue, it should be returned immediately."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        with handler._queue_lock:
            handler._message_queue.append({"command": "", "message": "hello"})
            handler._input_available.set()

        result = handler._input_callback()
        assert result == {"command": "", "message": "hello"}
        assert len(handler._message_queue) == 0

    def test_returns_none_when_stopped(self) -> None:
        """When _running is False and queue is empty, return None."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        handler._running.clear()
        result = handler._input_callback()
        assert result is None

    def test_waits_for_message(self) -> None:
        """Should wait for input_available when queue is empty."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

        def _add_message_after_delay() -> None:
            import time

            time.sleep(0.05)
            with handler._queue_lock:
                handler._message_queue.append({"command": "", "message": "delayed"})
                handler._input_available.set()

        thread = threading.Thread(target=_add_message_after_delay, daemon=True)
        thread.start()
        result = handler._input_callback()
        assert result == {"command": "", "message": "delayed"}

    def test_multiple_messages_returned_in_order(self) -> None:
        """Messages should be returned in FIFO order."""
        manifest = self._make_manifest()
        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        with handler._queue_lock:
            handler._message_queue.append({"command": "", "message": "first"})
            handler._message_queue.append({"command": "", "message": "second"})
            handler._message_queue.append({"command": "", "message": "third"})
            handler._input_available.set()

        assert handler._input_callback() == {"command": "", "message": "first"}
        assert handler._input_callback() == {"command": "", "message": "second"}
        assert handler._input_callback() == {"command": "", "message": "third"}


class TestUpdateCallback:
    """Test the _update_callback method."""

    def _make_handler(self) -> ChatWebSocketHandler:
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
        return ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

    def test_sends_update_when_ws_send_set(self) -> None:
        """When ws_send is set, sends update message."""
        handler = self._make_handler()
        sent: list[str] = []

        def ws_send(data: str) -> None:
            sent.append(data)

        handler._ws_send = ws_send
        handler._update_callback({"type": "chat", "text": "hello"}, "test-task")

        assert len(sent) == 1
        data = json.loads(sent[0])
        assert data["type"] == "update"
        assert data["data"]["text"] == "hello"
        assert data["task_name"] == "test-task"

    def test_no_ws_send_no_error(self) -> None:
        """When ws_send is None, no error should occur."""
        handler = self._make_handler()
        handler._update_callback({"type": "chat", "text": "hello"}, None)

    def test_save_result_sent_before_update(self) -> None:
        """When pending_save_callback is True and save_conv is present, send save_result first."""
        handler = self._make_handler()
        sent: list[str] = []

        def ws_send(data: str) -> None:
            sent.append(data)

        handler._ws_send = ws_send
        handler._pending_save_callback = True
        handler._update_callback(
            {"tiz-internal": {"save_conv": "base64content"}}, "test-task"
        )

        assert len(sent) == 2
        first = json.loads(sent[0])
        assert first["type"] == "save_result"
        assert first["contents"] == "base64content"
        second = json.loads(sent[1])
        assert second["type"] == "update"
        assert handler._pending_save_callback is False

    def test_save_result_empty_feedback(self) -> None:
        """When save_conv is empty, no save_result is sent and pending stays True."""
        handler = self._make_handler()
        sent: list[str] = []

        def ws_send(data: str) -> None:
            sent.append(data)

        handler._ws_send = ws_send
        handler._pending_save_callback = True
        handler._update_callback({"tiz-internal": {"save_conv": ""}}, "test-task")

        assert len(sent) == 1  # Only update, no save_result
        assert json.loads(sent[0])["type"] == "update"
        assert (
            handler._pending_save_callback is True
        )  # Not cleared because feedback was falsy

    def test_save_result_exception_handled(self) -> None:
        """Exception during save_result send should be handled gracefully."""
        handler = self._make_handler()
        save_sent: list[bool] = []

        def ws_send(_data: str) -> None:
            save_sent.append(True)
            if len(save_sent) == 1:
                raise RuntimeError("Send failed on save")

        handler._ws_send = ws_send
        handler._pending_save_callback = True
        handler._update_callback(
            {"tiz-internal": {"save_conv": "base64content"}}, "test-task"
        )
        assert handler._pending_save_callback is False

    def test_save_result_no_ws_send(self) -> None:
        """When ws_send is None, save result should be skipped."""
        handler = self._make_handler()
        handler._pending_save_callback = True
        handler._update_callback(
            {"tiz-internal": {"save_conv": "base64content"}}, "test-task"
        )
        assert handler._pending_save_callback is False


class TestConfirmCallback:
    """Test the _confirm_callback method."""

    def _make_handler(self) -> ChatWebSocketHandler:
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
        return ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

    def test_returns_true_when_confirmed(self) -> None:
        """When confirm_result is set to True, return True."""
        handler = self._make_handler()
        sent: list[str] = []

        def ws_send(data: str) -> None:
            sent.append(data)

        handler._ws_send = ws_send
        with handler._queue_lock:
            handler._confirm_result = True
        handler._confirm_available.set()

        result = handler._confirm_callback({"action": "run_tool"}, None, "test-task")
        assert result is True
        assert len(sent) == 1
        msg = json.loads(sent[0])
        assert msg["type"] == "confirm"
        assert msg["data"] == {"action": "run_tool"}
        assert msg["fmt_serialized"] is None
        assert msg["task_name"] == "test-task"

    def test_returns_false_when_denied(self) -> None:
        """When confirm_result is set to False, return False."""
        handler = self._make_handler()
        sent: list[str] = []

        def ws_send(data: str) -> None:
            sent.append(data)

        handler._ws_send = ws_send
        with handler._queue_lock:
            handler._confirm_result = False
        handler._confirm_available.set()

        result = handler._confirm_callback({"action": "delete_file"}, None, "test-task")
        assert result is False
        assert len(sent) == 1
        msg = json.loads(sent[0])
        assert msg["type"] == "confirm"
        assert msg["data"] == {"action": "delete_file"}
        assert msg["fmt_serialized"] is None
        assert msg["task_name"] == "test-task"

    def test_returns_false_when_stopped(self) -> None:
        """When _running is False, return False."""
        handler = self._make_handler()

        def ws_send(data: str) -> None:
            pass

        handler._ws_send = ws_send
        handler._running.clear()

        result = handler._confirm_callback({"action": "run_tool"}, None, "test-task")
        assert result is False

    def test_no_ws_send_waits_for_confirm(self) -> None:
        """When ws_send is None, still waits for confirm result."""
        handler = self._make_handler()
        with handler._queue_lock:
            handler._confirm_result = True
        handler._confirm_available.set()

        result = handler._confirm_callback({"action": "run_tool"}, None, "test-task")
        assert result is True

    def test_sends_fmt_serialized(self) -> None:
        """The fmt_serialized field should contain str(fmt)."""
        handler = self._make_handler()
        sent: list[str] = []

        def ws_send(data: str) -> None:
            sent.append(data)

        handler._ws_send = ws_send
        with handler._queue_lock:
            handler._confirm_result = True
        handler._confirm_available.set()

        def fmt_func(data: dict, flag: bool) -> str | None:
            return f"formatted {data} {flag}"

        handler._confirm_callback(
            {"action": "run_tool", "arguments": {"a": "b"}}, fmt_func, "test-task"
        )
        assert len(sent) == 1
        msg = json.loads(sent[0])
        assert msg["fmt_serialized"] == "formatted {'a': 'b'} True"
        assert msg["type"] == "confirm"
        assert msg["data"] == {"action": "run_tool", "arguments": {"a": "b"}}
        assert msg["task_name"] == "test-task"


class TestHandleMessage:
    """Test the handle_message method."""

    def _make_handler(self) -> ChatWebSocketHandler:
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
        return ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

    def test_interrupt_message(self) -> None:
        """Interrupt message should call _interrupt."""
        handler = self._make_handler()
        with patch.object(handler, "_interrupt") as mock_interrupt:
            handler.handle_message({"type": "interrupt"})
            mock_interrupt.assert_called_once()

    def test_confirm_response_true(self) -> None:
        """Confirm response message should set confirm_result."""
        handler = self._make_handler()
        handler.handle_message({"type": "confirm_response", "confirm": True})
        assert handler._confirm_result is True

    def test_confirm_response_false(self) -> None:
        """Confirm response message should set confirm_result to False."""
        handler = self._make_handler()
        handler.handle_message({"type": "confirm_response", "confirm": False})
        assert handler._confirm_result is False

    def test_regular_message(self) -> None:
        """Regular message should be queued."""
        handler = self._make_handler()
        handler.handle_message({"command": "", "message": "hello"})
        with handler._queue_lock:
            assert len(handler._message_queue) == 1
            assert handler._message_queue[0]["message"] == "hello"

    def test_command_message(self) -> None:
        """Command message should be queued."""
        handler = self._make_handler()
        handler.handle_message({"command": "/help", "message": ""})
        with handler._queue_lock:
            assert len(handler._message_queue) == 1
            assert handler._message_queue[0]["command"] == "/help"

    def test_save_command(self) -> None:
        """Save command should set pending_save_callback and queue filename."""
        handler = self._make_handler()
        handler.handle_message({"command": "/save", "message": "chat.json"})
        assert handler._pending_save_callback is True
        with handler._queue_lock:
            assert handler._message_queue[0]["command"] == "/save"
            assert handler._message_queue[0]["message"] == "chat.json"
            assert handler._message_queue[0]["contents"] == "1"

    def test_attach_with_content(self) -> None:
        """Attach command with contents should queue with filename name."""
        handler = self._make_handler()
        handler.handle_message(
            {"command": "/attach", "message": "/path/to/file.txt", "contents": "base64"}
        )
        with handler._queue_lock:
            assert len(handler._message_queue) == 1
            assert handler._message_queue[0]["message"] == "file.txt"
            assert handler._message_queue[0]["contents"] == "base64"

    def test_load_with_content(self) -> None:
        """Load command with contents should queue with filename name."""
        handler = self._make_handler()
        handler.handle_message(
            {"command": "/load", "message": "/path/to/conv.json", "contents": "base64"}
        )
        with handler._queue_lock:
            assert len(handler._message_queue) == 1
            assert handler._message_queue[0]["message"] == "conv.json"
            assert handler._message_queue[0]["contents"] == "base64"

    def test_files_list(self) -> None:
        """Files list should queue attach messages for each file."""
        handler = self._make_handler()
        handler.handle_message(
            {
                "files": [
                    {"name": "file1.txt", "content": "Y29udGVudDE="},
                    {"name": "file2.txt", "content": "Y29udGVudDI="},
                ]
            }
        )
        with handler._queue_lock:
            assert len(handler._message_queue) == 2
            assert handler._message_queue[0]["command"] == "/attach"
            assert handler._message_queue[0]["message"] == "file1.txt"
            assert handler._message_queue[1]["command"] == "/attach"
            assert handler._message_queue[1]["message"] == "file2.txt"

    def test_files_with_command_and_message(self) -> None:
        """Files list with command and message should queue both file and text messages."""
        handler = self._make_handler()
        handler.handle_message(
            {
                "command": "",
                "message": "process these",
                "files": [
                    {"name": "data.csv", "content": "MSwyLDMK"},
                ],
            }
        )
        with handler._queue_lock:
            assert len(handler._message_queue) == 2
            assert handler._message_queue[0]["command"] == "/attach"
            assert handler._message_queue[1]["command"] == ""
            assert handler._message_queue[1]["message"] == "process these"


class TestInterrupt:
    """Test the _interrupt method."""

    def _make_handler(self) -> ChatWebSocketHandler:
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
        return ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

    def test_interrupt_active_thread(self) -> None:
        """When chat_thread is alive, interrupt should call _set_async_keyboard_interrupt."""
        handler = self._make_handler()
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = True
        handler._chat_thread = mock_thread

        with patch("tiz.web_api._set_async_keyboard_interrupt") as mock_interrupt:
            handler._interrupt()
            mock_interrupt.assert_called_once_with(mock_thread)

    def test_interrupt_no_thread(self) -> None:
        """When chat_thread is None, interrupt should not error."""
        handler = self._make_handler()
        handler._chat_thread = None
        handler._interrupt()

    def test_interrupt_dead_thread(self) -> None:
        """When chat_thread is not alive, interrupt should not call set_async."""
        handler = self._make_handler()
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = False
        handler._chat_thread = mock_thread

        with patch("tiz.web_api._set_async_keyboard_interrupt") as mock_interrupt:
            handler._interrupt()
            mock_interrupt.assert_not_called()


class TestParseNamedWsPath:
    """Test the _parse_named_ws_path function."""

    def test_valid_path(self) -> None:
        """A valid path like /chat/ws should return ('chat', '/ws')."""
        result = _parse_named_ws_path("/chat/ws", {"chat"})
        assert result == ("chat", "/ws")

    def test_path_with_extra_slash(self) -> None:
        """A path with extra slashes should be handled."""
        result = _parse_named_ws_path("//chat/ws", {"chat"})
        assert result == ("chat", "/ws")

    def test_just_slash(self) -> None:
        """Just '/' should return (None, '/').
        After strip('/'), parts == [''], len(parts) == 1 != 2, so falls through."""
        result = _parse_named_ws_path("/", {"chat"})
        assert result == (None, "/")

    def test_empty_string(self) -> None:
        """Empty string should return (None, '')."""
        result = _parse_named_ws_path("", set())
        assert result == (None, "")

    def test_non_ws_path(self) -> None:
        """A path without /ws suffix should return (None, path)."""
        result = _parse_named_ws_path("/chat/other", {"chat"})
        assert result == (None, "/chat/other")

    def test_only_ws_suffix(self) -> None:
        """A path like /ws should return (None, '/ws')."""
        result = _parse_named_ws_path("/ws", {"chat"})
        assert result == (None, "/ws")

    def test_deep_path(self) -> None:
        """A deeper path like /a/b/ws should return (None, path)."""
        result = _parse_named_ws_path("/a/b/ws", {"chat"})
        assert result == (None, "/a/b/ws")

    def test_with_base_prefix(self) -> None:
        """A path like /base/chat/ws should return ('chat', '/ws')."""
        result = _parse_named_ws_path("/base/chat/ws", {"chat"})
        assert result == ("chat", "/ws")

    def test_unknown_endpoint(self) -> None:
        """An unknown endpoint name should return (None, path)."""
        result = _parse_named_ws_path("/unknown/ws", {"chat"})
        assert result == (None, "/unknown/ws")

    def test_deep_base_prefix(self) -> None:
        """A deeper base path like /a/b/chat/ws should work."""
        result = _parse_named_ws_path("/a/b/chat/ws", {"chat"})
        assert result == ("chat", "/ws")

    def test_stripped_equals_name_ws(self) -> None:
        """When stripped path equals {name}/ws (no leading slash), line 337 is hit.
        This happens when endpoint name contains '/' so the split('/')[0] doesn't
        match the full name, and endswith check also misses."""
        result = _parse_named_ws_path("a/b/ws", {"a/b"})
        assert result == ("a/b", "/ws")


class TestGuessMime:
    """Test the _guess_mime function."""

    def test_html(self) -> None:
        assert _guess_mime(".html") == "text/html"

    def test_css(self) -> None:
        assert _guess_mime(".css") == "text/css"

    def test_js(self) -> None:
        assert _guess_mime(".js") == "application/javascript"

    def test_json(self) -> None:
        assert _guess_mime(".json") == "application/json"

    def test_png(self) -> None:
        assert _guess_mime(".png") == "image/png"

    def test_jpg(self) -> None:
        assert _guess_mime(".jpg") == "image/jpeg"

    def test_jpeg(self) -> None:
        assert _guess_mime(".jpeg") == "image/jpeg"

    def test_gif(self) -> None:
        assert _guess_mime(".gif") == "image/gif"

    def test_svg(self) -> None:
        assert _guess_mime(".svg") == "image/svg+xml"

    def test_ico(self) -> None:
        assert _guess_mime(".ico") == "image/x-icon"

    def test_txt(self) -> None:
        assert _guess_mime(".txt") == "text/plain"

    def test_unknown(self) -> None:
        assert _guess_mime(".unknown") == "application/octet-stream"

    def test_no_dot(self) -> None:
        assert _guess_mime("") == "application/octet-stream"


class TestSlugify:
    """Test the _slugify function."""

    def test_simple_text(self) -> None:
        assert _slugify("Hello World") == "hello-world"

    def test_already_slug(self) -> None:
        assert _slugify("hello-world") == "hello-world"

    def test_special_chars(self) -> None:
        assert _slugify("Hello, World! @#$%") == "hello-world"

    def test_multiple_spaces(self) -> None:
        assert _slugify("hello   world") == "hello-world"

    def test_leading_trailing_spaces(self) -> None:
        assert _slugify("  hello world  ") == "hello-world"

    def test_unicode(self) -> None:
        assert _slugify("café") == "café"

    def test_empty_string(self) -> None:
        assert _slugify("") == ""

    def test_only_special_chars(self) -> None:
        assert _slugify("!!!") == ""


class TestApp:
    """Test the App class."""

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_add_endpoint_duplicate_raises(self) -> None:
        """Adding an endpoint with duplicate name should raise ValueError."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)
        with pytest.raises(ValueError, match="Endpoint 'test' already exists"):
            app.add_endpoint("test", ep_config)

    def test_endpoint_names_property(self) -> None:
        """endpoint_names should return list of endpoint names."""
        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("alpha", ep_config)
        app.add_endpoint("beta", ep_config)
        names = app.endpoint_names
        assert sorted(names) == ["alpha", "beta"]

    def test_get_endpoint_returns_none(self) -> None:
        """get_endpoint for non-existent name should return None."""
        app = App()
        assert app.get_endpoint("nonexistent") is None

    def test_serve_static_empty_path_uses_index(self, tmp_path: Path) -> None:
        """When file_path is empty, 'index.html' should be served."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html>index</html>")

        status, headers, body = self._run_async(_serve_static_async(static_dir, ""))
        assert status == HTTPStatus.OK
        assert b"index" in body

    def test_serve_static_empty_path_slash(self, tmp_path: Path) -> None:
        """When file_path is '/', 'index.html' should be served."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html>index</html>")

        status, headers, body = self._run_async(_serve_static_async(static_dir, "/"))
        assert status == HTTPStatus.OK
        assert b"index" in body

    def test_serve_static_hidden_file_refused(self, tmp_path: Path) -> None:
        """Hidden files (starting with '.') should not be served."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        hidden_file = static_dir / ".secret"
        hidden_file.write_text("secret content")

        status, headers, body = self._run_async(
            _serve_static_async(static_dir, "/.secret")
        )
        assert status == HTTPStatus.NOT_FOUND

    def test_serve_static_hidden_in_subdir(self, tmp_path: Path) -> None:
        """Files in hidden subdirectories should not be served."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        hidden_dir = static_dir / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "file.txt").write_text("content")

        status, headers, body = self._run_async(
            _serve_static_async(static_dir, "/.hidden/file.txt")
        )
        assert status == HTTPStatus.NOT_FOUND

    def test_serve_static_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Path traversal outside static_dir should not be served."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("outside")

        status, headers, body = self._run_async(
            _serve_static_async(static_dir, "/../outside.txt")
        )
        assert status == HTTPStatus.NOT_FOUND

    def test_input_callback_continue_branch(self) -> None:
        """_input_callback should handle timeout on wait and continue."""
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

        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

        def _add_and_stop() -> None:
            import time

            time.sleep(0.15)
            with handler._queue_lock:
                handler._message_queue.append({"command": "", "message": "after"})
                handler._input_available.set()

        thread = threading.Thread(target=_add_and_stop, daemon=True)
        thread.start()
        result = handler._input_callback()
        assert result == {"command": "", "message": "after"}

    def test_confirm_callback_continue_branch(self) -> None:
        """_confirm_callback should handle timeout on wait and continue."""
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

        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

        def _set_confirm() -> None:
            import time

            time.sleep(0.15)
            with handler._queue_lock:
                handler._confirm_result = True
            handler._confirm_available.set()

        thread = threading.Thread(target=_set_confirm, daemon=True)
        thread.start()
        result = handler._confirm_callback({"action": "test"}, None, "test-task")
        assert result is True

    def test_ws_handler_invalid_path(self) -> None:
        """_ws_handler should return early for non-WS paths."""
        from websockets.server import WebSocketServerProtocol

        app = App()
        mock_ws = MagicMock(spec=WebSocketServerProtocol)

        async def _test() -> None:
            await app._ws_handler(mock_ws, "/invalid/path")

        self._run_async(_test())

    def test_ws_handler_no_endpoint(self) -> None:
        """_ws_handler should return early for non-existent endpoint."""
        from websockets.server import WebSocketServerProtocol

        app = App()
        mock_ws = MagicMock(spec=WebSocketServerProtocol)

        async def _test() -> None:
            await app._ws_handler(mock_ws, "/nonexistent/ws")

        self._run_async(_test())

    def test_ws_handler_no_manifest(self) -> None:
        """_ws_handler should return early when endpoint has no manifest."""
        from websockets.server import WebSocketServerProtocol

        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("empty", ep_config)
        mock_ws = MagicMock(spec=WebSocketServerProtocol)

        async def _test() -> None:
            await app._ws_handler(mock_ws, "/empty/ws")

        self._run_async(_test())

    def test_ws_handler_endpoint_not_found(self) -> None:
        """_ws_handler: when get_endpoint returns None after parse success (line 479-480)."""
        from websockets.server import WebSocketServerProtocol

        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)
        mock_ws = MagicMock(spec=WebSocketServerProtocol)

        # Make get_endpoint return None even though name was parsed
        with patch.object(app, "get_endpoint", return_value=None):

            async def _test() -> None:
                await app._ws_handler(mock_ws, "/test/ws")

            self._run_async(_test())

    def test_run_in_thread(self) -> None:
        """run_in_thread should start a daemon thread."""
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

        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))

        def ws_send(data: str) -> None:
            pass

        with patch("tiz.web_api.InteractiveChat", autospec=True) as mock_ic:
            mock_chat_instance = mock_ic.return_value
            hold = threading.Event()

            def _run_and_hold() -> None:
                hold.wait(timeout=2)
                handler._running.clear()

            mock_chat_instance.run.side_effect = _run_and_hold

            thread = handler.run_in_thread(ws_send)
            assert thread is not None
            assert thread.daemon is True
            # Check _chat_thread while the thread is still inside _run()
            assert handler._chat_thread is thread
            assert handler._ws_send is ws_send
            hold.set()
            thread.join(timeout=2)
            assert not thread.is_alive()

    def test_ws_handler_send_connection_closed_through_app(self) -> None:
        """ConnectionClosed in send_worker should be caught by _ws_handler."""
        import asyncio as _asyncio
        import threading as _threading

        from websockets.exceptions import ConnectionClosed

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

        app = App()
        ep_config = EndpointConfig(
            manifest=manifest,
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)

        async def _test() -> None:
            send_called = _threading.Event()

            class MockWS:
                async def send(self, _data):
                    send_called.set()
                    raise ConnectionClosed(None, None)

                def __aiter__(self):
                    return _Iter(self)

            class _Iter:
                def __init__(self, ws):
                    self.ws = ws
                    self.count = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.count > 0:
                        raise StopAsyncIteration
                    self.count += 1
                    for _ in range(20):
                        if send_called.is_set():
                            break
                        await _asyncio.sleep(0.1)
                    return json.dumps({"command": "", "message": "hello"})

            def _patched_run(handler_instance: ChatWebSocketHandler) -> None:
                handler_instance.manifest.meta.ephemeral_sandbox = True
                handler_instance.manifest.meta.delete_sandbox_on_exit = True
                import time

                time.sleep(0.2)
                handler_instance._update_callback(
                    {"type": "chat", "text": "hello"}, None
                )
                send_called.wait(timeout=5)
                handler_instance._running.clear()

            mock_ws = MockWS()
            with patch.object(ChatWebSocketHandler, "_run", _patched_run):
                await app._ws_handler(mock_ws, "/test/ws")  # type: ignore[arg-type]

            assert send_called.is_set(), "send should have been called"

        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_ws_handler_recv_connection_closed_through_app(self) -> None:
        """ConnectionClosed in recv loop should be caught by _ws_handler."""
        import asyncio as _asyncio

        from websockets.exceptions import ConnectionClosed

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

        app = App()
        ep_config = EndpointConfig(
            manifest=manifest,
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)

        async def _test() -> None:
            class MockWS:
                async def send(self, data):
                    pass

                def __aiter__(self):
                    return _RecvIter()

            class _RecvIter:
                def __init__(self):
                    self.count = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    await _asyncio.sleep(0)
                    if self.count == 0:
                        self.count += 1
                        return json.dumps({"command": "", "message": "hello"})
                    raise ConnectionClosed(None, None)

            def _patched_run(handler_instance: ChatWebSocketHandler) -> None:
                handler_instance.manifest.meta.ephemeral_sandbox = True
                handler_instance.manifest.meta.delete_sandbox_on_exit = True
                import time

                time.sleep(0.1)
                handler_instance._update_callback(
                    {"type": "chat", "text": "hello"}, None
                )
                time.sleep(0.3)
                handler_instance._running.clear()

            mock_ws = MockWS()
            with patch.object(ChatWebSocketHandler, "_run", _patched_run):
                await app._ws_handler(mock_ws, "/test/ws")  # type: ignore[arg-type]

        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_ws_handler_invalid_json_through_app(self) -> None:
        """Invalid JSON from websocket should be logged and skipped."""
        import asyncio as _asyncio

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

        app = App()
        ep_config = EndpointConfig(
            manifest=manifest,
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)

        async def _test() -> None:
            class MockWS:
                def __init__(self):
                    self.messages = [
                        b"not valid json",
                        json.dumps({"command": "", "message": "ok"}).encode(),
                    ]

                async def send(self, data):
                    pass

                def __aiter__(self):
                    return _Iter(self)

            class _Iter:
                def __init__(self, ws):
                    self.ws = ws
                    self.idx = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.idx >= len(self.ws.messages):
                        raise StopAsyncIteration
                    msg = self.ws.messages[self.idx]
                    self.idx += 1
                    return msg

            def _patched_run(handler_instance: ChatWebSocketHandler) -> None:
                handler_instance.manifest.meta.ephemeral_sandbox = True
                handler_instance.manifest.meta.delete_sandbox_on_exit = True
                import time

                time.sleep(0.3)
                handler_instance._running.clear()

            mock_ws = MockWS()
            with patch.object(ChatWebSocketHandler, "_run", _patched_run):
                await app._ws_handler(mock_ws, "/test/ws")  # type: ignore[arg-type]

        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_ws_handler_non_dict_json_through_app(self) -> None:
        """Non-dict JSON from websocket (null, number, array, string) should be skipped."""
        import asyncio as _asyncio

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

        app = App()
        ep_config = EndpointConfig(
            manifest=manifest,
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)

        async def _test() -> None:
            class MockWS:
                def __init__(self):
                    self.messages = [
                        json.dumps(None).encode(),
                        json.dumps(42).encode(),
                        json.dumps([]).encode(),
                        json.dumps("string").encode(),
                        json.dumps({"command": "", "message": "ok"}).encode(),
                    ]

                async def send(self, data):
                    pass

                def __aiter__(self):
                    return _Iter(self)

            class _Iter:
                def __init__(self, ws):
                    self.ws = ws
                    self.idx = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.idx >= len(self.ws.messages):
                        raise StopAsyncIteration
                    msg = self.ws.messages[self.idx]
                    self.idx += 1
                    return msg

            def _patched_run(handler_instance: ChatWebSocketHandler) -> None:
                handler_instance.manifest.meta.ephemeral_sandbox = True
                handler_instance.manifest.meta.delete_sandbox_on_exit = True
                import time

                time.sleep(0.3)
                handler_instance._running.clear()

            mock_ws = MockWS()
            with patch.object(ChatWebSocketHandler, "_run", _patched_run):
                await app._ws_handler(mock_ws, "/test/ws")  # type: ignore[arg-type]

        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_ws_handler_send_worker_exits_on_running_false(self) -> None:
        """send_worker loop should exit when handler._running becomes False."""
        import asyncio as _asyncio

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

        app = App()
        ep_config = EndpointConfig(
            manifest=manifest,
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)

        async def _test() -> None:
            class MockWS:
                async def send(self, data):
                    pass

                def __aiter__(self):
                    return _Iter()

            class _Iter:
                def __init__(self):
                    self.done = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.done:
                        raise StopAsyncIteration
                    self.done = True
                    return json.dumps({"command": "", "message": "go"}).encode()

            def _patched_run(handler_instance: ChatWebSocketHandler) -> None:
                handler_instance.manifest.meta.ephemeral_sandbox = True
                handler_instance.manifest.meta.delete_sandbox_on_exit = True
                import time

                # This sends an update which queues a message for the send_worker
                handler_instance._update_callback(
                    {"type": "chat", "text": "ping"}, None
                )
                time.sleep(0.3)
                handler_instance._running.clear()

            mock_ws = MockWS()
            with patch.object(ChatWebSocketHandler, "_run", _patched_run):
                await app._ws_handler(mock_ws, "/test/ws")  # type: ignore[arg-type]

        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_close(self) -> None:
        """close should set _running to False and set events."""
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

        handler = ChatWebSocketHandler(manifest=manifest, base_path=Path("/tmp"))
        assert handler._running.is_set()
        handler.close()
        assert not handler._running.is_set()
        assert handler._input_available.is_set()
        assert handler._confirm_available.is_set()
        assert handler._stop_sleep_event.is_set()

    def test_send_worker_immediate_exit_when_running_false(self) -> None:
        """send_worker should exit immediately when handler._running is already False."""
        import asyncio as _asyncio

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

        app = App()
        ep_config = EndpointConfig(
            manifest=manifest,
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)

        async def _test() -> None:
            class MockWS:
                async def send(self, data):
                    pass

                def __aiter__(self):
                    return _Iter()

            class _Iter:
                def __init__(self):
                    self.done = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    raise StopAsyncIteration

            def _patched_run(handler_instance: ChatWebSocketHandler) -> None:
                handler_instance.manifest.meta.ephemeral_sandbox = True
                handler_instance.manifest.meta.delete_sandbox_on_exit = True
                # Don't send anything, just set running to False quickly
                handler_instance._running.clear()

            mock_ws = MockWS()
            with patch.object(ChatWebSocketHandler, "_run", _patched_run):
                await app._ws_handler(mock_ws, "/test/ws")  # type: ignore[arg-type]

        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_send_worker_continue_then_exit_when_running_false(self) -> None:
        """send_worker: continue from TimeoutError, then exit when _running becomes False."""
        import asyncio as _asyncio

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

        app = App()
        ep_config = EndpointConfig(
            manifest=manifest,
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)

        async def _test() -> None:
            keep_alive = [True]

            class _Iter:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    # Stay alive for 0.6s to let send_worker exit naturally
                    if keep_alive[0]:
                        await _asyncio.sleep(0.05)
                        return json.dumps({"command": "", "message": "x"}).encode()
                    raise StopAsyncIteration

            class MockWS:
                async def send(self, data):
                    pass

                def __aiter__(self):
                    return _Iter()

            def _patched_run(handler_instance: ChatWebSocketHandler) -> None:
                handler_instance.manifest.meta.ephemeral_sandbox = True
                handler_instance.manifest.meta.delete_sandbox_on_exit = True
                import time

                time.sleep(0.35)
                handler_instance._running.clear()
                time.sleep(0.3)
                keep_alive[0] = False

            mock_ws = MockWS()
            with patch.object(ChatWebSocketHandler, "_run", _patched_run):
                await app._ws_handler(mock_ws, "/test/ws")  # type: ignore[arg-type]

        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()


class TestAppRun:
    """Test the App.run method."""

    def test_run_keyboard_interrupt(self) -> None:
        """KeyboardInterrupt during App.run should be handled gracefully."""
        from unittest.mock import AsyncMock

        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        with patch("tiz.web_api.serve") as mock_serve:
            mock_serve.return_value.__aenter__ = AsyncMock(
                side_effect=KeyboardInterrupt()
            )
            mock_serve.return_value.__aexit__ = AsyncMock(return_value=None)
            with patch("builtins.print"):
                app.run(host="127.0.0.1", port=0)

    def test_run_with_endpoints_and_static(self) -> None:
        """App.run should print endpoint info and handle static dir."""
        from unittest.mock import AsyncMock

        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test1", ep_config)
        app.add_endpoint("test2", ep_config)
        app._static_dir = Path("/tmp")

        mock_serve_instance = MagicMock()
        mock_serve_instance.__aenter__ = AsyncMock(side_effect=KeyboardInterrupt())
        mock_serve_instance.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("tiz.web_api.serve", return_value=mock_serve_instance),
            patch("builtins.print"),
        ):
            app.run(host="127.0.0.1", port=0)

    def test_run_await_future_raises_kbi(self) -> None:
        """App.run should handle KeyboardInterrupt from asyncio.Future() in _run_server."""
        from unittest.mock import AsyncMock

        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)

        mock_serve_instance = MagicMock()
        mock_serve_instance.__aenter__ = AsyncMock(return_value=None)
        mock_serve_instance.__aexit__ = AsyncMock(return_value=None)

        # _run_server does: async with serve(...): await asyncio.Future()
        # We need the serve context to enter OK, then the await to raise KBI
        class _KbiFuture:
            def __await__(self):
                raise KeyboardInterrupt()
                yield  # type: ignore[unreachable]

        with (
            patch("tiz.web_api.serve", return_value=mock_serve_instance),
            patch("tiz.web_api.asyncio.Future", return_value=_KbiFuture()),
            patch("builtins.print"),
        ):
            app.run(host="127.0.0.1", port=0)

    def test_run_unix_socket_keyboard_interrupt(self, tmp_path: Path) -> None:
        """App.run with Unix socket path should handle KeyboardInterrupt."""
        from unittest.mock import AsyncMock

        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)
        socket_path = str(tmp_path / "test.sock")

        with patch("tiz.web_api.serve") as mock_serve:
            mock_serve.return_value.__aenter__ = AsyncMock(
                side_effect=KeyboardInterrupt()
            )
            mock_serve.return_value.__aexit__ = AsyncMock(return_value=None)
            with patch("builtins.print"):
                app.run(host="127.0.0.1", port=0, path=socket_path)

    def test_run_unix_socket_with_static(self, tmp_path: Path) -> None:
        """App.run with Unix socket path should print endpoint info and handle static dir."""
        from unittest.mock import AsyncMock

        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test1", ep_config)
        app._static_dir = Path("/tmp")
        socket_path = str(tmp_path / "test.sock")

        mock_serve_instance = MagicMock()
        mock_serve_instance.__aenter__ = AsyncMock(side_effect=KeyboardInterrupt())
        mock_serve_instance.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("tiz.web_api.serve", return_value=mock_serve_instance),
            patch("builtins.print"),
        ):
            app.run(host="127.0.0.1", port=0, path=socket_path)

    def test_run_unix_socket_await_future_raises_kbi(self, tmp_path: Path) -> None:
        """App.run with Unix socket should handle KeyboardInterrupt from asyncio.Future()."""
        from unittest.mock import AsyncMock

        app = App()
        ep_config = EndpointConfig(
            manifest=None,  # type: ignore[arg-type]
            base_path=Path("/tmp"),
        )
        app.add_endpoint("test", ep_config)
        socket_path = str(tmp_path / "test.sock")

        mock_serve_instance = MagicMock()
        mock_serve_instance.__aenter__ = AsyncMock(return_value=None)
        mock_serve_instance.__aexit__ = AsyncMock(return_value=None)

        class _KbiFuture:
            def __await__(self):
                raise KeyboardInterrupt()
                yield  # type: ignore[unreachable]

        with (
            patch("tiz.web_api.serve", return_value=mock_serve_instance),
            patch("tiz.web_api.asyncio.Future", return_value=_KbiFuture()),
            patch("builtins.print"),
        ):
            app.run(host="127.0.0.1", port=0, path=socket_path)


class TestRunSimple:
    """Test the run_simple function."""

    def test_run_simple_parse_error(self, tmp_path: Path) -> None:
        """When config parsing fails, run_simple should log error and return."""
        config_path = tmp_path / "nonexistent.yaml"
        with patch("tiz.web_api.logger") as mock_logger:
            from tiz.web_api import run_simple

            run_simple(
                base_path=tmp_path,
                config_path=config_path,
                host="127.0.0.1",
                port=0,
            )
            mock_logger.error.assert_called()

    def test_run_simple_config_parse_exception(self, tmp_path: Path) -> None:
        """When parse_web_config raises, run_simple logs error."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: ::\n")
        with patch("tiz.web_api.logger") as mock_logger:
            from tiz.web_api import run_simple

            run_simple(base_path=tmp_path, config_path=config_path)
            mock_logger.error.assert_called()

    def test_run_simple_success_path(self, tmp_path: Path) -> None:
        """run_simple should create app and add slugified endpoints."""
        from tiz.web_api import run_simple

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "static_dir: static\n"
            "endpoints:\n"
            "  my endpoint:\n"
            "    manifest:\n"
            "      path: /nonexistent.yaml\n"
        )
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        with patch("tiz.web_api.parse_web_config") as mock_parse:
            mock_config = MagicMock()
            mock_config.static_dir = static_dir
            mock_ep = MagicMock()
            mock_config.endpoints = {"my endpoint": mock_ep}
            mock_parse.return_value = mock_config

            with patch.object(App, "run") as mock_run:
                run_simple(
                    base_path=tmp_path,
                    config_path=config_path,
                    host="127.0.0.1",
                    port=8080,
                )
                mock_run.assert_called_once_with(host="127.0.0.1", port=8080, path=None)

    def test_run_simple_unix_socket(self, tmp_path: Path) -> None:
        """run_simple should pass path to App.run."""
        from tiz.web_api import run_simple

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "static_dir: static\n"
            "endpoints:\n"
            "  my endpoint:\n"
            "    manifest:\n"
            "      path: /nonexistent.yaml\n"
        )
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        socket_path = str(tmp_path / "test.sock")

        with patch("tiz.web_api.parse_web_config") as mock_parse:
            mock_config = MagicMock()
            mock_config.static_dir = static_dir
            mock_ep = MagicMock()
            mock_config.endpoints = {"my endpoint": mock_ep}
            mock_parse.return_value = mock_config

            with patch.object(App, "run") as mock_run:
                run_simple(
                    base_path=tmp_path,
                    config_path=config_path,
                    host="127.0.0.1",
                    port=8080,
                    path=socket_path,
                )
                mock_run.assert_called_once_with(
                    host="127.0.0.1", port=8080, path=socket_path
                )

        with patch("tiz.web_api.parse_web_config") as mock_parse:
            mock_config = MagicMock()
            mock_config.static_dir = static_dir
            mock_ep = MagicMock()
            mock_config.endpoints = {"my endpoint": mock_ep}
            mock_parse.return_value = mock_config

            with patch.object(App, "run") as mock_run:
                run_simple(
                    base_path=tmp_path,
                    config_path=config_path,
                    host="127.0.0.1",
                    port=8080,
                )
                mock_run.assert_called_once_with(host="127.0.0.1", port=8080, path=None)

    def test_run_simple_app_run_exception(self, tmp_path: Path) -> None:
        """When App.run raises, run_simple logs error."""
        from tiz.web_api import run_simple

        config_path = tmp_path / "config.yaml"
        config_path.write_text("static_dir: static\nendpoints: {}\n")
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        with patch("tiz.web_api.parse_web_config") as mock_parse:
            mock_config = MagicMock()
            mock_config.static_dir = static_dir
            mock_config.endpoints = {}
            mock_parse.return_value = mock_config

            with (
                patch.object(App, "run", side_effect=RuntimeError("Server failed")),
                patch("tiz.web_api.logger") as mock_logger,
            ):
                run_simple(base_path=tmp_path, config_path=config_path)
                mock_logger.error.assert_called()
