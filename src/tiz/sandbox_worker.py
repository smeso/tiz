#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("tiz-worker")

_MAX_CONNECTIONS = 50
_CONNECTION_SEMAPHORE = threading.Semaphore(_MAX_CONNECTIONS)
_MAX_BUFFER_SIZE = 1024 * 1024  # 1 MB
_HAS_RG: bool | None = None
_TOOL_LOCK = threading.Lock()
_MAX_LIST_DIR_ENTRIES = 1000
_CUSTOM_TOOLS_DIR = Path(os.environ.get("TIZ_CUSTOM_TOOLS_DIR", "/opt/tiz_tools"))
_TIZ_VERSION = "0.1.0"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko);"
    f" compatible; tiz/{_TIZ_VERSION}; +https://github.com/smeso/tiz"
)

_WEBFETCH_CACHE: OrderedDict[str, tuple[int, dict[str, str], str]] = OrderedDict()
_WEBFETCH_CACHE_MAX = int(os.environ.get("TIZ_WEBFETCH_CACHE_MAX", "256"))
_WEBFETCH_CACHE_LOCK = threading.Lock()
_WEBFETCH_RATE_LIMIT_LOCK = threading.Lock()
_WEBFETCH_LAST_CALL: dict[str, float] = {}
_WEBFETCH_MIN_INTERVAL = float(os.environ.get("TIZ_WEBFETCH_MIN_INTERVAL", "0.5"))

_WEBSEARCH_CACHE: OrderedDict[str, str] = OrderedDict()
_WEBSEARCH_CACHE_MAX = int(os.environ.get("TIZ_WEBSEARCH_CACHE_MAX", "128"))
_WEBSEARCH_CACHE_LOCK = threading.Lock()
_WEBSEARCH_RATE_LIMIT_LOCK = threading.Lock()
_WEBSEARCH_LAST_CALL: float = 0.0
_WEBSEARCH_MIN_INTERVAL = float(os.environ.get("TIZ_WEBSEARCH_MIN_INTERVAL", "1.0"))
_WEBSEARCH_URL = os.environ.get(
    "TIZ_WEBSEARCH_URL", "https://html.duckduckgo.com/html/"
)


def _load_custom_tools() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    if not _CUSTOM_TOOLS_DIR.is_dir():
        return tools
    for py_file in sorted(_CUSTOM_TOOLS_DIR.iterdir()):
        if not py_file.is_file() or py_file.suffix != ".py":
            continue
        tool_name = py_file.stem
        try:
            spec = importlib.util.spec_from_file_location(tool_name, py_file)
            if spec is None or spec.loader is None:  # pragma: no cover
                logger.warning(
                    "Cannot load tool %s: unable to create module spec", py_file
                )
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "handle"):
                tools[tool_name] = module.handle
                logger.info("Loaded custom tool: %s from %s", tool_name, py_file)
            else:
                logger.warning(
                    "Cannot load tool %s: module has no 'handle' attribute", py_file
                )
        except Exception:
            logger.warning(
                "Cannot load tool %s: failed to import module", py_file, exc_info=True
            )
    return tools


def _send_response(conn: socket.socket, response_dict: dict[str, Any]) -> None:
    resp_data = json.dumps(response_dict).encode("utf-8") + b"\n"
    conn.sendall(struct.pack(">I", len(resp_data)) + resp_data)


def _resolve_path(
    params: dict[str, Any],
    key: str = "path",
    default: str = "",
) -> Path:
    return Path(params.get(key, default)).expanduser().resolve(strict=False)


def _check_has_rg() -> bool:
    global _HAS_RG
    if _HAS_RG is not None:
        return _HAS_RG
    try:
        _HAS_RG = (
            subprocess.run(["which", "rg"], capture_output=True, check=False).returncode
            == 0
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        _HAS_RG = False
    return _HAS_RG


def _tool_cargo_fetch(params: dict[str, Any]) -> tuple[str, bool]:
    project_path = params.get("path", "")
    if not project_path:
        return "ERROR: path is required", True
    raw_timeout = params.get("timeout", 300)
    timeout = max(1, min(raw_timeout, 600))
    resolved = Path(project_path).expanduser().resolve(strict=False)
    cargo_toml = resolved / "Cargo.toml"
    if not cargo_toml.is_file():
        return f"ERROR: Cargo.toml not found at {cargo_toml}", True
    try:
        result = subprocess.run(
            ["cargo", "fetch", "--manifest-path", str(cargo_toml)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(resolved),
        )
        out = result.stdout + result.stderr
        if result.returncode != 0:
            out += f"\n(exit code: {result.returncode})"
        return out, result.returncode != 0
    except subprocess.TimeoutExpired:
        return "Cargo fetch timed out", True
    except FileNotFoundError:
        return "ERROR: cargo not found", True


def _tool_uv_sync(params: dict[str, Any]) -> tuple[str, bool]:
    project_path = params.get("path", "")
    if not project_path:
        return "ERROR: path is required", True
    raw_timeout = params.get("timeout", 300)
    timeout = max(1, min(raw_timeout, 600))
    resolved = Path(project_path).expanduser().resolve(strict=False)
    if not resolved.is_dir():
        return f"ERROR: directory not found at {resolved}", True
    cmd = ["uv", "sync", "--project", str(resolved)]
    group = params.get("group", [])
    if type(group) is not list:
        return "ERROR: group must be a list", True
    for g in group:
        cmd.extend(["--group", g])
    extra = params.get("extra", [])
    if type(extra) is not list:
        return "ERROR: extra must be a list", True
    for e in extra:
        cmd.extend(["--extra", e])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(resolved),
        )
        out = result.stdout + result.stderr
        if result.returncode != 0:
            out += f"\n(exit code: {result.returncode})"
        return out, result.returncode != 0
    except subprocess.TimeoutExpired:
        return "uv sync timed out", True
    except FileNotFoundError:
        return "ERROR: uv not found", True


def _tool_uv_python_install(params: dict[str, Any]) -> tuple[str, bool]:
    version = params.get("version", "")
    if not version:
        return "ERROR: version is required", True
    raw_timeout = params.get("timeout", 300)
    timeout = max(1, min(raw_timeout, 600))
    try:
        result = subprocess.run(
            ["uv", "python", "install", "--no-bin", "-f", version],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = result.stdout + result.stderr
        if result.returncode != 0:
            out += f"\n(exit code: {result.returncode})"
        return out, result.returncode != 0
    except subprocess.TimeoutExpired:
        return "uv python install timed out", True
    except FileNotFoundError:
        return "ERROR: uv not found", True


def _tool_bash(params: dict[str, Any]) -> tuple[str, bool]:
    cmd = params.get("command", "")
    raw_timeout = params.get("timeout", 30)
    timeout = max(1, min(raw_timeout, 300))
    cwd = params.get("cwd")
    env_override = params.get("env")

    if cwd is not None and not Path(cwd).expanduser().is_dir():
        return f"ERROR: cwd not found: {cwd}", True

    if env_override is not None:
        if not isinstance(env_override, dict):
            return "ERROR: env must be a dict", True
        for k, v in env_override.items():
            if not isinstance(v, str):
                return (
                    f"ERROR: env value for '{k}' must be a string, got {type(v).__name__}",
                    True,
                )

    try:
        env = None
        if env_override is not None:
            env = {**os.environ, **env_override}
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
            env=env,
        )
        out = result.stdout + result.stderr
        if result.returncode != 0:
            out += f"\n(exit code: {result.returncode})"
        return out, result.returncode != 0
    except subprocess.TimeoutExpired:
        return "Command timed out", True


def _tool_read(params: dict[str, Any]) -> tuple[str, bool]:
    resolved = _resolve_path(params)
    view_range = params.get("view_range")

    if not resolved.is_file():
        return f"ERROR: file not found {resolved}", True

    try:
        file_size = resolved.stat().st_size
        if file_size > _MAX_BUFFER_SIZE:
            return (
                f"ERROR: file too large ({file_size} bytes, max {_MAX_BUFFER_SIZE})",
                True,
            )
        lines = resolved.read_text(errors="replace", encoding="utf-8").splitlines()
    except OSError as exc:
        return f"Error reading file: {exc}", True

    if view_range is not None:
        if not isinstance(view_range, (list, tuple)) or len(view_range) != 2:
            return "ERROR: view_range must be a two-element array", True
        start, end = view_range
        if type(start) is not int or type(end) is not int:
            return "ERROR: view_range values must be integers", True
        if start < 1 or end < 1:
            return "ERROR: view_range values must be positive", True
        if start > end:
            return "ERROR: view_range start must be <= end", True
        offset = start - 1
        limit = end - start + 1
        end_idx = min(len(lines), offset + limit)
        out = "\n".join(f"{i + 1}\t{lines[i]}" for i in range(offset, end_idx))
        return out, False

    out = "\n".join(f"{i + 1}\t{lines[i]}" for i in range(len(lines)))
    return out, False


def _tool_edit(params: dict[str, Any]) -> tuple[str, bool]:
    resolved = _resolve_path(params)

    if not resolved.is_file():
        return f"ERROR: file not found {resolved}", True

    try:
        content = resolved.read_text(errors="replace", encoding="utf-8")
        old_str = params.get("old_string", "")
        expected_replacements = params.get("expected_replacements", 1)
        if type(expected_replacements) is not int:
            return "ERROR: expected_replacements must be an integer", True

        count = content.count(old_str)
        if count == 0:
            return "ERROR: old_string not found exactly in file", True

        if expected_replacements == -1:
            new_content = content.replace(old_str, params.get("new_string", ""))
            replacements = count
        elif expected_replacements != 1:
            if count != expected_replacements:
                return (
                    f"ERROR: expected {expected_replacements} occurrences but found {count}",
                    True,
                )
            new_content = content.replace(
                old_str, params.get("new_string", ""), expected_replacements
            )
            replacements = expected_replacements
        else:
            if count > 1:
                return "ERROR: old_string matches multiple locations", True
            new_content = content.replace(old_str, params.get("new_string", ""), 1)
            replacements = 1

        resolved.write_text(new_content, encoding="utf-8")
        return f"Edited {resolved} ({replacements} replacement(s) made)", False
    except OSError as exc:
        return f"Error editing file: {exc}", True


def _tool_write(params: dict[str, Any]) -> tuple[str, bool]:
    resolved = _resolve_path(params)

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        content = params.get("contents", "")
        resolved.write_text(content, encoding="utf-8")
        return f"Wrote {resolved}", False
    except OSError as exc:
        return f"Error writing file: {exc}", True


def _tool_glob(params: dict[str, Any]) -> tuple[str, bool]:
    resolved = _resolve_path(params, default=".")

    if not resolved.is_dir():
        return f"ERROR: directory not found {resolved}", True

    pattern = params.get("pattern", "")
    matches: list[str] = []
    try:
        for p in resolved.glob(pattern):
            if not p.is_file():
                continue
            rel = p.relative_to(resolved)
            if ".git" in rel.parts:
                continue
            matches.append(str(p))
            if len(matches) >= 100:
                break

        out = "\n".join(sorted(matches[:100]))
        return out or "No matches", False
    except OSError as exc:  # pragma: no cover
        return f"Error searching directories: {exc}", True  # pragma: no cover


def _tool_grep(params: dict[str, Any]) -> tuple[str, bool]:
    resolved = _resolve_path(params, default=".")
    pattern = params.get("pattern", "")
    glob_filter = params.get("glob")
    use_regex = params.get("regex", False)
    max_results = params.get("max_results", 100)
    if type(max_results) is not int or max_results < 1:
        return "ERROR: max_results must be a positive integer", True

    try:
        has_rg = _check_has_rg()
        if has_rg:
            cmd = ["rg", "--no-heading", "--line-number", "--color=never"]
            if glob_filter:
                cmd += ["-g", glob_filter]
            if not use_regex:
                cmd.append("-F")
            if params.get("case_insensitive"):
                cmd.append("-i")
            cmd += ["-m", str(max_results), "--", pattern, str(resolved)]
        else:
            if glob_filter:
                # Use find to locate files matching the glob, then grep on them
                find_cmd = ["find", str(resolved), "-type", "f", "-name", glob_filter]
                try:
                    find_result = subprocess.run(
                        find_cmd, capture_output=True, text=True, check=True
                    )
                    files = find_result.stdout.splitlines()
                    if not files:
                        return "No matches", False
                    grep_cmd = ["grep", "-rnH", "--color=never"]
                    if not use_regex:
                        grep_cmd.append("-F")
                    if params.get("case_insensitive"):
                        grep_cmd.append("-i")
                    grep_cmd += ["-m", str(max_results), "--", pattern, "--"]
                    grep_cmd += files
                    result = subprocess.run(
                        grep_cmd, capture_output=True, text=True, check=False
                    )
                    lines = result.stdout.splitlines()
                    out = "\n".join(lines[:max_results])
                    return out or "No matches", False
                except subprocess.SubprocessError as exc:
                    return f"Error running grep: {exc}", True
            cmd = ["grep", "-rn", "--color=never", "-m", str(max_results)]
            if not use_regex:
                cmd.append("-F")
            else:
                cmd.append("-P")
            if params.get("case_insensitive"):
                cmd.append("-i")
            cmd += ["--", pattern, str(resolved)]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        lines = result.stdout.splitlines()
        out = "\n".join(lines[:max_results])
        return out or "No matches", False
    except subprocess.SubprocessError as exc:
        return f"Error running grep: {exc}", True  # pragma: no cover


def _tool_insert(params: dict[str, Any]) -> tuple[str, bool]:
    resolved = _resolve_path(params)
    content = params.get("content", "")
    line_number = params.get("line_number", -1)
    if type(line_number) is not int:
        return "ERROR: line_number must be an integer", True

    if not resolved.is_file():
        return f"ERROR: file not found {resolved}", True

    try:
        lines = resolved.read_text(errors="replace", encoding="utf-8").splitlines()
        if line_number == -1:
            lines.append(content.rstrip("\n"))
            inserted = 1
        else:
            if line_number < 1 or line_number > len(lines) + 1:
                return (
                    f"ERROR: line_number {line_number} out of range (1-{len(lines) + 1})",
                    True,
                )
            insert_lines = content.rstrip("\n").split("\n")
            for i, line in enumerate(insert_lines):
                lines.insert(line_number - 1 + i, line)
            inserted = len(insert_lines)

        resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Inserted {inserted} line(s) into {resolved}", False
    except OSError as exc:
        return f"Error inserting content: {exc}", True


def _tool_list_dir(params: dict[str, Any]) -> tuple[str, bool]:
    resolved = _resolve_path(params, default=".")
    recursive = params.get("recursive", False)
    show_hidden = params.get("show_hidden", False)

    if not resolved.is_dir():
        return f"ERROR: directory not found {resolved}", True

    try:
        entries: list[dict[str, Any]] = []
        if recursive:
            for root, dirs, files in os.walk(resolved, followlinks=False):
                if not show_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in dirs + files:
                    if len(entries) >= _MAX_LIST_DIR_ENTRIES:
                        break
                    if not show_hidden and name.startswith("."):
                        continue
                    full = Path(root) / name
                    if full.is_symlink():
                        entry_type = "symlink"
                        st = full.lstat()
                    elif full.is_dir():
                        entry_type = "directory"
                        st = full.stat()
                    else:
                        entry_type = "file"
                        st = full.stat()
                    entry: dict[str, Any] = {
                        "name": name,
                        "path": str(full),
                        "type": entry_type,
                        "size": st.st_size if entry_type == "file" else None,
                        "modified": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)
                        ),
                    }
                    entries.append(entry)
                if len(entries) >= _MAX_LIST_DIR_ENTRIES:
                    break
        else:
            for item in resolved.iterdir():
                if len(entries) >= _MAX_LIST_DIR_ENTRIES:
                    break
                if not show_hidden and item.name.startswith("."):
                    continue
                if item.is_symlink():
                    entry_type = "symlink"
                    st = item.lstat()
                elif item.is_dir():
                    entry_type = "directory"
                    st = item.stat()
                else:
                    entry_type = "file"
                    st = item.stat()
                entry = {
                    "name": item.name,
                    "path": str(item),
                    "type": entry_type,
                    "size": st.st_size if entry_type == "file" else None,
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)
                    ),
                }
                entries.append(entry)

        return json.dumps(entries, indent=2), False
    except OSError as exc:
        return f"Error listing directory: {exc}", True


def _tool_patch(params: dict[str, Any]) -> tuple[str, bool]:
    patch_content = params.get("patch", "")
    strip = params.get("strip", 0)
    if type(strip) is not int or strip < 0:
        return "ERROR: strip must be a non-negative integer", True
    reverse = params.get("reverse", False)
    cwd = params.get("cwd")
    patch_file: str | None = None

    try:
        if cwd is not None:
            cwd = str(Path(cwd).expanduser())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch_content)
            patch_file = f.name

        cmd = ["patch", "-p", str(strip)]
        if reverse:
            cmd.append("-R")
        cmd += ["--batch", "--no-backup-if-mismatch", "-i", patch_file]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, cwd=cwd
        )

        out = result.stdout
        if result.stderr:
            out += result.stderr
        if result.returncode != 0:
            out += f"\n(exit code: {result.returncode})"
        return out, result.returncode != 0
    except (subprocess.SubprocessError, OSError) as exc:
        return f"Error applying patch: {exc}", True
    finally:
        if patch_file is not None and Path(patch_file).exists():
            Path(patch_file).unlink()


def _tool_read_multi(params: dict[str, Any]) -> tuple[str, bool]:
    paths = params.get("paths", [])
    if not isinstance(paths, list):
        return "ERROR: paths must be an array", True

    results: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p).expanduser()
        resolved = path.resolve(strict=False)
        try:
            if not resolved.is_file():
                results.append(
                    {
                        "path": str(resolved),
                        "content": None,
                        "error": f"File not found: {resolved}",
                    }
                )
                continue
            file_size = resolved.stat().st_size
            if file_size > _MAX_BUFFER_SIZE:
                results.append(
                    {
                        "path": str(resolved),
                        "content": None,
                        "error": f"File too large ({file_size} bytes, max {_MAX_BUFFER_SIZE})",
                    }
                )
                continue
            content = resolved.read_text(errors="replace", encoding="utf-8")
            results.append({"path": str(resolved), "content": content, "error": None})
        except OSError as exc:
            results.append({"path": str(resolved), "content": None, "error": str(exc)})

    return json.dumps(results, indent=2), False


def _tool_metadata(params: dict[str, Any]) -> tuple[str, bool]:
    path = Path(params.get("path", "")).expanduser()
    try:
        is_symlink = path.is_symlink()
    except OSError:
        is_symlink = False
    resolved = path.resolve(strict=False)

    if not resolved.exists() and not is_symlink:
        return json.dumps(
            {"exists": False, "error": f"Path not found: {resolved}"}
        ), False

    try:
        st = path.lstat() if is_symlink else resolved.lstat()
        file_type: str | None
        if is_symlink:
            file_type = "symlink"
        elif resolved.is_dir():
            file_type = "directory"
        elif resolved.is_file():
            file_type = "file"
        else:
            file_type = None

        return json.dumps(
            {
                "exists": True,
                "type": file_type,
                "size": None if is_symlink else st.st_size,
                "created": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(st.st_ctime)
                ),
                "modified": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)
                ),
                "permissions": oct(st.st_mode)[-3:],
                "error": None,
            },
            indent=2,
        ), False
    except OSError as exc:  # pragma: no cover
        return json.dumps(
            {"exists": False, "error": str(exc)}
        ), True  # pragma: no cover


def _rate_limit_domain(domain: str) -> None:
    with _WEBFETCH_RATE_LIMIT_LOCK:
        last = _WEBFETCH_LAST_CALL.get(domain, 0.0)
        now = time.monotonic()
        elapsed = now - last
        need_sleep = max(0.0, _WEBFETCH_MIN_INTERVAL - elapsed)
        if need_sleep > 0:
            time.sleep(need_sleep)
        _WEBFETCH_LAST_CALL[domain] = time.monotonic()


def _cache_get(url: str) -> tuple[int, dict[str, str], str] | None:
    with _WEBFETCH_CACHE_LOCK:
        entry = _WEBFETCH_CACHE.get(url)
        if entry is not None:
            _WEBFETCH_CACHE.move_to_end(url)
        return entry


def _cache_set(url: str, status: int, headers: dict[str, str], body: str) -> None:
    with _WEBFETCH_CACHE_LOCK:
        _WEBFETCH_CACHE[url] = (status, headers, body)
        while len(_WEBFETCH_CACHE) > _WEBFETCH_CACHE_MAX:
            _WEBFETCH_CACHE.popitem(last=False)


def _tool_webfetch(params: dict[str, Any]) -> tuple[str, bool]:
    url = params.get("url", "")
    if not isinstance(url, str) or not (
        url.startswith("http://") or url.startswith("https://")
    ):
        return "ERROR: url must start with http:// or https://", True

    method = params.get("method", "GET")
    if type(method) is not str or method.upper() not in (
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "HEAD",
        "PATCH",
    ):
        return (
            "ERROR: method must be one of GET, POST, PUT, DELETE, HEAD, PATCH",
            True,
        )
    method = method.upper()

    raw = params.get("raw", False)
    if type(raw) is not bool:
        return "ERROR: raw must be a boolean", True

    extra_headers = params.get("headers", {}) or {}
    if not isinstance(extra_headers, dict):
        return json.dumps(
            {"url": url, "error": "headers must be a dict"}, indent=2
        ), True

    request_body = params.get("body")

    raw_timeout = params.get("timeout", 30)
    if type(raw_timeout) is not int:
        return "ERROR: timeout must be an integer", True
    timeout = max(1, min(raw_timeout, 120))

    max_redirects = params.get("max_redirects", 5)
    if type(max_redirects) is not int:
        return "ERROR: max_redirects must be an integer", True
    max_redirects = max(0, min(max_redirects, 20))

    user_agent = params.get("user_agent", DEFAULT_USER_AGENT)

    domain = urllib.parse.urlparse(url).hostname or ""
    cache_key = f"{method}:{url}:{request_body or ''}"

    if method == "GET" and not raw:
        cached = _cache_get(cache_key)
        if cached is not None:
            status, headers, body = cached
            return (
                json.dumps(
                    {
                        "url": url,
                        "status": status,
                        "headers": headers,
                        "body": body,
                        "cached": True,
                    },
                    indent=2,
                ),
                False,
            )

    _rate_limit_domain(domain)

    headers = {"User-Agent": user_agent}
    headers.update(extra_headers)

    try:
        s = requests.Session()
        s.max_redirects = max_redirects

        resp = s.request(
            method=method,
            url=url,
            headers=headers,
            data=request_body,
            timeout=timeout,
            allow_redirects=max_redirects > 0,
        )

        resp_headers = dict(resp.headers)
        resp_text = resp.text

        max_body = 100 * 1024 if not raw else 1024 * 1024
        body = resp_text[:max_body]
        truncated = len(resp_text) > max_body

        max_body_kb = max_body // 1024
        result: dict[str, Any] = {
            "url": url,
            "status": resp.status_code,
            "headers": resp_headers,
            "body": body,
        }
        if truncated:
            result["truncated"] = True
            if raw:
                result["note"] = f"Response body truncated at {max_body_kb}KB."
            else:
                result["note"] = (
                    f"Response body truncated at {max_body_kb}KB."
                    " Use raw=true to get full content."
                )

        if method == "GET" and resp.status_code < 400:
            _cache_set(cache_key, resp.status_code, resp_headers, body)

        return json.dumps(result, indent=2), resp.status_code >= 400

    except requests.Timeout:
        return json.dumps({"url": url, "error": "Request timed out"}, indent=2), True
    except requests.ConnectionError as exc:
        return json.dumps(
            {"url": url, "error": f"Connection error: {exc}"}, indent=2
        ), True
    except requests.RequestException as exc:
        return json.dumps(
            {"url": url, "error": f"Request failed: {exc}"}, indent=2
        ), True


def _websearch_rate_limit() -> None:
    global _WEBSEARCH_LAST_CALL
    with _WEBSEARCH_RATE_LIMIT_LOCK:
        now = time.monotonic()
        elapsed = now - _WEBSEARCH_LAST_CALL
        need_sleep = max(0.0, _WEBSEARCH_MIN_INTERVAL - elapsed)
        if need_sleep > 0:
            time.sleep(need_sleep)
        _WEBSEARCH_LAST_CALL = time.monotonic()


def _websearch_cache_get(key: str) -> str | None:
    with _WEBSEARCH_CACHE_LOCK:
        entry = _WEBSEARCH_CACHE.get(key)
        if entry is not None:
            _WEBSEARCH_CACHE.move_to_end(key)
        return entry


def _websearch_cache_set(key: str, value: str) -> None:
    with _WEBSEARCH_CACHE_LOCK:
        _WEBSEARCH_CACHE[key] = value
        while len(_WEBSEARCH_CACHE) > _WEBSEARCH_CACHE_MAX:
            _WEBSEARCH_CACHE.popitem(last=False)


def _parse_ddg_results(html_text: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    blocks = re.findall(
        r"<!-- This is the visible part -->(.*?)</div>\s*</div>",
        html_text,
        re.DOTALL,
    )
    for block in blocks:
        title_match = re.search(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        snippet_match = re.search(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        if not title_match:
            continue
        href = title_match.group(1)
        title = re.sub(r"<.*?>", "", title_match.group(2)).strip()
        snippet = (
            re.sub(r"<.*?>", "", snippet_match.group(1)).strip()
            if snippet_match
            else ""
        )
        if "uddg=" in href:
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            href = urllib.parse.unquote_plus(qs.get("uddg", [href])[0])
        results.append({"title": title, "href": href, "body": snippet})
    return results


def _tool_websearch(params: dict[str, Any]) -> tuple[str, bool]:
    query = params.get("query", "")
    if not isinstance(query, str) or not query:
        return "ERROR: query is required and must be a non-empty string", True

    region = params.get("region", "wt-wt")
    timelimit = params.get("timelimit")
    page = params.get("page", 1)
    if type(page) is not int or page < 1:
        return "ERROR: page must be a positive integer", True
    max_results = params.get("max_results", 10)
    if type(max_results) is not int or max_results < 1:
        return "ERROR: max_results must be a positive integer", True
    max_results = min(max_results, 50)
    timeout = params.get("timeout", 10)
    if type(timeout) is not int or timeout < 1 or timeout > 30:
        return "ERROR: timeout must be an integer between 1 and 30", True

    user_agent = params.get("user_agent", DEFAULT_USER_AGENT)
    safe_search = params.get("safe_search", "off")

    cache_key = f"{query}|{region}|{timelimit or ''}|{page}|{safe_search}"
    cached = _websearch_cache_get(cache_key)
    if cached is not None:
        data = json.loads(cached)
        return json.dumps(
            {"results": data[:max_results], "cached": True}, indent=2
        ), False

    _websearch_rate_limit()

    payload: dict[str, str] = {"q": query, "b": "", "l": region}
    if page > 1:
        payload["s"] = str(15 * (page - 1))
    if timelimit:
        payload["df"] = timelimit
    if safe_search == "on":
        payload["kp"] = "-2"
    elif safe_search == "moderate":
        payload["kp"] = "-1"
    elif safe_search == "off":
        payload["kp"] = "1"

    headers = {"User-Agent": user_agent}

    try:
        resp = requests.post(
            _WEBSEARCH_URL,
            data=payload,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.Timeout:
        return json.dumps(
            {"query": query, "error": "Request timed out"}, indent=2
        ), True
    except requests.ConnectionError as exc:
        return json.dumps(
            {"query": query, "error": f"Connection error: {exc}"}, indent=2
        ), True
    except requests.RequestException as exc:
        return json.dumps(
            {"query": query, "error": f"Request failed: {exc}"}, indent=2
        ), True

    # Strip tags from snippet text in the raw HTML before parsing
    raw_results = _parse_ddg_results(resp.text)

    _websearch_cache_set(cache_key, json.dumps(raw_results))

    return json.dumps(
        {"query": query, "results": raw_results[:max_results]},
        indent=2,
    ), False


HANDLERS: dict[str, Any] = {
    "Bash": _tool_bash,
    "CargoFetch": _tool_cargo_fetch,
    "ReadFile": _tool_read,
    "Edit": _tool_edit,
    "WriteFile": _tool_write,
    "Glob": _tool_glob,
    "Grep": _tool_grep,
    "InsertFile": _tool_insert,
    "ListDir": _tool_list_dir,
    "ApplyPatch": _tool_patch,
    "ReadMulti": _tool_read_multi,
    "FileMetadata": _tool_metadata,
    "UvSync": _tool_uv_sync,
    "UvPythonInstall": _tool_uv_python_install,
    "WebFetch": _tool_webfetch,
    "WebSearch": _tool_websearch,
}
HANDLERS.update(_load_custom_tools())


def run_tool(name: str, params: dict[str, Any]) -> tuple[str, bool]:
    handler = HANDLERS.get(name)
    if not handler:
        logger.warning("Unknown tool requested: %s", name)
        return f"Unknown tool: {name}", True
    try:
        logger.debug("Running tool: %s", name)
        result = handler(params)
        return result  # type: ignore[no-any-return]
    except Exception as exc:
        logger.exception("Tool %s execution failed", name)
        return f"Tool execution failed unexpectedly: {str(exc)}", True


def handle_connection(conn: socket.socket) -> None:
    try:
        with conn:
            header = b""
            while True:
                while len(header) < 4:
                    chunk = conn.recv(4 - len(header))
                    if not chunk:
                        return
                    header += chunk
                msg_len = struct.unpack(">I", header)[0]
                header = b""
                if msg_len > _MAX_BUFFER_SIZE:
                    logger.warning("Request too large: %d bytes", msg_len)
                    _send_response(conn, {"error": "Request too large"})
                    continue
                body = b""
                while len(body) < msg_len:
                    chunk = conn.recv(min(65536, msg_len - len(body)))
                    if not chunk:
                        return
                    body += chunk
                try:
                    request = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON in request")
                    _send_response(conn, {"error": "Invalid JSON"})
                    continue

                command = request.get("name")
                if not command:
                    logger.warning("Request missing 'name' field")
                    _send_response(conn, {"error": "Missing 'name' field"})
                    continue

                with _TOOL_LOCK:
                    result_text, is_error = run_tool(command, request)
                _send_response(
                    conn,
                    {
                        "result": result_text,
                        "error": is_error,
                    },
                )
    finally:
        _CONNECTION_SEMAPHORE.release()


def main() -> None:
    args = sys.argv[1:]
    verbose = False
    debug = False
    while args and args[0].startswith("-"):
        if args[0] == "-v":
            verbose = True
        elif args[0] == "-vv":
            debug = True
        else:
            print(
                f"Usage: {sys.argv[0]} [-v] [-vv] <unix-socket-path>", file=sys.stderr
            )
            sys.exit(1)
        args = args[1:]
    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} [-v] [-vv] <unix-socket-path>", file=sys.stderr)
        sys.exit(1)

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    socket_path = args[0]
    socket_p = Path(socket_path)

    if socket_p.exists():
        socket_p.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    socket_p.chmod(0o600)
    server.listen(5)
    logger.info("Listening on %s", socket_path)

    signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(0))

    try:
        while True:  # pragma: no cover
            conn, _ = server.accept()
            if not _CONNECTION_SEMAPHORE.acquire(blocking=False):
                conn.close()
                continue
            try:
                thread = threading.Thread(
                    target=handle_connection, args=(conn,), daemon=True
                )
                thread.start()
            except Exception:
                _CONNECTION_SEMAPHORE.release()
                raise
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if socket_p.exists():  # pragma: no branch
            socket_p.unlink()


if __name__ == "__main__":  # pragma: no branch
    main()
