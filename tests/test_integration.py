# ruff: noqa: ARG002
"""Full integration tests between tools and sandbox_worker via Unix socket.

These tests start a real sandbox-worker server in a background thread and
connect real tool instances to it, exercising the complete socket-based
communication path for every tool.
"""

import http.server
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a stream socket, looping until full."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly")
        buf += chunk
    return buf


pytestmark = pytest.mark.accurate_cov


@pytest.fixture(scope="module")
def socket_path_server(tmp_path_factory) -> str:
    """Provide a temporary Unix socket path for sandbox_server."""
    fn = tmp_path_factory.mktemp("sandbox_server") / "server.sock"
    return str(fn)


@pytest.fixture(scope="module")
def sandbox_server(socket_path_server: str):
    worker_path = (
        Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--branch",
            str(worker_path),
            socket_path_server,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    for _ in range(100):
        p = Path(socket_path_server)
        if p.exists():
            mode = p.stat().st_mode & 0o777
            if mode == 0o600:
                break
        time.sleep(0.05)
    assert not proc.poll(), proc.stdout.read().decode("utf-8")
    yield proc
    proc.send_signal(2)  # SIGINT for graceful coverage flush
    rc = proc.wait(timeout=5)
    assert rc == 0


# ===================================================================
# Bash integration tests
# ===================================================================


class TestBashIntegration:
    def test_echo(self, sandbox_server, socket_path_server):
        from tiz.tools.bash import Bash

        tool = Bash(socket_path_server)
        result = tool.run({"command": "echo hello_world"})
        assert "hello_world" in result

    def test_pwd(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.bash import Bash

        tool = Bash(socket_path_server)
        result = tool.run({"command": "pwd", "cwd": str(tmp_path)})
        assert str(tmp_path) in result

    def test_env_override(self, sandbox_server, socket_path_server):
        from tiz.tools.bash import Bash

        tool = Bash(socket_path_server)
        result = tool.run({"command": "echo $MYVAR", "env": {"MYVAR": "testval"}})
        assert "testval" in result

    def test_timeout(self, sandbox_server, socket_path_server):
        from tiz.tools.bash import Bash

        tool = Bash(socket_path_server)
        result = tool.run({"command": "sleep 10", "timeout": 1})
        assert "timed out" in result

    def test_exit_code(self, sandbox_server, socket_path_server):
        from tiz.tools.bash import Bash

        tool = Bash(socket_path_server)
        result = tool.run({"command": "exit 42"})
        assert "exit code: 42" in result


# ===================================================================
# ReadFile integration tests
# ===================================================================


class TestReadFileIntegration:
    def test_read_full(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.readfile import ReadFile

        f = tmp_path / "file.txt"
        f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        tool = ReadFile(socket_path_server)
        result = tool.run({"path": str(f)})
        assert result == "1\talpha\n2\tbeta\n3\tgamma"

    def test_read_range(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.readfile import ReadFile

        f = tmp_path / "file.txt"
        f.write_text("one\ntwo\nthree\n", encoding="utf-8")
        tool = ReadFile(socket_path_server)
        result = tool.run({"path": str(f), "view_range": [2, 3]})
        assert result == "2\ttwo\n3\tthree"

    def test_read_not_found(self, sandbox_server, socket_path_server):
        from tiz.tools.readfile import ReadFile

        tool = ReadFile(socket_path_server)
        result = tool.run({"path": "/nonexistent_xyz.txt"})
        assert "ERROR: file not found" in result


# ===================================================================
# WriteFile integration tests
# ===================================================================


class TestWriteFileIntegration:
    def test_write_new_file(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.writefile import WriteFile

        tool = WriteFile(socket_path_server)
        target = tmp_path / "new.txt"
        result = tool.run({"path": str(target), "contents": "hello integration"})
        assert "Wrote" in result
        assert target.read_text(encoding="utf-8") == "hello integration"

    def test_write_nested_dirs(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.writefile import WriteFile

        tool = WriteFile(socket_path_server)
        target = tmp_path / "a" / "b" / "c.txt"
        result = tool.run({"path": str(target), "contents": "nested"})
        assert "Wrote" in result
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "nested"

    def test_write_empty(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.writefile import WriteFile

        tool = WriteFile(socket_path_server)
        target = tmp_path / "empty.txt"
        result = tool.run({"path": str(target), "contents": ""})
        assert "Wrote" in result
        assert target.exists()
        assert target.read_text(encoding="utf-8") == ""


# ===================================================================
# Edit integration tests
# ===================================================================


class TestEditIntegration:
    def test_single_replace(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.edit import Edit

        f = tmp_path / "edit.txt"
        f.write_text("hello world\n", encoding="utf-8")
        tool = Edit(socket_path_server)
        result = tool.run(
            {
                "path": str(f),
                "old_string": "world",
                "new_string": "integration",
            }
        )
        assert "Edited" in result
        assert f.read_text(encoding="utf-8") == "hello integration\n"

    def test_replace_all(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.edit import Edit

        f = tmp_path / "edit.txt"
        f.write_text("a b a b a\n", encoding="utf-8")
        tool = Edit(socket_path_server)
        result = tool.run(
            {
                "path": str(f),
                "old_string": "a",
                "new_string": "X",
                "expected_replacements": -1,
            }
        )
        assert "Edited" in result
        assert f.read_text(encoding="utf-8") == "X b X b X\n"

    def test_not_found(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.edit import Edit

        f = tmp_path / "edit.txt"
        f.write_text("hello\n", encoding="utf-8")
        tool = Edit(socket_path_server)
        result = tool.run(
            {
                "path": str(f),
                "old_string": "missing",
                "new_string": "x",
            }
        )
        assert "ERROR: old_string not found" in result

    def test_file_not_found(self, sandbox_server, socket_path_server):
        from tiz.tools.edit import Edit

        tool = Edit(socket_path_server)
        result = tool.run(
            {
                "path": "/nonexistent.txt",
                "old_string": "a",
                "new_string": "b",
            }
        )
        assert "ERROR: file not found" in result


# ===================================================================
# Glob integration tests
# ===================================================================


class TestGlobIntegration:
    def test_glob_py_files(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.glob_tool import Glob

        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "b.txt").write_text("", encoding="utf-8")
        tool = Glob(socket_path_server)
        result = tool.run({"pattern": "*.py", "path": str(tmp_path)})
        assert result == str(tmp_path / "a.py")

    def test_glob_no_matches(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.glob_tool import Glob

        tool = Glob(socket_path_server)
        result = tool.run({"pattern": "*.xyz", "path": str(tmp_path)})
        assert result == "No matches"

    def test_glob_dir_not_found(self, sandbox_server, socket_path_server):
        from tiz.tools.glob_tool import Glob

        tool = Glob(socket_path_server)
        result = tool.run({"pattern": "*", "path": "/nonexistent_dir"})
        assert "ERROR: directory not found" in result


# ===================================================================
# Grep integration tests
# ===================================================================


class TestGrepIntegration:
    def test_grep_literal(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.grep import Grep

        f = tmp_path / "search.txt"
        f.write_text("hello world\nfoo bar\n", encoding="utf-8")
        tool = Grep(socket_path_server)
        result = tool.run({"pattern": "hello", "path": str(tmp_path), "regex": False})
        assert result == f"{f}:1:hello world"

    def test_grep_no_matches(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.grep import Grep

        f = tmp_path / "search.txt"
        f.write_text("nothing here\n", encoding="utf-8")
        tool = Grep(socket_path_server)
        result = tool.run({"pattern": "zzz", "path": str(tmp_path), "regex": False})
        assert result == "No matches"

    def test_grep_case_insensitive(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.grep import Grep

        f = tmp_path / "search.txt"
        f.write_text("HELLO\nworld\n", encoding="utf-8")
        tool = Grep(socket_path_server)
        result = tool.run(
            {
                "pattern": "hello",
                "path": str(tmp_path),
                "regex": False,
                "case_insensitive": True,
            }
        )
        assert result == f"{f}:1:HELLO"


# ===================================================================
# InsertFile integration tests
# ===================================================================


class TestInsertFileIntegration:
    def test_insert_at_end(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.insertfile import InsertFile

        f = tmp_path / "insert.txt"
        f.write_text("line1\nline2\n", encoding="utf-8")
        tool = InsertFile(socket_path_server)
        result = tool.run({"path": str(f), "content": "line3", "line_number": -1})
        assert "Inserted" in result
        content = f.read_text(encoding="utf-8")
        assert content == "line1\nline2\nline3\n"

    def test_insert_at_line(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.insertfile import InsertFile

        f = tmp_path / "insert.txt"
        f.write_text("line1\nline2\n", encoding="utf-8")
        tool = InsertFile(socket_path_server)
        result = tool.run({"path": str(f), "content": "inserted", "line_number": 2})
        assert "Inserted" in result
        content = f.read_text(encoding="utf-8")
        assert content == "line1\ninserted\nline2\n"

    def test_insert_file_not_found(self, sandbox_server, socket_path_server):
        from tiz.tools.insertfile import InsertFile

        tool = InsertFile(socket_path_server)
        result = tool.run({"path": "/nonexistent.txt", "content": "new"})
        assert "ERROR: file not found" in result


# ===================================================================
# ListDir integration tests
# ===================================================================


class TestListDirIntegration:
    def test_list_dir_flat(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.listdir import ListDir

        (tmp_path / "file.txt").write_text("", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path)})
        entries = json.loads(result)
        names = {e["name"] for e in entries}
        assert len(names) == 2
        assert "file.txt" in names
        assert "subdir" in names

    def test_list_dir_recursive(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.listdir import ListDir

        (tmp_path / "a.txt").write_text("", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path), "recursive": True})
        entries = json.loads(result)
        names = {e["name"] for e in entries}
        assert len(names) == 3
        assert "a.txt" in names
        assert "b.txt" in names
        assert "sub" in names

    def test_list_dir_hidden_excluded(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        (tmp_path / ".hidden").write_text("", encoding="utf-8")
        (tmp_path / "visible").write_text("", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path)})
        entries = json.loads(result)
        names = {e["name"] for e in entries}
        assert len(names) == 1
        assert "visible" in names

    def test_list_dir_show_hidden(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.listdir import ListDir

        (tmp_path / ".hidden").write_text("", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path), "show_hidden": True})
        entries = json.loads(result)
        names = {e["name"] for e in entries}
        assert len(names) == 1
        assert ".hidden" in names

    def test_list_dir_not_found(self, sandbox_server, socket_path_server):
        from tiz.tools.listdir import ListDir

        tool = ListDir(socket_path_server)
        result = tool.run({"path": "/nonexistent_dir_xyz"})
        assert "ERROR: directory not found" in result


# ===================================================================
# ApplyPatch integration tests
# ===================================================================


class TestApplyPatchIntegration:
    def test_patch_invalid(self, sandbox_server, socket_path_server):
        from tiz.tools.applypatch import ApplyPatch

        tool = ApplyPatch(socket_path_server)
        result = tool.run({"patch": "not a valid patch at all"})
        assert "exit code" in result or "Error" in result


# ===================================================================
# ReadMulti integration tests
# ===================================================================


class TestReadMultiIntegration:
    def test_read_multiple(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.readmulti import ReadMulti

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content_a", encoding="utf-8")
        f2.write_text("content_b", encoding="utf-8")
        tool = ReadMulti(socket_path_server)
        result = tool.run({"paths": [str(f1), str(f2)]})
        entries = json.loads(result)
        assert len(entries) == 2
        assert entries[0]["content"] == "content_a"
        assert entries[1]["content"] == "content_b"

    def test_read_mixed_found_not_found(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.readmulti import ReadMulti

        f = tmp_path / "exists.txt"
        f.write_text("data", encoding="utf-8")
        tool = ReadMulti(socket_path_server)
        result = tool.run({"paths": [str(f), "/nonexistent_file_xyz"]})
        assert result.startswith("ERROR: ")
        json_part = result[len("ERROR: ") :]
        entries = json.loads(json_part)
        assert len(entries) == 2
        assert entries[0]["content"] == "data"
        assert entries[1]["content"] is None


# ===================================================================
# FileMetadata integration tests
# ===================================================================


class TestFileMetadataIntegration:
    def test_metadata_file(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.filemetadata import FileMetadata

        f = tmp_path / "meta.txt"
        f.write_text("hello", encoding="utf-8")
        tool = FileMetadata(socket_path_server)
        result = tool.run({"path": str(f)})
        data = json.loads(result)
        assert data["exists"] is True
        assert data["type"] == "file"
        assert data["size"] == 5
        assert data["error"] is None

    def test_metadata_directory(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.filemetadata import FileMetadata

        tool = FileMetadata(socket_path_server)
        result = tool.run({"path": str(tmp_path)})
        data = json.loads(result)
        assert data["exists"] is True
        assert data["type"] == "directory"

    def test_metadata_not_found(self, sandbox_server, socket_path_server):
        from tiz.tools.filemetadata import FileMetadata

        tool = FileMetadata(socket_path_server)
        result = tool.run({"path": "/nonexistent_meta_xyz"})
        data = json.loads(result)
        assert data["exists"] is False
        assert "not found" in data["error"]


# ===================================================================
# handle_connection integration tests
# ===================================================================


class TestHandleConnectionIntegration:
    def test_invalid_json(self, sandbox_server, socket_path_server):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(socket_path_server)
            bad = b"not json at all\n"
            sock.sendall(struct.pack(">I", len(bad)) + bad)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            response = json.loads(data.decode("utf-8"))
            assert "error" in response
        finally:
            sock.close()

    def test_missing_name_field(self, sandbox_server, socket_path_server):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(socket_path_server)
            msg = json.dumps({"params": {}}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            response = json.loads(data.decode("utf-8"))
            assert "error" in response
        finally:
            sock.close()

    def test_unknown_tool(self, sandbox_server, socket_path_server):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(socket_path_server)
            msg = (
                json.dumps({"name": "NonExistentTool", "params": {}}).encode("utf-8")
                + b"\n"
            )
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            response = json.loads(data.decode("utf-8"))
            assert "Unknown tool" in response["result"]
            assert response["error"] is True
        finally:
            sock.close()

    def test_request_too_large(self, sandbox_server, socket_path_server):
        """Send a message whose length prefix exceeds MAX_BUFFER_SIZE."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(socket_path_server)
            huge_len = 1024 * 1024 + 1  # > _MAX_BUFFER_SIZE
            sock.sendall(struct.pack(">I", huge_len))
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            response = json.loads(data.decode("utf-8"))
            assert "error" in response
        finally:
            sock.close()


# ===================================================================
# Additional coverage tests for sandbox_worker.py
# ===================================================================


class TestReadFileCoverage:
    def test_read_file_too_large(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.readfile import ReadFile

        f = tmp_path / "large.txt"
        f.write_text("x" * (1024 * 1024 + 1), encoding="utf-8")
        tool = ReadFile(socket_path_server)
        result = tool.run({"path": str(f)})
        assert "ERROR: file too large" in result

    def test_read_view_range_bad_type(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.readfile import ReadFile

        f = tmp_path / "file.txt"
        f.write_text("hello\n", encoding="utf-8")
        tool = ReadFile(socket_path_server)
        result = tool.run({"path": str(f), "view_range": "bad"})
        assert "ERROR: view_range must be a list of two integers" in result

    def test_read_view_range_bad_length(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.readfile import ReadFile

        f = tmp_path / "file.txt"
        f.write_text("hello\n", encoding="utf-8")
        tool = ReadFile(socket_path_server)
        result = tool.run({"path": str(f), "view_range": [1, 2, 3]})
        assert "ERROR: view_range must be a list of two integers" in result

    def test_read_view_range_non_int(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.readfile import ReadFile

        f = tmp_path / "file.txt"
        f.write_text("hello\n", encoding="utf-8")
        tool = ReadFile(socket_path_server)
        result = tool.run({"path": str(f), "view_range": ["a", 2]})
        assert "ERROR: view_range must contain integers only" in result

    def test_read_view_range_negative(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.readfile import ReadFile

        f = tmp_path / "file.txt"
        f.write_text("hello\n", encoding="utf-8")
        tool = ReadFile(socket_path_server)
        result = tool.run({"path": str(f), "view_range": [0, 2]})
        assert "ERROR: view_range values must be positive" in result

    def test_read_view_range_start_gt_end(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.readfile import ReadFile

        f = tmp_path / "file.txt"
        f.write_text("hello\nworld\n", encoding="utf-8")
        tool = ReadFile(socket_path_server)
        result = tool.run({"path": str(f), "view_range": [3, 1]})
        assert "ERROR: view_range[0] must be <= view_range[1]" in result


class TestEditCoverage:
    def test_edit_expected_replacements_mismatch(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.edit import Edit

        f = tmp_path / "edit.txt"
        f.write_text("a b a\n", encoding="utf-8")
        tool = Edit(socket_path_server)
        result = tool.run(
            {
                "path": str(f),
                "old_string": "a",
                "new_string": "X",
                "expected_replacements": 5,
            }
        )
        assert "expected 5 occurrences but found 2" in result

    def test_edit_multiple_matches_default(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.edit import Edit

        f = tmp_path / "edit.txt"
        f.write_text("a b a\n", encoding="utf-8")
        tool = Edit(socket_path_server)
        result = tool.run(
            {
                "path": str(f),
                "old_string": "a",
                "new_string": "X",
            }
        )
        assert "ERROR: old_string matches multiple locations" in result

    def test_edit_expected_replacements_non_one(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.edit import Edit

        f = tmp_path / "edit.txt"
        f.write_text("a b a\n", encoding="utf-8")
        tool = Edit(socket_path_server)
        result = tool.run(
            {
                "path": str(f),
                "old_string": "a",
                "new_string": "X",
                "expected_replacements": 2,
            }
        )
        assert "Edited" in result
        assert f.read_text(encoding="utf-8") == "X b X\n"


class TestGlobCoverage:
    def test_glob_skips_git_dir(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.glob_tool import Glob

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("", encoding="utf-8")
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        tool = Glob(socket_path_server)
        result = tool.run({"pattern": "*.py", "path": str(tmp_path)})
        assert result == str(tmp_path / "a.py")

    def test_glob_over_100_matches(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.glob_tool import Glob

        for i in range(105):
            (tmp_path / f"file_{i}.txt").write_text("", encoding="utf-8")
        tool = Glob(socket_path_server)
        result = tool.run({"pattern": "*.txt", "path": str(tmp_path)})
        lines = result.strip().split("\n")
        assert len(lines) <= 100


class TestGrepCoverage:
    def test_grep_with_glob_filter(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.grep import Grep

        f = tmp_path / "search.py"
        f.write_text("hello world\n", encoding="utf-8")
        (tmp_path / "other.txt").write_text("hello world\n", encoding="utf-8")
        tool = Grep(socket_path_server)
        result = tool.run(
            {
                "pattern": "hello",
                "path": str(tmp_path),
                "glob": "*.py",
            }
        )
        assert result == f"{f}:1:hello world"

    def test_grep_with_regex(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.grep import Grep

        f = tmp_path / "search.txt"
        f.write_text("hello123\n", encoding="utf-8")
        tool = Grep(socket_path_server)
        result = tool.run(
            {
                "pattern": r"hello\d+",
                "path": str(tmp_path),
                "regex": True,
            }
        )
        assert result == f"{f}:1:hello123"


class TestInsertFileCoverage:
    def test_insert_line_out_of_range(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.insertfile import InsertFile

        f = tmp_path / "insert.txt"
        f.write_text("line1\nline2\n", encoding="utf-8")
        tool = InsertFile(socket_path_server)
        result = tool.run({"path": str(f), "content": "new", "line_number": 100})
        assert "line_number 100 out of range" in result

    def test_insert_multiple_lines(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.insertfile import InsertFile

        f = tmp_path / "insert.txt"
        f.write_text("line1\nline3\n", encoding="utf-8")
        tool = InsertFile(socket_path_server)
        result = tool.run(
            {
                "path": str(f),
                "content": "line2a\nline2b",
                "line_number": 2,
            }
        )
        assert "Inserted 2 line(s)" in result
        content = f.read_text(encoding="utf-8")
        assert content == "line1\nline2a\nline2b\nline3\n"


class TestListDirCoverage:
    def test_list_dir_recursive_skips_git(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("", encoding="utf-8")
        (tmp_path / "a.txt").write_text("", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path), "recursive": True})
        entries = json.loads(result)
        names = {e["name"] for e in entries}
        assert len(names) == 1
        assert "a.txt" in names
        assert ".git" not in names
        assert "config" not in names

    def test_list_dir_recursive_hidden_excluded(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        (tmp_path / ".hidden").write_text("", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path), "recursive": True})
        entries = json.loads(result)
        names = {e["name"] for e in entries}
        assert len(names) == 1
        assert ".hidden" not in names
        assert "visible.txt" in names

    def test_list_dir_recursive_with_symlink(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        target = tmp_path / "target.txt"
        target.write_text("data", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path), "recursive": True})
        entries = json.loads(result)
        names = {e["name"]: e for e in entries}
        assert len(names) == 2
        assert names["link.txt"]["type"] == "symlink"
        assert names["target.txt"]["type"] == "file"

    def test_list_dir_non_recursive_hidden_excluded(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        (tmp_path / ".hidden").write_text("", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path)})
        entries = json.loads(result)
        names = {e["name"] for e in entries}
        assert len(names) == 1
        assert "visible.txt" in names
        assert ".hidden" not in names

    def test_list_dir_non_recursive_with_symlink(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        target = tmp_path / "target.txt"
        target.write_text("data", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path)})
        entries = json.loads(result)
        names = {e["name"]: e for e in entries}
        assert len(names) == 2
        assert names["link.txt"]["type"] == "symlink"
        assert names["target.txt"]["type"] == "file"


class TestApplyPatchCoverage:
    def test_patch_reverse(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.applypatch import ApplyPatch

        f = tmp_path / "file.txt"
        f.write_text("hello world\n", encoding="utf-8")
        patch_content = (
            "--- a/file.txt\n"
            "+++ b/file.txt\n"
            "@@ -1 +1 @@\n"
            "-hello world\n"
            "+hello integration\n"
        )
        tool = ApplyPatch(socket_path_server)
        result = tool.run(
            {
                "patch": patch_content,
                "cwd": str(tmp_path),
                "strip": 1,
            }
        )
        assert f.read_text(encoding="utf-8") == "hello integration\n"

        reverse_patch = (
            "--- a/file.txt\n"
            "+++ b/file.txt\n"
            "@@ -1 +1 @@\n"
            "-hello world\n"
            "+hello integration\n"
        )
        result = tool.run(
            {
                "patch": reverse_patch,
                "cwd": str(tmp_path),
                "strip": 1,
                "reverse": True,
            }
        )
        assert "patching file file.txt" in result
        assert f.read_text(encoding="utf-8") == "hello world\n"

    def test_patch_with_stderr(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.applypatch import ApplyPatch

        f = tmp_path / "file.txt"
        f.write_text("hello world\n", encoding="utf-8")
        patch_content = (
            "--- a/file.txt\n"
            "+++ b/file.txt\n"
            "@@ -1 +1 @@\n"
            "-hello world\n"
            "+hello integration\n"
        )
        tool = ApplyPatch(socket_path_server)
        result = tool.run(
            {
                "patch": patch_content,
                "cwd": str(tmp_path),
                "strip": 1,
            }
        )
        assert "patching file file.txt" in result
        assert "hello integration" in f.read_text(encoding="utf-8")


class TestReadMultiCoverage:
    def test_read_multi_paths_not_list(self, sandbox_server, socket_path_server):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(socket_path_server)
            msg = (
                json.dumps({"name": "ReadMulti", "paths": "not_a_list"}).encode("utf-8")
                + b"\n"
            )
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            response = json.loads(data.decode("utf-8"))
            assert "ERROR: paths must be an array" in response["result"]
        finally:
            sock.close()

    def test_read_multi_file_too_large(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.readmulti import ReadMulti

        f = tmp_path / "large.txt"
        f.write_text("x" * (1024 * 1024 + 1), encoding="utf-8")
        tool = ReadMulti(socket_path_server)
        result = tool.run({"paths": [str(f)]})
        assert result.startswith("ERROR: ")
        json_part = result[len("ERROR: ") :]
        entries = json.loads(json_part)
        assert entries[0]["content"] is None
        assert "File too large" in entries[0]["error"]


class TestFileMetadataCoverage:
    def test_metadata_symlink(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.filemetadata import FileMetadata

        target = tmp_path / "target.txt"
        target.write_text("data", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        tool = FileMetadata(socket_path_server)
        result = tool.run({"path": str(link)})
        data = json.loads(result)
        assert data["exists"] is True
        assert data["type"] == "symlink"

    def test_metadata_fifo(self, sandbox_server, socket_path_server, tmp_path):
        import os as os_mod

        from tiz.tools.filemetadata import FileMetadata

        fifo_path = tmp_path / "myfifo"
        os_mod.mkfifo(str(fifo_path))
        tool = FileMetadata(socket_path_server)
        result = tool.run({"path": str(fifo_path)})
        data = json.loads(result)
        assert data["exists"] is True
        assert data["type"] is None


class TestHandleConnectionMoreCoverage:
    def test_connection_closed_mid_body(self, sandbox_server, socket_path_server):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(socket_path_server)
        msg = (
            json.dumps({"name": "Bash", "params": {"command": "echo hi"}}).encode(
                "utf-8"
            )
            + b"\n"
        )
        sock.sendall(struct.pack(">I", len(msg)))
        sock.sendall(msg[:5])
        sock.close()
        time.sleep(0.2)


class TestListDirMaxEntries:
    def test_list_dir_recursive_max_entries(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        for i in range(1005):
            (tmp_path / f"file_{i}.txt").write_text("x", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path), "recursive": True})
        entries = json.loads(result)
        assert len(entries) <= 1000

    def test_list_dir_non_recursive_max_entries(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        for i in range(1005):
            (tmp_path / f"file_{i}.txt").write_text("x", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run({"path": str(tmp_path)})
        entries = json.loads(result)
        assert len(entries) <= 1000


class TestOSErrorCoverage:
    """Integration tests covering OSError paths in sandbox_worker handlers."""

    def test_read_oserror(self, sandbox_server, socket_path_server):
        from tiz.tools.readfile import ReadFile

        tool = ReadFile(socket_path_server)
        result = tool.run({"path": "/proc/self/mem"})
        assert "Error reading file" in result

    def test_write_oserror(self, sandbox_server, socket_path_server):
        from tiz.tools.writefile import WriteFile

        tool = WriteFile(socket_path_server)
        result = tool.run({"path": "/dev/full", "contents": "data"})
        assert "Error writing file" in result

    def test_edit_oserror(self, sandbox_server, socket_path_server):
        from tiz.tools.edit import Edit

        tool = Edit(socket_path_server)
        result = tool.run(
            {
                "path": "/proc/self/mem",
                "old_string": "a",
                "new_string": "b",
            }
        )
        assert "Error editing file" in result

    def test_insert_oserror(self, sandbox_server, socket_path_server):
        from tiz.tools.insertfile import InsertFile

        tool = InsertFile(socket_path_server)
        result = tool.run({"path": "/proc/self/mem", "content": "new"})
        assert "Error inserting content" in result

    def test_read_multi_oserror(self, sandbox_server, socket_path_server):
        from tiz.tools.readmulti import ReadMulti

        tool = ReadMulti(socket_path_server)
        result = tool.run({"paths": ["/proc/self/mem"]})
        assert result.startswith("ERROR: ")
        json_part = result[len("ERROR: ") :]
        entries = json.loads(json_part)
        assert entries[0]["content"] is None
        assert entries[0]["error"] is not None


class TestRunToolException:
    """Test the run_tool exception path via socket."""

    def test_handler_type_error(self, sandbox_server, socket_path_server):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(socket_path_server)
            msg = json.dumps({"name": "ReadFile", "path": 123}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            response = json.loads(data.decode("utf-8"))
            assert "Tool execution failed unexpectedly" in response["result"]
            assert response["error"] is True
        finally:
            sock.close()


class TestGrepFallbackIntegration:
    """Test grep fallback when rg is not available."""

    @staticmethod
    def _setup_fallback_env(tmp_path):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        for name in ("which", "grep", "find"):
            src = subprocess.run(
                ["which", name], capture_output=True, text=True, check=True
            ).stdout.strip()
            bindir.joinpath(name).symlink_to(src)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        return env

    @staticmethod
    def _start_fallback_worker(worker_path, sock_path, env):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    @staticmethod
    def _fallback_grep(sock_path, params):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(sock_path)
            msg = json.dumps({"name": "Grep", **params}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    def test_grep_fallback_via_restricted_path(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_fb.sock")
        env = self._setup_fallback_env(tmp_path)
        proc = self._start_fallback_worker(worker_path, sock_path, env)
        try:
            f = tmp_path / "search.txt"
            f.write_text("hello grep fallback\n", encoding="utf-8")
            resp = self._fallback_grep(
                sock_path,
                {
                    "pattern": "hello",
                    "path": str(tmp_path),
                    "regex": False,
                },
            )
            assert "hello grep fallback" in resp["result"]
            assert resp["error"] is False
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_grep_fallback_case_insensitive(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_cs.sock")
        env = self._setup_fallback_env(tmp_path)
        proc = self._start_fallback_worker(worker_path, sock_path, env)
        try:
            f = tmp_path / "search.txt"
            f.write_text("HELLO WORLD\n", encoding="utf-8")
            resp = self._fallback_grep(
                sock_path,
                {
                    "pattern": "hello",
                    "path": str(tmp_path),
                    "regex": False,
                    "case_insensitive": True,
                },
            )
            assert "HELLO WORLD" in resp["result"]
            assert resp["error"] is False
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_grep_fallback_regex(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_re.sock")
        env = self._setup_fallback_env(tmp_path)
        proc = self._start_fallback_worker(worker_path, sock_path, env)
        try:
            f = tmp_path / "search.txt"
            f.write_text("hello123\n", encoding="utf-8")
            resp = self._fallback_grep(
                sock_path,
                {
                    "pattern": r"hello[0-9]",
                    "path": str(tmp_path),
                    "regex": True,
                },
            )
            assert "hello123" in resp["result"]
            assert resp["error"] is False
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_grep_fallback_glob_filter(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_gf.sock")
        env = self._setup_fallback_env(tmp_path)
        proc = self._start_fallback_worker(worker_path, sock_path, env)
        try:
            f = tmp_path / "search.py"
            f.write_text("found in py\n", encoding="utf-8")
            (tmp_path / "search.txt").write_text("found in txt\n", encoding="utf-8")
            resp = self._fallback_grep(
                sock_path,
                {
                    "pattern": "found",
                    "path": str(tmp_path),
                    "glob": "*.py",
                },
            )
            assert f"{f}:1:found in py" == resp["result"]
            assert resp["error"] is False
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_grep_fallback_glob_filter_no_matches(self, tmp_path):
        """Fallback grep with glob_filter and no matching files (line 398)."""
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_gf_nomatch.sock")
        env = self._setup_fallback_env(tmp_path)
        proc = self._start_fallback_worker(worker_path, sock_path, env)
        try:
            # Create a .py file but search only *.txt via glob
            (tmp_path / "search.py").write_text("found in py\n", encoding="utf-8")
            resp = self._fallback_grep(
                sock_path,
                {
                    "pattern": "found",
                    "path": str(tmp_path),
                    "glob": "*.txt",
                },
            )
            assert resp["result"] == "No matches"
            assert resp["error"] is False
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_grep_fallback_glob_filter_case_insensitive(self, tmp_path):
        """Fallback grep with glob_filter + case_insensitive (lines 400->403)."""
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_gf_ci.sock")
        env = self._setup_fallback_env(tmp_path)
        proc = self._start_fallback_worker(worker_path, sock_path, env)
        try:
            f = tmp_path / "search.py"
            f.write_text("HELLO WORLD\n", encoding="utf-8")
            resp = self._fallback_grep(
                sock_path,
                {
                    "pattern": "hello",
                    "path": str(tmp_path),
                    "glob": "*.py",
                    "case_insensitive": True,
                },
            )
            assert "HELLO WORLD" in resp["result"]
            assert resp["error"] is False
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_grep_fallback_glob_filter_regex(self, tmp_path):
        """Fallback grep with glob_filter + regex=True (line 400->402 branch)."""
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_gf_re.sock")
        env = self._setup_fallback_env(tmp_path)
        proc = self._start_fallback_worker(worker_path, sock_path, env)
        try:
            f = tmp_path / "search.py"
            f.write_text("hello123\n", encoding="utf-8")
            resp = self._fallback_grep(
                sock_path,
                {
                    "pattern": r"hello[0-9]",
                    "path": str(tmp_path),
                    "glob": "*.py",
                    "regex": True,
                },
            )
            assert "hello123" in resp["result"]
            assert resp["error"] is False
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_grep_fallback_glob_filter_subprocess_error(self, tmp_path):
        """Fallback grep with glob_filter and broken find (lines 412-413)."""
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_gf_err.sock")
        bindir = tmp_path / "bin_broken_find"
        bindir.mkdir()
        # Symlink which and grep normally
        for name in ("which", "grep"):
            src = subprocess.run(
                ["which", name], capture_output=True, text=True, check=True
            ).stdout.strip()
            bindir.joinpath(name).symlink_to(src)
        # Create a broken find that exits non-zero
        (bindir / "find").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        (bindir / "find").chmod(0o755)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        proc = self._start_fallback_worker(worker_path, sock_path, env)
        try:
            f = tmp_path / "search.txt"
            f.write_text("some content\n", encoding="utf-8")
            resp = self._fallback_grep(
                sock_path,
                {
                    "pattern": "content",
                    "path": str(tmp_path),
                    "glob": "*.txt",
                },
            )
            assert "Error running grep" in resp["result"]
            assert resp["error"] is True
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestSandboxWorkerMain:
    def test_main_wrong_args(self):
        """main() with wrong number of args should exit."""

        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert "Usage" in result.stderr

    def test_main_unknown_flag(self):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                "--bad",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert "Usage" in result.stderr

    def test_main_too_many_args(self):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                "sock1",
                "sock2",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert "Usage" in result.stderr

    def test_main_verbose(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "verbose.sock")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                "-v",
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(100):
            if Path(sock_path).exists():
                break
            time.sleep(0.05)
        proc.send_signal(2)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        stdout = proc.stdout.read().decode("utf-8")
        assert "Listening on" in stdout

    def test_main_debug(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "debug.sock")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                "-vv",
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(100):
            if Path(sock_path).exists():
                break
            time.sleep(0.05)
        proc.send_signal(2)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        stdout = proc.stdout.read().decode("utf-8")
        assert "Listening on" in stdout

    def test_main_existing_socket(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "existing.sock")
        Path(sock_path).touch()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        proc.send_signal(2)
        proc.wait(timeout=5)
        assert proc.returncode == 0


class TestCargoFetchCoverage:
    """Cover _tool_cargo_fetch subprocess and timeout paths (lines 102-125)."""

    @staticmethod
    def _start_worker_with_fake_cargo(
        sock_path: str, fake_cargo_script: str
    ) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        bindir = Path(sock_path).parent / "fake_bin"
        bindir.mkdir(exist_ok=True)
        (bindir / "cargo").write_text(fake_cargo_script, encoding="utf-8")
        (bindir / "cargo").chmod(0o755)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir) + ":" + env.get("PATH", "")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def test_cargo_fetch_missing_path(self, sandbox_server, socket_path_server):
        from tiz.tools.cargofetch import CargoFetch

        tool = CargoFetch(socket_path_server)
        result = tool.run({})
        assert "ERROR" in result
        assert "path" in result.lower()

    def test_cargo_fetch_missing_cargo_toml(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.cargofetch import CargoFetch

        tool = CargoFetch(socket_path_server)
        result = tool.run({"path": str(tmp_path / "nonexistent")})
        assert "ERROR" in result
        assert "Cargo.toml not found" in result

    def test_cargo_fetch_timeout(self, sandbox_server, socket_path_server, tmp_path):
        """Cover TimeoutExpired handler via a slow fake cargo."""
        from tiz.tools.cargofetch import CargoFetch

        sock_path = str(tmp_path / "cargo_timeout.sock")
        proc = self._start_worker_with_fake_cargo(
            sock_path,
            "#!/bin/sh\nsleep 10\necho done\n",
        )
        try:
            project_dir = tmp_path / "cargo_project_t"
            project_dir.mkdir()
            (project_dir / "Cargo.toml").write_text(
                '[package]\nname = "test"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            tool = CargoFetch(sock_path)
            result = tool.run({"path": str(project_dir), "timeout": 1})
            assert "timed out" in result.lower()
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_cargo_fetch_subprocess_ok(self, tmp_path):
        """Cover subprocess.run success path with fake cargo."""
        from tiz.tools.cargofetch import CargoFetch

        sock_path = str(tmp_path / "cargo_ok.sock")
        proc = self._start_worker_with_fake_cargo(
            sock_path,
            "#!/bin/sh\necho 'fetch completed successfully'\n",
        )
        try:
            project_dir = tmp_path / "cargo_project_ok"
            project_dir.mkdir()
            (project_dir / "Cargo.toml").write_text(
                '[package]\nname = "test"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            tool = CargoFetch(sock_path)
            result = tool.run({"path": str(project_dir)})
            assert "fetch completed successfully" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_cargo_fetch_subprocess_error(self, tmp_path):
        """Cover subprocess.run non-zero exit path."""
        from tiz.tools.cargofetch import CargoFetch

        sock_path = str(tmp_path / "cargo_err.sock")
        proc = self._start_worker_with_fake_cargo(
            sock_path,
            "#!/bin/sh\necho 'error message' >&2\nexit 1\n",
        )
        try:
            project_dir = tmp_path / "cargo_project_err"
            project_dir.mkdir()
            (project_dir / "Cargo.toml").write_text(
                '[package]\nname = "test"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            tool = CargoFetch(sock_path)
            result = tool.run({"path": str(project_dir)})
            assert "error message" in result
            assert "exit code: 1" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestUvSyncCoverage:
    """Cover _tool_uv_sync subprocess and timeout paths (lines 129-151)."""

    @staticmethod
    def _start_worker_with_fake_uv(
        sock_path: str, fake_uv_script: str
    ) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        bindir = Path(sock_path).parent / "fake_bin_uv"
        bindir.mkdir(exist_ok=True)
        (bindir / "uv").write_text(fake_uv_script, encoding="utf-8")
        (bindir / "uv").chmod(0o755)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir) + ":" + env.get("PATH", "")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def test_uv_sync_missing_path(self, sandbox_server, socket_path_server):
        from tiz.tools.uvsync import UvSync

        tool = UvSync(socket_path_server)
        result = tool.run({})
        assert "ERROR" in result
        assert "path" in result.lower()

    def test_uv_sync_dir_not_found(self, sandbox_server, socket_path_server, tmp_path):
        from tiz.tools.uvsync import UvSync

        tool = UvSync(socket_path_server)
        result = tool.run({"path": str(tmp_path / "nonexistent_dir")})
        assert "ERROR" in result
        assert "directory not found" in result

    def test_uv_sync_timeout(self, tmp_path):
        """Cover TimeoutExpired handler."""
        from tiz.tools.uvsync import UvSync

        sock_path = str(tmp_path / "uv_sync_timeout.sock")
        proc = self._start_worker_with_fake_uv(
            sock_path,
            "#!/bin/sh\nsleep 10\necho done\n",
        )
        try:
            project_dir = tmp_path / "uv_project_t"
            project_dir.mkdir()
            tool = UvSync(sock_path)
            result = tool.run({"path": str(project_dir), "timeout": 1})
            assert "timed out" in result.lower()
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_uv_sync_subprocess_ok(self, tmp_path):
        """Cover subprocess.run success path."""
        from tiz.tools.uvsync import UvSync

        sock_path = str(tmp_path / "uv_sync_ok.sock")
        proc = self._start_worker_with_fake_uv(
            sock_path,
            "#!/bin/sh\necho 'synced successfully'\n",
        )
        try:
            project_dir = tmp_path / "uv_project_ok"
            project_dir.mkdir()
            tool = UvSync(sock_path)
            result = tool.run({"path": str(project_dir)})
            assert "synced successfully" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_uv_sync_subprocess_error(self, tmp_path):
        """Cover subprocess.run non-zero exit path."""
        from tiz.tools.uvsync import UvSync

        sock_path = str(tmp_path / "uv_sync_err.sock")
        proc = self._start_worker_with_fake_uv(
            sock_path,
            "#!/bin/sh\necho 'sync error' >&2\nexit 1\n",
        )
        try:
            project_dir = tmp_path / "uv_project_err"
            project_dir.mkdir()
            tool = UvSync(sock_path)
            result = tool.run({"path": str(project_dir)})
            assert "sync error" in result
            assert "exit code: 1" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_uv_sync_with_group(self, tmp_path):
        """Cover --group cmd.extend path (line 139)."""
        from tiz.tools.uvsync import UvSync

        sock_path = str(tmp_path / "uv_sync_group.sock")
        proc = self._start_worker_with_fake_uv(
            sock_path,
            '#!/bin/sh\nfor arg in "$@"; do echo "$arg"; done\n',
        )
        try:
            project_dir = tmp_path / "uv_project_group"
            project_dir.mkdir()
            tool = UvSync(sock_path)
            result = tool.run({"path": str(project_dir), "group": ["dev", "test"]})
            assert "--group" in result
            assert "dev" in result
            assert "test" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_uv_sync_with_extra(self, tmp_path):
        """Cover --extra cmd.extend path (line 141)."""
        from tiz.tools.uvsync import UvSync

        sock_path = str(tmp_path / "uv_sync_extra.sock")
        proc = self._start_worker_with_fake_uv(
            sock_path,
            '#!/bin/sh\nfor arg in "$@"; do echo "$arg"; done\n',
        )
        try:
            project_dir = tmp_path / "uv_project_extra"
            project_dir.mkdir()
            tool = UvSync(sock_path)
            result = tool.run(
                {"path": str(project_dir), "extra": ["feature1", "feature2"]}
            )
            assert "--extra" in result
            assert "feature1" in result
            assert "feature2" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestUvPythonInstallCoverage:
    """Cover _tool_uv_python_install subprocess and timeout paths (lines 155-173)."""

    @staticmethod
    def _start_worker_with_fake_uv(
        sock_path: str, fake_uv_script: str
    ) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        bindir = Path(sock_path).parent / "fake_bin_uv_pi"
        bindir.mkdir(exist_ok=True)
        (bindir / "uv").write_text(fake_uv_script, encoding="utf-8")
        (bindir / "uv").chmod(0o755)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir) + ":" + env.get("PATH", "")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def test_uv_python_install_missing_version(
        self, sandbox_server, socket_path_server
    ):
        from tiz.tools.uvpythoninstall import UvPythonInstall

        tool = UvPythonInstall(socket_path_server)
        result = tool.run({})
        assert "ERROR" in result
        assert "version" in result.lower()

    def test_uv_python_install_timeout(self, tmp_path):
        """Cover TimeoutExpired handler."""
        from tiz.tools.uvpythoninstall import UvPythonInstall

        sock_path = str(tmp_path / "uv_pi_timeout.sock")
        proc = self._start_worker_with_fake_uv(
            sock_path,
            "#!/bin/sh\nsleep 10\necho done\n",
        )
        try:
            tool = UvPythonInstall(sock_path)
            result = tool.run({"version": "3.99.0", "timeout": 1})
            assert "timed out" in result.lower()
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_uv_python_install_subprocess_ok(self, tmp_path):
        """Cover subprocess.run success path."""
        from tiz.tools.uvpythoninstall import UvPythonInstall

        sock_path = str(tmp_path / "uv_pi_ok.sock")
        proc = self._start_worker_with_fake_uv(
            sock_path,
            "#!/bin/sh\necho 'installed successfully'\n",
        )
        try:
            tool = UvPythonInstall(sock_path)
            result = tool.run({"version": "3.99.0"})
            assert "installed successfully" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_uv_python_install_subprocess_error(self, tmp_path):
        """Cover subprocess.run non-zero exit path."""
        from tiz.tools.uvpythoninstall import UvPythonInstall

        sock_path = str(tmp_path / "uv_pi_err.sock")
        proc = self._start_worker_with_fake_uv(
            sock_path,
            "#!/bin/sh\necho 'install error' >&2\nexit 1\n",
        )
        try:
            tool = UvPythonInstall(sock_path)
            result = tool.run({"version": "3.99.0"})
            assert "install error" in result
            assert "exit code: 1" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestCheckHasRgNoWhich:
    """Cover _check_has_rg FileNotFoundError (lines 37-38) by removing `which` from PATH."""

    def test_check_has_rg_file_not_found(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "no_which.sock")
        bindir = tmp_path / "bin_no_which"
        bindir.mkdir()
        grep_src = subprocess.run(
            ["which", "grep"], capture_output=True, text=True, check=True
        ).stdout.strip()
        bindir.joinpath("grep").symlink_to(grep_src)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        try:
            from tiz.tools.grep import Grep

            tool = Grep(sock_path)
            f = tmp_path / "search.txt"
            f.write_text("test data\n", encoding="utf-8")
            result = tool.run(
                {"pattern": "test", "path": str(tmp_path), "regex": False}
            )
            assert "test data" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestGlobOSError:
    """Test glob with restricted directory access."""

    def test_glob_no_matches_with_restricted(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.glob_tool import Glob

        subdir = tmp_path / "accessible_dir"
        subdir.mkdir()
        (subdir / "a.txt").write_text("data", encoding="utf-8")
        tool = Glob(socket_path_server)
        result = tool.run({"pattern": "*.py", "path": str(subdir)})
        assert result == "No matches"


class TestListDirOSError:
    """Cover _tool_list_dir OSError handler (lines 328-329) via unreadable directory."""

    def test_list_dir_oserror_unreadable(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        subdir = tmp_path / "unreadable"
        subdir.mkdir()
        (subdir / "file.txt").write_text("data", encoding="utf-8")
        subdir.chmod(0o111)
        try:
            tool = ListDir(socket_path_server)
            result = tool.run({"path": str(subdir)})
            assert "Error listing directory" in result
        finally:
            subdir.chmod(0o755)


class TestPatchOSError:
    """Cover _tool_patch OSError handler (lines 359-360) by removing patch from PATH."""

    def test_patch_oserror(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "no_patch.sock")
        bindir = tmp_path / "empty_bin"
        bindir.mkdir()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.settimeout(5)
                sock.connect(sock_path)
                msg = (
                    json.dumps({"name": "ApplyPatch", "patch": "dummy"}).encode("utf-8")
                    + b"\n"
                )
                sock.sendall(struct.pack(">I", len(msg)) + msg)
                length_data = _recv_exact(sock, 4)
                payload_length = struct.unpack(">I", length_data)[0]
                data = _recv_exact(sock, payload_length)
                response = json.loads(data.decode("utf-8"))
                assert "Error applying patch" in response["result"]
                assert response["error"] is True
            finally:
                sock.close()
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestGrepSubprocessError:
    """Test that grep works with fallback (rg not available)."""

    def test_grep_fallback_no_rg(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "grep_fb2.sock")
        bindir = tmp_path / "bin_no_rg"
        bindir.mkdir()
        for name in ("which", "grep"):
            src = subprocess.run(
                ["which", name], capture_output=True, text=True, check=True
            ).stdout.strip()
            bindir.joinpath(name).symlink_to(src)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        try:
            from tiz.tools.grep import Grep

            tool = Grep(sock_path)
            f = tmp_path / "search.txt"
            f.write_text("hello fallback\n", encoding="utf-8")
            result = tool.run(
                {"pattern": "hello", "path": str(tmp_path), "regex": False}
            )
            assert "hello fallback" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestMetadataOSError:
    """Test metadata with restricted file system scenarios."""

    def test_metadata_unreadable_parent(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.filemetadata import FileMetadata

        parent = tmp_path / "restricted_parent"
        parent.mkdir()
        inner = parent / "inner.txt"
        inner.write_text("data", encoding="utf-8")
        parent.chmod(0o111)
        try:
            tool = FileMetadata(socket_path_server)
            result = tool.run({"path": str(inner)})
            data = json.loads(result)
            assert data["exists"] is True
            assert data["type"] == "file"
        finally:
            parent.chmod(0o755)


class TestSigtermHandler:
    """Cover the SIGTERM signal handler in main()."""

    def test_sigterm_handled(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "sigterm.sock")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=5)
        assert rc == 0


class TestGlobOSErrorCoverage:
    """Cover _tool_glob OSError handler (lines 186-187).

    The OSError handler in _tool_glob is defensive: os.walk() in Python 3.12
    catches scandir errors on the top-level directory and silently returns,
    so the handler cannot be reached via normal filesystem operations.
    This test validates that glob handles inaccessible directories gracefully.
    """

    def test_glob_unreadable_directory_returns_no_matches(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.glob_tool import Glob

        subdir = tmp_path / "locked_dir"
        subdir.mkdir()
        (subdir / "a.txt").write_text("data", encoding="utf-8")
        subdir.chmod(0o000)
        try:
            tool = Glob(socket_path_server)
            result = tool.run({"pattern": "*.txt", "path": str(subdir)})
            assert result == "No matches"
        finally:
            subdir.chmod(0o755)


class TestGrepSubprocessErrorCoverage:
    """Cover _tool_grep SubprocessError handler (lines 224-225).

    The SubprocessError handler in _tool_grep is defensive: subprocess.run()
    with check=False and no timeout never raises SubprocessError, only OSError
    subclasses like FileNotFoundError which propagate to run_tool's handler.
    """

    def test_grep_subprocess_error_no_binaries(self, tmp_path):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "no_grep_rg.sock")
        bindir = tmp_path / "bin_empty"
        bindir.mkdir()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        try:
            from tiz.tools.grep import Grep

            tool = Grep(sock_path)
            f = tmp_path / "search.txt"
            f.write_text("hello\n", encoding="utf-8")
            result = tool.run(
                {"pattern": "hello", "path": str(tmp_path), "regex": False}
            )
            assert "Tool execution failed unexpectedly" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestMetadataIsSymlinkOSError:
    """Cover _tool_metadata is_symlink OSError handler (lines 407-408)."""

    def test_metadata_parent_not_executable(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.filemetadata import FileMetadata

        parent = tmp_path / "noexec_dir"
        parent.mkdir()
        inner = parent / "inner.txt"
        inner.write_text("data", encoding="utf-8")
        parent.chmod(0o666)
        try:
            tool = FileMetadata(socket_path_server)
            result = tool.run({"path": str(inner)})
            assert "Tool execution failed unexpectedly" in result
        finally:
            parent.chmod(0o755)


class TestMetadataLstatOSError:
    """Test metadata for a broken symlink (symlink whose target has been removed)."""

    def test_metadata_broken_symlink(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.filemetadata import FileMetadata

        target = tmp_path / "target.txt"
        target.write_text("data", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        target.unlink()
        tool = FileMetadata(socket_path_server)
        result = tool.run({"path": str(link)})
        data = json.loads(result)
        assert data["exists"] is True
        assert data["type"] == "symlink"
        assert data["size"] is None
        assert data["error"] is None
        assert "created" in data
        assert "modified" in data
        assert "permissions" in data


class TestSandboxWorkerImport:
    """Cover the __name__ == "__main__" branch by importing the module."""

    def test_import_sandbox_worker(self):
        import tiz.sandbox_worker  # noqa: F811

        assert tiz.sandbox_worker._MAX_BUFFER_SIZE == 1024 * 1024


# ===================================================================
# WebFetch integration tests (cover _tool_webfetch, _rate_limit_domain,
# _cache_get, _cache_set)
# ===================================================================


class SimpleHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler for testing WebFetch tool."""

    def do_GET(self):
        if self.path == "/test":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"hello world")
        elif self.path == "/redirect":
            self.send_response(301)
            self.send_header("Location", "/test")
            self.end_headers()
        elif self.path == "/large":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"x" * (200 * 1024))  # > 100KB truncation limit
        elif self.path == "/huge":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", "1050000")
            self.end_headers()
            self.wfile.write(b"x" * 1050000)  # > 1MB truncation limit
        elif self.path == "/status404":
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"not found")
        elif self.path == "/headers_echo":
            body = json.dumps(dict(self.headers)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"echo": body}).encode())

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    """Start a simple HTTP server on localhost and yield the port."""
    server = http.server.HTTPServer(("127.0.0.1", 0), SimpleHTTPHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


class TestWebFetchCoverage:
    """Integration tests for the WebFetch tool covering _tool_webfetch."""

    def _send_request(self, sock_path: str, payload: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(10)
            sock.connect(sock_path)
            msg = json.dumps({"name": "WebFetch", **payload}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    def test_get_success(self, sandbox_server, socket_path_server, http_server):
        port = http_server
        url = f"http://127.0.0.1:{port}/test"
        resp = self._send_request(socket_path_server, {"url": url})
        assert resp["error"] is False
        result = json.loads(resp["result"])
        assert result["url"] == url
        assert result["status"] == 200
        assert result["body"] == "hello world"

    def test_post_with_body(self, sandbox_server, socket_path_server, http_server):
        port = http_server
        url = f"http://127.0.0.1:{port}/test"
        resp = self._send_request(
            socket_path_server,
            {"url": url, "method": "POST", "body": "hello from post"},
        )
        assert resp["error"] is False
        result = json.loads(resp["result"])
        assert result["status"] == 200
        assert json.loads(result["body"])["echo"] == "hello from post"

    def test_caching(self, sandbox_server, socket_path_server, http_server):
        port = http_server
        url = f"http://127.0.0.1:{port}/test"
        # First call - uncached
        resp1 = self._send_request(socket_path_server, {"url": url})
        assert resp1["error"] is False
        result1 = json.loads(resp1["result"])
        assert "cached" not in result1
        # Second call - cached
        resp2 = self._send_request(socket_path_server, {"url": url})
        assert resp2["error"] is False
        result2 = json.loads(resp2["result"])
        assert result2.get("cached") is True

    def test_invalid_url(self, sandbox_server, socket_path_server):
        resp = self._send_request(socket_path_server, {"url": "ftp://invalid"})
        assert resp["error"] is True
        assert "ERROR: url must start with http" in resp["result"]

    def test_connection_refused(self, sandbox_server, socket_path_server):
        """Connect to a port that nothing is listening on."""
        url = "http://127.0.0.1:1/nonexistent"
        resp = self._send_request(socket_path_server, {"url": url})
        assert resp["error"] is True
        result = json.loads(resp["result"])
        assert "error" in result

    def test_redirect_followed(self, sandbox_server, socket_path_server, http_server):
        port = http_server
        url = f"http://127.0.0.1:{port}/redirect"
        resp = self._send_request(socket_path_server, {"url": url})
        assert resp["error"] is False
        result = json.loads(resp["result"])
        assert result["status"] == 200
        assert result["body"] == "hello world"

    def test_404_error_status(self, sandbox_server, socket_path_server, http_server):
        port = http_server
        url = f"http://127.0.0.1:{port}/status404"
        resp = self._send_request(socket_path_server, {"url": url})
        assert resp["error"] is True
        result = json.loads(resp["result"])
        assert result["status"] == 404

    def test_raw_mode(self, sandbox_server, socket_path_server, http_server):
        port = http_server
        url = f"http://127.0.0.1:{port}/large"
        resp = self._send_request(socket_path_server, {"url": url, "raw": True})
        assert resp["error"] is False
        result = json.loads(resp["result"])
        assert "truncated" not in result
        assert len(result["body"]) == 200 * 1024

    def test_truncated_response(self, sandbox_server, socket_path_server, http_server):
        port = http_server
        url = f"http://127.0.0.1:{port}/large"
        resp = self._send_request(socket_path_server, {"url": url, "raw": False})
        assert resp["error"] is False
        result = json.loads(resp["result"])
        assert result.get("truncated") is True
        assert "truncated" in result.get("note", "")

    def test_non_dict_headers(self, sandbox_server, socket_path_server, http_server):
        """Passing non-dict headers should return an error."""
        port = http_server
        url = f"http://127.0.0.1:{port}/test"
        resp = self._send_request(
            socket_path_server, {"url": url, "headers": "invalid"}
        )
        assert resp["error"] is True
        result = json.loads(resp["result"])
        assert result.get("error") == "headers must be a dict"

    def test_custom_user_agent(self, sandbox_server, socket_path_server, http_server):
        port = http_server
        url = f"http://127.0.0.1:{port}/headers_echo"
        resp = self._send_request(
            socket_path_server,
            {"url": url, "user_agent": "TestAgent/1.0"},
        )
        assert resp["error"] is False
        result = json.loads(resp["result"])
        headers = json.loads(result["body"])
        assert headers.get("User-Agent") == "TestAgent/1.0"

    def test_max_redirects_zero(self, sandbox_server, socket_path_server, http_server):
        """Redirects not allowed when max_redirects=0 - should get error."""
        port = http_server
        url = f"http://127.0.0.1:{port}/redirect"
        resp = self._send_request(socket_path_server, {"url": url, "max_redirects": 0})
        assert resp["error"] is True
        result = json.loads(resp["result"])
        assert "error" in result  # Request failed / exceeded 0 redirects

    def test_webfetch_timeout_connection_error(
        self, sandbox_server, socket_path_server
    ):
        """Connect to a private IP that will get connection refused."""
        url = "http://127.0.0.1:1/"
        resp = self._send_request(socket_path_server, {"url": url, "timeout": 2})
        assert resp["error"] is True
        result = json.loads(resp["result"])
        assert "error" in result

    def test_post_not_cached(self, sandbox_server, socket_path_server, http_server):
        """POST requests should not be cached."""
        port = http_server
        url = f"http://127.0.0.1:{port}/test"
        resp = self._send_request(
            socket_path_server, {"url": url, "method": "POST", "body": "data"}
        )
        assert resp["error"] is False
        result = json.loads(resp["result"])
        assert "cached" not in result or not result.get("cached")

    def test_timeout_clamped(self, sandbox_server, socket_path_server, http_server):
        """timeout should be clamped between 1 and 120."""
        port = http_server
        url = f"http://127.0.0.1:{port}/test"
        resp = self._send_request(socket_path_server, {"url": url, "timeout": 999})
        assert resp["error"] is False
        result = json.loads(resp["result"])
        assert result["status"] == 200

    def test_non_int_max_redirects(
        self, sandbox_server, socket_path_server, http_server
    ):
        """Non-integer max_redirects should return an error."""
        port = http_server
        url = f"http://127.0.0.1:{port}/test"
        resp = self._send_request(
            socket_path_server, {"url": url, "max_redirects": "abc"}
        )
        assert resp["error"] is True
        assert "max_redirects must be an integer" in resp["result"]

    def test_exception_catch(self, sandbox_server, socket_path_server):
        """Test requests.RequestException handler with bad URL."""
        url = "http://192.0.2.1/test"
        resp = self._send_request(socket_path_server, {"url": url, "timeout": 2})
        assert resp["error"] is True

    def test_webfetch_head_method(
        self, sandbox_server, socket_path_server, http_server
    ):
        """Test HEAD method returns no body."""
        port = http_server
        url = f"http://127.0.0.1:{port}/test"
        resp = self._send_request(socket_path_server, {"url": url, "method": "HEAD"})
        assert resp["error"] is False
        result = json.loads(resp["result"])
        assert result["status"] == 200

    def test_custom_headers(self, sandbox_server, socket_path_server, http_server):
        """Send custom headers."""
        port = http_server
        url = f"http://127.0.0.1:{port}/headers_echo"
        resp = self._send_request(
            socket_path_server,
            {"url": url, "headers": {"X-Custom": "test-value"}},
        )
        assert resp["error"] is False
        result = json.loads(resp["result"])
        headers = json.loads(result["body"])
        assert headers.get("X-Custom") == "test-value"


class TestWebFetchRateLimiting:
    """Cover _rate_limit_domain sleep path (line 501)."""

    @staticmethod
    def _start_worker(sock_path, extra_env):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env.update(extra_env)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def _send(self, sock_path: str, payload: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(10)
            sock.connect(sock_path)
            msg = json.dumps({"name": "WebFetch", **payload}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    def test_rate_limit_sleep(self, tmp_path, http_server):
        """Two rapid POSTs trigger the rate-limit sleep."""
        port = http_server
        sock_path = str(tmp_path / "ratelimit.sock")
        proc = self._start_worker(sock_path, {"TIZ_WEBFETCH_MIN_INTERVAL": "0.2"})
        try:
            url = f"http://127.0.0.1:{port}/test"
            resp1 = self._send(sock_path, {"url": url, "method": "POST", "body": "1"})
            assert resp1["error"] is False
            # Immediate second request - should trigger rate limit
            resp2 = self._send(sock_path, {"url": url, "method": "POST", "body": "2"})
            assert resp2["error"] is False
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestWebFetchCacheEviction:
    """Cover _cache_set popitem (line 517)."""

    @staticmethod
    def _start_worker(sock_path, extra_env):
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env.update(extra_env)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def _send(self, sock_path: str, payload: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(10)
            sock.connect(sock_path)
            msg = json.dumps({"name": "WebFetch", **payload}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    def test_cache_eviction(self, tmp_path):
        """Fill tiny cache to trigger eviction."""
        sock_path = str(tmp_path / "cache_evict.sock")
        proc = self._start_worker(
            sock_path,
            {
                "TIZ_WEBFETCH_CACHE_MAX": "3",
                "TIZ_WEBFETCH_MIN_INTERVAL": "0.001",
            },
        )
        try:
            servers = []
            try:
                for _ in range(5):
                    h = http.server.HTTPServer(("127.0.0.1", 0), SimpleHTTPHandler)
                    p = h.server_address[1]
                    t = threading.Thread(target=h.serve_forever, daemon=True)
                    t.start()
                    servers.append(h)
                    url = f"http://127.0.0.1:{p}/test"
                    resp = self._send(sock_path, {"url": url})
                    assert resp["error"] is False
            finally:
                for h in servers:
                    h.shutdown()
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestWebFetchRequestTimeout:
    """Cover the requests.Timeout exception handler (line 602)."""

    def test_timeout_with_slow_server(self, tmp_path):
        """Connect to a server that accepts but never responds, causing timeout."""
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "timeout_test.sock")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            },
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        # Start a server that accepts connections but never sends data
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        port = server_sock.getsockname()[1]
        server_sock.listen(1)
        server_sock.settimeout(2)

        def accept_and_hold():
            try:
                conn, _ = server_sock.accept()
                time.sleep(1.5)
                conn.close()
            except Exception:
                pass

        t = threading.Thread(target=accept_and_hold, daemon=True)
        t.start()
        time.sleep(0.1)
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.settimeout(10)
                sock.connect(sock_path)
                url = f"http://127.0.0.1:{port}/"
                msg = (
                    json.dumps({"name": "WebFetch", "url": url, "timeout": 1}).encode(
                        "utf-8"
                    )
                    + b"\n"
                )
                sock.sendall(struct.pack(">I", len(msg)) + msg)
                length_data = _recv_exact(sock, 4)
                payload_length = struct.unpack(">I", length_data)[0]
                data = _recv_exact(sock, payload_length)
                response = json.loads(data.decode("utf-8"))
                assert response["error"] is True
                result = json.loads(response["result"])
                assert "Request timed out" in result.get("error", "")
            finally:
                sock.close()
        finally:
            server_sock.close()
            proc.send_signal(2)
            proc.wait(timeout=5)


# ===================================================================
# Custom tools loading integration tests (cover _load_custom_tools)
# ===================================================================


class TestCustomToolsLoading:
    """Cover _load_custom_tools via separate worker processes."""

    @staticmethod
    def _start_worker_with_tools_dir(worker_path, sock_path, tools_dir, env=None):
        """Start a sandbox worker with custom tools dir in a modified env."""
        my_env = os.environ.copy()
        my_env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        my_env["TIZ_CUSTOM_TOOLS_DIR"] = str(tools_dir)
        if env:
            my_env.update(env)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=my_env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    @staticmethod
    def _send_raw(sock_path, payload):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(sock_path)
            msg = json.dumps(payload).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    def test_custom_tool_loaded_and_executed(self, tmp_path):
        """Create a valid custom tool and call it via the socket."""
        tools_dir = tmp_path / "custom_tools"
        tools_dir.mkdir()
        handler_code = """def handle(params):
    req_params = params.get("params", {})
    name_val = req_params.get("name", "world") if isinstance(req_params, dict) else "world"
    return "Hello, " + name_val + "!", False
"""
        (tools_dir / "hello.py").write_text(handler_code, encoding="utf-8")
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "custom.sock")
        proc = self._start_worker_with_tools_dir(worker_path, sock_path, tools_dir)
        try:
            resp = self._send_raw(
                sock_path, {"name": "hello", "params": {"name": "test"}}
            )
            assert resp["error"] is False
            assert "Hello" in resp["result"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_custom_tool_no_handle_attr(self, tmp_path):
        """Custom module without handle attribute should not crash."""
        tools_dir = tmp_path / "custom_tools_no_handle"
        tools_dir.mkdir()
        (tools_dir / "bad_tool.py").write_text("not_a_handle = 42\n", encoding="utf-8")
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "no_handle.sock")
        proc = self._start_worker_with_tools_dir(worker_path, sock_path, tools_dir)
        try:
            resp = self._send_raw(sock_path, {"name": "Bash", "command": "echo ok"})
            assert resp["error"] is False
            assert "ok" in resp["result"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_custom_tool_import_error(self, tmp_path):
        """Custom module that raises on import should not crash worker."""
        tools_dir = tmp_path / "custom_tools_import_err"
        tools_dir.mkdir()
        (tools_dir / "crash.py").write_text(
            "raise RuntimeError('simulated import failure')\n", encoding="utf-8"
        )
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "import_err.sock")
        proc = self._start_worker_with_tools_dir(worker_path, sock_path, tools_dir)
        try:
            resp = self._send_raw(sock_path, {"name": "Bash", "command": "echo ok"})
            assert resp["error"] is False
            assert "ok" in resp["result"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_custom_tool_non_py_file_skipped(self, tmp_path):
        """Non-.py files in custom tools dir should be skipped."""
        tools_dir = tmp_path / "custom_tools_skip"
        tools_dir.mkdir()
        (tools_dir / "readme.txt").write_text("not a tool", encoding="utf-8")
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "skip.sock")
        proc = self._start_worker_with_tools_dir(worker_path, sock_path, tools_dir)
        try:
            resp = self._send_raw(sock_path, {"name": "Bash", "command": "echo ok"})
            assert resp["error"] is False
            assert "ok" in resp["result"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_custom_tool_spec_none(self, tmp_path):
        """Custom module where spec_from_file_location returns None."""
        tools_dir = tmp_path / "custom_tools_spec_none"
        tools_dir.mkdir()
        # An empty file causes spec to be None
        (tools_dir / "__init__.py").write_text("", encoding="utf-8")
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "spec_none.sock")
        proc = self._start_worker_with_tools_dir(worker_path, sock_path, tools_dir)
        try:
            resp = self._send_raw(sock_path, {"name": "Bash", "command": "echo ok"})
            assert resp["error"] is False
            assert "ok" in resp["result"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


# ===================================================================
# WebSearch integration tests (cover _tool_websearch, _websearch_rate_limit,
# _websearch_cache_get, _websearch_cache_set, _parse_ddg_results)
# ===================================================================


class _WebSearchTestBase:
    """Shared helpers for WebSearch tests."""

    @staticmethod
    def _send_request(sock_path: str, payload: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(15)
            sock.connect(sock_path)
            msg = json.dumps({"name": "WebSearch", **payload}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    @staticmethod
    def _start_worker(
        sock_path: str, env_overrides: dict | None = None
    ) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc


class TestWebSearchCoverage:
    """Integration tests for the WebSearch tool covering _tool_websearch.

    Validation tests use the shared sandbox_server. Network-dependent
    tests use a dedicated worker with TIZ_WEBSEARCH_URL pointed at a
    non-routable address (192.0.2.1) to guarantee connection errors.
    """

    def test_empty_query(self, sandbox_server, socket_path_server):
        resp = _WebSearchTestBase._send_request(socket_path_server, {"query": ""})
        assert resp["error"] is True
        assert "query is required" in resp["result"]

    def test_non_string_query(self, sandbox_server, socket_path_server):
        resp = _WebSearchTestBase._send_request(socket_path_server, {"query": 123})
        assert resp["error"] is True
        assert "query is required" in resp["result"]

    def test_invalid_page(self, sandbox_server, socket_path_server):
        resp = _WebSearchTestBase._send_request(
            socket_path_server, {"query": "test", "page": "bad"}
        )
        assert resp["error"] is True
        assert "page must be a positive integer" in resp["result"]

    def test_negative_max_results(self, sandbox_server, socket_path_server):
        resp = _WebSearchTestBase._send_request(
            socket_path_server, {"query": "test", "max_results": -5}
        )
        assert resp["error"] is True
        assert "max_results must be a positive integer" in resp["result"]

    def test_large_max_results(self, sandbox_server, socket_path_server):
        resp = _WebSearchTestBase._send_request(
            socket_path_server, {"query": "test", "max_results": "big"}
        )
        assert resp["error"] is True
        assert "max_results must be a positive integer" in resp["result"]

    def test_invalid_timeout(self, sandbox_server, socket_path_server):
        resp = _WebSearchTestBase._send_request(
            socket_path_server, {"query": "test", "timeout": "bad"}
        )
        assert resp["error"] is True
        assert "timeout must be an integer between 1 and 30" in resp["result"]

    def test_timeout_too_large(self, sandbox_server, socket_path_server):
        resp = _WebSearchTestBase._send_request(
            socket_path_server, {"query": "test", "timeout": 999}
        )
        assert resp["error"] is True
        assert "timeout must be an integer between 1 and 30" in resp["result"]

    def test_timeout_too_small(self, sandbox_server, socket_path_server):
        resp = _WebSearchTestBase._send_request(
            socket_path_server, {"query": "test", "timeout": 0}
        )
        assert resp["error"] is True
        assert "timeout must be an integer between 1 and 30" in resp["result"]

    def test_connection_error(self, tmp_path):
        """Trigger ConnectionError using unreachable URL."""
        sock_path = str(tmp_path / "ws_conn_err.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "test query", "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_rate_limit_and_cache_get(self, tmp_path):
        """Two rapid queries: first calls rate_limit + cache_get, second also calls cache_get."""
        sock_path = str(tmp_path / "ws_ratelimit.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp1 = _WebSearchTestBase._send_request(
                sock_path, {"query": "rate limit test", "timeout": 2}
            )
            assert resp1["error"] is True
            resp2 = _WebSearchTestBase._send_request(
                sock_path, {"query": "rate limit test", "timeout": 2}
            )
            assert resp2["error"] is True
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_page_gt_one(self, tmp_path):
        """page > 1 reaches payload construction and connection error."""
        sock_path = str(tmp_path / "ws_page_gt1.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "test", "page": 3, "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_timelimit_param(self, tmp_path):
        """timelimit param reaches payload and connection error."""
        sock_path = str(tmp_path / "ws_timelimit.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "test", "timelimit": "week", "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_safe_search_on(self, tmp_path):
        """safe_search on reaches payload and connection error."""
        sock_path = str(tmp_path / "ws_safesearch_on.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "test", "safe_search": "on", "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_safe_search_moderate(self, tmp_path):
        """safe_search moderate reaches payload and connection error."""
        sock_path = str(tmp_path / "ws_safesearch_mod.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "test", "safe_search": "moderate", "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_custom_user_agent(self, tmp_path):
        """custom user_agent reaches payload and connection error."""
        sock_path = str(tmp_path / "ws_ua.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "test", "user_agent": "TestBot/1.0", "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_websearch_timeout_error(self, tmp_path):
        """Trigger requests.Timeout via a slow server."""
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        sock_path = str(tmp_path / "ws_timeout.sock")
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        port = server_sock.getsockname()[1]
        server_sock.listen(1)
        server_sock.settimeout(2)

        def accept_and_hold():
            try:
                conn, _ = server_sock.accept()
                time.sleep(5)
                conn.close()
            except Exception:
                pass

        t = threading.Thread(target=accept_and_hold, daemon=True)
        t.start()
        time.sleep(0.1)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{port}/",
            },
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.settimeout(10)
                sock.connect(sock_path)
                msg = (
                    json.dumps(
                        {"name": "WebSearch", "query": "test", "timeout": 1}
                    ).encode("utf-8")
                    + b"\n"
                )
                sock.sendall(struct.pack(">I", len(msg)) + msg)
                length_data = _recv_exact(sock, 4)
                payload_length = struct.unpack(">I", length_data)[0]
                data = _recv_exact(sock, payload_length)
                response = json.loads(data.decode("utf-8"))
                assert response["error"] is True
                result = json.loads(response["result"])
                assert "error" in result
            finally:
                sock.close()
        finally:
            server_sock.close()
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestWebSearchProxyCoverage:
    """Cover additional _tool_websearch paths using an unreachable URL."""

    def test_websearch_connection_error(self, tmp_path):
        """Trigger ConnectionError in _tool_websearch."""
        sock_path = str(tmp_path / "ws_proxy_err.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "connection error test", "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestWebSearchCacheEviction:
    """Cover _websearch_cache_set popitem path with a local mock server."""

    def test_websearch_cache_eviction(self, tmp_path):
        """Fill tiny cache to trigger eviction via a mock DDG server."""

        class DDGMockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = (
                    "<html>"
                    "<!-- This is the visible part -->"
                    '<div class="result"><a class="result__a" '
                    'href="http://example.com/a">Title A</a>'
                    '<a class="result__snippet">Snippet A</a></div>'
                    "</div></div>"
                    "</html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        mock_server = http.server.HTTPServer(("127.0.0.1", 0), DDGMockHandler)
        mock_port = mock_server.server_address[1]
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()

        sock_path = str(tmp_path / "ws_cache_evict.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path,
            {
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{mock_port}/html/",
                "TIZ_WEBSEARCH_CACHE_MAX": "3",
                "TIZ_WEBSEARCH_MIN_INTERVAL": "0.001",
            },
        )
        try:
            for i in range(5):
                resp = _WebSearchTestBase._send_request(
                    sock_path,
                    {"query": f"eviction test {i}", "timeout": 5},
                )
                assert resp["error"] is False
                result = json.loads(resp["result"])
                assert "results" in result
        finally:
            mock_server.shutdown()
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_websearch_connection_error_with_params(self, tmp_path):
        """Cover all param branches reaching the network (local mock server)."""

        class DDGMockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html></html>")

            def log_message(self, *args):
                pass

        mock_server = http.server.HTTPServer(("127.0.0.1", 0), DDGMockHandler)
        mock_port = mock_server.server_address[1]
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()
        sock_path = str(tmp_path / "ws_params.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path,
            {
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{mock_port}/html/",
                "TIZ_WEBSEARCH_MIN_INTERVAL": "0.001",
            },
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path,
                {
                    "query": "eviction test",
                    "region": "us-en",
                    "timelimit": "month",
                    "page": 1,
                    "safe_search": "moderate",
                    "user_agent": "TestAgent/1.0",
                    "timeout": 5,
                },
            )
            assert resp["error"] is False
            result = json.loads(resp["result"])
            assert "results" in result
        finally:
            mock_server.shutdown()
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestWebSearchCacheHit:
    """Cover _websearch_cache_get cached hit path using a local mock server."""

    def test_websearch_cache_hit(self, tmp_path):
        """First request is uncached, second returns cached result."""

        class DDGMockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = (
                    "<html>"
                    "<!-- This is the visible part -->"
                    '<div class="result"><a class="result__a" '
                    'href="http://example.com/b">Title B</a>'
                    '<a class="result__snippet">Snippet B</a></div>'
                    "</div></div>"
                    "</html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        mock_server = http.server.HTTPServer(("127.0.0.1", 0), DDGMockHandler)
        mock_port = mock_server.server_address[1]
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()

        sock_path = str(tmp_path / "ws_cache_hit.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path,
            {
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{mock_port}/html/",
                "TIZ_WEBSEARCH_MIN_INTERVAL": "0.001",
            },
        )
        try:
            resp1 = _WebSearchTestBase._send_request(
                sock_path, {"query": "cache hit test", "timeout": 5}
            )
            assert resp1["error"] is False
            result1 = json.loads(resp1["result"])
            assert "cached" not in result1 or result1.get("cached") is not True

            resp2 = _WebSearchTestBase._send_request(
                sock_path, {"query": "cache hit test", "timeout": 5}
            )
            assert resp2["error"] is False
            result2 = json.loads(resp2["result"])
            assert result2.get("cached") is True
        finally:
            mock_server.shutdown()
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_websearch_request_exception(self, tmp_path):
        """Cover requests.RequestException handler via unreachable URL."""
        sock_path = str(tmp_path / "ws_req_exc.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://192.0.2.1/html/"}
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "request exception test", "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestWebSearchExtraCoverage:
    """Cover remaining branches in _parse_ddg_results and _tool_websearch."""

    def test_parse_ddg_no_title_match(self, tmp_path):
        """Block without result__a triggers continue (line 669)."""

        class DDGMockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = (
                    "<html>"
                    "<!-- This is the visible part -->"
                    '<div class="result">'
                    "<span>no link here</span>"
                    '<a class="result__snippet">Snippet only</a></div>'
                    "</div></div>"
                    "</html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        mock_server = http.server.HTTPServer(("127.0.0.1", 0), DDGMockHandler)
        mock_port = mock_server.server_address[1]
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()

        sock_path = str(tmp_path / "ws_no_title.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path,
            {
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{mock_port}/html/",
                "TIZ_WEBSEARCH_MIN_INTERVAL": "0.001",
            },
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "no title match", "timeout": 5}
            )
            assert resp["error"] is False
            result = json.loads(resp["result"])
            assert result["results"] == []
        finally:
            mock_server.shutdown()
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_parse_ddg_uddg_url(self, tmp_path):
        """Result with uddg= href triggers URL parsing (lines 678-680)."""

        class DDGMockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = (
                    "<html>"
                    "<!-- This is the visible part -->"
                    '<div class="result"><a class="result__a" '
                    'href="https://example.com/l/?uddg=https%3A%2F%2Freal-site.com%2Fpage">'
                    "Redirected Title</a>"
                    '<a class="result__snippet">Redirected snippet</a></div>'
                    "</div></div>"
                    "</html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        mock_server = http.server.HTTPServer(("127.0.0.1", 0), DDGMockHandler)
        mock_port = mock_server.server_address[1]
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()

        sock_path = str(tmp_path / "ws_uddg.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path,
            {
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{mock_port}/html/",
                "TIZ_WEBSEARCH_MIN_INTERVAL": "0.001",
            },
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "uddg test", "timeout": 5}
            )
            assert resp["error"] is False
            result = json.loads(resp["result"])
            assert len(result["results"]) == 1
            assert result["results"][0]["href"] == "https://real-site.com/page"
        finally:
            mock_server.shutdown()
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_safe_search_unknown_value(self, tmp_path):
        """safe_search not in (on, moderate, off) hits branch 725->728."""

        class DDGMockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = (
                    "<html>"
                    "<!-- This is the visible part -->"
                    '<div class="result"><a class="result__a" '
                    'href="http://example.com/x">Title X</a>'
                    '<a class="result__snippet">Snippet X</a></div>'
                    "</div></div>"
                    "</html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        mock_server = http.server.HTTPServer(("127.0.0.1", 0), DDGMockHandler)
        mock_port = mock_server.server_address[1]
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()

        sock_path = str(tmp_path / "ws_safesearch_unknown.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path,
            {
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{mock_port}/html/",
                "TIZ_WEBSEARCH_MIN_INTERVAL": "0.001",
            },
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path,
                {
                    "query": "unknown safe_search",
                    "safe_search": "custom_mode",
                    "timeout": 5,
                },
            )
            assert resp["error"] is False
            result = json.loads(resp["result"])
            assert len(result["results"]) == 1
        finally:
            mock_server.shutdown()
            proc.send_signal(2)
            proc.wait(timeout=5)

    def test_websearch_http_error(self, tmp_path):
        """HTTP error (500) triggers raise_for_status -> RequestException (lines 746-747)."""

        class DDGMockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Internal Server Error")

            def log_message(self, *args):
                pass

        mock_server = http.server.HTTPServer(("127.0.0.1", 0), DDGMockHandler)
        mock_port = mock_server.server_address[1]
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()

        sock_path = str(tmp_path / "ws_http_error.sock")
        proc = _WebSearchTestBase._start_worker(
            sock_path,
            {
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{mock_port}/html/",
                "TIZ_WEBSEARCH_MIN_INTERVAL": "0.001",
            },
        )
        try:
            resp = _WebSearchTestBase._send_request(
                sock_path, {"query": "http error test", "timeout": 5}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
            assert "Request failed" in result["error"]
        finally:
            mock_server.shutdown()
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestCargoFetchRawCoverage:
    """Hit server-side validation (empty path/version) via raw socket."""

    @staticmethod
    def _send_raw(sock_path: str, payload: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(sock_path)
            msg = json.dumps(payload).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    def test_cargo_fetch_empty_path(self, sandbox_server, socket_path_server):
        resp = self._send_raw(socket_path_server, {"name": "CargoFetch", "path": ""})
        assert resp["error"] is True
        assert "path is required" in resp["result"]

    def test_uv_sync_empty_path(self, sandbox_server, socket_path_server):
        resp = self._send_raw(socket_path_server, {"name": "UvSync", "path": ""})
        assert resp["error"] is True
        assert "path is required" in resp["result"]

    def test_uv_python_install_empty_version(self, sandbox_server, socket_path_server):
        resp = self._send_raw(
            socket_path_server, {"name": "UvPythonInstall", "version": ""}
        )
        assert resp["error"] is True
        assert "version is required" in resp["result"]


# ===================================================================
# WebSearch rate limit sleep coverage (line 706)
# ===================================================================


class TestWebSearchRateLimitSleep:
    """Cover _websearch_rate_limit sleep path (line 706).

    Two rapid uncached requests to a fast mock server ensure the second
    call hits time.sleep() because the elapsed time is below the min
    interval.
    """

    @staticmethod
    def _send_request(sock_path: str, payload: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(10)
            sock.connect(sock_path)
            msg = json.dumps({"name": "WebSearch", **payload}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    @staticmethod
    def _start_worker(
        sock_path: str, env_overrides: dict | None = None
    ) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def test_rate_limit_sleep_websearch(self, tmp_path):
        """Two rapid uncached requests trigger time.sleep in rate limiter."""

        class FastMockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = (
                    "<html>"
                    "<!-- This is the visible part -->"
                    '<div class="result"><a class="result__a" '
                    'href="http://example.com/a">Title A</a>'
                    '<a class="result__snippet">Snippet A</a></div>'
                    "</div></div>"
                    "</html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        mock_server = http.server.HTTPServer(("127.0.0.1", 0), FastMockHandler)
        mock_port = mock_server.server_address[1]
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()

        sock_path = str(tmp_path / "ws_rl_sleep.sock")
        proc = self._start_worker(
            sock_path,
            {
                "TIZ_WEBSEARCH_URL": f"http://127.0.0.1:{mock_port}/html/",
                "TIZ_WEBSEARCH_MIN_INTERVAL": "0.2",
            },
        )
        try:
            # Two different queries (different cache keys) to skip cache
            resp1 = self._send_request(
                sock_path, {"query": "rate limit sleep alpha", "timeout": 5}
            )
            assert resp1["error"] is False

            resp2 = self._send_request(
                sock_path, {"query": "rate limit sleep beta", "timeout": 5}
            )
            assert resp2["error"] is False
        finally:
            mock_server.shutdown()
            proc.send_signal(2)
            proc.wait(timeout=5)


class TestWebSearchConnectionError:
    """Cover requests.ConnectionError handler in _tool_websearch (line 818).

    Using 127.0.0.1:1 reliably produces ConnectionError (ECONNREFUSED)
    on any machine regardless of network access restrictions.
    """

    @staticmethod
    def _send_request(sock_path: str, payload: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(10)
            sock.connect(sock_path)
            msg = json.dumps({"name": "WebSearch", **payload}).encode("utf-8") + b"\n"
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            return json.loads(data.decode("utf-8"))
        finally:
            sock.close()

    @staticmethod
    def _start_worker(
        sock_path: str, env_overrides: dict | None = None
    ) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def test_websearch_connection_error(self, tmp_path):
        """Connection to 127.0.0.1:1 reliably yields ConnectionError."""
        sock_path = str(tmp_path / "ws_conn_err2.sock")
        proc = self._start_worker(
            sock_path, {"TIZ_WEBSEARCH_URL": "http://127.0.0.1:1/html/"}
        )
        try:
            resp = self._send_request(
                sock_path, {"query": "connection error coverage", "timeout": 2}
            )
            assert resp["error"] is True
            result = json.loads(resp["result"])
            assert "error" in result
            assert "Connection" in result["error"] or "connection" in result["error"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


# ===================================================================
# ReadFile server-side validation via raw socket
# (bypasses tool client-side pre-validation)
# ===================================================================


def _send_raw_socket(sock_path: str, payload: dict) -> dict:
    """Send a raw JSON payload via Unix socket and return parsed response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(5)
        sock.connect(sock_path)
        msg = json.dumps(payload).encode("utf-8") + b"\n"
        sock.sendall(struct.pack(">I", len(msg)) + msg)
        length_data = _recv_exact(sock, 4)
        payload_length = struct.unpack(">I", length_data)[0]
        data = _recv_exact(sock, payload_length)
        return json.loads(data.decode("utf-8"))
    finally:
        sock.close()


class TestReadFileServerValidation:
    """Cover server-side view_range validation in _tool_read (lines 230-237)."""

    def test_view_range_not_list(self, sandbox_server, socket_path_server, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello\nworld\n", encoding="utf-8")
        resp = _send_raw_socket(
            socket_path_server,
            {"name": "ReadFile", "path": str(f), "view_range": "not_a_list"},
        )
        assert resp["error"] is True
        assert "two-element array" in resp["result"]

    def test_view_range_wrong_length(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        f = tmp_path / "file.txt"
        f.write_text("hello\nworld\n", encoding="utf-8")
        resp = _send_raw_socket(
            socket_path_server,
            {"name": "ReadFile", "path": str(f), "view_range": [1]},
        )
        assert resp["error"] is True
        assert "two-element array" in resp["result"]

    def test_view_range_non_int(self, sandbox_server, socket_path_server, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello\nworld\n", encoding="utf-8")
        resp = _send_raw_socket(
            socket_path_server,
            {"name": "ReadFile", "path": str(f), "view_range": ["a", 2]},
        )
        assert resp["error"] is True
        assert "integers" in resp["result"]

    def test_view_range_negative(self, sandbox_server, socket_path_server, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello\nworld\n", encoding="utf-8")
        resp = _send_raw_socket(
            socket_path_server,
            {"name": "ReadFile", "path": str(f), "view_range": [0, 2]},
        )
        assert resp["error"] is True
        assert "positive" in resp["result"]

    def test_view_range_start_gt_end(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        f = tmp_path / "file.txt"
        f.write_text("hello\nworld\n", encoding="utf-8")
        resp = _send_raw_socket(
            socket_path_server,
            {"name": "ReadFile", "path": str(f), "view_range": [3, 1]},
        )
        assert resp["error"] is True
        assert "start must be <= end" in resp["result"]


# ===================================================================
# Edit server-side validation via raw socket
# ===================================================================


class TestEditServerValidation:
    """Cover server-side expected_replacements validation in _tool_edit (line 260)."""

    def test_expected_replacements_non_int(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        f = tmp_path / "edit.txt"
        f.write_text("hello world\n", encoding="utf-8")
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "Edit",
                "path": str(f),
                "old_string": "hello",
                "new_string": "hi",
                "expected_replacements": "not_an_int",
            },
        )
        assert resp["error"] is True
        assert "expected_replacements must be an integer" in resp["result"]


# ===================================================================
# Additional Glob coverage
# ===================================================================


class TestGlobNonFileDirectory:
    """Cover _tool_glob non-file continue (line 316)."""

    def test_glob_skips_directory_matching_pattern(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.glob_tool import Glob

        (tmp_path / "a.txt").mkdir()
        (tmp_path / "b.txt").write_text("data", encoding="utf-8")
        tool = Glob(socket_path_server)
        result = tool.run({"pattern": "*.txt", "path": str(tmp_path)})
        lines = result.strip().split("\n")
        assert str(tmp_path / "b.txt") in lines
        assert str(tmp_path / "a.txt") not in lines


class TestGlobGitRecursive:
    """Cover _tool_glob .git in parts continue (line 319)."""

    def test_glob_recursive_skips_git_py_files(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.glob_tool import Glob

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "hook.py").write_text("", encoding="utf-8")
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        tool = Glob(socket_path_server)
        # Use recursive glob pattern to descend into .git/
        result = tool.run({"pattern": "**/*.py", "path": str(tmp_path)})
        lines = result.strip().split("\n")
        assert str(tmp_path / "a.py") in lines
        assert ".git" not in result


# ===================================================================
# Additional ListDir coverage - recursive with show_hidden (line 412->414 arc)
# ===================================================================


class TestListDirRecursiveShowHidden:
    """Cover _tool_list_dir recursive show_hidden branch (line 412->414)."""

    def test_list_dir_recursive_show_hidden(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        from tiz.tools.listdir import ListDir

        (tmp_path / ".hidden").write_text("", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("", encoding="utf-8")
        tool = ListDir(socket_path_server)
        result = tool.run(
            {"path": str(tmp_path), "recursive": True, "show_hidden": True}
        )
        entries = json.loads(result)
        names = {e["name"] for e in entries}
        assert len(names) == 2
        assert ".hidden" in names
        assert "visible.txt" in names


# ===================================================================
# WebFetch raw truncated note (line 692)
# ===================================================================


class TestWebFetchRawTruncation:
    """Alternative approach using the existing http_server fixture with /huge path."""

    def test_raw_truncated_note_via_http_server(
        self, sandbox_server, socket_path_server, http_server
    ):
        port = http_server
        url = f"http://127.0.0.1:{port}/huge"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(15)
            sock.connect(socket_path_server)
            msg = (
                json.dumps(
                    {"name": "WebFetch", "url": url, "raw": True, "timeout": 10}
                ).encode("utf-8")
                + b"\n"
            )
            sock.sendall(struct.pack(">I", len(msg)) + msg)
            length_data = _recv_exact(sock, 4)
            payload_length = struct.unpack(">I", length_data)[0]
            data = _recv_exact(sock, payload_length)
            response = json.loads(data.decode("utf-8"))
            assert response["error"] is False
            result = json.loads(response["result"])
            assert result.get("truncated") is True
            assert "truncated" in result.get("note", "")
        finally:
            sock.close()


# ===================================================================
# Bash server-side validation via raw socket
# ===================================================================


class TestBashServerValidation:
    """Cover _tool_bash validations bypassed by the tool client."""

    def test_bash_cwd_not_found(self, sandbox_server, socket_path_server):
        resp = _send_raw_socket(
            socket_path_server,
            {"name": "Bash", "command": "echo hi", "cwd": "/nonexistent_cwd_xyz"},
        )
        assert resp["error"] is True
        assert "cwd not found" in resp["result"]

    def test_bash_env_not_dict(self, sandbox_server, socket_path_server):
        resp = _send_raw_socket(
            socket_path_server,
            {"name": "Bash", "command": "echo hi", "env": "not_a_dict"},
        )
        assert resp["error"] is True
        assert "env must be a dict" in resp["result"]

    def test_bash_env_value_not_str(self, sandbox_server, socket_path_server):
        resp = _send_raw_socket(
            socket_path_server,
            {"name": "Bash", "command": "echo hi", "env": {"MYVAR": 123}},
        )
        assert resp["error"] is True
        assert "env value for 'MYVAR' must be a string" in resp["result"]


# ===================================================================
# CargoFetch FileNotFoundError coverage
# ===================================================================


class TestCargoFetchNotFound:
    """Cover _tool_cargo_fetch FileNotFoundError (lines 140-141)."""

    @staticmethod
    def _start_no_cargo_worker(sock_path: str) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        bindir = Path(sock_path).parent / "empty_bin_cargo"
        bindir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def test_cargo_fetch_not_found(self, tmp_path):
        sock_path = str(tmp_path / "cargo_not_found.sock")
        proc = self._start_no_cargo_worker(sock_path)
        try:
            project_dir = tmp_path / "cargo_project_nf"
            project_dir.mkdir()
            (project_dir / "Cargo.toml").write_text(
                '[package]\nname = "test"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            resp = _send_raw_socket(
                sock_path,
                {
                    "name": "CargoFetch",
                    "path": str(project_dir),
                },
            )
            assert resp["error"] is True
            assert "cargo not found" in resp["result"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


# ===================================================================
# UvSync server-side validation via raw socket
# ===================================================================


class TestUvSyncServerValidation:
    """Cover _tool_uv_sync validations bypassed by the tool client."""

    def test_uv_sync_group_not_list(self, sandbox_server, socket_path_server, tmp_path):
        project_dir = tmp_path / "uv_group_test"
        project_dir.mkdir()
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "UvSync",
                "path": str(project_dir),
                "group": "not_a_list",
            },
        )
        assert resp["error"] is True
        assert "group must be a list" in resp["result"]

    def test_uv_sync_extra_not_list(self, sandbox_server, socket_path_server, tmp_path):
        project_dir = tmp_path / "uv_extra_test"
        project_dir.mkdir()
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "UvSync",
                "path": str(project_dir),
                "extra": "not_a_list",
            },
        )
        assert resp["error"] is True
        assert "extra must be a list" in resp["result"]


class TestUvSyncNotFound:
    """Cover _tool_uv_sync FileNotFoundError (lines 179-180)."""

    @staticmethod
    def _start_no_uv_worker(sock_path: str) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        bindir = Path(sock_path).parent / "empty_bin_uv"
        bindir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def test_uv_sync_not_found(self, tmp_path):
        sock_path = str(tmp_path / "uv_sync_nf.sock")
        proc = self._start_no_uv_worker(sock_path)
        try:
            project_dir = tmp_path / "uv_project_nf"
            project_dir.mkdir()
            resp = _send_raw_socket(
                sock_path,
                {
                    "name": "UvSync",
                    "path": str(project_dir),
                },
            )
            assert resp["error"] is True
            assert "uv not found" in resp["result"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


# ===================================================================
# UvPythonInstall FileNotFoundError coverage
# ===================================================================


class TestUvPythonInstallNotFound:
    """Cover _tool_uv_python_install FileNotFoundError (lines 203-204)."""

    @staticmethod
    def _start_no_uv_worker(sock_path: str) -> subprocess.Popen:
        worker_path = (
            Path(__file__).resolve().parent.parent / "src" / "tiz" / "sandbox_worker.py"
        )
        bindir = Path(sock_path).parent / "empty_bin_uv_pi"
        bindir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["PATH"] = str(bindir)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--branch",
                str(worker_path),
                sock_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for _ in range(100):
            p = Path(sock_path)
            if p.exists() and p.stat().st_mode & 0o777 == 0o600:
                break
            time.sleep(0.05)
        assert not proc.poll(), proc.stdout.read().decode("utf-8")
        return proc

    def test_uv_python_install_not_found(self, tmp_path):
        sock_path = str(tmp_path / "uv_pi_nf.sock")
        proc = self._start_no_uv_worker(sock_path)
        try:
            resp = _send_raw_socket(
                sock_path,
                {
                    "name": "UvPythonInstall",
                    "version": "3.99.0",
                },
            )
            assert resp["error"] is True
            assert "uv not found" in resp["result"]
        finally:
            proc.send_signal(2)
            proc.wait(timeout=5)


# ===================================================================
# Grep server-side validation via raw socket
# ===================================================================


class TestGrepServerValidation:
    """Cover _tool_grep max_results validation (line 374)."""

    def test_grep_max_results_not_positive_int(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        f = tmp_path / "search.txt"
        f.write_text("hello\n", encoding="utf-8")
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "Grep",
                "pattern": "hello",
                "path": str(tmp_path),
                "max_results": "bad",
            },
        )
        assert resp["error"] is True
        assert "max_results must be a positive integer" in resp["result"]


# ===================================================================
# InsertFile server-side validation via raw socket
# ===================================================================


class TestInsertFileServerValidation:
    """Cover _tool_insert line_number validation (line 411)."""

    def test_insert_line_number_not_int(
        self, sandbox_server, socket_path_server, tmp_path
    ):
        f = tmp_path / "insert.txt"
        f.write_text("line1\nline2\n", encoding="utf-8")
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "InsertFile",
                "path": str(f),
                "content": "new",
                "line_number": "not_an_int",
            },
        )
        assert resp["error"] is True
        assert "line_number must be an integer" in resp["result"]


# ===================================================================
# ApplyPatch server-side validation via raw socket
# ===================================================================


class TestPatchServerValidation:
    """Cover _tool_patch strip validation (line 514)."""

    def test_patch_strip_not_int(self, sandbox_server, socket_path_server):
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "ApplyPatch",
                "patch": "dummy",
                "strip": "not_an_int",
            },
        )
        assert resp["error"] is True
        assert "strip must be a non-negative integer" in resp["result"]

    def test_patch_strip_negative(self, sandbox_server, socket_path_server):
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "ApplyPatch",
                "patch": "dummy",
                "strip": -1,
            },
        )
        assert resp["error"] is True
        assert "strip must be a non-negative integer" in resp["result"]


# ===================================================================
# Patch finally block cleanup coverage (line 544->exit)
# ===================================================================


class TestPatchFinallyCleanup:
    """Cover _tool_patch finally block when patch_file is None (line 544->exit)."""

    def test_patch_finally_patch_file_none(self, sandbox_server, socket_path_server):
        """Trigger an exception before tempfile creation by passing cwd as list."""
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "ApplyPatch",
                "patch": "dummy",
                "cwd": ["not", "a", "string"],
            },
        )
        # TypeError from Path(list) propagates up to run_tool's generic handler
        assert resp["error"] is True
        assert "unexpectedly" in resp["result"]


# ===================================================================
# Metadata lstat OSError coverage (lines 626-627)
# ===================================================================


# _tool_metadata lstat OSError at lines 626-627 is unreachable from
# normal filesystem operations because:
# - For symlinks, is_symlink returns True (path.lstat succeeds) and resolved.exists()
#   raises OSError first (before the try block) due to permission denied on target.
# - For non-symlinks, resolved.exists() also raises OSError first.
# The code is marked pragma: no cover in sandbox_worker.py.


# ===================================================================
# WebFetch server-side validation via raw socket
# ===================================================================


class TestWebFetchServerValidation:
    """Cover _tool_webfetch validations bypassed by the tool client."""

    def test_webfetch_method_not_valid_string(self, sandbox_server, socket_path_server):
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "WebFetch",
                "url": "http://127.0.0.1:1/",
                "method": 123,
            },
        )
        assert resp["error"] is True
        assert "method must be one of" in resp["result"]

    def test_webfetch_raw_not_bool(self, sandbox_server, socket_path_server):
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "WebFetch",
                "url": "http://127.0.0.1:1/",
                "raw": "not_a_bool",
            },
        )
        assert resp["error"] is True
        assert "raw must be a boolean" in resp["result"]

    def test_webfetch_timeout_not_int(self, sandbox_server, socket_path_server):
        resp = _send_raw_socket(
            socket_path_server,
            {
                "name": "WebFetch",
                "url": "http://127.0.0.1:1/",
                "timeout": "not_an_int",
            },
        )
        assert resp["error"] is True
        assert "timeout must be an integer" in resp["result"]
