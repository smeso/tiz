"""Tests for tiz.conversion_sandbox module."""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tiz.conversion_sandbox import (
    MIMETYPE_TO_SCRIPT,
    ConversionSandbox,
)
from tiz.sandbox_container import CONTAINER_MOUNT_SCRIPTS
from tiz.tools.base import Tool


class MockTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.last_run: dict[str, Any] | None = None
        self._result: str = "DONE"

    @staticmethod
    def prompt() -> str:
        return json.dumps({"name": "Bash", "description": "Run bash"})

    @staticmethod
    def fname() -> str:
        return "Bash"

    def run(self, args: dict[str, Any]) -> str:
        self.last_run = args
        command = args.get("command", "")
        cwd = args.get("cwd")
        parts = shlex.split(command)
        if len(parts) >= 3:
            out_dir = Path(cwd) / parts[-1] if cwd else Path(parts[-1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "page-1.jpg").write_bytes(b"image1")
            (out_dir / "page-2.jpg").write_bytes(b"image2")
        return self._result

    def format_confirmation(
        self, _args: dict[str, Any], _markdown: bool = False
    ) -> str | None:
        return None


class MockToolNoOutput(MockTool):
    def run(self, args: dict[str, Any]) -> str:
        self.last_run = args
        return self._result


class MockToolWithBytes(MockTool):
    def run(self, args: dict[str, Any]) -> str:
        self.last_run = args
        command = args.get("command", "")
        cwd = args.get("cwd")
        parts = shlex.split(command)
        if len(parts) >= 3:
            out_dir = Path(cwd) / parts[-1] if cwd else Path(parts[-1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "frame.jpg").write_bytes(b"image_bytes")
        return self._result


class MockToolAudio(MockTool):
    def run(self, args: dict[str, Any]) -> str:
        self.last_run = args
        command = args.get("command", "")
        cwd = args.get("cwd")
        parts = shlex.split(command)
        if len(parts) >= 3:
            out_path = Path(cwd) / parts[-1] if cwd else Path(parts[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"audio_data")
        return self._result


class MockToolAudioNoOutput(MockTool):
    def run(self, args: dict[str, Any]) -> str:
        self.last_run = args
        command = args.get("command", "")
        cwd = args.get("cwd")
        parts = shlex.split(command)
        if len(parts) >= 3:
            out_path = Path(cwd) / parts[-1] if cwd else Path(parts[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
        return self._result


def _patch_container_shared(tmp_path: Path) -> Any:
    return patch(
        "tiz.conversion_sandbox.CONTAINER_MOUNT_OWN_SHARED",
        str(tmp_path),
    )


def test_supported_mimetypes(tmp_path: Path) -> None:
    tool = MockTool()
    sandbox = ConversionSandbox(tool, str(tmp_path))
    expected = list(MIMETYPE_TO_SCRIPT.keys())
    assert sorted(sandbox.supported_mimetypes()) == sorted(expected)


def test_convert_unsupported_mimetype_returns_none(tmp_path: Path) -> None:
    tool = MockTool()
    sandbox = ConversionSandbox(tool, str(tmp_path))
    assert sandbox.convert("text/plain", "path") is None


def test_convert_with_path_success(tmp_path: Path) -> None:
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("pdf content")
        result = sandbox.convert("application/pdf", str(src))
    assert result is not None
    assert len(result) == 2
    assert result[0] == (b"image1", "image/jpeg")
    assert result[1] == (b"image2", "image/jpeg")
    assert tool.last_run is not None
    assert "pdf2pics.sh" in tool.last_run["command"]
    assert tool.last_run["timeout"] == 120
    assert tool.last_run["cwd"] is not None
    parts = shlex.split(tool.last_run["command"])
    assert len(parts) == 3
    assert parts[0] == f"{CONTAINER_MOUNT_SCRIPTS}/pdf2pics.sh"
    assert "/" not in parts[1]
    assert "/" not in parts[2]


def test_convert_with_bytes_success(tmp_path: Path) -> None:
    tool = MockToolWithBytes()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("application/pdf", b"pdf bytes")
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"image_bytes", "image/jpeg")


def test_convert_doc_mimetype(tmp_path: Path) -> None:
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "document.doc"
        src.write_text("doc content")
        result = sandbox.convert("application/msword", str(src))
    assert result is not None
    assert len(result) == 2
    assert tool.last_run is not None
    assert "doc2pics.sh" in tool.last_run["command"]


def test_convert_docx_mimetype(tmp_path: Path) -> None:
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "document.docx"
        src.write_text("docx content")
        result = sandbox.convert(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            str(src),
        )
    assert result is not None
    assert len(result) == 2
    assert tool.last_run is not None
    assert "doc2pics.sh" in tool.last_run["command"]


def test_convert_video_mimetype(tmp_path: Path) -> None:
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "video.mp4"
        src.write_text("video content")
        result = sandbox.convert("video/mp4", str(src))
    assert result is not None
    assert len(result) == 2
    assert tool.last_run is not None
    assert "video2pics.sh" in tool.last_run["command"]


def test_convert_video_mpeg_mimetype(tmp_path: Path) -> None:
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "video.mpeg"
        src.write_text("video content")
        result = sandbox.convert("video/mpeg", str(src))
    assert result is not None
    assert len(result) == 2
    assert tool.last_run is not None
    assert "video2pics.sh" in tool.last_run["command"]


def test_convert_video_quicktime_mimetype(tmp_path: Path) -> None:
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "video.mov"
        src.write_text("video content")
        result = sandbox.convert("video/quicktime", str(src))
    assert result is not None
    assert len(result) == 2
    assert tool.last_run is not None
    assert "video2pics.sh" in tool.last_run["command"]


def test_convert_video_x_msvideo_mimetype(tmp_path: Path) -> None:
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "video.avi"
        src.write_text("video content")
        result = sandbox.convert("video/x-msvideo", str(src))
    assert result is not None
    assert len(result) == 2
    assert tool.last_run is not None
    assert "video2pics.sh" in tool.last_run["command"]


def test_convert_video_matroska_mimetype(tmp_path: Path) -> None:
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "video.mkv"
        src.write_text("video content")
        result = sandbox.convert("video/x-matroska", str(src))
    assert result is not None
    assert len(result) == 2
    assert tool.last_run is not None
    assert "video2pics.sh" in tool.last_run["command"]


def test_convert_video_bytes_input_mimetypes(tmp_path: Path) -> None:
    tool = MockToolWithBytes()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("video/mpeg", b"video bytes")
    assert result is not None
    assert len(result) == 1

    tool2 = MockToolWithBytes()
    with _patch_container_shared(tmp_path):
        sandbox2 = ConversionSandbox(tool2, str(tmp_path))
        result2 = sandbox2.convert("video/quicktime", b"video bytes")
    assert result2 is not None
    assert len(result2) == 1

    tool3 = MockToolWithBytes()
    with _patch_container_shared(tmp_path):
        sandbox3 = ConversionSandbox(tool3, str(tmp_path))
        result3 = sandbox3.convert("video/x-msvideo", b"video bytes")
    assert result3 is not None
    assert len(result3) == 1

    tool4 = MockToolWithBytes()
    with _patch_container_shared(tmp_path):
        sandbox4 = ConversionSandbox(tool4, str(tmp_path))
        result4 = sandbox4.convert("video/x-matroska", b"video bytes")
    assert result4 is not None
    assert len(result4) == 1


def test_convert_tool_does_not_return_done_returns_none(tmp_path: Path) -> None:
    tool = MockTool()
    tool._result = "FAILED"
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("content")
        result = sandbox.convert("application/pdf", str(src))
    assert result is None


def test_convert_no_output_files_returns_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    tool = MockToolNoOutput()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("content")
        result = sandbox.convert("application/pdf", str(src))
    assert result is None
    assert "Script reported DONE but no output files found" in caplog.text


def test_convert_with_bytes_input_uses_extension(tmp_path: Path) -> None:
    tool = MockToolWithBytes()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("application/msword", b"doc content")
    assert result is not None
    assert len(result) == 1
    assert tool.last_run is not None
    parts = shlex.split(tool.last_run["command"])
    assert parts[1] == "input.doc"


def test_convert_creates_base_path(tmp_path: Path) -> None:
    base = tmp_path / "nonexistent" / "sessions"
    tool = MockTool()
    sandbox = ConversionSandbox(tool, str(base))
    assert base.exists()
    src = tmp_path / "input.pdf"
    src.write_text("content")
    with _patch_container_shared(base):
        result = sandbox.convert("application/pdf", str(src))
    assert result is not None


def test_convert_with_bytes_video_input(tmp_path: Path) -> None:
    tool = MockToolWithBytes()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("video/mp4", b"video bytes")
    assert result is not None
    assert len(result) == 1
    assert tool.last_run is not None
    assert "video2pics.sh" in tool.last_run["command"]


def test_convert_tool_result_not_done_returns_none(
    tmp_path: Path,
) -> None:
    class MockToolRunsButNoDone(MockTool):
        def run(self, args: dict[str, Any]) -> str:
            self.last_run = args
            return "SOMETHING ELSE"

    tool = MockToolRunsButNoDone()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("content")
        result = sandbox.convert("application/pdf", str(src))
    assert result is None


def test_convert_audio_mpeg_path_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "audio.mp3"
        src.write_text("mp3 content")
        result = sandbox.convert("audio/mpeg", str(src))
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")
    assert tool.last_run is not None
    assert "audio_convert.sh" in tool.last_run["command"]
    assert tool.last_run["timeout"] == 120
    parts = shlex.split(tool.last_run["command"])
    assert len(parts) == 3
    assert parts[0] == f"{CONTAINER_MOUNT_SCRIPTS}/audio_convert.sh"
    assert parts[1] == "audio.mp3"
    assert parts[2].endswith(".wav")
    assert "/" not in parts[1]
    assert "/" not in parts[2]


def test_convert_audio_wav_bytes_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("audio/wav", b"wav data")
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")
    assert tool.last_run is not None
    assert "audio_convert.sh" in tool.last_run["command"]


def test_convert_audio_ogg_path_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "audio.ogg"
        src.write_text("ogg content")
        result = sandbox.convert("audio/ogg", str(src))
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")
    assert tool.last_run is not None
    assert "audio_convert.sh" in tool.last_run["command"]


def test_convert_audio_flac_path_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "audio.flac"
        src.write_text("flac content")
        result = sandbox.convert("audio/flac", str(src))
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")


def test_convert_audio_x_wav_bytes_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("audio/x-wav", b"wav data")
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")
    assert tool.last_run is not None
    assert "audio_convert.sh" in tool.last_run["command"]


def test_convert_audio_x_m4a_bytes_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("audio/x-m4a", b"m4a data")
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")


def test_convert_audio_aac_bytes_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("audio/aac", b"aac data")
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")


def test_convert_audio_x_aiff_bytes_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("audio/x-aiff", b"aiff data")
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")


def test_convert_audio_webm_bytes_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("audio/webm", b"webm data")
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")


def test_convert_audio_x_ms_wma_bytes_success(tmp_path: Path) -> None:
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        result = sandbox.convert("audio/x-ms-wma", b"wma data")
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"audio_data", "audio/wav")


def test_convert_audio_tool_not_done_returns_none(tmp_path: Path) -> None:
    tool = MockToolAudio()
    tool._result = "FAILED"
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "audio.mp3"
        src.write_text("content")
        result = sandbox.convert("audio/mpeg", str(src))
    assert result is None


def test_convert_audio_no_output_file_returns_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    tool = MockToolAudioNoOutput()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "audio.mp3"
        src.write_text("content")
        result = sandbox.convert("audio/mpeg", str(src))
    assert result is None
    assert "Script reported DONE but audio output file" in caplog.text


def test_convert_audio_supported_mimetypes_included(tmp_path: Path) -> None:
    tool = MockTool()
    sandbox = ConversionSandbox(tool, str(tmp_path))
    supported = sandbox.supported_mimetypes()
    assert "audio/mpeg" in supported
    assert "audio/wav" in supported
    assert "audio/ogg" in supported
    assert "audio/flac" in supported


def test_convert_write_bytes_failure_returns_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Simulate OSError when writing input bytes."""
    caplog.set_level(logging.ERROR)
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        with (
            patch("pathlib.Path.write_bytes", side_effect=OSError("disk full")),
        ):
            result = sandbox.convert("application/pdf", b"pdf bytes")
    assert result is None
    assert "Failed to write input bytes" in caplog.text


def test_convert_copy_source_failure_returns_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Simulate OSError when copying source file."""
    caplog.set_level(logging.ERROR)
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("content")
        with patch("shutil.copy", side_effect=OSError("permission denied")):
            result = sandbox.convert("application/pdf", str(src))
    assert result is None
    assert "Failed to copy source file" in caplog.text


def test_convert_audio_read_failure_returns_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Simulate OSError when reading audio output file."""
    caplog.set_level(logging.ERROR)
    tool = MockToolAudio()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "audio.mp3"
        src.write_text("mp3 content")
        with patch("pathlib.Path.read_bytes", side_effect=OSError("read error")):
            result = sandbox.convert("audio/mpeg", str(src))
    assert result is None
    assert "Failed to read audio output file" in caplog.text


def test_convert_output_file_partial_read_failure_returns_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Simulate OSError when reading one of many output files - now returns None."""
    caplog.set_level(logging.WARNING)
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("content")

        original_read_bytes = Path.read_bytes
        call_count = 0

        def _side_effect(self_path: Path) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return original_read_bytes(self_path)
            raise OSError("read error on second file")

        with (
            patch("pathlib.Path.read_bytes", _side_effect),
            patch(
                "tiz.conversion_sandbox.ConversionSandbox._guess_mime",
                return_value="image/jpeg",
            ),
        ):
            result = sandbox.convert("application/pdf", str(src))
    assert result is None
    assert "Failed to read output file" in caplog.text


def test_convert_output_files_all_unreadable_returns_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Simulate OSError for all output files, resulting in no results."""
    caplog.set_level(logging.WARNING)
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("content")
        with patch(
            "pathlib.Path.read_bytes",
            side_effect=OSError("unreadable"),
        ):
            result = sandbox.convert("application/pdf", str(src))
    assert result is None
    assert "Failed to read output file" in caplog.text


def test_guess_mime_with_unknown_extension(tmp_path: Path) -> None:
    """Test _guess_mime falls back to application/octet-stream."""
    tool = MockTool()
    sandbox = ConversionSandbox(tool, str(tmp_path))
    result = sandbox._guess_mime(Path("file.unknown_extension_xyz"))
    assert result == "application/octet-stream"


def test_guess_mime_with_known_extension(tmp_path: Path) -> None:
    """Test _guess_mime returns correct MIME type for known extensions."""
    tool = MockTool()
    sandbox = ConversionSandbox(tool, str(tmp_path))
    assert sandbox._guess_mime(Path("test.jpg")) == "image/jpeg"
    assert sandbox._guess_mime(Path("test.png")) == "image/png"
    assert sandbox._guess_mime(Path("test.webp")) == "image/webp"
    assert sandbox._guess_mime(Path("test.bmp")) == "image/bmp"
    assert sandbox._guess_mime(Path("test.gif")) == "image/gif"
    assert sandbox._guess_mime(Path("test.tiff")) == "image/tiff"
    assert sandbox._guess_mime(Path("test.tif")) == "image/tiff"


def test_convert_with_webp_output(tmp_path: Path) -> None:
    """Test that .webp output files get the correct MIME type."""

    class MockToolWebP(MockTool):
        def run(self, args: dict[str, Any]) -> str:
            self.last_run = args
            command = args.get("command", "")
            cwd = args.get("cwd")
            parts = shlex.split(command)
            if len(parts) >= 3:
                out_dir = Path(cwd) / parts[-1] if cwd else Path(parts[-1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "frame.webp").write_bytes(b"webp_data")
            return self._result

    tool = MockToolWebP()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("content")
        result = sandbox.convert("application/pdf", str(src))
    assert result is not None
    assert len(result) == 1
    assert result[0] == (b"webp_data", "image/webp")


def test_convert_custom_timeout(tmp_path: Path) -> None:
    """Test that a custom timeout is used."""
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path), timeout=300)
        src = tmp_path / "input.pdf"
        src.write_text("content")
        result = sandbox.convert("application/pdf", str(src))
    assert result is not None
    assert tool.last_run is not None
    assert tool.last_run["timeout"] == 300


def test_convert_default_timeout(tmp_path: Path) -> None:
    """Test that the default timeout is 120."""
    tool = MockTool()
    with _patch_container_shared(tmp_path):
        sandbox = ConversionSandbox(tool, str(tmp_path))
        src = tmp_path / "input.pdf"
        src.write_text("content")
        result = sandbox.convert("application/pdf", str(src))
    assert result is not None
    assert tool.last_run is not None
    assert tool.last_run["timeout"] == 120


def test_guess_mime_audio_extensions(tmp_path: Path) -> None:
    """Test that _guess_mime maps audio extensions"""
    tool = MockTool()
    sandbox = ConversionSandbox(tool, str(tmp_path))
    assert sandbox._guess_mime(Path("output.wav")) == "audio/x-wav"
    assert sandbox._guess_mime(Path("output.mp3")) == "audio/mpeg"
    assert sandbox._guess_mime(Path("output.mp4")) == "video/mp4"
    assert sandbox._guess_mime(Path("output.pdf")) == "application/pdf"
