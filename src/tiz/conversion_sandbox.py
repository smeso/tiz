"""Conversion sandbox for file format conversion."""

from __future__ import annotations

import shlex
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tiz.log import get_logger
from tiz.sandbox_container import CONTAINER_MOUNT_OWN_SHARED, CONTAINER_MOUNT_SCRIPTS

if TYPE_CHECKING:  # pragma: no cover
    import logging

    from tiz.tools.base import Tool

__all__ = ["ConversionSandbox", "MIMETYPE_TO_SCRIPT"]

MIMETYPE_TO_SCRIPT: dict[str, str] = {
    "application/msword": "doc2pics.sh",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "doc2pics.sh",
    "application/pdf": "pdf2pics.sh",
    "video/mp4": "video2pics.sh",
    "video/mpeg": "video2pics.sh",
    "video/quicktime": "video2pics.sh",
    "video/x-msvideo": "video2pics.sh",
    "video/x-matroska": "video2pics.sh",
    "audio/mpeg": "audio_convert.sh",
    "audio/wav": "audio_convert.sh",
    "audio/ogg": "audio_convert.sh",
    "audio/flac": "audio_convert.sh",
    "audio/x-wav": "audio_convert.sh",
    "audio/x-m4a": "audio_convert.sh",
    "audio/aac": "audio_convert.sh",
    "audio/x-aiff": "audio_convert.sh",
    "audio/webm": "audio_convert.sh",
    "audio/x-ms-wma": "audio_convert.sh",
}

_MIMETYPE_EXT: dict[str, str] = {
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/pdf": ".pdf",
    "video/mp4": ".mp4",
    "video/mpeg": ".mpeg",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-wav": ".wav",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/x-aiff": ".aiff",
    "audio/webm": ".webm",
    "audio/x-ms-wma": ".wma",
}

_EXT_MIMETYPE: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}
_EXT_MIMETYPE.update({ext: mime for mime, ext in _MIMETYPE_EXT.items()})

logger: logging.Logger = get_logger(__name__)


class ConversionSandbox:
    """Sandbox for converting files between formats using container scripts."""

    def __init__(self, tool: Tool, base_path: str, timeout: int = 120) -> None:
        """Initialize the sandbox.

        Parameters
        ----------
        tool:
            The tool instance for running conversion commands.
        base_path:
            Base directory for temporary conversion directories.
        timeout:
            Timeout in seconds for conversion commands.
        """
        self._tool = tool
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

    def supported_mimetypes(self) -> list[str]:
        """Return list of supported MIME types."""
        return list(MIMETYPE_TO_SCRIPT.keys())

    def _guess_mime(self, path: Path) -> str:
        """Guess the MIME type of a file based on its extension."""
        return _EXT_MIMETYPE.get(path.suffix.lower(), "application/octet-stream")

    def _prepare_input(
        self, mimetype: str, src_path: str | bytes, tmp_path: Path
    ) -> Path | None:
        """Prepare the input file in the temporary directory.

        Parameters
        ----------
        mimetype:
            The MIME type of the source file.
        src_path:
            Path to the source file, or raw bytes.
        tmp_path:
            The temporary directory path.

        Returns
        -------
        Path | None
            The path to the prepared input file, or None on failure.
        """
        if isinstance(src_path, bytes):
            ext = _MIMETYPE_EXT.get(mimetype, "")
            src_file = tmp_path / f"input{ext}"
            try:
                src_file.write_bytes(src_path)
            except OSError:
                logger.exception("Failed to write input bytes to %s", src_file)
                return None
        else:
            src_file = tmp_path / Path(src_path).name
            try:
                shutil.copy(src_path, src_file)
            except OSError:
                logger.exception(
                    "Failed to copy source file %s to %s",
                    src_path,
                    src_file,
                )
                return None
        return src_file

    def _read_audio_output(self, out_file: Path) -> list[tuple[bytes, str]] | None:
        """Read audio output file.

        Parameters
        ----------
        out_file:
            Path to the expected audio output file.

        Returns
        -------
        list[tuple[bytes, str]] | None
            A list of (data, mimetype) tuples, or None on failure.
        """
        if not out_file.exists():
            logger.warning(
                "Script reported DONE but audio output file %s does not exist",
                out_file,
            )
            return None
        try:
            return [(out_file.read_bytes(), "audio/wav")]
        except OSError:
            logger.exception("Failed to read audio output file %s", out_file)
            return None

    def _read_directory_output(self, out_dir: Path) -> list[tuple[bytes, str]] | None:
        """Read output files from a directory.

        Parameters
        ----------
        out_dir:
            The directory containing output files.

        Returns
        -------
        list[tuple[bytes, str]] | None
            A list of (data, mimetype) tuples, or None on failure.
        """
        output_files = sorted(out_dir.iterdir()) if out_dir.exists() else []
        if not output_files:
            logger.warning(
                "Script reported DONE but no output files found in %s",
                out_dir,
            )
            return None

        results: list[tuple[bytes, str]] = []
        for f in output_files:
            try:
                data = f.read_bytes()
            except OSError:
                logger.warning("Failed to read output file %s", f)
                return None
            results.append((data, self._guess_mime(f)))

        return results

    def convert(
        self, mimetype: str, src_path: str | bytes
    ) -> list[tuple[bytes, str]] | None:
        """Convert a file from the given MIME type.

        Parameters
        ----------
        mimetype:
            The MIME type of the source file.
        src_path:
            Path to the source file, or raw bytes of the source file.

        Returns
        -------
        list[tuple[bytes, str]] | None
            A list of (data, mimetype) tuples for the converted output files,
            or None if conversion failed.
        """
        if mimetype not in MIMETYPE_TO_SCRIPT:
            return None

        script = MIMETYPE_TO_SCRIPT[mimetype]

        with tempfile.TemporaryDirectory(dir=str(self._base_path)) as tmpdir:
            tmp_path = Path(tmpdir)

            src_file = self._prepare_input(mimetype, src_path, tmp_path)
            if src_file is None:
                return None

            is_audio = mimetype.startswith("audio/")
            if is_audio:
                out_file = tmp_path / "output.wav"
                cmd = (
                    f"{shlex.quote(f'{CONTAINER_MOUNT_SCRIPTS}/{script}')}"
                    f" {shlex.quote(src_file.name)}"
                    f" {shlex.quote(out_file.name)}"
                )
            else:
                out_dir = tmp_path / "output"
                cmd = (
                    f"{shlex.quote(f'{CONTAINER_MOUNT_SCRIPTS}/{script}')}"
                    f" {shlex.quote(src_file.name)}"
                    f" {shlex.quote(out_dir.name)}"
                )

            result = self._tool.run(
                {
                    "command": cmd,
                    "timeout": self._timeout,
                    "cwd": str(Path(CONTAINER_MOUNT_OWN_SHARED) / tmp_path.name),
                },
            )

            if "DONE" not in result:
                logger.info(result.strip())
                return None

            if is_audio:
                return self._read_audio_output(out_file)

            return self._read_directory_output(out_dir)
