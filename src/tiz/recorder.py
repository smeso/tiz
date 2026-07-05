"""Audio recording utilities using pyaudio."""

import contextlib
import math
import os
import struct
import sys
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path

__all__ = ["record_audio"]

CHUNK: int = 1024
CHANNELS: int = 1

_stderr_lock = threading.Lock()


@contextlib.contextmanager
def _suppress_stderr() -> Iterator[None]:
    """Suppress stderr output from C libraries (e.g. ALSA)."""
    with _stderr_lock:
        try:
            stderr_fd = sys.stderr.fileno()
        except (AttributeError, OSError):
            yield
            return
        old_stderr = os.dup(stderr_fd)
        try:
            null_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(null_fd, stderr_fd)
            finally:
                os.close(null_fd)
            try:
                yield
            finally:
                os.dup2(old_stderr, stderr_fd)
        finally:
            os.close(old_stderr)


def record_audio(
    output_path: str | Path,
    max_seconds: float = 60,
    silence_threshold: float = 0.01,
    silence_duration: float = 5.0,
    sample_rate: int = 16000,
) -> Path:
    """Record audio from the microphone to a WAV file.

    Records PCM s16le mono audio at the specified sample rate.
    Stops on KeyboardInterrupt, after max_seconds, or after
    silence_duration seconds of continuous silence.

    Args:
        output_path: Path to save the WAV file.
        max_seconds: Maximum recording duration in seconds (default 60).
        silence_threshold: Normalized RMS amplitude below which is
            considered silence (default 0.01). Set to 0 to disable.
        silence_duration: Seconds of continuous silence before auto-stop
            (default 5.0). Set to 0 to disable.
        sample_rate: Sample rate in Hz (default 16000).

    Returns:
        Path to the recorded audio file.

    Raises:
        ImportError: If pyaudio is not installed.
        OSError: If the audio device cannot be opened.
    """
    try:
        import pyaudio
    except ImportError:
        msg = (
            "pyaudio is required for audio recording. "
            "Install it with: pip install tizbot[audio]"
        )
        raise ImportError(msg) from None

    output_path = Path(output_path)

    # Suppress ALSA/PulseAudio stderr noise from C library calls
    with _suppress_stderr():
        p = pyaudio.PyAudio()

    try:
        with _suppress_stderr():
            stream = p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=sample_rate,
                input=True,
                frames_per_buffer=CHUNK,
            )

        if stream is None:
            msg = "Failed to open audio stream"
            raise OSError(msg)

        frames: list[bytes] = []
        silence_start: float | None = None
        start_time = time.monotonic()

        print("Recording... Press Ctrl+C to stop.", file=sys.stderr)

        try:
            while True:
                with _suppress_stderr():
                    try:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                    except OSError as exc:
                        msg = f"Audio stream read failed: {exc}"
                        raise OSError(msg) from exc
                frames.append(data)

                # Silence detection — skip if data is empty
                if data and silence_duration > 0 and silence_threshold > 0:
                    # Ensure even-length data for struct.unpack
                    data = data[: len(data) & ~1]
                    if not data:
                        continue
                    values = struct.unpack(f"<{len(data) // 2}h", data)
                    sq_sum = sum(v * v for v in values)
                    rms = math.sqrt(sq_sum / len(values))
                    normalized_rms = rms / 32768.0

                    if normalized_rms < silence_threshold:
                        if silence_start is None:
                            silence_start = time.monotonic()
                        elif time.monotonic() - silence_start >= silence_duration:
                            print("Silence detected, stopping.", file=sys.stderr)
                            break
                    else:
                        silence_start = None

                # Max duration check
                if time.monotonic() - start_time >= max_seconds:
                    print(
                        f"Reached maximum recording time ({max_seconds}s).",
                        file=sys.stderr,
                    )
                    break

        except KeyboardInterrupt:
            print("\nRecording stopped by user.", file=sys.stderr)

        finally:
            with contextlib.suppress(Exception):
                stream.stop_stream()
            stream.close()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename to avoid partial/corrupt
        # output on write failure.
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(frames))
            tmp_path.replace(output_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    finally:
        p.terminate()

    return output_path
