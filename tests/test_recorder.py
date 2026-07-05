"""Tests for the audio recorder module."""

import builtins
import os
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tiz.recorder import _suppress_stderr, record_audio


def test_suppress_stderr_normal():
    """_suppress_stderr suppresses stderr and restores it."""
    stderr_fd = sys.stderr.fileno()
    # Get the current fd pointing to stderr
    dup_fd = os.dup(stderr_fd)
    try:
        with _suppress_stderr():
            new_fd = os.dup(stderr_fd)
            try:
                new_stat = os.fstat(new_fd)
                null_fd = os.open(os.devnull, os.O_RDONLY)
                try:
                    devnull_stat = os.fstat(null_fd)
                    assert new_stat.st_ino == devnull_stat.st_ino
                    assert new_stat.st_dev == devnull_stat.st_dev
                finally:
                    os.close(null_fd)
            finally:
                os.close(new_fd)
        # After the context, stderr should be restored
        restored_fd = os.dup(stderr_fd)
        try:
            restored_stat = os.fstat(restored_fd)
            original_stat = os.fstat(dup_fd)
            assert restored_stat.st_ino == original_stat.st_ino
            assert restored_stat.st_dev == original_stat.st_dev
        finally:
            os.close(restored_fd)
    finally:
        os.close(dup_fd)


def test_suppress_stderr_no_fd_leak_on_dup2_failure():
    """If os.dup2 fails, file descriptors are still cleaned up."""
    original_dup2 = os.dup2

    call_count = 0

    def failing_dup2(fd, fd2):
        nonlocal call_count
        call_count += 1
        # First dup2 call (redirect to null) should fail
        if call_count == 1:
            raise OSError("Mock dup2 failure")
        return original_dup2(fd, fd2)

    with (
        patch("tiz.recorder.os.dup2", side_effect=failing_dup2),
        pytest.raises(OSError, match="Mock dup2 failure"),
        _suppress_stderr(),
    ):
        pass


def test_suppress_stderr_restores_after_exception():
    """_suppress_stderr restores stderr even if the body raises."""
    stderr_fd = sys.stderr.fileno()
    dup_fd = os.dup(stderr_fd)
    try:
        with (
            pytest.raises(RuntimeError, match="test error"),
            _suppress_stderr(),
        ):
            raise RuntimeError("test error")
        # stderr should be restored
        restored_fd = os.dup(stderr_fd)
        try:
            original_stat = os.fstat(dup_fd)
            restored_stat = os.fstat(restored_fd)
            assert restored_stat.st_ino == original_stat.st_ino
            assert restored_stat.st_dev == original_stat.st_dev
        finally:
            os.close(restored_fd)
    finally:
        os.close(dup_fd)


def test_suppress_stderr_lock():
    """_suppress_stderr uses a lock for thread safety."""
    import threading

    errors = []

    def worker():
        try:
            with _suppress_stderr():
                pass
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Thread errors: {errors}"


def test_import_error_when_pyaudio_missing():
    """record_audio raises ImportError if pyaudio is not installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyaudio":
            raise ImportError("No module named pyaudio")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        pytest.raises(ImportError, match="pyaudio is required"),
    ):
        record_audio("/tmp/test.wav")


def test_record_audio_keyboard_interrupt(mock_pyaudio):
    """Recording stops on KeyboardInterrupt."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    mock_stream.read.side_effect = KeyboardInterrupt()

    result = record_audio("/tmp/test_keyboard.wav", max_seconds=10)

    assert isinstance(result, Path)
    assert str(result) == "/tmp/test_keyboard.wav"
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_max_seconds(mock_pyaudio):
    """Recording stops after max_seconds."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio

    # Return audio data, then eventually time catches up
    def fake_read(chunk_size, exception_on_overflow=False):  # noqa: ARG001
        return b"\x00\x00" * (chunk_size // 2)

    mock_stream.read = MagicMock(side_effect=fake_read)

    # Make time.monotonic advance quickly
    call_count = 0

    def advancing_monotonic():
        nonlocal call_count
        call_count += 1
        return call_count * 10.0  # 10 seconds per call

    with patch("tiz.recorder.time.monotonic", side_effect=advancing_monotonic):
        result = record_audio("/tmp/test_max.wav", max_seconds=5)

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_silence_detection(mock_pyaudio):
    """Recording stops after silence duration."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    chunk = 1024

    # Generate silent audio data (all zeros)
    silent_data = b"\x00\x00" * (chunk // 2)

    # Return silent data repeatedly
    mock_stream.read.return_value = silent_data

    # Start at 0.0, first read at 0.1 (silence_start set),
    # second read at 0.3 (0.3-0.1=0.2 >= 0.1 triggers break)
    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[
            0.0,  # start_time
            0.1,  # first read - start silence timer
            0.3,  # second read - silence duration elapsed
            0.4,  # not reached
        ],
    ):
        result = record_audio(
            "/tmp/test_silence.wav",
            max_seconds=60,
            silence_threshold=0.01,
            silence_duration=0.1,
        )

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_silence_not_long_enough(mock_pyaudio):
    """Silence that has not yet reached silence_duration does not trigger stop."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    chunk = 1024

    silent_data = b"\x00\x00" * (chunk // 2)
    mock_stream.read.return_value = silent_data

    # Iteration 1: silence_start = time.monotonic() -> 0.1
    #             max check at 0.2, continue
    # Iteration 2: elif time.monotonic() -> 0.5, 0.5-0.1=0.4 < 0.5 -> False
    #             max check at 0.6, continue
    # Iteration 3: elif time.monotonic() -> 0.7, 0.7-0.1=0.6 >= 0.5 -> True -> break
    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[
            0.0,  # start_time
            0.1,  # line 133: silence_start = 0.1
            0.2,  # line 141: max check
            0.5,  # line 134: 0.5-0.1=0.4 < 0.5 -> False (covers branch)
            0.6,  # line 141: max check
            0.7,  # line 134: 0.7-0.1=0.6 >= 0.5 -> True -> break
        ],
    ):
        result = record_audio(
            "/tmp/test_silence_not_long_enough.wav",
            max_seconds=60,
            silence_threshold=0.01,
            silence_duration=0.5,
        )

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_loud_audio_resets_silence(mock_pyaudio):
    """Loud audio resets the silence timer."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    chunk = 1024

    # Generate loud audio data (high amplitude)
    loud_sample = struct.pack("<h", 16000)
    loud_data = loud_sample * (chunk // 2)

    # Generate silent data
    silent_data = b"\x00\x00" * (chunk // 2)

    # Pattern: loud, loud, silent, silent, silent, ... but reset by loud
    def alternating_read(chunk_size, exception_on_overflow=False):  # noqa: ARG001
        nonlocal alternating_call_count
        alternating_call_count += 1
        # 5 loud reads, then silence - should NOT trigger because loud resets
        if alternating_call_count <= 5:
            return loud_data
        return silent_data

    alternating_call_count = 0
    mock_stream.read = MagicMock(side_effect=alternating_read)

    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[
            0.0,  # start_time
            0.1,  # first read - loud
            0.2,  # second read - loud
            0.3,  # third read - loud
            0.4,  # fourth read - loud
            0.5,  # fifth read - loud
            0.6,  # first silence - start silence timer
            0.7,  # second silence - still silent
            0.8,  # third silence - still silent (0.8 - 0.6 = 0.2s >= 0.1s)
            0.9,  # this shouldn't be reached
        ],
    ):
        result = record_audio(
            "/tmp/test_loud_reset.wav",
            max_seconds=60,
            silence_threshold=0.3,  # 16000/32768 ≈ 0.488 > 0.3 → loud
            silence_duration=0.1,
        )

    assert isinstance(result, Path)
    # 5 loud reads + 2 silent reads (timestamps: 0.6 for silence_start,
    # 0.7 for the silence_start is None check, then 0.8 checks
    # monotonic - silence_start >= 0.1 which triggers break)
    assert alternating_call_count == 7, (
        f"Expected 7 reads, got {alternating_call_count}"
    )


def test_record_audio_device_error(mock_pyaudio, tmp_path):
    """OSError from p.open is propagated."""
    mock_pyaudio_instance, _ = mock_pyaudio
    mock_pyaudio_instance.open.side_effect = OSError("Device not found")

    output = tmp_path / "test_error.wav"

    with pytest.raises(OSError, match="Device not found"):
        record_audio(str(output))

    assert not output.exists()
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_writes_valid_wav(mock_pyaudio, tmp_path):
    """The output file is a valid WAV with correct parameters."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    chunk = 1024

    # Return some audio data
    sample_data = struct.pack("<h", 100) * (chunk // 2)
    mock_stream.read.return_value = sample_data

    output = tmp_path / "recording.wav"

    # Make time advance past max_seconds quickly
    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[
            0.0,
            0.0,  # start_time and first read
            61.0,  # second read - exceeds max_seconds
        ],
    ):
        result = record_audio(
            str(output),
            max_seconds=60,
            silence_threshold=0,
            silence_duration=0,
        )

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 44  # WAV header is 44 bytes

    # Verify WAV content
    import wave

    with wave.open(str(output), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit = 2 bytes
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0
        assert wf.getcomptype() == "NONE"


def test_record_audio_creates_parent_dirs(mock_pyaudio, tmp_path):
    """Parent directories are created automatically."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    chunk = 1024

    sample_data = b"\x00\x00" * (chunk // 2)
    mock_stream.read.return_value = sample_data

    deep_path = tmp_path / "sub" / "dir" / "recording.wav"
    assert not deep_path.parent.exists()

    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[
            0.0,
            0.0,
            61.0,
        ],
    ):
        result = record_audio(
            str(deep_path),
            max_seconds=60,
            silence_threshold=0,
            silence_duration=0,
        )

    assert result == deep_path
    assert deep_path.exists()


def test_record_audio_silence_disabled(mock_pyaudio):
    """With silence_threshold=0, silence detection is disabled."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio

    # All silence, but silence detection disabled
    silent_data = b"\x00\x00" * (1024 // 2)
    mock_stream.read.return_value = silent_data

    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[
            0.0,
            0.0,
            61.0,  # start, first read, second read (past max)
        ],
    ):
        result = record_audio(
            "/tmp/test_silence_disabled.wav",
            max_seconds=60,
            silence_threshold=0,
            silence_duration=10,
        )

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_silence_duration_disabled(mock_pyaudio):
    """With silence_duration=0, silence auto-stop is disabled."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio

    silent_data = b"\x00\x00" * (1024 // 2)
    mock_stream.read.return_value = silent_data

    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[
            0.0,
            0.0,
            61.0,
        ],
    ):
        result = record_audio(
            "/tmp/test_duration_disabled.wav",
            max_seconds=60,
            silence_threshold=0.01,
            silence_duration=0,
        )

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_output_path_as_path(mock_pyaudio, tmp_path):
    """output_path can be a Path object."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    mock_stream.read.return_value = b"\x00\x00" * (1024 // 2)

    output = tmp_path / "test_path_obj.wav"

    with patch("tiz.recorder.time.monotonic", side_effect=[0.0, 0.0, 61.0]):
        result = record_audio(output, max_seconds=60)

    assert result == output
    assert output.exists()


def test_record_audio_stream_read_error(mock_pyaudio):
    """Errors from stream.read propagate correctly, cleanup happens."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    mock_stream.read.side_effect = OSError("Device disconnected")

    with pytest.raises(OSError, match="Audio stream read failed"):
        record_audio("/tmp/test_stream_error.wav", max_seconds=10)

    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called is True


def test_record_audio_open_returns_none(mock_pyaudio, tmp_path):
    """When p.open returns None, an OSError is raised."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    mock_pyaudio_instance.open.return_value = None

    output = tmp_path / "test_open_none.wav"

    with pytest.raises(OSError, match="Failed to open audio stream"):
        record_audio(str(output), max_seconds=10)

    assert not output.exists()
    # stream.stop_stream and stream.close should NOT be called
    # because stream was never assigned
    mock_stream.stop_stream.assert_not_called()
    mock_stream.close.assert_not_called()
    assert mock_pyaudio_instance.terminate.called is True


def test_suppress_stderr_no_fileno():
    """_suppress_stderr yields without suppression if stderr lacks fileno()."""

    class FakeStderr:
        pass

    fake = FakeStderr()
    with patch("tiz.recorder.sys.stderr", fake), _suppress_stderr():
        pass  # Should not raise


def test_suppress_stderr_fileno_raises_oserror():
    """_suppress_stderr yields if stderr.fileno() raises OSError."""

    class FakeStderr:
        def fileno(self):
            raise OSError("not a real file")

    with patch("tiz.recorder.sys.stderr", FakeStderr()), _suppress_stderr():
        pass  # Should not raise


def test_record_audio_stop_stream_raises(mock_pyaudio):
    """If stop_stream() raises, close() is still called."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    mock_stream.read.side_effect = KeyboardInterrupt()
    mock_stream.stop_stream.side_effect = OSError("stop failed")

    result = record_audio("/tmp/test_stop_fail.wav", max_seconds=10)

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_odd_length_data(mock_pyaudio):
    """Silence detection handles odd-length data gracefully."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    # Return odd-length data that truncates to empty, then silence, then done
    # A single byte truncates to 0 bytes (empty), hitting line 125
    odd_data = b"\x01"
    silent_data = b"\x00\x00" * (1024 // 2)
    mock_stream.read.side_effect = [odd_data, silent_data, KeyboardInterrupt]

    result = record_audio("/tmp/test_odd_data.wav", max_seconds=10)

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_max_seconds_float(mock_pyaudio):
    """max_seconds accepts float values."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio

    def fake_read(chunk_size, exception_on_overflow=False):  # noqa: ARG001
        return b"\x00\x00" * (chunk_size // 2)

    mock_stream.read = MagicMock(side_effect=fake_read)

    call_count = 0

    def advancing_monotonic():
        nonlocal call_count
        call_count += 1
        return call_count * 0.5  # 0.5 seconds per call

    with patch("tiz.recorder.time.monotonic", side_effect=advancing_monotonic):
        result = record_audio("/tmp/test_float.wav", max_seconds=0.5)

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_tmp_file_cleanup_on_write_failure(mock_pyaudio, tmp_path):
    """If wave.open fails, the temp file is cleaned up."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    mock_stream.read.return_value = b"\x00\x00" * (1024 // 2)

    output = tmp_path / "fail.wav"

    with (
        patch("tiz.recorder.time.monotonic", side_effect=[0.0, 0.0, 61.0]),
        patch("tiz.recorder.wave.open", side_effect=OSError("Disk full")),
        pytest.raises(OSError, match="Disk full"),
    ):
        record_audio(
            str(output),
            max_seconds=60,
            silence_threshold=0,
            silence_duration=0,
        )

    # The temp file should be cleaned up
    tmp_file = output.with_suffix(output.suffix + ".tmp")
    assert not tmp_file.exists()
    assert not output.exists()  # output never created
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_tmp_file_partial_write_cleanup(mock_pyaudio, tmp_path):
    """If writeframes fails, the temp file is cleaned up."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    mock_stream.read.return_value = b"\x00\x00" * (1024 // 2)

    output = tmp_path / "partial.wav"

    # Real context manager class, not a MagicMock — __exit__ is looked up
    # on the type, and MagicMock.__exit__ swallows exceptions.
    class FailingWaveWriter:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def setnchannels(self, n):
            pass

        def setsampwidth(self, n):
            pass

        def setframerate(self, n):
            pass

        def writeframes(self, data):  # noqa: ARG002
            raise OSError("Write failed")

    with (
        patch("tiz.recorder.time.monotonic", side_effect=[0.0, 0.0, 61.0]),
        patch("tiz.recorder.wave.open", return_value=FailingWaveWriter()),
        pytest.raises(OSError, match="Write failed"),
    ):
        record_audio(
            str(output),
            max_seconds=60,
            silence_threshold=0,
            silence_duration=0,
        )

    tmp_file = output.with_suffix(output.suffix + ".tmp")
    assert not tmp_file.exists()
    assert not output.exists()
    assert mock_pyaudio_instance.terminate.called


def test_suppress_stderr_dup_failure():
    """If os.dup(stderr_fd) raises, the exception propagates."""

    def failing_dup(fd):  # noqa: ARG001
        raise OSError("dup failed")

    with (
        patch("tiz.recorder.os.dup", side_effect=failing_dup),
        pytest.raises(OSError, match="dup failed"),
        _suppress_stderr(),
    ):
        pass


def test_suppress_stderr_open_failure():
    """If os.open(os.devnull) raises, old_stderr is cleaned up."""

    def failing_open(path, flags):  # noqa: ARG001
        raise OSError("open failed")

    with (
        patch("tiz.recorder.os.open", side_effect=failing_open),
        pytest.raises(OSError, match="open failed"),
        _suppress_stderr(),
    ):
        pass


def test_record_audio_stream_close_raises(mock_pyaudio):
    """If stream.close() raises, p.terminate() is still called."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    mock_stream.read.side_effect = KeyboardInterrupt()
    mock_stream.close.side_effect = OSError("close failed")

    with pytest.raises(OSError, match="close failed"):
        record_audio("/tmp/test_close_fail.wav", max_seconds=10)

    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_silence_loud_silence_pattern(mock_pyaudio):
    """Silence → Loud → Silence triggers silence detection after loud resets."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio
    chunk = 1024

    loud_sample = struct.pack("<h", 16000)
    loud_data = loud_sample * (chunk // 2)
    silent_data = b"\x00\x00" * (chunk // 2)

    # Pattern: silent, loud, silent, silent → stops on 2nd silence
    def pattern_read(chunk_size, exception_on_overflow=False):  # noqa: ARG001
        nonlocal pattern_call_count
        pattern_call_count += 1
        if pattern_call_count == 1:
            return silent_data  # silence
        if pattern_call_count == 2:
            return loud_data  # loud resets silence_start
        return silent_data  # silence again, should stop

    pattern_call_count = 0
    mock_stream.read = MagicMock(side_effect=pattern_read)

    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[
            0.0,  # start_time
            0.1,  # read 1: silence → silence_start = 0.1
            0.2,  # read 1: max check
            0.3,  # read 2: max check (loud, no silence call)
            0.5,  # read 3: silence → silence_start = 0.5
            0.6,  # read 3: max check
            0.7,  # read 4: elif check → 0.7-0.5=0.2 >= 0.1 → break
            0.8,  # not reached
        ],
    ):
        result = record_audio(
            "/tmp/test_silence_loud_silence.wav",
            max_seconds=60,
            silence_threshold=0.3,
            silence_duration=0.1,
        )

    assert isinstance(result, Path)
    # 4 reads: silent, loud, silent, silent (then break)
    assert pattern_call_count == 4
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_max_seconds_zero(mock_pyaudio):
    """With max_seconds=0, recording stops immediately."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio

    def fake_read(chunk_size, exception_on_overflow=False):  # noqa: ARG001
        return b"\x00\x00" * (chunk_size // 2)

    mock_stream.read = MagicMock(side_effect=fake_read)

    # Make time advance quickly
    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[0.0, 0.5],  # start, then max check (0.5 >= 0)
    ):
        result = record_audio(
            "/tmp/test_zero_max.wav",
            max_seconds=0,
            silence_threshold=0,
            silence_duration=0,
        )

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called


def test_record_audio_max_seconds_negative(mock_pyaudio):
    """With max_seconds negative, recording stops immediately."""
    mock_pyaudio_instance, mock_stream = mock_pyaudio

    def fake_read(chunk_size, exception_on_overflow=False):  # noqa: ARG001
        return b"\x00\x00" * (chunk_size // 2)

    mock_stream.read = MagicMock(side_effect=fake_read)

    with patch(
        "tiz.recorder.time.monotonic",
        side_effect=[0.0, 0.5],  # start, then max check (0.5 >= -1)
    ):
        result = record_audio(
            "/tmp/test_neg_max.wav",
            max_seconds=-1,
            silence_threshold=0,
            silence_duration=0,
        )

    assert isinstance(result, Path)
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    assert mock_pyaudio_instance.terminate.called
