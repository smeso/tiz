# mypy: disable-error-code="arg-type"
"""Tests for tiz.chat module."""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tiz.audio_inference_clients import AudioInferenceClient
from tiz.chat import Chat
from tiz.conversion_sandbox import ConversionSandbox
from tiz.manifest_parser import ConfirmationSpec


class DummyTool:
    @staticmethod
    def prompt() -> str:
        return json.dumps(
            {
                "name": "dummy_tool",
                "description": "A dummy tool",
                "parameters": {"type": "object", "properties": {}},
            }
        )

    @staticmethod
    def fname() -> str:
        return "dummy_tool"

    def run(self, args: dict) -> str:
        if not hasattr(self, "calls"):
            self.calls = [args]
        else:
            self.calls.append(args)
        return json.dumps({"result": "ok"})

    def format_confirmation(self, _args: dict, _markdown: bool = False) -> str | None:
        return None


class DummyToolError:
    @staticmethod
    def prompt() -> str:
        return json.dumps(
            {
                "name": "error_tool",
                "description": "A tool that errors",
                "parameters": {"type": "object", "properties": {}},
            }
        )

    @staticmethod
    def fname() -> str:
        return "error_tool"

    def run(self, args: dict) -> str:
        if not hasattr(self, "calls"):
            self.calls = [args]
        else:
            self.calls.append(args)
        return json.dumps({"error": "tool error"})

    def format_confirmation(self, _args: dict, _markdown: bool = False) -> str | None:
        return None


class MockInferenceClient:
    def __init__(
        self,
        context_size: int = 4096,
        chat_response: dict | None = None,
        input_modes: list[str] | None = None,
        token_count: int = 10,
        preserve_thinking: bool = False,
    ) -> None:
        self._context_size = context_size
        self._chat_response = chat_response or {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        }
        self._input_modes = input_modes or ["text", "image", "audio", "video", "file"]
        self._token_count = token_count
        self._preserve_thinking = preserve_thinking

    def get_context_size(self) -> int:
        return self._context_size

    def count_tokens(self, _messages: list) -> int:
        return self._token_count

    def input_modes(self) -> list[str]:
        return self._input_modes

    @property
    def preserve_thinking(self) -> bool:
        return self._preserve_thinking

    def chat(
        self,
        messages,  # noqa: ARG002
        sampling_params=None,  # noqa: ARG002
        update_callback=None,  # noqa: ARG002
        stream=False,  # noqa: ARG002
        timeout=None,  # noqa: ARG002
    ):
        return self._chat_response


class MockConversionSandbox(ConversionSandbox):
    def __init__(self) -> None:
        pass

    def supported_mimetypes(self) -> list[str]:
        return [
            "application/msword",
            "application/vnd.ms-excel",
            "video/mp4",
            "audio/wav",
            "audio/x-wav",
            "audio/mpeg",
        ]

    def convert(
        self, mimetype: str, _src_path: str | bytes
    ) -> list[tuple[bytes, str]] | None:
        if mimetype == "application/msword":
            return [(b"converted_image_data", "image/png")]
        if mimetype == "video/mp4":
            return [(b"video_frame_data", "image/jpeg")]
        if mimetype in ("audio/wav", "audio/x-wav", "audio/mpeg"):
            return [(b"converted_audio_data", "audio/wav")]
        return None


# ===========================================================================
# Chat __init__ tests
# ===========================================================================


def test_chat_init_defaults():
    client = MockInferenceClient()
    chat = Chat(client)
    assert chat.client is client
    assert chat.sys_prompt == "You are a helpful assistant."
    assert chat.update_callback is None
    assert chat.conv == [{"role": "system", "content": "You are a helpful assistant."}]
    assert chat.tools is None
    assert chat.tool_definitions == []
    assert chat.conversion_sandbox is None
    assert chat.ctx_size is None
    assert chat.ctx_ratio == 0.9


def test_chat_init_with_custom_sys_prompt():
    client = MockInferenceClient()
    chat = Chat(client, sys_prompt="Custom prompt")
    assert chat.sys_prompt == "Custom prompt"
    assert chat.conv == [{"role": "system", "content": "Custom prompt"}]


def test_chat_init_with_tools():
    client = MockInferenceClient()
    tools = [DummyTool()]
    chat = Chat(client, tools=tools)
    assert chat.tools is not None
    assert "dummy_tool" in chat.tools
    assert len(chat.tool_definitions) == 1
    assert chat.tool_definitions[0] == {
        "type": "function",
        "function": {
            "name": "dummy_tool",
            "description": "A dummy tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_chat_init_with_multiple_tools():
    client = MockInferenceClient()
    tools = [DummyTool(), DummyToolError()]
    chat = Chat(client, tools=tools)
    assert len(chat.tools) == 2
    assert len(chat.tool_definitions) == 2
    assert chat.tool_definitions == [
        {
            "type": "function",
            "function": {
                "name": "dummy_tool",
                "description": "A dummy tool",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "error_tool",
                "description": "A tool that errors",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def test_chat_init_with_update_callback():
    client = MockInferenceClient()
    callback = MagicMock()
    chat = Chat(client, update_callback=callback)
    assert chat.update_callback is callback


def test_chat_init_with_conversion_sandbox():
    client = MockInferenceClient()
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    assert chat.conversion_sandbox is sandbox


def test_chat_init_with_custom_ctx_ratio():
    client = MockInferenceClient()
    chat = Chat(client, ctx_ratio=0.5)
    assert chat.ctx_ratio == 0.5


def test_chat_init_ctx_ratio_zero_raises():
    client = MockInferenceClient()
    with pytest.raises(ValueError, match="ctx_ratio must be in"):
        Chat(client, ctx_ratio=0.0)


def test_chat_init_ctx_ratio_negative_raises():
    client = MockInferenceClient()
    with pytest.raises(ValueError, match="ctx_ratio must be in"):
        Chat(client, ctx_ratio=-0.1)


def test_chat_init_ctx_ratio_greater_than_one_raises():
    client = MockInferenceClient()
    with pytest.raises(ValueError, match="ctx_ratio must be in"):
        Chat(client, ctx_ratio=1.5)


def test_chat_init_ctx_ratio_one_is_valid():
    client = MockInferenceClient()
    chat = Chat(client, ctx_ratio=1.0)
    assert chat.ctx_ratio == 1.0


def test_chat_init_with_confirm_callback():
    client = MockInferenceClient()

    def confirm(_args: dict, _fmt: object, _subtask: str | None = None) -> bool:
        return True

    chat = Chat(client, confirm_callback=confirm)
    assert chat.confirm_callback is confirm
    assert chat.confirm_callback({"foo": "bar"}, lambda _a, _m: None, None) is True


def test_chat_init_confirm_callback_defaults_to_false():
    client = MockInferenceClient()
    chat = Chat(client)
    assert chat.confirm_callback is not None
    assert chat.confirm_callback({"any": "args"}, lambda _a, _m: None, None) is False
    assert chat.confirm_callback({}, lambda _a, _m: None, None) is False


def test_chat_init_confirm_callback_with_none():
    client = MockInferenceClient()
    chat = Chat(client, confirm_callback=None)
    assert chat.confirm_callback is not None
    assert chat.confirm_callback({"test": 123}, lambda _a, _m: None, None) is False


def test_chat_init_with_audio_inference_client():
    client = MockInferenceClient()

    class DummyAudioClient(AudioInferenceClient):
        def transcribe(self, audio: bytes | str) -> str:  # noqa: ARG002
            return ""

    audio_client = DummyAudioClient()
    chat = Chat(client, audio_inference_client=audio_client)
    assert chat.audio_inference_client is audio_client


def test_chat_init_without_audio_inference_client():
    client = MockInferenceClient()
    chat = Chat(client)
    assert chat.audio_inference_client is None


# ===========================================================================
# append_pdf tests
# ===========================================================================


# ===========================================================================
# reset_usage tests
# ===========================================================================


def test_reset_usage():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.usage["prompt_tokens"] = 100
    chat.usage["completion_tokens"] = 100
    chat.usage["prompt_time"] = 100
    chat.usage["completion_time"] = 100
    chat.usage["cost"] = 100
    chat.reset_usage()
    assert chat.usage["prompt_tokens"] == 0
    assert chat.usage["completion_tokens"] == 0
    assert chat.usage["prompt_time"] == 0
    assert chat.usage["completion_time"] == 0
    assert chat.usage["cost"] == 0


# ===========================================================================
# append tests
# ===========================================================================


def test_append_basic():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    assert chat.conv[-1] == {"role": "user", "content": "hello"}


def test_append_with_extras():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("assistant", "hi", extras={"reasoning": "thought"})
    assert chat.conv[-1] == {
        "role": "assistant",
        "content": "hi",
        "reasoning": "thought",
    }


def test_append_with_none_message():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("assistant", None)
    assert chat.conv[-1]["content"] is None


def test_append_with_none_extras():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello", extras=None)
    assert chat.conv[-1] == {"role": "user", "content": "hello"}


def test_append_with_empty_extras():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello", extras={})
    assert chat.conv[-1] == {"role": "user", "content": "hello"}


def test_append_with_extras_content_key_collision():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello", extras={"content": "overridden", "extra_key": "val"})
    assert chat.conv[-1] == {
        "role": "user",
        "content": "overridden",
        "extra_key": "val",
    }


# ===========================================================================
# append_media tests
# ===========================================================================


def test_append_media_with_path(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"fake_image_data")
    chat.append_media("image", path=str(img_path))
    assert len(chat.conv) == 2
    msg = chat.conv[-1]
    assert msg == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,ZmFrZV9pbWFnZV9kYXRh"},
            }
        ],
    }


def test_append_media_with_data():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_media("image", data=b"fake_data", mimetype="image/png")
    assert len(chat.conv) == 2
    msg = chat.conv[-1]
    assert msg == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,ZmFrZV9kYXRh"},
            }
        ],
    }


def test_append_media_no_path_no_data_raises():
    client = MockInferenceClient()
    chat = Chat(client)
    with pytest.raises(ValueError, match="Either path or data must be provided"):
        chat.append_media("image")


def test_append_media_no_mimetype_no_path_raises():
    client = MockInferenceClient()
    chat = Chat(client)
    with pytest.raises(ValueError, match="mimetype must be provided or guessable"):
        chat.append_media("image", data=b"data")


def test_append_media_guesses_mimetype_from_path(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake_jpeg")
    chat.append_media("image", path=str(img_path))
    msg = chat.conv[-1]
    assert msg == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,ZmFrZV9qcGVn"},
            }
        ],
    }


def test_append_media_path_unguessable_mimetype_raises(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    unknown_path = tmp_path / "file.unknownxyz"
    unknown_path.write_bytes(b"data")
    with pytest.raises(ValueError, match="mimetype must be provided or guessable"):
        chat.append_media("image", path=str(unknown_path))


# ===========================================================================
# append_image tests
# ===========================================================================


def test_append_image_with_path(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    img_path = tmp_path / "photo.png"
    img_path.write_bytes(b"png_data")
    chat.append_image(path=str(img_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,cG5nX2RhdGE="},
            }
        ],
    }


def test_append_image_with_data():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_image(data=b"img_data", mimetype="image/gif")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/gif;base64,aW1nX2RhdGE="},
            }
        ],
    }


# ===========================================================================
# append_video tests
# ===========================================================================


def test_append_video_with_path(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    vid_path = tmp_path / "clip.mp4"
    vid_path.write_bytes(b"mp4_data")
    chat.append_video(path=str(vid_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,bXA0X2RhdGE="},
            }
        ],
    }


def test_append_video_with_data():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_video(data=b"vid_data", mimetype="video/mp4")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,dmlkX2RhdGE="},
            }
        ],
    }


# ===========================================================================
# append_audio tests
# ===========================================================================


def test_append_audio_with_path(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    audio_path = tmp_path / "sound.wav"
    audio_path.write_bytes(b"wav_data")
    chat.append_audio(path=str(audio_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "d2F2X2RhdGE=", "format": "wav"},
            }
        ],
    }


def test_append_audio_with_data_wav():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_audio(data=b"wav_data", mimetype="audio/wav")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "d2F2X2RhdGE=", "format": "wav"},
            }
        ],
    }


def test_append_audio_with_data_mp3():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_audio(data=b"mp3_data", mimetype="audio/mpeg")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "bXAzX2RhdGE=", "format": "mp3"},
            }
        ],
    }


def test_append_audio_with_data_mp3_direct():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_audio(data=b"mp3_data", mimetype="audio/mp3")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "bXAzX2RhdGE=", "format": "mp3"},
            }
        ],
    }


def test_append_audio_with_data_ogg():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_audio(data=b"ogg_data", mimetype="audio/ogg")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "b2dnX2RhdGE=", "format": "ogg"},
            }
        ],
    }


def test_append_audio_with_data_flac():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_audio(data=b"flac_data", mimetype="audio/flac")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "ZmxhY19kYXRh", "format": "flac"},
            }
        ],
    }


def test_append_audio_with_data_unknown_mimetype():
    client = MockInferenceClient()
    chat = Chat(client)
    with pytest.raises(ValueError, match="Unsupported or missing audio format"):
        chat.append_audio(data=b"data", mimetype="audio/xyz")


def test_append_audio_no_path_no_data_raises():
    client = MockInferenceClient()
    chat = Chat(client)
    with pytest.raises(ValueError, match="Either path or data must be provided"):
        chat.append_audio()


def test_append_audio_guesses_mimetype(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    audio_path = tmp_path / "sound.mp3"
    audio_path.write_bytes(b"mp3_data")
    chat.append_audio(path=str(audio_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "bXAzX2RhdGE=", "format": "mp3"},
            }
        ],
    }


def test_append_audio_data_no_mimetype():
    client = MockInferenceClient()
    chat = Chat(client)
    with pytest.raises(ValueError, match="Unsupported or missing audio format"):
        chat.append_audio(data=b"data")


def test_append_audio_path_unguessable_mimetype(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    audio_path = tmp_path / "sound.unknownxyz"
    audio_path.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported or missing audio format"):
        chat.append_audio(path=str(audio_path))


def test_append_audio_with_sandbox_converts_wav(tmp_path: Path):
    client = MockInferenceClient()
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    audio_path = tmp_path / "sound.wav"
    audio_path.write_bytes(b"wav_data")
    chat.append_audio(path=str(audio_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "Y29udmVydGVkX2F1ZGlvX2RhdGE=",
                    "format": "wav",
                },
            }
        ],
    }


def test_append_audio_with_sandbox_converts_mpeg(tmp_path: Path):
    client = MockInferenceClient()
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    audio_path = tmp_path / "sound.mp3"
    audio_path.write_bytes(b"mp3_data")
    chat.append_audio(path=str(audio_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "Y29udmVydGVkX2F1ZGlvX2RhdGE=",
                    "format": "wav",
                },
            }
        ],
    }


def test_append_audio_with_sandbox_converts_data():
    client = MockInferenceClient()
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    chat.append_audio(data=b"wav_data", mimetype="audio/wav")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "Y29udmVydGVkX2F1ZGlvX2RhdGE=",
                    "format": "wav",
                },
            }
        ],
    }


def test_append_audio_with_sandbox_convert_fails_raises():
    sandbox = MockConversionSandbox()
    sandbox.convert = lambda _m, _s: None  # type: ignore[method-assign]
    client = MockInferenceClient()
    chat = Chat(client, conversion_sandbox=sandbox)
    with pytest.raises(ValueError, match="Audio conversion failed for audio/wav"):
        chat.append_audio(data=b"wav_data", mimetype="audio/wav")


def test_append_audio_with_sandbox_unsupported_mimetype(tmp_path: Path):
    client = MockInferenceClient()
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    audio_path = tmp_path / "sound.ogg"
    audio_path.write_bytes(b"ogg_data")
    chat.append_audio(path=str(audio_path))
    # Unsupported mimetype - sandbox not used, passes through as-is
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "b2dnX2RhdGE=", "format": "ogg"},
            }
        ],
    }


def test_append_audio_no_sandbox_works_normally(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)  # no sandbox
    audio_path = tmp_path / "sound.mp3"
    audio_path.write_bytes(b"mp3_data")
    chat.append_audio(path=str(audio_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "bXAzX2RhdGE=", "format": "mp3"},
            }
        ],
    }


def test_append_audio_with_audio_inference_client(tmp_path: Path):
    client = MockInferenceClient()

    class DummyAudioClient(AudioInferenceClient):
        def transcribe(self, audio: bytes | str) -> str:  # noqa: ARG002
            return "  transcribed text  "

    audio_client = DummyAudioClient()
    chat = Chat(client, audio_inference_client=audio_client)
    audio_path = tmp_path / "speech.mp3"
    audio_path.write_bytes(b"mp3_data")
    chat.append_audio(path=str(audio_path))
    assert chat.conv[-1] == {"role": "user", "content": "transcribed text"}


def test_append_audio_with_audio_inference_client_and_callback():
    client = MockInferenceClient()
    callback_calls = []

    def callback(msg, _subtask_name=None):
        callback_calls.append((msg, _subtask_name))

    class DummyAudioClient(AudioInferenceClient):
        def transcribe(self, audio: bytes | str) -> str:  # noqa: ARG002
            return "  hello world  "

    audio_client = DummyAudioClient()
    chat = Chat(
        client,
        audio_inference_client=audio_client,
        update_callback=callback,
    )
    chat.append_audio(data=b"wav_data", mimetype="audio/wav")
    assert chat.conv[-1] == {"role": "user", "content": "hello world"}
    assert callback_calls == [({"tiz-internal": {"transcribe": "hello world"}}, None)]


def test_append_audio_with_audio_inference_client_and_sandbox(tmp_path: Path):
    client = MockInferenceClient()
    sandbox = MockConversionSandbox()

    class DummyAudioClient(AudioInferenceClient):
        def transcribe(self, audio: bytes | str) -> str:
            assert audio == b"converted_audio_data"
            return "transcribed from converted"

    audio_client = DummyAudioClient()
    chat = Chat(
        client,
        conversion_sandbox=sandbox,
        audio_inference_client=audio_client,
    )
    audio_path = tmp_path / "speech.mp3"
    audio_path.write_bytes(b"mp3_data")
    chat.append_audio(path=str(audio_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": "transcribed from converted",
    }


def test_append_audio_with_audio_inference_client_no_callback():
    client = MockInferenceClient()

    class DummyAudioClient(AudioInferenceClient):
        def transcribe(self, audio: bytes | str) -> str:  # noqa: ARG002
            return "  hello world  "

    audio_client = DummyAudioClient()
    chat = Chat(client, audio_inference_client=audio_client)
    chat.append_audio(data=b"wav_data", mimetype="audio/wav")
    assert chat.conv[-1] == {"role": "user", "content": "hello world"}


def test_append_pdf_with_path(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"pdf_data")
    chat.append_pdf(path=str(pdf_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {"filename": "doc.pdf", "file_data": "cGRmX2RhdGE="},
            }
        ],
    }


def test_append_pdf_with_data():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append_pdf(data=b"pdf_data")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {"type": "file", "file": {"filename": "", "file_data": "cGRmX2RhdGE="}}
        ],
    }


def test_append_pdf_no_path_no_data_raises():
    client = MockInferenceClient()
    chat = Chat(client)
    with pytest.raises(ValueError, match="Either path or data must be provided"):
        chat.append_pdf()


# ===========================================================================
# append_file tests
# ===========================================================================


def test_append_file_audio_when_supported(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "audio"])
    chat = Chat(client)
    audio_path = tmp_path / "sound.wav"
    audio_path.write_bytes(b"wav_data")
    chat.append_file(str(audio_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "d2F2X2RhdGE=", "format": "wav"},
            }
        ],
    }


def test_append_file_video_when_supported(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "video"])
    chat = Chat(client)
    vid_path = tmp_path / "clip.mp4"
    vid_path.write_bytes(b"mp4_data")
    chat.append_file(str(vid_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,bXA0X2RhdGE="},
            }
        ],
    }


def test_append_file_pdf_when_supported(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "file"])
    chat = Chat(client)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"pdf_data")
    chat.append_file(str(pdf_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {"filename": "doc.pdf", "file_data": "cGRmX2RhdGE="},
            }
        ],
    }


def test_append_file_image_when_supported(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    chat = Chat(client)
    img_path = tmp_path / "photo.png"
    img_path.write_bytes(b"png_data")
    chat.append_file(str(img_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,cG5nX2RhdGE="},
            }
        ],
    }


def test_append_file_convert_via_sandbox(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    doc_path = tmp_path / "document.doc"
    doc_path.write_bytes(b"doc_data")
    chat.append_file(str(doc_path))
    assert chat.conv[-2] == {
        "role": "user",
        "content": "These are pages from the document document.doc.",
    }
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,Y29udmVydGVkX2ltYWdlX2RhdGE="
                },
            }
        ],
    }


def test_append_file_sandbox_convert_returns_none(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    xls_path = tmp_path / "data.xls"
    xls_path.write_bytes(b"xls_data")
    # application/vnd.ms-excel is in sandbox.supported_mimetypes but
    # MockConversionSandbox returns None for it
    with pytest.raises(ValueError, match="Unsupported file type"):
        chat.append_file(str(xls_path))


def test_append_file_unsupported_type_raises(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text"])
    chat = Chat(client)
    doc_path = tmp_path / "data.doc"
    doc_path.write_bytes(b"doc_data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        chat.append_file(str(doc_path))


def test_append_file_text_plain(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text"])
    chat = Chat(client)
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("hello world")
    chat.append_file(str(txt_path))
    assert chat.conv[-1] == {"role": "user", "content": "hello world"}


def test_append_file_text_plain_with_data(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text"])
    chat = Chat(client)
    txt_path = tmp_path / "notes.txt"
    chat.append_file(str(txt_path), data=b"data from bytes")
    assert chat.conv[-1] == {"role": "user", "content": "data from bytes"}


def test_append_file_unknown_mimetype_raises(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    unknown_path = tmp_path / "file.xyz123unknown"
    unknown_path.write_bytes(b"data")
    with pytest.raises(ValueError, match="Could not determine mimetype"):
        chat.append_file(str(unknown_path))


def test_append_file_with_data_audio(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "audio"])
    chat = Chat(client)
    audio_path = tmp_path / "sound.wav"
    chat.append_file(str(audio_path), data=b"wav_data")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "d2F2X2RhdGE=", "format": "wav"},
            }
        ],
    }


def test_append_file_with_data_video(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "video"])
    chat = Chat(client)
    vid_path = tmp_path / "clip.mp4"
    chat.append_file(str(vid_path), data=b"mp4_data")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,bXA0X2RhdGE="},
            }
        ],
    }


def test_append_file_with_data_pdf(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "file"])
    chat = Chat(client)
    pdf_path = tmp_path / "doc.pdf"
    chat.append_file(str(pdf_path), data=b"pdf_data")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {"filename": "doc.pdf", "file_data": "cGRmX2RhdGE="},
            }
        ],
    }


def test_append_file_with_data_image(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    chat = Chat(client)
    img_path = tmp_path / "photo.png"
    chat.append_file(str(img_path), data=b"png_data")
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,cG5nX2RhdGE="},
            }
        ],
    }


def test_append_file_sandbox_with_data_convert(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    doc_path = tmp_path / "document.doc"
    chat.append_file(str(doc_path), data=b"doc_data")
    assert chat.conv[-2] == {
        "role": "user",
        "content": "These are pages from the document document.doc.",
    }
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,Y29udmVydGVkX2ltYWdlX2RhdGE="
                },
            }
        ],
    }


def test_append_file_convert_video_via_sandbox(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    vid_path = tmp_path / "clip.mp4"
    vid_path.write_bytes(b"mp4_data")
    chat.append_file(str(vid_path))
    assert chat.conv[-2] == {
        "role": "user",
        "content": "These are frames from the video file clip.mp4.",
    }
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,dmlkZW9fZnJhbWVfZGF0YQ=="},
            }
        ],
    }


def test_append_file_convert_video_via_sandbox_with_data(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    vid_path = tmp_path / "clip.mp4"
    chat.append_file(str(vid_path), data=b"mp4_data")
    assert chat.conv[-2] == {
        "role": "user",
        "content": "These are frames from the video file clip.mp4.",
    }
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,dmlkZW9fZnJhbWVfZGF0YQ=="},
            }
        ],
    }


def test_append_file_audio_not_routed_through_image_via_sandbox(tmp_path: Path):
    """Audio should not be routed through append_image even with sandbox."""
    client = MockInferenceClient(input_modes=["text", "image"])
    sandbox = MockConversionSandbox()
    chat = Chat(client, conversion_sandbox=sandbox)
    audio_path = tmp_path / "sound.mp3"
    audio_path.write_bytes(b"mp3_data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        chat.append_file(str(audio_path))


def test_append_file_with_contents(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    chat = Chat(client)
    img_path = tmp_path / "photo.png"
    img_path.write_bytes(b"png_from_contents")
    chat.append_file(str(img_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,cG5nX2Zyb21fY29udGVudHM="},
            }
        ],
    }


def test_append_file_with_contents_and_filename(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "file"])
    chat = Chat(client)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"pdf_data")
    chat.append_file(str(pdf_path))
    assert chat.conv[-1] == {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {"filename": "doc.pdf", "file_data": "cGRmX2RhdGE="},
            }
        ],
    }


# ===========================================================================
# send_message tests
# ===========================================================================


def test_send_message_basic():
    client = MockInferenceClient()
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result == {
        "reasoning": "",
        "message": "hi",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
        "tool_calls": [],
    }


def test_send_message_with_files(tmp_path: Path):
    client = MockInferenceClient(input_modes=["text", "image"])
    chat = Chat(client)
    img_path = tmp_path / "photo.png"
    img_path.write_bytes(b"png_data")
    result = chat.send_message("look at this", files=[str(img_path)])
    assert result == {
        "reasoning": "",
        "message": "hi",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
        "tool_calls": [],
    }


def test_send_message_empty_choices():
    client = MockInferenceClient(chat_response={"choices": []})
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result == {
        "reasoning": "",
        "message": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
        "tool_calls": [],
    }


def test_send_message_no_message_in_choice():
    client = MockInferenceClient(chat_response={"choices": [{"other": "data"}]})
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result == {
        "reasoning": "",
        "message": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
        "tool_calls": [],
    }


def test_send_message_with_tool_calls():
    tool = DummyTool()

    class TwoStepClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"test": "ok"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    two_step = TwoStepClient()
    chat = Chat(two_step, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert hasattr(tool, "calls")
    assert tool.calls == [{"test": "ok"}]


def test_send_message_tool_call_malformed_json(caplog):
    class MalformedToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": "{bad json}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    tool = DummyTool()
    client = MalformedToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert chat.conv[-2] == {
        "role": "tool",
        "content": '{"error": "malformed JSON in tool arguments"}',
        "tool_call_id": "tc1",
        "name": "dummy_tool",
    }
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Malformed JSON in tool call" in caplog.records[0].message


def test_send_message_tool_call_unknown_tool(caplog):
    class UnknownToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "nonexistent_tool",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    tool = DummyTool()
    client = UnknownToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert chat.conv[-2] == {
        "role": "tool",
        "content": "ERROR: command does not exist",
        "tool_call_id": "tc1",
        "name": "nonexistent_tool",
    }
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Unknown tool call" in caplog.records[0].message


def test_send_message_with_usage_and_timings():
    client = MockInferenceClient(
        chat_response={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost": 0.01,
            },
            "timings": {
                "prompt_time": 1.5,
                "completion_time": 2.0,
            },
        }
    )
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 5
    assert result["cost"] == pytest.approx(0.01)
    assert result["prompt_time"] == pytest.approx(1.5)
    assert result["completion_time"] == pytest.approx(2.0)


def test_send_message_no_usage_no_timings():
    client = MockInferenceClient(
        chat_response={"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    )
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result["prompt_tokens"] == 0
    assert result["completion_time"] == 0
    assert result["completion_tokens"] == 0
    assert result["cost"] == 0
    assert result["prompt_time"] == 0


def test_send_message_with_reasoning_content():
    client = MockInferenceClient(
        chat_response={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "thinking",
                    }
                }
            ]
        }
    )
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result["reasoning"] == "thinking"
    assert result["message"] == "answer"


def test_send_message_tool_calls_without_tools_defined():
    class ToolCallNoToolsClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "tool_calls": [
                                {
                                    "id": "tc1",
                                    "function": {
                                        "name": "dummy_tool",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

    client = ToolCallNoToolsClient()
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result["message"] == "done"
    assert result["tool_calls"] == []


def test_send_message_with_timeout():
    client = MockInferenceClient()
    chat = Chat(client)
    with patch.object(client, "chat", wraps=client.chat) as mock_chat:
        chat.send_message("hello", timeout=30.0)
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["timeout"] == 30.0


# ===========================================================================
# compress_context tests
# ===========================================================================


def test_compress_context():
    class CompressClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "summary"}}
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "final"}}]}

    client = CompressClient()
    chat = Chat(client)
    chat.append("user", "msg1")
    chat.append("assistant", "resp1")
    chat.append("user", "msg2")
    chat.append("assistant", "resp2")
    chat.append("user", "msg3")
    chat.append("assistant", "resp3")
    chat.append("user", "msg4")
    chat.append("assistant", "resp4")

    chat.compress_context()
    assert len(chat.conv) == 7
    assert chat.conv[2] == {
        "role": "system",
        "content": "[Context compressed: 3 previous messages summarized]\nsummary",
    }


def test_compress_context_returns_usage():
    class UsageClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "summary"}}
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 10,
                        "cost": 0.02,
                    },
                    "timings": {"prompt_time": 1.0, "completion_time": 0.5},
                }
            return {"choices": [{"message": {"role": "assistant", "content": "final"}}]}

    client = UsageClient()
    chat = Chat(client)
    chat.append("user", "msg1")
    chat.append("assistant", "resp1")
    chat.append("user", "msg2")
    chat.append("assistant", "resp2")
    chat.append("user", "msg3")
    chat.append("assistant", "resp3")
    chat.append("user", "msg4")
    chat.append("assistant", "resp4")

    usage = chat.compress_context()
    assert usage["prompt_tokens"] == 50
    assert usage["completion_tokens"] == 10
    assert usage["cost"] == 0.02
    assert usage["prompt_time"] == 1.0
    assert usage["completion_time"] == 0.5
    assert usage["cached_tokens"] == 0
    assert usage["cache_write_tokens"] == 0


def test_compress_context_small_conv_less_than_4_returns_empty_usage():
    client = MockInferenceClient()
    chat = Chat(client)
    # conv has only 1 message (system)
    assert len(chat.conv) == 1
    result = chat.compress_context()
    assert result == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
    }
    assert len(chat.conv) == 1  # unchanged


def test_compress_context_small_conv_len_2_returns_empty_usage():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    assert len(chat.conv) == 2
    result = chat.compress_context()
    assert result == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
    }
    assert len(chat.conv) == 2  # unchanged


def test_compress_context_small_conv_len_3_returns_empty_usage():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    chat.append("assistant", "hi")
    assert len(chat.conv) == 3
    result = chat.compress_context()
    assert result == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
    }
    assert len(chat.conv) == 3  # unchanged


def test_compress_context_off_by_one_fixed():
    """Verify the off-by-one bug is fixed: no message is silently dropped."""

    class OffByOneClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "summary"}}
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "final"}}]}

    client = OffByOneClient()
    chat = Chat(client)
    # Add messages A..J (10 items total including system)
    # system, user:A, assistant:B, user:C, assistant:D, user:E, assistant:F,
    # user:G, assistant:H, user:I, assistant:J
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for i, label in enumerate(labels):
        role = "user" if i % 2 == 0 else "assistant"
        chat.append(role, f"msg{label}")

    assert len(chat.conv) == 11  # system + 10 messages
    chat.compress_context()
    # After fix: no message should be silently dropped.
    # conv = [system, user:A, summary, assistant:F, user:G, assistant:H, user:I, assistant:J]
    assert len(chat.conv) == 8
    # Before fix, assistant:msgF (index 6 = 2+i) was dropped
    assert chat.conv[3] == {"role": "assistant", "content": "msgF"}
    assert chat.conv[4] == {"role": "user", "content": "msgG"}
    assert chat.conv[5] == {"role": "assistant", "content": "msgH"}


def test_compress_context_when_send_message_returns_none_message():
    """compress_context handles None message from send_message gracefully."""

    class NoneSummaryClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [{"message": {"role": "assistant", "content": None}}]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "final"}}]}

    client = NoneSummaryClient()
    chat = Chat(client)
    chat.append("user", "msg1")
    chat.append("assistant", "resp1")
    chat.append("user", "msg2")
    chat.append("assistant", "resp2")
    chat.append("user", "msg3")
    chat.append("assistant", "resp3")
    chat.append("user", "msg4")
    chat.append("assistant", "resp4")

    chat.compress_context()
    # Should not crash - summary with None content becomes "None"
    assert len(chat.conv) == 7
    assert chat.conv[2]["role"] == "system"
    assert "[Context compressed:" in chat.conv[2]["content"]


def test_send_context_compression_propagates_usage():
    class CompressionUsageClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 100

        def count_tokens(self, messages):  # noqa: ARG002
            self.call_count += 1
            if self.call_count <= 2:
                return 95
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            if self.call_count <= 3:
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "summary"}}
                    ],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 8,
                        "cost": 0.01,
                    },
                    "timings": {"prompt_time": 0.5, "completion_time": 0.3},
                }
            return {"choices": [{"message": {"role": "assistant", "content": "final"}}]}

    client = CompressionUsageClient()
    chat = Chat(client, ctx_ratio=0.9)
    for i in range(10):
        chat.append("user", f"msg{i}")
        chat.append("assistant", f"resp{i}")
    result = chat.send()
    assert "compression_usage" in result
    assert result["compression_usage"]["prompt_tokens"] == 40
    assert result["compression_usage"]["completion_tokens"] == 8
    assert result["compression_usage"]["cost"] == 0.01
    assert result["compression_usage"]["prompt_time"] == 0.5
    assert result["compression_usage"]["completion_time"] == 0.3
    assert result["compression_usage"]["cached_tokens"] == 0
    assert result["compression_usage"]["cache_write_tokens"] == 0


def test_send_message_with_compression_accumulates_usage():
    class AccClient:
        def __init__(self):
            self.saw_summary = False

        def get_context_size(self):
            return 100

        def count_tokens(self, messages):
            # Sub-task has summarization system prompt, return low token count
            if any("summarize" in (m.get("content") or "") for m in messages):
                return 10
            # Main conv: trigger compression first time, then low
            if not self.saw_summary:
                return 95
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):
            messages = kwargs.get("messages", [])
            is_summary = any(
                "summarize them" in (m.get("content") or "") for m in messages
            )
            if is_summary:
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "summary"}}
                    ],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 8,
                        "cost": 0.01,
                    },
                    "timings": {"prompt_time": 0.5, "completion_time": 0.3},
                }
            self.saw_summary = True
            return {
                "choices": [{"message": {"role": "assistant", "content": "final"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.05},
                "timings": {"prompt_time": 2.0, "completion_time": 1.0},
            }

    client = AccClient()
    chat = Chat(client, ctx_ratio=0.9)
    for i in range(10):
        chat.append("user", f"msg{i}")
        chat.append("assistant", f"resp{i}")
    result = chat.send_message("one more")
    # Compression usage (40 + 8 + 0.01 + 0.5 + 0.3) + request usage (100 + 20 + 0.05 + 2.0 + 1.0)
    assert result["prompt_tokens"] == 140
    assert result["completion_tokens"] == 28
    assert result["cost"] == pytest.approx(0.06)
    assert result["prompt_time"] == pytest.approx(2.5)
    assert result["completion_time"] == pytest.approx(1.3)
    assert result["message"] == "final"


# ===========================================================================
# send tests
# ===========================================================================


def test_send_basic():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    result = chat.send()
    assert result == {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


def test_send_with_tool_definitions():
    client = MockInferenceClient()
    tool = DummyTool()
    chat = Chat(client, tools=[tool])
    chat.append("user", "hello")
    result = chat.send()
    assert result == {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


def test_send_with_update_callback():
    callback_calls = []

    def callback(msg, _subtask_name=None):
        callback_calls.append(msg)

    class StreamingClient:
        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):
            if kwargs.get("update_callback"):
                kwargs["update_callback"]({"delta": {"content": "hi"}}, None)
            return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    client = StreamingClient()
    chat = Chat(client, update_callback=callback)
    chat.append("user", "hello")
    result = chat.send()
    assert result == {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    assert callback_calls == [{"delta": {"content": "hi"}}]


def test_send_with_tool_calls_in_stream():
    callback_calls = []

    def callback(msg, _subtask_name=None):
        callback_calls.append(msg)

    class StreamingToolClient:
        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):
            if kwargs.get("update_callback"):
                kwargs["update_callback"](
                    {
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc1",
                                    "type": "function",
                                    "function": {"name": "tool", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                )
            return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    client = StreamingToolClient()
    chat = Chat(client, update_callback=callback)
    chat.append("user", "hello")
    result = chat.send()
    assert result == {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "tool", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }
    assert callback_calls == [{"delta": {"content": ""}}]


def test_send_with_tool_calls_multiple_chunks():
    callback_calls = []

    def callback(msg, _subtask_name=None):
        callback_calls.append(msg)

    class MultiChunkToolClient:
        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):
            if kwargs.get("update_callback"):
                cb = kwargs["update_callback"]
                cb(
                    {
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc1",
                                    "type": "function",
                                    "function": {"name": "tool"},
                                }
                            ],
                        }
                    }
                )
                cb(
                    {
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "{"},
                                }
                            ],
                        }
                    }
                )
                cb(
                    {
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "}"},
                                }
                            ],
                        }
                    }
                )
            return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    client = MultiChunkToolClient()
    chat = Chat(client, update_callback=callback)
    chat.append("user", "hello")
    result = chat.send()
    assert result == {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "tool", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }
    assert callback_calls == [
        {"delta": {"content": ""}},
        {"delta": {"content": ""}},
        {"delta": {"content": ""}},
    ]


def test_send_no_choices_in_result_with_tool_calls():
    class NoChoicesClient:
        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):
            if kwargs.get("update_callback"):
                kwargs["update_callback"](
                    {
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc1",
                                    "type": "function",
                                }
                            ],
                        }
                    }
                )
            return {}

    client = NoChoicesClient()
    chat = Chat(client, update_callback=lambda _msg, _subtask_name=None: None)
    chat.append("user", "hello")
    result = chat.send()
    assert result == {
        "choices": [
            {"message": {"tool_calls": [{"index": 0, "id": "tc1", "type": "function"}]}}
        ]
    }


def test_send_choices_but_no_message_with_tool_calls():
    class NoMessageClient:
        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):
            if kwargs.get("update_callback"):
                kwargs["update_callback"](
                    {
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc1",
                                    "type": "function",
                                }
                            ],
                        }
                    },
                    None,
                )
            return {"choices": [{}]}

    client = NoMessageClient()
    chat = Chat(client, update_callback=lambda _msg, _subtask_name=None: None)
    chat.append("user", "hello")
    result = chat.send()
    assert "message" in result["choices"][0]
    assert "tool_calls" in result["choices"][0]["message"]


def test_send_without_callback():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    result = chat.send()
    assert "choices" in result


def test_send_context_compression_trigger():
    class CompressionClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 100

        def count_tokens(self, messages):  # noqa: ARG002
            self.call_count += 1
            if self.call_count <= 2:
                return 95
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            if self.call_count <= 3:
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "summary"}}
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "final"}}]}

    client = CompressionClient()
    chat = Chat(client, ctx_ratio=0.9)
    for i in range(10):
        chat.append("user", f"msg{i}")
        chat.append("assistant", f"resp{i}")
    result = chat.send()
    assert "choices" in result


def test_send_compression_no_token_reduction_breaks():
    """When compression does not reduce token count, break to avoid infinite loop."""

    class NoReduceClient:
        def __init__(self):
            self.iterations = 0

        def get_context_size(self):
            return 100

        def count_tokens(self, messages):  # noqa: ARG002
            return 95

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.iterations += 1
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = NoReduceClient()
    chat = Chat(client, ctx_ratio=0.9)
    # Add enough messages so compress_context is a no-op (len < 4)
    # but token count stays above threshold - loop should break on first iteration
    chat.append("user", "hello")
    result = chat.send()
    assert result["choices"][0]["message"]["content"] == "done"


def test_send_compression_max_attempts_guard():
    """When compression reduces tokens but not enough after 3 attempts, break."""

    class SlowReduceClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 100

        def count_tokens(self, messages):
            self.call_count += 1
            # Subtask (system has "summarizer") should not enter its own loop
            if any("summarizer" in (m.get("content") or "") for m in messages):
                return 10
            # Main conv: slowly decreasing token count
            # call 1: 95 (enter loop), call 2 is subtask,
            # call 3: 93 (<95, continue), call 4 is subtask,
            # call 5: 91 (<93, continue), call 6 is subtask,
            # call 7: 89 (<91, continue, compression_attempts=3, break)
            return 96 - self.call_count

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            return {
                "choices": [{"message": {"role": "assistant", "content": "summary"}}]
            }

    client = SlowReduceClient()
    chat = Chat(client, ctx_ratio=0.9)
    for i in range(10):
        chat.append("user", f"msg{i}")
        chat.append("assistant", f"resp{i}")
    result = chat.send()
    assert result["choices"][0]["message"]["content"] == "summary"


# ===========================================================================
# _deep_merge_dicts tests
# ===========================================================================


def test_deep_merge_dicts_basic():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = Chat._deep_merge_dicts(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_dicts_nested():
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3, "z": 4}}
    result = Chat._deep_merge_dicts(base, override)
    assert result == {"a": {"x": 1, "y": 3, "z": 4}}


def test_deep_merge_dicts_string_concat():
    base = {"a": "hello "}
    override = {"a": "world"}
    result = Chat._deep_merge_dicts(base, override)
    assert result["a"] == "hello world"


def test_deep_merge_dicts_override_non_string():
    base = {"a": "hello"}
    override = {"a": 123}
    result = Chat._deep_merge_dicts(base, override)
    assert result["a"] == 123


def test_deep_merge_dicts_override_non_dict_with_dict():
    base = {"a": 1}
    override = {"a": {"x": 2}}
    result = Chat._deep_merge_dicts(base, override)
    assert result["a"] == {"x": 2}


# ===========================================================================
# _merge_tool_calls tests
# ===========================================================================


def test_merge_tool_calls_basic():
    tool_calls = {
        0: [
            {"index": 0, "id": "tc1", "type": "function"},
            {"index": 0, "function": {"name": "tool"}},
        ]
    }
    result = Chat._merge_tool_calls(tool_calls)
    assert len(result) == 1
    assert result[0]["id"] == "tc1"
    assert result[0]["function"]["name"] == "tool"


def test_merge_tool_calls_string_concat():
    tool_calls = {
        0: [
            {"function": {"name": "my_"}},
            {"function": {"name": "tool"}},
        ]
    }
    result = Chat._merge_tool_calls(tool_calls)
    assert result[0]["function"]["name"] == "my_tool"
    assert len(result) == 1


def test_merge_tool_calls_string_concat_top_level():
    tool_calls = {
        0: [
            {"type": "func"},
            {"type": "tion"},
        ]
    }
    result = Chat._merge_tool_calls(tool_calls)
    assert result[0]["type"] == "function"
    assert len(result) == 1


def test_merge_tool_calls_nested_dict_merge():
    tool_calls = {
        0: [
            {"function": {"args": {"a": 1}}},
            {"function": {"args": {"b": 2}}},
        ]
    }
    result = Chat._merge_tool_calls(tool_calls)
    assert result[0]["function"]["args"] == {"a": 1, "b": 2}
    assert len(result) == 1


def test_merge_tool_calls_multiple_indices():
    tool_calls = {
        0: [{"index": 0, "id": "tc1"}],
        1: [{"index": 1, "id": "tc2"}],
    }
    result = Chat._merge_tool_calls(tool_calls)
    assert len(result) == 2


def test_merge_tool_calls_override_value():
    tool_calls = {
        0: [
            {"index": 0},
            {"index": "1"},
        ]
    }
    result = Chat._merge_tool_calls(tool_calls)
    assert result[0]["index"] == "1"
    assert len(result) == 1


def test_merge_tool_calls_empty():
    result = Chat._merge_tool_calls({})
    assert result == []


# ===========================================================================
# _build_request_data tests
# ===========================================================================


def test_build_request_data_basic():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    data = chat._build_request_data()
    assert data == {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
    }


def test_build_request_data_with_tool_calls():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append(
        "assistant",
        "hi",
        extras={"tool_calls": [{"id": "tc1"}]},
    )
    data = chat._build_request_data()
    assert data == {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "hi", "tool_calls": [{"id": "tc1"}]},
        ]
    }


def test_build_request_data_with_tool_call_id():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("tool", "result", extras={"tool_call_id": "tc1", "name": "tool"})
    data = chat._build_request_data()
    assert data == {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "tc1",
                "name": "tool",
            },
        ]
    }


# ===========================================================================
# save tests
# ===========================================================================


def test_save(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    save_path = tmp_path / "conv.json"
    chat.save(str(save_path))
    assert save_path.exists()
    data = json.loads(save_path.read_text())
    assert data == chat.conv


# ===========================================================================
# load tests
# ===========================================================================


def test_load(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    save_path = tmp_path / "conv.json"
    conv_data = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    save_path.write_text(json.dumps(conv_data))
    chat.load(str(save_path))
    assert chat.conv == conv_data


def test_save_returns_encoded_string():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    chat.append("assistant", "hi")
    result = chat.save()
    assert isinstance(result, str)
    decoded = json.loads(result)
    assert decoded == chat.conv


def test_save_returns_empty_string_when_path_given(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    result = chat.save(str(tmp_path / "conv.json"))
    assert result == ""


def test_load_from_encoded_string():
    client = MockInferenceClient()
    conv_data = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    encoded = json.dumps(conv_data)
    # Create a fresh chat and load from encoded data
    chat2 = Chat(client)
    chat2.load(data=encoded)
    assert chat2.conv == conv_data


def test_load_no_args_raises():
    client = MockInferenceClient()
    chat = Chat(client)
    with pytest.raises(ValueError, match="Either path or data must be provided"):
        chat.load()


def test_load_non_existent_file_raises(tmp_path: Path):
    client = MockInferenceClient()
    chat = Chat(client)
    with pytest.raises(FileNotFoundError):
        chat.load(str(tmp_path / "nonexistent.json"))


def test_send_message_empty_choice_dict():
    class EmptyChoiceClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            return {"choices": [{"message": {}}]}

    client = EmptyChoiceClient()
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result == {
        "reasoning": "",
        "message": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
        "tool_calls": [],
    }


def test_merge_tool_calls_nested_dict_merge_deep():
    tool_calls = {
        0: [
            {"function": {"args": {"a": {"nested": 1}}}},
            {"function": {"args": {"a": {"other": 2}}}},
        ]
    }
    result = Chat._merge_tool_calls(tool_calls)
    assert result == [{"function": {"args": {"a": {"nested": 1, "other": 2}}}}]


def test_merge_tool_calls_dict_key_directly():
    tool_calls = {
        0: [
            {"function": {"name": "tool", "args": {"a": 1}}},
            {"function": {"args": {"b": 2}}},
        ]
    }
    result = Chat._merge_tool_calls(tool_calls)
    assert result == [{"function": {"name": "tool", "args": {"a": 1, "b": 2}}}]


def test_build_request_data_with_reasoning_and_preserve_thinking():
    client = MockInferenceClient()
    client._preserve_thinking = True
    chat = Chat(client)
    chat.append("assistant", "hi", extras={"reasoning": "thinking"})
    chat.append("user", "hello")
    data = chat._build_request_data()
    assert data == {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "assistant",
                "content": "hi",
                "reasoning_content": "thinking",
            },
            {"role": "user", "content": "hello"},
        ]
    }


# ===========================================================================
# replay tests
# ===========================================================================


def test_replay_removes_last_assistant_and_sends():
    class ReplayClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"resp{self.call_count}",
                        }
                    }
                ]
            }

    client = ReplayClient()
    chat = Chat(client)
    # Use send_message which calls send and appends the assistant response
    result1 = chat.send_message("hello")
    assert result1["message"] == "resp1"
    assert chat.conv[-1]["role"] == "assistant"
    assert chat.conv[-1]["content"] == "resp1"
    assert len(chat.conv) == 3

    result = chat.replay()
    assert result["message"] == "resp2"
    assert chat.conv[-1]["content"] == "resp2"
    assert len(chat.conv) == 3


def test_replay_removes_multiple_assistant_messages():
    class ReplayClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"resp{self.call_count}",
                        }
                    }
                ]
            }

    client = ReplayClient()
    chat = Chat(client)
    chat.append("user", "hello")
    chat.append("assistant", "extra assistant")
    chat.append("assistant", "extra assistant2")
    assert len(chat.conv) == 4

    result = chat.replay()
    assert result["message"] == "resp1"
    assert chat.conv[-1]["content"] == "resp1"
    assert len(chat.conv) == 3


def test_replay_empty_conv_does_not_crash():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.conv = []
    result = chat.replay()
    assert result["message"] == "hi"


def test_replay_with_timeout():
    class TimeoutReplayClient:
        def __init__(self):
            self.call_count = 0
            self.last_timeout = None

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):
            self.call_count += 1
            self.last_timeout = kwargs.get("timeout")
            return {"choices": [{"message": {"role": "assistant", "content": "resp"}}]}

    client = TimeoutReplayClient()
    chat = Chat(client)
    chat.append("user", "hello")
    chat.send()
    result = chat.replay(timeout=42.0)
    assert result["message"] == "resp"
    assert client.last_timeout == 42.0


def test_replay_with_no_assistant_messages():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("system", "custom system prompt")
    chat.append("user", "hello")
    chat.append("tool", "result", extras={"tool_call_id": "tc1", "name": "tool"})
    result = chat.replay()
    assert result["message"] == "hi"


# ===========================================================================
# send_message with no message (message=None) tests
# ===========================================================================


def test_send_message_no_message_sends_only_history():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "pre-existing message")
    result = chat.send_message()
    assert result["message"] == "hi"
    assert sum(1 for m in chat.conv if m.get("role") == "user") == 1


def test_send_message_with_none_message():
    client = MockInferenceClient()
    chat = Chat(client)
    chat.append("user", "hello")
    result = chat.send_message(message=None)
    assert result["message"] == "hi"
    assert sum(1 for m in chat.conv if m.get("role") == "user") == 1


# ===========================================================================
# send_message result['tool_calls'] tests
# ===========================================================================


def test_send_message_result_tool_calls_single_tool():
    tool = DummyTool()

    class SingleToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"key": "value"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = SingleToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert result["tool_calls"] == [("dummy_tool", {"key": "value"})]
    assert result["reasoning"] == ""
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["prompt_time"] == 0
    assert result["completion_time"] == 0
    assert result["cost"] == 0


def test_send_message_result_tool_calls_multiple_tools():
    class TwoToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tools",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"a": 1}',
                                        },
                                    },
                                    {
                                        "id": "tc2",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"b": 2}',
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    tool = DummyTool()
    client = TwoToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tools")
    assert result["message"] == "done"
    assert result["tool_calls"] == [
        ("dummy_tool", {"a": 1}),
        ("dummy_tool", {"b": 2}),
    ]
    assert result["reasoning"] == ""
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["prompt_time"] == 0
    assert result["completion_time"] == 0
    assert result["cost"] == 0


def test_send_message_result_tool_calls_tool_not_found(caplog):
    class NotFoundToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "nonexistent_tool",
                                            "arguments": '{"x": "y"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    tool = DummyTool()
    client = NotFoundToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert result["tool_calls"] == []
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Unknown tool call" in caplog.records[0].message


def test_send_message_result_tool_calls_malformed_json(caplog):
    class BadJsonToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": "{bad json}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    tool = DummyTool()
    client = BadJsonToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert result["tool_calls"] == []
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Malformed JSON" in caplog.records[0].message


def test_send_message_result_tool_calls_with_usage_and_timings():
    class UsageToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"foo": "bar"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 20,
                        "cost": 0.05,
                    },
                    "timings": {
                        "prompt_time": 2.0,
                        "completion_time": 1.5,
                    },
                }
            return {
                "choices": [{"message": {"role": "assistant", "content": "final"}}],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "cost": 0.02,
                },
                "timings": {
                    "prompt_time": 1.0,
                    "completion_time": 0.5,
                },
            }

    tool = DummyTool()
    client = UsageToolClient()
    chat = Chat(client, tools=[tool])
    chat.reset_usage()
    result = chat.send_message("run tool")
    assert result["message"] == "final"
    assert result["tool_calls"] == [("dummy_tool", {"foo": "bar"})]
    assert result["prompt_tokens"] == 80
    assert result["completion_tokens"] == 30
    assert result["cost"] == 0.07
    assert result["prompt_time"] == 3.0
    assert result["completion_time"] == 2.0
    assert result["reasoning"] == ""


def test_send_message_result_tool_calls_with_reasoning():
    class ReasoningToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "thinking about tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"z": 100}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "final answer",
                            "reasoning_content": "final thinking",
                        }
                    }
                ]
            }

    tool = DummyTool()
    client = ReasoningToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "final answer"
    assert result["reasoning"] == "final thinking"
    assert result["tool_calls"] == [("dummy_tool", {"z": 100})]


def test_send_message_result_tool_calls_mixed_found_and_not_found():
    class MixedToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"ok": true}',
                                        },
                                    },
                                    {
                                        "id": "tc2",
                                        "function": {
                                            "name": "missing_tool",
                                            "arguments": '{"x": 1}',
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    tool = DummyTool()
    client = MixedToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run mixed")
    assert result["message"] == "done"
    assert result["tool_calls"] == [("dummy_tool", {"ok": True})]


def test_send_message_result_tool_calls_accumulated_across_rounds():
    class DoubleRoundClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"first": 1}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            if self.call_count == 2:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc2",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"second": 2}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {
                "choices": [{"message": {"role": "assistant", "content": "all done"}}]
            }

    tool = DummyTool()
    client = DoubleRoundClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("multi round")
    assert result["message"] == "all done"
    assert result["tool_calls"] == [
        ("dummy_tool", {"first": 1}),
        ("dummy_tool", {"second": 2}),
    ]
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["prompt_time"] == 0
    assert result["completion_time"] == 0
    assert result["cost"] == 0
    assert result["reasoning"] == ""


# ===========================================================================
# tools_confirmations tests
# ===========================================================================


def test_chat_init_tools_confirmations_default():
    client = MockInferenceClient()
    chat = Chat(client)
    assert chat.tools_confirmations == {}


def test_chat_init_tools_confirmations_with_value():
    client = MockInferenceClient()
    confirmations = {
        "edit": [
            ConfirmationSpec(type="exact", key="command", value="write"),
        ]
    }
    chat = Chat(client, tools_confirmations=confirmations)
    assert chat.tools_confirmations == confirmations


def test_check_confirmation_any():
    spec = ConfirmationSpec(type="any", key=None, value=None)
    assert Chat._check_confirmation(spec, {"foo": "bar"})
    assert Chat._check_confirmation(spec, {})


def test_check_confirmation_exact_match():
    spec = ConfirmationSpec(type="exact", key="command", value="write")
    assert Chat._check_confirmation(spec, {"command": "write"})


def test_check_confirmation_exact_no_match():
    spec = ConfirmationSpec(type="exact", key="command", value="write")
    assert not Chat._check_confirmation(spec, {"command": "read"})


def test_check_confirmation_exact_key_missing():
    spec = ConfirmationSpec(type="exact", key="command", value="write")
    assert not Chat._check_confirmation(spec, {"other": "write"})


def test_check_confirmation_regexp_match():
    spec = ConfirmationSpec(type="regexp", key="path", value=re.compile(r"\.py$"))
    assert Chat._check_confirmation(spec, {"path": "/tmp/file.py"})


def test_check_confirmation_regexp_no_match():
    spec = ConfirmationSpec(type="regexp", key="path", value=re.compile(r"\.py$"))
    assert not Chat._check_confirmation(spec, {"path": "/tmp/file.txt"})


def test_check_confirmation_regexp_key_missing():
    spec = ConfirmationSpec(type="regexp", key="path", value=re.compile(r"\.py$"))
    assert not Chat._check_confirmation(spec, {"other": "file.py"})


def test_check_confirmation_exact_not_string_value():
    spec = ConfirmationSpec(type="exact", key="count", value="42")
    assert Chat._check_confirmation(spec, {"count": "42"})
    # The method uses str(arg_value) so int 42 also becomes "42"
    assert Chat._check_confirmation(spec, {"count": 42})
    assert not Chat._check_confirmation(spec, {"count": 43})


def test_check_confirmation_exact_none_key_returns_false():
    spec = ConfirmationSpec(type="exact", key=None, value="write")
    assert not Chat._check_confirmation(spec, {})


def test_check_confirmation_regexp_none_key_returns_false():
    spec = ConfirmationSpec(type="regexp", key=None, value=re.compile(r".*"))
    assert not Chat._check_confirmation(spec, {})


def test_check_confirmation_unknown_type():
    spec = ConfirmationSpec(type="unknown", key="x", value="y")
    assert not Chat._check_confirmation(spec, {"x": "y"})


def test_check_confirmation_arg_value_none():
    spec = ConfirmationSpec(type="exact", key="x", value="y")
    assert not Chat._check_confirmation(spec, {"x": None})


def test_check_confirmation_exact_mismatched_value_type():
    """If spec type is 'exact' but value is not str, returns False."""
    spec = ConfirmationSpec(type="exact", key="x", value=re.compile(r".*"))
    assert not Chat._check_confirmation(spec, {"x": "y"})


def test_check_confirmation_regexp_mismatched_value_type():
    """If spec type is 'regexp' but value is not re.Pattern, returns False."""
    spec = ConfirmationSpec(type="regexp", key="x", value="not_a_pattern")
    assert not Chat._check_confirmation(spec, {"x": "y"})


def test_tool_call_needs_confirmation_no_entry():
    client = MockInferenceClient()
    chat = Chat(client)
    assert not chat._tool_call_needs_confirmation("some_tool", {})


def test_tool_call_needs_confirmation_no_match():
    client = MockInferenceClient()
    chat = Chat(
        client,
        tools_confirmations={
            "some_tool": [
                ConfirmationSpec(type="exact", key="command", value="write"),
            ]
        },
    )
    assert not chat._tool_call_needs_confirmation("some_tool", {"command": "read"})


def test_tool_call_needs_confirmation_match():
    client = MockInferenceClient()
    chat = Chat(
        client,
        tools_confirmations={
            "some_tool": [
                ConfirmationSpec(type="exact", key="command", value="write"),
            ]
        },
    )
    assert chat._tool_call_needs_confirmation("some_tool", {"command": "write"})


def test_tool_call_needs_confirmation_different_tool():
    client = MockInferenceClient()
    chat = Chat(
        client,
        tools_confirmations={
            "tool_a": [
                ConfirmationSpec(type="exact", key="cmd", value="danger"),
            ]
        },
    )
    assert not chat._tool_call_needs_confirmation("tool_b", {"cmd": "danger"})


def test_tool_call_needs_confirmation_multiple_specs_match_any():
    client = MockInferenceClient()
    chat = Chat(
        client,
        tools_confirmations={
            "tool": [
                ConfirmationSpec(type="exact", key="cmd", value="read"),
                ConfirmationSpec(type="exact", key="cmd", value="write"),
            ]
        },
    )
    assert chat._tool_call_needs_confirmation("tool", {"cmd": "write"})
    assert chat._tool_call_needs_confirmation("tool", {"cmd": "read"})
    assert not chat._tool_call_needs_confirmation("tool", {"cmd": "delete"})


def test_tool_call_needs_confirmation_any_type():
    client = MockInferenceClient()
    chat = Chat(
        client,
        tools_confirmations={
            "tool": [ConfirmationSpec(type="any", key=None, value=None)],
        },
    )
    assert chat._tool_call_needs_confirmation("tool", {"foo": "bar"})
    assert chat._tool_call_needs_confirmation("tool", {})


def test_tool_call_needs_confirmation_regexp():
    client = MockInferenceClient()
    chat = Chat(
        client,
        tools_confirmations={
            "tool": [
                ConfirmationSpec(type="regexp", key="path", value=re.compile(r"\.py$")),
            ]
        },
    )
    assert chat._tool_call_needs_confirmation("tool", {"path": "file.py"})
    assert not chat._tool_call_needs_confirmation("tool", {"path": "file.txt"})


def test_tool_call_needs_confirmation_empty_specs_list():
    client = MockInferenceClient()
    chat = Chat(
        client,
        tools_confirmations={"tool": []},
    )
    assert not chat._tool_call_needs_confirmation("tool", {"anything": "value"})


# ===========================================================================
# send_message with tools_confirmations tests
# ===========================================================================


def test_send_message_tool_confirmation_rejected():
    tool = DummyTool()

    class ConfirmClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"path": "/etc/passwd"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = ConfirmClient()
    confirm_callback = MagicMock(return_value=False)
    chat = Chat(
        client,
        tools=[tool],
        confirm_callback=confirm_callback,
        tools_confirmations={
            "dummy_tool": [
                ConfirmationSpec(type="regexp", key="path", value=re.compile(r"/etc/")),
            ]
        },
    )
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    confirm_callback.assert_called_once()
    args, _ = confirm_callback.call_args
    assert args[0] == {"tool": "dummy_tool", "arguments": {"path": "/etc/passwd"}}
    assert callable(args[1])
    # Not appended to usage['tool_calls']
    assert result["tool_calls"] == []
    # Check that the rejected tool result was appended
    assert any(
        m.get("content") == '{"error": "Tool call rejected by user confirmation"}'
        for m in chat.conv
    )


def test_send_message_tool_confirmation_accepted():
    tool = DummyTool()

    class ConfirmClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"path": "/etc/passwd"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = ConfirmClient()
    confirm_callback = MagicMock(return_value=True)
    chat = Chat(
        client,
        tools=[tool],
        confirm_callback=confirm_callback,
        tools_confirmations={
            "dummy_tool": [
                ConfirmationSpec(type="regexp", key="path", value=re.compile(r"/etc/")),
            ]
        },
    )
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    confirm_callback.assert_called_once()
    args, _ = confirm_callback.call_args
    assert args[0] == {"tool": "dummy_tool", "arguments": {"path": "/etc/passwd"}}
    assert callable(args[1])
    assert result["tool_calls"] == [("dummy_tool", {"path": "/etc/passwd"})]


def test_send_message_tool_confirmations_no_match_does_not_ask():
    tool = DummyTool()

    class NoMatchClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"path": "/safe/file"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = NoMatchClient()
    confirm_callback = MagicMock()
    chat = Chat(
        client,
        tools=[tool],
        confirm_callback=confirm_callback,
        tools_confirmations={
            "dummy_tool": [
                ConfirmationSpec(type="regexp", key="path", value=re.compile(r"/etc/")),
            ]
        },
    )
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    confirm_callback.assert_not_called()
    assert result["tool_calls"] == [("dummy_tool", {"path": "/safe/file"})]


def test_send_message_tool_confirmations_different_tool_no_interference():
    tool = DummyTool()
    error_tool = DummyToolError()

    class MultiClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tools",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"path": "/etc/passwd"}',
                                        },
                                    },
                                    {
                                        "id": "tc2",
                                        "function": {
                                            "name": "error_tool",
                                            "arguments": '{"path": "/etc/shadow"}',
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = MultiClient()
    confirm_callback = MagicMock(return_value=False)
    chat = Chat(
        client,
        tools=[tool, error_tool],
        confirm_callback=confirm_callback,
        tools_confirmations={
            "dummy_tool": [
                ConfirmationSpec(type="regexp", key="path", value=re.compile(r"/etc/")),
            ]
        },
    )
    result = chat.send_message("run tools")
    assert result["message"] == "done"
    # dummy_tool should have been rejected
    assert result["tool_calls"] == [("error_tool", {"path": "/etc/shadow"})]


def test_tool_confirmation_with_any_type():
    """When type is 'any', any call to that tool should require confirmation."""
    tool = DummyTool()

    class AnyTypeClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"foo": "bar"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = AnyTypeClient()
    confirm_callback = MagicMock(return_value=False)
    chat = Chat(
        client,
        tools=[tool],
        confirm_callback=confirm_callback,
        tools_confirmations={
            "dummy_tool": [ConfirmationSpec(type="any", key=None, value=None)],
        },
    )
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    confirm_callback.assert_called_once()
    assert result["tool_calls"] == []


def test_tool_confirmation_with_exact_type():
    """When type is 'exact', only matching key/value should require confirmation."""
    tool = DummyTool()

    class ExactTypeClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tool",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"command": "delete"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = ExactTypeClient()
    confirm_callback = MagicMock(return_value=True)
    chat = Chat(
        client,
        tools=[tool],
        confirm_callback=confirm_callback,
        tools_confirmations={
            "dummy_tool": [
                ConfirmationSpec(type="exact", key="command", value="delete"),
            ]
        },
    )
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    confirm_callback.assert_called_once()
    assert result["tool_calls"] == [("dummy_tool", {"command": "delete"})]


def test_tool_confirmation_rejected_then_continue_with_other_tools():
    """Rejected tool should not prevent other tools in the same round from executing."""
    tool = DummyTool()

    class RejectOneClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "using tools",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"cmd": "danger"}',
                                        },
                                    },
                                    {
                                        "id": "tc2",
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": '{"cmd": "safe"}',
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = RejectOneClient()
    confirm_callback = MagicMock()
    # Reject "danger", accept anything else
    confirm_callback.side_effect = lambda _args, _fmt, _subtask=None: (
        _args.get("arguments", {}).get("cmd") != "danger"
    )
    chat = Chat(
        client,
        tools=[tool],
        confirm_callback=confirm_callback,
        tools_confirmations={
            "dummy_tool": [ConfirmationSpec(type="any", key=None, value=None)],
        },
    )
    result = chat.send_message("run tools")
    assert result["message"] == "done"
    assert result["tool_calls"] == [("dummy_tool", {"cmd": "safe"})]


# ===========================================================================
# send_message with tool.run() exception tests
# ===========================================================================


def test_send_message_tool_run_exception(caplog):
    class FailingTool:
        @staticmethod
        def prompt() -> str:
            return json.dumps(
                {
                    "name": "failing_tool",
                    "description": "A tool that fails",
                    "parameters": {"type": "object", "properties": {}},
                }
            )

        @staticmethod
        def fname() -> str:
            return "failing_tool"

        def run(self, args: dict) -> str:  # noqa: ARG002
            msg = "something went wrong"
            raise RuntimeError(msg)

        def format_confirmation(
            self, _args: dict, _markdown: bool = False
        ) -> str | None:
            return None

    class FailingToolClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "name": "failing_tool",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    tool = FailingTool()
    client = FailingToolClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert result["tool_calls"] == [("failing_tool", {})]
    assert chat.conv[-2] == {
        "role": "tool",
        "content": '{"error": "Tool execution failed: something went wrong"}',
        "tool_call_id": "tc1",
        "name": "failing_tool",
    }
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Tool failing_tool (tc1) raised:" in caplog.records[0].message


def test_send_message_tool_none_message():
    """Test that a null message in choices[0] triggers early return."""
    client = MockInferenceClient(chat_response={"choices": [{"message": None}]})
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result == {
        "reasoning": "",
        "message": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "prompt_time": 0,
        "completion_time": 0,
        "cost": 0,
        "tool_calls": [],
    }


def test_send_message_tool_name_missing():
    """Test that missing tool name defaults to 'unknown' does not crash."""


def test_send_message_choice_without_role_key():
    """Message dict that is truthy but lacks a 'role' key should not crash."""
    client = MockInferenceClient(
        chat_response={
            "choices": [{"message": {"content": "hi"}}]  # no "role" key
        }
    )
    chat = Chat(client)
    result = chat.send_message("hello")
    assert result["message"] == "hi"
    assert len(chat.conv) == 3
    assert chat.conv[-1]["role"] == "assistant"  # default role used

    tool = DummyTool()

    class NoNameClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "function": {
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    client = NoNameClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert chat.conv[-2] == {
        "role": "tool",
        "content": "ERROR: command does not exist",
        "tool_call_id": "tc1",
        "name": "unknown",
    }


def test_send_message_tool_id_missing():
    """Test that tool call with no id defaults to 'unknown'."""

    class NoIdClient:
        def __init__(self):
            self.call_count = 0

        def get_context_size(self):
            return 4096

        def count_tokens(self, messages):  # noqa: ARG002
            return 10

        def input_modes(self):
            return ["text"]

        @property
        def preserve_thinking(self):
            return False

        def chat(self, **kwargs):  # noqa: ARG002
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "dummy_tool",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    tool = DummyTool()
    client = NoIdClient()
    chat = Chat(client, tools=[tool])
    result = chat.send_message("run tool")
    assert result["message"] == "done"
    assert result["tool_calls"] == [("dummy_tool", {})]
