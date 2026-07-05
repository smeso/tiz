"""Chat session management for the tiz agentic AI chatbot.

The Chat class manages conversation history, tool execution, context
compression, and interaction with the LLM service.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tiz.audio_inference_clients import AudioInferenceClient
from tiz.conversion_sandbox import ConversionSandbox
from tiz.inference_clients import InferenceClient
from tiz.log import get_logger
from tiz.manifest_parser import ConfirmationSpec
from tiz.tools.base import Tool

logger = get_logger(__name__)


class Chat:
    """Manages a chat session with the LLM, including tool execution.

    The Chat class maintains conversation history, handles tool calls
    from the LLM, manages context window compression, and provides
    both streaming and non-streaming message sending.
    """

    def __init__(
        self,
        client: InferenceClient,
        sys_prompt: str = "You are a helpful assistant.",
        update_callback: Callable[[dict[str, Any], str | None], None] | None = None,
        tools: Sequence[Tool] | None = None,
        conversion_sandbox: ConversionSandbox | None = None,
        audio_inference_client: AudioInferenceClient | None = None,
        ctx_ratio: float = 0.9,
        confirm_callback: Callable[
            [dict[str, Any], Callable[[dict[str, Any], bool], str | None], str | None],
            bool,
        ]
        | None = None,
        tools_confirmations: dict[str, list[ConfirmationSpec]] | None = None,
        subtask_name: str | None = None,
    ) -> None:
        """Initialize a Chat session.

        Args:
            client: An InferenceClient instance for LLM communication.
            sys_prompt: System prompt for the conversation.
            update_callback: Optional callback for streaming updates.
            tools: Optional list of tools available to the LLM.
            conversion_sandbox: Optional sandbox for file conversion.
            audio_inference_client: Optional audio inference client for
                speech-to-text transcription.
            ctx_ratio: Ratio of context window to use before compression.
            confirm_callback: Optional callback to confirm tool execution.
                Takes the call details dict, the tool's format_confirmation
                function, and an optional subtask name.
            tools_confirmations: Optional dict mapping tool names to lists of
                ConfirmationSpec that require user confirmation before execution.
            subtask_name: Optional name of the subtask requesting confirmation.
        """
        self.client = client
        self.sys_prompt = sys_prompt
        self.update_callback = update_callback
        self.confirm_callback = (
            confirm_callback
            if confirm_callback is not None
            else lambda _, __, ___: False
        )
        self.tools_confirmations = tools_confirmations or {}
        self.subtask_name = subtask_name
        self.conv: list[dict[str, Any]] = []
        self.reset_usage()
        self.tools: dict[str, Tool] | None = None
        self.tool_definitions: list[dict[str, Any]] = []
        self.conversion_sandbox = conversion_sandbox
        self.audio_inference_client = audio_inference_client

        if tools:
            self.tools = {tool.fname(): tool for tool in tools}
            self.tool_definitions = [
                {"type": "function", "function": json.loads(tool.prompt())}
                for tool in tools
            ]

        self.append("system", self.sys_prompt)
        self.ctx_size = self.client.get_context_size()
        if not 0 < ctx_ratio <= 1:
            raise ValueError(f"ctx_ratio must be in (0, 1], got {ctx_ratio}")
        self.ctx_ratio = ctx_ratio
        self._current_tool_calls: dict[int, list[dict[str, Any]]] = {}

    def reset_usage(self) -> None:
        self.usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "prompt_time": 0,
            "completion_time": 0,
            "cost": 0,
            "tool_calls": [],
        }

    def append(
        self,
        role: str,
        message: str | None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Append a message to the conversation history.

        Args:
            role: The role of the message sender (e.g., user, assistant, tool).
            message: The message content.
            extras: Optional extra fields to include in the message.
        """
        msg: dict[str, Any] = {"role": role, "content": message}
        if extras:
            msg.update(extras)
        self.conv.append(msg)

    @staticmethod
    def _read_file(path: str) -> bytes:
        """Read binary data from a file.

        Args:
            path: The file path to read from.

        Returns:
            The binary contents of the file.
        """
        with Path(path).open("rb") as fd:
            return fd.read()

    def append_media(
        self,
        media_type: str,
        path: str | None = None,
        mimetype: str | None = None,
        data: bytes | None = None,
    ) -> None:
        """Append media (image, video, etc.) to the conversation.

        Args:
            media_type: The type of media (e.g., "image", "video").
            path: Path to the media file. Required if data is not provided.
            mimetype: The MIME type of the media.
            data: Binary data of the media. Required if path is not provided.
        """
        if data is None and path is not None:
            data = self._read_file(path)
        if data is None:
            raise ValueError("Either path or data must be provided")
        if mimetype is None and path is not None:
            guessed = mimetypes.guess_type(path)[0]
            if guessed is not None:
                mimetype = guessed
        if mimetype is None:
            raise ValueError("mimetype must be provided or guessable from path")

        b64 = base64.b64encode(data).decode()
        msg = {
            "type": f"{media_type}_url",
            f"{media_type}_url": {"url": f"data:{mimetype};base64,{b64}"},
        }
        self.conv.append({"role": "user", "content": [msg]})

    def append_image(
        self,
        path: str | None = None,
        mimetype: str | None = None,
        data: bytes | None = None,
    ) -> None:
        """Append an image to the conversation.

        Args:
            path: Path to the image file. Required if data is not provided.
            mimetype: The MIME type of the image.
            data: Binary data of the image. Required if path is not provided.
        """
        self.append_media("image", path, mimetype, data)

    def append_video(
        self,
        path: str | None = None,
        mimetype: str | None = None,
        data: bytes | None = None,
    ) -> None:
        """Append a video to the conversation.

        Args:
            path: Path to the video file. Required if data is not provided.
            mimetype: The MIME type of the video.
            data: Binary data of the video. Required if path is not provided.
        """
        self.append_media("video", path, mimetype, data)

    def append_audio(
        self,
        path: str | None = None,
        mimetype: str | None = None,
        data: bytes | None = None,
    ) -> None:
        """Append audio to the conversation.

        If a conversion_sandbox is available and supports the audio mimetype,
        the audio is first converted (e.g. to wav) before attaching.

        If an audio_inference_client is available, the audio is transcribed
        and the text is appended to the conversation instead of raw audio.

        Args:
            path: Path to the audio file. Required if data is not provided.
            mimetype: The MIME type of the audio.
            data: Binary data of the audio. Required if path is not provided.
        """
        if data is None and path is not None:
            data = self._read_file(path)
        if data is None:
            raise ValueError("Either path or data must be provided")
        if mimetype is None and path is not None:
            guessed = mimetypes.guess_type(path)[0]
            if guessed is not None:
                mimetype = guessed

        if (
            self.conversion_sandbox is not None
            and mimetype is not None
            and mimetype in self.conversion_sandbox.supported_mimetypes()
        ):
            converted = self.conversion_sandbox.convert(mimetype, data)
            if converted is not None:
                data = converted[0][0]
                mimetype = converted[0][1]
            else:
                raise ValueError(f"Audio conversion failed for {mimetype}")

        if self.audio_inference_client is not None:
            text = self.audio_inference_client.transcribe(data).strip()
            self.append("user", text)
            if self.update_callback is not None:
                self.update_callback(
                    {
                        "tiz-internal": {
                            "transcribe": text,
                        }
                    },
                    None,
                )
            return

        audio_format = ""
        if mimetype is not None:
            if "wav" in mimetype:
                audio_format = "wav"
            elif "mp3" in mimetype:
                audio_format = "mp3"
            elif "ogg" in mimetype:
                audio_format = "ogg"
            elif "flac" in mimetype:
                audio_format = "flac"
            elif "mpeg" in mimetype:
                audio_format = "mp3"

        if not audio_format:
            raise ValueError(
                f"Unsupported or missing audio format for mimetype: {mimetype}"
            )

        b64 = base64.b64encode(data).decode()
        msg = {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": audio_format},
        }
        self.conv.append({"role": "user", "content": [msg]})

    def append_pdf(
        self,
        path: str | None = None,
        data: bytes | None = None,
    ) -> None:
        """Append a PDF to the conversation.

        Args:
            path: Path to the PDF file. Required if data is not provided.
            data: Binary data of the PDF. Required if path is not provided.
        """
        filename = Path(path).name if path is not None else ""
        if data is None and path is not None:
            data = self._read_file(path)
        if data is None:
            raise ValueError("Either path or data must be provided")

        b64 = base64.b64encode(data).decode()
        msg = {
            "type": "file",
            "file": {"filename": filename, "file_data": b64},
        }
        self.conv.append({"role": "user", "content": [msg]})

    def append_file(
        self,
        path: str,
        data: bytes | None = None,
    ) -> None:
        """Append a file to the conversation, converting if necessary.

        Images are attached directly. Other supported file types are
        converted via the conversion sandbox.

        Args:
            path: Path to the file (used as the file reference even when
                contents is provided).
            data: Optional binary content of the file.
        """

        mimetype = mimetypes.guess_type(path)[0]
        if mimetype is None:
            raise ValueError(f"Could not determine mimetype for {path}")

        input_modes = self.client.input_modes()

        if mimetype.startswith("audio/") and (
            "audio" in input_modes or self.audio_inference_client is not None
        ):
            self.append_audio(path, mimetype, data)
            return

        if mimetype.startswith("video/") and "video" in input_modes:
            self.append_video(path, mimetype, data)
            return

        if mimetype == "application/pdf" and "file" in input_modes:
            self.append_pdf(path, data)
            return

        if mimetype.startswith("image/") and "image" in input_modes:
            self.append_image(path, mimetype, data)
            return

        if mimetype.startswith("text/"):
            if data is None:
                text = Path(path).read_text(encoding="utf-8")
            else:
                text = data.decode("utf-8")
            self.append("user", text)
            return

        if (
            self.conversion_sandbox is not None
            and (data is not None or path is not None)
            and mimetype in self.conversion_sandbox.supported_mimetypes()
            and "image" in input_modes
            and not mimetype.startswith("audio/")
        ):
            src: str | bytes = data if data is not None else path
            converted = self.conversion_sandbox.convert(mimetype, src)
            if converted is not None:
                filename = Path(path).name
                if mimetype.startswith("video/"):
                    self.append(
                        "user",
                        f"These are frames from the video file {filename}.",
                    )
                else:
                    self.append(
                        "user",
                        f"These are pages from the document {filename}.",
                    )
                for conv_data, conv_mimetype in converted:
                    self.append_image(data=conv_data, mimetype=conv_mimetype)
                return

        raise ValueError(f"Unsupported file type: {mimetype} for {path}")

    def replay(self, timeout: float | None = None) -> dict[str, Any]:
        """Remove the last assistant message(s) from history and re-send.

        Strips the most recent assistant turn and issues a new request
        using the remaining conversation history.

        Args:
            timeout: Optional request timeout in seconds.

        Returns:
            A dict with keys: reasoning, message, prompt_tokens,
            completion_tokens, cached_tokens, cache_write_tokens,
            prompt_time, completion_time, cost, tool_calls.
        """
        while self.conv and self.conv[-1]["role"] == "assistant":
            self.conv.pop()
        return self.send_message(timeout=timeout)

    def send_message(
        self,
        message: str | None = None,
        files: list[str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a message to the LLM and handle the response.

        Processes any tool calls in the response iteratively.

        Args:
            message: The user message to send.
            files: Optional list of file paths to attach.
            timeout: Optional request timeout in seconds.

        Returns:
            A dict with keys: reasoning, message, prompt_tokens,
            completion_tokens, cached_tokens, cache_write_tokens,
            prompt_time, completion_time, cost, tool_calls.
        """
        logger.debug(
            "Sending message with %d tool definitions", len(self.tool_definitions)
        )
        if files:
            for f in files:
                self.append_file(f)

        if message is not None:
            self.append("user", message)

        while True:
            res = self.send(timeout=timeout)

            choices = res.get("choices", [])
            if (
                not choices
                or "message" not in choices[0]
                or choices[0]["message"] is None
            ):
                return {
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
            choice = choices[0]["message"]
            tool_calls = choice.get("tool_calls")

            assistant_extras: dict[str, Any] = {
                "reasoning": choice.get("reasoning_content", ""),
            }
            if tool_calls:
                assistant_extras["tool_calls"] = tool_calls

            if choice:
                self.append(
                    choice.get("role", "assistant"),
                    choice.get("content"),
                    assistant_extras,
                )

            usage = res.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cached_tokens = usage.get("cached_tokens", 0)
                cache_write_tokens = usage.get("cache_write_tokens", 0)
                cost = usage.get("cost", 0)
                self.usage["prompt_tokens"] += prompt_tokens
                self.usage["completion_tokens"] += completion_tokens
                self.usage["cached_tokens"] += cached_tokens
                self.usage["cache_write_tokens"] += cache_write_tokens
                self.usage["cost"] += cost
            timings = res.get("timings", {})
            if timings:
                prompt_time = timings.get("prompt_time", 0)
                completion_time = timings.get("completion_time", 0)
                self.usage["prompt_time"] += prompt_time
                self.usage["completion_time"] += completion_time

            compression_usage = res.get("compression_usage", {})
            if compression_usage:
                self.usage["prompt_tokens"] += compression_usage.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += compression_usage.get(
                    "completion_tokens", 0
                )
                self.usage["cached_tokens"] += compression_usage.get("cached_tokens", 0)
                self.usage["cache_write_tokens"] += compression_usage.get(
                    "cache_write_tokens", 0
                )
                self.usage["prompt_time"] += compression_usage.get("prompt_time", 0)
                self.usage["completion_time"] += compression_usage.get(
                    "completion_time", 0
                )
                self.usage["cost"] += compression_usage.get("cost", 0)

            if tool_calls and self.tools:
                for tool_call in tool_calls:
                    tc_id = tool_call.get("id") or "unknown"
                    function_info = tool_call.get("function", {})
                    name = function_info.get("name") or "unknown"
                    arguments_str = function_info.get("arguments", "{}")

                    try:
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        result = json.dumps(
                            {"error": "malformed JSON in tool arguments"}
                        )
                        logger.warning(
                            "Malformed JSON in tool call %s: %s",
                            tc_id,
                            arguments_str,
                        )
                    else:
                        if name and name in self.tools:
                            tool_obj = self.tools[name]
                            if self._tool_call_needs_confirmation(
                                name, arguments
                            ) and not self.confirm_callback(
                                {
                                    "tool": name,
                                    "arguments": arguments,
                                },
                                tool_obj.format_confirmation,
                                self.subtask_name,
                            ):
                                result = json.dumps(
                                    {"error": "Tool call rejected by user confirmation"}
                                )
                                logger.warning(
                                    "Tool call %s (%s) rejected by user",
                                    tc_id,
                                    name,
                                )
                                self.append(
                                    "tool",
                                    result,
                                    {
                                        "tool_call_id": tc_id,
                                        "name": name,
                                    },
                                )
                                continue
                            self.usage["tool_calls"].append((name, arguments))
                            try:
                                result = self.tools[name].run(arguments)
                            except Exception as exc:
                                result = json.dumps(
                                    {"error": f"Tool execution failed: {exc}"}
                                )
                                logger.warning(
                                    "Tool %s (%s) raised: %s", name, tc_id, exc
                                )
                        else:
                            result = "ERROR: command does not exist"
                            logger.warning("Unknown tool call: %s (%s)", tc_id, name)

                    self.append(
                        "tool",
                        result,
                        {"tool_call_id": tc_id, "name": name},
                    )
                continue

            return {
                "reasoning": choice.get("reasoning_content", ""),
                "message": choice.get("content", ""),
                "prompt_tokens": self.usage["prompt_tokens"],
                "completion_tokens": self.usage["completion_tokens"],
                "cached_tokens": self.usage["cached_tokens"],
                "cache_write_tokens": self.usage["cache_write_tokens"],
                "prompt_time": self.usage["prompt_time"],
                "completion_time": self.usage["completion_time"],
                "cost": self.usage["cost"],
                "tool_calls": self.usage["tool_calls"],
            }

    @staticmethod
    def _check_confirmation(spec: ConfirmationSpec, arguments: dict[str, Any]) -> bool:
        """Check if a ConfirmationSpec matches the given arguments.

        If type is 'any', always matches.
        If type is 'exact', matches if the key exists and the value equals the spec's value.
        If type is 'regexp', matches if the key exists and the value matches the regex pattern.

        Returns:
            True if the specification matches the arguments.
        """
        if spec.type == "any":
            return True
        if spec.key is None:
            return False
        arg_value = arguments.get(spec.key)
        if arg_value is None:
            return False
        arg_str = str(arg_value)
        if spec.type == "exact":
            return isinstance(spec.value, str) and arg_str == spec.value
        if spec.type == "regexp":
            if not isinstance(spec.value, re.Pattern):
                return False
            return bool(spec.value.search(arg_str))
        return False

    def _tool_call_needs_confirmation(
        self, name: str, arguments: dict[str, Any]
    ) -> bool:
        """Check if a tool call needs user confirmation.

        Args:
            name: The tool name.
            arguments: The tool arguments.

        Returns:
            True if any confirmation spec matches the tool call.
        """
        if name not in self.tools_confirmations:
            return False
        for spec in self.tools_confirmations[name]:
            if self._check_confirmation(spec, arguments):
                return True
        return False

    def compress_context(self) -> dict[str, Any]:
        """Compress the conversation context to fit within the token limit.

        Summarizes the first half of the conversation (excluding the
        first 2 messages) to reduce token count.

        Returns:
            Usage information from the compression request.
        """
        if len(self.conv) < 4:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "prompt_time": 0,
                "completion_time": 0,
                "cost": 0,
            }

        new_conv_prefix = self.conv[:2]
        i = (len(self.conv) - 2) // 2
        to_compress = json.dumps(self.conv[2 : 2 + i])

        sub_task = Chat(
            client=self.client,
            sys_prompt=(
                "You are a summarizer subtask. You receive LLM "
                "conversations in JSON and you have to summarize them "
                "to optimize context usage.\n"
                "You should paraphrase text and cut command output. "
                "Do not summarize command output or code, you can only cut "
                "it or remove parts, but do not replace them.\n"
                "You should instead replace normal text with more "
                "concise sentences without repetitions, dropping "
                "anything that is not very important and rewriting "
                "important things to use less tokens."
            ),
        )
        summary = sub_task.send_message(to_compress)

        self.conv = (
            new_conv_prefix
            + [
                {
                    "role": "system",
                    "content": (
                        f"[Context compressed: {i} previous messages "
                        f"summarized]\n{summary['message']}"
                    ),
                }
            ]
            + self.conv[2 + i :]
        )

        return {
            "prompt_tokens": summary.get("prompt_tokens", 0),
            "completion_tokens": summary.get("completion_tokens", 0),
            "cached_tokens": summary.get("cached_tokens", 0),
            "cache_write_tokens": summary.get("cache_write_tokens", 0),
            "prompt_time": summary.get("prompt_time", 0),
            "completion_time": summary.get("completion_time", 0),
            "cost": summary.get("cost", 0),
        }

    def send(self, timeout: float | None = None) -> dict[str, Any]:
        """Send the current conversation to the LLM.

        Handles context compression if the token count approaches
        the context size limit.

        Args:
            timeout: Optional request timeout in seconds.

        Returns:
            The LLM response dict.
        """
        data = self._build_request_data()
        tcount = self.client.count_tokens(data["messages"])

        compression_usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "prompt_time": 0,
            "completion_time": 0,
            "cost": 0,
        }
        max_compressions = 3
        compression_attempts = 0
        while int(round(self.ctx_size * self.ctx_ratio)) <= tcount:
            cu = self.compress_context()
            for key in compression_usage:
                compression_usage[key] += cu.get(key, 0)
            data = self._build_request_data()
            new_tcount = self.client.count_tokens(data["messages"])
            if new_tcount >= tcount:
                break
            tcount = new_tcount
            compression_attempts += 1
            if compression_attempts >= max_compressions:
                break

        sampling_params: dict[str, Any] = {}
        if self.tool_definitions:
            sampling_params["tools"] = self.tool_definitions

        self._current_tool_calls = {}

        cb = self.update_callback
        if cb is not None:

            def _wrapped_callback(
                msg: dict[str, Any], _subtask_name: str | None = None
            ) -> None:
                delta = msg.get("delta", {})
                tool_calls = delta.pop("tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        index = tc.get("index", 0)
                        if index not in self._current_tool_calls:
                            self._current_tool_calls[index] = []
                        self._current_tool_calls[index].append(tc)
                cb(msg, self.subtask_name)

            callback = _wrapped_callback
        else:
            callback = None

        result = self.client.chat(
            messages=data["messages"],
            sampling_params=sampling_params,
            update_callback=callback,
            stream=self.update_callback is not None,
            timeout=timeout,
        )
        if self._current_tool_calls:
            merged_tool_calls = self._merge_tool_calls(self._current_tool_calls)
            if "choices" in result and result["choices"]:
                if "message" not in result["choices"][0]:
                    result["choices"][0]["message"] = {}
            else:
                result["choices"] = [{"message": {}}]
            result["choices"][0]["message"]["tool_calls"] = merged_tool_calls
            self._current_tool_calls = {}
        if compression_usage and any(v for v in compression_usage.values()):
            result["compression_usage"] = compression_usage
        return result

    @staticmethod
    def _deep_merge_dicts(
        base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Chat._deep_merge_dicts(result[key], value)
            elif (
                key in result
                and isinstance(value, str)
                and isinstance(result[key], str)
            ):
                result[key] += value
            else:
                result[key] = value
        return result

    @staticmethod
    def _merge_tool_calls(
        tool_calls: dict[int, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for chunks in tool_calls.values():
            merged_call: dict[str, Any] = {}
            for chunk in chunks:
                for key, value in chunk.items():
                    if key in merged_call:
                        if isinstance(value, str) and isinstance(merged_call[key], str):
                            merged_call[key] += value
                        elif isinstance(value, dict) and isinstance(
                            merged_call[key], dict
                        ):
                            merged_call[key] = Chat._deep_merge_dicts(
                                merged_call[key], value
                            )
                        else:
                            merged_call[key] = value
                    else:
                        merged_call[key] = value
            merged.append(merged_call)
        return merged

    def _build_request_data(self) -> dict[str, Any]:
        """Build the messages list for the LLM request.

        Returns:
            A dict with a "messages" key containing the conversation.
        """
        messages: list[dict[str, Any]] = []
        for entry in self.conv:
            msg: dict[str, Any] = {
                "role": entry["role"],
                "content": entry.get("content"),
            }
            if self.client.preserve_thinking and "reasoning" in entry:
                msg["reasoning_content"] = entry["reasoning"]
            if "tool_calls" in entry:
                msg["tool_calls"] = entry["tool_calls"]
            if "tool_call_id" in entry:
                msg["tool_call_id"] = entry["tool_call_id"]
            if "name" in entry:
                msg["name"] = entry["name"]
            messages.append(msg)
        return {"messages": messages}

    def save(self, path: str | None = None) -> str:
        """Save the conversation history to a JSON file.

        If path is None, returns the conversation as a JSON string.

        Args:
            path: The file path to save to, or None to return encoded string.

        Returns:
            JSON string if path is None, otherwise an empty string.
        """
        data = json.dumps(self.conv)
        if path is None:
            return data
        with Path(path).open("w", encoding="utf-8") as fd:
            fd.write(data)
        return ""

    def load(self, path: str | None = None, data: str | None = None) -> None:
        """Load conversation history from a JSON file or JSON string.

        Either path or data must be provided.

        Args:
            path: The file path to load from.
            data: JSON string of the conversation.
        """
        if data is not None:
            self.conv = json.loads(data)
        elif path is not None:
            with Path(path).open(encoding="utf-8") as fd:
                self.conv = json.load(fd)
        else:
            raise ValueError("Either path or data must be provided")
