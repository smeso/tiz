"""Inference clients for communicating with LLM backends.

This module provides a base ``InferenceClient`` class, an
``OpenAICompatibleClient`` base with shared functionality, and concrete
implementations for:

* ``LlamaCpp`` - a self-hosted llama.cpp server
* ``DwarfStar4`` - a lightweight inference backend
* ``OpenRouter`` - the OpenRouter.ai API gateway
* ``AnthropicClient`` - the Anthropic API (Claude models)
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import requests

from tiz import __version__
from tiz.log import get_logger

logger = get_logger(__name__)

USER_AGENT = (
    f"Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko);"
    f" compatible; tiz/{__version__}; +https://github.com/smeso/tiz"
)

OPENROUTER_REFERER = "https://github.com/smeso/tiz"
OPENROUTER_APP_TITLE = "tiz"
OPENROUTER_CATEGORIES = (
    "cli-agent,general-chat,personal-agent,code-assistant,creative-writing"
)


def validate_host(host: str) -> str:
    """Validate that *host* has a scheme and netloc."""
    parsed = urlparse(host)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"Invalid host URL '{host}': must include scheme (e.g. http://) and hostname"
        )
    return host


SamplingParams = dict[str, Any]
"""Dictionary of sampling parameters.

Common keys (backend-specific keys may vary):

- ``temperature`` (float): Controls randomness.  Lower values are more deterministic.
- ``top_k`` (int): Limits sampling to the top-k most likely tokens.
- ``top_p`` (float): Nucleus sampling threshold.
- ``min_p`` (float): Minimum probability threshold relative to the top token.
"""


class StreamStatus(Enum):
    """Status codes for stream processing outcomes."""

    RETURN = "return"
    ERROR = "error"
    RETRY = "retry"


@dataclass
class _StreamOutcome:
    """Typed result from _process_stream_response."""

    status: StreamStatus
    content: str
    reasoning: str
    retries: int
    error: RuntimeError | None = None
    result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class InferenceClient(ABC):
    """Abstract base class for LLM inference clients.

    Subclasses must implement the abstract methods to provide backend-specific
    behaviour for chat completions, token counting, model enumeration and
    tool-support detection.
    """

    def __init__(
        self,
        default_model: str = "",
        sampling_params: SamplingParams | None = None,
        preserve_thinking: bool = False,
    ) -> None:
        self._default_model = default_model
        self._sampling_params: SamplingParams = sampling_params or {}
        self._preserve_thinking = preserve_thinking

    def set_default_model(self, model: str) -> None:
        """Set or change the default model."""
        self._default_model = model

    @property
    def sampling_params(self) -> SamplingParams:
        """Return the default sampling parameters."""
        return self._sampling_params

    @sampling_params.setter
    def sampling_params(self, params: SamplingParams) -> None:
        """Set the default sampling parameters."""
        self._sampling_params = params

    @property
    def preserve_thinking(self) -> bool:
        """Return whether thinking/reasoning content should be preserved."""
        return self._preserve_thinking

    def _resolve_model(self, model: str) -> str:
        """Return the provided model, falling back to the default if empty."""
        return model if model else self._default_model

    @abstractmethod
    def get_models(self) -> list[str]:
        """Return a list of available model identifiers."""
        ...  # pragma: no cover

    @abstractmethod
    def count_tokens(self, messages: list[dict[str, str]], _model: str = "") -> int:
        """Count the tokens in a list of messages.

        Args:
            messages: The list of message dicts with ``role`` and ``content``.
            model: The model identifier to use for token counting.

        Returns:
            The number of input tokens.

        """
        ...  # pragma: no cover

    @abstractmethod
    def tools_support(self, model: str = "") -> bool:
        """Return whether the current model supports tool/function calling."""
        ...  # pragma: no cover

    @abstractmethod
    def input_modes(self, model: str = "") -> list[str]:
        """Return a list of supported input modalities (e.g. ``'text'``, ``'image'``)."""
        ...  # pragma: no cover

    @abstractmethod
    def output_modes(self, model: str = "") -> list[str]:
        """Return a list of supported output modalities (e.g. ``'text'``, ``'image'``)."""
        ...  # pragma: no cover

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        sampling_params: SamplingParams | None = None,
        stream: bool = False,
        update_callback: Callable[[dict[str, Any], str | None], None] | None = None,
        model: str = "",
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: The conversation history (list of role/content dicts).
            sampling_params: Optional dictionary of sampling parameters
                (temperature, top_k, top_p, min_p, ...).
            stream: Whether to stream the response.
            update_callback: Optional callback invoked for each streaming chunk.
                Takes the chunk data and an optional subtask name.
            model: The model identifier to use.
            max_retries: Maximum number of retries on 500 errors.
            timeout: Optional request timeout in seconds to override the default.

        Returns:
            A dict containing the chat completion response.

        """
        ...  # pragma: no cover

    @abstractmethod
    def get_context_size(self, model: str = "") -> int:
        """Return the current context window size for the given model."""
        ...  # pragma: no cover

    @abstractmethod
    def get_credits(self) -> dict[str, float]:
        """Return the account credit balance and usage."""
        ...  # pragma: no cover

    @abstractmethod
    def is_up(self) -> bool:
        """Check if the inference server is reachable."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# OpenAI-compatible base client
# ---------------------------------------------------------------------------


class OpenAICompatibleClient(InferenceClient):
    """Base class for OpenAI-compatible API clients.

    Provides shared implementation for chat completions, model listing,
    and retry logic used by both LlamaCpp and OpenRouter clients.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        message_timeout: float | None = None,
        verify_ssl: bool = True,
        ca_cert: str | None = None,
        default_model: str = "",
        sampling_params: SamplingParams | None = None,
        preserve_thinking: bool = False,
        api_key: str | None = None,
    ) -> None:
        super().__init__(default_model, sampling_params, preserve_thinking)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.message_timeout = message_timeout
        self.verify_ssl = verify_ssl
        self.ca_cert = ca_cert
        self.url = f"{self.base_url}/v1/chat/completions"
        self.url_models = f"{self.base_url}/v1/models"
        self.headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._models_cache: list[dict[str, Any]] | None = None
        self._models_cache_time: float = 0.0
        self._first_update_time: float | None = None
        self._last_update_time: float | None = None
        self._request_start_time: float = 0.0

    @property
    def _verify(self) -> bool | str:
        """Return the SSL verification setting for requests."""
        if self.ca_cert:
            return self.ca_cert
        return self.verify_ssl

    def _get_cached_models(self) -> list[dict[str, Any]]:
        cache_ttl = 3600
        if (
            self._models_cache is not None
            and time.monotonic() - self._models_cache_time < cache_ttl
        ):
            return self._models_cache
        resp = requests.get(
            self.url_models,
            headers=self.headers,
            timeout=self.timeout,
            verify=self._verify,
        )
        resp.raise_for_status()
        data = self._parse_json(resp)
        models: list[dict[str, Any]] = data.get("data", [])
        self._models_cache = models
        self._models_cache_time = time.monotonic()
        return models

    @staticmethod
    def _parse_json(resp: requests.Response) -> Any:
        """Parse JSON from a response, raising a clear error on failure."""
        content_type = resp.headers.get("Content-Type", "")
        allowed = ("application/json", "text/plain")
        if not any(ct in content_type for ct in allowed):
            raise requests.exceptions.InvalidJSONError(
                f"Unexpected Content-Type: {content_type}. "
                f"Response ({resp.status_code}): {resp.text[:200]}"
            )
        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise requests.exceptions.InvalidJSONError(
                    f"Expected JSON object but got {type(data).__name__}: {resp.text[:200]}"
                )
            return data
        except json.JSONDecodeError as e:
            raise requests.exceptions.InvalidJSONError(
                f"Failed to parse JSON response ({resp.status_code}): {resp.text[:200]}"
            ) from e

    def get_models(self) -> list[str]:
        """Return a list of available model identifiers."""
        data = self._get_cached_models()
        models: list[str] = [model["id"] for model in data if "id" in model]
        return models

    def _chat_non_stream(
        self,
        data: dict[str, Any],
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a non-streaming chat request."""
        retries = 0
        while True:
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    json=data,
                    timeout=timeout or self.message_timeout or self.timeout,
                    verify=self._verify,
                )
                resp.raise_for_status()
                result = self._parse_json(resp)
                assert isinstance(result, dict)
                return result
            except (
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                status_code = None
                if (
                    isinstance(e, requests.exceptions.HTTPError)
                    and e.response is not None
                ):
                    status_code = e.response.status_code
                if status_code in {500, 502, 503, 504, 429} or isinstance(
                    e,
                    (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
                ):
                    retries += 1
                    if retries > max_retries:
                        raise
                    backoff = self._backoff_delay(retries)
                    time.sleep(backoff)
                else:
                    raise

    def _build_chat_data(
        self,
        messages: list[dict[str, Any]],
        sampling_params: SamplingParams | None,
        model: str,
    ) -> dict[str, Any]:
        """Build the common chat request payload."""
        merged_params = self._sampling_params.copy()
        if sampling_params is not None:
            merged_params.update(sampling_params)
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if "role" not in m or not m.get("role"):
                raise ValueError(f"Message missing required 'role' field: {m}")
            msg: dict[str, Any] = {
                "role": m["role"],
                "content": m.get("content", ""),
            }
            if "tool_calls" in m:
                msg["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                msg["tool_call_id"] = m["tool_call_id"]
            if "name" in m:
                msg["name"] = m["name"]
            msgs.append(msg)
        data: dict[str, Any] = {
            "messages": msgs,
            **merged_params,
        }
        if model:
            data["model"] = model
        return data

    def chat(
        self,
        messages: list[dict[str, Any]],
        sampling_params: SamplingParams | None = None,
        stream: bool = False,
        update_callback: Callable[[dict[str, Any], str | None], None] | None = None,
        model: str = "",
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: The conversation history (list of role/content dicts).
            sampling_params: Optional dictionary of sampling parameters
                (temperature, top_k, top_p, min_p, ...).
            stream: Whether to stream the response.
            update_callback: Optional callback invoked for each streaming chunk.
                Takes the chunk data and an optional subtask name.
            model: The model identifier to use.
            max_retries: Maximum number of retries on 500 errors.
            timeout: Optional request timeout in seconds to override the default.

        Returns:
            A dict containing the chat completion response.

        """
        resolved_model = self._resolve_model(model)
        data = self._build_chat_data(messages, sampling_params, resolved_model)

        if stream:
            if update_callback is None:

                def _noop_callback(
                    _chunk: dict[str, Any], _subtask_name: str | None = None
                ) -> None:
                    pass

                update_callback = _noop_callback
            return self._chat_stream(data, update_callback, max_retries, timeout)

        return self._chat_non_stream(data, max_retries, timeout)

    def _prepare_stream_data(self, data: dict[str, Any]) -> None:
        """Hook to add backend-specific fields to streaming request data."""
        pass

    def _process_stream_delta(self, delta: dict[str, Any]) -> dict[str, Any]:
        """Hook to transform delta before callback."""
        return delta

    def _build_stream_callback(
        self, delta: dict[str, Any], delta_type: str, _msg: dict[str, Any]
    ) -> dict[str, Any]:
        """Hook to build the callback dict. Subclasses can add extra fields."""
        return {"delta": delta, "delta_type": delta_type}

    def _build_stream_result(
        self,
        content: str,
        reasoning: str,
        msg: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the final result dict from streaming response."""
        usage = msg.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens_details = usage.get("prompt_tokens_details", {})
        cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
        cache_write_tokens = prompt_tokens_details.get("cache_write_tokens", 0)

        if prompt_tokens is None:
            prompt_tokens = 0
        if completion_tokens is None:
            completion_tokens = 0
        if cached_tokens is None:
            cached_tokens = 0
        if cache_write_tokens is None:
            cache_write_tokens = 0

        prompt_time = (
            self._first_update_time or time.monotonic()
        ) - self._request_start_time
        completion_time = (self._last_update_time or time.monotonic()) - (
            self._first_update_time or time.monotonic()
        )

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
            "timings": {
                "prompt_time": prompt_time,
                "completion_time": completion_time,
            },
            "model": msg.get("model", data.get("model")),
            "provider": self.base_url,
        }

    def _handle_stream_error(
        self, msg: dict[str, Any], retries: int, max_retries: int
    ) -> tuple[bool, int]:
        """Handle an error in the stream. Return (should_retry, updated_retries)."""
        if "error" in msg:
            error_code = msg["error"].get("code")
            retryable_codes = {500, 502, 503, 504, 429}
            if error_code in retryable_codes:
                retries += 1
                if retries > max_retries:
                    raise RuntimeError(f"LLM error after {max_retries} retries: {msg}")
                backoff = self._backoff_delay(retries)
                time.sleep(backoff)
                return True, retries
            raise RuntimeError(f"LLM error: {msg}")
        return False, retries

    def _should_stop_streaming(self, msg: dict[str, Any]) -> bool:
        return bool(msg.get("done"))

    def _on_stream_update(self, msg: dict[str, Any]) -> None:
        """Hook called for each non-choice message (e.g. progress updates)."""
        pass

    def _on_stream_delta(self, _delta: dict[str, Any], delta_type: str) -> None:
        if not delta_type:
            return
        now = time.monotonic()
        if self._first_update_time is None:
            self._first_update_time = now
        self._last_update_time = now

    @staticmethod
    def _backoff_delay(retries: int) -> float:
        """Compute exponential backoff with jitter."""
        return float(min(2**retries, 60)) + random.uniform(0, 1)

    @staticmethod
    def _iter_stream_lines(
        resp: requests.Response,
    ) -> Generator[dict[str, Any], None, None]:
        for line in resp.iter_lines():
            if line is None or not line.startswith(b"data: "):
                continue
            data_part = line[6:].strip()
            if data_part == b"[DONE]":
                yield {"done": True}
                continue
            try:
                msg = json.loads(data_part)
            except json.JSONDecodeError:
                logger.debug("Failed to decode SSE line: %s", data_part)
                continue
            yield msg

    def _process_stream_response(
        self,
        data: dict[str, Any],
        update_callback: Callable[[dict[str, Any], str | None], None],
        max_retries: int,
        retries: int,
        timeout: float | None,
        content: str,
        reasoning: str,
    ) -> _StreamOutcome:
        done_detected = False
        finish_reason = False
        should_retry = False
        result_msg: dict[str, Any] = {}
        has_usage_or_timings = False

        with requests.post(
            self.url,
            headers=self.headers,
            json=data,
            stream=True,
            timeout=timeout or self.message_timeout or self.timeout,
            verify=self._verify,
        ) as resp:
            try:
                processing_active = True
                for msg in self._iter_stream_lines(resp):
                    if not processing_active:
                        break
                    if self._should_stop_streaming(msg):
                        done_detected = True
                        processing_active = False
                    else:
                        should_retry, retries = self._handle_stream_error(
                            msg, retries, max_retries
                        )
                        if should_retry:
                            processing_active = False
                        else:
                            choices = msg.get("choices", [])
                            if not choices:
                                self._on_stream_update(msg)

                            for choice in choices:
                                finish = choice.get("finish_reason")
                                if finish is not None:
                                    finish_reason = True
                                delta = choice.get("delta", {})
                                if delta and not finish_reason:
                                    delta = self._process_stream_delta(delta)
                                    delta_type = "reasoning"
                                    if "reasoning" in delta and delta["reasoning"]:
                                        reasoning += delta["reasoning"]
                                    if "content" in delta and delta["content"]:
                                        content += delta["content"]
                                        delta_type = "content"
                                    callback_data = self._build_stream_callback(
                                        delta, delta_type, msg
                                    )
                                    self._on_stream_delta(delta, delta_type)
                                    update_callback(callback_data, None)

                            if "usage" in msg or "timings" in msg:
                                result_msg = msg
                                has_usage_or_timings = True
                                processing_active = False
            except requests.exceptions.ChunkedEncodingError:
                retries += 1
                if retries > max_retries:
                    return _StreamOutcome(
                        status=StreamStatus.ERROR,
                        error=RuntimeError(
                            f"Stream failed after {max_retries} retries: ChunkedEncodingError"
                        ),
                        content=content,
                        reasoning=reasoning,
                        retries=retries,
                    )
                backoff = self._backoff_delay(retries)
                time.sleep(backoff)
                return _StreamOutcome(
                    status=StreamStatus.RETRY,
                    content=content,
                    reasoning=reasoning,
                    retries=retries,
                )

        if should_retry:
            return _StreamOutcome(
                status=StreamStatus.RETRY,
                content=content,
                reasoning=reasoning,
                retries=retries,
            )

        success = done_detected or has_usage_or_timings or finish_reason
        if not success:
            return _StreamOutcome(
                status=StreamStatus.ERROR,
                error=RuntimeError("Stream ended without finish_reason"),
                content=content,
                reasoning=reasoning,
                retries=retries,
            )

        return _StreamOutcome(
            status=StreamStatus.RETURN,
            result=self._build_stream_result(content, reasoning, result_msg, data),
            content=content,
            reasoning=reasoning,
            retries=retries,
        )

    def _chat_stream(
        self,
        data: dict[str, Any],
        update_callback: Callable[[dict[str, Any], str | None], None],
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self._first_update_time = None
        self._last_update_time = None
        self._request_start_time = time.monotonic()
        data["stream"] = True
        self._prepare_stream_data(data)

        content = ""
        reasoning = ""
        retries = 0

        while True:
            outcome = self._process_stream_response(
                data,
                update_callback,
                max_retries,
                retries,
                timeout,
                content,
                reasoning,
            )
            if outcome.status == StreamStatus.RETURN:
                assert outcome.result is not None
                return outcome.result
            if outcome.status == StreamStatus.ERROR:
                assert outcome.error is not None
                raise outcome.error
            assert outcome.error is None
            content = outcome.content
            reasoning = outcome.reasoning
            retries = outcome.retries


# ---------------------------------------------------------------------------
# LlamaCpp client
# ---------------------------------------------------------------------------


class LlamaCpp(OpenAICompatibleClient):
    """Client for a self-hosted llama.cpp server.

    The llama.cpp server exposes an OpenAI-compatible API at ``/v1``.  In
    addition it provides llama.cpp-specific endpoints such as ``/props``
    (server capabilities) and ``/v1/messages/count_tokens``.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:8080",
        timeout: float = 5,
        message_timeout: float | None = 120,
        verify_ssl: bool = True,
        ca_cert: str | None = None,
        default_model: str = "",
        sampling_params: SamplingParams | None = None,
        preserve_thinking: bool = False,
        api_key: str | None = None,
    ) -> None:
        """Initialise the LlamaCpp client.

        Args:
            host: Base URL of the llama.cpp server (no trailing slash).
            timeout: Request timeout in seconds.
            message_timeout: Optional per-message timeout override.
            verify_ssl: Whether to verify SSL certificates.
            ca_cert: Optional path to a CA certificate bundle file.
        """
        super().__init__(
            validate_host(host) if host else "http://127.0.0.1:8080",
            timeout=timeout,
            message_timeout=message_timeout,
            verify_ssl=verify_ssl,
            ca_cert=ca_cert,
            default_model=default_model,
            sampling_params=sampling_params,
            preserve_thinking=preserve_thinking,
            api_key=api_key,
        )
        self.url_token_count = f"{self.base_url}/v1/messages/count_tokens"

    def _prepare_stream_data(self, data: dict[str, Any]) -> None:
        data["return_progress"] = True

    def _process_stream_delta(self, delta: dict[str, Any]) -> dict[str, Any]:
        if "reasoning_content" in delta:
            delta["reasoning"] = delta["reasoning_content"]
            del delta["reasoning_content"]
        return delta

    def _build_stream_callback(
        self, delta: dict[str, Any], delta_type: str, msg: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "delta": delta,
            "delta_type": delta_type,
            "prompt_progress": msg.get("prompt_progress"),
        }

    def _build_stream_result(
        self,
        content: str,
        reasoning: str,
        msg: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        timings = msg.get("timings", {})
        cache_n = timings.get("cache_n", 0)
        prompt_n = timings.get("prompt_n", 0) + cache_n
        cache_write_n = prompt_n - cache_n if prompt_n >= cache_n else 0
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": prompt_n,
                "completion_tokens": timings.get("predicted_n", 0),
                "cached_tokens": cache_n,
                "cache_write_tokens": cache_write_n,
            },
            "timings": {
                "prompt_time": timings.get("prompt_ms", 0) / 1000,
                "completion_time": timings.get("predicted_ms", 0) / 1000,
            },
            "model": data.get("model"),
            "provider": "LlamaCpp/" + self.base_url,
        }

    def get_context_size(self, model: str = "") -> int:
        """Return the current context window size from the server props."""
        resolved = self._resolve_model(model)
        resp = requests.get(
            f"{self.base_url}/props",
            headers=self.headers,
            timeout=self.timeout,
            verify=self._verify,
        )
        resp.raise_for_status()
        data = self._parse_json(resp)
        settings = data.get("default_generation_settings")
        if not settings or "n_ctx" not in settings:
            raise ValueError("Model response contains no data")
        loaded_model = data.get("model")
        if loaded_model and resolved and loaded_model != resolved:
            logger.warning(
                "Requested model '%s' but server has '%s' loaded",
                resolved,
                loaded_model,
            )
        return int(settings["n_ctx"])

    def get_credits(self) -> dict[str, float]:
        """Return credit balance (always zero for self-hosted)."""
        return {"total_credits": 0.0, "total_usage": 0.0}

    def is_up(self) -> bool:
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=5,
                verify=self._verify,
            )
            data = self._parse_json(resp)
            assert isinstance(data, dict)
        except Exception as e:
            logger.debug("LlamaCpp health check failed: %s", e)
            return False
        return bool(data.get("status") == "ok")

    def count_tokens(self, messages: list[dict[str, str]], _model: str = "") -> int:
        """Count tokens via the llama.cpp ``/v1/messages/count_tokens`` endpoint."""
        resolved = self._resolve_model(_model)
        resp = requests.post(
            self.url_token_count,
            json={"messages": messages, "model": resolved},
            headers=self.headers,
            timeout=self.timeout,
            verify=self._verify,
        )
        resp.raise_for_status()
        return int(self._parse_json(resp)["input_tokens"])

    def tools_support(self, _model: str = "") -> bool:
        """Check tool-calling capability via the llama.cpp ``/props`` endpoint."""
        resp = requests.get(
            f"{self.base_url}/props",
            headers=self.headers,
            timeout=self.timeout,
            verify=self._verify,
        )
        resp.raise_for_status()
        data = self._parse_json(resp)
        caps = data.get("chat_template_caps")
        if not caps:
            raise ValueError(
                "Missing 'chat_template_caps' in /props response; "
                "cannot determine tool support"
            )
        return bool(caps.get("supports_tools") and caps.get("supports_tool_calls"))

    def input_modes(self, _model: str = "") -> list[str]:
        """Return the supported input modalities via the llama.cpp ``/props`` endpoint."""
        resp = requests.get(
            f"{self.base_url}/props",
            headers=self.headers,
            timeout=self.timeout,
            verify=self._verify,
        )
        resp.raise_for_status()
        data = self._parse_json(resp)
        mods = data.get("modalities", {})
        modes: list[str] = ["text"]
        if "vision" in mods and mods["vision"]:
            modes.append("image")
        if "audio" in mods and mods["audio"]:
            modes.append("audio")
        return modes

    def output_modes(self, _model: str = "") -> list[str]:
        """Return the supported output modalities for llama.cpp."""
        return ["text"]


# ---------------------------------------------------------------------------
# DwarfStar4 client
# ---------------------------------------------------------------------------


class DwarfStar4(OpenAICompatibleClient):
    """Client for a DwarfStar4 API-compatible inference server.

    DwarfStar4 is a lightweight, OpenAI-compatible inference backend.

    * Tool support is always enabled.
    * Only text input and text output are supported.
    * No credits endpoint is exposed.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:8080",
        timeout: float = 5,
        message_timeout: float | None = 120,
        verify_ssl: bool = True,
        ca_cert: str | None = None,
        default_model: str = "",
        sampling_params: SamplingParams | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            validate_host(host) if host else "http://127.0.0.1:8080",
            timeout=timeout,
            message_timeout=message_timeout,
            verify_ssl=verify_ssl,
            ca_cert=ca_cert,
            default_model=default_model,
            sampling_params=sampling_params,
            preserve_thinking=True,
            api_key=api_key,
        )

    def get_context_size(self, _model: str = "") -> int:
        """Return the fixed context size for DwarfStar4."""
        return 1000000

    def is_up(self) -> bool:
        """Check if the DwarfStar4 server is reachable via the models endpoint."""
        try:
            resp = requests.get(
                self.url_models,
                headers=self.headers,
                timeout=5,
                verify=self._verify,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug("DwarfStar4 health check failed: %s", e)
            return False

    def get_credits(self) -> dict[str, float]:
        """Return credit balance (always zero for DwarfStar4)."""
        return {"total_credits": 0.0, "total_usage": 0.0}

    def count_tokens(self, messages: list[dict[str, str]], _model: str = "") -> int:
        """Estimate token count using DeepSeek-specific heuristics.

        DeepSeek's BPE tokenizer averages ~3 characters per token for typical
        text, with ~5 tokens of overhead per message for role markers and
        chat-template formatting tokens.
        """
        total_chars = 0
        for msg in messages:
            total_chars += len(msg.get("content", "") or "")
            if self._preserve_thinking:
                total_chars += len(msg.get("reasoning_content", "") or "")
        num_messages = len(messages)
        return max(1, total_chars // 3 + num_messages * 5)

    def tools_support(self, _model: str = "") -> bool:
        """Return True - DwarfStar4 always supports tool calling."""
        return True

    def input_modes(self, _model: str = "") -> list[str]:
        """Return the supported input modalities (always text for DwarfStar4)."""
        return ["text"]

    def output_modes(self, _model: str = "") -> list[str]:
        """Return the supported output modalities (always text for DwarfStar4)."""
        return ["text"]

    def _prepare_stream_data(self, data: dict[str, Any]) -> None:
        data["stream_options"] = {"include_usage": True}

    def _process_stream_delta(self, delta: dict[str, Any]) -> dict[str, Any]:
        if "reasoning_content" in delta:
            delta["reasoning"] = delta["reasoning_content"]
            del delta["reasoning_content"]
        return delta

    def _build_stream_result(
        self,
        content: str,
        reasoning: str,
        msg: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        result = super()._build_stream_result(content, reasoning, msg, data)
        result["provider"] = "DwarfStar4/" + self.base_url
        return result


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------


class OpenRouter(OpenAICompatibleClient):
    """Client for the OpenRouter.ai API gateway.

    OpenRouter aggregates many LLM providers behind a single OpenAI-compatible
    endpoint.

    * The ``HTTP-Referer`` and ``X-Title`` headers are required for analytics.
    * Token counting is not available as a standalone endpoint; we estimate via
      a dry-run chat call.
    """

    BASE_URL = "https://openrouter.ai/api"

    def __init__(
        self,
        api_key: str,
        timeout: float = 60.0,
        message_timeout: float | None = None,
        verify_ssl: bool = True,
        ca_cert: str | None = None,
        default_model: str = "openrouter/free",
        sampling_params: SamplingParams | None = None,
        preserve_thinking: bool = False,
    ) -> None:
        super().__init__(
            self.BASE_URL,
            timeout=timeout,
            message_timeout=message_timeout,
            verify_ssl=verify_ssl,
            ca_cert=ca_cert,
            default_model=default_model,
            sampling_params=sampling_params,
            preserve_thinking=preserve_thinking,
            api_key=api_key,
        )
        self.url_credits = f"{self.base_url}/v1/credits"
        if os.environ.get("TIZ_DROP_OPENROUTER_REPORTING") != "1":
            self.headers.update(
                {
                    "HTTP-Referer": OPENROUTER_REFERER,
                    "X-Title": OPENROUTER_APP_TITLE,
                    "X-OpenRouter-Categories": OPENROUTER_CATEGORIES,
                }
            )

    @staticmethod
    def _strip_model_suffix(model: str) -> str:
        for suffix in (":nitro", ":exacto"):
            if model.endswith(suffix):
                model = model[: -len(suffix)]
        return model

    def _find_model(self, model: str) -> dict[str, Any] | None:
        """Look up model data by resolved model name (with suffix stripped)."""
        resolved = self._strip_model_suffix(self._resolve_model(model))
        models = self._get_cached_models()
        for model_data in models:
            if model_data.get("id") == resolved:
                return model_data
        return None

    def get_context_size(self, model: str = "") -> int:
        model_data = self._find_model(model)
        if model_data is None:
            resolved = self._strip_model_suffix(self._resolve_model(model))
            raise ValueError(f"Model '{resolved}' not found in OpenRouter response")
        context_length = model_data.get("context_length")
        if context_length is None:
            resolved = self._strip_model_suffix(self._resolve_model(model))
            raise ValueError(f"Could not determine context size for model {resolved}")
        return int(context_length)

    def is_up(self) -> bool:
        """Check if the OpenRouter API is reachable via the models endpoint."""
        try:
            resp = requests.get(
                self.url_models,
                headers=self.headers,
                timeout=5,
                verify=self._verify,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug("OpenRouter health check failed: %s", e)
            return False

    def get_credits(self) -> dict[str, float]:
        """Retrieve credit balance and usage from OpenRouter."""
        resp = requests.get(
            self.url_credits,
            headers=self.headers,
            timeout=self.timeout,
            verify=self._verify,
        )
        resp.raise_for_status()
        data = self._parse_json(resp)
        credits_data = data.get("data", {})
        return {
            "total_credits": float(credits_data.get("total_credits", 0)),
            "total_usage": float(credits_data.get("total_usage", 0)),
        }

    def count_tokens(self, messages: list[dict[str, str]], _model: str = "") -> int:
        """Estimate token count without making an API call.

        Uses a heuristic of ~4 characters per token plus a small overhead per
        message for formatting tokens.  This is a rough approximation and may
        differ from the true tokenization of the target model.
        """
        total_chars = 0
        for msg in messages:
            total_chars += len(msg.get("content", "") or "")
            if self._preserve_thinking:
                total_chars += len(msg.get("reasoning_content", "") or "")
        num_messages = len(messages)
        return max(1, total_chars // 4 + num_messages * 4)

    def tools_support(self, model: str = "") -> bool:
        model_data = self._find_model(model)
        if model_data is None:
            return False
        supported_params = model_data.get("supported_parameters", [])
        return "tools" in supported_params

    def input_modes(self, model: str = "") -> list[str]:
        model_data = self._find_model(model)
        if model_data is None:
            return ["text"]
        architecture = model_data.get("architecture", {})
        result: list[str] = architecture.get("input_modalities", ["text"])
        return result

    def output_modes(self, model: str = "") -> list[str]:
        model_data = self._find_model(model)
        if model_data is None:
            return ["text"]
        architecture = model_data.get("architecture", {})
        result: list[str] = architecture.get("output_modalities", ["text"])
        return result

    def _build_stream_result(
        self,
        content: str,
        reasoning: str,
        msg: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        result = super()._build_stream_result(content, reasoning, msg, data)
        usage = msg.get("usage", {})
        cost = usage.get("cost", 0)
        cost_details = usage.get("cost_details", 0)
        if cost is None:
            cost = 0
        if cost_details is None:
            cost_details = 0
        result["usage"]["cost"] = cost
        result["usage"]["cost_details"] = cost_details
        result["provider"] = "OpenRouter/" + str(msg.get("provider") or "Unknown")
        return result


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------


class AnthropicClient(InferenceClient):
    """Client for the Anthropic API (Claude models).

    Communicates with the Anthropic Messages API endpoint directly.
    Supports both streaming and non-streaming chat completions,
    thinking/reasoning content, and tool use.

    * Token counting uses a heuristic (no standalone endpoint).
    """

    BASE_URL = "https://api.anthropic.com"

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-5",
        sampling_params: SamplingParams | None = None,
        preserve_thinking: bool = False,
        timeout: float = 60.0,
        message_timeout: float | None = None,
    ) -> None:
        """Initialise the Anthropic client.

        Args:
            api_key: The Anthropic API key.
            default_model: Default model identifier.
            sampling_params: Optional default sampling parameters.
            preserve_thinking: Whether to preserve thinking/reasoning content.
            timeout: Default request timeout in seconds.
            message_timeout: Optional per-message timeout override.
        """
        super().__init__(default_model, sampling_params, preserve_thinking)
        self.api_key = api_key
        self.timeout = timeout
        self.message_timeout = message_timeout
        self.url = f"{self.BASE_URL}/v1/messages"
        self.headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        self._first_update_time: float | None = None
        self._last_update_time: float | None = None
        self._request_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_anthropic_sse(
        resp: requests.Response,
    ) -> Generator[dict[str, Any], None, None]:
        """Parse Anthropic SSE stream, yielding event data dicts.

        Each yielded dict includes the event type under the ``_event`` key.
        """
        current_event = ""
        current_data_lines: list[str] = []
        for raw_line in resp.iter_lines():
            if raw_line is None:
                continue
            decoded: str = (
                raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            )
            if decoded.startswith("event: "):
                current_event = decoded[7:]
            elif decoded.startswith("data: "):
                current_data_lines.append(decoded[6:])
            elif decoded == "":
                if current_data_lines:
                    data_str = "".join(current_data_lines).strip()
                    if data_str:
                        try:
                            data = json.loads(data_str)
                            data["_event"] = current_event
                            yield data
                        except json.JSONDecodeError:
                            pass
                current_event = ""
                current_data_lines = []
        if current_data_lines:
            data_str = "".join(current_data_lines).strip()
            if data_str:
                try:
                    data = json.loads(data_str)
                    data["_event"] = current_event
                    yield data
                except json.JSONDecodeError:
                    pass

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool definitions to Anthropic format."""
        result: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                result.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
        return result

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Convert OpenAI-format messages to Anthropic format.

        Returns ``(system, anthropic_messages)``.
        """
        system_parts: list[dict[str, Any]] = []
        anthropic_msgs: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "system":
                if content:
                    system_parts.append({"type": "text", "text": content})
                continue

            if role == "tool":
                blocks: list[dict[str, Any]] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id or "",
                        "content": content or "",
                    }
                ]
                anthropic_msgs.append({"role": "user", "content": blocks})
                continue

            if role == "assistant" and tool_calls:
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "input": args,
                        }
                    )
                anthropic_msgs.append({"role": "assistant", "content": blocks})
                continue

            anthropic_msgs.append({"role": role, "content": content or ""})

        system: str | list[dict[str, Any]] | None = None
        if len(system_parts) == 1:
            system = system_parts[0]["text"]
        elif len(system_parts) > 1:
            system = system_parts

        return system, anthropic_msgs

    @staticmethod
    def _backoff_delay(retries: int) -> float:
        """Compute exponential backoff with jitter."""
        return float(min(2**retries, 60)) + random.uniform(0, 1)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_models(self) -> list[str]:
        """Return the configured model (Anthropic does not expose a public list)."""
        return [self._resolve_model("")]

    def count_tokens(self, messages: list[dict[str, str]], _model: str = "") -> int:
        """Estimate token count using a heuristic.

        Anthropic's tokenizer averages ~4 characters per token, with ~5
        tokens of overhead per message for role markers and formatting.
        """
        total_chars = 0
        for msg in messages:
            total_chars += len(msg.get("content", "") or "")
            if self._preserve_thinking:
                total_chars += len(msg.get("reasoning_content", "") or "")
        num_messages = len(messages)
        return max(1, total_chars // 4 + num_messages * 5)

    def tools_support(self, _model: str = "") -> bool:
        """Return True - all Claude models support tool use."""
        return True

    def input_modes(self, _model: str = "") -> list[str]:
        """Return the supported input modalities for Claude models."""
        return ["text", "image"]

    def output_modes(self, _model: str = "") -> list[str]:
        """Return the supported output modalities for Claude models."""
        return ["text"]

    def get_context_size(self, _model: str = "") -> int:
        """Return the context window size for Claude models (1M)."""
        return 1000000

    def get_credits(self) -> dict[str, float]:
        """Return credit balance (always zero - not exposed via API)."""
        return {"total_credits": 0.0, "total_usage": 0.0}

    def is_up(self) -> bool:
        """Check if the Anthropic API is reachable."""
        try:
            resp = requests.get(
                self.BASE_URL,
                headers=self.headers,
                timeout=5,
            )
            return resp.status_code < 500
        except Exception as e:
            logger.debug("Anthropic health check failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Chat implementation
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        sampling_params: SamplingParams | None = None,
        stream: bool = False,
        update_callback: Callable[[dict[str, Any], str | None], None] | None = None,
        model: str = "",
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to the Anthropic API.

        Args:
            messages: The conversation history (list of role/content dicts).
            sampling_params: Optional dictionary of sampling parameters.
            stream: Whether to stream the response.
            update_callback: Optional callback invoked for each streaming chunk.
            model: The model identifier to use.
            max_retries: Maximum number of retries on 500 errors.
            timeout: Optional request timeout in seconds.

        Returns:
            A dict containing the chat completion response.
        """
        resolved_model = self._resolve_model(model)
        system, anthropic_msgs = self._convert_messages(messages)

        merged_params = self._sampling_params.copy()
        if sampling_params is not None:
            merged_params.update(sampling_params)

        data: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": merged_params.pop("max_tokens", 4096),
            "messages": anthropic_msgs,
        }
        if system is not None:
            data["system"] = system

        tools_raw = merged_params.pop("tools", None)
        if tools_raw:
            converted = self._convert_tools(tools_raw)
            if converted:
                data["tools"] = converted

        data.update(merged_params)

        if stream:
            data["stream"] = True
            if update_callback is None:

                def _noop_callback(
                    _chunk: dict[str, Any], _subtask_name: str | None = None
                ) -> None:
                    pass

                update_callback = _noop_callback
            return self._chat_stream(data, update_callback, max_retries, timeout)

        return self._chat_non_stream(data, max_retries, timeout)

    def _chat_non_stream(
        self,
        data: dict[str, Any],
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a non-streaming chat request."""
        retries = 0
        while True:
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    json=data,
                    timeout=timeout or self.message_timeout or self.timeout,
                )
                resp.raise_for_status()
                result = resp.json()
                return self._build_result(result, data)
            except (
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                status_code = None
                if (
                    isinstance(e, requests.exceptions.HTTPError)
                    and e.response is not None
                ):
                    status_code = e.response.status_code
                if status_code in {500, 502, 503, 504, 429} or isinstance(
                    e,
                    (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
                ):
                    retries += 1
                    if retries > max_retries:
                        raise
                    backoff = self._backoff_delay(retries)
                    time.sleep(backoff)
                else:
                    raise

    def _chat_stream(
        self,
        data: dict[str, Any],
        update_callback: Callable[[dict[str, Any], str | None], None],
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a streaming chat request and process SSE events."""
        self._first_update_time = None
        self._last_update_time = None
        self._request_start_time = time.monotonic()

        retries = 0
        content_blocks: list[dict[str, Any]] = []
        text_content = ""
        thinking_content = ""
        input_tokens = 0
        output_tokens = 0

        while True:
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    json=data,
                    stream=True,
                    timeout=timeout or self.message_timeout or self.timeout,
                )
                resp.raise_for_status()

                for event in self._parse_anthropic_sse(resp):
                    event_type = event.get("_event", "")
                    if event_type == "message_start":
                        msg = event.get("message", {})
                        usage = msg.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                    elif event_type == "content_block_start":
                        block = event.get("content_block", {})
                        content_blocks.append(block)
                        block_type = block.get("type", "")
                        if block_type == "thinking":
                            thinking_content += block.get("thinking", "")
                            delta_info: dict[str, Any] = {
                                "delta": {"reasoning": block.get("thinking", "")},
                                "delta_type": "reasoning",
                            }
                            now = time.monotonic()
                            if self._first_update_time is None:
                                self._first_update_time = now
                            self._last_update_time = now
                            update_callback(delta_info, None)
                        elif block_type == "tool_use":
                            pass
                        elif block_type == "text":
                            text_content += block.get("text", "")
                            delta_info = {
                                "delta": {"content": block.get("text", "")},
                                "delta_type": "content",
                            }
                            now = time.monotonic()
                            if self._first_update_time is None:
                                self._first_update_time = now
                            self._last_update_time = now
                            update_callback(delta_info, None)
                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        delta_type = delta.get("type", "")
                        if delta_type == "thinking_delta":
                            thinking_text = delta.get("thinking", "")
                            thinking_content += thinking_text
                            delta_info = {
                                "delta": {"reasoning": thinking_text},
                                "delta_type": "reasoning",
                            }
                            now = time.monotonic()
                            if self._first_update_time is None:
                                self._first_update_time = now
                            self._last_update_time = now
                            update_callback(delta_info, None)
                        elif delta_type == "text_delta":
                            text = delta.get("text", "")
                            text_content += text
                            delta_info = {
                                "delta": {"content": text},
                                "delta_type": "content",
                            }
                            now = time.monotonic()
                            if self._first_update_time is None:
                                self._first_update_time = now
                            self._last_update_time = now
                            update_callback(delta_info, None)
                        elif delta_type == "input_json_delta":
                            idx = event.get("index", len(content_blocks) - 1)
                            if (
                                0 <= idx < len(content_blocks)
                                and content_blocks[idx].get("type") == "tool_use"
                            ):
                                partial = delta.get("partial_json", "")
                                if partial:
                                    buf = content_blocks[idx].get("__json_buf", "")
                                    buf += partial
                                    content_blocks[idx]["__json_buf"] = buf
                    elif event_type == "content_block_stop":
                        idx = event.get("index", len(content_blocks) - 1)
                        if (
                            0 <= idx < len(content_blocks)
                            and content_blocks[idx].get("type") == "tool_use"
                        ):
                            buf = content_blocks[idx].pop("__json_buf", None)
                            if buf:
                                with contextlib.suppress(json.JSONDecodeError):
                                    content_blocks[idx]["input"] = json.loads(buf)
                    elif event_type == "message_delta":
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", output_tokens)
                    elif event_type == "message_stop":
                        break

                break
            except (
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                status_code = None
                if (
                    isinstance(e, requests.exceptions.HTTPError)
                    and e.response is not None
                ):
                    status_code = e.response.status_code
                if status_code in {500, 502, 503, 504, 429} or isinstance(
                    e,
                    (
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.ChunkedEncodingError,
                    ),
                ):
                    retries += 1
                    if retries > max_retries:
                        raise RuntimeError(
                            f"Stream failed after {max_retries} retries"
                        ) from e
                    backoff = self._backoff_delay(retries)
                    time.sleep(backoff)
                    continue
                raise

        # Extract tool calls from content blocks
        tool_calls_list: list[dict[str, Any]] = []
        for block in content_blocks:
            if block.get("type") == "tool_use":
                tool_calls_list.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        prompt_time = (self._first_update_time or time.monotonic()) - (
            self._request_start_time or time.monotonic()
        )
        completion_time = (self._last_update_time or time.monotonic()) - (
            self._first_update_time or time.monotonic()
        )

        result: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text_content,
                        "reasoning_content": thinking_content,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {
                "prompt_time": prompt_time,
                "completion_time": completion_time,
            },
            "model": data.get("model", ""),
            "provider": "Anthropic/" + self.BASE_URL,
        }

        if tool_calls_list:
            result["choices"][0]["message"]["tool_calls"] = tool_calls_list

        return result

    def _build_result(
        self, api_result: dict[str, Any], request_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert an Anthropic API response to the standard response format.

        Args:
            api_result: The raw response from the Anthropic API.
            request_data: The request data dict (for model name fallback).

        Returns:
            A dict with ``choices``, ``usage``, ``timings``, ``model``,
            ``provider`` keys.
        """
        content_blocks: list[dict[str, Any]] = api_result.get("content", [])
        text_content = ""
        thinking_content = ""
        tool_calls_list = []

        for block in content_blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                text_content += block.get("text", "")
            elif block_type == "thinking":
                thinking_content += block.get("thinking", "")
            elif block_type == "tool_use":
                tool_calls_list.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        usage = api_result.get("usage", {})
        result: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text_content,
                        "reasoning_content": thinking_content,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0, "completion_time": 0},
            "model": api_result.get("model", request_data.get("model", "")),
            "provider": "Anthropic/" + self.BASE_URL,
        }

        if tool_calls_list:
            result["choices"][0]["message"]["tool_calls"] = tool_calls_list

        return result
