# ruff: noqa: ARG002
"""Tests for inference_clients module."""

import json
import os
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from tiz.inference_clients import (
    USER_AGENT,
    AnthropicClient,
    DwarfStar4,
    InferenceClient,
    LlamaCpp,
    OpenRouter,
)

# ---------------------------------------------------------------------------
# Concrete subclass of InferenceClient for testing abstract base
# ---------------------------------------------------------------------------


class DummyClient(InferenceClient):
    def get_models(self):
        return ["dummy-model"]

    def count_tokens(self, _messages, _model=""):
        return 10

    def tools_support(self, _model=""):
        return True

    def input_modes(self, _model=""):
        return ["text"]

    def output_modes(self, _model=""):
        return ["text"]

    def chat(
        self,
        _messages,
        _sampling_params=None,
        _stream=False,
        _update_callback=None,
        _model="",
        _max_retries=3,
    ):
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    def get_context_size(self, _model=""):
        return 4096

    def get_credits(self):
        return {"total_credits": 0.0, "total_usage": 0.0}

    def is_up(self):
        return True


# ===========================================================================
# InferenceClient (abstract base) tests
# ===========================================================================


def test_inference_client_default_model():
    client = DummyClient(default_model="test-model")
    assert client._default_model == "test-model"


def test_inference_client_set_default_model():
    client = DummyClient()
    client.set_default_model("new-model")
    assert client._default_model == "new-model"


def test_inference_client_resolve_model_with_value():
    client = DummyClient(default_model="default")
    assert client._resolve_model("explicit") == "explicit"


def test_inference_client_resolve_model_empty_fallback():
    client = DummyClient(default_model="default")
    assert client._resolve_model("") == "default"


def test_inference_client_abstract_methods():
    with pytest.raises(TypeError):
        InferenceClient()


def test_inference_client_sampling_params_getter():
    client = DummyClient(sampling_params={"temperature": 0.5})
    assert client.sampling_params == {"temperature": 0.5}


def test_inference_client_sampling_params_setter():
    client = DummyClient(sampling_params={"temperature": 0.5})
    client.sampling_params = {"temperature": 0.9, "top_p": 0.8}
    assert client.sampling_params == {"temperature": 0.9, "top_p": 0.8}


def test_inference_client_sampling_params_default_empty():
    client = DummyClient()
    assert client.sampling_params == {}


def test_inference_client_preserve_thinking_true():
    client = DummyClient(preserve_thinking=True)
    assert client.preserve_thinking is True


def test_inference_client_preserve_thinking_false():
    client = DummyClient(preserve_thinking=False)
    assert client.preserve_thinking is False


def test_inference_client_preserve_thinking_default():
    client = DummyClient()
    assert client.preserve_thinking is False


def test_strip_model_suffix_nitro():
    assert OpenRouter._strip_model_suffix("some-model:nitro") == "some-model"


def test_strip_model_suffix_exacto():
    assert OpenRouter._strip_model_suffix("some-model:exacto") == "some-model"


def test_strip_model_suffix_no_suffix():
    assert OpenRouter._strip_model_suffix("some-model") == "some-model"


# ===========================================================================
# OpenAICompatibleClient shared functionality tests (via LlamaCpp)
# ===========================================================================


class TestOpenAICompatibleClientInit:
    def test_init_strips_trailing_slash(self):
        client = LlamaCpp(host="http://127.0.0.1:8080/")
        assert client.base_url == "http://127.0.0.1:8080"
        assert client.url == "http://127.0.0.1:8080/v1/chat/completions"
        assert client.url_models == "http://127.0.0.1:8080/v1/models"

    def test_init_default_values(self):
        client = LlamaCpp()
        assert client.timeout == 5.0
        assert client.verify_ssl is True
        assert client.headers == {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def test_init_custom_timeout_and_verify(self):
        client = LlamaCpp(host="http://localhost", timeout=30.0, verify_ssl=False)
        assert client.timeout == 30.0
        assert client.verify_ssl is False

    def test_init_message_timeout_default(self):
        client = LlamaCpp()
        assert client.message_timeout == 120

    def test_init_message_timeout_custom(self):
        client = LlamaCpp(message_timeout=120.0)
        assert client.message_timeout == 120.0

    def test_init_message_timeout_openrouter(self):
        client = OpenRouter(api_key="sk-test", message_timeout=300.0)
        assert client.message_timeout == 300.0


class TestParseJson:
    def test_parse_json_success(self):
        resp = MagicMock()
        resp.headers = {"Content-Type": "application/json"}
        resp.json.return_value = {"key": "value"}
        result = LlamaCpp._parse_json(resp)
        assert result == {"key": "value"}

    def test_parse_json_text_plain(self):
        resp = MagicMock()
        resp.headers = {"Content-Type": "text/plain"}
        resp.json.return_value = {"key": "value"}
        result = LlamaCpp._parse_json(resp)
        assert result == {"key": "value"}

    def test_parse_json_unexpected_content_type(self):
        resp = MagicMock()
        resp.headers = {"Content-Type": "text/html"}
        resp.status_code = 500
        resp.text = "Internal Server Error"
        with pytest.raises(
            requests.exceptions.InvalidJSONError, match="Unexpected Content-Type"
        ):
            LlamaCpp._parse_json(resp)

    def test_parse_json_decode_error(self):
        resp = MagicMock()
        resp.headers = {"Content-Type": "application/json"}
        resp.status_code = 200
        resp.text = "not valid json"
        resp.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        with pytest.raises(
            requests.exceptions.InvalidJSONError, match="Failed to parse JSON"
        ):
            LlamaCpp._parse_json(resp)

    def test_parse_json_returns_list_raises(self):
        resp = MagicMock()
        resp.headers = {"Content-Type": "application/json"}
        resp.json.return_value = ["item1", "item2"]
        resp.status_code = 200
        resp.text = '["item1", "item2"]'
        with pytest.raises(
            requests.exceptions.InvalidJSONError, match="Expected JSON object"
        ):
            LlamaCpp._parse_json(resp)


def _make_mock_response(json_data, headers=None, status_code=200):
    mock_resp = MagicMock()
    mock_resp.json.return_value = json_data
    mock_resp.headers = headers or {"Content-Type": "application/json"}
    mock_resp.status_code = status_code
    mock_resp.text = json.dumps(json_data)
    return mock_resp


def _make_stream_mock(iter_lines_data, headers=None):
    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = iter_lines_data
    mock_resp.headers = headers or {"Content-Type": "application/json"}
    mock_resp.json.return_value = {}
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestGetModels:
    def test_get_models_success(self):
        client = LlamaCpp()
        mock_resp = _make_mock_response({"data": [{"id": "model1"}, {"id": "model2"}]})
        with patch(
            "tiz.inference_clients.requests.get", return_value=mock_resp
        ) as mock_get:
            models = client.get_models()
        assert models == ["model1", "model2"]
        mock_get.assert_called_once()

    def test_get_models_empty_data(self):
        client = LlamaCpp()
        mock_resp = _make_mock_response({"data": []})
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            models = client.get_models()
        assert models == []

    def test_get_models_no_data_key(self):
        client = LlamaCpp()
        mock_resp = _make_mock_response({"other": "stuff"})
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            models = client.get_models()
        assert models == []

    def test_get_models_filters_missing_id(self):
        client = LlamaCpp()
        mock_resp = _make_mock_response({"data": [{"id": "model1"}, {"no_id": True}]})
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            models = client.get_models()
        assert models == ["model1"]


class TestChatNonStream:
    def test_chat_non_stream_success(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            result = client._chat_non_stream({"messages": []})
        assert result == mock_resp.json.return_value
        mock_post.assert_called_once()

    def test_chat_non_stream_retry_on_500(self):
        client = LlamaCpp()
        error_resp = MagicMock()
        error_resp.status_code = 500
        http_error = requests.exceptions.HTTPError(response=error_resp)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", side_effect=[http_error, mock_resp]
        ) as mock_post:
            result = client._chat_non_stream({"messages": []}, max_retries=3)
        assert mock_post.call_count == 2
        assert result == mock_resp.json.return_value

    def test_chat_non_stream_retry_exhausted(self):
        client = LlamaCpp()
        error_resp = MagicMock()
        error_resp.status_code = 500
        http_error = requests.exceptions.HTTPError(response=error_resp)
        with (
            patch("tiz.inference_clients.requests.post", side_effect=http_error),
            patch("time.sleep"),
            pytest.raises(requests.exceptions.HTTPError),
        ):
            client._chat_non_stream({"messages": []}, max_retries=0)

    def test_chat_non_stream_no_retry_on_400(self):
        client = LlamaCpp()
        error_resp = MagicMock()
        error_resp.status_code = 400
        http_error = requests.exceptions.HTTPError(response=error_resp)
        with (
            patch("tiz.inference_clients.requests.post", side_effect=http_error),
            pytest.raises(requests.exceptions.HTTPError),
        ):
            client._chat_non_stream({"messages": []})

    def test_chat_non_stream_retryable_codes(self):
        client = LlamaCpp()
        for code in [500, 502, 503, 504, 429]:
            error_resp = MagicMock()
            error_resp.status_code = code
            http_error = requests.exceptions.HTTPError(response=error_resp)
            with (
                patch(
                    "tiz.inference_clients.requests.post",
                    side_effect=[http_error, http_error, http_error, http_error],
                ) as mock_post,
                patch("time.sleep"),
                pytest.raises(requests.exceptions.HTTPError),
            ):
                client._chat_non_stream({"messages": []}, max_retries=3)
            assert mock_post.call_count == 4

    def test_chat_non_stream_uses_message_timeout(self):
        client = LlamaCpp(message_timeout=300.0)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_non_stream({"messages": []})
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 300.0

    def test_chat_non_stream_uses_message_timeout_over_timeout(self):
        client = LlamaCpp(timeout=60.0, message_timeout=120.0)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_non_stream({"messages": []})
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 120.0

    def test_chat_non_stream_explicit_timeout_overrides_message_timeout(self):
        client = LlamaCpp(timeout=60.0, message_timeout=120.0)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_non_stream({"messages": []}, timeout=30.0)
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 30.0

    def test_chat_non_stream_falls_back_to_timeout_when_no_message_timeout(self):
        client = LlamaCpp(timeout=60.0, message_timeout=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_non_stream({"messages": []})
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 60.0


class TestBuildChatData:
    def test_build_chat_data_basic(self):
        client = LlamaCpp()
        data = client._build_chat_data(
            [{"role": "user", "content": "hi"}], None, "model1"
        )
        assert data["messages"] == [{"role": "user", "content": "hi"}]
        assert data["model"] == "model1"
        assert len(data) == 2

    def test_build_chat_data_with_sampling_params(self):
        client = LlamaCpp()
        data = client._build_chat_data(
            [{"role": "user", "content": "hi"}],
            {"temperature": 0.7},
            "model1",
        )
        assert data["temperature"] == 0.7
        assert data["messages"] == [{"role": "user", "content": "hi"}]
        assert data["model"] == "model1"
        assert len(data) == 3

    def test_build_chat_data_no_model(self):
        client = LlamaCpp()
        data = client._build_chat_data([{"role": "user", "content": "hi"}], None, "")
        assert "model" not in data
        assert data["messages"] == [{"role": "user", "content": "hi"}]
        assert len(data) == 1

    def test_build_chat_data_empty_sampling_params(self):
        client = LlamaCpp()
        data = client._build_chat_data(
            [{"role": "user", "content": "hi"}], {}, "model1"
        )
        assert data["messages"] == [{"role": "user", "content": "hi"}]
        assert data["model"] == "model1"
        assert len(data) == 2

    def test_build_chat_data_missing_role_raises(self):
        client = LlamaCpp()
        with pytest.raises(ValueError, match="Message missing required 'role' field"):
            client._build_chat_data([{"content": "hi"}], None, "model1")

    def test_build_chat_data_empty_role_raises(self):
        client = LlamaCpp()
        with pytest.raises(ValueError, match="Message missing required 'role' field"):
            client._build_chat_data([{"role": "", "content": "hi"}], None, "model1")


class TestChat:
    def test_chat_non_stream(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "hi"}],
                model="test",
                stream=False,
            )
        assert result == mock_resp.json.return_value

    def test_chat_stream_with_callback(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}',
            ]
        )
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            client.chat(
                [{"role": "user", "content": "hi"}],
                model="test",
                stream=True,
                update_callback=callback,
            )
        assert len(callback_calls) == 1
        assert callback_calls[0] == {
            "delta": {"content": "hi"},
            "delta_type": "content",
            "prompt_progress": None,
        }

    def test_chat_stream_without_callback_creates_noop(self):
        client = LlamaCpp()
        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "usage": {}}',
            ]
        )
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "hi"}],
                model="test",
                stream=True,
            )
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "test",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }


class TestChatStream:
    def test_stream_content_and_reasoning(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"reasoning": "thinking"}, "finish_reason": null}]}',
                b'data: {"choices": [{"delta": {"content": "answer"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "timings": {"prompt_n": 5, "predicted_n": 3, "prompt_ms": 7000, "predicted_ms": 3000}}',
            ]
        )
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback, max_retries=3)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "thinking",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "cached_tokens": 0,
                "cache_write_tokens": 5,
            },
            "timings": {"prompt_time": 7.0, "completion_time": 3.0},
            "model": "test",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }
        assert result["choices"][0]["message"]["content"] == "answer"
        assert result["choices"][0]["message"]["reasoning_content"] == "thinking"
        assert callback_calls[0] == {
            "delta": {"reasoning": "thinking"},
            "delta_type": "reasoning",
            "prompt_progress": None,
        }
        assert callback_calls[1] == {
            "delta": {"content": "answer"},
            "delta_type": "content",
            "prompt_progress": None,
        }
        assert len(callback_calls) == 2

    def test_stream_skips_non_data_lines(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b"some other line",
                None,
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
            ]
        )
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert len(callback_calls) == 1
        assert callback_calls[0] == {
            "delta": {"content": "hi"},
            "delta_type": "content",
            "prompt_progress": None,
        }
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "test",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }

    def test_stream_handles_json_decode_error(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b"data: not json",
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
            ]
        )
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert len(callback_calls) == 1
        assert callback_calls[0] == {
            "delta": {"content": "hi"},
            "delta_type": "content",
            "prompt_progress": None,
        }
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "test",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }

    def test_stream_no_choices_triggers_on_stream_update(self):
        client = LlamaCpp()
        update_calls = []

        def callback(chunk, _subtask_name=None):
            pass

        def mock_on_update(msg):
            update_calls.append(msg)

        mock_resp = _make_stream_mock(
            [
                b'data: {"prompt_progress": {"processed": 10, "total": 100}}',
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
            ]
        )
        with (
            patch("tiz.inference_clients.requests.post", return_value=mock_resp),
            patch.object(client, "_on_stream_update", mock_on_update),
        ):
            client._chat_stream({"model": "test"}, callback)
        assert len(update_calls) == 1
        assert update_calls[0] == {"prompt_progress": {"processed": 10, "total": 100}}

    def test_stream_error_raises_runtime_error(self):
        client = LlamaCpp()

        def callback(chunk, _subtask_name=None):
            pass

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with (
            patch("tiz.inference_clients.requests.post", return_value=mock_resp),
            pytest.raises(RuntimeError, match="Stream ended without finish_reason"),
        ):
            client._chat_stream({"model": "test"}, callback)

    def test_stream_uses_message_timeout(self):
        client = LlamaCpp(message_timeout=300.0)
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
            ]
        )
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_stream({"model": "test"}, callback)
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 300.0

    def test_stream_uses_message_timeout_over_timeout(self):
        client = LlamaCpp(timeout=60.0, message_timeout=120.0)
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
            ]
        )
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_stream({"model": "test"}, callback)
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 120.0

    def test_stream_explicit_timeout_overrides_message_timeout(self):
        client = LlamaCpp(timeout=60.0, message_timeout=120.0)
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
            ]
        )
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_stream({"model": "test"}, callback, timeout=30.0)
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 30.0

    def test_stream_falls_back_to_timeout_when_no_message_timeout(self):
        client = LlamaCpp(timeout=60.0, message_timeout=None)
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
            ]
        )
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_stream({"model": "test"}, callback)
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 60.0


class TestHandleStreamError:
    def test_handle_stream_error_retryable(self):
        client = LlamaCpp()
        msg = {"error": {"code": 500}}
        with patch("time.sleep"):
            should_retry, retries = client._handle_stream_error(msg, 0, 3)
        assert should_retry is True
        assert retries == 1

    def test_handle_stream_error_non_retryable(self):
        client = LlamaCpp()
        msg = {"error": {"code": 400}}
        with pytest.raises(RuntimeError, match="LLM error"):
            client._handle_stream_error(msg, 0, 3)

    def test_handle_stream_error_max_retries_exceeded(self):
        client = LlamaCpp()
        msg = {"error": {"code": 500}}
        with (
            patch("time.sleep"),
            pytest.raises(RuntimeError, match="LLM error after 3 retries"),
        ):
            client._handle_stream_error(msg, 3, 3)

    def test_handle_stream_error_no_error(self):
        client = LlamaCpp()
        msg = {"choices": []}
        should_retry, retries = client._handle_stream_error(msg, 0, 3)
        assert should_retry is False
        assert retries == 0


class TestStreamHooks:
    def test_prepare_stream_data_default(self):
        client = LlamaCpp()
        data = {"model": "test"}
        # Call parent's _prepare_stream_data via super - LlamaCpp overrides it
        # so test the LlamaCpp version
        client._prepare_stream_data(data)
        assert len(data) == 2
        assert data["model"] == "test"
        assert data["return_progress"] is True

    def test_process_stream_delta_default(self):
        client = LlamaCpp()
        delta = {"content": "hi"}
        result = client._process_stream_delta(delta)
        assert result == delta

    def test_build_stream_callback_default(self):
        client = LlamaCpp()
        result = client._build_stream_callback({"content": "hi"}, "content", {})
        assert result["delta"] == {"content": "hi"}
        assert result["delta_type"] == "content"
        assert result["prompt_progress"] is None
        assert len(result) == 3

    def test_build_stream_result_default(self):
        client = LlamaCpp()
        msg = {
            "timings": {
                "prompt_n": 1,
                "predicted_n": 2,
                "prompt_ms": 3000,
                "predicted_ms": 5000,
            }
        }
        result = client._build_stream_result(
            "content", "reasoning", msg, {"model": "m"}
        )
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "content",
                        "reasoning_content": "reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "cached_tokens": 0,
                "cache_write_tokens": 1,
            },
            "timings": {"prompt_time": 3.0, "completion_time": 5.0},
            "model": "m",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }

    def test_should_stop_streaming_default(self):
        client = LlamaCpp()
        assert client._should_stop_streaming({}) is False

    def test_on_stream_update_default(self):
        client = LlamaCpp()
        assert client._on_stream_update({}) is None

    def test_on_stream_delta_default(self):
        client = LlamaCpp()
        assert client._on_stream_delta({}, "content") is None


# ===========================================================================
# LlamaCpp tests
# ===========================================================================


class TestLlamaCppInit:
    def test_init_default(self):
        client = LlamaCpp()
        assert client.base_url == "http://127.0.0.1:8080"
        assert (
            client.url_token_count == "http://127.0.0.1:8080/v1/messages/count_tokens"
        )

    def test_init_custom(self):
        client = LlamaCpp(host="http://custom:9090", timeout=30, verify_ssl=False)
        assert client.base_url == "http://custom:9090"
        assert client.timeout == 30
        assert client.verify_ssl is False

    def test_init_ca_cert(self):
        client = LlamaCpp(
            host="http://127.0.0.1:8080",
            ca_cert="/path/to/ca.pem",
        )
        assert client.ca_cert == "/path/to/ca.pem"
        assert client._verify == "/path/to/ca.pem"

    def test_verify_property_without_ca_cert(self):
        client = LlamaCpp(verify_ssl=False)
        assert client._verify is False


class TestLlamaCppPrepareStreamData:
    def test_prepare_stream_data(self):
        client = LlamaCpp()
        data = {"model": "test"}
        client._prepare_stream_data(data)
        assert data["return_progress"] is True
        assert data["model"] == "test"
        assert len(data) == 2


class TestLlamaCppProcessStreamDelta:
    def test_process_stream_delta_converts_reasoning_content(self):
        client = LlamaCpp()
        delta = {"reasoning_content": "thinking", "content": "hi"}
        result = client._process_stream_delta(delta)
        assert result["reasoning"] == "thinking"
        assert "reasoning_content" not in result
        assert result["content"] == "hi"
        assert len(result) == 2

    def test_process_stream_delta_no_change(self):
        client = LlamaCpp()
        delta = {"content": "hi"}
        result = client._process_stream_delta(delta)
        assert result == delta


class TestLlamaCppBuildStreamCallback:
    def test_build_stream_callback_includes_prompt_progress(self):
        client = LlamaCpp()
        msg = {"prompt_progress": {"processed": 50, "total": 100}}
        result = client._build_stream_callback({"content": "hi"}, "content", msg)
        assert result["prompt_progress"] == {"processed": 50, "total": 100}
        assert result["delta"] == {"content": "hi"}
        assert result["delta_type"] == "content"
        assert len(result) == 3

    def test_build_stream_callback_no_progress(self):
        client = LlamaCpp()
        msg = {}
        result = client._build_stream_callback({"content": "hi"}, "content", msg)
        assert result["prompt_progress"] is None
        assert result["delta"] == {"content": "hi"}
        assert result["delta_type"] == "content"
        assert len(result) == 3


class TestLlamaCppBuildStreamResult:
    def test_build_stream_result(self):
        client = LlamaCpp()
        msg = {
            "timings": {
                "prompt_n": 10,
                "predicted_n": 20,
                "prompt_ms": 100.0,
                "predicted_ms": 400.0,
            }
        }
        data = {"model": "llama-model"}
        result = client._build_stream_result("content", "reasoning", msg, data)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "content",
                        "reasoning_content": "reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 0,
                "cache_write_tokens": 10,
            },
            "timings": {"prompt_time": 0.1, "completion_time": 0.4},
            "model": "llama-model",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }

    def test_build_stream_result_empty_timings(self):
        client = LlamaCpp()
        msg = {"timings": {}}
        data = {"model": "llama-model"}
        result = client._build_stream_result("content", "reasoning", msg, data)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "content",
                        "reasoning_content": "reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "llama-model",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }


class TestLlamaCppGetContextSize:
    def test_get_context_size_success(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "default_generation_settings": {"n_ctx": 4096},
            "model": "test-model",
        }
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            size = client.get_context_size("test-model")
        assert size == 4096

    def test_get_context_size_no_settings(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"default_generation_settings": {}}
        mock_resp.headers = {"Content-Type": "application/json"}
        with (
            patch("tiz.inference_clients.requests.get", return_value=mock_resp),
            pytest.raises(ValueError, match="no data"),
        ):
            client.get_context_size("test-model")

    def test_get_context_size_model_mismatch_warning(self, caplog):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "default_generation_settings": {"n_ctx": 4096},
            "model": "different-model",
        }
        mock_resp.headers = {"Content-Type": "application/json"}
        with (
            patch("tiz.inference_clients.requests.get", return_value=mock_resp),
            caplog.at_level("WARNING"),
        ):
            client.get_context_size("requested-model")
        assert (
            "Requested model 'requested-model' but server has 'different-model'"
            in caplog.text
        )


class TestLlamaCppGetCredits:
    def test_get_credits_returns_zero(self):
        client = LlamaCpp()
        credits = client.get_credits()
        assert credits == {"total_credits": 0.0, "total_usage": 0.0}


class TestLlamaCppIsUp:
    def test_is_up_success(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.is_up() is True

    def test_is_up_failure_status(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "error"}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.is_up() is False

    def test_is_up_exception(self):
        client = LlamaCpp()
        with patch(
            "tiz.inference_clients.requests.get",
            side_effect=Exception("connection failed"),
        ):
            assert client.is_up() is False


class TestLlamaCppCountTokens:
    def test_count_tokens_success(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"input_tokens": 42}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            count = client.count_tokens(
                [{"role": "user", "content": "hello"}], _model="test"
            )
        assert count == 42
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs == {
            "json": {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "test",
            },
            "headers": {"Content-Type": "application/json", "User-Agent": USER_AGENT},
            "timeout": 5,
            "verify": True,
        }


class TestLlamaCppToolsSupport:
    def test_tools_support_true(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "chat_template_caps": {
                "supports_tools": True,
                "supports_tool_calls": True,
            }
        }
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.tools_support() is True

    def test_tools_support_false_missing_caps(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.headers = {"Content-Type": "application/json"}
        with (
            patch("tiz.inference_clients.requests.get", return_value=mock_resp),
            pytest.raises(ValueError, match="Missing.*chat_template_caps"),
        ):
            client.tools_support()

    def test_tools_support_false_no_tools(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "chat_template_caps": {"supports_tools": False, "supports_tool_calls": True}
        }
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.tools_support() is False

    def test_tools_support_false_no_tool_calls(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "chat_template_caps": {"supports_tools": True, "supports_tool_calls": False}
        }
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.tools_support() is False


# ===========================================================================
# DwarfStar4 tests
# ===========================================================================


class TestDwarfStar4Init:
    def test_init_default(self):
        client = DwarfStar4()
        assert client.base_url == "http://127.0.0.1:8080"
        assert client._default_model == ""
        assert client.timeout == 5.0
        assert client.verify_ssl is True
        assert client.headers == {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def test_init_custom_host_and_timeout(self):
        client = DwarfStar4(host="http://custom:9090", timeout=30, verify_ssl=False)
        assert client.base_url == "http://custom:9090"
        assert client.timeout == 30
        assert client.verify_ssl is False

    def test_init_with_api_key(self):
        client = DwarfStar4(api_key="sk-test-key")
        assert client.headers["Authorization"] == "Bearer sk-test-key"

    def test_init_with_ca_cert(self):
        client = DwarfStar4(ca_cert="/path/to/ca.pem")
        assert client._verify == "/path/to/ca.pem"

    def test_init_default_model_empty(self):
        client = DwarfStar4()
        assert client._default_model == ""

    def test_init_custom_default_model(self):
        client = DwarfStar4(default_model="my-model")
        assert client._default_model == "my-model"

    def test_init_custom_message_timeout(self):
        client = DwarfStar4(message_timeout=300.0)
        assert client.message_timeout == 300.0

    def test_init_sampling_params(self):
        client = DwarfStar4(sampling_params={"temperature": 0.5})
        assert client.sampling_params == {"temperature": 0.5}

    def test_init_preserve_thinking(self):
        client = DwarfStar4()
        assert client.preserve_thinking is True


class TestDwarfStar4GetContextSize:
    def test_get_context_size_success(self):
        client = DwarfStar4()
        mock_resp = _make_mock_response({"data": [{"id": "model1"}, {"id": "model2"}]})
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            size = client.get_context_size("model1")
        assert size == 1000000

    def test_get_context_size_model_not_found(self):
        client = DwarfStar4()
        mock_resp = _make_mock_response({"data": [{"id": "model1"}]})
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            size = client.get_context_size("nonexistent")
        assert size == 1000000

    def test_get_context_size_empty_default(self):
        client = DwarfStar4()
        mock_resp = _make_mock_response({"data": [{"id": "model1"}]})
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            size = client.get_context_size()
        assert size == 1000000

    def test_get_context_size_resolved_via_default(self):
        client = DwarfStar4(default_model="model2")
        mock_resp = _make_mock_response({"data": [{"id": "model1"}, {"id": "model2"}]})
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            size = client.get_context_size()
        assert size == 1000000


class TestDwarfStar4IsUp:
    def test_is_up_success(self):
        client = DwarfStar4()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.is_up() is True

    def test_is_up_failure(self):
        client = DwarfStar4()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.is_up() is False

    def test_is_up_exception(self):
        client = DwarfStar4()
        with patch(
            "tiz.inference_clients.requests.get",
            side_effect=Exception("connection failed"),
        ):
            assert client.is_up() is False


class TestDwarfStar4GetCredits:
    def test_get_credits_returns_zero(self):
        client = DwarfStar4()
        credits = client.get_credits()
        assert credits == {"total_credits": 0.0, "total_usage": 0.0}


class TestDwarfStar4CountTokens:
    def test_count_tokens_heuristic(self):
        client = DwarfStar4()
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = client.count_tokens(messages)
        assert count == 16

    def test_count_tokens_empty_content(self):
        client = DwarfStar4()
        count = client.count_tokens([{"role": "user", "content": ""}])
        assert count == 5

    def test_count_tokens_missing_content(self):
        client = DwarfStar4()
        count = client.count_tokens([{"role": "user"}])
        assert count == 5

    def test_count_tokens_empty_messages(self):
        client = DwarfStar4()
        count = client.count_tokens([])
        assert count == 1

    def test_count_tokens_with_preserve_thinking(self):
        client = DwarfStar4()
        client._preserve_thinking = True
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = client.count_tokens(messages)
        assert count == 16

    def test_count_tokens_with_reasoning_content_preserved(self):
        client = DwarfStar4()
        client._preserve_thinking = True
        messages = [
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Let me think...",
            },
        ]
        count = client.count_tokens(messages)
        total_chars = len("Final answer") + len("Let me think...")
        assert count == max(1, total_chars // 3 + 1 * 5)

    def test_count_tokens_reasoning_content_not_counted_when_not_preserved(self):
        client = DwarfStar4()
        client._preserve_thinking = False
        messages = [
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Let me think...",
            },
        ]
        count = client.count_tokens(messages)
        total_chars = len("Final answer")
        assert count == max(1, total_chars // 3 + 1 * 5)


class TestDwarfStar4ToolsSupport:
    def test_tools_support_always_true(self):
        client = DwarfStar4()
        assert client.tools_support() is True

    def test_tools_support_with_model_arg(self):
        client = DwarfStar4()
        assert client.tools_support("any-model") is True


class TestDwarfStar4InputModes:
    def test_input_modes_always_text(self):
        client = DwarfStar4()
        assert client.input_modes() == ["text"]

    def test_input_modes_with_model_arg(self):
        client = DwarfStar4()
        assert client.input_modes("any-model") == ["text"]


class TestDwarfStar4OutputModes:
    def test_output_modes_always_text(self):
        client = DwarfStar4()
        assert client.output_modes() == ["text"]

    def test_output_modes_with_model_arg(self):
        client = DwarfStar4()
        assert client.output_modes("any-model") == ["text"]


class TestDwarfStar4PrepareStreamData:
    def test_prepare_stream_data_adds_stream_options(self):
        client = DwarfStar4()
        data = {"model": "test-model"}
        client._prepare_stream_data(data)
        assert data["stream_options"] == {"include_usage": True}
        assert data["model"] == "test-model"
        assert len(data) == 2

    def test_prepare_stream_data_with_existing_fields(self):
        client = DwarfStar4()
        data = {"model": "test-model", "temperature": 0.7, "max_tokens": 100}
        client._prepare_stream_data(data)
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 100
        assert data["stream_options"] == {"include_usage": True}

    def test_prepare_stream_data_no_model(self):
        client = DwarfStar4()
        data = {}
        client._prepare_stream_data(data)
        assert data["stream_options"] == {"include_usage": True}
        assert len(data) == 1

    def test_prepare_stream_data_does_not_override(self):
        client = DwarfStar4()
        data = {"stream_options": {"include_usage": False}}
        client._prepare_stream_data(data)
        assert data["stream_options"] == {"include_usage": True}


class TestDwarfStar4ProcessStreamDelta:
    def test_process_stream_delta_converts_reasoning_content(self):
        client = DwarfStar4()
        delta = {"reasoning_content": "thinking", "content": "hi"}
        result = client._process_stream_delta(delta)
        assert result["reasoning"] == "thinking"
        assert "reasoning_content" not in result
        assert result["content"] == "hi"
        assert len(result) == 2

    def test_process_stream_delta_no_change(self):
        client = DwarfStar4()
        delta = {"content": "hi"}
        result = client._process_stream_delta(delta)
        assert result == delta

    def test_process_stream_delta_only_reasoning_content(self):
        client = DwarfStar4()
        delta = {"reasoning_content": "thinking"}
        result = client._process_stream_delta(delta)
        assert result["reasoning"] == "thinking"
        assert "reasoning_content" not in result
        assert len(result) == 1


class TestDwarfStar4Chat:
    def test_chat_non_stream_does_not_include_stream_options(self):
        client = DwarfStar4()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            result = client.chat(
                [{"role": "user", "content": "hello"}],
                model="test",
                stream=False,
            )
        assert result == {"choices": [{"message": {"content": "hi"}}]}
        call_kwargs = mock_post.call_args[1]
        json_data = call_kwargs["json"]
        assert "stream_options" not in json_data

    def test_chat_stream_includes_stream_options(self):
        client = DwarfStar4()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "usage": {}}',
            ]
        )
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            result = client.chat(
                [{"role": "user", "content": "hello"}],
                model="test",
                stream=True,
                update_callback=callback,
            )
        assert len(callback_calls) == 1
        assert callback_calls[0] == {
            "delta": {"content": "hi"},
            "delta_type": "content",
        }
        assert result["choices"] == [
            {
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "",
                }
            }
        ]
        assert result["model"] == "test"
        assert result["provider"] == "DwarfStar4/http://127.0.0.1:8080"
        call_kwargs = mock_post.call_args[1]
        json_data = call_kwargs["json"]
        assert json_data.get("stream_options") == {"include_usage": True}

    def test_chat_with_default_model(self):
        client = DwarfStar4(default_model="my-model")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}])
        assert result == {"choices": [{"message": {"content": "ok"}}]}
        call_kwargs = mock_post.call_args[1]
        json_data = call_kwargs["json"]
        assert "my-model" in json_data.get("model", "")
        assert "stream_options" not in json_data


# ===========================================================================
# OpenRouter tests
# ===========================================================================


class TestOpenRouterInit:
    def test_init_with_api_key(self):
        client = OpenRouter(api_key="test-key")
        assert client.base_url == OpenRouter.BASE_URL
        assert client.headers["Authorization"] == "Bearer test-key"
        assert client.headers["HTTP-Referer"] == "https://github.com/smeso/tiz"
        assert client.headers["X-Title"] == "tiz"

    def test_init_no_key_raises(self):
        with pytest.raises(TypeError, match="missing 1 required positional"):
            OpenRouter()  # type: ignore[call-arg]

    def test_init_default_model(self):
        client = OpenRouter(api_key="test-key")
        assert client._default_model == "openrouter/free"

    def test_init_drop_reporting_when_env_is_1(self):
        with patch.dict(os.environ, {"TIZ_DROP_OPENROUTER_REPORTING": "1"}, clear=True):
            client = OpenRouter(api_key="test-key")
        assert "HTTP-Referer" not in client.headers
        assert "X-Title" not in client.headers
        assert "X-OpenRouter-Categories" not in client.headers
        assert client.headers["Authorization"] == "Bearer test-key"

    def test_init_drop_reporting_when_env_is_not_1(self):
        for val in ("0", "true", "", "yes"):
            with patch.dict(
                os.environ, {"TIZ_DROP_OPENROUTER_REPORTING": val}, clear=True
            ):
                client = OpenRouter(api_key="test-key")
            assert client.headers.get("HTTP-Referer") == "https://github.com/smeso/tiz"
            assert client.headers.get("X-Title") == "tiz"

    def test_init_drop_reporting_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            client = OpenRouter(api_key="test-key")
        assert client.headers.get("HTTP-Referer") == "https://github.com/smeso/tiz"
        assert client.headers.get("X-Title") == "tiz"


class TestOpenRouterGetCachedModels:
    def test_get_cached_models_fetches_and_caches(self):
        client = OpenRouter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "model1"}]}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            models = client._get_cached_models()
        assert models == [{"id": "model1"}]
        assert client._models_cache == [{"id": "model1"}]

    def test_get_cached_models_uses_cache(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "cached"}]
        client._models_cache_time = time.monotonic()
        with patch("tiz.inference_clients.requests.get") as mock_get:
            models = client._get_cached_models()
        assert models == [{"id": "cached"}]
        mock_get.assert_not_called()

    def test_get_cached_models_cache_expired(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "old"}]
        client._models_cache_time = time.monotonic() - 7200
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "new"}]}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            models = client._get_cached_models()
        assert models == [{"id": "new"}]


class TestOpenRouterGetContextSize:
    def test_get_context_size_success(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "model1", "context_length": 8192}]
        client._models_cache_time = time.monotonic()
        size = client.get_context_size("model1")
        assert size == 8192

    def test_get_context_size_model_not_found(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "model1", "context_length": 8192}]
        client._models_cache_time = time.monotonic()
        with pytest.raises(ValueError, match="not found"):
            client.get_context_size("model2")

    def test_get_context_size_missing_context_length(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "model1"}]
        client._models_cache_time = time.monotonic()
        with pytest.raises(ValueError, match="Could not determine context size"):
            client.get_context_size("model1")


class TestOpenRouterIsUp:
    def test_is_up_success(self):
        client = OpenRouter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.is_up() is True

    def test_is_up_failure(self):
        client = OpenRouter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.is_up() is False

    def test_is_up_exception(self):
        client = OpenRouter(api_key="test-key")
        with patch("tiz.inference_clients.requests.get", side_effect=Exception("fail")):
            assert client.is_up() is False


class TestOpenRouterGetCredits:
    def test_get_credits_success(self):
        client = OpenRouter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"total_credits": 100.0, "total_usage": 25.5}
        }
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            credits = client.get_credits()
        assert credits == {"total_credits": 100.0, "total_usage": 25.5}

    def test_get_credits_defaults(self):
        client = OpenRouter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            credits = client.get_credits()
        assert credits == {"total_credits": 0.0, "total_usage": 0.0}


class TestOpenRouterCountTokens:
    def test_count_tokens_heuristic(self):
        client = OpenRouter(api_key="test-key")
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = client.count_tokens(messages)
        assert count == 12

    def test_count_tokens_empty_content(self):
        client = OpenRouter(api_key="test-key")
        messages = [{"role": "user", "content": ""}]
        count = client.count_tokens(messages)
        assert count == 4

    def test_count_tokens_missing_content(self):
        client = OpenRouter(api_key="test-key")
        messages = [{"role": "user"}]
        count = client.count_tokens(messages)
        assert count == 4

    def test_count_tokens_with_reasoning_content_preserved(self):
        client = OpenRouter(api_key="test-key")
        client._preserve_thinking = True
        messages = [
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Let me think...",
            },
        ]
        count = client.count_tokens(messages)
        total_chars = len("Final answer") + len("Let me think...")
        assert count == max(1, total_chars // 4 + 1 * 4)

    def test_count_tokens_reasoning_content_not_counted_when_not_preserved(self):
        client = OpenRouter(api_key="test-key")
        client._preserve_thinking = False
        messages = [
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Let me think...",
            },
        ]
        count = client.count_tokens(messages)
        total_chars = len("Final answer")
        assert count == max(1, total_chars // 4 + 1 * 4)


class TestOpenRouterToolsSupport:
    def test_tools_support_true(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [
            {"id": "model1", "supported_parameters": ["temperature", "tools"]}
        ]
        client._models_cache_time = time.monotonic()
        assert client.tools_support("model1") is True

    def test_tools_support_false(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [
            {"id": "model1", "supported_parameters": ["temperature"]}
        ]
        client._models_cache_time = time.monotonic()
        assert client.tools_support("model1") is False

    def test_tools_support_model_not_found(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "model1"}]
        client._models_cache_time = time.monotonic()
        assert client.tools_support("model2") is False


class TestOpenRouterShouldStopStreaming:
    def test_should_stop_streaming_done_key(self):
        client = OpenRouter(api_key="test-key")
        assert client._should_stop_streaming({"done": True}) is True

    def test_should_stop_streaming_not_done(self):
        client = OpenRouter(api_key="test-key")
        assert client._should_stop_streaming({"done": False}) is False

    def test_should_stop_streaming_no_done_key(self):
        client = OpenRouter(api_key="test-key")
        assert client._should_stop_streaming({}) is False


class TestOpenRouterOnStreamDelta:
    def test_on_stream_delta_first_call(self):
        client = OpenRouter(api_key="test-key")
        client._first_update_time = None
        client._last_update_time = None
        client._on_stream_delta({"content": "hi"}, "content")
        assert client._first_update_time is not None
        assert client._last_update_time is not None

    def test_on_stream_delta_subsequent_call(self):
        client = OpenRouter(api_key="test-key")
        t1 = time.monotonic() - 10
        t2 = time.monotonic() - 5
        client._first_update_time = t1
        client._last_update_time = t2
        client._on_stream_delta({"content": "hi"}, "content")
        assert client._first_update_time == t1
        assert client._last_update_time >= t2

    def test_on_stream_delta_empty_type(self):
        client = OpenRouter(api_key="test-key")
        client._first_update_time = None
        client._on_stream_delta({"content": "hi"}, "")
        assert client._first_update_time is None


class TestBackoffDelay:
    def test_backoff_delay_first_retry(self):
        delay = LlamaCpp._backoff_delay(1)
        assert 2 <= delay < 3

    def test_backoff_delay_clamped_at_60(self):
        delay = LlamaCpp._backoff_delay(10)
        assert 60 <= delay < 61

    def test_backoff_delay_zero_retries(self):
        delay = LlamaCpp._backoff_delay(0)
        assert 1 <= delay < 2


class TestOpenRouterBuildStreamResult:
    def test_build_stream_result(self):
        client = OpenRouter(api_key="test-key")
        client._first_update_time = 100.0
        client._last_update_time = 101.0
        client._request_start_time = 98.0
        msg = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cost": 0.001,
                "cost_details": 0.0005,
            },
            "model": "or-model",
            "provider": "nonex",
        }
        data = {"model": "fallback-model"}
        result = client._build_stream_result("content", "reasoning", msg, data)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "content",
                        "reasoning_content": "reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0.001,
                "cost_details": 0.0005,
            },
            "timings": {"prompt_time": 2.0, "completion_time": 1.0},
            "model": "or-model",
            "provider": "OpenRouter/nonex",
        }

    def test_build_stream_result_null_usage_fields(self):
        client = OpenRouter(api_key="test-key")
        client._first_update_time = None
        client._last_update_time = None
        client._request_start_time = 100.0
        with patch("tiz.inference_clients.time.monotonic", return_value=100.0):
            msg = {
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "cost": None,
                    "cost_details": None,
                }
            }
            data = {"model": "m"}
            result = client._build_stream_result("content", "reasoning", msg, data)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "content",
                        "reasoning_content": "reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0,
                "cost_details": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "m",
            "provider": "OpenRouter/Unknown",
        }

    def test_build_stream_result_zero_time(self):
        client = OpenRouter(api_key="test-key")
        now = 100.0
        client._first_update_time = now
        client._last_update_time = now
        client._request_start_time = now
        msg = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        data = {"model": "m"}
        result = client._build_stream_result("content", "reasoning", msg, data)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "content",
                        "reasoning_content": "reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0,
                "cost_details": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "m",
            "provider": "OpenRouter/Unknown",
        }


class TestOpenRouterChatStream:
    def test_chat_stream_resets_timings(self):
        client = OpenRouter(api_key="test-key")
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = _make_stream_mock(
            [
                b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
                b'data: {"choices": [{"finish_reason": "stop"}], "usage": {}}',
            ]
        )
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert client._first_update_time is not None
        assert callback_calls == [{"delta": {"content": "hi"}, "delta_type": "content"}]
        assert result["choices"] == [
            {
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "",
                }
            }
        ]
        assert result["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "cost": 0,
            "cost_details": 0,
        }
        assert result["model"] == "test"
        assert result["provider"] == "OpenRouter/Unknown"
        assert result["timings"]["prompt_time"] >= 0
        assert result["timings"]["completion_time"] >= 0


class TestOpenRouterChat:
    def test_chat_delegates_to_parent(self):
        client = OpenRouter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "hello"}],
                sampling_params={"temperature": 0.5},
                stream=False,
                model="test-model",
            )
        assert result == {"choices": [{"message": {"content": "hi"}}]}


# ===========================================================================
# Additional coverage tests
# ===========================================================================


class TestChatNonStreamAdditional:
    def test_chat_non_stream_http_error_no_response(self):
        client = LlamaCpp()
        response_object = requests.Response()
        response_object.status_code = 500
        http_error = requests.exceptions.HTTPError(
            "Fake error", response=response_object
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[http_error, mock_resp],
            ) as mock_post,
            patch("time.sleep"),
        ):
            result = client._chat_non_stream({"messages": []}, max_retries=3)
        assert result == {"choices": []}
        assert mock_post.call_count == 2

    def test_chat_non_stream_connection_error_retry(self):
        client = LlamaCpp()
        conn_error = requests.exceptions.ConnectionError()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[conn_error, mock_resp],
            ) as mock_post,
            patch("time.sleep"),
        ):
            result = client._chat_non_stream({"messages": []}, max_retries=3)
        assert result == {"choices": []}
        assert mock_post.call_count == 2


class TestOpenAICompatibleBuildStreamResult:
    def test_build_stream_result_base_class(self):
        from tiz.inference_clients import OpenAICompatibleClient

        class TestClient(OpenAICompatibleClient):
            def get_models(self):
                return []

            def count_tokens(self, messages, _model=""):
                return 0

            def tools_support(self, model=""):
                return False

            def chat(
                self,
                messages,
                sampling_params=None,
                stream=False,
                update_callback=None,
                model="",
                max_retries=3,
            ):
                return {}

            def get_context_size(self, model=""):
                return 0

            def get_credits(self):
                return {}

            def is_up(self):
                return True

            def input_modes(self, model=""):
                return ["text"]

            def output_modes(self, model=""):
                return ["text"]

        client = TestClient(base_url="http://test")
        result = client._build_stream_result("content", "reasoning", {}, {"model": "m"})
        assert result["choices"] == [
            {
                "message": {
                    "role": "assistant",
                    "content": "content",
                    "reasoning_content": "reasoning",
                }
            }
        ]
        assert result["model"] == "m"
        assert result["provider"] == "http://test"
        assert result["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }
        assert "timings" in result

    def test_build_stream_result_base_class_with_none_details(self):
        from tiz.inference_clients import OpenAICompatibleClient

        class TestClient(OpenAICompatibleClient):
            def get_models(self):
                return []

            def count_tokens(self, messages, _model=""):
                return 0

            def tools_support(self, model=""):
                return False

            def chat(
                self,
                messages,
                sampling_params=None,
                stream=False,
                update_callback=None,
                model="",
                max_retries=3,
            ):
                return {}

            def get_context_size(self, model=""):
                return 0

            def get_credits(self):
                return {}

            def is_up(self):
                return True

            def input_modes(self, model=""):
                return ["text"]

            def output_modes(self, model=""):
                return ["text"]

        client = TestClient(base_url="http://test")
        msg = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "prompt_tokens_details": {
                    "cached_tokens": None,
                    "cache_write_tokens": None,
                },
            }
        }
        result = client._build_stream_result(
            "content", "reasoning", msg, {"model": "m"}
        )
        assert result["usage"]["cached_tokens"] == 0
        assert result["usage"]["cache_write_tokens"] == 0


class TestChatStreamAdditional:
    def test_stream_retry_request_continue(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        first_resp = MagicMock()
        first_resp.iter_lines.return_value = [
            b'data: {"error": {"code": 500}}',
        ]
        first_resp.headers = {"Content-Type": "application/json"}
        first_resp.json.return_value = {}
        first_resp.__enter__ = MagicMock(return_value=first_resp)
        first_resp.__exit__ = MagicMock(return_value=False)

        second_resp = MagicMock()
        second_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
        ]
        second_resp.headers = {"Content-Type": "application/json"}
        second_resp.json.return_value = {}
        second_resp.__enter__ = MagicMock(return_value=second_resp)
        second_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[first_resp, second_resp],
            ) as _,
            patch("time.sleep"),
        ):
            result = client._chat_stream({"model": "test"}, callback, max_retries=3)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "test",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }

    def test_stream_done_sentinel_via_iter_lines(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b"data: [DONE]",
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert len(callback_calls) == 1
        assert callback_calls[0] == {
            "delta": {"content": "hi"},
            "delta_type": "content",
            "prompt_progress": None,
        }
        assert result["choices"][0]["message"]["content"] == "hi"
        assert result["model"] == "test"
        assert result["provider"] == "LlamaCpp/http://127.0.0.1:8080"

    def test_stream_finish_reason_with_timings(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "test",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }

    def test_stream_break_on_processing_inactive(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b'data: {"choices": [{"finish_reason": "stop"}], "timings": {"prompt_n": 1}}',
            b'data: {"choices": [{"delta": {"content": "should-not-process"}, "finish_reason": null}]}',
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["content"] == "hi"
        assert len(callback_calls) == 1

    def test_stream_finish_reason_no_usage_or_timings(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b'data: {"choices": [{"finish_reason": "stop"}]}',
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "test",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }

    def test_stream_chunked_encoding_error_retry_succeeds(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        first_resp = MagicMock()
        first_resp.iter_lines.side_effect = requests.exceptions.ChunkedEncodingError(
            "connection closed"
        )
        first_resp.headers = {"Content-Type": "application/json"}
        first_resp.json.return_value = {}
        first_resp.__enter__ = MagicMock(return_value=first_resp)
        first_resp.__exit__ = MagicMock(return_value=False)

        second_resp = MagicMock()
        second_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b'data: {"choices": [{"finish_reason": "stop"}], "timings": {}}',
        ]
        second_resp.headers = {"Content-Type": "application/json"}
        second_resp.json.return_value = {}
        second_resp.__enter__ = MagicMock(return_value=second_resp)
        second_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[first_resp, second_resp],
            ) as mock_post,
            patch("time.sleep"),
        ):
            result = client._chat_stream({"model": "test"}, callback, max_retries=3)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0.0, "completion_time": 0.0},
            "model": "test",
            "provider": "LlamaCpp/http://127.0.0.1:8080",
        }
        assert mock_post.call_count == 2

    def test_stream_chunked_encoding_error_exhausted_retries(self):
        client = LlamaCpp()
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        fail_resp = MagicMock()
        fail_resp.iter_lines.side_effect = requests.exceptions.ChunkedEncodingError(
            "connection closed"
        )
        fail_resp.headers = {"Content-Type": "application/json"}
        fail_resp.json.return_value = {}
        fail_resp.__enter__ = MagicMock(return_value=fail_resp)
        fail_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[fail_resp] * 5,
            ),
            patch("time.sleep"),
            pytest.raises(RuntimeError, match="Stream failed after 3 retries"),
        ):
            client._chat_stream({"model": "test"}, callback, max_retries=3)


class TestOpenRouterAdditional:
    def test_build_stream_result_no_model_in_msg(self):
        client = OpenRouter(api_key="test-key")
        client._first_update_time = 100.0
        client._last_update_time = 102.0
        client._request_start_time = 99.0
        msg = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "provider": "nonex",
        }
        data = {"model": "fallback-model"}
        result = client._build_stream_result("content", "reasoning", msg, data)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "content",
                        "reasoning_content": "reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0,
                "cost_details": 0,
            },
            "timings": {"prompt_time": 1.0, "completion_time": 2.0},
            "model": "fallback-model",
            "provider": "OpenRouter/nonex",
        }

    def test_build_stream_result_no_provider_in_msg(self):
        client = OpenRouter(api_key="test-key")
        client._first_update_time = 100.0
        client._last_update_time = 101.0
        client._request_start_time = 98.0
        msg = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model": "or-model",
        }
        data = {"model": "fallback-model"}
        result = client._build_stream_result("content", "reasoning", msg, data)
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "content",
                        "reasoning_content": "reasoning",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0,
                "cost_details": 0,
            },
            "timings": {"prompt_time": 2.0, "completion_time": 1.0},
            "model": "or-model",
            "provider": "OpenRouter/Unknown",
        }

    def test_chat_with_stream_true_and_no_callback(self):
        client = OpenRouter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b'data: {"choices": [{"finish_reason": "stop"}], "usage": {}}',
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "hello"}],
                stream=True,
                model="test-model",
            )
        assert result["choices"] == [
            {
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "",
                }
            }
        ]
        assert result["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "cost": 0,
            "cost_details": 0,
        }
        assert result["model"] == "test-model"
        assert result["provider"] == "OpenRouter/Unknown"
        assert result["timings"]["prompt_time"] >= 0
        assert result["timings"]["completion_time"] >= 0

    def test_get_cached_models_empty_data(self):
        client = OpenRouter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            models = client._get_cached_models()
        assert models == []
        assert client._models_cache == []

    def test_count_tokens_with_zero_messages(self):
        client = OpenRouter(api_key="test-key")
        count = client.count_tokens([])
        assert count == 1

    def test_tools_support_with_empty_cache(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = []
        client._models_cache_time = time.monotonic()
        assert client.tools_support("model1") is False


class TestOpenAICompatibleStreamHooks:
    def test_stream_done_detected_via_custom_check(self):
        from tiz.inference_clients import OpenAICompatibleClient

        class TestClient(OpenAICompatibleClient):
            def get_models(self):
                return []

            def count_tokens(self, messages, _model=""):
                return 0

            def tools_support(self, model=""):
                return False

            def chat(
                self,
                messages,
                sampling_params=None,
                stream=False,
                update_callback=None,
                model="",
                max_retries=3,
            ):
                return {}

            def get_context_size(self, model=""):
                return 0

            def get_credits(self):
                return {}

            def is_up(self):
                return True

            def input_modes(self, model=""):
                return ["text"]

            def output_modes(self, model=""):
                return ["text"]

            def _should_stop_streaming(self, msg):
                return bool(msg.get("custom_stop"))

        client = TestClient(base_url="http://test")
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b'data: {"custom_stop": true}',
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"] == [
            {
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "",
                }
            }
        ]
        assert result["model"] == "test"
        assert result["provider"] == "http://test"
        assert "usage" in result
        assert "timings" in result
        assert callback_calls == [{"delta": {"content": "hi"}, "delta_type": "content"}]

    def test_stream_done_detected_via_parsed_msg(self):
        from tiz.inference_clients import OpenAICompatibleClient

        class TestClient(OpenAICompatibleClient):
            def get_models(self):
                return []

            def count_tokens(self, messages, _model=""):
                return 0

            def tools_support(self, model=""):
                return False

            def chat(
                self,
                messages,
                sampling_params=None,
                stream=False,
                update_callback=None,
                model="",
                max_retries=3,
            ):
                return {}

            def get_context_size(self, model=""):
                return 0

            def get_credits(self):
                return {}

            def is_up(self):
                return True

            def input_modes(self, model=""):
                return ["text"]

            def output_modes(self, model=""):
                return ["text"]

            def _should_stop_streaming(self, msg):
                return bool(msg.get("done"))

        client = TestClient(base_url="http://test")
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}',
            b'data: {"done": true}',
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"] == [
            {
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "",
                }
            }
        ]
        assert result["model"] == "test"
        assert result["provider"] == "http://test"
        assert "usage" in result
        assert "timings" in result
        assert callback_calls == [{"delta": {"content": "hi"}, "delta_type": "content"}]

    def test_stream_done_detected_returns_result(self):
        from tiz.inference_clients import OpenAICompatibleClient

        class TestClient(OpenAICompatibleClient):
            def get_models(self):
                return []

            def count_tokens(self, messages, _model=""):
                return 0

            def tools_support(self, model=""):
                return False

            def chat(
                self,
                messages,
                sampling_params=None,
                stream=False,
                update_callback=None,
                model="",
                max_retries=3,
            ):
                return {}

            def get_context_size(self, model=""):
                return 0

            def get_credits(self):
                return {}

            def is_up(self):
                return True

            def input_modes(self, model=""):
                return ["text"]

            def output_modes(self, model=""):
                return ["text"]

            def _should_stop_streaming(self, msg):
                return bool(msg.get("done"))

        client = TestClient(base_url="http://test")
        callback_calls = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hello"}, "finish_reason": null}]}',
            b'data: {"done": true}',
        ]
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"] == [
            {
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning_content": "",
                }
            }
        ]
        assert result["model"] == "test"
        assert result["provider"] == "http://test"
        assert "usage" in result
        assert "timings" in result
        assert callback_calls == [
            {"delta": {"content": "hello"}, "delta_type": "content"}
        ]


class TestBuildChatDataWithToolFields:
    def test_build_chat_data_with_tool_calls(self):
        client = LlamaCpp()
        tool_calls = [{"id": "tc1", "type": "function", "function": {"name": "foo"}}]
        data = client._build_chat_data(
            [{"role": "assistant", "content": "", "tool_calls": tool_calls}],
            None,
            "model1",
        )
        assert data == {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc1", "type": "function", "function": {"name": "foo"}}
                    ],
                }
            ],
            "model": "model1",
        }

    def test_build_chat_data_with_tool_call_id(self):
        client = LlamaCpp()
        data = client._build_chat_data(
            [{"role": "tool", "content": "result", "tool_call_id": "tc1"}],
            None,
            "model1",
        )
        assert data == {
            "messages": [{"role": "tool", "content": "result", "tool_call_id": "tc1"}],
            "model": "model1",
        }

    def test_build_chat_data_with_name(self):
        client = LlamaCpp()
        data = client._build_chat_data(
            [{"role": "tool", "content": "result", "name": "foo"}],
            None,
            "model1",
        )
        assert data == {
            "messages": [{"role": "tool", "content": "result", "name": "foo"}],
            "model": "model1",
        }

    def test_build_chat_data_with_all_tool_fields(self):
        client = LlamaCpp()
        tool_calls = [{"id": "tc1", "type": "function", "function": {"name": "foo"}}]
        data = client._build_chat_data(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                    "name": "bar",
                }
            ],
            None,
            "model1",
        )
        assert data == {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc1", "type": "function", "function": {"name": "foo"}}
                    ],
                    "name": "bar",
                }
            ],
            "model": "model1",
        }


class TestLlamaCppInputModes:
    def test_input_modes_text_only(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"modalities": {}}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            modes = client.input_modes()
        assert modes == ["text"]

    def test_input_modes_with_vision(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"modalities": {"vision": True}}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            modes = client.input_modes()
        assert modes == ["text", "image"]

    def test_input_modes_with_audio(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"modalities": {"audio": True}}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            modes = client.input_modes()
        assert modes == ["text", "audio"]

    def test_input_modes_with_vision_and_audio(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"modalities": {"vision": True, "audio": True}}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            modes = client.input_modes()
        assert modes == ["text", "image", "audio"]

    def test_input_modes_vision_false(self):
        client = LlamaCpp()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"modalities": {"vision": False}}
        mock_resp.headers = {"Content-Type": "application/json"}
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            modes = client.input_modes()
        assert modes == ["text"]


class TestLlamaCppOutputModes:
    def test_output_modes(self):
        client = LlamaCpp()
        assert client.output_modes() == ["text"]


class TestOpenRouterInputModes:
    def test_input_modes_found(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [
            {"id": "model1", "architecture": {"input_modalities": ["text", "image"]}}
        ]
        client._models_cache_time = time.monotonic()
        assert client.input_modes("model1") == ["text", "image"]

    def test_input_modes_not_found_returns_text(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "model1"}]
        client._models_cache_time = time.monotonic()
        assert client.input_modes("model2") == ["text"]

    def test_input_modes_default_model(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [
            {"id": "openrouter/free", "architecture": {"input_modalities": ["text"]}}
        ]
        client._models_cache_time = time.monotonic()
        assert client.input_modes() == ["text"]


class TestOpenRouterOutputModes:
    def test_output_modes_found(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [
            {"id": "model1", "architecture": {"output_modalities": ["text", "image"]}}
        ]
        client._models_cache_time = time.monotonic()
        assert client.output_modes("model1") == ["text", "image"]

    def test_output_modes_not_found_returns_text(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "model1"}]
        client._models_cache_time = time.monotonic()
        assert client.output_modes("model2") == ["text"]

    def test_output_modes_default_model(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [
            {"id": "openrouter/free", "architecture": {"output_modalities": ["text"]}}
        ]
        client._models_cache_time = time.monotonic()
        assert client.output_modes() == ["text"]


class TestValidateHost:
    def test_validate_host_valid_http(self):
        from tiz.inference_clients import validate_host

        assert validate_host("http://localhost:8080") == "http://localhost:8080"

    def test_validate_host_valid_https(self):
        from tiz.inference_clients import validate_host

        assert validate_host("https://example.com") == "https://example.com"

    def test_validate_host_missing_scheme(self):
        from tiz.inference_clients import validate_host

        with pytest.raises(ValueError, match="must include scheme"):
            validate_host("localhost:8080")

    def test_validate_host_empty_string(self):
        from tiz.inference_clients import validate_host

        with pytest.raises(ValueError, match="must include scheme"):
            validate_host("")


class TestOpenRouterFindModel:
    def test_find_model_found(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [
            {"id": "model1", "context_length": 8192},
            {"id": "model2", "context_length": 4096},
        ]
        client._models_cache_time = time.monotonic()
        result = client._find_model("model1")
        assert result == {"id": "model1", "context_length": 8192}

    def test_find_model_not_found(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "model1"}]
        client._models_cache_time = time.monotonic()
        assert client._find_model("model2") is None

    def test_find_model_strips_suffix(self):
        client = OpenRouter(api_key="test-key")
        client._models_cache = [{"id": "model1", "context_length": 8192}]
        client._models_cache_time = time.monotonic()
        result = client._find_model("model1:nitro")
        assert result == {"id": "model1", "context_length": 8192}

    def test_find_model_uses_default(self):
        client = OpenRouter(api_key="test-key", default_model="model2")
        client._models_cache = [
            {"id": "model1"},
            {"id": "model2", "context_length": 4096},
        ]
        client._models_cache_time = time.monotonic()
        result = client._find_model("")
        assert result == {"id": "model2", "context_length": 4096}


class TestLlamaCppInitHostValidation:
    def test_init_valid_host(self):
        client = LlamaCpp(host="http://localhost:9090")
        assert client.base_url == "http://localhost:9090"

    def test_init_host_missing_scheme_raises(self):
        with pytest.raises(ValueError, match="must include scheme"):
            LlamaCpp(host="localhost:8080")

    def test_init_empty_host_falls_back_to_default(self):
        client = LlamaCpp(host="")
        assert client.base_url == "http://127.0.0.1:8080"


class TestDwarfStar4InitHostValidation:
    def test_init_valid_host(self):
        client = DwarfStar4(host="http://localhost:9090")
        assert client.base_url == "http://localhost:9090"

    def test_init_host_missing_scheme_raises(self):
        with pytest.raises(ValueError, match="must include scheme"):
            DwarfStar4(host="localhost:8080")

    def test_init_empty_host_falls_back_to_default(self):
        client = DwarfStar4(host="")
        assert client.base_url == "http://127.0.0.1:8080"


# ===========================================================================
# AnthropicClient tests
# ===========================================================================


class TestAnthropicClientInit:
    def test_init_default_values(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert client.api_key == "sk-ant-test"
        assert client._default_model == "claude-sonnet-5"
        assert client.timeout == 60.0
        assert client.message_timeout is None
        assert client.url == "https://api.anthropic.com/v1/messages"
        assert client.headers["x-api-key"] == "sk-ant-test"
        assert client.headers["anthropic-version"] == "2023-06-01"
        assert client.headers["Content-Type"] == "application/json"
        assert client._preserve_thinking is False
        assert client.sampling_params == {}

    def test_init_custom_model_and_params(self):
        client = AnthropicClient(
            api_key="sk-ant-test",
            default_model="claude-opus-4-20250514",
            sampling_params={"temperature": 0.5, "max_tokens": 8192},
            preserve_thinking=True,
            timeout=120.0,
            message_timeout=300.0,
        )
        assert client._default_model == "claude-opus-4-20250514"
        assert client.sampling_params == {"temperature": 0.5, "max_tokens": 8192}
        assert client.preserve_thinking is True
        assert client.timeout == 120.0
        assert client.message_timeout == 300.0


class TestAnthropicClientConvertMessages:
    def test_system_message_single(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [{"role": "system", "content": "You are a helpful assistant."}]
        )
        assert system == "You are a helpful assistant."
        assert msgs == []

    def test_system_message_with_user_message(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        )
        assert system == "You are helpful."
        assert msgs == [{"role": "user", "content": "Hello"}]

    def test_multiple_system_messages(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [
                {"role": "system", "content": "First."},
                {"role": "system", "content": "Second."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert system == [
            {"type": "text", "text": "First."},
            {"type": "text", "text": "Second."},
        ]
        assert msgs == [{"role": "user", "content": "Hi"}]

    def test_tool_result_message(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Let me search"},
                {
                    "role": "tool",
                    "content": "Result data",
                    "tool_call_id": "tc_123",
                },
            ]
        )
        assert msgs == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Let me search"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc_123",
                        "content": "Result data",
                    }
                ],
            },
        ]

    def test_assistant_with_tool_calls(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [
                {"role": "user", "content": "Search"},
                {
                    "role": "assistant",
                    "content": "I will search",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q": "test"}',
                            },
                        }
                    ],
                },
            ]
        )
        assert msgs == [
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will search"},
                    {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "search",
                        "input": {"q": "test"},
                    },
                ],
            },
        ]

    def test_assistant_tool_calls_no_content(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [
                {"role": "user", "content": "Search"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
            ]
        )
        assert msgs == [
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "search",
                        "input": {},
                    }
                ],
            },
        ]

    def test_assistant_tool_calls_malformed_json_args(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [
                {"role": "user", "content": "Search"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": "not-json",
                            },
                        }
                    ],
                },
            ]
        )
        assert msgs == [
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "search",
                        "input": {},
                    }
                ],
            },
        ]

    def test_regular_user_assistant_messages(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        assert system is None
        assert msgs == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_empty_system_content_no_system(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages(
            [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}]
        )
        assert system is None
        assert msgs == [{"role": "user", "content": "Hi"}]

    def test_tool_result_no_tool_call_id(self):
        client = AnthropicClient(api_key="sk-ant-test")
        system, msgs = client._convert_messages([{"role": "tool", "content": "result"}])
        assert msgs == [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "", "content": "result"}
                ],
            }
        ]


class TestAnthropicClientConvertTools:
    def test_convert_tools_empty(self):
        result = AnthropicClient._convert_tools([])
        assert result == []

    def test_convert_tools_basic(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ]
        result = AnthropicClient._convert_tools(tools)
        assert result == [
            {
                "name": "search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ]

    def test_convert_tools_without_function_type(self):
        tools = [{"type": "not_function", "name": "test"}]
        result = AnthropicClient._convert_tools(tools)
        assert result == []

    def test_convert_tools_missing_fields(self):
        tools = [{"type": "function", "function": {}}]
        result = AnthropicClient._convert_tools(tools)
        assert result == [
            {
                "name": "",
                "description": "",
                "input_schema": {},
            }
        ]


class TestAnthropicClientBackoffDelay:
    def test_backoff_delay_first_retry(self):
        delay = AnthropicClient._backoff_delay(1)
        assert 2 <= delay < 3

    def test_backoff_delay_clamped_at_60(self):
        delay = AnthropicClient._backoff_delay(10)
        assert 60 <= delay < 61

    def test_backoff_delay_zero_retries(self):
        delay = AnthropicClient._backoff_delay(0)
        assert 1 <= delay < 2


class TestAnthropicClientGetModels:
    def test_get_models_returns_default(self):
        client = AnthropicClient(api_key="sk-ant-test")
        models = client.get_models()
        assert models == ["claude-sonnet-5"]

    def test_get_models_with_default_model_set(self):
        client = AnthropicClient(
            api_key="sk-ant-test", default_model="claude-opus-4-20250514"
        )
        models = client.get_models()
        assert models == ["claude-opus-4-20250514"]


class TestAnthropicClientCountTokens:
    def test_count_tokens_heuristic(self):
        client = AnthropicClient(api_key="sk-ant-test")
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = client.count_tokens(messages)
        total_chars = len("Hello world") + len("Hi there")
        assert count == max(1, total_chars // 4 + 2 * 5)

    def test_count_tokens_empty_content(self):
        client = AnthropicClient(api_key="sk-ant-test")
        count = client.count_tokens([{"role": "user", "content": ""}])
        assert count == 5

    def test_count_tokens_missing_content(self):
        client = AnthropicClient(api_key="sk-ant-test")
        count = client.count_tokens([{"role": "user"}])
        assert count == 5

    def test_count_tokens_empty_messages(self):
        client = AnthropicClient(api_key="sk-ant-test")
        count = client.count_tokens([])
        assert count == 1

    def test_count_tokens_with_preserve_thinking(self):
        client = AnthropicClient(api_key="sk-ant-test", preserve_thinking=True)
        messages = [
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Let me think...",
            },
        ]
        count = client.count_tokens(messages)
        total_chars = len("Final answer") + len("Let me think...")
        assert count == max(1, total_chars // 4 + 1 * 5)

    def test_count_tokens_reasoning_not_counted_when_not_preserved(self):
        client = AnthropicClient(api_key="sk-ant-test", preserve_thinking=False)
        messages = [
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Let me think...",
            },
        ]
        count = client.count_tokens(messages)
        total_chars = len("Final answer")
        assert count == max(1, total_chars // 4 + 1 * 5)


class TestAnthropicClientToolsSupport:
    def test_tools_support_always_true(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert client.tools_support() is True
        assert client.tools_support("any-model") is True


class TestAnthropicClientInputModes:
    def test_input_modes_include_text_and_image(self):
        client = AnthropicClient(api_key="sk-ant-test")
        modes = client.input_modes()
        assert "text" in modes
        assert "image" in modes

    def test_input_modes_with_model_arg_ignored(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert client.input_modes("any-model") == ["text", "image"]


class TestAnthropicClientOutputModes:
    def test_output_modes_text_only(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert client.output_modes() == ["text"]


class TestAnthropicClientGetContextSize:
    def test_get_context_size(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert client.get_context_size() == 1000000
        assert client.get_context_size("any-model") == 1000000


class TestAnthropicClientGetCredits:
    def test_get_credits_returns_zero(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert client.get_credits() == {"total_credits": 0.0, "total_usage": 0.0}


class TestAnthropicClientIsUp:
    def test_is_up_success(self):
        client = AnthropicClient(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch(
            "tiz.inference_clients.requests.get", return_value=mock_resp
        ) as mock_get:
            assert client.is_up() is True
        mock_get.assert_called_once_with(
            AnthropicClient.BASE_URL,
            headers=client.headers,
            timeout=5,
        )

    def test_is_up_server_error_still_responds(self):
        client = AnthropicClient(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.is_up() is True

    def test_is_up_5xx_failure(self):
        client = AnthropicClient(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("tiz.inference_clients.requests.get", return_value=mock_resp):
            assert client.is_up() is False

    def test_is_up_exception(self):
        client = AnthropicClient(api_key="sk-ant-test")
        with patch(
            "tiz.inference_clients.requests.get",
            side_effect=Exception("connection failed"),
        ):
            assert client.is_up() is False


class TestAnthropicClientBuildResult:
    def test_build_result_text_only(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-sonnet-5",
        }
        result = client._build_result(api_result, {"model": "fallback"})
        assert result == {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                        "reasoning_content": "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "timings": {"prompt_time": 0, "completion_time": 0},
            "model": "claude-sonnet-5",
            "provider": "Anthropic/https://api.anthropic.com",
        }

    def test_build_result_with_thinking(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [
                {"type": "thinking", "thinking": "Let me reason..."},
                {"type": "text", "text": "Final answer"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "model": "claude-sonnet-5",
        }
        result = client._build_result(api_result, {})
        assert result["choices"][0]["message"]["content"] == "Final answer"
        assert (
            result["choices"][0]["message"]["reasoning_content"] == "Let me reason..."
        )

    def test_build_result_with_tool_use(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [
                {"type": "text", "text": "I will search"},
                {
                    "type": "tool_use",
                    "id": "tu_123",
                    "name": "search",
                    "input": {"q": "hello"},
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._build_result(api_result, {"model": "m"})
        assert result["choices"][0]["message"]["tool_calls"] == [
            {
                "id": "tu_123",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"q": "hello"}',
                },
            }
        ]

    def test_build_result_empty_content(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [],
            "usage": {},
            "model": "claude-sonnet-5",
        }
        result = client._build_result(api_result, {"model": "fallback"})
        assert result["choices"][0]["message"]["content"] == ""
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0
        assert result["model"] == "claude-sonnet-5"

    def test_build_result_fallback_model(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "hi"}],
            "usage": {},
        }
        result = client._build_result(api_result, {"model": "claude-sonnet-5"})
        assert result["model"] == "claude-sonnet-5"


class TestAnthropicClientChatNonStream:
    def test_chat_non_stream_success(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-sonnet-5",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                model="claude-sonnet-5",
                stream=False,
            )
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        mock_post.assert_called_once()

    def test_chat_non_stream_with_sampling_params(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {},
            "model": "claude-sonnet-5",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                sampling_params={"temperature": 0.7},
                stream=False,
            )
        assert result["choices"][0]["message"]["content"] == "Hello!"
        call_kwargs = mock_post.call_args.kwargs
        json_data = call_kwargs["json"]
        assert json_data["temperature"] == 0.7  # type: ignore[comparison-overlap]
        assert json_data["max_tokens"] == 4096  # type: ignore[comparison-overlap]

    def test_chat_non_stream_with_tools(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [
                {"type": "text", "text": "Searching..."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "search",
                    "input": {"q": "hello"},
                },
            ],
            "usage": {},
            "model": "claude-sonnet-5",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object"},
                },
            }
        ]
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            result = client.chat(
                [{"role": "user", "content": "Search"}],
                sampling_params={"tools": tools},
                stream=False,
            )
        assert result["choices"][0]["message"]["tool_calls"] == [
            {
                "id": "tu_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"q": "hello"}',
                },
            }
        ]
        call_kwargs = mock_post.call_args.kwargs
        json_data = call_kwargs["json"]
        assert "tools" in json_data
        assert json_data["tools"] == [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]

    def test_chat_non_stream_retry_on_500(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {},
        }
        error_resp = MagicMock()
        error_resp.status_code = 500
        http_error = requests.exceptions.HTTPError(response=error_resp)
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[http_error, mock_resp],
            ) as mock_post,
            patch("time.sleep"),
        ):
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            )
        assert mock_post.call_count == 2
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_chat_non_stream_retry_exhausted(self):
        client = AnthropicClient(api_key="sk-ant-test")
        error_resp = MagicMock()
        error_resp.status_code = 500
        http_error = requests.exceptions.HTTPError(response=error_resp)
        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=http_error,
            ),
            patch("time.sleep"),
            pytest.raises(requests.exceptions.HTTPError),
        ):
            client._chat_non_stream({"model": "test"}, max_retries=0)

    def test_chat_non_stream_no_retry_on_400(self):
        client = AnthropicClient(api_key="sk-ant-test")
        error_resp = MagicMock()
        error_resp.status_code = 400
        http_error = requests.exceptions.HTTPError(response=error_resp)
        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=http_error,
            ),
            pytest.raises(requests.exceptions.HTTPError),
        ):
            client._chat_non_stream({"model": "test"})

    def test_chat_non_stream_uses_message_timeout(self):
        client = AnthropicClient(api_key="sk-ant-test", message_timeout=300.0)
        api_result = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_non_stream({"model": "test"})
        assert mock_post.call_args.kwargs["timeout"] == 300.0

    def test_chat_non_stream_explicit_timeout_overrides_message_timeout(self):
        client = AnthropicClient(api_key="sk-ant-test", message_timeout=300.0)
        api_result = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_non_stream({"model": "test"}, timeout=30.0)
        assert mock_post.call_args.kwargs["timeout"] == 30.0


class TestAnthropicClientChatStream:
    def test_chat_stream_text_content(self):
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hello"}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                stream=True,
                update_callback=callback,
            )
        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["model"] == "claude-sonnet-5"
        assert "Anthropic" in result["provider"]
        assert len(callback_calls) >= 1

    def test_chat_stream_with_thinking(self):
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 5, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": "Let me"}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": " think..."}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Answer: "}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "42"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "What is 6*7?"}],
                stream=True,
                update_callback=callback,
            )
        assert result["choices"][0]["message"]["content"] == "Answer: 42"
        assert result["choices"][0]["message"]["reasoning_content"] == "Let me think..."
        assert result["usage"]["prompt_tokens"] == 5
        assert result["usage"]["completion_tokens"] == 10

    def test_chat_stream_with_tool_use(self):
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Searching"}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 15}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "Search"}],
                stream=True,
                update_callback=callback,
            )
        assert result["choices"][0]["message"]["content"] == "Searching..."
        assert result["choices"][0]["message"]["tool_calls"] == [
            {
                "id": "tu_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ]

    def test_chat_stream_without_callback_creates_noop(self):
        client = AnthropicClient(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hi"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                stream=True,
            )
        assert result["choices"][0]["message"]["content"] == "Hi"

    def test_chat_stream_chunked_encoding_retry_succeeds(self):
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        fail_resp = MagicMock()
        fail_resp.iter_lines.side_effect = requests.exceptions.ChunkedEncodingError(
            "connection closed"
        )
        fail_resp.headers = {"Content-Type": "application/json"}
        fail_resp.status_code = 200
        fail_resp.__enter__ = MagicMock(return_value=fail_resp)
        fail_resp.__exit__ = MagicMock(return_value=False)

        success_resp = MagicMock()
        success_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "OK"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        success_resp.status_code = 200
        success_resp.__enter__ = MagicMock(return_value=success_resp)
        success_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[fail_resp, success_resp],
            ),
            patch("time.sleep"),
        ):
            result = client._chat_stream(
                {"model": "test"},
                callback,
                max_retries=3,
            )
        assert result["choices"][0]["message"]["content"] == "OK"

    def test_chat_stream_chunked_encoding_retry_exhausted(self):
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        fail_resp = MagicMock()
        fail_resp.iter_lines.side_effect = requests.exceptions.ChunkedEncodingError(
            "connection closed"
        )
        fail_resp.headers = {"Content-Type": "application/json"}
        fail_resp.status_code = 200
        fail_resp.__enter__ = MagicMock(return_value=fail_resp)
        fail_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[fail_resp] * 5,
            ),
            patch("time.sleep"),
            pytest.raises(RuntimeError, match="Stream failed after 3 retries"),
        ):
            client._chat_stream(
                {"model": "test"},
                callback,
                max_retries=3,
            )

    def test_chat_stream_uses_message_timeout(self):
        client = AnthropicClient(api_key="sk-ant-test", message_timeout=300.0)
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hi"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_stream({"model": "test"}, callback)
        assert mock_post.call_args.kwargs["timeout"] == 300.0

    def test_chat_stream_explicit_timeout_overrides_message_timeout(self):
        client = AnthropicClient(api_key="sk-ant-test", message_timeout=300.0)
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hi"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client._chat_stream({"model": "test"}, callback, timeout=30.0)
        assert mock_post.call_args.kwargs["timeout"] == 30.0


class TestAnthropicClientParseAnthropicSSE:
    def test_parse_sse_basic(self):
        client = AnthropicClient(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"id": "msg_1"}}',
            b"",
        ]
        events = list(client._parse_anthropic_sse(mock_resp))
        assert len(events) == 1
        assert events[0]["_event"] == "message_start"
        assert events[0]["message"]["id"] == "msg_1"

    def test_parse_sse_skips_non_data_lines(self):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"some random line",
            b"",
            None,
        ]
        events = list(AnthropicClient._parse_anthropic_sse(mock_resp))
        assert events == []

    def test_parse_sse_empty_data(self):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b"data:   ",
            b"",
        ]
        events = list(AnthropicClient._parse_anthropic_sse(mock_resp))
        assert len(events) == 0

    def test_parse_sse_malformed_json(self):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: ping",
            b"data: not-json",
            b"",
        ]
        events = list(AnthropicClient._parse_anthropic_sse(mock_resp))
        assert events == []

    def test_parse_sse_multiple_events(self):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: ping",
            b'data: {"type": "ping"}',
            b"",
            b"event: message_start",
            b'data: {"type": "message_start", "message": {}}',
            b"",
        ]
        events = list(AnthropicClient._parse_anthropic_sse(mock_resp))
        assert len(events) == 2
        assert events[0]["_event"] == "ping"
        assert events[1]["_event"] == "message_start"

    def test_parse_sse_multiline_data_no_event(self):
        """When no event type is set, default to empty string."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'data: {"key": "value"}',
            b"",
        ]
        events = list(AnthropicClient._parse_anthropic_sse(mock_resp))
        assert len(events) == 1
        assert events[0]["_event"] == ""


class TestAnthropicClientSetDefaultModel:
    def test_set_default_model(self):
        client = AnthropicClient(api_key="sk-ant-test")
        client.set_default_model("claude-opus-4-20250514")
        assert client._default_model == "claude-opus-4-20250514"


class TestAnthropicClientResolveModel:
    def test_resolve_model_with_value(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert (
            client._resolve_model("claude-opus-4-20250514") == "claude-opus-4-20250514"
        )

    def test_resolve_model_empty_fallback(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert client._resolve_model("") == "claude-sonnet-5"

    def test_resolve_model_custom_default(self):
        client = AnthropicClient(
            api_key="sk-ant-test", default_model="claude-opus-4-20250514"
        )
        assert client._resolve_model("") == "claude-opus-4-20250514"


class TestAnthropicClientAdditional:
    def test_parse_sse_trailing_data_without_newline(self):
        """Cover trailing data lines at end-of-stream."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {}}',
        ]
        events = list(AnthropicClient._parse_anthropic_sse(mock_resp))
        assert len(events) == 1
        assert events[0]["_event"] == "message_start"

    def test_parse_sse_trailing_malformed_json(self):
        """Cover trailing data with malformed JSON."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: ping",
            b"data: bad-json",
        ]
        events = list(AnthropicClient._parse_anthropic_sse(mock_resp))
        assert events == []

    def test_chat_with_system_as_list(self):
        """Cover system parameter as a list of content blocks."""
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {},
            "model": "claude-sonnet-5",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client.chat(messages, stream=False)
        call_kwargs = mock_post.call_args.kwargs
        json_data = call_kwargs["json"]
        assert "system" in json_data
        assert json_data["system"] == [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ]

    def test_chat_non_stream_connection_error_retry(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "OK"}],
            "usage": {},
        }
        conn_error = requests.exceptions.ConnectionError()
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[conn_error, mock_resp],
            ),
            patch("time.sleep"),
        ):
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            )
        assert result["choices"][0]["message"]["content"] == "OK"

    def test_chat_non_stream_timeout_retry(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "OK"}],
            "usage": {},
        }
        timeout_error = requests.exceptions.Timeout()
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[timeout_error, mock_resp],
            ),
            patch("time.sleep"),
        ):
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            )
        assert result["choices"][0]["message"]["content"] == "OK"

    def test_chat_non_stream_http_error_no_response_raises(self):
        client = AnthropicClient(api_key="sk-ant-test")
        http_error = requests.exceptions.HTTPError("bad request")
        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=http_error,
            ),
            pytest.raises(requests.exceptions.HTTPError),
        ):
            client._chat_non_stream({"model": "test"})

    def test_chat_stream_with_non_text_blocks(self):
        """Cover content_block_start for non-text/non-thinking blocks like
        tool_use and input_json_delta events."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 5, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{"q": "hello"}"}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 10}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["tool_calls"] == [
            {
                "id": "tu_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ]
        assert result["usage"]["prompt_tokens"] == 5
        assert result["usage"]["completion_tokens"] == 10

    def test_chat_stream_first_update_time_set(self):
        """Cover the branch where _first_update_time is None and gets set."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hel"}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["content"] == "Hello"
        assert len(callback_calls) == 2

    def test_default_model_via_resolve(self):
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client.chat(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            )
        call_kwargs = mock_post.call_args.kwargs
        json_data = call_kwargs["json"]
        assert json_data["model"] == "claude-sonnet-5"

    def test_provider_in_result(self):
        client = AnthropicClient(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {},
        }
        mock_resp.status_code = 200
        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client.chat(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            )
        assert result["provider"] == "Anthropic/https://api.anthropic.com"

    def test_parse_sse_trailing_empty_data(self):
        """Cover trailing data lines where data_str is empty after strip (line 1271)."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"data:   ",
        ]
        events = list(AnthropicClient._parse_anthropic_sse(mock_resp))
        assert events == []

    def test_chat_with_empty_tools_conversion(self):
        """Cover tools=sampling_param where converted list is empty (line 1465)."""
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {"content": [{"type": "text", "text": "Hi"}], "usage": {}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_result
        mock_resp.status_code = 200
        tools = [{"type": "not_function", "name": "test"}]
        with patch(
            "tiz.inference_clients.requests.post", return_value=mock_resp
        ) as mock_post:
            client.chat(
                [{"role": "user", "content": "Hi"}],
                sampling_params={"tools": tools},
                stream=False,
            )
        call_kwargs = mock_post.call_args.kwargs
        json_data = call_kwargs["json"]
        assert "tools" not in json_data

    def test_chat_stream_no_message_stop_event(self):
        """Cover stream loop ending without message_stop (line 1556->1628)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hello"}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test_model"}, callback)
        assert result["choices"][0]["message"]["content"] == "Hello"
        assert result["model"] == "test_model"

    def test_chat_stream_thinking_delta_before_first_update(self):
        """Cover thinking_delta arriving when _first_update_time is None (line 1603)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "think..."}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["reasoning_content"] == "think..."
        assert len(callback_calls) >= 1

    def test_chat_stream_text_delta_before_first_update(self):
        """Cover text_delta arriving when _first_update_time is None (line 1615)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["content"] == "Hi"
        assert len(callback_calls) >= 1

    def test_chat_stream_input_json_delta(self):
        """Cover input_json_delta branch (line 1618-1619)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "input_json_delta"}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["tool_calls"] == [
            {
                "id": "tu_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ]

    def test_chat_stream_unknown_event_type(self):
        """Cover unknown event type falling through all elif branches (line 1625->1556)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: ping",
            b'data: {"type": "ping"}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hi"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["content"] == "Hi"

    def test_chat_stream_unknown_content_block_type(self):
        """Cover content_block_start with unknown block type (line 1580->1556)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "unknown_block", "data": "test"}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hello"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["content"] == "Hello"

    def test_build_result_unknown_block_type(self):
        """Cover _build_result with unknown block type (line 1715->1709)."""
        client = AnthropicClient(api_key="sk-ant-test")
        api_result = {
            "content": [
                {"type": "unknown", "data": "test"},
                {"type": "text", "text": "Hello"},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "model": "claude-sonnet-5",
        }
        result = client._build_result(api_result, {"model": "fallback"})
        assert result["choices"][0]["message"]["content"] == "Hello"
        assert result["usage"]["prompt_tokens"] == 5
        assert result["usage"]["completion_tokens"] == 3

    def test_chat_stream_unknown_delta_type(self):
        """Cover content_block_delta with unknown delta type (line 1618->1556)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "Hi"}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "unknown_delta_type"}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert result["choices"][0]["message"]["content"] == "Hi"
        assert len(callback_calls) >= 1

    def test_chat_stream_thinking_block_start_already_set(self):
        """Cover thinking block_start when _first_update_time already set (line 1574->1576)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": "Let me "}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "think"}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": "more "}}',
            b"",
            b"event: content_block_delta",
            b'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "thoughts"}}',
            b"",
            b"event: content_block_stop",
            b'data: {"type": "content_block_stop"}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tiz.inference_clients.requests.post", return_value=mock_resp):
            result = client._chat_stream({"model": "test"}, callback)
        assert (
            result["choices"][0]["message"]["reasoning_content"]
            == "Let me thinkmore thoughts"
        )

    def test_chat_stream_http_error_with_response_retry(self):
        """Cover Anthropic _chat_stream HTTPError with a response object (line 1690)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        error_resp = MagicMock()
        error_resp.status_code = 503
        http_error = requests.exceptions.HTTPError(response=error_resp)

        success_resp = MagicMock()
        success_resp.iter_lines.return_value = [
            b"event: message_start",
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 0}}}',
            b"",
            b"event: content_block_start",
            b'data: {"type": "content_block_start", "content_block": {"type": "text", "text": "OK"}}',
            b"",
            b"event: message_delta",
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}',
            b"",
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
            b"",
        ]
        success_resp.status_code = 200
        success_resp.__enter__ = MagicMock(return_value=success_resp)
        success_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=[http_error, success_resp],
            ),
            patch("time.sleep"),
        ):
            result = client._chat_stream({"model": "test"}, callback, max_retries=3)
        assert result["choices"][0]["message"]["content"] == "OK"

    def test_chat_stream_non_retryable_http_error_raises(self):
        """Cover Anthropic _chat_stream non-retryable HTTPError raising (line 1707)."""
        client = AnthropicClient(api_key="sk-ant-test")
        callback_calls: list[dict[str, Any]] = []

        def callback(chunk, _subtask_name=None):
            callback_calls.append(chunk)

        error_resp = MagicMock()
        error_resp.status_code = 400
        http_error = requests.exceptions.HTTPError(response=error_resp)

        with (
            patch(
                "tiz.inference_clients.requests.post",
                side_effect=http_error,
            ),
            pytest.raises(requests.exceptions.HTTPError),
        ):
            client._chat_stream({"model": "test"}, callback, max_retries=3)
