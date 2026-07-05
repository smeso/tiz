"""Tests for audio_inference_clients module."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from tiz.audio_inference_clients import AudioInferenceClient, WhisperCpp


class DummyClient(AudioInferenceClient):
    """Concrete subclass used across multiple tests."""

    def transcribe(self, _audio):
        return ""


@pytest.fixture
def dummy_client():
    """Return a fresh DummyClient instance."""
    return DummyClient()


# ===========================================================================
# AudioInferenceClient (abstract base) tests
# ===========================================================================


def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        AudioInferenceClient()  # type: ignore[abstract]


def test_init_without_sampling_params():
    client = DummyClient()
    assert client._sampling_params == {}
    assert client._language is None
    assert client._prompt is None


def test_init_with_sampling_params():
    client = DummyClient(sampling_params={"temperature": 0.5, "language": "en"})
    assert client._sampling_params == {"temperature": 0.5, "language": "en"}
    assert client._language is None
    assert client._prompt is None


def test_init_with_language():
    client = DummyClient(language="fr")
    assert client._language == "fr"
    assert client._prompt is None


def test_init_with_prompt():
    client = DummyClient(prompt="Hello, I am")
    assert client._prompt == "Hello, I am"
    assert client._language is None


def test_init_with_language_and_prompt():
    client = DummyClient(language="de", prompt="Bitte schreiben")
    assert client._language == "de"
    assert client._prompt == "Bitte schreiben"


def test_sampling_params_getter():
    client = DummyClient(sampling_params={"temperature": 0.5})
    assert client.sampling_params == {"temperature": 0.5}


def test_sampling_params_setter():
    client = DummyClient()
    client.sampling_params = {"temperature": 0.9, "top_p": 0.8}
    assert client._sampling_params == {"temperature": 0.9, "top_p": 0.8}


def test_language_property():
    client = DummyClient(language="en")
    assert client.language == "en"

    client2 = DummyClient()
    assert client2.language is None


def test_prompt_property():
    client = DummyClient(prompt="Hello")
    assert client.prompt == "Hello"

    client2 = DummyClient()
    assert client2.prompt is None


# ===========================================================================
# WhisperCpp tests
# ===========================================================================


class TestWhisperCppInit:
    def test_default_values(self):
        client = WhisperCpp()
        assert client.host == "http://127.0.0.1:8080"
        assert client.timeout == 5
        assert client.inference_timeout is None
        assert client.verify_ssl is True
        assert client.ca_cert is None
        assert client.url == "http://127.0.0.1:8080/inference"
        assert client._sampling_params == {}
        assert client._language is None
        assert client._prompt is None
        assert client.headers == {
            "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; tiz/0.1.0; +https://github.com/smeso/tiz",
        }

    def test_strips_trailing_slash(self):
        client = WhisperCpp(host="http://127.0.0.1:8080/")
        assert client.host == "http://127.0.0.1:8080"
        assert client.url == "http://127.0.0.1:8080/inference"

    def test_custom_values(self):
        client = WhisperCpp(
            host="http://localhost:9000",
            timeout=30.0,
            inference_timeout=60.0,
            verify_ssl=False,
            ca_cert="/path/to/ca.pem",
            sampling_params={"temperature": 0.2},
        )
        assert client.host == "http://localhost:9000"
        assert client.timeout == 30.0
        assert client.inference_timeout == 60.0
        assert client.verify_ssl is False
        assert client.ca_cert == "/path/to/ca.pem"
        assert client.url == "http://localhost:9000/inference"
        assert client._sampling_params == {"temperature": 0.2}

    def test_custom_language_and_prompt(self):
        client = WhisperCpp(language="fr", prompt="Bonjour")
        assert client._language == "fr"
        assert client._prompt == "Bonjour"
        assert client.language == "fr"
        assert client.prompt == "Bonjour"


class TestVerifyProperty:
    def test_verify_true(self):
        client = WhisperCpp()
        assert client._verify is True

    def test_verify_false(self):
        client = WhisperCpp(verify_ssl=False)
        assert client._verify is False

    def test_verify_with_ca_cert(self):
        client = WhisperCpp(ca_cert="/etc/ssl/certs/ca.pem")
        assert client._verify == "/etc/ssl/certs/ca.pem"


class TestAudioBytes:
    def test_bytes_unchanged(self):
        client = WhisperCpp()
        audio = b"raw audio data"
        assert client._audio_bytes(audio) is audio

    def test_base64_decoded(self):
        client = WhisperCpp()
        original = b"raw audio data"
        encoded = base64.b64encode(original).decode("ascii")
        assert client._audio_bytes(encoded) == original

    def test_empty_bytes(self):
        client = WhisperCpp()
        with pytest.raises(ValueError, match="Audio bytes data is empty"):
            client._audio_bytes(b"")

    def test_empty_base64_string(self):
        client = WhisperCpp()
        with pytest.raises(ValueError, match="Base64-encoded audio string is empty"):
            client._audio_bytes("")

    def test_invalid_base64_string(self):
        client = WhisperCpp()
        with pytest.raises(ValueError, match="Invalid base64-encoded audio"):
            client._audio_bytes("not-valid-base64!!!")

    def test_whitespace_only_base64_string(self):
        client = WhisperCpp()
        with pytest.raises(ValueError, match="Base64-encoded audio string is empty"):
            client._audio_bytes("   ")


class TestTranscribe:
    @patch("tiz.audio_inference_clients.requests.post")
    def test_basic_transcribe(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hello world"}
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        result = client.transcribe(b"audio data")

        assert result == "hello world"
        mock_post.assert_called_once_with(
            client.url,
            headers=client.headers,
            files={
                "file": ("audio.wav", b"audio data", "application/octet-stream"),
            },
            data={"language": "auto", "response_format": "json"},
            timeout=5,
            verify=True,
        )

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_with_language(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "bonjour"}
        mock_post.return_value = mock_resp

        client = WhisperCpp(language="fr")
        result = client.transcribe(b"audio")

        assert result == "bonjour"
        assert mock_post.call_args[1]["data"]["language"] == "fr"

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_with_prompt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "transcribed"}
        mock_post.return_value = mock_resp

        client = WhisperCpp(prompt="Hello, I am")
        result = client.transcribe(b"audio")

        assert result == "transcribed"
        assert mock_post.call_args[1]["data"]["prompt"] == "Hello, I am"

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_base64_audio(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "decoded"}
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        encoded = base64.b64encode(b"audio data").decode("ascii")
        result = client.transcribe(encoded)

        assert result == "decoded"
        sent_files = mock_post.call_args[1]["files"]
        assert sent_files["file"][1] == b"audio data"

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_with_language_and_prompt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "guten tag"}
        mock_post.return_value = mock_resp

        client = WhisperCpp(language="de", prompt="Bitte schreiben")
        result = client.transcribe(b"audio")

        assert result == "guten tag"
        data = mock_post.call_args[1]["data"]
        assert data["language"] == "de"
        assert data["prompt"] == "Bitte schreiben"

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_with_sampling_params(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hello"}
        mock_post.return_value = mock_resp

        client = WhisperCpp(sampling_params={"temperature": 0.3, "best_of": 5})
        client.transcribe(b"audio")

        data = mock_post.call_args[1]["data"]
        assert data["temperature"] == "0.3"
        assert data["best_of"] == "5"
        assert data["language"] == "auto"

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_uses_inference_timeout(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hi"}
        mock_post.return_value = mock_resp

        client = WhisperCpp(timeout=5, inference_timeout=60)
        client.transcribe(b"audio")

        assert mock_post.call_args[1]["timeout"] == 60

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_falls_back_to_timeout(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hi"}
        mock_post.return_value = mock_resp

        client = WhisperCpp(timeout=5, inference_timeout=None)
        client.transcribe(b"audio")

        assert mock_post.call_args[1]["timeout"] == 5

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_empty_json_response_raises_runtimeerror(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        with pytest.raises(RuntimeError, match="missing.*text"):
            client.transcribe(b"audio")

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_text_key_missing_raises_runtimeerror(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"other": "data"}
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        with pytest.raises(RuntimeError, match="missing.*text"):
            client.transcribe(b"audio")

    @patch("tiz.audio_inference_clients.requests.post")
    def test_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        with pytest.raises(RuntimeError, match="WhisperCpp request failed"):
            client.transcribe(b"audio")

    @patch("tiz.audio_inference_clients.requests.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("connection failed")

        client = WhisperCpp()
        with pytest.raises(RuntimeError, match="WhisperCpp request failed"):
            client.transcribe(b"audio")

    @patch("tiz.audio_inference_clients.requests.post")
    def test_timeout_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        client = WhisperCpp()
        with pytest.raises(RuntimeError, match="WhisperCpp request failed"):
            client.transcribe(b"audio")

    @patch("tiz.audio_inference_clients.requests.post")
    def test_json_decode_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
        mock_resp.status_code = 200
        mock_resp.text = "not json"
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        with pytest.raises(RuntimeError, match="invalid JSON"):
            client.transcribe(b"audio")

    @patch("tiz.audio_inference_clients.requests.post")
    def test_with_verify_ssl_false(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "insecure"}
        mock_post.return_value = mock_resp

        client = WhisperCpp(verify_ssl=False)
        client.transcribe(b"audio")

        assert mock_post.call_args[1]["verify"] is False

    @patch("tiz.audio_inference_clients.requests.post")
    def test_with_ca_cert_verify(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "secure"}
        mock_post.return_value = mock_resp

        client = WhisperCpp(ca_cert="/etc/ssl/custom.pem")
        client.transcribe(b"audio")

        assert mock_post.call_args[1]["verify"] == "/etc/ssl/custom.pem"

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_json_array_response_raises_runtimeerror(self, mock_post):
        """Server returns a JSON array instead of dict (Bug #1)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [1, 2, 3]
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        with pytest.raises(RuntimeError, match="missing.*text"):
            client.transcribe(b"audio")

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_non_string_text_returns_string(self, mock_post):
        """Server returns {"text": 123} - must be converted to str (Bug #2)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": 123}
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        result = client.transcribe(b"audio")

        assert result == "123"

    @patch("tiz.audio_inference_clients.requests.post")
    def test_transcribe_null_text_returns_empty_string(self, mock_post):
        """Server returns {"text": null} - should return empty string."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": None}
        mock_post.return_value = mock_resp

        client = WhisperCpp()
        result = client.transcribe(b"audio")

        assert result == ""


class TestWhisperCppHostValidation:
    def test_accepts_valid_http_host(self):
        client = WhisperCpp(host="http://example.com:8080")
        assert client.host == "http://example.com:8080"

    def test_accepts_valid_https_host(self):
        client = WhisperCpp(host="https://whisper.example.com")
        assert client.host == "https://whisper.example.com"

    def test_rejects_host_missing_scheme(self):
        with pytest.raises(ValueError, match="Invalid host URL"):
            WhisperCpp(host="127.0.0.1:8080")

    def test_rejects_empty_host(self):
        with pytest.raises(ValueError, match="Invalid host URL"):
            WhisperCpp(host="")
