"""Audio inference clients for speech-to-text backends.

This module provides a base ``AudioInferenceClient`` class and concrete
implementations for:

* ``WhisperCpp`` - a self-hosted whisper.cpp server
"""

from __future__ import annotations

import base64
import binascii
import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from tiz.inference_clients import USER_AGENT, SamplingParams, validate_host


class AudioInferenceClient(ABC):
    """Abstract base class for audio inference clients.

    Subclasses must implement the abstract ``transcribe`` method to provide
    backend-specific behaviour for speech-to-text.
    """

    def __init__(
        self,
        sampling_params: SamplingParams | None = None,
        language: str | None = None,
        prompt: str | None = None,
    ) -> None:
        super().__init__()
        self._sampling_params: SamplingParams = (
            sampling_params if sampling_params else {}
        )
        self._language: str | None = language
        self._prompt: str | None = prompt

    @property
    def sampling_params(self) -> SamplingParams:
        """Return the default sampling parameters."""
        return self._sampling_params

    @sampling_params.setter
    def sampling_params(self, params: SamplingParams) -> None:
        """Set the default sampling parameters."""
        self._sampling_params = params

    @property
    def language(self) -> str | None:
        """Return the language code."""
        return self._language

    @property
    def prompt(self) -> str | None:
        """Return the prompt."""
        return self._prompt

    @abstractmethod
    def transcribe(
        self,
        audio: bytes | str,
    ) -> str:
        """Transcribe an audio file.

        Args:
            audio: The audio data as raw bytes or a base64-encoded string.

        Returns:
            The transcribed text.

        """

        ...  # pragma: no cover


class WhisperCpp(AudioInferenceClient):
    """Client for a self-hosted whisper.cpp server.

    Communicates with the whisper.cpp HTTP server via ``multipart/form-data``
    POST requests to the ``/inference`` endpoint.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:8080",
        timeout: float = 5,
        inference_timeout: float | None = None,
        verify_ssl: bool = True,
        ca_cert: str | None = None,
        sampling_params: SamplingParams | None = None,
        language: str | None = None,
        prompt: str | None = None,
        max_retries: int = 3,
    ) -> None:
        """Initialise the WhisperCpp client.

        Args:
            host: Base URL of the whisper.cpp server (no trailing slash).
            timeout: Request timeout in seconds.
            inference_timeout: Optional per-inference timeout override.
            verify_ssl: Whether to verify SSL certificates.
            ca_cert: Optional path to a CA certificate bundle file.
            language: Optional language code (e.g. ``"en"``, ``"fr"``).
                When ``None`` the server will attempt to detect the language.
            prompt: Optional initial prompt to guide the transcription.
            max_retries: Maximum number of retries on server errors.
        """
        super().__init__(sampling_params, language, prompt)
        self.host = validate_host(host).rstrip("/")
        self.timeout = timeout
        self.inference_timeout = inference_timeout
        self.verify_ssl = verify_ssl
        self.ca_cert = ca_cert
        self.url = f"{self.host}/inference"
        self.headers: dict[str, str] = {
            "User-Agent": USER_AGENT,
        }
        self.max_retries = max_retries

    @property
    def _verify(self) -> bool | str:
        """Return the SSL verification setting for requests."""
        if self.ca_cert:
            return self.ca_cert
        return self.verify_ssl

    def _audio_bytes(self, audio: bytes | str) -> bytes:
        """Convert audio input to raw bytes.

        If a string is provided it is assumed to be a base64-encoded payload.
        """
        if isinstance(audio, str):
            if not audio.strip():
                raise ValueError("Base64-encoded audio string is empty")
            try:
                return base64.b64decode(audio, validate=True)
            except (binascii.Error, ValueError) as e:
                raise ValueError(f"Invalid base64-encoded audio: {e}") from e
        if len(audio) == 0:
            raise ValueError("Audio bytes data is empty")
        return audio

    @staticmethod
    def _backoff_delay(retries: int) -> float:
        """Compute exponential backoff with jitter."""
        return float(min(2**retries, 60)) + random.uniform(0, 1)

    def transcribe(
        self,
        audio: bytes | str,
    ) -> str:
        """Transcribe an audio file via the whisper.cpp ``/inference`` endpoint.

        Args:
            audio: The audio data as raw bytes or a base64-encoded string.

        Returns:
            The transcribed text.

        Raises:
            RuntimeError: On request/JSON failure or empty response.

        """
        retries = 0
        while True:
            try:
                return self._transcribe_once(audio)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                retries += 1
                if retries > self.max_retries:
                    raise RuntimeError(
                        f"WhisperCpp request failed after {self.max_retries} retries: {e}"
                    ) from e
                backoff = self._backoff_delay(retries)
                time.sleep(backoff)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                if status_code in {500, 502, 503, 504, 429}:
                    retries += 1
                    if retries > self.max_retries:
                        raise RuntimeError(
                            f"WhisperCpp request failed after {self.max_retries} retries: {e}"
                        ) from e
                    backoff = self._backoff_delay(retries)
                    time.sleep(backoff)
                else:
                    raise RuntimeError(f"WhisperCpp request failed: {e}") from e

    def _transcribe_once(
        self,
        audio: bytes | str,
    ) -> str:
        """Execute a single transcription request (no retry logic)."""
        timeout = self.inference_timeout or self.timeout
        audio_data = self._audio_bytes(audio)
        files: dict[str, Any] = {
            "file": ("audio.wav", audio_data, "application/octet-stream"),
        }
        data: dict[str, Any] = {}

        for key, value in self._sampling_params.items():
            data[key] = str(value)

        if self._language is not None:
            data["language"] = self._language
        else:
            data["language"] = "auto"

        if self._prompt is not None:
            data["prompt"] = self._prompt

        data["response_format"] = "json"

        resp = requests.post(
            self.url,
            headers=self.headers,
            files=files,
            data=data,
            timeout=timeout,
            verify=self._verify,
        )
        resp.raise_for_status()

        try:
            result: Any = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"WhisperCpp returned invalid JSON (status {resp.status_code}): "
                f"{resp.text[:200]}"
            ) from e

        if not isinstance(result, dict) or "text" not in result:
            raise RuntimeError(f"WhisperCpp response missing 'text' key: {result}")

        text = result["text"]
        if text is None:
            return ""
        return str(text)
