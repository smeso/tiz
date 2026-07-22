"""Tests for src/tiz/base_task_executor.py."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tiz.base_task_executor import (
    BaseTaskExecutor,
    TaskResources,
    _make_sandbox_name,
)
from tiz.conversion_sandbox import ConversionSandbox
from tiz.manifest_executor import ManifestExecutor
from tiz.manifest_parser import (
    DEFAULT_COLOR_INPUT,
    DEFAULT_COLOR_REASONING,
    ConfirmationSpec,
    InferenceEngineSpec,
    Manifest,
    ManifestMeta,
    PromptsAction,
    SubagentSpec,
    TaskSpec,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def _make_meta(**kwargs: Any) -> ManifestMeta:
    return ManifestMeta(
        version=kwargs.get("version", "1"),
        parallelism=kwargs.get("parallelism", 1),
        committer_name=kwargs.get("committer_name", "Tester"),
        committer_email=kwargs.get("committer_email", "tester@example.com"),
        container_engine=kwargs.get("container_engine"),
        color=kwargs.get("color", True),
        color_reasoning=kwargs.get("color_reasoning", DEFAULT_COLOR_REASONING),
        color_input=kwargs.get("color_input", DEFAULT_COLOR_INPUT),
        hide_reasoning=kwargs.get("hide_reasoning", False),
        use_host_timezone=kwargs.get("use_host_timezone", True),
        save_full_logs=kwargs.get("save_full_logs", False),
        save_full_toolcalls=kwargs.get("save_full_toolcalls", True),
        save_full_usage_details=kwargs.get("save_full_usage_details", True),
        summarizer_context_ratio=kwargs.get("summarizer_context_ratio", 0.9),
        verbosity=kwargs.get("verbosity", 0),
        delete_sandbox_on_exit=kwargs.get("delete_sandbox_on_exit", False),
        ephemeral_sandbox=kwargs.get("ephemeral_sandbox"),
    )


def _make_llamacpp_engine(**kwargs: Any) -> InferenceEngineSpec:
    return InferenceEngineSpec(
        engine_type=kwargs.get("engine_type", "llamacpp"),
        host=kwargs.get("host", "http://localhost:8080"),
        model=kwargs.get("model", "test-model"),
        api_key=kwargs.get("api_key", ""),
        name=kwargs.get("name", "test-engine"),
        timeout=kwargs.get("timeout", 5),
        verify_ssl=kwargs.get("verify_ssl", True),
        ca_cert=kwargs.get("ca_cert"),
        sampling_params=kwargs.get("sampling_params"),
        preserve_thinking=kwargs.get("preserve_thinking", False),
    )


def _make_dwarfstar4_engine(**kwargs: Any) -> InferenceEngineSpec:
    return InferenceEngineSpec(
        engine_type=kwargs.get("engine_type", "dwarfstar4"),
        host=kwargs.get("host", "http://localhost:9090"),
        model=kwargs.get("model", "test-ds4-model"),
        api_key=kwargs.get("api_key", "sk-test-ds4"),
        name=kwargs.get("name", "test-ds4-engine"),
        timeout=kwargs.get("timeout", 5),
        verify_ssl=kwargs.get("verify_ssl", True),
        ca_cert=kwargs.get("ca_cert"),
        sampling_params=kwargs.get("sampling_params"),
        message_timeout=kwargs.get("message_timeout"),
        preserve_thinking=kwargs.get("preserve_thinking", False),
    )


def _make_openrouter_engine(**kwargs: Any) -> InferenceEngineSpec:
    return InferenceEngineSpec(
        engine_type=kwargs.get("engine_type", "openrouter"),
        host=kwargs.get("host", ""),
        model=kwargs.get("model", "test-model"),
        api_key=kwargs.get("api_key", "sk-test"),
        name=kwargs.get("name", "test-engine"),
        timeout=kwargs.get("timeout", 5),
        verify_ssl=kwargs.get("verify_ssl", True),
        ca_cert=kwargs.get("ca_cert"),
        sampling_params=kwargs.get("sampling_params"),
        preserve_thinking=kwargs.get("preserve_thinking", False),
    )


def _make_anthropic_engine(**kwargs: Any) -> InferenceEngineSpec:
    return InferenceEngineSpec(
        engine_type=kwargs.get("engine_type", "anthropic"),
        host=kwargs.get("host", ""),
        model=kwargs.get("model", "claude-sonnet-5"),
        api_key=kwargs.get("api_key", "sk-ant-test"),
        name=kwargs.get("name", "test-anthropic-engine"),
        timeout=kwargs.get("timeout", 60),
        verify_ssl=kwargs.get("verify_ssl", True),
        ca_cert=kwargs.get("ca_cert"),
        sampling_params=kwargs.get("sampling_params"),
        message_timeout=kwargs.get("message_timeout"),
        preserve_thinking=kwargs.get("preserve_thinking", False),
    )


def _make_task(
    name: str = "test_task",
    tools: list[ToolSpec] | None = None,
    actions: list[Any] | None = None,
    inference_engine: str | None = None,
    allow_parallel_run: bool = True,
    **kwargs: Any,
) -> TaskSpec:
    return TaskSpec(
        name=name,
        worker_image=kwargs.get("worker_image", "tiz-worker:latest"),
        worker_image_containerfile=kwargs.get("worker_image_containerfile"),
        tools=tools or [],
        readonly_sandbox=kwargs.get("readonly_sandbox", False),
        project=kwargs.get("project"),
        sys_prompt=kwargs.get("sys_prompt"),
        sys_prompt_custom=kwargs.get("sys_prompt_custom"),
        actions=actions or [],
        allow_parallel_run=allow_parallel_run,
        force_copy_files=kwargs.get("force_copy_files", []),
        inference_engine=inference_engine,
        tmpfs_root=kwargs.get("tmpfs_root", False),
        subagents=kwargs.get("subagents", []),
        extra_container_args=kwargs.get("extra_container_args"),
    )


def _make_subagent_spec(
    name: str = "sub1",
    worker_image: str = "tiz-worker:latest",
    worker_image_containerfile: str | None = None,
    description: str | None = "A test sub-agent",
    sys_prompt: str | None = None,
    sys_prompt_custom: str | None = None,
    inference_engine: str | None = None,
    tools: list[ToolSpec] | None = None,
) -> SubagentSpec:
    return SubagentSpec(
        name=name,
        worker_image=worker_image,
        worker_image_containerfile=worker_image_containerfile,
        description=description,
        sys_prompt=sys_prompt,
        sys_prompt_custom=sys_prompt_custom,
        inference_engine=inference_engine,
        tools=tools or [],
    )


def _make_manifest(
    tasks: list[TaskSpec] | None = None,
    engines: list[InferenceEngineSpec] | None = None,
    **meta_kwargs: Any,
) -> Manifest:
    return Manifest(
        meta=_make_meta(**meta_kwargs),
        tasks=tasks or [],
        inference_engines=engines or [],
    )


def _make_executor(
    manifest: Manifest,
    base_path: Path,
    containerfiles_dirs: list[Path] | None = None,
) -> ManifestExecutor:
    executor = ManifestExecutor(manifest=manifest, base_path=base_path)
    if containerfiles_dirs is not None:
        executor._containerfiles_dirs = containerfiles_dirs
    return executor


@pytest.fixture
def mock_chat() -> Any:
    with patch("tiz.manifest_executor.Chat") as m:
        instance = MagicMock()
        instance.send_message.return_value = {
            "message": "ok",
            "prompt_tokens": 10,
            "prompt_time": 0.1,
            "completion_tokens": 20,
            "completion_time": 0.2,
            "cost": 0.0,
            "tool_calls": ["tc"],
        }
        m.return_value = instance
        yield m, instance


@pytest.fixture
def mock_discover_tools() -> Any:
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools", return_value={}
    ) as m:
        yield m


@pytest.fixture
def mock_sleep() -> Any:
    with patch("time.sleep") as m:
        yield m


# ---------------------------------------------------------------------------
# _make_sandbox_name
# ---------------------------------------------------------------------------


def test_make_sandbox_name_simple() -> None:
    assert _make_sandbox_name("hello") == "hello"


def test_make_sandbox_name_special_chars() -> None:
    assert _make_sandbox_name("My Task!") == "my_task_"


def test_make_sandbox_name_uppercase() -> None:
    assert _make_sandbox_name("HELLO") == "hello"


def test_make_sandbox_name_spaces() -> None:
    assert _make_sandbox_name("my task name") == "my_task_name"


def test_make_sandbox_name_mixed() -> None:
    result = _make_sandbox_name("Test-123@#$")
    assert result == "test_123___"


# ---------------------------------------------------------------------------
# _cleanup_resources (instance method)
# ---------------------------------------------------------------------------


def _make_cleanup_executor(
    tmp_path: Path, delete_on_exit: bool = False
) -> ManifestExecutor:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        delete_sandbox_on_exit=delete_on_exit,
    )
    return ManifestExecutor(manifest=manifest, base_path=tmp_path)


def test_cleanup_resources_both_set(tmp_path: Path) -> None:
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=False)
    manager = MagicMock()
    sandbox = MagicMock()
    executor._cleanup_resources(manager, "my-sandbox", sandbox)
    manager.kill_all_containers.assert_called_once_with("my-sandbox")
    sandbox.validate_git_project_dir.assert_called_once()


def test_cleanup_resources_manager_none(tmp_path: Path) -> None:
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=False)
    sandbox = MagicMock()
    executor._cleanup_resources(None, None, sandbox)
    sandbox.validate_git_project_dir.assert_called_once()
    sandbox.kill_all_containers.assert_not_called()


def test_cleanup_resources_sandbox_none(tmp_path: Path) -> None:
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=False)
    manager = MagicMock()
    executor._cleanup_resources(manager, "s1", None)
    manager.kill_all_containers.assert_called_once_with("s1")


def test_cleanup_resources_both_none(tmp_path: Path) -> None:
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=False)
    executor._cleanup_resources(None, None, None)


def test_cleanup_resources_delete_on_exit_both_set(tmp_path: Path) -> None:
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=True)
    manager = MagicMock()
    sandbox = MagicMock()
    executor._cleanup_resources(manager, "my-sandbox", sandbox)
    sandbox.sync_to_original_auto_rebase.assert_called_once_with()
    manager.kill_and_delete_sandbox.assert_called_once_with("my-sandbox")
    manager.kill_all_containers.assert_not_called()
    sandbox.validate_git_project_dir.assert_not_called()


def test_cleanup_resources_delete_on_exit_manager_none(tmp_path: Path) -> None:
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=True)
    sandbox = MagicMock()
    executor._cleanup_resources(None, None, sandbox)
    sandbox.sync_to_original_auto_rebase.assert_called_once_with()
    sandbox.validate_git_project_dir.assert_not_called()


def test_cleanup_resources_delete_on_exit_sandbox_name_none(tmp_path: Path) -> None:
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=True)
    manager = MagicMock()
    executor._cleanup_resources(manager, None, None)
    manager.kill_and_delete_sandbox.assert_not_called()


def test_cleanup_resources_delete_on_exit_both_none(tmp_path: Path) -> None:
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=True)
    executor._cleanup_resources(None, None, None)


def test_cleanup_resources_validate_git_oserror_logged(
    tmp_path: Path,
) -> None:
    """validate_git_project_dir exception in _cleanup_resources propagates."""
    executor = _make_cleanup_executor(tmp_path, delete_on_exit=False)
    manager = MagicMock()
    sandbox = MagicMock()
    sandbox.validate_git_project_dir.side_effect = ValueError("git validation error")
    with pytest.raises(ValueError, match="git validation error"):
        executor._cleanup_resources(manager, "my-sandbox", sandbox)
    manager.kill_all_containers.assert_called_once_with("my-sandbox")
    sandbox.validate_git_project_dir.assert_called_once()


# build_client  (static)
# ---------------------------------------------------------------------------


def test_build_client_llamacpp() -> None:
    with patch("tiz.base_task_executor.LlamaCpp") as mock_cls:
        engine = _make_llamacpp_engine()
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once_with(
        host="http://localhost:8080",
        timeout=5,
        message_timeout=None,
        verify_ssl=True,
        ca_cert=None,
        default_model="test-model",
        sampling_params=None,
        preserve_thinking=False,
    )


def test_build_client_llamacpp_no_model() -> None:
    with patch("tiz.base_task_executor.LlamaCpp") as mock_cls:
        engine = _make_llamacpp_engine(model=None)
        ManifestExecutor.build_client(engine)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["default_model"] == ""


def test_build_client_llamacpp_llama_cpp_variant() -> None:
    with patch("tiz.base_task_executor.LlamaCpp") as mock_cls:
        engine = _make_llamacpp_engine(engine_type="llama.cpp")
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()


def test_build_client_llamacpp_sampling_params() -> None:
    with patch("tiz.base_task_executor.LlamaCpp") as mock_cls:
        engine = _make_llamacpp_engine(sampling_params={"temperature": 0.7})
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["sampling_params"] == {"temperature": 0.7}


def test_build_client_openrouter() -> None:
    with patch("tiz.base_task_executor.OpenRouter") as mock_cls:
        engine = _make_openrouter_engine()
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once_with(
        api_key="sk-test",
        timeout=5,
        message_timeout=None,
        verify_ssl=True,
        ca_cert=None,
        default_model="test-model",
        sampling_params=None,
        preserve_thinking=False,
    )


def test_build_client_openrouter_no_model() -> None:
    with patch("tiz.base_task_executor.OpenRouter") as mock_cls:
        engine = _make_openrouter_engine(model=None)
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["default_model"] == "openrouter/free"


def test_build_client_openrouter_sampling_params() -> None:
    with patch("tiz.base_task_executor.OpenRouter") as mock_cls:
        engine = _make_openrouter_engine(sampling_params={"top_p": 0.9})
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["sampling_params"] == {"top_p": 0.9}


def test_build_client_llamacpp_preserve_thinking() -> None:
    with patch("tiz.base_task_executor.LlamaCpp") as mock_cls:
        engine = _make_llamacpp_engine(preserve_thinking=True)
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["preserve_thinking"] is True


def test_build_client_openrouter_preserve_thinking() -> None:
    with patch("tiz.base_task_executor.OpenRouter") as mock_cls:
        engine = _make_openrouter_engine(preserve_thinking=True)
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["preserve_thinking"] is True


def test_build_client_dwarfstar4() -> None:
    with patch("tiz.base_task_executor.DwarfStar4") as mock_cls:
        engine = _make_dwarfstar4_engine()
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once_with(
        host="http://localhost:9090",
        timeout=5,
        message_timeout=None,
        verify_ssl=True,
        ca_cert=None,
        default_model="test-ds4-model",
        sampling_params=None,
        api_key="sk-test-ds4",
    )


def test_build_client_dwarfstar4_no_model() -> None:
    with patch("tiz.base_task_executor.DwarfStar4") as mock_cls:
        engine = _make_dwarfstar4_engine(model=None)
        ManifestExecutor.build_client(engine)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["default_model"] == ""


def test_build_client_dwarfstar4_ds4_abbreviation() -> None:
    with patch("tiz.base_task_executor.DwarfStar4") as mock_cls:
        engine = _make_dwarfstar4_engine(engine_type="ds4")
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()


def test_build_client_dwarfstar4_sampling_params() -> None:
    with patch("tiz.base_task_executor.DwarfStar4") as mock_cls:
        engine = _make_dwarfstar4_engine(sampling_params={"temperature": 0.5})
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["sampling_params"] == {"temperature": 0.5}


def test_build_client_unknown_engine() -> None:
    engine = _make_llamacpp_engine(engine_type="unknown")
    with pytest.raises(ValueError, match="Unknown inference engine type"):
        ManifestExecutor.build_client(engine)


def test_build_client_anthropic() -> None:
    with patch("tiz.base_task_executor.AnthropicClient") as mock_cls:
        engine = _make_anthropic_engine()
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once_with(
        api_key="sk-ant-test",
        default_model="claude-sonnet-5",
        sampling_params=None,
        preserve_thinking=False,
        timeout=60,
        message_timeout=None,
    )


def test_build_client_anthropic_claude_variant() -> None:
    with patch("tiz.base_task_executor.AnthropicClient") as mock_cls:
        engine = _make_anthropic_engine(engine_type="claude")
        ManifestExecutor.build_client(engine)
    mock_cls.assert_called_once()


def test_build_client_anthropic_custom_model() -> None:
    with patch("tiz.base_task_executor.AnthropicClient") as mock_cls:
        engine = _make_anthropic_engine(model="claude-opus-4")
        ManifestExecutor.build_client(engine)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["default_model"] == "claude-opus-4"


def test_build_client_anthropic_no_model_fallback() -> None:
    with patch("tiz.base_task_executor.AnthropicClient") as mock_cls:
        engine = _make_anthropic_engine(model=None)
        ManifestExecutor.build_client(engine)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["default_model"] == "claude-sonnet-5"


def test_build_client_anthropic_sampling_params() -> None:
    with patch("tiz.base_task_executor.AnthropicClient") as mock_cls:
        engine = _make_anthropic_engine(sampling_params={"temperature": 0.3})
        ManifestExecutor.build_client(engine)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["sampling_params"] == {"temperature": 0.3}


def test_build_client_anthropic_preserve_thinking() -> None:
    with patch("tiz.base_task_executor.AnthropicClient") as mock_cls:
        engine = _make_anthropic_engine(preserve_thinking=True)
        ManifestExecutor.build_client(engine)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["preserve_thinking"] is True


def test_build_client_anthropic_custom_timeout() -> None:
    with patch("tiz.base_task_executor.AnthropicClient") as mock_cls:
        engine = _make_anthropic_engine(timeout=120, message_timeout=300)
        ManifestExecutor.build_client(engine)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["timeout"] == 120
    assert call_kwargs["message_timeout"] == 300


def test_build_client_anthropic_no_api_key() -> None:
    with patch("tiz.base_task_executor.AnthropicClient") as mock_cls:
        engine = _make_anthropic_engine(api_key=None)
        ManifestExecutor.build_client(engine)
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["api_key"] == ""


# ---------------------------------------------------------------------------
# _send_with_retry
# ---------------------------------------------------------------------------


def test_send_with_retry_success_first_try() -> None:
    chat = MagicMock()
    chat.send_message.return_value = {"message": "ok"}
    result = ManifestExecutor._send_with_retry(chat, "hello")
    assert result == {"message": "ok"}
    chat.send_message.assert_called_once_with("hello", timeout=None)


def test_send_with_retry_success_with_timeout() -> None:
    chat = MagicMock()
    chat.send_message.return_value = {"message": "ok"}
    result = ManifestExecutor._send_with_retry(chat, "hello", timeout=30.0)
    assert result == {"message": "ok"}
    chat.send_message.assert_called_once_with("hello", timeout=30.0)


def test_send_with_retry_fails_then_replay_succeeds(mock_sleep: Any) -> None:  # noqa: ARG001
    chat = MagicMock()
    chat.send_message.side_effect = RuntimeError("first fail")
    chat.replay.return_value = {"message": "recovered"}
    result = ManifestExecutor._send_with_retry(chat, "hello", max_retries=3)
    assert result == {"message": "recovered"}
    chat.send_message.assert_called_once_with("hello", timeout=None)
    assert chat.replay.call_count == 1


def test_send_with_retry_fails_all_attempts(mock_sleep: Any) -> None:  # noqa: ARG001
    chat = MagicMock()
    chat.send_message.side_effect = RuntimeError("first fail")
    chat.replay.side_effect = [
        RuntimeError("retry 1 fail"),
        RuntimeError("retry 2 fail"),
        RuntimeError("retry 3 fail"),
    ]
    with pytest.raises(RuntimeError, match="retry 3 fail"):
        ManifestExecutor._send_with_retry(chat, "hello", max_retries=3)
    assert chat.send_message.call_count == 1
    assert chat.replay.call_count == 3


def test_send_with_retry_fails_on_second_retry_then_succeeds(mock_sleep: Any) -> None:  # noqa: ARG001
    chat = MagicMock()
    chat.send_message.side_effect = RuntimeError("first fail")
    chat.replay.side_effect = [
        RuntimeError("retry 1 fail"),
        RuntimeError("retry 2 fail"),
        {"message": "recovered"},
    ]
    result = ManifestExecutor._send_with_retry(chat, "hello", max_retries=3)
    assert result == {"message": "recovered"}
    assert chat.send_message.call_count == 1
    assert chat.replay.call_count == 3


def test_send_with_retry_passes_timeout_to_replay(mock_sleep: Any) -> None:  # noqa: ARG001
    chat = MagicMock()
    chat.send_message.side_effect = RuntimeError("first fail")
    chat.replay.return_value = {"message": "ok"}
    result = ManifestExecutor._send_with_retry(
        chat, "hello", timeout=15.0, max_retries=1
    )
    assert result == {"message": "ok"}
    chat.send_message.assert_called_once_with("hello", timeout=15.0)
    chat.replay.assert_called_once_with(timeout=15.0)


def test_send_with_retry_use_replay_success_first_try(mock_sleep: Any) -> None:  # noqa: ARG001
    chat = MagicMock()
    chat.replay.return_value = {"message": "replayed"}
    result = ManifestExecutor._send_with_retry(chat, "irrelevant", use_replay=True)
    assert result == {"message": "replayed"}
    chat.replay.assert_called_once_with(timeout=None)
    chat.send_message.assert_not_called()


def test_send_with_retry_use_replay_retries_on_failure(mock_sleep: Any) -> None:  # noqa: ARG001
    chat = MagicMock()
    chat.replay.side_effect = [
        RuntimeError("first fail"),
        RuntimeError("second fail"),
        {"message": "recovered"},
    ]
    result = ManifestExecutor._send_with_retry(
        chat, "irrelevant", max_retries=3, use_replay=True
    )
    assert result == {"message": "recovered"}
    assert chat.replay.call_count == 3
    chat.send_message.assert_not_called()


def test_send_with_retry_use_replay_all_fail(mock_sleep: Any) -> None:  # noqa: ARG001
    chat = MagicMock()
    chat.replay.side_effect = [
        RuntimeError("fail 1"),
        RuntimeError("fail 2"),
        RuntimeError("fail 3"),
        RuntimeError("fail 4"),
    ]
    with pytest.raises(RuntimeError, match="fail 4"):
        ManifestExecutor._send_with_retry(
            chat, "irrelevant", max_retries=3, use_replay=True
        )
    assert chat.replay.call_count == 4  # first try + 3 retries
    chat.send_message.assert_not_called()


def test_send_with_retry_use_replay_passes_timeout(mock_sleep: Any) -> None:  # noqa: ARG001
    chat = MagicMock()
    chat.replay.return_value = {"message": "ok"}
    result = ManifestExecutor._send_with_retry(
        chat, "irrelevant", timeout=42.0, max_retries=1, use_replay=True
    )
    assert result == {"message": "ok"}
    chat.replay.assert_called_once_with(timeout=42.0)
    chat.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# _create_default_client
# ---------------------------------------------------------------------------


def test_create_default_client(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine()
    engine2 = _make_openrouter_engine()
    manifest = _make_manifest(engines=[engine, engine2])
    with patch.object(ManifestExecutor, "build_client") as mock_build:
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
        assert executor._default_client is mock_client
        mock_build.assert_called_once_with(engine)


def test_create_default_client_no_engines_raises(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[])
    with pytest.raises(ValueError, match="No inference engines configured"):
        ManifestExecutor(manifest=manifest, base_path=tmp_path)


# ---------------------------------------------------------------------------
# _get_client_for_task
# ---------------------------------------------------------------------------


def test_get_client_default_engine_none(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine()
    manifest = _make_manifest(engines=[engine])
    with patch.object(ManifestExecutor, "build_client") as mock_build:
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
        mock_build.reset_mock()

        task = _make_task(inference_engine=None)
        client = executor._get_client_for_task(task)
        assert client is mock_client


def test_get_client_named_found(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine(name="my-llama")
    engine2 = _make_llamacpp_engine(name="my-llama2")
    manifest = _make_manifest(engines=[engine2, engine])
    with patch.object(ManifestExecutor, "build_client") as mock_build:
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
        mock_build.reset_mock()

        task = _make_task(inference_engine="my-llama")
        client = executor._get_client_for_task(task)
        assert client is mock_client
        mock_build.assert_called_once_with(engine)


def test_get_client_named_not_found(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine(name="my-llama")
    manifest = _make_manifest(engines=[engine])
    with patch.object(ManifestExecutor, "build_client") as mock_build:
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

        task = _make_task(inference_engine="nonexistent")
        with pytest.raises(ValueError, match="not found in manifest"):
            executor._get_client_for_task(task)


def test_get_client_no_engines_and_default_client() -> None:
    manifest = _make_manifest()
    mock_client = MagicMock()
    executor = ManifestExecutor.__new__(ManifestExecutor)
    executor.manifest = manifest
    executor._engine = None
    executor._update_callback = None
    executor._task_usage = {}
    executor._task_usage_lock = threading.Lock()
    executor._jinja_env = MagicMock()
    executor._default_client = mock_client

    task = _make_task()
    client = executor._get_client_for_task(task)
    assert client is mock_client


# ---------------------------------------------------------------------------
# _accumulate_usage / task_usage
# ---------------------------------------------------------------------------


def test_accumulate_usage_new_task(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine()
    manifest = _make_manifest(engines=[engine])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._accumulate_usage(
        "task1",
        {
            "prompt_tokens": 100,
            "prompt_time": 1.5,
            "completion_tokens": 200,
            "completion_time": 2.5,
            "cost": 0.05,
            "tool_calls": ["call1"],
        },
    )

    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 100
    assert usage["task1"]["prompt_time"] == 1.5
    assert usage["task1"]["completion_tokens"] == 200
    assert usage["task1"]["completion_time"] == 2.5
    assert usage["task1"]["cost"] == 0.05
    assert usage["task1"]["tool_calls"] == ["call1"]


def test_accumulate_usage_existing_task(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine()
    manifest = _make_manifest(engines=[engine])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._accumulate_usage(
        "task1",
        {
            "prompt_tokens": 100,
            "prompt_time": 1.0,
            "completion_tokens": 50,
            "completion_time": 0.5,
            "cost": 0.01,
            "tool_calls": ["call1"],
        },
    )
    executor._accumulate_usage(
        "task1",
        {
            "prompt_tokens": 200,
            "prompt_time": 2.0,
            "completion_tokens": 100,
            "completion_time": 1.0,
            "cost": 0.02,
            "tool_calls": ["call2"],
        },
    )

    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 300
    assert usage["task1"]["prompt_time"] == 3.0
    assert usage["task1"]["completion_tokens"] == 150
    assert usage["task1"]["completion_time"] == 1.5
    assert usage["task1"]["cost"] == 0.03
    assert usage["task1"]["tool_calls"] == ["call1", "call2"]


def test_accumulate_usage_missing_keys(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine()
    manifest = _make_manifest(engines=[engine])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._accumulate_usage("task1", {})
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 0
    assert usage["task1"]["tool_calls"] == []


def test_accumulate_usage_no_tool_calls_in_result(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine()
    manifest = _make_manifest(engines=[engine])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._accumulate_usage("task1", {"prompt_tokens": 10})
    usage = executor.task_usage
    assert usage["task1"]["tool_calls"] == []


def test_accumulate_usage_no_tool_calls_key_in_usage(tmp_path: Path) -> None:
    """When a task entry exists but has no tool_calls key, one is created."""
    engine = _make_llamacpp_engine()
    manifest = _make_manifest(engines=[engine])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._task_usage["task1"] = {
        "test-engine": {
            "prompt_tokens": 5,
            "prompt_time": 0.0,
            "completion_tokens": 0,
            "completion_time": 0.0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "cost": 0,
        },
    }
    executor._accumulate_usage(
        "task1",
        {
            "prompt_tokens": 10,
            "tool_calls": ["x"],
        },
    )
    usage = executor.task_usage
    assert usage["task1"]["tool_calls"] == ["x"]


def test_accumulate_usage_with_explicit_engine_name(tmp_path: Path) -> None:
    """_accumulate_usage with explicit engine_name uses it instead of looking up engine."""
    engine = _make_llamacpp_engine(name="e1")
    manifest = _make_manifest(
        tasks=[_make_task(name="t1", inference_engine="e1")],
        engines=[engine],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._accumulate_usage(
        "t1",
        {"prompt_tokens": 100, "completion_tokens": 200, "cost": 0.05},
        engine_name="custom-engine",
    )

    usage = executor.task_usage
    assert usage["t1"]["prompt_tokens"] == 100
    assert usage["t1"]["completion_tokens"] == 200
    assert usage["t1"]["cost"] == 0.05


def test_task_usage_returns_copy(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine()
    manifest = _make_manifest(engines=[engine])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._accumulate_usage("task1", {"prompt_tokens": 10})
    usage1 = executor.task_usage
    executor._accumulate_usage("task1", {"prompt_tokens": 20})
    usage2 = executor.task_usage

    assert usage1["task1"]["prompt_tokens"] == 10
    assert usage2["task1"]["prompt_tokens"] == 30


# ---------------------------------------------------------------------------
# _save_full_usage
# ---------------------------------------------------------------------------


def test_save_full_usage_writes_log_when_enabled(
    mock_chat: Any,
    mock_discover_tools: Any,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0.001,
        "tool_calls": ["tc"],
    }
    task = _make_task(name="t1", actions=[PromptsAction(message_groups=[["hi"]])])
    manifest = _make_manifest(
        tasks=[task],
        engines=[_make_llamacpp_engine()],
        save_full_usage_details=True,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    usage_dir = tmp_path / "logs" / "usage"
    assert usage_dir.is_dir()
    log_files = list(usage_dir.glob("*.log"))
    assert len(log_files) == 1
    assert "test_engine" in str(log_files[0])
    assert "t1" in str(log_files[0])
    data = json.loads(log_files[0].read_text(encoding="utf-8"))
    assert data["prompt_tokens"] == 1
    assert data["completion_tokens"] == 2
    assert "tool_calls" not in data


def test_save_full_usage_skipped_when_disabled(
    mock_chat: Any,
    mock_discover_tools: Any,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0.001,
        "tool_calls": ["tc"],
    }
    task = _make_task(name="t1", actions=[PromptsAction(message_groups=[["hi"]])])
    manifest = _make_manifest(
        tasks=[task],
        engines=[_make_llamacpp_engine()],
        save_full_usage_details=False,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    usage_dir = tmp_path / "logs" / "usage"
    assert not usage_dir.is_dir()


def test_save_full_usage_splits_per_task_and_engine(
    mock_chat: Any,
    mock_discover_tools: Any,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 10,
        "prompt_time": 0.5,
        "completion_tokens": 20,
        "completion_time": 1.0,
        "cost": 0.01,
        "tool_calls": ["tc1", "tc2"],
    }
    task1 = _make_task(
        name="t1", actions=[PromptsAction(message_groups=[["a"], ["b"]])]
    )
    task2 = _make_task(name="t2", actions=[PromptsAction(message_groups=[["c"]])])
    manifest = _make_manifest(
        tasks=[task1, task2],
        engines=[_make_llamacpp_engine()],
        parallelism=2,
        save_full_usage_details=True,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    usage_dir = tmp_path / "logs" / "usage"
    log_files = list(usage_dir.glob("*.log"))
    assert len(log_files) == 2
    filenames = [f.name for f in log_files]
    for fn in filenames:
        assert fn.endswith(".log")
        assert "test_engine" in fn
    t1_files = [f for f in log_files if "_t1." in f.name]
    t2_files = [f for f in log_files if "_t2." in f.name]
    assert len(t1_files) == 1
    assert len(t2_files) == 1
    data1 = json.loads(t1_files[0].read_text(encoding="utf-8"))
    data2 = json.loads(t2_files[0].read_text(encoding="utf-8"))
    assert data1["prompt_tokens"] == 20
    assert data1["completion_tokens"] == 40
    assert data1["cost"] == 0.02
    assert "tool_calls" not in data1
    assert data2["prompt_tokens"] == 10
    assert data2["completion_tokens"] == 20
    assert data2["cost"] == 0.01
    assert "tool_calls" not in data2


def test_save_full_usage_direct_call(tmp_path: Path) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine(name="my-custom-engine")],
        save_full_usage_details=True,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "t1",
        {
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0.002,
            "tool_calls": ["tc"],
        },
    )
    executor._save_full_usage()

    usage_dir = tmp_path / "logs" / "usage"
    assert usage_dir.is_dir()
    log_files = list(usage_dir.glob("*.log"))
    assert len(log_files) == 1
    assert "my_custom_engine" in str(log_files[0])
    assert "_t1." in log_files[0].name
    data = json.loads(log_files[0].read_text(encoding="utf-8"))
    assert data == {
        "prompt_tokens": 5,
        "prompt_time": 0.1,
        "completion_tokens": 10,
        "completion_time": 0.2,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "cost": 0.002,
    }
    assert "tool_calls" not in data


def test_save_full_usage_disabled_direct_call(tmp_path: Path) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        save_full_usage_details=False,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "t1",
        {
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0.002,
            "tool_calls": ["tc"],
        },
    )
    executor._save_full_usage()

    usage_dir = tmp_path / "logs" / "usage"
    assert not usage_dir.is_dir()


def test_save_full_usage_mkdir_oserror(caplog: Any, tmp_path: Path) -> None:
    """OSError on mkdir in _save_full_usage is caught and logged."""
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_usage_details=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "t1", {"prompt_tokens": 5, "prompt_time": 0.1, "completion_tokens": 10}
    )
    with patch.object(Path, "mkdir", side_effect=OSError(13, "Permission denied")):
        caplog.clear()
        executor._save_full_usage()
        assert "Failed to create usage log dir" in caplog.text
        assert not (tmp_path / "logs" / "usage").exists()


def test_save_full_usage_write_oserror(caplog: Any, tmp_path: Path) -> None:
    """OSError on file write in _save_full_usage is caught and logged."""
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_usage_details=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "t1", {"prompt_tokens": 5, "prompt_time": 0.1, "completion_tokens": 10}
    )
    usage_dir = tmp_path / "logs" / "usage"
    usage_dir.mkdir(parents=True, exist_ok=True)
    with patch.object(Path, "open", side_effect=OSError(13, "Permission denied")):
        caplog.clear()
        executor._save_full_usage()
        assert "Failed to write usage log" in caplog.text


# ---------------------------------------------------------------------------
# _discover_tools
# ---------------------------------------------------------------------------


def test_discover_tools_builtin() -> None:
    tools = BaseTaskExecutor._discover_tools()
    assert isinstance(tools, dict)
    assert len(tools) > 0
    for name, cls in tools.items():
        assert isinstance(name, str)
        assert isinstance(cls, type)


def test_discover_tools_with_paths() -> None:
    tools = BaseTaskExecutor._discover_tools(paths=[Path("/nonexistent")])
    assert isinstance(tools, dict)
    assert len(tools) > 0


def test_discover_tools_with_user_dir(tmp_path: Path) -> None:
    user_dir = tmp_path / "user_tools"
    user_dir.mkdir()
    tool_file = user_dir / "mytool.py"
    tool_file.write_text(
        "from tiz.tools.base import SocketTool\n"
        "class MyTool(SocketTool):\n"
        "    @staticmethod\n"
        "    def fname(): return 'my_tool'\n"
        "    @staticmethod\n"
        "    def prompt(): return '{}'\n"
        "    def run(self, args): return 'ok'\n"
    )
    tools = BaseTaskExecutor._discover_tools(paths=[user_dir])
    assert len(tools) > 0
    assert "my_tool" in tools


# ---------------------------------------------------------------------------
# _discover_tools_from_dir
# ---------------------------------------------------------------------------


def test_discover_tools_from_dir_not_a_dir(tmp_path: Path) -> None:
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path / "no_such_dir")
    assert result == {}


def test_discover_tools_from_dir_empty(tmp_path: Path) -> None:
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
    assert result == {}


def test_discover_tools_from_dir_with_underscore_prefix(tmp_path: Path) -> None:
    f = tmp_path / "_private.py"
    f.write_text("from tiz.tools.base import SocketTool\nclass X(SocketTool): pass\n")
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
    assert result == {}


def test_discover_tools_from_dir_already_imported(tmp_path: Path) -> None:
    import sys

    mod_name = "user_tools.already_loaded"
    sys.modules[mod_name] = MagicMock()
    try:
        f = tmp_path / "already_loaded.py"
        f.write_text(
            "from tiz.tools.base import SocketTool\n"
            "class AlreadyLoaded(SocketTool):\n"
            "    @staticmethod\n"
            "    def fname(): return 'already_loaded'\n"
            "    @staticmethod\n"
            "    def prompt(): return '{}'\n"
            "    def run(self, args): return 'ok'\n"
        )
        result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
        assert "already_loaded" not in result
    finally:
        sys.modules.pop(mod_name, None)


def test_discover_tools_from_dir_invalid_spec(tmp_path: Path) -> None:
    """A file name that produces None for spec_from_file_location."""
    f = tmp_path / "bad_mod.py"
    f.write_text("this is invalid python syntax >>>>>>")
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
    assert isinstance(result, dict)


def test_discover_tools_from_dir_module_load_error(tmp_path: Path) -> None:
    f = tmp_path / "broken.py"
    f.write_text("raise RuntimeError('nope')\n")
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
    assert result == {}


def test_discover_tools_from_dir_finds_tool(tmp_path: Path) -> None:
    f = tmp_path / "valid_tool.py"
    f.write_text(
        "from tiz.tools.base import SocketTool\n"
        "class ValidTool(SocketTool):\n"
        "    @staticmethod\n"
        "    def fname(): return 'valid_tool'\n"
        "    @staticmethod\n"
        "    def prompt(): import json; return json.dumps({'name': 'valid_tool'})\n"
        "    def run(self, args): return 'ok'\n"
    )
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
    assert "valid_tool" in result


def test_discover_tools_from_dir_skips_non_subclasses(tmp_path: Path) -> None:
    f = tmp_path / "not_a_tool.py"
    f.write_text("class Something: pass\n")
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
    assert result == {}


def test_discover_tools_from_dir_skips_socket_tool_base(tmp_path: Path) -> None:
    f = tmp_path / "just_base.py"
    f.write_text("from tiz.tools.base import SocketTool\n")
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
    assert result == {}


def test_discover_tools_from_dir_skips_private_attrs(tmp_path: Path) -> None:
    f = tmp_path / "tools.py"
    f.write_text(
        "from tiz.tools.base import SocketTool\n"
        "class _PrivateTool(SocketTool):\n"
        "    @staticmethod\n"
        "    def fname(): return '_private'\n"
        "    @staticmethod\n"
        "    def prompt(): return '{}'\n"
        "    def run(self, args): return 'ok'\n"
    )
    result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
    assert "_private" not in result


def test_discover_tools_from_dir_spec_loader_none(tmp_path: Path) -> None:
    """A file whose spec has a None loader triggers the spec.loader is None branch."""
    import importlib.util as iu

    f = tmp_path / "no_loader.py"
    f.write_text("pass\n")

    original_spec_from_file_location = iu.spec_from_file_location

    def fake_spec(*args: Any, **kwargs: Any) -> Any:
        s = original_spec_from_file_location(*args, **kwargs)
        if s is not None:
            s.loader = None
        return s

    with patch.object(iu, "spec_from_file_location", side_effect=fake_spec):
        result = BaseTaskExecutor._discover_tools_from_dir(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# _get_engine_name
# ---------------------------------------------------------------------------


def test_get_engine_name_from_task_inference_engine(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="e1")
    engine2 = _make_llamacpp_engine(name="e2")
    manifest = _make_manifest(
        tasks=[_make_task(name="t1", inference_engine="e2")],
        engines=[engine1, engine2],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    assert executor._get_engine_name("t1") == "e2"


def test_get_engine_name_falls_back_to_first_engine(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="e1")
    manifest = _make_manifest(
        tasks=[_make_task(name="t1", inference_engine=None)],
        engines=[engine1],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    assert executor._get_engine_name("t1") == "e1"


def test_get_engine_name_no_engines_returns_unknown() -> None:
    executor = ManifestExecutor.__new__(ManifestExecutor)
    executor._base_path = Path("/tmp")
    executor.manifest = _make_manifest(
        tasks=[_make_task(name="t1", inference_engine=None)]
    )
    assert executor._get_engine_name("t1") == "unknown"


# ---------------------------------------------------------------------------
# _save_chat_log
# ---------------------------------------------------------------------------


def test_save_chat_log_skipped_when_save_full_logs_false(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=False)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    chat = MagicMock()
    executor._save_chat_log(chat, "task1")
    # _save_conv_log is not called, so no file is created
    assert not (tmp_path / "logs").exists()


def test_save_chat_log_writes_file(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    chat = MagicMock()
    conv = [{"role": "user", "content": "hello"}]
    chat.conv = conv

    executor._save_chat_log(chat, "mytask")
    log_dir = tmp_path / "logs" / "conversations"
    files = list(log_dir.iterdir())
    assert len(files) == 1
    filepath = files[0]
    assert "mytask" in filepath.name
    assert "test_engine" in filepath.name
    assert filepath.name.endswith(".json")
    with filepath.open() as f:
        data = json.load(f)
    assert data == conv


def test_save_chat_log_creates_directory(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    chat = MagicMock()
    chat.conv = [{"role": "user", "content": "hi"}]

    assert not (tmp_path / "logs").exists()
    executor._save_chat_log(chat, "task1")
    assert (tmp_path / "logs" / "conversations").is_dir()
    assert len(list((tmp_path / "logs" / "conversations").iterdir())) == 1


def test_save_chat_log_with_engine_name_from_task(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="my-custom-engine")
    manifest = _make_manifest(
        save_full_logs=True,
        tasks=[_make_task(name="t1", inference_engine="my-custom-engine")],
        engines=[engine1],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    chat = MagicMock()
    chat.conv = [{"role": "user", "content": "hello"}]

    executor._save_chat_log(chat, "t1")
    log_dir = tmp_path / "logs" / "conversations"
    files = list(log_dir.iterdir())
    assert len(files) == 1
    assert "my_custom_engine" in files[0].name


def test_save_chat_log_with_special_task_name(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="test-engine")
    manifest = _make_manifest(
        save_full_logs=True,
        tasks=[_make_task(name="My Special Task!")],
        engines=[engine1],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    chat = MagicMock()
    chat.conv = [{"role": "user", "content": "hello"}]

    executor._save_chat_log(chat, "My Special Task!")
    log_dir = tmp_path / "logs" / "conversations"
    files = list(log_dir.iterdir())
    assert len(files) == 1
    assert "my_special_task_" in files[0].name


# ---------------------------------------------------------------------------
# _save_toolcalls_log
# ---------------------------------------------------------------------------


def test_save_toolcalls_log_skipped_when_save_full_toolcalls_false(
    tmp_path: Path,
) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_toolcalls=False
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "task1", {"tool_calls": [("read_file", {"path": "/tmp"})]}
    )
    executor._save_toolcalls_log("task1")
    assert not (tmp_path / "logs" / "tools").exists()


def test_save_toolcalls_log_no_tool_calls_does_not_create_file(tmp_path: Path) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_toolcalls=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._save_toolcalls_log("task1")
    tools_dir = tmp_path / "logs" / "tools"
    assert tools_dir.is_dir()
    assert len(list(tools_dir.iterdir())) == 0


def test_save_toolcalls_log_writes_file(tmp_path: Path) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_toolcalls=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "mytask",
        {"tool_calls": [("read_file", {"path": "/tmp"}), ("bash", {"command": "ls"})]},
    )
    executor._save_toolcalls_log("mytask")
    tools_dir = tmp_path / "logs" / "tools"
    files = list(tools_dir.iterdir())
    assert len(files) == 1
    filepath = files[0]
    assert "mytask" in str(filepath)
    assert "test_engine" in str(filepath)
    assert str(filepath).endswith(".json")
    with filepath.open() as f:
        data = json.load(f)
    assert data == [["read_file", {"path": "/tmp"}], ["bash", {"command": "ls"}]]


def test_save_toolcalls_log_creates_directory(tmp_path: Path) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_toolcalls=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage("task1", {"tool_calls": [("edit", {"old_string": "a"})]})
    assert not (tmp_path / "logs").exists()
    executor._save_toolcalls_log("task1")
    assert (tmp_path / "logs" / "tools").is_dir()


def test_save_toolcalls_log_with_engine_name_from_task(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="my-custom-engine")
    manifest = _make_manifest(
        save_full_toolcalls=True,
        tasks=[_make_task(name="t1", inference_engine="my-custom-engine")],
        engines=[engine1],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage("t1", {"tool_calls": [("grep", {"pattern": "foo"})]})
    executor._save_toolcalls_log("t1")
    files = list((tmp_path / "logs" / "tools").iterdir())
    assert len(files) == 1
    assert "my_custom_engine" in str(files[0])


def test_save_toolcalls_log_with_special_task_name(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="test-engine")
    manifest = _make_manifest(
        save_full_toolcalls=True,
        tasks=[_make_task(name="My Special Task!")],
        engines=[engine1],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "My Special Task!", {"tool_calls": [("glob", {"pattern": "*.py"})]}
    )
    executor._save_toolcalls_log("My Special Task!")
    files = list((tmp_path / "logs" / "tools").iterdir())
    assert len(files) == 1
    assert "my_special_task_" in str(files[0])


def test_save_toolcalls_log_empty_tool_calls_list_does_not_write(
    tmp_path: Path,
) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_toolcalls=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage("task1", {"tool_calls": []})
    executor._save_toolcalls_log("task1")
    tools_dir = tmp_path / "logs" / "tools"
    assert tools_dir.is_dir()
    assert len(list(tools_dir.iterdir())) == 0


def test_save_toolcalls_log_task_no_usage_does_not_write(tmp_path: Path) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_toolcalls=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._save_toolcalls_log("nonexistent")
    tools_dir = tmp_path / "logs" / "tools"
    assert tools_dir.is_dir()
    assert len(list(tools_dir.iterdir())) == 0


def test_save_toolcalls_log_write_oserror(caplog: Any, tmp_path: Path) -> None:
    """OSError on file write in _save_toolcalls_log is caught and logged."""
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_toolcalls=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "task1", {"tool_calls": [("read_file", {"path": "/tmp"})]}
    )
    tools_dir = tmp_path / "logs" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(Path, "open", side_effect=OSError(13, "Permission denied")):
        caplog.clear()
        executor._save_toolcalls_log("task1")
        assert "Failed to write toolcalls log" in caplog.text


def test_save_toolcalls_log_mkdir_oserror(caplog: Any, tmp_path: Path) -> None:
    """OSError on mkdir in _save_toolcalls_log is caught and logged."""
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], save_full_toolcalls=True
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._accumulate_usage(
        "task1", {"tool_calls": [("read_file", {"path": "/tmp"})]}
    )
    with patch.object(Path, "mkdir", side_effect=OSError(13, "Permission denied")):
        caplog.clear()
        executor._save_toolcalls_log("task1")
        assert "Failed to create toolcalls log dir" in caplog.text
        assert not (tmp_path / "logs" / "tools").exists()


# ---------------------------------------------------------------------------
# resolve_prompt
# ---------------------------------------------------------------------------


def test_resolve_prompt_default(tmp_path: Path) -> None:
    """resolve_prompt returns default when sys_prompt is None and no custom prompt."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task()
    result = executor.resolve_prompt(task)
    assert result.startswith("You are Tiz")


def test_resolve_prompt_empty_spec(tmp_path: Path) -> None:
    """resolve_prompt treats empty sys_prompt string like None, falling back to 'coding'."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt="")
    result = executor.resolve_prompt(task)
    assert result.startswith("You are Tiz")


def test_resolve_prompt_custom(tmp_path: Path) -> None:
    """resolve_prompt uses sys_prompt_custom when provided."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="Be terse.")
    result = executor.resolve_prompt(task)
    assert result == "Be terse."


def test_resolve_prompt_custom_with_template(tmp_path: Path) -> None:
    """resolve_prompt renders jinja2 template from sys_prompt_custom."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="Hello {{committer_name}}")
    result = executor.resolve_prompt(task)
    assert result == "Hello Tester"


def test_resolve_prompt_custom_jinja_undefined_fallback(tmp_path: Path) -> None:
    """resolve_prompt uses jinja2 silent Undefined, missing vars become empty."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="Hello {{undefined_var}}")
    result = executor.resolve_prompt(task)
    assert result == "Hello {{undefined_var}}"


def test_resolve_prompt_spec_from_file(tmp_path: Path) -> None:
    """resolve_prompt loads sys_prompt from a file in _prompts_dirs."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "my_prompt.txt"
    prompt_file.write_text("I am a test prompt.")

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._prompts_dirs = [prompts_dir]
    task = _make_task(sys_prompt="my_prompt")
    result = executor.resolve_prompt(task)
    assert result == "I am a test prompt."


def test_resolve_prompt_spec_j2_template(tmp_path: Path) -> None:
    """resolve_prompt renders .j2 template file."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "tmpl.j2"
    prompt_file.write_text("Hello {{committer_name}}")

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._prompts_dirs = [prompts_dir]
    task = _make_task(sys_prompt="tmpl")
    result = executor.resolve_prompt(task)
    assert result == "Hello Tester"


def test_resolve_prompt_spec_from_plain(tmp_path: Path) -> None:
    """resolve_prompt returns plain file contents without rendering."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "tmpl"
    prompt_file.write_text("Hello {{committer_name}}")

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._prompts_dirs = [prompts_dir]
    task = _make_task(sys_prompt="tmpl")
    result = executor.resolve_prompt(task)
    assert result == "Hello {{committer_name}}"


def test_resolve_prompt_spec_from_txt(tmp_path: Path) -> None:
    """resolve_prompt returns .txt file contents without rendering."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "tmpl.txt"
    prompt_file.write_text("Hello {{committer_name}}")

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._prompts_dirs = [prompts_dir]
    task = _make_task(sys_prompt="tmpl")
    result = executor.resolve_prompt(task)
    assert result == "Hello {{committer_name}}"


def test_resolve_prompt_spec_not_found(tmp_path: Path) -> None:
    """resolve_prompt returns default when sys_prompt file not found."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._prompts_dirs = [prompts_dir]
    task = _make_task(sys_prompt="nonexistent")
    result = executor.resolve_prompt(task)
    assert result == "You are a helpful assistant."


def test_resolve_prompt_from_data_dir(tmp_path: Path) -> None:
    """resolve_prompt finds prompt file from the tiz/data/prompts dir."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    assert len(executor._prompts_dirs) >= 2
    data_prompts_dir = executor._prompts_dirs[1]
    assert data_prompts_dir.is_dir()

    task = _make_task(sys_prompt="coding")
    result = executor.resolve_prompt(task)
    assert result != "You are a helpful assistant."
    assert len(result) > 0


def test_containerfile_from_data_dir(tmp_path: Path) -> None:
    """Containerfile lookup uses the data dir when base_path has no match."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    assert len(executor._containerfiles_dirs) >= 2
    data_cf_dir = executor._containerfiles_dirs[1]
    assert data_cf_dir.is_dir()
    cf_file = data_cf_dir / "Containerfile.tiz-worker"
    assert cf_file.is_file()


def test_resolve_prompt_with_sandbox_context(tmp_path: Path) -> None:
    """resolve_prompt with sandbox sets git_repo context."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    sandbox = MagicMock()
    type(sandbox).project_dir = property(lambda _self: project_dir)  # type: ignore[assignment]
    sandbox.is_git_repo.return_value = False

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="git={{git_repo}}")
    result = executor.resolve_prompt(task, sandbox)
    assert "git=no" in result


def test_resolve_prompt_with_agents_md(tmp_path: Path) -> None:
    """resolve_prompt reads AGENTS.md from sandbox project dir and passes it to context."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    agents_md = project_dir / "AGENTS.md"
    agents_md.write_text("Custom agents content", encoding="utf-8")

    sandbox = MagicMock()
    type(sandbox).project_dir = property(lambda _self: project_dir)  # type: ignore[assignment]

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="{{git_repo}} {{AGENTS_md}}")
    result = executor.resolve_prompt(task, sandbox)
    assert result == "no Custom agents content"


def test_resolve_prompt_custom_empty_string(tmp_path: Path) -> None:
    """resolve_prompt treats empty custom string like None."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="")
    result = executor.resolve_prompt(task)
    assert result.startswith("You are Tiz")


def test_resolve_prompt_with_sandbox_no_project(tmp_path: Path) -> None:
    """resolve_prompt handles sandbox with non-existent project_dir gracefully."""
    sandbox = MagicMock()
    type(sandbox).project_dir = property(lambda _self: tmp_path / "nonexistent")  # type: ignore[assignment]

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="git={{git_repo}}")
    result = executor.resolve_prompt(task, sandbox)
    assert "git=no" in result


def test_resolve_prompt_without_sandbox(tmp_path: Path) -> None:
    """resolve_prompt with sandbox=None sets git_repo=no."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="git={{git_repo}}")
    result = executor.resolve_prompt(task)
    assert "git=no" in result


def test_resolve_prompt_with_open_git_repo_context(tmp_path: Path) -> None:
    """resolve_prompt detects a git repo in the project dir."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    import subprocess

    subprocess.run(["git", "init"], cwd=project_dir, capture_output=True, check=True)

    sandbox = MagicMock()
    type(sandbox).project_dir = property(lambda _self: project_dir)  # type: ignore[assignment]

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(sys_prompt_custom="git={{git_repo}}")
    result = executor.resolve_prompt(task, sandbox)
    assert "git=yes" in result


def test_resolve_prompt_tools_access_context(tmp_path: Path) -> None:
    sandbox = MagicMock()
    type(sandbox).project_dir = property(lambda _self: tmp_path / "nonexistent")  # type: ignore[assignment]

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(
        sys_prompt_custom="wp={{tools_with_project_access}} wop={{tools_without_project_access}} wi={{tools_with_internet_access}} woi={{tools_without_internet_access}}",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="bash", network="internet", disk_mode="nodisk"),
            ToolSpec(name="grep", network="none", disk_mode="ro-disk"),
        ],
    )
    result = executor.resolve_prompt(task, sandbox)
    assert "wp=read_file, grep" in result
    assert "wop=bash" in result
    assert "wi=bash" in result
    assert "woi=read_file, grep" in result


def test_resolve_prompt_tools_access_context_none(tmp_path: Path) -> None:
    sandbox = MagicMock()
    type(sandbox).project_dir = property(lambda _self: tmp_path / "nonexistent")  # type: ignore[assignment]

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(
        sys_prompt_custom="wp={{tools_with_project_access is defined}}",
        tools=[],
    )
    result = executor.resolve_prompt(task, sandbox)
    assert result == "wp=False"


def test_resolve_prompt_tools_deduplication(tmp_path: Path) -> None:
    """resolve_prompt skips duplicate tool names when building context."""
    sandbox = MagicMock()
    type(sandbox).project_dir = property(lambda _self: tmp_path / "nonexistent")  # type: ignore[assignment]

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(
        sys_prompt_custom="wp={{tools_with_project_access}}",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
        ],
    )
    result = executor.resolve_prompt(task, sandbox)
    assert result == "wp=read_file"


def test_resolve_prompt_tools_access_context_undefined_without_tools(
    tmp_path: Path,
) -> None:
    sandbox = MagicMock()
    type(sandbox).project_dir = property(lambda _self: tmp_path / "nonexistent")  # type: ignore[assignment]

    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    task = _make_task(
        sys_prompt_custom="wp={{tools_with_project_access is defined}}",
    )
    result = executor.resolve_prompt(task, sandbox)
    assert "wp=False" in result


# ---------------------------------------------------------------------------
# _create_task_resources
# ---------------------------------------------------------------------------


def test_create_task_resources_no_tools(tmp_path: Path) -> None:
    task = _make_task(name="no_tools_task", tools=[])
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    resources = executor._create_task_resources(task=task)
    assert isinstance(resources, TaskResources)
    assert resources.tool_instances == []
    assert resources.tools_confirmations == {}
    assert resources.sandbox is None
    assert resources.manager is None
    assert resources.sandbox_name is None
    assert resources.action_lock is None
    assert resources.conversion_sandbox is None


def test_create_task_resources_no_tools_default_task(tmp_path: Path) -> None:
    """When task=None, falls back to manifest.tasks[0]."""
    task = _make_task(name="default_task", tools=[])
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    resources = executor._create_task_resources(task=None)
    assert isinstance(resources, TaskResources)
    assert resources.tool_instances == []
    assert resources.tools_confirmations == {}
    assert resources.sandbox is None
    assert resources.manager is None
    assert resources.sandbox_name is None
    assert resources.action_lock is None
    assert resources.conversion_sandbox is None


def test_create_task_resources_no_tools_with_conversion_container(
    tmp_path: Path,
) -> None:
    task = _make_task(name="conv_no_tools_task", tools=[])
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        container_engine="podman",
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch(
            "tiz.base_task_executor.BaseTaskExecutor._discover_tools", return_value={}
        ),
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        conversion_container_mock = MagicMock()
        conversion_container_mock.worker_socket_path = "/tmp/conv_sock"
        conversion_container_mock.shared_dir = tmp_path / "conversion_shared"
        mgr_instance.create_container.return_value = conversion_container_mock
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx

        resources = executor._create_task_resources(
            task=task,
            create_conversion_container=True,
        )

    assert resources.tool_instances == []
    assert resources.tools_confirmations == {}
    assert resources.sandbox is sandbox_dirs
    assert resources.manager is mgr_instance
    assert resources.sandbox_name == "conv_no_tools_task"
    assert resources.conversion_sandbox is not None
    assert isinstance(resources.conversion_sandbox, ConversionSandbox)
    assert mgr_instance.create_sandbox.call_count == 1
    assert mgr_instance.create_container.call_count == 1
    conv_call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert conv_call_kwargs["network"] == "none"
    assert conv_call_kwargs["mount_project"] is False
    assert conv_call_kwargs["read_only_project"] is True
    assert resources.action_lock is lock_ctx


def test_create_task_resources_no_tools_no_subagents_skips_setup(
    tmp_path: Path,
) -> None:
    """When no tools and no subagents, sandbox is skipped."""
    task = _make_task(name="no_tools_no_sub", tools=[], subagents=[])
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    resources = executor._create_task_resources(task=task)
    assert resources.tool_instances == []
    assert resources.tools_confirmations == {}
    assert resources.sandbox is None
    assert resources.manager is None
    assert resources.sandbox_name is None
    assert resources.action_lock is None
    assert resources.conversion_sandbox is None


def test_create_task_resources_with_engine(tmp_path: Path) -> None:
    task = _make_task(
        name="engine_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        container_engine="podman",
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        resources = executor._create_task_resources(task=task)

    assert resources.sandbox_name == "engine_task"
    mock_mgr_cls.assert_called_once_with(base_path=tmp_path, engine="podman")
    mgr_instance.create_sandbox.assert_called_once()
    mgr_instance.create_container.assert_called_once()
    assert len(resources.tool_instances) == 1
    assert resources.sandbox is sandbox_dirs
    assert resources.manager is mgr_instance
    assert resources.conversion_sandbox is None


def test_create_task_resources_with_conversion_container(
    tmp_path: Path,
) -> None:
    task = _make_task(
        name="conv_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        container_engine="podman",
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        container_mock.shared_dir = None
        conversion_container_mock = MagicMock()
        conversion_container_mock.worker_socket_path = "/tmp/conv_sock"
        conversion_container_mock.shared_dir = tmp_path / "conversion_shared"
        mgr_instance.create_container.side_effect = [
            container_mock,
            conversion_container_mock,
        ]
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        resources = executor._create_task_resources(
            task=task,
            create_conversion_container=True,
        )

    assert resources.sandbox_name == "conv_task"
    assert len(resources.tool_instances) == 1
    assert resources.sandbox is sandbox_dirs
    assert resources.manager is mgr_instance
    assert resources.conversion_sandbox is not None
    assert isinstance(resources.conversion_sandbox, ConversionSandbox)
    assert mgr_instance.create_container.call_count == 2
    conv_call_kwargs = mgr_instance.create_container.call_args_list[1].kwargs
    assert conv_call_kwargs["network"] == "none"
    assert conv_call_kwargs["mount_project"] is False
    assert conv_call_kwargs["read_only_project"] is True


def test_create_task_resources_with_conversion_container_extra_args(
    tmp_path: Path,
) -> None:
    task = _make_task(
        name="conv_extra_args_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
        extra_container_args=["--cap-add=NET_ADMIN"],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        container_engine="podman",
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        container_mock.shared_dir = None
        conversion_container_mock = MagicMock()
        conversion_container_mock.worker_socket_path = "/tmp/conv_sock"
        conversion_container_mock.shared_dir = tmp_path / "conversion_shared"
        mgr_instance.create_container.side_effect = [
            container_mock,
            conversion_container_mock,
        ]
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        resources = executor._create_task_resources(
            task=task,
            create_conversion_container=True,
        )

    assert resources.sandbox_name == "conv_extra_args_task"
    assert len(resources.tool_instances) == 1
    assert resources.sandbox is sandbox_dirs
    assert resources.manager is mgr_instance
    assert resources.conversion_sandbox is not None
    assert isinstance(resources.conversion_sandbox, ConversionSandbox)
    assert mgr_instance.create_container.call_count == 2
    # Tool container gets extra_run_args
    tool_call_kwargs = mgr_instance.create_container.call_args_list[0].kwargs
    assert tool_call_kwargs["extra_run_args"] == ["--cap-add=NET_ADMIN"]
    # Conversion container gets extra_run_args
    conv_call_kwargs = mgr_instance.create_container.call_args_list[1].kwargs
    assert conv_call_kwargs["network"] == "none"
    assert conv_call_kwargs["mount_project"] is False
    assert conv_call_kwargs["read_only_project"] is True
    assert conv_call_kwargs["extra_run_args"] == ["--cap-add=NET_ADMIN"]


def test_create_task_resources_conversion_container_no_socket(
    tmp_path: Path,
) -> None:
    task = _make_task(
        name="conv_nosock_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        container_engine="podman",
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        conversion_container_mock = MagicMock()
        conversion_container_mock.worker_socket_path = None
        mgr_instance.create_container.side_effect = [
            container_mock,
            conversion_container_mock,
        ]
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        resources = executor._create_task_resources(
            task=task,
            create_conversion_container=True,
        )

    assert resources.conversion_sandbox is None


def test_create_task_resources_conversion_container_socket_but_no_shared_dir(
    tmp_path: Path,
) -> None:
    """Conversion container has a socket but no shared_dir - ConversionSandbox should be None."""
    task = _make_task(
        name="conv_noshared_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        container_engine="podman",
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        conversion_container_mock = MagicMock()
        conversion_container_mock.worker_socket_path = "/tmp/conv_sock"
        conversion_container_mock.shared_dir = None
        mgr_instance.create_container.side_effect = [
            container_mock,
            conversion_container_mock,
        ]
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        resources = executor._create_task_resources(
            task=task,
            create_conversion_container=True,
        )

    assert resources.conversion_sandbox is None


def test_create_task_resources_engine_none_falls_back(tmp_path: Path) -> None:
    task = _make_task(
        name="fallback_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        mock_mgr_cls.available_engine.return_value = "docker"
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    mock_mgr_cls.assert_called_once()
    engine_arg = mock_mgr_cls.call_args.kwargs["engine"]
    assert engine_arg == "docker"


def test_create_task_resources_with_containerfile(tmp_path: Path) -> None:
    task = _make_task(
        name="cf_task",
        worker_image_containerfile="FROM ubuntu",
        worker_image="tiz-worker-img:latest",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    mgr_instance.build_image.assert_called_once_with(
        containerfile="FROM ubuntu",
        tag="tiz-worker-img:latest",
        delete_existing=False,
    )


def test_create_task_resources_containerfile_from_dir(tmp_path: Path) -> None:
    task = _make_task(
        name="cf_dir_task",
        worker_image="custom-img:latest",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path, containerfiles_dirs=[])

    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir()
    cf_file = cf_dir / "Containerfile.custom-img:latest"
    cf_file.write_text("FROM alpine")
    executor._containerfiles_dirs = [cf_dir]

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    mgr_instance.build_image.assert_called_once_with(
        containerfile="FROM alpine",
        tag="custom-img:latest",
        delete_existing=False,
    )


def test_create_task_resources_containerfile_not_found(tmp_path: Path) -> None:
    task = _make_task(
        name="cf_not_found",
        worker_image="nonexistent:latest",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir()
    executor = _make_executor(manifest, tmp_path, containerfiles_dirs=[cf_dir])

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    mgr_instance.build_image.assert_not_called()


def test_create_task_resources_containerfile_loopback(tmp_path: Path) -> None:
    task = _make_task(
        name="cf_loopback",
        worker_image="myworker:latest",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    cf_dir_1 = tmp_path / "cf_empty"
    cf_dir_1.mkdir()
    cf_dir_2 = tmp_path / "cf_data"
    cf_dir_2.mkdir()
    cf_path = cf_dir_2 / "Containerfile.myworker:latest"
    cf_path.write_text("FROM alpine")
    executor = _make_executor(
        manifest, tmp_path, containerfiles_dirs=[cf_dir_1, cf_dir_2]
    )

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    mgr_instance.build_image.assert_called_once_with(
        containerfile="FROM alpine",
        tag="myworker:latest",
        delete_existing=False,
    )


def test_create_task_resources_readonly_sandbox(tmp_path: Path) -> None:
    task = _make_task(
        name="ro_task",
        readonly_sandbox=True,
        tools=[ToolSpec(name="read_file", network="none", disk_mode="ro-disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    mgr_instance.create_container.assert_called_once()
    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["read_only_project"] is True
    assert call_kwargs["mount_project"] is True
    assert call_kwargs["use_host_timezone"] is True


def test_create_task_resources_readonly_sandbox_with_disk_mode_raises(
    tmp_path: Path,
) -> None:
    """readonly_sandbox=True with disk_mode='disk' raises ValueError."""
    tool_cls = MagicMock()
    tool_cls.fname.return_value = "write_file"

    task = _make_task(
        name="ro_disk_task",
        readonly_sandbox=True,
        tools=[ToolSpec(name="write_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        mock_disc.return_value = {"write_file": tool_cls}

        with pytest.raises(
            ValueError,
            match="readonly_sandbox=True cannot have tools with disk_mode='disk'",
        ):
            executor._create_task_resources(task=task)

    mgr_instance.kill_all_containers.assert_called_once_with("ro_disk_task")


def test_create_task_resources_extra_container_args(tmp_path: Path) -> None:
    task = _make_task(
        name="extra_args_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
        extra_container_args=["--cap-add=NET_ADMIN"],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    mgr_instance.create_container.assert_called_once()
    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["extra_run_args"] == ["--cap-add=NET_ADMIN"]


def test_create_task_resources_writable_sandbox_disk_mode(
    tmp_path: Path,
) -> None:
    task = _make_task(
        name="rw_task",
        readonly_sandbox=False,
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["read_only_project"] is False
    assert call_kwargs["mount_project"] is True
    assert call_kwargs["use_host_timezone"] is True


def test_create_task_resources_ro_disk_mode(tmp_path: Path) -> None:
    task = _make_task(
        name="ro_disk_task",
        readonly_sandbox=False,
        tools=[ToolSpec(name="read_file", network="none", disk_mode="ro-disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["read_only_project"] is True
    # ro-disk -> mount_project should be True (project is mounted, but read-only)
    assert call_kwargs["mount_project"] is True


def test_create_task_resources_nodisk_mode(tmp_path: Path) -> None:
    task = _make_task(
        name="nodisk_task",
        readonly_sandbox=False,
        tools=[ToolSpec(name="bash", network="internet", disk_mode="nodisk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "bash"
        mock_disc.return_value = {"bash": tool_cls}

        executor._create_task_resources(task=task)

    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["read_only_project"] is True
    assert call_kwargs["mount_project"] is False
    assert call_kwargs["use_host_timezone"] is True


def test_create_task_resources_multiple_combos(tmp_path: Path) -> None:
    task = _make_task(
        name="multi_combo_task",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="bash", network="internet", disk_mode="nodisk"),
        ],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls_rf = MagicMock()
        tool_cls_rf.fname.return_value = "read_file"
        tool_cls_bash = MagicMock()
        tool_cls_bash.fname.return_value = "bash"
        mock_disc.return_value = {"read_file": tool_cls_rf, "bash": tool_cls_bash}

        resources = executor._create_task_resources(task=task)

    assert mgr_instance.create_container.call_count == 2
    call1_kwargs = mgr_instance.create_container.call_args_list[0].kwargs
    call2_kwargs = mgr_instance.create_container.call_args_list[1].kwargs
    combos = {
        (c["network"], c["read_only_project"], c["mount_project"])
        for c in [call1_kwargs, call2_kwargs]
    }
    assert combos == {("none", False, True), ("internet", True, False)}
    assert len(resources.tool_instances) == 2
    for c in [call1_kwargs, call2_kwargs]:
        assert c["use_host_timezone"] is True


def test_create_task_resources_shared_combo_single_container(
    tmp_path: Path,
) -> None:
    task = _make_task(
        name="shared_combo_task",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="write_file", network="none", disk_mode="disk"),
        ],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls_rf = MagicMock()
        tool_cls_rf.fname.return_value = "read_file"
        tool_instance_rf = MagicMock()
        tool_instance_rf.socket_path = "/tmp/sock"
        tool_cls_rf.return_value = tool_instance_rf
        tool_cls_wf = MagicMock()
        tool_cls_wf.fname.return_value = "write_file"
        tool_instance_wf = MagicMock()
        tool_instance_wf.socket_path = "/tmp/sock"
        tool_cls_wf.return_value = tool_instance_wf
        mock_disc.return_value = {
            "read_file": tool_cls_rf,
            "write_file": tool_cls_wf,
        }

        resources = executor._create_task_resources(task=task)

    assert mgr_instance.create_container.call_count == 1
    assert len(resources.tool_instances) == 2
    assert resources.tool_instances[0].socket_path == "/tmp/sock"
    assert resources.tool_instances[1].socket_path == "/tmp/sock"


def test_create_task_resources_project_and_force_copy(tmp_path: Path) -> None:
    task = _make_task(
        name="proj_task",
        project="/some/project",
        force_copy_files=["*.env", "config/*"],
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    call_kwargs = mgr_instance.create_sandbox.call_args.kwargs
    assert call_kwargs["project_path"] == "/some/project"
    assert call_kwargs["force_copy_files"] == ["*.env", "config/*"]
    assert call_kwargs["committer_name"] == "Tester"
    assert call_kwargs["committer_email"] == "tester@example.com"


def test_create_task_resources_project_none(tmp_path: Path) -> None:
    task = _make_task(
        name="no_proj_task",
        project=None,
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    call_kwargs = mgr_instance.create_sandbox.call_args.kwargs
    assert call_kwargs["project_path"] is None


def test_create_task_resources_force_copy_none(tmp_path: Path) -> None:
    task = _make_task(
        name="no_copy_task",
        force_copy_files=[],
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        executor._create_task_resources(task=task)

    call_kwargs = mgr_instance.create_sandbox.call_args.kwargs
    assert call_kwargs["force_copy_files"] is None


def test_create_task_resources_subagents_rejected(tmp_path: Path) -> None:
    """SubAgents tool in task.tools raises ValueError."""
    tool_cls = MagicMock()
    tool_cls.fname.return_value = "SubAgents"

    task = _make_task(
        name="subagents_task",
        tools=[ToolSpec(name="SubAgents", network="none", disk_mode="nodisk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch(
            "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
            return_value={"SubAgents": tool_cls},
        ),
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock

        with pytest.raises(
            ValueError,
            match="SubAgents is a special tool that cannot be used as a regular tool",
        ):
            executor._create_task_resources(task=task)

    mgr_instance.kill_all_containers.assert_called_once_with("subagents_task")
    sandbox_dirs.validate_git_project_dir.assert_called_once()


def test_create_task_resources_tool_not_discovered(tmp_path: Path) -> None:
    task = _make_task(
        name="missing_tool_task",
        tools=[ToolSpec(name="nonexistent_tool", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch(
            "tiz.base_task_executor.BaseTaskExecutor._discover_tools", return_value={}
        ),
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock

        with pytest.raises(ValueError, match="Unknown tool"):
            executor._create_task_resources(task=task)

    mgr_instance.kill_all_containers.assert_called_once_with("missing_tool_task")


def test_create_task_resources_socket_path_none_raises(tmp_path: Path) -> None:
    task = _make_task(
        name="sock_none_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = None
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        with pytest.raises(ValueError, match="No container socket found for tool"):
            executor._create_task_resources(task=task)

    mgr_instance.kill_all_containers.assert_called_once_with("sock_none_task")


def test_create_task_resources_create_sandbox_raises(tmp_path: Path) -> None:
    task = _make_task(
        name="sandbox_fail_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        mgr_instance.create_sandbox.side_effect = RuntimeError(
            "sandbox creation failed"
        )
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        with pytest.raises(RuntimeError, match="sandbox creation failed"):
            executor._create_task_resources(task=task)

    mgr_instance.kill_all_containers.assert_called_once_with("sandbox_fail_task")


def test_create_task_resources_create_container_raises(tmp_path: Path) -> None:
    task = _make_task(
        name="container_fail_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        mgr_instance.create_container.side_effect = RuntimeError("container failed")
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        with pytest.raises(RuntimeError, match="container failed"):
            executor._create_task_resources(task=task)

    mgr_instance.kill_all_containers.assert_called_once_with("container_fail_task")
    sandbox_dirs.validate_git_project_dir.assert_called_once()


def test_create_task_resources_validate_git_called_on_raise(
    tmp_path: Path,
) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        mgr_instance.create_container.side_effect = [
            container_mock,
            RuntimeError("second container failed"),
        ]
        tool_cls_rf = MagicMock()
        tool_cls_rf.fname.return_value = "read_file"
        tool_cls_bash = MagicMock()
        tool_cls_bash.fname.return_value = "bash"
        mock_disc.return_value = {"read_file": tool_cls_rf, "bash": tool_cls_bash}

        task2 = _make_task(
            name="validate_task",
            tools=[
                ToolSpec(name="read_file", network="none", disk_mode="disk"),
                ToolSpec(name="bash", network="internet", disk_mode="nodisk"),
            ],
            actions=[PromptsAction(message_groups=[["x"]])],
        )

        with pytest.raises(RuntimeError, match="second container failed"):
            executor._create_task_resources(task=task2)

    mgr_instance.kill_all_containers.assert_called_once_with("validate_task")
    sandbox_dirs.validate_git_project_dir.assert_called_once()


def test_create_task_resources_validate_git_exception_logged(
    caplog: Any,
    tmp_path: Path,
) -> None:
    """When validate_git_project_dir raises in except block, the original exception is preserved."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        sandbox_dirs.validate_git_project_dir.side_effect = ValueError(
            "git validation failed"
        )
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        mgr_instance.create_container.side_effect = [
            container_mock,
            RuntimeError("second container failed"),
        ]
        tool_cls_rf = MagicMock()
        tool_cls_rf.fname.return_value = "read_file"
        tool_cls_bash = MagicMock()
        tool_cls_bash.fname.return_value = "bash"
        mock_disc.return_value = {"read_file": tool_cls_rf, "bash": tool_cls_bash}

        task2 = _make_task(
            name="validate_fail_task",
            tools=[
                ToolSpec(name="read_file", network="none", disk_mode="disk"),
                ToolSpec(name="bash", network="internet", disk_mode="nodisk"),
            ],
            actions=[PromptsAction(message_groups=[["x"]])],
        )

        caplog.clear()
        with pytest.raises(RuntimeError, match="second container failed"):
            executor._create_task_resources(task=task2)

    mgr_instance.kill_all_containers.assert_called_once_with("validate_fail_task")
    sandbox_dirs.validate_git_project_dir.assert_called_once()
    assert "Failed to validate git project dir during cleanup" in caplog.text


def test_create_task_resources_two_containers_two_tools(tmp_path: Path) -> None:
    task = _make_task(
        name="two_containers",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="bash", network="internet", disk_mode="nodisk"),
        ],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        sock1 = MagicMock()
        sock1.worker_socket_path = "/tmp/sock1"
        sock2 = MagicMock()
        sock2.worker_socket_path = "/tmp/sock2"
        mgr_instance.create_container.side_effect = [sock1, sock2]
        tool_cls_rf = MagicMock()
        tool_cls_rf.fname.return_value = "read_file"
        tool_instance_rf = MagicMock()
        tool_instance_rf.socket_path = "/tmp/sock1"
        tool_cls_rf.return_value = tool_instance_rf
        tool_cls_bash = MagicMock()
        tool_cls_bash.fname.return_value = "bash"
        tool_instance_bash = MagicMock()
        tool_instance_bash.socket_path = "/tmp/sock2"
        tool_cls_bash.return_value = tool_instance_bash
        mock_disc.return_value = {"read_file": tool_cls_rf, "bash": tool_cls_bash}

        resources = executor._create_task_resources(task=task)

    assert mgr_instance.create_container.call_count == 2
    assert len(resources.tool_instances) == 2
    assert resources.tool_instances[0].socket_path == "/tmp/sock1"
    assert resources.tool_instances[1].socket_path == "/tmp/sock2"


def test_create_task_resources_ephemeral_sandbox_name(tmp_path: Path) -> None:
    """When ephemeral_sandbox is True, sandbox_name is a random base64 string."""
    task = _make_task(
        name="eph_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(
        tasks=[task],
        engines=[_make_llamacpp_engine()],
        ephemeral_sandbox=True,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        resources = executor._create_task_resources(task=task)

    assert resources.sandbox_name is not None
    assert len(resources.sandbox_name) > 0


def test_create_task_resources_sandbox_name_special_chars(
    tmp_path: Path,
) -> None:
    task = _make_task(
        name="Special Task!@#",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        resources = executor._create_task_resources(task=task)

    assert resources.sandbox_name == "special_task___"
    mgr_instance.create_container.assert_called_once()
    assert (
        mgr_instance.create_container.call_args.kwargs["sandbox_name"]
        == "special_task___"
    )


def test_create_task_resources_deduplicates_combos(tmp_path: Path) -> None:
    task = _make_task(
        name="dedup_task",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="write_file", network="none", disk_mode="disk"),
            ToolSpec(name="edit", network="none", disk_mode="disk"),
        ],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls_rf = MagicMock()
        tool_cls_rf.fname.return_value = "read_file"
        tool_instance_rf = MagicMock()
        tool_instance_rf.socket_path = "/tmp/sock"
        tool_cls_rf.return_value = tool_instance_rf
        tool_cls_wf = MagicMock()
        tool_cls_wf.fname.return_value = "write_file"
        tool_instance_wf = MagicMock()
        tool_instance_wf.socket_path = "/tmp/sock"
        tool_cls_wf.return_value = tool_instance_wf
        tool_cls_edit = MagicMock()
        tool_cls_edit.fname.return_value = "edit"
        tool_instance_edit = MagicMock()
        tool_instance_edit.socket_path = "/tmp/sock"
        tool_cls_edit.return_value = tool_instance_edit
        mock_disc.return_value = {
            "read_file": tool_cls_rf,
            "write_file": tool_cls_wf,
            "edit": tool_cls_edit,
        }

        resources = executor._create_task_resources(task=task)

    assert mgr_instance.create_container.call_count == 1
    assert len(resources.tool_instances) == 3
    for tool in resources.tool_instances:
        assert tool.socket_path == "/tmp/sock"


def test_create_task_resources_verbosity_not_used(tmp_path: Path) -> None:
    """Verify verbosity setting has no effect on sandbox setup (regression check)."""
    task = _make_task(
        name="verbosity_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], verbosity=3)
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        mock_disc.return_value = {"read_file": tool_cls}

        resources = executor._create_task_resources(task=task)

    assert resources.sandbox_name == "verbosity_task"
    mgr_instance.create_container.assert_called_once()
    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["verbose"] == 3
    assert call_kwargs["use_host_timezone"] is True


def test_create_task_resources_build_image_raises(tmp_path: Path) -> None:
    task = _make_task(
        name="build_fail_task",
        worker_image_containerfile="FROM ubuntu",
        worker_image="tiz-worker:latest",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch(
            "tiz.base_task_executor.BaseTaskExecutor._discover_tools", return_value={}
        ),
    ):
        mgr_instance = MagicMock()
        mgr_instance.build_image.side_effect = RuntimeError("build failed")
        mock_mgr_cls.return_value = mgr_instance

        with pytest.raises(RuntimeError, match="build failed"):
            executor._create_task_resources(task=task)

    mgr_instance.kill_all_containers.assert_called_once_with("build_fail_task")
    mgr_instance.create_sandbox.assert_not_called()


def test_create_task_resources_manager_none_on_error(tmp_path: Path) -> None:
    task = _make_task(
        name="manager_none_task",
        worker_image_containerfile="FROM ubuntu",
        worker_image="tiz-worker:latest",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch(
            "tiz.base_task_executor.BaseTaskExecutor._discover_tools", return_value={}
        ),
    ):
        mock_mgr_cls.return_value = None
        mock_mgr_cls.available_engine.return_value = "docker"

        with pytest.raises(AttributeError):
            executor._create_task_resources(task=task)


def test_get_client_for_task_sub_agent_no_engines(tmp_path: Path) -> None:
    """When no inference engines configured, returns default client."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    # Simulate no engines scenario
    executor.manifest.inference_engines = []
    sub_spec = _make_subagent_spec(inference_engine="some-engine")
    client = executor._get_client_for_task_sub_agent(sub_spec)
    assert client is executor._default_client


# _create_sub_agent_chat_factory
# ---------------------------------------------------------------------------


def test_create_sub_agent_chat_factory_no_tools(tmp_path: Path) -> None:
    """Factory creates a Chat with no tools when sub-agent has no tools."""
    sub_spec = _make_subagent_spec(
        name="mysub",
        sys_prompt_custom="You are a sub-agent.",
        inference_engine="test-engine",
    )
    manifest = _make_manifest(
        tasks=[
            _make_task(
                name="main",
                tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
            )
        ],
        engines=[_make_llamacpp_engine(name="test-engine")],
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.Chat") as mock_chat_cls,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx

        factory = executor._create_sub_agent_chat_factory(
            sub_spec, "main_sandbox", mgr_instance, manifest.tasks[0]
        )
        assert callable(factory)

        factory()

        mock_chat_cls.assert_called_once()
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["sys_prompt"] == "You are a sub-agent."
        assert call_kwargs["tools"] is None


def test_create_sub_agent_chat_factory_with_tools(tmp_path: Path) -> None:
    """Factory creates containers and tools for sub-agent."""
    sub_spec = _make_subagent_spec(
        name="mysub",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
    )
    manifest = _make_manifest(
        tasks=[_make_task(name="main")],
        engines=[_make_llamacpp_engine()],
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.Chat") as mock_chat_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sub_sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "bash"
        tool_instance = MagicMock()
        tool_cls.return_value = tool_instance
        mock_disc.return_value = {"bash": tool_cls}

        factory = executor._create_sub_agent_chat_factory(
            sub_spec, "main_sandbox", mgr_instance, manifest.tasks[0]
        )
        factory()

        assert mgr_instance.create_container.call_count == 1
        mock_chat_cls.assert_called_once()
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 1


def test_create_sub_agent_chat_factory_unknown_tool_raises(tmp_path: Path) -> None:
    sub_spec = _make_subagent_spec(
        name="mysub",
        tools=[ToolSpec(name="nonexistent", network="none", disk_mode="nodisk")],
    )
    manifest = _make_manifest(
        tasks=[_make_task(name="main")],
        engines=[_make_llamacpp_engine()],
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        mock_disc.return_value = {}

        factory = executor._create_sub_agent_chat_factory(
            sub_spec, "main_sandbox", mgr_instance, manifest.tasks[0]
        )
        with pytest.raises(ValueError, match="Unknown tool"):
            factory()


def test_resolve_sub_agent_prompt_spec_from_j2_undefined_fallback(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "myagent.j2").write_text("You are {{ name }}.", encoding="utf-8")
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    executor._prompts_dirs = [prompts_dir]
    sub_spec = _make_subagent_spec(sys_prompt="myagent")
    result = executor._resolve_sub_agent_prompt(sub_spec)
    # UndefinedError is caught and raw contents returned
    assert result == "You are {{ name }}."


def test_create_sub_agent_chat_factory_subagents_nested_rejected(
    tmp_path: Path,
) -> None:
    sub_spec = _make_subagent_spec(
        name="mysub",
        tools=[ToolSpec(name="SubAgents", network="none", disk_mode="nodisk")],
    )
    manifest = _make_manifest(
        tasks=[_make_task(name="main")],
        engines=[_make_llamacpp_engine()],
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "SubAgents"
        mock_disc.return_value = {"SubAgents": tool_cls}

        factory = executor._create_sub_agent_chat_factory(
            sub_spec, "main_sandbox", mgr_instance, manifest.tasks[0]
        )
        with pytest.raises(ValueError, match="SubAgents is a special tool"):
            factory()


def test_create_sub_agent_chat_factory_socket_none_raises(tmp_path: Path) -> None:
    sub_spec = _make_subagent_spec(
        name="mysub",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
    )
    manifest = _make_manifest(
        tasks=[_make_task(name="main")],
        engines=[_make_llamacpp_engine()],
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        container_mock = MagicMock()
        container_mock.worker_socket_path = None
        mgr_instance.create_container.return_value = container_mock
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "bash"
        mock_disc.return_value = {"bash": tool_cls}

        factory = executor._create_sub_agent_chat_factory(
            sub_spec, "main_sandbox", mgr_instance, manifest.tasks[0]
        )
        with pytest.raises(ValueError, match="No container socket found for tool"):
            factory()


def test_resolve_sub_agent_prompt_spec_not_found(tmp_path: Path) -> None:
    """When spec file is not found, returns fallback."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    executor._prompts_dirs = [tmp_path / "nonexistent"]
    sub_spec = _make_subagent_spec(sys_prompt="nonexistent_spec")
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result == "You are a helpful assistant."


# _get_client_for_task_sub_agent
# ---------------------------------------------------------------------------


def test_get_client_for_task_sub_agent_no_engines_returns_default(
    tmp_path: Path,
) -> None:
    """When no engines are configured but default client exists, return it."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine(name="myengine")])
    executor = _make_executor(manifest, tmp_path)
    # Override default_client to simulate the no-engines-but-default scenario
    executor._default_client = MagicMock()
    sub_spec = _make_subagent_spec(inference_engine=None)
    client = executor._get_client_for_task_sub_agent(sub_spec)
    assert client is executor._default_client


def test_get_client_for_task_sub_agent_engine_not_found_raises(tmp_path: Path) -> None:
    """When sub-agent specifies an engine not in manifest, raises ValueError."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine(name="myengine")])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(inference_engine="nonexistent")
    with pytest.raises(ValueError, match="not found in manifest"):
        executor._get_client_for_task_sub_agent(sub_spec)


def test_get_client_for_task_sub_agent_no_engine_name(tmp_path: Path) -> None:
    """Returns default client when sub-agent has no engine specified."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(inference_engine=None)
    client = executor._get_client_for_task_sub_agent(sub_spec)
    assert client is executor._default_client


def test_get_client_for_task_sub_agent_found(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine(name="myengine")])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(inference_engine="myengine")
    client = executor._get_client_for_task_sub_agent(sub_spec)
    assert client is not None


def test_get_client_for_task_sub_agent_not_found(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine(name="myengine")])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(inference_engine="other-engine")
    with pytest.raises(ValueError, match="not found in manifest"):
        executor._get_client_for_task_sub_agent(sub_spec)


# _resolve_sub_agent_prompt
# ---------------------------------------------------------------------------


def test_resolve_sub_agent_prompt_custom(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(sys_prompt_custom="You are a sub-agent.")
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result == "You are a sub-agent."


def test_resolve_sub_agent_prompt_custom_with_template(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    executor._context = {"role": "helper"}
    sub_spec = _make_subagent_spec(sys_prompt_custom="You are a {{ role }}.")
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result == "You are a helper."


def test_resolve_sub_agent_prompt_custom_jinja_undefined_fallback(
    tmp_path: Path,
) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(sys_prompt_custom="You are a {{ undefined_var }}.")
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result == "You are a {{ undefined_var }}."


def test_resolve_sub_agent_prompt_spec_from_file(tmp_path: Path) -> None:
    """Prompts directory is searched for the spec file."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "myagent.txt").write_text("You are MyAgent.", encoding="utf-8")
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    # Override prompts dirs
    executor._prompts_dirs = [prompts_dir, tmp_path / "prompts"]
    sub_spec = _make_subagent_spec(sys_prompt="myagent")
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result == "You are MyAgent."


def test_resolve_sub_agent_prompt_default(tmp_path: Path) -> None:
    """Uses 'coding' as default spec when no sys_prompt given."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec()  # no sys_prompt or custom
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result is not None
    assert len(result) > 0


def test_create_task_resources_subagents_image_build_from_dir(
    tmp_path: Path,
) -> None:
    """Sub-agent with different image finds containerfile from directory."""
    containerfile_dir = tmp_path / "containerfiles"
    containerfile_dir.mkdir()
    (containerfile_dir / "Containerfile.tiz-custom:latest").write_text(
        "FROM alpine", encoding="utf-8"
    )
    sub_spec = _make_subagent_spec(
        name="sub1",
        worker_image="tiz-custom:latest",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
    )
    task = _make_task(
        name="cf_dir_sub",
        tools=[],
        subagents=[sub_spec],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)
    # Override containerfiles_dirs to use our test directory
    executor._containerfiles_dirs = [containerfile_dir]

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        mock_disc.return_value = {}
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        executor._create_task_resources(task=task)

    mgr_instance.build_image.assert_any_call(
        containerfile="FROM alpine",
        tag="tiz-custom:latest",
        delete_existing=False,
    )


def test_create_task_resources_subagents_image_not_found_in_dir(
    tmp_path: Path,
) -> None:
    """Sub-agent with different image but no containerfile found."""
    sub_spec = _make_subagent_spec(
        name="sub1",
        worker_image="tiz-custom:latest",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
    )
    task = _make_task(
        name="cf_not_found",
        tools=[],
        subagents=[sub_spec],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        mock_disc.return_value = {}
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        executor._create_task_resources(task=task)

    # No build_image call for the sub-agent image since no containerfile found
    mgr_instance.build_image.assert_not_called()


def test_resolve_sub_agent_prompt_spec_from_j2(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "myagent.j2").write_text("You are {{ name }}.", encoding="utf-8")
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    executor._prompts_dirs = [prompts_dir]
    executor._context = {"name": "Agent007"}
    sub_spec = _make_subagent_spec(sys_prompt="myagent")
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result == "You are Agent007."


def test_resolve_sub_agent_prompt_custom_empty_string(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(sys_prompt_custom="")
    result = executor._resolve_sub_agent_prompt(sub_spec)
    # Falls through to 'coding' prompt since custom is empty
    assert result is not None
    assert len(result) > 0


def test_resolve_sub_agent_prompt_with_tools_context(tmp_path: Path) -> None:
    """Sub-agent prompt includes tool access context from sub_spec.tools."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(
        sys_prompt_custom="wp={{tools_with_project_access}} wop={{tools_without_project_access}} wi={{tools_with_internet_access}} woi={{tools_without_internet_access}}",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="bash", network="internet", disk_mode="nodisk"),
            ToolSpec(name="grep", network="none", disk_mode="ro-disk"),
        ],
    )
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert "wp=read_file, grep" in result
    assert "wop=bash" in result
    assert "wi=bash" in result
    assert "woi=read_file, grep" in result


def test_resolve_sub_agent_prompt_tools_context_deduplication(tmp_path: Path) -> None:
    """Duplicate tool names in sub_spec.tools are deduplicated."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(
        sys_prompt_custom="wp={{tools_with_project_access}}",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
        ],
    )
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result == "wp=read_file"


def test_resolve_sub_agent_prompt_tools_context_none(tmp_path: Path) -> None:
    """When sub_spec.tools is empty, context keys are not set (undefined)."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(
        sys_prompt_custom="wp={{tools_with_project_access is defined}}",
        tools=[],
    )
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert result == "wp=False"


def test_resolve_sub_agent_prompt_date_datetime_uuid(tmp_path: Path) -> None:
    """Sub-agent prompt context includes date, datetime, uuid."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(
        sys_prompt_custom="date={{date}} datetime={{datetime}} uuid={{uuid}}",
    )
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert "date=" in result
    assert "datetime=" in result
    assert "uuid=" in result


def test_resolve_sub_agent_prompt_shared_dirs_and_committer(tmp_path: Path) -> None:
    """Sub-agent prompt context includes shared dirs and committer info."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    sub_spec = _make_subagent_spec(
        sys_prompt_custom="shared={{general_shared_dir}} own={{own_shared_dir}} name={{committer_name}} email={{committer_email}}",
    )
    result = executor._resolve_sub_agent_prompt(sub_spec)
    assert "shared=" in result
    assert "own=" in result
    assert "name=Tester" in result
    assert "email=tester@example.com" in result


# _save_conv_log
# ---------------------------------------------------------------------------


def test_save_conv_log_skipped_when_save_full_logs_false(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=False)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    conv: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
    executor._save_conv_log(conv, "task1")
    assert not (tmp_path / "logs").exists()


def test_save_conv_log_writes_file(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    conv: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]

    executor._save_conv_log(conv, "mytask")
    log_dir = tmp_path / "logs" / "conversations"
    files = list(log_dir.iterdir())
    assert len(files) == 1
    filepath = files[0]
    assert "mytask" in filepath.name
    assert "test_engine" in filepath.name
    assert filepath.name.endswith(".json")
    with filepath.open() as f:
        data = json.load(f)
    assert data == conv


def test_save_conv_log_creates_directory(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    conv: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]

    assert not (tmp_path / "logs").exists()
    executor._save_conv_log(conv, "task1")
    assert (tmp_path / "logs" / "conversations").is_dir()
    assert len(list((tmp_path / "logs" / "conversations").iterdir())) == 1


def test_save_conv_log_with_engine_name_from_task(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="my-custom-engine")
    manifest = _make_manifest(
        save_full_logs=True,
        tasks=[_make_task(name="t1", inference_engine="my-custom-engine")],
        engines=[engine1],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    conv: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]

    executor._save_conv_log(conv, "t1")
    log_dir = tmp_path / "logs" / "conversations"
    files = list(log_dir.iterdir())
    assert len(files) == 1
    assert "my_custom_engine" in files[0].name


def test_save_conv_log_with_special_task_name(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="test-engine")
    manifest = _make_manifest(
        save_full_logs=True,
        tasks=[_make_task(name="My Special Task!")],
        engines=[engine1],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    conv: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]

    executor._save_conv_log(conv, "My Special Task!")
    log_dir = tmp_path / "logs" / "conversations"
    files = list(log_dir.iterdir())
    assert len(files) == 1
    assert "my_special_task_" in files[0].name


def test_save_conv_log_mkdir_oserror(caplog: Any, tmp_path: Path) -> None:
    """OSError on mkdir is caught and logged."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    conv: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]

    with patch.object(Path, "mkdir", side_effect=OSError(13, "Permission denied")):
        caplog.clear()
        executor._save_conv_log(conv, "task1")
        assert "Failed to create conversation log dir" in caplog.text
        assert not (tmp_path / "logs").exists()


def test_save_conv_log_write_oserror(caplog: Any, tmp_path: Path) -> None:
    """OSError on file write is caught and logged."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    conv: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]

    log_dir = tmp_path / "logs" / "conversations"
    log_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(Path, "open", side_effect=OSError(13, "Permission denied")):
        caplog.clear()
        executor._save_conv_log(conv, "task1")
        assert "Failed to write conversation log" in caplog.text


# _create_sub_agents_tool
# ---------------------------------------------------------------------------


def test_create_sub_agents_tool_no_subagents(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    task = _make_task(name="main")
    mgr = MagicMock()
    result = executor._create_sub_agents_tool(task, "sandbox", mgr)
    assert result is None


def test_create_sub_agents_tool_with_subagents(tmp_path: Path) -> None:
    sub_spec = _make_subagent_spec(name="mysub")
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    task = _make_task(name="main", subagents=[sub_spec])
    mgr = MagicMock()

    with patch.object(executor, "_create_sub_agent_chat_factory") as mock_factory:
        mock_factory.return_value = lambda: MagicMock()
        result = executor._create_sub_agents_tool(task, "sandbox", mgr)
        assert result is not None
        assert "mysub" in result._subagents
        assert callable(result._subagents["mysub"]["chat_factory"])
        assert result._subagents["mysub"]["description"] == "A test sub-agent"
        assert callable(result._subagents["mysub"]["usage_callback"])


def test_create_sub_agents_tool_callback_accumulates_usage_and_saves_log(
    tmp_path: Path,
) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        save_full_logs=True,
    )
    executor = _make_executor(manifest, tmp_path)
    task = _make_task(name="main", subagents=[_make_subagent_spec(name="mysub")])
    mgr = MagicMock()

    with patch.object(executor, "_create_sub_agent_chat_factory") as mock_factory:
        mock_factory.return_value = lambda: MagicMock()
        result = executor._create_sub_agents_tool(task, "sandbox", mgr)

    callback = result._subagents["mysub"]["usage_callback"]
    usage = {"prompt_tokens": 50, "completion_tokens": 25, "cost": 0.01}
    conv: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
    callback(usage, conv)

    # Usage should be accumulated under "main/mysub"
    task_usage = executor.task_usage
    assert "main/mysub" in task_usage
    assert task_usage["main/mysub"]["prompt_tokens"] == 50
    assert task_usage["main/mysub"]["completion_tokens"] == 25
    assert task_usage["main/mysub"]["cost"] == 0.01

    # Conversation log should be saved
    log_dir = tmp_path / "logs" / "conversations"
    files = list(log_dir.iterdir())
    assert len(files) == 1
    assert "main_mysub" in files[0].name


# _create_task_resources with subagents
# ---------------------------------------------------------------------------


def test_create_task_tools_and_sandbox_no_tools_no_subagents_skips_setup(
    tmp_path: Path,
) -> None:
    """When no tools and no subagents, sandbox is skipped."""
    task = _make_task(name="no_tools_no_sub", tools=[], subagents=[])
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = _make_executor(manifest, tmp_path)
    resources = executor._create_task_resources(task=task)
    assert resources.tool_instances == []
    assert resources.sandbox is None
    assert resources.manager is None
    assert resources.sandbox_name is None


def test_create_task_tools_and_sandbox_with_subagents_no_tools(tmp_path: Path) -> None:
    """Subagents only (no regular tools) still creates sandbox."""
    sub_spec = _make_subagent_spec(
        name="sub1",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
    )
    task = _make_task(name="sub_only", tools=[], subagents=[sub_spec])
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        mock_disc.return_value = {}
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        resources = executor._create_task_resources(task=task)

    assert len(resources.tool_instances) == 1
    assert resources.tool_instances[0] is mock_sub_tool
    assert resources.sandbox is not None
    assert resources.manager is not None
    assert resources.sandbox_name == "sub_only"
    assert resources.action_lock is lock_ctx
    mock_create_sub.assert_called_once_with(task, "sub_only", mgr_instance)


def test_create_task_tools_and_sandbox_with_subagents_always_created(
    tmp_path: Path,
) -> None:
    """Subagents are always created when configured. No flag needed."""
    sub_spec = _make_subagent_spec(name="sub1")
    task = _make_task(
        name="sub_always",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
        subagents=[sub_spec],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "bash"
        bash_instance = MagicMock()
        bash_instance.socket_path = "/tmp/sock"
        tool_cls.return_value = bash_instance
        mock_disc.return_value = {"bash": tool_cls}
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        resources = executor._create_task_resources(task=task)

    assert len(resources.tool_instances) == 2
    assert resources.tool_instances[0] is bash_instance
    assert resources.tool_instances[1] is mock_sub_tool
    mock_create_sub.assert_called_once_with(task, "sub_always", mgr_instance)


def test_create_task_tools_and_sandbox_with_subagents_different_image(
    tmp_path: Path,
) -> None:
    """Sub-agent with different worker image triggers image build when containerfile found."""
    sub_spec = _make_subagent_spec(
        name="sub1",
        worker_image="tiz-custom:latest",
        worker_image_containerfile="FROM alpine",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
    )
    task = _make_task(
        name="diff_img",
        tools=[],
        subagents=[sub_spec],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        mock_disc.return_value = {}
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        executor._create_task_resources(task=task)

    # Build image should have been called for the sub-agent's image
    mgr_instance.build_image.assert_any_call(
        containerfile="FROM alpine",
        tag="tiz-custom:latest",
        delete_existing=False,
    )


def test_create_task_tools_and_sandbox_with_subagents_same_image_no_build(
    tmp_path: Path,
) -> None:
    """Sub-agent with same image as task does not trigger extra build."""
    sub_spec = _make_subagent_spec(
        name="sub1",
        worker_image="tiz-worker:latest",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
    )
    task = _make_task(
        name="same_img",
        tools=[],
        subagents=[sub_spec],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        mock_disc.return_value = {}
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        executor._create_task_resources(task=task)

    # Only the task's image build (same image) - not a second one
    build_calls = mgr_instance.build_image.call_args_list
    assert len(build_calls) <= 1


def test_create_task_tools_and_sandbox_subagents_with_containerfile(
    tmp_path: Path,
) -> None:
    """Sub-agent with different image and containerfile triggers build."""
    sub_spec = _make_subagent_spec(
        name="sub1",
        worker_image="tiz-custom:latest",
        worker_image_containerfile="FROM alpine",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
    )
    task = _make_task(
        name="cf_sub",
        tools=[],
        subagents=[sub_spec],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        mock_disc.return_value = {}
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        executor._create_task_resources(task=task)

    mgr_instance.build_image.assert_any_call(
        containerfile="FROM alpine",
        tag="tiz-custom:latest",
        delete_existing=False,
    )


def test_create_task_tools_and_sandbox_no_tools_with_subagents(tmp_path: Path) -> None:
    """Task with subagents but no tools still creates sandbox."""
    sub_spec = _make_subagent_spec(name="sub1")
    task = _make_task(name="sub_no_tools", tools=[], subagents=[sub_spec])
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        resources = executor._create_task_resources(task=task)

    assert len(resources.tool_instances) == 1
    assert resources.tool_instances[0] is mock_sub_tool
    assert resources.sandbox is not None
    assert resources.sandbox_name == "sub_no_tools"
    mgr_instance.create_sandbox.assert_called_once()


def test_create_task_tools_and_sandbox_with_subagents_and_regular_tools(
    tmp_path: Path,
) -> None:
    """Task with both regular tools and subagents has both in tool list."""
    sub_spec = _make_subagent_spec(name="sub1")
    task = _make_task(
        name="both",
        tools=[ToolSpec(name="bash", network="none", disk_mode="nodisk")],
        subagents=[sub_spec],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.BaseTaskExecutor._discover_tools") as mock_disc,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        mgr_instance.create_container.return_value = container_mock
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        tool_cls = MagicMock()
        tool_cls.fname.return_value = "bash"
        bash_instance = MagicMock()
        bash_instance.socket_path = "/tmp/sock"
        tool_cls.return_value = bash_instance
        mock_disc.return_value = {"bash": tool_cls}
        mock_sub_tool = MagicMock()
        mock_create_sub.return_value = mock_sub_tool

        resources = executor._create_task_resources(task=task)

    assert len(resources.tool_instances) == 2
    assert resources.tool_instances[0] is bash_instance
    assert resources.tool_instances[1] is mock_sub_tool
    mock_create_sub.assert_called_once_with(task, "both", mgr_instance)


def test_create_task_tools_and_sandbox_subagents_returns_none_skips_append(
    tmp_path: Path,
) -> None:
    """When _create_sub_agents_tool returns None, no tool is appended."""
    sub_spec = _make_subagent_spec(name="sub1")
    task = _make_task(
        name="sub_none",
        tools=[],
        subagents=[sub_spec],
    )
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()], container_engine="podman"
    )
    executor = _make_executor(manifest, tmp_path)

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch.object(executor, "_create_sub_agents_tool") as mock_create_sub,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        sandbox_dirs = MagicMock()
        mgr_instance.create_sandbox.return_value = sandbox_dirs
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx
        mock_create_sub.return_value = None

        resources = executor._create_task_resources(task=task)

    assert len(resources.tool_instances) == 0
    assert resources.sandbox is not None
    assert resources.sandbox_name == "sub_none"
    mock_create_sub.assert_called_once_with(task, "sub_none", mgr_instance)


# ---------------------------------------------------------------------------
# _create_sub_agent_chat_factory with callbacks
# ---------------------------------------------------------------------------


def test_create_sub_agent_chat_factory_with_update_callback(tmp_path: Path) -> None:
    """update_callback is propagated to the Chat when set on executor."""
    update_callback = MagicMock()
    sub_spec = _make_subagent_spec(
        name="mysub",
        sys_prompt_custom="You are a sub-agent.",
        inference_engine="test-engine",
    )
    manifest = _make_manifest(
        tasks=[_make_task(name="main")],
        engines=[_make_llamacpp_engine(name="test-engine")],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._update_callback = update_callback

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.Chat") as mock_chat_cls,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx

        factory = executor._create_sub_agent_chat_factory(
            sub_spec, "main_sandbox", mgr_instance, manifest.tasks[0]
        )
        factory()

    mock_chat_cls.assert_called_once()
    call_kwargs = mock_chat_cls.call_args.kwargs
    assert call_kwargs["update_callback"] is not None
    # Verify the callback works (Chat passes subtask_name as second arg)
    call_kwargs["update_callback"]({"msg": "test"}, "mysub")
    update_callback.assert_called_once_with({"msg": "test"}, "mysub")


def test_create_sub_agent_chat_factory_with_confirm_callback(tmp_path: Path) -> None:
    """confirm_callback is propagated to the Chat when set on executor."""
    confirm_callback = MagicMock()
    sub_spec = _make_subagent_spec(
        name="mysub",
        sys_prompt_custom="You are a sub-agent.",
        inference_engine="test-engine",
    )
    manifest = _make_manifest(
        tasks=[_make_task(name="main")],
        engines=[_make_llamacpp_engine(name="test-engine")],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor._confirm_callback = confirm_callback

    with (
        patch("tiz.base_task_executor.SandboxManager") as mock_mgr_cls,
        patch("tiz.base_task_executor.Chat") as mock_chat_cls,
    ):
        mgr_instance = MagicMock()
        mock_mgr_cls.return_value = mgr_instance
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        mgr_instance.sandbox_lock.return_value = lock_ctx

        factory = executor._create_sub_agent_chat_factory(
            sub_spec, "main_sandbox", mgr_instance, manifest.tasks[0]
        )
        factory()

    mock_chat_cls.assert_called_once()
    call_kwargs = mock_chat_cls.call_args.kwargs
    assert call_kwargs["confirm_callback"] is not None
    # Verify the callback was called (Chat passes subtask_name as third arg)
    call_kwargs["confirm_callback"]({"call": "test"}, lambda _d, _b: "result", "mysub")
    assert confirm_callback.call_count == 1
    args, kwargs = confirm_callback.call_args
    assert args[0] == {"call": "test"}
    assert args[2] == "mysub"


# ---------------------------------------------------------------------------
# _create_tool_instances_from_specs (direct, default_readonly=None)
# ---------------------------------------------------------------------------


def test_create_tool_instances_from_specs_default_readonly_none(tmp_path: Path) -> None:
    """When default_readonly is None, read_only_project is derived from disk_mode (nodisk -> True)."""
    tool_specs = [ToolSpec(name="bash", network="none", disk_mode="nodisk")]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    mock_manager = MagicMock()
    container_mock = MagicMock()
    container_mock.worker_socket_path = "/tmp/sock"
    mock_manager.create_container.return_value = container_mock
    tool_cls = MagicMock()
    tool_cls.fname.return_value = "bash"
    tool_instance = MagicMock()
    tool_cls.return_value = tool_instance

    tools, confirmations = executor._create_tool_instances_from_specs(
        tool_specs,
        {"bash": tool_cls},
        "sandbox1",
        "tiz-worker:latest",
        mock_manager,
        error_context="direct",
        default_readonly=None,
    )

    assert len(tools) == 1
    assert confirmations == {}
    mock_manager.create_container.assert_called_once()
    call_kwargs = mock_manager.create_container.call_args.kwargs
    # nodisk -> read_only_project should be True (disk_mode != "disk")
    assert call_kwargs["read_only_project"] is True
    # nodisk -> mount_project should be False
    assert call_kwargs["mount_project"] is False
    tool_cls.assert_called_once_with(socket_path="/tmp/sock")


def test_create_tool_instances_from_specs_extra_run_args(tmp_path: Path) -> None:
    """extra_run_args is passed to create_container."""
    tool_specs = [ToolSpec(name="bash", network="none", disk_mode="nodisk")]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    mock_manager = MagicMock()
    container_mock = MagicMock()
    container_mock.worker_socket_path = "/tmp/sock"
    mock_manager.create_container.return_value = container_mock
    tool_cls = MagicMock()
    tool_cls.fname.return_value = "bash"
    tool_instance = MagicMock()
    tool_cls.return_value = tool_instance

    extra_args = ["--cap-add=NET_ADMIN", "--device=/dev/fuse"]
    tools, confirmations = executor._create_tool_instances_from_specs(
        tool_specs,
        {"bash": tool_cls},
        "sandbox1",
        "tiz-worker:latest",
        mock_manager,
        error_context="direct",
        default_readonly=None,
        extra_run_args=extra_args,
    )

    assert len(tools) == 1
    assert confirmations == {}
    mock_manager.create_container.assert_called_once()
    call_kwargs = mock_manager.create_container.call_args.kwargs
    assert call_kwargs["extra_run_args"] == extra_args


def test_create_tool_instances_from_specs_default_readonly_none_disk(
    tmp_path: Path,
) -> None:
    """When default_readonly is None and disk_mode is 'disk', read_only_project is False."""
    tool_specs = [ToolSpec(name="write_file", network="none", disk_mode="disk")]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    mock_manager = MagicMock()
    container_mock = MagicMock()
    container_mock.worker_socket_path = "/tmp/sock"
    mock_manager.create_container.return_value = container_mock
    tool_cls = MagicMock()
    tool_cls.fname.return_value = "write_file"
    tool_instance = MagicMock()
    tool_cls.return_value = tool_instance

    tools, confirmations = executor._create_tool_instances_from_specs(
        tool_specs,
        {"write_file": tool_cls},
        "sandbox1",
        "tiz-worker:latest",
        mock_manager,
        error_context="direct",
        default_readonly=None,
    )

    assert len(tools) == 1
    assert confirmations == {}

    call_kwargs = mock_manager.create_container.call_args.kwargs
    # disk -> read_only_project should be False (disk_mode == "disk")
    assert call_kwargs["read_only_project"] is False
    # disk -> mount_project should be True
    assert call_kwargs["mount_project"] is True


def test_create_tool_instances_from_specs_with_confirmations(tmp_path: Path) -> None:
    """When ToolSpec has confirmations, they are returned in the confirmations dict."""
    tool_specs = [
        ToolSpec(
            name="bash",
            network="none",
            disk_mode="nodisk",
            confirmations=[ConfirmationSpec(type="exact", key="command", value="rm")],
        ),
        ToolSpec(name="read_file", network="none", disk_mode="nodisk"),
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    mock_manager = MagicMock()
    container_mock = MagicMock()
    container_mock.worker_socket_path = "/tmp/sock"
    mock_manager.create_container.return_value = container_mock
    bash_cls = MagicMock()
    bash_cls.fname.return_value = "bash"
    bash_instance = MagicMock()
    bash_cls.return_value = bash_instance
    read_cls = MagicMock()
    read_cls.fname.return_value = "read_file"
    read_instance = MagicMock()
    read_cls.return_value = read_instance

    tools, confirmations = executor._create_tool_instances_from_specs(
        tool_specs,
        {"bash": bash_cls, "read_file": read_cls},
        "sandbox1",
        "tiz-worker:latest",
        mock_manager,
        error_context="direct",
        default_readonly=None,
    )

    assert len(tools) == 2
    assert "bash" in confirmations
    assert len(confirmations["bash"]) == 1
    assert confirmations["bash"][0].type == "exact"
    assert confirmations["bash"][0].key == "command"
    assert confirmations["bash"][0].value == "rm"
    # read_file has no confirmations
    assert "read_file" not in confirmations
