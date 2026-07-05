# ruff: noqa: ARG001,ARG002,SIM117
# mypy: disable-error-code="attr-defined,misc"
"""Tests for src/tiz/manifest_executor.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tiz.manifest_executor import ManifestExecutor
from tiz.manifest_parser import (
    DEFAULT_COLOR_INPUT,
    DEFAULT_COLOR_REASONING,
    CmdAction,
    InferenceEngineSpec,
    IteratorAction,
    Manifest,
    ManifestMeta,
    PromptsAction,
    RepeaterAction,
    ScoringAction,
    TaskSpec,
    ToolSpec,
)


class _UnknownAction:
    """Action class that does not match any isinstance check in _run_task."""

    pass


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
        extra_container_args=kwargs.get("extra_container_args"),
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
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        }
        instance.conv = []
        m.return_value = instance
        yield m, instance


@pytest.fixture
def mock_manager(tmp_path: Path) -> Any:
    with (
        patch("tiz.base_task_executor.SandboxManager") as m,
    ):
        instance = MagicMock()
        sandbox_dirs = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        type(sandbox_dirs).project_dir = property(lambda _self: project_dir)  # type: ignore[assignment]
        sandbox_dirs.is_git_repo.return_value = True
        sandbox_dirs.git_capture_branch.return_value = "main"
        sandbox_dirs.git_create_branch.return_value = None
        sandbox_dirs.git_finalize_branches.return_value = None
        sandbox_dirs.sync_to_original_auto_rebase.return_value = None
        instance.create_sandbox.return_value = sandbox_dirs
        container_mock = MagicMock()
        container_mock.worker_socket_path = "/tmp/sock"
        instance.create_container.return_value = container_mock
        lock_ctx = MagicMock()
        lock_ctx.__enter__ = MagicMock(return_value=None)
        lock_ctx.__exit__ = MagicMock(return_value=None)
        instance.sandbox_lock.return_value = lock_ctx
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
# __init__
# ---------------------------------------------------------------------------


def test_init_basic(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    assert executor.manifest is manifest
    assert executor._update_callback is None


def test_init_with_base_path(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    assert executor._base_path == tmp_path


def test_init_prompts_and_containerfiles_dirs(tmp_path: Path) -> None:
    """Verify _prompts_dirs and _containerfiles_dirs include base_path and data dirs."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    _data_dir = Path(__file__).resolve().parent.parent / "src" / "tiz" / "data"

    assert len(executor._prompts_dirs) == 2
    assert executor._prompts_dirs[0] == tmp_path / "prompts"
    assert executor._prompts_dirs[1] == _data_dir / "prompts"

    assert len(executor._containerfiles_dirs) == 2
    assert executor._containerfiles_dirs[0] == tmp_path / "containerfiles"
    assert executor._containerfiles_dirs[1] == _data_dir / "containerfiles"


def test_init_with_engine(tmp_path: Path) -> None:
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        version="1",
        parallelism=1,
        committer_name="Tester",
        committer_email="tester@example.com",
        container_engine="podman",
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    assert executor._engine == "podman"


def test_init_with_callback(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    cb = MagicMock()
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    assert executor._update_callback is cb


def test_init_no_engine_raises(tmp_path: Path) -> None:
    manifest = _make_manifest()
    with pytest.raises(ValueError, match="No inference engines configured"):
        ManifestExecutor(manifest=manifest, base_path=tmp_path)


# ---------------------------------------------------------------------------
# _run_message_group
# ---------------------------------------------------------------------------


def test_run_message_group(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "resp",
        "prompt_tokens": 5,
        "prompt_time": 0.1,
        "completion_tokens": 10,
        "completion_time": 0.2,
        "cost": 0.0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    executor._run_message_group(
        "task1",
        ["msg1", "msg2"],
        "system",
        [],
        {},
        client,
    )

    assert chat_instance.send_message.call_count == 2
    chat_instance.send_message.assert_any_call("msg1", timeout=None)
    chat_instance.send_message.assert_any_call("msg2", timeout=None)
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 10
    assert usage["task1"]["completion_tokens"] == 20
    assert usage["task1"]["prompt_time"] == pytest.approx(0.2)
    assert usage["task1"]["completion_time"] == pytest.approx(0.4)
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert len(usage["task1"]["tool_calls"]) == 2


# ---------------------------------------------------------------------------
# _run_prompts_action
# ---------------------------------------------------------------------------


def test_run_prompts_action_sequential(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = PromptsAction(
        message_groups=[["m1", "m2"], ["m3"]],
        parallel_message_groups=False,
    )
    executor._run_prompts_action("task1", action, "sys", [], {}, client)

    assert chat_instance.send_message.call_count == 3
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls == ["m1", "m2", "m3"]
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 3
    assert usage["task1"]["completion_tokens"] == 6
    assert usage["task1"]["prompt_time"] == pytest.approx(0.3)
    assert usage["task1"]["completion_time"] == pytest.approx(0.6)
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert len(usage["task1"]["tool_calls"]) == 3


def test_run_prompts_action_parallel(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], parallelism=2)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = PromptsAction(
        message_groups=[["m1"], ["m2"]],
        parallel_message_groups=True,
    )
    executor._run_prompts_action("task1", action, "sys", [], {}, client)

    assert chat_instance.send_message.call_count == 2
    sent = {c[0][0] for c in chat_instance.send_message.call_args_list}
    assert sent == {"m1", "m2"}
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 2
    assert usage["task1"]["completion_tokens"] == 4
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.2)
    assert usage["task1"]["completion_time"] == pytest.approx(0.4)
    assert len(usage["task1"]["tool_calls"]) == 2


# ---------------------------------------------------------------------------
# _run_iterator_action
# ---------------------------------------------------------------------------


def test_run_iterator_action_sequential(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "line1\nline2\n",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "resp",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "resp",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = IteratorAction(
        input_prompt="gen",
        prompt_groups=[["prompt {{item}}"]],
        parallel_message_groups=False,
    )
    executor._run_iterator_action("task1", action, "sys", [], {}, client)

    assert chat_instance.send_message.call_count == 3
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "gen"
    assert "line1" in calls[1]
    assert "line2" in calls[2]
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 15
    assert usage["task1"]["completion_tokens"] == 30
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.30000000000000004)
    assert usage["task1"]["completion_time"] == pytest.approx(0.6000000000000001)
    assert len(usage["task1"]["tool_calls"]) == 3


def test_run_iterator_action_parallel(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "a\nb\n",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "resp",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "resp",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], parallelism=2)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = IteratorAction(
        input_prompt="gen",
        prompt_groups=[["p {{item}}"]],
        parallel_message_groups=True,
    )
    executor._run_iterator_action("task1", action, "sys", [], {}, client)

    assert chat_instance.send_message.call_count == 3
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "gen"
    assert any(c == "p a" for c in calls[1:])
    assert any(c == "p b" for c in calls[1:])
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 15
    assert usage["task1"]["completion_tokens"] == 30
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.30000000000000004)
    assert usage["task1"]["completion_time"] == pytest.approx(0.6000000000000001)
    assert len(usage["task1"]["tool_calls"]) == 3


def test_run_iterator_action_empty_output(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "",
        "prompt_tokens": 5,
        "prompt_time": 0.1,
        "completion_tokens": 10,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = IteratorAction(
        input_prompt="gen",
        prompt_groups=[["p {{item}}"]],
    )
    executor._run_iterator_action("task1", action, "sys", [], {}, client)

    assert chat_instance.send_message.call_count == 1
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 5
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["completion_tokens"] == 10


# ---------------------------------------------------------------------------
# _run_repeater_action
# ---------------------------------------------------------------------------


def test_run_repeater_action(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = RepeaterAction(
        repeat=3,
        prompt_groups=[["p{{item}}"]],
    )
    executor._run_repeater_action("task1", action, "sys", [], {}, client)

    assert chat_instance.send_message.call_count == 3
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls == ["p1", "p2", "p3"]
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 3
    assert usage["task1"]["completion_tokens"] == 6
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.30000000000000004)
    assert usage["task1"]["completion_time"] == pytest.approx(0.6000000000000001)
    assert len(usage["task1"]["tool_calls"]) == 3


def test_run_repeater_action_multiple_groups(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = RepeaterAction(
        repeat=2,
        prompt_groups=[["g1"], ["g2"]],
    )
    executor._run_repeater_action("task1", action, "sys", [], {}, client)

    assert chat_instance.send_message.call_count == 4
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls == ["g1", "g2", "g1", "g2"]
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 4
    assert usage["task1"]["completion_tokens"] == 8
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.4)
    assert usage["task1"]["completion_time"] == pytest.approx(0.8)
    assert len(usage["task1"]["tool_calls"]) == 4


def test_run_repeater_jinja_rendering(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = RepeaterAction(
        repeat=1,
        prompt_groups=[["hello {{item}}"]],
    )
    executor._run_repeater_action("task1", action, "sys", [], {}, client)

    assert chat_instance.send_message.call_count == 1
    args = chat_instance.send_message.call_args_list[0]
    assert args[0][0] == "hello 1"
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 1
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["completion_tokens"] == 2


# ---------------------------------------------------------------------------
# _run_scoring_action
# ---------------------------------------------------------------------------


def test_run_scoring_action_rounds_gt1_no_iterator(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r1",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score this",
        rounds=2,
        scoring_rounds=1,
        prompt_groups=[["run {{item}}"]],
        iterator_input=None,
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    assert chat_instance.send_message.call_count == 3
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "run {{item}}"
    assert calls[1] == "run {{item}}"
    assert "score this" in calls[2]
    assert "tiz/scoring_task1_r0" in calls[2]
    assert "tiz/scoring_task1_r1" in calls[2]

    assert sandbox.git_create_branch.call_count == 2
    branch_calls = [c[0][0] for c in sandbox.git_create_branch.call_args_list]
    assert branch_calls == ["tiz/scoring_task1_r0", "tiz/scoring_task1_r1"]

    sandbox.git_finalize_branches.assert_called_once()
    finalize_args = sandbox.git_finalize_branches.call_args
    assert finalize_args[0][0] == "main"
    assert finalize_args[0][1] == "tiz/scoring_task1_r1"
    assert finalize_args[0][2] == [
        "tiz/scoring_task1_r0",
        "tiz/scoring_task1_r1",
    ]

    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 3
    assert usage["task1"]["completion_tokens"] == 6
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.30000000000000004)
    assert usage["task1"]["completion_time"] == pytest.approx(0.6000000000000001)
    assert len(usage["task1"]["tool_calls"]) == 3


def test_run_scoring_action_rounds_gt1_with_iterator(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "itemA\n",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r1",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="pick best",
        rounds=2,
        scoring_rounds=1,
        prompt_groups=[["do {{item}}"]],
        iterator_input="generate items",
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    assert chat_instance.send_message.call_count == 4
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "generate items"
    assert calls[1] == "do itemA"
    assert calls[2] == "do itemA"
    assert "pick best" in calls[3]
    assert "tiz/scoring_task1_r0" in calls[3]
    assert "tiz/scoring_task1_r1" in calls[3]

    assert sandbox.git_create_branch.call_count == 2
    branch_calls = [c[0][0] for c in sandbox.git_create_branch.call_args_list]
    assert branch_calls == ["tiz/scoring_task1_r0", "tiz/scoring_task1_r1"]

    sandbox.git_finalize_branches.assert_called_once()
    finalize_args = sandbox.git_finalize_branches.call_args
    assert finalize_args[0][1] == "tiz/scoring_task1_r1"

    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 4
    assert usage["task1"]["completion_tokens"] == 8
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.4)
    assert usage["task1"]["completion_time"] == pytest.approx(0.8)
    assert len(usage["task1"]["tool_calls"]) == 4


def test_run_scoring_action_rounds_gt1_multiple_scoring_rounds(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r1",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=2,
        scoring_rounds=3,
        prompt_groups=[["p"]],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    assert chat_instance.send_message.call_count == 5
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "p"
    assert calls[1] == "p"
    assert "score" in calls[2]
    assert "score" in calls[3]
    assert "score" in calls[4]

    assert sandbox.git_create_branch.call_count == 2
    sandbox.git_finalize_branches.assert_called_once()
    finalize_args = sandbox.git_finalize_branches.call_args
    assert len(finalize_args[0][2]) == 2

    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 5
    assert usage["task1"]["completion_tokens"] == 10
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.5)
    assert usage["task1"]["completion_time"] == pytest.approx(1.0)
    assert len(usage["task1"]["tool_calls"]) == 5


def test_run_scoring_action_rounds_gt1_no_tools_raises(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    action = ScoringAction(
        scoring_prompt="score",
        rounds=3,
        scoring_rounds=2,
        prompt_groups=[["p"]],
    )

    with pytest.raises(RuntimeError, match="Scoring action requires a sandbox"):
        executor._run_task(
            _make_task(
                name="no_tools_rounds_gt1",
                tools=[],
                actions=[action],
            )
        )


def test_run_scoring_action_no_iterator(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "winner",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score this",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["run {{item}}"]],
        iterator_input=None,
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    assert chat_instance.send_message.call_count == 2
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "run {{item}}"
    assert "score this" in calls[1]
    assert "tiz/scoring_task1_r0" in calls[1]
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 2
    assert usage["task1"]["completion_tokens"] == 4
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.2)
    assert usage["task1"]["completion_time"] == pytest.approx(0.4)
    assert len(usage["task1"]["tool_calls"]) == 2


def test_run_scoring_action_with_iterator(mock_chat: Any, tmp_path: Path) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "itemA\n",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="pick best",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["do {{item}}"]],
        iterator_input="generate items",
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    assert chat_instance.send_message.call_count == 3
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "generate items"
    assert calls[1] == "do itemA"
    assert "pick best" in calls[2]
    assert "tiz/scoring_task1_r0" in calls[2]
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 3
    assert usage["task1"]["completion_tokens"] == 6
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.30000000000000004)
    assert usage["task1"]["completion_time"] == pytest.approx(0.6000000000000001)
    assert len(usage["task1"]["tool_calls"]) == 3


def test_run_scoring_action_no_tools_raises(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["p"]],
    )

    with pytest.raises(RuntimeError, match="Scoring action requires a sandbox"):
        executor._run_task(
            _make_task(
                name="no_tools_task",
                tools=[],
                actions=[action],
            )
        )


def test_run_scoring_action_not_a_git_repo_raises(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.is_git_repo.return_value = False

    with pytest.raises(
        RuntimeError,
        match="Scoring action requires a git repository in the sandbox",
    ):
        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)


def test_run_scoring_action_zero_rounds_raises(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify that rounds=0 raises ValueError."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=0,
        scoring_rounds=1,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.is_git_repo.return_value = True

    with pytest.raises(ValueError, match="Scoring rounds must be at least 1"):
        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)


def test_run_scoring_action_negative_rounds_raises(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify that negative rounds raise ValueError."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=-1,
        scoring_rounds=1,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.is_git_repo.return_value = True

    with pytest.raises(ValueError, match="Scoring rounds must be at least 1"):
        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)


def test_run_scoring_action_even_scoring_rounds_raises(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=2,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.is_git_repo.return_value = True

    with pytest.raises(ValueError, match="Scoring rounds must be odd"):
        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)


def test_run_scoring_action_negative_odd_scoring_rounds_raises(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify that negative odd numbers like -3 also raise (regression test)."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=-3,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.is_git_repo.return_value = True

    with pytest.raises(ValueError, match="Scoring rounds must be positive"):
        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)


def test_run_scoring_action_zero_scoring_rounds_raises(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify that scoring_rounds=0 also raises."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=0,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.is_git_repo.return_value = True

    with pytest.raises(ValueError, match="Scoring rounds must be positive"):
        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)


def test_scoring_branch_case_insensitive(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "TIZ/SCORING_TASK1_R0",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    sandbox.git_finalize_branches.assert_called_once()
    call_args = sandbox.git_finalize_branches.call_args
    assert call_args[0][1] == "tiz/scoring_task1_r0"
    sandbox.git_capture_branch.assert_called_once()
    sandbox.git_create_branch.assert_called_once_with("tiz/scoring_task1_r0")


def test_scoring_no_scores_falls_back_to_original(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "invalid_branch",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "original-branch"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    call_args = sandbox.git_finalize_branches.call_args
    assert call_args[0][1] == "original-branch"


def test_scoring_multiple_rounds(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = lambda *_, **__: {
        "message": "tiz/scoring_task1_r0",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=3,
        prompt_groups=[["run"]],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    assert sandbox.git_finalize_branches.call_count == 1
    assert chat_instance.send_message.call_count == 4
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "run"
    assert "score" in calls[1] and "tiz/scoring_task1_r0" in calls[1]
    assert "score" in calls[2] and "tiz/scoring_task1_r0" in calls[2]
    assert "score" in calls[3] and "tiz/scoring_task1_r0" in calls[3]
    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 4
    assert usage["task1"]["completion_tokens"] == 8
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.4)
    assert usage["task1"]["completion_time"] == pytest.approx(0.8)
    assert len(usage["task1"]["tool_calls"]) == 4


# ---------------------------------------------------------------------------
# _run_task – no tools path
# ---------------------------------------------------------------------------


def test_run_task_no_tools(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "done",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    with patch("tiz.base_task_executor.logger") as mock_logger:
        executor._run_task(
            _make_task(
                name="simple",
                tools=[],
                actions=[PromptsAction(message_groups=[["hello"]])],
            )
        )
    mock_logger.info.assert_any_call(
        "Task %s has no tools; skipping sandbox setup", "simple"
    )


def test_run_task_prompts_action(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "resp",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._run_task(
        _make_task(
            name="t1",
            tools=[],
            actions=[PromptsAction(message_groups=[["say hi"]])],
        )
    )
    assert chat_instance.send_message.call_count == 1
    chat_instance.send_message.assert_called_once_with("say hi", timeout=None)


def test_run_task_iterator_action(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "l1\n",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "r",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._run_task(
        _make_task(
            name="t1",
            tools=[],
            actions=[
                IteratorAction(input_prompt="gen", prompt_groups=[["p {{item}}"]])
            ],
        )
    )
    assert chat_instance.send_message.call_count == 2
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "gen"
    assert calls[1] == "p l1"
    usage = executor.task_usage
    assert usage["t1"]["prompt_tokens"] == 2
    assert usage["t1"]["completion_tokens"] == 4
    assert usage["t1"]["cached_tokens"] == 0
    assert usage["t1"]["cache_write_tokens"] == 0
    assert usage["t1"]["cost"] == 0.0
    assert usage["t1"]["prompt_time"] == pytest.approx(0.2)
    assert usage["t1"]["completion_time"] == pytest.approx(0.4)
    assert len(usage["t1"]["tool_calls"]) == 2


def test_run_task_repeater_action(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._run_task(
        _make_task(
            name="t1",
            tools=[],
            actions=[RepeaterAction(repeat=2, prompt_groups=[["p{{item}}"]])],
        )
    )
    assert chat_instance.send_message.call_count == 2
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls == ["p1", "p2"]
    usage = executor.task_usage
    assert usage["t1"]["prompt_tokens"] == 2
    assert usage["t1"]["completion_tokens"] == 4
    assert usage["t1"]["cached_tokens"] == 0
    assert usage["t1"]["cache_write_tokens"] == 0
    assert usage["t1"]["cost"] == 0.0
    assert usage["t1"]["prompt_time"] == pytest.approx(0.2)
    assert usage["t1"]["completion_time"] == pytest.approx(0.4)
    assert len(usage["t1"]["tool_calls"]) == 2


def test_run_task_keyboard_interrupt_with_tools(
    mock_manager: Any, mock_chat: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = KeyboardInterrupt
    _, mgr_instance = mock_manager
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        task = _make_task(
            name="kb_task",
            tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
            actions=[PromptsAction(message_groups=[["do it"]])],
        )
        with pytest.raises(KeyboardInterrupt):
            executor._run_task(task)

    mgr_instance.kill_all_containers.assert_called_once()


def test_run_task_cmd_sync(
    mock_manager: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, mgr_instance = mock_manager
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    task = _make_task(
        name="sync_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[CmdAction(command="sync")],
    )
    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor._run_task(task)

    mgr_instance.create_sandbox.assert_called_once()
    mgr_instance.kill_all_containers.assert_called_once()


def test_run_task_cmd_not_sync_logs_warning(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """CmdAction with command other than 'sync' logs a warning."""
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    with patch("tiz.manifest_executor.logger") as mock_logger:
        executor._run_task(
            _make_task(
                name="non_sync",
                tools=[],
                actions=[
                    CmdAction(command="other"),
                    PromptsAction(message_groups=[["hi"]]),
                ],
            )
        )
    mock_logger.warning.assert_called_once_with(
        "Unsupported CmdAction command '%s' in task '%s'; ignoring",
        "other",
        "non_sync",
    )
    # The CmdAction("other") is logged and ignored, PromptsAction runs
    chat_instance.send_message.assert_called_once_with("hi", timeout=None)


def test_run_task_cmd_sync_no_tools_warns(mock_chat: Any, tmp_path: Path) -> None:
    """CmdAction 'sync' with no tools logs a warning but does not raise."""
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    with patch("tiz.manifest_executor.logger") as mock_logger:
        executor._run_task(
            _make_task(
                name="sync_no_tools",
                tools=[],
                actions=[CmdAction(command="sync")],
            )
        )
    mock_logger.warning.assert_called_once_with(
        "CmdAction 'sync' requires a sandbox with tools; "
        "task '%s' has no tools configured",
        "sync_no_tools",
    )


# ---------------------------------------------------------------------------
# execute  – serial
# ---------------------------------------------------------------------------


def test_execute_no_tasks(mock_discover_tools: Any, tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()


def test_execute_single_task_serial(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    task = _make_task(name="t1", actions=[PromptsAction(message_groups=[["hi"]])])
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    with patch("tiz.manifest_executor.logger") as mock_logger:
        executor.execute()
    mock_logger.info.assert_any_call("Executing serial task %s", "t1")
    mock_logger.info.assert_any_call("Serial task %s completed", "t1")


# ---------------------------------------------------------------------------
# execute  – parallel tasks grouped
# ---------------------------------------------------------------------------


def test_execute_parallel_tasks_grouped(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    task1 = _make_task(
        name="t1",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["a"]])],
    )
    task2 = _make_task(
        name="t2",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["b"]])],
    )
    manifest = _make_manifest(
        tasks=[task1, task2],
        engines=[_make_llamacpp_engine()],
        parallelism=2,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    with patch("tiz.manifest_executor.logger") as mock_logger:
        executor.execute()
    assert any(
        "Executing" in str(c) and "parallel" in str(c)
        for c in mock_logger.info.call_args_list
    )
    usage = executor.task_usage
    assert len(usage) == 2
    assert usage["t1"]["prompt_tokens"] == 1
    assert usage["t2"]["prompt_tokens"] == 1
    calls = {c[0][0] for c in chat_instance.send_message.call_args_list}
    assert usage["t1"]["cached_tokens"] == 0
    assert usage["t1"]["cache_write_tokens"] == 0
    assert usage["t1"]["cost"] == 0.0
    assert usage["t1"]["prompt_time"] == pytest.approx(0.1)
    assert usage["t1"]["completion_time"] == pytest.approx(0.2)
    assert usage["t1"]["completion_tokens"] == 2
    assert usage["t2"]["cached_tokens"] == 0
    assert usage["t2"]["cache_write_tokens"] == 0
    assert usage["t2"]["cost"] == 0.0
    assert usage["t2"]["prompt_time"] == pytest.approx(0.1)
    assert usage["t2"]["completion_time"] == pytest.approx(0.2)
    assert usage["t2"]["completion_tokens"] == 2
    assert calls == {"a", "b"}


def test_execute_mixed_parallel_and_serial_tasks(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    task_parallel = _make_task(
        name="par",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["a"]])],
    )
    task_serial = _make_task(
        name="ser",
        allow_parallel_run=False,
        actions=[PromptsAction(message_groups=[["b"]])],
    )
    task_parallel2 = _make_task(
        name="par2",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["c"]])],
    )
    manifest = _make_manifest(
        tasks=[task_parallel, task_serial, task_parallel2],
        engines=[_make_llamacpp_engine()],
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    usage = executor.task_usage
    assert len(usage) == 3
    for name in ("par", "ser", "par2"):
        assert name in usage
        assert usage[name]["prompt_tokens"] == 1
        assert usage[name]["completion_tokens"] == 2
        assert usage[name]["cached_tokens"] == 0
        assert usage[name]["cache_write_tokens"] == 0
        assert usage[name]["cost"] == 0.0
        assert usage[name]["prompt_time"] == pytest.approx(0.1)
        assert usage[name]["completion_time"] == pytest.approx(0.2)
    calls = {c[0][0] for c in chat_instance.send_message.call_args_list}
    assert calls == {"a", "b", "c"}


def test_execute_serial_task_grouping(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify that consecutive serial tasks each get their own group."""
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    t1 = _make_task(
        name="s1",
        allow_parallel_run=False,
        actions=[PromptsAction(message_groups=[["a"]])],
    )
    t2 = _make_task(
        name="s2",
        allow_parallel_run=False,
        actions=[PromptsAction(message_groups=[["b"]])],
    )
    manifest = _make_manifest(tasks=[t1, t2], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor.execute()
    usage = executor.task_usage
    assert "s1" in usage
    assert "s2" in usage
    assert usage["s1"]["prompt_tokens"] == 1
    assert usage["s2"]["prompt_tokens"] == 1
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert "a" in calls
    assert usage["s1"]["cached_tokens"] == 0
    assert usage["s1"]["cache_write_tokens"] == 0
    assert usage["s1"]["cost"] == 0.0
    assert usage["s1"]["prompt_time"] == pytest.approx(0.1)
    assert usage["s1"]["completion_time"] == pytest.approx(0.2)
    assert usage["s1"]["completion_tokens"] == 2
    assert usage["s2"]["cached_tokens"] == 0
    assert usage["s2"]["cache_write_tokens"] == 0
    assert usage["s2"]["cost"] == 0.0
    assert usage["s2"]["prompt_time"] == pytest.approx(0.1)
    assert usage["s2"]["completion_time"] == pytest.approx(0.2)
    assert usage["s2"]["completion_tokens"] == 2
    assert "b" in calls


def test_execute_parallel_then_serial(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    t1 = _make_task(
        name="p1",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["a"]])],
    )
    t2 = _make_task(
        name="p2",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["b"]])],
    )
    t3 = _make_task(
        name="s1",
        allow_parallel_run=False,
        actions=[PromptsAction(message_groups=[["c"]])],
    )
    manifest = _make_manifest(tasks=[t1, t2, t3], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    usage = executor.task_usage
    assert len(usage) == 3
    for name in ("p1", "p2", "s1"):
        assert name in usage
        assert usage[name]["prompt_tokens"] == 1
        assert usage[name]["cached_tokens"] == 0
        assert usage[name]["cache_write_tokens"] == 0
        assert usage[name]["cost"] == 0.0
        assert usage[name]["prompt_time"] == pytest.approx(0.1)
        assert usage[name]["completion_time"] == pytest.approx(0.2)
    calls = {c[0][0] for c in chat_instance.send_message.call_args_list}
    assert calls == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# execute  – task failure logging
# ---------------------------------------------------------------------------


def test_execute_parallel_task_failure_logged(
    mock_chat: Any, mock_discover_tools: Any, mock_sleep: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = RuntimeError("fail")
    chat_instance.replay.side_effect = RuntimeError("fail")
    t1 = _make_task(
        name="fail_t",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["a"]])],
    )
    t2 = _make_task(
        name="ok_t",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["b"]])],
    )
    manifest = _make_manifest(
        tasks=[t1, t2], engines=[_make_llamacpp_engine()], parallelism=2
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    with patch("tiz.manifest_executor.logger") as mock_logger:
        executor.execute()
    mock_logger.exception.assert_called()


def test_execute_serial_task_failure(
    mock_chat: Any, mock_discover_tools: Any, mock_sleep: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = RuntimeError("fail")
    chat_instance.replay.side_effect = RuntimeError("fail")
    t1 = _make_task(
        name="fail_t",
        allow_parallel_run=False,
        actions=[PromptsAction(message_groups=[["a"]])],
    )
    manifest = _make_manifest(tasks=[t1], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    with pytest.raises(RuntimeError, match="fail"):
        executor.execute()


def test_execute_parallel_task_failure_continues(
    mock_chat: Any, mock_discover_tools: Any, mock_sleep: Any, tmp_path: Path
) -> None:
    """Parallel failures shouldn't stop other tasks."""
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    t1 = _make_task(
        name="ok_t",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["a"]])],
    )
    t2 = _make_task(
        name="fail_t",
        allow_parallel_run=True,
        actions=[PromptsAction(message_groups=[["b"]])],
    )

    call_count = 0

    def side_effect(*_a: Any, **_kw: Any) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("fail")
        return {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        }

    chat_instance.send_message.side_effect = side_effect
    chat_instance.replay.side_effect = RuntimeError("fail")
    manifest = _make_manifest(
        tasks=[t1, t2], engines=[_make_llamacpp_engine()], parallelism=2
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    usage = executor.task_usage
    assert len(usage) == 1
    all_calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert "a" in all_calls
    assert "b" in all_calls
    for _task_name, task_usage in usage.items():
        assert task_usage["prompt_tokens"] == 1
        assert task_usage["completion_tokens"] == 2
        assert task_usage["cached_tokens"] == 0
        assert task_usage["cache_write_tokens"] == 0
        assert task_usage["cost"] == 0.0
        assert task_usage["prompt_time"] == pytest.approx(0.1)
        assert task_usage["completion_time"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# execute with tools – sandbox setup
# ---------------------------------------------------------------------------


def test_execute_task_with_tools(
    mock_chat: Any, mock_manager: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        task = _make_task(
            name="with_tools",
            tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
            actions=[PromptsAction(message_groups=[["do it"]])],
        )
        manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
        executor.execute()

    mgr_instance.create_sandbox.assert_called_once()
    mgr_instance.build_image.assert_not_called()
    mgr_instance.kill_all_containers.assert_called_once()


def test_execute_task_with_worker_image(
    mock_chat: Any, mock_manager: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="img_task",
        worker_image_containerfile="FROM ubuntu",
        worker_image="tiz-worker-img:latest",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor.execute()

    mgr_instance.build_image.assert_called_once_with(
        containerfile="FROM ubuntu",
        tag="tiz-worker-img:latest",
        delete_existing=False,
    )


def test_execute_task_with_containerfile_from_dir(
    mock_chat: Any, mock_manager: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="img_task_cf_dir",
        worker_image="tiz-worker-img:latest",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir()
    cf_file = cf_dir / "Containerfile.tiz-worker-img:latest"
    cf_file.write_text("FROM alpine")

    tool_cls = MagicMock()
    executor._containerfiles_dirs = [cf_dir]
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor.execute()

    mgr_instance.build_image.assert_called_once_with(
        containerfile="FROM alpine",
        tag="tiz-worker-img:latest",
        delete_existing=False,
    )


def test_execute_task_with_tool_not_found_raises(
    mock_manager: Any, mock_chat: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="missing_tool",
        tools=[ToolSpec(name="nonexistent_tool", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    with patch("tiz.base_task_executor.SandboxManager", return_value=mgr_instance):
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

        with patch(
            "tiz.base_task_executor.BaseTaskExecutor._discover_tools", return_value={}
        ):
            with pytest.raises(ValueError, match="Unknown tool"):
                executor.execute()

    mgr_instance.create_sandbox.assert_called_once()
    mgr_instance.kill_all_containers.assert_called_once()


def test_execute_task_combo_sock_none_raises(
    mock_manager: Any, mock_chat: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="bad_combo",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["x"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])

    with patch("tiz.base_task_executor.SandboxManager", return_value=mgr_instance):
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

        tool_cls = MagicMock()
        tool_cls.fname.return_value = "read_file"
        container_mock = MagicMock()
        container_mock.worker_socket_path = None
        mgr_instance.create_container.return_value = container_mock

        with patch(
            "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
            return_value={"read_file": tool_cls},
        ):
            with pytest.raises(ValueError, match="No container socket found for tool"):
                executor.execute()

    mgr_instance.kill_all_containers.assert_called_once()


# ---------------------------------------------------------------------------
# execute – all action types within a task
# ---------------------------------------------------------------------------


def test_execute_task_with_iterator_action(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "item1\n",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    task = _make_task(
        name="iter_task",
        tools=[],
        actions=[IteratorAction(input_prompt="gen", prompt_groups=[["p {{item}}"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    assert chat_instance.send_message.call_count == 2
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "gen"
    assert calls[1] == "p item1"
    usage = executor.task_usage
    assert usage["iter_task"]["prompt_tokens"] == 2
    assert usage["iter_task"]["completion_tokens"] == 4
    assert usage["iter_task"]["cached_tokens"] == 0
    assert usage["iter_task"]["cache_write_tokens"] == 0
    assert usage["iter_task"]["cost"] == 0.0
    assert usage["iter_task"]["prompt_time"] == pytest.approx(0.2)
    assert usage["iter_task"]["completion_time"] == pytest.approx(0.4)
    assert len(usage["iter_task"]["tool_calls"]) == 2


def test_execute_task_with_repeater_action(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    task = _make_task(
        name="rep_task",
        tools=[],
        actions=[RepeaterAction(repeat=2, prompt_groups=[["p{{item}}"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    assert chat_instance.send_message.call_count == 2
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls == ["p1", "p2"]
    usage = executor.task_usage
    assert usage["rep_task"]["prompt_tokens"] == 2
    assert usage["rep_task"]["completion_tokens"] == 4
    assert usage["rep_task"]["cached_tokens"] == 0
    assert usage["rep_task"]["cache_write_tokens"] == 0
    assert usage["rep_task"]["cost"] == 0.0
    assert usage["rep_task"]["prompt_time"] == pytest.approx(0.2)
    assert usage["rep_task"]["completion_time"] == pytest.approx(0.4)
    assert len(usage["rep_task"]["tool_calls"]) == 2


def test_execute_task_with_scoring_action(
    mock_chat: Any, mock_manager: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    _, mgr_instance = mock_manager
    chat_instance.send_message.side_effect = [
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_scoring_task_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    task = _make_task(
        name="scoring_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[
            ScoringAction(
                scoring_prompt="pick best",
                rounds=1,
                scoring_rounds=1,
                prompt_groups=[["p"]],
            )
        ],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor.execute()

    assert chat_instance.send_message.call_count == 2
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "p"
    assert "pick best" in calls[1]
    assert "tiz/scoring_scoring_task_r0" in calls[1]
    usage = executor.task_usage
    assert usage["scoring_task"]["prompt_tokens"] == 2
    assert usage["scoring_task"]["completion_tokens"] == 4
    assert usage["scoring_task"]["cached_tokens"] == 0
    assert usage["scoring_task"]["cache_write_tokens"] == 0
    assert usage["scoring_task"]["cost"] == 0.0
    assert usage["scoring_task"]["prompt_time"] == pytest.approx(0.2)
    assert usage["scoring_task"]["completion_time"] == pytest.approx(0.4)
    assert len(usage["scoring_task"]["tool_calls"]) == 2


def test_execute_task_with_scoring_action_iterator_input(
    mock_chat: Any, mock_manager: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    _, mgr_instance = mock_manager
    chat_instance.send_message.side_effect = [
        {
            "message": "item1\n",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_scorer_task_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    task = _make_task(
        name="scorer_task",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[
            ScoringAction(
                scoring_prompt="pick",
                rounds=1,
                scoring_rounds=1,
                prompt_groups=[["p {{item}}"]],
                iterator_input="gen",
            )
        ],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor.execute()
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "gen"
    assert calls[1] == "p item1"
    assert "pick" in calls[2]
    assert "tiz/scoring_scorer_task_r0" in calls[2]
    usage = executor.task_usage
    assert usage["scorer_task"]["prompt_tokens"] == 3
    assert usage["scorer_task"]["completion_tokens"] == 6
    assert usage["scorer_task"]["cached_tokens"] == 0
    assert usage["scorer_task"]["cache_write_tokens"] == 0
    assert usage["scorer_task"]["cost"] == 0.0
    assert usage["scorer_task"]["prompt_time"] == pytest.approx(0.30000000000000004)
    assert usage["scorer_task"]["completion_time"] == pytest.approx(0.6000000000000001)
    assert len(usage["scorer_task"]["tool_calls"]) == 3


def test_multiple_actions_same_task(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    task = _make_task(
        name="multi_action",
        tools=[],
        actions=[
            PromptsAction(message_groups=[["first"]]),
            RepeaterAction(repeat=1, prompt_groups=[["second {{item}}"]]),
            PromptsAction(message_groups=[["third"]]),
        ],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    assert chat_instance.send_message.call_count == 3
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "first"
    assert calls[1] == "second 1"
    assert calls[2] == "third"
    usage = executor.task_usage
    assert usage["multi_action"]["prompt_tokens"] == 3
    assert usage["multi_action"]["completion_tokens"] == 6
    assert usage["multi_action"]["cached_tokens"] == 0
    assert usage["multi_action"]["cache_write_tokens"] == 0
    assert usage["multi_action"]["cost"] == 0.0
    assert usage["multi_action"]["prompt_time"] == pytest.approx(0.30000000000000004)
    assert usage["multi_action"]["completion_time"] == pytest.approx(0.6000000000000001)
    assert len(usage["multi_action"]["tool_calls"]) == 3


# ---------------------------------------------------------------------------
# execute with tools – readonly sandbox paths
# ---------------------------------------------------------------------------


def test_execute_task_readonly_sandbox(
    mock_manager: Any, mock_chat: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="ro_task",
        readonly_sandbox=True,
        tools=[ToolSpec(name="read_file", network="none", disk_mode="ro-disk")],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor.execute()

    mgr_instance.create_container.assert_called_once()
    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["read_only_project"] is True
    assert call_kwargs["use_host_timezone"] is True


def test_execute_task_readonly_false_for_disk_mode(
    mock_manager: Any, mock_chat: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="rw_task",
        readonly_sandbox=False,
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor.execute()

    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["read_only_project"] is False
    assert call_kwargs["use_host_timezone"] is True


def test_execute_task_ro_disk_mode(
    mock_manager: Any, mock_chat: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="ro_disk",
        readonly_sandbox=False,
        tools=[ToolSpec(name="read_file", network="none", disk_mode="ro-disk")],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor.execute()

    call_kwargs = mgr_instance.create_container.call_args.kwargs
    assert call_kwargs["read_only_project"] is True
    assert call_kwargs["use_host_timezone"] is True


# ---------------------------------------------------------------------------
# execute with tools – multiple tool combos
# ---------------------------------------------------------------------------


def test_execute_task_multiple_tool_combos(
    mock_manager: Any, mock_chat: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="multi_combo",
        tools=[
            ToolSpec(name="read_file", network="none", disk_mode="disk"),
            ToolSpec(name="bash", network="internet", disk_mode="nodisk"),
        ],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls_rf = MagicMock()
    tool_cls_rf.fname.return_value = "read_file"
    tool_cls_bash = MagicMock()
    tool_cls_bash.fname.return_value = "bash"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls_rf, "bash": tool_cls_bash},
    ):
        executor.execute()

    assert mgr_instance.create_container.call_count == 2
    call1_kwargs = mgr_instance.create_container.call_args_list[0].kwargs
    call2_kwargs = mgr_instance.create_container.call_args_list[1].kwargs
    combos = {
        (c["network"], c["read_only_project"]) for c in [call1_kwargs, call2_kwargs]
    }
    assert combos == {("none", False), ("internet", True)}
    for c in [call1_kwargs, call2_kwargs]:
        assert c["sandbox_name"] == "multi_combo"
        assert c["image"] == "tiz-worker:latest"
        assert c["use_host_timezone"] is True


# ---------------------------------------------------------------------------
# execute with project_path and force_copy_files
# ---------------------------------------------------------------------------


def test_execute_task_with_project_and_force_copy(
    mock_manager: Any, mock_chat: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    _, mgr_instance = mock_manager
    task = _make_task(
        name="proj_task",
        project="/some/project",
        force_copy_files=["*.env"],
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        executor.execute()

    call_kwargs = mgr_instance.create_sandbox.call_args.kwargs
    assert call_kwargs["project_path"] == "/some/project"
    assert call_kwargs["sandbox_name"] == "proj_task"
    assert call_kwargs["force_copy_files"] == ["*.env"]
    assert call_kwargs["committer_name"] == "Tester"
    assert call_kwargs["committer_email"] == "tester@example.com"


def test_run_prompts_action_parallel_failure_logged(
    mock_chat: Any, mock_discover_tools: Any, mock_sleep: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = RuntimeError("boom")
    chat_instance.replay.side_effect = RuntimeError("boom")
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], parallelism=2)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = PromptsAction(
        message_groups=[["m1"], ["m2"]],
        parallel_message_groups=True,
    )
    with patch("tiz.manifest_executor.logger") as mock_logger:
        executor._run_prompts_action("task1", action, "sys", [], {}, client)
    mock_logger.exception.assert_called()


def test_run_iterator_action_parallel_failure_logged(
    mock_chat: Any, mock_discover_tools: Any, mock_sleep: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.replay.side_effect = RuntimeError("boom")
    chat_instance.send_message.side_effect = [
        {
            "message": "a\nb\n",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        RuntimeError("boom"),
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], parallelism=2)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = IteratorAction(
        input_prompt="gen",
        prompt_groups=[["p {{item}}"]],
        parallel_message_groups=True,
    )
    with patch("tiz.manifest_executor.logger") as mock_logger:
        executor._run_iterator_action("task1", action, "sys", [], {}, client)
    mock_logger.exception.assert_called()


def test_run_scoring_action_empty_scoring_prompt_raises(
    mock_chat: Any, mock_manager: Any, tmp_path: Path
) -> None:
    """Cover the ValueError raised when scoring_prompt is empty."""
    _, mgr_instance = mock_manager
    action = ScoringAction(
        scoring_prompt="",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["p"]],
    )
    task = _make_task(
        name="empty_prompt",
        tools=[ToolSpec(name="read_file", network="none", disk_mode="disk")],
        actions=[action],
    )
    mgr_manifest = _make_manifest(tasks=[task], engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=mgr_manifest, base_path=tmp_path)

    tool_cls = MagicMock()
    tool_cls.fname.return_value = "read_file"
    with patch(
        "tiz.base_task_executor.BaseTaskExecutor._discover_tools",
        return_value={"read_file": tool_cls},
    ):
        with pytest.raises(ValueError, match="Scoring prompt is required"):
            executor.execute()


# ---------------------------------------------------------------------------
# _run_scoring_line branch cleanup on exception (Bug 2)
# ---------------------------------------------------------------------------


def test_run_scoring_line_branch_cleanup_on_exception(
    mock_chat: Any, mock_discover_tools: Any, mock_sleep: Any, tmp_path: Path
) -> None:
    """Verify that if _run_scoring_line raises mid-way, branches are cleaned up."""
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        RuntimeError("mid-way failure"),
    ]
    chat_instance.replay.side_effect = RuntimeError("mid-way failure")
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=2,
        scoring_rounds=1,
        prompt_groups=[["p"]],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    with pytest.raises(RuntimeError, match="mid-way failure"):
        executor._run_scoring_line(
            "task1", action, "sys", [], {}, sandbox, None, client
        )

    # First branch was created and the second round failed
    assert sandbox.git_create_branch.call_count == 2
    # Should have cleaned up the branch that was created
    sandbox.git_finalize_branches.assert_called()
    # The cleanup should checkout original branch and finalize with original as winner
    cleanup_calls = sandbox.git_finalize_branches.call_args_list
    assert any(args[0][0] == "main" and args[0][1] == "main" for args in cleanup_calls)
    sandbox.git_checkout.assert_called_with("main")


def test_run_scoring_line_branch_cleanup_suppresses_errors(
    mock_chat: Any, mock_discover_tools: Any, mock_sleep: Any, tmp_path: Path
) -> None:
    """Verify that cleanup errors during exception handling are suppressed."""
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        RuntimeError("fail"),
    ]
    chat_instance.replay.side_effect = RuntimeError("fail")
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=2,
        scoring_rounds=1,
        prompt_groups=[["p"]],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    # Simulate that git_checkout also fails during cleanup
    sandbox.git_checkout.return_value = None
    sandbox.git_finalize_branches.side_effect = RuntimeError("cleanup fail")

    with pytest.raises(RuntimeError, match="fail"):
        executor._run_scoring_line(
            "task1", action, "sys", [], {}, sandbox, None, client
        )

    # The original exception should propagate, not the cleanup failures
    assert sandbox.git_finalize_branches.called


# ---------------------------------------------------------------------------
# _resolve_engine_names_to_clients
# ---------------------------------------------------------------------------


def test_resolve_engine_names_to_clients_empty(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    clients = executor._resolve_engine_names_to_clients([])
    assert clients == []


def test_resolve_engine_names_to_clients_single(tmp_path: Path) -> None:
    engine = _make_llamacpp_engine(name="my-llama")
    engine2 = _make_openrouter_engine(name="my-openrouter")
    manifest = _make_manifest(engines=[engine, engine2])
    with patch.object(ManifestExecutor, "build_client") as mock_build:
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
        mock_build.reset_mock()

        clients = executor._resolve_engine_names_to_clients(["my-llama"])
        assert len(clients) == 1
        assert clients[0] is mock_client
        mock_build.assert_called_once_with(engine)


def test_resolve_engine_names_to_clients_multiple(tmp_path: Path) -> None:
    engine1 = _make_llamacpp_engine(name="e1")
    engine2 = _make_openrouter_engine(name="e2")
    manifest = _make_manifest(engines=[engine1, engine2])
    with patch.object(ManifestExecutor, "build_client") as mock_build:
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_client3 = MagicMock()
        mock_build.side_effect = [mock_client1, mock_client2, mock_client3]
        executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
        mock_build.reset_mock()

        mock_build.side_effect = [mock_client1, mock_client2]
        clients = executor._resolve_engine_names_to_clients(["e1", "e2"])
        assert len(clients) == 2
        assert clients[0] is mock_client1
        assert clients[1] is mock_client2
        assert mock_build.call_count == 2
        mock_build.assert_any_call(engine1)
        mock_build.assert_any_call(engine2)


def test_resolve_engine_names_to_clients_not_found(tmp_path: Path) -> None:
    manifest = _make_manifest(engines=[_make_llamacpp_engine(name="e1")])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    with pytest.raises(ValueError, match="not found in manifest"):
        executor._resolve_engine_names_to_clients(["nonexistent"])


# ---------------------------------------------------------------------------
# scoring action with engines / scoring_engines cycling
# ---------------------------------------------------------------------------


def test_run_scoring_action_with_engines_cycling(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r1",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    engine1 = _make_llamacpp_engine(name="e1")
    engine2 = _make_openrouter_engine(name="e2")
    manifest = _make_manifest(engines=[engine1, engine2])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=3,
        scoring_rounds=1,
        prompt_groups=[["p"]],
        engines=["e1", "e2"],
        scoring_engines=["e2"],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    with patch.object(executor, "_resolve_engine_names_to_clients") as mock_resolve:
        round_client1 = MagicMock()
        round_client2 = MagicMock()
        scoring_client = MagicMock()
        mock_resolve.side_effect = lambda names: {
            ("e1", "e2"): [round_client1, round_client2],
            ("e2",): [scoring_client],
        }.get(tuple(names), [])

        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

        assert mock_resolve.call_count == 2
        mock_resolve.assert_any_call(["e1", "e2"])
        mock_resolve.assert_any_call(["e2"])

        chat_calls = mock_chat[0].call_args_list
        assert len(chat_calls) == 4
        assert chat_calls[0].kwargs["client"] is round_client1
        assert chat_calls[1].kwargs["client"] is round_client2
        assert chat_calls[2].kwargs["client"] is round_client1
        assert chat_calls[3].kwargs["client"] is scoring_client

    assert chat_instance.send_message.call_count == 4
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "p"
    assert calls[1] == "p"
    assert calls[2] == "p"
    assert "score" in calls[3]

    sandbox.git_finalize_branches.assert_called_once()


def test_run_scoring_action_with_scoring_engines_cycling(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    engine1 = _make_llamacpp_engine(name="e1")
    engine2 = _make_openrouter_engine(name="e2")
    manifest = _make_manifest(engines=[engine1, engine2])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=3,
        prompt_groups=[["p"]],
        engines=[],
        scoring_engines=["e1", "e2"],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    with patch.object(executor, "_resolve_engine_names_to_clients") as mock_resolve:
        scoring_client1 = MagicMock()
        scoring_client2 = MagicMock()
        mock_resolve.side_effect = lambda names: {
            ("e1", "e2"): [scoring_client1, scoring_client2],
        }.get(tuple(names), [])

        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

        assert mock_resolve.call_count == 2
        mock_resolve.assert_any_call([])
        mock_resolve.assert_any_call(["e1", "e2"])

        chat_calls = mock_chat[0].call_args_list
        assert len(chat_calls) == 4
        assert chat_calls[0].kwargs["client"] is client
        assert chat_calls[1].kwargs["client"] is scoring_client1
        assert chat_calls[2].kwargs["client"] is scoring_client2
        assert chat_calls[3].kwargs["client"] is scoring_client1

    assert chat_instance.send_message.call_count == 4
    sandbox.git_finalize_branches.assert_called_once()


def test_run_scoring_action_engines_cycle_wraps(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """3 rounds, 2 engines: round 0->engine[0], round 1->engine[1], round 2->engine[0]."""
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    engine1 = _make_llamacpp_engine(name="e1")
    engine2 = _make_openrouter_engine(name="e2")
    manifest = _make_manifest(engines=[engine1, engine2])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=3,
        scoring_rounds=1,
        prompt_groups=[["p"]],
        engines=["e1", "e2"],
        scoring_engines=[],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    with patch.object(executor, "_resolve_engine_names_to_clients") as mock_resolve:
        round_client1 = MagicMock()
        round_client2 = MagicMock()
        mock_resolve.side_effect = lambda names: {
            ("e1", "e2"): [round_client1, round_client2],
        }.get(tuple(names), [])

        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

        assert mock_resolve.call_count == 2
        mock_resolve.assert_any_call(["e1", "e2"])
        mock_resolve.assert_any_call([])

        chat_calls = mock_chat[0].call_args_list
        assert len(chat_calls) == 4
        assert chat_calls[0].kwargs["client"] is round_client1
        assert chat_calls[1].kwargs["client"] is round_client2
        assert chat_calls[2].kwargs["client"] is round_client1
        assert chat_calls[3].kwargs["client"] is client

    assert chat_instance.send_message.call_count == 4


def test_run_scoring_action_with_iterator_and_engines(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "itemA\n",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    engine1 = _make_llamacpp_engine(name="e1")
    manifest = _make_manifest(engines=[engine1])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="pick best",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["do {{item}}"]],
        iterator_input="generate items",
        engines=["e1"],
        scoring_engines=["e1"],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    assert chat_instance.send_message.call_count == 3
    calls = [c[0][0] for c in chat_instance.send_message.call_args_list]
    assert calls[0] == "generate items"
    assert calls[1] == "do itemA"
    assert "pick best" in calls[2]

    usage = executor.task_usage
    assert usage["task1"]["prompt_tokens"] == 3
    assert usage["task1"]["completion_tokens"] == 6
    assert usage["task1"]["cached_tokens"] == 0
    assert usage["task1"]["cache_write_tokens"] == 0
    assert usage["task1"]["cost"] == 0.0
    assert usage["task1"]["prompt_time"] == pytest.approx(0.30000000000000004)
    assert usage["task1"]["completion_time"] == pytest.approx(0.6000000000000001)
    assert len(usage["task1"]["tool_calls"]) == 3


# ---------------------------------------------------------------------------
# update_callback tiz-internal structured dicts
# ---------------------------------------------------------------------------


def test_callback_starting_task(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )

    executor._run_task(
        _make_task(
            name="mytask", tools=[], actions=[PromptsAction(message_groups=[["hi"]])]
        )
    )

    assert cb.called
    calls = cb.call_args_list
    starting_call = calls[0]
    assert starting_call[0][0] == {
        "tiz-internal": {"status": "starting_task", "task": "mytask"}
    }


def test_callback_completed_task(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )

    executor._run_task(
        _make_task(
            name="mytask", tools=[], actions=[PromptsAction(message_groups=[["hi"]])]
        )
    )

    completed_call = cb.call_args_list[-1]
    assert completed_call[0][0] == {
        "tiz-internal": {"status": "completed_task", "task": "mytask"}
    }


def test_callback_executing_action(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )

    executor._run_task(
        _make_task(
            name="mytask", tools=[], actions=[PromptsAction(message_groups=[["hi"]])]
        )
    )

    action_call = cb.call_args_list[1]
    assert action_call[0][0] == {
        "tiz-internal": {
            "status": "executing_action",
            "task": "mytask",
            "action_idx": 1,
            "total_actions": 1,
            "action_type": "PromptsAction",
        }
    }


def test_callback_message_group(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "resp",
        "prompt_tokens": 5,
        "prompt_time": 0.1,
        "completion_tokens": 10,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    executor._run_message_group("task1", ["m1", "m2"], "sys", [], {}, client)

    assert cb.call_count == 2
    assert cb.call_args_list[0][0][0] == {
        "tiz-internal": {
            "action": "message_group",
            "status": "sending",
            "task": "task1",
            "message": 1,
            "total_messages": 2,
            "prompt": "m1",
        }
    }
    assert cb.call_args_list[1][0][0] == {
        "tiz-internal": {
            "action": "message_group",
            "status": "sending",
            "task": "task1",
            "message": 2,
            "total_messages": 2,
            "prompt": "m2",
        }
    }


def test_callback_prompts_parallel(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], parallelism=2)
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = PromptsAction(
        message_groups=[["m1"], ["m2"]],
        parallel_message_groups=True,
    )
    executor._run_prompts_action("task1", action, "sys", [], {}, client)

    parallel_call = cb.call_args_list[0]
    assert parallel_call[0][0] == {
        "tiz-internal": {
            "action": "prompts",
            "status": "running",
            "task": "task1",
            "prompts_groups": 2,
            "prompts_parallel": True,
        }
    }


def test_callback_prompts_sequential(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = PromptsAction(
        message_groups=[["m1"], ["m3"]],
        parallel_message_groups=False,
    )
    executor._run_prompts_action("task1", action, "sys", [], {}, client)

    assert cb.call_args_list[0][0][0] == {
        "tiz-internal": {
            "action": "prompts",
            "status": "group",
            "task": "task1",
            "group": 1,
            "total_groups": 2,
        }
    }
    assert cb.call_args_list[2][0][0] == {
        "tiz-internal": {
            "action": "prompts",
            "status": "group",
            "task": "task1",
            "group": 2,
            "total_groups": 2,
        }
    }


def test_callback_iterator_generating_items(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "",
        "prompt_tokens": 5,
        "prompt_time": 0.1,
        "completion_tokens": 10,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = IteratorAction(input_prompt="gen", prompt_groups=[["p {{item}}"]])
    executor._run_iterator_action("task1", action, "sys", [], {}, client)

    assert cb.call_args_list[0][0][0] == {
        "tiz-internal": {
            "action": "iterator",
            "status": "generating_items",
            "task": "task1",
            "prompt": "gen",
        }
    }


def test_callback_iterator_processing_line(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "hello\n",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "resp",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "resp",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = IteratorAction(
        input_prompt="gen",
        prompt_groups=[["p {{item}}"], ["q {{item}}"]],
        parallel_message_groups=False,
    )
    executor._run_iterator_action("task1", action, "sys", [], {}, client)

    processing_calls = [
        c
        for c in cb.call_args_list
        if c[0][0]["tiz-internal"].get("status") == "processing_line"
    ]
    assert len(processing_calls) == 2
    assert processing_calls[0][0][0] == {
        "tiz-internal": {
            "action": "iterator",
            "status": "processing_line",
            "task": "task1",
            "line": "hello",
            "group": 1,
            "total_groups": 2,
            "current_line": 1,
            "total_lines": 1,
        }
    }
    assert processing_calls[1][0][0] == {
        "tiz-internal": {
            "action": "iterator",
            "status": "processing_line",
            "task": "task1",
            "line": "hello",
            "group": 2,
            "total_groups": 2,
            "current_line": 1,
            "total_lines": 1,
        }
    }


def test_callback_repeater_iteration(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = RepeaterAction(repeat=2, prompt_groups=[["p{{item}}"]])
    executor._run_repeater_action("task1", action, "sys", [], {}, client)

    assert cb.call_count >= 2
    assert cb.call_args_list[0][0][0] == {
        "tiz-internal": {
            "action": "repeater",
            "status": "iteration",
            "task": "task1",
            "iteration": 1,
            "total_iterations": 2,
        }
    }
    assert cb.call_args_list[1][0][0] == {
        "tiz-internal": {
            "action": "repeater",
            "status": "iteration",
            "task": "task1",
            "iteration": 2,
            "total_iterations": 2,
        }
    }


def test_callback_scoring_round_with_engines(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r2",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    cb = MagicMock()
    engine1 = _make_llamacpp_engine(name="e1")
    engine2 = _make_openrouter_engine(name="e2")
    manifest = _make_manifest(engines=[engine1, engine2])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=3,
        scoring_rounds=1,
        prompt_groups=[["p"]],
        engines=["e1", "e2"],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    with patch.object(executor, "_resolve_engine_names_to_clients") as mock_resolve:
        rc1, rc2, sc1, sc2 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        mock_resolve.side_effect = lambda names: {
            ("e1", "e2"): [rc1, rc2],
            (): [sc1, sc2],
        }.get(tuple(names), [])

        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    round_calls = [
        c for c in cb.call_args_list if c[0][0]["tiz-internal"].get("status") == "round"
    ]
    assert len(round_calls) == 3
    assert round_calls[0][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "round",
            "task": "task1",
            "round": 1,
            "total_rounds": 3,
            "group": 1,
            "total_groups": 1,
            "engine": "e1",
        }
    }
    assert round_calls[1][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "round",
            "task": "task1",
            "round": 2,
            "total_rounds": 3,
            "group": 1,
            "total_groups": 1,
            "engine": "e2",
        }
    }
    assert round_calls[2][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "round",
            "task": "task1",
            "round": 3,
            "total_rounds": 3,
            "group": 1,
            "total_groups": 1,
            "engine": "e1",
        }
    }


def test_callback_scoring_round(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=2,
        scoring_rounds=1,
        prompt_groups=[["p"]],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    round_calls = [
        c for c in cb.call_args_list if c[0][0]["tiz-internal"].get("status") == "round"
    ]
    assert len(round_calls) == 2
    assert round_calls[0][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "round",
            "task": "task1",
            "round": 1,
            "total_rounds": 2,
            "group": 1,
            "total_groups": 1,
            "engine": None,
        }
    }
    assert round_calls[1][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "round",
            "task": "task1",
            "round": 2,
            "total_rounds": 2,
            "group": 1,
            "total_groups": 1,
            "engine": None,
        }
    }


def test_callback_scoring_scoring_step_with_engines(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    cb = MagicMock()
    engine1 = _make_llamacpp_engine(name="e1")
    engine2 = _make_openrouter_engine(name="e2")
    manifest = _make_manifest(engines=[engine1, engine2])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=3,
        prompt_groups=[["p"]],
        scoring_engines=["e1", "e2"],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    with patch.object(executor, "_resolve_engine_names_to_clients") as mock_resolve:
        rc1, sc1, sc2 = MagicMock(), MagicMock(), MagicMock()
        mock_resolve.side_effect = lambda names: {
            (): [rc1],
            ("e1", "e2"): [sc1, sc2],
        }.get(tuple(names), [])

        executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    scoring_calls = [
        c
        for c in cb.call_args_list
        if c[0][0]["tiz-internal"].get("status") == "scoring_step"
    ]
    assert len(scoring_calls) == 3
    _prompt = (
        "You are a scoring assistant. Given the following branches and the "
        "scoring prompt, select the best branch.\n"
        "\n"
        "Scoring prompt: score\n"
        "\n"
        "Branches:\n"
        "- tiz/scoring_task1_r0\n"
        "\n"
        "Return only the name of the best branch.\n"
        "Do not add any preamble or any other text."
    )
    assert scoring_calls[0][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "scoring_step",
            "task": "task1",
            "scoring_step": 1,
            "total_scoring_steps": 3,
            "prompt": _prompt,
            "engine": "e1",
        }
    }
    assert scoring_calls[1][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "scoring_step",
            "task": "task1",
            "scoring_step": 2,
            "total_scoring_steps": 3,
            "prompt": _prompt,
            "engine": "e2",
        }
    }
    assert scoring_calls[2][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "scoring_step",
            "task": "task1",
            "scoring_step": 3,
            "total_scoring_steps": 3,
            "prompt": _prompt,
            "engine": "e1",
        }
    }


def test_callback_scoring_scoring_step(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "ok",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="score",
        rounds=1,
        scoring_rounds=3,
        prompt_groups=[["p"]],
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    scoring_calls = [
        c
        for c in cb.call_args_list
        if c[0][0]["tiz-internal"].get("status") == "scoring_step"
    ]
    assert len(scoring_calls) == 3
    _prompt = (
        "You are a scoring assistant. Given the following branches and the "
        "scoring prompt, select the best branch.\n"
        "\n"
        "Scoring prompt: score\n"
        "\n"
        "Branches:\n"
        "- tiz/scoring_task1_r0\n"
        "\n"
        "Return only the name of the best branch.\n"
        "Do not add any preamble or any other text."
    )
    assert scoring_calls[0][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "scoring_step",
            "task": "task1",
            "scoring_step": 1,
            "total_scoring_steps": 3,
            "prompt": _prompt,
            "engine": None,
        }
    }
    assert scoring_calls[1][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "scoring_step",
            "task": "task1",
            "scoring_step": 2,
            "total_scoring_steps": 3,
            "prompt": _prompt,
            "engine": None,
        }
    }
    assert scoring_calls[2][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "scoring_step",
            "task": "task1",
            "scoring_step": 3,
            "total_scoring_steps": 3,
            "prompt": _prompt,
            "engine": None,
        }
    }


def test_callback_scoring_generating_iterator_items(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    _, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "itA\n",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    cb = MagicMock()
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(
        manifest=manifest, base_path=tmp_path, update_callback=cb
    )
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="pick",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["p {{item}}"]],
        iterator_input="gen items",
    )
    sandbox = MagicMock()
    sandbox.is_git_repo.return_value = True
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    _scoring_prompt = (
        "You are a scoring assistant. Given the following branches and the "
        "scoring prompt, select the best branch.\n"
        "\n"
        "Scoring prompt: pick\n"
        "\n"
        "Branches:\n"
        "- tiz/scoring_task1_r0\n"
        "\n"
        "Return only the name of the best branch.\n"
        "Do not add any preamble or any other text."
    )
    assert cb.call_args_list[0][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "generating_iterator_items",
            "task": "task1",
            "prompt": "gen items",
        }
    }
    assert cb.call_args_list[1][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "round",
            "task": "task1",
            "round": 1,
            "total_rounds": 1,
            "group": 1,
            "total_groups": 1,
            "engine": None,
        }
    }
    assert cb.call_args_list[2][0][0] == {
        "tiz-internal": {
            "action": "message_group",
            "status": "sending",
            "task": "task1",
            "message": 1,
            "total_messages": 1,
            "prompt": "p itA",
        }
    }
    assert cb.call_args_list[3][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "scoring_step",
            "task": "task1",
            "scoring_step": 1,
            "total_scoring_steps": 1,
            "prompt": _scoring_prompt,
            "engine": None,
        }
    }
    assert cb.call_args_list[4][0][0] == {
        "tiz-internal": {
            "action": "scoring",
            "status": "winner_selected",
            "task": "task1",
            "winner": "tiz/scoring_task1_r0",
            "winner_engine": None,
            "votes": 1,
            "scoring_engines": [None],
        }
    }
    assert cb.call_count == 5


# ---------------------------------------------------------------------------
# save_full_toolcalls integration
# ---------------------------------------------------------------------------


def test_save_full_toolcalls_writes_logs(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify _save_toolcalls_log creates log files when save_full_toolcalls is True."""
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": [{"tool": "read_file", "args": ["test.txt"]}],
    }
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        save_full_toolcalls=True,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._run_task(
        _make_task(
            name="toolcall_task",
            tools=[],
            actions=[PromptsAction(message_groups=[["do it"]])],
        )
    )

    log_dir = tmp_path / "logs" / "tools"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) >= 1
    content = log_files[0].read_text()
    assert "read_file" in content
    assert "test.txt" in content


def test_save_full_toolcalls_skipped_when_disabled(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify _save_toolcalls_log is skipped when save_full_toolcalls is False."""
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": [{"tool": "read_file"}],
    }
    manifest = _make_manifest(
        engines=[_make_llamacpp_engine()],
        save_full_toolcalls=False,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    executor._run_task(
        _make_task(
            name="toolcall_skip",
            tools=[],
            actions=[PromptsAction(message_groups=[["do it"]])],
        )
    )

    log_dir = tmp_path / "logs" / "tools"
    assert not log_dir.exists()


# ---------------------------------------------------------------------------
# save_full_usage_details integration
# ---------------------------------------------------------------------------


def test_save_full_usage_details_writes_logs(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify _save_full_usage creates usage log files when save_full_usage_details is True."""
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    task = _make_task(
        name="usage_task",
        tools=[],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(
        tasks=[task],
        engines=[_make_llamacpp_engine()],
        save_full_usage_details=True,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    log_dir = tmp_path / "logs" / "usage"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) >= 1
    content = log_files[0].read_text()
    assert "prompt_tokens" in content


def test_save_full_usage_details_skipped_when_disabled(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """Verify _save_full_usage is skipped when save_full_usage_details is False."""
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    task = _make_task(
        name="usage_skip",
        tools=[],
        actions=[PromptsAction(message_groups=[["do it"]])],
    )
    manifest = _make_manifest(
        tasks=[task],
        engines=[_make_llamacpp_engine()],
        save_full_usage_details=False,
    )
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    executor.execute()

    log_dir = tmp_path / "logs" / "usage"
    assert not log_dir.exists()


# ---------------------------------------------------------------------------
# save_full_logs integration – log is saved via _run_message_group
# ---------------------------------------------------------------------------


def test_run_message_group_saves_log_when_enabled(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "resp",
        "prompt_tokens": 5,
        "prompt_time": 0.1,
        "completion_tokens": 10,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    executor._run_message_group("task1", ["msg1"], "sys", [], {}, client)

    log_dir = tmp_path / "logs" / "conversations"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) == 1
    assert log_files[0].suffix == ".json"
    assert "task1" in log_files[0].name
    assert str(tmp_path / "logs" / "conversations") in str(log_files[0])


def test_run_message_group_skips_log_when_disabled(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "resp",
        "prompt_tokens": 5,
        "prompt_time": 0.1,
        "completion_tokens": 10,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=False)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    executor._run_message_group("task1", ["msg1"], "sys", [], {}, client)

    chat_instance.save.assert_not_called()


def test_run_prompts_action_saves_logs_when_enabled(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = PromptsAction(
        message_groups=[["m1", "m2"], ["m3"]],
        parallel_message_groups=False,
    )
    executor._run_prompts_action("task1", action, "sys", [], {}, client)

    log_dir = tmp_path / "logs" / "conversations"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) >= 1
    # Multiple saves may share the same timestamp second, resulting in fewer files


def test_run_iterator_action_saves_input_chat_log(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "line1\n",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "resp",
            "prompt_tokens": 5,
            "prompt_time": 0.1,
            "completion_tokens": 10,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = IteratorAction(
        input_prompt="gen",
        prompt_groups=[["p {{item}}"]],
    )
    executor._run_iterator_action("task1", action, "sys", [], {}, client)

    log_dir = tmp_path / "logs" / "conversations"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) >= 1  # saves may share the same second, producing fewer files


def test_run_repeater_action_saves_logs_when_enabled(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "r",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = RepeaterAction(
        repeat=2,
        prompt_groups=[["p{{item}}"]],
    )
    executor._run_repeater_action("task1", action, "sys", [], {}, client)

    log_dir = tmp_path / "logs" / "conversations"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) >= 1  # saves may share the same second, producing fewer files


def test_run_scoring_action_saves_input_chat_log(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    mock_chat_cls, chat_instance = mock_chat
    chat_instance.send_message.side_effect = [
        {
            "message": "itemA\n",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "done",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
        {
            "message": "tiz/scoring_task1_r0",
            "prompt_tokens": 1,
            "prompt_time": 0.1,
            "completion_tokens": 2,
            "completion_time": 0.2,
            "cost": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "tool_calls": ["tc"],
        },
    ]
    manifest = _make_manifest(engines=[_make_llamacpp_engine()], save_full_logs=True)
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)
    client = MagicMock()

    action = ScoringAction(
        scoring_prompt="pick best",
        rounds=1,
        scoring_rounds=1,
        prompt_groups=[["do {{item}}"]],
        iterator_input="generate items",
    )
    sandbox = MagicMock()
    sandbox.git_capture_branch.return_value = "main"
    sandbox.git_create_branch.return_value = None
    sandbox.git_finalize_branches.return_value = None
    sandbox.is_git_repo.return_value = True

    executor._run_scoring_action("task1", action, "sys", [], {}, sandbox, client)

    log_dir = tmp_path / "logs" / "conversations"
    assert log_dir.exists()
    log_files = list(log_dir.iterdir())
    assert len(log_files) >= 1


# ---------------------------------------------------------------------------
# fallthrough – action type does not match any isinstance check
# ---------------------------------------------------------------------------


def test_run_task_unknown_action_type(
    mock_chat: Any, mock_discover_tools: Any, tmp_path: Path
) -> None:
    """An action that doesn't match any isinstance check is silently ignored."""
    _, chat_instance = mock_chat
    chat_instance.send_message.return_value = {
        "message": "ok",
        "prompt_tokens": 1,
        "prompt_time": 0.1,
        "completion_tokens": 2,
        "completion_time": 0.2,
        "cost": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "tool_calls": ["tc"],
    }
    manifest = _make_manifest(engines=[_make_llamacpp_engine()])
    executor = ManifestExecutor(manifest=manifest, base_path=tmp_path)

    task = _make_task(
        name="unknown_action",
        tools=[],
        actions=[_UnknownAction()],
    )

    # Should not raise – unknown action is simply ignored
    executor._run_task(task)

    # No messages sent since no known action processed
    chat_instance.send_message.assert_not_called()
