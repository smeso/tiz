"""Tests for manifest_parser.py."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from tiz.manifest_parser import (
    CmdAction,
    ConfirmationSpec,
    InferenceEngineSpec,
    IteratorAction,
    Manifest,
    ManifestMeta,
    ManifestParser,
    PromptsAction,
    RepeaterAction,
    ScoringAction,
    SubagentSpec,
    TaskSpec,
    ToolSpec,
    _merge,
    merge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_META = {
    "version": "0",
    "parallelism": 4,
}

_MINIMAL_TASK = {
    "name": "test-task",
}


def _make_data(
    meta: dict[str, Any] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    inference_engines: list[dict[str, Any]] | None = None,
    audio_inference_engines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if meta is not None:
        result["meta"] = meta
    if tasks is not None:
        result["tasks"] = tasks
    if inference_engines is not None:
        result["inference_engines"] = inference_engines
    if audio_inference_engines is not None:
        result["audio_inference_engines"] = audio_inference_engines
    return result


def _make_parser_with_tools(tools_data: list[dict[str, Any]]) -> ManifestParser:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "tools": tools_data,
            }
        ],
    )
    return ManifestParser(data=data, path=None)


def _make_parser_with_actions(actions: list[dict[str, Any]]) -> ManifestParser:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "actions": actions,
            }
        ],
    )
    return ManifestParser(data=data, path=None)


# ---------------------------------------------------------------------------
# merge() and _merge()
# ---------------------------------------------------------------------------


def test_merge_single_manifest() -> None:
    parser = ManifestParser(data=_make_data(meta=_MINIMAL_META), path=None)
    m1 = Manifest(
        meta=parser.meta,
        tasks=parser.tasks,
        inference_engines=parser.inference_engines,
    )
    result = merge([m1])
    assert result.meta.version == "0"
    assert result.meta.parallelism == 4
    assert result.meta.committer_name is None
    assert result.meta.committer_email is None
    assert result.meta.container_engine is None
    assert result.meta.color is None
    assert result.tasks == []
    assert result.inference_engines == []
    assert result.audio_inference_engines == []


def test_merge_no_manifests_raises() -> None:
    with pytest.raises(ValueError, match="At least one manifest is required"):
        merge([])


def test_merge_version_mismatch_raises() -> None:
    m1 = Manifest(
        meta=ManifestMeta(
            version="1.0", parallelism=4, committer_name="A", committer_email="a@x"
        ),
        tasks=[],
        inference_engines=[],
    )
    m2 = Manifest(
        meta=ManifestMeta(
            version="2.0", parallelism=2, committer_name="B", committer_email="b@x"
        ),
        tasks=[],
        inference_engines=[],
    )
    with pytest.raises(ValueError, match="Manifest version mismatch"):
        _merge(m1, m2)


def test_merge_two_manifests() -> None:
    m1 = Manifest(
        meta=ManifestMeta(
            version="1.0",
            parallelism=4,
            committer_name="Alice",
            committer_email="alice@x",
        ),
        tasks=[
            TaskSpec(
                name="task1",
                worker_image="img1",
                worker_image_containerfile=None,
                tools=[],
                readonly_sandbox=False,
                project=None,
                sys_prompt=None,
                sys_prompt_custom=None,
                actions=[],
                allow_parallel_run=True,
                force_copy_files=[],
                inference_engine=None,
                tmpfs_root=False,
                extra_container_args=None,
            )
        ],
        inference_engines=[
            InferenceEngineSpec(
                engine_type="llamacpp",
                host="h",
                model="m",
                api_key="k",
                name="e1",
                timeout=5,
                preserve_thinking=False,
            )
        ],
    )
    m2 = Manifest(
        meta=ManifestMeta(
            version="1.0",
            parallelism=2,
            committer_name="Bob",
            committer_email="bob@x",
            container_engine="podman",
            color=False,
        ),
        tasks=[
            TaskSpec(
                name="task2",
                worker_image="img2",
                worker_image_containerfile=None,
                tools=[],
                readonly_sandbox=True,
                project=None,
                sys_prompt=None,
                sys_prompt_custom=None,
                actions=[],
                allow_parallel_run=False,
                force_copy_files=[],
                inference_engine=None,
                tmpfs_root=False,
                extra_container_args=None,
            )
        ],
        inference_engines=[],
    )
    result = _merge(m1, m2)
    assert result.meta.version == "1.0"
    assert result.meta.parallelism == 2
    assert result.meta.committer_name == "Bob"
    assert result.meta.committer_email == "bob@x"
    assert result.meta.container_engine == "podman"
    assert result.meta.color is False
    assert len(result.tasks) == 2
    assert result.tasks[0].name == "task1"
    assert result.tasks[1].name == "task2"
    assert len(result.inference_engines) == 1
    assert result.inference_engines[0].name == "e1"
    assert result.inference_engines[0].engine_type == "llamacpp"
    assert result.inference_engines[0].host == "h"
    assert result.inference_engines[0].model == "m"
    assert result.inference_engines[0].api_key == "k"
    assert result.inference_engines[0].timeout == 5
    assert result.audio_inference_engines == []


def test_merge_three_manifests() -> None:
    m1 = Manifest(
        meta=ManifestMeta(
            version="1.0", parallelism=3, committer_name="A", committer_email="a@x"
        ),
        tasks=[],
        inference_engines=[],
    )
    m2 = Manifest(
        meta=ManifestMeta(
            version="1.0", parallelism=2, committer_name="B", committer_email="b@x"
        ),
        tasks=[],
        inference_engines=[],
    )
    m3 = Manifest(
        meta=ManifestMeta(
            version="1.0", parallelism=5, committer_name="C", committer_email="c@x"
        ),
        tasks=[],
        inference_engines=[],
    )
    result = merge([m1, m2, m3])
    assert result.meta.version == "1.0"
    assert result.meta.parallelism == 5
    assert result.meta.committer_name == "C"
    assert result.meta.committer_email == "c@x"
    assert result.tasks == []
    assert result.inference_engines == []
    assert result.audio_inference_engines == []


def test_merge_propagates_engine_and_color() -> None:
    m1 = Manifest(
        meta=ManifestMeta(
            version="1.0",
            parallelism=4,
            committer_name="A",
            committer_email="a@x",
            container_engine="docker",
            color=True,
            color_reasoning="#000000",
            color_input="#ffffff",
            hide_reasoning=True,
            use_host_timezone=True,
            save_full_logs=True,
            save_full_toolcalls=False,
            save_full_usage_details=False,
            verbosity=1,
            summarizer_context_ratio=0.5,
            ring_bell=False,
            tools_default_user_agent="AgentA/1.0",
            ephemeral_sandbox=True,
        ),
        tasks=[],
        inference_engines=[],
    )
    m2 = Manifest(
        meta=ManifestMeta(
            version="1.0",
            parallelism=2,
            committer_name="B",
            committer_email="b@x",
            container_engine="podman",
            color=False,
            color_reasoning="#ffffff",
            color_input="#000000",
            hide_reasoning=False,
            use_host_timezone=False,
            save_full_logs=False,
            save_full_toolcalls=True,
            save_full_usage_details=True,
            verbosity=2,
            summarizer_context_ratio=0.8,
            delete_sandbox_on_exit=True,
            ring_bell=True,
            tools_default_user_agent="AgentB/1.0",
            ephemeral_sandbox=False,
        ),
        tasks=[],
        inference_engines=[],
    )
    result = _merge(m1, m2)
    assert result.meta.container_engine == "podman"
    assert result.meta.color is False
    assert result.meta.color_reasoning == "#ffffff"
    assert result.meta.color_input == "#000000"
    assert result.meta.hide_reasoning is False
    assert result.meta.use_host_timezone is False
    assert result.meta.save_full_logs is False
    assert result.meta.save_full_toolcalls is True
    assert result.meta.save_full_usage_details is True
    assert result.meta.verbosity == 2
    assert result.meta.summarizer_context_ratio == 0.8
    assert result.meta.delete_sandbox_on_exit is True
    assert result.meta.ring_bell is True
    assert result.meta.tools_default_user_agent == "AgentB/1.0"
    assert result.meta.ephemeral_sandbox is False


def test_merge_verbosity_last_wins() -> None:
    m1 = Manifest(
        meta=ManifestMeta(
            version="1.0",
            parallelism=4,
            committer_name="A",
            committer_email="a@x",
            verbosity=2,
        ),
        tasks=[],
        inference_engines=[],
    )
    m2 = Manifest(
        meta=ManifestMeta(
            version="1.0",
            parallelism=2,
            committer_name="B",
            committer_email="b@x",
            verbosity=0,
        ),
        tasks=[],
        inference_engines=[],
    )
    result = _merge(m1, m2)
    assert result.meta.verbosity == 0

    m3 = Manifest(
        meta=ManifestMeta(
            version="1.0",
            parallelism=4,
            committer_name="C",
            committer_email="c@x",
            verbosity=0,
        ),
        tasks=[],
        inference_engines=[],
    )
    m4 = Manifest(
        meta=ManifestMeta(
            version="1.0",
            parallelism=2,
            committer_name="D",
            committer_email="d@x",
            verbosity=1,
        ),
        tasks=[],
        inference_engines=[],
    )
    result = _merge(m3, m4)
    assert result.meta.verbosity == 1

    result = _merge(m4, m3)
    assert result.meta.verbosity == 0

    result = _merge(m2, m2)
    assert result.meta.verbosity == 0

    result = _merge(m1, m1)
    assert result.meta.verbosity == 2


# ---------------------------------------------------------------------------
# ManifestParser.__init__
# ---------------------------------------------------------------------------


def test_init_with_data() -> None:
    data = _make_data(meta=_MINIMAL_META, tasks=[_MINIMAL_TASK])
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.version == "0"
    assert parser.meta.parallelism == 4
    assert len(parser.tasks) == 1
    assert parser.tasks[0].name == "test-task"


def test_init_with_empty_data() -> None:
    parser = ManifestParser(data={}, path=None)
    assert parser.meta.version == "0"
    assert parser.meta.parallelism is None
    assert parser.meta.committer_name is None
    assert parser.meta.committer_email is None
    assert parser.tasks == []
    assert parser.inference_engines == []
    assert parser.audio_inference_engines == []


def test_init_with_path(tmp_path: Path) -> None:
    manifest_content = yaml.dump(_make_data(meta=_MINIMAL_META, tasks=[_MINIMAL_TASK]))
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(manifest_content, encoding="utf-8")
    parser = ManifestParser(data={}, path=manifest_file)
    assert parser.meta.version == "0"
    assert len(parser.tasks) == 1
    assert parser.tasks[0].name == "test-task"


def test_init_with_missing_tasks() -> None:
    data = _make_data(meta=_MINIMAL_META)
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks == []


def test_init_with_none_tasks() -> None:
    data = {"meta": _MINIMAL_META, "tasks": None}
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks == []


def test_init_with_path_and_data_override(tmp_path: Path) -> None:
    manifest_content = yaml.dump(
        {"meta": {"version": "0", "parallelism": 2}, "tasks": [{"name": "from-yaml"}]}
    )
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(manifest_content, encoding="utf-8")
    parser = ManifestParser(
        data={
            "meta": {"version": "0", "parallelism": 8},
            "tasks": [{"name": "override"}],
        },
        path=manifest_file,
    )
    assert parser.meta.version == "0"
    assert parser.meta.parallelism == 8
    assert len(parser.tasks) == 1
    assert parser.tasks[0].name == "override"


def test_get_manifest() -> None:
    data = _make_data(meta=_MINIMAL_META, tasks=[_MINIMAL_TASK])
    parser = ManifestParser(data=data, path=None)
    manifest = parser.get_manifest()
    assert isinstance(manifest, Manifest)
    assert manifest.meta.version == "0"
    assert manifest.meta.parallelism == 4
    assert len(manifest.tasks) == 1
    assert manifest.tasks[0].name == "test-task"
    assert manifest.inference_engines == []
    assert manifest.audio_inference_engines == []


# ---------------------------------------------------------------------------
# _parse_meta
# ---------------------------------------------------------------------------


def test_parse_meta_full() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "parallelism": 8,
            "committer_name": "Tester",
            "committer_email": "test@x",
            "container_engine": "docker",
            "color": False,
        }
    )
    with patch("shutil.which", return_value="/usr/bin/docker"):
        parser = ManifestParser(data=data, path=None)
    assert parser.meta.version == "0"
    assert parser.meta.parallelism == 8
    assert parser.meta.committer_name == "Tester"
    assert parser.meta.committer_email == "test@x"
    assert parser.meta.container_engine == "docker"
    assert parser.meta.color is False


def test_parse_meta_default_committer() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.committer_name is None
    assert parser.meta.committer_email is None


def test_parse_meta_committer_none_uses_default() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "committer_name": None,
            "committer_email": None,
        }
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.committer_name is None
    assert parser.meta.committer_email is None


def test_parse_meta_version_as_int() -> None:
    data = _make_data(meta={"version": 0})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.version == "0"


def test_parse_meta_version_not_zero_raises() -> None:
    data = _make_data(meta={"version": "1.0"})
    with pytest.raises(ValueError, match="Manifest 'version' must be '0'"):
        ManifestParser(data=data, path=None)


def test_parse_meta_version_as_int_nonzero_raises() -> None:
    data = _make_data(meta={"version": 1})
    with pytest.raises(ValueError, match="Manifest 'version' must be '0'"):
        ManifestParser(data=data, path=None)


def test_parse_meta_parallelism_as_string() -> None:
    data = _make_data(meta={"version": "0", "parallelism": "4"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.parallelism == 4


def test_parse_meta_container_engine() -> None:
    data = _make_data(meta={"version": "0", "container_engine": "con"})
    with patch("shutil.which", return_value="/usr/bin/con"):
        parser = ManifestParser(data=data, path=None)
    assert parser.meta.container_engine == "con"


def test_parse_meta_container_engine_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.container_engine is None


def test_parse_meta_color_true() -> None:
    data = _make_data(meta={"version": "0", "color": True})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color is True


def test_parse_meta_color_false() -> None:
    data = _make_data(meta={"version": "0", "color": False})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color is False


def test_parse_meta_color_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color is None


def test_parse_meta_engine_as_int() -> None:
    data = _make_data(meta={"version": "0", "container_engine": 42})
    with patch("shutil.which", return_value="/usr/bin/42"):
        parser = ManifestParser(data=data, path=None)
    assert parser.meta.container_engine == "42"


def test_parse_meta_container_engine_not_in_path_raises() -> None:
    data = _make_data(meta={"version": "0", "container_engine": "nonexistent"})
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(
            ValueError, match="Container engine 'nonexistent' not found in PATH"
        ),
    ):
        ManifestParser(data=data, path=None)


def test_parse_meta_color_reasoning_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_reasoning is None


def test_parse_meta_color_reasoning_custom() -> None:
    data = _make_data(meta={"version": "0", "color_reasoning": "#FF0000"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_reasoning == "#ff0000"


def test_parse_meta_color_reasoning_3_digit_expands() -> None:
    data = _make_data(meta={"version": "0", "color_reasoning": "#f0A"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_reasoning == "#ff00aa"


def test_parse_meta_color_reasoning_3_digit_uppercase() -> None:
    data = _make_data(meta={"version": "0", "color_reasoning": "#ABC"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_reasoning == "#aabbcc"


def test_parse_meta_color_reasoning_lowercase() -> None:
    data = _make_data(meta={"version": "0", "color_reasoning": "#aBcDeF"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_reasoning == "#abcdef"


def test_parse_meta_color_reasoning_invalid_no_hash_raises() -> None:
    data = _make_data(meta={"version": "0", "color_reasoning": "ff0000"})
    with pytest.raises(ValueError, match="Invalid color_reasoning"):
        ManifestParser(data=data, path=None)


def test_parse_meta_color_reasoning_invalid_hex_raises() -> None:
    data = _make_data(meta={"version": "0", "color_reasoning": "#xyz123"})
    with pytest.raises(ValueError, match="Invalid color_reasoning"):
        ManifestParser(data=data, path=None)


def test_parse_meta_color_reasoning_invalid_length_raises() -> None:
    data = _make_data(meta={"version": "0", "color_reasoning": "#1234567"})
    with pytest.raises(ValueError, match="Invalid color_reasoning"):
        ManifestParser(data=data, path=None)


def test_parse_meta_color_reasoning_invalid_2_digit_raises() -> None:
    data = _make_data(meta={"version": "0", "color_reasoning": "#12"})
    with pytest.raises(ValueError, match="Invalid color_reasoning"):
        ManifestParser(data=data, path=None)


def test_parse_meta_color_input_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_input is None


def test_parse_meta_color_input_custom() -> None:
    data = _make_data(meta={"version": "0", "color_input": "#FF0000"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_input == "#ff0000"


def test_parse_meta_color_input_3_digit_expands() -> None:
    data = _make_data(meta={"version": "0", "color_input": "#f0A"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_input == "#ff00aa"


def test_parse_meta_color_input_3_digit_uppercase() -> None:
    data = _make_data(meta={"version": "0", "color_input": "#ABC"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_input == "#aabbcc"


def test_parse_meta_color_input_lowercase() -> None:
    data = _make_data(meta={"version": "0", "color_input": "#aBcDeF"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color_input == "#abcdef"


def test_parse_meta_color_input_invalid_no_hash_raises() -> None:
    data = _make_data(meta={"version": "0", "color_input": "ff0000"})
    with pytest.raises(ValueError, match="Invalid color_input"):
        ManifestParser(data=data, path=None)


def test_parse_meta_color_input_invalid_hex_raises() -> None:
    data = _make_data(meta={"version": "0", "color_input": "#xyz123"})
    with pytest.raises(ValueError, match="Invalid color_input"):
        ManifestParser(data=data, path=None)


def test_parse_meta_color_input_invalid_length_raises() -> None:
    data = _make_data(meta={"version": "0", "color_input": "#1234567"})
    with pytest.raises(ValueError, match="Invalid color_input"):
        ManifestParser(data=data, path=None)


def test_parse_meta_color_input_invalid_2_digit_raises() -> None:
    data = _make_data(meta={"version": "0", "color_input": "#12"})
    with pytest.raises(ValueError, match="Invalid color_input"):
        ManifestParser(data=data, path=None)


def test_parse_meta_hide_reasoning_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.hide_reasoning is None


def test_parse_meta_hide_reasoning_true() -> None:
    data = _make_data(meta={"version": "0", "hide_reasoning": True})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.hide_reasoning is True


def test_parse_meta_hide_reasoning_false() -> None:
    data = _make_data(meta={"version": "0", "hide_reasoning": False})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.hide_reasoning is False


def test_parse_meta_use_host_timezone_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.use_host_timezone is None


def test_parse_meta_use_host_timezone_true() -> None:
    data = _make_data(meta={"version": "0", "use_host_timezone": True})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.use_host_timezone is True


def test_parse_meta_use_host_timezone_false() -> None:
    data = _make_data(meta={"version": "0", "use_host_timezone": False})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.use_host_timezone is False


def test_parse_meta_save_full_logs_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_logs is None


def test_parse_meta_save_full_logs_true() -> None:
    data = _make_data(meta={"version": "0", "save_full_logs": True})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_logs is True


def test_parse_meta_save_full_logs_false() -> None:
    data = _make_data(meta={"version": "0", "save_full_logs": False})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_logs is False


def test_parse_meta_save_full_toolcalls_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_toolcalls is None


def test_parse_meta_save_full_toolcalls_false() -> None:
    data = _make_data(meta={"version": "0", "save_full_toolcalls": False})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_toolcalls is False


def test_parse_meta_save_full_toolcalls_true() -> None:
    data = _make_data(meta={"version": "0", "save_full_toolcalls": True})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_toolcalls is True


def test_parse_meta_save_full_usage_details_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_usage_details is None


def test_parse_meta_save_full_usage_details_false() -> None:
    data = _make_data(meta={"version": "0", "save_full_usage_details": False})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_usage_details is False


def test_parse_meta_verbosity_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.verbosity is None


def test_parse_meta_verbosity_0() -> None:
    data = _make_data(meta={"version": "0", "verbosity": 0})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.verbosity == 0


def test_parse_meta_verbosity_1() -> None:
    data = _make_data(meta={"version": "0", "verbosity": 1})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.verbosity == 1


def test_parse_meta_verbosity_2() -> None:
    data = _make_data(meta={"version": "0", "verbosity": 2})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.verbosity == 2


def test_parse_meta_verbosity_as_string() -> None:
    data = _make_data(meta={"version": "0", "verbosity": "1"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.verbosity == 1


def test_parse_meta_verbosity_invalid_raises() -> None:
    data = _make_data(meta={"version": "0", "verbosity": 3})
    with pytest.raises(ValueError, match="Manifest 'verbosity' must be 0, 1, or 2"):
        ManifestParser(data=data, path=None)


def test_parse_meta_verbosity_negative_raises() -> None:
    data = _make_data(meta={"version": "0", "verbosity": -1})
    with pytest.raises(ValueError, match="Manifest 'verbosity' must be 0, 1, or 2"):
        ManifestParser(data=data, path=None)


def test_parse_meta_summarizer_context_ratio_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.summarizer_context_ratio is None


def test_parse_meta_summarizer_context_ratio_custom() -> None:
    data = _make_data(meta={"version": "0", "summarizer_context_ratio": 0.5})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.summarizer_context_ratio == 0.5


def test_parse_meta_summarizer_context_ratio_as_string() -> None:
    data = _make_data(meta={"version": "0", "summarizer_context_ratio": "0.75"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.summarizer_context_ratio == 0.75


def test_parse_meta_all_new_options() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "parallelism": 4,
            "color": False,
            "color_reasoning": "#AbCdEf",
            "color_input": "#123456",
            "hide_reasoning": True,
            "use_host_timezone": False,
            "save_full_logs": True,
            "save_full_toolcalls": False,
            "save_full_usage_details": False,
            "verbosity": 2,
            "summarizer_context_ratio": 0.3,
        }
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color is False
    assert parser.meta.color_reasoning == "#abcdef"
    assert parser.meta.color_input == "#123456"
    assert parser.meta.hide_reasoning is True
    assert parser.meta.use_host_timezone is False
    assert parser.meta.save_full_logs is True
    assert parser.meta.save_full_toolcalls is False
    assert parser.meta.save_full_usage_details is False
    assert parser.meta.verbosity == 2
    assert parser.meta.summarizer_context_ratio == 0.3


def test_parse_meta_delete_sandbox_on_exit_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.delete_sandbox_on_exit is None


def test_parse_meta_delete_sandbox_on_exit_true() -> None:
    data = _make_data(meta={"version": "0", "delete_sandbox_on_exit": True})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.delete_sandbox_on_exit is True


def test_parse_meta_delete_sandbox_on_exit_false() -> None:
    data = _make_data(meta={"version": "0", "delete_sandbox_on_exit": False})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.delete_sandbox_on_exit is False


def test_parse_meta_ephemeral_sandbox_default() -> None:
    data = _make_data(meta={"version": "0"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.ephemeral_sandbox is None
    assert parser.meta.delete_sandbox_on_exit is None


def test_parse_meta_ephemeral_sandbox_true_implies_delete() -> None:
    data = _make_data(meta={"version": "0", "ephemeral_sandbox": True})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.ephemeral_sandbox is True
    assert parser.meta.delete_sandbox_on_exit is True


def test_parse_meta_ephemeral_sandbox_true_with_delete_true() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "ephemeral_sandbox": True,
            "delete_sandbox_on_exit": True,
        }
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.ephemeral_sandbox is True
    assert parser.meta.delete_sandbox_on_exit is True


def test_parse_meta_ephemeral_sandbox_false_with_delete_false() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "ephemeral_sandbox": False,
            "delete_sandbox_on_exit": False,
        }
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.ephemeral_sandbox is False
    assert parser.meta.delete_sandbox_on_exit is False


def test_parse_meta_ephemeral_sandbox_false_with_delete_true() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "ephemeral_sandbox": False,
            "delete_sandbox_on_exit": True,
        }
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.ephemeral_sandbox is False
    assert parser.meta.delete_sandbox_on_exit is True


def test_parse_meta_ephemeral_sandbox_true_delete_false_raises() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "ephemeral_sandbox": True,
            "delete_sandbox_on_exit": False,
        }
    )
    with pytest.raises(
        ValueError,
        match="If 'ephemeral_sandbox' is True then 'delete_sandbox_on_exit' must be True",
    ):
        ManifestParser(data=data, path=None)


# ---------------------------------------------------------------------------
# _parse_inference_engines
# ---------------------------------------------------------------------------
def test_parse_inference_engine_no_api_key() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "no-key-engine",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].api_key is None
    assert parser.inference_engines[0].engine_type == "llamacpp"
    assert parser.inference_engines[0].name == "no-key-engine"


def test_parse_inference_engine_single_engine_plain_key() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "gpt-engine",
                "model": "gpt-4",
                "api_key": "plain:sk-123",
                "host": "https://api.llamacpp.com",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert len(parser.inference_engines) == 1
    eng = parser.inference_engines[0]
    assert eng.engine_type == "llamacpp"
    assert eng.name == "gpt-engine"
    assert eng.model == "gpt-4"
    assert eng.api_key == "sk-123"
    assert eng.host == "https://api.llamacpp.com"
    assert eng.timeout == 5
    assert eng.verify_ssl is True
    assert eng.ca_cert is None
    assert eng.sampling_params is None
    assert eng.preserve_thinking is False


def test_parse_inference_engine_with_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_API_KEY", "secret-key-val")
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "gpt-engine",
                "model": "gpt-4",
                "api_key": "env:MY_API_KEY",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].api_key == "secret-key-val"


def test_parse_inference_engine_with_missing_env_key_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "gpt-engine",
                "api_key": "env:NONEXISTENT_VAR_XYZ",
            }
        ],
    )
    with pytest.raises(
        ValueError, match="Environment variable 'NONEXISTENT_VAR_XYZ' is not set"
    ):
        ManifestParser(data=data, path=None)


def test_parse_inference_engine_with_stdin_key() -> None:
    with patch("getpass.getpass", return_value="stdin-key") as mock_getpass:
        data = _make_data(
            meta=_MINIMAL_META,
            inference_engines=[
                {
                    "type": "llamacpp",
                    "name": "gpt-engine",
                    "api_key": "stdin",
                }
            ],
        )
        parser = ManifestParser(data=data, path=None)
        assert parser.inference_engines[0].api_key == "stdin-key"
        mock_getpass.assert_called_once()


def test_parse_inference_engine_raw_key() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "gpt-engine",
                "api_key": "raw-key",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].api_key == "raw-key"


def test_parse_inference_engine_missing_type_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "name": "gpt-engine",
            }
        ],
    )
    with pytest.raises(ValueError, match="Inference engine 'type' is required"):
        ManifestParser(data=data, path=None)


def test_parse_inference_engine_missing_name_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
            }
        ],
    )
    with pytest.raises(ValueError, match="Inference engine 'name' is required"):
        ManifestParser(data=data, path=None)


def test_parse_inference_engine_full_options() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "full-engine",
                "model": "gpt-4",
                "api_key": "plain:k",
                "host": "http://localhost",
                "timeout": 30,
                "message_timeout": 120,
                "verify_ssl": False,
                "ca_cert": "/path/to/cert",
                "sampling_params": {"temperature": 0.5},
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    eng = parser.inference_engines[0]
    assert eng.engine_type == "llamacpp"
    assert eng.name == "full-engine"
    assert eng.model == "gpt-4"
    assert eng.api_key == "k"
    assert eng.host == "http://localhost"
    assert eng.timeout == 30
    assert eng.message_timeout == 120
    assert eng.verify_ssl is False
    assert eng.ca_cert == "/path/to/cert"
    assert eng.sampling_params == {"temperature": 0.5}


def test_parse_inference_engine_message_timeout_default() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "api_key": "plain:k",
                "timeout": 10,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    eng = parser.inference_engines[0]
    assert eng.message_timeout is None


def test_parse_inference_engine_message_timeout_zero() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "api_key": "plain:k",
                "timeout": 10,
                "message_timeout": 0,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    eng = parser.inference_engines[0]
    assert eng.message_timeout == 0


def test_parse_inference_engine_invalid_sampling_params_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e",
                "api_key": "plain:k",
                "sampling_params": "not-a-dict",
            }
        ],
    )
    with pytest.raises(ValueError, match="'sampling_params' must be a dict"):
        ManifestParser(data=data, path=None)


def test_parse_inference_engine_empty_name_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "",
            }
        ],
    )
    with pytest.raises(ValueError, match="Inference engine 'name' is required"):
        ManifestParser(data=data, path=None)


def test_parse_inference_engine_empty_type_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "  ",
                "name": "e",
            }
        ],
    )
    with pytest.raises(ValueError, match="Inference engine 'type' is required"):
        ManifestParser(data=data, path=None)


def test_parse_inference_engine_preserve_thinking_default() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "api_key": "plain:k",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].preserve_thinking is False


def test_parse_inference_engine_preserve_thinking_true() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "api_key": "plain:k",
                "preserve_thinking": True,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].preserve_thinking is True


def test_parse_inference_engine_preserve_thinking_false() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "api_key": "plain:k",
                "preserve_thinking": False,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].preserve_thinking is False


def test_parse_inference_engine_multiple_engines() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {"type": "llamacpp", "name": "e1", "api_key": "plain:k1"},
            {"type": "openrouter", "name": "e2", "api_key": "plain:k2"},
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert len(parser.inference_engines) == 2
    assert parser.inference_engines[0].engine_type == "llamacpp"
    assert parser.inference_engines[0].name == "e1"
    assert parser.inference_engines[0].api_key == "k1"
    assert parser.inference_engines[1].engine_type == "openrouter"
    assert parser.inference_engines[1].name == "e2"
    assert parser.inference_engines[1].api_key == "k2"


def test_parse_inference_engine_no_engines() -> None:
    data = _make_data(meta=_MINIMAL_META, inference_engines=[])
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines == []


# ---------------------------------------------------------------------------
# _parse_audio_inference_engines
# ---------------------------------------------------------------------------


def test_parse_audio_inference_engine_minimal() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "audio-1",
                "host": "http://localhost:8080",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert len(parser.audio_inference_engines) == 1
    eng = parser.audio_inference_engines[0]
    assert eng.engine_type == "whispercpp"
    assert eng.name == "audio-1"
    assert eng.host == "http://localhost:8080"
    assert eng.timeout == 5
    assert eng.inference_timeout is None
    assert eng.verify_ssl is True
    assert eng.ca_cert is None
    assert eng.sampling_params is None
    assert eng.language is None
    assert eng.prompt is None


def test_parse_audio_inference_engine_type_whispercpp_case_insensitive() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "WHISPERCPP",
                "name": "audio-1",
                "host": "http://localhost:8080",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].engine_type == "whispercpp"


def test_parse_audio_inference_engine_type_whisper_dot_cpp() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whisper.cpp",
                "name": "audio-1",
                "host": "http://localhost:8080",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].engine_type == "whisper.cpp"


def test_parse_audio_inference_engine_type_whisper_dot_cpp_case_insensitive() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "WHISPER.CPP",
                "name": "audio-1",
                "host": "http://localhost:8080",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].engine_type == "whisper.cpp"


def test_parse_audio_inference_engine_missing_type_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "name": "audio-1",
                "host": "http://localhost:8080",
            }
        ],
    )
    with pytest.raises(ValueError, match="Audio inference engine 'type' is required"):
        ManifestParser(data=data, path=None)


def test_parse_audio_inference_engine_empty_type_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "   ",
                "name": "audio-1",
                "host": "http://localhost:8080",
            }
        ],
    )
    with pytest.raises(
        ValueError,
        match="Audio inference engine 'type' must be 'whispercpp' or 'whisper.cpp'",
    ):
        ManifestParser(data=data, path=None)


def test_parse_audio_inference_engine_invalid_type_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "invalid-engine",
                "name": "audio-1",
                "host": "http://localhost:8080",
            }
        ],
    )
    with pytest.raises(
        ValueError,
        match="Audio inference engine 'type' must be 'whispercpp' or 'whisper.cpp'",
    ):
        ManifestParser(data=data, path=None)


def test_parse_audio_inference_engine_missing_name_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "host": "http://localhost:8080",
            }
        ],
    )
    with pytest.raises(ValueError, match="Audio inference engine 'name' is required"):
        ManifestParser(data=data, path=None)


def test_parse_audio_inference_engine_empty_name_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "",
                "host": "http://localhost:8080",
            }
        ],
    )
    with pytest.raises(ValueError, match="Audio inference engine 'name' is required"):
        ManifestParser(data=data, path=None)


def test_parse_audio_inference_engine_full_options() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "full-audio",
                "host": "http://audio-host:9090",
                "timeout": 30,
                "inference_timeout": 120,
                "verify_ssl": False,
                "ca_cert": "/path/to/ca.pem",
                "sampling_params": {"temperature": 0.2},
                "language": "en",
                "prompt": "Transcribe the following",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert len(parser.audio_inference_engines) == 1
    eng = parser.audio_inference_engines[0]
    assert eng.engine_type == "whispercpp"
    assert eng.name == "full-audio"
    assert eng.host == "http://audio-host:9090"
    assert eng.timeout == 30
    assert eng.inference_timeout == 120
    assert eng.verify_ssl is False
    assert eng.ca_cert == "/path/to/ca.pem"
    assert eng.sampling_params == {"temperature": 0.2}
    assert eng.language == "en"
    assert eng.prompt == "Transcribe the following"


def test_parse_audio_inference_engine_custom_timeout() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "timeout-engine",
                "host": "http://localhost:8080",
                "timeout": 15,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].timeout == 15


def test_parse_audio_inference_engine_inference_timeout_zero() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "inference_timeout": 0,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].inference_timeout == 0


def test_parse_audio_inference_engine_verify_ssl_false() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "verify_ssl": False,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].verify_ssl is False


def test_parse_audio_inference_engine_ca_cert() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "ca_cert": "/custom/ca.pem",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].ca_cert == "/custom/ca.pem"


def test_parse_audio_inference_engine_invalid_sampling_params_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "sampling_params": "not-a-dict",
            }
        ],
    )
    with pytest.raises(ValueError, match="'sampling_params' must be a dict"):
        ManifestParser(data=data, path=None)


def test_parse_audio_inference_engine_language() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "language": "fr",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].language == "fr"


def test_parse_audio_inference_engine_prompt() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "prompt": "This is a test prompt",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].prompt == "This is a test prompt"


def test_parse_audio_inference_engine_multiple_engines() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "engine-a",
                "host": "http://host-a:8080",
            },
            {
                "type": "whisper.cpp",
                "name": "engine-b",
                "host": "http://host-b:8080",
            },
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert len(parser.audio_inference_engines) == 2
    assert parser.audio_inference_engines[0].engine_type == "whispercpp"
    assert parser.audio_inference_engines[0].name == "engine-a"
    assert parser.audio_inference_engines[1].engine_type == "whisper.cpp"
    assert parser.audio_inference_engines[1].name == "engine-b"


def test_parse_audio_inference_engine_no_engines() -> None:
    data = _make_data(meta=_MINIMAL_META, audio_inference_engines=[])
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines == []


def test_parse_audio_inference_engine_hyphenated_keys() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "inference-timeout": 60,
                "verify-ssl": False,
                "ca-cert": "/ca.pem",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    eng = parser.audio_inference_engines[0]
    assert eng.inference_timeout == 60
    assert eng.verify_ssl is False
    assert eng.ca_cert == "/ca.pem"


def test_parse_audio_inference_engine_underscore_key_wins() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "inference_timeout": 120,
                "inference-timeout": 60,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].inference_timeout == 120


# ---------------------------------------------------------------------------
# _parse_tools
# ---------------------------------------------------------------------------


def test_parse_tools_simple_tool() -> None:
    parser = _make_parser_with_tools([{"BashTool": "internet"}])
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert isinstance(tools[0], ToolSpec)
    assert tools[0].name == "BashTool"
    assert tools[0].network == "internet"
    assert tools[0].disk_mode == "nodisk"


def test_parse_tools_tool_no_network() -> None:
    parser = _make_parser_with_tools([{"ToolA": "disk"}])
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "ToolA"
    assert tools[0].network == "none"
    assert tools[0].disk_mode == "disk"


def test_parse_tools_tool_multiple_network_flags() -> None:
    parser = _make_parser_with_tools([{"ToolA": ["internet", "ro-disk"]}])
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "ToolA"
    assert tools[0].network == "internet"
    assert tools[0].disk_mode == "ro-disk"


def test_parse_tools_tool_disk_mode_precedence() -> None:
    parser = _make_parser_with_tools([{"ToolA": ["nodisk", "ro-disk", "disk"]}])
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].disk_mode == "disk"


def test_parse_tools_meta_tool_expansion() -> None:
    parser = _make_parser_with_tools([{"standard_file_manipulation": "internet"}])
    tools = parser.tasks[0].tools
    assert len(tools) == 9
    assert tools[0].name == "Edit"
    expected_tools = ManifestParser._META_TOOL["standard_file_manipulation"]
    assert [t.name for t in tools] == expected_tools
    for tool in tools:
        assert tool.network == "internet"
        assert tool.disk_mode == "nodisk"


def test_parse_tools_meta_tool_expansion_dedup() -> None:
    parser = _make_parser_with_tools(
        [
            {"standard_file_manipulation": ["internet", "disk"]},
            {"standard_file_manipulation": ["internet", "disk"]},
        ]
    )
    tools = parser.tasks[0].tools
    assert len(tools) == 9


def test_parse_tools_empty_tools() -> None:
    parser = _make_parser_with_tools([])
    assert parser.tasks[0].tools == []


def test_parse_tools_tool_is_str_wrapped_in_list() -> None:
    parser = _make_parser_with_tools([{"ToolA": "internet"}])
    tools = parser.tasks[0].tools
    assert isinstance(tools[0].network, str)
    assert tools[0].network == "internet"


def test_parse_tools_tool_unknown_token_raises() -> None:
    with pytest.raises(ValueError, match="Unknown disk or network modes"):
        _make_parser_with_tools([{"ToolA": ["internet", "unknown-mode"]}])


def test_parse_tools_ns_network_mode() -> None:
    parser = _make_parser_with_tools([{"ToolA": "ns:mynet"}])
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "ToolA"
    assert tools[0].network == "ns:mynet"
    assert tools[0].disk_mode == "nodisk"


def test_parse_tools_with_config_dict() -> None:
    parser = _make_parser_with_tools(
        [
            {
                "ToolA": [
                    "internet",
                    "disk",
                    {"config": {"user_agent": "CustomAgent/1.0"}},
                ]
            }
        ]
    )
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "ToolA"
    assert tools[0].network == "internet"
    assert tools[0].disk_mode == "disk"
    assert tools[0].config == {"user_agent": "CustomAgent/1.0"}


def test_parse_tools_with_multiple_config_dicts() -> None:
    parser = _make_parser_with_tools(
        [
            {
                "ToolA": [
                    "internet",
                    {"config": {"key1": "val1"}},
                    {"config": {"key2": "val2"}},
                ]
            }
        ]
    )
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "ToolA"
    assert tools[0].config == {"key1": "val1", "key2": "val2"}


def test_parse_tools_webfetch_with_default_user_agent() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "parallelism": 1,
            "tools_default_user_agent": "DefaultAgent/1.0",
        },
        tasks=[
            {
                "name": "t",
                "tools": [{"WebFetch": "internet"}],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "WebFetch"
    assert tools[0].config.get("user_agent") == "DefaultAgent/1.0"


def test_parse_tools_webfetch_with_explicit_user_agent_not_overridden() -> None:
    """WebFetch with explicit user_agent in config should not be overridden by default."""
    data = _make_data(
        meta={
            "version": "0",
            "parallelism": 1,
            "tools_default_user_agent": "DefaultAgent/1.0",
        },
        tasks=[
            {
                "name": "t",
                "tools": [
                    {
                        "WebFetch": [
                            "internet",
                            {"config": {"user_agent": "ExplicitAgent/1.0"}},
                        ]
                    }
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "WebFetch"
    assert tools[0].config.get("user_agent") == "ExplicitAgent/1.0"


# ---------------------------------------------------------------------------
# _parse_actions
# ---------------------------------------------------------------------------


def test_parse_actions_prompts_single_group() -> None:
    parser = _make_parser_with_actions([{"prompts": [["hello", "world"]]}])
    actions = parser.tasks[0].actions
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, PromptsAction)
    assert action.message_groups == [["hello", "world"]]
    assert action.parallel_message_groups is False


def test_parse_actions_prompts_multiple_groups() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "prompts": [
                    ["hello", "world"],
                    ["foo", "bar"],
                ],
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, PromptsAction)
    assert len(action.message_groups) == 2
    assert action.message_groups[0] == ["hello", "world"]
    assert action.message_groups[1] == ["foo", "bar"]
    assert action.parallel_message_groups is False


def test_parse_actions_prompts_parallel() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "prompts": ["msg"],
                "parallel_message_groups": True,
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, PromptsAction)
    assert action.parallel_message_groups is True
    assert action.message_groups == [["msg"]]


def test_parse_actions_prompts_single_string_group() -> None:
    parser = _make_parser_with_actions([{"prompts": [["single-prompt"]]}])
    action = parser.tasks[0].actions[0]
    assert isinstance(action, PromptsAction)
    assert action.message_groups == [["single-prompt"]]


def test_parse_actions_prompts_empty_returns_no_action() -> None:
    parser = _make_parser_with_actions([{"prompts": []}])
    assert parser.tasks[0].actions == []


def test_parse_actions_cmd_action() -> None:
    parser = _make_parser_with_actions([{"cmd": "SYNC"}])
    action = parser.tasks[0].actions[0]
    assert isinstance(action, CmdAction)
    assert action.command == "sync"


def test_parse_actions_cmd_action_lowercased() -> None:
    parser = _make_parser_with_actions([{"cmd": "MyCommand"}])
    action = parser.tasks[0].actions[0]
    assert isinstance(action, CmdAction)
    assert action.command == "mycommand"


def test_parse_actions_iterator_action() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "iterator": {
                    "input": "generate list",
                    "prompts": [
                        ["process {item}", "next {item}"],
                        ["final {item}"],
                    ],
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, IteratorAction)
    assert action.input_prompt == "generate list"
    assert action.prompt_groups == [
        ["process {item}", "next {item}"],
        ["final {item}"],
    ]
    assert action.parallel_message_groups is False


def test_parse_actions_iterator_action_parallel() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "iterator": {
                    "input": "gen",
                    "prompts": [["step"]],
                    "parallel_message_groups": True,
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, IteratorAction)
    assert action.parallel_message_groups is True
    assert action.input_prompt == "gen"
    assert action.prompt_groups == [["step"]]


def test_parse_actions_iterator_action_empty_prompts_skips() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "iterator": {
                    "input": "gen",
                    "prompts": [],
                },
            }
        ]
    )
    assert (
        len([a for a in parser.tasks[0].actions if isinstance(a, IteratorAction)]) == 0
    )


def test_parse_actions_iterator_action_non_list_prompt_group() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "iterator": {
                    "input": "gen",
                    "prompts": ["single-item"],
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, IteratorAction)
    assert action.prompt_groups == [["single-item"]]


def test_parse_actions_repeater_action() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "repeater": {
                    "repeat": 5,
                    "prompts": [
                        ["repeat-msg-1"],
                        ["repeat-msg-2"],
                    ],
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, RepeaterAction)
    assert action.repeat == 5
    assert action.prompt_groups == [
        ["repeat-msg-1"],
        ["repeat-msg-2"],
    ]


def test_parse_actions_repeater_action_default_repeat() -> None:
    parser = _make_parser_with_actions([{"repeater": {"prompts": [["msg"]]}}])
    action = parser.tasks[0].actions[0]
    assert isinstance(action, RepeaterAction)
    assert action.repeat == 1
    assert action.prompt_groups == [["msg"]]


def test_parse_actions_repeater_action_empty_prompts_skips() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "repeater": {
                    "repeat": 3,
                    "prompts": [],
                },
            }
        ]
    )
    assert (
        len([a for a in parser.tasks[0].actions if isinstance(a, RepeaterAction)]) == 0
    )


def test_parse_actions_repeater_non_list_prompt_group() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "repeater": {
                    "repeat": 1,
                    "prompts": ["single-item"],
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, RepeaterAction)
    assert action.repeat == 1
    assert action.prompt_groups == [["single-item"]]


def test_parse_actions_scoring_action() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "scoring": {
                    "scoring_prompt": "score it",
                    "rounds": 3,
                    "scoring_rounds": 2,
                    "prompts": [
                        ["option-a"],
                        ["option-b"],
                    ],
                    "iterator_input": "generate",
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, ScoringAction)
    assert action.scoring_prompt == "score it"
    assert action.rounds == 3
    assert action.scoring_rounds == 2
    assert action.prompt_groups == [["option-a"], ["option-b"]]
    assert action.iterator_input == "generate"
    assert action.engines == []
    assert action.scoring_engines == []


def test_parse_actions_scoring_action_defaults() -> None:
    parser = _make_parser_with_actions([{"scoring": {"prompts": [["a"]]}}])
    action = parser.tasks[0].actions[0]
    assert isinstance(action, ScoringAction)
    assert action.scoring_prompt == ""
    assert action.rounds == 1
    assert action.scoring_rounds == 1
    assert action.prompt_groups == [["a"]]
    assert action.iterator_input is None


def test_parse_actions_scoring_action_empty_prompts_skips() -> None:
    parser = _make_parser_with_actions([{"scoring": {"prompts": []}}])
    assert (
        len([a for a in parser.tasks[0].actions if isinstance(a, ScoringAction)]) == 0
    )


def test_parse_actions_scoring_non_list_prompt_group() -> None:
    parser = _make_parser_with_actions([{"scoring": {"prompts": ["single-item"]}}])
    action = parser.tasks[0].actions[0]
    assert isinstance(action, ScoringAction)
    assert action.prompt_groups == [["single-item"]]


def test_parse_actions_scoring_engines_defaults() -> None:
    parser = _make_parser_with_actions([{"scoring": {"prompts": [["a"]]}}])
    action = parser.tasks[0].actions[0]
    assert isinstance(action, ScoringAction)
    assert action.engines == []
    assert action.scoring_engines == []


def test_parse_actions_scoring_engines_list() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "scoring": {
                    "prompts": [["a"]],
                    "engines": ["e1", "e2"],
                    "scoring_engines": ["se1", "se2"],
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, ScoringAction)
    assert action.engines == ["e1", "e2"]
    assert action.scoring_engines == ["se1", "se2"]


def test_parse_actions_scoring_engines_single_string() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "scoring": {
                    "prompts": [["a"]],
                    "engines": "single-engine",
                    "scoring_engines": "single-scoring-engine",
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, ScoringAction)
    assert action.engines == ["single-engine"]
    assert action.scoring_engines == ["single-scoring-engine"]


def test_parse_actions_scoring_engines_none() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "scoring": {
                    "prompts": [["a"]],
                    "engines": None,
                    "scoring_engines": None,
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, ScoringAction)
    assert action.engines == []
    assert action.scoring_engines == []


def test_parse_actions_multiple_action_types() -> None:
    parser = _make_parser_with_actions(
        [
            {"prompts": [["hello"]]},
            {"cmd": "sync"},
            {"iterator": {"input": "gen", "prompts": [["step"]]}},
            {"repeater": {"repeat": 2, "prompts": [["r"]]}},
            {"scoring": {"prompts": [["s"]]}},
        ]
    )
    actions = parser.tasks[0].actions
    assert len(actions) == 5

    assert isinstance(actions[0], PromptsAction)
    assert actions[0].message_groups == [["hello"]]
    assert actions[0].parallel_message_groups is False

    assert isinstance(actions[1], CmdAction)
    assert actions[1].command == "sync"

    assert isinstance(actions[2], IteratorAction)
    assert actions[2].input_prompt == "gen"
    assert actions[2].prompt_groups == [["step"]]
    assert actions[2].parallel_message_groups is False

    assert isinstance(actions[3], RepeaterAction)
    assert actions[3].repeat == 2
    assert actions[3].prompt_groups == [["r"]]

    assert isinstance(actions[4], ScoringAction)
    assert actions[4].scoring_prompt == ""
    assert actions[4].rounds == 1
    assert actions[4].scoring_rounds == 1
    assert actions[4].prompt_groups == [["s"]]
    assert actions[4].iterator_input is None
    assert actions[4].engines == []
    assert actions[4].scoring_engines == []


def test_parse_actions_no_actions() -> None:
    data = _make_data(meta=_MINIMAL_META, tasks=[{"name": "t", "actions": []}])
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].actions == []


def test_parse_actions_none_actions() -> None:
    data = _make_data(meta=_MINIMAL_META, tasks=[{"name": "t", "actions": None}])
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].actions == []


def test_parse_actions_unknown_action_entry_raises() -> None:
    with pytest.raises(ValueError, match="Unknown action type in manifest"):
        _make_parser_with_actions([{"unknown_key": "value"}, {"prompts": [["hello"]]}])


# ---------------------------------------------------------------------------
# _parse_task
# ---------------------------------------------------------------------------


def test_parse_task_minimal_task() -> None:
    data = _make_data(meta=_MINIMAL_META, tasks=[{"name": "t"}])
    parser = ManifestParser(data=data, path=None)
    task = parser.tasks[0]
    assert task.name == "t"
    assert task.worker_image == "tiz-worker:latest"
    assert task.worker_image_containerfile is None
    assert task.tools == []
    assert task.readonly_sandbox is False
    assert task.project is None
    assert task.sys_prompt is None
    assert task.sys_prompt_custom is None
    assert task.actions == []
    assert task.allow_parallel_run is True
    assert task.force_copy_files == []
    assert task.inference_engine is None
    assert task.tmpfs_root is False
    assert task.extra_container_args is None
    assert task.dedicated_audio_engine is None
    assert task.subagents == []


def test_parse_task_all_fields() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "complex-task",
                "worker_image": "custom- worker:v2",
                "worker_image_containerfile": "/path/to/Dockerfile",
                "tools": [{"ToolA": "internet"}],
                "readonly_sandbox": True,
                "project": "/my/project",
                "sys_prompt": "custom prompt",
                "sys_prompt_custom": "custom override",
                "allow_parallel_run": False,
                "force_copy_files": ["/src", "/lib"],
                "inference_engine": "my-engine",
                "tmpfs_root": True,
                "extra_container_args": [
                    "--cap-add=SYS_ADMIN",
                    "--security-opt=apparmor:unconfined",
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    task = parser.tasks[0]
    assert task.name == "complex-task"
    assert task.worker_image == "custom- worker:v2"
    assert task.worker_image_containerfile == "/path/to/Dockerfile"
    assert len(task.tools) == 1
    assert task.tools[0].name == "ToolA"
    assert task.tools[0].network == "internet"
    assert task.tools[0].disk_mode == "nodisk"
    assert task.readonly_sandbox is True
    assert task.project == "/my/project"
    assert task.sys_prompt == "custom prompt"
    assert task.sys_prompt_custom == "custom override"
    assert task.allow_parallel_run is False
    assert task.force_copy_files == ["/src", "/lib"]
    assert task.inference_engine == "my-engine"
    assert task.tmpfs_root is True
    assert task.extra_container_args == [
        "--cap-add=SYS_ADMIN",
        "--security-opt=apparmor:unconfined",
    ]
    assert task.dedicated_audio_engine is None
    assert task.subagents == []


def test_parse_task_empty_name_raises() -> None:
    data = _make_data(meta=_MINIMAL_META, tasks=[{"name": ""}])
    with pytest.raises(ValueError, match="Task name is required and cannot be empty"):
        ManifestParser(data=data, path=None)


def test_parse_task_missing_name_raises() -> None:
    data = _make_data(meta=_MINIMAL_META, tasks=[{}])
    with pytest.raises(ValueError, match="Task name is required and cannot be empty"):
        ManifestParser(data=data, path=None)


def test_parse_task_force_copy_files_string() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "force_copy_files": "/single/path",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].force_copy_files == ["/single/path"]


def test_parse_task_empty_force_copy_files() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "force_copy_files": [],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].force_copy_files == []


def test_parse_task_extra_container_args_default_none() -> None:
    data = _make_data(meta=_MINIMAL_META, tasks=[{"name": "t"}])
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].extra_container_args is None


def test_parse_task_extra_container_args_list() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "extra_container_args": ["--cap-add=NET_ADMIN", "--device=/dev/fuse"],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].extra_container_args == [
        "--cap-add=NET_ADMIN",
        "--device=/dev/fuse",
    ]


def test_parse_task_extra_container_args_string() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "extra_container_args": "--cap-add=SYS_ADMIN",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].extra_container_args == ["--cap-add=SYS_ADMIN"]


def test_parse_task_extra_container_args_empty_list() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "extra_container_args": [],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].extra_container_args == []


def test_parse_task_extra_container_args_hyphenated() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "extra-container-args": ["--cap-add=NET_ADMIN"],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].extra_container_args == ["--cap-add=NET_ADMIN"]


def test_parse_task_extra_container_args_underscore_wins() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "extra_container_args": ["--cap-add=SYS_ADMIN"],
                "extra-container-args": ["--cap-add=NET_ADMIN"],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].extra_container_args == ["--cap-add=SYS_ADMIN"]


# ---------------------------------------------------------------------------
# _resolve_key
# ---------------------------------------------------------------------------


def test_resolve_key_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "env-value")
    result = ManifestParser._resolve_key("env:TEST_KEY", "engine")
    assert result == "env-value"


def test_resolve_key_env_prefix_missing_raises() -> None:
    with pytest.raises(
        ValueError, match="Environment variable 'MISSING_XYZ' is not set"
    ):
        ManifestParser._resolve_key("env:MISSING_XYZ", "engine")


def test_resolve_key_plain_prefix() -> None:
    result = ManifestParser._resolve_key("plain:my-secret", "engine")
    assert result == "my-secret"


def test_resolve_key_stdin() -> None:
    with patch("getpass.getpass", return_value="stdin-value") as mock_getpass:
        result = ManifestParser._resolve_key("stdin", "my-engine")
        assert result == "stdin-value"
        mock_getpass.assert_called_once_with("Enter API key for 'my-engine': ")


def test_resolve_key_raw_key() -> None:
    result = ManifestParser._resolve_key("rawkey123", "engine")
    assert result == "rawkey123"


def test_resolve_key_empty_key() -> None:
    result = ManifestParser._resolve_key("", "engine")
    assert result == ""


def test_resolve_key_file_prefix(tmp_path: Path) -> None:
    file_path = tmp_path / "keyfile.txt"
    file_path.write_text("my-secret-key\n", encoding="utf-8")
    result = ManifestParser._resolve_key(f"file:{file_path}", "engine")
    assert result == "my-secret-key"


def test_resolve_key_file_prefix_no_newline(tmp_path: Path) -> None:
    file_path = tmp_path / "keyfile.txt"
    file_path.write_text("my-secret-key", encoding="utf-8")
    result = ManifestParser._resolve_key(f"file:{file_path}", "engine")
    assert result == "my-secret-key"


def test_resolve_key_file_prefix_multiple_newlines(tmp_path: Path) -> None:
    file_path = tmp_path / "keyfile.txt"
    file_path.write_text("my-secret-key\n\n\n", encoding="utf-8")
    result = ManifestParser._resolve_key(f"file:{file_path}", "engine")
    assert result == "my-secret-key\n\n"


def test_resolve_key_file_prefix_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    result = ManifestParser._resolve_key(f"file:{file_path}", "engine")
    assert result == ""


def test_resolve_key_file_prefix_only_newline(tmp_path: Path) -> None:
    file_path = tmp_path / "newline_only.txt"
    file_path.write_text("\n", encoding="utf-8")
    result = ManifestParser._resolve_key(f"file:{file_path}", "engine")
    assert result == ""


def test_resolve_key_file_prefix_relative_path_raises() -> None:
    with pytest.raises(ValueError, match="must be an absolute path"):
        ManifestParser._resolve_key("file:relative/path/key.txt", "engine")


def test_resolve_key_file_prefix_nonexistent_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "nonexistent.txt"
    with pytest.raises(FileNotFoundError):
        ManifestParser._resolve_key(f"file:{file_path}", "engine")


def test_resolve_key_file_prefix_non_utf8(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(UnicodeDecodeError):
        ManifestParser._resolve_key(f"file:{file_path}", "engine")


def test_resolve_key_file_prefix_in_engine(tmp_path: Path) -> None:
    file_path = tmp_path / "api_key.txt"
    file_path.write_text("file-based-key\n", encoding="utf-8")
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "file-key-engine",
                "model": "gpt-4",
                "api_key": f"file:{file_path}",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].api_key == "file-based-key"


# ---------------------------------------------------------------------------
# _expand_tool_name
# ---------------------------------------------------------------------------


def test_expand_tool_name_known_meta_tool() -> None:
    parser = ManifestParser(data={}, path=None)
    result = parser._expand_tool_name("standard_file_manipulation")
    assert result == ManifestParser._META_TOOL["standard_file_manipulation"]
    assert result == [
        "Edit",
        "FileMetadata",
        "Glob",
        "Grep",
        "InsertFile",
        "ListDir",
        "ReadFile",
        "ReadMulti",
        "WriteFile",
    ]


def test_expand_tool_name_hyphenated_warns() -> None:
    parser = ManifestParser(data={}, path=None)
    result = parser._expand_tool_name("standard-file-manipulation")
    assert result == [
        "Edit",
        "FileMetadata",
        "Glob",
        "Grep",
        "InsertFile",
        "ListDir",
        "ReadFile",
        "ReadMulti",
        "WriteFile",
    ]


def test_expand_tool_name_unknown_tool_returns_single() -> None:
    parser = ManifestParser(data={}, path=None)
    result = parser._expand_tool_name("BashTool")
    assert result == ["BashTool"]


def test_expand_tool_name_unknown_hyphenated_returns_single() -> None:
    parser = ManifestParser(data={}, path=None)
    result = parser._expand_tool_name("some-hyphenated-tool")
    assert result == ["some-hyphenated-tool"]


# ---------------------------------------------------------------------------
# _parse_confirmations
# ---------------------------------------------------------------------------


def test_parse_confirmations_default_type_exact() -> None:
    result = ManifestParser._parse_confirmations(
        [{"key": "target_file", "value": "/etc/passwd"}]
    )
    assert len(result) == 1
    assert result[0].type == "exact"
    assert result[0].key == "target_file"
    assert result[0].value == "/etc/passwd"
    assert isinstance(result[0].value, str)


def test_parse_confirmations_exact_type() -> None:
    result = ManifestParser._parse_confirmations(
        [{"type": "exact", "key": "file", "value": "test.txt"}]
    )
    assert len(result) == 1
    assert result[0].type == "exact"
    assert result[0].key == "file"
    assert result[0].value == "test.txt"
    assert isinstance(result[0].value, str)


def test_parse_confirmations_regexp_type() -> None:
    result = ManifestParser._parse_confirmations(
        [{"type": "regexp", "key": "file", "value": r"\.txt$"}]
    )
    assert len(result) == 1
    assert result[0].type == "regexp"
    assert result[0].key == "file"
    assert hasattr(result[0].value, "search")
    assert result[0].value.pattern == r"\.txt$"  # type: ignore[union-attr]


def test_parse_confirmations_any_type() -> None:
    result = ManifestParser._parse_confirmations([{"type": "any"}])
    assert len(result) == 1
    assert result[0].type == "any"
    assert result[0].key is None
    assert result[0].value is None


def test_parse_confirmations_multiple() -> None:
    result = ManifestParser._parse_confirmations(
        [
            {"type": "exact", "key": "file", "value": "a.txt"},
            {"type": "regexp", "key": "path", "value": r"/tmp/.*"},
            {"type": "any"},
        ]
    )
    assert len(result) == 3
    assert result[0].type == "exact"
    assert result[0].key == "file"
    assert result[0].value == "a.txt"
    assert result[1].type == "regexp"
    assert result[1].key == "path"
    assert isinstance(result[1].value, re.Pattern)
    assert result[1].value.pattern == r"/tmp/.*"
    assert result[2].type == "any"
    assert result[2].key is None
    assert result[2].value is None


def test_parse_confirmations_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="Invalid confirmation type 'invalid'"):
        ManifestParser._parse_confirmations(
            [{"type": "invalid", "key": "k", "value": "v"}]
        )


def test_parse_confirmations_exact_missing_key_raises() -> None:
    with pytest.raises(
        ValueError, match="Confirmation of type 'exact' must have a 'key'"
    ):
        ManifestParser._parse_confirmations([{"type": "exact", "value": "v"}])


def test_parse_confirmations_exact_missing_value_raises() -> None:
    with pytest.raises(
        ValueError, match="Confirmation of type 'exact' must have a 'value'"
    ):
        ManifestParser._parse_confirmations([{"type": "exact", "key": "k"}])


def test_parse_confirmations_regexp_missing_key_raises() -> None:
    with pytest.raises(
        ValueError, match="Confirmation of type 'regexp' must have a 'key'"
    ):
        ManifestParser._parse_confirmations([{"type": "regexp", "value": r"\d+"}])


def test_parse_confirmations_regexp_missing_value_raises() -> None:
    with pytest.raises(
        ValueError, match="Confirmation of type 'regexp' must have a 'value'"
    ):
        ManifestParser._parse_confirmations([{"type": "regexp", "key": "k"}])


def test_parse_confirmations_any_ignores_key_value() -> None:
    """'any' type should not require key or value."""
    result = ManifestParser._parse_confirmations(
        [{"type": "any", "key": "ignored", "value": "ignored"}]
    )
    assert len(result) == 1
    assert result[0].type == "any"
    assert result[0].key is None
    assert result[0].value is None


def test_parse_confirmations_empty_list() -> None:
    result = ManifestParser._parse_confirmations([])
    assert result == []


# ---------------------------------------------------------------------------
# ToolSpec with confirmations in _parse_tools
# ---------------------------------------------------------------------------


def test_parse_tools_with_confirmations_exact() -> None:
    parser = _make_parser_with_tools(
        [
            {
                "BashTool": [
                    "internet",
                    {
                        "confirmations": [
                            {"type": "exact", "key": "command", "value": "rm -rf /"}
                        ]
                    },
                ]
            }
        ]
    )
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "BashTool"
    assert len(tools[0].confirmations) == 1
    assert tools[0].confirmations[0].type == "exact"
    assert tools[0].confirmations[0].key == "command"
    assert tools[0].confirmations[0].value == "rm -rf /"


def test_parse_tools_with_confirmations_and_config() -> None:
    parser = _make_parser_with_tools(
        [
            {
                "ToolA": [
                    "internet",
                    "disk",
                    {"config": {"user_agent": "Agent/1.0"}},
                    {
                        "confirmations": [
                            {"type": "any"},
                            {"type": "regexp", "key": "url", "value": r"https?://.*"},
                        ]
                    },
                ]
            }
        ]
    )
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].name == "ToolA"
    assert tools[0].network == "internet"
    assert tools[0].disk_mode == "disk"
    assert tools[0].config == {"user_agent": "Agent/1.0"}
    assert len(tools[0].confirmations) == 2
    assert tools[0].confirmations[0].type == "any"
    assert tools[0].confirmations[0].key is None
    assert tools[0].confirmations[0].value is None
    assert tools[0].confirmations[1].type == "regexp"
    assert tools[0].confirmations[1].key == "url"
    assert isinstance(tools[0].confirmations[1].value, re.Pattern)
    assert tools[0].confirmations[1].value.pattern == r"https?://.*"


# ---------------------------------------------------------------------------
# _to_bool, _to_int, _to_float
# ---------------------------------------------------------------------------


def test_to_bool_with_string_true() -> None:
    assert ManifestParser._to_bool("true") is True
    assert ManifestParser._to_bool("True") is True
    assert ManifestParser._to_bool("TRUE") is True
    assert ManifestParser._to_bool("1") is True
    assert ManifestParser._to_bool("yes") is True


def test_to_bool_with_string_false() -> None:
    assert ManifestParser._to_bool("false") is False
    assert ManifestParser._to_bool("False") is False
    assert ManifestParser._to_bool("FALSE") is False
    assert ManifestParser._to_bool("0") is False
    assert ManifestParser._to_bool("no") is False
    assert ManifestParser._to_bool("") is False


def test_to_bool_with_bool() -> None:
    assert ManifestParser._to_bool(True) is True
    assert ManifestParser._to_bool(False) is False


def test_to_bool_with_int() -> None:
    assert ManifestParser._to_bool(1) is True
    assert ManifestParser._to_bool(0) is False


def test_to_bool_with_invalid_type_raises() -> None:
    """_to_bool should raise for non-str, non-bool, non-int types."""
    with pytest.raises(ValueError, match="Expected a boolean or string value"):
        ManifestParser._to_bool([])
    with pytest.raises(ValueError, match="Expected a boolean or string value"):
        ManifestParser._to_bool({})


def test_to_int_with_bool_raises() -> None:
    """_to_int should raise for boolean values."""
    with pytest.raises(ValueError, match="'test_field' must be an integer, got True"):
        ManifestParser._to_int(True, "test_field")
    with pytest.raises(ValueError, match="'test_field' must be an integer, got False"):
        ManifestParser._to_int(False, "test_field")


def test_to_float_with_bool_raises() -> None:
    """_to_float should raise for boolean values (B3)."""
    with pytest.raises(
        ValueError, match="'summarizer_context_ratio' must be a number, got True"
    ):
        ManifestParser._to_float(True, "summarizer_context_ratio")
    with pytest.raises(
        ValueError, match="'summarizer_context_ratio' must be a number, got False"
    ):
        ManifestParser._to_float(False, "summarizer_context_ratio")


def test_parse_meta_summarizer_context_ratio_bool_raises() -> None:
    """summarizer_context_ratio: true (boolean) should raise (B3)."""
    data = _make_data(meta={"version": "0", "summarizer_context_ratio": True})
    with pytest.raises(
        ValueError, match="'summarizer_context_ratio' must be a number, got True"
    ):
        ManifestParser(data=data, path=None)


def test_validate_color_invalid_4_digit_raises() -> None:
    """_validate_color should reject 4-digit hex colors."""
    with pytest.raises(ValueError, match="Invalid color_reasoning"):
        ManifestParser._validate_color("#abcd", "color_reasoning")


def test_validate_color_invalid_5_digit_raises() -> None:
    """_validate_color should reject 5-digit hex colors."""
    with pytest.raises(ValueError, match="Invalid color_reasoning"):
        ManifestParser._validate_color("#abcde", "color_reasoning")


def test_default_worker_image_constant() -> None:
    """_DEFAULT_WORKER_IMAGE constant should be 'tiz-worker:latest'."""
    from tiz.manifest_parser import _DEFAULT_WORKER_IMAGE

    assert _DEFAULT_WORKER_IMAGE == "tiz-worker:latest"
    assert SubagentSpec(name="test").worker_image == _DEFAULT_WORKER_IMAGE


def test_to_bool_verify_ssl_string_false() -> None:
    """verify_ssl: 'false' (quoted) should be False, not True."""
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "api_key": "plain:k",
                "verify_ssl": "false",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].verify_ssl is False


def test_to_bool_preserve_thinking_string_true() -> None:
    """preserve_thinking: 'true' (quoted) should be True."""
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "api_key": "plain:k",
                "preserve_thinking": "true",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].preserve_thinking is True


def test_to_bool_tmpfs_root_string_true() -> None:
    """tmpfs_root: 'true' (quoted) in task should be True."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t", "tmpfs_root": "true"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].tmpfs_root is True


def test_to_bool_allow_parallel_string_false() -> None:
    """allow_parallel_run: 'false' (quoted) should be False."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t", "allow_parallel_run": "false"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].allow_parallel_run is False


def test_to_bool_readonly_sandbox_string_true() -> None:
    """readonly_sandbox: 'true' (quoted) should be True."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t", "readonly_sandbox": "true"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].readonly_sandbox is True


def test_to_bool_audio_verify_ssl_string_false() -> None:
    """Audio engine verify_ssl: 'false' (quoted) should be False."""
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "e",
                "host": "http://localhost:8080",
                "verify_ssl": "false",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].verify_ssl is False


def test_to_bool_meta_color_string_false() -> None:
    """Meta color: 'false' (quoted) should be False."""
    data = _make_data(meta={"version": "0", "color": "false"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.color is False


def test_to_bool_meta_hide_reasoning_string_true() -> None:
    """Meta hide_reasoning: 'true' (quoted) should be True."""
    data = _make_data(meta={"version": "0", "hide_reasoning": "true"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.hide_reasoning is True


def test_to_bool_meta_ephemeral_string_true() -> None:
    """Meta ephemeral_sandbox: 'true' (quoted) should be True and set delete."""
    data = _make_data(meta={"version": "0", "ephemeral_sandbox": "true"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.ephemeral_sandbox is True
    assert parser.meta.delete_sandbox_on_exit is True


def test_to_bool_meta_save_full_logs_string_false() -> None:
    """Meta save_full_logs: 'false' (quoted) should be False."""
    data = _make_data(meta={"version": "0", "save_full_logs": "false"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_logs is False


def test_to_bool_meta_save_full_logs_string_true() -> None:
    """Meta save_full_logs: 'true' (quoted) should be True."""
    data = _make_data(meta={"version": "0", "save_full_logs": "true"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_logs is True


def test_to_bool_meta_save_full_toolcalls_string_false() -> None:
    """Meta save_full_toolcalls: 'false' (quoted) should be False."""
    data = _make_data(meta={"version": "0", "save_full_toolcalls": "false"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_toolcalls is False


def test_to_bool_meta_save_full_toolcalls_string_true() -> None:
    """Meta save_full_toolcalls: 'true' (quoted) should be True."""
    data = _make_data(meta={"version": "0", "save_full_toolcalls": "true"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_toolcalls is True


def test_to_bool_meta_save_full_usage_details_string_false() -> None:
    """Meta save_full_usage_details: 'false' (quoted) should be False."""
    data = _make_data(meta={"version": "0", "save_full_usage_details": "false"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_usage_details is False


def test_to_bool_meta_save_full_usage_details_string_true() -> None:
    """Meta save_full_usage_details: 'true' (quoted) should be True."""
    data = _make_data(meta={"version": "0", "save_full_usage_details": "true"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.save_full_usage_details is True


def test_to_bool_meta_ring_bell_string_false() -> None:
    """Meta ring_bell: 'false' (quoted) should be False."""
    data = _make_data(meta={"version": "0", "ring_bell": "false"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.ring_bell is False


def test_to_bool_meta_ring_bell_string_true() -> None:
    """Meta ring_bell: 'true' (quoted) should be True."""
    data = _make_data(meta={"version": "0", "ring_bell": "true"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.ring_bell is True


def test_to_bool_meta_delete_sandbox_on_exit_string_false() -> None:
    """Meta delete_sandbox_on_exit: 'false' (quoted) should be False."""
    data = _make_data(meta={"version": "0", "delete_sandbox_on_exit": "false"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.delete_sandbox_on_exit is False


def test_to_bool_meta_delete_sandbox_on_exit_string_true() -> None:
    """Meta delete_sandbox_on_exit: 'true' (quoted) should be True."""
    data = _make_data(meta={"version": "0", "delete_sandbox_on_exit": "true"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.delete_sandbox_on_exit is True


def test_to_int_parallelism_string() -> None:
    """Parallelism as string should convert."""
    data = _make_data(meta={"version": "0", "parallelism": "8"})
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.parallelism == 8


def test_to_int_invalid_raises() -> None:
    """Invalid parallelism value should raise."""
    data = _make_data(meta={"version": "0", "parallelism": "abc"})
    with pytest.raises(ValueError, match="'parallelism' must be an integer"):
        ManifestParser(data=data, path=None)


def test_to_int_verbosity_invalid_raises() -> None:
    """Invalid verbosity value should raise."""
    data = _make_data(meta={"version": "0", "verbosity": "abc"})
    with pytest.raises(ValueError, match="'verbosity' must be an integer"):
        ManifestParser(data=data, path=None)


def test_to_float_invalid_raises() -> None:
    """Invalid summarizer_context_ratio value should raise."""
    data = _make_data(meta={"version": "0", "summarizer_context_ratio": "abc"})
    with pytest.raises(ValueError, match="'summarizer_context_ratio' must be a number"):
        ManifestParser(data=data, path=None)


def test_load_yaml_with_list_raises(tmp_path: Path) -> None:
    """YAML file with a list should raise descriptive error."""
    manifest_file = tmp_path / "list.yaml"
    manifest_file.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a YAML mapping.*list"):
        ManifestParser(data={"meta": _MINIMAL_META}, path=manifest_file)


def test_load_yaml_with_scalar_raises(tmp_path: Path) -> None:
    """YAML file with a scalar should raise descriptive error."""
    manifest_file = tmp_path / "scalar.yaml"
    manifest_file.write_text("just a string", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a YAML mapping.*str"):
        ManifestParser(data={"meta": _MINIMAL_META}, path=manifest_file)


def test_parse_tools_meta_tool_expansion_with_confirmations() -> None:
    parser = _make_parser_with_tools(
        [
            {
                "standard_file_manipulation": [
                    "internet",
                    {
                        "confirmations": [
                            {"type": "exact", "key": "file", "value": "important.txt"}
                        ]
                    },
                ]
            }
        ]
    )
    tools = parser.tasks[0].tools
    assert len(tools) == 9
    for tool in tools:
        assert len(tool.confirmations) == 1
        assert tool.confirmations[0].type == "exact"
        assert tool.confirmations[0].key == "file"
        assert tool.confirmations[0].value == "important.txt"


def test_parse_tools_no_confirmations() -> None:
    parser = _make_parser_with_tools([{"BashTool": "internet"}])
    tools = parser.tasks[0].tools
    assert len(tools) == 1
    assert tools[0].confirmations == []


# ---------------------------------------------------------------------------
# ConfirmationSpec dataclass
# ---------------------------------------------------------------------------


def test_confirmation_spec_defaults() -> None:
    spec = ConfirmationSpec()
    assert spec.type == "exact"
    assert spec.key is None
    assert spec.value is None


def test_confirmation_spec_full() -> None:
    spec = ConfirmationSpec(type="regexp", key="k", value=re.compile(r"\d+"))
    assert spec.type == "regexp"
    assert spec.key == "k"
    assert isinstance(spec.value, re.Pattern)
    assert spec.value.pattern == r"\d+"


# ---------------------------------------------------------------------------
# _get_key
# ---------------------------------------------------------------------------


def test_get_key_underscore_preferred() -> None:
    parser = ManifestParser(data={}, path=None)
    raw = {"worker_image": "img1", "worker-image": "img2"}
    result = parser._get_key(raw, "worker_image")
    assert result == "img1"


def test_get_key_hyphenated_fallback() -> None:
    parser = ManifestParser(data={}, path=None)
    raw = {"worker-image": "img1"}
    result = parser._get_key(raw, "worker_image")
    assert result == "img1"


def test_get_key_missing_default() -> None:
    parser = ManifestParser(data={}, path=None)
    raw: dict[str, Any] = {}
    result = parser._get_key(raw, "worker_image", "default-val")
    assert result == "default-val"


def test_get_key_missing_no_default() -> None:
    parser = ManifestParser(data={}, path=None)
    raw: dict[str, Any] = {}
    result = parser._get_key(raw, "worker_image")
    assert result is None


def test_get_key_underscore_empty_does_not_fallback_to_hyphenated() -> None:
    parser = ManifestParser(data={}, path=None)
    raw = {"worker_image": "", "worker-image": "img2"}
    result = parser._get_key(raw, "worker_image")
    assert result == ""


def test_get_key_hyphenated_only_warns() -> None:
    parser = ManifestParser(data={}, path=None)
    raw = {"worker-image": "img1"}
    result = parser._get_key(raw, "worker_image")
    assert result == "img1"


# ---------------------------------------------------------------------------
# Hyphenated key backward compatibility
# ---------------------------------------------------------------------------


def test_hyphenated_worker_image() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "worker-image": "my-image:v1",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].worker_image == "my-image:v1"


def test_hyphenated_readonly_sandbox() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "readonly-sandbox": True,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].readonly_sandbox is True


def test_hyphenated_meta_key_warns() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "tools-default-user-agent": "Agent/1.0",
        }
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.tools_default_user_agent == "Agent/1.0"


def test_underscore_meta_key_wins() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "tools_default_user_agent": "UnderscoreAgent/1.0",
            "tools-default-user-agent": "HyphenAgent/1.0",
        }
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.meta.tools_default_user_agent == "UnderscoreAgent/1.0"


def test_hyphenated_inference_engine_key() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "message-timeout": 60,
                "preserve-thinking": True,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    eng = parser.inference_engines[0]
    assert eng.message_timeout == 60
    assert eng.preserve_thinking is True


def test_underscore_inference_engine_key_wins() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "e1",
                "message_timeout": 120,
                "message-timeout": 60,
                "preserve_thinking": True,
                "preserve-thinking": False,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    eng = parser.inference_engines[0]
    assert eng.message_timeout == 120
    assert eng.preserve_thinking is True


def test_hyphenated_scoring_action_keys() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "scoring": {
                    "scoring-prompt": "score it",
                    "scoring-rounds": 3,
                    "prompts": [["a"]],
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, ScoringAction)
    assert action.scoring_prompt == "score it"
    assert action.scoring_rounds == 3


def test_hyphenated_parallel_message_groups() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "prompts": ["msg"],
                "parallel-message-groups": True,
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, PromptsAction)
    assert action.parallel_message_groups is True


def test_hyphenated_iterator_parallel() -> None:
    parser = _make_parser_with_actions(
        [
            {
                "iterator": {
                    "input": "gen",
                    "prompts": [["step"]],
                    "parallel-message-groups": True,
                },
            }
        ]
    )
    action = parser.tasks[0].actions[0]
    assert isinstance(action, IteratorAction)
    assert action.parallel_message_groups is True


def test_hyphenated_force_copy_files() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "force-copy-files": "/path/to/file",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].force_copy_files == ["/path/to/file"]


def test_hyphenated_allow_parallel_run() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "allow-parallel-run": False,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].allow_parallel_run is False


def test_hyphenated_worker_image_containerfile() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "worker-image-containerfile": "/path/to/Dockerfile",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].worker_image_containerfile == "/path/to/Dockerfile"


def test_underscore_worker_image_containerfile() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "worker_image_containerfile": "/path/to/Dockerfile",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].worker_image_containerfile == "/path/to/Dockerfile"


def test_hyphenated_meta_keys_all() -> None:
    data = _make_data(
        meta={
            "version": "0",
            "container-engine": "docker",
            "color-reasoning": "#ff0000",
            "color-input": "#00ff00",
            "hide-reasoning": True,
            "use-host-timezone": True,
            "save-full-logs": True,
            "save-full-toolcalls": False,
            "save-full-usage-details": False,
            "summarizer-context-ratio": 0.5,
            "ring-bell": True,
            "delete-sandbox-on-exit": True,
        }
    )
    with patch("shutil.which", return_value="/usr/bin/docker"):
        parser = ManifestParser(data=data, path=None)
    assert parser.meta.container_engine == "docker"
    assert parser.meta.color_reasoning == "#ff0000"
    assert parser.meta.color_input == "#00ff00"
    assert parser.meta.hide_reasoning is True
    assert parser.meta.use_host_timezone is True
    assert parser.meta.save_full_logs is True
    assert parser.meta.save_full_toolcalls is False
    assert parser.meta.save_full_usage_details is False
    assert parser.meta.summarizer_context_ratio == 0.5
    assert parser.meta.ring_bell is True
    assert parser.meta.delete_sandbox_on_exit is True


def test_hyphenated_meta_tool_expansion() -> None:
    """Hyphenated meta-tool name should still work (backward compat)."""
    parser = _make_parser_with_tools([{"standard-file-manipulation": "internet"}])
    tools = parser.tasks[0].tools
    assert len(tools) == 9


# ---------------------------------------------------------------------------
# SubagentSpec dataclass
# ---------------------------------------------------------------------------


def test_subagent_spec_defaults() -> None:
    spec = SubagentSpec(name="agent1")
    assert spec.name == "agent1"
    assert spec.worker_image == "tiz-worker:latest"
    assert spec.worker_image_containerfile is None
    assert spec.description is None
    assert spec.sys_prompt is None
    assert spec.sys_prompt_custom is None
    assert spec.inference_engine is None
    assert spec.tools == []


def test_subagent_spec_full() -> None:
    tools = [ToolSpec(name="Bash", network="internet", disk_mode="nodisk")]
    spec = SubagentSpec(
        name="agent1",
        worker_image="custom-agent:v2",
        worker_image_containerfile="/path/to/Dockerfile",
        description="A test agent",
        sys_prompt="system prompt",
        sys_prompt_custom="custom prompt",
        inference_engine="my-engine",
        tools=tools,
    )
    assert spec.name == "agent1"
    assert spec.worker_image == "custom-agent:v2"
    assert spec.worker_image_containerfile == "/path/to/Dockerfile"
    assert spec.description == "A test agent"
    assert spec.sys_prompt == "system prompt"
    assert spec.sys_prompt_custom == "custom prompt"
    assert spec.inference_engine == "my-engine"
    assert len(spec.tools) == 1
    assert spec.tools[0].name == "Bash"


# ---------------------------------------------------------------------------
# _parse_subagents
# ---------------------------------------------------------------------------


def test_parse_subagents_single_minimal() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {"name": "sa1"},
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    subagents = parser.tasks[0].subagents
    assert len(subagents) == 1
    assert subagents[0].name == "sa1"
    assert subagents[0].worker_image == "tiz-worker:latest"
    assert subagents[0].worker_image_containerfile is None
    assert subagents[0].description is None
    assert subagents[0].sys_prompt is None
    assert subagents[0].sys_prompt_custom is None
    assert subagents[0].inference_engine is None
    assert subagents[0].tools == []


def test_parse_subagents_single_full() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "worker_image": "custom-agent:v2",
                        "worker_image_containerfile": "/path/to/Dockerfile",
                        "description": "A test subagent",
                        "sys_prompt": "You are helpful",
                        "sys_prompt_custom": "Custom override",
                        "inference_engine": "gpt-4",
                        "tools": [{"Bash": "internet"}],
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    subagents = parser.tasks[0].subagents
    assert len(subagents) == 1
    assert subagents[0].name == "sa1"
    assert subagents[0].worker_image == "custom-agent:v2"
    assert subagents[0].worker_image_containerfile == "/path/to/Dockerfile"
    assert subagents[0].description == "A test subagent"
    assert subagents[0].sys_prompt == "You are helpful"
    assert subagents[0].sys_prompt_custom == "Custom override"
    assert subagents[0].inference_engine == "gpt-4"
    assert len(subagents[0].tools) == 1
    assert subagents[0].tools[0].name == "Bash"
    assert subagents[0].tools[0].network == "internet"
    assert subagents[0].tools[0].disk_mode == "nodisk"


def test_parse_subagents_multiple() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {"name": "sa1"},
                    {
                        "name": "sa2",
                        "description": "Second agent",
                        "tools": [{"WebFetch": "internet"}],
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    subagents = parser.tasks[0].subagents
    assert len(subagents) == 2
    assert subagents[0].name == "sa1"
    assert subagents[0].description is None
    assert subagents[1].name == "sa2"
    assert subagents[1].description == "Second agent"
    assert len(subagents[1].tools) == 1
    assert subagents[1].tools[0].name == "WebFetch"


def test_parse_subagents_empty_name_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {"name": ""},
                ],
            }
        ],
    )
    with pytest.raises(
        ValueError, match="Subagent 'name' is required and cannot be empty"
    ):
        ManifestParser(data=data, path=None)


def test_parse_subagents_missing_name_raises() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {"description": "no name"},
                ],
            }
        ],
    )
    with pytest.raises(
        ValueError, match="Subagent 'name' is required and cannot be empty"
    ):
        ManifestParser(data=data, path=None)


def test_parse_subagents_empty_tools() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {"name": "sa1", "tools": []},
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].tools == []


def test_parse_subagents_none_tools() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {"name": "sa1", "tools": None},
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].tools == []


def test_parse_subagents_none_subagents_field() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": None,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents == []


def test_parse_subagents_no_subagents() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents == []


def test_parse_subagents_empty_subagents() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents == []


def test_parse_subagents_subagent_tool_expansion() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "tools": [{"standard_file_manipulation": "internet"}],
                    }
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    subagents = parser.tasks[0].subagents
    assert len(subagents) == 1
    assert len(subagents[0].tools) == 9
    assert subagents[0].tools[0].name == "Edit"
    assert subagents[0].tools[0].network == "internet"


def test_parse_subagents_task_with_actions_and_subagents() -> None:
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "actions": [{"prompts": [["hello"]]}],
                "subagents": [
                    {
                        "name": "sa1",
                        "inference_engine": "local",
                        "tools": [{"Bash": "internet"}],
                    }
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    task = parser.tasks[0]
    assert len(task.actions) == 1
    assert isinstance(task.actions[0], PromptsAction)
    assert len(task.subagents) == 1
    assert task.subagents[0].name == "sa1"
    assert task.subagents[0].inference_engine == "local"
    assert task.subagents[0].tools[0].name == "Bash"


def test_parse_subagents_hyphenated_worker_image() -> None:
    """Hyphenated worker-image should work in subagent entries."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "worker-image": "sub-agent-img:v1",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].worker_image == "sub-agent-img:v1"


def test_parse_subagents_underscore_worker_image_wins() -> None:
    """Underscore worker_image should win over hyphenated in subagent."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "worker_image": "underscore-img:v1",
                        "worker-image": "hyphen-img:v1",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].worker_image == "underscore-img:v1"


def test_parse_subagents_hyphenated_worker_image_containerfile() -> None:
    """Hyphenated worker-image-containerfile should work in subagent entries."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "worker-image-containerfile": "/path/to/Containerfile",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert (
        parser.tasks[0].subagents[0].worker_image_containerfile
        == "/path/to/Containerfile"
    )


def test_parse_subagents_underscore_worker_image_containerfile_wins() -> None:
    """Underscore worker_image_containerfile should win in subagent."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "worker_image_containerfile": "/path/to/Dockerfile",
                        "worker-image-containerfile": "/path/to/Containerfile",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert (
        parser.tasks[0].subagents[0].worker_image_containerfile == "/path/to/Dockerfile"
    )


def test_parse_subagents_default_worker_image() -> None:
    """Subagent without worker_image should get default."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {"name": "sa1"},
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].worker_image == "tiz-worker:latest"
    assert parser.tasks[0].subagents[0].worker_image_containerfile is None


def test_parse_subagents_inherits_task_worker_image() -> None:
    """Subagent should inherit worker_image from parent task."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "worker_image": "custom-worker:v1",
                "subagents": [
                    {"name": "sa1"},
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].worker_image == "custom-worker:v1"
    assert parser.tasks[0].subagents[0].worker_image_containerfile is None
    assert parser.tasks[0].subagents[0].inference_engine is None
    assert parser.tasks[0].subagents[0].sys_prompt is None
    assert parser.tasks[0].subagents[0].sys_prompt_custom is None


def test_parse_subagents_inherits_task_inference_engine() -> None:
    """Subagent should inherit inference_engine from parent task."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "inference_engine": "gpt-4",
                "subagents": [
                    {"name": "sa1"},
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].inference_engine == "gpt-4"
    assert parser.tasks[0].subagents[0].worker_image == "tiz-worker:latest"


def test_parse_subagents_inherits_task_sys_prompt() -> None:
    """Subagent should inherit sys_prompt from parent task."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "sys_prompt": "You are helpful",
                "sys_prompt_custom": "Custom override",
                "subagents": [
                    {"name": "sa1"},
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].sys_prompt == "You are helpful"
    assert parser.tasks[0].subagents[0].sys_prompt_custom == "Custom override"


def test_parse_subagents_explicit_overrides_task_defaults() -> None:
    """Subagent explicit values should override inherited task defaults."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "worker_image": "worker:v1",
                "inference_engine": "gpt-4",
                "sys_prompt": "task prompt",
                "sys_prompt_custom": "task custom",
                "subagents": [
                    {
                        "name": "sa1",
                        "worker_image": "sub-worker:v2",
                        "inference_engine": "claude-3",
                        "sys_prompt": "sub prompt",
                        "sys_prompt_custom": "sub custom",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    sub = parser.tasks[0].subagents[0]
    assert sub.worker_image == "sub-worker:v2"
    assert sub.inference_engine == "claude-3"
    assert sub.sys_prompt == "sub prompt"
    assert sub.sys_prompt_custom == "sub custom"


def test_parse_subagents_inherits_task_worker_image_containerfile() -> None:
    """Subagent should inherit worker_image_containerfile from parent task."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "worker_image_containerfile": "/path/to/Dockerfile",
                "subagents": [
                    {"name": "sa1"},
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert (
        parser.tasks[0].subagents[0].worker_image_containerfile == "/path/to/Dockerfile"
    )


def test_parse_subagents_mixed_inheritance() -> None:
    """Subagent should inherit some values from task while overriding others."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "worker_image": "task-worker:v1",
                "worker_image_containerfile": "/path/to/Dockerfile",
                "inference_engine": "gpt-4",
                "sys_prompt": "task prompt",
                "subagents": [
                    {
                        "name": "sa1",
                        "worker_image": "override-worker:v2",
                        "inference_engine": "claude-3",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    sub = parser.tasks[0].subagents[0]
    assert sub.worker_image == "override-worker:v2"
    assert sub.worker_image_containerfile == "/path/to/Dockerfile"
    assert sub.inference_engine == "claude-3"
    assert sub.sys_prompt == "task prompt"
    assert sub.sys_prompt_custom is None


def test_parse_subagents_hyphenated_description() -> None:
    """Hyphenated description should work in subagent entries."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "description": "A test agent",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].description == "A test agent"


def test_parse_subagents_hyphenated_sys_prompt() -> None:
    """Hyphenated sys-prompt should work in subagent entries."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "sys_prompt": "task prompt",
                "subagents": [
                    {
                        "name": "sa1",
                        "sys-prompt": "hyphenated prompt",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].sys_prompt == "hyphenated prompt"


def test_parse_subagents_underscore_sys_prompt_wins() -> None:
    """Underscore sys_prompt should win over hyphenated in subagent."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "sys_prompt": "underscore prompt",
                        "sys-prompt": "hyphenated prompt",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].sys_prompt == "underscore prompt"


def test_parse_subagents_hyphenated_sys_prompt_custom() -> None:
    """Hyphenated sys-prompt-custom should work in subagent entries."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "sys-prompt-custom": "hyphenated custom",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].sys_prompt_custom == "hyphenated custom"


def test_parse_subagents_underscore_sys_prompt_custom_wins() -> None:
    """Underscore sys_prompt_custom should win over hyphenated in subagent."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "sys_prompt_custom": "underscore custom",
                        "sys-prompt-custom": "hyphenated custom",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].sys_prompt_custom == "underscore custom"


def test_parse_subagents_hyphenated_inference_engine() -> None:
    """Hyphenated inference-engine should work in subagent entries."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "inference-engine": "hyphenated-engine",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].inference_engine == "hyphenated-engine"


def test_parse_subagents_underscore_inference_engine_wins() -> None:
    """Underscore inference_engine should win over hyphenated in subagent."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "subagents": [
                    {
                        "name": "sa1",
                        "inference_engine": "underscore-engine",
                        "inference-engine": "hyphenated-engine",
                    },
                ],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].subagents[0].inference_engine == "underscore-engine"


# ---------------------------------------------------------------------------
# B1: Empty YAML file handling
# ---------------------------------------------------------------------------


def test_init_with_empty_yaml_file(tmp_path: Path) -> None:
    """Empty YAML file should not crash (B1)."""
    manifest_file = tmp_path / "empty.yaml"
    manifest_file.write_text("", encoding="utf-8")
    parser = ManifestParser(data={"meta": _MINIMAL_META}, path=manifest_file)
    assert parser.meta.version == "0"
    assert parser.meta.parallelism == 4
    assert parser.tasks == []


def test_init_with_empty_yaml_file_and_override(tmp_path: Path) -> None:
    """Empty YAML with data override should work (B1)."""
    manifest_file = tmp_path / "empty.yaml"
    manifest_file.write_text("", encoding="utf-8")
    parser = ManifestParser(
        data={"meta": _MINIMAL_META, "tasks": [{"name": "from-data"}]},
        path=manifest_file,
    )
    assert len(parser.tasks) == 1
    assert parser.tasks[0].name == "from-data"


def test_init_with_yaml_containing_only_null(tmp_path: Path) -> None:
    """YAML file with 'null' content should not crash (B1)."""
    manifest_file = tmp_path / "null.yaml"
    manifest_file.write_text("null", encoding="utf-8")
    parser = ManifestParser(data={"meta": _MINIMAL_META}, path=manifest_file)
    assert parser.meta.version == "0"


# ---------------------------------------------------------------------------
# B2: Hyphenated keys in _parse_task
# ---------------------------------------------------------------------------


def test_hyphenated_sys_prompt() -> None:
    """sys-prompt should be recognized via _get_key (B2)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t", "sys-prompt": "hyphenated sys prompt"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].sys_prompt == "hyphenated sys prompt"


def test_hyphenated_sys_prompt_custom() -> None:
    """sys-prompt-custom should be recognized via _get_key (B2)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t", "sys-prompt-custom": "hyphenated custom"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].sys_prompt_custom == "hyphenated custom"


def test_hyphenated_inference_engine() -> None:
    """inference-engine should be recognized via _get_key (B2)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t", "inference-engine": "hyphenated-engine"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].inference_engine == "hyphenated-engine"


def test_hyphenated_dedicated_audio_engine() -> None:
    """dedicated-audio-engine should be recognized via _get_key (B2)."""
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {"type": "whispercpp", "name": "audio-1", "host": "http://localhost:8080"}
        ],
        tasks=[{"name": "t", "dedicated-audio-engine": "audio-1"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].dedicated_audio_engine == "audio-1"


def test_hyphenated_tmpfs_root() -> None:
    """tmpfs-root should be recognized via _get_key (B2)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t", "tmpfs-root": True}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].tmpfs_root is True


def test_underscore_sys_prompt_wins_over_hyphenated() -> None:
    """Underscore sys_prompt should win over hyphenated (B2)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "sys_prompt": "underscore prompt",
                "sys-prompt": "hyphenated prompt",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].sys_prompt == "underscore prompt"


# ---------------------------------------------------------------------------
# P1: dedicated_audio_engine validation
# ---------------------------------------------------------------------------


def test_dedicated_audio_engine_valid_reference() -> None:
    """dedicated_audio_engine must reference an existing audio engine (P1)."""
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {"type": "whispercpp", "name": "audio-1", "host": "http://localhost:8080"}
        ],
        tasks=[{"name": "t", "dedicated_audio_engine": "audio-1"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].dedicated_audio_engine == "audio-1"


def test_dedicated_audio_engine_invalid_reference_no_validation() -> None:
    """dedicated_audio_engine with non-existent engine should not raise (no validation)."""
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {"type": "whispercpp", "name": "audio-1", "host": "http://localhost:8080"}
        ],
        tasks=[{"name": "t", "dedicated_audio_engine": "nonexistent-engine"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].dedicated_audio_engine == "nonexistent-engine"


def test_dedicated_audio_engine_none_when_no_engines() -> None:
    """No dedicated_audio_engine set should work even with no audio engines."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "t"}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].dedicated_audio_engine is None


# ---------------------------------------------------------------------------
# P2: Task name validation improvement
# ---------------------------------------------------------------------------


def test_parse_task_name_non_string_converted_to_string() -> None:
    """Task name as non-string should be converted to string (P2)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": 42}],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].name == "42"


def test_parse_task_name_whitespace_only_raises() -> None:
    """Task name with only whitespace should raise (P2)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[{"name": "   "}],
    )
    with pytest.raises(ValueError, match="Task name is required and cannot be empty"):
        ManifestParser(data=data, path=None)


# ---------------------------------------------------------------------------
# P3: _expand_tool_name with unknown hyphenated tool
# ---------------------------------------------------------------------------


def test_expand_tool_name_unknown_hyphenated_passes_through() -> None:
    """Unknown hyphenated tool name should pass through (P3)."""
    parser = ManifestParser(data={}, path=None)
    result = parser._expand_tool_name("Web-Fetch")
    assert result == ["Web-Fetch"]


# ---------------------------------------------------------------------------
# B2: Multiple network modes should raise error
# ---------------------------------------------------------------------------


def test_parse_tools_multiple_networks_raises() -> None:
    """Multiple network modes on a tool should raise ValueError (B2)."""
    with pytest.raises(
        ValueError,
        match="specifies multiple network modes.*Only one network mode is allowed",
    ):
        _make_parser_with_tools([{"ToolA": ["internet", "ns:mynet"]}])


# ---------------------------------------------------------------------------
# B3: force_copy_files stringification
# ---------------------------------------------------------------------------


def test_parse_task_force_copy_files_int() -> None:
    """force_copy_files with int should be stringified (B3)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "force_copy_files": 42,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].force_copy_files == ["42"]


def test_parse_task_force_copy_files_list_with_int() -> None:
    """force_copy_files list with int items should be stringified (B3)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "force_copy_files": [42, "/path"],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].force_copy_files == ["42", "/path"]


# ---------------------------------------------------------------------------
# B4: Action value type validation
# ---------------------------------------------------------------------------


def test_parse_actions_repeater_not_dict_raises() -> None:
    """repeater with non-dict value should raise ValueError (B4)."""
    with pytest.raises(ValueError, match="'repeater' must be a dict"):
        _make_parser_with_actions([{"repeater": "foo"}])


def test_parse_actions_scoring_not_dict_raises() -> None:
    """scoring with non-dict value should raise ValueError (B4)."""
    with pytest.raises(ValueError, match="'scoring' must be a dict"):
        _make_parser_with_actions([{"scoring": "foo"}])


def test_parse_actions_iterator_not_dict_raises() -> None:
    """iterator with non-dict value should raise ValueError (B4)."""
    with pytest.raises(ValueError, match="'iterator' must be a dict"):
        _make_parser_with_actions([{"iterator": "foo"}])


def test_parse_actions_prompts_not_list_raises() -> None:
    """prompts with non-list value should raise ValueError (B4)."""
    with pytest.raises(ValueError, match="'prompts' must be a list"):
        _make_parser_with_actions([{"prompts": "foo"}])


def test_parse_actions_cmd_not_str_int_or_float_raises() -> None:
    """cmd with non-string non-number value should raise ValueError (B4)."""
    with pytest.raises(ValueError, match="'cmd' must be a string"):
        _make_parser_with_actions([{"cmd": []}])
    with pytest.raises(ValueError, match="'cmd' must be a string"):
        _make_parser_with_actions([{"cmd": {}}])


# ---------------------------------------------------------------------------
# B5: _parse_confirmations validates list type
# ---------------------------------------------------------------------------


def test_parse_confirmations_dict_raises() -> None:
    """confirmations as dict should raise ValueError (B5)."""
    with pytest.raises(ValueError, match="'confirmations' must be a list"):
        ManifestParser._parse_confirmations({"type": "exact", "key": "k", "value": "v"})


# ---------------------------------------------------------------------------
# B6: _parse_tools validates config/confirmations types
# ---------------------------------------------------------------------------


def test_parse_tools_config_not_dict_raises() -> None:
    """config as non-dict in tool should raise ValueError (B6)."""
    with pytest.raises(ValueError, match="'config' for tool 'ToolA' must be a dict"):
        _make_parser_with_tools(
            [
                {
                    "ToolA": [
                        "internet",
                        {"config": "not-a-dict"},
                    ]
                }
            ]
        )


def test_parse_tools_confirmations_not_list_raises() -> None:
    """confirmations as non-list in tool should raise ValueError (B6)."""
    with pytest.raises(
        ValueError, match="'confirmations' for tool 'ToolA' must be a list"
    ):
        _make_parser_with_tools(
            [
                {
                    "ToolA": [
                        "internet",
                        {"confirmations": "not-a-list"},
                    ]
                }
            ]
        )


# ---------------------------------------------------------------------------
# B8: Inference/audio engine name stringification
# ---------------------------------------------------------------------------


def test_parse_inference_engine_name_as_int() -> None:
    """Inference engine name as int should be stringified (B8)."""
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": 42,
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.inference_engines[0].name == "42"


def test_parse_inference_engine_name_whitespace_only_raises() -> None:
    """Inference engine name with only whitespace should raise (B8)."""
    data = _make_data(
        meta=_MINIMAL_META,
        inference_engines=[
            {
                "type": "llamacpp",
                "name": "   ",
            }
        ],
    )
    with pytest.raises(ValueError, match="Inference engine 'name' is required"):
        ManifestParser(data=data, path=None)


def test_parse_audio_inference_engine_name_as_int() -> None:
    """Audio inference engine name as int should be stringified (B8)."""
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": 42,
                "host": "http://localhost:8080",
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.audio_inference_engines[0].name == "42"


def test_parse_audio_inference_engine_name_whitespace_only_raises() -> None:
    """Audio inference engine name with only whitespace should raise (B8)."""
    data = _make_data(
        meta=_MINIMAL_META,
        audio_inference_engines=[
            {
                "type": "whispercpp",
                "name": "   ",
                "host": "http://localhost:8080",
            }
        ],
    )
    with pytest.raises(ValueError, match="Audio inference engine 'name' is required"):
        ManifestParser(data=data, path=None)


# ---------------------------------------------------------------------------
# S6: _parse_prompt_groups validates list type
# ---------------------------------------------------------------------------


def test_parse_prompt_groups_non_list_raises() -> None:
    """_parse_prompt_groups with non-list should raise ValueError (S6)."""
    with pytest.raises(ValueError, match="'prompts' must be a list"):
        ManifestParser._parse_prompt_groups("hello")


# ---------------------------------------------------------------------------
# S11: Non-UTF8 YAML file raises ValueError with path
# ---------------------------------------------------------------------------


def test_load_yaml_non_utf8_raises(tmp_path: Path) -> None:
    """Non-UTF8 YAML file should raise ValueError with path info (S11)."""
    manifest_file = tmp_path / "bad_encoding.bin"
    manifest_file.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ValueError, match="is not valid UTF-8"):
        ManifestParser(data={"meta": _MINIMAL_META}, path=manifest_file)


# ---------------------------------------------------------------------------
# S12: extra_container_args list items stringified
# ---------------------------------------------------------------------------


def test_parse_task_extra_container_args_list_with_int() -> None:
    """extra_container_args list with non-string items should be stringified (S12)."""
    data = _make_data(
        meta=_MINIMAL_META,
        tasks=[
            {
                "name": "t",
                "extra_container_args": [123, "--flag"],
            }
        ],
    )
    parser = ManifestParser(data=data, path=None)
    assert parser.tasks[0].extra_container_args == ["123", "--flag"]
