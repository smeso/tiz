"""Parse tiz manifest files.

Manifests are YAML files with Jinja2 templating support. Multiple
manifests can be parsed and merged.  The parser creates data
specifications for sandboxes, containers, tool instances and Chat sessions,
and defines the tasks using threads for parallelism.
"""

from __future__ import annotations

import dataclasses
import functools
import getpass
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from tiz.log import get_logger

logger = get_logger(__name__)

_VALID_NETWORKS = {"none", "internet"}
_VALID_DISK_MODES: dict[str, int] = {"disk": 3, "ro-disk": 2, "ro_disk": 2, "nodisk": 1}

DEFAULT_COLOR_REASONING = "#758182"
DEFAULT_COLOR_INPUT = "#ff00ff"
_DEFAULT_WORKER_IMAGE = "tiz-worker:latest"


# ---------------------------------------------------------------------------
# Data classes for parsed manifest sections
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ConfirmationSpec:
    """A confirmation specification for a tool.

    When type is 'any', no key or value is needed.
    When type is 'exact', value is a string for exact matching.
    When type is 'regexp', value is a compiled regex pattern.
    """

    type: str = "exact"
    key: str | None = None
    value: str | re.Pattern[str] | None = None


@dataclasses.dataclass
class ToolSpec:
    """A tool specification from the manifest."""

    name: str
    network: str
    disk_mode: str
    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    confirmations: list[ConfirmationSpec] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PromptsAction:
    """A prompts action: one or more sequential chat sessions."""

    message_groups: list[list[str]]
    parallel_message_groups: bool = False


@dataclasses.dataclass
class IteratorAction:
    """An iterator action: generates input via chat, splits by line, iterates prompts."""

    input_prompt: str
    prompt_groups: list[list[str]]
    parallel_message_groups: bool = False


@dataclasses.dataclass
class CmdAction:
    """A command action (e.g. sync)."""

    command: str


@dataclasses.dataclass
class RepeaterAction:
    """A repeater action: runs prompt groups repeatedly with a jinja counter."""

    repeat: int
    prompt_groups: list[list[str]]


@dataclasses.dataclass
class ScoringAction:
    """A scoring action: evaluates prompts across branches and selects a winner."""

    scoring_prompt: str
    rounds: int
    scoring_rounds: int
    prompt_groups: list[list[str]]
    iterator_input: str | None = None
    engines: list[str] = dataclasses.field(default_factory=list)
    scoring_engines: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SubagentSpec:
    """A subagent specification within a task."""

    name: str
    worker_image: str = _DEFAULT_WORKER_IMAGE
    worker_image_containerfile: str | None = None
    description: str | None = None
    sys_prompt: str | None = None
    sys_prompt_custom: str | None = None
    inference_engine: str | None = None
    tools: list[ToolSpec] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TaskSpec:
    """A single task defined in the manifest."""

    name: str
    worker_image: str
    worker_image_containerfile: str | None
    tools: list[ToolSpec]
    readonly_sandbox: bool
    project: str | None
    sys_prompt: str | None
    sys_prompt_custom: str | None
    actions: list[
        PromptsAction | CmdAction | IteratorAction | RepeaterAction | ScoringAction
    ]
    allow_parallel_run: bool
    force_copy_files: list[str]
    inference_engine: str | None = None
    dedicated_audio_engine: str | None = None
    tmpfs_root: bool = False
    subagents: list[SubagentSpec] = dataclasses.field(default_factory=list)
    extra_container_args: list[str] | None = None


@dataclasses.dataclass
class InferenceEngineSpec:
    """An inference engine specification from the manifest."""

    engine_type: str
    host: str
    model: str
    name: str
    timeout: int
    api_key: str | None = None
    message_timeout: int | None = None
    verify_ssl: bool = True
    ca_cert: str | None = None
    sampling_params: dict[str, Any] | None = None
    preserve_thinking: bool = False


@dataclasses.dataclass
class AudioInferenceEngineSpec:
    """An audio inference engine specification from the manifest."""

    engine_type: str
    name: str
    host: str
    timeout: int = 5
    inference_timeout: int | None = None
    verify_ssl: bool = True
    ca_cert: str | None = None
    sampling_params: dict[str, Any] | None = None
    language: str | None = None
    prompt: str | None = None


@dataclasses.dataclass
class ManifestMeta:
    """Top-level metadata."""

    version: str
    parallelism: int | None = None
    committer_name: str | None = None
    committer_email: str | None = None
    container_engine: str | None = None
    color: bool | None = None
    color_reasoning: str | None = None
    color_input: str | None = None
    hide_reasoning: bool | None = None
    use_host_timezone: bool | None = None
    save_full_logs: bool | None = None
    save_full_toolcalls: bool | None = None
    save_full_usage_details: bool | None = None
    summarizer_context_ratio: float | None = None
    verbosity: int | None = None
    tools_default_user_agent: str | None = None
    ring_bell: bool | None = None
    delete_sandbox_on_exit: bool | None = None
    ephemeral_sandbox: bool | None = None


@dataclasses.dataclass
class Manifest:
    """A fully parsed manifest."""

    meta: ManifestMeta
    tasks: list[TaskSpec] = dataclasses.field(default_factory=list)
    inference_engines: list[InferenceEngineSpec] = dataclasses.field(
        default_factory=list
    )
    audio_inference_engines: list[AudioInferenceEngineSpec] = dataclasses.field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _merge(m1: Manifest, m2: Manifest) -> Manifest:
    if m1.meta.version != m2.meta.version:
        raise ValueError(
            f"Manifest version mismatch: {m1.meta.version} vs {m2.meta.version}"
        )

    def _pick(v1: Any, v2: Any) -> Any:
        return v2 if v2 is not None else v1

    return Manifest(
        meta=ManifestMeta(
            version=m1.meta.version,
            parallelism=_pick(m1.meta.parallelism, m2.meta.parallelism),
            committer_name=_pick(m1.meta.committer_name, m2.meta.committer_name),
            committer_email=_pick(m1.meta.committer_email, m2.meta.committer_email),
            container_engine=_pick(m1.meta.container_engine, m2.meta.container_engine),
            color=_pick(m1.meta.color, m2.meta.color),
            color_reasoning=_pick(m1.meta.color_reasoning, m2.meta.color_reasoning),
            color_input=_pick(m1.meta.color_input, m2.meta.color_input),
            hide_reasoning=_pick(m1.meta.hide_reasoning, m2.meta.hide_reasoning),
            use_host_timezone=_pick(
                m1.meta.use_host_timezone, m2.meta.use_host_timezone
            ),
            save_full_logs=_pick(m1.meta.save_full_logs, m2.meta.save_full_logs),
            save_full_toolcalls=_pick(
                m1.meta.save_full_toolcalls, m2.meta.save_full_toolcalls
            ),
            save_full_usage_details=_pick(
                m1.meta.save_full_usage_details, m2.meta.save_full_usage_details
            ),
            summarizer_context_ratio=_pick(
                m1.meta.summarizer_context_ratio, m2.meta.summarizer_context_ratio
            ),
            verbosity=_pick(m1.meta.verbosity, m2.meta.verbosity),
            tools_default_user_agent=_pick(
                m1.meta.tools_default_user_agent, m2.meta.tools_default_user_agent
            ),
            ring_bell=_pick(m1.meta.ring_bell, m2.meta.ring_bell),
            delete_sandbox_on_exit=_pick(
                m1.meta.delete_sandbox_on_exit, m2.meta.delete_sandbox_on_exit
            ),
            ephemeral_sandbox=_pick(
                m1.meta.ephemeral_sandbox, m2.meta.ephemeral_sandbox
            ),
        ),
        tasks=m1.tasks + m2.tasks,
        inference_engines=m1.inference_engines + m2.inference_engines,
        audio_inference_engines=m1.audio_inference_engines + m2.audio_inference_engines,
    )


def merge(manifests: list[Manifest]) -> Manifest:
    """Merge an arbitrary number of manifests."""
    if not manifests:
        raise ValueError("At least one manifest is required")
    return functools.reduce(_merge, manifests)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class ManifestParser:
    """Encapsulate all manifest parsing and rendering logic."""

    _META_TOOL: dict[str, list[str]] = {
        "standard_file_manipulation": [
            "Edit",
            "FileMetadata",
            "Glob",
            "Grep",
            "InsertFile",
            "ListDir",
            "ReadFile",
            "ReadMulti",
            "WriteFile",
        ],
    }

    @staticmethod
    def _get_key(raw: dict[str, Any], key: str, default: Any = None) -> Any:
        """Get a dict value by underscore key, warning if hyphenated version is used."""
        if key in raw:
            return raw[key]
        hyphenated = key.replace("_", "-")
        if hyphenated in raw:
            logger.warning(
                "Key '%s' uses hyphens; use '%s' with underscores instead",
                hyphenated,
                key,
            )
            return raw[hyphenated]
        return default

    @staticmethod
    def _to_bool(v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes")
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        raise ValueError(f"Expected a boolean or string value, got {v!r}")

    @staticmethod
    def _to_int(v: Any, field_name: str) -> int:
        if isinstance(v, bool):
            raise ValueError(f"'{field_name}' must be an integer, got {v!r}")
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"'{field_name}' must be an integer, got {v!r}") from None

    @staticmethod
    def _to_float(v: Any, field_name: str) -> float:
        if isinstance(v, bool):
            raise ValueError(f"'{field_name}' must be a number, got {v!r}")
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"'{field_name}' must be a number, got {v!r}") from None

    def __init__(
        self,
        data: dict[str, Any],
        path: Path | None,
    ) -> None:
        self.data = self._load_yaml(path) if path else data
        self.data.update(data)
        self.meta = self._parse_meta(self.data.get("meta", {}))
        self.inference_engines = self._parse_inference_engines(
            self.data.get("inference_engines", [])
        )
        self.audio_inference_engines = self._parse_audio_inference_engines(
            self.data.get("audio_inference_engines", [])
        )
        raw_tasks: list[dict[str, Any]] = self.data.get("tasks", []) or []
        self.tasks = [self._parse_task(t) for t in raw_tasks]

    def get_manifest(self) -> Manifest:
        return Manifest(
            meta=self.meta,
            tasks=self.tasks,
            inference_engines=self.inference_engines,
            audio_inference_engines=self.audio_inference_engines,
        )

    def _expand_tool_name(self, name: str) -> list[str]:
        # Check both underscore and hyphenated versions, warn on hyphenated
        if name in self._META_TOOL:
            return self._META_TOOL[name]
        underscore_name = name.replace("-", "_")
        if underscore_name in self._META_TOOL:
            logger.warning(
                "Tool '%s' uses hyphens; use '%s' with underscores instead",
                name,
                underscore_name,
            )
            return self._META_TOOL[underscore_name]
        if "-" in name:
            logger.warning(
                "Tool '%s' contains hyphens but does not match any meta tool; "
                "tool will be passed through as-is",
                name,
            )
        return [name]

    def _parse_meta(self, raw_meta: dict[str, Any]) -> ManifestMeta:
        version = str(self._get_key(raw_meta, "version", "0"))
        if version != "0":
            raise ValueError("Manifest 'version' must be '0'")
        parallelism = self._get_key(raw_meta, "parallelism")
        parallelism = (
            self._to_int(parallelism, "parallelism")
            if parallelism is not None
            else None
        )

        committer_name = self._get_key(raw_meta, "committer_name")
        committer_email = self._get_key(raw_meta, "committer_email")
        committer_name = str(committer_name) if committer_name is not None else None
        committer_email = str(committer_email) if committer_email is not None else None
        container_engine = self._get_key(raw_meta, "container_engine")
        container_engine = (
            str(container_engine) if container_engine is not None else None
        )
        if container_engine is not None and shutil.which(container_engine) is None:
            raise ValueError(f"Container engine '{container_engine}' not found in PATH")
        color = self._get_key(raw_meta, "color")
        color = self._to_bool(color) if color is not None else None
        raw_color_reasoning = self._get_key(raw_meta, "color_reasoning")
        color_reasoning: str | None = (
            self._validate_color(str(raw_color_reasoning), "color_reasoning")
            if raw_color_reasoning is not None
            else None
        )
        raw_color_input = self._get_key(raw_meta, "color_input")
        color_input: str | None = (
            self._validate_color(str(raw_color_input), "color_input")
            if raw_color_input is not None
            else None
        )
        raw_hide_reasoning = self._get_key(raw_meta, "hide_reasoning")
        hide_reasoning: bool | None = (
            self._to_bool(raw_hide_reasoning)
            if raw_hide_reasoning is not None
            else None
        )
        raw_use_host_timezone = self._get_key(raw_meta, "use_host_timezone")
        use_host_timezone: bool | None = (
            self._to_bool(raw_use_host_timezone)
            if raw_use_host_timezone is not None
            else None
        )
        raw_save_full_logs = self._get_key(raw_meta, "save_full_logs")
        save_full_logs: bool | None = (
            self._to_bool(raw_save_full_logs)
            if raw_save_full_logs is not None
            else None
        )
        raw_save_full_toolcalls = self._get_key(raw_meta, "save_full_toolcalls")
        save_full_toolcalls: bool | None = (
            self._to_bool(raw_save_full_toolcalls)
            if raw_save_full_toolcalls is not None
            else None
        )
        raw_save_full_usage_details = self._get_key(raw_meta, "save_full_usage_details")
        save_full_usage_details: bool | None = (
            self._to_bool(raw_save_full_usage_details)
            if raw_save_full_usage_details is not None
            else None
        )
        raw_summarizer_context_ratio = self._get_key(
            raw_meta, "summarizer_context_ratio"
        )
        summarizer_context_ratio: float | None = (
            self._to_float(raw_summarizer_context_ratio, "summarizer_context_ratio")
            if raw_summarizer_context_ratio is not None
            else None
        )
        raw_verbosity = self._get_key(raw_meta, "verbosity")
        verbosity: int | None = (
            self._to_int(raw_verbosity, "verbosity")
            if raw_verbosity is not None
            else None
        )
        if verbosity is not None and verbosity not in (0, 1, 2):
            raise ValueError(
                f"Manifest 'verbosity' must be 0, 1, or 2, got {verbosity}"
            )
        raw_user_agent = self._get_key(raw_meta, "tools_default_user_agent")
        tools_default_user_agent: str | None = (
            str(raw_user_agent) if raw_user_agent is not None else None
        )
        raw_ring_bell = self._get_key(raw_meta, "ring_bell")
        ring_bell: bool | None = (
            self._to_bool(raw_ring_bell) if raw_ring_bell is not None else None
        )
        raw_delete_sandbox_on_exit = self._get_key(raw_meta, "delete_sandbox_on_exit")
        delete_sandbox_on_exit: bool | None = (
            self._to_bool(raw_delete_sandbox_on_exit)
            if raw_delete_sandbox_on_exit is not None
            else None
        )
        raw_ephemeral_sandbox = self._get_key(raw_meta, "ephemeral_sandbox")
        ephemeral_sandbox: bool | None = (
            self._to_bool(raw_ephemeral_sandbox)
            if raw_ephemeral_sandbox is not None
            else None
        )
        if ephemeral_sandbox is True and delete_sandbox_on_exit is False:
            raise ValueError(
                "If 'ephemeral_sandbox' is True then 'delete_sandbox_on_exit' must be True"
            )
        if ephemeral_sandbox is True and delete_sandbox_on_exit is None:
            delete_sandbox_on_exit = True
        return ManifestMeta(
            version=version,
            parallelism=parallelism,
            committer_name=committer_name,
            committer_email=committer_email,
            container_engine=container_engine,
            color=color,
            color_reasoning=color_reasoning,
            color_input=color_input,
            hide_reasoning=hide_reasoning,
            use_host_timezone=use_host_timezone,
            save_full_logs=save_full_logs,
            save_full_toolcalls=save_full_toolcalls,
            save_full_usage_details=save_full_usage_details,
            summarizer_context_ratio=summarizer_context_ratio,
            verbosity=verbosity,
            tools_default_user_agent=tools_default_user_agent,
            ring_bell=ring_bell,
            delete_sandbox_on_exit=delete_sandbox_on_exit,
            ephemeral_sandbox=ephemeral_sandbox,
        )

    def _parse_inference_engines(
        self, raw_engines: list[dict[str, Any]]
    ) -> list[InferenceEngineSpec]:
        engines: list[InferenceEngineSpec] = []
        for entry in raw_engines:
            engine_type = str(entry.get("type", "")).strip().lower()
            if not engine_type:
                raise ValueError("Inference engine 'type' is required")
            name = entry.get("name", "")
            if not name:
                raise ValueError("Inference engine 'name' is required")
            name = str(name).strip()
            if not name:
                raise ValueError("Inference engine 'name' is required")
            model = str(entry.get("model", ""))
            api_key = self._get_key(entry, "api_key")
            if api_key is not None:
                api_key = self._resolve_key(str(api_key), name)
            host = str(entry.get("host", ""))
            timeout = self._to_int(entry.get("timeout", 5), "timeout")
            raw_message_timeout = self._get_key(entry, "message_timeout")
            message_timeout: int | None = (
                self._to_int(raw_message_timeout, "message_timeout")
                if raw_message_timeout is not None
                else None
            )
            verify_ssl = self._to_bool(self._get_key(entry, "verify_ssl", True))
            ca_cert_val = self._get_key(entry, "ca_cert")
            ca_cert: str | None = str(ca_cert_val) if ca_cert_val is not None else None
            sampling_params = self._get_key(entry, "sampling_params")
            if sampling_params is not None and not isinstance(sampling_params, dict):
                raise ValueError("'sampling_params' must be a dict")
            preserve_thinking = self._to_bool(
                self._get_key(entry, "preserve_thinking", False)
            )
            engines.append(
                InferenceEngineSpec(
                    engine_type=engine_type,
                    host=host,
                    model=model,
                    api_key=api_key,
                    name=name,
                    timeout=timeout,
                    message_timeout=message_timeout,
                    verify_ssl=verify_ssl,
                    ca_cert=ca_cert,
                    sampling_params=sampling_params,
                    preserve_thinking=preserve_thinking,
                )
            )
        return engines

    def _parse_audio_inference_engines(
        self, raw_engines: list[dict[str, Any]]
    ) -> list[AudioInferenceEngineSpec]:
        engines: list[AudioInferenceEngineSpec] = []
        for entry in raw_engines:
            raw_engine_type = entry.get("type", "")
            if not raw_engine_type:
                raise ValueError("Audio inference engine 'type' is required")
            engine_type = str(raw_engine_type).strip().lower()
            if engine_type not in ("whispercpp", "whisper.cpp"):
                raise ValueError(
                    f"Audio inference engine 'type' must be 'whispercpp' or "
                    f"'whisper.cpp', got '{raw_engine_type}'"
                )
            name = entry.get("name", "")
            if not name:
                raise ValueError("Audio inference engine 'name' is required")
            name = str(name).strip()
            if not name:
                raise ValueError("Audio inference engine 'name' is required")
            host = str(entry.get("host", ""))
            timeout = self._to_int(entry.get("timeout", 5), "timeout")
            raw_inference_timeout = self._get_key(entry, "inference_timeout")
            inference_timeout: int | None = (
                self._to_int(raw_inference_timeout, "inference_timeout")
                if raw_inference_timeout is not None
                else None
            )
            verify_ssl = self._to_bool(self._get_key(entry, "verify_ssl", True))
            ca_cert_val = self._get_key(entry, "ca_cert")
            ca_cert: str | None = str(ca_cert_val) if ca_cert_val is not None else None
            sampling_params = self._get_key(entry, "sampling_params")
            if sampling_params is not None and not isinstance(sampling_params, dict):
                raise ValueError("'sampling_params' must be a dict")
            language = self._get_key(entry, "language")
            language = str(language) if language is not None else None
            prompt = entry.get("prompt")
            prompt = str(prompt) if prompt is not None else None
            engines.append(
                AudioInferenceEngineSpec(
                    engine_type=engine_type,
                    name=name,
                    host=host,
                    timeout=timeout,
                    inference_timeout=inference_timeout,
                    verify_ssl=verify_ssl,
                    ca_cert=ca_cert,
                    sampling_params=sampling_params,
                    language=language,
                    prompt=prompt,
                )
            )
        return engines

    def _parse_tools(self, raw_tools: list[dict[str, Any]]) -> list[ToolSpec]:
        seen: set[str] = set()
        tools: list[ToolSpec] = []
        for entry in raw_tools:
            for tool_name, network_list in entry.items():
                if not isinstance(network_list, list):
                    network_list = [network_list]

                tool_config: dict[str, Any] = {}
                confirmations: list[ConfirmationSpec] = []
                remaining: list[Any] = []
                for item in network_list:
                    if isinstance(item, dict) and "config" in item:
                        if not isinstance(item["config"], dict):
                            raise ValueError(
                                f"'config' for tool '{tool_name}' must be a dict, "
                                f"got {type(item['config']).__name__}"
                            )
                        tool_config.update(item["config"])
                    elif isinstance(item, dict) and "confirmations" in item:
                        if not isinstance(item["confirmations"], list):
                            raise ValueError(
                                f"'confirmations' for tool '{tool_name}' must be a list, "
                                f"got {type(item['confirmations']).__name__}"
                            )
                        confirmations.extend(
                            self._parse_confirmations(item["confirmations"])
                        )
                    else:
                        remaining.append(item)
                network_list = remaining

                disk_modes = [n for n in network_list if n in _VALID_DISK_MODES]
                nets = [
                    n
                    for n in network_list
                    if n in _VALID_NETWORKS or n.startswith("ns:")
                ]
                unknown = [
                    n
                    for n in network_list
                    if n not in _VALID_DISK_MODES
                    and n not in _VALID_NETWORKS
                    and not n.startswith("ns:")
                ]
                if unknown:
                    raise ValueError(
                        f"Unknown disk or network modes for tool '{tool_name}': {unknown}"
                    )

                if not disk_modes:
                    disk_mode = "nodisk"
                else:
                    disk_mode = max(disk_modes, key=lambda d: _VALID_DISK_MODES[d])
                if not nets:
                    nets = ["none"]

                if len(nets) > 1:
                    raise ValueError(
                        f"Tool '{tool_name}' specifies multiple network modes: {nets}. "
                        f"Only one network mode is allowed."
                    )
                nets_sorted = sorted(nets) if nets else ["none"]
                expanded = self._expand_tool_name(tool_name)
                for ename in expanded:
                    canonical = f"{ename}:{','.join(nets_sorted)}:{disk_mode}"
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    if (
                        ename in ("WebFetch", "WebSearch")
                        and self.meta.tools_default_user_agent is not None
                        and "user_agent" not in tool_config
                    ):
                        tool_config["user_agent"] = self.meta.tools_default_user_agent
                    tools.append(
                        ToolSpec(
                            name=ename,
                            network=nets_sorted[0],
                            disk_mode=disk_mode,
                            config=tool_config,
                            confirmations=confirmations,
                        )
                    )
        return tools

    @staticmethod
    def _parse_confirmations(
        raw_confirmations: Any,
    ) -> list[ConfirmationSpec]:
        if not isinstance(raw_confirmations, list):
            raise ValueError(
                f"'confirmations' must be a list, got {type(raw_confirmations).__name__}"
            )
        result: list[ConfirmationSpec] = []
        for entry in raw_confirmations:
            conf_type = entry.get("type", "exact")
            if conf_type not in ("exact", "regexp", "any"):
                raise ValueError(
                    f"Invalid confirmation type '{conf_type}': "
                    f"must be 'exact', 'regexp', or 'any'"
                )
            key: str | None = None
            value: str | re.Pattern[str] | None = None
            if conf_type != "any":
                raw_key = entry.get("key")
                if raw_key is None:
                    raise ValueError(
                        f"Confirmation of type '{conf_type}' must have a 'key'"
                    )
                key = str(raw_key)
                raw_value = entry.get("value")
                if raw_value is None:
                    raise ValueError(
                        f"Confirmation of type '{conf_type}' must have a 'value'"
                    )
                if conf_type == "regexp":
                    value = re.compile(str(raw_value))
                else:
                    value = str(raw_value)
            result.append(ConfirmationSpec(type=conf_type, key=key, value=value))
        return result

    @staticmethod
    def _parse_prompt_groups(raw_groups: Any) -> list[list[str]]:
        if not isinstance(raw_groups, list):
            raise ValueError(
                f"'prompts' must be a list, got {type(raw_groups).__name__}"
            )
        result: list[list[str]] = []
        for group in raw_groups:
            if isinstance(group, list):
                result.append([str(m) for m in group])
            else:
                result.append([str(group)])
        return result

    def _parse_subagents(
        self,
        raw_subagents: list[dict[str, Any]],
        task_worker_image: str = _DEFAULT_WORKER_IMAGE,
        task_worker_image_containerfile: str | None = None,
        task_sys_prompt: str | None = None,
        task_sys_prompt_custom: str | None = None,
        task_inference_engine: str | None = None,
    ) -> list[SubagentSpec]:
        result: list[SubagentSpec] = []
        for entry in raw_subagents:
            name = entry.get("name")
            if not name:
                raise ValueError("Subagent 'name' is required and cannot be empty")
            name = str(name)
            worker_image: str = self._get_key(entry, "worker_image", task_worker_image)
            worker_image_containerfile: str | None = self._get_key(
                entry, "worker_image_containerfile", task_worker_image_containerfile
            )
            description: str | None = self._get_key(entry, "description")
            description = str(description) if description is not None else None
            sys_prompt: str | None = self._get_key(entry, "sys_prompt", task_sys_prompt)
            sys_prompt_custom: str | None = self._get_key(
                entry, "sys_prompt_custom", task_sys_prompt_custom
            )
            inference_engine: str | None = self._get_key(
                entry, "inference_engine", task_inference_engine
            )
            raw_tools = entry.get("tools", []) or []
            tools = self._parse_tools(raw_tools) if raw_tools else []
            result.append(
                SubagentSpec(
                    name=name,
                    worker_image=worker_image,
                    worker_image_containerfile=worker_image_containerfile,
                    description=description,
                    sys_prompt=sys_prompt,
                    sys_prompt_custom=sys_prompt_custom,
                    inference_engine=inference_engine,
                    tools=tools,
                )
            )
        return result

    def _parse_actions(
        self, raw_actions: list[dict[str, Any]]
    ) -> list[
        PromptsAction | CmdAction | IteratorAction | RepeaterAction | ScoringAction
    ]:
        actions: list[
            PromptsAction | CmdAction | IteratorAction | RepeaterAction | ScoringAction
        ] = []
        for entry in raw_actions:
            if "repeater" in entry:
                repeater_def = entry["repeater"]
                if not isinstance(repeater_def, dict):
                    raise ValueError(
                        f"'repeater' must be a dict, got {type(repeater_def).__name__}"
                    )
                repeat = self._to_int(repeater_def.get("repeat", 1), "repeat")
                raw_prompt_groups = repeater_def.get("prompts", [])
                prompt_groups = self._parse_prompt_groups(raw_prompt_groups)
                if prompt_groups:
                    actions.append(
                        RepeaterAction(
                            repeat=repeat,
                            prompt_groups=prompt_groups,
                        )
                    )
            elif "scoring" in entry:
                scoring_def = entry["scoring"]
                if not isinstance(scoring_def, dict):
                    raise ValueError(
                        f"'scoring' must be a dict, got {type(scoring_def).__name__}"
                    )
                scoring_prompt = str(self._get_key(scoring_def, "scoring_prompt", ""))
                rounds = self._to_int(scoring_def.get("rounds", 1), "rounds")
                scoring_rounds = self._to_int(
                    self._get_key(scoring_def, "scoring_rounds", 1), "scoring_rounds"
                )
                iterator_input = self._get_key(scoring_def, "iterator_input")
                if iterator_input is not None:
                    iterator_input = str(iterator_input)
                engines: list[str] = scoring_def.get("engines", []) or []
                if isinstance(engines, str):
                    engines = [engines]
                engines = [str(e) for e in engines]
                scoring_engines: list[str] = (
                    self._get_key(scoring_def, "scoring_engines", []) or []
                )
                if isinstance(scoring_engines, str):
                    scoring_engines = [scoring_engines]
                scoring_engines = [str(e) for e in scoring_engines]
                raw_prompt_groups = scoring_def.get("prompts", [])
                sc_prompt_groups = self._parse_prompt_groups(raw_prompt_groups)
                if sc_prompt_groups:
                    actions.append(
                        ScoringAction(
                            scoring_prompt=scoring_prompt,
                            rounds=rounds,
                            scoring_rounds=scoring_rounds,
                            prompt_groups=sc_prompt_groups,
                            iterator_input=iterator_input,
                            engines=engines,
                            scoring_engines=scoring_engines,
                        )
                    )
            elif "iterator" in entry:
                iterator_def = entry["iterator"]
                if not isinstance(iterator_def, dict):
                    raise ValueError(
                        f"'iterator' must be a dict, got {type(iterator_def).__name__}"
                    )
                input_prompt = str(iterator_def.get("input", ""))
                raw_prompt_groups = iterator_def.get("prompts", [])
                it_prompt_groups = self._parse_prompt_groups(raw_prompt_groups)
                parallel = self._to_bool(
                    self._get_key(iterator_def, "parallel_message_groups", False)
                )
                if it_prompt_groups:
                    actions.append(
                        IteratorAction(
                            input_prompt=input_prompt,
                            prompt_groups=it_prompt_groups,
                            parallel_message_groups=parallel,
                        )
                    )
            elif "prompts" in entry:
                raw_groups = entry["prompts"]
                if not isinstance(raw_groups, list):
                    raise ValueError(
                        f"'prompts' must be a list, got {type(raw_groups).__name__}"
                    )
                groups = self._parse_prompt_groups(raw_groups)
                if groups:
                    parallel = self._to_bool(
                        self._get_key(entry, "parallel_message_groups", False)
                    )
                    actions.append(
                        PromptsAction(
                            message_groups=groups,
                            parallel_message_groups=parallel,
                        )
                    )
            elif "cmd" in entry:
                if not isinstance(entry["cmd"], (str, int, float)):
                    raise ValueError(
                        f"'cmd' must be a string, got {type(entry['cmd']).__name__}"
                    )
                actions.append(CmdAction(command=str(entry["cmd"]).lower()))
            else:
                raise ValueError(f"Unknown action type in manifest: {entry}")
        return actions

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8", errors="strict") as fd:
                result: Any = yaml.safe_load(fd)
        except UnicodeDecodeError as e:
            raise ValueError(f"File '{path}' is not valid UTF-8: {e}") from None
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise ValueError(
                f"Expected a YAML mapping (dict) at {path}, got {type(result).__name__}"
            )
        return result

    def _parse_task(self, raw_task: dict[str, Any]) -> TaskSpec:
        name_raw = raw_task.get("name")
        if name_raw is None:
            raise ValueError("Task name is required and cannot be empty")
        name = str(name_raw).strip()
        if name == "":
            raise ValueError("Task name is required and cannot be empty")
        worker_image: str = self._get_key(
            raw_task, "worker_image", _DEFAULT_WORKER_IMAGE
        )
        worker_image_containerfile: str | None = self._get_key(
            raw_task, "worker_image_containerfile"
        )
        tools = self._parse_tools(raw_task.get("tools", []))
        readonly = self._to_bool(self._get_key(raw_task, "readonly_sandbox", False))
        project: str | None = self._get_key(raw_task, "project")
        sys_prompt: str | None = self._get_key(raw_task, "sys_prompt")
        sys_prompt_custom: str | None = self._get_key(raw_task, "sys_prompt_custom")
        inference_engine: str | None = self._get_key(raw_task, "inference_engine")
        dedicated_audio_engine: str | None = self._get_key(
            raw_task, "dedicated_audio_engine"
        )

        tmpfs_root = self._to_bool(self._get_key(raw_task, "tmpfs_root", False))

        raw_actions: list[dict[str, Any]] = raw_task.get("actions", []) or []
        actions = self._parse_actions(raw_actions)

        allow_parallel = self._to_bool(
            self._get_key(raw_task, "allow_parallel_run", True)
        )
        force_copy_files_raw = self._get_key(raw_task, "force_copy_files", [])
        force_copy_files: list[str] = (
            [str(x) for x in force_copy_files_raw]
            if isinstance(force_copy_files_raw, list)
            else [str(force_copy_files_raw)]
        )

        raw_extra_container_args = self._get_key(raw_task, "extra_container_args")
        extra_container_args: list[str] | None = (
            [str(x) for x in raw_extra_container_args]
            if isinstance(raw_extra_container_args, list)
            else (
                [str(raw_extra_container_args)]
                if raw_extra_container_args is not None
                else None
            )
        )

        raw_subagents: list[dict[str, Any]] = raw_task.get("subagents", []) or []
        subagents = (
            self._parse_subagents(
                raw_subagents,
                task_worker_image=worker_image,
                task_worker_image_containerfile=worker_image_containerfile,
                task_sys_prompt=sys_prompt,
                task_sys_prompt_custom=sys_prompt_custom,
                task_inference_engine=inference_engine,
            )
            if raw_subagents
            else []
        )

        return TaskSpec(
            name=name,
            worker_image=worker_image,
            worker_image_containerfile=worker_image_containerfile,
            tools=tools,
            readonly_sandbox=readonly,
            project=project,
            sys_prompt=sys_prompt,
            sys_prompt_custom=sys_prompt_custom,
            actions=actions,
            allow_parallel_run=allow_parallel,
            force_copy_files=force_copy_files,
            inference_engine=inference_engine,
            dedicated_audio_engine=dedicated_audio_engine,
            tmpfs_root=tmpfs_root,
            subagents=subagents,
            extra_container_args=extra_container_args,
        )

    @staticmethod
    def _resolve_key(raw_key: str, name: str) -> str:
        if raw_key.startswith("env:"):
            env_var = raw_key[4:]
            value = os.environ.get(env_var)
            if value is None:
                raise ValueError(f"Environment variable '{env_var}' is not set")
            return value
        if raw_key.startswith("plain:"):
            return raw_key[6:]
        if raw_key.startswith("file:"):
            file_path = Path(raw_key[5:])
            if not file_path.is_absolute():
                raise ValueError(f"File path '{file_path}' must be an absolute path")
            content = file_path.read_text(encoding="utf-8")
            if content.endswith("\n"):
                content = content[:-1]
            return content
        if raw_key == "stdin":
            return getpass.getpass(f"Enter API key for '{name}': ")
        return raw_key

    @staticmethod
    def _validate_color(value: str, field_name: str) -> str:
        if not re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", value):
            raise ValueError(
                f"Invalid {field_name} '{value}': must be a # followed by 3 or 6 hex digits"
            )
        value = value.lower()
        if len(value) == 4:
            r, g, b = value[1], value[2], value[3]
            value = f"#{r}{r}{g}{g}{b}{b}"
        return value
