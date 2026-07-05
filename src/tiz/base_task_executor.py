"""Abstract base class for task executors with shared usage tracking and logging."""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import importlib
import importlib.resources
import importlib.util
import json
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, UndefinedError

from tiz.chat import Chat
from tiz.conversion_sandbox import ConversionSandbox
from tiz.inference_clients import (
    DwarfStar4,
    InferenceClient,
    LlamaCpp,
    OpenRouter,
)
from tiz.log import get_logger
from tiz.manifest_parser import (
    ConfirmationSpec,
    InferenceEngineSpec,
    Manifest,
    SubagentSpec,
    TaskSpec,
    ToolSpec,
)
from tiz.sandbox_container import (
    CONTAINER_MOUNT_OWN_SHARED,
    CONTAINER_MOUNT_PROJECT,
    CONTAINER_MOUNT_SHARED,
)
from tiz.sandbox_dirs import (
    TIZ_COMMIT_AUTHOR_EMAIL,
    TIZ_COMMIT_AUTHOR_NAME,
    SandboxDirs,
)
from tiz.sandbox_manager import CONTAINERFILES_DIR, SandboxManager
from tiz.tools.base import SocketTool, Tool
from tiz.tools.bash import Bash
from tiz.tools.subagents import SubAgents

logger = get_logger(__name__)


@dataclasses.dataclass
class TaskResources:
    """Container for the resources created by _create_task_resources."""

    tool_instances: list[Tool]
    tools_confirmations: dict[str, list[ConfirmationSpec]]
    sandbox: SandboxDirs | None
    manager: SandboxManager | None
    sandbox_name: str | None
    action_lock: contextlib.AbstractContextManager[None] | None
    conversion_sandbox: ConversionSandbox | None


_SANDBOX_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _make_sandbox_name(task_name: str) -> str:
    return _SANDBOX_NAME_RE.sub("_", task_name.lower())


_PROMPTS_DIR = "prompts"


class BaseTaskExecutor:
    def __init__(
        self,
        manifest: Manifest,
        base_path: Path = Path(),
        update_callback: Callable[[dict[str, Any], str | None], None] | None = None,
        context: dict[str, Any] | None = None,
        confirm_callback: Callable[
            [dict[str, Any], Callable[[dict[str, Any], bool], str | None], str | None],
            bool,
        ]
        | None = None,
    ) -> None:
        self.manifest = manifest
        self._base_path = base_path
        self._update_callback = update_callback
        self._confirm_callback = confirm_callback
        self._task_usage: dict[str, dict[str, dict[str, Any]]] = {}
        self._task_usage_lock = threading.Lock()
        self._default_client = self._create_default_client()
        self._jinja_env = Environment(autoescape=False, undefined=StrictUndefined)
        self._context = context or {}
        self._prompts_dirs, self._containerfiles_dirs = self._get_data_dirs()
        self._engine = self.manifest.meta.container_engine

    def _get_data_dirs(self) -> tuple[list[Path], list[Path]]:
        _data_dir = Path(str(importlib.resources.files("tiz") / "data"))
        prompts_dirs = [
            self._base_path / _PROMPTS_DIR,
            _data_dir / _PROMPTS_DIR,
        ]
        containerfiles_dirs = [
            self._base_path / CONTAINERFILES_DIR,
            _data_dir / CONTAINERFILES_DIR,
        ]
        return prompts_dirs, containerfiles_dirs

    @staticmethod
    def _discover_tools(
        paths: list[Path] | None = None,
    ) -> dict[str, type[Tool]]:
        """Import all tools from tiz.tools and index by fname.

        Also discovers tools from python files in the given paths.
        """
        discovered_tools: dict[str, type[Tool]] = {}

        import tiz.tools as tools_pkg

        for class_name in tools_pkg.__all__:
            cls = getattr(tools_pkg, class_name)
            discovered_tools[cls.fname()] = cls

        if paths:
            for path_ in paths:
                discovered_tools.update(
                    BaseTaskExecutor._discover_tools_from_dir(path_)
                )

        return discovered_tools

    @staticmethod
    def _discover_tools_from_dir(dirpath: Path) -> dict[str, type[Tool]]:
        """Discover Tool subclasses from Python files in *dirpath*."""
        discovered_tools: dict[str, type[Tool]] = {}

        if not dirpath.is_dir():
            return discovered_tools

        for py_file in sorted(dirpath.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            mod_name = f"user_tools.{py_file.stem}"
            if mod_name in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(mod_name, py_file)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                sys.modules[mod_name] = mod
            except Exception as exc:
                logger.warning("Failed to load tool module '%s': %s", py_file.name, exc)
                sys.modules.pop(mod_name, None)
                continue
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Tool)
                    and obj is not Tool
                    and obj is not SocketTool
                ):
                    discovered_tools[obj.fname()] = obj

        return discovered_tools

    def _create_sub_agent_chat_factory(
        self,
        sub_spec: SubagentSpec,
        sandbox_name: str,
        manager: SandboxManager,
        task: TaskSpec,
    ) -> Callable[[], Chat]:
        """Create a factory that returns a Chat instance for a sub-agent.

        Each invocation creates a fresh Chat with dedicated containers in the main
        sandbox, holding locks as necessary.
        """
        discovered_tools = BaseTaskExecutor._discover_tools([self._base_path / "tools"])

        def factory() -> Chat:
            tool_instances, tools_confirmations = (
                self._create_tool_instances_from_specs(
                    sub_spec.tools or [],
                    discovered_tools,
                    sandbox_name,
                    sub_spec.worker_image,
                    manager,
                    error_context=f"sub-agent '{sub_spec.name}' tools",
                    default_readonly=task.readonly_sandbox,
                    extra_run_args=task.extra_container_args,
                )
            )
            client = self._get_client_for_task_sub_agent(sub_spec)
            sys_prompt = self._resolve_sub_agent_prompt(sub_spec)

            _sub_name = sub_spec.name
            update_callback = (
                (lambda msg, _sn=_sub_name, _cb=self._update_callback: _cb(msg, _sn))
                if self._update_callback is not None
                else None
            )
            confirm_callback = (
                (
                    lambda call, fmt, _sn=_sub_name, _cb=self._confirm_callback: _cb(
                        call, fmt, _sn
                    )
                )
                if self._confirm_callback is not None
                else None
            )

            return Chat(
                client=client,
                sys_prompt=sys_prompt,
                tools=tool_instances or None,
                update_callback=update_callback,
                confirm_callback=confirm_callback,
                tools_confirmations=tools_confirmations or None,
                ctx_ratio=self.manifest.meta.summarizer_context_ratio or 0.9,
                subtask_name=_sub_name,
            )

        return factory

    def _get_client_for_task_sub_agent(self, sub_spec: SubagentSpec) -> InferenceClient:
        return self._get_client_for_task_engine(
            sub_spec.inference_engine, f"sub-agent '{sub_spec.name}'"
        )

    @staticmethod
    def _categorize_tools(tool_specs: list[ToolSpec]) -> dict[str, str]:
        """Categorize tool specs by disk_mode and network, returning context entries."""
        categories: dict[str, list[str]] = {
            "tools_with_project_access": [],
            "tools_without_project_access": [],
            "tools_with_internet_access": [],
            "tools_without_internet_access": [],
        }
        seen_names: set[str] = set()
        for ts in tool_specs:
            if ts.name not in seen_names:
                seen_names.add(ts.name)
                if ts.disk_mode in ("disk", "ro-disk"):
                    categories["tools_with_project_access"].append(ts.name)
                else:
                    categories["tools_without_project_access"].append(ts.name)
                if ts.network == "internet":
                    categories["tools_with_internet_access"].append(ts.name)
                else:
                    categories["tools_without_internet_access"].append(ts.name)
        return {k: ", ".join(v) for k, v in categories.items() if v}

    def _resolve_sub_agent_prompt(self, sub_spec: SubagentSpec) -> str:
        meta = self.manifest.meta

        context: dict[str, Any] = {}
        now = datetime.now().astimezone()
        context["date"] = now.strftime("%Y-%m-%d")
        context["datetime"] = now.isoformat()
        context["uuid"] = str(uuid.uuid4())
        context["general_shared_dir"] = CONTAINER_MOUNT_SHARED
        context["own_shared_dir"] = CONTAINER_MOUNT_OWN_SHARED
        context["committer_name"] = meta.committer_name
        context["committer_email"] = meta.committer_email

        context.update(self._categorize_tools(sub_spec.tools))

        return self._resolve_prompt_text(
            sub_spec.sys_prompt, sub_spec.sys_prompt_custom, extra_context=context
        )

    def _create_sub_agents_tool(
        self,
        task: TaskSpec,
        sandbox_name: str,
        manager: SandboxManager,
    ) -> Tool | None:
        """Build a SubAgents tool from the task's sub_agents config."""
        if not task.subagents:
            return None

        subagents: dict[str, dict[str, Any]] = {}
        for sub_spec in task.subagents:
            factory = self._create_sub_agent_chat_factory(
                sub_spec, sandbox_name, manager, task
            )
            task_sub_name = f"{task.name}/{sub_spec.name}"

            def make_callback(
                _task_sub_name: str = task_sub_name,
            ) -> Callable[[dict[str, Any], list[dict[str, Any]]], None]:
                def callback(usage: dict[str, Any], conv: list[dict[str, Any]]) -> None:
                    self._accumulate_usage(_task_sub_name, usage)
                    self._save_conv_log(conv, _task_sub_name)

                return callback

            subagents[sub_spec.name] = {
                "chat": factory,
                "description": sub_spec.description or "",
                "usage_callback": make_callback(),
            }

        return SubAgents(subagents=subagents)

    def _save_conv_log(self, conv: list[dict[str, Any]], task_name: str) -> None:
        if not self.manifest.meta.save_full_logs:
            return
        engine_name = self._get_engine_name(task_name)
        slug_task = _make_sandbox_name(task_name)
        slug_engine = _make_sandbox_name(engine_name)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self._base_path / "logs" / "conversations"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create conversation log dir %s: %s", log_dir, exc)
            return
        log_path = log_dir / f"{slug_task}-{slug_engine}-{now}.json"
        try:
            with log_path.open("w") as f:
                json.dump(conv, f)
        except OSError as exc:
            logger.warning("Failed to write conversation log %s: %s", log_path, exc)

    def _resolve_containerfile_content(
        self, worker_image: str, worker_image_containerfile: str | None
    ) -> str | None:
        if worker_image_containerfile:
            return worker_image_containerfile
        for cf_dir in self._containerfiles_dirs:
            cf_path = cf_dir / f"Containerfile.{worker_image}"
            if cf_path.is_file():
                return cf_path.read_text(encoding="utf-8")
        return None

    def _build_worker_image(
        self,
        manager: SandboxManager,
        worker_image: str,
        worker_image_containerfile: str | None,
    ) -> None:
        cf_content = self._resolve_containerfile_content(
            worker_image, worker_image_containerfile
        )
        if cf_content:
            manager.build_image(
                containerfile=cf_content,
                tag=worker_image,
                delete_existing=False,
            )

    def _create_tool_instances_from_specs(
        self,
        tool_specs: list[ToolSpec],
        discovered_tools: dict[str, type[Tool]],
        sandbox_name: str,
        worker_image: str,
        manager: SandboxManager,
        error_context: str = "tools",
        default_readonly: bool | None = None,
        extra_run_args: list[str] | None = None,
    ) -> tuple[list[Tool], dict[str, list[ConfirmationSpec]]]:
        if not tool_specs:
            return [], {}

        tool_combos: set[tuple[str, str]] = set()
        for ts in tool_specs:
            tool_combos.add((ts.disk_mode, ts.network))

        sock_paths: dict[tuple[str, str], str] = {}
        for disk_mode, network in tool_combos:
            if default_readonly is not None:
                if default_readonly and disk_mode == "disk":
                    raise ValueError(
                        f"Task with readonly_sandbox=True cannot have tools "
                        f"with disk_mode='disk' in {error_context}"
                    )
                read_only_project = default_readonly or disk_mode != "disk"
            else:
                read_only_project = disk_mode != "disk"

            mount_project = disk_mode != "nodisk"

            host_tz = (
                True
                if self.manifest.meta.use_host_timezone is None
                else self.manifest.meta.use_host_timezone
            )

            container = manager.create_container(
                sandbox_name=sandbox_name,
                image=worker_image,
                network=network,
                read_only_project=read_only_project,
                mount_project=mount_project,
                verbose=self.manifest.meta.verbosity or 0,
                use_host_timezone=host_tz,
                extra_run_args=extra_run_args,
            )
            if container.worker_socket_path is not None:
                sock_paths[(disk_mode, network)] = str(container.worker_socket_path)

        tool_instances: list[Tool] = []
        tools_confirmations: dict[str, list[ConfirmationSpec]] = {}
        for ts in tool_specs:
            cls = discovered_tools.get(ts.name)
            if cls is not None:
                if cls.fname() == "SubAgents":
                    raise ValueError(
                        "SubAgents is a special tool that cannot be used "
                        f"as a regular tool in {error_context}"
                    )
                combo_sock = sock_paths.get((ts.disk_mode, ts.network))
                if combo_sock is None:
                    raise ValueError(
                        f"No container socket found for tool '{ts.name}' "
                        f"with disk_mode='{ts.disk_mode}', network='{ts.network}' "
                        f"in {error_context}"
                    )
                instance = cls(socket_path=combo_sock, **ts.config)
                tool_instances.append(instance)
                if ts.confirmations:
                    tools_confirmations[ts.name] = ts.confirmations
            else:
                raise ValueError(f"Unknown tool: {ts.name}")
        return tool_instances, tools_confirmations

    def _get_client_for_task_engine(
        self, engine_name: str | None, context: str
    ) -> InferenceClient:
        if not self.manifest.inference_engines:
            return self._default_client
        if engine_name is None:
            return self._default_client
        for e in self.manifest.inference_engines:
            if e.name == engine_name:
                return self.build_client(e)
        raise ValueError(
            f"Inference engine '{engine_name}' referenced by {context} "
            "not found in manifest"
        )

    def _resolve_prompt_text(
        self,
        spec: str | None,
        custom: str | None,
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        if custom is not None and custom != "":
            try:
                tmpl = self._jinja_env.from_string(custom)
                return tmpl.render(**self._context, **(extra_context or {}))
            except UndefinedError:
                logger.warning(
                    "Undefined variable in custom prompt template: %s", custom
                )
                return custom
        prompt_spec = spec or "coding"
        for base_dir in self._prompts_dirs:
            for suffix in (".j2", ".txt", ""):
                candidate = base_dir / f"{prompt_spec}{suffix}"
                if candidate.is_file():
                    contents = candidate.read_text(encoding="utf-8")
                    if str(candidate).endswith(".j2"):
                        try:
                            tmpl = self._jinja_env.from_string(contents)
                            return tmpl.render(**self._context, **(extra_context or {}))
                        except UndefinedError:
                            logger.warning(
                                "Undefined variable in prompt template '%s': %s",
                                candidate,
                                contents,
                            )
                            return contents
                    else:
                        return contents
        return "You are a helpful assistant."

    def _create_task_resources(
        self,
        task: TaskSpec | None = None,
        create_conversion_container: bool = False,
    ) -> TaskResources:
        """Create task resources: tools, sandbox, containers, and related state.

        Returns a TaskResources dataclass with all created resources.
        """
        tool_instances: list[Tool] = []
        tools_confirmations: dict[str, list[ConfirmationSpec]] = {}
        sandbox: SandboxDirs | None = None
        manager: SandboxManager | None = None
        sandbox_name: str | None = None
        action_lock: contextlib.AbstractContextManager[None] | None = None
        conversion_sandbox_obj: ConversionSandbox | None = None

        if task is None:
            task = self.manifest.tasks[0]

        if not task.tools and not task.subagents and not create_conversion_container:
            logger.info("Task %s has no tools; skipping sandbox setup", task.name)
            return TaskResources(
                tool_instances=tool_instances,
                tools_confirmations=tools_confirmations,
                sandbox=sandbox,
                manager=manager,
                sandbox_name=sandbox_name,
                action_lock=action_lock,
                conversion_sandbox=conversion_sandbox_obj,
            )

        effective_engine = self._engine or SandboxManager.available_engine()
        sandbox_name = _make_sandbox_name(task.name)
        if self.manifest.meta.ephemeral_sandbox:
            sandbox_name = (
                base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().strip("=")
            )
        manager = SandboxManager(base_path=self._base_path, engine=effective_engine)

        try:
            self._build_worker_image(
                manager, task.worker_image, task.worker_image_containerfile
            )

            # Build images for sub-agents with different worker images
            if task.subagents:
                for sub_spec in task.subagents:
                    if sub_spec.worker_image == task.worker_image:
                        continue
                    self._build_worker_image(
                        manager,
                        sub_spec.worker_image,
                        sub_spec.worker_image_containerfile,
                    )

            sandbox = manager.create_sandbox(
                sandbox_name=sandbox_name,
                project_path=task.project or None,
                force_copy_files=task.force_copy_files or None,
                committer_name=self.manifest.meta.committer_name
                or TIZ_COMMIT_AUTHOR_NAME,
                committer_email=self.manifest.meta.committer_email
                or TIZ_COMMIT_AUTHOR_EMAIL,
            )

            if task.tools:
                discovered_tools = BaseTaskExecutor._discover_tools(
                    [self._base_path / "tools"]
                )
                tool_instances, tools_confirmations = (
                    self._create_tool_instances_from_specs(
                        task.tools,
                        discovered_tools,
                        sandbox_name,
                        task.worker_image,
                        manager,
                        error_context="task.tools",
                        default_readonly=task.readonly_sandbox,
                        extra_run_args=task.extra_container_args,
                    )
                )

            if task.subagents:
                sub_agents_tool = self._create_sub_agents_tool(
                    task, sandbox_name, manager
                )
                if sub_agents_tool is not None:
                    tool_instances.append(sub_agents_tool)

            if create_conversion_container:
                conv_host_tz = (
                    True
                    if self.manifest.meta.use_host_timezone is None
                    else self.manifest.meta.use_host_timezone
                )
                conversion_container = manager.create_container(
                    sandbox_name=sandbox_name,
                    image=task.worker_image,
                    network="none",
                    read_only_project=True,
                    mount_project=False,
                    verbose=self.manifest.meta.verbosity or 0,
                    use_host_timezone=conv_host_tz,
                    extra_run_args=task.extra_container_args,
                )
                if conversion_container.worker_socket_path is not None:
                    conversion_bash = Bash(
                        socket_path=str(conversion_container.worker_socket_path)
                    )
                    if conversion_container.shared_dir is not None:
                        conversion_sandbox_obj = ConversionSandbox(
                            tool=conversion_bash,
                            base_path=str(conversion_container.shared_dir),
                        )

            action_lock = manager.sandbox_lock(sandbox_name)
        except Exception:
            if manager is not None and sandbox_name is not None:
                manager.kill_all_containers(sandbox_name)
            if sandbox is not None:
                try:
                    sandbox.validate_git_project_dir()
                except Exception as exc:
                    logger.warning(
                        "Failed to validate git project dir during cleanup: %s", exc
                    )
            raise

        return TaskResources(
            tool_instances=tool_instances,
            tools_confirmations=tools_confirmations,
            sandbox=sandbox,
            manager=manager,
            sandbox_name=sandbox_name,
            action_lock=action_lock,
            conversion_sandbox=conversion_sandbox_obj,
        )

    @staticmethod
    def build_client(
        engine_spec: InferenceEngineSpec,
    ) -> InferenceClient:
        et = engine_spec.engine_type.lower()
        if et in ("llamacpp", "llama.cpp"):
            model = engine_spec.model or ""
            return LlamaCpp(
                host=engine_spec.host,
                timeout=engine_spec.timeout,
                message_timeout=engine_spec.message_timeout,
                verify_ssl=engine_spec.verify_ssl,
                ca_cert=engine_spec.ca_cert,
                default_model=model,
                sampling_params=engine_spec.sampling_params,
                preserve_thinking=engine_spec.preserve_thinking,
            )
        if et in ("dwarfstar4", "ds4"):
            model = engine_spec.model or ""
            return DwarfStar4(
                host=engine_spec.host,
                timeout=engine_spec.timeout,
                message_timeout=engine_spec.message_timeout,
                verify_ssl=engine_spec.verify_ssl,
                ca_cert=engine_spec.ca_cert,
                default_model=model,
                sampling_params=engine_spec.sampling_params,
                api_key=engine_spec.api_key,
            )
        if et == "openrouter":
            model = engine_spec.model or "openrouter/free"
            return OpenRouter(
                api_key=engine_spec.api_key or "",
                timeout=engine_spec.timeout,
                message_timeout=engine_spec.message_timeout,
                verify_ssl=engine_spec.verify_ssl,
                ca_cert=engine_spec.ca_cert,
                default_model=model,
                sampling_params=engine_spec.sampling_params,
                preserve_thinking=engine_spec.preserve_thinking,
            )
        raise ValueError(f"Unknown inference engine type: {engine_spec.engine_type}")

    def _create_default_client(self) -> InferenceClient:
        if not self.manifest.inference_engines:
            raise ValueError("No inference engines configured and no client provided")
        first_engine = self.manifest.inference_engines[0]
        return self.build_client(first_engine)

    def _get_client_for_task(self, task: TaskSpec) -> InferenceClient:
        return self._get_client_for_task_engine(
            task.inference_engine, f"task '{task.name}'"
        )

    def _get_engine_name(self, task_name: str) -> str:
        # Handle sub-agent names like "taskname/subagentname" -> "taskname"
        base_name = task_name.split("/", 1)[0]
        for task in self.manifest.tasks:
            if task.name == base_name:
                if task.inference_engine:
                    return task.inference_engine
                break
        if self.manifest.inference_engines:
            return self.manifest.inference_engines[0].name
        return "unknown"

    @staticmethod
    def _send_with_retry(
        chat: Chat,
        message: str,
        timeout: float | None = None,
        max_retries: int = 3,
        use_replay: bool = False,
    ) -> dict[str, Any]:
        try:
            if use_replay:
                return chat.replay(timeout=timeout)
            return chat.send_message(message, timeout=timeout)
        except Exception as exc:
            last_exception: Exception = exc
            for attempt in range(max_retries):
                delay = 2**attempt
                time.sleep(delay)
                try:
                    return chat.replay(timeout=timeout)
                except Exception as e:
                    last_exception = e
            raise last_exception from exc

    def resolve_prompt(self, task: TaskSpec, sandbox: SandboxDirs | None = None) -> str:
        meta = self.manifest.meta

        project_dir = Path(sandbox.project_dir) if sandbox is not None else None

        context: dict[str, Any] = {}
        if project_dir is not None and project_dir.exists():
            agents_path = project_dir / "AGENTS.md"
            if agents_path.is_file():
                context["AGENTS_md"] = agents_path.read_text(encoding="utf-8")
            context["git_repo"] = (
                "yes" if SandboxDirs.is_git_repo(project_dir) else "no"
            )
            context["project_dir"] = CONTAINER_MOUNT_PROJECT
        else:
            context["git_repo"] = "no"
            context["project_dir"] = ""
        now = datetime.now().astimezone()
        context["date"] = now.strftime("%Y-%m-%d")
        context["datetime"] = now.isoformat()
        context["uuid"] = str(uuid.uuid4())
        context["general_shared_dir"] = CONTAINER_MOUNT_SHARED
        context["own_shared_dir"] = CONTAINER_MOUNT_OWN_SHARED
        context["committer_name"] = meta.committer_name
        context["committer_email"] = meta.committer_email

        context.update(self._categorize_tools(task.tools))
        return self._resolve_prompt_text(
            task.sys_prompt, task.sys_prompt_custom, extra_context=context
        )

    def _accumulate_usage(
        self,
        task_name: str,
        result: dict[str, Any],
        engine_name: str | None = None,
    ) -> None:
        if engine_name is None:
            engine_name = self._get_engine_name(task_name)
        with self._task_usage_lock:
            if task_name not in self._task_usage:
                self._task_usage[task_name] = {}
            engine_usages = self._task_usage[task_name]
            if engine_name not in engine_usages:
                engine_usages[engine_name] = {
                    "prompt_tokens": 0,
                    "prompt_time": 0.0,
                    "completion_tokens": 0,
                    "completion_time": 0.0,
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.0,
                    "tool_calls": [],
                }
            usage = engine_usages[engine_name]
            for key in (
                "prompt_tokens",
                "prompt_time",
                "completion_tokens",
                "completion_time",
                "cached_tokens",
                "cache_write_tokens",
                "cost",
            ):
                usage[key] += result.get(key, 0) or 0
            if "tool_calls" in result:
                usage.setdefault("tool_calls", []).extend(result["tool_calls"])

    @property
    def task_usage(self) -> dict[str, dict[str, Any]]:
        with self._task_usage_lock:
            result: dict[str, dict[str, Any]] = {}
            for task_name, engine_usages in self._task_usage.items():
                flat: dict[str, Any] = {
                    "prompt_tokens": 0,
                    "prompt_time": 0.0,
                    "completion_tokens": 0,
                    "completion_time": 0.0,
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.0,
                    "tool_calls": [],
                }
                for usage in engine_usages.values():
                    flat["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    flat["prompt_time"] += usage.get("prompt_time", 0)
                    flat["completion_tokens"] += usage.get("completion_tokens", 0)
                    flat["completion_time"] += usage.get("completion_time", 0)
                    flat["cached_tokens"] += usage.get("cached_tokens", 0)
                    flat["cache_write_tokens"] += usage.get("cache_write_tokens", 0)
                    flat["cost"] += usage.get("cost", 0)
                    flat["tool_calls"].extend(
                        copy.deepcopy(usage.get("tool_calls", []))
                    )
                result[task_name] = flat
            return result

    def _save_chat_log(self, chat: Chat, task_name: str) -> None:
        if not self.manifest.meta.save_full_logs:
            return
        self._save_conv_log(chat.conv, task_name)

    def _save_toolcalls_log(self, task_name: str) -> None:
        if not self.manifest.meta.save_full_toolcalls:
            return
        slug_task = _make_sandbox_name(task_name)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self._base_path / "logs" / "tools"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create toolcalls log dir %s: %s", log_dir, exc)
            return
        with self._task_usage_lock:
            engine_usages = self._task_usage.get(task_name, {})
            for engine_name, usage in engine_usages.items():
                tool_calls = usage.get("tool_calls", [])
                if not tool_calls:
                    continue
                slug_engine = _make_sandbox_name(engine_name)
                log_path = log_dir / f"{slug_task}-{slug_engine}-{now}.json"
                try:
                    with log_path.open("w") as f:
                        json.dump(tool_calls, f)
                except OSError as exc:
                    logger.warning(
                        "Failed to write toolcalls log for %s/%s: %s",
                        task_name,
                        engine_name,
                        exc,
                    )

    def _cleanup_resources(
        self,
        manager: SandboxManager | None,
        sandbox_name: str | None,
        sandbox: SandboxDirs | None,
    ) -> None:
        if self.manifest.meta.delete_sandbox_on_exit:
            if sandbox is not None:
                sandbox.sync_to_original_auto_rebase()
            if manager is not None and sandbox_name is not None:
                manager.kill_and_delete_sandbox(sandbox_name)
        else:
            if manager is not None and sandbox_name is not None:
                manager.kill_all_containers(sandbox_name)
            if sandbox is not None:
                sandbox.validate_git_project_dir()

    def _save_full_usage(self) -> None:
        if not self.manifest.meta.save_full_usage_details:
            return
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self._base_path / "logs" / "usage"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create usage log dir %s: %s", log_dir, exc)
            return
        with self._task_usage_lock:
            usage_snapshot = {
                k: {ek: copy.deepcopy(ev) for ek, ev in v.items()}
                for k, v in self._task_usage.items()
            }
        for task_name, engine_usages in usage_snapshot.items():
            for engine_name, usage in engine_usages.items():
                slug_engine = _make_sandbox_name(engine_name)
                slug_task = _make_sandbox_name(task_name)
                log_path = log_dir / f"{now}_{slug_engine}_{slug_task}.log"
                saved = {k: v for k, v in usage.items() if k != "tool_calls"}
                try:
                    with log_path.open("w") as f:
                        json.dump(saved, f)
                except OSError as exc:
                    logger.warning("Failed to write usage log %s: %s", log_path, exc)
