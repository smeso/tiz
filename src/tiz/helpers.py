from __future__ import annotations

import re
import subprocess
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from pathlib import Path

from tiz.base_task_executor import BaseTaskExecutor
from tiz.interactive_chat import InteractiveChat
from tiz.log import get_logger
from tiz.manifest_executor import ManifestExecutor
from tiz.manifest_parser import (
    InferenceEngineSpec,
    IteratorAction,
    Manifest,
    ManifestParser,
    PromptsAction,
    merge,
)
from tiz.sandbox_dirs import (
    TIZ_COMMIT_AUTHOR_EMAIL,
    TIZ_COMMIT_AUTHOR_NAME,
)
from tiz.sandbox_manager import SandboxManager

logger = get_logger(__name__)


def _format_exc(exc: Exception, verbosity: int = 0) -> str:
    if verbosity >= 2:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    tb = (
        traceback.extract_tb(exc.__traceback__)
        if exc.__traceback__
        else traceback.StackSummary()
    )
    project_frames = [
        f for f in tb if "tiz" in f.filename or "site-packages" not in f.filename
    ]
    if not project_frames:
        project_frames = tb[-1:] if tb else []
    parts = [str(exc)]
    for f in project_frames:
        parts.append(f"  at {f.filename}:{f.lineno}")
    return "\n".join(parts)


def get_credits(engines: list[InferenceEngineSpec]) -> list[dict[str, str | float]]:
    """Get credits info for each inference engine.

    Returns a list of dicts with keys: name, total_credits, total_usage, remaining.
    """
    results: list[dict[str, str | float]] = []
    for engine in engines:
        client = BaseTaskExecutor.build_client(engine)
        credits = client.get_credits()
        total_credits = credits.get("total_credits", 0.0)
        total_usage = credits.get("total_usage", 0.0)
        remaining = total_credits - total_usage
        results.append(
            {
                "name": engine.name,
                "total_credits": total_credits,
                "total_usage": total_usage,
                "remaining": remaining,
            },
        )
    return results


def exec_cmd(
    manifest: Manifest,
    base_path: Path,
    task_name: str | None = None,
    cmd_args: list[str] | None = None,
    extra_run_args: list[str] | None = None,
) -> str | None:
    """Execute a command inside a sandboxed container."""
    if not manifest.tasks:
        return "Error: no tasks found in manifests"

    if task_name is None:
        task = manifest.tasks[0]
    else:
        matching = [t for t in manifest.tasks if t.name == task_name]
        if not matching:
            return f"Error: task '{task_name}' not found in manifests"
        task = matching[0]

    engine = manifest.meta.container_engine or SandboxManager.available_engine()
    if engine is None:
        return "Error: no container engine (podman or docker) found"

    manager = SandboxManager(base_path=base_path, engine=engine)

    sandbox_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", task.name.lower())
    try:
        manager.create_sandbox(
            sandbox_name=sandbox_name,
            project_path=task.project or None,
            force_copy_files=task.force_copy_files or None,
            committer_name=manifest.meta.committer_name or TIZ_COMMIT_AUTHOR_NAME,
            committer_email=manifest.meta.committer_email or TIZ_COMMIT_AUTHOR_EMAIL,
        )
    except Exception as exc:
        return (
            f"Error creating sandbox: {_format_exc(exc, manifest.meta.verbosity or 0)}"
        )

    cmd_to_run = cmd_args or ["/bin/bash", "-l"]

    image = task.worker_image
    try:
        container = manager.create_container(
            sandbox_name=sandbox_name,
            image=image,
            network="internet",
            read_only_project=task.readonly_sandbox,
            extra_run_args=extra_run_args,
            verbose=manifest.meta.verbosity or 0,
            use_host_timezone=(
                True
                if manifest.meta.use_host_timezone is None
                else manifest.meta.use_host_timezone
            ),
        )
    except Exception as exc:
        manager.kill_and_delete_sandbox(sandbox_name)
        return f"Error creating container: {_format_exc(exc, manifest.meta.verbosity or 0)}"

    container_id = container.container_id
    if container_id is None:
        manager.kill_and_delete_sandbox(sandbox_name)
        return "Error: container ID is None"

    logger.info("Starting container for task '%s'...", task.name)
    exec_cmd_list = [engine, "exec", "-it", container_id, *cmd_to_run]
    try:
        proc = subprocess.Popen(exec_cmd_list)
        try:
            proc.wait()
            logger.info("Exiting...")
        except KeyboardInterrupt:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            return "interrupted"
        return None
    finally:
        container.stop(timeout=0)


def parse_manifest(
    base_path: Path,
    manifests: list[Path],
    default_options: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[Manifest | None, str | None]:
    """Parse manifest files and merge them into a single Manifest object."""
    manifest_paths: list[Path] = []

    config_yaml = base_path / "config.yaml"
    if config_yaml.is_file():
        manifest_paths.append(config_yaml)

    config_d_dir = base_path / "config.d"
    if config_d_dir.is_dir():
        manifest_paths.extend(sorted(config_d_dir.glob("*.yaml")))

    for m in manifests:
        if m.is_absolute() or m.exists():
            manifest_paths.append(m)
        else:
            manifest_paths.append(base_path / "manifests" / m)

    parsed: list[Manifest] = []
    try:
        if default_options:
            parser = ManifestParser(data=default_options, path=None)
            parsed.append(parser.get_manifest())

        for p in manifest_paths:
            parser = ManifestParser(data={}, path=p)
            parsed.append(parser.get_manifest())

        if options:
            parser = ManifestParser(data=options, path=None)
            parsed.append(parser.get_manifest())

        manifest = merge(parsed)
    except Exception as exc:
        return None, f"Manifest parsing/merge error: {_format_exc(exc)}"

    if not can_parallelize(manifest):
        manifest.meta.parallelism = 1

    return manifest, None


def ask_confirmations(manifest: Manifest) -> bool:
    """Check if any tool in any task (or subtask) has confirmations configured."""
    for task in manifest.tasks:
        for tool in task.tools:
            if tool.confirmations:
                return True
        for subagent in task.subagents:
            for tool in subagent.tools:
                if tool.confirmations:
                    return True
    return False


def can_parallelize(manifest: Manifest) -> bool:
    """Check if the manifest can actually benefit from parallelism.

    Returns True if meta.parallelism > 1 and there are either:
    - 2+ consecutive tasks with allow_parallel_run=True, or
    - an action with parallel_message_groups=True AND more than 1 message group.
    """
    if manifest.meta.parallelism is None or manifest.meta.parallelism <= 1:
        return False

    # Check for 2+ consecutive tasks that allow parallel run
    consecutive_count = 0
    for task in manifest.tasks:
        if task.allow_parallel_run:
            consecutive_count += 1
            if consecutive_count >= 2:
                return True
        else:
            consecutive_count = 0

    # Check for actions with parallel_message_groups AND multiple groups
    for task in manifest.tasks:
        for action in task.actions:
            if (
                isinstance(action, PromptsAction)
                and action.parallel_message_groups
                and len(action.message_groups) > 1
            ):
                return True
            if isinstance(action, IteratorAction) and action.parallel_message_groups:
                # IteratorAction usually has multiple items and doesn't need multiple
                # message_groups to parallelize
                return True

    logger.warning(
        "Cannot parallelize: no consecutive parallel tasks (need 2+) and no "
        "parallelizable actions with multiple groups found",
    )
    return False


def run(
    manifest: Manifest,
    base_path: Path,
    update_callback: Callable[[dict[str, Any], str | None], None] | None = None,
    context: dict[str, Any] | None = None,
    confirm_callback: Callable[
        [dict[str, Any], Callable[[dict[str, Any], bool], str | None], str | None],
        bool,
    ]
    | None = None,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Execute all tasks in the manifest."""
    try:
        executor = ManifestExecutor(
            manifest=manifest,
            base_path=base_path,
            update_callback=update_callback,
            context=context,
            confirm_callback=confirm_callback,
        )
    except KeyboardInterrupt:
        return {}, "interrupted"
    except Exception as exc:
        return (
            {},
            f"Manifest execution error: {_format_exc(exc, manifest.meta.verbosity or 0)}",
        )
    try:
        executor.execute()
    except KeyboardInterrupt:
        return executor.task_usage, "interrupted"
    except Exception as exc:
        return (
            executor.task_usage,
            f"Manifest execution error: {_format_exc(exc, manifest.meta.verbosity or 0)}",
        )

    return executor.task_usage, None


def chat(
    manifest: Manifest,
    base_path: Path,
    task_name: str | None = None,
    update_callback: Callable[[dict[str, Any], str | None], None] | None = None,
    input_callback: Callable[[], dict[str, str] | None] | None = None,
    context: dict[str, Any] | None = None,
    confirm_callback: Callable[
        [dict[str, Any], Callable[[dict[str, Any], bool], str | None], str | None],
        bool,
    ]
    | None = None,
    *,
    enable_recording: bool = False,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Run an interactive chat session for the manifest."""
    try:
        interactive = InteractiveChat(
            manifest=manifest,
            base_path=base_path,
            task_name=task_name,
            update_callback=update_callback,
            input_callback=input_callback,
            context=context,
            confirm_callback=confirm_callback,
            enable_recording=enable_recording,
        )
    except KeyboardInterrupt:
        return {}, "interrupted"
    except Exception as exc:
        return (
            {},
            f"Interactive chat error: {_format_exc(exc, manifest.meta.verbosity or 0)}",
        )
    try:
        interactive.run()
    except KeyboardInterrupt:
        return interactive.task_usage, "interrupted"
    except Exception as exc:
        return (
            interactive.task_usage,
            f"Interactive chat error: {_format_exc(exc, manifest.meta.verbosity or 0)}",
        )

    return interactive.task_usage, None
