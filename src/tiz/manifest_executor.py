"""Execute tasks defined in a parsed manifest."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from tiz.base_task_executor import BaseTaskExecutor, _make_sandbox_name
from tiz.chat import Chat
from tiz.log import get_logger
from tiz.manifest_parser import (
    CmdAction,
    ConfirmationSpec,
    IteratorAction,
    PromptsAction,
    RepeaterAction,
    ScoringAction,
    TaskSpec,
)

if TYPE_CHECKING:
    from tiz.inference_clients import InferenceClient
    from tiz.sandbox_dirs import SandboxDirs
    from tiz.sandbox_manager import SandboxManager
    from tiz.tools.base import Tool

logger = get_logger(__name__)

_CTX_RATIO_FALLBACK = 0.9

_SCORING_TEMPLATE = (
    "You are a scoring assistant. Given the following branches and the "
    "scoring prompt, select the best branch.\n"
    "\n"
    "Scoring prompt: {{ scoring_prompt }}\n"
    "\n"
    "Branches:\n"
    "{% for branch in branches %}"
    "- {{ branch }}\n"
    "{% endfor %}"
    "\n"
    "Return only the name of the best branch.\n"
    "Do not add any preamble or any other text."
)


class ManifestExecutor(BaseTaskExecutor):
    """Execute tasks defined in a parsed manifest."""

    def _emit_event(self, data: dict[str, Any]) -> None:
        if self._update_callback is not None:
            self._update_callback(data, None)

    def _make_chat(
        self,
        client: InferenceClient,
        sys_prompt: str,
        tool_instances: list[Tool],
        tools_confirmations: dict[str, list[ConfirmationSpec]],
    ) -> Chat:
        return Chat(
            client=client,
            sys_prompt=sys_prompt,
            tools=tool_instances,
            update_callback=self._update_callback,
            confirm_callback=self._confirm_callback,
            tools_confirmations=tools_confirmations,
            ctx_ratio=self.manifest.meta.summarizer_context_ratio
            or _CTX_RATIO_FALLBACK,
        )

    def _resolve_engine_names_to_clients(
        self, names: list[str]
    ) -> list[InferenceClient]:
        clients: list[InferenceClient] = []
        for name in names:
            engine_spec = None
            for e in self.manifest.inference_engines:
                if e.name == name:
                    engine_spec = e
                    break
            if engine_spec is None:
                raise ValueError(
                    f"Inference engine '{name}' referenced in scoring action "
                    f"not found in manifest"
                )
            clients.append(self.build_client(engine_spec))
        return clients

    def _run_iterator_action(
        self,
        task_name: str,
        action: IteratorAction,
        sys_prompt: str,
        tool_instances: list[Tool],
        tools_confirmations: dict[str, list[ConfirmationSpec]],
        client: InferenceClient,
    ) -> None:
        self._emit_event(
            {
                "tiz-internal": {
                    "action": "iterator",
                    "status": "generating_items",
                    "task": task_name,
                    "prompt": action.input_prompt,
                }
            }
        )
        input_chat = self._make_chat(
            client=client,
            sys_prompt=sys_prompt,
            tool_instances=tool_instances,
            tools_confirmations=tools_confirmations,
        )
        result = self._send_with_retry(input_chat, action.input_prompt)
        self._accumulate_usage(task_name, result)
        self._save_chat_log(input_chat, task_name)

        output: str = result.get("message", "")
        lines = [line.strip() for line in output.splitlines() if line.strip()]

        total_lines = len(lines)
        if action.parallel_message_groups:
            with ThreadPoolExecutor(
                max_workers=max(1, self.manifest.meta.parallelism or 1)
            ) as pool:
                futures: dict[Future[None], str] = {
                    pool.submit(
                        self._run_iterator_line,
                        task_name,
                        line,
                        action,
                        sys_prompt,
                        tool_instances,
                        tools_confirmations,
                        client,
                        line_idx=idx + 1,
                        total_lines=total_lines,
                    ): line
                    for idx, line in enumerate(lines)
                }
                for future in as_completed(futures):
                    line = futures[future]
                    try:
                        future.result()
                        logger.info("Iterator line %s completed", line)
                    except Exception:
                        logger.exception("Iterator line %s failed", line)
        else:
            for idx, line in enumerate(lines):
                self._run_iterator_line(
                    task_name,
                    line,
                    action,
                    sys_prompt,
                    tool_instances,
                    tools_confirmations,
                    client,
                    line_idx=idx + 1,
                    total_lines=total_lines,
                )

    def _run_iterator_line(
        self,
        task_name: str,
        line: str,
        action: IteratorAction,
        sys_prompt: str,
        tool_instances: list[Tool],
        tools_confirmations: dict[str, list[ConfirmationSpec]],
        client: InferenceClient,
        line_idx: int,
        total_lines: int,
    ) -> None:
        for pg_idx, prompt_group in enumerate(action.prompt_groups):
            self._emit_event(
                {
                    "tiz-internal": {
                        "action": "iterator",
                        "status": "processing_line",
                        "task": task_name,
                        "line": line,
                        "group": pg_idx + 1,
                        "total_groups": len(action.prompt_groups),
                        "current_line": line_idx,
                        "total_lines": total_lines,
                    }
                }
            )
            rendered_group = [
                self._jinja_env.from_string(m).render(item=line) for m in prompt_group
            ]
            self._run_message_group(
                task_name,
                rendered_group,
                sys_prompt,
                tool_instances,
                tools_confirmations,
                client,
            )

    def _run_repeater_action(
        self,
        task_name: str,
        action: RepeaterAction,
        sys_prompt: str,
        tool_instances: list[Tool],
        tools_confirmations: dict[str, list[ConfirmationSpec]],
        client: InferenceClient,
    ) -> None:
        for i in range(1, action.repeat + 1):
            self._emit_event(
                {
                    "tiz-internal": {
                        "action": "repeater",
                        "status": "iteration",
                        "task": task_name,
                        "iteration": i,
                        "total_iterations": action.repeat,
                    }
                }
            )
            for prompt_group in action.prompt_groups:
                chat = self._make_chat(
                    client=client,
                    sys_prompt=sys_prompt,
                    tool_instances=tool_instances,
                    tools_confirmations=tools_confirmations,
                )
                for prompt in prompt_group:
                    tmpl = self._jinja_env.from_string(prompt)
                    rendered = tmpl.render(item=str(i))
                    result = self._send_with_retry(chat, rendered)
                    self._accumulate_usage(task_name, result)
                self._save_chat_log(chat, task_name)

    def _run_scoring_line(
        self,
        task_name: str,
        action: ScoringAction,
        sys_prompt: str,
        tool_instances: list[Tool],
        tools_confirmations: dict[str, list[ConfirmationSpec]],
        sandbox: SandboxDirs,
        line: str | None,
        client: InferenceClient,
    ) -> None:
        original_branch = sandbox.git_capture_branch()

        scores: dict[str, int] = defaultdict(int)

        round_clients = self._resolve_engine_names_to_clients(action.engines)
        scoring_clients = self._resolve_engine_names_to_clients(action.scoring_engines)
        round_client_names = action.engines
        scoring_client_names = action.scoring_engines

        branch_engines: dict[str, str | None] = {}
        scoring_votes: dict[str, list[str | None]] = {}

        branches: list[str] = []
        try:
            for round_idx in range(action.rounds):
                round_engine_name = (
                    round_client_names[round_idx % len(round_client_names)]
                    if round_client_names
                    else None
                )
                for pg_idx, prompt_group in enumerate(action.prompt_groups):
                    self._emit_event(
                        {
                            "tiz-internal": {
                                "action": "scoring",
                                "status": "round",
                                "task": task_name,
                                "round": round_idx + 1,
                                "total_rounds": action.rounds,
                                "group": pg_idx + 1,
                                "total_groups": len(action.prompt_groups),
                                "engine": (
                                    round_client_names[
                                        round_idx % len(round_client_names)
                                    ]
                                    if round_client_names
                                    else None
                                ),
                            },
                        },
                    )
                    branch_name = (
                        f"tiz/scoring_{_make_sandbox_name(task_name)}_r{len(branches)}"
                    )
                    sandbox.git_create_branch(branch_name)
                    branches.append(branch_name)
                    branch_engines[branch_name] = round_engine_name
                    if line is not None:
                        rendered_group = [
                            self._jinja_env.from_string(m).render(item=line)
                            for m in prompt_group
                        ]
                    else:
                        rendered_group = list(prompt_group)
                    round_client = (
                        round_clients[round_idx % len(round_clients)]
                        if round_clients
                        else client
                    )
                    self._run_message_group(
                        task_name,
                        rendered_group,
                        sys_prompt,
                        tool_instances,
                        tools_confirmations,
                        round_client,
                        engine_name=round_engine_name,
                    )
                    sandbox.git_checkout(original_branch)

            tmpl = self._jinja_env.from_string(_SCORING_TEMPLATE)
            rendered = tmpl.render(
                scoring_prompt=action.scoring_prompt,
                branches=branches,
            )
            for scr_idx in range(action.scoring_rounds):
                scoring_engine_name = (
                    scoring_client_names[scr_idx % len(scoring_client_names)]
                    if scoring_client_names
                    else None
                )
                self._emit_event(
                    {
                        "tiz-internal": {
                            "action": "scoring",
                            "status": "scoring_step",
                            "task": task_name,
                            "scoring_step": scr_idx + 1,
                            "total_scoring_steps": action.scoring_rounds,
                            "prompt": rendered,
                            "engine": (
                                scoring_client_names[
                                    scr_idx % len(scoring_client_names)
                                ]
                                if scoring_client_names
                                else None
                            ),
                        },
                    },
                )
                scoring_client = (
                    scoring_clients[scr_idx % len(scoring_clients)]
                    if scoring_clients
                    else client
                )
                scoring_chat = self._make_chat(
                    client=scoring_client,
                    sys_prompt=sys_prompt,
                    tool_instances=tool_instances,
                    tools_confirmations=tools_confirmations,
                )
                result = self._send_with_retry(scoring_chat, rendered)
                self._accumulate_usage(
                    task_name, result, engine_name=scoring_engine_name
                )
                self._save_chat_log(scoring_chat, task_name)

                selected = result.get("message", "").strip()
                if selected in branches:
                    scores[selected] += 1
                    scoring_votes.setdefault(selected, []).append(scoring_engine_name)
                else:
                    for branch in branches:
                        if branch.lower() == selected.lower():
                            scores[branch] += 1
                            scoring_votes.setdefault(branch, []).append(
                                scoring_engine_name
                            )
                            break

            winner = max(scores, key=lambda k: scores[k]) if scores else original_branch

            self._emit_event(
                {
                    "tiz-internal": {
                        "action": "scoring",
                        "status": "winner_selected",
                        "task": task_name,
                        "winner": winner,
                        "winner_engine": branch_engines.get(winner),
                        "votes": scores.get(winner, 0),
                        "scoring_engines": scoring_votes.get(winner, []),
                    },
                },
            )

            sandbox.git_finalize_branches(original_branch, winner, branches)
        except:
            for branch in branches:
                with contextlib.suppress(Exception):
                    sandbox.git_checkout(original_branch)
                    sandbox.git_finalize_branches(
                        original_branch,
                        original_branch,
                        [branch],
                    )
            raise

    def _run_scoring_action(
        self,
        task_name: str,
        action: ScoringAction,
        sys_prompt: str,
        tool_instances: list[Tool],
        tools_confirmations: dict[str, list[ConfirmationSpec]],
        sandbox: SandboxDirs,
        client: InferenceClient,
    ) -> None:
        if not sandbox.is_git_repo(sandbox.project_dir):
            raise RuntimeError(
                f"Scoring action requires a git repository in the sandbox "
                f"project directory for task '{task_name}'"
            )
        if action.rounds < 1:
            raise ValueError(
                f"Scoring rounds must be at least 1 for task '{task_name}', "
                f"got {action.rounds}"
            )
        if action.scoring_rounds < 1:
            raise ValueError(
                f"Scoring rounds must be positive for task '{task_name}', "
                f"got {action.scoring_rounds}"
            )
        if action.scoring_rounds % 2 == 0:
            raise ValueError(
                f"Scoring rounds must be odd for task '{task_name}', "
                f"got {action.scoring_rounds}"
            )
        if not action.scoring_prompt:
            raise ValueError(
                f"Scoring prompt is required for scoring action in task '{task_name}'"
            )
        if action.iterator_input:
            self._emit_event(
                {
                    "tiz-internal": {
                        "action": "scoring",
                        "status": "generating_iterator_items",
                        "task": task_name,
                        "prompt": action.iterator_input,
                    }
                }
            )
            input_chat = self._make_chat(
                client=client,
                sys_prompt=sys_prompt,
                tool_instances=tool_instances,
                tools_confirmations=tools_confirmations,
            )
            result = self._send_with_retry(input_chat, action.iterator_input)
            self._accumulate_usage(task_name, result)
            self._save_chat_log(input_chat, task_name)

            output: str = result.get("message", "")
            lines = [line.strip() for line in output.splitlines() if line.strip()]

            for line in lines:
                self._run_scoring_line(
                    task_name,
                    action,
                    sys_prompt,
                    tool_instances,
                    tools_confirmations,
                    sandbox,
                    line,
                    client,
                )
        else:
            self._run_scoring_line(
                task_name,
                action,
                sys_prompt,
                tool_instances,
                tools_confirmations,
                sandbox,
                None,
                client,
            )

    def _run_message_group(
        self,
        task_name: str,
        msg_group: list[str],
        sys_prompt: str,
        tool_instances: list[Tool],
        tools_confirmations: dict[str, list[ConfirmationSpec]],
        client: InferenceClient,
        engine_name: str | None = None,
    ) -> None:
        chat = self._make_chat(
            client=client,
            sys_prompt=sys_prompt,
            tool_instances=tool_instances,
            tools_confirmations=tools_confirmations,
        )
        for i, msg in enumerate(msg_group):
            self._emit_event(
                {
                    "tiz-internal": {
                        "action": "message_group",
                        "status": "sending",
                        "task": task_name,
                        "message": i + 1,
                        "total_messages": len(msg_group),
                        "prompt": msg,
                    }
                }
            )
            result = self._send_with_retry(chat, msg)
            self._accumulate_usage(task_name, result, engine_name=engine_name)
        self._save_chat_log(chat, task_name)

    def _run_prompts_action(
        self,
        task_name: str,
        action: PromptsAction,
        sys_prompt: str,
        tool_instances: list[Tool],
        tools_confirmations: dict[str, list[ConfirmationSpec]],
        client: InferenceClient,
    ) -> None:
        if action.parallel_message_groups:
            self._emit_event(
                {
                    "tiz-internal": {
                        "action": "prompts",
                        "status": "running",
                        "task": task_name,
                        "prompts_groups": len(action.message_groups),
                        "prompts_parallel": True,
                    }
                }
            )
            with ThreadPoolExecutor(
                max_workers=max(1, self.manifest.meta.parallelism or 1)
            ) as pool:
                futures: dict[Future[None], int] = {
                    pool.submit(
                        self._run_message_group,
                        task_name,
                        msg_group,
                        sys_prompt,
                        tool_instances,
                        tools_confirmations,
                        client,
                    ): idx
                    for idx, msg_group in enumerate(action.message_groups)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                        logger.info("Message group %d completed", idx)
                    except Exception:
                        logger.exception("Message group %d failed", idx)
        else:
            for g_idx, msg_group in enumerate(action.message_groups):
                self._emit_event(
                    {
                        "tiz-internal": {
                            "action": "prompts",
                            "status": "group",
                            "task": task_name,
                            "group": g_idx + 1,
                            "total_groups": len(action.message_groups),
                        }
                    }
                )
                self._run_message_group(
                    task_name,
                    msg_group,
                    sys_prompt,
                    tool_instances,
                    tools_confirmations,
                    client,
                )

    def execute(self) -> None:
        """Execute all tasks from the manifest, respecting parallelism settings."""
        tasks = self.manifest.tasks
        parallelism = max(1, self.manifest.meta.parallelism or 1)

        groups: list[list[TaskSpec]] = []
        current_group: list[TaskSpec] = []
        for task in tasks:
            if task.allow_parallel_run:
                current_group.append(task)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([task])
        if current_group:
            groups.append(current_group)

        for group in groups:
            if len(group) > 1:
                logger.info(
                    "Executing %d parallel tasks with pool size %d",
                    len(group),
                    parallelism,
                )
                with ThreadPoolExecutor(max_workers=parallelism) as pool:
                    futures: dict[Future[None], str] = {
                        pool.submit(self._run_task, task): task.name for task in group
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            future.result()
                            logger.info("Task %s completed", name)
                        except Exception:
                            logger.exception("Task %s failed", name)
            else:
                logger.info("Executing serial task %s", group[0].name)
                self._run_task(group[0])
                logger.info("Serial task %s completed", group[0].name)

        self._save_full_usage()

    def _run_task(self, task: TaskSpec) -> None:
        self._emit_event(
            {
                "tiz-internal": {
                    "status": "starting_task",
                    "task": task.name,
                }
            }
        )
        tool_instances: list[Tool] = []
        tools_confirmations: dict[str, list[ConfirmationSpec]] = {}
        sandbox: SandboxDirs | None = None
        manager: SandboxManager | None = None
        sandbox_name: str | None = None

        try:
            resources = self._create_task_resources(
                task=task,
            )
            tool_instances = resources.tool_instances
            tools_confirmations = resources.tools_confirmations
            sandbox = resources.sandbox
            manager = resources.manager
            sandbox_name = resources.sandbox_name
            action_lock = resources.action_lock

            client = self._get_client_for_task(task)

            if action_lock is None:
                action_lock = contextlib.nullcontext()
            with action_lock:
                for action_idx, action in enumerate(task.actions):
                    action_type = type(action).__name__
                    self._emit_event(
                        {
                            "tiz-internal": {
                                "status": "executing_action",
                                "task": task.name,
                                "action_idx": action_idx + 1,
                                "total_actions": len(task.actions),
                                "action_type": action_type,
                            }
                        }
                    )
                    if isinstance(action, IteratorAction):
                        self._run_iterator_action(
                            task.name,
                            action,
                            self.resolve_prompt(task, sandbox),
                            tool_instances,
                            tools_confirmations,
                            client,
                        )
                    elif isinstance(action, RepeaterAction):
                        self._run_repeater_action(
                            task.name,
                            action,
                            self.resolve_prompt(task, sandbox),
                            tool_instances,
                            tools_confirmations,
                            client,
                        )
                    elif isinstance(action, ScoringAction):
                        if not task.tools or sandbox is None:
                            raise RuntimeError(
                                f"Scoring action requires a sandbox with tools; "
                                f"task '{task.name}' has no tools configured"
                            )
                        self._run_scoring_action(
                            task.name,
                            action,
                            self.resolve_prompt(task, sandbox),
                            tool_instances,
                            tools_confirmations,
                            sandbox,
                            client,
                        )
                    elif isinstance(action, PromptsAction):
                        self._run_prompts_action(
                            task.name,
                            action,
                            self.resolve_prompt(task, sandbox),
                            tool_instances,
                            tools_confirmations,
                            client,
                        )
                    elif isinstance(action, CmdAction):
                        if action.command == "sync":
                            if sandbox is None:
                                logger.warning(
                                    "CmdAction 'sync' requires a sandbox with tools; "
                                    "task '%s' has no tools configured",
                                    task.name,
                                )
                            else:
                                sandbox.sync_to_original_auto_rebase()
                        else:
                            logger.warning(
                                "Unsupported CmdAction command '%s' in task '%s'; "
                                "ignoring",
                                action.command,
                                task.name,
                            )
                if sandbox:
                    sandbox.sync_to_original_auto_rebase()

            self._emit_event(
                {
                    "tiz-internal": {
                        "status": "completed_task",
                        "task": task.name,
                    }
                }
            )

            self._save_toolcalls_log(task.name)
        except KeyboardInterrupt:
            print("\nExiting...")
            raise
        finally:
            self._cleanup_resources(manager, sandbox_name, sandbox)
