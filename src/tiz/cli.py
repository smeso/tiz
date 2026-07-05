"""CLI entrypoint for tiz."""

# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import shlex
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable  # pragma: no cover

from tiz.autocomplete import autocomplete, shellcode
from tiz.helpers import (
    _format_exc,
    ask_confirmations,
    exec_cmd,
    get_credits,
    parse_manifest,
    run,
)
from tiz.helpers import (
    chat as helpers_chat,
)
from tiz.interactive_chat import InteractiveChat
from tiz.log import get_logger, logging_init
from tiz.manifest_parser import (
    DEFAULT_COLOR_INPUT,
    DEFAULT_COLOR_REASONING,
    InferenceEngineSpec,
    ManifestMeta,
)
from tiz.sandbox_dirs import TIZ_COMMIT_AUTHOR_EMAIL, TIZ_COMMIT_AUTHOR_NAME
from tiz.sandbox_manager import SandboxManager
from tiz.web_api import run_simple as web_run_simple

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None  # type: ignore[assignment]  # pragma: no cover

logger = get_logger(__name__)

if readline is not None:  # pragma: no branch

    class _PathCompleter:
        def __init__(self) -> None:
            self._matches: list[str] = []

        def complete(self, text: str, state: int) -> str | None:
            if state == 0:
                needle = str(Path(text or "").expanduser())
                if needle.endswith(os.path.sep):
                    dir_part = Path(needle)
                    prefix = ""
                else:
                    dir_part = Path(needle).parent if needle else Path()
                    prefix = Path(needle).name if needle else ""
                try:
                    entries = sorted(dir_part.iterdir())
                except OSError:
                    entries = []
                self._matches = []
                for entry in entries:
                    if entry.name.startswith(prefix):
                        full = str(dir_part / entry.name)
                        if entry.is_dir():
                            full += os.path.sep
                        self._matches.append(full)
            try:
                return self._matches[state]
            except IndexError:
                return None

    class _ChatCompleter:
        def __init__(self) -> None:
            self._matches: list[str] = []
            self._path_completer: _PathCompleter | None = None

        def complete(self, text: str, state: int) -> str | None:
            line = readline.get_line_buffer()
            stripped = line.strip()
            if " " in line:
                cmd = line.split(maxsplit=1)[0]
                if cmd in ("/attach", "/load", "/save"):
                    if self._path_completer is None:
                        self._path_completer = _PathCompleter()
                    return self._path_completer.complete(text, state)
            if not stripped.startswith("/"):
                return None
            if state == 0:
                chat_commands = [
                    k.split()[0] for k, _ in InteractiveChat.BASE_COMMANDS
                ] + ["/record"]
                self._matches = sorted(c for c in chat_commands if c.startswith(text))
            try:
                return self._matches[state]
            except IndexError:
                return None

    _chat_completer = _ChatCompleter()


def _parse_hex_color(hex_color: str) -> str:
    hc = hex_color.lstrip("#")
    if len(hc) != 6 or not all(c in "0123456789abcdefABCDEF" for c in hc):
        raise argparse.ArgumentTypeError(
            f"invalid hex color value: {hex_color!r} (expected 6 hex digits after optional #)"
        )
    r, g, b = int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)
    return f"{r};{g};{b}"


_ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;]*[a-zA-Z]"  # CSI sequences
    r"|\x1b\][0-9;]*[\x07\x1b]"  # OSC sequences (terminated by BEL or ST)
    r"|\x1b[X^_]"  # SOS, PM, APC (single-char intermediates)
    r"|\x1b[NP]"  # DCS, SOS (2-char)
    r"|\x1b"  # bare escape
)


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences and potentially harmful control chars from AI output."""
    return _ANSI_ESCAPE_RE.sub("", text)


def _is_tty() -> bool:
    return sys.stdout.isatty()


def _set_terminal_title(title: str) -> None:
    """Set window and tab terminal titles via ANSI escape sequences (OSC 0, OSC 30, OSC 31)."""
    if not _is_tty():
        return
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.write(f"\033]30;{title}\007")
    sys.stdout.write(f"\033]31;{title}\007")
    sys.stdout.flush()


def _maybe_ring_bell(manifest_meta: ManifestMeta | None) -> None:
    """Ring terminal bell if enabled and running in a TTY."""
    if manifest_meta is not None and manifest_meta.ring_bell and _is_tty():
        sys.stdout.write("\a")
        sys.stdout.flush()


class StreamUpdater:
    def __init__(
        self,
        hide_reasoning: bool,
        color_reasoning: str,
        color_input: str,
        color: bool = True,
        command: str = "tiz",
        manifests: list[str] | None = None,
        parallelism: int = 1,
    ) -> None:
        self.reasoning = False
        self._reasoning_token_count = 0
        self._hide_reasoning = hide_reasoning
        self._color_reasoning = _parse_hex_color(color_reasoning)
        self._color = color
        self._command = command
        self._manifest_names = manifests or []
        self._color_input = _parse_hex_color(color_input)
        self._prev_title_parts: list[str] = []
        self._parallelism = parallelism
        self._lock = threading.Lock()

    def _update_title(
        self, internal: dict[str, Any], subtask_name: str | None = None
    ) -> None:
        if not _is_tty():
            return
        parts: list[str] = ["tiz"]
        parts.append(self._command)
        if self._manifest_names:
            parts.append(":".join(self._manifest_names))

        status = internal.get("status", "")
        task = internal.get("task", "")
        action = internal.get("action", internal.get("action_type", ""))

        if status == "starting_task":
            parts.append(f"{task}: starting")
        elif status == "completed_task":
            parts.append(f"{task}: completed")
        elif status == "executing_action":
            action_idx = internal.get("action_idx", 0)
            total = internal.get("total_actions", 0)
            atype = internal.get("action_type", "action")
            parts.append(f"{task}: {atype} {action_idx}/{total}")
        elif action == "message_group" and status == "sending":
            msg_num = internal.get("message", 0)
            total = internal.get("total_messages", 0)
            parts = self._prev_title_parts + [f"msg {msg_num}/{total}"]
        elif action == "prompts":
            if status == "running":
                parts.append(f"{task}: prompts (parallel)")
            elif status == "group":
                g = internal.get("group", 0)
                total = internal.get("total_groups", 0)
                parts.append(f"{task}: prompt group {g}/{total}")
        elif action == "iterator":
            if status == "generating_items":
                parts.append(f"{task}: generating items")
            elif status == "processing_line":
                line_idx = internal.get("current_line", 0)
                total = internal.get("total_lines", 0)
                parts.append(f"{task}: line {line_idx}/{total}")
        elif action == "repeater" and status == "iteration":
            it = internal.get("iteration", 0)
            total = internal.get("total_iterations", 0)
            parts.append(f"{task}: iteration {it}/{total}")
        elif action == "scoring":
            if status == "round":
                r = internal.get("round", 0)
                total = internal.get("total_rounds", 0)
                parts.append(f"{task}: scoring round {r}/{total}")
            elif status == "scoring_step":
                s = internal.get("scoring_step", 0)
                total = internal.get("total_scoring_steps", 0)
                parts.append(f"{task}: scoring step {s}/{total}")
            elif status == "winner_selected":
                winner = internal.get("winner", "")
                votes = internal.get("votes", 0)
                parts.append(f"{task}: scoring: {winner} ({votes} votes)")
            elif status == "generating_iterator_items":
                parts.append(f"{task}: generating scoring items")
        else:
            parts.append(f"{task}")

        if subtask_name is not None:
            parts.append(subtask_name)

        self._prev_title_parts = parts
        title = ": ".join(parts)
        _set_terminal_title(title)
        if self._parallelism > 1:
            print(title, file=sys.stderr)

    def _print_transcribe(self, text: str) -> None:
        if self._color:
            sys.stdout.write(
                f"\033[1m\033[38;2;{self._color_input}mTranscription:\033[0m {text}\n"
            )
        else:
            sys.stdout.write(f"Transcription: {text}\n")
        sys.stdout.flush()

    def __call__(self, msg: dict[str, Any], subtask_name: str | None = None) -> None:
        if self._parallelism > 1:
            with self._lock:
                if "tiz-internal" in msg:
                    internal = msg["tiz-internal"]
                    self._update_title(internal, subtask_name)
            return
        if "tiz-internal" in msg:
            internal = msg["tiz-internal"]
            feedback = internal.get("interactive_chat_feedback")
            if feedback is not None:
                print(feedback)
            usage = internal.get("interactive_chat_usage")
            if usage is not None and internal.get("interactive_chat_usage_accumulated"):
                _print_usage(usage)
            prompt = internal.get("prompt")
            if prompt is not None:
                sys.stdout.write("\n")
                if self._color:
                    sys.stdout.write(f"\033[1m\033[38;2;{self._color_input}m")
                    sys.stdout.write(f"{prompt}")
                    sys.stdout.write("\033[0m\n")
                else:
                    sys.stdout.write(f"prompt> {prompt}\n")
                sys.stdout.flush()
            transcribe = internal.get("transcribe")
            if transcribe is not None:
                self._print_transcribe(transcribe)
            self._update_title(internal, subtask_name)
            return
        if "delta" in msg and "reasoning" in msg["delta"] and msg["delta"]["reasoning"]:
            if not self.reasoning:
                self.reasoning = True
                if self._color:
                    sys.stdout.write(f"\033[38;2;{self._color_reasoning}m")
                else:
                    sys.stdout.write("<think>")
            if not self._hide_reasoning:
                sys.stdout.write(f"{_strip_ansi(msg['delta']['reasoning'])}")
            else:
                self._reasoning_token_count += 1
                if self._reasoning_token_count % 10 == 0:
                    sys.stdout.write(".")
        elif "delta" in msg and "content" in msg["delta"] and msg["delta"]["content"]:
            if subtask_name is None:
                if self.reasoning:
                    if self._color:
                        sys.stdout.write("\033[0m\n")
                    else:
                        sys.stdout.write("</think>\n")
                    self.reasoning = False
                self._reasoning_token_count = 0
                sys.stdout.write(f"{_strip_ansi(msg['delta']['content'])}")
            else:
                if not self.reasoning:
                    self.reasoning = True
                    if self._color:
                        sys.stdout.write(f"\033[38;2;{self._color_reasoning}m")
                    else:
                        sys.stdout.write("<think>")
                if not self._hide_reasoning:
                    sys.stdout.write(f"{_strip_ansi(msg['delta']['content'])}")
                else:
                    self._reasoning_token_count += 1
                    if self._reasoning_token_count % 10 == 0:
                        sys.stdout.write(".")
        elif "prompt_progress" in msg and msg["prompt_progress"]:
            if self.reasoning:
                if self._color:
                    sys.stdout.write("\033[0m\n")
                else:
                    sys.stdout.write("</think>\n")
                self.reasoning = False
            self._reasoning_token_count = 0
            p = msg["prompt_progress"]
            print(
                f"Progress: {int(round(p['processed'] / p['total'] * 100))}% ({p['processed']}/{p['total']})"
            )
        else:
            if not self.reasoning:
                self.reasoning = True
                if self._color:
                    sys.stdout.write(f"\033[38;2;{self._color_reasoning}m")
                else:
                    sys.stdout.write("<think>")
            sys.stdout.write(".")
        sys.stdout.flush()


class StreamInput:
    def __init__(self, color_input: str, color: bool = True) -> None:
        self._color_input = _parse_hex_color(color_input)
        self._color = color
        if readline is not None:
            readline.set_completer(_chat_completer.complete)
            readline.set_completer_delims(" \t\n")
            readline.parse_and_bind("tab: complete")

    def __call__(self) -> dict[str, str] | None:
        while True:
            try:
                if self._color:
                    if readline is not None:
                        prompt = f"\n\001\033[1m\033[38;2;{self._color_input}m\002> "
                    else:
                        prompt = f"\n\033[1m\033[38;2;{self._color_input}m> "
                    iret = input(prompt)
                    sys.stdout.write("\033[0m")
                else:
                    iret = input("\n> ")

                raw = iret.strip()
                if not raw:
                    return {"message": "", "command": ""}

                if raw.startswith("/"):
                    parts = raw.split(maxsplit=1)
                    cmd = parts[0]
                    arg = parts[1] if len(parts) > 1 else ""
                    return {"message": arg, "command": cmd}

                return {"message": raw, "command": ""}
            except EOFError:
                return None
            except KeyboardInterrupt:
                if readline is not None and readline.get_line_buffer():
                    sys.stdout.write("\n")
                    continue
                raise


class ToolConfirm:
    def __init__(
        self,
        color_input: str,
        color: bool = True,
        color_reasoning: str = DEFAULT_COLOR_REASONING,
    ) -> None:
        self._color_input = _parse_hex_color(color_input)
        self._color = color
        self._color_reasoning = _parse_hex_color(color_reasoning)

    def __call__(
        self,
        call_details: dict[str, Any],
        format_confirmation: Callable[[dict[str, Any], bool], str | None],
        subtask_name: str | None = None,
    ) -> bool:
        args = call_details.get("arguments", {})
        tool_name = call_details.get("tool", "unknown")
        confirmation = format_confirmation(args, False)
        if confirmation is None:
            confirmation = str(args)
        if self._color:
            sys.stdout.write(
                f"\n\033[1m\033[38;2;{self._color_input}m{confirmation}\033[0m\n"
            )
        else:
            sys.stdout.write(f"{confirmation}\n")
        sys.stdout.flush()
        if subtask_name is not None:
            source = f"{subtask_name} ({tool_name})"
        else:
            source = tool_name
        prompt = (
            f"Execute \033[1m\033[38;2;{self._color_input}m{source}\033[0m? \033[1m\033[38;2;{self._color_input}mYES\033[0m/n: "
            if self._color
            else f"Execute {source}? YES/n: "
        )
        try:
            while True:
                response = input(prompt)
                stripped = response.strip()
                if stripped == "YES":
                    return True
                if stripped.lower() in ("n", "no"):
                    return False
        except (EOFError, KeyboardInterrupt):
            return False
        finally:
            if self._color:
                sys.stdout.write(f"\033[38;2;{self._color_reasoning}m")
                sys.stdout.flush()


def _setup_logging(verbose: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging_init(level)


def _build_sb_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    sb_parser = subparsers.add_parser("sb", help="Sandbox operations")

    sb_subparsers = sb_parser.add_subparsers(dest="sb_command")

    list_parser = sb_subparsers.add_parser("list", help="List sandboxes")
    list_parser.add_argument(
        "sandbox_name",
        nargs="?",
        default=None,
        help="Sandbox name (list all if omitted)",
    )

    ls_parser = sb_subparsers.add_parser("ls", help="List sandboxes")
    ls_parser.add_argument(
        "sandbox_name",
        nargs="?",
        default=None,
        help="Sandbox name (list all if omitted)",
    )

    containers_parser = sb_subparsers.add_parser(
        "containers", help="List containers for a sandbox"
    )
    containers_parser.add_argument(
        "sandbox_name",
        nargs="?",
        default=None,
        help="Sandbox name (containers for all sandboxes if omitted)",
    )

    cleanup_parser = sb_subparsers.add_parser(
        "cleanup", help="Cleanup sandbox resources"
    )
    cleanup_parser.add_argument(
        "--untracked-only",
        action="store_true",
        default=False,
        help="Only cleanup untracked containers",
    )
    cleanup_parser.add_argument(
        "--dead-only",
        action="store_true",
        default=False,
        help="Only cleanup dead entries",
    )

    kill_parser = sb_subparsers.add_parser("kill", help="Kill containers")
    kill_parser.add_argument(
        "sandbox_name",
        nargs="?",
        default=None,
        help="Sandbox name (kill all sandbox containers if omitted)",
    )
    kill_parser.add_argument(
        "--do-not-ask",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )

    rm_parser = sb_subparsers.add_parser("rm", help="Delete a sandbox")
    rm_parser.add_argument("sandbox_name", help="Sandbox name to delete")
    rm_parser.add_argument(
        "--do-not-ask",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )

    sync_parser = sb_subparsers.add_parser("sync", help="Sync a sandbox")
    sync_subparsers = sync_parser.add_subparsers(dest="sync_direction")
    from_parser = sync_subparsers.add_parser(
        "from-origin", help="Sync from origin to sandbox"
    )
    from_parser.add_argument("sandbox_name", help="Sandbox name")
    from_parser.add_argument(
        "--force", action="store_true", default=False, help="Force sync"
    )
    to_parser = sync_subparsers.add_parser(
        "to-origin", help="Sync to origin from sandbox"
    )
    to_parser.add_argument("sandbox_name", help="Sandbox name")
    to_parser.add_argument(
        "--force", action="store_true", default=False, help="Force sync"
    )

    build_parser = sb_subparsers.add_parser("build", help="Build container images")
    build_subparsers = build_parser.add_subparsers(dest="build_command")
    image_parser = build_subparsers.add_parser(
        "image", help="Build a single image with a given tag"
    )
    image_parser.add_argument("tag", help="Image tag (e.g. tiz-worker-js:latest)")
    build_subparsers.add_parser("all", help="Build all default containerfiles")

    rm_images_parser = sb_subparsers.add_parser(
        "rm-images", help="Remove all tiz-prefixed images"
    )
    rm_images_parser.add_argument(
        "--do-not-ask",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )

    logs_parser = sb_subparsers.add_parser("logs", help="Show container logs")
    logs_parser.add_argument("sandbox_name", help="Sandbox name")
    logs_parser.add_argument(
        "--container",
        default=None,
        help="Container name (show logs for all containers if omitted)",
    )
    logs_parser.add_argument(
        "--separate",
        action="store_true",
        default=False,
        help="Show stdout and stderr separately",
    )

    rm_all_parser = sb_subparsers.add_parser("rm-all", help="Remove all sandboxes")
    rm_all_parser.add_argument(
        "--do-not-ask",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )

    return cast(argparse.ArgumentParser, sb_parser)


def _confirm(do_not_ask: bool, msg: str) -> bool:
    if do_not_ask:
        return True
    response = input(f"{msg} [y/N] ")
    return response.lower() in ("y", "yes")


def _handle_sb(
    args: argparse.Namespace, config_dir: Path, sb_parser: argparse.ArgumentParser
) -> int:
    sb_cmd = args.sb_command

    try:
        if sb_cmd is None:
            sb_parser.print_help()
            return 1

        engine = SandboxManager.available_engine()
        if engine is None:
            print(
                "Error: no container engine (podman or docker) found",
                file=sys.stderr,
            )
            return 1
        manager = SandboxManager(base_path=config_dir, engine=engine)

        if sb_cmd in ("list", "ls"):
            names = manager.list_sandboxes()
            if args.sandbox_name:
                if args.sandbox_name not in names:
                    print(
                        f"Error: Sandbox '{args.sandbox_name}' not found",
                        file=sys.stderr,
                    )
                    return 1
                print(args.sandbox_name)
            else:
                for name in names:
                    print(name)
            return 0

        if sb_cmd == "containers":
            if args.sandbox_name:
                containers = manager.get_sandbox_containers(args.sandbox_name)
                for c in containers:
                    print(
                        f"{c.container_id or '?'}\t{c.container_name or '?'}\t{c.engine}\t"
                        f"{'running' if c.is_running() else 'stopped'}"
                    )
            else:
                all_containers = manager.list_containers()
                for entry in all_containers:
                    print(
                        f"{entry.get('sandbox_name', '?')}\t"
                        f"{entry.get('container_id', '?')}\t"
                        f"{entry.get('container_name', '?')}\t"
                        f"{entry.get('engine', '?')}"
                    )
            return 0

        if sb_cmd == "cleanup":
            removed_untracked: list[str] = []
            removed_dead: list[str] = []
            if not args.dead_only:
                removed_untracked = manager.cleanup_untracked_containers()
            if not args.untracked_only:
                removed_dead = manager.cleanup_dead_entries()
            for cid in removed_untracked:
                print(f"Removed untracked container: {cid}")
            for cid in removed_dead:
                print(f"Removed dead entry: {cid}")
            return 0

        if sb_cmd == "kill":
            if args.sandbox_name:
                if not _confirm(
                    args.do_not_ask,
                    f"Kill all containers for sandbox '{args.sandbox_name}'?",
                ):
                    return 0
                manager.kill_all_containers(args.sandbox_name)
                print(f"Killed containers for sandbox '{args.sandbox_name}'")
            else:
                names = manager.list_sandboxes()
                if not names:
                    print("Error: No sandboxes found", file=sys.stderr)
                    return 0
                if not _confirm(
                    args.do_not_ask,
                    "Kill all containers for ALL sandboxes?",
                ):
                    return 0
                for name in names:
                    manager.kill_all_containers(name)
                    print(f"Killed containers for sandbox '{name}'")
            return 0

        if sb_cmd == "rm":
            if not _confirm(
                args.do_not_ask,
                f"Delete sandbox '{args.sandbox_name}' (kills containers too)?",
            ):
                return 0
            manager.kill_and_delete_sandbox(args.sandbox_name)
            print(f"Deleted sandbox '{args.sandbox_name}'")
            return 0

        if sb_cmd == "sync":
            if args.sync_direction == "from-origin":
                manager.sync_from_original(args.sandbox_name, force=args.force)
                print(f"Synced sandbox '{args.sandbox_name}' from origin")
            elif args.sync_direction == "to-origin":
                manager.sync_to_original(args.sandbox_name, force=args.force)
                print(f"Synced sandbox '{args.sandbox_name}' to origin")
            else:
                print(
                    "Error: sync direction required (from-origin | to-origin)",
                    file=sys.stderr,
                )
                return 1
            return 0

        if sb_cmd == "build":
            containerfiles_dirs = SandboxManager.get_containerfiles_dirs(config_dir)

            def _find_containerfile(tag: str) -> Path | None:
                containerfile_name = f"Containerfile.{tag.split(':')[0]}"
                for cf_dir in containerfiles_dirs:
                    cf_path = cf_dir / containerfile_name
                    if cf_path.is_file():
                        return cf_path
                return None

            if args.build_command == "image":
                tag = args.tag
                if not tag.startswith("tiz-worker"):
                    print(
                        f"Error: tag must start with 'tiz-worker', got '{tag}'",
                        file=sys.stderr,
                    )
                    return 1
                cf_path = _find_containerfile(tag)
                if cf_path is None:
                    print(
                        f"Error: no default containerfile for tag '{tag}'",
                        file=sys.stderr,
                    )
                    return 1
                content = cf_path.read_text(encoding="utf-8")
                manager.build_image(
                    containerfile=content, tag=tag, delete_existing=True
                )
                print(f"Built image '{tag}'")
            elif args.build_command == "all":
                built = False
                for cf_dir in containerfiles_dirs:
                    if not cf_dir.is_dir():
                        continue
                    for cf in sorted(cf_dir.iterdir(), key=str):
                        if not cf.name.startswith("Containerfile."):
                            continue
                        tag = cf.name[len("Containerfile.") :] + ":latest"
                        content = cf.read_text(encoding="utf-8")
                        manager.build_image(
                            containerfile=content, tag=tag, delete_existing=True
                        )
                        print(f"Built image '{tag}'")
                        built = True
                if not built:
                    print(
                        "Error: no default containerfiles found",
                        file=sys.stderr,
                    )
                    return 1
            else:
                print(
                    "Error: build command required (image | all)",
                    file=sys.stderr,
                )
                return 1
            return 0

        if sb_cmd == "logs":
            containers = manager.get_sandbox_containers(args.sandbox_name)
            if args.container:
                containers = [
                    c for c in containers if c.container_name == args.container
                ]
                if not containers:
                    print(
                        f"Container '{args.container}' not found in sandbox '{args.sandbox_name}'",
                        file=sys.stderr,
                    )
                    return 1
            for c in containers:
                label = f"{c.container_name or c.container_id}"
                try:
                    logs = c.get_container_logs(separate=args.separate)
                except RuntimeError as e:
                    print(
                        f"Error getting logs for {label}: {e}",
                        file=sys.stderr,
                    )
                    continue
                if args.separate:
                    stdout, stderr_content = logs  # type: ignore[misc]
                    if len(containers) > 1:
                        print(f"=== {label} STDOUT ===")
                    print(stdout, end="")
                    if stderr_content:
                        if len(containers) > 1:
                            print(f"=== {label} STDERR ===")
                        print(stderr_content, end="")
                else:
                    if len(containers) > 1:
                        print(f"=== {label} ===")
                    print(logs, end="")
            return 0

        if sb_cmd == "rm-images":
            for image in manager.delete_tiz_worker_images(dry_run=True):
                print(image)
            if not _confirm(
                args.do_not_ask,
                "Remove all tiz-prefixed images?",
            ):
                return 0
            ret = manager.delete_tiz_worker_images()
            print("Removed all tiz-prefixed images:")
            for image in ret:
                print(image)
            return 0

        if sb_cmd == "rm-all":
            if not _confirm(
                args.do_not_ask,
                "Remove ALL sandboxes?",
            ):
                return 0
            manager.kill_and_delete_all_sandboxes()
            print("Removed all sandboxes")
            return 0

        return 0

    except Exception as exc:
        print(f"Error: {_format_exc(exc, args.verbose)}", file=sys.stderr)
        return 1


def _build_completion_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    completion_parser = subparsers.add_parser(
        "completion", help="Output shell autocompletion script"
    )
    completion_parser.add_argument(
        "shell",
        nargs="?",
        default="bash",
        choices=["bash", "zsh", "tcsh", "fish"],
        help="Shell type (default: bash)",
    )
    return cast(argparse.ArgumentParser, completion_parser)


def _build_stats_parser(subparsers: argparse._SubParsersAction) -> None:
    stats_parser = subparsers.add_parser("stats", help="Log statistics")
    stats_subparsers = stats_parser.add_subparsers(dest="stats_command")

    usage_parser = stats_subparsers.add_parser(
        "usage", help="Show usage per inference engine"
    )
    usage_parser.add_argument(
        "--period",
        choices=["daily", "weekly", "monthly", "yearly"],
        default="daily",
        help="Aggregation period (default: daily)",
    )
    usage_parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD or 'today', default: 30 days ago)",
    )
    usage_parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD or 'today', default: today)",
    )
    usage_parser.add_argument(
        "--engine",
        type=str,
        action="append",
        default=None,
        dest="engine_filter",
        help="Filter by inference engine (regexp, can be specified multiple times)",
    )
    usage_parser.add_argument(
        "--task",
        type=str,
        action="append",
        default=None,
        dest="task_filter",
        help="Filter by task name (regexp, can be specified multiple times)",
    )
    usage_parser.add_argument(
        "--group-by-task",
        action="store_true",
        default=False,
        help="Group output by task name instead of inference engine",
    )


def _parse_date_arg(s: str) -> date:
    if s.lower() == "today":
        return date.today()
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_date_from_filename(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def _load_usage_logs(log_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    usage_dir = log_dir / "usage"
    if not usage_dir.is_dir():
        return records
    for fpath in sorted(usage_dir.iterdir()):
        if fpath.suffix == ".log":
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                data["_file"] = fpath.name
                records.append(data)
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping malformed/unreadable log file: %s", fpath)
                continue
    return records


def _extract_engine_from_filename(filename: str) -> str:
    parts = _parse_log_filename(filename)
    return parts[0] if parts else "unknown"


def _extract_task_from_filename(filename: str) -> str:
    parts = _parse_log_filename(filename)
    return parts[1] if parts else "unknown"


_LOG_FILENAME_RE = re.compile(r"\d{8}_\d{6}_(.+?)_(.+?)\.log")


def _parse_log_filename(filename: str) -> tuple[str, str] | None:
    m = _LOG_FILENAME_RE.match(filename)
    if m:
        return (m.group(1), m.group(2))
    return None


def _handle_stats(args: argparse.Namespace, config_dir: Path) -> int:
    log_dir = config_dir / "logs"
    cmd = args.stats_command

    if cmd == "usage":
        return _handle_stats_usage(args, log_dir)

    print("Error: stats command required (usage)", file=sys.stderr)
    return 1


def _round_date(d: date, period: str) -> date:
    match period:
        case "daily":
            return d
        case "weekly":
            return d - timedelta(days=d.weekday())
        case "monthly":
            return d.replace(day=1)
        case "yearly":
            return d.replace(month=1, day=1)
        case _:
            return d


_USAGE_HEADER_COLUMNS: list[tuple[str, int, str]] = [
    ("Period", 14, "<"),
    ("Input", 12, ">"),
    ("Output", 12, ">"),
    ("Cached", 12, ">"),
    ("CWrite", 12, ">"),
    ("Cost", 14, ">"),
]
_USAGE_INDENT = 2
_USAGE_OUTER_SEP_WIDTH = 82
_USAGE_INNER_SEP_WIDTH = 76


def _handle_stats_usage(args: argparse.Namespace, log_dir: Path) -> int:
    to_date = _parse_date_arg(args.to_date) if args.to_date else date.today()
    from_date = (
        _parse_date_arg(args.from_date)
        if args.from_date
        else to_date - timedelta(days=30)
    )
    period = args.period

    records = _load_usage_logs(log_dir)
    if not records:
        print("No usage logs found.")
        return 0

    engine_filter: list[re.Pattern] | None = (
        [re.compile(p) for p in args.engine_filter] if args.engine_filter else None
    )
    task_filter: list[re.Pattern] | None = (
        [re.compile(p) for p in args.task_filter] if args.task_filter else None
    )
    group_by_task = args.group_by_task

    buckets: dict[str, dict[str, Any]] = {}
    for rec in records:
        log_date_str = rec.get("_file", "00000000_000000")[:8]
        try:
            log_date = _parse_date_from_filename(log_date_str)
        except ValueError:
            logger.warning(
                "Skipping malformed log file: %s", rec.get("_file", "unknown")
            )
            continue
        if log_date < from_date or log_date > to_date:
            continue
        engine = _extract_engine_from_filename(rec.get("_file", ""))
        task = _extract_task_from_filename(rec.get("_file", ""))

        if engine_filter is not None and not any(
            p.fullmatch(engine) for p in engine_filter
        ):
            continue
        if task_filter is not None and not any(p.fullmatch(task) for p in task_filter):
            continue

        bucket = _round_date(log_date, period)

        group_key: str = task if group_by_task else engine

        if group_key not in buckets:
            buckets[group_key] = {}
        key = bucket.isoformat()
        if key not in buckets[group_key]:
            buckets[group_key][key] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0.0,
            }
        b = buckets[group_key][key]
        b["prompt_tokens"] += rec.get("prompt_tokens", 0) or 0
        b["completion_tokens"] += rec.get("completion_tokens", 0) or 0
        b["cached_tokens"] += rec.get("cached_tokens", 0) or 0
        b["cache_write_tokens"] += rec.get("cache_write_tokens", 0) or 0
        b["cost"] += rec.get("cost", 0) or 0.0

    if not buckets:
        print("No usage data found.")
        return 0

    for group_key in sorted(buckets.keys()):
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        total_cache_write = 0
        total_cost = 0.0
        group_buckets = buckets[group_key]
        label = f" [{group_key}]"
        print(f"{label}")
        print(f"{'':-<{_USAGE_OUTER_SEP_WIDTH}}")
        header_parts = []
        for col_name, col_width, col_align in _USAGE_HEADER_COLUMNS:
            header_parts.append(f"{col_name:{col_align}{col_width}}")
        print(f"{' ' * _USAGE_INDENT}{' '.join(header_parts)}")
        print(f"{'': >{_USAGE_INDENT}}{'':-<{_USAGE_INNER_SEP_WIDTH}}")
        for period_key in sorted(group_buckets.keys()):
            b = group_buckets[period_key]
            total_prompt += b["prompt_tokens"]
            total_completion += b["completion_tokens"]
            total_cached += b["cached_tokens"]
            total_cache_write += b["cache_write_tokens"]
            total_cost += b["cost"]
            print(
                f"  {period_key:<14} {b['prompt_tokens'] - b['cached_tokens']:>12,} {b['completion_tokens']:>12,}"
                f" {b['cached_tokens']:>12,} {b['cache_write_tokens']:>12,}"
                f" {b['cost']:>13.10f}$"
            )
        print(f"{'': >{_USAGE_INDENT}}{'':-<{_USAGE_INNER_SEP_WIDTH}}")
        print(
            f"  {'TOTAL':<14} {total_prompt - total_cached:>12,} {total_completion:>12,}"
            f" {total_cached:>12,} {total_cache_write:>12,}"
            f" {total_cost:>13.10f}$"
        )
        print()

    return 0


def _parse_context(context_args: list[str]) -> dict[str, str]:
    ctx: dict[str, str] = {}
    for arg in context_args:
        if "=" not in arg:
            raise ValueError(f"invalid context format '{arg}', expected key=value")
        key, value = arg.split("=", 1)
        ctx[key] = value
    return ctx


def _print_usage(
    result: dict[str, dict[str, Any]],
    running_time: str | None = None,
) -> None:
    prompt_tokens = 0
    prompt_time: float = 0
    completion_tokens = 0
    completion_time: float = 0
    cached_tokens = 0
    cache_write_tokens = 0
    cost: float = 0
    tool_calls: list[Any] = []
    for item in result.values():
        prompt_tokens += item.get("prompt_tokens", 0)
        prompt_time += item.get("prompt_time", 0)
        completion_tokens += item.get("completion_tokens", 0)
        completion_time += item.get("completion_time", 0)
        cached_tokens += item.get("cached_tokens", 0)
        cache_write_tokens += item.get("cache_write_tokens", 0)
        cost += item.get("cost", 0)
        if "tool_calls" in item:
            tool_calls.extend(item["tool_calls"])
    print("\n")
    prompt_tokens -= cached_tokens
    if running_time is not None:
        print(f"Running time:\t{running_time}")
    input_rate = round(prompt_tokens / prompt_time, 2) if prompt_time else 0.0
    output_rate = (
        round(completion_tokens / completion_time, 2) if completion_time else 0.0
    )
    print(f"Usage: input:\t{prompt_tokens} ({input_rate} tk/s)")
    print(f"      output:\t{completion_tokens} ({output_rate} tk/s)")
    print(f"      cached:\t{cached_tokens}")
    print(f"      cwrite:\t{cache_write_tokens}")
    print(f"Credits spent:\t{cost:.10f} $")
    print("Tools usage:")
    for name, count in sorted(
        collections.Counter(x[0] for x in tool_calls).items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {name}: {count}")


def _handle_chat(
    manifest: Any,
    base_path: Path,
    task_name: str | None = None,
    context: dict[str, str] | None = None,
    manifest_names: list[str] | None = None,
) -> int:
    update = StreamUpdater(
        hide_reasoning=manifest.meta.hide_reasoning,
        color_reasoning=manifest.meta.color_reasoning,
        color_input=manifest.meta.color_input,
        color=manifest.meta.color,
        command="chat",
        manifests=manifest_names,
        parallelism=manifest.meta.parallelism or 1,
    )
    _set_terminal_title("tiz: chat")
    inp = StreamInput(manifest.meta.color_input, color=manifest.meta.color)
    confirm = ToolConfirm(
        manifest.meta.color_input,
        color=manifest.meta.color,
        color_reasoning=manifest.meta.color_reasoning,
    )
    chat_result, err = helpers_chat(
        manifest=manifest,
        base_path=base_path,
        task_name=task_name,
        update_callback=update,
        input_callback=inp,
        context=context,
        confirm_callback=confirm,
        enable_recording=True,
    )
    sys.stdout.write("\033[0m\n" if manifest.meta.color else "\n")
    if err:
        print(f"Error: {err}", file=sys.stderr)
    _print_usage(chat_result)
    _set_terminal_title("tiz")
    return 1 if err else 0


def _handle_exec(
    manifest: Any,
    base_path: Path,
    task_name: str | None,
    cmd_args: list[str],
    extra_run_args: list[str] | None = None,
) -> int:
    err = exec_cmd(
        manifest=manifest,
        base_path=base_path,
        task_name=task_name,
        cmd_args=cmd_args,
        extra_run_args=extra_run_args,
    )
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


def _build_credits_parser(subparsers: argparse._SubParsersAction) -> None:
    credits_parser = subparsers.add_parser(
        "credits", help="Show credits for all inference engines"
    )
    credits_parser.add_argument(
        "-m",
        "--manifest",
        type=Path,
        action="append",
        default=[],
        required=True,
        help="Manifest file (can be specified multiple times)",
    )


def _handle_credits(manifest: Any) -> int:
    engines: list[InferenceEngineSpec] = manifest.inference_engines
    if not engines:
        print("No inference engines found in manifest(s).", file=sys.stderr)
        return 1

    credits_data = get_credits(engines)
    for entry in credits_data:
        print(
            f"[{entry['name']}]  Total credits: ${entry['total_credits']:.4f}  "
            f"Total usage: ${entry['total_usage']:.4f}  "
            f"Remaining: ${entry['remaining']:.4f}"
        )
    return 0


def _build_web_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    web_parser = subparsers.add_parser("web", help="Run the web API server")
    web_parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the WebConfig YAML file",
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )
    web_parser.add_argument(
        "--socket-path",
        type=Path,
        default=None,
        help="Unix socket path (overrides host/port)",
    )
    return cast(argparse.ArgumentParser, web_parser)


def _handle_web(args: argparse.Namespace, base_path: Path) -> int:
    web_run_simple(
        base_path=base_path,
        config_path=args.config_path,
        host=args.host,
        port=args.port,
        path=args.socket_path,
    )
    return 0


def get_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the tiz CLI."""
    parser = argparse.ArgumentParser(
        prog="tiz", description="tiz - an agentic AI bot and much more"
    )
    parser.add_argument(
        "-c",
        "--config-dir",
        type=Path,
        default=Path.home() / ".tiz",
        help="Base path for configuration",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity level (can be used up to twice)",
    )
    parser.add_argument(
        "--hide-reasoning",
        action="store_true",
        default=None,
        help="Hide reasoning output",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=None,
        help="Maximum parallel tasks",
    )
    parser.add_argument(
        "--container-engine",
        type=str,
        default=None,
        help="Container engine to use (e.g. docker, podman)",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        default=None,
        help="Enable colored output",
    )
    parser.add_argument(
        "--no-color",
        action="store_false",
        dest="color",
        default=None,
        help="Disable colored output",
    )
    parser.add_argument(
        "--use-host-timezone",
        action="store_true",
        default=None,
        help="Use host timezone for containers",
    )
    parser.add_argument(
        "--no-use-host-timezone",
        action="store_false",
        dest="use_host_timezone",
        default=None,
        help="Do not use host timezone for containers",
    )
    parser.add_argument(
        "--color-input",
        type=str,
        default=None,
        help="Hex color for input (e.g. #ff00ff)",
    )
    parser.add_argument(
        "--color-reasoning",
        type=str,
        default=None,
        help="Hex color for reasoning (e.g. #758182)",
    )
    parser.add_argument(
        "--summarizer-context-ratio",
        type=float,
        default=None,
        help="Summarizer context ratio (default: 0.9)",
    )
    parser.add_argument(
        "--committer-name",
        type=str,
        default=None,
        help="Committer name for git operations",
    )
    parser.add_argument(
        "--committer-email",
        type=str,
        default=None,
        help="Committer email for git operations",
    )
    parser.add_argument(
        "--save-full-logs",
        action="store_true",
        default=None,
        help="Save full logs",
    )
    parser.add_argument(
        "--no-save-full-logs",
        action="store_false",
        dest="save_full_logs",
        default=None,
        help="Do not save full logs",
    )
    parser.add_argument(
        "--save-full-toolcalls",
        action="store_true",
        default=None,
        help="Save full tool calls",
    )
    parser.add_argument(
        "--no-save-full-toolcalls",
        action="store_false",
        dest="save_full_toolcalls",
        default=None,
        help="Do not save full tool calls",
    )
    parser.add_argument(
        "--save-full-usage-details",
        action="store_true",
        default=None,
        help="Save full usage details",
    )
    parser.add_argument(
        "--no-save-full-usage-details",
        action="store_false",
        dest="save_full_usage_details",
        default=None,
        help="Do not save full usage details",
    )
    parser.add_argument(
        "--ring-bell",
        action="store_true",
        default=None,
        help="Ring terminal bell on completion",
    )
    parser.add_argument(
        "--no-ring-bell",
        action="store_false",
        dest="ring_bell",
        default=None,
        help="Do not ring terminal bell on completion",
    )
    parser.add_argument(
        "--delete-sandbox-on-exit",
        action="store_true",
        default=None,
        help="Delete sandbox container on exit",
    )
    parser.add_argument(
        "--no-delete-sandbox-on-exit",
        action="store_false",
        dest="delete_sandbox_on_exit",
        default=None,
        help="Do not delete sandbox container on exit",
    )
    parser.add_argument(
        "--ephemeral-sandbox",
        action="store_true",
        default=None,
        help="Use ephemeral sandbox (implies --delete-sandbox-on-exit)",
    )
    parser.add_argument(
        "--no-ephemeral-sandbox",
        action="store_false",
        dest="ephemeral_sandbox",
        default=None,
        help="Do not use ephemeral sandbox",
    )
    subparsers = parser.add_subparsers(dest="command")
    chat_parser = subparsers.add_parser(
        "chat", help="Start an interactive chat session"
    )
    chat_parser.add_argument(
        "-m",
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="Manifest file (can be specified multiple times)",
    )
    chat_parser.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="Task name for the chat (default: first task)",
    )
    chat_parser.add_argument(
        "--context",
        type=str,
        action="append",
        default=[],
        help="Context key=value pairs (can be specified multiple times)",
    )

    run_parser = subparsers.add_parser("run", help="Run Tiz")
    run_parser.add_argument(
        "-m",
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="Manifest file (can be specified multiple times)",
    )
    run_parser.add_argument(
        "--context",
        type=str,
        action="append",
        default=[],
        help="Context key=value pairs (can be specified multiple times)",
    )
    exec_parser = subparsers.add_parser("exec", help="Exec into a sandbox container")
    exec_parser.add_argument(
        "-m",
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="Manifest file (can be specified multiple times)",
    )
    exec_parser.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="Task name to exec into (default: first task)",
    )
    exec_parser.add_argument(
        "--extra-run-args",
        type=str,
        action="append",
        default=None,
        dest="extra_run_args",
        help="Extra arguments to pass to container run command (can be specified multiple times)",
    )
    exec_parser.add_argument(
        "cmd_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Command to run inside the container (default: /bin/bash)",
    )
    parser._sb_parser = _build_sb_parser(subparsers)  # type: ignore[attr-defined]
    _build_stats_parser(subparsers)
    _build_completion_parser(subparsers)
    _build_credits_parser(subparsers)
    _build_web_parser(subparsers)
    return parser


def main() -> int:
    parser = get_parser()
    sb_parser: argparse.ArgumentParser = cast(
        argparse.ArgumentParser,
        parser._sb_parser,  # type: ignore[attr-defined]
    )

    autocomplete(parser)

    args = parser.parse_args()

    if args.color is None:
        args.color = _is_tty() and "NO_COLOR" not in os.environ

    _setup_logging(args.verbose)

    if args.command == "sb":
        return _handle_sb(args, args.config_dir, sb_parser)

    if args.command == "stats":
        return _handle_stats(args, args.config_dir)

    if args.command == "completion":
        print(shellcode(shell=args.shell))
        return 0

    if args.command == "web":
        return _handle_web(args, args.config_dir)

    if args.command not in ("exec", "run", "chat", "credits"):
        parser.print_help()
        return 1

    if not args.manifest:
        parser.error("at least one manifest is required (--manifest)")

    default_options: dict[str, Any] = {}
    default_options["meta"] = {
        "version": "0",
        "parallelism": 1,
        "committer_name": TIZ_COMMIT_AUTHOR_NAME,
        "committer_email": TIZ_COMMIT_AUTHOR_EMAIL,
        "color": True,
        "color_reasoning": DEFAULT_COLOR_REASONING,
        "color_input": DEFAULT_COLOR_INPUT,
        "hide_reasoning": False,
        "use_host_timezone": True,
        "save_full_logs": False,
        "save_full_toolcalls": True,
        "save_full_usage_details": True,
        "summarizer_context_ratio": 0.9,
        "verbosity": 0,
        "ring_bell": False,
        "delete_sandbox_on_exit": False,
        "ephemeral_sandbox": False,
    }

    options: dict[str, Any] = {}
    if args.verbose in (1, 2):
        options["verbosity"] = args.verbose
    elif args.verbose > 2:
        options["verbosity"] = 2

    for opt_key in (
        "hide_reasoning",
        "parallelism",
        "container_engine",
        "color",
        "color_input",
        "color_reasoning",
        "summarizer_context_ratio",
        "committer_name",
        "committer_email",
        "save_full_logs",
        "save_full_toolcalls",
        "save_full_usage_details",
        "use_host_timezone",
        "ring_bell",
        "delete_sandbox_on_exit",
        "ephemeral_sandbox",
    ):
        val = getattr(args, opt_key, None)
        if val is not None:
            options[opt_key] = val

    manifest, err = parse_manifest(
        base_path=args.config_dir,
        manifests=args.manifest,
        default_options=default_options,
        options={"meta": options},
    )
    if err or manifest is None:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # These fields have defaults applied in parse_manifest, so they're never None here.
    # Assert to satisfy mypy about the narrowed types.
    assert manifest.meta.hide_reasoning is not None
    assert manifest.meta.color_reasoning is not None
    assert manifest.meta.color_input is not None
    assert manifest.meta.color is not None

    if (manifest.meta.parallelism or 1) > 1 and ask_confirmations(manifest):
        manifest.meta.parallelism = 1
        logger.warning(
            "Parallelism forced to 1 because some tools have confirmations, "
            "which are incompatible with parallel execution."
        )

    if manifest.meta.verbosity != args.verbose:
        _setup_logging(manifest.meta.verbosity or 0)

    try:
        context = (
            _parse_context(args.context) if args.command in ("chat", "run") else {}
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    manifest_names = [p.stem for p in args.manifest]

    if args.command == "credits":
        return _handle_credits(manifest)

    if args.command == "chat":
        return _handle_chat(
            manifest, args.config_dir, args.task, context, manifest_names
        )

    if args.command == "exec":
        # extra_run_args are appended as individual strings; shell-split and flatten
        xargs = args.extra_run_args
        if xargs is not None:
            xargs = [shlex.split(a) for a in xargs]
            xargs = [item for sublist in xargs for item in sublist]
        rc = _handle_exec(
            manifest,
            args.config_dir,
            args.task,
            args.cmd_args,
            extra_run_args=xargs,
        )
        _maybe_ring_bell(manifest.meta)
        return rc

    update = StreamUpdater(
        hide_reasoning=manifest.meta.hide_reasoning,
        color_reasoning=manifest.meta.color_reasoning,
        color_input=manifest.meta.color_input,
        color=manifest.meta.color,
        command="run",
        manifests=manifest_names,
        parallelism=manifest.meta.parallelism or 1,
    )
    confirm = ToolConfirm(
        manifest.meta.color_input,
        color=manifest.meta.color,
        color_reasoning=manifest.meta.color_reasoning,
    )

    _set_terminal_title("tiz: run")

    r: dict[str, dict[str, Any]] = {}
    start = time.monotonic()
    try:
        r, err = run(
            manifest=manifest,
            base_path=args.config_dir,
            update_callback=update,
            confirm_callback=confirm,
            context=context,
        )
    except KeyboardInterrupt:
        err = "interrupted"
    finally:
        end = time.monotonic()
        sys.stdout.write("\033[0m\n" if manifest.meta.color else "\n")
        if err:
            print(f"Error: {err}", file=sys.stderr)
        _print_usage(r, running_time=f"{timedelta(seconds=end - start)}")
        _set_terminal_title("tiz")
        _maybe_ring_bell(manifest.meta)
    return 1 if err else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
