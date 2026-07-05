from __future__ import annotations

import base64
import contextlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tiz.audio_inference_clients import AudioInferenceClient, WhisperCpp
from tiz.base_task_executor import BaseTaskExecutor
from tiz.chat import Chat
from tiz.log import get_logger
from tiz.manifest_parser import AudioInferenceEngineSpec, Manifest, TaskSpec
from tiz.recorder import record_audio

logger = get_logger(__name__)


class InteractiveChat(BaseTaskExecutor):
    def __init__(
        self,
        *,
        manifest: Manifest,
        base_path: Path = Path(),
        task_name: str | None = None,
        update_callback: Callable[[dict[str, Any], str | None], None] | None = None,
        input_callback: Callable[[], dict[str, str] | None] | None = None,
        context: dict[str, Any] | None = None,
        confirm_callback: Callable[
            [dict[str, Any], Callable[[dict[str, Any], bool], str | None], str | None],
            bool,
        ]
        | None = None,
        enable_recording: bool = False,
        required_prefix: str | None = None,
        in_band_files: bool = False,
        enable_help: bool = True,
        no_exit_on_kbdint: bool = False,
    ) -> None:
        super().__init__(
            manifest=manifest,
            base_path=base_path,
            update_callback=update_callback,
            context=context,
            confirm_callback=confirm_callback,
        )
        self.input_callback = input_callback
        self.enable_recording = enable_recording
        self.required_prefix = (required_prefix or "").strip() or None
        self.in_band_files = in_band_files
        self.enable_help = enable_help
        self.no_exit_on_kbdint = no_exit_on_kbdint

        if not manifest.tasks:
            raise ValueError("No tasks found in manifest")

        if task_name is not None:
            task = next(
                (t for t in manifest.tasks if t.name == task_name),
                None,
            )
            if task is None:
                raise ValueError(f"Task '{task_name}' not found in manifest")
        else:
            task = manifest.tasks[0]
        self.task = task
        self.client = self._get_client_for_task(self.task)
        self.audio_inference_client = self._get_audio_client_for_task(self.task)

        resources = self._create_task_resources(
            task=self.task,
            create_conversion_container=True,
        )
        self._tool_instances = resources.tool_instances
        self._tools_confirmations = resources.tools_confirmations
        self._sandbox = resources.sandbox
        self._manager = resources.manager
        self._sandbox_name = resources.sandbox_name
        action_lock = resources.action_lock
        self._conversion_sandbox = resources.conversion_sandbox
        self._action_lock: contextlib.AbstractContextManager[None] = (
            action_lock if action_lock is not None else contextlib.nullcontext()
        )

        try:
            self.chat = self._create_chat()
        except:
            self._cleanup_resources(self._manager, self._sandbox_name, self._sandbox)
            raise

    def _create_chat(self) -> Chat:
        sys_prompt = self.resolve_prompt(self.task, self._sandbox)
        return Chat(
            client=self.client,
            sys_prompt=sys_prompt,
            tools=self._tool_instances or None,
            update_callback=self._update_callback,
            confirm_callback=self._confirm_callback,
            tools_confirmations=self._tools_confirmations or None,
            conversion_sandbox=self._conversion_sandbox,
            audio_inference_client=self.audio_inference_client,
            ctx_ratio=self.manifest.meta.summarizer_context_ratio or 0.9,
        )

    @staticmethod
    def _build_audio_client(spec: AudioInferenceEngineSpec) -> AudioInferenceClient:
        if spec.engine_type in ("whispercpp", "whisper.cpp"):
            return WhisperCpp(
                host=spec.host,
                timeout=spec.timeout,
                inference_timeout=spec.inference_timeout,
                verify_ssl=spec.verify_ssl,
                ca_cert=spec.ca_cert,
                sampling_params=spec.sampling_params,
                language=spec.language,
                prompt=spec.prompt,
            )
        raise ValueError(f"Unknown audio inference engine type: {spec.engine_type}")

    def _get_audio_client_for_task(self, task: TaskSpec) -> AudioInferenceClient | None:
        if not task.dedicated_audio_engine:
            return None
        for e in self.manifest.audio_inference_engines:
            if e.name == task.dedicated_audio_engine:
                return self._build_audio_client(e)
        raise ValueError(
            f"Audio engine '{task.dedicated_audio_engine}' referenced by task "
            f"'{task.name}' not found in manifest"
        )

    BASE_COMMANDS: tuple[tuple[str, str], ...] = (
        ("/help", "show this help message"),
        ("/quit", "exit the interactive session"),
        ("/exit", "exit the interactive session"),
        ("/clear", "start a new conversation (discard history)"),
        ("/replay", "replay the last assistant response"),
        ("/usage", "show current usage"),
        ("/attach <file>", "attach a file to the conversation"),
        ("/load <file>", "load conversation history from a JSON file"),
        ("/save <file>", "save conversation history to a JSON file"),
        ("/sync-to", "sync sandbox changes to the original project"),
        ("/sync-from", "sync sandbox changes from the original project"),
    )

    @property
    def commands(self) -> dict[str, str]:
        cmds = dict(self.BASE_COMMANDS)
        if self.enable_recording:
            cmds["/record"] = "record audio from microphone and attach it"
        return cmds

    def _show_help(self) -> None:
        if not self.enable_help:
            return
        self._feedback("Available commands:")
        for cmd, desc in self.commands.items():
            self._feedback(f"  {cmd:<20} {desc}")

    def _show_usage(self, accumulated: bool = False) -> None:
        if self._update_callback is not None:
            self._update_callback(
                {
                    "tiz-internal": {
                        "interactive_chat_usage": self.task_usage
                        if accumulated
                        else self.chat.usage,
                        "interactive_chat_usage_accumulated": accumulated,
                    }
                },
                None,
            )

    def _usage_hint(self, text: str) -> None:
        if self.enable_help:
            self._feedback(text)

    def _feedback(self, text: str) -> None:
        if self._update_callback is not None:
            self._update_callback(
                {"tiz-internal": {"interactive_chat_feedback": text}},
                None,
            )

    def _handle_replay(self, _raw: dict[str, str]) -> bool:
        try:
            result = self._send_with_retry(self.chat, "", use_replay=True)
            self._accumulate_usage(self.task.name, result)
            self._show_usage()
        except KeyboardInterrupt:
            self._feedback("")
        return True

    def _handle_clear(self, _raw: dict[str, str]) -> bool:
        self.chat = self._create_chat()
        self._feedback("Conversation cleared.")
        return True

    def _handle_usage(self, _raw: dict[str, str]) -> bool:
        self._show_usage(accumulated=True)
        return True

    def _handle_attach(self, raw: dict[str, str]) -> bool:
        arg = raw.get("message", "")
        if not arg:
            self._usage_hint("Usage: /attach <file>")
            return True
        try:
            if self.in_band_files:
                contents = base64.b64decode(raw.get("contents", ""))
                self.chat.append_file(arg, data=contents)
            else:
                self.chat.append_file(arg)
            self._feedback(f"Attached: {arg}")
        except Exception as e:
            logger.info("Error in /attach: %s", e)
            self._feedback(f"Error: {e}")
        return True

    def _handle_load(self, raw: dict[str, str]) -> bool:
        arg = raw.get("message", "")
        if not arg:
            self._usage_hint("Usage: /load <file>")
            return True
        try:
            if self.in_band_files:
                data = base64.b64decode(raw.get("contents", "")).decode()
                self.chat.load(data=data)
            else:
                self.chat.load(arg)
            self._feedback(f"Loaded conversation from: {arg}")
        except Exception as e:
            logger.info("Error in /load: %s", e)
            self._feedback(f"Error: {e}")
        return True

    def _handle_save(self, raw: dict[str, str]) -> bool:
        arg = raw.get("message", "")
        if not arg:
            self._usage_hint("Usage: /save <file>")
            return True
        try:
            if self.in_band_files:
                saved = self.chat.save()
                if "contents" in raw:
                    saved = base64.b64encode(saved.encode()).decode()
                    if self._update_callback is not None:
                        self._update_callback(
                            {"tiz-internal": {"save_conv": saved}},
                            None,
                        )
                else:
                    Path(arg).write_text(saved)
                    self._feedback(f"Saved conversation to: {arg}")
            else:
                self.chat.save(arg)
                self._feedback(f"Saved conversation to: {arg}")
        except Exception as e:
            logger.info("Error in /save: %s", e)
            self._feedback(f"Error: {e}")
        return True

    def _handle_sync_to(self, _raw: dict[str, str]) -> bool:
        if self._manager is not None and self._sandbox_name is not None:
            self._manager.sync_to_original(self._sandbox_name)
            self._feedback("Synced sandbox changes to original project.")
        else:
            self._feedback("No sandbox to sync from.")
        return True

    def _handle_sync_from(self, _raw: dict[str, str]) -> bool:
        if self._manager is not None and self._sandbox_name is not None:
            self._manager.sync_from_original(self._sandbox_name)
            self._feedback("Synced original project changes to sandbox.")
        else:
            self._feedback("No sandbox to sync to.")
        return True

    def _handle_quit(self, _raw: dict[str, str]) -> bool:
        return False

    def _handle_help(self, _raw: dict[str, str]) -> bool:
        self._show_help()
        return True

    def _handle_record(self, _raw: dict[str, str]) -> bool:
        if not self.enable_recording:
            self._feedback("Unknown command: /record")
            return True
        filepath: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                filepath = f.name
            record_audio(filepath)
            self.chat.append_file(filepath)
            self._feedback("Attached recording")
            try:
                result = self._send_with_retry(self.chat, "", use_replay=True)
                self._accumulate_usage(self.task.name, result)
                self._show_usage()
            except KeyboardInterrupt:
                self._feedback("")
                return True
        except Exception as e:
            logger.info("Error in /record: %s", e)
            self._feedback(f"Error: {e}")
        finally:
            if filepath is not None:
                with contextlib.suppress(OSError):
                    Path(filepath).unlink()
        return True

    def run(self) -> None:
        self._show_help()
        self._feedback("")

        dispatch: dict[str, Callable[[dict[str, str]], bool]] = {
            "/help": self._handle_help,
            "/quit": self._handle_quit,
            "/exit": self._handle_quit,
            "/clear": self._handle_clear,
            "/replay": self._handle_replay,
            "/usage": self._handle_usage,
            "/attach": self._handle_attach,
            "/load": self._handle_load,
            "/save": self._handle_save,
            "/sync-to": self._handle_sync_to,
            "/sync-from": self._handle_sync_from,
            "/record": self._handle_record,
        }

        try:
            with self._action_lock:
                while True:
                    try:
                        raw = (
                            self.input_callback()
                            if self.input_callback is not None
                            else None
                        )
                        if raw is None:
                            break
                    except KeyboardInterrupt:
                        self._feedback("")
                        if self.no_exit_on_kbdint:
                            continue
                        break
                    cmd = raw.get("command", "")
                    arg = raw.get("message", "")

                    if not cmd and not arg:
                        continue

                    if cmd.startswith("/"):
                        handler = dispatch.get(cmd)
                        if handler is not None:
                            should_continue = handler(raw)
                            if not should_continue:
                                break
                        else:
                            self._feedback(f"Unknown command: {cmd}")
                    else:
                        if self.required_prefix:
                            if not arg.lower().startswith(self.required_prefix.lower()):
                                logger.info(
                                    "Input ignored: does not start with "
                                    "required prefix '%s'",
                                    self.required_prefix,
                                )
                                continue
                            arg = arg[len(self.required_prefix) :].strip()
                        if not arg:
                            continue
                        try:
                            result = self._send_with_retry(self.chat, arg)
                            self._accumulate_usage(self.task.name, result)
                            self._show_usage()
                        except KeyboardInterrupt:
                            self._feedback("")
        finally:
            try:
                if self._sandbox:
                    self._sandbox.sync_to_original_auto_rebase()
            except Exception:
                logger.exception("Failed to sync sandbox to original")
            self._cleanup_resources(self._manager, self._sandbox_name, self._sandbox)
            try:
                self._save_chat_log(self.chat, self.task.name)
            except Exception:
                logger.exception("Failed to save chat log")
            try:
                self._save_toolcalls_log(self.task.name)
            except Exception:
                logger.exception("Failed to save tool calls log")
            try:
                self._save_full_usage()
            except Exception:
                logger.exception("Failed to save full usage")
