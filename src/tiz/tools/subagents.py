"""SubAgents tool for delegating tasks to sub-agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from tiz.tools.base import Tool

logger = logging.getLogger(__name__)


class SubAgents(Tool):
    """Tool to delegate tasks to named sub-agents.

    Each sub-agent is created by a factory function that returns a Chat instance.
    A new Chat instance is created for every invocation, ensuring isolation.
    The LLM can invoke a sub-agent by name with a prompt for that sub-agent.
    """

    def __init__(
        self,
        subagents: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__()
        self._subagents: dict[str, dict[str, Any]] = {}
        for name, config in subagents.items():
            if not isinstance(config, dict):
                logger.warning("Skipping sub-agent '%s': config is not a dict", name)
                continue
            chat_factory = config.get("chat")
            raw_desc = config.get("description", "")
            description = raw_desc if isinstance(raw_desc, str) else ""
            sub_callback = config.get("usage_callback")
            if callable(chat_factory):
                entry: dict[str, Any] = {
                    "chat_factory": chat_factory,
                    "description": description,
                }
                if callable(sub_callback):
                    entry["usage_callback"] = sub_callback
                elif sub_callback is not None:
                    logger.warning(
                        "Sub-agent '%s': 'usage_callback' is not callable, ignoring",
                        name,
                    )
                self._subagents[name] = entry
            else:
                logger.warning("Skipping sub-agent '%s': 'chat' is not callable", name)

    def prompt(self) -> str:
        subagent_lines: list[str] = []
        for name, cfg in sorted(self._subagents.items()):
            desc = cfg.get("description", "")
            if desc:
                subagent_lines.append(f"- `{name}`: {desc}")
            else:
                subagent_lines.append(f"- `{name}`")

        if subagent_lines:
            available_str = "Available sub-agents:\n" + "\n".join(subagent_lines)
        else:
            available_str = "No sub-agents are currently available."

        return json.dumps(
            {
                "name": self.fname(),
                "description": (
                    f"Delegate a task to a named sub-agent. Each sub-agent "
                    f"has its own capabilities and context. Use this when you "
                    f"need to perform a distinct task that is better handled "
                    f"by a specialized sub-agent.\n\n{available_str}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the sub-agent to invoke. See the tool description for available sub-agents and their capabilities.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The prompt or task to send to the sub-agent.",
                        },
                    },
                    "required": ["name", "prompt"],
                    "additionalProperties": False,
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "SubAgents"

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        name = args.get("name") or ""
        prompt = (args.get("prompt") or "")[:80]
        if markdown:
            safe_name = name.replace("`", "")
            safe_prompt = prompt.replace("`", "")
            return f"Invoke sub-agent `{safe_name}` with: `{safe_prompt}`"
        return f"Invoke sub-agent {name} with: '{prompt}'"

    def run(self, args: dict[str, Any]) -> str:
        name = args.get("name", "")
        prompt = args.get("prompt", "")

        if not name or not isinstance(name, str):
            return f"ERROR: {self.fname()} takes a mandatory 'name' argument!"
        if not prompt or not isinstance(prompt, str):
            return f"ERROR: {self.fname()} takes a mandatory 'prompt' argument!"

        if name not in self._subagents:
            available = ", ".join(sorted(self._subagents.keys()))
            if available:
                return f"ERROR: unknown sub-agent '{name}'. Available: {available}"
            return f"ERROR: unknown sub-agent '{name}'. No sub-agents available."

        subagent = self._subagents[name]
        chat_factory = subagent["chat_factory"]
        try:
            chat = chat_factory()
            if not hasattr(chat, "send_message"):
                logger.error(
                    "chat_factory for '%s' did not return a Chat-like object",
                    name,
                )
                return "ERROR: sub-agent call failed"
            result = chat.send_message(prompt)
        except Exception:
            logger.exception("sub-agent call failed for '%s'", name)
            return "ERROR: sub-agent call failed"
        callback = subagent.get("usage_callback")
        if callback is not None:
            try:
                callback(chat.usage, chat.conv)
            except Exception:
                logger.exception("usage_callback failed for '%s'", name)
        msg = result.get("message")
        return str(msg) if msg is not None else ""
