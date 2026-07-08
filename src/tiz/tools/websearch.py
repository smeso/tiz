"""Web search tool using DuckDuckGo."""

import json
from typing import Any

from tiz.sandbox_worker import DEFAULT_USER_AGENT
from tiz.tools.base import SocketTool

DEFAULT_MAX_RESULTS = 10
MAX_RESULTS_LIMIT = 50
DEFAULT_TIMEOUT = 10
TIMEOUT_MIN = 1
TIMEOUT_MAX = 30
DEFAULT_REGION = "wt-wt"
DEFAULT_SAFE_SEARCH = "off"
DEFAULT_PAGE = 1


class WebSearch(SocketTool):
    """Tool to perform web searches via DuckDuckGo."""

    def __init__(self, socket_path: str, user_agent: str | None = None) -> None:
        super().__init__(socket_path)
        self.user_agent = user_agent or DEFAULT_USER_AGENT

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        query = args.get("query", "")
        if markdown:
            return f"Search web: `{self._safe_md(query)}`"
        return f"Search web: {query}"

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": self.fname(),
                "description": (
                    "Search the web using DuckDuckGo. Returns a list of search results "
                    "with titles, URLs, and snippets."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (required).",
                        },
                        "region": {
                            "type": "string",
                            "description": (
                                "Region code for localized results "
                                "(default: 'wt-wt' for worldwide). "
                                "Examples: 'us-en', 'de-de', 'fr-fr'."
                            ),
                        },
                        "timelimit": {
                            "type": "string",
                            "description": (
                                "Time limit for results. "
                                "'d' = past day, 'w' = past week, "
                                "'m' = past month, 'y' = past year. "
                                "Omit for all time."
                            ),
                            "enum": ["d", "w", "m", "y"],
                        },
                        "page": {
                            "type": "integer",
                            "description": (
                                "Page number for pagination (default: 1). "
                                "Each page returns up to 15 results. "
                                "Use with max_results to control total results."
                            ),
                        },
                        "safe_search": {
                            "type": "string",
                            "description": (
                                "Safe search level: 'off', 'on', or 'moderate'. "
                                "Default is 'off'."
                            ),
                            "enum": ["off", "on", "moderate"],
                        },
                        "max_results": {
                            "type": "integer",
                            "description": (
                                f"Maximum number of results to return "
                                f"(default: {DEFAULT_MAX_RESULTS}, max: {MAX_RESULTS_LIMIT})."
                            ),
                        },
                        "timeout": {
                            "type": "integer",
                            "description": (
                                f"Request timeout in seconds "
                                f"({TIMEOUT_MIN}-{TIMEOUT_MAX}, default {DEFAULT_TIMEOUT})."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "WebSearch"

    def run(self, args: dict[str, Any]) -> str:
        args = dict(args)  # copy to avoid mutating caller's dict
        if "query" not in args or not args["query"]:
            return f"ERROR: {self.fname()} takes a mandatory 'query' argument!"
        query = args["query"]
        if not isinstance(query, str):
            return f"ERROR: {self.fname()}: query must be a string"
        region = args.get("region", DEFAULT_REGION)
        if not isinstance(region, str):
            return f"ERROR: {self.fname()}: region must be a string"
        timelimit = args.get("timelimit")
        if timelimit is not None and timelimit not in ("d", "w", "m", "y"):
            return (
                f"ERROR: {self.fname()}: timelimit must be one of 'd', 'w', 'm', or 'y'"
            )
        page = args.get("page", DEFAULT_PAGE)
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            return f"ERROR: {self.fname()}: page must be a positive integer"
        max_results = args.get("max_results", DEFAULT_MAX_RESULTS)
        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or max_results < 1
        ):
            return f"ERROR: {self.fname()}: max_results must be a positive integer"
        max_results = min(max_results, MAX_RESULTS_LIMIT)
        timeout = args.get("timeout", DEFAULT_TIMEOUT)
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or timeout < TIMEOUT_MIN
            or timeout > TIMEOUT_MAX
        ):
            return f"ERROR: {self.fname()}: timeout must be an integer between {TIMEOUT_MIN} and {TIMEOUT_MAX}"
        safe_search = args.get("safe_search")
        if safe_search is not None and safe_search not in ("off", "on", "moderate"):
            return (
                f"ERROR: {self.fname()}: safe_search must be 'off', 'on', or 'moderate'"
            )

        call_args: dict[str, Any] = {
            "query": query,
            "region": region,
            "page": page,
            "max_results": max_results,
            "timeout": timeout,
            "user_agent": self.user_agent,
            "safe_search": safe_search
            if safe_search is not None
            else DEFAULT_SAFE_SEARCH,
        }
        if timelimit is not None:
            call_args["timelimit"] = timelimit
        return self._call(call_args)
