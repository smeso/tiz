"""Tests for the WebSearch tool."""

import json
from unittest.mock import patch

from tiz.tools.websearch import WebSearch


class TestWebSearchPrompt:
    def test_prompt_returns_valid_json(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "WebSearch"
        assert "description" in data
        assert "parameters" in data

    def test_prompt_has_required_fields(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert "query" in data["parameters"]["required"]
        assert data["parameters"]["properties"]["query"]["type"] == "string"

    def test_prompt_has_optional_fields(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        props = data["parameters"]["properties"]
        assert "region" in props
        assert "timelimit" in props
        assert "page" in props
        assert "max_results" in props
        assert "timeout" in props
        assert "safe_search" in props

    def test_prompt_timelimit_enum(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        timelimit = data["parameters"]["properties"]["timelimit"]
        assert timelimit["enum"] == ["d", "w", "m", "y"]

    def test_prompt_safe_search_enum(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        safe_search = data["parameters"]["properties"]["safe_search"]
        assert safe_search["enum"] == ["off", "on", "moderate"]


class TestWebSearchFname:
    def test_fname(self) -> None:
        assert WebSearch.fname() == "WebSearch"


class TestWebSearchInit:
    def test_default_user_agent(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        assert tool.user_agent is not None
        assert "tiz" in tool.user_agent

    def test_custom_user_agent(self, socket_path: str) -> None:
        tool = WebSearch(socket_path, user_agent="CustomAgent/1.0")
        assert tool.user_agent == "CustomAgent/1.0"

    def test_none_user_agent_uses_default(self, socket_path: str) -> None:
        tool = WebSearch(socket_path, user_agent=None)
        assert tool.user_agent is not None
        assert "tiz" in tool.user_agent


class TestWebSearchRunValidation:
    def test_missing_query(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({})
        assert result == "ERROR: WebSearch takes a mandatory 'query' argument!"

    def test_empty_query(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": ""})
        assert result == "ERROR: WebSearch takes a mandatory 'query' argument!"

    def test_query_not_string(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": 123})
        assert result == "ERROR: WebSearch: query must be a string"

    def test_valid_query(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "hello world"})
        assert result == "ok"
        mock_call.assert_called_once()

    def test_region_not_string(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "region": 123})
        assert result == "ERROR: WebSearch: region must be a string"

    def test_valid_region(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "region": "us-en"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["region"] == "us-en"

    def test_default_region(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["region"] == "wt-wt"

    def test_timelimit_invalid(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "timelimit": "x"})
        assert (
            result == "ERROR: WebSearch: timelimit must be one of 'd', 'w', 'm', or 'y'"
        )

    def test_timelimit_valid_day(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "timelimit": "d"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["timelimit"] == "d"

    def test_timelimit_valid_week(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "timelimit": "w"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["timelimit"] == "w"

    def test_timelimit_valid_month(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "timelimit": "m"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["timelimit"] == "m"

    def test_timelimit_valid_year(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "timelimit": "y"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["timelimit"] == "y"

    def test_timelimit_none_omitted(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert "timelimit" not in call_args

    def test_page_invalid_type(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "page": "1"})
        assert result == "ERROR: WebSearch: page must be a positive integer"

    def test_page_too_low(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "page": 0})
        assert result == "ERROR: WebSearch: page must be a positive integer"

    def test_page_negative(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "page": -1})
        assert result == "ERROR: WebSearch: page must be a positive integer"

    def test_page_valid(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "page": 3})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["page"] == 3

    def test_page_default(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["page"] == 1

    def test_max_results_invalid_type(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "max_results": "10"})
        assert result == "ERROR: WebSearch: max_results must be a positive integer"

    def test_max_results_too_low(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "max_results": 0})
        assert result == "ERROR: WebSearch: max_results must be a positive integer"

    def test_max_results_negative(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "max_results": -5})
        assert result == "ERROR: WebSearch: max_results must be a positive integer"

    def test_max_results_capped_at_50(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "max_results": 100})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["max_results"] == 50

    def test_max_results_default(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["max_results"] == 10

    def test_max_results_valid_value(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "max_results": 25})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["max_results"] == 25

    def test_max_results_at_limit(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "max_results": 50})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["max_results"] == 50

    def test_timeout_invalid_type(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "timeout": "10"})
        assert result == "ERROR: WebSearch: timeout must be an integer between 1 and 30"

    def test_timeout_too_low(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "timeout": 0})
        assert result == "ERROR: WebSearch: timeout must be an integer between 1 and 30"

    def test_timeout_too_high(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "timeout": 31})
        assert result == "ERROR: WebSearch: timeout must be an integer between 1 and 30"

    def test_timeout_negative(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "timeout": -5})
        assert result == "ERROR: WebSearch: timeout must be an integer between 1 and 30"

    def test_timeout_valid(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "timeout": 15})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["timeout"] == 15

    def test_timeout_default(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["timeout"] == 10

    def test_user_agent_injected(self, socket_path: str) -> None:
        tool = WebSearch(socket_path, user_agent="TestAgent/1.0")
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["user_agent"] == "TestAgent/1.0"

    def test_name_not_in_call_args(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        args = {"query": "test", "name": "WebSearch"}
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            tool.run(args)
        assert "name" in args
        call_args = mock_call.call_args[0][0]
        assert "name" not in call_args

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(
            WebSearch, "_call", return_value='{"result": "ok"}'
        ) as mock_call:
            result = tool.run({"query": "test query"})
        assert result == '{"result": "ok"}'
        mock_call.assert_called_once()

    def test_all_params_passed(self, socket_path: str) -> None:
        tool = WebSearch(socket_path, user_agent="Agent/1.0")
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run(
                {
                    "query": "python programming",
                    "region": "us-en",
                    "timelimit": "m",
                    "page": 2,
                    "max_results": 15,
                    "timeout": 20,
                    "safe_search": "on",
                }
            )
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["query"] == "python programming"
        assert call_args["region"] == "us-en"
        assert call_args["timelimit"] == "m"
        assert call_args["page"] == 2
        assert call_args["max_results"] == 15
        assert call_args["timeout"] == 20
        assert call_args["user_agent"] == "Agent/1.0"
        assert call_args["safe_search"] == "on"

    def test_safe_search_invalid(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "safe_search": "invalid"})
        assert (
            result == "ERROR: WebSearch: safe_search must be 'off', 'on', or 'moderate'"
        )

    def test_safe_search_off(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "safe_search": "off"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["safe_search"] == "off"

    def test_safe_search_on(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "safe_search": "on"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["safe_search"] == "on"

    def test_safe_search_moderate(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test", "safe_search": "moderate"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["safe_search"] == "moderate"

    def test_safe_search_default_off(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        with patch.object(WebSearch, "_call", return_value="ok") as mock_call:
            result = tool.run({"query": "test"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["safe_search"] == "off"

    def test_safe_search_empty_string(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "safe_search": ""})
        assert (
            result == "ERROR: WebSearch: safe_search must be 'off', 'on', or 'moderate'"
        )

    def test_timelimit_empty_string(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.run({"query": "test", "timelimit": ""})
        assert (
            result == "ERROR: WebSearch: timelimit must be one of 'd', 'w', 'm', or 'y'"
        )


class TestWebSearchFormatConfirmation:
    def test_format_confirmation(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.format_confirmation(
            {"query": "python programming"}, markdown=False
        )
        assert result == "Search web: python programming"

    def test_format_confirmation_markdown(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.format_confirmation({"query": "test query"}, markdown=True)
        assert result == "Search web: `test query`"

    def test_format_confirmation_empty_query(self, socket_path: str) -> None:
        tool = WebSearch(socket_path)
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Search web: "

    def test_format_confirmation_markdown_sanitizes_backticks(
        self, socket_path: str
    ) -> None:
        tool = WebSearch(socket_path)
        result = tool.format_confirmation(
            {"query": "python `code` test"}, markdown=True
        )
        assert result == "Search web: `python code test`"

    def test_format_confirmation_nonmarkdown_keeps_backticks(
        self, socket_path: str
    ) -> None:
        tool = WebSearch(socket_path)
        result = tool.format_confirmation(
            {"query": "python `code` test"}, markdown=False
        )
        assert result == "Search web: python `code` test"
