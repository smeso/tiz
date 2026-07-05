"""Tests for the Grep tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.grep import Grep


class TestGrepPrompt:
    def test_prompt_returns_valid_json(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "Grep"

    def test_prompt_has_required_fields(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["parameters"]["required"] == ["pattern"]

    def test_prompt_has_description(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        data = json.loads(tool.prompt())
        desc = data.get("description")
        assert desc is not None
        assert len(desc) > 0

    def test_prompt_parameters_type_is_object(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        data = json.loads(tool.prompt())
        assert data["parameters"]["type"] == "object"

    def test_prompt_has_all_properties(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        data = json.loads(tool.prompt())
        props = data["parameters"]["properties"]
        expected = {
            "pattern",
            "path",
            "glob",
            "regex",
            "max_results",
            "case_insensitive",
        }
        assert set(props.keys()) == expected

    def test_prompt_property_types(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        data = json.loads(tool.prompt())
        props = data["parameters"]["properties"]
        assert props["pattern"]["type"] == "string"
        assert props["path"]["type"] == "string"
        assert props["glob"]["type"] == "string"
        assert props["regex"]["type"] == "boolean"
        assert props["max_results"]["type"] == "integer"
        assert props["case_insensitive"]["type"] == "boolean"

    def test_prompt_property_descriptions(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        data = json.loads(tool.prompt())
        props = data["parameters"]["properties"]
        for _name, prop in props.items():
            assert "description" in prop
            assert len(prop["description"]) > 0

    def test_prompt_max_results_constraints(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        data = json.loads(tool.prompt())
        max_results = data["parameters"]["properties"]["max_results"]
        assert max_results["minimum"] == 1
        assert max_results["maximum"] == 10000

    def test_prompt_description_and_type_are_present(self, socket_path: str) -> None:
        """Validate that description and type are in the prompt top-level."""
        tool = Grep(socket_path)
        data = json.loads(tool.prompt())
        assert "description" in data
        assert data["description"] != ""
        assert data["parameters"]["type"] == "object"


class TestGrepFname:
    def test_fname(self) -> None:
        assert Grep.fname() == "Grep"


class TestGrepRunValidation:
    def test_missing_pattern(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({})
        assert result == "ERROR: Grep takes a mandatory 'pattern' argument!"

    def test_empty_pattern(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": ""})
        assert result == "ERROR: Grep takes a mandatory 'pattern' argument!"

    def test_pattern_too_large(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: pattern exceeds maximum allowed size"

    def test_invalid_regex_pattern(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "[invalid", "regex": True})
        assert result.startswith("ERROR: invalid regex pattern:")

    def test_valid_regex_pattern(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "valid.*regex", "regex": True})
        assert result == "match"
        mock_call.assert_called_once_with(
            {"pattern": "valid.*regex", "regex": True, "max_results": 100}
        )

    def test_literal_pattern_skips_regex_validation(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "[invalid", "regex": False})
        assert result == "match"
        mock_call.assert_called_once_with(
            {"pattern": "[invalid", "regex": False, "max_results": 100}
        )

    def test_invalid_max_results_zero(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "max_results": 0})
        assert result == "ERROR: max_results must be an integer between 1 and 10000"

    def test_invalid_max_results_too_high(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "max_results": 10001})
        assert result == "ERROR: max_results must be an integer between 1 and 10000"

    def test_invalid_max_results_type(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "max_results": "100"})
        assert result == "ERROR: max_results must be an integer between 1 and 10000"

    def test_invalid_max_results_bool(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "max_results": True})
        assert result == "ERROR: max_results must be an integer between 1 and 10000"

    def test_invalid_max_results_negative(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "max_results": -1})
        assert result == "ERROR: max_results must be an integer between 1 and 10000"

    def test_invalid_pattern_int(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": 123})
        assert result == "ERROR: Grep takes a mandatory 'pattern' argument!"

    def test_invalid_pattern_bool(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": True})
        assert result == "ERROR: Grep takes a mandatory 'pattern' argument!"

    def test_invalid_pattern_list(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": ["hello"]})
        assert result == "ERROR: Grep takes a mandatory 'pattern' argument!"

    def test_invalid_pattern_dict(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": {"key": "val"}})
        assert result == "ERROR: Grep takes a mandatory 'pattern' argument!"

    def test_invalid_pattern_none(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": None})
        assert result == "ERROR: Grep takes a mandatory 'pattern' argument!"

    def test_invalid_pattern_float(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": 3.14})
        assert result == "ERROR: Grep takes a mandatory 'pattern' argument!"

    def test_valid_max_results(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "test", "max_results": 50})
        assert result == "match"
        mock_call.assert_called_once_with({"pattern": "test", "max_results": 50})

    def test_default_regex_not_sent(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            tool.run({"pattern": "test"})
        call_args = mock_call.call_args[0][0]
        assert "regex" not in call_args

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "hello"})
        assert result == "match"
        mock_call.assert_called_once_with({"pattern": "hello", "max_results": 100})

    def test_args_not_mutated(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        original = {"pattern": "hello"}
        with patch.object(Grep, "_call", return_value="match"):
            tool.run(original)
        assert original == {"pattern": "hello"}

    def test_invalid_path_not_string(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "path": 123})
        assert result == "ERROR: path must be a string"

    def test_invalid_path_bool(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "path": True})
        assert result == "ERROR: path must be a string"

    def test_invalid_path_list(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "path": ["/src"]})
        assert result == "ERROR: path must be a string"

    def test_invalid_path_dict(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "path": {"dir": "/src"}})
        assert result == "ERROR: path must be a string"

    def test_invalid_path_none(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "path": None})
        assert result == "ERROR: path must be a string"

    def test_invalid_path_float(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "path": 1.5})
        assert result == "ERROR: path must be a string"

    def test_valid_path_string(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "test", "path": "/src"})
        assert result == "match"
        mock_call.assert_called_once_with(
            {"pattern": "test", "path": "/src", "max_results": 100}
        )

    def test_invalid_glob_not_string(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "glob": 456})
        assert result == "ERROR: glob must be a string"

    def test_invalid_glob_bool(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "glob": False})
        assert result == "ERROR: glob must be a string"

    def test_invalid_glob_list(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "glob": ["*.py"]})
        assert result == "ERROR: glob must be a string"

    def test_invalid_glob_dict(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "glob": {"ext": "py"}})
        assert result == "ERROR: glob must be a string"

    def test_invalid_glob_none(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "glob": None})
        assert result == "ERROR: glob must be a string"

    def test_invalid_glob_float(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "glob": 3.14})
        assert result == "ERROR: glob must be a string"

    def test_valid_glob_string(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "test", "glob": "*.py"})
        assert result == "match"
        mock_call.assert_called_once_with(
            {"pattern": "test", "glob": "*.py", "max_results": 100}
        )

    def test_path_validation_before_max_results(self, socket_path: str) -> None:
        """path validation should happen before max_results validation."""
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "path": 123, "max_results": 0})
        assert result == "ERROR: path must be a string"

    def test_invalid_regex_bool(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "regex": "yes"})
        assert result == "ERROR: regex must be a boolean"

    def test_invalid_regex_int(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "regex": 1})
        assert result == "ERROR: regex must be a boolean"

    def test_invalid_regex_list(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "regex": ["true"]})
        assert result == "ERROR: regex must be a boolean"

    def test_invalid_regex_dict(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "regex": {"a": 1}})
        assert result == "ERROR: regex must be a boolean"

    def test_invalid_regex_none(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "regex": None})
        assert result == "ERROR: regex must be a boolean"

    def test_invalid_regex_float(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "regex": 1.5})
        assert result == "ERROR: regex must be a boolean"

    def test_invalid_case_insensitive_bool(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "case_insensitive": "true"})
        assert result == "ERROR: case_insensitive must be a boolean"

    def test_invalid_case_insensitive_int(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "case_insensitive": 1})
        assert result == "ERROR: case_insensitive must be a boolean"

    def test_invalid_case_insensitive_list(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "case_insensitive": ["true"]})
        assert result == "ERROR: case_insensitive must be a boolean"

    def test_invalid_case_insensitive_dict(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "case_insensitive": {"a": 1}})
        assert result == "ERROR: case_insensitive must be a boolean"

    def test_invalid_case_insensitive_none(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "case_insensitive": None})
        assert result == "ERROR: case_insensitive must be a boolean"

    def test_invalid_case_insensitive_float(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "hello", "case_insensitive": 1.5})
        assert result == "ERROR: case_insensitive must be a boolean"

    def test_valid_case_insensitive(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "hello", "case_insensitive": True})
        assert result == "match"
        mock_call.assert_called_once_with(
            {"pattern": "hello", "case_insensitive": True, "max_results": 100}
        )

    def test_valid_case_insensitive_false(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "hello", "case_insensitive": False})
        assert result == "match"
        mock_call.assert_called_once_with(
            {"pattern": "hello", "case_insensitive": False, "max_results": 100}
        )

    def test_regex_validation_before_case_insensitive(self, socket_path: str) -> None:
        """regex validation should happen before case_insensitive validation."""
        tool = Grep(socket_path)
        result = tool.run(
            {"pattern": "hello", "regex": "bad", "case_insensitive": "bad"}
        )
        assert result == "ERROR: regex must be a boolean"

    def test_invalid_path_too_large(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "path": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_invalid_glob_too_large(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "glob": "x" * (MAX_INPUT_SIZE + 1)})
        assert result == "ERROR: glob exceeds maximum allowed size"

    def test_path_size_validation_before_glob(self, socket_path: str) -> None:
        """path size validation should happen before glob validation."""
        tool = Grep(socket_path)
        result = tool.run(
            {"pattern": "test", "path": "x" * (MAX_INPUT_SIZE + 1), "glob": "bad"}
        )
        assert result == "ERROR: path exceeds maximum allowed size"

    def test_glob_validation_before_max_results(self, socket_path: str) -> None:
        """glob validation should happen before max_results validation."""
        tool = Grep(socket_path)
        result = tool.run({"pattern": "test", "glob": 456, "max_results": 0})
        assert result == "ERROR: glob must be a string"

    def test_strips_name_from_args(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "hello", "name": "WrongName"})
        assert result == "match"
        call_args = mock_call.call_args[0][0]
        assert "name" not in call_args
        assert call_args["pattern"] == "hello"

    def test_strips_description_from_args(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        with patch.object(Grep, "_call", return_value="match") as mock_call:
            result = tool.run({"pattern": "hello", "description": "some description"})
        assert result == "match"
        call_args = mock_call.call_args[0][0]
        assert "description" not in call_args
        assert call_args["pattern"] == "hello"


class TestGrepFormatConfirmation:
    def test_format_confirmation(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.format_confirmation({"pattern": "hello"}, markdown=False)
        assert result == "Grep for 'hello' in ."

    def test_format_confirmation_markdown(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.format_confirmation(
            {"pattern": "TODO", "path": "/src"}, markdown=True
        )
        assert result == "Grep for `TODO` in `/src`"

    def test_format_confirmation_empty_pattern(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Grep for '' in ."

    def test_format_confirmation_path_none(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.format_confirmation(
            {"pattern": "hello", "path": None}, markdown=False
        )
        assert result == "Grep for 'hello' in None"

    def test_format_confirmation_path_empty_string(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.format_confirmation(
            {"pattern": "hello", "path": ""}, markdown=False
        )
        assert result == "Grep for 'hello' in "

    def test_format_confirmation_empty_pattern_empty_path(
        self, socket_path: str
    ) -> None:
        tool = Grep(socket_path)
        result = tool.format_confirmation({}, markdown=True)
        assert result == "Grep for `` in `.`"

    def test_format_confirmation_markdown_empty_path(self, socket_path: str) -> None:
        tool = Grep(socket_path)
        result = tool.format_confirmation(
            {"pattern": "hello", "path": ""}, markdown=True
        )
        assert result == "Grep for `hello` in ``"

    def test_format_confirmation_markdown_backticks_in_pattern(
        self, socket_path: str
    ) -> None:
        """Backticks in pattern should be stripped in markdown mode."""
        tool = Grep(socket_path)
        result = tool.format_confirmation(
            {"pattern": "`TODO`", "path": "/src"}, markdown=True
        )
        assert result == "Grep for `TODO` in `/src`"

    def test_format_confirmation_markdown_backticks_in_path(
        self, socket_path: str
    ) -> None:
        """Backticks in path should be stripped in markdown mode."""
        tool = Grep(socket_path)
        result = tool.format_confirmation(
            {"pattern": "hello", "path": "/tmp/foo`bar"}, markdown=True
        )
        assert result == "Grep for `hello` in `/tmp/foobar`"

    def test_format_confirmation_markdown_backticks_in_both(
        self, socket_path: str
    ) -> None:
        """Backticks in both pattern and path should be stripped in markdown mode."""
        tool = Grep(socket_path)
        result = tool.format_confirmation(
            {"pattern": "`TODO`", "path": "/tmp/foo`bar"}, markdown=True
        )
        assert result == "Grep for `TODO` in `/tmp/foobar`"

    def test_format_confirmation_backticks_preserved_non_markdown(
        self, socket_path: str
    ) -> None:
        """Backticks should be preserved verbatim when markdown=False."""
        tool = Grep(socket_path)
        result = tool.format_confirmation(
            {"pattern": "`TODO`", "path": "/tmp/foo`bar"}, markdown=False
        )
        assert result == "Grep for '`TODO`' in /tmp/foo`bar"
