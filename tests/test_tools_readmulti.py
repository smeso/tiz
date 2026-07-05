"""Tests for the ReadMulti tool."""

import json
from unittest.mock import patch

from tiz.tools.base import MAX_INPUT_SIZE
from tiz.tools.readmulti import MAX_PATHS, ReadMulti

_TEST_SOCKET = "/tmp/test.sock"


class TestReadMultiPrompt:
    def test_prompt_returns_valid_json(self) -> None:
        tool = ReadMulti(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "ReadMulti"

    def test_prompt_has_required_fields(self) -> None:
        tool = ReadMulti(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["parameters"]["required"] == ["paths"]

    def test_prompt_includes_max_paths(self) -> None:
        tool = ReadMulti(_TEST_SOCKET)
        data = json.loads(tool.prompt())
        desc = data["parameters"]["properties"]["paths"]["description"]
        assert str(MAX_PATHS) in desc

    def test_prompt_has_valid_description(self) -> None:
        tool = ReadMulti(_TEST_SOCKET)
        data = json.loads(tool.prompt())
        assert data["description"] == "Read multiple files at once."

    def test_prompt_parameters_type_is_object(self) -> None:
        tool = ReadMulti(_TEST_SOCKET)
        data = json.loads(tool.prompt())
        assert data["parameters"]["type"] == "object"

    def test_prompt_paths_schema(self) -> None:
        tool = ReadMulti(_TEST_SOCKET)
        data = json.loads(tool.prompt())
        paths = data["parameters"]["properties"]["paths"]
        assert paths["type"] == "array"
        assert paths["items"] == {"type": "string"}


class TestReadMultiFname:
    def test_fname(self) -> None:
        assert ReadMulti.fname() == "ReadMulti"


class TestReadMultiRunValidation:
    def test_missing_paths(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({})
        assert result == "ERROR: ReadMulti takes a mandatory 'paths' argument!"

    def test_empty_paths(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": []})
        assert result == "ERROR: ReadMulti takes a mandatory 'paths' argument!"

    def test_paths_not_a_list(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": "not_a_list"})
        assert result == "ERROR: ReadMulti takes a mandatory 'paths' argument!"

    def test_paths_is_none(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": None})
        assert result == "ERROR: ReadMulti takes a mandatory 'paths' argument!"

    def test_paths_contains_non_string(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": [1, 2, 3]})
        assert result == "ERROR: ReadMulti 'paths' must contain only non-empty strings!"

    def test_paths_contains_none(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": ["a.txt", None]})
        assert result == "ERROR: ReadMulti 'paths' must contain only non-empty strings!"

    def test_paths_contains_mixed(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": ["a.txt", 42, "b.txt"]})
        assert result == "ERROR: ReadMulti 'paths' must contain only non-empty strings!"

    def test_paths_contains_empty_string(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": ["a.txt", ""]})
        assert result == "ERROR: ReadMulti 'paths' must contain only non-empty strings!"

    def test_paths_contains_only_empty_string(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": [""]})
        assert result == "ERROR: ReadMulti 'paths' must contain only non-empty strings!"

    def test_paths_exceeds_max(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        result = tool.run({"paths": ["f" + str(i) for i in range(MAX_PATHS + 1)]})
        assert result == f"ERROR: ReadMulti allows a maximum of {MAX_PATHS} paths"

    def test_path_exceeds_max_input_size(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        huge_path = "x" * (MAX_INPUT_SIZE + 1)
        result = tool.run({"paths": ["small.txt", huge_path]})
        assert result == "ERROR: ReadMulti path exceeds maximum allowed size"

    def test_serialized_input_exceeds_max_input_size(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        large_path = "x" * (MAX_INPUT_SIZE // 2)
        with patch.object(ReadMulti, "_call", return_value="called") as mock_call:
            result = tool.run({"paths": [large_path, large_path]})
        assert result == "called"
        mock_call.assert_called_once_with({"paths": [large_path, large_path]})

    def test_paths_at_max_limit(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        paths = ["f" + str(i) for i in range(MAX_PATHS)]
        with patch.object(ReadMulti, "_call", return_value="results") as mock_call:
            result = tool.run({"paths": paths})
        assert result == "results"
        mock_call.assert_called_once_with({"paths": paths})

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        with patch.object(ReadMulti, "_call", return_value="results") as mock_call:
            result = tool.run({"paths": ["a.txt", "b.txt"]})
        assert result == "results"
        mock_call.assert_called_once_with({"paths": ["a.txt", "b.txt"]})

    def test_strips_description_key(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        with patch.object(ReadMulti, "_call", return_value="results") as mock_call:
            result = tool.run(
                {
                    "paths": ["a.txt"],
                    "description": "Read multiple files",
                    "name": "ReadMulti",
                }
            )
        assert result == "results"
        mock_call.assert_called_once_with({"paths": ["a.txt"]})

    def test_strips_name_key(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        with patch.object(ReadMulti, "_call", return_value="results") as mock_call:
            result = tool.run({"paths": ["a.txt"], "name": "ReadMulti"})
        assert result == "results"
        mock_call.assert_called_once_with({"paths": ["a.txt"]})

    def test_does_not_mutate_original_args(self, socket_path: str) -> None:
        tool = ReadMulti(socket_path)
        args = {"paths": ["a.txt"], "description": "test"}
        with patch.object(ReadMulti, "_call", return_value="results") as mock_call:
            result = tool.run(args)
        assert result == "results"
        assert "description" in args
        mock_call.assert_called_once_with({"paths": ["a.txt"]})


class TestReadMultiFormatConfirmation:
    def test_format_confirmation(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation(
            {"paths": ["a.txt", "b.txt", "c.txt"]}, markdown=False
        )
        assert result == "Read 3 file(s): a.txt, b.txt, c.txt"

    def test_format_confirmation_markdown(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation({"paths": ["a.txt", "b.txt"]}, markdown=True)
        assert result == "Read 2 file(s): `a.txt`, `b.txt`"

    def test_format_confirmation_empty_paths(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Read 0 file(s): "

    def test_format_confirmation_more_than_3(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation(
            {"paths": ["a", "b", "c", "d", "e"]}, markdown=False
        )
        assert result == "Read 5 file(s): a, b, c, d, e"

    def test_format_confirmation_non_string_paths(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation({"paths": [123, 456, 789]}, markdown=False)
        assert result == "Read 3 file(s): 123, 456, 789"

    def test_format_confirmation_non_string_paths_markdown(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation({"paths": [123, 456]}, markdown=True)
        assert result == "Read 2 file(s): `123`, `456`"

    def test_format_confirmation_non_string_paths_mixed(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation(
            {"paths": ["a.txt", 42, "b.txt"]}, markdown=False
        )
        assert result == "Read 3 file(s): a.txt, 42, b.txt"

    def test_format_confirmation_paths_is_none(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation({"paths": None}, markdown=False)
        assert result == "Read 0 file(s): "

    def test_format_confirmation_paths_is_none_markdown(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation({"paths": None}, markdown=True)
        assert result == "Read 0 file(s): "

    def test_format_confirmation_markdown_backtick_stripping(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation(
            {"paths": ["`a.txt`", "b.txt"]}, markdown=True
        )
        assert result == "Read 2 file(s): `a.txt`, `b.txt`"

    def test_format_confirmation_non_markdown_backticks_preserved(self) -> None:
        tool = ReadMulti("/tmp/test.sock")
        result = tool.format_confirmation(
            {"paths": ["`a.txt`", "b.txt"]}, markdown=False
        )
        assert result == "Read 2 file(s): `a.txt`, b.txt"
