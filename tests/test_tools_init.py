"""Tests for the tools __init__ module."""

import importlib
import json
from typing import Any

from tiz.tools import __all__
from tiz.tools.base import SocketTool, Tool

_tools = importlib.import_module("tiz.tools")


def _instantiate(name: str) -> Any:
    """Create an instance of a tool class with minimal required args."""
    cls = getattr(_tools, name)
    if issubclass(cls, SocketTool):
        return cls(socket_path="/tmp/test.sock")
    return cls(subagents={})


def test_all_exports_exist() -> None:
    for name in __all__:
        cls = getattr(_tools, name, None)
        assert cls is not None, f"{name} not found in tiz.tools"


def test_all_classes_have_fname() -> None:
    for name in __all__:
        cls = getattr(_tools, name)
        fname = cls.fname()
        assert isinstance(fname, str)


def test_all_classes_have_run() -> None:
    for name in __all__:
        cls = getattr(_tools, name)
        assert callable(getattr(cls, "run", None)), f"{name}.run is not callable"


def test_all_prompts_return_valid_json() -> None:
    for name in __all__:
        instance = _instantiate(name)
        prompt = instance.prompt()
        data = json.loads(prompt)
        assert "name" in data, f"{name}.prompt() missing 'name'"
        assert data["name"] == instance.fname(), (
            f"{name}: prompt name '{data['name']}' != fname '{instance.fname()}'"
        )
        assert "description" in data, f"{name}.prompt() missing 'description'"
        assert isinstance(data["description"], str) and data["description"], (
            f"{name}.prompt() has empty description"
        )


def test_all_fnames_are_unique() -> None:
    fnames = []
    for name in __all__:
        cls = getattr(_tools, name)
        fnames.append(cls.fname())
    assert len(fnames) == len(set(fnames))


def test_all_classes_have_format_confirmation() -> None:
    for name in __all__:
        cls = getattr(_tools, name)
        assert callable(getattr(cls, "format_confirmation", None)), (
            f"{name}.format_confirmation is not callable"
        )


def test_all_format_confirmations_return_valid_types() -> None:
    for name in __all__:
        instance = _instantiate(name)
        result = instance.format_confirmation({"test": "data"})
        assert result is None or isinstance(result, str), (
            f"{name}.format_confirmation() returned {type(result).__name__}"
        )
        result_md = instance.format_confirmation({"test": "data"}, markdown=True)
        assert result_md is None or isinstance(result_md, str), (
            f"{name}.format_confirmation(markdown=True) returned {type(result_md).__name__}"
        )


def test_all_classes_inherit_from_tool() -> None:
    for name in __all__:
        cls = getattr(_tools, name)
        assert issubclass(cls, Tool), f"{name} is not a subclass of Tool"


def test_all_prompts_have_valid_parameters_schema() -> None:
    for name in __all__:
        instance = _instantiate(name)
        prompt = instance.prompt()
        data = json.loads(prompt)
        params = data["parameters"]
        assert params["type"] == "object", f"{name}: parameters.type is not 'object'"
        assert "properties" in params, f"{name}: parameters missing 'properties'"
        assert isinstance(params["properties"], dict), (
            f"{name}: parameters.properties is not a dict"
        )
        assert params["properties"], f"{name}: parameters.properties is empty"
        for prop_name, prop in params["properties"].items():
            assert "description" in prop, (
                f"{name}: property '{prop_name}' missing 'description'"
            )
            assert isinstance(prop["description"], str) and prop["description"], (
                f"{name}: property '{prop_name}' has empty description"
            )
            has_type = "type" in prop
            has_oneof = "oneOf" in prop
            assert has_type or has_oneof, (
                f"{name}: property '{prop_name}' must have 'type' or 'oneOf'"
            )
        if "additionalProperties" in params:
            assert params["additionalProperties"] is False, (
                f"{name}: additionalProperties should be False if present, "
                f"got {params['additionalProperties']}"
            )
        assert "required" in params, f"{name}: parameters missing 'required'"
        for req_prop in params["required"]:
            assert req_prop in params["properties"], (
                f"{name}: required property '{req_prop}' not in properties"
            )


def test_subagents_prompt_with_subagents() -> None:
    from tiz.tools.subagents import SubAgents

    instance = SubAgents(
        subagents={
            "agent1": {
                "chat": lambda: None,
                "description": "First agent for testing",
            },
        },
    )
    prompt = instance.prompt()
    data = json.loads(prompt)
    assert data["name"] == "SubAgents"
    assert isinstance(data["description"], str) and data["description"]
    assert "First agent for testing" in prompt
    assert "agent1" in prompt
    params = data["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "required" in params
    assert "name" in params["properties"]
    assert "prompt" in params["properties"]
    assert params.get("additionalProperties") is False


def test_subagents_prompt_no_subagents() -> None:
    from tiz.tools.subagents import SubAgents

    instance = SubAgents(subagents={})
    prompt = instance.prompt()
    data = json.loads(prompt)
    assert data["name"] == "SubAgents"
    assert isinstance(data["description"], str) and data["description"]
    assert "No sub-agents are currently available" in prompt
    params = data["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "required" in params


def test_fname_matches_class_name() -> None:
    for name in __all__:
        cls = getattr(_tools, name)
        assert cls.fname() == name, f"{name}: fname() != class name"


def test_all_exports_are_sorted() -> None:
    assert __all__ == sorted(__all__), "__all__ is not sorted alphabetically"
