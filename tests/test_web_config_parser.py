"""Tests for web_config_parser module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from tiz.manifest_parser import Manifest
from tiz.web_config_parser import (
    EndpointConfig,
    WebConfig,
    parse_web_config,
)


def _make_valid_manifest_dict() -> dict[str, Any]:
    """Return a minimal valid manifest dict."""
    return {
        "meta": {"version": "0"},
        "tasks": [],
    }


def _write_yaml(path: Path, data: Any) -> Path:
    """Write data as YAML to path and return the path."""
    path.write_text(yaml.dump(data))
    return path


def _prepare_manifest(tmp_path: Path, name: str = "manifest.yaml") -> Path:
    """Create a manifests dir and a valid manifest file, return the file path."""
    d = tmp_path / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    mf = d / name
    _write_yaml(mf, _make_valid_manifest_dict())
    return mf


def test_parse_web_config_simple(tmp_path: Path) -> None:
    """Parse a config with one endpoint referencing an external manifest file."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    from tiz.web_config_parser import _STATIC_DIR_FALLBACK

    result = parse_web_config(tmp_path, config_file)
    assert isinstance(result, WebConfig)
    assert "chat1" in result.endpoints
    ep = result.endpoints["chat1"]
    assert isinstance(ep, EndpointConfig)
    assert isinstance(ep.manifest, Manifest)
    assert ep.base_path == tmp_path
    assert ep.context == {}
    assert ep.task_name is None
    assert result.static_dir == _STATIC_DIR_FALLBACK


def test_parse_web_config_multiple_manifests(tmp_path: Path) -> None:
    """Parse a config with multiple manifest files per endpoint."""
    _prepare_manifest(tmp_path, "base.yaml")
    _prepare_manifest(tmp_path, "overrides.yaml")

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["base.yaml", "overrides.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert "chat1" in result.endpoints
    ep = result.endpoints["chat1"]
    assert isinstance(ep.manifest, Manifest)
    assert ep.base_path == tmp_path


def test_parse_web_config_with_options(tmp_path: Path) -> None:
    """Parse a config with options passed as inline overrides."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "options": {"meta": {"verbosity": 2}},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert "chat1" in result.endpoints
    assert result.endpoints["chat1"].manifest.meta.verbosity == 2
    assert result.endpoints["chat1"].base_path == tmp_path


def test_parse_web_config_options_not_dict(tmp_path: Path) -> None:
    """Non-dict options raises TypeError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "options": "not a dict",
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'options' must be a dict"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_with_all_fields(tmp_path: Path) -> None:
    """Parse a config with all optional fields set."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "context": {"user": "alice", "mode": "test"},
                "task_name": "main_task",
                "description": "A test endpoint",
                "suggestions": ["ask about weather", "help me write"],
                "options": {"meta": {"parallelism": 2}},
            },
        },
        "static_dir": "/var/www/static",
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    ep = result.endpoints["chat1"]
    assert ep.context == {"user": "alice", "mode": "test"}
    assert ep.task_name == "main_task"
    assert ep.description == "A test endpoint"
    assert ep.suggestions == ["ask about weather", "help me write"]
    assert ep.base_path == tmp_path
    assert result.static_dir == Path("/var/www/static")


def test_parse_web_config_description(tmp_path: Path) -> None:
    """Parse a config with description field."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "description": "Chat endpoint for testing",
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    ep = result.endpoints["chat1"]
    assert ep.description == "Chat endpoint for testing"
    assert ep.base_path == tmp_path


def test_parse_web_config_description_default_none(tmp_path: Path) -> None:
    """Description defaults to None when not provided."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].description is None


def test_parse_web_config_description_not_str(tmp_path: Path) -> None:
    """Non-string description raises TypeError."""
    _prepare_manifest(tmp_path)

    config: dict[str, Any] = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "description": 123,
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'description' must be a string"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_empty_endpoints(tmp_path: Path) -> None:
    """Parse a config with no endpoints."""
    config: dict[str, Any] = {"endpoints": {}}
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints == {}


def test_parse_web_config_blank_file(tmp_path: Path) -> None:
    """An empty config file raises ValueError."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_null_yaml(tmp_path: Path) -> None:
    """A file containing only '~' (null) raises ValueError."""
    config_file = tmp_path / "null.yaml"
    config_file.write_text("~\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_top_level_list(tmp_path: Path) -> None:
    """A top-level YAML list raises TypeError."""
    config_file = tmp_path / "list.yaml"
    _write_yaml(config_file, ["endpoints", {"chat1": {"manifests": ["m.yaml"]}}])
    with pytest.raises(TypeError, match="Web config must be a dict, got list"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_top_level_scalar(tmp_path: Path) -> None:
    """A top-level YAML scalar raises TypeError."""
    config_file = tmp_path / "scalar.yaml"
    _write_yaml(config_file, "just a string")
    with pytest.raises(TypeError, match="Web config must be a dict, got str"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_top_level_int(tmp_path: Path) -> None:
    """A top-level YAML integer raises TypeError."""
    config_file = tmp_path / "int.yaml"
    _write_yaml(config_file, 42)
    with pytest.raises(TypeError, match="Web config must be a dict, got int"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_file_not_found(tmp_path: Path) -> None:
    """Non-existent path raises FileNotFoundError."""
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        parse_web_config(tmp_path, missing)


def test_parse_web_config_is_directory(tmp_path: Path) -> None:
    """A directory path raises IsADirectoryError."""
    with pytest.raises(IsADirectoryError, match="directory"):
        parse_web_config(tmp_path, tmp_path)


def test_parse_web_config_endpoints_not_dict(tmp_path: Path) -> None:
    """Non-dict endpoints raises TypeError."""
    config: dict[str, Any] = {"endpoints": ["not", "a", "dict"]}
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'endpoints' must be a dict"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_endpoint_not_dict(tmp_path: Path) -> None:
    """Endpoint config that is not a dict raises TypeError."""
    config: dict[str, Any] = {
        "endpoints": {"chat1": "just a string"},
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="Endpoint 'chat1' config must be a dict"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_missing_manifests_key(tmp_path: Path) -> None:
    """Missing 'manifests' key raises ValueError."""
    config: dict[str, Any] = {
        "endpoints": {"chat1": {}},
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(ValueError, match="manifests"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_manifests_not_list(tmp_path: Path) -> None:
    """Manifests that is not a list raises TypeError."""
    config: dict[str, Any] = {
        "endpoints": {"chat1": {"manifests": "not a list"}},
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'manifests' must be a list"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_manifest_entry_not_string(tmp_path: Path) -> None:
    """A manifest entry that is not a string raises TypeError."""
    config: dict[str, Any] = {
        "endpoints": {"chat1": {"manifests": [12345]}},
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="must be a string path"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_context_not_dict(tmp_path: Path) -> None:
    """Non-dict context raises TypeError."""
    _prepare_manifest(tmp_path)

    config: dict[str, Any] = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "context": "should be a dict",
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'context' must be a dict"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_task_name_not_str(tmp_path: Path) -> None:
    """Non-string task_name raises TypeError."""
    _prepare_manifest(tmp_path)

    config: dict[str, Any] = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "task_name": 42,
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'task_name' must be a string"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_manifest_from_subdir(tmp_path: Path) -> None:
    """Manifest paths are resolved relative to the config file directory."""
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifests_dir / "chat.yaml"
    _write_yaml(manifest_file, _make_valid_manifest_dict())

    config = {
        "endpoints": {
            "chat1": {"manifests": ["chat.yaml"]},
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert "chat1" in result.endpoints


def test_parse_web_config_context_is_copied(tmp_path: Path) -> None:
    """The context dict is copied, not referenced."""
    _prepare_manifest(tmp_path)

    original_context: dict[str, Any] = {"key": "value"}
    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "context": original_context,
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    ep_context = result.endpoints["chat1"].context
    assert ep_context == {"key": "value"}
    # Mutating the original should not affect the parsed context
    original_context["key"] = "changed"
    assert ep_context == {"key": "value"}


def test_parse_web_config_preserves_empty_context(tmp_path: Path) -> None:
    """An empty context is left as an empty dict."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "context": {},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].context == {}


def test_parse_web_config_string_path(tmp_path: Path) -> None:
    """Accepts a plain string path argument."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {"manifests": ["manifest.yaml"]},
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, str(config_file))
    assert "chat1" in result.endpoints


def test_parse_web_config_manifest_not_found(tmp_path: Path) -> None:
    """A manifest file that doesn't exist raises ValueError."""
    config = {
        "endpoints": {
            "chat1": {"manifests": ["nonexistent.yaml"]},
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    with pytest.raises(ValueError, match="manifest parsing error"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_merged_manifests(tmp_path: Path) -> None:
    """Two endpoints with different manifest sets are independent."""
    _prepare_manifest(tmp_path, "a.yaml")
    _prepare_manifest(tmp_path, "b.yaml")

    config = {
        "endpoints": {
            "alpha": {"manifests": ["a.yaml"], "task_name": "task_a"},
            "beta": {"manifests": ["b.yaml"], "task_name": "task_b"},
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["alpha"].manifest is not result.endpoints["beta"].manifest
    assert result.endpoints["alpha"].task_name == "task_a"
    assert result.endpoints["beta"].task_name == "task_b"


def test_parse_web_config_endpoint_static_dir_override(tmp_path: Path) -> None:
    """Endpoint-level static_dir is ignored (only top-level matters)."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "static_dir": "/endpoint/static",
            },
        },
        "static_dir": "/global/static",
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    # Endpoint-level static_dir is now ignored; top-level wins
    assert result.static_dir == Path("/global/static")


def test_parse_web_config_endpoint_static_dir_inherits_global(tmp_path: Path) -> None:
    """Top-level static_dir sets WebConfig.static_dir."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
        "static_dir": "/global/static",
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.static_dir == Path("/global/static")


def test_parse_web_config_endpoint_static_dir_fallback_package(tmp_path: Path) -> None:
    """Endpoint falls back to package data dir when no static_dir is set anywhere."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    from tiz.web_config_parser import _STATIC_DIR_FALLBACK

    result = parse_web_config(tmp_path, config_file)
    assert result.static_dir == _STATIC_DIR_FALLBACK


def test_parse_web_config_static_dir_not_str(tmp_path: Path) -> None:
    """Non-string top-level static_dir raises TypeError."""
    _prepare_manifest(tmp_path)

    config: dict[str, Any] = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
        "static_dir": 123,
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'static_dir' must be a string path, got int"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_static_dir_not_str_list(tmp_path: Path) -> None:
    """Non-string top-level static_dir as list raises TypeError."""
    _prepare_manifest(tmp_path)

    config: dict[str, Any] = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
        "static_dir": ["path1", "path2"],
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'static_dir' must be a string path, got list"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_static_dir_none(tmp_path: Path) -> None:
    """None static_dir at top-level falls back to package default."""
    _prepare_manifest(tmp_path)

    config: dict[str, Any] = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
        "static_dir": None,
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    from tiz.web_config_parser import _STATIC_DIR_FALLBACK

    result = parse_web_config(tmp_path, config_file)
    assert result.static_dir == _STATIC_DIR_FALLBACK


def test_parse_web_config_endpoint_static_dir_not_str(tmp_path: Path) -> None:
    """Non-string endpoint static_dir is silently ignored (field no longer supported on endpoint)."""
    _prepare_manifest(tmp_path)

    config: dict[str, Any] = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "static_dir": 42,
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    from tiz.web_config_parser import _STATIC_DIR_FALLBACK

    result = parse_web_config(tmp_path, config_file)
    assert isinstance(result, WebConfig)
    assert result.static_dir == _STATIC_DIR_FALLBACK


def test_web_config_post_init_fallback() -> None:
    """WebConfig.__post_init__ resolves None static_dir to fallback."""
    from tiz.web_config_parser import _STATIC_DIR_FALLBACK

    config = WebConfig(endpoints={})
    assert config.static_dir == _STATIC_DIR_FALLBACK


def test_web_config_post_init_no_endpoint_propagation() -> None:
    """WebConfig.__post_init__ no longer propagates static_dir to endpoints (field removed)."""
    manifest = MagicMock(spec=Manifest)
    ep = EndpointConfig(manifest=manifest, base_path=Path("/tmp"))
    assert not hasattr(ep, "static_dir")


def test_web_config_post_init_empty_endpoints() -> None:
    """WebConfig.__post_init__ handles empty endpoints dict."""
    config = WebConfig(
        endpoints={},
        static_dir=Path("/custom"),
    )
    assert config.static_dir == Path("/custom")
    assert config.endpoints == {}


def test_parse_web_config_suggestions_default(tmp_path: Path) -> None:
    """Suggestions field defaults to empty list when not provided."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].suggestions == []


def test_parse_web_config_suggestions_parsed(tmp_path: Path) -> None:
    """Suggestions field is parsed correctly."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "suggestions": ["hello", "world"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].suggestions == ["hello", "world"]


def test_parse_web_config_suggestions_not_list(tmp_path: Path) -> None:
    """Non-list suggestions raises TypeError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "suggestions": "not a list",
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    with pytest.raises(TypeError, match="'suggestions' must be a list"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_suggestions_entry_not_str(tmp_path: Path) -> None:
    """A non-string entry in suggestions raises TypeError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "suggestions": [42],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    with pytest.raises(TypeError, match="'suggestions\\[0\\]' must be a string"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_suggestions_copied(tmp_path: Path) -> None:
    """The suggestions list is copied, not referenced."""
    _prepare_manifest(tmp_path)

    original: list[str] = ["hello", "world"]
    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "suggestions": original,
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    ep_suggestions = result.endpoints["chat1"].suggestions
    assert ep_suggestions == ["hello", "world"]
    original.append("extra")
    assert ep_suggestions == ["hello", "world"]


def test_parse_web_config_suggestions_empty_list(tmp_path: Path) -> None:
    """An empty suggestions list is preserved."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "suggestions": [],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].suggestions == []


def test_parse_web_config_auth_headers_default(tmp_path: Path) -> None:
    """auth_headers defaults to empty dict when not provided."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].auth_headers == {}


def test_parse_web_config_auth_headers_parsed(tmp_path: Path) -> None:
    """auth_headers is parsed correctly from YAML."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "auth_headers": {
                    "Authorization": "Bearer token123",
                    "X-API-Key": "abc123",
                },
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].auth_headers == {
        "Authorization": "Bearer token123",
        "X-API-Key": "abc123",
    }


def test_parse_web_config_auth_headers_not_dict(tmp_path: Path) -> None:
    """Non-dict auth_headers raises TypeError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "auth_headers": "not a dict",
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    with pytest.raises(TypeError, match="'auth_headers' must be a dict"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_auth_headers_key_not_str(tmp_path: Path) -> None:
    """Non-string key in auth_headers raises TypeError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "auth_headers": {42: "value"},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    with pytest.raises(TypeError, match="'auth_headers' keys must be strings"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_auth_headers_value_not_str(tmp_path: Path) -> None:
    """Non-string value in auth_headers raises TypeError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "auth_headers": {"Authorization": 123},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    with pytest.raises(
        TypeError, match="'auth_headers\\['Authorization'\\]' must be a string"
    ):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_auth_headers_empty_dict(tmp_path: Path) -> None:
    """An empty auth_headers dict is preserved."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "auth_headers": {},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].auth_headers == {}


def test_parse_web_config_auth_headers_copied(tmp_path: Path) -> None:
    """The auth_headers dict is copied, not referenced."""
    _prepare_manifest(tmp_path)

    original: dict[str, str] = {"Authorization": "Bearer token"}
    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "auth_headers": original,
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    ep_auth = result.endpoints["chat1"].auth_headers
    assert ep_auth == {"Authorization": "Bearer token"}
    original["X-Custom"] = "value"
    assert ep_auth == {"Authorization": "Bearer token"}


def test_parse_web_config_auth_headers_with_all_fields(tmp_path: Path) -> None:
    """auth_headers works alongside other endpoint fields."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "context": {"user": "alice"},
                "task_name": "main_task",
                "description": "Test endpoint",
                "suggestions": ["hello"],
                "auth_headers": {"Authorization": "Bearer xyz"},
                "options": {"meta": {"parallelism": 2}},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    ep = result.endpoints["chat1"]
    assert ep.context == {"user": "alice"}
    assert ep.task_name == "main_task"
    assert ep.description == "Test endpoint"
    assert ep.suggestions == ["hello"]
    assert ep.auth_headers == {"Authorization": "Bearer xyz"}


def test_endpoint_config_auth_headers_field_default() -> None:
    """EndpointConfig auth_headers defaults to empty dict."""
    manifest = MagicMock(spec=Manifest)
    ep = EndpointConfig(manifest=manifest, base_path=Path("/tmp"))
    assert ep.auth_headers == {}


def test_parse_web_config_endpoint_name_not_str(tmp_path: Path) -> None:
    """Non-string endpoint name raises TypeError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            123: {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="Endpoint name must be a string, got int"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_context_key_not_str(tmp_path: Path) -> None:
    """Non-string context key raises TypeError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "context": {42: "value"},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)
    with pytest.raises(TypeError, match="'context' keys must be strings"):
        parse_web_config(tmp_path, config_file)


def test_parse_web_config_relative_static_dir(tmp_path: Path) -> None:
    """Relative static_dir is resolved against config file parent."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
        "static_dir": "relative/static",
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    expected = (tmp_path / "relative/static").resolve()
    assert result.static_dir == expected


def test_parse_web_config_absolute_static_dir(tmp_path: Path) -> None:
    """Absolute static_dir is used as-is."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
        "static_dir": "/absolute/path",
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.static_dir == Path("/absolute/path")


def test_parse_web_config_context_value_not_str(tmp_path: Path) -> None:
    """Non-string context values are allowed (context is dict[str, Any])."""
    _prepare_manifest(tmp_path)

    config: dict[str, Any] = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "context": {"count": 42, "enabled": True, "tags": ["a", "b"]},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert result.endpoints["chat1"].context == {
        "count": 42,
        "enabled": True,
        "tags": ["a", "b"],
    }
    # Verify it's a copy
    config["endpoints"]["chat1"]["context"]["count"] = 99
    assert result.endpoints["chat1"].context["count"] == 42


# ──────────────────────────────────────────────
# Bug-fix tests (2.2 – default_options parameter)
# ──────────────────────────────────────────────


def test_parse_web_config_default_options_none(tmp_path: Path) -> None:
    """default_options=None does not alter the manifest."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file, default_options=None)
    ep = result.endpoints["chat1"]
    assert ep.base_path == tmp_path
    assert ep.manifest.meta.verbosity is None
    assert ep.manifest.meta.parallelism is None or ep.manifest.meta.parallelism == 1


def test_parse_web_config_default_options_empty_dict(tmp_path: Path) -> None:
    """default_options={} does not alter the manifest."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file, default_options={})
    ep = result.endpoints["chat1"]
    assert ep.base_path == tmp_path
    assert ep.manifest.meta.verbosity is None


def test_parse_web_config_default_options_override(tmp_path: Path) -> None:
    """default_options with meta.verbosity sets the base manifest value."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(
        tmp_path, config_file, default_options={"meta": {"verbosity": 1}}
    )
    ep = result.endpoints["chat1"]
    assert ep.base_path == tmp_path
    assert ep.manifest.meta.verbosity == 1


def test_parse_web_config_default_options_with_endpoint_options(tmp_path: Path) -> None:
    """Endpoint-level options override default_options."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "options": {"meta": {"verbosity": 2}},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(
        tmp_path, config_file, default_options={"meta": {"verbosity": 1}}
    )
    ep = result.endpoints["chat1"]
    assert ep.base_path == tmp_path
    # Endpoint-level options take precedence
    assert ep.manifest.meta.verbosity == 2


# ──────────────────────────────────────────────
# Bug-fix tests (2.3 – empty-string static_dir)
# ──────────────────────────────────────────────


def test_parse_web_config_static_dir_empty_string_raises(tmp_path: Path) -> None:
    """Empty-string static_dir raises ValueError."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
        "static_dir": "",
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    with pytest.raises(ValueError, match="non-empty"):
        parse_web_config(tmp_path, config_file)


# ──────────────────────────────────────────────
# Bug-fix tests (2.4 – unknown/extra keys)
# ──────────────────────────────────────────────


def test_parse_web_config_unknown_endpoint_key_is_silent(tmp_path: Path) -> None:
    """Unknown keys in endpoint config are silently ignored."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
                "unknown_field": "some_value",
                "another_unknown": {"nested": "data"},
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    ep = result.endpoints["chat1"]
    assert ep.base_path == tmp_path
    assert isinstance(ep.manifest, Manifest)


def test_parse_web_config_unknown_top_level_key_is_silent(tmp_path: Path) -> None:
    """Unknown top-level keys in config are silently ignored."""
    _prepare_manifest(tmp_path)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["manifest.yaml"],
            },
        },
        "unknown_top_key": "should be ignored",
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    assert "chat1" in result.endpoints
    assert result.endpoints["chat1"].base_path == tmp_path


# ──────────────────────────────────────────────
# Bug-fix tests (2.5 – merged manifest content)
# ──────────────────────────────────────────────


def test_parse_web_config_multiple_manifests_merged_content(tmp_path: Path) -> None:
    """Multiple manifests are merged correctly, with later files overriding earlier ones."""
    base_manifest = _make_valid_manifest_dict()
    base_manifest["meta"]["verbosity"] = 1
    base_manifest["meta"]["parallelism"] = 2
    base_manifest["meta"]["color"] = True
    d = tmp_path / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    _write_yaml(d / "base.yaml", base_manifest)

    override_manifest = _make_valid_manifest_dict()
    override_manifest["meta"]["verbosity"] = 2
    override_manifest["meta"]["color"] = None
    _write_yaml(d / "overrides.yaml", override_manifest)

    config = {
        "endpoints": {
            "chat1": {
                "manifests": ["base.yaml", "overrides.yaml"],
            },
        },
    }
    config_file = tmp_path / "web_config.yaml"
    _write_yaml(config_file, config)

    result = parse_web_config(tmp_path, config_file)
    ep = result.endpoints["chat1"]
    assert ep.base_path == tmp_path
    # 'verbosity' is set in override -> should be 2
    assert ep.manifest.meta.verbosity == 2
    # 'parallelism' is inherited from base (2) but gets reset to 1 by
    # can_parallelize since there are no parallel tasks
    assert ep.manifest.meta.parallelism == 1
    # 'color' is in base (True) and None in override -> picks base value
    assert ep.manifest.meta.color is True
    # 'color' explicitly set to None in override -> picks base value of True
    assert ep.manifest.meta.color is True


# ──────────────────────────────────────────────
# Bug-fix tests – verify base_path in existing tests
# ──────────────────────────────────────────────
