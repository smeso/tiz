"""Configuration models and YAML parsing for WebSocket endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tiz.helpers import parse_manifest as _parse_manifest
from tiz.manifest_parser import Manifest

__all__ = [
    "EndpointConfig",
    "MANIFEST_VERSION",
    "WebConfig",
    "parse_web_config",
]

MANIFEST_VERSION = "0"

_STATIC_DIR_FALLBACK = Path(__file__).resolve().parent / "data" / "web_static"


@dataclass
class EndpointConfig:
    """Configuration for a named WebSocket endpoint."""

    manifest: Manifest
    base_path: Path
    context: dict[str, Any] = field(default_factory=dict)
    task_name: str | None = None
    description: str | None = None
    suggestions: list[str] = field(default_factory=list)
    auth_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class WebConfig:
    """Configuration for a web application with multiple WebSocket endpoints."""

    endpoints: dict[str, EndpointConfig] = field(default_factory=dict)
    static_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.static_dir is None:
            self.static_dir = _STATIC_DIR_FALLBACK
        self.static_dir = self.static_dir.resolve()


def parse_web_config(
    base_path: Path, path: str | Path, default_options: dict[str, Any] | None = None
) -> WebConfig:
    """Parse a YAML file and return a WebConfig object.

    The YAML file supports the following structure:

    .. code-block:: yaml

        meta:
          version: "0"
        endpoints:
          chat1:
            manifests:
              - path/to/manifest.yaml
            options:
              meta:
                parallelism: 2
            context:
              key: value
            task_name: mytask
            description: A test endpoint
            suggestions:
              - what time is it?
            auth_headers:
              Authorization: Bearer token123
        static_dir: /path/to/static

    ``static_dir`` is a top-level configuration.  If missing, a fallback path
    inside the tiz package is used.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Web config file not found: {config_path}")
    if not config_path.is_file():
        raise IsADirectoryError(f"Expected a file, got directory: {config_path}")

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Web config file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise TypeError(f"Web config must be a dict, got {type(raw).__name__}")

    meta_raw = raw.get("meta")
    if meta_raw is None:
        raise ValueError("Web config is missing required 'meta' field")
    if not isinstance(meta_raw, dict):
        raise TypeError(f"'meta' must be a dict, got {type(meta_raw).__name__}")
    version_raw = meta_raw.get("version")
    if version_raw is None:
        raise ValueError("Web config 'meta' is missing required 'version' field")
    if not isinstance(version_raw, str):
        raise TypeError(
            f"'meta.version' must be a string, got {type(version_raw).__name__}"
        )
    if version_raw != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported meta version {version_raw!r}, expected {MANIFEST_VERSION!r}"
        )

    endpoints_raw = raw.get("endpoints", {})
    if not isinstance(endpoints_raw, dict):
        raise TypeError(
            f"'endpoints' must be a dict, got {type(endpoints_raw).__name__}"
        )

    endpoints: dict[str, EndpointConfig] = {}
    for name, ep_data in endpoints_raw.items():
        if not isinstance(name, str):
            raise TypeError(
                f"Endpoint name must be a string, got {type(name).__name__}"
            )
        if not isinstance(ep_data, dict):
            raise TypeError(
                f"Endpoint {name!r} config must be a dict, got {type(ep_data).__name__}"
            )

        manifests_raw = ep_data.get("manifests")
        if manifests_raw is None:
            raise ValueError(f"Endpoint {name!r} is missing required 'manifests' key")
        if not isinstance(manifests_raw, list):
            raise TypeError(
                f"Endpoint {name!r} 'manifests' must be a list, "
                f"got {type(manifests_raw).__name__}"
            )

        manifest_paths: list[Path] = []
        for i, m in enumerate(manifests_raw):
            if not isinstance(m, str):
                raise TypeError(
                    f"Endpoint {name!r} 'manifests[{i}]' must be a string path, "
                    f"got {type(m).__name__}"
                )
            manifest_paths.append(Path(m))

        options: dict[str, Any] | None = ep_data.get("options")
        if options is not None and not isinstance(options, dict):
            raise TypeError(
                f"Endpoint {name!r} 'options' must be a dict, "
                f"got {type(options).__name__}"
            )

        context_raw: Any = ep_data.get("context", {})
        if not isinstance(context_raw, dict):
            raise TypeError(
                f"Endpoint {name!r} 'context' must be a dict, "
                f"got {type(context_raw).__name__}"
            )
        context: dict[str, Any] = {}
        for k, v in context_raw.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"Endpoint {name!r} 'context' keys must be strings, "
                    f"got {type(k).__name__}"
                )
            context[k] = v
        task_name: str | None = ep_data.get("task_name")
        if task_name is not None and not isinstance(task_name, str):
            raise TypeError(
                f"Endpoint {name!r} 'task_name' must be a string, "
                f"got {type(task_name).__name__}"
            )

        description: str | None = ep_data.get("description")
        if description is not None and not isinstance(description, str):
            raise TypeError(
                f"Endpoint {name!r} 'description' must be a string, "
                f"got {type(description).__name__}"
            )

        suggestions_raw: Any = ep_data.get("suggestions", [])
        if not isinstance(suggestions_raw, list):
            raise TypeError(
                f"Endpoint {name!r} 'suggestions' must be a list, "
                f"got {type(suggestions_raw).__name__}"
            )
        suggestions: list[str] = []
        for i, s in enumerate(suggestions_raw):
            if not isinstance(s, str):
                raise TypeError(
                    f"Endpoint {name!r} 'suggestions[{i}]' must be a string, "
                    f"got {type(s).__name__}"
                )
            suggestions.append(s)

        auth_headers_raw: Any = ep_data.get("auth_headers", {})
        if not isinstance(auth_headers_raw, dict):
            raise TypeError(
                f"Endpoint {name!r} 'auth_headers' must be a dict, "
                f"got {type(auth_headers_raw).__name__}"
            )
        auth_headers: dict[str, str] = {}
        for k, v in auth_headers_raw.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"Endpoint {name!r} 'auth_headers' keys must be strings, "
                    f"got {type(k).__name__}"
                )
            if not isinstance(v, str):
                raise TypeError(
                    f"Endpoint {name!r} 'auth_headers[{k!r}]' must be a string, "
                    f"got {type(v).__name__}"
                )
            auth_headers[k] = v

        manifest, error = _parse_manifest(
            base_path=base_path,
            manifests=manifest_paths,
            default_options=default_options,
            options=options,
        )
        if error is not None or manifest is None:
            raise ValueError(f"Endpoint {name!r} manifest parsing error: {error}")

        endpoints[name] = EndpointConfig(
            manifest=manifest,
            base_path=base_path,
            context=context,
            task_name=task_name,
            description=description,
            suggestions=suggestions,
            auth_headers=auth_headers,
        )

    web_static_dir_raw: Any = raw.get("static_dir")
    if web_static_dir_raw is not None and not isinstance(web_static_dir_raw, str):
        raise TypeError(
            f"'static_dir' must be a string path, got {type(web_static_dir_raw).__name__}"
        )
    if isinstance(web_static_dir_raw, str) and not web_static_dir_raw:
        raise ValueError(
            "'static_dir' must be a non-empty string path, got empty string"
        )
    web_static_dir: Path | None = None
    if web_static_dir_raw:
        static_p = Path(web_static_dir_raw)
        if not static_p.is_absolute():
            static_p = config_path.parent / static_p
        web_static_dir = static_p.resolve()

    return WebConfig(
        endpoints=endpoints,
        static_dir=web_static_dir,
    )
