"""Shell autocompletion for tiz CLI using argcomplete."""

from __future__ import annotations

__all__ = ["autocomplete", "shellcode"]

import argparse
import os
from pathlib import Path
from typing import Any

try:
    import argcomplete

    _HAS_ARGCOMPLETE = True
except ImportError:  # pragma: no cover
    _HAS_ARGCOMPLETE = False

from tiz.sandbox_dirs import SandboxDirs
from tiz.sandbox_manager import SandboxManager


def _resolve_config_dir(parsed_args: argparse.Namespace) -> Path:
    """Extract config_dir from parsed_args, defaulting to ``~/.tiz``."""
    config_dir = getattr(parsed_args, "config_dir", None)
    if config_dir is None:
        config_dir = Path.home() / ".tiz"
    elif not isinstance(config_dir, Path):
        config_dir = Path(config_dir)
    elif type(config_dir) is not Path:
        config_dir = Path(str(config_dir))
    return config_dir


def _sandbox_name_completer(
    prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    """Return sandbox names matching *prefix* by querying SandboxManager.

    Uses ``parsed_args.config_dir`` if available (set via ``--config-dir``),
    otherwise defaults to ``~/.tiz``.  Returns an empty list on any error
    (e.g. no container engine, missing directory).
    """
    try:
        config_dir = _resolve_config_dir(parsed_args)
        engine = SandboxManager.available_engine()
        if engine is None:
            return []
        sandboxes_dir = config_dir / "sandboxes"
        names = SandboxDirs.list_all(sandboxes_dir)
        return [name for name in names if name.startswith(prefix)]
    except Exception:
        return []


def _image_tag_completer(
    prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    """Return available image tags matching *prefix* by scanning containerfiles dirs.

    Looks for ``Containerfile.<name>`` files in the containerfiles directories
    (user config dir and package data dir). Converts them to ``<name>:latest``
    tags. Returns an empty list on any error.
    """
    try:
        config_dir = _resolve_config_dir(parsed_args)
        dirs = SandboxManager.get_containerfiles_dirs(config_dir)
        tags: list[str] = []
        for d in dirs:
            if not d.is_dir():
                continue
            for child in d.iterdir():
                if child.is_file() and child.name.startswith("Containerfile."):
                    name = child.name[len("Containerfile.") :]
                    if not name:
                        continue
                    if name.startswith(prefix):
                        tags.append(name)
        return sorted(set(tags))
    except Exception:
        return []


def _manifest_path_completer(
    prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    """Return manifest file paths matching *prefix*.

    Looks for files (any extension) in the current working directory and in
    the ``manifests/`` subdirectory under the config dir.  Returns an empty
    list on any error.
    """
    try:
        config_dir = _resolve_config_dir(parsed_args)

        prefix_path = Path(prefix)
        base_prefix = prefix_path.name
        dir_part = prefix[: len(prefix) - len(base_prefix)]

        candidates: list[str] = []

        cwd = Path.cwd()
        if cwd.is_dir():
            for child in cwd.iterdir():
                if child.is_file() and child.name.startswith(base_prefix):
                    candidates.append(dir_part + child.name)

        manifest_dir = config_dir / "manifests"
        if manifest_dir.is_dir():
            for child in manifest_dir.iterdir():
                if child.is_file() and child.name.startswith(prefix):
                    candidates.append(child.name)

        return sorted(set(candidates))
    except Exception:
        return []


def _set_sandbox_completers(parser: argparse.ArgumentParser) -> None:
    """Walk *parser* and attach completers to arguments.

    Attaches the sandbox-name completer to every ``sandbox_name`` positional,
    the image-tag completer to every ``tag`` positional, and the manifest-path
    completer to every ``manifest`` argument.
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                if isinstance(subparser, argparse.ArgumentParser):
                    _set_sandbox_completers(subparser)
        elif action.dest == "sandbox_name":
            action.completer = _sandbox_name_completer  # type: ignore[attr-defined]  # argcomplete protocol
        elif action.dest == "tag":
            action.completer = _image_tag_completer  # type: ignore[attr-defined]  # argcomplete protocol
        elif action.dest == "manifest":
            action.completer = _manifest_path_completer  # type: ignore[attr-defined]  # argcomplete protocol


def autocomplete(parser: argparse.ArgumentParser) -> None:
    """Enable shell autocompletion for the given parser using argcomplete.

    Call this before ``parser.parse_args()``. Does nothing if argcomplete
    is not installed or if ``_ARGCOMPLETE`` is not set (normal execution).
    """
    if not _HAS_ARGCOMPLETE:  # pragma: no cover
        return
    if "_ARGCOMPLETE" not in os.environ:
        return
    _set_sandbox_completers(parser)
    argcomplete.autocomplete(parser)


def shellcode(shell: str = "bash") -> str:
    """Return the shell snippet to enable tab completion for tiz.

    Parameters
    ----------
    shell : str
        One of ``"bash"``, ``"zsh"``, ``"tcsh"``, or ``"fish"``.

    Returns
    -------
    str
        Shell code that can be ``eval``\\u2019d or added to ``~/.bashrc``.
    """
    if not _HAS_ARGCOMPLETE:  # pragma: no cover
        return "# argcomplete not installed"
    return str(argcomplete.shellcode(["tiz"], shell=shell))
