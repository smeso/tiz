"""Manage sandbox directories and project synchronization."""

from __future__ import annotations

import contextlib
import fcntl
import fnmatch
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import git

from tiz.log import get_logger

logger = get_logger(__name__)

META_CONTAINERS_FILE = "containers.json"
META_PROJECT_PATH_FILE = "project_path.txt"
SHARED_GENERAL_DIR = "general"
SHARED_SPECIFIC_DIR = "specific"

TIZ_PATCHES_DIR = "tiz_patches"

TIZ_COMMIT_AUTHOR_NAME = "Tiz"
TIZ_COMMIT_AUTHOR_EMAIL = "tiz@example.com"

_INVALID_NAME_RE = re.compile(r"(?:\.\.|[/\\\x00])")


class SandboxDirs:
    """Manage files and directories in a sandbox base path.

    Sandboxes are created under <base_path>/<sandbox_name>/ and
    persist after the program exits.  Each sandbox contains:
      - ``shared/`` – a directory with sub-directories for shared artefacts:
          - ``shared/general/`` – shared by all containers in the sandbox
          - ``shared/specific/<container_name>/`` – shared only by that container
      - ``project/`` – a copy of the user-supplied project directory
      - ``containers.json`` – metadata about running containers
      - ``project_path.txt`` – the original project path that was copied
    """

    _LOCK_TIMEOUT = 1
    _LOCK_POLL_INTERVAL = 0.1

    def __init__(
        self,
        sandbox_name: str,
        base_path: Path,
        project_path: str | Path | None = None,
        commit_author_name: str = TIZ_COMMIT_AUTHOR_NAME,
        commit_author_email: str = TIZ_COMMIT_AUTHOR_EMAIL,
    ) -> None:
        """Create a sandbox manager.

        Parameters
        ----------
        sandbox_name:
            Unique name for the sandbox.
        base_path:
            Base path for sandbox storage.
        project_path:
            Optional path to a project directory.  When provided the
            ``project/`` sub-directory is populated on ``create``.
        commit_author_name:
            Name used for git commits created by the sync.
        commit_author_email:
            Email used for git commits created by the sync.
        """
        if _INVALID_NAME_RE.search(sandbox_name):
            raise ValueError(
                f"Sandbox name '{sandbox_name}' contains invalid "
                f"characters (no '..', path separators, or null bytes)."
            )
        self._sandbox_name = sandbox_name
        self._base_path = base_path
        self._sandbox_dir = self._base_path / sandbox_name
        self._shared_dir = self._sandbox_dir / "shared"
        self._shared_general_dir = self._shared_dir / SHARED_GENERAL_DIR
        self._project_dir = self._sandbox_dir / "project"
        self._meta_containers_path = self._sandbox_dir / META_CONTAINERS_FILE
        self._meta_project_path_path = self._sandbox_dir / META_PROJECT_PATH_FILE
        self._commit_author_name = commit_author_name
        self._commit_author_email = commit_author_email
        self._lock_path = self._sandbox_dir / ".lock"

        if self._sandbox_dir.exists():
            if self._meta_project_path_path.exists():
                saved_path = self._meta_project_path_path.read_text(
                    encoding="utf-8"
                ).strip()
                if not saved_path:
                    raise ValueError(
                        f"Sandbox '{self._sandbox_name}' has an empty project path"
                    )
                self._original_project_path: Path | None = Path(saved_path).resolve()
            else:
                self._original_project_path = None
            if (
                project_path is not None
                and self._original_project_path != Path(project_path).resolve()
            ):
                raise ValueError(
                    f"Sandbox '{self._sandbox_name}' was created with project path "
                    f"'{self._original_project_path}', but '{project_path}' was provided"
                )
        else:
            self._original_project_path = (
                Path(project_path).resolve() if project_path else None
            )
        if (
            self._original_project_path is not None
            and not self._original_project_path.exists()
        ):
            if self._sandbox_dir.exists():
                logger.warning(
                    "Original project path '%s' does not exist (sandbox '%s' exists).",
                    self._original_project_path,
                    self._sandbox_name,
                )
                self._original_project_path = None
            else:
                raise FileNotFoundError(
                    f"Original project path '{self._original_project_path}' does not exist"
                )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def sandbox_lock(self) -> Iterator[None]:
        """Context manager that acquires an exclusive lock for the sandbox."""
        try:
            fd = os.open(
                str(self._lock_path),
                os.O_CREAT | os.O_WRONLY,
                0o600,
            )
        except OSError as exc:
            raise FileNotFoundError(
                f"Sandbox directory '{self._sandbox_dir}' does not exist. "
                "Call create() first."
            ) from exc
        try:
            timeout = time.monotonic() + self._LOCK_TIMEOUT
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() > timeout:
                        raise TimeoutError(
                            f"Could not acquire lock for sandbox '{self._sandbox_name}'"
                        ) from None
                    time.sleep(self._LOCK_POLL_INTERVAL)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def validate_git_project_dir(self) -> None:
        """Validate the sandbox project directory's git repository.

        If ``.git`` exists it is checked for symlinks, then the
        repository is validated and hooks are removed.
        """
        if not self._project_dir.exists():
            return
        git_dir = self._project_dir / ".git"
        if git_dir.exists():
            if git_dir.is_symlink():
                raise ValueError(
                    f".git directory at '{git_dir}' is a symlink; refusing to use."
                )
            for item in git_dir.rglob("*"):
                if item.is_symlink():
                    raise ValueError(
                        f"Symlink found inside .git directory: '{item}'; "
                        "refusing to use."
                    )
            self.remove_git_hooks()
            if not self.is_git_repo(self._project_dir):
                raise ValueError(
                    f"Directory '{self._project_dir}' contains a .git directory "
                    "but is not a valid git repository."
                )
            self._check_git_protocol_file_allow(self._project_dir)
            self._check_git_submodules_protocol_file_allow()

    @staticmethod
    def _check_git_protocol_file_allow(repo_path: Path) -> None:
        """Check that ``protocol.file.allow`` is unset or set to ``never``."""
        try:
            repo = git.Repo(repo_path)
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            return
        try:
            try:
                value = repo.git.config("--local", "--get", "protocol.file.allow")
            except git.GitCommandError:
                return
            if value and value.strip().lower() != "never":
                raise ValueError(
                    f"Git config 'protocol.file.allow' is set to '{value.strip()}' "
                    f"in repository at '{repo_path}'; must be 'never' or unset."
                )
        finally:
            repo.close()

    def _check_git_submodules_protocol_file_allow(self) -> None:
        """Check ``protocol.file.allow`` in all submodules."""
        repo = git.Repo(self._project_dir)
        try:
            for submodule in repo.submodules:
                sub_path = self._project_dir / submodule.path
                if sub_path.exists():
                    self._check_git_protocol_file_allow(sub_path)
        finally:
            repo.close()

    @staticmethod
    def is_git_repo(path: Path) -> bool:
        try:
            repo = git.Repo(path)
            repo.close()
            return True
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            return False

    @staticmethod
    def _matches_gitignore(
        rel: Path, patterns: list[str], is_dir: bool = False
    ) -> bool:
        """Return ``True`` if *rel* matches any gitignore pattern."""
        if not patterns:
            return False

        def _last_part_matches(
            rel: Path, effective: str, dir_only: bool, is_dir: bool
        ) -> bool:
            """Return True if *rel* matches the effective pattern."""
            if "/" not in effective and effective.lstrip("*") != "":
                for i, part in enumerate(rel.parts):
                    if fnmatch.fnmatch(part, effective) and (
                        not dir_only or i < len(rel.parts) - 1 or is_dir
                    ):
                        return True
                return False
            elif effective.startswith("**/"):
                rest = effective[3:]
                for i in range(len(rel.parts)):
                    rel_suffix = str(Path(*rel.parts[i:]))
                    if fnmatch.fnmatch(rel_suffix, rest):
                        return True
                return False
            elif "**/" in effective:
                prefix, suffix = effective.split("**/", 1)
                prefix_parts = [p for p in prefix.split("/") if p]
                remaining_parts = []
                if len(rel.parts) >= len(prefix_parts) and all(
                    fnmatch.fnmatch(rel.parts[i], prefix_parts[i])
                    for i in range(len(prefix_parts))
                ):
                    remaining_parts = list(rel.parts[len(prefix_parts) :])
                if remaining_parts:
                    for j in range(len(remaining_parts)):
                        candidate = str(Path(*remaining_parts[j:]))
                        if fnmatch.fnmatch(candidate, suffix):
                            suffix_parts = suffix.split("/")
                            # If suffix doesn't consume all of remaining_parts,
                            # matched item is a directory (has children after it)
                            if (
                                not dir_only
                                or j + len(suffix_parts) < len(remaining_parts)
                                or is_dir
                            ):
                                return True
                return False
            else:
                pattern_parts = effective.split("/")
                return (
                    len(rel.parts) == len(pattern_parts)
                    and all(
                        fnmatch.fnmatch(rel.parts[i], pattern_parts[i])
                        for i in range(len(pattern_parts))
                    )
                    and (not dir_only or is_dir)
                )

        result = False
        for pattern in patterns:
            negation = pattern.startswith("!")
            effective = (pattern[1:] if negation else pattern).strip()
            if not effective:
                continue
            dir_only = effective.endswith("/")
            if dir_only:
                effective = effective.rstrip("/")
            effective = effective.lstrip("/")
            if _last_part_matches(rel, effective, dir_only, is_dir):
                result = not negation
        return result

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def create(self, *, force_copy_files: list[str] | None = None) -> None:
        """Create the sandbox directory structure.

        Parameters
        ----------
        force_copy_files:
            List of file patterns to copy even if untracked or in
            ``.gitignore``.
        """
        logger.info(
            "Creating sandbox '%s' at %s", self._sandbox_name, self._sandbox_dir
        )
        try:
            self._sandbox_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Sandbox '{self._sandbox_name}' already exists at {self._sandbox_dir}"
            ) from exc
        with self.sandbox_lock():
            self._shared_general_dir.mkdir(parents=True, mode=0o700, exist_ok=False)

            if self._original_project_path is not None:
                self._copy_project(
                    self._original_project_path, force_copy_files=force_copy_files
                )
                self._meta_project_path_path.write_text(
                    str(self._original_project_path), encoding="utf-8"
                )

            self._write_containers_meta([])

    def _copy_project(
        self,
        src: Path,
        *,
        force_copy_files: list[str] | None = None,
    ) -> None:
        """Copy *src* into ``project/``.

        If *src* is a git repository it is cloned via GitPython
        (without hard-links so the sandbox is fully independent).
        Otherwise a plain copy is performed.

        Parameters
        ----------
        force_copy_files:
            List of file patterns to copy even if they are untracked
            or listed in ``.gitignore``.
        """
        if self.is_git_repo(src):
            orig_repo = git.Repo(src)
            if orig_repo.is_dirty(untracked_files=True):
                raise RuntimeError(f"Original repository at '{src}' is dirty")
            orig_repo.close()
            cloned_repo = git.Repo.clone_from(
                str(src), str(self._project_dir), multi_options=["--recurse-submodules"]
            )
            try:
                with cloned_repo.config_writer() as config:
                    config.set_value("user", "name", self._commit_author_name)
                    config.set_value("user", "email", self._commit_author_email)
            finally:
                cloned_repo.close()
            if force_copy_files:
                self._force_copy_files(src, self._project_dir, force_copy_files)
        else:
            self._copy_project_with_gitignore(
                src, self._project_dir, force_copy_files=force_copy_files
            )

    def _copy_project_with_gitignore(
        self, src: Path, dst: Path, *, force_copy_files: list[str] | None = None
    ) -> None:
        """Copy *src* into *dst*, honoring any ``.gitignore`` files."""
        ignore_patterns = self._read_gitignore_patterns(src)
        shutil.copytree(
            src,
            dst,
            ignore=lambda dir_path, names: {
                name
                for name in names
                if self._matches_gitignore(
                    Path(dir_path).joinpath(name).relative_to(src),
                    ignore_patterns,
                    is_dir=(
                        not (Path(dir_path) / name).is_symlink()
                        and (Path(dir_path) / name).is_dir()
                    ),
                )
            },
        )
        if force_copy_files:
            self._force_copy_files(src, dst, force_copy_files)

    def _force_copy_files(self, src: Path, dst: Path, patterns: list[str]) -> None:
        """Copy files matching *patterns* from *src* into *dst*,
        bypassing git tracking and .gitignore rules.
        """
        for pattern in patterns:
            for item in self._rglob_no_symlinks(src, pattern):
                if item.is_dir():
                    continue
                rel = item.relative_to(src)
                dst_file = dst / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(item), str(dst_file), follow_symlinks=False)

    @staticmethod
    def _rglob_no_symlinks(root: Path, pattern: str) -> list[Path]:
        """Like ``Path.rglob`` but does not follow directory symlinks.

        *pattern* is matched against the relative path from *root*, not just
        the file name.
        """
        results: list[Path] = []
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                for entry in list(current.iterdir()):
                    if entry.is_symlink():
                        rel = entry.relative_to(root)
                        if fnmatch.fnmatch(str(rel), pattern):
                            results.append(entry)
                        continue
                    if entry.is_dir():
                        stack.append(entry)
                    else:
                        rel = entry.relative_to(root)
                        if fnmatch.fnmatch(str(rel), pattern):
                            results.append(entry)
            except PermissionError:
                continue
        return results

    @staticmethod
    def _read_gitignore_patterns(src: Path) -> list[str]:
        """Return patterns from ``.gitignore`` in *src*."""
        patterns = [f"{TIZ_PATCHES_DIR}"]
        gitignore = src / ".gitignore"
        if not gitignore.exists():
            return patterns
        for line in gitignore.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped)
        return patterns

    def exists(self) -> bool:
        """Return ``True`` if the sandbox directory exists."""
        return self._sandbox_dir.exists()

    def remove(self, *, force: bool = False) -> None:
        """Delete the sandbox directory."""
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            raise RuntimeError(
                "shutil.rmtree is vulnerable to symlink attacks "
                "on this platform; refusing to remove sandbox."
            )
        with self.sandbox_lock():
            if not force and self._meta_containers_path.exists():
                try:
                    meta = json.loads(
                        self._meta_containers_path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError:
                    raise RuntimeError(
                        "Sandbox container metadata is corrupted. "
                        "Use force=True to remove."
                    ) from None
                if meta:
                    raise RuntimeError(
                        "Sandbox container metadata is not empty. Use force=True to remove."
                    )
            shutil.rmtree(self._sandbox_dir)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @classmethod
    def list_all(cls, base_path: Path) -> list[str]:
        """Return sorted names of all existing sandboxes.

        Parameters
        ----------
        base_path:
            Base path to list sandboxes from.
        """
        if not base_path.exists():
            return []
        return sorted(
            entry.name
            for entry in base_path.iterdir()
            if entry.is_dir() and not _INVALID_NAME_RE.search(entry.name)
        )

    @property
    def sandbox_name(self) -> str:
        return self._sandbox_name

    @property
    def project_dir(self) -> Path:
        return self._project_dir

    @property
    def shared_general_dir(self) -> Path:
        return self._shared_general_dir

    def shared_container_dir(self, container_name: str) -> Path:
        """Return the per-container shared directory."""
        if _INVALID_NAME_RE.search(container_name):
            raise ValueError(
                f"Container name '{container_name}' contains invalid "
                f"characters (no '..', path separators, or null bytes)."
            )
        return self._shared_dir / SHARED_SPECIFIC_DIR / container_name

    @property
    def original_project_path(self) -> Path | None:
        return self._original_project_path

    @staticmethod
    def _remove_hooks_from_dir(hooks_dir: Path) -> None:
        """Remove all non-sample hook files from *hooks_dir*."""
        if not hooks_dir.exists():
            return
        if hooks_dir.is_symlink():
            hooks_dir.unlink()
            return
        for hook in hooks_dir.iterdir():
            if hook.is_file() and not hook.name.endswith(".sample"):
                hook.unlink()

    def remove_git_hooks(self) -> None:
        """Remove all git hooks from the sandbox project directory
        and all its submodules."""
        if not self.is_git_repo(self._project_dir):
            return
        hooks_dir = self._project_dir / ".git" / "hooks"

        if hooks_dir.is_symlink():
            hooks_dir.unlink()
            return
        if not hooks_dir.exists():
            return

        project_dir_resolved = self._project_dir.resolve()
        hooks_dir_resolved = hooks_dir.resolve()

        try:
            hooks_dir_resolved.relative_to(project_dir_resolved)
        except ValueError:
            raise RuntimeError(
                f"Git hooks directory '{hooks_dir}' resolves outside the project "
                f"directory '{project_dir_resolved}'; refusing to remove hooks."
            ) from None

        self._remove_hooks_from_dir(hooks_dir_resolved)

        modules_dir = hooks_dir_resolved.parent / "modules"
        if (
            modules_dir.exists()
            and modules_dir.is_dir()
            and not modules_dir.is_symlink()
        ):
            for submodule_dir in modules_dir.iterdir():
                sub_hooks_dir = submodule_dir / "hooks"
                self._remove_hooks_from_dir(sub_hooks_dir)

    # ------------------------------------------------------------------
    # container metadata
    # ------------------------------------------------------------------

    @property
    def containers_meta(self) -> list[dict[str, Any]]:
        if not self._meta_containers_path.exists():
            return []
        with self.sandbox_lock():
            return self._containers_meta

    @property
    def _containers_meta(self) -> list[dict[str, Any]]:
        if not self._meta_containers_path.exists():
            return []
        data = json.loads(self._meta_containers_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError("Expected list in container metadata")
        return data

    def add_container_meta(self, container_info: dict[str, Any]) -> None:
        if "container_id" not in container_info:
            raise ValueError("container_info must contain 'container_id' key")
        allowed_keys = {
            "container_id",
            "engine",
            "network",
            "container_name",
        }
        filtered = {k: v for k, v in container_info.items() if k in allowed_keys}
        with self.sandbox_lock():
            containers = self._containers_meta
            containers.append(filtered)
            self._write_containers_meta(containers)

    def remove_container_meta(self, container_id: str) -> None:
        with self.sandbox_lock():
            containers = self._containers_meta
            containers = [
                c for c in containers if c.get("container_id") != container_id
            ]
            self._write_containers_meta(containers)

    def replace_containers_meta(self, containers: list[dict[str, Any]]) -> None:
        """Replace all container metadata with a new list.

        Parameters
        ----------
        containers:
            The complete list of container metadata dictionaries.
        """
        with self.sandbox_lock():
            self._write_containers_meta(containers)

    def _write_containers_meta(self, containers: list[dict[str, Any]]) -> None:
        tmp = self._meta_containers_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(containers, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self._meta_containers_path)

    # ------------------------------------------------------------------
    # sync  original -> sandbox
    # ------------------------------------------------------------------

    def sync_from_original(self, *, force: bool = False) -> None:
        """Sync changes from the original project directory to the sandbox.

        Raises ``RuntimeError`` if the sync would overwrite local
        modifications in the sandbox unless *force* is ``True``.
        """
        original_path = self._original_project_path
        if original_path is None or not original_path.exists():
            raise ValueError("Original project path not found")

        if self.is_git_repo(original_path):
            repo = git.Repo(original_path)
            try:
                if repo.is_dirty(untracked_files=True):
                    raise RuntimeError("Original project has uncommitted changes.")
            finally:
                repo.close()

        if not self._project_dir.exists():
            raise RuntimeError("Sandbox project directory does not exist")

        self.validate_git_project_dir()

        if self.is_git_repo(self._project_dir) and self.is_git_repo(original_path):
            self._sync_git_from_original(original_path, force=force)
        else:
            self._sync_copy_from_original(original_path)

    def _sync_git_from_original(
        self, original_path: Path, *, force: bool = False
    ) -> None:
        """Pull changes from *original_path* into the sandbox's git repo.

        Resets or merges every local branch to match the corresponding
        branch in the original, then checks out the same branch as the
        original repository.

        When *force* is ``False`` a fast-forward pull or a clean rebase is attempted.
        When *force* is ``True`` the sandbox is hard-reset to match
        the original.
        """
        remote_name = "tiz-original-tmp"

        orig_repo = git.Repo(original_path)
        try:
            try:
                orig_branch = orig_repo.active_branch.name
            except TypeError:
                raise RuntimeError(
                    f"Original repository at '{original_path}' is in detached HEAD state; "
                    f"cannot sync."
                ) from None
        finally:
            orig_repo.close()

        repo = git.Repo(self._project_dir)
        prev_branch = repo.active_branch.name
        try:
            if repo.is_dirty(untracked_files=True):
                repo.git.add("--all")
                if repo.index.diff("HEAD"):
                    repo.git.commit(
                        "-m",
                        "tiz sync",
                        "--author",
                        f"{self._commit_author_name} <{self._commit_author_email}>",
                    )
            initial_remote = next(
                (r for r in repo.remotes if r.name == remote_name), None
            )
            if initial_remote is not None:
                repo.delete_remote(initial_remote)
            remote = repo.create_remote(remote_name, str(original_path))
            remote.fetch()

            for branch in repo.heads:
                branch_name = branch.name
                remote_ref_prefix = f"{remote_name}/{branch_name}"
                matching_refs = [r for r in remote.refs if r.name == remote_ref_prefix]
                if not matching_refs:
                    continue
                target_ref = matching_refs[0]
                repo.git.checkout(branch_name)
                if force:
                    repo.git.reset("--hard", target_ref.commit.hexsha)
                else:
                    try:
                        repo.git.merge(
                            "--ff-only", target_ref.commit.hexsha, "--no-edit"
                        )
                    except git.GitCommandError:
                        try:
                            repo.git.rebase(target_ref.commit.hexsha)
                        except git.GitCommandError as exc:
                            repo.git.rebase("--abort")
                            repo.git.checkout(prev_branch)
                            raise RuntimeError(
                                f"Could not merge or rebase branch '{branch_name}' "
                                f"from original: {exc.stderr}"
                            ) from exc
            repo.git.checkout(orig_branch)
            repo.git.submodule("update", "--init", "--recursive")
        finally:
            with contextlib.suppress(git.GitCommandError):
                repo.delete_remote(repo.remote(remote_name))
            repo.close()

    def _sync_copy_from_original(self, original_path: Path) -> None:
        self._copytree_update(original_path, self._project_dir)

    def _copytree_update(self, src: Path, dst: Path) -> None:
        """Copy *src* into *dst*, only writing files whose mtime differs."""
        ignore_patterns = self._read_gitignore_patterns(src)
        src_files = set()
        for item in src.rglob("*"):
            if item.is_dir():
                continue
            rel = item.relative_to(src)
            if self._matches_gitignore(rel, ignore_patterns):
                continue
            src_files.add(rel)
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_file.exists():
                src_mtime = item.stat().st_mtime
                dst_mtime = dst_file.stat().st_mtime
                src_size = item.stat().st_size
                dst_size = dst_file.stat().st_size
                if src_size == dst_size and math.isclose(
                    src_mtime, dst_mtime, rel_tol=0, abs_tol=0.01
                ):
                    continue
            shutil.copy2(str(item), str(dst_file), follow_symlinks=False)

        if dst.exists():
            for item in list(dst.rglob("*")):
                if item.is_dir():
                    continue
                rel = item.relative_to(dst)
                if rel not in src_files and not self._matches_gitignore(
                    rel, ignore_patterns
                ):
                    item.unlink()

            for item in sorted(dst.rglob("*"), reverse=True):
                if item.is_dir() and not any(item.iterdir()):
                    with contextlib.suppress(OSError):
                        item.rmdir()

        dst.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # sync  sandbox -> original
    # ------------------------------------------------------------------

    def list_sandbox_only_branches(self) -> set[str]:
        """Return names of local branches in the sandbox that are not
        in the original project repository.
        """
        self.validate_git_project_dir()
        original_path = self._original_project_path
        if original_path is None or not original_path.exists():
            raise ValueError("Original project path not found")
        if not self.is_git_repo(self._project_dir):
            return set()
        if not self.is_git_repo(original_path):
            return set()

        orig_repo = git.Repo(original_path)
        try:
            orig_branches = {h.name for h in orig_repo.heads}
        finally:
            orig_repo.close()

        sandbox_repo = git.Repo(self._project_dir)
        try:
            sandbox_branches = {h.name for h in sandbox_repo.heads}
            return sandbox_branches - orig_branches
        finally:
            sandbox_repo.close()

    def sync_to_original_auto_rebase(self) -> None:
        """Sync sandbox changes to the original, automatically rebasing
        if a fast-forward conflict occurs.

        On failure due to non-fast-forward, it pulls from the original
        into the sandbox and retries once.
        """
        if self._original_project_path is None:
            return
        try:
            self.sync_to_original()
        except RuntimeError as exc:
            if "Could not fast forward from sandbox" in str(exc):
                self.sync_from_original()
                self.sync_to_original()
            else:
                raise

    def may_need_to_sync_to_original(self) -> bool:
        """Return ``True`` if the sandbox may have changes that should be
        synced back to the original project directory.

        For git repositories this checks whether the HEAD commit of the
        current branch in the sandbox matches the same branch in the
        original repository.

        For non-git directories it checks whether a patch would be produced
        by sync_to_original.
        """
        original_path = self._original_project_path
        if original_path is None or not original_path.exists():
            return False
        if not self._project_dir.exists():
            return False

        if self.is_git_repo(self._project_dir) and self.is_git_repo(original_path):
            return self._git_may_need_to_sync(original_path)
        return self._non_git_may_need_to_sync(original_path)

    def _git_may_need_to_sync(self, original_path: Path) -> bool:
        self.validate_git_project_dir()
        sandbox_repo = git.Repo(self._project_dir)
        try:
            if sandbox_repo.is_dirty(untracked_files=True):
                return True
            try:
                sandbox_branch_name = sandbox_repo.active_branch.name
            except TypeError:
                return True
            head_commit = sandbox_repo.head.commit.hexsha
        finally:
            sandbox_repo.close()

        orig_repo = git.Repo(original_path)
        try:
            orig_branches = {h.name for h in orig_repo.heads}
            if sandbox_branch_name not in orig_branches:
                return True
            orig_head_commit = orig_repo.commit(sandbox_branch_name).hexsha
            return head_commit != orig_head_commit
        except (git.GitCommandError, TypeError):
            return True
        finally:
            orig_repo.close()

    def _non_git_may_need_to_sync(self, original_path: Path) -> bool:
        return bool(self._sync_patch_helper(original_path))

    def sync_to_original(
        self, *, force: bool = False, all_branches: bool = False
    ) -> None:
        """Sync changes in the sandbox project dir back to the original
        code base.

        When **both** the sandbox and the original are git repositories
        commits are pulled from the sandbox into the original.
        Otherwise a unified diff patch is saved inside ``tiz_patches/``
        in the original project directory.

        Raises ``RuntimeError`` if the sync would overwrite local
        modifications unless *force* is ``True``.
        """
        original_path = self._original_project_path
        if original_path is None or not original_path.exists():
            raise ValueError("Original project path not found")

        self.validate_git_project_dir()

        if self.is_git_repo(self._project_dir) and self.is_git_repo(original_path):
            self._sync_git_pull_from_sandbox(
                original_path, force=force, all_branches=all_branches
            )
        else:
            self._sync_patch(original_path)

    def _sync_git_pull_from_sandbox(
        self,
        original_path: Path,
        *,
        force: bool = False,
        all_branches: bool = False,
    ) -> None:
        """Pull changes from the sandbox repo into *original_path*.

        Commits local sandbox changes, adds the sandbox as a temporary
        remote to the original repo, and fetches/merges.

        Parameters
        ----------
        force:
            When ``True`` perform hard resets instead of merges.
        all_branches:
            When ``True`` pull all branches from the sandbox instead of
            only the currently active one.
        """
        sandbox_repo = git.Repo(self._project_dir)
        try:
            if sandbox_repo.is_dirty(untracked_files=True):
                sandbox_repo.git.add("--all")
                if sandbox_repo.index.diff("HEAD"):
                    sandbox_repo.git.commit(
                        "-m",
                        "tiz sync",
                        "--author",
                        f"{self._commit_author_name} <{self._commit_author_email}>",
                    )
            try:
                sandbox_branch_name = sandbox_repo.active_branch.name
            except TypeError:
                sandbox_branch_name = (
                    f"tiz-detached-{int(time.time())}-{time.monotonic_ns()}"
                )
                sandbox_repo.git.branch(sandbox_branch_name)
                sandbox_repo.git.checkout(sandbox_branch_name)

            remote_name = "tiz-sandbox-tmp"
            orig_repo = git.Repo(original_path)
            try:
                if orig_repo.is_dirty(untracked_files=True):
                    raise RuntimeError(
                        f"Original repository at '{original_path}' is dirty"
                    )
                try:
                    orig_branch_name = orig_repo.active_branch.name
                except TypeError:
                    raise RuntimeError(
                        f"Original repository at '{original_path}' is in detached HEAD state; "
                        f"cannot sync."
                    ) from None
                try:
                    initial_remote = next(
                        (r for r in orig_repo.remotes if r.name == remote_name), None
                    )
                    if initial_remote is not None:
                        orig_repo.delete_remote(initial_remote)
                    orig_repo.create_remote(remote_name, str(self._project_dir))
                    remote = orig_repo.remote(remote_name)
                    remote.fetch()
                    if not all_branches:
                        remote_ref_prefix = f"{remote_name}/{orig_branch_name}"
                        matching_refs = [
                            r for r in remote.refs if r.name == remote_ref_prefix
                        ]
                        if matching_refs:
                            remote_head = matching_refs[0]
                            if force:
                                orig_repo.git.reset("--hard", remote_head.commit.hexsha)
                            else:
                                try:
                                    orig_repo.git.merge(
                                        "--ff-only",
                                        remote_head.commit.hexsha,
                                        "--no-edit",
                                    )
                                except git.GitCommandError as exc:
                                    raise RuntimeError(
                                        f"Could not fast forward from sandbox: {exc.stderr}"
                                    ) from None
                    else:
                        orig_repo_heads = {h.name for h in orig_repo.heads}
                        for ref in remote.refs:
                            branch_to_sync = ref.name.split("/")[-1]
                            if branch_to_sync == "HEAD":
                                continue
                            if branch_to_sync in orig_repo_heads:
                                orig_repo.git.checkout(branch_to_sync)
                            else:
                                orig_repo.git.checkout("-b", branch_to_sync)
                            if force:
                                orig_repo.git.reset("--hard", ref.commit.hexsha)
                            else:
                                try:
                                    orig_repo.git.merge(
                                        "--ff-only", ref.commit.hexsha, "--no-edit"
                                    )
                                except git.GitCommandError:
                                    continue
                    orig_repo.git.submodule("update", "--init", "--recursive")
                finally:
                    cleanup_remote = next(
                        (r for r in orig_repo.remotes if r.name == remote_name), None
                    )
                    if cleanup_remote is not None:
                        orig_repo.delete_remote(cleanup_remote)
                    orig_repo.git.checkout(orig_branch_name)
            finally:
                orig_repo.close()
        finally:
            sandbox_repo.close()

    def _sync_patch_helper(self, original_path: Path) -> str | None:
        """Create a unified diff patch."""
        tmp_dir_orig = Path(tempfile.mkdtemp(prefix="tiz_patch_"))
        tmp_orig = tmp_dir_orig / "a"
        tmp_sandbox = tmp_dir_orig / "b"
        try:
            self._copy_project_with_gitignore(original_path, tmp_orig)
            self._copy_project_with_gitignore(self._project_dir, tmp_sandbox)

            diff_result = subprocess.run(
                [
                    "diff",
                    "-ruNp",
                    "--no-dereference",
                    "a/",
                    "b/",
                ],
                cwd=tmp_dir_orig,
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "LC_ALL": "C"},
            )

            if diff_result.returncode == 0 and not diff_result.stdout:
                return None
            if diff_result.returncode == 2:
                raise RuntimeError(f"diff command failed: {diff_result.stderr.strip()}")

            return diff_result.stdout
        finally:
            shutil.rmtree(tmp_dir_orig, ignore_errors=True)

    def git_capture_branch(self) -> str:
        """Capture the current git branch state.

        If the repository is in a detached HEAD state, a branch is
        created from the current commit.  Returns the name of the
        current (or newly created) branch.
        """
        repo = git.Repo(self._project_dir)
        try:
            original_branch = repo.active_branch.name
        except TypeError:
            original_branch = f"tiz-detached-{int(time.time())}-{time.monotonic_ns()}"
            repo.git.branch(original_branch)
            repo.git.checkout(original_branch)
        finally:
            repo.close()
        return original_branch

    def git_create_branch(self, branch_name: str) -> None:
        """Create and check out a new branch."""
        repo = git.Repo(self._project_dir)
        try:
            repo.git.checkout("-b", branch_name)
        finally:
            repo.close()

    def git_checkout(self, branch_name: str) -> None:
        """Check out an existing branch."""
        repo = git.Repo(self._project_dir)
        try:
            repo.git.checkout(branch_name)
        finally:
            repo.close()

    def git_finalize_branches(
        self,
        original_branch: str,
        winner: str,
        branches: list[str],
    ) -> None:
        """Finalize branch selection.

        Checks out the original branch, resets it to the winning
        branch, and deletes all temporary branches.
        """
        repo = git.Repo(self._project_dir)
        try:
            repo.git.checkout(original_branch)
            with contextlib.suppress(git.GitCommandError):
                repo.git.reset("--hard", winner)
            for branch in branches:
                with contextlib.suppress(git.GitCommandError):
                    repo.delete_head(branch, force=True)
        finally:
            repo.close()

    def _sync_patch(self, original_path: Path) -> None:
        """Create a unified diff patch in *original_path*/tiz_patches/."""
        patches_dir = original_path / TIZ_PATCHES_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        patch_name = f"tiz_sync_{timestamp}.patch"
        patch_path = patches_dir / patch_name
        patch = self._sync_patch_helper(original_path)
        if patch:
            patches_dir.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(patch, encoding="utf-8")
