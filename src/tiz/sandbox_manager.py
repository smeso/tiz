"""Manage all tiz sandbox containers across all sandboxes."""

from __future__ import annotations

import contextlib
import fcntl
import importlib.resources
import os
import random
import re
import shutil
import string
import subprocess
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from tiz.log import get_logger
from tiz.sandbox_container import ContainerMeta, SandboxContainer
from tiz.sandbox_dirs import (
    _INVALID_NAME_RE,
    TIZ_COMMIT_AUTHOR_EMAIL,
    TIZ_COMMIT_AUTHOR_NAME,
    SandboxDirs,
)

TIZ_WORKER_PREFIX = "tiz_"
TIZ_TOOLS_WORKER_DIR = "tools_worker"
_MAX_BUILD_RECURSION = 10
CONTAINERFILES_DIR = "containerfiles"

logger = get_logger(__name__)


class SandboxManager:
    """Manage all tiz containers across all sandboxes."""

    @staticmethod
    def available_engine() -> str | None:
        """Return ``"docker"`` or ``"podman"`` if available, else ``None``."""
        for engine in ("podman", "docker"):
            if shutil.which(engine) is not None:
                return engine
        return None

    @staticmethod
    def available_engines() -> list[str]:
        """Return the sorted list of available container engines."""
        engines = []
        for engine in ("docker", "podman"):
            if shutil.which(engine) is not None:
                engines.append(engine)
        return sorted(engines)

    def __init__(self, base_path: Path, engine: str | None = None) -> None:
        if engine is None:
            detected = SandboxManager.available_engine()
            if detected is None:
                raise RuntimeError("No container engine (podman or docker) found")
            engine = detected
        if shutil.which(engine) is None:
            raise ValueError(f"Engine '{engine}' not found in PATH")
        self._engine = engine
        self._manager_base_path = base_path
        self._base_path = base_path / "sandboxes"

    def list_sandboxes(self) -> list[str]:
        return SandboxDirs.list_all(base_path=self._base_path)

    def _iter_sandboxes(self) -> list[SandboxDirs]:
        names = self.list_sandboxes()
        result = []
        for name in names:
            try:
                result.append(SandboxDirs(sandbox_name=name, base_path=self._base_path))
            except (ValueError, FileNotFoundError, OSError) as exc:
                logger.warning("Skipping corrupted sandbox '%s': %s", name, exc)
        return result

    def get_sandbox_containers(self, sandbox_name: str) -> list[SandboxContainer]:
        """Return a list of all SandboxContainer objects for a given sandbox.

        Note: The caller should hold the manager lock for *sandbox_name* to
        ensure consistency.  Methods that call this under the lock (e.g.
        ``_kill_all_containers_locked``) are safe.
        """
        containers: list[SandboxContainer] = []
        sandbox = SandboxDirs(sandbox_name=sandbox_name, base_path=self._base_path)
        for meta in sandbox.containers_meta:
            cmeta = ContainerMeta(**meta)
            if cmeta.engine != self._engine:
                raise ValueError(
                    f"Container engine mismatch: expected '{self._engine}', "
                    f"got '{cmeta.engine}'"
                )
            container = SandboxContainer(sandbox_dirs=sandbox, meta=cmeta)
            containers.append(container)
        return containers

    def list_containers(self) -> list[dict[str, object]]:
        """Return a list of all tiz containers with their metadata."""
        result: list[dict[str, object]] = []
        for sandbox in self._iter_sandboxes():
            with self.sandbox_lock(sandbox.sandbox_name):
                for meta in sandbox.containers_meta:
                    entry: dict[str, object] = {
                        "sandbox_name": sandbox.sandbox_name,
                        **meta,
                    }
                    result.append(entry)
        return result

    def create_sandbox(
        self,
        sandbox_name: str,
        project_path: str | None = None,
        *,
        force_copy_files: list[str] | None = None,
        committer_name: str = TIZ_COMMIT_AUTHOR_NAME,
        committer_email: str = TIZ_COMMIT_AUTHOR_EMAIL,
    ) -> SandboxDirs:
        """Create a new sandbox using SandboxDirs.

        Parameters
        ----------
        sandbox_name:
            Name of the sandbox to create.
        project_path:
            Optional path to the project directory to copy into the sandbox.
        force_copy_files:
            List of file patterns to copy even if untracked or in .gitignore.
        committer_name:
            Name used for git commits created by the sync.
        committer_email:
            Email used for git commits created by the sync.
        """
        logger.info("Creating sandbox '%s'", sandbox_name)
        sandbox = SandboxDirs(
            sandbox_name=sandbox_name,
            project_path=project_path,
            base_path=self._base_path,
            commit_author_name=committer_name,
            commit_author_email=committer_email,
        )
        try:
            sandbox.create(force_copy_files=force_copy_files)
        except FileExistsError:
            with self.sandbox_lock(sandbox_name):
                if sandbox.containers_meta:
                    raise RuntimeError(
                        f"Sandbox '{sandbox_name}' already exists and has running containers"
                    ) from None
                if sandbox.original_project_path is not None:
                    sandbox.sync_from_original()
        return sandbox

    @contextlib.contextmanager
    def sandbox_lock(
        self,
        sandbox_name: str,
        timeout: float = 1.0,
        poll_interval: float = 0.1,
    ) -> Iterator[None]:
        """Context manager that acquires a manager-level exclusive lock for a sandbox.

        This uses a separate lock file (``.manager_lock``) from the one
        used by ``SandboxDirs`` (``.lock``).  The lock hierarchy is always:
        manager-level lock first, then ``SandboxDirs``-level lock.  Callers
        that already hold the manager lock must not acquire the dirs lock
        directly without being aware of this hierarchy.

        Parameters
        ----------
        sandbox_name:
            Name of the sandbox to lock.
        timeout:
            Maximum seconds to wait for the lock.
        poll_interval:
            Seconds between lock acquisition attempts.
        """
        sandbox_dir = self._base_path / sandbox_name
        lock_path = sandbox_dir / ".manager_lock"

        if _INVALID_NAME_RE.search(sandbox_name):
            raise ValueError(f"Invalid sandbox name '{sandbox_name}'")

        if not sandbox_dir.exists():
            raise FileNotFoundError(
                f"Sandbox directory '{sandbox_dir}' does not exist."
            )

        fd = os.open(
            str(lock_path),
            os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Could not acquire manager lock for sandbox '{sandbox_name}'"
                        ) from None
                    time.sleep(poll_interval)
            try:
                yield
            finally:
                sandbox_dirs = SandboxDirs(
                    sandbox_name=sandbox_name, base_path=self._base_path
                )
                sandbox_dirs.validate_git_project_dir()
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def create_container(
        self,
        sandbox_name: str,
        image: str,
        *,
        container_name: str | None = None,
        mount_project: bool = True,
        network: str = "none",
        read_only_project: bool = False,
        extra_run_args: list[str] | None = None,
        verbose: int = 0,
        use_host_timezone: bool = True,
    ) -> SandboxContainer:
        """Create and start a new sandbox container.

        Parameters
        ----------
        sandbox_name:
            Name of the sandbox to create the container in.
        image:
            Container image to use.
        container_name:
            Optional human-readable name. Must start with
            "tiz_worker_SANDBOXNAME_" and be unique within the sandbox.
        mount_project:
            Mount the project directory into the container. Set to ``False``
            to skip mounting the project directory entirely.
        network:
            Container network mode.  ``"internet"`` is accepted as an
            alias for ``"private"``.
        read_only_project:
            Mount the project directory read-only.
        extra_run_args:
            Additional arguments for the container run command.
        use_host_timezone:
            When ``True`` (default) bind-mount ``/etc/localtime`` read-only
            inside the container so it uses the host's timezone.
        """
        sandbox = SandboxDirs(sandbox_name=sandbox_name, base_path=self._base_path)
        if network == "internet":
            network = "private"
        prefix = f"{TIZ_WORKER_PREFIX}{sandbox_name[-10:]}_"

        existing_names = {
            meta.get("container_name")
            for meta in sandbox.containers_meta
            if meta.get("container_name") is not None
        }

        if container_name is None:
            for _ in range(100):
                suffix = "".join(
                    random.choices(string.ascii_lowercase + string.digits, k=3)
                )
                candidate = f"{prefix}{suffix}"
                if candidate not in existing_names:
                    container_name = candidate
                    break
            else:
                raise RuntimeError(
                    f"Could not generate unique container name for sandbox '{sandbox_name}'"
                )
        elif not container_name.startswith(prefix):
            raise ValueError(
                f"Container name must start with '{prefix}', got '{container_name}'"
            )
        elif container_name in existing_names:
            raise ValueError(f"Container name '{container_name}' is already taken")

        container = SandboxContainer(sandbox_dirs=sandbox, engine=self._engine)
        tools_worker_path: Path | None = None
        if self._manager_base_path.exists():
            tools_candidate = self._manager_base_path / TIZ_TOOLS_WORKER_DIR
            if tools_candidate.is_dir():
                tools_worker_path = tools_candidate
        container.start(
            image=image,
            container_name=container_name,
            network=network,
            read_only_project=read_only_project,
            mount_project=mount_project,
            extra_run_args=extra_run_args,
            verbose=verbose,
            use_host_timezone=use_host_timezone,
            custom_tools_dir=tools_worker_path,
        )
        return container

    def _kill_all_containers_locked(self, sandbox_name: str) -> None:
        """Kill all containers for a given sandbox.

        Caller must hold the manager lock for *sandbox_name*.
        """
        containers = self.get_sandbox_containers(sandbox_name)
        for container in containers:
            if container.container_id is not None and container.exists():
                container.stop()

    def kill_all_containers(self, sandbox_name: str) -> None:
        """Kill all containers for a given sandbox.

        Parameters
        ----------
        sandbox_name:
            Name of the sandbox whose containers should be killed.
        """
        with self.sandbox_lock(sandbox_name):
            self._kill_all_containers_locked(sandbox_name)

    def kill_and_delete_sandbox(self, sandbox_name: str) -> None:
        """Kill all containers for a given sandbox and delete the sandbox.

        Parameters
        ----------
        sandbox_name:
            Name of the sandbox to kill and delete.
        """
        with self.sandbox_lock(sandbox_name):
            self._kill_all_containers_locked(sandbox_name)
            sandbox = SandboxDirs(sandbox_name=sandbox_name, base_path=self._base_path)
            sandbox.remove(force=True)

    def kill_and_delete_all_sandboxes(self) -> None:
        """Kill all containers and delete all sandboxes."""
        for sandbox in self._iter_sandboxes():
            with contextlib.suppress(FileNotFoundError):
                self.kill_and_delete_sandbox(sandbox.sandbox_name)

    def delete_container(
        self,
        sandbox_name: str,
        container_id: str,
    ) -> None:
        """Stop and remove a container by ID.

        Parameters
        ----------
        sandbox_name:
            Name of the sandbox the container belongs to.
        container_id:
            The container ID to delete.
        """
        with self.sandbox_lock(sandbox_name):
            sandbox = SandboxDirs(sandbox_name=sandbox_name, base_path=self._base_path)
            for meta_dict in sandbox.containers_meta:
                if meta_dict.get("container_id") == container_id:
                    cmeta = ContainerMeta(**meta_dict)
                    if cmeta.engine != self._engine:
                        raise ValueError(
                            f"Container engine mismatch: expected '{self._engine}', "
                            f"got '{cmeta.engine}'"
                        )
                    container = SandboxContainer(sandbox_dirs=sandbox, meta=cmeta)
                    container.stop()
                    return
            raise FileNotFoundError(
                f"Container '{container_id}' not found in sandbox '{sandbox_name}'"
            )

    def delete_tiz_worker_images(self, dry_run: bool = False) -> list[str]:
        """Delete all images with a name that starts with 'tiz-worker'."""
        result = subprocess.run(
            [self._engine, "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []

        ret = []
        for line in result.stdout.splitlines():
            full_ref = line.strip()
            short_name = full_ref.split("/")[-1]
            if short_name.startswith("tiz-worker"):
                if not dry_run:
                    rmi_result = subprocess.run(
                        [self._engine, "rmi", "-fi", full_ref],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    if rmi_result.returncode == 0:
                        ret.append(short_name)
                else:
                    ret.append(short_name)
        return ret

    def sync_from_original(self, sandbox_name: str, *, force: bool = False) -> None:
        with self.sandbox_lock(sandbox_name):
            sandbox = SandboxDirs(sandbox_name=sandbox_name, base_path=self._base_path)
            sandbox.sync_from_original(force=force)

    def sync_to_original(self, sandbox_name: str, *, force: bool = False) -> None:
        with self.sandbox_lock(sandbox_name):
            sandbox = SandboxDirs(sandbox_name=sandbox_name, base_path=self._base_path)
            sandbox.sync_to_original(force=force)

    @staticmethod
    def get_containerfiles_dirs(base_path: Path) -> list[Path]:
        dirs = [base_path / CONTAINERFILES_DIR]
        try:
            data_dir = Path(
                str(importlib.resources.files("tiz") / "data" / CONTAINERFILES_DIR)
            )
            dirs.append(data_dir)
        except (TypeError, ModuleNotFoundError, FileNotFoundError):
            pass
        return dirs

    def _resolve_and_build(
        self,
        *,
        containerfile: str,
        tag: str,
        timestamp: datetime | None = None,
        delete_existing: bool = False,
        recursion_level: int = 0,
    ) -> None:
        if recursion_level > _MAX_BUILD_RECURSION:
            raise RuntimeError(
                f"Maximum recursion depth ({_MAX_BUILD_RECURSION}) exceeded "
                f"while resolving dependencies for image '{tag}'"
            )

        base_path = self._manager_base_path
        containerfiles_dirs = self.get_containerfiles_dirs(base_path)

        for line in containerfile.splitlines():
            m = re.match(r"^\s*FROM\s+(\S+)", line, re.IGNORECASE)
            if not m:
                continue
            image_name = m.group(1)
            if image_name.split(":")[0] == tag.split(":")[0]:
                continue
            result = subprocess.run(
                [self._engine, "image", "inspect", image_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                continue
            dep_content = None
            for cf_dir in containerfiles_dirs:
                cf_path = cf_dir / f"Containerfile.{image_name.split(':')[0]}"
                if cf_path.is_file():
                    dep_content = cf_path.read_text(encoding="utf-8")
                    break
            if dep_content is not None:
                self._resolve_and_build(
                    containerfile=dep_content,
                    tag=image_name,
                    timestamp=timestamp,
                    recursion_level=recursion_level + 1,
                )
            else:
                logger.warning(
                    "Dependency Containerfile for '%s' not found in %s",
                    image_name.split(":")[0],
                    containerfiles_dirs,
                )

        self._build_image_internal(
            containerfile=containerfile,
            tag=tag,
            timestamp=timestamp,
            delete_existing=delete_existing,
        )

    def _build_image_internal(
        self,
        *,
        containerfile: str,
        tag: str,
        timestamp: datetime | None = None,
        delete_existing: bool = False,
    ) -> None:
        if not tag.startswith("tiz-worker"):
            raise ValueError(f"Image tag must start with 'tiz-worker', got '{tag}'")
        if delete_existing:
            subprocess.run(
                [self._engine, "rmi", "-fi", tag],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            result = subprocess.run(
                [self._engine, "image", "inspect", tag],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return
        cmd = [
            self._engine,
            "build",
            "--force-rm",
            "--pull=newer",
            "--squash",
        ]
        if timestamp is not None:
            cmd.extend(["--timestamp", str(int(timestamp.timestamp()))])
        cmd.extend(["-f"])
        with tempfile.TemporaryDirectory(prefix="tiz_build_") as tmpdir:
            containerfile_path = Path(tmpdir) / "Containerfile"
            containerfile_path.write_text(containerfile, encoding="utf-8")
            cmd.append(str(containerfile_path))
            cmd.extend(["-t", tag, tmpdir])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to build image: {result.stderr.strip()}")

    def build_image(
        self,
        *,
        containerfile: str,
        tag: str = "tiz-worker:latest",
        timestamp: datetime | None = None,
        delete_existing: bool = False,
    ) -> None:
        """Build a container image from a Containerfile string.

        Parameters
        ----------
        containerfile:
            The Containerfile/Dockerfile content as a string.
        tag:
            Tag for the built image. Must start with 'tiz-worker'.
        timestamp:
            Optional datetime to use as the --timestamp argument.
        delete_existing:
            If ``True``, delete any existing image with the same tag
            before building.
        """
        if not tag.startswith("tiz-worker"):
            raise ValueError(f"Image tag must start with 'tiz-worker', got '{tag}'")
        self._resolve_and_build(
            containerfile=containerfile,
            tag=tag,
            timestamp=timestamp,
            delete_existing=delete_existing,
            recursion_level=0,
        )

    def cleanup_dead_entries(self) -> list[str]:
        """Remove dead entries from containers.json where the container no longer exists.

        Returns
        -------
        list[str]
            List of container IDs that were removed from metadata.
        """
        removed: list[str] = []
        for sandbox in self._iter_sandboxes():
            with self.sandbox_lock(sandbox.sandbox_name):
                metas = sandbox.containers_meta
                remaining = []
                for meta in metas:
                    cid = meta.get("container_id")
                    if cid is None:
                        continue
                    cmd_result = subprocess.run(
                        [self._engine, "inspect", cid],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if cmd_result.returncode == 0:
                        remaining.append(meta)
                    else:
                        removed.append(cid)
                sandbox.replace_containers_meta(remaining)
        return removed

    def cleanup_untracked_containers(self) -> list[str]:
        """Kill and remove running containers with tiz prefixes that are not in any containers.json.

        Collects tracked container IDs from all sandboxes first, then queries
        running containers once.  This avoids killing containers belonging to
        other sandboxes.

        Returns
        -------
        list[str]
            List of container IDs that were killed and removed.
        """
        untracked: list[str] = []

        # Collect tracked IDs across all sandboxes first
        all_tracked_ids: set[str] = set()
        for sandbox in self._iter_sandboxes():
            with self.sandbox_lock(sandbox.sandbox_name):
                for meta in sandbox.containers_meta:
                    cid = meta.get("container_id")
                    if cid is not None:
                        all_tracked_ids.add(cid)

        result = subprocess.run(
            [self._engine, "ps", "--format", "{{.ID}} {{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return untracked

        for line in result.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            cid = parts[0]
            if cid in all_tracked_ids:
                continue
            name = parts[1] if len(parts) > 1 else ""
            if name.startswith(TIZ_WORKER_PREFIX):
                subprocess.run(
                    [self._engine, "stop", cid],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                subprocess.run(
                    [self._engine, "rm", "-fi", cid],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                untracked.append(cid)

        return untracked
