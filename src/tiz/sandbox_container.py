"""Manage Docker/Podman sandbox containers."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from importlib.resources import files as _rfiles
from pathlib import Path
from typing import Literal, overload

from tiz.log import get_logger
from tiz.sandbox_dirs import SandboxDirs

logger = get_logger(__name__)

_DEFAULT_SUBPROCESS_TIMEOUT = 60

SANDBOX_WORKER_PATH = Path(str(_rfiles("tiz") / "sandbox_worker.py"))
WORKER_SCRIPTS_DIR = Path(str(_rfiles("tiz") / "worker_scripts"))
CONTAINER_MOUNT_PROJECT = "/opt/project"
CONTAINER_MOUNT_SHARED = "/opt/shared"
CONTAINER_MOUNT_OWN_SHARED = "/opt/container_shared"
CONTAINER_MOUNT_SCRIPTS = "/opt/scripts"
CONTAINER_CWD_PROJECT = CONTAINER_MOUNT_PROJECT
CONTAINER_CWD_SHARED = CONTAINER_MOUNT_OWN_SHARED
CONTAINER_MOUNT_WORKER = "/usr/local/bin/worker.py"
CONTAINER_WORKER_SOCK_BASENAME = "exe.sock"
CONTAINER_WORKER_SOCK_PATH = str(
    Path(CONTAINER_MOUNT_OWN_SHARED) / CONTAINER_WORKER_SOCK_BASENAME
)

CONTAINER_CMD_PREFIX = (
    "/bin/bash",
    "-lc",
)
DEFAULT_NETWORK = "none"
DEFAULT_IMAGE = "tiz-worker:latest"
_ALREADY_STOPPED_INDICATORS = (
    "is not running",
    "already stopped",
    "no such container",
    "not running",
)
_VALID_CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

OCI_START_HOOK_NAME = "firewall.json"
OCI_HOOK_SCRIPT_CONTENT = """\
#!/bin/sh

set -e

/usr/sbin/iptables -A OUTPUT -d 10.0.0.0/8 -j DROP
/usr/sbin/iptables -A OUTPUT -d 172.16.0.0/12 -j DROP
/usr/sbin/iptables -A OUTPUT -d 192.168.0.0/16 -j DROP
/usr/sbin/iptables -A OUTPUT -d 169.254.0.0/16 -j DROP
/usr/sbin/iptables -A OUTPUT -d 224.0.0.0/4 -j DROP

/usr/sbin/ip6tables -A OUTPUT -d fc00::/7 -j DROP
/usr/sbin/ip6tables -A OUTPUT -d fe80::/10 -j DROP
/usr/sbin/ip6tables -A OUTPUT -d ff00::/8 -j DROP
"""


@dataclasses.dataclass(frozen=True)
class ContainerMeta:
    """Serialisable metadata for a single container instance."""

    container_id: str
    engine: str
    network: str
    container_name: str | None


class SandboxContainer:
    """Represent and manage a single sandbox container."""

    def __init__(
        self,
        sandbox_dirs: SandboxDirs,
        engine: str = "podman",
        meta: ContainerMeta | None = None,
    ) -> None:
        """Create a sandbox container manager.

        Parameters
        ----------
        sandbox_dirs:
            ``SandboxDirs`` instance for this sandbox.
        engine:
            Container engine CLI to use – ``"docker"`` or ``"podman"``.
        meta:
            Optional ``ContainerMeta`` to restore container state.
            engine will be ignored if this is used.
        """
        self._container_id: str | None
        self._container_name: str | None
        if meta is not None:
            self._engine = meta.engine
            self._container_id = meta.container_id
            self._container_name = meta.container_name
        else:
            if shutil.which(engine) is None:
                raise ValueError(f"Container engine '{engine}' not found in PATH")
            self._engine = engine
            self._container_id = None
            self._container_name = None

        self._sandbox_dirs = sandbox_dirs

    def _run_subprocess(
        self, cmd: list[str], timeout: int = _DEFAULT_SUBPROCESS_TIMEOUT
    ) -> subprocess.CompletedProcess[str]:
        """Run a subprocess and return the result."""
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        read_only_project: bool = False,
        image: str = DEFAULT_IMAGE,
        network: str = DEFAULT_NETWORK,
        *,
        container_name: str,
        verbose: int = 0,
        extra_run_args: list[str] | None = None,
        oci_hooks: bool = True,
        mount_project: bool = True,
        use_host_timezone: bool = True,
        custom_tools_dir: Path | None = None,
        tmpfs_root: bool = False,
    ) -> None:
        """Start a new sandbox container.

        Parameters
        ----------
        read_only_project:
            Mount the project directory read-only when ``True``.
        image:
            Container image to use.
        network:
            Container network mode.
        container_name:
            Human-readable container name.
        verbose:
            Verbosity level passed to worker.py (0, 1 for -v, 2 for -vv).
        extra_run_args:
            Additional arguments to pass to the container run command.
        oci_hooks:
            When ``True`` (default) OCI hooks are configured to restrict
            outgoing network traffic via iptables.
        mount_project:
            When ``True`` (default) mount the project directory into
            the container. Set to ``False`` to skip the mount entirely.
        use_host_timezone:
            When ``True`` (default) bind-mount ``/etc/localtime`` read-only
            inside the container so it uses the host's timezone.
        custom_tools_dir:
            Optional path to a tools_worker directory on the host to
            mount read-only at ``/opt/tiz_tools`` inside the container.
        tmpfs_root:
            When ``True``, use ``--image-volume=tmpfs`` instead of
            ``--read-only`` so that writable tmpfs overlays are used
            for the container rootfs.
        """
        if self._container_id is not None:
            raise RuntimeError("Container already started")
        if not _VALID_CONTAINER_NAME_RE.match(container_name):
            raise ValueError(
                f"Invalid container name {container_name!r}: "
                "must start with A-Za-z0-9 and contain only A-Za-z0-9_.- characters"
            )
        logger.info("Starting container '%s' with image '%s'", container_name, image)
        project_dir = self._sandbox_dirs.project_dir
        shared_general_dir = self._sandbox_dirs.shared_general_dir
        own_shared_dir = self._sandbox_dirs.shared_container_dir(container_name)
        own_shared_dir.mkdir(parents=True, exist_ok=True)

        rootfs_arg = "--image-volume=tmpfs" if tmpfs_root else "--read-only"
        cmd = [
            self._engine,
            "run",
            "--pull=never",
            "-d",
            "--rm",
            "--no-hosts",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            rootfs_arg,
            "--userns=keep-id:uid=1000,gid=1000",
        ]
        cmd.extend(["--name", container_name])
        cmd.extend(["--network", network])

        should_mount_project = mount_project and project_dir.exists()
        if should_mount_project:
            project_ro = "ro" if read_only_project else "rw"
            cmd.extend(
                [
                    "--mount",
                    f"type=bind,source={project_dir},target={CONTAINER_MOUNT_PROJECT},{project_ro},nosuid,nodev",
                ]
            )
        cmd.extend(
            [
                "--mount",
                f"type=bind,source={shared_general_dir},target={CONTAINER_MOUNT_SHARED},rw,nosuid,nodev",
            ]
        )
        cmd.extend(
            [
                "--mount",
                f"type=bind,source={SANDBOX_WORKER_PATH},target={CONTAINER_MOUNT_WORKER},ro",
            ]
        )
        cmd.extend(
            [
                "--mount",
                f"type=bind,source={WORKER_SCRIPTS_DIR},target={CONTAINER_MOUNT_SCRIPTS},ro,nosuid,nodev",
            ]
        )
        cmd.extend(
            [
                "--mount",
                f"type=bind,source={own_shared_dir},target={CONTAINER_MOUNT_OWN_SHARED},rw,nosuid,nodev",
            ]
        )

        if use_host_timezone:
            cmd.extend(
                [
                    "--mount",
                    "type=bind,source=/etc/localtime,target=/etc/localtime,ro",
                ]
            )

        if custom_tools_dir is not None:
            cmd.extend(
                [
                    "--mount",
                    f"type=bind,source={custom_tools_dir},target=/opt/tiz_tools,ro,nosuid,nodev",
                ]
            )

        cwd = CONTAINER_CWD_PROJECT if should_mount_project else CONTAINER_CWD_SHARED
        cmd.extend(["--workdir", cwd])

        if extra_run_args is not None:
            cmd.extend(extra_run_args)

        with tempfile.TemporaryDirectory(prefix="tiz_oci_hooks_") as tmpdirname:
            if oci_hooks:
                hooks_dir = Path(tmpdirname)
                hook_script = hooks_dir / "hook.sh"
                hook_script.write_text(OCI_HOOK_SCRIPT_CONTENT, encoding="utf-8")
                hook_script.chmod(0o755)
                hook_spec = {
                    "version": "1.0.0",
                    "hook": {
                        "path": str(hook_script),
                        "args": [str(hook_script)],
                    },
                    "when": {"always": True},
                    "stages": ["createContainer"],
                }
                (hooks_dir / OCI_START_HOOK_NAME).write_text(
                    json.dumps(hook_spec, indent=2) + "\n",
                    encoding="utf-8",
                )
                cmd.extend(["--hooks-dir", str(hooks_dir)])

            cmd.append(image)

            worker_cmd = list(CONTAINER_CMD_PREFIX)
            inner_cmd = f"exec /usr/bin/env python3 {CONTAINER_MOUNT_WORKER}"
            if verbose > 0:
                inner_cmd += " -" + "v" * verbose
            inner_cmd += f" {CONTAINER_WORKER_SOCK_PATH}"
            worker_cmd.append(inner_cmd)
            cmd.extend(worker_cmd)

            run_result = self._run_subprocess(cmd)

        if run_result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container: {shlex.join(cmd)}: {run_result.stderr.strip()}"
            )
        self._container_name = container_name
        container_id_val = run_result.stdout.strip()
        self._container_id = container_id_val
        meta = ContainerMeta(
            container_id=container_id_val,
            engine=self._engine,
            network=network,
            container_name=container_name,
        )
        self._sandbox_dirs.add_container_meta(dataclasses.asdict(meta))

    def stop(self, timeout: int = 3) -> None:
        """Stop and remove a container."""
        if self._container_id is None:
            raise RuntimeError("No container has been started")
        logger.info("Stopping container '%s'", self._container_id)

        stop_cmd = [self._engine, "stop", "-t", str(timeout), self._container_id]
        stop_result = self._run_subprocess(stop_cmd)
        if stop_result.returncode != 0:
            stderr_lower = stop_result.stderr.lower()
            if not any(
                indicator in stderr_lower for indicator in _ALREADY_STOPPED_INDICATORS
            ):
                raise RuntimeError(
                    f"Failed to stop container: {stop_result.stderr.strip()}"
                )

        rm_cmd = [self._engine, "rm", self._container_id]
        rm_result = self._run_subprocess(rm_cmd)
        if (
            rm_result.returncode != 0
            and "no such container" not in rm_result.stderr.lower()
        ):
            raise RuntimeError(
                f"Failed to remove container: {rm_result.stderr.strip()}"
            )
        cid_to_remove = self._container_id
        self._container_id = None
        self._container_name = None
        self._sandbox_dirs.remove_container_meta(cid_to_remove)

    def is_running(self) -> bool:
        """Return ``True`` if the container is currently running."""
        if self._container_id is None:
            return False
        cmd = [
            self._engine,
            "inspect",
            "--format",
            "{{.State.Running}}",
            self._container_id,
        ]
        result = self._run_subprocess(cmd)
        if result.returncode != 0:
            return False
        return result.stdout.strip().lower() == "true"

    def exists(self) -> bool:
        """Return ``True`` if the container exists."""
        if self._container_id is None:
            return False
        cmd = [
            self._engine,
            "inspect",
            "--format",
            "{{.Id}}",
            self._container_id,
        ]
        result = self._run_subprocess(cmd)
        if result.returncode != 0:
            return False
        return result.stdout.strip().lower() == self._container_id.lower()

    def exec_in_container(
        self,
        cmd: list[str] | None = None,
        *,
        interactive: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.Popen:
        """Run a command or interactive shell inside the running container.

        Parameters
        ----------
        cmd:
            Command to execute inside the container.
            When ``None`` an interactive shell (``/bin/bash``) is launched.
        interactive:
            When ``True`` attach the terminal to the container process
            (passes ``-it`` to the container engine).

        Returns
        -------
        subprocess.CompletedProcess[str] | subprocess.Popen
            A ``CompletedProcess`` for non-interactive invocations,
            or a ``Popen`` instance for interactive sessions.
        """
        if self._container_id is None:
            raise RuntimeError("No container has been started")
        if not self.is_running():
            raise RuntimeError("Container is not running")

        exec_cmd = [self._engine, "exec"]
        if interactive:
            exec_cmd.extend(["-it"])
        exec_cmd.append(self._container_id)
        if cmd is None:
            exec_cmd.extend(["/bin/bash", "-l"])
        else:
            exec_cmd.extend(cmd)

        if interactive:
            return subprocess.Popen(exec_cmd)
        return self._run_subprocess(exec_cmd)

    def __enter__(self) -> SandboxContainer:
        """Enter the runtime context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the runtime context, stopping the container if it was started."""
        if self._container_id is not None:
            with contextlib.suppress(RuntimeError):
                self.stop()

    @property
    def shared_dir(self) -> Path | None:
        """Return the shared directory path."""
        if self._container_name is None:
            return None
        return self._sandbox_dirs.shared_container_dir(self._container_name)

    @property
    def worker_socket_path(self) -> Path | None:
        """Return the host path to the worker Unix socket."""
        own_shared_dir = self.shared_dir
        if own_shared_dir is None:
            return None
        return own_shared_dir / CONTAINER_WORKER_SOCK_BASENAME

    @property
    def container_name(self) -> str | None:
        """Return the container name."""
        return self._container_name

    @overload
    def get_container_logs(
        self, *, separate: Literal[True]
    ) -> tuple[str, str]: ...  # pragma: no cover

    @overload
    def get_container_logs(
        self, *, separate: Literal[False] = False
    ) -> str: ...  # pragma: no cover

    def get_container_logs(self, *, separate: bool = False) -> str | tuple[str, str]:
        """Return the container's logs.

        Parameters
        ----------
        separate:
            When ``True`` return ``(stdout, stderr)`` as a tuple.
            When ``False`` return stdout and stderr concatenated.

        Returns
        -------
        str | tuple[str, str]
            Combined logs, or a ``(stdout, stderr)`` tuple.
        """
        if self._container_id is None:
            raise RuntimeError("No container has been started")
        cmd = [self._engine, "logs", self._container_id]
        if separate:
            result = self._run_subprocess(cmd)
        else:
            result = subprocess.run(
                cmd,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            err_msg = (result.stderr or "").strip() or (result.stdout or "").strip()
            raise RuntimeError(f"Failed to get container logs: {err_msg}")
        if separate:
            return (result.stdout, result.stderr)
        return result.stdout

    # ------------------------------------------------------------------
    # engine info
    # ------------------------------------------------------------------

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def sandbox_name(self) -> str:
        return self._sandbox_dirs.sandbox_name

    @property
    def container_id(self) -> str | None:
        return self._container_id
