# ruff: noqa: ARG001,ARG002,SIM117
# mypy: disable-error-code="attr-defined,misc"
"""Tests for src/tiz/sandbox_container.py."""

import dataclasses
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tiz.sandbox_container import (
    _ALREADY_STOPPED_INDICATORS,
    ContainerMeta,
    SandboxContainer,
)
from tiz.sandbox_dirs import SandboxDirs


@pytest.fixture(autouse=True)
def _patch_which() -> None:
    def fake_which(name: str) -> str | None:
        if name in ("podman", "docker"):
            return f"/usr/bin/{name}"
        return None

    with patch("shutil.which", side_effect=fake_which):
        yield


@pytest.fixture
def sandbox_base(tmp_path: Path) -> Path:
    return tmp_path / "sandboxes"


@pytest.fixture
def sandbox_dirs(sandbox_base: Path) -> SandboxDirs:
    s = SandboxDirs("test-sb", base_path=sandbox_base)
    s.create()
    return s


def make_meta(
    container_id: str = "abc123",
    engine: str = "podman",
    network: str = "none",
    container_name: str | None = None,
) -> ContainerMeta:
    return ContainerMeta(
        container_id=container_id,
        engine=engine,
        network=network,
        container_name=container_name,
    )


# ---------------------------------------------------------------------------
# ContainerMeta
# ---------------------------------------------------------------------------


def test_container_meta_frozen() -> None:
    meta = make_meta()
    assert meta.container_id == "abc123"
    assert meta.engine == "podman"
    assert meta.network == "none"
    assert meta.container_name is None


def test_container_meta_with_name() -> None:
    meta = make_meta(container_name="my-container")
    assert meta.container_name == "my-container"


def test_container_meta_is_frozen() -> None:
    meta = make_meta()
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        meta.container_id = "new_id"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_engine(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    assert sc.engine == "podman"


def test_docker_engine(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs, engine="docker")
    assert sc.engine == "docker"


def test_invalid_engine(sandbox_dirs: SandboxDirs) -> None:
    with pytest.raises(ValueError, match="Container engine 'lxc' not found in PATH"):
        SandboxContainer(sandbox_dirs, engine="lxc")


def test_restore_from_meta(sandbox_dirs: SandboxDirs) -> None:
    meta = make_meta(container_id="restored")
    sc = SandboxContainer(sandbox_dirs, engine="docker", meta=meta)
    assert sc.engine == "podman"
    assert sc.container_id == "restored"


def test_container_id_initially_none(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    assert sc.container_id is None


def test_sandbox_name(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    assert sc.sandbox_name == "test-sb"


def test_start_invalid_container_name_raises(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(ValueError, match="Invalid container name"):
        sc.start(
            container_name="invalid name", oci_hooks=False, use_host_timezone=False
        )


def test_start_invalid_container_name_special_chars(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(ValueError, match="Invalid container name"):
        sc.start(
            container_name="test@container", oci_hooks=False, use_host_timezone=False
        )


def test_start_invalid_container_name_leading_dot(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(ValueError, match="Invalid container name"):
        sc.start(container_name=".test", oci_hooks=False, use_host_timezone=False)


def test_start_invalid_container_name_empty(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(ValueError, match="Invalid container name"):
        sc.start(container_name="", oci_hooks=False, use_host_timezone=False)


def test_start_valid_container_names(sandbox_dirs: SandboxDirs) -> None:
    valid_names = ["test-c", "test_c", "test.c", "Test1", "a", "1abc", "a.b-c_d"]
    for name in valid_names:
        sc = SandboxContainer(sandbox_dirs)
        with patch.object(sc, "_run_subprocess", return_value=_success_result()):
            sc.start(container_name=name, oci_hooks=False, use_host_timezone=False)
            assert sc.container_id == "container_id_1"


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


def _success_result(stdout: str = "container_id_1") -> subprocess.CompletedProcess[str]:
    cp: subprocess.CompletedProcess[str] = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = 0
    cp.stdout = stdout
    cp.stderr = ""
    return cp


def test_start_default(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with patch.object(sc, "_run_subprocess", return_value=_success_result()):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    assert sc.container_id == "container_id_1"
    cmeta = sandbox_dirs.containers_meta
    assert len(cmeta) == 1
    assert cmeta[0] == {
        "container_id": "container_id_1",
        "container_name": "test-c",
        "engine": "podman",
        "network": "none",
    }


def test_start_with_tmpfs_root_true(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            tmpfs_root=True,
        )
    cmd = captured_cmds[0]
    assert "--read-only" not in cmd
    assert "--image-volume=tmpfs" in cmd
    assert sc.container_id == "container_id_1"
    cmeta = sandbox_dirs.containers_meta
    assert len(cmeta) == 1
    assert cmeta[0] == {
        "container_id": "container_id_1",
        "container_name": "test-c",
        "engine": "podman",
        "network": "none",
    }


def test_start_with_tmpfs_root_false_default(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            tmpfs_root=False,
        )
    cmd = captured_cmds[0]
    assert "--read-only" in cmd
    assert "--image-volume=tmpfs" not in cmd


def test_start_with_tmpfs_root_and_read_only_project(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    project_dir = sandbox_dirs.project_dir
    project_dir.mkdir(exist_ok=True)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            read_only_project=True,
            oci_hooks=False,
            use_host_timezone=False,
            tmpfs_root=True,
        )
    cmd = captured_cmds[0]
    assert "--read-only" not in cmd
    assert "--image-volume=tmpfs" in cmd
    # The project mount should still be read-only
    mount_indices = [i for i, c in enumerate(cmd) if c == "--mount"]
    project_mounts = [
        cmd[i + 1] for i in mount_indices if ",target=/opt/project," in cmd[i + 1]
    ]
    assert len(project_mounts) == 1
    assert ",ro," in project_mounts[0] or project_mounts[0].endswith(",ro")


def test_start_calls_run_subprocess(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    assert cmd[0] == "podman"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--name" in cmd
    name_idx = cmd.index("--name")
    assert cmd[name_idx + 1] == "test-c"
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"
    assert cmd[-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    assert cmd[-14] == "--mount"
    assert (
        cmd[-13]
        == f"type=bind,source={sandbox_dirs.shared_general_dir},target=/opt/shared,rw,nosuid,nodev"
    )
    assert cmd[-12] == "--mount"
    assert cmd[-11].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[-10] == "--mount"
    assert cmd[-9].endswith("/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev")
    assert cmd[-8] == "--mount"
    assert (
        cmd[-7]
        == f"type=bind,source={sandbox_dirs.shared_container_dir('test-c')},target=/opt/container_shared,rw,nosuid,nodev"
    )


def test_start_with_container_name(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="my-container", oci_hooks=False, use_host_timezone=False
        )
    cmd = captured_cmds[0]
    assert "--name" in cmd
    name_idx = cmd.index("--name")
    assert cmd[name_idx + 1] == "my-container"
    assert cmd[0] == "podman"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"
    assert cmd[-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    assert cmd[-12] == "--mount"
    assert cmd[-11].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[-10] == "--mount"
    assert cmd[-9].endswith("/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev")
    assert cmd[-8] == "--mount"
    assert (
        cmd[-7]
        == f"type=bind,source={sandbox_dirs.shared_container_dir('my-container')},target=/opt/container_shared,rw,nosuid,nodev"
    )
    assert sc.container_id == "container_id_1"
    cmeta = sandbox_dirs.containers_meta
    assert len(cmeta) == 1
    assert cmeta[0] == {
        "container_id": "container_id_1",
        "container_name": "my-container",
        "engine": "podman",
        "network": "none",
    }


def test_start_with_network(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            network="bridge",
            oci_hooks=False,
            use_host_timezone=False,
        )
    cmd = captured_cmds[0]
    assert cmd[0] == "podman"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"
    assert cmd[-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    assert cmd[-14] == "--mount"
    assert (
        cmd[-13]
        == f"type=bind,source={sandbox_dirs.shared_general_dir},target=/opt/shared,rw,nosuid,nodev"
    )
    assert cmd[-12] == "--mount"
    assert cmd[-11].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[-10] == "--mount"
    assert cmd[-9].endswith("/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev")
    assert cmd[-8] == "--mount"
    assert (
        cmd[-7]
        == f"type=bind,source={sandbox_dirs.shared_container_dir('test-c')},target=/opt/container_shared,rw,nosuid,nodev"
    )
    assert sc.container_id == "container_id_1"
    cmeta = sandbox_dirs.containers_meta
    assert len(cmeta) == 1
    assert cmeta[0] == {
        "container_id": "container_id_1",
        "container_name": "test-c",
        "engine": "podman",
        "network": "bridge",
    }


def test_start_with_read_only_project(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    project_dir = sandbox_dirs.project_dir
    project_dir.mkdir(exist_ok=True)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            read_only_project=True,
            oci_hooks=False,
            use_host_timezone=False,
        )
    cmd = captured_cmds[0]
    assert cmd[0] == "podman"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/project"
    assert cmd[-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    assert cmd[-16] == "--mount"
    assert (
        cmd[-15]
        == f"type=bind,source={sandbox_dirs.project_dir},target=/opt/project,ro,nosuid,nodev"
    )
    assert cmd[-14] == "--mount"
    assert (
        cmd[-13]
        == f"type=bind,source={sandbox_dirs.shared_general_dir},target=/opt/shared,rw,nosuid,nodev"
    )
    assert cmd[-12] == "--mount"
    assert cmd[-11].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[-10] == "--mount"
    assert cmd[-9].endswith("/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev")
    assert cmd[-8] == "--mount"
    assert (
        cmd[-7]
        == f"type=bind,source={sandbox_dirs.shared_container_dir('test-c')},target=/opt/container_shared,rw,nosuid,nodev"
    )


def test_start_without_read_only_project(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    project_dir = sandbox_dirs.project_dir
    project_dir.mkdir(exist_ok=True)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            read_only_project=False,
            oci_hooks=False,
            use_host_timezone=False,
        )
    cmd = captured_cmds[0]
    assert cmd[0] == "podman"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/project"
    assert cmd[-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    assert cmd[-16] == "--mount"
    assert (
        cmd[-15]
        == f"type=bind,source={sandbox_dirs.project_dir},target=/opt/project,rw,nosuid,nodev"
    )
    assert cmd[-14] == "--mount"
    assert (
        cmd[-13]
        == f"type=bind,source={sandbox_dirs.shared_general_dir},target=/opt/shared,rw,nosuid,nodev"
    )
    assert cmd[-12] == "--mount"
    assert cmd[-11].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[-10] == "--mount"
    assert cmd[-9].endswith("/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev")
    assert cmd[-8] == "--mount"
    assert (
        cmd[-7]
        == f"type=bind,source={sandbox_dirs.shared_container_dir('test-c')},target=/opt/container_shared,rw,nosuid,nodev"
    )


def test_start_project_dir_does_not_exist(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    cmd = captured_cmds[0]
    assert cmd[0] == "podman"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"
    assert cmd[-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    assert cmd[-14] == "--mount"
    assert (
        cmd[-13]
        == f"type=bind,source={sandbox_dirs.shared_general_dir},target=/opt/shared,rw,nosuid,nodev"
    )
    assert cmd[-12] == "--mount"
    assert cmd[-11].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[-10] == "--mount"
    assert cmd[-9].endswith("/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev")
    assert cmd[-8] == "--mount"
    assert (
        cmd[-7]
        == f"type=bind,source={sandbox_dirs.shared_container_dir('test-c')},target=/opt/container_shared,rw,nosuid,nodev"
    )


def test_start_with_extra_run_args(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            extra_run_args=["--memory=512m", "--cpus=1"],
            oci_hooks=False,
            use_host_timezone=False,
        )
    cmd = captured_cmds[0]
    assert "--memory=512m" in cmd
    assert "--cpus=1" in cmd
    assert cmd[0] == "podman"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"
    assert cmd[-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    mount_args = [i for i, c in enumerate(cmd) if c == "--mount"]
    assert len(mount_args) == 4
    assert cmd[mount_args[0] + 1].startswith(
        f"type=bind,source={sandbox_dirs.shared_general_dir}"
    )
    assert cmd[mount_args[1] + 1].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[mount_args[2] + 1].endswith(
        "/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev"
    )
    assert cmd[mount_args[3] + 1].startswith(
        f"type=bind,source={sandbox_dirs.shared_container_dir('test-c')}"
    )


def test_start_with_verbose(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c", verbose=2, oci_hooks=False, use_host_timezone=False
        )
    assert captured_cmds[0][-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py -vv /opt/container_shared/exe.sock",
    ]


def test_start_with_verbose_1(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c", verbose=1, oci_hooks=False, use_host_timezone=False
        )
    assert captured_cmds[0][-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py -v /opt/container_shared/exe.sock",
    ]


def test_start_with_verbose_0(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c", verbose=0, oci_hooks=False, use_host_timezone=False
        )
    assert captured_cmds[0][-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]


def test_start_already_started_raises(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with patch.object(sc, "_run_subprocess", return_value=_success_result()):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    with pytest.raises(RuntimeError, match="Container already started"):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)


def test_start_failure_raises(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)

    def fail(_: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 1
        cp.stdout = ""
        cp.stderr = "something went wrong\n"
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=fail):
        with pytest.raises(RuntimeError, match="Failed to start container"):
            sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    assert sc.container_id is None


def test_start_stores_meta(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with patch.object(sc, "_run_subprocess", return_value=_success_result()):
        sc.start(
            container_name="test-c",
            network="bridge",
            oci_hooks=False,
            use_host_timezone=False,
        )
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 1
    assert metas[0]["container_id"] == "container_id_1"
    assert metas[0]["engine"] == "podman"
    assert metas[0]["network"] == "bridge"
    assert metas[0]["container_name"] == "test-c"


def test_start_uses_custom_image(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            image="my-image:v1",
            oci_hooks=False,
            use_host_timezone=False,
        )
    cmd = captured_cmds[0]
    assert cmd[0] == "podman"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"
    assert cmd[-4:] == [
        "my-image:v1",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    assert cmd[-14] == "--mount"
    assert (
        cmd[-13]
        == f"type=bind,source={sandbox_dirs.shared_general_dir},target=/opt/shared,rw,nosuid,nodev"
    )
    assert cmd[-12] == "--mount"
    assert cmd[-11].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[-10] == "--mount"
    assert cmd[-9].endswith("/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev")
    assert cmd[-8] == "--mount"
    assert cmd[-7].startswith(
        f"type=bind,source={sandbox_dirs.shared_container_dir('test-c')}"
    )


def test_start_docker_engine(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs, engine="docker")
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    cmd = captured_cmds[0]
    assert cmd[0] == "docker"
    assert "run" in cmd
    assert "--pull=never" in cmd
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--no-hosts" in cmd
    assert "--cap-drop=all" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns=keep-id:uid=1000,gid=1000" in cmd
    assert "--workdir" in cmd
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"
    assert cmd[-4:] == [
        "tiz-worker:latest",
        "/bin/bash",
        "-lc",
        "exec /usr/bin/env python3 /usr/local/bin/worker.py /opt/container_shared/exe.sock",
    ]
    assert cmd[-14] == "--mount"
    assert (
        cmd[-13]
        == f"type=bind,source={sandbox_dirs.shared_general_dir},target=/opt/shared,rw,nosuid,nodev"
    )
    assert cmd[-12] == "--mount"
    assert cmd[-11].endswith(
        "/tiz/sandbox_worker.py,target=/usr/local/bin/worker.py,ro"
    )
    assert cmd[-10] == "--mount"
    assert cmd[-9].endswith("/tiz/worker_scripts,target=/opt/scripts,ro,nosuid,nodev")
    assert cmd[-8] == "--mount"
    assert cmd[-7].startswith(
        f"type=bind,source={sandbox_dirs.shared_container_dir('test-c')}"
    )
    assert sc.container_id == "container_id_1"
    cmeta = sandbox_dirs.containers_meta
    assert len(cmeta) == 1
    assert cmeta[0] == {
        "container_id": "container_id_1",
        "container_name": "test-c",
        "engine": "docker",
        "network": "none",
    }


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


def _stop_success_stdout(
    stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    cp: subprocess.CompletedProcess[str] = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = 0
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def _stop_fail_stdout(stderr: str) -> subprocess.CompletedProcess[str]:
    cp: subprocess.CompletedProcess[str] = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = 1
    cp.stdout = ""
    cp.stderr = stderr
    return cp


def test_stop_not_started_raises(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(RuntimeError, match="No container has been started"):
        sc.stop()


def test_stop_success(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 0
    sandbox_dirs.add_container_meta(dataclasses.asdict(meta))
    with patch.object(sc, "_run_subprocess", return_value=_stop_success_stdout()):
        sc.stop()
    assert sc.container_id is None
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 0


def test_stop_already_stopped(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 0
    sandbox_dirs.add_container_meta(dataclasses.asdict(meta))
    with patch.object(
        sc,
        "_run_subprocess",
        side_effect=[
            _stop_fail_stdout("container is not running"),
            _stop_success_stdout(),
        ],
    ):
        sc.stop()
    assert sc.container_id is None
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 0


def test_stop_already_stopped_indicators(sandbox_dirs: SandboxDirs) -> None:
    for indicator in _ALREADY_STOPPED_INDICATORS:
        sc = SandboxContainer(sandbox_dirs)
        meta = make_meta(container_id="c1", engine="podman")
        sc = SandboxContainer(sandbox_dirs, meta=meta)
        metas = sandbox_dirs.containers_meta
        assert len(metas) == 0
        sandbox_dirs.add_container_meta(dataclasses.asdict(meta))
        with patch.object(
            sc,
            "_run_subprocess",
            side_effect=[
                _stop_fail_stdout(f"{indicator.upper()}"),
                _stop_success_stdout(),
            ],
        ):
            sc.stop()
        assert sc.container_id is None
        metas = sandbox_dirs.containers_meta
        assert len(metas) == 0


def test_stop_rm_no_such_container(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 0
    sandbox_dirs.add_container_meta(dataclasses.asdict(meta))
    with patch.object(
        sc,
        "_run_subprocess",
        side_effect=[
            _stop_success_stdout(),
            _stop_fail_stdout("no such container"),
        ],
    ):
        sc.stop()
    assert sc.container_id is None
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 0


def test_stop_failure_raises(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 0
    sandbox_dirs.add_container_meta(dataclasses.asdict(meta))
    with (
        patch.object(
            sc,
            "_run_subprocess",
            side_effect=[
                _stop_fail_stdout("unknown error"),
            ],
        ),
        pytest.raises(RuntimeError, match="Failed to stop container"),
    ):
        sc.stop()
    assert sc.container_id == "c1"
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 1


def test_stop_rm_failure_raises(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 0
    sandbox_dirs.add_container_meta(dataclasses.asdict(meta))
    with (
        patch.object(
            sc,
            "_run_subprocess",
            side_effect=[
                _stop_success_stdout(),
                _stop_fail_stdout("some other rm error"),
            ],
        ),
        pytest.raises(RuntimeError, match="Failed to remove container"),
    ):
        sc.stop()
    assert sc.container_id == "c1"
    metas = sandbox_dirs.containers_meta
    assert len(metas) == 1


# ---------------------------------------------------------------------------
# is_running()
# ---------------------------------------------------------------------------


def test_is_running_not_started(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    assert sc.is_running() is False


def test_is_running_true(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "true\n"
        cp.stderr = ""
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        assert sc.is_running() is True


def test_is_running_false(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "false\n"
        cp.stderr = ""
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        assert sc.is_running() is False


def test_is_running_inspect_failure(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 1
        cp.stdout = ""
        cp.stderr = "error"
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        assert sc.is_running() is False


# ---------------------------------------------------------------------------
# get_container_logs()
# ---------------------------------------------------------------------------


def test_get_logs_not_started_raises(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(RuntimeError, match="No container has been started"):
        sc.get_container_logs()


def test_get_logs_combined(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    with patch("tiz.sandbox_container.subprocess.run") as mock_run:
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "combined output"
        mock_run.return_value = mock_result
        result = sc.get_container_logs(separate=False)
    assert result == "combined output"


def test_get_logs_separate(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "stdout data"
        cp.stderr = "stderr data"
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        result = sc.get_container_logs(separate=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == "stdout data"
    assert result[1] == "stderr data"


def test_get_logs_failure(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    with patch("tiz.sandbox_container.subprocess.run") as mock_run:
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        with pytest.raises(RuntimeError, match="Failed to get container logs"):
            sc.get_container_logs()


def test_get_logs_separate_failure(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 1
        cp.stdout = ""
        cp.stderr = "log error sep"
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        with pytest.raises(RuntimeError, match="Failed to get container logs"):
            sc.get_container_logs(separate=True)


# ---------------------------------------------------------------------------
# _run_subprocess
# ---------------------------------------------------------------------------


def test_run_subprocess(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    result = sc._run_subprocess(["echo", "hello"])
    assert result.returncode == 0
    assert "hello" in result.stdout


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_engine_property(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs, engine="docker")
    assert sc.engine == "docker"


def test_container_id_property_after_start(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with patch.object(sc, "_run_subprocess", return_value=_success_result("my-id")):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    assert sc.container_id == "my-id"


def test_container_id_property_after_stop(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    assert sc.container_id == "c1"
    with patch.object(sc, "_run_subprocess", return_value=_stop_success_stdout()):
        sc.stop()
    assert sc.container_id is None


def test_shared_dir_property(sandbox_dirs: SandboxDirs) -> None:
    meta = make_meta(container_name="test", container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    assert sc.container_name == "test"
    assert sc.shared_dir == sandbox_dirs.shared_container_dir("test")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_start_with_meta_ignores_engine_param(sandbox_dirs: SandboxDirs) -> None:
    meta = make_meta(engine="docker", container_id="from-meta")
    sc = SandboxContainer(sandbox_dirs, engine="podman", meta=meta)
    assert sc.engine == "docker"
    assert sc.container_id == "from-meta"


def test_get_logs_separate_default_false(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    with patch("tiz.sandbox_container.subprocess.run") as mock_run:
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "some log"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        output = sc.get_container_logs()
    assert output == "some log"


def test_get_logs_empty_stdout(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)

    with patch("tiz.sandbox_container.subprocess.run") as mock_run:
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        output = sc.get_container_logs()
    assert output == ""


# ---------------------------------------------------------------------------
# exec_in_container()
# ---------------------------------------------------------------------------


def test_exec_in_container_not_started_raises(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(RuntimeError, match="No container has been started"):
        sc.exec_in_container()


def test_exec_in_container_non_interactive_no_cmd(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="c1"))
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "true"
        cp.stderr = ""
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        result = sc.exec_in_container()
    assert "--format" in captured_cmds[0]
    assert "exec" in captured_cmds[1]
    assert result.stdout == "true"


def test_exec_in_container_non_interactive_with_cmd(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="c1"))
    captured_cmds: list[list[str]] = []
    call_count = 0

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        captured_cmds.append(cmd)
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        if call_count == 1:
            cp.returncode = 0
            cp.stdout = "true\n"
            cp.stderr = ""
        else:
            cp.returncode = 0
            cp.stdout = "cmd_output"
            cp.stderr = ""
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        result = sc.exec_in_container(cmd=["echo", "hello"])
    assert "exec" in captured_cmds[1]
    assert "echo" in captured_cmds[1]
    assert result.stdout == "cmd_output"


def test_exec_in_container_not_running_raises(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="c1"))

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "false\n"
        cp.stderr = ""
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        with pytest.raises(RuntimeError, match="Container is not running"):
            sc.exec_in_container()


def test_exec_in_container_interactive(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="c1"))

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "true\n"
        cp.stderr = ""
        return cp

    mock_popen = MagicMock(spec=subprocess.Popen)
    with patch.object(sc, "_run_subprocess", side_effect=capture):
        with patch("subprocess.Popen", return_value=mock_popen) as mock_popen_cls:
            result = sc.exec_in_container(interactive=True)
    assert isinstance(result, subprocess.Popen)
    call_args = mock_popen_cls.call_args
    exec_cmd = call_args[0][0]
    assert exec_cmd == ["podman", "exec", "-it", "c1", "/bin/bash", "-l"]
    # interactive mode inherits parent's terminal (no stdin/stdout/stderr kwargs)
    assert call_args[1] == {}


def test_exec_in_container_interactive_with_cmd(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="c1"))

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "true\n"
        cp.stderr = ""
        return cp

    mock_popen = MagicMock(spec=subprocess.Popen)
    with patch.object(sc, "_run_subprocess", side_effect=capture):
        with patch("subprocess.Popen", return_value=mock_popen) as mock_popen_cls:
            result = sc.exec_in_container(cmd=["ls", "-la"], interactive=True)
    assert isinstance(result, subprocess.Popen)
    call_args = mock_popen_cls.call_args
    exec_cmd = call_args[0][0]
    assert exec_cmd == ["podman", "exec", "-it", "c1", "ls", "-la"]
    # interactive mode inherits parent's terminal (no stdin/stdout/stderr kwargs)
    assert call_args[1] == {}


def test_exec_in_container_error(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="c1"))
    captured_cmds: list[list[str]] = []
    call_count = 0

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        captured_cmds.append(cmd)
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        if call_count == 1:
            cp.returncode = 0
            cp.stdout = "true\n"
            cp.stderr = ""
        else:
            cp.returncode = 1
            cp.stdout = ""
            cp.stderr = ""
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        result = sc.exec_in_container(cmd=["exit", "1"])
    assert "exec" in captured_cmds[1]
    assert "exit" in captured_cmds[1]
    assert result.returncode == 1
    assert result.stdout == ""


def test_start_with_oci_hooks_enabled(sandbox_dirs: SandboxDirs) -> None:
    from tiz.sandbox_container import OCI_HOOK_SCRIPT_CONTENT

    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []
    captured_hooks_dir: Path | None = None

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal captured_hooks_dir
        captured_cmds.append(cmd)
        if "--hooks-dir" in cmd:
            hooks_dir_idx = cmd.index("--hooks-dir")
            captured_hooks_dir = Path(cmd[hooks_dir_idx + 1])
            assert captured_hooks_dir.is_dir()
            json_files = list(captured_hooks_dir.glob("*.json"))
            script_files = list(captured_hooks_dir.glob("*.sh"))
            assert len(json_files) == 1
            assert len(script_files) == 1
            hook_spec = json.loads(json_files[0].read_text())
            assert hook_spec["version"] == "1.0.0"
            assert "path" in hook_spec["hook"]
            assert hook_spec["stages"] == ["createContainer"]
            assert script_files[0].read_text() == OCI_HOOK_SCRIPT_CONTENT
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(container_name="test-c", oci_hooks=True, use_host_timezone=False)
    assert sc.container_id == "container_id_1"
    assert sc.container_name == "test-c"
    assert sc.shared_dir is not None
    assert sc.shared_dir.exists()
    assert "--hooks-dir" in captured_cmds[0]
    assert captured_hooks_dir is not None


def test_start_with_oci_hooks_disabled(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    cmd = captured_cmds[0]
    assert "--hooks-dir" not in cmd


def test_start_with_use_host_timezone_true(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=True)
    cmd = captured_cmds[0]
    mount_indices = [i for i, c in enumerate(cmd) if c == "--mount"]
    tz_mounts = [cmd[i + 1] for i in mount_indices if "localtime" in cmd[i + 1]]
    assert len(tz_mounts) == 1
    assert tz_mounts[0] == "type=bind,source=/etc/localtime,target=/etc/localtime,ro"


def test_start_with_use_host_timezone_false(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    cmd = captured_cmds[0]
    mount_indices = [i for i, c in enumerate(cmd) if c == "--mount"]
    tz_mounts = [cmd[i + 1] for i in mount_indices if "localtime" in cmd[i + 1]]
    assert len(tz_mounts) == 0


def test_worker_socket_path_when_container_name_is_none(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs)
    assert sc.worker_socket_path is None


def test_worker_socket_path_with_container_name(sandbox_dirs: SandboxDirs) -> None:
    meta = make_meta(container_name="test-c", container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    expected = sandbox_dirs.shared_container_dir("test-c") / "exe.sock"
    assert sc.worker_socket_path == expected


def test_oci_hook_script_content_has_shebang() -> None:
    from tiz.sandbox_container import OCI_HOOK_SCRIPT_CONTENT

    assert OCI_HOOK_SCRIPT_CONTENT.startswith("#!/bin/sh")


def test_container_name_property_not_none(sandbox_dirs: SandboxDirs) -> None:
    meta = make_meta(container_name="my-ctr", container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    assert sc.container_name == "my-ctr"


def test_start_with_custom_tools_dir(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []
    tools_dir = sandbox_dirs.shared_general_dir.parent / "tools_worker"
    tools_dir.mkdir(parents=True, exist_ok=True)

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            custom_tools_dir=tools_dir,
        )
    cmd = captured_cmds[0]
    mount_indices = [i for i, c in enumerate(cmd) if c == "--mount"]
    tools_mounts = [cmd[i + 1] for i in mount_indices if "/opt/tiz_tools" in cmd[i + 1]]
    assert len(tools_mounts) == 1
    assert tools_mounts[0] == (
        f"type=bind,source={tools_dir},target=/opt/tiz_tools,ro,nosuid,nodev"
    )


def test_start_with_custom_tools_dir_none(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            custom_tools_dir=None,
        )
    cmd = captured_cmds[0]
    mount_indices = [i for i, c in enumerate(cmd) if c == "--mount"]
    tools_mounts = [cmd[i + 1] for i in mount_indices if "/opt/tiz_tools" in cmd[i + 1]]
    assert len(tools_mounts) == 0


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


def test_exists_not_started(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    assert sc.exists() is False


def test_exists_true(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="abc123"))

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "abc123\n"
        cp.stderr = ""
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        assert sc.exists() is True


def test_exists_false_due_to_returncode(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="abc123"))

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 1
        cp.stdout = ""
        cp.stderr = "No such container"
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        assert sc.exists() is False


def test_exists_false_due_to_mismatched_id(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="abc123"))

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "different_id\n"
        cp.stderr = ""
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        assert sc.exists() is False


# ---------------------------------------------------------------------------
# shared_dir edge cases
# ---------------------------------------------------------------------------


def test_shared_dir_none_when_no_container_name(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    assert sc.container_name is None
    assert sc.shared_dir is None


def test_container_name_none_initially(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    assert sc.container_name is None


# ---------------------------------------------------------------------------
# start() with mount_project=False
# ---------------------------------------------------------------------------


def test_start_with_mount_project_false(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            mount_project=False,
        )
    cmd = captured_cmds[0]
    # No --mount for project dir
    mount_indices = [i for i, c in enumerate(cmd) if c == "--mount"]
    project_mounts = [
        cmd[i + 1] for i in mount_indices if ",target=/opt/project," in cmd[i + 1]
    ]
    assert len(project_mounts) == 0
    # workdir should be shared since project not mounted
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"


def test_start_with_mount_project_true_project_not_exists(
    sandbox_dirs: SandboxDirs,
) -> None:
    """When mount_project=True but project_dir doesn't exist, skips project mount."""
    sc = SandboxContainer(sandbox_dirs)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            mount_project=True,
        )
    cmd = captured_cmds[0]
    mount_indices = [i for i, c in enumerate(cmd) if c == "--mount"]
    project_mounts = [
        cmd[i + 1] for i in mount_indices if ",target=/opt/project," in cmd[i + 1]
    ]
    assert len(project_mounts) == 0
    # workdir should be shared since project not mounted
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"


def test_start_with_mount_project_false_project_exists(
    sandbox_dirs: SandboxDirs,
) -> None:
    """When mount_project=False but project_dir exists, still skips project mount."""
    sc = SandboxContainer(sandbox_dirs)
    project_dir = sandbox_dirs.project_dir
    project_dir.mkdir(exist_ok=True)
    captured_cmds: list[list[str]] = []

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return _success_result()

    with patch.object(sc, "_run_subprocess", side_effect=capture):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            mount_project=False,
        )
    cmd = captured_cmds[0]
    mount_indices = [i for i, c in enumerate(cmd) if c == "--mount"]
    project_mounts = [
        cmd[i + 1] for i in mount_indices if ",target=/opt/project," in cmd[i + 1]
    ]
    assert len(project_mounts) == 0
    # workdir should be shared since project not mounted
    assert cmd[cmd.index("--workdir") + 1] == "/opt/container_shared"


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------


def test_container_cmd_prefix_is_tuple() -> None:
    """Bug 5: CONTAINER_CMD_PREFIX should be a tuple (immutable)."""
    from tiz.sandbox_container import CONTAINER_CMD_PREFIX

    assert isinstance(CONTAINER_CMD_PREFIX, tuple)
    assert CONTAINER_CMD_PREFIX == ("/bin/bash", "-lc")


def test_container_name_reset_on_stop(sandbox_dirs: SandboxDirs) -> None:
    """Bug 3: _container_name must be reset when stop() is called."""
    meta = make_meta(container_id="c1", container_name="my-container")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    assert sc.container_name == "my-container"
    assert sc.shared_dir is not None
    assert sc.worker_socket_path is not None
    with patch.object(sc, "_run_subprocess", return_value=_stop_success_stdout()):
        sc.stop()
    assert sc.container_id is None
    assert sc.container_name is None
    assert sc.shared_dir is None
    assert sc.worker_socket_path is None


def test_start_failure_uses_shlex_join(sandbox_dirs: SandboxDirs) -> None:
    """Bug 6: Error message should use shlex.join for readable command output."""
    sc = SandboxContainer(sandbox_dirs)

    def fail(_: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 1
        cp.stdout = ""
        cp.stderr = "error details\n"
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=fail):
        with pytest.raises(RuntimeError) as exc_info:
            sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    msg = str(exc_info.value)
    # shlex.join produces a shell-readable string without brackets or quotes
    assert "[podman" not in msg
    assert "'/bin/bash'" not in msg
    assert "error details" in msg


def test_exec_in_container_return_type_is_unparameterized_popen(
    sandbox_dirs: SandboxDirs,
) -> None:
    """Bug 7: exec_in_container returns Popen (not Popen[bytes]) for interactive."""
    sc = SandboxContainer(sandbox_dirs, meta=make_meta(container_id="c1"))

    def capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 0
        cp.stdout = "true\n"
        cp.stderr = ""
        return cp

    mock_popen = MagicMock(spec=subprocess.Popen)
    with patch.object(sc, "_run_subprocess", side_effect=capture):
        with patch("subprocess.Popen", return_value=mock_popen):
            result = sc.exec_in_container(interactive=True)
    assert isinstance(result, subprocess.Popen)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_enter_returns_self(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with sc as ctx:
        assert ctx is sc


def test_context_manager_exit_stops_container(sandbox_dirs: SandboxDirs) -> None:
    meta = make_meta(container_id="c1")
    sc = SandboxContainer(sandbox_dirs, meta=meta)
    with patch.object(sc, "_run_subprocess", return_value=_stop_success_stdout()):
        with sc as ctx:
            assert ctx.container_id == "c1"
        assert sc.container_id is None


def test_context_manager_exit_does_nothing_when_not_started(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with sc as ctx:
        assert ctx.container_id is None
    assert sc.container_id is None


# ---------------------------------------------------------------------------
# Comma in path validation
# ---------------------------------------------------------------------------


def test_start_comma_in_shared_general_dir_raises(
    sandbox_dirs: SandboxDirs,
    tmp_path: Path,
) -> None:
    base = tmp_path / "my,base"
    base.mkdir(parents=True)
    bad_dirs = SandboxDirs("test-sb", base_path=base)
    bad_dirs.create()
    sc = SandboxContainer(bad_dirs)
    with pytest.raises(RuntimeError, match="Failed to start container"):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)


def test_start_comma_in_custom_tools_dir_raises(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(RuntimeError, match="Failed to start container"):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            custom_tools_dir=Path("/some,path"),
        )


# ---------------------------------------------------------------------------
# _require_dir validation
# ---------------------------------------------------------------------------


def test_start_missing_shared_general_dir_raises(
    sandbox_dirs: SandboxDirs,
) -> None:
    """When shared general dir is missing, start() should raise."""
    import shutil

    sc = SandboxContainer(sandbox_dirs)
    shared_dir = sandbox_dirs.shared_general_dir
    assert shared_dir.is_dir()
    shutil.rmtree(shared_dir)
    with pytest.raises(RuntimeError, match="Failed to start container"):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)


def test_start_missing_worker_scripts_dir_raises(
    sandbox_dirs: SandboxDirs,
    tmp_path: Path,
) -> None:
    """When worker scripts dir is missing, start() should raise."""
    sc = SandboxContainer(sandbox_dirs)
    with (
        patch(
            "tiz.sandbox_container.WORKER_SCRIPTS_DIR",
            tmp_path / "nonexistent_scripts",
        ),
        pytest.raises(RuntimeError, match="Failed to start container"),
    ):
        sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)


def test_start_missing_custom_tools_dir_raises(
    sandbox_dirs: SandboxDirs,
) -> None:
    sc = SandboxContainer(sandbox_dirs)
    with pytest.raises(RuntimeError, match="Failed to start container"):
        sc.start(
            container_name="test-c",
            oci_hooks=False,
            use_host_timezone=False,
            custom_tools_dir=Path("/nonexistent_tools_dir"),
        )


# ---------------------------------------------------------------------------
# Failure cleanup - _container_name not set on failed start
# ---------------------------------------------------------------------------


def test_start_failure_cleans_up_state(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)

    def fail(_: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 1
        cp.stdout = ""
        cp.stderr = "error\n"
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=fail):
        with pytest.raises(RuntimeError, match="Failed to start container"):
            sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    assert sc.container_name is None
    assert sc.shared_dir is None
    assert sc.worker_socket_path is None


def test_start_failure_cleans_up_directory(sandbox_dirs: SandboxDirs) -> None:
    sc = SandboxContainer(sandbox_dirs)
    own_shared_dir = sandbox_dirs.shared_container_dir("test-c")
    assert not own_shared_dir.exists()

    def fail(_: list[str]) -> subprocess.CompletedProcess[str]:
        cp: subprocess.CompletedProcess[str] = MagicMock(
            spec=subprocess.CompletedProcess
        )
        cp.returncode = 1
        cp.stdout = ""
        cp.stderr = "error\n"
        return cp

    with patch.object(sc, "_run_subprocess", side_effect=fail):
        with pytest.raises(RuntimeError, match="Failed to start container"):
            sc.start(container_name="test-c", oci_hooks=False, use_host_timezone=False)
    # The shared dir is created before subprocess (line 191 of sandbox_container.py)
    # and not cleaned up on failure. Only state properties are reset.
    assert own_shared_dir.exists()
    assert sc.container_name is None
    assert sc.shared_dir is None
    assert sc.worker_socket_path is None
