# ruff: noqa: ARG001,ARG002,SIM117
# mypy: disable-error-code="attr-defined,misc"
"""Tests for src/tiz/sandbox_manager.py."""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tiz.sandbox_container import SandboxContainer
from tiz.sandbox_dirs import SandboxDirs
from tiz.sandbox_manager import TIZ_TOOLS_WORKER_DIR, TIZ_WORKER_PREFIX, SandboxManager


@pytest.fixture
def sandbox_base(tmp_path: Path) -> Path:
    return tmp_path


def _auto_detect(sandbox_base: Path) -> SandboxManager:
    """Helper for tests that don't use the manager fixture."""
    patches = [
        patch.object(SandboxManager, "available_engine", return_value="podman"),
        patch("shutil.which", return_value="/usr/bin/podman"),
    ]
    for p in patches:
        p.start()
    try:
        return SandboxManager(base_path=sandbox_base)
    finally:
        for p in patches:
            with contextlib.suppress(ValueError):
                p.stop()


def make_meta_dict(
    container_id: str,
    engine: str = "podman",
    network: str = "none",
    container_name: str | None = None,
) -> dict:
    return {
        "container_id": container_id,
        "engine": engine,
        "network": network,
        "container_name": container_name,
    }


# ---------------------------------------------------------------------------
# Static methods: available_engine / available_engines
# ---------------------------------------------------------------------------


def test_available_engine_returns_one() -> None:
    def fake_which(cmd: str) -> str | None:
        if cmd == "docker":
            return None
        if cmd == "podman":
            return "/usr/bin/podman"
        return None

    with patch.object(shutil, "which", side_effect=fake_which):
        result = SandboxManager.available_engine()
        assert result == "podman"


def test_available_engine_returns_one_2() -> None:
    def fake_which(cmd: str) -> str | None:
        if cmd == "docker":
            return "/usr/bin/docker"
        if cmd == "podman":
            return None
        return None

    with patch.object(shutil, "which", side_effect=fake_which):
        result = SandboxManager.available_engine()
        assert result == "docker"


def test_available_engine_podman_preferred() -> None:
    def fake_which(cmd: str) -> str | None:
        if cmd == "docker":
            return "/usr/bin/docker"
        if cmd == "podman":
            return "/usr/bin/podman"
        return None

    with patch.object(shutil, "which", side_effect=fake_which):
        result = SandboxManager.available_engine()
        assert result == "podman"


def test_available_engine_returns_none() -> None:
    with patch.object(shutil, "which", return_value=None):
        result = SandboxManager.available_engine()
        assert result is None


def test_available_engines_both() -> None:
    with patch.object(shutil, "which", return_value="/usr/bin/exe"):
        result = SandboxManager.available_engines()
        assert len(result) == 2
        assert "docker" in result
        assert "podman" in result


def test_available_engines_only_docker() -> None:
    def fake_which(cmd: str) -> str | None:
        if cmd == "docker":
            return "/usr/bin/docker"
        return None

    with patch.object(shutil, "which", side_effect=fake_which):
        result = SandboxManager.available_engines()
        assert result == ["docker"]


def test_available_engines_only_podman() -> None:
    def fake_which(cmd: str) -> str | None:
        if cmd == "podman":
            return "/usr/bin/podman"
        return None

    with patch.object(shutil, "which", side_effect=fake_which):
        result = SandboxManager.available_engines()
        assert result == ["podman"]


def test_available_engines_none() -> None:
    with patch.object(shutil, "which", return_value=None):
        result = SandboxManager.available_engines()
        assert result == []


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_auto_detect_engine(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    assert mgr._engine == "podman"


def test_init_auto_detect_docker(sandbox_base: Path) -> None:
    with (
        patch.object(SandboxManager, "available_engine", return_value="docker"),
        patch("shutil.which", return_value="/usr/bin/docker"),
    ):
        mgr = SandboxManager(base_path=sandbox_base)
    assert mgr._engine == "docker"


def test_init_no_engine_available(sandbox_base: Path) -> None:
    with patch.object(SandboxManager, "available_engine", return_value=None):
        with pytest.raises(RuntimeError, match="No container engine"):
            SandboxManager(base_path=sandbox_base)


def test_init_explicit_engine(sandbox_base: Path) -> None:
    with (
        patch.object(SandboxManager, "available_engine", return_value="podman"),
        patch("shutil.which", return_value="/usr/bin/podman"),
    ):
        mgr = SandboxManager(base_path=sandbox_base, engine="podman")
    assert mgr._engine == "podman"


def test_init_engine_not_in_path(sandbox_base: Path) -> None:
    with patch("shutil.which", return_value=None):
        with pytest.raises(ValueError, match="Engine 'podman' not found"):
            SandboxManager(base_path=sandbox_base, engine="podman")


def test_init_explicit_engine_in_path(sandbox_base: Path) -> None:
    def fake_which(name: str) -> str | None:
        if name == "podman":
            return "/usr/bin/podman"
        return None

    with (
        patch.object(SandboxManager, "available_engine", return_value=None),
        patch("shutil.which", side_effect=fake_which),
    ):
        mgr = SandboxManager(base_path=sandbox_base, engine="podman")
    assert mgr._engine == "podman"


# ---------------------------------------------------------------------------
# _iter_sandboxes
# ---------------------------------------------------------------------------


def test_iter_sandboxes_empty(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    result = mgr._iter_sandboxes()
    assert result == []


def test_iter_sandboxes_with_sandboxes(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb1 = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb1.create()
    sb2 = SandboxDirs("sb2", base_path=sandbox_base / "sandboxes")
    sb2.create()
    result = mgr._iter_sandboxes()
    assert len(result) == 2
    names = [s.sandbox_name for s in result]
    assert names == ["sb1", "sb2"]


# ---------------------------------------------------------------------------
# list_sandboxes
# ---------------------------------------------------------------------------


def test_list_sandboxes_empty(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    result = mgr.list_sandboxes()
    assert result == []


def test_list_sandboxes_with_sandboxes(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb1 = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb1.create()
    sb2 = SandboxDirs("sb2", base_path=sandbox_base / "sandboxes")
    sb2.create()
    result = mgr.list_sandboxes()
    assert result == ["sb1", "sb2"]


# ---------------------------------------------------------------------------
# get_sandbox_containers
# ---------------------------------------------------------------------------


def test_get_sandbox_containers_empty(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    result = mgr.get_sandbox_containers("sb1")
    assert result == []


def test_get_sandbox_containers_with_meta(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))
    result = mgr.get_sandbox_containers("sb1")
    assert len(result) == 1
    assert result[0].container_id == "c1"
    assert result[0].engine == "podman"


def test_get_sandbox_containers_engine_mismatch(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1", engine="docker"))
    with pytest.raises(ValueError, match="Container engine mismatch"):
        mgr.get_sandbox_containers("sb1")


def test_get_sandbox_containers_meta_engine_none(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    meta_dict = make_meta_dict("c1")
    meta_dict["engine"] = None
    sb.add_container_meta(meta_dict)
    with pytest.raises(ValueError, match="Container engine mismatch"):
        mgr.get_sandbox_containers("sb1")


def test_get_sandbox_containers_non_existent(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    result = mgr.get_sandbox_containers("does_not_exist")
    assert result == []


def test_get_sandbox_containers_unknown_keys(sandbox_base: Path) -> None:
    """ContainerMeta is a frozen dataclass; unknown keys in containers.json raise TypeError."""
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    # Write directly to simulate a manually-edited containers.json with unknown keys
    bad_meta = {
        "container_id": "c1",
        "engine": "podman",
        "network": "none",
        "container_name": None,
        "unknown_key": "boom",
    }
    sb.replace_containers_meta([bad_meta])
    with pytest.raises(TypeError):
        mgr.get_sandbox_containers("sb1")


# ---------------------------------------------------------------------------
# list_containers
# ---------------------------------------------------------------------------


def test_list_containers_empty(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    result = mgr.list_containers()
    assert result == []


def test_list_containers_with_data(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1", container_name="worker"))
    result = mgr.list_containers()
    assert len(result) == 1
    assert result[0]["sandbox_name"] == "sb1"
    assert result[0]["container_id"] == "c1"
    assert result[0]["container_name"] == "worker"
    assert result[0]["engine"] == "podman"
    assert result[0]["network"] == "none"


def test_list_containers_multiple_sandboxes(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb1 = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb1.create()
    sb1.add_container_meta(make_meta_dict("c1"))
    sb2 = SandboxDirs("sb2", base_path=sandbox_base / "sandboxes")
    sb2.create()
    sb2.add_container_meta(make_meta_dict("c2"))
    result = mgr.list_containers()
    assert len(result) == 2
    assert result[0]["sandbox_name"] == "sb1"
    assert result[0]["container_id"] == "c1"
    assert result[0]["container_name"] is None
    assert result[0]["engine"] == "podman"
    assert result[0]["network"] == "none"
    assert result[1]["sandbox_name"] == "sb2"
    assert result[1]["container_id"] == "c2"
    assert result[1]["container_name"] is None
    assert result[1]["engine"] == "podman"
    assert result[1]["network"] == "none"


# ---------------------------------------------------------------------------
# create_sandbox
# ---------------------------------------------------------------------------


def test_create_sandbox(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = mgr.create_sandbox("new_sb")
    assert sb.sandbox_name == "new_sb"
    assert sb.exists()


def test_create_sandbox_with_project(sandbox_base: Path, tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "file.txt").write_text("hello")
    mgr = _auto_detect(sandbox_base)
    sb = mgr.create_sandbox("proj_sb", project_path=str(project))
    assert sb.exists()
    assert sb.project_dir.exists()
    assert (sb.project_dir / "file.txt").exists()


def test_create_sandbox_with_force_copy(sandbox_base: Path, tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "file.env").write_text("hello")
    (project / "file2.env").write_text("hello")
    (project / ".gitignore").write_text("*.env\n")
    mgr = _auto_detect(sandbox_base)
    sb = mgr.create_sandbox(
        "fc_sb", project_path=str(project), force_copy_files=["file.env"]
    )
    assert sb.exists()
    assert (sb.project_dir / ".gitignore").exists()
    assert (sb.project_dir / "file.env").exists()
    assert not (sb.project_dir / "file2.env").exists()


def test_create_sandbox_already_exists_no_containers(
    sandbox_base: Path, tmp_path: Path
) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "file.txt").write_text("original")
    mgr = _auto_detect(sandbox_base)
    sb1 = mgr.create_sandbox("existing_sb", project_path=str(project))
    assert sb1.exists()
    assert (sb1.project_dir / "file.txt").read_text() == "original"
    (project / "file.txt").write_text("updated")
    mtime = (project / "file.txt").stat().st_mtime + 5
    os.utime(project / "file.txt", (mtime, mtime))
    sb2 = mgr.create_sandbox("existing_sb", project_path=str(project))
    assert sb2.sandbox_name == "existing_sb"
    assert sb2.exists()
    assert (sb2.project_dir / "file.txt").read_text() == "updated"


def test_create_sandbox_already_exists_with_containers(
    sandbox_base: Path, tmp_path: Path
) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "file.txt").write_text("data")
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs(
        "occupied_sb", base_path=sandbox_base / "sandboxes", project_path=str(project)
    )
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))
    with pytest.raises(RuntimeError, match="already exists and has running containers"):
        mgr.create_sandbox("occupied_sb", project_path=str(project))


def test_create_sandbox_already_exists_no_project_path(
    sandbox_base: Path,
) -> None:
    mgr = _auto_detect(sandbox_base)
    sb1 = mgr.create_sandbox("empty_sb")
    assert sb1.exists()
    sb2 = mgr.create_sandbox("empty_sb")
    assert sb2.sandbox_name == "empty_sb"
    assert sb2.exists()


# ---------------------------------------------------------------------------
# create_container
# ---------------------------------------------------------------------------


def test_create_container_auto_name(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_explicit_name(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container(
            "sb1", image="myimg", container_name=f"{TIZ_WORKER_PREFIX}sb1_leo"
        )
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=f"{TIZ_WORKER_PREFIX}sb1_leo",
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_name_prefix_invalid(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    with pytest.raises(ValueError, match="Container name must start with"):
        mgr.create_container("sb1", image="myimg", container_name="bad_name")


def test_create_container_name_prefix_invalid_2(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    with pytest.raises(ValueError, match="Container name must start with"):
        mgr.create_container(
            "sb1", image="myimg", container_name=f"{TIZ_WORKER_PREFIX}sxx_0"
        )


def test_create_container_name_already_taken(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    meta_dict = make_meta_dict("c1", container_name=f"{TIZ_WORKER_PREFIX}sb1_0")
    sb.add_container_meta(meta_dict)
    with pytest.raises(ValueError, match="is already taken"):
        mgr.create_container(
            "sb1", image="myimg", container_name=f"{TIZ_WORKER_PREFIX}sb1_0"
        )


def test_create_container_auto_name_skips_existing(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(
        make_meta_dict("c1", container_name=f"{TIZ_WORKER_PREFIX}sb1_aaa")
    )
    sb.add_container_meta(
        make_meta_dict("c2", container_name=f"{TIZ_WORKER_PREFIX}sb1_bbb")
    )
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    assert name not in {f"{TIZ_WORKER_PREFIX}sb1_aaa", f"{TIZ_WORKER_PREFIX}sb1_bbb"}


def test_create_container_auto_name_exhausted(sandbox_base: Path) -> None:
    """Cover the for...else fallthrough when all 100 random names are taken."""
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(
        make_meta_dict("c1", container_name=f"{TIZ_WORKER_PREFIX}sb1_zzz")
    )
    with (
        patch("tiz.sandbox_manager.random.choices", return_value=["z", "z", "z"]),
        patch("tiz.sandbox_manager.SandboxContainer"),
    ):
        with pytest.raises(
            RuntimeError, match="Could not generate unique container name"
        ):
            mgr.create_container("sb1", image="myimg")


def test_create_container_network_internet_rewritten(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg", network="internet")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="private",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_network_internet_passed_to_start(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    captured_kwargs: dict = {}

    class CapturingSandboxContainer:
        def __init__(self, **kwargs: object) -> None:
            pass

        def start(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    with patch(
        "tiz.sandbox_manager.SandboxContainer", return_value=CapturingSandboxContainer()
    ):
        mgr.create_container("sb1", image="myimg", network="internet")
    assert captured_kwargs.get("network") == "private"
    assert captured_kwargs.get("use_host_timezone") is True


def test_create_container_network_none_not_rewritten(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg", network="none")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_network_bridge_not_rewritten(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg", network="bridge")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="bridge",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_network_default_none_not_rewritten(
    sandbox_base: Path,
) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_with_extra_args(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container(
            "sb1",
            image="myimg",
            network="bridge",
            read_only_project=True,
            extra_run_args=["--memory=512m"],
        )
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="bridge",
        read_only_project=True,
        mount_project=True,
        extra_run_args=["--memory=512m"],
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_base_path_does_not_exist(sandbox_base: Path) -> None:
    """Cover branch where _manager_base_path.exists() is False."""
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with (
        patch("tiz.sandbox_manager.SandboxContainer", return_value=sc),
        patch.object(type(mgr._manager_base_path), "exists", return_value=False),
    ):
        mgr.create_container("sb1", image="myimg")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_with_tools_worker_dir(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    tools_worker = sandbox_base / TIZ_TOOLS_WORKER_DIR
    tools_worker.mkdir()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=tools_worker,
    )


def test_create_container_base_path_exists_but_no_tools_worker_dir(
    sandbox_base: Path,
) -> None:
    """Cover branch where _manager_base_path exists but tools_worker is not a directory."""
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    # sandbox_base exists (it's a dir), but TIZ_TOOLS_WORKER_DIR does not exist
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg")
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_mount_project_false(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg", mount_project=False)
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=False,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_verbose_1(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg", verbose=1)
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=1,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_verbose_2(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg", verbose=2)
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=2,
        use_host_timezone=True,
        custom_tools_dir=None,
    )


def test_create_container_use_host_timezone_false(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sc = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=sc):
        mgr.create_container("sb1", image="myimg", use_host_timezone=False)
    name = sc.start.call_args[1]["container_name"]
    assert name.startswith(f"{TIZ_WORKER_PREFIX}sb1_")
    assert len(name) == len(f"{TIZ_WORKER_PREFIX}sb1_") + 3
    sc.start.assert_called_once_with(
        image="myimg",
        container_name=name,
        network="none",
        read_only_project=False,
        mount_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=False,
        custom_tools_dir=None,
    )


# ---------------------------------------------------------------------------
# kill_all_containers
# ---------------------------------------------------------------------------


def test_kill_all_containers(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))
    mock_container = MagicMock(spec=SandboxContainer)
    mock_container.container_id = "c1"
    mock_container.is_running.return_value = True
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=mock_container):
        mgr.kill_all_containers("sb1")
    mock_container.stop.assert_called_once()


def test_kill_all_containers_not_running(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))
    mock_container = MagicMock(spec=SandboxContainer)
    mock_container.container_id = "c1"
    mock_container.exists.return_value = False
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=mock_container):
        mgr.kill_all_containers("sb1")
    mock_container.stop.assert_not_called()


def test_kill_all_containers_empty(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    mgr.kill_all_containers("sb1")


def test_kill_all_containers_engine_mismatch(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1", engine="docker"))
    with pytest.raises(ValueError, match="Container engine mismatch"):
        mgr.kill_all_containers("sb1")


# ---------------------------------------------------------------------------
# kill_and_delete_sandbox
# ---------------------------------------------------------------------------


def test_kill_and_delete_sandbox(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))
    mock_container = MagicMock(spec=SandboxContainer)
    mock_container.container_id = "c1"
    mock_container.exists.return_value = True
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=mock_container):
        mgr.kill_and_delete_sandbox("sb1")
    mock_container.stop.assert_called_once()
    assert not sb.exists()


def test_kill_and_delete_sandbox_empty(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    mgr.kill_and_delete_sandbox("sb1")
    assert not sb.exists()


def test_kill_and_delete_sandbox_engine_mismatch(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1", engine="docker"))
    with pytest.raises(ValueError, match="Container engine mismatch"):
        mgr.kill_and_delete_sandbox("sb1")


# ---------------------------------------------------------------------------
# kill_and_delete_all_sandboxes
# ---------------------------------------------------------------------------


def test_kill_and_delete_all_sandboxes(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb1 = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb1.create()
    sb2 = SandboxDirs("sb2", base_path=sandbox_base / "sandboxes")
    sb2.create()
    with patch.object(mgr, "kill_and_delete_sandbox") as mock_kill_delete:
        mgr.kill_and_delete_all_sandboxes()
    assert mock_kill_delete.call_count == 2
    mock_kill_delete.assert_any_call("sb1")
    mock_kill_delete.assert_any_call("sb2")


def test_kill_and_delete_all_sandboxes_empty(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    with patch.object(mgr, "kill_and_delete_sandbox") as mock_kill_delete:
        mgr.kill_and_delete_all_sandboxes()
    mock_kill_delete.assert_not_called()


def test_kill_and_delete_all_sandboxes_race_file_not_found(
    sandbox_base: Path,
) -> None:
    mgr = _auto_detect(sandbox_base)
    sb1 = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb1.create()
    sb2 = SandboxDirs("sb2", base_path=sandbox_base / "sandboxes")
    sb2.create()

    call_count = 0

    def side_effect(name: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise FileNotFoundError(f"Sandbox '{name}' already deleted")

    with patch.object(mgr, "kill_and_delete_sandbox", side_effect=side_effect):
        mgr.kill_and_delete_all_sandboxes()

    assert call_count == 2


# ---------------------------------------------------------------------------
# delete_container
# ---------------------------------------------------------------------------


def test_delete_container(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))
    mock_container = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=mock_container):
        mgr.delete_container("sb1", "c1")
    mock_container.stop.assert_called_once()


def test_delete_container_multiple_meta_match_second(
    sandbox_base: Path,
) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))
    sb.add_container_meta(make_meta_dict("c2"))
    mock_container = MagicMock(spec=SandboxContainer)
    with patch("tiz.sandbox_manager.SandboxContainer", return_value=mock_container):
        mgr.delete_container("sb1", "c2")
    mock_container.stop.assert_called_once()


def test_delete_container_not_found(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    with pytest.raises(FileNotFoundError, match="Container 'c1' not found"):
        mgr.delete_container("sb1", "c1")


def test_delete_container_engine_mismatch(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1", engine="docker"))
    with pytest.raises(ValueError, match="Container engine mismatch"):
        mgr.delete_container("sb1", "c1")


def test_delete_container_container_id_none(sandbox_base: Path) -> None:
    """When meta contains container_id=None, delete_container should not match."""
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(
        {
            "container_id": None,
            "engine": "podman",
            "network": "none",
            "container_name": None,
        }
    )
    with pytest.raises(FileNotFoundError, match="Container 'c1' not found"):
        mgr.delete_container("sb1", "c1")


# ---------------------------------------------------------------------------
# delete_tiz_worker_images
# ---------------------------------------------------------------------------


def test_delete_tiz_worker_images(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    images_result = MagicMock(spec=subprocess.CompletedProcess)
    images_result.returncode = 0
    images_result.stdout = "tiz-worker:latest\nubuntu:22.04\ntiz-worker-python:v2\n"
    rmi_result = MagicMock(spec=subprocess.CompletedProcess)
    rmi_result.returncode = 0
    rmi_result.stdout = ""
    rmi_result.stderr = ""
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append((cmd, kwargs))
        if cmd[1] == "images":
            return images_result
        return rmi_result

    with patch("subprocess.run", side_effect=capture):
        mgr.delete_tiz_worker_images()
    rmi_calls = [(c, k) for c, k in call_args if c[1] == "rmi"]
    assert len(rmi_calls) == 2
    assert "tiz-worker:latest" in rmi_calls[0][0]
    assert "tiz-worker-python:v2" in rmi_calls[1][0]
    for _, kwargs in rmi_calls:
        assert kwargs.get("stdout") is subprocess.DEVNULL
        assert kwargs.get("stderr") is subprocess.DEVNULL
        assert kwargs.get("check") is False
        assert "capture_output" not in kwargs


def test_delete_tiz_worker_images_no_matching(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    images_result = MagicMock(spec=subprocess.CompletedProcess)
    images_result.returncode = 0
    images_result.stdout = "ubuntu:22.04\nnginx:latest\n"
    rmi_result = MagicMock(spec=subprocess.CompletedProcess)
    rmi_result.returncode = 0
    rmi_result.stdout = ""
    rmi_result.stderr = ""
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append((cmd, kwargs))
        if cmd[1] == "images":
            return images_result
        return rmi_result

    with patch("subprocess.run", side_effect=capture):
        mgr.delete_tiz_worker_images()
    rmi_calls = [(c, k) for c, k in call_args if c[1] == "rmi"]
    assert len(rmi_calls) == 0


def test_delete_tiz_worker_images_list_failure(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    images_result = MagicMock(spec=subprocess.CompletedProcess)
    images_result.returncode = 0
    images_result.stdout = ""
    images_result.stderr = "error"
    rmi_result = MagicMock(spec=subprocess.CompletedProcess)
    rmi_result.returncode = 0
    rmi_result.stdout = ""
    rmi_result.stderr = ""
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append((cmd, kwargs))
        if cmd[1] == "images":
            return images_result
        return rmi_result

    with patch("subprocess.run", side_effect=capture):
        mgr.delete_tiz_worker_images()
    rmi_calls = [(c, k) for c, k in call_args if c[1] == "rmi"]
    assert len(rmi_calls) == 0


def test_delete_tiz_worker_images_command_failure(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    images_result = MagicMock(spec=subprocess.CompletedProcess)
    images_result.returncode = 1
    images_result.stdout = ""
    images_result.stderr = "error"

    with patch("subprocess.run", return_value=images_result) as mock_run:
        mgr.delete_tiz_worker_images()
    mock_run.assert_called_once()


def test_delete_tiz_worker_images_dry_run_with_matching(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    images_result = MagicMock(spec=subprocess.CompletedProcess)
    images_result.returncode = 0
    images_result.stdout = "tiz-worker:latest\nubuntu:22.04\ntiz-worker-py:v2\n"
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append((cmd, kwargs))
        return images_result

    with patch("subprocess.run", side_effect=capture):
        result = mgr.delete_tiz_worker_images(dry_run=True)

    assert result == ["tiz-worker:latest", "tiz-worker-py:v2"]
    rmi_calls = [(c, k) for c, k in call_args if c[1] == "rmi"]
    assert len(rmi_calls) == 0


# ---------------------------------------------------------------------------
# build_image
# ---------------------------------------------------------------------------


def test_build_image_invalid_tag(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    with pytest.raises(ValueError, match="Image tag must start with"):
        mgr.build_image(containerfile="FROM ubuntu", tag="wrong-tag")


def test_build_image_success(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    inspect_result = MagicMock(spec=subprocess.CompletedProcess)
    inspect_result.returncode = 1
    inspect_result.stdout = ""
    inspect_result.stderr = ""
    build_result = MagicMock(spec=subprocess.CompletedProcess)
    build_result.returncode = 0
    build_result.stdout = ""
    build_result.stderr = ""
    call_args: list[list[str]] = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        if cmd[1] == "image" and cmd[2] == "inspect":
            return inspect_result
        return build_result

    with patch("subprocess.run", side_effect=capture):
        mgr.build_image(containerfile="FROM ubuntu")
    assert any(c[1] == "build" for c in call_args)
    build_call = next(c for c in call_args if c[1] == "build")
    assert build_call[0] == "podman"
    assert build_call[1] == "build"
    assert build_call[2] == "--force-rm"
    assert build_call[3] == "--pull=newer"
    assert build_call[4] == "--squash"
    assert build_call[5] == "-f"
    assert build_call[6].endswith("/Containerfile")
    assert build_call[7] == "-t"
    assert build_call[8] == "tiz-worker:latest"


def test_build_image_with_timestamp(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    inspect_result = MagicMock(spec=subprocess.CompletedProcess)
    inspect_result.returncode = 1
    inspect_result.stdout = ""
    inspect_result.stderr = ""
    build_result = MagicMock(spec=subprocess.CompletedProcess)
    build_result.returncode = 0
    build_result.stdout = ""
    build_result.stderr = ""
    call_args: list[list[str]] = []
    ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        if cmd[1] == "image" and cmd[2] == "inspect":
            return inspect_result
        return build_result

    with patch("subprocess.run", side_effect=capture):
        mgr.build_image(containerfile="FROM ubuntu", timestamp=ts)

    build_call = next(c for c in call_args if c[1] == "build")
    assert "--timestamp" in build_call
    ts_idx = build_call.index("--timestamp")
    assert build_call[ts_idx + 1] == str(int(ts.timestamp()))


def test_build_image_delete_existing(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    rmi_result = MagicMock(spec=subprocess.CompletedProcess)
    rmi_result.returncode = 0
    rmi_result.stdout = ""
    rmi_result.stderr = ""
    build_result = MagicMock(spec=subprocess.CompletedProcess)
    build_result.returncode = 0
    build_result.stdout = ""
    build_result.stderr = ""
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        if cmd[1] == "rmi":
            return rmi_result
        return build_result

    with patch("subprocess.run", side_effect=capture):
        mgr.build_image(containerfile="FROM ubuntu", delete_existing=True)

    rmi_calls = [c for c in call_args if c[1] == "rmi"]
    assert len(rmi_calls) == 1
    assert "tiz-worker:latest" in rmi_calls[0]
    build_calls = [c for c in call_args if c[1] == "build"]
    assert len(build_calls) == 1


def test_build_image_already_exists(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    inspect_result = MagicMock(spec=subprocess.CompletedProcess)
    inspect_result.returncode = 0
    inspect_result.stdout = "exists"
    inspect_result.stderr = ""
    call_args: list[list[str]] = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        return inspect_result

    with patch("subprocess.run", side_effect=capture):
        mgr.build_image(containerfile="FROM ubuntu")
    build_calls = [c for c in call_args if c[1] == "build"]
    assert len(build_calls) == 0


def test_build_image_failure(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    build_result = MagicMock(spec=subprocess.CompletedProcess)
    build_result.returncode = 1
    build_result.stdout = ""
    build_result.stderr = "build failed\n"

    with patch("subprocess.run", return_value=build_result):
        with pytest.raises(RuntimeError, match="Failed to build image"):
            mgr.build_image(containerfile="FROM ubuntu")


# ---------------------------------------------------------------------------
# cleanup_dead_entries
# ---------------------------------------------------------------------------


def test_cleanup_dead_entries(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("live_c"))
    sb.add_container_meta(make_meta_dict("dead_c"))

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = MagicMock(spec=subprocess.CompletedProcess)
        if cmd[1] == "inspect" and cmd[2] == "live_c":
            result.returncode = 0
        else:
            result.returncode = 1
        result.stdout = ""
        result.stderr = ""
        return result

    with patch("subprocess.run", side_effect=capture):
        removed = mgr.cleanup_dead_entries()

    assert len(removed) == 1
    assert "dead_c" in removed
    assert "live_c" not in removed
    remaining = sb.containers_meta
    assert len(remaining) == 1
    assert remaining[0]["container_id"] == "live_c"


def test_cleanup_dead_entries_none_removed(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch("subprocess.run", side_effect=capture):
        removed = mgr.cleanup_dead_entries()

    assert removed == []
    remaining = sb.containers_meta
    assert len(remaining) == 1
    assert remaining[0]["container_id"] == "c1"


def test_cleanup_dead_entries_no_container_id(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb._write_containers_meta([{"engine": "podman", "network": "none"}])

    with patch("subprocess.run") as mock_run:
        removed = mgr.cleanup_dead_entries()

    inspect_calls = [c for c in mock_run.call_args_list if c[0][0][1] == "inspect"]
    assert len(inspect_calls) == 0
    assert removed == []
    remaining = sb.containers_meta
    assert len(remaining) == 0


# ---------------------------------------------------------------------------
# cleanup_untracked_containers
# ---------------------------------------------------------------------------


def test_cleanup_untracked_containers(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))

    ps_result = MagicMock(spec=subprocess.CompletedProcess)
    ps_result.returncode = 0
    ps_result.stdout = f"c1 tracked_name\nabc123 {TIZ_WORKER_PREFIX}sb1_99\n"
    stop_result = MagicMock(spec=subprocess.CompletedProcess)
    stop_result.returncode = 0
    stop_result.stdout = ""
    stop_result.stderr = ""
    rm_result = MagicMock(spec=subprocess.CompletedProcess)
    rm_result.returncode = 0
    rm_result.stdout = ""
    rm_result.stderr = ""
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        if cmd[1] == "ps":
            return ps_result
        if cmd[1] == "stop":
            return stop_result
        if cmd[1] == "rm":
            return rm_result
        return ps_result

    with patch("subprocess.run", side_effect=capture):
        removed = mgr.cleanup_untracked_containers()

    assert any("abc123" in c for c in call_args if c[1] == "stop")
    assert any("abc123" in c for c in call_args if c[1] == "rm")
    assert all("c1" not in c for c in call_args if c[1] == "stop")
    assert all("c1" not in c for c in call_args if c[1] == "rm")
    assert "abc123" in removed
    assert "c1" not in removed


def test_cleanup_untracked_containers_ps_failure(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))

    ps_result = MagicMock(spec=subprocess.CompletedProcess)
    ps_result.returncode = 1
    ps_result.stdout = ""
    ps_result.stderr = "error"

    with patch("subprocess.run", return_value=ps_result):
        removed = mgr.cleanup_untracked_containers()

    assert removed == []


def test_cleanup_untracked_containers_no_untracked(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb.add_container_meta(make_meta_dict("c1"))

    ps_result = MagicMock(spec=subprocess.CompletedProcess)
    ps_result.returncode = 0
    ps_result.stdout = "c1 tracked_name\nother_worker some_name\n"

    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        return ps_result

    with patch("subprocess.run", side_effect=capture):
        removed = mgr.cleanup_untracked_containers()

    assert removed == []
    ps_calls = [c for c in call_args if c[1] == "ps"]
    assert len(ps_calls) == 1


def test_cleanup_untracked_containers_empty_ps_output(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    ps_result = MagicMock(spec=subprocess.CompletedProcess)
    ps_result.returncode = 0
    ps_result.stdout = ""

    with patch("subprocess.run", return_value=ps_result) as mock_run:
        removed = mgr.cleanup_untracked_containers()

    assert removed == []
    mock_run.assert_called_once()


def test_cleanup_untracked_containers_blank_lines(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()

    ps_result = MagicMock(spec=subprocess.CompletedProcess)
    ps_result.returncode = 0
    ps_result.stdout = f"\n\nabc123 {TIZ_WORKER_PREFIX}sb1_5\n\n"
    stop_result = MagicMock(spec=subprocess.CompletedProcess)
    stop_result.returncode = 0
    stop_result.stdout = ""
    stop_result.stderr = ""
    rm_result = MagicMock(spec=subprocess.CompletedProcess)
    rm_result.returncode = 0
    rm_result.stdout = ""
    rm_result.stderr = ""
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        if cmd[1] == "ps":
            return ps_result
        if cmd[1] == "stop":
            return stop_result
        if cmd[1] == "rm":
            return rm_result
        return ps_result

    with patch("subprocess.run", side_effect=capture):
        removed = mgr.cleanup_untracked_containers()

    assert len(removed) == 1
    assert "abc123" in removed
    subprocess_calls = [c for c in call_args if isinstance(c, list)]
    assert any(c == ["podman", "stop", "abc123"] for c in subprocess_calls)
    assert any(c == ["podman", "rm", "-fi", "abc123"] for c in subprocess_calls)


def test_cleanup_untracked_containers_tracked_id_none(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    sb._write_containers_meta(
        [{"engine": "podman", "network": "none", "container_name": "name_only"}]
    )

    ps_result = MagicMock(spec=subprocess.CompletedProcess)
    ps_result.returncode = 0
    ps_result.stdout = f"abc123 {TIZ_WORKER_PREFIX}sb1_99\n"
    stop_result = MagicMock(spec=subprocess.CompletedProcess)
    stop_result.returncode = 0
    stop_result.stdout = ""
    stop_result.stderr = ""
    rm_result = MagicMock(spec=subprocess.CompletedProcess)
    rm_result.returncode = 0
    rm_result.stdout = ""
    rm_result.stderr = ""
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        if cmd[1] == "ps":
            return ps_result
        if cmd[1] == "stop":
            return stop_result
        if cmd[1] == "rm":
            return rm_result
        return ps_result

    with patch("subprocess.run", side_effect=capture):
        removed = mgr.cleanup_untracked_containers()

    assert len(removed) == 1
    assert "abc123" in removed
    subprocess_calls = [c for c in call_args if isinstance(c, list)]
    assert any(c == ["podman", "stop", "abc123"] for c in subprocess_calls)
    assert any(c == ["podman", "rm", "-fi", "abc123"] for c in subprocess_calls)


# ---------------------------------------------------------------------------
# sandbox_lock
# ---------------------------------------------------------------------------


def test_sandbox_lock_invalid_name(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    with pytest.raises(ValueError, match="Invalid sandbox name"):
        with mgr.sandbox_lock("../evil"):
            pass


def test_sandbox_lock_sandbox_not_found(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    with pytest.raises(FileNotFoundError, match="Sandbox directory .* does not exist"):
        with mgr.sandbox_lock("nonexistent"):
            pass


def test_sandbox_lock_success(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    lock_path = sandbox_base / "sandboxes" / "sb1" / ".manager_lock"
    assert not lock_path.exists()
    with mgr.sandbox_lock("sb1"):
        assert lock_path.exists()
    fd = os.open(str(lock_path), os.O_RDONLY)
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pytest.fail("Lock was not released after exiting context manager")
    finally:
        os.close(fd)


def test_sandbox_lock_timeout(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    lock_path = sandbox_base / "sandboxes" / "sb1" / ".manager_lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="Could not acquire manager lock"):
            with mgr.sandbox_lock("sb1", timeout=0.2, poll_interval=0.05):
                pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_sandbox_lock_releases_on_exception(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    with pytest.raises(ValueError, match="boom"):
        with mgr.sandbox_lock("sb1"):
            raise ValueError("boom")
    lock_path = sandbox_base / "sandboxes" / "sb1" / ".manager_lock"
    fd = os.open(str(lock_path), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pytest.fail("Lock was not released after exception")
    finally:
        os.close(fd)


def test_sandbox_lock_validates_git(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    with patch.object(SandboxDirs, "validate_git_project_dir") as mock_validate:
        with mgr.sandbox_lock("sb1"):
            pass
    mock_validate.assert_called_once()


def test_sandbox_validate_on_exception(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    sb = SandboxDirs("sb1", base_path=sandbox_base / "sandboxes")
    sb.create()
    with pytest.raises(ValueError, match="boom"):
        with patch.object(SandboxDirs, "validate_git_project_dir") as mock_validate:
            with mgr.sandbox_lock("sb1"):
                raise ValueError("boom")
    mock_validate.assert_called_once()


# ---------------------------------------------------------------------------
# sync_from_original / sync_to_original
# ---------------------------------------------------------------------------


def test_sync_from_original(sandbox_base: Path, tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "file.txt").write_text("original")
    mgr = _auto_detect(sandbox_base)
    mgr.create_sandbox("sb_sync_from", project_path=str(project))
    (project / "file.txt").write_text("updated")
    mtime = (project / "file.txt").stat().st_mtime + 5
    os.utime(project / "file.txt", (mtime, mtime))
    with (
        patch.object(mgr, "sandbox_lock", wraps=mgr.sandbox_lock) as mock_lock,
        patch.object(SandboxDirs, "sync_from_original") as mock_sync,
    ):
        mgr.sync_from_original("sb_sync_from")
    mock_lock.assert_called_once_with("sb_sync_from")
    mock_sync.assert_called_once_with(force=False)


def test_sync_from_original_with_force(sandbox_base: Path, tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "file.txt").write_text("original")
    mgr = _auto_detect(sandbox_base)
    mgr.create_sandbox("sb_sync_from_f", project_path=str(project))
    with (
        patch.object(mgr, "sandbox_lock", wraps=mgr.sandbox_lock) as mock_lock,
        patch.object(SandboxDirs, "sync_from_original") as mock_sync,
    ):
        mgr.sync_from_original("sb_sync_from_f", force=True)
    mock_lock.assert_called_once_with("sb_sync_from_f")
    mock_sync.assert_called_once_with(force=True)


def test_sync_to_original(sandbox_base: Path, tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "file.txt").write_text("data")
    mgr = _auto_detect(sandbox_base)
    mgr.create_sandbox("sb_sync_to", project_path=str(project))
    with (
        patch.object(mgr, "sandbox_lock", wraps=mgr.sandbox_lock) as mock_lock,
        patch.object(SandboxDirs, "sync_to_original") as mock_sync,
    ):
        mgr.sync_to_original("sb_sync_to")
    mock_lock.assert_called_once_with("sb_sync_to")
    mock_sync.assert_called_once_with(force=False)


def test_sync_to_original_with_force(sandbox_base: Path, tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "file.txt").write_text("data")
    mgr = _auto_detect(sandbox_base)
    mgr.create_sandbox("sb_sync_to_f", project_path=str(project))
    with (
        patch.object(mgr, "sandbox_lock", wraps=mgr.sandbox_lock) as mock_lock,
        patch.object(SandboxDirs, "sync_to_original") as mock_sync,
    ):
        mgr.sync_to_original("sb_sync_to_f", force=True)
    mock_lock.assert_called_once_with("sb_sync_to_f")
    mock_sync.assert_called_once_with(force=True)


def test_sync_from_original_non_existent(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mgr.sync_from_original("nonexistent")


def test_sync_to_original_non_existent(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mgr.sync_to_original("nonexistent")


# ---------------------------------------------------------------------------
# get_containerfiles_dirs
# ---------------------------------------------------------------------------


def test_get_containerfiles_dirs_returns_list(sandbox_base: Path) -> None:
    _auto_detect(sandbox_base)
    dirs = SandboxManager.get_containerfiles_dirs(sandbox_base)
    assert len(dirs) >= 1
    assert dirs[0] == sandbox_base / "containerfiles"


def test_get_containerfiles_dirs_exception_caught(sandbox_base: Path) -> None:
    with patch(
        "tiz.sandbox_manager.importlib.resources.files",
        side_effect=FileNotFoundError,
    ):
        dirs = SandboxManager.get_containerfiles_dirs(sandbox_base)
        assert len(dirs) == 1
        assert dirs[0] == sandbox_base / "containerfiles"


# ---------------------------------------------------------------------------
# _build_image_internal
# ---------------------------------------------------------------------------


def test_build_image_internal_invalid_tag(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    with pytest.raises(ValueError, match="Image tag must start with 'tiz-worker'"):
        mgr._build_image_internal(containerfile="FROM ubuntu", tag="invalid")


# ---------------------------------------------------------------------------
# _resolve_and_build
# ---------------------------------------------------------------------------


def test_resolve_and_build_recursion_depth(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    with (
        patch.object(mgr, "_build_image_internal"),
        pytest.raises(RuntimeError, match="Maximum recursion depth"),
    ):
        mgr._resolve_and_build(
            containerfile="FROM tiz-worker-base",
            tag="tiz-worker:latest",
            recursion_level=11,
        )


def test_resolve_and_build_non_from_line(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    inspect_result = MagicMock(spec=subprocess.CompletedProcess)
    inspect_result.returncode = 0
    inspect_result.stdout = ""
    inspect_result.stderr = ""
    with (
        patch("subprocess.run", return_value=inspect_result),
        patch.object(mgr, "_build_image_internal") as mock_build,
    ):
        mgr._resolve_and_build(
            containerfile="# comment\nRUN echo hello\nFROM ubuntu:22.04",
            tag="tiz-worker:latest",
        )
    mock_build.assert_called_once()


def test_resolve_and_build_skip_same_tag(sandbox_base: Path) -> None:
    mgr = _auto_detect(sandbox_base)
    inspect_result = MagicMock(spec=subprocess.CompletedProcess)
    inspect_result.returncode = 1
    inspect_result.stdout = ""
    inspect_result.stderr = ""
    with (
        patch("subprocess.run", return_value=inspect_result),
        patch.object(mgr, "_build_image_internal") as mock_build,
    ):
        mgr._resolve_and_build(
            containerfile="FROM tiz-worker-base:v1",
            tag="tiz-worker:latest",
        )
    mock_build.assert_called_once()


def test_resolve_and_build_skip_self_reference(sandbox_base: Path) -> None:
    """When FROM matches the tag's repository, skip dependency resolution."""
    mgr = _auto_detect(sandbox_base)
    with (
        patch.object(mgr, "_build_image_internal") as mock_build,
    ):
        mgr._resolve_and_build(
            containerfile="FROM tiz-worker-base:latest",
            tag="tiz-worker-base:v2",
        )
    mock_build.assert_called_once()


def test_resolve_and_build_dep_found(tmp_path: Path) -> None:
    mgr = _auto_detect(tmp_path)
    cf_dir = tmp_path / "containerfiles"
    cf_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "sandboxes").mkdir(parents=True, exist_ok=True)
    dep_cf_path = cf_dir / "Containerfile.tiz-worker-base"
    dep_cf_path.write_text("FROM scratch")

    dep_inspect = MagicMock(spec=subprocess.CompletedProcess)
    dep_inspect.returncode = 1
    build_result = MagicMock(spec=subprocess.CompletedProcess)
    build_result.returncode = 0

    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append(cmd)
        if cmd[1] == "image" and cmd[2] == "inspect":
            return dep_inspect
        return build_result

    with (
        patch("subprocess.run", side_effect=capture),
        patch.object(
            SandboxManager,
            "get_containerfiles_dirs",
            return_value=[cf_dir],
        ),
    ):
        mgr._resolve_and_build(
            containerfile="FROM tiz-worker-base:latest",
            tag="tiz-worker-thing:latest",
        )

    build_calls = [c for c in call_args if c[1] == "build"]
    assert len(build_calls) == 2


def test_iter_sandboxes_skips_corrupted(
    sandbox_base: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_iter_sandboxes should skip sandboxes whose SandboxDirs init fails."""
    mgr = _auto_detect(sandbox_base)
    sb_good = SandboxDirs("good_sb", base_path=sandbox_base / "sandboxes")
    sb_good.create()
    # Create a corrupted sandbox: write an empty project_path.txt
    sandboxes_dir = sandbox_base / "sandboxes"
    bad_dir = sandboxes_dir / "bad_sb"
    bad_dir.mkdir()
    (bad_dir / "project_path.txt").write_text("", encoding="utf-8")

    result = mgr._iter_sandboxes()
    assert len(result) == 1
    assert result[0].sandbox_name == "good_sb"
    assert "bad_sb" in caplog.text


def test_delete_tiz_worker_images_rmi_failure(sandbox_base: Path) -> None:
    """delete_tiz_worker_images should not add images to result when rmi fails."""
    mgr = _auto_detect(sandbox_base)
    images_result = MagicMock(spec=subprocess.CompletedProcess)
    images_result.returncode = 0
    images_result.stdout = "docker.io/library/tiz-worker:latest\n"
    rmi_result = MagicMock(spec=subprocess.CompletedProcess)
    rmi_result.returncode = 1
    rmi_result.stdout = ""
    rmi_result.stderr = "error"
    call_args = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        call_args.append((cmd, kwargs))
        if cmd[1] == "images":
            return images_result
        return rmi_result

    with patch("subprocess.run", side_effect=capture):
        result = mgr.delete_tiz_worker_images()

    assert result == []
    rmi_calls = [(c, k) for c, k in call_args if c[1] == "rmi"]
    assert len(rmi_calls) == 1
    assert "docker.io/library/tiz-worker:latest" in rmi_calls[0][0]
