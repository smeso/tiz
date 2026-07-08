"""Tests for src/tiz/sandbox_dirs.py."""

import fcntl
import os
import shutil
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import git
import pytest

from tiz.sandbox_dirs import TIZ_PATCHES_DIR, SandboxDirs

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _init_git_repo(
    repo_path: Path,
    *,
    author_name: str = "Test",
    author_email: str = "test@example.com",
    initial_branch: str = "main",
) -> None:
    """Create a git repo."""
    repo_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "-c",
            f"init.defaultBranch={initial_branch}",
            "init",
            "--initial-branch",
            initial_branch,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", author_email],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", author_name],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


def _commit_file(
    repo_path: Path,
    name: str,
    content: str,
    *,
    author_name: str = "Test",
    author_email: str = "test@example.com",
) -> None:
    """Write *name* with *content* and commit it."""
    (repo_path / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            f"add {name}",
            "--author",
            f"{author_name} <{author_email}>",
        ],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def sandbox_base(tmp_path: Path) -> Path:
    return tmp_path / "sandboxes"


@pytest.fixture
def fake_original_project(tmp_path: Path) -> Path:
    proj = tmp_path / "original_project"
    proj.mkdir(parents=True)
    (proj / "README.md").write_text("# Hello", encoding="utf-8")
    (proj / "main.py").write_text("print('hello')\n", encoding="utf-8")
    return proj


@pytest.fixture
def git_original_project(tmp_path: Path) -> Generator[Path, None, None]:
    proj = tmp_path / "git_original"
    _init_git_repo(proj)
    _commit_file(proj, "README.md", "# Root")
    subdir = proj / "pkg"
    subdir.mkdir()
    (subdir / "__init__.py").write_text("", encoding="utf-8")
    _commit_file(proj, "pkg/__init__.py", "")
    yield proj


# ===========================================================================
# SandboxDirs – construction
# ===========================================================================


def test_invalid_name() -> None:
    with pytest.raises(ValueError, match="invalid"):
        SandboxDirs("../evil", base_path=Path("/tmp"))

    with pytest.raises(ValueError, match="invalid"):
        SandboxDirs("foo/bar", base_path=Path("/tmp"))

    with pytest.raises(ValueError, match="invalid"):
        SandboxDirs("foo\x00bar", base_path=Path("/tmp"))

    with pytest.raises(ValueError, match="invalid"):
        SandboxDirs("/foobar", base_path=Path("/tmp"))


def test_valid_name() -> None:
    s = SandboxDirs("my-sandbox_1", base_path=Path("/tmp"))
    assert s._sandbox_name == "my-sandbox_1"
    assert s.sandbox_name == "my-sandbox_1"


def test_patches_dir() -> None:
    assert TIZ_PATCHES_DIR == "tiz_patches"


def test_custom_base_path(sandbox_base: Path) -> None:
    s = SandboxDirs("test-sb", base_path=sandbox_base)
    assert s._base_path == sandbox_base


# ===========================================================================
# SandboxDirs – create / exists / remove
# ===========================================================================


def test_create_basic(sandbox_base: Path) -> None:
    s = SandboxDirs("sb1", base_path=sandbox_base)
    s.create()
    assert s.exists()
    assert s.shared_general_dir.exists()
    assert not (s._sandbox_dir / "project").exists()
    assert (s._sandbox_dir / "shared").exists()
    assert (s._sandbox_dir / "containers.json").exists()
    assert (s._sandbox_dir / "containers.json").read_text().strip() == "[]"


def test_create_already_exists(sandbox_base: Path) -> None:
    s = SandboxDirs("sb1", base_path=sandbox_base)
    s.create()
    with pytest.raises(FileExistsError):
        s.create()


def test_create_with_project(fake_original_project: Path, sandbox_base: Path) -> None:
    s = SandboxDirs(
        "sb2", project_path=str(fake_original_project), base_path=sandbox_base
    )
    s.create()
    assert s.exists()
    assert s.shared_general_dir.exists()
    assert (s._sandbox_dir / "shared").exists()
    assert (s._sandbox_dir / "containers.json").exists()
    assert (s._sandbox_dir / "containers.json").read_text().strip() == "[]"
    assert s.project_dir.exists()
    assert (s.project_dir / "README.md").exists()
    assert not (s.project_dir / ".git").exists()
    assert s.original_project_path == fake_original_project.resolve()


def test_remove(sandbox_base: Path) -> None:
    s = SandboxDirs("sb5", base_path=sandbox_base)
    s.create()
    s.remove()
    assert not s.exists()


def test_shared_container_dir_invalid_name(sandbox_base: Path) -> None:
    s = SandboxDirs("sb_inv_container", base_path=sandbox_base)
    s.create()
    with pytest.raises(ValueError, match="invalid"):
        s.shared_container_dir("../etc")
    with pytest.raises(ValueError, match="invalid"):
        s.shared_container_dir("foo/bar")


def test_remove_corrupted_metadata(sandbox_base: Path) -> None:
    s = SandboxDirs("sb_corrupt", base_path=sandbox_base)
    s.create()
    s._meta_containers_path.write_text("not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupted"):
        s.remove()
    s.remove(force=True)
    assert not s.exists()


def test_remove_with_containers_fails(sandbox_base: Path) -> None:
    s = SandboxDirs("sb6", base_path=sandbox_base)
    s.create()
    s.add_container_meta({"container_id": "abc"})
    with pytest.raises(RuntimeError, match="container metadata"):
        s.remove()


def test_remove_with_containers_force(sandbox_base: Path) -> None:
    s = SandboxDirs("sb6b", base_path=sandbox_base)
    s.create()
    s.add_container_meta({"container_id": "abc"})
    s.remove(force=True)
    assert not s.exists()


def test_reopen_existing_with_different_project_raises(
    fake_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb7", project_path=str(fake_original_project), base_path=sandbox_base
    )
    s.create()
    other_proj = sandbox_base.parent / "other_project"
    other_proj.mkdir(parents=True)
    with pytest.raises(ValueError, match="was created with project path"):
        SandboxDirs("sb7", project_path=str(other_proj), base_path=sandbox_base)


def test_nonexistent_original_project_raises(sandbox_base: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SandboxDirs(
            "sb8",
            project_path="/nonexistent/path/xyz",
            base_path=sandbox_base,
        )


def test_existing_sandbox_with_missing_original_project(
    sandbox_base: Path, fake_original_project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Construct a SandboxDirs for an existing sandbox whose original project
    path no longer exists. The constructor should warn and set
    original_project_path to None."""
    s = SandboxDirs(
        "sb_missing_orig",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    s.create()
    # Now remove the original project
    shutil.rmtree(fake_original_project)
    assert not fake_original_project.exists()
    # Re-constructing should not raise
    s2 = SandboxDirs(
        "sb_missing_orig",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    assert s2.original_project_path is None
    # Verify warning was logged
    assert any(
        "Original project path" in r.message and "does not exist" in r.message
        for r in caplog.records
    )


# ===========================================================================
# SandboxDirs – list_all
# ===========================================================================


def test_list_all_empty(sandbox_base: Path) -> None:
    assert SandboxDirs.list_all(base_path=sandbox_base) == []


def test_list_all(sandbox_base: Path) -> None:
    SandboxDirs("bbb", base_path=sandbox_base).create()
    SandboxDirs("aaa", base_path=sandbox_base).create()
    # Create out of order to verify list_all sorts regardless of creation order
    assert SandboxDirs.list_all(base_path=sandbox_base) == ["aaa", "bbb"]


def test_list_all_nonexistent_base() -> None:
    base = Path("/does/not/exist/xyz")
    assert SandboxDirs.list_all(base_path=base) == []


# ===========================================================================
# SandboxDirs – lock
# ===========================================================================


def test_lock_fail_on_nonexistent_sandbox(sandbox_base: Path) -> None:
    s = SandboxDirs("sb9", base_path=sandbox_base)
    with pytest.raises(FileNotFoundError):
        s.sandbox_lock().__enter__()


def test_lock_acquire(sandbox_base: Path) -> None:
    s = SandboxDirs("sb10", base_path=sandbox_base)
    s.create()
    with s.sandbox_lock():
        lock_path = s._sandbox_dir / ".lock"
        assert lock_path.exists()


def test_lock_timeout(tmp_path: Path) -> None:
    s = SandboxDirs("sb11", base_path=tmp_path)
    s.create()
    fd = os.open(str(s._lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with (  # noqa: SIM117 (try/finally prevents full merge)
            pytest.raises(TimeoutError),
            patch("tiz.sandbox_dirs.SandboxDirs._LOCK_TIMEOUT", 0.05),
            patch("tiz.sandbox_dirs.SandboxDirs._LOCK_POLL_INTERVAL", 0.01),
        ):
            with s.sandbox_lock():
                raise AssertionError()  # pragma: no cover
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ===========================================================================
# SandboxDirs – container metadata
# ===========================================================================


def test_container_meta_crud(sandbox_base: Path) -> None:
    s = SandboxDirs("sb12", base_path=sandbox_base)
    s.create()
    assert s.containers_meta == []

    s.add_container_meta({"container_id": "c1", "image": "nginx"})
    assert len(s.containers_meta) == 1

    s.add_container_meta({"container_id": "c2", "image": "redis"})
    assert len(s.containers_meta) == 2

    s.remove_container_meta("c1")
    assert len(s.containers_meta) == 1
    assert s.containers_meta[0]["container_id"] == "c2"


def test_add_container_meta_missing_key(sandbox_base: Path) -> None:
    s = SandboxDirs("sb13", base_path=sandbox_base)
    s.create()
    with pytest.raises(ValueError, match="container_id"):
        s.add_container_meta({"image": "nginx"})


# ===========================================================================
# SandboxDirs – git detection
# ===========================================================================


def test_is_git_repo_true(git_original_project: Path) -> None:
    assert SandboxDirs.is_git_repo(git_original_project) is True


def test_is_git_repo_false(tmp_path: Path) -> None:
    non_git = tmp_path / "not_repo"
    non_git.mkdir()
    assert SandboxDirs.is_git_repo(non_git) is False


def test_is_git_repo_nonexistent(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist"
    assert SandboxDirs.is_git_repo(path) is False


# ===========================================================================
# SandboxDirs – _matches_gitignore
# ===========================================================================


def test_matches_gitignore_empty() -> None:
    assert SandboxDirs._matches_gitignore(Path("foo.pyc"), []) is False


def test_matches_gitignore_simple() -> None:
    patterns = ["*.pyc", "__pycache__/"]
    assert SandboxDirs._matches_gitignore(Path("foo.pyc"), patterns) is True
    assert SandboxDirs._matches_gitignore(Path("sub/foo.pyc"), patterns) is True
    assert (
        SandboxDirs._matches_gitignore(Path("__pycache__"), patterns, is_dir=True)
        is True
    )
    assert (
        SandboxDirs._matches_gitignore(Path("__pycache__/something"), patterns) is True
    )
    assert (
        SandboxDirs._matches_gitignore(Path("some/__pycache__/something"), patterns)
        is True
    )
    assert SandboxDirs._matches_gitignore(Path("something"), patterns) is False
    assert SandboxDirs._matches_gitignore(Path("__pycache__"), patterns) is False


def test_matches_gitignore_slash_prefix() -> None:
    assert SandboxDirs._matches_gitignore(Path("build"), ["/build"]) is True


def test_matches_gitignore_path_pattern() -> None:
    assert SandboxDirs._matches_gitignore(Path("src/foo.pyc"), ["src/*.pyc"]) is True
    assert SandboxDirs._matches_gitignore(Path("foo.pyc"), ["src/*.pyc"]) is False
    assert SandboxDirs._matches_gitignore(Path("src/foo.py"), ["src/*.pyc"]) is False
    assert (
        SandboxDirs._matches_gitignore(Path("src/no/foo.pyc"), ["src/*.pyc"]) is False
    )
    assert SandboxDirs._matches_gitignore(Path("src/foo.pyc"), ["src/**.pyc"]) is True
    assert (
        SandboxDirs._matches_gitignore(Path("src/no/foo.pyc"), ["src/**.pyc"]) is False
    )
    assert SandboxDirs._matches_gitignore(Path("foo.pyc"), ["src/**.pyc"]) is False
    assert SandboxDirs._matches_gitignore(Path("src/foo.pyc"), ["src/***.pyc"]) is True
    assert (
        SandboxDirs._matches_gitignore(Path("src/no/foo.pyc"), ["src/***.pyc"]) is False
    )
    assert SandboxDirs._matches_gitignore(Path("foo.pyc"), ["src/***.pyc"]) is False
    assert (
        SandboxDirs._matches_gitignore(Path("src/no/foo.pyc"), ["src/**/*.pyc"]) is True
    )
    assert SandboxDirs._matches_gitignore(Path("src/foo.pyc"), ["src/**/*.pyc"]) is True
    assert SandboxDirs._matches_gitignore(Path("foo.pyc"), ["src/**/*.pyc"]) is False


def test_matches_gitignore_negation_ignored() -> None:
    assert (
        SandboxDirs._matches_gitignore(Path("important.log"), ["!important.log"])
        is False
    )
    assert (
        SandboxDirs._matches_gitignore(
            Path("important.log"), ["*.log", "!important.log"]
        )
        is False
    )
    assert (
        SandboxDirs._matches_gitignore(
            Path("notimportant.log"), ["*.log", "!important.log"]
        )
        is True
    )
    assert SandboxDirs._matches_gitignore(Path("important.log"), ["*.log"]) is True
    assert (
        SandboxDirs._matches_gitignore(
            Path("important.log"), ["!important.log", "*.log"]
        )
        is True
    )


def test_matches_gitignore_rel_name_match() -> None:
    patterns = ["*.pyc"]
    assert SandboxDirs._matches_gitignore(Path("foo.pyc"), patterns) is True


def test_matches_gitignore_slash_trailing() -> None:
    patterns = ["build/"]
    assert SandboxDirs._matches_gitignore(Path("build"), patterns, is_dir=True) is True
    assert (
        SandboxDirs._matches_gitignore(Path("build"), patterns, is_dir=False) is False
    )
    assert SandboxDirs._matches_gitignore(Path("build/file.txt"), patterns) is True


def test_matches_gitignore_pattern_with_slash_no_match() -> None:
    patterns = ["src/*.pyc"]
    assert SandboxDirs._matches_gitignore(Path("build/output.log"), patterns) is False


def test_matches_gitignore_double_star() -> None:
    patterns = ["**/temp"]
    assert SandboxDirs._matches_gitignore(Path("temp"), patterns) is True
    assert SandboxDirs._matches_gitignore(Path("sub/temp"), patterns) is True
    assert SandboxDirs._matches_gitignore(Path("a/b/temp"), patterns) is True
    assert SandboxDirs._matches_gitignore(Path("a/b/txemp"), patterns) is False


def test_matches_gitignore_double_star_with_prefix() -> None:
    patterns = ["dir/**/temp"]
    assert SandboxDirs._matches_gitignore(Path("dir/temp"), patterns) is True
    assert SandboxDirs._matches_gitignore(Path("dir/sub/temp"), patterns) is True
    assert SandboxDirs._matches_gitignore(Path("dir/s/u/b/temp"), patterns) is True
    assert SandboxDirs._matches_gitignore(Path("dir/sub/texmp"), patterns) is False
    assert SandboxDirs._matches_gitignore(Path("dir/something"), patterns) is False


def test_matches_gitignore_single_star_in_double_star_context() -> None:
    patterns = ["*.py"]
    assert SandboxDirs._matches_gitignore(Path("foo.py"), patterns) is True


def test_matches_gitignore_empty_after_strip() -> None:
    patterns = ["/", "!"]
    assert SandboxDirs._matches_gitignore(Path("foo"), patterns) is False


def test_matches_gitignore_double_star_no_match_at_all() -> None:
    patterns = ["**/nonexistent"]
    assert SandboxDirs._matches_gitignore(Path("src/main.py"), patterns) is False


def test_matches_gitignore_trailing_slash_no_match_file() -> None:
    """Pattern with trailing / should not match a file with the same name."""
    patterns = ["build/"]
    assert (
        SandboxDirs._matches_gitignore(Path("build"), patterns, is_dir=False) is False
    )


def test_matches_gitignore_trailing_slash_path_no_match_file() -> None:
    """Pattern with trailing / in a path should not match a file."""
    patterns = ["dir/build/"]
    assert (
        SandboxDirs._matches_gitignore(Path("dir/build"), patterns, is_dir=False)
        is False
    )


def test_matches_gitignore_double_star_trailing_slash_no_match_file() -> None:
    """Pattern like dir/**/foo/ should not match a file."""
    patterns = ["dir/**/foo/"]
    assert (
        SandboxDirs._matches_gitignore(Path("dir/sub/foo"), patterns, is_dir=False)
        is False
    )


def test_matches_gitignore_double_star_prefix_no_match() -> None:
    patterns = ["other/**/test"]
    assert SandboxDirs._matches_gitignore(Path("src/some/test"), patterns) is False


# ===========================================================================
# SandboxDirs – force_copy_files
# ===========================================================================


def test_create_force_copy_files(
    sandbox_base: Path, git_original_project: Path
) -> None:
    untracked = git_original_project / "secret.env"
    untracked.write_text("KEY=val\n", encoding="utf-8")
    s = SandboxDirs(
        "sb14",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    with pytest.raises(RuntimeError, match="is dirty"):
        s.create()


def test_create_force_copy_files_gitignore(
    sandbox_base: Path, git_original_project: Path
) -> None:
    untracked = git_original_project / "secret.env"
    untracked.write_text("KEY=val\n", encoding="utf-8")
    _commit_file(git_original_project, ".gitignore", "*.env")
    s = SandboxDirs(
        "sb14",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create(force_copy_files=["*.env"])
    assert (s.project_dir / "secret.env").exists()


def test_create_force_copy_files_non_git(
    fake_original_project: Path, sandbox_base: Path
) -> None:
    (fake_original_project / "ignored.tmp").write_text("x\n", encoding="utf-8")
    (fake_original_project / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    s = SandboxDirs(
        "sb15",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    s.create(force_copy_files=["*.tmp"])
    assert (s.project_dir / "ignored.tmp").exists()


# ===========================================================================
# SandboxDirs – remove_git_hooks
# ===========================================================================


def test_remove_git_hooks(sandbox_base: Path, git_original_project: Path) -> None:
    s = SandboxDirs(
        "sb16", project_path=str(git_original_project), base_path=sandbox_base
    )
    s.create()
    hooks_dir = s.project_dir / ".git" / "hooks"
    hook = hooks_dir / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    sample = hooks_dir / "pre-commit.sample"
    sample.write_text("#!/bin/sh\n", encoding="utf-8")
    s.remove_git_hooks()
    assert not hook.exists()
    assert sample.exists()


def test_remove_git_hooks_not_a_repo(
    sandbox_base: Path, fake_original_project: Path
) -> None:
    s = SandboxDirs(
        "sb17", project_path=str(fake_original_project), base_path=sandbox_base
    )
    s.create()
    # should silently do nothing
    s.remove_git_hooks()


def test_existing_sandbox_without_project_path(sandbox_base: Path) -> None:
    sb_dir = sandbox_base / "sb62"
    sb_dir.mkdir(parents=True)
    (sb_dir / "shared").mkdir()
    (sb_dir / "containers.json").write_text("[]", encoding="utf-8")
    s = SandboxDirs("sb62", base_path=sandbox_base)
    assert s.original_project_path is None


# ===========================================================================
# SandboxDirs – sync_from_original
# ===========================================================================


def test_sync_from_original_no_path(sandbox_base: Path) -> None:
    s = SandboxDirs("sb24", base_path=sandbox_base)
    s.create()
    with pytest.raises(ValueError, match="Original project path"):
        s.sync_from_original()


def test_sync_from_original_untracked(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb25",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (s.project_dir / "README.md").write_text("modified\n", encoding="utf-8")
    repo = git.Repo(s.project_dir)
    assert repo.is_dirty(untracked_files=True)
    s.sync_from_original()
    assert not repo.is_dirty(untracked_files=True)
    assert (s.project_dir / "README.md").read_text() == "modified\n"
    repo.close()


def test_sync_from_original_git(git_original_project: Path, sandbox_base: Path) -> None:
    s = SandboxDirs(
        "sb26",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(git_original_project, "new.txt", "hello")
    s.sync_from_original()
    assert (s.project_dir / "new.txt").exists()
    assert (s.project_dir / "new.txt").read_text() == "hello"
    repo = git.Repo(s.project_dir)
    assert not repo.is_dirty(untracked_files=True)
    orig_repo = git.Repo(git_original_project)
    assert repo.active_branch.commit.hexsha == orig_repo.active_branch.commit.hexsha


def test_sync_from_original_dirty_original_raises(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb26b",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (git_original_project / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        s.sync_from_original()
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        s.sync_from_original(force=True)


def test_sync_from_original_copy_non_git(tmp_path: Path, sandbox_base: Path) -> None:
    orig = tmp_path / "orig_sync"
    orig.mkdir()
    (orig / "a.txt").write_text("a\n", encoding="utf-8")
    s = SandboxDirs("sb27", project_path=str(orig), base_path=sandbox_base)
    s.create()
    time.sleep(0.05)
    (orig / "a.txt").write_text("b\n", encoding="utf-8")
    s.sync_from_original()
    assert (s.project_dir / "a.txt").read_text(encoding="utf-8") == "b\n"


def test_sync_from_original_detached_head(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb27b",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    subprocess.run(
        ["git", "checkout", "--detach"],
        cwd=git_original_project,
        check=True,
        capture_output=True,
    )
    with pytest.raises(RuntimeError, match="detached HEAD"):
        s.sync_from_original()


def test_sync_from_original_no_sandbox_project_dir(
    sandbox_base: Path, fake_original_project: Path
) -> None:
    s = SandboxDirs(
        "sb27c", project_path=str(fake_original_project), base_path=sandbox_base
    )
    s.create()
    shutil.rmtree(s.project_dir)
    with pytest.raises(RuntimeError, match="project directory does not exist"):
        s.sync_from_original()


def test_sync_from_original_merge_conflict_then_force(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb27d",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(git_original_project, "README.md", "# changed in original\n")
    pd = s.project_dir
    _commit_file(pd, "README.md", "# changed in sandbox\n")
    with pytest.raises(RuntimeError, match="not merge or rebase"):
        s.sync_from_original()
    assert (s.project_dir / "README.md").read_text() == "# changed in sandbox\n"
    s.sync_from_original(force=True)
    assert (s.project_dir / "README.md").read_text() == "# changed in original\n"


# ===========================================================================
# SandboxDirs – sync_to_original
# ===========================================================================


def test_sync_to_original_no_path(sandbox_base: Path) -> None:
    s = SandboxDirs("sb28", base_path=sandbox_base)
    s.create()
    with pytest.raises(ValueError, match="Original project path"):
        s.sync_to_original()


def test_sync_to_original_git(git_original_project: Path, sandbox_base: Path) -> None:
    s = SandboxDirs(
        "sb29",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (s.project_dir / "feature.txt").write_text("feature content\n", encoding="utf-8")
    assert not (git_original_project / "feature.txt").exists()
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "feature.txt").read_text() == "feature content\n"


def test_sync_to_original_dirty_original_raises(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb30",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (git_original_project / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="is dirty"):
        s.sync_to_original()
    assert not (s.project_dir / "dirty.txt").exists()


def test_sync_to_original_git_force(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb31",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(s.project_dir, "branch_file.txt", "branch content\n")
    _commit_file(git_original_project, "original_diverge.txt", "orig\n")

    with pytest.raises(RuntimeError, match="not fast forward"):
        s.sync_to_original()
    assert not (git_original_project / "branch_file.txt").exists()
    assert (git_original_project / "original_diverge.txt").exists()
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original(force=True)
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "branch_file.txt").read_text() == "branch content\n"
    assert not (git_original_project / "original_diverge.txt").exists()


def test_sync_from_original_with_rebase(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb31b",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd, "branch_file.txt", "branch content\n")
    _commit_file(git_original_project, "original_diverge.txt", "orig\n")

    s.sync_from_original()
    assert (pd / "branch_file.txt").read_text() == "branch content\n"
    assert (pd / "original_diverge.txt").read_text() == "orig\n"


def test_sync_to_original_no_changes_non_git(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_sync_to_nochange"
    orig.mkdir()
    (orig / "a.txt").write_text("a\n", encoding="utf-8")
    s = SandboxDirs("sb32b", project_path=str(orig), base_path=sandbox_base)
    s.create()
    assert s.may_need_to_sync_to_original() is False
    s.sync_to_original()
    patches_dir = orig / TIZ_PATCHES_DIR
    assert not patches_dir.exists()


# ===========================================================================
# SandboxDirs – accessors
# ===========================================================================


def test_get_project_dir(sandbox_base: Path) -> None:
    s = SandboxDirs("sb35", base_path=sandbox_base)
    s.create()
    assert s.project_dir == (sandbox_base / "sb35" / "project")


def test_get_shared_general_dir(sandbox_base: Path) -> None:
    s = SandboxDirs("sb36g", base_path=sandbox_base)
    s.create()
    assert s.shared_general_dir == (sandbox_base / "sb36g" / "shared" / "general")


def test_shared_container_dir(sandbox_base: Path) -> None:
    s = SandboxDirs("sb36c", base_path=sandbox_base)
    s.create()
    result = s.shared_container_dir("my-container")
    assert result == (sandbox_base / "sb36c" / "shared" / "specific" / "my-container")


def test_get_original_project_path(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs("sb37", base_path=sandbox_base)
    assert s.original_project_path is None
    s3 = SandboxDirs(
        "sb37c", project_path=str(git_original_project), base_path=sandbox_base
    )
    assert s3.original_project_path == git_original_project.resolve()


# ===========================================================================
# SandboxDirs – _copytree_update
# ===========================================================================


def test_copytree_update_new_file(tmp_path: Path) -> None:
    src = tmp_path / "src_ctu"
    dst = tmp_path / "dst_ctu"
    src.mkdir()
    dst.mkdir()
    (src / "x.txt").write_text("hello\n", encoding="utf-8")
    s = SandboxDirs("sb38", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert (dst / "x.txt").exists()


def test_copytree_update_deletes_removed(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src_ctu2"
    dst = tmp_path / "dst_ctu2"
    src.mkdir()
    dst.mkdir()
    (dst / "old.txt").write_text("old\n", encoding="utf-8")
    s = SandboxDirs("sb39", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert not (dst / "old.txt").exists()


def test_copytree_update_skipped_same_mtime_and_size(tmp_path: Path) -> None:
    src = tmp_path / "src_ctu3"
    dst = tmp_path / "dst_ctu3"
    src.mkdir()
    dst.mkdir()
    src_file = src / "x.txt"
    dst_file = dst / "x.txt"
    src_file.write_text("hello!\n", encoding="utf-8")
    dst_file.write_text("hello1\n", encoding="utf-8")
    # Force same mtime and same size (both 7 bytes)
    mtime = 1000000
    os.utime(src_file, (mtime, mtime))
    os.utime(dst_file, (mtime, mtime))
    s = SandboxDirs("sb39b", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert dst_file.exists()
    assert dst_file.read_text() == "hello1\n"


def test_copytree_update_non_file_items(tmp_path: Path) -> None:
    src = tmp_path / "src_ctu4"
    dst = tmp_path / "dst_ctu4"
    src.mkdir()
    dst.mkdir()
    (src / "sym").mkdir()
    s = SandboxDirs("sb39c", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert not (dst / "sym").exists()
    (src / "sym" / "test").write_text("hello\n", encoding="utf-8")
    s._copytree_update(src, dst)
    assert (dst / "sym").exists()
    assert (dst / "sym" / "test").exists()


def test_copytree_update_empty_dirs(tmp_path: Path) -> None:
    src = tmp_path / "src_ctu5"
    dst = tmp_path / "dst_ctu5"
    src.mkdir()
    dst.mkdir()
    (dst / "empty_dir").mkdir()
    (dst / "nested" / "deep").mkdir(parents=True)
    (dst / "nested" / "deep" / "old.txt").write_text("old\n", encoding="utf-8")
    s = SandboxDirs("sb39d", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert not (dst / "nested").exists()
    assert not (dst / "empty_dir").exists()


def test_copytree_update_gitignore(tmp_path: Path) -> None:
    src = tmp_path / "src_ctu6"
    dst = tmp_path / "dst_ctu6"
    src.mkdir()
    dst.mkdir()
    (src / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (src / "keep.txt").write_text("keep\n", encoding="utf-8")
    (src / "skip.log").write_text("skip\n", encoding="utf-8")
    s = SandboxDirs("sb39e", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert (dst / "keep.txt").exists()
    assert (dst / ".gitignore").exists()
    assert not (dst / "skip.log").exists()
    assert (src / "skip.log").exists()


def test_copytree_update_dst_not_exists(tmp_path: Path) -> None:
    src = tmp_path / "src_no_dst"
    src.mkdir()
    dst = tmp_path / "dst_no_dst"
    s = SandboxDirs("sb75", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert dst.exists()


def test_copytree_update_different_mtime(tmp_path: Path) -> None:
    src = tmp_path / "src_ctu3"
    dst = tmp_path / "dst_ctu3"
    src.mkdir()
    dst.mkdir()
    src_file = src / "x.txt"
    dst_file = dst / "x.txt"
    src_file.write_text("hello\n", encoding="utf-8")
    dst_file.write_text("hello1\n", encoding="utf-8")
    mtime = 1000000
    os.utime(src_file, (mtime + 1, mtime + 1))
    os.utime(dst_file, (mtime, mtime))
    s = SandboxDirs("sb39b2", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert dst_file.exists()
    assert dst_file.read_text() == "hello\n"


def test_copytree_update_gitignore_existing(tmp_path: Path) -> None:
    src = tmp_path / "src_ctu6"
    dst = tmp_path / "dst_ctu6"
    src.mkdir()
    dst.mkdir()
    (src / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (src / "keep.txt").write_text("keep\n", encoding="utf-8")
    (src / "skip.log").write_text("skip\n", encoding="utf-8")
    (dst / "skip.log").write_text("keep1\n", encoding="utf-8")
    (dst / "keep.log").write_text("keep2\n", encoding="utf-8")
    s = SandboxDirs("sb39e2", base_path=tmp_path)
    s._copytree_update(src, dst)
    assert (dst / "keep.txt").read_text() == "keep\n"
    assert (dst / ".gitignore").exists()
    assert (src / "skip.log").read_text() == "skip\n"
    assert (dst / "skip.log").read_text() == "keep1\n"
    assert (dst / "keep.log").read_text() == "keep2\n"


# ===========================================================================
# SandboxDirs – _read_gitignore_patterns
# ===========================================================================


def test_read_gitignore_patterns_no_file(tmp_path: Path) -> None:
    assert SandboxDirs._read_gitignore_patterns(tmp_path / "nonexistent") == [
        f"{TIZ_PATCHES_DIR}"
    ]


def test_read_gitignore_patterns(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "*.pyc\n# comment\n\n__pycache__/\n!important.log\n", encoding="utf-8"
    )
    patterns = SandboxDirs._read_gitignore_patterns(tmp_path)
    assert patterns == [f"{TIZ_PATCHES_DIR}", "*.pyc", "__pycache__/", "!important.log"]


# ===========================================================================
# SandboxDirs – remove_git_hooks symlink attack
# ===========================================================================


def test_remove_git_hooks_symlink_attack(tmp_path: Path, sandbox_base: Path) -> None:
    orig = tmp_path / "orig_hooks"
    _init_git_repo(orig)
    _commit_file(orig, "README.md", "# root")
    s = SandboxDirs("sb41", project_path=str(orig), base_path=sandbox_base)
    s.create()
    hooks_dir = s.project_dir / ".git" / "hooks"
    shutil.rmtree(hooks_dir)
    outside = tmp_path / "outside_project"
    outside.mkdir()
    evil_hook = outside / "evil_hook"
    evil_hook.write_text("# evil\n", encoding="utf-8")
    hooks_dir.symlink_to(outside)
    s.remove_git_hooks()
    assert evil_hook.exists()
    assert not hooks_dir.exists()


def test_remove_git_hooks_git_symlink_attack(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_hooks"
    _init_git_repo(orig)
    _commit_file(orig, "README.md", "# root")
    s = SandboxDirs("sb41", project_path=str(orig), base_path=sandbox_base)
    s.create()
    pd = s.project_dir
    new_git = (pd / ".git").rename(tmp_path / "outside_project")
    (pd / ".git").symlink_to(new_git)
    with pytest.raises(RuntimeError, match="resolves outside"):
        s.remove_git_hooks()


def test_remove_git_hooks_no_hooks_dir(
    sandbox_base: Path, git_original_project: Path
) -> None:
    s = SandboxDirs(
        "sb41b", project_path=str(git_original_project), base_path=sandbox_base
    )
    s.create()
    hooks_dir = s.project_dir / ".git" / "hooks"
    shutil.rmtree(hooks_dir)
    # should silently do nothing
    s.remove_git_hooks()


def test_remove_git_hooks_submodule(tmp_path: Path, sandbox_base: Path) -> None:
    orig = tmp_path / "orig_submod"
    _init_git_repo(orig)
    _commit_file(orig, "m.txt", "main\n")
    s = SandboxDirs("sb100", project_path=str(orig), base_path=sandbox_base)
    s.create()
    pd = s.project_dir
    sub_hooks_dir = pd / ".git" / "modules" / "mysub" / "hooks"
    sub_hooks_dir.mkdir(parents=True)
    sub_hook = sub_hooks_dir / "pre-commit"
    sub_hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    sample = sub_hooks_dir / "pre-commit.sample"
    sample.write_text("#!/bin/sh\n", encoding="utf-8")
    hooks_dir = pd / ".git" / "hooks"
    hook = hooks_dir / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    s.remove_git_hooks()
    assert not hook.exists()
    assert not sub_hook.exists()
    assert sample.exists()


def test_remove_git_hooks_submodule_no_hooks_dir(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_submod2"
    _init_git_repo(orig)
    _commit_file(orig, "m.txt", "main\n")
    s = SandboxDirs("sb101", project_path=str(orig), base_path=sandbox_base)
    s.create()
    pd = s.project_dir
    sub_hooks_dir = pd / ".git" / "modules" / "mysub" / "hooks"
    sub_hooks_dir.mkdir(parents=True)
    (sub_hooks_dir / "pre-commit").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    shutil.rmtree(sub_hooks_dir)
    hook = pd / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    s.remove_git_hooks()
    assert not hook.exists()


def test_remove_git_hooks_submodule_sample_files_preserved(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_submod3"
    _init_git_repo(orig)
    _commit_file(orig, "m.txt", "main\n")
    s = SandboxDirs("sb102", project_path=str(orig), base_path=sandbox_base)
    s.create()
    pd = s.project_dir
    sub_hooks_dir = pd / ".git" / "modules" / "mysub" / "hooks"
    sub_hooks_dir.mkdir(parents=True)
    sample = sub_hooks_dir / "pre-commit.sample"
    sample.write_text("#!/bin/sh\n", encoding="utf-8")
    sub_hook = sub_hooks_dir / "pre-commit"
    sub_hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    s.remove_git_hooks()
    assert sample.exists()
    assert not sub_hook.exists()


def test_remove_git_hooks_submodule_symlink(tmp_path: Path, sandbox_base: Path) -> None:
    orig = tmp_path / "orig_submod4"
    _init_git_repo(orig)
    _commit_file(orig, "m.txt", "main\n")
    s = SandboxDirs("sb103", project_path=str(orig), base_path=sandbox_base)
    s.create()
    pd = s.project_dir
    sub_hooks_dir = pd / ".git" / "modules" / "mysub" / "hooks"
    sub_hooks_dir.mkdir(parents=True)
    (sub_hooks_dir / "pre-commit").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    shutil.rmtree(sub_hooks_dir)
    outside = tmp_path / "outside4"
    outside.mkdir()
    outside_hook = outside / "evil"
    outside_hook.write_text("#!/bin/sh\necho evil\n", encoding="utf-8")
    sub_hooks_dir.symlink_to(outside)
    s.remove_git_hooks()
    assert not sub_hooks_dir.exists()
    assert outside_hook.exists()


# ===========================================================================
# SandboxDirs – get_containers_meta edge cases
# ===========================================================================


def test_get_containers_meta_empty_sandbox(sandbox_base: Path) -> None:
    s = SandboxDirs("sb42", base_path=sandbox_base)
    s.create()
    # Remove the containers file
    (s._sandbox_dir / "containers.json").unlink()
    assert s.containers_meta == []


def test_get_containers_meta_invalid_type(sandbox_base: Path) -> None:
    s = SandboxDirs("sb42b", base_path=sandbox_base)
    s.create()
    (s._sandbox_dir / "containers.json").write_text(
        '{"key": "value"}', encoding="utf-8"
    )
    with pytest.raises(TypeError, match="Expected list"):
        _ = s.containers_meta


# ===========================================================================
# SandboxDirs – remove symlink attack check
# ===========================================================================


def test_remove_symlink_attack_check(sandbox_base: Path) -> None:
    s = SandboxDirs("sb43", base_path=sandbox_base)
    s.create()
    with (
        patch.object(shutil.rmtree, "avoids_symlink_attacks", False),
        pytest.raises(RuntimeError, match="symlink attacks"),
    ):
        s.remove()


# ===========================================================================
# SandboxDirs – sync_from_original _sync_git_from_original edge cases
# ===========================================================================


def test_sync_from_original_with_existing_remote(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb45",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    repo = git.Repo(str(s.project_dir))
    repo.create_remote("tiz-original-tmp", "/some/path")
    repo.close()
    _commit_file(git_original_project, "new_for_remote.txt", "content")
    s.sync_from_original()
    assert (s.project_dir / "new_for_remote.txt").read_text() == "content"


def test_sync_from_original_no_matching_branch_ref(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb45b",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    repo_s = git.Repo(str(pd))
    repo_s.git.checkout("-b", "extra-branch")
    repo_s.close()
    _commit_file(git_original_project, "some_file.txt", "content")
    s.sync_from_original()
    assert (s.project_dir / "some_file.txt").read_text() == "content"


def test_sync_from_original_merge_then_rebase(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb45c",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(git_original_project, "shared.txt", "original")
    pd = s.project_dir
    _commit_file(pd, "shared.txt", "sandbox version\n")
    with pytest.raises(RuntimeError, match="not merge or rebase"):
        s.sync_from_original()
    assert (s.project_dir / "shared.txt").read_text() == "sandbox version\n"
    s.sync_from_original(force=True)
    assert (s.project_dir / "shared.txt").read_text() == "original"


# ===========================================================================
# SandboxDirs – _copy_project_with_gitignore comment line
# ===========================================================================


def test_copy_project_with_gitignore_comment_lines(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_gitignore_comments"
    orig.mkdir()
    (orig / ".gitignore").write_text(
        "# This is a comment\n*.log\n\n# Another comment\n", encoding="utf-8"
    )
    (orig / "keep.txt").write_text("keep\n", encoding="utf-8")
    (orig / "skip.log").write_text("skip\n", encoding="utf-8")
    s = SandboxDirs("sb46", project_path=str(orig), base_path=sandbox_base)
    s.create()
    assert (s.project_dir / "keep.txt").exists()
    assert not (s.project_dir / "skip.log").exists()


# ===========================================================================
# SandboxDirs – _force_copy_files skips non-files
# ===========================================================================


def test_force_copy_files_skips_directories(tmp_path: Path, sandbox_base: Path) -> None:
    orig = tmp_path / "orig_force_dir"
    orig.mkdir()
    (orig / "subdir").mkdir()
    s = SandboxDirs("sb47", project_path=str(orig), base_path=sandbox_base)
    s.create(force_copy_files=["subdir"])
    assert (s.project_dir / "subdir").exists()
    assert (s.project_dir / "subdir").is_dir()
    assert list((s.project_dir / "subdir").iterdir()) == []


def test_replace_containers_meta(sandbox_base: Path) -> None:
    s = SandboxDirs("sb_replace_meta", base_path=sandbox_base)
    s.create()
    s.replace_containers_meta([{"container_id": "c1"}])
    assert len(s.containers_meta) == 1
    assert s.containers_meta[0]["container_id"] == "c1"
    s.replace_containers_meta([])
    assert s.containers_meta == []


def test_rglob_no_symlinks_skips_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "rglob_test"
    root.mkdir()
    (root / "real.txt").write_text("real\n", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("deep\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("leaked\n", encoding="utf-8")
    sym = root / "symlink_to_outside"
    sym.symlink_to(outside)

    result = SandboxDirs._rglob_no_symlinks(root, "*.txt")
    paths = sorted(str(p.relative_to(root)) for p in result)
    assert "real.txt" in paths
    assert "sub/deep.txt" in paths
    assert "symlink_to_outside" not in paths


def test_rglob_no_symlinks_matching_symlink(tmp_path: Path) -> None:
    root = tmp_path / "rglob_sym_match"
    root.mkdir()
    outside = tmp_path / "outside_sym"
    outside.mkdir()
    (outside / "target.txt").write_text("target\n", encoding="utf-8")
    sym = root / "link_to_outside"
    sym.symlink_to(outside)

    result = SandboxDirs._rglob_no_symlinks(root, "link_to_outside")
    assert len(result) == 1
    assert result[0].is_symlink()


def test_force_copy_files_with_symlink_to_directory(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_sym_dir"
    orig.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
    (orig / "link_to_outside").symlink_to(outside)
    s = SandboxDirs("sb_sym_dir", project_path=str(orig), base_path=sandbox_base)
    s.create(force_copy_files=["link_to_outside"])
    assert (s.project_dir / "link_to_outside").exists()


def test_rglob_no_symlinks_permission_error(tmp_path: Path) -> None:
    root = tmp_path / "rglob_perm"
    root.mkdir()
    no_perm = root / "no_access"
    no_perm.mkdir()
    no_perm.chmod(0o000)
    (root / "real.txt").write_text("real\n", encoding="utf-8")
    try:
        result = SandboxDirs._rglob_no_symlinks(root, "*.txt")
        assert any("real.txt" in str(p) for p in result)
    finally:
        no_perm.chmod(0o755)


def test_rglob_no_symlinks_relative_path_pattern(tmp_path: Path) -> None:
    """_rglob_no_symlinks matches pattern against relative path, not just name."""
    root = tmp_path / "rglob_rel"
    root.mkdir()
    sub = root / "src"
    sub.mkdir()
    (sub / "foo.py").write_text("x\n", encoding="utf-8")
    (sub / "bar.py").write_text("x\n", encoding="utf-8")
    (root / "readme.txt").write_text("x\n", encoding="utf-8")

    # Pattern "**/*.py" should match files in subdirectories
    result = SandboxDirs._rglob_no_symlinks(root, "**/*.py")
    paths = sorted(str(p.relative_to(root)) for p in result)
    assert paths == ["src/bar.py", "src/foo.py"]

    # Pattern "*.txt" should still match at root level
    result_txt = SandboxDirs._rglob_no_symlinks(root, "*.txt")
    paths_txt = sorted(str(p.relative_to(root)) for p in result_txt)
    assert paths_txt == ["readme.txt"]


def test_rglob_no_symlinks_double_star_prefix(tmp_path: Path) -> None:
    """Pattern like src/**/deep/*.py matches nested relative paths."""
    root = tmp_path / "rglob_dstar"
    root.mkdir()
    nested = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("x\n", encoding="utf-8")
    (root / "a" / "other.py").write_text("x\n", encoding="utf-8")

    result = SandboxDirs._rglob_no_symlinks(root, "a/**/deep.py")
    paths = sorted(str(p.relative_to(root)) for p in result)
    assert paths == ["a/b/c/deep.py"]


def test_rglob_no_symlinks_symlink_matches_rel_pattern(tmp_path: Path) -> None:
    """Symlinks matched against relative path pattern."""
    root = tmp_path / "rglob_sym_rel"
    root.mkdir()
    sub = root / "sub"
    sub.mkdir()
    (sub / "real.txt").write_text("x\n", encoding="utf-8")
    outside = tmp_path / "other"
    outside.mkdir()
    sym = sub / "link"
    sym.symlink_to(outside)

    result = SandboxDirs._rglob_no_symlinks(root, "sub/link")
    assert len(result) == 1
    assert result[0].is_symlink()

    # Symlink to a directory should not be followed
    result2 = SandboxDirs._rglob_no_symlinks(root, "sub/*.txt")
    paths2 = sorted(str(p.relative_to(root)) for p in result2)
    assert paths2 == ["sub/real.txt"]


def test_rglob_no_symlinks_path_with_dir_pattern(tmp_path: Path) -> None:
    """Pattern like src/*.py matches .py files under src/ (incl. subdirs)."""
    root = tmp_path / "rglob_dirpat"
    root.mkdir()
    src = root / "src"
    src.mkdir()
    (src / "foo.py").write_text("x\n", encoding="utf-8")
    nested = src / "nested"
    nested.mkdir()
    (nested / "bar.py").write_text("x\n", encoding="utf-8")

    # * in fnmatch matches across '/' so both files are found
    result = SandboxDirs._rglob_no_symlinks(root, "src/*.py")
    paths = sorted(str(p.relative_to(root)) for p in result)
    assert paths == ["src/foo.py", "src/nested/bar.py"]


def test_rglob_no_symlinks_no_match(tmp_path: Path) -> None:
    """Pattern that matches no files returns empty list."""
    root = tmp_path / "rglob_nomatch"
    root.mkdir()
    (root / "foo.txt").write_text("x\n", encoding="utf-8")
    result = SandboxDirs._rglob_no_symlinks(root, "*.py")
    assert result == []


def test_add_container_meta_filters_unknown_keys(sandbox_base: Path) -> None:
    s = SandboxDirs("sb_filter_keys", base_path=sandbox_base)
    s.create()
    s.add_container_meta(
        {
            "container_id": "c1",
            "clearly_unknown_key": "should_be_filtered",
            "another_unknown": "also_should_be_filtered",
        }
    )
    meta = s.containers_meta
    assert len(meta) == 1
    assert meta[0]["container_id"] == "c1"
    assert "clearly_unknown_key" not in meta[0]
    assert "another_unknown" not in meta[0]


# ===========================================================================
# SandboxDirs – sync_to_original _sync_patch
# ===========================================================================


def test_sync_to_original_non_git_creates_patch(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_patch"
    orig.mkdir()
    (orig / "a.txt").write_text("original\n", encoding="utf-8")
    s = SandboxDirs("sb48", project_path=str(orig), base_path=sandbox_base)
    s.create()
    (s.project_dir / "a.txt").write_text("modified\n", encoding="utf-8")
    (s.project_dir / "new.txt").write_text("new file\n", encoding="utf-8")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is True
    patches_dir = orig / TIZ_PATCHES_DIR
    assert patches_dir.exists()
    patch_files = list(patches_dir.glob("*.patch"))
    assert len(patch_files) == 1
    patch_name = patch_files[0].name
    assert patch_name.endswith(".patch")
    assert patch_name.startswith("tiz_sync_")
    patch_contents = patch_files[0].read_text(encoding="utf-8")
    assert patch_contents
    assert "--- " in patch_contents
    assert "+++ " in patch_contents
    assert "@@ " in patch_contents
    assert "-original" in patch_contents
    assert "+modified" in patch_contents
    assert "+new file" in patch_contents
    # Verify the patch can be applied cleanly
    result = subprocess.run(
        ["patch", "-p1", "--dry-run", "-i", str(patch_files[0])],
        cwd=orig,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"patch dry-run failed: {result.stderr}"


def test_sync_to_original_non_git_no_changes(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_patch_unchanged"
    orig.mkdir()
    (orig / "a.txt").write_text("same\n", encoding="utf-8")
    s = SandboxDirs("sb48b", project_path=str(orig), base_path=sandbox_base)
    s.create()
    assert s.may_need_to_sync_to_original() is False
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    patches_dir = orig / TIZ_PATCHES_DIR
    patch_files = list(patches_dir.glob("*.patch"))
    assert len(patch_files) == 0


def test_sync_to_original_non_git_no_changes_gitignore(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_patch_unchanged"
    orig.mkdir()
    (orig / "a.txt").write_text("same\n", encoding="utf-8")
    (orig / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (orig / "testn.log").write_text("test")
    s = SandboxDirs("sb48b", project_path=str(orig), base_path=sandbox_base)
    s.create()
    (s.project_dir / "test.log").write_text("test")
    assert s.may_need_to_sync_to_original() is False
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    patches_dir = orig / TIZ_PATCHES_DIR
    patch_files = list(patches_dir.glob("*.patch"))
    assert len(patch_files) == 0


# ===========================================================================
# SandboxDirs – _sync_git_pull_from_sandbox "nothing to commit" exception
# ===========================================================================


def test_sync_git_pull_nothing_to_commit_exception(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb50",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    assert s.may_need_to_sync_to_original() is False
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    pd = s.project_dir
    assert (git_original_project / "README.md").read_text() == (
        pd / "README.md"
    ).read_text()


def test_sync_git_pull_existing_temp_remote(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb51",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    repo = git.Repo(str(git_original_project))
    repo.create_remote("tiz-sandbox-tmp", "/some/path")
    repo.close()
    _commit_file(s.project_dir, "feature2.txt", "feature2")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "feature2.txt").read_text() == "feature2"


def test_sync_git_pull_no_commit(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb51b",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (s.project_dir / "feature2.txt").write_text("feature2")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "feature2.txt").read_text() == "feature2"
    repo = git.Repo(str(git_original_project))
    assert not repo.is_dirty(untracked_files=True)
    repo.close()
    repo = git.Repo(str(s.project_dir))
    assert not repo.is_dirty(untracked_files=True)
    repo.close()


def test_sync_git_pull_no_remote_refs(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb51c",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    assert s.may_need_to_sync_to_original() is False
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False


def test_sync_from_original_dirty_original_non_git(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_dirty_non_git"
    orig.mkdir()
    (orig / "a.txt").write_text("a\n", encoding="utf-8")
    s = SandboxDirs("sb53", project_path=str(orig), base_path=sandbox_base)
    s.create()
    time.sleep(0.05)
    (orig / "a.txt").write_text("b\n", encoding="utf-8")
    assert s.may_need_to_sync_to_original() is True
    s.sync_from_original()
    assert (s.project_dir / "a.txt").read_text(encoding="utf-8") == "b\n"


def test_sync_git_from_original_preexisting_remote(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb55",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    repo = git.Repo(str(s.project_dir))
    repo.create_remote("tiz-original-tmp", "http://example.com")
    repo.close()
    _commit_file(git_original_project, "file_after_remote.txt", "content")
    s.sync_from_original()
    assert (s.project_dir / "file_after_remote.txt").read_text() == "content"


def test_sync_to_original_force_reset(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb57",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(git_original_project, "orig_extra.txt", "extra")
    _commit_file(s.project_dir, "sand_only.txt", "sand")
    assert s.may_need_to_sync_to_original() is True
    with pytest.raises(RuntimeError, match="not fast forward"):
        s.sync_to_original()
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original(force=True)
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "sand_only.txt").read_text() == "sand"


def test_sync_to_original_and_from(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb57b",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(git_original_project, "orig_extra.txt", "extra")
    _commit_file(pd, "sand_only.txt", "sand")
    assert s.may_need_to_sync_to_original() is True
    with pytest.raises(RuntimeError, match="not fast forward"):
        s.sync_to_original()
    s.sync_from_original()
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "sand_only.txt").read_text() == "sand"
    assert (git_original_project / "orig_extra.txt").read_text() == "extra"
    assert (pd / "sand_only.txt").read_text() == "sand"
    assert (pd / "orig_extra.txt").read_text() == "extra"


def test_get_containers_meta_no_file_before_create(sandbox_base: Path) -> None:
    s = SandboxDirs("sb58", base_path=sandbox_base)
    assert s.containers_meta == []


def test_sync_from_original_non_git_path(tmp_path: Path, sandbox_base: Path) -> None:
    orig = tmp_path / "orig_sync_non_git"
    orig.mkdir()
    (orig / "file.txt").write_text("content\n", encoding="utf-8")
    s = SandboxDirs("sb59", project_path=str(orig), base_path=sandbox_base)
    s.create()
    time.sleep(0.05)
    (orig / "file.txt").write_text("updated\n", encoding="utf-8")
    (orig / "new_file.txt").write_text("new\n", encoding="utf-8")
    s.sync_from_original()
    assert (s.project_dir / "file.txt").read_text() == "updated\n"
    assert (s.project_dir / "new_file.txt").read_text() == "new\n"


def test_sync_patch_diff_failure(
    tmp_path: Path,
    sandbox_base: Path,
) -> None:
    orig = tmp_path / "orig_diff_fail"
    orig.mkdir()
    (orig / "a.txt").write_text("content\n", encoding="utf-8")
    s = SandboxDirs("sb60", project_path=str(orig), base_path=sandbox_base)
    s.create()
    (s.project_dir / "a.txt").write_text("modified\n", encoding="utf-8")
    failed_result = subprocess.CompletedProcess(
        args=[], returncode=2, stdout="", stderr="diff: error"
    )
    with (
        patch("subprocess.run", return_value=failed_result),
        pytest.raises(RuntimeError, match="diff command failed"),
    ):
        s.sync_to_original()
    patches_dir = orig / TIZ_PATCHES_DIR
    assert not patches_dir.exists()


def test_sync_from_original_rebase_succeeds(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb66",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd, "sandbox_add.txt", "sandbox\n")
    s.sync_from_original()
    assert (pd / "sandbox_add.txt").read_text() == "sandbox\n"


def test_sync_from_original_rebase_fails(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb67",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd, "README.md", "sandbox version\n")
    _commit_file(git_original_project, "README.md", "original version\n")
    with pytest.raises(RuntimeError, match="not merge or rebase"):
        s.sync_from_original()
    s.sync_from_original(force=True)
    assert (s.project_dir / "README.md").read_text() == "original version\n"


def test_sync_git_from_original_no_cleanup_remote(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb76",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(git_original_project, "cleanup_test.txt", "cleanup content")
    s.sync_from_original()
    assert (s.project_dir / "cleanup_test.txt").read_text() == "cleanup content"


def test_empty_project_path_raises(
    sandbox_base: Path, fake_original_project: Path
) -> None:
    s = SandboxDirs(
        "sb80", project_path=str(fake_original_project), base_path=sandbox_base
    )
    s.create()
    s._meta_project_path_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty project path"):
        SandboxDirs("sb80", base_path=sandbox_base)


def test_get_containers_meta_private_empty(sandbox_base: Path) -> None:
    s = SandboxDirs("sb84", base_path=sandbox_base)
    assert s._containers_meta == []


def test_sync_from_original_git_commit_error_raises(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_commit_err",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    (pd / "sandbox_only.txt").write_text("added\n", encoding="utf-8")
    error = git.GitCommandError("commit", 1, "", "some other error")

    orig_repo = git.Repo(str(git_original_project))
    sandbox_repo = git.Repo(str(pd))
    try:

        class MockGit:
            def add(self, *_a, **_kw):
                pass

            def commit(self, *_a, **_kw):
                raise error

            def checkout(self, *_a, **_kw):
                pass

            def config(self, *_a, **_kw):
                raise git.GitCommandError("config", 1)

            def merge(self, *_a, **_kw):
                pass

            def rebase(self, *_a, **_kw):
                pass

            def submodule(self, *_a, **_kw):
                pass

        class MockIndex:
            @staticmethod
            def diff(*_a, **_kw):
                return ["something"]

        class MockSandboxRepo:
            def is_dirty(self, **kwargs):  # noqa: ARG002
                return True

            @property
            def index(self):
                return MockIndex()

            @property
            def git(self):
                return MockGit()

            @property
            def active_branch(self):
                return sandbox_repo.active_branch

            @property
            def heads(self):
                return sandbox_repo.heads

            @property
            def remotes(self):
                return []

            @property
            def submodules(self):
                return []

            def create_remote(self, *_a, **_kw):
                return None

            def delete_remote(self, *_a, **_kw):
                pass

            def remote(self, *_a, **_kw):
                return None

            def close(self):
                pass

        mock_sandbox = MockSandboxRepo()
        orig_ab = sandbox_repo.active_branch

        def repo_factory(path, *_a, **_kw):
            if str(Path(path).resolve()) == str(git_original_project.resolve()):
                m = type("MO", (), {})()

                class FakeAB:
                    name = orig_ab.name

                m.active_branch = FakeAB()
                m.is_dirty = lambda **_: False
                m.close = lambda: None
                return m
            return mock_sandbox

        with (
            patch("git.Repo", side_effect=repo_factory),
            pytest.raises(git.GitCommandError, match="some other error"),
        ):
            s.sync_from_original()
    finally:
        sandbox_repo.close()
        orig_repo.close()


def test_sync_to_original_git_commit_error_raises(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_git_commit_err2",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    (pd / "sandbox_only2.txt").write_text("added\n", encoding="utf-8")
    error = git.GitCommandError("commit", 1, "", "some other error message")
    sandbox_repo = git.Repo(str(pd))
    orig_repo = git.Repo(str(git_original_project))
    try:

        class MockGit2:
            def add(self, *_a, **_kw):
                pass

            def commit(self, *_a, **_kw):
                raise error

            def config(self, *_a, **_kw):
                raise git.GitCommandError("config", 1)

            def submodule(self, *_a, **_kw):
                pass

        class MockIndex2:
            @staticmethod
            def diff(*_a, **_kw):
                return ["something"]

        class MockSandboxRepo2:
            def is_dirty(self, **kwargs):  # noqa: ARG002
                return True

            @property
            def index(self):
                return MockIndex2()

            @property
            def active_branch(self):
                return sandbox_repo.active_branch

            @property
            def git(self):
                return MockGit2()

            @property
            def submodules(self):
                return []

            def close(self):
                pass

        mock_sandbox = MockSandboxRepo2()
        call_count = {"n": 0}

        def repo_factory2(path, *_a, **_kw):
            call_count["n"] += 1
            if "git_original" in str(path) or str(
                git_original_project.resolve()
            ) in str(Path(path).resolve()):
                m = type("MO2", (), {})()
                m.is_dirty = lambda **_: False
                m.remotes = []
                m.close = lambda: None
                return m
            return mock_sandbox

        with (
            patch("git.Repo", side_effect=repo_factory2),
            pytest.raises(git.GitCommandError, match="some other error message"),
        ):
            s.sync_to_original()
    finally:
        sandbox_repo.close()
        orig_repo.close()


def test_sync_to_original_original_detached_head(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_detached_orig",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    subprocess.run(
        ["git", "checkout", "--detach"],
        cwd=git_original_project,
        check=True,
        capture_output=True,
    )
    with pytest.raises(RuntimeError, match="detached HEAD"):
        s.sync_to_original()


def test_sync_to_original_no_matching_branch_ref(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_no_match_sync_to",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    repo_s = git.Repo(str(s.project_dir))
    repo_s.git.checkout("-b", "extra-branch-sync-to")
    repo_s.close()
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is True
    repo_s = git.Repo(str(git_original_project))
    assert "extra-branch-sync-to" not in {h.name for h in repo_s.heads}
    s.sync_to_original(all_branches=True)
    assert s.may_need_to_sync_to_original() is False
    assert "extra-branch-sync-to" in {h.name for h in repo_s.heads}
    repo_s.close()


def test_sync_from_original_git_commit_nothing_to_commit(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """When a repo with untracked files is created, the commit may fail with
    'nothing to commit' if the repo was already clean.  This should not
    raise."""
    s = SandboxDirs(
        "sb_no_commit_err",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    no_err = git.GitCommandError(
        "commit", 1, "nothing to commit, working tree clean", ""
    )
    orig_repo = git.Repo(str(git_original_project))
    sandbox_repo = git.Repo(str(s.project_dir))
    try:

        class MockRemote:
            def fetch(self, *_a, **_kw):
                pass

            @property
            def refs(self):
                return []

        class MockGit:
            def add(self, *_a, **_kw):
                pass

            def commit(self, *_a, **_kw):
                raise no_err

            def checkout(self, *_a, **_kw):
                pass

            def config(self, *_a, **_kw):
                raise git.GitCommandError("config", 1)

            def merge(self, *_a, **_kw):
                pass

            def rebase(self, *_a, **_kw):
                pass

            def submodule(self, *_a, **_kw):
                pass

        mock_sandbox = type("MS", (), {})()
        mock_sandbox.is_dirty = lambda **_: True
        active_branch_name = sandbox_repo.active_branch.name

        class FakeAB:
            name = active_branch_name

        class FakeIndex:
            @staticmethod
            def diff(*_a, **_kw):
                return []

        mock_sandbox.index = FakeIndex()
        mock_sandbox.active_branch = FakeAB()
        mock_sandbox.git = MockGit()
        mock_sandbox.heads = []
        mock_sandbox.submodules = []
        mock_sandbox.remotes = []
        mock_sandbox.create_remote = lambda *_a, **_kw: MockRemote()
        mock_sandbox.delete_remote = lambda *_a, **_kw: None
        mock_sandbox.remote = lambda *_a, **_kw: MockRemote()
        mock_sandbox.close = sandbox_repo.close

        def repo_factory(path, *_a, **_kw):
            if str(Path(path).resolve()) == str(git_original_project.resolve()):
                m = type("MO", (), {})()
                m.active_branch = sandbox_repo.active_branch
                m.is_dirty = lambda **_: False
                m.close = lambda: None
                return m
            return mock_sandbox

        with patch("git.Repo", side_effect=repo_factory):
            s.sync_from_original()
    finally:
        sandbox_repo.close()
        orig_repo.close()


def test_sync_to_original_git_commit_nothing_to_commit(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Same pattern but for sync_to_original."""
    s = SandboxDirs(
        "sb_no_commit_err2",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    no_err = git.GitCommandError(
        "commit", 1, "nothing to commit, working tree clean", ""
    )
    sandbox_repo = git.Repo(str(s.project_dir))
    orig_repo = git.Repo(str(git_original_project))
    try:

        class MockRemote2:
            def fetch(self, *_a, **_kw):
                pass

            @property
            def refs(self):
                return []

        class MockGit2:
            def add(self, *_a, **_kw):
                pass

            def commit(self, *_a, **_kw):
                raise no_err

            def checkout(self, *_a, **_kw):
                pass

            def config(self, *_a, **_kw):
                raise git.GitCommandError("config", 1)

            def submodule(self, *_a, **_kw):
                pass

        mock_sandbox = type("MS2", (), {})()
        mock_sandbox.is_dirty = lambda **_: True
        mock_sandbox.active_branch = sandbox_repo.active_branch
        mock_sandbox.git = MockGit2()

        class FakeIndex2:
            @staticmethod
            def diff(*_a, **_kw):
                return []

        mock_sandbox.index = FakeIndex2()
        mock_sandbox.submodules = []
        mock_sandbox.close = lambda: None

        def repo_factory2(path, *_a, **_kw):
            if str(Path(path).resolve()) == str(git_original_project.resolve()):
                m = type("MO2", (), {})()
                m.is_dirty = lambda **_: False
                m.remotes = []
                m.create_remote = lambda *_a, **_kw: MockRemote2()
                m.delete_remote = lambda *_a, **_kw: None
                m.git = MockGit2()
                m.remote = lambda *_a, **_kw: MockRemote2()
                try:
                    m.active_branch = sandbox_repo.active_branch
                except TypeError:
                    raise RuntimeError("detached") from None
                m.close = lambda: None
                return m
            return mock_sandbox

        with patch("git.Repo", side_effect=repo_factory2):
            s.sync_to_original()
    finally:
        sandbox_repo.close()
        orig_repo.close()


# ===========================================================================
# SandboxDirs – validate_git_project_dir
# ===========================================================================


def test_validate_git_project_dir_no_project_dir(sandbox_base: Path) -> None:
    s = SandboxDirs("sb_validate_no_proj", base_path=sandbox_base)
    s.create()
    assert not s.project_dir.exists()
    s.validate_git_project_dir()


def test_validate_git_project_dir_no_git_dir(
    fake_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_no_git",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    s.create()
    s.validate_git_project_dir()


def test_validate_git_project_dir_git_symlink_attack(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_git_sym",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    git_dir = pd / ".git"
    shutil.rmtree(git_dir)
    outside = sandbox_base.parent / "outside_git"
    outside.mkdir()
    git_dir.symlink_to(outside)
    with pytest.raises(ValueError, match="is a symlink"):
        s.validate_git_project_dir()


def test_validate_git_project_dir_git_internal_symlink(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_git_int_sym",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    git_dir = s.project_dir / ".git"
    evil = git_dir / "evil_symlink"
    evil.symlink_to(sandbox_base.parent)
    with pytest.raises(ValueError, match="Symlink found inside .git"):
        s.validate_git_project_dir()


def test_validate_git_project_dir_valid(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_valid",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    s.validate_git_project_dir()


def test_validate_git_project_dir_invalid_repo(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_invalid",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    git_dir = pd / ".git"
    shutil.rmtree(git_dir / "objects")
    with pytest.raises(ValueError, match="not a valid git repository"):
        s.validate_git_project_dir()


def test_validate_git_project_dir_protocol_file_allow_unset(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_proto_unset",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    s.validate_git_project_dir()


def test_validate_git_project_dir_protocol_file_allow_never(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_proto_never",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    repo = git.Repo(s.project_dir)
    try:
        repo.git.config("protocol.file.allow", "never")
    finally:
        repo.close()
    s.validate_git_project_dir()


def test_validate_git_project_dir_protocol_file_allow_always(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_proto_always",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    repo = git.Repo(s.project_dir)
    try:
        repo.git.config("protocol.file.allow", "always")
    finally:
        repo.close()
    with pytest.raises(ValueError, match="protocol.file.allow"):
        s.validate_git_project_dir()


def test_validate_git_project_dir_protocol_file_allow_if_asked(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_validate_proto_ifasked",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    repo = git.Repo(s.project_dir)
    try:
        repo.git.config("protocol.file.allow", "ifAsked")
    finally:
        repo.close()
    with pytest.raises(ValueError, match="protocol.file.allow"):
        s.validate_git_project_dir()


def test_validate_git_project_dir_protocol_file_allow_submodule(
    git_original_project: Path, sandbox_base: Path
) -> None:
    sub_path = git_original_project.parent / "sub_repo"
    _init_git_repo(sub_path)
    _commit_file(sub_path, "sub_file.txt", "sub content\n")

    s = SandboxDirs("sb_validate_proto_sub", base_path=sandbox_base)
    s.create()
    pd = s.project_dir
    pd.mkdir(parents=True, exist_ok=True)
    _init_git_repo(pd)
    _commit_file(pd, "init.txt", "init\n")
    subprocess.run(
        [
            "git",
            "-C",
            str(pd),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(sub_path),
            "mysub",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(pd), "commit", "-m", "add submodule"],
        check=True,
        capture_output=True,
    )

    sub_dst = pd / "mysub"
    sub_repo = git.Repo(sub_dst)
    try:
        sub_repo.git.config("protocol.file.allow", "always")
    finally:
        sub_repo.close()

    with pytest.raises(ValueError, match="protocol.file.allow"):
        s.validate_git_project_dir()


def test_validate_git_project_dir_protocol_file_allow_invalid_repo(
    tmp_path: Path,
) -> None:
    """_check_git_protocol_file_allow should handle InvalidGitRepositoryError."""
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()
    SandboxDirs._check_git_protocol_file_allow(non_repo)


def test_validate_git_project_dir_protocol_file_allow_non_existent_path(
    tmp_path: Path,
) -> None:
    """_check_git_protocol_file_allow should handle NoSuchPathError."""
    non_existent = tmp_path / "does_not_exist"
    SandboxDirs._check_git_protocol_file_allow(non_existent)


def test_validate_git_project_dir_submodule_path_not_exists(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Submodule path that doesn't exist should be skipped."""
    sub_path = git_original_project.parent / "sub_repo_inex"
    _init_git_repo(sub_path)
    _commit_file(sub_path, "sub_file.txt", "sub content\n")

    s = SandboxDirs("sb_validate_proto_sub_nex", base_path=sandbox_base)
    s.create()
    pd = s.project_dir
    pd.mkdir(parents=True, exist_ok=True)
    _init_git_repo(pd)
    _commit_file(pd, "init.txt", "init\n")
    subprocess.run(
        [
            "git",
            "-C",
            str(pd),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(sub_path),
            "mysub_nex",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(pd), "commit", "-m", "add submodule nex"],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(pd / "mysub_nex")

    s.validate_git_project_dir()


def test_validate_git_project_dir_protocol_file_allow_submodule_unset(
    git_original_project: Path, sandbox_base: Path
) -> None:
    sub_path = git_original_project.parent / "sub_repo_ok"
    _init_git_repo(sub_path)
    _commit_file(sub_path, "sub_file.txt", "sub content\n")

    s = SandboxDirs("sb_validate_proto_sub_ok", base_path=sandbox_base)
    s.create()
    pd = s.project_dir
    pd.mkdir(parents=True, exist_ok=True)
    _init_git_repo(pd)
    _commit_file(pd, "init.txt", "init\n")
    subprocess.run(
        [
            "git",
            "-C",
            str(pd),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(sub_path),
            "mysub_ok",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(pd), "commit", "-m", "add submodule ok"],
        check=True,
        capture_output=True,
    )

    s.validate_git_project_dir()


# ===========================================================================
# SandboxDirs – sync_from_original with multiple branches
# ===========================================================================


def repo_checkout_create(path: Path, name: str):
    repo = git.Repo(str(path))
    try:
        repo.git.checkout("-b", name)
    finally:
        repo.close()


def repo_checkout(path: Path, name: str):
    repo = git.Repo(str(path))
    try:
        repo.git.checkout(name)
    finally:
        repo.close()


def repo_has_branch(path: Path, name: str):
    repo = git.Repo(str(path))
    try:
        return name in {h.name for h in repo.heads}
    finally:
        repo.close()


def test_sync_from_original_multiple_branches(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Original has main and feature branches with new commits; sync
    updates both branches in the sandbox."""
    repo_checkout_create(git_original_project, "feature")
    _commit_file(git_original_project, "feature_base.txt", "feature base\n")
    repo_checkout(git_original_project, "main")

    s = SandboxDirs(
        "sb_multi_from",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    repo_checkout(pd, "feature")
    _commit_file(pd, "feature_base.txt", "feature base2\n")
    repo_checkout(pd, "main")

    _commit_file(git_original_project, "main_update.txt", "main update\n")
    repo_checkout(git_original_project, "feature")
    _commit_file(git_original_project, "feature_file.txt", "feature content\n")
    repo_checkout(git_original_project, "main")

    s.sync_from_original()

    assert (pd / "main_update.txt").read_text() == "main update\n"
    repo_checkout(pd, "feature")
    assert (pd / "feature_file.txt").read_text() == "feature content\n"
    assert (pd / "feature_base.txt").read_text() == "feature base2\n"
    repo_checkout(git_original_project, "feature")
    assert (git_original_project / "feature_base.txt").read_text() == "feature base\n"


def test_sync_from_original_multiple_branches_sandbox_has_local_commits(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Original and sandbox both have separate branches with local commits;
    sync merges without force."""
    s = SandboxDirs(
        "sb_multi_from2",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    _commit_file(git_original_project, "main_update.txt", "main update\n")

    _commit_file(pd, "sandbox_main.txt", "sandbox\n")

    repo_checkout_create(pd, "feature")
    _commit_file(pd, "sandbox_feature.txt", "sandbox feature\n")
    repo_checkout(pd, "main")

    repo_checkout_create(git_original_project, "feature")
    _commit_file(git_original_project, "orig_feature.txt", "orig feature\n")
    repo_checkout(git_original_project, "main")

    s.sync_from_original()

    assert (git_original_project / "main_update.txt").read_text() == "main update\n"
    assert (pd / "main_update.txt").read_text() == "main update\n"
    assert (pd / "sandbox_main.txt").read_text() == "sandbox\n"
    repo_checkout(git_original_project, "feature")
    repo_checkout(pd, "feature")
    assert (git_original_project / "orig_feature.txt").read_text() == "orig feature\n"
    assert (pd / "sandbox_feature.txt").read_text() == "sandbox feature\n"
    assert (pd / "orig_feature.txt").read_text() == "orig feature\n"


def test_sync_from_original_multiple_branches_force(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Force sync overwrites local branch commits in sandbox with original."""
    s = SandboxDirs(
        "sb_multi_from3",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    _commit_file(pd, "sandbox_local.txt", "local sandbox\n")
    repo_checkout_create(pd, "feature")
    _commit_file(pd, "local_feature.txt", "local feature\n")
    repo_checkout(pd, "main")

    _commit_file(git_original_project, "orig_update.txt", "orig\n")
    repo_checkout_create(git_original_project, "feature")
    _commit_file(git_original_project, "orig_feature.txt", "orig feature\n")
    repo_checkout(git_original_project, "main")

    s.sync_from_original(force=True)

    assert (pd / "orig_update.txt").read_text() == "orig\n"
    assert (git_original_project / "orig_update.txt").read_text() == "orig\n"
    assert not (pd / "sandbox_local.txt").exists()
    assert not (git_original_project / "sandbox_local.txt").exists()

    repo_checkout(pd, "feature")
    repo_checkout(git_original_project, "feature")
    assert (pd / "orig_feature.txt").read_text() == "orig feature\n"
    assert (git_original_project / "orig_feature.txt").read_text() == "orig feature\n"
    assert not (pd / "local_feature.txt").exists()
    assert not (git_original_project / "local_feature.txt").exists()


def test_sync_from_original_extra_branch_in_sandbox_skipped(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Sandbox has a branch not present in original; sync leaves it untouched."""
    s = SandboxDirs(
        "sb_extra_br",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    repo_checkout_create(pd, "sandbox-only")
    _commit_file(pd, "sandbox_only.txt", "sandbox only\n")
    repo_checkout(pd, "main")

    _commit_file(git_original_project, "orig_new.txt", "orig new\n")

    s.sync_from_original()

    assert (pd / "orig_new.txt").read_text() == "orig new\n"
    repo_checkout(pd, "sandbox-only")
    assert (pd / "sandbox_only.txt").read_text() == "sandbox only\n"


def test_sync_from_original_extra_branch_in_sandbox_skipped_force(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Sandbox has a branch not present in original; sync leaves it untouched."""
    s = SandboxDirs(
        "sb_extra_br2",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    repo_checkout_create(pd, "sandbox-only")
    _commit_file(pd, "sandbox_only.txt", "sandbox only\n")
    repo_checkout(pd, "main")

    _commit_file(git_original_project, "orig_new.txt", "orig new\n")

    s.sync_from_original(force=True)

    assert (pd / "orig_new.txt").read_text() == "orig new\n"
    repo_checkout(pd, "sandbox-only")
    assert (pd / "sandbox_only.txt").read_text() == "sandbox only\n"


# ===========================================================================
# SandboxDirs – sync_to_original with multiple branches
# ===========================================================================


def test_sync_to_original_multiple_branches_same_active(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Both original and sandbox have main and feature branches; sandbox
    commits on main are synced to original."""
    s = SandboxDirs(
        "sb_multi_to",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    repo_checkout_create(pd, "feature")
    _commit_file(pd, "feature_file.txt", "feature content\n")
    repo_checkout(pd, "main")
    _commit_file(pd, "sandbox_main.txt", "sandbox main\n")

    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False

    assert (git_original_project / "sandbox_main.txt").read_text() == "sandbox main\n"
    assert not (git_original_project / "feature_file.txt").exists()
    assert not repo_has_branch(git_original_project, "feature")


def test_sync_to_original_multiple_branches_force(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Both original and sandbox have main and feature; original has extra
    commits on main; force sync overwrites main in original."""
    s = SandboxDirs(
        "sb_multi_to2",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    _commit_file(git_original_project, "orig_extra.txt", "orig extra\n")

    _commit_file(pd, "sandbox_file.txt", "sandbox file\n")

    repo_checkout_create(pd, "feature")
    _commit_file(pd, "feature_sandbox.txt", "feature sb\n")
    repo_checkout(pd, "main")

    assert s.may_need_to_sync_to_original() is True
    with pytest.raises(RuntimeError, match="not fast forward"):
        s.sync_to_original()
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original(force=True)
    assert s.may_need_to_sync_to_original() is False

    assert (git_original_project / "sandbox_file.txt").read_text() == "sandbox file\n"
    assert not (git_original_project / "orig_extra.txt").exists()
    assert not (git_original_project / "feature_sandbox.txt").exists()
    assert not repo_has_branch(git_original_project, "feature")


def test_sync_to_original_multiple_branches_clean_sandbox_syncs(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Sandbox and original both have main branches, but sandbox has
    no new commits to push; sync is a no-op."""
    s = SandboxDirs(
        "sb_multi_to3",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    repo_checkout_create(pd, "feature")
    repo_checkout(pd, "main")

    assert s.may_need_to_sync_to_original() is False
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    assert not repo_has_branch(git_original_project, "feature")
    s.sync_to_original(all_branches=True)
    assert s.may_need_to_sync_to_original() is False
    assert repo_has_branch(git_original_project, "feature")


def test_sync_to_original_feature_branch_active_in_original(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Original checks out feature branch before sync; sandbox changes land
    on feature branch in original."""
    s = SandboxDirs(
        "sb_multi_to4",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    repo_checkout_create(git_original_project, "feature")

    repo_checkout_create(pd, "feature")
    _commit_file(pd, "feature_sync.txt", "feature sync content\n")
    repo_checkout(pd, "main")
    _commit_file(pd, "main_file.txt", "main sync content\n")
    repo_checkout(pd, "feature")

    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False

    assert (
        git_original_project / "feature_sync.txt"
    ).read_text() == "feature sync content\n"
    assert not (git_original_project / "main_file.txt").exists()
    repo_checkout(git_original_project, "main")
    assert not (git_original_project / "main_file.txt").exists()


def test_sync_to_original_and_from_roundtrip_multiple_branches(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Full round-trip: sync_to_original on main, sync_from_original picks up
    new commits on both branches."""
    s = SandboxDirs(
        "sb_roundtrip_mb",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    _commit_file(pd, "sandbox_main.txt", "sandbox main\n")
    repo_checkout_create(pd, "feature")
    _commit_file(pd, "sandbox_feature.txt", "sandbox feature\n")
    repo_checkout(pd, "main")

    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "sandbox_main.txt").read_text() == "sandbox main\n"
    assert not repo_has_branch(git_original_project, "feature")
    assert not (pd / "main.txt").exists()
    _commit_file(git_original_project, "main.txt", "orig main\n")

    s.sync_from_original()

    assert (pd / "main.txt").read_text() == "orig main\n"
    assert (pd / "sandbox_main.txt").read_text() == "sandbox main\n"
    repo_checkout(pd, "feature")
    assert (pd / "sandbox_feature.txt").read_text() == "sandbox feature\n"


def test_sync_from_original_multiple_branches_no_matching_ref_for_sandbox_branch(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Sandbox has a branch that doesn't exist in original; sync updates
    matching branches and leaves the extra branch alone."""
    s = SandboxDirs(
        "sb_multi_from4",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    repo_checkout_create(pd, "sandbox-only-branch")
    _commit_file(pd, "sandbox_only.txt", "sandbox only\n")
    repo_checkout(pd, "main")

    _commit_file(git_original_project, "main_update.txt", "main update\n")

    s.sync_from_original()

    assert not repo_has_branch(git_original_project, "sandbox-only-branch")
    assert (pd / "main_update.txt").read_text() == "main update\n"
    repo_checkout(pd, "sandbox-only-branch")
    assert (pd / "sandbox_only.txt").read_text() == "sandbox only\n"


def test_sync_to_original_multiple_branches_no_matching_branch_in_original(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Sandbox has main and feature; original only has main. Sync to
    original only pushes main."""
    s = SandboxDirs(
        "sb_multi_to5",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir

    repo_checkout_create(pd, "sandbox-feature")
    _commit_file(pd, "feature_sb.txt", "feature sb\n")
    repo_checkout(pd, "main")
    _commit_file(pd, "main_sb.txt", "main sb\n")

    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False

    assert not repo_has_branch(git_original_project, "sandbox-feature")
    assert (git_original_project / "main_sb.txt").read_text() == "main sb\n"
    assert not (git_original_project / "feature_sb.txt").exists()

    s.sync_to_original(all_branches=True)
    assert s.may_need_to_sync_to_original() is False
    assert repo_has_branch(git_original_project, "sandbox-feature")
    assert (git_original_project / "main_sb.txt").read_text() == "main sb\n"


# ===========================================================================
# SandboxDirs – list_sandbox_only_branches
# ===========================================================================


def test_list_sandbox_only_branches_no_path(sandbox_base: Path) -> None:
    s = SandboxDirs("sb_list_no_path", base_path=sandbox_base)
    s.create()
    with pytest.raises(ValueError, match="Original project path"):
        s.list_sandbox_only_branches()


def test_list_sandbox_only_branches_non_git_sandbox(
    fake_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_list_nongit_sb",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    s.create()
    assert s.list_sandbox_only_branches() == set()


def test_list_sandbox_only_branches_non_git_original(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_nongit"
    orig.mkdir()
    (orig / "a.txt").write_text("a\n", encoding="utf-8")
    s = SandboxDirs(
        "sb_list_nongit_orig",
        project_path=str(orig),
        base_path=sandbox_base,
    )
    s.create()
    assert s.list_sandbox_only_branches() == set()


def test_list_sandbox_only_branches_git_sandbox_non_git_original(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """Sandbox is a git repo but original isn't – list_sandbox_only_branches
    returns empty set."""
    orig = git_original_project.parent / "non_git_orig_list"
    orig.mkdir()
    (orig / "a.txt").write_text("a\n", encoding="utf-8")
    s = SandboxDirs(
        "sb_list_git_sb_non_git_orig",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    # Replace the original project path with a non-git directory
    s._original_project_path = orig
    assert s.list_sandbox_only_branches() == set()


def test_list_sandbox_only_branches_extra_branch_in_sandbox(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_list_extra_branch",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    repo_checkout_create(pd, "sandbox-only-list")
    repo_checkout(pd, "main")
    result = s.list_sandbox_only_branches()
    assert len(result) == 1
    assert "sandbox-only-list" in result
    assert "main" not in result


def test_list_sandbox_only_branches_no_extra_branches(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_list_no_extra",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    result = s.list_sandbox_only_branches()
    assert result == set()


# ===========================================================================
# SandboxDirs – sync_to_original_auto_rebase
# ===========================================================================


def test_sync_to_original_auto_rebase_no_original_path(sandbox_base: Path) -> None:
    s = SandboxDirs("sb_auto_rebase_no_path", base_path=sandbox_base)
    s.create()
    assert s._original_project_path is None
    s.sync_to_original_auto_rebase()


def test_sync_to_original_auto_rebase_succeeds(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_auto_rebase_ok",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (s.project_dir / "auto_rebase.txt").write_text("content\n", encoding="utf-8")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original_auto_rebase()
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "auto_rebase.txt").read_text() == "content\n"


def test_sync_to_original_auto_rebase_retries_on_ff_failure(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_auto_rebase_retry",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd, "sb_file.txt", "sandbox\n")
    _commit_file(git_original_project, "orig_file.txt", "original\n")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original_auto_rebase()
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "sb_file.txt").read_text() == "sandbox\n"
    assert (git_original_project / "orig_file.txt").read_text() == "original\n"


def test_sync_to_original_auto_rebase_reraises_other_error(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_auto_rebase_err2",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (s.project_dir / "feature.txt").write_text("content\n", encoding="utf-8")
    (git_original_project / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="is dirty"):
        s.sync_to_original_auto_rebase()


def test_may_need_to_sync_git_checkout_error(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_checkout_err",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()

    pd = s.project_dir
    orig_repo = git.Repo(str(git_original_project))
    sandbox_repo = git.Repo(str(pd))
    active_name = sandbox_repo.active_branch.name
    try:

        class MockSBR:
            def __init__(self, *a, **kw) -> None:
                pass

            @property
            def active_branch(self):
                return sandbox_repo.active_branch

            @property
            def heads(self):
                return sandbox_repo.heads

            @property
            def head(self):
                return sandbox_repo.head

            @property
            def git(self):
                return self

            @staticmethod
            def config(*_a, **_kw):
                raise git.GitCommandError("config", 1)

            @staticmethod
            def checkout(*_a, **_kw):
                raise git.GitCommandError("checkout", 1)

            @property
            def submodules(self):
                return []

            def is_dirty(self, **kwargs):  # noqa: ARG002
                return False

            def close(self) -> None:
                pass

        def repo_factory(path, *_a, **_kw):
            if str(Path(path).resolve()) == str(git_original_project.resolve()):
                m = type("MO", (), {})()

                class FakeAB:
                    name = active_name

                m.active_branch = FakeAB()
                m.head = sandbox_repo.head
                m.heads = sandbox_repo.heads

                class FakeCommitSHARaising:
                    @property
                    def hexsha(self):
                        return "abc"

                def commit_raiser(_name):
                    raise git.GitCommandError("rev-parse", 1)

                m.commit = commit_raiser
                m.close = lambda: None
                return m
            return MockSBR()

        with patch("git.Repo", side_effect=repo_factory):
            assert s.may_need_to_sync_to_original() is True
    finally:
        sandbox_repo.close()
        orig_repo.close()


def apply_one_patch(target_dir: Path) -> None:
    """Apply the first patch file found in *target_dir*/tiz_patches/ using ``patch -Z``.

    Parameters
    ----------
    target_dir:
        Directory containing the ``tiz_patches/`` subdirectory.
        Defaults to ``self._original_project_path``.
    """
    if target_dir is None or not target_dir.exists():
        raise ValueError("Target project path not found")
    patches_dir = target_dir / TIZ_PATCHES_DIR
    if not patches_dir.exists():
        raise ValueError("Target has no patches dir")
    patch_files = sorted(patches_dir.glob("*.patch"))
    if not patch_files:
        raise ValueError("Target has no patches")
    if len(patch_files) != 1:
        raise ValueError("Target has too many patches")
    patch_path = patch_files[0]
    result = subprocess.run(
        ["patch", "-Z", "-p1", "-i", str(patch_path)],
        cwd=target_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"patch command failed: {result.stderr.strip()}")


def test_may_need_to_sync_non_git_symlink_in_orig(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_symlink_test"
    orig.mkdir()
    (orig / "file.txt").write_text("same\n", encoding="utf-8")
    s = SandboxDirs("sb_non_git_sym", project_path=str(orig), base_path=sandbox_base)
    s.create()
    # Add symlink to project dir only (orig doesn't have it)
    target = tmp_path / "outside"
    target.mkdir()
    (target / "link").write_text("outside\n", encoding="utf-8")
    outside_file = s.project_dir / "link"
    outside_file.symlink_to(target / "link")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    apply_one_patch(orig)
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_no_original_path(sandbox_base: Path) -> None:
    s = SandboxDirs("sb_may_need_1", base_path=sandbox_base)
    s.create()
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_non_git_files_same(
    fake_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_2",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    s.create()
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_non_git_files_changed(
    fake_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_3",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    s.create()
    time.sleep(0.05)
    (s.project_dir / "README.md").write_text("changed\n", encoding="utf-8")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    apply_one_patch(fake_original_project)
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_non_git_file_missing_in_sandbox(
    fake_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_4",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (s.project_dir / "README.md").unlink()
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    apply_one_patch(fake_original_project)
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_non_git_file_extra_in_sandbox(
    fake_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_5",
        project_path=str(fake_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (s.project_dir / "new_file.txt").write_text("extra\n", encoding="utf-8")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    apply_one_patch(fake_original_project)
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_non_git_with_gitignore(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_may_need_gitignore"
    orig.mkdir()
    (orig / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (orig / "keep.txt").write_text("same\n", encoding="utf-8")
    (orig / "skip.log").write_text("ignore me\n", encoding="utf-8")
    s = SandboxDirs("sb_may_need_6", project_path=str(orig), base_path=sandbox_base)
    s.create()
    (s.project_dir / "skip.log").write_text("changed ignored\n", encoding="utf-8")
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_git_same_commit(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_7",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_git_different_commit(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_8",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(s.project_dir, "new_file.txt", "new\n")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_git_missing_branch(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_9",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    repo_checkout_create(s.project_dir, "other-branch")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original(all_branches=True)
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_git_detached_head(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_10",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    subprocess.run(
        ["git", "checkout", "--detach"],
        cwd=git_original_project,
        check=True,
        capture_output=True,
    )
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_project_dir_not_exists(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_11",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    shutil.rmtree(s.project_dir)
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_git_dirty(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_dirty",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    (s.project_dir / "untracked.txt").write_text("untracked\n")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original_auto_rebase()
    assert s.may_need_to_sync_to_original() is False


def test_may_need_to_sync_git_detached_head_in_sandbox(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_may_need_detached_sb",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    subprocess.run(
        ["git", "checkout", "--detach"],
        cwd=s.project_dir,
        check=True,
        capture_output=True,
    )
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original(all_branches=True)
    assert s.may_need_to_sync_to_original() is False


def test_non_git_may_need_sync_symlink_not_file(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_sym"
    orig.mkdir()
    (orig / "file.txt").write_text("same\n", encoding="utf-8")
    link_target = tmp_path / "target"
    link_target.mkdir()
    (link_target / "real.txt").write_text("real\n", encoding="utf-8")
    s = SandboxDirs("sb_nongit_link", project_path=str(orig), base_path=sandbox_base)
    s.create()
    # Add symlink to original after sandbox creation so sandbox doesn't have it
    (orig / "link").symlink_to(link_target)
    assert s.may_need_to_sync_to_original() is True


def test_sync_to_original_all_branches_force(
    git_original_project: Path, sandbox_base: Path
) -> None:
    """sync_to_original with all_branches=True and force=True resets original
    branches to match sandbox, covering the git.reset --hard path."""
    s = SandboxDirs(
        "sb_all_branches_force",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd, "sand_main.txt", "sandbox main\n")
    repo_checkout_create(pd, "feature")
    _commit_file(pd, "feature.txt", "sandbox feature\n")
    repo_checkout(pd, "main")
    _commit_file(git_original_project, "orig_extra.txt", "orig extra\n")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original(all_branches=True, force=True)
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "sand_main.txt").read_text() == "sandbox main\n"
    assert not (git_original_project / "orig_extra.txt").exists()
    repo_checkout(git_original_project, "feature")
    assert (git_original_project / "feature.txt").read_text() == "sandbox feature\n"


def test_non_git_may_need_sync_symlink_in_proj(
    tmp_path: Path, sandbox_base: Path
) -> None:
    orig = tmp_path / "orig_sym_proj"
    orig.mkdir()
    (orig / "file.txt").write_text("same\n", encoding="utf-8")
    s = SandboxDirs(
        "sb_nongit_link_proj", project_path=str(orig), base_path=sandbox_base
    )
    s.create()
    link_target = tmp_path / "link_target"
    link_target.mkdir()
    (link_target / "real.txt").write_text("real\n", encoding="utf-8")
    (s.project_dir / "link").symlink_to(link_target)
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    apply_one_patch(orig)
    assert s.may_need_to_sync_to_original() is False


def test_git_capture_branch_normal(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_git_capture",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    result = s.git_capture_branch()
    assert result == "main"


def test_git_capture_branch_detached(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_git_capture_det",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    subprocess.run(
        ["git", "checkout", "--detach"],
        cwd=pd,
        check=True,
        capture_output=True,
    )
    result = s.git_capture_branch()
    assert result.startswith("tiz-detached-")
    repo = git.Repo(str(pd))
    try:
        assert repo.active_branch.name == result
    finally:
        repo.close()


def test_git_create_branch(git_original_project: Path, sandbox_base: Path) -> None:
    s = SandboxDirs(
        "sb_git_create",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    s.git_create_branch("new-feature")
    repo = git.Repo(str(s.project_dir))
    try:
        assert "new-feature" in {h.name for h in repo.heads}
        assert repo.active_branch.name == "new-feature"
    finally:
        repo.close()


def test_git_checkout(git_original_project: Path, sandbox_base: Path) -> None:
    s = SandboxDirs(
        "sb_git_checkout",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    s.git_create_branch("feature-branch")
    repo = git.Repo(str(s.project_dir))
    try:
        assert repo.active_branch.name == "feature-branch"
    finally:
        repo.close()
    s.git_checkout("main")
    repo = git.Repo(str(s.project_dir))
    try:
        assert repo.active_branch.name == "main"
    finally:
        repo.close()


def test_git_finalize_branches(git_original_project: Path, sandbox_base: Path) -> None:
    s = SandboxDirs(
        "sb_git_finalize",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd, "sandbox_file.txt", "sandbox content\n")
    repo = git.Repo(str(pd))
    try:
        repo.git.checkout("-b", "temp-branch")
    finally:
        repo.close()

    _commit_file(pd, "temp_file.txt", "temp content\n")

    repo = git.Repo(str(pd))
    try:
        winner_commit = repo.head.commit.hexsha
        repo.git.checkout("-b", "temp-branch2")
    finally:
        repo.close()

    _commit_file(pd, "temp_file2.txt", "temp2 content\n")
    branches_to_delete = ["temp-branch", "temp-branch2"]

    s.git_finalize_branches("main", "temp-branch", branches_to_delete)

    repo = git.Repo(str(pd))
    try:
        assert repo.active_branch.name == "main"
        assert repo.head.commit.hexsha == winner_commit
        assert "temp-branch" not in {h.name for h in repo.heads}
        assert "temp-branch2" not in {h.name for h in repo.heads}
    finally:
        repo.close()


def test_git_finalize_branches_nonexistent_branch(
    git_original_project: Path, sandbox_base: Path
) -> None:
    s = SandboxDirs(
        "sb_git_finalize_no_exist",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    s.git_finalize_branches("main", "main", ["does-not-exist"])
    assert s.project_dir.exists()


# ===========================================================================
# SandboxDirs – git repos with submodules
# ===========================================================================


def _git_env_for_submodules() -> dict[str, str]:
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "protocol.file.allow",
        "GIT_CONFIG_VALUE_0": "always",
    }


@pytest.fixture
def git_repo_with_submodule(tmp_path: Path) -> Generator[Path, None, None]:
    proj = tmp_path / "git_with_sub"
    sub_path = tmp_path / "sub_lib"
    _init_git_repo(sub_path)
    _commit_file(sub_path, "lib.py", "def hello(): return 42\n")
    _init_git_repo(proj)
    _commit_file(proj, "README.md", "# Project with submodule\n")
    subprocess.run(
        [
            "git",
            "-C",
            str(proj),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(sub_path),
            "lib",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(proj), "commit", "-m", "add submodule"],
        check=True,
        capture_output=True,
    )
    yield proj


def test_create_with_submodule(
    git_repo_with_submodule: Path, sandbox_base: Path, monkeypatch: Any
) -> None:
    for k, v in _git_env_for_submodules().items():
        monkeypatch.setenv(k, v)
    s = SandboxDirs(
        "sb_sub_create",
        project_path=str(git_repo_with_submodule),
        base_path=sandbox_base,
    )
    s.create()
    assert s.exists()
    assert s.project_dir.exists()
    assert (s.project_dir / "README.md").exists()
    assert (s.project_dir / "lib" / "lib.py").exists()
    assert (s.project_dir / ".gitmodules").exists()
    repo = git.Repo(s.project_dir)
    try:
        assert len(repo.submodules) == 1
        assert repo.submodules[0].name == "lib"
    finally:
        repo.close()


def test_sync_from_original_with_submodule(
    git_repo_with_submodule: Path, sandbox_base: Path, monkeypatch: Any
) -> None:
    for k, v in _git_env_for_submodules().items():
        monkeypatch.setenv(k, v)
    s = SandboxDirs(
        "sb_sub_sync_from",
        project_path=str(git_repo_with_submodule),
        base_path=sandbox_base,
    )
    s.create()
    sub_path = git_repo_with_submodule.parent / "sub_lib"
    _commit_file(sub_path, "lib.py", "def hello(): return 99\n")
    subprocess.run(
        ["git", "-C", str(git_repo_with_submodule), "checkout", "main"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo_with_submodule / "lib"),
            "pull",
            "origin",
            "main",
        ],
        check=True,
        env={**os.environ, **_git_env_for_submodules()},
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo_with_submodule),
            "add",
            "lib",
            ".gitmodules",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo_with_submodule),
            "commit",
            "-m",
            "update submodule",
            "--author",
            "Test <test@example.com>",
        ],
        check=True,
        capture_output=True,
    )
    s.sync_from_original()
    assert (s.project_dir / "lib" / "lib.py").read_text() == "def hello(): return 99\n"


def test_sync_from_original_submodule_force(
    git_repo_with_submodule: Path, sandbox_base: Path, monkeypatch: Any
) -> None:
    for k, v in _git_env_for_submodules().items():
        monkeypatch.setenv(k, v)
    s = SandboxDirs(
        "sb_sub_sync_from_force",
        project_path=str(git_repo_with_submodule),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd / "lib", "lib.py", "def hello(): return 777\n")
    _commit_file(git_repo_with_submodule, "README.md", "# Updated project\n")
    sub_path = git_repo_with_submodule.parent / "sub_lib"
    _commit_file(sub_path, "lib.py", "def hello(): return 55\n")
    subprocess.run(
        ["git", "-C", str(git_repo_with_submodule), "checkout", "main"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo_with_submodule / "lib"),
            "pull",
            "origin",
            "main",
        ],
        check=True,
        env={**os.environ, **_git_env_for_submodules()},
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo_with_submodule),
            "add",
            "lib",
            ".gitmodules",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo_with_submodule),
            "commit",
            "-m",
            "update submodule",
            "--author",
            "Test <test@example.com>",
        ],
        check=True,
        capture_output=True,
    )
    s.sync_from_original(force=True)
    assert (pd / "README.md").read_text() == "# Updated project\n"
    repo = git.Repo(pd / "lib")
    try:
        assert (pd / "lib" / "lib.py").read_text() == "def hello(): return 55\n"
    finally:
        repo.close()


def test_sync_to_original_with_submodule(
    git_repo_with_submodule: Path, sandbox_base: Path, monkeypatch: Any
) -> None:
    for k, v in _git_env_for_submodules().items():
        monkeypatch.setenv(k, v)
    s = SandboxDirs(
        "sb_sub_sync_to",
        project_path=str(git_repo_with_submodule),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(s.project_dir, "sandbox_file.txt", "sandbox content\n")
    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    assert (
        git_repo_with_submodule / "sandbox_file.txt"
    ).read_text() == "sandbox content\n"
    repo = git.Repo(git_repo_with_submodule)
    try:
        assert len(repo.submodules) == 1
    finally:
        repo.close()


def test_sync_to_original_with_submodule_force(
    git_repo_with_submodule: Path, sandbox_base: Path, monkeypatch: Any
) -> None:
    for k, v in _git_env_for_submodules().items():
        monkeypatch.setenv(k, v)
    s = SandboxDirs(
        "sb_sub_sync_to_force",
        project_path=str(git_repo_with_submodule),
        base_path=sandbox_base,
    )
    s.create()
    _commit_file(git_repo_with_submodule, "orig_extra.txt", "orig extra\n")
    _commit_file(s.project_dir, "sandbox_file.txt", "sandbox file\n")
    assert s.may_need_to_sync_to_original() is True
    with pytest.raises(RuntimeError, match="not fast forward"):
        s.sync_to_original()
    s.sync_to_original(force=True)
    assert s.may_need_to_sync_to_original() is False
    assert not (git_repo_with_submodule / "orig_extra.txt").exists()
    assert (
        git_repo_with_submodule / "sandbox_file.txt"
    ).read_text() == "sandbox file\n"


def test_sync_from_and_to_original_with_submodule_roundtrip(
    git_repo_with_submodule: Path, sandbox_base: Path, monkeypatch: Any
) -> None:
    for k, v in _git_env_for_submodules().items():
        monkeypatch.setenv(k, v)
    s = SandboxDirs(
        "sb_sub_roundtrip",
        project_path=str(git_repo_with_submodule),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd, "sandbox_main.txt", "sandbox main\n")
    _commit_file(git_repo_with_submodule, "orig_main.txt", "orig main\n")
    assert s.may_need_to_sync_to_original() is True
    with pytest.raises(RuntimeError, match="not fast forward"):
        s.sync_to_original()
    s.sync_from_original()
    s.sync_to_original()
    assert s.may_need_to_sync_to_original() is False
    assert (
        git_repo_with_submodule / "sandbox_main.txt"
    ).read_text() == "sandbox main\n"
    assert (git_repo_with_submodule / "orig_main.txt").read_text() == "orig main\n"
    assert (pd / "sandbox_main.txt").read_text() == "sandbox main\n"
    assert (pd / "orig_main.txt").read_text() == "orig main\n"


def test_sync_to_original_all_branches_skips_head_ref(
    git_original_project: Path, sandbox_base: Path, monkeypatch: Any
) -> None:
    """When sync_to_original with all_branches=True, HEAD ref from the remote
    should be skipped (line 1018 in sandbox_dirs.py)."""
    s = SandboxDirs(
        "sb_head_ref_skip",
        project_path=str(git_original_project),
        base_path=sandbox_base,
    )
    s.create()
    pd = s.project_dir
    _commit_file(pd, "sandbox_file.txt", "sandbox file\n")

    orig_fetch = git.Remote.fetch

    def patched_fetch(self, *args: Any, **kwargs: Any) -> Any:
        result = orig_fetch(self, *args, **kwargs)
        # After fetch, write a HEAD ref into the remote tracking namespace
        refs_dir = Path(self.repo.git_dir) / "refs" / "remotes" / self.name
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "HEAD").write_text(self.repo.head.commit.hexsha + "\n")
        return result

    monkeypatch.setattr(git.Remote, "fetch", patched_fetch)

    assert s.may_need_to_sync_to_original() is True
    s.sync_to_original(all_branches=True)
    assert s.may_need_to_sync_to_original() is False
    assert (git_original_project / "sandbox_file.txt").read_text() == "sandbox file\n"
