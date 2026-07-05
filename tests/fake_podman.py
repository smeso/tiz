#!/usr/bin/env python3
"""Fake podman that simulates container lifecycle for tiz integration tests.

This script reads mount arguments to find the worker script and shared
directory, then launches sandbox_worker.py directly on the host.
Container IDs are stored in a temporary directory for verification.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_STORAGE: Path | None = None


def _get_storage() -> Path:
    global _STORAGE
    if _STORAGE is None:
        _STORAGE = Path(
            os.environ.get("FAKE_PODMAN_STORAGE", "/tmp/fake-podman-storage")
        )
        _STORAGE.mkdir(parents=True, exist_ok=True)
    return _STORAGE


def start_worker(worker_path: str, sock_path: str) -> subprocess.Popen:
    # AF_UNIX socket paths are limited to ~108 bytes. If the expected path
    # is too long, create the socket at a shorter temp path and symlink it.
    actual_sock_path = sock_path
    sock_link: Path | None = None
    if len(sock_path) >= 100:
        tmp_sock = Path("/tmp") / f"fps-{os.urandom(4).hex()}.sock"
        actual_sock_path = str(tmp_sock)
        sock_link = Path(sock_path)

    proc = subprocess.Popen(
        [sys.executable, str(worker_path), actual_sock_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    for _ in range(200):
        if Path(actual_sock_path).exists():
            break
        time.sleep(0.05)
    else:
        proc.terminate()
        try:
            err = proc.communicate(timeout=3)[1].decode()
        except subprocess.TimeoutExpired:
            err = "(timeout reading stderr)"
        raise RuntimeError(
            f"Worker socket {actual_sock_path} never appeared. "
            f"Worker stderr: {err[:500]}"
        )

    # Symlink from the expected (long) path to the actual socket so
    # callers that check the original path find it.
    if sock_link is not None:
        sock_link.parent.mkdir(parents=True, exist_ok=True)
        sock_link.symlink_to(actual_sock_path)

    return proc


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("fake-podman: no args", file=sys.stderr)
        sys.exit(1)

    cmd = args[0]
    storage = _get_storage()

    if cmd == "run" and "-d" in args:
        container_name = None
        worker_src = None
        own_shared = None
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--name" and i + 1 < len(args):
                container_name = args[i + 1]
                i += 2
                continue
            if a == "--mount" and i + 1 < len(args):
                parts = args[i + 1].split(",")
                source = None
                target = None
                for p in parts:
                    if p.startswith("source="):
                        source = p[7:]
                    elif p.startswith("target="):
                        target = p[7:]
                if target == "/usr/local/bin/worker.py":
                    worker_src = source
                elif (
                    target is not None
                    and "/container_shared" in target
                    and source is not None
                ):
                    own_shared = Path(source)
                i += 2
                continue
            if a == "--network" and i + 1 < len(args):
                i += 2
                continue
            if a == "--workdir" and i + 1 < len(args):
                i += 2
                continue
            if a.startswith("--"):
                i += 1
                continue
            i += 1

        if container_name is None:
            container_name = "unknown"

        container_id = "fake_" + os.urandom(4).hex()
        cid_file = storage / container_id
        cid_file.write_text(json.dumps({"name": container_name}), encoding="utf-8")

        if worker_src and own_shared is not None and own_shared.exists():
            sock_path = str(own_shared / "exe.sock")
            start_worker(worker_src, sock_path)

        print(container_id)
        sys.exit(0)

    elif cmd == "exec":
        cid = args[1]
        cmd_args = args[2:] if len(args) > 2 else ["/bin/bash", "-l"]
        cid_file = storage / cid
        if not cid_file.exists():
            print(f"Error: container {cid} not found", file=sys.stderr)
            sys.exit(1)
        proc = subprocess.run(cmd_args)
        sys.exit(proc.returncode)

    elif cmd == "inspect":
        # Support: inspect cid, inspect --format FMT cid, inspect cid --format FMT
        fmt = None
        cid = None
        positional = [a for a in args[1:] if not a.startswith("--")]
        if positional:
            cid = positional[-1]
        if "--format" in args:
            fmt_idx = args.index("--format") + 1
            if fmt_idx < len(args):
                fmt = args[fmt_idx]
        if cid is None:
            print("Error: missing container id", file=sys.stderr)
            sys.exit(1)
        cid_file = storage / cid
        if cid_file.exists():
            if fmt is not None:
                if fmt == "{{.State.Running}}":
                    print("true")
                elif fmt == "{{.Id}}":
                    print(cid)
                else:
                    print(f"Warning: unknown format {fmt}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(json.dumps([{"State": {"Running": True, "Status": "running"}}]))
            sys.exit(0)
        if fmt is not None:
            print("false")
        sys.exit(1)

    elif cmd in ("stop", "rm"):
        cid = args[-1]
        cid_file = storage / cid
        if cid_file.exists():
            cid_file.unlink()
        sys.exit(0)

    elif cmd == "images":
        print("REPOSITORY   TAG       IMAGE ID")
        print("tiz-worker   latest    abc123")
        sys.exit(0)

    elif cmd in ("rmi", "build", "version"):
        if cmd == "version":
            print("podman version 4.9.0")
        sys.exit(0)

    else:
        print(f"Unknown podman command: {args}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
