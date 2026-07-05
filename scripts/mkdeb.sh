#!/bin/bash
# Build a Debian package for tiz and run package linters.
set -euo pipefail

projdir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"

# dpkg-buildpackage outputs to the parent directory of the source tree.
# Since the project root is on a read-only filesystem, we build in a
# temporary copy so the .deb lands in a writable location.
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

echo "=== Copying source to build directory ==="
cp -a "$projdir/." "$tmpdir/source"
cd "$tmpdir/source"

echo "=== Building Debian package ==="
dpkg-buildpackage --no-sign -b

echo ""
echo "=== Copying package to project root ==="
cp "$tmpdir"/tiz_*.deb "$projdir/" 2>/dev/null || true

echo ""
echo "=== Lintian package check ==="
lintian --info --display-experimental --pedantic \
    "$projdir"/tiz_*.deb 2>&1 || true

echo ""
echo "=== Package contents ==="
dpkg-deb --info "$projdir"/tiz_*.deb

echo ""
echo "OK: Debian package built successfully."
