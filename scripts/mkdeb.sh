#!/bin/bash
# Build a Debian package for tiz and run package linters.
set -euo pipefail

projdir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"
src_pkg_dir="$projdir/src/tiz"

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
deb_path=$(find "$tmpdir" -maxdepth 1 -name 'tiz_*.deb' -print -quit 2>/dev/null)
if [[ -n "$deb_path" ]]; then
    cp "$deb_path" "$projdir/"
fi

echo ""
echo "=== Package info ==="
dpkg-deb --info "$projdir"/tiz_*.deb

echo ""
echo "=== Verifying package contents ==="

# shellcheck disable=SC2012  # only one .deb file expected
deb_file=$(ls "$projdir"/tiz_*.deb 2>/dev/null | head -1)
if [[ -z "$deb_file" ]]; then
    echo "ERROR: no .deb package found in project root" >&2
    exit 1
fi

# Collect all files from the deb (exclude dpkg metadata under ./DEBIAN/)
mapfile -t installed_files < <(dpkg-deb --contents "$deb_file" \
    | awk '{for (i=6; i<=NF; i++) printf "%s%s", $i, (i<NF ? OFS : ORS)}' \
    | grep -v '^\./DEBIAN/' \
    | sed 's|^\.||')

# Build list of expected files from the Python package tree.
expected_files=()

# All Python source files (*.py) from src/tiz/
while IFS= read -r -d '' f; do
    rel="${f#"$src_pkg_dir"}"
    expected_files+=( "/usr/lib/python3/dist-packages/tiz${rel}" )
done < <(find "$src_pkg_dir" -name '*.py' -type f -print0)

# py.typed marker
expected_files+=( "/usr/lib/python3/dist-packages/tiz/py.typed" )

# Data files (containerfiles, prompts, web_static, etc.)
while IFS= read -r -d '' f; do
    rel="${f#"$src_pkg_dir"}"
    expected_files+=( "/usr/lib/python3/dist-packages/tiz${rel}" )
done < <(find "$src_pkg_dir/data" -type f -print0)

# Worker scripts
while IFS= read -r -d '' f; do
    rel="${f#"$src_pkg_dir"}"
    expected_files+=( "/usr/lib/python3/dist-packages/tiz${rel}" )
done < <(find "$src_pkg_dir/worker_scripts" -type f -print0)

# Man page and shell completions (from debian/rules)
expected_files+=( "/usr/share/man/man1/tiz.1.gz" )
expected_files+=( "/usr/share/bash-completion/completions/tiz" )
expected_files+=( "/usr/share/zsh/vendor-completions/_tiz" )
expected_files+=( "/usr/share/fish/vendor_completions.d/tiz.fish" )

# Console script entry point (from pyproject.toml)
expected_files+=( "/usr/bin/tiz" )

# Documentation (from debian/tiz.docs) - dh_installdocs compresses with gzip
expected_files+=( "/usr/share/doc/tiz/README.md.gz" )

# Check that every expected file is present in the deb.
missing=()
for ef in "${expected_files[@]}"; do
    found=false
    for installed in "${installed_files[@]}"; do
        if [[ "$installed" == "$ef" ]]; then
            found=true
            break
        fi
    done
    if ! $found; then
        missing+=( "$ef" )
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: the following expected files are missing from the package:" >&2
    for m in "${missing[@]}"; do
        echo "  $m" >&2
    done
    exit 1
fi
echo "All expected files are present."

echo ""
echo "=== Lintian package check ==="
lintian --info --display-experimental --pedantic \
    "$deb_file" 2>&1 || true

echo ""
echo "OK: Debian package built successfully."
