#!/usr/bin/env bash
# Check that the version string is consistent across all files.
set -euo pipefail

errors=0

# Extract version from pyproject.toml
pyproj_ver=$(grep -oP '^version\s*=\s*"\K[^"]+' /opt/project/pyproject.toml)
echo "pyproject.toml:       $pyproj_ver"

# Extract version from debian/changelog (first line)
changelog_ver=$(head -1 /opt/project/debian/changelog | grep -oP '^tiz \(\K[^)]+' | grep -oP '^[0-9]+\.[0-9]+\.[0-9]+')
echo "debian/changelog:     $changelog_ver"

# Extract fallback version from src/tiz/__init__.py
init_ver=$(grep -oP '__version__\s*=\s*"\K[^"]+' /opt/project/src/tiz/__init__.py | head -1)
echo "src/tiz/__init__.py:  $init_ver"

# Extract the fallback version from src/tiz/sandbox_worker.py
sw_ver=$(grep -oP '_tiz_version\s*=\s*"\K[^"]+' /opt/project/src/tiz/sandbox_worker.py)
echo "src/tiz/sandbox_worker.py: $sw_ver"

# Compare
if [[ "$pyproj_ver" != "$changelog_ver" ]]; then
    echo "ERROR: debian/changelog version ($changelog_ver) != pyproject.toml ($pyproj_ver)" >&2
    ((errors++))
fi

if [[ "$pyproj_ver" != "$init_ver" ]]; then
    echo "ERROR: src/tiz/__init__.py version ($init_ver) != pyproject.toml ($pyproj_ver)" >&2
    ((errors++))
fi

if [[ "$pyproj_ver" != "$sw_ver" ]]; then
    echo "ERROR: src/tiz/sandbox_worker.py version ($sw_ver) != pyproject.toml ($pyproj_ver)" >&2
    ((errors++))
fi

if (( errors > 0 )); then
    echo "FAILED: $errors version mismatch(es)" >&2
    exit 1
fi

echo "OK - all versions match ($pyproj_ver)"
