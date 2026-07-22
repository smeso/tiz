#!/bin/bash
# Sign the Release file of an APT repository previously created by mkrepo.sh.
set -euo pipefail

gpg_key_id="85F0580B9DACBE6E"

projdir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"
repo_dir="$projdir/repo"
dist="stable"

echo ""
echo "=== Signing Release file ==="

cd "$repo_dir/dists/$dist" || exit 1

# Check that Release file exists (i.e. mkrepo.sh has been run)
if [[ ! -f Release ]]; then
    echo "ERROR: No Release file found at $repo_dir/dists/$dist/Release" >&2
    echo "Run scripts/mkrepo.sh first to create the repository structure." >&2
    exit 1
fi

gpg --batch --yes --detach-sign \
    --armor \
    --local-user "$gpg_key_id" \
    --output Release.gpg \
    Release

# InRelease (clearsigned inline)
gpg --batch --yes --clearsign \
    --local-user "$gpg_key_id" \
    --output InRelease \
    Release

cd "$projdir"

echo ""
echo "=== Repository structure (signed) ==="
find "$repo_dir" -type f | sed "s|$repo_dir/||"

echo ""
echo "OK: Signed APT repository in $repo_dir"
