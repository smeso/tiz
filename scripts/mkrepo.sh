#!/bin/bash
# Build a Debian package and create an unsigned APT repository containing it.
# Run scripts/signrepo.sh separately to sign the Release file.

set -e

projdir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

"$projdir"/scripts/mkdeb.sh

# Locate the built .deb
# shellcheck disable=SC2012  # only one .deb file expected
deb_file=$(ls "$projdir"/tiz_*.deb 2>/dev/null | head -1)
if [[ -z "$deb_file" ]]; then
    echo "ERROR: no .deb package found after build" >&2
    exit 1
fi

repo_dir="$projdir/repo"
dist="stable"
component="main"
arch="all"

echo ""
echo "=== Creating APT repository structure ==="
repodir="$repo_dir/dists/$dist/$component/binary-$arch"
mkdir -p "$repodir"
cp "$deb_file" "$repodir/"

# Build Packages index
cd "$repo_dir" || exit 1
dpkg-scanpackages --multiversion "dists/$dist" > "$repodir/Packages"
gzip -9 -c "$repodir/Packages" > "$repodir/Packages.gz"

# Build Release file
cat > "dists/$dist/Release" <<EOF
Origin: tiz
Label: tiz APT Repository
Suite: $dist
Codename: $dist
Date: $(date -Ru)
Architectures: $arch
Components: $component
Description: tiz - agentic chatbot and harness using sandboxed tools
EOF

# Compute hashes
cd "dists/$dist" || exit 1

hash_lines=""

while IFS= read -r -d '' f; do
    rel="${f#./}"
    md5=$(md5sum "$f" | cut -d' ' -f1)
    sha1=$(sha1sum "$f" | cut -d' ' -f1)
    sha256=$(sha256sum "$f" | cut -d' ' -f1)
    sha512=$(sha512sum "$f" | cut -d' ' -f1)
    size=$(stat -c%s "$f")
    hash_lines+="MD5Sum: $md5 $size $rel"$'\n'
    hash_lines+="SHA1: $sha1 $size $rel"$'\n'
    hash_lines+="SHA256: $sha256 $size $rel"$'\n'
    hash_lines+="SHA512: $sha512 $size $rel"$'\n'
done < <(find "$component" -type f -print0 | sort -z)

# Append hashes to Release
{
    echo "MD5Sum:"
    echo "$hash_lines" | grep '^MD5Sum:' | cut -d' ' -f2-
    echo "SHA1:"
    echo "$hash_lines" | grep '^SHA1:' | cut -d' ' -f2-
    echo "SHA256:"
    echo "$hash_lines" | grep '^SHA256:' | cut -d' ' -f2-
    echo "SHA512:"
    echo "$hash_lines" | grep '^SHA512:' | cut -d' ' -f2-
} >> Release

cd "$projdir"

echo ""
echo "=== Repository structure ==="
find "$repo_dir" -type f | sed "s|$repo_dir/||"

echo ""
echo "OK: Unsigned APT repository created in $repo_dir"
echo ""
echo "Run:  scripts/signrepo.sh"
echo "to sign the Release file and make the repository usable."
echo ""
echo "To use it, add the following to /etc/apt/sources.list.d/tiz.list:"
echo "  deb [signed-by=<path-to-pubkey>] file:$repo_dir $dist $component"
echo ""
echo "Export the public key with:"
echo "  gpg --armor --export <key-id>"
