#!/usr/bin/env sh
# Build the Linux artefact.
#
#   packaging/linux/build.sh [out-dir]
#
# NOT YET RUN. This script is *configuration*: the
# same staging and verification steps as the Windows path, expressed for a Linux runner. The CI
# `package` job is what will execute it for the first time. Do not read its existence as a claim
# that a Linux artefact has been produced.
#
# The payload carries this project's own binary, its documentation and a launcher; it carries no
# Minecraft assets.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace=$(CDPATH= cd -- "$here/../.." && pwd)
out=${1:-"$workspace/target/package"}

version=$(sed -n 's/^version *= *"\(.*\)".*/\1/p' "$workspace/Cargo.toml" | head -n 1)
arch=$(uname -m)
triple="linux-$arch"
name="mc-rs-$version-$triple"
stage="$out/$name"

echo "building $name (profile: dist, default features off)"
# `offline` is dev-only and on by default; a release
# build must turn it off.
(cd "$workspace" && cargo build --profile dist -p mc-client --no-default-features)

binary="$workspace/target/dist/mc-client"
[ -f "$binary" ] || { echo "built binary not found: $binary" >&2; exit 1; }

rm -rf "$stage"
mkdir -p "$stage"
cp "$binary" "$stage/mc-client"
chmod +x "$stage/mc-client"
cp "$here/mc-rs.sh" "$stage/mc-rs"
chmod +x "$stage/mc-rs"
cp "$here/mc-rs.desktop" "$stage/mc-rs.desktop"

cat > "$stage/README.txt" <<EOF
mc-rs $version ($triple)
========================

A native Rust client for Minecraft 1.8.9 servers (protocol 47).

WHAT IS IN THIS ARCHIVE
-----------------------
This project's own binary, this file, and a launcher script. That is all.

WHAT IS DELIBERATELY NOT IN IT
------------------------------
Any file belonging to Mojang: the 1.8.9 jar, block models, textures, sounds, language files,
the asset index, the asset object store. Redistributing them is not permitted, so this build
does not carry them and no configuration of it will.

WHAT YOU NEED
-------------
An existing Minecraft installation made by the official launcher, with the 1.8.9 profile
started at least once. ./mc-rs finds it under \$XDG_DATA_HOME/minecraft or ~/.minecraft, or
wherever MC_RS_MINECRAFT_DIR points.

HOW TO RUN
----------
./mc-rs [any vanilla client option]

Nothing is copied out of your Minecraft directory; it is read in place.

LOGGING IN
----------
This build uses the Microsoft account flow. The dev-only offline login (no sign-in) is compiled
out of release artefacts.
EOF

cat > "$stage/NOTICE.txt" <<'EOF'
mc-rs — NOTICE

This software is an independent client implementation. It is not affiliated with, endorsed by,
or a product of Mojang Studios or Microsoft.

It contains no Minecraft game assets and no Minecraft code. It reads the assets of an existing,
separately obtained Minecraft installation from the machine it runs on, and does not copy or
redistribute them.

Minecraft is a trademark of Mojang AB.
EOF

[ -f "$workspace/LICENSE" ] && cp "$workspace/LICENSE" "$stage/LICENSE.txt" || \
  echo "note: no LICENSE at the workspace root; the artefact ships without one"

sha=$(sha256sum "$stage/mc-client" | cut -d' ' -f1)
cat > "$stage/BUILD-INFO.txt" <<EOF
# mc-rs build manifest, written by packaging/*/build.*
version: $version
triple: $triple
profile: dist
default-features: off
features:
binary: mc-client
binary-bytes: $(wc -c < "$stage/mc-client" | tr -d ' ')
binary-sha256: $sha
built-utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

# One verifier implementation for all three platforms, deliberately: a second copy of the
# forbidden-path rules written in shell is a second copy that can drift, and the rule it enforces
# is a legal one. `pwsh` is preinstalled on GitHub's ubuntu and macOS runners.
if command -v pwsh >/dev/null 2>&1; then
  pwsh -File "$here/../scripts/verify-payload.ps1" -Payload "$stage" -ExpectBinary "mc-client"
else
  echo "ERROR: pwsh not found; cannot verify the payload carries no Minecraft assets." >&2
  echo "Refusing to produce an unverified artefact. Install PowerShell or run the verifier" >&2
  echo "on another machine against $stage." >&2
  exit 1
fi

tarball="$out/$name.tar.gz"
rm -f "$tarball"
tar -czf "$tarball" -C "$out" "$name"

echo
echo "artefact: $tarball"
echo "payload:  $stage"
