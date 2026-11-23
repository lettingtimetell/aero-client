#!/usr/bin/env sh
# Build the macOS artefact.
#
#   packaging/macos/build.sh [out-dir]
#
# NOT YET RUN (see the same note in packaging/linux/build.sh). This is configuration for a macOS runner, not an
# artefact that has been produced.
#
# Two macOS specifics that are easy to get wrong and are handled here:
#   * The bundle layout is `mc-rs.app/Contents/{MacOS,Resources}` with `Info.plist` at
#     `Contents/`; the executable named by `CFBundleExecutable` must be the one in `MacOS/`.
#   * A downloaded, unsigned bundle is quarantined by Gatekeeper. This build does not sign or
#     notarize — that needs an Apple Developer identity, which this project does not have — so
#     the README tells the user the `xattr -dr com.apple.quarantine` incantation rather than
#     pretending the artefact is signed.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace=$(CDPATH= cd -- "$here/../.." && pwd)
out=${1:-"$workspace/target/package"}

version=$(sed -n 's/^version *= *"\(.*\)".*/\1/p' "$workspace/Cargo.toml" | head -n 1)
arch=$(uname -m)
triple="macos-$arch"
name="mc-rs-$version-$triple"
stage="$out/$name"
app="$stage/mc-rs.app"

echo "building $name (profile: dist, default features off)"
(cd "$workspace" && cargo build --profile dist -p mc-client --no-default-features)

binary="$workspace/target/dist/mc-client"
[ -f "$binary" ] || { echo "built binary not found: $binary" >&2; exit 1; }

rm -rf "$stage"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$binary" "$app/Contents/MacOS/mc-client"
chmod +x "$app/Contents/MacOS/mc-client"
cp "$here/mc-rs.sh" "$app/Contents/MacOS/mc-rs"
chmod +x "$app/Contents/MacOS/mc-rs"
sed "s/@VERSION@/$version/g" "$here/Info.plist" > "$app/Contents/Info.plist"

cat > "$stage/README.txt" <<EOF
mc-rs $version ($triple)
========================

A native Rust client for Minecraft 1.8.9 servers (protocol 47).

WHAT IS IN THIS ARCHIVE
-----------------------
mc-rs.app, containing this project's own binary and a launcher. That is all.

WHAT IS DELIBERATELY NOT IN IT
------------------------------
Any file belonging to Mojang. Redistributing them is not permitted, so this build does not
carry them and no configuration of it will.

WHAT YOU NEED
-------------
An existing Minecraft installation made by the official launcher, with the 1.8.9 profile
started at least once. The launcher finds it at
~/Library/Application Support/minecraft, or wherever MC_RS_MINECRAFT_DIR points.

GATEKEEPER
----------
This bundle is not signed or notarized (that needs an Apple Developer identity). macOS will
refuse to open it after download until you clear the quarantine attribute:

    xattr -dr com.apple.quarantine /path/to/mc-rs.app

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

sha=$(shasum -a 256 "$app/Contents/MacOS/mc-client" | cut -d' ' -f1)
cat > "$stage/BUILD-INFO.txt" <<EOF
# mc-rs build manifest, written by packaging/*/build.*
version: $version
triple: $triple
profile: dist
default-features: off
features:
binary: mc-rs.app/Contents/MacOS/mc-client
binary-bytes: $(wc -c < "$app/Contents/MacOS/mc-client" | tr -d ' ')
binary-sha256: $sha
built-utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

if command -v pwsh >/dev/null 2>&1; then
  pwsh -File "$here/../scripts/verify-payload.ps1" -Payload "$stage" \
    -ExpectBinary "mc-rs.app/Contents/MacOS/mc-client"
else
  echo "ERROR: pwsh not found; cannot verify the payload carries no Minecraft assets." >&2
  echo "Refusing to produce an unverified artefact." >&2
  exit 1
fi

tarball="$out/$name.tar.gz"
rm -f "$tarball"
tar -czf "$tarball" -C "$out" "$name"

echo
echo "artefact: $tarball"
echo "payload:  $stage"
