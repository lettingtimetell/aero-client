#!/usr/bin/env sh
# Launcher, macOS. Lives at mc-rs.app/Contents/MacOS/mc-rs and is the bundle's
# CFBundleExecutable.
#
# This artefact ships no Minecraft assets. It
# locates the user's own installation and points the client at it.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -n "${MC_RS_MINECRAFT_DIR:-}" ]; then
  mcdir="$MC_RS_MINECRAFT_DIR"
else
  mcdir="$HOME/Library/Application Support/minecraft"
fi

index="$mcdir/assets/indexes/1.8.json"
jar="$mcdir/versions/1.8.9/1.8.9.jar"

fail() {
  # Launched from Finder there is no terminal to print to, so the message is also shown as a
  # dialog. `osascript` is always present on macOS.
  echo "mc-rs: $1" >&2
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display dialog \"mc-rs: $1\" buttons {\"OK\"} with icon stop" >/dev/null 2>&1 || true
  fi
  exit 3
}

[ -d "$mcdir" ] || fail "no Minecraft installation found at $mcdir. This build deliberately ships no Minecraft assets; install Minecraft with the official launcher and run 1.8.9 once, or set MC_RS_MINECRAFT_DIR."
[ -f "$index" ] || fail "found $mcdir but no asset index at $index. Run 1.8.9 once in the official launcher."
[ -f "$jar" ] || fail "found $mcdir but no version jar at $jar. Install the 1.8.9 profile and start it once."

exec "$here/mc-client" \
  --assetsDir "$mcdir/assets" \
  --assetIndex 1.8 \
  --versionJar "$jar" \
  "$@"
