#!/usr/bin/env sh
# Launcher, Linux.
#
# This artefact ships no Minecraft assets. It
# locates the user's own installation and points the client at it, and refuses to start with a
# clear message rather than rendering placeholder blocks when there isn't one.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -n "${MC_RS_MINECRAFT_DIR:-}" ]; then
  candidates="$MC_RS_MINECRAFT_DIR"
else
  candidates="${XDG_DATA_HOME:-$HOME/.local/share}/minecraft $HOME/.minecraft"
fi

mcdir=""
for candidate in $candidates; do
  if [ -d "$candidate" ]; then mcdir="$candidate"; break; fi
done

if [ -z "$mcdir" ]; then
  echo "mc-rs: no Minecraft installation found." >&2
  echo "  looked in:$(for c in $candidates; do printf ' %s' "$c"; done)" >&2
  echo >&2
  echo "This build deliberately ships no Minecraft assets. Install Minecraft with the official" >&2
  echo "launcher and run 1.8.9 once, or set MC_RS_MINECRAFT_DIR to an existing installation." >&2
  exit 3
fi

index="$mcdir/assets/indexes/1.8.json"
jar="$mcdir/versions/1.8.9/1.8.9.jar"

if [ ! -f "$index" ]; then
  echo "mc-rs: found $mcdir but no asset index for 1.8." >&2
  echo "  expected: $index" >&2
  echo "Run 1.8.9 once in the official launcher so it downloads the assets." >&2
  exit 3
fi
if [ ! -f "$jar" ]; then
  echo "mc-rs: found $mcdir but no 1.8.9 version jar." >&2
  echo "  expected: $jar" >&2
  echo "Install the 1.8.9 profile in the official launcher and start it once." >&2
  exit 3
fi

exec "$here/mc-client" \
  --assetsDir "$mcdir/assets" \
  --assetIndex 1.8 \
  --versionJar "$jar" \
  "$@"
