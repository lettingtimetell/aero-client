#!/bin/bash
# Profile-guided dist build of mc-client.exe and examples/skywars_bench.exe from the CURRENT tree.
#
#   tools/pgo_build.sh OUT_BINS_DIR
#
# Steps (first measured 2026-11-06: SkyWars -7.8%, e1-16 -8.2%, terrain -1.5% steady / -8.5% load;
# identical screenshots and work):
#   1. instrumented build (-Cprofile-generate) into tgt/pgo-gen
#   2. clear old profiles (also the ones build scripts/proc-macros write during step 1)
#   3. train: tools/pgo_train.py runs SkyWars (pipelined + serial), terrain fixed + orbit, e1-16
#   4. merge with the toolchain's llvm-profdata (rustup component llvm-tools-preview)
#   5. optimised build (-Cprofile-use) into tgt/pgo-use, copied to OUT_BINS_DIR
# Rerun after code changes: a stale profile no longer matches changed functions.
# Flags go through CARGO_ENCODED_RUSTFLAGS: RUSTFLAGS splits paths that contain spaces. Both builds
# get the same --remap-path-prefix list as install-mc-rs.ps1, so panic messages name no folder of
# this PC, and identical flags keep the PGO profile matching.
# Takes ~25 min and leaves target/ (and the installed client) untouched. Not timed: no idle PC needed.
set -euo pipefail
OUT=${1:?usage: tools/pgo_build.sh OUT_BINS_DIR}
# Paths come from this script's location, so the repository can live anywhere. Git Bash's `pwd -W`
# gives the D:/... form that cargo and Windows Python need; plain `pwd` elsewhere.
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && { pwd -W 2>/dev/null || pwd; })"
ROOT="$(cd "$W/../../.." && { pwd -W 2>/dev/null || pwd; })"
# Get-RemapPathFlags of install-mc-rs.ps1: each folder a source path can start with, mapped to a
# neutral name; rustc applies the last match, so the shortest folder comes first. US is the
# separator CARGO_ENCODED_RUSTFLAGS splits on.
US=$'\x1f'
REMAP=""
while IFS=$'\t' read -r _length from to; do
  REMAP+="${US}--remap-path-prefix=${from}=${to}"
done < <(printf '%s\t%s\n' \
    "$(cygpath -m "${USERPROFILE:?}")" '~' \
    "$(dirname "$ROOT")" project \
    "$ROOT" mc-rs \
    "$(cygpath -m "${CARGO_HOME:-$USERPROFILE/.cargo}")" cargo \
    "$(cygpath -m "${RUSTUP_HOME:-$USERPROFILE/.rustup}")" rustup \
    "$W/tgt/pgo-gen" target \
    "$W/tgt/pgo-use" target |
  awk -F'\t' '{ print length($1) "\t" $0 }' | sort -s -n -k1,1)
PROFRAW="$W/pgo/profraw"
MERGED="$W/pgo/merged.profdata"
PROFDATA="$(rustc --print sysroot)/lib/rustlib/x86_64-pc-windows-msvc/bin/llvm-profdata.exe"
[ -x "$PROFDATA" ] || { echo "llvm-profdata missing: rustup component add llvm-tools-preview"; exit 2; }
[ -e "$OUT" ] && { echo "output exists: $OUT"; exit 2; }

cd "$ROOT"
echo "[1/5] instrumented build $(date +%T)"
CARGO_TARGET_DIR="$W/tgt/pgo-gen" CARGO_ENCODED_RUSTFLAGS="-Cprofile-generate=$PROFRAW$REMAP" \
  cargo build --profile dist -p mc-client --bin mc-client --example skywars_bench

echo "[2/5] clear profiles $(date +%T)"
mkdir -p "$PROFRAW"
rm -f "${PROFRAW:?}"/*.profraw

echo "[3/5] train $(date +%T)"
python -I "$W/tools/pgo_train.py" "$ROOT/optimization" \
  "$W/tgt/pgo-gen/dist" "$W/results/pgo-train-$(date +%Y%m%d-%H%M%S)"

echo "[4/5] merge $(date +%T)"
"$PROFDATA" merge -o "$MERGED" "$PROFRAW"/*.profraw

echo "[5/5] optimised build $(date +%T)"
CARGO_TARGET_DIR="$W/tgt/pgo-use" CARGO_ENCODED_RUSTFLAGS="-Cprofile-use=$MERGED$REMAP" \
  cargo build --profile dist -p mc-client --bin mc-client --example skywars_bench
mkdir -p "$OUT/examples"
cp "$W/tgt/pgo-use/dist/mc-client.exe" "$OUT/"
cp "$W/tgt/pgo-use/dist/examples/skywars_bench.exe" "$OUT/examples/"
sha256sum "$OUT/mc-client.exe" "$OUT/examples/skywars_bench.exe"
echo "done $(date +%T)"
