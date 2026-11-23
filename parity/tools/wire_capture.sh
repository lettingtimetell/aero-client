#!/usr/bin/env bash
# W-CAP: regenerate the vanilla-side wire fixtures.
#
# Compiles parity/wire/java/WireOracle.java against MCP-919-main/bin/minecraft -- the REAL 1.8.9
# client classes, compiled from the same src/ tree the audit read -- and runs it. Output lands in
# parity/wire/fixtures/ as one .bin per case plus manifest.tsv.
#
#   bash parity/tools/wire_capture.sh            regenerate in place
#   bash parity/tools/wire_capture.sh --check    regenerate to a temp dir and diff; non-zero if
#                                                the committed fixtures no longer match vanilla
#
# The fixtures are committed on purpose: the Rust test must run with no JDK and no decompiled
# reference workspace.
# This script is what you run when you ADD a case, not on every build.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # parity/
root="$(cd "$here/.." && pwd)"                            # repo root
mcp="$root/MCP-919-main"
src="$here/wire/java/WireOracle.java"
fixtures="$here/wire/fixtures"

[ -d "$mcp/bin/minecraft" ] || { echo "FAIL: $mcp/bin/minecraft not found - the reference workspace has not been recompiled" >&2; exit 1; }
command -v javac >/dev/null || { echo "FAIL: javac not on PATH (Temurin 21 expected)" >&2; exit 1; }

win() { if command -v cygpath >/dev/null; then cygpath -w "$1"; else printf '%s' "$1"; fi; }
sep=';'
case "$(uname -s 2>/dev/null || echo unknown)" in Linux|Darwin) sep=':' ;; esac

cp="$(win "$mcp/bin/minecraft")"
while IFS= read -r jar; do
    cp="$cp$sep$(win "$jar")"
done < <(find "$mcp/jars/libraries" -name '*.jar' | sort)

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

javac -nowarn -encoding UTF-8 -cp "$cp" -d "$(win "$out")" "$(win "$src")"

target="$fixtures"
if [ "${1:-}" = "--check" ]; then
    target="$out/fixtures"
fi
mkdir -p "$target"

# The oracle prints one netty platform-detection line to stderr on startup; keep stdout clean.
java -cp "$cp$sep$(win "$out")" WireOracle "$(win "$target")"

if [ "${1:-}" = "--check" ]; then
    if diff -r --brief "$fixtures" "$target"; then
        echo "OK: committed fixtures still match what vanilla produces"
    else
        echo "DRIFT: the committed fixtures do not match this run. Either the oracle changed, or"
        echo "       something edited fixtures/ by hand. Fixtures are vanilla's output, never ours."
        exit 1
    fi
else
    echo "regenerated $(ls "$fixtures"/*.bin | wc -l) fixtures in $fixtures"
    echo "now run: cargo test -p mc-parity --test wire_c2s"
fi
