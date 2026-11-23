#!/usr/bin/env bash
# Regenerate the vanilla-side case file for tests/actual_state_java_oracle.rs.
#
#   bash crates/mc-render/tests/actual_state_oracle/run_oracle.sh <cases.tsv> [cases-per-meta]
#   MC_ACTUAL_STATE_ORACLE=<cases.tsv> cargo test -p mc-render --test actual_state_java_oracle -- --ignored
#
# Compiles ActualStateOracle.java against MCP-919-main/bin/minecraft -- the real 1.8.9 client
# classes, recompiled from the decompiled src tree -- and runs it. Needs a JDK on PATH. The classpath
# recipe is parity/tools/wire_capture.sh's.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../../.." && pwd)"   # the MCP-919-main checkout that holds mc-rs/
mcp="$root/MCP-919-main"
[ -d "$mcp/bin/minecraft" ] || { echo "FAIL: $mcp/bin/minecraft not found" >&2; exit 1; }
[ $# -ge 1 ] || { echo "usage: $0 <cases.tsv> [cases-per-meta]" >&2; exit 2; }

win() { if command -v cygpath >/dev/null; then cygpath -w "$1"; else printf '%s' "$1"; fi; }
sep=';'
case "$(uname -s 2>/dev/null || echo unknown)" in Linux|Darwin) sep=':' ;; esac

cp="$(win "$mcp/bin/minecraft")"
while IFS= read -r jar; do
    cp="$cp$sep$(win "$jar")"
done < <(find "$mcp/jars/libraries" -name '*.jar' | sort)

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
javac -nowarn -encoding UTF-8 -cp "$cp" -d "$(win "$out")" "$(win "$here/ActualStateOracle.java")"
java -cp "$cp$sep$(win "$out")" ActualStateOracle "$(win "$1")" "${2:-48}"
