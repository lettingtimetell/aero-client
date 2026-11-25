#!/usr/bin/env bash
# Protocol-5 (1.7.10) wire fixtures: regenerate them from the real 1.7.10 classes.
#
# Compiles parity/wire/java/WireOracle17.java against the 1.7.10 client classes compiled from the
# decompiled 1.7.10 sources, runs it, and writes parity/wire/fixtures-v5/ (one .bin per frame,
# manifest.tsv, decode.tsv, and *.inflated.bin for the chunk packets).
#
#   bash parity/tools/wire_capture_v5.sh            regenerate in place
#   bash parity/tools/wire_capture_v5.sh --check    regenerate to a temp dir and diff
#
# Inputs (environment, all required except JAVA_HOME):
#   MC17_CLASSES  directory of the compiled 1.7.10 client classes (javac, JDK 8, of the decompiled
#                 1.7.10 src/minecraft tree against the libraries below)
#   MC17_LIBS     directory holding the 1.7.10 libraries listed by Mojang's 1.7.10 version file
#                 (*.jar; native jars are skipped)
#   JAVA_HOME     a JDK 8 (javac and java are taken from $JAVA_HOME/bin, else from PATH)
#
# The fixtures are committed: the Rust tests (cargo test -p mc-protocol --test v5_wire_oracle)
# run without a JDK or the decompiled sources.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # parity/
src="$here/wire/java/WireOracle17.java"
fixtures="$here/wire/fixtures-v5"
classes="${MC17_CLASSES:?set MC17_CLASSES to the compiled 1.7.10 classes}"
libs="${MC17_LIBS:?set MC17_LIBS to the 1.7.10 libraries directory}"

bin=""
if [ -n "${JAVA_HOME:-}" ]; then
    bin="$JAVA_HOME/bin/"
fi

[ -f "$classes/net/minecraft/network/PacketBuffer.class" ] || { echo "FAIL: no 1.7.10 classes in $classes (set MC17_CLASSES)" >&2; exit 1; }
[ -d "$libs" ] || { echo "FAIL: no 1.7.10 libraries in $libs (set MC17_LIBS)" >&2; exit 1; }

win() { if command -v cygpath >/dev/null; then cygpath -w "$1"; else printf '%s' "$1"; fi; }
sep=';'
case "$(uname -s 2>/dev/null || echo unknown)" in Linux|Darwin) sep=':' ;; esac

cp="$(win "$classes")"
while IFS= read -r jar; do
    cp="$cp$sep$(win "$jar")"
done < <(find "$libs" -name '*.jar' ! -name '*natives*' | sort)

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

"${bin}javac" -nowarn -encoding UTF-8 -cp "$cp" -d "$(win "$out")" "$(win "$src")"

target="$fixtures"
if [ "${1:-}" = "--check" ]; then
    target="$out/fixtures-v5"
fi
mkdir -p "$target"

"${bin}java" -cp "$cp$sep$(win "$out")" WireOracle17 "$(win "$target")"

if [ "${1:-}" = "--check" ]; then
    if diff -r --brief "$fixtures" "$target"; then
        echo "OK: committed fixtures still match what the 1.7.10 classes produce"
    else
        echo "DRIFT: the committed fixtures do not match this run."
        exit 1
    fi
else
    echo "regenerated $(ls "$fixtures"/*.bin | wc -l) files in $fixtures"
    echo "now run: cargo test -p mc-protocol --test v5_wire_oracle"
fi
