"""Write THIRD-PARTY-NOTICES.txt for the shipped client (the zip's app\\mc-client.exe).

    python -I third_party_notices.py OUT.txt [--package mc-client] [--target TRIPLE]
                                     [--no-default-features] [--all-features] [--features LIST]

    package-aero-client.ps1 runs it while staging the zip, with the zip's own feature flags:
    python -I third_party_notices.py <app>\\licenses\\THIRD-PARTY-NOTICES.txt \\
        --no-default-features --features mimalloc

Aero Client itself is licensed under the GNU General Public License, version 3 or (at your
option) any later version (GPL-3.0-or-later); the zip carries the version 3 text as
app\\LICENSE.txt (a copy of mc-rs/LICENSE). This file
lists what the program contains that keeps a licence of its own.

Which crates: exactly what `cargo tree -p <package> <features> -e normal,no-proc-macro --target
<triple>` lists, minus this workspace's own crates. That is every library linked into the
program for that feature set and target; build-script and development dependencies, and
compile-time procedural macros (code generators that run during the build and are not part of
the program), are left out. For each crate: name, version, licence expression (from
`cargo metadata`) and the text of every LICENSE*, LICENCE*, COPYING*, NOTICE*, UNLICENSE and
COPYRIGHT* file in its package folder, plus a licence file the manifest names. Identical texts are
printed once and referred to by number. A crate under a licence that asks the distributor to say
where its source is (MPL-2.0, LGPL, GPL, EPL, CDDL) also gets a source line (its crates.io page for
the exact version). Then the licence of the bundled Inter typeface, the notice of the fdlibm
routine the client's StrictMath-compatible logarithm is transliterated from (read from the comment
beside that function in mc-math), and the notices of shader code that follows published code
(the notice files listed in SHADER_NOTICES, beside the shaders). A missing notice stops the
script.

Offline by design (`--offline --locked`): after a build, every crate is in cargo's cache, and the
lock file decides the versions, so the notices match the build exactly. The output names no local
path: only crate names, versions and the file names inside each package. The script refuses to
write a file that contains this PC's profile, cargo home or workspace path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]          # mc-rs/
INTER_DIR = WORKSPACE / "crates/mc-ui/assets/fonts/inter"
INTER_OFL = INTER_DIR / "OFL.txt"
FDLIBM_SOURCE = WORKSPACE / "crates/mc-math/src/math_helper.rs"
# Notice files beside shader code that follows published code under a licence that asks for its
# notice. Each file says what follows what, then quotes the notice word for word.
SHADER_NOTICES = [
    WORKSPACE / "crates/mc-render/src/post/NOTICE-ambient-occlusion.txt",
]
NOTICE_FILE = re.compile(r"^(licen[cs]e|copying|notice|unlicense|copyright)", re.IGNORECASE)
# Licences whose terms ask whoever distributes the program to tell recipients where the source of
# the licensed code is (MPL-2.0 section 3.2(a), and the copyleft families).
SOURCE_LICENCES = re.compile(r"\b(MPL|LGPL|GPL|EPL|CDDL)-")
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
# Licence texts that name no copyright holder, so one package's copy serves another that ships none.
STANDARD_TEXTS = {
    "Apache-2.0": re.compile(r"^\s*Apache License\s*\n\s*Version 2\.0, January 2004", re.M),
    "BSL-1.0": re.compile(r"Boost Software License - Version 1\.0"),
    "MPL-2.0": re.compile(r"Mozilla Public License Version 2\.0"),
}


def cargo(args: list[str]) -> str:
    out = subprocess.run(["cargo", *args, "--offline", "--locked"], cwd=WORKSPACE,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        sys.exit(f"cargo {args[0]} failed:\n{out.stderr.strip()}")
    return out.stdout


def linked_crates(package: str, target: str, features: list[str]) -> list[tuple[str, str]]:
    """(name, version) of every package in the program, the root included, in first-seen order."""
    tree = cargo(["tree", "-p", package, *features, "-e", "normal,no-proc-macro",
                  "--target", target, "--prefix", "none", "--format", "{p}"])
    seen: list[tuple[str, str]] = []
    for line in tree.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, rest = line.partition(" v")
        version = rest.split(" ", 1)[0]
        if not name or not version:
            sys.exit(f"cannot read a cargo tree line: {line[:60]!r}")
        if (name, version) not in seen:
            seen.append((name, version))
    return seen


def normalized(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def licence_files(package_dir: Path, named: str | None) -> list[tuple[str, str]]:
    """(file name relative to the package, normalized text), sorted by name."""
    found: dict[str, str] = {}
    for entry in sorted(package_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_file() and NOTICE_FILE.match(entry.name):
            found[entry.name] = normalized(entry.read_bytes())
    if named:
        path = (package_dir / named).resolve()
        if path.is_file() and package_dir.resolve() in path.parents:
            found.setdefault(path.relative_to(package_dir.resolve()).as_posix(),
                             normalized(path.read_bytes()))
    return sorted(found.items(), key=lambda kv: kv[0].lower())


def inter_licences() -> list[str]:
    """The typeface's licence text (OFL.txt), then any other licence file in the font folder whose
    text differs (the Display styles come from a later release whose file names another year)."""
    if not INTER_OFL.is_file():
        sys.exit("no OFL.txt in the Inter font folder")
    main_text = normalized(INTER_OFL.read_bytes())
    out = [main_text]
    for entry in sorted(INTER_DIR.iterdir(), key=lambda p: p.name.lower()):
        if entry == INTER_OFL or not entry.is_file() or not NOTICE_FILE.match(entry.name):
            continue
        text = normalized(entry.read_bytes())
        if text != main_text:
            out += ["", f"The same licence as it reads in {entry.name}:", "", text]
    return out


def source_line(pkg: dict) -> str | None:
    """Where the exact source of a crate is, for licences that ask the distributor to say so."""
    if not SOURCE_LICENCES.search(pkg.get("license") or ""):
        return None
    if pkg.get("source") == CRATES_IO:
        return f"https://crates.io/crates/{pkg['name']}/{pkg['version']}"
    return pkg.get("repository") or pkg.get("homepage") or "not published"


def fdlibm_notice() -> str:
    """Sun's fdlibm notice, word for word: the `// ====` block above `strict_log` in mc-math's
    math_helper.rs (that function is transliterated from fdlibm 5.3 `e_log.c`)."""
    lines = normalized(FDLIBM_SOURCE.read_bytes()).split("\n")
    # The file also uses `// ====` lines as section separators: take the delimited block that
    # holds Sun's notice, not merely the first one.
    marks = [i for i, line in enumerate(lines) if line.strip().startswith("// ====")]
    for start, end in zip(marks, marks[1:]):
        if "Copyright (C) 1993 by Sun Microsystems" in "\n".join(lines[start:end]):
            block = [line.strip() for line in lines[start:end + 1]]
            return "\n".join(line[3:] if line.startswith("// ") else line[2:] for line in block)
    sys.exit("no fdlibm notice found beside strict_log in mc-math's math_helper.rs")


def shader_notice(path: Path) -> str:
    """A shader notice file, word for word. It must quote a copyright line."""
    if not path.is_file():
        sys.exit(f"missing shader notice file {path.name}")
    text = normalized(path.read_bytes())
    if "Copyright" not in text:
        sys.exit(f"{path.name} quotes no copyright notice")
    return text


def forbidden_paths() -> list[str]:
    """Folders of this PC that must never appear in the output, in both slash styles."""
    roots = [str(WORKSPACE), str(WORKSPACE.parent), os.environ.get("USERPROFILE", ""),
             os.environ.get("CARGO_HOME", ""), str(Path.home())]
    out = []
    for root in roots:
        if len(root) > 3:
            low = root.lower()
            out.extend({low, low.replace("\\", "/"), low.replace("/", "\\")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--package", default="mc-client")
    ap.add_argument("--target", default="x86_64-pc-windows-msvc")
    # The same feature flags as cargo's, passed on to `cargo tree`.
    ap.add_argument("--no-default-features", action="store_true")
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--features", default="")
    a = ap.parse_args()
    features = (["--no-default-features"] if a.no_default_features else []) \
        + (["--all-features"] if a.all_features else []) \
        + (["--features", a.features] if a.features else [])

    meta = json.loads(cargo(["metadata", "--format-version", "1", "--filter-platform", a.target]))
    members = set(meta["workspace_members"])
    by_key = {(p["name"], p["version"]): p for p in meta["packages"]}
    crates = []
    root_version = "?"
    for name, version in linked_crates(a.package, a.target, features):
        pkg = by_key.get((name, version))
        if pkg is None:
            sys.exit(f"{name} {version} is in cargo tree but not in cargo metadata")
        if pkg["id"] in members:
            if name == a.package:
                root_version = version
            continue
        crates.append(pkg)
    crates.sort(key=lambda p: (p["name"].lower(), p["version"]))

    texts: dict[str, int] = {}          # digest -> text number
    bodies: list[str] = []
    users: list[list[str]] = []
    entries = []
    for pkg in crates:
        files = licence_files(Path(pkg["manifest_path"]).parent, pkg.get("license_file"))
        refs = []
        for file_name, body in files:
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if digest not in texts:
                texts[digest] = len(bodies) + 1
                bodies.append(body)
                users.append([])
            number = texts[digest]
            users[number - 1].append(f"{pkg['name']} {pkg['version']}")
            refs.append(f"{file_name} [text {number}]")
        named = pkg.get("license_file")
        licence = pkg.get("license") or (f"see {named}" if named else "not stated")
        entries.append((pkg, licence, refs, []))

    # A package without licence files: point to the standard text of its licence, when another
    # package ships that text (only licences whose text names no copyright holder).
    standard: dict[str, int] = {}
    for spdx, marker in STANDARD_TEXTS.items():
        for number, body in enumerate(bodies, start=1):
            if marker.search(body):
                standard[spdx] = number
                break
    for pkg, licence, refs, standard_refs in entries:
        if refs:
            continue
        for spdx in re.findall(r"[A-Za-z0-9.+-]+", licence):
            if spdx in standard:
                number = standard[spdx]
                standard_refs.append(f"standard {spdx} text [text {number}]")
                users[number - 1].append(f"{pkg['name']} {pkg['version']}")

    feature_note = " ".join(features) if features else "default features"
    out: list[str] = [
        "Aero Client - third-party notices",
        "",
        "Aero Client is free software: you can redistribute it and/or modify it under the terms of",
        "the GNU General Public License as published by the Free Software Foundation, either",
        "version 3 of the License, or (at your option) any later version (GPL-3.0-or-later).",
        "Copyright (C) 2026 pvps and contributors. The licence text is app\\LICENSE.txt.",
        "",
        "Aero Client's program (app\\mc-client.exe) also contains open-source Rust libraries, the",
        "Inter typeface, a math routine derived from fdlibm and shader code that follows",
        "published code. Each keeps its own licence; their licence and notice texts follow.",
        "Compile-time code generators (procedural macros) that run during the build and are not",
        "part of the program are not listed.",
        "",
        f"Generated from the lock file for {a.package} {root_version}, target {a.target}, "
        f"{feature_note}, by mc-rs/packaging/scripts/third_party_notices.py.",
        "",
        "Contents",
        f"  1. Rust libraries ({len(entries)})",
        "  2. The Inter typeface",
        "  3. A math routine derived from fdlibm",
        "  4. Shader code that follows published code",
        f"  5. Licence and notice texts ({len(bodies)} distinct)",
        "",
        "=" * 78,
        f"1. Rust libraries ({len(entries)})",
        "=" * 78,
        "",
    ]
    for pkg, licence, refs, standard_refs in entries:
        out.append(f"{pkg['name']} {pkg['version']}")
        out.append(f"  Licence: {licence}")
        if refs:
            out.append("  Files: " + ", ".join(refs))
        else:
            source = pkg.get("repository") or pkg.get("homepage") or ""
            out.append("  Files: none in the published package"
                       + "".join(f"; {ref}" for ref in standard_refs)
                       + (f"; source: {source}" if source else ""))
        where = source_line(pkg)
        if where:
            out.append(f"  Source code: {where}")
        out.append("")
    out += ["=" * 78, "2. The Inter typeface", "=" * 78, "",
            "The title screen uses the Inter typeface, embedded in the program. Its licence:", "",
            *inter_licences(), ""]
    out += ["=" * 78, "3. A math routine derived from fdlibm", "=" * 78, "",
            "The program's StrictMath-compatible logarithm (vanilla's random numbers depend on it)",
            "is transliterated from fdlibm 5.3 (e_log.c). Its notice:", "",
            fdlibm_notice(), ""]
    out += ["=" * 78, "4. Shader code that follows published code", "=" * 78, ""]
    for path in SHADER_NOTICES:
        out += [shader_notice(path), ""]
    out += ["=" * 78, f"5. Licence and notice texts ({len(bodies)} distinct)", "=" * 78, ""]
    for number, (body, used_by) in enumerate(zip(bodies, users), start=1):
        out.append(f"---- [text {number}] used by: {', '.join(used_by)}")
        out.append("")
        out.append(body)
        out.append("")
    document = "\n".join(out).rstrip() + "\n"

    low = document.lower()
    leaked = [p for p in forbidden_paths() if p in low]
    if leaked:
        sys.exit("refusing to write: the notices contain a folder of this PC")
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(document.replace("\n", "\r\n").encode("utf-8"))
    print(f"wrote {target.name}: {len(entries)} libraries, {len(bodies)} distinct licence texts, "
          f"{len(document) // 1024} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
