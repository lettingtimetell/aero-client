# Contributing to Aero Client

How to build, run, test and package the client, and what a pull request has to show. Everything
here was written for, and has only been run on, **Windows 10/11 x64**. Commands are PowerShell,
run from the repository root unless a step says `cd mc-rs`.

1. [Rules for changes](#1-rules-for-changes)
2. [Pull request checklist](#2-pull-request-checklist)
3. [Prerequisites](#3-prerequisites)
4. [Your Minecraft 1.8.9 files](#4-your-minecraft-189-files)
5. [Building](#5-building)
6. [Running and playing](#6-running-and-playing)
7. [Tests, formatting and lints](#7-tests-formatting-and-lints)
8. [Measuring performance](#8-measuring-performance)
9. [Reference data and the Java generators](#9-reference-data-and-the-java-generators)
10. [Sign-in](#10-sign-in)
11. [Packaging](#11-packaging)
12. [Releases and updates](#12-releases-and-updates)
13. [CI](#13-ci)
14. [Environment variables](#14-environment-variables)
15. [Display latency on NVIDIA GPUs](#15-display-latency-on-nvidia-gpus)
16. [Troubleshooting](#16-troubleshooting)
17. [Licensing of contributions](#17-licensing-of-contributions)
18. [Security problems](#18-security-problems)

## 1. Rules for changes

- **Vanilla stays vanilla.** With a mod off, or a setting at its vanilla default, the client must
  behave exactly like vanilla 1.8.9: the same packets, in the same order, from the same tick phase,
  and the same picture. Check a change against what the 1.8.9 client really does (its classes and
  methods may be named in comments and tests, for example `java: RenderGuardian.doRender`), not
  against a comment in this code. The parity tests (`mc-parity`, the `*_java_parity` and
  `*_oracle` tests) encode this; extend them when you change behaviour. Two differences in what
  the client sends are deliberate. The client brand: after joining a server the client always
  sends `Aero` in the `MC|Brand` message where vanilla sends `vanilla`. And typed chat (the chat
  box, the sleep screen's, macros) is sent up to 256
  characters, always, where vanilla cuts at 100; a server's `run_command` click keeps vanilla's
  100.
- **No regressions in ms/frame.** Performance is measured in milliseconds per frame, not FPS. A
  change on a hot path (the frame loop, meshing, entity rendering, packet handling) must not raise
  the mean, the p99 or input-to-present; show a before/after measurement (section 8). A feature
  that is switched off costs nothing: no per-frame allocation, locking, hashing or string
  formatting on hot paths. Never undo an existing rendering or performance optimisation (batched
  and indexed entity rendering, mesh snapshots and terrain queues, shared atlases, culling, caches,
  pipelined submit) to make a change easier.
- **Self-referencing only.** Code, comments, strings, docs and test names describe this project in
  its own terms. Do not name, link to or compare with any third-party game client, mod, launcher or
  project. Two exceptions: references the code needs to work (for example a folder name that
  resource packs use) and licence notices, which stay word for word.
- **Privacy and secrets.** No account names, e-mail addresses, player names, server names from
  recordings or absolute local paths in code, comments, strings or files the client writes. Never
  log, print, store in plain text or put in a crash report an access token, refresh token or other
  secret; tests use obviously fake values. Tests find their files relative to
  `env!("CARGO_MANIFEST_DIR")` or through an environment variable.
- **Settings live in the mod catalog** (`mc-rs/crates/mc-mods`). An option id is a key in the
  player's settings file: never rename or reuse one. Labels use US spelling and sentence case.
- **`debug_assert!` is compiled out of release builds.** Never put a call with an effect inside
  one; bind the result first.
- **No Minecraft files and no Java source** in the repository (`.gitignore` blocks `*.java`,
  `*.class` and `*.jar`; the packaging check rejects Minecraft files in a package).

## 2. Pull request checklist

- `cargo fmt --all --check` is clean.
- `cargo clippy -p <crate> --all-targets -- -D warnings` is clean for every crate you touched.
- `cargo test -p <crate>` passes for every crate you touched, with your Minecraft 1.8.9 files in
  place (section 4). A test you did not run is not evidence.
- Behaviour changes come with a test against vanilla's behaviour (cite the vanilla class and
  method). Visual changes come with before/after pictures.
- Hot-path changes come with an A/B measurement in ms/frame (section 8).
- No third-party names, no personal data, no Minecraft files (section 1).
- Anything taken from elsewhere (code, shaders, fonts, images, sounds) says where it comes from
  and under which licence (section 17).
- The description says what you changed, why, and how you verified it.

Security problems are not reported in pull requests or issues: see section 18.

## 3. Prerequisites

| You need | For | Notes |
|---|---|---|
| Windows 10/11 x64 and a GPU with a Vulkan or DirectX 12 driver | everything | wgpu chooses the backend |
| [Rust](https://rustup.rs) via rustup, target `x86_64-pc-windows-msvc` | everything | `mc-rs\rust-toolchain.toml` selects the stable toolchain with rustfmt and clippy; rustup installs it on the first `cargo` command |
| Visual Studio 2022 Build Tools, workload **Desktop development with C++** | everything | the MSVC linker, and the C compiler for the `mimalloc` allocator |
| `rustup component add llvm-tools-preview` | PGO builds | provides `llvm-profdata`, which merges the training profiles |
| Python 3.10 or newer on PATH as `python` | PGO builds, the zip's licence notices, the benchmark tool | standard library only |
| [Inno Setup 6](https://jrsoftware.org/isinfo.php), 6.6 or newer | the installer (section 11) | optional: without it packaging builds the zip only |
| A Minecraft: Java Edition account | playing | you sign in on the client's Accounts screen (section 10) |
| Minecraft 1.8.9 installed with the official launcher and started once | the 1.8.9 files for playing, PGO training and tests | any folder holding the 1.8.9 jar and assets works too (section 4) |
| Git, and roughly 20 GB of free disk | cloning and build folders | the PGO builds use their own target folders |

Paths with spaces are fine: the scripts pass compiler flags through `CARGO_ENCODED_RUSTFLAGS`,
which does not split on spaces the way `RUSTFLAGS` does.

`mc-rs\.cargo\config.toml` limits cargo to 4 parallel jobs to keep the PC usable during long
builds; raise `jobs` there if your machine has headroom. The same file sets `MC_NO_AUDIO=1` for
every process cargo starts (tests, `cargo run`), so test clients stay silent. Set `MC_NO_AUDIO=0`
for a run that should play sound.

## 4. Your Minecraft 1.8.9 files

This repository contains no Minecraft files. Three things need them on disk:

- **Development runs** started without `--versionJar`/`--assetsDir`. The client then looks in
  `MC_JARS_DIR`, or else in `MCP-919-main\jars\` (or `jars\`) at the repository root.
- **The PGO training runs** of `install-mc-rs.ps1` and `package-aero-client.ps1`. Without the files
  they would train on untextured scenes, so both scripts stop before building unless they find the
  1.8.9 jar (`-SkipVanillaCheck` overrides that).
- **The tests** that load real vanilla models, textures, sounds and NBT (section 7).

After installing Minecraft 1.8.9 with the official launcher and starting it once, run:

```powershell
.\setup-vanilla-files.ps1
```

It copies, from the official launcher's folder (`%APPDATA%\.minecraft`) unless you name another
game folder with `-MinecraftDir` (any folder that holds the 1.8.9 jar under `versions\1.8.9\` or
`libraries\com\mojang\minecraft\1.8.9\`, and `assets\`) or the files themselves:

| Into `MCP-919-main\jars\` | From |
|---|---|
| `versions\1.8.9\1.8.9.jar` | `.minecraft\versions\1.8.9\1.8.9.jar`, or the jar given with `-JarPath` |
| `assets\indexes\1.8.json` and the ~720 objects it lists (about 110 MB) | `.minecraft\assets\`, or the folder given with `-AssetsDir` |
| `servers.dat`, `saves\MCP Development World\level.dat` | embedded in the script: two tiny reference NBT files (a one-entry server list and an empty development world) that the `mc-nbt`/`mc-ui` golden tests compare byte for byte |

It takes a few seconds and skips files that are already there, so running it again is harmless.
Options: `-MinecraftDir`, `-JarPath`, `-AssetsDir`, `-Destination`.

`MCP-919-main\` is git-ignored. It is named after the local reference workspace (the decompiled
vanilla 1.8.9 source) the port was developed against, which held the same files at the same place.
Some tests look only there, so keep the default destination, and never commit anything from it.
Playing does not need this step: `launch-mc-rs.ps1` finds the same files itself (section 6), and
uses this copy when there is one.

## 5. Building

| Build | Command | Result | Time |
|---|---|---|---|
| Development build | `cd mc-rs`, then `cargo build -p mc-client` | `mc-rs\target\debug\mc-client.exe` | a few minutes the first time |
| Optimised, without PGO | `cd mc-rs`, then `cargo build --profile dist -p mc-client` | `mc-rs\target\dist\mc-client.exe` | longer (fat LTO) |
| Optimised for A/B timing | `cd mc-rs`, then `cargo build --profile perf-ab -p mc-client` | `mc-rs\target\perf-ab\mc-client.exe` | faster than `dist` (thin LTO, incremental) |
| **Installed client (PGO)** | `.\install-mc-rs.ps1` | `mc-rs\target\release\mc-client.exe` and `mc-client.install.json` | about 25 min |
| Portable zip for other players | `.\package-aero-client.ps1` | `release\Aero Client.zip` | about 35 min |

The `dev` profile compiles the workspace at `opt-level = 1` and all dependencies at
`opt-level = 3`, because the mesher and physics are unusable unoptimised. The first build compiles
several hundred crates, including wgpu.

Two rules matter more than anything else here:

- **Never run `cargo build --release -p mc-client`.** `mc-rs\target\release\mc-client.exe` is
  where `install-mc-rs.ps1` puts the PGO client, and a plain release build overwrites it with a
  slower one (PGO measured 7.8% less frame time in the SkyWars benchmark, with lower input
  latency). For tests and benchmarks use `--profile dist`, `--profile perf-ab` or your own
  `CARGO_TARGET_DIR`.
- **A PGO profile only fits the code it was trained on.** After any code change, run
  `install-mc-rs.ps1` again before relying on the client's performance. `launch-mc-rs.ps1` warns on
  every launch when the code is newer than the installed client, when the exe is not the one
  `install-mc-rs.ps1` installed, or when there is no install stamp. The warning never blocks the
  launch.

### The optimised client (PGO)

```powershell
.\install-mc-rs.ps1
```

Close the game first: the script refuses to run while `mc-client` is running. The PC does not need
to be idle. The steps:

| Step | What happens | Where |
|---|---|---|
| check | the 1.8.9 jar is where the training runs will look (section 4) | |
| 1 | instrumented build (`-Cprofile-generate`) of `mc-client` and the `skywars_bench` example | `mc-rs\optimization\scratch\round2-2026-11-05\tgt\pgo-gen` |
| 2 | training: the SkyWars benchmark (pipelined and serial submit), dense terrain with a fixed and an orbiting camera, and an entity scene. Windows open in the background and close by themselves. | `...\round2-2026-11-05\results\pgo-train-<time>` |
| 3 | merge the profiles with `llvm-profdata` | `...\round2-2026-11-05\pgo\merged.profdata` |
| 4 | optimised build (`-Cprofile-use`, `dist` profile: fat LTO, one codegen unit) | `...\round2-2026-11-05\tgt\pgo-use` |
| 5 | smoke test: a 600-frame replay, muted, in a background window | a temporary folder |
| 6 | back up the previous client, install the new one, write the stamp | `mc-rs\optimization\installed-backups\`, `mc-rs\target\release\` |

All of those folders are git-ignored. The training inputs are committed: the plans in
`mc-rs\optimization\scratch\round2-2026-11-05\plans\` (they name files as `{MC_RS}\...`, which
`pgo_train.py` expands), the replay captures they name, and `mc-rs\optimization\quick_ab.py`, which
`pgo_train.py` imports for the benchmark's base options.

Options: `-Prebuilt <dir>` installs a folder that already holds a PGO-built `mc-client.exe` (for
example the output of `bash mc-rs/optimization/scratch/round2-2026-11-05/tools/pgo_build.sh <dir>`);
it is refused if any source file is newer than that exe. `-SkipVanillaCheck` builds even without
the 1.8.9 files (a less representative profile). The stamp `mc-rs\target\release\mc-client.install.json`
records the exe's hash and the newest source file at build time; `launch-mc-rs.ps1` checks both.

## 6. Running and playing

- **On any server:** `.\launch-mc-rs.ps1` starts the installed client with your Minecraft 1.8.9
  files; `.\launch-mc-rs.ps1 -Build debug` starts your development build instead. Sign in on the
  client's Accounts screen (section 10).
- **Offline-mode servers only (development):**
  `cargo run -p mc-client -- --offline DevName --server localhost --port 25565 --gameDir <scratch folder>`.
  The `offline` feature is a development login with no Microsoft account. It is on by default in
  development builds and compiled out of the portable zip.
- **Replay a recorded session (no server):**
  `cargo run -p mc-client -- --replay crates\mc-parity\captures\spigot-1.8.8-perfworld-vd16.mccap --offline BenchBot --frames 2000 --gameDir <scratch folder>`.

Always pass a scratch `--gameDir` for experiments, or the client writes `options.txt` and its other
settings files into the current folder. `launch-mc-rs.ps1` uses `mc-rs\` as the game directory, so
your settings, logs, screenshots and resource packs live there (all git-ignored).

`launch-mc-rs.ps1` gives the client its game directory and the Minecraft 1.8.9 files, nothing
else: it reads no account and passes no sign-in (section 10). It takes the files from, first found
wins, `-JarPath`/`-AssetsDir`, the copy `setup-vanilla-files.ps1` made in `MCP-919-main\jars\`,
then the official launcher's folder (`%APPDATA%\.minecraft`). Options:

| Option | Meaning |
|---|---|
| `-Server <host> [-Port <n>]` | join a server straight away |
| `-Build debug` | start the development build (`target\debug`) instead |
| `-ClientPath <exe>` | start a specific exe |
| `-MinecraftDir <dir>` | look for the 1.8.9 files in this game folder only (the jar under `versions\1.8.9\` or `libraries\com\mojang\minecraft\1.8.9\`, and `assets\`) |
| `-JarPath <jar>`, `-AssetsDir <dir>` | the 1.8.9 client jar, and the assets folder holding `indexes\1.8.json`, given directly |
| `-CheckOnly` | check the client and the 1.8.9 files, print where they are, then stop |

The launcher turns on pipelined frame submission (`MC_PIPELINED_SUBMIT=1`) unless you set it
yourself; set `MC_PIPELINED_SUBMIT=0` first for the serial loop. In game, **Right Shift** opens the
mod menu (rebind it in the menu's Keybinds view: Menu, "Open and close this menu").

## 7. Tests, formatting and lints

```powershell
cd mc-rs
cargo fmt --all --check
cargo clippy -p <crate> --all-targets -- -D warnings
cargo test -p <crate>
```

- `cargo clippy --workspace --all-targets -- -D warnings` checks everything at once. It currently
  stops on four known findings in test and experiment code (`mc-render`'s
  `entity/render/indexed_experiment.rs`, `mc-ui`'s `tests/x6_nametags.rs`); add `--keep-going` to
  see every crate's result.
- **Run `setup-vanilla-files.ps1` first** (section 4). Tests that need the vanilla files *fail*
  rather than skip when they are missing, on purpose: a skipping test passes and proves nothing.
  Most of them honour `MC_JARS_DIR`, but a few only look at `MCP-919-main\jars\`, which is why the
  setup script writes there. Without the files, `mc-assets`' vanilla tests and three golden-file
  tests of `mc-nbt` fail, and most of `mc-client`'s tests need them too; every other crate's
  tests pass without them (CI runs exactly those, section 13).
- Tests that render need a GPU adapter. Without one (for example on a CI runner) they print a
  `skip:` line and pass.
- No test talks to a real server or to an online service: network tests run against small local
  stand-ins (a fake game server, a fake session or release server on loopback).
- `cargo test -p mc-parity --test wire_c2s` checks every serverbound packet against bytes the real
  1.8.9 client wrote, kept in `parity\wire\fixtures\` (outside `mc-rs`; the test finds it by
  walking up from the crate). `parity\tools\wire_capture.sh` is the record of how they were made.
- The `mc-client` test binaries are large; the `test` profile in `mc-rs\Cargo.toml` turns off
  debug info to keep linking fast. Run one test file with `cargo test -p mc-client --test <name>`.
- The Python tools in `mc-rs\optimization\` have their own tests:
  `cd mc-rs\optimization; python -m unittest test_quick_ab_window test_benchmark_lease test_frame_metrics`.

## 8. Measuring performance

Report milliseconds per frame (mean and p99), never FPS, and compare two builds of the same scene:

1. Build the control and the candidate with `--profile perf-ab` (or `dist`), each in its own
   `CARGO_TARGET_DIR`, including the benchmark example:
   `cargo build --profile perf-ab -p mc-client --bin mc-client --example skywars_bench`.
2. Write a plan: copy `mc-rs\optimization\scratch\round2-2026-11-05\plans\pm-latency-sw.json`,
   replace every `{MC_RS}` with the absolute path of your `mc-rs` folder, and point its `A`
   (control) and `B` (candidate) entries at the two `skywars_bench.exe` files. The scene is
   lockstep: frame N is the same scene in every build, whatever its frame rate.
3. Run it: `python mc-rs\optimization\quick_ab.py run --plan <plan.json> --output <absolute dir>`.
   It runs A and B in ABBA order (an A/A noise baseline when both are the same exe) and prints
   ms/frame with 95% intervals; `quick_ab.py analyze --output <dir>` reprints the result.

Optimised builds have no vanilla-jar path compiled in: set `MC_JARS_DIR` (section 4) or pass
`--versionJar`. Close other heavy programs while measuring; timings taken while the PC compiles are
noise. For a single run, `MC_FRAME_TRACE=<file.csv>` makes the client write per-frame timings, and
`python mc-rs\optimization\frame_metrics.py <file.csv>` summarises them.

## 9. Reference data and the Java generators

The port is checked against what the real 1.8.9 client does, captured by small Java programs that
ran the decompiled Minecraft classes of a local reference workspace. Several of them transcribe
Minecraft code, so **no `.java` file is in this repository**. Everything they produced is
committed, so building and testing need no Java:

| Data | Made by |
|---|---|
| `mc-rs/crates/mc-math/src/tables/*.bin`, `mc-math/tests/fixtures/*.bin` | `mc-math/javaref/generate.ps1` |
| `mc-rs/crates/mc-physics/tests/fixtures/*.bin` | `mc-physics/javaref/generate*.ps1` |
| `mc-rs/crates/mc-world/tests/fixtures/{light,placement}/*` and the biome tables | `mc-world/javaref/generate*.ps1` |
| `mc-rs/crates/mc-data/extractor/out/*` | `mc-data/extractor/run.ps1` and `run-debug-grid.ps1` |
| `mc-rs/crates/mc-data/src/generated/*.rs` | `mc-data/extractor/codegen.py` and `generate_debug_states.py` from `out/` (Python, runnable here) |
| `mc-rs/crates/mc-client/javaref/*.txt`, the Java traces in `mc-client/tests/fixtures/`, the tables in the `*_java_parity` and `*_oracle` tests | the `generate-*.ps1` scripts in `mc-client/javaref/`, `mc-render/javaref/` and `mc-assets/javaref/`, and `mc-render/tests/actual_state_oracle/run_oracle.sh` |
| `mc-rs/crates/mc-ui/tests/fixtures/{achievements,hud_debug}_golden.json` | `mc-ui/tests/javaref/gen_*_golden.py` (Python re-derivations, runnable here) |
| the panorama tables in `mc-ui`'s `menu_panorama` test | `mc-ui/tests/javaref/gen_panorama_golden.ps1` |
| `parity/wire/fixtures/*.bin` | `parity/tools/wire_capture.sh` |
| `mc-rs/crates/mc-nbt/tests/data/java_writeutf.nbt` | a JVM's `DataOutputStream.writeUTF` |

The `generate*.ps1` and `*.sh` scripts are kept as the record of how each data set was made; the
tests that read the data name them. Running one needs its Java source, which is not published, and
the decompiled, recompiled vanilla 1.8.9 workspace at `MCP-919-main\` (git-ignored). The Python
generators run from the committed data alone.

## 10. Sign-in

Players sign in inside the client, on its own **Accounts** screen: the account corner at the top
right of the title screen, Options > Accounts..., or the mod menu's Aero Title Screen > Accounts
(so it also works with the vanilla title screen). **Add account** > **Sign in with Microsoft**
starts a device-code sign-in: the screen shows a code and an **Open page** button, the player
enters the code on Microsoft's own page in their browser, and the client then signs in to Xbox
Live and Minecraft's services and checks that the account owns Minecraft: Java Edition. The
password never passes through the client. An added account whose Minecraft token is about to run
out is refreshed before the next server join. At start the client uses the account chosen last
time; otherwise a session passed on its command line (below); otherwise the account used most
recently.

**Builds.** The Microsoft sign-in needs an Azure application (client) ID compiled in:
`AERO_MSA_CLIENT_ID` in the environment of `cargo build` (it is never written in the source).
Without it, "Add account" says that sign-in is not set up in this build. For development,
`MC_RS_MSA_CLIENT_ID=<id>` (or `{"client_id": "<id>"}` in `auth.json` in the config folder)
overrides the built-in ID at run time. Minecraft's services accept an Azure application only after
Mojang has approved it; until then sign-in stops at "Invalid app registration" (HTTP 403), and the
client says so.

**A session from a launcher.** Like the vanilla client, the client accepts a session on its command
line, the way a launcher starts vanilla 1.8.9: `--username`, `--uuid` and `--accessToken`, all
three (`--userType` is accepted and ignored). The Accounts screen lists it as "From the launcher";
it is used as given and never saved. Nothing in this repository reads another program's accounts
or tokens: the client takes a session only from its command line and its own Accounts screen.
`launch-mc-rs.ps1` passes no session at all, and neither does the packaged client when it starts
itself (section 11): it only finds the 1.8.9 files.

**Saved sign-ins** live in `accounts.json` in the per-user config folder (`%APPDATA%\mc-rs\config`
on Windows; `MC_RS_CONFIG_DIR` overrides it), never in the game folder that players copy and share.
The tokens in it are sealed with Windows DPAPI for the current user (on other platforms with
AES-256-GCM and a key file next to it). Only the account in use is unsealed, tokens are never
written to logs, and removing an account deletes its saved sign-in.

## 11. Packaging

```powershell
.\package-aero-client.ps1          # about 35 minutes; -SkipBuild re-zips the last build
```

It produces `release\Aero Client.zip` for players who do not build anything: `Aero Client.exe`,
which players double-click, a plain-language README, and the client, `app\mc-client.exe`.
`Aero Client.exe` is `aero-start` from the `mc-shell` crate: it only starts `app\mc-client.exe`.
The client, started with no arguments next to its updater marker `app\aero-install.json`, starts
itself the way a player expects: it finds the player's own Minecraft 1.8.9 files (the folder
chosen last time, then the official launcher's, else it asks once with a folder picker and keeps
the answer for the Windows user), keeps its logs in `game\logs`, and shows a plain message when
something goes wrong. Players sign in on the client's Accounts screen. The package holds no
script of any kind (no PowerShell, no `.cmd` or `.bat`). Build it with `AERO_MSA_CLIENT_ID` set
(section 10): the zip's players can sign in only there, so a zip built without it cannot sign in
at all, and the script warns about it. Compared with the client
`install-mc-rs.ps1` installs:

- it is built with `--no-default-features --features mimalloc,windowed`, so the development-only
  offline login is compiled out, and the client is a windowed program (Windows' GUI subsystem):
  no console window opens beside the game;
- the C runtime is linked statically (`-Ctarget-feature=+crt-static`), so players need no Visual
  C++ Redistributable;
- it uses its own build folders (`tgt\pkg-gen`, `tgt\pkg-use`, and `tgt\pkg-starter` for
  `Aero Client.exe`, which is built the same way without PGO) and profile, and never touches your
  installed client or settings.

**No Minecraft files in anything distributed.** A package ships this project's programs, their
licences and notices; the client finds the player's own Minecraft installation. Never the 1.8.9
jar, its assets or anything extracted from them. `mc-rs\packaging\scripts\verify-payload.ps1`
enforces this: it fails the build if anything that looks like a Minecraft file is in the package,
and its own tests (`verify-payload.tests.ps1`) plant such a file to prove it rejects it. The script
also has it check that the package holds every file players and the updater need, and no script
(`.ps1`, `.cmd`, `.bat`). Before zipping, the script checks that both programs are windowed,
carry the logo as their icon (their build scripts compile it with the Windows SDK's `rc.exe`,
which the MSVC build tools install; without it cargo only warns), import no Visual C++ runtime DLL
and name no folder, account or address of the PC that built them, that the client contains no
offline login, and that every library `Aero Client.exe` links is listed in the notices; then it
smoke-tests a replay and the title screen.

`app\LICENSE.txt` in the zip is the GPL-3.0 text (copied from `mc-rs\LICENSE`), and `app\licenses\`
holds the font licence and `THIRD-PARTY-NOTICES.txt`: every Rust library compiled into that exe with
its licence and notice files (and, for the MPL-2.0 ones, where their source is), the font's
licence, and the notices of published code the client follows (such as the logarithm in
`mc-math`). `mc-rs\packaging\scripts\third_party_notices.py` writes it during packaging from the
lock file (offline), for exactly the zip's features and target.

**Optional code signing.** Without these options nothing changes. With one, the script signs
`Aero Client.exe` and `app\mc-client.exe` before zipping, with SHA-256 and an RFC 3161 timestamp,
so the signature stays valid after the certificate expires:

- `-SignThumbprint <thumbprint>`: a code signing certificate in `Cert:\CurrentUser\My` (on a
  hardware token, from a cloud signing service, or any other). Find the thumbprint with
  `Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert`.
- `-SignDlib <dll> -SignMetadata <json>`: Azure Artifact Signing (signtool with Microsoft's dlib and
  a metadata JSON, as Microsoft's setup guide describes).
- `-TimestampUrl <url>` overrides the timestamp server. The environment variables
  `AERO_SIGN_THUMBPRINT`, `AERO_SIGN_DLIB` and `AERO_SIGN_METADATA` do the same as the options.

Signing needs `signtool.exe`: install "Windows SDK Signing Tools for Desktop Apps" with the Visual
Studio Installer or the Windows SDK installer. Check a signed file with
`Get-AuthenticodeSignature '.\release\Aero Client\app\mc-client.exe'`. A signature shows who
published the file; Windows SmartScreen still builds reputation for it over time, so expect its
warning on new releases either way.

**The installer.** After the zip, the script builds `release\Aero-Client-Setup.exe` from the same
staged folder with Inno Setup 6's `ISCC.exe`: `$env:ISCC` when set, else Inno Setup 6 installed for
the current user (`%LOCALAPPDATA%\Programs\Inno Setup 6`) or for everyone (`Program Files
(x86)\Inno Setup 6`, where GitHub's Windows runners have it). Without it the script builds the zip
only and says so. Use 6.6 or newer: the script compiles with 6.3 or later, but its wizard pictures
need 6.5.2 and its dark mode (it follows Windows' setting) 6.6. The installer script is
`mc-rs\packaging\aero\installer\aero-client.iss`; `make-installer-assets.ps1` next to it draws the
program icon (`app\aero-client.ico`, which the zip carries too) and the wizard pictures from the
logo renders in `mc-rs\crates\mc-client\assets\icon` (add a size there rather than a picture
anywhere else). The installer:

- installs for the current Windows user, without an administrator prompt, into
  `%LOCALAPPDATA%\Programs\Aero Client` unless the player chooses another folder on its folder
  page, which shows on a first install only (an upgrade keeps the folder). The folder is always
  named `Aero Client`: the page adds that name to a typed folder (Browse adds it too), and Setup
  refuses any other (`/DIR=` on its command line), so its uninstaller only ever deletes inside
  such a folder. It must be one the player can change without administrator rights, because the
  in-game updater writes there: the page refuses Program Files, Windows, a network path and any
  folder where Setup cannot make a test folder. A folder that already holds a portable copy is
  fine: Setup replaces the program and keeps `game\`;
- installs exactly the zip's folder (`Aero Client.exe`, `README.txt`, `app\`), so the in-game
  updater updates an installed copy the way it updates an unzipped one, and removes the
  `Play Aero Client.cmd` that earlier packages had. Keep `app\aero-client.ico` in every release:
  the shortcuts use it as their icon, and an update replaces `app\` as a whole;
- adds "Aero Client" to the Start menu and, unless the player unticks it, the desktop (both start
  `app\mc-client.exe` with no arguments, so a pinned shortcut and the running game share one
  taskbar button), and offers to start the game on its last page;
- upgrades an existing installation in place (its `AppId` must never change), after closing a
  running client (it asks first);
- registers in Settings > Apps, where the client keeps the version in step after an in-game
  update (only for the entry whose `InstallLocation` is its own folder);
- uninstalls the program, what the updater added (`app\`, `update\`), the shortcuts and the
  Settings entry. It keeps the player's `game\` folder unless they agree to delete it (a silent
  uninstall always keeps it), and it never touches saved sign-ins (`%APPDATA%\mc-rs`), which
  portable copies on the same PC share.

With `-SignThumbprint` or `-SignDlib` the script signs Setup and its uninstaller as well.

**Release files.** `mc-rs\packaging\scripts\release-assets.ps1 -Key <signing key>` (or `-Unsigned`)
turns the zip and the installer into the files of a GitHub release, in `release\v<version>\`, and
prints the `gh release create` command that publishes them:

| File | What it is |
|---|---|
| `Aero-Client-Setup.exe` | the installer |
| `Aero-Client-Portable.zip` | the portable zip, byte for byte the versioned one |
| `Aero-Client-<version>-windows-x64.zip` | the portable zip under the name the in-game updater downloads |
| `Aero-Client-<version>-windows-x64.zip.sha256` and `.sig` | its SHA-256 and update signature (section 12) |
| `SHA256SUMS.txt` | the SHA-256 of the installer and both zips, in `sha256sum -c` format |

**Never rename `Aero-Client-Setup.exe` or `Aero-Client-Portable.zip`.** The website's download
buttons link `https://github.com/lettingtimetell/aero-client/releases/latest/download/Aero-Client-Setup.exe`
and `.../releases/latest/download/Aero-Client-Portable.zip`, which always download the newest
release's file of exactly that name. A release without them, or one published as a draft or a
prerelease (which `latest` skips), breaks those buttons.

`mc-rs\packaging\` also holds the earlier generic packaging scripts (`windows\build.ps1`,
`linux\build.sh`, `macos\build.sh`), which stage the client with a launcher that looks for the
platform's standard Minecraft folder (`MC_RS_MINECRAFT_DIR` overrides it). Only the Windows script
has ever been run.

## 12. Releases and updates

No release has been published yet. The client's updater (`mc-rs/crates/mc-update`) reads the
latest release of one GitHub repository (`mc_update::DEFAULT_REPO`; a build can name another with
`AERO_UPDATE_REPO=owner/name`) at most once per launch and every 6 hours, only in a packaged
portable install and while "Check for updates" is on in the Aero title screen's options. It skips
drafts and prereleases and compares the release tag (`v0.2.0`) with the client's version (the
workspace version in `mc-rs/Cargo.toml`) by semver.

A release carries the portable zip as `Aero-Client-<version>-windows-x64.zip`, with its SHA-256 in
`Aero-Client-<version>-windows-x64.zip.sha256` (`<hex>  <zip name>`, or GitHub's own asset
digest) and an Ed25519 signature in `Aero-Client-<version>-windows-x64.zip.sig` (base64, over
`aero-client-release-v1`, the asset name, the version and the SHA-256, one per line). A client built
with `AERO_UPDATE_PUBKEY` (the base64 public key) downloads an update, checks the size, the SHA-256
and the signature, unpacks it into a staging folder and installs it on the next restart, rolling
back if the new version fails to start. A client built without a key only tells the player that a
new version exists. Only a packaged portable install updates itself: `package-aero-client.ps1`
writes the version marker `app\aero-install.json` into the zip. A copy installed by
`Aero-Client-Setup.exe` is one too: the installer installs the same folder.

`aero-update-sign` makes the key pair and the release files; its file header
(`mc-rs/crates/mc-update/src/bin/aero-update-sign.rs`) has every step. In short, from `mc-rs`:

1. Once: `cargo run -p mc-update --bin aero-update-sign -- keygen --out <folder outside the repository>\aero-update-signing.pk8`.
   It prints the public key. Keep the private key offline and backed up: whoever holds it can
   sign an update every client accepts.
2. Set `AERO_UPDATE_PUBKEY` to the printed public key, set the release's version in
   `mc-rs/Cargo.toml`, and run `package-aero-client.ps1`.
3. `packaging\scripts\release-assets.ps1 -Key <...>.pk8` runs
   `aero-update-sign sign --key <...>.pk8 --zip "..\release\Aero Client.zip" --version <version>`,
   which writes the three updater files, checks them with `aero-update-sign verify` the way the
   client will, and adds the installer, the portable zip and `SHA256SUMS.txt` (section 11).
4. Publish a release tagged `v<version>` (not a draft, not a prerelease) with every file in
   `release\v<version>\`; the script prints the `gh release create` command.

## 13. CI

`.github/workflows/ci.yml` **only runs when started by hand** (Actions, then ci, then Run
workflow), on Windows runners, and has not been run on GitHub yet. Its jobs:

- **lint**: `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings`
  (red until the four known findings of section 7 are fixed);
- **test**: the tests that pass without Minecraft files (section 7), the parity suite, a wider
  round-trip fuzzing pass, the benchmark tool's Python tests and the payload guard's self-test;
- **package**: the generic packaging scripts for Windows, Linux and macOS (the Linux and macOS
  legs have never run anywhere, so they report without failing the workflow).

Add `push` and `pull_request` to its `on:` block once a run is green.

## 14. Environment variables

| Variable | Effect |
|---|---|
| `MC_JARS_DIR` | folder with `versions\1.8.9\1.8.9.jar` and `assets\` for development runs and most tests (default: `MCP-919-main\jars`, then `jars`, at the repository root) |
| `MC_NO_AUDIO` | `1` mutes the client (set for everything cargo runs by `mc-rs\.cargo\config.toml`) |
| `MC_PIPELINED_SUBMIT` | `1` encodes, submits and presents frames on a second thread (the launcher's default); `0` for the serial loop |
| `MC_BACKGROUND_WINDOW` | `1` opens the window without focus and ignores input (benchmarks and smoke tests) |
| `MC_FRAME_TRACE` | path of a per-frame timing CSV to write |
| `AERO_MSA_CLIENT_ID` | at build time: the Azure application ID compiled in for the client's own Microsoft sign-in (section 10) |
| `MC_RS_MSA_CLIENT_ID` | at run time: overrides that ID (development) |
| `MC_RS_CONFIG_DIR` | the per-user config folder (default `%APPDATA%\mc-rs\config`), where saved sign-ins live |
| `AERO_UPDATE_PUBKEY` | at build time: the updater's Ed25519 public key (base64); without it the updater only notifies (section 12) |
| `AERO_UPDATE_REPO` | at build time: `owner/name` of the repository whose releases the updater reads |
| `MC_RS_MINECRAFT_DIR` | Minecraft folder for the generic packaging launchers |
| `PROPTEST_CASES`, `MC_PARITY_FUZZ_ITERS` | widen the property tests and the parity fuzzing |

## 15. Display latency on NVIDIA GPUs

On NVIDIA GPUs, set the per-program option for `mc-client.exe` in the NVIDIA Control Panel:
**Vulkan/OpenGL present method = Prefer layered on DXGI Swapchain**. Frames then take Windows'
fastest presentation path ("Hardware: Independent Flip") instead of going through the desktop
compositor. In fullscreen play, input-to-photon latency measured 6.75 ms without it and 0.70 ms
with it. The client cannot detect the setting, so check it again after a driver update.

## 16. Troubleshooting

| Problem | Fix |
|---|---|
| `install-mc-rs.ps1`: "No Minecraft 1.8.9 jar where the PGO training runs look for it" | run `.\setup-vanilla-files.ps1`, or set `MC_JARS_DIR` |
| "llvm-profdata is missing" | `rustup component add llvm-tools-preview` |
| `python` not found during training | install Python 3 and tick "Add to PATH" |
| `link.exe` not found, or a C compiler error from `mimalloc` | install the Visual Studio Build Tools workload "Desktop development with C++" |
| "mc-client is running; close the game first" | close the game; the installer replaces the running exe |
| The launcher warns the installed client is out of date | run `.\install-mc-rs.ps1` again (any code change needs a new PGO build) |
| `launch-mc-rs.ps1`: "The Minecraft 1.8.9 jar was not found" or "The Minecraft 1.8 assets were not found" | install Minecraft 1.8.9 with the official launcher and start it once, or name the folder that holds the files with `-MinecraftDir` (section 6) |
| "You're not signed in" when joining a server | add your Microsoft account on the Accounts screen (section 10) |
| "Microsoft sign-in isn't set up in this build of Aero Client" | build with `AERO_MSA_CLIENT_ID` set, or set `MC_RS_MSA_CLIENT_ID` for a development run (section 10) |
| `mc-assets` tests fail with "cannot run: ... looked under ..." | run `.\setup-vanilla-files.ps1` |
| An optimised build exits at start | it has no vanilla-jar path compiled in: set `MC_JARS_DIR` or pass `--versionJar` |
| Builds make the PC unresponsive | lower `jobs` in `mc-rs\.cargo\config.toml` |
| `cargo fmt --all` fails with "The filename or extension is too long (os error 206)" | Windows' command-line limit (rustfmt gets every crate's target files in one command line): clone to a shorter path, or check one crate at a time with `cargo fmt -p <crate> --check` |
| Scripts are blocked by the execution policy | `powershell -ExecutionPolicy Bypass -File .\install-mc-rs.ps1`, or `launch-mc-rs.cmd` for playing |

## 17. Licensing of contributions

Aero Client is licensed under the GNU General Public License, version 3 or (at your option) any
later version (GPL-3.0-or-later). By sending a pull request you agree that your contribution is
licensed under the same terms and that you have the right to license it that way; you keep the
copyright in it (the project's notice reads "Copyright (C) 2026 pvps and contributors"). Code,
shaders, fonts, images or sounds taken from elsewhere are accepted only under a licence
compatible with GPL-3.0-or-later: say in the pull request where they come from and under which
licence, so their notice can be kept.

Write features from your own description of the behaviour you want: don't copy code, data tables,
wording or images from other game clients or mods.

## 18. Security problems

Report a security problem privately, as README.md describes in "Reporting security problems"
(GitHub's private vulnerability reporting on the repository's Security tab), never in a public
issue or pull request. A client that merely behaves unlike vanilla 1.8.9 is an ordinary bug: open
an issue for it.
