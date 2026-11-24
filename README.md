# Aero Client

A Minecraft 1.8.9 PvP client written in Rust: a native, multiplayer-only port of the 1.8.9 client
(protocol 47) that joins unmodified 1.8.9 servers, with a built-in mod menu. In the code it is
called **mc-rs**.

> **Experimental.** Aero Client aims to behave exactly like the original Minecraft 1.8.9 client,
> and parity tests check much of it, but we can't guarantee that it matches in every case.
> Servers decide which clients and mods they allow, and some check how clients behave: use your
> own discretion when you play on servers with it. You do so at your own risk, and we are not
> responsible for bans or any other action a server takes against your account.

> Not an official Minecraft product. Not approved by or associated with Mojang or Microsoft.
> This repository contains no Minecraft game files and no Minecraft Java source code. Building,
> testing and playing need your own Minecraft: Java Edition account and its 1.8.9 files.

## Download

Aero Client is in beta. Get it from **[aero-client.com](https://aero-client.com)** or from this
repository's [releases](https://github.com/lettingtimetell/aero-client/releases), in one of two
forms:

- `Aero-Client-Setup.exe`, the installer: installs Aero Client for your Windows account, with
  shortcuts on the desktop and in the Start menu, and needs no administrator rights;
- `Aero-Client-Portable.zip`, the portable version: unpack it into any folder and start
  `Play Aero Client`.

You need Windows 10 or 11 (64-bit) and a Microsoft account that owns Minecraft: Java Edition.
Start version 1.8.9 once from the official Minecraft Launcher, so that its files are on your PC,
then sign in on Aero Client's Accounts screen.

## Features

- A wgpu + winit renderer with pipelined frame submission, built with profile-guided
  optimisation (PGO)
- Behaviour checked against the original client by parity tests: maths, physics, lighting, the
  wire format and rendering
- A mod menu on Right Shift (rebindable) with search, profiles and pins: PvP and HUD mods (FPS,
  keystrokes, zoom, crosshair, hitboxes, nametags, markers and more)
- Optional graphics, each zero-cost when off: TAA/FXAA anti-aliasing, screen-space ambient
  occlusion, anisotropic filtering, per-pixel motion blur, menu blur, volumetric clouds and
  built-in shader effects
- Replays: record a session, play it back, export it as an MP4 video; the replay browser shows
  your recordings by folder and sorts them by date, name, server, length, size and more
- Animated title screens, and a now-playing widget for the music playing on your PC
- A lower frame rate while the game window is in the background or minimized, to save CPU
- Its own Microsoft sign-in (see [Signing in](#signing-in))
- Tells servers that it is Aero Client: when it joins a server it sends the client brand `Aero`
  where vanilla sends `vanilla`
- An update check against this repository's releases for the downloaded client (installed or
  portable); with a signing key compiled in, it downloads, verifies and installs signed updates

## Status

- **Experimental** (see the notice above). With a mod off, or a setting at its vanilla default,
  the client is meant to behave exactly like vanilla 1.8.9. A few client features are on by
  default and can each be switched off: the Aero title screen, a lower frame rate in the
  background, Borderless Fullscreen, Raw Mouse Input and Modern Keybinds (the last two change
  how your input is read), and three chat settings (chat kept between servers, a 1,000-line
  history, shadows under chat symbols). Two differences have no switch: the client brand above,
  and chat you type can be up to 256 characters (vanilla allows 100; a server that really runs
  1.8 disconnects you for a longer message). A new player's server list starts with one entry,
  Mega Skywars (`megaskywars.com`, a server run by the project's publisher), which can be edited
  or removed like any other, and the Aero title screen has a Discord button that opens the
  project's Discord invite in your browser.
- **Windows x86_64** is the only platform that has been built and played. The Linux and macOS
  packaging scripts in `mc-rs/packaging/` have never been run.
- Releases are built on a maintainer's PC with `package-aero-client.ps1` (a PGO build) and signed
  for the in-client updater; see CONTRIBUTING.md, "Packaging". CI (`.github/workflows/ci.yml`)
  runs only when started by hand; see CONTRIBUTING.md, "CI".

## Quick start (Windows)

Everything else (prerequisites, tests, the rules for pull requests, packaging, troubleshooting) is
in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

```powershell
# Once: copy your own Minecraft 1.8.9 files to where builds and tests look for them
.\setup-vanilla-files.ps1

# Build the optimised client from the current code and install it (about 25 minutes)
.\install-mc-rs.ps1

# Play
.\launch-mc-rs.ps1          # or double-click launch-mc-rs.cmd
```

For development: `cd mc-rs`, then `cargo build -p mc-client` (an unoptimised build) and
`cargo test -p <crate>`.

## Signing in

You sign in inside the client, with the Microsoft account that owns Minecraft: Java Edition. On
the title screen, click the account corner at the top right ("Not signed in"), or open
Options > Accounts..., then **Add account** and **Sign in with Microsoft**. The client shows a
code; you enter it on Microsoft's own page in your browser, and the client checks that the
account owns the game. Your password never passes through the client.

The sign-in is kept for your Windows user on this PC (sealed with Windows DPAPI in the per-user
config folder, never in the game folder that players copy and share), so the next start is
already signed in. The Accounts screen adds, switches and removes accounts; removing one deletes
its saved sign-in.

The scripts in this repository never read an account or a token: `launch-mc-rs.ps1` and the
portable zip's launcher only find the Minecraft 1.8.9 game files and start the client. Like the
vanilla client, the client also accepts a session that a launcher passes on its command line
(`--username`, `--uuid`, `--accessToken`, the way the official launcher starts 1.8.9); the
Accounts screen lists it as "From the launcher".

The Microsoft sign-in works only in builds made with an Azure application (client) ID that
Minecraft's services accept, compiled in from `AERO_MSA_CLIENT_ID` (CONTRIBUTING.md, "Sign-in").

## Repository layout

| Path | What it is |
|---|---|
| `mc-rs/` | The Rust workspace (`Cargo.toml`, `Cargo.lock`, the crates in `mc-rs/crates/`) |
| `setup-vanilla-files.ps1` | Copies your Minecraft 1.8.9 files into the git-ignored `MCP-919-main/jars/` |
| `install-mc-rs.ps1` | PGO build of the current code, installed as `mc-rs/target/release/mc-client.exe` |
| `launch-mc-rs.ps1`, `launch-mc-rs.cmd` | Starts the installed client |
| `package-aero-client.ps1` | Builds `release/Aero Client.zip`, the portable client for other players |
| `mc-rs/packaging/` | Templates for that zip, its licence-notice generator, the payload check, and the generic Windows/Linux/macOS packaging scripts |
| `mc-rs/optimization/` | The ms/frame benchmark tool (`quick_ab.py`), the PGO training inputs and the benchmark fixtures the tests replay |
| `parity/wire/` | Serverbound packets as the real 1.8.9 client writes them, read by the `mc-parity` wire tests |

### Crates

| Crate | Role |
|---|---|
| `mc-client` | The game binary: config, sign-in, menus, session lifecycle, mods |
| `mc-render` | wgpu renderer: window shell, terrain mesher, entities, particles, sky, post effects |
| `mc-ui` | Font rendering, widgets, screens and container UIs |
| `mc-world` | Client world: chunk storage, block access, lighting, entity arena |
| `mc-physics` | Player physics, collision and client-side prediction |
| `mc-protocol` | Protocol 47 packets, codec, encryption and compression |
| `mc-assets` | Asset index, resource packs, block models, blockstates, texture atlas |
| `mc-audio` | Positional audio and music |
| `mc-auth` | Microsoft-account authentication and the server-join handshake |
| `mc-data` | Static game tables: blocks, items, entities, biomes, potions |
| `mc-math` | Vectors, AABBs and bit-exact `MathHelper` arithmetic |
| `mc-nbt` | NBT read/write with Java `DataInput` semantics |
| `mc-mods` | Mod framework: catalog, options, settings store, HUD layout |
| `mc-media` | Now-playing information and media controls from the Windows media session |
| `mc-replay` | Replay files: the recording container, its writer and reader, clips and the replay library |
| `mc-video` | Video and audio encoding for replay exports (H.264/H.265 and AAC in MP4) |
| `mc-parity` | Vanilla parity harness and the session recorder: capture, replay and diff |
| `mc-update` | Update check, download, verification (SHA-256 and an Ed25519 signature) and staged install of the portable client |
| `mc-core` | Shared conventions: tracing setup, profiling scopes |

## Reporting security problems

Please don't report a security problem in a public issue. Use **Report a vulnerability** on this
repository's **Security** tab (GitHub's private vulnerability reporting), and include:

- what an attacker controls: a server you join, a file someone shares (a replay, a resource
  pack, a settings file), a release download, or something else;
- the steps or a file that show the problem, and the commit or version you tested;
- what happens: a crash or hang, a file written or read outside where it belongs, a program
  started, a token or other secret exposed, or an update accepted that should have been refused.

A client that merely behaves unlike vanilla 1.8.9 is an ordinary bug: open an issue for it.

## License

Copyright (C) 2026 pvps and contributors.

Aero Client is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version (`GPL-3.0-or-later`). The full text of version 3
is in [LICENSE](LICENSE).

Aero Client is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

### Third-party content

- The Inter typeface in `mc-rs/crates/mc-ui/assets/fonts/inter/`, embedded in the client for the
  title screen: SIL Open Font License 1.1 (`OFL.txt` and `LICENSE-Inter-4.1.txt` in that folder).
- Code that follows published code whose licence asks for its notice keeps that notice next to
  it: the logarithm in `mc-rs/crates/mc-math/src/math_helper.rs`, and the ambient occlusion
  shader's horizon search (`NOTICE-ambient-occlusion.txt` beside
  `mc-rs/crates/mc-render/src/post/ao.wgsl`, MIT).
- The Rust dependencies listed in `mc-rs/Cargo.lock` are downloaded by cargo, each under its own
  licence; every one is compatible with GPL-3.0-or-later. The portable zip carries their licence and
  notice texts, and the ones above, in `app\licenses\THIRD-PARTY-NOTICES.txt`; the GPL text itself
  is `app\LICENSE.txt`.

## Not in this repository

- Minecraft game files (the 1.8.9 jar, textures, sounds) and the decompiled Minecraft Java source
  (the local reference workspace the port was developed against).
- The Java reference generators that captured vanilla behaviour. The data they produced is
  committed, so building and testing need no Java.
- Build outputs, benchmark run archives, screenshots and personal settings.
- Development notes: plans, work logs, research notes and benchmark reports.
