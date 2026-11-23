#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! The Aero Client updater: it finds a newer portable release on GitHub, downloads and verifies
//! it, stages it next to the running client, and swaps it in on restart, with a rollback when the
//! new version does not start.
//!
//! # Which builds update themselves
//!
//! Only a packaged portable install: `Aero Client\app\mc-client.exe` next to the marker
//! `app\aero-install.json` that the packaging script writes, with `Aero Client\Play Aero
//! Client.cmd` beside `app\` ([`install::InstallPaths::detect`]). Development builds and
//! `cargo run` have no marker and never check, download or swap anything.
//!
//! # The steps
//!
//! 1. **Check** ([`github`], [`net`]): `GET https://api.github.com/repos/<REPO>/releases/latest`
//!    in the background, at most once per launch and once per [`CHECK_INTERVAL`], with no
//!    credentials and a User-Agent naming the client only. Drafts and prereleases are skipped;
//!    the tag (`v0.2.0`) must be a newer version than this build's ([`version`]).
//! 2. **Download** ([`net`]), only when the player clicks Update, which only a build with a
//!    signing key offers (without one the client announces the release and links its page):
//!    HTTPS only, redirects only to GitHub's own hosts, and the asset's size in the release data
//!    is a hard cap.
//! 3. **Verify** ([`verify`]): the SHA-256 must equal the asset's `digest` in the release data
//!    (or a `.sha256` asset), and when this build has a public key compiled in
//!    (`AERO_UPDATE_PUBKEY`) an Ed25519 signature (`.sig` asset) over the asset's name, version
//!    and hash must verify. With a key compiled in, the signature is required.
//! 4. **Stage** ([`stage`]): the zip is extracted into `update\staging\<version>`, refusing
//!    absolute paths, `..`, drive letters, links, device names and oversized entries.
//! 5. **Apply on restart** ([`helper`], [`swap`]): the player clicks "Restart to update" on the
//!    title screen (never with a world loaded). A copy of the running client waits for it to
//!    exit, runs the new binary's self-test, swaps `app\` by renames (the old one is kept in
//!    `update\previous\`) and starts `Play Aero Client.cmd`. If the new version exits before
//!    its title screen shows, the old `app\` is put back and started instead.
//!
//! The player's `game\` folder is never read or written.
//!
//! # Release conventions
//!
//! A release is tagged `v<version>` and carries [`github::asset_name`]
//! (`Aero-Client-<version>-windows-x64.zip`), plus `<asset>.sha256` and, once a signing key
//! exists, `<asset>.sig`. The `aero-update-sign` tool in this crate (`src/bin/aero-update-sign.rs`)
//! writes all three from the packaged zip and makes the key pair; its header has the steps.

pub mod error;
pub mod github;
pub mod helper;
pub mod install;
pub mod net;
pub mod service;
pub mod stage;
pub mod state;
pub mod swap;
pub mod verify;
pub mod version;

#[cfg(test)]
mod tests;

use std::time::Duration;

pub use error::UpdateError;

/// This build's version: the workspace version.
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The product name, as the packaging marker and the release zip's top folder spell it.
pub const PRODUCT: &str = "Aero Client";

/// The repository releases come from, `owner/name`. To move it, change this one constant, or
/// build with `AERO_UPDATE_REPO=owner/name` set, which wins over it.
pub const DEFAULT_REPO: &str = "lettingtimetell/aero-client";

/// The repository this build reads releases from ([`DEFAULT_REPO`] unless the build set
/// `AERO_UPDATE_REPO`).
pub const REPO: &str = match option_env!("AERO_UPDATE_REPO") {
    Some(repo) => repo,
    None => DEFAULT_REPO,
};

/// The least time between two release checks, across launches.
pub const CHECK_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

/// Seconds since the Unix epoch now (0 if the clock is before it).
#[must_use]
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}
