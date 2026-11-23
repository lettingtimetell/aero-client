//! Locating the real 1.8.9 data, **without ever skipping**.
//!
//! # Why this module exists
//!
//! Five suites in this directory used to open with some spelling of
//!
//! ```ignore
//! let Some(jar) = jar() else {
//!     eprintln!("skip: no vanilla jar in this checkout");
//!     return;
//! };
//! ```
//!
//! A `#[test]` that returns early **passes**, and `cargo test` swallows the stdout/stderr of a
//! passing test. So on any machine where the jar was not exactly where the helper looked, the
//! suite reported a clean green run of tests that examined nothing — while claiming, by name,
//! to cover the whole asset pipeline (`vanilla_assets`, `s09_model_bake`,
//! `s09_particle_texture`, `s21_reinstall`, `s23_cache`).
//!
//! This is the same failure the sibling `codegen.py` bugs had. `validate_blockstates` derived
//! the repo root from its *output* directory and skipped; worse, `load_sounds` printed a
//! warning and wrote a **valid, empty** `sounds.rs`, so a regenerate on an assetless checkout
//! silently deleted the 246-entry sound registry and exited 0. A validator that cannot find
//! its input and passes anyway is worse than no validator, because it reads as a pass.
//!
//! So: [`locate_jar`] and [`locate_assets`] return `Err` with the path they looked at, and
//! [`vanilla_jar`]/[`assets_dir`] panic with it. Every suite that uses this module carries a
//! `..._fails_loudly_when_the_data_is_absent` negative control that drives the `Err` branch
//! directly, without touching process-wide environment, and *also* asserts the real data is
//! found — so the control cannot pass for the wrong reason.
//!
//! `tests/blockstate_coverage.rs` keeps its own private copy of this logic on purpose: it is
//! the oracle these five are measured against, and it must not be able to fail because a
//! shared test helper changed underneath it.
//!
//! # Overriding the location
//!
//! `MC_JARS_DIR` points at a `jars/` directory. It is honoured by every helper here, including
//! [`assets_dir`] — which the old `s09_particle_texture`/`s21_reinstall`/`s23_cache` helpers
//! did not do; they hard-coded three candidate paths and had no override at all.

#![allow(dead_code)] // each test binary uses a different subset of this module

use std::path::{Path, PathBuf};

/// `$MC_JARS_DIR`, or `<repo root>/MCP-919-main/jars`.
///
/// Unlike a `-> Option<PathBuf>`, this never reports "not found" by vanishing: the directory
/// may well not exist, and the caller's error message names it.
pub fn jars_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        return PathBuf::from(dir);
    }
    // crates/mc-assets -> crates -> mc-rs -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("CARGO_MANIFEST_DIR should have three ancestors")
        .join("MCP-919-main")
        .join("jars")
}

/// The vanilla jar under a `jars/` directory, or why it is not usable.
///
/// Split out from [`vanilla_jar`] so the negative controls can exercise the absent branch
/// without mutating process-wide environment (which would race the other tests in the binary).
pub fn locate_jar(jars: &Path) -> Result<PathBuf, String> {
    if !jars.is_dir() {
        return Err(format!("{} is not a directory", jars.display()));
    }
    let p = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    if !p.is_file() {
        return Err(format!("{} is not a file", p.display()));
    }
    Ok(p)
}

/// The unpacked `assets/` tree (the launcher's object store) under a `jars/` directory.
///
/// Presence is decided by `indexes/1.8.json`, not by the directory alone: an empty `assets/`
/// would otherwise satisfy the check and every test would then fail deep inside the index
/// parser with a much worse message.
pub fn locate_assets(jars: &Path) -> Result<PathBuf, String> {
    if !jars.is_dir() {
        return Err(format!("{} is not a directory", jars.display()));
    }
    let p = jars.join("assets");
    let index = p.join("indexes").join("1.8.json");
    if !index.is_file() {
        return Err(format!("{} is not a file", index.display()));
    }
    Ok(p)
}

fn explain(what: &str, why: String, jars: &Path) -> String {
    format!(
        "{what} cannot run: {why}\n\
         looked under {}\n\
         set MC_JARS_DIR to a jars/ directory containing versions/1.8.9/1.8.9.jar (and, for\n\
         the asset-index tests, assets/indexes/1.8.json).\n\
         this test does not skip on purpose -- it used to, and a skipping test passes, so the\n\
         suite reported a clean run of tests that examined nothing.",
        jars.display()
    )
}

/// The vanilla jar, or a panic naming every path it looked at.
#[track_caller]
pub fn vanilla_jar(what: &str) -> PathBuf {
    let jars = jars_dir();
    match locate_jar(&jars) {
        Ok(p) => p,
        Err(why) => panic!("{}", explain(what, why, &jars)),
    }
}

/// The unpacked `assets/` tree, or a panic naming every path it looked at.
#[track_caller]
pub fn assets_dir(what: &str) -> PathBuf {
    let jars = jars_dir();
    match locate_assets(&jars) {
        Ok(p) => p,
        Err(why) => panic!("{}", explain(what, why, &jars)),
    }
}

/// A directory that is guaranteed not to exist, for the negative controls.
pub fn definitely_missing() -> PathBuf {
    jars_dir().join("this-path-does-not-exist-MC_JARS_DIR-negative-control")
}
