#![warn(missing_docs)]
// Every `unsafe` call is in `sys` (Windows only), which allows it.
#![deny(unsafe_code)]

//! The Windows calls of Aero Client's packaged start, and the portable folder's starter.
//!
//! `mc-client` is `#![forbid(unsafe_code)]`, so what its packaged start needs from Windows beyond
//! the standard library is here, behind safe functions:
//!
//! - [`dialog`]: plain pop-ups (`MessageBoxW`) and the shell's folder picker (`IFileOpenDialog`);
//! - [`registry`]: one string value of an existing key under `HKEY_CURRENT_USER` (the Settings >
//!   Apps entry `Aero-Client-Setup.exe` writes);
//! - [`stdio`]: this process's standard output and error, sent to files.
//!
//! [`start`] is what `Aero Client.exe` does (the `aero-start` binary, which the packaging copies
//! to the portable folder's root): start `app\mc-client.exe` with its own arguments and exit, or
//! say plainly why it can't.
//!
//! Other platforms build the same API: messages go to standard error and everything else answers
//! that it isn't available. Only the Windows package uses any of it.

pub mod dialog;
pub mod registry;
pub mod start;
pub mod stdio;

#[cfg(windows)]
mod sys;
