//! Sending this process's standard output and error to files.
//!
//! Rust's standard library asks Windows for the standard handles at every write, so once they
//! are replaced, `println!`, `eprintln!`, the log writer and a panic message all land in the new
//! files. Code in C libraries that writes through the C runtime's own descriptors is not
//! redirected (the runtime copied the handles when the process started).

use std::fs::File;
use std::io;

/// Make `out` this process's standard output and `err` its standard error, for the rest of its
/// life (the files are never closed).
///
/// # Errors
///
/// Windows refused (nothing changed when the first one fails), and on other platforms.
pub fn redirect(out: File, err: File) -> io::Result<()> {
    #[cfg(windows)]
    {
        crate::sys::set_std_handles(out, err)
    }
    #[cfg(not(windows))]
    {
        let _ = (out, err);
        Err(io::ErrorKind::Unsupported.into())
    }
}
