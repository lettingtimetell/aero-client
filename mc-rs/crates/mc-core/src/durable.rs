//! Crash- and power-safe replacement of the small files a player cares about: `options.txt`, the
//! mod settings, profiles, waypoints and `servers.dat`.
//!
//! [`write_atomic`] writes the new contents to a sibling temporary file, flushes that file to the
//! disk ([`std::fs::File::sync_all`], `FlushFileBuffers` on Windows) and only then renames it over
//! the destination. A crash, a killed process or a power cut at any moment leaves either the
//! complete old file or the complete new one:
//!
//! * a plain [`std::fs::write`] truncates the destination before writing, so dying half-way
//!   leaves an empty or cut-off file;
//! * a rename without the flush can reach the disk before the data does, so a power cut right
//!   after it leaves a file of the right length full of zeros.
//!
//! The rename replaces the destination in one step on every platform the client runs on
//! (`MoveFileExW` with `MOVEFILE_REPLACE_EXISTING` on Windows, `rename(2)` on Unix); the old
//! file is never deleted first. Two behaviours of a plain write are kept on purpose: a
//! destination marked read-only is left alone (some players lock `options.txt` that way), and a
//! destination that is a symbolic link has its target replaced, not the link.
//!
//! Small files only: the whole contents are passed in, and the call blocks until the disk has them
//! (a few milliseconds on an SSD). Never call it per frame.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// How many times a failed rename is retried. On Windows an antivirus scanner or the search
/// indexer may hold the destination open for a moment, which fails the rename.
const RENAME_ATTEMPTS: u32 = 3;
/// The pause between those attempts.
const RENAME_RETRY_DELAY: Duration = Duration::from_millis(15);

/// `path` with `suffix` appended to its file name: `options.txt` + `.tmp` is `options.txt.tmp`.
#[must_use]
pub fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Replace `path` with `bytes`, atomically and durably, through `<path>.tmp`.
///
/// # Errors
///
/// The temporary file could not be written or flushed, the destination is marked read-only, or
/// the rename failed. The destination is untouched in every error case, and the temporary file is
/// removed.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let target = resolve_link(path);
    let temporary = sibling(&target, ".tmp");
    write_atomic_via(&target, &temporary, bytes)
}

/// [`write_atomic`] through an explicit temporary file, which must be on the same volume as
/// `path` (in practice: in the same folder). `servers.dat` uses vanilla's `servers.dat_tmp`.
///
/// # Errors
///
/// As [`write_atomic`].
pub fn write_atomic_via(path: &Path, temporary: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if is_read_only(path) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("{} is marked read-only", path.display()),
        ));
    }
    let result = write_and_flush(temporary, bytes).and_then(|()| rename_over(temporary, path));
    if result.is_err() {
        // Never leave a partial temporary behind: the next save starts clean either way.
        let _ = std::fs::remove_file(temporary);
    }
    result
}

/// Copy `path` to `<path><suffix>` so that a file the client could not read survives the next
/// save. Returns the copy's path, or `None` when there was nothing to copy or the copy failed
/// (the caller logs; a failed backup never stops the game).
#[must_use]
pub fn keep_aside(path: &Path, suffix: &str) -> Option<PathBuf> {
    let backup = sibling(path, suffix);
    match std::fs::copy(path, &backup) {
        Ok(_) => Some(backup),
        Err(_) => None,
    }
}

fn write_and_flush(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn rename_over(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut attempt = 1;
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => {
                sync_parent(to);
                return Ok(());
            }
            Err(error)
                if attempt < RENAME_ATTEMPTS && error.kind() != std::io::ErrorKind::NotFound =>
            {
                attempt += 1;
                std::thread::sleep(RENAME_RETRY_DELAY);
            }
            Err(error) => return Err(error),
        }
    }
}

/// Unix only: make the rename itself durable by flushing the folder. NTFS journals the rename
/// with the rest of its metadata, and Windows has no portable way to flush a folder handle.
fn sync_parent(path: &Path) {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if let Ok(folder) = std::fs::File::open(parent) {
            let _ = folder.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

fn is_read_only(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.permissions().readonly())
}

/// The file a symbolic link points at (a plain write follows the link; a rename would replace
/// it). Anything that is not a readable link is returned as it is.
fn resolve_link(path: &Path) -> PathBuf {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
        }
        _ => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mc-core-durable-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn writes_a_new_file_and_replaces_an_existing_one() {
        let dir = temp_dir("replace");
        let path = dir.join("options.txt");
        write_atomic(&path, b"first").expect("create");
        assert_eq!(std::fs::read(&path).expect("read"), b"first");
        write_atomic(&path, b"second, longer").expect("replace");
        assert_eq!(std::fs::read(&path).expect("read"), b"second, longer");
        write_atomic(&path, b"3").expect("shorter");
        assert_eq!(std::fs::read(&path).expect("read"), b"3");
        assert!(
            !sibling(&path, ".tmp").exists(),
            "the temporary is consumed by the rename"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The property the module exists for: when the new contents cannot be written, the old file
    /// is still there, complete. Negative control: the plain `fs::write` this replaces destroys
    /// the old contents before it can fail.
    #[test]
    fn a_failed_write_leaves_the_old_file_complete() {
        let dir = temp_dir("failed");
        let path = dir.join("servers.dat");
        std::fs::write(&path, b"the player's server list").expect("seed");
        // A folder where the temporary file has to go makes creating it fail.
        let blocker = sibling(&path, ".tmp");
        std::fs::create_dir_all(&blocker).expect("blocker");
        assert!(write_atomic(&path, b"new").is_err());
        assert_eq!(
            std::fs::read(&path).expect("read"),
            b"the player's server list"
        );
        std::fs::remove_dir_all(&blocker).expect("unblock");

        // Negative control: truncate-then-write loses everything when the write is cut short.
        {
            let file = std::fs::File::create(&path).expect("truncating open");
            drop(file); // "crash" right after the truncate
        }
        assert_eq!(std::fs::read(&path).expect("read").len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_temporary_from_an_earlier_crash_is_overwritten() {
        let dir = temp_dir("stale");
        let path = dir.join("mc-rs-mods.json");
        std::fs::write(sibling(&path, ".tmp"), b"half a file from a crash").expect("stale");
        write_atomic(&path, b"{}").expect("write");
        assert_eq!(std::fs::read(&path).expect("read"), b"{}");
        assert!(!sibling(&path, ".tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_read_only_destination_is_left_alone() {
        let dir = temp_dir("readonly");
        let path = dir.join("options.txt");
        std::fs::write(&path, b"locked by the player").expect("seed");
        let mut permissions = std::fs::metadata(&path).expect("meta").permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&path, permissions.clone()).expect("lock");

        let error = write_atomic(&path, b"overwritten").expect_err("read-only must refuse");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(std::fs::read(&path).expect("read"), b"locked by the player");
        assert!(!sibling(&path, ".tmp").exists());

        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        std::fs::set_permissions(&path, permissions).expect("unlock");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_explicit_temporary_name_is_used_and_consumed() {
        let dir = temp_dir("via");
        let path = dir.join("servers.dat");
        let temporary = dir.join("servers.dat_tmp");
        write_atomic_via(&path, &temporary, b"nbt").expect("write");
        assert_eq!(std::fs::read(&path).expect("read"), b"nbt");
        assert!(!temporary.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn keep_aside_copies_and_reports_the_backup() {
        let dir = temp_dir("aside");
        let path = dir.join("options.txt");
        assert_eq!(keep_aside(&path, ".bad"), None, "nothing to copy");
        std::fs::write(&path, b"\0\0\0").expect("seed");
        let backup = keep_aside(&path, ".bad").expect("copied");
        assert_eq!(backup, dir.join("options.txt.bad"));
        assert_eq!(std::fs::read(&backup).expect("read"), b"\0\0\0");
        assert!(path.exists(), "a copy, not a move");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
