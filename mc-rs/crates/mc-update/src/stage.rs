//! Extracting a verified release zip into its staging folder, safely.
//!
//! The zip's bytes are already verified (its SHA-256, and its signature when this build has a
//! key), so this is defence in depth against a bad package as much as against an attack: nothing
//! in the zip can land outside the staging folder, and nothing but plain files and folders is
//! created. zip 8.6's own `extract` helpers and its streaming reader are **not** used (they have
//! open advisories about link targets and names); entries are read one by one with
//! `ZipArchive::by_index`, in central-directory order, and each must pass all of these:
//!
//! - the crate's `enclosed_name()` must accept it, **and** this module's own check: printable
//!   ASCII only, `/` or `\` as separators, no leading separator, no `:` (drive letters, NTFS
//!   streams), no empty, `.` or `..` component, none ending in a dot or space, none of
//!   `< > " | ? *`, no Windows device name (`CON`, `NUL`, `COM1`...), bounded lengths;
//! - it must be under the top folder `Aero Client/`; only `app/...` and the [`ROOT_FILES`]
//!   (`Aero Client.exe`, `Play Aero Client.cmd`, `README.txt`) are extracted (anything else,
//!   `game/` included, is skipped and logged);
//! - no symbolic link and nothing but a plain file or folder (Unix mode bits checked too);
//! - no path twice (compared case-insensitively, as Windows does);
//! - at most [`Limits::max_entries`] entries; each file's bytes are **counted as read** against
//!   [`Limits::max_entry_bytes`] and the running total against [`Limits::max_total_bytes`] (the
//!   declared size is not trusted, and must match the count);
//! - files are created with `create_new` inside a staging folder this module creates empty.
//!
//! The result must contain `app/mc-client.exe` and `app/aero-install.json` naming this release's
//! version. The root files are optional: those a release has replace the
//! install's, the others stay as they are (the helper starts `app\mc-client.exe` after an update,
//! never a root file).

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::UpdateError;
use crate::install::{InstallMarker, APP_DIR, EXE_NAME, LAUNCHER_CMD, LAUNCHER_EXE, MARKER_FILE};

/// The release zip's top folder.
pub const TOP_DIR: &str = "Aero Client";

/// Files next to `app/` that an update replaces too: the starter, the older `.cmd`, the readme.
pub const ROOT_FILES: &[&str] = &[LAUNCHER_EXE, LAUNCHER_CMD, "README.txt"];

/// The longest entry name accepted, in bytes.
const MAX_NAME_BYTES: usize = 240;
/// The longest path component accepted.
const MAX_COMPONENT_BYTES: usize = 120;

/// Caps on what one release may extract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Entries in the zip.
    pub max_entries: usize,
    /// Bytes of one file.
    pub max_entry_bytes: u64,
    /// Bytes of all files.
    pub max_total_bytes: u64,
}

impl Default for Limits {
    /// Far above today's package (about 15 files, 50 MB unpacked), far below harm.
    fn default() -> Limits {
        Limits {
            max_entries: 4096,
            max_entry_bytes: 512 * 1024 * 1024,
            max_total_bytes: 1024 * 1024 * 1024,
        }
    }
}

/// A release extracted into a staging folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Staged {
    /// The extracted `Aero Client` folder.
    pub root: PathBuf,
    /// Files written.
    pub files: usize,
    /// Bytes written.
    pub bytes: u64,
    /// The SHA-256 of `app/mc-client.exe`.
    pub exe_sha256: String,
}

fn archive_error(message: impl Into<String>) -> UpdateError {
    UpdateError::Archive(message.into())
}

/// The checked components of an entry name (see the module docs), without a directory's
/// trailing separator.
pub fn checked_components(name: &str) -> Result<Vec<&str>, UpdateError> {
    let refuse = |why: &str| archive_error(format!("unsafe entry name ({why})"));
    if name.is_empty() || name.len() > MAX_NAME_BYTES {
        return Err(refuse("length"));
    }
    if !name.bytes().all(|byte| (0x20..0x7f).contains(&byte)) {
        return Err(refuse("not printable ASCII"));
    }
    if name.starts_with(['/', '\\']) {
        return Err(refuse("absolute"));
    }
    if name.contains(':') {
        return Err(refuse("drive or stream"));
    }
    let trimmed = name
        .strip_suffix('/')
        .or_else(|| name.strip_suffix('\\'))
        .unwrap_or(name);
    let parts: Vec<&str> = trimmed.split(['/', '\\']).collect();
    for part in &parts {
        check_component(part).map_err(refuse)?;
    }
    Ok(parts)
}

fn check_component(part: &str) -> Result<(), &'static str> {
    if part.is_empty() || part == "." || part == ".." {
        return Err("empty, . or ..");
    }
    if part.len() > MAX_COMPONENT_BYTES {
        return Err("component length");
    }
    if part.ends_with(['.', ' ']) || part.starts_with(' ') {
        return Err("leading or trailing dot or space");
    }
    if part.contains(['<', '>', '"', '|', '?', '*']) {
        return Err("reserved character");
    }
    let stem = part.split('.').next().unwrap_or(part).to_ascii_uppercase();
    let device = matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
    ) || ((stem.starts_with("COM") || stem.starts_with("LPT"))
        && stem.len() == 4
        && stem.as_bytes()[3].is_ascii_digit());
    if device {
        return Err("device name");
    }
    Ok(())
}

/// Where an entry goes: `None` to skip it, else its path below the staging folder.
fn destination(parts: &[&str], is_dir: bool) -> Result<Option<PathBuf>, UpdateError> {
    if parts.first() != Some(&TOP_DIR) {
        return Err(archive_error(format!(
            "an entry outside the {TOP_DIR} folder"
        )));
    }
    let keep = match parts {
        [_] => is_dir,
        [_, app, ..] if *app == APP_DIR => true,
        [_, file] => !is_dir && ROOT_FILES.contains(file),
        _ => false,
    };
    Ok(keep.then(|| parts.iter().collect()))
}

/// Copy `reader` into `out`, counting bytes as they are read; more than `budget` is an error.
fn copy_counted(
    reader: &mut impl Read,
    out: &mut impl Write,
    budget: u64,
    buffer: &mut [u8],
) -> Result<u64, UpdateError> {
    let mut copied = 0_u64;
    loop {
        let read = reader
            .read(buffer)
            .map_err(|error| archive_error(format!("reading an entry: {error}")))?;
        if read == 0 {
            return Ok(copied);
        }
        copied += read as u64;
        if copied > budget {
            return Err(UpdateError::Size(
                "an entry is larger than allowed".to_owned(),
            ));
        }
        out.write_all(&buffer[..read])
            .map_err(|error| UpdateError::io("writing a staged file", error))?;
    }
}

/// Extract the release zip at `zip_path` into `dest` (which must not exist; it is created) and
/// check the result is release `version`'s package. On any error the caller deletes `dest`.
pub fn stage_release(
    zip_path: &Path,
    dest: &Path,
    version: &str,
    limits: &Limits,
) -> Result<Staged, UpdateError> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| UpdateError::io("creating the staging folder", error))?;
    }
    fs::create_dir(dest).map_err(|error| UpdateError::io("creating the staging folder", error))?;
    let file =
        File::open(zip_path).map_err(|error| UpdateError::io("opening the download", error))?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file))
        .map_err(|error| archive_error(format!("not a zip: {error}")))?;
    if archive.len() > limits.max_entries {
        return Err(archive_error("too many entries"));
    }
    let mut seen = HashSet::new();
    let mut buffer = vec![0_u8; 1 << 16];
    let (mut files, mut total) = (0_usize, 0_u64);
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| archive_error(format!("entry {index}: {error}")))?;
        let name = entry.name().to_owned();
        if entry.enclosed_name().is_none() {
            return Err(archive_error("an entry escapes its folder"));
        }
        let parts = checked_components(&name)?;
        if entry.is_symlink() {
            return Err(archive_error("a link in the release"));
        }
        let is_dir = entry.is_dir();
        if !is_dir && !entry.is_file() {
            return Err(archive_error("an entry that is not a file or folder"));
        }
        if let Some(mode) = entry.unix_mode() {
            let kind = mode & 0o170_000;
            if kind != 0 && kind != 0o100_000 && kind != 0o040_000 {
                return Err(archive_error("an entry that is not a file or folder"));
            }
        }
        let Some(relative) = destination(&parts, is_dir)? else {
            tracing::info!(entry = %name, "release entry skipped (not part of the install)");
            continue;
        };
        if !seen.insert(parts.join("/").to_ascii_lowercase()) {
            return Err(archive_error("a path listed twice"));
        }
        let target = dest.join(&relative);
        if is_dir {
            fs::create_dir_all(&target)
                .map_err(|error| UpdateError::io("creating a staged folder", error))?;
            continue;
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| UpdateError::io("creating a staged folder", error))?;
        }
        let mut out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .map_err(|error| UpdateError::io("creating a staged file", error))?;
        let budget = limits
            .max_entry_bytes
            .min(limits.max_total_bytes.saturating_sub(total));
        let copied = copy_counted(&mut entry, &mut out, budget, &mut buffer)?;
        if copied != entry.size() {
            return Err(archive_error("an entry's size differs from its header"));
        }
        out.sync_all()
            .map_err(|error| UpdateError::io("writing a staged file", error))?;
        total += copied;
        files += 1;
    }
    let root = dest.join(TOP_DIR);
    check_package(&root, version)?;
    let exe_sha256 = crate::verify::sha256_file(&root.join(APP_DIR).join(EXE_NAME))
        .map_err(|error| UpdateError::io("hashing the staged client", error))?;
    Ok(Staged {
        root,
        files,
        bytes: total,
        exe_sha256,
    })
}

/// Whether the extracted `Aero Client` folder at `root` is release `version`'s package.
pub fn check_package(root: &Path, version: &str) -> Result<(), UpdateError> {
    let app = root.join(APP_DIR);
    for required in [app.join(EXE_NAME), app.join(MARKER_FILE)] {
        if !required.is_file() {
            return Err(archive_error(format!(
                "the release has no {}",
                required.strip_prefix(root).unwrap_or(&required).display()
            )));
        }
    }
    let marker = InstallMarker::read(&app.join(MARKER_FILE))
        .map_err(|_| archive_error("the release's install marker is unusable"))?;
    if marker.version != version {
        return Err(archive_error(format!(
            "the release's marker says {} for release {version}",
            marker.version
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{package_zip, write_zip, ZipItem};

    fn stage(zip: &[u8], version: &str) -> (tempfile::TempDir, Result<Staged, UpdateError>) {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("release.zip");
        fs::write(&zip_path, zip).unwrap();
        let dest = dir.path().join("staging").join(version);
        let result = stage_release(&zip_path, &dest, version, &Limits::default());
        (dir, result)
    }

    /// Every file under `dir`, relative, with `/`.
    fn listing(dir: &Path) -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(next) = stack.pop() {
            for entry in fs::read_dir(&next).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let relative = path.strip_prefix(dir).unwrap();
                    out.push(relative.to_string_lossy().replace('\\', "/"));
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn a_good_package_is_staged_with_only_the_install_files() {
        let zip = package_zip(
            "0.2.0",
            &[ZipItem::file("Aero Client/game/options.txt", b"x")],
        );
        let (dir, result) = stage(&zip, "0.2.0");
        let staged = result.expect("staged");
        assert_eq!(
            staged.root,
            dir.path().join("staging").join("0.2.0").join(TOP_DIR)
        );
        assert_eq!(
            listing(&dir.path().join("staging")),
            [
                "0.2.0/Aero Client/Aero Client.exe",
                "0.2.0/Aero Client/Play Aero Client.cmd",
                "0.2.0/Aero Client/README.txt",
                "0.2.0/Aero Client/app/aero-install.json",
                "0.2.0/Aero Client/app/licenses/NOTICE.txt",
                "0.2.0/Aero Client/app/mc-client.exe",
            ],
            "game/ is never extracted"
        );
        assert_eq!(
            staged.exe_sha256,
            crate::verify::sha256_file(&staged.root.join("app").join(EXE_NAME)).unwrap()
        );
        assert_eq!(staged.files, 6);
    }

    #[test]
    fn backslash_separators_from_windows_zips_work() {
        let items = [
            ZipItem::file("Aero Client\\app\\mc-client.exe", b"exe"),
            ZipItem::file(
                "Aero Client\\app\\aero-install.json",
                InstallMarker::new("0.2.0").to_json().as_bytes(),
            ),
            ZipItem::file("Aero Client\\Aero Client.exe", b"starter"),
            ZipItem::file("Aero Client\\Play Aero Client.cmd", b"cmd"),
        ];
        let (dir, result) = stage(&write_zip(&items), "0.2.0");
        let staged = result.expect("staged");
        assert_eq!(
            fs::read(staged.root.join("Aero Client.exe")).unwrap(),
            b"starter"
        );
        drop(dir);
    }

    /// A release needs its client and its marker; the root files are optional
    /// (an install keeps the ones a release doesn't bring), and an older release's
    /// `app\launcher.ps1` is just another file of `app\`.
    #[test]
    fn the_root_files_are_optional_and_no_launcher_script_is_needed() {
        let marker = InstallMarker::new("0.2.0").to_json();
        let app_only = [
            ZipItem::file("Aero Client/app/mc-client.exe", b"exe"),
            ZipItem::file("Aero Client/app/aero-install.json", marker.as_bytes()),
        ];
        let (dir, result) = stage(&write_zip(&app_only), "0.2.0");
        result.expect("app\\ alone is a release");
        assert_eq!(
            listing(&dir.path().join("staging")),
            [
                "0.2.0/Aero Client/app/aero-install.json",
                "0.2.0/Aero Client/app/mc-client.exe",
            ]
        );

        let with_script = package_zip(
            "0.2.0",
            &[ZipItem::file("Aero Client/app/launcher.ps1", b"# launcher")],
        );
        let (dir, result) = stage(&with_script, "0.2.0");
        let staged = result.expect("an older layout still stages");
        assert!(staged.root.join("app").join("launcher.ps1").is_file());
        assert!(staged.root.join("Aero Client.exe").is_file());
        drop(dir);

        // A starter anywhere but next to app\ is not one of the root files.
        let (dir, result) = stage(
            &package_zip(
                "0.2.0",
                &[ZipItem::file("Aero Client/extra/Aero Client.exe", b"x")],
            ),
            "0.2.0",
        );
        result.expect("staged");
        assert!(!dir.path().join("staging/0.2.0/Aero Client/extra").exists());
    }

    #[test]
    fn hostile_names_are_refused() {
        for name in [
            "../evil.txt",
            "Aero Client/../../evil.txt",
            "Aero Client/app/../../evil.txt",
            "Aero Client\\app\\..\\..\\evil.txt",
            "/Aero Client/app/x",
            "\\Aero Client\\app\\x",
            "C:/Windows/evil.dll",
            "C:evil.txt",
            "Aero Client/app/x.exe:stream",
            "Aero Client/app//x",
            "Aero Client/./app/x",
            "Aero Client/app/x.",
            "Aero Client/app/x ",
            "Aero Client/app/ x",
            "Aero Client/app/CON",
            "Aero Client/app/nul.txt",
            "Aero Client/app/com1.dll",
            "Aero Client/app/LPT9",
            "Aero Client/app/a<b",
            "Aero Client/app/a|b",
            "Aero Client/app/a?b",
            "Aero Client/app/caf\u{e9}.txt",
            "Aero Client/app/tab\there",
            "Other/app/mc-client.exe",
            "app/mc-client.exe",
        ] {
            let zip = package_zip("0.2.0", &[ZipItem::file(name, b"evil")]);
            let (dir, result) = stage(&zip, "0.2.0");
            assert!(result.is_err(), "{name:?} was accepted");
            assert!(!dir.path().join("evil.txt").exists());
            assert!(!dir.path().parent().unwrap().join("evil.txt").exists());
        }
    }

    #[test]
    fn links_and_duplicates_are_refused() {
        let zip = package_zip(
            "0.2.0",
            &[ZipItem::symlink("Aero Client/app/link", "../../x")],
        );
        let (_dir, result) = stage(&zip, "0.2.0");
        assert!(matches!(result, Err(UpdateError::Archive(_))), "{result:?}");

        let zip = package_zip(
            "0.2.0",
            &[ZipItem::file("Aero Client/app/MC-CLIENT.EXE", b"second")],
        );
        let (_dir, result) = stage(&zip, "0.2.0");
        assert!(result.is_err(), "the same file twice (case-insensitively)");
    }

    #[test]
    fn size_caps_count_the_bytes_actually_read() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("big.zip");
        let big = vec![0_u8; 300_000];
        fs::write(
            &zip_path,
            package_zip("0.2.0", &[ZipItem::file("Aero Client/app/big.bin", &big)]),
        )
        .unwrap();
        let small = Limits {
            max_entries: 100,
            max_entry_bytes: 100_000,
            max_total_bytes: 10_000_000,
        };
        let result = stage_release(&zip_path, &dir.path().join("a"), "0.2.0", &small);
        assert!(matches!(result, Err(UpdateError::Size(_))), "{result:?}");
        let total = Limits {
            max_entries: 100,
            max_entry_bytes: 10_000_000,
            max_total_bytes: 200_000,
        };
        let result = stage_release(&zip_path, &dir.path().join("b"), "0.2.0", &total);
        assert!(matches!(result, Err(UpdateError::Size(_))), "{result:?}");
        let few = Limits {
            max_entries: 3,
            ..Limits::default()
        };
        let result = stage_release(&zip_path, &dir.path().join("c"), "0.2.0", &few);
        assert!(matches!(result, Err(UpdateError::Archive(_))), "{result:?}");
        // The default limits take it.
        stage_release(
            &zip_path,
            &dir.path().join("d"),
            "0.2.0",
            &Limits::default(),
        )
        .expect("staged");
    }

    #[test]
    fn the_package_must_be_complete_and_for_this_version() {
        // The marker names another version.
        let zip = package_zip("0.2.0", &[]);
        let (_dir, result) = stage(&zip, "0.3.0");
        assert!(matches!(result, Err(UpdateError::Archive(_))), "{result:?}");
        // No client.
        let items = [
            ZipItem::file(
                "Aero Client/app/aero-install.json",
                InstallMarker::new("0.2.0").to_json().as_bytes(),
            ),
            ZipItem::file("Aero Client/Aero Client.exe", b"starter"),
            ZipItem::file("Aero Client/Play Aero Client.cmd", b"cmd"),
        ];
        let (_dir, result) = stage(&write_zip(&items), "0.2.0");
        assert!(matches!(result, Err(UpdateError::Archive(_))), "{result:?}");
        // No marker.
        let items = [
            ZipItem::file("Aero Client/app/mc-client.exe", b"exe"),
            ZipItem::file("Aero Client/Aero Client.exe", b"starter"),
        ];
        let (_dir, result) = stage(&write_zip(&items), "0.2.0");
        assert!(matches!(result, Err(UpdateError::Archive(_))), "{result:?}");
        // Not a zip at all.
        let (_dir, result) = stage(b"PK not really", "0.2.0");
        assert!(matches!(result, Err(UpdateError::Archive(_))), "{result:?}");
    }

    #[test]
    fn the_staging_folder_must_be_new() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("r.zip");
        fs::write(&zip_path, package_zip("0.2.0", &[])).unwrap();
        let dest = dir.path().join("dest");
        fs::create_dir(&dest).unwrap();
        assert!(stage_release(&zip_path, &dest, "0.2.0", &Limits::default()).is_err());
    }
}
