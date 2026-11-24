//! Where a portable install keeps what, and how a build knows it is one.
//!
//! ```text
//! Aero Client\
//!   Play Aero Client.cmd      what the player double-clicks
//!   README.txt
//!   app\                      swapped as a whole by an update
//!     mc-client.exe
//!     launcher.ps1
//!     aero-install.json       the marker the packaging script writes (InstallMarker)
//!     licenses\ ...
//!   game\                     the player's settings, worlds, logs: never touched
//!   update\                   the updater's own folder, created on first use
//!     state.json              what was checked, offered, staged, tried (crate::state)
//!     download\               the zip while it downloads
//!     staging\<version>\      the extracted release
//!     previous\               the app\ an update replaced (one, for a rollback)
//!     failed\                 an app\ that did not start
//!     helper\                 the copy of the old client that applies an update
//!     update.log              the helper's log
//! ```

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::error::UpdateError;

/// The folder that holds the client inside a portable install.
pub const APP_DIR: &str = "app";
/// The client's file name.
pub const EXE_NAME: &str = "mc-client.exe";
/// The packaging marker's file name, in `app\`.
pub const MARKER_FILE: &str = "aero-install.json";
/// What the player starts, next to `app\`.
pub const LAUNCHER_CMD: &str = "Play Aero Client.cmd";
/// The launcher script in `app\`.
pub const LAUNCHER_SCRIPT: &str = "launcher.ps1";
/// The updater's folder, next to `app\`.
pub const UPDATE_DIR: &str = "update";
/// The marker format this build reads and the packaging script writes.
pub const MARKER_FORMAT: u32 = 1;
/// The only platform packaged today.
pub const PLATFORM: &str = "windows-x64";
/// The largest marker read.
const MAX_MARKER_BYTES: u64 = 4096;

/// `app\aero-install.json`: written by the packaging script, so only a packaged portable build
/// has it.
///
/// ```json
/// {"format": 1, "product": "Aero Client", "version": "0.1.0", "platform": "windows-x64"}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstallMarker {
    /// [`MARKER_FORMAT`].
    pub format: u32,
    /// [`crate::PRODUCT`].
    pub product: String,
    /// The packaged version.
    pub version: String,
    /// [`PLATFORM`].
    pub platform: String,
}

impl InstallMarker {
    /// The marker for a package of `version`.
    #[must_use]
    pub fn new(version: &str) -> InstallMarker {
        InstallMarker {
            format: MARKER_FORMAT,
            product: crate::PRODUCT.to_owned(),
            version: version.to_owned(),
            platform: PLATFORM.to_owned(),
        }
    }

    /// Read and check a marker file: this format, this product, this platform, a version.
    pub fn read(path: &Path) -> Result<InstallMarker, UpdateError> {
        let bytes = read_small(path, MAX_MARKER_BYTES)
            .map_err(|error| UpdateError::io("reading the install marker", error))?;
        // Windows PowerShell's UTF-8 writers put a byte order mark first.
        let json = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
        let marker: InstallMarker = serde_json::from_slice(json)
            .map_err(|_| UpdateError::NotPortable("the install marker is malformed".to_owned()))?;
        if marker.format != MARKER_FORMAT
            || marker.product != crate::PRODUCT
            || marker.platform != PLATFORM
            || crate::version::parse_release_tag(&marker.version).is_none()
        {
            return Err(UpdateError::NotPortable(
                "the install marker is for another product, format or platform".to_owned(),
            ));
        }
        Ok(marker)
    }

    /// The marker as the packaging script writes it.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Read at most `cap` bytes of `path` (an error if it is longer).
pub(crate) fn read_small(path: &Path, cap: u64) -> std::io::Result<Vec<u8>> {
    use std::io::Read as _;
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(cap + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file too large",
        ));
    }
    Ok(bytes)
}

/// The folders of one portable install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallPaths {
    /// The `Aero Client` folder.
    pub root: PathBuf,
    /// `app\`.
    pub app: PathBuf,
    /// `update\`.
    pub update: PathBuf,
    /// The marker in `app\`.
    pub marker: InstallMarker,
}

impl InstallPaths {
    /// The install this process runs from, if it is a packaged portable one (Windows only).
    pub fn detect() -> Result<InstallPaths, UpdateError> {
        if !cfg!(windows) {
            return Err(UpdateError::NotPortable(
                "only the Windows package updates itself".to_owned(),
            ));
        }
        let exe = std::env::current_exe()
            .map_err(|error| UpdateError::io("finding the running client", error))?;
        InstallPaths::from_exe(&exe)
    }

    /// The install `exe` belongs to: it must be `<root>\app\<exe>` with the marker beside it and
    /// `Play Aero Client.cmd` in `<root>`.
    pub fn from_exe(exe: &Path) -> Result<InstallPaths, UpdateError> {
        let app = exe
            .parent()
            .ok_or_else(|| UpdateError::NotPortable("the client has no folder".to_owned()))?;
        let in_app = app
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case(APP_DIR));
        if !in_app {
            return Err(UpdateError::NotPortable(
                "the client is not in an app folder".to_owned(),
            ));
        }
        let root = app
            .parent()
            .ok_or_else(|| UpdateError::NotPortable("the app folder has no parent".to_owned()))?;
        InstallPaths::at_root(root)
    }

    /// The install in `root` (`root\app\aero-install.json` and `root\Play Aero Client.cmd` must
    /// exist).
    pub fn at_root(root: &Path) -> Result<InstallPaths, UpdateError> {
        let app = root.join(APP_DIR);
        let marker_path = app.join(MARKER_FILE);
        if !marker_path.is_file() {
            return Err(UpdateError::NotPortable(
                "no install marker (a development build)".to_owned(),
            ));
        }
        if !root.join(LAUNCHER_CMD).is_file() {
            return Err(UpdateError::NotPortable(format!("no {LAUNCHER_CMD}")));
        }
        let marker = InstallMarker::read(&marker_path)?;
        Ok(InstallPaths {
            root: root.to_path_buf(),
            app,
            update: root.join(UPDATE_DIR),
            marker,
        })
    }

    /// `app\mc-client.exe`.
    #[must_use]
    pub fn exe(&self) -> PathBuf {
        self.app.join(EXE_NAME)
    }

    /// `Play Aero Client.cmd`.
    #[must_use]
    pub fn launcher_cmd(&self) -> PathBuf {
        self.root.join(LAUNCHER_CMD)
    }

    /// `update\state.json`.
    #[must_use]
    pub fn state_file(&self) -> PathBuf {
        self.update.join("state.json")
    }

    /// `update\download\`.
    #[must_use]
    pub fn download_dir(&self) -> PathBuf {
        self.update.join("download")
    }

    /// `update\staging\`.
    #[must_use]
    pub fn staging_root(&self) -> PathBuf {
        self.update.join("staging")
    }

    /// `update\staging\<version>\`: `version` is a parsed semver string, so it is a plain name.
    #[must_use]
    pub fn staging_dir(&self, version: &str) -> PathBuf {
        self.staging_root().join(version)
    }

    /// The extracted release's `Aero Client` folder in [`InstallPaths::staging_dir`].
    #[must_use]
    pub fn staged_root(&self, version: &str) -> PathBuf {
        self.staging_dir(version).join(crate::stage::TOP_DIR)
    }

    /// `update\previous\`.
    #[must_use]
    pub fn previous_dir(&self) -> PathBuf {
        self.update.join("previous")
    }

    /// `update\failed\`.
    #[must_use]
    pub fn failed_dir(&self) -> PathBuf {
        self.update.join("failed")
    }

    /// `update\helper\`.
    #[must_use]
    pub fn helper_dir(&self) -> PathBuf {
        self.update.join("helper")
    }

    /// The helper's copy of the old client.
    #[must_use]
    pub fn helper_exe(&self) -> PathBuf {
        self.helper_dir().join("aero-update-helper.exe")
    }

    /// `update\update.log`.
    #[must_use]
    pub fn log_file(&self) -> PathBuf {
        self.update.join("update.log")
    }
}

/// The portable install this process runs from, detected once (`None` for every development
/// build, after one file check).
pub fn portable_install() -> Option<&'static InstallPaths> {
    static INSTALL: OnceLock<Option<InstallPaths>> = OnceLock::new();
    INSTALL
        .get_or_init(|| match InstallPaths::detect() {
            Ok(paths) => {
                if paths.marker.version != crate::CURRENT_VERSION {
                    tracing::warn!(
                        marker = %paths.marker.version,
                        build = crate::CURRENT_VERSION,
                        "the install marker names another version than this build"
                    );
                }
                Some(paths)
            }
            Err(error) => {
                tracing::debug!(%error, "updates off");
                None
            }
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fake_install;

    #[test]
    fn a_packaged_layout_is_detected_from_its_exe() {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), "0.1.0");
        let found = InstallPaths::from_exe(&paths.exe()).unwrap();
        assert_eq!(found, paths);
        assert_eq!(found.marker.version, "0.1.0");
        assert_eq!(found.update, dir.path().join("Aero Client").join("update"));
    }

    /// A copy installed by Aero-Client-Setup.exe (packaging/aero/installer):
    /// the same layout, in `%LOCALAPPDATA%\Programs\Aero Client`, plus the uninstaller's files
    /// next to `app\` and the shortcuts' icon in it. It is a packaged install like an unzipped
    /// one, and an update from a release zip swaps `app\` (the icon with it) while the
    /// uninstaller's files and `game\` stay where they are.
    #[test]
    fn a_copy_installed_by_setup_is_detected_and_updated_like_an_unzipped_one() {
        let dir = tempfile::tempdir().unwrap();
        let programs = dir.path().join("AppData").join("Local").join("Programs");
        let paths = fake_install(&programs, "0.1.0");
        std::fs::write(paths.app.join("aero-client.ico"), "icon 0.1.0").unwrap();
        std::fs::write(paths.root.join("unins000.exe"), "uninstaller").unwrap();
        std::fs::write(paths.root.join("unins000.dat"), "uninstall log").unwrap();
        std::fs::create_dir_all(paths.root.join("game")).unwrap();
        std::fs::write(paths.root.join("game").join("options.txt"), "mine").unwrap();

        let found = InstallPaths::from_exe(&paths.exe()).unwrap();
        assert_eq!(found, paths);
        assert_eq!(found.root, programs.join("Aero Client"));
        assert_eq!(found.update, programs.join("Aero Client").join(UPDATE_DIR));

        let zip = dir.path().join("release.zip");
        let icon = crate::tests::ZipItem::file("Aero Client/app/aero-client.ico", b"icon 0.2.0");
        std::fs::write(&zip, crate::tests::package_zip("0.2.0", &[icon])).unwrap();
        let limits = crate::stage::Limits::default();
        let staged =
            crate::stage::stage_release(&zip, &paths.staging_dir("0.2.0"), "0.2.0", &limits)
                .unwrap();
        crate::swap::swap_in(&paths, &staged.root).unwrap();

        let read = |path: &Path| std::fs::read_to_string(path).unwrap();
        assert_eq!(read(&paths.exe()), "exe 0.2.0");
        assert_eq!(
            read(&paths.app.join("aero-client.ico")),
            "icon 0.2.0",
            "the shortcuts' icon comes with the new app"
        );
        assert_eq!(read(&paths.root.join("unins000.exe")), "uninstaller");
        assert_eq!(read(&paths.root.join("unins000.dat")), "uninstall log");
        assert_eq!(read(&paths.root.join("game").join("options.txt")), "mine");
        let updated = InstallPaths::from_exe(&paths.exe()).unwrap();
        assert_eq!(
            updated.marker.version, "0.2.0",
            "what the launcher shows in Settings"
        );
    }

    #[test]
    fn development_layouts_are_not_portable() {
        let dir = tempfile::tempdir().unwrap();
        // target\debug\mc-client.exe: not in app\.
        let exe = dir.path().join("target").join("debug").join(EXE_NAME);
        assert!(InstallPaths::from_exe(&exe).is_err());
        // app\ without a marker.
        let paths = fake_install(dir.path(), "0.1.0");
        std::fs::remove_file(paths.app.join(MARKER_FILE)).unwrap();
        assert!(InstallPaths::from_exe(&paths.exe()).is_err());
        // A marker but no Play Aero Client.cmd.
        let other = tempfile::tempdir().unwrap();
        let paths = fake_install(other.path(), "0.1.0");
        std::fs::remove_file(paths.launcher_cmd()).unwrap();
        assert!(InstallPaths::from_exe(&paths.exe()).is_err());
    }

    #[test]
    fn markers_for_other_products_formats_or_platforms_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(MARKER_FILE);
        let good = InstallMarker::new("0.1.0");
        std::fs::write(&path, good.to_json()).unwrap();
        assert_eq!(InstallMarker::read(&path).unwrap(), good);
        for bad in [
            InstallMarker {
                format: 2,
                ..good.clone()
            },
            InstallMarker {
                product: "Other".to_owned(),
                ..good.clone()
            },
            InstallMarker {
                platform: "linux-x64".to_owned(),
                ..good.clone()
            },
            InstallMarker {
                version: "next".to_owned(),
                ..good.clone()
            },
        ] {
            std::fs::write(&path, bad.to_json()).unwrap();
            assert!(InstallMarker::read(&path).is_err(), "{bad:?}");
        }
        std::fs::write(&path, "{not json").unwrap();
        assert!(InstallMarker::read(&path).is_err());
        // What the packaging script writes (PowerShell's ConvertTo-Json layout), with or
        // without a byte order mark.
        let script = "{\r\n    \"format\":  1,\r\n    \"product\":  \"Aero Client\",\r\n    \
                      \"version\":  \"0.1.0\",\r\n    \"platform\":  \"windows-x64\"\r\n}";
        let with_bom = [b"\xEF\xBB\xBF".as_slice(), script.as_bytes()].concat();
        for bytes in [script.as_bytes().to_vec(), with_bom] {
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(InstallMarker::read(&path).unwrap(), good);
        }
        std::fs::write(&path, vec![b' '; 5000]).unwrap();
        assert!(InstallMarker::read(&path).is_err(), "too large");
    }
}
