//! What `Aero Client.exe` does: the `aero-start` binary, which the packaging copies to the
//! portable folder's root under that name. It starts `app\mc-client.exe` with its own
//! arguments, in its own folder, and exits; started with no arguments, the client makes its
//! packaged start. Nothing stays running and no console window ever opens.
//!
//! When the client can't be started it says why in a message: the zip was not extracted (the
//! starter runs from a zip viewer's temporary copy, as the old `Play Aero Client.cmd` noticed),
//! the client is missing (an antivirus may have removed it), or Windows refused to start it.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};

/// The title of every message.
pub const TITLE: &str = "Aero Client";

/// When the starter runs from inside the zip.
pub const FROM_ZIP: &str = "Aero Client can't start from inside the zip file.\n\n\
    Right-click the zip file and choose \"Extract All...\". Then open the extracted \
    \"Aero Client\" folder and double-click \"Aero Client\".";

/// When `app\mc-client.exe` is missing.
pub const MISSING: &str = "Aero Client can't start: the game file app\\mc-client.exe is \
    missing.\n\nYour antivirus may have removed it. Extract the zip again, then start Aero \
    Client.";

/// `<folder>\app\mc-client.exe`.
#[must_use]
pub fn client_path(folder: &Path) -> PathBuf {
    folder.join("app").join("mc-client.exe")
}

/// Whether `folder` looks like where a zip viewer copied a program it runs from inside a zip:
/// the temporary folder `temp` or anything in it, or a folder named the way Windows'
/// (`Temp1_<zip name>.zip`), 7-Zip's (`7zO...`) or WinRAR's (`Rar$EX...`) viewer names it.
/// Letter case doesn't matter, as on Windows.
#[must_use]
pub fn looks_like_zip_preview(folder: &Path, temp: &Path) -> bool {
    let lower = |path: &Path| {
        path.components()
            .map(|part| part.as_os_str().to_string_lossy().to_lowercase())
            .collect::<Vec<_>>()
    };
    let (folder, temp) = (lower(folder), lower(temp));
    let in_temp = !temp.is_empty() && folder.starts_with(&temp);
    in_temp
        || folder.iter().any(|part| {
            part.ends_with(".zip") || part.starts_with("7zo") || part.starts_with("rar$ex")
        })
}

/// Start the client in `folder` (`app\mc-client.exe`, working folder `folder`) with `args`,
/// without a console window even for a client built with one. The error is the message to show.
///
/// # Errors
///
/// [`FROM_ZIP`] or [`MISSING`] when the client isn't there, or a message naming Windows' reason
/// when it refused to start it.
pub fn start_client(
    folder: &Path,
    args: impl IntoIterator<Item = OsString>,
) -> Result<Child, String> {
    let client = client_path(folder);
    if !client.is_file() {
        let from_zip = looks_like_zip_preview(folder, &std::env::temp_dir());
        return Err(if from_zip { FROM_ZIP } else { MISSING }.to_owned());
    }
    let mut command = Command::new(client);
    command.args(args).current_dir(folder);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command.spawn().map_err(|error| {
        format!(
            "Aero Client can't start: Windows couldn't start app\\mc-client.exe ({error}).\n\n\
             Your antivirus may have blocked it. Extract the zip again, then start Aero Client."
        )
    })
}

/// Let the process `pid` bring its window to the front. The starter may (the player started it)
/// and passes that on, so the game's window opens in front rather than behind the folder.
pub fn allow_set_foreground(pid: u32) -> bool {
    #[cfg(windows)]
    {
        crate::sys::allow_set_foreground(pid)
    }
    #[cfg(not(windows))]
    {
        let _ = pid;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zip_viewers_temporary_copy_is_recognised() {
        let temp = Path::new(r"C:\Users\Player\AppData\Local\Temp");
        for folder in [
            // Windows' own viewer, "Run" in its warning.
            r"C:\Users\Player\AppData\Local\Temp\Temp1_Aero-Client-1.0.0-windows-x64.zip\Aero Client",
            // 7-Zip and WinRAR.
            r"C:\Users\Player\AppData\Local\Temp\7zO0A1B2C3D\Aero Client",
            r"C:\Users\Player\AppData\Local\Temp\Rar$EXa1234.5678\Aero Client",
            // Anything else in the temporary folder, in any letter case.
            r"c:\users\player\appdata\local\temp\some-viewer\Aero Client",
            // The same viewers with the temporary folder elsewhere.
            r"D:\Temp\Temp2_Aero Client.zip\Aero Client",
            r"D:\Scratch\7zO11\Aero Client",
        ] {
            assert!(looks_like_zip_preview(Path::new(folder), temp), "{folder}");
        }
        for folder in [
            r"C:\Users\Player\Downloads\Aero Client",
            r"C:\Users\Player\AppData\Local\Programs\Aero Client",
            r"D:\Games\Aero Client",
            // A folder whose name only begins like the temporary folder's.
            r"C:\Users\Player\AppData\Local\Temporary Games\Aero Client",
        ] {
            assert!(!looks_like_zip_preview(Path::new(folder), temp), "{folder}");
        }
        assert!(
            !looks_like_zip_preview(Path::new(r"D:\Games\Aero Client"), Path::new("")),
            "no temporary folder known"
        );
    }

    #[test]
    fn a_missing_client_is_explained() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("Aero Client");
        std::fs::create_dir_all(&folder).unwrap();
        let error = start_client(&folder, []).unwrap_err();
        // The test's own folder is in the temporary folder, so this reads as a zip viewer's copy.
        assert_eq!(error, FROM_ZIP);
        assert_eq!(
            client_path(&folder),
            folder.join("app").join("mc-client.exe")
        );
    }
}
