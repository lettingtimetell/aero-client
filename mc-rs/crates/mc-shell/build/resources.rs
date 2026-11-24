//! The Windows resources of Aero Client's executables, compiled by their build scripts:
//!
//! - the logo as the file's icon (what Explorer, shortcuts and a pinned taskbar button show);
//! - the version information (the file's Properties > Details, and the name Task Manager gives
//!   the process);
//! - an application manifest: pop-ups and the folder picker in the current Windows style, sharp
//!   at any display scale (the game's window was already per-monitor aware: winit asks for it).
//!
//! Shared by `mc-shell/build.rs` (`Aero Client.exe`) and `mc-client/build.rs` (`mc-client.exe`)
//! through `#[path]`. No crate is needed: an ICO file holds the PNG pictures as they are, and the
//! Windows SDK's resource compiler (`rc.exe`, part of the Visual Studio build tools the MSVC
//! toolchain already needs) turns a short script into a `.res` file that the linker takes as an
//! input. Without `rc.exe` the executable is built without these resources and a cargo warning
//! says so: the build never fails because of them.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The logo's pictures in `mc-client/assets/icon/`, `aero-<side>.png`: 16 to 48 px for lists and
/// small icons at 100 % to 200 % display scaling, up to 256 px for Explorer's large icons.
pub fn icon_pngs(folder: &Path) -> Vec<PathBuf> {
    [16, 20, 24, 32, 40, 48, 64, 128, 256]
        .iter()
        .map(|side| folder.join(format!("aero-{side}.png")))
        .collect()
}

/// One executable's resources.
pub struct Exe<'a> {
    /// The binary target they go into; `None` for every binary of the package.
    pub bin: Option<&'a str>,
    /// The icon's pictures: square PNG files, 256 px at most.
    pub icon_pngs: Vec<PathBuf>,
    /// The file description (Task Manager's name for the process).
    pub description: &'a str,
    /// The internal name.
    pub internal_name: &'a str,
    /// The file name the executable is shipped under.
    pub original_filename: &'a str,
    /// A file name stem for the generated files in `OUT_DIR`.
    pub stem: &'a str,
}

/// Compile `exe`'s resources and link them into its binary (Windows and MSVC targets only).
pub fn embed(exe: &Exe<'_>) {
    for png in &exe.icon_pngs {
        println!("cargo:rerun-if-changed={}", png.display());
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        println!(
            "cargo:warning={} is built without its icon: only MSVC targets get resources",
            exe.original_filename
        );
        return;
    }
    match compile(exe) {
        Ok(res) => match exe.bin {
            Some(bin) => println!("cargo:rustc-link-arg-bin={bin}={}", res.display()),
            None => println!("cargo:rustc-link-arg-bins={}", res.display()),
        },
        Err(why) => println!(
            "cargo:warning={} is built without its icon and version information: {why}",
            exe.original_filename
        ),
    }
}

/// Write the icon, the manifest and the script into `OUT_DIR` and compile them into a `.res`.
fn compile(exe: &Exe<'_>) -> Result<PathBuf, String> {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?);
    let ico = format!("{}.ico", exe.stem);
    let manifest = format!("{}.manifest", exe.stem);
    let script = out.join(format!("{}.rc", exe.stem));
    let res = out.join(format!("{}.res", exe.stem));
    let write = |name: &Path, bytes: &[u8]| {
        std::fs::write(name, bytes).map_err(|error| format!("writing {}: {error}", name.display()))
    };
    write(&out.join(&ico), &icon_file(&exe.icon_pngs)?)?;
    write(&out.join(&manifest), MANIFEST.as_bytes())?;
    write(&script, rc_script(exe, &ico, &manifest).as_bytes())?;
    let rc = find_rc().ok_or("no resource compiler (rc.exe from the Windows SDK) was found")?;
    // The script names the icon and the manifest relative to `OUT_DIR`, its working folder.
    let output = Command::new(&rc)
        .arg("/nologo")
        .arg("/fo")
        .arg(&res)
        .arg(&script)
        .current_dir(&out)
        .output()
        .map_err(|error| format!("running {}: {error}", rc.display()))?;
    if !output.status.success() {
        let said = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "rc.exe failed ({}): {}",
            output.status,
            said.trim()
        ));
    }
    Ok(res)
}

/// The application manifest: the Common Controls 6 library (pop-ups in the current style), per
/// monitor DPI awareness from the first instruction (winit asks for the same once the window is
/// made, so the game is unchanged), and no elevation.
const MANIFEST: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
    </dependentAssembly>
  </dependency>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, PerMonitor</dpiAwareness>
    </windowsSettings>
  </application>
</assembly>
"#;

/// The resource script: icon 1 (the one Explorer shows), the version information and the
/// manifest (resource type 24, id 1: the one Windows reads when it starts the program).
fn rc_script(exe: &Exe<'_>, ico: &str, manifest: &str) -> String {
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let [major, minor, patch] = numeric_version(&version);
    // A double quote inside a resource string is written twice.
    let text = |value: &str| value.replace('"', "\"\"");
    format!(
        "1 ICON \"{ico}\"\n\
         1 VERSIONINFO\n\
         FILEVERSION {major},{minor},{patch},0\n\
         PRODUCTVERSION {major},{minor},{patch},0\n\
         FILEFLAGSMASK 0x3F\n\
         FILEFLAGS 0x0\n\
         FILEOS 0x40004\n\
         FILETYPE 0x1\n\
         FILESUBTYPE 0x0\n\
         BEGIN\n\
         BLOCK \"StringFileInfo\"\n\
         BEGIN\n\
         BLOCK \"040904B0\"\n\
         BEGIN\n\
         VALUE \"FileDescription\", \"{description}\"\n\
         VALUE \"FileVersion\", \"{version}\"\n\
         VALUE \"InternalName\", \"{internal}\"\n\
         VALUE \"OriginalFilename\", \"{original}\"\n\
         VALUE \"ProductName\", \"Aero Client\"\n\
         VALUE \"ProductVersion\", \"{version}\"\n\
         END\n\
         END\n\
         BLOCK \"VarFileInfo\"\n\
         BEGIN\n\
         VALUE \"Translation\", 0x409, 1200\n\
         END\n\
         END\n\
         1 24 \"{manifest}\"\n",
        description = text(exe.description),
        version = text(&version),
        internal = text(exe.internal_name),
        original = text(exe.original_filename),
    )
}

/// `major.minor.patch` of a semver version as numbers (anything else is 0).
fn numeric_version(version: &str) -> [u16; 3] {
    let mut parts = version
        .split(['.', '-', '+'])
        .map(|part| part.parse::<u16>().unwrap_or(0));
    [
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    ]
}

/// An ICO file of the PNG pictures, smallest first: the six-byte header, a sixteen-byte entry
/// per picture (its size, 0 meaning 256, and where its bytes are), then the PNG files unchanged.
fn icon_file(pngs: &[PathBuf]) -> Result<Vec<u8>, String> {
    let mut pictures = Vec::with_capacity(pngs.len());
    for path in pngs {
        let bytes =
            std::fs::read(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
        let side = match png_size(&bytes) {
            Some((width, height)) if width == height && (1..=256).contains(&width) => width,
            _ => {
                return Err(format!(
                    "{} is not a square PNG of 256 px or less",
                    path.display()
                ))
            }
        };
        pictures.push((side, bytes));
    }
    if pictures.is_empty() {
        return Err("no icon pictures".to_owned());
    }
    pictures.sort_by_key(|(side, _)| *side);
    let count = u16::try_from(pictures.len()).map_err(|_| "too many icon pictures")?;
    let mut file = Vec::new();
    file.extend_from_slice(&0_u16.to_le_bytes()); // reserved
    file.extend_from_slice(&1_u16.to_le_bytes()); // an icon, not a cursor
    file.extend_from_slice(&count.to_le_bytes());
    let mut offset = 6 + 16 * u32::from(count);
    for (side, bytes) in &pictures {
        let size = u32::try_from(bytes.len()).map_err(|_| "an icon picture is too large")?;
        let dimension = u8::try_from(*side).unwrap_or(0); // 256 is written as 0
        file.extend_from_slice(&[dimension, dimension, 0, 0]); // width, height, palette, reserved
        file.extend_from_slice(&1_u16.to_le_bytes()); // colour planes
        file.extend_from_slice(&32_u16.to_le_bytes()); // bits per pixel
        file.extend_from_slice(&size.to_le_bytes());
        file.extend_from_slice(&offset.to_le_bytes());
        offset += size;
    }
    for (_, bytes) in &pictures {
        file.extend_from_slice(bytes);
    }
    Ok(file)
}

/// A PNG file's width and height, from its header.
fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((width, height))
}

/// The Windows SDK's `rc.exe`: the one a developer prompt names, then one on `PATH`, then the
/// newest Windows 10/11 SDK in its standard place, for this PC's processor.
fn find_rc() -> Option<PathBuf> {
    let architectures: &[&str] = match std::env::consts::ARCH {
        "aarch64" => &["arm64", "x64", "x86"],
        "x86" => &["x86"],
        _ => &["x64", "x86"],
    };
    let in_folder = |folder: &Path| {
        architectures
            .iter()
            .map(|architecture| folder.join(architecture).join("rc.exe"))
            .find(|rc| rc.is_file())
    };
    if let Some(folder) = std::env::var_os("WindowsSdkVerBinPath") {
        if let Some(rc) = in_folder(Path::new(&folder)) {
            return Some(rc);
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        if let Some(rc) = std::env::split_paths(&path)
            .map(|folder| folder.join("rc.exe"))
            .find(|rc| rc.is_file())
        {
            return Some(rc);
        }
    }
    for program_files in ["ProgramFiles(x86)", "ProgramFiles"] {
        let Some(base) = std::env::var_os(program_files) else {
            continue;
        };
        let bin = Path::new(&base).join("Windows Kits").join("10").join("bin");
        let mut versions: Vec<(Vec<u32>, PathBuf)> = std::fs::read_dir(&bin)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                let numbers = name
                    .split('.')
                    .map(|part| part.parse::<u32>().ok())
                    .collect::<Option<Vec<u32>>>()?;
                Some((numbers, entry.path()))
            })
            .collect();
        versions.sort();
        for (_, folder) in versions.iter().rev() {
            if let Some(rc) = in_folder(folder) {
                return Some(rc);
            }
        }
        if let Some(rc) = in_folder(&bin) {
            return Some(rc);
        }
    }
    None
}
