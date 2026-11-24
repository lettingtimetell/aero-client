//! Every `unsafe` call of the crate (Windows only; the crate root denies `unsafe_code` elsewhere).
//!
//! The message box, the folder picker, COM and the foreground permission use the `windows`
//! crate's bindings. The registry and standard-handle functions are imported here with that
//! crate's own `link!` macro (the raw-dylib import every one of its bindings uses): their binding
//! modules are `windows` features no other crate of the client enables, and a new feature would
//! rebuild `windows` and everything above it (wgpu, cpal, the renderer) for every build of the
//! client. Their declarations are the documented Win32 signatures, with the registry key handle
//! as a pointer-sized integer.
#![allow(unsafe_code)]

use std::ffi::{c_void, OsString};
use std::fs::File;
use std::io;
use std::os::windows::ffi::OsStringExt as _;
use std::os::windows::io::{AsRawHandle as _, IntoRawHandle as _};
use std::path::PathBuf;

use windows::core::{HRESULT, PCWSTR};
use windows::Win32::Foundation::ERROR_CANCELLED;
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_INPROC_SERVER,
    COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE,
};
use windows::Win32::UI::Shell::{
    FileOpenDialog, IFileOpenDialog, FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_PATHMUSTEXIST,
    FOS_PICKFOLDERS, SIGDN_FILESYSPATH,
};
use windows::Win32::UI::WindowsAndMessaging::{
    AllowSetForegroundWindow, MessageBoxW, IDOK, IDYES, MB_ICONERROR, MB_ICONINFORMATION,
    MB_ICONQUESTION, MB_ICONWARNING, MB_OK, MB_OKCANCEL, MB_SETFOREGROUND, MB_YESNO,
};

use crate::dialog::{Answer, Buttons, Icon};

/// A registry key handle (`HKEY`).
type Hkey = isize;
/// `HKEY_CURRENT_USER`: the predefined handle `0x80000001`, sign-extended as Windows defines it.
const HKEY_CURRENT_USER: Hkey = 0x8000_0001_u32 as i32 as isize;
const KEY_SET_VALUE: u32 = 0x0002;
const REG_SZ: u32 = 1;
const RRF_RT_REG_SZ: u32 = 0x0000_0002;
const ERROR_SUCCESS: u32 = 0;
const ERROR_FILE_NOT_FOUND: u32 = 2;
const ERROR_MORE_DATA: u32 = 234;
/// The longest registry string read (bytes): far above any folder path or version.
const MAX_REGISTRY_STRING_BYTES: u32 = 64 * 1024;
/// `STD_OUTPUT_HANDLE` and `STD_ERROR_HANDLE`: `(DWORD)-11` and `(DWORD)-12`.
const STD_OUTPUT_HANDLE: u32 = -11_i32 as u32;
const STD_ERROR_HANDLE: u32 = -12_i32 as u32;

windows::core::link!("advapi32.dll" "system" fn RegGetValueW(
    hkey: isize, lpsubkey: *const u16, lpvalue: *const u16, dwflags: u32, pdwtype: *mut u32,
    pvdata: *mut c_void, pcbdata: *mut u32) -> u32);
windows::core::link!("advapi32.dll" "system" fn RegOpenKeyExW(
    hkey: isize, lpsubkey: *const u16, uloptions: u32, samdesired: u32, phkresult: *mut isize)
    -> u32);
windows::core::link!("advapi32.dll" "system" fn RegSetValueExW(
    hkey: isize, lpvaluename: *const u16, reserved: u32, dwtype: u32, lpdata: *const u8,
    cbdata: u32) -> u32);
windows::core::link!("advapi32.dll" "system" fn RegCloseKey(hkey: isize) -> u32);
windows::core::link!("kernel32.dll" "system" fn SetStdHandle(
    nstdhandle: u32, hhandle: *mut c_void) -> i32);

/// `text` as a null-terminated UTF-16 string (an embedded NUL would end it early).
fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A Win32 error code as an [`io::Error`] (`2`, file not found, is [`io::ErrorKind::NotFound`]).
fn win32_error(code: u32) -> io::Error {
    io::Error::from_raw_os_error(code as i32)
}

/// `MessageBoxW` with no owner window, brought to the front.
pub(crate) fn message_box(title: &str, text: &str, icon: Icon, buttons: Buttons) -> Answer {
    let icon = match icon {
        Icon::Information => MB_ICONINFORMATION,
        Icon::Warning => MB_ICONWARNING,
        Icon::Error => MB_ICONERROR,
        Icon::Question => MB_ICONQUESTION,
    };
    let kind = match buttons {
        Buttons::Ok => MB_OK,
        Buttons::OkCancel => MB_OKCANCEL,
        Buttons::YesNo => MB_YESNO,
    };
    let (title, text) = (wide(title), wide(text));
    // SAFETY: both strings are null-terminated UTF-16 buffers that outlive the call, and no
    // owner window is passed.
    let answer = unsafe {
        MessageBoxW(
            None,
            PCWSTR(text.as_ptr()),
            PCWSTR(title.as_ptr()),
            icon | kind | MB_SETFOREGROUND,
        )
    };
    match (buttons, answer) {
        (Buttons::YesNo, IDYES) => Answer::Yes,
        (Buttons::YesNo, _) => Answer::No,
        (_, IDOK) => Answer::Ok,
        _ => Answer::Cancel,
    }
}

/// This thread's membership of a single-threaded COM apartment (what the shell's dialogs need),
/// left on drop.
struct Apartment {
    entered: bool,
}

impl Apartment {
    fn enter() -> Apartment {
        // SAFETY: no pointers (the reserved argument is null). A success (S_OK, or S_FALSE when
        // this thread already was in an apartment of this kind) is balanced by exactly one
        // CoUninitialize in `drop`, on this thread: the value never leaves `pick_folder`.
        let result =
            unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) };
        Apartment {
            entered: result.is_ok(),
        }
    }
}

impl Drop for Apartment {
    fn drop(&mut self) {
        if self.entered {
            // SAFETY: balances the successful CoInitializeEx in `enter`, on the same thread.
            unsafe { CoUninitialize() };
        }
    }
}

/// The shell's folder picker (`IFileOpenDialog` with `FOS_PICKFOLDERS`), titled `title`: the
/// chosen folder, or `None` when the player cancels or the dialog can't be shown.
pub(crate) fn pick_folder(title: &str) -> Option<PathBuf> {
    let _apartment = Apartment::enter();
    let title = wide(title);
    let picked = (|| -> windows::core::Result<Option<PathBuf>> {
        // SAFETY: COM is initialised on this thread (`_apartment`); the class id is the
        // documented one for this interface, created in-process.
        let dialog: IFileOpenDialog =
            unsafe { CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER)? };
        // SAFETY (this block): `dialog` is a live interface used only on this thread, and the
        // title is a null-terminated UTF-16 buffer that outlives the call.
        unsafe {
            let options = dialog.GetOptions()?;
            dialog.SetOptions(
                options
                    | FOS_PICKFOLDERS
                    | FOS_FORCEFILESYSTEM
                    | FOS_PATHMUSTEXIST
                    | FOS_NOCHANGEDIR,
            )?;
            dialog.SetTitle(PCWSTR(title.as_ptr()))?;
        }
        // SAFETY: as above; no owner window. Cancelling is reported as ERROR_CANCELLED.
        match unsafe { dialog.Show(None) } {
            Ok(()) => {}
            Err(error) if error.code() == HRESULT::from_win32(ERROR_CANCELLED.0) => {
                return Ok(None)
            }
            Err(error) => return Err(error),
        }
        // SAFETY: the dialog was shown and confirmed, so it holds a result.
        let item = unsafe { dialog.GetResult()? };
        // SAFETY: a live item; SIGDN_FILESYSPATH asks for its file system path, which the
        // FOS_FORCEFILESYSTEM option guarantees it has.
        let name = unsafe { item.GetDisplayName(SIGDN_FILESYSPATH)? };
        // SAFETY: `name` is the null-terminated string the call above allocated with the COM
        // allocator; it is read once, then freed once, and not used after.
        let path = unsafe {
            let path = OsString::from_wide(name.as_wide());
            CoTaskMemFree(Some(name.0 as *const c_void));
            path
        };
        Ok(Some(PathBuf::from(path)))
    })();
    picked.unwrap_or_else(|error| {
        eprintln!("the folder picker could not be shown: {error}");
        None
    })
}

/// The `REG_SZ` value `name` of `HKEY_CURRENT_USER\<key>`.
pub(crate) fn read_current_user_string(key: &str, name: &str) -> io::Result<Option<String>> {
    let (key, name) = (wide(key), wide(name));
    // The size first, then the value; again (a few times) if it grew in between.
    let mut bytes: u32 = 0;
    for _ in 0..4 {
        let mut buffer = vec![0_u16; (bytes as usize).div_ceil(2)];
        let mut size = bytes;
        let data: *mut c_void = if buffer.is_empty() {
            std::ptr::null_mut()
        } else {
            buffer.as_mut_ptr().cast()
        };
        // SAFETY: the key and value names are null-terminated UTF-16 buffers; `data` is null
        // (asking for the size only) or points to `buffer`, whose `size` bytes are writable;
        // `size` and the null type pointer are what the API documents.
        let status = unsafe {
            RegGetValueW(
                HKEY_CURRENT_USER,
                key.as_ptr(),
                name.as_ptr(),
                RRF_RT_REG_SZ,
                std::ptr::null_mut(),
                data,
                &mut size,
            )
        };
        match status {
            ERROR_SUCCESS if !data.is_null() => {
                // `size` counts bytes, the terminating NUL included.
                buffer.truncate(size as usize / 2);
                while buffer.last() == Some(&0) {
                    buffer.pop();
                }
                return String::from_utf16(&buffer)
                    .map(Some)
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "not UTF-16"));
            }
            ERROR_SUCCESS if size == 0 => return Ok(Some(String::new())),
            ERROR_SUCCESS | ERROR_MORE_DATA if size <= MAX_REGISTRY_STRING_BYTES => bytes = size,
            ERROR_SUCCESS | ERROR_MORE_DATA => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "value too long"))
            }
            ERROR_FILE_NOT_FOUND => return Ok(None),
            other => return Err(win32_error(other)),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::Interrupted,
        "the value kept changing",
    ))
}

/// Set the `REG_SZ` value `name` of the existing key `HKEY_CURRENT_USER\<key>`.
pub(crate) fn set_current_user_string(key: &str, name: &str, value: &str) -> io::Result<()> {
    let (key, name, value) = (wide(key), wide(name), wide(value));
    let bytes = u32::try_from(value.len() * 2)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "value too long"))?;
    let mut handle: Hkey = 0;
    // SAFETY: the key name is a null-terminated UTF-16 buffer and `handle` a writable handle
    // slot. Opening (never creating) the key: a missing one is an error here.
    let status = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            key.as_ptr(),
            0,
            KEY_SET_VALUE,
            &mut handle,
        )
    };
    if status != ERROR_SUCCESS {
        return Err(win32_error(status));
    }
    // SAFETY: `handle` was opened above with KEY_SET_VALUE; the value name is null-terminated;
    // the data are `value`'s bytes, the terminating NUL included as REG_SZ requires.
    let status = unsafe {
        RegSetValueExW(
            handle,
            name.as_ptr(),
            0,
            REG_SZ,
            value.as_ptr().cast(),
            bytes,
        )
    };
    // SAFETY: closes the handle opened above, once; it is not used after.
    unsafe { RegCloseKey(handle) };
    if status == ERROR_SUCCESS {
        Ok(())
    } else {
        Err(win32_error(status))
    }
}

/// Make `out` and `err` this process's standard output and error. They stay open until the
/// process ends.
pub(crate) fn set_std_handles(out: File, err: File) -> io::Result<()> {
    let set = |which: u32, file: &File| {
        // SAFETY: SetStdHandle only stores the handle value. The handle is a valid file handle
        // that is never closed once stored (see below), so it stays valid for every later write.
        if unsafe { SetStdHandle(which, file.as_raw_handle()) } == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    };
    set(STD_OUTPUT_HANDLE, &out)?;
    // Kept open for the rest of the process: it is its standard output now.
    let _ = out.into_raw_handle();
    set(STD_ERROR_HANDLE, &err)?;
    let _ = err.into_raw_handle();
    Ok(())
}

/// Let the process `pid` bring its window to the front (this process may, having been started
/// by the player).
pub(crate) fn allow_set_foreground(pid: u32) -> bool {
    // SAFETY: no pointers; it fails harmlessly when this process may not grant the permission.
    unsafe { AllowSetForegroundWindow(pid) }.is_ok()
}
