//! Windows' DLL search order, made safe once at start-up.
//!
//! A DLL loaded by bare name is looked for in the program's folder, the system folders, then the
//! **current directory** and every folder on `PATH`. The launcher starts the client in the
//! player's `game` folder, which players zip and send to each other, and some DLLs the client
//! loads by name are not part of Windows: the DX12 back-end first tries `dxcompiler.dll` (DXC)
//! before the system's `d3dcompiler_47.dll`, and a PC without a Vulkan driver has no
//! `vulkan-1.dll`. A `dxcompiler.dll` left in a shared `game` folder therefore ran as soon as a
//! DX12 device was created. After [`harden_dll_search_order`], a `LoadLibrary` or `LoadLibraryEx`
//! that names no search folders of its own looks only in the program's folder, `System32` and
//! folders added with `AddDllDirectory` (none are).
//!
//! What it cannot cover: the DLLs the executable imports statically are resolved before `main`
//! (the program's folder first, then `System32`). All of them are Windows' own DLLs, so planting
//! one needs write access to the program's folder, which could replace the program itself.

/// `MC_W21_SL_DLL_SEARCH=0` keeps Windows' default search order (to check a graphics driver,
/// overlay or audio device that stopped loading against it). Read once, at start-up.
pub const DLL_SEARCH_ENV: &str = "MC_W21_SL_DLL_SEARCH";

#[cfg(windows)]
mod sys {
    /// The program's folder, `System32` and `AddDllDirectory` folders.
    pub const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;

    #[link(name = "kernel32")]
    extern "system" {
        pub fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
    }
}

/// Make every later by-name DLL load skip the current directory and `PATH` (see the module
/// docs). Call it first thing in `main`, before any thread or graphics, audio or video library
/// starts. Returns whether the safe order is now in effect: `false` when
/// [`DLL_SEARCH_ENV`] is `0`, off Windows, or if Windows refused (logged).
pub fn harden_dll_search_order() -> bool {
    if std::env::var_os(DLL_SEARCH_ENV).is_some_and(|value| value == "0") {
        return false;
    }
    set_default_dll_directories()
}

#[cfg(windows)]
fn set_default_dll_directories() -> bool {
    // SAFETY: `SetDefaultDllDirectories` takes one flags value and no pointers, and only changes
    // which folders this process searches for DLLs loaded later (Windows 8 and later; the client
    // needs Windows 10).
    let ok = unsafe { sys::SetDefaultDllDirectories(sys::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) };
    if ok == 0 {
        tracing::warn!(
            error = %std::io::Error::last_os_error(),
            "the safe DLL search order could not be set"
        );
    }
    ok != 0
}

#[cfg(not(windows))]
fn set_default_dll_directories() -> bool {
    false
}
