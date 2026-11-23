//! Give the memory the allocator has freed back to the OS at the
//! points where vanilla calls `System.gc()`.
//!
//! The game binary's allocator (mimalloc, `mc-client`'s `mimalloc` feature) keeps the pages a
//! thread freed for a short delay before it decommits them, so that a buffer freed and allocated
//! again from one frame to the next is not decommitted and faulted back in every frame (the
//! reason the binary uses it: with no delay the title screen takes ~51,000 page faults a second
//! instead of ~500). That delayed purge only runs when the thread that owns those pages
//! allocates or frees in the same place again, so the memory of a world that was just torn down
//! stays committed and resident while the menus are up.
//!
//! [`collect_freed_memory`] asks the allocator to run those pending purges now
//! (`mi_collect(true)`): the calling thread's heap, the segments of threads that have exited, and
//! the shared arenas. [`collect_on_thread`] does the same from a worker thread for its own heap.
//! They are called when a world is left (`Minecraft.loadWorld(null)`, which ends with
//! `System.gc()`), never per frame, and touch no data: every block that is live stays where it
//! is. Measured: leaving the SkyWars capture gives back 18-35 MB of working
//! set and 29-81 MB of commit at once (1-7 ms, once), and the commit stays ~10 MB lower once the
//! menu has settled; at start-up (warm or cold bake cache) there is nothing pending, so it is not
//! called there. `MC_W21_MB_COLLECT=0` (read once) turns it off. Without the `mimalloc` feature
//! (the system allocator) it does nothing.
//!
//! [`process_memory`] and [`total_physical_memory`] read the OS's own counters for this process
//! and machine in microseconds. The F3 overlay and crash reports used to get the same figures by
//! starting `tasklist.exe` and `powershell.exe` (once at start-up, then both every second while
//! F3 is open), because the crates that need them may not call the OS directly (`unsafe` lives
//! here only).
//!
//! [`map_file_for_process`] maps a file that is read once and kept all session (the mod menu's
//! system fonts, 2.8 MB) instead of copying it into private memory.

use std::sync::OnceLock;

/// Kill switch: `0` keeps the allocator's own purge timing everywhere.
pub const COLLECT_ENV: &str = "MC_W21_MB_COLLECT";

/// Whether the transition collects run (read once; default on).
#[must_use]
pub fn collect_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(COLLECT_ENV).is_none_or(|value| value != "0"))
}

#[cfg(feature = "mimalloc")]
mod ffi {
    // Links the allocator's static library into every binary that enables this feature (the
    // binary that installs mimalloc as its global allocator links it anyway).
    use libmimalloc_sys as _;

    extern "C" {
        /// `void mi_collect(bool force)`: free empty pages and run every pending purge of the
        /// calling thread's heap, collect abandoned segments and purge the arenas. Thread-safe.
        pub(super) fn mi_collect(force: bool);
    }
}

/// Run the allocator's pending purges for the calling thread now. Returns how long it took, or
/// `None` when nothing ran (switch off, or not built with the allocator).
pub fn collect_on_thread() -> Option<std::time::Duration> {
    if !collect_enabled() {
        return None;
    }
    #[cfg(feature = "mimalloc")]
    {
        let started = std::time::Instant::now();
        // SAFETY: `mi_collect` takes no pointers and is safe to call from any thread at any time;
        // with the system allocator as the global one it collects mimalloc's own (empty) heap.
        unsafe { ffi::mi_collect(true) };
        Some(started.elapsed())
    }
    #[cfg(not(feature = "mimalloc"))]
    {
        None
    }
}

/// This process's memory as the OS counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemory {
    /// Bytes of the process's pages in RAM (`WorkingSetSize`; what `tasklist` calls "Mem Usage"
    /// and `Get-Process` `WorkingSet64`).
    pub working_set: u64,
    /// Bytes of private memory committed (`PrivateUsage`; `Get-Process` `PrivateMemorySize64`).
    pub private: u64,
}

#[cfg(windows)]
mod os {
    #[repr(C)]
    #[derive(Default)]
    pub(super) struct Counters {
        pub cb: u32,
        pub page_fault_count: u32,
        pub peak_working_set: usize,
        pub working_set: usize,
        pub quota_peak_paged_pool: usize,
        pub quota_paged_pool: usize,
        pub quota_peak_non_paged_pool: usize,
        pub quota_non_paged_pool: usize,
        pub pagefile_usage: usize,
        pub peak_pagefile_usage: usize,
        pub private_usage: usize,
    }

    #[repr(C)]
    #[derive(Default)]
    pub(super) struct MemoryStatus {
        pub length: u32,
        pub memory_load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page_file: u64,
        pub avail_page_file: u64,
        pub total_virtual: u64,
        pub avail_virtual: u64,
        pub avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub(super) fn GetCurrentProcess() -> isize;
        /// `PROCESS_MEMORY_COUNTERS_EX` through psapi's kernel32 export.
        pub(super) fn K32GetProcessMemoryInfo(
            process: isize,
            counters: *mut Counters,
            cb: u32,
        ) -> i32;
        pub(super) fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32;
    }
}

/// This process's working set and private bytes, straight from the OS (a few microseconds, no
/// child process). `None` off Windows or if the call fails.
#[must_use]
pub fn process_memory() -> Option<ProcessMemory> {
    #[cfg(windows)]
    {
        let mut counters = os::Counters {
            cb: std::mem::size_of::<os::Counters>() as u32,
            ..os::Counters::default()
        };
        // SAFETY: the pseudo-handle of this process and an out-parameter of the declared size.
        let ok = unsafe {
            os::K32GetProcessMemoryInfo(os::GetCurrentProcess(), &mut counters, counters.cb)
        };
        (ok != 0).then_some(ProcessMemory {
            working_set: counters.working_set as u64,
            private: counters.private_usage as u64,
        })
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// The machine's physical memory visible to the OS (`GlobalMemoryStatusEx`'s `ullTotalPhys`, the
/// figure WMI's `Win32_ComputerSystem.TotalPhysicalMemory` reports). `None` off Windows or on
/// failure.
#[must_use]
pub fn total_physical_memory() -> Option<u64> {
    #[cfg(windows)]
    {
        let mut status = os::MemoryStatus {
            length: std::mem::size_of::<os::MemoryStatus>() as u32,
            ..os::MemoryStatus::default()
        };
        // SAFETY: an out-parameter whose `dwLength` names its size, as the API requires.
        let ok = unsafe { os::GlobalMemoryStatusEx(&mut status) };
        (ok != 0 && status.total_phys > 0).then_some(status.total_phys)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Kill switch for [`map_file_for_process`]: `0` reads such files into memory as before.
pub const MAP_FILES_ENV: &str = "MC_W21_MB_MAP_FILES";

/// Whether [`map_file_for_process`] maps (read once; default on).
#[must_use]
pub fn map_files_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(MAP_FILES_ENV).is_none_or(|value| value != "0"))
}

#[cfg(windows)]
mod map {
    use std::ffi::c_void;

    pub(super) const GENERIC_READ: u32 = 0x8000_0000;
    pub(super) const FILE_SHARE_READ: u32 = 0x0000_0001;
    pub(super) const OPEN_EXISTING: u32 = 3;
    pub(super) const FILE_ATTRIBUTE_NORMAL: u32 = 0x0000_0080;
    pub(super) const INVALID_HANDLE_VALUE: isize = -1;
    pub(super) const PAGE_READONLY: u32 = 0x02;
    pub(super) const FILE_MAP_READ: u32 = 0x0004;

    #[link(name = "kernel32")]
    extern "system" {
        pub(super) fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *const c_void,
            disposition: u32,
            flags: u32,
            template: isize,
        ) -> isize;
        pub(super) fn GetFileSizeEx(file: isize, size: *mut i64) -> i32;
        pub(super) fn CreateFileMappingW(
            file: isize,
            security: *const c_void,
            protect: u32,
            maximum_size_high: u32,
            maximum_size_low: u32,
            name: *const u16,
        ) -> isize;
        pub(super) fn MapViewOfFile(
            mapping: isize,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        pub(super) fn CloseHandle(handle: isize) -> i32;
    }
}

/// The whole file at `path`, mapped read-only for the rest of the process instead of copied
/// into private memory. For files that are read once and kept for the whole session (the menu's
/// system fonts): the pages are the OS's file cache, shared with every other program using the
/// file, so they cost this process no private memory and the OS can drop and re-read them.
///
/// The file is opened with a share mode that denies writers and that handle is kept until the
/// process exits, so the bytes behind the slice can never change; the view is never unmapped.
/// `None` (the caller reads the file instead) off Windows, with the switch off, for an empty or
/// over-1-GiB file, or when the file cannot be opened that way (for example another program
/// already has it open for writing).
#[must_use]
pub fn map_file_for_process(path: &std::path::Path) -> Option<&'static [u8]> {
    if !map_files_enabled() {
        return None;
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;

        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `wide` is a NUL-terminated path; every handle is checked before use and the
        // mapping handle is closed once the view exists (the view keeps the section alive).
        // `file` is deliberately never closed: its share mode (read only) keeps every writer
        // out, which is what makes the `'static` slice immutable for as long as it exists.
        unsafe {
            let file = map::CreateFileW(
                wide.as_ptr(),
                map::GENERIC_READ,
                map::FILE_SHARE_READ,
                std::ptr::null(),
                map::OPEN_EXISTING,
                map::FILE_ATTRIBUTE_NORMAL,
                0,
            );
            if file == map::INVALID_HANDLE_VALUE {
                return None;
            }
            let mut size = 0_i64;
            if map::GetFileSizeEx(file, &mut size) == 0 || size <= 0 || size > (1 << 30) {
                map::CloseHandle(file);
                return None;
            }
            let mapping = map::CreateFileMappingW(
                file,
                std::ptr::null(),
                map::PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            );
            if mapping == 0 {
                map::CloseHandle(file);
                return None;
            }
            let view = map::MapViewOfFile(mapping, map::FILE_MAP_READ, 0, 0, 0);
            map::CloseHandle(mapping);
            if view.is_null() {
                map::CloseHandle(file);
                return None;
            }
            Some(std::slice::from_raw_parts(
                view.cast::<u8>().cast_const(),
                size as usize,
            ))
        }
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        None
    }
}

// ---- A crash report's facts from the OS directly -------------------
//
// A crash report names the local time twice (its file name and its "Time:" line) and the
// operating system. They came from three `powershell.exe` runs per report (about a second and
// 1.5-1.9 CPU-seconds each, while the game was going down); these are the same values from the
// OS's own calls, in-process and in microseconds.

#[cfg(windows)]
mod facts {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub(super) struct SystemTime {
        pub year: u16,
        pub month: u16,
        pub day_of_week: u16,
        pub day: u16,
        pub hour: u16,
        pub minute: u16,
        pub second: u16,
        pub milliseconds: u16,
    }

    #[repr(C)]
    pub(super) struct FileTime {
        pub low: u32,
        pub high: u32,
    }

    #[repr(C)]
    pub(super) struct OsVersionInfo {
        pub size: u32,
        pub major: u32,
        pub minor: u32,
        pub build: u32,
        pub platform: u32,
        pub service_pack: [u16; 128],
    }

    pub(super) const LOCALE_SSHORTDATE: u32 = 0x1F;
    pub(super) const LOCALE_SSHORTTIME: u32 = 0x79;
    pub(super) const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;

    #[link(name = "kernel32")]
    extern "system" {
        pub(super) fn FileTimeToSystemTime(file: *const FileTime, system: *mut SystemTime) -> i32;
        pub(super) fn SystemTimeToTzSpecificLocalTime(
            zone: *const c_void,
            universal: *const SystemTime,
            local: *mut SystemTime,
        ) -> i32;
        pub(super) fn GetLocaleInfoEx(
            locale: *const u16,
            kind: u32,
            out: *mut u16,
            len: i32,
        ) -> i32;
        pub(super) fn GetDateFormatEx(
            locale: *const u16,
            flags: u32,
            date: *const SystemTime,
            format: *const u16,
            out: *mut u16,
            len: i32,
            calendar: *const u16,
        ) -> i32;
        pub(super) fn GetTimeFormatEx(
            locale: *const u16,
            flags: u32,
            time: *const SystemTime,
            format: *const u16,
            out: *mut u16,
            len: i32,
        ) -> i32;
        pub(super) fn LoadLibraryExW(name: *const u16, file: isize, flags: u32) -> isize;
        pub(super) fn GetProcAddress(module: isize, name: *const u8) -> *const c_void;
        pub(super) fn FreeLibrary(module: isize) -> i32;
        pub(super) fn GlobalFree(memory: *mut u16) -> *mut u16;
    }

    #[link(name = "ntdll")]
    extern "system" {
        pub(super) fn RtlGetVersion(info: *mut OsVersionInfo) -> i32;
    }

    /// `BrandingFormatString` (winbrand.dll): the format with the system's branding tokens
    /// filled in, in memory the caller frees with `GlobalFree`.
    pub(super) type BrandingFormatString = unsafe extern "system" fn(*const u16) -> *mut u16;
}

/// The local `[year, month, day, hour, minute, second]` of
/// `unix_seconds` in the OS's current time zone, daylight saving included
/// (`SystemTimeToTzSpecificLocalTime`): what `DateTimeOffset.ToLocalTime()` gives for a moment of
/// this year. `None` off Windows or on failure.
#[must_use]
pub fn local_time_fields(unix_seconds: i64) -> Option<[u16; 6]> {
    #[cfg(windows)]
    {
        let local = local_system_time(unix_seconds)?;
        Some([
            local.year,
            local.month,
            local.day,
            local.hour,
            local.minute,
            local.second,
        ])
    }
    #[cfg(not(windows))]
    {
        let _ = unix_seconds;
        None
    }
}

#[cfg(windows)]
fn local_system_time(unix_seconds: i64) -> Option<facts::SystemTime> {
    // FILETIME counts 100-nanosecond ticks from 1601-01-01 UTC.
    let ticks = unix_seconds
        .checked_add(11_644_473_600)?
        .checked_mul(10_000_000)?;
    let ticks = u64::try_from(ticks).ok()?;
    let file_time = facts::FileTime {
        low: ticks as u32,
        high: (ticks >> 32) as u32,
    };
    let mut universal = facts::SystemTime::default();
    let mut local = facts::SystemTime::default();
    // SAFETY: in- and out-parameters of the declared layouts; a null zone is the active one.
    let ok = unsafe {
        facts::FileTimeToSystemTime(&file_time, &mut universal) != 0
            && facts::SystemTimeToTzSpecificLocalTime(std::ptr::null(), &universal, &mut local) != 0
    };
    ok.then_some(local)
}

/// `unix_seconds` as the user's short date (its `yyyy` cut to `yy`, as
/// Java 8's default short `DateFormat`) and short time, joined by a space, in the user's locale
/// with their own settings (`GetLocaleInfoEx`, `GetDateFormatEx`, `GetTimeFormatEx` for the user
/// default locale: the data .NET's `CultureInfo.CurrentCulture` reads). `None` off Windows or on
/// failure.
#[must_use]
pub fn local_short_date_time(unix_seconds: i64) -> Option<String> {
    #[cfg(windows)]
    {
        let local = local_system_time(unix_seconds)?;
        let date_pattern = wide_text(|out, len| {
            // SAFETY: the user default locale (null name) and an out-buffer of `len` units.
            unsafe { facts::GetLocaleInfoEx(std::ptr::null(), facts::LOCALE_SSHORTDATE, out, len) }
        })?
        .replace("yyyy", "yy");
        let time_pattern = wide_text(|out, len| {
            // SAFETY: as above.
            unsafe { facts::GetLocaleInfoEx(std::ptr::null(), facts::LOCALE_SSHORTTIME, out, len) }
        })?;
        let date_pattern = nul_terminated(&date_pattern);
        let time_pattern = nul_terminated(&time_pattern);
        let date = wide_text(|out, len| {
            // SAFETY: a valid SYSTEMTIME, a NUL-terminated picture and an out-buffer of `len`.
            unsafe {
                facts::GetDateFormatEx(
                    std::ptr::null(),
                    0,
                    &local,
                    date_pattern.as_ptr(),
                    out,
                    len,
                    std::ptr::null(),
                )
            }
        })?;
        let time = wide_text(|out, len| {
            // SAFETY: as above.
            unsafe {
                facts::GetTimeFormatEx(std::ptr::null(), 0, &local, time_pattern.as_ptr(), out, len)
            }
        })?;
        Some(format!("{date} {time}"))
    }
    #[cfg(not(windows))]
    {
        let _ = unix_seconds;
        None
    }
}

/// The operating system's name as WMI's
/// `Win32_OperatingSystem.Caption` gives it: "Microsoft " and the system's own long product name
/// (`BrandingFormatString("%WINDOWS_LONG%")`, "Windows 11 Home"), which is right where the
/// registry's product name still says Windows 10. `None` off Windows or on failure.
#[must_use]
pub fn windows_caption() -> Option<String> {
    #[cfg(windows)]
    {
        let library = nul_terminated("winbrand.dll");
        // SAFETY: a NUL-terminated name, loaded from System32 only (never the game folder); the
        // export is checked for null before it is called with a NUL-terminated format, its
        // result is copied and then freed as its documentation asks, and the library released.
        unsafe {
            let module =
                facts::LoadLibraryExW(library.as_ptr(), 0, facts::LOAD_LIBRARY_SEARCH_SYSTEM32);
            if module == 0 {
                return None;
            }
            let export = facts::GetProcAddress(module, c"BrandingFormatString".as_ptr().cast());
            let caption = if export.is_null() {
                None
            } else {
                let branding = std::mem::transmute::<
                    *const std::ffi::c_void,
                    facts::BrandingFormatString,
                >(export);
                let format = nul_terminated("%WINDOWS_LONG%");
                let text = branding(format.as_ptr());
                if text.is_null() {
                    None
                } else {
                    let mut len = 0;
                    while *text.add(len) != 0 && len < 1024 {
                        len += 1;
                    }
                    let long = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
                    facts::GlobalFree(text);
                    let long = long.trim().to_owned();
                    (!long.is_empty()).then(|| {
                        if long.starts_with("Microsoft") {
                            long
                        } else {
                            format!("Microsoft {long}")
                        }
                    })
                }
            };
            facts::FreeLibrary(module);
            caption
        }
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// The running Windows version as WMI's `Win32_OperatingSystem.Version`
/// gives it, `major.minor.build` ("10.0.26200"), from `RtlGetVersion` (which, unlike
/// `GetVersionEx`, does not depend on the executable's compatibility manifest). `None` off
/// Windows or on failure.
#[must_use]
pub fn windows_version() -> Option<String> {
    #[cfg(windows)]
    {
        let mut info = facts::OsVersionInfo {
            size: std::mem::size_of::<facts::OsVersionInfo>() as u32,
            major: 0,
            minor: 0,
            build: 0,
            platform: 0,
            service_pack: [0; 128],
        };
        // SAFETY: an out-parameter whose size field names its size, as the call requires.
        let status = unsafe { facts::RtlGetVersion(&mut info) };
        (status == 0 && info.major > 0)
            .then(|| format!("{}.{}.{}", info.major, info.minor, info.build))
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// `text` as UTF-16 with a terminating NUL.
#[cfg(windows)]
fn nul_terminated(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Text an NLS call writes into a buffer: `fill(buffer, capacity)` returns the units written,
/// counting the terminating NUL, or 0 on failure.
#[cfg(windows)]
fn wide_text(fill: impl FnOnce(*mut u16, i32) -> i32) -> Option<String> {
    let mut buffer = [0_u16; 256];
    let written = fill(buffer.as_mut_ptr(), buffer.len() as i32);
    let len = usize::try_from(written).ok()?.checked_sub(1)?;
    (len > 0 && len < buffer.len()).then(|| String::from_utf16_lossy(&buffer[..len]))
}

/// [`collect_on_thread`] for the calling thread (the game thread when a world is left), and the
/// same on each terrain worker as a background job (they free what the game thread released of
/// theirs only when they run). Never blocks on the workers. `reason` is logged with the time.
pub fn collect_freed_memory(reason: &'static str) {
    if !collect_enabled() {
        return;
    }
    mc_core::profile_scope!("alloc.collect");
    crate::terrain::jobs::collect_worker_heaps();
    if let Some(took) = collect_on_thread() {
        tracing::debug!(
            reason,
            us = took.as_micros() as u64,
            "returned freed allocator pages to the OS"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are this process's and they are live: a touched 64 MiB block raises both the
    /// working set and the private bytes by about that much, and they fall back when it is freed.
    /// (The test binary uses the system allocator, which takes a block this size straight from
    /// the OS and gives it back on free.)
    #[cfg(windows)]
    #[test]
    fn process_memory_follows_a_real_allocation() {
        const SIZE: usize = 64 << 20;
        let before = process_memory().expect("the OS reports this process's memory");
        assert!(before.working_set > 1 << 20, "{before:?}");
        assert!(before.private > 1 << 20, "{before:?}");
        assert!(
            before.working_set < 64 << 30 && before.private < 64 << 30,
            "{before:?}"
        );

        // One cycle: allocate and touch the block, sample, free it, sample. `black_box` keeps an
        // optimised build from deleting the block (only one byte of it is ever read). Other
        // tests allocate and free on other threads meanwhile, so a cycle may be disturbed: a few
        // attempts are allowed, and dead counters fail every one of them.
        let cycle = || {
            let before = process_memory().expect("still readable");
            let block = std::hint::black_box(vec![1_u8; SIZE]);
            let during = process_memory().expect("still readable");
            assert_eq!(std::hint::black_box(&block)[SIZE - 1], 1);
            drop(block);
            let after = process_memory().expect("still readable");
            let grew = during.private >= before.private + (SIZE as u64) * 3 / 4
                && during.working_set >= before.working_set + (SIZE as u64) * 3 / 4;
            let fell = after.private + (SIZE as u64) / 2 <= during.private;
            (
                grew && fell,
                format!("{before:?} -> {during:?} -> {after:?}"),
            )
        };
        let mut seen = Vec::new();
        for _ in 0..5 {
            let (ok, trace) = cycle();
            if ok {
                return;
            }
            seen.push(trace);
        }
        panic!("the counters did not follow a touched 64 MiB block in 5 tries: {seen:#?}");
    }

    #[cfg(windows)]
    #[test]
    fn total_physical_memory_is_the_machines() {
        let total = total_physical_memory().expect("the OS reports its physical memory");
        let process = process_memory().expect("the OS reports this process's memory");
        assert!(total >= 512 << 20, "{total}");
        assert!(total < 1 << 44, "{total}");
        assert!(total > process.working_set, "{total} vs {process:?}");
    }

    /// A mapped file reads exactly like the file, and while it is mapped nobody can open it for
    /// writing (the guarantee behind the `'static` slice). Missing and empty files are `None`,
    /// so the caller reads them the old way.
    #[cfg(windows)]
    #[test]
    fn a_mapped_file_is_the_file_and_stays_unchanged() {
        let dir = std::env::temp_dir().join(format!("mc-w21-mb-map-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("mapped.bin");
        // A few pages, left in the temp folder afterwards: the handle that keeps writers out is
        // held until this test process exits (by design), so the test cannot delete the file.
        let content: Vec<u8> = (0..70_000_u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &content).expect("write the test file");

        let mapped = map_file_for_process(&path);
        if !map_files_enabled() {
            assert!(mapped.is_none());
            return;
        }
        let mapped = mapped.expect("an ordinary file maps");
        assert_eq!(mapped, content.as_slice());
        assert!(
            std::fs::OpenOptions::new().write(true).open(&path).is_err(),
            "a writer must be refused while the file is mapped"
        );
        assert_eq!(std::fs::read(&path).expect("readers are fine"), content);

        assert!(map_file_for_process(&dir.join("missing.bin")).is_none());
        let empty = dir.join("empty.bin");
        std::fs::write(&empty, []).expect("write the empty file");
        assert!(map_file_for_process(&empty).is_none());
    }

    #[cfg(not(windows))]
    #[test]
    fn the_counters_are_windows_only() {
        assert_eq!(process_memory(), None);
        assert_eq!(total_physical_memory(), None);
    }

    /// The collect runs exactly when the binary's allocator is mimalloc and the switch is on, and
    /// changes nothing a caller can see.
    #[test]
    fn collect_runs_only_with_the_allocator_and_keeps_live_data() {
        let kept: Vec<u64> = (0..100_000).collect();
        let ran = collect_on_thread();
        assert_eq!(
            ran.is_some(),
            cfg!(feature = "mimalloc") && collect_enabled()
        );
        collect_freed_memory("test");
        assert!(kept.iter().enumerate().all(|(i, v)| *v == i as u64));
    }
}
