//! Every `unsafe` call in the crate lives here (the crate root denies `unsafe_code` elsewhere):
//! - the worker thread's membership of the multithreaded COM apartment;
//! - the player's Windows audio-session volume, through Core Audio: every active output device
//!   (`IMMDeviceEnumerator`), its sessions (`IAudioSessionManager2`), the ones whose process is the
//!   player's (`IAudioSessionControl2::GetProcessId`), and their `ISimpleAudioVolume`;
//! - naming a session's process (`QueryFullProcessImageNameW`).
//!
//! The Core Audio bindings are `unsafe fn`s because they are raw COM calls. The interfaces used
//! here are all created on the worker thread, used only there, and released before the thread
//! leaves the apartment (see `gsmtc::WindowsBackend`'s field order).
//!
//! This is the level shown in the Windows volume mixer for the player, which is not the player's
//! own in-app slider (Spotify keeps that separately). Setting it is what "Spotify's volume" can
//! mean without a Spotify login.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use windows::core::{Interface, PWSTR};
use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Media::Audio::{
    eRender, AudioSessionState, AudioSessionStateActive, AudioSessionStateExpired,
    IAudioSessionControl, IAudioSessionControl2, IAudioSessionManager2, IMMDeviceEnumerator,
    ISimpleAudioVolume, MMDeviceEnumerator, DEVICE_STATE_ACTIVE,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED,
};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};

use crate::apps;

/// While no session of the player is known, look again after this long, doubling with each
/// empty search up to `RESCAN_INTERVAL` (a player that streams to another device through Spotify
/// Connect has no session here for as long as that lasts). `VolumeFinder::nudge` starts over.
const SEARCH_INTERVAL: Duration = Duration::from_secs(3);

/// While sessions are known, look again this often anyway (a new output device gets a new one).
const RESCAN_INTERVAL: Duration = Duration::from_secs(30);

/// The current thread's membership of the multithreaded COM apartment, left on drop.
pub(crate) struct ComApartment {
    entered: bool,
    /// COM membership belongs to a thread: keep this on the thread that entered (`!Send`).
    _same_thread: PhantomData<*const ()>,
}

impl ComApartment {
    /// Enters the multithreaded apartment. WinRT works without this too (it enters on demand),
    /// but `CoCreateInstance` for Core Audio needs it.
    pub(crate) fn enter_mta() -> Self {
        // SAFETY: no memory-safety preconditions; the reserved argument must be null (`None`).
        // Success (S_OK, or S_FALSE if this thread was already in the MTA) is balanced by exactly
        // one CoUninitialize in `drop`, on this same thread, since the type is `!Send`.
        let result = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if let Err(error) = result.ok() {
            tracing::warn!("COM init failed on the media worker: {error}");
        }
        Self {
            entered: result.is_ok(),
            _same_thread: PhantomData,
        }
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.entered {
            // SAFETY: balances the successful CoInitializeEx in `enter_mta`, on the same thread.
            unsafe { CoUninitialize() };
        }
    }
}

/// One audio session of the player's process.
struct AppSession {
    control: IAudioSessionControl,
    volume: ISimpleAudioVolume,
}

/// Finds the followed player's audio sessions and reads or sets their volume.
#[derive(Default)]
pub(crate) struct VolumeFinder {
    enumerator: Option<IMMDeviceEnumerator>,
    /// The executable whose sessions are wanted (lowercase), from the player's app id.
    exe: Option<String>,
    /// Its sessions found by the last search, the one playing sound first.
    sessions: Vec<AppSession>,
    last_search: Option<Instant>,
    /// Empty searches in a row (see `SEARCH_INTERVAL`).
    misses: u32,
}

impl VolumeFinder {
    /// The volume of the player's audio session, `0.0..=1.0`. `None` when it has none yet (a
    /// player gets one when it first plays sound) or the app id names no process.
    pub(crate) fn level(&mut self, app_id: &str) -> Option<f32> {
        self.update(app_id);
        let session = self.sessions.first()?;
        // SAFETY: a live interface obtained on this thread; the call only writes the returned
        // value.
        unsafe { session.volume.GetMasterVolume() }.ok()
    }

    /// Sets the volume of every audio session of the player (normally one), so they stay
    /// together. Returns whether any accepted it.
    pub(crate) fn set_level(&mut self, app_id: &str, level: f32) -> bool {
        if !level.is_finite() {
            return false;
        }
        let level = level.clamp(0.0, 1.0);
        self.update(app_id);
        let mut applied = false;
        for session in &self.sessions {
            // SAFETY: a live interface obtained on this thread; the level is in 0..=1 as the API
            // requires, and a null event-context pointer is allowed (no context).
            let set = unsafe { session.volume.SetMasterVolume(level, std::ptr::null()) };
            applied |= set.is_ok();
        }
        applied
    }

    /// Something happened that may have given the player an audio session (it started playing,
    /// or announced new media): if none is known, drop back to the shortest wait, so the next
    /// search comes at most `SEARCH_INTERVAL` after the last one. However often this is called,
    /// searches never come closer together than that.
    pub(crate) fn nudge(&mut self) {
        if self.sessions.is_empty() {
            self.misses = 0;
        }
    }

    fn update(&mut self, app_id: &str) {
        let exe = apps::audio_exe_for(app_id);
        if exe != self.exe {
            self.exe = exe;
            self.sessions.clear();
            self.last_search = None;
            self.misses = 0;
        }
        // Drop sessions that ended (the player restarted, or the output device went away).
        self.sessions.retain(|session| !expired(&session.control));
        let now = Instant::now();
        if self
            .last_search
            .is_some_and(|at| now < at + search_interval(!self.sessions.is_empty(), self.misses))
        {
            return;
        }
        let Some(exe) = self.exe.clone() else {
            return;
        };
        self.last_search = Some(now);
        self.sessions = self.search(&exe);
        self.misses = if self.sessions.is_empty() {
            self.misses.saturating_add(1)
        } else {
            0
        };
    }

    fn enumerator(&mut self) -> Option<&IMMDeviceEnumerator> {
        if self.enumerator.is_none() {
            // SAFETY: this thread is in the COM apartment (`ComApartment`); the class id is the
            // documented MMDeviceEnumerator and the call only writes the returned interface.
            let created = unsafe {
                CoCreateInstance::<_, IMMDeviceEnumerator>(&MMDeviceEnumerator, None, CLSCTX_ALL)
            };
            match created {
                Ok(enumerator) => self.enumerator = Some(enumerator),
                Err(error) => tracing::debug!("no audio device enumerator: {error}"),
            }
        }
        self.enumerator.as_ref()
    }

    /// The live sessions of `exe` on every active output device, the one playing sound first.
    fn search(&mut self, exe: &str) -> Vec<AppSession> {
        let mut found = Vec::new();
        let Some(enumerator) = self.enumerator() else {
            return found;
        };
        // SAFETY (every Core Audio call in this function): the interfaces are live and were
        // obtained on this thread; each call only writes its out-value, which the bindings return.
        let Ok(devices) = (unsafe { enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE) })
        else {
            return found;
        };
        // Process names, so each process is opened once per search.
        let mut is_player: HashMap<u32, bool> = HashMap::new();
        // SAFETY: as above.
        let device_count = unsafe { devices.GetCount() }.unwrap_or(0);
        for device_index in 0..device_count {
            // SAFETY: as above.
            let Ok(device) = (unsafe { devices.Item(device_index) }) else {
                continue;
            };
            // SAFETY: as above; null activation parameters are what IAudioSessionManager2 takes.
            let activated = unsafe { device.Activate::<IAudioSessionManager2>(CLSCTX_ALL, None) };
            let Ok(manager) = activated else {
                continue;
            };
            // SAFETY: as above.
            let Ok(list) = (unsafe { manager.GetSessionEnumerator() }) else {
                continue;
            };
            // SAFETY: as above.
            let session_count = unsafe { list.GetCount() }.unwrap_or(0);
            for session_index in 0..session_count {
                // SAFETY: as above; the index is below the count just read.
                let Ok(control) = (unsafe { list.GetSession(session_index) }) else {
                    continue;
                };
                let Ok(control2) = control.cast::<IAudioSessionControl2>() else {
                    continue;
                };
                // SAFETY: as above. A session spanning processes reports the one that made it.
                let Ok(pid) = (unsafe { control2.GetProcessId() }) else {
                    continue;
                };
                // Pid 0 is the system-sounds session.
                if pid == 0 || expired(&control) {
                    continue;
                }
                let matches = *is_player.entry(pid).or_insert_with(|| {
                    process_exe_name(pid).is_some_and(|name| name.eq_ignore_ascii_case(exe))
                });
                if !matches {
                    continue;
                }
                if let Ok(volume) = control.cast::<ISimpleAudioVolume>() {
                    found.push(AppSession { control, volume });
                }
            }
        }
        found.sort_by_key(|session| state(&session.control) != Some(AudioSessionStateActive));
        found
    }
}

/// The wait before the next search: `RESCAN_INTERVAL` while sessions are known; otherwise
/// `SEARCH_INTERVAL` after the first empty search, doubling with each further one up to
/// `RESCAN_INTERVAL`.
fn search_interval(found: bool, misses: u32) -> Duration {
    if found {
        return RESCAN_INTERVAL;
    }
    let doublings = misses.saturating_sub(1).min(4);
    SEARCH_INTERVAL
        .saturating_mul(1 << doublings)
        .min(RESCAN_INTERVAL)
}

fn state(control: &IAudioSessionControl) -> Option<AudioSessionState> {
    // SAFETY: a live interface obtained on this thread; the call only writes the returned state.
    unsafe { control.GetState() }.ok()
}

/// A session that ended, or can no longer be asked.
fn expired(control: &IAudioSessionControl) -> bool {
    state(control).is_none_or(|state| state == AudioSessionStateExpired)
}

/// The executable file name of process `pid` (for example `Spotify.exe`).
fn process_exe_name(pid: u32) -> Option<String> {
    // SAFETY: no memory-safety preconditions; the handle it returns is closed below.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut path = [0u16; 1024];
    let mut length = path.len() as u32;
    let buffer = PWSTR(path.as_mut_ptr());
    let format = PROCESS_NAME_WIN32;
    // SAFETY: `buffer` points at `path`, valid for writes of `length` UTF-16 units and alive past
    // the call, which writes at most that many and stores the count written (without the
    // terminator) in `length`.
    let named = unsafe { QueryFullProcessImageNameW(handle, format, buffer, &mut length) };
    // SAFETY: `handle` is the live handle opened above and is closed exactly once.
    let _ = unsafe { CloseHandle(handle) };
    named.ok()?;
    let path = String::from_utf16_lossy(path.get(..length as usize)?);
    path.rsplit('\\').next().map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read-only: names this test process, which proves the pid -> executable step that ties an
    /// audio session to the player.
    #[test]
    fn the_current_process_is_named_by_its_executable() {
        let name = process_exe_name(std::process::id()).expect("own process");
        let own = std::env::current_exe().expect("current exe");
        let own = own
            .file_name()
            .and_then(|name| name.to_str())
            .expect("file name");
        assert!(name.eq_ignore_ascii_case(own), "{name} vs {own}");
        assert_eq!(
            process_exe_name(0),
            None,
            "the idle process cannot be opened"
        );
    }

    /// **Read-only** diagnostic: lists every audio session on every active output device with its
    /// process, state and volume. Never changes anything. Run with
    /// `cargo test -p mc-media --lib ffi::tests::list_audio_sessions -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads this PC's real audio sessions (read-only)"]
    fn list_audio_sessions() {
        let _apartment = ComApartment::enter_mta();
        let mut finder = VolumeFinder::default();
        let enumerator = finder.enumerator().expect("enumerator").clone();
        // SAFETY: as in `VolumeFinder::search`: live interfaces from this thread, out-values only.
        let devices = unsafe { enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE) }
            .expect("devices");
        // SAFETY: as above.
        let device_count = unsafe { devices.GetCount() }.unwrap_or(0);
        println!("{device_count} active output device(s)");
        for device_index in 0..device_count {
            // SAFETY: as above.
            let device = unsafe { devices.Item(device_index) }.expect("device");
            // SAFETY: as above.
            let manager = unsafe { device.Activate::<IAudioSessionManager2>(CLSCTX_ALL, None) }
                .expect("session manager");
            // SAFETY: as above.
            let list = unsafe { manager.GetSessionEnumerator() }.expect("sessions");
            // SAFETY: as above.
            let session_count = unsafe { list.GetCount() }.unwrap_or(0);
            println!("device {device_index}: {session_count} session(s)");
            for session_index in 0..session_count {
                // SAFETY: as above.
                let control = unsafe { list.GetSession(session_index) }.expect("session");
                let control2 = control.cast::<IAudioSessionControl2>().expect("control2");
                // SAFETY: as above.
                let pid = unsafe { control2.GetProcessId() };
                let name = pid.as_ref().ok().and_then(|&pid| process_exe_name(pid));
                let volume = control.cast::<ISimpleAudioVolume>().ok();
                // SAFETY: as above.
                let level = volume.and_then(|volume| unsafe { volume.GetMasterVolume() }.ok());
                println!(
                    "  pid {pid:?} exe {name:?} state {:?} volume {level:?}",
                    state(&control).map(|state| state.0)
                );
            }
        }
        println!(
            "Spotify.exe level via the finder: {:?}",
            finder.level("Spotify.exe")
        );
        let rounds = 20;
        let started = Instant::now();
        for _ in 0..rounds {
            finder.last_search = None;
            let _ = finder.level("Spotify.exe");
        }
        println!(
            "one session search (enumerator kept): {:?}",
            started.elapsed() / rounds
        );
    }

    #[test]
    fn searches_back_off_while_the_player_has_no_audio_session() {
        let seconds = |found, misses| search_interval(found, misses).as_secs();
        assert_eq!(seconds(true, 0), 30, "known sessions: an occasional rescan");
        assert_eq!(
            [0, 1, 2, 3, 4, 5, 50].map(|misses| seconds(false, misses)),
            [3, 3, 6, 12, 24, 30, 30]
        );
    }

    /// Read-only: a COM apartment can be entered and left on a fresh thread, and a lookup for an
    /// app id that names no process finds no sessions without touching any device.
    #[test]
    fn an_app_id_without_a_process_reports_no_volume() {
        std::thread::spawn(|| {
            let _apartment = ComApartment::enter_mta();
            let mut finder = VolumeFinder::default();
            assert_eq!(finder.level("308046B0AF4A39CB"), None);
            assert!(finder.enumerator.is_none(), "no device was enumerated");
        })
        .join()
        .expect("thread");
    }
}
