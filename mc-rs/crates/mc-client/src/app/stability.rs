//! Loading the player's own files so that nothing the PC does to them costs the player their
//! settings or their server list.
//!
//! Both loaders never fail, as vanilla's never do, and both keep a copy (`<file>.bad`) of a file
//! they could not use before anything can overwrite it: the client saves `options.txt` on every
//! options change and `servers.dat` at every exit, so a damaged file that was simply replaced by
//! defaults was gone for good. The writers are atomic and flushed
//! ([`mc_core::durable::write_atomic`]), so a crash or power cut no longer makes such files.

use std::path::Path;
use std::time::Duration;

use mc_render::settings::{GameSettings, OptionsFileState, DAMAGED_SUFFIX};
use mc_ui::screens::multiplayer::ServerList;

/// The cycle the menu animation clocks (title screen, mod menu, replay viewer) wrap at.
pub const UI_ANIMATION_CYCLE: Duration = Duration::from_secs(3600);

/// Seconds into a repeating `cycle`, as an `f32` that keeps sub-millisecond precision however
/// long the client has been running: the wrap is done on the integer [`Duration`] before the
/// conversion. The menu clocks used `elapsed.as_secs_f32() % 3600.0`, which converts first, so
/// after 10 hours of uptime they moved in 4 ms steps and after a day in 8 ms steps (visible
/// stutter at 144 Hz and up). Within the first cycle the value is bit-identical to that.
#[must_use]
pub fn cycle_seconds(elapsed: Duration, cycle: Duration) -> f32 {
    let wrapped = elapsed.as_nanos() % cycle.as_nanos().max(1);
    // Below `cycle`, so it fits in `u64` nanoseconds for any cycle shorter than 584 years.
    Duration::from_nanos(u64::try_from(wrapped).unwrap_or(u64::MAX)).as_secs_f32()
}

/// Set when the shell reports the GPU device lost ([`note_device_lost`]); the game then ends
/// with a report.
static DEVICE_LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The shell told the client its GPU device was lost (`ShellApp::device_lost`): the run ends
/// with a crash report, so the exit hook must leave the shutdown to `main`.
pub fn note_device_lost() {
    DEVICE_LOST.store(true, std::sync::atomic::Ordering::Release);
}

impl crate::app::Client {
    /// Whether the game is ending because of a crash, a failed load or a lost GPU device: `main`
    /// then writes a report, which must describe the live client (its world above all) before
    /// the shutdown unloads it. A terrain worker's panic, any panic the hook recorded, a fatal
    /// load error, or a lost device.
    #[must_use]
    pub fn ending_with_a_report(&self) -> bool {
        crate::app::crash::has_crashed()
            || self.fatal_error().is_some()
            || mc_render::terrain::mesh_worker_panics() > 0
            || DEVICE_LOST.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// `options.txt`: its readable lines, the defaults for the rest. java: `GameSettings.loadOptions`.
#[must_use]
pub fn load_options(path: &Path) -> GameSettings {
    let (settings, state) = GameSettings::load_keeping_damaged(path);
    if let OptionsFileState::Damaged { reason, backup } = state {
        tracing::warn!(
            %reason,
            copy_kept = backup.is_some(),
            "options.txt is damaged: its readable lines are used, the rest is at the defaults, \
             and the file was copied to options.txt.bad"
        );
    }
    settings
}

/// `servers.dat`, or an empty list. java: `ServerList.loadServerList`, which logs and starts
/// empty too. A file that exists but cannot be read or parsed is first copied to
/// `servers.dat.bad`: this client writes the list at every exit (vanilla only after an edit), so
/// without the copy the player's damaged list was replaced by an empty one at the first quit.
#[must_use]
pub fn load_server_list(path: &Path) -> ServerList {
    let error = match std::fs::read(path) {
        Ok(bytes) => match ServerList::read(&bytes) {
            Ok(list) => return list,
            Err(error) => error.to_string(),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return ServerList::new(),
        Err(error) => error.to_string(),
    };
    let copy_kept = mc_core::durable::keep_aside(path, DAMAGED_SUFFIX).is_some();
    tracing::warn!(
        %error,
        copy_kept,
        "servers.dat unreadable; starting with an empty list (the file was copied to \
         servers.dat.bad)"
    );
    ServerList::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_ui::screens::multiplayer::ServerEntry;

    fn list_of(names: &[&str]) -> ServerList {
        let mut list = ServerList::new();
        for name in names {
            list.add(ServerEntry::new(*name, format!("{name}.example.net")));
        }
        list
    }

    fn write_list(path: &Path, list: &ServerList) {
        let mc_nbt::Nbt::Compound(root) = list.to_nbt() else {
            unreachable!("a server list is a compound")
        };
        mc_nbt::safe_write(path, &root).expect("write servers.dat");
    }

    #[test]
    fn a_good_server_list_loads_and_makes_no_copy() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        assert!(load_server_list(&path).is_empty(), "missing: a first run");
        write_list(&path, &list_of(&["alpha", "beta"]));
        assert_eq!(load_server_list(&path).len(), 2);
        assert!(!dir.path().join("servers.dat.bad").exists());
    }

    /// The bug: a damaged `servers.dat` loaded as an empty list, and the save at the next exit
    /// replaced it. Now the damaged bytes survive as `servers.dat.bad`, and the exit's save
    /// (simulated here) no longer destroys the only copy.
    #[test]
    fn a_damaged_server_list_is_kept_aside_before_the_exit_save() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        write_list(&path, &list_of(&["alpha", "beta", "gamma"]));
        let mut damaged = std::fs::read(&path).expect("read");
        let cut = damaged.len() / 2;
        damaged.truncate(cut); // what a crash mid-write left behind with the old writer
        std::fs::write(&path, &damaged).expect("damage");

        let loaded = load_server_list(&path);
        assert!(
            loaded.is_empty(),
            "an unreadable list starts empty, as vanilla's does"
        );
        // The exit save, as `ShutdownTarget::save_server_list` performs it.
        write_list(&path, &loaded);

        let backup = dir.path().join("servers.dat.bad");
        assert_eq!(std::fs::read(&backup).expect("copy kept"), damaged);
    }

    #[test]
    fn options_with_a_foreign_byte_keep_every_other_option() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("options.txt");
        std::fs::write(
            &path,
            b"guiScale:3\nlastServer:\xE9\nmouseSensitivity:0.9\n",
        )
        .expect("w");
        let settings = load_options(&path);
        assert_eq!(settings.gui_scale, 3);
        assert_eq!(settings.mouse_sensitivity, 0.9);
        assert!(!dir.path().join("options.txt.bad").exists(), "not damage");
    }

    /// Long uptime: ten hours and one millisecond in, the old clock lost the millisecond (36,000 s
    /// is past the point where an `f32` can hold milliseconds); the wrapped one keeps it.
    #[test]
    fn menu_clocks_keep_their_precision_after_hours() {
        let late = Duration::from_secs(36_000) + Duration::from_millis(1);
        let old = late.as_secs_f32() % 3600.0;
        assert_eq!(
            old, 0.0,
            "negative control: the old clock dropped the millisecond"
        );
        let new = cycle_seconds(late, UI_ANIMATION_CYCLE);
        assert!((new - 0.001).abs() < 1e-6, "{new}");
        // A day in, a 2 ms frame still moves the clock by 2 ms (to within one f32 step at
        // 1234 s, 0.12 ms). The old clock moved by 0 or 7.8 ms.
        let day = Duration::from_secs(86_400 + 1234);
        let next = day + Duration::from_millis(2);
        let step = cycle_seconds(next, UI_ANIMATION_CYCLE) - cycle_seconds(day, UI_ANIMATION_CYCLE);
        assert!((step - 0.002).abs() < 2.5e-4, "{step}");
        let old_step = next.as_secs_f32() % 3600.0 - day.as_secs_f32() % 3600.0;
        assert!(
            (old_step - 0.002).abs() > 1e-3,
            "negative control: {old_step}"
        );
    }

    /// Within the first cycle nothing changes, bit for bit.
    #[test]
    fn menu_clocks_are_unchanged_in_the_first_hour() {
        for millis in [0_u64, 1, 16, 999, 61_234, 1_799_999, 3_599_999] {
            let elapsed = Duration::from_millis(millis) + Duration::from_nanos(123);
            assert_eq!(
                cycle_seconds(elapsed, UI_ANIMATION_CYCLE).to_bits(),
                (elapsed.as_secs_f32() % 3600.0).to_bits(),
                "{elapsed:?}"
            );
        }
    }

    #[test]
    fn options_full_of_zeros_are_kept_aside() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("options.txt");
        std::fs::write(&path, [0_u8; 512]).expect("zeros");
        let settings = load_options(&path);
        assert_eq!(settings.gui_scale, GameSettings::default().gui_scale);
        assert!(dir.path().join("options.txt.bad").exists());
    }
}
