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

/// A brand-new game folder, one with nothing at all at
/// `servers.dat`, gets the list a new player starts with ([`ServerList::for_new_player`]), written
/// at once: from the first launch it is an ordinary entry the player can edit, move or delete, and
/// nothing ever adds it again. Whatever is already at the path is the player's and is left exactly
/// as it is: a list, an empty list, a damaged file, a folder or a link. So is a path whose state
/// cannot be read. Not vanilla, whose first list is empty.
///
/// The game binary's start calls this once (`Client::new_deferred`), before the list is loaded and
/// before anything can save it. Returns whether the file was written.
pub fn seed_new_player_server_list(path: &Path) -> bool {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        _ => return false,
    }
    if let Some(folder) = path
        .parent()
        .filter(|folder| !folder.as_os_str().is_empty())
    {
        if let Err(error) = std::fs::create_dir_all(folder) {
            tracing::warn!(%error, "could not create the game folder for the first servers.dat");
            return false;
        }
    }
    let mc_nbt::Nbt::Compound(root) = ServerList::for_new_player().to_nbt() else {
        unreachable!("a server list is a compound")
    };
    match mc_nbt::safe_write(path, &root) {
        Ok(()) => {
            tracing::info!("new game folder: servers.dat starts with the default server");
            true
        }
        Err(error) => {
            tracing::warn!(%error, "could not write the first servers.dat");
            false
        }
    }
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

    /// A folder with no `servers.dat` gets the one-entry list, on
    /// disk at once, and the load that follows reads it; the next start finds a file and writes
    /// nothing.
    #[test]
    fn a_new_game_folder_starts_with_the_default_server_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        assert!(seed_new_player_server_list(&path), "a new folder is seeded");
        let bytes = std::fs::read(&path).expect("written at once");
        let saved = ServerList::read(&bytes).expect("a valid servers.dat");
        assert_eq!(saved, ServerList::for_new_player());
        assert_eq!(saved.get(0).map(|e| e.name.as_str()), Some("Mega Skywars"));
        assert_eq!(load_server_list(&path), saved);
        assert!(!dir.path().join("servers.dat_tmp").exists());
        assert!(!seed_new_player_server_list(&path), "seeded once only");
        assert_eq!(std::fs::read(&path).expect("read"), bytes);
    }

    /// Any file already there is the player's list, even an empty one: never seeded, not a byte
    /// changed.
    #[test]
    fn an_existing_server_list_is_never_seeded() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        for list in [ServerList::new(), list_of(&["alpha", "beta"])] {
            write_list(&path, &list);
            let before = std::fs::read(&path).expect("read");
            assert!(
                !seed_new_player_server_list(&path),
                "{} entries",
                list.len()
            );
            assert_eq!(std::fs::read(&path).expect("read"), before);
            assert_eq!(load_server_list(&path), list);
        }
    }

    /// The player deletes the default server (the list screen writes the shorter list at once):
    /// every later start loads the list as it is.
    #[test]
    fn a_deleted_default_server_stays_deleted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        assert!(seed_new_player_server_list(&path));
        let mut list = load_server_list(&path);
        assert!(list.remove(0).is_some());
        write_list(&path, &list);
        for _ in 0..2 {
            assert!(!seed_new_player_server_list(&path));
            assert!(load_server_list(&path).is_empty(), "not added back");
        }
    }

    /// A damaged file is not a new folder: it is not replaced by the default list, on this start
    /// or the next; the load keeps its copy aside as before.
    #[test]
    fn a_damaged_server_list_is_not_overwritten_with_the_default() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        for damaged in [b"".to_vec(), vec![0xDE, 0xAD, 0xBE, 0xEF], {
            write_list(&path, &list_of(&["alpha", "beta", "gamma"]));
            let mut bytes = std::fs::read(&path).expect("read");
            bytes.truncate(bytes.len() / 2);
            bytes
        }] {
            std::fs::write(&path, &damaged).expect("damage");
            assert!(!seed_new_player_server_list(&path));
            assert_eq!(std::fs::read(&path).expect("read"), damaged, "untouched");
            assert!(load_server_list(&path).is_empty());
            let backup = dir.path().join("servers.dat.bad");
            assert_eq!(std::fs::read(&backup).expect("copy kept"), damaged);
            assert!(!seed_new_player_server_list(&path), "still not new");
            assert_eq!(std::fs::read(&path).expect("read"), damaged);
        }
    }

    /// Something that is not a file still means the folder is not new.
    #[test]
    fn a_folder_named_servers_dat_is_left_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        std::fs::create_dir(&path).expect("folder");
        assert!(!seed_new_player_server_list(&path));
        assert!(path.is_dir());
    }

    /// A first start into a game folder that does not exist yet creates it for the list.
    #[test]
    fn a_missing_game_folder_is_created_for_the_first_list() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("new").join("game").join("servers.dat");
        assert!(seed_new_player_server_list(&path));
        assert_eq!(load_server_list(&path), ServerList::for_new_player());
    }

    /// The game binary's start (`Client::new_deferred`) seeds a new folder before it loads the
    /// list, so the list the client holds has the entry from the start, with nothing pinging or
    /// saving first. A player who deletes it never sees it again, and a folder with a list (even
    /// an empty one) keeps it. Skips without the vanilla jar.
    #[test]
    fn the_game_binarys_start_seeds_only_a_new_folder() {
        use crate::app::config::{ClientConfig, LoginMode};
        use mc_ui::screens::Action;

        let boot = |dir: &Path| {
            let config = ClientConfig {
                game_dir: dir.to_path_buf(),
                login: LoginMode::Offline("Dev".to_owned()),
                ..ClientConfig::default()
            }
            .with_discovered_assets();
            if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
                eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
                return None;
            }
            Some(crate::app::Client::new_deferred(config).expect("the boot assets load"))
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.dat");
        let Some(mut client) = boot(dir.path()) else {
            return;
        };
        assert_eq!(client.servers, ServerList::for_new_player());
        assert_eq!(load_server_list(&path), client.servers, "on disk already");

        // The player deletes it from the server list, which writes the list at once.
        client.handle_action(Action::OpenMultiplayer);
        let crate::app::driver::Ui::Multiplayer(screen) = &mut client.ui else {
            panic!("the server list is open")
        };
        screen.select(0);
        assert!(screen.delete_selected().is_some());
        client.handle_action(Action::SaveServers);
        drop(client);
        let client = boot(dir.path()).expect("boots again");
        assert!(client.servers.is_empty(), "never added back");
        assert!(load_server_list(&path).is_empty());

        // A folder that already has a list, an empty one included, keeps it.
        let other = tempfile::tempdir().expect("tempdir");
        let path = other.path().join("servers.dat");
        write_list(&path, &ServerList::new());
        let before = std::fs::read(&path).expect("read");
        let client = boot(other.path()).expect("boots");
        assert!(client.servers.is_empty());
        assert_eq!(std::fs::read(&path).expect("read"), before);
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
