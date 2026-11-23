//! The Bookmark key, and when "Record every game" splits.
//!
//! * **Bookmark key** (the `replay` mod's `key_bookmark`, unbound by default). Pressed in game
//!   while a recording runs, it drops a marker labelled [`mc_replay::BOOKMARK_LABEL`] at that
//!   moment: in the footer's metadata (what the replay list counts, without opening a block) and
//!   as a marker record (tag 3), so a crash keeps it (`Library::repair_unfinished` takes it
//!   back). While the clip key's session buffer runs, the press is kept too, and a clip saved
//!   later carries the bookmarks inside its window. A small line beside the recording dot
//!   confirms it for a moment: "Bookmark 3", "Bookmark kept for clips" or "Not recording".
//! * **Splitting.** "Record every game" makes a file per game by splitting at world changes.
//!   A world left after less than [`SPLIT_WORLD_MIN`] (a lobby hop: a network passes you through
//!   lobbies for a few seconds at a time) no longer makes a file of its own: it stays at the
//!   start of the file of the world that follows. The user's library had ten 1-14 s files of
//!   lobby hops among 23.
//!
//! Cost: nothing per frame while no confirmation shows (an `Option` test in the HUD hook); a
//! bookmark is one marker record. The confirmation's text is made once, at the press.

use std::time::{Duration, Instant};

use mc_replay::{Marker, ReplayTime, StreamId, BOOKMARK_LABEL};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::widget::DrawList;

use super::super::super::Client;
use super::{Conn, RecState, MIC_RING_SPAN};

/// "Record every game" splits at a world change only when the world being left lasted this
/// long; a shorter one (a lobby hop) stays in the file of the world that follows.
pub(super) const SPLIT_WORLD_MIN: ReplayTime = 20_000_000;

/// The confirmation shows this long, then fades for [`FLASH_FADE`].
const FLASH_SHOW: Duration = Duration::from_millis(1500);
const FLASH_FADE: Duration = Duration::from_millis(400);

/// What the Bookmark key's last press did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FlashKind {
    /// A bookmark went into the recording.
    Recording,
    /// No recording runs, but the clip key's buffer does: kept for clips.
    ForClips,
    /// Nothing records: nothing was kept.
    NotRecording,
}

/// The confirmation of the Bookmark key's last press.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Flash {
    pub(super) kind: FlashKind,
    /// Made once, at the press ("Bookmark 3").
    pub(super) text: String,
}

impl RecState {
    /// The Bookmark key, pressed at `now` (real time; the confirmation's clock).
    pub(super) fn bookmark_key(&mut self, now: Instant) {
        if self.blocked {
            return;
        }
        let flash = match self.conn.as_deref_mut() {
            Some(conn) => bookmark(conn),
            None => Flash {
                kind: FlashKind::NotRecording,
                text: "Not recording".to_owned(),
            },
        };
        tracing::debug!(text = %flash.text, "replay bookmark key");
        self.bookmark_flash = Some((flash, now));
    }
}

/// Drop a bookmark into whatever records on `conn`, at this frame's time.
fn bookmark(conn: &mut Conn) -> Flash {
    let now = conn.now;
    let mut flash = Flash {
        kind: FlashKind::NotRecording,
        text: "Not recording".to_owned(),
    };
    if conn.buffer.is_some() {
        // Only the last few minutes can end up in a clip (Clip length's maximum).
        conn.bookmarks
            .retain(|&time| now.saturating_sub(time) <= MIC_RING_SPAN);
        conn.bookmarks.push(now);
        flash = Flash {
            kind: FlashKind::ForClips,
            text: "Bookmark kept for clips".to_owned(),
        };
    }
    if let Some(recording) = conn.recording.as_mut() {
        let time = recording.time(now);
        recording.markers.push(Marker {
            time,
            label: BOOKMARK_LABEL.to_owned(),
        });
        recording
            .writer
            .push(time, StreamId::Marker, BOOKMARK_LABEL.as_bytes());
        let count = recording
            .markers
            .iter()
            .filter(|marker| marker.label == BOOKMARK_LABEL)
            .count();
        flash = Flash {
            kind: FlashKind::Recording,
            text: format!("Bookmark {count}"),
        };
    }
    flash
}

/// A replay's file name for the log: never the folder it is in (the game folder's path holds
/// the Windows account name, and people share their logs).
pub(super) fn log_name(path: &std::path::Path) -> std::borrow::Cow<'_, str> {
    path.file_name()
        .map_or(std::borrow::Cow::Borrowed("?"), |name| {
            name.to_string_lossy()
        })
}

/// The bookmarks inside a clip's `window` (session times; the clip writer retimes them).
pub(super) fn clip_bookmarks(conn: &Conn, window: (ReplayTime, ReplayTime)) -> Vec<Marker> {
    conn.bookmarks
        .iter()
        .filter(|&&time| time >= window.0 && time <= window.1)
        .map(|&time| Marker {
            time,
            label: BOOKMARK_LABEL.to_owned(),
        })
        .collect()
}

impl Client {
    /// The Bookmark key's confirmation, top right beside the recording dot: the timeline's amber
    /// diamond and "Bookmark 3" for a second and a half, then a short fade. Not under F3 (its
    /// right column is there); hidden with F1 like the rest of the HUD (this draws from the HUD).
    pub(super) fn rec_draw_bookmark_flash(&mut self, list: &mut DrawList, ctx: &ScreenCtx) {
        let Some((flash, at)) = self.rec.bookmark_flash.as_mut() else {
            return;
        };
        let age = at.elapsed();
        if age >= FLASH_SHOW + FLASH_FADE {
            self.rec.bookmark_flash = None;
            return;
        }
        if self.debug.visible {
            return;
        }
        let fade = age.saturating_sub(FLASH_SHOW).as_secs_f32() / FLASH_FADE.as_secs_f32();
        let alpha = ((1.0 - fade).clamp(0.0, 1.0) * 255.0) as u32;
        // The font treats an alpha under 4 as opaque.
        if alpha < 8 {
            return;
        }
        let kind = flash.kind;
        // Borrowed out and back, so drawing needs no copy of the text.
        let text = std::mem::take(&mut flash.text);
        let width = self.font.string_width(&text);
        // Right-aligned just left of the recording dot (`rec_draw_dot`: 5 px at width - 9).
        let right = ctx.width - 12;
        let x = right - width;
        let y = 3;
        let text_rgb = match kind {
            FlashKind::NotRecording => 0x00A0_A0A0,
            FlashKind::Recording | FlashKind::ForClips => 0x00F0_F0F0,
        };
        if kind != FlashKind::NotRecording {
            // A 5x5 diamond, like the markers on the replay timeline.
            let amber = (alpha << 24) | 0x00FF_B300;
            let (dx, dy) = (x - 8, y + 1);
            list.rect(dx + 2, dy, dx + 3, dy + 1, amber);
            list.rect(dx + 1, dy + 1, dx + 4, dy + 2, amber);
            list.rect(dx, dy + 2, dx + 5, dy + 3, amber);
            list.rect(dx + 1, dy + 3, dx + 4, dy + 4, amber);
            list.rect(dx + 2, dy + 4, dx + 3, dy + 5, amber);
        }
        let color = ((alpha << 24) | text_rgb) as i32;
        self.font
            .draw_string_with_shadow(list, &text, x as f32, y as f32, color);
        if let Some((flash, _)) = self.rec.bookmark_flash.as_mut() {
            flash.text = text;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use mc_mods::mods::replay::ids;
    use mc_parity::capture::{Capture, CaptureMeta, Direction, Recorder};
    use mc_protocol::play::{s2c, S2CPlay};
    use mc_protocol::{PacketBuffer, State};
    use mc_replay::{Library, LibraryEntry, Reader, ReplayKind, StreamId, BOOKMARK_LABEL};

    use super::super::super::super::Client;
    use super::FlashKind;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{NetHandle, ReplaySource};
    use crate::session::{JoinSettings, Session};

    /// An offline client with its game dir in a temp folder (needs `MC_JARS_DIR`).
    fn client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    fn open(client: &mut Client, capture: &Capture, path: &Path, step_micros: u64) -> usize {
        let source = ReplaySource::from_capture_keeping_bytes(
            capture,
            path.to_path_buf(),
            step_micros,
            true,
        )
        .expect("a replay source");
        let frames = (source.summary().span_micros / step_micros) as usize + 20;
        client.session = Some(Session::new(
            NetHandle::replay(source),
            JoinSettings::default(),
        ));
        client.current_server_address = Some("skywars.test".to_owned());
        frames
    }

    fn drive(
        client: &mut Client,
        frames: usize,
        step: Duration,
        mut each: impl FnMut(&mut Client, usize),
    ) {
        let start = Instant::now();
        client.rec.sync(&client.mods.settings);
        for frame in 0..frames {
            client.rec_frame(start + step * frame as u32);
            client.poll_in_place();
            each(client, frame);
        }
    }

    /// Leave the session and wait until every recording and clip is on disk.
    fn finish(client: &mut Client) {
        client.session = None;
        client.rec_frame(Instant::now() + Duration::from_secs(120));
        client.rec_shutdown();
    }

    fn frame(packet: &S2CPlay) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    fn markers_recorded(path: &Path) -> Vec<(u64, Vec<u8>)> {
        Reader::open(path)
            .expect("opens")
            .records()
            .map(|r| r.expect("record"))
            .filter(|r| r.stream == StreamId::Marker)
            .map(|r| (r.time, r.payload))
            .collect()
    }

    fn in_its_date_folder(library: &Library, entry: &LibraryEntry) -> bool {
        let root = if entry.is_clip {
            library.clips_dir()
        } else {
            library.replays_dir()
        };
        entry.path.parent() == Some(&mc_replay::date_folder(root, entry.meta.title_unix_ms()))
    }

    /// The Bookmark key on the SkyWars scene with "Record every game" and the clip key: two
    /// presses become two bookmarks in the recording (footer and marker records), the clip
    /// key's later clip carries the one inside its window, the confirmation counts them, and
    /// both files land in their date folders.
    #[test]
    fn bookmarks_go_into_the_recording_and_the_clips_inside_their_window() {
        let Some((mut client, dir)) = client() else {
            return;
        };
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../optimization/scratch/skywars-scene-2026-11-05/captures")
            .join("sw-mega-doubles-mid.mccap");
        let capture = Capture::read_file(&path).expect("the SkyWars bench capture");
        client.mods.settings.set_enabled("replay", true);
        client.mods.settings.set_key("replay", ids::KEY_CLIP, 88); // F12
        client
            .mods
            .settings
            .set_key("replay", ids::KEY_BOOKMARK, 48); // B
        let frames = open(&mut client, &capture, &path, 50_000);
        assert!(frames > 1_550, "the scene is about 80 s");
        let mut flashes = Vec::new();
        drive(
            &mut client,
            frames,
            Duration::from_millis(50),
            |client, frame| {
                // 20 s and 50 s in; the clip key at 75 s keeps 45-75 s.
                if frame == 400 || frame == 1_000 {
                    client.rec.bookmark_key(Instant::now());
                    let flash = client.rec.bookmark_flash.as_ref().map(|(f, _)| f.clone());
                    flashes.push(flash.expect("a confirmation"));
                }
                if frame == 1_500 {
                    client.rec.clip_key();
                }
            },
        );
        // The confirmation draws (beside the dot), then goes.
        let ctx = mc_ui::screen_stack::ScreenCtx::new(client.resolution(854, 480), 0.0);
        let mut list = mc_ui::widget::DrawList::new();
        client.rec_draw_bookmark_flash(&mut list, &ctx);
        assert!(!list.vertices().is_empty(), "the confirmation is drawn");
        if let Some((_, at)) = client.rec.bookmark_flash.as_mut() {
            *at -= Duration::from_secs(3);
        }
        let mut list = mc_ui::widget::DrawList::new();
        client.rec_draw_bookmark_flash(&mut list, &ctx);
        assert!(list.vertices().is_empty() && client.rec.bookmark_flash.is_none());
        finish(&mut client);

        assert_eq!(
            flashes
                .iter()
                .map(|f| (f.kind, f.text.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (FlashKind::Recording, "Bookmark 1"),
                (FlashKind::Recording, "Bookmark 2")
            ]
        );
        let library = Library::new(dir.path());
        let full = library.list_dir(false);
        assert_eq!(full.len(), 1, "{full:?}");
        let recording = &full[0];
        assert!(
            in_its_date_folder(&library, recording),
            "{}",
            recording.path.display()
        );
        let meta = &recording.meta;
        assert_eq!(meta.bookmark_count(), 2);
        let bookmarks: Vec<u64> = meta
            .markers
            .iter()
            .filter(|m| m.label == BOOKMARK_LABEL)
            .map(|m| m.time)
            .collect();
        // File time = session time here (recording from the first frame), 50 ms per frame.
        let near = |t: u64, want: u64| t.abs_diff(want) <= 150_000;
        assert!(
            near(bookmarks[0], 20_000_000) && near(bookmarks[1], 50_000_000),
            "{bookmarks:?}"
        );
        // The clip key also marks the recording ("Clip"), as before.
        assert_eq!(meta.markers.len(), 3);
        let records = markers_recorded(&recording.path);
        assert_eq!(
            records.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            meta.markers.iter().map(|m| m.time).collect::<Vec<_>>(),
            "every marker is also a record (a crash keeps it)"
        );

        let clips = library.list_dir(true);
        assert_eq!(clips.len(), 1);
        let clip = &clips[0];
        assert!(
            in_its_date_folder(&library, clip),
            "{}",
            clip.path.display()
        );
        let ReplayKind::Clip { start, end } = clip.meta.kind else {
            panic!("a clip");
        };
        assert_eq!(
            clip.meta.bookmark_count(),
            1,
            "only the 50 s bookmark is in 45-75 s"
        );
        let inside = clip.meta.markers[0].time;
        assert!((start..=end).contains(&inside));
        assert!(
            near(end - inside, 25_000_000),
            "25 s before the clip key: {}",
            end - inside
        );
    }

    /// Lobby hops: "Record every game" keeps a world shorter than 20 s in the file of the world
    /// after it, and still splits after a long one. (With the old rule, a file at least 5 s old
    /// split at any world change, so this script made three files.)
    #[test]
    fn a_lobby_hop_stays_in_the_file_of_the_game_that_follows() {
        let Some((mut client, dir)) = client() else {
            return;
        };
        client.mods.settings.set_enabled("replay", true);
        let join = S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 7,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 1,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        });
        let respawn = |dimension: i32| {
            S2CPlay::Respawn(s2c::Respawn {
                dimension,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })
        };
        let chat = |text: &str| {
            S2CPlay::Chat(s2c::Chat {
                component: format!(r#"{{"text":"{text}"}}"#),
                position: 0,
            })
        };
        // A main lobby for 8 s, a proxy switch (two Respawns) into a game for 32 s, a switch
        // into the next lobby.
        let script: Vec<(f64, S2CPlay)> = vec![
            (0.0, join),
            (1.0, chat("main lobby")),
            (8.0, respawn(-1)),
            (8.05, respawn(0)),
            (9.0, chat("the game")),
            (40.0, respawn(1)),
            (40.05, respawn(0)),
            (41.0, chat("the next lobby")),
        ];
        let mut recorder = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "hops".into(),
            started_at_unix_millis: 0,
        });
        for (seconds, packet) in &script {
            recorder.record_at(
                Direction::Clientbound,
                State::Play,
                (seconds * 1e6) as u64,
                &frame(packet),
            );
        }
        let capture = recorder.finish();
        let frames = open(&mut client, &capture, Path::new("hops.mccap"), 50_000);
        drive(&mut client, frames, Duration::from_millis(50), |_, _| {});
        finish(&mut client);
        let mut files = Library::new(dir.path()).list_dir(false);
        // The synthetic clock runs this 41 s script in about a millisecond of wall
        // time, so both files can carry the same wall-clock start (the test failed now and then
        // under parallel load, the second file sorting first). The first file is the 40 s one.
        files.sort_by_key(|e| (e.meta.started_unix_ms, std::cmp::Reverse(e.meta.duration)));
        let durations: Vec<u64> = files.iter().map(|e| e.meta.duration / 1_000_000).collect();
        assert_eq!(
            files.len(),
            2,
            "lobby + game, then the next lobby: {durations:?}"
        );
        let resets = &files[0].meta.world_resets;
        assert_eq!(
            resets.len(),
            3,
            "the hop is a world reset inside the file: {resets:?}"
        );
        let paths: Vec<PathBuf> = files.iter().map(|e| e.path.clone()).collect();
        let texts: Vec<String> = Reader::open(&paths[0])
            .unwrap()
            .records()
            .map(|r| r.unwrap())
            .filter(|r| r.stream == StreamId::Packet)
            .map(|r| String::from_utf8_lossy(&r.payload).into_owned())
            .filter(|t| t.contains("lobby") || t.contains("game"))
            .collect();
        assert_eq!(
            texts.len(),
            2,
            "main lobby and the game in the first file: {texts:?}"
        );
    }

    fn gpu(size: (u32, u32)) -> Option<mc_render::Gpu> {
        match mc_render::Gpu::headless(
            size.0,
            size.1,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("skip: no wgpu adapter available on this machine");
                None
            }
            Err(error) => panic!("headless GPU: {error}"),
        }
    }

    fn save(gpu: &mc_render::Gpu, size: (u32, u32), path: &Path) {
        let pixels = gpu.read_pixels().expect("readback");
        image::save_buffer(
            path,
            &pixels,
            size.0,
            size.1,
            image::ExtendedColorType::Rgba8,
        )
        .expect("write the PNG");
        eprintln!("wrote {}", path.display());
    }

    /// Review picture (opt-in, `MC_RA_REVIEW_DIR`; the replays from `MC_RA_REFS_DIR`, copies of
    /// the user's): the real client's Replays browser over the title screen, after its listing
    /// thread moved the flat files into their date folders.
    #[test]
    fn review_the_client_browser_after_organizing_the_users_replays() {
        let (Some(out), Some(refs)) = (
            std::env::var_os("MC_RA_REVIEW_DIR").map(PathBuf::from),
            std::env::var_os("MC_RA_REFS_DIR").map(PathBuf::from),
        ) else {
            eprintln!("skip: set MC_RA_REVIEW_DIR and MC_RA_REFS_DIR");
            return;
        };
        std::fs::create_dir_all(&out).expect("review dir");
        let size = (1920, 1080);
        let Some(mut gpu) = gpu(size) else { return };
        let Some((mut client, dir)) = client() else {
            return;
        };
        let library = Library::new(dir.path());
        std::fs::create_dir_all(library.replays_dir()).unwrap();
        let mut flat = Vec::new();
        for item in std::fs::read_dir(&refs).unwrap().flatten() {
            let to = library.replays_dir().join(item.file_name());
            std::fs::copy(item.path(), &to).unwrap();
            let file = std::fs::OpenOptions::new().write(true).open(&to).unwrap();
            file.set_modified(std::time::SystemTime::now() - Duration::from_secs(3600))
                .unwrap();
            flat.push(to);
        }
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, size.0, size.1);
        let mut camera = crate::app::driver::menu_camera();
        camera.set_viewport(size.0, size.1);
        client.frame(&gpu, &mut camera, 0.0);
        client.open_replay_browser();
        // The listing thread organizes first, then lists.
        for _ in 0..200 {
            client.frame(&gpu, &mut camera, 0.0);
            if flat.iter().all(|path| !path.exists()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        for _ in 0..40 {
            client.frame(&gpu, &mut camera, 0.0);
            std::thread::sleep(Duration::from_millis(25));
        }
        // Every flat copy moved into its date folder, and the browser lists them there.
        assert!(flat.iter().all(|path| !path.exists()), "nothing left flat");
        let listed = library.list_dir(false);
        assert_eq!(listed.len(), flat.len());
        assert!(listed
            .iter()
            .all(|entry| in_its_date_folder(&library, entry)));
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &camera, 0.0);
        save(
            &gpu,
            size,
            &out.join("client-browser-organized-1920x1080.png"),
        );
        for entry in &listed {
            eprintln!("{}", entry.path.strip_prefix(dir.path()).unwrap().display());
        }
    }

    /// Review picture (opt-in, `MC_RA_REVIEW_DIR`): the SkyWars scene in game, recording, a
    /// moment after the Bookmark key: "Bookmark 1" beside the recording dot.
    #[test]
    fn review_the_bookmark_confirmation_in_game() {
        let Some(out) = std::env::var_os("MC_RA_REVIEW_DIR").map(PathBuf::from) else {
            eprintln!("skip: set MC_RA_REVIEW_DIR");
            return;
        };
        std::fs::create_dir_all(&out).expect("review dir");
        let size = (1280, 720);
        let Some(mut gpu) = gpu(size) else { return };
        let Some((mut client, _dir)) = client() else {
            return;
        };
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../optimization/scratch/skywars-scene-2026-11-05/captures")
            .join("sw-mega-doubles-mid.mccap");
        let capture = Capture::read_file(&path).expect("the SkyWars bench capture");
        client.mods.settings.set_enabled("replay", true);
        client
            .mods
            .settings
            .set_key("replay", ids::KEY_BOOKMARK, 48);
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, size.0, size.1);
        let mut camera = crate::app::driver::menu_camera();
        camera.set_viewport(size.0, size.1);
        let _ = open(&mut client, &capture, &path, 50_000);
        // The main loop's order: the recorder's frame, the packets, a tick (50 ms of capture per
        // frame), and every few frames a rendered frame (terrain uploads).
        let start = Instant::now();
        client.rec.sync(&client.mods.settings);
        for frame in 0..400u32 {
            client.rec_frame(start + Duration::from_millis(50) * frame);
            client.poll_in_place();
            client.tick(&mc_physics::state::MoveInput::default());
            if frame % 8 == 0 {
                client.frame(&gpu, &mut camera, 0.0);
            }
        }
        if !matches!(client.ui, super::super::super::Ui::InGame) {
            client.show_screen(None);
        }
        client.rec.bookmark_key(Instant::now());
        for _ in 0..3 {
            client.frame(&gpu, &mut camera, 0.0);
        }
        assert!(client.rec.recording(), "recording the scene");
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &camera, 0.0);
        save(&gpu, size, &out.join("in-game-bookmark-1280x720.png"));
        client.session = None;
        client.rec_shutdown();
    }
}
