//! A replay frame reached through a **checkpoint** is
//! byte-identical to the same frame reached by applying the replay from its start.
//!
//! The real client on a headless GPU opens a real `.aeroreplay` (set `MC_RC_REPLAY`; the test
//! skips without it, without the vanilla jar or without a GPU adapter) and:
//!
//! 1. **Reference pass.** Moves forward from the start through every target time in order
//!    (`replay_render_at`, the blocking seek the exporter uses: each step fast-applies the records
//!    after the previous one, exactly what playing does, and takes checkpoints on the way). At
//!    each target it lets the terrain settle, reads the frame back, and writes down a digest of
//!    the replayed state (tick, player, every entity's position and animation, particles and their
//!    random stream, chat, HUD, environment).
//! 2. **Checkpoint pass.** Jumps to the targets again in a scrambled order, backwards each time,
//!    so every one is reached by restoring a checkpoint and fast-applying from it. The frame and
//!    the digest must be identical.
//! 3. **Controls.** A frame half a second later must differ from the target's (the comparison is
//!    sensitive), and with restores deliberately broken (the client-side fields left as they
//!    were, as a forgotten field list would) the digest must differ.
//!
//! ```text
//! MC_RC_REPLAY=<file.aeroreplay> MC_JARS_DIR=<jars> cargo test -p mc-client \
//!     --test rc_seek_determinism -- --ignored --nocapture --test-threads=1
//! ```
//! Optional: `MC_RC_TARGETS=95.5,212.25,...` (seconds) picks the targets, `MC_RC_PNG_DIR=<dir>`
//! writes every compared frame as a PNG.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mc_client::app::config::LoginMode;
use mc_client::{Client, ClientConfig};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList};

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;
/// Default targets (seconds): spread over the first ten minutes, off any checkpoint grid.
const DEFAULT_TARGETS: [f64; 5] = [47.31, 133.77, 251.9, 388.05, 597.6];

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    png_dir: Option<PathBuf>,
    _dir: tempfile::TempDir,
}

impl Rig {
    fn new(replay: &Path) -> Option<Rig> {
        let dir = tempfile::tempdir().ok()?;
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("SeekBot".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar (set MC_JARS_DIR)");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let gpu = match Gpu::headless(
            WIDTH,
            HEIGHT,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("SKIP: no headless GPU ({error})");
                return None;
            }
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, WIDTH, HEIGHT);
        client.replay_open(replay).expect("the replay opens");
        let mut camera = Camera::default();
        camera.set_viewport(WIDTH, HEIGHT);
        let png_dir = std::env::var_os("MC_RC_PNG_DIR").map(PathBuf::from);
        if let Some(dir) = &png_dir {
            std::fs::create_dir_all(dir).expect("png dir");
        }
        Some(Rig {
            client,
            gpu,
            phases,
            camera,
            png_dir,
            _dir: dir,
        })
    }

    /// One frame of the paused replay, read back.
    fn render(&mut self) -> Vec<u8> {
        self.client.frame(&self.gpu, &mut self.camera, 0.0);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 0.0);
        let pixels = frame.read_pixels(&self.gpu).expect("read back");
        frame.present(&self.gpu.queue);
        pixels
    }

    /// Seek (blocking), let every chunk mesh and upload, and read the frame back. With
    /// `as_a_jump`, the state vanilla keeps from *drawn* frames (the boss bar's countdown and the
    /// wither sky tint) is cleared first, as every jump (a rebuild or a restore) leaves it: the
    /// reference pass moves forward with frames drawn at each target, which a jump never has.
    fn frame_at(&mut self, seconds: f64, as_a_jump: bool) -> Vec<u8> {
        self.client.replay_render_at((seconds * 1e6) as u64);
        if as_a_jump {
            self.client.replay_debug_forget_drawn_frames();
        }
        let mut settled = 0;
        for _ in 0..2000 {
            self.render();
            if self.client.replay_terrain_settled() {
                settled += 1;
                if settled >= 3 {
                    return self.render();
                }
            }
        }
        panic!("the terrain never settled at {seconds} s");
    }

    fn save(&self, name: &str, pixels: &[u8]) {
        let Some(dir) = &self.png_dir else {
            return;
        };
        let path = dir.join(format!("{name}.png"));
        image::save_buffer(&path, pixels, WIDTH, HEIGHT, image::ColorType::Rgba8)
            .expect("write png");
    }

    /// The replayed state that frames are drawn from, as text. Entity arena handles and the
    /// scoreboard and tab-list revision numbers (process-wide counters) are left out: they are
    /// identities, not state.
    fn digest(&mut self) -> String {
        let mut out = String::new();
        let client = &mut self.client;
        let _ = writeln!(out, "tick {}", client.tick_count());
        let _ = writeln!(out, "time {}", client.replay_time());
        if let Some(session) = client.session() {
            if let Some(predictor) = session.predictor() {
                let _ = writeln!(out, "player {:?}", predictor.state);
            }
            let state = session.state();
            let _ = writeln!(out, "local {:?}", state.local);
            let _ = writeln!(out, "env {:?}", state.env);
            let _ = writeln!(out, "entities {}", state.entities.len());
            for (id, tracked) in state.entities.iter() {
                let pos = tracked.position(&state.world);
                let _ = writeln!(
                    out,
                    "entity {id} {:?} {:?} {:?}",
                    pos,
                    tracked.server_position(),
                    tracked.render_params(0.5)
                );
            }
        } else {
            out.push_str("no session\n");
        }
        let _ = writeln!(out, "particles {}", client.particles.len());
        let rng = client.particles.rng_mut().clone().next_i64();
        let _ = writeln!(out, "particle rng {rng}");
        for particle in client.particles.particles() {
            let _ = writeln!(out, "particle {particle:?}");
        }
        let _ = writeln!(out, "chat {}", client.chat.lines().len());
        for line in client.chat.lines() {
            let _ = writeln!(out, "chat {line:?}");
        }
        let _ = writeln!(out, "hud {:?}", client.hud);
        let _ = writeln!(out, "player hud {:?}", client.player_hud);
        // The first-person phase's tick state lives in the renderer, outside the
        // session (a restore once left the held item out of the hand).
        out.push_str(&hand_digest(client));
        // The world border's two stamps stay in. They were wall-clock readings (a
        // rebuild stamped a later time than the pass that took a checkpoint), so this used to
        // strip them with `without_wall_clock_stamps`; a replay now runs the border on the
        // recording's clock, and they must match like everything else.
        without_hash_indexes(&out)
    }
}

/// Seek to `t` (blocking), draw one frame (the per-frame visibility gates are part of
/// the compared state) and return [`hand_digest`].
fn hand_at(rig: &mut Rig, t: f64) -> String {
    rig.client.replay_render_at((t * 1e6) as u64);
    rig.render();
    hand_digest(&rig.client)
}

/// The first-person hand's tick state (equip progress, the latched item, swing, use) and
/// whether a model is latched, as one line.
fn hand_digest(client: &Client) -> String {
    match client.replay_debug_first_person() {
        Some(held) => format!(
            "hand {:?} model {}\n",
            held.render_state(),
            held.has_item_model()
        ),
        None => "hand none\n".to_owned(),
    }
}

/// Drop every `index: {...}` map from a Debug dump. An NBT compound keeps a name -> position
/// `HashMap` beside its ordered entries; each decoded item stack gets a fresh `RandomState`, so
/// that map prints in a different order every time the same packet is decoded. It is derived
/// from the entries, which stay in the dump.
fn without_hash_indexes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("index: {") {
        out.push_str(&rest[..at]);
        let mut depth = 0usize;
        let mut end = rest.len();
        for (offset, ch) in rest[at..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = at + offset + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str("index: _");
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// Replace the number after each `name: ` with `_`. The world border keeps
/// `System.currentTimeMillis()` stamps of when its packet was applied (vanilla `WorldBorder`),
/// which were wall-clock readings in replays too until replays moved onto the recording's
/// clock; the digest no longer strips them (kept for its unit test and for other wall-clock
/// fields a future digest might meet).
fn without_wall_clock_stamps(text: &str, names: &[&str]) -> String {
    let mut out = text.to_owned();
    for name in names {
        let key = format!("{name}: ");
        let mut result = String::with_capacity(out.len());
        let mut rest = out.as_str();
        while let Some(at) = rest.find(&key) {
            let value_start = at + key.len();
            result.push_str(&rest[..value_start]);
            let value_len = rest[value_start..]
                .find(|c: char| !(c.is_ascii_digit() || c == '-'))
                .unwrap_or(rest.len() - value_start);
            result.push('_');
            rest = &rest[value_start + value_len..];
        }
        result.push_str(rest);
        out = result;
    }
    out
}

fn first_difference(a: &str, b: &str) -> String {
    for (index, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {index}:\n  reference: {x}\n  restored:  {y}");
        }
    }
    format!(
        "length: {} vs {} lines",
        a.lines().count(),
        b.lines().count()
    )
}

fn differing_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// The path a viewer takes most: watch (frames drawn between the records, as in the window),
/// then jump back. A checkpoint taken during that playback must restore to the same frame a
/// rebuild from the start of the file reaches.
#[test]
#[ignore = "needs MC_RC_REPLAY, the vanilla jar and a GPU; run with --ignored"]
fn a_checkpoint_taken_while_watching_restores_exactly() {
    use mc_render::ShellApp as _;
    let Some(replay) = std::env::var_os("MC_RC_REPLAY").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_RC_REPLAY to an .aeroreplay");
        return;
    };
    let Some(mut rig) = Rig::new(&replay) else {
        return;
    };
    let (start, end) = rig.client.replay_view_range();
    // Watch from 100 s to 240 s: with a checkpoint every interval (70 s on the hour-long
    // replay, 10 s on short ones), at least one is taken while frames are being drawn.
    let watch_from = 100.0f64.max(start as f64 / 1e6);
    let watch_to = (watch_from + 140.0).min(end as f64 / 1e6 - 2.0);
    assert!(watch_to > watch_from + 20.0, "the replay is long enough");
    rig.frame_at(watch_from, false);
    let taken_before = rig
        .client
        .replay_checkpoint_stats()
        .map_or(0, |stats| stats.taken);
    rig.client.replay_set_speed(8.0);
    rig.client.replay_play();
    let mut frames = 0u32;
    while rig.client.replay_time() < (watch_to * 1e6) as u64 && rig.client.replay_is_playing() {
        rig.client.before_ticks(&mut rig.camera);
        rig.client.frame(&rig.gpu, &mut rig.camera, 0.0);
        let frame = rig.gpu.acquire().expect("acquire");
        rig.phases.render(&rig.gpu, frame.view(), &rig.camera, 0.0);
        frame.present(&rig.gpu.queue);
        frames += 1;
    }
    rig.client.replay_pause();
    let stats = rig.client.replay_checkpoint_stats().expect("open");
    eprintln!("watched {watch_from} s -> {watch_to} s in {frames} drawn frames: {stats:?}");
    assert!(
        stats.taken > taken_before,
        "no checkpoint was taken while watching: {stats:?}"
    );
    // Just after the middle of the watched stretch, so the restore uses one taken while
    // watching.
    let target = watch_from + (watch_to - watch_from) * 0.75 + 0.37;
    let restored_before = stats.restored;
    let restored = rig.frame_at(target, false);
    let restored_digest = rig.digest();
    let stats = rig.client.replay_checkpoint_stats().expect("open");
    assert!(
        stats.restored > restored_before,
        "the jump back restored: {stats:?}"
    );
    rig.save("watch_restored", &restored);
    // The reference: a rebuild from the start of the file.
    rig.frame_at(target + 1.0, false);
    rig.client.replay_drop_checkpoints();
    let rebuilt = rig.frame_at(target, false);
    let rebuilt_digest = rig.digest();
    rig.save("watch_rebuilt", &rebuilt);
    let bytes = differing_bytes(&rebuilt, &restored);
    eprintln!(
        "watched-then-restored vs rebuilt at {target:.2} s: {bytes} bytes differ, state {}",
        if rebuilt_digest == restored_digest {
            "identical"
        } else {
            "DIFFERS"
        }
    );
    rig.client.replay_close();
    assert_eq!(bytes, 0, "the frames differ");
    assert!(
        rebuilt_digest == restored_digest,
        "the state differs: {}",
        first_difference(&rebuilt_digest, &restored_digest)
    );
}

#[test]
fn hash_index_maps_are_dropped_from_digests_and_nothing_else() {
    let text = "a { x: 1, index: {\"lvl\": 0, \"id\": {1}} } b index: {} c";
    assert_eq!(
        without_hash_indexes(text),
        "a { x: 1, index: _ } b index: _ c"
    );
    assert_eq!(without_hash_indexes("no maps"), "no maps");
    assert_eq!(
        without_wall_clock_stamps(
            "B { start_time_millis: 1794641609760, end_time_millis: -5, speed_millis: 7 }",
            &["start_time_millis", "end_time_millis"]
        ),
        "B { start_time_millis: _, end_time_millis: _, speed_millis: 7 }"
    );
}

#[test]
#[ignore = "needs MC_RC_REPLAY, the vanilla jar and a GPU; run with --ignored"]
fn a_frame_reached_through_a_checkpoint_is_byte_identical() {
    let Some(replay) = std::env::var_os("MC_RC_REPLAY").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_RC_REPLAY to an .aeroreplay");
        return;
    };
    let targets: Vec<f64> = std::env::var("MC_RC_TARGETS").map_or_else(
        |_| DEFAULT_TARGETS.to_vec(),
        |list| {
            list.split(',')
                .filter_map(|value| value.trim().parse().ok())
                .collect()
        },
    );
    let Some(mut rig) = Rig::new(&replay) else {
        return;
    };
    let (start, end) = rig.client.replay_view_range();
    let targets: Vec<f64> = targets
        .into_iter()
        .filter(|t| (*t * 1e6) as u64 > start && ((*t + 1.0) * 1e6) as u64 <= end)
        .collect();
    assert!(targets.len() >= 2, "at least two targets inside the replay");

    // 1. The reference: forward from the start through every target, in order.
    let mut reference = Vec::new();
    for (index, &t) in targets.iter().enumerate() {
        let frame = rig.frame_at(t, true);
        let digest = rig.digest();
        rig.save(&format!("ref_{index}_{t}"), &frame);
        eprintln!(
            "reference {t:>8.2} s: {} entities, {} particles",
            rig.client.session().map_or(0, |s| s.state().entities.len()),
            rig.client.particles.len()
        );
        reference.push((t, frame, digest));
    }
    let stats = rig
        .client
        .replay_checkpoint_stats()
        .expect("a replay is open");
    eprintln!("after the reference pass: {stats:?}");
    assert!(stats.off.is_none(), "checkpoints switched off: {stats:?}");
    assert!(stats.taken >= 2, "checkpoints were taken: {stats:?}");

    // Control: half a second later the picture moves.
    let (t0, frame0, _) = &reference[reference.len() - 1];
    let later = rig.frame_at(t0 + 0.5, true);
    let moved = differing_bytes(frame0, &later);
    eprintln!("control: {moved} bytes differ half a second later");
    assert!(
        moved > 0,
        "the scene did not move: the comparison would be blind"
    );

    // 2. Every target again, each reached backwards through a checkpoint.
    let mut order: Vec<usize> = (0..reference.len()).rev().collect();
    order.rotate_left(1);
    let mut failures = Vec::new();
    for &index in &order {
        let (t, frame, digest) = &reference[index];
        let restored_before = rig
            .client
            .replay_checkpoint_stats()
            .map_or(0, |s| s.restored);
        // No help from the test here: the restore itself must leave what a jump leaves.
        let again = rig.frame_at(*t, false);
        let again_digest = rig.digest();
        let stats = rig.client.replay_checkpoint_stats().expect("open");
        assert!(
            stats.restored > restored_before,
            "the seek to {t} s did not restore a checkpoint: {stats:?}"
        );
        rig.save(&format!("chk_{index}_{t}"), &again);
        let bytes = differing_bytes(frame, &again);
        let same_state = *digest == again_digest;
        eprintln!(
            "target {t:>8.2} s via checkpoint: {bytes} bytes differ, state {}, restore {:.1} ms",
            if same_state { "identical" } else { "DIFFERS" },
            stats.restore_ms_last
        );
        if bytes != 0 || !same_state {
            failures.push(format!(
                "{t} s: {bytes} bytes differ; state: {}",
                if same_state {
                    "identical".to_owned()
                } else {
                    first_difference(digest, &again_digest)
                }
            ));
        }
    }

    // 3. The old path against the new one with nothing normalised: a rebuild from the start of
    //    the file (checkpoints dropped first) and a restore must reach the same frame.
    for &index in &[reference.len() / 2, reference.len() - 1] {
        let t = reference[index].0;
        // Past the target, then back with no checkpoint left: a rebuild from the start.
        rig.frame_at(t + 1.0, false);
        rig.client.replay_drop_checkpoints();
        let rebuilds_before = rig.client.replay_stats().map_or(0, |s| s.rebuilds);
        let rebuilt = rig.frame_at(t, false);
        let rebuilt_digest = rig.digest();
        let rebuilds = rig.client.replay_stats().map_or(0, |s| s.rebuilds);
        assert!(
            rebuilds > rebuilds_before,
            "the seek back rebuilt the world"
        );
        // Away and back, so the second visit restores one of the checkpoints the rebuild took.
        rig.frame_at(t + 30.0, false);
        let before = rig
            .client
            .replay_checkpoint_stats()
            .map_or(0, |s| s.restored);
        let restored = rig.frame_at(t, false);
        let restored_digest = rig.digest();
        let stats = rig.client.replay_checkpoint_stats().expect("open");
        assert!(
            stats.restored > before,
            "the second visit restored: {stats:?}"
        );
        let bytes = differing_bytes(&rebuilt, &restored);
        let same_state = rebuilt_digest == restored_digest;
        eprintln!(
            "rebuild vs restore at {t:>8.2} s: {bytes} bytes differ, state {} (rebuilds so far {rebuilds})",
            if same_state { "identical" } else { "DIFFERS" }
        );
        if bytes != 0 || !same_state {
            failures.push(format!(
                "rebuild vs restore at {t} s: {bytes} bytes differ; state: {}",
                if same_state {
                    "identical".to_owned()
                } else {
                    first_difference(&rebuilt_digest, &restored_digest)
                }
            ));
        }
    }

    // 4. Negative control: restores that forget the client-side fields must be caught.
    rig.client.replay_debug_break_checkpoints(true);
    let (t, _, digest) = &reference[0];
    let far = reference[reference.len() - 1].0;
    rig.frame_at(far, true);
    rig.frame_at(*t, false);
    let broken = rig.digest();
    rig.client.replay_debug_break_checkpoints(false);
    eprintln!(
        "negative control (broken restore) at {t} s: state {}",
        if broken == *digest {
            "identical (BLIND)"
        } else {
            "differs (caught)"
        }
    );
    assert_ne!(
        broken, *digest,
        "a restore that kept the live client-side fields went unnoticed: the digest is blind"
    );

    let stats = rig.client.replay_checkpoint_stats().expect("open");
    eprintln!("final: {stats:?}");
    rig.client.replay_close();
    assert!(
        failures.is_empty(),
        "not identical:\n{}",
        failures.join("\n")
    );
}

/// A jump back through a checkpoint keeps the held item in the
/// hand. The first-person phase (the model latched at the last equip swap, the equip and swing
/// state) lives in the renderer, outside the session; restores used to leave it as
/// `unload_world` does, so the item vanished from the hand until the player switched items
/// (the harness: 4101 pixels at its tick 99). Many targets, state only (no terrain
/// settling), so it stays quick; the frames are covered by the tests above, whose digest now
/// includes the hand.
#[test]
#[ignore = "needs MC_RC_REPLAY, the vanilla jar and a GPU; run with --ignored"]
fn a_jump_back_keeps_the_held_item_in_the_hand() {
    let Some(replay) = std::env::var_os("MC_RC_REPLAY").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_RC_REPLAY to an .aeroreplay");
        return;
    };
    let Some(mut rig) = Rig::new(&replay) else {
        return;
    };
    let (start, end) = rig.client.replay_view_range();
    let span = end.saturating_sub(start) as f64 / 1e6;
    assert!(span > 60.0, "a replay of at least a minute");
    // 24 targets spread over the replay, off any checkpoint grid (3.3 s past each step).
    let targets: Vec<f64> = (1..=24)
        .map(|k| start as f64 / 1e6 + span * f64::from(k) / 25.0 + 3.3)
        .filter(|t| ((t + 1.0) * 1e6) as u64 <= end)
        .collect();
    let reference: Vec<(f64, String)> =
        targets.iter().map(|&t| (t, hand_at(&mut rig, t))).collect();
    let rendering = reference
        .iter()
        .filter(|(_, hand)| hand.contains("item_to_render: Some"))
        .count();
    let holding = reference
        .iter()
        .filter(|(_, hand)| hand.contains("item_to_render: Some") && hand.ends_with("model true\n"))
        .count();
    eprintln!(
        "{holding} of {} reference targets hold a latched item ({rendering} show an item)",
        reference.len()
    );
    // A replay whose player has an empty slot selected at every target (a lobby, say) has no
    // item that a jump could lose: the held-item check is blind there, which is the replay's
    // doing, not the seek's, so say so and still compare the equip and swing state after every
    // jump. Items shown without a latched model at every target still fail: that is the loss
    // this test was written for.
    if rendering == 0 {
        eprintln!(
            "NOTE: the hand is empty at every target of this replay, so the held-item check is \
             blind here (the equip and swing state are still compared); run it on a replay \
             where the player holds items to see it bite"
        );
    } else {
        assert!(
            holding > 0,
            "no target holds an item: the test would be blind; the first targets' hands:\n{}",
            reference
                .iter()
                .take(3)
                .map(|(t, hand)| format!("  {t:.2} s: {hand}"))
                .collect::<String>()
        );
    }

    // Backwards, so every target restores a checkpoint.
    let mut failures = Vec::new();
    for (t, hand) in reference.iter().rev() {
        let before = rig
            .client
            .replay_checkpoint_stats()
            .map_or(0, |s| s.restored);
        let again = hand_at(&mut rig, *t);
        let restored = rig
            .client
            .replay_checkpoint_stats()
            .map_or(0, |s| s.restored)
            > before;
        if *hand != again {
            failures.push(format!(
                "{t:.2} s (restored {restored}):\n  ref {hand}  got {again}"
            ));
        }
    }
    let stats = rig.client.replay_checkpoint_stats().expect("open");
    eprintln!(
        "hand after jumps: {} of {} differ; {stats:?}",
        failures.len(),
        reference.len()
    );

    // Negative control: a restore that leaves the render phases as `unload_world` does (what
    // restores did before this fix; everything else restored as usual) must lose the hand.
    rig.client.replay_debug_skip_render_restore(true);
    let mut caught = 0;
    for (t, hand) in reference.iter().rev() {
        if hand_at(&mut rig, *t) != *hand {
            caught += 1;
        }
    }
    rig.client.replay_debug_skip_render_restore(false);
    eprintln!(
        "negative control: {caught} of {} targets differ",
        reference.len()
    );
    rig.client.replay_close();
    // The control only bites where the held item stays the same between a checkpoint and its
    // target: the hour-long replay (a checkpoint every 70 s) catches 8 of 24; a short PvP replay
    // (every 10 s, items switched all the time, each switch re-latching the hand) may catch
    // none. That makes the control blind there, not the fix wrong: say so, do not fail.
    if caught == 0 {
        eprintln!(
            "NOTE: the negative control found nothing on this replay (the hand re-latched \
             between every checkpoint and its target); run it on a long replay to see it bite"
        );
    }
    assert!(
        failures.is_empty(),
        "the hand differs after a jump:\n{}",
        failures.join("\n")
    );
}
