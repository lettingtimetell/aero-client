//! Replay QA: replays that move, vanish or are cut short, and
//! seeks over real recordings.
//!
//! The real client (`Client::new` on the real assets), headless: no window, no GPU (the export's
//! first seek runs the same way, with no frames drawn). Needs the vanilla jar (`MC_JARS_DIR`);
//! every test skips without it. The tests on real recordings also need `MC_RQ_REPLAYS`, a folder
//! of `.aeroreplay` files (copies: each file is copied again into a temporary folder first).
//!
//! ```text
//! MC_JARS_DIR=<jars> MC_RQ_REPLAYS=<folder> cargo test -p mc-client --test nw_rq_replay_qa \
//!     -- --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_client::app::config::LoginMode;
use mc_client::{Client, ClientConfig};
use mc_replay::{ReplayMeta, StreamId, Writer, WriterOptions};

/// A headless client whose game folder is `dir`, or `None` (skip) without the vanilla jar.
fn client(dir: &Path) -> Option<Client> {
    let config = ClientConfig {
        game_dir: dir.to_path_buf(),
        login: LoginMode::Offline("QaBot".to_owned()),
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
        eprintln!("SKIP: no vanilla jar (set MC_JARS_DIR)");
        return None;
    }
    Some(Client::new(config, &mut |_, _| {}).expect("the real assets must load"))
}

/// A small, finished replay file (30 s, a few packet records), as the tests write them.
fn write_replay(path: &Path) {
    std::fs::create_dir_all(path.parent().expect("a folder")).expect("the folder");
    let mut meta = ReplayMeta {
        title: "QA".to_owned(),
        server: "example.invalid".to_owned(),
        started_unix_ms: 1_794_537_660_000,
        ..ReplayMeta::default()
    };
    let mut writer = Writer::create(path, &meta, WriterOptions::default()).expect("the writer");
    for i in 0..10u64 {
        writer.push(i * 100_000, StreamId::Packet, &[0x02, 0, 0]);
    }
    meta.duration = 30_000_000;
    writer
        .finish(meta)
        .wait(Duration::from_secs(20))
        .expect("the file is written");
}

/// Every `.aeroreplay` in `MC_RQ_REPLAYS`, or none.
fn real_replays() -> Vec<PathBuf> {
    let Some(dir) = std::env::var_os("MC_RQ_REPLAYS").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_RQ_REPLAYS to a folder of .aeroreplay copies");
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("MC_RQ_REPLAYS is a folder")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext == mc_replay::EXTENSION)
        })
        .collect();
    files.sort();
    files
}

fn name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// RD's harness "hung twice" when the library moved its replay into a date folder. A
/// replay path that no longer leads to a replay (moved, renamed, deleted, or the empty
/// placeholder the library's mover creates for a moment) must fail to open at once with a
/// message that names no folder, and a replay that vanishes while it is open must keep playing
/// (its records are in memory) and close cleanly.
#[test]
fn a_replay_that_moved_or_vanished_fails_cleanly_and_an_open_one_plays_on() {
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(mut client) = client(dir.path()) else {
        return;
    };
    let flat = dir.path().join("replays").join("a.aeroreplay");
    write_replay(&flat);

    // Moved into its date folder, as the library does on the first start.
    let moved = dir
        .path()
        .join("replays")
        .join("2026")
        .join("11 November")
        .join("13")
        .join("a.aeroreplay");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&flat, &moved).unwrap();
    let started = Instant::now();
    let error = client.replay_open(&flat).expect_err("the old path is gone");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "opening a missing file stalls"
    );
    assert!(!client.replay_is_active());
    let folder = dir.path().to_string_lossy().into_owned();
    assert!(
        !error.contains(&folder),
        "the message names the folder (the account name is in it): {error}"
    );
    eprintln!("moved: {error}");

    // The library's mover keeps an empty placeholder at the new name for a moment.
    let placeholder = dir.path().join("replays").join("b.aeroreplay");
    std::fs::write(&placeholder, b"").unwrap();
    let error = client
        .replay_open(&placeholder)
        .expect_err("an empty file is no replay");
    assert!(!client.replay_is_active());
    eprintln!("placeholder: {error}");

    // Open at its new place, then the file is deleted (another client's storage limit, the
    // player in Explorer) and renamed: playback goes on from memory, seeks both ways work.
    client.replay_open(&moved).expect("the moved replay opens");
    let (start, end) = client.replay_view_range();
    client.replay_render_at(start);
    let renamed = moved.with_file_name("renamed.aeroreplay");
    std::fs::rename(&moved, &renamed).unwrap();
    client.replay_render_at(end);
    std::fs::remove_file(&renamed).unwrap();
    client.replay_render_at(start + (end - start) / 3);
    client.replay_render_at(end);
    assert!(client.replay_is_active());
    assert_eq!(client.replay_time(), end);
    client.replay_close();
    assert!(!client.replay_is_active());
    client
        .replay_open(&renamed)
        .expect_err("a deleted replay does not open again");
}

/// Crash recovery: a recording cut anywhere (the game or the PC crashed mid-write) opens and
/// plays what it holds, or is refused with a message; it never panics or hangs. Cuts through
/// the header, the first block, the middle, the last block and the footer of each real replay.
#[test]
fn cut_copies_of_real_replays_open_and_play_what_they_hold() {
    let replays = real_replays();
    if replays.is_empty() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(mut client) = client(dir.path()) else {
        return;
    };
    let cut_dir = dir.path().join("cut");
    std::fs::create_dir_all(&cut_dir).unwrap();
    for source in &replays {
        let bytes = std::fs::read(source).expect("read the copy");
        let full = mc_replay::read_info(source).expect("the original reads");
        let full_end = full.meta.window().1;
        let len = bytes.len();
        let cuts = [
            20,
            len / 2000,
            len / 100,
            len / 10,
            len / 3,
            len / 2,
            len * 77 / 100,
            len * 999 / 1000,
            len - 41,
            len - 3,
        ];
        for cut in cuts {
            let path = cut_dir.join(format!("cut-{cut}.aeroreplay"));
            std::fs::write(&path, &bytes[..cut]).unwrap();
            let started = Instant::now();
            match client.replay_open(&path) {
                Ok(()) => {
                    let (start, end) = client.replay_view_range();
                    client.replay_render_at(end);
                    client.replay_render_at(start + (end - start) / 2);
                    client.replay_render_at(start);
                    let stats = client.replay_stats().expect("open");
                    eprintln!(
                        "{} cut at {cut}/{len}: opened, view {:.1}..{:.1} s of {:.1} s, {} ticks, \
                         {} packet errors, {:.1} s",
                        name(source),
                        start as f64 / 1e6,
                        end as f64 / 1e6,
                        full_end as f64 / 1e6,
                        stats.ticks,
                        stats.packet_errors,
                        started.elapsed().as_secs_f64()
                    );
                    assert!(end <= full_end, "a cut copy is never longer than the file");
                    if cut >= len * 999 / 1000 {
                        assert!(
                            end >= full_end * 9 / 10,
                            "a copy cut in its last block or footer keeps nearly everything"
                        );
                    }
                    client.replay_close();
                }
                Err(error) => {
                    eprintln!("{} cut at {cut}/{len}: refused: {error}", name(source));
                    assert!(
                        cut < len / 10,
                        "a cut after a tenth of the file still opens"
                    );
                }
            }
            assert!(!client.replay_is_active());
            std::fs::remove_file(&path).ok();
        }
    }
}

/// Every real replay: open, then seek through it forward, backward, across its world resets
/// (server switches, dimension changes, respawns) and to both ends, with checkpoints and
/// without; no panic, no packet that fails to decode, the time lands where asked.
#[test]
fn every_real_replay_seeks_everywhere_without_errors() {
    let replays = real_replays();
    if replays.is_empty() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(mut client) = client(dir.path()) else {
        return;
    };
    for source in &replays {
        let copy = dir.path().join(name(source));
        std::fs::copy(source, &copy).expect("copy");
        let meta = mc_replay::read_info(&copy).expect("reads").meta;
        client.replay_open(&copy).expect("opens");
        let (start, end) = client.replay_view_range();
        let span = end - start;
        let mut targets: Vec<u64> = vec![
            end,
            start,
            start + span / 2,
            start + span / 7,
            end.saturating_sub(1_000_000).max(start),
            start + span * 3 / 4,
        ];
        // Just before and after every world reset.
        for &reset in &meta.world_resets {
            for at in [
                reset.saturating_sub(250_000),
                reset + 250_000,
                reset + 2_000_000,
            ] {
                targets.push(at.clamp(start, end));
            }
        }
        let started = Instant::now();
        for (index, &target) in targets.iter().enumerate() {
            if index == targets.len() / 2 {
                // The second half without checkpoints: the rebuild path.
                client.replay_drop_checkpoints();
            }
            client.replay_render_at(target);
            assert_eq!(
                client.replay_time(),
                target,
                "{}: the seek landed",
                name(source)
            );
        }
        let stats = client.replay_stats().expect("open");
        eprintln!(
            "{}: {} seeks in {:.1} s, {} rebuilds, {} ticks, {} packets, {} packet errors, \
             {} of {} player ticks re-simulated differently",
            name(source),
            targets.len(),
            started.elapsed().as_secs_f64(),
            stats.rebuilds,
            stats.ticks,
            stats.packets,
            stats.packet_errors,
            stats.player_mismatches,
            stats.player_ticks
        );
        assert_eq!(
            stats.packet_errors,
            0,
            "{}: packets failed to decode",
            name(source)
        );
        client.replay_close();
    }
}
