//! The library's date folders, moving old flat files into them
//! without ever losing or overwriting one, names and titles in local time, the storage cap with
//! folders, and bookmarks (markers) surviving a crash.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use chrono::{FixedOffset, TimeZone as _};
use mc_replay::{
    date_folder, date_folder_in, localize_title_in, read_info, restamp_in, Library, Marker,
    ReplayKind, ReplayMeta, StreamId, Writer, WriterOptions, BOOKMARK_LABEL,
};

/// 2026-11-14 02:05:23.186 UTC = 2026-11-13 21:05:23 EST: the start of a recording made on a
/// computer in UTC, `2026-11-14 02.05.23 sky.example.org`.
const SAMPLE_START_MS: u64 = 1_794_621_923_186;

fn est() -> FixedOffset {
    FixedOffset::west_opt(5 * 3600).unwrap()
}

fn utc() -> FixedOffset {
    FixedOffset::east_opt(0).unwrap()
}

fn small_blocks() -> WriterOptions {
    WriterOptions {
        level: 3,
        block_bytes: 4096,
        block_interval: Duration::from_secs(3600),
        sync_interval: None,
    }
}

fn meta(title: &str, started_unix_ms: u64) -> ReplayMeta {
    ReplayMeta {
        title: title.to_owned(),
        server: "mc.example.net".to_owned(),
        started_unix_ms,
        ..ReplayMeta::default()
    }
}

/// A small finished replay at `path` (its parent made as needed), backdated past the library's
/// two-minute "still being recorded" grace.
fn write_replay(path: &Path, meta: &ReplayMeta, records: usize) {
    let mut writer = Writer::create(path, meta, small_blocks()).expect("create");
    for i in 0..records {
        let payload = vec![(i % 251) as u8; 40 + i % 300];
        writer.push(i as u64 * 50_000, StreamId::Packet, &payload);
    }
    let mut last = meta.clone();
    last.duration = records as u64 * 50_000;
    writer
        .finish(last)
        .wait(Duration::from_secs(30))
        .expect("finish");
    backdate(path);
}

fn backdate(path: &Path) {
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_modified(SystemTime::now() - Duration::from_secs(600))
        .unwrap();
}

fn names(paths: &[PathBuf], root: &Path) -> Vec<String> {
    let mut out: Vec<String> = paths
        .iter()
        .map(|p| {
            p.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    out.sort();
    out
}

#[test]
fn date_folders_are_year_month_day_in_the_computers_zone() {
    let root = Path::new("replays");
    // A recording made in UTC: Nov 14 there, still Nov 13 in a UTC-5 zone.
    assert_eq!(
        date_folder_in(root, SAMPLE_START_MS, &est()),
        root.join("2026").join("11 November").join("13")
    );
    assert_eq!(
        date_folder_in(root, SAMPLE_START_MS, &utc()),
        root.join("2026").join("11 November").join("14")
    );
    // Single digits are padded so every sort order is right.
    let march_5 = 1_772_700_000_000; // 2026-03-05 08:40 UTC
    assert_eq!(
        date_folder_in(root, march_5, &utc()),
        root.join("2026").join("03 March").join("05")
    );
    // The library's own path for a new file is in this computer's date folder.
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let path = library.unique_path_at(
        false,
        "2026-11-13 21.05.23 sky.example.org",
        SAMPLE_START_MS,
    );
    assert_eq!(
        path,
        date_folder(library.replays_dir(), SAMPLE_START_MS)
            .join("2026-11-13 21.05.23 sky.example.org.aeroreplay")
    );
    let clip = library.unique_path_at(true, "clip", SAMPLE_START_MS);
    assert!(clip.starts_with(library.clips_dir()));
    assert_eq!(
        clip.parent().unwrap(),
        date_folder(library.clips_dir(), SAMPLE_START_MS)
    );
}

#[test]
fn listing_finds_flat_files_date_folders_and_player_folders_but_keeps_clips_apart() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let replays = library.replays_dir().to_path_buf();
    let flat = replays.join("2026-11-13 20.51.45 EXAMPLE.NET.aeroreplay");
    let dated = library.unique_path_at(false, "dated", SAMPLE_START_MS);
    let mine = replays.join("best fights").join("ace.aeroreplay");
    let flat_clip = library.clips_dir().join("old clip.aeroreplay");
    let dated_clip = library.unique_path_at(true, "new clip", SAMPLE_START_MS);
    for (i, path) in [&flat, &dated, &mine, &flat_clip, &dated_clip]
        .iter()
        .enumerate()
    {
        write_replay(
            path,
            &meta(&format!("r{i}"), SAMPLE_START_MS + i as u64),
            50,
        );
    }
    // Not replays: ignored.
    std::fs::write(replays.join("notes.txt"), b"hello").unwrap();
    let full: Vec<PathBuf> = library
        .list_dir(false)
        .into_iter()
        .map(|e| e.path)
        .collect();
    let clips: Vec<PathBuf> = library.list_dir(true).into_iter().map(|e| e.path).collect();
    assert_eq!(
        names(&full, &replays),
        names(&[flat, dated, mine], &replays)
    );
    assert_eq!(
        names(&clips, &replays),
        names(&[flat_clip, dated_clip], &replays)
    );
    let usage = library.usage();
    assert_eq!((usage.full_count, usage.clip_count), (3, 2));
    assert_eq!(library.list().len(), 5);
}

/// The heart of the lane's safety promise: old flat files move into their date folders, a name
/// clash never overwrites anything, and files that may be in use, unfinished or unreadable stay.
#[test]
fn organize_moves_old_flat_files_and_never_loses_or_overwrites_one() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let replays = library.replays_dir().to_path_buf();
    let clips = library.clips_dir().to_path_buf();

    let start = SAMPLE_START_MS - 3_600_000;
    let a = replays.join("2026-11-13 20.05.23 a.aeroreplay");
    let b = replays.join("2026-11-13 20.05.23 b.aeroreplay");
    let clip = clips.join("2026-11-13 20.06.00 c.aeroreplay");
    write_replay(&a, &meta("a", start), 120);
    write_replay(&b, &meta("b", start + 1000), 80);
    let mut clip_meta = meta("c", start);
    clip_meta.kind = ReplayKind::Clip {
        start: 10_000_000,
        end: 37_000_000,
    };
    write_replay(&clip, &clip_meta, 60);

    // The date folder already holds a different file with b's name: it must survive untouched,
    // and b must arrive as "(2)".
    let b_dir = date_folder(&replays, start + 1000);
    std::fs::create_dir_all(&b_dir).unwrap();
    let squatter = b_dir.join("2026-11-13 20.05.23 b.aeroreplay");
    std::fs::write(&squatter, b"someone else's file").unwrap();

    // Left alone: changed a moment ago (could be another client's live recording), unfinished
    // (a crash: no footer), not a replay at all.
    let live = replays.join("live.aeroreplay");
    let mut writer = Writer::create(&live, &meta("live", start), small_blocks()).unwrap();
    writer.push(0, StreamId::Packet, b"x");
    writer
        .finish(meta("live", start))
        .wait(Duration::from_secs(10))
        .unwrap();
    let unfinished = replays.join("unfinished.aeroreplay");
    std::fs::copy(&a, &unfinished).unwrap();
    let len = std::fs::metadata(&unfinished).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&unfinished)
        .unwrap()
        .set_len(len - 20)
        .unwrap();
    backdate(&unfinished);
    let junk = replays.join("junk.aeroreplay");
    std::fs::write(&junk, b"not a replay").unwrap();
    backdate(&junk);

    let before: Vec<(PathBuf, Vec<u8>)> = [&a, &b, &clip]
        .iter()
        .map(|p| ((*p).clone(), std::fs::read(p).unwrap()))
        .collect();
    let started = Instant::now();
    let moved = library.organize();
    let took = started.elapsed();

    let to_of = |from: &Path| -> PathBuf {
        moved
            .iter()
            .find(|(f, _)| f == from)
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| panic!("{} was not moved", from.display()))
    };
    assert_eq!(
        moved.len(),
        3,
        "a, b and the clip move; nothing else: {moved:?}"
    );
    // The names are this computer's stamps (EST here: unchanged; elsewhere: restamped).
    let local = |name: &str, at: u64| mc_replay::restamp(name, at).unwrap_or(name.to_owned());
    assert_eq!(
        to_of(&a),
        date_folder(&replays, start).join(local("2026-11-13 20.05.23 a.aeroreplay", start))
    );
    if local("2026-11-13 20.05.23 b.aeroreplay", start + 1000) == "2026-11-13 20.05.23 b.aeroreplay"
    {
        assert_eq!(
            to_of(&b),
            b_dir.join("2026-11-13 20.05.23 b (2).aeroreplay"),
            "a name clash takes the next free name"
        );
    }
    // A clip's date is its key press (the window's end).
    assert_eq!(
        to_of(&clip),
        date_folder(&clips, start + 37_000)
            .join(local("2026-11-13 20.06.00 c.aeroreplay", start + 37_000))
    );
    for (from, bytes) in &before {
        assert!(!from.exists(), "{} still flat", from.display());
        assert_eq!(&std::fs::read(to_of(from)).unwrap(), bytes, "byte for byte");
    }
    assert_eq!(
        std::fs::read(&squatter).unwrap(),
        b"someone else's file",
        "an existing file is never overwritten"
    );
    for stays in [&live, &unfinished, &junk] {
        assert!(stays.exists(), "{} must stay where it is", stays.display());
    }
    // No placeholder is left behind anywhere.
    let all = library.list_dir(false);
    assert!(all.iter().all(|e| e.size_bytes > 0));
    assert!(
        library.organize().is_empty(),
        "a second pass has nothing to move"
    );
    eprintln!("organize took {took:?}");
}

/// A move that fails (here: another program holds the file open without letting it be moved)
/// leaves the file where it was, still listed, and removes the empty placeholder it made; the
/// next pass, once the file is free, moves it.
#[cfg(windows)]
#[test]
fn a_move_that_fails_leaves_the_file_listed_and_no_placeholder() {
    use std::os::windows::fs::OpenOptionsExt as _;
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let path = library
        .replays_dir()
        .join("2026-11-13 20.00.00 x.aeroreplay");
    write_replay(&path, &meta("x", SAMPLE_START_MS), 30);
    let bytes = std::fs::read(&path).unwrap();
    // FILE_SHARE_READ only: reading works, renaming does not.
    let held = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(1)
        .open(&path)
        .unwrap();
    assert!(
        library.organize().is_empty(),
        "nothing moves while the file is held"
    );
    assert!(path.exists());
    let target_dir = date_folder(library.replays_dir(), SAMPLE_START_MS);
    let left: Vec<_> = std::fs::read_dir(&target_dir)
        .map(|read| read.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    assert!(left.is_empty(), "no placeholder is left behind: {left:?}");
    assert_eq!(library.list_dir(false).len(), 1, "still listed where it is");
    drop(held);
    let moved = library.organize();
    assert_eq!(moved.len(), 1);
    assert_eq!(std::fs::read(&moved[0].1).unwrap(), bytes);
}

#[test]
fn names_stamped_in_another_zone_get_the_local_stamp_and_chosen_names_stay() {
    let remote = "2026-11-14 02.05.23 sky.example.org.aeroreplay";
    assert_eq!(
        restamp_in(remote, SAMPLE_START_MS, &est()).as_deref(),
        Some("2026-11-13 21.05.23 sky.example.org.aeroreplay")
    );
    // On a computer in the recording's zone the name is already right.
    assert_eq!(restamp_in(remote, SAMPLE_START_MS, &utc()), None);
    // India, +5:30: also a zone.
    let india = FixedOffset::east_opt(5 * 3600 + 1800).unwrap();
    assert_eq!(
        restamp_in(remote, SAMPLE_START_MS, &india).as_deref(),
        Some("2026-11-14 07.35.23 sky.example.org.aeroreplay")
    );
    // Our own local name (a second of slack either way): kept.
    assert_eq!(
        restamp_in(
            "2026-11-13 21.05.24 sky.example.org.aeroreplay",
            SAMPLE_START_MS,
            &est()
        ),
        None
    );
    // Not a whole zone offset away (7 minutes): a name someone typed; kept.
    assert_eq!(
        restamp_in(
            "2026-11-13 21.12.23 sky.example.org.aeroreplay",
            SAMPLE_START_MS,
            &est()
        ),
        None
    );
    // No stamp, or the stamp glued to more digits: kept.
    assert_eq!(
        restamp_in("best fight.aeroreplay", SAMPLE_START_MS, &est()),
        None
    );
    assert_eq!(
        restamp_in("2026-11-14 02.05.234.aeroreplay", SAMPLE_START_MS, &est()),
        None
    );
}

#[test]
fn default_titles_from_another_zone_show_local_time() {
    let remote = "sky.example.org \u{b7} Nov 14, 2:05 AM";
    assert_eq!(
        localize_title_in(remote, SAMPLE_START_MS, &est()).as_deref(),
        Some("sky.example.org \u{b7} Nov 13, 9:05 PM")
    );
    assert_eq!(localize_title_in(remote, SAMPLE_START_MS, &utc()), None);
    // Already local (the user's own recordings): unchanged.
    assert_eq!(
        localize_title_in(
            "SkyWars \u{b7} Dwarf Fortress \u{b7} Nov 13, 9:05 PM",
            SAMPLE_START_MS,
            &est()
        ),
        None
    );
    // A renamed title, or one with no time: unchanged.
    assert_eq!(
        localize_title_in("4 kills off spawn", SAMPLE_START_MS, &est()),
        None
    );
    assert_eq!(
        localize_title_in("Bed Wars \u{b7} EXAMPLE.NET", SAMPLE_START_MS, &est()),
        None
    );
    // Over New Year: 1:00 AM on Jan 1 in UTC is 8:00 PM on Dec 31 in EST.
    let new_year = 1_798_765_200_000; // 2027-01-01 01:00 UTC
    assert_eq!(
        localize_title_in("x \u{b7} Jan 1, 1:00 AM", new_year, &est()).as_deref(),
        Some("x \u{b7} Dec 31, 8:00 PM")
    );
}

#[test]
fn the_cap_deletes_only_where_the_library_writes_and_prunes_empty_date_folders() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let replays = library.replays_dir().to_path_buf();
    // Oldest of all, but in a folder the player made: never deleted, never counted.
    let kept = replays.join("keep these").join("final.aeroreplay");
    write_replay(&kept, &meta("final", 1_000), 100);
    let old_flat = replays.join("old.aeroreplay");
    write_replay(&old_flat, &meta("old", 2_000), 100);
    let day = 86_400_000;
    let dated: Vec<PathBuf> = (0..3)
        .map(|i| {
            let at = SAMPLE_START_MS + i * day;
            let path = library.unique_path_at(false, &format!("game {i}"), at);
            write_replay(&path, &meta(&format!("game {i}"), at), 100);
            path
        })
        .collect();
    let clip = library.unique_path_at(true, "clip", 3_000);
    write_replay(&clip, &meta("clip", 3_000), 100);

    let each = std::fs::metadata(&dated[0]).unwrap().len();
    // Room for two of the four managed recordings: the oldest two go.
    let deleted = library.enforce_cap(each * 2 + 10, None);
    assert_eq!(deleted, vec![old_flat.clone(), dated[0].clone()]);
    assert!(kept.exists() && clip.exists() && dated[1].exists() && dated[2].exists());
    // dated[0]'s day folder is gone (it was alone in it); its month folder stays (dated[1]
    // shares it), unless the days fell in different months.
    assert!(
        !dated[0].parent().unwrap().exists(),
        "the empty day folder is removed"
    );
    assert!(dated[1].parent().unwrap().exists());
    // Everything managed over the cap: all of it goes, the player's folder and clips stay.
    let deleted = library.enforce_cap(0, None);
    assert_eq!(deleted.len(), 2);
    assert!(kept.exists() && clip.exists());
    assert!(
        !replays.join("2026").exists(),
        "empty year, month and day folders are pruned"
    );
    // Deleting from the browser prunes too, and works in sub-folders.
    let lone = library.unique_path_at(false, "lone", SAMPLE_START_MS);
    write_replay(&lone, &meta("lone", SAMPLE_START_MS), 10);
    library.rename(&lone, "Renamed").unwrap();
    assert_eq!(read_info(&lone).unwrap().meta.title, "Renamed");
    library.delete(&lone).unwrap();
    assert!(!replays.join("2026").exists());
    // Outside the library: refused.
    let outside = dir.path().join("elsewhere.aeroreplay");
    write_replay(&outside, &meta("x", 1), 5);
    assert!(library.delete(&outside).is_err());
    assert!(outside.exists());
}

#[test]
fn bookmarks_are_counted_from_the_footer_and_survive_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let path = library.unique_path_at(false, "crash", SAMPLE_START_MS);
    let mut writer =
        Writer::create(&path, &meta("crash", SAMPLE_START_MS), small_blocks()).unwrap();
    let mut expected = Vec::new();
    for i in 0..400u64 {
        let time = i * 25_000;
        writer.push(time, StreamId::Packet, &[(i % 200) as u8; 60]);
        if i == 100 || i == 250 || i == 300 {
            let label = if i == 250 { "Clip" } else { BOOKMARK_LABEL };
            writer.push(time, StreamId::Marker, label.as_bytes());
            expected.push(Marker {
                time,
                label: label.to_owned(),
            });
        }
    }
    writer.flush();
    // Wait for the worker to write what was sealed.
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut last = u64::MAX;
    loop {
        std::thread::sleep(Duration::from_millis(60));
        let now = writer.stats().blocks;
        if now == last && now > 0 {
            break;
        }
        assert!(Instant::now() < deadline);
        last = now;
    }
    // The game dies here: a copy of the file as it is on disk has no footer.
    let crashed = path.with_file_name("crashed.aeroreplay");
    std::fs::copy(&path, &crashed).unwrap();
    let mut full = meta("crash", SAMPLE_START_MS);
    full.markers = expected.clone();
    writer.finish(full).wait(Duration::from_secs(10)).unwrap();
    std::fs::remove_file(&path).unwrap();
    backdate(&crashed);

    let before = read_info(&crashed).unwrap();
    assert!(before.recovered);
    assert!(
        before.meta.markers.is_empty(),
        "the header knows no markers"
    );
    assert_eq!(library.repair_unfinished(), vec![crashed.clone()]);
    let after = read_info(&crashed).unwrap();
    assert!(!after.recovered);
    assert_eq!(
        after.meta.markers, expected,
        "taken back from the marker records"
    );
    assert_eq!(
        after.meta.bookmark_count(),
        2,
        "the clip marker is not a bookmark"
    );
}

#[test]
fn the_title_instant_is_where_the_part_to_watch_begins() {
    let mut m = meta("x", 1_000_000);
    assert_eq!(m.title_unix_ms(), 1_000_000);
    m.view_start = 30_000_000;
    assert_eq!(m.title_unix_ms(), 1_030_000, "the start key's press");
    m.kind = ReplayKind::Clip {
        start: 50_000_000,
        end: 80_000_000,
    };
    assert_eq!(m.title_unix_ms(), 1_080_000, "the clip key's press");
    m.started_unix_ms = 0;
    assert_eq!(m.title_unix_ms(), 0, "unknown");
}

/// Real replays (copies; `MC_RA_REFS_DIR` points at a folder of them, never a live game
/// folder): flat files, among them one stamped in UTC. Copied into a temp
/// game folder, organized: each lands in its local day, byte for byte, and the UTC-stamped one
/// gets this computer's stamp.
#[test]
fn the_users_flat_replays_are_organized_into_their_local_days() {
    let Some(refs) = std::env::var_os("MC_RA_REFS_DIR") else {
        eprintln!("SKIP: set MC_RA_REFS_DIR to a folder of .aeroreplay copies");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    std::fs::create_dir_all(library.replays_dir()).unwrap();
    let mut originals = Vec::new();
    for item in std::fs::read_dir(&refs).unwrap().flatten() {
        let from = item.path();
        if from.extension().and_then(|e| e.to_str()) != Some("aeroreplay") {
            continue;
        }
        let to = library.replays_dir().join(item.file_name());
        std::fs::copy(&from, &to).unwrap();
        backdate(&to);
        originals.push((to, std::fs::read(&from).unwrap()));
    }
    assert!(!originals.is_empty(), "no replays in MC_RA_REFS_DIR");
    let moved = library.organize();
    assert_eq!(
        moved.len(),
        originals.len(),
        "every flat file moves: {moved:?}"
    );
    for (from, bytes) in &originals {
        let (_, to) = moved.iter().find(|(f, _)| f == from).expect("moved");
        let meta = read_info(to).unwrap().meta;
        let name = from.file_name().unwrap().to_str().unwrap();
        let expected = mc_replay::restamp(name, meta.title_unix_ms()).unwrap_or(name.to_owned());
        assert_eq!(
            to,
            &date_folder(library.replays_dir(), meta.title_unix_ms()).join(&expected)
        );
        assert_eq!(&std::fs::read(to).unwrap(), bytes, "{name}: byte for byte");
        eprintln!(
            "{name}  ->  {}",
            to.strip_prefix(dir.path()).unwrap().display()
        );
    }
    // In a UTC-5 zone the file stamped in UTC (02.05.23 on the 14th) is the 13th.
    let local = chrono::Local
        .timestamp_opt(1_794_621_923, 0)
        .single()
        .map(|t| t.offset().local_minus_utc());
    if local == Some(-5 * 3600) {
        let remote = moved
            .iter()
            .find(|(f, _)| f.to_string_lossy().contains("02.05.23"))
            .map(|(_, t)| t.clone());
        if let Some(remote) = remote {
            let day = Path::new("2026").join("11 November").join("13");
            assert!(remote.parent().is_some_and(|parent| parent.ends_with(&day)));
            let name = remote.file_name().and_then(|n| n.to_str()).unwrap_or("");
            assert!(name.starts_with("2026-11-13 21.05.23 "), "{name}");
        }
    }
    assert!(library.organize().is_empty());
}
