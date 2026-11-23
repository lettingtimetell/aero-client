//! The library as the browser's Folders view lists it. Every folder under a tab's root (empty
//! ones too), each named by its key; the clips kept apart; folder names that are odd but legal;
//! links and junctions below the two roots never followed (a junction is made with `mklink /J`,
//! which needs no special rights on Windows); and the check made before Explorer opens a folder.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use mc_replay::{
    folder_key, is_safe_folder, Library, ReplayMeta, StreamId, Writer, WriterOptions, MAX_DEPTH,
};

fn small_blocks() -> WriterOptions {
    WriterOptions {
        level: 3,
        block_bytes: 4096,
        block_interval: Duration::from_secs(3600),
        sync_interval: None,
    }
}

/// A small finished replay at `path` (its parent made as needed), backdated past the library's
/// two-minute "still being recorded" grace.
fn write_replay(path: &Path, title: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let meta = ReplayMeta {
        title: title.to_owned(),
        server: "mc.example.net".to_owned(),
        started_unix_ms: 1_794_621_923_186,
        ..ReplayMeta::default()
    };
    let mut writer = Writer::create(path, &meta, small_blocks()).expect("create");
    for i in 0..20u64 {
        writer.push(i * 50_000, StreamId::Packet, &[1, 2, 3]);
    }
    let mut last = meta.clone();
    last.duration = 1_000_000;
    writer
        .finish(last)
        .wait(Duration::from_secs(30))
        .expect("finish");
    let old = SystemTime::now() - Duration::from_secs(3600);
    let file = std::fs::File::options().write(true).open(path).unwrap();
    file.set_modified(old).unwrap();
}

fn keys(root: &Path, dirs: &[PathBuf]) -> Vec<String> {
    let mut keys: Vec<String> = dirs
        .iter()
        .map(|d| folder_key(root, d).expect("a folder under the root"))
        .collect();
    keys.sort();
    keys
}

fn titles(entries: &[mc_replay::LibraryEntry]) -> Vec<String> {
    let mut titles: Vec<String> = entries.iter().map(|e| e.meta.title.clone()).collect();
    titles.sort();
    titles
}

/// A directory junction at `link` to `target` (Windows: `mklink /J`). `false` when it could not
/// be made (another OS, or a refusal): the caller then skips that part, loudly.
fn junction(link: &Path, target: &Path) -> bool {
    if !cfg!(windows) {
        return false;
    }
    let made = std::process::Command::new("cmd")
        .arg("/C")
        .arg("mklink")
        .arg("/J")
        .arg(link)
        .arg(target)
        .output()
        .is_ok_and(|out| out.status.success());
    made && std::fs::symlink_metadata(link).is_ok_and(|m| m.file_type().is_symlink())
}

#[test]
fn the_tree_lists_every_folder_with_its_key_and_keeps_clips_apart() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let replays = library.replays_dir().to_path_buf();
    let clips = library.clips_dir().to_path_buf();
    write_replay(&replays.join("flat.aeroreplay"), "flat");
    write_replay(
        &replays
            .join("2026")
            .join("11 November")
            .join("13")
            .join("a.aeroreplay"),
        "dated",
    );
    write_replay(&replays.join("Best games").join("b.aeroreplay"), "mine");
    std::fs::create_dir_all(replays.join("Empty one")).unwrap();
    write_replay(&clips.join("2026").join("c.aeroreplay"), "clip");
    std::fs::create_dir_all(clips.join("Empty clips")).unwrap();
    // Not a replay: ignored (its folder still lists).
    std::fs::create_dir_all(replays.join("Notes")).unwrap();
    std::fs::write(replays.join("Notes").join("n.txt"), b"hello").unwrap();

    let full = library.list_tree(false);
    assert_eq!(titles(&full.entries), ["dated", "flat", "mine"]);
    assert_eq!(
        keys(&replays, &full.folders),
        [
            "2026",
            "2026/11 November",
            "2026/11 November/13",
            "Best games",
            "Empty one",
            "Notes"
        ],
        "every folder, empty ones too, and never the clips folder"
    );
    let clip_tree = library.list_tree(true);
    assert_eq!(titles(&clip_tree.entries), ["clip"]);
    assert!(clip_tree.entries.iter().all(|e| e.is_clip));
    assert_eq!(keys(&clips, &clip_tree.folders), ["2026", "Empty clips"]);
    // The same replays as the plain listing.
    assert_eq!(full.entries.len(), library.list_dir(false).len());
    assert_eq!(library.tab_root(true), clips.as_path());
    // A missing library lists nothing and makes nothing.
    let none = tempfile::tempdir().unwrap();
    let empty = Library::new(none.path()).list_tree(false);
    assert!(empty.entries.is_empty() && empty.folders.is_empty());
    assert!(!none.path().join("replays").exists());
}

#[test]
fn odd_but_legal_folder_names_keep_their_names_as_keys() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let replays = library.replays_dir().to_path_buf();
    let names = [
        "with spaces",
        "dots.in.name",
        "#hash %percent &amp",
        "\u{e9}t\u{e9} \u{2603} \u{1f3ae}",
        "\u{202e}rtl",
        "  leading spaces",
        "a",
    ];
    for name in names {
        write_replay(&replays.join(name).join("r.aeroreplay"), name);
    }
    let tree = library.list_tree(false);
    let mut want: Vec<String> = names.iter().map(|n| (*n).to_owned()).collect();
    want.sort();
    assert_eq!(keys(&replays, &tree.folders), want);
    let mut entry_keys: Vec<String> = tree
        .entries
        .iter()
        .map(|e| folder_key(&replays, e.path.parent().unwrap()).unwrap())
        .collect();
    entry_keys.sort();
    assert_eq!(entry_keys, want);
}

#[test]
fn keys_are_only_made_for_plain_folders_under_the_root() {
    let root = Path::new("game").join("replays");
    assert_eq!(folder_key(&root, &root).as_deref(), Some(""));
    assert_eq!(
        folder_key(&root, &root.join("a").join("b c")).as_deref(),
        Some("a/b c")
    );
    assert_eq!(folder_key(&root, Path::new("game")), None, "above the root");
    assert_eq!(folder_key(&root, Path::new("elsewhere")), None);
    assert_eq!(folder_key(&root, &root.join("..").join("x")), None, "`..`");
}

#[test]
fn folders_down_to_the_depth_limit_list_and_deeper_ones_do_not() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let mut deep = library.replays_dir().to_path_buf();
    for level in 1..=MAX_DEPTH + 1 {
        deep.push(format!("L{level}"));
        write_replay(&deep.join("r.aeroreplay"), &format!("depth {level}"));
    }
    let tree = library.list_tree(false);
    assert_eq!(tree.entries.len(), MAX_DEPTH, "one replay per listed level");
    assert_eq!(tree.folders.len(), MAX_DEPTH);
    assert!(!titles(&tree.entries).contains(&format!("depth {}", MAX_DEPTH + 1)));
    // Eight levels: a date folder (three) inside folders of one's own.
    assert_eq!(tree.entries.len(), 8);
}

#[test]
fn junctions_below_the_roots_are_never_followed() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    write_replay(&outside.path().join("secret.aeroreplay"), "outside");
    std::fs::create_dir_all(outside.path().join("deeper")).unwrap();
    write_replay(
        &outside.path().join("deeper").join("more.aeroreplay"),
        "outside deeper",
    );
    let library = Library::new(dir.path());
    let replays = library.replays_dir().to_path_buf();
    write_replay(&replays.join("mine").join("r.aeroreplay"), "inside");
    let link = replays.join("link out");
    if !junction(&link, outside.path()) {
        eprintln!("SKIP: no junction could be made here (mklink /J)");
        return;
    }
    // Through the junction the outside replays are reachable on disk...
    assert!(link.join("secret.aeroreplay").is_file());
    // ...but the library never goes through it.
    let tree = library.list_tree(false);
    assert_eq!(titles(&tree.entries), ["inside"]);
    assert_eq!(keys(&replays, &tree.folders), ["mine"]);
    assert_eq!(library.list().len(), 1);
    assert_eq!(library.usage().full_count, 1);
    // Explorer is never sent there, nor anywhere outside, nor to a file.
    assert!(is_safe_folder(&replays, &replays.join("mine")));
    assert!(is_safe_folder(&replays, &replays));
    assert!(!is_safe_folder(&replays, &link));
    assert!(!is_safe_folder(&replays, &link.join("deeper")));
    assert!(!is_safe_folder(&replays, outside.path()));
    assert!(!is_safe_folder(
        &replays,
        &replays.join("mine").join("r.aeroreplay")
    ));
    assert!(!is_safe_folder(&replays, &replays.join("missing")));
    assert!(!is_safe_folder(
        &replays,
        &replays.join("mine").join("..").join("..")
    ));

    // A clips folder that is itself a junction is the player's own (clips on another drive):
    // it lists as before (the recorder writes through it too), but a junction below it does not.
    let elsewhere = tempfile::tempdir().unwrap();
    write_replay(
        &elsewhere.path().join("2026").join("c.aeroreplay"),
        "clip on another drive",
    );
    let clips = library.clips_dir().to_path_buf();
    if !junction(&clips, elsewhere.path()) {
        eprintln!("SKIP: no clips junction could be made");
        return;
    }
    assert!(junction(
        &elsewhere.path().join("link deeper"),
        outside.path()
    ));
    let clip_tree = library.list_tree(true);
    assert_eq!(titles(&clip_tree.entries), ["clip on another drive"]);
    assert_eq!(keys(&clips, &clip_tree.folders), ["2026"]);
    assert!(is_safe_folder(&clips, &clips));
    assert!(is_safe_folder(&clips, &clips.join("2026")));
    assert!(!is_safe_folder(&clips, &clips.join("link deeper")));
    // The recordings' walk leaves the clips folder out as before.
    assert_eq!(titles(&library.list_tree(false).entries), ["inside"]);
}

/// How long a listing takes over a real library (a copy of a game folder), warm file cache. By
/// hand: `MC_BROWSER_REVIEW_GAMEDIR=<copy> cargo test -p mc-replay --test library_tree_links
/// listing_time -- --ignored --nocapture`.
#[test]
#[ignore = "timing over a real library; run by hand with MC_BROWSER_REVIEW_GAMEDIR"]
fn listing_time_over_a_real_library() {
    let Some(dir) = std::env::var_os("MC_BROWSER_REVIEW_GAMEDIR") else {
        eprintln!("skip: set MC_BROWSER_REVIEW_GAMEDIR");
        return;
    };
    let library = Library::new(Path::new(&dir));
    let mut listed = 0;
    let start = std::time::Instant::now();
    for _ in 0..20 {
        listed = library.list_tree(false).entries.len() + library.list_tree(true).entries.len();
    }
    let each = start.elapsed() / 20;
    let files = u32::try_from(listed.max(1)).unwrap_or(1);
    eprintln!(
        "{listed} replays listed in {each:?} ({:?} a replay, warm file cache)",
        each / files
    );
    for entry in library.list_tree(false).entries {
        let start = std::time::Instant::now();
        for _ in 0..20 {
            let _ = mc_replay::read_info(&entry.path);
        }
        eprintln!(
            "  {:>6} KB: {:?} a read",
            entry.size_bytes / 1000,
            start.elapsed() / 20
        );
    }
}

#[test]
fn a_symbolic_link_to_a_folder_is_not_followed_either() {
    #[cfg(windows)]
    {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        write_replay(&outside.path().join("o.aeroreplay"), "outside");
        let library = Library::new(dir.path());
        let replays = library.replays_dir().to_path_buf();
        write_replay(&replays.join("r.aeroreplay"), "inside");
        let link = replays.join("symlinked");
        if std::os::windows::fs::symlink_dir(outside.path(), &link).is_err() {
            eprintln!("SKIP: symbolic links need Developer Mode or admin rights here");
            return;
        }
        let tree = library.list_tree(false);
        assert_eq!(titles(&tree.entries), ["inside"]);
        assert!(tree.folders.is_empty());
        assert!(!is_safe_folder(&replays, &link));
    }
}
