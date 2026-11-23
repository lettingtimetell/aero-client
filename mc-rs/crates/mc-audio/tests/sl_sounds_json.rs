//! A resource pack's `sounds.json` is someone else's file (and a server can offer a pack). Its
//! parsed form must stay within a small multiple of a real one's, whatever the file holds.
//!
//! This binary counts the bytes live on the test's own thread (a counting global allocator; the
//! library is unchanged), so the bound is measured.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use mc_assets::{FolderResourcePack, ResourceManager};

struct Counting;

thread_local! {
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

fn grew(bytes: usize) {
    let _ = LIVE.try_with(|live| {
        live.set(live.get() + bytes as i64);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(live.get())));
    });
}

fn shrank(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - bytes as i64));
}

// SAFETY: forwards every call to the system allocator unchanged; only counts on the side.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        // SAFETY: the caller's contract is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        grew(new_size);
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn peak_during<T>(run: impl FnOnce() -> T) -> (T, i64) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    let out = run();
    (out, PEAK.with(Cell::get) - before)
}

const MIB: i64 = 1 << 20;

/// A pack whose `sounds.json` is ~20 MB of one-letter sound names (it deflates to a few dozen
/// kilobytes in a zip): one event listing 5 million sounds. Before the cap, the same file at
/// 60 MB took the loading process to a 1.5 GB peak (measured).
#[test]
fn a_sounds_json_of_millions_of_tiny_entries_stays_bounded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let assets = dir.path().join("assets/minecraft");
    std::fs::create_dir_all(&assets).expect("pack folders");
    let entries = 5_000_000usize;
    let mut json = String::with_capacity(entries * 4 + 64);
    json.push_str(r#"{"bomb":{"category":"master","sounds":["#);
    for i in 0..entries {
        if i > 0 {
            json.push(',');
        }
        json.push_str("\"a\"");
    }
    json.push_str("]}}");
    let file_len = json.len() as i64;
    std::fs::write(assets.join("sounds.json"), json).expect("write");

    let mut resources = ResourceManager::new();
    resources.push_pack(Box::new(
        FolderResourcePack::open(dir.path()).expect("open folder pack"),
    ));
    let (registry, peak) = peak_during(|| mc_audio::SoundRegistry::load(&resources));
    let registry = registry.expect("loading never fails");
    eprintln!(
        "sounds.json {} MiB: peak {} MiB while loading, {} events",
        file_len / MIB,
        peak / MIB,
        registry.len()
    );
    assert!(
        peak < 128 * MIB,
        "loading sounds.json peaked at {} MiB",
        peak / MIB
    );
    assert!(registry.is_empty(), "the oversized file is skipped");
}

/// A real-sized `sounds.json` still loads.
#[test]
fn a_real_sized_sounds_json_still_loads() {
    let dir = tempfile::tempdir().expect("tempdir");
    let assets = dir.path().join("assets/minecraft");
    std::fs::create_dir_all(&assets).expect("pack folders");
    std::fs::write(
        assets.join("sounds.json"),
        r#"{"custom.click":{"category":"master","sounds":["custom/click"]}}"#,
    )
    .expect("write");
    std::fs::create_dir_all(assets.join("sounds/custom")).expect("sound folder");
    std::fs::write(assets.join("sounds/custom/click.ogg"), b"OggS").expect("sound file");
    let mut resources = ResourceManager::new();
    resources.push_pack(Box::new(
        FolderResourcePack::open(dir.path()).expect("open folder pack"),
    ));
    let registry = mc_audio::SoundRegistry::load(&resources).expect("load");
    assert_eq!(registry.len(), 1);
}
