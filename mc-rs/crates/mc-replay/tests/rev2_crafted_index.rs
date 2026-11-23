//! A replay file's footer index is read before any block is checked.
//! Room for the records is reserved up front from it, so an index that claims more than the
//! file can hold must not make the viewer reserve hundreds of megabytes (or abort when the
//! allocator refuses) for a file of a few hundred bytes. Replays are shared between players.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use mc_replay::{Reader, RecordStore, ReplayMeta};

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

/// The most bytes live on this thread at once while `run` ran, above what was live before it.
fn peak_during<T>(run: impl FnOnce() -> T) -> (T, i64) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    let out = run();
    (out, PEAK.with(Cell::get) - before)
}

/// A finished file (header, footer, tail) with no block, whose index claims one block of
/// `u32::MAX` stored bytes, raw bytes and records.
fn claiming_index(meta_json: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"AEROREPL");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    out.extend_from_slice(meta_json.as_bytes());
    out.extend_from_slice(&crc32fast::hash(meta_json.as_bytes()).to_le_bytes());
    let footer_offset = out.len() as u64;
    let mut body = Vec::new();
    body.extend_from_slice(b"ARIX");
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&footer_offset.to_le_bytes()); // where the block would start
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // stored bytes
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // raw bytes
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // records
    body.extend_from_slice(&0u64.to_le_bytes()); // first time
    body.extend_from_slice(&0u64.to_le_bytes()); // last time
    body.push(1); // zstd
    body.push(1); // keyframe
    body.extend_from_slice(&1u16.to_le_bytes()); // packets
    body.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    body.extend_from_slice(meta_json.as_bytes());
    let crc = crc32fast::hash(&body);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&footer_offset.to_le_bytes());
    out.extend_from_slice(b"AEROEND1");
    out
}

const LIMIT: i64 = 8 << 20;

#[test]
fn an_index_claiming_more_than_the_file_reserves_little() {
    let bytes = claiming_index(&ReplayMeta::default().to_json());
    let file_len = bytes.len();
    assert!(file_len < 4096, "a small file: {file_len} bytes");
    let reader = Reader::from_bytes(bytes.clone()).expect("header and footer are well formed");
    assert_eq!(reader.blocks().len(), 1, "the footer's index is in use");
    assert_eq!(
        reader.blocks()[0].records,
        u32::MAX,
        "the claim is the index's"
    );

    let (held, peak) = peak_during(|| RecordStore::read_all(&reader).map(|_| ()));
    assert!(held.is_err(), "there is no block behind the index");
    assert!(
        peak < LIMIT,
        "holding every record reserved {peak} bytes for a {file_len}-byte file"
    );

    let reader = Reader::from_bytes(bytes).expect("as above");
    let (in_file, peak) = peak_during(|| RecordStore::read_all_in_file(reader, 1024).map(|_| ()));
    assert!(in_file.is_err(), "there is no block behind the index");
    assert!(
        peak < LIMIT,
        "keeping payloads in the file reserved {peak} bytes for a {file_len}-byte file"
    );
}
