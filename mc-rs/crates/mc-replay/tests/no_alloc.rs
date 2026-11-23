//! The game thread's side of recording does not allocate per record: proven with a counting
//! global allocator (this test binary's own; the library is unchanged).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::{Duration, Instant};

use mc_replay::{ReplayMeta, StreamId, Writer, WriterOptions};

struct Counting;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn count() {
    let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
}

// SAFETY: forwards every call to the system allocator unchanged; only counts on the side.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller's contract is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

#[test]
fn pushing_records_allocates_nothing_once_warm() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.aeroreplay");
    let options = WriterOptions {
        block_bytes: 64 * 1024,
        sync_interval: None,
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&path, &ReplayMeta::default(), options).unwrap();
    let packet = [0x15u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let chunk = vec![0x21u8; 9000];
    let start = Instant::now();
    let mut frame = 0u64;
    let mut run = |writer: &mut Writer, frames: u64| {
        for _ in 0..frames {
            frame += 1;
            let time = frame * 1000;
            for _ in 0..6 {
                writer.push(time, StreamId::Packet, &packet);
            }
            if frame.is_multiple_of(50) {
                writer.push(time, StreamId::Packet, &chunk);
            }
            writer.push(time, StreamId::Client(2), &packet[..8]);
            writer.tick(start + Duration::from_millis(frame));
            // Let real time pass now and then, as frames do: a block is sealed about every
            // 230 frames here, and the worker returns its buffer long before the next one.
            if frame.is_multiple_of(100) {
                std::thread::sleep(Duration::from_millis(3));
            }
        }
    };
    // Warm up: several blocks sealed, compressed and their buffers returned.
    run(&mut writer, 20_000);
    std::thread::sleep(Duration::from_millis(300));
    let blocks_before = writer.stats().blocks;

    let before = allocs();
    run(&mut writer, 20_000);
    let during = allocs() - before;
    std::thread::sleep(Duration::from_millis(300));
    let sealed = writer.stats().blocks - blocks_before;
    let records = 20_000 * 7 + 20_000 / 50;
    println!("{records} records, {sealed} blocks sealed, {during} allocations on this thread");
    assert!(
        sealed >= 10,
        "the run must cross many block boundaries: {sealed}"
    );
    // The only allocations left are the channel's own node blocks (one per 31 messages).
    assert!(
        during <= sealed / 8 + 2,
        "{during} allocations for {records} records"
    );

    writer
        .finish(ReplayMeta::default())
        .wait(Duration::from_secs(10))
        .unwrap();
}
