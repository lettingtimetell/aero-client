//! What the filled-map texture cache holds in memory.
//!
//! Each map image is 64 KiB of RGBA. Stored inline in the cache's hash table, every bucket, used
//! or not, cost 64 KiB: past 56 maps (a lobby's leaderboards and map walls) the table took a
//! fresh 8 MiB block for 128 buckets, 16 MiB past 112, and `clear` (leaving the world) kept it
//! for the rest of the session. Now the cache holds what its maps need, an update of a map it
//! has costs no allocation (servers resend map images often), and leaving gives everything back.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use mc_ui::map::{MapTextureCache, MAP_TEXTURE_BYTES};
use mc_world::MapData;

struct Counting;

thread_local! {
    /// Bytes this thread holds, and the allocations it made (only this test's thread matters,
    /// so the harness's own allocations on other threads cannot disturb the figures).
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

fn book(bytes: usize, sign: i64) {
    let _ = LIVE.try_with(|live| live.set(live.get() + sign * bytes as i64));
    if sign > 0 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

// SAFETY: forwards every call to `System` unchanged; the counters are thread-local cells.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            book(layout.size(), 1);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        book(layout.size(), -1);
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            book(layout.size(), -1);
            book(new_size, 1);
        }
        new
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn live() -> i64 {
    LIVE.with(Cell::get)
}

fn allocations() -> u64 {
    ALLOCATIONS.with(Cell::get)
}

#[test]
fn map_textures_hold_what_they_need_and_leaving_the_world_gives_it_back() {
    const MAPS: i32 = 60;
    let images = i64::from(MAPS) * MAP_TEXTURE_BYTES as i64;
    let maps: Vec<MapData> = (0..MAPS).map(MapData::new).collect();
    let baseline = live();

    let mut cache = MapTextureCache::default();
    for map in &maps {
        cache.update(map).expect("a blank map converts");
    }
    let held = live() - baseline;
    assert!(
        held >= images && held < images + 64 * 1024,
        "{MAPS} maps hold {held} bytes: their images are {images}, the rest is the table"
    );

    // A server sends a map's image again: no allocation, the same bytes held.
    let before = allocations();
    for map in &maps {
        cache.update(map).expect("a blank map converts");
    }
    assert_eq!(allocations(), before, "updating a known map allocates");
    assert_eq!(live() - baseline, held);
    assert_eq!(cache.len(), MAPS as usize);
    assert_eq!(cache.get(7).map(|texture| texture.map_id), Some(7));

    // Leaving the world: the images and the table are given back.
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(
        live() - baseline,
        0,
        "bytes still held after the cache was cleared"
    );
}
