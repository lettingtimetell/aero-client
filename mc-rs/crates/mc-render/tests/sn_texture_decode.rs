//! A skin or cape PNG's header cannot make the decoder allocate the image it claims.
//!
//! Skins and capes are fetched for URLs a server supplies (unsigned head textures), and a PNG of a
//! few dozen bytes can claim a 4096 x 4096 surface. This binary counts heap bytes with its own
//! global allocator, so it holds a single test: libtest threads would otherwise share the counter.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use mc_render::entity::skin::{decode_cape, decode_skin, SkinModel};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to `System`; the counters only observe sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(live, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Heap bytes `work` adds at its peak, over what was live when it started.
fn peak_growth(work: impl FnOnce()) -> usize {
    let start = LIVE.load(Ordering::SeqCst);
    PEAK.store(start, Ordering::SeqCst);
    work();
    PEAK.load(Ordering::SeqCst).saturating_sub(start)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A PNG of a few dozen bytes whose header claims `side` x `side` RGBA pixels.
fn png_claiming(side: u32) -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        png.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
        png.extend_from_slice(&body);
        png.extend_from_slice(&crc32(&body).to_be_bytes());
    };
    let mut header = Vec::new();
    header.extend_from_slice(&side.to_be_bytes());
    header.extend_from_slice(&side.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &header);
    chunk(b"IDAT", &[0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
    chunk(b"IEND", &[]);
    png
}

#[test]
fn a_texture_header_cannot_make_the_decoder_allocate_its_claimed_size() {
    const MIB: usize = 1024 * 1024;
    let hostile = png_claiming(4096);
    assert!(hostile.len() < 100);

    // Negative control: the decode the skin and cape loaders used before allocates the whole
    // claimed surface (4096 x 4096 x 4 bytes = 64 MiB) before it finds there is no pixel data.
    let unbounded = peak_growth(|| {
        assert!(image::load_from_memory(&hostile).is_err());
    });
    assert!(
        unbounded >= 64 * MIB,
        "control allocated only {unbounded} bytes"
    );

    // The loaders now refuse the header first.
    let skin = peak_growth(|| assert!(decode_skin(&hostile, SkinModel::Classic).is_err()));
    let cape = peak_growth(|| assert!(decode_cape(&hostile).is_err()));
    eprintln!("peak heap growth: unbounded {unbounded} B, skin {skin} B, cape {cape} B");
    assert!(skin < MIB, "skin decode allocated {skin} bytes");
    assert!(cape < MIB, "cape decode allocated {cape} bytes");
}
