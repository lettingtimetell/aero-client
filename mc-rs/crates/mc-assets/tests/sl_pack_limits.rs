//! Resource packs are other people's files (and a server can offer one). Nothing a pack holds may
//! make the client allocate far beyond the pack's own limits.
//!
//! This binary counts the bytes live on the test's own thread (a counting global allocator; the
//! library is unchanged), so every bound below is measured.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Cursor;

use mc_assets::{FolderResourcePack, ResourceLocation, ResourceManager, ResourcePack};

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

const MIB: i64 = 1 << 20;

#[test]
fn a_huge_pack_mcmeta_in_a_folder_pack_is_not_read_whole() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("assets/minecraft")).expect("pack folders");
    // 1 GiB of zeros (the file system fills it lazily): not JSON, and far past any real one.
    // Before the cap, opening the pack read all of it (measured: a 200 MiB file, 200 MiB).
    let mcmeta = std::fs::File::create(dir.path().join("pack.mcmeta")).expect("create");
    mcmeta.set_len(1 << 30).expect("size the file");
    drop(mcmeta);

    let (pack, peak) = peak_during(|| FolderResourcePack::open(dir.path()));
    let pack = pack.expect("the pack still opens");
    assert!(pack.metadata().is_none(), "no usable metadata in it");
    eprintln!("peak while opening: {} MiB", peak / MIB);
    // The per-resource cap (64 MiB): `read_to_end`'s doubling holds up to three times that for
    // a moment, whatever the file's size.
    assert!(peak < 256 * MIB, "opening read {peak} bytes");
}

/// `.mcmeta` JSON of `bytes` length: valid pack metadata plus an array of millions of zeros
/// (parsed whole, each element costs dozens of bytes).
fn mcmeta_bomb(bytes: usize) -> String {
    let mut json = String::from(r#"{"pack":{"pack_format":1,"description":"x"},"x":[0"#);
    while json.len() + 2 < bytes {
        json.push_str(",0");
    }
    json.push_str("]}");
    json
}

/// A zip pack's `pack.mcmeta` is parsed whenever the pack is opened (the
/// pack list opens every pack in the folder, selected or not); a texture's `.mcmeta` sidecar
/// whenever it loads. Both are limited to what metadata can need.
#[test]
fn oversized_mcmeta_files_are_refused_before_they_are_parsed() {
    use std::io::Write as _;

    let bomb = mcmeta_bomb(4 << 20);
    let dir = tempfile::tempdir().expect("tempdir");
    let zip_path = dir.path().join("bomb.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&zip_path).expect("zip"));
    let deflated = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("pack.mcmeta", deflated).expect("entry");
    zip.write_all(bomb.as_bytes()).expect("write");
    zip.start_file("assets/minecraft/textures/blocks/stone.png", deflated)
        .expect("entry");
    let mut stone_png = Cursor::new(Vec::new());
    image::RgbaImage::new(16, 16)
        .write_to(&mut stone_png, image::ImageFormat::Png)
        .expect("encode");
    zip.write_all(stone_png.get_ref()).expect("write");
    zip.start_file(
        "assets/minecraft/textures/blocks/stone.png.mcmeta",
        deflated,
    )
    .expect("entry");
    zip.write_all(bomb.as_bytes()).expect("write");
    zip.finish().expect("finish");
    eprintln!(
        "zip: {} KiB holding two 4 MiB .mcmeta files",
        std::fs::metadata(&zip_path).expect("size").len() >> 10
    );

    let (pack, peak) = peak_during(|| mc_assets::ZipResourcePack::open(&zip_path));
    let pack = pack.expect("the pack still opens");
    eprintln!("open: peak {} MiB", peak / MIB);
    assert!(pack.metadata().is_none(), "no usable metadata");
    assert!(peak < 8 * MIB, "opening the pack peaked at {peak} bytes");

    let mut resources = ResourceManager::new();
    resources.push_pack(Box::new(pack));
    let stone = ResourceLocation::minecraft("textures/blocks/stone.png");
    let (result, peak) = peak_during(|| resources.load_png_rgba(&stone));
    eprintln!("texture with its sidecar: peak {} MiB", peak / MIB);
    assert!(
        result.is_err(),
        "a texture whose sidecar is too large fails, as an unreadable one"
    );
    assert!(peak < 8 * MIB, "reading the texture peaked at {peak} bytes");
}

/// A real 1x1 PNG whose header is rewritten to claim `width` x `height` (the pixel data stays a
/// few bytes: a "PNG bomb" costs nothing to make).
fn png_claiming(width: u32, height: u32) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image::RgbaImage::new(1, 1)
        .write_to(&mut out, image::ImageFormat::Png)
        .expect("encode");
    let mut png = out.into_inner();
    // Signature (8), then IHDR: length (4), "IHDR" (4), width, height, ..., CRC.
    assert_eq!(&png[12..16], b"IHDR");
    png[16..20].copy_from_slice(&width.to_be_bytes());
    png[20..24].copy_from_slice(&height.to_be_bytes());
    let crc = crc32(&png[12..29]);
    png[29..33].copy_from_slice(&crc.to_be_bytes());
    png
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[test]
fn a_png_claiming_50000_by_50000_is_refused_before_it_is_allocated() {
    let bomb = png_claiming(50_000, 50_000);
    assert!(bomb.len() < 200, "{} bytes", bomb.len());

    let dir = tempfile::tempdir().expect("tempdir");
    let blocks = dir.path().join("assets/minecraft/textures/blocks");
    std::fs::create_dir_all(&blocks).expect("pack folders");
    std::fs::write(blocks.join("stone.png"), &bomb).expect("write");
    let mut resources = ResourceManager::new();
    resources.push_pack(Box::new(
        FolderResourcePack::open(dir.path()).expect("open"),
    ));
    let stone = ResourceLocation::minecraft("textures/blocks/stone.png");

    // A texture of its own (GUI, entity, sky... sheets).
    let (result, peak) = peak_during(|| resources.load_png_rgba(&stone));
    assert!(result.is_err(), "refused");
    assert!(peak < MIB, "load_png_rgba allocated {peak} bytes");
    let (result, peak) = peak_during(|| mc_assets::decode_texture_png(&bomb));
    assert!(result.is_err(), "refused");
    assert!(peak < MIB, "decode_texture_png allocated {peak} bytes");
    // The atlas sprites' own decode (`image`'s 512 MiB allocation cap).
    let (result, peak) =
        peak_during(|| image::load_from_memory_with_format(&bomb, image::ImageFormat::Png));
    assert!(result.is_err(), "refused");
    assert!(peak < MIB, "the sprite decode allocated {peak} bytes");
    // The pack itself still answers: the file is there, it just never decodes.
    assert!(resources.resource(&stone).is_ok());
}
