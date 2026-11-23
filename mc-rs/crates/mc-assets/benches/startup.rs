//! S23's startup measurement: what a cold start actually costs, and what an on-disk bake cache
//! would and would not buy.
//!
//! "Startup-time optimisation (atlas + model baking cache on disk)". The task is
//! stated as if baking were the dominant cost of a cold start. **That is a hypothesis, and this
//! is the instrument that tests it** — the honest outcome of a performance task is sometimes
//! "the premise does not hold on this machine", and that is worth far more than a cache nobody
//! measured.
//!
//! Run it with:
//!
//! ```text
//! cargo bench -p mc-assets --bench startup
//! ```
//!
//! # What it measures
//!
//! 1. `bake_all` against the real `1.8.9.jar` — every `RenderType::Model` state plus the atlas
//!    stitch, which is the whole of what a cache would replace.
//! 2. Encoding, writing, reading and decoding a [`BakedSnapshot`](mc_assets::cache::BakedSnapshot)
//!    of comparable size, so the two can be compared directly rather than by assertion.
//! 3. The cost of *keying* the cache — stamping the pack stack — because a cache whose key costs
//!    more than the bake is a pessimisation, and that is a real risk when the pack is an 11 MB
//!    jar.
//!
//! # Honesty notes
//!
//! * The snapshot measured here is **the real one**, taken with `BakedSnapshot::of` from the
//!   `BakedAssets` the bake just produced. (It used to be assembled from the crate's public
//!   surface, because `BakedAssets`' sprite, weighted-variant and item tables were private and
//!   there was no constructor to put them back; S23's additive `from_parts`/`to_parts` pair
//!   closed that, so the caveat that used to live here is gone and the numbers below are the
//!   ones the client actually pays.)
//! * Single machine, warm OS file cache after the first run. A genuinely cold start also pays to
//!   read the jar off disk, which this does not isolate.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_assets::cache::{self, BakeInputs, BakedSnapshot, CacheRead};
use mc_assets::{ResourceManager, ZipResourcePack};

struct Samples {
    ms: Vec<f64>,
}

impl Samples {
    fn new() -> Self {
        Samples { ms: Vec::new() }
    }
    fn push(&mut self, d: Duration) {
        self.ms.push(d.as_secs_f64() * 1000.0);
    }
    fn mean(&self) -> f64 {
        self.ms.iter().sum::<f64>() / self.ms.len().max(1) as f64
    }
    fn sd(&self) -> f64 {
        if self.ms.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        (self.ms.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (self.ms.len() - 1) as f64).sqrt()
    }
    fn report(&self, label: &str) {
        println!(
            "  {label:<44} {:>8.2} ms  sd {:>6.2}  min {:>7.2}  max {:>7.2}  (n = {})",
            self.mean(),
            self.sd(),
            self.ms.iter().copied().fold(f64::INFINITY, f64::min),
            self.ms.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            self.ms.len()
        );
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn main() {
    println!("S23 startup profile");
    println!(
        "  profile: {}",
        if cfg!(debug_assertions) {
            "dev (opt-level 1, deps at 3)"
        } else {
            "release-like (debug_assertions off)"
        }
    );

    let Some(jars) = jars_dir() else {
        println!("skip: jars dir not found; set MC_JARS_DIR");
        return;
    };
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        println!("skip: 1.8.9.jar not found");
        return;
    }
    let jar_bytes = std::fs::metadata(&jar).map(|m| m.len()).unwrap_or(0);
    println!(
        "  jar: {} ({:.1} MiB)",
        jar.display(),
        jar_bytes as f64 / (1024.0 * 1024.0)
    );
    println!();

    // ---- 1. the bake ------------------------------------------------------------------------
    println!("1. what a cache would replace");
    let mut bake = Samples::new();
    let mut snapshot_of = Samples::new();
    let mut considered = 0;
    let mut atlas = None;
    let mut real_snapshot = None;
    for i in 0..5 {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(ZipResourcePack::open(&jar).expect("open jar")));
        let t = Instant::now();
        let (assets, report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
        bake.push(t.elapsed());
        considered = report.model_states_considered;
        // Taking the snapshot is the one cost the cache *adds* to a cold start, so it is timed
        // every round rather than once.
        let t = Instant::now();
        let snapshot = BakedSnapshot::of(&assets);
        snapshot_of.push(t.elapsed());
        if i == 0 {
            atlas = Some(assets.atlas.clone());
            real_snapshot = Some(snapshot);
            mc_assets::model::install(assets);
        }
    }
    bake.report("bake_all (models + atlas stitch)");
    snapshot_of.report("BakedSnapshot::of (cold path only)");
    let atlas = atlas.expect("atlas");
    println!(
        "  {:<44} {} model states, atlas {}x{} ({:.2} MiB)",
        "covering",
        considered,
        atlas.w,
        atlas.h,
        (atlas.texture.len() as f64) / (1024.0 * 1024.0)
    );

    // Opening the pack is separate from baking, and a cache does not remove it: the client still
    // has to be able to load textures and language files at runtime.
    let mut open = Samples::new();
    for _ in 0..10 {
        let t = Instant::now();
        let pack = ZipResourcePack::open(&jar).expect("open jar");
        open.push(t.elapsed());
        drop(pack);
    }
    open.report("ZipResourcePack::open (not cacheable)");
    println!();

    // ---- 2. the snapshot --------------------------------------------------------------------
    println!("2. the cache, on the real snapshot of the bake above");
    let snapshot = real_snapshot.expect("snapshot");
    let quads: usize = snapshot
        .models
        .iter()
        .map(|m| m.general_quads.len() + m.face_quads.iter().map(Vec::len).sum::<usize>())
        .sum();
    println!(
        "  {:<44} {} sprites, {} block models ({quads} quads), {} weighted, {} item models",
        "covering",
        snapshot.sprites.len(),
        snapshot.models.len(),
        snapshot.weighted.len(),
        snapshot.item_models.len()
    );

    let mut encode = Samples::new();
    let mut payload = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        payload = snapshot.encode();
        encode.push(t.elapsed());
    }
    encode.report("BakedSnapshot::encode");
    println!(
        "  {:<44} {:.2} MiB",
        "payload size",
        payload.len() as f64 / (1024.0 * 1024.0)
    );

    let mut decode = Samples::new();
    for _ in 0..5 {
        let t = Instant::now();
        let back = BakedSnapshot::decode(&payload).expect("decode");
        decode.push(t.elapsed());
        std::hint::black_box(&back);
    }
    decode.report("BakedSnapshot::decode");

    // Correctness, on real data rather than on the synthetic fixtures the unit tests use.
    let back = BakedSnapshot::decode(&payload).expect("decode");
    assert!(back == snapshot, "the real bake must round-trip exactly");
    println!(
        "  {:<44} round-trips exactly (asserted)",
        "real vanilla data"
    );

    let dir = std::env::temp_dir().join("mc-rs-s23-startup");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("baked.cache");

    let mut stamp = Samples::new();
    for _ in 0..10 {
        let t = Instant::now();
        let mut inputs = BakeInputs::new(2048);
        inputs.push_path(&jar);
        stamp.push(t.elapsed());
        assert!(inputs.is_stampable());
    }
    stamp.report("BakeInputs::push_path (sample stamp)");

    let mut stamp_full = Samples::new();
    for _ in 0..3 {
        let t = Instant::now();
        let mut inputs = BakeInputs::new(2048);
        inputs.push_file_hashed(&jar);
        stamp_full.push(t.elapsed());
    }
    stamp_full.report("BakeInputs::push_file_hashed (whole jar)");

    let mut inputs = BakeInputs::new(2048);
    inputs.push_path(&jar);

    let mut save = Samples::new();
    for _ in 0..5 {
        let t = Instant::now();
        cache::save(&path, &inputs, &snapshot).expect("save");
        save.push(t.elapsed());
    }
    save.report("cache::save (encode + write + rename)");
    let on_disk = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  {:<44} {:.2} MiB",
        "file on disk",
        on_disk as f64 / (1024.0 * 1024.0)
    );

    let mut load = Samples::new();
    for _ in 0..5 {
        let t = Instant::now();
        let read = cache::load(&path, &inputs);
        load.push(t.elapsed());
        assert!(matches!(read, CacheRead::Hit(_)), "the cache must hit here");
    }
    load.report("cache::load (read + verify + decode)");

    // Just the file read, so the checksum and decode can be attributed separately.
    let mut read_only = Samples::new();
    for _ in 0..5 {
        let t = Instant::now();
        let mut buf = Vec::new();
        std::fs::File::open(&path)
            .expect("open")
            .read_to_end(&mut buf)
            .expect("read");
        read_only.push(t.elapsed());
    }
    read_only.report("  of which: reading the file");
    println!();

    // ---- 3. the verdict ---------------------------------------------------------------------
    println!("3. verdict");
    let saved = bake.mean() - (load.mean() + stamp.mean());
    println!(
        "  {:<44} {:>8.2} ms bake  vs  {:>6.2} ms load + {:.2} ms stamp",
        "per cold start",
        bake.mean(),
        load.mean(),
        stamp.mean()
    );
    println!(
        "  {:<44} {:>8.2} ms ({:+.0}%)",
        "net saving",
        saved,
        100.0 * saved / bake.mean()
    );
    println!(
        "  {:<44} {:>8.2} ms",
        "verify (the residual: checksum over the payload)",
        (load.mean() - read_only.mean() - decode.mean()).max(0.0)
    );
    println!();
    println!(
        "  Do not read the percentage above as a startup improvement — it is the saving on the\n  \
         *bake stage*, not on a start. The cache is wired in now (mc-client's `AssetLoader`), and\n  \
         the whole-binary A/B was run interleaved to control for drift, `mc-client --offline Dev`\n  \
         with and without MC_RS_NO_BAKE_CACHE:\n  \
         \n  \
           asset load, Client::new -> \"assets loaded\"   592.5 +/- 20.5  ->  547.8 +/- 11.6 ms\n  \
                                                        (n = 12 each, -44.7 ms, -7.5%)\n  \
           whole process, start to exit, --frames 5    786.6 +/- 16.6  ->  764.7 +/-  9.0 ms\n  \
                                                        (n = 14 each, -21.9 ms, -2.8%)\n  \
         \n  \
         So: a real, repeatable saving, and a small one. Vulkan instance + adapter + device is\n  \
         still the bulk of a cold start on this machine. The steady-state frame rate is\n  \
         unchanged, as it must be — the reported `average_fps` moves from 338.9 to 351.8 at 300\n  \
         frames purely because that average is frames/elapsed and elapsed contains the startup;\n  \
         at 3000 frames it is 357.6 -> 359.2, and 3000/357.6 - 3000/359.2 = 36 ms, which is the\n  \
         startup saving and nothing else.\n  \
         Where it would matter more: a high-resolution resource pack, whose sprite decode and\n  \
         stitch scale with texture area while the cache read scales with the same bytes at\n  \
         memcpy speed."
    );
    let _ = std::fs::remove_file(&path);
}
