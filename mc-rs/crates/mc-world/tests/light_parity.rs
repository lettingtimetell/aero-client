//! Differential test against the **real** 1.8.9 game.
//!
//! `javaref/LightRefGen.java` runs the recompiled vanilla classes under a JDK, builds a genuine
//! `net.minecraft.world.World` out of real `Chunk`s and `ExtendedBlockStorage`s, fills them with
//! `Chunk.fillChunk`, drives a scripted list of `World.setBlockState(pos, state, 3)` calls
//! through vanilla's own `checkLight` / `checkLightFor` / `getRawLight` / `relightBlock` /
//! `generateSkylightMap`, and dumps every light nibble in a probe box. This test replays the
//! identical scenario through `mc-world` and compares **every probed cell exactly** — no
//! tolerance, these are 4-bit integers.
//!
//! Both sides also compute an FNV-1a digest of every chunk payload they build. If the two
//! payload builders ever drift the digests fail first, so a green light comparison can never be
//! green because both sides were fed different terrain.
//!
//! Regenerate with:
//!
//! ```text
//! powershell -ExecutionPolicy Bypass -File crates/mc-world/javaref/generate.ps1
//! ```
//!
//! The fixtures are committed, so this test needs no JDK and no jars.

use std::path::PathBuf;

use mc_data::BlockState;
use mc_math::{BlockPos, ChunkPos};
use mc_world::{SkyBlock, World};

struct Reader<'a> {
    d: &'a [u8],
    o: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.d[self.o];
        self.o += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_be_bytes(self.d[self.o..self.o + 2].try_into().unwrap());
        self.o += 2;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_be_bytes(self.d[self.o..self.o + 4].try_into().unwrap());
        self.o += 4;
        v
    }
    fn i64(&mut self) -> i64 {
        let v = i64::from_be_bytes(self.d[self.o..self.o + 8].try_into().unwrap());
        self.o += 8;
        v
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let v = &self.d[self.o..self.o + n];
        self.o += n;
        v
    }
}

#[derive(Debug, Clone, Copy)]
struct Fill {
    x0: i32,
    y0: i32,
    z0: i32,
    x1: i32,
    y1: i32,
    z1: i32,
    state: u16,
}

impl Fill {
    fn contains(self, x: i32, y: i32, z: i32) -> bool {
        (self.x0..=self.x1).contains(&x)
            && (self.y0..=self.y1).contains(&y)
            && (self.z0..=self.z1).contains(&z)
    }
}

struct Fixture {
    name: String,
    has_sky: bool,
    radius: i32,
    fills: Vec<Fill>,
    digests: Vec<(ChunkPos, i64)>,
    ops: Vec<(BlockPos, u16)>,
    probe: (i32, i32, i32, i32, i32, i32),
    probes: Vec<u8>,
}

impl Fixture {
    /// The terrain state at a world position: the last fill that contains it, else air.
    /// Mirrors `LightRefGen.Scenario.stateAt`.
    fn state_at(&self, x: i32, y: i32, z: i32) -> u16 {
        self.fills
            .iter()
            .rev()
            .find(|f| f.contains(x, y, z))
            .map_or(0, |f| f.state)
    }

    /// One column's `S21PacketChunkData` payload. Byte-for-byte the same rule as
    /// `LightRefGen.Scenario.payload`, which the digests below prove.
    fn payload(&self, cx: i32, cz: i32) -> Vec<u8> {
        let sky_len = if self.has_sky { 16 * 2048 } else { 0 };
        let mut out = Vec::with_capacity(16 * 8192 + 16 * 2048 + sky_len + 256);
        for s in 0..16i32 {
            for y in 0..16i32 {
                for z in 0..16i32 {
                    for x in 0..16i32 {
                        let st = self.state_at((cx << 4) + x, (s << 4) + y, (cz << 4) + z);
                        out.push((st & 0xFF) as u8);
                        out.push((st >> 8) as u8);
                    }
                }
            }
        }
        out.resize(out.len() + 16 * 2048, 0); // block light
        out.resize(out.len() + sky_len, 0); // sky light (absent without sky — H6)
        out.resize(out.len() + 256, 1); // biomes: plains
        out
    }
}

/// 64-bit FNV-1a, matching `LightRefGen.fnv1a`. Java has no unsigned long, so the reference
/// value arrives as a signed `i64` and is compared as one.
fn fnv1a(data: &[u8]) -> i64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h as i64
}

fn load(name: &str) -> Fixture {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "light"]
        .iter()
        .collect::<PathBuf>()
        .join(format!("{name}.bin"));
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut r = Reader { d: &data, o: 0 };
    assert_eq!(r.bytes(8), b"MCLIGHT1", "bad magic in {}", path.display());
    let name_len = r.u16() as usize;
    let name = String::from_utf8(r.bytes(name_len).to_vec()).expect("utf-8 name");
    let has_sky = r.u8() == 1;
    let radius = r.i32();

    let fill_count = r.i32() as usize;
    let mut fills = Vec::with_capacity(fill_count);
    for _ in 0..fill_count {
        fills.push(Fill {
            x0: r.i32(),
            y0: r.i32(),
            z0: r.i32(),
            x1: r.i32(),
            y1: r.i32(),
            z1: r.i32(),
            state: r.u16(),
        });
    }

    let digest_count = r.i32() as usize;
    let mut digests = Vec::with_capacity(digest_count);
    for _ in 0..digest_count {
        digests.push((ChunkPos::new(r.i32(), r.i32()), r.i64()));
    }

    let op_count = r.i32() as usize;
    let mut ops = Vec::with_capacity(op_count);
    for _ in 0..op_count {
        ops.push((BlockPos::new(r.i32(), r.i32(), r.i32()), r.u16()));
    }

    let probe = (r.i32(), r.i32(), r.i32(), r.i32(), r.i32(), r.i32());
    let (x0, y0, z0, x1, y1, z1) = probe;
    let n = ((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)) as usize;
    let probes = r.bytes(n).to_vec();
    assert_eq!(r.o, data.len(), "trailing bytes in {}", path.display());

    Fixture {
        name,
        has_sky,
        radius,
        fills,
        digests,
        ops,
        probe,
        probes,
    }
}

/// Replay one fixture and compare every probed cell against vanilla's answer.
fn check(name: &str) {
    let f = load(name);
    assert_eq!(f.name, name);

    let mut w = World::new();
    // H6: this has to be set before any chunk packet, or the sky-light region is misread.
    w.set_has_sky_light(f.has_sky);
    for cx in -f.radius..=f.radius {
        for cz in -f.radius..=f.radius {
            let payload = f.payload(cx, cz);
            let pos = ChunkPos::new(cx, cz);
            let expected = f
                .digests
                .iter()
                .find(|(p, _)| *p == pos)
                .unwrap_or_else(|| panic!("no digest for {pos:?}"))
                .1;
            assert_eq!(
                fnv1a(&payload),
                expected,
                "{name}: the Rust and Java chunk-payload builders disagree at {pos:?}; \
                 nothing below this line would mean anything"
            );
            w.apply_chunk_data(pos, true, 0xFFFF, &payload)
                .expect("well-formed payload");
        }
    }

    for &(pos, state) in &f.ops {
        w.set_block(pos, BlockState::from_raw(state));
    }

    let (x0, y0, z0, x1, y1, z1) = f.probe;
    let mut i = 0usize;
    let mut total = 0usize;
    let mut samples = Vec::new();
    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                let want = f.probes[i];
                i += 1;
                let p = BlockPos::new(x, y, z);
                let got = (w.light_for(SkyBlock::Sky, p) << 4) | w.light_for(SkyBlock::Block, p);
                if got != want {
                    total += 1;
                    if samples.len() < 20 {
                        samples.push(format!(
                            "  {p:?}: vanilla sky={} block={}, mc-world sky={} block={}",
                            want >> 4,
                            want & 15,
                            got >> 4,
                            got & 15
                        ));
                    }
                }
            }
        }
    }
    assert!(
        total == 0,
        "{name}: {total} of {i} probed cells disagree with vanilla (first {}):\n{}",
        samples.len(),
        samples.join("\n")
    );
}

#[test]
fn torch_sealed_room() {
    check("torch_sealed_room");
}

#[test]
fn torch_removed() {
    check("torch_removed");
}

#[test]
fn two_torches_one_removed() {
    check("two_torches_one_removed");
}

#[test]
fn mined_shaft_and_tunnel() {
    check("mined_shaft_and_tunnel");
}

#[test]
fn shaft_resealed() {
    check("shaft_resealed");
}

#[test]
fn roof_over_open_ground() {
    check("roof_over_open_ground");
}

#[test]
fn across_a_chunk_boundary() {
    check("across_a_chunk_boundary");
}

#[test]
fn mixed_opacities() {
    check("mixed_opacities");
}

#[test]
fn nether_glowstone() {
    check("nether_glowstone");
}

/// A guard on the guard: the fixtures must actually contain light. A generator bug that emitted
/// an all-zero probe array would make every test above pass vacuously.
#[test]
fn the_fixtures_are_not_vacuously_dark() {
    for name in [
        "torch_sealed_room",
        "mined_shaft_and_tunnel",
        "roof_over_open_ground",
        "across_a_chunk_boundary",
        "mixed_opacities",
        "nether_glowstone",
    ] {
        let f = load(name);
        let lit = f.probes.iter().filter(|&&b| b != 0).count();
        assert!(
            lit > 50,
            "{name}: only {lit} of {} probed cells are lit at all",
            f.probes.len()
        );
    }
    // ...and `torch_removed` must be the opposite: vanilla really does take it all back out.
    let removed = load("torch_removed");
    assert!(
        removed.probes.iter().all(|&b| b == 0),
        "torch_removed should end completely dark"
    );
}

#[test]
fn roof_one_block() {
    check("roof_one_block");
}

#[test]
fn slab_only() {
    check("slab_only");
}
