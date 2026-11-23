//! Phase 1**: what `mc-world`'s chunk storage actually costs, measured
//! before any palette is built.
//!
//! T4.1 asks for chunk-storage palette compaction and blockstate dedup, unmeasured until a heap
//! profile exists; §3.3 marks it **[want, low]**.
//! This bench is that heap profile, plus the read/build A/B that decides whether the compaction
//! is affordable — because `mesh_walk` reads every one of a section's 4096 cells through
//! `World::block_at`, and `mesh_section` serial is 1.155 ms/section (§1.6). A memory win that
//! costs meshing time lands directly on the join burst, which §1.5 identifies as the *only*
//! thing in the whole document that is visible at 360 Hz.
//!
//! Run it with:
//!
//! ```text
//! cargo bench -p mc-world --bench chunk_memory
//! cargo bench -p mc-world --bench chunk_memory -- --reps 40 --max-columns 1089
//! ```
//!
//! # What is measured, and what is not
//!
//! * **Static layout** — `size_of`, from the compiler. Measured, and asserted, so a future
//!   layout change fails this bench loudly instead of silently invalidating its table.
//! * **Loaded-set bytes** — the worlds are *actually built* at each of F3's column counts, so
//!   the totals are construction-backed rather than arithmetic alone. `fill_from_packet` writes
//!   every byte of every section, so none of it is a lazily-committed page.
//! * **Palette headroom** — computed over **real 1.8.8 server bytes**, decoded from
//!   `mc-parity`'s committed captures through this crate's own `World::apply_chunk_data`.
//!   §1.2 of the plan is explicit that that server's spawn is *a minigame lobby, not terrain*,
//!   so its entropy is **lower** than survival terrain's, which makes every palette saving
//!   reported from it an **optimistic bound**. A synthetic dense-terrain census is printed
//!   beside it as the pessimistic end, and is labelled `assumed`.
//! * **Read, build and write cost** — a paletted mirror is built from the same real sections and
//!   exercised with the same access patterns, **interleaved in one process** (§0 rule 1), with
//!   an **A/A control** so the reader can see what "the change did nothing" measures like
//!   (§0 rule 2). The paletted read is verified against the real storage at **every** position
//!   of every section before any timing is taken.
//!
//! # Which numbers here are solid and which are provisional
//!
//! Everything in §1, §2, §3 and §6 is **byte accounting**: `size_of`, allocation counting and
//! arithmetic over content that was actually decoded. None of it is a timing, so none of it is
//! affected by what else the machine is doing.
//!
//! Every figure in §4 and §5 **is** a timing, and the run they were first taken on had four
//! other `cargo bench` runs on the same 16 threads. They are therefore marked
//! **provisional** and want a re-run on a quiet machine. What survives the contention is the
//! *shape*: each comparison is interleaved in one process and carries an A/A control, so a
//! difference larger than the control band is real even when the absolute means drift. Three
//! runs at 20/30/40 reps put the A/A control between −1.5 % and +0.8 % while `E4 vs E3` stayed
//! between +211 % and +226 % — the conclusions below rest on gaps of that size, not on gaps the
//! contention could manufacture.
//!
//! # Relationship to F5
//!
//! F5 records that `mesh_section` is called **8 × frames rendered** and never settles —
//! 786 986 calls in 390 s, ≈2018 calls/s, ≈450 re-meshes of every section of a static world.
//! Chunk reads are therefore currently paid far more often than they should be, which raises the
//! price of any change that slows `block_at` down. It does **not** raise the value of the memory
//! saving, and F5 is a defect that will be fixed, so §4 reports the delta **per `mesh_section`
//! call** and scales it by F5's observed rate only as an aside — sizing a storage decision
//! against a bug would be a mistake.
//!
//! # What no part of this can measure, and the cross-check that covers it
//!
//! Allocator-level overhead per `Box<ChunkSection>`. `mc-world` is `#![forbid(unsafe_code)]` and
//! the no-`unsafe` rule exempts only `mc-render`, so there is no `#[global_allocator]` here — a counting
//! allocator needs `unsafe impl GlobalAlloc`. The omission is one-directional: the real resident
//! figure can only be *larger* than the one printed.
//!
//! It was bounded from outside instead, by polling the process's working set while it runs. Two
//! runs differing only in how many columns §2 builds:
//!
//! ```powershell
//! $p = Start-Process $exe -ArgumentList "--reps","1","--read-columns","0","--max-columns","4225" `
//!        -RedirectStandardOutput "$env:TEMP\ws.txt" -NoNewWindow -PassThru
//! $peak = 0; while (-not $p.HasExited) { $p.Refresh(); if ($p.WorkingSet64 -gt $peak) { $peak = $p.WorkingSet64 } }
//! ```
//!
//! | `--max-columns` | §2 predicts | peak working set | over |
//! |---|---|---|---|
//! | 289  | 20.43 MiB  | 22.13 MiB  | +1.70 MiB |
//! | 4225 | 298.62 MiB | 312.02 MiB | +13.40 MiB (+4.5 %) |
//!
//! The **delta** between the two runs cancels the process baseline: 289.89 MiB measured against
//! 278.19 MiB predicted, i.e. the arithmetic in §2 is right to within **+4.2 %**, and low, as
//! predicted. The residue is the `HashMap` table (§2 projects +2.9 MiB across those two sizes)
//! plus Windows' heap rounding on 23 616 extra 12 288-byte blocks. `--hold-ms N` holds the
//! process open at the end if a snapshot is easier to take than a poll.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_data::BlockState;
use mc_math::{BlockPos, ChunkPos};
use mc_world::{expected_chunk_data_len, Chunk, ChunkSection, World};

/// §1.6/F3: **162.44 MiB of resident vertex bytes for 2500 sections**. Every ratio below is
/// taken against this, because it is the number that decides whether CPU chunk storage is worth
/// attacking at all.
const F3_VERTEX_BYTES_TOTAL: f64 = 162.44 * 1024.0 * 1024.0;
/// Sections behind [`F3_VERTEX_BYTES_TOTAL`] (`frame_loop --radius 12`).
const F3_SECTIONS: f64 = 2500.0;
/// §1.6: `mesh_section` serial, one thread, ms per section.
const F3_MESH_MS_PER_SECTION: f64 = 1.155;
/// F5: 786 986 `mesh_section` calls in 390 s on a static world. Reported as an aside only.
const F5_CALLS_PER_SECOND: f64 = 786_986.0 / 390.0;

/// F3's projection columns, so this table lies beside it: (render distance, columns).
const F3_COLUMNS: [(u32, usize); 4] = [(8, 289), (12, 625), (16, 1089), (32, 4225)];
/// F3's "non-vacant sections @ ~6/column" assumption, carried over unchanged for comparability.
const F3_SECTIONS_PER_COLUMN: usize = 6;

const KIB: f64 = 1024.0;
const MIB: f64 = 1024.0 * 1024.0;

// =============================================================================================
// Statistics — mean, sd, min, max, n. A best-of-N is not a result (§0 rule 3).
// =============================================================================================

struct Samples {
    ms: Vec<f64>,
}

impl Samples {
    fn empty() -> Self {
        Samples { ms: Vec::new() }
    }

    fn push(&mut self, d: Duration) {
        self.ms.push(d.as_secs_f64() * 1000.0);
    }

    fn mean(&self) -> f64 {
        if self.ms.is_empty() {
            return 0.0;
        }
        self.ms.iter().sum::<f64>() / self.ms.len() as f64
    }

    fn sd(&self) -> f64 {
        if self.ms.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let var =
            self.ms.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (self.ms.len() - 1) as f64;
        var.sqrt()
    }

    fn min(&self) -> f64 {
        self.ms.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.ms.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    fn report(&self, label: &str) {
        println!(
            "  {label:<36} {:>9.4} ms  sd {:>7.4}  min {:>9.4}  max {:>9.4}  (n = {})",
            self.mean(),
            self.sd(),
            self.min(),
            self.max(),
            self.ms.len()
        );
    }
}

/// Mean/sd/min/p50/p90/max over integer observations.
struct Spread {
    v: Vec<usize>,
}

impl Spread {
    fn of(mut v: Vec<usize>) -> Self {
        v.sort_unstable();
        Spread { v }
    }

    fn mean(&self) -> f64 {
        if self.v.is_empty() {
            return 0.0;
        }
        self.v.iter().sum::<usize>() as f64 / self.v.len() as f64
    }

    fn sd(&self) -> f64 {
        if self.v.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let var = self
            .v
            .iter()
            .map(|&x| (x as f64 - m) * (x as f64 - m))
            .sum::<f64>()
            / (self.v.len() - 1) as f64;
        var.sqrt()
    }

    fn at(&self, q: f64) -> usize {
        if self.v.is_empty() {
            return 0;
        }
        let i = ((self.v.len() - 1) as f64 * q).round() as usize;
        self.v[i]
    }

    fn report(&self, label: &str) {
        if self.v.is_empty() {
            println!("  {label}: no samples");
            return;
        }
        println!(
            "  {label:<36} mean {:>7.2}  sd {:>6.2}  min {:>4}  p50 {:>4}  p90 {:>4}  max {:>4}  (n = {})",
            self.mean(),
            self.sd(),
            self.v[0],
            self.at(0.5),
            self.at(0.9),
            self.v[self.v.len() - 1],
            self.v.len()
        );
    }
}

// =============================================================================================
// A paletted section, as 1.13+ `PalettedContainer` would store one
// =============================================================================================

/// Bit-packed indices into a linear palette of the distinct states a section actually contains.
///
/// Deliberately built in the form that is **fastest to read**, not the form that is smallest:
/// the index width is rounded up to a power of two (1/2/4/8/16), so unpacking is a shift and a
/// mask, with no division and no word-straddling. Vanilla's `BitArray` uses an arbitrary width
/// and a magic-number reciprocal divide — a little denser, materially slower. Since this bench
/// exists to decide whether a palette is worth its read cost, the palette gets the benefit of
/// every doubt: the timings are a **lower bound** on what a real implementation would cost, and
/// the byte figures are within a couple of percent of the arbitrary-width form on the measured
/// data (both are printed).
///
/// A single-state section stores `bits = 0` and no words at all, matching vanilla's
/// `SingleValuePalette` — the case a palette wins most on.
#[derive(Clone)]
struct PalettedSection {
    palette: Vec<u16>,
    /// 0 (uniform), 1, 2, 4, 8 or 16.
    bits: u32,
    /// `log2(64 / bits)`: how far to shift a cell index to reach its word.
    word_shift: u32,
    /// `64 / bits - 1`.
    lane_mask: usize,
    value_mask: u64,
    words: Vec<u64>,
}

impl PalettedSection {
    /// Builds in one pass with an insertion-ordered linear palette — vanilla's `LinearPalette`,
    /// and the cheapest correct build there is at the measured entropy (a handful of entries
    /// means a handful of comparisons per cell). A sort-based build was tried first and is
    /// several times slower; using it would have overstated the decode cost measured in §5.
    fn build(cells: &[u16; 4096]) -> Self {
        let mut palette: Vec<u16> = Vec::with_capacity(16);
        let mut indices = [0u16; 4096];
        for (i, &c) in cells.iter().enumerate() {
            let p = match palette.iter().position(|&x| x == c) {
                Some(p) => p,
                None => {
                    palette.push(c);
                    palette.len() - 1
                }
            };
            indices[i] = p as u16;
        }
        palette.shrink_to_fit();
        Self::pack(palette, &indices)
    }

    fn pack(palette: Vec<u16>, indices: &[u16; 4096]) -> Self {
        let bits = pow2_bits(palette.len());
        if bits == 0 {
            return PalettedSection {
                palette,
                bits: 0,
                word_shift: 0,
                lane_mask: 0,
                value_mask: 0,
                words: Vec::new(),
            };
        }
        let per_word = 64 / bits;
        let mut words = vec![0u64; 4096 / per_word as usize];
        for (i, &index) in indices.iter().enumerate() {
            let lane = (i as u32 % per_word) * bits;
            words[i / per_word as usize] |= u64::from(index) << lane;
        }
        PalettedSection {
            palette,
            bits,
            word_shift: per_word.trailing_zeros(),
            lane_mask: per_word as usize - 1,
            value_mask: (1u64 << bits) - 1,
            words,
        }
    }

    #[inline]
    fn get(&self, index: usize) -> u16 {
        if self.bits == 0 {
            return self.palette[0];
        }
        let word = self.words[index >> self.word_shift];
        let lane = (index & self.lane_mask) as u32 * self.bits;
        self.palette[((word >> lane) & self.value_mask) as usize]
    }

    #[inline]
    fn raw_index(&self, index: usize) -> u16 {
        if self.bits == 0 {
            return 0;
        }
        let word = self.words[index >> self.word_shift];
        let lane = (index & self.lane_mask) as u32 * self.bits;
        ((word >> lane) & self.value_mask) as u16
    }

    #[inline]
    fn write_index(&mut self, index: usize, pi: u64) {
        let lane = (index & self.lane_mask) as u32 * self.bits;
        let w = &mut self.words[index >> self.word_shift];
        *w = (*w & !(self.value_mask << lane)) | (pi << lane);
    }

    /// Writes one cell. A state already in the palette is a shift and a mask; a state that is
    /// not either appends to the palette (free while the width still holds it) or forces a
    /// **repack of all 4096 indices**. That repack is the block-change (`S23`/`S22`) cost a
    /// palette adds, which is why both arms are measured separately.
    fn set(&mut self, index: usize, value: u16) {
        if let Some(pi) = self.palette.iter().position(|&x| x == value) {
            if self.bits != 0 {
                self.write_index(index, pi as u64);
            }
            return;
        }
        // The new state still fits the current index width: append and write one lane.
        if self.bits != 0 && pow2_bits(self.palette.len() + 1) == self.bits {
            self.palette.push(value);
            self.write_index(index, (self.palette.len() - 1) as u64);
            return;
        }
        // The width has to grow, so every index moves.
        let mut indices = [0u16; 4096];
        for (i, slot) in indices.iter_mut().enumerate() {
            *slot = self.raw_index(i);
        }
        let mut palette = std::mem::take(&mut self.palette);
        palette.push(value);
        indices[index] = (palette.len() - 1) as u16;
        *self = Self::pack(palette, &indices);
    }

    /// Heap bytes this representation owns, plus its own inline size.
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.words.len() * 8 + self.palette.capacity() * 2
    }

    /// Index bytes only, so the block-array comparison is like for like.
    fn index_bytes(&self) -> usize {
        self.words.len() * 8
    }
}

/// The smallest power-of-two bit width that indexes `distinct` entries; 0 for at most one.
fn pow2_bits(distinct: usize) -> u32 {
    match distinct {
        0 | 1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        17..=256 => 8,
        _ => 16,
    }
}

/// The arbitrary width vanilla actually uses, for the byte comparison only.
fn exact_bits(distinct: usize) -> u32 {
    if distinct <= 1 {
        return 0;
    }
    let mut b = 1u32;
    while (1usize << b) < distinct {
        b += 1;
    }
    b
}

/// A paletted mirror of [`Chunk`]: same shape, same sixteen optional sections.
struct PalettedChunk {
    sections: [Option<Box<PalettedSection>>; 16],
    block_light: [Option<Box<[u8; 2048]>>; 16],
    sky_light: [Option<Box<[u8; 2048]>>; 16],
    biomes: [u8; 256],
}

impl PalettedChunk {
    fn empty() -> Self {
        PalettedChunk {
            sections: std::array::from_fn(|_| None),
            block_light: std::array::from_fn(|_| None),
            sky_light: std::array::from_fn(|_| None),
            biomes: [0xFF; 256],
        }
    }
}

/// A paletted mirror of [`World`]'s chunk map, so the read A/B runs at the level the mesher
/// actually reads at (`World::block_at`, `HashMap` lookup included) rather than only at the
/// level the palette changes.
struct PalettedWorld {
    chunks: HashMap<ChunkPos, PalettedChunk>,
}

impl PalettedWorld {
    /// The same decision tree as `World::block_at`, step for step.
    #[inline]
    fn block_at(&self, p: BlockPos) -> u16 {
        if !(0..256).contains(&p.y) {
            return 0;
        }
        let (x, z) = ((p.x & 15) as usize, (p.z & 15) as usize);
        match self.chunks.get(&ChunkPos::new(p.x >> 4, p.z >> 4)) {
            Some(chunk) => match &chunk.sections[(p.y >> 4) as usize] {
                Some(section) => section.get(local_index(x, (p.y & 15) as usize, z)),
                None => 0,
            },
            None => 0,
        }
    }

    fn bytes(&self) -> usize {
        let mut total = 0;
        for chunk in self.chunks.values() {
            total += std::mem::size_of::<PalettedChunk>();
            for s in chunk.sections.iter().flatten() {
                total += s.bytes();
            }
            total += chunk.block_light.iter().flatten().count() * 2048;
            total += chunk.sky_light.iter().flatten().count() * 2048;
        }
        total
    }
}

/// `ExtendedBlockStorage`'s index: `y << 8 | z << 4 | x`.
#[inline]
const fn local_index(x: usize, y: usize, z: usize) -> usize {
    (y << 8) | (z << 4) | x
}

// =============================================================================================
// A cheap integer hasher, for one control arm only
// =============================================================================================

/// `rustc-hash`'s multiply-rotate hash, written out because the workspace has no such
/// dependency and this bench must not add one to the crate's *library*.
///
/// It exists to answer one question: how much of a `World::block_at` call is `SipHash-1-3` on a
/// `ChunkPos`? `std`'s default hasher is DoS-resistant, which is the right default for
/// untrusted keys and the wrong one for a key the client generates itself and probes 4096 times
/// per section. This is a **measurement, not a proposal** — swapping `World`'s hasher is a
/// separate decision with its own review.
#[derive(Default, Clone, Copy)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(u64::from(b));
        }
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.add(i as u32 as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

type FxBuild = std::hash::BuildHasherDefault<FxHasher>;

/// Lets one generic sweep run over maps whose value is a `Chunk`, a `Box<Chunk>` or a `&Chunk`,
/// so the three probe arms differ **only** in the map's entry size and hasher — same loop body,
/// same monomorphised shape, nothing for the optimiser to treat differently.
trait AsChunk {
    fn as_chunk(&self) -> &Chunk;
}

impl AsChunk for Chunk {
    #[inline]
    fn as_chunk(&self) -> &Chunk {
        self
    }
}

impl AsChunk for Box<Chunk> {
    #[inline]
    fn as_chunk(&self) -> &Chunk {
        self
    }
}

impl AsChunk for &Chunk {
    #[inline]
    fn as_chunk(&self) -> &Chunk {
        self
    }
}

/// `World::block_at`'s decision tree, over any of the maps above. One probe per cell, exactly
/// like `mesh_walk`.
fn sweep_chunk_map<V: AsChunk, S: std::hash::BuildHasher>(
    map: &HashMap<ChunkPos, V, S>,
    positions: &[(ChunkPos, i32)],
) -> u64 {
    let mut acc = 0u64;
    for &(pos, sy) in positions {
        for y in 0..16i32 {
            for z in 0..16i32 {
                for x in 0..16i32 {
                    let bp = BlockPos::new(pos.x * 16 + x, sy * 16 + y, pos.z * 16 + z);
                    let id = if (0..256).contains(&bp.y) {
                        match map.get(&ChunkPos::new(bp.x >> 4, bp.z >> 4)) {
                            Some(c) => {
                                c.as_chunk()
                                    .block_at((bp.x & 15) as u8, bp.y as u8, (bp.z & 15) as u8)
                                    .id
                            }
                            None => 0,
                        }
                    } else {
                        0
                    };
                    acc = acc.wrapping_add(u64::from(id));
                }
            }
        }
    }
    acc
}

// =============================================================================================
// Reading `mc-parity`'s committed captures — real 1.8.8 server chunk bytes
// =============================================================================================

/// One column payload lifted straight off the wire, in the shape `World::apply_chunk_data` takes.
struct RawColumn {
    pos: ChunkPos,
    ground_up: bool,
    bitmask: u16,
    has_sky_light: bool,
    data: Vec<u8>,
}

fn capture_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_CAPTURE_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let p = here
        .parent()?
        .join("mc-parity")
        .join("tests")
        .join("fixtures");
    p.is_dir().then_some(p)
}

fn be_u16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(o..o + 2)?.try_into().ok()?))
}

fn be_u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

fn be_i32(b: &[u8], o: usize) -> Option<i32> {
    Some(i32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

fn varint(b: &[u8], i: &mut usize) -> Option<u32> {
    let mut v = 0u32;
    for shift in 0..5u32 {
        let byte = *b.get(*i)?;
        *i += 1;
        v |= u32::from(byte & 0x7F) << (shift * 7);
        if byte & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

/// Splits an `MCCAP001` container into its `(direction, state, frame)` records.
///
/// The container layout is documented in `mc-parity`'s `capture.rs`; it is re-read here rather
/// than linked against, because `mc-parity` depends on this crate (and on `mc-render`), and a
/// bench is not worth that cycle. This parser is total: every malformed shape returns `None` and
/// the capture is skipped, since the point is a measurement, not a decoder.
fn capture_frames(bytes: &[u8]) -> Option<Vec<(u8, u8, Vec<u8>)>> {
    if bytes.get(0..8)? != b"MCCAP001" {
        return None;
    }
    let flags = *bytes.get(9)?;
    let addr_len = be_u16(bytes, 22)? as usize;
    let mut o = 24 + addr_len;
    let frame_count = be_u32(bytes, o)? as usize;
    let uncompressed_len = be_u32(bytes, o + 4)? as usize;
    let stored_len = be_u32(bytes, o + 8)? as usize;
    o += 12;
    let stored = bytes.get(o..o + stored_len)?;
    let payload = if flags & 0x01 != 0 {
        use std::io::Read;
        let mut out = Vec::with_capacity(uncompressed_len);
        flate2::read::ZlibDecoder::new(stored)
            .take(uncompressed_len as u64 + 1)
            .read_to_end(&mut out)
            .ok()?;
        out
    } else {
        stored.to_vec()
    };
    if payload.len() != uncompressed_len {
        return None;
    }

    let mut frames = Vec::with_capacity(frame_count);
    let mut i = 0usize;
    for _ in 0..frame_count {
        let direction = *payload.get(i)?;
        let state = *payload.get(i + 1)?;
        let len = be_u32(&payload, i + 10)? as usize;
        let frame = payload.get(i + 14..i + 14 + len)?.to_vec();
        i += 14 + len;
        frames.push((direction, state, frame));
    }
    Some(frames)
}

/// Pulls every `S21PacketChunkData` (0x21) and `S26PacketMapChunkBulk` (0x26) column out of a
/// capture, in arrival order. Field order per protocol 47's packet classes.
fn columns_of(frames: &[(u8, u8, Vec<u8>)]) -> Vec<RawColumn> {
    let mut out = Vec::new();
    // Dimension sky flag (H6). Bulk packets carry it; a lone 0x21 does not, so it inherits the
    // last bulk packet's, defaulting to the overworld exactly as `World` does.
    let mut has_sky_light = true;
    for (direction, state, frame) in frames {
        // clientbound (0), Play (3)
        if *direction != 0 || *state != 3 {
            continue;
        }
        let mut i = 0usize;
        let Some(id) = varint(frame, &mut i) else {
            continue;
        };
        match id {
            0x21 => {
                let (Some(cx), Some(cz), Some(&gu), Some(mask)) = (
                    be_i32(frame, i),
                    be_i32(frame, i + 4),
                    frame.get(i + 8),
                    be_u16(frame, i + 9),
                ) else {
                    continue;
                };
                let mut j = i + 11;
                let Some(size) = varint(frame, &mut j) else {
                    continue;
                };
                let Some(data) = frame.get(j..j + size as usize) else {
                    continue;
                };
                out.push(RawColumn {
                    pos: ChunkPos::new(cx, cz),
                    ground_up: gu != 0,
                    bitmask: mask,
                    has_sky_light,
                    data: data.to_vec(),
                });
            }
            0x26 => {
                let Some(&sky) = frame.get(i) else { continue };
                has_sky_light = sky != 0;
                let mut j = i + 1;
                let Some(count) = varint(frame, &mut j) else {
                    continue;
                };
                let mut metas = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    let (Some(cx), Some(cz), Some(mask)) =
                        (be_i32(frame, j), be_i32(frame, j + 4), be_u16(frame, j + 8))
                    else {
                        break;
                    };
                    j += 10;
                    metas.push((cx, cz, mask));
                }
                if metas.len() != count as usize {
                    continue;
                }
                for (cx, cz, mask) in metas {
                    let len = expected_chunk_data_len(mask, has_sky_light, true);
                    let Some(data) = frame.get(j..j + len) else {
                        break;
                    };
                    j += len;
                    out.push(RawColumn {
                        pos: ChunkPos::new(cx, cz),
                        ground_up: true,
                        bitmask: mask,
                        has_sky_light,
                        data: data.to_vec(),
                    });
                }
            }
            _ => {}
        }
    }
    out
}

/// Every real column payload available, in arrival order, tagged with its capture.
fn load_real_columns() -> Vec<RawColumn> {
    let Some(dir) = capture_dir() else {
        println!("  skip: no .mccap fixtures found (set MC_CAPTURE_DIR)");
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        println!("  skip: cannot read {}", dir.display());
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "mccap"))
        .collect();
    paths.sort();

    let mut out = Vec::new();
    for path in paths {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Some(frames) = capture_frames(&bytes) else {
            println!("  {}: not a readable MCCAP001, skipped", path.display());
            continue;
        };
        let cols = columns_of(&frames);
        println!(
            "  {}: {} frames, {} chunk-column payloads",
            path.file_name().unwrap_or_default().to_string_lossy(),
            frames.len(),
            cols.len()
        );
        out.extend(cols);
    }
    out
}

/// Replays every captured column through the **production** entry point, tiling the capture's
/// footprint until at least `target_columns` are loaded.
///
/// The lobby only ever sent 49 columns (§1.2: *"the client meshed 138 sections total"*), which is
/// far too small a working set to time a read path against — the whole thing fits in L2 and both
/// storages look free. Tiling repeats the captured footprint side by side so the `HashMap`, the
/// section allocations and the cache pressure are the size a real render distance produces, while
/// every byte of content stays real. Pass `--read-columns 0` to measure the untiled capture.
fn replay_into_world(columns: &[RawColumn], target_columns: usize) -> World {
    let mut world = World::new();
    let mut rejected = 0usize;

    let xs: Vec<i32> = columns.iter().map(|c| c.pos.x).collect();
    let zs: Vec<i32> = columns.iter().map(|c| c.pos.z).collect();
    let (Some(&min_x), Some(&max_x), Some(&min_z), Some(&max_z)) = (
        xs.iter().min(),
        xs.iter().max(),
        zs.iter().min(),
        zs.iter().max(),
    ) else {
        return world;
    };
    let (w, h) = (max_x - min_x + 1, max_z - min_z + 1);

    let mut tiles = 0usize;
    'tiling: for ring in 0..1024i32 {
        for tx in 0..=ring {
            for tz in 0..=ring {
                if tx != ring && tz != ring {
                    continue; // already emitted by an inner ring
                }
                for col in columns {
                    world.set_has_sky_light(col.has_sky_light);
                    let pos = ChunkPos::new(col.pos.x + tx * w, col.pos.z + tz * h);
                    if world
                        .apply_chunk_data(pos, col.ground_up, col.bitmask, &col.data)
                        .is_err()
                    {
                        rejected += 1;
                    }
                }
                tiles += 1;
                if world.loaded_chunk_count() >= target_columns {
                    break 'tiling;
                }
            }
        }
    }
    world.drain_dirty_sections();
    println!(
        "  replayed {} payloads x {tiles} tile(s) through World::apply_chunk_data: \
         {rejected} rejected, {} columns loaded",
        columns.len(),
        world.loaded_chunk_count()
    );
    world
}

// =============================================================================================
// Section census
// =============================================================================================

/// Reads a whole section out through the crate's **public** API, in storage index order.
fn cells_of(section: &ChunkSection) -> [u16; 4096] {
    let mut cells = [0u16; 4096];
    for y in 0..16u8 {
        for z in 0..16u8 {
            for x in 0..16u8 {
                cells[local_index(x as usize, y as usize, z as usize)] = section.block(x, y, z).id;
            }
        }
    }
    cells
}

fn light_of(section: &ChunkSection, sky: bool) -> [u8; 2048] {
    let mut out = [0u8; 2048];
    for y in 0..16u8 {
        for z in 0..16u8 {
            for x in 0..16u8 {
                let v = if sky {
                    section.sky_light(x, y, z)
                } else {
                    section.block_light(x, y, z)
                };
                let idx = local_index(x as usize, y as usize, z as usize);
                if idx & 1 == 0 {
                    out[idx >> 1] = (out[idx >> 1] & 0xF0) | v;
                } else {
                    out[idx >> 1] = (out[idx >> 1] & 0x0F) | (v << 4);
                }
            }
        }
    }
    out
}

/// Aggregated census over a set of sections.
#[derive(Default)]
struct Census {
    label: String,
    sections: usize,
    distinct: Vec<usize>,
    pow2_bytes: usize,
    exact_bytes: usize,
    palette_bytes: usize,
    uniform_block_light: usize,
    uniform_sky_light: usize,
    light_measured: usize,
}

impl Census {
    fn add_cells(&mut self, cells: &[u16; 4096]) {
        let mut d = cells.to_vec();
        d.sort_unstable();
        d.dedup();
        let distinct = d.len();
        self.sections += 1;
        self.distinct.push(distinct);
        self.pow2_bytes += 4096 * pow2_bits(distinct) as usize / 8;
        self.exact_bytes += 4096 * exact_bits(distinct) as usize / 8;
        self.palette_bytes += distinct * 2;
    }

    fn add_light(&mut self, block_light: &[u8; 2048], sky_light: &[u8; 2048]) {
        self.light_measured += 1;
        self.uniform_block_light += usize::from(block_light.iter().all(|&b| b == block_light[0]));
        self.uniform_sky_light += usize::from(sky_light.iter().all(|&b| b == sky_light[0]));
    }

    /// Mean bytes a paletted block array would occupy at power-of-two index width.
    fn pow2_mean(&self) -> f64 {
        if self.sections == 0 {
            return 0.0;
        }
        (self.pow2_bytes + self.palette_bytes) as f64 / self.sections as f64
    }

    fn report(&self) {
        if self.sections == 0 {
            println!("  {}: no sections", self.label);
            return;
        }
        println!("  -- {} — {} sections --", self.label, self.sections);
        Spread::of(self.distinct.clone()).report("distinct blockstates / section");
        let cur = self.sections * 8192;
        let pow2 = self.pow2_bytes + self.palette_bytes;
        let exact = self.exact_bytes + self.palette_bytes;
        println!(
            "  block arrays: current {:>9} B -> pow2-width {:>9} B ({:.1} %), exact-width {:>9} B ({:.1} %)",
            cur,
            pow2,
            100.0 * pow2 as f64 / cur as f64,
            exact,
            100.0 * exact as f64 / cur as f64,
        );
        println!(
            "  per section:  current      8192 B -> pow2-width {:>9.0} B,            exact-width {:>9.0} B",
            pow2 as f64 / self.sections as f64,
            exact as f64 / self.sections as f64,
        );
        if self.light_measured > 0 {
            println!(
                "  uniform light arrays: block {}/{} ({:.1} %), sky {}/{} ({:.1} %)  <- a second, \
                 cheaper dedup target",
                self.uniform_block_light,
                self.light_measured,
                100.0 * self.uniform_block_light as f64 / self.light_measured as f64,
                self.uniform_sky_light,
                self.light_measured,
                100.0 * self.uniform_sky_light as f64 / self.light_measured as f64,
            );
        }
    }
}

// =============================================================================================
// Synthetic dense terrain — the pessimistic end of the entropy range
// =============================================================================================

/// A survival-shaped section: stone with the six 1.8 ore types, stone variants, dirt/gravel
/// pockets, cave air, a surface band and a canopy. **`assumed`, not measured** — there is no
/// survival world in this environment (open question 2). It exists only to bracket
/// the lobby census from the other side.
fn synthetic_terrain_cells(section_y: usize, seed: u64) -> [u16; 4096] {
    // Raw packed ids (`block << 4 | meta`), the packing a chunk section stores.
    const AIR: u16 = 0;
    const STONE: u16 = 1 << 4;
    const GRANITE: u16 = (1 << 4) | 1;
    const DIORITE: u16 = (1 << 4) | 3;
    const ANDESITE: u16 = (1 << 4) | 5;
    const GRASS: u16 = 2 << 4;
    const DIRT: u16 = 3 << 4;
    const COARSE_DIRT: u16 = (3 << 4) | 2;
    const GRAVEL: u16 = 13 << 4;
    const COAL: u16 = 16 << 4;
    const IRON: u16 = 15 << 4;
    const GOLD: u16 = 14 << 4;
    const REDSTONE: u16 = 73 << 4;
    const LAPIS: u16 = 21 << 4;
    const DIAMOND: u16 = 56 << 4;
    const WATER: u16 = 9 << 4;
    const LAVA: u16 = 11 << 4;
    const LOG: u16 = 17 << 4;
    const LEAVES: u16 = 18 << 4;

    let mut cells = [AIR; 4096];
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for y in 0..16usize {
        let world_y = section_y * 16 + y;
        for z in 0..16usize {
            for x in 0..16usize {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let state = if world_y < 60 {
                    match rng % 64 {
                        0 => GRANITE,
                        1 => DIORITE,
                        2 => ANDESITE,
                        3 => GRAVEL,
                        4 => DIRT,
                        5 => COAL,
                        6 => IRON,
                        7 if world_y < 32 => GOLD,
                        8 if world_y < 16 => DIAMOND,
                        9 if world_y < 16 => REDSTONE,
                        10 if world_y < 32 => LAPIS,
                        11 if world_y < 12 => LAVA,
                        12..=15 => AIR, // caves
                        _ => STONE,
                    }
                } else if world_y < 64 {
                    match rng % 32 {
                        0..=2 => DIRT,
                        3 => COARSE_DIRT,
                        4 => GRAVEL,
                        5..=7 => WATER,
                        _ => GRASS,
                    }
                } else {
                    match rng % 128 {
                        0 => LOG,
                        1..=6 => LEAVES,
                        _ => AIR,
                    }
                };
                cells[local_index(x, y, z)] = state;
            }
        }
    }
    cells
}

fn synthetic_census() -> Census {
    let mut census = Census {
        label: "synthetic dense terrain (ASSUMED — no survival world here)".into(),
        ..Census::default()
    };
    for i in 0..96usize {
        census.add_cells(&synthetic_terrain_cells(i % 6, i as u64 + 1));
    }
    census
}

// =============================================================================================
// Report sections
// =============================================================================================

fn static_layout() {
    use std::mem::{align_of, size_of};

    println!("\n=== 1. Static layout (measured — compiler) ===\n");
    let section = size_of::<ChunkSection>();
    let chunk = size_of::<Chunk>();
    let entry = size_of::<(ChunkPos, Chunk)>();

    println!(
        "  ChunkSection  size_of = {section} B, align = {}",
        align_of::<ChunkSection>()
    );
    println!(
        "    blocks       [u16; 4096]  {:>6} B  {:>5.1} %",
        8192,
        100.0 * 8192.0 / section as f64
    );
    println!(
        "    block_light  [u8;  2048]  {:>6} B  {:>5.1} %",
        2048,
        100.0 * 2048.0 / section as f64
    );
    println!(
        "    sky_light    [u8;  2048]  {:>6} B  {:>5.1} %",
        2048,
        100.0 * 2048.0 / section as f64
    );
    println!(
        "  Chunk         size_of = {chunk} B, align = {}",
        align_of::<Chunk>()
    );
    println!(
        "    sections     [Option<Box<ChunkSection>>; 16]  {:>6} B  (niche-packed, 8 B/slot)",
        16 * size_of::<Option<Box<ChunkSection>>>()
    );
    println!(
        "    biomes       [u8; 256]                        {:>6} B",
        256
    );
    println!(
        "    light bookkeeping (A12-01)                    {:>6} B           (2 x [i32; 256] heightmaps, [bool; 256] gap flags, 3 scalars)",
        2 * 1024 + 256 + 12
    );
    println!("  World map entry (ChunkPos, Chunk) = {entry} B");
    println!(
        "  size_of::<World>() = {} B — the fixed part: HashMap + hecs::World + HashSet + flag",
        size_of::<World>()
    );
    println!(
        "\n  A section is {:.0} KiB and **content-independent**: a section holding one distinct\n  \
         state costs exactly what a section holding 4096 distinct states costs. That is the\n  \
         entire premise a palette attacks — and it is also why nothing here has to be sampled.",
        section as f64 / KIB
    );

    // Pin the arithmetic: if the layout changes, this bench's whole table has to be redone.
    //
    // The `Chunk` figure was `16 * 8 + 256` when this bench was written and had gone stale:
    // A12-01 gave `Chunk` vanilla's `heightMap` and `precipitationHeightMap` (`[i32; 256]`
    // each, `Chunk.java:56/66`), `updateSkylightColumns` (`[bool; 256]`, `:88`) and the three
    // scalars `heightMapMinimum` / `isGapLightingUpdated` / `queuedLightChecks`. That is
    // 2297 bytes of light bookkeeping on top of the original 384, padded to the struct's
    // 8-byte alignment. The pin is restored rather than deleted: it is the only thing that
    // makes the percentages in the table above trustworthy.
    assert_eq!(section, 8192 + 2048 + 2048, "ChunkSection layout changed");
    assert_eq!(
        chunk,
        (16 * 8 + 256 + 2 * 1024 + 256 + 4 + 1 + 4usize).next_multiple_of(8),
        "Chunk layout changed"
    );
}

/// Builds real `World`s at F3's column counts and reports what they weigh.
fn loaded_set_table(max_columns: usize) {
    println!("\n=== 2. Loaded-set cost (measured — worlds actually built and filled) ===\n");
    println!(
        "  Sections/column follows F3's `~6/column` assumption so the two tables line up.\n  \
         Every column below is a real `World::apply_chunk_data` call with a well-formed payload."
    );
    println!(
        "\n  {:<4} {:>8} {:>10} {:>14} {:>13} {:>13} {:>11}",
        "rd", "columns", "sections", "sections MiB", "columns MiB", "total MiB", "build ms"
    );

    for (rd, columns) in F3_COLUMNS {
        if columns > max_columns {
            println!("  {rd:<4} {columns:>8}   skipped (above --max-columns)");
            continue;
        }
        let bitmask: u16 = (1u16 << F3_SECTIONS_PER_COLUMN) - 1;
        let payload = vec![0u8; expected_chunk_data_len(bitmask, true, true)];

        let started = Instant::now();
        let mut world = World::new();
        let side = (columns as f64).sqrt().ceil() as i32;
        let mut placed = 0usize;
        'outer: for cx in 0..side {
            for cz in 0..side {
                if placed == columns {
                    break 'outer;
                }
                world
                    .apply_chunk_data(ChunkPos::new(cx, cz), true, bitmask, &payload)
                    .expect("well-formed payload");
                placed += 1;
            }
        }
        let build = started.elapsed();
        assert_eq!(world.loaded_chunk_count(), columns);

        let sections: usize = world
            .loaded_chunks()
            .map(|(_, c)| c.sections().iter().flatten().count())
            .sum();
        let section_bytes = sections * std::mem::size_of::<ChunkSection>();
        let column_bytes = columns * std::mem::size_of::<Chunk>();
        println!(
            "  {rd:<4} {columns:>8} {sections:>10} {:>14.2} {:>13.3} {:>13.2} {:>11.1}",
            section_bytes as f64 / MIB,
            column_bytes as f64 / MIB,
            (section_bytes + column_bytes) as f64 / MIB,
            build.as_secs_f64() * 1000.0,
        );
    }

    println!(
        "\n  Projected (arithmetic only) — the same worlds at 16 sections/column, which is\n  \
              what `mc-render`'s `fixture::blank_world` and therefore `frame_loop` actually build:"
    );
    for (rd, columns) in F3_COLUMNS {
        let sections = columns * 16;
        let bytes =
            sections * std::mem::size_of::<ChunkSection>() + columns * std::mem::size_of::<Chunk>();
        println!(
            "    rd {rd:<3} {columns:>6} x 16 = {sections:>7} sections x 12288 B + {columns:>6} x 384 B = {:>8.2} MiB",
            bytes as f64 / MIB
        );
    }

    let entry = std::mem::size_of::<(ChunkPos, Chunk)>() + 1;
    println!(
        "\n  Not counted above: the `HashMap<ChunkPos, Chunk>` table itself. hashbrown keeps\n  \
         `next_pow2(ceil(n * 8/7))` buckets of {} B + 1 control byte — at 4225 columns that is\n  \
         8192 x {} B = {:.2} MiB (projected: {:.2} % of the rd-32 total).",
        std::mem::size_of::<(ChunkPos, Chunk)>(),
        entry,
        8192.0 * entry as f64 / MIB,
        100.0 * 8192.0 * entry as f64 / (4225.0 * 6.0 * 12288.0 + 4225.0 * 384.0),
    );
}

/// Censuses every section a real `World` holds after replaying the captures.
fn real_census(world: &World) -> Census {
    let mut census = Census {
        label: "real 1.8.8 server capture (MEASURED — lobby, not terrain)".into(),
        ..Census::default()
    };
    let mut per_column = Vec::new();
    for (_, chunk) in world.loaded_chunks() {
        let mut n = 0usize;
        for section in chunk.sections().iter().flatten() {
            census.add_cells(&cells_of(section));
            census.add_light(&light_of(section, false), &light_of(section, true));
            n += 1;
        }
        per_column.push(n);
    }
    Spread::of(per_column).report("loaded sections / column");
    census
}

/// The same sections in both storages, read with the mesher's pattern, interleaved, plus A/A.
fn read_path_ab(world: &World, reps: usize) -> Option<()> {
    println!("\n=== 4. Read path: current vs paletted (measured, interleaved, same process) ===\n");
    let mut paletted = PalettedWorld {
        chunks: HashMap::new(),
    };
    let mut positions: Vec<(ChunkPos, i32)> = Vec::new();
    let mut direct_bytes = 0usize;

    for (pos, chunk) in world.loaded_chunks() {
        let mut pchunk = PalettedChunk::empty();
        pchunk.biomes = *chunk.biomes();
        for (i, slot) in chunk.sections().iter().enumerate() {
            if let Some(section) = slot.as_deref() {
                pchunk.sections[i] = Some(Box::new(PalettedSection::build(&cells_of(section))));
                pchunk.block_light[i] = Some(Box::new(light_of(section, false)));
                pchunk.sky_light[i] = Some(Box::new(light_of(section, true)));
                direct_bytes += std::mem::size_of::<ChunkSection>();
                positions.push((pos, i as i32));
            }
        }
        direct_bytes += std::mem::size_of::<Chunk>();
        paletted.chunks.insert(pos, pchunk);
    }
    if positions.is_empty() {
        println!("  skip: no real sections to read");
        return None;
    }
    positions.sort_by_key(|&(p, y)| (p.x, p.z, y));

    let paletted_bytes = paletted.bytes();
    let paletted_index_bytes: usize = paletted
        .chunks
        .values()
        .flat_map(|c| c.sections.iter().flatten())
        .map(|s| s.index_bytes())
        .sum();
    println!(
        "  {} sections, both storages live in this process at once:\n    \
         current   {:>10} B  ({:.3} MiB)\n    \
         paletted  {:>10} B  ({:.3} MiB, {:.1} % of current — light arrays carried unchanged)\n    \
         block arrays alone: {} B -> {} B of packed indices ({:.1} %)",
        positions.len(),
        direct_bytes,
        direct_bytes as f64 / MIB,
        paletted_bytes,
        paletted_bytes as f64 / MIB,
        100.0 * paletted_bytes as f64 / direct_bytes as f64,
        positions.len() * 8192,
        paletted_index_bytes,
        100.0 * paletted_index_bytes as f64 / (positions.len() * 8192) as f64,
    );

    // Correctness gate before any timing: every position of every section, not sampled.
    let mut checked = 0usize;
    for &(pos, sy) in &positions {
        for y in 0..16i32 {
            for z in 0..16i32 {
                for x in 0..16i32 {
                    let bp = BlockPos::new(pos.x * 16 + x, sy * 16 + y, pos.z * 16 + z);
                    assert_eq!(
                        world.block_at(bp).id,
                        paletted.block_at(bp),
                        "paletted read diverged at {bp:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    println!("  verified: {checked} positions read identically from both storages (exhaustive)\n");

    // `mesh_walk`'s exact pattern: y, then z, then x, one `block_at` per cell.
    let sweep_direct = |w: &World| -> u64 {
        let mut acc = 0u64;
        for &(pos, sy) in &positions {
            for y in 0..16i32 {
                for z in 0..16i32 {
                    for x in 0..16i32 {
                        let bp = BlockPos::new(pos.x * 16 + x, sy * 16 + y, pos.z * 16 + z);
                        acc = acc.wrapping_add(u64::from(w.block_at(bp).id));
                    }
                }
            }
        }
        acc
    };
    let sweep_paletted = |w: &PalettedWorld| -> u64 {
        let mut acc = 0u64;
        for &(pos, sy) in &positions {
            for y in 0..16i32 {
                for z in 0..16i32 {
                    for x in 0..16i32 {
                        let bp = BlockPos::new(pos.x * 16 + x, sy * 16 + y, pos.z * 16 + z);
                        acc = acc.wrapping_add(u64::from(w.block_at(bp)));
                    }
                }
            }
        }
        acc
    };

    let mut a = Samples::empty();
    let mut b = Samples::empty();
    let mut control = Samples::empty();
    let mut sink = 0u64;

    sink = sink.wrapping_add(sweep_direct(world)); // warm-up, discarded
    sink = sink.wrapping_add(sweep_paletted(&paletted));

    for _ in 0..reps {
        let t = Instant::now();
        let v = sweep_direct(world);
        a.push(t.elapsed());
        sink = sink.wrapping_add(v);

        let t = Instant::now();
        let v = sweep_paletted(&paletted);
        b.push(t.elapsed());
        sink = sink.wrapping_add(v);

        let t = Instant::now();
        let v = sweep_direct(world);
        control.push(t.elapsed());
        sink = sink.wrapping_add(v);
    }

    let cells = positions.len() as f64 * 4096.0;
    a.report("A  current, World::block_at");
    control.report("A' current again — the control");
    b.report("B  paletted, same pattern");
    println!(
        "\n  per cell:  A {:.3} ns   A' {:.3} ns   B {:.3} ns",
        a.mean() * 1e6 / cells,
        control.mean() * 1e6 / cells,
        b.mean() * 1e6 / cells,
    );
    println!(
        "  A' vs A: {:+.2} %   <- what \"the change did nothing\" measures like here\n  \
         B  vs A: {:+.2} %",
        100.0 * (control.mean() - a.mean()) / a.mean(),
        100.0 * (b.mean() - a.mean()) / a.mean(),
    );
    let delta_per_section = (b.mean() - a.mean()) / positions.len() as f64;
    println!(
        "  per-section sweep: A {:.4} ms, B {:.4} ms, delta {:+.4} ms/section",
        a.mean() / positions.len() as f64,
        b.mean() / positions.len() as f64,
        delta_per_section,
    );
    println!(
        "  against §1.6's `mesh_section` serial {F3_MESH_MS_PER_SECTION} ms/section: {:+.2} % of\n  \
         one section's mesh time — and that is a LOWER bound, because `mesh_walk` also reads\n  \
         neighbours for AO, cull-face, light and fluid height, so real meshing does several\n  \
         `block_at` calls per cell, not one.",
        100.0 * delta_per_section / F3_MESH_MS_PER_SECTION,
    );
    println!(
        "  extrapolated to §1.6's 2500-section join burst: {:+.1} ms added to a 287 ms mesh_all",
        delta_per_section * 2500.0,
    );
    println!(
        "  aside, NOT a sizing input: F5 measures `mesh_section` at {F5_CALLS_PER_SECOND:.0}\n  \
         calls/s (786 986 in 390 s, 8 x frames rendered, ~450 re-meshes per section of a static\n  \
         world), so at today's call rate this delta recurs at {:+.1} ms of CPU per second. F5 is\n  \
         a defect due to be fixed; the per-call number above is the one that should decide T4.1.",
        delta_per_section * F5_CALLS_PER_SECOND,
    );

    // Isolation: the same sweep with the chunk lookup removed, so the palette's own cost shows
    // undiluted by the `HashMap` probe both storages pay equally.
    let direct_sections: Vec<&ChunkSection> = positions
        .iter()
        .filter_map(|&(pos, sy)| world.chunk(pos)?.section(sy as u8))
        .collect();
    let pal_sections: Vec<&PalettedSection> = positions
        .iter()
        .filter_map(|&(pos, sy)| paletted.chunks.get(&pos)?.sections[sy as usize].as_deref())
        .collect();

    let sweep_sec_direct = || {
        let mut acc = 0u64;
        for s in &direct_sections {
            for y in 0..16u8 {
                for z in 0..16u8 {
                    for x in 0..16u8 {
                        acc = acc.wrapping_add(u64::from(s.block(x, y, z).id));
                    }
                }
            }
        }
        acc
    };
    // `local_index` is `y << 8 | z << 4 | x`, so 0..4096 is the same visit order as the loop above.
    let sweep_sec_pal = || {
        let mut acc = 0u64;
        for s in &pal_sections {
            for i in 0..4096usize {
                acc = acc.wrapping_add(u64::from(s.get(i)));
            }
        }
        acc
    };

    let mut sa = Samples::empty();
    let mut sb = Samples::empty();
    let mut sc = Samples::empty();
    sink = sink.wrapping_add(sweep_sec_direct());
    sink = sink.wrapping_add(sweep_sec_pal());
    for _ in 0..reps {
        let t = Instant::now();
        let v = sweep_sec_direct();
        sa.push(t.elapsed());
        sink = sink.wrapping_add(v);

        let t = Instant::now();
        let v = sweep_sec_pal();
        sb.push(t.elapsed());
        sink = sink.wrapping_add(v);

        let t = Instant::now();
        let v = sweep_sec_direct();
        sc.push(t.elapsed());
        sink = sink.wrapping_add(v);
    }
    println!("\n  Isolated — the chunk looked up ONCE per section instead of once per cell:");
    sa.report("C  current, ChunkSection::block");
    sc.report("C' current again — the control");
    sb.report("D  paletted, shift + mask + index");
    println!(
        "  per cell:  C {:.3} ns   C' {:.3} ns   D {:.3} ns    C' vs C {:+.2} %,  D vs C {:+.2} %",
        sa.mean() * 1e6 / cells,
        sc.mean() * 1e6 / cells,
        sb.mean() * 1e6 / cells,
        100.0 * (sc.mean() - sa.mean()) / sa.mean(),
        100.0 * (sb.mean() - sa.mean()) / sa.mean(),
    );
    println!(
        "  CAVEAT, stated because it inflates D/C: C's inner loop is a contiguous scan the\n  \
         compiler auto-vectorises ({:.2} cycles/cell at 4.7 GHz), and D's palette indirection\n  \
         cannot be. `mesh_walk` branches per cell and calls into the model baker, so it gets no\n  \
         such vectorisation either — the honest palette figure is the A/B pair above, not this\n  \
         one. What this pair *is* good for is the next number.",
        sa.mean() * 1e-3 * 4.7e9 / cells,
    );

    // What *is* a `World::block_at` made of? Four arms, same data, same generic sweep body,
    // interleaved, each changing exactly one thing from the one before:
    //
    //   E1 : `HashMap<ChunkPos, Chunk>` — a clone of `World`'s own layout. Its job is to
    //        reproduce A; if it does not, the difference is `World`, not the map.
    //   E2 : `HashMap<ChunkPos, Box<Chunk>>` — the same map with 16 B entries instead of 392 B.
    //   E3 : E2 with `SipHash-1-3` swapped for the multiply-rotate integer hash.
    //   E4 : E3 with paletted storage. `&PalettedChunk` is 8 B, the same entry size as `Box`,
    //        so E4 differs from E3 only in how a section stores its blocks.
    let by_value: HashMap<ChunkPos, Chunk> =
        world.loaded_chunks().map(|(p, c)| (p, c.clone())).collect();
    let boxed: HashMap<ChunkPos, Box<Chunk>> = world
        .loaded_chunks()
        .map(|(p, c)| (p, Box::new(c.clone())))
        .collect();
    let boxed_fx: HashMap<ChunkPos, &Chunk, FxBuild> =
        boxed.iter().map(|(p, c)| (*p, c.as_ref())).collect();
    let pal_fx: HashMap<ChunkPos, &PalettedChunk, FxBuild> =
        paletted.chunks.iter().map(|(p, c)| (*p, c)).collect();

    let sweep_pal_fx = || {
        let mut acc = 0u64;
        for &(pos, sy) in &positions {
            for y in 0..16i32 {
                for z in 0..16i32 {
                    for x in 0..16i32 {
                        let bp = BlockPos::new(pos.x * 16 + x, sy * 16 + y, pos.z * 16 + z);
                        let id = if (0..256).contains(&bp.y) {
                            match pal_fx.get(&ChunkPos::new(bp.x >> 4, bp.z >> 4)) {
                                Some(c) => match &c.sections[(bp.y >> 4) as usize] {
                                    Some(s) => s.get(local_index(
                                        (bp.x & 15) as usize,
                                        (bp.y & 15) as usize,
                                        (bp.z & 15) as usize,
                                    )),
                                    None => 0,
                                },
                                None => 0,
                            }
                        } else {
                            0
                        };
                        acc = acc.wrapping_add(u64::from(id));
                    }
                }
            }
        }
        acc
    };

    let mut e1 = Samples::empty();
    let mut e2 = Samples::empty();
    let mut e3 = Samples::empty();
    let mut e4 = Samples::empty();
    sink = sink.wrapping_add(sweep_chunk_map(&by_value, &positions));
    sink = sink.wrapping_add(sweep_pal_fx());
    for _ in 0..reps {
        let t = Instant::now();
        let v = sweep_chunk_map(&by_value, &positions);
        e1.push(t.elapsed());
        sink = sink.wrapping_add(v);

        let t = Instant::now();
        let v = sweep_chunk_map(&boxed, &positions);
        e2.push(t.elapsed());
        sink = sink.wrapping_add(v);

        let t = Instant::now();
        let v = sweep_chunk_map(&boxed_fx, &positions);
        e3.push(t.elapsed());
        sink = sink.wrapping_add(v);

        let t = Instant::now();
        let v = sweep_pal_fx();
        e4.push(t.elapsed());
        sink = sink.wrapping_add(v);
    }
    println!("\n  Decomposing the per-cell chunk probe (same sweep body, same data, interleaved):");
    e1.report("E1 HashMap<_, Chunk>, std hash");
    e2.report("E2 HashMap<_, Box<Chunk>>, std hash");
    e3.report("E3 = E2 + fast integer hash");
    e4.report("E4 = E3 + paletted sections");
    println!(
        "  per cell:  A {:.3} ns | E1 {:.3} | E2 {:.3} | E3 {:.3} | E4 {:.3}",
        a.mean() * 1e6 / cells,
        e1.mean() * 1e6 / cells,
        e2.mean() * 1e6 / cells,
        e3.mean() * 1e6 / cells,
        e4.mean() * 1e6 / cells,
    );
    println!(
        "  E1 vs A  {:+.2} %  <- the sweep reproducing `World::block_at`; near 0 means `World`\n  \
         {:>17}    itself adds nothing and E2..E4 are measuring the real thing\n  \
         E2 vs E1 {:+.2} %  <- `Chunk` by value (392 B entries) vs boxed (16 B)\n  \
         E3 vs E2 {:+.2} %  <- `SipHash-1-3` vs a multiply-rotate integer hash\n  \
         E4 vs E3 {:+.2} %  <- **the palette's read cost, with nothing left to hide behind**",
        100.0 * (e1.mean() - a.mean()) / a.mean(),
        "",
        100.0 * (e2.mean() - e1.mean()) / e1.mean(),
        100.0 * (e3.mean() - e2.mean()) / e2.mean(),
        100.0 * (e4.mean() - e3.mean()) / e3.mean(),
    );

    println!("\n  (checksum {sink:#x} — printed so no sweep can be optimised away)");
    Some(())
}

/// The decode path (`fill_from_packet`) and the block-change path, both A/B.
fn build_and_write_ab(columns: &[RawColumn], reps: usize) {
    println!("\n=== 5. Decode and write paths (measured, interleaved) ===\n");
    // One well-formed ground-up payload per distinct column, deduplicated to the last send.
    let mut by_pos: HashMap<ChunkPos, &RawColumn> = HashMap::new();
    for col in columns {
        if col.ground_up && col.bitmask != 0 {
            by_pos.insert(col.pos, col);
        }
    }
    let mut payloads: Vec<&RawColumn> = by_pos.into_values().collect();
    payloads.sort_by_key(|c| (c.pos.x, c.pos.z));
    if payloads.is_empty() {
        println!("  skip: no real column payloads");
        return;
    }
    let sections: usize = payloads
        .iter()
        .map(|c| c.bitmask.count_ones() as usize)
        .sum();
    println!(
        "  {} real column payloads, {sections} sections total",
        payloads.len()
    );

    let mut a = Samples::empty();
    let mut b = Samples::empty();
    let mut control = Samples::empty();
    let mut sink = 0u64;

    // Each decoded set is dropped *before* the next timing starts. Leaving them all alive to the
    // end of the iteration was tried first and made the A/A control read +84 %, purely from
    // allocator state — a good demonstration of why the control is here at all.
    for _ in 0..reps {
        let t = Instant::now();
        let built = decode_current(&payloads);
        a.push(t.elapsed());
        sink = sink.wrapping_add(built.len() as u64);
        drop(built);

        let t = Instant::now();
        let built = decode_paletted(&payloads);
        b.push(t.elapsed());
        sink = sink.wrapping_add(built.len() as u64);
        drop(built);

        let t = Instant::now();
        let built = decode_current(&payloads);
        control.push(t.elapsed());
        sink = sink.wrapping_add(built.len() as u64);
        drop(built);
    }
    a.report("A  Chunk::fill_from_packet");
    control.report("A' fill_from_packet again — control");
    b.report("B  decode + palette build");
    println!(
        "\n  per section: A {:.4} ms, B {:.4} ms, delta {:+.4} ms\n  \
         A' vs A {:+.2} % (control) — B vs A {:+.2} %\n  \
         Every section the client receives pays this once, on the join burst §1.5 measures at\n  \
         1335 fps and a 3.16 ms worst `drain_mesh_queue`.",
        a.mean() / sections as f64,
        b.mean() / sections as f64,
        (b.mean() - a.mean()) / sections as f64,
        100.0 * (control.mean() - a.mean()) / a.mean(),
        100.0 * (b.mean() - a.mean()) / a.mean(),
    );

    // Block-change writes: in-palette (cheap) and novel-state (a full repack).
    let sample = payloads[0];
    let mut chunk = Chunk::empty();
    if chunk
        .fill_from_packet(
            sample.ground_up,
            sample.bitmask,
            sample.has_sky_light,
            &sample.data,
        )
        .is_err()
    {
        println!("  skip: sample payload rejected");
        return;
    }
    let Some(idx) = (0..16u8).find(|&i| chunk.section(i).is_some()) else {
        println!("  skip: sample column has no sections");
        return;
    };
    let cells = cells_of(chunk.section(idx).expect("checked"));

    let mut w_direct = Samples::empty();
    let mut w_hit = Samples::empty();
    let mut w_miss = Samples::empty();
    let known = cells[0];

    const WRITES: usize = 16_384;
    const NOVEL: usize = 64;
    for _ in 0..reps {
        let mut section = chunk.clone();
        let mut p = PalettedSection::build(&cells);

        let t = Instant::now();
        let s = section.section_mut(idx).expect("present");
        for i in 0..WRITES {
            s.set_block(
                (i & 15) as u8,
                ((i >> 4) & 15) as u8,
                ((i >> 8) & 15) as u8,
                BlockState::from_raw(known),
            );
        }
        w_direct.push(t.elapsed());
        sink = sink.wrapping_add(u64::from(s.block(0, 0, 0).id));

        let t = Instant::now();
        for i in 0..WRITES {
            p.set(i & 4095, known);
        }
        w_hit.push(t.elapsed());

        // States the palette has never held. Each one either appends (free while the index
        // width still holds it) or forces a repack of all 4096 indices.
        let mut q = PalettedSection::build(&cells);
        let t = Instant::now();
        for n in 0..NOVEL {
            q.set(n, 0xF00u16.wrapping_add(n as u16) | 1);
        }
        w_miss.push(t.elapsed());
        sink = sink.wrapping_add(u64::from(q.get(7)));
    }
    println!();
    w_direct.report(&format!("write: current, {WRITES} set_block"));
    w_hit.report(&format!("write: paletted, {WRITES} in-palette"));
    w_miss.report(&format!("write: paletted, {NOVEL} novel states"));
    println!(
        "  per write: current {:.2} ns, paletted in-palette {:.2} ns, paletted novel {:.0} ns",
        w_direct.mean() * 1e6 / WRITES as f64,
        w_hit.mean() * 1e6 / WRITES as f64,
        w_miss.mean() * 1e6 / NOVEL as f64,
    );
    println!("  (checksum {sink:#x})");
}

fn decode_current(payloads: &[&RawColumn]) -> Vec<Chunk> {
    let mut out = Vec::with_capacity(payloads.len());
    for col in payloads {
        let mut chunk = Chunk::empty();
        if chunk
            .fill_from_packet(col.ground_up, col.bitmask, col.has_sky_light, &col.data)
            .is_ok()
        {
            out.push(chunk);
        }
    }
    out
}

fn decode_paletted(payloads: &[&RawColumn]) -> Vec<PalettedChunk> {
    let mut out = Vec::with_capacity(payloads.len());
    for col in payloads {
        let mut chunk = PalettedChunk::empty();
        let selected: Vec<usize> = (0..16usize)
            .filter(|s| col.bitmask & (1 << s) != 0)
            .collect();
        let mut i = 0usize;
        for &s in &selected {
            let mut cells = [0u16; 4096];
            for cell in cells.iter_mut() {
                *cell = u16::from(col.data[i]) | (u16::from(col.data[i + 1]) << 8);
                i += 2;
            }
            chunk.sections[s] = Some(Box::new(PalettedSection::build(&cells)));
        }
        for &s in &selected {
            let mut light = [0u8; 2048];
            light.copy_from_slice(&col.data[i..i + 2048]);
            chunk.block_light[s] = Some(Box::new(light));
            i += 2048;
        }
        if col.has_sky_light {
            for &s in &selected {
                let mut light = [0u8; 2048];
                light.copy_from_slice(&col.data[i..i + 2048]);
                chunk.sky_light[s] = Some(Box::new(light));
                i += 2048;
            }
        }
        if col.ground_up {
            chunk.biomes.copy_from_slice(&col.data[i..i + 256]);
        }
        out.push(chunk);
    }
    out
}

// =============================================================================================
// Verdict arithmetic
// =============================================================================================

fn verdict(real: &Census, synth: &Census) {
    println!("\n=== 6. The ratio that decides T4.1 ===\n");
    let per_section_now = std::mem::size_of::<ChunkSection>() as f64;
    let vertex_per_section = F3_VERTEX_BYTES_TOTAL / F3_SECTIONS;

    println!(
        "  F3   (measured): {:.2} MiB of GPU vertex bytes for {:.0} sections = {:.2} KiB/section.",
        F3_VERTEX_BYTES_TOTAL / MIB,
        F3_SECTIONS,
        vertex_per_section / KIB
    );
    println!(
        "  Here (measured): {:.2} KiB of CPU chunk storage per section.",
        per_section_now / KIB
    );
    println!(
        "  ==> CPU chunk storage is {:.1} % of the GPU vertex bytes for the same section.\n",
        100.0 * per_section_now / vertex_per_section
    );

    for census in [real, synth] {
        if census.sections == 0 {
            continue;
        }
        let saved = 8192.0 - census.pow2_mean();
        println!("  -- if T4.1 shipped, sized from: {} --", census.label);
        println!(
            "    block array 8192 B -> {:.0} B;  section {:.0} B -> {:.0} B;  saving {:.2} KiB/section",
            census.pow2_mean(),
            per_section_now,
            per_section_now - saved,
            saved / KIB
        );
        println!(
            "    = {:.1} % of the section's CPU storage, and {:.2} % of its GPU vertex bytes",
            100.0 * saved / per_section_now,
            100.0 * saved / vertex_per_section
        );
        println!(
            "    {:<4} {:>8} {:>9} {:>12} {:>13} {:>11} {:>15}",
            "rd",
            "columns",
            "sections",
            "CPU now MiB",
            "CPU T4.1 MiB",
            "saved MiB",
            "F3 vertex MiB"
        );
        for (rd, columns) in F3_COLUMNS {
            let sections = columns * F3_SECTIONS_PER_COLUMN;
            let now = sections as f64 * per_section_now + columns as f64 * 384.0;
            let after = sections as f64 * (per_section_now - saved) + columns as f64 * 384.0;
            let vertex = sections as f64 * vertex_per_section;
            println!(
                "    {rd:<4} {columns:>8} {sections:>9} {:>12.1} {:>13.1} {:>11.1} {:>15.1}",
                now / MIB,
                after / MIB,
                (now - after) / MIB,
                vertex / MIB
            );
        }
        println!();
    }

    println!(
        "  Arithmetic behind every projected row: sections = columns x {} (F3's own assumption);\n  \
         CPU now = sections x {} B + columns x 384 B; CPU after = sections x (12288 - saving) B\n  \
         + columns x 384 B; F3 vertex = sections x {:.0} B (= 162.44 MiB / 2500).",
        F3_SECTIONS_PER_COLUMN,
        std::mem::size_of::<ChunkSection>(),
        vertex_per_section,
    );
}

// =============================================================================================

struct Args {
    reps: usize,
    max_columns: usize,
    read_columns: usize,
    hold_ms: u64,
}

fn parse_args() -> Args {
    let mut args = Args {
        reps: 20,
        max_columns: 4225,
        read_columns: 1089,
        hold_ms: 0,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    while i < raw.len() {
        let value: Option<u64> = raw.get(i + 1).and_then(|v| v.parse().ok());
        match raw[i].as_str() {
            "--reps" => args.reps = value.unwrap_or(args.reps as u64) as usize,
            "--max-columns" => args.max_columns = value.unwrap_or(args.max_columns as u64) as usize,
            "--read-columns" => {
                args.read_columns = value.unwrap_or(args.read_columns as u64) as usize;
            }
            "--hold-ms" => args.hold_ms = value.unwrap_or(args.hold_ms),
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    args
}

fn main() {
    let args = parse_args();
    println!(
        "mc-world chunk storage: memory cost\n\
         reps = {}, max columns = {}, read-path columns = {}, build = {}",
        args.reps,
        args.max_columns,
        args.read_columns,
        if cfg!(debug_assertions) {
            "debug_assertions ON — NOT a release measurement"
        } else {
            "optimised"
        }
    );

    static_layout();
    loaded_set_table(args.max_columns);

    println!("\n=== 3. Palette headroom ===\n");
    let columns = load_real_columns();
    let (real, world) = if columns.is_empty() {
        (Census::default(), World::new())
    } else {
        let world = replay_into_world(&columns, args.read_columns);
        (real_census(&world), world)
    };
    real.report();
    println!();
    let synth = synthetic_census();
    synth.report();

    if read_path_ab(&world, args.reps).is_some() {
        build_and_write_ab(&columns, args.reps);
    }
    verdict(&real, &synth);

    if args.hold_ms > 0 {
        println!(
            "\nholding for {} ms (read the process working set from outside)",
            args.hold_ms
        );
        std::thread::sleep(Duration::from_millis(args.hold_ms));
    }
}
