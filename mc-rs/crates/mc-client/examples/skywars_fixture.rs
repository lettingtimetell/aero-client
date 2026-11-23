//! SkyWars benchmark scene generator: `sw-mega-doubles-mid` (scene spec v1, 2026-11-05).
//!
//! Writes a deterministic `.mccap` capture, its recipe sidecar `<stem>.recipe.json` (schema
//! `mc-rs-skywars-scene/1`, consumed by `skywars_bench.rs`) and a statistics report
//! `<stem>.stats.md`. The scene is a mid-game Mega Doubles fight (40 players, 20 teams of 2, 26
//! alive) on a real exported SkyWars map, seen in first person by a player trading hits with a
//! Sharpness sword. Spec: `optimization/scratch/skywars-scene-2026-11-05/scene-spec.md`.
//!
//! ```text
//! cargo run -p mc-client --example skywars_fixture -- --map <maps/out/dwarf_fortress> \
//!     --variant mega-doubles-mid --seed 1 --output <dir>/sw-mega-doubles-mid.mccap \
//!     [--local-knockback on|off] [--antixray]
//! ```
//!
//! # Structure: a small server emulator, not a packet script
//!
//! 1. **World.** The map's `chunks.bin` columns within 8 chunks (Chebyshev) of the viewer, in
//!    Spigot's `PlayerChunkMap.ChunkCoordComparator` order around the viewer, cage glass stripped
//!    (sections that become empty are dropped from the bitmask, as Spigot would), sent as `S26`
//!    bulks of at most 10 columns, one bulk per tick (Spigot `EntityPlayer.t_` sends one bulk per
//!    tick).
//! 2. **Scripted state.** Every player near the fight has per-tick state: position (f64, encoded
//!    to 1/32 block only inside the tracker, from the true position, so rounding never
//!    accumulates), yaw/pitch/head yaw, onGround, sprint/sneak/use flags, health, absorption,
//!    arrows-in-body, equipment, potion effects. A simplified vanilla player physics (ground
//!    friction, air drag, gravity, jumps, knockback, terrain steps from the decoded map) moves
//!    them; small role AIs (duelist, archer, bridger, looter, traveller) drive the inputs.
//! 3. **Tracker.** A port of the 1.8.8 server's
//!    `EntityTrackerEntry.updatePlayerList`/`updatePlayerEntity`
//!    with Spigot's tracking ranges (players 48, items/XP 32, everything else 64) and
//!    CraftBukkit's spawn-time head rotation. It alone emits S0C/S0E/S11/S13/S15-S19/S1C/S20 and
//!    the tracker S12. Melee knockback S12 goes to the victim only (`EntityPlayer`), projectile
//!    and rod knockback through `velocityChanged` to every tracker.
//! 4. **Event emitters.** Attacks and hit responses (swing lockout, hurt-resistance window,
//!    crit/magic-crit echoes, armour/sword durability S04/S2F, hurt and step sounds within 16
//!    blocks), projectiles, bridge block changes, chest actions, deaths/drops/XP, kill effects,
//!    sidebar/tab/chat.
//!
//! The viewer (entity 1) is driven by the bench wrapper from the recipe's `local` block (clicks,
//! keys, aim, approach). The generator models the viewer with the same rules, so the server's
//! reactions to the viewer's damaging hits are placed at tick `T+1` for a press at tick `T`.
//!
//! **Determinism:** no wall clock, no `Recorder::record`, no `CaptureMeta::new`, `BTreeMap`/`Vec`
//! only in anything that reaches the output, and seeded SplitMix64 streams (one per actor).

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mc_data::{block_by_name, item_by_name, item_props, potion, sound_event, BlockState};
use mc_math::BlockPos;
use mc_nbt::{Nbt, NbtCompound};
use mc_parity::{Capture, CaptureMeta, Direction, Recorder};
use mc_protocol::chunk::{self, ChunkBulkColumn, MapChunkBulk};
use mc_protocol::login::LoginSuccess;
use mc_protocol::packet::{Packet, State};
use mc_protocol::play::s2c;
use mc_protocol::{Angle, ItemStack, MetadataEntry, MetadataValue, PacketBuffer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

// =============================================================================================
// Constants
// =============================================================================================

const PROTOCOL: i32 = 47;
const SCHEMA: &str = "mc-rs-skywars-scene/1";
const SCENE_ID: &str = "sw-mega-doubles-mid";
const VARIANT: &str = "mega-doubles-mid";
const K: u64 = 50;
const LOCKSTEP_MICROS: u64 = 1_000;
const TICK_MICROS: u64 = 50_000;
const FIGHT_START: i64 = 300;
const FIGHT_END: i64 = 1_500;
const TAIL_END: i64 = 1_600;
const SEGMENTS: [(&str, i64, i64); 3] = [
    ("sw_join", 0, FIGHT_START),
    ("sw_fight", FIGHT_START, FIGHT_END),
    ("sw_tail", FIGHT_END, TAIL_END),
];
const LOCAL_EID: i32 = 1;
const LOCAL_UUID: u128 = 0x0000_0000_0000_4000_8000_0000_0000_0001;
const LOCAL_NAME: &str = "BenchBot";
const VIEW_DISTANCE: i32 = 8;
const MAX_BULK_COLUMNS: usize = 10;
/// Spigot `entity-tracking-range` defaults.
const RANGE_PLAYER: f64 = 48.0;
const RANGE_MISC: f64 = 32.0;
const RANGE_OTHER: f64 = 64.0;
/// The `approach` keep-within distance written into the recipe and used by the viewer model.
const KEEP_WITHIN: f64 = 2.6;
/// Melee knockback profile. Vanilla is 0.4 horizontal + 0.5 per sprint/Knockback level and
/// 0.4 (+0.1) vertical; minigame servers often use custom, unpublished values. The scene keeps
/// vanilla's base and vertical but halves the sprint bonus so trades stay inside reach, which is
/// what the research's 1.5 damaging hits/s each way implies.
const KB_BASE_H: f64 = 0.3;
const KB_SPRINT_H: f64 = 0.1;
const KB_BASE_V: f64 = 0.34;
const KB_SPRINT_V: f64 = 0.04;
/// Ticks after being hit during which a fighter's own clicks do not connect (the knockback
/// throws the aim off). Without it both sides of every trade hit in the same tick and fly apart.
const HIT_RECOVERY_TICKS: i64 = 3;
/// Plugin particle range for projectile-trail cosmetics (vanilla `WorldServer.spawnParticle`
/// uses 16 blocks; trail plugins send further). The packets keep `long_distance = false`, so
/// the client still culls them beyond 16 blocks.
const TRAIL_RANGE: f64 = 64.0;
/// Mojang's real signed `textures` payload (Notch), identical to `optimization_fixture.rs`. Every
/// tab row carries it: one shared skin, never an invented signature.
const SIGNED_TEXTURES_VALUE: &str = "ewogICJ0aW1lc3RhbXAiIDogMTc4NzY5MTA3NTE3MiwKICAicHJvZmlsZUlkIiA6ICIwNjlhNzlmNDQ0ZTk0NzI2YTViZWZjYTkwZTM4YWFmNSIsCiAgInByb2ZpbGVOYW1lIiA6ICJOb3RjaCIsCiAgInNpZ25hdHVyZVJlcXVpcmVkIiA6IHRydWUsCiAgInRleHR1cmVzIiA6IHsKICAgICJTS0lOIiA6IHsKICAgICAgInVybCIgOiAiaHR0cDovL3RleHR1cmVzLm1pbmVjcmFmdC5uZXQvdGV4dHVyZS8yOTIwMDlhNDkyNWI1OGYwMmM3N2RhZGMzZWNlZjA3ZWE0Yzc0NzJmNjRlMGZkYzMyY2U1NTIyNDg5MzYyNjgwIgogICAgfQogIH0KfQ==";
const SIGNED_TEXTURES_SIGNATURE: &str = "oqgTiH1ZWCjCcU9a5bOJpskVzloygYgbdNZ16Xtn1RgKrRsPLutWL9Uok/rkg4re1O+TbsKL0lKYzQtZH0+OfWdfWIoWE4bBanHuhejGWu69EWVvcxlpEiNhZptzER9KfPGP5Q7SPsT+4MaSj/AbJP+yKWNjriYZZFV00bWSW/aexY6PJSoXMD/R57gn0TucdaZp1nrbfgL9DXf+6R6T8ou2Lz4qsPAoB2CJI2I8F9HPsLJ/dUinrp8gmjfKnvvJUf+a6XeG+BESWqqChuiD+XJjvnZSyM3Fs1zdb+RkqvFNqT4W14ohyh3mxyX3Q2S/8S+Q2axT47nnjvy9WMVpaDay1o8pquzy+zRqPps1tRFTScrp4w4jVdSzm8xL4FhNqm2H/QUoAYczVUyAfBx4Q8w9daelERlNOcQAIEeauBzan/728P/F4Rxc24Wl9iV5Qb+XeE/858QHNplwLIcSv4Wfp6G29++0XVljN7K2Vtbilp9LipVrDG028XMFY6eY/lb1n9C4hiMWF/N4YTXz8Tm+IBIBqKpgeUCzb3ZmB7bvVubPUgWdL3K7piiClRQ8ssIAN67DEOm/sTgyv0eFitWiZopPhBvgizOBPDv2/GSa3WdFRiNAtaxrqOZVO3W4HUn62XNz+leI640BMjAYMvIcpNU2x0R+S8ILf7v0zcM=";
/// `EntityLivingBase.sprintingSpeedBoostModifierUUID`.
const SPRINT_MODIFIER_UUID: u128 = 0x662A_6B8D_DA3E_4C1C_8813_96EA_6097_278D;
/// `Potion.moveSpeed` attribute modifier UUID.
const SPEED_MODIFIER_UUID: u128 = 0x91AE_AA56_376B_4498_935B_2F7F_6807_0635;
const ENCH_PROTECTION: i16 = 0;
const ENCH_FEATHER_FALLING: i16 = 2;
const ENCH_PROJ_PROTECTION: i16 = 4;
const ENCH_SHARPNESS: i16 = 16;
const ENCH_UNBREAKING: i16 = 34;
const ENCH_POWER: i16 = 48;
const EFFECT_SPEED: u8 = 1;
const EFFECT_REGENERATION: u8 = 10;
const EFFECT_ABSORPTION: u8 = 22;
/// Splash Regeneration II and splash Instant Health II potion damage values.
const POTION_SPLASH_REGEN_II: i32 = 16_417;
const PARTICLE_FIREWORKS_SPARK: i32 = 3;
const PARTICLE_NOTE: i32 = 23;
const PARTICLE_HEART: i32 = 34;
const CHEST_BLOCK_ID: i32 = 54;
const HEADER_TEXT: &str = "You are playing on §e§lMC.EXAMPLE.NET";
const FOOTER_TEXT: &str = "§bRanks, Boosters & MORE! §c§lSTORE.EXAMPLE.NET";

/// Server tick of a moment given in seconds from the start of `sw_fight`.
fn ft(seconds: f64) -> i64 {
    FIGHT_START + (seconds * 20.0).round() as i64
}

fn segment_of(tick: i64) -> usize {
    if tick < FIGHT_START {
        0
    } else if tick < FIGHT_END {
        1
    } else {
        2
    }
}

// =============================================================================================
// Small math helpers
// =============================================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct V3 {
    x: f64,
    y: f64,
    z: f64,
}

impl V3 {
    const ZERO: V3 = V3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    const fn new(x: f64, y: f64, z: f64) -> V3 {
        V3 { x, y, z }
    }
    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    fn scale(self, k: f64) -> V3 {
        V3::new(self.x * k, self.y * k, self.z * k)
    }
    fn hlen(self) -> f64 {
        (self.x * self.x + self.z * self.z).sqrt()
    }
    fn len(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    fn hdist(self, o: V3) -> f64 {
        self.sub(o).hlen()
    }
    fn dist(self, o: V3) -> f64 {
        self.sub(o).len()
    }
    fn hnorm(self) -> V3 {
        let l = self.hlen();
        if l < 1e-9 {
            V3::ZERO
        } else {
            V3::new(self.x / l, 0.0, self.z / l)
        }
    }
    fn json(self) -> Value {
        json!([round3(self.x), round3(self.y), round3(self.z)])
    }
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

/// Minecraft yaw (0 = +Z, 90 = -X) from `from` towards `to`.
fn yaw_to(from: V3, to: V3) -> f32 {
    let d = to.sub(from);
    (-d.x).atan2(d.z).to_degrees() as f32
}

/// Minecraft pitch (positive looks down) from an eye position to a point.
fn pitch_to(eye: V3, to: V3) -> f32 {
    let d = to.sub(eye);
    (-(d.y.atan2(d.hlen()))).to_degrees() as f32
}

fn dir_of_yaw(yaw: f32) -> V3 {
    let r = f64::from(yaw).to_radians();
    V3::new(-r.sin(), 0.0, r.cos())
}

fn wrap180(a: f32) -> f32 {
    let mut a = a % 360.0;
    if a >= 180.0 {
        a -= 360.0;
    }
    if a < -180.0 {
        a += 360.0;
    }
    a
}

/// Keeps a yaw continuous (server yaw is unbounded, like the client's), turning at most `step`.
fn turn_toward(current: f32, target: f32, step: f32) -> f32 {
    current + wrap180(target - current).clamp(-step, step)
}

/// `MathHelper.floor_double(v * 32.0)`.
fn fix32(v: f64) -> i32 {
    (v * 32.0).floor() as i32
}

/// `MathHelper.floor_float(deg * 256.0F / 360.0F)`, computed in f32 as Java does.
fn rot256(deg: f32) -> i32 {
    let f = deg * 256.0 / 360.0;
    let i = f as i32;
    if f < i as f32 {
        i - 1
    } else {
        i
    }
}

/// `S12PacketEntityVelocity` encoding: clamp to ±3.9 blocks/tick, then `(int)(v * 8000)`.
fn vel8000(v: f64) -> i16 {
    (v.clamp(-3.9, 3.9) * 8000.0) as i32 as i16
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

// =============================================================================================
// Seeded RNG (SplitMix64): one independent stream per actor
// =============================================================================================

#[derive(Clone, Debug)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64, stream: u64) -> Rng {
        let mut r = Rng(seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03));
        r.next();
        r
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
    fn chance(&mut self, p: f64) -> bool {
        self.f() < p
    }
}

// =============================================================================================
// CLI
// =============================================================================================

#[derive(Clone, Debug)]
struct Args {
    map: PathBuf,
    seed: u64,
    output: PathBuf,
    local_knockback: bool,
    antixray: bool,
}

fn parse_args(raw: Vec<String>) -> Result<Args, String> {
    let mut map = None;
    let mut variant = None;
    let mut seed = 1u64;
    let mut output = None;
    let mut local_knockback = true;
    let mut antixray = false;
    let mut it = raw.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--map" => map = Some(PathBuf::from(it.next().ok_or("--map needs a value")?)),
            "--variant" => variant = Some(it.next().ok_or("--variant needs a value")?),
            "--seed" => {
                let v = it.next().ok_or("--seed needs a value")?;
                seed = v
                    .parse::<u64>()
                    .map_err(|_| format!("--seed must be a non-negative integer, got {v:?}"))?;
                if seed > i64::MAX as u64 {
                    return Err("--seed must fit an i64 (the recipe stores it as one)".into());
                }
            }
            "--output" => output = Some(PathBuf::from(it.next().ok_or("--output needs a value")?)),
            "--local-knockback" => {
                local_knockback = match it.next().as_deref() {
                    Some("on") => true,
                    Some("off") => false,
                    other => {
                        return Err(format!(
                            "--local-knockback must be on or off, got {other:?}"
                        ))
                    }
                }
            }
            "--antixray" => antixray = true,
            "-h" | "--help" => {
                println!(
                    "usage: skywars_fixture --map <maps/out/<map>> --variant {VARIANT} [--seed N] \
                     --output <file.mccap> [--local-knockback on|off] [--antixray]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let variant = variant.ok_or("--variant is required")?;
    if variant != VARIANT {
        return Err(format!(
            "--variant {variant:?} is not built yet; v1 has only {VARIANT}"
        ));
    }
    Ok(Args {
        map: map.ok_or("--map is required")?,
        seed,
        output: output.ok_or("--output is required")?,
        local_knockback,
        antixray,
    })
}

// =============================================================================================
// Map input: manifest + chunks.bin
// =============================================================================================

#[derive(Clone, Debug)]
struct Column {
    x: i32,
    z: i32,
    mask: u16,
    data: Vec<u8>,
}

#[derive(Debug)]
struct MapInput {
    name: String,
    display_name: String,
    chunks_file: &'static str,
    sha256: String,
    columns: Vec<Column>,
    manifest: Value,
}

fn load_map(dir: &Path, antixray: bool) -> Result<MapInput, String> {
    let manifest_path = dir.join("manifest.json");
    let manifest_bytes =
        fs::read(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let chunks_file = if antixray {
        "chunks.antixray.bin"
    } else {
        "chunks.bin"
    };
    let path = dir.join(chunks_file);
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let sha256 = hex(&Sha256::digest(&bytes));
    if let Some(expected) = manifest["files"][chunks_file]["sha256"].as_str() {
        if expected != sha256 {
            return Err(format!(
                "{}: SHA-256 {sha256} does not match the manifest's {expected}",
                path.display()
            ));
        }
    }
    let mut columns = Vec::new();
    let mut seen = BTreeSet::new();
    let mut at = 0usize;
    while at < bytes.len() {
        let h = bytes
            .get(at..at + 14)
            .ok_or_else(|| format!("{}: truncated record header at {at}", path.display()))?;
        let x = i32::from_le_bytes([h[0], h[1], h[2], h[3]]);
        let z = i32::from_le_bytes([h[4], h[5], h[6], h[7]]);
        let mask = u16::from_le_bytes([h[8], h[9]]);
        let len = u32::from_le_bytes([h[10], h[11], h[12], h[13]]) as usize;
        at += 14;
        let data = bytes
            .get(at..at + len)
            .ok_or_else(|| format!("{}: truncated payload for ({x},{z})", path.display()))?
            .to_vec();
        at += len;
        let want = mc_world::expected_chunk_data_len(mask, true, true);
        if len != want {
            return Err(format!("({x},{z}): payload {len} bytes, expected {want}"));
        }
        if !seen.insert((x, z)) {
            return Err(format!("duplicate column ({x},{z})"));
        }
        columns.push(Column { x, z, mask, data });
    }
    if let Some(n) = manifest["column_count"].as_u64() {
        if n as usize != columns.len() {
            return Err(format!(
                "{}: {} records, manifest says {n}",
                path.display(),
                columns.len()
            ));
        }
    }
    Ok(MapInput {
        name: manifest["name"].as_str().unwrap_or("unknown").to_owned(),
        display_name: manifest["display_name"]
            .as_str()
            .unwrap_or("Unknown")
            .to_owned(),
        chunks_file,
        sha256,
        columns,
        manifest,
    })
}

/// `PlayerChunkMap.ChunkCoordComparator` (Spigot 1.8.8, javap in `maps/server-ref`): squared
/// distance to the player's chunk, then a quadrant tie-break. A strict total order.
fn spigot_chunk_order(px: i32, pz: i32, a: (i32, i32), b: (i32, i32)) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    let (ax, az) = (a.0 - px, a.1 - pz);
    let (bx, bz) = (b.0 - px, b.1 - pz);
    let i = (ax - bx) * (ax + bx) + (az - bz) * (az + bz);
    let r = if i != 0 {
        i
    } else if ax < 0 {
        if bx < 0 {
            bz - az
        } else {
            -1
        }
    } else if bx < 0 {
        1
    } else {
        az - bz
    };
    r.cmp(&0)
}

fn section_ordinal(mask: u16, section: i32) -> Option<usize> {
    if !(0..16).contains(&section) || mask & (1 << section) == 0 {
        return None;
    }
    Some((mask & ((1u16 << section) - 1)).count_ones() as usize)
}

/// Rebuild a column payload keeping only the listed sections (ascending).
fn rebuild_payload(col: &Column, keep: u16) -> Vec<u8> {
    let n = col.mask.count_ones() as usize;
    let (blocks, rest) = col.data.split_at(n * 8192);
    let (block_light, rest) = rest.split_at(n * 2048);
    let (sky_light, biomes) = rest.split_at(n * 2048);
    let mut out_blocks = Vec::new();
    let mut out_bl = Vec::new();
    let mut out_sl = Vec::new();
    for s in 0..16 {
        if let Some(ord) = section_ordinal(col.mask, s) {
            if keep & (1 << s) != 0 {
                out_blocks.extend_from_slice(&blocks[ord * 8192..(ord + 1) * 8192]);
                out_bl.extend_from_slice(&block_light[ord * 2048..(ord + 1) * 2048]);
                out_sl.extend_from_slice(&sky_light[ord * 2048..(ord + 1) * 2048]);
            }
        }
    }
    let mut payload = out_blocks;
    payload.extend_from_slice(&out_bl);
    payload.extend_from_slice(&out_sl);
    payload.extend_from_slice(biomes);
    payload
}

#[derive(Debug, Default)]
struct StripReport {
    blocks: usize,
    by_block: BTreeMap<String, usize>,
    sections_dropped: usize,
    cages: usize,
}

/// Every glass / stained glass / pane block in the shell around each cage's
/// `interior_bbox` (bbox expanded by 1) becomes air. Glass has light opacity 0, so the stored
/// light stays valid. A section left with no non-air block is dropped from the bitmask, as
/// Spigot's `ExtendedBlockStorage.isEmpty()` would.
fn strip_cages(columns: &mut [Column], manifest: &Value) -> Result<StripReport, String> {
    let glass: BTreeMap<u16, &str> = ["glass", "stained_glass", "glass_pane", "stained_glass_pane"]
        .iter()
        .map(|n| {
            (
                block_by_name(n)
                    .unwrap_or_else(|| panic!("mc-data has no block {n}"))
                    .0,
                *n,
            )
        })
        .collect();
    let index: BTreeMap<(i32, i32), usize> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| ((c.x, c.z), i))
        .collect();
    let mut report = StripReport::default();
    let mut touched = BTreeSet::new();
    let cages = manifest["cages"]["list"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for cage in &cages {
        let bb: Vec<i32> = cage["interior_bbox"]
            .as_array()
            .ok_or("cage without interior_bbox")?
            .iter()
            .map(|v| v.as_i64().unwrap_or(0) as i32)
            .collect();
        if bb.len() != 6 {
            return Err("cage interior_bbox must have 6 values".into());
        }
        report.cages += 1;
        for x in bb[0] - 1..=bb[3] + 1 {
            for y in bb[1] - 1..=bb[4] + 1 {
                for z in bb[2] - 1..=bb[5] + 1 {
                    let Some(&ci) = index.get(&(x >> 4, z >> 4)) else {
                        continue;
                    };
                    let col = &mut columns[ci];
                    let Some(ord) = section_ordinal(col.mask, y >> 4) else {
                        continue;
                    };
                    let idx = (((y & 15) << 8) | ((z & 15) << 4) | (x & 15)) as usize;
                    let off = ord * 8192 + idx * 2;
                    let state = u16::from_le_bytes([col.data[off], col.data[off + 1]]);
                    if let Some(name) = glass.get(&(state >> 4)) {
                        col.data[off] = 0;
                        col.data[off + 1] = 0;
                        report.blocks += 1;
                        *report.by_block.entry((*name).to_owned()).or_default() += 1;
                        touched.insert(ci);
                    }
                }
            }
        }
    }
    for ci in touched {
        let col = &columns[ci];
        let mut keep = col.mask;
        for s in 0..16 {
            if let Some(ord) = section_ordinal(col.mask, s) {
                let empty = col.data[ord * 8192..(ord + 1) * 8192]
                    .iter()
                    .all(|&b| b == 0);
                if empty {
                    keep &= !(1 << s);
                    report.sections_dropped += 1;
                }
            }
        }
        if keep != col.mask {
            let payload = rebuild_payload(col, keep);
            let col = &mut columns[ci];
            col.data = payload;
            col.mask = keep;
            debug_assert_eq!(
                col.data.len(),
                mc_world::expected_chunk_data_len(col.mask, true, true)
            );
        }
    }
    Ok(report)
}

// =============================================================================================
// Terrain model (decoded with mc_world) and the walkable-surface grid
// =============================================================================================

struct Terrain {
    chunks: BTreeMap<(i32, i32), mc_world::Chunk>,
    /// Blocks placed during the scene (E9's bridge).
    overlay: BTreeMap<(i32, i32, i32), u16>,
}

#[derive(Clone, Copy, Debug)]
struct Probe {
    wall: bool,
    floor: Option<f64>,
}

impl Terrain {
    fn new(columns: &[Column]) -> Result<Terrain, String> {
        let mut chunks = BTreeMap::new();
        for c in columns {
            let mut chunk = mc_world::Chunk::empty();
            chunk
                .fill_from_packet(true, c.mask, true, &c.data)
                .map_err(|e| format!("decode ({},{}): {e:?}", c.x, c.z))?;
            chunks.insert((c.x, c.z), chunk);
        }
        Ok(Terrain {
            chunks,
            overlay: BTreeMap::new(),
        })
    }

    fn state(&self, x: i32, y: i32, z: i32) -> u16 {
        if let Some(&s) = self.overlay.get(&(x, y, z)) {
            return s;
        }
        if !(0..256).contains(&y) {
            return 0;
        }
        match self.chunks.get(&(x >> 4, z >> 4)) {
            Some(c) => c.block_at((x & 15) as u8, y as u8, (z & 15) as u8).id,
            None => 0,
        }
    }

    /// Top of the block's collision boxes, if it has any.
    fn top_of(&self, x: i32, y: i32, z: i32) -> Option<f64> {
        let boxes = BlockState::from_raw(self.state(x, y, z))
            .props()
            .collision
            .boxes();
        if boxes.is_empty() {
            return None;
        }
        Some(f64::from(y) + boxes.iter().fold(0.0f64, |m, b| m.max(b.max_y)))
    }

    /// The floor under a body at `(x, feet, z)` (highest collision top at or below
    /// `feet + 0.6`, the vanilla step height) and whether the body span is blocked.
    fn probe(&self, x: f64, z: f64, feet: f64) -> Probe {
        let bx = x.floor() as i32;
        let bz = z.floor() as i32;
        let mut wall = false;
        let lo = (feet + 0.6).floor() as i32;
        let hi = (feet + 1.79).floor() as i32;
        for y in lo..=hi {
            let boxes = BlockState::from_raw(self.state(bx, y, bz))
                .props()
                .collision
                .boxes();
            for b in boxes {
                let b_lo = f64::from(y) + b.min_y;
                let b_hi = f64::from(y) + b.max_y;
                if b_hi > feet + 0.6 + 1e-6 && b_lo < feet + 1.8 {
                    wall = true;
                }
            }
        }
        let mut floor = None;
        let start = (feet + 0.6).floor() as i32;
        for y in (start - 72..=start).rev() {
            if y < 0 {
                break;
            }
            if let Some(top) = self.top_of(bx, y, bz) {
                if top <= feet + 0.6 + 1e-6 {
                    floor = Some(top);
                    break;
                }
            }
        }
        Probe { wall, floor }
    }

    /// A column with no colliding block at all: the void.
    fn is_void_column(&self, x: i32, z: i32) -> bool {
        (0..=140).all(|y| self.top_of(x, y, z).is_none())
    }

    /// The top-most walkable surface of a column: a colliding block no taller than a full
    /// block, not a chest. Plants, torches and liquids have no collision and are skipped.
    fn surface(&self, x: i32, z: i32) -> Option<Cell> {
        for y in (1..=140).rev() {
            let st = self.state(x, y, z);
            let boxes = BlockState::from_raw(st).props().collision.boxes();
            if boxes.is_empty() {
                continue;
            }
            let max_y = boxes.iter().fold(0.0f64, |m, b| m.max(b.max_y));
            let id = st >> 4;
            if max_y > 1.0 || id == 54 || id == 146 || id == 130 {
                return None;
            }
            return Some(Cell {
                h: f64::from(y) + max_y,
                top_y: y,
            });
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    /// Feet height when standing on this column.
    h: f64,
    top_y: i32,
}

struct Grid {
    x0: i32,
    z0: i32,
    w: i32,
    d: i32,
    cells: Vec<Option<Cell>>,
}

impl Grid {
    fn build(terrain: &Terrain, x0: i32, z0: i32, x1: i32, z1: i32) -> Grid {
        let w = x1 - x0 + 1;
        let d = z1 - z0 + 1;
        let mut cells = Vec::with_capacity((w * d) as usize);
        for z in z0..=z1 {
            for x in x0..=x1 {
                cells.push(terrain.surface(x, z));
            }
        }
        Grid {
            x0,
            z0,
            w,
            d,
            cells,
        }
    }
    fn get(&self, x: i32, z: i32) -> Option<Cell> {
        let (ix, iz) = (x - self.x0, z - self.z0);
        if ix < 0 || iz < 0 || ix >= self.w || iz >= self.d {
            return None;
        }
        self.cells[(iz * self.w + ix) as usize]
    }
    fn set(&mut self, x: i32, z: i32, cell: Option<Cell>) {
        let (ix, iz) = (x - self.x0, z - self.z0);
        if ix >= 0 && iz >= 0 && ix < self.w && iz < self.d {
            self.cells[(iz * self.w + ix) as usize] = cell;
        }
    }
    /// Largest radius (blocks) within which every column is walkable within `tol` of `h0`.
    fn open_radius(&self, x: i32, z: i32, h0: f64, tol: f64, max_r: i32) -> i32 {
        let mut r = 0;
        for rr in 1..=max_r {
            for dz in -rr..=rr {
                for dx in -rr..=rr {
                    let d2 = dx * dx + dz * dz;
                    if d2 > rr * rr || d2 <= (rr - 1) * (rr - 1) {
                        continue;
                    }
                    match self.get(x + dx, z + dz) {
                        Some(c) if (c.h - h0).abs() <= tol => {}
                        _ => return r,
                    }
                }
            }
            r = rr;
        }
        r
    }
}

/// Every standable floor of a column (a colliding block no taller than a full block, not a
/// chest, with two non-colliding blocks above), top first. Multi-level, so a walkway under an
/// arch is found even though the arch is the column's top block.
fn column_floors(terrain: &Terrain, x: i32, z: i32) -> Vec<f64> {
    let mut out = Vec::new();
    for y in (1..=140).rev() {
        let st = terrain.state(x, y, z);
        let boxes = BlockState::from_raw(st).props().collision.boxes();
        if boxes.is_empty() {
            continue;
        }
        let id = st >> 4;
        let max_y = boxes.iter().fold(0.0f64, |m, b| m.max(b.max_y));
        if max_y > 1.0 || id == 54 || id == 146 || id == 130 {
            continue;
        }
        if terrain.top_of(x, y + 1, z).is_some() || terrain.top_of(x, y + 2, z).is_some() {
            continue;
        }
        out.push(f64::from(y) + max_y);
    }
    out
}

/// Dijkstra over standable floors (8-neighbour, no corner cutting, up ≤ 1.25 with a jump, down
/// ≤ 3). Nodes are (x, z, floor height), so stacked floors are separate. Deterministic: integer
/// costs, ties broken by the node key. Returns column-centre waypoints at floor height.
fn walk_path(terrain: &Terrain, from: V3, to: V3, limit: u32) -> Option<Vec<V3>> {
    const DIRS: [(i32, i32, u32); 8] = [
        (1, 0, 10),
        (-1, 0, 10),
        (0, 1, 10),
        (0, -1, 10),
        (1, 1, 14),
        (1, -1, 14),
        (-1, 1, 14),
        (-1, -1, 14),
    ];
    let mut cache: BTreeMap<(i32, i32), Vec<f64>> = BTreeMap::new();
    let mut floors = |x: i32, z: i32| -> Vec<f64> {
        cache
            .entry((x, z))
            .or_insert_with(|| column_floors(terrain, x, z))
            .clone()
    };
    let best_floor = |list: &[f64], h: f64| -> Option<f64> {
        list.iter()
            .copied()
            .filter(|f| (-3.0..=1.25).contains(&(f - h)))
            .min_by(|a, b| {
                (a - h)
                    .abs()
                    .partial_cmp(&(b - h).abs())
                    .unwrap_or(Ordering::Equal)
            })
    };
    let key = |h: f64| (h * 1000.0).round() as i64;
    let start_col = (from.x.floor() as i32, from.z.floor() as i32);
    let goal_col = (to.x.floor() as i32, to.z.floor() as i32);
    let h0 = floors(start_col.0, start_col.1)
        .into_iter()
        .min_by(|a, b| {
            (a - from.y)
                .abs()
                .partial_cmp(&(b - from.y).abs())
                .unwrap_or(Ordering::Equal)
        })?;
    let start = (start_col.0, start_col.1, key(h0));
    let mut dist: BTreeMap<(i32, i32, i64), u32> = BTreeMap::new();
    let mut prev: BTreeMap<(i32, i32, i64), (i32, i32, i64)> = BTreeMap::new();
    let mut heap = BinaryHeap::new();
    dist.insert(start, 0);
    heap.push(Reverse((0u32, start)));
    while let Some(Reverse((cost, node))) = heap.pop() {
        let (x, z, hk) = node;
        let h = hk as f64 / 1000.0;
        if (x, z) == goal_col && (h - to.y).abs() <= 1.6 {
            let mut out = vec![node];
            let mut cur = node;
            while let Some(&p) = prev.get(&cur) {
                out.push(p);
                cur = p;
            }
            out.reverse();
            return Some(
                out.into_iter()
                    .map(|(x, z, hk)| {
                        V3::new(f64::from(x) + 0.5, hk as f64 / 1000.0, f64::from(z) + 0.5)
                    })
                    .collect(),
            );
        }
        if cost > limit || dist.get(&node).is_some_and(|&d| d < cost) {
            continue;
        }
        for &(dx, dz, c) in &DIRS {
            let (nx, nz) = (x + dx, z + dz);
            let Some(nh) = best_floor(&floors(nx, nz), h) else {
                continue;
            };
            if dx != 0 && dz != 0 {
                // No corner cutting: both orthogonal neighbours must be walkable without a jump.
                let flat = |f: Option<f64>| f.is_some_and(|f| (f - h).abs() <= 0.6);
                let a = best_floor(&floors(x + dx, z), h);
                let b = best_floor(&floors(x, z + dz), h);
                if !flat(a) || !flat(b) || (nh - h).abs() > 0.6 {
                    continue;
                }
            }
            // Prefer flat routes: a jump-up costs three extra blocks, a drop half of one.
            let extra = if nh - h > 0.6 {
                30
            } else if h - nh > 0.6 {
                5
            } else {
                0
            };
            let nc = cost + c + extra;
            let next = (nx, nz, key(nh));
            if dist.get(&next).is_none_or(|&d| nc < d) {
                dist.insert(next, nc);
                prev.insert(next, node);
                heap.push(Reverse((nc, next)));
            }
        }
    }
    None
}

fn cell_centre(grid: &Grid, x: i32, z: i32) -> Option<V3> {
    grid.get(x, z)
        .map(|c| V3::new(f64::from(x) + 0.5, c.h, f64::from(z) + 0.5))
}

// =============================================================================================
// Scene plan: anchor and group placements, chosen by scanning the decoded columns
// =============================================================================================

#[derive(Clone, Debug)]
struct ChestPlan {
    pos: (i32, i32, i32),
    access: V3,
}

#[derive(Clone, Debug)]
struct BridgePlan {
    island_spawn: V3,
    start: V3,
    blocks: Vec<(i32, i32, i32)>,
    landing: V3,
    dir: V3,
    block_state: u16,
}

#[derive(Clone, Debug)]
struct ScenePlan {
    centre: V3,
    anchor: V3,
    anchor_report: Value,
    open_dir: V3,
    e1_start: V3,
    ally_home: V3,
    b_centre: V3,
    b_edge: V3,
    b_void_dir: V3,
    archer7: V3,
    archer8: V3,
    bridge: BridgePlan,
    e11_start: V3,
    chests: Vec<ChestPlan>,
    pearl_landing_hint: V3,
    placements_report: Value,
}

fn mid_bbox(manifest: &Value) -> [i32; 6] {
    let bb = manifest["islands"]["mid"]["bbox"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|v| v.as_i64().unwrap_or(0) as i32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if bb.len() == 6 {
        [bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]]
    } else {
        [-100_000, 0, -100_000, 100_000, 255, 100_000]
    }
}

fn in_mid(bb: &[i32; 6], x: i32, z: i32) -> bool {
    x >= bb[0] && x <= bb[3] && z >= bb[2] && z <= bb[5]
}

fn mid_chests(manifest: &Value, bb: &[i32; 6]) -> Vec<((i32, i32, i32), i64)> {
    let mut out = Vec::new();
    for c in manifest["chests"].as_array().cloned().unwrap_or_default() {
        let p: Vec<i32> = c["pos"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect())
            .unwrap_or_default();
        if p.len() == 3 && in_mid(bb, p[0], p[2]) {
            out.push(((p[0], p[1], p[2]), c["facing_meta"].as_i64().unwrap_or(2)));
        }
    }
    out
}

const ANCHOR_REQUIRED_OPEN: i32 = 7;
const ANCHOR_REQUESTED_MAX: f64 = 15.0;

fn plan_scene(terrain: &mut Terrain, grid: &Grid, manifest: &Value) -> Result<ScenePlan, String> {
    let centre = V3::new(
        manifest["centre"]["x"]
            .as_f64()
            .ok_or("manifest centre.x")?,
        manifest["centre"]["y"]
            .as_f64()
            .ok_or("manifest centre.y")?,
        manifest["centre"]["z"]
            .as_f64()
            .ok_or("manifest centre.z")?,
    );
    let bb = mid_bbox(manifest);
    let chests = mid_chests(manifest, &bb);

    // ---- anchor: open, gentle mid-island ground, nearest the centre --------------------------
    // Rule: walkable column inside the mid island whose open radius (every column within R is
    // walkable and within 1 block of its height) is at least ANCHOR_REQUIRED_OPEN; among those,
    // most mid chests 10-15 blocks away (E10's loot), then nearest the centre, then
    // x, z. The spec asks for ≤ 15 blocks from the centre; the report records what was found.
    let mut best: Option<(usize, f64, i32, i32, i32, f64)> = None;
    let mut best_within: (i32, f64) = (0, f64::MAX);
    let mut considered = 0usize;
    for z in bb[2]..=bb[5] {
        for x in bb[0]..=bb[3] {
            let Some(c) = grid.get(x, z) else {
                continue;
            };
            considered += 1;
            let p = V3::new(f64::from(x) + 0.5, c.h, f64::from(z) + 0.5);
            let dc = p.hdist(centre);
            let r = grid.open_radius(x, z, c.h, 1.0, 10);
            if dc <= ANCHOR_REQUESTED_MAX && r > best_within.0 {
                best_within = (r, dc);
            }
            if r < ANCHOR_REQUIRED_OPEN {
                continue;
            }
            let near_chests = chests
                .iter()
                .filter(|((cx, _, cz), _)| {
                    let d = p.hdist(V3::new(f64::from(*cx) + 0.5, 0.0, f64::from(*cz) + 0.5));
                    (10.0..=15.0).contains(&d)
                })
                .count()
                .min(4);
            let better = match &best {
                None => true,
                Some((bc, bd, _, bx, bz, _)) => {
                    (near_chests, -dc, -x, -z)
                        .partial_cmp(&(*bc, -*bd, -*bx, -*bz))
                        .unwrap_or(Ordering::Equal)
                        == Ordering::Greater
                }
            };
            if better {
                best = Some((near_chests, dc, r, x, z, c.h));
            }
        }
    }
    let (anchor_chests, anchor_dist, anchor_r, ax, az, ah) =
        best.ok_or("no open ground with the required radius on the mid island")?;
    let anchor = V3::new(f64::from(ax) + 0.5, ah, f64::from(az) + 0.5);
    let anchor_report = json!({
        "column": [ax, grid.get(ax, az).map(|c| c.top_y).unwrap_or(0), az],
        "feet": anchor.json(),
        "distance_from_centre": round3(anchor_dist),
        "open_radius_blocks": anchor_r,
        "mid_chests_10_to_15_blocks": anchor_chests,
        "columns_considered": considered,
        "rule": "walkable mid-island column with every column within R walkable and within 1 block of its height, R >= 7; then most mid chests 10-15 blocks away, nearest the centre, then x, z",
        "requested": "open, flat mid-island ground within 15 blocks of manifest.centre",
        "best_open_radius_within_15_blocks": best_within.0,
        "note": if anchor_dist > ANCHOR_REQUESTED_MAX {
            format!("no column within {ANCHOR_REQUESTED_MAX} blocks of the centre has open radius >= {ANCHOR_REQUIRED_OPEN} (best there: {}); the centre is the fortress plaza, a 9x7 floor with void quarter-holes. Anchor moved to the nearest open ring ground.", best_within.0)
        } else {
            "within the requested distance".to_owned()
        },
    });

    // ---- open direction at A: the direction with the longest gentle walk -------------------
    let mut open_dir = V3::new(0.0, 0.0, 1.0);
    let mut open_best = -1.0;
    for k in 0..32 {
        let a = f64::from(k) * std::f64::consts::TAU / 32.0;
        let d = V3::new(a.cos(), 0.0, a.sin());
        let mut run = 0.0;
        let mut s = 0.5;
        while s <= 24.0 {
            let p = anchor.add(d.scale(s));
            match grid.get(p.x.floor() as i32, p.z.floor() as i32) {
                Some(c) if (c.h - ah).abs() <= 2.0 => run = s,
                _ => break,
            }
            s += 0.5;
        }
        if run > open_best + 1e-9 {
            open_best = run;
            open_dir = d;
        }
    }

    // ---- group A: E1 opposite the open side (pushes the viewer back towards open ground) ----
    let e1_start = find_walkable_near(grid, anchor.add(open_dir.scale(2.7)), 2.0)
        .unwrap_or_else(|| anchor.add(open_dir.scale(2.7)));
    // Teammate and E2 share a home 5.5-7 blocks from A, perpendicular to the open direction.
    let mut ally_home = None;
    'home: for dist in [6.0, 5.5, 6.5, 7.0, 5.0] {
        for k in 0..16 {
            let side = if k % 2 == 0 { 1.0 } else { -1.0 };
            let a = (std::f64::consts::FRAC_PI_2 + f64::from(k / 2) * 0.2) * side;
            let d = V3::new(
                open_dir.x * a.cos() - open_dir.z * a.sin(),
                0.0,
                open_dir.x * a.sin() + open_dir.z * a.cos(),
            );
            let p = anchor.add(d.scale(dist));
            let (px, pz) = (p.x.floor() as i32, p.z.floor() as i32);
            if let Some(c) = grid.get(px, pz) {
                if grid.open_radius(px, pz, c.h, 1.0, 4) >= 3 && (c.h - ah).abs() <= 1.0 {
                    ally_home = Some(V3::new(f64::from(px) + 0.5, c.h, f64::from(pz) + 0.5));
                    break 'home;
                }
            }
        }
    }
    let ally_home = ally_home.ok_or("no open ground 5-7 blocks from the anchor for T/E2")?;

    // ---- group B: 16-24 blocks from A, open, with a void edge within 8 blocks --------------
    let mut b_best: Option<(f64, V3, V3, V3)> = None;
    for z in bb[2]..=bb[5] {
        for x in bb[0]..=bb[3] {
            let Some(c) = grid.get(x, z) else {
                continue;
            };
            let p = V3::new(f64::from(x) + 0.5, c.h, f64::from(z) + 0.5);
            let d = p.hdist(anchor);
            if !(16.0..=24.0).contains(&d) || (c.h - ah).abs() > 3.0 {
                continue;
            }
            let r = grid.open_radius(x, z, c.h, 1.0, 6);
            if r < 4 {
                continue;
            }
            let Some((edge, void_dir, edge_d)) = nearest_void_edge(terrain, grid, x, z, 8) else {
                continue;
            };
            if edge_d < 4.0 {
                continue;
            }
            // Prefer ~17.5 blocks from A (the fight's near side then dips inside the 16-block
            // particle radius), a roomy spot, and an edge 4-8 blocks away.
            let score = -(d - 17.5).abs() + f64::from(r) * 0.5 - (edge_d - 5.5).abs() * 0.3;
            let better = match &b_best {
                None => true,
                Some((s, bp, _, _)) => {
                    score > *s + 1e-9 || ((score - *s).abs() <= 1e-9 && (p.x, p.z) < (bp.x, bp.z))
                }
            };
            if better {
                b_best = Some((score, p, edge, void_dir));
            }
        }
    }
    let (_, b_centre, b_edge, b_void_dir) =
        b_best.ok_or("no group-B spot 16-24 blocks from the anchor with a void edge")?;

    // ---- group C: two archers 30-45 blocks from A, 18-34 apart, clear line of sight, the
    // segment between them crossing as much void as possible ("across the mid ring") ---------
    let mut cand = Vec::new();
    for z in (bb[2]..=bb[5]).step_by(2) {
        for x in (bb[0]..=bb[3]).step_by(2) {
            let Some(c) = grid.get(x, z) else {
                continue;
            };
            let p = V3::new(f64::from(x) + 0.5, c.h, f64::from(z) + 0.5);
            let d = p.hdist(anchor);
            if (30.0..=45.0).contains(&d) && grid.open_radius(x, z, c.h, 1.0, 3) >= 2 {
                cand.push(p);
            }
        }
    }
    let mut c_best: Option<(f64, V3, V3)> = None;
    for i in 0..cand.len() {
        for j in i + 1..cand.len() {
            let (a, b) = (cand[i], cand[j]);
            let sep = a.hdist(b);
            if !(18.0..=34.0).contains(&sep) || !line_of_sight(terrain, a, b) {
                continue;
            }
            let samples = 24;
            let mut void = 0;
            for k in 1..samples {
                let p = a.add(b.sub(a).scale(f64::from(k) / f64::from(samples)));
                if grid.get(p.x.floor() as i32, p.z.floor() as i32).is_none() {
                    void += 1;
                }
            }
            let score = f64::from(void) - (sep - 26.0).abs() * 0.1;
            let better = match &c_best {
                None => true,
                Some((s, pa, pb)) => {
                    score > *s + 1e-9
                        || ((score - *s).abs() <= 1e-9
                            && (a.x, a.z, b.x, b.z) < (pa.x, pa.z, pb.x, pb.z))
                }
            };
            if better {
                c_best = Some((score, a, b));
            }
        }
    }
    let (_, archer7, archer8) = c_best.ok_or("no bow-duel pair 30-45 blocks from the anchor")?;

    // ---- E10's chests: the 4 mid chests nearest A by walking distance ---------------------
    let mut reach = Vec::new();
    for &((cx, cy, cz), facing) in &chests {
        let front = match facing {
            2 => (cx, cz - 1),
            3 => (cx, cz + 1),
            4 => (cx - 1, cz),
            _ => (cx + 1, cz),
        };
        let mut options = vec![
            front,
            (cx, cz - 1),
            (cx, cz + 1),
            (cx - 1, cz),
            (cx + 1, cz),
        ];
        options.dedup();
        for (sx, sz) in options {
            // Stand on the floor level the chest sits on.
            let Some(h) = column_floors(terrain, sx, sz)
                .into_iter()
                .find(|h| (h - f64::from(cy)).abs() <= 0.6)
            else {
                continue;
            };
            let access = V3::new(f64::from(sx) + 0.5, h, f64::from(sz) + 0.5);
            if let Some(path) = walk_path(terrain, anchor, access, 1_500) {
                reach.push((path.len(), (cx, cy, cz), access));
                break;
            }
        }
    }
    reach.sort_by_key(|a| (a.0, a.1));
    let chests: Vec<ChestPlan> = reach
        .iter()
        .take(4)
        .map(|&(_, pos, access)| ChestPlan { pos, access })
        .collect();
    if chests.len() < 4 {
        return Err(format!(
            "only {} mid chests are reachable from the anchor",
            chests.len()
        ));
    }

    // ---- E9's bridge: a spawn island outside the 48-block square, gap towards A ------------
    let bridge = plan_bridge(terrain, grid, manifest, anchor)?;
    let e11_start = bridge.start.sub(bridge.dir.scale(2.0));
    let e11_start = find_walkable_near(grid, e11_start, 3.0).unwrap_or(bridge.start);

    // ---- E1's pearl lands ~6 blocks away on the open side -----------------------------------
    let pearl_landing_hint = find_walkable_near(grid, anchor.add(open_dir.scale(6.0)), 2.5)
        .unwrap_or_else(|| anchor.add(open_dir.scale(6.0)));

    let placements_report = json!({
        "open_direction": open_dir.json(),
        "e1_start": e1_start.json(),
        "ally_home": ally_home.json(),
        "group_b_centre": b_centre.json(),
        "group_b_distance_from_anchor": round3(b_centre.hdist(anchor)),
        "group_b_void_edge": b_edge.json(),
        "archer_e7": archer7.json(),
        "archer_e8": archer8.json(),
        "archers_distance_from_anchor": [round3(archer7.hdist(anchor)), round3(archer8.hdist(anchor))],
        "archer_separation": round3(archer7.hdist(archer8)),
        "chests": chests.iter().map(|c| json!({"pos": [c.pos.0, c.pos.1, c.pos.2], "distance_from_anchor": round3(V3::new(f64::from(c.pos.0) + 0.5, 0.0, f64::from(c.pos.2) + 0.5).hdist(anchor))})).collect::<Vec<_>>(),
        "bridge": {
            "island_spawn": bridge.island_spawn.json(),
            "start": bridge.start.json(),
            "landing": bridge.landing.json(),
            "blocks": bridge.blocks.len(),
        },
        "e11_start": e11_start.json(),
        "pearl_landing": pearl_landing_hint.json(),
    });
    let _ = terrain; // terrain is mutated later by the server (bridge overlay)
    Ok(ScenePlan {
        centre,
        anchor,
        anchor_report,
        open_dir,
        e1_start,
        ally_home,
        b_centre,
        b_edge,
        b_void_dir,
        archer7,
        archer8,
        bridge,
        e11_start,
        chests,
        pearl_landing_hint,
        placements_report,
    })
}

fn find_walkable_near(grid: &Grid, p: V3, radius: f64) -> Option<V3> {
    let (cx, cz) = (p.x.floor() as i32, p.z.floor() as i32);
    let r = radius.ceil() as i32;
    let mut best: Option<(f64, i32, i32)> = None;
    for dz in -r..=r {
        for dx in -r..=r {
            let (x, z) = (cx + dx, cz + dz);
            let Some(_) = grid.get(x, z) else {
                continue;
            };
            let d = V3::new(f64::from(x) + 0.5, 0.0, f64::from(z) + 0.5).hdist(p);
            if d <= radius && best.is_none_or(|(bd, bx, bz)| (d, x, z) < (bd, bx, bz)) {
                best = Some((d, x, z));
            }
        }
    }
    best.and_then(|(_, x, z)| cell_centre(grid, x, z))
}

/// Nearest walkable column (within `max` blocks) that has a void column as a 4-neighbour.
fn nearest_void_edge(
    terrain: &Terrain,
    grid: &Grid,
    x: i32,
    z: i32,
    max: i32,
) -> Option<(V3, V3, f64)> {
    let mut best: Option<(f64, i32, i32, i32, i32)> = None;
    for dz in -max..=max {
        for dx in -max..=max {
            let (ex, ez) = (x + dx, z + dz);
            if grid.get(ex, ez).is_none() {
                continue;
            }
            for (vx, vz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                if grid.get(ex + vx, ez + vz).is_none() && terrain.is_void_column(ex + vx, ez + vz)
                {
                    let d = f64::from(dx * dx + dz * dz).sqrt();
                    if d <= f64::from(max)
                        && best.is_none_or(|(bd, bx, bz, _, _)| (d, ex, ez) < (bd, bx, bz))
                    {
                        best = Some((d, ex, ez, vx, vz));
                    }
                }
            }
        }
    }
    let (d, ex, ez, vx, vz) = best?;
    Some((
        cell_centre(grid, ex, ez)?,
        V3::new(f64::from(vx), 0.0, f64::from(vz)),
        d,
    ))
}

fn line_of_sight(terrain: &Terrain, a: V3, b: V3) -> bool {
    let ea = a.add(V3::new(0.0, 1.62, 0.0));
    let eb = b.add(V3::new(0.0, 1.0, 0.0));
    let n = (ea.dist(eb) * 4.0).ceil() as i32;
    for k in 1..n {
        let p = ea.add(eb.sub(ea).scale(f64::from(k) / f64::from(n)));
        if terrain
            .top_of(p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
            .is_some()
        {
            return false;
        }
    }
    true
}

fn plan_bridge(
    terrain: &Terrain,
    grid: &Grid,
    manifest: &Value,
    anchor: V3,
) -> Result<BridgePlan, String> {
    let stone = block_by_name("stone").ok_or("mc-data has no stone")?.0 << 4;
    let mut best: Option<(f64, BridgePlan)> = None;
    for cage in manifest["cages"]["list"]
        .as_array()
        .cloned()
        .unwrap_or_default()
    {
        let s: Vec<f64> = cage["spawn"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
            .unwrap_or_default();
        if s.len() != 3 {
            continue;
        }
        let spawn = V3::new(s[0], s[1], s[2]);
        let (dx, dz) = (anchor.x - spawn.x, anchor.z - spawn.z);
        if dx.abs() <= RANGE_PLAYER && dz.abs() <= RANGE_PLAYER {
            continue; // E9 must start outside the viewer's tracking square
        }
        // Island standing point under the (stripped) cage.
        let Some(island) = find_walkable_near(grid, spawn, 2.0) else {
            continue;
        };
        // Bridge straight towards the anchor, rasterised 4-connected so every block shares an
        // edge with the previous one (a diagonal bridge is a staircase of edge-adjacent blocks).
        let dir = anchor.sub(island).hnorm();
        let (sx, sz) = (
            if dir.x > 1e-9 {
                1
            } else if dir.x < -1e-9 {
                -1
            } else {
                0
            },
            if dir.z > 1e-9 {
                1
            } else if dir.z < -1e-9 {
                -1
            } else {
                0
            },
        );
        let origin = island;
        let perp = |x: i32, z: i32| {
            let p = V3::new(f64::from(x) + 0.5, 0.0, f64::from(z) + 0.5).sub(origin);
            (p.x * dir.z - p.z * dir.x).abs()
        };
        let step = |x: i32, z: i32| -> (i32, i32) {
            if sx == 0 {
                (x, z + sz)
            } else if sz == 0 || perp(x + sx, z) <= perp(x, z + sz) {
                (x + sx, z)
            } else {
                (x, z + sz)
            }
        };
        let (mut x, mut z) = (island.x.floor() as i32, island.z.floor() as i32);
        let mut edge = None;
        for _ in 0..30 {
            let (nx, nz) = step(x, z);
            match grid.get(nx, nz) {
                Some(_) => {
                    x = nx;
                    z = nz;
                }
                None => {
                    edge = grid.get(x, z).map(|c| (x, z, c));
                    break;
                }
            }
        }
        let Some((ex, ez, edge_cell)) = edge else {
            continue;
        };
        let by = edge_cell.top_y;
        let mut blocks = Vec::new();
        let (mut bx, mut bz) = step(ex, ez);
        let mut landing = None;
        for _ in 0..60 {
            if !terrain.is_void_column(bx, bz) {
                if let Some(c) = grid.get(bx, bz) {
                    let dh = c.h - f64::from(by + 1);
                    if (-2.5..=1.0).contains(&dh) {
                        landing = Some(V3::new(f64::from(bx) + 0.5, c.h, f64::from(bz) + 0.5));
                    }
                }
                break;
            }
            blocks.push((bx, by, bz));
            let next = step(bx, bz);
            bx = next.0;
            bz = next.1;
        }
        let Some(landing) = landing else {
            continue;
        };
        if blocks.len() < 8 {
            continue;
        }
        // E9 must enter the viewer's 48-block tracking square during or right after the bridge.
        let (lx, lz) = (anchor.x - landing.x, anchor.z - landing.z);
        if lx.abs() > RANGE_PLAYER - 4.0 || lz.abs() > RANGE_PLAYER - 4.0 {
            continue;
        }
        let start = V3::new(f64::from(ex) + 0.5, edge_cell.h, f64::from(ez) + 0.5);
        // Prefer the longest gap (more placements), then the nearest island.
        let score = blocks.len() as f64 - spawn.hdist(anchor) * 0.01;
        let plan = BridgePlan {
            island_spawn: island,
            start,
            blocks,
            landing,
            dir,
            block_state: stone,
        };
        let better = match &best {
            None => true,
            Some((s, p)) => {
                score > *s + 1e-9
                    || ((score - *s).abs() <= 1e-9
                        && (plan.start.x, plan.start.z) < (p.start.x, p.start.z))
            }
        };
        if better {
            best = Some((score, plan));
        }
    }
    best.map(|(_, p)| p).ok_or_else(|| {
        "no spawn island outside the tracking square has a straight void gap to mid".to_owned()
    })
}

// =============================================================================================
// Items
// =============================================================================================

fn item_id(name: &str) -> i16 {
    item_by_name(name)
        .unwrap_or_else(|| panic!("mc-data has no item {name}"))
        .0 as i16
}

fn stack(name: &str, count: i32) -> ItemStack {
    ItemStack {
        item_id: item_id(name),
        count,
        damage: 0,
        nbt: None,
    }
}

fn max_damage(name: &str) -> i16 {
    item_props(item_by_name(name).unwrap_or_else(|| panic!("mc-data has no item {name}")))
        .max_damage as i16
}

fn worn(name: &str, rng: &mut Rng, lo: f64, hi: f64) -> ItemStack {
    let mut s = stack(name, 1);
    s.damage = (f64::from(max_damage(name)) * rng.range(lo, hi)) as i16;
    s
}

fn enchanted(mut s: ItemStack, enchants: &[(i16, i16)]) -> ItemStack {
    let list: Vec<Nbt> = enchants
        .iter()
        .map(|&(id, lvl)| {
            let mut c = NbtCompound::new();
            c.insert("id", Nbt::Short(id));
            c.insert("lvl", Nbt::Short(lvl));
            Nbt::Compound(c)
        })
        .collect();
    let mut tag = s.nbt.take().unwrap_or_default();
    tag.insert("ench", Nbt::list(list));
    s.nbt = Some(tag);
    s
}

fn firework_rocket() -> ItemStack {
    let mut explosion = NbtCompound::new();
    explosion.insert("Flicker", Nbt::Byte(1));
    explosion.insert("Trail", Nbt::Byte(1));
    explosion.insert("Type", Nbt::Byte(1));
    explosion.insert("Colors", Nbt::IntArray(vec![0x00FF_5555, 0x00FF_FF55]));
    explosion.insert("FadeColors", Nbt::IntArray(vec![0x00FF_FFFF]));
    let mut fireworks = NbtCompound::new();
    fireworks.insert("Flight", Nbt::Byte(1));
    fireworks.insert("Explosions", Nbt::list(vec![Nbt::Compound(explosion)]));
    let mut tag = NbtCompound::new();
    tag.insert("Fireworks", Nbt::Compound(fireworks));
    let mut s = stack("fireworks", 1);
    s.nbt = Some(tag);
    s
}

/// What one player's gear means for the cosmetic damage model.
#[derive(Clone, Debug, Default)]
struct GearStats {
    sword_damage: f32,
    sharpness: i16,
    armour_points: i32,
    protection_epf: i32,
    power: i16,
}

/// Equipment array in S04 slot order: 0 held, 1 boots, 2 leggings, 3 chestplate, 4 helmet.
type Equip = [Option<ItemStack>; 5];

fn armour_points(name: &str) -> i32 {
    item_props(item_by_name(name).unwrap_or_else(|| panic!("mc-data has no item {name}")))
        .armor_reduction as i32
}

/// Mixes of diamond and iron armour, 1-3 enchanted pieces (glint), Sharpness II-III
/// diamond or iron swords.
fn fighter_gear(rng: &mut Rng, diamond_bias: f64) -> (Equip, ItemStack, GearStats) {
    let slots = [
        ("boots", 1usize),
        ("leggings", 2usize),
        ("chestplate", 3usize),
        ("helmet", 4usize),
    ];
    let mut equip: Equip = [None, None, None, None, None];
    let mut stats = GearStats::default();
    let enchanted_pieces = rng.int(1, 3) as usize;
    let mut order: Vec<usize> = vec![0, 1, 2, 3];
    for i in (1..order.len()).rev() {
        let j = rng.int(0, i as i64) as usize;
        order.swap(i, j);
    }
    for (k, &(piece, slot)) in slots.iter().enumerate() {
        let material = if rng.chance(diamond_bias) {
            "diamond"
        } else {
            "iron"
        };
        let name = format!("{material}_{piece}");
        let mut s = worn(&name, rng, 0.05, 0.6);
        stats.armour_points += armour_points(&name);
        if order[..enchanted_pieces].contains(&k) {
            let (ench, lvl) = if piece == "boots" && rng.chance(0.4) {
                (ENCH_FEATHER_FALLING, rng.int(1, 4) as i16)
            } else if rng.chance(0.25) {
                (ENCH_PROJ_PROTECTION, rng.int(1, 3) as i16)
            } else {
                (ENCH_PROTECTION, rng.int(1, 4) as i16)
            };
            if ench == ENCH_PROTECTION {
                stats.protection_epf += ((6 + i32::from(lvl * lvl)) as f32 / 3.0 * 0.75) as i32;
            }
            s = enchanted(s, &[(ench, lvl)]);
        }
        equip[slot] = Some(s);
    }
    let sword_material = if rng.chance(diamond_bias) {
        "diamond"
    } else {
        "iron"
    };
    let sword_name = format!("{sword_material}_sword");
    let sharp = rng.int(2, 3) as i16;
    let mut sword = enchanted(
        worn(&sword_name, rng, 0.02, 0.3),
        &[(ENCH_SHARPNESS, sharp)],
    );
    if rng.chance(0.25) {
        sword = enchanted(sword, &[(ENCH_SHARPNESS, sharp), (ENCH_UNBREAKING, 1)]);
    }
    stats.sword_damage = 1.0 + item_props(item_by_name(&sword_name).expect("sword")).attack_damage;
    stats.sharpness = sharp;
    equip[0] = Some(sword.clone());
    (equip, sword, stats)
}

// =============================================================================================
// DataWatcher model (vanilla HashMap iteration order) and attribute snapshots
// =============================================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WKind {
    Player,
    Item,
    Arrow,
    Firework,
    Plain,
}

#[derive(Clone, Debug)]
struct Watcher {
    order: Vec<u8>,
    values: BTreeMap<u8, MetadataValue>,
    dirty: BTreeSet<u8>,
}

impl Watcher {
    fn new(kind: WKind) -> Watcher {
        let mut w = Watcher {
            order: Vec::new(),
            values: BTreeMap::new(),
            dirty: BTreeSet::new(),
        };
        // Entity.entityInit (Entity.java:286-290), in registration order.
        w.add(0, MetadataValue::Byte(0));
        w.add(1, MetadataValue::Short(300));
        w.add(3, MetadataValue::Byte(0));
        w.add(2, MetadataValue::String(String::new()));
        w.add(4, MetadataValue::Byte(0));
        match kind {
            WKind::Player => {
                // EntityLivingBase.entityInit (:213-216) then EntityPlayer.entityInit (:199-202).
                w.add(7, MetadataValue::Int(0));
                w.add(8, MetadataValue::Byte(0));
                w.add(9, MetadataValue::Byte(0));
                w.add(6, MetadataValue::Float(20.0));
                w.add(16, MetadataValue::Byte(0));
                w.add(17, MetadataValue::Float(0.0));
                w.add(18, MetadataValue::Int(0));
                w.add(10, MetadataValue::Byte(0x7F)); // all skin layers, from the client's C15
            }
            WKind::Item => w.add(10, MetadataValue::ItemStack(None)),
            WKind::Arrow => w.add(16, MetadataValue::Byte(0)),
            WKind::Firework => w.add(8, MetadataValue::ItemStack(None)),
            WKind::Plain => {}
        }
        w
    }
    fn add(&mut self, idx: u8, v: MetadataValue) {
        self.order.push(idx);
        self.values.insert(idx, v);
    }
    /// `DataWatcher.updateObject`: marks the slot changed when the value differs (ItemStacks
    /// compare by reference in Java, so setting one always marks it).
    fn set(&mut self, idx: u8, v: MetadataValue) {
        let changed = matches!(v, MetadataValue::ItemStack(_)) || self.values.get(&idx) != Some(&v);
        if changed {
            self.values.insert(idx, v);
            self.dirty.insert(idx);
        }
    }
    fn byte(&self, idx: u8) -> i8 {
        match self.values.get(&idx) {
            Some(MetadataValue::Byte(b)) => *b,
            _ => 0,
        }
    }
    fn set_flag(&mut self, bit: u8, on: bool) {
        let cur = self.byte(0);
        let next = if on {
            cur | (1 << bit)
        } else {
            cur & !(1 << bit)
        };
        self.set(0, MetadataValue::Byte(next));
    }
    /// Java `HashMap<Integer, ...>` iteration: bucket `key & (n-1)`, insertion order within a
    /// bucket, table size doubling past a 0.75 load factor.
    fn vanilla_order(&self) -> Vec<u8> {
        let mut n = 16usize;
        while self.order.len() * 4 > n * 3 {
            n *= 2;
        }
        let mut keys: Vec<(usize, usize, u8)> = self
            .order
            .iter()
            .enumerate()
            .map(|(rank, &k)| ((k as usize) & (n - 1), rank, k))
            .collect();
        keys.sort_unstable();
        keys.into_iter().map(|(_, _, k)| k).collect()
    }
    fn all(&self) -> Vec<MetadataEntry> {
        self.vanilla_order()
            .into_iter()
            .map(|k| MetadataEntry {
                index: k,
                value: self.values[&k].clone(),
            })
            .collect()
    }
    fn take_changed(&mut self) -> Vec<MetadataEntry> {
        let out = self
            .vanilla_order()
            .into_iter()
            .filter(|k| self.dirty.contains(k))
            .map(|k| MetadataEntry {
                index: k,
                value: self.values[&k].clone(),
            })
            .collect();
        self.dirty.clear();
        out
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveEffect {
    id: u8,
    amplifier: u8,
    until: i64,
}

/// `PotionHelper.calcPotionLiquidColor` over the active effects (all show particles).
fn potion_colour(effects: &[ActiveEffect]) -> i32 {
    if effects.is_empty() {
        return 0;
    }
    let (mut r, mut g, mut b, mut n) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for e in effects {
        let c = potion(e.id).map_or(0, |p| p.liquid_color);
        for _ in 0..=e.amplifier {
            r += ((c >> 16) & 255) as f32 / 255.0;
            g += ((c >> 8) & 255) as f32 / 255.0;
            b += (c & 255) as f32 / 255.0;
            n += 1.0;
        }
    }
    let r = (r / n * 255.0) as i32;
    let g = (g / n * 255.0) as i32;
    let b = (b / n * 255.0) as i32;
    (r << 16) | (g << 8) | b
}

fn movement_speed_snapshot(
    sprinting: bool,
    speed_amp: Option<u8>,
) -> s2c::AttributePropertySnapshot {
    let mut modifiers = Vec::new();
    if let Some(amp) = speed_amp {
        modifiers.push(s2c::AttributeModifierSnapshot {
            uuid: Uuid::from_u128(SPEED_MODIFIER_UUID),
            amount: 0.200_000_002_980_232_24 * f64::from(amp + 1),
            operation: s2c::AttributeModifierOperation::Multiply,
        });
    }
    if sprinting {
        modifiers.push(s2c::AttributeModifierSnapshot {
            uuid: Uuid::from_u128(SPRINT_MODIFIER_UUID),
            amount: 0.300_000_011_920_928_96,
            operation: s2c::AttributeModifierOperation::Multiply,
        });
    }
    s2c::AttributePropertySnapshot {
        key: "generic.movementSpeed".into(),
        base_value: 0.100_000_001_490_116_12,
        modifiers,
    }
}

// =============================================================================================
// Server entities
// =============================================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Viewer,
    Ally,
    E(u8),
    Far(u8),
}

impl Role {
    fn label(self) -> String {
        match self {
            Role::Viewer => "V".into(),
            Role::Ally => "T".into(),
            Role::E(n) => format!("E{n}"),
            Role::Far(n) => format!("F{n}"),
        }
    }
}

#[derive(Clone, Debug)]
enum Mode {
    Idle,
    Duel,
    Path { points: VecDeque<V3>, sprint: bool },
    Archer,
    Bridge,
    Loot,
    Retreat { target: V3 },
    Falling,
    Dead,
}

#[derive(Clone, Copy, Debug, Default)]
struct Input {
    wish: V3,
    mag: f64,
    jump: bool,
}

#[derive(Clone, Debug)]
struct Player {
    eid: i32,
    uuid: Uuid,
    name: String,
    team: usize,
    role: Role,
    alive: bool,
    /// Has an emulated body near mid, i.e. a tracker entry.
    scene: bool,
    pos: V3,
    vel: V3,
    yaw: f32,
    pitch: f32,
    on_ground: bool,
    airborne_flag: bool,
    sprinting: bool,
    using: bool,
    health: f32,
    absorption: f32,
    arrows_in: i8,
    effects: Vec<ActiveEffect>,
    equip: Equip,
    sword: Option<ItemStack>,
    gear: GearStats,
    watcher: Watcher,
    attrs_dirty: bool,
    last_damage: i64,
    last_swing: i64,
    rng: Rng,
    input: Input,
    mode: Mode,
    opponent: Option<usize>,
    home: V3,
    home_radius: f64,
    strafe: f64,
    strafe_until: i64,
    desired_r: f64,
    sprint_resume: i64,
    next_wtap: i64,
    use_until: i64,
    allow_void: bool,
    kb_pending: Option<V3>,
    walked: f64,
    next_step: f64,
    ping: i32,
    spectator: bool,
    held_restore: Option<(i64, ItemStack)>,
    cps: f64,
    accuracy: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObjKind {
    Item,
    XpOrb(i16),
    Arrow { shooter: usize, crit: bool },
    Snowball,
    Egg,
    Pearl,
    Potion(i32),
    Hook,
    Firework,
}

#[derive(Clone, Debug)]
struct Obj {
    eid: i32,
    kind: ObjKind,
    owner: Option<usize>,
    pos: V3,
    vel: V3,
    yaw: f32,
    pitch: f32,
    on_ground: bool,
    alive: bool,
    stuck: bool,
    watcher: Watcher,
    velocity_changed: bool,
    target: Option<usize>,
    trail: Option<i32>,
    hooked: Option<usize>,
    retract_at: i64,
    explode_at: i64,
    collect_by: Option<usize>,
    born: i64,
    gravity: f64,
    drag: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ent {
    P(usize),
    O(usize),
}

#[derive(Clone, Debug)]
struct Entry {
    ent: Ent,
    eid: i32,
    range: f64,
    freq: i64,
    send_velocity: bool,
    is_arrow: bool,
    enc: [i32; 3],
    enc_rot: [i32; 2],
    last_head: i32,
    counter: i64,
    since_teleport: i64,
    on_ground: bool,
    last_check: V3,
    first_done: bool,
    last_motion: V3,
    tracked: bool,
    removed: bool,
}

// =============================================================================================
// Statistics
// =============================================================================================

#[derive(Default)]
struct Stats {
    by_seg: [BTreeMap<i32, u64>; 3],
    bytes_by_seg: [u64; 3],
    fight_per_second: Vec<u64>,
    tracked_samples: Vec<Value>,
    counters: BTreeMap<String, u64>,
    sections_sent: usize,
    columns_sent: usize,
    bulks: usize,
    last_chunk_tick: i64,
    last_block_change_tick_in_join: i64,
}

impl Stats {
    fn bump(&mut self, key: &str) {
        *self.counters.entry(key.to_owned()).or_default() += 1;
    }
}

fn packet_name(id: i32) -> &'static str {
    match id {
        0x00 => "S00 KeepAlive",
        0x01 => "S01 JoinGame",
        0x02 => "S02 Chat",
        0x03 => "S03 TimeUpdate",
        0x04 => "S04 EntityEquipment",
        0x05 => "S05 SpawnPosition",
        0x06 => "S06 UpdateHealth",
        0x08 => "S08 PlayerPosLook",
        0x09 => "S09 HeldItemChange",
        0x0B => "S0B Animation",
        0x0C => "S0C SpawnPlayer",
        0x0D => "S0D CollectItem",
        0x0E => "S0E SpawnObject",
        0x11 => "S11 SpawnXPOrb",
        0x12 => "S12 EntityVelocity",
        0x13 => "S13 DestroyEntities",
        0x15 => "S15 EntityRelMove",
        0x16 => "S16 EntityLook",
        0x17 => "S17 EntityLookMove",
        0x18 => "S18 EntityTeleport",
        0x19 => "S19 EntityHeadLook",
        0x1A => "S19 EntityStatus (0x1A)",
        0x1C => "S1C EntityMetadata",
        0x1D => "S1D EntityEffect",
        0x1F => "S1F SetExperience",
        0x20 => "S20 EntityProperties",
        0x21 => "S21 ChunkData",
        0x22 => "S22 MultiBlockChange",
        0x23 => "S23 BlockChange",
        0x24 => "S24 BlockAction",
        0x26 => "S26 MapChunkBulk",
        0x28 => "S28 Effect",
        0x29 => "S29 SoundEffect",
        0x2A => "S2A Particles",
        0x2C => "S2C SpawnGlobalEntity",
        0x2F => "S2F SetSlot",
        0x30 => "S30 WindowItems",
        0x33 => "S33 UpdateSign",
        0x35 => "S35 UpdateTileEntity",
        0x38 => "S38 PlayerListItem",
        0x39 => "S39 PlayerAbilities",
        0x3B => "S3B ScoreboardObjective",
        0x3C => "S3C UpdateScore",
        0x3D => "S3D DisplayScoreboard",
        0x3E => "S3E Teams",
        0x41 => "S41 ServerDifficulty",
        0x44 => "S44 WorldBorder",
        0x47 => "S47 PlayerListHeaderFooter",
        _ => "other",
    }
}

// =============================================================================================
// Packet encoding (copied from optimization_fixture.rs)
// =============================================================================================

fn encode_packet<P: Packet>(packet: &P) -> Vec<u8> {
    let mut bytes = PacketBuffer::new();
    bytes.write_varint(P::ID);
    packet.write(&mut bytes);
    bytes.as_slice().to_vec()
}

fn chat_json(text: &str) -> String {
    json!({ "text": text }).to_string()
}

/// A tile entity's description packet, sent right after the bulk carrying its column
/// (`EntityPlayerMP` sends `getDescriptionPacket()` for each tile entity of a sent chunk).
#[derive(Clone, Debug)]
enum TilePacket {
    Sign(s2c::UpdateSign),
    Data(s2c::UpdateTileEntity),
}

/// Tile description packets keyed by the chunk column that carries them.
type ColumnTiles = Vec<((i32, i32), TilePacket)>;

/// S33 for signs and S35 for skulls (action 4), flower pots (5) and banners (6) from
/// the manifest's `other_tile_entities`. Chests, furnaces and enchanting tables have no
/// description packet in 1.8.
fn tile_entity_packets(manifest: &Value) -> Result<ColumnTiles, String> {
    use base64::Engine as _;
    let mut out = Vec::new();
    for te in manifest["other_tile_entities"]
        .as_array()
        .cloned()
        .unwrap_or_default()
    {
        let id = te["id"].as_str().unwrap_or("");
        let p: Vec<i32> = te["pos"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect())
            .unwrap_or_default();
        if p.len() != 3 {
            continue;
        }
        let position = BlockPos::new(p[0], p[1], p[2]);
        let column = (p[0] >> 4, p[2] >> 4);
        match id {
            "Sign" => {
                let line = |k: &str| te["nbt"][k].as_str().unwrap_or("\"\"").to_owned();
                out.push((
                    column,
                    TilePacket::Sign(s2c::UpdateSign {
                        position,
                        lines: [line("Text1"), line("Text2"), line("Text3"), line("Text4")],
                    }),
                ));
            }
            "Skull" | "Banner" | "FlowerPot" => {
                let action = match id {
                    "Skull" => 4,
                    "FlowerPot" => 5,
                    _ => 6,
                };
                let b64 = te["nbt_b64"]
                    .as_str()
                    .ok_or_else(|| format!("{id} at {p:?} has no nbt_b64"))?;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| format!("{id} at {p:?}: {e}"))?;
                let (_, nbt) =
                    mc_nbt::from_slice(&bytes).map_err(|e| format!("{id} at {p:?}: {e:?}"))?;
                let Nbt::Compound(compound) = nbt else {
                    return Err(format!("{id} at {p:?}: NBT root is not a compound"));
                };
                out.push((
                    column,
                    TilePacket::Data(s2c::UpdateTileEntity {
                        position,
                        action,
                        nbt: Some(compound),
                    }),
                ));
            }
            _ => {}
        }
    }
    Ok(out)
}

// =============================================================================================
// The server emulator
// =============================================================================================

const NAMES_FAR: [&str; 27] = [
    "Mossy_Fern",
    "QuietLark",
    "Ember_Wick",
    "TidalHollow",
    "Kestrel_Nine",
    "Pale_Juniper",
    "Copper_Vane",
    "Hollow_Brook",
    "NightOwlz",
    "Slate_Rider",
    "Amber_Quill",
    "FrostLantern",
    "Gravel_Kid",
    "Linden_Ash",
    "Rook_Ninety",
    "Salt_Marsh",
    "Velvet_Comet",
    "Briar_Patch",
    "Ninefold",
    "Thistle_Pup",
    "Cobalt_Sun",
    "Drift_Willow",
    "Larkspur_K",
    "Onyx_Harbor",
    "Pebble_Duke",
    "Quarry_Fox",
    "Wren_Tally",
];

struct Server<'a> {
    t: i64,
    plan: &'a ScenePlan,
    terrain: Terrain,
    grid: Grid,
    players: Vec<Player>,
    objs: Vec<Obj>,
    entries: Vec<Entry>,
    next_eid: i32,
    out: Vec<(i64, State, Vec<u8>)>,
    destroy: Vec<i32>,
    bulks: Vec<Vec<usize>>,
    columns: Vec<Column>,
    chunks_sent: BTreeSet<(i32, i32)>,
    presses: BTreeMap<i64, u8>,
    jumps: BTreeSet<i64>,
    local_kb: bool,
    players_left: i32,
    kills: i32,
    timeline: Vec<Value>,
    stats: Stats,
    rng: Rng,
    seed: u64,
    kill_tick_e1: i64,
    e5_killed: bool,
    e6_pushed: bool,
    bridge_next: usize,
    bridge_times: Vec<i64>,
    loot_step: usize,
    loot_wait_until: i64,
    loot_phase: u8,
    archer_state: [(i64, i64, bool, u32); 2],
    sounds_used: BTreeSet<String>,
    xp_total: i32,
    tile_packets: usize,
    tiles: ColumnTiles,
    look_track: Vec<Value>,
}

/// Index helpers for the scene roster.
const IDX_V: usize = 0;
const IDX_T: usize = 1;
fn idx_e(n: u8) -> usize {
    1 + n as usize
}

impl<'a> Server<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        plan: &'a ScenePlan,
        terrain: Terrain,
        grid: Grid,
        columns: Vec<Column>,
        bulks: Vec<Vec<usize>>,
        tiles: ColumnTiles,
        seed: u64,
        local_kb: bool,
    ) -> Server<'a> {
        let mut s = Server {
            t: 0,
            plan,
            terrain,
            grid,
            players: Vec::new(),
            objs: Vec::new(),
            entries: Vec::new(),
            next_eid: 2_000,
            out: Vec::new(),
            destroy: Vec::new(),
            bulks,
            columns,
            chunks_sent: BTreeSet::new(),
            presses: BTreeMap::new(),
            jumps: BTreeSet::new(),
            local_kb,
            players_left: 26,
            kills: 0,
            timeline: Vec::new(),
            stats: Stats::default(),
            rng: Rng::new(seed, 0xEEE),
            seed,
            kill_tick_e1: ft(45.0),
            e5_killed: false,
            e6_pushed: false,
            bridge_next: 0,
            bridge_times: Vec::new(),
            loot_step: 0,
            loot_wait_until: 0,
            loot_phase: 0,
            archer_state: [(0, 0, false, 0); 2],
            sounds_used: BTreeSet::new(),
            xp_total: 125,
            tile_packets: 0,
            tiles,
            look_track: Vec::new(),
        };
        s.stats.fight_per_second = vec![0; ((FIGHT_END - FIGHT_START) / 20) as usize];
        s.build_roster();
        s.build_inputs();
        s
    }

    // ---------------------------------------------------------------------------------------
    // Roster, gear, initial state
    // ---------------------------------------------------------------------------------------

    fn new_player(&mut self, idx: usize, role: Role, name: &str, team: usize, eid: i32) -> Player {
        let mut rng = Rng::new(self.seed, 0x1000 + idx as u64);
        let uuid = if role == Role::Viewer {
            Uuid::from_u128(LOCAL_UUID)
        } else {
            Uuid::from_u128(0x0000_0000_0000_4000_8000_5357_0000_0000 + idx as u128)
        };
        let ping = rng.int(18, 160) as i32;
        Player {
            eid,
            uuid,
            name: name.to_owned(),
            team,
            role,
            alive: true,
            scene: false,
            pos: V3::ZERO,
            vel: V3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            on_ground: true,
            airborne_flag: false,
            sprinting: false,
            using: false,
            health: 20.0,
            absorption: 0.0,
            arrows_in: 0,
            effects: Vec::new(),
            equip: [None, None, None, None, None],
            sword: None,
            gear: GearStats::default(),
            watcher: Watcher::new(WKind::Player),
            attrs_dirty: false,
            last_damage: -100,
            last_swing: -100,
            rng,
            input: Input::default(),
            mode: Mode::Idle,
            opponent: None,
            home: V3::ZERO,
            home_radius: 3.0,
            strafe: 1.0,
            strafe_until: 0,
            desired_r: 2.7,
            sprint_resume: 0,
            next_wtap: 0,
            use_until: -1,
            allow_void: false,
            kb_pending: None,
            walked: 0.0,
            next_step: 1.0,
            ping,
            spectator: false,
            held_restore: None,
            cps: 12.0,
            accuracy: 0.5,
        }
    }

    fn build_roster(&mut self) {
        let plan = self.plan;
        // Viewer and teammate (team 0, green).
        let v = self.new_player(IDX_V, Role::Viewer, LOCAL_NAME, 0, LOCAL_EID);
        self.players.push(v);
        let t = self.new_player(IDX_T, Role::Ally, "BenchAlly", 0, 101);
        self.players.push(t);
        let named = [
            "Rival_E1",
            "Rodder_E2",
            "Brawl_E3",
            "Brawl_E4",
            "Brawl_E5",
            "Brawl_E6",
            "Archer_E7",
            "Archer_E8",
            "Bridger_E9",
            "Looter_E10",
            "Rival_E11",
        ];
        let teams = [1usize, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6];
        for (k, name) in named.iter().enumerate() {
            let idx = idx_e(k as u8 + 1);
            let p = self.new_player(idx, Role::E(k as u8 + 1), name, teams[k], 102 + k as i32);
            self.players.push(p);
        }
        // 27 far players: F1..F13 alive (some the teammates of E7/E8/E10), F14..F27 dead.
        let far_teams = [
            4usize, 5, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17,
            17, 18, 18, 19, 19,
        ];
        for k in 0..27 {
            let idx = 13 + k;
            let mut p = self.new_player(
                idx,
                Role::Far(k as u8 + 1),
                NAMES_FAR[k],
                far_teams[k],
                120 + k as i32,
            );
            if k >= 13 {
                p.alive = false;
                p.spectator = true;
            }
            self.players.push(p);
        }

        // ---- scene bodies, gear and starting positions ------------------------------------
        let a = plan.anchor;
        let mut grng = Rng::new(self.seed, 0x6EA6);
        // Viewer: full diamond, Prot II helmet/chest, Sharpness III diamond sword.
        {
            let v = &mut self.players[IDX_V];
            v.scene = true;
            v.pos = a;
            v.yaw = yaw_to(a, plan.e1_start);
            v.equip = [
                Some(enchanted(stack("diamond_sword", 1), &[(ENCH_SHARPNESS, 3)])),
                Some(stack("diamond_boots", 1)),
                Some(stack("diamond_leggings", 1)),
                Some(enchanted(
                    stack("diamond_chestplate", 1),
                    &[(ENCH_PROTECTION, 2)],
                )),
                Some(enchanted(
                    stack("diamond_helmet", 1),
                    &[(ENCH_PROTECTION, 2)],
                )),
            ];
            v.sword = v.equip[0].clone();
            v.gear = GearStats {
                sword_damage: 1.0
                    + item_props(item_by_name("diamond_sword").expect("sword")).attack_damage,
                sharpness: 3,
                armour_points: 20,
                protection_epf: 4,
                power: 0,
            };
            for slot in 1..5 {
                if let Some(s) = v.equip[slot].as_mut() {
                    s.damage = (f64::from(s.damage) + 40.0) as i16;
                }
            }
        }
        let starts: [(u8, V3); 11] = [
            (1, plan.e1_start),
            (2, plan.ally_home.add(V3::new(1.3, 0.0, 0.4))),
            (3, plan.b_centre.add(V3::new(-1.5, 0.0, -1.2))),
            (4, plan.b_centre.add(V3::new(1.4, 0.0, 1.1))),
            (5, plan.b_centre.add(V3::new(-1.2, 0.0, 1.4))),
            (6, plan.b_centre.add(V3::new(1.6, 0.0, -1.0))),
            (7, plan.archer7),
            (8, plan.archer8),
            (9, plan.bridge.start),
            (10, plan.chests[0].access),
            (11, plan.e11_start),
        ];
        // Teammate.
        {
            let (equip, sword, gear) = fighter_gear(&mut grng, 0.8);
            let tp = self.grid_snap(plan.ally_home.add(V3::new(-1.3, 0.0, -0.4)));
            let p = &mut self.players[IDX_T];
            p.scene = true;
            p.pos = tp;
            p.equip = equip;
            p.sword = Some(sword);
            p.gear = gear;
            p.home = plan.ally_home;
            p.opponent = Some(idx_e(2));
            p.mode = Mode::Duel;
            p.effects.push(ActiveEffect {
                id: EFFECT_REGENERATION,
                amplifier: 0,
                until: i64::MAX,
            });
            p.cps = 11.0;
            p.accuracy = 0.75;
        }
        for (n, start) in starts {
            let idx = idx_e(n);
            let (equip, sword, gear) =
                fighter_gear(&mut grng, if n == 1 || n == 11 { 0.7 } else { 0.45 });
            let snapped = self.grid_snap(start);
            let p = &mut self.players[idx];
            p.scene = true;
            p.pos = snapped;
            p.equip = equip;
            p.sword = Some(sword);
            p.gear = gear;
            p.cps = 12.0;
            p.accuracy = 0.75;
            match n {
                1 => {
                    p.opponent = Some(IDX_V);
                    p.mode = Mode::Duel;
                    p.home = a;
                    p.home_radius = 2.5;
                    p.cps = 14.0;
                    p.accuracy = 0.85;
                    p.effects.push(ActiveEffect {
                        id: EFFECT_SPEED,
                        amplifier: 1,
                        until: i64::MAX,
                    });
                }
                2 => {
                    p.opponent = Some(IDX_T);
                    p.mode = Mode::Duel;
                    p.home = plan.ally_home;
                }
                3..=6 => {
                    p.opponent = Some(idx_e(match n {
                        3 => 5,
                        4 => 6,
                        5 => 3,
                        _ => 4,
                    }));
                    p.mode = Mode::Duel;
                    p.home = plan.b_centre;
                    p.home_radius = 3.5;
                }
                7 | 8 => {
                    p.opponent = Some(idx_e(if n == 7 { 8 } else { 7 }));
                    p.mode = Mode::Archer;
                    p.home = snapped;
                    p.home_radius = 1.5;
                    let power = 2 + (n % 2) as i16;
                    p.gear.power = power;
                    p.equip[0] = Some(enchanted(
                        worn("bow", &mut grng, 0.1, 0.4),
                        &[(ENCH_POWER, power)],
                    ));
                }
                9 => {
                    p.mode = Mode::Bridge;
                    p.home = snapped;
                    p.equip[0] = Some(stack("stone", 64));
                }
                10 => {
                    p.mode = Mode::Loot;
                    p.home = snapped;
                    p.equip[0] = Some(worn("iron_sword", &mut grng, 0.1, 0.3));
                    p.equip[4] = None; // gets a helmet from the first chest
                }
                11 => {
                    p.mode = Mode::Idle;
                    p.home = snapped;
                    p.cps = 14.0;
                    p.accuracy = 0.85;
                }
                _ => {}
            }
        }
        // Facing and initial watcher state for every scene body.
        for i in 0..self.players.len() {
            if !self.players[i].scene {
                continue;
            }
            let target = match self.players[i].opponent {
                Some(j) => self.players[j].pos,
                None => self.plan.anchor,
            };
            let p = &mut self.players[i];
            if i != IDX_V {
                p.yaw = yaw_to(p.pos, target);
            }
            p.pitch = 4.0;
            p.watcher.set(6, MetadataValue::Float(p.health));
            let colour = potion_colour(&p.effects);
            p.watcher.set(7, MetadataValue::Int(colour));
            p.watcher.dirty.clear();
            let next_wtap = 20 + p.rng.int(0, 20);
            p.next_wtap = next_wtap;
        }
        // Tracker entries for every scene body, ascending entity id (the viewer's own first).
        let mut order: Vec<usize> = (0..self.players.len())
            .filter(|&i| self.players[i].scene)
            .collect();
        order.sort_by_key(|&i| self.players[i].eid);
        for i in order {
            let p = &self.players[i];
            let e = Entry {
                ent: Ent::P(i),
                eid: p.eid,
                range: RANGE_PLAYER,
                freq: 2,
                send_velocity: false,
                is_arrow: false,
                enc: [fix32(p.pos.x), fix32(p.pos.y), fix32(p.pos.z)],
                enc_rot: [rot256(p.yaw), rot256(p.pitch)],
                last_head: rot256(p.yaw),
                counter: 0,
                since_teleport: 0,
                on_ground: p.on_ground,
                last_check: p.pos,
                first_done: false,
                last_motion: V3::ZERO,
                tracked: false,
                removed: false,
            };
            self.entries.push(e);
        }
    }

    fn grid_snap(&self, p: V3) -> V3 {
        let (x, z) = (p.x.floor() as i32, p.z.floor() as i32);
        match self.grid.get(x, z) {
            Some(c) => V3::new(p.x, c.h, p.z),
            None => find_walkable_near(&self.grid, p, 3.0).unwrap_or(p),
        }
    }

    /// 14 CPS with jitter (0/1 per tick, occasionally 2) whenever the viewer has a
    /// target; a one-tick `jump` every 10-14 ticks during the fight.
    fn build_inputs(&mut self) {
        let mut rng = Rng::new(self.seed, 0x7A9);
        let windows = [(FIGHT_START, self.kill_tick_e1), (ft(48.0), FIGHT_END)];
        for (from, to) in windows {
            for t in from..to {
                let r = rng.f();
                let c = if r < 0.35 {
                    0
                } else if r < 0.95 {
                    1
                } else {
                    2
                };
                if c > 0 {
                    self.presses.insert(t, c);
                }
            }
        }
        // The kill at +45 s is the server's reaction to a press at the tick before.
        self.presses.entry(self.kill_tick_e1 - 1).or_insert(1);
        let mut t = FIGHT_START + rng.int(2, 8);
        while t < FIGHT_END {
            self.jumps.insert(t);
            t += rng.int(10, 14);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Emission
    // ---------------------------------------------------------------------------------------

    fn emit<P: Packet>(&mut self, p: &P) {
        let bytes = encode_packet(p);
        let seg = segment_of(self.t);
        *self.stats.by_seg[seg].entry(P::ID).or_default() += 1;
        self.stats.bytes_by_seg[seg] += bytes.len() as u64;
        if seg == 1 {
            let sec = ((self.t - FIGHT_START) / 20) as usize;
            if let Some(slot) = self.stats.fight_per_second.get_mut(sec) {
                *slot += 1;
            }
        }
        self.out.push((self.t, State::Play, bytes));
    }

    fn event(&mut self, kind: &str, detail: Value) {
        self.timeline
            .push(json!({"tick": self.t, "event": kind, "detail": detail}));
    }

    fn tracked(&self, i: usize) -> bool {
        if i == IDX_V {
            return true;
        }
        self.entries
            .iter()
            .any(|e| e.ent == Ent::P(i) && e.tracked && !e.removed)
    }

    fn obj_tracked(&self, o: usize) -> bool {
        self.entries
            .iter()
            .any(|e| e.ent == Ent::O(o) && e.tracked && !e.removed)
    }

    /// `WorldManager.playSound`/`playSoundToNearExcept`: an S29 to the viewer when within
    /// 16 blocks (more for volume > 1). `except` is the source player (never the viewer here).
    fn sound(&mut self, name: &str, pos: V3, volume: f32, pitch: f32) {
        let range = if volume > 1.0 {
            16.0 * f64::from(volume)
        } else {
            16.0
        };
        if self.players[IDX_V].pos.dist(pos) > range {
            return;
        }
        self.sounds_used.insert(name.to_owned());
        self.emit(&s2c::SoundEffect {
            sound_name: name.into(),
            x_fixed: (pos.x * 8.0) as i32,
            y_fixed: (pos.y * 8.0) as i32,
            z_fixed: (pos.z * 8.0) as i32,
            volume,
            pitch_quantized: (pitch * 63.0).clamp(0.0, 255.0) as u8,
        });
        self.stats.bump("sounds");
    }

    fn chat(&mut self, text: &str) {
        self.emit(&s2c::Chat {
            component: chat_json(text),
            position: 0,
        });
        self.stats.bump("chat_lines");
        self.event("chat", json!(text));
    }

    // ---------------------------------------------------------------------------------------
    // Join (tick 0)
    // ---------------------------------------------------------------------------------------

    fn join(&mut self) {
        self.out.push((
            0,
            State::Login,
            encode_packet(&LoginSuccess {
                uuid: Uuid::from_u128(LOCAL_UUID),
                username: LOCAL_NAME.into(),
            }),
        ));
        self.emit(&s2c::JoinGame {
            entity_id: LOCAL_EID,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 40,
            level_type: "default".into(),
            reduced_debug_info: false,
        });
        self.emit(&s2c::ServerDifficulty {
            difficulty: s2c::ServerDifficultyLevel::Normal,
        });
        let a = self.plan.anchor;
        self.emit(&s2c::SpawnPosition {
            position: BlockPos::new(a.x.floor() as i32, a.y.floor() as i32, a.z.floor() as i32),
        });
        self.emit(&s2c::PlayerAbilities {
            flags: 0,
            fly_speed: 0.05,
            walk_speed: 0.1,
        });
        self.emit(&s2c::HeldItemChange { slot: 0 });
        self.scoreboard_init();
        // Tab list: 40 rows, one real signed textures payload shared by all.
        let props = vec![s2c::ProfileProperty {
            name: "textures".into(),
            value: SIGNED_TEXTURES_VALUE.into(),
            signature: Some(SIGNED_TEXTURES_SIGNATURE.into()),
        }];
        let rows: Vec<s2c::AddPlayerEntry> = self
            .players
            .iter()
            .map(|p| s2c::AddPlayerEntry {
                uuid: p.uuid,
                username: p.name.clone(),
                properties: props.clone(),
                game_mode: if p.spectator {
                    s2c::GameMode::Spectator
                } else {
                    s2c::GameMode::Survival
                },
                ping: p.ping,
                display_name: None,
            })
            .collect();
        self.emit(&s2c::PlayerListItem::AddPlayer(rows));
        self.emit(&s2c::PlayerListHeaderFooter {
            header: chat_json(HEADER_TEXT),
            footer: chat_json(FOOTER_TEXT),
        });
        let v = &self.players[IDX_V];
        let (vpos, vyaw) = (v.pos, v.yaw);
        self.emit(&s2c::PlayerPositionLook {
            x: vpos.x,
            y: vpos.y,
            z: vpos.z,
            yaw: vyaw,
            pitch: 6.0,
            flags: 0,
        });
        self.players[IDX_V].pitch = 6.0;
        self.emit(&s2c::WorldBorder::Initialize {
            x: 0.0,
            z: 0.0,
            old_diameter: 60_000_000.0,
            new_diameter: 60_000_000.0,
            speed_millis: 0,
            world_size: 29_999_984,
            warning_blocks: 5,
            warning_time: 15,
        });
        self.time_update();
        // Inventory (window 0): armour 5-8, main 9-35, hotbar 36-44.
        let v = &self.players[IDX_V];
        let mut items: Vec<Option<ItemStack>> = vec![None; 45];
        items[5] = v.equip[4].clone();
        items[6] = v.equip[3].clone();
        items[7] = v.equip[2].clone();
        items[8] = v.equip[1].clone();
        items[36] = v.equip[0].clone();
        items[37] = Some(enchanted(stack("bow", 1), &[(ENCH_POWER, 1)]));
        items[38] = Some(stack("fishing_rod", 1));
        items[39] = Some(stack("ender_pearl", 16));
        items[40] = Some(stack("stone", 64));
        items[41] = Some(stack("golden_apple", 2));
        items[42] = Some(stack("water_bucket", 1));
        items[43] = Some(stack("snowball", 16));
        items[9] = Some(stack("arrow", 12));
        items[10] = Some(stack("planks", 32));
        self.emit(&s2c::WindowItems {
            window_id: 0,
            items,
        });
        self.emit(&s2c::UpdateHealth {
            health: 20.0,
            food_level: 20,
            saturation_level: 5.0,
        });
        self.emit_xp();
        self.event("join", json!({"anchor": self.plan.anchor.json()}));
    }

    fn emit_xp(&mut self) {
        // Vanilla level curve (1.8): xpBarCap = 7 + 2*level for levels 0-15.
        let mut level = 0;
        let mut rest = self.xp_total;
        while rest >= 7 + 2 * level {
            rest -= 7 + 2 * level;
            level += 1;
        }
        self.emit(&s2c::SetExperience {
            experience_bar: rest as f32 / (7 + 2 * level) as f32,
            level,
            total_experience: self.xp_total,
        });
    }

    fn time_update(&mut self) {
        self.emit(&s2c::TimeUpdate {
            total_world_time: 4_812_000 + self.t,
            world_time: -6_000,
        });
    }

    // ---------------------------------------------------------------------------------------
    // HUD: teams, sidebar, tab
    // ---------------------------------------------------------------------------------------

    fn team_info(display: &str, prefix: &str, suffix: &str, colour: i8) -> s2c::TeamInfo {
        s2c::TeamInfo {
            display_name: display.into(),
            prefix: prefix.into(),
            suffix: suffix.into(),
            friendly_flags: 3,
            name_tag_visibility: "always".into(),
            color: colour,
        }
    }

    fn timer_text(&self) -> String {
        let remaining = (165 - self.t / 20).max(0);
        format!("§aRefill {}:{:02}", remaining / 60, remaining % 60)
    }

    fn scoreboard_init(&mut self) {
        // One scoreboard team per game team: the viewer's green, every enemy team red.
        let mut members: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for p in &self.players {
            members.entry(p.team).or_default().push(p.name.clone());
        }
        for (team, players) in members {
            let (prefix, colour) = if team == 0 { ("§a", 10) } else { ("§c", 12) };
            self.emit(&s2c::Teams {
                team_name: if team == 0 {
                    "you".to_owned()
                } else {
                    format!("sw_t{team:02}")
                },
                action: s2c::TeamsAction::Create {
                    info: Self::team_info(&format!("Team {}", team + 1), prefix, "", colour),
                    players,
                },
            });
        }
        // Sidebar line teams (dynamic text lives in prefixes/suffixes).
        let timer = self.timer_text();
        for (name, entry, prefix, suffix) in [
            ("sw_l_timer", "§2§r", timer.as_str(), ""),
            ("sw_l_left", "§4§a", "Players left: ", "26"),
            ("sw_l_kills", "§6§a", "Kills: ", "0"),
        ] {
            self.emit(&s2c::Teams {
                team_name: name.into(),
                action: s2c::TeamsAction::Create {
                    info: Self::team_info(name, prefix, suffix, -1),
                    players: vec![entry.into()],
                },
            });
        }
        self.emit(&s2c::ScoreboardObjective {
            objective_name: "sw_sidebar".into(),
            mode: s2c::ScoreboardObjectiveMode::Create {
                display_name: "§e§lSKYWARS".into(),
                render_type: s2c::ScoreboardRenderType::Integer,
            },
        });
        self.emit(&s2c::DisplayScoreboard {
            position: 1,
            objective_name: "sw_sidebar".into(),
        });
        let lines = [
            "§711/05/26 §8m27Q",
            "§1",
            "Next Event:",
            "§2§r",
            "§3",
            "§4§a",
            "§5",
            "§6§a",
            "§7",
            "Map: §aDwarf Fortress",
            "Mode: §6Mega Doubles",
            "§8",
            "§ewww.example.net",
        ];
        for (k, line) in lines.iter().enumerate() {
            self.emit(&s2c::UpdateScore {
                player_name: (*line).into(),
                objective_name: "sw_sidebar".into(),
                action: s2c::UpdateScoreAction::Change {
                    value: (lines.len() - k) as i32,
                },
            });
        }
    }

    fn sidebar_line(&mut self, team: &str, prefix: &str, suffix: &str) {
        self.emit(&s2c::Teams {
            team_name: team.into(),
            action: s2c::TeamsAction::UpdateInfo {
                info: Self::team_info(team, prefix, suffix, -1),
            },
        });
    }

    fn hud_tick(&mut self) {
        if self.t > 0 && self.t % 20 == 0 {
            let text = self.timer_text();
            self.sidebar_line("sw_l_timer", &text, "");
        }
        if self.t > 0 && self.t % 600 == 0 {
            let rows: Vec<s2c::UpdateLatencyEntry> = (0..self.players.len())
                .map(|i| {
                    let p = &mut self.players[i];
                    let ping = (p.ping + p.rng.int(-12, 12) as i32).max(8);
                    p.ping = ping;
                    s2c::UpdateLatencyEntry { uuid: p.uuid, ping }
                })
                .collect();
            self.emit(&s2c::PlayerListItem::UpdateLatency(rows));
        }
    }

    fn players_left_changed(&mut self) {
        self.players_left -= 1;
        let left = self.players_left.to_string();
        self.sidebar_line("sw_l_left", "Players left: ", &left);
    }

    fn make_spectator(&mut self, i: usize) {
        let uuid = self.players[i].uuid;
        self.emit(&s2c::PlayerListItem::UpdateGameMode(vec![
            s2c::UpdateGameModeEntry {
                uuid,
                game_mode: s2c::GameMode::Spectator,
            },
        ]));
    }

    // ---------------------------------------------------------------------------------------
    // Chunks (ticks 1..): one bulk per tick, then the tracker spawns entities in those chunks
    // ---------------------------------------------------------------------------------------

    fn send_chunks(&mut self) {
        if self.t < 1 {
            return;
        }
        let k = (self.t - 1) as usize;
        let Some(batch) = self.bulks.get(k).cloned() else {
            return;
        };
        if batch.len() == 1 {
            let c = self.columns[batch[0]].clone();
            self.emit(&chunk::ChunkData {
                chunk_x: c.x,
                chunk_z: c.z,
                ground_up_continuous: true,
                primary_bit_mask: c.mask,
                data: c.data,
            });
        } else {
            let columns: Vec<ChunkBulkColumn> = batch
                .iter()
                .map(|&i| {
                    let c = &self.columns[i];
                    ChunkBulkColumn {
                        chunk_x: c.x,
                        chunk_z: c.z,
                        primary_bit_mask: c.mask,
                        data: c.data.clone(),
                    }
                })
                .collect();
            let bulk = MapChunkBulk {
                has_sky_light: true,
                columns,
            };
            let size = encode_packet(&bulk).len();
            assert!(
                size < 2_097_151,
                "bulk packet of {size} bytes exceeds the frame limit"
            );
            self.emit(&bulk);
        }
        self.stats.bulks += 1;
        self.stats.last_chunk_tick = self.t;
        let mut sent = Vec::new();
        for &i in &batch {
            let c = &self.columns[i];
            self.stats.columns_sent += 1;
            self.stats.sections_sent += c.mask.count_ones() as usize;
            sent.push((c.x, c.z));
        }
        for pos in &sent {
            self.chunks_sent.insert(*pos);
        }
        let tiles: Vec<TilePacket> = self
            .tiles
            .iter()
            .filter(|(column, _)| sent.contains(column))
            .map(|(_, p)| p.clone())
            .collect();
        for packet in tiles {
            match packet {
                TilePacket::Sign(p) => self.emit(&p),
                TilePacket::Data(p) => self.emit(&p),
            }
            self.tile_packets += 1;
        }
        // EntityTracker.func_85172_a: entities in the newly sent chunks become visible.
        for k in 0..self.entries.len() {
            let pos = self.entity_pos(self.entries[k].ent);
            let ch = ((pos.x.floor() as i32) >> 4, (pos.z.floor() as i32) >> 4);
            if sent.contains(&ch) {
                self.viewer_check(k);
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Tracker (EntityTrackerEntry port)
    // ---------------------------------------------------------------------------------------

    fn entity_pos(&self, ent: Ent) -> V3 {
        match ent {
            Ent::P(i) => self.players[i].pos,
            Ent::O(o) => self.objs[o].pos,
        }
    }

    /// `EntityTrackerEntry.updatePlayerEntity` for the one player that matters: the viewer.
    fn viewer_check(&mut self, k: usize) {
        let e = self.entries[k].clone();
        if e.removed || e.ent == Ent::P(IDX_V) {
            return;
        }
        let vpos = self.players[IDX_V].pos;
        // Java: `encodedPosX / 32` is integer division (truncation), then widened.
        let d0 = vpos.x - f64::from(e.enc[0] / 32);
        let d1 = vpos.z - f64::from(e.enc[2] / 32);
        let visible = match e.ent {
            Ent::P(i) => self.players[i].alive && !self.players[i].spectator,
            Ent::O(o) => self.objs[o].alive,
        };
        let in_range =
            d0 >= -e.range && d0 <= e.range && d1 >= -e.range && d1 <= e.range && visible;
        if in_range {
            if !e.tracked {
                let pos = self.entity_pos(e.ent);
                let ch = ((pos.x.floor() as i32) >> 4, (pos.z.floor() as i32) >> 4);
                if self.chunks_sent.contains(&ch) {
                    self.entries[k].tracked = true;
                    self.spawn_packets(k);
                }
            }
        } else if e.tracked {
            self.entries[k].tracked = false;
            self.destroy.push(e.eid);
        }
    }

    fn spawn_packets(&mut self, k: usize) {
        let e = self.entries[k].clone();
        match e.ent {
            Ent::P(i) => {
                let p = self.players[i].clone();
                let held = p.equip[0].as_ref().map_or(0, |s| s.item_id);
                self.emit(&s2c::SpawnPlayer {
                    entity_id: p.eid,
                    player_uuid: p.uuid,
                    x: fix32(p.pos.x),
                    y: fix32(p.pos.y),
                    z: fix32(p.pos.z),
                    yaw: Angle::from_degrees(p.yaw),
                    pitch: Angle::from_degrees(p.pitch),
                    current_item: held,
                    metadata: p.watcher.all(),
                });
                self.emit(&s2c::EntityMetadata {
                    entity_id: p.eid,
                    metadata: p.watcher.all(),
                });
                let speed = p
                    .effects
                    .iter()
                    .find(|e| e.id == EFFECT_SPEED)
                    .map(|e| e.amplifier);
                self.emit(&s2c::EntityProperties {
                    entity_id: p.eid,
                    properties: vec![
                        s2c::AttributePropertySnapshot {
                            key: "generic.maxHealth".into(),
                            base_value: 20.0,
                            modifiers: vec![],
                        },
                        movement_speed_snapshot(p.sprinting, speed),
                    ],
                });
                for slot in 0..5 {
                    if let Some(s) = &p.equip[slot] {
                        self.emit(&s2c::EntityEquipment {
                            entity_id: p.eid,
                            equipment_slot: slot as i16,
                            item: Some(s.clone()),
                        });
                    }
                }
                // CraftBukkit: "Fix for nonsensical head yaw" broadcasts S19 at spawn.
                let head = rot256(p.yaw);
                self.entries[k].last_head = head;
                self.emit(&s2c::EntityHeadLook {
                    entity_id: p.eid,
                    yaw: Angle(head as u8),
                });
                for eff in &p.effects {
                    self.emit(&s2c::EntityEffect {
                        entity_id: p.eid,
                        effect_id: eff.id as i8,
                        amplifier: eff.amplifier as i8,
                        duration: (eff.until - self.t).clamp(1, 32_767) as i32,
                        show_particles: true,
                    });
                }
                self.stats.bump("player_spawns");
            }
            Ent::O(o) => {
                let ob = self.objs[o].clone();
                let vel = (vel8000(ob.vel.x), vel8000(ob.vel.y), vel8000(ob.vel.z));
                match ob.kind {
                    ObjKind::XpOrb(value) => self.emit(&s2c::SpawnExperienceOrb {
                        entity_id: ob.eid,
                        x: fix32(ob.pos.x),
                        y: fix32(ob.pos.y),
                        z: fix32(ob.pos.z),
                        xp_value: value,
                    }),
                    _ => {
                        let (ty, data) = match ob.kind {
                            ObjKind::Item => (2, 1),
                            ObjKind::Arrow { shooter, .. } => (60, self.players[shooter].eid),
                            ObjKind::Snowball => (61, 0),
                            ObjKind::Egg => (62, 0),
                            ObjKind::Pearl => (65, 0),
                            ObjKind::Potion(d) => (73, d),
                            ObjKind::Hook => (90, self.players[ob.owner.unwrap_or(0)].eid),
                            ObjKind::Firework => (76, 0),
                            ObjKind::XpOrb(_) => unreachable!(),
                        };
                        self.emit(&s2c::SpawnObject {
                            entity_id: ob.eid,
                            object_type: ty,
                            x: fix32(ob.pos.x),
                            y: fix32(ob.pos.y),
                            z: fix32(ob.pos.z),
                            pitch: Angle(rot256(ob.pitch) as u8),
                            yaw: Angle(rot256(ob.yaw) as u8),
                            data,
                            velocity: if data > 0 { Some(vel) } else { None },
                        });
                    }
                }
                self.emit(&s2c::EntityMetadata {
                    entity_id: ob.eid,
                    metadata: ob.watcher.all(),
                });
                if e.send_velocity {
                    self.emit(&s2c::EntityVelocity {
                        entity_id: ob.eid,
                        motion_x: vel.0,
                        motion_y: vel.1,
                        motion_z: vel.2,
                    });
                }
                self.entries[k].last_motion = ob.vel;
                self.stats.bump("object_spawns");
            }
        }
    }

    fn add_obj(&mut self, mut obj: Obj) -> usize {
        obj.eid = self.next_eid;
        self.next_eid += 1;
        obj.born = self.t;
        let (range, freq, send_velocity) = match obj.kind {
            ObjKind::Item | ObjKind::XpOrb(_) => (RANGE_MISC, 20, true),
            ObjKind::Arrow { .. } => (RANGE_OTHER, 20, false),
            ObjKind::Hook => (RANGE_OTHER, 5, true),
            _ => (RANGE_OTHER, 10, true),
        };
        let o = self.objs.len();
        let e = Entry {
            ent: Ent::O(o),
            eid: obj.eid,
            range,
            freq,
            send_velocity,
            is_arrow: matches!(obj.kind, ObjKind::Arrow { .. }),
            enc: [fix32(obj.pos.x), fix32(obj.pos.y), fix32(obj.pos.z)],
            enc_rot: [rot256(obj.yaw), rot256(obj.pitch)],
            last_head: rot256(obj.yaw),
            counter: 0,
            since_teleport: 0,
            on_ground: obj.on_ground,
            last_check: obj.pos,
            first_done: true,
            last_motion: obj.vel,
            tracked: false,
            removed: false,
        };
        self.objs.push(obj);
        self.entries.push(e);
        let k = self.entries.len() - 1;
        // EntityTracker.addEntityToTracker -> updatePlayerEntities: spawn right away.
        self.viewer_check(k);
        o
    }

    fn remove_obj(&mut self, o: usize) {
        if !self.objs[o].alive {
            return;
        }
        self.objs[o].alive = false;
        if let Some(k) = self.entries.iter().position(|e| e.ent == Ent::O(o)) {
            if self.entries[k].tracked {
                self.destroy.push(self.entries[k].eid);
            }
            self.entries[k].tracked = false;
            self.entries[k].removed = true;
        }
    }

    fn remove_player_body(&mut self, i: usize) {
        if let Some(k) = self.entries.iter().position(|e| e.ent == Ent::P(i)) {
            if self.entries[k].tracked {
                self.destroy.push(self.entries[k].eid);
            }
            self.entries[k].tracked = false;
            self.entries[k].removed = true;
        }
    }

    fn tracker_tick(&mut self) {
        let mut viewer_moved = false;
        for k in 0..self.entries.len() {
            if self.entries[k].removed {
                continue;
            }
            if self.update_entry(k) {
                viewer_moved = true;
            }
        }
        if viewer_moved {
            for k in 0..self.entries.len() {
                self.viewer_check(k);
            }
        }
    }

    /// `EntityTrackerEntry.updatePlayerList`. Returns true when the viewer's own entry moved
    /// more than 4 blocks (vanilla then re-checks every entry for that player).
    fn update_entry(&mut self, k: usize) -> bool {
        let ent = self.entries[k].ent;
        let is_self = ent == Ent::P(IDX_V);
        let (pos, vel, yaw, pitch, on_ground, airborne, dirty) = match ent {
            Ent::P(i) => {
                let p = &self.players[i];
                (
                    p.pos,
                    p.vel,
                    p.yaw,
                    p.pitch,
                    p.on_ground,
                    p.airborne_flag,
                    !p.watcher.dirty.is_empty(),
                )
            }
            Ent::O(o) => {
                let ob = &self.objs[o];
                (
                    ob.pos,
                    ob.vel,
                    ob.yaw,
                    ob.pitch,
                    ob.on_ground,
                    false,
                    !ob.watcher.dirty.is_empty(),
                )
            }
        };
        let mut viewer_moved = false;
        {
            let e = &mut self.entries[k];
            if !e.first_done || pos.sub(e.last_check).len().powi(2) > 16.0 {
                e.last_check = pos;
                e.first_done = true;
                if is_self {
                    viewer_moved = true;
                } else {
                    self.viewer_check(k);
                }
            }
        }
        let e = self.entries[k].clone();
        if e.counter % e.freq == 0 || airborne || dirty {
            let mut e = e;
            e.since_teleport += 1;
            let kx = fix32(pos.x);
            let ky = fix32(pos.y);
            let kz = fix32(pos.z);
            let l1 = rot256(yaw);
            let i2 = rot256(pitch);
            let (j2, k2, i) = (kx - e.enc[0], ky - e.enc[1], kz - e.enc[2]);
            let flag = j2.abs() >= 4 || k2.abs() >= 4 || i.abs() >= 4 || e.counter % 60 == 0;
            let flag1 = (l1 - e.enc_rot[0]).abs() >= 4 || (i2 - e.enc_rot[1]).abs() >= 4;
            let send = e.tracked && !is_self;
            if e.counter > 0 || e.is_arrow {
                let in_rel = (-128..128).contains(&j2)
                    && (-128..128).contains(&k2)
                    && (-128..128).contains(&i)
                    && e.since_teleport <= 400
                    && e.on_ground == on_ground;
                if in_rel {
                    if (!flag || !flag1) && !e.is_arrow {
                        if flag {
                            if send {
                                self.emit(&s2c::EntityRelMove {
                                    entity_id: e.eid,
                                    delta_x: j2 as i8,
                                    delta_y: k2 as i8,
                                    delta_z: i as i8,
                                    on_ground,
                                });
                            }
                        } else if flag1 && send {
                            self.emit(&s2c::EntityLook {
                                entity_id: e.eid,
                                yaw: Angle(l1 as u8),
                                pitch: Angle(i2 as u8),
                                on_ground,
                            });
                        }
                    } else if send {
                        self.emit(&s2c::EntityLookAndRelMove {
                            entity_id: e.eid,
                            delta_x: j2 as i8,
                            delta_y: k2 as i8,
                            delta_z: i as i8,
                            yaw: Angle(l1 as u8),
                            pitch: Angle(i2 as u8),
                            on_ground,
                        });
                    }
                } else {
                    e.on_ground = on_ground;
                    e.since_teleport = 0;
                    if send {
                        self.emit(&s2c::EntityTeleport {
                            entity_id: e.eid,
                            x: kx,
                            y: ky,
                            z: kz,
                            yaw: Angle(l1 as u8),
                            pitch: Angle(i2 as u8),
                            on_ground,
                        });
                    }
                }
            }
            if e.send_velocity {
                let d = vel.sub(e.last_motion);
                let d4 = d.x * d.x + d.y * d.y + d.z * d.z;
                if d4 > 0.02 * 0.02 || (d4 > 0.0 && vel.x == 0.0 && vel.y == 0.0 && vel.z == 0.0) {
                    e.last_motion = vel;
                    if send {
                        self.emit(&s2c::EntityVelocity {
                            entity_id: e.eid,
                            motion_x: vel8000(vel.x),
                            motion_y: vel8000(vel.y),
                            motion_z: vel8000(vel.z),
                        });
                    }
                }
            }
            // sendMetadataToAllAssociatedPlayers sends to the trackers plus the player itself. The
            // self-echo is deliberately NOT emitted: on a real server it only repeats what the
            // client already sent (C0B sprint/sneak edges), but here the viewer's state is a model,
            // while the real client's sprint state depends on runtime input and on the vanilla
            // sprint reset after each hit. Echoing the model would overwrite the client's
            // watched flags and re-add the sprint speed modifier (S20) the client just removed.
            let to_viewer = e.tracked && !is_self;
            match ent {
                Ent::P(pi) => {
                    let changed = self.players[pi].watcher.take_changed();
                    if !changed.is_empty() && to_viewer {
                        self.emit(&s2c::EntityMetadata {
                            entity_id: e.eid,
                            metadata: changed,
                        });
                    }
                    if self.players[pi].attrs_dirty {
                        self.players[pi].attrs_dirty = false;
                        if to_viewer {
                            let p = &self.players[pi];
                            let speed = p
                                .effects
                                .iter()
                                .find(|x| x.id == EFFECT_SPEED)
                                .map(|x| x.amplifier);
                            let snap = movement_speed_snapshot(p.sprinting, speed);
                            self.emit(&s2c::EntityProperties {
                                entity_id: e.eid,
                                properties: vec![snap],
                            });
                        }
                    }
                }
                Ent::O(oi) => {
                    let changed = self.objs[oi].watcher.take_changed();
                    if !changed.is_empty() && e.tracked {
                        self.emit(&s2c::EntityMetadata {
                            entity_id: e.eid,
                            metadata: changed,
                        });
                    }
                }
            }
            if flag {
                e.enc = [kx, ky, kz];
            }
            if flag1 {
                e.enc_rot = [l1, i2];
            }
            // Head look (inside the update block, as in vanilla).
            if let Ent::P(_) = ent {
                let l = rot256(yaw);
                if (l - e.last_head).abs() >= 4 {
                    if send {
                        self.emit(&s2c::EntityHeadLook {
                            entity_id: e.eid,
                            yaw: Angle(l as u8),
                        });
                    }
                    e.last_head = l;
                }
            }
            if let Ent::P(pi) = ent {
                self.players[pi].airborne_flag = false;
            }
            e.counter += 1;
            self.entries[k] = e;
        } else {
            self.entries[k].counter += 1;
        }
        // velocityChanged: non-melee knockback, to trackers and the entity itself.
        let vc = match ent {
            Ent::O(o) => std::mem::take(&mut self.objs[o].velocity_changed),
            Ent::P(_) => false,
        };
        if vc && self.entries[k].tracked {
            let ev = self.entries[k].eid;
            self.emit(&s2c::EntityVelocity {
                entity_id: ev,
                motion_x: vel8000(vel.x),
                motion_y: vel8000(vel.y),
                motion_z: vel8000(vel.z),
            });
        }
        viewer_moved
    }

    fn flush_destroy(&mut self) {
        if self.destroy.is_empty() {
            return;
        }
        let ids = std::mem::take(&mut self.destroy);
        self.emit(&s2c::DestroyEntities { entity_ids: ids });
    }

    // ---------------------------------------------------------------------------------------
    // Player state mutators (keep watcher/attributes in step)
    // ---------------------------------------------------------------------------------------

    fn set_sprint(&mut self, i: usize, on: bool) {
        let p = &mut self.players[i];
        if p.sprinting != on {
            p.sprinting = on;
            p.watcher.set_flag(3, on);
            p.attrs_dirty = true;
            self.stats.bump("sprint_toggles");
        }
    }

    fn set_using(&mut self, i: usize, on: bool) {
        let p = &mut self.players[i];
        if p.using != on {
            p.using = on;
            p.watcher.set_flag(4, on);
        }
    }

    fn set_health(&mut self, i: usize, health: f32) {
        let p = &mut self.players[i];
        let h = health.clamp(0.0, 20.0);
        if (p.health - h).abs() < f32::EPSILON {
            return;
        }
        p.health = h;
        p.watcher.set(6, MetadataValue::Float(h));
        if i == IDX_V {
            self.emit(&s2c::UpdateHealth {
                health: h,
                food_level: 20,
                saturation_level: 5.0,
            });
        }
    }

    fn refresh_potion_colour(&mut self, i: usize) {
        let t = self.t;
        let p = &mut self.players[i];
        p.effects.retain(|e| e.until > t);
        let c = potion_colour(&p.effects);
        p.watcher.set(7, MetadataValue::Int(c));
    }

    fn set_equipment(&mut self, i: usize, slot: usize, item: Option<ItemStack>) {
        self.players[i].equip[slot] = item.clone();
        if i == IDX_V {
            return;
        }
        if self.tracked(i) {
            let eid = self.players[i].eid;
            self.emit(&s2c::EntityEquipment {
                entity_id: eid,
                equipment_slot: slot as i16,
                item,
            });
        }
    }

    fn hold(&mut self, i: usize, item: ItemStack, restore_after: i64) {
        let sword = self.players[i].sword.clone();
        self.set_equipment(i, 0, Some(item));
        if let Some(sw) = sword {
            self.players[i].held_restore = Some((self.t + restore_after, sw));
        }
    }

    /// `EntityLivingBase.swingItem` on the server: S0B 0 to trackers only when a new swing may
    /// start (no swing in progress, or `swingProgressInt >= 3` of the 6-tick swing).
    fn swing(&mut self, i: usize) {
        let t = self.t;
        let p = &mut self.players[i];
        if t - p.last_swing < 4 {
            return;
        }
        p.last_swing = t;
        if i != IDX_V && self.tracked(i) {
            let eid = self.players[i].eid;
            self.emit(&s2c::Animation {
                entity_id: eid,
                animation_type: 0,
            });
            self.stats.bump("swing_broadcasts");
        }
    }

    fn falling(&self, i: usize) -> bool {
        let p = &self.players[i];
        !p.on_ground && p.vel.y < 0.0
    }

    // ---------------------------------------------------------------------------------------
    // Melee
    // ---------------------------------------------------------------------------------------

    /// `EntityPlayer.attackTargetEntityWithCurrentItem` for one damaging hit, with every packet
    /// the viewer would receive. `kill` makes it the scripted killing blow.
    fn melee_hit(&mut self, a: usize, v: usize, kill: bool, push_dir: Option<V3>) {
        let t = self.t;
        let crit = self.falling(a);
        let a_pos = self.players[a].pos;
        let a_yaw = self.players[a].yaw;
        let a_sprint = self.players[a].sprinting;
        let sharp = self.players[a].gear.sharpness;
        let base = self.players[a].gear.sword_damage + 1.25 * f32::from(sharp);
        // Swing first: the client sends C0A before C02.
        self.swing(a);
        // ---- victim -----------------------------------------------------------------------
        let v_eid = self.players[v].eid;
        let mut dmg = base * if crit { 1.5 } else { 1.0 };
        let armour = self.players[v].gear.armour_points.min(20);
        dmg *= (25 - armour) as f32 / 25.0;
        let epf = self.players[v].gear.protection_epf.min(20);
        dmg *= 1.0 - epf as f32 * 0.03;
        self.players[v].last_damage = t;
        if v == IDX_V || self.tracked(v) {
            self.emit(&s2c::EntityStatus {
                entity_id: v_eid,
                opcode: 2,
            });
        }
        if v != IDX_V {
            let p = self.players[v].pos;
            let pitch = self.players[v].rng.range(0.9, 1.1) as f32;
            self.sound("game.player.hurt", p, 1.0, pitch);
        }
        // Knockback (EntityLivingBase.knockBack + the sprint bonus).
        let vp = self.players[v].pos;
        let mut d = a_pos.sub(vp);
        if d.hlen() < 1e-4 {
            d = V3::new(0.01, 0.0, 0.01);
        }
        let f = d.hlen();
        let mut kb = self.players[v].vel.scale(0.5);
        let away = match push_dir {
            Some(dir) => dir.scale(-1.0),
            None => d.scale(1.0 / f),
        };
        kb.x -= away.x * KB_BASE_H;
        kb.z -= away.z * KB_BASE_H;
        kb.y += KB_BASE_V;
        if kb.y > KB_BASE_V {
            kb.y = KB_BASE_V;
        }
        if a_sprint {
            let fwd = match push_dir {
                Some(dir) => dir,
                None => dir_of_yaw(a_yaw),
            };
            kb.x += fwd.x * KB_SPRINT_H;
            kb.z += fwd.z * KB_SPRINT_H;
            kb.y += KB_SPRINT_V;
        }
        // Keep fights on their ground: when the victim has drifted out of its home zone, half of
        // the push points home (server knockback is often custom; the direction is a scene choice,
        // recorded in the recipe). This bounds combo drift.
        if push_dir.is_none() {
            let (home, home_r) = if v == IDX_V {
                (self.plan.anchor, 3.0)
            } else {
                (self.players[v].home, self.players[v].home_radius + 1.0)
            };
            let off = home.sub(vp);
            if off.hlen() > home_r {
                let h = V3::new(kb.x, 0.0, kb.z);
                let mag = h.hlen();
                let blend = h
                    .hnorm()
                    .scale(0.5)
                    .add(off.hnorm().scale(0.5))
                    .hnorm()
                    .scale(mag);
                kb.x = blend.x;
                kb.z = blend.z;
            }
        }
        if v == IDX_V {
            if self.local_kb {
                self.emit(&s2c::EntityVelocity {
                    entity_id: LOCAL_EID,
                    motion_x: vel8000(kb.x),
                    motion_y: vel8000(kb.y),
                    motion_z: vel8000(kb.z),
                });
                self.players[IDX_V].kb_pending = Some(kb);
            }
        } else {
            self.players[v].kb_pending = Some(kb);
            self.players[v].airborne_flag = true;
        }
        // Crit and magic-crit echoes go to everyone tracking the attacker, and the attacker.
        if a == IDX_V || self.tracked(a) {
            if crit {
                self.emit(&s2c::Animation {
                    entity_id: v_eid,
                    animation_type: 4,
                });
                self.stats.bump("crit_echoes");
            }
            if sharp > 0 {
                self.emit(&s2c::Animation {
                    entity_id: v_eid,
                    animation_type: 5,
                });
                self.stats.bump("magic_crit_echoes");
            }
        }
        // Armour durability: every piece loses 1 (S04 to trackers, S2F to the victim).
        for slot in 1..5usize {
            let Some(mut s) = self.players[v].equip[slot].clone() else {
                continue;
            };
            s.damage += 1;
            if v == IDX_V {
                self.players[IDX_V].equip[slot] = Some(s.clone());
                self.emit(&s2c::SetSlot {
                    window_id: 0,
                    slot: (9 - slot) as i16,
                    item: Some(s),
                });
            } else {
                self.set_equipment(v, slot, Some(s));
            }
        }
        // Health (cosmetic model: never below 1 except at the scripted kill).
        let (h, abs) = (self.players[v].health, self.players[v].absorption);
        let mut left = dmg;
        if abs > 0.0 {
            let used = abs.min(left);
            left -= used;
            self.players[v].absorption = abs - used;
            let na = self.players[v].absorption;
            self.players[v].watcher.set(17, MetadataValue::Float(na));
        }
        let nh = if kill { 0.0 } else { (h - left).max(1.0) };
        self.set_health(v, nh);
        // ---- attacker ---------------------------------------------------------------------
        if let Some(mut sw) = self.players[a].sword.clone() {
            sw.damage += 1;
            self.players[a].sword = Some(sw.clone());
            if a == IDX_V {
                self.players[IDX_V].equip[0] = Some(sw.clone());
                self.emit(&s2c::SetSlot {
                    window_id: 0,
                    slot: 36,
                    item: Some(sw),
                });
            } else if self.players[a].equip[0].as_ref().map(|s| s.item_id) == Some(sw.item_id) {
                self.set_equipment(a, 0, Some(sw));
            }
        }
        if a_sprint {
            self.set_sprint(a, false);
            let p = &mut self.players[a];
            p.sprint_resume = t + p.rng.int(2, 4);
            p.vel.x *= 0.6;
            p.vel.z *= 0.6;
        }
        self.stats.bump(if v == IDX_V {
            "hits_on_viewer"
        } else if a == IDX_V {
            "viewer_hits"
        } else {
            "remote_hits"
        });
        if kill {
            self.kill_player(v, Some(a), false);
        }
    }

    fn hurt_ok(&self, v: usize) -> bool {
        self.t - self.players[v].last_damage >= 10
    }

    // ---------------------------------------------------------------------------------------
    // Deaths, drops, kill effects
    // ---------------------------------------------------------------------------------------

    fn kill_player(&mut self, v: usize, killer: Option<usize>, void: bool) {
        let t = self.t;
        let v_eid = self.players[v].eid;
        let pos = self.players[v].pos;
        let tracked = self.tracked(v);
        if tracked {
            self.emit(&s2c::EntityStatus {
                entity_id: v_eid,
                opcode: 3,
            });
        }
        self.set_health(v, 0.0);
        let pitch = self.players[v].rng.range(0.9, 1.1) as f32;
        self.sound("game.player.die", pos, 1.0, pitch);
        self.players[v].alive = false;
        self.players[v].mode = if void { Mode::Falling } else { Mode::Dead };
        let victim = self.players[v].name.clone();
        let killer_name = killer
            .map(|k| self.players[k].name.clone())
            .unwrap_or_default();
        let killer_colour = if killer == Some(IDX_V) || killer == Some(IDX_T) {
            "§a"
        } else {
            "§c"
        };
        if void {
            self.chat(&format!(
                "§c{victim} §ewas thrown into the void by {killer_colour}{killer_name}§e."
            ));
        } else {
            self.chat(&format!(
                "§c{victim} §ewas killed by {killer_colour}{killer_name}§e."
            ));
        }
        self.players_left_changed();
        self.make_spectator(v);
        if !void {
            // ~20 dropped stacks (EntityPlayer.dropItem with dropAround) and 3 XP orbs.
            let drops = self.death_drops(v);
            let base = pos.add(V3::new(0.0, 1.32, 0.0));
            for s in drops {
                let (fspeed, ang, yaw) = {
                    let r = &mut self.players[v].rng;
                    (
                        r.range(0.0, 0.5),
                        r.range(0.0, std::f64::consts::TAU),
                        r.range(0.0, 360.0) as f32,
                    )
                };
                let mut w = Watcher::new(WKind::Item);
                w.set(10, MetadataValue::ItemStack(Some(s)));
                w.dirty.clear();
                self.add_obj(Obj {
                    eid: 0,
                    kind: ObjKind::Item,
                    owner: None,
                    pos: base,
                    vel: V3::new(
                        -ang.sin() * fspeed,
                        0.200_000_002_980_232_24,
                        ang.cos() * fspeed,
                    ),
                    yaw,
                    pitch: 0.0,
                    on_ground: false,
                    alive: true,
                    stuck: false,
                    watcher: w,
                    velocity_changed: false,
                    target: None,
                    trail: None,
                    hooked: None,
                    retract_at: 0,
                    explode_at: 0,
                    collect_by: None,
                    born: t,
                    gravity: 0.04,
                    drag: 0.98,
                });
                self.stats.bump("dropped_items");
            }
            let collector = killer.unwrap_or(IDX_V);
            for value in [37i16, 17, 7] {
                let (vx, vy, vz) = {
                    let r = &mut self.players[v].rng;
                    (
                        (r.f() * 0.2 - 0.1) * 2.0,
                        r.f() * 0.2 * 2.0,
                        (r.f() * 0.2 - 0.1) * 2.0,
                    )
                };
                self.add_obj(Obj {
                    eid: 0,
                    kind: ObjKind::XpOrb(value),
                    owner: None,
                    pos: pos.add(V3::new(0.0, 0.5, 0.0)),
                    vel: V3::new(vx, vy, vz),
                    yaw: 0.0,
                    pitch: 0.0,
                    on_ground: false,
                    alive: true,
                    stuck: false,
                    watcher: Watcher::new(WKind::Plain),
                    velocity_changed: false,
                    target: None,
                    trail: None,
                    hooked: None,
                    retract_at: 0,
                    explode_at: 0,
                    collect_by: Some(collector),
                    born: t,
                    gravity: 0.03,
                    drag: 0.98,
                });
                self.stats.bump("xp_orbs");
            }
        }
        self.event(
            "death",
            json!({"victim": victim, "killer": killer_name, "void": void, "pos": pos.json()}),
        );
    }

    fn death_drops(&mut self, v: usize) -> Vec<ItemStack> {
        let mut out: Vec<ItemStack> = Vec::new();
        for slot in [4usize, 3, 2, 1] {
            if let Some(s) = self.players[v].equip[slot].clone() {
                out.push(s);
            }
        }
        if let Some(s) = self.players[v].sword.clone() {
            out.push(s);
        }
        let r = &mut self.players[v].rng;
        let extra: [(&str, i32); 16] = [
            ("bow", 1),
            ("arrow", r.int(4, 16) as i32),
            ("fishing_rod", 1),
            ("ender_pearl", r.int(1, 3) as i32),
            ("stone", 64),
            ("planks", r.int(12, 48) as i32),
            ("golden_apple", r.int(1, 3) as i32),
            ("snowball", r.int(6, 16) as i32),
            ("cooked_beef", r.int(4, 16) as i32),
            ("diamond", r.int(1, 5) as i32),
            ("iron_ingot", r.int(2, 9) as i32),
            ("experience_bottle", r.int(8, 32) as i32),
            ("lava_bucket", 1),
            ("water_bucket", 1),
            ("egg", r.int(4, 16) as i32),
            ("diamond_pickaxe", 1),
        ];
        for (name, n) in extra {
            if out.len() >= 20 {
                break;
            }
            out.push(stack(name, n));
        }
        out
    }

    // ---------------------------------------------------------------------------------------
    // Projectiles
    // ---------------------------------------------------------------------------------------

    /// Launch velocity (speed `speed`) from `from` that lands on `to` under the given gravity and
    /// drag (vanilla projectile integration: move, then drag, then gravity).
    fn solve_launch(from: V3, to: V3, speed: f64, gravity: f64, drag: f64) -> V3 {
        let dir = to.sub(from).hnorm();
        let mut lo = -60.0f64;
        let mut hi = 60.0f64;
        for _ in 0..40 {
            let mid = (lo + hi) * 0.5;
            let (s, c) = mid.to_radians().sin_cos();
            let mut p = from;
            let mut v = V3::new(dir.x * c * speed, s * speed, dir.z * c * speed);
            let target_h = to.hdist(from);
            let mut y_at = None;
            for _ in 0..200 {
                let np = p.add(v);
                if np.hdist(from) >= target_h {
                    let frac =
                        (target_h - p.hdist(from)) / (np.hdist(from) - p.hdist(from)).max(1e-9);
                    y_at = Some(p.y + (np.y - p.y) * frac);
                    break;
                }
                p = np;
                v = v.scale(drag);
                v.y -= gravity;
            }
            match y_at {
                Some(y) if y < to.y => lo = mid,
                Some(_) => hi = mid,
                None => lo = mid,
            }
        }
        let (s, c) = ((lo + hi) * 0.5).to_radians().sin_cos();
        V3::new(dir.x * c * speed, s * speed, dir.z * c * speed)
    }

    fn proj_obj(kind: ObjKind, owner: usize, pos: V3, vel: V3, gravity: f64, drag: f64) -> Obj {
        let watcher = Watcher::new(match kind {
            ObjKind::Arrow { .. } => WKind::Arrow,
            ObjKind::Firework => WKind::Firework,
            _ => WKind::Plain,
        });
        Obj {
            eid: 0,
            kind,
            owner: Some(owner),
            pos,
            vel,
            yaw: (vel.x.atan2(vel.z)).to_degrees() as f32,
            pitch: (vel.y.atan2(vel.hlen())).to_degrees() as f32,
            on_ground: false,
            alive: true,
            stuck: false,
            watcher,
            velocity_changed: false,
            target: None,
            trail: None,
            hooked: None,
            retract_at: 0,
            explode_at: 0,
            collect_by: None,
            born: 0,
            gravity,
            drag,
        }
    }

    fn eye(&self, i: usize) -> V3 {
        self.players[i].pos.add(V3::new(0.0, 1.62, 0.0))
    }

    /// Snowball/egg throw at `target` (spec §6: ×8, 40% with a trail), a hit or a scripted miss.
    fn throw(&mut self, thrower: usize, target: usize, egg: bool, hit: bool, trail: Option<i32>) {
        if !self.players[thrower].alive
            || !self.players[target].alive
            || !self.players[thrower].scene
        {
            return;
        }
        let item = if egg {
            stack("egg", 16)
        } else {
            stack("snowball", 16)
        };
        self.hold(thrower, item, 4);
        self.swing(thrower);
        let from = self.eye(thrower).sub(V3::new(0.0, 0.1, 0.0));
        let mut aim = self.players[target].pos.add(V3::new(0.0, 1.0, 0.0));
        aim = aim.add(self.players[target].vel.scale(6.0));
        if !hit {
            let side = V3::new(-(aim.z - from.z), 0.0, aim.x - from.x).hnorm();
            aim = aim.add(side.scale(1.8)).add(V3::new(0.0, -0.9, 0.0));
        }
        let vel = Self::solve_launch(from, aim, 1.5, 0.03, 0.99);
        let pitch = 1.0 / (self.players[thrower].rng.range(0.8, 1.2) as f32 * 0.4 + 0.8);
        self.sound("random.bow", self.players[thrower].pos, 0.5, pitch * 0.4);
        let mut o = Self::proj_obj(
            if egg { ObjKind::Egg } else { ObjKind::Snowball },
            thrower,
            from,
            vel,
            0.03,
            0.99,
        );
        o.target = if hit { Some(target) } else { None };
        o.trail = trail;
        self.add_obj(o);
        self.stats.bump("throwables");
        let (tn, vn) = (
            self.players[thrower].name.clone(),
            self.players[target].name.clone(),
        );
        self.event(
            "throw",
            json!({"by": tn, "at": vn, "kind": if egg { "egg" } else { "snowball" }, "hit": hit, "trail": trail}),
        );
    }

    /// A projectile/rod hit on a player: hurt, knockback through `velocityChanged` (all trackers
    /// and the victim), and a small cosmetic damage.
    fn projectile_hit(&mut self, v: usize, from: V3, horizontal: f64, damage: f32, arrow: bool) {
        let v_eid = self.players[v].eid;
        if v == IDX_V || self.tracked(v) {
            self.emit(&s2c::EntityStatus {
                entity_id: v_eid,
                opcode: 2,
            });
        }
        if v != IDX_V {
            let p = self.players[v].pos;
            self.sound("game.player.hurt", p, 1.0, 1.0);
        }
        let dir = self.players[v].pos.sub(from).hnorm();
        let mut kb = self.players[v].vel.scale(0.5);
        kb.x += dir.x * horizontal;
        kb.z += dir.z * horizontal;
        kb.y = (kb.y + 0.4).min(0.4) + 0.1;
        self.players[v].last_damage = self.t;
        if v == IDX_V {
            if self.local_kb {
                self.emit(&s2c::EntityVelocity {
                    entity_id: LOCAL_EID,
                    motion_x: vel8000(kb.x),
                    motion_y: vel8000(kb.y),
                    motion_z: vel8000(kb.z),
                });
                self.players[IDX_V].kb_pending = Some(kb);
            }
        } else {
            self.players[v].kb_pending = Some(kb);
            self.players[v].airborne_flag = true;
            if self.tracked(v) {
                self.emit(&s2c::EntityVelocity {
                    entity_id: v_eid,
                    motion_x: vel8000(kb.x),
                    motion_y: vel8000(kb.y),
                    motion_z: vel8000(kb.z),
                });
            }
        }
        if arrow {
            let n = self.players[v].arrows_in.saturating_add(1);
            self.players[v].arrows_in = n;
            self.players[v].watcher.set(9, MetadataValue::Byte(n));
        }
        let h = self.players[v].health;
        self.set_health(v, (h - damage).max(1.0));
        self.stats.bump("projectile_hits");
    }

    fn step_objects(&mut self) {
        let t = self.t;
        for o in 0..self.objs.len() {
            if !self.objs[o].alive {
                continue;
            }
            let kind = self.objs[o].kind;
            match kind {
                ObjKind::Item => self.step_item(o, false),
                ObjKind::XpOrb(_) => self.step_xp(o),
                ObjKind::Firework => {
                    let ob = &mut self.objs[o];
                    ob.vel.x *= 1.15;
                    ob.vel.z *= 1.15;
                    ob.vel.y += 0.04;
                    ob.pos = ob.pos.add(ob.vel);
                    if t >= ob.explode_at {
                        let eid = ob.eid;
                        if self.obj_tracked(o) {
                            self.emit(&s2c::EntityStatus {
                                entity_id: eid,
                                opcode: 17,
                            });
                        }
                        self.remove_obj(o);
                        self.event("firework_explode", json!({"eid": eid}));
                    }
                }
                ObjKind::Hook => self.step_hook(o),
                _ => self.step_projectile(o),
            }
        }
    }

    fn step_item(&mut self, o: usize, _orb: bool) {
        let ob = self.objs[o].clone();
        if ob.on_ground && ob.vel.hlen() < 1e-3 {
            return;
        }
        let mut pos = ob.pos;
        let mut vel = ob.vel;
        vel.y -= ob.gravity;
        let probe = self.terrain.probe(pos.x + vel.x, pos.z + vel.z, pos.y);
        if !probe.wall {
            pos.x += vel.x;
            pos.z += vel.z;
        } else {
            vel.x = 0.0;
            vel.z = 0.0;
        }
        let floor = self
            .terrain
            .probe(pos.x, pos.z, pos.y.max(pos.y + vel.y))
            .floor;
        let ny = pos.y + vel.y;
        let mut on_ground = false;
        match floor {
            Some(h) if ny <= h => {
                pos.y = h;
                vel.y = 0.0;
                on_ground = true;
            }
            _ => pos.y = ny,
        }
        let f = if on_ground { 0.588 } else { ob.drag };
        vel.x *= f;
        vel.z *= f;
        vel.y *= 0.98;
        if on_ground && vel.hlen() < 0.003 {
            vel = V3::ZERO;
        }
        if pos.y < -64.0 {
            self.remove_obj(o);
            return;
        }
        let ob = &mut self.objs[o];
        ob.pos = pos;
        ob.vel = vel;
        ob.on_ground = on_ground;
    }

    fn step_xp(&mut self, o: usize) {
        let t = self.t;
        let Some(c) = self.objs[o].collect_by else {
            self.step_item(o, true);
            return;
        };
        if !self.players[c].alive {
            self.step_item(o, true);
            return;
        }
        let feet = self.players[c].pos;
        let target = feet.add(V3::new(0.0, 0.9, 0.0));
        {
            // EntityXPOrb.onUpdate: drift towards the closest player within 8 blocks.
            let ob = &mut self.objs[o];
            let d = target.sub(ob.pos);
            let dist = d.len();
            if dist < 8.0 && t - ob.born > 8 {
                let k = (1.0 - dist / 8.0).powi(2) * 0.1;
                ob.vel = ob.vel.add(d.scale(1.0 / dist.max(1e-6)).scale(k));
                ob.on_ground = false;
            }
        }
        self.step_item(o, true);
        let ob = &self.objs[o];
        let age = t - ob.born;
        let touching = ob.pos.hdist(feet) < 1.0 && (ob.pos.y - feet.y).abs() < 2.0;
        // Collected on touch, or after 3.5 s at the latest (the client animates either way).
        if (touching && age > 20) || age >= 70 {
            let ObjKind::XpOrb(value) = ob.kind else {
                return;
            };
            let (eid, collector) = (ob.eid, self.players[c].eid);
            if self.obj_tracked(o) {
                self.emit(&s2c::CollectItem {
                    collected_entity_id: eid,
                    collector_entity_id: collector,
                });
            }
            if c == IDX_V {
                let p = self.players[IDX_V].pos;
                let pitch = 0.5 * (self.rng.range(-0.7, 0.7) as f32 + 1.8);
                self.sound("random.orb", p, 0.1, pitch);
                self.xp_total += i32::from(value);
                self.emit_xp();
            }
            self.remove_obj(o);
            self.stats.bump("xp_collected");
        }
    }

    fn step_projectile(&mut self, o: usize) {
        let ob = self.objs[o].clone();
        if ob.stuck {
            return;
        }
        let from = ob.pos;
        let to = ob.pos.add(ob.vel);
        // Entity hit along the segment (scripted target only; misses fly past).
        if let Some(target) = ob.target {
            let tp = self.players[target].pos;
            let steps = 8;
            for s in 1..=steps {
                let p = from.add(to.sub(from).scale(f64::from(s) / f64::from(steps)));
                let rel = p.sub(tp);
                if rel.x.abs() <= 0.6
                    && rel.z.abs() <= 0.6
                    && rel.y >= -0.2
                    && rel.y <= 2.0
                    && self.players[target].alive
                {
                    self.projectile_impact(o, p, Some(target));
                    return;
                }
            }
        }
        // Block hit.
        let n = ((to.sub(from).len() * 4.0).ceil() as i32).max(1);
        for s in 1..=n {
            let p = from.add(to.sub(from).scale(f64::from(s) / f64::from(n)));
            let (bx, by, bz) = (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
            if let Some(top) = self.terrain.top_of(bx, by, bz) {
                if p.y <= top {
                    self.projectile_impact(o, p, None);
                    return;
                }
            }
        }
        let ob = &mut self.objs[o];
        ob.pos = to;
        ob.vel = ob.vel.scale(ob.drag);
        ob.vel.y -= ob.gravity;
        ob.yaw = (ob.vel.x.atan2(ob.vel.z)).to_degrees() as f32;
        ob.pitch = (ob.vel.y.atan2(ob.vel.hlen())).to_degrees() as f32;
        let (pos, trail) = (ob.pos, ob.trail);
        if pos.y < -64.0 {
            self.remove_obj(o);
            return;
        }
        if let Some(particle) = trail {
            if self.players[IDX_V].pos.dist(pos) <= TRAIL_RANGE {
                self.emit(&s2c::Particles {
                    particle_id: particle,
                    long_distance: false,
                    x: pos.x as f32,
                    y: pos.y as f32,
                    z: pos.z as f32,
                    offset_x: 0.0,
                    offset_y: 0.0,
                    offset_z: 0.0,
                    particle_speed: 0.0,
                    particle_count: 1,
                    extra_data: vec![],
                });
                self.stats.bump("trail_particle_packets");
            }
        }
    }

    fn projectile_impact(&mut self, o: usize, at: V3, hit: Option<usize>) {
        let ob = self.objs[o].clone();
        let owner = ob.owner.unwrap_or(IDX_V);
        match ob.kind {
            ObjKind::Arrow { .. } => {
                if let Some(v) = hit {
                    let power = self.players[owner].gear.power;
                    self.projectile_hit(v, ob.pos, 0.3, 3.0 + f32::from(power), true);
                    self.sound("random.bowhit", self.players[v].pos, 1.0, 1.1);
                    self.remove_obj(o);
                } else {
                    let ob = &mut self.objs[o];
                    ob.pos = at;
                    ob.vel = V3::ZERO;
                    ob.stuck = true;
                    ob.on_ground = true;
                    self.stats.bump("arrows_stuck");
                }
            }
            ObjKind::Snowball | ObjKind::Egg => {
                if let Some(v) = hit {
                    self.projectile_hit(v, ob.pos, 0.35, 0.0, false);
                }
                self.objs[o].pos = at;
                self.remove_obj(o);
            }
            ObjKind::Pearl => {
                self.objs[o].pos = at;
                self.remove_obj(o);
                // EntityEnderPearl.onImpact: teleport the thrower, 5 fall damage.
                let landing = self.grid_snap(at);
                let p = &mut self.players[owner];
                p.pos = landing;
                p.vel = V3::ZERO;
                p.on_ground = true;
                let h = p.health;
                let eid = p.eid;
                if self.tracked(owner) {
                    self.emit(&s2c::EntityStatus {
                        entity_id: eid,
                        opcode: 2,
                    });
                }
                self.sound("game.player.hurt", landing, 1.0, 1.0);
                self.players[owner].last_damage = self.t;
                self.set_health(owner, (h - 5.0).max(1.0));
                self.event("pearl_landed", json!({"at": landing.json()}));
            }
            ObjKind::Potion(damage) => {
                self.objs[o].pos = at;
                self.emit(&s2c::Effect {
                    effect_id: 2002,
                    position: BlockPos::new(
                        at.x.floor() as i32,
                        at.y.floor() as i32,
                        at.z.floor() as i32,
                    ),
                    data: damage,
                    disable_relative_volume: false,
                });
                self.remove_obj(o);
                // Regeneration II for players within 4 blocks.
                for i in 0..self.players.len() {
                    if !self.players[i].scene || !self.players[i].alive {
                        continue;
                    }
                    if self.players[i].pos.dist(at) <= 4.0 {
                        let until = self.t + 300;
                        let p = &mut self.players[i];
                        p.effects.retain(|e| e.id != EFFECT_REGENERATION);
                        p.effects.push(ActiveEffect {
                            id: EFFECT_REGENERATION,
                            amplifier: 1,
                            until,
                        });
                        self.refresh_potion_colour(i);
                    }
                }
                self.event("potion_splash", json!({"at": at.json(), "damage": damage}));
            }
            _ => {}
        }
    }

    fn step_hook(&mut self, o: usize) {
        let t = self.t;
        let ob = self.objs[o].clone();
        let owner = ob.owner.unwrap_or(IDX_V);
        if t >= ob.retract_at {
            self.remove_obj(o);
            self.swing(owner);
            self.event("rod_retract", json!({"by": self.players[owner].name}));
            return;
        }
        if let Some(h) = ob.hooked {
            let p = self.players[h].pos.add(V3::new(0.0, 1.44, 0.0));
            let ob = &mut self.objs[o];
            ob.vel = p.sub(ob.pos);
            ob.pos = p;
            return;
        }
        if ob.stuck {
            return;
        }
        let to = ob.pos.add(ob.vel);
        if let Some(target) = ob.target {
            let tp = self.players[target].pos;
            for s in 1..=8 {
                let p = ob.pos.add(ob.vel.scale(f64::from(s) / 8.0));
                let rel = p.sub(tp);
                if rel.hlen() <= 0.9 && rel.y > -0.3 && rel.y < 2.1 {
                    self.objs[o].hooked = Some(target);
                    self.objs[o].pos = p;
                    let from = self.players[owner].pos;
                    self.projectile_hit(target, from, 0.4, 0.0, false);
                    self.event("rod_hit", json!({"on": self.players[target].name}));
                    return;
                }
            }
        }
        if let Some(floor) = self.terrain.probe(to.x, to.z, to.y).floor {
            if to.y <= floor {
                let ob = &mut self.objs[o];
                ob.pos = V3::new(to.x, floor, to.z);
                ob.vel = V3::ZERO;
                ob.stuck = true;
                ob.on_ground = true;
                return;
            }
        }
        let ob = &mut self.objs[o];
        ob.pos = to;
        ob.vel = ob.vel.scale(0.92);
        ob.vel.y -= 0.04;
    }

    // ---------------------------------------------------------------------------------------
    // Player physics (simplified vanilla) and step sounds
    // ---------------------------------------------------------------------------------------

    fn step_player(&mut self, i: usize) {
        if !self.players[i].scene {
            return;
        }
        if matches!(self.players[i].mode, Mode::Dead) {
            return;
        }
        let input = self.players[i].input;
        let speed_mult = self.players[i]
            .effects
            .iter()
            .find(|e| e.id == EFFECT_SPEED)
            .map_or(1.0, |e| 1.0 + 0.2 * f64::from(e.amplifier + 1));
        let before = self.players[i].pos;
        {
            let p = &mut self.players[i];
            if let Some(kb) = p.kb_pending.take() {
                p.vel = kb;
                if kb.y > 0.0 {
                    p.on_ground = false;
                }
            }
            let base = 0.1 * speed_mult * if p.sprinting { 1.3 } else { 1.0 };
            let accel = if p.on_ground {
                base
            } else if p.sprinting {
                0.026
            } else {
                0.02
            };
            let mut mag = input.mag * 0.98;
            if p.using {
                mag *= 0.2;
            }
            p.vel.x += input.wish.x * accel * mag;
            p.vel.z += input.wish.z * accel * mag;
            if input.jump && p.on_ground {
                p.vel.y = 0.42;
                if p.sprinting {
                    let d = dir_of_yaw(p.yaw);
                    p.vel.x += d.x * 0.2;
                    p.vel.z += d.z * 0.2;
                }
                p.on_ground = false;
                p.airborne_flag = true;
            }
        }
        let p = self.players[i].clone();
        let mut pos = p.pos;
        let mut vel = p.vel;
        let mut on_ground = p.on_ground;
        // Horizontal move with per-axis sliding (vanilla resolves collisions axis by axis).
        let blocked = |terrain: &Terrain, x: f64, z: f64| {
            let probe = terrain.probe(x, z, p.pos.y);
            let unsafe_floor = match probe.floor {
                Some(h) => h < p.pos.y - 3.5,
                None => true,
            };
            probe.wall || (unsafe_floor && !p.allow_void)
        };
        let (nx, nz) = (p.pos.x + p.vel.x, p.pos.z + p.vel.z);
        if !blocked(&self.terrain, nx, nz) {
            pos.x = nx;
            pos.z = nz;
        } else if !blocked(&self.terrain, nx, p.pos.z) {
            pos.x = nx;
            vel.z = 0.0;
        } else if !blocked(&self.terrain, p.pos.x, nz) {
            pos.z = nz;
            vel.x = 0.0;
        } else {
            vel.x = 0.0;
            vel.z = 0.0;
        }
        let floor = self.terrain.probe(pos.x, pos.z, pos.y).floor;
        if on_ground && vel.y <= 0.0 {
            match floor {
                Some(h) if h >= pos.y - 0.05 => {
                    pos.y = h.max(pos.y.min(h));
                    vel.y = 0.0;
                }
                _ => on_ground = false,
            }
        }
        if !on_ground {
            let ny = pos.y + vel.y;
            let fl = self.terrain.probe(pos.x, pos.z, pos.y.max(ny)).floor;
            match fl {
                Some(h) if vel.y <= 0.0 && ny <= h => {
                    pos.y = h;
                    vel.y = 0.0;
                    on_ground = true;
                }
                _ => {
                    pos.y = ny;
                    vel.y = (vel.y - 0.08) * 0.98;
                }
            }
        }
        let fr = if on_ground { 0.546 } else { 0.91 };
        vel.x *= fr;
        vel.z *= fr;
        if on_ground {
            vel.y = 0.0;
        }
        {
            let q = &mut self.players[i];
            q.pos = pos;
            q.vel = vel;
            q.on_ground = on_ground;
        }
        // Server-side step sounds (Entity.moveEntity -> playStepSound -> EntityPlayer.playSound,
        // sent to everyone within 16 blocks except the walker).
        if i != IDX_V && on_ground && !p.using {
            let moved = pos.hdist(before);
            let q = &mut self.players[i];
            q.walked += moved * 0.6;
            if q.walked > q.next_step {
                q.next_step = q.walked.floor() + 1.0;
                let under = self.terrain.state(
                    pos.x.floor() as i32,
                    (pos.y - 0.2).floor() as i32,
                    pos.z.floor() as i32,
                );
                let info = mc_data::block_info(BlockState::from_raw(under).block());
                let name = format!("step.{}", info.step_sound);
                if sound_event(&name).is_some() {
                    self.sound(
                        &name,
                        pos,
                        0.15 * info.step_sound_volume,
                        info.step_sound_pitch,
                    );
                    self.stats.bump("step_sounds");
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // AI
    // ---------------------------------------------------------------------------------------

    fn in_fight(&self) -> bool {
        (FIGHT_START..FIGHT_END).contains(&self.t)
    }

    fn face(&mut self, i: usize, target: V3, step: f32) {
        let eye = self.eye(i);
        let p = &mut self.players[i];
        let ty = yaw_to(p.pos, target);
        p.yaw = turn_toward(p.yaw, ty, step);
        p.pitch = pitch_to(eye, target).clamp(-60.0, 60.0);
    }

    fn viewer_target(&self) -> Option<usize> {
        let t = self.t;
        let e1 = idx_e(1);
        let e11 = idx_e(11);
        if (FIGHT_START..self.kill_tick_e1).contains(&t) && self.players[e1].alive {
            Some(e1)
        } else if (ft(48.0)..FIGHT_END).contains(&t) && self.players[e11].alive {
            Some(e11)
        } else {
            None
        }
    }

    /// The viewer model: the wrapper's `aim_entity` + `approach` + `jump` keys.
    fn viewer_ai(&mut self) {
        let t = self.t;
        let mut input = Input::default();
        let mut forward = false;
        if let Some(j) = self.viewer_target() {
            let target = self.players[j].pos.add(V3::new(0.0, 0.9, 0.0));
            self.face(IDX_V, target, 180.0);
            let d = self.players[j].pos.sub(self.players[IDX_V].pos);
            if d.hlen() > KEEP_WITHIN {
                input.wish = d.hnorm();
                input.mag = 1.0;
                forward = true;
            }
        }
        input.jump = self.jumps.contains(&t);
        // Sprint follows the held forward key (approach holds sprint with forward); the server
        // sees the client's START/STOP_SPRINTING one tick later.
        let resume_ok = t >= self.players[IDX_V].sprint_resume;
        self.set_sprint(IDX_V, forward && resume_ok);
        self.players[IDX_V].input = input;
    }

    /// Resolve the viewer's presses from tick `t-1` (server reaction at `t`).
    fn viewer_hits(&mut self) {
        let t = self.t;
        let Some(&count) = self.presses.get(&(t - 1)) else {
            return;
        };
        if count == 0 {
            return;
        }
        // The target the viewer was aiming at when it pressed.
        let target = {
            let saved = self.t;
            self.t = t - 1;
            let tg = self.viewer_target();
            self.t = saved;
            tg
        };
        let Some(j) = target else {
            return;
        };
        if !self.players[j].alive {
            return;
        }
        if j == idx_e(1) && t == self.kill_tick_e1 {
            self.melee_hit(IDX_V, j, true, None);
            self.firework_kill_effect(j);
            self.viewer_kill_rewards(j);
            return;
        }
        let dist = self.players[j].pos.hdist(self.players[IDX_V].pos);
        if dist <= 3.4 && self.hurt_ok(j) {
            self.melee_hit(IDX_V, j, false, None);
        }
    }

    fn viewer_kill_rewards(&mut self, j: usize) {
        self.kills += 1;
        let kills = self.kills.to_string();
        self.sidebar_line("sw_l_kills", "Kills: ", &kills);
        self.chat("§6+5 coins!");
        self.chat("§b+1 Soul");
        let name = self.players[j].name.clone();
        self.event("viewer_kill", json!({"victim": name}));
    }

    fn firework_kill_effect(&mut self, j: usize) {
        let pos = self.players[j].pos.add(V3::new(0.0, 1.0, 0.0));
        let mut o = Self::proj_obj(
            ObjKind::Firework,
            IDX_V,
            pos,
            V3::new(0.0, 0.05, 0.0),
            0.0,
            1.0,
        );
        o.watcher
            .set(8, MetadataValue::ItemStack(Some(firework_rocket())));
        o.watcher.dirty.clear();
        o.explode_at = self.t + 12;
        o.yaw = 0.0;
        o.pitch = 90.0;
        self.add_obj(o);
        self.sound("fireworks.launch", pos, 3.0, 1.0);
        self.event("kill_effect", json!({"kind": "Firework", "at": pos.json()}));
    }

    fn lightning_kill_effect(&mut self, j: usize) {
        let pos = self.players[j].pos;
        let eid = self.next_eid;
        self.next_eid += 1;
        self.emit(&s2c::SpawnGlobalEntity {
            entity_id: eid,
            entity_type: 1,
            x_fixed: fix32(pos.x),
            y_fixed: fix32(pos.y),
            z_fixed: fix32(pos.z),
        });
        self.event(
            "kill_effect",
            json!({"kind": "Lightning", "at": pos.json()}),
        );
    }

    fn duel_ai(&mut self, i: usize) {
        let t = self.t;
        let Some(j) = self.players[i].opponent else {
            self.idle_ai(i);
            return;
        };
        if !self.players[j].alive || (j == IDX_V && !self.players[IDX_V].alive) {
            self.idle_ai(i);
            return;
        }
        let me = self.players[i].pos;
        let opp = self.players[j].pos;
        let to = opp.sub(me);
        let dist = to.hlen();
        {
            let p = &mut self.players[i];
            if t >= p.strafe_until {
                p.strafe = if p.rng.chance(0.6) {
                    -p.strafe
                } else {
                    p.strafe
                };
                p.strafe_until = t + p.rng.int(6, 20);
            }
            if t % 40 == 0 {
                p.desired_r = p.rng.range(2.3, 2.9);
            }
        }
        let (strafe, desired_r, home, home_r) = {
            let p = &self.players[i];
            (p.strafe, p.desired_r, p.home, p.home_radius)
        };
        let mut input = Input {
            mag: 1.0,
            ..Input::default()
        };
        // Close in whenever out of the orbit band (knockback separates fighters; both sides
        // sprint back in, the victim steering in the air); strafe around the opponent inside it.
        let chasing = dist > desired_r + 0.25;
        let mut wish = if chasing || !self.players[i].on_ground {
            to.hnorm()
        } else {
            let out = to.hnorm().scale(-1.0);
            let tangent = V3::new(-out.z, 0.0, out.x).scale(strafe * 0.7);
            let err = (desired_r - dist).clamp(-1.0, 1.0) * 1.5;
            tangent.add(out.scale(err))
        };
        // E1/E11 prefer the far side of the viewer from the anchor, so their hits push the
        // viewer back towards open ground.
        if j == IDX_V && !chasing {
            let va = opp.sub(self.plan.anchor);
            if va.hlen() > 1.5 {
                let bias = opp.add(va.hnorm().scale(desired_r));
                wish = wish.add(bias.sub(me).hnorm().scale(0.6));
            }
        }
        // Anchoring: every fight belongs to a home point; the pull grows past its radius, so
        // knockback can never walk a fight off across the island (the viewer follows E1/E11).
        // A soft zone first (the fight may roam 2 blocks past the radius while the opponent is
        // inside it), then a hard pull.
        let opp_inside = opp.hdist(home) <= home_r + 2.0;
        let slack = if opp_inside { 2.0 } else { 0.0 };
        let excess = me.hdist(home) - home_r - slack;
        if excess > 0.0 {
            wish = wish
                .hnorm()
                .add(home.sub(me).hnorm().scale(0.8 + 1.5 * excess));
        }
        input.wish = wish.hnorm();
        // W-tap / sprint resets.
        let fighting = self.in_fight();
        {
            let using = self.players[i].using;
            let p = &mut self.players[i];
            let mut want = !using && t >= p.sprint_resume;
            if fighting && t >= p.next_wtap && !chasing {
                p.sprint_resume = t + p.rng.int(2, 3);
                p.next_wtap = t + p.rng.int(14, 30);
                want = false;
            }
            let _ = want;
        }
        let want_sprint = {
            let p = &self.players[i];
            !p.using && t >= p.sprint_resume
        };
        self.set_sprint(i, want_sprint);
        // Jumps (~0.4/s on top of the knockback airtime) and steps.
        {
            let p = &mut self.players[i];
            if p.on_ground && p.rng.chance(0.02) {
                input.jump = true;
            }
        }
        if self.step_ahead(i, input.wish) {
            input.jump = true;
        }
        // Block-hit sometimes.
        if fighting {
            let using_now = t < self.players[i].use_until;
            if !using_now && self.players[i].using && self.players[i].use_until >= 0 {
                self.set_using(i, false);
            }
            if !using_now && !chasing && self.players[i].rng.chance(0.02) {
                let d = self.players[i].rng.int(4, 7);
                self.players[i].use_until = t + d;
                self.set_using(i, true);
            }
        }
        // Face the opponent's body, with the constant small aim corrections of a real mouse.
        let target = opp.add(V3::new(0.0, 1.1, 0.0));
        self.face(i, target, 40.0);
        {
            let p = &mut self.players[i];
            p.yaw += p.rng.range(-3.0, 3.0) as f32;
            p.pitch = (p.pitch + p.rng.range(-2.0, 2.0) as f32).clamp(-60.0, 60.0);
        }
        self.players[i].input = input;
        // Clicks and damaging hits.
        if !fighting || self.players[i].using {
            return;
        }
        // Scripted kill: E3 kills E5 at +12 s (first moment E5 is hittable and within reach,
        // forced by +13 s at the latest).
        let e5 = idx_e(5);
        if i == idx_e(3) && j == e5 && !self.e5_killed && t >= ft(12.0) {
            if (dist <= 3.3 && self.hurt_ok(j)) || t >= ft(13.0) {
                self.e5_killed = true;
                self.melee_hit(i, j, true, None);
                self.lightning_kill_effect(e5);
                self.players[idx_e(3)].opponent = Some(idx_e(6));
                self.players[idx_e(3)].home = self.plan.b_centre;
            }
            return;
        }
        let cps = self.players[i].cps;
        let click = self.players[i].rng.chance(cps / 20.0) && dist <= 4.5;
        if !click {
            return;
        }
        self.swing(i);
        let accuracy = self.players[i].accuracy;
        let recovered = t - self.players[i].last_damage >= HIT_RECOVERY_TICKS;
        if dist <= 3.3 && recovered && self.hurt_ok(j) && self.players[i].rng.chance(accuracy) {
            if j == e5 && t >= ft(11.5) && !self.e5_killed {
                return; // E5's killing blow is reserved for E3
            }
            if i == idx_e(6) && t >= ft(33.0) {
                return; // E6 is retreating to the edge
            }
            if j == idx_e(6) && t >= ft(33.0) && !self.e6_pushed {
                return; // the void knock at +35 s is scripted
            }
            if j == idx_e(1) && (self.kill_tick_e1 - 12..self.kill_tick_e1).contains(&t) {
                return; // keep E1 hittable for the viewer's kill at +45 s
            }
            self.melee_hit(i, j, false, None);
        }
    }

    /// True when the next column in the wish direction is a 0.6-1.25 block step (jump needed).
    fn step_ahead(&self, i: usize, wish: V3) -> bool {
        let p = &self.players[i];
        if !p.on_ground || wish.hlen() < 0.1 {
            return false;
        }
        let ahead = p.pos.add(wish.scale(0.7));
        let probe = self.terrain.probe(ahead.x, ahead.z, p.pos.y);
        if !probe.wall {
            return false;
        }
        let up = self.terrain.probe(ahead.x, ahead.z, p.pos.y + 1.0);
        !up.wall && up.floor.is_some_and(|h| h - p.pos.y <= 1.25 + 1e-6)
    }

    fn idle_ai(&mut self, i: usize) {
        let t = self.t;
        let mut input = Input::default();
        let home = self.players[i].home;
        let me = self.players[i].pos;
        if me.hdist(home) > 2.5 {
            input.wish = home.sub(me).hnorm();
            input.mag = 0.6;
        }
        if self.step_ahead(i, input.wish) {
            input.jump = true;
        }
        self.set_sprint(i, false);
        if self.players[i].using {
            self.set_using(i, false);
        }
        // Look around slowly; face the viewer now and then.
        let look = if (t / 60) % 2 == 0 {
            self.players[IDX_V].pos.add(V3::new(0.0, 1.0, 0.0))
        } else {
            let a = f64::from((t / 60) as i32) * 1.7;
            me.add(V3::new(a.cos() * 5.0, 1.0, a.sin() * 5.0))
        };
        self.face(i, look, 6.0);
        self.players[i].input = input;
    }

    fn path_ai(&mut self, i: usize) {
        let Mode::Path { mut points, sprint } = self.players[i].mode.clone() else {
            return;
        };
        let me = self.players[i].pos;
        while let Some(next) = points.front() {
            if me.hdist(*next) < 0.45 {
                points.pop_front();
            } else {
                break;
            }
        }
        let mut input = Input::default();
        match points.front().copied() {
            Some(next) => {
                input.wish = next.sub(me).hnorm();
                input.mag = 1.0;
                let look = next.add(V3::new(0.0, 1.5, 0.0));
                self.face(i, look, 30.0);
                // Jump for steps, and whenever the body stalls against an obstacle.
                let stalled = self.players[i].on_ground && self.players[i].vel.hlen() < 0.02;
                if self.step_ahead(i, input.wish) || (stalled && next.y > me.y + 0.3) {
                    input.jump = true;
                }
                self.set_sprint(i, sprint);
                self.players[i].mode = Mode::Path { points, sprint };
            }
            None => {
                self.set_sprint(i, false);
                self.players[i].mode = Mode::Idle;
            }
        }
        self.players[i].input = input;
    }

    fn start_path(&mut self, i: usize, to: V3, sprint: bool) -> bool {
        let me = self.players[i].pos;
        match walk_path(&self.terrain, me, to, 4_000) {
            Some(points) => {
                let points: VecDeque<V3> = points.into_iter().skip(1).collect();
                self.players[i].mode = Mode::Path { points, sprint };
                true
            }
            None => false,
        }
    }

    fn archer_ai(&mut self, i: usize, slot: usize) {
        let t = self.t;
        let Some(j) = self.players[i].opponent else {
            self.idle_ai(i);
            return;
        };
        let bursts = [(ft(2.0), ft(20.0)), (ft(34.0), ft(52.0))];
        let in_burst = bursts.iter().any(|&(a, b)| (a..b).contains(&t));
        let target = self.players[j].pos.add(V3::new(0.0, 1.1, 0.0));
        self.face(i, target, 25.0);
        // Small strafes around the home spot.
        let mut input = Input::default();
        {
            let p = &mut self.players[i];
            if t >= p.strafe_until {
                p.strafe = -p.strafe;
                p.strafe_until = t + p.rng.int(10, 30);
            }
        }
        let me = self.players[i].pos;
        let home = self.players[i].home;
        let side = target.sub(me).hnorm();
        let mut wish = V3::new(-side.z, 0.0, side.x).scale(self.players[i].strafe);
        if me.hdist(home) > 1.5 {
            wish = home.sub(me).hnorm();
        }
        input.wish = wish;
        input.mag = 0.5;
        self.set_sprint(i, false);
        let (draw_start, release_at, drawing, shots) = self.archer_state[slot];
        if !in_burst || shots >= 24 || !self.players[j].alive {
            if drawing {
                self.set_using(i, false);
                self.archer_state[slot].2 = false;
            }
            self.players[i].input = input;
            return;
        }
        if !drawing && t >= release_at + 6 && t >= draw_start {
            // Start drawing: 70% full charge (20 ticks), else 8-14 ticks.
            let full = self.players[i].rng.chance(0.7);
            let ticks = if full {
                20
            } else {
                self.players[i].rng.int(8, 14)
            };
            self.archer_state[slot] = (t, t + ticks, true, shots);
            self.set_using(i, true);
        } else if drawing && t >= release_at {
            let ticks = (release_at - draw_start) as f64;
            let tt = ticks / 20.0;
            let f = ((tt * tt + tt * 2.0) / 3.0).min(1.0);
            let crit = f >= 1.0;
            let hit = self.players[i].rng.chance(0.4);
            self.set_using(i, false);
            let wait = self.players[i].rng.int(4, 10);
            self.archer_state[slot] = (t + wait, t, false, shots + 1);
            self.shoot_arrow(i, j, f * 3.0, crit, hit);
        }
        self.players[i].input = input;
    }

    fn shoot_arrow(&mut self, i: usize, j: usize, speed: f64, crit: bool, hit: bool) {
        let from = self.eye(i).sub(V3::new(0.0, 0.1, 0.0));
        let mut aim = self.players[j].pos.add(V3::new(0.0, 1.0, 0.0));
        if !hit {
            let r = &mut self.players[i].rng;
            let side = V3::new(-(aim.z - from.z), 0.0, aim.x - from.x).hnorm();
            let s = if r.chance(0.5) { 1.0 } else { -1.0 };
            aim = aim.add(side.scale(s * r.range(1.2, 2.6))).add(V3::new(
                0.0,
                r.range(-1.5, -0.6),
                0.0,
            ));
        }
        let vel = Self::solve_launch(from, aim, speed.max(0.6), 0.05, 0.99);
        let mut o = Self::proj_obj(
            ObjKind::Arrow { shooter: i, crit },
            i,
            from,
            vel,
            0.05,
            0.99,
        );
        if crit {
            o.watcher.set(16, MetadataValue::Byte(1));
            o.watcher.dirty.clear();
        }
        o.target = if hit { Some(j) } else { None };
        // E7 owns a "Notes" projectile-trail cosmetic (server trails follow every projectile).
        if i == idx_e(7) {
            o.trail = Some(PARTICLE_NOTE);
        }
        self.add_obj(o);
        let pitch = 1.0 / (self.players[i].rng.range(0.0, 1.0) as f32 * 0.4 + 1.2)
            + (speed as f32 / 3.0) * 0.5;
        self.sound("random.bow", self.players[i].pos, 1.0, pitch);
        self.stats.bump("arrows_shot");
        if crit {
            self.stats.bump("crit_arrows");
        }
    }

    fn bridge_ai(&mut self, i: usize) {
        let t = self.t;
        let plan = &self.plan.bridge;
        let n = plan.blocks.len();
        // Place blocks on schedule.
        while self.bridge_next < n
            && self
                .bridge_times
                .get(self.bridge_next)
                .is_some_and(|&bt| bt <= t)
        {
            let (bx, by, bz) = plan.blocks[self.bridge_next];
            let state = plan.block_state;
            self.terrain.overlay.insert((bx, by, bz), state);
            self.grid.set(
                bx,
                bz,
                Some(Cell {
                    h: f64::from(by + 1),
                    top_y: by,
                }),
            );
            self.swing(i);
            self.emit(&s2c::BlockChange {
                position: BlockPos::new(bx, by, bz),
                block_state_id: i32::from(state),
            });
            if t < FIGHT_START && t > 20 {
                self.stats.last_block_change_tick_in_join = t;
            }
            let held = self.players[i].equip[0].clone();
            if let Some(mut s) = held {
                s.count -= 1;
                if s.count <= 0 {
                    s.count = 64;
                }
                self.set_equipment(i, 0, Some(s));
            }
            let at = V3::new(
                f64::from(bx) + 0.5,
                f64::from(by) + 0.5,
                f64::from(bz) + 0.5,
            );
            self.sound("dig.stone", at, 1.0, 0.8);
            self.bridge_next += 1;
            self.stats.bump("bridge_blocks");
        }
        // Walk onto the newest block (or the landing once finished).
        let target = if self.bridge_next == 0 {
            plan.start
        } else if self.bridge_next < n {
            let (bx, by, bz) = plan.blocks[self.bridge_next - 1];
            V3::new(f64::from(bx) + 0.5, f64::from(by + 1), f64::from(bz) + 0.5)
        } else {
            plan.landing
        };
        let me = self.players[i].pos;
        let mut input = Input::default();
        let d = target.sub(me);
        if d.hlen() > 0.25 {
            input.wish = d.hnorm();
            input.mag = (d.hlen() * 2.0).min(1.0);
        }
        if self.bridge_next >= n && me.hdist(plan.landing) < 0.6 {
            self.players[i].home = plan.landing.add(plan.dir.scale(2.0));
            self.players[i].mode = Mode::Idle;
        }
        let look = me.add(plan.dir.scale(3.0)).add(V3::new(0.0, -0.5, 0.0));
        self.face(i, look, 20.0);
        self.players[i].input = input;
    }

    fn loot_ai(&mut self, i: usize) {
        let t = self.t;
        let chests = self.plan.chests.clone();
        if t < ft(1.0) || self.loot_step >= chests.len() {
            self.idle_ai(i);
            return;
        }
        let c = &chests[self.loot_step];
        let chest_centre = V3::new(
            f64::from(c.pos.0) + 0.5,
            f64::from(c.pos.1) + 0.5,
            f64::from(c.pos.2) + 0.5,
        );
        match self.loot_phase {
            0 => {
                // Walk to the chest's access cell.
                if self.players[i].pos.hdist(c.access) < 0.6 {
                    self.loot_phase = 1;
                    self.players[i].mode = Mode::Loot;
                    self.players[i].input = Input::default();
                } else {
                    if !matches!(self.players[i].mode, Mode::Path { .. })
                        && !self.start_path(i, c.access, true)
                    {
                        // No path (should not happen: chests were chosen by reachability).
                        self.event(
                            "warning",
                            json!(format!("E10 found no path to chest {}", self.loot_step)),
                        );
                        self.loot_step = chests.len();
                        return;
                    }
                    self.path_ai(i);
                    if matches!(self.players[i].mode, Mode::Idle) {
                        self.players[i].mode = Mode::Loot;
                        self.loot_phase = 1;
                    }
                }
            }
            1 => {
                // Open: swing (right-click), BlockAction 1/1, chest sound from the lid update.
                self.players[i].mode = Mode::Loot;
                self.face(i, chest_centre, 60.0);
                self.swing(i);
                self.emit(&s2c::BlockAction {
                    position: BlockPos::new(c.pos.0, c.pos.1, c.pos.2),
                    data1: 1,
                    data2: 1,
                    block_id: CHEST_BLOCK_ID,
                });
                let pitch = self.players[i].rng.range(0.9, 1.0) as f32;
                self.sound("random.chestopen", chest_centre, 0.5, pitch);
                let wait = self.players[i].rng.int(30, 55);
                self.loot_wait_until = t + wait;
                self.loot_phase = 2;
                self.stats.bump("chest_opens");
                self.event("chest_open", json!({"pos": [c.pos.0, c.pos.1, c.pos.2]}));
            }
            2 => {
                self.face(i, chest_centre, 20.0);
                self.players[i].input = Input::default();
                if t >= self.loot_wait_until {
                    self.emit(&s2c::BlockAction {
                        position: BlockPos::new(c.pos.0, c.pos.1, c.pos.2),
                        data1: 1,
                        data2: 0,
                        block_id: CHEST_BLOCK_ID,
                    });
                    self.loot_wait_until = t + 6;
                    self.loot_phase = 3;
                    // Equip what was found (spec: S04 armour equips).
                    let mut rng = self.players[i].rng.clone();
                    match self.loot_step {
                        0 => self.set_equipment(
                            i,
                            4,
                            Some(enchanted(
                                worn("diamond_helmet", &mut rng, 0.0, 0.1),
                                &[(ENCH_PROTECTION, 4)],
                            )),
                        ),
                        2 => self.set_equipment(
                            i,
                            3,
                            Some(worn("iron_chestplate", &mut rng, 0.0, 0.1)),
                        ),
                        _ => {}
                    }
                    self.players[i].rng = rng;
                }
            }
            _ => {
                self.players[i].input = Input::default();
                if t >= self.loot_wait_until {
                    let pitch = self.players[i].rng.range(0.9, 1.0) as f32;
                    self.sound("random.chestclosed", chest_centre, 0.5, pitch);
                    self.loot_step += 1;
                    self.loot_phase = 0;
                    self.players[i].mode = Mode::Loot;
                    self.players[i].home = c.access;
                }
            }
        }
    }

    fn retreat_ai(&mut self, i: usize, target: V3) {
        let me = self.players[i].pos;
        let mut input = Input::default();
        if me.hdist(target) > 0.3 {
            input.wish = target.sub(me).hnorm();
            input.mag = 1.0;
        }
        if self.step_ahead(i, input.wish) {
            input.jump = true;
        }
        if let Some(j) = self.players[i].opponent {
            let look = self.players[j].pos.add(V3::new(0.0, 1.1, 0.0));
            self.face(i, look, 30.0);
        }
        self.set_sprint(i, false);
        self.players[i].input = input;
    }

    fn ai_tick(&mut self) {
        self.viewer_ai();
        for i in 1..self.players.len() {
            if !self.players[i].scene {
                continue;
            }
            // Held-item restore after throws/rods/gapples.
            if let Some((at, item)) = self.players[i].held_restore.clone() {
                if self.t >= at {
                    self.players[i].held_restore = None;
                    if self.players[i].alive {
                        self.set_equipment(i, 0, Some(item));
                    }
                }
            }
            if !self.players[i].alive {
                if matches!(self.players[i].mode, Mode::Falling) {
                    self.players[i].input = Input::default();
                }
                continue;
            }
            if i == idx_e(10) && self.loot_step < self.plan.chests.len() {
                self.loot_ai(i);
                continue;
            }
            match self.players[i].mode.clone() {
                Mode::Duel => self.duel_ai(i),
                Mode::Idle | Mode::Dead => self.idle_ai(i),
                Mode::Path { .. } => self.path_ai(i),
                Mode::Archer => {
                    let slot = if i == idx_e(7) { 0 } else { 1 };
                    self.archer_ai(i, slot);
                }
                Mode::Bridge => self.bridge_ai(i),
                Mode::Loot => self.loot_ai(i),
                Mode::Retreat { target } => self.retreat_ai(i, target),
                Mode::Falling => {
                    self.players[i].input = Input::default();
                }
            }
            if self.t >= FIGHT_END {
                // Tail: fights stop; nobody attacks or throws, bodies settle.
                let p = &mut self.players[i];
                p.input.mag *= 0.0;
                p.input.jump = false;
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Scheduled events
    // ---------------------------------------------------------------------------------------

    fn events_tick(&mut self) {
        let t = self.t;
        let e = idx_e;
        // Bridge schedule: 3 bursts at 4.5 blocks/s.
        if t == 0 {
            let n = self.plan.bridge.blocks.len();
            let per = [
                n / 3 + usize::from(!n.is_multiple_of(3)),
                n / 3 + usize::from(n % 3 > 1),
                n / 3,
            ];
            let starts = [ft(8.0), ft(16.0), ft(24.0)];
            let mut times = Vec::new();
            for (b, &count) in per.iter().enumerate() {
                for k in 0..count {
                    times.push(starts[b] + (k as f64 * 20.0 / 4.5).round() as i64);
                }
            }
            self.bridge_times = times;
        }
        // Snowballs/eggs (8, 3 with trails) and rods.
        let throws: [(f64, u8, usize, bool, bool, Option<i32>); 8] = [
            (4.0, 1, IDX_V, false, true, Some(PARTICLE_HEART)),
            (9.5, 3, e(5), true, true, None),
            (15.0, 0, e(2), false, true, Some(PARTICLE_FIREWORKS_SPARK)),
            (22.0, 2, IDX_T, true, false, None),
            (27.5, 4, e(6), false, true, None),
            (33.5, 1, IDX_V, true, false, Some(PARTICLE_NOTE)),
            (44.0, 2, IDX_T, false, true, None),
            (52.0, 0, e(2), false, true, None),
        ];
        for (sec, by, at, egg, hit, trail) in throws {
            if t == ft(sec) {
                let thrower = if by == 0 { IDX_T } else { e(by) };
                self.throw(thrower, at, egg, hit, trail);
            }
        }
        for (sec, hit) in [(8.0, true), (38.0, false)] {
            if t == ft(sec) {
                self.rod_cast(e(2), IDX_T, hit);
            }
        }
        if t == ft(20.0) {
            self.pearl(e(1));
        }
        if t == ft(25.0) {
            self.regen_splash(e(2), IDX_T);
        }
        if t == ft(30.0) {
            self.gapple_start(e(1));
        }
        if (ft(30.0)..=ft(30.0) + 32).contains(&t) {
            self.gapple_tick(e(1), t - ft(30.0));
        }
        // E6: retreat to the void edge, then E4 knocks it off (+35 s).
        if t == ft(32.0) && self.players[e(6)].alive {
            let target = self.plan.b_edge.sub(self.plan.b_void_dir.scale(0.2));
            self.players[e(6)].home = target;
            self.players[e(6)].home_radius = 0.6;
            if !self.start_path(e(6), self.plan.b_edge, false) {
                self.players[e(6)].mode = Mode::Retreat { target };
            }
            // The attackers follow E6 to the edge.
            let inner = self.plan.b_edge.sub(self.plan.b_void_dir.scale(2.5));
            for k in [e(4), e(3)] {
                self.players[k].home = inner;
                self.players[k].home_radius = 2.0;
            }
        }
        if t >= ft(35.0) && !self.e6_pushed && self.players[e(6)].alive {
            let six = self.players[e(6)].pos;
            let four = self.players[e(4)].pos;
            let at_edge = six.hdist(self.plan.b_edge) < 1.2;
            if (at_edge && six.hdist(four) < 4.0) || t >= ft(36.0) {
                self.e6_pushed = true;
                self.players[e(6)].allow_void = true;
                self.players[e(6)].last_damage = -100;
                self.players[e(4)].opponent = Some(e(6));
                let dir = self.plan.b_void_dir;
                self.melee_hit(e(4), e(6), false, Some(dir));
                if let Some(kb) = self.players[e(6)].kb_pending.as_mut() {
                    kb.x = dir.x * 0.9;
                    kb.z = dir.z * 0.9;
                }
                self.players[e(6)].mode = Mode::Falling;
                self.event(
                    "void_knock",
                    json!({"victim": self.players[e(6)].name, "by": self.players[e(4)].name}),
                );
            }
        }
        if self.e6_pushed && self.players[e(6)].alive && self.players[e(6)].pos.y < 0.0 {
            self.kill_player(e(6), Some(e(4)), true);
            self.remove_player_body(e(6));
            self.players[e(4)].opponent = None;
            self.players[e(3)].opponent = None;
            self.players[e(4)].mode = Mode::Idle;
            self.players[e(3)].mode = Mode::Idle;
            let side = V3::new(-self.plan.b_void_dir.z, 0.0, self.plan.b_void_dir.x).scale(1.6);
            self.players[e(4)].home = self.grid_snap(self.plan.b_centre.add(side));
            self.players[e(3)].home = self.grid_snap(self.plan.b_centre.sub(side));
        }
        // E5's body leaves 20 ticks after its death (spectator respawn).
        for v in [e(5), e(1)] {
            if !self.players[v].alive
                && !self.players[v].spectator
                && matches!(self.players[v].mode, Mode::Dead)
            {
                let died = self.timeline.iter().rev().find(|ev| {
                    ev["event"] == "death"
                        && ev["detail"]["victim"] == self.players[v].name.as_str()
                });
                if let Some(ev) = died {
                    if t >= ev["tick"].as_i64().unwrap_or(t) + 20 {
                        self.players[v].spectator = true;
                        self.remove_player_body(v);
                    }
                }
            }
        }
        // E11 crosses E9's bridge and engages the viewer from +48 s.
        if t == ft(30.0) {
            let goal = self.plan.anchor.add(self.plan.open_dir.scale(-3.5));
            let goal = find_walkable_near(&self.grid, goal, 3.0).unwrap_or(self.plan.anchor);
            self.players[e(11)].home = goal;
            if !self.start_path(e(11), goal, true) {
                self.event("warning", json!("E11 found no path over the bridge"));
            }
        }
        if t == ft(48.0) && self.players[e(11)].alive {
            self.players[e(11)].opponent = Some(IDX_V);
            self.players[e(11)].mode = Mode::Duel;
            self.players[e(11)].home = self.plan.anchor;
            self.players[e(11)].home_radius = 2.5;
            self.event("opponent_switch", json!({"to": self.players[e(11)].name}));
        }
        // Off-screen deaths (chat, tab, Players left) and one team chat line.
        let offscreen: [(f64, usize, usize, bool); 4] = [
            (5.0, 13 + 1, 13 + 8, false),
            (17.0, 13 + 5, 13 + 9, true),
            (29.0, 13 + 10, 13 + 11, false),
            (52.0, 13 + 3, 13 + 6, false),
        ];
        for (sec, victim, killer, void) in offscreen {
            if t == ft(sec) && self.players[victim].alive {
                self.players[victim].alive = false;
                self.players[victim].spectator = true;
                let (vn, kn) = (
                    self.players[victim].name.clone(),
                    self.players[killer].name.clone(),
                );
                if void {
                    self.chat(&format!("§c{vn} §ewas thrown into the void by §c{kn}§e."));
                } else {
                    self.chat(&format!("§c{vn} §ewas killed by §c{kn}§e."));
                }
                self.players_left_changed();
                self.make_spectator(victim);
            }
        }
        if t == ft(40.0) {
            self.chat("§9[TEAM] §aBenchAlly§f: they are low, push mid");
        }
        // Regen/absorption expiry refreshes the potion colour.
        for i in 0..self.players.len() {
            if self.players[i].scene
                && self.players[i].alive
                && self.players[i].effects.iter().any(|x| x.until == t)
            {
                self.refresh_potion_colour(i);
            }
        }
        // Cosmetic regeneration (keeps the scripted fighters alive; spec deaths are scripted).
        if t % 20 == 10 {
            for i in 0..self.players.len() {
                let p = &self.players[i];
                if !p.scene || !p.alive || p.health >= 20.0 {
                    continue;
                }
                let h = p.health;
                let heal = if h < 8.0 {
                    3.0
                } else if h < 14.0 {
                    2.0
                } else {
                    1.0
                };
                self.set_health(i, (h + heal).min(20.0));
            }
        }
    }

    fn rod_cast(&mut self, i: usize, target: usize, hit: bool) {
        if !self.players[i].alive || !self.players[target].alive {
            return;
        }
        self.hold(i, stack("fishing_rod", 1), 24);
        self.swing(i);
        let from = self.eye(i).sub(V3::new(0.0, 0.1, 0.0));
        let flight = (self.players[target].pos.dist(from) / 1.2).ceil();
        let mut aim = self.players[target]
            .pos
            .add(self.players[target].vel.scale(flight))
            .add(V3::new(0.0, 1.0, 0.0));
        if !hit {
            aim = aim.add(V3::new(1.6, -1.0, -1.2));
        }
        let vel = Self::solve_launch(from, aim, 1.2, 0.04, 0.92);
        let pitch = 0.4 / (self.players[i].rng.range(0.8, 1.2) as f32);
        self.sound("random.bow", self.players[i].pos, 0.5, pitch);
        let mut o = Self::proj_obj(ObjKind::Hook, i, from, vel, 0.04, 0.92);
        o.target = if hit { Some(target) } else { None };
        o.retract_at = self.t + 20;
        self.add_obj(o);
        self.stats.bump("rod_casts");
        self.event("rod_cast", json!({"by": self.players[i].name, "hit": hit}));
    }

    fn pearl(&mut self, i: usize) {
        if !self.players[i].alive {
            return;
        }
        self.hold(i, stack("ender_pearl", 3), 3);
        self.swing(i);
        let from = self.eye(i).sub(V3::new(0.0, 0.1, 0.0));
        let me = self.players[i].pos;
        // Six blocks away on walkable ground, preferring the open side of the anchor.
        let mut landing = self.plan.pearl_landing_hint;
        if landing.hdist(me) < 4.5 || landing.hdist(me) > 8.0 {
            let dir = self.plan.open_dir;
            landing =
                find_walkable_near(&self.grid, me.add(dir.scale(6.0)), 2.5).unwrap_or(landing);
        }
        let vel = Self::solve_launch(from, landing.add(V3::new(0.0, 0.1, 0.0)), 1.5, 0.03, 0.99);
        self.sound("random.bow", me, 0.5, 0.4);
        self.add_obj(Self::proj_obj(ObjKind::Pearl, i, from, vel, 0.03, 0.99));
        self.event(
            "pearl_throw",
            json!({"by": self.players[i].name, "towards": landing.json()}),
        );
    }

    fn regen_splash(&mut self, i: usize, at: usize) {
        if !self.players[i].alive || !self.players[at].alive {
            return;
        }
        let mut potion_item = stack("potion", 1);
        potion_item.damage = POTION_SPLASH_REGEN_II as i16;
        self.hold(i, potion_item, 4);
        self.swing(i);
        let from = self.eye(i).sub(V3::new(0.0, 0.1, 0.0));
        let feet = self.players[at].pos;
        let vel = Self::solve_launch(from, feet.add(V3::new(0.0, 0.05, 0.0)), 0.5, 0.05, 0.99);
        self.sound("random.bow", self.players[i].pos, 0.5, 0.4);
        self.add_obj(Self::proj_obj(
            ObjKind::Potion(POTION_SPLASH_REGEN_II),
            i,
            from,
            vel,
            0.05,
            0.99,
        ));
        self.event(
            "potion_throw",
            json!({"by": self.players[i].name, "at": self.players[at].name}),
        );
    }

    fn gapple_start(&mut self, i: usize) {
        if !self.players[i].alive {
            return;
        }
        let sword = self.players[i].sword.clone();
        self.set_equipment(i, 0, Some(stack("golden_apple", 3)));
        self.players[i].held_restore = None;
        self.set_using(i, true);
        self.players[i].use_until = self.t + 32;
        let _ = sword;
        self.event("gapple_start", json!({"by": self.players[i].name}));
    }

    /// `EntityPlayer.onUpdate` item use: "random.eat" at use counts <= 25 divisible by 4 (sent to
    /// others by `EntityPlayer.playSound`), then `ItemFood.onItemUseFinish` (burp, absorption,
    /// Regeneration II).
    fn gapple_tick(&mut self, i: usize, elapsed: i64) {
        if !self.players[i].alive {
            return;
        }
        let remaining = 32 - elapsed;
        let pos = self.players[i].pos;
        if remaining > 0 && remaining <= 25 && remaining % 4 == 0 {
            let vol = 0.5 + 0.5 * self.players[i].rng.int(0, 1) as f32;
            let pitch = self.players[i].rng.range(0.8, 1.2) as f32;
            self.sound("random.eat", pos, vol, pitch);
        }
        if elapsed == 32 {
            self.set_using(i, false);
            let pitch = self.players[i].rng.range(0.9, 1.0) as f32;
            self.sound("random.burp", pos, 0.5, pitch);
            let t = self.t;
            {
                let p = &mut self.players[i];
                p.absorption = 4.0;
                p.watcher.set(17, MetadataValue::Float(4.0));
                p.effects
                    .retain(|e| e.id != EFFECT_REGENERATION && e.id != EFFECT_ABSORPTION);
                p.effects.push(ActiveEffect {
                    id: EFFECT_REGENERATION,
                    amplifier: 1,
                    until: t + 100,
                });
                p.effects.push(ActiveEffect {
                    id: EFFECT_ABSORPTION,
                    amplifier: 0,
                    until: t + 2_400,
                });
            }
            self.refresh_potion_colour(i);
            self.set_equipment(i, 0, Some(stack("golden_apple", 2)));
            if let Some(sw) = self.players[i].sword.clone() {
                self.players[i].held_restore = Some((t + 2, sw));
            }
            self.event("gapple_eaten", json!({"by": self.players[i].name}));
        }
    }

    // ---------------------------------------------------------------------------------------
    // Main loop
    // ---------------------------------------------------------------------------------------

    fn sample(&mut self) {
        if self.t % 20 != 0 {
            return;
        }
        // Debug dumps to stderr (`SKYWARS_DEBUG_PAIRS`, `SKYWARS_DEBUG`); they never change the
        // generated bytes.
        if std::env::var_os("SKYWARS_DEBUG_PAIRS").is_some() && self.in_fight() {
            let pairs = [
                (IDX_V, idx_e(1)),
                (IDX_T, idx_e(2)),
                (idx_e(3), idx_e(5)),
                (idx_e(4), idx_e(6)),
            ];
            let mut line = format!("t={}", self.t);
            for (a, b) in pairs {
                let pa = &self.players[a];
                let pb = &self.players[b];
                let _ = write!(
                    line,
                    " {}-{}:{:.1}({},{})",
                    pa.role.label(),
                    pb.role.label(),
                    pa.pos.hdist(pb.pos),
                    if pa.on_ground { 'g' } else { 'a' },
                    if pb.on_ground { 'g' } else { 'a' }
                );
            }
            let six = &self.players[idx_e(6)];
            let _ = write!(
                line,
                " E6edge:{:.1} mode={:?}",
                six.pos.hdist(self.plan.b_edge),
                std::mem::discriminant(&six.mode)
            );
            eprintln!("{line}");
        }
        if self.t % 100 == 0 && std::env::var_os("SKYWARS_DEBUG").is_some() {
            for (i, p) in self.players.iter().enumerate() {
                if !p.scene {
                    continue;
                }
                let tracked = self.tracked(i);
                eprintln!(
                    "t={} {:>4} pos=({:.1},{:.1},{:.1}) hp={:.1} alive={} tracked={} mode={:?} opp={:?} dV={:.1}",
                    self.t,
                    p.role.label(),
                    p.pos.x,
                    p.pos.y,
                    p.pos.z,
                    p.health,
                    p.alive,
                    tracked,
                    std::mem::discriminant(&p.mode),
                    p.opponent.map(|j| self.players[j].role.label()),
                    p.pos.hdist(self.players[IDX_V].pos)
                );
            }
        }
        let mut players = 0;
        let mut items = 0;
        let mut xp = 0;
        let mut projectiles = 0;
        let mut other = 0;
        for e in &self.entries {
            if !e.tracked || e.removed {
                continue;
            }
            match e.ent {
                Ent::P(_) => players += 1,
                Ent::O(o) => match self.objs[o].kind {
                    ObjKind::Item => items += 1,
                    ObjKind::XpOrb(_) => xp += 1,
                    ObjKind::Firework => other += 1,
                    _ => projectiles += 1,
                },
            }
        }
        self.stats.tracked_samples.push(json!({
            "tick": self.t, "players": players, "items": items, "xp_orbs": xp,
            "projectiles": projectiles, "other": other,
            "total": players + items + xp + projectiles + other,
        }));
    }

    fn look_keyframes(&mut self) {
        // look_track covers the gap between E1's death and E11's arrival and the tail; the
        // wrapper's `look` ranges (aim_entity) take precedence where they apply.
        let t = self.t;
        let gap = (self.kill_tick_e1..ft(48.0)).contains(&t) || t >= FIGHT_END;
        if gap && t % 5 == 0 {
            let v = self.players[IDX_V].pos.add(V3::new(0.0, 1.62, 0.0));
            let target = if t < ft(48.0) && t < self.kill_tick_e1 + 30 {
                self.players[idx_e(1)].pos.add(V3::new(
                    0.0,
                    2.5 + (t - self.kill_tick_e1) as f64 * 0.15,
                    0.0,
                ))
            } else {
                self.players[idx_e(11)].pos.add(V3::new(0.0, 1.1, 0.0))
            };
            let yaw = yaw_to(v, target);
            let pitch = pitch_to(v, target).clamp(-70.0, 70.0);
            self.look_track
                .push(json!([t, round3(f64::from(yaw)), round3(f64::from(pitch))]));
        }
    }

    fn run(&mut self) {
        for t in 0..TAIL_END {
            self.t = t;
            if t == 0 {
                self.join();
            }
            self.send_chunks();
            if t > 0 && t % 20 == 0 {
                self.time_update();
            }
            self.hud_tick();
            self.events_tick();
            if self.in_fight() {
                self.viewer_hits();
            }
            self.ai_tick();
            for i in 0..self.players.len() {
                self.step_player(i);
            }
            self.step_objects();
            self.tracker_tick();
            self.flush_destroy();
            self.sample();
            self.look_keyframes();
        }
    }
}

// =============================================================================================
// Generation, recipe, stats
// =============================================================================================

struct Generated {
    capture: Capture,
    recipe: Value,
    stats_md: String,
    summary: String,
}

fn generate(
    map_dir: &Path,
    seed: u64,
    local_kb: bool,
    antixray: bool,
) -> Result<Generated, String> {
    let mut map = load_map(map_dir, antixray)?;
    let strip = strip_cages(&mut map.columns, &map.manifest)?;
    let mut terrain = Terrain::new(&map.columns)?;
    let bb = mid_bbox(&map.manifest);
    let bounds = &map.manifest["bounds"];
    let bmin: Vec<i32> = bounds["min"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect())
        .unwrap_or_else(|| vec![bb[0], 0, bb[2]]);
    let bmax: Vec<i32> = bounds["max"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect())
        .unwrap_or_else(|| vec![bb[3], 255, bb[5]]);
    let grid = Grid::build(&terrain, bmin[0] - 4, bmin[2] - 4, bmax[0] + 4, bmax[2] + 4);
    let plan = plan_scene(&mut terrain, &grid, &map.manifest)?;

    // Columns within VIEW_DISTANCE (Chebyshev) of the viewer's chunk, Spigot order.
    let px = (plan.anchor.x as i32) >> 4;
    let pz = (plan.anchor.z as i32) >> 4;
    let mut selected: Vec<Column> = map
        .columns
        .iter()
        .filter(|c| (c.x - px).abs() <= VIEW_DISTANCE && (c.z - pz).abs() <= VIEW_DISTANCE)
        .cloned()
        .collect();
    selected.sort_by(|a, b| spigot_chunk_order(px, pz, (a.x, a.z), (b.x, b.z)));
    let bulks: Vec<Vec<usize>> = (0..selected.len())
        .collect::<Vec<_>>()
        .chunks(MAX_BULK_COLUMNS)
        .map(<[usize]>::to_vec)
        .collect();
    let columns_expected = ((2 * VIEW_DISTANCE + 1) * (2 * VIEW_DISTANCE + 1)) as usize;

    let tiles = tile_entity_packets(&map.manifest)?;
    let mut server = Server::new(&plan, terrain, grid, selected, bulks, tiles, seed, local_kb);
    server.run();

    // Self-checks that would make the capture misleading.
    let mut warnings: Vec<String> = Vec::new();
    for name in &server.sounds_used {
        if sound_event(name).is_none() {
            return Err(format!("sound {name:?} is not a 1.8.9 sound event"));
        }
    }
    if server.stats.columns_sent != columns_expected {
        warnings.push(format!(
            "sent {} columns, a full {}x{} square would be {columns_expected}",
            server.stats.columns_sent,
            2 * VIEW_DISTANCE + 1,
            2 * VIEW_DISTANCE + 1
        ));
    }
    if server.stats.last_block_change_tick_in_join > 20 {
        return Err("block change after tick 20 inside sw_join".into());
    }
    for (name, ok) in [
        ("E5 died", !server.players[idx_e(5)].alive),
        ("E6 died", !server.players[idx_e(6)].alive),
        ("E1 died", !server.players[idx_e(1)].alive),
        ("viewer alive", server.players[IDX_V].alive),
    ] {
        if !ok {
            return Err(format!("choreography check failed: {name}"));
        }
    }

    // ---- capture ------------------------------------------------------------------------
    let mut recorder = Recorder::new(CaptureMeta {
        protocol: PROTOCOL,
        address: format!("synthetic-{SCENE_ID}"),
        started_at_unix_millis: 0,
    });
    let mut last = 0u64;
    for (tick, state, payload) in &server.out {
        let micros = *tick as u64 * TICK_MICROS;
        debug_assert!(micros >= last);
        last = micros;
        recorder.record_at(Direction::Clientbound, *state, micros, payload);
    }
    let capture = recorder.finish();

    // ---- recipe -------------------------------------------------------------------------
    let e1 = server.players[idx_e(1)].eid;
    let e11 = server.players[idx_e(11)].eid;
    let mut keys: Vec<Value> = server
        .jumps
        .iter()
        .map(|&t| json!({"from": t, "to": t + 1, "down": ["jump"]}))
        .collect();
    keys.push(json!({"from": ft(30.0), "to": ft(30.0) + 40, "down": ["playerlist"]}));
    let presses: Vec<Value> = server
        .presses
        .iter()
        .map(|(&t, &c)| json!([t, c]))
        .collect();
    let segments: Vec<Value> = SEGMENTS
        .iter()
        .map(|(n, a, b)| json!({"name": n, "start_tick": a, "end_tick": b}))
        .collect();
    let mut expected = serde_json::Map::new();
    for (s, (name, _, _)) in SEGMENTS.iter().enumerate() {
        let mut m = serde_json::Map::new();
        for (&id, &n) in &server.stats.by_seg[s] {
            m.insert(packet_name(id).to_owned(), json!(n));
        }
        m.insert(
            "_total".into(),
            json!(server.stats.by_seg[s].values().sum::<u64>()),
        );
        m.insert("_bytes".into(), json!(server.stats.bytes_by_seg[s]));
        expected.insert((*name).to_owned(), Value::Object(m));
    }
    let fight_secs = (FIGHT_END - FIGHT_START) as f64 / 20.0;
    let fight_total: u64 = server.stats.by_seg[1].values().sum();
    let entities: Vec<Value> = server
        .players
        .iter()
        .map(|p| {
            json!({
                "name": p.name, "role": p.role.label(), "entity_id": p.eid, "team": p.team,
                "uuid": p.uuid.to_string(), "alive_at_fight_start": !matches!(p.role, Role::Far(n) if n > 13),
                "scene_body": p.scene,
            })
        })
        .collect();
    let e5_death = server
        .timeline
        .iter()
        .find(|ev| {
            ev["event"] == "death"
                && ev["detail"]["victim"] == server.players[idx_e(5)].name.as_str()
        })
        .and_then(|ev| ev["tick"].as_i64())
        .unwrap_or(ft(12.0));
    let screenshots = json!([
        [280, "join_settled"],
        [FIGHT_START + 5, "fight_start"],
        [e5_death + 2, "lightning_kill"],
        [ft(16.0), "mid_fight"],
        [ft(31.0), "tab_held"],
        [server.kill_tick_e1 + 13, "firework_kill"],
        [ft(55.0), "late_fight"],
        [TAIL_END - 10, "tail"]
    ]);
    let recipe = json!({
        "schema": SCHEMA,
        "k": K,
        "lockstep_micros": LOCKSTEP_MICROS,
        "seed": seed,
        "segments": segments,
        "screenshots": screenshots,
        "local": {
            "entity_id": LOCAL_EID,
            "attack_presses": presses,
            "keys": keys,
            "hotbar": [[FIGHT_START, 0]],
            "look": [
                {"from": FIGHT_START, "to": server.kill_tick_e1, "aim_entity": e1},
                {"from": ft(48.0), "to": FIGHT_END, "aim_entity": e11}
            ],
            "look_track": server.look_track,
            "approach": [
                {"from": FIGHT_START, "to": server.kill_tick_e1, "entity": e1, "keep_within": KEEP_WITHIN, "sprint": true},
                {"from": ft(48.0), "to": FIGHT_END, "entity": e11, "keep_within": KEEP_WITHIN, "sprint": true}
            ],
            "notes": "presses are taps; the server reacts to a press at tick T at tick T+1 (S0B 5/4 echo, S19 hurt, S1C/S04, target knockback in its path). Hotbar stays on slot 0 (Sharpness III diamond sword)."
        },
        "scene": SCENE_ID,
        "variant": VARIANT,
        "status": "UNQUALIFIED_DIAGNOSTIC",
        "scoring_eligible": false,
        "scoring_note": "false until the A/A gate (two identical wrapper runs, identical sidecar rows) passes",
        "replay_args": ["--replayCamera", "fixed", "--replayStep", LOCKSTEP_MICROS.to_string(), "--lockstepMicros", LOCKSTEP_MICROS.to_string()],
        "segment_ticks": {"sw_join": [0, FIGHT_START], "sw_fight": [FIGHT_START, FIGHT_END], "sw_tail": [FIGHT_END, TAIL_END]},
        "map": {
            "name": map.name,
            "display_name": map.display_name,
            "chunks_file": map.chunks_file,
            "chunks_sha256": map.sha256,
            "antixray": antixray,
            "view_distance_chebyshev": VIEW_DISTANCE,
            "viewer_chunk": [px, pz],
            "columns_sent": server.stats.columns_sent,
            "sections_sent": server.stats.sections_sent,
            "bulk_packets": server.stats.bulks,
            "bulk_columns_max": MAX_BULK_COLUMNS,
            "last_chunk_tick": server.stats.last_chunk_tick,
            "cage_blocks_stripped": strip.blocks,
            "cage_blocks_stripped_by_block": strip.by_block,
            "cage_sections_dropped": strip.sections_dropped,
            "cages": strip.cages,
            "tile_entity_packets": server.tile_packets,
            "time": {"world_time": -6000, "every_ticks": 20, "weather": "clear"},
        },
        "anchor": plan.anchor_report,
        "placements": plan.placements_report,
        "local_knockback": local_kb,
        "skins": "single-signed",
        "tracking_ranges": {"players": RANGE_PLAYER, "items_xp": RANGE_MISC, "other": RANGE_OTHER},
        "entities": entities,
        "events": server.timeline,
        "expected_packets": Value::Object(expected),
        "fight_packets_per_second": round3(fight_total as f64 / fight_secs),
        "fight_packets_per_second_series": server.stats.fight_per_second,
        "tracked_entities": server.stats.tracked_samples,
        "counters": server.stats.counters,
        "sounds_used": server.sounds_used,
        "warnings": warnings,
        "clocks": {
            "capture": "every packet of server tick T stamped T*50000 us; JoinGame at 0 us; frames in non-decreasing time",
            "wall_time": "not used",
            "rng": "SplitMix64 streams derived from the seed, one per actor",
        },
        "generation": "skywars_fixture.rs: mc-protocol Packet::write, mc-data registry lookups, mc_world chunk decode; EntityTrackerEntry emulation",
    });

    let stats_md = stats_markdown(&server, &plan, &strip, &map, local_kb, &recipe);
    let summary = format!(
        "{SCENE_ID}: {} frames, {} columns ({} sections) in {} bulks, last chunk tick {}, {} cage blocks stripped, fight {:.1} packets/s\n",
        capture.len(),
        server.stats.columns_sent,
        server.stats.sections_sent,
        server.stats.bulks,
        server.stats.last_chunk_tick,
        strip.blocks,
        fight_total as f64 / fight_secs,
    );
    Ok(Generated {
        capture,
        recipe,
        stats_md,
        summary,
    })
}

fn stats_markdown(
    server: &Server<'_>,
    plan: &ScenePlan,
    strip: &StripReport,
    map: &MapInput,
    local_kb: bool,
    recipe: &Value,
) -> String {
    let mut s = String::new();
    let fight_secs = (FIGHT_END - FIGHT_START) as f64 / 20.0;
    let _ = writeln!(s, "# {SCENE_ID} capture statistics\n");
    let _ = writeln!(
        s,
        "Generated by `skywars_fixture.rs` (seed {}, local knockback {}). Map `{}` ({}), `{}` SHA-256 `{}`.\n",
        server.seed,
        if local_kb { "on" } else { "off" },
        map.name,
        map.display_name,
        map.chunks_file,
        map.sha256
    );
    let _ = writeln!(s, "## World\n");
    let _ = writeln!(
        s,
        "- Columns sent: {} ({} sections) in {} packets (S26 of <= {MAX_BULK_COLUMNS}, one per tick), ticks 1-{}.",
        server.stats.columns_sent, server.stats.sections_sent, server.stats.bulks, server.stats.last_chunk_tick
    );
    let _ = writeln!(
        s,
        "- Cage blocks stripped: {} across {} cages ({:?}); sections dropped as empty: {}.",
        strip.blocks, strip.cages, strip.by_block, strip.sections_dropped
    );
    let _ = writeln!(
        s,
        "- Anchor: feet {:?}, {:.1} blocks from the centre, open radius {} (see recipe `anchor`).\n",
        [round3(plan.anchor.x), round3(plan.anchor.y), round3(plan.anchor.z)],
        plan.anchor.hdist(plan.centre),
        recipe["anchor"]["open_radius_blocks"]
    );
    let _ = writeln!(s, "## Packets by segment\n");
    let _ = writeln!(
        s,
        "| packet | sw_join | sw_fight | sw_tail | sw_fight per s |"
    );
    let _ = writeln!(s, "|---|---:|---:|---:|---:|");
    let mut ids: BTreeSet<i32> = BTreeSet::new();
    for seg in &server.stats.by_seg {
        ids.extend(seg.keys().copied());
    }
    for id in &ids {
        let n = |k: usize| server.stats.by_seg[k].get(id).copied().unwrap_or(0);
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {:.2} |",
            packet_name(*id),
            n(0),
            n(1),
            n(2),
            n(1) as f64 / fight_secs
        );
    }
    let totals: Vec<u64> = (0..3)
        .map(|k| server.stats.by_seg[k].values().sum())
        .collect();
    let _ = writeln!(
        s,
        "| **total** | **{}** | **{}** | **{}** | **{:.1}** |\n",
        totals[0],
        totals[1],
        totals[2],
        totals[1] as f64 / fight_secs
    );
    let series = &server.stats.fight_per_second;
    let (min, max) = (
        series.iter().copied().min().unwrap_or(0),
        series.iter().copied().max().unwrap_or(0),
    );
    let _ = writeln!(
        s,
        "Fight window: {:.1} packets/s mean, {min}-{max} per second. Per-second series: {:?}\n",
        totals[1] as f64 / fight_secs,
        series
    );
    let _ = writeln!(s, "## Tracked entities over time (every second)\n");
    let _ = writeln!(
        s,
        "| tick | players | items | xp | projectiles | other | total |"
    );
    let _ = writeln!(s, "|---:|---:|---:|---:|---:|---:|---:|");
    for row in server.stats.tracked_samples.iter().step_by(5) {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} |",
            row["tick"],
            row["players"],
            row["items"],
            row["xp_orbs"],
            row["projectiles"],
            row["other"],
            row["total"]
        );
    }
    let _ = writeln!(s, "\n## Event counters\n");
    for (k, v) in &server.stats.counters {
        let _ = writeln!(s, "- {k}: {v}");
    }
    let c = |k: &str| server.stats.counters.get(k).copied().unwrap_or(0) as f64;
    let _ = writeln!(s, "\n## Comparison with research rates (fight window)\n");
    let _ = writeln!(s, "| quantity | this capture | research | note |");
    let _ = writeln!(s, "|---|---:|---|---|");
    let per = |id: i32| server.stats.by_seg[1].get(&id).copied().unwrap_or(0) as f64 / fight_secs;
    let _ = writeln!(s, "| total inbound packets/s | {:.1} | 150-300 (spec §9, ~10 tracked players) | see deviations |", totals[1] as f64 / fight_secs);
    let _ = writeln!(
        s,
        "| viewer damaging hits/s | {:.2} | 1.5 (pvp §2a) | press-driven, hurt window + reach |",
        c("viewer_hits") / fight_secs
    );
    let _ = writeln!(
        s,
        "| hits on the viewer/s | {:.2} | 1.5 (pvp §2a) | |",
        c("hits_on_viewer") / fight_secs
    );
    let _ = writeln!(
        s,
        "| remote damaging hits/s (all pairs) | {:.2} | 1.5 per fighter each way | 3 melee pairs |",
        c("remote_hits") / fight_secs
    );
    let _ = writeln!(
        s,
        "| S0B swing broadcasts/s | {:.1} | <=5 per fighter | |",
        c("swing_broadcasts") / fight_secs
    );
    let _ = writeln!(
        s,
        "| S15/S16/S17 move+look/s | {:.1} | ~14 per fighter (pvp §2b) | |",
        per(0x15) + per(0x16) + per(0x17)
    );
    let _ = writeln!(
        s,
        "| S18 teleport/s | {:.1} | ~4 per fighter | onGround flips |",
        per(0x18)
    );
    let _ = writeln!(
        s,
        "| S19 head look/s | {:.1} | ~12 per fighter | |",
        per(0x19)
    );
    let _ = writeln!(
        s,
        "| S1C metadata/s | {:.1} | ~6-9 per fighter | sprint, use, health |",
        per(0x1C)
    );
    let _ = writeln!(
        s,
        "| S20 properties/s | {:.1} | ~3 per fighter | sprint toggles |",
        per(0x20)
    );
    let _ = writeln!(
        s,
        "| S04 equipment/s | {:.1} | ~8 per fighter | durability |",
        per(0x04)
    );
    let _ = writeln!(
        s,
        "| S29 sounds/s | {:.1} | 10-20 (real servers) | hurt + step sounds within 16 blocks |",
        per(0x29)
    );
    let _ = writeln!(s, "| S2A particle packets/s | {:.2} | 10-40 trail particles/s | trails: 3 of 8 throwables + E7's arrows; crit/sprint particles are client-generated |", per(0x2A));
    let _ = writeln!(
        s,
        "| chat lines/min | {:.1} | ~10 | |",
        c("chat_lines") / (fight_secs / 60.0)
    );
    s
}

fn recipe_path(output: &Path, suffix: &str) -> Result<PathBuf, String> {
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| "--output must have a filename stem".to_owned())?;
    Ok(output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}{suffix}")))
}

/// Exclusive creation of every output (never overwrites), cleaned up on failure. Adapted from
/// `optimization_fixture.rs::reserve_and_write` to three files.
fn reserve_and_write(files: &[(&Path, &[u8])]) -> Result<(), String> {
    let mut reserved: Vec<(PathBuf, File)> = Vec::new();
    for (path, _) in files {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => reserved.push((path.to_path_buf(), file)),
            Err(error) => {
                for (p, _) in reserved {
                    let _ = fs::remove_file(p);
                }
                if error.kind() == io::ErrorKind::AlreadyExists {
                    return Err(format!(
                        "refusing to overwrite existing output {}",
                        path.display()
                    ));
                }
                return Err(format!("cannot reserve output {}: {error}", path.display()));
            }
        }
    }
    let result = (|| {
        for (k, (_, bytes)) in files.iter().enumerate() {
            reserved[k].1.write_all(bytes)?;
            reserved[k].1.flush()?;
        }
        Ok::<(), io::Error>(())
    })();
    if let Err(error) = result {
        let paths: Vec<PathBuf> = reserved.iter().map(|(p, _)| p.clone()).collect();
        drop(reserved);
        for p in paths {
            let _ = fs::remove_file(p);
        }
        return Err(format!("writing outputs failed: {error}"));
    }
    Ok(())
}

/// Capture limits enforced by `Capture::parse` (study §1.1).
fn check_limits(capture: &Capture, encoded: &[u8]) -> Result<(), String> {
    if capture.len() > 1_000_000 {
        return Err(format!(
            "{} frames exceed the 1,000,000 frame limit",
            capture.len()
        ));
    }
    let payload: usize = capture.frames().iter().map(|f| f.payload.len() + 14).sum();
    if payload > 256 * 1024 * 1024 {
        return Err(format!("{payload} payload bytes exceed 256 MiB"));
    }
    if let Some(f) = capture
        .frames()
        .iter()
        .find(|f| f.payload.len() > 2_097_151)
    {
        return Err(format!(
            "a {} byte frame exceeds 2,097,151",
            f.payload.len()
        ));
    }
    Capture::parse(encoded).map_err(|e| format!("capture does not re-parse: {e:?}"))?;
    Ok(())
}

fn run(args: &Args) -> Result<String, String> {
    let generated = generate(&args.map, args.seed, args.local_knockback, args.antixray)?;
    let report = mc_parity::Replayer::new(&generated.capture).run();
    if !report.is_clean() {
        return Err(format!("Replayer is not clean: {}", report.summary()));
    }
    let bytes = generated
        .capture
        .encode()
        .map_err(|e| format!("encoding capture failed: {e:?}"))?;
    check_limits(&generated.capture, &bytes)?;
    let mut recipe = serde_json::to_vec_pretty(&generated.recipe)
        .map_err(|e| format!("encoding recipe failed: {e}"))?;
    recipe.push(b'\n');
    let recipe_file = recipe_path(&args.output, ".recipe.json")?;
    let stats_file = recipe_path(&args.output, ".stats.md")?;
    if recipe_file == args.output || stats_file == args.output {
        return Err("--output must not end in .recipe.json or .stats.md".into());
    }
    reserve_and_write(&[
        (&args.output, &bytes),
        (&recipe_file, &recipe),
        (&stats_file, generated.stats_md.as_bytes()),
    ])?;
    Ok(format!(
        "{}{}\nwrote {} ({} bytes)\nwrote {}\nwrote {}\nreplayer: {}\n",
        generated.stats_md,
        generated.summary,
        args.output.display(),
        bytes.len(),
        recipe_file.display(),
        stats_file.display(),
        report.summary()
    ))
}

fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1).collect()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

// =============================================================================================
// Tests
// =============================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The exported map lives in the scene's scratch folder; `SKYWARS_MAP_DIR` overrides it.
    /// Missing data fails the test rather than passing vacuously.
    fn map_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("SKYWARS_MAP_DIR") {
            return PathBuf::from(dir);
        }
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../optimization/scratch/skywars-scene-2026-11-05/maps/out/dwarf_fortress");
        assert!(
            dir.join("chunks.bin").is_file(),
            "map export missing at {} (set SKYWARS_MAP_DIR)",
            dir.display()
        );
        dir
    }

    #[test]
    fn generation_is_byte_identical_and_replayer_clean() {
        let dir = map_dir();
        let first = generate(&dir, 1, true, false).expect("first generation");
        let second = generate(&dir, 1, true, false).expect("second generation");
        let a = first.capture.encode().expect("encode first");
        let b = second.capture.encode().expect("encode second");
        assert_eq!(a.len(), b.len(), "capture sizes differ");
        assert!(a == b, "captures differ");
        assert_eq!(
            serde_json::to_vec_pretty(&first.recipe).expect("recipe"),
            serde_json::to_vec_pretty(&second.recipe).expect("recipe"),
            "recipes differ"
        );
        assert_eq!(first.stats_md, second.stats_md);
        let report = mc_parity::Replayer::new(&first.capture).run();
        assert!(report.is_clean(), "{}", report.summary());
        check_limits(&first.capture, &a).expect("limits");
        // Frames are in non-decreasing time and the first Play packet is JoinGame at 0 us.
        let frames = first.capture.frames();
        assert!(frames.windows(2).all(|w| w[0].micros <= w[1].micros));
        let first_play = frames
            .iter()
            .find(|f| f.state == State::Play)
            .expect("play frame");
        assert_eq!(first_play.packet_id(), Some(s2c::JoinGame::ID));
        assert_eq!(first_play.micros, 0);
        // The recipe's local block names the viewer and the two opponents.
        assert_eq!(first.recipe["schema"], SCHEMA);
        assert_eq!(first.recipe["local"]["entity_id"], LOCAL_EID);
        assert_eq!(first.recipe["scoring_eligible"], false);
    }

    #[test]
    fn different_seed_changes_the_capture() {
        let dir = map_dir();
        let a = generate(&dir, 1, true, false)
            .expect("seed 1")
            .capture
            .encode()
            .expect("enc");
        let b = generate(&dir, 2, true, false)
            .expect("seed 2")
            .capture
            .encode()
            .expect("enc");
        assert_ne!(a, b);
    }

    #[test]
    fn spigot_order_is_a_total_order() {
        let mut v: Vec<(i32, i32)> = Vec::new();
        for x in -3..=3 {
            for z in -3..=3 {
                v.push((x, z));
            }
        }
        v.sort_by(|a, b| spigot_chunk_order(0, 0, *a, *b));
        assert_eq!(v[0], (0, 0));
        for w in v.windows(2) {
            assert_eq!(spigot_chunk_order(0, 0, w[0], w[1]), Ordering::Less);
        }
    }

    #[test]
    fn watcher_order_matches_java_hashmap() {
        let w = Watcher::new(WKind::Arrow);
        assert_eq!(w.vanilla_order(), vec![0, 16, 1, 2, 3, 4]);
        let p = Watcher::new(WKind::Player);
        assert_eq!(
            p.vanilla_order(),
            vec![0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 16, 17, 18]
        );
    }

    #[test]
    fn tile_entities_become_s33_and_s35() {
        use base64::Engine as _;
        let mut skull = NbtCompound::new();
        skull.insert("id", Nbt::String("Skull".into()));
        skull.insert("SkullType", Nbt::Byte(3));
        skull.insert("Rot", Nbt::Byte(4));
        let bytes = mc_nbt::to_vec("", &Nbt::Compound(skull.clone())).expect("nbt");
        let manifest = json!({"other_tile_entities": [
            {"id": "Sign", "pos": [-17, 70, 33], "nbt": {"Text1": "{\"text\":\"mid\"}", "Text2": "\"\"", "Text3": "\"\"", "Text4": "\"\""}},
            {"id": "Skull", "pos": [5, 64, -3], "nbt_b64": base64::engine::general_purpose::STANDARD.encode(&bytes)},
            {"id": "Furnace", "pos": [0, 64, 0]}
        ]});
        let packets = tile_entity_packets(&manifest).expect("tiles");
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].0, (-2, 2));
        match &packets[0].1 {
            TilePacket::Sign(p) => assert_eq!(p.lines[0], "{\"text\":\"mid\"}"),
            other => panic!("expected a sign, got {other:?}"),
        }
        match &packets[1].1 {
            TilePacket::Data(p) => {
                assert_eq!(p.action, 4);
                assert_eq!(p.nbt.as_ref(), Some(&skull));
            }
            other => panic!("expected S35, got {other:?}"),
        }
    }

    #[test]
    fn rot256_floors_like_java() {
        assert_eq!(rot256(0.0), 0);
        assert_eq!(rot256(-1.0), -1);
        assert_eq!(rot256(359.0), 255);
        assert_eq!(fix32(-0.01), -1);
    }
}
