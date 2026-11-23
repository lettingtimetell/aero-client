//! The greenhouse, block by block: a barrel-vaulted glass house with white quartz ribs, a
//! diagonal quartz brace overgrown with vines, lush trees and plants, a winding raised quartz
//! walkway with glass railings and sea-lantern lights, and two pools the shader waterfalls land
//! in.
//!
//! Everything here is deterministic: the only randomness is [`hash`] of block coordinates and a
//! fixed salt, so every run (and every machine) builds the same blocks.
//!
//! Coordinates: `+x` is screen-left for the camera, which looks down `+z` (vanilla yaw 0);
//! `+y` is up. The interior spans `x` [`X_MIN`]..=[`X_MAX`] and `z` [`Z_MIN`]..=[`Z_MAX`], and
//! [`FLOOR`] is the first air layer above the grass.

use mc_data::BlockState;

/// The first air layer above the ground; the grass is at `FLOOR - 1`.
pub(super) const FLOOR: i32 = 10;
/// Interior bounds (the glass walls sit one block outside them).
pub(super) const X_MIN: i32 = 0;
pub(super) const X_MAX: i32 = 47;
pub(super) const Z_MIN: i32 = 0;
pub(super) const Z_MAX: i32 = 87;
/// The last layer of the vertical side walls.
const WALL_TOP: i32 = FLOOR + 14;
/// How far the vault rises above [`WALL_TOP`] at the ridge.
const RISE: f32 = 13.0;
/// Rib spacing along `z`.
const RIB_STEP: i32 = 8;

/// The voxel bounds the scene may touch (inclusive).
pub(super) const VOX_MIN: [i32; 3] = [-2, 0, -2];
pub(super) const VOX_MAX: [i32; 3] = [49, 47, 89];

/// One shader waterfall: a square column of falling water above a round pool.
#[derive(Debug, Clone, Copy)]
pub(super) struct Fall {
    /// Column centre.
    pub(super) x: f32,
    pub(super) z: f32,
    /// Half the column's width.
    pub(super) half: f32,
    /// Where the water leaves the spout.
    pub(super) top: f32,
    /// The pool's water surface.
    pub(super) pool_y: f32,
    /// The pool's water radius.
    pub(super) pool_r: f32,
    /// Per-fall noise offset.
    pub(super) seed: f32,
}

/// A dense block grid over [`VOX_MIN`]..=[`VOX_MAX`].
pub(super) struct Voxels {
    cells: Vec<BlockState>,
}

impl Voxels {
    const SX: i32 = VOX_MAX[0] - VOX_MIN[0] + 1;
    const SY: i32 = VOX_MAX[1] - VOX_MIN[1] + 1;
    const SZ: i32 = VOX_MAX[2] - VOX_MIN[2] + 1;

    fn new() -> Self {
        Self {
            cells: vec![BlockState::AIR; (Self::SX * Self::SY * Self::SZ) as usize],
        }
    }

    fn index(x: i32, y: i32, z: i32) -> Option<usize> {
        let (lx, ly, lz) = (x - VOX_MIN[0], y - VOX_MIN[1], z - VOX_MIN[2]);
        if lx < 0 || ly < 0 || lz < 0 || lx >= Self::SX || ly >= Self::SY || lz >= Self::SZ {
            return None;
        }
        Some(((ly * Self::SZ + lz) * Self::SX + lx) as usize)
    }

    /// The block at a position; air outside the grid.
    pub(super) fn get(&self, x: i32, y: i32, z: i32) -> BlockState {
        Self::index(x, y, z).map_or(BlockState::AIR, |i| self.cells[i])
    }

    fn set(&mut self, x: i32, y: i32, z: i32, s: BlockState) {
        if let Some(i) = Self::index(x, y, z) {
            self.cells[i] = s;
        }
    }

    fn set_if_air(&mut self, x: i32, y: i32, z: i32, s: BlockState) -> bool {
        if self.get(x, y, z) == BlockState::AIR {
            self.set(x, y, z, s);
            true
        } else {
            false
        }
    }

    fn fill(&mut self, from: [i32; 3], to: [i32; 3], s: BlockState) {
        for y in from[1].min(to[1])..=from[1].max(to[1]) {
            for z in from[2].min(to[2])..=from[2].max(to[2]) {
                for x in from[0].min(to[0])..=from[0].max(to[0]) {
                    self.set(x, y, z, s);
                }
            }
        }
    }

    /// Whether any block in the 16³ section `(sx, sy, sz)` is not air.
    pub(super) fn section_occupied(&self, sx: i32, sy: i32, sz: i32) -> bool {
        for y in sy * 16..sy * 16 + 16 {
            for z in sz * 16..sz * 16 + 16 {
                for x in sx * 16..sx * 16 + 16 {
                    if self.get(x, y, z) != BlockState::AIR {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// The block state for a registry name and metadata.
fn st(name: &str, meta: u8) -> BlockState {
    let block = mc_data::block_by_name(name)
        .unwrap_or_else(|| panic!("{name} is not a registered 1.8.9 block"));
    BlockState::new(block, meta)
}

/// A deterministic 32-bit hash of a block position and a salt (a lowbias32 finaliser over a
/// mixed key).
pub(super) fn hash(x: i32, y: i32, z: i32, salt: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F)
        ^ salt.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// [`hash`] as a float in `0..1`.
pub(super) fn rand01(x: i32, y: i32, z: i32, salt: u32) -> f32 {
    (hash(x, y, z, salt) >> 8) as f32 / 16_777_216.0
}

/// Smooth 2-D value noise in `0..1` with a cell size of `scale` blocks.
fn value_noise(x: f32, z: f32, scale: f32, salt: u32) -> f32 {
    let (fx, fz) = (x / scale, z / scale);
    let (ix, iz) = (fx.floor() as i32, fz.floor() as i32);
    let (tx, tz) = (fx - fx.floor(), fz - fz.floor());
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sz) = (s(tx), s(tz));
    let n = |a: i32, b: i32| rand01(a, 0, b, salt);
    let top = n(ix, iz) + (n(ix + 1, iz) - n(ix, iz)) * sx;
    let bottom = n(ix, iz + 1) + (n(ix + 1, iz + 1) - n(ix, iz + 1)) * sx;
    top + (bottom - top) * sz
}

/// The palette, resolved once per build.
struct Palette {
    grass: BlockState,
    dirt: BlockState,
    quartz: BlockState,
    quartz_pillar_y: BlockState,
    quartz_chiseled: BlockState,
    glass: BlockState,
    pane: BlockState,
    sea_lantern: BlockState,
    prismarine_bricks: BlockState,
    quartz_stairs: u16,
    vine: u16,
    tallgrass: u16,
    double_plant: u16,
    red_flower: u16,
    yellow_flower: u16,
    leaves: u16,
    leaves2: u16,
    log: u16,
    log2: u16,
}

impl Palette {
    fn new() -> Self {
        let id = |name: &str| mc_data::block_by_name(name).expect(name).0;
        Self {
            grass: st("minecraft:grass", 0),
            dirt: st("minecraft:dirt", 0),
            quartz: st("minecraft:quartz_block", 0),
            quartz_pillar_y: st("minecraft:quartz_block", 2),
            quartz_chiseled: st("minecraft:quartz_block", 1),
            glass: st("minecraft:glass", 0),
            pane: st("minecraft:glass_pane", 0),
            sea_lantern: st("minecraft:sea_lantern", 0),
            prismarine_bricks: st("minecraft:prismarine", 1),
            quartz_stairs: id("minecraft:quartz_stairs"),
            vine: id("minecraft:vine"),
            tallgrass: id("minecraft:tallgrass"),
            double_plant: id("minecraft:double_plant"),
            red_flower: id("minecraft:red_flower"),
            yellow_flower: id("minecraft:yellow_flower"),
            leaves: id("minecraft:leaves"),
            leaves2: id("minecraft:leaves2"),
            log: id("minecraft:log"),
            log2: id("minecraft:log2"),
        }
    }

    fn of(id: u16, meta: u8) -> BlockState {
        BlockState::new(mc_data::BlockId(id), meta)
    }
}

/// A wood type: the log block and variant, and the leaves block and variant.
#[derive(Clone, Copy)]
enum Wood {
    Oak,
    Birch,
    Jungle,
    DarkOak,
    Acacia,
}

impl Wood {
    fn log(self, p: &Palette, axis: u8) -> BlockState {
        match self {
            Wood::Oak => Palette::of(p.log, axis),
            Wood::Birch => Palette::of(p.log, 2 | axis),
            Wood::Jungle => Palette::of(p.log, 3 | axis),
            Wood::Acacia => Palette::of(p.log2, axis),
            Wood::DarkOak => Palette::of(p.log2, 1 | axis),
        }
    }

    fn leaves(self, p: &Palette) -> BlockState {
        // `| 4`: no decay. Rendering only reads the variant.
        match self {
            Wood::Oak => Palette::of(p.leaves, 4),
            Wood::Birch => Palette::of(p.leaves, 2 | 4),
            Wood::Jungle => Palette::of(p.leaves, 3 | 4),
            Wood::Acacia => Palette::of(p.leaves2, 4),
            Wood::DarkOak => Palette::of(p.leaves2, 1 | 4),
        }
    }
}

/// The finished scene: the blocks, the waterfalls, and the walkway's centre line (which the
/// camera follows).
pub(super) struct Greenhouse {
    pub(super) voxels: Voxels,
    pub(super) falls: Vec<Fall>,
}

/// The vault's inner surface height at column `x`.
fn vault_top(x: i32) -> i32 {
    let centre = (X_MIN + X_MAX) as f32 * 0.5;
    let half = (X_MAX - X_MIN) as f32 * 0.5 + 1.0;
    let u = ((x as f32 - centre) / half).clamp(-1.0, 1.0);
    WALL_TOP + (RISE * (1.0 - u * u).max(0.0).sqrt()).round() as i32
}

/// The walkway's centre line, as a Catmull-Rom spline through these `(x, z)` points.
const PATH: [[f32; 2]; 11] = [
    [21.0, -6.0],
    [21.0, 1.0],
    [23.0, 9.0],
    [28.0, 17.0],
    [30.0, 26.0],
    [26.5, 36.0],
    [20.0, 45.0],
    [17.5, 56.0],
    [19.5, 66.0],
    [24.0, 76.0],
    [27.0, 92.0],
];

/// The walkway's half-width.
const PATH_HALF: f32 = 2.4;

/// Sample the walkway spline densely: `(x, z, arc length)`.
fn path_samples() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    let mut length = 0.0f32;
    let mut last: Option<[f32; 2]> = None;
    for seg in 0..PATH.len() - 1 {
        let p0 = PATH[seg.saturating_sub(1)];
        let p1 = PATH[seg];
        let p2 = PATH[seg + 1];
        let p3 = PATH[(seg + 2).min(PATH.len() - 1)];
        for step in 0..64 {
            let t = step as f32 / 64.0;
            let (t2, t3) = (t * t, t * t * t);
            let c = |a: f32, b: f32, c: f32, d: f32| {
                0.5 * (2.0 * b
                    + (-a + c) * t
                    + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                    + (-a + 3.0 * b - 3.0 * c + d) * t3)
            };
            let p = [c(p0[0], p1[0], p2[0], p3[0]), c(p0[1], p1[1], p2[1], p3[1])];
            if let Some(q) = last {
                length += ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
            }
            last = Some(p);
            out.push([p[0], p[1], length]);
        }
    }
    out
}

/// Distance from `(x, z)` to the walkway centre line, with the arc length of the nearest sample.
fn path_distance(samples: &[[f32; 3]], x: f32, z: f32) -> (f32, f32) {
    let mut best = (f32::MAX, 0.0);
    for s in samples {
        let d = (s[0] - x).powi(2) + (s[1] - z).powi(2);
        if d < best.0 {
            best = (d, s[2]);
        }
    }
    (best.0.sqrt(), best.1)
}

/// Build the whole scene.
pub(super) fn build() -> Greenhouse {
    let p = Palette::new();
    let mut v = Voxels::new();

    ground(&mut v, &p);
    let samples = path_samples();
    let falls = pools(&mut v, &p);
    walkway(&mut v, &p, &samples);
    shell(&mut v, &p);
    braces(&mut v, &p);
    trees(&mut v, &p, &samples, &falls);
    ground_cover(&mut v, &p, &samples, &falls);

    Greenhouse { voxels: v, falls }
}

fn ground(v: &mut Voxels, p: &Palette) {
    v.fill(
        [VOX_MIN[0], FLOOR - 3, VOX_MIN[2]],
        [VOX_MAX[0], FLOOR - 2, VOX_MAX[2]],
        p.dirt,
    );
    v.fill(
        [VOX_MIN[0], FLOOR - 1, VOX_MIN[2]],
        [VOX_MAX[0], FLOOR - 1, VOX_MAX[2]],
        p.grass,
    );
}

/// The glass walls, the far end wall, the vault, its quartz ribs, ridge and purlins.
fn shell(v: &mut Voxels, p: &Palette) {
    let (xw0, xw1) = (X_MIN - 1, X_MAX + 1);
    let z_far = Z_MAX + 1;
    // The vault profile, including the walls (columns xw0 and xw1 reach down to the floor).
    let top = |x: i32| -> i32 {
        if x <= xw0 || x >= xw1 {
            WALL_TOP
        } else {
            vault_top(x)
        }
    };
    for x in xw0..=xw1 {
        let t = top(x);
        // Close the surface: fill down to one above the lower neighbour so no gap shows.
        let lo = if x == xw0 || x == xw1 {
            FLOOR
        } else {
            (top(x - 1).min(top(x + 1)) + 1).min(t)
        };
        for z in Z_MIN - 1..=z_far {
            for y in lo..=t {
                v.set(x, y, z, p.glass);
            }
        }
        // The far end wall under the vault.
        for y in FLOOR..t {
            v.set(x, y, z_far, p.glass);
        }
    }
    // Ribs: a quartz arch every RIB_STEP blocks, one block thick, with pillar quartz down the
    // walls; plus the ridge and two purlins along the length.
    let mut z = Z_MIN + 2;
    while z <= z_far {
        for x in xw0..=xw1 {
            let t = top(x);
            let lo = if x == xw0 || x == xw1 {
                FLOOR
            } else {
                (top(x - 1).min(top(x + 1)) + 1).min(t)
            };
            for y in lo..=t {
                let wall = x == xw0 || x == xw1;
                v.set(x, y, z, if wall { p.quartz_pillar_y } else { p.quartz });
            }
            // A flange one block inside the arch near the springing, where it reads as a rib.
            if !(x == xw0 || x == xw1) && (x - xw0 <= 6 || xw1 - x <= 6) {
                v.set(x, lo - 1, z, p.quartz);
            }
        }
        // Inner pilasters at the wall feet.
        for y in FLOOR..=WALL_TOP {
            v.set(xw0 + 1, y, z, p.quartz_pillar_y);
            v.set(xw1 - 1, y, z, p.quartz_pillar_y);
        }
        z += RIB_STEP;
    }
    let centre = (X_MIN + X_MAX) / 2;
    for x in [centre, centre + 1, X_MIN + 7, X_MAX - 7] {
        let t = top(x);
        for z in Z_MIN - 1..=z_far {
            v.set(x, t, z, p.quartz);
        }
    }
    // A cornice at the wall tops.
    for z in Z_MIN - 1..=z_far {
        v.set(xw0, WALL_TOP, z, p.quartz);
        v.set(xw1, WALL_TOP, z, p.quartz);
        v.set(xw0, FLOOR, z, p.quartz);
        v.set(xw1, FLOOR, z, p.quartz);
    }
    for x in xw0..=xw1 {
        v.set(x, FLOOR, z_far, p.quartz);
    }
}

/// A great diagonal quartz brace on the left, rising from the near foreground (where it stands
/// in a grass bed) along the walkway to the vault, in the `y`-`z` plane. Stairs smooth its top
/// and bottom edges; vines climb the face it shows the walkway.
fn braces(v: &mut Voxels, p: &Palette) {
    // Centre line y - z = C: from (z = 4, y = FLOOR) up toward the ridge.
    let c = FLOOR - 4;
    let (x0, x1) = (35, 37);
    for z in VOX_MIN[2]..=Z_MAX {
        for y in FLOOR..=VOX_MAX[1] {
            let s = y - z;
            if (c - 1..=c + 1).contains(&s) {
                for x in x0..=x1 {
                    if y < vault_top(x) {
                        v.set(x, y, z, p.quartz);
                    }
                }
            }
        }
    }
    // Stairs: the upper edge (y - z = C + 2) has the beam to its south, so they face south;
    // the lower edge (y - z = C - 2) gets upside-down stairs facing north.
    // (facing: 0 east, 1 west, 2 south, 3 north; 4 = upside down.)
    for z in VOX_MIN[2]..=Z_MAX {
        for x in x0..=x1 {
            let y_up = c + 2 + z;
            if y_up >= FLOOR && y_up < vault_top(x) {
                v.set_if_air(x, y_up, z, Palette::of(p.quartz_stairs, 2));
            }
            let y_dn = c - 2 + z;
            if y_dn >= FLOOR && y_dn < vault_top(x) {
                v.set_if_air(x, y_dn, z, Palette::of(p.quartz_stairs, 3 | 4));
            }
        }
    }
    // Vines on the side facing the walkway (x0 - 1), clinging to the brace (east bit), in
    // patches.
    for z in VOX_MIN[2]..=Z_MAX {
        for y in FLOOR..=VOX_MAX[1] {
            if !(c - 2..=c + 2).contains(&(y - z)) || v.get(x0, y, z) == BlockState::AIR {
                continue;
            }
            let n = value_noise(z as f32, y as f32 * 1.2, 4.0, 77);
            if n > 0.4 {
                v.set_if_air(x0 - 1, y, z, Palette::of(p.vine, 8));
                if rand01(x0, y, z, 5) < 0.35 {
                    v.set_if_air(x0 - 1, y - 1, z, Palette::of(p.vine, 8));
                }
            }
        }
    }
}

/// The walkway: a raised quartz path with glass-pane railings and sea lanterns set into its
/// edges.
fn walkway(v: &mut Voxels, p: &Palette, samples: &[[f32; 3]]) {
    for z in VOX_MIN[2]..=Z_MAX {
        for x in X_MIN..=X_MAX {
            let (d, arc) = path_distance(samples, x as f32 + 0.5, z as f32 + 0.5);
            if d >= PATH_HALF {
                continue;
            }
            let edge = d >= PATH_HALF - 1.0;
            let light = edge && (arc as i32).rem_euclid(5) == 0;
            v.set(
                x,
                FLOOR,
                z,
                if light {
                    p.sea_lantern
                } else if edge {
                    p.quartz_chiseled
                } else {
                    p.quartz
                },
            );
            if edge {
                v.set(x, FLOOR + 1, z, p.pane);
            }
        }
    }
    // The railing stops where the walkway leaves the house, so the near end is open.
    for z in VOX_MIN[2]..Z_MIN {
        for x in X_MIN..=X_MAX {
            if v.get(x, FLOOR + 1, z) == p.pane {
                v.set(x, FLOOR + 1, z, BlockState::AIR);
            }
        }
    }
}

/// Two round pools with quartz rims; returns the waterfalls above them.
fn pools(v: &mut Voxels, p: &Palette) -> Vec<Fall> {
    let mut falls = Vec::new();
    for (i, (cx, cz)) in [(37.5f32, 44.5f32), (11.5, 48.5)].into_iter().enumerate() {
        let r = 3.6f32;
        for z in (cz - r - 2.0) as i32..=(cz + r + 2.0) as i32 {
            for x in (cx - r - 2.0) as i32..=(cx + r + 2.0) as i32 {
                let d = ((x as f32 + 0.5 - cx).powi(2) + (z as f32 + 0.5 - cz).powi(2)).sqrt();
                if d < r - 0.6 {
                    // Water: dig to a prismarine floor two below.
                    v.set(x, FLOOR - 1, z, BlockState::AIR);
                    v.set(x, FLOOR - 2, z, p.prismarine_bricks);
                } else if d < r + 0.5 {
                    v.set(x, FLOOR, z, p.quartz);
                    v.set(x, FLOOR - 1, z, p.quartz);
                }
            }
        }
        let half = 0.85;
        let top = vault_top(cx as i32) as f32 - 1.0;
        // The spout: a quartz collar under the vault with a sea-lantern ring.
        let (ix, iz) = (cx.floor() as i32, cz.floor() as i32);
        for y in top as i32 - 1..=vault_top(ix) {
            for z in iz - 2..=iz + 2 {
                for x in ix - 2..=ix + 2 {
                    let ring = (x - ix).abs() == 2 || (z - iz).abs() == 2;
                    if ring {
                        let lantern =
                            y == top as i32 - 1 && ((x - ix).abs() + (z - iz).abs()) % 2 == 0;
                        v.set(x, y, z, if lantern { p.sea_lantern } else { p.quartz });
                    }
                }
            }
        }
        falls.push(Fall {
            x: cx,
            z: cz,
            half,
            top: top - 1.0,
            pool_y: FLOOR as f32 + 0.65,
            pool_r: r - 0.5,
            seed: 17.3 * (i as f32 + 1.0),
        });
    }
    falls
}

/// Smooth 3-D value noise in `0..1` with a cell size of `scale` blocks.
fn value_noise3(x: f32, y: f32, z: f32, scale: f32, salt: u32) -> f32 {
    let (fx, fy, fz) = (x / scale, y / scale, z / scale);
    let (ix, iy, iz) = (fx.floor() as i32, fy.floor() as i32, fz.floor() as i32);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (tx, ty, tz) = (s(fx - fx.floor()), s(fy - fy.floor()), s(fz - fz.floor()));
    let mut acc = 0.0;
    for (dz, wz) in [(0, 1.0 - tz), (1, tz)] {
        for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
            for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                acc += wx * wy * wz * rand01(ix + dx, iy + dy, iz + dz, salt);
            }
        }
    }
    acc
}

/// A blob of leaves: an ellipsoid whose surface is pushed in and out by smooth noise (lumps a
/// few blocks across) plus a little per-block roughness.
fn canopy(v: &mut Voxels, c: [f32; 3], r: [f32; 3], leaves: BlockState, salt: u32) {
    let lo = [
        c[0] - r[0] * 1.3 - 1.0,
        c[1] - r[1] * 1.3 - 1.0,
        c[2] - r[2] * 1.3 - 1.0,
    ];
    let hi = [
        c[0] + r[0] * 1.3 + 1.0,
        c[1] + r[1] * 1.3 + 1.0,
        c[2] + r[2] * 1.3 + 1.0,
    ];
    for y in lo[1] as i32..=hi[1] as i32 {
        for z in lo[2] as i32..=hi[2] as i32 {
            for x in lo[0] as i32..=hi[0] as i32 {
                let (px, py, pz) = (x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let d = ((px - c[0]) / r[0]).powi(2)
                    + ((py - c[1]) / r[1]).powi(2)
                    + ((pz - c[2]) / r[2]).powi(2);
                let lump = value_noise3(px, py, pz, 2.6, salt) - 0.5;
                let edge = 1.0 + lump * 0.75 - 0.18 * rand01(x, y, z, salt ^ 0x55);
                if d < edge {
                    v.set_if_air(x, y, z, leaves);
                }
            }
        }
    }
}

/// A line of logs (bark on all sides when diagonal).
fn branch(v: &mut Voxels, from: [f32; 3], to: [f32; 3], log: BlockState) {
    let d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    let n = d[0].abs().max(d[1].abs()).max(d[2].abs()).ceil().max(1.0) as i32;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, y, z) = (
            (from[0] + d[0] * t).floor() as i32,
            (from[1] + d[1] * t).floor() as i32,
            (from[2] + d[2] * t).floor() as i32,
        );
        let s = v.get(x, y, z);
        if s == BlockState::AIR || is_leaves(s) {
            v.set(x, y, z, log);
        }
    }
}

fn is_leaves(s: BlockState) -> bool {
    matches!(
        mc_data::block_info(s.block()).name,
        "minecraft:leaves" | "minecraft:leaves2"
    )
}

/// Hang vines down the outside of a canopy.
fn hang_vines(v: &mut Voxels, p: &Palette, lo: [i32; 3], hi: [i32; 3], salt: u32, chance: f32) {
    // facing bit for "the vine clings to the block on this side": south 1, west 2, north 4,
    // east 8 (the leaf is on that side of the vine cell).
    let dirs: [([i32; 2], u8); 4] = [([0, -1], 1), ([1, 0], 2), ([0, 1], 4), ([-1, 0], 8)];
    for y in (lo[1]..=hi[1]).rev() {
        for z in lo[2]..=hi[2] {
            for x in lo[0]..=hi[0] {
                if !is_leaves(v.get(x, y, z)) {
                    continue;
                }
                for (k, (d, bit)) in dirs.iter().enumerate() {
                    let (vx, vz) = (x + d[0], z + d[1]);
                    if v.get(vx, y, vz) != BlockState::AIR
                        || rand01(x, y, z, salt + k as u32) > chance
                    {
                        continue;
                    }
                    let len = 2 + (hash(x, y, z, salt + 9) % 9) as i32;
                    for j in 0..len {
                        let yy = y - j;
                        if yy <= FLOOR + 1 || !v.set_if_air(vx, yy, vz, Palette::of(p.vine, *bit)) {
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// One tree. `trunk` is the trunk's height above the floor; `crown` the canopy radius.
#[allow(clippy::too_many_arguments)]
fn tree(
    v: &mut Voxels,
    p: &Palette,
    wood: Wood,
    x: i32,
    z: i32,
    trunk: i32,
    crown: f32,
    thick: bool,
    vines: f32,
) {
    let salt = hash(x, 0, z, 991);
    let log = wood.log(p, 0);
    let bark = wood.log(p, 12);
    let leaves = wood.leaves(p);
    let span = if thick { 1 } else { 0 };
    for y in FLOOR..FLOOR + trunk {
        for dz in 0..=span {
            for dx in 0..=span {
                v.set(x + dx, y, z + dz, log);
            }
        }
    }
    // Root flare on the big trunks.
    if thick {
        for (dx, dz) in [(-1, 0), (2, 1), (0, -1), (1, 2), (-1, 1), (2, 0)] {
            if rand01(x + dx, 1, z + dz, salt) < 0.7 {
                v.set_if_air(x + dx, FLOOR, z + dz, bark);
            }
        }
    }
    let base = [
        x as f32 + 0.5 + span as f32 * 0.5,
        (FLOOR + trunk) as f32,
        z as f32 + 0.5 + span as f32 * 0.5,
    ];
    let tall = matches!(wood, Wood::Birch);
    let squash = if tall { 1.25 } else { 0.78 };
    // A main crown, a domed top, and satellite blobs on branches at varied heights.
    canopy(
        v,
        [base[0], base[1] + crown * 0.15, base[2]],
        [crown, crown * squash, crown],
        leaves,
        salt,
    );
    canopy(
        v,
        [base[0], base[1] + crown * 0.75, base[2]],
        [crown * 0.6, crown * 0.5 * squash, crown * 0.6],
        leaves,
        salt ^ 0x9E37,
    );
    let n = 5 + (salt % 3) as usize;
    for i in 0..n {
        let a = (i as f32 / n as f32) * std::f32::consts::TAU + (salt % 100) as f32 * 0.07;
        let reach = crown * (0.7 + 0.35 * rand01(i as i32, 1, 0, salt));
        let tip = [
            base[0] + a.cos() * reach,
            base[1] - crown * 0.45 + rand01(i as i32, 2, 0, salt) * crown * 0.9,
            base[2] + a.sin() * reach,
        ];
        branch(v, [base[0], base[1] - crown * 0.4, base[2]], tip, bark);
        let r = crown * (0.42 + 0.25 * rand01(i as i32, 3, 0, salt));
        canopy(
            v,
            tip,
            [r, r * 0.8 * squash, r],
            leaves,
            salt + i as u32 + 1,
        );
    }
    if vines > 0.0 {
        let c = crown.ceil() as i32 + 3;
        hang_vines(
            v,
            p,
            [base[0] as i32 - c, FLOOR + 2, base[2] as i32 - c],
            [base[0] as i32 + c, base[1] as i32 + c, base[2] as i32 + c],
            salt,
            vines,
        );
    }
}

/// A low bush: a log stub under a small leaf blob.
fn bush(v: &mut Voxels, p: &Palette, wood: Wood, x: i32, z: i32, r: f32) {
    let salt = hash(x, 1, z, 313);
    v.set_if_air(x, FLOOR, z, wood.log(p, 0));
    canopy(
        v,
        [x as f32 + 0.5, FLOOR as f32 + r * 0.45, z as f32 + 0.5],
        [r, r * 0.62, r],
        wood.leaves(p),
        salt,
    );
}

fn trees(v: &mut Voxels, p: &Palette, samples: &[[f32; 3]], falls: &[Fall]) {
    // (wood, x, z, trunk, crown, thick, vines)
    let big: [(Wood, i32, i32, i32, f32, bool, f32); 16] = [
        // Foreground framing.
        (Wood::Oak, 42, 8, 11, 6.0, true, 0.10),
        (Wood::DarkOak, 4, 11, 10, 7.0, true, 0.12),
        (Wood::Oak, 41, 21, 8, 4.0, false, 0.12),
        (Wood::Jungle, 7, 20, 12, 4.8, true, 0.28),
        // Mid ground.
        (Wood::Jungle, 5, 30, 17, 5.5, true, 0.16),
        (Wood::Jungle, 41, 31, 15, 5.0, true, 0.30),
        (Wood::Birch, 8, 38, 10, 3.2, false, 0.0),
        (Wood::Oak, 31, 55, 12, 6.0, true, 0.15),
        (Wood::Jungle, 4, 50, 18, 5.5, true, 0.35),
        (Wood::Acacia, 43, 58, 10, 5.0, false, 0.10),
        (Wood::Oak, 44, 46, 9, 4.5, false, 0.15),
        // Far.
        (Wood::Jungle, 25, 70, 19, 6.5, true, 0.30),
        (Wood::Oak, 8, 72, 12, 6.0, true, 0.15),
        (Wood::Birch, 40, 74, 10, 3.8, false, 0.0),
        (Wood::DarkOak, 13, 84, 10, 6.0, true, 0.10),
        (Wood::Jungle, 38, 84, 18, 6.0, true, 0.30),
    ];
    for (wood, x, z, trunk, crown, thick, vines) in big {
        tree(v, p, wood, x, z, trunk, crown, thick, vines);
    }
    // Bushes along the walkway and around the pools, wherever they do not block either.
    let mut placed = 0;
    'outer: for z in (Z_MIN + 1..Z_MAX).step_by(2) {
        for x in (X_MIN + 1..X_MAX).step_by(2) {
            let jx = x + (hash(x, 3, z, 41) % 2) as i32;
            let jz = z + (hash(x, 4, z, 41) % 2) as i32;
            let (d, _) = path_distance(samples, jx as f32 + 0.5, jz as f32 + 0.5);
            let near_pool = falls.iter().any(|f| {
                ((jx as f32 + 0.5 - f.x).powi(2) + (jz as f32 + 0.5 - f.z).powi(2)).sqrt()
                    < f.pool_r + 2.5
            });
            if !(PATH_HALF + 1.2..=PATH_HALF + 6.0).contains(&d) || near_pool {
                continue;
            }
            if rand01(jx, 5, jz, 43) < 0.16 {
                let wood = match hash(jx, 6, jz, 47) % 4 {
                    0 => Wood::Birch,
                    1 => Wood::Jungle,
                    _ => Wood::Oak,
                };
                bush(v, p, wood, jx, jz, 1.2 + rand01(jx, 7, jz, 53) * 0.9);
                placed += 1;
                if placed > 60 {
                    break 'outer;
                }
            }
        }
    }
}

/// Grass, ferns, flowers and tall double plants on every free grass block.
fn ground_cover(v: &mut Voxels, p: &Palette, samples: &[[f32; 3]], falls: &[Fall]) {
    for z in Z_MIN..=Z_MAX {
        for x in X_MIN..=X_MAX {
            if v.get(x, FLOOR - 1, z) != p.grass || v.get(x, FLOOR, z) != BlockState::AIR {
                continue;
            }
            let (d, _) = path_distance(samples, x as f32 + 0.5, z as f32 + 0.5);
            let wet = falls.iter().any(|f| {
                ((x as f32 + 0.5 - f.x).powi(2) + (z as f32 + 0.5 - f.z).powi(2)).sqrt()
                    < f.pool_r + 4.0
            });
            let shade = v.get(x, FLOOR + 6, z) != BlockState::AIR
                || v.get(x, FLOOR + 9, z) != BlockState::AIR;
            let lush = value_noise(x as f32, z as f32, 6.0, 101);
            let flowers = value_noise(x as f32, z as f32, 4.0, 202);
            let r = rand01(x, FLOOR, z, 303);
            let r2 = rand01(x, FLOOR, z, 304);
            // Leave a clear verge of short grass right beside the walkway.
            let verge = d < PATH_HALF + 0.9;
            let plant = |id: u16, meta: u8| Palette::of(id, meta);
            if verge {
                if r < 0.35 {
                    v.set(x, FLOOR, z, plant(p.tallgrass, 1));
                }
                continue;
            }
            if flowers > 0.62 && r < 0.55 {
                let pick = hash(x, 9, z, 305) % 9;
                let s = match pick {
                    0 => plant(p.yellow_flower, 0),
                    1 => plant(p.red_flower, 1), // blue orchid
                    2 => plant(p.red_flower, 2), // allium
                    3 => plant(p.red_flower, 3), // azure bluet
                    4 => plant(p.red_flower, 8), // oxeye daisy
                    5 => plant(p.red_flower, 6), // white tulip
                    6 => plant(p.red_flower, 7), // pink tulip
                    7 => plant(p.red_flower, 0), // poppy
                    _ => plant(p.red_flower, 1),
                };
                v.set(x, FLOOR, z, s);
                continue;
            }
            if r2 < 0.10 + 0.12 * lush && v.get(x, FLOOR + 1, z) == BlockState::AIR {
                // Double plants: large fern in shade and by water, else grass, roses, peonies.
                let variant = if shade || wet {
                    3
                } else {
                    match hash(x, 8, z, 306) % 6 {
                        0 => 4,
                        1 => 5,
                        2 => 1,
                        _ => 2,
                    }
                };
                v.set(x, FLOOR, z, plant(p.double_plant, variant));
                v.set(x, FLOOR + 1, z, plant(p.double_plant, 8));
                continue;
            }
            let density = 0.55 + 0.4 * lush;
            if r < density {
                let fern = (shade || wet) && r2 < 0.75 || r2 < 0.18;
                v.set(x, FLOOR, z, plant(p.tallgrass, if fern { 2 } else { 1 }));
            }
        }
    }
}
