//! Tile-entity renderers: chest, ender chest, sign, banner, skull, beacon.
//!
//! java: `client/renderer/tileentity/` — `TileEntityChestRenderer`,
//! `TileEntityEnderChestRenderer`, `TileEntitySignRenderer`, `TileEntityBannerRenderer`,
//! `TileEntitySkullRenderer`, `TileEntityBeaconRenderer`.
//!
//! These are not entities: they have no yaw, no limb swing and no interpolation, and they are
//! drawn from block state and a small amount of tile data. What each one really needs from a
//! port is its **placement** — the exact matrix that turns a block position and a metadata
//! value into a model transform — plus the one or two animated numbers it has. That is what
//! this module is; the box geometry itself is in [`crate::entity::models::tileent`].
//!
//! All of them share the same flip as living entities: `scale(1, -1, -1)` (chest) or
//! `scale(-1, -1, 1)` (skull), because the models are authored Y-down.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::context::{Gpu, DEPTH_FORMAT};
use crate::frame::Camera;
use crate::pipeline::{PassTarget, RenderPhase};

use super::models::ModelKind;

#[cfg(test)]
use super::EntityVertex;

/// Which tile entity is being drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TileKind {
    /// java: `TileEntityChestRenderer` with `chestBasic`
    Chest,
    /// java: `TileEntityChestRenderer` with `chestTrap`
    TrappedChest,
    /// java: `TileEntityChestRenderer`'s double-chest branch
    LargeChest,
    /// java: `TileEntityChestRenderer`'s double-chest branch with `getChestType() == 1`.
    ///
    /// `TileEntityChestRenderer.java:14-19` declares **six** chest textures and
    /// `:66-113` selects among them on `(double, christmas, chestType)`. Without this variant a
    /// double *trapped* chest could only resolve to [`TileKind::LargeChest`] and would bind
    /// `normal_double.png` — visually identical to an ordinary double chest, which is a real if
    /// minor gameplay tell.
    LargeTrappedChest,
    /// java: `TileEntityEnderChestRenderer`
    EnderChest,
    /// java: `TileEntitySignRenderer`
    Sign,
    /// java: `TileEntityBannerRenderer`
    Banner,
    /// java: `TileEntitySkullRenderer`
    Skull,
    /// java: `TileEntityBeaconRenderer`
    Beacon,
}

impl TileKind {
    /// The model this renderer draws, if it draws a box model.
    ///
    /// The beacon has none — it is two textured shells generated per frame, see
    /// [`BeaconBeam`]. The skull's model depends on its variant ([`SkullVariant::model`]).
    #[must_use]
    pub fn model(self) -> Option<ModelKind> {
        match self {
            Self::Chest | Self::TrappedChest | Self::EnderChest => Some(ModelKind::Chest),
            Self::LargeChest | Self::LargeTrappedChest => Some(ModelKind::LargeChest),
            Self::Sign => Some(ModelKind::Sign),
            Self::Banner => Some(ModelKind::Banner),
            Self::Skull => Some(ModelKind::SkeletonHead),
            Self::Beacon => None,
        }
    }

    /// The texture this renderer binds, where it is fixed.
    #[must_use]
    pub fn texture(self) -> Option<&'static str> {
        match self {
            Self::Chest => Some("textures/entity/chest/normal.png"),
            Self::TrappedChest => Some("textures/entity/chest/trapped.png"),
            Self::LargeChest => Some("textures/entity/chest/normal_double.png"),
            Self::LargeTrappedChest => Some("textures/entity/chest/trapped_double.png"),
            Self::EnderChest => Some("textures/entity/chest/ender.png"),
            Self::Sign => Some("textures/entity/sign.png"),
            Self::Banner => Some("textures/entity/banner_base.png"),
            Self::Beacon => Some("textures/entity/beacon_beam.png"),
            Self::Skull => None,
        }
    }

    /// The chest kind for `(trapped, double)`. java: the `getChestType() == 1` fork
    /// `TileEntityChestRenderer` takes twice, once per model.
    #[must_use]
    pub fn chest(trapped: bool, double: bool) -> TileKind {
        match (trapped, double) {
            (false, false) => Self::Chest,
            (true, false) => Self::TrappedChest,
            (false, true) => Self::LargeChest,
            (true, true) => Self::LargeTrappedChest,
        }
    }
}

/// The Christmas chest textures, which override the normal/trapped pair on December 24-26.
///
/// java: `TileEntityChestRenderer`'s constructor (`:24-31`) —
/// `calendar.get(2) + 1 == 12 && calendar.get(5) >= 24 && calendar.get(5) <= 26`. Taken as an
/// explicit `(month, day)` rather than read from the clock so it is testable and so the caller
/// samples the calendar once at startup, exactly as the Java constructor does.
#[must_use]
pub fn is_christmas(month: u32, day: u32) -> bool {
    month == 12 && (24..=26).contains(&day)
}

/// The chest texture, including the two Christmas variants [`TileKind::texture`] cannot reach.
///
/// java: `TileEntityChestRenderer.renderTileEntityAt:66-113` — the `destroyStage >= 0` /
/// `isChristmas` / `getChestType() == 1` / else chain, run once for the single model and once
/// for the double one.
#[must_use]
pub fn chest_texture(kind: TileKind, christmas: bool) -> &'static str {
    match (kind, christmas) {
        (TileKind::Chest | TileKind::TrappedChest, true) => "textures/entity/chest/christmas.png",
        (TileKind::LargeChest | TileKind::LargeTrappedChest, true) => {
            "textures/entity/chest/christmas_double.png"
        }
        (kind, _) => kind.texture().unwrap_or("textures/entity/chest/normal.png"),
    }
}

/// The chest lid's X rotation, in radians.
///
/// java: `TileEntityChestRenderer.renderTileEntityAt` —
/// `f = 1 - f; f = 1 - f*f*f; chestLid.rotateAngleX = -(f * PI / 2)`. The cubic is what gives
/// the lid its snap: it is nearly fully open a third of the way through the animation.
/// `neighbour_lid` mirrors the renderer's `adjacentChestZNeg`/`adjacentChestXNeg` probes.  The
/// normal double-chest draw is the half with neither of those neighbours, so its slice is empty;
/// synchronization comes from both `TileEntityChest` halves receiving the same block event and
/// ticking their own lid state.  Keep the max rule because it is still the exact Java helper for
/// an incomplete/malformed adjacency graph and for the direct render seam.
#[must_use]
pub fn chest_lid_angle(prev_lid: f32, lid: f32, partial_ticks: f32, neighbour_lid: &[f32]) -> f32 {
    let mut f = prev_lid + (lid - prev_lid) * partial_ticks;
    for &n in neighbour_lid {
        if n > f {
            f = n;
        }
    }
    f = 1.0 - f;
    f = 1.0 - f * f * f;
    -(f * core::f32::consts::PI / 2.0)
}

/// The chest's model matrix.
///
/// java: `translate(x, y + 1, z + 1)`, `scale(1, -1, -1)`, `translate(0.5, 0.5, 0.5)`,
/// `rotate(yaw, Y)`, `translate(-0.5, -0.5, -0.5)`. `yaw` is `0` for a chest facing south,
/// `90` west, `180` north, `270` east — java derives it from `EnumFacing` in a `switch`.
///
/// `double_offset` is the extra `translate(1, 0, 0)` or `translate(0, 0, -1)` a large chest's
/// far half gets.
#[must_use]
pub fn chest_matrix(rel: Vec3, yaw_degrees: f32, double_offset: Vec3) -> Mat4 {
    Mat4::from_translation(rel + Vec3::new(0.0, 1.0, 1.0))
        * Mat4::from_scale(Vec3::new(1.0, -1.0, -1.0))
        * Mat4::from_translation(Vec3::new(0.5, 0.5, 0.5))
        * Mat4::from_translation(double_offset)
        * Mat4::from_rotation_y(yaw_degrees.to_radians())
        * Mat4::from_translation(Vec3::new(-0.5, -0.5, -0.5))
}

/// Scale shared by the sign and the banner. java: `float f = 0.6666667F`
pub const SIGN_SCALE: f32 = 0.666_666_7;

/// The standing-sign / wall-sign matrix.
///
/// java: `TileEntitySignRenderer.renderTileEntityAt`. A standing sign turns by
/// `metadata * 360 / 16` degrees; a wall sign snaps to one of four facings and is pushed back
/// by `translate(0, -0.3125, -0.4375)` so it hangs flat against the block behind it.
#[must_use]
pub fn sign_matrix(rel: Vec3, standing: bool, metadata: i32) -> Mat4 {
    if standing {
        let yaw = (metadata * 360) as f32 / 16.0;
        Mat4::from_translation(rel + Vec3::new(0.5, 0.75 * SIGN_SCALE, 0.5))
            * Mat4::from_rotation_y((-yaw).to_radians())
    } else {
        let yaw = wall_facing_degrees(metadata);
        Mat4::from_translation(rel + Vec3::new(0.5, 0.75 * SIGN_SCALE, 0.5))
            * Mat4::from_rotation_y((-yaw).to_radians())
            * Mat4::from_translation(Vec3::new(0.0, -0.3125, -0.4375))
    }
}

/// The standing-banner / wall-banner matrix.
///
/// java: `TileEntityBannerRenderer.renderTileEntityAt` — the same shape as the sign except a
/// wall banner sits at `y - 0.25 * f` rather than `y + 0.75 * f`.
#[must_use]
pub fn banner_matrix(rel: Vec3, standing: bool, metadata: i32) -> Mat4 {
    if standing {
        let yaw = (metadata * 360) as f32 / 16.0;
        Mat4::from_translation(rel + Vec3::new(0.5, 0.75 * SIGN_SCALE, 0.5))
            * Mat4::from_rotation_y((-yaw).to_radians())
    } else {
        let yaw = wall_facing_degrees(metadata);
        Mat4::from_translation(rel + Vec3::new(0.5, -0.25 * SIGN_SCALE, 0.5))
            * Mat4::from_rotation_y((-yaw).to_radians())
            * Mat4::from_translation(Vec3::new(0.0, -0.3125, -0.4375))
    }
}

/// The four wall facings a wall sign/banner metadata maps to, in degrees.
///
/// java: the `if (k == 2) f2 = 180; if (k == 4) f2 = 90; if (k == 5) f2 = -90;` chain — note
/// metadata 3 falls through to `0`, and so does anything unexpected.
#[must_use]
pub fn wall_facing_degrees(metadata: i32) -> f32 {
    match metadata {
        2 => 180.0,
        4 => 90.0,
        5 => -90.0,
        _ => 0.0,
    }
}

/// The five skull types a `TileEntitySkull` can hold.
///
/// java: `TileEntitySkullRenderer.renderSkull`'s `switch (p_180543_6_)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkullVariant {
    /// java: case 0
    Skeleton,
    /// java: case 1
    WitherSkeleton,
    /// java: case 2 — uses the humanoid head model.
    Zombie,
    /// java: case 3 — uses the humanoid head model and the profile's own skin.
    Player,
    /// java: case 4
    Creeper,
}

impl SkullVariant {
    /// Parses the tile entity's stored type byte, saturating on anything unexpected the way
    /// vanilla's `default` arm does.
    #[must_use]
    pub fn from_id(id: u8) -> Self {
        match id {
            1 => Self::WitherSkeleton,
            2 => Self::Zombie,
            3 => Self::Player,
            4 => Self::Creeper,
            _ => Self::Skeleton,
        }
    }

    /// Which head model this variant uses.
    ///
    /// java: only cases 2 and 3 swap `modelbase` to `humanoidHead`; the creeper's head uses the
    /// *skeleton* model even though it binds the creeper texture.
    #[must_use]
    pub fn model(self) -> ModelKind {
        match self {
            Self::Zombie | Self::Player => ModelKind::HumanoidHead,
            _ => ModelKind::SkeletonHead,
        }
    }

    /// The texture, or `None` for a player head, whose texture is the player's skin.
    #[must_use]
    pub fn texture(self) -> Option<&'static str> {
        match self {
            Self::Skeleton => Some("textures/entity/skeleton/skeleton.png"),
            Self::WitherSkeleton => Some("textures/entity/skeleton/wither_skeleton.png"),
            Self::Zombie => Some("textures/entity/zombie/zombie.png"),
            Self::Creeper => Some("textures/entity/creeper/creeper.png"),
            Self::Player => None,
        }
    }
}

/// Where a skull is attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkullMount {
    /// On the floor — free rotation from the tile's own rotation field.
    Floor,
    /// On the north face of the block behind it.
    North,
    /// On the south face.
    South,
    /// On the west face.
    West,
    /// On the east face.
    East,
}

/// The skull's matrix and the yaw handed to the model.
///
/// java: `TileEntitySkullRenderer.renderSkull` — a wall skull is offset to `0.74`/`0.26` on the
/// mounting axis and sits at `y + 0.25`, and its yaw is **overwritten** by the facing rather
/// than added to it, so a wall skull ignores the rotation field entirely. A floor skull keeps
/// its rotation and sits at `y` with no lift.
#[must_use]
pub fn skull_transform(rel: Vec3, mount: SkullMount, rotation_degrees: f32) -> (Mat4, f32) {
    let (offset, yaw) = match mount {
        SkullMount::Floor => (Vec3::new(0.5, 0.0, 0.5), rotation_degrees),
        SkullMount::North => (Vec3::new(0.5, 0.25, 0.74), rotation_degrees),
        SkullMount::South => (Vec3::new(0.5, 0.25, 0.26), 180.0),
        SkullMount::West => (Vec3::new(0.74, 0.25, 0.5), 270.0),
        SkullMount::East => (Vec3::new(0.26, 0.25, 0.5), 90.0),
    };
    let m = Mat4::from_translation(rel + offset) * Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0));
    (m, yaw)
}

impl SkullMount {
    /// Which face a skull's metadata mounts it on.
    ///
    /// java: `TileEntitySkullRenderer.renderTileEntityAt:32` —
    /// `EnumFacing.getFront(te.getBlockMetadata() & 7)`, then `renderSkull`'s switch. `UP` (1) is
    /// the floor case; `NORTH`(2), `SOUTH`(3), `WEST`(4) are named and **everything else**,
    /// including `DOWN`(0), falls through the `EAST`/`default` arm — which is why this saturates
    /// rather than returning `Option`. This conversion had no caller.
    #[must_use]
    pub fn from_meta(metadata: u8) -> SkullMount {
        match metadata & 7 {
            1 => SkullMount::Floor,
            2 => SkullMount::North,
            3 => SkullMount::South,
            4 => SkullMount::West,
            _ => SkullMount::East,
        }
    }
}

/// The skull's yaw in degrees from its stored rotation nibble.
///
/// java: `TileEntitySkullRenderer.renderTileEntityAt:32` — `(float)(te.getSkullRotation() * 360)
/// / 16.0F`. A wall skull's [`skull_transform`] overwrites it; a floor skull keeps it.
#[must_use]
pub fn skull_rotation_degrees(rotation: u8) -> f32 {
    f32::from(u16::from(rotation) * 360) / 16.0
}

/// The banner cloth's X rotation, in radians.
///
/// java: `TileEntityBannerRenderer.java:64-66` —
/// `f3 = (posX*7 + posY*9 + posZ*13) + totalWorldTime + partialTicks;`
/// `bannerSlate.rotateAngleX = (-0.0125F + 0.01F * cos(f3 * PI * 0.02F)) * PI`.
///
/// The position hash is the whole point: it desynchronises adjacent banners so a row of them
/// ripples instead of moving in lockstep. A grep for `0.0125` across the tree
/// found nothing before this.
///
/// `world_time` is `World.getTotalWorldTime()`, or `0` for a banner drawn without a world (the
/// `flag` fork at `TileEntityBannerRenderer.java:29`, which is how the item-form banner is
/// drawn perfectly still).
#[must_use]
pub fn banner_slate_angle(x: i32, y: i32, z: i32, world_time: i64, partial_ticks: f32) -> f32 {
    let hash = x * 7 + y * 9 + z * 13;
    let f3 = hash as f32 + world_time as f32 + partial_ticks;
    (-0.012_5 + 0.01 * mc_math::cos(f3 * core::f32::consts::PI * 0.02)) * core::f32::consts::PI
}

/// Scale of the sign/banner **model** under [`sign_matrix`]. java: `scale(f, -f, -f)`.
#[must_use]
pub fn sign_model_matrix(base: Mat4) -> Mat4 {
    base * Mat4::from_scale(Vec3::new(SIGN_SCALE, -SIGN_SCALE, -SIGN_SCALE))
}

/// Text scale on a sign face. java: `float f3 = 0.015625F * f`.
pub const SIGN_TEXT_SCALE: f32 = 0.015_625 * SIGN_SCALE;

/// The matrix the four lines of sign text are laid out under.
///
/// java: `TileEntitySignRenderer.java:104-108` — after the board is drawn, the stack is
/// `translate(0, 0.5F * f, 0.07F * f)` then `scale(f3, -f3, f3)`, and `drawString` then works in
/// **GUI pixels** with `y = j * 10 - signText.length * 5` per line. The `-f3` on Y is what makes
/// GUI-space "down" come out as world "down".
///
/// `base` is [`sign_matrix`]'s result.
#[must_use]
pub fn sign_text_matrix(base: Mat4) -> Mat4 {
    base * Mat4::from_translation(Vec3::new(0.0, 0.5 * SIGN_SCALE, 0.07 * SIGN_SCALE))
        * Mat4::from_scale(Vec3::new(
            SIGN_TEXT_SCALE,
            -SIGN_TEXT_SCALE,
            SIGN_TEXT_SCALE,
        ))
}

/// The baseline `y` of sign line `line` in the text matrix's GUI-pixel space.
///
/// java: `j * 10 - te.signText.length * 5` with `signText.length == 4`, i.e. `line * 10 - 20`.
#[must_use]
pub fn sign_line_y(line: usize) -> f32 {
    line as f32 * 10.0 - 20.0
}

/// The mob spawner's miniature-entity matrix.
///
/// java: `TileEntityMobSpawnerRenderer.renderTileEntityAt:13-16` plus `renderMob:26-34` —
/// `translate(x + 0.5, y, z + 0.5)`, `translate(0, 0.4, 0)`,
/// `rotate(interpolated_rotation * 10, Y)`, `rotate(-30, X)`, `translate(0, -0.4, 0)`,
/// `scale(0.4375)`.
#[must_use]
pub fn spawner_matrix(rel: Vec3, prev_rotation: f64, rotation: f64, partial_ticks: f32) -> Mat4 {
    let spin = (prev_rotation + (rotation - prev_rotation) * f64::from(partial_ticks)) as f32;
    Mat4::from_translation(rel + Vec3::new(0.5, 0.0, 0.5))
        * Mat4::from_translation(Vec3::new(0.0, 0.4, 0.0))
        * Mat4::from_rotation_y((spin * 10.0).to_radians())
        * Mat4::from_rotation_x((-30.0f32).to_radians())
        * Mat4::from_translation(Vec3::new(0.0, -0.4, 0.0))
        * Mat4::from_scale(Vec3::splat(0.437_5))
}

/// The enchantment table book's matrix.
///
/// java: `TileEntityEnchantmentTableRenderer.java:18-37` — `translate(x + 0.5, y + 0.75,
/// z + 0.5)`, the `0.1 + sin(tickCount * 0.1) * 0.01` bob, the **shortest-arc** interpolation of
/// `bookRotation` (which is why the two angles are unwrapped here rather than lerped naively),
/// then `rotate(-f2 * 180 / PI, Y)` and `rotate(80, Z)`.
#[must_use]
pub fn enchant_book_matrix(
    rel: Vec3,
    tick_count: i32,
    prev_rotation: f32,
    rotation: f32,
    partial_ticks: f32,
) -> Mat4 {
    let f = tick_count as f32 + partial_ticks;
    let mut delta = rotation - prev_rotation;
    while delta >= core::f32::consts::PI {
        delta -= core::f32::consts::PI * 2.0;
    }
    while delta < -core::f32::consts::PI {
        delta += core::f32::consts::PI * 2.0;
    }
    let angle = prev_rotation + delta * partial_ticks;
    Mat4::from_translation(rel + Vec3::new(0.5, 0.75, 0.5))
        * Mat4::from_translation(Vec3::new(0.0, 0.1 + mc_math::sin(f * 0.1) * 0.01, 0.0))
        * Mat4::from_rotation_y(-angle)
        * Mat4::from_rotation_z(80.0f32.to_radians())
}

/// The two page-flip phases `ModelBook` is posed with.
///
/// java: `TileEntityEnchantmentTableRenderer.java:39-62` — `frac(pageFlip + 0.25) * 1.6 - 0.3`
/// and the same with `+ 0.75`, each clamped to `0..1`. Vanilla writes the truncation as
/// `f3 - (float) MathHelper.truncateDoubleToInt(f3)`. Despite its name, that helper is the
/// shifted cast `(int)(value + 1024.0D) - 1024`; use its exact mc-math port rather than Rust's
/// truncation-toward-zero or a merely common-range `floor` approximation. `pageFlip` is an
/// unbounded signed random walk, so negative phases are reachable in normal play.
#[must_use]
pub fn enchant_page_flips(prev_flip: f32, flip: f32, partial_ticks: f32) -> (f32, f32) {
    let base = prev_flip + (flip - prev_flip) * partial_ticks;
    let phase = |offset: f32| {
        let v = base + offset;
        let whole = mc_math::truncate_f64_to_i32(f64::from(v)) as f32;
        ((v - whole) * 1.6 - 0.3).clamp(0.0, 1.0)
    };
    (phase(0.25), phase(0.75))
}

/// One beacon beam shell: four quads around the block column.
///
/// java: `TileEntityBeaconRenderer.renderTileEntityAt` draws two shells per beam segment — an
/// **inner** one whose four corners orbit the block centre at radius `0.2` and whose V range is
/// stretched `2.5x`, drawn opaque while the Java code issues a dead blend-function setup, and an **outer** static
/// `0.2..0.8` box at alpha `0.125` with straight alpha blending. Both scroll downward with
/// world time, which is where the beam's "rising" look comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeaconBeam {
    /// The four rotating corners of the inner shell, as `(x, z)` offsets inside the block.
    pub inner_corners: [[f64; 2]; 4],
    /// Inner shell V at the segment's base and top.
    pub inner_v: [f64; 2],
    /// Outer shell V at the segment's base and top.
    pub outer_v: [f64; 2],
    /// Alpha of the outer shell. java: `0.125F`
    pub outer_alpha: f32,
}

/// Builds a beam segment's animated geometry.
///
/// `world_time` is `World.getTotalWorldTime()`, `height` the segment's block height, and
/// `beam_alpha` the `shouldBeamRender()` fade in `0..1`.
#[must_use]
pub fn beacon_beam(
    world_time: i64,
    partial_ticks: f32,
    height: i32,
    beam_alpha: f32,
) -> BeaconBeam {
    let d0 = world_time as f64 + f64::from(partial_ticks);
    // java: MathHelper.func_181162_h(-d0 * 0.2 - floor(-d0 * 0.1)) — the fractional part.
    let d1 = mc_math::frac_f64(-d0 * 0.2 - f64::from(mc_math::floor_f64(-d0 * 0.1)));
    let d2 = d0 * 0.025 * -1.5;
    let r = 0.2;
    // java: the four corners are at d2 + 3PI/4, PI/4, 5PI/4, 7PI/4.
    let angles = [
        2.356_194_490_192_345,
        core::f64::consts::FRAC_PI_4,
        3.926_990_816_987_241_4,
        5.497_787_143_782_138,
    ];
    let inner_corners = core::array::from_fn(|i| {
        let a = d2 + angles[i];
        [0.5 + a.cos() * r, 0.5 + a.sin() * r]
    });
    let base = -1.0 + d1;
    BeaconBeam {
        inner_corners,
        inner_v: [base, f64::from(height as f32 * beam_alpha) * 2.5 + base],
        outer_v: [base, f64::from(height as f32 * beam_alpha) + base],
        outer_alpha: 0.125,
    }
}

// ---------------------------------------------------------------------------------------------
// Beacon vertex emission and GPU pipeline
// ---------------------------------------------------------------------------------------------
//
// java: `TileEntityBeaconRenderer.renderTileEntityAt`, read past its own doc comment above — the
// two shells' actual `GlStateManager` calls are, in draw order:
//
// inner:  `disableBlend()`, `depthMask(true)`,  `tryBlendFuncSeparate(770, 1, 1, 0)` (dead: blend
//         is off so the func never applies) — an **opaque**, alpha-tested draw that writes depth.
// outer:  `enableBlend()`,  `depthMask(false)`, `tryBlendFuncSeparate(770, 771, 1, 0)` — standard
//         straight-alpha blending (`SrcAlpha`/`OneMinusSrcAlpha`) that does not write depth.
//
// Depth *testing* is never touched by either shell, so both stay depth-tested against the world
// the whole time — only the write mask changes. [`BeaconBeam`]'s own doc calls the inner shell
// "additive"; that describes how the bright gradient texture *reads* once alpha-tested and drawn
// opaquely, not an additive blend mode — there is no additive blend function ever enabled here.

/// One vertex of beacon-beam geometry: position, the scrolling texture's UV, and the segment's
/// RGBA tint. java: `DefaultVertexFormats.POSITION_TEX_COLOR`; the inner shell uses alpha `1.0F`
/// while the outer shell uses [`BeaconBeam::outer_alpha`] (`0.125F`) before the shared
/// `alphaFunc(516, 0.1F)` test.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct BeaconVertex {
    /// Camera-relative position, in blocks.
    pub pos: [f32; 3],
    /// Texture coordinate. `v` regularly leaves `0..1` (the inner shell stretches to `2.5x` the
    /// segment height), which is why vanilla sets `GL_REPEAT` on both axes — see
    /// [`BeaconPhase::new`]'s sampler.
    pub uv: [f32; 2],
    /// Straight RGBA.
    pub color: [f32; 4],
}

/// The vertex layout [`BeaconVertex`] is fed to [`BeaconPhase`]'s pipelines with.
pub const BEACON_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: core::mem::size_of::<BeaconVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 12,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 20,
            shader_location: 2,
        },
    ],
};

/// Builds one quad's six vertices (a 0-1-2/0-2-3 fan) from the corner pair `(a, b)`.
///
/// java: the shared per-quad vertex pattern every one of `renderTileEntityAt`'s sixteen
/// `worldrenderer.pos(...)` calls follows — top of `a`, bottom of `a`, bottom of `b`, top of `b`,
/// each `u` fixed at `1` for `a` and `0` for `b`.
#[allow(clippy::too_many_arguments)]
fn push_shell_quad(
    xz: [f32; 2],
    y_bottom: f32,
    y_top: f32,
    v_bottom: f32,
    v_top: f32,
    a: [f64; 2],
    b: [f64; 2],
    color: [f32; 4],
    out: &mut Vec<BeaconVertex>,
) {
    let pa = [xz[0] + a[0] as f32, xz[1] + a[1] as f32];
    let pb = [xz[0] + b[0] as f32, xz[1] + b[1] as f32];
    let quad = [
        BeaconVertex {
            pos: [pa[0], y_top, pa[1]],
            uv: [1.0, v_top],
            color,
        },
        BeaconVertex {
            pos: [pa[0], y_bottom, pa[1]],
            uv: [1.0, v_bottom],
            color,
        },
        BeaconVertex {
            pos: [pb[0], y_bottom, pb[1]],
            uv: [0.0, v_bottom],
            color,
        },
        BeaconVertex {
            pos: [pb[0], y_top, pb[1]],
            uv: [0.0, v_top],
            color,
        },
    ];
    for &i in &[0usize, 1, 2, 0, 2, 3] {
        out.push(quad[i]);
    }
}

/// Emits the inner shell's 24 vertices (4 quads).
///
/// `xz` is the camera-relative `(x, z)` of the tile entity's block (i.e. its unrotated, un-
/// offset corner — [`BeaconBeam::inner_corners`] already carries the `0.5 + ...` that centres
/// each corner inside the block). `y_bottom`/`y_top` are the segment's `i`/`k` world heights,
/// already made camera-relative by the caller. `color` is the segment's `getColors()` RGB.
///
/// java: the four `worldrenderer.pos` groups after `depthMask(true)` — connected in the order
/// `(corner0, corner1)`, `(corner3, corner2)`, `(corner1, corner3)`, `(corner2, corner0)`, which
/// is *not* the plain `0-1-2-3` loop the outer shell uses (see
/// [`emit_beacon_outer_shell`]) — transcribed exactly, not "fixed", because that is what the
/// orbiting corners actually connect to in the Java.
pub fn emit_beacon_inner_shell(
    xz: [f32; 2],
    y_bottom: f32,
    y_top: f32,
    beam: &BeaconBeam,
    color: [f32; 3],
    out: &mut Vec<BeaconVertex>,
) {
    let v_bottom = beam.inner_v[0] as f32;
    let v_top = beam.inner_v[1] as f32;
    let c = beam.inner_corners;
    // java: every rotating inner-shell vertex is emitted with alpha `1.0F` before the first
    // tessellator draw. Only the later static outer shell uses `0.125F`.
    let rgba = [color[0], color[1], color[2], 1.0];
    for &(a, b) in &[(0usize, 1usize), (3, 2), (1, 3), (2, 0)] {
        push_shell_quad(xz, y_bottom, y_top, v_bottom, v_top, c[a], c[b], rgba, out);
    }
}

/// Emits the outer shell's 24 vertices (4 quads) — a static `0.2..0.8` box, alpha
/// [`BeaconBeam::outer_alpha`].
///
/// java: the four `worldrenderer.pos` groups after `depthMask(false)`, which connect the static
/// corners `(0.2,0.2)`, `(0.8,0.2)`, `(0.8,0.8)`, `(0.2,0.8)` in the plain `0-1`, `2-3`, `1-2`,
/// `3-0` loop order — unlike the inner shell, this one *is* a simple untwisted box because the
/// four corners never move relative to each other.
pub fn emit_beacon_outer_shell(
    xz: [f32; 2],
    y_bottom: f32,
    y_top: f32,
    beam: &BeaconBeam,
    color: [f32; 3],
    out: &mut Vec<BeaconVertex>,
) {
    let v_bottom = beam.outer_v[0] as f32;
    let v_top = beam.outer_v[1] as f32;
    let corners: [[f64; 2]; 4] = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]];
    let rgba = [color[0], color[1], color[2], beam.outer_alpha];
    for &(a, b) in &[(0usize, 1usize), (2, 3), (1, 2), (3, 0)] {
        push_shell_quad(
            xz, y_bottom, y_top, v_bottom, v_top, corners[a], corners[b], rgba, out,
        );
    }
}

const BEACON_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

// java: `GlStateManager.alphaFunc(516, 0.1F)` — GL_GREATER, ref 0.1, shared by both shells.
@fragment
fn fs_inner(in: VsOut) -> @location(0) vec4<f32> {
    let s = textureSample(tex, samp, in.uv) * in.color;
    if (s.a < 0.1) {
        discard;
    }
    // Opaque: blending is disabled for this shell (`GlStateManager.disableBlend()`).
    return vec4<f32>(s.rgb, 1.0);
}

@fragment
fn fs_outer(in: VsOut) -> @location(0) vec4<f32> {
    let s = textureSample(tex, samp, in.uv) * in.color;
    if (s.a < 0.1) {
        discard;
    }
    return s;
}
"#;

/// java: `TileEntityBeaconRenderer` binds `textures/entity/beacon_beam.png` through
/// `TextureUtil`, whose `GL_RGBA` allocation preserves the uploaded bytes on sampling.
const JAVA_BEACON_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One uploaded texture, mirroring [`super::render::EntityPhase`]'s bind-group bookkeeping.
struct BeaconTexture {
    bind_group: wgpu::BindGroup,
}

/// The depth portion of one beacon shell's immutable pipeline descriptor.
///
/// java: `Minecraft` establishes `glDepthFunc(GL_LEQUAL)` globally and
/// `TileEntityBeaconRenderer` leaves it unchanged for both shells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BeaconDepthDescriptor {
    write_depth: bool,
    compare: wgpu::CompareFunction,
}

const fn beacon_depth_descriptor(write_depth: bool) -> BeaconDepthDescriptor {
    BeaconDepthDescriptor {
        write_depth,
        compare: wgpu::CompareFunction::LessEqual,
    }
}

/// Draws beacon beams: the opaque, depth-writing inner shell and the translucent,
/// depth-testing-but-not-writing outer shell, in that order — see the module-level comment
/// above [`BeaconVertex`] for exactly which vanilla GL state each corresponds to.
///
/// java: `TileEntityBeaconRenderer`. Unlike [`super::render::EntityPhase`], there is only ever
/// one texture (`textures/entity/beacon_beam.png`, [`TileKind::Beacon`]'s fixed texture) so this
/// phase has no per-draw texture switching to batch.
pub struct BeaconPhase {
    pipeline_inner: wgpu::RenderPipeline,
    pipeline_outer: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: Option<BeaconTexture>,
    inner: Vec<BeaconVertex>,
    outer: Vec<BeaconVertex>,
    buffer_inner: Option<wgpu::Buffer>,
    buffer_outer: Option<wgpu::Buffer>,
    capacity_inner: u64,
    capacity_outer: u64,
    uploaded_inner: u32,
    uploaded_outer: u32,
    enabled: bool,
}

impl BeaconPhase {
    /// Builds both pipelines. Cheap enough to do at startup; holds no per-beam state.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render beacon"),
                source: wgpu::ShaderSource::Wgsl(BEACON_WGSL.into()),
            });
        let texture_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render beacon texture"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render beacon layout"),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(&texture_layout)],
                immediate_size: 0,
            });
        let outer_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let build =
            |label: &str, entry: &str, write_depth: bool, blend: Option<wgpu::BlendState>| {
                let depth = beacon_depth_descriptor(write_depth);
                gpu.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(label),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs"),
                            compilation_options: Default::default(),
                            buffers: &[Some(BEACON_VERTEX_LAYOUT)],
                        },
                        // java: `disableCull()`.
                        primitive: wgpu::PrimitiveState {
                            cull_mode: None,
                            ..Default::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: DEPTH_FORMAT,
                            depth_write_enabled: Some(depth.write_depth),
                            // Neither shell ever disables the depth *test* — only the write mask
                            // changes between them (see the module comment above).
                            depth_compare: Some(depth.compare),
                            stencil: wgpu::StencilState::default(),
                            bias: wgpu::DepthBiasState::default(),
                        }),
                        multisample: wgpu::MultisampleState {
                            count: gpu.sample_count(),
                            ..Default::default()
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some(entry),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: gpu.config.format,
                                blend,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    })
            };
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render beacon sampler"),
            // java: `glTexParameterf(..., GL_TEXTURE_WRAP_S/T, GL_REPEAT)` — the inner shell's V
            // range reaches 2.5x the segment height, well outside `0..1`.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            pipeline_inner: build("mc-render beacon/inner", "fs_inner", true, None),
            pipeline_outer: build(
                "mc-render beacon/outer",
                "fs_outer",
                false,
                Some(outer_blend),
            ),
            texture_layout,
            sampler,
            texture: None,
            inner: Vec::new(),
            outer: Vec::new(),
            buffer_inner: None,
            buffer_outer: None,
            capacity_inner: 0,
            capacity_outer: 0,
            uploaded_inner: 0,
            uploaded_outer: 0,
            enabled: true,
        }
    }

    /// Uploads (or replaces) `textures/entity/beacon_beam.png`. `rgba` must be `width * height *
    /// 4` bytes.
    ///
    /// # Panics
    /// If `rgba` is not exactly `width * height * 4` bytes.
    pub fn upload_texture(&mut self, gpu: &Gpu, rgba: &[u8], width: u32, height: u32) {
        assert_eq!(
            rgba.len(),
            (width * height * 4) as usize,
            "beacon beam texture is {} bytes, expected {width}x{height}x4",
            rgba.len()
        );
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mc-render beacon_beam.png"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: JAVA_BEACON_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            texture.as_image_copy(),
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-render beacon_beam.png"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.texture = Some(BeaconTexture { bind_group });
    }

    /// True once [`BeaconPhase::upload_texture`] has run.
    #[must_use]
    pub fn has_texture(&self) -> bool {
        self.texture.is_some()
    }

    /// Drops all per-frame geometry. Call once at the top of the frame.
    pub fn begin(&mut self) {
        self.inner.clear();
        self.outer.clear();
    }

    /// Queues one beam segment's inner-shell vertices (see [`emit_beacon_inner_shell`]).
    pub fn push_inner(&mut self, verts: &[BeaconVertex]) {
        self.inner.extend_from_slice(verts);
    }

    /// Queues one beam segment's outer-shell vertices (see [`emit_beacon_outer_shell`]).
    pub fn push_outer(&mut self, verts: &[BeaconVertex]) {
        self.outer.extend_from_slice(verts);
    }

    fn upload(
        gpu: &Gpu,
        verts: &[BeaconVertex],
        buffer: &mut Option<wgpu::Buffer>,
        capacity: &mut u64,
    ) {
        if verts.is_empty() {
            return;
        }
        let needed = std::mem::size_of_val(verts) as u64;
        if *capacity < needed || buffer.is_none() {
            let cap = needed.next_power_of_two().max(4096);
            *buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc-render beacon vertices"),
                size: cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            *capacity = cap;
        }
        if let Some(buf) = buffer {
            gpu.write_buffer_staged(buf, 0, bytemuck::cast_slice(verts));
        }
    }

    /// Uploads this frame's geometry. Call once after the last `push_inner`/`push_outer`.
    pub fn finish(&mut self, gpu: &Gpu) {
        self.uploaded_inner = self.inner.len() as u32;
        self.uploaded_outer = self.outer.len() as u32;
        Self::upload(
            gpu,
            &self.inner,
            &mut self.buffer_inner,
            &mut self.capacity_inner,
        );
        Self::upload(
            gpu,
            &self.outer,
            &mut self.buffer_outer,
            &mut self.capacity_outer,
        );
    }

    /// Vertices queued this frame, `(inner, outer)`.
    #[must_use]
    pub fn vertex_counts(&self) -> (usize, usize) {
        (self.inner.len(), self.outer.len())
    }

    /// Turns the phase off without unregistering it.
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }
}

impl RenderPhase for BeaconPhase {
    fn name(&self) -> &'static str {
        "beacon"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.record_to(gpu, PassTarget::Own { enc, view });
    }

    // One pass, nothing else on the encoder: it joins the world pass.
    fn shares_world_pass(&self) -> bool {
        true
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, _cam: &Camera) {
        self.record_to(gpu, PassTarget::Shared(pass));
    }
}

impl BeaconPhase {
    /// The phase's draws, into a pass of its own or the shared world pass.
    fn record_to(&self, gpu: &Gpu, target: PassTarget<'_>) {
        let Some(tex) = &self.texture else {
            return;
        };
        if self.uploaded_inner == 0 && self.uploaded_outer == 0 {
            return;
        }
        let mut own = None;
        let pass = target.begin(gpu, "beacon", &mut own);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &tex.bind_group, &[]);
        // Draw order matters: the inner shell writes depth first, so the outer shell (drawn
        // second, depth-tested but not depth-writing) is correctly hidden by anything the inner
        // shell already put closer to the camera — java draws them in this order too.
        if self.uploaded_inner > 0 {
            if let Some(buf) = &self.buffer_inner {
                pass.set_pipeline(&self.pipeline_inner);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..self.uploaded_inner, 0..1);
            }
        }
        if self.uploaded_outer > 0 {
            if let Some(buf) = &self.buffer_outer {
                pass.set_pipeline(&self.pipeline_outer);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..self.uploaded_outer, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::TileModels;
    use super::*;

    #[test]
    fn beacon_shell_descriptors_keep_java_lequal_and_distinct_write_masks() {
        let inner = beacon_depth_descriptor(true);
        let outer = beacon_depth_descriptor(false);
        assert_eq!(inner.compare, wgpu::CompareFunction::LessEqual);
        assert_eq!(outer.compare, wgpu::CompareFunction::LessEqual);
        assert!(inner.write_depth);
        assert!(!outer.write_depth);
    }

    #[test]
    fn beacon_depth_control_does_not_change_unrelated_tile_kind_selection() {
        assert_eq!(TileKind::Chest.model(), Some(ModelKind::Chest));
        assert_eq!(
            TileKind::Chest.texture(),
            Some("textures/entity/chest/normal.png")
        );
        assert_eq!(TileKind::Beacon.model(), None);
    }

    #[test]
    fn negative_enchant_page_phases_use_the_java_shifted_cast_not_rust_truncation() {
        // base -0.5 makes the two sampled phases -0.25 and +0.25. Java's shifted cast returns
        // -1 for the first, giving fractional part 0.75 and a 0.9-open page. `f32::trunc`
        // returns zero and incorrectly clamps that page completely shut.
        let (left, right) = enchant_page_flips(-1.0, 0.0, 0.5);
        assert!((left - 0.9).abs() < 1e-6, "negative phase opened to {left}");
        assert!(
            (right - 0.1).abs() < 1e-6,
            "positive control opened to {right}"
        );
    }

    #[test]
    fn enchant_page_phase_preserves_shifted_cast_behavior_below_minus_1024() {
        // `truncateDoubleToInt` is floor only while value + 1024 stays non-negative. At
        // -1024.25 Java computes `(int)-0.25 - 1024 == -1024`, not floor(-1024.25) == -1025.
        // This pins the exact helper rather than the report's common-range floor shortcut.
        let (left, right) = enchant_page_flips(-1024.5, -1024.5, 0.0);
        assert_eq!(left, 0.0);
        assert!(
            (right - 0.1).abs() < 1e-6,
            "adjacent control opened to {right}"
        );
    }

    #[test]
    fn positive_enchant_page_phases_keep_the_existing_fractional_cycle() {
        let (left, right) = enchant_page_flips(0.5, 0.5, 0.0);
        assert!((left - 0.9).abs() < 1e-6);
        assert!((right - 0.1).abs() < 1e-6);
    }

    #[test]
    fn a_closed_chest_lid_is_flat_and_an_open_one_is_a_right_angle() {
        assert_eq!(chest_lid_angle(0.0, 0.0, 0.0, &[]), 0.0);
        let open = chest_lid_angle(1.0, 1.0, 1.0, &[]);
        assert!(
            (open + core::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "fully open is -PI/2, got {open}"
        );
    }

    #[test]
    fn the_lid_snaps_open_because_the_easing_is_cubic() {
        // java: f = 1 - (1 - f)^3 — a third of the way through, the lid is already 70% open.
        let third = chest_lid_angle(0.0, 1.0, 1.0 / 3.0, &[]).abs();
        let half_travel = core::f32::consts::FRAC_PI_2 * 0.5;
        assert!(
            third > half_travel,
            "{third} should already be past halfway"
        );
        let want = (1.0 - (1.0 - 1.0f32 / 3.0).powi(3)) * core::f32::consts::FRAC_PI_2;
        assert!((third - want).abs() < 1e-6);
    }

    #[test]
    fn a_double_chest_opens_with_whichever_half_is_further_along() {
        let alone = chest_lid_angle(0.0, 0.2, 1.0, &[]);
        let with_neighbour = chest_lid_angle(0.0, 0.2, 1.0, &[0.9]);
        assert!(
            with_neighbour < alone,
            "the neighbour's larger lid must win: {with_neighbour} vs {alone}"
        );
        // A smaller neighbour changes nothing.
        assert_eq!(
            chest_lid_angle(0.0, 0.9, 1.0, &[0.1]),
            chest_lid_angle(0.0, 0.9, 1.0, &[])
        );
    }

    #[test]
    fn a_neighbour_lid_changes_the_live_double_chest_pose() {
        let mut models = TileModels::new(false);
        let mut own_only = Vec::new();
        let mut neighbour_wins = Vec::new();

        // The two halves are intentionally asymmetric.  If emit_chest ignores the adjacent
        // interpolation input, both vertex streams are identical and this negative control bites.
        models.emit_chest(
            Vec3::ZERO,
            2,
            false,
            true,
            Vec3::ZERO,
            0.1,
            0.4,
            &[],
            0.5,
            &mut own_only,
        );
        models.emit_chest(
            Vec3::ZERO,
            2,
            false,
            true,
            Vec3::ZERO,
            0.1,
            0.4,
            &[0.9],
            0.5,
            &mut neighbour_wins,
        );
        assert_eq!(own_only.len(), neighbour_wins.len());
        assert!(
            own_only
                .iter()
                .zip(&neighbour_wins)
                .any(|(a, b)| a.pos != b.pos),
            "the larger adjacent lid must affect the emitted double-chest pose"
        );
    }

    #[test]
    fn the_chest_matrix_flips_y_and_z_and_lands_on_the_block() {
        let m = chest_matrix(Vec3::ZERO, 0.0, Vec3::ZERO);
        // The model's origin corner (0, 0, 0) maps to the block's top-back corner because of
        // the -1 scale on Y and Z, then the +1 translate on both.
        let o = m.transform_point3(Vec3::ZERO);
        assert!((o.y - 1.0).abs() < 1e-6, "{o}");
        assert!((o.z - 1.0).abs() < 1e-6, "{o}");
        // A unit step along model +Y goes *down* in the world.
        let up = m.transform_vector3(Vec3::Y);
        assert!(up.y < 0.0);
    }

    #[test]
    fn a_large_chests_far_half_is_offset_by_a_whole_block() {
        let a = chest_matrix(Vec3::ZERO, 0.0, Vec3::ZERO).transform_point3(Vec3::ZERO);
        let b = chest_matrix(Vec3::ZERO, 0.0, Vec3::X).transform_point3(Vec3::ZERO);
        assert!((b.x - a.x - 1.0).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn standing_signs_turn_in_sixteenths_of_a_circle() {
        // java: metadata * 360 / 16, negated.
        for (meta, want) in [(0, 0.0), (4, -90.0), (8, -180.0), (12, -270.0)] {
            let m = sign_matrix(Vec3::ZERO, true, meta);
            let f = m.transform_vector3(Vec3::Z).normalize();
            let expect = Mat4::from_rotation_y((want as f32).to_radians())
                .transform_vector3(Vec3::Z)
                .normalize();
            assert!((f - expect).length() < 1e-5, "meta {meta}: {f} vs {expect}");
        }
    }

    #[test]
    fn wall_signs_snap_to_four_facings_and_hang_back() {
        assert_eq!(wall_facing_degrees(2), 180.0);
        assert_eq!(wall_facing_degrees(3), 0.0);
        assert_eq!(wall_facing_degrees(4), 90.0);
        assert_eq!(wall_facing_degrees(5), -90.0);
        assert_eq!(
            wall_facing_degrees(99),
            0.0,
            "unexpected metadata falls through to 0"
        );

        let standing = sign_matrix(Vec3::ZERO, true, 0).transform_point3(Vec3::ZERO);
        let wall = sign_matrix(Vec3::ZERO, false, 3).transform_point3(Vec3::ZERO);
        assert!(
            (wall.y - (standing.y - 0.3125)).abs() < 1e-6,
            "{wall} vs {standing}"
        );
        assert!((wall.z - (standing.z - 0.4375)).abs() < 1e-6);
    }

    #[test]
    fn a_wall_banner_sits_lower_than_a_standing_one() {
        let standing = banner_matrix(Vec3::ZERO, true, 0).transform_point3(Vec3::ZERO);
        let wall = banner_matrix(Vec3::ZERO, false, 3).transform_point3(Vec3::ZERO);
        // java: +0.75 * f versus -0.25 * f, then the extra -0.3125.
        assert!((standing.y - 0.75 * SIGN_SCALE).abs() < 1e-6);
        assert!(
            (wall.y - (-0.25 * SIGN_SCALE - 0.3125)).abs() < 1e-6,
            "{wall}"
        );
    }

    #[test]
    fn skull_variants_pick_the_right_model_and_texture() {
        assert_eq!(SkullVariant::from_id(0), SkullVariant::Skeleton);
        assert_eq!(
            SkullVariant::from_id(7),
            SkullVariant::Skeleton,
            "default arm"
        );
        assert_eq!(SkullVariant::Zombie.model(), ModelKind::HumanoidHead);
        assert_eq!(SkullVariant::Player.model(), ModelKind::HumanoidHead);
        assert_eq!(
            SkullVariant::Creeper.model(),
            ModelKind::SkeletonHead,
            "the creeper head uses the skeleton model despite its own texture"
        );
        assert_eq!(
            SkullVariant::Player.texture(),
            None,
            "a player head wears its skin"
        );
        assert_eq!(
            SkullVariant::WitherSkeleton.texture(),
            Some("textures/entity/skeleton/wither_skeleton.png")
        );
    }

    #[test]
    fn a_wall_skull_ignores_its_rotation_field() {
        // java: the switch *assigns* p_180543_5_, discarding the caller's rotation.
        for (mount, want) in [
            (SkullMount::South, 180.0),
            (SkullMount::West, 270.0),
            (SkullMount::East, 90.0),
        ] {
            let (_, yaw) = skull_transform(Vec3::ZERO, mount, 45.0);
            assert_eq!(yaw, want, "{mount:?}");
        }
        // North is the one facing that leaves the rotation alone, and the floor keeps it too.
        assert_eq!(skull_transform(Vec3::ZERO, SkullMount::North, 45.0).1, 45.0);
        assert_eq!(skull_transform(Vec3::ZERO, SkullMount::Floor, 45.0).1, 45.0);
    }

    #[test]
    fn a_floor_skull_sits_on_the_block_and_a_wall_skull_is_lifted() {
        let floor = skull_transform(Vec3::ZERO, SkullMount::Floor, 0.0)
            .0
            .transform_point3(Vec3::ZERO);
        let wall = skull_transform(Vec3::ZERO, SkullMount::North, 0.0)
            .0
            .transform_point3(Vec3::ZERO);
        assert_eq!(floor.y, 0.0);
        assert!((wall.y - 0.25).abs() < 1e-6);
        assert!((wall.z - 0.74).abs() < 1e-6);
    }

    #[test]
    fn the_beacon_inner_shell_orbits_at_radius_a_fifth() {
        let b = beacon_beam(0, 0.0, 4, 1.0);
        for c in b.inner_corners {
            let d = ((c[0] - 0.5).powi(2) + (c[1] - 0.5).powi(2)).sqrt();
            assert!((d - 0.2).abs() < 1e-9, "corner {c:?} is {d} from centre");
        }
        // java lists the corners at d2 + 3PI/4, PI/4, 5PI/4, 7PI/4 — so index 0 is opposite
        // index 3, and index 1 opposite index 2, *not* 0-2 as a naive reading suggests. The
        // quad winding depends on that ordering, so it is worth pinning.
        for (a, c) in [(0, 3), (1, 2)] {
            let across = ((b.inner_corners[a][0] - b.inner_corners[c][0]).powi(2)
                + (b.inner_corners[a][1] - b.inner_corners[c][1]).powi(2))
            .sqrt();
            assert!((across - 0.4).abs() < 1e-9, "{a} vs {c}: {across}");
        }
        // Adjacent corners are a quarter turn apart.
        let side = ((b.inner_corners[0][0] - b.inner_corners[1][0]).powi(2)
            + (b.inner_corners[0][1] - b.inner_corners[1][1]).powi(2))
        .sqrt();
        assert!(
            (side - 0.4 / core::f64::consts::SQRT_2).abs() < 1e-9,
            "{side}"
        );
    }

    #[test]
    fn the_beacon_shell_turns_with_world_time() {
        let a = beacon_beam(0, 0.0, 4, 1.0).inner_corners[0];
        let b = beacon_beam(100, 0.0, 4, 1.0).inner_corners[0];
        assert!(
            (a[0] - b[0]).abs() > 1e-3 || (a[1] - b[1]).abs() > 1e-3,
            "{a:?} {b:?}"
        );
    }

    #[test]
    fn the_inner_beam_stretches_its_texture_two_and_a_half_times() {
        let b = beacon_beam(0, 0.0, 10, 1.0);
        let inner_span = b.inner_v[1] - b.inner_v[0];
        let outer_span = b.outer_v[1] - b.outer_v[0];
        assert!((inner_span - 25.0).abs() < 1e-9, "{inner_span}");
        assert!((outer_span - 10.0).abs() < 1e-9, "{outer_span}");
        assert!((inner_span / outer_span - 2.5).abs() < 1e-9);
        assert_eq!(b.outer_alpha, 0.125);
    }

    #[test]
    fn a_fading_beam_shrinks_its_texture_span_not_its_geometry() {
        // java: `(float)height * f` where f is shouldBeamRender()'s 0..1 fade.
        let full = beacon_beam(0, 0.0, 10, 1.0);
        let half = beacon_beam(0, 0.0, 10, 0.5);
        assert!(((half.outer_v[1] - half.outer_v[0]) - 5.0).abs() < 1e-9);
        assert_eq!(full.inner_corners, half.inner_corners);
    }

    #[test]
    fn tile_kinds_map_to_the_right_models_and_textures() {
        assert_eq!(TileKind::LargeChest.model(), Some(ModelKind::LargeChest));
        assert_eq!(TileKind::Chest.model(), Some(ModelKind::Chest));
        assert_eq!(TileKind::EnderChest.model(), Some(ModelKind::Chest));
        assert_eq!(
            TileKind::EnderChest.texture(),
            Some("textures/entity/chest/ender.png")
        );
        assert_eq!(TileKind::Beacon.model(), None);
        assert_eq!(
            TileKind::Skull.texture(),
            None,
            "the skull picks per variant"
        );
    }

    // --------------------------------------------------------------- beacon vertex emission

    #[test]
    fn each_shell_emits_four_quads_as_twenty_four_vertices() {
        let beam = beacon_beam(0, 0.0, 4, 1.0);
        let mut inner = Vec::new();
        emit_beacon_inner_shell([0.0, 0.0], 0.0, 4.0, &beam, [1.0, 0.5, 0.2], &mut inner);
        assert_eq!(inner.len(), 4 * 6);

        let mut outer = Vec::new();
        emit_beacon_outer_shell([0.0, 0.0], 0.0, 4.0, &beam, [1.0, 0.5, 0.2], &mut outer);
        assert_eq!(outer.len(), 4 * 6);
    }

    #[test]
    fn the_inner_shell_carries_the_java_opaque_alpha() {
        let beam = beacon_beam(0, 0.0, 4, 1.0);
        let mut out = Vec::new();
        emit_beacon_inner_shell([0.0, 0.0], 0.0, 4.0, &beam, [1.0, 0.5, 0.2], &mut out);
        for v in &out {
            assert_eq!(v.color, [1.0, 0.5, 0.2, 1.0]);
        }
    }

    #[test]
    fn the_outer_shell_carries_the_faded_alpha() {
        let beam = beacon_beam(0, 0.0, 4, 1.0);
        let mut out = Vec::new();
        emit_beacon_outer_shell([0.0, 0.0], 0.0, 4.0, &beam, [1.0, 0.5, 0.2], &mut out);
        for v in &out {
            assert_eq!(v.color, [1.0, 0.5, 0.2, 0.125]);
        }
    }

    #[test]
    fn shells_are_offset_by_the_callers_block_position() {
        let beam = beacon_beam(0, 0.0, 4, 1.0);
        let mut at_origin = Vec::new();
        emit_beacon_outer_shell([0.0, 0.0], 0.0, 4.0, &beam, [1.0, 1.0, 1.0], &mut at_origin);
        let mut moved = Vec::new();
        emit_beacon_outer_shell([10.0, -3.0], 0.0, 4.0, &beam, [1.0, 1.0, 1.0], &mut moved);
        for (a, b) in at_origin.iter().zip(&moved) {
            assert!((b.pos[0] - a.pos[0] - 10.0).abs() < 1e-5);
            assert!((b.pos[2] - a.pos[2] + 3.0).abs() < 1e-5);
            assert_eq!(a.pos[1], b.pos[1], "y is untouched by the xz offset");
        }
    }

    #[test]
    fn the_outer_shell_spans_the_static_point_two_to_point_eight_box() {
        let beam = beacon_beam(0, 0.0, 4, 1.0);
        let mut out = Vec::new();
        emit_beacon_outer_shell([0.0, 0.0], 0.0, 4.0, &beam, [1.0, 1.0, 1.0], &mut out);
        let (mut lo_x, mut hi_x) = (f32::MAX, f32::MIN);
        for v in &out {
            lo_x = lo_x.min(v.pos[0]);
            hi_x = hi_x.max(v.pos[0]);
        }
        assert!((lo_x - 0.2).abs() < 1e-5, "{lo_x}");
        assert!((hi_x - 0.8).abs() < 1e-5, "{hi_x}");
    }

    #[test]
    fn the_inner_shell_orbits_inside_the_outer_shells_footprint() {
        // Radius 0.2 about the 0.5 centre spans 0.3..0.7, strictly inside the outer shell's
        // static 0.2..0.8 box.
        let beam = beacon_beam(0, 0.0, 4, 1.0);
        let mut out = Vec::new();
        emit_beacon_inner_shell([0.0, 0.0], 0.0, 4.0, &beam, [1.0, 1.0, 1.0], &mut out);
        for v in &out {
            assert!(
                v.pos[0] >= 0.3 - 1e-6 && v.pos[0] <= 0.7 + 1e-6,
                "{}",
                v.pos[0]
            );
            assert!(
                v.pos[2] >= 0.3 - 1e-6 && v.pos[2] <= 0.7 + 1e-6,
                "{}",
                v.pos[2]
            );
        }
    }

    #[test]
    fn shell_vertices_span_from_the_bottom_to_the_top_height() {
        let beam = beacon_beam(0, 0.0, 4, 1.0);
        let mut out = Vec::new();
        emit_beacon_inner_shell([0.0, 0.0], 2.0, 6.0, &beam, [1.0, 1.0, 1.0], &mut out);
        let ys: Vec<f32> = out.iter().map(|v| v.pos[1]).collect();
        assert!(ys.iter().any(|&y| (y - 2.0).abs() < 1e-6));
        assert!(ys.iter().any(|&y| (y - 6.0).abs() < 1e-6));
        assert!(ys.iter().all(|&y| (2.0..=6.0).contains(&y)));
    }

    // --------------------------------------------------------------------------- real GPU

    fn headless(width: u32, height: u32) -> Option<crate::context::Gpu> {
        use crate::context::GpuOptions;
        use crate::RenderError;
        match crate::context::Gpu::headless(
            width,
            height,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(g) => Some(g),
            Err(RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                None
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        }
    }

    fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * width + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    }

    /// The end-to-end proof: a real beam, posed and uploaded, drawn into an offscreen target and
    /// read back. Closes the "no vertex emission and no additive blend pipeline state" gap.
    #[test]
    fn a_beacon_beam_reaches_the_frame_buffer() {
        let Some(mut gpu) = headless(96, 96) else {
            return;
        };
        let mut phase = BeaconPhase::new(&gpu);
        // A non-white, fully opaque synthetic texture: real pixels, not a resource-pack fetch.
        // The channel separation makes an sRGB upload observable in the readback below.
        const BEAM_TEXEL: [u8; 4] = [200, 120, 60, 255];
        phase.upload_texture(&gpu, &BEAM_TEXEL, 1, 1);
        assert!(phase.has_texture());

        let beam = beacon_beam(0, 0.0, 6, 1.0);
        let mut inner = Vec::new();
        emit_beacon_inner_shell([0.0, 0.0], -1.6, 4.4, &beam, [1.0, 1.0, 0.6], &mut inner);
        let mut outer = Vec::new();
        emit_beacon_outer_shell([0.0, 0.0], -1.6, 4.4, &beam, [1.0, 1.0, 0.6], &mut outer);

        phase.begin();
        phase.push_inner(&inner);
        phase.push_outer(&outer);
        phase.finish(&gpu);
        assert_eq!(phase.vertex_counts(), (inner.len(), outer.len()));

        let mut phases = crate::pipeline::PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(crate::pipeline::PhaseOrder::ENTITIES, Box::new(phase));
        let mut cam = Camera::default();
        cam.set_viewport(96, 96);
        cam.update();
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, frame.view(), &cam, 0.0);
        frame.present(&gpu.queue);
        let pixels = gpu.read_pixels().expect("read back");

        let lit = pixels
            .chunks(4)
            .filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8)
            .count();
        assert!(
            lit > 20,
            "the beam should cover a visible sliver, got {lit} lit pixels"
        );
        assert!(
            pixels
                .chunks(4)
                .any(|p| { p[0] >= 170 && p[1] >= 95 && (25..=50).contains(&p[2]) }),
            "the raw Java GL_RGBA beam texel should survive sampling and [1, 1, 0.6] modulation"
        );
    }

    /// The inner shell writes depth and draws opaque; a beam segment placed *behind* another
    /// opaque draw must be hidden, which only happens if the inner pipeline's depth test and the
    /// outer pipeline's depth test are both really wired to the shared depth buffer.
    #[test]
    fn the_inner_shell_is_depth_tested_against_the_rest_of_the_scene() {
        let Some(mut gpu) = headless(64, 64) else {
            return;
        };
        // An opaque red wall 2 blocks away, drawn first so it writes depth.
        let mut wall = super::super::render::EntityPhase::new(&gpu);
        wall.upload_texture(&gpu, "wall", &[255, 0, 0, 255], 1, 1);
        let mut wverts = Vec::new();
        for (x, y) in [
            (-4.0f32, -4.0f32),
            (4.0, -4.0),
            (4.0, 4.0),
            (-4.0, -4.0),
            (4.0, 4.0),
            (-4.0, 4.0),
        ] {
            wverts.push(EntityVertex {
                pos: [x, y, 2.0],
                uv: [0.5, 0.5],
                normal: [0.0, 0.0, -1.0],
                // A02-07 added `EntityVertex::color`; `Default` is opaque white.
                ..EntityVertex::default()
            });
        }
        wall.begin();
        wall.push("wall", &wverts);
        wall.finish(&gpu);

        // A beam behind the wall, centred directly ahead of the camera at roughly z = 5, which
        // the wall's depth must hide. `xz` shifts the shell's own 0.3..0.7-around-0.5 footprint
        // so its centre lands near `(x, z) = (0, 5)`.
        let mut phase = BeaconPhase::new(&gpu);
        phase.upload_texture(&gpu, &[255, 255, 0, 255], 1, 1);
        let beam = beacon_beam(0, 0.0, 4, 1.0);
        let mut inner = Vec::new();
        emit_beacon_inner_shell([-0.5, 4.5], -2.0, 2.0, &beam, [1.0, 1.0, 1.0], &mut inner);
        phase.begin();
        phase.push_inner(&inner);
        phase.finish(&gpu);

        let mut phases = crate::pipeline::PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(crate::pipeline::PhaseOrder::ENTITIES, Box::new(wall));
        phases.insert(crate::pipeline::PhaseOrder::PARTICLES, Box::new(phase));
        let mut cam = Camera::default();
        cam.set_viewport(64, 64);
        cam.update();
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, frame.view(), &cam, 0.0);
        frame.present(&gpu.queue);
        let pixels = gpu.read_pixels().expect("read back");

        // The whole frame is covered by the red wall; no yellow beam pixel should survive.
        let yellow_leaked = pixels
            .chunks(4)
            .any(|p| p[0] > 200 && p[1] > 200 && p[2] < 60);
        assert!(
            !yellow_leaked,
            "the beam behind the wall must be depth-occluded"
        );
        // Sanity: the wall itself did draw. `EntityPhase`'s shader applies directional
        // lighting (java: `RenderHelper.enableStandardItemLighting`), which dims the raw
        // [255, 0, 0] texture to as little as `0.4x` depending on the quad's normal — so this
        // checks the wall's colour dominates, not that it survived undimmed.
        let centre = pixel(&pixels, 64, 32, 32);
        assert!(
            centre[0] > 90 && centre[0] > centre[1] * 3,
            "wall did not draw: {centre:?}"
        );
    }
}
