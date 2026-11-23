//! Fluid rendering — `BlockFluidRenderer`.
//!
//! Water and lava have no JSON model. Their top face is a quad whose four corners each sit at
//! a height interpolated from the 2×2 column of fluid cells that touches that corner, which is
//! what makes a flowing stream slope smoothly instead of stepping. The still/flowing texture
//! choice and the top face's UV rotation both come from the flow direction, computed from the
//! surrounding fluid levels.
//!
//! # Texture availability
//!
//! Vanilla registers `blocks/{water,lava}_{still,flow}` into the block atlas explicitly, from
//! `BlockFluidRenderer.initAtlasSprites`, because no JSON model references them. S09's
//! `bake_all` originally stitched only the textures the baked models reference, so those four
//! sprites were absent and [`mc_assets::model::sprite`] answered with its placeholder — the
//! geometry below was right and every fluid UV was wrong.
//!
//! **Closed.** `bake_all` now also stitches `mc_assets::model::BUILTIN_TEXTURES`, its port of
//! `ModelBakery.LOCATIONS_BUILTIN_TEXTURES`. [`fluid_sprites_available`] is kept as the runtime
//! probe (it is still `false` before `install` has run, or under a resource pack that removed
//! them) and `s11_fluid.rs` asserts both that it is now `true` and that a rendered water surface
//! samples the real `water_still` texels.
//
// java: client/renderer/BlockFluidRenderer.java, block/BlockLiquid.java

use mc_assets::{model::sprite, sprite::UvRect, ResourceLocation};
use mc_data::{block_props, BlockState, Material};
use mc_math::{BlockPos, Facing};
use mc_world::World;

use super::light::{block_byte, mixed_brightness, sky_byte};
use super::vertex::{Buffer, TerrainVertex};

/// The four sprite locations vanilla's fluid renderer binds.
/// java: `BlockFluidRenderer.initAtlasSprites`.
pub const FLUID_SPRITES: [&str; 4] = [
    "textures/blocks/water_still",
    "textures/blocks/water_flow",
    "textures/blocks/lava_still",
    "textures/blocks/lava_flow",
];

/// Whether the installed atlas actually contains the four fluid textures. `false` means fluid
/// geometry will be built with placeholder UVs (see the module docs).
///
/// Two independent signals, because either alone has a plausible false positive: the sprite
/// must have a real sub-rectangle of the atlas (the placeholder's is the whole sheet) **and**
/// more than one animation frame (all four fluid textures are animated in vanilla, and the
/// placeholder has exactly one).
#[must_use]
pub fn fluid_sprites_available() -> bool {
    FLUID_SPRITES.iter().all(|p| {
        let loc = ResourceLocation::minecraft(*p);
        let s = sprite(&loc);
        s.uv != UvRect::FULL && s.frames.len() > 1
    })
}

fn uv_for(path: &str) -> UvRect {
    sprite(&ResourceLocation::minecraft(path)).uv
}

/// The four [`FLUID_SPRITES`] UVs, looked up once per installed asset generation and thread.
///
/// Every fluid cell used to build two `ResourceLocation`s (four `String`s) and do two
/// string-keyed sprite lookups (MESH-9). The answer only changes when the asset registry is
/// (re)installed, which bumps [`mc_assets::model::installed_generation`] — the same generation
/// `sprite` reads its registry through — so the cache can never serve another pack's UVs.
fn fluid_uvs() -> [UvRect; 4] {
    thread_local! {
        static CACHE: std::cell::Cell<Option<(u64, [UvRect; 4])>> =
            const { std::cell::Cell::new(None) };
    }
    // Under `with_thread_registry` this cache, keyed by the
    // installed generation, would serve the installed pack's UVs: bypass it.
    if mc_assets::model::thread_registry_overridden() {
        return FLUID_SPRITES.map(uv_for);
    }
    let generation = mc_assets::model::installed_generation();
    CACHE.with(|cache| match cache.get() {
        Some((cached, uvs)) if cached == generation => uvs,
        _ => {
            let uvs = FLUID_SPRITES.map(uv_for);
            cache.set(Some((generation, uvs)));
            uvs
        }
    })
}

/// `BlockLiquid.getLiquidHeightPercent` — the fraction of the block the fluid does *not* fill.
#[must_use]
pub fn liquid_height_percent(meta: u8) -> f32 {
    let meta = if meta >= 8 { 0 } else { meta };
    f32::from(meta + 1) / 9.0
}

/// The fluid level of a cell, or `-1` when it is not this fluid.
/// java: `BlockLiquid.getLevel`.
fn level(world: &World, pos: BlockPos, material: Material) -> i32 {
    let s = world.block_at(pos);
    if block_props(s).material == material {
        i32::from(s.meta())
    } else {
        -1
    }
}

/// java: `BlockLiquid.getEffectiveFlowDecay`.
fn effective_flow_decay(world: &World, pos: BlockPos, material: Material) -> i32 {
    let l = level(world, pos, material);
    if l >= 8 {
        0
    } else {
        l
    }
}

/// The interpolated corner height. java: `BlockFluidRenderer.getFluidHeight`.
///
/// Returns `1.0` when any of the four columns has the same fluid directly above it, so a
/// submerged cell is a full block with no visible surface.
#[must_use]
pub fn fluid_height(world: &World, pos: BlockPos, material: Material) -> f32 {
    let mut i = 0u32;
    let mut f = 0.0f32;
    for j in 0..4i32 {
        let p = pos.add(-(j & 1), 0, -((j >> 1) & 1));
        if block_props(world.block_at(p.up())).material == material {
            return 1.0;
        }
        let s = world.block_at(p);
        let m = block_props(s).material;
        if m != material {
            if !m.is_solid() {
                f += 1.0;
                i += 1;
            }
        } else {
            let k = s.meta();
            if k >= 8 || k == 0 {
                f += liquid_height_percent(k) * 10.0;
                i += 10;
            }
            f += liquid_height_percent(k);
            i += 1;
        }
    }
    if i == 0 {
        return 1.0;
    }
    1.0 - f / i as f32
}

/// Normalize a flow vector with vanilla's `Vec3.normalize` semantics.
///
/// `MathHelper.sqrt_double` narrows the square root to `float`, and vectors shorter than
/// `1.0E-4D` become the exact zero vector. Both details matter before the falling-fluid
/// adjustment and before the final flow-direction lookup.
/// java: `Vec3.normalize` / `MathHelper.sqrt_double`.
#[inline]
fn normalize_flow_vector(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let length = f64::from(mc_math::sqrt_d(x * x + y * y + z * z));
    if length < 1.0E-4 {
        (0.0, 0.0, 0.0)
    } else {
        (x / length, y / length, z / length)
    }
}

/// The flow angle in radians, or `None` for a still cell.
/// java: `BlockLiquid.getFlowDirection` / `MathHelper.atan2`.
#[inline]
fn flow_angle(x: f64, z: f64) -> Option<f32> {
    if x == 0.0 && z == 0.0 {
        None
    } else {
        Some((mc_math::atan2(z, x) - std::f64::consts::FRAC_PI_2) as f32)
    }
}

/// The flow angle in radians, or `None` for a still cell.
/// java: `BlockLiquid.getFlowDirection` / `getFlowVector`.
#[must_use]
pub fn flow_direction(world: &World, pos: BlockPos, material: Material) -> Option<f32> {
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    let mut vz = 0.0f64;
    let i = effective_flow_decay(world, pos, material);

    for facing in Facing::HORIZONTALS {
        let np = pos.relative(facing);
        let j = effective_flow_decay(world, np, material);
        if j < 0 {
            if !block_props(world.block_at(np)).material.blocks_movement() {
                let j = effective_flow_decay(world, np.down(), material);
                if j >= 0 {
                    let k = j - (i - 8);
                    vx += f64::from((np.x - pos.x) * k);
                    vy += f64::from((np.y - pos.y) * k);
                    vz += f64::from((np.z - pos.z) * k);
                }
            }
        } else {
            let l = j - i;
            vx += f64::from((np.x - pos.x) * l);
            vy += f64::from((np.y - pos.y) * l);
            vz += f64::from((np.z - pos.z) * l);
        }
    }

    // java: a falling source (level >= 8) with a solid horizontal neighbour is pushed down.
    if i32::from(world.block_at(pos).meta()) >= 8 {
        for facing in Facing::HORIZONTALS {
            let np = pos.relative(facing);
            if is_block_solid(world, np, material) || is_block_solid(world, np.up(), material) {
                let normalized = normalize_flow_vector(vx, vy, vz);
                vx = normalized.0;
                vy = normalized.1;
                vz = normalized.2;
                vy -= 6.0;
                break;
            }
        }
    }

    let normalized = normalize_flow_vector(vx, vy, vz);
    flow_angle(normalized.0, normalized.2)
}

/// java: `BlockLiquid.isBlockSolid` — the same material is never solid *to itself*, and ice is
/// explicitly excluded so a frozen surface does not make the water under it flow sideways.
fn is_block_solid(world: &World, pos: BlockPos, material: Material) -> bool {
    let m = block_props(world.block_at(pos)).material;
    if m == material || m == Material::Ice {
        false
    } else {
        m.is_solid()
    }
}

/// java: `BlockLiquid.shouldRenderSides` — the top face's underside is only drawn when some
/// column in the 3×3 around it is neither this fluid nor a full block.
fn should_render_sides(world: &World, pos: BlockPos, material: Material) -> bool {
    for i in -1..=1 {
        for j in -1..=1 {
            let p = pos.add(i, 0, j);
            let props = block_props(world.block_at(p));
            if props.material != material && !props.full_block {
                return true;
            }
        }
    }
    false
}

/// java: `BlockLiquid.shouldSideBeRendered`.
///
/// The settings matter: the chain ends in the virtual `isOpaqueCube()` call that `BlockLeaves`
/// overrides to `!fancyGraphics` (`Block.java:468-471`), so under Fast graphics a fluid face
/// against a leaf block is culled and under Fancy it is not.
fn side_visible(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    side: Facing,
    settings: super::cull_face::MeshSettings,
) -> bool {
    super::cull_face::should_side_be_rendered(world, state, pos, side, settings)
}

/// Mesh one fluid cell. java: `BlockFluidRenderer.renderFluid`.
#[allow(clippy::too_many_lines)] // A faithful port of one 220-line Java method.
pub fn render_fluid(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    local: [f32; 3],
    settings: super::cull_face::MeshSettings,
    out: &mut Buffer,
) {
    let material = block_props(state).material;
    let lava = material == Material::Lava;
    let uvs = fluid_uvs();
    let (still, flow) = if lava {
        (uvs[2], uvs[3])
    } else {
        (uvs[0], uvs[1])
    };

    let tint = super::tint::color_multiplier(world, state, pos);
    let (tr, tg, tb) = (
        ((tint >> 16) & 255) as f32 / 255.0,
        ((tint >> 8) & 255) as f32 / 255.0,
        (tint & 255) as f32 / 255.0,
    );

    let up = side_visible(world, state, pos, Facing::Up, settings);
    let down = side_visible(world, state, pos, Facing::Down, settings);
    let sides = [
        side_visible(world, state, pos, Facing::North, settings),
        side_visible(world, state, pos, Facing::South, settings),
        side_visible(world, state, pos, Facing::West, settings),
        side_visible(world, state, pos, Facing::East, settings),
    ];
    if !up && !down && !sides.iter().any(|b| *b) {
        return;
    }

    // Corner heights, in vanilla's (nw, sw, se, ne) order.
    let mut h0 = fluid_height(world, pos, material);
    let mut h1 = fluid_height(world, pos.south(), material);
    let mut h2 = fluid_height(world, pos.east().south(), material);
    let mut h3 = fluid_height(world, pos.east(), material);
    const EPS: f32 = 0.001;

    let (x, y, z) = (local[0], local[1], local[2]);
    // Shader-pack meshing: water flagged for shader water, and no per-face shade
    // under a pack's sun shading. With no pack both are vanilla's (material 0, the face factor).
    let pack = settings.pack;
    let mat = if pack.flags && !lava {
        super::phase::pack::material::WATER
    } else {
        0
    };
    let face = |f: f32| if pack.flat_shade { 1.0 } else { f };

    if up {
        let angle = flow_direction(world, pos, material);
        let uv = if angle.is_some() { flow } else { still };
        h0 -= EPS;
        h1 -= EPS;
        h2 -= EPS;
        h3 -= EPS;

        let (u0, v0, u1, v1, u2, v2, u3, v3) = match angle {
            None => (
                uv.interpolate_u(0.0),
                uv.interpolate_v(0.0),
                uv.interpolate_u(0.0),
                uv.interpolate_v(16.0),
                uv.interpolate_u(16.0),
                uv.interpolate_v(16.0),
                uv.interpolate_u(16.0),
                uv.interpolate_v(0.0),
            ),
            Some(a) => {
                let s = mc_math::sin(a) * 0.25;
                let c = mc_math::cos(a) * 0.25;
                (
                    uv.interpolate_u(8.0 + (-c - s) * 16.0),
                    uv.interpolate_v(8.0 + (-c + s) * 16.0),
                    uv.interpolate_u(8.0 + (-c + s) * 16.0),
                    uv.interpolate_v(8.0 + (c + s) * 16.0),
                    uv.interpolate_u(8.0 + (c + s) * 16.0),
                    uv.interpolate_v(8.0 + (c - s) * 16.0),
                    uv.interpolate_u(8.0 + (c - s) * 16.0),
                    uv.interpolate_v(8.0 + (-c - s) * 16.0),
                )
            }
        };

        let br = mixed_brightness(world, state, pos);
        let light = [sky_byte(br), block_byte(br), mat, 0];
        let color = shade(tr, tg, tb, 1.0);
        let top = [
            TerrainVertex::new([x, y + h0, z], [u0, v0], color, light),
            TerrainVertex::new([x, y + h1, z + 1.0], [u1, v1], color, light),
            TerrainVertex::new([x + 1.0, y + h2, z + 1.0], [u2, v2], color, light),
            TerrainVertex::new([x + 1.0, y + h3, z], [u3, v3], color, light),
        ];
        out.push_quad(top);

        // java: the underside of the surface, so the top face is visible from below too.
        if should_render_sides(world, pos.up(), material) {
            out.push_quad([top[0], top[3], top[2], top[1]]);
        }
    }

    if down {
        let br = mixed_brightness(world, state, pos.down());
        let light = [sky_byte(br), block_byte(br), mat, 0];
        // java: `worldRendererIn.pos(...).color(f3, f3, f3, 1.0F)` with `f3 = 0.5F`
        // (`BlockFluidRenderer.java:53`, `:146-149`) — the bottom face is a flat grey and is
        // the **only** face that does not take the `colorMultiplier` tint. In a swamp
        // (`waterColorMultiplier = 14745518`) tinting it turns the underside of the water
        // green.
        let color = shade(1.0, 1.0, 1.0, face(0.5));
        out.push_quad([
            TerrainVertex::new([x, y, z + 1.0], [still.min_u, still.max_v], color, light),
            TerrainVertex::new([x, y, z], [still.min_u, still.min_v], color, light),
            TerrainVertex::new([x + 1.0, y, z], [still.max_u, still.min_v], color, light),
            TerrainVertex::new(
                [x + 1.0, y, z + 1.0],
                [still.max_u, still.max_v],
                color,
                light,
            ),
        ]);
    }

    for (i, visible) in sides.iter().enumerate() {
        if !visible {
            continue;
        }
        let (dx, dz) = match i {
            0 => (0, -1),
            1 => (0, 1),
            2 => (-1, 0),
            _ => (1, 0),
        };
        let np = pos.add(dx, 0, dz);

        let (ha, hb, xa, xb, za, zb) = match i {
            0 => (h0, h3, x, x + 1.0, z + EPS, z + EPS),
            1 => (h2, h1, x + 1.0, x, z + 1.0 - EPS, z + 1.0 - EPS),
            2 => (h1, h0, x + EPS, x + EPS, z + 1.0, z),
            _ => (h3, h2, x + 1.0 - EPS, x + 1.0 - EPS, z, z + 1.0),
        };

        let u0 = flow.interpolate_u(0.0);
        let u1 = flow.interpolate_u(8.0);
        let va = flow.interpolate_v((1.0 - ha) * 16.0 * 0.5);
        let vb = flow.interpolate_v((1.0 - hb) * 16.0 * 0.5);
        let vbot = flow.interpolate_v(8.0);

        let br = mixed_brightness(world, state, np);
        let light = [sky_byte(br), block_byte(br), mat, 0];
        // java: north/south sides get 0.8, west/east 0.6 — the same constants as
        // `FaceBakery.getFaceBrightness`.
        let f = if i < 2 { 0.8 } else { 0.6 };
        let color = shade(tr, tg, tb, face(f));

        let quad = [
            TerrainVertex::new([xa, y + ha, za], [u0, va], color, light),
            TerrainVertex::new([xb, y + hb, zb], [u1, vb], color, light),
            TerrainVertex::new([xb, y, zb], [u1, vbot], color, light),
            TerrainVertex::new([xa, y, za], [u0, vbot], color, light),
        ];
        out.push_quad(quad);
        // java: the same four vertices again in reverse, so the side is double-sided.
        out.push_quad([quad[3], quad[2], quad[1], quad[0]]);
    }
}

fn shade(r: f32, g: f32, b: f32, f: f32) -> [u8; 4] {
    [
        ((r * f).clamp(0.0, 1.0) * 255.0) as u8,
        ((g * f).clamp(0.0, 1.0) * 255.0) as u8,
        ((b * f).clamp(0.0, 1.0) * 255.0) as u8,
        255,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::fixture;
    use mc_math::ChunkSectionPos;

    #[test]
    fn liquid_height_percent_matches_vanilla() {
        // java: (meta + 1) / 9, with meta >= 8 folded to 0.
        assert!((liquid_height_percent(0) - 1.0 / 9.0).abs() < 1e-6);
        assert!((liquid_height_percent(7) - 8.0 / 9.0).abs() < 1e-6);
        assert!((liquid_height_percent(8) - 1.0 / 9.0).abs() < 1e-6);
        assert!((liquid_height_percent(15) - 1.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn a_submerged_cell_is_a_full_block() {
        let mut w = fixture::blank_world(1);
        let water = fixture::state_of("minecraft:water");
        fixture::fill(
            &mut w,
            mc_math::BlockPos::new(0, 60, 0),
            mc_math::BlockPos::new(4, 62, 4),
            water,
        );
        // Water directly above => height 1.0 exactly.
        assert_eq!(
            fluid_height(&w, mc_math::BlockPos::new(2, 61, 2), Material::Water),
            1.0
        );
    }

    #[test]
    fn a_lone_source_block_is_eight_ninths_high() {
        let mut w = fixture::blank_world(1);
        let water = fixture::state_of("minecraft:water");
        w.set_block(mc_math::BlockPos::new(2, 60, 2), water);
        // Three of the four sampled columns are air (non-solid => contribute 1.0 each), the
        // fourth is the source: 1/9 * 11 over 11 samples.
        let h = fluid_height(&w, mc_math::BlockPos::new(2, 60, 2), Material::Water);
        assert!(h > 0.0 && h < 1.0, "{h}");
    }

    #[test]
    fn a_still_pool_has_no_flow_direction() {
        let mut w = fixture::blank_world(1);
        let water = fixture::state_of("minecraft:water");
        fixture::fill(
            &mut w,
            mc_math::BlockPos::new(0, 60, 0),
            mc_math::BlockPos::new(6, 60, 6),
            water,
        );
        assert_eq!(
            flow_direction(&w, mc_math::BlockPos::new(3, 60, 3), Material::Water),
            None
        );
    }

    #[test]
    fn flow_vector_uses_vanilla_cutoff_before_direction() {
        // Vec3.normalize narrows sqrt_double to float and rejects lengths strictly below
        // 1e-4. The just-below case catches the old `len > 0.0` normalization mutant.
        assert_eq!(normalize_flow_vector(0.000_099, 0.0, 0.0), (0.0, 0.0, 0.0));
        // At just above the cutoff the float sqrt is not exactly the input when widened back
        // to f64. Pinning its bits catches a mutant that uses f64::sqrt directly (which returns
        // exactly 1.0 here), while retaining Java's `MathHelper.sqrt_double` narrowing.
        let normalized = normalize_flow_vector(0.000_101, 0.0, 0.0);
        assert_eq!(normalized.0.to_bits(), 4_607_182_418_897_566_124);
        assert_eq!(normalized.1, 0.0);
        assert_eq!(normalized.2, 0.0);
    }

    #[test]
    fn flow_angle_preserves_vanilla_axes_quadrants_and_atan2() {
        // These are the float-cast vanilla MathHelper.atan2 results, not libm atan2 results.
        // (1, 2) is deliberately non-trivial: libm differs by multiple float ulps here.
        let cases = [
            (1.0, 0.0, 3_217_625_051),
            (0.0, 1.0, 0),
            (-1.0, 0.0, 1_070_141_403),
            (0.0, -1.0, 3_226_013_659),
            (1.0, 1.0, 3_209_236_443),
            (-1.0, 1.0, 1_061_752_795),
            (-1.0, -1.0, 3_229_307_857),
            (1.0, -1.0, 3_222_719_460),
            // Java is `MathHelper.atan2(z, x) - PI/2`, not `-atan2(z, x)`; this
            // non-cardinal case also bites a replacement with the platform libm atan2.
            (1.0, 2.0, 3_203_228_461),
        ];
        for (x, z, expected_bits) in cases {
            assert_eq!(
                flow_angle(x, z).expect("non-zero flow").to_bits(),
                expected_bits,
                "flow angle for ({x}, {z})"
            );
        }
        assert_eq!(flow_angle(0.0, 0.0), None);
    }

    #[test]
    fn flowing_surface_uv_angle_uses_the_live_vanilla_direction() {
        let mut w = fixture::blank_world(1);
        let water = fixture::state_of("minecraft:water");
        let center = BlockPos::new(3, 60, 3);
        w.set_block(center, water);
        w.set_block(center.east(), BlockState::new(water.block(), 1));
        w.set_block(center.south(), BlockState::new(water.block(), 2));

        // Neighbor level deltas produce (vx, vz) = (1, 2), exercising the renderer's live
        // top-face UV orientation rather than only testing the math helper in isolation.
        assert_eq!(
            flow_direction(&w, center, Material::Water)
                .expect("flowing water")
                .to_bits(),
            // The vector is normalized before atan2: (1, 2) / sqrt_double(5), so this is
            // intentionally distinct from the raw-coordinate helper control above.
            3_203_228_352
        );
    }

    #[test]
    fn a_water_surface_produces_a_top_face_on_the_translucent_layer() {
        let mut w = fixture::blank_world(1);
        let water = fixture::state_of("minecraft:water");
        fixture::fill(
            &mut w,
            mc_math::BlockPos::new(0, 16, 0),
            mc_math::BlockPos::new(15, 16, 15),
            water,
        );
        let mesh = super::super::mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
        assert!(
            !mesh.translucent.is_empty(),
            "water must land on the translucent layer"
        );
        assert!(mesh.solid.is_empty());
        // 256 cells: each contributes a top face (+ its mirrored underside where the 3x3 has a
        // non-full block) and a bottom face; the edge cells add sides. Just assert the top
        // faces all exist at the interpolated height rather than at y=1.
        let tops = mesh
            .translucent
            .vertices()
            .iter()
            .filter(|v| v.position()[1] > 0.0 && v.position()[1] < 1.0)
            .count();
        assert!(tops > 0, "no interpolated surface heights emitted");
    }

    /// java: `BlockFluidRenderer.renderFluid` gives the bottom face `color(f3, f3, f3, 1.0F)`
    /// with `f3 = 0.5F` (`:53`, `:146-149`), while the top (`f24 = f4 * f`, `:120-122`) and the
    /// sides (`f32 = f4 * f31 * f`, `:237-239`) all multiply by `colorMultiplier`. The bottom
    /// face is the one face vanilla leaves untinted.
    ///
    /// Swampland (biome 6) is the only 1.8.9 biome with a non-white `waterColorMultiplier`
    /// (`14745518` = `0xE0FFAE`, `BiomeGenSwamp.java:28`), so it is the only biome in which the
    /// difference is observable at all — which is exactly why it is the oracle here. Its
    /// green channel is `0xFF`, so the negative control has to be the **red** channel: the
    /// tinted faces must come out below `0xE0/2` while the bottom face is a flat `127`.
    #[test]
    fn the_underside_of_water_is_flat_grey_and_not_biome_tinted() {
        let mut w = World::new();
        w.set_has_sky_light(true);
        for cx in -1..=1 {
            for cz in -1..=1 {
                let payload = fixture::blank_column_payload(true, 6);
                w.apply_chunk_data(mc_math::ChunkPos::new(cx, cz), true, 0xFFFF, &payload)
                    .expect("swamp column");
            }
        }
        assert_eq!(super::super::tint::biome_at(&w, BlockPos::new(2, 20, 2)), 6);
        let water = fixture::state_of("minecraft:water");
        fixture::fill(
            &mut w,
            BlockPos::new(0, 20, 0),
            BlockPos::new(15, 20, 15),
            water,
        );
        let mesh = super::super::mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
        let verts = mesh.translucent.vertices();
        assert!(!verts.is_empty());

        // The bottom face is the only quad with all four vertices on the cell's floor plane;
        // a side quad has two vertices there and two at the fluid height.
        let bottom: Vec<_> = verts
            .chunks_exact(4)
            .filter(|q| q.iter().all(|v| v.position()[1] == 4.0))
            .collect();
        assert_eq!(bottom.len(), 256, "one bottom face per cell");
        for q in &bottom {
            for v in q.iter() {
                assert_eq!(
                    v.color,
                    [127, 127, 127, 255],
                    "the underside must not carry the swamp water tint"
                );
            }
        }

        // Negative control: the tint is live on this world. Some other face must be tinted, and
        // its red channel must be below the flat grey — `0xE0 = 224`, so a top face is 224 and
        // a shaded side is less, but *any* tinted face has red != green.
        let tinted = verts
            .iter()
            .find(|v| v.color[0] != v.color[1] || v.color[1] != v.color[2]);
        let tinted = tinted.expect("no tinted face at all: the oracle cannot bite");
        assert!(
            tinted.color[0] < tinted.color[1],
            "swamp water is 0xE0FFAE, so red < green: {:?}",
            tinted.color
        );
    }
}
