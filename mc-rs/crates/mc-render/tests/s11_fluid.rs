//! Fluid textures: the S11 -> S09 request, closed.
//!
//! `tests/s11_render.rs` used to carry `the_fluid_textures_are_known_to_be_missing_from_the_atlas`,
//! a deliberate test of the *gap*: no vanilla JSON model references
//! `blocks/{water,lava}_{still,flow}` (`BlockFluidRenderer.initAtlasSprites` binds them by name
//! instead), so S09's `bake_all` never stitched them, `sprite()` answered with its placeholder,
//! and every fluid UV addressed the whole atlas. `bake_all` now also stitches
//! `mc_assets::model::BUILTIN_TEXTURES`, its port of `ModelBakery.LOCATIONS_BUILTIN_TEXTURES`,
//! and this file is the inverted test.
//!
//! Three independent links, so the loop is genuinely closed rather than self-consistent:
//!
//! 1. **The mesher's UVs are the sprite's rect** — asserted here, exactly, against the four
//!    corners `BlockFluidRenderer`'s still-water branch produces (`getInterpolatedU(0/16)`).
//! 2. **That rect holds the real pixels** — `mc-assets`'
//!    `the_water_sprite_region_is_a_byte_exact_copy_of_frame_zero` byte-diffs the stitched
//!    region against `water_still.png` frame 0.
//! 3. **What reaches the framebuffer is that texel** — asserted here on the real GPU, with the
//!    expected colour computed on the CPU from the atlas bytes through the whole chain (plain
//!    GL_RGBA byte sample, vertex colour, lightmap, alpha blend) rather than compared against a
//!    snapshot.

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{fixture, fluid, mesh_section, TerrainPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use std::path::{Path, PathBuf};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_atlas() -> Option<mc_assets::Atlas> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return None;
    }
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let (assets, report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
    assert!(report.is_clean());
    let atlas = assets.atlas.clone();
    mc_assets::model::install(assets);
    let _ = mc_render::terrain::tint::load_colormaps(&rm);
    Some(atlas)
}

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        WIDTH,
        HEIGHT,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => {
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

/// A slab of still water floating in air, so nothing is drawn behind it and the destination of
/// the alpha blend is the (black) clear colour.
fn water_slab() -> mc_world::World {
    let mut w = fixture::blank_world(1);
    fixture::fill(
        &mut w,
        BlockPos::new(2, 20, 2),
        BlockPos::new(13, 20, 13),
        fixture::state_of("minecraft:water"),
    );
    w
}

/// The inverted gap test.
#[test]
fn the_fluid_textures_are_in_the_atlas_now() {
    let Some(_atlas) = vanilla_atlas() else {
        return;
    };
    assert!(
        fluid::fluid_sprites_available(),
        "bake_all must stitch water_still/water_flow/lava_still/lava_flow"
    );
    for path in fluid::FLUID_SPRITES {
        let loc = mc_assets::ResourceLocation::minecraft(path);
        let s = mc_assets::sprite(&loc);
        assert_ne!(s.uv, mc_assets::sprite::UvRect::FULL, "{path}");
        assert!(s.frames.len() > 1, "{path} should be animated");
    }
}

/// Link 1: the four UVs a still-water top face emits are *exactly* the `water_still` sprite's
/// rect corners — java: `BlockFluidRenderer.renderFluid`'s `f12 < -999` branch, which is
/// `getInterpolatedU(0)`/`getInterpolatedU(16)` and the same for V.
#[test]
fn a_still_water_top_face_uses_the_water_still_sprite_rect_exactly() {
    let Some(_atlas) = vanilla_atlas() else {
        return;
    };

    let world = water_slab();
    let mesh = mesh_section(&world, ChunkSectionPos::new(0, 1, 0));
    assert!(!mesh.translucent.is_empty(), "water must mesh");

    let loc = mc_assets::ResourceLocation::minecraft("textures/blocks/water_still");
    let uv = mc_assets::sprite(&loc).uv;
    let expected = [
        [uv.min_u, uv.min_v],
        [uv.min_u, uv.max_v],
        [uv.max_u, uv.max_v],
        [uv.max_u, uv.min_v],
    ];

    // The top face of an interior cell: the first quad whose four vertices all sit at the
    // interpolated surface height (0 < y < 1) rather than at the cell's floor.
    let mut found = 0usize;
    for i in 0..mesh.translucent.quad_count() {
        let q = mesh.translucent.quad(i);
        // Section-local: the slab is at world y = 20 in section y = 1, so local y = 4.
        if !q
            .iter()
            .all(|v| v.position()[1] > 4.5 && v.position()[1] < 5.0)
        {
            continue;
        }
        // Skip the mirrored underside (java: the `shouldRenderSides(pos.up())` copy), which is
        // the same four vertices in the order `0, 3, 2, 1`. Told apart geometrically rather
        // than by UV, because it shares vertex 0 with the face it mirrors.
        if q[1].position()[2] <= q[0].position()[2] {
            continue;
        }
        for (v, e) in q.iter().zip(expected) {
            // To within T1.1's UV quantum, and no looser. The vertex format stores UV as a
            // `Unorm16` in units of 1/32768 of the atlas, so a sprite rect that is not itself on
            // that grid moves by at most half a quantum on the way in.
            //
            // The water sprites are exactly that case, and it is a **premise failure worth
            // recording**: `vertex.rs` reasons that a 512x512 atlas with sixteenth subdivisions
            // puts every UV on a 1/8192 grid, and `water_still`'s `min_u` (0.562_519_55, i.e.
            // 288.010 texels) is not. `mc_assets` insets animated sprites. Measured error here:
            // 1.1e-5 of the atlas, 0.006 of a texel, against a `Nearest` sampler — and
            // `s23_batching.rs` renders this same pool and still gets 0 of 307 200 framebuffer
            // bytes different, which is the assertion that actually matters.
            const UV_QUANTUM: f32 = 1.0 / 32768.0;
            let got = v.texcoord();
            for k in 0..2 {
                assert!(
                    (got[k] - e[k]).abs() <= UV_QUANTUM / 2.0 + f32::EPSILON,
                    "quad {i} UV mismatch: {got:?} vs {e:?} (quantum {UV_QUANTUM})"
                );
            }
        }
        found += 1;
    }
    assert!(found > 50, "only {found} still-water top faces matched");

    // ...and those are not the whole-atlas placeholder they used to be.
    assert!(
        uv.max_u - uv.min_u < 0.2 && uv.max_v - uv.min_v < 0.2,
        "the water sprite still spans the whole atlas: {uv:?}"
    );
}

/// Java samples the atlas uploaded with `GL_RGBA` as normalized bytes.
fn atlas_byte(c: u8) -> f32 {
    f32::from(c) / 255.0
}

/// Link 3, on the real GPU: look straight down at the floating water slab and check one
/// framebuffer pixel against the exact `water_still` texel its UV addresses.
///
/// Looking straight down is what makes this exact rather than approximate: the water surface is
/// then a plane of constant eye-space depth, so every point on it shares one clip `w` and the
/// projection restricted to that plane is *affine*. A chosen world point therefore maps to a
/// known pixel and a known UV with no perspective correction to model.
#[test]
fn a_rendered_water_surface_samples_the_real_water_texels() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    let world = water_slab();
    let mesh = mesh_section(&world, ChunkSectionPos::new(0, 1, 0));

    // Straight down over the middle of the slab.
    let mut cam = Camera {
        pos: DVec3::new(8.0, 40.0, 8.0),
        yaw: 0.0,
        pitch: 90.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    cam.update();

    let mut phases = PhaseList::new();
    // Black, so the alpha blend's destination contributes nothing and `out = src.rgb * src.a`.
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    let shared = TerrainPhase::install(&gpu, &atlas, &mut phases);

    // Read the surface height and the vertex colour/light straight off the mesh: this test is
    // about the *UV*, and tint/brightness are S11's own already-tested business.
    let (surface_y, vcolor, vlight) = {
        let mut out = None;
        'outer: for i in 0..mesh.translucent.quad_count() {
            let q = mesh.translucent.quad(i);
            // An interior top face, at the interpolated height, over block (7, 20, 7).
            if q[0].position()[0].floor() as i32 == 7
                && q[0].position()[2].floor() as i32 == 7
                && q.iter()
                    .all(|v| v.position()[1] > 4.5 && v.position()[1] < 5.0)
            {
                out = Some((q[0].position()[1], q[0].color, q[0].light));
                break 'outer;
            }
        }
        out.expect("no interior top face over block (7,*,7)")
    };
    // java: getFluidHeight over four all-source columns gives 1 - 1/9, then renderFluid's
    // `f11 = 0.001` epsilon. Section-local y, so the section origin's 16 is already subtracted.
    let expected_h = 1.0 - 1.0 / 9.0 - 0.001;
    // Half a T1.1 position quantum, and no looser. 8/9 of a block is not a multiple of 1/2048,
    // so the packed vertex format cannot carry it exactly; the tolerance is the format's, stated,
    // rather than a round number picked to make the test pass. Measured error: 1.94e-4 blocks.
    const POS_QUANTUM: f32 = 1.0 / 2048.0;
    assert!(
        (surface_y - (4.0 + expected_h)).abs() <= POS_QUANTUM / 2.0 + f32::EPSILON,
        "surface at local y {surface_y}, expected {} (quantum {POS_QUANTUM})",
        4.0 + expected_h
    );

    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        r.upload(&gpu, mesh, cam.pos);
    }
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let px = gpu.read_pixels().expect("readback");

    // Sample 5.5/16 into block (7, 7)'s top face, which is the centre of texel (5, 5).
    let (tx, ty) = (5usize, 5usize);
    let frac = (tx as f64 + 0.5) / 16.0;
    let world_pt = DVec3::new(
        7.0 + frac,
        16.0 + f64::from(surface_y),
        7.0 + (ty as f64 + 0.5) / 16.0,
    );
    let rel = cam.relative(world_pt);
    let clip = cam.view_proj() * glam::Vec4::new(rel.x, rel.y, rel.z, 1.0);
    let ndc = clip.truncate() / clip.w;
    let sx = ((ndc.x * 0.5 + 0.5) * WIDTH as f32).floor() as u32;
    let sy = ((1.0 - (ndc.y * 0.5 + 0.5)) * HEIGHT as f32).floor() as u32;
    assert!(sx < WIDTH && sy < HEIGHT, "sample point off screen");

    let o = ((sy * WIDTH + sx) * 4) as usize;
    let got = [px[o], px[o + 1], px[o + 2], px[o + 3]];

    // The expected colour, from the source PNG rather than from the atlas, so a wrong stitch
    // offset cannot cancel out.
    let jar = jars_dir()
        .expect("jars")
        .join("versions")
        .join("1.8.9")
        .join("1.8.9.jar");
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(mc_assets::ZipResourcePack::open(&jar).unwrap()));
    let direct = mc_assets::sprite::load_sprite(
        &rm,
        &mc_assets::ResourceLocation::minecraft("textures/blocks/water_still"),
    )
    .expect("water_still");
    let t = (ty * 16 + tx) * 4;
    let texel = &direct.frames[0].rgba[t..t + 4];

    // The lightmap texel the vertex's (sky, block) bytes address. `lm_uv` lands exactly on a
    // texel centre for every legal light level, so LINEAR filtering does no blending.
    let lm = mc_render::sky::lightmap::Lightmap::generate(
        mc_render::sky::lightmap::LightmapParams::default(),
    )
    .sample(vlight[0] / 16, vlight[1] / 16);

    let src_a = f32::from(texel[3]) / 255.0;
    let mut expected = [0u8; 3];
    for c in 0..3 {
        let normalized = atlas_byte(texel[c]);
        let modulated = normalized * (f32::from(vcolor[c]) / 255.0) * (f32::from(lm[c]) / 255.0);
        // wgpu::BlendState::ALPHA_BLENDING over a black destination.
        expected[c] = (modulated * src_a * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    eprintln!(
        "pixel ({sx},{sy}) = {got:?}; texel {texel:?}, vertex colour {vcolor:?}, lightmap {lm:?} \
         -> expected {expected:?}"
    );
    for c in 0..3 {
        assert!(
            got[c].abs_diff(expected[c]) <= 2,
            "channel {c}: framebuffer {got:?} vs expected {expected:?} from water_still texel \
             ({tx},{ty}) = {texel:?}"
        );
    }
    // ...and the sanity that makes the tolerance meaningful: water is unmistakably blue.
    assert!(
        got[2] > got[0] + 40 && got[2] > got[1] + 30,
        "the surface is not blue ({got:?}) — this is what a wrong sprite looks like"
    );
}
