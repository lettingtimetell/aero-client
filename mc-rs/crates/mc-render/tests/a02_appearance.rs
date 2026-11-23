//! Two entity appearance fixes, checked on a real GPU.
//!
//! These are the assertions that could not pass before the entity vertex grew a colour channel
//! and `EntityPhase` grew its three pipelines, and that no unit test can make: they read pixels
//! back out of a headless frame and compare them.
//!
//! Every test skips cleanly (printing `SKIP:`) when no wgpu adapter is available.

use mc_render::entity::layers::LayerBlend;
use mc_render::entity::render::EntityPhase;
use mc_render::entity::EntityVertex;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const SIZE: u32 = 96;

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        SIZE,
        SIZE,
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

/// A screen-filling quad four blocks in front of the camera, facing it.
fn quad(z: f32, color: [f32; 4]) -> Vec<EntityVertex> {
    let r = 4.0f32;
    let corners = [(-r, -r), (r, -r), (r, r), (-r, -r), (r, r), (-r, r)];
    corners
        .into_iter()
        .map(|(x, y)| EntityVertex {
            pos: [x, y, z],
            uv: [0.5, 0.5],
            // Facing the camera, which is looking down +Z by default.
            normal: [0.0, 0.0, -1.0],
            color,
        })
        .collect()
}

/// Renders one prepared phase and reads the centre pixel back as RGBA.
fn centre_pixel(gpu: &mut Gpu, phase: EntityPhase) -> [u8; 4] {
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));

    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back the frame");
    let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// The whole point of the colour channel: a tinted draw and an untinted
/// draw of the same geometry against the same texture must produce different pixels.
///
/// java: `GlStateManager.color`, which is what `LayerSheepWool` (`:31`) sets from
/// `EntitySheep.getDyeRgb` and what `RendererLivingEntity.setBrightness` sets for the hurt flash.
/// Before this the entity fragment shader had no colour input at all, so a red sheep and a white
/// sheep were byte-identical.
#[test]
fn a_tinted_draw_and_an_untinted_draw_differ_on_screen() {
    let Some(mut gpu) = headless() else { return };

    let white = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "flat", &[255, 255, 255, 255], 1, 1);
        phase.begin();
        phase.push("flat", &quad(4.0, EntityVertex::WHITE));
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };
    // EntitySheep.getDyeRgb(RED) = (0.6, 0.2, 0.2).
    let red = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "flat", &[255, 255, 255, 255], 1, 1);
        phase.begin();
        phase.push("flat", &quad(4.0, [0.6, 0.2, 0.2, 1.0]));
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };

    assert!(white[0] > 8, "the untinted quad drew something: {white:?}");
    assert_ne!(white, red, "the tint reached the framebuffer");
    // The tint is a multiply, so every channel is darker and green/blue lose the most.
    assert!(red[0] < white[0], "R {red:?} vs {white:?}");
    assert!(red[1] < white[1], "G {red:?} vs {white:?}");
    assert!(
        red[0] > red[1],
        "a 0.6/0.2/0.2 tint reads red, not grey: {red:?}"
    );
}

/// `Render.renderEntityOnFire` changes only the fixed-function lighting bit: it inherits the
/// world's `GL_GREATER, 0.1` alpha cutout and depth mask.  It is therefore neither the lit
/// ordinary entity pipeline nor an additive/translucent layer.
#[test]
fn fire_overlay_is_unlit_but_still_an_opaque_depth_writing_alpha_cutout() {
    let Some(mut gpu) = headless() else { return };

    let unlit = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "flat", &[255, 255, 255, 255], 1, 1);
        phase.begin();
        let mut geometry = quad(3.0, EntityVertex::WHITE);
        // A side-facing normal makes the ordinary entity light visibly less than white. The
        // overlay's `disableLighting()` must leave the sampled white texel unchanged instead.
        for vertex in &mut geometry {
            vertex.normal = [1.0, 0.0, 0.0];
        }
        phase.push_blended("flat", LayerBlend::OpaqueUnlit, &geometry);
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };
    assert!(
        unlit.iter().take(3).all(|&channel| channel > 240),
        "unlit fire must not receive the entity normal lighting: {unlit:?}"
    );

    let depth_writes = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "flame", &[255, 255, 255, 255], 1, 1);
        phase.upload_texture(&gpu, "behind", &[255, 0, 0, 255], 1, 1);
        phase.begin();
        phase.push_blended(
            "flame",
            LayerBlend::OpaqueUnlit,
            &quad(3.0, EntityVertex::WHITE),
        );
        // This is submitted second, as a later entity would be. The closer flame's normal
        // depth write must still occlude it.
        phase.push("behind", &quad(4.0, EntityVertex::WHITE));
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };
    assert!(
        depth_writes[0] > 240 && depth_writes[1] > 240 && depth_writes[2] > 240,
        "fire did not retain the world depth mask: {depth_writes:?}"
    );

    let alpha_cutout = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "transparent-flame", &[255, 255, 255, 0], 1, 1);
        phase.upload_texture(&gpu, "behind", &[255, 0, 0, 255], 1, 1);
        phase.begin();
        phase.push_blended(
            "transparent-flame",
            LayerBlend::OpaqueUnlit,
            &quad(3.0, EntityVertex::WHITE),
        );
        phase.push("behind", &quad(4.0, EntityVertex::WHITE));
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };
    assert!(
        alpha_cutout[0] > 100 && alpha_cutout[1] < 8 && alpha_cutout[2] < 8,
        "a fully transparent flame texel must be discarded, not write depth: {alpha_cutout:?}"
    );
}

/// The second appearance fix *and* the depth-function correction it depends on.
///
/// java: `LayerSpiderEyes`/`LayerEndermanEyes` re-render the **main model** — identical geometry
/// at identical depth — under an eyes texture with `blendFunc(1, 1)`. Two things have to be right
/// for that to appear at all:
///
/// 1. The depth comparison must be `GL_LEQUAL`, which `Minecraft.startGame` (`:540`) sets
///    globally. Under `Less` (what this pipeline had) every fragment of the second draw fails and
///    the layer is invisible no matter how correctly it was emitted.
/// 2. The pipeline must actually blend additively rather than overwrite.
///
/// So: draw a dim body, then draw the same quad again as an additive layer, and the result must
/// be *brighter than either alone* — which is only possible if both are true.
#[test]
fn an_additive_layer_at_the_bodys_own_depth_brightens_it() {
    let Some(mut gpu) = headless() else { return };

    let body_only = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "flat", &[80, 80, 80, 255], 1, 1);
        phase.begin();
        phase.push("flat", &quad(4.0, EntityVertex::WHITE));
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };

    let body_plus_eyes = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "flat", &[80, 80, 80, 255], 1, 1);
        phase.upload_texture(&gpu, "eyes", &[200, 0, 0, 255], 1, 1);
        phase.begin();
        phase.push("flat", &quad(4.0, EntityVertex::WHITE));
        // Exactly the same vertices — this is `getMainModel().render(...)` a second time.
        phase.push_blended(
            "eyes",
            LayerBlend::Additive,
            &quad(4.0, EntityVertex::WHITE),
        );
        phase.finish(&gpu);
        assert_eq!(
            phase.batch_count(),
            2,
            "two blend modes cannot share one draw"
        );
        centre_pixel(&mut gpu, phase)
    };

    assert!(body_only[0] > 8, "the body drew: {body_only:?}");
    assert!(
        body_plus_eyes[0] > body_only[0],
        "the additive layer must reach the screen and brighten: {body_only:?} -> {body_plus_eyes:?}"
    );
    // Additive, not replacement: the green channel keeps the body's contribution even though the
    // eyes texture has none.
    assert!(
        body_plus_eyes[1] >= body_only[1],
        "additive never darkens: {body_only:?} -> {body_plus_eyes:?}"
    );
}

/// java: `RendererLivingEntity.renderModel` (`:261`) draws an invisible
/// but still-visible-to-you entity at `color(1, 1, 1, 0.15)` with `blendFunc(770, 771)`.
///
/// The ghost must be *nearly* the background, but not identical to it and not the solid draw.
#[test]
fn the_invisibility_ghost_blends_instead_of_replacing() {
    let Some(mut gpu) = headless() else { return };

    let solid = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "flat", &[255, 255, 255, 255], 1, 1);
        phase.begin();
        phase.push("flat", &quad(4.0, EntityVertex::WHITE));
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };
    let ghost = {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "flat", &[255, 255, 255, 255], 1, 1);
        phase.begin();
        phase.push_blended("flat", LayerBlend::Ghost, &quad(4.0, [1.0, 1.0, 1.0, 0.15]));
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };

    assert!(ghost[0] > 0, "the ghost is visible: {ghost:?}");
    assert!(
        ghost[0] < solid[0] / 2,
        "a 0.15-alpha ghost is far dimmer than the solid body: {ghost:?} vs {solid:?}"
    );
}

/// java: `RendererLivingEntity.setBrightness` sets `GL_COMBINE_ALPHA = GL_REPLACE` on all three
/// combiner units (texture 0, then previous, previous), so while the red hurt/death tint is on,
/// the fragment alpha is the texel's and the ghost's `color(1, 1, 1, 0.15)` alpha is dropped: a
/// hurt invisible teammate flashes at full skin opacity. mc-rs multiplied the tint pass by the
/// vertex alpha and kept the 15% ghost (H7 2026-11-11).
#[test]
fn a_hurt_ghost_flashes_at_the_skins_own_opacity() {
    let Some(mut gpu) = headless() else { return };

    let hurt = |gpu: &mut Gpu, blend: LayerBlend, color: [f32; 4]| {
        let mut phase = EntityPhase::new(gpu);
        phase.upload_texture(gpu, "flat", &[255, 255, 255, 255], 1, 1);
        phase.begin();
        phase.push_blended("flat", blend, &quad(4.0, color));
        phase.finish(gpu);
        centre_pixel(gpu, phase)
    };
    let solid = hurt(&mut gpu, LayerBlend::LivingHurt, EntityVertex::WHITE);
    let ghost = hurt(&mut gpu, LayerBlend::LivingHurtGhost, [1.0, 1.0, 1.0, 0.15]);
    assert!(
        solid[0] > solid[1],
        "control: the hurt tint is red: {solid:?}"
    );
    assert_eq!(
        [ghost[0], ghost[1], ghost[2]],
        [solid[0], solid[1], solid[2]],
        "the hurt combiner's alpha is the texel's, so the ghost draws like the solid body"
    );
}

/// java: `Render.renderShadow` draws under the entity pass's `alphaFunc(GL_GREATER, 0.1)`, which
/// fixed-function GL applies to the *combined* alpha (shadow texel times the cell's vertex
/// alpha). A cell whose alpha has faded to 0.08 (14-16 blocks away in full light, or a jumping
/// entity, or dim light) is discarded outright; mc-rs tested the bare texel and kept drawing a
/// faint disc (H7 2026-11-11). Control: the same cell at 0.2 draws.
#[test]
fn a_faded_shadow_cell_is_discarded_by_the_combined_alpha_test() {
    let Some(mut gpu) = headless() else { return };

    let shadow_cell = |gpu: &mut Gpu, alpha: f32| {
        let mut phase = EntityPhase::new(gpu);
        phase.upload_texture(gpu, "disc", &[255, 255, 255, 255], 1, 1);
        phase.begin();
        phase.push_blended(
            "disc",
            LayerBlend::Translucent,
            &quad(4.0, [1.0, 1.0, 1.0, alpha]),
        );
        phase.finish(gpu);
        centre_pixel(gpu, phase)
    };
    let faded = shadow_cell(&mut gpu, 0.08);
    let visible = shadow_cell(&mut gpu, 0.2);
    assert_eq!(
        [faded[0], faded[1], faded[2]],
        [0, 0, 0],
        "texel 1.0 * alpha 0.08 fails GL_GREATER 0.1: {faded:?}"
    );
    assert!(visible[0] > 0, "control: alpha 0.2 passes: {visible:?}");
}

/// java: the ghost's lowered `alphaFunc(516, 0.003921569)` is still tested against
/// `texel * 0.15`, so a faint skin texel (13/255) survives as a ~2/255 smudge. Under the
/// ordinary 0.1 test (the `Translucent` family) the same texel is discarded, which is why the
/// ghost has its own pipeline.
#[test]
fn the_ghost_keeps_vanillas_lowered_threshold_on_the_combined_alpha() {
    let Some(mut gpu) = headless() else { return };

    let faint = |gpu: &mut Gpu, blend: LayerBlend| {
        let mut phase = EntityPhase::new(gpu);
        phase.upload_texture(gpu, "faint", &[255, 255, 255, 13], 1, 1);
        phase.begin();
        phase.push_blended("faint", blend, &quad(4.0, [1.0, 1.0, 1.0, 0.15]));
        phase.finish(gpu);
        centre_pixel(gpu, phase)
    };
    let ghost = faint(&mut gpu, LayerBlend::Ghost);
    let ordinary = faint(&mut gpu, LayerBlend::Translucent);
    assert!(ghost[0] > 0, "13/255 * 0.15 clears 1/255: {ghost:?}");
    assert_eq!(
        [ordinary[0], ordinary[1], ordinary[2]],
        [0, 0, 0],
        "control: it does not clear 0.1: {ordinary:?}"
    );
}

/// The alpha test still cuts out fully transparent texels for a translucent draw.
#[test]
fn a_translucent_draw_still_cuts_out_transparent_texels() {
    let Some(mut gpu) = headless() else { return };

    let mut phase = EntityPhase::new(&gpu);
    phase.upload_texture(&gpu, "hole", &[255, 255, 255, 0], 1, 1);
    phase.begin();
    phase.push_blended(
        "hole",
        LayerBlend::Translucent,
        &quad(4.0, [1.0, 1.0, 1.0, 0.15]),
    );
    phase.finish(&gpu);
    let px = centre_pixel(&mut gpu, phase);
    assert_eq!(
        [px[0], px[1], px[2]],
        [0, 0, 0],
        "a fully transparent texel is discarded, not blended: {px:?}"
    );
}

/// The UV scroll: `LayerCreeperCharge` translates the *texture* matrix, and
/// the equivalent here is shifting the batch's UVs. Against a two-texel texture that is a visible
/// colour change.
#[test]
fn a_scrolled_layer_samples_a_different_texel() {
    let Some(mut gpu) = headless() else { return };

    // A 2x1 texture: red at u < 0.5, blue at u >= 0.5.
    let texels = [255u8, 0, 0, 255, 0, 0, 255, 255];
    let mut sample = |offset: [f32; 2]| {
        let mut phase = EntityPhase::new(&gpu);
        phase.upload_texture(&gpu, "split", &texels, 2, 1);
        let mut verts = quad(4.0, EntityVertex::WHITE);
        // Start every vertex sampling the left (red) half.
        for v in &mut verts {
            v.uv = [0.25, 0.5];
        }
        mc_render::entity::offset_uv(&mut verts, offset);
        phase.begin();
        phase.push("split", &verts);
        phase.finish(&gpu);
        centre_pixel(&mut gpu, phase)
    };

    let unscrolled = sample([0.0, 0.0]);
    let scrolled = sample([0.5, 0.0]);
    assert!(
        unscrolled[0] > unscrolled[2],
        "unscrolled samples the red half: {unscrolled:?}"
    );
    assert!(
        scrolled[2] > scrolled[0],
        "scrolled by half a texture samples the blue half: {scrolled:?}"
    );
}
