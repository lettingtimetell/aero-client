//! 3D skins on a real GPU frame.
//!
//! A player is drawn through the ordinary [`EntityPhase`] (one texture, one batch) twice: the
//! flat vanilla body (`pose_and_render`) and the 3D Skins body (`skin3d::emit_player_body`).
//!
//! * With a transparent outer layer the two frames are byte-identical: the base body is
//!   untouched and an empty layer adds nothing.
//! * With a "hair" outer layer the 3D frame differs, and its silhouette reaches further out
//!   (the voxel hat stands 0.75 texels proud where the flat hat stands 0.5).
//!
//! `MC_X5_RENDER_OUT=<dir>` also writes the frames as PNGs; `MC_X5_SKIN=<png>` renders that
//! skin instead of the synthetic one (both for looking, not for the assertions). The GPU tests
//! skip cleanly (printing `SKIP:`) without an adapter.

use glam::Vec3;
use mc_render::entity::anim::{AnimParams, EntityModel};
use mc_render::entity::models::biped::PlayerModel;
use mc_render::entity::render::{entity_matrix, EntityPhase, EntityTransform};
use mc_render::entity::skin3d::{emit_player_body, SkinLayerMesh, ALL_WEAR};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const SIZE: u32 = 256;

fn headless(size: u32) -> Option<Gpu> {
    match Gpu::headless(
        size,
        size,
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

/// A 64x64 skin: an opaque base (a different colour per body part) and an outer layer drawn
/// by `outer(x, y)` (`None` = transparent).
fn skin(outer: impl Fn(usize, usize) -> Option<[u8; 3]>) -> Vec<u8> {
    let outer_rects: [(usize, usize, usize, usize); 6] = [
        (32, 0, 32, 16),
        (16, 32, 24, 16),
        (48, 48, 16, 16),
        (40, 32, 16, 16),
        (0, 48, 16, 16),
        (0, 32, 16, 16),
    ];
    let mut rgba = vec![0u8; 64 * 64 * 4];
    for y in 0..64 {
        for x in 0..64 {
            let in_outer = outer_rects
                .iter()
                .any(|&(u, v, w, h)| x >= u && x < u + w && y >= v && y < v + h);
            const SKIN: [u8; 4] = [205, 160, 125, 255];
            const SHIRT: [u8; 4] = [40, 90, 200, 255];
            const TROUSERS: [u8; 4] = [60, 50, 120, 255];
            let upper = (16..32).contains(&y);
            let legs = (upper && x < 16) || (y >= 48 && (16..32).contains(&x));
            let px = if in_outer {
                outer(x, y).map_or([0, 0, 0, 0], |c| [c[0], c[1], c[2], 255])
            } else if upper && (16..40).contains(&x) {
                SHIRT
            } else if legs {
                TROUSERS
            } else {
                // Head and both arms.
                SKIN
            };
            rgba[(y * 64 + x) * 4..][..4].copy_from_slice(&px);
        }
    }
    rgba
}

/// Brown hair: the hat's top face, the top three rows of its sides and back, a two-row fringe
/// on the front, and a jacket hem.
fn hair_skin() -> Vec<u8> {
    skin(|x, y| {
        let brown = Some([90, 55, 25]);
        // Hat: top (40..48, 0..8); sides/front/back rows start at y = 8.
        if (40..48).contains(&x) && y < 8 {
            return brown;
        }
        if (32..64).contains(&x) && (8..16).contains(&y) {
            let front = (40..48).contains(&x);
            let rows = if front { 2 } else { 3 };
            return (y < 8 + rows).then_some([90, 55, 25]);
        }
        // Jacket front (20..28, 36..48): a white hem on the last two rows.
        if (20..28).contains(&x) && (46..48).contains(&y) {
            return Some([235, 235, 235]);
        }
        None
    })
}

fn user_skin() -> Option<Vec<u8>> {
    let path = std::env::var_os("MC_X5_SKIN")?;
    let bytes = std::fs::read(path).ok()?;
    let texture =
        mc_render::entity::skin::decode_skin(&bytes, mc_render::entity::SkinModel::Classic).ok()?;
    Some(texture.rgba.to_vec())
}

/// Renders one player with `skin`, flat or 3D, facing `body_yaw`, and reads the frame back.
fn render(gpu: &mut Gpu, rgba: &[u8], voxels: bool, body_yaw: f32) -> Vec<u8> {
    render_on(gpu, rgba, voxels, body_yaw, wgpu::Color::BLACK)
}

fn render_on(
    gpu: &mut Gpu,
    rgba: &[u8],
    voxels: bool,
    body_yaw: f32,
    clear: wgpu::Color,
) -> Vec<u8> {
    let mut params = AnimParams::still();
    params.entity.child = false;
    let base = entity_matrix(&EntityTransform {
        rel: Vec3::new(0.0, -1.55, 2.2),
        body_yaw,
        ..Default::default()
    });
    let mut model = PlayerModel::new(0.0, false);
    let mut verts = Vec::new();
    if voxels {
        let mesh = SkinLayerMesh::build(rgba, 64, 64, false).expect("64x64 skin");
        emit_player_body(&mut model, &params, base, &mesh, ALL_WEAR, &mut verts);
    } else {
        model.pose_and_render(&params, base, &mut verts);
    }
    let mut phase = EntityPhase::new(gpu);
    phase.upload_texture(gpu, "skin", rgba, 64, 64);
    phase.begin();
    phase.push("skin", &verts);
    phase.finish(gpu);
    let mut list = PhaseList::new();
    list.clear_color = clear;
    list.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    list.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read back the frame")
}

fn lit(px: &[u8]) -> bool {
    px[0] > 4 || px[1] > 4 || px[2] > 4
}

/// `(left, top, right, bottom)` of the lit pixels.
fn bounds(pixels: &[u8]) -> (u32, u32, u32, u32) {
    let (mut l, mut t, mut r, mut b) = (SIZE, SIZE, 0, 0);
    for y in 0..SIZE {
        for x in 0..SIZE {
            if lit(&pixels[((y * SIZE + x) * 4) as usize..][..4]) {
                l = l.min(x);
                t = t.min(y);
                r = r.max(x);
                b = b.max(y);
            }
        }
    }
    (l, t, r, b)
}

fn lit_count(pixels: &[u8]) -> usize {
    pixels.chunks_exact(4).filter(|px| lit(px)).count()
}

fn dump(name: &str, pixels: &[u8]) {
    let Some(dir) = std::env::var_os("MC_X5_RENDER_OUT") else {
        return;
    };
    let path = std::path::Path::new(&dir).join(format!("{name}.png"));
    image::save_buffer(&path, pixels, SIZE, SIZE, image::ColorType::Rgba8)
        .expect("write the debug frame");
}

#[test]
fn an_empty_outer_layer_renders_byte_identical_to_the_flat_body() {
    let Some(mut gpu) = headless(SIZE) else {
        return;
    };
    let bare = skin(|_, _| None);
    for yaw in [180.0, 135.0] {
        let flat = render(&mut gpu, &bare, false, yaw);
        let voxel = render(&mut gpu, &bare, true, yaw);
        assert!(lit_count(&flat) > 1000, "the player reached the frame");
        assert!(
            flat == voxel,
            "yaw {yaw}: an empty layer must not change a pixel"
        );
    }
}

#[test]
fn voxel_hair_stands_out_of_the_head_on_screen() {
    let Some(mut gpu) = headless(SIZE) else {
        return;
    };
    let hair = hair_skin();
    // Facing the camera and turned away three quarters.
    for (yaw, name) in [(180.0, "front"), (115.0, "side")] {
        let flat = render(&mut gpu, &hair, false, yaw);
        let voxel = render(&mut gpu, &hair, true, yaw);
        dump(&format!("x5_hair_{name}_flat"), &flat);
        dump(&format!("x5_hair_{name}_3d"), &voxel);
        assert!(flat != voxel, "{name}: the 3D layer must change the frame");
        let (fl, ft, fr, _) = bounds(&flat);
        let (vl, vt, vr, _) = bounds(&voxel);
        // The voxel hat is 9.5 texels wide where the flat hat is 9: the head's silhouette grows.
        assert!(
            vt <= ft && (vr - vl) >= (fr - fl),
            "{name}: flat {fl}..{fr} top {ft}, 3D {vl}..{vr} top {vt}"
        );
        assert!(
            vt < ft || (vr - vl) > (fr - fl),
            "{name}: the hair must reach past the flat hat somewhere"
        );
    }
    if let Some(user) = user_skin() {
        let grey = wgpu::Color {
            r: 0.45,
            g: 0.55,
            b: 0.7,
            a: 1.0,
        };
        for (yaw, name) in [(180.0, "front"), (115.0, "side"), (30.0, "back")] {
            dump(
                &format!("x5_user_{name}_flat"),
                &render_on(&mut gpu, &user, false, yaw, grey),
            );
            dump(
                &format!("x5_user_{name}_3d"),
                &render_on(&mut gpu, &user, true, yaw, grey),
            );
        }
    }
}
