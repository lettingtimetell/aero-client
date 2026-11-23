//! Anisotropic filtering of the block atlas on a real GPU.
//!
//! The atlas is synthetic: 64x64, four mip levels (so sprites sit on a 16-texel grid, as the
//! stitcher guarantees for mipmapLevels 4), one sprite of vertical black/white stripes (4 texels
//! each) in the middle cell, pure red in every other cell. The floor is 16 x 128 blocks of that
//! sprite, seen at a grazing angle; a second camera looks straight down at it from half a block.
//!
//! * Off is today: a renderer that never had AF, one switched on and off again, one at level 1
//!   and one whose atlas has no mip levels all render byte-identical frames, and switching off
//!   frees the AF pipelines.
//! * Magnified texels are untouched: close up, AF on is byte-identical to off.
//! * Minified, AF keeps the stripes that mipmapping greys out: more contrast far away, more with
//!   a higher level.
//! * No bleeding: with AF at 16x not one pixel picks up red from the neighbouring sprites.

use std::cell::RefCell;
use std::rc::Rc;

use mc_math::{ChunkSectionPos, DVec3};
use mc_render::terrain::mesher::ChunkMesh;
use mc_render::terrain::{TerrainPhase, TerrainRenderer, TerrainVertex};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 200;
const ATLAS: u32 = 64;
const CELL: u32 = 16;
/// Number of stripes texels per colour.
const STRIPE: u32 = 4;

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        WIDTH,
        HEIGHT,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => {
            gpu.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(gpu)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

/// Level 0: stripes in cell (1, 1), red elsewhere. Then 2x2 box-filtered levels down to 4x4, which
/// is what per-sprite mips give when every sprite is cell-aligned.
fn atlas(mip_levels: u32) -> mc_assets::Atlas {
    let mut level0 = vec![0u8; (ATLAS * ATLAS * 4) as usize];
    for y in 0..ATLAS {
        for x in 0..ATLAS {
            let i = ((y * ATLAS + x) * 4) as usize;
            let in_sprite = (CELL..2 * CELL).contains(&x) && (CELL..2 * CELL).contains(&y);
            let rgb = if in_sprite {
                if (x / STRIPE).is_multiple_of(2) {
                    [255, 255, 255]
                } else {
                    [0, 0, 0]
                }
            } else {
                [255, 0, 0]
            };
            level0[i..i + 3].copy_from_slice(&rgb);
            level0[i + 3] = 255;
        }
    }
    let mut mipmaps = Vec::new();
    let mut prev = level0.clone();
    let mut size = ATLAS;
    for _ in 0..mip_levels {
        let half = size / 2;
        let mut next = vec![0u8; (half * half * 4) as usize];
        for y in 0..half {
            for x in 0..half {
                for c in 0..4 {
                    let at =
                        |xx: u32, yy: u32| u32::from(prev[((yy * size + xx) * 4 + c) as usize]);
                    let sum = at(2 * x, 2 * y)
                        + at(2 * x + 1, 2 * y)
                        + at(2 * x, 2 * y + 1)
                        + at(2 * x + 1, 2 * y + 1);
                    next[((y * half + x) * 4 + c) as usize] = ((sum + 2) / 4) as u8;
                }
            }
        }
        mipmaps.push(next.clone());
        prev = next;
        size = half;
    }
    mc_assets::Atlas {
        texture: level0,
        mipmaps,
        w: ATLAS,
        h: ATLAS,
    }
}

/// One section of floor at local y = 0: 16 x 16 block quads, each mapping the whole striped
/// sprite (u along x, v along z), wound counter-clockwise seen from above.
fn floor_section(z: i32) -> ChunkMesh {
    let mut mesh = ChunkMesh::empty(ChunkSectionPos::new(0, 0, z));
    let (u0, u1) = (CELL as f32 / ATLAS as f32, 2.0 * CELL as f32 / ATLAS as f32);
    let (v0, v1) = (u0, u1);
    let color = [255, 255, 255, 255];
    // Full sky light, no block light: the neutral daylight column of the lightmap.
    let light = [240, 0, 0, 0];
    for bz in 0..16 {
        for bx in 0..16 {
            let (x, zz) = (bx as f32, bz as f32);
            mesh.solid.push_quad([
                TerrainVertex::new([x, 0.0, zz], [u0, v0], color, light),
                TerrainVertex::new([x, 0.0, zz + 1.0], [u0, v1], color, light),
                TerrainVertex::new([x + 1.0, 0.0, zz + 1.0], [u1, v1], color, light),
                TerrainVertex::new([x + 1.0, 0.0, zz], [u1, v0], color, light),
            ]);
        }
    }
    mesh
}

/// Eye 1.62 above the floor at its near edge, looking down the 128-block run.
fn grazing_camera() -> Camera {
    let mut cam = Camera {
        pos: DVec3::new(8.3, 1.62, -1.0),
        yaw: 0.0,
        pitch: 6.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    cam.update();
    cam
}

/// Half a block above the floor, looking straight down: every texel covers dozens of pixels.
fn close_camera() -> Camera {
    let mut cam = Camera {
        pos: DVec3::new(8.3, 0.5, 8.4),
        yaw: 0.0,
        pitch: 90.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    cam.update();
    cam
}

struct Scene {
    phases: PhaseList,
    terrain: Rc<RefCell<TerrainRenderer>>,
}

fn scene(gpu: &Gpu, mip_levels: u32, cam: &Camera) -> Scene {
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    let terrain = TerrainPhase::install(gpu, &atlas(mip_levels), &mut phases);
    {
        let mut r = terrain.borrow_mut();
        r.render_distance = 12;
        for z in 0..8 {
            r.upload(gpu, floor_section(z), cam.pos);
        }
    }
    Scene { phases, terrain }
}

fn frame(gpu: &mut Gpu, scene: &mut Scene, cam: &Camera) -> Vec<u8> {
    let target = gpu.acquire().expect("acquire");
    scene.phases.render(gpu, target.view(), cam, 0.0);
    drop(target);
    gpu.read_pixels().expect("readback")
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * WIDTH + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// Mean absolute difference between horizontally adjacent pixels along row `y` over the floor:
/// how much stripe contrast survives at that distance.
fn row_contrast(pixels: &[u8], y: u32) -> f32 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for x in WIDTH / 4..WIDTH * 3 / 4 {
        let a = pixel(pixels, x, y);
        let b = pixel(pixels, x + 1, y);
        if is_clear(a) || is_clear(b) {
            continue; // past the floor's end
        }
        sum += (f32::from(a[1]) - f32::from(b[1])).abs();
        n += 1.0;
    }
    if n == 0.0 {
        0.0
    } else {
        sum / n
    }
}

/// The screen row the floor at distance `z` (blocks ahead of the eye) lands on.
fn row_at_depth(cam: &Camera, z: f32) -> u32 {
    let clip = cam.view_proj() * glam::Vec4::new(0.0, -1.62, z, 1.0);
    let ndc_y = clip.y / clip.w;
    ((1.0 - ndc_y) * 0.5 * HEIGHT as f32) as u32
}

#[test]
fn off_is_todays_frame_and_switching_off_frees_the_pipelines() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let cam = grazing_camera();
    let mut fresh = scene(&gpu, 4, &cam);
    let never = frame(&mut gpu, &mut fresh, &cam);
    assert!(!fresh.terrain.borrow().anisotropy_allocated());

    let mut toggled = scene(&gpu, 4, &cam);
    toggled.terrain.borrow_mut().set_anisotropy(16);
    let on = frame(&mut gpu, &mut toggled, &cam);
    assert!(toggled.terrain.borrow().anisotropy_allocated());
    assert_ne!(on, never, "AF at 16x changes a grazing floor");
    toggled.terrain.borrow_mut().set_anisotropy(0);
    assert!(
        !toggled.terrain.borrow().anisotropy_allocated(),
        "off frees it"
    );
    assert_eq!(
        frame(&mut gpu, &mut toggled, &cam),
        never,
        "off again = never on"
    );

    toggled.terrain.borrow_mut().set_anisotropy(1);
    assert_eq!(toggled.terrain.borrow().anisotropy(), 0, "1x is off");
    assert_eq!(frame(&mut gpu, &mut toggled, &cam), never);

    // No mip levels (Mipmap Levels 0): AF stays inactive and the frame is the plain one.
    let mut flat_plain = scene(&gpu, 0, &cam);
    let flat_off = frame(&mut gpu, &mut flat_plain, &cam);
    let mut flat = scene(&gpu, 0, &cam);
    flat.terrain.borrow_mut().set_anisotropy(16);
    assert!(!flat.terrain.borrow().anisotropy_active());
    assert_eq!(frame(&mut gpu, &mut flat, &cam), flat_off);
    assert!(!flat.terrain.borrow().anisotropy_allocated());
}

#[test]
fn magnified_texels_stay_pixel_sharp() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let cam = close_camera();
    let mut off_scene = scene(&gpu, 4, &cam);
    let off = frame(&mut gpu, &mut off_scene, &cam);
    // The close-up really is nearest-neighbour stripes: only pure black and white texels.
    let grey = off
        .chunks_exact(4)
        .filter(|p| p[1] != 0 && p[1] < 200 && !is_clear([p[0], p[1], p[2], p[3]]))
        .count();
    assert!(
        grey < (WIDTH * HEIGHT / 50) as usize,
        "{grey} grey pixels close up"
    );
    for level in [2, 4, 8, 16] {
        let mut on_scene = scene(&gpu, 4, &cam);
        on_scene.terrain.borrow_mut().set_anisotropy(level);
        let on = frame(&mut gpu, &mut on_scene, &cam);
        assert!(on_scene.terrain.borrow().anisotropy_allocated());
        assert_eq!(on, off, "{level}x close up must be byte-identical to off");
    }
}

#[test]
fn minified_stripes_keep_their_contrast_and_more_with_a_higher_level() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let cam = grazing_camera();
    let rows = [
        ("10 blocks", row_at_depth(&cam, 10.0)),
        ("20 blocks", row_at_depth(&cam, 20.0)),
        ("40 blocks", row_at_depth(&cam, 40.0)),
    ];
    let mut off_scene = scene(&gpu, 4, &cam);
    let off = frame(&mut gpu, &mut off_scene, &cam);
    let mut previous: Vec<f32> = rows.iter().map(|&(_, y)| row_contrast(&off, y)).collect();
    eprintln!("off: {previous:?}");
    for level in [2, 4, 8, 16] {
        let mut on_scene = scene(&gpu, 4, &cam);
        on_scene.terrain.borrow_mut().set_anisotropy(level);
        let on = frame(&mut gpu, &mut on_scene, &cam);
        let contrast: Vec<f32> = rows.iter().map(|&(_, y)| row_contrast(&on, y)).collect();
        eprintln!("{level}x: {contrast:?}");
        for (i, &(label, _)) in rows.iter().enumerate() {
            assert!(
                contrast[i] >= previous[i] * 0.95,
                "{level}x at {label}: {} after {}",
                contrast[i],
                previous[i]
            );
        }
        previous = contrast;
    }
    // At 16x the far rows keep clearly more stripe contrast than mipmapping alone.
    let off_far: Vec<f32> = rows.iter().map(|&(_, y)| row_contrast(&off, y)).collect();
    for (i, &(label, _)) in rows.iter().enumerate().skip(1) {
        assert!(
            previous[i] > off_far[i] * 1.5 + 2.0,
            "16x at {label}: {} vs off {}",
            previous[i],
            off_far[i]
        );
    }
}

#[test]
fn no_tap_reads_a_neighbouring_sprite() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let cam = grazing_camera();
    // The lightmap may tint the grey sprite a little: measure that on the off frame, where every
    // tap is nearest-within-level inside the sprite.
    let mut off_scene = scene(&gpu, 4, &cam);
    let off = frame(&mut gpu, &mut off_scene, &cam);
    let tint = off
        .chunks_exact(4)
        .filter(|p| !is_clear([p[0], p[1], p[2], p[3]]))
        .map(|p| i32::from(p[0]) - i32::from(p[1]))
        .max()
        .unwrap_or(0);
    for level in [4, 16] {
        let mut on_scene = scene(&gpu, 4, &cam);
        on_scene.terrain.borrow_mut().set_anisotropy(level);
        let on = frame(&mut gpu, &mut on_scene, &cam);
        let mut floor_pixels = 0;
        for p in on.chunks_exact(4) {
            if is_clear([p[0], p[1], p[2], p[3]]) {
                continue;
            }
            floor_pixels += 1;
            assert!(
                i32::from(p[0]) - i32::from(p[1]) <= tint + 2,
                "{level}x: a floor pixel picked up red: {p:?} (tint {tint})"
            );
        }
        assert!(floor_pixels > (WIDTH * HEIGHT / 4) as usize);
    }
}

fn is_clear(p: [u8; 4]) -> bool {
    p == [0, 0, 255, 255]
}

/// Writes the grazing floor off and at 16x to `X8_DUMP_DIR` for a human to look at.
#[test]
#[ignore = "writes PNGs for a visual check; needs X8_DUMP_DIR"]
fn dump_images_for_a_visual_check() {
    let Some(dir) = std::env::var_os("X8_DUMP_DIR") else {
        eprintln!("SKIP: set X8_DUMP_DIR");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("dump dir");
    let Some(mut gpu) = headless() else {
        return;
    };
    let cam = grazing_camera();
    for level in [0, 4, 16] {
        let mut s = scene(&gpu, 4, &cam);
        s.terrain.borrow_mut().set_anisotropy(level);
        let pixels = frame(&mut gpu, &mut s, &cam);
        image::RgbaImage::from_raw(WIDTH, HEIGHT, pixels)
            .expect("size")
            .save(dir.join(format!("x8_aniso_{level}x.png")))
            .expect("save png");
    }
}
