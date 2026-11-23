//! GPU readbacks for the two mods that draw through mc-render:
//!
//! * **Block Overlay** (`BlockOverlayPhase::selection_style`): the fill and the wide outline
//!   reach the framebuffer where the block projects, a box behind the camera draws nothing, and
//!   `None` is vanilla's frame byte for byte, also after the style was on.
//! * **Fire Height** (`OverlayPhase::fire_style`): the offset lowers the real `fire_layer_1`
//!   quads, the alpha fades them, a zero offset at vanilla's alpha is vanilla's frame byte for
//!   byte, and `None` is vanilla's frame after the style was on.
//!
//! Same technique as `gap_a01_block_overlay.rs`: the real phase through the real `PhaseList`
//! into a headless target. The fire tests skip (printing `SKIP:`) without the vanilla jar; point
//! `MC_JARS_DIR` at a `jars/` directory to override its location.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use mc_assets::{ResourceManager, ZipResourcePack};
use mc_math::{BlockPos, DVec3};
use mc_render::post::overlay::{FireOverlayStyle, FIRST_PERSON_FIRE_ALPHA};
use mc_render::post::OverlayPhase;
use mc_render::terrain::{fixture, BlockOverlayPhase, SelectionBox, SelectionStyle};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;
/// The clear colour: mid grey, so both darkening (vanilla's black line) and colour show.
const GREY: wgpu::Color = wgpu::Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};

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
            // Turn wgpu validation errors into test failures instead of silent no-draws.
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

/// A phase the test keeps a handle on between frames (the client's `Shared`).
struct Handle<T>(Rc<RefCell<T>>);

impl<T: RenderPhase> RenderPhase for Handle<T> {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.0.borrow_mut().record(gpu, enc, view, cam);
    }

    fn name(&self) -> &'static str {
        self.0.borrow().name()
    }

    fn enabled(&self) -> bool {
        self.0.borrow().enabled()
    }
}

/// Render `phase` alone at `order` over the grey clear colour and read the frame back.
fn render<T: RenderPhase + 'static>(
    gpu: &mut Gpu,
    cam: &Camera,
    order: PhaseOrder,
    phase: &Rc<RefCell<T>>,
) -> Vec<u8> {
    let mut phases = PhaseList::new();
    phases.clear_color = GREY;
    phases.insert(order, Box::new(Handle(Rc::clone(phase))));
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    drop(frame);
    gpu.read_pixels().expect("readback")
}

fn at(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * WIDTH + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// Every pixel that differs from `base`, as `(x, y)`.
fn changed_pixels(base: &[u8], other: &[u8]) -> Vec<(u32, u32)> {
    assert_eq!(base.len(), other.len());
    (0..base.len())
        .step_by(4)
        .filter(|&i| base[i..i + 4] != other[i..i + 4])
        .map(|i| {
            let px = (i / 4) as u32;
            (px % WIDTH, px / WIDTH)
        })
        .collect()
}

// ---- Block Overlay ------------------------------------------------------------------------

/// Eye at the centre of a block face, looking down `+Z` (vanilla yaw `0` is south).
fn camera() -> Camera {
    let mut cam = Camera {
        pos: DVec3::new(8.5, 17.5, 0.5),
        yaw: 0.0,
        pitch: 0.0,
        ..Camera::default()
    };
    cam.set_viewport(WIDTH, HEIGHT);
    cam.update();
    cam
}

/// Project a world point to framebuffer pixels through the matrices the shaders use.
fn project(cam: &Camera, world: DVec3) -> (f32, f32) {
    let ndc = (cam.proj * cam.view).project_point3(cam.relative(world));
    (
        (ndc.x * 0.5 + 0.5) * WIDTH as f32,
        (0.5 - ndc.y * 0.5) * HEIGHT as f32,
    )
}

fn overlay(gpu: &Gpu, selection: Option<SelectionBox>) -> Rc<RefCell<BlockOverlayPhase>> {
    let mut phase = BlockOverlayPhase::new(gpu, None);
    phase.selection = selection;
    Rc::new(RefCell::new(phase))
}

/// Opaque red, 4 px wide; green fill at 50%.
fn loud_style() -> SelectionStyle {
    SelectionStyle::from_argb(0xFFFF_0000, 4.0, 0x8000_FF00)
}

fn stone_at(pos: BlockPos) -> SelectionBox {
    SelectionBox::for_block(pos, fixture::state_of("minecraft:stone"))
}

#[test]
fn the_styled_box_fills_the_face_and_draws_a_wide_coloured_outline() {
    let Some(mut gpu) = gpu() else { return };
    let cam = camera();
    let pos = BlockPos::new(8, 17, 8);
    let empty_phase = overlay(&gpu, None);
    let empty = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &empty_phase);
    let vanilla_phase = overlay(&gpu, Some(stone_at(pos)));
    let vanilla = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &vanilla_phase);
    let styled_phase = overlay(&gpu, Some(stone_at(pos)));
    styled_phase.borrow_mut().selection_style = Some(loud_style());
    let styled = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &styled_phase);

    // The face's centre: untouched by vanilla's wireframe, filled by the style. Only the north
    // face is front-facing from here, so the fill is exactly one 50% layer over the grey.
    let (cx, cy) = project(&cam, DVec3::new(8.5, 17.5, 8.0));
    let (cx, cy) = (cx as u32, cy as u32);
    assert_eq!(at(&vanilla, cx, cy), at(&empty, cx, cy));
    let fill = at(&styled, cx, cy);
    for (got, want) in fill.iter().zip([64u8, 191, 64]) {
        assert!(
            got.abs_diff(want) <= 2,
            "fill pixel {fill:?}, expected ~[64, 191, 64]"
        );
    }

    // The near face's top edge, at its midpoint: vanilla's line is at most 2 px tall there, the
    // styled outline 3 to 5 px of pure red.
    let (ex, ey) = project(&cam, DVec3::new(8.5, 18.002, 7.998));
    let (ex, ey) = (ex as u32, ey.round() as i64);
    let column = |frame: &[u8], keep: &dyn Fn([u8; 4]) -> bool| {
        (ey - 8..=ey + 8)
            .filter(|&y| keep(at(frame, ex, y as u32)))
            .count()
    };
    let red = column(&styled, &|p| p[0] > 240 && p[1] < 15 && p[2] < 15);
    assert!(
        (3..=5).contains(&red),
        "{red} red outline pixels across the top edge"
    );
    let dark = column(&vanilla, &|p| p[0] < 100);
    assert!(
        (1..=2).contains(&dark),
        "{dark} vanilla line pixels across the top edge"
    );

    // The whole outline is many times vanilla's area, and stays around the block.
    let vanilla_px = changed_pixels(&empty, &vanilla);
    let styled_px = changed_pixels(&empty, &styled);
    assert!(styled_px.len() > 3 * vanilla_px.len());
    let (min_x, max_x) = (
        styled_px.iter().map(|p| p.0).min().unwrap(),
        styled_px.iter().map(|p| p.0).max().unwrap(),
    );
    let (vmin_x, vmax_x) = (
        vanilla_px.iter().map(|p| p.0).min().unwrap(),
        vanilla_px.iter().map(|p| p.0).max().unwrap(),
    );
    assert!(
        min_x + 3 >= vmin_x && max_x <= vmax_x + 3,
        "styled box spans x {min_x}..{max_x}, vanilla's {vmin_x}..{vmax_x}"
    );
}

#[test]
fn switching_the_style_off_restores_vanillas_frame_byte_for_byte() {
    let Some(mut gpu) = gpu() else { return };
    let cam = camera();
    let phase = overlay(&gpu, Some(stone_at(BlockPos::new(8, 17, 8))));
    let before = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &phase);
    phase.borrow_mut().selection_style = Some(loud_style());
    let on = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &phase);
    phase.borrow_mut().selection_style = None;
    let after = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &phase);
    assert!(
        !changed_pixels(&before, &on).is_empty(),
        "the style drew nothing"
    );
    assert!(
        changed_pixels(&before, &after).is_empty(),
        "mod off after on must be vanilla's exact frame"
    );
}

#[test]
fn a_styled_box_behind_or_around_the_camera_draws_nothing_wrong() {
    let Some(mut gpu) = gpu() else { return };
    let cam = camera();
    let empty_phase = overlay(&gpu, None);
    let empty = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &empty_phase);

    // Behind: both ends of every edge are behind the near plane, so nothing may smear across.
    let behind = overlay(&gpu, Some(stone_at(BlockPos::new(8, 17, -8))));
    behind.borrow_mut().selection_style = Some(loud_style());
    let frame = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &behind);
    assert!(
        changed_pixels(&empty, &frame).is_empty(),
        "a styled box behind the camera must be clipped away entirely"
    );

    // Around the eye (the block the camera is in): every face is seen from behind, so there is
    // no fill; the edges crossing the near plane are clipped in the vertex shader (no NaN, no
    // validation error) and the screen centre stays clear.
    let around = overlay(&gpu, Some(stone_at(BlockPos::new(8, 17, 0))));
    around.borrow_mut().selection_style = Some(loud_style());
    let frame = render(&mut gpu, &cam, PhaseOrder::BLOCK_OVERLAY, &around);
    let (cx, cy) = (WIDTH / 2, HEIGHT / 2);
    assert_eq!(at(&frame, cx, cy), at(&empty, cx, cy));
}

// ---- Fire Height --------------------------------------------------------------------------

fn fire_phase(gpu: &Gpu, rm: &ResourceManager) -> Rc<RefCell<OverlayPhase>> {
    let mut phase = OverlayPhase::new(gpu);
    phase.load_fire_texture(gpu, rm);
    assert!(phase.has_fire_texture(), "fire_layer_1 loads from the jar");
    phase.fire = true;
    Rc::new(RefCell::new(phase))
}

fn fire_camera() -> Camera {
    let mut cam = Camera::default();
    cam.set_viewport(WIDTH, HEIGHT);
    cam.update();
    cam
}

/// The topmost row any pixel differs from the clear grey in.
fn top_row(pixels: &[u8]) -> Option<u32> {
    let grey = at(pixels, 0, 0);
    (0..HEIGHT).find(|&y| (0..WIDTH).any(|x| at(pixels, x, y)[..3] != grey[..3]))
}

#[test]
fn fire_height_lowers_and_fades_the_first_person_fire() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let Some(mut gpu) = gpu() else { return };
    let cam = fire_camera();
    let phase = fire_phase(&gpu, &rm);
    let vanilla = render(&mut gpu, &cam, PhaseOrder::POST, &phase);
    let vanilla_top = top_row(&vanilla).expect("vanilla fire draws");
    // Vanilla's quads reach about 79% of the way up; the flame tips are a little lower.
    assert!(
        vanilla_top < HEIGHT / 2,
        "vanilla fire starts at row {vanilla_top} of {HEIGHT}"
    );

    // A zero offset at vanilla's alpha is vanilla's frame exactly.
    phase.borrow_mut().fire_style = Some(FireOverlayStyle {
        height_offset: 0.0,
        alpha: FIRST_PERSON_FIRE_ALPHA,
    });
    let zero = render(&mut gpu, &cam, PhaseOrder::POST, &phase);
    assert!(changed_pixels(&vanilla, &zero).is_empty());

    // The default -0.3: each 0.1 is about 14% of the screen, so ~43% (~82 px at 192) lower.
    phase.borrow_mut().fire_style = Some(FireOverlayStyle {
        height_offset: -0.3,
        alpha: FIRST_PERSON_FIRE_ALPHA,
    });
    let lowered = render(&mut gpu, &cam, PhaseOrder::POST, &phase);
    let lowered_top = top_row(&lowered).expect("lowered fire still draws");
    let lowered_by = lowered_top - vanilla_top;
    assert!(
        (60..=110).contains(&lowered_by),
        "the flames moved down {lowered_by} px ({vanilla_top} -> {lowered_top})"
    );

    // Half opacity. The fire blend replaces the target's alpha with the last layer's, so every
    // pixel a fire quad covers (alpha below the clear's 255) carries exactly half vanilla's
    // alpha. Colours move toward the grey: by exactly half where one copy covers a pixel, a
    // little less where the two copies overlap (two half layers are 70% opaque, not 49.5%), so
    // only the total is bounded.
    phase.borrow_mut().fire_style = Some(FireOverlayStyle {
        height_offset: 0.0,
        alpha: FIRST_PERSON_FIRE_ALPHA * 0.5,
    });
    let faded = render(&mut gpu, &cam, PhaseOrder::POST, &phase);
    let grey = at(&vanilla, 0, 0);
    let (mut covered, mut vanilla_distance, mut faded_distance) = (0u32, 0i64, 0i64);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let (v, f) = (at(&vanilla, x, y), at(&faded, x, y));
            if v[3] == 255 {
                assert_eq!(v, f, "({x}, {y}) is outside the fire quads");
                continue;
            }
            covered += 1;
            assert!(
                (2 * i32::from(f[3]) - i32::from(v[3])).abs() <= 2,
                "({x}, {y}) alpha: vanilla {v:?}, faded {f:?}"
            );
            for c in 0..3 {
                vanilla_distance += i64::from(v[c].abs_diff(grey[c]));
                faded_distance += i64::from(f[c].abs_diff(grey[c]));
            }
        }
    }
    let ratio = faded_distance as f64 / vanilla_distance as f64;
    println!(
        "fire: vanilla top row {vanilla_top}, lowered {lowered_top} (down {lowered_by} px of \
         {HEIGHT}); half opacity: {covered} px, colour distance ratio {ratio:.3}"
    );
    assert!(
        covered > 1_000,
        "only {covered} pixels under the fire quads"
    );
    assert!(
        (0.45..=0.75).contains(&ratio),
        "half opacity moved the colours {ratio:.3} of the way back to the grey"
    );

    // Mod off again: vanilla's frame.
    phase.borrow_mut().fire_style = None;
    let after = render(&mut gpu, &cam, PhaseOrder::POST, &phase);
    assert!(changed_pixels(&vanilla, &after).is_empty());
}
