//! 1.7 Animations' "1.7 item edges" (option id
//! `translucent_held`, `mc_render::pipeline::old_item_edges`) on a real GPU with the real 1.8.9
//! jar.
//!
//! An "older pack" diamond sword is generated here and put in a folder pack over the jar: a solid
//! blade core, a semi-transparent (alpha 140) ring around it and a faint (alpha 14) anti-aliased
//! fringe, the kind of edge MC-70336 is about. The real bake builds vanilla's model from it, and
//! the real `FirstPersonPhase` draws it held in first person over a magenta clear, once with the
//! option off (vanilla 1.8.9) and once on.
//!
//! * Control: the vanilla iron sword (alpha only 0 or 255) is byte-identical either way, and the
//!   solid pipelines are never built for it.
//! * The old-pack sword: off, magenta shows through the gaps beside the faint fringe and through
//!   the semi-transparent ring; on, every pixel that was the item stays the item, and the gaps
//!   are closed.
//!
//! With `MC_RFX_PNG_DIR` set, the frames and a zoomed before/after strip are written there.
//! Skips (loudly) without a wgpu adapter or the vanilla jar; set `MC_JARS_DIR`.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use mc_assets::{Atlas, BakedModel, FolderResourcePack, ResourceManager, ZipResourcePack};
use mc_render::pipeline::FirstPersonPhase;
use mc_render::{
    Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase, SceneTextures,
};

const W: u32 = 640;
const H: u32 = 360;
const MAGENTA: [u8; 3] = [255, 0, 255];

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let dir = PathBuf::from(dir);
        return dir.is_dir().then_some(dir);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let dir = root.join("MCP-919-main").join("jars");
    dir.is_dir().then_some(dir)
}

/// A 16x16 sword like an older pack's: the blade runs from the bottom-left to the top-right with
/// a solid core, a semi-transparent ring and a faint anti-aliased fringe; a solid brown handle.
fn old_pack_sword() -> Vec<u8> {
    let mut rgba = vec![0u8; 16 * 16 * 4];
    for y in 0..16i32 {
        for x in 0..16i32 {
            // Distance from the blade's centre line x + y = 15, in texels.
            let d = ((x + y - 15) as f32).abs() / std::f32::consts::SQRT_2;
            let along = x - y; // -15 at the handle end .. 15 at the tip
            let texel = if along < -9 {
                // The handle: two texels wide, solid.
                (d <= 1.0).then_some([110, 70, 30, 255])
            } else if d <= 0.75 {
                Some([120, 220, 255, 255])
            } else if d <= 1.5 {
                Some([170, 240, 255, 140])
            } else if d <= 2.3 {
                Some([200, 250, 255, 14])
            } else {
                None
            };
            if let Some(texel) = texel {
                rgba[((y * 16 + x) * 4) as usize..][..4].copy_from_slice(&texel);
            }
        }
    }
    rgba
}

/// The client's `Shared`: the list owns one handle, the test keeps the other.
#[derive(Clone)]
struct Handle(Rc<RefCell<FirstPersonPhase>>);

impl RenderPhase for Handle {
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
        "first-person"
    }

    fn enabled(&self) -> bool {
        self.0.borrow().enabled()
    }

    fn needs_scene_color(&self) -> bool {
        self.0.borrow().needs_scene_color()
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.0.borrow_mut().record_scene(gpu, enc, view, cam, scene);
    }
}

/// Render `model` held in first person, fully raised, over magenta. Returns the RGBA frame and
/// whether the phase drew it with 1.7's edges.
fn render_held(
    gpu: &mut Gpu,
    atlas: &Atlas,
    model: BakedModel,
    old_edges: bool,
) -> (Vec<u8>, bool, bool) {
    let gui_3d = model.gui_3d;
    let mut phase = FirstPersonPhase::new(gpu);
    phase.set_atlas(gpu, atlas);
    phase.set_item_model(Some(model), gui_3d);
    phase.render_state.prev_equipped_progress = 1.0;
    phase.render_state.equipped_progress = 1.0;
    phase.old_item_edges = old_edges;
    let handle = Handle(Rc::new(RefCell::new(phase)));

    let mut camera = Camera::default();
    camera.set_viewport(W, H);
    // A wide hand FOV puts the whole blade in the picture (both draws share the view).
    camera.hand_fov = 120.0;
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::FIRST_PERSON, Box::new(handle.clone()));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &camera, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let phase = handle.0.borrow();
    (
        pixels,
        phase.item_drawn_with_old_edges(),
        phase.old_edge_pipelines_built(),
    )
}

fn is_magenta(pixels: &[u8], index: usize) -> bool {
    pixels[index * 4..index * 4 + 3] == MAGENTA
}

/// Pixels that show some magenta mixed into the item (blended over the clear), but are not the
/// clear itself.
fn tinted(pixels: &[u8], index: usize) -> bool {
    let [r, g, b] = [
        pixels[index * 4],
        pixels[index * 4 + 1],
        pixels[index * 4 + 2],
    ];
    !is_magenta(pixels, index) && r > 150 && b > 150 && g < 200 && g + 40 < r.min(b)
}

/// Write the frames and a zoomed before/after strip of the item's bounding box.
fn save_pngs(dir: &Path, vanilla: &[u8], old: &[u8]) {
    std::fs::create_dir_all(dir).expect("png dir");
    let frame =
        |pixels: &[u8]| image::RgbaImage::from_raw(W, H, pixels.to_vec()).expect("frame size");
    frame(vanilla)
        .save(dir.join("item_edges_vanilla_frame.png"))
        .expect("save");
    frame(old)
        .save(dir.join("item_edges_17_frame.png"))
        .expect("save");
    // Bounding box of everything that is not the clear in either frame.
    let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0, 0);
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            if !is_magenta(vanilla, i) || !is_magenta(old, i) {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    let (x0, y0) = (x0.saturating_sub(4), y0.saturating_sub(4));
    let (x1, y1) = ((x1 + 4).min(W - 1), (y1 + 4).min(H - 1));
    let (cw, ch) = (x1 - x0 + 1, y1 - y0 + 1);
    const ZOOM: u32 = 3;
    const GAP: u32 = 12;
    let mut strip = image::RgbaImage::from_pixel(
        cw * ZOOM * 2 + GAP,
        ch * ZOOM,
        image::Rgba([40, 40, 40, 255]),
    );
    for (panel, pixels) in [vanilla, old].into_iter().enumerate() {
        let left = panel as u32 * (cw * ZOOM + GAP);
        for y in 0..ch * ZOOM {
            for x in 0..cw * ZOOM {
                let i = (((y0 + y / ZOOM) * W + x0 + x / ZOOM) * 4) as usize;
                let p = [pixels[i], pixels[i + 1], pixels[i + 2], 255];
                strip.put_pixel(left + x, y, image::Rgba(p));
            }
        }
    }
    strip
        .save(dir.join("item_edges_before_after.png"))
        .expect("save");
}

#[test]
fn old_item_edges_close_the_gaps_and_leave_vanilla_items_alone() {
    let Some(mut gpu) = gpu() else {
        return;
    };
    let Some(jar) = jars_dir()
        .map(|dir| dir.join("versions").join("1.8.9").join("1.8.9.jar"))
        .filter(|jar| jar.is_file())
    else {
        eprintln!("SKIP: no vanilla jar; set MC_JARS_DIR");
        return;
    };
    // The "older pack": one generated diamond sword texture.
    let pack = tempfile::tempdir().expect("tempdir");
    let items = pack.path().join("assets/minecraft/textures/items");
    std::fs::create_dir_all(&items).expect("pack dirs");
    image::RgbaImage::from_raw(16, 16, old_pack_sword())
        .expect("16x16")
        .save(items.join("diamond_sword.png"))
        .expect("write the pack's sword");

    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).expect("open the jar")));
    rm.push_pack(Box::new(
        FolderResourcePack::open(pack.path()).expect("open the pack"),
    ));
    let (baked, _report) = mc_assets::bake_all(&rm, 4096).expect("bake the jar and the pack");
    let atlas = baked.atlas.clone();
    mc_assets::install(baked);
    let model = |name: &str| {
        let id = mc_data::item_by_name(name).expect("a real 1.8.9 item");
        let model = mc_assets::item_model_for(id).clone();
        assert!(
            !model.is_empty() && !model.gui_3d,
            "{name}: a flat generated model"
        );
        model
    };

    // Control: the vanilla iron sword is untouched by the option, pixel for pixel.
    let iron = model("iron_sword");
    let (iron_off, _, _) = render_held(&mut gpu, &atlas, iron.clone(), false);
    let (iron_on, solid, built) = render_held(&mut gpu, &atlas, iron, true);
    assert!(!solid && !built, "a 0/255 item must keep the vanilla path");
    assert_eq!(iron_off, iron_on, "the option changed a vanilla item");

    // The older pack's sword.
    let sword = model("diamond_sword");
    let (vanilla, solid, built) = render_held(&mut gpu, &atlas, sword.clone(), false);
    assert!(!solid && !built, "off is vanilla");
    let (old, solid, built) = render_held(&mut gpu, &atlas, sword, true);
    assert!(
        solid && built,
        "on: the 1.7-edged copy, drawn without blending"
    );

    let pixels = (W * H) as usize;
    let item = |frame: &[u8]| (0..pixels).filter(|&i| !is_magenta(frame, i)).count();
    let tint = |frame: &[u8]| (0..pixels).filter(|&i| tinted(frame, i)).count();
    // Every pixel the vanilla draw covered is still covered: 1.7 draws a superset.
    let lost = (0..pixels)
        .filter(|&i| !is_magenta(&vanilla, i) && is_magenta(&old, i))
        .count();
    let closed = (0..pixels)
        .filter(|&i| is_magenta(&vanilla, i) && !is_magenta(&old, i))
        .count();
    eprintln!(
        "item pixels: vanilla {} -> 1.7 {}; gap pixels closed {closed}; lost {lost}; \
         magenta-tinted (see-through) pixels: vanilla {} -> 1.7 {}",
        item(&vanilla),
        item(&old),
        tint(&vanilla),
        tint(&old),
    );
    assert_eq!(lost, 0, "1.7 edges must never uncover the background");
    assert!(
        closed > 20,
        "the side gaps beside the faint fringe stay open: {closed}"
    );
    assert!(
        tint(&vanilla) > 20 && tint(&old) == 0,
        "the semi-transparent ring: blended in vanilla, solid in 1.7"
    );

    if let Some(dir) = std::env::var_os("MC_RFX_PNG_DIR") {
        save_pngs(Path::new(&dir), &vanilla, &old);
    }
}
