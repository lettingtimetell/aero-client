//! The UI phase split behind the menu blur.
//!
//! The HUD and the open screen are one draw list. `UiPhase::set_split` draws the list in two
//! passes, the HUD half at `PhaseOrder::UI` and the screen half through `UiScreenPhase` at
//! `PhaseOrder::UI_SCREEN`, so the menu blur (`PhaseOrder::MENU_BLUR`, between them) can blur
//! the HUD with the world and leave the screen sharp. These tests render real frames on a
//! headless GPU and read them back. They skip (printing `SKIP:`) without an adapter; the text
//! batches need the vanilla jar and are left out without it.
//!
//! * Split anywhere (at the HUD's end, inside a batch, at either end of the list) with nothing
//!   between the halves: byte-identical to one unsplit pass.
//! * With the blur between the halves: the HUD half is blurred, the screen's opaque panel is
//!   exactly what an unblurred frame shows there.
//! * Split at 0 (the mod's "Blur HUD" off): the world is blurred, the HUD is not.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::post::blur::MenuBlurPhase;
use mc_render::{
    Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase,
    SceneTextures, HEADLESS_FORMAT,
};
use mc_ui::font::{FontData, FontRenderer};
use mc_ui::widget::{DrawList, UiPhase, UiScreenPhase};

const W: u32 = 320;
const H: u32 = 240;

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
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn font() -> Option<FontRenderer> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("SKIP (text batches only): jars/versions/1.8.9/1.8.9.jar not found");
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(FontRenderer::new(FontData::load(&rm).ok()?, false, false))
}

/// The client's `Shared<UiPhase>`: the list owns one handle, the screen phase another.
#[derive(Clone)]
struct SharedUi(Rc<RefCell<UiPhase>>);

impl RenderPhase for SharedUi {
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
        "ui"
    }
}

/// The client's `Shared<MenuBlurPhase>`.
#[derive(Clone)]
struct SharedBlur(Rc<RefCell<MenuBlurPhase>>);

impl RenderPhase for SharedBlur {
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
        MenuBlurPhase::NAME
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

/// The "world": vertical stripes four pixels wide, drawn before the UI.
struct World {
    pipeline: wgpu::RenderPipeline,
}

const WORLD_WGSL: &str = r#"
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let on = f32((u32(pos.x) / 4u) % 2u);
    return vec4<f32>(0.2 + 0.6 * on, 0.5, 0.8 - 0.6 * on, 1.0);
}
"#;

impl World {
    fn new(gpu: &Gpu) -> World {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("x10 world"),
                source: wgpu::ShaderSource::Wgsl(WORLD_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("x10 world"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("x10 world"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HEADLESS_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        World { pipeline }
    }
}

impl RenderPhase for World {
    fn name(&self) -> &'static str {
        "test/world"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, "x10 world", DepthAttachment::None);
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }
}

/// HUD rows and the open screen, the way `build_ui` orders them. Returns the list and the vertex
/// count at the end of the HUD (where the client splits it).
///
/// HUD: black/white stripes along the bottom (the "hotbar") and a translucent chat backdrop with
/// text. Screen: vanilla's dark gradient over everything, an opaque panel, text on the panel.
/// Solid merging is on, as in the live frame, so solid rects extend neighbouring text batches
/// and some batches straddle the split.
fn scene(font: &mut Option<FontRenderer>) -> (DrawList, usize) {
    let mut list = DrawList::new();
    list.enable_solid_merging();
    for i in 0..40 {
        let x = i * 8;
        list.rect(x, 200, x + 4, 232, 0xFFFF_FFFF);
        list.rect(x + 4, 200, x + 8, 232, 0xFF00_0000);
    }
    list.rect(2, 150, 160, 190, 0x8000_0000);
    if let Some(font) = font {
        font.draw_string_with_shadow(&mut list, "<Steve> gg", 4.0, 160.0, 0xFF_FFFF);
    }
    let hud_end = list.vertices().len();
    // `GuiScreen.drawWorldBackground`: 0xC0101010 -> 0xD0101010.
    list.gradient_rect(0, 0, W as i32, H as i32, 0xC010_1010, 0xD010_1010);
    if let Some(font) = font {
        font.draw_string(&mut list, "Inventory", 120.0, 40.0, 0x40_4040, false);
    }
    list.rect(100, 60, 220, 140, 0xFFC6_C6C6);
    list.rect(110, 70, 128, 88, 0xFF8B_8B8B);
    if let Some(font) = font {
        font.draw_string_with_shadow(&mut list, "64", 118.0, 80.0, 0xFF_FFFF);
    }
    (list, hud_end)
}

fn new_ui(gpu: &Gpu, font: &Option<FontRenderer>) -> Rc<RefCell<UiPhase>> {
    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    if let Some(font) = font {
        phase.renderer.upload_ascii_font(gpu, font.data());
    }
    Rc::new(RefCell::new(phase))
}

/// One frame: the world, the UI (split at `split`), optionally the blur between the halves.
fn render(
    gpu: &mut Gpu,
    font: &mut Option<FontRenderer>,
    split: Option<usize>,
    blur: Option<&SharedBlur>,
) -> Vec<u8> {
    let ui = new_ui(gpu, font);
    let (list, _) = scene(font);
    ui.borrow_mut().submit(list);
    ui.borrow_mut().set_split(split);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(World::new(gpu)));
    phases.insert(PhaseOrder::UI, Box::new(SharedUi(ui.clone())));
    if let Some(blur) = blur {
        phases.insert(PhaseOrder::MENU_BLUR, Box::new(blur.clone()));
    }
    phases.insert(PhaseOrder::UI_SCREEN, Box::new(UiScreenPhase(ui.clone())));
    assert_eq!(
        phases.names().filter(|n| *n == UiScreenPhase::NAME).count(),
        1
    );
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

/// Max minus min of the red channel along row `y`, over `x0..x1`.
fn contrast(px: &[u8], y: u32, x0: u32, x1: u32) -> i32 {
    let row: Vec<i32> = (x0..x1).map(|x| i32::from(pixel(px, x, y)[0])).collect();
    row.iter().max().unwrap() - row.iter().min().unwrap()
}

fn blur(radius: f32) -> SharedBlur {
    let mut phase = MenuBlurPhase::new();
    // `radius` physical pixels on this 240-pixel-high frame.
    phase.set_radius(radius * 1080.0 / H as f32);
    phase.set_active(true);
    SharedBlur(Rc::new(RefCell::new(phase)))
}

#[test]
fn a_split_list_draws_exactly_the_pixels_of_one_pass() {
    let Some(mut gpu) = gpu() else { return };
    let mut font = font();
    let (list, hud_end) = scene(&mut font);
    let len = list.vertices().len();
    assert!(0 < hud_end && hud_end < len);
    // A batch straddling the HUD's end exists only with solid merging and text; find one
    // anyway so the partial-batch path is exercised whenever it can be.
    let straddle = list
        .batches()
        .iter()
        .find(|b| b.range.start < hud_end && b.range.end > hud_end)
        .map(|b| (b.range.start + b.range.end) / 2);
    let reference = render(&mut gpu, &mut font, None, None);
    let mut splits = vec![0, hud_end, hud_end + 3, len / 2, len - 1, len];
    splits.extend(straddle);
    for at in splits {
        let got = render(&mut gpu, &mut font, Some(at), None);
        assert!(
            got == reference,
            "splitting at vertex {at} of {len} changed the frame"
        );
    }
}

#[test]
fn the_blur_between_the_halves_blurs_the_hud_and_leaves_the_screen_sharp() {
    let Some(mut gpu) = gpu() else { return };
    let mut font = font();
    let (_, hud_end) = scene(&mut font);
    let reference = render(&mut gpu, &mut font, None, None);
    let blurred = render(&mut gpu, &mut font, Some(hud_end), Some(&blur(8.0)));

    // The opaque panel belongs to the screen half: exactly its own colour, blur or not.
    for (x, y) in [(105, 65), (200, 130), (115, 75)] {
        assert_eq!(
            pixel(&blurred, x, y),
            pixel(&reference, x, y),
            "panel pixel ({x}, {y})"
        );
    }
    // The HUD stripes are under the gradient in both frames; blurred they flatten out.
    let sharp_contrast = contrast(&reference, 216, 40, 280);
    let blurred_contrast = contrast(&blurred, 216, 40, 280);
    assert!(
        sharp_contrast > 40,
        "the HUD stripes show: {sharp_contrast}"
    );
    assert!(
        blurred_contrast * 6 < sharp_contrast,
        "the HUD stripes are blurred: {blurred_contrast} vs {sharp_contrast}"
    );
    // So is the world above the HUD.
    let world_sharp = contrast(&reference, 20, 10, 90);
    let world_blurred = contrast(&blurred, 20, 10, 90);
    assert!(
        world_blurred * 6 < world_sharp,
        "{world_blurred} vs {world_sharp}"
    );
}

/// Not a check: writes what the vanilla pause menu looks like over a real in-game screenshot
/// (`mc-rs/screenshots/`, world plus HUD) with no blur and with the blur at a few radii, into
/// `featwave-2026-11-09/scratch/X10/`, for choosing the default radius by eye.
/// `cargo test -p mc-ui --test x10_ui_split -- --ignored preview`
#[test]
#[ignore = "writes preview PNGs for a human to look at"]
fn preview_the_pause_menu_over_a_real_screenshot() {
    use mc_assets::{Language, ResourceLocation};
    use mc_ui::screen_stack::ScreenCtx;
    use mc_ui::screens::pause::PauseScreen;
    use mc_ui::widget::{ScaledResolution, UiTexture};

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let shot = root.join("mc-rs/screenshots/screenshot_2026-09-13_22.28.08.png");
    let Ok(image) = image::open(&shot) else {
        eprintln!("SKIP: {} not found", shot.display());
        return;
    };
    let image = image.to_rgba8();
    let (w, h) = image.dimensions();
    let out = root.join("featwave-2026-11-09/scratch/X10");
    std::fs::create_dir_all(&out).unwrap();
    let Some(jar) = jars_dir().map(|d| d.join("versions/1.8.9/1.8.9.jar")) else {
        return;
    };
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).unwrap()));
    let lang = Language::load(&rm, &["en_US"]);
    let mut font = FontRenderer::new(FontData::load(&rm).unwrap(), false, false);
    let Ok(mut gpu) = Gpu::headless(w, h, GpuOptions::default()) else {
        return;
    };
    let ctx = ScreenCtx::new(ScaledResolution::new(w, h, 2, false), 0.0);
    let backdrop_key = UiTexture::Resource(ResourceLocation::minecraft("x10/backdrop.png"));

    for radius in [0.0f32, 8.0, 14.0, 20.0, 28.0] {
        // The "world and HUD": the screenshot, 1:1.
        let mut backdrop = UiPhase::new(&gpu, w, h, 1, false);
        backdrop
            .renderer
            .upload(&gpu, backdrop_key.clone(), image.as_raw(), w, h);
        let mut list = DrawList::new();
        list.blit_scaled(
            &backdrop_key,
            0,
            0,
            0.0,
            0.0,
            w as i32,
            h as i32,
            w as i32,
            h as i32,
            w as f32,
            h as f32,
        );
        backdrop.submit(list);
        // The pause menu, GUI scale 2, drawn after the blur.
        let mut menu = UiPhase::new(&gpu, w, h, 2, false);
        menu.renderer.upload_ascii_font(&gpu, font.data());
        let mut screen = PauseScreen::default();
        screen.init(&ctx, &lang);
        let mut list = DrawList::new();
        screen.draw(&mut list, &mut font, &ctx, (0, 0), &lang);
        let _ = menu.renderer.ensure_list_textures(&gpu, &rm, &list);
        menu.submit(list);

        let mut phases = PhaseList::new();
        phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(backdrop));
        if radius > 0.0 {
            let mut phase = MenuBlurPhase::new();
            phase.set_radius(radius);
            phase.set_active(true);
            phases.insert(
                PhaseOrder::MENU_BLUR,
                Box::new(SharedBlur(Rc::new(RefCell::new(phase)))),
            );
        }
        phases.insert(PhaseOrder::UI_SCREEN, Box::new(menu));
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
        let pixels = gpu.read_pixels().expect("readback");
        let path = out.join(format!("pause-radius-{radius:02}.png"));
        image::save_buffer(&path, &pixels, w, h, image::ColorType::Rgba8).unwrap();
        eprintln!("wrote {}", path.display());
    }
}

#[test]
fn a_split_at_zero_blurs_the_world_but_not_the_hud() {
    let Some(mut gpu) = gpu() else { return };
    let mut font = font();
    let reference = render(&mut gpu, &mut font, None, None);
    let blurred = render(&mut gpu, &mut font, Some(0), Some(&blur(8.0)));
    // The HUD stripes are opaque and draw after the blur, so they (and the gradient over them)
    // are pixel for pixel what the unblurred frame shows.
    for x in 40..280 {
        assert_eq!(
            pixel(&blurred, x, 216),
            pixel(&reference, x, 216),
            "HUD pixel ({x}, 216)"
        );
    }
    let world_sharp = contrast(&reference, 20, 10, 90);
    let world_blurred = contrast(&blurred, 20, 10, 90);
    assert!(
        world_blurred * 6 < world_sharp,
        "{world_blurred} vs {world_sharp}"
    );
}
