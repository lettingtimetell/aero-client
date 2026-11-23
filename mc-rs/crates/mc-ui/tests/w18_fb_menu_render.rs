//! Where the two new options sit in the mod menu, rendered for real
//! (wgpu, the vanilla font from the jar) at 1280x720: Nametags' settings with **Always show mob
//! names** on, **Confirm Disconnect** selected in Chat & Menus, and a search for "disconnect".
//! The
//! harness is `nw_mods_menu_render.rs`'s, trimmed.
//!
//! Each frame runs under a wgpu validation scope and must put near-white title text in the first
//! window. With `MC_W18_FB_PNG=<dir>` every frame is saved there (`menu-<state>.png`).
//!
//! Needs the vanilla jar (`MC_JARS_DIR`) and a GPU adapter; skips loudly without either.

use std::future::Future;
use std::time::{Duration, Instant};

use mc_mods::{DynamicRows, ModSettings};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::modhud::ModHud;
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::text::{MenuText, ATLAS_SIZE};
use mc_ui::screens::modmenu::{MenuEnv, MenuInput, ModMenuScreen, View};
use mc_ui::widget::{DrawList, UiPhase};

const SIZE: (u32, u32) = (1280, 720);

/// wgpu's native error-scope futures are ready at once; poll without an executor.
fn ready<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    for _ in 0..1000 {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        std::thread::yield_now();
    }
    panic!("the error scope never resolved");
}

fn jar_assets() -> Option<mc_assets::ResourceManager> {
    let jars = std::path::PathBuf::from(std::env::var_os("MC_JARS_DIR")?);
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = mc_assets::ZipResourcePack::open(&jar).ok()?;
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

/// A plain "world" behind the glass: sky bands over grass.
fn scene(w: u32, h: u32) -> DrawList {
    let mut list = DrawList::new();
    let band_h = (h / 12).max(1) as i32;
    for band in 0..12_i32 {
        let b = band as u32;
        let shade = 0xFF00_0000 | ((40 + b * 14) << 16) | ((90 + b * 10) << 8) | 230;
        list.rect(0, band * band_h, w as i32, band * band_h + band_h, shade);
    }
    list.rect(0, (h * 2 / 3) as i32, w as i32, h as i32, 0xFF3C_7A2E);
    list
}

struct Menu {
    settings: ModSettings,
    hud: ModHud,
    rows: Vec<DynamicRows>,
    events: Vec<mc_mods::ModEvent>,
    text: MenuText,
    font: FontRenderer,
    screen: ModMenuScreen,
    t0: Instant,
}

impl Menu {
    fn new(rm: &mc_assets::ResourceManager) -> Menu {
        let settings = ModSettings::new();
        let t0 = Instant::now();
        Menu {
            rows: vec![DynamicRows::default(); settings.catalog().len()],
            screen: ModMenuScreen::new(settings.catalog(), t0),
            settings,
            hud: ModHud::new(),
            events: Vec::new(),
            text: MenuText::load(),
            font: FontRenderer::load(rm, false, false).expect("vanilla font"),
            t0,
        }
    }

    fn at(&self, ms: u64) -> Instant {
        self.t0 + Duration::from_millis(ms)
    }

    fn gui_scale(&self) -> f32 {
        // Vanilla's automatic GUI scale: the largest that keeps 320x240.
        let (w, h) = SIZE;
        let mut scale = 1;
        while w / (scale + 1) >= 320 && h / (scale + 1) >= 240 {
            scale += 1;
        }
        scale as f32
    }

    fn build(&mut self, ms: u64) -> (DrawList, GlassFrame) {
        let mut list = DrawList::new();
        let now = self.at(ms);
        let gui_scale = self.gui_scale();
        let mut env = MenuEnv {
            text: &mut self.text,
            mc_font: &mut self.font,
            settings: &self.settings,
            rows: &self.rows,
            hud: &self.hud,
            screen: (SIZE.0 as f32, SIZE.1 as f32),
            gui_scale,
            now,
            time_s: 1.5,
            game: mc_ui::screens::modmenu::MenuGame::default(),
        };
        let frame = self.screen.build(&mut env, &mut list);
        (list, frame)
    }

    fn input(&mut self, ms: u64) -> MenuInput<'_> {
        let now = self.t0 + Duration::from_millis(ms);
        let scale = self.gui_scale();
        MenuInput {
            settings: &mut self.settings,
            hud: &mut self.hud,
            events: &mut self.events,
            gui_screen: (SIZE.0 as f32 / scale, SIZE.1 as f32 / scale),
            gui_scale: scale,
            now,
            alt: false,
            shift: false,
        }
    }

    fn type_text(&mut self, text: &str, ms: u64) {
        for c in text.chars() {
            self.key(30, Some(c), ms);
        }
    }

    fn key(&mut self, code: i32, typed: Option<char>, ms: u64) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        {
            let mut input = self.input(ms);
            screen.key(code, typed, &mut input);
        }
        self.screen = screen;
    }

    fn click(&mut self, (x, y): (f32, f32), ms: u64) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        {
            let mut input = self.input(ms);
            screen.mouse_pressed(x, y, 0, &mut input);
            screen.mouse_released(x, y, 0, &mut input);
        }
        self.screen = screen;
    }

    fn open_view(&mut self, view: View, ms: u64) {
        let nav = self
            .screen
            .nav_rect(view)
            .expect("the entry is in window 1");
        self.click(nav.center(), ms);
        self.build(ms);
    }

    /// Open a mod's settings as the user does: search its name, click the result, Esc, then its
    /// home category in window 1.
    fn open_mod(&mut self, mod_id: &str, ms: u64) {
        let m = self.settings.index_of(mod_id).expect("mod");
        let home = mc_mods::menu_layout::CATEGORIES
            .iter()
            .position(|category| {
                category
                    .sections
                    .iter()
                    .any(|section| section.mods.contains(&mod_id))
            })
            .and_then(|c| u8::try_from(c).ok())
            .expect("placed in a category");
        let name = self.settings.catalog()[m].name;
        self.type_text(name, ms);
        self.build(ms);
        let row = self.screen.row_rect(m).expect("a search result");
        self.click((row.x + row.w * 0.3, row.y + row.h * 0.5), ms);
        self.key(1, None, ms);
        self.build(ms);
        self.open_view(View::Category(home), ms);
        assert_eq!(self.screen.selected(), Some(m), "{mod_id} open");
    }

    /// Frames from `from` to `to` ms (animations and eased scrolling), returning the last.
    fn settle(&mut self, from: u64, to: u64) -> (DrawList, GlassFrame) {
        let mut last = self.build(from);
        for ms in (from + 16..=to).step_by(16) {
            last = self.build(ms);
        }
        last
    }
}

fn render(
    gpu: &mut Gpu,
    rm: &mc_assets::ResourceManager,
    menu: &mut Menu,
    list: DrawList,
    frame: GlassFrame,
) -> Vec<u8> {
    let (w, h) = SIZE;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ui = UiPhase::new(gpu, w, h, 1, false);
    ui.submit(scene(w, h));
    let mut phase = ModMenuPhase::new(gpu);
    menu.text.mark_dirty();
    if let Some(pixels) = menu.text.take_upload() {
        phase.upload_atlas(gpu, pixels, ATLAS_SIZE);
    }
    if !menu.text.is_segoe() {
        let _ = phase.renderer.ensure_list_textures(gpu, rm, &list);
    }
    phase.submit(list, frame);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(ui));
    phases.insert(PhaseOrder(PhaseOrder::UI.0 + 50), Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    if let Some(error) = ready(scope.pop()) {
        panic!("wgpu validation error drawing the menu: {error}");
    }
    gpu.read_pixels().expect("readback")
}

/// The first window must show near-white title text (the frame was really drawn); the frame is
/// saved when `MC_W18_FB_PNG` is set.
fn check_and_save(px: &[u8], frame: &GlassFrame, state: &str) {
    let (w, h) = SIZE;
    assert_eq!(frame.rects.len(), 4, "{state}: three windows and Edit HUD");
    let [x0, y0, _, _] = frame.rects[0].rect;
    let mut brightest = 0u8;
    for y in (y0 as u32 + 6)..(y0 as u32 + (h / 18).max(30)) {
        for x in (x0 as u32 + 6)..(x0 as u32 + (w / 16).max(80)) {
            let i = ((y * w + x) * 4) as usize;
            brightest = brightest.max(px[i].min(px[i + 1]).min(px[i + 2]));
        }
    }
    assert!(brightest > 200, "{state}: title text drawn ({brightest})");
    if let Some(dir) = std::env::var_os("MC_W18_FB_PNG") {
        let path = std::path::Path::new(&dir).join(format!("menu-{state}.png"));
        image::save_buffer(&path, px, w, h, image::ExtendedColorType::Rgba8)
            .expect("write the PNG");
        eprintln!("saved {}", path.display());
    }
}

#[test]
fn the_new_options_render_where_the_report_says() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: no vanilla jar (set MC_JARS_DIR)");
        return;
    };
    let mut gpu = match Gpu::headless(
        SIZE.0,
        SIZE.1,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            return;
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    };
    let mut menu = Menu::new(&rm);
    // Nametags on, with the new option on, so its row shows its "on" state.
    menu.settings.set_enabled("nametags", true);
    assert!(menu
        .settings
        .set_bool("nametags", mc_mods::mods::nametags::ALWAYS_MOB_NAMES, true));
    menu.settings
        .set_enabled(mc_mods::mods::smart_disconnect::ID, true);
    menu.screen = ModMenuScreen::new(menu.settings.catalog(), menu.t0);
    menu.settle(0, 500);

    menu.open_mod("nametags", 500);
    let (list, frame) = menu.settle(516, 1200);
    let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
    check_and_save(&px, &frame, "w18-fb-nametags");

    menu.open_mod(mc_mods::mods::smart_disconnect::ID, 1200);
    let (list, frame) = menu.settle(1216, 1900);
    let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
    check_and_save(&px, &frame, "w18-fb-smart-disconnect");

    // Search: "disconnect" finds Confirm Disconnect.
    menu.type_text("disconnect", 1900);
    let (list, frame) = menu.settle(1916, 2600);
    let m = menu
        .settings
        .index_of(mc_mods::mods::smart_disconnect::ID)
        .unwrap();
    assert!(
        menu.screen.row_rect(m).is_some(),
        "a search for disconnect lists Confirm Disconnect"
    );
    let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
    check_and_save(&px, &frame, "w18-fb-search-disconnect");
}
