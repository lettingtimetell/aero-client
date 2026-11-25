//! The mod menu rendered for real (wgpu, Segoe UI, the vanilla
//! font from the jar) at 1280x720, 1920x1080 and 2560x1440, in three states: just opened,
//! searching "zoom", and Color Grading open with two saved profiles. The menu is three windows now, and a second test renders the plan's new states
//! (`ma-*` pictures: My setup, a category, settings, a search for "color", Keybinds, ...).
//!
//! Each frame runs under a wgpu validation scope and must put near-white title text inside the
//! first window. With `MC_X1_PNG_DIR=<dir>` every frame is saved there
//! (`menu-<state>-<width>x<height>.png`) for a human look at the layout.
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
    let jars = match std::env::var("MC_JARS_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)?
            .join("MCP-919-main")
            .join("jars"),
    };
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = mc_assets::ZipResourcePack::open(&jar).ok()?;
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

/// A colourful "world" behind the glass: sky bands, terrain, bright posts.
fn scene(w: u32, h: u32) -> DrawList {
    let mut list = DrawList::new();
    let band_h = (h / 12).max(1) as i32;
    for band in 0..12_i32 {
        let b = band as u32;
        let shade = 0xFF00_0000 | ((40 + b * 14) << 16) | ((90 + b * 10) << 8) | 230;
        list.rect(0, band * band_h, w as i32, band * band_h + band_h, shade);
    }
    list.rect(0, (h * 2 / 3) as i32, w as i32, h as i32, 0xFF3C_7A2E);
    let step = (w / 16) as i32;
    for post in 0..16 {
        let x = post * step + step / 4;
        list.rect(
            x,
            (h * 5 / 12) as i32,
            x + step / 6,
            (h * 5 / 6) as i32,
            0xFFFF_E080,
        );
    }
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
    size: (u32, u32),
    t0: Instant,
    /// No world (the title screen): Edit HUD greys out.
    no_world: bool,
}

impl Menu {
    fn new(rm: &mc_assets::ResourceManager, size: (u32, u32)) -> Menu {
        let mut settings = ModSettings::new();
        for id in ["fps", "keystrokes", "toggle_sprint", "zoom", "crosshair"] {
            settings.set_enabled(id, true);
        }
        let t0 = Instant::now();
        Menu {
            rows: vec![DynamicRows::default(); settings.catalog().len()],
            screen: ModMenuScreen::new(settings.catalog(), t0),
            settings,
            hud: ModHud::new(),
            events: Vec::new(),
            text: MenuText::load(),
            font: FontRenderer::load(rm, false, false).expect("vanilla font"),
            size,
            t0,
            no_world: false,
        }
    }

    fn at(&self, ms: u64) -> Instant {
        self.t0 + Duration::from_millis(ms)
    }

    fn gui_scale(&self) -> f32 {
        // Vanilla's automatic GUI scale: the largest that keeps 320x240.
        let (w, h) = self.size;
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
            screen: (self.size.0 as f32, self.size.1 as f32),
            gui_scale,
            now,
            time_s: 1.5,
            game: mc_ui::screens::modmenu::MenuGame {
                no_world: self.no_world,
                ..mc_ui::screens::modmenu::MenuGame::default()
            },
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
            gui_screen: (self.size.0 as f32 / scale, self.size.1 as f32 / scale),
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

    /// One key press (`typed`: the character it makes, if any).
    fn key(&mut self, code: i32, typed: Option<char>, ms: u64) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        {
            let mut input = self.input(ms);
            screen.key(code, typed, &mut input);
        }
        self.screen = screen;
    }

    /// A click (`button` 0 left, 1 right) at `(x, y)`.
    fn click(&mut self, (x, y): (f32, f32), button: u8, ms: u64) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        {
            let mut input = self.input(ms);
            screen.mouse_pressed(x, y, button, &mut input);
            screen.mouse_released(x, y, button, &mut input);
        }
        self.screen = screen;
    }

    /// Open window 1's entry `view` (a frame must have been built).
    fn open_view(&mut self, view: View, ms: u64) {
        let nav = self
            .screen
            .nav_rect(view)
            .expect("the entry is in window 1");
        self.click(nav.center(), 0, ms);
        self.build(ms);
    }

    /// Open a mod's settings as the user does (three windows), even when
    /// its row is below the fold: search its name, click the result, Esc, then its home category
    /// in window 1 (window 2 scrolls to the selected row).
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
        self.click((row.x + row.w * 0.3, row.y + row.h * 0.5), 0, ms);
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
    let (w, h) = menu.size;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ui = UiPhase::new(gpu, w, h, 1, false);
    ui.submit(scene(w, h));
    let mut phase = ModMenuPhase::new(gpu);
    // A new phase starts without the glyph atlas: upload it whole, as the client does for a
    // rebuilt phase (`menu_atlas_stale`), not only when new glyphs appeared.
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

/// `window`: a window whose title must be near white (a search dims the windows without a
/// match, so the "search" state checks the window that holds Zoom).
fn check_and_save(px: &[u8], size: (u32, u32), frame: &GlassFrame, state: &str, window: usize) {
    let (w, h) = size;
    assert_eq!(
        frame.rects.len(),
        4,
        "{state}: three windows and the taskbar's Edit HUD"
    );
    // Near-white title text inside that window's title bar.
    let [x0, y0, _, _] = frame.rects[window].rect;
    let mut brightest = 0u8;
    for y in (y0 as u32 + 6)..(y0 as u32 + (h / 18).max(30)) {
        for x in (x0 as u32 + 6)..(x0 as u32 + (w / 16).max(80)) {
            let i = ((y * w + x) * 4) as usize;
            brightest = brightest.max(px[i].min(px[i + 1]).min(px[i + 2]));
        }
    }
    assert!(
        brightest > 200,
        "{state} {w}x{h}: title text drawn ({brightest})"
    );
    if let Some(dir) = std::env::var_os("MC_X1_PNG_DIR") {
        let path = std::path::Path::new(&dir).join(format!("menu-{state}-{w}x{h}.png"));
        image::save_buffer(&path, px, w, h, image::ExtendedColorType::Rgba8)
            .expect("write the PNG");
        eprintln!("saved {}", path.display());
    }
}

#[test]
fn the_menu_renders_opened_searching_and_grading_at_720p_1080p_and_1440p() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    for size in [(1280, 720), (1920, 1080), (2560, 1440)] {
        let mut gpu = match Gpu::headless(
            size.0,
            size.1,
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

        // Just opened.
        let mut menu = Menu::new(&rm, size);
        let (list, frame) = menu.settle(0, 1000);
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "open", 0);

        // Searching "zoom": Zoom with its matching options inline, the rest hidden or dimmed.
        menu.type_text("zoom", 1000);
        let (list, frame) = menu.settle(1016, 1500);
        assert_eq!(menu.screen.search_query(), "zoom");
        let zoom = menu.settings.index_of("zoom").unwrap();
        assert!(menu.screen.row_rect(zoom).is_some(), "Zoom is listed");
        assert!(
            menu.screen
                .row_rect(menu.settings.index_of("fps").unwrap())
                .is_none(),
            "FPS is hidden"
        );
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        // Window 2 lists the results (window 1 is always titled).
        check_and_save(&px, size, &frame, "search", 1);

        // Color Grading open, a grade on, two profiles saved (the window scrolls to show it).
        let mut menu = Menu::new(&rm, size);
        menu.settings.set_enabled("color_grading", true);
        let grading = menu.settings.index_of("color_grading").unwrap();
        menu.settings.set_f32("color_grading", "saturation", 1.25);
        menu.settings.save_profile_at(grading, "Warm sunset");
        menu.settings.set_f32("color_grading", "master_slope", 1.1);
        menu.settings.save_profile_at(grading, "Crisp");
        menu.settle(0, 1000);
        menu.open_mod("color_grading", 1000);
        let (list, frame) = menu.settle(1016, 1600);
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "grading", 0);
    }
}

/// The three-window menu in the states the plan's pictures ask
/// for, each at 1280x720, 1920x1080 and 2560x1440: My setup (the
/// first open), a category with a mod open, settings with headings and a rule, a search for
/// "color" and Keybinds; plus at 1080p Particles' tabs, Chat Macros' paired rows and the menu
/// without a world. Saved as `ma-<state>-<w>x<h>.png` under `MC_X1_PNG_DIR`.
///
/// `MC_MA_SETTINGS=<file>` lays the menu out over a settings file (a copy of the user's
/// `mc-rs-mods.json`); otherwise over a small sample setup (a few mods on, one tuned, one
/// pinned).
#[test]
fn the_three_window_menu_renders_its_states_at_720p_1080p_and_1440p() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    let settings_file = std::env::var_os("MC_MA_SETTINGS");
    for size in [(1280, 720), (1920, 1080), (2560, 1440)] {
        let mut gpu = match Gpu::headless(
            size.0,
            size.1,
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
        let mut menu = Menu::new(&rm, size);
        match &settings_file {
            Some(path) => {
                let (settings, report) = ModSettings::load(std::path::Path::new(path));
                assert!(!report.corrupt, "{report:?}");
                menu.settings = settings;
            }
            None => {
                menu.settings.set_f32("hitboxes", "width", 3.0);
                let fullbright = menu.settings.index_of("fullbright").unwrap();
                menu.settings.set_pinned_at(fullbright, true);
            }
        }
        menu.screen = ModMenuScreen::new(menu.settings.catalog(), menu.t0);

        // My setup: the first open of a session with something changed.
        let (list, frame) = menu.settle(0, 1000);
        assert_eq!(menu.screen.view(), Some(View::MySetup));
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "ma-mysetup", 0);

        // A category with a mod open: HUD, FPS (its STYLE block).
        menu.open_mod("fps", 1000);
        let (list, frame) = menu.settle(1016, 1500);
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "ma-category", 0);

        // Settings with headings and a rule: Crosshair.
        menu.open_mod("crosshair", 1500);
        let (list, frame) = menu.settle(1516, 2000);
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "ma-settings", 0);

        // A search for "color": both spellings, mods first, then settings by mod.
        menu.type_text("color", 2000);
        let (list, frame) = menu.settle(2016, 2500);
        assert_eq!(menu.screen.search_query(), "color");
        let grading = menu.settings.index_of("color_grading").unwrap();
        assert!(
            menu.screen.row_rect(grading).is_some(),
            "Color Grading is a result"
        );
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "ma-search", 0);

        // Keybinds (Esc clears the search first).
        menu.key(1, None, 2500);
        menu.build(2500);
        menu.open_view(View::Keybinds, 2500);
        let (list, frame) = menu.settle(2516, 3000);
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "ma-keybinds", 0);

        if size != (1920, 1080) {
            continue;
        }
        // Particles: the picker as tabs over the two sliders it switches.
        menu.open_mod("particles", 3000);
        let (list, frame) = menu.settle(3016, 3500);
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "ma-particles", 0);

        // Chat Macros: a macro's key and text on one row.
        menu.open_mod("chat_macros", 3500);
        let (list, frame) = menu.settle(3516, 4000);
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "ma-macros", 0);

        // No world (the title screen): Edit HUD is greyed out.
        menu.no_world = true;
        let (list, frame) = menu.settle(4016, 4300);
        assert!(menu.screen.edit_hud_rect().is_none());
        let px = render(&mut gpu, &rm, &mut menu, list, frame.clone());
        check_and_save(&px, size, &frame, "ma-noworld", 0);
    }
}

// ---- The menu extras ------------------------------------------

/// The menu, with the client's facts plugged in: Minecraft's controls (the Keybinds clash
/// check), `options.txt` settings (live warnings) and a game folder (profiles).
struct McMenu {
    settings: ModSettings,
    hud: ModHud,
    rows: Vec<DynamicRows>,
    events: Vec<mc_mods::ModEvent>,
    text: MenuText,
    font: FontRenderer,
    screen: ModMenuScreen,
    size: (u32, u32),
    t0: Instant,
    vanilla: mc_render::input::KeyBindings,
    options: mc_render::GameSettings,
    game_dir: std::path::PathBuf,
}

impl McMenu {
    fn new(
        rm: &mc_assets::ResourceManager,
        size: (u32, u32),
        settings: ModSettings,
        game_dir: &std::path::Path,
    ) -> McMenu {
        let t0 = Instant::now();
        McMenu {
            rows: vec![DynamicRows::default(); settings.catalog().len()],
            screen: ModMenuScreen::new(settings.catalog(), t0),
            settings,
            hud: ModHud::new(),
            events: Vec::new(),
            text: MenuText::load(),
            font: FontRenderer::load(rm, false, false).expect("vanilla font"),
            size,
            t0,
            vanilla: mc_render::input::KeyBindings::vanilla(),
            options: mc_render::GameSettings::default(),
            game_dir: game_dir.to_path_buf(),
        }
    }

    fn gui_scale(&self) -> f32 {
        let (w, h) = self.size;
        let mut scale = 1;
        while w / (scale + 1) >= 320 && h / (scale + 1) >= 240 {
            scale += 1;
        }
        scale as f32
    }

    fn build(&mut self, ms: u64) -> (DrawList, GlassFrame) {
        let mut list = DrawList::new();
        let now = self.t0 + Duration::from_millis(ms);
        let gui_scale = self.gui_scale();
        self.hud.sync(&self.settings);
        let mut env = MenuEnv {
            text: &mut self.text,
            mc_font: &mut self.font,
            settings: &self.settings,
            rows: &self.rows,
            hud: &self.hud,
            screen: (self.size.0 as f32, self.size.1 as f32),
            gui_scale,
            now,
            time_s: 1.5,
            game: mc_ui::screens::modmenu::MenuGame {
                no_world: false,
                vanilla_keys: Some(&self.vanilla),
                options: Some(&self.options),
                game_dir: Some(&self.game_dir),
            },
        };
        let frame = self.screen.build(&mut env, &mut list);
        (list, frame)
    }

    fn with_input(&mut self, ms: u64, f: impl FnOnce(&mut ModMenuScreen, &mut MenuInput<'_>)) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        {
            let scale = self.gui_scale();
            let mut input = MenuInput {
                settings: &mut self.settings,
                hud: &mut self.hud,
                events: &mut self.events,
                gui_screen: (self.size.0 as f32 / scale, self.size.1 as f32 / scale),
                gui_scale: scale,
                now: self.t0 + Duration::from_millis(ms),
                alt: false,
                shift: false,
            };
            f(&mut screen, &mut input);
        }
        self.screen = screen;
    }

    fn click(&mut self, (x, y): (f32, f32), ms: u64) {
        self.with_input(ms, |screen, input| {
            screen.mouse_pressed(x, y, 0, input);
            screen.mouse_released(x, y, 0, input);
        });
    }

    fn hover(&mut self, (x, y): (f32, f32), ms: u64) {
        self.with_input(ms, |screen, input| screen.pointer_moved(x, y, input));
    }

    fn key(&mut self, code: i32, typed: Option<char>, ms: u64) {
        self.with_input(ms, |screen, input| {
            screen.key(code, typed, input);
        });
    }

    fn open_view(&mut self, view: View, ms: u64) {
        let nav = self
            .screen
            .nav_rect(view)
            .expect("the entry is in window 1");
        self.click(nav.center(), ms);
        self.build(ms);
    }

    /// As `Menu::open_mod`: search its name, click the result, Esc, then its home category.
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
        for c in name.chars() {
            self.key(30, Some(c), ms);
        }
        self.build(ms);
        let row = self.screen.row_rect(m).expect("a search result");
        self.click((row.x + row.w * 0.3, row.y + row.h * 0.5), ms);
        self.key(1, None, ms);
        self.build(ms);
        self.open_view(View::Category(home), ms);
        assert_eq!(self.screen.selected(), Some(m), "{mod_id} open");
    }

    fn settle(&mut self, from: u64, to: u64) -> (DrawList, GlassFrame) {
        let mut last = self.build(from);
        for ms in (from + 16..=to).step_by(16) {
            last = self.build(ms);
        }
        last
    }
}

/// `render` for an [`McMenu`].
fn render_mc(
    gpu: &mut Gpu,
    rm: &mc_assets::ResourceManager,
    menu: &mut McMenu,
    list: DrawList,
    frame: GlassFrame,
) -> Vec<u8> {
    let (w, h) = menu.size;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ui = UiPhase::new(gpu, w, h, 1, false);
    ui.submit(scene(w, h));
    let mut phase = ModMenuPhase::new(gpu);
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

/// Save `px` as `mc-<state>-<w>x<h>.png` under `MC_X1_PNG_DIR`.
fn save_mc(px: &[u8], (w, h): (u32, u32), state: &str) {
    if let Some(dir) = std::env::var_os("MC_X1_PNG_DIR") {
        let path = std::path::Path::new(&dir).join(format!("mc-{state}-{w}x{h}.png"));
        image::save_buffer(&path, px, w, h, image::ExtendedColorType::Rgba8)
            .expect("write the PNG");
        eprintln!("saved {}", path.display());
    }
}

/// How many pixels in `rect` are amber (the clash and warning colour): proves a warning drew.
fn amber_pixels(px: &[u8], (w, _): (u32, u32), rect: [f32; 4]) -> usize {
    let [x0, y0, rw, rh] = rect;
    let mut count = 0;
    for y in (y0 as u32)..((y0 + rh) as u32) {
        for x in (x0 as u32)..((x0 + rw) as u32) {
            let i = ((y * w + x) * 4) as usize;
            let (r, g, b) = (px[i], px[i + 1], px[i + 2]);
            if r > 200 && g > 140 && g < 215 && b < 120 {
                count += 1;
            }
        }
    }
    count
}

/// The states, each saved as `mc-<state>-<w>x<h>.png` under `MC_X1_PNG_DIR`: Keybinds with
/// clashes (Zoom on Q, Minecraft's Drop Item; a chat macro on Markers' key while Chat Macros is
/// off), Profiles (two saved, one picked, what loading it changes), FPS on the shared HUD style
/// (with "Move on screen") and Toggle Sprint on its own style, a live warning (Anti-Aliasing
/// with MSAA on) and the HUD editor's tray; at 1280x720, 1920x1080 and 2560x1440 for the first
/// three. The pause menu and title screen buttons are in the next test.
#[test]
fn the_menu_extras_render_at_720p_1080p_and_1440p() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    // A game folder with two saved profiles (rebuilt each run).
    let game_dir = std::env::temp_dir().join(format!("mc-rs-mc-render-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&game_dir);
    std::fs::create_dir_all(&game_dir).expect("game folder");
    {
        let library = mc_mods::profiles::ProfileLibrary::open(&game_dir, mc_mods::catalog::all());
        let mut bedwars = ModSettings::new();
        for id in [
            "fps",
            "cps",
            "keystrokes",
            "toggle_sprint",
            "zoom",
            "hitboxes",
        ] {
            bedwars.set_enabled(id, true);
        }
        bedwars.set_f32("hitboxes", "width", 2.5);
        library.save("Bedwars", &bedwars, None);
        let mut skywars = ModSettings::new();
        for id in ["fps", "zoom", "old_animations", "fullbright"] {
            skywars.set_enabled(id, true);
        }
        library.save("SkyWars", &skywars, None);
        library.wait_idle();
    }
    // The current setup.
    let mut settings = ModSettings::new();
    for id in ["fps", "toggle_sprint", "zoom", "crosshair", "markers"] {
        settings.set_enabled(id, true);
    }
    settings.set_key("zoom", "key", 16);
    settings.set_key("chat_macros", "key1", 44);
    settings.set_text("chat_macros", "text1", "/play bedwars_four_four");
    for size in [(1280, 720), (1920, 1080), (2560, 1440)] {
        let mut gpu = match Gpu::headless(
            size.0,
            size.1,
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
        let mut menu = McMenu::new(&rm, size, settings.clone(), &game_dir);
        menu.settle(0, 900);

        // Keybinds with clashes.
        menu.open_view(View::Keybinds, 900);
        let (list, frame) = menu.settle(916, 1400);
        let px = render_mc(&mut gpu, &rm, &mut menu, list, frame.clone());
        assert_eq!(frame.rects.len(), 4);
        assert!(
            amber_pixels(&px, size, frame.rects[0].rect) > 10,
            "window 1's Keybinds entry warns"
        );
        save_mc(&px, size, "keybinds");

        // Profiles: Bedwars picked.
        menu.open_view(View::Profiles, 1400);
        for ms in (1416..1600).step_by(16) {
            menu.build(ms);
            std::thread::sleep(Duration::from_millis(2));
        }
        let row = menu.screen.profile_row_rect(0).expect("Bedwars is listed");
        menu.click(row.center(), 1600);
        menu.hover((5.0, 5.0), 1610);
        let (list, frame) = menu.settle(1616, 2000);
        assert!(
            menu.screen.profile_load_rect().is_some(),
            "Bedwars can be loaded"
        );
        let px = render_mc(&mut gpu, &rm, &mut menu, list, frame.clone());
        save_mc(&px, size, "profiles");

        // FPS on the HUD style, its summary unfolded; then Toggle Sprint on its own style.
        menu.open_mod("fps", 2000);
        menu.build(2010);
        let fold = menu
            .screen
            .hud_style_fold_rect()
            .expect("FPS uses the HUD style");
        menu.click(fold.center(), 2020);
        menu.hover((5.0, 5.0), 2030);
        let (list, frame) = menu.settle(2040, 2400);
        let px = render_mc(&mut gpu, &rm, &mut menu, list, frame.clone());
        save_mc(&px, size, "hudstyle");
        if size != (1920, 1080) {
            continue;
        }
        menu.open_mod("toggle_sprint", 2400);
        let (list, frame) = menu.settle(2416, 2800);
        let px = render_mc(&mut gpu, &rm, &mut menu, list, frame.clone());
        save_mc(&px, size, "hudstyle-custom");

        // A live warning: Anti-Aliasing while MSAA is on.
        menu.options.msaa_samples = 4;
        menu.open_mod("anti_aliasing", 2800);
        let (list, frame) = menu.settle(2816, 3200);
        let px = render_mc(&mut gpu, &rm, &mut menu, list, frame.clone());
        assert!(
            amber_pixels(&px, size, frame.rects[2].rect) > 50,
            "window 3 shows the warning"
        );
        save_mc(&px, size, "warning");

        // The HUD editor's tray.
        let edit = menu.screen.edit_hud_rect().expect("Edit HUD");
        menu.click(edit.center(), 3200);
        menu.settle(3216, 3700);
        let tray = menu.screen.tray_rect().expect("the tray's pill");
        menu.click(tray.center(), 3700);
        menu.hover((tray.x - 40.0, tray.y - 120.0), 3710);
        let (list, frame) = menu.settle(3716, 3900);
        assert!(menu.screen.hud_editing());
        let px = render_mc(&mut gpu, &rm, &mut menu, list, frame);
        save_mc(&px, size, "tray");
    }
    let _ = std::fs::remove_dir_all(&game_dir);
}

/// The Mods buttons: the pause menu, where "Open to LAN" was (hovered), and the
/// Aero title screen (hovered). Saved as `mc-pause-<w>x<h>.png` and `mc-title-<w>x<h>.png`.
#[test]
fn the_mods_buttons_render_on_the_pause_menu_and_the_title_screen() {
    use mc_ui::screen_stack::ScreenCtx;
    use mc_ui::screens::aero_title::font::TitleFont;
    use mc_ui::screens::aero_title::phase::AeroTitlePhase;
    use mc_ui::screens::aero_title::{AeroButton, AeroTitleEnv, AeroTitleScreen};
    use mc_ui::screens::pause::PauseScreen;
    use mc_ui::widget::ScaledResolution;

    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    let lang = mc_assets::Language::load(&rm, &["en_US"]);
    for size in [(1280, 720), (1920, 1080)] {
        let (w, h) = size;
        let mut gpu = match Gpu::headless(
            w,
            h,
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
        let mut font = FontRenderer::load(&rm, false, false).expect("vanilla font");

        // The pause menu over a world, the Mods button hovered.
        let mut scale = 1;
        while w / (scale + 1) >= 320 && h / (scale + 1) >= 240 {
            scale += 1;
        }
        let res = ScaledResolution::new(w, h, scale as i32, false);
        let ctx = ScreenCtx::new(res, 0.0);
        let mut pause = PauseScreen::new();
        pause.init(&ctx, &lang);
        let (gw, gh) = (ctx.width, ctx.height);
        let mods_at = (gw / 2 + 2 + 49, gh / 4 + 96 - 16 + 10);
        assert_eq!(
            pause.mouse_clicked(mods_at.0, mods_at.1, 0),
            mc_ui::screens::Action::OpenModMenu,
            "the Mods button is where Open to LAN was"
        );
        let mut list = scene(gw as u32, gh as u32);
        pause.draw(&mut list, &mut font, &ctx, mods_at, &lang);
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut ui = UiPhase::new(&gpu, w, h, scale as i32, false);
        let missing = ui.renderer.ensure_list_textures(&gpu, &rm, &list);
        assert!(missing.is_empty(), "the pause menu's textures: {missing:?}");
        ui.submit(list);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::UI, Box::new(ui));
        let target = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, target.view(), &Camera::default(), 0.0);
        if let Some(error) = ready(scope.pop()) {
            panic!("wgpu validation error drawing the pause menu: {error}");
        }
        let px = gpu.read_pixels().expect("readback");
        save_mc(&px, size, "pause");

        // The Aero title screen, the Mods button hovered.
        let mut text = MenuText::load();
        let mut title = TitleFont::embedded();
        let labels = AeroButton::ALL.map(|b| b.english().to_owned());
        let t0 = Instant::now();
        let mut screen = AeroTitleScreen::new(t0);
        let sz = (w as f32, h as f32);
        let mut list = DrawList::new();
        let mut glass = GlassFrame::default();
        for ms in [0u64, 400, 1200, 1500, 1800, 2200] {
            if ms == 1200 {
                let r = screen.button_rect(AeroButton::Mods);
                screen.pointer_moved(r.x + r.w * 0.6, r.y + r.h * 0.5);
            }
            list = DrawList::new();
            let mut env = AeroTitleEnv {
                text: &mut text,
                title_font: &mut title,
                mc_font: &mut font,
                labels: &labels,
                version: "Minecraft 1.8.9",
                game_version: mc_version::GameVersion::V1_8_9,
                copyright: "Copyright Mojang AB. Do not distribute!",
                screen: sz,
                now: t0 + Duration::from_millis(ms),
                time_s: ms as f32 / 1000.0,
            };
            glass = screen.build(&mut env, &mut list);
        }
        // One pane per button, the version slider's and the link pill's in the bottom-right
        // corner.
        assert_eq!(glass.rects.len(), AeroButton::COUNT + 2);
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut ui = UiPhase::new(&gpu, w, h, 1, false);
        let mut backdrop = DrawList::new();
        backdrop.gradient_rect(0, 0, w as i32, h as i32, 0xFF6F_B8EA, 0xFF2C_6E31);
        ui.submit(backdrop);
        let mut phase = AeroTitlePhase::new(&gpu);
        if let Some(pixels) = text.take_upload() {
            phase.upload_text_atlas(&gpu, pixels);
        }
        if let Some((pixels, tw, th)) = title.take_upload() {
            phase.upload_title(&gpu, pixels, tw, th);
        }
        if !text.is_segoe() {
            let _ = phase.renderer().ensure_list_textures(&gpu, &rm, &list);
        }
        phase.submit(list, glass);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::UI, Box::new(ui));
        phases.insert(PhaseOrder(PhaseOrder::UI.0 + 40), Box::new(phase));
        let target = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, target.view(), &Camera::default(), 0.0);
        if let Some(error) = ready(scope.pop()) {
            panic!("wgpu validation error drawing the title screen: {error}");
        }
        let px = gpu.read_pixels().expect("readback");
        save_mc(&px, size, "title");
    }
}
