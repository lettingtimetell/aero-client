//! Bug report: "symbols in the chat dont render with a background shadow like
//! text".
//!
//! A symbol is any glyph `ascii.png` does not have (hearts, stars, swords, check marks, arrows):
//! it is drawn from a unicode page. With the unicode font off, vanilla 1.8.9 draws its shadow
//! exactly under it (`FontRenderer.renderStringAtPos`, FontRenderer.java:479-494), where the
//! glyph covers it, so no shadow shows. `FontRenderer::set_symbol_shadows` keeps it one pixel
//! down and right, like a letter's. Through the real font renderer, the real chat box and the
//! real `UI 800` phase on a headless wgpu device, read back:
//!
//! * [`only_symbol_shadows_change`]: the chat at 1920x1080, GUI scales 1-4. A chat of letters
//!   only is byte-identical either way. With symbols, every pixel that changes lies in a
//!   symbol's shadow footprint, some do, and batches and pipeline binds are equal.
//! * [`gallery`]: the chat box (and a Tab list with symbols in its names) at GUI 1-4 both ways,
//!   saved as PNGs when `MC_W18_FA_PNG_DIR=<dir>` is set.
//! * [`the_chat_settings_window_shows_symbol_shadows`]: the mod menu's Chat settings (Chat off,
//!   as by default), saved as `menu-chat-<w>x<h>.png` in the same folder.
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent (`MC_JARS_DIR`).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_assets::{Language, ResourceManager, ZipResourcePack};
use mc_mods::{DynamicRows, ModSettings};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::component::ChatComponent;
use mc_ui::font::{FontRenderer, FontTexture, GlyphQuad, SolidRect, TextSink};
use mc_ui::modhud::ModHud;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::chat::{ChatLayout, ChatLog};
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::text::{MenuText, ATLAS_SIZE};
use mc_ui::screens::modmenu::{MenuEnv, MenuInput, ModMenuScreen, View};
use mc_ui::screens::tab_list::{TabListOverlay, TabRow};
use mc_ui::widget::{DrawList, ScaledResolution, Transform, UiPhase, UiTexture};

/// The user's window.
const SIZE: (u32, u32) = (1920, 1080);

/// Chat lines with the symbols servers put in chat, newest last. The last one has letters only.
const SYMBOL_LINES: [&str; 8] = [
    "\u{a7}7[\u{a7}6100\u{a7}e\u{272b}\u{a7}7] \u{a7}b[MVP\u{a7}c+\u{a7}b] Steve\u{a7}f: gg",
    "\u{a7}c\u{2764} \u{a7}fHealth \u{a7}c20\u{2764}  \u{a7}a\u{2714} \u{a7}aQuest complete!",
    "\u{a7}6\u{2694} \u{a7}eYou killed \u{a7}bAlex\u{a7}e! \u{a7}6+10 coins \u{a7}d\u{272a}",
    "\u{a7}e\u{279c} \u{a7}7Sending you to \u{a7}amini3B \u{a7}8\u{25b6} \u{a7}f\u{2726} \u{2620} \
     \u{273f} \u{2756} \u{26a1}",
    "\u{a7}a\u{a7}l\u{25ac}\u{25ac}\u{25ac}\u{25ac}\u{25ac}\u{25ac}\u{25ac}\u{25ac}\u{25ac}\
     \u{25ac}\u{25ac}\u{25ac}\u{25ac}\u{25ac}",
    "\u{a7}f\u{2192} \u{2190} \u{2191} \u{2193} \u{2713} \u{2717} \u{2714} \u{2716} \u{2665} \
     \u{2605} \u{2606} \u{2666} \u{2663} \u{2660} \u{266a} \u{263a} \u{2022}",
    "\u{a7}c\u{a7}l\u{2764} FINAL KILL! \u{a7}r\u{a7}7Steve was killed by Alex. \u{a7}b\u{2714}",
    "\u{a7}eThe quick brown fox jumps over the lazy dog. gjpqy_",
];

/// The same chat with letters only.
const LETTER_LINES: [&str; 4] = [
    "\u{a7}7[\u{a7}6100\u{a7}7] \u{a7}b[MVP\u{a7}c+\u{a7}b] Steve\u{a7}f: gg",
    "\u{a7}c\u{a7}lFINAL KILL! \u{a7}r\u{a7}7Steve was killed by Alex.",
    "\u{a7}nUnderlined\u{a7}r \u{a7}mstruck\u{a7}r \u{a7}oitalic \u{a7}l\u{a7}obold italic",
    "\u{a7}eThe quick brown fox jumps over the lazy dog. gjpqy_",
];

/// Symbol shadows off (vanilla) and on.
const MODES: [bool; 2] = [false, true];

fn mode_name(on: bool) -> &'static str {
    if on {
        "symbol-shadows"
    } else {
        "vanilla"
    }
}

fn gpu(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
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

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

fn png_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MC_W18_FA_PNG_DIR")?);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Render `list` at `resolution` on black: the frame, the pipeline binds and the batches.
fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    resolution: ScaledResolution,
    gui_option: i32,
    list: DrawList,
) -> (Vec<u8>, u64, usize) {
    let (w, h) = resolution.display_size();
    let (w, h) = (w as u32, h as u32);
    gpu.resize(w, h);
    let mut phase = UiPhase::new(gpu, w, h, gui_option, false);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    let binds = phase.renderer.pipeline_binds(&list);
    let batches = list.batches().len();
    phase.set_resolution(resolution);
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    assert_eq!(pixels.len(), (w * h * 4) as usize, "readback size");
    (pixels, binds, batches)
}

/// A dim, varied backdrop so shadows read as they do over a world (sky bands, then grass).
fn backdrop(d: &mut DrawList, w: i32, h: i32) {
    let bands = 12;
    for band in 0..bands {
        let b = band as u32;
        let shade = 0xFF00_0000u32 | ((30 + b * 9) << 16) | ((60 + b * 8) << 8) | (90 + b * 6);
        d.rect(0, band * h / bands, w, (band + 1) * h / bands, shade);
    }
    d.rect(0, h * 2 / 3, w, h, 0xFF2E_5A24);
}

fn chat_log(font: &FontRenderer, layout: &ChatLayout, lang: &Language, lines: &[&str]) -> ChatLog {
    let mut log = ChatLog::new();
    for text in lines {
        log.add_message(
            ChatComponent::text(*text),
            0,
            1000,
            layout,
            font,
            lang,
            false,
        );
    }
    log
}

/// The closed chat box over the backdrop, as the HUD draws it.
fn chat_scene(
    font: &mut FontRenderer,
    ctx: &ScreenCtx,
    lang: &Language,
    lines: &[&str],
) -> DrawList {
    let mut d = DrawList::new();
    backdrop(&mut d, ctx.width, ctx.height);
    let layout = ChatLayout::from_settings(&mc_render::GameSettings::default());
    let log = chat_log(font, &layout, lang, lines);
    log.draw(&mut d, font, ctx, 1010, &layout, false);
    d
}

/// Collects the glyphs of the shadow pass.
#[derive(Default)]
struct ShadowGlyphs {
    shadow: bool,
    quads: Vec<GlyphQuad>,
}

impl TextSink for ShadowGlyphs {
    fn shadow_pass(&mut self, shadow: bool) {
        self.shadow = shadow;
    }
    fn glyph(&mut self, quad: GlyphQuad) {
        if self.shadow {
            self.quads.push(quad);
        }
    }
    fn bar(&mut self, _: SolidRect) {}
}

/// Every symbol's shadow quad in GUI space, as the chat places its rows: row `n` (0 = newest,
/// at the bottom) at `translate(2, height - 28)` and `y = -9 n - 8`.
fn symbol_shadow_quads(font: &mut FontRenderer, ctx: &ScreenCtx, lines: &[&str]) -> Vec<GlyphQuad> {
    let mut out = Vec::new();
    for (row, text) in lines.iter().rev().enumerate() {
        let mut sink = ShadowGlyphs::default();
        let y = -(row as i32) * 9 - 8;
        font.draw_string(&mut sink, text, 0.0, y as f32, -1, true);
        let t = Transform::translate(2.0, (ctx.height - 28) as f32);
        out.extend(
            sink.quads
                .into_iter()
                .filter(|q| matches!(q.texture, FontTexture::UnicodePage(_)))
                .map(|q| GlyphQuad {
                    pos: q.pos.map(|p| t.apply(p)),
                    ..q
                }),
        );
    }
    out
}

/// Physical pixel boxes `[x0, y0, x1, y1)` around `quads` (GUI space), one pixel of slack.
fn footprints(quads: &[GlyphQuad], factor: f32) -> Vec<[i64; 4]> {
    quads
        .iter()
        .map(|q| {
            let xs = q.pos.map(|p| p[0] * factor);
            let ys = q.pos.map(|p| p[1] * factor);
            let lo = |v: [f32; 4]| v.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = |v: [f32; 4]| v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            [
                lo(xs).floor() as i64 - 1,
                lo(ys).floor() as i64 - 1,
                hi(xs).ceil() as i64 + 1,
                hi(ys).ceil() as i64 + 1,
            ]
        })
        .collect()
}

#[test]
fn only_symbol_shadows_change() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let Some(mut gpu) = gpu(SIZE.0, SIZE.1) else {
        return;
    };
    let lang = Language::load(&rm, &["en_US"]);
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let (w, h) = SIZE;
    let mut report = Vec::new();
    for gui in 1..=4 {
        let res = ScaledResolution::new(w, h, gui, false);
        assert_eq!(res.scale_factor(), gui, "1920x1080 allows GUI {gui}");
        let ctx = ScreenCtx::new(res, 0.0);
        let factor = gui as f32;

        // Letters only: byte-identical either way, with identical batching.
        let mut letters = Vec::new();
        for on in MODES {
            font.set_symbol_shadows(on);
            let list = chat_scene(&mut font, &ctx, &lang, &LETTER_LINES);
            letters.push(render(&mut gpu, &rm, res, gui, list));
        }
        assert_eq!(
            letters[1].1, letters[0].1,
            "gui{gui}: letters, pipeline binds"
        );
        assert_eq!(letters[1].2, letters[0].2, "gui{gui}: letters, batches");
        assert!(letters[1].0 == letters[0].0, "gui{gui}: a letter changed");

        // With symbols: every changed pixel lies in a symbol's shadow footprint (where vanilla
        // drew it, under the symbol, or where it is drawn now), some pixels change, and the
        // batching is unchanged.
        font.set_symbol_shadows(false);
        let list = chat_scene(&mut font, &ctx, &lang, &SYMBOL_LINES);
        let (vanilla, binds, batches) = render(&mut gpu, &rm, res, gui, list);
        let vanilla_quads = symbol_shadow_quads(&mut font, &ctx, &SYMBOL_LINES);
        assert!(vanilla_quads.len() > 40, "the chat has symbols");
        font.set_symbol_shadows(true);
        let list = chat_scene(&mut font, &ctx, &lang, &SYMBOL_LINES);
        let (frame, on_binds, on_batches) = render(&mut gpu, &rm, res, gui, list);
        assert_eq!(on_binds, binds, "gui{gui}: pipeline binds");
        assert_eq!(on_batches, batches, "gui{gui}: batches");
        let mut boxes = footprints(&vanilla_quads, factor);
        boxes.extend(footprints(
            &symbol_shadow_quads(&mut font, &ctx, &SYMBOL_LINES),
            factor,
        ));
        let mut changed = 0usize;
        let mut outside = Vec::new();
        for (i, (a, b)) in vanilla
            .chunks_exact(4)
            .zip(frame.chunks_exact(4))
            .enumerate()
        {
            if a == b {
                continue;
            }
            changed += 1;
            let (x, y) = ((i as u32 % w) as i64, (i as u32 / w) as i64);
            let inside = boxes
                .iter()
                .any(|b| x >= b[0] && x < b[2] && y >= b[1] && y < b[3]);
            if !inside && outside.len() < 5 {
                outside.push((x, y));
            }
        }
        let line = format!(
            "gui{gui}: {changed} pixels changed, all in symbol shadows: {}; {batches} batches, \
             {binds} pipeline binds either way",
            outside.is_empty()
        );
        assert!(outside.is_empty(), "{line}; outside at {outside:?}");
        assert!(changed > 100, "{line}: the shadows did not appear");
        report.push(line);
    }
    font.set_symbol_shadows(false);
    eprintln!("{}", report.join("\n"));
}

/// The `FontRenderer.java` parity row's alpha test, through the real chat box. java:
/// `GuiNewChat.drawChat` draws a line's backdrop (`drawRect`, alpha `l1 / 2`) and then its text
/// with `drawStringWithShadow`, which enables the alpha test (GREATER 0.1): once `l1` is 25 or
/// less, in the last ~0.3 s of the line's 10 s, the text is gone while the backdrop keeps fading
/// until `l1` reaches 3.
#[test]
fn a_fading_chat_lines_text_goes_at_a_tenth_of_its_alpha() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let lang = Language::load(&rm, &["en_US"]);
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let layout = ChatLayout::from_settings(&mc_render::GameSettings::default());
    let ctx = ScreenCtx::new(ScaledResolution::new(1920, 1080, 2, false), 0.0);
    let log = chat_log(&font, &layout, &lang, &["\u{a7}eHello \u{2764}"]);
    let mut seen = Vec::new();
    for age in 186..200 {
        let mut d = DrawList::new();
        log.draw(&mut d, &mut font, &ctx, 1000 + age, &layout, false);
        let count = |font_texture: bool| -> usize {
            d.batches()
                .iter()
                .filter(|b| matches!(b.texture, UiTexture::Font(_)) == font_texture)
                .map(|b| b.range.len())
                .sum()
        };
        // GuiNewChat's alpha, with the default text opacity (1.0 * 0.9 + 0.1).
        let fade = ((1.0 - f64::from(age) / 200.0) * 10.0).clamp(0.0, 1.0);
        let alpha = (255.0 * fade * fade) as i32;
        assert_eq!(
            count(true) > 0,
            alpha > 25,
            "age {age}: alpha {alpha}, text"
        );
        assert_eq!(
            count(false) > 0,
            alpha > 3,
            "age {age}: alpha {alpha}, backdrop"
        );
        seen.push((alpha, count(true) > 0));
    }
    assert!(
        seen.iter().any(|&(a, text)| a > 3 && a <= 25 && !text),
        "the tail was reached"
    );
}

/// The chat box (and a Tab list with symbols in its names) both ways at GUI 1-4.
/// `chat-gui<n>-<vanilla|symbol-shadows>.png` is the chat box region, `tab-gui<n>-...` the Tab
/// list.
#[test]
fn gallery() {
    let Some(dir) = png_dir() else {
        eprintln!("SKIP: set MC_W18_FA_PNG_DIR to save the pictures");
        return;
    };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let Some(mut gpu) = gpu(SIZE.0, SIZE.1) else {
        return;
    };
    let lang = Language::load(&rm, &["en_US"]);
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let (w, h) = SIZE;
    for gui in 1..=4u32 {
        let res = ScaledResolution::new(w, h, gui as i32, false);
        let ctx = ScreenCtx::new(res, 0.0);
        for mode in MODES {
            font.set_symbol_shadows(mode);
            // The chat box: 330 GUI px wide, its rows and two pixels of margin.
            let list = chat_scene(&mut font, &ctx, &lang, &SYMBOL_LINES);
            let (frame, _, _) = render(&mut gpu, &rm, res, gui as i32, list);
            let rows = SYMBOL_LINES.len() as u32;
            let top = (ctx.height as u32 - 28 - 9 * rows - 2) * gui;
            let bottom = (ctx.height as u32 - 26) * gui;
            save_crop(
                &dir.join(format!("chat-gui{gui}-{}.png", mode_name(mode))),
                &frame,
                w,
                [0, top, 330 * gui, bottom],
            );

            let list = tab_scene(&mut font, &ctx);
            let (frame, _, _) = render(&mut gpu, &rm, res, gui as i32, list);
            let half = ctx.width as u32 / 2;
            save_crop(
                &dir.join(format!("tab-gui{gui}-{}.png", mode_name(mode))),
                &frame,
                w,
                [(half - 150) * gui, 0, (half + 150) * gui, 110 * gui],
            );
        }
    }
    font.set_symbol_shadows(false);
}

fn tab_scene(font: &mut FontRenderer, ctx: &ScreenCtx) -> DrawList {
    let mut d = DrawList::new();
    backdrop(&mut d, ctx.width, ctx.height);
    let names = [
        "\u{a7}6[MVP\u{a7}c++\u{a7}6] Steve \u{a7}e\u{272b}",
        "\u{a7}b[MVP\u{a7}c+\u{a7}b] Alex \u{a7}c\u{2764}",
        "\u{a7}a[VIP] Herobrine \u{a7}a\u{2714}",
        "\u{a7}7Player \u{a7}d\u{272a}\u{a7}7 \u{2694}",
    ];
    let mut rows: Vec<TabRow> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let mut row = TabRow::new(format!("Player{i}"), 40 + 60 * i as i32);
            row.display = (*name).to_owned();
            row
        })
        .collect();
    mc_ui::screens::tab_list::sort_rows(&mut rows);
    let mut tab = TabListOverlay::new();
    tab.update(true, 0);
    tab.draw(
        &mut d,
        font,
        ctx.width,
        &rows,
        Some("\u{a7}b\u{2726} \u{a7}eYou are playing on \u{a7}e\u{a7}lA SERVER \u{a7}b\u{2726}"),
        Some("\u{a7}a\u{2764} Ranks, Boosters & MORE! \u{a7}c\u{2764}"),
        None,
        false,
    );
    d
}

/// Save `rect = [x0, y0, x1, y1)` of an RGBA `frame` that is `width` pixels wide.
fn save_crop(path: &Path, frame: &[u8], width: u32, rect: [u32; 4]) {
    let [x0, y0, x1, y1] = rect;
    let mut out = Vec::with_capacity(((x1 - x0) * (y1 - y0) * 4) as usize);
    for y in y0..y1 {
        let start = ((y * width + x0) * 4) as usize;
        let end = ((y * width + x1) * 4) as usize;
        out.extend_from_slice(&frame[start..end]);
    }
    image::save_buffer(path, &out, x1 - x0, y1 - y0, image::ColorType::Rgba8)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

// ---------------------------------------------------------------------------------------------
// The option in the mod menu. The harness is `nw_mods_menu_render.rs`'s,
// trimmed to opening one mod's settings.

/// The mod menu over a plain backdrop, driven like a player drives it.
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
}

impl Menu {
    fn new(rm: &ResourceManager, size: (u32, u32)) -> Menu {
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
            size,
            t0,
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
        let gui_scale = self.gui_scale();
        let mut env = MenuEnv {
            text: &mut self.text,
            mc_font: &mut self.font,
            settings: &self.settings,
            rows: &self.rows,
            hud: &self.hud,
            screen: (self.size.0 as f32, self.size.1 as f32),
            gui_scale,
            now: self.t0 + Duration::from_millis(ms),
            time_s: 1.5,
            game: mc_ui::screens::modmenu::MenuGame::default(),
        };
        let frame = self.screen.build(&mut env, &mut list);
        (list, frame)
    }

    fn with_input(&mut self, ms: u64, act: impl FnOnce(&mut ModMenuScreen, &mut MenuInput<'_>)) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
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
        act(&mut screen, &mut input);
        self.screen = screen;
    }

    fn click(&mut self, (x, y): (f32, f32), ms: u64) {
        self.with_input(ms, |screen, input| {
            screen.mouse_pressed(x, y, 0, input);
            screen.mouse_released(x, y, 0, input);
        });
    }

    /// Search the mod's name, click the result, Esc, then its home category in window 1.
    fn open_mod(&mut self, mod_id: &str, ms: u64) {
        let m = self.settings.index_of(mod_id).expect("mod");
        let home = mc_mods::menu_layout::CATEGORIES
            .iter()
            .position(|c| c.sections.iter().any(|s| s.mods.contains(&mod_id)))
            .and_then(|c| u8::try_from(c).ok())
            .expect("placed in a category");
        let name = self.settings.catalog()[m].name;
        for c in name.chars() {
            self.with_input(ms, |screen, input| {
                screen.key(30, Some(c), input);
            });
        }
        self.build(ms);
        let row = self.screen.row_rect(m).expect("a search result");
        self.click((row.x + row.w * 0.3, row.y + row.h * 0.5), ms);
        self.with_input(ms, |screen, input| {
            screen.key(1, None, input);
        });
        self.build(ms);
        let nav = self
            .screen
            .nav_rect(View::Category(home))
            .expect("window 1");
        self.click(nav.center(), ms);
        self.build(ms);
        assert_eq!(self.screen.selected(), Some(m), "{mod_id} open");
    }

    fn settle(&mut self, from: u64, to: u64) -> (DrawList, GlassFrame) {
        let mut last = self.build(from);
        for ms in (from + 16..=to).step_by(16) {
            last = self.build(ms);
        }
        last
    }

    fn render(&mut self, gpu: &mut Gpu, rm: &ResourceManager, list: DrawList, frame: GlassFrame) {
        let (w, h) = self.size;
        let mut ui = UiPhase::new(gpu, w, h, 1, false);
        let mut backdrop_list = DrawList::new();
        backdrop(&mut backdrop_list, w as i32, h as i32);
        ui.submit(backdrop_list);
        let mut phase = ModMenuPhase::new(gpu);
        self.text.mark_dirty();
        if let Some(pixels) = self.text.take_upload() {
            phase.upload_atlas(gpu, pixels, ATLAS_SIZE);
        }
        if !self.text.is_segoe() {
            let _ = phase.renderer.ensure_list_textures(gpu, rm, &list);
        }
        phase.submit(list, frame);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::UI, Box::new(ui));
        phases.insert(PhaseOrder(PhaseOrder::UI.0 + 50), Box::new(phase));
        let target = gpu.acquire().expect("acquire offscreen frame");
        phases.render(gpu, target.view(), &Camera::default(), 0.0);
    }
}

/// Mods > Chat with the Chat mod off (its default): "Symbol shadows" sits under its own ALL TEXT
/// heading. Saved as `menu-chat-<w>x<h>.png` when `MC_W18_FA_PNG_DIR` is set.
#[test]
fn the_chat_settings_window_shows_symbol_shadows() {
    let Some(dir) = png_dir() else {
        eprintln!("SKIP: set MC_W18_FA_PNG_DIR to save the pictures");
        return;
    };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let size = (1920, 1080);
    let Some(mut gpu) = gpu(size.0, size.1) else {
        return;
    };
    let mut menu = Menu::new(&rm, size);
    assert!(!menu.settings.enabled("chat"), "Chat is off by default");
    menu.settle(0, 500);
    menu.open_mod("chat", 500);
    let (list, frame) = menu.settle(516, 1200);
    menu.render(&mut gpu, &rm, list, frame);
    let px = gpu.read_pixels().expect("readback");
    let path = dir.join(format!("menu-chat-{}x{}.png", size.0, size.1));
    image::save_buffer(&path, &px, size.0, size.1, image::ColorType::Rgba8)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}
