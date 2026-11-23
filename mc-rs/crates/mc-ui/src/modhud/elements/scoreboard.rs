//! Scoreboard HUD element (`scoreboard`): the vanilla sidebar as a draggable element.
//!
//! While the mod is on, the client skips `Hud::draw_scoreboard` and hands the same cached
//! [`Sidebar`] to this element (`HudData::sidebar`, re-cloned only when the cache was rebuilt).
//! The paint is the vanilla HUD's own [`paint_sidebar`] with the mod's [`SidebarStyle`], so the
//! two share every pixel rule (and any fix to the vanilla draw).
//!
//! # Position: vanilla's pivot
//!
//! Vanilla does not centre the board vertically: its bottom is at `height / 2 + rows * 9 / 3`,
//! so the board grows up by 6 and down by 3 pixels per row. The 9-point anchor model cannot
//! express that, so for the three middle-row anchors this element overrides
//! [`HudElement::resolve`] and its inverse [`HudElement::placement_for`]: the offset is
//! measured from the screen's middle line to the board's pivot, `h - rows * 9 / 3` below its
//! top. The default placement (`CenterRight`, `x = -1`, `y = 0`) is therefore vanilla's box
//! exactly, right edge at `width - 1`, for any row count. Top and bottom anchors use the
//! ordinary model (the board grows away from the pinned edge).
//!
//! In the HUD editor with no sidebar, a five-row sample titled "Scoreboard" is drawn instead.

use mc_mods::{resolve_rect, HudAnchor, HudPlacement, HudRect, ModSettings};

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement};
use crate::screens::hud::{paint_sidebar, sidebar_text_width, Sidebar, SidebarStyle};
use crate::widget::DrawList;

/// The editor sample's title.
const SAMPLE_TITLE: &str = "Scoreboard";
/// The editor sample's row count.
const SAMPLE_ROWS: i32 = 5;

/// What a measured width depends on: sidebar revision, font layout generation, the numbers
/// option and whether it was the sample.
type MeasureKey = (u64, u64, bool, bool);

/// The `scoreboard` element.
#[derive(Debug)]
pub struct ScoreboardElement {
    hide: bool,
    style: SidebarStyle,
    /// The editor sample, built on first use.
    sample: Option<Sidebar>,
    /// This frame draws the sample, not the live sidebar.
    use_sample: bool,
    /// The inputs `width`, `rows` and `font_height` were measured from.
    measured: Option<MeasureKey>,
    /// [`sidebar_text_width`] of what is drawn.
    width: i32,
    /// Visible rows of what is drawn.
    rows: i32,
    /// The font's line height at the last measure.
    font_height: i32,
}

impl Default for ScoreboardElement {
    fn default() -> Self {
        ScoreboardElement {
            hide: false,
            style: SidebarStyle::VANILLA,
            sample: None,
            use_sample: false,
            measured: None,
            width: 0,
            rows: 0,
            font_height: 9,
        }
    }
}

/// A five-row sidebar for the editor, top row "Line 1".
fn sample_sidebar() -> Sidebar {
    let scores: Vec<(String, i32)> = (1..=SAMPLE_ROWS)
        .map(|points| (format!("Line {}", SAMPLE_ROWS + 1 - points), points))
        .collect();
    Sidebar::from_scores(SAMPLE_TITLE, &scores, str::to_owned)
}

/// The anchors whose vertical position goes through vanilla's pivot.
fn is_middle_row(anchor: HudAnchor) -> bool {
    matches!(
        anchor,
        HudAnchor::CenterLeft | HudAnchor::Center | HudAnchor::CenterRight
    )
}

impl ScoreboardElement {
    /// How far below the board's top the screen's middle line falls in vanilla's layout, for an
    /// unscaled board `height` pixels tall: `height - rows * font_height / 3`, with vanilla's
    /// integer division (`rows * font_height` is `height - font_height - 1`).
    fn pivot(&self, height: f32) -> f32 {
        let font_height = self.font_height.max(1);
        let height = height.round() as i32;
        let content = (height - font_height - 1).max(0);
        (height - content / 3) as f32
    }

    /// This frame's unscaled size.
    fn size(&self) -> (f32, f32) {
        (
            (self.width + 4) as f32,
            ((self.rows + 1) * self.font_height + 1) as f32,
        )
    }
}

impl HudElement for ScoreboardElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("scoreboard", "scoreboard")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.hide = settings.bool("scoreboard", "hide");
        self.style = SidebarStyle {
            numbers: !settings.bool("scoreboard", "hide_numbers"),
            background: settings.f32("scoreboard", "background_opacity") / 100.0,
            shadow: settings.bool("scoreboard", "text_shadow"),
        };
        self.measured = None;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        if self.hide {
            return None;
        }
        let live = ctx
            .data
            .sidebar
            .as_ref()
            .filter(|sidebar| !sidebar.visible_rows().is_empty());
        let use_sample = live.is_none();
        if use_sample && !ctx.editing {
            return None;
        }
        self.use_sample = use_sample;
        let key = (
            ctx.data.sidebar_revision,
            font.layout_generation(),
            self.style.numbers,
            use_sample,
        );
        if self.measured != Some(key) {
            if use_sample && self.sample.is_none() {
                self.sample = Some(sample_sidebar());
            }
            let sidebar = match (live, self.sample.as_ref()) {
                (Some(live), _) => live,
                (None, Some(sample)) => sample,
                (None, None) => return None,
            };
            self.width = sidebar_text_width(font, sidebar, self.style.numbers);
            self.rows = sidebar.visible_rows().len() as i32;
            self.font_height = font.font_height();
            self.measured = Some(key);
        }
        Some(self.size())
    }

    fn draw(&mut self, ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        mc_core::profile_scope!("hud.mod.scoreboard");
        let sidebar = if self.use_sample {
            self.sample.as_ref()
        } else {
            ctx.data.sidebar.as_ref()
        };
        let Some(sidebar) = sidebar else {
            return;
        };
        // Local origin = the board's top-left plate corner: vanilla's text column is 2 pixels in,
        // and its baseline is the board's bottom edge.
        let height = (self.rows + 1) * self.font_height + 1;
        paint_sidebar(d, font, sidebar, 2, height, self.width, self.style);
    }

    /// Vanilla draws the sidebar with F3 open, so this element does too.
    fn draws_under_debug(&self) -> bool {
        true
    }

    fn resolve(&self, placement: &HudPlacement, size: (f32, f32), screen: (f32, f32)) -> HudRect {
        let mut rect = resolve_rect(placement, size, screen);
        let placement = placement.sanitized();
        if is_middle_row(placement.anchor) {
            let y = (screen.1 / 2.0).floor() + placement.y - self.pivot(size.1) * placement.scale;
            rect.y = y.clamp(0.0, (screen.1 - rect.h).max(0.0));
        }
        rect
    }

    fn placement_for(&self, rect: &HudRect, scale: f32, screen: (f32, f32)) -> HudPlacement {
        let placement = mc_mods::placement_from_rect(rect, scale, screen);
        if !is_middle_row(placement.anchor) {
            return placement;
        }
        let scale = placement.scale;
        let pivot = self.pivot(rect.h / scale) * scale;
        HudPlacement {
            y: rect.y + pivot - (screen.1 / 2.0).floor(),
            ..placement
        }
        .sanitized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::HudData;
    use crate::widget::test_support::uniform_font;

    fn ctx<'a>(
        data: &'a HudData,
        settings: &'a ModSettings,
        screen: (f32, f32),
        editing: bool,
    ) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen,
            pixel_scale: 2.0,
            editing,
        }
    }

    /// `rows` rows named "Row <points>", ascending, like `getSortedScores`.
    pub(crate) fn sidebar(rows: i32) -> Sidebar {
        let scores: Vec<(String, i32)> = (1..=rows).map(|p| (format!("Row {p}"), p)).collect();
        Sidebar::from_scores("Title", &scores, str::to_owned)
    }

    fn data_with(sidebar: Option<Sidebar>, revision: u64) -> HudData {
        HudData {
            sidebar,
            sidebar_revision: revision,
            ..HudData::default()
        }
    }

    #[test]
    fn nothing_without_a_sidebar_and_a_sample_in_the_editor() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ScoreboardElement::default();
        element.sync(&settings);
        let none = data_with(None, 0);
        assert_eq!(
            element.layout(&ctx(&none, &settings, (640.0, 360.0), false), &font),
            None
        );
        let size = element
            .layout(&ctx(&none, &settings, (640.0, 360.0), true), &font)
            .expect("sample");
        // "Scoreboard" is the widest line: 10 glyphs of 5 pixels.
        assert_eq!(size, (54.0, 55.0));
        let mut d = DrawList::new();
        let mut font = font;
        element.draw(
            &ctx(&none, &settings, (640.0, 360.0), true),
            &mut font,
            &mut d,
        );
        assert!(!d.is_empty());
        // An empty sidebar is no sidebar.
        let empty = data_with(Some(sidebar(0)), 1);
        assert_eq!(
            element.layout(&ctx(&empty, &settings, (640.0, 360.0), false), &font),
            None
        );
    }

    #[test]
    fn hide_draws_nothing() {
        let mut settings = ModSettings::new();
        settings.set_bool("scoreboard", "hide", true);
        let font = uniform_font();
        let mut element = ScoreboardElement::default();
        element.sync(&settings);
        let data = data_with(Some(sidebar(3)), 1);
        assert_eq!(
            element.layout(&ctx(&data, &settings, (640.0, 360.0), true), &font),
            None
        );
    }

    #[test]
    fn the_width_is_measured_only_when_its_inputs_change() {
        let mut settings = ModSettings::new();
        settings.set_bool("scoreboard", "hide_numbers", false);
        let font = uniform_font();
        let mut element = ScoreboardElement::default();
        element.sync(&settings);
        let data = data_with(Some(sidebar(3)), 7);
        let with_numbers = element
            .layout(&ctx(&data, &settings, (640.0, 360.0), false), &font)
            .unwrap();
        // "Row 3: §c3": 6 glyphs of 5 pixels, two spaces of 4, and a zero-width `§c`: 38 + 4.
        assert_eq!(with_numbers, (42.0, 37.0));
        assert_eq!(
            element.measured,
            Some((7, font.layout_generation(), true, false))
        );
        settings.set_bool("scoreboard", "hide_numbers", true);
        element.sync(&settings);
        let without = element
            .layout(&ctx(&data, &settings, (640.0, 360.0), false), &font)
            .unwrap();
        // Names alone: the title "Title" (25) is wider than "Row 3" (24).
        assert_eq!(without, (29.0, 37.0));
    }

    #[test]
    fn the_default_placement_is_vanillas_box_for_any_row_count_and_screen() {
        let mut settings = ModSettings::new();
        settings.set_bool("scoreboard", "hide_numbers", false);
        let font = uniform_font();
        let default = mc_mods::catalog::all()
            .iter()
            .find(|d| d.id == "scoreboard")
            .unwrap()
            .hud_elements[0]
            .default_placement;
        for rows in [1, 2, 5, 14, 15] {
            for screen in [
                (640.0, 360.0),
                (427.0, 240.0),
                (683.0, 384.0),
                (480.0, 271.0),
            ] {
                let mut element = ScoreboardElement::default();
                element.sync(&settings);
                let data = data_with(Some(sidebar(rows)), 1);
                let size = element
                    .layout(&ctx(&data, &settings, screen, false), &font)
                    .unwrap();
                let rect = element.resolve(&default, size, screen);
                // Vanilla: text column at W - width - 3, plates from 2 left of it to W - 1,
                // bottom at H / 2 + rows * 9 / 3, header plate top (rows + 1) * 9 + 1 above.
                let (w, h) = (screen.0 as i32, screen.1 as i32);
                let text_width = sidebar_text_width(&font, &sidebar(rows), true);
                let baseline = h / 2 + rows * 9 / 3;
                assert_eq!(
                    rect.x as i32,
                    w - text_width - 3 - 2,
                    "{rows} rows on {screen:?}"
                );
                assert_eq!(rect.right() as i32, w - 1);
                assert_eq!(rect.bottom() as i32, baseline, "{rows} rows on {screen:?}");
                assert_eq!(rect.y as i32, baseline - (rows + 1) * 9 - 1);
            }
        }
    }

    #[test]
    fn hiding_the_numbers_keeps_vanillas_right_edge_and_rows() {
        let font = uniform_font();
        let default = mc_mods::catalog::all()
            .iter()
            .find(|d| d.id == "scoreboard")
            .unwrap()
            .hud_elements[0]
            .default_placement;
        let screen = (427.0, 240.0);
        let data = data_with(Some(sidebar(5)), 1);
        let mut numbers = ModSettings::new();
        numbers.set_bool("scoreboard", "hide_numbers", false);
        let mut with = ScoreboardElement::default();
        with.sync(&numbers);
        let size = with
            .layout(&ctx(&data, &numbers, screen, false), &font)
            .unwrap();
        let vanilla_box = with.resolve(&default, size, screen);
        // The mod's own defaults hide the numbers.
        let defaults = ModSettings::new();
        let mut without = ScoreboardElement::default();
        without.sync(&defaults);
        let size = without
            .layout(&ctx(&data, &defaults, screen, false), &font)
            .unwrap();
        let narrow = without.resolve(&default, size, screen);
        assert!(narrow.w < vanilla_box.w);
        assert_eq!(narrow.right(), vanilla_box.right());
        assert_eq!((narrow.y, narrow.h), (vanilla_box.y, vanilla_box.h));
    }

    #[test]
    fn placement_for_inverts_resolve_on_every_anchor_and_scale() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ScoreboardElement::default();
        element.sync(&settings);
        let data = data_with(Some(sidebar(6)), 1);
        let screen = (640.0, 361.0);
        let size = element
            .layout(&ctx(&data, &settings, screen, false), &font)
            .unwrap();
        for scale in [1.0, 0.5, 1.5, 2.0] {
            let w = size.0 * scale;
            let h = size.1 * scale;
            for (x, y) in [
                (4.0, 3.0),
                (300.0, 2.0),
                (600.0 - w, 5.0),
                (2.0, 150.0),
                (310.0, 160.0),
                (638.0 - w, 140.0),
                (3.0, 359.0 - h),
                (300.0, 358.0 - h),
                (637.0 - w, 357.0 - h),
            ] {
                let rect = HudRect::new(x, y, w, h);
                let placement = element.placement_for(&rect, scale, screen);
                let back = element.resolve(&placement, size, screen);
                assert!(
                    (back.x - rect.x).abs() < 1e-3 && (back.y - rect.y).abs() < 1e-3,
                    "{rect:?} at {scale} came back as {back:?} via {placement:?}"
                );
            }
        }
    }

    fn paint(element: &mut ScoreboardElement, settings: &ModSettings, data: &HudData) -> DrawList {
        let mut font = uniform_font();
        let screen = (640.0, 360.0);
        element.sync(settings);
        element.layout(&ctx(data, settings, screen, false), &font);
        let mut d = DrawList::new();
        element.draw(&ctx(data, settings, screen, false), &mut font, &mut d);
        d
    }

    /// The alpha of every vertex in batches that are (`solid`) or are not plates.
    fn alphas(d: &DrawList, solid: bool) -> Vec<f32> {
        d.batches()
            .iter()
            .filter(|b| (b.texture == crate::widget::UiTexture::Solid) == solid)
            .flat_map(|b| d.vertices()[b.range.clone()].iter().map(|v| v.color[3]))
            .collect()
    }

    #[test]
    fn options_reach_the_paint() {
        let mut settings = ModSettings::new();
        let data = data_with(Some(sidebar(3)), 1);
        let mut element = ScoreboardElement::default();
        let plain = paint(&mut element, &settings, &data);
        assert!(
            alphas(&plain, true).iter().all(|&a| a > 0.3),
            "vanilla plates"
        );
        settings.set_f32("scoreboard", "background_opacity", 0.0);
        let bare = paint(&mut element, &settings, &data);
        assert!(alphas(&bare, true).is_empty(), "0% draws no plate at all");
        settings.set_f32("scoreboard", "background_opacity", 50.0);
        let half = paint(&mut element, &settings, &data);
        let expected = (0x50 as f32 * 0.5).round() / 255.0;
        assert!(alphas(&half, true)
            .iter()
            .any(|&a| (a - expected).abs() < 1e-6));
        settings.set_bool("scoreboard", "text_shadow", true);
        let shadowed = paint(&mut element, &settings, &data);
        assert!(
            shadowed.vertices().len() > half.vertices().len(),
            "shadow pass"
        );
        assert!(
            alphas(&shadowed, false).iter().all(|&a| a == 1.0),
            "with the shadow on, the text and its shadow are opaque"
        );
    }
}

/// GPU readbacks for both M2 elements: at its default placement each one renders the
/// same pixels as the vanilla draw it replaces, on odd and even scaled widths and heights. Real
/// `wgpu` on a headless target (skipped without an adapter), over a coloured background so the
/// black plates show.
#[cfg(test)]
mod readback_tests {
    use super::*;
    use crate::modhud::elements::boss_bar::{BossBarElement, HudBoss};
    use crate::modhud::{HudData, ModHud};
    use crate::screen_stack::ScreenCtx;
    use crate::screens::hud::Hud;
    use crate::widget::test_support::uniform_font;
    use crate::widget::{ScaledResolution, UiPhase, UiTexture};
    use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

    /// Physical size and GUI scale: scaled 427x240, 428x241 and 641x361.
    const TARGETS: [(u32, u32, i32); 3] = [(854, 480, 2), (856, 482, 2), (641, 361, 1)];

    /// A 256x256 icons sheet whose boss-bar rows are opaque and change colour along x, so a
    /// one-pixel shift or a wrong fill width changes the picture.
    fn icons_sheet() -> Vec<u8> {
        let mut rgba = vec![0u8; 256 * 256 * 4];
        for y in 74..84usize {
            for x in 0..256usize {
                let i = (y * 256 + x) * 4;
                rgba[i] = x as u8;
                rgba[i + 1] = if y >= 79 { 220 } else { 40 };
                rgba[i + 2] = (y * 3) as u8;
                rgba[i + 3] = 255;
            }
        }
        rgba
    }

    struct Target {
        gpu: Gpu,
        width: u32,
        height: u32,
        gui_scale: i32,
    }

    fn headless_target(width: u32, height: u32, gui_scale: i32) -> Option<Target> {
        let options = GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        };
        match Gpu::headless(width, height, options) {
            Ok(gpu) => Some(Target {
                gpu,
                width,
                height,
                gui_scale,
            }),
            Err(RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter available on this machine");
                None
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        }
    }

    impl Target {
        fn ctx(&self) -> ScreenCtx {
            let resolution = ScaledResolution::new(self.width, self.height, self.gui_scale, false);
            ScreenCtx::new(resolution, 0.0)
        }

        /// Render `list` through a fresh UI phase (test font and icons uploaded) and read back.
        fn render(&mut self, list: DrawList) -> Vec<u8> {
            let font = uniform_font();
            let mut phase = UiPhase::new(&self.gpu, self.width, self.height, self.gui_scale, false);
            phase.renderer.upload_ascii_font(&self.gpu, font.data());
            phase.renderer.upload(
                &self.gpu,
                UiTexture::icons().clone(),
                &icons_sheet(),
                256,
                256,
            );
            phase.submit(list);
            let mut phases = PhaseList::new();
            phases.clear_color = wgpu::Color {
                r: 0.35,
                g: 0.55,
                b: 0.75,
                a: 1.0,
            };
            phases.insert(PhaseOrder::UI, Box::new(phase));
            let frame = self.gpu.acquire().expect("acquire offscreen frame");
            phases.render(&self.gpu, frame.view(), &Camera::default(), 0.0);
            self.gpu.read_pixels().expect("readback")
        }

        /// Assert two readbacks are identical, naming the first differing pixel.
        fn assert_same(&self, actual: &[u8], expected: &[u8], what: &str) {
            assert_eq!(actual.len(), expected.len());
            if let Some(i) = (0..actual.len() / 4)
                .find(|&i| actual[i * 4..i * 4 + 4] != expected[i * 4..i * 4 + 4])
            {
                let (x, y) = (i as u32 % self.width, i as u32 / self.width);
                let differing = (0..actual.len() / 4)
                    .filter(|&i| actual[i * 4..i * 4 + 4] != expected[i * 4..i * 4 + 4])
                    .count();
                panic!(
                    "{what}: {differing} pixels differ, first at ({x}, {y}): {:?} vs vanilla {:?}",
                    &actual[i * 4..i * 4 + 4],
                    &expected[i * 4..i * 4 + 4],
                );
            }
        }
    }

    /// One element through the real pipeline (`ModHud`), at the placement in `settings`.
    fn element_list(
        settings: &ModSettings,
        data: &HudData,
        ctx: &ScreenCtx,
        element: Box<dyn HudElement>,
    ) -> (DrawList, Option<HudRect>) {
        let mut hud = ModHud::with_elements(mc_mods::catalog::all(), vec![element]);
        hud.sync(settings);
        let hud_ctx = HudCtx {
            data,
            settings,
            screen: (ctx.width as f32, ctx.height as f32),
            pixel_scale: ctx.resolution.scale_factor() as f32,
            editing: false,
        };
        let mut list = DrawList::new();
        hud.draw(&hud_ctx, &mut uniform_font(), &mut list);
        (list, hud.slots()[0].rect)
    }

    /// A minigame-style board: formatted names, ascending points, a bold coloured title.
    fn board(rows: i32) -> Sidebar {
        let scores: Vec<(String, i32)> = (1..=rows)
            .map(|p| (format!("\u{a7}a{p} \u{a7}fPlayer{}", p * 7), p))
            .collect();
        Sidebar::from_scores("\u{a7}e\u{a7}lSKYWARS", &scores, str::to_owned)
    }

    #[test]
    fn the_boss_bar_element_at_its_default_placement_renders_vanillas_pixels() {
        let long = "W".repeat(45);
        for (w, h, scale) in TARGETS {
            let Some(mut target) = headless_target(w, h, scale) else {
                return;
            };
            let ctx = target.ctx();
            for (name, health) in [
                ("Wither", 0.37),
                ("\u{a7}dEnder Dragon \u{a7}7| \u{a7}eEvent 5:00", 1.0),
                (long.as_str(), 0.0),
            ] {
                let mut hud = Hud::new();
                hud.set_boss_status(health * 300.0, 300.0, name, false);
                let mut vanilla = DrawList::new();
                hud.draw_boss_health(&mut vanilla, &mut uniform_font(), &ctx);
                let mut settings = ModSettings::new();
                settings.set_enabled("boss_bar", true);
                let data = HudData {
                    boss: HudBoss {
                        shown: true,
                        name: name.to_owned(),
                        health_scale: hud.boss_health_scale(),
                    },
                    ..HudData::default()
                };
                let (element, rect) =
                    element_list(&settings, &data, &ctx, Box::<BossBarElement>::default());
                assert!(rect.is_some(), "drawn");
                let expected = target.render(vanilla);
                let blank = target.render(DrawList::new());
                assert_ne!(expected, blank, "the vanilla bar drew something");
                let actual = target.render(element);
                target.assert_same(&actual, &expected, &format!("{name:?} on {w}x{h}@{scale}"));
            }
        }
    }

    #[test]
    fn the_scoreboard_element_at_its_default_placement_renders_vanillas_pixels() {
        for (w, h, scale) in TARGETS {
            let Some(mut target) = headless_target(w, h, scale) else {
                return;
            };
            let ctx = target.ctx();
            for rows in [1, 5, 15] {
                let sidebar = board(rows);
                let mut vanilla = DrawList::new();
                Hud::new().draw_scoreboard(&mut vanilla, &mut uniform_font(), &ctx, &sidebar);
                // The mod's defaults minus "hide red numbers" are vanilla's look.
                let mut settings = ModSettings::new();
                settings.set_enabled("scoreboard", true);
                settings.set_bool("scoreboard", "hide_numbers", false);
                let data = HudData {
                    sidebar: Some(sidebar),
                    sidebar_revision: 1,
                    ..HudData::default()
                };
                let (element, rect) =
                    element_list(&settings, &data, &ctx, Box::<ScoreboardElement>::default());
                assert!(rect.is_some(), "drawn");
                let expected = target.render(vanilla);
                let actual = target.render(element);
                target.assert_same(
                    &actual,
                    &expected,
                    &format!("{rows} rows on {w}x{h}@{scale}"),
                );
            }
        }
    }

    /// Negative control for the two readbacks above: one GUI pixel off the default placement is
    /// a visible difference, so "identical" there means placed exactly.
    #[test]
    fn a_one_pixel_offset_from_the_default_is_detected() {
        let Some(mut target) = headless_target(854, 480, 2) else {
            return;
        };
        let ctx = target.ctx();

        let mut hud = Hud::new();
        hud.set_boss_status(150.0, 300.0, "Wither", false);
        let mut vanilla = DrawList::new();
        hud.draw_boss_health(&mut vanilla, &mut uniform_font(), &ctx);
        let mut settings = ModSettings::new();
        settings.set_enabled("boss_bar", true);
        settings.set_placement(
            "boss_bar",
            "boss_bar",
            HudPlacement::new(HudAnchor::TopCenter, 0.0, 3.0),
        );
        let data = HudData {
            boss: HudBoss {
                shown: true,
                name: "Wither".to_owned(),
                health_scale: hud.boss_health_scale(),
            },
            ..HudData::default()
        };
        let (moved, _) = element_list(&settings, &data, &ctx, Box::<BossBarElement>::default());
        assert_ne!(
            target.render(moved),
            target.render(vanilla),
            "boss bar 1 px low"
        );

        let sidebar = board(5);
        let mut vanilla = DrawList::new();
        Hud::new().draw_scoreboard(&mut vanilla, &mut uniform_font(), &ctx, &sidebar);
        let mut settings = ModSettings::new();
        settings.set_enabled("scoreboard", true);
        settings.set_bool("scoreboard", "hide_numbers", false);
        settings.set_placement(
            "scoreboard",
            "scoreboard",
            HudPlacement::new(HudAnchor::CenterRight, -2.0, 0.0),
        );
        let data = HudData {
            sidebar: Some(sidebar),
            sidebar_revision: 1,
            ..HudData::default()
        };
        let (moved, _) = element_list(&settings, &data, &ctx, Box::<ScoreboardElement>::default());
        assert_ne!(
            target.render(moved),
            target.render(vanilla),
            "sidebar 1 px left"
        );
    }

    #[test]
    fn a_moved_and_scaled_scoreboard_draws_only_inside_its_rect() {
        let Some(mut target) = headless_target(854, 480, 2) else {
            return;
        };
        let ctx = target.ctx();
        let mut settings = ModSettings::new();
        settings.set_enabled("scoreboard", true);
        settings.set_placement(
            "scoreboard",
            "scoreboard",
            HudPlacement::new(HudAnchor::TopLeft, 4.0, 6.0).scaled(1.5),
        );
        let data = HudData {
            sidebar: Some(board(5)),
            sidebar_revision: 1,
            ..HudData::default()
        };
        let (list, rect) =
            element_list(&settings, &data, &ctx, Box::<ScoreboardElement>::default());
        let rect = rect.expect("drawn");
        assert_eq!((rect.x, rect.y), (4.0, 6.0));
        let blank = target.render(DrawList::new());
        let pixels = target.render(list);
        let device = |v: f32| (v * 2.0).ceil() as u32;
        let (x0, y0) = (device(rect.x), device(rect.y));
        let (x1, y1) = (device(rect.right()), device(rect.bottom()));
        let mut changed = 0;
        for y in 0..480u32 {
            for x in 0..854u32 {
                let i = ((y * 854 + x) * 4) as usize;
                if pixels[i..i + 4] != blank[i..i + 4] {
                    changed += 1;
                    assert!(
                        x >= x0 && x < x1 && y >= y0 && y < y1,
                        "({x}, {y}) drawn outside {rect:?}"
                    );
                }
            }
        }
        assert!(changed > 0, "the board drew something");
    }
}
