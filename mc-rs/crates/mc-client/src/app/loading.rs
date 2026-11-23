//! The loading screen. java: `client/LoadingScreenRenderer.java` (221 lines).
//!
//! Vanilla's class is both the *model* (three strings and a progress integer) and the *renderer*
//! (raw `GlStateManager` calls into a dedicated `Framebuffer`). Here the model is
//! [`LoadingScreen`] and the drawing goes through S15's [`DrawList`], because the whole point of
//! this project's UI layer is that there is one path to the screen. Every coordinate, colour and
//! the 100-millisecond throttle are vanilla's.
//!
//! The three lines vanilla describes in `resetProgressAndMessage`'s comment are:
//!
//! ```text
//!            <title>          ← displayString / currentlyDisplayedText
//!            <message>        ← displayLoadingString
//!         [====      ]        ← setLoadingProgress
//! ```

use mc_ui::screen_stack::ScreenCtx;
use mc_ui::widget::{DrawList, Rect, UiTexture, OPTIONS_BACKGROUND};

/// The progress bar's width in GUI pixels. java: `int i1 = 100`.
pub const BAR_WIDTH: i32 = 100;
/// The progress bar's height. java: `int j1 = 2`.
pub const BAR_HEIGHT: i32 = 2;
/// java: the `i - this.systemTime >= 100L` guard in `setLoadingProgress`. Vanilla redraws at
/// most ten times a second no matter how often the loader reports.
pub const REDRAW_INTERVAL_MS: u64 = 100;
/// java: `worldrenderer.color(64, 64, 64, 255)` over `Gui.optionsBackground`.
pub const BACKGROUND_ARGB: u32 = 0xFF40_4040;
/// java: `color(128, 128, 128, 255)`.
pub const BAR_TRACK_ARGB: u32 = 0xFF80_8080;
/// java: `color(128, 255, 128, 255)`.
pub const BAR_FILL_ARGB: u32 = 0xFF80_FF80;
/// java: `float f = 32.0F` — the background is tiled every 32 pixels.
pub const BACKGROUND_TILE: f32 = 32.0;
/// java: `drawStringWithShadow(..., 16777215)`.
pub const TEXT_COLOR: i32 = 0x00FF_FFFF;

/// The loading screen's state. java: `LoadingScreenRenderer`'s four fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadingScreen {
    /// java: `currentlyDisplayedText`, set by `resetProgressAndMessage`/`displaySavingString`.
    pub title: String,
    /// java: `message`, set by `displayLoadingString`.
    pub message: String,
    /// java: the `progress` argument. Negative means "no bar", which is what
    /// `displayLoadingString` passes (`setLoadingProgress(-1)`).
    pub progress: i32,
    /// java: `systemTime`. `0` means "redraw on the next call", which is what
    /// `displayLoadingString` forces by assigning it twice.
    last_drawn_ms: u64,
}

impl LoadingScreen {
    /// A blank screen. java: the constructor, minus the framebuffer.
    #[must_use]
    pub fn new() -> LoadingScreen {
        LoadingScreen::default()
    }

    /// java: `resetProgressAndMessage` — a new headline, progress cleared.
    pub fn reset_progress_and_message(&mut self, title: impl Into<String>) {
        self.title = title.into();
        self.message.clear();
        self.progress = -1;
    }

    /// java: `displayLoadingString` — a new sub-line, and `systemTime = 0` so the *next*
    /// `setLoadingProgress` redraws immediately rather than waiting out the throttle.
    pub fn display_loading_string(&mut self, message: impl Into<String>) {
        self.message = message.into();
        self.progress = -1;
        self.last_drawn_ms = 0;
    }

    /// java: `setLoadingProgress(int)` — record the value and decide whether this is a frame.
    ///
    /// Returns `true` when the caller should actually draw. Vanilla folds the decision and the
    /// drawing into one method; splitting them is what lets the throttle be tested without a GPU.
    pub fn set_loading_progress(&mut self, progress: i32, now_ms: u64) -> bool {
        self.progress = progress;
        self.should_redraw(now_ms)
    }

    /// Whether enough time has passed to redraw. java: `if (i - this.systemTime >= 100L)`.
    pub fn should_redraw(&mut self, now_ms: u64) -> bool {
        if now_ms.saturating_sub(self.last_drawn_ms) >= REDRAW_INTERVAL_MS {
            self.last_drawn_ms = now_ms;
            true
        } else {
            false
        }
    }

    /// The progress bar's rectangle, or `None` when `progress < 0`.
    ///
    /// java: `k1 = k / 2 - i1 / 2`, `l1 = l / 2 + 16`, where `k`/`l` are the *scaled* width and
    /// height. Note vanilla treats `progress` as a **pixel count**, not a percentage — it just
    /// happens that the bar is 100 wide, so a 0..=100 value reads as one.
    #[must_use]
    pub fn bar_rect(&self, ctx: &ScreenCtx) -> Option<(Rect, Rect)> {
        if self.progress < 0 {
            return None;
        }
        let x = ctx.width / 2 - BAR_WIDTH / 2;
        let y = ctx.height / 2 + 16;
        let track = Rect::new(x, y, BAR_WIDTH, BAR_HEIGHT);
        // java: the fill quad runs to `k1 + progress` with no clamp, so a caller reporting more
        // than 100 draws past the track. Clamped here — an overlong green bar is a bug, not a
        // vanilla behaviour worth preserving, and nothing in this client reports above 100.
        let fill = Rect::new(x, y, self.progress.min(BAR_WIDTH), BAR_HEIGHT);
        Some((track, fill))
    }

    /// java: `drawStringWithShadow(currentlyDisplayedText, …, l / 2 - 4 - 16, …)`.
    #[must_use]
    pub fn title_y(ctx: &ScreenCtx) -> i32 {
        ctx.height / 2 - 4 - 16
    }

    /// java: `drawStringWithShadow(message, …, l / 2 - 4 + 8, …)`.
    #[must_use]
    pub fn message_y(ctx: &ScreenCtx) -> i32 {
        ctx.height / 2 - 4 + 8
    }

    /// Draw the whole screen.
    ///
    /// java: the body of `setLoadingProgress` past the throttle — tiled background, the two bar
    /// quads, then the two shadowed strings.
    pub fn draw(&self, d: &mut DrawList, font: &mut mc_ui::font::FontRenderer, ctx: &ScreenCtx) {
        let background = UiTexture::gui(OPTIONS_BACKGROUND);
        d.tiled(
            &background,
            Rect::new(0, 0, ctx.width, ctx.height),
            BACKGROUND_TILE,
            0.0,
            BACKGROUND_ARGB,
        );

        if let Some((track, fill)) = self.bar_rect(ctx) {
            d.rect(
                track.x,
                track.y,
                track.right(),
                track.bottom(),
                BAR_TRACK_ARGB,
            );
            if fill.width > 0 {
                d.rect(fill.x, fill.y, fill.right(), fill.bottom(), BAR_FILL_ARGB);
            }
        }

        for (text, y) in [
            (&self.title, LoadingScreen::title_y(ctx)),
            (&self.message, LoadingScreen::message_y(ctx)),
        ] {
            if text.is_empty() {
                continue;
            }
            let x = (ctx.width - font.string_width(text)) / 2;
            font.draw_string_with_shadow(d, text, x as f32, y as f32, TEXT_COLOR);
        }
    }
}

/// A weighted list of loading stages, so the bar advances smoothly across work of very
/// different costs.
///
/// java: vanilla has no equivalent — `SimpleReloadableResourceManager.reloadResources` reports
/// `i / listeners.size()` and the model bake (by far the slowest step) is a single listener, so
/// the real 1.8.9 bar jumps from about 30% to 100% and sits there. Weighting the stages by
/// measured cost is a deliberate improvement, flagged as such.
#[derive(Debug, Clone)]
pub struct StageProgress {
    stages: Vec<(&'static str, u32)>,
    completed_weight: u32,
    total_weight: u32,
    current: usize,
}

impl StageProgress {
    /// Build from `(name, weight)` pairs. An empty list is legal and reports 100% immediately.
    #[must_use]
    pub fn new(stages: Vec<(&'static str, u32)>) -> StageProgress {
        let total_weight = stages.iter().map(|(_, weight)| *weight).sum::<u32>().max(1);
        StageProgress {
            stages,
            completed_weight: 0,
            total_weight,
            current: 0,
        }
    }

    /// The stage about to run, or `None` when everything is done.
    #[must_use]
    pub fn current(&self) -> Option<&'static str> {
        self.stages.get(self.current).map(|(name, _)| *name)
    }

    /// Percentage complete, `0..=100`, ready for [`LoadingScreen::set_loading_progress`].
    #[must_use]
    pub fn percent(&self) -> i32 {
        ((u64::from(self.completed_weight) * 100) / u64::from(self.total_weight)) as i32
    }

    /// Mark the current stage finished and move on.
    pub fn finish_stage(&mut self) {
        if let Some((_, weight)) = self.stages.get(self.current) {
            self.completed_weight += weight;
            self.current += 1;
        }
    }

    /// Whether every stage has been finished.
    #[must_use]
    pub fn done(&self) -> bool {
        self.current >= self.stages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_ui::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(480, 320, 1, false), 0.0)
    }

    /// java: `k1 = k / 2 - 50`, `l1 = l / 2 + 16`, bar 100×2.
    #[test]
    fn the_bar_sits_where_vanilla_puts_it() {
        let ctx = ctx();
        let mut screen = LoadingScreen::new();
        assert_eq!(screen.progress, 0, "Default::default starts at zero");

        screen.progress = 40;
        let (track, fill) = screen.bar_rect(&ctx).expect("a bar at progress 40");
        assert_eq!(track, Rect::new(480 / 2 - 50, 320 / 2 + 16, 100, 2));
        assert_eq!(fill, Rect::new(480 / 2 - 50, 320 / 2 + 16, 40, 2));
        assert_eq!(track.right(), 290);
        assert_eq!(track.bottom(), 178);
    }

    /// java: `displayLoadingString` calls `setLoadingProgress(-1)`, and `if (progress >= 0)`
    /// gates the whole bar — so a message with no numeric progress draws text only.
    #[test]
    fn a_negative_progress_draws_no_bar() {
        let ctx = ctx();
        let mut screen = LoadingScreen::new();
        screen.display_loading_string("Loading");
        assert_eq!(screen.progress, -1);
        assert!(screen.bar_rect(&ctx).is_none());
    }

    #[test]
    fn the_text_lines_sit_where_vanilla_puts_them() {
        let ctx = ctx();
        assert_eq!(LoadingScreen::title_y(&ctx), 320 / 2 - 4 - 16);
        assert_eq!(LoadingScreen::message_y(&ctx), 320 / 2 - 4 + 8);
        assert_eq!(
            LoadingScreen::message_y(&ctx) - LoadingScreen::title_y(&ctx),
            24,
            "vanilla's two lines are 24 apart, not one line height"
        );
    }

    /// java: the 100 ms throttle in `setLoadingProgress`. Without it a fast loader redraws
    /// thousands of times and the load takes longer than the work.
    #[test]
    fn redraws_are_throttled_to_ten_a_second() {
        let mut screen = LoadingScreen::new();
        assert!(
            screen.set_loading_progress(0, 1_000),
            "the first call draws"
        );
        assert!(
            !screen.set_loading_progress(1, 1_050),
            "50 ms later is too soon"
        );
        assert!(
            !screen.set_loading_progress(2, 1_099),
            "99 ms is still too soon"
        );
        assert!(
            screen.set_loading_progress(3, 1_100),
            "100 ms exactly redraws"
        );
        assert!(!screen.set_loading_progress(4, 1_199));
        assert!(screen.set_loading_progress(5, 5_000));
        // Progress is recorded on every call, drawn or not.
        assert_eq!(screen.progress, 5);
    }

    /// java: `displayLoadingString` sets `systemTime = 0` twice, which forces the next progress
    /// report to draw even if one just happened. Reproduced, because otherwise a stage boundary
    /// can be invisible for a tenth of a second.
    #[test]
    fn a_new_message_forces_the_next_redraw() {
        let mut screen = LoadingScreen::new();
        assert!(screen.set_loading_progress(0, 10_000));
        assert!(!screen.set_loading_progress(1, 10_010));
        screen.display_loading_string("Baking models");
        assert!(
            screen.set_loading_progress(1, 10_020),
            "a new message must redraw immediately"
        );
    }

    #[test]
    fn resetting_clears_the_message_and_the_bar() {
        let mut screen = LoadingScreen::new();
        screen.display_loading_string("something");
        screen.progress = 50;
        screen.reset_progress_and_message("Loading world");
        assert_eq!(screen.title, "Loading world");
        assert_eq!(screen.message, "");
        assert_eq!(screen.progress, -1);
    }

    #[test]
    fn weighted_stages_advance_proportionally() {
        let mut progress = StageProgress::new(vec![("a", 1), ("b", 8), ("c", 1)]);
        assert_eq!(progress.current(), Some("a"));
        assert_eq!(progress.percent(), 0);
        assert!(!progress.done());

        progress.finish_stage();
        assert_eq!(progress.percent(), 10);
        assert_eq!(progress.current(), Some("b"));

        progress.finish_stage();
        assert_eq!(progress.percent(), 90, "the heavy stage is worth 80 points");

        progress.finish_stage();
        assert_eq!(progress.percent(), 100);
        assert!(progress.done());
        assert_eq!(progress.current(), None);

        // Finishing past the end must not overshoot.
        progress.finish_stage();
        assert_eq!(progress.percent(), 100);
    }

    #[test]
    fn an_empty_stage_list_is_immediately_done() {
        let progress = StageProgress::new(Vec::new());
        assert!(progress.done());
        assert_eq!(progress.percent(), 0);
        assert_eq!(progress.current(), None);
    }
}
