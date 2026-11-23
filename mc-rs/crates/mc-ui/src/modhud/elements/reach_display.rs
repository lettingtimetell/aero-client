//! The reach readout (`reach_display`): `2.87 blocks` in the shared readout box (see
//! [`super::fps`]), at least [`REACH_MIN_WIDTH`] wide.
//!
//! `HudData::reach` is filled by the client (`mc-client/src/app/mods/m1_hud.rs`) when
//! your own melee attack is sent: the distance from your eyes to the point where the look ray
//! entered the target's box expanded by its collision border, i.e. vanilla's own
//! `objectMouseOver.hitVec` for that attack. Survival hits therefore never read above 3.00.
//! After `reset_seconds` without a hit the readout goes back to `0.00 blocks` (or hides).

use std::fmt::Write as _;

use mc_mods::ModSettings;

use super::fps::{draw_readout, readout_size};
use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement, HudTextStyle};
use crate::widget::DrawList;

/// The narrowest the reach box gets, in GUI pixels.
pub const REACH_MIN_WIDTH: f32 = 65.0;

/// What the HUD editor shows when there is no recent hit to place it by.
const PREVIEW_DISTANCE: f64 = 2.87;

/// The `reach_display` element.
#[derive(Debug)]
pub struct ReachDisplayElement {
    style: HudTextStyle,
    decimals: usize,
    reset_ms: u64,
    hide_when_idle: bool,
    /// The distance (as bits, `None` = idle) the text was built from; `None` forces a rebuild.
    shown: Option<Option<u64>>,
    font_generation: u64,
    text: String,
    scratch: String,
    size: (f32, f32),
}

impl Default for ReachDisplayElement {
    fn default() -> Self {
        ReachDisplayElement {
            style: HudTextStyle::default(),
            decimals: 2,
            reset_ms: 3000,
            hide_when_idle: false,
            shown: None,
            font_generation: u64::MAX,
            text: String::new(),
            scratch: String::new(),
            size: (0.0, 0.0),
        }
    }
}

impl ReachDisplayElement {
    /// The distance to show this frame: the last hit while it is recent, the editor preview
    /// while placing, `None` when idle.
    fn current(&self, ctx: &HudCtx<'_>) -> Option<f64> {
        let now = ctx.data.now_ms;
        let live = ctx
            .data
            .reach
            .filter(|reach| now.saturating_sub(reach.at_ms) <= self.reset_ms)
            .map(|reach| reach.distance);
        match live {
            None if ctx.editing => Some(PREVIEW_DISTANCE),
            live => live,
        }
    }
}

impl HudElement for ReachDisplayElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("reach_display", "reach_display")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.style = HudTextStyle::read(settings, "reach_display");
        self.decimals = settings.f32("reach_display", "decimals").clamp(0.0, 3.0) as usize;
        self.reset_ms = (settings
            .f32("reach_display", "reset_seconds")
            .clamp(1.0, 10.0)
            * 1000.0) as u64;
        self.hide_when_idle = settings.bool("reach_display", "hide_when_idle");
        self.shown = None;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        let distance = self.current(ctx);
        if distance.is_none() && self.hide_when_idle {
            return None;
        }
        let key = distance.map(f64::to_bits);
        if self.shown != Some(key) || font.layout_generation() != self.font_generation {
            self.shown = Some(key);
            self.font_generation = font.layout_generation();
            self.scratch.clear();
            // The measured value, rounded to nearest at `decimals` places; never clamped or
            // biased (vanilla's own survival cutoff, > 3.0 is a miss, bounds it at 3.00).
            let _ = write!(
                self.scratch,
                "{:.*} blocks",
                self.decimals,
                distance.unwrap_or(0.0)
            );
            self.style.decorate(&self.scratch, &mut self.text);
            self.size = readout_size(&self.style, font, &self.text, REACH_MIN_WIDTH);
        }
        Some(self.size)
    }

    fn draw(&mut self, _ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        draw_readout(&self.style, d, font, &self.text, self.size.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::{HudData, HudReach};
    use crate::widget::test_support::uniform_font;

    fn ctx<'a>(data: &'a HudData, settings: &'a ModSettings, editing: bool) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen: (960.0, 540.0),
            pixel_scale: 2.0,
            editing,
        }
    }

    fn hit(distance: f64, at_ms: u64, now_ms: u64) -> HudData {
        HudData {
            now_ms,
            reach: Some(HudReach { distance, at_ms }),
            ..HudData::default()
        }
    }

    #[test]
    fn a_recent_hit_shows_then_resets_after_the_timeout() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ReachDisplayElement::default();
        element.sync(&settings);
        let size = element
            .layout(&ctx(&hit(2.8712, 1000, 1000), &settings, false), &font)
            .unwrap();
        assert_eq!(element.text, "2.87 blocks");
        assert_eq!(size.0, REACH_MIN_WIDTH);
        // Still shown at exactly the default 3 s...
        element.layout(&ctx(&hit(2.8712, 1000, 4000), &settings, false), &font);
        assert_eq!(element.text, "2.87 blocks");
        // ...and idle after it.
        element.layout(&ctx(&hit(2.8712, 1000, 4001), &settings, false), &font);
        assert_eq!(element.text, "0.00 blocks");
        // Before any hit at all.
        element.layout(&ctx(&HudData::default(), &settings, false), &font);
        assert_eq!(element.text, "0.00 blocks");
    }

    #[test]
    fn decimals_hide_when_idle_and_the_editor_preview() {
        let mut settings = ModSettings::new();
        settings.set_f32("reach_display", "decimals", 0.0);
        settings.set_bool("reach_display", "hide_when_idle", true);
        settings.set_f32("reach_display", "reset_seconds", 1.0);
        let font = uniform_font();
        let mut element = ReachDisplayElement::default();
        element.sync(&settings);
        element.layout(&ctx(&hit(2.6, 0, 500), &settings, false), &font);
        assert_eq!(element.text, "3 blocks");
        let idle = hit(2.6, 0, 1001);
        assert!(element
            .layout(&ctx(&idle, &settings, false), &font)
            .is_none());
        // The editor still shows something to drag.
        assert!(element
            .layout(&ctx(&idle, &settings, true), &font)
            .is_some());
        assert_eq!(element.text, "3 blocks");
        settings.set_f32("reach_display", "decimals", 3.0);
        element.sync(&settings);
        element.layout(&ctx(&idle, &settings, true), &font);
        assert_eq!(element.text, "2.870 blocks");
    }
}
