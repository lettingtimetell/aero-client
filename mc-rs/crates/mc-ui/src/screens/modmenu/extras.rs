//! Window 3's extras.
//!
//! * **Live warnings,** shown only while they are true, under a mod's header: "Does nothing
//!   right now: Mipmap Levels is Off in Video Settings", "Does nothing while MSAA is on". They
//!   read the client's `options.txt` settings ([`super::MenuGame::options`]); without them (tests,
//!   harnesses) none shows.
//! * **The shared HUD style's STYLE block** (decision D5, `mc_mods::hud_style`): a readout's STYLE
//!   heading gets a "HUD style / Custom" choice. On the HUD style its rows edit the shared style
//!   (every readout on it changes together) and fold away behind a summary line, so up to 39
//!   repeated settings disappear from view; on Custom they are its own, as before.

use mc_mods::{hud_style, presets, ModSettings, OptionValue};
use mc_render::GameSettings;

use super::keybinds::draw_warn_mark;
use super::layout::Area;
use super::paint::{self, css, white, Fill, Painter, Rect};
use super::state::{NavRow, Pane, Target};
use super::text::{TextStyle, Weight};
use super::widgets::{draw_checker, draw_focus_ring, draw_wrapped, shadowed};
use super::{MenuEnv, MenuInput, ModMenuScreen};

/// The extras' own hit targets (`Target::Extras`). Must stay `Copy` and `PartialEq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExtrasTarget {
    /// A readout's style choice: (mod, use the HUD style).
    StyleMode(usize, bool),
    /// Show or fold the HUD style's settings inside a readout that uses it.
    StyleOpen,
    /// The HUD editor's tray: open or close it.
    Tray,
    /// The tray: turn this HUD mod on, so it shows up to be placed.
    TrayAdd(usize),
    /// The tray's panel itself: swallows clicks so nothing under it starts a drag.
    TrayPanel,
}

/// The extras' state.
#[derive(Debug, Default)]
pub(super) struct ExtrasState {
    /// The HUD style's settings show inside readouts that use it (folded by default).
    pub(super) style_open: bool,
    /// The HUD editor's tray is open.
    pub(super) tray_open: bool,
    /// The HUD slot "Move on screen" or the tray picked, and when (it glows for a moment).
    pub(super) picked: Option<(usize, std::time::Instant)>,
}

/// What [`ModMenuScreen::draw_style_block`] drew.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum StyleBlock {
    /// Draw the block's rows as usual from this y.
    Rows(f32),
    /// The block is done (folded): continue below this y.
    Done(f32),
}

/// The warning about mod `id` that is true now, if any.
#[must_use]
pub(super) fn live_warning(id: &str, options: &GameSettings) -> Option<String> {
    let msaa = options.msaa_samples;
    match id {
        "anisotropic_filtering" if options.mipmap_levels <= 0 => Some(
            "Does nothing right now: Mipmap Levels is Off in Video Settings (yours: 0). Set it \
             to 1 or more."
                .to_owned(),
        ),
        "anti_aliasing" | "motion_blur" | "color_grading" if msaa > 1 => Some(format!(
            "Does nothing while MSAA is on (yours: {msaa}x, msaaSamples in options.txt). Set \
             msaaSamples to 1 and restart to use it."
        )),
        "shaders_classic" | "shaders_dreamy" | "shaders_underwater" if msaa > 1 => Some(format!(
            "Stays off while MSAA is on (yours: {msaa}x, msaaSamples in options.txt). Set \
             msaaSamples to 1 and restart to use it."
        )),
        "volumetric_clouds" if options.clouds == 0 => {
            Some("Hidden right now: Clouds is Off in Video Settings.".to_owned())
        }
        _ => None,
    }
}

/// "White text, shadow, dark box": the shared style in words.
fn style_summary(settings: &ModSettings) -> String {
    let flag = |id: &str| matches!(settings.hud_style_value(id), Some(OptionValue::Bool(true)));
    let mut parts = vec!["Text"];
    if flag(presets::TEXT_SHADOW_ID) {
        parts.push("shadow");
    }
    if flag(presets::BACKGROUND_ID) {
        parts.push("background");
    }
    if flag(presets::BRACKETS_ID) {
        parts.push("[brackets]");
    }
    parts.join(", ")
}

impl ModMenuScreen {
    /// Under window 3's header: warnings about mod `m` that are true right now. Returns the y
    /// below what it drew (`y` when nothing applies).
    pub(super) fn draw_mod_warnings(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        area: &Area,
        y: f32,
    ) -> f32 {
        let Some(options) = env.game.options else {
            return y;
        };
        let Some(descriptor) = env.settings.catalog().get(m) else {
            return y;
        };
        let Some(text) = live_warning(descriptor.id, options) else {
            return y;
        };
        let s = area.s;
        let style = TextStyle::new(11.5 * s);
        let (_, lh) = env.text.metrics(style);
        let text_x = area.x + 26.0 * s;
        let text_w = area.right() - text_x - 10.0 * s;
        // The box goes under the text, so count its lines first (as `draw_wrapped` breaks them).
        let mut lines = 0;
        let mut rest = text.trim();
        while !rest.is_empty() {
            let mut cut = env
                .text
                .fit_words(rest, text_w, style, env.mc_font)
                .min(rest.len());
            while cut > 0 && !rest.is_char_boundary(cut) {
                cut -= 1;
            }
            if cut == 0 {
                cut = rest.chars().next().map_or(rest.len(), char::len_utf8);
            }
            rest = rest[cut..].trim_start();
            lines += 1;
        }
        let h = lines.max(1) as f32 * (lh + 2.0 * s) + 14.0 * s;
        let r = Rect::new(area.x, y, area.w, h);
        painter.round_rect(r, 10.0 * s, Fill::Solid(css(255, 196, 82, 0.10)));
        painter.round_rect_stroke(r, 10.0 * s, 1.0, 0.5, css(255, 196, 82, 0.55));
        draw_warn_mark(
            painter,
            area.x + 13.0 * s,
            y + 7.0 * s + lh * 0.5,
            s,
            css(255, 196, 82, 0.95),
        );
        draw_wrapped(
            painter,
            env,
            &text,
            text_x,
            y + 7.0 * s,
            text_w,
            style,
            css(255, 226, 170, 0.97),
            s,
        );
        r.bottom() + 10.0 * s
    }

    /// Mod `m`'s STYLE block at `y` (under its heading): for a readout, the "HUD style /
    /// Custom" choice, then (on the HUD style) a summary line that unfolds the shared rows.
    /// [`StyleBlock::Rows`] lets the usual rows follow (they read and write the shared style
    /// while the readout uses it); [`StyleBlock::Done`] skips them (folded).
    pub(super) fn draw_style_block(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        area: &Area,
        y: f32,
    ) -> StyleBlock {
        let settings = env.settings;
        if !settings.hud_style_capable_at(m) {
            return StyleBlock::Rows(y);
        }
        let s = area.s;
        let shared = settings.uses_hud_style_at(m);
        let details = Pane::Details.index();
        // The choice: two segments.
        let style = TextStyle::new(11.5 * s).weight(Weight::Semibold);
        let x0 = area.x + 14.0 * s;
        let seg_h = 26.0 * s;
        let labels = [("HUD style", true), ("Custom", false)];
        let widths: Vec<f32> = labels
            .iter()
            .map(|(label, _)| env.text.measure(label, style, env.mc_font) + 26.0 * s)
            .collect();
        let total: f32 = widths.iter().sum();
        let bar = Rect::new(x0, y + 2.0 * s, total, seg_h);
        painter.round_rect(bar, seg_h * 0.5, Fill::Solid(css(0, 0, 0, 0.22)));
        painter.round_rect_stroke(bar, seg_h * 0.5, 1.0, 0.5, white(0.22));
        let mut x = x0;
        let index = self.nav[details].len();
        let focused =
            self.keyboard && self.pane == Pane::Details && self.cursors[details] == Some(index);
        if focused {
            draw_focus_ring(painter, bar, seg_h * 0.5, s);
        }
        for ((label, hud), w) in labels.into_iter().zip(&widths) {
            let seg = Rect::new(x, bar.y, *w, seg_h);
            let on = hud == shared;
            let hit = area.hit(painter, seg);
            let hover = hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
            if on {
                painter.round_rect(seg, seg_h * 0.5, Fill::Solid(white(0.26)));
                painter.round_rect_stroke(seg, seg_h * 0.5, 1.0, 0.5, white(0.62));
            } else if hover {
                painter.round_rect(seg, seg_h * 0.5, Fill::Solid(white(0.10)));
            }
            let tw = env.text.measure(label, style, env.mc_font);
            let (_, lh) = env.text.metrics(style);
            shadowed(
                painter,
                env,
                label,
                seg.x + (seg.w - tw) * 0.5,
                seg.y + (seg.h - lh) * 0.5,
                style,
                white(if on { 0.98 } else { 0.70 }),
                s,
            );
            if let Some(r) = hit {
                self.hits
                    .push((r, Target::Extras(ExtrasTarget::StyleMode(m, hud))));
                if r.contains(self.cursor.0, self.cursor.1) {
                    let tip = if hud {
                        "Use the HUD style: one look shared by every readout set to it."
                    } else {
                        "Use this readout's own style settings."
                    };
                    self.set_tooltip(tip, self.cursor);
                }
            }
            x += w;
        }
        self.nav[details].push(NavRow {
            enter: Target::Extras(ExtrasTarget::StyleMode(m, !shared)),
            space: Some(Target::Extras(ExtrasTarget::StyleMode(m, !shared))),
            span: area.span(bar.y, bar.bottom()),
        });
        let mut y = bar.bottom() + 8.0 * s;
        let note_style = TextStyle::new(11.0 * s);
        if !shared {
            let note = if hud_style::uses_hud_style_by_default(settings.catalog()[m]) {
                "This readout's own look. Pick HUD style to match the other readouts."
            } else {
                "Its own look, which differs from the HUD style out of the box. Pick HUD style \
                 to match the other readouts."
            };
            y = draw_wrapped(
                painter,
                env,
                note,
                x0,
                y,
                area.right() - x0,
                note_style,
                white(0.55),
                s,
            ) + 6.0 * s;
            return StyleBlock::Rows(y);
        }
        // On the HUD style: a summary row with swatches; a click unfolds the shared rows.
        let flashed = self.flash_options.iter().any(|&o| {
            settings.catalog()[m]
                .options
                .get(o)
                .is_some_and(|option| hud_style::slot(option.id).is_some())
        });
        let open = self.extras.style_open || flashed;
        let row = Rect::new(area.x, y, area.w, 30.0 * s);
        let hit = area.hit(painter, row);
        let hover = hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        if hover {
            painter.round_rect(row, 8.0 * s, Fill::Solid(white(0.07)));
        }
        let users = settings.hud_style_users();
        let mut sx = x0;
        for id in [presets::TEXT_COLOR_ID, presets::BACKGROUND_COLOR_ID] {
            if let Some(OptionValue::Color(argb)) = settings.hud_style_value(id) {
                let swatch = Rect::new(sx, row.y + 8.0 * s, 16.0 * s, 14.0 * s);
                draw_checker(painter, swatch, s);
                painter.round_rect(swatch, 4.0 * s, Fill::Solid(paint::argb(*argb)));
                painter.round_rect_stroke(swatch, 4.0 * s, 1.0, 0.5, white(0.45));
                sx = swatch.right() + 5.0 * s;
            }
        }
        let summary = format!(
            "{}  \u{b7}  shared by {users} readout{}",
            style_summary(settings),
            if users == 1 { "" } else { "s" }
        );
        let (_, lh) = env.text.metrics(note_style);
        let chevron = if open { "Hide" } else { "Edit" };
        let cstyle = TextStyle::new(11.0 * s).weight(Weight::Semibold);
        let cw = env.text.measure(chevron, cstyle, env.mc_font);
        let saved = painter.clip();
        let clip = Rect::new(sx, row.y, row.right() - cw - 18.0 * s - sx, row.h);
        painter.set_clip(Some(saved.map_or(clip, |c| c.intersect(&clip))));
        shadowed(
            painter,
            env,
            &summary,
            sx + 3.0 * s,
            row.y + (row.h - lh) * 0.5,
            note_style,
            white(0.72),
            s,
        );
        painter.set_clip(saved);
        shadowed(
            painter,
            env,
            chevron,
            row.right() - cw - 10.0 * s,
            row.y + (row.h - lh) * 0.5,
            cstyle,
            white(if hover { 0.98 } else { 0.78 }),
            s,
        );
        if let Some(r) = hit {
            self.hits.push((r, Target::Extras(ExtrasTarget::StyleOpen)));
            if r.contains(self.cursor.0, self.cursor.1) {
                let tip = if open {
                    "Fold the HUD style's settings away."
                } else {
                    "Show the HUD style's settings. Changes apply to every readout set to HUD style."
                };
                self.set_tooltip(tip, self.cursor);
            }
        }
        self.nav[details].push(NavRow {
            enter: Target::Extras(ExtrasTarget::StyleOpen),
            space: Some(Target::Extras(ExtrasTarget::StyleOpen)),
            span: area.span(row.y, row.bottom()),
        });
        y = row.bottom() + 4.0 * s;
        if open {
            y = draw_wrapped(
                painter,
                env,
                "These change every readout set to HUD style.",
                x0,
                y,
                area.right() - x0,
                note_style,
                white(0.55),
                s,
            ) + 4.0 * s;
            StyleBlock::Rows(y)
        } else {
            StyleBlock::Done(y + 2.0 * s)
        }
    }

    /// Where the HUD style's summary row (a click shows or folds its settings) was drawn last
    /// frame in window 3.
    #[must_use]
    pub fn hud_style_fold_rect(&self) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::Extras(ExtrasTarget::StyleOpen))
    }

    /// Where the HUD editor's "Add to HUD" pill (the tray) was drawn last frame.
    #[must_use]
    pub fn tray_rect(&self) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::Extras(ExtrasTarget::Tray))
    }

    /// A click on one of the extras' targets (`button`: 0 left, 1 right).
    pub(super) fn activate_extras(
        &mut self,
        target: ExtrasTarget,
        button: u8,
        input: &mut MenuInput<'_>,
    ) {
        if button != 0 {
            return;
        }
        match target {
            ExtrasTarget::StyleMode(m, hud) => {
                input.settings.set_uses_hud_style_at(m, hud);
                // The rows under it changed meaning: the cached layout stays valid (same rows).
            }
            ExtrasTarget::StyleOpen => self.extras.style_open = !self.extras.style_open,
            ExtrasTarget::Tray => self.extras.tray_open = !self.extras.tray_open,
            ExtrasTarget::TrayAdd(m) => {
                input.settings.set_enabled_at(m, true);
                // It glows where it appears, and the arrows nudge it.
                if let Some(slot) = input.hud.slots().iter().position(|s| s.mod_index == m) {
                    self.pick_hud_slot(slot, input);
                }
            }
            ExtrasTarget::TrayPanel => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warnings_show_only_while_true() {
        let mut options = GameSettings {
            mipmap_levels: 4,
            msaa_samples: 1,
            clouds: 2,
            ..GameSettings::default()
        };
        for id in [
            "anisotropic_filtering",
            "anti_aliasing",
            "motion_blur",
            "color_grading",
            "shaders_classic",
            "volumetric_clouds",
            "fps",
        ] {
            assert_eq!(live_warning(id, &options), None, "{id}");
        }
        options.mipmap_levels = 0;
        assert!(live_warning("anisotropic_filtering", &options)
            .unwrap()
            .contains("Mipmap Levels"));
        options.msaa_samples = 4;
        for id in [
            "anti_aliasing",
            "motion_blur",
            "color_grading",
            "shaders_dreamy",
        ] {
            assert!(live_warning(id, &options).unwrap().contains("4x"), "{id}");
        }
        options.clouds = 0;
        assert!(live_warning("volumetric_clouds", &options).is_some());
        assert_eq!(live_warning("fps", &options), None);
    }
}
