//! Keystrokes (`keystrokes`): W/A/S/D, the mouse buttons with the CPS counter inside, and the
//! space bar, lit as you press them.
//!
//! ```text
//!        [ W ]
//!  [ A ] [ S ] [ D ]
//!  [ LMB  ] [  RMB ]
//!  [  12 CPS  4 CPS ]   (half-size, inside the mouse keys)
//!  [ ──────────────  ]  the space bar: a short line, no text
//! ```
//!
//! * The state is the **key bindings'** (`HudData::keys`), so a remapped key lights the right
//!   box and nothing lights while a screen is open, which un-presses every binding as vanilla's
//!   `setIngameNotInFocus` does.
//! * A press shows at once; a release fades linearly back over `fade_ms`. A press during the
//!   fade snaps straight back to pressed.
//! * A click released within one frame still lights its mouse key for that frame: the counted
//!   presses (`HudData::click_presses`) moved since the last frame.
//! * Movement labels are the bound key's name when it is 1 to 3 characters
//!   (`HudData::move_key_labels`, rebuilt by the client only when a binding changes), else the
//!   default letter. The CPS strings are rebuilt only when a count changes.
//! * Square keys, 18 px with 2 px gaps, 58 px wide in all (research `keystrokes.md`).

use std::fmt::Write as _;

use mc_mods::ModSettings;

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement};
use crate::widget::{DrawList, Transform, UiTexture};

/// The element's width; every layout is this wide.
const WIDTH: f32 = 58.0;
/// The space bar's height.
const SPACE_HEIGHT: f32 = 10.0;
/// The line drawn on the space bar.
const SPACE_LINE: f32 = 20.0;

/// Key indices.
const LMB: usize = 4;
const RMB: usize = 5;
const SPACE: usize = 6;

/// What a movement key shows when its binding's name is not 1 to 3 characters long.
const DEFAULT_LABELS: [&str; 4] = ["W", "A", "S", "D"];
const MOUSE_LABELS: [&str; 2] = ["LMB", "RMB"];

/// `from` to `to`, per 8-bit channel, alpha included.
#[must_use]
fn lerp_argb(from: u32, to: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    [0, 8, 16, 24].into_iter().fold(0, |out, shift| {
        let a = ((from >> shift) & 0xFF) as f32;
        let b = ((to >> shift) & 0xFF) as f32;
        out | (((a + (b - a) * t).round() as u32) << shift)
    })
}

/// Vanilla's text shadow colour for `argb`: the colour at a quarter brightness, same alpha.
#[must_use]
fn shadow_argb(argb: u32) -> u32 {
    (argb & 0xFF00_0000) | ((argb & 0x00FC_FCFC) >> 2)
}

/// One key's press state.
#[derive(Debug, Clone, Copy, Default)]
struct KeyState {
    down: bool,
    /// When it was last released (`HudData::now_ms` clock); `None` until the first release.
    released_at_ms: Option<u64>,
}

/// The `keystrokes` element.
#[derive(Debug)]
pub struct KeystrokesElement {
    show_mouse: bool,
    show_cps: bool,
    show_space: bool,
    fade_ms: u64,
    pressed_color: u32,
    pressed_text_color: u32,
    /// Released text colour (`text_color`).
    text_color: u32,
    /// Released key colour (`background_color`).
    key_color: u32,
    shadow: bool,
    /// W, A, S, D, LMB, RMB, space.
    keys: [KeyState; 7],
    /// `HudData::click_presses` as of the last frame; `None` before the first.
    seen_presses: Option<[u64; 2]>,
    /// The movement labels as the client last sent them (empty: the default letter).
    labels: [String; 4],
    /// Widths of the W, A, S, D, LMB and RMB labels.
    label_widths: [f32; 6],
    /// The counts `cps_text` was built from; `None` forces a rebuild.
    cps_shown: [Option<u32>; 2],
    cps_text: [String; 2],
    cps_widths: [f32; 2],
    font_generation: u64,
    /// This frame's `HudData::now_ms`.
    now_ms: u64,
    size: (f32, f32),
}

impl Default for KeystrokesElement {
    fn default() -> Self {
        KeystrokesElement {
            show_mouse: true,
            show_cps: true,
            show_space: true,
            fade_ms: 75,
            pressed_color: 0x6FFF_FFFF,
            pressed_text_color: 0xFF00_0000,
            text_color: 0xFFFF_FFFF,
            key_color: 0x6F00_0000,
            shadow: false,
            keys: [KeyState::default(); 7],
            seen_presses: None,
            labels: Default::default(),
            label_widths: [0.0; 6],
            cps_shown: [None; 2],
            cps_text: Default::default(),
            cps_widths: [0.0; 2],
            font_generation: u64::MAX,
            now_ms: 0,
            size: (0.0, 0.0),
        }
    }
}

impl KeystrokesElement {
    /// `(x, y, width, height)` of W, A, S, D, LMB, RMB and the space bar. Rows move up when a
    /// row above them is hidden.
    fn rects(&self) -> [(f32, f32, f32, f32); 7] {
        let space_y = if self.show_mouse { 60.0 } else { 40.0 };
        [
            (20.0, 0.0, 18.0, 18.0),
            (0.0, 20.0, 18.0, 18.0),
            (20.0, 20.0, 18.0, 18.0),
            (40.0, 20.0, 18.0, 18.0),
            (0.0, 40.0, 28.0, 18.0),
            (30.0, 40.0, 28.0, 18.0),
            (0.0, space_y, WIDTH, SPACE_HEIGHT),
        ]
    }

    fn visible(&self, key: usize) -> bool {
        match key {
            LMB | RMB => self.show_mouse,
            SPACE => self.show_space,
            _ => true,
        }
    }

    /// The fill and text colours of `key` this frame, and whether it is held.
    fn colors(&self, key: usize) -> (u32, u32, bool) {
        let state = self.keys[key];
        if state.down {
            return (self.pressed_color, self.pressed_text_color, true);
        }
        let t = match state.released_at_ms {
            Some(at) if self.fade_ms > 0 => {
                self.now_ms.saturating_sub(at) as f32 / self.fade_ms as f32
            }
            _ => 1.0,
        };
        (
            lerp_argb(self.pressed_color, self.key_color, t),
            lerp_argb(self.pressed_text_color, self.text_color, t),
            false,
        )
    }

    fn label(&self, key: usize) -> &str {
        match key {
            0..=3 if !self.labels[key].is_empty() => &self.labels[key],
            0..=3 => DEFAULT_LABELS[key],
            _ => MOUSE_LABELS[key - LMB],
        }
    }
}

impl HudElement for KeystrokesElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("keystrokes", "keystrokes")
    }

    fn sync(&mut self, settings: &ModSettings) {
        const ID: &str = "keystrokes";
        self.show_mouse = settings.bool(ID, "show_mouse");
        self.show_cps = settings.bool(ID, "show_cps");
        self.show_space = settings.bool(ID, "show_space");
        self.fade_ms = settings.f32(ID, "fade_ms").clamp(0.0, 250.0) as u64;
        self.pressed_color = settings.color(ID, "pressed_color");
        self.pressed_text_color = settings.color(ID, "pressed_text_color");
        self.text_color = settings.color(ID, "text_color");
        self.key_color = settings.color(ID, "background_color");
        self.shadow = settings.bool(ID, "text_shadow");
        self.cps_shown = [None; 2];
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        let data = ctx.data;
        self.now_ms = data.now_ms;
        let font_changed = font.layout_generation() != self.font_generation;
        if font_changed {
            self.font_generation = font.layout_generation();
            self.cps_shown = [None; 2];
            for (key, label) in MOUSE_LABELS.iter().enumerate() {
                self.label_widths[LMB + key] = font.string_width(label).max(0) as f32;
            }
        }
        for key in 0..4 {
            let sent = data.move_key_labels[key].as_str();
            if font_changed || self.labels[key] != sent {
                self.labels[key].clear();
                self.labels[key].push_str(sent);
                self.label_widths[key] = font.string_width(self.label(key)).max(0) as f32;
            }
        }
        if self.show_mouse && self.show_cps {
            for (i, count) in [data.cps_left, data.cps_right].into_iter().enumerate() {
                if self.cps_shown[i] != Some(count) {
                    self.cps_shown[i] = Some(count);
                    self.cps_text[i].clear();
                    let _ = write!(self.cps_text[i], "{count} CPS");
                    self.cps_widths[i] = font.string_width(&self.cps_text[i]).max(0) as f32;
                }
            }
        }
        // A counted click since the last frame lights its key for this frame even if the button
        // is already up again (a click shorter than a frame).
        let presses = data.click_presses;
        let latched = self.seen_presses.map_or([false; 2], |seen| {
            [presses[0] != seen[0], presses[1] != seen[1]]
        });
        self.seen_presses = Some(presses);
        let keys = data.keys;
        let down = [
            keys.forward,
            keys.left,
            keys.back,
            keys.right,
            keys.attack || latched[0],
            keys.use_item || latched[1],
            keys.jump,
        ];
        for (state, down) in self.keys.iter_mut().zip(down) {
            if state.down && !down {
                state.released_at_ms = Some(data.now_ms);
            }
            state.down = down;
        }
        let mut height = 38.0;
        if self.show_mouse {
            height += 20.0;
        }
        if self.show_space {
            height += 12.0;
        }
        self.size = (WIDTH, height);
        Some(self.size)
    }

    fn draw(&mut self, _ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        mc_core::profile_scope!("hud.mod.keystrokes");
        let rects = self.rects();
        let colors: [(u32, u32, bool); 7] = std::array::from_fn(|key| self.colors(key));
        // All the fills first, then all the text: two texture runs instead of fourteen.
        for (key, &(x, y, w, h)) in rects.iter().enumerate() {
            if self.visible(key) {
                let fill = colors[key].0;
                d.quad_rect(
                    &UiTexture::Solid,
                    (x, y, x + w, y + h),
                    (0.0, 0.0, 1.0, 1.0),
                    fill,
                );
            }
        }
        let cps = self.show_mouse && self.show_cps;
        for (key, &(x, y, w, _)) in rects.iter().enumerate().take(SPACE) {
            if !self.visible(key) {
                continue;
            }
            let (_, text, held) = colors[key];
            let label_x = x + ((w - self.label_widths[key] + 1.0) * 0.5).floor();
            let label_y = y + if cps && key >= LMB { 3.0 } else { 5.0 };
            // Shadow under black pressed text only muddies it.
            let shadow = self.shadow && !held;
            font.draw_string(d, self.label(key), label_x, label_y, text as i32, shadow);
        }
        if cps {
            d.with_transform(Transform::scale(0.5), |d| {
                for (i, key) in [LMB, RMB].into_iter().enumerate() {
                    let (x, y, w, _) = rects[key];
                    let (_, text, held) = colors[key];
                    // Half-size text: positions are in half pixels, centred on whole ones.
                    let left = (x * 2.0 + (w * 2.0 - self.cps_widths[i] + 1.0) * 0.5).floor();
                    let shadow = self.shadow && !held;
                    let top = (y + 12.0) * 2.0;
                    font.draw_string(d, &self.cps_text[i], left, top, text as i32, shadow);
                }
            });
        }
        if self.show_space {
            let (x, y, w, _) = rects[SPACE];
            let (_, text, held) = colors[SPACE];
            let left = x + (w - SPACE_LINE) * 0.5;
            let top = y + 4.0;
            if self.shadow && !held {
                let shadow = shadow_argb(text);
                let rect = (left + 1.0, top + 1.0, left + SPACE_LINE + 1.0, top + 2.0);
                d.quad_rect(&UiTexture::Solid, rect, (0.0, 0.0, 1.0, 1.0), shadow);
            }
            let rect = (left, top, left + SPACE_LINE, top + 1.0);
            d.quad_rect(&UiTexture::Solid, rect, (0.0, 0.0, 1.0, 1.0), text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::{HudData, HudKeys};
    use crate::widget::draw::unpack_argb;
    use crate::widget::test_support::uniform_font;

    fn ctx<'a>(data: &'a HudData, settings: &'a ModSettings) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen: (960.0, 540.0),
            pixel_scale: 2.0,
            editing: false,
        }
    }

    fn at(now_ms: u64, keys: HudKeys) -> HudData {
        HudData {
            now_ms,
            keys,
            ..HudData::default()
        }
    }

    /// The fill colour of the `n`th drawn key (fills come first, one quad each).
    fn fill(d: &DrawList, n: usize) -> [f32; 4] {
        d.vertices()[n * 6].color
    }

    fn frame(element: &mut KeystrokesElement, data: &HudData) -> DrawList {
        let settings = ModSettings::new();
        let mut font = uniform_font();
        element.layout(&ctx(data, &settings), &font).unwrap();
        let mut d = DrawList::new();
        element.draw(&ctx(data, &settings), &mut font, &mut d);
        d
    }

    #[test]
    fn rows_drop_out_and_move_up() {
        let mut settings = ModSettings::new();
        let font = uniform_font();
        let mut element = KeystrokesElement::default();
        element.sync(&settings);
        let data = HudData::default();
        assert_eq!(
            element.layout(&ctx(&data, &settings), &font),
            Some((58.0, 70.0))
        );
        settings.set_bool("keystrokes", "show_space", false);
        element.sync(&settings);
        assert_eq!(
            element.layout(&ctx(&data, &settings), &font),
            Some((58.0, 58.0))
        );
        settings.set_bool("keystrokes", "show_space", true);
        settings.set_bool("keystrokes", "show_mouse", false);
        element.sync(&settings);
        assert_eq!(
            element.layout(&ctx(&data, &settings), &font),
            Some((58.0, 50.0))
        );
        assert_eq!(element.rects()[SPACE].1, 40.0, "the space bar moves up");
    }

    #[test]
    fn a_press_is_instant_and_a_release_fades() {
        let settings = ModSettings::new();
        let mut element = KeystrokesElement::default();
        element.sync(&settings);
        let released = unpack_argb(0x6F00_0000);
        let pressed = unpack_argb(0x6FFF_FFFF);
        let idle = frame(&mut element, &at(1000, HudKeys::default()));
        assert_eq!(fill(&idle, 0), released, "never pressed: released colour");
        let w_down = HudKeys {
            forward: true,
            ..HudKeys::default()
        };
        assert_eq!(fill(&frame(&mut element, &at(1016, w_down)), 0), pressed);
        // Released at 1032: still the pressed colour on that frame, halfway at +37.5 ms...
        assert_eq!(
            fill(&frame(&mut element, &at(1032, HudKeys::default())), 0),
            pressed
        );
        let half = fill(&frame(&mut element, &at(1032 + 37, HudKeys::default())), 0);
        assert!(half[0] > 0.4 && half[0] < 0.6, "{half:?}");
        // ...and back to the released colour after the 75 ms fade.
        assert_eq!(
            fill(&frame(&mut element, &at(1032 + 75, HudKeys::default())), 0),
            released
        );
        // A press during a fade snaps straight back to pressed.
        frame(&mut element, &at(2000, w_down));
        frame(&mut element, &at(2016, HudKeys::default()));
        assert_eq!(fill(&frame(&mut element, &at(2020, w_down)), 0), pressed);
    }

    #[test]
    fn a_click_inside_one_frame_still_lights_the_key() {
        let settings = ModSettings::new();
        let mut element = KeystrokesElement::default();
        element.sync(&settings);
        let mut data = at(1000, HudKeys::default());
        data.click_presses = [10, 3];
        let released = unpack_argb(0x6F00_0000);
        let pressed = unpack_argb(0x6FFF_FFFF);
        // The first frame only learns the counters.
        assert_eq!(fill(&frame(&mut element, &data), LMB), released);
        data.now_ms = 1016;
        data.click_presses = [11, 3];
        let d = frame(&mut element, &data);
        assert_eq!(
            fill(&d, LMB),
            pressed,
            "pressed and released between two frames"
        );
        assert_eq!(fill(&d, RMB), released);
        // From the next frame it fades like any release.
        data.now_ms = 1032;
        frame(&mut element, &data);
        data.now_ms = 1032 + 75;
        assert_eq!(fill(&frame(&mut element, &data), LMB), released);
    }

    #[test]
    fn labels_follow_the_bindings_and_cps_strings_are_cached() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = KeystrokesElement::default();
        element.sync(&settings);
        let mut data = HudData {
            cps_left: 12,
            cps_right: 4,
            ..HudData::default()
        };
        data.move_key_labels[0] = "UP".to_owned();
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.label(0), "UP");
        assert_eq!(element.label(1), "A", "no name sent: the default letter");
        assert_eq!(element.label(LMB), "LMB");
        assert_eq!(element.cps_text, ["12 CPS".to_owned(), "4 CPS".to_owned()]);
        element.cps_text[0].push('!');
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.cps_text[0], "12 CPS!", "same count: not rebuilt");
        data.cps_left = 13;
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.cps_text[0], "13 CPS");
    }

    #[test]
    fn colour_helpers() {
        assert_eq!(lerp_argb(0x6FFF_FFFF, 0x6F00_0000, 0.0), 0x6FFF_FFFF);
        assert_eq!(lerp_argb(0x6FFF_FFFF, 0x6F00_0000, 1.0), 0x6F00_0000);
        assert_eq!(lerp_argb(0x0000_0000, 0xFF00_00FF, 0.5), 0x8000_0080);
        assert_eq!(shadow_argb(0xFFFF_FFFF), 0xFF3F_3F3F);
    }
}
