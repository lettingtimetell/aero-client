//! The version slider on the Aero title screen: "1.7.10 | 1.8.9", the Minecraft version the next
//! connection plays.
//!
//! A slim glass row at the top of the button column, under the title and above Multiplayer, as
//! wide as the buttons and drawn with their glass. The versions sit side by side, oldest first
//! ([`GameVersion::ALL`]), and a bright knob in the mod menu's switch style rests behind the
//! selected one, sliding (eased) to another when it is picked. Clicking a version picks it;
//! dragging the knob picks the version it is released nearest to. The row pops in right after
//! the title, before the buttons.
//!
//! The slider only reports a pick ([`VersionSlider::mouse_released`]). The client saves it and
//! hands the selection back every frame ([`AeroTitleEnv::game_version`]), so the knob always
//! shows what the next connection will play, wherever it was changed. Nothing is allocated per
//! frame: the labels are the versions' own `&'static str` names.

use mc_version::GameVersion;

use super::layout::Layout;
use super::{draw_button_glass, haloed, AeroTitleEnv, ButtonLook, Layer, CRISP_ALPHA};
use crate::screens::modmenu::anim;
use crate::screens::modmenu::glass::{GlassFrame, GlassRect};
use crate::screens::modmenu::paint::{black, white, Fill, Painter, Rect};
use crate::screens::modmenu::text::{TextStyle, Weight};

/// How many versions the slider shows.
const COUNT: usize = GameVersion::ALL.len();
/// Between the row's edge and the knob, in design pixels (1080p).
const INSET: f32 = 4.0;
/// How far the pointer moves, after a press on the knob, before the knob follows it (design
/// pixels): a click that wobbles a little is still a click.
const DRAG_SLOP: f32 = 4.0;
/// The rate the knob eases to the selected version at (per second; see [`anim::approach`]).
const KNOB_RATE: f32 = 16.0;
/// The rate the hover and press highlights ease at, as the buttons'.
const HIGHLIGHT_RATE: f32 = 20.0;
/// The labels' em size, in design pixels.
const LABEL_SIZE: f32 = 17.0;

/// A press on the slider, waiting for its release.
#[derive(Debug, Clone, Copy)]
struct Press {
    /// Where it landed (physical pixels).
    x: f32,
    /// The knob's position then.
    knob: f32,
    /// It landed on the knob, so moving the pointer drags the knob.
    on_knob: bool,
    /// The pointer has moved far enough to drag.
    dragging: bool,
}

/// The slider's state on one title screen.
#[derive(Debug, Default)]
pub struct VersionSlider {
    /// The row in the last frame (physical pixels).
    rect: Rect,
    /// Physical pixels per design pixel in the last frame.
    s: f32,
    /// Whether the row had started to appear in the last frame.
    shown: bool,
    /// The version the knob rests on: the selection, or the version just picked.
    selected: GameVersion,
    /// The knob's eased position: `0` on the oldest version, `COUNT - 1` on the newest. `None`
    /// before the first frame, so the knob starts where it rests.
    knob: Option<f32>,
    /// Eased hover highlight, 0..1.
    hover: f32,
    /// Eased press highlight, 0..1.
    press: f32,
    pressed: Option<Press>,
}

impl VersionSlider {
    /// The row's rectangle in the last frame (physical pixels), once it has started to appear.
    #[must_use]
    pub fn rect(&self) -> Option<Rect> {
        self.shown.then_some(self.rect)
    }

    /// Where `version` sits in the row in the last frame: a click there picks it.
    #[must_use]
    pub fn version_rect(&self, version: GameVersion) -> Rect {
        slot(self.track(), index_of(version) as f32)
    }

    /// The version the knob rests on.
    #[must_use]
    pub fn selected(&self) -> GameVersion {
        self.selected
    }

    /// Whether the slider is on screen at `(x, y)`.
    fn takes(&self, x: f32, y: f32) -> bool {
        self.shown && self.rect.contains(x, y)
    }

    /// Whether a press on the slider is waiting for its release.
    #[must_use]
    pub fn has_press(&self) -> bool {
        self.pressed.is_some()
    }

    /// Forget a press whose release will never come (the window lost focus while it was held):
    /// the knob stops following the pointer and goes back to the selected version.
    pub fn cancel_press(&mut self) {
        self.pressed = None;
    }

    /// A mouse button went down at `(x, y)`. Returns whether the slider took it: it takes every
    /// press on itself, and only the left button picks. A press that misses the row still ends
    /// the press before it, as a press on the title screen's buttons does.
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8) -> bool {
        self.pressed = None;
        if !self.takes(x, y) {
            return false;
        }
        self.pressed = (button == 0).then(|| {
            let knob = self.knob_position();
            Press {
                x,
                knob,
                on_knob: slot(self.track(), knob).contains(x, y),
                dragging: false,
            }
        });
        true
    }

    /// A mouse button went up at `(x, y)`: the version picked, when it is not the selected one.
    /// A click picks the version under the pointer (released on the slider, like a button); a
    /// drag of the knob picks the version the knob is nearest to, wherever the pointer is.
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8) -> Option<GameVersion> {
        if button != 0 {
            return None;
        }
        let press = self.pressed.take()?;
        let index = if self.drags(&press, x) {
            self.drag_position(&press, x).round() as usize
        } else if self.takes(x, y) {
            self.index_at(x)
        } else {
            return None;
        };
        let version = *GameVersion::ALL.get(index)?;
        if version == self.selected {
            return None;
        }
        self.selected = version;
        Some(version)
    }

    /// Draw this frame's slider (called by `AeroTitleScreen::build_layers` after the buttons, so
    /// its glass pane follows theirs). `clock` is the screen's `(elapsed ms since its pop-in,
    /// seconds since the last frame, cursor)`.
    pub(super) fn build(
        &mut self,
        painter: &mut Painter<'_>,
        crisp: Option<&mut Painter<'_>>,
        env: &mut AeroTitleEnv<'_>,
        layout: &Layout,
        glass: &mut GlassFrame,
        (elapsed, dt, cursor): (f32, f32, (f32, f32)),
    ) {
        let s = layout.s;
        self.rect = layout.version;
        self.s = s;
        self.selected = env.game_version;
        let progress = anim::window_open_progress(elapsed, super::VERSION_POP);
        self.shown = progress > 0.0;

        // ---- the press, the highlights and the knob ----
        if let Some(press) = self.pressed.as_mut() {
            if press.on_knob && (cursor.0 - press.x).abs() > DRAG_SLOP * s {
                press.dragging = true;
            }
        }
        let over = self.shown && self.rect.contains(cursor.0, cursor.1);
        let k = anim::approach(dt, HIGHLIGHT_RATE);
        ease(&mut self.hover, if over { 1.0 } else { 0.0 }, k);
        let pressing = over && self.pressed.is_some();
        ease(&mut self.press, if pressing { 1.0 } else { 0.0 }, k);
        // Under the pointer while it is dragged, otherwise easing to the selected version.
        let target = index_of(self.selected) as f32;
        let knob = match self.pressed {
            Some(press) if press.dragging => self.drag_position(&press, cursor.0),
            _ => {
                let mut knob = self.knob.unwrap_or(target);
                ease(&mut knob, target, anim::approach(dt, KNOB_RATE));
                knob
            }
        };
        self.knob = Some(knob);
        if !self.shown {
            return;
        }

        // ---- the pane ----
        let slide = (1.0 - progress) * anim::SLIDE_DIP * s;
        // A pressed slider sinks in a little, as a pressed button does.
        let rect = self.rect.inset(self.press * s);
        glass.rects.push(GlassRect {
            rect: [rect.x, rect.y + slide, rect.w, rect.h],
            radius: layout.radius,
            alpha: progress,
        });
        let look = ButtonLook {
            rect,
            radius: layout.radius,
            hover: self.hover,
            press: self.press,
            icon_scale: 1.0,
        };
        painter.alpha = progress;
        painter.offset = (0.0, slide);
        draw_button_glass(painter, look, s);
        let dragging = self.pressed.is_some_and(|press| press.dragging);
        let hovered = (over && !dragging)
            .then(|| self.index_at(cursor.0))
            .filter(|&index| index != index_of(self.selected));
        let marks = Marks {
            rect,
            radius: layout.radius,
            knob,
            hover: self.hover,
            hovered,
            s,
        };
        marks.draw(painter, env, Layer::Full);
        if let Some(crisp) = crisp {
            crisp.alpha = progress * CRISP_ALPHA;
            crisp.offset = (0.0, slide);
            marks.draw(crisp, env, Layer::Crisp);
        }
    }

    /// Where the knob slides: the row inset by [`INSET`].
    fn track(&self) -> Rect {
        track(self.rect, self.s)
    }

    /// The version slot under `x` (the nearest one for an `x` beside the track).
    fn index_at(&self, x: f32) -> usize {
        let track = self.track();
        let slot_w = track.w / COUNT as f32;
        if slot_w <= 0.0 {
            return 0;
        }
        (((x - track.x) / slot_w).floor().max(0.0) as usize).min(COUNT - 1)
    }

    /// The knob's position now (where it rests before the first frame).
    fn knob_position(&self) -> f32 {
        self.knob.unwrap_or_else(|| index_of(self.selected) as f32)
    }

    /// Tests: the knob's position in the last frame (`0` on 1.7.10, `1` on 1.8.9).
    #[cfg(test)]
    pub(super) fn knob(&self) -> f32 {
        self.knob_position()
    }

    /// Whether `press`, released at `x`, dragged the knob.
    fn drags(&self, press: &Press, x: f32) -> bool {
        press.on_knob && (press.dragging || (x - press.x).abs() > DRAG_SLOP * self.s)
    }

    /// The knob's position while `press` drags it and the pointer is at `x`.
    fn drag_position(&self, press: &Press, x: f32) -> f32 {
        let slot_w = (self.track().w / COUNT as f32).max(1.0);
        (press.knob + (x - press.x) / slot_w).clamp(0.0, (COUNT - 1) as f32)
    }
}

/// The slider's marks for one frame: drawn under the VHS finish and once more, crisp, over it.
struct Marks {
    rect: Rect,
    radius: f32,
    /// The knob's position.
    knob: f32,
    hover: f32,
    /// A version under the pointer other than the selected one: it lights up faintly.
    hovered: Option<usize>,
    s: f32,
}

impl Marks {
    fn draw(&self, painter: &mut Painter<'_>, env: &mut AeroTitleEnv<'_>, layer: Layer) {
        let (rect, radius, s) = (self.rect, self.radius, self.s);
        // The buttons' top highlight and light border.
        painter.line(
            rect.x + radius * 0.8,
            rect.y + 1.5,
            rect.right() - radius * 0.8,
            rect.y + 1.5,
            1.0,
            white(0.26 + 0.14 * self.hover),
        );
        painter.round_rect_stroke(rect, radius, 1.0, 0.5, white(0.40 + 0.30 * self.hover));
        let track = track(rect, s);
        let inner_radius = (radius - INSET * s).max(s);
        if layer == Layer::Full {
            if let Some(index) = self.hovered {
                let fill = Fill::Solid(white(0.10 * self.hover));
                painter.round_rect(slot(track, index as f32), inner_radius, fill);
            }
            // The knob: the mod menu switch's bright knob, stretched to a pill behind a label.
            let knob = slot(track, self.knob);
            painter.round_rect_shadow(
                knob,
                inner_radius,
                [6.0 * s, 2.0 * s, 6.0 * s, 8.0 * s],
                black(0.24),
            );
            let fill = Fill::css_angle(knob, 180.0, white(0.96), white(0.80));
            painter.round_rect(knob, inner_radius, fill);
        }
        // The versions: white on the glass, dark on the knob (as on the update card's main
        // button). A label the knob is sliding over is cut at the knob's edge.
        let style = TextStyle::new(LABEL_SIZE * s).weight(Weight::Semibold);
        let (_, line_h) = env.text.metrics(style);
        let knob = slot(track, self.knob);
        // Room for the white labels' halo.
        let margin = 4.0 * s;
        let glass = [
            Rect::new(
                rect.x - margin,
                rect.y - margin,
                knob.x - rect.x + margin,
                rect.h + 2.0 * margin,
            ),
            Rect::new(
                knob.right(),
                rect.y - margin,
                rect.right() + margin - knob.right(),
                rect.h + 2.0 * margin,
            ),
        ];
        let saved = painter.clip();
        for (index, version) in GameVersion::ALL.into_iter().enumerate() {
            let label = version.name();
            let place = slot(track, index as f32);
            let width = env.text.measure(label, style, env.mc_font);
            let x = (place.x + (place.w - width) * 0.5).round();
            let y = (place.y + (place.h - line_h) * 0.5).round();
            let bounds = Rect::new(
                x - margin,
                y - margin,
                width + 2.0 * margin,
                line_h + 2.0 * margin,
            );
            for part in glass.into_iter().filter(|part| overlaps(*part, bounds)) {
                painter.set_clip(Some(within(saved, part)));
                let color = white(0.95);
                match layer {
                    Layer::Full => haloed(painter, env, label, x, y, style, color, (s, 0.18)),
                    Layer::Crisp => {
                        env.text
                            .draw(painter, env.mc_font, label, x, y, style, color);
                    }
                }
            }
            if overlaps(knob, bounds) {
                painter.set_clip(Some(within(saved, knob)));
                env.text
                    .draw(painter, env.mc_font, label, x, y, style, black(0.84));
            }
        }
        painter.set_clip(saved);
    }
}

/// Whether `a` and `b` share any area.
fn overlaps(a: Rect, b: Rect) -> bool {
    let shared = a.intersect(&b);
    shared.w > 0.0 && shared.h > 0.0
}

/// `rect` inside the clip already in force, if any.
fn within(clip: Option<Rect>, rect: Rect) -> Rect {
    clip.map_or(rect, |clip| clip.intersect(&rect))
}

/// Where the knob slides in a `row` drawn at scale `s`.
fn track(row: Rect, s: f32) -> Rect {
    row.inset(INSET * s)
}

/// The slot of `track` at `position` (a version's index; the knob's may be fractional).
fn slot(track: Rect, position: f32) -> Rect {
    let w = track.w / COUNT as f32;
    Rect::new(track.x + w * position, track.y, w, track.h)
}

/// `version`'s place in [`GameVersion::ALL`].
fn index_of(version: GameVersion) -> usize {
    GameVersion::ALL
        .iter()
        .position(|candidate| *candidate == version)
        .unwrap_or(0)
}

/// Move `value` the fraction `k` of the way to `target`, settling on it when close.
fn ease(value: &mut f32, target: f32, k: f32) {
    *value += (target - *value) * k;
    if (target - *value).abs() < 0.002 {
        *value = target;
    }
}
