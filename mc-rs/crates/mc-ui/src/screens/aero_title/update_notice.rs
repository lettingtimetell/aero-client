//! The update card on the Aero title screen: "Aero Client 0.2.0 is available" with Update,
//! Release notes and Later; the download's progress; "Restart to update" once it is ready.
//!
//! It sits at the bottom left, on the button column's left edge, just above the version line,
//! and only while the client has something to say about an update ([`NoticeView`]); the account
//! corner is at the top right. The card is one Aero glass pane like the buttons (same tint,
//! sheen, border and halo'd white text), pops in after them, and its small buttons ease their
//! hover and press the way the big ones do. Clicks: a press on the card is the card's; releasing
//! over the button the press landed on activates it ([`UpdateNotice::mouse_released`]).
//!
//! Nothing is allocated per frame: the client replaces the view only when the updater's status
//! changes ([`UpdateNotice::set_view`]), and the progress text is rewritten in place only when
//! the whole percent changes.

use std::fmt::Write as _;
use std::time::Instant;

use super::layout::Layout;
use super::{draw_button_glass, haloed, ms_since, AeroButton, AeroTitleEnv, ButtonLook, Layer};
use crate::screens::modmenu::anim;
use crate::screens::modmenu::glass::{GlassFrame, GlassRect};
use crate::screens::modmenu::paint::{black, white, Fill, Painter, Rect, Rgba};
use crate::screens::modmenu::text::{TextStyle, Weight};

/// The most buttons a card has.
pub const MAX_BUTTONS: usize = 3;

/// The card's least width, in design pixels (1080p): the button column's, so the left column
/// lines up. It grows with its text up to [`MAX_CARD_W`].
pub const MIN_CARD_W: f32 = super::layout::BUTTON_W;
/// The card's greatest width, in design pixels.
pub const MAX_CARD_W: f32 = 560.0;
/// Inner padding.
const PAD: f32 = 16.0;
/// From the card's bottom to the version line's top.
const FOOTER_GAP: f32 = 14.0;
/// Small button height.
const BUTTON_H: f32 = 32.0;
/// Space between small buttons.
const BUTTON_GAP: f32 = 8.0;
/// Text padding inside a small button.
const BUTTON_PAD_X: f32 = 14.0;
/// Progress bar height.
const BAR_H: f32 = 6.0;
const HIGHLIGHT_RATE: f32 = 20.0;

/// One of the card's buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeButton {
    /// Download the available release.
    Update,
    /// Open the release page in the browser.
    ReleaseNotes,
    /// Don't offer this version again.
    Later,
    /// Quit and let the helper swap the new version in.
    Restart,
    /// Stop the download.
    Cancel,
    /// Download again after a failure.
    TryAgain,
    /// Close a message.
    Ok,
}

impl NoticeButton {
    /// The label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            NoticeButton::Update => "Update",
            NoticeButton::ReleaseNotes => "Release notes",
            NoticeButton::Later => "Later",
            NoticeButton::Restart => "Restart to update",
            NoticeButton::Cancel => "Cancel",
            NoticeButton::TryAgain => "Try again",
            NoticeButton::Ok => "OK",
        }
    }
}

/// What the card says. The client builds one when the updater's status changes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoticeView {
    /// The headline, "Aero Client 0.2.0 is available".
    pub title: String,
    /// A second line, or empty.
    pub detail: String,
    /// Draw a progress bar (the value comes from [`UpdateNotice::set_progress`]).
    pub progress: bool,
    /// The buttons, left to right; the first is drawn as the main one. At most [`MAX_BUTTONS`].
    pub buttons: Vec<NoticeButton>,
}

/// The card's state on one title screen.
#[derive(Debug, Default)]
pub struct UpdateNotice {
    view: Option<NoticeView>,
    /// The client's generation of `view` (`0`: never set).
    generation: u64,
    /// When the current view first appeared (its own pop-in).
    shown_at: Option<Instant>,
    /// Download progress, `0..=1`.
    progress: f32,
    /// The whole percent `progress_text` shows.
    percent: Option<u32>,
    progress_text: String,
    card: Rect,
    visible: bool,
    rects: [Rect; MAX_BUTTONS],
    hover: [f32; MAX_BUTTONS],
    press: [f32; MAX_BUTTONS],
    pressed: Option<usize>,
}

impl UpdateNotice {
    /// The generation of the view last set (`0` before any): the client compares its own and
    /// calls [`UpdateNotice::set_view`] only when they differ.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Show `view` (or nothing), as of the client's `generation`. A card that was hidden pops
    /// in; one already up keeps its place while its text changes (Update -> the download's
    /// progress -> "Restart to update").
    pub fn set_view(&mut self, generation: u64, view: Option<NoticeView>) {
        self.generation = generation;
        if view.is_none() {
            self.shown_at = None;
        }
        self.view = view;
        self.pressed = None;
        self.percent = None;
    }

    /// The download's progress, `done` of `total` bytes (a store per frame; the text changes with
    /// the whole percent).
    pub fn set_progress(&mut self, done: u64, total: u64) {
        let fraction = if total == 0 {
            0.0
        } else {
            (done as f64 / total as f64).clamp(0.0, 1.0)
        };
        self.progress = fraction as f32;
        let percent = (fraction * 100.0).floor() as u32;
        if self.percent != Some(percent) {
            self.percent = Some(percent);
            self.progress_text.clear();
            let _ = write!(self.progress_text, "{percent}%");
        }
    }

    /// Whether the card is on screen at `(x, y)`.
    #[must_use]
    pub fn takes(&self, x: f32, y: f32) -> bool {
        self.visible && self.card.contains(x, y)
    }

    /// Whether a press on the card is waiting for its release.
    #[must_use]
    pub fn has_press(&self) -> bool {
        self.pressed.is_some()
    }

    /// The button under `(x, y)`.
    #[must_use]
    pub fn hit(&self, x: f32, y: f32) -> Option<NoticeButton> {
        self.hit_index(x, y)
            .and_then(|i| self.view.as_ref()?.buttons.get(i).copied())
    }

    fn hit_index(&self, x: f32, y: f32) -> Option<usize> {
        if !self.visible {
            return None;
        }
        let count = self
            .view
            .as_ref()
            .map_or(0, |view| view.buttons.len().min(MAX_BUTTONS));
        (0..count).find(|&i| self.rects[i].contains(x, y))
    }

    /// A mouse button went down at `(x, y)`. Returns whether the card took it.
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8) -> bool {
        if !self.takes(x, y) {
            return false;
        }
        self.pressed = (button == 0).then(|| self.hit_index(x, y)).flatten();
        true
    }

    /// A mouse button went up at `(x, y)`: the button to activate, when the press landed on it.
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8) -> Option<NoticeButton> {
        if button != 0 {
            return None;
        }
        let pressed = self.pressed.take()?;
        if self.hit_index(x, y) != Some(pressed) {
            return None;
        }
        self.view.as_ref()?.buttons.get(pressed).copied()
    }

    /// The card's rectangle in the last frame.
    #[must_use]
    pub fn card(&self) -> Option<Rect> {
        self.visible.then_some(self.card)
    }

    /// The small buttons' rectangles in the last frame.
    #[must_use]
    pub fn button_rects(&self) -> &[Rect] {
        let count = self
            .view
            .as_ref()
            .map_or(0, |view| view.buttons.len().min(MAX_BUTTONS));
        &self.rects[..if self.visible { count } else { 0 }]
    }

    /// Draw this frame's card (called by `AeroTitleScreen::build_layers` after the footer).
    /// `clock` is the screen's `(elapsed ms since its pop-in, seconds since the last frame,
    /// cursor)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        &mut self,
        painter: &mut Painter<'_>,
        crisp: Option<&mut Painter<'_>>,
        env: &mut AeroTitleEnv<'_>,
        layout: &Layout,
        glass: &mut GlassFrame,
        (elapsed, dt, cursor): (f32, f32, (f32, f32)),
    ) {
        let Some(view) = self.view.as_ref() else {
            self.visible = false;
            return;
        };
        let now = env.now;
        let shown_at = *self.shown_at.get_or_insert(now);
        // After the buttons' pop-in, and with its own when it appears later.
        let progress = anim::window_open_progress(elapsed, AeroButton::COUNT + 1)
            .min(anim::window_open_progress(ms_since(shown_at, now), 0));
        self.visible = progress > 0.0;
        if !self.visible {
            return;
        }
        let s = layout.s;
        let title_style = TextStyle::new(17.0 * s).weight(Weight::Semibold);
        let detail_style = TextStyle::new(14.0 * s);
        let label_style = TextStyle::new(14.0 * s).weight(Weight::Semibold);
        let (_, title_h) = env.text.metrics(title_style);
        let (_, detail_h) = env.text.metrics(detail_style);
        let (_, label_h) = env.text.metrics(label_style);
        let (_, footer_h) = env
            .text
            .metrics(TextStyle::new(14.0 * s).weight(Weight::Semibold));
        let has_detail = !view.detail.is_empty() || view.progress;
        let buttons = view.buttons.len().min(MAX_BUTTONS);

        // ---- layout: bottom left, above the version line ----
        let mut height = PAD * s + title_h;
        if has_detail {
            height += 4.0 * s + detail_h;
        }
        if view.progress {
            height += 8.0 * s + BAR_H * s;
        }
        if buttons > 0 {
            height += 12.0 * s + BUTTON_H * s;
        }
        height += PAD * s;
        // As wide as its widest line (the buttons' row included), within the bounds.
        let mut widths = [0.0_f32; MAX_BUTTONS];
        for (width, button) in widths.iter_mut().zip(&view.buttons) {
            *width = (env.text.measure(button.label(), label_style, env.mc_font)
                + 2.0 * BUTTON_PAD_X * s)
                .max(72.0 * s)
                .round();
        }
        let row = widths[..buttons].iter().sum::<f32>()
            + BUTTON_GAP * s * buttons.saturating_sub(1) as f32;
        let title_w = env.text.measure(&view.title, title_style, env.mc_font);
        let detail_w = env.text.measure(&view.detail, detail_style, env.mc_font);
        let content = row.max(title_w).max(detail_w) + 2.0 * PAD * s;
        let card_w = content.clamp(MIN_CARD_W * s, MAX_CARD_W * s).round();
        let bottom = layout.footer_bottom - footer_h - FOOTER_GAP * s;
        let card = Rect::new(
            layout.left,
            (bottom - height).round(),
            card_w,
            height.round(),
        );
        self.card = card;
        let mut x = card.x + PAD * s;
        let button_y = card.bottom() - PAD * s - BUTTON_H * s;
        for (i, w) in widths[..buttons].iter().enumerate() {
            self.rects[i] = Rect::new(x.round(), button_y.round(), *w, (BUTTON_H * s).round());
            x += w + BUTTON_GAP * s;
        }

        // ---- highlights ease towards the pointer ----
        let k = anim::approach(dt, HIGHLIGHT_RATE);
        for i in 0..MAX_BUTTONS {
            let over = i < buttons && self.rects[i].contains(cursor.0, cursor.1);
            let targets = [
                (&mut self.hover[i], if over { 1.0 } else { 0.0 }),
                (
                    &mut self.press[i],
                    if over && self.pressed == Some(i) {
                        1.0
                    } else {
                        0.0
                    },
                ),
            ];
            for (value, target) in targets {
                *value += (target - *value) * k;
                if (target - *value).abs() < 0.002 {
                    *value = target;
                }
            }
        }

        // ---- the pane ----
        let slide = (1.0 - progress) * anim::SLIDE_DIP * s;
        glass.rects.push(GlassRect {
            rect: [card.x, card.y + slide, card.w, card.h],
            radius: layout.radius,
            alpha: progress,
        });
        painter.alpha = progress;
        painter.offset = (0.0, slide);
        let look = ButtonLook {
            rect: card,
            radius: layout.radius,
            hover: 0.0,
            press: 0.0,
            icon_scale: 1.0,
        };
        draw_button_glass(painter, look, s);
        painter.line(
            card.x + layout.radius * 0.8,
            card.y + 1.5,
            card.right() - layout.radius * 0.8,
            card.y + 1.5,
            1.0,
            white(0.26),
        );
        painter.round_rect_stroke(card, layout.radius, 1.0, 0.5, white(0.40));
        let progress_text = self.progress_text.as_str();
        let marks = Marks {
            view,
            card,
            rects: &self.rects,
            hover: &self.hover,
            press: &self.press,
            bar: self.progress,
            progress_text,
            s,
            styles: (title_style, detail_style, label_style),
            heights: (title_h, detail_h, label_h),
        };
        marks.draw(painter, env, Layer::Full);
        if let Some(crisp) = crisp {
            crisp.alpha = progress * super::CRISP_ALPHA;
            crisp.offset = (0.0, slide);
            marks.draw(crisp, env, Layer::Crisp);
        }
    }
}

/// The card's text, bar and buttons: drawn once under the VHS finish and once, crisp, over it.
struct Marks<'a> {
    view: &'a NoticeView,
    card: Rect,
    rects: &'a [Rect; MAX_BUTTONS],
    hover: &'a [f32; MAX_BUTTONS],
    press: &'a [f32; MAX_BUTTONS],
    bar: f32,
    progress_text: &'a str,
    s: f32,
    styles: (TextStyle, TextStyle, TextStyle),
    heights: (f32, f32, f32),
}

impl Marks<'_> {
    fn text(
        painter: &mut Painter<'_>,
        env: &mut AeroTitleEnv<'_>,
        layer: Layer,
        (text, x, y): (&str, f32, f32),
        style: TextStyle,
        (color, s): (Rgba, f32),
    ) {
        match layer {
            Layer::Full => haloed(painter, env, text, x, y, style, color, (s, 0.18)),
            Layer::Crisp => {
                env.text
                    .draw(painter, env.mc_font, text, x, y, style, color);
            }
        }
    }

    fn draw(&self, painter: &mut Painter<'_>, env: &mut AeroTitleEnv<'_>, layer: Layer) {
        let s = self.s;
        let (title_style, detail_style, label_style) = self.styles;
        let (title_h, detail_h, label_h) = self.heights;
        let x = self.card.x + PAD * s;
        let mut y = self.card.y + PAD * s;
        let title = (self.view.title.as_str(), x, y);
        Self::text(painter, env, layer, title, title_style, (white(1.0), s));
        y += title_h;
        if !self.view.detail.is_empty() || self.view.progress {
            y += 4.0 * s;
            let detail = if self.view.detail.is_empty() {
                self.progress_text
            } else {
                self.view.detail.as_str()
            };
            Self::text(
                painter,
                env,
                layer,
                (detail, x, y),
                detail_style,
                (white(0.86), s),
            );
            if self.view.progress && !self.view.detail.is_empty() {
                // The percentage at the right end of the detail line.
                let width = env
                    .text
                    .measure(self.progress_text, detail_style, env.mc_font);
                let right = self.card.right() - PAD * s - width;
                let percent = (self.progress_text, right, y);
                Self::text(painter, env, layer, percent, detail_style, (white(0.86), s));
            }
            y += detail_h;
        }
        if self.view.progress && layer == Layer::Full {
            y += 8.0 * s;
            let track = Rect::new(x, y, self.card.w - 2.0 * PAD * s, BAR_H * s);
            painter.round_rect(track, BAR_H * s * 0.5, Fill::Solid(white(0.18)));
            let fill = Rect::new(
                track.x,
                track.y,
                (track.w * self.bar).max(BAR_H * s),
                track.h,
            );
            painter.round_rect(fill, BAR_H * s * 0.5, Fill::Solid(white(0.92)));
        }
        for (i, button) in self.view.buttons.iter().take(MAX_BUTTONS).enumerate() {
            let rect = self.rects[i].inset(self.press[i] * 1.0 * s);
            let (hover, press) = (self.hover[i], self.press[i]);
            let main = i == 0;
            let radius = rect.h * 0.5;
            if layer == Layer::Full {
                let fill = if main {
                    white(0.86 + 0.10 * hover - 0.12 * press)
                } else {
                    white(0.12 + 0.10 * hover + 0.06 * press)
                };
                painter.round_rect(rect, radius, Fill::Solid(fill));
                if !main {
                    painter.round_rect_stroke(rect, radius, 1.0, 0.5, white(0.42 + 0.28 * hover));
                }
            }
            let label = button.label();
            let width = env.text.measure(label, label_style, env.mc_font);
            let tx = (rect.x + (rect.w - width) * 0.5).round();
            let ty = (rect.y + (rect.h - label_h) * 0.5).round();
            if main {
                // Dark text on the light main button: no halo.
                env.text.draw(
                    painter,
                    env.mc_font,
                    label,
                    tx,
                    ty,
                    label_style,
                    black(0.86),
                );
            } else {
                Self::text(
                    painter,
                    env,
                    layer,
                    (label, tx, ty),
                    label_style,
                    (white(1.0), s),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(buttons: &[NoticeButton]) -> NoticeView {
        NoticeView {
            title: "Aero Client 0.2.0 is available".to_owned(),
            detail: String::new(),
            progress: false,
            buttons: buttons.to_vec(),
        }
    }

    #[test]
    fn progress_text_changes_only_with_the_whole_percent() {
        let mut notice = UpdateNotice::default();
        notice.set_progress(0, 0);
        assert_eq!(notice.progress_text, "0%");
        notice.set_progress(421, 1000);
        assert_eq!(notice.progress_text, "42%");
        let capacity = notice.progress_text.capacity();
        notice.set_progress(429, 1000);
        assert_eq!(notice.progress_text, "42%");
        notice.set_progress(1000, 1000);
        assert_eq!(notice.progress_text, "100%");
        assert_eq!(
            notice.progress_text.capacity(),
            capacity,
            "rewritten in place"
        );
        notice.set_progress(5000, 1000);
        assert_eq!(notice.progress_text, "100%", "clamped");
    }

    #[test]
    fn a_click_is_a_press_and_release_on_the_same_button() {
        let mut notice = UpdateNotice::default();
        notice.set_view(1, Some(view(&[NoticeButton::Update, NoticeButton::Later])));
        assert_eq!(notice.generation(), 1);
        // Laid out by hand (a frame would do it).
        notice.visible = true;
        notice.card = Rect::new(0.0, 0.0, 400.0, 100.0);
        notice.rects[0] = Rect::new(10.0, 60.0, 80.0, 30.0);
        notice.rects[1] = Rect::new(100.0, 60.0, 80.0, 30.0);
        assert!(notice.takes(5.0, 5.0));
        assert!(!notice.takes(500.0, 5.0));
        assert_eq!(notice.hit(20.0, 70.0), Some(NoticeButton::Update));
        assert_eq!(notice.hit(110.0, 70.0), Some(NoticeButton::Later));
        assert_eq!(notice.hit(300.0, 70.0), None);

        assert!(notice.mouse_pressed(20.0, 70.0, 0));
        assert!(notice.has_press());
        assert_eq!(
            notice.mouse_released(25.0, 75.0, 0),
            Some(NoticeButton::Update)
        );
        assert!(!notice.has_press());
        // Pressed on one, released on the other: nothing.
        assert!(notice.mouse_pressed(20.0, 70.0, 0));
        assert_eq!(notice.mouse_released(110.0, 70.0, 0), None);
        // The right button presses nothing.
        assert!(notice.mouse_pressed(20.0, 70.0, 1));
        assert!(!notice.has_press());
        // A press on the card's background is taken, activates nothing.
        assert!(notice.mouse_pressed(300.0, 20.0, 0));
        assert_eq!(notice.mouse_released(300.0, 20.0, 0), None);
        // A new view drops a pending press; no view, nothing to hit.
        assert!(notice.mouse_pressed(20.0, 70.0, 0));
        notice.set_view(2, None);
        assert!(!notice.has_press());
        notice.visible = false;
        assert!(!notice.takes(5.0, 5.0));
        assert_eq!(notice.hit(20.0, 70.0), None);
    }

    #[test]
    fn labels_are_short_and_sentence_case() {
        for button in [
            NoticeButton::Update,
            NoticeButton::ReleaseNotes,
            NoticeButton::Later,
            NoticeButton::Restart,
            NoticeButton::Cancel,
            NoticeButton::TryAgain,
            NoticeButton::Ok,
        ] {
            let label = button.label();
            assert!(!label.is_empty() && label.len() <= 18, "{label}");
            assert!(
                label.chars().next().is_some_and(char::is_uppercase),
                "{label}"
            );
        }
    }
}
