//! The replay screens' Aero look, built from the mod menu's pieces ([`Painter`], [`MenuText`],
//! the glass) so the browser, the viewer's controls and the export window read as the same
//! family as the Right Shift menu and the title screen: translucent dark glass, a light 1 px
//! border, the soft text shadow, white type, and one accent (Aero blue) for the main action and
//! the timeline selection.
//!
//! Everything here draws in **physical pixels**; sizes are design pixels for a 1920x1080 window
//! times the scale `s` ([`scale`]).

use std::time::Instant;

use super::icons::{self, Icon};
use crate::font::FontRenderer;
use crate::screens::modmenu::paint::{black, css, white, Fill, Painter, Rect, Rgba};
use crate::screens::modmenu::text::{MenuText, TextStyle, Weight};
use crate::widget::UiTexture;

/// The accent: the main action (Watch, Export), the timeline selection, focus rings.
pub const ACCENT: Rgba = css(86, 170, 255, 1.0);
/// A brighter accent for text and thin marks on dark glass.
pub const ACCENT_TEXT: Rgba = css(150, 205, 255, 1.0);
/// Destructive actions once armed (Delete?).
pub const DANGER: Rgba = css(255, 96, 96, 1.0);
/// A finished export.
pub const SUCCESS: Rgba = css(110, 220, 140, 1.0);

/// How long the pointer rests on a control before its tooltip shows.
pub const TOOLTIP_DELAY_MS: f32 = 450.0;

/// Physical pixels per design pixel. The height decides (1080p is 1), but a small window keeps
/// the small text readable: never under 0.8 (720p) while the layouts still fit (the viewer's
/// control row needs about 1440 design pixels across, the export window about 800 down).
#[must_use]
pub fn scale(screen: (f32, f32)) -> f32 {
    let natural = screen.1 / 1080.0;
    let fit = (screen.0 / 1440.0).min(screen.1 / 800.0);
    let s = natural.max(0.8_f32.min(fit));
    if s.is_finite() {
        s.clamp(0.45, 4.0)
    } else {
        1.0
    }
}

/// Milliseconds from `since` to `now` (zero when `now` is earlier).
#[must_use]
pub fn ms_since(since: Instant, now: Instant) -> f32 {
    now.saturating_duration_since(since).as_secs_f32() * 1000.0
}

/// Mix two colours.
#[must_use]
pub fn mix(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// `color` with its alpha multiplied by `a`.
#[must_use]
pub fn fade(color: Rgba, a: f32) -> Rgba {
    [color[0], color[1], color[2], color[3] * a]
}

/// The fonts the screens draw with.
pub struct Fonts<'a> {
    /// Segoe UI (or the Minecraft font).
    pub text: &'a mut MenuText,
    /// The Minecraft font (fallback and measuring).
    pub mc: &'a mut FontRenderer,
}

impl Fonts<'_> {
    /// Text with the menu's soft shadow (`text-shadow: 0 1px 1px rgba(0,0,0,.38)`, a touch
    /// stronger: the backdrop can be a bright sky). Returns the advance.
    #[allow(clippy::too_many_arguments)]
    pub fn shadowed(
        &mut self,
        painter: &mut Painter<'_>,
        text: &str,
        x: f32,
        y: f32,
        style: TextStyle,
        color: Rgba,
        s: f32,
    ) -> f32 {
        self.text.draw(
            painter,
            self.mc,
            text,
            x,
            y + (1.0 * s).max(1.0),
            style,
            black(0.45 * color[3]),
        );
        self.text.draw(painter, self.mc, text, x, y, style, color)
    }

    /// [`Fonts::shadowed`] centred vertically on `cy`.
    #[allow(clippy::too_many_arguments)]
    pub fn shadowed_mid(
        &mut self,
        painter: &mut Painter<'_>,
        text: &str,
        x: f32,
        cy: f32,
        style: TextStyle,
        color: Rgba,
        s: f32,
    ) -> f32 {
        let (_, line_h) = self.text.metrics(style);
        self.shadowed(painter, text, x, cy - line_h * 0.5, style, color, s)
    }

    /// Plain text (no shadow) centred vertically on `cy`.
    pub fn plain_mid(
        &mut self,
        painter: &mut Painter<'_>,
        text: &str,
        x: f32,
        cy: f32,
        style: TextStyle,
        color: Rgba,
    ) -> f32 {
        let (_, line_h) = self.text.metrics(style);
        self.text
            .draw(painter, self.mc, text, x, cy - line_h * 0.5, style, color)
    }

    /// The advance width of `text`.
    #[must_use]
    pub fn width(&self, text: &str, style: TextStyle) -> f32 {
        self.text.measure(text, style, self.mc)
    }

    /// The line height of `style`.
    #[must_use]
    pub fn line_h(&self, style: TextStyle) -> f32 {
        self.text.metrics(style).1
    }

    /// `text` cut to fit `max_w` with a trailing ellipsis (whole characters).
    #[must_use]
    pub fn ellipsize(&self, text: &str, style: TextStyle, max_w: f32) -> String {
        if self.width(text, style) <= max_w {
            return text.to_owned();
        }
        let ellipsis = "\u{2026}";
        let room = (max_w - self.width(ellipsis, style)).max(0.0);
        let mut end = 0;
        for (index, c) in text.char_indices() {
            let next = index + c.len_utf8();
            if self.width(&text[..next], style) > room {
                break;
            }
            end = next;
        }
        format!("{}{ellipsis}", text[..end].trim_end())
    }
}

/// The text styles, at scale `s`.
pub mod styles {
    use super::{TextStyle, Weight};

    /// A window title.
    #[must_use]
    pub fn title(s: f32) -> TextStyle {
        TextStyle::new(17.0 * s).weight(Weight::Semibold)
    }

    /// The line under a window title.
    #[must_use]
    pub fn subtitle(s: f32) -> TextStyle {
        TextStyle::new(12.0 * s)
    }

    /// Body text.
    #[must_use]
    pub fn body(s: f32) -> TextStyle {
        TextStyle::new(14.0 * s)
    }

    /// Small body text (second lines).
    #[must_use]
    pub fn small(s: f32) -> TextStyle {
        TextStyle::new(12.0 * s)
    }

    /// A button label.
    #[must_use]
    pub fn button(s: f32) -> TextStyle {
        TextStyle::new(13.0 * s).weight(Weight::Semibold)
    }

    /// A section label: small caps.
    #[must_use]
    pub fn section(s: f32) -> TextStyle {
        TextStyle::new(10.5 * s).weight(Weight::Bold).caps(0.14)
    }

    /// Numbers that change every frame (the time readout).
    #[must_use]
    pub fn readout(s: f32) -> TextStyle {
        TextStyle::new(14.0 * s).weight(Weight::Semibold)
    }
}

/// The chrome of an Aero window (the mod menu's `draw_window` look): drop shadow, the dark
/// translucent body, the 135-degree sheen, an optional title bar of `title_h` with its divider,
/// the top highlight and the light border. The glass itself is a [`GlassRect`] the caller adds.
///
/// [`GlassRect`]: crate::screens::modmenu::glass::GlassRect
pub fn window(painter: &mut Painter<'_>, rect: Rect, radius: f32, title_h: f32, s: f32) {
    painter.round_rect_shadow(
        rect,
        radius,
        [26.0 * s, 6.0 * s, 26.0 * s, 52.0 * s],
        black(0.32),
    );
    // A darker body than the menu's: the replay windows sit on bright scenes (the greenhouse,
    // a noon sky) and carry long lines of small text.
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(rect, 180.0, black(0.30), black(0.40)),
    );
    let corner = Rect::new(rect.x, rect.y, rect.w * 0.64, rect.h * 0.64);
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(corner, 135.0, white(0.06), white(0.0)),
    );
    if title_h > 0.0 {
        let title = Rect::new(rect.x, rect.y, rect.w, title_h);
        let saved = painter.clip();
        painter.set_clip(Some(saved.map_or(title, |c| c.intersect(&title))));
        painter.round_rect(
            Rect::new(rect.x, rect.y, rect.w, title_h + radius),
            radius,
            Fill::css_angle(title, 180.0, white(0.06), white(0.01)),
        );
        painter.set_clip(saved);
        painter.rect(
            Rect::new(rect.x + 1.0, (rect.y + title_h).round(), rect.w - 2.0, 1.0),
            white(0.10),
        );
    }
    painter.line(
        rect.x + radius * 0.8,
        rect.y + 1.5,
        rect.right() - radius * 0.8,
        rect.y + 1.5,
        1.0,
        white(0.18),
    );
    painter.round_rect_stroke(rect, radius, 1.0, 0.5, white(0.40));
}

/// A panel over the live game (the viewer's bar, the export window): the window's chrome with a
/// darker body, because what is behind it moves and is as busy as a game gets (the hotbar sits
/// right under the controls in the player's view).
pub fn panel(painter: &mut Painter<'_>, rect: Rect, radius: f32, title_h: f32, s: f32) {
    painter.round_rect_shadow(
        rect,
        radius,
        [22.0 * s, 6.0 * s, 22.0 * s, 40.0 * s],
        black(0.34),
    );
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(rect, 180.0, css(14, 18, 26, 0.50), css(8, 10, 16, 0.62)),
    );
    let corner = Rect::new(rect.x, rect.y, rect.w * 0.5, rect.h * 0.9);
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(corner, 135.0, white(0.07), white(0.0)),
    );
    if title_h > 0.0 {
        painter.rect(
            Rect::new(rect.x + 1.0, (rect.y + title_h).round(), rect.w - 2.0, 1.0),
            white(0.10),
        );
    }
    painter.line(
        rect.x + radius * 0.8,
        rect.y + 1.5,
        rect.right() - radius * 0.8,
        rect.y + 1.5,
        1.0,
        white(0.20),
    );
    painter.round_rect_stroke(rect, radius, 1.0, 0.5, white(0.34));
}

/// How a button looks this frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Look {
    /// The pointer is over it.
    pub hover: bool,
    /// It is held down.
    pub press: bool,
    /// A toggle that is on, or the selected segment.
    pub active: bool,
    /// The main action: accent-tinted.
    pub primary: bool,
    /// An armed destructive action: red.
    pub danger: bool,
    /// Cannot be used now: dimmed, no hover.
    pub disabled: bool,
}

impl Look {
    /// A plain button with this hover and press.
    #[must_use]
    pub fn new(hover: bool, press: bool) -> Look {
        Look {
            hover,
            press,
            ..Look::default()
        }
    }
}

/// A glass pill button body (the menu taskbar's `.task-app`), accent or red when asked.
pub fn pill_body(painter: &mut Painter<'_>, rect: Rect, radius: f32, look: Look, s: f32) {
    let hover = look.hover && !look.disabled;
    painter.round_rect_shadow(
        rect,
        radius,
        [8.0 * s, 3.0 * s, 8.0 * s, 14.0 * s],
        black(if look.disabled { 0.10 } else { 0.24 }),
    );
    let (top, bottom, border) = if look.danger {
        (
            fade(DANGER, if hover { 0.62 } else { 0.48 }),
            fade(css(170, 30, 30, 1.0), 0.55),
            fade(css(255, 170, 170, 1.0), 0.85),
        )
    } else if look.primary {
        (
            fade(ACCENT, if hover { 0.70 } else { 0.55 }),
            fade(css(30, 100, 210, 1.0), if hover { 0.70 } else { 0.58 }),
            fade(css(190, 225, 255, 1.0), if hover { 0.95 } else { 0.75 }),
        )
    } else if look.active {
        (
            white(if hover { 0.34 } else { 0.28 }),
            white(0.10),
            white(0.75),
        )
    } else {
        (
            white(if hover { 0.17 } else { 0.08 }),
            black(0.16),
            white(if hover { 0.55 } else { 0.30 }),
        )
    };
    let (top, bottom) = if look.press {
        (mix(top, black(0.3), 0.5), mix(bottom, black(0.4), 0.5))
    } else {
        (top, bottom)
    };
    painter.round_rect(rect, radius, Fill::css_angle(rect, 180.0, top, bottom));
    painter.line(
        rect.x + radius,
        rect.y + 1.5,
        rect.right() - radius,
        rect.y + 1.5,
        1.0,
        white(if look.primary { 0.35 } else { 0.18 }),
    );
    painter.round_rect_stroke(rect, radius, 1.0, 0.5, border);
}

/// A pill button with an optional icon and a label, both centred as a group.
#[allow(clippy::too_many_arguments)]
pub fn pill(
    painter: &mut Painter<'_>,
    fonts: &mut Fonts<'_>,
    rect: Rect,
    icon: Option<Icon>,
    label: &str,
    look: Look,
    s: f32,
) {
    let radius = (rect.h * 0.5).min(16.0 * s);
    pill_body(painter, rect, radius, look, s);
    let style = styles::button(s);
    let text_w = if label.is_empty() {
        0.0
    } else {
        fonts.width(label, style)
    };
    let icon_w = if icon.is_some() { 16.0 * s } else { 0.0 };
    let gap = if icon.is_some() && !label.is_empty() {
        7.0 * s
    } else {
        0.0
    };
    let total = icon_w + gap + text_w;
    let mut x = rect.x + (rect.w - total) * 0.5;
    let cy = rect.y + rect.h * 0.5;
    let alpha = if look.disabled { 0.38 } else { 1.0 };
    if let Some(icon) = icon {
        icons::draw(
            painter,
            icon,
            (x + icon_w * 0.5, cy),
            16.0 * s,
            1.6 * s,
            white(0.96 * alpha),
        );
        x += icon_w + gap;
    }
    if !label.is_empty() {
        fonts.shadowed_mid(painter, label, x, cy, style, white(0.97 * alpha), s);
    }
}

/// A round (or square-ish) icon-only glass button.
pub fn icon_button(
    painter: &mut Painter<'_>,
    rect: Rect,
    icon: Icon,
    icon_size: f32,
    look: Look,
    s: f32,
) {
    let radius = rect.h.min(rect.w) * 0.5;
    let hover = look.hover && !look.disabled;
    if look.primary || look.active || look.danger || hover || look.press {
        pill_body(painter, rect, radius, look, s);
    }
    let alpha = if look.disabled { 0.35 } else { 1.0 };
    let (cx, cy) = rect.center();
    icons::draw(
        painter,
        icon,
        (cx, cy),
        icon_size,
        1.7 * s,
        white((if hover { 1.0 } else { 0.92 }) * alpha),
    );
}

/// A value badge (the menu's `.badge`) with optional `‹ ›` arrows; returns its rectangle.
#[allow(clippy::too_many_arguments)]
pub fn badge(
    painter: &mut Painter<'_>,
    fonts: &mut Fonts<'_>,
    text: &str,
    right: f32,
    cy: f32,
    arrows: bool,
    hover: bool,
    s: f32,
) -> Rect {
    let style = TextStyle::new(12.5 * s);
    let tw = fonts.width(text, style);
    let arrow_w = if arrows { 14.0 * s } else { 0.0 };
    let w = tw + 20.0 * s + arrow_w * 2.0;
    let h = 26.0 * s;
    let rect = Rect::new(right - w, cy - h * 0.5, w, h);
    painter.round_rect(
        rect,
        9.0 * s,
        Fill::Solid(black(if hover { 0.36 } else { 0.26 })),
    );
    painter.round_rect_stroke(
        rect,
        9.0 * s,
        1.0,
        0.5,
        white(if hover { 0.45 } else { 0.18 }),
    );
    if arrows {
        let a = white(if hover { 0.85 } else { 0.55 });
        let lx = rect.x + 11.0 * s;
        chevron(painter, (lx, cy), -1.0, 3.2 * s, 1.3 * s, a);
        let rx = rect.right() - 11.0 * s;
        chevron(painter, (rx, cy), 1.0, 3.2 * s, 1.3 * s, a);
    }
    fonts.plain_mid(
        painter,
        text,
        rect.x + 10.0 * s + arrow_w,
        cy,
        style,
        white(0.92),
    );
    rect
}

/// A `‹` (`dir` -1) or `›` (`dir` 1) chevron centred on `c`, `r` from centre to tip.
pub fn chevron(painter: &mut Painter<'_>, c: (f32, f32), dir: f32, r: f32, w: f32, color: Rgba) {
    let (cx, cy) = c;
    painter.line(cx - dir * r * 0.5, cy - r, cx + dir * r * 0.5, cy, w, color);
    painter.line(cx + dir * r * 0.5, cy, cx - dir * r * 0.5, cy + r, w, color);
}

/// A `⌄` chevron (pointing down) centred on `c`.
pub fn chevron_down(painter: &mut Painter<'_>, c: (f32, f32), r: f32, w: f32, color: Rgba) {
    let (cx, cy) = c;
    painter.line(cx - r, cy - r * 0.5, cx, cy + r * 0.5, w, color);
    painter.line(cx, cy + r * 0.5, cx + r, cy - r * 0.5, w, color);
}

/// The Aero switch (the menu's `draw_switch`): `t` 0 off .. 1 on.
pub fn switch(painter: &mut Painter<'_>, rect: Rect, t: f32, hover: bool, disabled: bool) {
    let r = rect.h * 0.5;
    let off_fill = black(if hover { 0.34 } else { 0.28 });
    let on_fill = fade(ACCENT, 0.80);
    let fill = mix(off_fill, on_fill, t);
    let a = if disabled { 0.4 } else { 1.0 };
    painter.round_rect(rect, r, Fill::Solid(fade(fill, a)));
    painter.round_rect_stroke(rect, r, 1.0, 0.5, white((0.30 + 0.45 * t) * a));
    let knob_r = r - 2.5;
    let cx = rect.x + r + (rect.w - 2.0 * r) * t;
    painter.circle(cx, rect.y + r + 0.5, knob_r, black(0.25 * a));
    painter.circle(cx, rect.y + r, knob_r, white((0.80 + 0.2 * t) * a));
}

/// A text field: a recessed dark well, a light border (accent when focused), the text clipped
/// to the well, and a blinking caret at the end while focused. `placeholder` shows when empty.
#[allow(clippy::too_many_arguments)]
pub fn text_field(
    painter: &mut Painter<'_>,
    fonts: &mut Fonts<'_>,
    rect: Rect,
    text: &str,
    placeholder: &str,
    focused: bool,
    caret_on: bool,
    s: f32,
) {
    let radius = 8.0 * s;
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(rect, 180.0, black(0.40), black(0.28)),
    );
    let border = if focused {
        fade(ACCENT, 0.95)
    } else {
        white(0.22)
    };
    painter.round_rect_stroke(rect, radius, if focused { 1.5 } else { 1.0 }, 0.5, border);
    let style = styles::body(s);
    let pad = 10.0 * s;
    let room = (rect.w - 2.0 * pad).max(1.0);
    let saved = painter.clip();
    let clip = Rect::new(rect.x + pad - 1.0, rect.y, room + 2.0, rect.h);
    painter.set_clip(Some(saved.map_or(clip, |c| c.intersect(&clip))));
    let cy = rect.y + rect.h * 0.5;
    if text.is_empty() {
        fonts.plain_mid(painter, placeholder, rect.x + pad, cy, style, white(0.40));
    } else {
        let tw = fonts.width(text, style);
        let x = if tw > room {
            rect.x + pad + room - tw
        } else {
            rect.x + pad
        };
        fonts.plain_mid(painter, text, x, cy, style, white(0.96));
        if focused && caret_on {
            let lh = fonts.line_h(style);
            painter.rect(
                Rect::new(
                    (x + tw + 1.0).min(rect.x + pad + room),
                    cy - lh * 0.45,
                    (1.0 * s).max(1.0),
                    lh * 0.9,
                ),
                white(0.9),
            );
        }
    }
    if text.is_empty() && focused && caret_on {
        let lh = fonts.line_h(style);
        painter.rect(
            Rect::new(rect.x + pad, cy - lh * 0.45, (1.0 * s).max(1.0), lh * 0.9),
            white(0.9),
        );
    }
    painter.set_clip(saved);
}

/// The segments of a [`segmented`] control with `n` segments in `rect`.
#[must_use]
pub fn segment_rects(rect: Rect, n: usize, s: f32) -> Vec<Rect> {
    let inner = rect.inset(3.0 * s);
    let w = inner.w / n.max(1) as f32;
    (0..n)
        .map(|i| Rect::new(inner.x + w * i as f32, inner.y, w, inner.h))
        .collect()
}

/// The segment of a [`segmented`] control under `(x, y)`.
#[must_use]
pub fn segment_at(rect: Rect, n: usize, s: f32, (x, y): (f32, f32)) -> Option<usize> {
    segment_rects(rect, n, s)
        .iter()
        .position(|r| r.contains(x, y))
}

/// A segmented control: `labels` side by side in one track, `selected` lit. Returns each
/// segment's rectangle. `hover` is the hovered segment.
#[allow(clippy::too_many_arguments)]
pub fn segmented(
    painter: &mut Painter<'_>,
    fonts: &mut Fonts<'_>,
    rect: Rect,
    labels: &[(Option<Icon>, &str)],
    selected: usize,
    hover: Option<usize>,
    disabled: &[bool],
    s: f32,
) -> Vec<Rect> {
    let radius = (rect.h * 0.5).min(14.0 * s);
    painter.round_rect(rect, radius, Fill::Solid(black(0.30)));
    painter.round_rect_stroke(rect, radius, 1.0, 0.5, white(0.20));
    let out = segment_rects(rect, labels.len(), s);
    for (i, (icon, label)) in labels.iter().enumerate() {
        let seg = out[i];
        let off = disabled.get(i).copied().unwrap_or(false);
        let on = i == selected;
        if on {
            let r = (seg.h * 0.5).min(11.0 * s);
            painter.round_rect_shadow(seg, r, [4.0 * s; 4], black(0.18));
            painter.round_rect(
                seg,
                r,
                Fill::css_angle(seg, 180.0, white(0.34), white(0.16)),
            );
            painter.round_rect_stroke(seg, r, 1.0, 0.5, white(0.65));
        } else if hover == Some(i) && !off {
            let r = (seg.h * 0.5).min(11.0 * s);
            painter.round_rect(seg, r, Fill::Solid(white(0.10)));
        }
        let style = styles::button(s);
        let alpha = if off {
            0.35
        } else if on {
            1.0
        } else {
            0.78
        };
        let tw = if label.is_empty() {
            0.0
        } else {
            fonts.width(label, style)
        };
        let iw = if icon.is_some() { 15.0 * s } else { 0.0 };
        let gap = if icon.is_some() && !label.is_empty() {
            6.0 * s
        } else {
            0.0
        };
        let mut x = seg.x + (seg.w - iw - gap - tw) * 0.5;
        let cy = seg.y + seg.h * 0.5;
        if let Some(icon) = icon {
            icons::draw(
                painter,
                *icon,
                (x + iw * 0.5, cy),
                15.0 * s,
                1.5 * s,
                white(0.95 * alpha),
            );
            x += iw + gap;
        }
        if !label.is_empty() {
            fonts.shadowed_mid(painter, label, x, cy, style, white(0.97 * alpha), s);
        }
    }
    out
}

/// A progress bar: a dark track and an accent fill to `t`.
pub fn progress_bar(painter: &mut Painter<'_>, rect: Rect, t: f32, color: Rgba) {
    let r = rect.h * 0.5;
    painter.round_rect(rect, r, Fill::Solid(black(0.35)));
    painter.round_rect_stroke(rect, r, 1.0, 0.5, white(0.16));
    let t = t.clamp(0.0, 1.0);
    if t > 0.0 {
        let filled = Rect::new(rect.x, rect.y, (rect.w * t).max(rect.h), rect.h);
        painter.round_rect(
            filled,
            r,
            Fill::css_angle(filled, 180.0, mix(color, white(1.0), 0.25), color),
        );
    }
}

/// A tooltip: a small dark glass label with the control's name and, dimmer, its key, placed
/// above `anchor` (below when there is no room), kept inside the screen.
#[allow(clippy::too_many_arguments)]
pub fn tooltip(
    painter: &mut Painter<'_>,
    fonts: &mut Fonts<'_>,
    anchor: Rect,
    label: &str,
    key: Option<&str>,
    screen: (f32, f32),
    alpha: f32,
    s: f32,
) {
    if alpha <= 0.0 {
        return;
    }
    let style = TextStyle::new(12.5 * s).weight(Weight::Semibold);
    let key_style = TextStyle::new(11.5 * s);
    let pad_x = 10.0 * s;
    let lw = fonts.width(label, style);
    let kw = key.map_or(0.0, |k| fonts.width(k, key_style) + 12.0 * s);
    let gap = if key.is_some() { 8.0 * s } else { 0.0 };
    let w = pad_x * 2.0 + lw + gap + kw;
    let h = 28.0 * s;
    let mut x = anchor.x + (anchor.w - w) * 0.5;
    x = x.clamp(8.0 * s, (screen.0 - w - 8.0 * s).max(8.0 * s));
    let mut y = anchor.y - h - 8.0 * s;
    if y < 8.0 * s {
        y = anchor.bottom() + 8.0 * s;
    }
    let rect = Rect::new(x.round(), y.round(), w.round(), h.round());
    let saved_alpha = painter.alpha;
    let saved_clip = painter.clip();
    painter.set_clip(None);
    painter.alpha = saved_alpha * alpha;
    painter.round_rect_shadow(
        rect,
        8.0 * s,
        [6.0 * s, 2.0 * s, 6.0 * s, 10.0 * s],
        black(0.30),
    );
    painter.round_rect(rect, 8.0 * s, Fill::Solid(css(18, 22, 30, 0.92)));
    painter.round_rect_stroke(rect, 8.0 * s, 1.0, 0.5, white(0.22));
    let cy = rect.y + rect.h * 0.5;
    fonts.plain_mid(painter, label, rect.x + pad_x, cy, style, white(0.97));
    if let Some(key) = key {
        let kx = rect.x + pad_x + lw + gap;
        let kr = Rect::new(kx, cy - 9.0 * s, kw, 18.0 * s);
        painter.round_rect(kr, 5.0 * s, Fill::Solid(white(0.12)));
        painter.round_rect_stroke(kr, 5.0 * s, 1.0, 0.5, white(0.25));
        fonts.plain_mid(painter, key, kx + 6.0 * s, cy, key_style, white(0.85));
    }
    painter.alpha = saved_alpha;
    painter.set_clip(saved_clip);
}

/// A toast: a small glass message near the bottom centre (or at `y`), fading with `alpha`.
#[allow(clippy::too_many_arguments)]
pub fn toast(
    painter: &mut Painter<'_>,
    fonts: &mut Fonts<'_>,
    text: &str,
    icon: Option<(Icon, Rgba)>,
    center_x: f32,
    y: f32,
    alpha: f32,
    s: f32,
) -> Rect {
    let style = TextStyle::new(13.5 * s).weight(Weight::Semibold);
    let tw = fonts.width(text, style);
    let iw = if icon.is_some() { 24.0 * s } else { 0.0 };
    let w = tw + iw + 32.0 * s;
    let h = 38.0 * s;
    let rect = Rect::new(
        (center_x - w * 0.5).round(),
        y.round(),
        w.round(),
        h.round(),
    );
    let saved = painter.alpha;
    painter.alpha = saved * alpha;
    painter.round_rect_shadow(
        rect,
        h * 0.5,
        [10.0 * s, 4.0 * s, 10.0 * s, 16.0 * s],
        black(0.3),
    );
    painter.round_rect(rect, h * 0.5, Fill::Solid(css(20, 26, 36, 0.90)));
    painter.round_rect_stroke(rect, h * 0.5, 1.0, 0.5, white(0.28));
    let cy = rect.y + h * 0.5;
    let mut x = rect.x + 16.0 * s;
    if let Some((icon, color)) = icon {
        icons::draw(painter, icon, (x + 8.0 * s, cy), 16.0 * s, 1.8 * s, color);
        x += iw;
    }
    fonts.plain_mid(painter, text, x, cy, style, white(0.97));
    painter.alpha = saved;
    rect
}

/// A filled convex polygon with an anti-aliased edge, through the painter's offset, alpha and
/// clip (the painter has no public polygon fill; this mirrors its triangle path).
pub fn fill_convex(painter: &mut Painter<'_>, points: &[[f32; 2]], color: Rgba) {
    let n = points.len();
    if !(3..=16).contains(&n) {
        return;
    }
    // Winding: positive area means clockwise in y-down space.
    let mut area = 0.0;
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        area += a[0] * b[1] - b[0] * a[1];
    }
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    // Offset each vertex along its corner's bisector by `d` (outward for d > 0).
    let offset = |d: f32| -> Vec<[f32; 2]> {
        (0..n)
            .map(|i| {
                let prev = points[(i + n - 1) % n];
                let p = points[i];
                let next = points[(i + 1) % n];
                let normal = |a: [f32; 2], b: [f32; 2]| {
                    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
                    // Outward for clockwise (y-down) polygons.
                    [sign * dy / len, -sign * dx / len]
                };
                let (n0, n1) = (normal(prev, p), normal(p, next));
                let mut m = [n0[0] + n1[0], n0[1] + n1[1]];
                let len = (m[0] * m[0] + m[1] * m[1]).sqrt().max(1e-6);
                m = [m[0] / len, m[1] / len];
                let cos = (m[0] * n0[0] + m[1] * n0[1]).max(0.35);
                [p[0] + m[0] * d / cos, p[1] + m[1] * d / cos]
            })
            .collect()
    };
    let inner = offset(-0.5);
    let outer = offset(0.5);
    let clear = [color[0], color[1], color[2], 0.0];
    for i in 1..n - 1 {
        emit(painter, [inner[0], inner[i], inner[i + 1]], [color; 3]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        emit(
            painter,
            [inner[i], outer[i], inner[j]],
            [color, clear, color],
        );
        emit(
            painter,
            [inner[j], outer[i], outer[j]],
            [color, clear, clear],
        );
    }
}

/// One solid triangle through the painter's offset, alpha, winding rule and clip.
fn emit(painter: &mut Painter<'_>, pos: [[f32; 2]; 3], colors: [Rgba; 3]) {
    let mut v: [([f32; 2], Rgba); 3] = [
        (pos[0], colors[0]),
        (pos[1], colors[1]),
        (pos[2], colors[2]),
    ];
    for (p, c) in &mut v {
        p[0] += painter.offset.0;
        p[1] += painter.offset.1;
        c[3] *= painter.alpha;
    }
    if v.iter().all(|(_, c)| c[3] <= 0.0) {
        return;
    }
    let cross = (v[1].0[0] - v[0].0[0]) * (v[2].0[1] - v[0].0[1])
        - (v[2].0[0] - v[0].0[0]) * (v[1].0[1] - v[0].0[1]);
    if cross > 0.0 {
        v.swap(1, 2);
    }
    let Some(clip) = painter
        .clip()
        .map(|c| c.offset(painter.offset.0, painter.offset.1))
    else {
        push(painter, &v);
        return;
    };
    let inside = |p: &[f32; 2]| {
        p[0] >= clip.x && p[0] <= clip.right() && p[1] >= clip.y && p[1] <= clip.bottom()
    };
    if v.iter().all(|(p, _)| inside(p)) {
        push(painter, &v);
        return;
    }
    // Sutherland-Hodgman against the four clip edges.
    let mut poly: Vec<([f32; 2], Rgba)> = v.to_vec();
    for (axis, limit, keep_greater) in [
        (0, clip.x, true),
        (0, clip.right(), false),
        (1, clip.y, true),
        (1, clip.bottom(), false),
    ] {
        let keep = |p: &[f32; 2]| {
            if keep_greater {
                p[axis] >= limit
            } else {
                p[axis] <= limit
            }
        };
        let mut out = Vec::with_capacity(poly.len() + 2);
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            let (ka, kb) = (keep(&a.0), keep(&b.0));
            if ka {
                out.push(a);
            }
            if ka != kb {
                let t = (limit - a.0[axis]) / (b.0[axis] - a.0[axis]);
                out.push((
                    [
                        a.0[0] + (b.0[0] - a.0[0]) * t,
                        a.0[1] + (b.0[1] - a.0[1]) * t,
                    ],
                    mix(a.1, b.1, t),
                ));
            }
        }
        poly = out;
        if poly.len() < 3 {
            return;
        }
    }
    for i in 1..poly.len() - 1 {
        push(painter, &[poly[0], poly[i], poly[i + 1]]);
    }
}

fn push(painter: &mut Painter<'_>, v: &[([f32; 2], Rgba); 3]) {
    painter.list.triangle(
        &UiTexture::Solid,
        [v[0].0, v[1].0, v[2].0],
        [[0.0, 0.0]; 3],
        [v[0].1, v[1].1, v[2].1],
    );
}

/// A soft vertical scrim (clear at the top to `alpha` black at the bottom) behind controls
/// that sit on the live world, so white text reads over a noon sky.
pub fn scrim(painter: &mut Painter<'_>, rect: Rect, alpha: f32) {
    painter.round_rect(
        rect,
        0.0,
        Fill::css_angle(rect, 180.0, black(0.0), black(alpha)),
    );
}

/// Hit regions recorded while drawing (immediate mode, like the mod menu): the next input event
/// tests the last frame's. Later entries are on top.
#[derive(Debug, Clone)]
pub struct Hits<T: Copy + PartialEq> {
    entries: Vec<(Rect, T, Option<Tip>)>,
}

impl<T: Copy + PartialEq> Default for Hits<T> {
    fn default() -> Self {
        Hits {
            entries: Vec::new(),
        }
    }
}

/// A control's tooltip: its name and its key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tip {
    /// What the control does, in two or three words.
    pub label: &'static str,
    /// Its keyboard shortcut, if any.
    pub key: Option<&'static str>,
}

impl Tip {
    /// A tooltip.
    #[must_use]
    pub const fn new(label: &'static str, key: Option<&'static str>) -> Tip {
        Tip { label, key }
    }
}

impl<T: Copy + PartialEq> Hits<T> {
    /// Forget the last frame's regions.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Record a region.
    pub fn push(&mut self, rect: Rect, target: T, tip: Option<Tip>) {
        if rect.w > 0.0 && rect.h > 0.0 {
            self.entries.push((rect, target, tip));
        }
    }

    /// The topmost target under `(x, y)`.
    #[must_use]
    pub fn at(&self, x: f32, y: f32) -> Option<T> {
        self.entries
            .iter()
            .rev()
            .find(|(rect, _, _)| rect.contains(x, y))
            .map(|(_, target, _)| *target)
    }

    /// The topmost region under `(x, y)` with its tooltip.
    #[must_use]
    pub fn entry_at(&self, x: f32, y: f32) -> Option<(Rect, T, Option<Tip>)> {
        self.entries
            .iter()
            .rev()
            .find(|(rect, _, _)| rect.contains(x, y))
            .copied()
    }

    /// The rectangle `target` was drawn at in the last frame.
    #[must_use]
    pub fn rect_of(&self, target: T) -> Option<Rect> {
        self.entries
            .iter()
            .rev()
            .find(|(_, t, _)| *t == target)
            .map(|(rect, _, _)| *rect)
    }

    /// Every recorded target (tests).
    #[must_use]
    pub fn targets(&self) -> Vec<T> {
        self.entries.iter().map(|(_, t, _)| *t).collect()
    }
}

/// The tooltip clock: which control the pointer rests on and since when.
#[derive(Debug, Clone, Copy, Default)]
pub struct TipClock {
    over: Option<(Rect, Tip)>,
    since: Option<Instant>,
    /// A press hides the tooltip until the pointer moves to another control.
    suppressed: bool,
}

impl TipClock {
    /// Feed this frame's hovered control; returns the tooltip to draw and its alpha.
    pub fn update(
        &mut self,
        hovered: Option<(Rect, Tip)>,
        now: Instant,
    ) -> Option<(Rect, Tip, f32)> {
        let same = match (&self.over, &hovered) {
            (Some((_, a)), Some((_, b))) => a == b,
            (None, None) => true,
            _ => false,
        };
        if !same {
            self.over = hovered;
            self.since = hovered.map(|_| now);
            self.suppressed = false;
        } else if let (Some(over), Some(new)) = (self.over.as_mut(), hovered) {
            // The control may have moved (a sliding window): follow it.
            over.0 = new.0;
        }
        let (rect, tip) = self.over?;
        if self.suppressed {
            return None;
        }
        let ms = ms_since(self.since?, now) - TOOLTIP_DELAY_MS;
        (ms > 0.0).then(|| (rect, tip, (ms / 120.0).min(1.0)))
    }

    /// A press: hide the tooltip until the pointer goes elsewhere.
    pub fn suppress(&mut self) {
        self.suppressed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::DrawList;

    #[test]
    fn the_scale_follows_the_height_with_a_readable_floor() {
        assert_eq!(scale((1920.0, 1080.0)), 1.0);
        assert_eq!(scale((3840.0, 2160.0)), 2.0);
        // 720p: 0.8, not 0.67, so 12 px text stays near 10 px.
        assert!((scale((1280.0, 720.0)) - 0.8).abs() < 0.001);
        // 4:3: the width decides so the control row fits.
        assert!((scale((1024.0, 768.0)) - 1024.0 / 1440.0).abs() < 0.001);
        // A tiny window: as large as still fits.
        assert!((scale((854.0, 480.0)) - 0.593).abs() < 0.001);
    }

    #[test]
    fn a_filled_triangle_faces_the_ui_pipeline_and_respects_the_clip() {
        for points in [
            [[0.0, 0.0], [10.0, 5.0], [0.0, 10.0]],
            [[0.0, 0.0], [0.0, 10.0], [10.0, 5.0]],
        ] {
            let mut list = DrawList::new();
            let mut painter = Painter::new(&mut list);
            fill_convex(&mut painter, &points, white(1.0));
            assert!(!list.is_empty());
            for t in list.vertices().chunks(3) {
                let cross = (t[1].pos[0] - t[0].pos[0]) * (t[2].pos[1] - t[0].pos[1])
                    - (t[2].pos[0] - t[0].pos[0]) * (t[1].pos[1] - t[0].pos[1]);
                assert!(cross <= 1e-3, "culled winding");
            }
            // Inside its bounds plus the fringe.
            assert!(list
                .vertices()
                .iter()
                .all(|v| v.pos[0] >= -1.5 && v.pos[0] <= 11.5));
        }
        let mut list = DrawList::new();
        let mut painter = Painter::new(&mut list);
        painter.set_clip(Some(Rect::new(0.0, 0.0, 5.0, 100.0)));
        fill_convex(
            &mut painter,
            &[[0.0, 0.0], [10.0, 5.0], [0.0, 10.0]],
            white(1.0),
        );
        assert!(list.vertices().iter().all(|v| v.pos[0] <= 5.0001));
    }

    #[test]
    fn hits_prefer_the_topmost_region() {
        let mut hits: Hits<u8> = Hits::default();
        hits.push(Rect::new(0.0, 0.0, 100.0, 100.0), 1, None);
        hits.push(
            Rect::new(10.0, 10.0, 10.0, 10.0),
            2,
            Some(Tip::new("x", None)),
        );
        assert_eq!(hits.at(15.0, 15.0), Some(2));
        assert_eq!(hits.at(50.0, 50.0), Some(1));
        assert_eq!(hits.at(150.0, 50.0), None);
        assert_eq!(hits.rect_of(2), Some(Rect::new(10.0, 10.0, 10.0, 10.0)));
    }

    #[test]
    fn tooltips_wait_then_fade_in_and_a_press_hides_them() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + std::time::Duration::from_millis(ms);
        let tip = Tip::new("Play", Some("Space"));
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        let mut clock = TipClock::default();
        assert!(clock.update(Some((rect, tip)), at(0)).is_none());
        assert!(clock.update(Some((rect, tip)), at(300)).is_none());
        let (_, shown, alpha) = clock.update(Some((rect, tip)), at(700)).expect("shown");
        assert_eq!(shown, tip);
        assert!(alpha > 0.99);
        clock.suppress();
        assert!(clock.update(Some((rect, tip)), at(800)).is_none());
        // Another control restarts the clock.
        let other = Tip::new("Pause", None);
        assert!(clock.update(Some((rect, other)), at(900)).is_none());
        assert!(clock.update(Some((rect, other)), at(1400)).is_some());
        assert!(clock.update(None, at(1500)).is_none());
    }
}
