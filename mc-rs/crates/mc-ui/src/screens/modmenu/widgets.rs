//! Shared drawing pieces (the Aero switch, value badge, slider track, pills, icons, text with its
//! soft shadow) and small helpers (time, sliders, colour conversions).

use std::time::Instant;

use mc_mods::{ModSettings, OptionKind, OptionValue, RowIcon};

use super::paint::{self, black, css, white, Fill, Painter, Rect, Rgba};
use super::text::{TextStyle, Weight};
use super::MenuEnv;

pub(super) const PILL_H: f32 = 38.0;

/// How long an armed "Reset to defaults" or profile delete waits for its second click.
pub(super) const ARM_MS: u128 = 3000;

pub(super) fn ms_since(since: Instant, now: Instant) -> f32 {
    now.saturating_duration_since(since).as_secs_f32() * 1000.0
}

/// An armed button (clicked once at `at`) still takes its confirming second click at `now`.
pub(super) fn still_armed(at: Instant, now: Instant) -> bool {
    now.saturating_duration_since(at).as_millis() < ARM_MS
}

// ---- The three-window menu's small pieces -----------------

/// The keyboard focus ring: a soft blue outline just outside `rect`.
pub(super) fn draw_focus_ring(painter: &mut Painter<'_>, rect: Rect, radius: f32, s: f32) {
    painter.round_rect_stroke(
        rect.inset(-2.0 * s),
        radius + 2.0 * s,
        1.5 * s,
        0.5,
        css(130, 190, 255, 0.95),
    );
}

/// A small push pin (pinned to My setup), centred near `(cx, cy)`, tilted like a pin stuck in
/// a board.
pub(super) fn draw_pin(painter: &mut Painter<'_>, cx: f32, cy: f32, s: f32, color: Rgba) {
    // The needle, down and to the left.
    painter.line(
        cx - 0.6 * s,
        cy + 0.6 * s,
        cx - 4.2 * s,
        cy + 4.2 * s,
        1.2 * s,
        color,
    );
    // The collar across the needle, then the round head.
    painter.line(
        cx - 2.6 * s,
        cy - 0.8 * s,
        cx + 0.8 * s,
        cy + 2.6 * s,
        1.6 * s,
        color,
    );
    painter.circle(cx + 1.4 * s, cy - 1.4 * s, 2.9 * s, color);
}

/// `text` shortened to fit `max_w`: whole when it fits, else cut at a word (or a character, for
/// one long word) with "…" after it.
pub(super) fn fit_ellipsis<'t>(
    env: &mut MenuEnv<'_>,
    text: &'t str,
    max_w: f32,
    style: TextStyle,
) -> std::borrow::Cow<'t, str> {
    if env.text.measure(text, style, env.mc_font) <= max_w {
        return std::borrow::Cow::Borrowed(text);
    }
    let ellipsis_w = env.text.measure("\u{2026}", style, env.mc_font);
    let room = (max_w - ellipsis_w).max(0.0);
    let mut cut = env
        .text
        .fit_words(text, room, style, env.mc_font)
        .min(text.len());
    if cut == 0 {
        // One word longer than the room: cut by characters.
        cut = text
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| env.text.measure(&text[..i], style, env.mc_font) <= room)
            .last()
            .unwrap_or(0);
    }
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let head = text[..cut].trim_end_matches([' ', ',', '.', ';', ':']);
    std::borrow::Cow::Owned(format!("{head}\u{2026}"))
}

/// `text` wrapped at words into lines `w` wide from `(x, y)`, with the design's text shadow.
/// Returns the y below the last line.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_wrapped(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    text: &str,
    x: f32,
    y: f32,
    w: f32,
    style: TextStyle,
    color: Rgba,
    s: f32,
) -> f32 {
    let (_, lh) = env.text.metrics(style);
    let mut rest = text.trim();
    let mut y = y;
    while !rest.is_empty() {
        let mut cut = env
            .text
            .fit_words(rest, w, style, env.mc_font)
            .min(rest.len());
        while cut > 0 && !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut == 0 {
            cut = rest.chars().next().map_or(rest.len(), char::len_utf8);
        }
        let (line, tail) = rest.split_at(cut);
        shadowed(painter, env, line.trim_end(), x, y, style, color, s);
        y += lh + 2.0 * s;
        rest = tail.trim_start();
    }
    y
}

/// The step a slider's badge resolves: its snap step, or for a continuous slider the last
/// decimal its badge shows.
pub(super) fn display_step(step: f32, decimals: u8) -> f32 {
    if step > 0.0 {
        step
    } else {
        10.0_f32.powi(-i32::from(decimals))
    }
}

/// A slider press or drag at the physical-pixel `x`: the pointer's exact position on the track
/// (the menu works in physical pixels, so there is no GUI-pixel rounding), snapped by the
/// store to the slider's step; a continuous slider snaps to the step its badge shows, so the
/// value is always exactly what the badge says.
pub(super) fn set_slider_from_x(
    settings: &mut ModSettings,
    m: usize,
    o: usize,
    track: Rect,
    x: f32,
) {
    let Some(option) = settings.catalog().get(m).and_then(|d| d.options.get(o)) else {
        return;
    };
    let t = ((x - track.x) / track.w.max(1.0)).clamp(0.0, 1.0);
    let value = match option.kind {
        OptionKind::Slider {
            min,
            max,
            step,
            decimals,
            ..
        } => {
            let raw = min + (max - min) * t;
            if step > 0.0 {
                raw
            } else {
                let unit = display_step(step, decimals);
                min + ((raw - min) / unit).round() * unit
            }
        }
        OptionKind::SliderLabeled { min, max, .. } => min + (max - min) * t,
        _ => return,
    };
    settings.set_value_at(m, o, OptionValue::Float(value));
}

/// Text with the design's soft dark shadow (`text-shadow: 0 1px 1px rgba(0,0,0,.38)`).
#[allow(clippy::too_many_arguments)]
pub(super) fn shadowed(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    text: &str,
    x: f32,
    y: f32,
    style: TextStyle,
    color: Rgba,
    s: f32,
) -> f32 {
    let shadow = black(0.38 * color[3]);
    env.text.draw(
        painter,
        env.mc_font,
        text,
        x,
        y + (1.0 * s).max(1.0),
        style,
        shadow,
    );
    env.text
        .draw(painter, env.mc_font, text, x, y, style, color)
}

/// The Aero switch: a pill track and a round knob, `t` = 0 off .. 1 on.
pub(super) fn draw_switch(painter: &mut Painter<'_>, rect: Rect, t: f32, hover: bool) {
    let r = rect.h * 0.5;
    let off_fill = black(if hover { 0.32 } else { 0.26 });
    let on_fill = white(0.62);
    let fill = [
        off_fill[0] + (on_fill[0] - off_fill[0]) * t,
        off_fill[1] + (on_fill[1] - off_fill[1]) * t,
        off_fill[2] + (on_fill[2] - off_fill[2]) * t,
        off_fill[3] + (on_fill[3] - off_fill[3]) * t,
    ];
    if t > 0.0 {
        painter.round_rect_shadow(rect, r, [5.0; 4], white(0.10 * t));
    }
    painter.round_rect(rect, r, Fill::Solid(fill));
    painter.round_rect_stroke(rect, r, 1.0, 0.5, white(0.30 + 0.45 * t));
    let knob_r = r - 2.5;
    let cx = rect.x + r + (rect.w - 2.0 * r) * t;
    painter.circle(cx, rect.y + r + 0.5, knob_r, black(0.25));
    painter.circle(cx, rect.y + r, knob_r, white(0.78 + 0.2 * t));
}

/// A right-aligned value badge (`.badge`), vertically centred in a `line_h` line ending at
/// `right`. Returns its rectangle.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_badge(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    text: &str,
    right: f32,
    y: f32,
    line_h: f32,
    s: f32,
    hover: bool,
    arrows: bool,
) -> Rect {
    let style = TextStyle::new(11.0 * s);
    let tw = env.text.measure(text, style, env.mc_font);
    let (_, lh) = env.text.metrics(style);
    let arrow_w = if arrows { 12.0 * s } else { 0.0 };
    let w = tw + 16.0 * s + arrow_w * 2.0;
    let h = lh + 6.0 * s;
    let rect = Rect::new(right - w, y + (line_h - h) * 0.5, w, h);
    painter.round_rect(
        rect,
        8.0 * s,
        Fill::Solid(black(if hover { 0.34 } else { 0.24 })),
    );
    painter.round_rect_stroke(
        rect,
        8.0 * s,
        1.0,
        0.5,
        white(if hover { 0.40 } else { 0.16 }),
    );
    if arrows {
        let cy = rect.y + rect.h * 0.5;
        let lx = rect.x + 9.0 * s;
        painter.line(
            lx + 2.0 * s,
            cy - 3.0 * s,
            lx - 1.0 * s,
            cy,
            1.1 * s,
            white(0.55),
        );
        painter.line(
            lx - 1.0 * s,
            cy,
            lx + 2.0 * s,
            cy + 3.0 * s,
            1.1 * s,
            white(0.55),
        );
        let rx = rect.right() - 9.0 * s;
        painter.line(
            rx - 2.0 * s,
            cy - 3.0 * s,
            rx + 1.0 * s,
            cy,
            1.1 * s,
            white(0.55),
        );
        painter.line(
            rx + 1.0 * s,
            cy,
            rx - 2.0 * s,
            cy + 3.0 * s,
            1.1 * s,
            white(0.55),
        );
    }
    env.text.draw(
        painter,
        env.mc_font,
        text,
        rect.x + 8.0 * s + arrow_w,
        rect.y + 3.0 * s,
        style,
        white(0.86),
    );
    rect
}

pub(super) fn draw_slider_track(
    painter: &mut Painter<'_>,
    track: Rect,
    t: f32,
    fill: Fill,
    s: f32,
) {
    let r = track.h * 0.5;
    painter.round_rect(track, r, Fill::Solid(white(0.16)));
    let filled = Rect::new(track.x, track.y, (track.w * t).max(track.h), track.h);
    painter.round_rect(filled, r, fill);
    let cx = track.x + track.w * t;
    let cy = track.y + track.h * 0.5;
    painter.circle(cx, cy + 0.6 * s, 6.5 * s, black(0.28));
    painter.circle(cx, cy, 6.0 * s, white(0.97));
}

pub(super) fn draw_checker(painter: &mut Painter<'_>, rect: Rect, s: f32) {
    let cell = (4.0 * s).max(2.0);
    let saved = painter.clip();
    painter.set_clip(Some(saved.map_or(rect, |c| c.intersect(&rect))));
    let cols = (rect.w / cell).ceil() as i32;
    let rows = (rect.h / cell).ceil() as i32;
    painter.rect(rect, white(0.85));
    for row in 0..rows {
        for col in 0..cols {
            if (row + col) % 2 == 0 {
                painter.rect(
                    Rect::new(
                        rect.x + col as f32 * cell,
                        rect.y + row as f32 * cell,
                        cell,
                        cell,
                    ),
                    css(150, 150, 150, 0.85),
                );
            }
        }
    }
    painter.set_clip(saved);
}

pub(super) fn draw_icon(painter: &mut Painter<'_>, icon: RowIcon, r: Rect, s: f32) {
    let (cx, cy) = r.center();
    let c = white(0.9);
    let lw = 1.3 * s;
    match icon {
        RowIcon::Eye | RowIcon::EyeOff => {
            let w = 6.0 * s;
            let h = 3.5 * s;
            painter.line(cx - w, cy, cx, cy - h, lw, c);
            painter.line(cx, cy - h, cx + w, cy, lw, c);
            painter.line(cx - w, cy, cx, cy + h, lw, c);
            painter.line(cx, cy + h, cx + w, cy, lw, c);
            painter.circle(cx, cy, 1.6 * s, c);
            if icon == RowIcon::EyeOff {
                painter.line(cx - w, cy + h + 1.0 * s, cx + w, cy - h - 1.0 * s, lw, c);
            }
        }
        RowIcon::Delete => {
            let d = 3.5 * s;
            painter.line(cx - d, cy - d, cx + d, cy + d, lw, c);
            painter.line(cx - d, cy + d, cx + d, cy - d, lw, c);
        }
        RowIcon::Play => {
            let d = 3.5 * s;
            painter.line(cx - d * 0.6, cy - d, cx + d, cy, lw, c);
            painter.line(cx + d, cy, cx - d * 0.6, cy + d, lw, c);
            painter.line(cx - d * 0.6, cy + d, cx - d * 0.6, cy - d, lw, c);
        }
        RowIcon::Stop => {
            let d = 3.0 * s;
            painter.round_rect(
                Rect::new(cx - d, cy - d, 2.0 * d, 2.0 * d),
                1.0 * s,
                Fill::Solid(c),
            );
        }
        RowIcon::Plus => {
            let d = 4.0 * s;
            painter.line(cx - d, cy, cx + d, cy, lw, c);
            painter.line(cx, cy - d, cx, cy + d, lw, c);
        }
        RowIcon::Refresh => {
            let d = 4.0 * s;
            let steps = 10;
            for i in 0..steps {
                let a0 = 0.6 + i as f32 / steps as f32 * 4.6;
                let a1 = 0.6 + (i + 1) as f32 / steps as f32 * 4.6;
                painter.line(
                    cx + a0.cos() * d,
                    cy + a0.sin() * d,
                    cx + a1.cos() * d,
                    cy + a1.sin() * d,
                    lw,
                    c,
                );
            }
            let tip = (cx + (5.2_f32).cos() * d, cy + (5.2_f32).sin() * d);
            painter.line(tip.0, tip.1, tip.0 + 2.5 * s, tip.1 - 0.5 * s, lw, c);
        }
    }
}

/// An Aero taskbar pill (`.task-app`): translucent, a light border, an inset underline when
/// hovered.
pub(super) fn draw_pill(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    rect: Rect,
    label: &str,
    hover: bool,
    s: f32,
) {
    draw_pill_body(painter, rect, hover, s);
    let style = TextStyle::new(12.5 * s).weight(Weight::Semibold);
    let tw = env.text.measure(label, style, env.mc_font);
    let (_, lh) = env.text.metrics(style);
    shadowed(
        painter,
        env,
        label,
        rect.x + (rect.w - tw) * 0.5,
        rect.y + (rect.h - lh) * 0.5,
        style,
        white(0.96),
        s,
    );
}

/// A pill's glass body: the `.task-app` gradient, border and hover underline, no label.
pub(super) fn draw_pill_body(painter: &mut Painter<'_>, rect: Rect, hover: bool, s: f32) {
    let r = 16.0 * s;
    painter.round_rect_shadow(
        rect,
        r,
        [10.0 * s, 4.0 * s, 10.0 * s, 18.0 * s],
        black(0.28),
    );
    painter.round_rect(
        rect,
        r,
        Fill::css_angle(
            rect,
            180.0,
            white(if hover { 0.16 } else { 0.075 }),
            black(0.13),
        ),
    );
    painter.round_rect_stroke(rect, r, 1.0, 0.5, white(if hover { 0.55 } else { 0.30 }));
    if hover {
        painter.line(
            rect.x + r,
            rect.bottom() - 2.0,
            rect.right() - r,
            rect.bottom() - 2.0,
            1.5,
            white(0.86),
        );
    }
}

/// `0xAARRGGBB` to `[h, s, v, a]`, all `0..=1`.
#[must_use]
pub fn argb_to_hsva(argb: u32) -> [f32; 4] {
    let [r, g, b, a] = paint::argb(argb);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d <= 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };
    let s = if max <= 0.0 { 0.0 } else { d / max };
    [h, s, max, a]
}

/// `[h, s, v, a]` to `0xAARRGGBB`.
#[must_use]
pub fn hsva_to_argb([h, s, v, a]: [f32; 4]) -> u32 {
    let h = (h.clamp(0.0, 1.0) * 6.0) % 6.0;
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let byte = |f: f32| ((f + m).clamp(0.0, 1.0) * 255.0).round() as u32;
    ((a.clamp(0.0, 1.0) * 255.0).round() as u32) << 24 | byte(r) << 16 | byte(g) << 8 | byte(b)
}
