//! The Spotify widget's "Aero" style: the mod menu's glass vocabulary ([`Painter`]: rounded
//! panels with a one-pixel anti-aliased fringe, the white sheen, top highlight and border) and
//! its Segoe UI text ([`MenuText`]), over a dark translucent tint instead of the menu's frosted
//! scene copy.
//!
//! # Cost
//!
//! The menu's real glass needs the frame behind it, which [`mc_render::PhaseList`] provides only
//! by switching the whole frame to an intermediate target with a full-frame copy and blit. This
//! style never asks for the scene: it is ordinary geometry in the UI pass the HUD already uses.
//!
//! Everything is drawn in **physical pixels** (a `1 / pixel` transform inside the element), so
//! fringes are one screen pixel and glyphs are rasterised at their on-screen size. Both halves
//! are cached and replayed:
//! * the shapes (panel, cover, buttons, bars) are rebuilt only when what they show changes: a
//!   new state, a hover, the progress bar crossing a whole pixel;
//! * the text is laid out once per string change into glyph quads (the time labels once a
//!   second), then replayed with the scroll offset and clipped to its column.

use crate::font::FontRenderer;
use crate::screens::modmenu::paint::{black, white, Fill, Painter, Rect, Rgba};
use crate::screens::modmenu::text::{MenuText, TextStyle, Weight};
use crate::widget::{DrawList, Transform, UiTexture};

use super::spotify::{
    cover_texture, marquee_offset, AeroText, SpotifyControl, SpotifyHud, SpotifyLayout, TimeLabels,
};

/// Panel corner radius, GUI pixels.
const RADIUS: f32 = 5.0;
/// Cover corner radius, GUI pixels.
const COVER_RADIUS: f32 = 3.0;
/// Title em size (Segoe UI Semibold), GUI pixels.
const TITLE_EM: f32 = 8.5;
/// Artist em size, GUI pixels.
const ARTIST_EM: f32 = 7.5;
/// Time label em size, GUI pixels.
const TIME_EM: f32 = 6.5;
/// Most outline points one rounded rectangle produces here (4 corners × 13).
const MAX_OUTLINE: usize = 52;

/// One glyph quad of a laid-out line, relative to the line's pen origin and top.
#[derive(Debug, Clone, Copy, PartialEq)]
struct GlyphQuad {
    texture: usize,
    pos: [f32; 4],
    uv: [f32; 4],
}

/// A line of text laid out once, replayed every frame.
#[derive(Debug, Default)]
struct GlyphLine {
    /// The string it was laid out for.
    text: String,
    /// The em size (physical pixels) it was laid out at.
    size: f32,
    /// Advance width, physical pixels.
    width: f32,
    /// The font's ascent at that size (the line's baseline sits this far below its top).
    ascent: f32,
    textures: Vec<UiTexture>,
    quads: Vec<GlyphQuad>,
}

impl GlyphLine {
    /// Lay `text` out (into the shared atlas) unless it already is, at this size.
    fn ensure(
        &mut self,
        text: &str,
        style: TextStyle,
        menu_text: &mut MenuText,
        font: &mut FontRenderer,
        scratch: &mut DrawList,
        force: bool,
    ) {
        if !force && self.size == style.size && self.text == text {
            return;
        }
        self.text.clear();
        self.text.push_str(text);
        self.size = style.size;
        self.textures.clear();
        self.quads.clear();
        scratch.clear();
        self.ascent = menu_text.metrics(style).0;
        let mut painter = Painter::new(scratch);
        self.width = menu_text.draw(&mut painter, font, text, 0.0, 0.0, style, white(1.0));
        // `Painter::textured` emits each glyph as two triangles, six vertices, both in the
        // glyph's batch: rebuild the axis-aligned quads from them.
        for batch in scratch.batches() {
            let texture = match self.textures.iter().position(|t| *t == batch.texture) {
                Some(index) => index,
                None => {
                    self.textures.push(batch.texture.clone());
                    self.textures.len() - 1
                }
            };
            for quad in scratch.vertices()[batch.range.clone()].chunks_exact(6) {
                let mut pos = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
                let mut uv = [0.0; 4];
                for v in quad {
                    pos[0] = pos[0].min(v.pos[0]);
                    pos[1] = pos[1].min(v.pos[1]);
                    pos[2] = pos[2].max(v.pos[0]);
                    pos[3] = pos[3].max(v.pos[1]);
                }
                for v in quad {
                    if v.pos[0] == pos[0] {
                        uv[0] = v.uv[0];
                    }
                    if v.pos[0] == pos[2] {
                        uv[2] = v.uv[0];
                    }
                    if v.pos[1] == pos[1] {
                        uv[1] = v.uv[1];
                    }
                    if v.pos[1] == pos[3] {
                        uv[3] = v.uv[1];
                    }
                }
                self.quads.push(GlyphQuad { texture, pos, uv });
            }
        }
        scratch.clear();
    }

    /// Emit the line with its pen at `(x, top)`, clipped to `x0..x1`. A `fade` wider than zero
    /// fades the text out over that many pixels at an edge it is cut at (a scrolling line).
    fn draw(&self, d: &mut DrawList, x: f32, top: f32, clip: (f32, f32), fade: f32, color: Rgba) {
        let (x0, x1) = clip;
        let fade_left = fade > 0.0 && x < x0;
        let fade_right = fade > 0.0 && x + self.width > x1;
        let alpha = |px: f32| {
            let mut a = 1.0f32;
            if fade_left {
                a = a.min((px - x0) / fade);
            }
            if fade_right {
                a = a.min((x1 - px) / fade);
            }
            color[3] * a.clamp(0.0, 1.0)
        };
        for q in &self.quads {
            let (mut l, t, mut r, b) = (q.pos[0] + x, q.pos[1] + top, q.pos[2] + x, q.pos[3] + top);
            if r <= x0 || l >= x1 || r <= l {
                continue;
            }
            let (mut u0, mut u1) = (q.uv[0], q.uv[2]);
            let width = r - l;
            if l < x0 {
                u0 += (q.uv[2] - q.uv[0]) * (x0 - l) / width;
                l = x0;
            }
            if r > x1 {
                u1 -= (q.uv[2] - q.uv[0]) * (r - x1) / width;
                r = x1;
            }
            let (left, right) = if fade_left || fade_right {
                let mut left = color;
                let mut right = color;
                left[3] = alpha(l);
                right[3] = alpha(r);
                (left, right)
            } else {
                (color, color)
            };
            d.quad(
                &self.textures[q.texture],
                [[l, t], [l, b], [r, t], [r, b]],
                [[u0, q.uv[1]], [u0, q.uv[3]], [u1, q.uv[1]], [u1, q.uv[3]]],
                [left, left, right, right],
            );
        }
    }
}

/// What the cached shapes show; any change rebuilds them.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ShapeKey {
    pixel: f32,
    /// Background opacity, `0.0..=1.0`: scales the glass panel only.
    background: f32,
    layout: SpotifyLayout,
    cover: bool,
    playing: bool,
    enabled: [bool; 3],
    lit: Option<SpotifyControl>,
    pressed: Option<SpotifyControl>,
    /// The volume knob's x, physical pixels (`None` without a volume).
    volume_x: Option<i32>,
    /// `(x0, x1, fill)` of the progress bar, physical pixels.
    progress: Option<(i32, i32, i32)>,
}

/// The Aero painter: caches, plus a Minecraft-font fallback when the client gave no Segoe UI.
#[derive(Debug, Default)]
pub(crate) struct AeroPainter {
    shapes: DrawList,
    shape_key: Option<ShapeKey>,
    title: GlyphLine,
    artist: GlyphLine,
    elapsed: GlyphLine,
    total: GlyphLine,
    /// The em scale the lines were laid out for, and which text renderer laid them out.
    text_key: Option<(f32, usize)>,
    scratch: DrawList,
    fallback: Option<MenuText>,
}

impl AeroPainter {
    /// Forget every cache (options changed).
    pub(crate) fn invalidate(&mut self) {
        self.shape_key = None;
        self.text_key = None;
    }

    /// How many triangles the cached shapes replay per frame (tests, the cost report).
    #[must_use]
    pub(crate) fn shape_triangles(&self) -> usize {
        self.shapes.vertices().len() / 3
    }

    /// How many glyph quads the text replays per frame, before clipping (tests, the cost
    /// report). The shadow doubles the title's and artist's.
    #[must_use]
    pub(crate) fn text_quads(&self) -> usize {
        self.title.quads.len() * 2
            + self.artist.quads.len() * 2
            + self.elapsed.quads.len()
            + self.total.quads.len()
    }

    /// Draw the widget. `pixel` is physical pixels per local GUI pixel; `t_ms` the time since
    /// the shown text appeared (the marquee clock); `background` the Background opacity
    /// (`0.0..=1.0`), which reaches the glass panel only.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &mut self,
        layout: &SpotifyLayout,
        data: &SpotifyHud,
        times: &TimeLabels,
        text: Option<&AeroText>,
        t_ms: u64,
        pixel: f32,
        background: f32,
        font: &mut FontRenderer,
        d: &mut DrawList,
    ) {
        let s = pixel.max(0.25);
        let has_times = layout.progress_y.is_some() && data.progress().is_some();

        // ---- text: laid out on change, into the shared atlas ----
        // Segoe UI from the client, or the Minecraft font when there is none (tests) or for a
        // line Segoe UI has no glyphs for (CJK titles: the Minecraft font's unicode pages).
        let mut shared = text.map(|t| t.0.borrow_mut());
        let fallback = self.fallback.get_or_insert_with(MenuText::minecraft_only);
        let source = text.map_or(0, |t| std::rc::Rc::as_ptr(&t.0) as usize);
        let title_style = TextStyle::new(TITLE_EM * s).weight(Weight::Semibold);
        let artist_style = TextStyle::new(ARTIST_EM * s);
        let time_style = TextStyle::new(TIME_EM * s);
        let key = (s, source);
        let stale = self.text_key != Some(key);
        let rescaled = matches!(self.text_key, Some((old, src)) if src == source && old != s);
        if let Some(menu_text) = shared.as_deref_mut() {
            if rescaled && menu_text.is_segoe() {
                // This widget changed size (the HUD editor's scroll-scale): start a fresh atlas,
                // so glyphs of old sizes never pile up until the atlas resets under cached
                // quads. The client uploads the new atlas before this frame reaches the GPU.
                // (Not on a first layout: another widget sharing the renderer may have quads
                // cached in it.)
                *menu_text = MenuText::load();
            }
        }
        self.text_key = Some(key);
        let lines = [
            (&mut self.title, data.title.as_str(), title_style),
            (&mut self.artist, data.artist.as_str(), artist_style),
            (
                &mut self.elapsed,
                if has_times {
                    times.elapsed.as_str()
                } else {
                    ""
                },
                time_style,
            ),
            (
                &mut self.total,
                if has_times { times.total.as_str() } else { "" },
                time_style,
            ),
        ];
        let changed = lines
            .iter()
            .any(|(line, text, style)| line.text != *text || line.size != style.size);
        if stale || changed {
            // Every line together, so a reset of a full atlas (new glyphs after many tracks)
            // cannot leave an earlier line pointing at cleared texels: the check below lays
            // the first line out again and compares.
            let scratch = &mut self.scratch;
            for (line, text, style) in lines {
                let menu_text = pick(shared.as_deref_mut(), fallback, text);
                line.ensure(text, style, menu_text, font, scratch, true);
            }
            let before = self.title.quads.clone();
            let menu_text = pick(shared.as_deref_mut(), fallback, &data.title);
            self.title.ensure(
                &data.title,
                title_style,
                menu_text,
                font,
                &mut self.scratch,
                true,
            );
            if self.title.quads != before {
                for (line, text, style) in [
                    (&mut self.artist, data.artist.as_str(), artist_style),
                    (
                        &mut self.elapsed,
                        if has_times {
                            times.elapsed.as_str()
                        } else {
                            ""
                        },
                        time_style,
                    ),
                    (
                        &mut self.total,
                        if has_times { times.total.as_str() } else { "" },
                        time_style,
                    ),
                ] {
                    let menu_text = pick(shared.as_deref_mut(), fallback, text);
                    line.ensure(text, style, menu_text, font, &mut self.scratch, true);
                }
            }
        }
        drop(shared);

        // ---- geometry in physical pixels ----
        let px = |v: f32| (v * s).round();
        let (x0, x1) = (px(layout.text_x), px(layout.text_x + layout.text_w));
        let progress = layout.progress_y.filter(|_| has_times).map(|_| {
            let label = self.elapsed.width.max(self.total.width).ceil();
            let bar0 = x0 + label + px(4.0);
            let bar1 = x1 - label - px(4.0);
            let fill = bar0 + ((bar1 - bar0).max(0.0) * data.progress().unwrap_or(0.0)).round();
            (bar0 as i32, bar1 as i32, fill as i32)
        });
        let volume_x = layout
            .volume
            .zip(data.volume)
            .map(|(track, v)| (px(track.x) + (px(track.w) * v.clamp(0.0, 1.0))).round() as i32);
        let lit = data
            .interactive
            .then_some(data.pressed.or(data.hover))
            .flatten();
        let key = ShapeKey {
            pixel: s,
            background: background.clamp(0.0, 1.0),
            layout: *layout,
            cover: data.cover,
            playing: data.playing,
            enabled: [data.can_previous, data.can_play_pause, data.can_next],
            lit,
            pressed: data.pressed.filter(|_| data.interactive),
            volume_x,
            progress,
        };
        if self.shape_key != Some(key) {
            self.shape_key = Some(key);
            self.shapes.clear();
            build_shapes(&mut self.shapes, &key, s);
        }

        d.push_transform(Transform::scale(1.0 / s));
        d.append(&self.shapes);
        let shadow = (0.4 * s).round().max(1.0);
        let clip = (x0, x1);
        // Baselines line up with the Minecraft font's, 7 pixels below its cell top.
        let row = |y: f32, line: &GlyphLine| ((y + 7.0) * s - line.ascent).round();
        let title_top = row(layout.title_y, &self.title);
        let artist_top = row(layout.artist_y, &self.artist);
        let width = x1 - x0;
        let title_dx = (marquee_offset((self.title.width - width) / s, t_ms) * s).round();
        let artist_dx = (marquee_offset((self.artist.width - width) / s, t_ms) * s).round();
        let title_x = x0 - title_dx;
        let artist_x = x0 - artist_dx;
        let fade = (6.0 * s).round();
        self.title
            .draw(d, title_x, title_top + shadow, clip, fade, black(0.38));
        self.title
            .draw(d, title_x, title_top, clip, fade, white(0.97));
        self.artist
            .draw(d, artist_x, artist_top + shadow, clip, fade, black(0.30));
        self.artist
            .draw(d, artist_x, artist_top, clip, fade, white(0.70));
        if let (Some(py), true) = (layout.progress_y, has_times) {
            let top = row(py, &self.elapsed);
            let all = (f32::MIN, f32::MAX);
            self.elapsed.draw(d, x0, top, all, 0.0, white(0.62));
            self.total
                .draw(d, x1 - self.total.width.round(), top, all, 0.0, white(0.62));
        }
        d.pop_transform();
    }
}

/// The renderer for `text`: the shared Segoe UI one if it has every glyph, else the
/// Minecraft-font fallback.
fn pick<'a>(
    shared: Option<&'a mut MenuText>,
    fallback: &'a mut MenuText,
    text: &str,
) -> &'a mut MenuText {
    match shared {
        Some(menu_text) if menu_text.covers(text) => menu_text,
        _ => fallback,
    }
}

/// Every shape of the widget, in physical pixels.
fn build_shapes(list: &mut DrawList, key: &ShapeKey, s: f32) {
    let l = &key.layout;
    let px = |v: f32| (v * s).round();
    let scaled = |r: Rect| {
        let (x, y) = (px(r.x), px(r.y));
        Rect::new(x, y, px(r.right()) - x, px(r.bottom()) - y)
    };
    let mut p = Painter::new(list);

    // ---- the glass panel (the menu window's vocabulary over a dark tint) ----
    // Background opacity scales these three (tint, sheen, border) and nothing else; at 0% the
    // panel is left out and the rest floats on the scene.
    let body = Rect::new(0.0, 0.0, px(l.size.0), px(l.size.1));
    let radius = RADIUS * s;
    let bg = key.background;
    if bg > 0.0 {
        // One gradient for the tint and the window's top-left sheen together (lighter
        // top-left, darker bottom-right): one fan instead of two.
        p.round_rect(
            body,
            radius,
            Fill::css_angle(body, 160.0, black(0.34 * bg), black(0.58 * bg)),
        );
        p.line(
            radius * 0.8,
            1.5,
            body.w - radius * 0.8,
            1.5,
            1.0,
            white(0.20 * bg),
        );
        p.round_rect_stroke(body, radius, 1.0, 0.5, white(0.34 * bg));
    }

    // ---- the cover ----
    if let Some(cover) = l.cover {
        let rect = scaled(cover);
        let r = COVER_RADIUS * s;
        if key.cover {
            textured_round_rect(p.list, cover_texture(), rect, r);
        } else {
            p.round_rect(rect, r, Fill::Solid(white(0.08)));
            p.round_rect_stroke(rect, r, 1.0, 0.5, white(0.12));
            note(&mut p, rect, white(0.45));
        }
    }

    // ---- progress ----
    if let (Some((x0, x1, fill)), Some(py)) = (key.progress, l.progress_y) {
        let h = (2.0 * s).round().max(2.0);
        let y = ((py + 3.5) * s - h * 0.5).round();
        let (x0, x1, fill) = (x0 as f32, x1 as f32, fill as f32);
        if x1 > x0 {
            p.round_rect(
                Rect::new(x0, y, x1 - x0, h),
                h * 0.5,
                Fill::Solid(white(0.18)),
            );
            if fill > x0 {
                let w = (fill - x0).max(h);
                p.round_rect(Rect::new(x0, y, w, h), h * 0.5, Fill::Solid(white(0.88)));
            }
        }
    }

    // ---- controls ----
    if let Some(buttons) = l.buttons {
        let controls = [
            SpotifyControl::Previous,
            SpotifyControl::PlayPause,
            SpotifyControl::Next,
        ];
        for ((rect, control), enabled) in buttons.into_iter().zip(controls).zip(key.enabled) {
            let r = scaled(rect);
            let (cx, cy) = r.center();
            let lit = key.lit == Some(control) && enabled;
            let pressed = key.pressed == Some(control) && enabled;
            let icon = 7.0 * s;
            if control == SpotifyControl::PlayPause {
                // The media flyout's round play button: white disc, dark glyph.
                let disc = if !enabled {
                    white(0.30)
                } else if pressed {
                    white(0.74)
                } else if lit {
                    white(1.0)
                } else {
                    white(0.90)
                };
                p.circle(cx, cy, r.h * 0.5, disc);
                let glyph = black(0.82);
                if key.playing {
                    pause_icon(&mut p, cx, cy, icon * 0.85, glyph);
                } else {
                    play_icon(&mut p, cx + 0.06 * icon, cy, icon * 0.8, glyph);
                }
            } else {
                if lit {
                    let wash = if pressed { white(0.26) } else { white(0.15) };
                    p.circle(cx, cy, r.h * 0.5, wash);
                }
                let color = if enabled { white(0.92) } else { white(0.30) };
                skip_icon(
                    &mut p,
                    cx,
                    cy,
                    icon * 0.8,
                    control == SpotifyControl::Next,
                    color,
                );
            }
        }
    }
    if let (Some(track), Some(knob), Some((sx, sy))) = (l.volume, key.volume_x, l.speaker) {
        let lit = matches!(key.lit, Some(SpotifyControl::Volume));
        speaker_icon(&mut p, px(sx), px(sy), 7.0 * s, white(0.78));
        let h = (2.0 * s).round().max(2.0);
        let x0 = px(track.x);
        let w = px(track.right()) - x0;
        let cy = ((track.y + track.h * 0.5) * s).round();
        let bar = Rect::new(x0, cy - h * 0.5, w, h);
        p.round_rect(bar, h * 0.5, Fill::Solid(white(0.16)));
        let filled = (knob as f32 - x0).max(h);
        p.round_rect(
            Rect::new(x0, bar.y, filled, h),
            h * 0.5,
            Fill::Solid(white(0.78)),
        );
        // The menu slider's knob (its drop shadow left out: a few dozen triangles for a
        // shading nobody sees at this size).
        let kr = if lit { 3.6 * s } else { 3.0 * s };
        p.circle(knob as f32, cy, kr, white(0.97));
    }
}

/// A solid triangle wound for the culling UI pipeline (as `Painter` does).
fn triangle(list: &mut DrawList, mut pts: [[f32; 2]; 3], color: Rgba) {
    let cross = (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
        - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]);
    if cross > 0.0 {
        pts.swap(1, 2);
    }
    list.triangle(&UiTexture::Solid, pts, [[0.0, 0.0]; 3], [color; 3]);
}

/// A triangle with anti-aliased edges: the fill plus a hairline along each edge.
fn smooth_triangle(p: &mut Painter<'_>, pts: [[f32; 2]; 3], color: Rgba) {
    triangle(p.list, pts, color);
    let edge = [color[0], color[1], color[2], color[3] * 0.55];
    for i in 0..3 {
        let (a, b) = (pts[i], pts[(i + 1) % 3]);
        p.line(a[0], a[1], b[0], b[1], 0.6, edge);
    }
}

/// ▶ centred on `(cx, cy)`, `size` tall.
fn play_icon(p: &mut Painter<'_>, cx: f32, cy: f32, size: f32, color: Rgba) {
    let h = size;
    let w = size * 0.87;
    smooth_triangle(
        p,
        [
            [cx - w * 0.5, cy - h * 0.5],
            [cx - w * 0.5, cy + h * 0.5],
            [cx + w * 0.5, cy],
        ],
        color,
    );
}

/// ⏸ centred on `(cx, cy)`, `size` tall.
fn pause_icon(p: &mut Painter<'_>, cx: f32, cy: f32, size: f32, color: Rgba) {
    // Whole pixels, no rounding: two crisp bars cost 4 triangles instead of 70.
    let bar = (size * 0.3).max(2.0).round();
    let gap = (size * 0.28).round();
    let (top, h) = ((cy - size * 0.5).round(), size.round());
    for x in [(cx - gap * 0.5).round() - bar, (cx + gap * 0.5).round()] {
        p.rect(Rect::new(x, top, bar, h), color);
    }
}

/// ⏭ (or ⏮ when `!next`) centred on `(cx, cy)`, `size` tall.
fn skip_icon(p: &mut Painter<'_>, cx: f32, cy: f32, size: f32, next: bool, color: Rgba) {
    let h = size;
    let w = size * 0.8;
    let bar = (size * 0.18).max(1.5);
    let dir = if next { 1.0 } else { -1.0 };
    let tip = cx + dir * (w * 0.5 - bar);
    let back = cx - dir * w * 0.5;
    smooth_triangle(
        p,
        [[back, cy - h * 0.5], [back, cy + h * 0.5], [tip, cy]],
        color,
    );
    let bar_x = if next { tip } else { tip - bar };
    p.rect(
        Rect::new(
            bar_x.round(),
            (cy - h * 0.5).round(),
            bar.round(),
            h.round(),
        ),
        color,
    );
}

/// A speaker with a sound wave, `size` tall, top-left at `(x, y)`.
fn speaker_icon(p: &mut Painter<'_>, x: f32, y: f32, size: f32, color: Rgba) {
    let u = size / 7.0;
    p.rect(
        Rect::new(
            x.round(),
            (y + 2.2 * u).round(),
            (2.0 * u).round(),
            (2.6 * u).round(),
        ),
        color,
    );
    smooth_triangle(
        p,
        [
            [x + 1.6 * u, y + 2.4 * u],
            [x + 4.2 * u, y],
            [x + 4.2 * u, y + 7.0 * u],
        ],
        color,
    );
    smooth_triangle(
        p,
        [
            [x + 1.6 * u, y + 2.4 * u],
            [x + 4.2 * u, y + 7.0 * u],
            [x + 1.6 * u, y + 4.6 * u],
        ],
        color,
    );
    p.line(
        x + 5.6 * u,
        y + 2.3 * u,
        x + 5.6 * u,
        y + 4.7 * u,
        (0.8 * u).max(1.0),
        color,
    );
}

/// An eighth note in the middle of `rect` (the missing-cover placeholder).
fn note(p: &mut Painter<'_>, rect: Rect, color: Rgba) {
    let (cx, cy) = rect.center();
    let u = rect.h / 32.0;
    p.circle(cx - 3.0 * u, cy + 5.0 * u, 3.2 * u, color);
    p.rect(
        Rect::new(cx - 0.2 * u, cy - 8.0 * u, 1.4 * u, 13.0 * u),
        color,
    );
    p.line(
        cx + 0.5 * u,
        cy - 7.5 * u,
        cx + 5.0 * u,
        cy - 3.5 * u,
        1.6 * u,
        color,
    );
}

/// The outline of a rounded rectangle grown by `grow` (clockwise from the top-left corner).
fn outline(rect: Rect, radius: f32, grow: f32, out: &mut [[f32; 2]; MAX_OUTLINE]) -> usize {
    let r0 = radius.clamp(0.0, rect.w.min(rect.h) * 0.5);
    let r = (r0 + grow).max(0.0);
    let segments = ((radius * 0.35).ceil() as usize).clamp(2, 12);
    let centres = [
        (rect.x + r0, rect.y + r0, std::f32::consts::PI),
        (rect.right() - r0, rect.y + r0, std::f32::consts::PI * 1.5),
        (rect.right() - r0, rect.bottom() - r0, 0.0),
        (rect.x + r0, rect.bottom() - r0, std::f32::consts::PI * 0.5),
    ];
    let mut n = 0;
    for (cx, cy, start) in centres {
        for i in 0..=segments {
            let a = start + std::f32::consts::FRAC_PI_2 * (i as f32 / segments as f32);
            out[n] = [cx + a.cos() * r, cy + a.sin() * r];
            n += 1;
        }
    }
    n
}

/// An image in a rounded rectangle with an anti-aliased edge: a fan whose UVs follow the
/// positions, plus a one-pixel fringe fading to clear (UVs run past the edge; the cover's
/// sampler clamps).
fn textured_round_rect(list: &mut DrawList, texture: &UiTexture, rect: Rect, radius: f32) {
    if rect.w <= 0.0 || rect.h <= 0.0 {
        return;
    }
    let mut inner = [[0.0; 2]; MAX_OUTLINE];
    let mut outer = [[0.0; 2]; MAX_OUTLINE];
    let n = outline(rect, radius, -0.5, &mut inner);
    outline(rect, radius, 0.5, &mut outer);
    let uv = |p: [f32; 2]| [(p[0] - rect.x) / rect.w, (p[1] - rect.y) / rect.h];
    let solid = white(1.0);
    let clear = white(0.0);
    let (cx, cy) = rect.center();
    let mut emit = |pts: [[f32; 2]; 3], colors: [Rgba; 3]| {
        let mut order = [0, 1, 2];
        let cross = (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
            - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]);
        if cross > 0.0 {
            order = [0, 2, 1];
        }
        list.triangle(
            texture,
            order.map(|i| pts[i]),
            order.map(|i| uv(pts[i])),
            order.map(|i| colors[i]),
        );
    };
    for i in 0..n {
        let j = (i + 1) % n;
        emit([[cx, cy], inner[i], inner[j]], [solid; 3]);
        emit([inner[i], outer[i], inner[j]], [solid, clear, solid]);
        emit([inner[j], outer[i], outer[j]], [solid, clear, clear]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::elements::spotify::{SpotifyElement, SpotifyStyle, MOD_ID};
    use crate::modhud::{HudCtx, HudData, HudElement};
    use crate::widget::test_support::uniform_font;
    use mc_mods::ModSettings;

    fn aero_settings() -> ModSettings {
        let mut settings = ModSettings::new();
        settings.set_enabled(MOD_ID, true);
        settings.set_choice(MOD_ID, "style", 1);
        settings
    }

    fn data() -> HudData {
        HudData {
            now_ms: 5_000,
            spotify: SpotifyHud {
                text_revision: 1,
                cover: true,
                ..SpotifyHud::sample()
            },
            ..HudData::default()
        }
    }

    fn draw(element: &mut SpotifyElement, data: &HudData, settings: &ModSettings) -> DrawList {
        let mut font = uniform_font();
        let ctx = HudCtx {
            data,
            settings,
            screen: (640.0, 360.0),
            pixel_scale: 3.0,
            editing: false,
        };
        element.layout(&ctx, &font).expect("a track is playing");
        let mut d = DrawList::new();
        d.with_transform(Transform::translate(400.0, 250.0), |d| {
            element.draw(&ctx, &mut font, d);
            assert_eq!(d.transform_depth(), 1, "the element pops what it pushes");
        });
        d
    }

    #[test]
    fn the_aero_style_stays_inside_its_rect_and_never_needs_the_scene() {
        let settings = aero_settings();
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        assert_eq!(element.options().style, SpotifyStyle::Aero);
        let d = draw(&mut element, &data(), &settings);
        let (w, h) = element.current_layout().size;
        assert!(!d.is_empty());
        // Positions are back in GUI pixels (the 1/pixel transform is popped), inside the
        // widget plus the half-pixel fringe.
        for v in d.vertices() {
            assert!(
                v.pos[0] >= 400.0 - 0.5 && v.pos[0] <= 400.0 + w + 0.5,
                "{:?}",
                v.pos
            );
            assert!(
                v.pos[1] >= 250.0 - 0.5 && v.pos[1] <= 250.0 + h + 0.5,
                "{:?}",
                v.pos
            );
        }
        assert_eq!(d.transform_depth(), 0, "balanced transforms");
        assert!(d.batches().iter().any(|b| &b.texture == cover_texture()));
    }

    #[test]
    fn shapes_and_text_are_replayed_until_something_changes() {
        let settings = aero_settings();
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        let mut data = data();
        let first = draw(&mut element, &data, &settings);
        let second = draw(&mut element, &data, &settings);
        assert_eq!(
            first, second,
            "an unchanged frame replays the same geometry"
        );
        // A hover changes the shapes; the next frame has more triangles (the hover wash).
        data.spotify.interactive = true;
        data.spotify.hover = Some(SpotifyControl::Next);
        let hovered = draw(&mut element, &data, &settings);
        assert!(hovered.vertices().len() > first.vertices().len());
    }

    #[test]
    fn the_per_frame_geometry_is_small() {
        let settings = aero_settings();
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        let d = draw(&mut element, &data(), &settings);
        let triangles = d.vertices().len() / 3;
        // The cost report quotes this number; keep it honest if the look changes.
        eprintln!(
            "aero: {triangles} triangles per frame, {} batches",
            d.batches().len()
        );
        assert!(triangles < 1_400, "{triangles}");
    }

    /// Per-frame cost of each style against the FPS readout, the cheapest HUD element. Timing
    /// only: `cargo test -p mc-ui --lib per_frame_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn per_frame_cost() {
        use crate::modhud::elements::fps::FpsElement;
        use crate::screens::modmenu::text::MenuText;
        use std::time::Instant;

        const FRAMES: u32 = 3000;
        let mut font = uniform_font();
        let text = AeroText::new(MenuText::load());
        let mut report = |name: &str, element: &mut dyn HudElement, settings: &ModSettings| {
            let mut data = data();
            data.spotify.aero_text = Some(text.clone());
            let mut d = DrawList::new();
            d.enable_solid_merging();
            let mut frame = |d: &mut DrawList, data: &HudData, font: &mut FontRenderer| {
                let ctx = HudCtx {
                    data,
                    settings,
                    screen: (640.0, 360.0),
                    pixel_scale: 3.0,
                    editing: false,
                };
                d.clear();
                element.layout(&ctx, font);
                d.with_transform(Transform::translate(400.0, 250.0), |d| {
                    element.draw(&ctx, font, d);
                });
            };
            for _ in 0..50 {
                frame(&mut d, &data, &mut font);
            }
            let start = Instant::now();
            for i in 0..FRAMES {
                // 3 ms per frame: the progress bar crosses a pixel now and then, the time
                // labels change every 333 frames, as at ~330 FPS.
                data.now_ms += 3;
                data.spotify.position_ms = Some(87_133 + u64::from(i) * 3);
                frame(&mut d, &data, &mut font);
            }
            let per = start.elapsed() / FRAMES;
            eprintln!(
                "{name}: {per:?} per frame, {} vertices, {} batches",
                d.vertices().len(),
                d.batches().len()
            );
        };
        let mut fps_settings = ModSettings::new();
        fps_settings.set_enabled("fps", true);
        let mut fps = FpsElement::default();
        fps.sync(&fps_settings);
        report("fps readout (reference)", &mut fps, &fps_settings);
        let mut mc_settings = ModSettings::new();
        mc_settings.set_enabled(MOD_ID, true);
        let mut mc = SpotifyElement::default();
        mc.sync(&mc_settings);
        report("spotify minecraft", &mut mc, &mc_settings);
        let aero_settings = aero_settings();
        let mut aero = SpotifyElement::default();
        aero.sync(&aero_settings);
        report("spotify aero", &mut aero, &aero_settings);
        let (shapes, glyphs) = aero.aero_geometry();
        eprintln!("aero replays {shapes} shape triangles and {glyphs} glyph quads");
    }

    /// The old per-triangle replay, kept as the reference `DrawList::append` must match.
    fn replay_each(src: &DrawList, d: &mut DrawList) {
        let vertices = src.vertices();
        for batch in src.batches() {
            for t in vertices[batch.range.clone()].chunks_exact(3) {
                d.triangle(
                    &batch.texture,
                    [t[0].pos, t[1].pos, t[2].pos],
                    [t[0].uv, t[1].uv, t[2].uv],
                    [t[0].color, t[1].color, t[2].color],
                );
            }
        }
    }

    #[test]
    fn append_matches_emitting_each_triangle() {
        let mut cached = DrawList::new();
        {
            let mut p = Painter::new(&mut cached);
            p.round_rect(
                Rect::new(0.0, 0.0, 40.0, 20.0),
                6.0,
                Fill::Solid(white(0.5)),
            );
        }
        textured_round_rect(
            &mut cached,
            cover_texture(),
            Rect::new(2.0, 2.0, 16.0, 16.0),
            3.0,
        );
        Painter::new(&mut cached).circle(30.0, 10.0, 4.0, black(0.3));
        assert!(cached.batches().len() >= 3, "solid, cover, solid");
        for merge in [false, true] {
            let mut a = DrawList::new();
            let mut b = DrawList::new();
            for d in [&mut a, &mut b] {
                if merge {
                    d.enable_solid_merging();
                }
                d.rect(0, 0, 5, 5, 0xFF00_FF00);
                d.push_transform(Transform::translate(3.0, 4.0).then_scale(0.5));
            }
            a.append(&cached);
            replay_each(&cached, &mut b);
            a.pop_transform();
            b.pop_transform();
            assert_eq!(a, b, "merge = {merge}");
        }
    }

    #[test]
    fn a_line_segoe_ui_cannot_draw_falls_back_to_the_minecraft_font() {
        let mut segoe = MenuText::load();
        if !segoe.is_segoe() {
            eprintln!("SKIP: Segoe UI is not installed");
            return;
        }
        assert!(segoe.covers("Shining Home - The Glasshouse, Renée"));
        assert!(!segoe.covers("夜に走る"), "Segoe UI has no kana or kanji");
        let mut fallback = MenuText::minecraft_only();
        assert!(fallback.covers("夜に走る"));
        assert!(pick(Some(&mut segoe), &mut fallback, "Shining Home").is_segoe());
        assert!(!pick(Some(&mut segoe), &mut fallback, "夜に走る").is_segoe());
        assert!(!pick(None, &mut fallback, "Shining Home").is_segoe());
    }

    #[test]
    fn a_long_title_is_clipped_to_the_text_column() {
        let settings = aero_settings();
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        let mut data = data();
        data.spotify.title = "A very long title that never fits in the widget".repeat(2);
        data.spotify.text_revision = 7;
        data.now_ms = 50_000;
        let d = draw(&mut element, &data, &settings);
        let l = *element.current_layout();
        let right = 400.0 + l.text_x + l.text_w + 0.01;
        // Glyph quads use the text texture (the MC-font fallback here: font sheets).
        for b in d.batches() {
            if matches!(b.texture, UiTexture::Font(_)) {
                for v in &d.vertices()[b.range.clone()] {
                    assert!(v.pos[0] <= right, "{:?} past {right}", v.pos);
                }
            }
        }
    }
}
