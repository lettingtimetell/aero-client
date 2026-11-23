//! Layout: the scale, the three window frames (Where, What, How), the window chrome, a window's
//! scrolled body ([`Area`]), one frame's `build` (the animation and the glass), the taskbar row
//! under the windows, and hover text.
//!
//! Three windows of 220, 330 and 440
//! design pixels instead of seven of 256, so the height limits the scale instead of the width
//! and the text is about 20% larger at 720p, 1080p and 1440p.

use super::anim;
use super::glass::{GlassFrame, GlassRect};
use super::keybinds::key_name;
use super::paint::{black, white, Fill, Painter, Rect};
use super::state::{Mode, Pane, Target, View};
use super::text::{TextStyle, Weight};
use super::widgets::{draw_pill, draw_pill_body, ms_since, shadowed, PILL_H};
use super::{MenuEnv, ModMenuScreen};
use crate::widget::DrawList;

// ---- design constants (CSS px of the Aero design) ------------------------------------------
/// The windows' widths, left to right: Where (the entries), What (the mods), How (the settings).
pub const WINDOW_WIDTHS: [f32; 3] = [220.0, 330.0, 440.0];
pub(super) const WINDOW_GAP: f32 = 16.0;
/// Space between the outer windows and the screen edge.
pub(super) const SIDE_MARGIN: f32 = 32.0;
/// The height the windows plus the taskbar row are designed to fit.
pub(super) const DESIGN_H: f32 = 900.0;
pub(super) const WINDOW_MAX_H: f32 = 680.0;
pub(super) const WINDOW_RADIUS: f32 = 25.0;
pub(super) const TITLE_H: f32 = 64.0;
pub(super) const BODY_PAD_X: f32 = 13.0;
pub(super) const BODY_PAD_TOP: f32 = 14.0;
pub(super) const BODY_PAD_BOTTOM: f32 = 18.0;
/// The taskbar's "Edit HUD" pill.
pub(super) const EDIT_W: f32 = 132.0;

/// glass.js `uDim`.
pub(super) const MENU_DIM: f32 = 0.35;
/// Mods whose open settings fade the menu dim out, so the game behind the glass shows the real
/// colours being tuned.
pub(super) const UNDIM_WHEN_EXPANDED: &[&str] = &["color_grading"];

/// The design width: the three windows, the gaps between them and a margin on both sides.
#[must_use]
pub fn design_width() -> f32 {
    let windows: f32 = WINDOW_WIDTHS.iter().sum();
    windows + 2.0 * WINDOW_GAP + 2.0 * SIDE_MARGIN
}

/// Physical pixels per design pixel: the three windows with their gaps and side margins fit
/// the width ([`design_width`]), the windows and the taskbar fit the height (at 16:9 the height
/// decides: 0.8 at 1280x720, 1.2 at 1920x1080, 1.6 at 2560x1440).
#[must_use]
pub fn px_per_dip(screen: (f32, f32)) -> f32 {
    let fit = (screen.0 / design_width()).min(screen.1 / DESIGN_H);
    if fit.is_finite() {
        fit.clamp(0.55, 3.0)
    } else {
        1.0
    }
}

/// The three windows in physical pixels, left to right, and the taskbar row's top, for a
/// screen at `s` physical pixels per design pixel.
pub(super) fn window_frames(screen: (f32, f32), s: f32) -> ([Rect; 3], f32) {
    let screen_dip = (screen.0 / s, screen.1 / s);
    let natural: f32 = WINDOW_WIDTHS.iter().sum();
    let gaps = 2.0 * WINDOW_GAP;
    // Narrower windows only when the scale is at its floor (a tiny game window).
    let fit = ((screen_dip.0 - 2.0 * SIDE_MARGIN - gaps) / natural).clamp(0.6, 1.0);
    let total = natural * fit + gaps;
    let window_h = (screen_dip.1 - 150.0).clamp(220.0, WINDOW_MAX_H);
    let left = (screen_dip.0 - total) * 0.5;
    let top = (screen_dip.1 - window_h) * 0.5 - 18.0;
    let mut x = left;
    let frames = WINDOW_WIDTHS.map(|w| {
        let rect = Rect::new(x * s, top * s, w * fit * s, window_h * s);
        x += w * fit + WINDOW_GAP;
        rect
    });
    (frames, (top + window_h + 22.0) * s)
}

/// A window's scrolled body, as [`ModMenuScreen::begin_body`] opens it: where the content starts
/// and how wide it is. Everything is in physical pixels.
#[derive(Debug, Clone, Copy)]
pub(super) struct Area {
    /// The content's left edge.
    pub(super) x: f32,
    /// The content's top, already moved by the window's scroll.
    pub(super) y: f32,
    /// The content's width.
    pub(super) w: f32,
    /// The visible part of the body: hit regions are clipped to it.
    pub(super) clip: Rect,
    /// Physical pixels per design pixel.
    pub(super) s: f32,
    /// Input is live (not closing, not in the HUD editor).
    pub(super) interactive: bool,
}

impl Area {
    /// The content's right edge.
    pub(super) fn right(&self) -> f32 {
        self.x + self.w
    }

    /// `rect` (drawn with `painter`) as a hit region inside the visible body, or `None` when no
    /// part of it shows or input is not live.
    pub(super) fn hit(&self, painter: &Painter<'_>, rect: Rect) -> Option<Rect> {
        if !self.interactive {
            return None;
        }
        let (ox, oy) = painter.offset;
        let r = rect.offset(ox, oy).intersect(&self.clip.offset(ox, oy));
        (r.w > 0.0 && r.h > 0.0).then_some(r)
    }

    /// A content span `(top, bottom)` in physical pixels as design pixels from the content's
    /// top (what [`ModMenuScreen::end_body`] scrolls into view).
    pub(super) fn span(&self, top: f32, bottom: f32) -> (f32, f32) {
        ((top - self.y) / self.s, (bottom - self.y) / self.s)
    }
}

/// The window chrome of the Aero design: shadow, glass gradients, title bar, borders.
pub(super) fn draw_chrome(painter: &mut Painter<'_>, rect: Rect, s: f32) {
    let radius = WINDOW_RADIUS * s;
    // box-shadow: 0 28px 65px rgba(0,0,0,.4), drawn outside the window only.
    painter.round_rect_shadow(
        rect,
        radius,
        [26.0 * s, 6.0 * s, 26.0 * s, 52.0 * s],
        black(0.30),
    );
    // background: linear-gradient(135deg, rgba(255,255,255,.055), transparent 32%),
    //             linear-gradient(180deg, rgba(255,255,255,.025), rgba(0,0,0,.08));
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(rect, 180.0, white(0.025), black(0.08)),
    );
    let corner = Rect::new(rect.x, rect.y, rect.w * 0.64, rect.h * 0.64);
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(corner, 135.0, white(0.055), white(0.0)),
    );
    // ::before: linear-gradient(112deg, rgba(255,255,255,.045), transparent 22%).
    let sheen = Rect::new(rect.x, rect.y, rect.w * 0.44, rect.h * 0.44);
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(sheen, 112.0, white(0.045), white(0.0)),
    );
    // Title bar: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.008)),
    // clipped to the window's rounded top.
    let title = Rect::new(rect.x, rect.y, rect.w, TITLE_H * s);
    painter.set_clip(Some(title));
    painter.round_rect(
        Rect::new(rect.x, rect.y, rect.w, (TITLE_H + WINDOW_RADIUS) * s),
        radius,
        Fill::css_angle(title, 180.0, white(0.055), white(0.008)),
    );
    painter.set_clip(None);
    // Body: .sidebar's linear-gradient(90deg, rgba(0,0,0,.12), rgba(0,0,0,.025)).
    let body_top = rect.y + TITLE_H * s;
    let body = Rect::new(rect.x, body_top, rect.w, rect.bottom() - body_top);
    painter.set_clip(Some(body));
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(body, 90.0, black(0.12), black(0.025)),
    );
    painter.set_clip(None);
    // border-bottom: 1px solid rgba(255,255,255,.1) under the title bar.
    painter.rect(
        Rect::new(rect.x + 1.0, body_top.round(), rect.w - 2.0, 1.0),
        white(0.10),
    );
    // inset 0 1px 0 rgba(255,255,255,.16): the top highlight.
    painter.line(
        rect.x + radius * 0.8,
        rect.y + 1.5,
        rect.right() - radius * 0.8,
        rect.y + 1.5,
        1.0,
        white(0.16),
    );
    // border: 1px, white at about 0.42 (the brief; the design's --line).
    painter.round_rect_stroke(rect, radius, 1.0, 0.5, white(0.42));
}

/// A window's title bar text: the title (shrunk, down to 12 dip, to fit) and the subtitle line
/// under it, leaving `reserve` physical pixels free at the right (a switch, a count).
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_title(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    rect: Rect,
    title: &str,
    subtitle: &str,
    reserve: f32,
    s: f32,
) {
    let pad = 18.0 * s;
    let room = (rect.w - 2.0 * pad - reserve).max(1.0);
    let saved = painter.clip();
    let clip = Rect::new(rect.x + pad - 2.0 * s, rect.y, room + 4.0 * s, TITLE_H * s);
    painter.set_clip(Some(saved.map_or(clip, |c| c.intersect(&clip))));
    let mut style = TextStyle::new(15.0 * s).weight(Weight::Semibold);
    let natural = env.text.measure(title, style, env.mc_font);
    if natural > room {
        style.size = (style.size * room / natural).max(12.0 * s);
    }
    shadowed(
        painter,
        env,
        title,
        rect.x + pad,
        rect.y + 14.0 * s,
        style,
        white(1.0),
        s,
    );
    shadowed(
        painter,
        env,
        subtitle,
        rect.x + pad,
        rect.y + 35.0 * s,
        TextStyle::new(11.0 * s),
        white(0.62),
        s,
    );
    painter.set_clip(saved);
}

impl ModMenuScreen {
    // ---- drawing ---------------------------------------------------------------------

    /// Draw one frame into `list` (physical pixels) and return the glass for it.
    pub fn build(&mut self, env: &mut MenuEnv<'_>, list: &mut DrawList) -> GlassFrame {
        self.hits.clear();
        for rows in &mut self.nav {
            rows.clear();
        }
        self.tooltip_at = None;
        self.search_found = 0;
        self.best = None;
        self.no_world = env.game.no_world;
        let now = env.now;
        let dt = self.last_frame.map_or(0.0, |last| {
            now.saturating_duration_since(last).as_secs_f32()
        });
        self.last_frame = Some(now);
        let catalog = env.settings.catalog();
        if self.switch_anim.len() != catalog.len() {
            self.switch_anim = (0..catalog.len())
                .map(|m| if env.settings.enabled_at(m) { 1.0 } else { 0.0 })
                .collect();
        }
        let k = anim::approach(dt, 22.0);
        for (m, t) in self.switch_anim.iter_mut().enumerate() {
            let target = if env.settings.enabled_at(m) { 1.0 } else { 0.0 };
            *t += (target - *t) * k;
            if (target - *t).abs() < 0.002 {
                *t = target;
            }
        }
        self.refresh_changed(env.settings);
        // The first open of a session: My setup, or HUD while My setup is empty.
        if self.view.is_none() {
            let view = if self.my_setup_len(env.settings) > 0 {
                View::MySetup
            } else {
                View::Category(0)
            };
            self.view = Some(view);
        }
        if self.selected.is_none() {
            self.selected = self.first_in_view(env.settings);
            self.reveal = self.selected;
        }
        // While Color Grading's settings are open the dim fades out, so the game
        // behind the glass shows its real (graded) colours, and back in when they close. Same
        // easing as the switches.
        let undim_target = match self.selected.and_then(|m| catalog.get(m)) {
            Some(descriptor) if UNDIM_WHEN_EXPANDED.contains(&descriptor.id) => 1.0,
            _ => 0.0,
        };
        self.undim += (undim_target - self.undim) * k;
        if (undim_target - self.undim).abs() < 0.002 {
            self.undim = undim_target;
        }
        // A new search starts the list at the top, where its best result is.
        if self.search != self.search_seen {
            self.search_seen.clone_from(&self.search);
            let list_pane = Pane::List.index();
            self.scroll_target[list_pane] = 0.0;
            self.cursors[list_pane] = None;
            if !self.searching() {
                self.narrow = None;
            }
        }
        self.refresh_found(env);
        let ks = anim::approach(dt, 18.0);
        for w in 0..Pane::ALL.len() {
            self.scroll_target[w] = self.scroll_target[w].clamp(0.0, self.max_scroll[w]);
            self.scroll[w] += (self.scroll_target[w] - self.scroll[w]) * ks;
        }

        let s = px_per_dip(env.screen);
        let mut glass = GlassFrame {
            px_per_dip: s,
            time: env.time_s,
            ..GlassFrame::default()
        };
        let closing = self.closing_since.map(|since| ms_since(since, now));
        let mut painter = Painter::new(list);

        // Window visibility: open/close animation, and fading out into the HUD editor.
        let editor_ms = ms_since(self.mode_since, now);
        let menu_visible = match self.mode {
            Mode::Menu => 1.0,
            Mode::HudEditor => anim::window_close_progress(editor_ms, 0, 1),
        };
        let dim_progress = match closing {
            Some(ms) => anim::dim_close_progress(ms),
            None => anim::dim_open_progress(ms_since(self.opened_at, now)),
        };
        glass.dim = MENU_DIM * dim_progress * menu_visible * (1.0 - self.undim);
        glass.strength = dim_progress;

        let (frames, taskbar_y) = window_frames(env.screen, s);
        let count = frames.len();
        if menu_visible > 0.0 {
            for (w, frame) in frames.into_iter().enumerate() {
                let progress = match closing {
                    Some(ms) => anim::window_close_progress(ms, w, count),
                    None => anim::window_open_progress(ms_since(self.opened_at, now), w),
                } * menu_visible;
                if progress <= 0.0 {
                    continue;
                }
                let slide = (1.0 - progress) * anim::SLIDE_DIP * s;
                let rect = Rect::new(frame.x, frame.y + slide, frame.w, frame.h);
                glass.rects.push(GlassRect {
                    rect: [rect.x, rect.y, rect.w, rect.h],
                    radius: WINDOW_RADIUS * s,
                    alpha: progress,
                });
                painter.alpha = progress;
                draw_chrome(&mut painter, rect, s);
                match Pane::ALL[w] {
                    Pane::Sidebar => self.draw_sidebar(&mut painter, env, rect, s),
                    Pane::List => self.draw_list(&mut painter, env, rect, s),
                    Pane::Details => self.draw_details(&mut painter, env, rect, s),
                }
                painter.set_clip(None);
            }
            // The Aero taskbar under the windows: "Edit HUD" and how to close.
            painter.alpha = dim_progress * menu_visible;
            self.draw_taskbar(&mut painter, env, &mut glass, taskbar_y, s);
            if self.mode == Mode::Menu && closing.is_none() {
                self.draw_tooltip(&mut painter, env, s);
            }
        }

        if self.mode == Mode::HudEditor && closing.is_none() {
            painter.alpha = 1.0;
            painter.set_clip(None);
            self.draw_editor(&mut painter, env, s);
        }
        glass
    }

    /// Input is live: the menu is up (not closing, not in the HUD editor).
    pub(super) fn interactive(&self) -> bool {
        self.closing_since.is_none() && self.mode == Mode::Menu
    }

    /// Start a window's scrolled body: clip to it, record its scroll region, and return where
    /// its content starts.
    pub(super) fn begin_body(
        &mut self,
        painter: &mut Painter<'_>,
        rect: Rect,
        pane: Pane,
        s: f32,
    ) -> Area {
        let interactive = self.interactive();
        let body_top = rect.y + TITLE_H * s;
        let clip = Rect::new(
            rect.x + 1.0,
            body_top + 1.0,
            rect.w - 2.0,
            rect.bottom() - body_top - 6.0 * s,
        );
        if interactive {
            self.hits.push((
                clip.offset(painter.offset.0, painter.offset.1),
                Target::Window(pane.index()),
            ));
        }
        painter.set_clip(Some(clip));
        let scroll = self.scroll[pane.index()] * s;
        Area {
            x: rect.x + BODY_PAD_X * s,
            y: body_top + BODY_PAD_TOP * s - scroll,
            w: rect.w - BODY_PAD_X * 2.0 * s,
            clip,
            s,
            interactive,
        }
    }

    /// End a window's body. `bottom` is the y below its last content. Sets how far the window
    /// scrolls, scrolls `reveal` (a span from [`Area::span`]) into view once, and draws a thin
    /// scrollbar when the content is taller than the window.
    pub(super) fn end_body(
        &mut self,
        painter: &mut Painter<'_>,
        pane: Pane,
        area: Area,
        bottom: f32,
        reveal: Option<(f32, f32)>,
    ) {
        let s = area.s;
        let w = pane.index();
        let content_h = (bottom - area.y) / s + BODY_PAD_BOTTOM;
        let visible_h = area.clip.h / s;
        self.max_scroll[w] = (content_h - visible_h).max(0.0);
        if let Some((top, bottom)) = reveal {
            let target = &mut self.scroll_target[w];
            // A little air around the row, more below it (it reads as "more follows").
            let top = (top - 12.0).max(0.0);
            let bottom = bottom + 28.0;
            if bottom - *target > visible_h {
                *target = top.min(bottom - visible_h);
            }
            if top < *target {
                *target = top;
            }
            *target = target.clamp(0.0, self.max_scroll[w]);
        }
        if self.max_scroll[w] > 0.0 {
            let clip = area.clip;
            let track_h = clip.h - 16.0 * s;
            let thumb_h = (track_h * visible_h / content_h).max(24.0 * s);
            let t = (self.scroll[w] / self.max_scroll[w]).clamp(0.0, 1.0);
            let thumb_y = clip.y + 8.0 * s + (track_h - thumb_h) * t;
            painter.round_rect(
                Rect::new(clip.right() - 5.0 * s, thumb_y, 3.0 * s, thumb_h),
                1.5 * s,
                Fill::Solid(white(0.28)),
            );
        }
        painter.set_clip(None);
    }

    /// The taskbar row at physical `y`: the "Edit HUD" pill (a glass pane), centred under the
    /// windows, and how to close. Skipped when it would not fit on screen. Without a world the
    /// pill is greyed out and says why on hover.
    fn draw_taskbar(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        glass: &mut GlassFrame,
        y: f32,
        s: f32,
    ) {
        let h = PILL_H * s;
        if y + h >= env.screen.1 {
            return;
        }
        let interactive = self.interactive();
        let edit_w = EDIT_W * s;
        let edit = Rect::new((env.screen.0 - edit_w) * 0.5, y, edit_w, h);
        glass.rects.push(GlassRect {
            rect: [edit.x, edit.y, edit.w, edit.h],
            radius: 18.0 * s,
            alpha: painter.alpha,
        });
        let hovered = interactive && edit.contains(self.cursor.0, self.cursor.1);
        let label = if self.return_to_hud {
            "Back to HUD"
        } else {
            "Edit HUD"
        };
        let label_style = TextStyle::new(12.5 * s).weight(Weight::Semibold);
        if self.no_world {
            draw_pill_body(painter, edit, false, s);
            let tw = env.text.measure(label, label_style, env.mc_font);
            let (_, lh) = env.text.metrics(label_style);
            shadowed(
                painter,
                env,
                label,
                edit.x + (edit.w - tw) * 0.5,
                edit.y + (edit.h - lh) * 0.5,
                label_style,
                white(0.38),
                s,
            );
            if hovered {
                self.set_tooltip(
                    "Edit HUD needs a world. Join a server, then open the menu again.",
                    self.cursor,
                );
            }
        } else {
            draw_pill(painter, env, edit, label, hovered, s);
            if interactive {
                self.hits.push((edit, Target::EditHud));
            }
        }
        // How to close and go back, right of the pill.
        let style = TextStyle::new(11.5 * s);
        let (_, lh) = env.text.metrics(style);
        let hint = format!(
            "{} closes  \u{b7}  Esc goes back",
            key_name(env.settings.menu_key())
        );
        shadowed(
            painter,
            env,
            &hint,
            edit.right() + 18.0 * s,
            y + (h - lh) * 0.5,
            style,
            white(0.55),
            s,
        );
    }

    /// Hover text for this frame, drawn on top of everything. The first caller in a frame wins.
    pub(super) fn set_tooltip(&mut self, text: &str, at: (f32, f32)) {
        if self.tooltip_at.is_none() && self.mode == Mode::Menu && self.drag.is_none() {
            self.tooltip.clear();
            self.tooltip.push_str(text);
            self.tooltip_at = Some(at);
        }
    }

    fn draw_tooltip(&mut self, painter: &mut Painter<'_>, env: &mut MenuEnv<'_>, s: f32) {
        let Some((x, y)) = self.tooltip_at else {
            return;
        };
        let text = std::mem::take(&mut self.tooltip);
        let style = TextStyle::new(11.0 * s);
        let (_, lh) = env.text.metrics(style);
        let max_w = 300.0 * s;
        let mut lines: Vec<&str> = Vec::new();
        let mut rest = text.trim();
        while !rest.is_empty() && lines.len() < 8 {
            let mut cut = env
                .text
                .fit_words(rest, max_w, style, env.mc_font)
                .min(rest.len());
            while cut > 0 && !rest.is_char_boundary(cut) {
                cut -= 1;
            }
            if cut == 0 {
                cut = rest.chars().next().map_or(rest.len(), char::len_utf8);
            }
            let (line, tail) = rest.split_at(cut);
            lines.push(line);
            rest = tail.trim_start();
        }
        let widest = lines
            .iter()
            .map(|line| env.text.measure(line, style, env.mc_font))
            .fold(0.0_f32, f32::max);
        let (w, h) = (
            widest + 16.0 * s,
            lines.len() as f32 * (lh + 2.0 * s) + 10.0 * s,
        );
        let mut bx = x + 14.0 * s;
        let mut by = y + 18.0 * s;
        if bx + w > env.screen.0 - 4.0 {
            bx = (x - w - 8.0 * s).max(4.0);
        }
        if by + h > env.screen.1 - 4.0 {
            by = (y - h - 8.0 * s).max(4.0);
        }
        let r = Rect::new(bx, by, w, h);
        painter.set_clip(None);
        painter.alpha = 1.0;
        painter.round_rect_shadow(r, 7.0 * s, [6.0 * s; 4], black(0.25));
        painter.round_rect(r, 7.0 * s, Fill::Solid(black(0.80)));
        painter.round_rect_stroke(r, 7.0 * s, 1.0, 0.5, white(0.22));
        for (i, line) in lines.iter().enumerate() {
            env.text.draw(
                painter,
                env.mc_font,
                line,
                r.x + 8.0 * s,
                r.y + 5.0 * s + i as f32 * (lh + 2.0 * s),
                style,
                white(0.92),
            );
        }
        drop(lines);
        self.tooltip = text;
    }
}
