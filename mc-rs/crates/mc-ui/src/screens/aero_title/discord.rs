//! The Discord button on the Aero title screen: the Discord logo and "Discord" on a small Aero
//! glass pill at the bottom right, just above vanilla's copyright line and right-aligned with it.
//! A click opens the Aero Client Discord server ([`DISCORD_URL`]) in the default browser.
//!
//! It pops in with the footer and wears the title screen's glass (the buttons' tint, sheen,
//! border and halo'd white marks); hovered, its glass turns Discord's Blurple, and pressed it
//! sinks in like the buttons. A click is a press and a release on it
//! ([`DiscordButton::mouse_released`]); the client opens the link. It never covers anything: it
//! has to sit below the button column (and so below the title and the account corner) and clear
//! of the update card, and in a window too small for that it is not drawn and takes no clicks.
//!
//! The logo is the Simple Icons project's `discord` path (CC0 1.0), unmodified, drawn in white as
//! Discord's brand guidelines allow, and filled with [`PathMesh`], tessellated once per size.
//! Discord and the Discord logo are trademarks of Discord Inc.

use std::f32::consts::TAU;
use std::sync::OnceLock;

use super::layout::Layout;
use super::update_notice::FOOTER_GAP;
use super::{haloed, mix, AeroButton, AeroTitleEnv, Layer, CRISP_ALPHA};
use crate::screens::modmenu::anim;
use crate::screens::modmenu::glass::{GlassFrame, GlassRect};
use crate::screens::modmenu::paint::{black, css, white, Fill, Painter, PathMesh, Rect, Rgba};
use crate::screens::modmenu::text::{TextStyle, Weight};

/// The link: the Aero Client Discord server.
pub const DISCORD_URL: &str = "https://discord.gg/kAtcMRwffy";
/// The label next to the logo.
pub const LABEL: &str = "Discord";
/// Discord's Blurple, `#5865F2`.
pub const BLURPLE: Rgba = css(0x58, 0x65, 0xF2, 1.0);

/// The pill's height, in design pixels (1080p, scaled like the rest of the screen).
pub const HEIGHT: f32 = 40.0;
/// The logo's width; its height follows the mark's proportions.
pub const LOGO_W: f32 = 22.0;
/// From the pill's left edge to the logo.
pub const PAD_LEFT: f32 = 14.0;
/// Between the logo and the label.
const GAP: f32 = 9.0;
/// After the label.
const PAD_RIGHT: f32 = 16.0;
/// The label's em size.
const LABEL_SIZE: f32 = 15.0;
/// The corner radius: the buttons' 16 at 56 tall, in proportion.
const RADIUS: f32 = 12.0;
/// The least space between the pill and the button column or the update card.
const CLEARANCE: f32 = 10.0;
/// The hover and press easing rate, as the buttons'.
const HIGHLIGHT_RATE: f32 = 20.0;
/// The size of the steps the logo's curves are cut into, in its grid's units (a 24-unit grid
/// drawn 88 pixels wide at the largest scale: a third of a pixel).
const FLATTEN_STEP: f32 = 0.09;

/// The Simple Icons `discord` path (CC0 1.0), verbatim, on its 24-unit grid.
const LOGO_PATH: &str = "M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z";

/// The pill's place this frame: right-aligned with the copyright line (the window's width less
/// the title's margin) and as far above it as the update card is above the version line, `size`
/// being its width and height in pixels and `footer_h` the footer line's height. `None` when
/// there is no room: it must lie inside the window, below the lowest button (the title and the
/// account corner are above that) and clear of the update card, `card`, by a small margin.
#[must_use]
pub fn place(layout: &Layout, size: (f32, f32), footer_h: f32, card: Option<Rect>) -> Option<Rect> {
    let s = layout.s;
    let right = layout.screen.0 - layout.left;
    let bottom = layout.footer_bottom - footer_h - FOOTER_GAP * s;
    let (w, h) = (size.0.round(), size.1.round());
    let rect = Rect::new((right - w).round(), (bottom - h).round(), w, h);
    let clearance = CLEARANCE * s;
    let lowest = layout
        .buttons
        .iter()
        .map(Rect::bottom)
        .fold(f32::NEG_INFINITY, f32::max);
    let apart = |other: Rect| {
        let overlap = rect.inset(-clearance).intersect(&other);
        overlap.w <= 0.0 || overlap.h <= 0.0
    };
    let fits = rect.x >= 0.0 && rect.y >= lowest + clearance && card.is_none_or(apart);
    fits.then_some(rect)
}

/// The Discord button's state on one title screen.
#[derive(Debug, Default)]
pub struct DiscordButton {
    /// Where it was in the last frame (physical pixels).
    rect: Rect,
    /// Whether it was on screen in the last frame: popping in or in, with room.
    shown: bool,
    /// A left press landed on it and waits for its release.
    pressed: bool,
    /// Eased hover highlight, 0..1.
    hover: f32,
    /// Eased press highlight, 0..1.
    press: f32,
    /// The logo tessellated for its size in the last frame (pixels per grid unit), rebuilt when
    /// the window's size changes.
    logo: Option<(f32, PathMesh)>,
}

impl DiscordButton {
    /// The button's rectangle in the last frame, while it is on screen.
    #[must_use]
    pub fn rect(&self) -> Option<Rect> {
        self.shown.then_some(self.rect)
    }

    /// Whether `(x, y)` is on the button (once it has started to appear).
    #[must_use]
    pub fn hit(&self, x: f32, y: f32) -> bool {
        self.shown && self.rect.contains(x, y)
    }

    /// Whether a press on the button is waiting for its release.
    #[must_use]
    pub fn has_press(&self) -> bool {
        self.pressed
    }

    /// A mouse button went down at `(x, y)`: whether the button took it. Every press on it is
    /// its own; only the left button's can become a click.
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8) -> bool {
        if !self.hit(x, y) {
            return false;
        }
        self.pressed = button == 0;
        true
    }

    /// A mouse button went up at `(x, y)`: whether that completes a click (the left button,
    /// pressed and released on the button).
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8) -> bool {
        if button != 0 {
            return false;
        }
        std::mem::take(&mut self.pressed) && self.hit(x, y)
    }

    /// Draw this frame's button (called by `AeroTitleScreen::build_layers` after the update
    /// card, whose rectangle this frame is `card`). `clock` is the screen's `(elapsed ms since
    /// its pop-in, seconds since the last frame, cursor)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        &mut self,
        painter: &mut Painter<'_>,
        crisp: Option<&mut Painter<'_>>,
        env: &mut AeroTitleEnv<'_>,
        layout: &Layout,
        glass: &mut GlassFrame,
        (elapsed, dt, cursor): (f32, f32, (f32, f32)),
        card: Option<Rect>,
    ) {
        let s = layout.s;
        let label_style = TextStyle::new(LABEL_SIZE * s).weight(Weight::Semibold);
        let (_, label_h) = env.text.metrics(label_style);
        let (_, footer_h) = env
            .text
            .metrics(TextStyle::new(14.0 * s).weight(Weight::Semibold));
        let label_w = env.text.measure(LABEL, label_style, env.mc_font);
        let size = (
            (PAD_LEFT + LOGO_W + GAP + PAD_RIGHT) * s + label_w,
            HEIGHT * s,
        );
        let placed = place(layout, size, footer_h, card);
        // With the footer, the last element of the pop-in.
        let progress = anim::window_open_progress(elapsed, AeroButton::COUNT);
        self.shown = placed.is_some() && progress > 0.0;
        if let Some(rect) = placed {
            self.rect = rect;
        }

        let over = self.shown && self.rect.contains(cursor.0, cursor.1);
        let k = anim::approach(dt, HIGHLIGHT_RATE);
        for (value, target) in [
            (&mut self.hover, if over { 1.0 } else { 0.0 }),
            (
                &mut self.press,
                if over && self.pressed { 1.0 } else { 0.0 },
            ),
        ] {
            *value += (target - *value) * k;
            if (target - *value).abs() < 0.002 {
                *value = target;
            }
        }
        if !self.shown {
            return;
        }

        let logo = logo();
        let scale = LOGO_W * s / logo.size[0];
        if self
            .logo
            .as_ref()
            .is_none_or(|(built, _)| (built - scale).abs() > 1e-4)
        {
            self.logo = Some((scale, logo.mesh(scale)));
        }
        let Some((_, mesh)) = self.logo.as_ref() else {
            return;
        };

        let slide = (1.0 - progress) * anim::SLIDE_DIP * s;
        let body = self.rect.inset(self.press * 1.5 * s);
        let radius = RADIUS * s;
        glass.rects.push(GlassRect {
            rect: [body.x, body.y + slide, body.w, body.h],
            radius,
            alpha: progress,
        });
        let marks = Marks {
            body,
            radius,
            hover: self.hover,
            logo: mesh,
            logo_centre: (
                body.x + (PAD_LEFT + LOGO_W * 0.5) * s,
                body.y + body.h * 0.5,
            ),
            label: (
                body.x + (PAD_LEFT + LOGO_W + GAP) * s,
                (body.y + (body.h - label_h) * 0.5).round(),
                label_style,
            ),
            s,
        };
        painter.alpha = progress;
        painter.offset = (0.0, slide);
        draw_body(painter, body, radius, (self.hover, self.press), s);
        marks.draw(painter, env, Layer::Full);
        if let Some(crisp) = crisp {
            crisp.alpha = progress * CRISP_ALPHA;
            crisp.offset = (0.0, slide);
            marks.draw(crisp, env, Layer::Crisp);
        }
    }
}

/// The pill's body: the title buttons' chrome (drop shadow, a dark tint that keeps white marks
/// readable over the bright scene, the 135-degree sheen), its tint easing to Blurple on hover
/// and to a deeper Blurple when pressed.
fn draw_body(
    painter: &mut Painter<'_>,
    body: Rect,
    radius: f32,
    (hover, press): (f32, f32),
    s: f32,
) {
    painter.round_rect_shadow(
        body,
        radius,
        [12.0 * s, 4.0 * s, 12.0 * s, 20.0 * s],
        black(0.20),
    );
    let [r, g, b, _] = BLURPLE;
    let top = mix(
        mix(black(0.22), [r, g, b, 0.80], hover),
        css(0x47, 0x52, 0xC9, 0.90),
        press,
    );
    let bottom = mix(
        mix(black(0.34), css(0x4B, 0x57, 0xD6, 0.88), hover),
        css(0x3A, 0x44, 0xAA, 0.94),
        press,
    );
    painter.round_rect(body, radius, Fill::css_angle(body, 180.0, top, bottom));
    let corner = Rect::new(body.x, body.y, body.w * 0.6, body.h * 1.4);
    painter.round_rect(
        body,
        radius,
        Fill::css_angle(corner, 135.0, white(0.10 + 0.06 * hover), white(0.0)),
    );
}

/// The pill's marks: drawn once under the greenhouse's VHS finish and once, crisp, over it.
struct Marks<'a> {
    body: Rect,
    radius: f32,
    hover: f32,
    logo: &'a PathMesh,
    logo_centre: (f32, f32),
    /// The label's left edge, top and style.
    label: (f32, f32, TextStyle),
    s: f32,
}

impl Marks<'_> {
    fn draw(&self, painter: &mut Painter<'_>, env: &mut AeroTitleEnv<'_>, layer: Layer) {
        let Marks {
            body,
            radius,
            hover,
            s,
            ..
        } = *self;
        // inset 0 1px 0 rgba(255,255,255,.2): the top highlight, and the 1 px light border.
        painter.line(
            body.x + radius * 0.8,
            body.y + 1.5,
            body.right() - radius * 0.8,
            body.y + 1.5,
            1.0,
            white(0.26 + 0.14 * hover),
        );
        painter.round_rect_stroke(body, radius, 1.0, 0.5, white(0.40 + 0.30 * hover));
        // The mark as it is, plain white: no halo or shadow (the brand's rules), the dark glass
        // behind it is its contrast.
        painter.fill_mesh(self.logo, self.logo_centre, white(1.0));
        let (x, y, style) = self.label;
        let color = white(0.95 + 0.05 * hover);
        match layer {
            Layer::Full => haloed(painter, env, LABEL, x, y, style, color, (s, 0.18)),
            Layer::Crisp => {
                env.text
                    .draw(painter, env.mc_font, LABEL, x, y, style, color);
            }
        }
    }
}

/// The Discord logo, parsed once from [`LOGO_PATH`].
struct Logo {
    /// The outline and the two eyes, on the 24-unit grid, moved so that the mark's bounding box
    /// is centred on the origin.
    outlines: Vec<Vec<[f32; 2]>>,
    /// The bounding box's centre in the path's own coordinates (what the move took away; tests
    /// use it to find points of the mark).
    #[cfg(test)]
    centre: [f32; 2],
    /// The mark's width and height in grid units.
    size: [f32; 2],
}

impl Logo {
    fn parse(path: &str) -> Logo {
        let mut outlines = parse_path(path);
        let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in outlines.iter().flatten() {
            min = [min[0].min(p[0]), min[1].min(p[1])];
            max = [max[0].max(p[0]), max[1].max(p[1])];
        }
        let centre = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
        for p in outlines.iter_mut().flatten() {
            *p = [p[0] - centre[0], p[1] - centre[1]];
        }
        Logo {
            outlines,
            #[cfg(test)]
            centre,
            size: [max[0] - min[0], max[1] - min[1]],
        }
    }

    /// The logo filled at `scale` pixels per grid unit, centred on the origin.
    fn mesh(&self, scale: f32) -> PathMesh {
        let outlines: Vec<Vec<[f32; 2]>> = self
            .outlines
            .iter()
            .map(|outline| {
                outline
                    .iter()
                    .map(|p| [p[0] * scale, p[1] * scale])
                    .collect()
            })
            .collect();
        PathMesh::new(&outlines)
    }
}

fn logo() -> &'static Logo {
    static LOGO: OnceLock<Logo> = OnceLock::new();
    LOGO.get_or_init(|| Logo::parse(LOGO_PATH))
}

/// An SVG path's `d` attribute as closed polygons: lines kept, cubic curves and elliptical arcs
/// cut into short steps. Understands M, L, H, V, C, A and Z, absolute and relative, with the
/// implicit repeats and the compact number syntax (`.0741.0741` is two numbers, `00-4.88` two
/// arc flags and a number). Stops at anything else.
fn parse_path(d: &str) -> Vec<Vec<[f32; 2]>> {
    let mut tokens = PathTokens {
        bytes: d.as_bytes(),
        at: 0,
    };
    let mut outlines = Vec::new();
    let mut points: Vec<[f32; 2]> = Vec::new();
    let (mut pen, mut start) = ([0.0_f32; 2], [0.0_f32; 2]);
    let mut command = 0_u8;
    while !tokens.done() {
        if let Some(letter) = tokens.command() {
            command = letter;
            if matches!(letter, b'Z' | b'z') {
                close(&mut points, &mut outlines);
                pen = start;
                continue;
            }
        }
        let relative = command.is_ascii_lowercase();
        let base = if relative { pen } else { [0.0, 0.0] };
        let point = |tokens: &mut PathTokens<'_>| -> Option<[f32; 2]> {
            Some([base[0] + tokens.number()?, base[1] + tokens.number()?])
        };
        match command.to_ascii_uppercase() {
            b'M' => {
                let Some(p) = point(&mut tokens) else { break };
                close(&mut points, &mut outlines);
                points.push(p);
                (pen, start) = (p, p);
                // More pairs after a move are lines.
                command = if relative { b'l' } else { b'L' };
            }
            b'L' => {
                let Some(p) = point(&mut tokens) else { break };
                points.push(p);
                pen = p;
            }
            b'H' => {
                let Some(x) = tokens.number() else { break };
                pen = [if relative { pen[0] + x } else { x }, pen[1]];
                points.push(pen);
            }
            b'V' => {
                let Some(y) = tokens.number() else { break };
                pen = [pen[0], if relative { pen[1] + y } else { y }];
                points.push(pen);
            }
            b'C' => {
                let (Some(c1), Some(c2), Some(end)) =
                    (point(&mut tokens), point(&mut tokens), point(&mut tokens))
                else {
                    break;
                };
                flatten_cubic(&mut points, [pen, c1, c2, end]);
                pen = end;
            }
            b'A' => {
                let (Some(rx), Some(ry), Some(rotation), Some(large), Some(sweep)) = (
                    tokens.number(),
                    tokens.number(),
                    tokens.number(),
                    tokens.flag(),
                    tokens.flag(),
                ) else {
                    break;
                };
                let Some(end) = point(&mut tokens) else { break };
                flatten_arc(&mut points, (pen, end), [rx, ry], rotation, (large, sweep));
                pen = end;
            }
            _ => break,
        }
    }
    close(&mut points, &mut outlines);
    outlines
}

/// End the polygon in `points` (kept when it has an area).
fn close(points: &mut Vec<[f32; 2]>, outlines: &mut Vec<Vec<[f32; 2]>>) {
    if points.len() >= 3 {
        outlines.push(std::mem::take(points));
    } else {
        points.clear();
    }
}

/// The tokens of a path's `d` attribute.
struct PathTokens<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl PathTokens<'_> {
    fn skip_separators(&mut self) {
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_whitespace() || *b == b',')
        {
            self.at += 1;
        }
    }

    fn done(&mut self) -> bool {
        self.skip_separators();
        self.at >= self.bytes.len()
    }

    /// A command letter, if one is next.
    fn command(&mut self) -> Option<u8> {
        self.skip_separators();
        let letter = *self.bytes.get(self.at)?;
        if !letter.is_ascii_alphabetic() || matches!(letter, b'e' | b'E') {
            return None;
        }
        self.at += 1;
        Some(letter)
    }

    /// A number: `12`, `-.0785`, `1.2495`, `2e-3`.
    fn number(&mut self) -> Option<f32> {
        self.skip_separators();
        let b = self.bytes;
        let start = self.at;
        let mut i = start;
        let digits = |i: &mut usize| {
            let from = *i;
            while b.get(*i).is_some_and(u8::is_ascii_digit) {
                *i += 1;
            }
            *i - from
        };
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let mut count = digits(&mut i);
        if b.get(i) == Some(&b'.') {
            i += 1;
            count += digits(&mut i);
        }
        if count == 0 {
            return None;
        }
        if matches!(b.get(i), Some(b'e' | b'E')) {
            let mut j = i + 1;
            if matches!(b.get(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if digits(&mut j) > 0 {
                i = j;
            }
        }
        self.at = i;
        std::str::from_utf8(&b[start..i]).ok()?.parse().ok()
    }

    /// An arc flag: a single `0` or `1`, which may run straight into what follows.
    fn flag(&mut self) -> Option<bool> {
        self.skip_separators();
        let flag = match self.bytes.get(self.at)? {
            b'0' => false,
            b'1' => true,
            _ => return None,
        };
        self.at += 1;
        Some(flag)
    }
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

/// The cubic curve `[start, c1, c2, end]` in steps of about [`FLATTEN_STEP`] (`start` is already
/// in `points`).
fn flatten_cubic(points: &mut Vec<[f32; 2]>, [p0, p1, p2, p3]: [[f32; 2]; 4]) {
    // The control polygon is never shorter than the curve.
    let hull = distance(p0, p1) + distance(p1, p2) + distance(p2, p3);
    let steps = (hull / FLATTEN_STEP).ceil().clamp(1.0, 96.0) as usize;
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        let u = 1.0 - t;
        let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
        points.push([
            w[0] * p0[0] + w[1] * p1[0] + w[2] * p2[0] + w[3] * p3[0],
            w[0] * p0[1] + w[1] * p1[1] + w[2] * p2[1] + w[3] * p3[1],
        ]);
    }
}

/// The elliptical arc from `from` to `to` with `radii`, the x axis turned `rotation` degrees and
/// the `(large arc, sweep)` flags, in steps of about [`FLATTEN_STEP`] (`from` is already in
/// `points`). SVG's endpoint to centre conversion (SVG 1.1, appendix F.6.5), radii too small to
/// reach scaled up as SVG does.
fn flatten_arc(
    points: &mut Vec<[f32; 2]>,
    (from, to): ([f32; 2], [f32; 2]),
    radii: [f32; 2],
    rotation: f32,
    (large, sweep): (bool, bool),
) {
    if distance(from, to) < 1e-7 {
        return;
    }
    let (mut rx, mut ry) = (radii[0].abs(), radii[1].abs());
    if rx < 1e-7 || ry < 1e-7 {
        points.push(to);
        return;
    }
    let (sin, cos) = rotation.to_radians().sin_cos();
    let (hx, hy) = ((from[0] - to[0]) * 0.5, (from[1] - to[1]) * 0.5);
    let (x1, y1) = (cos * hx + sin * hy, -sin * hx + cos * hy);
    let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry);
    if lambda > 1.0 {
        rx *= lambda.sqrt();
        ry *= lambda.sqrt();
    }
    let (rx2, ry2) = (rx * rx, ry * ry);
    let denominator = rx2 * y1 * y1 + ry2 * x1 * x1;
    let numerator = (rx2 * ry2 - denominator).max(0.0);
    let mut coef = if denominator > 0.0 {
        (numerator / denominator).sqrt()
    } else {
        0.0
    };
    if large == sweep {
        coef = -coef;
    }
    let (cx1, cy1) = (coef * rx * y1 / ry, -coef * ry * x1 / rx);
    let centre = [
        cos * cx1 - sin * cy1 + (from[0] + to[0]) * 0.5,
        sin * cx1 + cos * cy1 + (from[1] + to[1]) * 0.5,
    ];
    let angle =
        |u: [f32; 2], v: [f32; 2]| (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1]);
    let u = [(x1 - cx1) / rx, (y1 - cy1) / ry];
    let v = [(-x1 - cx1) / rx, (-y1 - cy1) / ry];
    let theta = angle([1.0, 0.0], u);
    let mut delta = angle(u, v);
    if !sweep && delta > 0.0 {
        delta -= TAU;
    } else if sweep && delta < 0.0 {
        delta += TAU;
    }
    let steps = (delta.abs() * rx.max(ry) / FLATTEN_STEP)
        .ceil()
        .clamp(1.0, 96.0) as usize;
    for i in 1..=steps {
        let (s, c) = (theta + delta * i as f32 / steps as f32).sin_cos();
        points.push([
            centre[0] + rx * cos * c - ry * sin * s,
            centre[1] + rx * sin * c + ry * cos * s,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::font::FontRenderer;
    use crate::screens::accounts::corner::corner_rect;
    use crate::screens::aero_title::font::TitleFont;
    use crate::screens::aero_title::layout::scale;
    use crate::screens::aero_title::update_notice::{NoticeButton, NoticeView};
    use crate::screens::aero_title::AeroTitleScreen;
    use crate::screens::modmenu::text::MenuText;
    use crate::widget::test_support::{composite_at, uniform_font};
    use crate::widget::DrawList;

    const COPYRIGHT: &str = "Copyright Mojang AB. Do not distribute!";
    const VERSION: &str = "Minecraft 1.8.9";
    /// About the greenhouse's brightness behind the bottom right corner.
    const BACKDROP: [f32; 3] = [0.74, 0.86, 0.76];

    struct Harness {
        text: MenuText,
        title: TitleFont,
        mc: FontRenderer,
        labels: [String; AeroButton::COUNT],
        screen: AeroTitleScreen,
        size: (f32, f32),
        t0: Instant,
    }

    impl Harness {
        fn new(size: (f32, f32)) -> Harness {
            let t0 = Instant::now();
            Harness {
                text: MenuText::minecraft_only(),
                title: TitleFont::embedded(),
                mc: uniform_font(),
                labels: AeroButton::ALL.map(|b| b.english().to_owned()),
                screen: AeroTitleScreen::new(t0),
                size,
                t0,
            }
        }

        fn build(&mut self, ms: u64) -> (DrawList, GlassFrame) {
            let mut list = DrawList::new();
            let mut env = AeroTitleEnv {
                text: &mut self.text,
                title_font: &mut self.title,
                mc_font: &mut self.mc,
                labels: &self.labels,
                version: VERSION,
                copyright: COPYRIGHT,
                screen: self.size,
                now: self.t0 + Duration::from_millis(ms),
                time_s: ms as f32 / 1000.0,
            };
            let glass = self.screen.build(&mut env, &mut list);
            (list, glass)
        }

        fn layout(&self) -> Layout {
            Layout::new(self.size, |size| self.title.cap_height(size))
        }

        /// The version and copyright lines' boxes, as `draw_footer` places them.
        fn footer(&self) -> (Rect, Rect) {
            let layout = self.layout();
            let style = TextStyle::new(14.0 * layout.s).weight(Weight::Semibold);
            let (_, line_h) = self.text.metrics(style);
            let top = layout.footer_bottom - line_h;
            let copyright_w = self.text.measure(COPYRIGHT, style, &self.mc);
            let version_w = self.text.measure(VERSION, style, &self.mc);
            let right = layout.screen.0 - layout.left;
            (
                Rect::new(layout.left, top, version_w, line_h),
                Rect::new(right - copyright_w, top, copyright_w, line_h),
            )
        }
    }

    fn overlaps(a: Rect, b: Rect) -> bool {
        let overlap = a.intersect(&b);
        overlap.w > 0.0 && overlap.h > 0.0
    }

    /// The tallest, widest card the updater can show: a title, a detail line, the progress bar
    /// and three buttons.
    fn widest_card() -> NoticeView {
        NoticeView {
            title: "Aero Client 10.20.30 is available to download right now".to_owned(),
            detail: "Get it from the release page, or update in the client.".to_owned(),
            progress: true,
            buttons: vec![
                NoticeButton::Update,
                NoticeButton::ReleaseNotes,
                NoticeButton::Later,
            ],
        }
    }

    /// Even-odd inside test on the logo's outlines, in the path's own coordinates.
    fn in_logo(x: f32, y: f32) -> bool {
        let logo = logo();
        let p = [x - logo.centre[0], y - logo.centre[1]];
        let crossings = logo
            .outlines
            .iter()
            .filter(|outline| {
                let n = outline.len();
                (0..n)
                    .filter(|&k| {
                        let (a, b) = (outline[k], outline[(k + n - 1) % n]);
                        (a[1] > p[1]) != (b[1] > p[1])
                            && p[0] < a[0] + (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1])
                    })
                    .count()
                    % 2
                    == 1
            })
            .count();
        crossings % 2 == 1
    }

    #[test]
    fn the_path_parser_reads_compact_svg_numbers_flags_and_commands() {
        let mut tokens = PathTokens {
            bytes: b" .0741.0741 0 00-4.8851-1.5152,2e1 -3E-1x",
            at: 0,
        };
        assert_eq!(tokens.number(), Some(0.0741));
        assert_eq!(tokens.number(), Some(0.0741));
        assert_eq!(tokens.number(), Some(0.0));
        assert_eq!(tokens.flag(), Some(false));
        assert_eq!(tokens.flag(), Some(false));
        assert_eq!(tokens.number(), Some(-4.8851));
        assert_eq!(tokens.number(), Some(-1.5152));
        assert_eq!(tokens.number(), Some(20.0));
        assert_eq!(tokens.number(), Some(-0.3));
        assert_eq!(tokens.number(), None, "a letter is not a number");
        assert_eq!(tokens.command(), Some(b'x'));
        assert!(tokens.done());

        // Relative lines after a relative move, H/V, and an implicit close at the end.
        let outlines = parse_path("m1 1 4 0v4h-4zM10 10L12 10 12 12");
        assert_eq!(outlines.len(), 2);
        assert_eq!(
            outlines[0],
            [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]]
        );
        assert_eq!(outlines[1], [[10.0, 10.0], [12.0, 10.0], [12.0, 12.0]]);
        // A half circle by arc ends where it should, through the far side.
        let arc = parse_path("M0 0A5 5 0 0 1 10 0Z");
        let last = *arc[0].last().unwrap();
        assert!(
            (last[0] - 10.0).abs() < 1e-4 && last[1].abs() < 1e-4,
            "{last:?}"
        );
        let lowest = arc[0].iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        assert!(
            (lowest + 5.0).abs() < 0.01,
            "sweep 1 goes through y = -5: {lowest}"
        );
    }

    /// The parsed mark is Simple Icons' Discord logo: 24 units wide, about 18.3 tall, the head
    /// with its two horns and two feet, and two eyes cut out of it.
    #[test]
    fn the_logo_is_the_discord_mark_with_its_eyes_cut_out() {
        let logo = logo();
        assert_eq!(logo.outlines.len(), 3, "the outline and two eyes");
        assert!((logo.size[0] - 24.0).abs() < 0.15, "{:?}", logo.size);
        assert!((logo.size[1] - 18.3).abs() < 0.2, "{:?}", logo.size);
        for (x, y, inside, what) in [
            (12.0, 7.0, true, "the forehead"),
            (12.0, 13.0, true, "between the eyes"),
            (2.0, 15.0, true, "the left cheek"),
            (22.0, 15.0, true, "the right cheek"),
            (8.02, 12.91, false, "the left eye"),
            (15.99, 12.91, false, "the right eye"),
            (12.0, 3.4, false, "between the horns"),
            (12.0, 20.0, false, "between the feet"),
            (1.0, 3.5, false, "outside, top left"),
            (23.5, 21.0, false, "outside, bottom right"),
        ] {
            assert_eq!(in_logo(x, y), inside, "{what} ({x}, {y})");
        }
    }

    /// Bottom right: right-aligned with the copyright line, `FOOTER_GAP` above it, and clear of
    /// everything else on the screen, the widest update card included, at every window size
    /// from a small 640x360 to 4K and odd shapes; in windows too small for it, it hides rather
    /// than overlap. (The title screen scales with the window, not the GUI scale.)
    #[test]
    fn it_sits_above_the_copyright_line_and_never_overlaps_anything() {
        for (w, h) in [
            (854.0, 480.0),
            (1280.0, 720.0),
            (1920.0, 1080.0),
            (2560.0, 1440.0),
            (3840.0, 2160.0),
            (1024.0, 768.0),
            (800.0, 600.0),
            (640.0, 480.0),
            (640.0, 360.0),
            (3440.0, 1440.0),
            (1080.0, 1920.0),
            (600.0, 800.0),
            (400.0, 300.0),
            (300.0, 200.0),
            (200.0, 150.0),
            (854.0, 120.0),
            (120.0, 90.0),
        ] {
            let mut harness = Harness::new((w, h));
            harness.screen.notice.set_view(1, Some(widest_card()));
            let _ = harness.build(5000);
            let layout = harness.layout();
            let (version, copyright) = harness.footer();
            let card = harness.screen.notice.card();
            let Some(pill) = harness.screen.discord.rect() else {
                assert!(
                    w < 640.0 || h < 360.0,
                    "{w}x{h}: hidden in a window that has room"
                );
                continue;
            };
            assert!(
                pill.x >= 0.0 && pill.y >= 0.0 && pill.right() <= w && pill.bottom() <= h,
                "{w}x{h}: outside the window: {pill:?}"
            );
            assert!(
                (pill.right() - copyright.right()).abs() <= 1.0,
                "{w}x{h}: right-aligned"
            );
            let gap = copyright.y - pill.bottom();
            assert!(
                (gap - FOOTER_GAP * layout.s).abs() <= 1.0,
                "{w}x{h}: {gap} above the copyright line"
            );
            let mut others = vec![version, copyright, corner_rect((w, h), 1.0e6)];
            others.extend(layout.buttons);
            others.extend(card);
            // The title: its em box, five and a half em wide, is above the first button.
            others.push(Rect::new(
                layout.left,
                layout.title_baseline - layout.title_size,
                layout.title_size * 5.5,
                layout.title_size * 1.2,
            ));
            for other in others {
                assert!(
                    !overlaps(pill, other),
                    "{w}x{h}: {pill:?} overlaps {other:?}"
                );
            }
        }
    }

    #[test]
    fn it_scales_with_the_window() {
        let rect_at = |size: (f32, f32)| {
            let mut harness = Harness::new(size);
            let _ = harness.build(5000);
            harness.screen.discord.rect().expect("shown")
        };
        let (hd, uhd) = (rect_at((1920.0, 1080.0)), rect_at((3840.0, 2160.0)));
        assert_eq!(hd.h, HEIGHT);
        assert_eq!(uhd.h, 2.0 * HEIGHT);
        assert!((uhd.w - 2.0 * hd.w).abs() <= 2.0, "{hd:?} {uhd:?}");
        let small = rect_at((854.0, 480.0));
        assert_eq!(small.h, (HEIGHT * scale((854.0, 480.0))).round());
    }

    #[test]
    fn a_click_is_a_press_and_release_on_it_once_it_has_appeared() {
        let mut h = Harness::new((1920.0, 1080.0));
        let _ = h.build(5000);
        let rect = h.screen.discord.rect().expect("shown");
        let (x, y) = rect.center();
        // A fresh screen: the footer, and the button with it, are the last to appear.
        h.screen.restart(h.t0 + Duration::from_millis(6000));
        let _ = h.build(6010);
        assert!(h.screen.discord.rect().is_none());
        assert!(!h.screen.discord.mouse_pressed(x, y, 0), "not there yet");
        let _ = h.build(9000);

        assert!(h.screen.discord.hit(x, y));
        assert!(h.screen.discord.mouse_pressed(x, y, 0));
        assert!(h.screen.discord.has_press());
        assert!(
            h.screen.discord.mouse_released(x + 3.0, y - 2.0, 0),
            "a click"
        );
        assert!(!h.screen.discord.has_press());
        // Released off it: no click.
        assert!(h.screen.discord.mouse_pressed(x, y, 0));
        assert!(!h.screen.discord.mouse_released(5.0, 5.0, 0));
        // The right button is taken but never clicks.
        assert!(h.screen.discord.mouse_pressed(x, y, 1));
        assert!(!h.screen.discord.has_press());
        assert!(!h.screen.discord.mouse_released(x, y, 1));
        // Elsewhere is not its own, and it is no title button.
        assert!(!h.screen.discord.mouse_pressed(5.0, 5.0, 0));
        let options = h.screen.button_rect(AeroButton::Options).center();
        assert!(!h.screen.discord.hit(options.0, options.1));
        assert_eq!(h.screen.hit(x, y), None);
    }

    #[test]
    fn hovering_turns_its_glass_blurple_and_pressing_sinks_it() {
        let mut h = Harness::new((1920.0, 1080.0));
        let (idle, _) = h.build(5000);
        let rect = h.screen.discord.rect().expect("shown");
        let blurple = |list: &DrawList| {
            list.vertices().iter().any(|v| {
                rect.inset(-2.0).contains(v.pos[0], v.pos[1])
                    && v.color[3] > 0.5
                    && v.color[2] > 0.75
                    && v.color[0] < 0.45
            })
        };
        assert!(!blurple(&idle), "dark glass at rest");
        let (x, y) = rect.center();
        h.screen.pointer_moved(x, y);
        let _ = h.build(5016);
        let (hovered, glass) = h.build(5400);
        assert!(blurple(&hovered), "Blurple when hovered");
        let pane = glass.rects.last().expect("its pane").rect;
        assert_eq!(pane, [rect.x, rect.y, rect.w, rect.h]);
        assert!(h.screen.discord.mouse_pressed(x, y, 0));
        let _ = h.build(5416);
        let (_, glass) = h.build(5800);
        let pane = glass.rects.last().expect("its pane").rect;
        assert!(
            pane[2] < rect.w && pane[0] > rect.x,
            "pressed, it sinks in: {pane:?}"
        );
    }

    /// The logo is on screen where it belongs: white on the face, the dark glass through the
    /// eyes, at 4K, 1080p and 854x480 (where the whole mark is about ten pixels wide).
    #[test]
    fn the_logo_draws_white_with_the_glass_showing_through_its_eyes() {
        let luma = |rgb: [f32; 3]| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        for size in [(3840.0, 2160.0), (1920.0, 1080.0), (854.0, 480.0)] {
            let mut h = Harness::new(size);
            let (list, _) = h.build(5000);
            let rect = h.screen.discord.rect().expect("shown");
            let s = scale(size);
            let logo = logo();
            let k = LOGO_W * s / logo.size[0];
            let centre = (
                rect.x + (PAD_LEFT + LOGO_W * 0.5) * s,
                rect.y + rect.h * 0.5,
            );
            let at = |x: f32, y: f32| {
                let p = (
                    centre.0 + (x - logo.centre[0]) * k + 0.013,
                    centre.1 + (y - logo.centre[1]) * k + 0.007,
                );
                luma(composite_at(&list, p, BACKDROP))
            };
            let face = at(12.0, 7.5);
            let cheek = at(3.0, 14.5);
            let eye = at(8.02, 12.91);
            let glass = luma(composite_at(
                &list,
                (
                    rect.right() - PAD_RIGHT * 0.5 * s,
                    rect.y + rect.h * 0.5 + 0.3,
                ),
                BACKDROP,
            ));
            let outside = at(12.0, 21.0);
            assert!(
                face > 0.97 && cheek > 0.97,
                "{size:?}: face {face}, cheek {cheek}"
            );
            assert!(
                eye < 0.75 && (eye - glass).abs() < 0.08,
                "{size:?}: eye {eye}, glass {glass}"
            );
            assert!(outside < 0.75, "{size:?}: under the jaw {outside}");
            assert!(
                luma(BACKDROP) - glass > 0.15,
                "{size:?}: the glass is darker"
            );
        }
    }

    /// The logo is tessellated once per size, not per frame.
    #[test]
    fn the_logo_mesh_is_kept_between_frames() {
        let mut h = Harness::new((1920.0, 1080.0));
        let _ = h.build(5000);
        let first = h
            .screen
            .discord
            .logo
            .as_ref()
            .map(|(k, m)| (*k, m.triangle_count()));
        let _ = h.build(5016);
        let second = h
            .screen
            .discord
            .logo
            .as_ref()
            .map(|(k, m)| (*k, m.triangle_count()));
        assert_eq!(first, second);
        let (_, triangles) = first.expect("built");
        assert!(triangles > 100 && triangles < 2000, "{triangles}");
        h.size = (3840.0, 2160.0);
        let _ = h.build(5032);
        let (k, _) = h
            .screen
            .discord
            .logo
            .as_ref()
            .map(|(k, m)| (*k, m))
            .expect("rebuilt");
        assert!((k - 2.0 * first.unwrap().0).abs() < 1e-4);
    }

    /// Pictures of the button, rasterised on the CPU (solid shapes only, so no label), idle and
    /// hovered at 1080p and 4K, for a person to look at: set `MC_DISCORD_PNG_DIR`.
    #[test]
    fn discord_button_pictures() {
        let Some(out) = std::env::var_os("MC_DISCORD_PNG_DIR").map(std::path::PathBuf::from) else {
            eprintln!("skip: set MC_DISCORD_PNG_DIR to write the pictures");
            return;
        };
        std::fs::create_dir_all(&out).expect("picture folder");
        for size in [(1920.0, 1080.0), (3840.0, 2160.0), (854.0, 480.0)] {
            for hover in [false, true] {
                let mut h = Harness::new(size);
                let _ = h.build(5000);
                let rect = h.screen.discord.rect().expect("shown");
                if hover {
                    h.screen.pointer_moved(rect.center().0, rect.center().1);
                    let _ = h.build(5016);
                }
                let (list, _) = h.build(5600);
                let area = rect.inset(-24.0 * scale(size));
                let (pw, ph) = (area.w.ceil() as u32, area.h.ceil() as u32);
                let mut image = image::RgbImage::new(pw, ph);
                for (px, py, pixel) in image.enumerate_pixels_mut() {
                    let at = (area.x + px as f32 + 0.513, area.y + py as f32 + 0.507);
                    let rgb = composite_at(&list, at, BACKDROP);
                    *pixel = image::Rgb(rgb.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8));
                }
                let name = format!(
                    "discord-{}x{}{}.png",
                    size.0,
                    size.1,
                    if hover { "-hover" } else { "" }
                );
                image.save(out.join(&name)).expect("write the picture");
                eprintln!("wrote {name}");
            }
        }
    }
}
