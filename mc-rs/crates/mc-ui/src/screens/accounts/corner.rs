//! The account corner on the Aero title screen: the account in use, its head and name, in an
//! Aero glass pill at the top right; clicking it opens the Accounts screen.
//!
//! It sits on the title's line: right-aligned at the title's own margin
//! ([`layout::MARGIN_X`]) so the two balance, vertically centred where the title's capitals are
//! (an eighth of the way down), as tall as a title-screen button. Its width follows the name.
//! It pops in with the first button and highlights and presses like the buttons do.
//!
//! Like the title screen it is drawn in physical pixels into the title phase's lists, and its
//! glass pane joins the frame's [`GlassFrame`]. The client draws it after
//! [`AeroTitleScreen::build_layers`](crate::screens::aero_title::AeroTitleScreen::build_layers)
//! and routes clicks to it first; it changes nothing in the title screen itself.

use std::time::Instant;

use crate::screens::aero_title::layout::{scale, BUTTON_H, BUTTON_RADIUS, MARGIN_X};
use crate::screens::aero_title::{AeroTitleEnv, CRISP_ALPHA};
use crate::screens::modmenu::anim;
use crate::screens::modmenu::glass::{GlassFrame, GlassRect};
use crate::screens::modmenu::paint::{argb, black, white, Fill, Painter, Rect, Rgba};
use crate::screens::modmenu::text::{TextStyle, Weight};
use crate::widget::draw::{DrawList, UiTexture};

/// What the corner shows: the client fills it when the account changes.
#[derive(Debug, Clone)]
pub struct CornerAccount {
    /// The account's name, or "Not signed in".
    pub name: String,
    /// The line under it: "Microsoft account", "From the launcher", "Add an account", ...
    pub caption: String,
    /// The skin sheet for the head and its height in texels (64, or 32 for a legacy sheet);
    /// `None` draws a plain placeholder.
    pub head: Option<(UiTexture, f32)>,
    /// The caption asks for something (not signed in, sign in again): drawn in amber.
    pub attention: bool,
}

/// Design sizes, in 1080p pixels (scaled like the title screen's: [`scale`]).
pub mod size {
    /// The head.
    pub const HEAD: f32 = 32.0;
    /// Inner padding on both sides.
    pub const PAD: f32 = 12.0;
    /// Between the head and the text.
    pub const GAP: f32 = 12.0;
    /// The chevron's column on the right.
    pub const CHEVRON: f32 = 22.0;
    /// The narrowest pill.
    pub const MIN_W: f32 = 200.0;
    /// The widest pill (a 16-character name fits well inside).
    pub const MAX_W: f32 = 380.0;
    /// The name's em size.
    pub const NAME: f32 = 17.0;
    /// The caption's em size.
    pub const CAPTION: f32 = 13.0;
}

/// The corner's place in the title screen's pop-in: with the first button.
const POP_INDEX: usize = 1;
/// The highlight easing rate, as the title screen's buttons.
const HIGHLIGHT_RATE: f32 = 20.0;
/// The caption colour when it asks for something.
const AMBER: u32 = 0xFFFF_D98A;

/// The pill's rectangle for a `screen`-sized window when its text is `text_w` pixels wide.
#[must_use]
pub fn corner_rect(screen: (f32, f32), text_w: f32) -> Rect {
    let s = scale(screen);
    let chrome = (size::PAD + size::HEAD + size::GAP + size::CHEVRON + size::PAD) * s;
    let w = (text_w + chrome)
        .clamp(size::MIN_W * s, size::MAX_W * s)
        .round();
    let h = (BUTTON_H * s).round();
    let right = screen.0 - (MARGIN_X * s).round();
    let centre = screen.1 / 8.0;
    Rect::new(right - w, (centre - h * 0.5).round(), w, h)
}

/// The corner's state; the client keeps one with the title screen.
#[derive(Debug, Clone, Default)]
pub struct AccountCorner {
    rect: Rect,
    shown: bool,
    pressed: bool,
    hover: f32,
    press: f32,
    last_frame: Option<Instant>,
}

impl AccountCorner {
    /// A corner that has not been drawn yet.
    #[must_use]
    pub fn new() -> AccountCorner {
        AccountCorner::default()
    }

    /// The pill's rectangle in the last frame (physical pixels).
    #[must_use]
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// Whether `(x, y)` is on the pill (once it has started to appear).
    #[must_use]
    pub fn hit(&self, x: f32, y: f32) -> bool {
        self.shown && self.rect.contains(x, y)
    }

    /// A mouse button went down at `(x, y)`: whether the corner took it.
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8) -> bool {
        self.pressed = button == 0 && self.hit(x, y);
        self.pressed
    }

    /// A mouse button went up at `(x, y)`: whether that completes a click on the corner (the
    /// press landed on it too).
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8) -> bool {
        if button != 0 {
            return false;
        }
        let pressed = std::mem::take(&mut self.pressed);
        pressed && self.hit(x, y)
    }

    /// Forget a press (the title screen went away under it).
    pub fn reset(&mut self) {
        self.pressed = false;
        self.hover = 0.0;
        self.press = 0.0;
        self.last_frame = None;
    }

    /// Draw one frame of the corner into `list` (and its crisp marks into `crisp`, when the
    /// title screen has a finish over it), and add its pane to `glass`. `opened_at` is the title
    /// screen's pop-in start; `cursor` the pointer in physical pixels.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &mut self,
        account: &CornerAccount,
        env: &mut AeroTitleEnv<'_>,
        opened_at: Instant,
        cursor: (f32, f32),
        list: &mut DrawList,
        crisp: Option<&mut DrawList>,
        glass: &mut GlassFrame,
    ) {
        let now = env.now;
        let dt = self.last_frame.map_or(0.0, |last| {
            now.saturating_duration_since(last).as_secs_f32()
        });
        self.last_frame = Some(now);
        let s = scale(env.screen);
        let name_style = TextStyle::new(size::NAME * s).weight(Weight::Semibold);
        let caption_style = TextStyle::new(size::CAPTION * s).weight(Weight::Semibold);
        let name_w = env.text.measure(&account.name, name_style, env.mc_font);
        let caption_w = env
            .text
            .measure(&account.caption, caption_style, env.mc_font);
        let rect = corner_rect(env.screen, name_w.max(caption_w));
        self.rect = rect;

        let elapsed = now.saturating_duration_since(opened_at).as_secs_f32() * 1000.0;
        let progress = anim::window_open_progress(elapsed, POP_INDEX);
        self.shown = progress > 0.0;
        let over = self.shown && rect.contains(cursor.0, cursor.1);
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

        let slide = (1.0 - progress) * anim::SLIDE_DIP * s;
        let body = rect.inset(self.press * 1.5 * s);
        let radius = BUTTON_RADIUS * s;
        glass.rects.push(GlassRect {
            rect: [body.x, body.y + slide, body.w, body.h],
            radius,
            alpha: progress,
        });
        let look = Look {
            body,
            radius,
            hover: self.hover,
            press: self.press,
            s,
            name_style,
            caption_style,
        };
        let mut painter = Painter::new(list);
        painter.alpha = progress;
        painter.offset = (0.0, slide);
        draw_body(&mut painter, &look);
        draw_marks(&mut painter, env, account, &look, true);
        if let Some(crisp) = crisp {
            let mut painter = Painter::new(crisp);
            painter.alpha = progress * CRISP_ALPHA;
            painter.offset = (0.0, slide);
            draw_marks(&mut painter, env, account, &look, false);
        }
    }
}

/// One frame's look.
struct Look {
    body: Rect,
    radius: f32,
    hover: f32,
    press: f32,
    s: f32,
    name_style: TextStyle,
    caption_style: TextStyle,
}

fn mix(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// The pill's body, the title-screen buttons' chrome: drop shadow, a tint that keeps white text
/// readable over the bright scene, and the 135-degree sheen.
fn draw_body(painter: &mut Painter<'_>, look: &Look) {
    let Look {
        body,
        radius,
        hover,
        press,
        s,
        ..
    } = *look;
    painter.round_rect_shadow(
        body,
        radius,
        [14.0 * s, 5.0 * s, 14.0 * s, 24.0 * s],
        black(0.20),
    );
    let top = mix(mix(black(0.22), black(0.10), hover), black(0.32), press);
    let bottom = mix(mix(black(0.34), black(0.24), hover), black(0.42), press);
    painter.round_rect(body, radius, Fill::css_angle(body, 180.0, top, bottom));
    let corner = Rect::new(body.x, body.y, body.w * 0.6, body.h * 1.4);
    painter.round_rect(
        body,
        radius,
        Fill::css_angle(corner, 135.0, white(0.10 + 0.06 * hover), white(0.0)),
    );
}

/// The pill's marks: highlight and border, the head, the name and caption, and the chevron.
/// `full` adds the soft dark halos (the copy under the title screen's finish); the crisp copy
/// over the finish has only the light marks.
fn draw_marks(
    painter: &mut Painter<'_>,
    env: &mut AeroTitleEnv<'_>,
    account: &CornerAccount,
    look: &Look,
    full: bool,
) {
    let Look {
        body,
        radius,
        hover,
        s,
        name_style,
        caption_style,
        ..
    } = *look;
    painter.line(
        body.x + radius * 0.8,
        body.y + 1.5,
        body.right() - radius * 0.8,
        body.y + 1.5,
        1.0,
        white(0.26 + 0.14 * hover),
    );
    painter.round_rect_stroke(body, radius, 1.0, 0.5, white(0.40 + 0.30 * hover));

    // The head: the skin's face and hat layers, with a thin light frame.
    let head = (size::HEAD * s).round();
    let head_rect = Rect::new(
        (body.x + size::PAD * s).round(),
        (body.y + (body.h - head) * 0.5).round(),
        head,
        head,
    );
    if full {
        painter.round_rect(head_rect.inset(-2.0 * s), 3.0 * s, Fill::Solid(black(0.25)));
    }
    match &account.head {
        Some((sheet, height)) => {
            let v = 1.0 / height.max(1.0);
            let face = [8.0 / 64.0, 8.0 * v, 16.0 / 64.0, 16.0 * v];
            let hat = [40.0 / 64.0, 8.0 * v, 48.0 / 64.0, 16.0 * v];
            painter.textured(sheet, head_rect, face, white(1.0));
            painter.textured(sheet, head_rect, hat, white(1.0));
        }
        None => draw_person(painter, head_rect, 3.0 * s),
    }
    painter.round_rect_stroke(head_rect.inset(-s), 3.0 * s, 1.0, 0.5, white(0.45));

    // The name over the caption, centred together on the pill's middle.
    let (_, name_h) = env.text.metrics(name_style);
    let (_, caption_h) = env.text.metrics(caption_style);
    let x = head_rect.right() + size::GAP * s;
    let top = body.y + (body.h - name_h - caption_h) * 0.5;
    let name_color = white(0.95 + 0.05 * hover);
    let caption_color = if account.attention {
        argb(AMBER)
    } else {
        white(0.78)
    };
    for (text, style, y, color) in [
        (account.name.as_str(), name_style, top, name_color),
        (
            account.caption.as_str(),
            caption_style,
            top + name_h,
            caption_color,
        ),
    ] {
        if full {
            haloed(painter, env, text, (x, y), style, color, s);
        } else {
            env.text
                .draw(painter, env.mc_font, text, x, y, style, color);
        }
    }

    // A chevron: the pill opens a screen.
    let cx = body.right() - (size::PAD + size::CHEVRON * 0.5) * s + hover * 2.0 * s;
    let cy = body.y + body.h * 0.5;
    let c = white(0.55 + 0.35 * hover);
    let lw = 1.7 * s;
    painter.line(cx - 2.5 * s, cy - 5.0 * s, cx + 2.5 * s, cy, lw, c);
    painter.line(cx + 2.5 * s, cy, cx - 2.5 * s, cy + 5.0 * s, lw, c);
}

/// The head's place when nobody is signed in: a plain person (a round head over rounded
/// shoulders) on a faint tile.
fn draw_person(painter: &mut Painter<'_>, tile: Rect, radius: f32) {
    painter.round_rect(tile, radius, Fill::Solid(white(0.18)));
    let (cx, h) = (tile.x + tile.w * 0.5, tile.h);
    painter.circle(cx, tile.y + h * 0.38, h * 0.18, white(0.85));
    let previous = painter.clip();
    painter.set_clip(Some(tile));
    painter.round_rect(
        Rect::new(cx - h * 0.32, tile.y + h * 0.64, h * 0.64, h * 0.5),
        h * 0.25,
        Fill::Solid(white(0.85)),
    );
    painter.set_clip(previous);
}

/// Text over a backdrop that can be as bright as the text: a soft dark halo (the glyphs faintly
/// at eight offsets), a drop shadow, then the text (as the title screen's labels).
fn haloed(
    painter: &mut Painter<'_>,
    env: &mut AeroTitleEnv<'_>,
    text: &str,
    (x, y): (f32, f32),
    style: crate::screens::modmenu::text::TextStyle,
    color: Rgba,
    s: f32,
) {
    let r = s.max(1.0);
    for (dx, dy) in [
        (-0.7, -0.7),
        (0.0, -1.0),
        (0.7, -0.7),
        (-1.0, 0.0),
        (1.0, 0.0),
        (-0.7, 0.7),
        (0.0, 1.0),
        (0.7, 0.7),
    ] {
        env.text.draw(
            painter,
            env.mc_font,
            text,
            x + dx * r,
            y + dy * r,
            style,
            black(0.16 * color[3]),
        );
    }
    env.text.draw(
        painter,
        env.mc_font,
        text,
        x,
        y + r,
        style,
        black(0.45 * color[3]),
    );
    env.text
        .draw(painter, env.mc_font, text, x, y, style, color);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::font::FontRenderer;
    use crate::screens::aero_title::font::TitleFont;
    use crate::screens::aero_title::layout::Layout;
    use crate::screens::aero_title::AeroButton;
    use crate::screens::modmenu::text::MenuText;
    use crate::widget::test_support::uniform_font;

    fn account(name: &str) -> CornerAccount {
        CornerAccount {
            name: name.to_owned(),
            caption: "Microsoft account".to_owned(),
            head: Some((UiTexture::gui("textures/entity/steve.png"), 64.0)),
            attention: false,
        }
    }

    struct Harness {
        text: MenuText,
        title: TitleFont,
        mc: FontRenderer,
        labels: [String; AeroButton::COUNT],
        t0: Instant,
    }

    impl Harness {
        fn new() -> Harness {
            Harness {
                text: MenuText::minecraft_only(),
                title: TitleFont::embedded(),
                mc: uniform_font(),
                labels: AeroButton::ALL.map(|b| b.english().to_owned()),
                t0: Instant::now(),
            }
        }

        fn build(
            &mut self,
            corner: &mut AccountCorner,
            account: &CornerAccount,
            screen: (f32, f32),
            ms: u64,
            cursor: (f32, f32),
        ) -> (DrawList, GlassFrame) {
            let mut list = DrawList::new();
            let mut glass = GlassFrame::default();
            let mut env = AeroTitleEnv {
                text: &mut self.text,
                title_font: &mut self.title,
                mc_font: &mut self.mc,
                labels: &self.labels,
                version: "Minecraft 1.8.9",
                game_version: mc_version::GameVersion::V1_8_9,
                copyright: "",
                screen,
                now: self.t0 + Duration::from_millis(ms),
                time_s: 0.0,
            };
            corner.build(
                account, &mut env, self.t0, cursor, &mut list, None, &mut glass,
            );
            (list, glass)
        }
    }

    #[test]
    fn the_pill_sits_on_the_titles_line_at_the_titles_margin() {
        for screen in [
            (1920.0, 1080.0),
            (2560.0, 1440.0),
            (1280.0, 720.0),
            (854.0, 480.0),
        ] {
            let s = scale(screen);
            let rect = corner_rect(screen, 120.0 * s);
            assert!((rect.right() - (screen.0 - (MARGIN_X * s).round())).abs() < 0.5);
            let centre = rect.y + rect.h * 0.5;
            assert!(
                (centre - screen.1 / 8.0).abs() <= 1.0,
                "{screen:?}: {rect:?}"
            );
            assert_eq!(rect.h, (BUTTON_H * s).round());
            assert!(rect.w >= (size::MIN_W * s).round() - 0.5);
            assert!(rect.w <= (size::MAX_W * s).round() + 0.5);
            assert!(rect.x > 0.0 && rect.y > 0.0, "{screen:?}: {rect:?}");
        }
        // A long name grows the pill up to its maximum, never past it.
        let wide = corner_rect((1920.0, 1080.0), 10_000.0);
        assert_eq!(wide.w, size::MAX_W);
    }

    #[test]
    fn the_pill_never_covers_the_title_or_the_buttons() {
        for screen in [
            (1920.0, 1080.0),
            (2560.0, 1440.0),
            (1280.0, 720.0),
            (1024.0, 768.0),
        ] {
            let title = TitleFont::embedded();
            let layout = Layout::new(screen, |size| title.cap_height(size));
            let pill = corner_rect(screen, 10_000.0);
            for button in layout.buttons {
                assert!(pill.intersect(&button).w <= 0.0, "{screen:?}");
            }
            // Right of the title's ink: the title is about 5 em wide at most.
            assert!(
                pill.x > layout.left + layout.title_size * 5.0,
                "{screen:?}: {pill:?}"
            );
        }
    }

    #[test]
    fn it_pops_in_with_the_first_button_and_adds_one_glass_pane() {
        let mut h = Harness::new();
        let mut corner = AccountCorner::new();
        let a = account("Alpha");
        let (list, glass) = h.build(&mut corner, &a, (1920.0, 1080.0), 0, (-1.0, -1.0));
        assert!(list.is_empty() && glass.rects.is_empty(), "not yet");
        assert!(!corner.hit(corner.rect().center().0, corner.rect().center().1));
        let (list, glass) = h.build(&mut corner, &a, (1920.0, 1080.0), 1000, (-1.0, -1.0));
        assert!(!list.is_empty());
        assert_eq!(glass.rects.len(), 1);
        let r = corner.rect();
        assert_eq!(glass.rects[0].rect, [r.x, r.y, r.w, r.h]);
        assert_eq!(glass.rects[0].alpha, 1.0);
        // The head is drawn from the skin sheet: two textured quads (face and hat).
        let skin = UiTexture::gui("textures/entity/steve.png");
        assert!(list.batches().iter().any(|batch| batch.texture == skin));
    }

    #[test]
    fn a_click_needs_the_press_and_the_release_on_the_pill() {
        let mut h = Harness::new();
        let mut corner = AccountCorner::new();
        let a = account("Alpha");
        let _ = h.build(&mut corner, &a, (1920.0, 1080.0), 1000, (-1.0, -1.0));
        let (x, y) = corner.rect().center();
        assert!(corner.mouse_pressed(x, y, 0));
        assert!(corner.mouse_released(x, y, 0));
        // Pressed on it, released elsewhere: no click.
        assert!(corner.mouse_pressed(x, y, 0));
        assert!(!corner.mouse_released(5.0, 5.0, 0));
        // Elsewhere, or another button: not taken.
        assert!(!corner.mouse_pressed(5.0, 5.0, 0));
        assert!(!corner.mouse_pressed(x, y, 1));
        assert!(!corner.mouse_released(x, y, 1));
    }

    #[test]
    fn hovering_changes_the_drawing_and_a_missing_head_still_draws() {
        let mut h = Harness::new();
        let mut corner = AccountCorner::new();
        let a = account("Alpha");
        let (idle, _) = h.build(&mut corner, &a, (1920.0, 1080.0), 1000, (-1.0, -1.0));
        let (x, y) = corner.rect().center();
        let _ = h.build(&mut corner, &a, (1920.0, 1080.0), 1016, (x, y));
        let (hovered, _) = h.build(&mut corner, &a, (1920.0, 1080.0), 1300, (x, y));
        assert_ne!(idle.vertices(), hovered.vertices());
        let none = CornerAccount {
            head: None,
            attention: true,
            ..account("Not signed in")
        };
        let mut fresh = AccountCorner::new();
        let (list, glass) = h.build(&mut fresh, &none, (1920.0, 1080.0), 1000, (0.0, 0.0));
        assert!(!list.is_empty());
        assert_eq!(glass.rects.len(), 1);
    }
}
