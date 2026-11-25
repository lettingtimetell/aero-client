//! The Aero title screen (`aero-title-2026-11-07`): "Aero Client" in Inter at the top
//! left, and Multiplayer, Options, Language and Quit Game as Aero glass buttons in a column under
//! it, over the block greenhouse (`mc_render::menu_scene`). It is vanilla's title screen
//! minus Singleplayer and Realms, which this multiplayer-only client cannot use anyway.
//!
//! Like the Shift+Tab menu ([`super::modmenu`]) this is plain state plus arithmetic:
//! [`AeroTitleScreen::build`] draws one frame into a [`DrawList`] in **physical pixels** and
//! returns the [`GlassFrame`] for [`phase::AeroTitlePhase`], a second instance of the menu's own
//! glass phase. It reuses the menu's pieces rather than copying them: the glass shader, the
//! [`Painter`], the Segoe UI [`MenuText`] for the labels, and [`anim`]'s easing and stagger, so
//! the title and then each button pop in exactly as the menu's windows do (fade in and rise
//! [`anim::SLIDE_DIP`] with an ease-out cubic over [`anim::OPEN_MS`], each [`anim::STAGGER_MS`]
//! after the one before).
//!
//! Clicks follow the menu too: a press marks a button (it darkens), and releasing over the same
//! button activates it ([`AeroTitleScreen::mouse_released`]). The client maps the button to
//! vanilla's own action ([`AeroButton::action`]) and plays vanilla's button sound.
//!
//! The column starts with the version slider ([`version_slider`]): which Minecraft version the
//! next connection plays, 1.7.10 or 1.8.9.
//!
//! Above the copyright line at the bottom right, a small glass button with the Discord logo
//! links to the Aero Client Discord server ([`discord`]).

// The Discord button (bottom right).
pub mod discord;
pub mod font;
pub mod icons;
pub mod layout;
pub mod phase;
// The update card (bottom left).
pub mod update_notice;
pub mod version_slider;

#[cfg(test)]
mod tests;

use std::time::Instant;

use mc_version::GameVersion;

use self::font::{TitleFont, TitleLook};
use self::icons::Icon;
use self::layout::Layout;
use super::modmenu::anim;
use super::modmenu::glass::{GlassFrame, GlassRect};
use super::modmenu::paint::{black, white, Fill, Painter, Rect, Rgba};
use super::modmenu::text::{MenuText, TextStyle, Weight};
use crate::font::FontRenderer;
use crate::screens::options::OptionsPage;
use crate::screens::Action;
use crate::widget::DrawList;

/// The title.
pub const TITLE: &str = "Aero Client";

/// The glass frost (mip bias) inside the buttons: frostier than the menu's 1.5, so the bright,
/// busy leaves behind a label blur into a calm backing.
const FROST: f32 = 3.4;
/// The rate the hover and press highlights ease at (per second; see [`anim::approach`]).
const HIGHLIGHT_RATE: f32 = 20.0;
/// How strongly the crisp copy of the buttons' marks covers their VHS-finished copy
/// ([`AeroTitleScreen::build_layers`]): about half the finish's blur stays on the small text.
pub const CRISP_ALPHA: f32 = 0.55;

/// The version slider's place in the pop-in ([`anim::window_open_progress`]): with Multiplayer,
/// right under it, so every other element keeps its time.
const VERSION_POP: usize = 1;

/// One of the buttons.
///
/// **Replays** (the replay browser) has no vanilla
/// equivalent. Buttons are sized by [`AeroButton::COUNT`]; a new button is a variant here plus
/// its entry in `ALL`, `index`, `lang_key`, `english` and `icon` (the Mods button).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AeroButton {
    /// Open the server list.
    Multiplayer,
    /// Open the replay browser.
    Replays,
    /// Open the mod menu over the title screen. Edit HUD is greyed out there: no world.
    Mods,
    /// Open the options screen.
    Options,
    /// Open the language page.
    Language,
    /// Quit the game.
    Quit,
}

impl AeroButton {
    /// How many buttons there are.
    pub const COUNT: usize = 6;

    /// Top to bottom.
    pub const ALL: [AeroButton; AeroButton::COUNT] = [
        AeroButton::Multiplayer,
        AeroButton::Replays,
        AeroButton::Mods,
        AeroButton::Options,
        AeroButton::Language,
        AeroButton::Quit,
    ];

    /// What vanilla's title screen does for the same button (`GuiMainMenu.actionPerformed`).
    /// Buttons vanilla does not have: [`AeroButton::Replays`] returns [`Action::None`] (the
    /// client handles it by variant), [`AeroButton::Mods`] the mod menu's
    /// [`Action::OpenModMenu`].
    #[must_use]
    pub fn action(self) -> Action {
        match self {
            AeroButton::Multiplayer => Action::OpenMultiplayer,
            AeroButton::Replays => Action::None,
            AeroButton::Mods => Action::OpenModMenu,
            AeroButton::Options => Action::OpenOptions,
            AeroButton::Language => Action::OpenOptionsPage(OptionsPage::Language),
            AeroButton::Quit => Action::Quit,
        }
    }

    /// The vanilla language key of the label (an Aero key, missing from every vanilla
    /// language, for buttons vanilla does not have: they use [`AeroButton::english`]).
    #[must_use]
    pub fn lang_key(self) -> &'static str {
        match self {
            AeroButton::Multiplayer => "menu.multiplayer",
            AeroButton::Replays => "aero.menu.replays",
            AeroButton::Mods => "aero.menu.mods",
            AeroButton::Options => "menu.options",
            AeroButton::Language => "options.language",
            AeroButton::Quit => "menu.quit",
        }
    }

    /// The label when the language has none (or Segoe UI cannot draw it).
    #[must_use]
    pub fn english(self) -> &'static str {
        match self {
            AeroButton::Multiplayer => "Multiplayer",
            AeroButton::Replays => "Replays",
            AeroButton::Mods => "Mods",
            AeroButton::Options => "Options",
            AeroButton::Language => "Language",
            AeroButton::Quit => "Quit Game",
        }
    }

    fn icon(self) -> Icon {
        match self {
            AeroButton::Multiplayer => Icon::Multiplayer,
            AeroButton::Replays => Icon::Replays,
            AeroButton::Mods => Icon::Mods,
            AeroButton::Options => Icon::Options,
            AeroButton::Language => Icon::Language,
            AeroButton::Quit => Icon::Quit,
        }
    }

    /// The button's place in [`AeroButton::ALL`].
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            AeroButton::Multiplayer => 0,
            AeroButton::Replays => 1,
            AeroButton::Mods => 2,
            AeroButton::Options => 3,
            AeroButton::Language => 4,
            AeroButton::Quit => 5,
        }
    }
}

/// The button labels in the game's language, from vanilla's own keys, without the trailing
/// "..." vanilla puts on buttons that open a screen ("Options...", "Language...").
#[must_use]
pub fn labels(lang: &mc_assets::Language) -> [String; AeroButton::COUNT] {
    AeroButton::ALL.map(|button| {
        let key = button.lang_key();
        let text = lang.tr(key);
        let text = text
            .trim()
            .trim_end_matches("...")
            .trim_end_matches('\u{2026}');
        if text.is_empty() || text == key {
            button.english().to_owned()
        } else {
            text.trim_end().to_owned()
        }
    })
}

/// What the client hands [`AeroTitleScreen::build`] each frame.
pub struct AeroTitleEnv<'a> {
    /// Segoe UI (or Minecraft-font) text for the labels and the footer.
    pub text: &'a mut MenuText,
    /// The title's font.
    pub title_font: &'a mut TitleFont,
    /// The Minecraft font (fallback text and measuring).
    pub mc_font: &'a mut FontRenderer,
    /// The button labels ([`labels`]).
    pub labels: &'a [String; AeroButton::COUNT],
    /// The bottom-left line (vanilla's version line).
    pub version: &'a str,
    /// The version the next connection plays: where the version slider's knob rests.
    pub game_version: GameVersion,
    /// The bottom-right line (vanilla's copyright line).
    pub copyright: &'a str,
    /// Window size in physical pixels.
    pub screen: (f32, f32),
    /// Real time now (the animation clock).
    pub now: Instant,
    /// The glass `uTime` in seconds.
    pub time_s: f32,
}

/// The title screen's state. The client keeps one while the title screen is up.
#[derive(Debug)]
pub struct AeroTitleScreen {
    opened_at: Instant,
    last_frame: Option<Instant>,
    cursor: (f32, f32),
    pressed: Option<AeroButton>,
    /// Eased hover highlight per button, 0..1.
    hover: [f32; AeroButton::COUNT],
    /// Eased press highlight per button, 0..1.
    press: [f32; AeroButton::COUNT],
    /// The buttons' rectangles in the last frame (hit regions; immediate mode, as the menu).
    rects: [Rect; AeroButton::COUNT],
    /// Whether each button had started to appear in the last frame.
    shown: [bool; AeroButton::COUNT],
    /// The update card (bottom left), empty unless the client has an
    /// update to show ([`update_notice::UpdateNotice::set_view`]).
    pub notice: update_notice::UpdateNotice,
    /// The Discord button (bottom right); the client opens the link.
    pub discord: discord::DiscordButton,
    /// The version slider at the top of the column. The client routes clicks on it before the
    /// buttons and saves what it picks ([`version_slider::VersionSlider::mouse_released`]).
    pub version_slider: version_slider::VersionSlider,
}

impl AeroTitleScreen {
    /// A title screen whose pop-in starts at `now`.
    #[must_use]
    pub fn new(now: Instant) -> AeroTitleScreen {
        AeroTitleScreen {
            opened_at: now,
            last_frame: None,
            cursor: (-1.0, -1.0),
            pressed: None,
            hover: [0.0; AeroButton::COUNT],
            press: [0.0; AeroButton::COUNT],
            rects: [Rect::default(); AeroButton::COUNT],
            shown: [false; AeroButton::COUNT],
            notice: update_notice::UpdateNotice::default(),
            discord: discord::DiscordButton::default(),
            version_slider: version_slider::VersionSlider::default(),
        }
    }

    /// When the pop-in started.
    #[must_use]
    pub fn opened_at(&self) -> Instant {
        self.opened_at
    }

    /// Restart the pop-in at `now` (the scene behind took a while to build: show the whole
    /// animation rather than its end). The hover and press easing keep their own clock.
    pub fn restart(&mut self, now: Instant) {
        self.opened_at = now;
    }

    /// Whether every element has finished popping in at `now`.
    #[must_use]
    pub fn settled(&self, now: Instant) -> bool {
        ms_since(self.opened_at, now) >= anim::OPEN_MS + AeroButton::COUNT as f32 * anim::STAGGER_MS
    }

    /// The button's rectangle in the last frame (physical pixels).
    #[must_use]
    pub fn button_rect(&self, button: AeroButton) -> Rect {
        self.rects[button.index()]
    }

    /// The button under `(x, y)`, among those already on screen.
    #[must_use]
    pub fn hit(&self, x: f32, y: f32) -> Option<AeroButton> {
        AeroButton::ALL.into_iter().find(|button| {
            let i = button.index();
            self.shown[i] && self.rects[i].contains(x, y)
        })
    }

    /// The cursor is at `(x, y)` (physical pixels).
    pub fn pointer_moved(&mut self, x: f32, y: f32) {
        self.cursor = (x, y);
    }

    /// A mouse button went down at `(x, y)`. Returns whether it landed on a button.
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8) -> bool {
        self.cursor = (x, y);
        if button != 0 {
            return false;
        }
        self.pressed = self.hit(x, y);
        self.pressed.is_some()
    }

    /// A mouse button went up at `(x, y)`: the button to activate, when it is the one the press
    /// landed on.
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8) -> Option<AeroButton> {
        self.cursor = (x, y);
        if button != 0 {
            return None;
        }
        let pressed = self.pressed.take()?;
        (self.hit(x, y) == Some(pressed)).then_some(pressed)
    }

    /// Draw one frame into `list` (physical pixels) and return the glass for it.
    pub fn build(&mut self, env: &mut AeroTitleEnv<'_>, list: &mut DrawList) -> GlassFrame {
        self.build_layers(env, list, None)
    }

    /// [`AeroTitleScreen::build`], and the buttons' and footer's crisp marks (labels, icons,
    /// borders) once more into `crisp`, at [`CRISP_ALPHA`], for a phase drawn **after** the
    /// greenhouse's VHS finish. The finish blurs the whole frame, UI included;
    /// laid over it, the crisp copy gives the small text a lighter version of it (measured: the
    /// full finish halved the labels' contrast). The title and the glass stay under the finish.
    pub fn build_layers(
        &mut self,
        env: &mut AeroTitleEnv<'_>,
        list: &mut DrawList,
        crisp: Option<&mut DrawList>,
    ) -> GlassFrame {
        let now = env.now;
        let dt = self.last_frame.map_or(0.0, |last| {
            now.saturating_duration_since(last).as_secs_f32()
        });
        self.last_frame = Some(now);
        let elapsed = ms_since(self.opened_at, now);
        let layout = {
            let font = &*env.title_font;
            Layout::new(env.screen, |size| font.cap_height(size))
        };
        let s = layout.s;
        self.rects = layout.buttons;

        // Highlights ease towards the pointer, the same way at any frame rate.
        let k = anim::approach(dt, HIGHLIGHT_RATE);
        for button in AeroButton::ALL {
            let i = button.index();
            let over = self.shown[i] && self.rects[i].contains(self.cursor.0, self.cursor.1);
            let hover_target = if over { 1.0 } else { 0.0 };
            let press_target = if over && self.pressed == Some(button) {
                1.0
            } else {
                0.0
            };
            for (value, target) in [
                (&mut self.hover[i], hover_target),
                (&mut self.press[i], press_target),
            ] {
                *value += (target - *value) * k;
                if (target - *value).abs() < 0.002 {
                    *value = target;
                }
            }
        }

        let mut glass = GlassFrame {
            px_per_dip: s,
            time: env.time_s,
            dim: 0.0,
            strength: 1.0,
            frost: FROST,
            // The buttons, plus one pane for the update card and one for the account pill
            // and one for the Discord button, so none of
            // them reallocates per frame. One more for the version slider.
            rects: Vec::with_capacity(AeroButton::COUNT + 4),
        };
        let mut painter = Painter::new(list);
        let mut crisp = crisp.map(Painter::new);

        // ---- the title: element 0 of the pop-in ----
        let progress = anim::window_open_progress(elapsed, 0);
        if progress > 0.0 {
            painter.alpha = progress;
            painter.offset = (0.0, (1.0 - progress) * anim::SLIDE_DIP * s);
            draw_title(&mut painter, env, &layout);
        }

        // ---- the buttons: elements 1 to COUNT ----
        for button in AeroButton::ALL {
            let i = button.index();
            let progress = anim::window_open_progress(elapsed, i + 1);
            self.shown[i] = progress > 0.0;
            if progress <= 0.0 {
                continue;
            }
            let slide = (1.0 - progress) * anim::SLIDE_DIP * s;
            // A pressed button sinks in a little.
            let rect = layout.buttons[i].inset(self.press[i] * 1.5 * s);
            glass.rects.push(GlassRect {
                rect: [rect.x, rect.y + slide, rect.w, rect.h],
                radius: layout.radius,
                alpha: progress,
            });
            let label = env.labels[i].as_str();
            let label = if env.text.covers(label) {
                label
            } else {
                button.english()
            };
            let look = ButtonLook {
                rect,
                radius: layout.radius,
                hover: self.hover[i],
                press: self.press[i],
                // The icon pops a little on top of the button's own rise.
                icon_scale: 0.82 + 0.18 * progress,
            };
            painter.alpha = progress;
            painter.offset = (0.0, slide);
            draw_button_glass(&mut painter, look, s);
            draw_button_marks(
                &mut painter,
                env,
                look,
                (button.icon(), label),
                Layer::Full,
                s,
            );
            if let Some(crisp) = crisp.as_mut() {
                crisp.alpha = progress * CRISP_ALPHA;
                crisp.offset = (0.0, slide);
                draw_button_marks(crisp, env, look, (button.icon(), label), Layer::Crisp, s);
            }
        }

        // ---- the version slider: element 1, with Multiplayer; its pane after the buttons' ----
        let cursor = self.cursor;
        self.version_slider.build(
            &mut painter,
            crisp.as_mut(),
            env,
            &layout,
            &mut glass,
            (elapsed, dt, cursor),
        );

        // ---- the footer, with the last button ----
        let progress = anim::window_open_progress(elapsed, AeroButton::COUNT);
        if progress > 0.0 {
            painter.alpha = progress;
            painter.offset = (0.0, 0.0);
            draw_footer(&mut painter, env, &layout, Layer::Full);
            if let Some(crisp) = crisp.as_mut() {
                crisp.alpha = progress * CRISP_ALPHA;
                crisp.offset = (0.0, 0.0);
                draw_footer(crisp, env, &layout, Layer::Crisp);
            }
        }

        // The update card, bottom left, after the buttons' pop-in (an
        // `Option` test when there is none).
        self.notice.build(
            &mut painter,
            crisp.as_mut(),
            env,
            &layout,
            &mut glass,
            (elapsed, dt, cursor),
        );

        // The Discord button, bottom right, clear of the card it may share the
        // bottom with in a narrow window.
        let card = self.notice.card();
        self.discord.build(
            &mut painter,
            crisp.as_mut(),
            env,
            &layout,
            &mut glass,
            (elapsed, dt, cursor),
            card,
        );
        glass
    }
}

/// Which copy of the marks is being drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layer {
    /// Everything: dark halos and shadows under the white marks (under the VHS finish).
    Full,
    /// Only the white marks themselves, for the part-strength copy over the finish. A halo drawn
    /// at part strength under part-strength text would darken the strokes it sits under.
    Crisp,
}

/// One button's look this frame.
#[derive(Debug, Clone, Copy)]
struct ButtonLook {
    rect: Rect,
    radius: f32,
    hover: f32,
    press: f32,
    icon_scale: f32,
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

fn draw_title(painter: &mut Painter<'_>, env: &mut AeroTitleEnv<'_>, layout: &Layout) {
    // The logo in the "Logo style" option's style (the client sets it on the title font).
    if env.title_font.prepare(TITLE, layout.title_size) {
        let x = layout.left - env.title_font.ink_left();
        env.title_font.draw_logo(painter, x, layout.title_baseline);
        return;
    }
    // No title font: Segoe UI (or the Minecraft font) at the same size.
    let style = TextStyle::new(layout.title_size);
    let (ascent, _) = env.text.metrics(style);
    let top = layout.title_baseline - ascent;
    let shadow: TitleLook = env.title_font.look();
    env.text.draw(
        painter,
        env.mc_font,
        TITLE,
        layout.left,
        top + 3.0 * layout.s,
        style,
        shadow.shadow,
    );
    env.text.draw(
        painter,
        env.mc_font,
        TITLE,
        layout.left,
        top,
        style,
        white(1.0),
    );
}

/// Text with the menu's soft dark shadow (`text-shadow: 0 1px 1px rgba(0,0,0,.38)`), a touch
/// stronger because the backdrop is bright.
#[allow(clippy::too_many_arguments)]
fn shadowed(
    painter: &mut Painter<'_>,
    env: &mut AeroTitleEnv<'_>,
    text: &str,
    x: f32,
    y: f32,
    style: TextStyle,
    color: Rgba,
    s: f32,
) -> f32 {
    env.text.draw(
        painter,
        env.mc_font,
        text,
        x,
        y + s.max(1.0),
        style,
        black(0.45 * color[3]),
    );
    env.text
        .draw(painter, env.mc_font, text, x, y, style, color)
}

/// One Aero glass button's body: the menu window's chrome at button size (drop shadow, a tint
/// that keeps white text readable over a bright scene, the 135-degree sheen). Hovering lifts the
/// tint; pressing darkens it. Its marks are [`draw_button_marks`].
fn draw_button_glass(painter: &mut Painter<'_>, look: ButtonLook, s: f32) {
    let ButtonLook {
        rect,
        radius,
        hover,
        press,
        ..
    } = look;
    painter.round_rect_shadow(
        rect,
        radius,
        [14.0 * s, 5.0 * s, 14.0 * s, 24.0 * s],
        black(0.20),
    );
    // Dark enough that a white label reads where the frosted greenhouse behind is near white
    // (its quartz beams); the hover lifts it, but the bar, border and chevron carry the hover.
    let top = mix(mix(black(0.22), black(0.10), hover), black(0.32), press);
    let bottom = mix(mix(black(0.34), black(0.24), hover), black(0.42), press);
    painter.round_rect(rect, radius, Fill::css_angle(rect, 180.0, top, bottom));
    // The 135-degree sheen in the top-left corner (the window's `linear-gradient(135deg, ...)`).
    let corner = Rect::new(rect.x, rect.y, rect.w * 0.6, rect.h * 1.4);
    painter.round_rect(
        rect,
        radius,
        Fill::css_angle(corner, 135.0, white(0.10 + 0.06 * hover), white(0.0)),
    );
}

/// One Aero glass button's marks: the top highlight, the 1 px light border, the icon and the
/// label, and while hovered the menu's bright left edge and a chevron. Drawn over
/// [`draw_button_glass`], and once more, crisp, after the VHS finish (see
/// [`AeroTitleScreen::build_layers`]).
fn draw_button_marks(
    painter: &mut Painter<'_>,
    env: &mut AeroTitleEnv<'_>,
    look: ButtonLook,
    (icon, label): (Icon, &str),
    layer: Layer,
    s: f32,
) {
    let ButtonLook {
        rect,
        radius,
        hover,
        icon_scale,
        ..
    } = look;
    // inset 0 1px 0 rgba(255,255,255,.2): the top highlight.
    painter.line(
        rect.x + radius * 0.8,
        rect.y + 1.5,
        rect.right() - radius * 0.8,
        rect.y + 1.5,
        1.0,
        white(0.26 + 0.14 * hover),
    );
    painter.round_rect_stroke(rect, radius, 1.0, 0.5, white(0.40 + 0.30 * hover));
    // The bright left edge of the menu's active row, grown by the hover.
    if hover > 0.01 {
        let bar_h = (rect.h - 26.0 * s) * hover;
        painter.round_rect(
            Rect::new(
                rect.x + 1.5 * s,
                rect.y + (rect.h - bar_h) * 0.5,
                3.0 * s,
                bar_h,
            ),
            1.5 * s,
            Fill::Solid(white(0.92 * hover)),
        );
    }

    let cy = rect.y + rect.h * 0.5;
    let icon_size = 24.0 * s * icon_scale * (1.0 + 0.06 * hover);
    if layer == Layer::Full {
        // A soft dark halo behind the icon keeps its thin strokes legible on bright glass.
        icons::draw(
            painter,
            icon,
            (rect.x + 32.0 * s, cy + 1.0 * s),
            icon_size,
            2.6 * s,
            black(0.16),
        );
    }
    icons::draw(
        painter,
        icon,
        (rect.x + 32.0 * s, cy),
        icon_size,
        1.8 * s,
        white(1.0),
    );

    let style = TextStyle::new(18.0 * s).weight(Weight::Semibold);
    let (_, line_h) = env.text.metrics(style);
    let (x, y) = (rect.x + 60.0 * s, cy - line_h * 0.5);
    let color = white(0.95 + 0.05 * hover);
    match layer {
        Layer::Full => haloed(painter, env, label, x, y, style, color, (s, 0.18)),
        Layer::Crisp => {
            env.text
                .draw(painter, env.mc_font, label, x, y, style, color);
        }
    }

    // The menu's "›" chevron on the right, faded in by the hover.
    if hover > 0.01 {
        let cx = rect.right() - 26.0 * s + (1.0 - hover) * -6.0 * s;
        let c = white(0.85 * hover);
        let lw = 1.7 * s;
        painter.line(cx - 2.5 * s, cy - 5.0 * s, cx + 2.5 * s, cy, lw, c);
        painter.line(cx + 2.5 * s, cy, cx - 2.5 * s, cy + 5.0 * s, lw, c);
    }
}

/// Text over a backdrop that can be as bright as the text itself (the greenhouse's white path
/// and beams): a soft dark halo, the glyphs drawn faintly at the eight offsets of two rings (one
/// and two pixels out; `halo` alpha on the inner ring, half on the outer), under the usual
/// shadow. `(s, halo)`: the scale and the halo strength.
#[allow(clippy::too_many_arguments)]
fn haloed(
    painter: &mut Painter<'_>,
    env: &mut AeroTitleEnv<'_>,
    text: &str,
    x: f32,
    y: f32,
    style: TextStyle,
    color: Rgba,
    (s, halo): (f32, f32),
) {
    for (ring, alpha) in [(2.0, halo * 0.5), (1.0, halo)] {
        let r = (s * ring).max(ring);
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
                black(alpha * color[3]),
            );
        }
    }
    shadowed(painter, env, text, x, y, style, color, s);
}

/// Vanilla's version and copyright lines, small and quiet, in the bottom corners.
fn draw_footer(
    painter: &mut Painter<'_>,
    env: &mut AeroTitleEnv<'_>,
    layout: &Layout,
    layer: Layer,
) {
    let s = layout.s;
    let style = TextStyle::new(14.0 * s).weight(Weight::Semibold);
    let (_, line_h) = env.text.metrics(style);
    let top = layout.footer_bottom - line_h;
    let color = white(0.86);
    let (version, copyright) = (env.version, env.copyright);
    let width = env.text.measure(copyright, style, env.mc_font);
    let right = layout.screen.0 - layout.left;
    for (text, x) in [(version, layout.left), (copyright, right - width)] {
        match layer {
            Layer::Full => haloed(painter, env, text, x, top, style, color, (s, 0.16)),
            Layer::Crisp => {
                env.text
                    .draw(painter, env.mc_font, text, x, top, style, color);
            }
        }
    }
}

/// A bright plain backdrop for when the greenhouse cannot be built (no vanilla jar, a failed
/// bake): sky at the top, foliage across the middle, a pale path at the bottom, as bright as the
/// real scene. `width`/`height` are the UI phase's scaled GUI size.
pub fn draw_stand_in_backdrop(list: &mut DrawList, width: i32, height: i32) {
    let band = |t: f32| (height as f32 * t).round() as i32;
    list.gradient_rect(0, 0, width, band(0.46), 0xFF6F_B8EA, 0xFFD2_EEFB);
    list.gradient_rect(0, band(0.46), width, band(0.74), 0xFF64_AE52, 0xFF2C_6E31);
    list.gradient_rect(0, band(0.74), width, height, 0xFFD8_E9D6, 0xFFF3_F8F1);
}

fn ms_since(since: Instant, now: Instant) -> f32 {
    now.saturating_duration_since(since).as_secs_f32() * 1000.0
}
