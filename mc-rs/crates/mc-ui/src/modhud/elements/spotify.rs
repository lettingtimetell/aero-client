//! The Spotify widget (`spotify`): the album cover, the
//! title and artist (scrolling when too long), a progress bar with times, and a control row
//! (previous, play/pause, next, and a volume slider while the player reports a volume).
//!
//! * **Data.** [`SpotifyHud`] in [`HudData::spotify`](crate::modhud::HudData) is plain data the
//!   client fills from `mc_media::NowPlaying` (`mc-client/src/app/mods/w9_spotify.rs`), only
//!   while the mod is on. The cover is a UI texture the client uploads under
//!   [`cover_texture`] when the art changes.
//! * **Layout.** [`SpotifyLayout`] is one pure function of the options, shared by the drawing
//!   and the client's click routing, so a button is hit exactly where it is drawn.
//! * **Two styles.** "Minecraft" (this file): the Minecraft font, a dark panel like the chat's,
//!   vanilla button art from `widgets.png` and pixel icons. "Aero" ([`super::spotify_aero`]):
//!   the mod menu's glass vocabulary and Segoe UI. Neither samples the scene: no extra pass, no
//!   scene copy, just quads in the UI pass the HUD already uses.
//! * **Cost.** Strings are rebuilt only when they change; the time labels once a second.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::rc::Rc;
use std::sync::LazyLock;

use mc_assets::ResourceLocation;
use mc_mods::ModSettings;

use crate::font::{FontRenderer, GlyphQuad, SolidRect, TextSink};
use crate::modhud::{HudCtx, HudElement};
use crate::screens::modmenu::paint::Rect;
use crate::screens::modmenu::text::MenuText;
use crate::widget::{DrawList, UiTexture};

use super::spotify_aero::AeroPainter;

/// The mod id (and the element id).
pub const MOD_ID: &str = "spotify";

/// The UI texture key the client uploads the album cover under (at most 128×128, linear).
#[must_use]
pub fn cover_texture() -> &'static UiTexture {
    static KEY: LazyLock<UiTexture> =
        LazyLock::new(|| UiTexture::Resource(ResourceLocation::new("mcrs", "spotify/cover")));
    &KEY
}

/// One of the widget's controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpotifyControl {
    /// Previous track.
    Previous,
    /// Play or pause.
    PlayPause,
    /// Next track.
    Next,
    /// The volume slider.
    Volume,
}

/// The `style` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpotifyStyle {
    /// The Minecraft font, a dark translucent panel, vanilla button art.
    #[default]
    Minecraft,
    /// The mod menu's glass look and Segoe UI.
    Aero,
}

/// The `controls` option: when the control row shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ControlsMode {
    /// Always.
    #[default]
    Always,
    /// Only while the cursor is free over the HUD (chat open), when it can be clicked.
    ChatOpen,
    /// Never (keys only).
    Never,
}

/// The widget's options, read once per settings change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotifyOptions {
    /// "Background opacity": the panel behind the widget, from none to the style's own
    /// darkness. Only the panel: the text, cover, progress bar and controls are always opaque
    /// (it used to fade the whole widget).
    pub background: f32,
    /// Look.
    pub style: SpotifyStyle,
    /// When the control row shows.
    pub controls: ControlsMode,
    /// Draw the album cover.
    pub cover: bool,
    /// Draw the progress bar and the times.
    pub progress: bool,
}

impl Default for SpotifyOptions {
    fn default() -> Self {
        SpotifyOptions {
            background: 1.0,
            style: SpotifyStyle::Minecraft,
            controls: ControlsMode::Always,
            cover: true,
            progress: true,
        }
    }
}

impl SpotifyOptions {
    /// The options of the `spotify` mod in `settings`.
    #[must_use]
    pub fn read(settings: &ModSettings) -> SpotifyOptions {
        use mc_mods::mods::spotify::ids;
        SpotifyOptions {
            background: (settings.f32(MOD_ID, ids::OPACITY) / 100.0).clamp(0.0, 1.0),
            style: match settings.choice(MOD_ID, ids::STYLE) {
                1 => SpotifyStyle::Aero,
                _ => SpotifyStyle::Minecraft,
            },
            controls: match settings.choice(MOD_ID, ids::CONTROLS) {
                1 => ControlsMode::ChatOpen,
                2 => ControlsMode::Never,
                _ => ControlsMode::Always,
            },
            cover: settings.bool(MOD_ID, ids::SHOW_COVER),
            progress: settings.bool(MOD_ID, ids::SHOW_PROGRESS),
        }
    }

    /// Whether the control row shows, given whether the cursor is free over the HUD.
    #[must_use]
    pub fn shows_controls(&self, interactive: bool) -> bool {
        match self.controls {
            ControlsMode::Always => true,
            ControlsMode::ChatOpen => interactive,
            ControlsMode::Never => false,
        }
    }
}

/// The Segoe UI text renderer the Aero style draws with. The client owns it and uploads its
/// glyph atlas into the UI renderer (under [`crate::screens::modmenu::text::atlas_key`]) whenever
/// [`MenuText::take_upload`] has pixels; the widget only lays glyphs out with it.
#[derive(Clone)]
pub struct AeroText(pub Rc<RefCell<MenuText>>);

impl AeroText {
    /// Wrap a text renderer for sharing.
    #[must_use]
    pub fn new(text: MenuText) -> AeroText {
        AeroText(Rc::new(RefCell::new(text)))
    }
}

impl std::fmt::Debug for AeroText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AeroText").finish_non_exhaustive()
    }
}

/// What the widget shows this frame. The client fills it in place each frame while the mod is
/// on (strings only when they change); with the mod off it stays at its default.
#[derive(Debug, Clone, Default)]
pub struct SpotifyHud {
    /// A matching player has a session. Without one the widget hides (outside the HUD editor).
    pub present: bool,
    /// Bumped whenever `title` or `artist` changed: the widget's text cache key.
    pub text_revision: u64,
    /// Track title.
    pub title: String,
    /// Artist line.
    pub artist: String,
    /// Playing (the button shows pause) or paused (it shows play).
    pub playing: bool,
    /// Elapsed time, extrapolated to this frame, if the player reports a timeline.
    pub position_ms: Option<u64>,
    /// Track length, if known.
    pub duration_ms: Option<u64>,
    /// The player's volume `0..=1`, if known (the slider shows only then). While the user drags
    /// the slider this is the dragged value.
    pub volume: Option<f32>,
    /// [`cover_texture`] holds this track's cover.
    pub cover: bool,
    /// The player accepts play/pause.
    pub can_play_pause: bool,
    /// The player accepts "next".
    pub can_next: bool,
    /// The player accepts "previous".
    pub can_previous: bool,
    /// The cursor is free over the HUD (chat is open): the controls are live.
    pub interactive: bool,
    /// The control under the cursor (only while `interactive`).
    pub hover: Option<SpotifyControl>,
    /// The control held down (a pressed button, or the volume slider while dragged).
    pub pressed: Option<SpotifyControl>,
    /// Segoe UI for the Aero style (the client sets it while that style is selected).
    pub aero_text: Option<AeroText>,
}

impl SpotifyHud {
    /// What the HUD editor shows when nothing is playing, so the widget can be placed.
    #[must_use]
    pub fn sample() -> SpotifyHud {
        SpotifyHud {
            present: true,
            text_revision: 0,
            title: "Song Title".to_owned(),
            artist: "Artist".to_owned(),
            playing: true,
            position_ms: Some(67_000),
            duration_ms: Some(204_000),
            volume: Some(0.6),
            cover: false,
            can_play_pause: true,
            can_next: true,
            can_previous: true,
            interactive: false,
            hover: None,
            pressed: None,
            aero_text: None,
        }
    }

    /// Whether the control is enabled (the player accepts it).
    #[must_use]
    pub fn accepts(&self, control: SpotifyControl) -> bool {
        match control {
            SpotifyControl::Previous => self.can_previous,
            SpotifyControl::PlayPause => self.can_play_pause,
            SpotifyControl::Next => self.can_next,
            SpotifyControl::Volume => self.volume.is_some(),
        }
    }

    /// The progress `0..=1`, if both the position and the length are known.
    #[must_use]
    pub fn progress(&self) -> Option<f32> {
        let (position, duration) = (self.position_ms?, self.duration_ms?);
        (duration > 0).then(|| (position as f64 / duration as f64).clamp(0.0, 1.0) as f32)
    }
}

// ---- layout ---------------------------------------------------------------------------------

/// The widget's width in GUI pixels (both styles).
pub const WIDTH: f32 = 160.0;
const PAD: f32 = 4.0;
const COVER: f32 = 32.0;
const INFO_H: f32 = 40.0;
const BUTTON_W: f32 = 18.0;
const BUTTON_H: f32 = 13.0;
const BUTTON_GAP: f32 = 2.0;
const CONTROLS_Y: f32 = INFO_H + 1.0;
const VOLUME_X: f32 = 79.0;
/// Extra grab margin around the volume track, left and right.
const VOLUME_GRAB: f32 = 4.0;

/// Where everything goes, in the element's local GUI pixels (before the HUD scale).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotifyLayout {
    /// The whole widget.
    pub size: (f32, f32),
    /// The cover square, when shown.
    pub cover: Option<Rect>,
    /// Left edge of the text column.
    pub text_x: f32,
    /// Width of the text column (lines longer than this scroll).
    pub text_w: f32,
    /// Top of the title's glyph cell (the Minecraft font's 8-pixel cell; Aero aligns baselines).
    pub title_y: f32,
    /// Top of the artist's glyph cell.
    pub artist_y: f32,
    /// Top of the time row's glyph cell, when the progress bar is shown.
    pub progress_y: Option<f32>,
    /// The previous / play-pause / next buttons, when the control row is shown.
    pub buttons: Option<[Rect; 3]>,
    /// The volume track (2–3 pixels tall), when the row is shown and the volume is known.
    pub volume: Option<Rect>,
    /// The speaker icon's top-left, next to the volume track.
    pub speaker: Option<(f32, f32)>,
}

impl SpotifyLayout {
    /// The layout for these options. `controls`: the row is shown (see
    /// [`SpotifyOptions::shows_controls`]); `volume`: the player reports a volume.
    #[must_use]
    pub fn new(options: &SpotifyOptions, controls: bool, volume: bool) -> SpotifyLayout {
        let cover = options.cover.then(|| Rect::new(PAD, PAD, COVER, COVER));
        let text_x = if options.cover {
            PAD + COVER + 5.0
        } else {
            PAD + 2.0
        };
        let text_w = WIDTH - PAD - 1.0 - text_x;
        let (title_y, artist_y, progress_y) = if options.progress {
            (5.0, 15.0, Some(26.0))
        } else {
            (10.0, 21.0, None)
        };
        let (buttons, volume, speaker, height) = if controls {
            let y = CONTROLS_Y;
            let button =
                |i: f32| Rect::new(PAD + i * (BUTTON_W + BUTTON_GAP), y, BUTTON_W, BUTTON_H);
            let volume =
                volume.then(|| Rect::new(VOLUME_X, y + 5.0, WIDTH - PAD - 1.0 - VOLUME_X, 3.0));
            let speaker = volume.map(|_| (VOLUME_X - 11.0, y + 3.0));
            (
                Some([button(0.0), button(1.0), button(2.0)]),
                volume,
                speaker,
                y + BUTTON_H + 3.0,
            )
        } else {
            (None, None, None, INFO_H)
        };
        SpotifyLayout {
            size: (WIDTH, height),
            cover,
            text_x,
            text_w,
            title_y,
            artist_y,
            progress_y,
            buttons,
            volume,
            speaker,
        }
    }

    /// The control at local `(x, y)`, if any.
    #[must_use]
    pub fn hit(&self, x: f32, y: f32) -> Option<SpotifyControl> {
        if let Some([previous, play, next]) = self.buttons {
            for (rect, control) in [
                (previous, SpotifyControl::Previous),
                (play, SpotifyControl::PlayPause),
                (next, SpotifyControl::Next),
            ] {
                if rect.contains(x, y) {
                    return Some(control);
                }
            }
        }
        let volume = self.volume?;
        let grab = Rect::new(
            volume.x - VOLUME_GRAB,
            CONTROLS_Y,
            volume.w + 2.0 * VOLUME_GRAB,
            BUTTON_H,
        );
        grab.contains(x, y).then_some(SpotifyControl::Volume)
    }

    /// The volume a press or drag at local `x` selects, `0..=1`.
    #[must_use]
    pub fn volume_at(&self, x: f32) -> Option<f32> {
        let track = self.volume?;
        Some(((x - track.x) / track.w.max(1.0)).clamp(0.0, 1.0))
    }
}

// ---- shared helpers ---------------------------------------------------------------------------

/// How far a line `overflow` pixels too long has scrolled `t_ms` after it appeared: it waits,
/// scrolls to its end at a reading pace, waits, and starts over.
#[must_use]
pub fn marquee_offset(overflow: f32, t_ms: u64) -> f32 {
    const HOLD_START_MS: u64 = 2500;
    const HOLD_END_MS: u64 = 1500;
    /// GUI pixels per second.
    const SPEED: f32 = 24.0;
    if overflow <= 0.0 {
        return 0.0;
    }
    let scroll_ms = (overflow / SPEED * 1000.0).ceil() as u64;
    let cycle = HOLD_START_MS + scroll_ms + HOLD_END_MS;
    let t = t_ms % cycle;
    if t < HOLD_START_MS {
        0.0
    } else if t < HOLD_START_MS + scroll_ms {
        ((t - HOLD_START_MS) as f32 * SPEED / 1000.0).min(overflow)
    } else {
        overflow
    }
}

/// `m:ss`, or `h:mm:ss` from an hour on.
pub fn write_time(out: &mut String, ms: u64) {
    let s = ms / 1000;
    if s >= 3600 {
        let _ = write!(out, "{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60);
    } else {
        let _ = write!(out, "{}:{:02}", s / 60, s % 60);
    }
}

/// A [`TextSink`] that clips Minecraft-font glyphs to the column `x0..x1` (texture coordinates
/// cut to match), for the scrolling title and artist.
pub struct ClipSink<'a> {
    /// Where the clipped quads go.
    pub list: &'a mut DrawList,
    /// Left edge of the column.
    pub x0: f32,
    /// Right edge of the column.
    pub x1: f32,
}

impl TextSink for ClipSink<'_> {
    fn glyph(&mut self, quad: GlyphQuad) {
        let left = quad.pos[0][0].min(quad.pos[1][0]);
        let right = quad.pos[2][0].max(quad.pos[3][0]);
        if right <= self.x0 || left >= self.x1 {
            return;
        }
        let mut quad = quad;
        if left < self.x0 || right > self.x1 {
            // The top edge runs corner 0 -> 2, the bottom edge 1 -> 3 (strip order).
            for (a, b) in [(0, 2), (1, 3)] {
                let (xa, xb) = (quad.pos[a][0], quad.pos[b][0]);
                let width = xb - xa;
                if width <= 0.0 {
                    return;
                }
                let (ua, ub) = (quad.uv[a][0], quad.uv[b][0]);
                let (na, nb) = (xa.max(self.x0), xb.min(self.x1));
                if nb <= na {
                    return;
                }
                quad.uv[a][0] = ua + (ub - ua) * (na - xa) / width;
                quad.uv[b][0] = ua + (ub - ua) * (nb - xa) / width;
                quad.pos[a][0] = na;
                quad.pos[b][0] = nb;
            }
        }
        self.list.glyph(quad);
    }

    fn bar(&mut self, rect: SolidRect) {
        let (x0, x1) = (rect.x0.max(self.x0), rect.x1.min(self.x1));
        if x1 > x0 {
            self.list.bar(SolidRect { x0, x1, ..rect });
        }
    }
}

/// Round `v` to whole physical pixels, `pixel` of them per local unit.
pub(crate) fn snap(v: f32, pixel: f32) -> f32 {
    if pixel > 0.0 {
        (v * pixel).round() / pixel
    } else {
        v
    }
}

// ---- the element ------------------------------------------------------------------------------

/// Text the Minecraft style measured, rebuilt only when the strings or the font change.
#[derive(Debug, Default)]
struct McText {
    revision: Option<u64>,
    font_generation: u64,
    title_w: f32,
    artist_w: f32,
}

/// The two time labels, rebuilt once a second.
#[derive(Debug, Default)]
pub(crate) struct TimeLabels {
    /// `(elapsed s, total s)` the strings were built for.
    key: Option<(u64, u64)>,
    /// Elapsed, `m:ss`.
    pub(crate) elapsed: String,
    /// Total, `m:ss`.
    pub(crate) total: String,
}

impl TimeLabels {
    /// Rebuild for `data` if the whole seconds changed. Returns whether the labels apply (the
    /// player reports a timeline).
    pub(crate) fn update(&mut self, data: &SpotifyHud) -> bool {
        let (Some(position), Some(duration)) = (data.position_ms, data.duration_ms) else {
            return false;
        };
        let key = (position / 1000, duration / 1000);
        if self.key != Some(key) {
            self.key = Some(key);
            self.elapsed.clear();
            write_time(&mut self.elapsed, position.min(duration));
            self.total.clear();
            write_time(&mut self.total, duration);
        }
        true
    }
}

/// The Spotify HUD element.
#[derive(Debug)]
pub struct SpotifyElement {
    options: SpotifyOptions,
    layout: SpotifyLayout,
    /// What the HUD editor shows when nothing plays.
    sample: SpotifyHud,
    /// When (`HudData::now_ms`) the shown text appeared: the marquee clock.
    text_since_ms: u64,
    text_revision: Option<u64>,
    mc: McText,
    times: TimeLabels,
    aero: AeroPainter,
}

impl Default for SpotifyElement {
    fn default() -> Self {
        let options = SpotifyOptions::default();
        SpotifyElement {
            options,
            layout: SpotifyLayout::new(&options, true, true),
            sample: SpotifyHud::sample(),
            text_since_ms: 0,
            text_revision: None,
            mc: McText::default(),
            times: TimeLabels::default(),
            aero: AeroPainter::default(),
        }
    }
}

impl SpotifyElement {
    /// The layout of the last `layout` call (tests and the render harness).
    #[must_use]
    pub fn current_layout(&self) -> &SpotifyLayout {
        &self.layout
    }

    /// The options the element synced (tests).
    #[must_use]
    pub fn options(&self) -> &SpotifyOptions {
        &self.options
    }

    /// What the Aero style replays each frame: `(cached shape triangles, glyph quads)` (the
    /// glyph quads before clipping, shadows included). Zero until it has drawn once.
    #[must_use]
    pub fn aero_geometry(&self) -> (usize, usize) {
        (self.aero.shape_triangles(), self.aero.text_quads())
    }

    /// The data to show: the live track, or the sample in the HUD editor.
    fn shown<'a>(&'a self, ctx: &'a HudCtx<'_>) -> Option<&'a SpotifyHud> {
        let live = &ctx.data.spotify;
        if live.present {
            Some(live)
        } else if ctx.editing {
            Some(&self.sample)
        } else {
            None
        }
    }

    fn draw_minecraft(
        &mut self,
        data: &SpotifyHud,
        now_ms: u64,
        pixel: f32,
        font: &mut FontRenderer,
        d: &mut DrawList,
    ) {
        let l = self.layout;
        let (w, h) = l.size;
        // The chat's backdrop: `drawRect(.., (opacity * 255 / 2) << 24)` at full opacity, scaled
        // by Background opacity (and left out at 0%). Nothing else fades.
        let backdrop = panel_alpha(0x7F, self.options.background);
        if backdrop != 0 {
            d.quad_rect(
                &UiTexture::Solid,
                (0.0, 0.0, w, h),
                (0.0, 0.0, 1.0, 1.0),
                backdrop << 24,
            );
        }

        if let Some(c) = l.cover {
            if data.cover {
                d.quad_rect(
                    cover_texture(),
                    (c.x, c.y, c.right(), c.bottom()),
                    (0.0, 0.0, 1.0, 1.0),
                    0xFFFF_FFFF,
                );
            } else {
                d.quad_rect(
                    &UiTexture::Solid,
                    (c.x, c.y, c.right(), c.bottom()),
                    (0.0, 0.0, 1.0, 1.0),
                    0xFF22_2222,
                );
                mc_note(d, c.x as i32 + 12, c.y as i32 + 10);
            }
        }

        // Title and artist: measured once per change, scrolled and clipped to the column.
        if self.mc.revision != Some(data.text_revision)
            || self.mc.font_generation != font.layout_generation()
        {
            self.mc.revision = Some(data.text_revision);
            self.mc.font_generation = font.layout_generation();
            self.mc.title_w = font.string_width(&data.title).max(0) as f32;
            self.mc.artist_w = font.string_width(&data.artist).max(0) as f32;
        }
        let t = now_ms.saturating_sub(self.text_since_ms);
        let (x0, x1) = (l.text_x, l.text_x + l.text_w);
        let title_dx = snap(marquee_offset(self.mc.title_w - l.text_w, t), pixel);
        let artist_dx = snap(marquee_offset(self.mc.artist_w - l.text_w, t), pixel);
        let mut sink = ClipSink { list: d, x0, x1 };
        font.draw_string(&mut sink, &data.title, x0 - title_dx, l.title_y, -1, true);
        font.draw_string(
            &mut sink,
            &data.artist,
            x0 - artist_dx,
            l.artist_y,
            0xFFAA_AAAA_u32 as i32,
            true,
        );

        // Progress: elapsed, the bar, total.
        if let Some(py) = l.progress_y {
            if self.times.update(data) {
                let total_w = font.string_width(&self.times.total).max(0) as f32;
                font.draw_string(d, &self.times.elapsed, x0, py, 0xFFAA_AAAA_u32 as i32, true);
                font.draw_string(
                    d,
                    &self.times.total,
                    x1 - total_w,
                    py,
                    0xFFAA_AAAA_u32 as i32,
                    true,
                );
                let bar_x0 = (x0 + total_w + 3.0).round();
                let bar_x1 = (x1 - total_w - 4.0).round();
                if bar_x1 > bar_x0 {
                    let y = py + 3.0;
                    d.quad_rect(
                        &UiTexture::Solid,
                        (bar_x0 + 1.0, y + 1.0, bar_x1 + 1.0, y + 3.0),
                        (0.0, 0.0, 1.0, 1.0),
                        0xFF3F_3F3F,
                    );
                    d.quad_rect(
                        &UiTexture::Solid,
                        (bar_x0, y, bar_x1, y + 2.0),
                        (0.0, 0.0, 1.0, 1.0),
                        0xFF55_5555,
                    );
                    let fill = data.progress().unwrap_or(0.0);
                    let x = bar_x0 + ((bar_x1 - bar_x0) * fill).round();
                    if x > bar_x0 {
                        d.quad_rect(
                            &UiTexture::Solid,
                            (bar_x0, y, x, y + 2.0),
                            (0.0, 0.0, 1.0, 1.0),
                            0xFFFF_FFFF,
                        );
                    }
                }
            }
        }

        // Controls: vanilla button art, pixel icons, the volume slider.
        if let Some(buttons) = l.buttons {
            let controls = [
                SpotifyControl::Previous,
                SpotifyControl::PlayPause,
                SpotifyControl::Next,
            ];
            for (rect, control) in buttons.into_iter().zip(controls) {
                let enabled = data.accepts(control);
                let lit = data.interactive
                    && (data.hover == Some(control) || data.pressed == Some(control));
                let state = if !enabled {
                    0
                } else if lit {
                    2
                } else {
                    1
                };
                mc_button(d, rect, state);
                // GuiButton's text colours: 0xE0E0E0, 0xFFFFA0 hovered, 0xA0A0A0 disabled.
                let color = match state {
                    0 => 0xFFA0_A0A0,
                    2 => 0xFFFF_FFA0,
                    _ => 0xFFE0_E0E0,
                };
                let icon = match control {
                    SpotifyControl::Previous => McIcon::Previous,
                    SpotifyControl::Next => McIcon::Next,
                    _ if data.playing => McIcon::Pause,
                    _ => McIcon::Play,
                };
                let iw = icon.width();
                let ix = rect.x as i32 + ((rect.w as i32 - iw) / 2);
                let iy = rect.y as i32 + 3;
                icon.draw(d, ix + 1, iy + 1, 0xFF38_3838);
                icon.draw(d, ix, iy, color);
            }
            if let (Some(track), Some((sx, sy))) = (l.volume, l.speaker) {
                let volume = data.volume.unwrap_or(0.0).clamp(0.0, 1.0);
                mc_speaker(d, sx as i32 + 1, sy as i32 + 1, volume, 0xFF38_3838);
                mc_speaker(d, sx as i32, sy as i32, volume, 0xFFE0_E0E0);
                let (x0, x1) = (track.x, track.right());
                let y = track.y.round();
                d.quad_rect(
                    &UiTexture::Solid,
                    (x0 + 1.0, y + 1.0, x1 + 1.0, y + 3.0),
                    (0.0, 0.0, 1.0, 1.0),
                    0xFF3F_3F3F,
                );
                d.quad_rect(
                    &UiTexture::Solid,
                    (x0, y, x1, y + 2.0),
                    (0.0, 0.0, 1.0, 1.0),
                    0xFF55_5555,
                );
                let knob = (x0 + (x1 - x0) * volume).round();
                if knob > x0 {
                    d.quad_rect(
                        &UiTexture::Solid,
                        (x0, y, knob, y + 2.0),
                        (0.0, 0.0, 1.0, 1.0),
                        0xFFFF_FFFF,
                    );
                }
                // A small vanilla-slider handle: black outline, button grey (white when held).
                let lit = data.interactive
                    && (data.hover == Some(SpotifyControl::Volume)
                        || data.pressed == Some(SpotifyControl::Volume));
                let (kx, ky) = (knob as i32, y as i32 + 1);
                d.rect(kx - 2, ky - 4, kx + 2, ky + 4, 0xFF00_0000);
                d.rect(
                    kx - 1,
                    ky - 3,
                    kx + 1,
                    ky + 3,
                    if lit { 0xFFFF_FFFF } else { 0xFFC6_C6C6 },
                );
            }
        }
    }
}

impl HudElement for SpotifyElement {
    fn id(&self) -> (&'static str, &'static str) {
        (MOD_ID, MOD_ID)
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.options = SpotifyOptions::read(settings);
        self.aero.invalidate();
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, _font: &FontRenderer) -> Option<(f32, f32)> {
        let data = self.shown(ctx)?;
        let controls = self.options.shows_controls(data.interactive || ctx.editing);
        let volume = data.volume.is_some();
        let revision = data.text_revision;
        let present = ctx.data.spotify.present;
        self.layout = SpotifyLayout::new(&self.options, controls, volume);
        // The marquee clock restarts with every new title.
        let key = if present { Some(revision) } else { None };
        if self.text_revision != key {
            self.text_revision = key;
            self.text_since_ms = ctx.data.now_ms;
        }
        Some(self.layout.size)
    }

    fn draw(&mut self, ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        // Physical pixels per local unit: the GUI scale times the HUD scale the pipeline pushed.
        let pixel = ctx.pixel_scale * d.transform().determinant().abs().sqrt();
        if ctx.data.spotify.present {
            self.draw_data(&ctx.data.spotify, ctx, pixel, font, d);
        } else if ctx.editing {
            // The editor's sample, moved out for the call (no per-frame clone).
            let sample = std::mem::take(&mut self.sample);
            self.draw_data(&sample, ctx, pixel, font, d);
            self.sample = sample;
        }
    }
}

impl SpotifyElement {
    fn draw_data(
        &mut self,
        data: &SpotifyHud,
        ctx: &HudCtx<'_>,
        pixel: f32,
        font: &mut FontRenderer,
        d: &mut DrawList,
    ) {
        // Background opacity reaches only the panel (each style applies it there); the text,
        // cover, progress bar and controls are drawn at full alpha.
        match self.options.style {
            SpotifyStyle::Minecraft => self.draw_minecraft(data, ctx.data.now_ms, pixel, font, d),
            SpotifyStyle::Aero => {
                // The client's Segoe UI renderer rides on the live data even with no track.
                let text = ctx.data.spotify.aero_text.as_ref();
                let t = ctx.data.now_ms.saturating_sub(self.text_since_ms);
                self.times.update(data);
                let background = self.options.background;
                self.aero.draw(
                    &self.layout,
                    data,
                    &self.times,
                    text,
                    t,
                    pixel,
                    background,
                    font,
                    d,
                );
            }
        }
    }
}

/// A panel's alpha byte, `full` at 100% Background opacity, scaled by `background`
/// (`0.0..=1.0`) and rounded.
#[must_use]
fn panel_alpha(full: u32, background: f32) -> u32 {
    if background >= 1.0 {
        full
    } else {
        (full as f32 * background.max(0.0)).round() as u32
    }
}

// ---- Minecraft-style pieces --------------------------------------------------------------------

/// A vanilla button at any size: the four corners of the 200×20 `widgets.png` sprite, the way
/// `GuiButton` halves it horizontally (here also vertically, for a 13-pixel row).
/// `state`: 0 disabled, 1 idle, 2 hovered (`GuiButton.getHoverState`).
fn mc_button(d: &mut DrawList, rect: Rect, state: i32) {
    let tex = UiTexture::widgets();
    let (x, y, w, h) = (rect.x as i32, rect.y as i32, rect.w as i32, rect.h as i32);
    let v = 46 + state * 20;
    let (lw, rw) = (w / 2, w - w / 2);
    let (th, bh) = (h / 2, h - h / 2);
    d.blit(tex, x, y, 0, v, lw, th);
    d.blit(tex, x + lw, y, 200 - rw, v, rw, th);
    d.blit(tex, x, y + th, 0, v + 20 - bh, lw, bh);
    d.blit(tex, x + lw, y + th, 200 - rw, v + 20 - bh, rw, bh);
}

/// The control icons as Minecraft-style pixel art (7 pixels tall).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McIcon {
    Previous,
    Play,
    Pause,
    Next,
}

impl McIcon {
    fn width(self) -> i32 {
        match self {
            McIcon::Play => 4,
            McIcon::Pause => 6,
            McIcon::Previous | McIcon::Next => 5,
        }
    }

    fn draw(self, d: &mut DrawList, x: i32, y: i32, argb: u32) {
        // Column `c` of a right-pointing triangle spans rows c..7-c.
        let right = |d: &mut DrawList, x: i32| {
            for c in 0..4 {
                d.rect(x + c, y + c, x + c + 1, y + 7 - c, argb);
            }
        };
        let left = |d: &mut DrawList, x: i32| {
            for c in 0..4 {
                d.rect(x + 3 - c, y + c, x + 4 - c, y + 7 - c, argb);
            }
        };
        match self {
            McIcon::Play => right(d, x),
            McIcon::Pause => {
                d.rect(x, y, x + 2, y + 7, argb);
                d.rect(x + 4, y, x + 6, y + 7, argb);
            }
            McIcon::Next => {
                right(d, x);
                d.rect(x + 4, y, x + 5, y + 7, argb);
            }
            McIcon::Previous => {
                d.rect(x, y, x + 1, y + 7, argb);
                left(d, x + 1);
            }
        }
    }
}

/// A 7-pixel speaker with one or two sound waves by volume.
fn mc_speaker(d: &mut DrawList, x: i32, y: i32, volume: f32, argb: u32) {
    d.rect(x, y + 2, x + 2, y + 5, argb);
    d.rect(x + 2, y + 1, x + 3, y + 6, argb);
    d.rect(x + 3, y, x + 4, y + 7, argb);
    if volume > 0.0 {
        d.rect(x + 5, y + 2, x + 6, y + 5, argb);
    }
    if volume > 0.5 {
        d.rect(x + 7, y + 1, x + 8, y + 6, argb);
    }
}

/// A pixel-art eighth note for a missing cover (8×11).
fn mc_note(d: &mut DrawList, x: i32, y: i32) {
    let c = 0xFF77_7777;
    d.rect(x + 3, y, x + 4, y + 9, c);
    d.rect(x + 4, y, x + 6, y + 1, c);
    d.rect(x + 6, y + 1, x + 7, y + 3, c);
    d.rect(x, y + 8, x + 3, y + 11, c);
    d.rect(x + 1, y + 7, x + 3, y + 8, c);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::HudData;
    use crate::widget::test_support::uniform_font;

    fn ctx<'a>(data: &'a HudData, settings: &'a ModSettings, editing: bool) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen: (640.0, 360.0),
            pixel_scale: 3.0,
            editing,
        }
    }

    fn playing() -> HudData {
        HudData {
            now_ms: 10_000,
            spotify: SpotifyHud {
                text_revision: 1,
                ..SpotifyHud::sample()
            },
            ..HudData::default()
        }
    }

    fn on() -> ModSettings {
        let mut settings = ModSettings::new();
        settings.set_enabled(MOD_ID, true);
        settings
    }

    /// "Background opacity" fades the panel and nothing else, in
    /// both styles. At 0% the panel is left out and every other vertex is exactly the 100% one;
    /// at 50% the panel's vertices (drawn first) keep position, UV and colour at half the alpha,
    /// and the text, cover, progress bar and controls are untouched. (Before this change every
    /// vertex of the widget faded, and 0% drew nothing at all.)
    #[test]
    fn background_opacity_fades_only_the_panel_in_both_styles() {
        let mut settings = on();
        let mut data = playing();
        data.spotify.cover = true;
        let mut font = uniform_font();
        for style in [0, 1] {
            settings.set_choice(MOD_ID, "style", style);
            let mut element = SpotifyElement::default();
            let mut render = |settings: &ModSettings| {
                element.sync(settings);
                let context = ctx(&data, settings, false);
                element.layout(&context, &font).unwrap();
                let mut d = DrawList::new();
                element.draw(&context, &mut font, &mut d);
                d
            };
            settings.set_f32(MOD_ID, "opacity", 100.0);
            let full = render(&settings);
            settings.set_f32(MOD_ID, "opacity", 50.0);
            let half = render(&settings);
            settings.set_f32(MOD_ID, "opacity", 0.0);
            let none = render(&settings);
            let (full, half, none) = (full.vertices(), half.vertices(), none.vertices());
            assert_eq!(full.len(), half.len(), "style {style}");
            let panel = full.len() - none.len();
            assert!(
                panel >= 6,
                "style {style}: 0% leaves the panel out ({panel})"
            );
            assert!(none.len() > 100, "style {style}: and draws the rest");
            let same = |a: &crate::widget::UiVertex, b: &crate::widget::UiVertex| {
                a.pos == b.pos && a.uv == b.uv && a.color == b.color
            };
            for (i, (a, b)) in full[panel..].iter().zip(none).enumerate() {
                assert!(
                    same(a, b),
                    "style {style}: vertex {i} after the panel moved at 0%"
                );
            }
            for (i, (a, b)) in full[panel..].iter().zip(&half[panel..]).enumerate() {
                assert!(
                    same(a, b),
                    "style {style}: vertex {i} after the panel faded at 50%"
                );
            }
            // Minecraft rounds its alpha byte (127 -> 64); Aero scales floats.
            let tolerance = if style == 0 { 0.6 / 255.0 } else { 1e-5 };
            for (a, b) in full[..panel].iter().zip(&half[..panel]) {
                assert_eq!((a.pos, a.uv), (b.pos, b.uv));
                assert_eq!(&a.color[..3], &b.color[..3]);
                assert!(
                    (b.color[3] - a.color[3] * 0.5).abs() <= tolerance,
                    "style {style}: {} -> {}",
                    a.color[3],
                    b.color[3]
                );
            }
            assert!(
                full[..panel].iter().any(|v| v.color[3] > 0.3),
                "style {style}: the faded prefix really is the dark panel"
            );
        }
    }

    #[test]
    fn the_layout_follows_the_options() {
        let full = SpotifyLayout::new(&SpotifyOptions::default(), true, true);
        assert_eq!(full.size, (WIDTH, 57.0));
        assert_eq!(full.cover, Some(Rect::new(4.0, 4.0, 32.0, 32.0)));
        assert!(full.buttons.is_some() && full.volume.is_some());
        // No volume reported: the buttons stay, the slider goes.
        let quiet = SpotifyLayout::new(&SpotifyOptions::default(), true, false);
        assert!(quiet.buttons.is_some() && quiet.volume.is_none() && quiet.speaker.is_none());
        // No controls: just the info block.
        let bare = SpotifyLayout::new(&SpotifyOptions::default(), false, true);
        assert_eq!(bare.size, (WIDTH, INFO_H));
        assert!(bare.buttons.is_none() && bare.volume.is_none());
        // No cover: the text column takes the whole width; no progress: the lines centre.
        let options = SpotifyOptions {
            cover: false,
            progress: false,
            ..SpotifyOptions::default()
        };
        let text_only = SpotifyLayout::new(&options, false, false);
        assert!(text_only.cover.is_none() && text_only.progress_y.is_none());
        assert!(text_only.text_w > full.text_w + 30.0);
        assert!(text_only.title_y > full.title_y);
    }

    #[test]
    fn hits_land_exactly_on_the_drawn_controls() {
        let l = SpotifyLayout::new(&SpotifyOptions::default(), true, true);
        let [previous, play, next] = l.buttons.unwrap();
        let centre = |r: Rect| r.center();
        assert_eq!(
            l.hit(centre(previous).0, centre(previous).1),
            Some(SpotifyControl::Previous)
        );
        assert_eq!(
            l.hit(centre(play).0, centre(play).1),
            Some(SpotifyControl::PlayPause)
        );
        assert_eq!(
            l.hit(centre(next).0, centre(next).1),
            Some(SpotifyControl::Next)
        );
        // The gap between two buttons, the cover and the text are not controls.
        assert_eq!(l.hit(previous.right() + 0.5, previous.y + 2.0), None);
        assert_eq!(l.hit(10.0, 10.0), None);
        assert_eq!(l.hit(100.0, 8.0), None);
        // The volume track grabs a few pixels around it, and maps x to 0..=1.
        let track = l.volume.unwrap();
        assert_eq!(
            l.hit(track.x - 2.0, track.y + 1.0),
            Some(SpotifyControl::Volume)
        );
        assert_eq!(l.volume_at(track.x), Some(0.0));
        assert_eq!(l.volume_at(track.right() + 10.0), Some(1.0));
        assert!((l.volume_at(track.x + track.w * 0.25).unwrap() - 0.25).abs() < 1e-6);
        // Every control is inside the widget.
        for r in [previous, play, next, track] {
            assert!(r.x >= 0.0 && r.right() <= l.size.0 && r.bottom() <= l.size.1);
        }
    }

    #[test]
    fn the_marquee_waits_scrolls_waits_and_restarts() {
        assert_eq!(
            marquee_offset(0.0, 99_999),
            0.0,
            "a line that fits never moves"
        );
        assert_eq!(marquee_offset(48.0, 0), 0.0);
        assert_eq!(marquee_offset(48.0, 2_400), 0.0, "holds first");
        let mid = marquee_offset(48.0, 3_500);
        assert!(mid > 20.0 && mid < 28.0, "{mid}");
        assert_eq!(
            marquee_offset(48.0, 2_500 + 2_000 + 100),
            48.0,
            "holds at the end"
        );
        assert_eq!(
            marquee_offset(48.0, 2_500 + 2_000 + 1_500),
            0.0,
            "and starts over"
        );
    }

    #[test]
    fn times_format_like_spotify() {
        let mut s = String::new();
        write_time(&mut s, 67_900);
        assert_eq!(s, "1:07");
        s.clear();
        write_time(&mut s, 3_725_000);
        assert_eq!(s, "1:02:05");
    }

    #[test]
    fn nothing_shows_without_a_player_except_in_the_editor() {
        let settings = on();
        let font = uniform_font();
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        let idle = HudData::default();
        assert_eq!(element.layout(&ctx(&idle, &settings, false), &font), None);
        let sample = element.layout(&ctx(&idle, &settings, true), &font);
        assert_eq!(
            sample,
            Some((WIDTH, 57.0)),
            "the editor gets the full sample"
        );
        let live = playing();
        assert!(element
            .layout(&ctx(&live, &settings, false), &font)
            .is_some());
    }

    #[test]
    fn controls_follow_the_mode_and_the_free_cursor() {
        let mut settings = on();
        let font = uniform_font();
        let mut element = SpotifyElement::default();
        settings.set_choice(MOD_ID, "controls", 1);
        element.sync(&settings);
        let mut data = playing();
        let (_, h) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(h, INFO_H, "chat closed: no row");
        data.spotify.interactive = true;
        let (_, h) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(h, 57.0, "chat open: the row appears");
        settings.set_choice(MOD_ID, "controls", 2);
        element.sync(&settings);
        let (_, h) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(h, INFO_H, "never");
    }

    #[test]
    fn the_minecraft_style_draws_the_cover_only_when_uploaded() {
        let settings = on();
        let mut font = uniform_font();
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        let mut data = playing();
        let uses_cover = |d: &DrawList| d.batches().iter().any(|b| &b.texture == cover_texture());
        element.layout(&ctx(&data, &settings, false), &font);
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings, false), &mut font, &mut d);
        assert!(
            !uses_cover(&d),
            "no art yet: the placeholder, never the cover key"
        );
        data.spotify.cover = true;
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings, false), &mut font, &mut d);
        assert!(uses_cover(&d));
        // Vanilla button art for the three buttons.
        assert!(d
            .batches()
            .iter()
            .any(|b| &b.texture == UiTexture::widgets()));
        // Everything stays inside the widget.
        let (w, h) = element.current_layout().size;
        for v in d.vertices() {
            assert!(v.pos[0] >= 0.0 && v.pos[0] <= w + 1.0, "{:?}", v.pos);
            assert!(v.pos[1] >= 0.0 && v.pos[1] <= h + 1.0, "{:?}", v.pos);
        }
    }

    #[test]
    fn a_long_title_scrolls_inside_its_column() {
        let settings = on();
        let mut font = uniform_font();
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        let mut data = playing();
        // 5 px per glyph: 60 glyphs is far wider than the 115-pixel column.
        data.spotify.title = "x".repeat(60);
        data.spotify.text_revision = 2;
        element.layout(&ctx(&data, &settings, false), &font);
        let l = *element.current_layout();
        let glyph_xs = |element: &mut SpotifyElement, data: &HudData, font: &mut FontRenderer| {
            let mut d = DrawList::new();
            element.draw(&ctx(data, &settings, false), font, &mut d);
            d.vertices()
                .iter()
                .filter(|v| v.pos[1] >= l.title_y && v.pos[1] <= l.title_y + 9.0)
                .filter(|v| v.pos[0] >= l.text_x - 0.01)
                .map(|v| v.pos[0])
                .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)))
        };
        let (lo, hi) = glyph_xs(&mut element, &data, &mut font);
        assert!(
            lo >= l.text_x - 0.01 && hi <= l.text_x + l.text_w + 0.01,
            "{lo} {hi}"
        );
        // Four seconds later the line has moved left, still clipped to the column.
        data.now_ms += 4_000;
        element.layout(&ctx(&data, &settings, false), &font);
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings, false), &mut font, &mut d);
        let moved = marquee_offset(
            font.string_width(&data.spotify.title) as f32 - l.text_w,
            4_000,
        );
        assert!(moved > 10.0);
        for v in d.vertices() {
            if v.pos[1] >= l.title_y && v.pos[1] < l.title_y + 8.0 && v.pos[0] > l.text_x {
                assert!(v.pos[0] <= l.text_x + l.text_w + 0.01, "{:?}", v.pos);
            }
        }
        // A new track restarts the clock.
        data.spotify.text_revision = 3;
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(element.text_since_ms, data.now_ms);
    }

    #[test]
    fn time_labels_are_rebuilt_once_a_second() {
        let mut labels = TimeLabels::default();
        let mut data = SpotifyHud::sample();
        assert!(labels.update(&data));
        assert_eq!(
            (labels.elapsed.as_str(), labels.total.as_str()),
            ("1:07", "3:24")
        );
        let ptr = labels.elapsed.as_ptr();
        data.position_ms = Some(67_900);
        labels.update(&data);
        assert_eq!(labels.elapsed.as_ptr(), ptr, "same second: untouched");
        data.position_ms = Some(68_000);
        labels.update(&data);
        assert_eq!(labels.elapsed, "1:08");
        data.duration_ms = None;
        assert!(!labels.update(&data), "no timeline: no labels");
    }

    #[test]
    fn disabled_and_hovered_buttons_use_the_vanilla_states() {
        let settings = on();
        let mut font = uniform_font();
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        let mut data = playing();
        data.spotify.can_previous = false;
        data.spotify.interactive = true;
        data.spotify.hover = Some(SpotifyControl::Next);
        element.layout(&ctx(&data, &settings, false), &font);
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings, false), &mut font, &mut d);
        let has = |argb: u32| {
            let c = crate::widget::unpack_argb(argb);
            d.vertices().iter().any(|v| v.color == c)
        };
        assert!(has(0xFFA0_A0A0), "disabled previous: grey icon");
        assert!(has(0xFFFF_FFA0), "hovered next: yellow icon");
        assert!(has(0xFFE0_E0E0), "idle play/pause");
    }
}
