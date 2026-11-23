//! The Spotify widget's client half: the media hub's
//! lifecycle, the widget's data, the cover and Segoe UI textures, clicks and keys.
//!
//! | hook | where | cost with the mod off |
//! |---|---|---|
//! | [`Client::w9_frame`] | `<<FB-W9:frame>>`, end of `Client::frame` | a `u64` compare, two `bool`s |
//! | [`Client::w9_shell_frame`] | `<<FB-W9:shell-frame>>`, `ShellApp::frame` | a `u64` compare, a `bool` |
//! | [`Client::w9_mouse_pressed`], [`Client::w9_mouse_released`] | `<<FB-W9:mouse>>` | a `bool` |
//! | [`W9State::fill_hud`] | `<<FB-W9:fill>>`, `fill_hud_data` (only while a HUD mod is on) | a `bool` |
//! | [`W9State::upload_textures`] | `<<FB-W9:upload>>`, `build_ui`, before `ensure_list_textures` | a few `bool`s |
//!
//! * **The hub lives only while the mod is on.** It is started when the mod turns on (the first
//!   frame after) and dropped when it turns off, which stops its worker. The game thread only
//!   calls the hub's cheap methods: `generation` every frame (an atomic load), `state` when that
//!   moved (a short lock and an `Arc` clone), `send` on a click or a key.
//! * **Clicks while chat is open.** Chat is the screen where the cursor is free over the live,
//!   undimmed HUD, and where a click outside the chat box does nothing in vanilla, so taking
//!   clicks on the widget there costs nothing. Not in containers (the HUD is dimmed behind them
//!   and a click outside the window is vanilla's "drop the held stack"), not in the HUD editor
//!   (clicks move and scale elements), not over the mod menu (its windows cover the HUD).
//! * **Keys** are optional (unbound by default) and work in game only, as press edges.
//! * **The cover** is uploaded only when `art_generation` changes (or after the renderer lost
//!   it), at most 128×128, with linear filtering.
//! * **Tests never reach real media sessions:** the hub comes from [`HubFactory`], which is
//!   `MediaHub::start` in the game and a fake under `cfg(test)`; the tests install their own
//!   fake and read its commands back.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mc_assets::TextureMetadata;
use mc_media::{AlbumArt, MediaCommand, MediaHub, NowPlaying, SourceFilter};
use mc_mods::mods::spotify::ids;
use mc_mods::ModSettings;
use mc_render::input::Input;
use mc_render::{Gpu, InputState};
use mc_ui::modhud::elements::spotify::{
    cover_texture, AeroText, SpotifyControl, SpotifyHud, SpotifyLayout, SpotifyOptions,
    SpotifyStyle, MOD_ID,
};
use mc_ui::modhud::ModHud;
use mc_ui::screens::modmenu::text::{self as menu_text, MenuText};
use mc_ui::widget::{UiRenderer, UiTexture};

use super::super::{Client, Ui};

/// How long a clicked play/pause state shows before the player has confirmed it.
const OPTIMISTIC: Duration = Duration::from_millis(1500);
/// After a volume drag, the dragged value shows this long while the player catches up.
const VOLUME_HOLD: Duration = Duration::from_millis(1000);
/// At most one `SetVolume` per this long while dragging (the worker rate-limits too).
const VOLUME_SEND_INTERVAL: Duration = Duration::from_millis(40);
/// The largest cover side uploaded.
const COVER_MAX: u32 = 128;

/// Makes the media hub when the mod turns on.
pub(crate) struct HubFactory(pub(crate) Box<dyn FnMut(SourceFilter) -> MediaHub>);

impl Default for HubFactory {
    fn default() -> Self {
        // A test that forgets to install its own fake still never reaches the user's real
        // sessions: under `cfg(test)` the default hub is a fake that nobody feeds.
        #[cfg(test)]
        {
            HubFactory(Box::new(|_| MediaHub::fake().0))
        }
        #[cfg(not(test))]
        {
            HubFactory(Box::new(MediaHub::start))
        }
    }
}

impl std::fmt::Debug for HubFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HubFactory(..)")
    }
}

/// The cover the widget shows, and what the UI renderer holds.
#[derive(Debug, Default)]
struct CoverState {
    /// The current track's art (kept to re-upload after a renderer rebuild).
    art: Option<Arc<AlbumArt>>,
    /// `art_generation` of `art`.
    generation: Option<u64>,
    /// The generation the UI renderer holds, if any.
    uploaded: Option<u64>,
}

/// A volume slider drag in progress.
#[derive(Debug, Clone, Copy)]
struct VolumeDrag {
    value: f32,
    sent: f32,
    sent_at: Instant,
}

/// The state on the client (`Client::w9`).
#[derive(Debug)]
pub(crate) struct W9State {
    /// The settings generation the snapshot below belongs to (0 = never synced).
    seen: u64,
    pub(crate) enabled: bool,
    options: SpotifyOptions,
    filter: SourceFilter,
    /// Play/pause, next, previous.
    keys: [Input; 3],
    keys_down: [bool; 3],
    /// The first key sample after enabling only records what is held (a held key is no press).
    keys_armed: bool,
    pub(crate) hub: Option<MediaHub>,
    /// The filter the running hub was given.
    hub_filter: SourceFilter,
    pub(crate) factory: HubFactory,
    /// The hub generation `now_playing` was read at.
    media_generation: u64,
    now_playing: Option<Arc<NowPlaying>>,
    /// The hub generation the widget's strings were last compared at.
    filled_generation: u64,
    text_revision: u64,
    cover: CoverState,
    /// The play/pause state just asked for, and when.
    optimistic: Option<(bool, Instant)>,
    drag: Option<VolumeDrag>,
    /// The value of the last drag, and when it ended.
    volume_hold: Option<(f32, Instant)>,
    /// The control held down by the mouse.
    pressed: Option<SpotifyControl>,
    /// Chat is open over a shown widget.
    interactive: bool,
    hover: Option<SpotifyControl>,
    /// The widget's index in `ModHud::slots()`.
    slot: Option<usize>,
    /// Segoe UI for the Aero style, while it is selected.
    aero_text: Option<AeroText>,
    atlas_key: Option<UiTexture>,
    atlas_uploaded: bool,
    /// `HudData::spotify` holds something to clear when the mod turns off.
    hud_filled: bool,
}

impl Default for W9State {
    fn default() -> Self {
        W9State {
            seen: 0,
            enabled: false,
            options: SpotifyOptions::default(),
            filter: SourceFilter::Spotify,
            keys: [Input::Unbound; 3],
            keys_down: [false; 3],
            keys_armed: false,
            hub: None,
            hub_filter: SourceFilter::Spotify,
            factory: HubFactory::default(),
            media_generation: 0,
            now_playing: None,
            filled_generation: 0,
            text_revision: 0,
            cover: CoverState::default(),
            optimistic: None,
            drag: None,
            volume_hold: None,
            pressed: None,
            interactive: false,
            hover: None,
            slot: None,
            aero_text: None,
            atlas_key: None,
            atlas_uploaded: false,
            hud_filled: false,
        }
    }
}

/// The commands the three keys send, in `W9State::keys` order.
const KEY_COMMANDS: [MediaCommand; 3] = [
    MediaCommand::PlayPause,
    MediaCommand::Next,
    MediaCommand::Previous,
];

/// Linear, clamped sampling: covers and glyphs are minified and must not wrap.
fn linear() -> TextureMetadata {
    TextureMetadata {
        blur: true,
        clamp: true,
        ..TextureMetadata::default()
    }
}

/// Usable art: a non-empty image whose pixel buffer matches its size.
fn art_is_valid(art: &AlbumArt) -> bool {
    art.width > 0
        && art.height > 0
        && art.rgba.len() == (art.width as usize) * (art.height as usize) * 4
}

/// The art at most [`COVER_MAX`] on a side (W8 already shrinks it; this is the backstop): a box
/// filter by the smallest whole factor that fits.
fn fit_cover(art: &AlbumArt) -> Option<(Cow<'_, [u8]>, u32, u32)> {
    if !art_is_valid(art) {
        return None;
    }
    let (w, h) = (art.width, art.height);
    if w <= COVER_MAX && h <= COVER_MAX {
        return Some((Cow::Borrowed(&art.rgba), w, h));
    }
    let k = w.max(h).div_ceil(COVER_MAX);
    let (ow, oh) = ((w / k).max(1), (h / k).max(1));
    let mut out = Vec::with_capacity((ow * oh * 4) as usize);
    for oy in 0..oh {
        for ox in 0..ow {
            let mut sum = [0u32; 4];
            for y in oy * k..(oy * k + k).min(h) {
                for x in ox * k..(ox * k + k).min(w) {
                    let i = ((y * w + x) * 4) as usize;
                    for (c, s) in sum.iter_mut().enumerate() {
                        *s += u32::from(art.rgba[i + c]);
                    }
                }
            }
            let n = k * k;
            out.extend(sum.map(|s| (s / n) as u8));
        }
    }
    Some((Cow::Owned(out), ow, oh))
}

impl W9State {
    /// Refresh the option snapshot when the settings generation moved.
    fn sync(&mut self, settings: &ModSettings) {
        if self.seen == settings.generation() {
            return;
        }
        self.seen = settings.generation();
        self.enabled = settings.enabled(MOD_ID);
        self.options = SpotifyOptions::read(settings);
        self.filter = match settings.choice(MOD_ID, ids::SOURCE) {
            1 => SourceFilter::Any,
            _ => SourceFilter::Spotify,
        };
        self.keys = [ids::KEY_PLAY_PAUSE, ids::KEY_NEXT, ids::KEY_PREVIOUS]
            .map(|id| Input::from_lwjgl(settings.key(MOD_ID, id)));
    }

    /// Start the hub when the mod is on, drop it when it is off, pass on a new source filter.
    fn drive_hub(&mut self) {
        if !self.enabled {
            if self.hub.is_some() {
                self.stop();
            }
            return;
        }
        match &self.hub {
            None => {
                self.hub = Some((self.factory.0)(self.filter));
                self.hub_filter = self.filter;
                self.media_generation = 0;
            }
            Some(hub) if self.hub_filter != self.filter => {
                hub.set_filter(self.filter);
                self.hub_filter = self.filter;
            }
            Some(_) => {}
        }
    }

    /// Drop the hub (its worker stops) and everything read from it. The cover's `uploaded`
    /// stays, so the upload hook frees the texture.
    fn stop(&mut self) {
        self.hub = None;
        self.media_generation = 0;
        self.now_playing = None;
        self.filled_generation = 0;
        self.cover.art = None;
        self.cover.generation = None;
        self.optimistic = None;
        self.drag = None;
        self.volume_hold = None;
        self.pressed = None;
        self.interactive = false;
        self.hover = None;
        self.keys_armed = false;
    }

    fn send(&self, command: MediaCommand) {
        if let Some(hub) = &self.hub {
            hub.send(command);
        }
    }

    /// Read the hub's state if it changed: one atomic load on a frame where it did not.
    fn poll(&mut self) {
        let Some(hub) = &self.hub else {
            return;
        };
        let generation = hub.generation();
        if generation == self.media_generation {
            return;
        }
        self.media_generation = generation;
        self.now_playing = hub.state().now_playing;
        let art = self.now_playing.as_ref().and_then(|now| {
            now.art
                .as_ref()
                .filter(|art| art_is_valid(art))
                .map(|art| (Arc::clone(art), now.art_generation))
        });
        match art {
            Some((art, generation)) => {
                if self.cover.generation != Some(generation) {
                    self.cover.art = Some(art);
                    self.cover.generation = Some(generation);
                }
            }
            None => {
                self.cover.art = None;
                self.cover.generation = None;
            }
        }
        if let Some(now) = self.now_playing.as_deref() {
            if self.optimistic.is_some_and(|(want, _)| want == now.playing) {
                self.optimistic = None;
            }
            if let (Some((value, _)), Some(volume)) = (self.volume_hold, now.volume) {
                if (volume - value).abs() < 0.01 {
                    self.volume_hold = None;
                }
            }
        }
    }

    /// Playing as shown: the state just clicked until the player confirms it (or a moment
    /// passes), else the player's.
    fn displayed_playing(&self, now: Instant) -> bool {
        let playing = self.now_playing.as_ref().is_some_and(|p| p.playing);
        match self.optimistic {
            Some((want, since)) if now.saturating_duration_since(since) < OPTIMISTIC => want,
            _ => playing,
        }
    }

    /// The volume as shown: the dragged value, then the released one for a moment, else the
    /// player's.
    fn displayed_volume(&self, now: Instant) -> Option<f32> {
        if let Some(drag) = self.drag {
            return Some(drag.value);
        }
        let player = self.now_playing.as_ref().and_then(|p| p.volume);
        match self.volume_hold {
            Some((value, since))
                if player.is_some() && now.saturating_duration_since(since) < VOLUME_HOLD =>
            {
                Some(value)
            }
            _ => player,
        }
    }

    fn accepts(&self, control: SpotifyControl, now: Instant) -> bool {
        let Some(playing) = self.now_playing.as_deref() else {
            return false;
        };
        match control {
            SpotifyControl::Previous => playing.can_previous,
            SpotifyControl::PlayPause => playing.can_play_pause,
            SpotifyControl::Next => playing.can_next,
            SpotifyControl::Volume => self.displayed_volume(now).is_some(),
        }
    }

    fn ensure_slot(&mut self, hud: &ModHud) {
        if self.slot.is_none() {
            let catalog = mc_mods::catalog::all();
            self.slot = hud
                .slots()
                .iter()
                .position(|slot| catalog[slot.mod_index].id == MOD_ID);
        }
    }

    /// The layout the widget was drawn with last frame, and the GUI point `cursor` in its
    /// local pixels. `None` when it was not drawn (no track, F3, the mod off).
    fn locate(
        &self,
        cursor: (i32, i32),
        hud: &ModHud,
        now: Instant,
    ) -> Option<(SpotifyLayout, (f32, f32))> {
        let slot = hud.slots().get(self.slot?)?;
        let rect = slot.rect?;
        if !slot.enabled || slot.placeholder {
            return None;
        }
        let scale = if slot.placement.scale > 0.0 {
            slot.placement.scale
        } else {
            1.0
        };
        let local = (
            (cursor.0 as f32 - rect.x) / scale,
            (cursor.1 as f32 - rect.y) / scale,
        );
        // Whether the control row was drawn is in the drawn height: the row may follow the
        // free cursor, and the click lands on what the user saw.
        let without = SpotifyLayout::new(&self.options, false, false).size.1;
        let controls = rect.h / scale > without + 0.5;
        let volume = self.displayed_volume(now).is_some();
        Some((SpotifyLayout::new(&self.options, controls, volume), local))
    }

    /// Hover and the volume drag, once per frame while the mod is on.
    fn update_pointer(&mut self, chat_open: bool, cursor: (i32, i32), hud: &ModHud, now: Instant) {
        self.ensure_slot(hud);
        self.interactive = chat_open && self.now_playing.is_some();
        if !self.interactive {
            self.hover = None;
            self.pressed = None;
            if self.drag.is_some() {
                self.finish_drag(now);
            }
            return;
        }
        let located = self.locate(cursor, hud, now);
        self.hover = located
            .and_then(|(layout, (x, y))| layout.hit(x, y))
            .filter(|&control| self.accepts(control, now));
        if self.drag.is_some() {
            let value = located.and_then(|(layout, (x, _))| layout.volume_at(x));
            let mut send = None;
            if let Some(drag) = self.drag.as_mut() {
                if let Some(value) = value {
                    drag.value = value;
                }
                if (drag.value - drag.sent).abs() >= 0.005
                    && now.saturating_duration_since(drag.sent_at) >= VOLUME_SEND_INTERVAL
                {
                    drag.sent = drag.value;
                    drag.sent_at = now;
                    send = Some(drag.value);
                }
            }
            if let Some(value) = send {
                self.send(MediaCommand::SetVolume(value));
            }
        }
    }

    /// A left press at `cursor` while chat is open. `Some` when it landed on a control (the
    /// press is consumed even on a disabled one); a command goes out only for an enabled one.
    fn press(&mut self, cursor: (i32, i32), hud: &ModHud, now: Instant) -> Option<SpotifyControl> {
        self.now_playing.as_ref()?;
        let (layout, (x, y)) = self.locate(cursor, hud, now)?;
        let control = layout.hit(x, y)?;
        if !self.accepts(control, now) {
            return Some(control);
        }
        self.pressed = Some(control);
        match control {
            SpotifyControl::PlayPause => {
                let playing = self.displayed_playing(now);
                self.optimistic = Some((!playing, now));
                self.send(MediaCommand::PlayPause);
            }
            SpotifyControl::Next => self.send(MediaCommand::Next),
            SpotifyControl::Previous => self.send(MediaCommand::Previous),
            SpotifyControl::Volume => {
                let value = layout.volume_at(x).unwrap_or(0.0);
                self.drag = Some(VolumeDrag {
                    value,
                    sent: value,
                    sent_at: now,
                });
                self.volume_hold = None;
                self.send(MediaCommand::SetVolume(value));
            }
        }
        Some(control)
    }

    /// The left button came up: end a press or a drag. `true` when one was in progress.
    fn release(&mut self, now: Instant) -> bool {
        let pressed = self.pressed.take();
        if self.drag.is_some() {
            self.finish_drag(now);
        }
        pressed.is_some()
    }

    fn finish_drag(&mut self, now: Instant) {
        if let Some(drag) = self.drag.take() {
            if drag.value != drag.sent {
                self.send(MediaCommand::SetVolume(drag.value));
            }
            self.volume_hold = Some((drag.value, now));
        }
    }

    /// The three optional keys, as press edges, in game only.
    fn keys_frame(&mut self, is_down: impl Fn(Input) -> bool, in_game: bool, now: Instant) {
        if self.keys.iter().all(|&key| key == Input::Unbound) {
            return;
        }
        let armed = std::mem::replace(&mut self.keys_armed, true);
        for (i, command) in KEY_COMMANDS.into_iter().enumerate() {
            let key = self.keys[i];
            let down = key != Input::Unbound && is_down(key);
            let pressed = armed && down && !self.keys_down[i];
            self.keys_down[i] = down;
            if pressed && in_game && self.hub.is_some() {
                if command == MediaCommand::PlayPause && self.now_playing.is_some() {
                    let playing = self.displayed_playing(now);
                    self.optimistic = Some((!playing, now));
                }
                self.send(command);
            }
        }
    }

    /// This frame's widget data. Strings are copied only when the hub
    /// published something new and they differ.
    pub(crate) fn fill_hud(&mut self, out: &mut SpotifyHud) {
        if !self.enabled {
            if self.hud_filled {
                *out = SpotifyHud::default();
                self.hud_filled = false;
            }
            return;
        }
        self.hud_filled = true;
        let now = Instant::now();
        let aero = self.options.style == SpotifyStyle::Aero;
        if aero != out.aero_text.is_some() {
            out.aero_text = aero.then(|| {
                self.aero_text
                    .get_or_insert_with(|| AeroText::new(MenuText::load()))
                    .clone()
            });
        }
        out.interactive = self.interactive;
        out.hover = self.hover;
        out.pressed = self.pressed;
        let Some(playing) = self.now_playing.as_deref() else {
            out.present = false;
            return;
        };
        out.present = true;
        if self.filled_generation != self.media_generation {
            self.filled_generation = self.media_generation;
            if out.title != playing.title || out.artist != playing.artist {
                out.title.clear();
                out.title.push_str(&playing.title);
                out.artist.clear();
                out.artist.push_str(&playing.artist);
                self.text_revision = self.text_revision.wrapping_add(1);
                out.text_revision = self.text_revision;
            }
            out.duration_ms = playing.duration_ms;
            out.can_play_pause = playing.can_play_pause;
            out.can_next = playing.can_next;
            out.can_previous = playing.can_previous;
        }
        out.playing = self.displayed_playing(now);
        // A pause just clicked freezes the bar where it was clicked.
        let at = match self.optimistic {
            Some((false, since)) if playing.playing && !out.playing => since,
            _ => now,
        };
        out.position_ms = playing.position_at(at);
        out.volume = self.displayed_volume(now);
        out.cover = self.cover.art.is_some();
    }

    /// In `build_ui` before the frame's textures are resolved: the cover
    /// when its generation changed (or the renderer lost it), the Segoe atlas when it gained
    /// glyphs. With the mod off, frees what an earlier session uploaded, once.
    pub(crate) fn upload_textures(&mut self, gpu: &Gpu, renderer: &mut UiRenderer) {
        if !self.enabled {
            if self.cover.uploaded.is_some() || self.atlas_uploaded || self.aero_text.is_some() {
                if self.cover.uploaded.take().is_some() {
                    renderer.invalidate(cover_texture());
                }
                if std::mem::take(&mut self.atlas_uploaded) {
                    if let Some(key) = &self.atlas_key {
                        renderer.invalidate(key);
                    }
                }
                self.aero_text = None;
            }
            return;
        }
        match &self.cover.art {
            Some(art) => {
                if self.cover.uploaded != self.cover.generation
                    || !renderer.has_texture(cover_texture())
                {
                    if let Some((rgba, w, h)) = fit_cover(art) {
                        renderer.upload_with_options(
                            gpu,
                            cover_texture().clone(),
                            &rgba,
                            w,
                            h,
                            &linear(),
                        );
                        self.cover.uploaded = self.cover.generation;
                    }
                }
            }
            None => {
                if self.cover.uploaded.take().is_some() {
                    renderer.invalidate(cover_texture());
                }
            }
        }
        if self.options.style != SpotifyStyle::Aero {
            // Back to the Minecraft style: the Segoe atlas (4 MB here and on the GPU) goes.
            if std::mem::take(&mut self.atlas_uploaded) {
                if let Some(key) = &self.atlas_key {
                    renderer.invalidate(key);
                }
            }
            self.aero_text = None;
            return;
        }
        if let Some(text) = &self.aero_text {
            let key = self.atlas_key.get_or_insert_with(menu_text::atlas_key);
            let mut text = text.0.borrow_mut();
            if self.atlas_uploaded && !renderer.has_texture(key) {
                // A new renderer (a resource reload): send the whole atlas again.
                text.mark_dirty();
            }
            if let Some(pixels) = text.take_upload() {
                renderer.upload_with_options(
                    gpu,
                    key.clone(),
                    pixels,
                    menu_text::ATLAS_SIZE,
                    menu_text::ATLAS_SIZE,
                    &linear(),
                );
                self.atlas_uploaded = true;
            }
        }
    }
}

impl Client {
    /// At the end of `Client::frame`: sync, the hub's lifecycle, its state,
    /// the hover and the volume drag.
    pub(crate) fn w9_frame(&mut self) {
        self.w9.sync(&self.mods.settings);
        if !self.w9.enabled && self.w9.hub.is_none() {
            return;
        }
        self.w9.drive_hub();
        if !self.w9.enabled {
            // Just turned off: release the strings and the Segoe renderer the HUD data holds.
            self.mods.hud_data.spotify = SpotifyHud::default();
            return;
        }
        self.w9.poll();
        let chat_open = matches!(self.ui, Ui::Chat(_));
        self.w9
            .update_pointer(chat_open, self.cursor, &self.mods.hud, Instant::now());
    }

    /// The optional keys, in game, not with F3 held (F3 combos).
    pub(crate) fn w9_shell_frame(&mut self, input: &InputState) {
        self.w9.sync(&self.mods.settings);
        if !self.w9.enabled {
            return;
        }
        let in_game = matches!(self.ui, Ui::InGame) && !self.debug_f3_held;
        self.w9
            .keys_frame(|key| input.is_down(key), in_game, Instant::now());
    }

    /// First in `mouse_clicked`: a left press on one of the widget's controls
    /// while chat is open. Returns `true` when it consumed the press.
    pub(crate) fn w9_mouse_pressed(&mut self, button: i32) -> bool {
        if !self.w9.enabled || button != 0 || !matches!(self.ui, Ui::Chat(_)) {
            return false;
        }
        let Some(control) = self.w9.press(self.cursor, &self.mods.hud, Instant::now()) else {
            return false;
        };
        // The Minecraft style's buttons are vanilla buttons: they click like them. The Aero
        // style follows the mod menu, which is silent.
        if self.w9.options.style == SpotifyStyle::Minecraft && self.w9.pressed == Some(control) {
            self.play_button_press();
        }
        true
    }

    /// First in `mouse_released`: ends a press or a volume drag. Returns
    /// `true` when one was in progress.
    pub(crate) fn w9_mouse_released(&mut self, button: i32) -> bool {
        if !self.w9.enabled || button != 0 {
            return false;
        }
        self.w9.release(Instant::now())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use mc_media::FakeControl;
    use mc_ui::modhud::HudCtx;
    use mc_ui::screen_stack::ScreenCtx;
    use mc_ui::widget::DrawList;

    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::session::{JoinSettings, Session};

    /// The fakes the hub factory made, newest last.
    type Fakes = Rc<RefCell<Vec<FakeControl>>>;

    fn fake_factory() -> (HubFactory, Fakes, Rc<RefCell<Vec<SourceFilter>>>) {
        let fakes: Fakes = Rc::default();
        let filters = Rc::new(RefCell::new(Vec::new()));
        let (f, l) = (Rc::clone(&fakes), Rc::clone(&filters));
        let factory = HubFactory(Box::new(move |filter| {
            let (hub, control) = MediaHub::fake();
            hub.set_filter(filter);
            l.borrow_mut().push(filter);
            f.borrow_mut().push(control);
            hub
        }));
        (factory, fakes, filters)
    }

    fn art(generation_colour: u8) -> Arc<AlbumArt> {
        Arc::new(AlbumArt {
            width: 4,
            height: 4,
            rgba: vec![generation_colour; 64],
        })
    }

    fn track(title: &str, playing: bool) -> NowPlaying {
        NowPlaying {
            source_app: "Spotify.exe".to_owned(),
            title: title.to_owned(),
            artist: "Artist".to_owned(),
            album: String::new(),
            playing,
            position_ms: Some(30_000),
            duration_ms: Some(200_000),
            sampled_at: Instant::now(),
            volume: Some(0.5),
            art: Some(art(200)),
            art_generation: 1,
            can_play_pause: true,
            can_next: true,
            can_previous: true,
        }
    }

    // ---- pure state tests (no client) ----------------------------------------------------

    fn enabled_state() -> (W9State, Fakes, ModSettings) {
        let (factory, fakes, _) = fake_factory();
        let mut state = W9State {
            factory,
            ..W9State::default()
        };
        let mut settings = ModSettings::new();
        settings.set_enabled(MOD_ID, true);
        state.sync(&settings);
        state.drive_hub();
        (state, fakes, settings)
    }

    #[test]
    fn the_hub_exists_only_while_the_mod_is_on() {
        let (factory, fakes, filters) = fake_factory();
        let mut state = W9State {
            factory,
            ..W9State::default()
        };
        let mut settings = ModSettings::new();
        state.sync(&settings);
        state.drive_hub();
        assert!(state.hub.is_none(), "off by default: no hub");
        assert!(fakes.borrow().is_empty());

        settings.set_enabled(MOD_ID, true);
        state.sync(&settings);
        state.drive_hub();
        state.drive_hub();
        assert!(state.hub.is_some());
        assert_eq!(fakes.borrow().len(), 1, "started once");
        assert_eq!(*filters.borrow(), vec![SourceFilter::Spotify]);

        // The source option reaches the running hub without a restart.
        settings.set_choice(MOD_ID, ids::SOURCE, 1);
        state.sync(&settings);
        state.drive_hub();
        assert_eq!(fakes.borrow().len(), 1);
        assert_eq!(fakes.borrow()[0].filter(), SourceFilter::Any);

        fakes.borrow()[0].set(Some(track("A", true)));
        state.poll();
        assert!(state.now_playing.is_some());
        settings.set_enabled(MOD_ID, false);
        state.sync(&settings);
        state.drive_hub();
        assert!(state.hub.is_none(), "turned off: the hub is dropped");
        assert!(state.now_playing.is_none() && state.cover.art.is_none());
        // No hub, no command.
        state.send(MediaCommand::Next);
        assert!(fakes.borrow()[0].take_commands().is_empty());
    }

    #[test]
    fn the_widget_data_follows_the_hub_and_strings_change_only_with_the_track() {
        let (mut state, fakes, _) = enabled_state();
        let mut out = SpotifyHud::default();
        state.poll();
        state.fill_hud(&mut out);
        assert!(!out.present, "no session yet");

        fakes.borrow()[0].set(Some(track("First", true)));
        state.poll();
        state.fill_hud(&mut out);
        assert!(out.present && out.playing && out.cover);
        assert_eq!(
            (out.title.as_str(), out.artist.as_str()),
            ("First", "Artist")
        );
        assert_eq!(out.duration_ms, Some(200_000));
        assert!(out.position_ms.unwrap() >= 30_000);
        assert_eq!(out.volume, Some(0.5));
        let revision = out.text_revision;
        let title_ptr = out.title.as_ptr();

        // A position-only update: no string work.
        fakes.borrow()[0].set(Some(NowPlaying {
            position_ms: Some(31_000),
            ..track("First", true)
        }));
        state.poll();
        state.fill_hud(&mut out);
        assert_eq!(out.text_revision, revision);
        assert_eq!(out.title.as_ptr(), title_ptr);

        fakes.borrow()[0].set(Some(track("Second", false)));
        state.poll();
        state.fill_hud(&mut out);
        assert_eq!(out.title, "Second");
        assert!(out.text_revision != revision);
        assert!(!out.playing);

        // The session goes away: the widget hides.
        fakes.borrow()[0].set(None);
        state.poll();
        state.fill_hud(&mut out);
        assert!(!out.present && state.cover.art.is_none());
    }

    #[test]
    fn the_cover_is_uploaded_only_when_its_generation_changes() {
        let (mut state, fakes, _) = enabled_state();
        fakes.borrow()[0].set(Some(track("A", true)));
        state.poll();
        assert_eq!(state.cover.generation, Some(1));
        let first = Arc::as_ptr(state.cover.art.as_ref().unwrap());
        // A new snapshot with the same generation keeps the art the widget already has.
        fakes.borrow()[0].set(Some(NowPlaying {
            art: Some(art(10)),
            ..track("A", true)
        }));
        state.poll();
        assert_eq!(Arc::as_ptr(state.cover.art.as_ref().unwrap()), first);
        fakes.borrow()[0].set(Some(NowPlaying {
            art: Some(art(10)),
            art_generation: 2,
            ..track("B", true)
        }));
        state.poll();
        assert_eq!(state.cover.generation, Some(2));
        assert_ne!(Arc::as_ptr(state.cover.art.as_ref().unwrap()), first);
        // Broken art is no art.
        fakes.borrow()[0].set(Some(NowPlaying {
            art: Some(Arc::new(AlbumArt {
                width: 8,
                height: 8,
                rgba: vec![0; 3],
            })),
            art_generation: 3,
            ..track("B", true)
        }));
        state.poll();
        assert!(state.cover.art.is_none());
    }

    #[test]
    fn oversized_art_is_box_filtered_to_128() {
        let big = AlbumArt {
            width: 300,
            height: 300,
            rgba: vec![100; 300 * 300 * 4],
        };
        let (rgba, w, h) = fit_cover(&big).unwrap();
        assert_eq!((w, h), (100, 100));
        assert!(w <= COVER_MAX && h <= COVER_MAX);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
        assert!(rgba.iter().all(|&b| b == 100));
        let small = art(7);
        assert!(matches!(fit_cover(&small), Some((Cow::Borrowed(_), 4, 4))));
    }

    #[test]
    fn keys_send_on_the_press_edge_in_game_only() {
        let (mut state, fakes, mut settings) = enabled_state();
        settings.set_key(MOD_ID, ids::KEY_NEXT, 49); // N
        state.sync(&settings);
        fakes.borrow()[0].set(Some(track("A", true)));
        state.poll();
        let n = Input::from_lwjgl(49);
        let now = Instant::now();
        // Held when sampling starts: not a press.
        state.keys_frame(|k| k == n, true, now);
        state.keys_frame(|_| false, true, now);
        assert!(fakes.borrow()[0].take_commands().is_empty());
        state.keys_frame(|k| k == n, true, now);
        state.keys_frame(|k| k == n, true, now);
        assert_eq!(
            fakes.borrow()[0].take_commands(),
            vec![MediaCommand::Next],
            "one press, no repeat while held"
        );
        // Typing in chat (not in game): nothing.
        state.keys_frame(|_| false, false, now);
        state.keys_frame(|k| k == n, false, now);
        assert!(fakes.borrow()[0].take_commands().is_empty());
    }

    // ---- through the real client: the HUD pipeline, chat and the mouse path ---------------

    struct Harness {
        client: Client,
        fakes: Fakes,
        _dir: tempfile::TempDir,
        _net: (
            tokio::sync::mpsc::Sender<crate::net::ToGame>,
            tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
        ),
    }

    const SIZE: (u32, u32) = (1920, 1080);

    /// An offline client in an (empty) world, the fake hub factory installed. Skips without
    /// the vanilla jar.
    fn harness() -> Option<Harness> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(Session::new(handle, JoinSettings::default()));
        client.ui = Ui::InGame;
        let (factory, fakes, _) = fake_factory();
        client.w9.factory = factory;
        Some(Harness {
            client,
            fakes,
            _dir: dir,
            _net: (to_game, to_net),
        })
    }

    /// One frame of the widget's path: the frame hook, then the mod HUD draw (which fills the
    /// HUD data and records the widget's on-screen rect).
    fn frame(client: &mut Client) -> DrawList {
        client.w9_frame();
        let resolution = client.resolution(SIZE.0, SIZE.1);
        let ctx = ScreenCtx::new(resolution, 0.0);
        let mut list = DrawList::new();
        client.draw_mod_hud(&mut list, &ctx);
        list
    }

    /// The GUI pixel at the centre of `control`, from the drawn rect and the layout.
    fn centre_of(client: &Client, control: SpotifyControl) -> (i32, i32) {
        let slot = &client.mods.hud.slots()[client.w9.slot.expect("slot found")];
        let rect = slot.rect.expect("the widget was drawn");
        let scale = slot.placement.scale;
        let layout = client
            .w9
            .locate((0, 0), &client.mods.hud, Instant::now())
            .expect("located")
            .0;
        let r = match control {
            SpotifyControl::Previous => layout.buttons.unwrap()[0],
            SpotifyControl::PlayPause => layout.buttons.unwrap()[1],
            SpotifyControl::Next => layout.buttons.unwrap()[2],
            SpotifyControl::Volume => layout.volume.unwrap(),
        };
        let (cx, cy) = r.center();
        (
            (rect.x + cx * scale).floor() as i32,
            (rect.y + cy * scale).floor() as i32,
        )
    }

    fn click(client: &mut Client, at: (i32, i32)) {
        client.cursor = at;
        let ctx = ScreenCtx::new(client.resolution(SIZE.0, SIZE.1), 0.0);
        client.mouse_clicked(0, ctx);
        client.mouse_released(0);
    }

    #[test]
    fn with_the_mod_off_there_is_no_hub_no_data_and_no_click() {
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        assert!(!client.mods.settings.enabled(MOD_ID), "off by default");
        let list = frame(client);
        assert!(client.w9.hub.is_none());
        assert!(h.fakes.borrow().is_empty(), "the factory never ran");
        assert!(!client.mods.hud_data.spotify.present);
        assert!(list.is_empty(), "nothing drawn");
        client.open_chat("");
        assert!(!client.w9_mouse_pressed(0));
        assert!(!client.w9_mouse_released(0));
    }

    #[test]
    fn clicks_while_chat_is_open_reach_the_hub_and_nothing_else() {
        let Some(mut h) = harness() else { return };
        let fakes = Rc::clone(&h.fakes);
        let client = &mut h.client;
        client.mods.settings.set_enabled(MOD_ID, true);
        frame(client);
        assert_eq!(
            fakes.borrow().len(),
            1,
            "turning the mod on started the hub"
        );
        fakes.borrow()[0].set(Some(track("Song", true)));
        let list = frame(client);
        assert!(client.mods.hud_data.spotify.present);
        assert!(!list.is_empty(), "the widget is drawn");
        assert!(
            list.batches().iter().any(|b| &b.texture == cover_texture()),
            "the cover is referenced (uploaded in build_ui before the draw)"
        );

        // In game the cursor is captured: a click (if one got here) is not the widget's.
        let play = centre_of(client, SpotifyControl::PlayPause);
        click(client, play);
        assert!(fakes.borrow()[0].take_commands().is_empty());

        client.open_chat("");
        frame(client);
        assert!(client.mods.hud_data.spotify.interactive);
        let play = centre_of(client, SpotifyControl::PlayPause);
        click(client, play);
        let next = centre_of(client, SpotifyControl::Next);
        click(client, next);
        let previous = centre_of(client, SpotifyControl::Previous);
        click(client, previous);
        assert_eq!(
            fakes.borrow()[0].take_commands(),
            vec![
                MediaCommand::PlayPause,
                MediaCommand::Next,
                MediaCommand::Previous
            ]
        );
        // The pause shows at once, before the player confirms it.
        frame(client);
        assert!(!client.mods.hud_data.spotify.playing);
        assert!(matches!(client.ui, Ui::Chat(_)), "chat stays open");

        // A click beside the widget is chat's, not ours.
        click(client, (2, 2));
        assert!(fakes.borrow()[0].take_commands().is_empty());

        // Hovering the next button lights it.
        client.cursor = centre_of(client, SpotifyControl::Next);
        frame(client);
        assert_eq!(
            client.mods.hud_data.spotify.hover,
            Some(SpotifyControl::Next)
        );
    }

    #[test]
    fn the_volume_slider_drags_and_rate_limits() {
        let Some(mut h) = harness() else { return };
        let fakes = Rc::clone(&h.fakes);
        let client = &mut h.client;
        client.mods.settings.set_enabled(MOD_ID, true);
        frame(client);
        fakes.borrow()[0].set(Some(track("Song", true)));
        frame(client);
        client.open_chat("");
        frame(client);
        let slot = &client.mods.hud.slots()[client.w9.slot.unwrap()];
        let (rect, scale) = (slot.rect.unwrap(), slot.placement.scale);
        let layout = client
            .w9
            .locate((0, 0), &client.mods.hud, Instant::now())
            .unwrap()
            .0;
        let track_rect = layout.volume.expect("the volume is known");
        let at = |t: f32| {
            (
                (rect.x + (track_rect.x + track_rect.w * t) * scale).round() as i32,
                (rect.y + (track_rect.y + 1.0) * scale).floor() as i32,
            )
        };
        // Press at a quarter: SetVolume right away.
        client.cursor = at(0.25);
        let ctx = ScreenCtx::new(client.resolution(SIZE.0, SIZE.1), 0.0);
        client.mouse_clicked(0, ctx);
        let first = fakes.borrow()[0].take_commands();
        assert_eq!(first.len(), 1);
        let MediaCommand::SetVolume(v) = first[0] else {
            panic!("{first:?}");
        };
        assert!((v - 0.25).abs() < 0.03, "{v}");
        // Drag to three quarters within the rate limit: shown at once, sent later.
        client.cursor = at(0.75);
        frame(client);
        let shown = client.mods.hud_data.spotify.volume.unwrap();
        assert!((shown - 0.75).abs() < 0.03, "{shown}");
        // Release: the final value goes out.
        client.mouse_released(0);
        let sent = fakes.borrow()[0].take_commands();
        let last = sent.last().copied();
        assert!(
            matches!(last, Some(MediaCommand::SetVolume(v)) if (v - 0.75).abs() < 0.03),
            "{sent:?}"
        );
        assert!(sent.len() <= 2, "{sent:?}");
        // The released value shows while the player catches up.
        frame(client);
        let shown = client.mods.hud_data.spotify.volume.unwrap();
        assert!((shown - 0.75).abs() < 0.03, "{shown}");
    }

    #[test]
    fn closing_chat_ends_interaction_and_turning_the_mod_off_drops_everything() {
        let Some(mut h) = harness() else { return };
        let fakes = Rc::clone(&h.fakes);
        let client = &mut h.client;
        client.mods.settings.set_enabled(MOD_ID, true);
        client.mods.settings.set_choice(MOD_ID, ids::STYLE, 1);
        frame(client);
        fakes.borrow()[0].set(Some(track("Song", true)));
        frame(client);
        assert!(
            client.mods.hud_data.spotify.aero_text.is_some(),
            "Aero: the client shares its Segoe UI renderer"
        );
        client.open_chat("");
        frame(client);
        assert!(client.mods.hud_data.spotify.interactive);
        client.show_screen(None);
        frame(client);
        assert!(!client.mods.hud_data.spotify.interactive);

        client.mods.settings.set_enabled(MOD_ID, false);
        frame(client);
        assert!(client.w9.hub.is_none(), "the hub is dropped");
        assert!(!client.mods.hud_data.spotify.present);
        assert!(client.mods.hud_data.spotify.aero_text.is_none());
        assert!(client.mods.hud_data.spotify.title.is_empty());
    }

    /// One `build_ui` with the UI renderer's upload counters reset first: whether the cover is
    /// resident afterwards, and how many texture bytes the frame uploaded.
    fn ui_frame(
        client: &mut Client,
        gpu: &mc_render::Gpu,
        ui: &super::super::Shared<mc_ui::widget::UiPhase>,
    ) -> (bool, u64) {
        client.w9_frame();
        ui.with(mc_ui::widget::UiPhase::begin_diagnostic_frame);
        client.build_ui(gpu, 0.0);
        ui.with(|phase| {
            (
                phase.renderer.has_texture(cover_texture()),
                phase
                    .diagnostic_stats()
                    .expect("diagnostics on")
                    .texture_upload_bytes,
            )
        })
    }

    /// The real render stack: the cover reaches the UI renderer in `build_ui`, once per art
    /// generation (64 bytes: the fake art is 4×4), and leaves it when the mod turns off.
    #[test]
    fn the_cover_is_uploaded_by_build_ui_once_per_generation() {
        let Some(mut h) = harness() else { return };
        let gpu = match mc_render::Gpu::headless(
            1280,
            720,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                return;
            }
            Err(error) => panic!("headless GPU: {error}"),
        };
        let fakes = Rc::clone(&h.fakes);
        let client = &mut h.client;
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, 1280, 720);
        let ui = client.render.as_ref().expect("installed").ui.clone();
        ui.with(|phase| phase.set_diagnostics_enabled(true));
        client.mods.settings.set_enabled(MOD_ID, true);
        client.w9_frame();
        fakes.borrow()[0].set(Some(track("Song", true)));

        let (resident, _) = ui_frame(client, &gpu, &ui);
        assert!(resident, "the first frame with art uploads it");
        // Steady state: nothing goes up again (the font sheets and widgets are resident too).
        for _ in 0..3 {
            let (resident, bytes) = ui_frame(client, &gpu, &ui);
            assert!(resident);
            assert_eq!(bytes, 0, "same generation: no upload");
        }
        // A position update keeps the generation: still nothing.
        fakes.borrow()[0].set(Some(NowPlaying {
            position_ms: Some(40_000),
            ..track("Song", true)
        }));
        assert_eq!(ui_frame(client, &gpu, &ui).1, 0);
        // New art: exactly one 4×4 upload.
        fakes.borrow()[0].set(Some(NowPlaying {
            art: Some(art(50)),
            art_generation: 2,
            ..track("Next song", true)
        }));
        assert_eq!(ui_frame(client, &gpu, &ui).1, 64);
        assert_eq!(ui_frame(client, &gpu, &ui).1, 0);

        client.mods.settings.set_enabled(MOD_ID, false);
        let (resident, bytes) = ui_frame(client, &gpu, &ui);
        assert!(!resident, "off: the cover texture is freed");
        assert_eq!(bytes, 0);
        assert!(client.w9.hub.is_none());
    }

    /// The real data path, **read-only**: W8's real hub on this PC's media session, the
    /// client's fill and cover upload, and both styles drawn through the client's own render
    /// stack. Nothing here can send a command (no click, no key reaches the widget). Saves
    /// `spotify_live_<style>.png` into `MC_W9_RENDER_DIR`. Run by hand:
    /// `cargo test -p mc-client --lib live_session -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads the real media session (read-only); run by hand"]
    fn live_session_renders_both_styles() {
        let Ok(out_dir) = std::env::var("MC_W9_RENDER_DIR") else {
            eprintln!("SKIP: set MC_W9_RENDER_DIR");
            return;
        };
        let Some(mut h) = harness() else { return };
        let (w, h_px) = (1280, 720);
        let mut gpu = match mc_render::Gpu::headless(
            w,
            h_px,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                return;
            }
            Err(error) => panic!("headless GPU: {error}"),
        };
        let client = &mut h.client;
        // The real worker. Read-only: this test never presses, clicks or binds anything.
        client.w9.factory = HubFactory(Box::new(MediaHub::start));
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, w, h_px);
        client.mods.settings.set_enabled(MOD_ID, true);
        let deadline = Instant::now() + Duration::from_secs(8);
        while client.w9.now_playing.is_none() && Instant::now() < deadline {
            client.w9_frame();
            std::thread::sleep(Duration::from_millis(50));
        }
        let Some(now) = client.w9.now_playing.clone() else {
            eprintln!("SKIP: no matching media session on this PC");
            return;
        };
        eprintln!(
            "live: {:?} by {:?}, playing {}, {:?}/{:?} ms, volume {:?}, art {:?} (gen {})",
            now.title,
            now.artist,
            now.playing,
            now.position_ms,
            now.duration_ms,
            now.volume,
            now.art.as_ref().map(|a| (a.width, a.height)),
            now.art_generation
        );
        let scale = client.resolution(w, h_px).scale_factor().max(1) as f32;
        for (style, name) in [(0, "minecraft"), (1, "aero")] {
            client.mods.settings.set_choice(MOD_ID, ids::STYLE, style);
            // The first frame lays out and uploads, the second draws with all of it resident.
            for _ in 0..2 {
                client.w9_frame();
                client.build_ui(&gpu, 0.0);
            }
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &mc_render::Camera::default(), 0.0);
            let pixels = gpu.read_pixels().expect("readback");
            let slot = &client.mods.hud.slots()[client.w9.slot.expect("slot")];
            let rect = slot.rect.expect("the widget was drawn");
            let frame = image::RgbaImage::from_raw(w, h_px, pixels).expect("frame size");
            let x = ((rect.x - 4.0) * scale).max(0.0) as u32;
            let y = ((rect.y - 4.0) * scale).max(0.0) as u32;
            let cw = (((rect.w + 8.0) * scale) as u32).min(w - x);
            let ch = (((rect.h + 8.0) * scale) as u32).min(h_px - y);
            let path = std::path::Path::new(&out_dir).join(format!("spotify_live_{name}.png"));
            image::imageops::crop_imm(&frame, x, y, cw, ch)
                .to_image()
                .save(&path)
                .expect("save");
            eprintln!("saved {}", path.display());
        }
    }

    #[test]
    fn the_hud_draws_through_the_real_pipeline_in_the_editor_sample() {
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        client.mods.settings.set_enabled(MOD_ID, true);
        client.mods.hud.sync(&client.mods.settings);
        let settings = &client.mods.settings;
        let data = mc_ui::modhud::HudData::default();
        let ctx = HudCtx {
            data: &data,
            settings,
            screen: (640.0, 360.0),
            pixel_scale: 3.0,
            editing: true,
        };
        let mut list = DrawList::new();
        client.mods.hud.draw(&ctx, &mut client.font, &mut list);
        client.w9.ensure_slot(&client.mods.hud);
        let slot = &client.mods.hud.slots()[client.w9.slot.expect("the widget has a slot")];
        let rect = slot.rect.expect("the editor shows the sample");
        assert!(!slot.placeholder, "a real sample, not the placeholder box");
        // Bottom right, 2 px in and 16 px up (clear of the chat input line).
        assert_eq!((rect.x + rect.w, rect.y + rect.h), (638.0, 344.0));
    }
}
