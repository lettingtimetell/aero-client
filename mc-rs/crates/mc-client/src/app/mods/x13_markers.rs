//! X13 Markers: the client half.
//!
//! What the player gets, all of it off until the mod is switched on in the Players window:
//!
//! * **Place.** The marker key (Z) marks what the crosshair is on: the first entity or block the
//!   look ray meets within [`MAX_DISTANCE`] blocks, else the point that far out along the ray.
//!   In game only (no screen open, so never while typing), never while F3 is held. A short ping
//!   plays that only this player hears.
//! * **An entity marker marks a spot.** It marks where the entity was hit when the key went
//!   down; it does not follow the entity. A marker that tracked a player through walls would be
//!   an ESP, which many servers ban. Invisible entities (hologram armour stands, the
//!   invisibility potion) and spectators are passed through: a marker never reveals what the
//!   player cannot see.
//! * **Removal.** Pressing the key on a marker (on its pin or distance as drawn, or within
//!   [`REPRESS_RADIUS`] of where the ray lands) removes it; the clear key removes them all. They
//!   also go on their own: after the set minutes, on a new world or dimension, on a disconnect,
//!   and on a teleport further than the set distance. One past the maximum replaces the oldest.
//!   Switching the mod off removes every marker. The rules are `mc_mods::mods::markers::set`.
//! * **Drawing.** A pin with the distance above it, through walls and at one size on screen
//!   (`mc_ui::marker_label`), in its own `NameplatePhase` at the waypoint labels' order (645:
//!   after the screen-space fog, before the hand), registered after them so markers draw on top.
//!   Hidden with F1, like every label.
//!
//! Nothing here sends a packet or writes a file.
//!
//! # Cost
//!
//! Off: the shell-frame hook is one `u64` compare and a `bool`. The tick, frame and event hooks
//! test `MarkerSet::is_empty` and return (switching the mod off empties the set). The phase is
//! registered at install as an empty wrapper that reports `enabled() == false`; its pipeline and
//! font texture are built the first time a marker is drawn.
//!
//! On with no marker up: the same plus two key-state lookups a frame.
//!
//! On, per marker and frame: one distance, one matrix, 12 solid vertices plus 6 per glyph
//! (`42m`: 18) into reused buffers; the text is laid out again only when the rounded distance
//! changes. A key press costs one ray of up to 256 blocks and one pass over the tracked
//! entities: per press, never per frame.

use mc_math::DVec3;
use mc_mods::mods::markers::set::{Marker, MarkerSet};
use mc_render::entity::NameplatePhase;
use mc_render::{Camera, Gpu, Input, InputState, PhaseList, RenderPhase};
use mc_ui::font::FontTexture;
use mc_ui::marker_label::{self, MarkerLabel};
use mc_ui::waypoint_label::{self, LabelScratch};

use super::super::{Client, Shared, Ui};
use super::m9_waypoints::{upload_sheet, LABEL_ORDER};
use crate::app::interact;
use crate::session::{RenderViewTarget, Session};
use crate::sync::ClientEvent;

const MOD_ID: &str = "markers";

/// How far the marker ray reaches, in blocks.
pub(crate) const MAX_DISTANCE: f64 = 256.0;

/// The ray is walked in pieces this long. Vanilla's block walk gives up after 200 block steps
/// (`World.rayTraceBlocks`, `k1 = 200`), which a diagonal 256-block ray needs up to 444 of; a
/// 64-block piece needs at most 111.
const SEGMENT: f64 = 64.0;

/// Pressing the key within this many blocks of a marker removes it instead of placing another.
pub(crate) const REPRESS_RADIUS: f64 = 1.0;

/// The ping: vanilla's note-block "pling", the sound servers use for notifications. Its sound
/// category is "record" (the Jukebox/Note Blocks slider).
pub(crate) const SOUND: &str = "note.pling";

/// Pitch of the ping when a marker is placed (two octaves up: short and bright).
pub(crate) const PLACE_PITCH: f32 = 2.0;

/// Pitch of the ping when markers are removed by a key (one octave lower than placing).
pub(crate) const REMOVE_PITCH: f32 = 1.0;

/// The ping's volume (non-positional, so at most full gain).
const SOUND_VOLUME: f32 = 0.6;

/// The settings the hooks read, refreshed only when the settings generation moves.
#[derive(Debug, Clone, Copy)]
struct Snapshot {
    seen: u64,
    enabled: bool,
    key: Input,
    clear_key: Input,
    lifetime_ms: u64,
    teleport_distance: f64,
    max: usize,
    /// `0xRRGGBB`.
    rgb: u32,
    sound: bool,
    /// `size` as a factor (1.0 = 100%).
    scale: f32,
}

impl Default for Snapshot {
    fn default() -> Self {
        Snapshot {
            seen: 0,
            enabled: false,
            key: Input::Unbound,
            clear_key: Input::Unbound,
            lifetime_ms: 5 * 60_000,
            teleport_distance: 200.0,
            max: 3,
            rgb: 0xFF_FF55,
            sound: true,
            scale: 1.0,
        }
    }
}

/// The label `NameplatePhase`, built the first time a marker is drawn.
#[derive(Default)]
struct MarkerPhase {
    inner: Option<NameplatePhase>,
    active: bool,
}

impl RenderPhase for MarkerPhase {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        if let Some(inner) = self.inner.as_mut() {
            inner.record(gpu, enc, view, cam);
        }
    }

    fn name(&self) -> &'static str {
        "markers/labels"
    }

    fn enabled(&self) -> bool {
        self.active && self.inner.is_some()
    }
}

/// Everything X13 keeps on the client (`Client::x13`).
#[derive(Default)]
pub(crate) struct MarkersState {
    snap: Snapshot,
    /// The markers that are up, and the removal rules.
    set: MarkerSet,
    /// The place and clear keys were down last frame.
    key_down: bool,
    clear_down: bool,
    /// One cached label per marker, in set order, and the set revision they were sized for.
    labels: Vec<MarkerLabel>,
    labels_for: Option<u64>,
    scratch: LabelScratch,
    /// This frame's markers as `(distance, index)`, far to near.
    order: Vec<(f64, usize)>,
    phase: Option<Shared<MarkerPhase>>,
    /// Font sheets uploaded to (or failed for) the phase, for `font_generation`.
    sheets: Vec<FontTexture>,
    font_generation: u64,
    /// The phase was left enabled by the last frame.
    active: bool,
}

impl MarkersState {
    /// Refresh the snapshot. One `u64` compare when nothing changed.
    fn sync(&mut self, settings: &mc_mods::ModSettings) {
        if self.snap.seen == settings.generation() {
            return;
        }
        let was_enabled = self.snap.enabled;
        let minutes = settings.f32(MOD_ID, "duration");
        self.snap = Snapshot {
            seen: settings.generation(),
            enabled: settings.enabled(MOD_ID),
            key: Input::from_lwjgl(settings.key(MOD_ID, "key")),
            clear_key: Input::from_lwjgl(settings.key(MOD_ID, "clear_key")),
            lifetime_ms: (f64::from(minutes.max(0.0)) * 60_000.0) as u64,
            teleport_distance: f64::from(settings.f32(MOD_ID, "teleport_distance")),
            max: settings.f32(MOD_ID, "max_markers").round().max(1.0) as usize,
            rgb: settings.color(MOD_ID, "color") & 0xFF_FFFF,
            sound: settings.bool(MOD_ID, "sound"),
            scale: settings.f32(MOD_ID, "size") / 100.0,
        };
        if self.snap.enabled && !was_enabled {
            // A key already held when the mod turns on is not a press.
            self.key_down = true;
            self.clear_down = true;
        }
        if !self.snap.enabled {
            // Off means no markers, so every hook returns at its first test.
            self.set.clear();
        } else {
            // A lowered maximum applies at once.
            self.set.limit(self.snap.max);
        }
    }
}

/// A marker's point as a vector.
fn marker_pos(marker: &Marker) -> DVec3 {
    let [x, y, z] = marker.pos;
    DVec3::new(x, y, z)
}

/// The events that replace the world: a JoinGame, a respawn into another dimension, and the
/// end of the connection.
fn is_world_change(event: &ClientEvent) -> bool {
    matches!(
        event,
        ClientEvent::JoinedGame(_)
            | ClientEvent::Respawned {
                changed_world: true,
                ..
            }
            | ClientEvent::Disconnected(_)
    )
}

/// The first block the ray from `eye` along `look` (a unit vector) hits within
/// [`MAX_DISTANCE`], as its exact hit point. Vanilla's own block walk (the crosshair's
/// selection boxes, liquids passed through), in [`SEGMENT`]-long pieces.
fn raytrace_far(world: &mc_world::World, eye: DVec3, look: DVec3) -> Option<DVec3> {
    let mut start = 0.0;
    while start < MAX_DISTANCE {
        let end = (start + SEGMENT).min(MAX_DISTANCE);
        let hit = interact::raytrace_blocks_with_options(
            world,
            eye + look * start,
            eye + look * end,
            false,
            false,
            false,
        )
        .and_then(interact::RayTrace::as_block);
        if let Some(hit) = hit {
            return Some(hit.hit_vec);
        }
        start = end;
    }
    None
}

impl Client {
    /// Register the phase as an empty wrapper. No GPU object is built here;
    /// that waits for the first marker. Runs again after a resource reload, which rebuilt the
    /// phase list, so everything derived from the old phase is dropped.
    pub(crate) fn install_marker_phase(&mut self, phases: &mut PhaseList) {
        let phase = Shared::new(MarkerPhase::default());
        // `insert` is stable: after the waypoint labels, which `install` registered just before.
        phases.insert(LABEL_ORDER, Box::new(phase.clone()));
        self.x13.phase = Some(phase);
        self.x13.active = false;
        self.x13.sheets.clear();
        self.x13.font_generation = 0;
    }

    /// A new world or the end of the
    /// connection removes every marker. One `is_empty` test while no marker is up.
    pub(crate) fn markers_events(&mut self, events: &[ClientEvent]) {
        if self.x13.set.is_empty() {
            return;
        }
        if events.iter().any(is_world_change) {
            self.x13.set.world_changed();
        }
    }

    /// The timeout and the far-teleport rule. One `is_empty` test
    /// while no marker is up.
    pub(crate) fn markers_tick(&mut self) {
        if self.x13.set.is_empty() {
            return;
        }
        self.x13.sync(&self.mods.settings);
        let now = self.markers_now_ms();
        let snap = self.x13.snap;
        self.x13.set.expire(now, snap.lifetime_ms);
        let feet = self
            .session
            .as_ref()
            .and_then(Session::predictor)
            .map(|predictor| predictor.state.pos);
        if let Some(pos) = feet {
            self.x13
                .set
                .track_position([pos.x, pos.y, pos.z], snap.teleport_distance);
        }
    }

    /// The place and clear keys, as press edges of the live input.
    /// Runs after `handle_screen_input` and the mouse look, so `self.ui` and the view are current.
    pub(crate) fn markers_shell_frame(&mut self, input: &InputState, partial_ticks: f32) {
        self.x13.sync(&self.mods.settings);
        if !self.x13.snap.enabled {
            return;
        }
        let place_down = input.is_down(self.x13.snap.key);
        let clear_down = input.is_down(self.x13.snap.clear_key);
        let place = place_down && !self.x13.key_down;
        let clear = clear_down && !self.x13.clear_down;
        self.x13.key_down = place_down;
        self.x13.clear_down = clear_down;
        // In game only, and never with F3 held (F3 + key is vanilla's debug chord space).
        if !(place || clear) || !matches!(self.ui, Ui::InGame) || self.debug_f3_held {
            return;
        }
        if clear {
            if self.x13.set.clear() {
                self.marker_sound(REMOVE_PITCH);
            }
        } else {
            self.press_marker_key(partial_ticks);
        }
    }

    /// This frame's pins. One `is_empty` test (and a `bool`) while no
    /// marker is up.
    pub(crate) fn markers_frame(&mut self, gpu: &Gpu, camera: &Camera) {
        if !self.x13.set.is_empty() {
            self.x13.sync(&self.mods.settings);
            // Leaving the server through a menu drops the session without an event.
            if self.session.is_none() {
                self.x13.set.world_changed();
            }
        }
        if self.x13.set.is_empty() {
            if self.x13.active {
                self.deactivate_marker_phase();
            }
            return;
        }
        self.draw_markers(gpu, camera);
    }

    /// Milliseconds since the client started: the clock markers are timed on.
    fn markers_now_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// The place key went down in game: remove the marker the crosshair is on, else put one
    /// where the ray lands.
    fn press_marker_key(&mut self, partial_ticks: f32) {
        let Some((eye, yaw, pitch)) = self.marker_view(partial_ticks) else {
            return;
        };
        let look = interact::look_vector(yaw, pitch);
        if let Some(id) = self.aimed_marker(eye, look, yaw, pitch) {
            self.x13.set.remove(id);
            self.marker_sound(REMOVE_PITCH);
            return;
        }
        let Some(target) = self.marker_target(eye, look, partial_ticks) else {
            return;
        };
        let target = [target.x, target.y, target.z];
        if let Some(id) = self.x13.set.nearest_within(target, REPRESS_RADIUS) {
            self.x13.set.remove(id);
            self.marker_sound(REMOVE_PITCH);
            return;
        }
        let now = self.markers_now_ms();
        let max = self.x13.snap.max;
        self.x13.set.place(target, now, max);
        self.marker_sound(PLACE_PITCH);
    }

    /// The eye and the look angles the crosshair uses (as `update_render_hit_result` does): the
    /// local player's interpolated eye, or the spectated entity's.
    fn marker_view(&self, partial_ticks: f32) -> Option<(DVec3, f32, f32)> {
        let session = self.session.as_ref()?;
        if session.render_view_is_local() {
            let predictor = session.predictor()?;
            let eye = self.interpolated_eye(&predictor.state, partial_ticks);
            Some((eye, predictor.state.yaw, predictor.state.pitch))
        } else {
            let eye = session.render_view_eye_position(partial_ticks)?;
            let (yaw, pitch) = session.render_view_rotation(partial_ticks)?;
            Some((eye, yaw, pitch))
        }
    }

    /// The marker whose drawn pin or distance the crosshair is on (the one nearest the
    /// crosshair's centre when several are).
    fn aimed_marker(&self, eye: DVec3, look: DVec3, yaw: f32, pitch: f32) -> Option<u64> {
        let x13 = &self.x13;
        let labels =
            x13.labels_for == Some(x13.set.revision()) && x13.labels.len() == x13.set.len();
        let mut best: Option<(f64, u64)> = None;
        for (index, marker) in x13.set.markers().iter().enumerate() {
            let rel = marker_pos(marker) - eye;
            // The distance box counts once it has been drawn; before that only the pin does.
            let width = match x13.labels.get(index) {
                Some(label) if labels && label.is_built() => label.width(),
                _ => 0,
            };
            if !marker_label::aimed_at(rel, look, yaw, pitch, x13.snap.scale, width) {
                continue;
            }
            let off_centre = -rel.normalize().dot(look);
            if best.is_none_or(|(score, _)| off_centre < score) {
                best = Some((off_centre, marker.id));
            }
        }
        best.map(|(_, id)| id)
    }

    /// Where a press lands: the nearest entity or block on the ray within [`MAX_DISTANCE`],
    /// else the point that far out along it.
    fn marker_target(&self, eye: DVec3, look: DVec3, partial_ticks: f32) -> Option<DVec3> {
        let session = self.session.as_ref()?;
        if !eye.is_finite() || !look.is_finite() {
            return None;
        }
        let block = raytrace_far(&session.state().world, eye, look);
        let reach = block.map_or(MAX_DISTANCE, |hit| (hit - eye).length());
        let entity = self.marker_entity_hit(session, eye, look, reach, partial_ticks);
        Some(entity.or(block).unwrap_or(eye + look * MAX_DISTANCE))
    }

    /// The nearest point within `reach` where the ray enters a visible entity's box, as drawn
    /// this frame. The same boxes and filter as the crosshair (`pick_result_at`), at any range,
    /// minus what the player cannot see and minus the ridden vehicle.
    fn marker_entity_hit(
        &self,
        session: &Session,
        eye: DVec3,
        look: DVec3,
        reach: f64,
        partial_ticks: f32,
    ) -> Option<DVec3> {
        use mc_data::EntityType;
        let state = session.state();
        let world = &state.world;
        let view_entity = match session.render_view_target() {
            RenderViewTarget::Entity { network_id, .. } => network_id,
            RenderViewTarget::Local => state.local.entity_id,
        };
        let riding = session.riding();
        let end = eye + look * reach;
        let mut best: Option<(f64, DVec3)> = None;
        for (network_id, tracked) in state.entities.iter() {
            if network_id == view_entity || riding == Some(network_id) {
                continue;
            }
            let kind = tracked.kind.entity_type();
            if kind.is_none() && tracked.uuid.is_none() {
                continue;
            }
            // A hanging entity lies flat on its wall: the wall behind it is the same spot.
            if matches!(
                kind,
                Some(EntityType::Painting | EntityType::ItemFrame | EntityType::LeashKnot)
            ) {
                continue;
            }
            let pos = tracked.render_position(world, partial_ticks);
            if (pos - eye).length_squared() > (reach + 4.0) * (reach + 4.0) {
                continue;
            }
            let collision = interact::CollisionState {
                dead: tracked.render.death_time >= interact::DEATH_ANIMATION_TICKS,
                marker: kind == Some(EntityType::ArmorStand)
                    && tracked
                        .with_watched(world, |w| w.armor_stand().is_some_and(|stand| stand.marker))
                        .unwrap_or(false),
            };
            if !interact::can_be_collided_with(kind, collision) {
                continue;
            }
            // Never mark what the player cannot see.
            if crate::app::appearance::is_invisible(world, tracked) {
                continue;
            }
            let spectating = tracked.uuid.is_some_and(|uuid| {
                state.players.get(&uuid).is_some_and(|info| {
                    info.game_mode == mc_protocol::play::s2c::GameMode::Spectator
                })
            });
            if spectating {
                continue;
            }
            let (width, height) = tracked.dimensions();
            let border = f64::from(interact::collision_border_size(kind));
            let bounds =
                interact::size_bounding_box(pos, width, height).expand(border, border, border);
            // Standing inside a box (a mount, a crowd): mark past it.
            if bounds.is_vec_inside(eye) {
                continue;
            }
            let Some((point, _)) = bounds.calculate_intercept(eye, end) else {
                continue;
            };
            let d = (point - eye).length_squared();
            if best.is_none_or(|(nearest, _)| d < nearest) {
                best = Some((d, point));
            }
        }
        best.map(|(_, point)| point)
    }

    /// The ping, unless the Sound option is off. Non-positional, through the one sound funnel
    /// (`Client::play_sound`), so the volume sliders apply and `MC_NO_AUDIO` silences it.
    fn marker_sound(&mut self, pitch: f32) {
        if !self.x13.snap.sound {
            return;
        }
        let mut request =
            mc_audio::PlayRequest::ambient(mc_assets::ResourceLocation::minecraft(SOUND));
        request.volume = SOUND_VOLUME;
        request.pitch = pitch;
        self.play_sound(request);
    }

    fn deactivate_marker_phase(&mut self) {
        self.x13.active = false;
        if let Some(phase) = &self.x13.phase {
            phase.with(|phase| phase.active = false);
        }
    }

    fn draw_markers(&mut self, gpu: &Gpu, camera: &Camera) {
        let Some(phase) = self.x13.phase.clone() else {
            return;
        };
        let anaglyph = self.settings.anaglyph;
        let hide_gui = self.hide_gui;

        // The GPU half, on first use.
        let built = phase.with(|phase| {
            phase.active = true;
            match phase.inner.as_mut() {
                Some(inner) => {
                    if inner.anaglyph_enabled() != anaglyph {
                        inner.set_anaglyph_with_gpu(gpu, anaglyph);
                    }
                    false
                }
                None => {
                    phase.inner = Some(NameplatePhase::new_with_anaglyph(gpu, anaglyph));
                    true
                }
            }
        });
        self.x13.active = true;

        let font_generation = self.font.layout_generation();
        let Client {
            x13, font, assets, ..
        } = self;
        if built || x13.font_generation != font_generation {
            x13.font_generation = font_generation;
            x13.sheets.clear();
        }
        // Labels are cached by index: re-sized (and so re-laid-out) when the list changes.
        let revision = x13.set.revision();
        if x13.labels_for != Some(revision) || x13.labels.len() != x13.set.len() {
            x13.labels.clear();
            x13.labels.resize_with(x13.set.len(), MarkerLabel::default);
            x13.labels_for = Some(revision);
        }

        // Far to near, so nearer pins draw on top.
        x13.order.clear();
        for (index, marker) in x13.set.markers().iter().enumerate() {
            x13.order
                .push(((marker_pos(marker) - camera.pos).length(), index));
        }
        x13.order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        let markers = x13.set.markers();
        phase.with(|phase| {
            let Some(inner) = phase.inner.as_mut() else {
                return;
            };
            inner.begin();
            if !hide_gui {
                for &(distance, index) in &x13.order {
                    let rel = marker_pos(&markers[index]) - camera.pos;
                    let Some(matrix) =
                        waypoint_label::label_matrix(rel, camera.yaw, camera.pitch, x13.snap.scale)
                    else {
                        continue;
                    };
                    let label = &mut x13.labels[index];
                    label.update(font, distance);
                    for &sheet in label.sheets() {
                        if !x13.sheets.contains(&sheet) {
                            x13.sheets.push(sheet);
                            upload_sheet(inner, gpu, font, &assets.resources, sheet);
                        }
                    }
                    label.emit(matrix, x13.snap.rgb, &mut x13.scratch, inner);
                }
            }
            inner.finish(gpu);
        });
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::session::JoinSettings;
    use mc_math::BlockPos;
    use mc_physics::state::MoveInput;
    use winit::keyboard::KeyCode;

    type Channels = (
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    );

    /// A real client over the vanilla jar, with a temp game dir (never the user's).
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// Join an empty world with no socket behind it, standing at `feet`, looking along
    /// `yaw`/`pitch`, in game.
    fn join(client: &mut Client, feet: DVec3, yaw: f32, pitch: f32) -> Channels {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(mc_protocol::play::s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                }),
            ))
            .expect("join channel");
        session.poll(16);
        let predictor = session
            .predictor_mut()
            .expect("JoinGame builds the predictor");
        predictor.state.pos = feet;
        predictor.state.yaw = yaw;
        predictor.state.pitch = pitch;
        client.session = Some(session);
        client.ui = Ui::InGame;
        (to_game, to_net)
    }

    fn look(client: &mut Client, yaw: f32, pitch: f32) {
        let predictor = client
            .session
            .as_mut()
            .and_then(Session::predictor_mut)
            .expect("joined");
        predictor.state.yaw = yaw;
        predictor.state.pitch = pitch;
    }

    fn stand_at(client: &mut Client, feet: DVec3) {
        client
            .session
            .as_mut()
            .and_then(Session::predictor_mut)
            .expect("joined")
            .state
            .pos = feet;
    }

    /// Load the whole column at chunk `(x, z)` as air.
    fn load_column(client: &mut Client, x: i32, z: i32) {
        let world = &mut client.session.as_mut().expect("joined").state_mut().world;
        let data = vec![0u8; mc_world::expected_chunk_data_len(0xFFFF, true, true)];
        world
            .apply_chunk_data(mc_math::ChunkPos::new(x, z), true, 0xFFFF, &data)
            .expect("an empty column decodes");
    }

    /// Stone at `pos`, in a column that must already be loaded.
    fn set_stone(client: &mut Client, pos: BlockPos) {
        let stone = mc_data::BlockState::new(mc_data::block_by_name("stone").expect("stone"), 0);
        let world = &mut client.session.as_mut().expect("joined").state_mut().world;
        assert!(
            world.is_chunk_loaded(pos.chunk_pos()),
            "load {pos:?}'s column first"
        );
        world.set_block(pos, stone);
        assert_eq!(world.block_at(pos), stone);
    }

    fn press(input: &mut InputState, code: KeyCode, down: bool) {
        let _ = input.on_key(code, down, false);
    }

    /// One full press and release of `code` through the shell-frame hook.
    fn tap(client: &mut Client, input: &mut InputState, code: KeyCode) {
        press(input, code, true);
        client.markers_shell_frame(input, 1.0);
        press(input, code, false);
        client.markers_shell_frame(input, 1.0);
    }

    fn positions(client: &Client) -> Vec<DVec3> {
        client.x13.set.markers().iter().map(marker_pos).collect()
    }

    fn eye_and_look(client: &Client) -> (DVec3, DVec3) {
        let (eye, yaw, pitch) = client.marker_view(1.0).expect("a view");
        (eye, interact::look_vector(yaw, pitch))
    }

    /// The last ping: `(event, pitch, positional)`.
    fn last_ping(client: &Client) -> Option<(String, f32, bool)> {
        client.audio.last_play().map(|record| {
            (
                record.request.event.path.clone(),
                record.request.pitch,
                record.request.position.is_some(),
            )
        })
    }

    /// Creeper `entity_id` with its feet at `feet` (wire fixed point), optionally invisible.
    fn spawn_creeper(client: &mut Client, entity_id: i32, feet: DVec3, invisible: bool) {
        let session = client.session.as_mut().expect("session");
        let metadata = if invisible {
            vec![mc_protocol::MetadataEntry {
                index: 0,
                value: mc_protocol::MetadataValue::Byte(0x20),
            }]
        } else {
            Vec::new()
        };
        crate::sync::apply(
            session.state_mut(),
            mc_protocol::play::S2CPlay::SpawnMob(mc_protocol::play::s2c::SpawnMob {
                entity_id,
                entity_type: 50,
                x: (feet.x * 32.0) as i32,
                y: (feet.y * 32.0) as i32,
                z: (feet.z * 32.0) as i32,
                yaw: mc_protocol::Angle(0),
                pitch: mc_protocol::Angle(0),
                head_pitch: mc_protocol::Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata,
            }),
            &mut Vec::new(),
        )
        .expect("creeper spawn applies");
    }

    #[test]
    fn with_the_mod_off_the_keys_do_nothing() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        assert!(!client.mods.settings.enabled(MOD_ID), "off by default");
        assert_eq!(client.mods.settings.key(MOD_ID, "key"), 44, "Z by default");
        let mut input = InputState::new();
        client.audio.clear_recent();
        client.markers_shell_frame(&input, 1.0);
        tap(&mut client, &mut input, KeyCode::KeyZ);
        assert!(client.x13.set.is_empty());
        assert!(client.audio.last_play().is_none(), "no ping");
        // Every other hook returns at its first test.
        client.markers_tick();
        client.markers_events(&[ClientEvent::Disconnected(
            crate::net::DisconnectReason::ClosedByClient,
        )]);
        assert!(client.x13.set.is_empty());
    }

    #[test]
    fn the_key_marks_the_block_face_under_the_crosshair_once_per_press() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        // Looking along +Z at a stone wall 19.5 blocks ahead.
        let (_to_game, mut to_net) = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        load_column(&mut client, 0, 1);
        set_stone(&mut client, BlockPos::new(0, 65, 20));
        client.mods.settings.set_enabled(MOD_ID, true);
        let mut input = InputState::new();
        // Whatever the join queued (client settings, brand) is not ours.
        while to_net.try_recv().is_ok() {}

        // Held when the mod turns on: not a press.
        press(&mut input, KeyCode::KeyZ, true);
        client.markers_shell_frame(&input, 1.0);
        assert!(client.x13.set.is_empty());
        press(&mut input, KeyCode::KeyZ, false);
        client.markers_shell_frame(&input, 1.0);

        press(&mut input, KeyCode::KeyZ, true);
        client.markers_shell_frame(&input, 1.0);
        client.markers_shell_frame(&input, 1.0); // still held: no repeat
        let (eye, _) = eye_and_look(&client);
        let placed = positions(&client);
        assert_eq!(placed.len(), 1, "{placed:?}");
        assert!(
            (placed[0] - DVec3::new(eye.x, eye.y, 20.0)).length() < 1e-3,
            "on the wall's north face, straight ahead: {} (eye {eye})",
            placed[0]
        );
        assert_eq!(
            last_ping(&client),
            Some((SOUND.to_owned(), PLACE_PITCH, false)),
            "a non-positional ping"
        );
        press(&mut input, KeyCode::KeyZ, false);
        client.markers_shell_frame(&input, 1.0);

        // F3 held (vanilla's debug chords) or a screen open: nothing.
        look(&mut client, 20.0, 0.0);
        client.debug_f3_held = true;
        tap(&mut client, &mut input, KeyCode::KeyZ);
        client.debug_f3_held = false;
        client.ui = Ui::Loading;
        tap(&mut client, &mut input, KeyCode::KeyZ);
        client.ui = Ui::InGame;
        assert_eq!(client.x13.set.len(), 1);

        // Nothing to hit: a point 256 blocks out along the ray.
        look(&mut client, 90.0, -30.0);
        tap(&mut client, &mut input, KeyCode::KeyZ);
        let (eye, ray) = eye_and_look(&client);
        let placed = positions(&client);
        assert_eq!(placed.len(), 2);
        assert!((placed[1] - (eye + ray * MAX_DISTANCE)).length() < 1e-6);

        // Purely local: not one packet left the client for any of it.
        client.markers_tick();
        assert!(to_net.try_recv().is_err(), "markers send nothing");
    }

    #[test]
    fn the_ray_reaches_past_one_vanilla_walk_to_a_block_230_blocks_away() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        // A diagonal ray: about 350 block steps to 230 blocks, more than vanilla's 200.
        let _net = join(&mut client, DVec3::new(0.5, 100.0, 0.5), 37.0, 8.0);
        let (eye, ray) = eye_and_look(&client);
        let far = eye + ray * 230.0;
        let block = BlockPos::new(
            mc_math::floor_f64(far.x),
            mc_math::floor_f64(far.y),
            mc_math::floor_f64(far.z),
        );
        let chunk = block.chunk_pos();
        load_column(&mut client, chunk.x, chunk.z);
        set_stone(&mut client, block);
        let world = &client.session.as_ref().unwrap().state().world;
        assert_eq!(
            interact::raytrace_blocks_with_options(
                world,
                eye,
                eye + ray * MAX_DISTANCE,
                false,
                false,
                false
            ),
            None,
            "a single vanilla walk gives up before it gets there"
        );
        client.mods.settings.set_enabled(MOD_ID, true);
        let mut input = InputState::new();
        client.markers_shell_frame(&input, 1.0);
        tap(&mut client, &mut input, KeyCode::KeyZ);
        let placed = positions(&client);
        assert_eq!(placed.len(), 1);
        let distance = (placed[0] - eye).length();
        assert!((228.0..=230.5).contains(&distance), "{distance} blocks");
        let on = |v: f64, lo: i32| {
            let lo = f64::from(lo);
            (lo - 1e-6..=lo + 1.0 + 1e-6).contains(&v)
        };
        let p = placed[0];
        assert!(
            on(p.x, block.x) && on(p.y, block.y) && on(p.z, block.z),
            "{p} is on {block:?}"
        );
    }

    #[test]
    fn an_entity_in_front_takes_the_marker_unless_it_is_invisible() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        // The creepers' column must be loaded, or they are not in the world yet (vanilla queues
        // an entity spawned in an unloaded chunk) and the ray could not meet them anyway.
        load_column(&mut client, 0, 0);
        load_column(&mut client, 0, 1);
        set_stone(&mut client, BlockPos::new(0, 65, 20));
        client.mods.settings.set_enabled(MOD_ID, true);
        let mut input = InputState::new();
        client.markers_shell_frame(&input, 1.0);

        // An invisible creeper 9.5 blocks ahead is passed through: the wall is marked.
        spawn_creeper(&mut client, 70, DVec3::new(0.5, 64.0, 10.0), true);
        let resident = |client: &Client, id: i32| {
            let session = client.session.as_ref().unwrap();
            session
                .state()
                .entities
                .iter()
                .any(|(network_id, _)| network_id == id)
        };
        assert!(
            resident(&client, 70),
            "in the world, so only its invisibility hides it"
        );
        tap(&mut client, &mut input, KeyCode::KeyZ);
        let placed = positions(&client);
        assert_eq!(placed.len(), 1);
        assert!((placed[0].z - 20.0).abs() < 1e-9, "the wall: {}", placed[0]);
        assert!(client.x13.set.clear());

        // A visible one in front of the wall takes the marker: the front of its box (half width
        // 0.3 plus the 0.1 collision border), where the ray met it.
        spawn_creeper(&mut client, 71, DVec3::new(0.5, 64.0, 15.0), false);
        assert!(resident(&client, 71));
        tap(&mut client, &mut input, KeyCode::KeyZ);
        let placed = positions(&client);
        assert_eq!(placed.len(), 1);
        assert!(
            (placed[0].z - 14.6).abs() < 1e-6,
            "the creeper: {}",
            placed[0]
        );

        // It marks a spot, not the entity: the marker stays when the creeper walks off.
        let session = client.session.as_mut().unwrap();
        crate::sync::apply(
            session.state_mut(),
            mc_protocol::play::S2CPlay::EntityTeleport(mc_protocol::play::s2c::EntityTeleport {
                entity_id: 71,
                x: 40 * 32,
                y: 64 * 32,
                z: 40 * 32,
                yaw: mc_protocol::Angle(0),
                pitch: mc_protocol::Angle(0),
                on_ground: true,
            }),
            &mut Vec::new(),
        )
        .expect("teleport applies");
        client.markers_tick();
        assert_eq!(positions(&client), placed);
    }

    #[test]
    fn pressing_on_a_marker_or_right_next_to_it_removes_it() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        load_column(&mut client, 0, 1);
        load_column(&mut client, -1, 1);
        for x in -2..=2 {
            set_stone(&mut client, BlockPos::new(x, 65, 20));
        }
        client.mods.settings.set_enabled(MOD_ID, true);
        let mut input = InputState::new();
        client.markers_shell_frame(&input, 1.0);

        // On its pin.
        tap(&mut client, &mut input, KeyCode::KeyZ);
        assert_eq!(client.x13.set.len(), 1);
        tap(&mut client, &mut input, KeyCode::KeyZ);
        assert!(client.x13.set.is_empty(), "the second press removed it");
        assert_eq!(
            last_ping(&client),
            Some((SOUND.to_owned(), REMOVE_PITCH, false)),
            "the lower ping"
        );

        // Beside the pin (2.6° off, past its 2.35° half width with the margin) but within a
        // block of it on the wall (0.89 blocks): still "on it".
        tap(&mut client, &mut input, KeyCode::KeyZ);
        let first = positions(&client)[0];
        look(&mut client, 2.6, 0.0);
        let (eye, ray) = eye_and_look(&client);
        let marker = client.x13.set.markers()[0];
        assert!(
            !marker_label::aimed_at(marker_pos(&marker) - eye, ray, 2.6, 0.0, 1.0, 0),
            "the crosshair is off the pin"
        );
        tap(&mut client, &mut input, KeyCode::KeyZ);
        assert!(client.x13.set.is_empty(), "removed by the 1-block rule");

        // Further off (6°, 2.1 blocks on the wall): a second marker.
        look(&mut client, 0.0, 0.0);
        tap(&mut client, &mut input, KeyCode::KeyZ);
        look(&mut client, 6.0, 0.0);
        tap(&mut client, &mut input, KeyCode::KeyZ);
        let placed = positions(&client);
        assert_eq!(placed.len(), 2, "{placed:?}");
        assert!((placed[0] - first).length() < 1e-9);
    }

    #[test]
    fn the_clear_key_removes_all_and_the_maximum_replaces_the_oldest() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        client.mods.settings.set_enabled(MOD_ID, true);
        client.mods.settings.set_key(MOD_ID, "clear_key", 45); // X
        client.mods.settings.set_f32(MOD_ID, "max_markers", 2.0);
        let mut input = InputState::new();
        client.markers_shell_frame(&input, 1.0);
        for yaw in [0.0, 30.0, 60.0] {
            look(&mut client, yaw, -10.0);
            tap(&mut client, &mut input, KeyCode::KeyZ);
        }
        let placed = positions(&client);
        assert_eq!(placed.len(), 2, "two at most");
        look(&mut client, 30.0, -10.0);
        let (eye, ray) = eye_and_look(&client);
        assert!(
            (placed[0] - (eye + ray * MAX_DISTANCE)).length() < 1e-6,
            "the first (yaw 0) was replaced"
        );
        // A lowered maximum applies at once.
        client.mods.settings.set_f32(MOD_ID, "max_markers", 1.0);
        client.markers_shell_frame(&input, 1.0);
        assert_eq!(client.x13.set.len(), 1);

        tap(&mut client, &mut input, KeyCode::KeyX);
        assert!(client.x13.set.is_empty(), "cleared");
        assert_eq!(
            last_ping(&client),
            Some((SOUND.to_owned(), REMOVE_PITCH, false))
        );
        // With the Sound option off, no ping.
        client.mods.settings.set_bool(MOD_ID, "sound", false);
        client.audio.clear_recent();
        tap(&mut client, &mut input, KeyCode::KeyZ);
        assert_eq!(client.x13.set.len(), 1);
        assert!(client.audio.last_play().is_none());
    }

    #[test]
    fn a_new_world_or_dimension_clears_markers_but_a_respawn_in_place_does_not() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let (to_game, _to_net) = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        client.mods.settings.set_enabled(MOD_ID, true);
        let mut input = InputState::new();
        client.markers_shell_frame(&input, 1.0);
        let respawn = |dimension: i32| {
            crate::net::ToGame::Packet(mc_protocol::play::S2CPlay::Respawn(
                mc_protocol::play::s2c::Respawn {
                    dimension,
                    difficulty: 2,
                    game_mode: 0,
                    level_type: "default".into(),
                },
            ))
        };

        // A death respawn in the same dimension keeps the world (and the markers), through the
        // real poll path.
        tap(&mut client, &mut input, KeyCode::KeyZ);
        to_game.try_send(respawn(0)).expect("respawn");
        client.poll_in_place();
        assert!(
            client.last_events.iter().any(|e| matches!(
                e,
                ClientEvent::Respawned {
                    changed_world: false,
                    ..
                }
            )),
            "{:?}",
            client.last_events
        );
        assert_eq!(client.x13.set.len(), 1);

        // Into the Nether: a new world.
        to_game.try_send(respawn(-1)).expect("respawn");
        client.poll_in_place();
        assert!(client.x13.set.is_empty(), "{:?}", client.last_events);

        // A JoinGame and a disconnect, straight through the hook.
        client.ui = Ui::InGame;
        for event in [
            ClientEvent::JoinedGame(Box::new(mc_protocol::play::s2c::JoinGame {
                entity_id: 1,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 2,
                max_players: 20,
                level_type: "default".into(),
                reduced_debug_info: false,
            })),
            ClientEvent::Disconnected(crate::net::DisconnectReason::ClosedByClient),
        ] {
            tap(&mut client, &mut input, KeyCode::KeyZ);
            assert_eq!(client.x13.set.len(), 1);
            client.markers_events(&[event]);
            assert!(client.x13.set.is_empty());
        }

        // Leaving through a menu drops the session without an event: the next frame clears.
        tap(&mut client, &mut input, KeyCode::KeyZ);
        client.session = None;
        let Ok(gpu) = Gpu::headless(64, 64, mc_render::GpuOptions::default()) else {
            eprintln!("SKIP: no wgpu adapter for the frame half");
            return;
        };
        client.markers_frame(&gpu, &super::super::super::menu_camera());
        assert!(client.x13.set.is_empty());
    }

    #[test]
    fn a_far_teleport_clears_markers_and_ordinary_movement_does_not() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        client.mods.settings.set_enabled(MOD_ID, true);
        let mut input = InputState::new();
        client.markers_shell_frame(&input, 1.0);
        tap(&mut client, &mut input, KeyCode::KeyZ);

        // Driven through the real 20 Hz tick. The default limit is 200 blocks.
        client.tick(&MoveInput::default());
        stand_at(&mut client, DVec3::new(60.5, 70.0, 30.5)); // an ender pearl's worth
        client.tick(&MoveInput::default());
        assert_eq!(client.x13.set.len(), 1, "65 blocks is not a far teleport");
        stand_at(&mut client, DVec3::new(360.5, 70.0, 30.5));
        client.tick(&MoveInput::default());
        assert!(client.x13.set.is_empty(), "300 blocks in one tick is");

        // The option moves the limit.
        client
            .mods
            .settings
            .set_f32(MOD_ID, "teleport_distance", 50.0);
        tap(&mut client, &mut input, KeyCode::KeyZ);
        client.tick(&MoveInput::default());
        stand_at(&mut client, DVec3::new(300.5, 70.0, 30.5));
        client.tick(&MoveInput::default());
        assert!(
            client.x13.set.is_empty(),
            "60 blocks is over a 50-block limit"
        );
    }

    #[test]
    fn markers_go_after_the_set_minutes() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        client.mods.settings.set_enabled(MOD_ID, true);
        client.mods.settings.set_f32(MOD_ID, "duration", 1.0);
        let mut input = InputState::new();
        client.markers_shell_frame(&input, 1.0);
        tap(&mut client, &mut input, KeyCode::KeyZ);
        client.tick(&MoveInput::default());
        assert_eq!(client.x13.set.len(), 1);
        // Move the client's clock 59 s, then past a minute, on.
        let Some(earlier) = client.started.checked_sub(Duration::from_secs(59)) else {
            eprintln!("skip: the clock cannot go back a minute here");
            return;
        };
        client.started = earlier;
        client.tick(&MoveInput::default());
        assert_eq!(client.x13.set.len(), 1, "59 s is not a minute");
        let Some(earlier) = client.started.checked_sub(Duration::from_secs(2)) else {
            return;
        };
        client.started = earlier;
        client.tick(&MoveInput::default());
        assert!(client.x13.set.is_empty(), "61 s is");
    }

    #[test]
    fn the_phase_is_built_on_first_use_draws_each_marker_and_goes_quiet_when_off() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let Ok(gpu) = Gpu::headless(64, 64, mc_render::GpuOptions::default()) else {
            eprintln!("SKIP: no wgpu adapter");
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        let mut camera = super::super::super::menu_camera();
        let state = |client: &Client| {
            client.x13.phase.as_ref().expect("installed").with(|phase| {
                (
                    phase.enabled(),
                    phase
                        .inner
                        .as_ref()
                        .map(|inner| (inner.vertex_counts().0, inner.glyph_vertex_counts().0)),
                )
            })
        };

        // Off: registered, nothing built, nothing recorded.
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(state(&client), (false, None));

        // On but no marker: still nothing built.
        client.mods.settings.set_enabled(MOD_ID, true);
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(state(&client), (false, None));

        // One marker 30 blocks ahead: backdrop, outline and pin (12) plus "30m" (18).
        let now = client.markers_now_ms();
        client.x13.set.place([0.5, 65.62, 30.5], now, 3);
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(state(&client), (true, Some((12, 18))));
        assert_eq!(client.x13.labels[0].text(), "30m");

        // F1 hides it.
        client.hide_gui = true;
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(state(&client), (true, Some((0, 0))));
        client.hide_gui = false;

        // Two markers: two pins.
        client.x13.set.place([10.5, 65.62, 30.5], now, 3);
        client.frame(&gpu, &mut camera, 0.0);
        let (enabled, counts) = state(&client);
        assert!(enabled);
        assert_eq!(counts.map(|(solid, _)| solid), Some(24));

        // Switching the mod off removes every marker and disables the phase.
        client.mods.settings.set_enabled(MOD_ID, false);
        client.frame(&gpu, &mut camera, 0.0);
        assert!(client.x13.set.is_empty());
        assert!(!state(&client).0);
        assert_eq!(phases.remove_named("markers/labels"), 1);
    }

    /// End to end through the installed stack (sky, fog, HUD, every phase): a real `Client`
    /// frame, rendered by its own `PhaseList` and read back. The marker's exact colour reaches
    /// the screen only while a marker is up.
    #[test]
    fn a_marker_reaches_the_screen_through_the_whole_client_frame() {
        const SIZE: u32 = 256;
        const COLOR: u32 = 0x12_EF34;
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let Ok(mut gpu) = Gpu::headless(
            SIZE,
            SIZE,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) else {
            eprintln!("SKIP: no wgpu adapter");
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5), 0.0, 0.0);
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE, SIZE);
        client.mods.settings.set_enabled(MOD_ID, true);
        client
            .mods
            .settings
            .set_color(MOD_ID, "color", 0xFF00_0000 | COLOR);
        client.mods.settings.set_f32(MOD_ID, "size", 200.0);

        let mut render = |client: &mut Client| {
            let mut camera = super::super::super::menu_camera();
            camera.set_viewport(SIZE, SIZE);
            client.frame(&gpu, &mut camera, 0.0);
            camera.update();
            let frame = gpu.acquire().expect("acquire an offscreen frame");
            phases.render(&gpu, frame.view(), &camera, 0.0);
            frame.present(&gpu.queue);
            let pixels = gpu.read_pixels().expect("read back");
            pixels
                .chunks(4)
                .filter(|p| p[..3] == [0x12, 0xEF, 0x34])
                .count()
        };

        assert_eq!(render(&mut client), 0, "no marker, no marker colour");
        // 30 blocks straight ahead (the player faces +Z).
        let now = client.markers_now_ms();
        client.x13.set.place([0.5, 65.62, 30.5], now, 3);
        let on = render(&mut client);
        assert!(on > 30, "the pin must reach the screen: {on} pixels");
        assert!(client.x13.set.clear());
        assert_eq!(render(&mut client), 0, "cleared, gone");
    }
}
