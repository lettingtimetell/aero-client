//! The client half of `anti_aliasing` and of `motion_blur`'s
//! per-pixel modes.
//!
//! Three render phases, registered by [`Client::install_x9_phases`] from `Client::install`:
//!
//! * `post/velocity` ([`VelocityPhase`]) and `post/aa` ([`AntiAliasPhase`]) at
//!   `PhaseOrder::TEMPORAL_AA` (643), in that order: the motion target, then TAA or FXAA;
//! * `post/velocity-blur` ([`VelocityBlurPhase`]) at `PhaseOrder::VELOCITY_BLUR` (644).
//!
//! Every frame ([`Client::x9_frame`], called where the camera is final) this file:
//!
//! * re-reads both mods when the settings generation moved;
//! * puts TAA's sub-pixel jitter on the camera (`Camera::jitter`), or zero;
//! * tells the phases whether a world is on screen, and resets TAA's history and the motion
//!   target's previous frame on a teleport (a new `Predictor::corrections`);
//! * hands the motion target the moving entities' boxes (TAA, and Camera + objects), plus one
//!   box per drawn label, measured by the nameplate pass ([`X9State::note_label`]): the label's text width, score line and the Nametags scale, your own F5 label too.
//!
//! `motion_blur`'s Frame blend mode stays in `m7_postfx.rs` (the M7 phase); that file passes it
//! no strength while a per-pixel mode is chosen.
//!
//! # Off
//!
//! Both mods off (or Frame blend): one `u64` compare and one `bool` test per frame, plus one
//! more `bool` while the camera still carries a jitter from the last TAA frame. The phases are
//! disabled, so `PhaseList` keeps its direct path, and the camera uniform is today's.

use std::collections::HashMap;

use mc_math::{DVec3, Vec3};
use mc_mods::mods::{anti_aliasing, motion_blur};
use mc_render::entity::nameplate::{
    Nameplate, NAMEPLATE_LIFT, NAMEPLATE_SCALE, NAMEPLATE_SCORE_GAP, NAMEPLATE_SCORE_RANGE_SQ,
    NAMEPLATE_SNEAK_DROP,
};
use mc_render::post::taa::{AaMode, AntiAliasPhase};
use mc_render::post::velocity::{ObjectMotion, VelocityConsumers, VelocityPhase, MAX_OBJECTS};
use mc_render::post::velocity_blur::VelocityBlurPhase;
use mc_render::{Camera, Gpu, PhaseList, PhaseOrder};
use mc_ui::font::FontRenderer;
use mc_ui::nameplate::NameplateRequest;

use super::super::{Client, Shared, Ui};

/// An entity slower than this (blocks per second) keeps the camera's motion.
const MOVING: f64 = 0.05;

/// Entities further than this (blocks) are left out: their motion is a pixel or less.
const OBJECT_RANGE: f64 = 96.0;

/// How far an entity's box grows past its hitbox sideways (blocks): arms and held items stick
/// out of the 0.6-wide box. Not downwards, or the ground under the feet would move with it.
const BOX_PAD: f32 = 0.3;

/// How far the box grows above the hitbox (blocks): enough for a short nametag, which is drawn
/// 0.5 above the head with depth and moves with the entity. Each drawn label also gets its own
/// box ([`LabelBox`]), as wide as its text.
const BOX_PAD_TOP: f32 = 0.9;

/// Slack around a label's box (blocks): the depth-rebuilt glyph surface sits a hair off its
/// plane.
const LABEL_PAD: f32 = 0.05;

/// The local player's hitbox (`EntityPlayer`): width and height in blocks.
const PLAYER_SIZE: (f32, f32) = (0.6, 1.8);

/// The local player's key in [`X9State::positions`] and for its own F5 label (network ids are
/// never this low).
pub(crate) const LOCAL_PLAYER_KEY: i32 = i32::MIN;

/// One drawn label's box around its entity's feet (blocks): `half` to each side on x and z,
/// `lo..hi` in height. A moving entity's label box carries the entity's motion, so TAA and
/// Camera + objects do not reproject the label as the wall behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LabelBox {
    half: f32,
    lo: f32,
    hi: f32,
}

/// The box a label covers when `mc_ui::nameplate::submit_cached` draws `plate` (its `rel` is
/// ignored: the box is around the feet). `name_px` and `score_px` are the measured string widths,
/// the score only when its line shows; `scale` is the Nametags scale.
///
/// Height: the rows `-1..=8` (plus the Nametags shadow row) below the anchor `height + 0.5`
/// (half a hitbox lower for a child), the name lifted over a score line, the deadmau5 lift, a
/// sneaking label's quarter-block drop. Sideways: half the widest line plus its backdrop, plus the
/// rows' reach, because the billboard tilts toward the camera with the pitch and turns with the
/// yaw; a square column covers every view.
pub(crate) fn label_box(
    plate: &Nameplate,
    name_px: i32,
    score_px: Option<i32>,
    scale: f32,
) -> LabelBox {
    // Blocks per label pixel. java: `renderLivingLabel`'s `scale(-f1, -f1, f1)`; label +y is down.
    let px = NAMEPLATE_SCALE * scale;
    let child_drop = if plate.child { plate.height / 2.0 } else { 0.0 };
    let anchor = plate.height - child_drop + NAMEPLATE_LIFT;
    // `Nameplate::layout`: deadmau5's rows sit 10 pixels higher (not while sneaking).
    let lift = if plate.text == "deadmau5" && !plate.sneaking {
        10.0
    } else {
        0.0
    };
    // `submit_cached`: a score line keeps the anchor and pushes the name up by the scaled gap.
    let gap = if score_px.is_some() {
        NAMEPLATE_SCORE_GAP * scale
    } else {
        0.0
    };
    let hi = anchor + gap + (1.0 + lift) * px;
    // The rows tilt toward the camera with the pitch, pivoting on their line's anchor, so a line
    // spans from its anchor to its level position (deadmau5's rows all sit above the anchor).
    let mut lo = (anchor + gap - (9.0 - lift) * px).min(anchor + gap);
    if score_px.is_some() {
        lo = lo.min(anchor - 9.0 * px);
    }
    // `Nameplate::matrix`: the sneak drop is label pixels after the rotation, so at most this far
    // down (and sideways once tilted); unscaled by the Nametags scale.
    let sneak = if plate.sneaking {
        NAMEPLATE_SNEAK_DROP * NAMEPLATE_SCALE
    } else {
        0.0
    };
    // java: the backdrop spans `-j - 1 .. j + 1` with `j = width / 2`; one more for the shadow.
    let widest = name_px.max(score_px.unwrap_or(0)).max(0);
    let half_px = (widest / 2) as f32 + 2.0;
    let reach_px = 9.0f32.max(1.0 + lift);
    LabelBox {
        half: (half_px + reach_px) * px + sneak + LABEL_PAD,
        lo: lo - sneak - LABEL_PAD,
        hi: hi + LABEL_PAD,
    }
}

/// A render position that moved further than this in one frame (blocks) teleported: no motion.
const ENTITY_JUMP: f64 = 8.0;

/// The three phases, shared with the `PhaseList`.
pub(crate) struct X9Phases {
    velocity: Shared<VelocityPhase>,
    aa: Shared<AntiAliasPhase>,
    blur: Shared<VelocityBlurPhase>,
}

/// What the client last pushed into the phases.
#[derive(Default)]
pub(crate) struct X9State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    aa_mode: AaMode,
    /// The per-pixel blur runs in Camera + objects mode: the motion target needs the entities.
    blur_objects: bool,
    /// Something of this lane can draw: the per-frame work is worth doing.
    active: bool,
    /// What the phases were last told about a world being on screen.
    in_world: bool,
    /// `Predictor::corrections` last frame: a change is a teleport.
    corrections: u64,
    /// The camera carries a non-zero jitter that must be cleared once TAA stops.
    jittering: bool,
    /// Moving entities, reused every frame.
    objects: Vec<ObjectMotion>,
    /// Every nearby entity's render position on the previous frame (network id; the local
    /// player is [`LOCAL_PLAYER_KEY`]), for exact per-frame displacements.
    positions: HashMap<i32, DVec3>,
    /// This frame's positions while they are collected; swapped with `positions` after.
    next_positions: HashMap<i32, DVec3>,
    /// The nameplate pass reports its labels ([`Self::note_label`]): set by the last frame's
    /// [`Client::x9_frame`] while TAA or Camera + objects runs in a world.
    wants_labels: bool,
    /// This frame's drawn labels (same keys as `positions`), filled by the nameplate pass,
    /// which runs before [`Client::x9_frame`], and emptied there.
    labels: HashMap<i32, LabelBox>,
    phases: Option<X9Phases>,
}

impl X9State {
    /// Whether the nameplate pass should call [`Self::note_label`] this frame. Read once per
    /// frame by the caller; `false` while TAA and Camera + objects are off.
    pub(crate) fn wants_labels(&self) -> bool {
        self.wants_labels
    }

    /// The nameplate pass drew (or skipped as out of range) `req` for the entity `key`: a network
    /// id, or [`LOCAL_PLAYER_KEY`] for your own F5 label. Measures it (two allocation-free
    /// string-width walks) and keeps its box for this frame's object pass.
    pub(crate) fn note_label(
        &mut self,
        key: i32,
        font: &FontRenderer,
        req: &NameplateRequest<'_>,
        scale: f32,
    ) {
        let plate = req.plate;
        if !plate.in_range(req.distance_sq) {
            return;
        }
        // The same condition `submit_cached` draws the score line under.
        let score_px = req
            .score
            .filter(|_| !plate.sneaking && req.distance_sq < NAMEPLATE_SCORE_RANGE_SQ)
            .map(|score| font.string_width(score));
        let label = label_box(plate, font.string_width(&plate.text), score_px, scale);
        self.labels.insert(key, label);
    }
}

impl Client {
    /// Register the three phases. Runs from `Client::install`, so also after a resource-pack
    /// reload, which rebuilds every phase; the new phases are synced on the next frame.
    pub(crate) fn install_x9_phases(&mut self, gpu: &Gpu, phases: &mut PhaseList) {
        let velocity = VelocityPhase::new(gpu);
        let output = velocity.output();
        let mut aa = AntiAliasPhase::new(gpu);
        aa.set_velocity(output.clone());
        let mut blur = VelocityBlurPhase::new(gpu);
        blur.set_velocity(output);
        let velocity = Shared::new(velocity);
        let aa = Shared::new(aa);
        let blur = Shared::new(blur);
        // Stable insertion: the motion target records before the resolve that reads it.
        phases.insert(PhaseOrder::TEMPORAL_AA, Box::new(velocity.clone()));
        phases.insert(PhaseOrder::TEMPORAL_AA, Box::new(aa.clone()));
        phases.insert(PhaseOrder::VELOCITY_BLUR, Box::new(blur.clone()));
        self.x9 = X9State {
            phases: Some(X9Phases { velocity, aa, blur }),
            // The camera's jitter, if any, is still cleared by the next frame.
            jittering: self.x9.jittering,
            ..X9State::default()
        };
    }

    /// The per-frame hook (module docs). `camera` is the one this frame renders with.
    pub(crate) fn x9_frame(&mut self, gpu: &Gpu, camera: &mut Camera, partial_ticks: f32) {
        let generation = self.mods.settings.generation();
        if self.x9.seen != generation {
            self.x9.seen = generation;
            self.sync_x9_settings();
        }
        if !self.x9.active {
            if self.x9.jittering {
                camera.jitter = Default::default();
                self.x9.jittering = false;
            }
            return;
        }
        let Some(phases) = self.x9.phases.as_ref() else {
            return;
        };
        let in_world = self.x9_world_on_screen();
        if in_world != self.x9.in_world {
            self.x9.in_world = in_world;
            self.x9.positions.clear();
            phases.velocity.with(|p| p.set_in_world(in_world));
            phases.aa.with(|p| p.set_in_world(in_world));
            phases.blur.with(|p| p.set_in_world(in_world));
        }
        let corrections = self.corrections();
        if corrections != self.x9.corrections {
            // An S08 teleport (or a world's first position): nothing on screen continues.
            self.x9.corrections = corrections;
            self.x9.positions.clear();
            phases.velocity.with(VelocityPhase::reset_history);
            phases.aa.with(AntiAliasPhase::reset_history);
        }
        let (width, height) = (gpu.config.width, gpu.config.height);
        camera.jitter = phases.aa.with(|p| p.next_jitter(width, height));
        self.x9.jittering = camera.jitter != Default::default();

        let wants_objects = in_world && (self.x9.aa_mode == AaMode::Taa || self.x9.blur_objects);
        // The next frame's nameplate pass measures its labels.
        self.x9.wants_labels = wants_objects;
        let velocity = phases.velocity.clone();
        if wants_objects {
            let mut objects = std::mem::take(&mut self.x9.objects);
            let previous = std::mem::take(&mut self.x9.positions);
            let mut current = std::mem::take(&mut self.x9.next_positions);
            current.clear();
            self.collect_moving_entities(
                camera,
                partial_ticks,
                &previous,
                &mut current,
                &mut objects,
            );
            velocity.with(|p| p.set_objects(&objects));
            self.x9.objects = objects;
            // This frame's positions become the previous ones; the old map's storage is reused.
            self.x9.positions = current;
            self.x9.next_positions = previous;
            self.x9.labels.clear();
        } else if !self.x9.objects.is_empty()
            || !self.x9.positions.is_empty()
            || !self.x9.labels.is_empty()
        {
            self.x9.objects.clear();
            self.x9.positions.clear();
            self.x9.labels.clear();
            velocity.with(|p| p.set_objects(&[]));
        }
    }

    /// Read both mods (only when the settings generation moved) and push them.
    fn sync_x9_settings(&mut self) {
        let settings = &self.mods.settings;
        let aa_mode = if settings.enabled("anti_aliasing") {
            match settings.choice("anti_aliasing", "mode") {
                anti_aliasing::MODE_FXAA => AaMode::Fxaa,
                _ => AaMode::Taa,
            }
        } else {
            AaMode::Off
        };
        let sharpness = settings.f32("anti_aliasing", "sharpness") / 100.0;
        let texture_edges = settings.bool("anti_aliasing", "texture_edges");
        let blur_mode = if settings.enabled("motion_blur") {
            settings.choice("motion_blur", "mode")
        } else {
            motion_blur::MODE_FRAME_BLEND
        };
        let strength = settings.f32("motion_blur", "strength") / 100.0;
        let exposure = motion_blur::exposure_seconds(strength);
        let blur_on = blur_mode != motion_blur::MODE_FRAME_BLEND && exposure > 0.0;
        let blur_objects = blur_on && blur_mode == motion_blur::MODE_CAMERA_OBJECTS;

        self.x9.aa_mode = aa_mode;
        self.x9.blur_objects = blur_objects;
        self.x9.active = aa_mode != AaMode::Off || blur_on;
        let Some(phases) = self.x9.phases.as_ref() else {
            return;
        };
        phases.aa.with(|p| {
            p.set_mode(aa_mode);
            p.set_sharpness(sharpness);
            p.set_texture_edges(texture_edges);
        });
        phases.velocity.with(|p| {
            p.set_consumers(VelocityConsumers {
                taa: aa_mode == AaMode::Taa,
                blur: blur_on,
                blur_objects,
            });
            p.set_exposure(exposure);
        });
        phases.blur.with(|p| p.set_on(blur_on));
        if !self.x9.active {
            // Everything off: forget the world state so turning a mod on resyncs it.
            self.x9.in_world = false;
            self.x9.wants_labels = false;
            self.x9.labels.clear();
            phases.velocity.with(|p| p.set_in_world(false));
            phases.aa.with(|p| p.set_in_world(false));
            phases.blur.with(|p| p.set_in_world(false));
        }
    }

    /// Whether a world is on screen: a session whose player exists and is past the
    /// downloading-terrain screen. The main menu, its panorama and loading screens are never
    /// anti-aliased or blurred here (same test as the M7 phase).
    fn x9_world_on_screen(&self) -> bool {
        !matches!(self.ui, Ui::Loading)
            && self.session.as_ref().is_some_and(|session| {
                session.predictor().is_some() && !session.terrain_download_active()
            })
    }

    /// Every entity that moves on screen this frame, as a camera-relative box plus its
    /// displacement since the previous frame (from `previous`, this frame's positions go into
    /// `current`) and its render velocity, for the motion target's object pass, and a second box
    /// with the same motion over its label when the nameplate pass drew one
    /// ([`X9State::note_label`]). The local player is included in third person, where its body
    /// moves with the camera. The entity a spectator looks out of in first person is left out:
    /// it is not drawn, and its box would hold the camera.
    fn collect_moving_entities(
        &self,
        camera: &Camera,
        partial_ticks: f32,
        previous: &HashMap<i32, DVec3>,
        current: &mut HashMap<i32, DVec3>,
        out: &mut Vec<ObjectMotion>,
    ) {
        out.clear();
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let state = session.state();
        let world = &state.world;
        let labels = &self.x9.labels;
        // `render_position(pt)` lerps last tick to this tick, and `pt` advances 20 per second.
        let ticks_per_second = f64::from(mc_render::TICKS_PER_SECOND);
        let mut push = |key: i32, feet: DVec3, velocity: DVec3, size: (f32, f32)| {
            let rel = camera.relative(feet);
            if f64::from(rel.length()) > OBJECT_RANGE {
                return;
            }
            current.insert(key, feet);
            let moved = previous
                .get(&key)
                .map(|before| feet - *before)
                .filter(|moved| moved.length() < ENTITY_JUMP)
                .unwrap_or(DVec3::ZERO);
            let still = moved.length() < 1e-6 && velocity.length() < MOVING;
            if still || out.len() + 2 > MAX_OBJECTS {
                return;
            }
            let (frame_delta, velocity) = (moved.as_vec3(), velocity.as_vec3());
            let half = size.0 * 0.5 + BOX_PAD;
            out.push(ObjectMotion {
                min: rel + Vec3::new(-half, 0.0, -half),
                max: rel + Vec3::new(half, size.1 + BOX_PAD_TOP, half),
                frame_delta,
                velocity,
            });
            // The label the nameplate pass drew for this entity, if any.
            if let Some(label) = labels.get(&key) {
                out.push(ObjectMotion {
                    min: rel + Vec3::new(-label.half, label.lo, -label.half),
                    max: rel + Vec3::new(label.half, label.hi, label.half),
                    frame_delta,
                    velocity,
                });
            }
        };
        // X6's first-person spectating view: that entity is not drawn.
        let hidden = super::x6_entities::first_person_view_entity(session, self.third_person_view);
        for (id, tracked) in state.entities.iter() {
            if hidden == Some(id) {
                continue;
            }
            let now = tracked.render_position(world, partial_ticks);
            let velocity = (tracked.render_position(world, 1.0)
                - tracked.render_position(world, 0.0))
                * ticks_per_second;
            push(id, now, velocity, tracked.dimensions());
        }
        if self.third_person_view != 0 && session.render_view_is_local() {
            if let Some(predictor) = session.predictor() {
                let prev = self
                    .camera_prev_pos
                    .filter(|_| self.camera_prev_corrections == predictor.corrections);
                let pos = predictor.state.pos;
                let start = prev.unwrap_or(pos);
                let feet = start + (pos - start) * f64::from(partial_ticks);
                push(
                    LOCAL_PLAYER_KEY,
                    feet,
                    (pos - start) * ticks_per_second,
                    PLAYER_SIZE,
                );
            }
        }
    }

    // A video export renders at fixed
    // timesteps, far from real time, so the motion target's blur exposure must use the export's
    // frame time instead of the wall clock (`None` puts the wall clock back).
    pub(crate) fn x9_set_fixed_dt(&mut self, dt: Option<f32>) {
        if let Some(phases) = self.x9.phases.as_ref() {
            phases.velocity.with(|phase| phase.set_fixed_dt(dt));
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
    use tokio::sync::mpsc::{Sender, UnboundedReceiver};

    /// A client on the real assets, or `None` (skip) without the vanilla jar.
    pub(crate) fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// JoinGame, then the first S08 closes the downloading-terrain screen.
    pub(crate) fn join_world(client: &mut Client) -> (Sender<ToGame>, UnboundedReceiver<ToNet>) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(Session::new(handle, JoinSettings::default()));
        to_game
            .try_send(ToGame::Packet(mc_protocol::play::S2CPlay::JoinGame(
                mc_protocol::play::s2c::JoinGame {
                    entity_id: 27,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                },
            )))
            .expect("join channel");
        client.poll();
        teleport(&to_game, 8.5);
        client.poll();
        (to_game, to_net)
    }

    fn teleport(to_game: &Sender<ToGame>, x: f64) {
        to_game
            .try_send(ToGame::Packet(
                mc_protocol::play::S2CPlay::PlayerPositionLook(
                    mc_protocol::play::s2c::PlayerPositionLook {
                        x,
                        y: 70.0,
                        z: 8.5,
                        yaw: 0.0,
                        pitch: 0.0,
                        flags: 0,
                    },
                ),
            ))
            .expect("position channel");
    }

    fn phases(client: &Client) -> &X9Phases {
        client.x9.phases.as_ref().expect("installed")
    }

    /// Settings reach the phases; the jitter is on the camera only while TAA runs in a world
    /// and is cleared (to exactly zero) the frame it stops; with both mods off nothing runs.
    #[test]
    fn settings_reach_the_phases_and_the_jitter_follows_taa() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let gpu = match Gpu::headless(64, 64, mc_render::GpuOptions::default()) {
            Ok(gpu) => gpu,
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter");
                return;
            }
            Err(error) => panic!("headless context: {error}"),
        };
        let mut list = PhaseList::new();
        client.install_x9_phases(&gpu, &mut list);
        let names: Vec<&str> = list.names().collect();
        assert_eq!(
            names,
            [
                VelocityPhase::NAME,
                AntiAliasPhase::NAME,
                VelocityBlurPhase::NAME
            ]
        );
        let mut camera = Camera::default();
        let zero = camera.jitter;

        // Defaults: both mods off. Nothing runs, no jitter.
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert!(!list.scene_mode());
        assert_eq!(camera.jitter, zero);

        // Anti-aliasing on (TAA by default) in the main menu: no world, no jitter.
        client.mods.settings.set_enabled("anti_aliasing", true);
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert_eq!(phases(&client).aa.with(|p| p.mode()), AaMode::Taa);
        assert!(!list.scene_mode(), "the main menu is never anti-aliased");
        assert_eq!(camera.jitter, zero);

        // In a world: the phases run and the camera is jittered by under half a pixel.
        let (to_game, _to_net) = join_world(&mut client);
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert!(client.x9_world_on_screen());
        assert!(list.scene_mode());
        assert_ne!(camera.jitter, zero);
        assert!(camera.jitter.abs().max_element() < 1.0 / 64.0);
        assert!(phases(&client).velocity.with(|p| p.consumers().taa));

        // A teleport resets the history (the counter moves).
        let before = client.x9.corrections;
        teleport(&to_game, 40.5);
        client.poll();
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert_ne!(client.x9.corrections, before);

        // FXAA: no jitter, still a pass; the motion target is not needed.
        client
            .mods
            .settings
            .set_choice("anti_aliasing", "mode", anti_aliasing::MODE_FXAA);
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert_eq!(camera.jitter, zero);
        assert!(list.scene_mode());
        assert!(!phases(&client).velocity.with(|p| p.consumers().any()));

        // Motion blur, Camera + objects: the motion target runs for the blur.
        client.mods.settings.set_enabled("anti_aliasing", false);
        client.mods.settings.set_enabled("motion_blur", true);
        client
            .mods
            .settings
            .set_choice("motion_blur", "mode", motion_blur::MODE_CAMERA_OBJECTS);
        client.x9_frame(&gpu, &mut camera, 0.0);
        let consumers = phases(&client).velocity.with(|p| p.consumers());
        assert!(consumers.blur && consumers.blur_objects && !consumers.taa);
        assert!(phases(&client).blur.with(|p| p.is_active()));

        // Frame blend is the M7 phase's: nothing of this lane runs.
        client
            .mods
            .settings
            .set_choice("motion_blur", "mode", motion_blur::MODE_FRAME_BLEND);
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert!(!list.scene_mode());

        // TAA again, then everything off: the jitter goes back to exactly zero.
        client.mods.settings.set_enabled("motion_blur", false);
        client.mods.settings.set_enabled("anti_aliasing", true);
        client
            .mods
            .settings
            .set_choice("anti_aliasing", "mode", anti_aliasing::MODE_TAA);
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert_ne!(camera.jitter, zero);
        client.mods.settings.set_enabled("anti_aliasing", false);
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert_eq!(camera.jitter, zero);
        assert!(!list.scene_mode());
        assert!(!phases(&client).aa.with(|p| p.history_allocated()));
    }

    /// X15 (review M1): every corner of a label, as `submit_cached` places it (backdrop plus the
    /// shadow pixel, the score line and the name lifted over it), lies in [`label_box`] at any
    /// view, scale, sneak, child and deadmau5 state; a long name gets a wide box and the Nametags
    /// scale scales it.
    #[test]
    fn a_label_box_holds_every_corner_of_its_label() {
        let corners = |plate: &Nameplate, width: i32, yaw: f32, pitch: f32, scale: f32| {
            let layout = plate.layout(width);
            let b = layout.backdrop;
            let m = plate.matrix_scaled(yaw, pitch, scale);
            // The Nametags shadow reaches one pixel right and down of the glyphs.
            [
                (b.x0, b.y0),
                (b.x1 + 1.0, b.y0),
                (b.x0, b.y1 + 1.0),
                (b.x1 + 1.0, b.y1 + 1.0),
            ]
            .map(|(x, y)| m.transform_point3(Vec3::new(x, y, 0.0)))
        };
        let mut checked = 0;
        for (text, width) in [
            ("Steve", 30),
            ("deadmau5", 46),
            ("[MVP++] A_Long_Name", 112),
        ] {
            for scale in [0.5, 1.0, 2.0] {
                for (sneaking, child, score) in [
                    (false, false, None),
                    (false, false, Some(70)),
                    (false, false, Some(150)),
                    (true, false, None),
                    (false, true, None),
                ] {
                    let name = Nameplate {
                        text: text.to_owned(),
                        rel: Vec3::ZERO,
                        height: 1.8,
                        child,
                        sneaking,
                    };
                    let label = label_box(&name, width, score, scale);
                    for yaw in [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 250.0] {
                        for pitch in [-90.0, -60.0, -20.0, 0.0, 20.0, 60.0, 90.0] {
                            let mut points = Vec::new();
                            match score {
                                Some(score_width) => {
                                    let line = Nameplate {
                                        text: "7 kills".to_owned(),
                                        ..name.clone()
                                    };
                                    points.extend(corners(&line, score_width, yaw, pitch, scale));
                                    let lifted = Nameplate {
                                        rel: Vec3::new(0.0, NAMEPLATE_SCORE_GAP * scale, 0.0),
                                        ..name.clone()
                                    };
                                    points.extend(corners(&lifted, width, yaw, pitch, scale));
                                }
                                None => points.extend(corners(&name, width, yaw, pitch, scale)),
                            }
                            for p in points {
                                assert!(
                                    p.x.abs() <= label.half
                                        && p.z.abs() <= label.half
                                        && (label.lo..=label.hi).contains(&p.y),
                                    "{text} x{scale} sneak {sneaking} child {child} score \
                                     {score:?} yaw {yaw} pitch {pitch}: {p:?} outside {label:?}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 5000);

        // Not loose: a short vanilla name stays within a block of the anchor and under a block
        // wide; a long one is wider than X9's old fixed 1-block half width, and twice the scale
        // is twice the box (less the fixed slack).
        let plate = |text: &str| Nameplate {
            text: text.to_owned(),
            rel: Vec3::ZERO,
            height: 1.8,
            child: false,
            sneaking: false,
        };
        let short = label_box(&plate("Steve"), 30, None, 1.0);
        assert!(short.half < 0.75 && short.hi - short.lo < 0.4, "{short:?}");
        let long = label_box(&plate("[MVP++] A_Long_Name"), 112, None, 1.0);
        assert!(long.half > 1.0, "{long:?}");
        let double = label_box(&plate("[MVP++] A_Long_Name"), 112, None, 2.0);
        assert!(((double.half - LABEL_PAD) - 2.0 * (long.half - LABEL_PAD)).abs() < 1e-4);
    }

    /// X15 (review M1), end to end without a GPU: a moving zombie with a measured long label
    /// gets its body box and a label box with the same motion; your own F5 label gets one too;
    /// the entity a spectator looks out of in first person gets none.
    #[test]
    fn moving_labels_get_boxes_and_the_spectated_entity_does_not() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let (to_game, _to_net) = join_world(&mut client);
        // A loaded column, or the tracker parks the zombie (not iterated, not spectatable).
        client
            .session
            .as_mut()
            .expect("joined")
            .state_mut()
            .world
            .load_empty_chunk(mc_math::ChunkPos::new(0, 0));
        to_game
            .try_send(ToGame::Packet(mc_protocol::play::S2CPlay::SpawnMob(
                mc_protocol::play::s2c::SpawnMob {
                    entity_id: 42,
                    entity_type: 54,
                    x: 8 * 32 + 16,
                    y: 70 * 32,
                    z: 12 * 32 + 16,
                    yaw: mc_protocol::Angle(0),
                    pitch: mc_protocol::Angle(0),
                    head_pitch: mc_protocol::Angle(0),
                    velocity_x: 0,
                    velocity_y: 0,
                    velocity_z: 0,
                    metadata: Vec::new(),
                },
            )))
            .expect("spawn channel");
        client.poll();
        let session = client.session.as_ref().expect("joined");
        let feet = session
            .state()
            .entities
            .get(42)
            .expect("the zombie spawned")
            .render_position(&session.state().world, 0.0);
        let local_pos = session.predictor().expect("a local player").state.pos;
        let corrections = session.predictor().expect("a local player").corrections;
        let camera = Camera {
            pos: local_pos + DVec3::new(0.0, 1.62, -4.0),
            ..Camera::default()
        };

        // The nameplate pass measures the zombie's label and your own.
        let name = Nameplate {
            text: "\u{a7}6[MVP\u{a7}c++\u{a7}6] A_Rather_Long_Name".to_owned(),
            rel: Vec3::ZERO,
            height: 1.95,
            child: false,
            sneaking: false,
        };
        fn request(plate: &Nameplate) -> NameplateRequest<'_> {
            NameplateRequest {
                plate,
                view_yaw: 0.0,
                view_pitch: 0.0,
                distance_sq: 16.0,
                score: None,
            }
        }
        client.x9.note_label(42, &client.font, &request(&name), 1.0);
        let own = Nameplate {
            text: "Dev".to_owned(),
            height: 1.8,
            ..name.clone()
        };
        client
            .x9
            .note_label(LOCAL_PLAYER_KEY, &client.font, &request(&own), 2.0);
        let zombie_label = client.x9.labels[&42];
        let own_label = client.x9.labels[&LOCAL_PLAYER_KEY];
        assert!(zombie_label.half > 1.0, "{zombie_label:?}");

        // The zombie moved half a block since the previous frame; you walk in F5.
        let previous: HashMap<i32, DVec3> = [(42, feet - DVec3::new(0.5, 0.0, 0.0))].into();
        let mut current = HashMap::new();
        let mut out = Vec::new();
        client.third_person_view = 1;
        client.camera_prev_pos = Some(local_pos - DVec3::new(0.3, 0.0, 0.0));
        client.camera_prev_corrections = corrections;
        client.collect_moving_entities(&camera, 1.0, &previous, &mut current, &mut out);
        assert_eq!(out.len(), 4, "zombie body + label, your body + label");
        let rel = camera.relative(feet);
        let close = |a: Vec3, b: Vec3| (a - b).length() < 1e-4;
        assert!(close(
            out[1].min,
            rel + Vec3::new(-zombie_label.half, zombie_label.lo, -zombie_label.half)
        ));
        assert!(close(
            out[1].max,
            rel + Vec3::new(zombie_label.half, zombie_label.hi, zombie_label.half)
        ));
        assert_eq!(out[1].frame_delta, out[0].frame_delta);
        assert!(close(out[0].frame_delta, Vec3::new(0.5, 0.0, 0.0)));
        let own_rel = camera.relative(local_pos);
        assert!(close(
            out[3].max,
            own_rel + Vec3::new(own_label.half, own_label.hi, own_label.half)
        ));
        assert_eq!(out[3].frame_delta, out[2].frame_delta);

        // Spectating the zombie in first person: no box at all (it is not drawn, and its box
        // would hold the camera); in F5 it is drawn and moves again.
        client.third_person_view = 0;
        client
            .session
            .as_mut()
            .expect("joined")
            .set_render_view_entity(42);
        client.collect_moving_entities(&camera, 1.0, &previous, &mut current, &mut out);
        assert!(out.is_empty(), "{out:?}");
        client.third_person_view = 1;
        client.collect_moving_entities(&camera, 1.0, &previous, &mut current, &mut out);
        assert_eq!(out.len(), 2, "the spectated zombie in F5: body + label");
    }

    /// X15 (review M1): the live nameplate pass (`upload_entities`) reports your own F5 label
    /// while TAA runs, not while it is off, and `x9_frame` consumes the reports every frame.
    #[test]
    fn the_nameplate_pass_reports_labels_only_while_taa_runs() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let gpu = match Gpu::headless(64, 64, mc_render::GpuOptions::default()) {
            Ok(gpu) => gpu,
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter");
                return;
            }
            Err(error) => panic!("headless context: {error}"),
        };
        let mut list = PhaseList::new();
        client.install(&gpu, &mut list, 64, 64);
        let (_to_game, _to_net) = join_world(&mut client);
        client
            .session
            .as_mut()
            .expect("joined")
            .state_mut()
            .local
            .username = "Dev".to_owned();
        client.mods.settings.set_enabled("nametags", true);
        client.x6_sync();
        client.third_person_view = 1;
        let mut camera = Camera::default();

        client.x9_frame(&gpu, &mut camera, 0.0);
        assert!(!client.x9.wants_labels(), "TAA off: the pass is not asked");
        client.upload_entities(&gpu, &camera, 0.0);
        assert!(client.x9.labels.is_empty());

        client.mods.settings.set_enabled("anti_aliasing", true);
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert!(client.x9.wants_labels());
        client.upload_entities(&gpu, &camera, 0.0);
        assert!(
            client.x9.labels.contains_key(&LOCAL_PLAYER_KEY),
            "your own F5 label is measured"
        );
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert!(client.x9.labels.is_empty(), "consumed every frame");

        client.mods.settings.set_enabled("anti_aliasing", false);
        client.x9_frame(&gpu, &mut camera, 0.0);
        assert!(!client.x9.wants_labels(), "off again: not asked");
    }
}
