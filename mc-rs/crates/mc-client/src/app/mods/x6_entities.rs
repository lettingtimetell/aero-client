//! Entity rendering mods and fixes.
//!
//! * **Stuck Arrows** (`stuck_arrows`, PvP, on by default): [`StuckArrows::visible`] gates
//!   `LayerArrow` for the remote-player and local F5 bodies (`upload_entities`).
//! * **Nametags** (`nametags`, Players): a [`NameplateStyle`] for every entity label (pushed to
//!   `mc_ui::nameplate::NameplateCache` and `NameplatePhase` when settings change) and your own
//!   label in third person ([`Client::x6_own_nametag`]). Aero adds "Always
//!   show mob names" ([`X6State::always_mob_names`]; the rest is `super::w18_fb`).
//! * **Hitboxes** (`hitboxes`, PvP): wide-line boxes in `mc_render::entity::hitbox::HitboxPhase`,
//!   replacing vanilla's thin F3+B lines while the mod is on ([`Client::x6_frame`]).
//! * **Spectating** (item 35): [`first_person_view_entity`], the entity a spectator looks out of,
//!   which vanilla does not draw in first person.
//!
//! Off, every hook costs one `u64` generation compare (in [`Client::x6_sync`]) plus a `bool`.

use mc_math::{DVec3, Vec3};
use mc_mods::ModSettings;
use mc_render::entity::hitbox::{HitboxPhase, HITBOX_PHASE_ORDER};
use mc_render::entity::nameplate::NameplateStyle;
use mc_render::entity::NameplatePhase;
use mc_render::terrain::frustum::Frustum;
use mc_render::{Camera, Gpu, PhaseList};

use super::super::{Client, Shared};
use crate::session::{RenderViewTarget, Session};
use crate::sync::entities::{TrackedEntity, TrackedKind};

/// The render-view entity a spectator looks out of, while the view is first person.
///
/// java: `RenderGlobal.renderEntities` draws an entity only when `entity2 != renderViewEntity ||
/// thirdPersonView != 0 || renderViewEntity.isPlayerSleeping()`, for the bodies and for the
/// spectator outline pass alike. `None` when the view is the local player's own, in third person,
/// or the spectated player is asleep. (`isPlayerSleeping` is `false` for every non-player.)
pub(crate) fn first_person_view_entity(session: &Session, third_person_view: u8) -> Option<i32> {
    if third_person_view != 0 {
        return None;
    }
    let RenderViewTarget::Entity { network_id, .. } = session.render_view_target() else {
        return None;
    };
    let tracked = session.state().entities.get_loaded(network_id)?;
    if tracked.kind == TrackedKind::Player && tracked.sleeping {
        return None;
    }
    Some(network_id)
}

/// Stuck Arrows' snapshot. The default matches the catalog: visible players only.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StuckArrows {
    shown: bool,
    on_invisible: bool,
}

impl Default for StuckArrows {
    fn default() -> Self {
        Self {
            shown: true,
            on_invisible: false,
        }
    }
}

impl StuckArrows {
    /// Whether `LayerArrow` draws for a player whose invisible flag is `invisible`.
    ///
    /// Off hides all arrows. "Only invisible players" selects invisible players instead of
    /// visible ones; disabling that option shows arrows on visible players only.
    pub(crate) fn visible(self, invisible: bool) -> bool {
        self.shown && (self.on_invisible == invisible)
    }
}

/// Nametags' snapshot: the label style, and whether to label yourself in F5.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Nametags {
    /// Show your own label in third person.
    own: bool,
    /// [`NameplateStyle::VANILLA`] while the mod is off.
    style: NameplateStyle,
    /// "Always show mob names": a named mob is labelled without the
    /// crosshair test (`super::w18_fb`). `false` while the mod is off.
    mob_names: bool,
}

/// The entity groups the Hitboxes mod can show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HitboxGroup {
    Player,
    Mob,
    Item,
    Projectile,
    Other,
}

impl HitboxGroup {
    /// Which group a tracked entity belongs to.
    pub(crate) fn of(tracked: &TrackedEntity) -> Self {
        use mc_data::EntityType as E;
        match tracked.kind {
            TrackedKind::Player => return Self::Player,
            TrackedKind::FishingBobber => return Self::Projectile,
            _ => {}
        }
        match tracked.kind.entity_type() {
            Some(E::Item | E::XPOrb) => Self::Item,
            Some(
                E::Arrow
                | E::Snowball
                | E::ThrownEgg
                | E::ThrownEnderpearl
                | E::ThrownPotion
                | E::ThrownExpBottle
                | E::Fireball
                | E::SmallFireball
                | E::WitherSkull
                | E::EyeOfEnderSignal
                | E::FireworksRocketEntity,
            ) => Self::Projectile,
            Some(kind) if kind.info().living => Self::Mob,
            _ => Self::Other,
        }
    }

    /// Living entities get the eye-height line (`instanceof EntityLivingBase`).
    fn living(self) -> bool {
        matches!(self, Self::Player | Self::Mob)
    }
}

/// Hitboxes' snapshot.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Hitboxes {
    enabled: bool,
    always: bool,
    players: bool,
    mobs: bool,
    items: bool,
    projectiles: bool,
    other: bool,
    color: [f32; 4],
    width: f32,
    eye: bool,
    look: bool,
}

/// Vanilla F3+B's eye-height colour (`255, 0, 0`).
const EYE_COLOR: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
/// Vanilla F3+B's look-vector colour (`0, 0, 255`).
const LOOK_COLOR: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
/// Boxes further than this from the camera are not drawn (vanilla draws them for every rendered
/// entity; 128 blocks covers the entity render range at any server view distance).
const HITBOX_RANGE_SQ: f64 = 128.0 * 128.0;

impl Hitboxes {
    /// While on, the mod replaces vanilla's thin F3+B lines.
    pub(crate) fn replaces_f3b(self) -> bool {
        self.enabled
    }

    /// Whether boxes are drawn this frame, given F3+B's toggle.
    pub(crate) fn active(self, debug_bounding_box: bool) -> bool {
        self.enabled && (self.always || debug_bounding_box)
    }

    fn shows(self, group: HitboxGroup) -> bool {
        match group {
            HitboxGroup::Player => self.players,
            HitboxGroup::Mob => self.mobs,
            HitboxGroup::Item => self.items,
            HitboxGroup::Projectile => self.projectiles,
            HitboxGroup::Other => self.other,
        }
    }

    /// One entity's box, eye line and look vector. `feet` is camera-relative.
    fn push(
        self,
        phase: &mut HitboxPhase,
        feet: Vec3,
        size: (f32, f32),
        eye: Option<f32>,
        look: Vec3,
    ) {
        let half = size.0 * 0.5;
        let min = Vec3::new(feet.x - half, feet.y, feet.z - half);
        let max = Vec3::new(feet.x + half, feet.y + size.1, feet.z + half);
        phase.push_box(min, max, self.width, self.color);
        if let Some(eye) = eye {
            if self.eye {
                phase.push_rect(min, max, feet.y + eye, self.width, EYE_COLOR);
            }
        }
        if self.look {
            let eye_height = eye.unwrap_or(size.1 * 0.85);
            let from = Vec3::new(feet.x, feet.y + eye_height, feet.z);
            phase.push_line(from, from + look * 2.0, self.width, LOOK_COLOR);
        }
    }
}

/// Everything this lane keeps on the client (`Client::x6`).
#[derive(Default)]
pub(crate) struct X6State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    pub(crate) arrows: StuckArrows,
    nametags: Nametags,
    pub(crate) hitboxes: Hitboxes,
    /// The hitbox phase, shared with the `PhaseList`. `None` until `Client::install`.
    hitbox_phase: Option<Shared<HitboxPhase>>,
    /// The phase holds this frame's (or a previous frame's) segments.
    hitboxes_drawn: bool,
    /// This frame's forced mob labels and holograms (used only while
    /// "Always show mob names" is on).
    pub(crate) mob_labels: super::w18_fb::MobNameLabels,
}

impl X6State {
    /// Refresh the snapshots when the settings generation moved; `true` when it did.
    fn sync(&mut self, s: &ModSettings) -> bool {
        if self.seen == s.generation() {
            return false;
        }
        self.seen = s.generation();
        self.arrows = StuckArrows {
            shown: s.enabled("stuck_arrows"),
            on_invisible: s.bool("stuck_arrows", "on_invisible"),
        };
        self.nametags = if s.enabled("nametags") {
            Nametags {
                own: s.bool("nametags", "own_nametag"),
                // Never larger or darker than vanilla. The descriptor
                // already clamps saved and edited values; this is the one place every label
                // (other players, your own in F5, sneaking) takes its style from.
                style: NameplateStyle {
                    text_shadow: s.bool("nametags", "text_shadow"),
                    background_alpha: (s.f32("nametags", "background") / 100.0)
                        .clamp(0.0, NameplateStyle::VANILLA.background_alpha),
                    scale: s
                        .f32("nametags", "scale")
                        .clamp(0.5, NameplateStyle::VANILLA.scale),
                },
                mob_names: s.bool("nametags", mc_mods::mods::nametags::ALWAYS_MOB_NAMES),
            }
        } else {
            Nametags::default()
        };
        self.hitboxes = if s.enabled("hitboxes") {
            let argb = s.color("hitboxes", "color");
            Hitboxes {
                enabled: true,
                always: s.choice("hitboxes", "show") == 0,
                players: s.bool("hitboxes", "players"),
                mobs: s.bool("hitboxes", "mobs"),
                items: s.bool("hitboxes", "items"),
                projectiles: s.bool("hitboxes", "projectiles"),
                other: s.bool("hitboxes", "other"),
                color: mc_render::entity::argb_to_rgba(argb),
                width: s.f32("hitboxes", "width"),
                eye: s.bool("hitboxes", "eye_line"),
                look: s.bool("hitboxes", "look_vector"),
            }
        } else {
            Hitboxes::default()
        };
        true
    }

    /// Nametags' "Always show mob names" (`false` while the mod is off).
    pub(crate) fn always_mob_names(&self) -> bool {
        self.nametags.mob_names
    }
}

/// `Entity.getVectorForRotation(pitch, yaw)`.
fn look_vector(yaw: f32, pitch: f32) -> Vec3 {
    let (yaw, pitch) = (yaw.to_radians(), pitch.to_radians());
    let f = mc_math::cos(-yaw - std::f32::consts::PI);
    let f1 = mc_math::sin(-yaw - std::f32::consts::PI);
    let f2 = -mc_math::cos(-pitch);
    let f3 = mc_math::sin(-pitch);
    Vec3::new(f1 * f2, f3, f * f2)
}

/// `prev + wrapDegrees(current - prev) * partial` (EntityLivingBase keeps `prev` within 180 of
/// `current`, so this is java's plain lerp).
fn lerp_degrees(prev: f32, current: f32, partial_ticks: f32) -> f32 {
    prev + mc_math::wrap_degrees_f32(current - prev) * partial_ticks
}

impl Client {
    /// Once per frame, before the entity passes: refresh the snapshots and, when settings
    /// changed, hand the nametag style to the label cache and the nameplate phase. One `u64`
    /// compare when nothing changed.
    pub(crate) fn x6_sync(&mut self) {
        if !self.x6.sync(&self.mods.settings) {
            return;
        }
        let style = self.x6.nametags.style;
        self.nameplate_cache.set_style(style);
        if let Some(render) = &self.render {
            render.nameplates.with(|phase| phase.set_style(style));
        }
    }

    /// `Client::install` (and every resource reload): register the hitbox phase (no GPU objects
    /// until the mod draws) and give the new nameplate phase the current style.
    pub(crate) fn x6_install(
        &mut self,
        phases: &mut PhaseList,
        nameplates: &Shared<NameplatePhase>,
    ) {
        let phase = Shared::new(HitboxPhase::new());
        phases.insert(HITBOX_PHASE_ORDER, Box::new(phase.clone()));
        self.x6.hitbox_phase = Some(phase);
        self.x6.hitboxes_drawn = false;
        self.x6.sync(&self.mods.settings);
        let style = self.x6.nametags.style;
        nameplates.with(|nameplates| nameplates.set_style(style));
        self.nameplate_cache.set_style(style);
    }

    /// Your own label in third person (the Nametags mod's "Show own nametag").
    ///
    /// java never draws it: `RendererLivingEntity.canRenderName` ends with `entity !=
    /// renderManager.livingPlayer`. Everything else follows that method for a player without the
    /// early team switch (which skips `thePlayer`): the GUI is on (F1 hides it), not invisible to
    /// yourself (`EntityPlayer.isInvisibleToPlayer`), and nobody rides you. The text is your team
    /// display name and the score line comes with it, as for any player.
    ///
    /// Returns `(text, feet, height, sneaking, username)` for the caller's label list.
    pub(crate) fn x6_own_nametag(
        &self,
        session: &Session,
        partial_ticks: f32,
    ) -> Option<(String, DVec3, f32, bool, String)> {
        if !self.x6.nametags.own
            || self.third_person_view == 0
            || self.hide_gui
            || !session.render_view_is_local()
        {
            return None;
        }
        let predictor = session.predictor()?;
        let local = &session.state().local;
        let username = local.username.as_str();
        if username.is_empty() {
            return None;
        }
        let board = &session.state().scoreboard;
        let invisible =
            crate::sync::watched::Watched::new(TrackedKind::Player, &local.metadata).is_invisible();
        let spectator = session
            .state()
            .players
            .get(&local.uuid)
            .is_some_and(|info| info.game_mode == mc_protocol::play::s2c::GameMode::Spectator);
        if invisible && !spectator && !board.sees_friendly_invisible(username, username) {
            return None;
        }
        let feet = super::super::interpolated_feet(
            self.camera_prev_pos
                .filter(|_| self.camera_prev_corrections == predictor.corrections),
            &predictor.state,
            partial_ticks,
        );
        Some((
            board.format_player_name(username),
            feet,
            predictor.state.height,
            predictor.state.sneaking,
            username.to_owned(),
        ))
    }

    /// The Hitboxes mod's per-frame work (camera final). Off: a `bool` test (plus one phase
    /// release the frame it is switched off).
    pub(crate) fn x6_frame(&mut self, gpu: &Gpu, camera: &Camera, partial_ticks: f32) {
        let hitboxes = self.x6.hitboxes;
        if !hitboxes.active(self.debug_bounding_box) {
            if self.x6.hitboxes_drawn {
                self.x6.hitboxes_drawn = false;
                if let Some(phase) = &self.x6.hitbox_phase {
                    if hitboxes.enabled {
                        // F3+B off in "With F3+B": keep the pipeline, draw nothing.
                        phase.with(|phase| {
                            phase.begin();
                            phase.finish(gpu);
                        });
                    } else {
                        // The mod went off: drop the GPU objects as well.
                        phase.with(HitboxPhase::release);
                    }
                }
            }
            return;
        }
        let Some(phase) = &self.x6.hitbox_phase else {
            return;
        };
        let Some(session) = &self.session else {
            // Left the server (or X3's in-game Join disconnected first):
            // empty the phase once, like F3+B off above, so it stops drawing the last world's
            // boxes behind the menus. The pipeline stays for the next world.
            if self.x6.hitboxes_drawn {
                self.x6.hitboxes_drawn = false;
                phase.with(|phase| {
                    phase.begin();
                    phase.finish(gpu);
                });
            }
            return;
        };
        let world = &session.state().world;
        let frustum = Frustum::from_camera(camera);
        let hidden = first_person_view_entity(session, self.third_person_view);
        let local_id = session.state().local.entity_id;
        phase.with(|phase| {
            phase.begin();
            for (network_id, tracked) in session.state().entities.iter() {
                if network_id == local_id || hidden == Some(network_id) {
                    continue;
                }
                let group = HitboxGroup::of(tracked);
                if !hitboxes.shows(group) {
                    continue;
                }
                // java: `debugBoundingBox && !entity.isInvisible()`, whoever can see the body.
                if tracked
                    .with_watched(world, |watched| watched.is_invisible())
                    .unwrap_or(false)
                {
                    continue;
                }
                let feet = tracked.render_position(world, partial_ticks);
                if (feet - camera.pos).length_squared() > HITBOX_RANGE_SQ {
                    continue;
                }
                let (width, height) = tracked.dimensions();
                let half = f64::from(width) * 0.5;
                if !frustum.contains_box(
                    feet - DVec3::new(half, 0.0, half),
                    feet + DVec3::new(half, f64::from(height), half),
                ) {
                    continue;
                }
                let current = world
                    .ecs
                    .get::<&mc_world::Rotation>(tracked.entity)
                    .map(|rotation| (rotation.yaw, rotation.pitch))
                    .unwrap_or((tracked.target_yaw, tracked.target_pitch));
                let pitch = lerp_degrees(tracked.prev_rotation_pitch, current.1, partial_ticks);
                // java: `EntityLivingBase.getLook` turns with the head; `Entity.getLook` with
                // the raw yaw.
                let yaw = if group.living() {
                    lerp_degrees(
                        tracked.render.prev_head_yaw,
                        tracked.render.head_yaw,
                        partial_ticks,
                    )
                } else {
                    lerp_degrees(tracked.prev_rotation_yaw, current.0, partial_ticks)
                };
                hitboxes.push(
                    phase,
                    camera.relative(feet),
                    (width, height),
                    group.living().then(|| tracked.eye_height(world)),
                    look_vector(yaw, pitch),
                );
            }
            // Yourself, in third person while the view is your own (F3+B draws it too).
            if hitboxes.players && self.third_person_view > 0 && session.render_view_is_local() {
                if let Some(predictor) = session.predictor() {
                    let local = &session.state().local;
                    let invisible =
                        crate::sync::watched::Watched::new(TrackedKind::Player, &local.metadata)
                            .is_invisible();
                    if !invisible {
                        let feet = super::super::interpolated_feet(
                            self.camera_prev_pos
                                .filter(|_| self.camera_prev_corrections == predictor.corrections),
                            &predictor.state,
                            partial_ticks,
                        );
                        let eye = (predictor.state.eye_pos().y - predictor.state.pos.y) as f32;
                        hitboxes.push(
                            phase,
                            camera.relative(feet),
                            (predictor.state.width, predictor.state.height),
                            Some(eye),
                            look_vector(predictor.state.yaw, predictor.state.pitch),
                        );
                    }
                }
            }
            phase.finish(gpu);
        });
        self.x6.hitboxes_drawn = true;
    }
}

/// Spectating through the real client: the `E1-50` capture (50 remote players) replayed on a
/// headless GPU, then the camera put inside one of them with `set_render_view_entity`, exactly
/// what `S43PacketCamera` does. Skips without an adapter, the vanilla jar or the capture.
#[cfg(test)]
mod spectator_tests {
    use std::path::{Path, PathBuf};

    use mc_physics::state::MoveInput;
    use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};

    use super::super::super::Client;
    use crate::app::campath::CameraScript;
    use crate::app::config::{ClientConfig, LoginMode};

    const SIZE: (u32, u32) = (640, 360);

    fn capture() -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)?
            .join("optimization")
            .join("fixtures")
            .join("generated-light-v2")
            .join("e1-50.mccap");
        path.is_file().then_some(path)
    }

    fn frame(client: &mut Client, gpu: &mut Gpu, phases: &mut PhaseList, camera: &mut Camera) {
        client.poll();
        client.tick(&MoveInput::default());
        client.frame(gpu, camera, 0.0);
        let target = gpu.acquire().expect("acquire offscreen frame");
        phases.render(gpu, target.view(), camera, 0.0);
    }

    /// Entity vertices within `radius` blocks of the camera-relative point `around`.
    fn vertices_near(client: &Client, around: [f32; 3], radius: f32) -> usize {
        client.render().expect("installed").entities.with(|phase| {
            phase
                .vertex_positions()
                .filter(|p| {
                    let d = [p[0] - around[0], p[1] - around[1], p[2] - around[2]];
                    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < radius * radius
                })
                .count()
        })
    }

    #[test]
    fn the_spectated_body_is_not_drawn_in_first_person_and_your_own_never_is() {
        let Some(capture) = capture() else {
            eprintln!("SKIP: the E1-50 capture is not in this checkout");
            return;
        };
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join(mc_mods::SETTINGS_FILE_NAME),
            r#"{"format":1,"mods":{"aero_title":{"enabled":false}}}"#,
        )
        .expect("write the scratch mod settings");
        let mut config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("E1Viewer".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return;
        }
        let mut gpu = match Gpu::headless(
            SIZE.0,
            SIZE.1,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                return;
            }
            Err(error) => panic!("headless GPU: {error}"),
        };
        config.replay = Some(capture);
        config.replay_step_micros = 50_000;
        config.replay_camera = CameraScript::Fixed;
        let mut client = Client::new(config, &mut |_, _| {}).expect("start");
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        let mut camera = Camera::default();
        client.connect("replay", 0);
        for _ in 0..120 {
            frame(&mut client, &mut gpu, &mut phases, &mut camera);
        }
        // The player nearest the middle of the view.
        let session = client.session.as_ref().expect("a replay session");
        let target = session
            .state()
            .entities
            .iter()
            .filter(|(_, tracked)| tracked.kind == super::TrackedKind::Player)
            .map(|(id, tracked)| {
                let at = tracked.render_position(&session.state().world, 0.0);
                ((at.x - 0.5).abs() + (at.z + 8.0).abs(), id)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, id)| id)
            .expect("the capture's players are spawned");

        // First person, looking out of the target: nothing of it around the camera.
        client.third_person_view = 0;
        client
            .session
            .as_mut()
            .expect("a replay session")
            .set_render_view_entity(target);
        frame(&mut client, &mut gpu, &mut phases, &mut camera);
        assert_eq!(
            vertices_near(&client, [0.0, 0.0, 0.0], 0.9),
            0,
            "the spectated body (head around the camera) is not drawn in first person"
        );

        // F5 while spectating: the target is drawn again (vanilla), your own body never is.
        client.third_person_view = 1;
        frame(&mut client, &mut gpu, &mut phases, &mut camera);
        let session = client.session.as_ref().expect("a replay session");
        let target_feet = session
            .state()
            .entities
            .get(target)
            .expect("still spawned")
            .render_position(&session.state().world, 0.0);
        let own_feet = session.predictor().expect("a local player").state.pos;
        let rel = |p: mc_math::DVec3| camera.relative(p).to_array();
        assert!(
            vertices_near(&client, rel(target_feet), 1.6) > 0,
            "in F5 the spectated player is drawn"
        );
        // The control for the first-person check: the head region the first-person camera sat
        // in does hold body vertices whenever the body is drawn.
        let target_eye = target_feet + mc_math::DVec3::new(0.0, 1.62, 0.0);
        assert!(
            vertices_near(&client, rel(target_eye), 0.9) > 0,
            "the head sits where the first-person camera was"
        );
        assert_eq!(
            vertices_near(&client, rel(own_feet), 1.2),
            0,
            "your own body is never drawn while you look out of another entity"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stuck_arrows_follow_the_two_switches() {
        let default = StuckArrows::default();
        assert!(default.visible(false) && !default.visible(true));
        let invisible_only = StuckArrows {
            shown: true,
            on_invisible: true,
        };
        assert!(invisible_only.visible(true));
        assert!(
            !invisible_only.visible(false),
            "invisible-only must not draw on visible players"
        );
        let default_on = StuckArrows {
            shown: true,
            on_invisible: false,
        };
        assert!(default_on.visible(false));
        assert!(
            !default_on.visible(true),
            "hidden on invisible players by default"
        );
        let off = StuckArrows {
            shown: false,
            on_invisible: true,
        };
        assert!(
            !off.visible(false) && !off.visible(true),
            "mod off hides every arrow"
        );
    }

    #[test]
    fn the_snapshot_reads_the_catalog_defaults() {
        let mut settings = ModSettings::new();
        let mut state = X6State::default();
        assert!(state.sync(&settings));
        assert!(
            !state.sync(&settings),
            "one u64 compare when nothing changed"
        );
        // Stuck Arrows is on by default, with arrows on invisible players hidden.
        assert!(state.arrows.visible(false));
        assert!(!state.arrows.visible(true));
        // Nametags and Hitboxes are off: vanilla labels, no boxes, F3+B untouched.
        assert!(state.nametags.style.is_vanilla());
        assert!(!state.nametags.own);
        assert!(!state.hitboxes.replaces_f3b());
        assert!(!state.hitboxes.active(true));

        settings.set_enabled("nametags", true);
        settings.set_enabled("hitboxes", true);
        assert!(state.sync(&settings));
        let style = state.nametags.style;
        assert!(style.text_shadow);
        assert!((style.background_alpha - 0.25).abs() < 1e-6);
        assert!((style.scale - 1.0).abs() < 1e-6);
        assert!(state.nametags.own);
        // Off by default, on with its option, never with the mod off.
        assert!(!state.always_mob_names());
        settings.set_bool("nametags", "always_mob_names", true);
        assert!(state.sync(&settings));
        assert!(state.always_mob_names());
        settings.set_enabled("nametags", false);
        assert!(state.sync(&settings));
        assert!(!state.always_mob_names(), "the mod's switch wins");
        settings.set_enabled("nametags", true);
        assert!(state.sync(&settings));
        assert!(state.hitboxes.active(false), "Show: Always");
        assert!(state.hitboxes.replaces_f3b());
        assert!(state.hitboxes.players && !state.hitboxes.mobs);
        assert_eq!(state.hitboxes.color, [1.0; 4]);
        assert!((state.hitboxes.width - 2.0).abs() < 1e-6);

        settings.set_choice("hitboxes", "show", 1);
        assert!(state.sync(&settings));
        assert!(!state.hitboxes.active(false), "With F3+B: off until F3+B");
        assert!(state.hitboxes.active(true));
    }

    #[test]
    fn the_look_vector_is_get_vector_for_rotation() {
        // Yaw 0 looks south (+Z), yaw 90 west (-X), pitch -90 straight up.
        let south = look_vector(0.0, 0.0);
        assert!(
            (south - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-4,
            "{south:?}"
        );
        let west = look_vector(90.0, 0.0);
        assert!(
            (west - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-4,
            "{west:?}"
        );
        let up = look_vector(0.0, -90.0);
        assert!((up - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-4, "{up:?}");
    }

    #[test]
    fn a_hitbox_is_a_box_an_eye_rect_and_a_look_line() {
        let mut phase = HitboxPhase::new();
        let all = Hitboxes {
            enabled: true,
            always: true,
            players: true,
            color: [1.0; 4],
            width: 3.0,
            eye: true,
            look: true,
            ..Hitboxes::default()
        };
        all.push(&mut phase, Vec3::ZERO, (0.6, 1.8), Some(1.62), Vec3::Z);
        assert_eq!(phase.segment_count(), 12 + 4 + 1);
        let look = phase.segments()[16];
        assert_eq!(look.color, LOOK_COLOR);
        assert!((look.a[1] - 1.62).abs() < 1e-6 && (look.b[2] - 2.0).abs() < 1e-6);
        assert!(phase.segments().iter().all(|s| (s.a[3] - 3.0).abs() < 1e-6));

        phase.begin();
        let bare = Hitboxes {
            eye: false,
            look: false,
            ..all
        };
        bare.push(&mut phase, Vec3::ZERO, (0.25, 0.25), None, Vec3::Z);
        assert_eq!(phase.segment_count(), 12, "a non-living box without extras");
    }
}

/// X15 (review L2): leaving the server empties the hitbox phase, so it stops drawing the last
/// world's boxes behind the menus.
#[cfg(test)]
mod x15_session_tests {
    use mc_render::RenderPhase;

    use super::super::x9_aa::tests::{join_world, offline_client};
    use super::*;

    #[test]
    fn leaving_the_world_empties_the_hitbox_phase() {
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
        let nameplates = Shared::new(NameplatePhase::new(&gpu));
        client.x6_install(&mut list, &nameplates);
        client.mods.settings.set_enabled("hitboxes", true);
        client.x6_sync();
        let (_to_game, _to_net) = join_world(&mut client);
        client.third_person_view = 1;
        let camera = Camera::default();
        let phase = client.x6.hitbox_phase.clone().expect("installed");

        // In a world, F5: your own box is drawn.
        client.x6_frame(&gpu, &camera, 0.0);
        assert!(phase.with(|p| p.enabled()), "your own box in F5");

        // Disconnected: the phase is emptied once and records nothing; the pipeline stays.
        client.session = None;
        client.x6_frame(&gpu, &camera, 0.0);
        assert!(
            !phase.with(|p| p.enabled()),
            "no stale boxes behind the menus"
        );
        assert!(!client.x6.hitboxes_drawn);
        assert!(
            phase.with(|p| p.has_gpu_resources()),
            "kept for the next world"
        );
        // Later frames without a world do nothing.
        client.x6_frame(&gpu, &camera, 0.0);
        assert!(!phase.with(|p| p.enabled()));
    }
}
