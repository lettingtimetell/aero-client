//! Dynamic FOV's per-source sliders, the client half.
//!
//! A request for "more sliders to dynamic fov for other specific fov changing things".
//! Vanilla 1.8.9 changes the field of view in two places:
//!
//! * `AbstractClientPlayer.getFovModifier` (`:109`), the per-tick target that
//!   `EntityRenderer.updateFovModifierHand` eases towards: flying (`x1.1`), then the
//!   `generic.movementSpeed` attribute through `(attribute / walkSpeed + 1) / 2`, then the bow.
//!   The attribute is the `0.1` base with the server's modifiers and the local sprint modifier.
//!   The vanilla sources are all operation 2 (multiply the total): sprint `+0.3`
//!   (`EntityLivingBase.sprintingSpeedBoostModifier`), Speed `+0.2` per level and Slowness
//!   `-0.15` per level (`Potion.moveSpeed`, `Potion.moveSlowdown`, applied by the server and
//!   sent in S20).
//! * `EntityRenderer.getFOVModifier` (`:551`), per frame, for the world camera and the held item
//!   alike: the death zoom (divide by `(1 - 500 / (deathTime + partialTicks + 500)) * 2 + 1`)
//!   and the underwater squeeze (`* 60 / 70`).
//!
//! Each slider scales its source **where vanilla computes it**. Sprinting, Speed and Slowness
//! multiply their own modifier's amount before the attribute is resolved again in Java's order
//! (so Speed II at 50 % is exactly Speed I, and sprinting with Speed still multiplies as in
//! vanilla); flying moves `x1.1` towards `x1`; underwater and dying move their factor towards
//! `1`. A modifier the client did not receive is never "removed": a server that sends no Speed
//! modifier keeps its FOV whatever the slider says. At 100 % every slider is vanilla bit for bit
//! (tests below), and at the defaults [`FovSources::from_settings`] is `None`, so each hook is
//! one `Option` test and allocates nothing.
//!
//! Order with the older controls in `m3_controls`: this file rebuilds the movement part of the
//! per-tick target from the sliders, then `strength` scales that part and `bow_zoom` keeps or
//! drops the bow (`Client::static_fov_target`), then vanilla's easing. Off ignores every slider
//! here: the movement and the bow are flattened, and the underwater and dying squeezes stay
//! vanilla, which is what off did before these sliders existed.

use mc_mods::mods::dynamic_fov::{
    DYING, FLYING, SLOWNESS_EFFECT, SPEED_EFFECT, SPRINTING, UNDERWATER,
};
use mc_mods::ModSettings;

use super::super::Client;
use crate::session::{AttributeSnapshot, RenderViewTarget};

const DYNAMIC_FOV: &str = "dynamic_fov";

/// The four movement sliders as shares (`1.0` = 100 % = vanilla).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MovementShares {
    /// Sprinting: scales the sprint modifier's amount.
    pub(crate) sprint: f64,
    /// Speed effect: scales `Potion.moveSpeed`'s modifier amount.
    pub(crate) speed: f64,
    /// Slowness effect: scales `Potion.moveSlowdown`'s modifier amount.
    pub(crate) slowness: f64,
    /// Flying: moves the `x1.1` towards `x1`.
    pub(crate) flying: f32,
}

impl MovementShares {
    /// Every movement slider at 100 %.
    pub(crate) const VANILLA: MovementShares = MovementShares {
        sprint: 1.0,
        speed: 1.0,
        slowness: 1.0,
        flying: 1.0,
    };
}

/// The per-source sliders while at least one of them is not at 100 % (`Client::m3`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FovSources {
    /// `None` while sprinting, Speed, Slowness and flying are all at 100 %.
    pub(crate) movement: Option<MovementShares>,
    /// The underwater share, `None` at 100 %.
    pub(crate) underwater: Option<f32>,
    /// The dying share, `None` at 100 %.
    pub(crate) dying: Option<f32>,
    /// The Speed- and Slowness-scaled attribute value (without sprint), with the S20 snapshot
    /// it was resolved from. Re-resolved only when a new snapshot arrives; the settings
    /// generation replaces the whole struct.
    potion_cache: Option<(AttributeSnapshot, f64)>,
}

impl FovSources {
    /// The sliders `settings` hold, or `None` when they change nothing: the mod is off (off
    /// ignores them) or every slider is at 100 %.
    #[must_use]
    pub(crate) fn from_settings(settings: &ModSettings) -> Option<FovSources> {
        if !settings.enabled(DYNAMIC_FOV) {
            return None;
        }
        let share = |id| f64::from(settings.f32(DYNAMIC_FOV, id).clamp(0.0, 100.0)) / 100.0;
        let movement = MovementShares {
            sprint: share(SPRINTING),
            speed: share(SPEED_EFFECT),
            slowness: share(SLOWNESS_EFFECT),
            flying: share(FLYING) as f32,
        };
        let movement = (movement != MovementShares::VANILLA).then_some(movement);
        let underwater = Some(share(UNDERWATER) as f32).filter(|share| *share < 1.0);
        let dying = Some(share(DYING) as f32).filter(|share| *share < 1.0);
        (movement.is_some() || underwater.is_some() || dying.is_some()).then_some(FovSources {
            movement,
            underwater,
            dying,
            potion_cache: None,
        })
    }
}

/// `PhysicsState::ai_move_speed` (the `generic.movementSpeed` value `getFovModifier` reads) with
/// the sprint modifier's amount scaled by `sprint_share`. `without_sprint` is the attribute
/// without the sprint modifier (Speed and Slowness already scaled), `sprint_amount` the sprint
/// modifier's amount while it is applied. Same order and precision, so a share of `1.0` is
/// `ai_move_speed` bit for bit.
#[must_use]
pub(crate) fn move_speed(
    without_sprint: f64,
    sprint_amount: Option<f64>,
    sprint_share: f64,
) -> f32 {
    let mut speed = without_sprint;
    if let Some(amount) = sprint_amount {
        speed *= 1.0 + amount * sprint_share;
    }
    speed.clamp(0.0, 1024.0) as f32
}

/// `AbstractClientPlayer.getFovModifier` without the bow (`:111-124`, `fov_modifier_target` in
/// the same order and precision), with flying's `x1.1` moved towards `x1` by `flying_share`. A
/// share of `1.0` is vanilla bit for bit.
#[must_use]
pub(crate) fn movement_target(
    flying: bool,
    move_speed: f32,
    walk_speed: f32,
    flying_share: f32,
) -> f32 {
    let mut f = 1.0_f32;
    if flying {
        f *= if flying_share >= 1.0 {
            1.1
        } else {
            1.0 + (1.1 - 1.0) * flying_share
        };
    }
    f = (f64::from(f) * ((f64::from(move_speed) / f64::from(walk_speed) + 1.0) / 2.0)) as f32;
    if walk_speed == 0.0 || !f.is_finite() {
        f = 1.0;
    }
    f
}

/// The underwater squeeze `* 60 / 70` moved towards `x1` by `share` (`< 1`; 100 % takes
/// vanilla's own expression in [`Client::underwater_fov`]).
#[must_use]
pub(crate) fn scaled_underwater(f: f32, share: f32) -> f32 {
    f * (1.0 + (60.0 / 70.0 - 1.0) * share)
}

/// The death zoom's divisor moved towards `1` by `share` (`< 1`; 100 % takes
/// `death_fov_modifier` itself in [`Client::dying_fov`]).
#[must_use]
pub(crate) fn scaled_death(f: f32, death_time: i32, partial_ticks: f32, share: f32) -> f32 {
    let f1 = death_time as f32 + partial_ticks;
    let divisor = (1.0_f32 - 500.0_f32 / (f1 + 500.0_f32)) * 2.0_f32 + 1.0_f32;
    f / (1.0 + (divisor - 1.0) * share)
}

/// The attribute without sprint for the local player: the predictor's value while Speed and
/// Slowness are at 100 % (vanilla's own number), otherwise the S20 snapshot resolved again with
/// the scaled potion modifiers, cached against that snapshot.
fn local_without_sprint(
    cache: &mut Option<(AttributeSnapshot, f64)>,
    snapshot: Option<&AttributeSnapshot>,
    predictor_value: f64,
    shares: MovementShares,
) -> f64 {
    if shares.speed >= 1.0 && shares.slowness >= 1.0 {
        return predictor_value;
    }
    let Some(snapshot) = snapshot else {
        // No S20 yet: the 0.1 base stands and there is no potion modifier to scale.
        return predictor_value;
    };
    if let Some((cached, value)) = cache.as_ref() {
        if cached == snapshot {
            return *value;
        }
    }
    let value = snapshot.resolved_without_sprint_with_potion_shares(shares.speed, shares.slowness);
    *cache = Some((snapshot.clone(), value));
    value
}

impl Client {
    /// The
    /// per-tick target with the movement sliders applied. `target` is vanilla's
    /// (`fov_modifier_target`) and is returned untouched while the movement sliders are at
    /// 100 % or there is no world; `bow_elapsed` is the draw that target already includes.
    pub(crate) fn fov_source_target(&mut self, target: f32, bow_elapsed: Option<i32>) -> f32 {
        let Some(sources) = self.m3.fov_sources.as_mut() else {
            return target;
        };
        let Some(shares) = sources.movement else {
            return target;
        };
        let Some(session) = self.session.as_ref() else {
            return target;
        };
        let movement = match session.render_view_target() {
            RenderViewTarget::Local => {
                let abilities = &session.state().local.abilities;
                let walk_speed = abilities.walk_speed;
                let speed = match session.predictor() {
                    Some(predictor) => {
                        let vanilla = predictor.state.movement_speed;
                        // The snapshot lookup only when a potion slider moved.
                        let without_sprint = if shares.speed >= 1.0 && shares.slowness >= 1.0 {
                            vanilla
                        } else {
                            local_without_sprint(
                                &mut sources.potion_cache,
                                session.local_movement_attribute(),
                                vanilla,
                                shares,
                            )
                        };
                        move_speed(
                            without_sprint,
                            predictor.state.sprint_modifier_amount,
                            shares.sprint,
                        )
                    }
                    // `update_fov_modifier_hand` reads the walk speed as the attribute here.
                    None => walk_speed,
                };
                movement_target(abilities.flying, speed, walk_speed, shares.flying)
            }
            RenderViewTarget::Entity { network_id, .. } => {
                // A spectated player: vanilla knows only their sprint flag (see
                // `update_fov_modifier_hand`); a non-player view entity keeps `1.0`.
                let tracked =
                    session.state().entities.get(network_id).filter(|tracked| {
                        tracked.kind == crate::sync::entities::TrackedKind::Player
                    });
                let Some(tracked) = tracked else {
                    return target;
                };
                let walk_speed = mc_physics::state::WALK_SPEED as f32;
                let boost = mc_physics::state::SPRINT_SPEED_BOOST as f32 * shares.sprint as f32;
                let speed = if tracked.render.entity.sprinting {
                    walk_speed * (1.0 + boost)
                } else {
                    walk_speed
                };
                movement_target(false, speed, walk_speed, shares.flying)
            }
        };
        match bow_elapsed {
            Some(elapsed) => movement * crate::app::interact::bow_fov_multiplier(elapsed),
            None => movement,
        }
    }

    /// `vanilla` is the caller's
    /// own `f * 60 / 70`, returned as is while the Underwater slider is at 100 %.
    #[must_use]
    pub(crate) fn underwater_fov(&self, f: f32, vanilla: f32) -> f32 {
        match self
            .m3
            .fov_sources
            .as_ref()
            .and_then(|sources| sources.underwater)
        {
            None => vanilla,
            Some(share) => scaled_underwater(f, share),
        }
    }

    /// `vanilla` is the caller's own
    /// `death_fov_modifier(f, ..)`, returned as is while the Dying slider is at 100 %.
    #[must_use]
    pub(crate) fn dying_fov(
        &self,
        f: f32,
        vanilla: f32,
        death_time: i32,
        partial_ticks: f32,
    ) -> f32 {
        match self
            .m3
            .fov_sources
            .as_ref()
            .and_then(|sources| sources.dying)
        {
            None => vanilla,
            Some(share) => scaled_death(f, death_time, partial_ticks, share),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{death_fov_modifier, fov_modifier_target};
    use super::*;
    use mc_math::DVec3;
    use mc_physics::state::{PhysicsState, SPRINT_SPEED_BOOST, WALK_SPEED};
    use mc_protocol::play::s2c::{
        AttributeModifierOperation as Op, AttributeModifierSnapshot, AttributePropertySnapshot,
    };

    const SPEED: uuid::Uuid = uuid::Uuid::from_u128(0x91AEAA56_376B_4498_935B_2F7F68070635);
    const SLOWNESS: uuid::Uuid = uuid::Uuid::from_u128(0x7107DE5E_7CE8_4030_940E_514C1F160890);
    const SPRINT: uuid::Uuid = uuid::Uuid::from_u128(0x662A6B8D_DA3E_4C1C_8813_96EA6097278D);
    /// A modifier no vanilla source owns (a server's own speed change): never scaled.
    const OTHER: uuid::Uuid = uuid::Uuid::from_u128(0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0);

    /// `Potion.moveSpeed`'s amount at `level` (`func_111183_a`: `amount * (amplifier + 1)`).
    fn speed(level: u8) -> AttributeModifierSnapshot {
        AttributeModifierSnapshot {
            uuid: SPEED,
            amount: 0.20000000298023224 * f64::from(level),
            operation: Op::Multiply,
        }
    }

    fn slowness(level: u8) -> AttributeModifierSnapshot {
        AttributeModifierSnapshot {
            uuid: SLOWNESS,
            amount: -0.15000000596046448 * f64::from(level),
            operation: Op::Multiply,
        }
    }

    fn sprint() -> AttributeModifierSnapshot {
        AttributeModifierSnapshot {
            uuid: SPRINT,
            amount: SPRINT_SPEED_BOOST,
            operation: Op::Multiply,
        }
    }

    fn other(amount: f64, operation: Op) -> AttributeModifierSnapshot {
        AttributeModifierSnapshot {
            uuid: OTHER,
            amount,
            operation,
        }
    }

    fn snapshot(modifiers: &[AttributeModifierSnapshot]) -> AttributeSnapshot {
        AttributeSnapshot::from_wire(WALK_SPEED, modifiers).expect("distinct UUIDs")
    }

    /// Every movement slider at 100 % is `getFovModifier` bit for bit: the same attribute as
    /// `PhysicsState::ai_move_speed` and the same target as `fov_modifier_target`, over walking,
    /// sprinting, flying, Speed and Slowness values, odd walk speeds and the zero-walk guard.
    #[test]
    fn every_movement_slider_at_100_percent_is_vanilla_bit_for_bit() {
        let attributes = [
            WALK_SPEED,
            WALK_SPEED * 1.2,
            WALK_SPEED * 1.4,
            WALK_SPEED * 0.85,
            WALK_SPEED * 0.85 * 1.2,
            WALK_SPEED * 2.0,
            0.0,
            0.123_456_789,
            5000.0,
        ];
        let mut checked = 0;
        for flying in [false, true] {
            for without_sprint in attributes {
                for sprint_amount in [None, Some(SPRINT_SPEED_BOOST), Some(0.5)] {
                    for walk_speed in [0.1_f32, 0.05, 0.2, 0.0, 0.123] {
                        let mut state = PhysicsState::new_player(DVec3::ZERO);
                        state.movement_speed = without_sprint;
                        state.sprint_modifier_amount = sprint_amount;
                        let speed = move_speed(without_sprint, sprint_amount, 1.0);
                        assert_eq!(speed.to_bits(), state.ai_move_speed().to_bits());
                        let vanilla =
                            fov_modifier_target(flying, state.ai_move_speed(), walk_speed, None);
                        let ours = movement_target(flying, speed, walk_speed, 1.0);
                        assert_eq!(
                            ours.to_bits(),
                            vanilla.to_bits(),
                            "flying {flying}, attribute {without_sprint}, sprint \
                             {sprint_amount:?}, walk {walk_speed}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 2 * 9 * 3 * 5);
    }

    /// Sprinting and flying scale their own change and nothing else; 0 % is walking on the
    /// ground, 50 % half the change (a lone multiplier scales the FOV change linearly).
    #[test]
    fn sprinting_and_flying_sliders_scale_only_their_own_change() {
        let walk = WALK_SPEED as f32;
        let sprint = Some(SPRINT_SPEED_BOOST);
        // `(flying, without sprint, sprint share, flying share)` to the movement target.
        let target = |flying, without_sprint, sprint_share, flying_share| {
            let speed = move_speed(without_sprint, sprint, sprint_share);
            movement_target(flying, speed, walk, flying_share)
        };
        let vanilla_sprint = target(false, WALK_SPEED, 1.0, 1.0);
        assert!((vanilla_sprint - 1.15).abs() < 1e-6, "{vanilla_sprint}");
        assert_eq!(target(false, WALK_SPEED, 0.0, 1.0), 1.0, "0 % is walking");
        let half = target(false, WALK_SPEED, 0.5, 1.0);
        assert!((half - 1.075).abs() < 1e-6, "{half}");
        // The sprint slider leaves flying alone, and the other way round.
        let flying_no_sprint = target(true, WALK_SPEED, 0.0, 1.0);
        assert!((flying_no_sprint - 1.1).abs() < 1e-6, "{flying_no_sprint}");
        assert_eq!(target(true, WALK_SPEED, 1.0, 0.0), vanilla_sprint);
        let half_flying = movement_target(true, walk, walk, 0.5);
        assert!((half_flying - 1.05).abs() < 1e-6, "{half_flying}");
        // Speed II stays in the attribute when only the sprint slider moves.
        let speed_two = target(false, WALK_SPEED * 1.4, 0.0, 1.0);
        assert!((speed_two - 1.2).abs() < 1e-6, "{speed_two}");
    }

    /// The Speed and Slowness sliders rescale their own S20 modifier and resolve the attribute
    /// again in Java's order: 100 % is the predictor's value bit for bit, Speed II at 50 % is
    /// exactly Speed I, 0 % drops the effect, and a modifier no vanilla source owns stays.
    #[test]
    fn potion_sliders_rescale_their_own_modifier_in_java_order() {
        let with_all = snapshot(&[
            speed(2),
            slowness(1),
            sprint(),
            other(0.05, Op::MultiplyBase),
        ]);
        let resolve =
            |modifiers: &[AttributeModifierSnapshot]| snapshot(modifiers).resolved_value();
        let vanilla_without_sprint =
            resolve(&[speed(2), slowness(1), other(0.05, Op::MultiplyBase)]);
        assert_eq!(
            with_all
                .resolved_without_sprint_with_potion_shares(1.0, 1.0)
                .to_bits(),
            vanilla_without_sprint.to_bits()
        );
        assert_eq!(
            with_all.resolved_without_sprint_with_potion_shares(0.5, 1.0),
            resolve(&[speed(1), slowness(1), other(0.05, Op::MultiplyBase)]),
            "Speed II at 50 % is Speed I"
        );
        assert_eq!(
            with_all.resolved_without_sprint_with_potion_shares(0.0, 1.0),
            resolve(&[
                AttributeModifierSnapshot {
                    amount: 0.0,
                    ..speed(2)
                },
                slowness(1),
                other(0.05, Op::MultiplyBase)
            ]),
        );
        // Slowness at 0 %: the base, the other modifier's `+5 %` of the base, then Speed II.
        let no_slowness = with_all.resolved_without_sprint_with_potion_shares(1.0, 0.0);
        let expected = (WALK_SPEED + WALK_SPEED * 0.05) * (1.0 + speed(2).amount);
        assert!(
            (no_slowness - expected).abs() < 1e-15,
            "{no_slowness} vs {expected}"
        );
        let neither = with_all.resolved_without_sprint_with_potion_shares(0.0, 0.0);
        let expected = WALK_SPEED + WALK_SPEED * 0.05;
        assert!(
            (neither - expected).abs() < 1e-15,
            "the other modifier stays"
        );
    }

    /// The cache resolves a snapshot once and follows a new one; 100 % never reads it.
    #[test]
    fn the_potion_cache_follows_the_snapshot() {
        let half_speed = MovementShares {
            speed: 0.5,
            ..MovementShares::VANILLA
        };
        let mut cache = None;
        let speed_two = snapshot(&[speed(2)]);
        let first = local_without_sprint(&mut cache, Some(&speed_two), 0.14, half_speed);
        assert_eq!(first, snapshot(&[speed(1)]).resolved_value());
        assert!(cache.is_some());
        let speed_four = snapshot(&[speed(4)]);
        let second = local_without_sprint(&mut cache, Some(&speed_four), 0.18, half_speed);
        assert_eq!(second, snapshot(&[speed(2)]).resolved_value());
        // Vanilla shares hand back the predictor's own value, snapshot or not.
        assert_eq!(
            local_without_sprint(&mut cache, Some(&speed_four), 0.18, MovementShares::VANILLA),
            0.18
        );
        // No S20 yet: nothing to scale.
        assert_eq!(local_without_sprint(&mut None, None, 0.1, half_speed), 0.1);
    }

    /// The real packet path: an S20 for the local player reaches the snapshot the sliders read,
    /// and at 100 % that snapshot resolves to the predictor's own attribute, bit for bit.
    #[test]
    fn an_s20_reaches_the_snapshot_and_100_percent_matches_the_predictor() {
        use crate::session::{JoinSettings, Session};
        let (handle, to_game, _to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let join = mc_protocol::play::s2c::JoinGame {
            entity_id: 3,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        };
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(join),
            ))
            .expect("join");
        let _ = session.poll_all();
        assert!(session.local_movement_attribute().is_none(), "no S20 yet");
        let properties = mc_protocol::play::s2c::EntityProperties {
            entity_id: 3,
            properties: vec![AttributePropertySnapshot {
                key: "generic.movementSpeed".into(),
                base_value: WALK_SPEED,
                modifiers: vec![speed(2), slowness(1), sprint()],
            }],
        };
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::EntityProperties(properties),
            ))
            .expect("properties");
        let _ = session.poll_all();
        // What every client tick does before the physics: the predictor takes the S20 value.
        session.push_server_movement_state();
        let attribute = session.local_movement_attribute().expect("the snapshot");
        let predictor = session.predictor().expect("a predictor after JoinGame");
        assert_eq!(
            attribute
                .resolved_without_sprint_with_potion_shares(1.0, 1.0)
                .to_bits(),
            predictor.state.movement_speed.to_bits(),
            "100 % is the predictor's attribute"
        );
        assert_eq!(
            predictor.state.sprint_modifier_amount,
            Some(SPRINT_SPEED_BOOST)
        );
        assert_eq!(
            attribute.resolved_without_sprint_with_potion_shares(0.0, 0.0),
            WALK_SPEED,
            "both potions at 0 % leave the base"
        );
    }

    /// Underwater and dying: the hooks are vanilla's expressions while the sliders are at 100 %
    /// (`from_settings` gives `None`), 0 % keeps the FOV, and the scaled curves meet vanilla at
    /// 100 % (within a rounding step: the hooks never use them there).
    #[test]
    fn underwater_and_dying_scale_their_squeeze() {
        for f in [1.0_f32, 1.15, 0.85, 70.0] {
            assert_eq!(scaled_underwater(f, 0.0), f);
            let vanilla = f * 60.0 / 70.0;
            assert!((scaled_underwater(f, 1.0) - vanilla).abs() <= vanilla * 1e-6);
            let half = scaled_underwater(f, 0.5);
            assert!((half - f * (1.0 - 0.5 / 7.0)).abs() <= f * 1e-6, "{half}");
            for (death_time, partial) in [(0, 0.0_f32), (5, 0.5), (19, 0.99)] {
                assert_eq!(scaled_death(f, death_time, partial, 0.0), f);
                let vanilla = death_fov_modifier(f, death_time, partial);
                let ours = scaled_death(f, death_time, partial, 1.0);
                assert!(
                    (ours - vanilla).abs() <= vanilla * 1e-6,
                    "{ours} vs {vanilla}"
                );
                let half = scaled_death(f, death_time, partial, 0.5);
                assert!(half <= f && half >= vanilla, "between still and vanilla");
            }
        }
    }

    /// A headless client over the real assets, its game dir a temp dir (so the mod settings
    /// file never lands in the user's `mc-rs`). `None` without the vanilla jar.
    fn headless_client() -> Option<(Client, tempfile::TempDir)> {
        use crate::app::config::{ClientConfig, LoginMode};
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config
            .version_jar
            .as_ref()
            .is_some_and(|path| path.is_file())
        {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// The tick hook is wired into `update_fov_modifier_hand`: at the defaults a sprinting
    /// player's FOV eases towards vanilla's 1.15 (the negative control), with Sprinting at 0 %
    /// it stays at the walking 1.0, and at 50 % it eases towards 1.075.
    #[test]
    fn the_tick_hook_applies_the_sprinting_slider() {
        use crate::session::{JoinSettings, Session};
        let Some((mut client, _dir)) = headless_client() else {
            return;
        };
        let (handle, to_game, _to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let join = mc_protocol::play::s2c::JoinGame {
            entity_id: 3,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        };
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(join),
            ))
            .expect("join");
        let _ = session.poll_all();
        session
            .predictor_mut()
            .expect("a predictor after JoinGame")
            .state
            .set_sprinting(true);
        client.session = Some(session);
        let eased = |client: &mut Client| {
            client.m3.sync(&client.mods.settings);
            client.fov_modifier_hand = 1.0;
            client.update_fov_modifier_hand();
            client.fov_modifier_hand
        };
        let vanilla = eased(&mut client);
        assert!(
            (vanilla - 1.075).abs() < 1e-6,
            "vanilla eases half way to 1.15: {vanilla}"
        );
        client.mods.settings.set_f32(DYNAMIC_FOV, SPRINTING, 0.0);
        assert_eq!(eased(&mut client), 1.0, "sprinting at 0 % is walking");
        client.mods.settings.set_f32(DYNAMIC_FOV, SPRINTING, 50.0);
        let half = eased(&mut client);
        assert!((half - 1.0375).abs() < 1e-6, "half way to 1.075: {half}");
        client.mods.settings.set_f32(DYNAMIC_FOV, SPRINTING, 100.0);
        assert_eq!(eased(&mut client), vanilla, "back to vanilla, bit for bit");

        // The frame hooks: the caller's vanilla value at the defaults, then the sliders.
        let f = 1.15_f32;
        let (water, death) = (f * 60.0 / 70.0, death_fov_modifier(f, 5, 0.5));
        assert_eq!(client.underwater_fov(f, water).to_bits(), water.to_bits());
        assert_eq!(
            client.dying_fov(f, death, 5, 0.5).to_bits(),
            death.to_bits()
        );
        client.mods.settings.set_f32(DYNAMIC_FOV, UNDERWATER, 0.0);
        client.mods.settings.set_f32(DYNAMIC_FOV, DYING, 0.0);
        client.m3.sync(&client.mods.settings);
        assert_eq!(
            client.underwater_fov(f, water),
            f,
            "underwater at 0 % keeps the FOV"
        );
        assert_eq!(
            client.dying_fov(f, death, 5, 0.5),
            f,
            "dying at 0 % keeps the FOV"
        );
        // Off ignores the sliders: the squeezes are vanilla again.
        client.mods.settings.set_enabled(DYNAMIC_FOV, false);
        client.m3.sync(&client.mods.settings);
        assert_eq!(client.underwater_fov(f, water).to_bits(), water.to_bits());
        assert_eq!(
            client.dying_fov(f, death, 5, 0.5).to_bits(),
            death.to_bits()
        );
    }

    /// The settings: the defaults and off are `None` (no state, no work); each slider sets only
    /// its own share; out-of-range values are clamped.
    #[test]
    fn sliders_read_as_shares_and_the_defaults_cost_nothing() {
        let mut settings = ModSettings::new();
        assert!(settings.enabled(DYNAMIC_FOV));
        assert_eq!(
            FovSources::from_settings(&settings),
            None,
            "defaults are vanilla"
        );
        settings.set_f32(DYNAMIC_FOV, SPRINTING, 40.0);
        let sources = FovSources::from_settings(&settings).expect("a slider moved");
        let movement = sources.movement.expect("a movement slider");
        assert!((movement.sprint - 0.4).abs() < 1e-6);
        assert_eq!(
            (movement.speed, movement.slowness, movement.flying),
            (1.0, 1.0, 1.0)
        );
        assert_eq!((sources.underwater, sources.dying), (None, None));
        settings.set_f32(DYNAMIC_FOV, SPRINTING, 100.0);
        settings.set_f32(DYNAMIC_FOV, UNDERWATER, 0.0);
        let sources = FovSources::from_settings(&settings).expect("underwater moved");
        assert_eq!(
            (sources.movement, sources.underwater, sources.dying),
            (None, Some(0.0), None)
        );
        settings.set_f32(DYNAMIC_FOV, UNDERWATER, 100.0);
        settings.set_f32(DYNAMIC_FOV, DYING, 25.0);
        assert_eq!(
            FovSources::from_settings(&settings).unwrap().dying,
            Some(0.25)
        );
        // Off ignores every slider (m3_controls flattens the movement and the bow).
        settings.set_enabled(DYNAMIC_FOV, false);
        assert_eq!(FovSources::from_settings(&settings), None);
    }
}
