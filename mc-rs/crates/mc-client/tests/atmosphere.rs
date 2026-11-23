//! S21 §13 integration: potion effects, lightning, eye-position-driven fog/clear colour, and the
//! HUD's `underwater`/`air`/`max_health` trio.
//!
//! Six real gaps this closes, all found while answering the plan's `S09 → S21` / `S20c → S21` /
//! `S21e | S14/S16` cross-section rows: `drive_atmosphere` never drove `FogInputs::medium`/
//! `biome_temperature`, `PhaseList::clear_color` was a fixed placeholder for every dimension,
//! `Weather::last_lightning_bolt` was hard-coded `0`, nothing fed the local player's active
//! potion effects to the HUD at all, nothing simulated `air`/`underwater` (`EntityLivingBase.
//! onEntityUpdate`'s decrement-while-submerged, damage-and-reset-at-`-20` state machine ran
//! nowhere in this client), and `S20PacketEntityProperties` reached `ClientEvent::
//! EntityProperties` but nothing read it, so `PlayerHud::max_health` never moved off its `20.0`
//! default. Packet-level, against the loopback double `smoke.rs`/`input_dispatch.rs` use —
//! asserting on `S1DPacketEntityEffect`/`S2CPacketSpawnGlobalEntity`/`S23PacketBlockChange`/
//! `S20PacketEntityProperties` reaching public client state, not on "something changed".

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_protocol::metadata::{MetadataEntry, MetadataValue};
use mc_protocol::play::s2c;
use mc_render::{Camera, PhaseList};
use tokio::net::TcpListener;
use uuid::Uuid;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// The id `accept_and_join` puts in `JoinGame`, so it becomes `session.state().local.entity_id`.
const LOCAL_ENTITY_ID: i32 = 42;

struct Rig {
    client: Client,
    gpu: mc_render::Gpu,
    phases: PhaseList,
    camera: Camera,
    peer: tokio::task::JoinHandle<common::RefPeer>,
}

impl Rig {
    fn pump(&mut self) {
        pump(
            &mut self.client,
            &mut self.gpu,
            &mut self.phases,
            &mut self.camera,
            &mc_physics::state::MoveInput::default(),
        );
    }

    fn pump_n(&mut self, n: usize) {
        for _ in 0..n {
            self.pump();
        }
    }

    fn finish(mut self) {
        self.client.disconnect();
        for _ in 0..10 {
            self.pump();
        }
        let handle = self.client.runtime().handle().clone();
        handle.block_on(async {
            tokio::time::timeout(Duration::from_secs(10), self.peer)
                .await
                .expect("the peer must finish")
                .expect("the peer task must not panic");
        });
    }
}

/// Join the loopback double, script `extra` packets right after join, and stop once the world
/// is ready. The extra packets ride the same connection the join used, so by the time
/// `pump_until` notices `in_game` they are already sitting in the client's inbound queue.
fn join_and_script<F, Fut>(extra: F) -> Option<Rig>
where
    F: FnOnce(common::RefPeer) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = common::RefPeer> + Send,
{
    let gpu = gpu(WIDTH, HEIGHT)?;
    let dir = Box::leak(Box::new(tempfile::tempdir().expect("tempdir")));
    let config = config(dir.path(), "Dev")?;
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let camera = mc_client::app::driver::menu_camera();

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let peer = client.runtime().handle().spawn(async move {
        let mut peer = accept_and_join(listener, LOCAL_ENTITY_ID).await;
        peer = extra(peer).await;
        peer.record_until_closed().await;
        peer
    });

    client.handle_action(mc_ui::screens::Action::Connect(Box::new(
        mc_ui::screens::multiplayer::ServerEntry::new("Loopback", format!("127.0.0.1:{port}")),
    )));

    let mut rig = Rig {
        client,
        gpu,
        phases,
        camera,
        peer,
    };
    pump_until(
        &mut rig.client,
        &mut rig.gpu,
        &mut rig.phases,
        &mut rig.camera,
        &mc_physics::state::MoveInput::default(),
        Duration::from_secs(30),
        "the client to reach the world",
        |client| {
            client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|s| s.world().is_chunk_loaded(CHUNK))
        },
    );
    Some(rig)
}

/// **The core of the `S21e → S14/S16` and `S09 → S21` rows.** `S1DPacketEntityEffect` for the
/// local player must reach `PlayerHud::effects`/`.absorption`/`.poisoned`;
/// `S1EPacketRemoveEntityEffect` clears a tracked effect; duration expiry alone retains it on a
/// remote world. Watcher-backed absorption remains authoritative until a metadata update; an effect
/// on a *different* entity must never leak onto the local player's HUD. All asserted on the wire
/// packets and public HUD state, not on an internal flag guess.
#[test]
fn entity_effect_packets_feed_the_huds_potion_list_absorption_and_flags() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        // Absorption (22), amplifier 1. Java's `PotionAbsorption` writes the player's watched
        // absorption float on the server; the client receives that authoritative value below.
        // The short effect duration expires naturally partway through the test.
        peer.send(&s2c::EntityEffect {
            entity_id: LOCAL_ENTITY_ID,
            effect_id: 22,
            amplifier: 1,
            duration: 3,
            show_particles: false,
        })
        .await;
        // java: GuiIngame.renderPlayerStats reads EntityPlayer's watched absorption float (index
        // 17). PotionAbsorption is a server-side writer of that watcher; an S1D effect packet
        // alone must not fabricate hearts on the client HUD.
        peer.send(&s2c::EntityMetadata {
            entity_id: LOCAL_ENTITY_ID,
            metadata: vec![MetadataEntry {
                index: 17,
                value: MetadataValue::Float(8.0),
            }],
        })
        .await;
        // Haste (3), amplifier 0: the "stays present" control.
        peer.send(&s2c::EntityEffect {
            entity_id: LOCAL_ENTITY_ID,
            effect_id: 3,
            amplifier: 0,
            duration: 200,
            show_particles: false,
        })
        .await;
        // Poison (19): added, then immediately removed by packet rather than by expiry.
        peer.send(&s2c::EntityEffect {
            entity_id: LOCAL_ENTITY_ID,
            effect_id: 19,
            amplifier: 0,
            duration: 200,
            show_particles: false,
        })
        .await;
        peer.send(&s2c::RemoveEntityEffect {
            entity_id: LOCAL_ENTITY_ID,
            effect_id: 19,
        })
        .await;
        // A different entity's effect must never leak onto the local player's HUD.
        peer.send(&s2c::EntityEffect {
            entity_id: LOCAL_ENTITY_ID + 1,
            effect_id: 16, // night vision
            amplifier: 0,
            duration: 999,
            show_particles: false,
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.pump_n(2);
    assert_eq!(
        rig.client.player_hud.absorption, 8.0,
        "the server's watched absorption float is authoritative"
    );
    assert!(
        !rig.client.player_hud.poisoned,
        "removed by packet before it ever ticked"
    );
    assert_eq!(
        rig.client.player_hud.effects.len(),
        2,
        "absorption + haste only — not poison (removed), not entity {}'s night vision",
        LOCAL_ENTITY_ID + 1
    );
    let absorption = rig
        .client
        .player_hud
        .effects
        .iter()
        .find(|e| e.name.starts_with("Absorption"))
        .expect("absorption in the list");
    assert_eq!(absorption.icon_index, 18); // mc_data::potion(22).status_icon_index
    assert!(absorption.beneficial);
    let haste = rig
        .client
        .player_hud
        .effects
        .iter()
        .find(|e| e.name == "Haste") // amplifier 0 -> no roman-numeral suffix
        .expect("haste in the list");
    assert!(haste.beneficial);

    // Absorption's 3-tick duration expires; haste's 200-tick one does not.
    rig.pump_n(6);
    assert_eq!(
        rig.client.player_hud.absorption, 8.0,
        "the watched absorption value persists until the server updates it"
    );
    // EntityLivingBase.updatePotionEffects removes expired effects only when !world.isRemote.
    assert_eq!(
        rig.client.player_hud.effects.len(),
        2,
        "remote expiry retains both effects"
    );
    assert!(rig
        .client
        .player_hud
        .effects
        .iter()
        .any(|effect| effect.name == "Haste"));
    assert!(rig
        .client
        .player_hud
        .effects
        .iter()
        .any(|effect| effect.name.starts_with("Absorption")));

    rig.finish();
}

/// **The `S20c → S21` lightning row.** `S2CPacketSpawnGlobalEntity` (entity type `1`, a lightning
/// bolt) must set `World.lastLightningBolt`. EntityLightningBolt.onUpdate can reflash up to
/// three times, separated by at most ten negative-state ticks. Once the entity expires, the
/// world's counter must count down to zero and stay there.
///
/// One `pump()` is `poll()` followed by `tick()`. If the packet is drained after that tick, the
/// externally visible value is `2` until the next tick; if it is drained before the tick, the
/// next observation is `1`. The packet's arrival point is a real network race (the peer sends it
/// as soon as the socket is up, independently of how many `pump()`s `join_and_script`'s own
/// join-wait needed), so this polls for the rise instead of asserting one specific call.
#[test]
fn a_lightning_bolt_sets_last_lightning_bolt_and_it_decays_to_zero_and_stays() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::SpawnGlobalEntity {
            entity_id: 9001,
            entity_type: 1,
            x_fixed: 8 * 32,
            y_fixed: 80 * 32,
            z_fixed: 8 * 32,
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    let mut observed = Vec::new();
    for _ in 0..64 {
        rig.pump();
        observed.push(rig.client.last_lightning_bolt());
    }
    assert!(
        observed.contains(&1),
        "the counter must have been externally observable at 1 at some point: {observed:?}"
    );
    assert!(
        observed.iter().all(|&v| v <= 2),
        "the Java handler stores 2 before the next tick decrements it: {observed:?}"
    );
    assert_eq!(
        observed.last(),
        Some(&0),
        "must settle at 0 within the complete bolt lifecycle: {observed:?}"
    );
    // Initial state 2 plus at most three reflashes cannot survive 48 ticks, even if every
    // nextInt(10) postpones the next flash as long as Java permits. Check a further 16 ticks
    // so a stale retained weather entity cannot keep re-triggering the world's flash.
    assert!(
        observed[48..].iter().all(|&v| v == 0),
        "must not re-trigger after the bolt expires: {observed:?}"
    );

    rig.finish();
}

/// **The `S20c → S21` clear-colour row, and the medium/biome half of `S09 → S21`.** Once the
/// world is live the clear colour must move off `PhaseList::new`'s fixed placeholder — proving
/// `drive_atmosphere` is actually wired to `ShellApp::frame`'s `ctx.phases.clear_color`, not
/// merely computing a value nobody reads. Spawn stands on a stone floor in daylight, so this also
/// pins the negative case for the eye-medium lookup: a clear colour with no lava/water tint.
#[test]
fn joining_the_world_moves_the_clear_colour_off_the_placeholder() {
    let placeholder = wgpu::Color {
        r: 0.043,
        g: 0.070,
        b: 0.145,
        a: 1.0,
    };
    let Some(mut rig) = join_and_script(|peer| async move { peer }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.pump_n(3);
    let color = rig.client.clear_color();
    assert_ne!(
        (color.r, color.g, color.b),
        (placeholder.r, placeholder.g, placeholder.b),
        "drive_atmosphere must have overwritten the placeholder by now"
    );
    // Not lava-tinted: `Environment::base_fog_color`'s overworld branch never produces a
    // near-lava colour (red channel dominant, green/blue near zero) from a temperate default
    // biome standing on stone in daylight.
    assert!(
        !(color.r > 0.4 && color.g < 0.15 && color.b < 0.05),
        "spawn is on stone, not lava: {color:?}"
    );
    for channel in [color.r, color.g, color.b] {
        assert!((0.0..=1.0).contains(&channel), "{color:?}");
    }

    rig.finish();
}

/// Where the local player's eye lands: spawn feet are `common::SPAWN` = `(8.5, 5.0, 8.5)`, and
/// standing eye height `1.62` (`mc_physics::state::PLAYER_EYE_HEIGHT`) puts the eye at
/// `y = 6.62`, whose containing block is `(8, 6, 8)`.
const EYE_BLOCK: mc_math::BlockPos = mc_math::BlockPos { x: 8, y: 6, z: 8 };
/// Still water, block id 9 meta 0 (`minecraft:water`; `9 << 4` in this port's packed
/// `BLOCK_STATE_IDS` scheme — see `common::STONE`'s own `1 << 4` for the same packing).
const WATER_BLOCK_STATE: i32 = 9 << 4;

/// **The `underwater`/`air` two-thirds of the `S21e | S14/S16` tail row.** The eye-position block
/// lookup `drive_atmosphere` already used for fog (`FogInputs::medium`) is reused
/// (`Client::eye_medium`) to drive `PlayerHud::underwater`, and `PlayerHud::air` drains by
/// exactly 1 every tick while submerged — java: `EntityLivingBase.onEntityUpdate`
/// (`EntityLivingBase.java:295`-`:328`) — resetting to `0` and continuing once it would go past
/// `-20`, not stopping there.
///
/// The water block rides the same connection the join used (see `join_and_script`'s own doc), so
/// exactly which `pump()` call first observes `underwater == true` is a real network race, the
/// same one `a_lightning_bolt_...` above already documents — this polls for the flip rather than
/// assuming a fixed tick offset, then reasons only about *relative* ticks from there.
#[test]
fn eye_submerged_in_water_drains_air_one_per_tick_and_resets_at_minus_twenty() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: EYE_BLOCK,
            block_state_id: WATER_BLOCK_STATE,
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    let mut air_at_flip = None;
    for _ in 0..200 {
        rig.pump();
        if rig.client.player_hud.underwater {
            air_at_flip = Some(rig.client.player_hud.air);
            break;
        }
    }
    let air_at_flip =
        air_at_flip.expect("the water block must land and flip `underwater` within 200 ticks");
    // The tick that first observes `underwater == true` already ran that same tick's decrement
    // (both come out of one `tick_air_supply` call), so this is `< 300`, never the untouched
    // default the HUD starts at.
    assert!(
        air_at_flip < 300,
        "air must already have decremented on the tick underwater flips: {air_at_flip}"
    );

    // Every following tick must drain exactly 1 while submerged.
    for steps in 1..=5 {
        rig.pump();
        assert_eq!(
            rig.client.player_hud.air,
            air_at_flip - steps,
            "air must decrement by exactly 1 per tick while submerged"
        );
    }

    // Run the clock forward to the `-20` boundary and confirm it resets to `0` there instead of
    // continuing past it, then keeps cycling rather than freezing at the reset value.
    let current_air = air_at_flip - 5;
    let ticks_to_boundary = current_air + 20;
    for _ in 0..ticks_to_boundary {
        rig.pump();
    }
    assert_eq!(
        rig.client.player_hud.air, 0,
        "must reset to 0 at -20 (java: EntityLivingBase.java:303-305), not keep counting down"
    );
    rig.pump();
    assert_eq!(
        rig.client.player_hud.air, -1,
        "the cycle must continue after the reset, not freeze at 0"
    );

    rig.finish();
}

/// Leaving the water resets `air` to `300` immediately, java: the `else { this.setAir(300); }`
/// branch — not a gradual ramp — and must also drop `underwater` back to `false`.
///
/// The two block changes are paced with a real sleep in the scripted peer (this client's
/// multi-threaded `tokio` runtime, `NetRuntime::new`, keeps making progress on that task between
/// this test's `pump()` calls) so the test can observe "went underwater and drained" as a
/// separate phase from "left the water again", rather than the two collapsing into one if they
/// landed back-to-back. Polling is wall-clock-bounded, not iteration-bounded (`pump_until`'s own
/// idiom): a fixed pump count raced the peer's `sleep` in an earlier version of this test — a
/// tight `pump()` loop can rack up hundreds of iterations well inside 500ms, exhausting the
/// budget before the second packet had even been sent.
#[test]
fn leaving_the_water_resets_air_to_full_immediately() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: EYE_BLOCK,
            block_state_id: WATER_BLOCK_STATE,
        })
        .await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        peer.send(&s2c::BlockChange {
            position: EYE_BLOCK,
            block_state_id: 0, // air
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut saw_drained = false;
    let mut left_again = false;
    while std::time::Instant::now() < deadline {
        rig.pump();
        if rig.client.player_hud.underwater && rig.client.player_hud.air < 300 {
            saw_drained = true;
        }
        if saw_drained && !rig.client.player_hud.underwater {
            left_again = true;
            break;
        }
    }
    assert!(
        saw_drained,
        "must have observed the submerged, draining state"
    );
    assert!(left_again, "must have left the water again");
    assert_eq!(
        rig.client.player_hud.air, 300,
        "must reset to 300 immediately on leaving, not ramp back up"
    );

    rig.finish();
}

/// Water breathing (potion id 13 — the same id `drive_atmosphere` already reads for fog) must
/// hold `air` exactly where it is while submerged: java's gate is on the *decrement*, not on
/// which branch runs, so air neither drains nor gets forced back to `300` while both submerged
/// and water-breathing.
///
/// The potion packet and the block change are sent back-to-back with no delay, so they are very
/// likely to land in the same client-side read and apply in the same tick — but not guaranteed
/// to; this gives a five-tick grace window before locking in the "stable" value it then holds
/// every following tick to, so a possible one-tick lead of the water block over the potion
/// packet cannot make the test racy.
#[test]
fn water_breathing_potion_prevents_air_from_draining_while_submerged() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: EYE_BLOCK,
            block_state_id: WATER_BLOCK_STATE,
        })
        .await;
        peer.send(&s2c::EntityEffect {
            entity_id: LOCAL_ENTITY_ID,
            effect_id: 13, // water breathing
            amplifier: 0,
            duration: 6000,
            show_particles: false,
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    let mut went_underwater = false;
    for _ in 0..200 {
        rig.pump();
        if rig.client.player_hud.underwater {
            went_underwater = true;
            break;
        }
    }
    assert!(went_underwater, "must have gone underwater");
    rig.pump_n(5);
    let stable = rig.client.player_hud.air;
    for _ in 0..50 {
        rig.pump();
        assert_eq!(
            rig.client.player_hud.air, stable,
            "once water breathing is active, air must stop draining entirely"
        );
    }

    rig.finish();
}

/// Invulnerable players (creative or spectator — the only two modes `WorldSettings.GameType.
/// configurePlayerCapabilities` sets `disableDamage` for) must not drain air while submerged
/// either, gated on `Abilities::invulnerable` — this client's decode of the real
/// `S39PacketPlayerAbilities` wire flag, not a re-derivation from game mode.
///
/// Same race tolerance as the water-breathing test above, for the same reason.
#[test]
fn invulnerable_players_do_not_drain_air_while_submerged() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: EYE_BLOCK,
            block_state_id: WATER_BLOCK_STATE,
        })
        .await;
        peer.send(&s2c::PlayerAbilities {
            flags: 0x01, // bit 0: invulnerable
            fly_speed: 0.05,
            walk_speed: 0.1,
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    let mut went_underwater = false;
    for _ in 0..200 {
        rig.pump();
        if rig.client.player_hud.underwater {
            went_underwater = true;
            break;
        }
    }
    assert!(went_underwater, "must have gone underwater");
    rig.pump_n(5);
    let stable = rig.client.player_hud.air;
    for _ in 0..50 {
        rig.pump();
        assert_eq!(
            rig.client.player_hud.air, stable,
            "invulnerable (creative/spectator) players must not drain air while submerged"
        );
    }

    rig.finish();
}

/// **The `max_health` third of the `S21e | S14/S16` tail row.** `S20PacketEntityProperties`
/// reaches `PlayerHud::max_health` only for the local player, and the value is the real
/// `ModifiableAttributeInstance.computeValue` three-pass order (`ModifiableAttributeInstance.
/// java:166`) — `Add` modifiers summed onto the base, then `MultiplyBase` modifiers summed and
/// added to *that* post-`Add` total (not compounded one at a time), then every `Multiply`
/// modifier scaling the running total — not a naive "read the base value" or a sequential fold
/// that would get the `MultiplyBase` group wrong.
#[test]
fn entity_properties_feeds_max_health_with_the_real_attribute_modifier_order() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        // base 20 -Add(+2,+2)-> 24 -MultiplyBase(0.25,0.25) summed onto 24, not compounded->
        // 24 + 24*0.5 = 36 -Multiply(1.0)-> 36 * 2.0 = 72.
        peer.send(&s2c::EntityProperties {
            entity_id: LOCAL_ENTITY_ID,
            properties: vec![s2c::AttributePropertySnapshot {
                key: "generic.maxHealth".into(),
                base_value: 20.0,
                modifiers: vec![
                    s2c::AttributeModifierSnapshot {
                        uuid: Uuid::from_u128(1),
                        amount: 2.0,
                        operation: s2c::AttributeModifierOperation::Add,
                    },
                    s2c::AttributeModifierSnapshot {
                        uuid: Uuid::from_u128(2),
                        amount: 2.0,
                        operation: s2c::AttributeModifierOperation::Add,
                    },
                    s2c::AttributeModifierSnapshot {
                        uuid: Uuid::from_u128(3),
                        amount: 0.25,
                        operation: s2c::AttributeModifierOperation::MultiplyBase,
                    },
                    s2c::AttributeModifierSnapshot {
                        uuid: Uuid::from_u128(4),
                        amount: 0.25,
                        operation: s2c::AttributeModifierOperation::MultiplyBase,
                    },
                    s2c::AttributeModifierSnapshot {
                        uuid: Uuid::from_u128(5),
                        amount: 1.0,
                        operation: s2c::AttributeModifierOperation::Multiply,
                    },
                ],
            }],
        })
        .await;
        // A later packet about a *different* entity must never clobber the local player's HUD
        // value — sent after the real one, so a broken `is_local_entity` filter (which would let
        // events apply in send order regardless of entity id) is caught by the final value
        // becoming this rather than staying `72.0`.
        peer.send(&s2c::EntityProperties {
            entity_id: LOCAL_ENTITY_ID + 1,
            properties: vec![s2c::AttributePropertySnapshot {
                key: "generic.maxHealth".into(),
                base_value: 999.0,
                modifiers: Vec::new(),
            }],
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.pump_n(3);
    assert_eq!(
        rig.client.player_hud.max_health, 72.0,
        "20 -Add(2,2)-> 24 -MultiplyBase(0.25,0.25) summed-> 36 -Multiply(1.0)-> 72, and the other \
         entity's 999.0 must not have leaked in"
    );

    rig.finish();
}
