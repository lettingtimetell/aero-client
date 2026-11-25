//! 1.7.10 support: the crosshair and the clicks of vanilla 1.7.10.
//!
//! `Minecraft.clickMouse`, `rightClickMouse` and `sendClickBlockToController`, the tick's
//! `EntityRenderer.getMouseOver(1.0F)` and `PlayerControllerMP.onStoppedUsingItem` as 1.7.10 runs
//! them, as `impl Client` methods. A child of `driver` (like `mods`) so it reaches `Client`'s
//! private state and helpers. Each 1.8.9 function these replace starts with a one-line hook,
//! `if self.plays_1_7_10() { return self.…_v5(); }`: a 1.8.9 session runs the unchanged 1.8.9
//! body and pays one predictable branch.
//!
//! The file sits next to `interact.rs` but is not its 1.7.10 half: it is the driver's
//! (`driver::interact_v5`, the clicks). `interact.rs`'s 1.7.10 half is `interact_rules_v5.rs`,
//! declared there as [`interact::v5`]: the rules these clicks apply (pick, ray trace, dig
//! speed, adventure exemption, the controller's dig methods).
//!
//! What 1.7.10 does differently here (`java17:` is vanilla 1.7.10, `java18:` 1.8.9):
//!
//! * the tick pick aims with the current `rotationYaw` from the eye-level `posY` and searches
//!   entities only 3.0 blocks out, with no MISS conversion (`EntityRenderer.getMouseOver`);
//! * a left click at nothing costs no 10-tick lockout (`Minecraft.clickMouse`);
//! * digging and swinging go on while an item is in use, and the swing follows the adventure
//!   exemption, not the controller's result (`Minecraft.sendClickBlockToController`);
//! * right clicks are not refused while a block is being dug, and an entity right click sends
//!   one C02 INTERACT (`Minecraft.rightClickMouse`);
//! * dig speed, adventure rules and the C08 hit offsets are 1.7.10's (`EntityPlayer`,
//!   `PlayerControllerMP`);
//! * the C07s carry 1.7.10's faces: `-1` for `resetBlockRemoving`'s ABORT and `255` for the
//!   use-item release (`PlayerControllerMP.onStoppedUsingItem`).
//!
//! The 1.7 Animations options "Using items stops mining" and "Swing at blocks while using" hook
//! the 1.8.9 bodies only, so they never run in 1.7.10 mode, where vanilla 1.7.10 already lets a
//! use start during a dig (the dig goes on) and swings at blocks while using.
//!
//! Packets: the protocol-5 conversion (`C2SPlay5::from_v47`) refuses every 1.8.9 C07, so each
//! 1.7.10 C07 is built as the protocol-5 packet ([`Client::send_digging_v5`]). The rest stays a
//! 1.8.9 `C2SPlay` whose meaning is identical (C0A, C09, C02 INTERACT/ATTACK, C08).

use super::*;
use mc_protocol::play::c2s;
use mc_protocol::v5::C2SPlay5;

/// java17: `C07PacketPlayerDigging` statuses (`PlayerControllerMP.java:144-419`).
const DIG_START: u8 = 0;
const DIG_ABORT: u8 = 1;
const DIG_STOP: u8 = 2;
const DIG_RELEASE_USE_ITEM: u8 = 5;
/// java17: `resetBlockRemoving`'s face `-1` (`PlayerControllerMP.java:187`), written as a byte.
const FACE_NONE: u8 = 0xFF;
/// java17: `onStoppedUsingItem`'s face `255` (`PlayerControllerMP.java:419`).
const FACE_USE_ITEM: u8 = 255;

/// java17: the tick-time `getPosition(1.0F)` of the local player, `(posX, posY, posZ)` with the
/// eye-level `posY = minY + 1.62F - ySize` (java17 `Entity.moveEntity`,
/// `EntityLivingBase.getPosition`). A 1.7.10 predictor carries that
/// value; one that is not (yet) a 1.7.10 predictor gives the same number with `ySize` 0.
fn eye_v5(predictor: &mc_physics::prediction::Predictor) -> DVec3 {
    predictor.v5().map_or_else(
        || {
            let pos = predictor.state.pos;
            DVec3::new(pos.x, pos.y + f64::from(1.62_f32), pos.z)
        },
        |v5| v5.eye_position(&predictor.state),
    )
}

/// What the tick's 1.7.10 pick knew beyond [`Client::hit`] (the `pick_v5` field).
#[derive(Default)]
pub(super) struct PickV5 {
    /// java17: `objectMouseOver == null` ([`interact::v5::pick`]'s `None`, or no player to pick
    /// for), which [`Client::hit`] shows as a miss. Only `clickMouse` tells the two apart
    /// (`Minecraft.java:1348-1356`); `rightClickMouse`, `sendClickBlockToController` and
    /// `middleClickMouse` treat a null target as they treat a MISS (`:1392-1395`, `:1314`,
    /// `:2261`).
    pub(super) null: bool,
    /// The picked entity as the pick saw it. 1.7.10 picks before the tick's packets
    /// (`Minecraft.java:1564-1569`), so an S13 can remove the entity before the click, and Java
    /// still clicks the `Entity` object `objectMouseOver` holds.
    entity: Option<PickedEntityV5>,
}

/// What a click on the picked entity needs once the tracker has dropped it.
#[derive(Clone)]
struct PickedEntityV5 {
    network_id: i32,
    kind: Option<mc_data::EntityType>,
    /// `EntityOtherPlayerMP`, whose client-side `attackEntityFrom` succeeds.
    remote_player: bool,
    /// `instanceof EntityLivingBase`.
    living: bool,
    /// An `EntityDragonPart` of a tracked dragon.
    dragon_part: bool,
    /// The held-independent half of its `interactWith` model.
    facts: Option<EntityFactsV5>,
}

/// The held-independent half of [`Client::entity_interaction_facts_v5`]: the entity's own facts.
#[derive(Clone)]
struct EntityFactsV5 {
    kind: Option<mc_data::EntityType>,
    /// The tracker's entity, which the local predictions act on; `None` once it is gone.
    generation: Option<mc_world::Entity>,
    facts: interact::EntityInteractionFacts,
    leashed_to_local: bool,
    command_minecart: Option<(String, String)>,
}

impl Client {
    /// java17: `EntityRenderer.getMouseOver(1.0F)`, called by `runTick` before
    /// `updateController` (`Minecraft.java:1564`), i.e. before the tick's inbound packets
    /// (`PlayerControllerMP.updateController`, `PlayerControllerMP.java:254-270`).
    ///
    /// It aims with `getLook(1.0F)`, which in 1.7.10 reads the current `rotationYaw`
    /// (`EntityLivingBase.java:1871-1885`): the yaw the frames since the last tick turned to,
    /// where 1.8.9 aims with last tick's `rotationYawHead` (java18
    /// `EntityLivingBase.java:2171-2183`). The origin is
    /// `getPosition(1.0F)`, the eye-level `posY` ([`eye_v5`]).
    pub(super) fn update_hit_result_v5(&mut self) {
        mc_core::profile_scope!("client.update_hit_result");
        let Some((eye, yaw, pitch)) =
            self.session
                .as_ref()
                .and_then(Session::predictor)
                .map(|predictor| {
                    (
                        eye_v5(predictor),
                        predictor.state.yaw,
                        predictor.state.pitch,
                    )
                })
        else {
            // java17: with no player, `getMouseOver` leaves `objectMouseOver` alone
            // (`EntityRenderer.java:293-298`): null until the first pick after the join.
            self.hit = HitResult::Miss;
            self.pick_v5 = PickV5 {
                null: true,
                entity: None,
            };
            return;
        };
        let picked = self.pick_result_at_v5(eye, yaw, pitch);
        self.hit = picked.unwrap_or(HitResult::Miss);
        let entity = match self.hit {
            HitResult::Entity(hit) => self
                .session
                .as_ref()
                .map(|session| self.picked_entity_v5(session, hit.network_id)),
            HitResult::Block(_) | HitResult::Miss => None,
        };
        self.pick_v5 = PickV5 {
            null: picked.is_none(),
            entity,
        };
    }

    /// What a click on `network_id` needs, as the tracker holds it at the pick.
    fn picked_entity_v5(&self, session: &Session, network_id: i32) -> PickedEntityV5 {
        let entities = &session.state().entities;
        let tracked = entities.get(network_id);
        PickedEntityV5 {
            network_id,
            kind: tracked.and_then(|tracked| tracked.kind.entity_type()),
            remote_player: tracked.is_some_and(|tracked| {
                interact::tracked_attack_result(tracked.kind)
                    == interact::LocalAttackResult::RemotePlayer
            }),
            living: tracked.is_some_and(|tracked| tracked.kind.is_living()),
            dragon_part: tracked.is_none() && entities.dragon_part(network_id).is_some(),
            facts: self.entity_facts_v5(session, network_id),
        }
    }

    /// [`Self::pick_result_at`] with 1.7.10's candidates and [`interact::v5::pick`]; `None` is
    /// Java's null `objectMouseOver`.
    ///
    /// java17: `getEntitiesWithinAABBExcludingEntity(renderViewEntity, box)` and the
    /// `canBeCollidedWith` test in the loop (`EntityRenderer.java:330-337`): the same chunk-order
    /// walk as 1.8.9 without its spectator filter (1.7.10 has no spectators). Border sizes are
    /// 1.7.10's; the entity boxes are whatever the tracker holds (1.7.10 sizes where its spawn
    /// path set them).
    fn pick_result_at_v5(&self, eye: DVec3, yaw: f32, pitch: f32) -> Option<HitResult> {
        let Some(session) = &self.session else {
            return None;
        };
        // java17: `getBlockReachDistance()` and `extendedReach()` read `currentGameType`
        // (`PlayerControllerMP.java:249-252`, `:438-441`).
        let creative =
            session.state().local.game_mode == mc_protocol::play::s2c::GameMode::Creative;
        let look = interact::look_vector(yaw, pitch);
        let local = session.state().local.entity_id;
        let block_reach = PlayerController::block_reach(creative);
        let entity_reach = interact::v5::entity_reach(block_reach, creative);
        let world = &session.state().world;
        let mut entities: Vec<(i32, i32, i32, interact::Pickable)> = session
            .state()
            .entities
            .iter()
            .filter(|(network_id, _)| *network_id != local)
            .filter_map(|(network_id, tracked)| {
                let kind = tracked.kind.entity_type();
                if kind.is_none() && tracked.uuid.is_none() {
                    return None;
                }
                let (width, height) = tracked.dimensions();
                let pos = tracked.position(world);
                let bounds = self.pick_bounds_v5(world, tracked, kind, pos, width, height);
                let border_size = interact::v5::collision_border_size(kind);
                if !interact::box_within_reach(bounds, border_size, eye, entity_reach) {
                    return None;
                }
                let state = interact::CollisionState {
                    dead: tracked.render.death_time >= interact::DEATH_ANIMATION_TICKS,
                    marker: false,
                };
                let chunk_x = mc_math::floor_f64(pos.x / 16.0);
                let chunk_z = mc_math::floor_f64(pos.z / 16.0);
                let slice_y = mc_math::floor_f64(pos.y / 16.0).clamp(0, 15);
                Some((
                    chunk_x,
                    chunk_z,
                    slice_y,
                    interact::Pickable {
                        network_id,
                        bounds,
                        border_size,
                        collidable: interact::can_be_collided_with(kind, state),
                    },
                ))
            })
            .collect();
        // java17: `Chunk.getEntitiesWithinAABBForEntity` appends `getParts()` after the dragon,
        // as 1.8.9 does (`Chunk.java`); see `pick_result_at`.
        for part in session.state().entities.dragon_parts() {
            if part.network_id == local
                || session
                    .state()
                    .entities
                    .get_loaded(part.network_id)
                    .is_some()
            {
                continue;
            }
            let border =
                interact::v5::collision_border_size(Some(mc_data::EntityType::EnderDragon));
            if !interact::box_within_reach(part.bounds(), border, eye, entity_reach) {
                continue;
            }
            let parent_pos = session
                .state()
                .entities
                .get(part.parent_network_id)
                .map(|parent| parent.position(world))
                .unwrap_or(part.position);
            entities.push((
                mc_math::floor_f64(parent_pos.x / 16.0),
                mc_math::floor_f64(parent_pos.z / 16.0),
                mc_math::floor_f64(parent_pos.y / 16.0).clamp(0, 15),
                interact::Pickable {
                    network_id: part.network_id,
                    bounds: part.bounds(),
                    border_size: border,
                    collidable: true,
                },
            ));
        }
        entities.sort_by_key(|(chunk_x, chunk_z, slice_y, _)| (*chunk_x, *chunk_z, *slice_y));
        interact::v5::pick(
            world,
            eye,
            look,
            block_reach,
            creative,
            session.riding(),
            entities.into_iter().map(|(_, _, _, pickable)| pickable),
        )
    }

    /// The box 1.7.10's `getMouseOver` intersects for one tracked entity.
    ///
    /// java17: a leash knot keeps the `setSize(0.5F, 0.5F)` box `setPosition` builds
    /// (`EntityLeashKnot.java:21-25`, `setDirection` is empty), unlike 1.8.9's constructor box.
    /// Item frames and paintings use `EntityHanging.setDirection`'s box
    /// ([`interact::v5::hanging_bounding_box`]) around their position, with 1.7.10's item frame
    /// size of 9 pixels (`EntityItemFrame.java:58-66`).
    fn pick_bounds_v5(
        &self,
        world: &mc_world::World,
        tracked: &crate::sync::TrackedEntity,
        kind: Option<mc_data::EntityType>,
        pos: DVec3,
        width: f32,
        height: f32,
    ) -> mc_math::Aabb {
        let fallback = || interact::size_bounding_box(pos, width, height);
        match kind {
            Some(mc_data::EntityType::ItemFrame) => world
                .ecs
                .get::<&crate::app::entrender::ObjectData>(tracked.entity)
                .ok()
                .map_or_else(fallback, |data| {
                    interact::v5::hanging_bounding_box(pos, data.0, 9, 9)
                }),
            Some(mc_data::EntityType::Painting) => world
                .ecs
                .get::<&crate::app::entrender::Hanging>(tracked.entity)
                .ok()
                .map_or_else(fallback, |hanging| {
                    interact::v5::hanging_bounding_box(
                        pos,
                        hanging.facing.horizontal_index().map_or(0, i32::from),
                        hanging.art.size_x,
                        hanging.art.size_y,
                    )
                }),
            _ => fallback(),
        }
    }

    /// [`Self::current_dig_context`] with 1.7.10's head-in-water test: `isInsideOfMaterial`
    /// at `posY + 0.12F`, 0.12 above the 1.8.9 eye (java17 `EntityPlayer.getBreakSpeed`,
    /// `EntityPlayer.java:749`).
    fn current_dig_context_v5(&self) -> interact::DigContext {
        let haste = self.active_effects.get(&3).map(|effect| effect.amplifier);
        let fatigue = self.active_effects.get(&4).map(|effect| effect.amplifier);
        let underwater = self.session.as_ref().is_some_and(|session| {
            session.predictor().is_some_and(|predictor| {
                let eye = eye_v5(predictor);
                interact::v5::is_inside_of_water(&session.state().world, eye.x, eye.y, eye.z)
            })
        });
        // java17: `EnchantmentHelper.getAquaAffinityModifier` (`EnchantmentHelper.java`), the
        // same armour scan as 1.8.9.
        let aqua_affinity = (5..=8).any(|slot| {
            self.player_container
                .slot_stack(slot)
                .is_some_and(|stack| stack_enchantment_level(stack, 6) > 0)
        });
        interact::DigContext::from_player(
            self.held_item(),
            haste,
            fatigue,
            underwater,
            aqua_affinity,
        )
    }

    /// java17: `Minecraft.clickMouse` (`Minecraft.java:1342-1384`).
    ///
    /// ```text
    /// if (leftClickCounter <= 0) {
    ///     thePlayer.swingItem();
    ///     if (objectMouseOver == null) { if (isNotCreative()) leftClickCounter = 10; }
    ///     else switch (SwitchMovingObjectType.field_152390_a[typeOfHit.ordinal()]) {
    ///         case 1: playerController.attackEntity(thePlayer, entityHit); break;
    ///         case 2: if (material == air) { if (isNotCreative()) leftClickCounter = 10; }
    ///                 else playerController.clickBlock(x, y, z, sideHit);
    ///     }
    /// }
    /// ```
    ///
    /// The switch map has ENTITY and BLOCK only (`Minecraft.java:2910-2934`), so a MISS does
    /// nothing: no 10-tick lockout (java18 sets it, `Minecraft.java:1554-1559`). A null
    /// `objectMouseOver` does cost it: the ray found no block it could not collide with, as up
    /// a ladder shaft ([`interact::v5::pick`], [`PickV5::null`]).
    pub(super) fn click_mouse_v5(&mut self) {
        let dig_context = self.current_dig_context_v5();
        self.controller.set_dig_context(dig_context);
        if !self.controller.left_click_ready() {
            return;
        }
        self.start_first_person_swing();
        if let Some(session) = &mut self.session {
            session.player_action(|out, packets| out.swing_arm(packets));
        }
        let creative = self.controller_is_creative();
        match self.hit {
            HitResult::Entity(entity) => self.attack_entity_v5(entity),
            HitResult::Block(hit) => {
                let solid = self
                    .session
                    .as_ref()
                    .is_some_and(|s| !s.state().world.block_at(hit.pos).props().is_air());
                if !solid {
                    self.controller.note_missed_swing(creative);
                    return;
                }
                let on_ground = self.player_on_ground();
                let held = self.held_item().cloned();
                let step = {
                    let Some(session) = &self.session else { return };
                    // java17: no world border (`clickBlock` has no border test).
                    self.controller.click_block_v5(
                        &session.state().world,
                        hit,
                        session.state().local.game_mode,
                        session.state().local.abilities.allow_edit,
                        held.as_ref(),
                        on_ground,
                    )
                };
                let destroy = match step {
                    DigStep::StartAndDestroy(pos, face) | DigStep::Finish(pos, face) => {
                        Some((pos, face))
                    }
                    DigStep::None | DigStep::Start(_, _) | DigStep::Abort => None,
                };
                self.apply_dig_step_v5(step);
                self.apply_local_block_click();
                if let Some((pos, face)) = destroy {
                    self.apply_local_block_destroy_v5(pos, face);
                }
            }
            HitResult::Miss if self.pick_v5.null => self.controller.note_missed_swing(creative),
            HitResult::Miss => {}
        }
    }

    /// java17: `PlayerControllerMP.attackEntity` (`PlayerControllerMP.java:373-378`): sync, C02
    /// ATTACK, then `attackTargetEntityWithCurrentItem`. The same as 1.8.9's (java18
    /// `PlayerControllerMP.java:495-506`); see the
    /// ENTITY arm of [`Self::click_mouse`], with two 1.7.10 differences:
    ///
    /// * the Knockback enchantment counts against an `EntityLivingBase` target only
    ///   (`EntityPlayer.java:1062-1066`), so a boat, minecart or dragon part hit with it while
    ///   not sprinting leaves the player's motion and sprint alone;
    /// * an entity the tick's packets removed after the pick is still attacked: Java holds its
    ///   `Entity` object, so the hit on a removed remote player still damps the motion and ends
    ///   the sprint (`STOP_SPRINTING` next tick) ([`PickV5`]).
    fn attack_entity_v5(&mut self, entity: interact::EntityHit) {
        // The reach display mod records the hit, as in 1.8.9.
        self.m1_note_attack(entity.hit_vec);
        let slot = self.held_slot();
        let enchantment_knockback = i32::from(
            self.held_item()
                .map(|stack| crate::app::abilities::enchantment_level(stack, 19))
                .unwrap_or(0),
        );
        let held = self.held_item().cloned();
        let removed = self.session.as_ref().is_some_and(|session| {
            let entities = &session.state().entities;
            entities.get(entity.network_id).is_none()
                && entities.dragon_part(entity.network_id).is_none()
        });
        let picked = self
            .pick_v5
            .entity
            .clone()
            .filter(|picked| removed && picked.network_id == entity.network_id);
        if let Some(picked) = picked {
            let knockback_level = if picked.living {
                enchantment_knockback
            } else {
                0
            };
            let hit_entity = self.session.as_mut().is_some_and(|session| {
                session.attack_removed_entity_v5(
                    entity.network_id,
                    picked.kind,
                    picked.remote_player,
                    slot,
                    knockback_level,
                    held.as_ref(),
                )
            });
            if hit_entity {
                self.apply_hit_entity_item_damage(held.as_ref());
            }
            return;
        }
        let (target_kind, living) = self
            .session
            .as_ref()
            .and_then(|session| {
                session
                    .state()
                    .entities
                    .get(entity.network_id)
                    .map(|tracked| (tracked.kind.entity_type(), tracked.kind.is_living()))
            })
            .unwrap_or((None, false));
        let knockback_level = if living { enchantment_knockback } else { 0 };
        if let Some(session) = &mut self.session {
            let hit_entity = session.attack_entity_with_knockback(
                entity.network_id,
                target_kind,
                slot,
                knockback_level,
                held.as_ref(),
            );
            if hit_entity {
                self.apply_hit_entity_item_damage(held.as_ref());
            }
        }
    }

    /// java17: `Minecraft.sendClickBlockToController` (`Minecraft.java:1309-1340`), every tick
    /// with whether the attack button is held.
    ///
    /// ```text
    /// if (!leftClick) leftClickCounter = 0;
    /// if (leftClickCounter <= 0) {
    ///     if (leftClick && objectMouseOver != null && typeOfHit == BLOCK) {
    ///         if (material != air) {
    ///             playerController.onPlayerDamageBlock(x, y, z, sideHit);
    ///             if (thePlayer.isCurrentToolAdventureModeExempt(x, y, z)) {
    ///                 effectRenderer.addBlockHitEffects(x, y, z, sideHit);
    ///                 thePlayer.swingItem();
    ///             }
    ///         }
    ///     } else playerController.resetBlockRemoving();
    /// }
    /// ```
    ///
    /// No `!isUsingItem()` gate, so blocking, eating or drawing a bow with attack held on a block
    /// still digs and swings every tick (java18 adds `!isUsingItem()`, `Minecraft.java:1503`).
    /// The swing does not depend on what
    /// `onPlayerDamageBlock` did (it returns `void`): it follows the exemption, read after the
    /// dig, so it also swings through the five-tick hit delay.
    pub(super) fn send_click_block_to_controller_v5(&mut self, left_click: bool) {
        let dig_context = self.current_dig_context_v5();
        self.controller.set_dig_context(dig_context);
        if !left_click {
            self.controller.release_attack();
        }
        if !self.controller.left_click_ready() {
            return;
        }
        let hit = match (left_click, self.hit) {
            (true, HitResult::Block(hit)) => hit,
            _ => {
                let step = self.controller.reset_block_removing_v5();
                self.apply_dig_step_v5(step);
                return;
            }
        };
        if self
            .session
            .as_ref()
            .is_none_or(|session| session.state().world.block_at(hit.pos).props().is_air())
        {
            return;
        }
        // java17: `onPlayerDamageBlock` starts with `syncCurrentPlayItem()` (`:197`).
        let slot = self.held_slot();
        if let Some(session) = &mut self.session {
            session.player_action(|out, packets| out.sync_held_item(slot, packets));
        }
        let on_ground = self.player_on_ground();
        let held = self.held_item().cloned();
        let tick = {
            let Some(session) = &self.session else { return };
            self.controller.damage_block_v5(
                &session.state().world,
                hit,
                session.state().local.game_mode,
                session.state().local.abilities.allow_edit,
                held.as_ref(),
                on_ground,
            )
        };
        if let Some(hit_sound) = tick.hit_sound {
            self.play_sound(hit_sound.play_request());
        }
        let destroy = match tick.step {
            DigStep::StartAndDestroy(pos, face) | DigStep::Finish(pos, face) => Some((pos, face)),
            DigStep::None | DigStep::Start(_, _) | DigStep::Abort => None,
        };
        self.apply_dig_step_v5(tick.step);
        self.apply_local_block_click();
        if let Some((pos, face)) = destroy {
            self.apply_local_block_destroy_v5(pos, face);
        }
        let exempt = self.session.as_ref().is_some_and(|session| {
            interact::v5::is_current_tool_adventure_mode_exempt(
                &session.state().world,
                hit.pos,
                self.held_item(),
                session.state().local.abilities.allow_edit,
            )
        });
        if exempt {
            if let Some(session) = self.session.as_ref() {
                mc_render::particle::packet::add_block_hit_effects(
                    &mut self.particles,
                    &session.state().world,
                    hit.pos,
                    hit.face,
                    &crate::app::particles::AtlasIcons,
                );
            }
            self.start_first_person_swing();
            if let Some(session) = &mut self.session {
                session.player_action(|out, packets| out.swing_arm(packets));
            }
        }
    }

    /// The dig reset of the seams that end a dig without the attack key: a screen opening
    /// (`prepare_modal_input`) and focus loss (`note_focus_changed_at`). A 1.8.9 session runs its
    /// own reset, unchanged; a 1.7.10 session runs `resetBlockRemoving` as 1.7.10 does, whose
    /// ABORT carries face -1 (java17 `PlayerControllerMP.java:183-193`) and so has to be built as
    /// the protocol-5 packet ([`Self::apply_dig_step_v5`]).
    pub(super) fn reset_block_removing_for_version(&mut self) {
        if self.plays_1_7_10() {
            let step = self.controller.reset_block_removing_v5();
            self.apply_dig_step_v5(step);
            return;
        }
        let step = self.controller.reset_block_removing();
        self.apply_dig_step(step);
    }

    /// The C07s of one controller step, in 1.7.10's order and with its faces.
    ///
    /// java17: `clickBlock` sends `ABORT(currentBlock, side)` before the START when a dig was in
    /// progress (`:150-155`); the finished dig sends STOP (`:233`); `resetBlockRemoving` sends
    /// `ABORT(currentBlock, -1)` (`:187`). The controller names the ABORT's block
    /// ([`PlayerController::take_pending_abort`]), as in 1.8.9.
    fn apply_dig_step_v5(&mut self, step: DigStep) {
        let abort = self.controller.take_pending_abort();
        match step {
            DigStep::None => {}
            DigStep::Start(pos, face) | DigStep::StartAndDestroy(pos, face) => {
                if let Some(previous) = abort {
                    self.send_digging_v5(DIG_ABORT, previous, face.index());
                }
                self.send_digging_v5(DIG_START, pos, face.index());
            }
            DigStep::Finish(pos, face) => self.send_digging_v5(DIG_STOP, pos, face.index()),
            DigStep::Abort => {
                if let Some(pos) = abort {
                    self.send_digging_v5(DIG_ABORT, pos, FACE_NONE);
                }
            }
        }
        // The 1.8.9 packet layer's record of an open START (`OutboundState::digging`) is never
        // set on a 1.7.10 connection: these C07s bypass it.
    }

    /// Queue one 1.7.10 `C07PacketPlayerDigging(status, x, y, z, face)`.
    ///
    /// java17: `writeByte(status)`, `writeInt(x)`, `writeByte(y)`, `writeInt(z)`,
    /// `writeByte(face)` (`C07PacketPlayerDigging.java:38-45`): `y` keeps its low eight bits, so
    /// `onPlayerDestroyBlock`'s `currentBlockY = -1` goes out as `0xFF`.
    fn send_digging_v5(&mut self, status: u8, pos: mc_math::BlockPos, face: u8) {
        if let Some(session) = &mut self.session {
            session.send_v5(C2SPlay5::PlayerDigging(
                mc_protocol::v5::c2s::PlayerDigging {
                    status,
                    x: pos.x,
                    y: pos.y as u8,
                    z: pos.z,
                    face,
                },
            ));
        }
    }

    /// [`Self::apply_local_block_destroy`] with 1.7.10's `onPlayerDestroyBlock`
    /// (`PlayerControllerMP.java:86-136`).
    ///
    /// The adventure gate is the 1.7.10 exemption (java17
    /// `EntityPlayer.isCurrentToolAdventureModeExempt`). 1.7.10's moving piston has no
    /// `onBlockDestroyedByPlayer` (java17 `BlockPistonMoving.java`), so nothing behind it is
    /// removed, and `ItemShears.onBlockDestroyed` has no wool case (java17
    /// `ItemShears.java:21-32`): shears that break wool are not damaged.
    fn apply_local_block_destroy_v5(&mut self, pos: mc_math::BlockPos, face: mc_math::Facing) {
        let held_slot = HOTBAR_SLOT_OFFSET + usize::from(self.held_slot());
        let held = self
            .player_container
            .slot_stack(held_slot)
            .cloned()
            .or_else(|| self.held_item().cloned());
        let Some((state, game_mode, exempt)) = self.session.as_ref().map(|session| {
            let world = &session.state().world;
            (
                world.block_at(pos),
                session.state().local.game_mode,
                interact::v5::is_current_tool_adventure_mode_exempt(
                    world,
                    pos,
                    held.as_ref(),
                    session.state().local.abilities.allow_edit,
                ),
            )
        }) else {
            return;
        };
        let creative = game_mode == mc_protocol::play::s2c::GameMode::Creative;
        // java17: `clickBlockCreative` extinguishes a fire on the clicked face first and then
        // destroys nothing (`PlayerControllerMP.java:52-58`; `World.extinguishFire`,
        // `World.java:2256-2297`).
        if creative {
            let fire_pos = pos.relative(face);
            let fire = self
                .session
                .as_ref()
                .is_some_and(|session| session.state().world.block_at(fire_pos).block().0 == 51);
            if fire {
                self.play_aux_sfx(1004, fire_pos, 0);
                if let Some(session) = &mut self.session {
                    session
                        .state_mut()
                        .world
                        .set_block(fire_pos, mc_data::BlockState::AIR);
                }
                return;
            }
        }
        if game_mode == mc_protocol::play::s2c::GameMode::Adventure && !exempt {
            return;
        }
        let sword = held.as_ref().is_some_and(|stack| {
            mc_data::item_props(mc_data::ItemId(stack.item_id as u16)).kind
                == mc_data::ItemKind::Sword
        });
        if (creative && sword) || state.props().is_air() {
            return;
        }
        let shears_on_wool = state.block().0 == 35
            && held.as_ref().is_some_and(|stack| {
                mc_data::item_props(mc_data::ItemId(stack.item_id as u16)).kind
                    == mc_data::ItemKind::Shears
            });
        let item_effect =
            interact::block_destroy_item_effect(state, held.as_ref()).filter(|_| !shears_on_wool);
        // java17: `playAuxSFX(2001, x, y, z, id + (meta << 12))` before `setBlockToAir`.
        self.play_aux_sfx(2001, pos, state.to_java_state_id());
        if let Some(session) = &mut self.session {
            session
                .state_mut()
                .world
                .set_block(pos, mc_data::BlockState::AIR);
        }
        self.controller.note_block_destroyed();
        if !creative {
            if let Some(effect) = item_effect {
                self.apply_block_destroy_item_effect(effect, held.as_ref());
            }
        }
    }

    /// java17: `Minecraft.middleClickMouse` (`Minecraft.java:2259-2377`): 1.8.9's pick block
    /// without the creative Ctrl branch that copies a tile entity's NBT into the stack (java18
    /// `Minecraft.java:2526-2529`, `:2618-2627`): 1.7.10 always goes through
    /// `InventoryPlayer.setCurrentItem` (java17 `InventoryPlayer.java:93-123`, its creative half
    /// `func_70439_a` `:198-222`, the same moves as java18's) and then, in creative, the C10.
    /// The block and entity items are the same in both (1.7.10 has no armour stand to pick).
    pub(super) fn middle_click_mouse_v5(&mut self) {
        // java17: `boolean var1 = this.thePlayer.capabilities.isCreativeMode;` (`:2263`).
        let creative = self.is_creative();
        let picked = match self.hit {
            HitResult::Block(hit) => self
                .session
                .as_ref()
                .and_then(|session| interact::pick_block_item(&session.state().world, hit.pos)),
            HitResult::Entity(entity) => self.session.as_ref().and_then(|session| {
                let tracked = session.state().entities.get_loaded(entity.network_id)?;
                let frame_item = tracked
                    .with_watched(&session.state().world, |watched| {
                        watched.frame_item().cloned()
                    })
                    .flatten();
                interact::pick_entity_item(
                    tracked.kind.entity_type(),
                    frame_item.as_ref(),
                    creative,
                )
            }),
            HitResult::Miss => None,
        };
        let Some(picked) = picked else {
            return;
        };
        if creative {
            self.apply_creative_pick(picked);
        } else {
            self.apply_survival_pick(picked);
        }
    }

    /// java17: `PlayerControllerMP.onStoppedUsingItem` (`PlayerControllerMP.java:416-421`):
    /// `syncCurrentPlayItem()`, `C07PacketPlayerDigging(5, 0, 0, 0, 255)`, then
    /// `stopUsingItem()`. Only the face differs from 1.8.9 (java18 `DOWN`,
    /// `PlayerControllerMP.java:576-581`); the bow's local release is
    /// the same (java17 `ItemBow.onPlayerStoppedUsing`, `ItemBow.java:25-82`).
    pub(super) fn release_item_use_v5(&mut self) {
        let used = self.controller.item_in_use().cloned();
        let used_action = self.controller.item_in_use_action();
        let time_left = self.controller.item_in_use_count();
        let used_slot = self
            .controller
            .item_in_use_slot()
            .unwrap_or_else(|| self.held_slot());
        let slot = self.held_slot();
        if let Some(session) = &mut self.session {
            session.player_action(|out, packets| out.sync_held_item(slot, packets));
        }
        self.send_digging_v5(
            DIG_RELEASE_USE_ITEM,
            mc_math::BlockPos::new(0, 0, 0),
            FACE_USE_ITEM,
        );
        if used_action == Some(interact::ItemUseAction::Bow) {
            if let Some(bow) = used {
                self.apply_local_bow_release(used_slot, bow, time_left);
            }
        }
        self.controller.stop_using_item();
    }

    /// java17: `Minecraft.rightClickMouse` (`Minecraft.java:1386-1449`).
    ///
    /// 1.8.9's body without its `if (!playerController.getIsHittingBlock())` wrapper: a right
    /// click is handled, and `rightClickDelayTimer` set, while a block is being dug (java18
    /// `Minecraft.java:1572`). An entity gets `interactWithEntitySendPacket` only: one C02
    /// INTERACT, no INTERACT_AT (java18 `PlayerControllerMP.java:523-529`).
    pub(super) fn right_click_mouse_v5(&mut self) {
        self.controller.note_right_click();
        let mut flag = true;
        let held = self.held_item().cloned();
        match self.hit {
            HitResult::Entity(entity) => {
                if self.entity_right_click_v5(entity, held.as_ref()) {
                    flag = false;
                }
            }
            HitResult::Block(hit) => {
                let solid = self
                    .session
                    .as_ref()
                    .is_some_and(|s| !s.state().world.block_at(hit.pos).props().is_air());
                if solid {
                    let equip_before = self.held_slot_object();
                    if self.on_player_right_click_v5(hit, held.as_ref()) {
                        flag = false;
                        self.start_first_person_swing();
                        if let Some(session) = &mut self.session {
                            session.player_action(|out, packets| out.swing_arm(packets));
                        }
                    }
                    // java17: `else if (var2.stackSize != var6 || isInCreativeMode())
                    // itemRenderer.resetEquippedProgress()` (`Minecraft.java:1428-1435`).
                    if crate::app::first_person_equip::block_click_resets_equip(
                        equip_before,
                        self.held_slot_object(),
                        self.controller_is_creative(),
                    ) {
                        if let Some(render) = &self.render {
                            render
                                .first_person
                                .with(|phase| phase.render_state.reset_equipped_progress());
                        }
                    }
                    if held.is_none() {
                        return;
                    }
                }
            }
            HitResult::Miss => {}
        }
        // java17: `if (var1) { var7 = getCurrentItem(); if (var7 != null && sendUseItem(..))
        // resetEquippedProgress2(); }` (`Minecraft.java:1438-1447`), re-reading the slot.
        if flag {
            if let Some(stack) = self.held_item().cloned() {
                self.send_use_item(Some(stack));
            }
        }
    }

    /// java17: `PlayerControllerMP.interactWithEntitySendPacket` (`:380-385`): sync, one C02
    /// INTERACT, then `EntityPlayer.interactWith`, whose result says whether the click was
    /// consumed (no fallback C08).
    ///
    /// The local `interactWith` model is 1.8.9's classifier, which matches 1.7.10 on the client
    /// for every interaction the two versions share (`EntityPlayer.interactWith`, java17
    /// `:992-1035`, java18 `:1220-1273`, minus spectators), with these 1.7.10 differences:
    /// a command minecart opens its editor in every game mode (java17
    /// `EntityMinecartCommandBlock.java:100-107`), and there is no armour stand and no
    /// INTERACT_AT. The breeding, milking, lead and horse-food differences are in
    /// [`interact::v5::entity_classifier_inputs`].
    fn entity_right_click_v5(
        &mut self,
        entity: interact::EntityHit,
        held: Option<&mc_protocol::ItemStack>,
    ) -> bool {
        let slot = self.held_slot();
        let facts = self
            .session
            .as_ref()
            .and_then(|session| self.entity_interaction_facts_v5(session, entity, held));
        if let Some(session) = &mut self.session {
            session.player_action(|out, packets| {
                out.sync_held_item(slot, packets);
                packets.push(mc_protocol::play::C2SPlay::UseEntity(c2s::UseEntity {
                    entity_id: entity.network_id,
                    action: c2s::UseEntityAction::Interact,
                }));
                // java17: `EntityCreeper.interact` ignites from flint and steel and calls
                // `player.swingItem()` (`EntityCreeper.java:210-228`), after the C02.
                if facts.as_ref().is_some_and(|facts| facts.swing_after) {
                    out.swing_arm(packets);
                }
            });
        }
        // Vanilla holds a reference to `entityHit` (an entity removed since the pick answers
        // from the pick's facts); one this port never resolved keeps the conservative consumed
        // answer (no fallback C08), as 1.8.9 does.
        let Some(facts) = facts else {
            return true;
        };
        if let Some(session) = &mut self.session {
            if let Some((horse_id, generation)) = facts.horse_mount_intent {
                session.note_horse_mount_intent(horse_id, generation);
            }
        }
        if let Some((pig_entity, prediction)) = facts.saddle_prediction {
            self.apply_saddle_prediction(pig_entity, slot, prediction, held);
        }
        if let Some((network_id, generation, prediction)) = facts.name_tag_prediction {
            self.apply_name_tag_prediction(network_id, generation, slot, prediction, held);
        }
        if let Some((network_id, generation, prediction)) = facts.local_entity_prediction {
            self.apply_entity_local_prediction(network_id, generation, slot, prediction, held);
        }
        if let Some((command, last_output_json)) = facts.command_minecart.clone() {
            self.open_command_minecart_screen(entity.network_id, command, last_output_json);
        }
        if facts.swing_after {
            self.start_first_person_swing();
        }
        // java17: `rightClickMouse`'s final `if (var7 != null)`: an empty hand on a command
        // minecart has no fallback packet either way.
        facts.consumed || (facts.command_minecart.is_some() && held.is_none())
    }
}

/// What `EntityPlayer.interactWith` does on the client for one entity right click, gathered
/// before the packets go out (the closure that sends them borrows the session mutably).
struct EntityInteractionV5 {
    consumed: bool,
    swing_after: bool,
    saddle_prediction: Option<(mc_world::Entity, interact::SaddleInteraction)>,
    name_tag_prediction: Option<(i32, mc_world::Entity, interact::NameTagInteraction)>,
    local_entity_prediction: Option<(i32, mc_world::Entity, interact::EntityLocalInteraction)>,
    horse_mount_intent: Option<(i32, mc_world::Entity)>,
    command_minecart: Option<(String, String)>,
}

impl EntityInteractionV5 {
    /// `interactWith` returned false and predicts nothing locally.
    fn declined(swing_after: bool, command_minecart: Option<(String, String)>) -> Self {
        Self {
            consumed: false,
            swing_after,
            saddle_prediction: None,
            name_tag_prediction: None,
            local_entity_prediction: None,
            horse_mount_intent: None,
            command_minecart,
        }
    }
}

impl Client {
    /// The client-side `EntityPlayer.interactWith` model for one entity right click: the
    /// entity's facts ([`Self::entity_facts_v5`]), then 1.8.9's classifiers with 1.7.10's inputs
    /// ([`interact::v5::entity_classifier_inputs`]) and its child breeding rule
    /// ([`interact::v5::child_breeding_declined`]).
    ///
    /// An entity the tick's packets removed after the pick is still the one Java interacts with
    /// (`objectMouseOver.entityHit`): its facts are the pick's ([`PickV5`]), and only the local
    /// predictions, which act on the tracked entity, are left out. `None`, an entity this port
    /// never resolved, keeps the conservative consumed answer.
    fn entity_interaction_facts_v5(
        &self,
        session: &Session,
        entity: interact::EntityHit,
        held: Option<&mc_protocol::ItemStack>,
    ) -> Option<EntityInteractionV5> {
        // java17: an `EntityDragonPart` has no `interactFirst` and is not an
        // `EntityLivingBase`, so `interactWith` returns false (`EntityPlayer.java:992-1035`).
        if session
            .state()
            .entities
            .dragon_part(entity.network_id)
            .is_some()
        {
            return Some(EntityInteractionV5::declined(false, None));
        }
        let EntityFactsV5 {
            kind,
            generation,
            facts,
            leashed_to_local,
            command_minecart,
        } = match self.entity_facts_v5(session, entity.network_id) {
            Some(live) => live,
            None => {
                let picked = self
                    .pick_v5
                    .entity
                    .as_ref()
                    .filter(|picked| picked.network_id == entity.network_id)?;
                if picked.dragon_part {
                    return Some(EntityInteractionV5::declined(false, None));
                }
                EntityFactsV5 {
                    generation: None,
                    ..picked.facts.clone()?
                }
            }
        };
        let swing_after = kind == Some(mc_data::EntityType::Creeper)
            && held.is_some_and(|stack| stack.item_id == 259);
        let base_state = facts.base;
        let (held, facts) = interact::v5::entity_classifier_inputs(kind, held, facts);
        let held = held.as_ref();
        if interact::v5::child_breeding_declined(kind, held, facts) {
            return Some(EntityInteractionV5::declined(swing_after, command_minecart));
        }
        let name_tag_override_reached =
            interact::name_tag_override_reached(kind, facts, leashed_to_local);
        let consumed =
            interact::entity_interaction_consumed_with_facts(kind, held, facts).unwrap_or(true);
        let horse_mount_intent = generation
            .filter(|_| {
                kind == Some(mc_data::EntityType::EntityHorse)
                    && interact::horse_mount_intent(held, facts)
            })
            .map(|generation| (entity.network_id, generation));
        let saddle_prediction = generation.zip(
            interact::saddle_item_interaction(kind, held, base_state)
                .filter(|prediction| prediction.set_saddled || prediction.decrement_stack),
        );
        let name_tag_prediction = generation.and_then(|generation| {
            name_tag_override_reached
                .then(|| interact::name_tag_interaction(kind, held, base_state.creative))
                .flatten()
                .map(|prediction| (entity.network_id, generation, prediction))
        });
        let local_entity_prediction = generation.and_then(|generation| {
            interact::entity_local_interaction(kind, held, facts)
                .map(|prediction| (entity.network_id, generation, prediction))
        });
        Some(EntityInteractionV5 {
            consumed,
            swing_after,
            saddle_prediction,
            name_tag_prediction,
            local_entity_prediction,
            horse_mount_intent,
            command_minecart,
        })
    }

    /// The held-independent half of the client's `interactWith` model for a tracked entity,
    /// read as the 1.8.9 right-click arm reads it (`right_click_mouse`); `None` once the tracker
    /// has dropped the entity.
    fn entity_facts_v5(&self, session: &Session, network_id: i32) -> Option<EntityFactsV5> {
        let state = session.state();
        let tracked = state.entities.get_loaded(network_id)?;
        let world = &state.world;
        let kind = tracked.kind.entity_type();
        // java17: `EntityMinecartCommandBlock.interactFirst` opens the editor on the client in
        // every game mode (`EntityMinecartCommandBlock.java:100-105`) and returns false.
        let command_minecart = if kind == Some(mc_data::EntityType::MinecartCommandBlock) {
            tracked
                .with_watched(world, |watched| {
                    watched.command_minecart().map(|command| {
                        (
                            command.command.to_owned(),
                            command.last_output_json.to_owned(),
                        )
                    })
                })
                .flatten()
        } else {
            None
        };
        let child = tracked
            .with_watched(world, |watched| watched.is_child())
            .unwrap_or(false);
        let pig_saddled = tracked
            .with_watched(world, |watched| watched.is_saddled())
            .unwrap_or(false);
        let sheep_sheared = tracked
            .with_watched(world, |watched| watched.is_sheared())
            .unwrap_or(false);
        let sheep_fleece_metadata = (kind == Some(mc_data::EntityType::Sheep)).then(|| {
            tracked
                .with_watched(world, |watched| {
                    watched
                        .fleece_colour()
                        .map(|colour| colour.metadata() as u8)
                        .unwrap_or(0)
                })
                .unwrap_or(0)
        });
        let creative = state.local.game_mode == mc_protocol::play::s2c::GameMode::Creative;
        let base_state = interact::EntityInteractionState {
            child,
            creative,
            pig_saddled,
            sheep_sheared,
        };
        let rideable_occupancy = if matches!(
            kind,
            Some(mc_data::EntityType::MinecartRideable | mc_data::EntityType::EntityHorse)
        ) {
            if session.riding() == Some(network_id) {
                interact::RideableOccupancy::Player
            } else {
                state
                    .entities
                    .iter()
                    .find(|(_, rider)| rider.riding_entity() == Some(network_id))
                    .map_or(interact::RideableOccupancy::Empty, |(_, rider)| {
                        if rider.kind == crate::sync::entities::TrackedKind::Player {
                            interact::RideableOccupancy::Player
                        } else {
                            interact::RideableOccupancy::Other
                        }
                    })
            }
        } else {
            interact::RideableOccupancy::Unknown
        };
        let (
            horse_tame,
            horse_type,
            horse_chested,
            wolf_tamed,
            wolf_angry,
            wolf_hurt,
            wolf_collar_dye_damage,
            ocelot_tamed,
            villager_alive,
            tame_owner,
        ) = tracked
            .with_watched(world, |watched| {
                let horse = watched.horse();
                let tame_owner = watched
                    .owner_id()
                    .and_then(|owner| owner.parse::<uuid::Uuid>().ok())
                    .is_some_and(|owner| owner == state.local.uuid);
                (
                    horse.map(|horse| horse.tame),
                    horse.map(|horse| horse.horse_type),
                    horse.map(|horse| horse.chested),
                    Some(watched.is_tamed()),
                    Some(watched.is_angry()),
                    watched.wolf_health().map(|health| health < 20.0),
                    watched.collar_colour().map(|colour| colour.dye_damage()),
                    Some(watched.is_tamed()),
                    watched.health().map(|health| health > 0.0),
                    Some(tame_owner),
                )
            })
            .unwrap_or_default();
        let horse_hurt = (kind == Some(mc_data::EntityType::EntityHorse))
            .then(|| {
                let max_health =
                    tracked.attribute_value_any(&["generic.maxHealth", "maxHealth", "Max Health"]);
                tracked
                    .with_watched(world, |watched| {
                        horse_hurt_from_health(watched.health(), max_health)
                    })
                    .flatten()
            })
            .flatten();
        let facts = interact::EntityInteractionFacts {
            base: base_state,
            sheep_fleece_metadata,
            rideable_occupancy,
            sneaking: session
                .predictor()
                .map(|predictor| predictor.state.sneaking),
            local_player_entity_id: Some(state.local.entity_id),
            leashed_to: tracked.leashed_to(),
            tame_owner,
            lead_creative: Some(state.local.abilities.creative_mode),
            horse_tame,
            horse_type,
            horse_chested,
            wolf_tamed,
            wolf_angry,
            wolf_hurt,
            wolf_collar_dye_damage,
            ocelot_tamed,
            villager_alive,
            villager_trading: Some(false),
            animal_breeding_ready: tracked.animal_breeding_ready(world),
            horse_hurt,
            horse_temper: tracked.horse_temper(),
            furnace_minecart_cart_position: world
                .ecs
                .get::<&mc_world::entity::Position>(tracked.entity)
                .ok()
                .map(|position| position.0),
            furnace_minecart_player_position: session
                .predictor()
                .map(|predictor| predictor.state.pos),
            furnace_minecart_creative: Some(state.local.abilities.creative_mode),
        };
        Some(EntityFactsV5 {
            kind,
            generation: Some(tracked.entity),
            facts,
            leashed_to_local: tracked.leashed_to() == Some(state.local.entity_id),
            command_minecart,
        })
    }

    /// java17: `PlayerControllerMP.onPlayerRightClick` (`PlayerControllerMP.java:296-342`).
    ///
    /// ```text
    /// syncCurrentPlayItem();
    /// float f = (float)hitVector.xCoord - (float)x; ...
    /// if ((!player.isSneaking() || player.getHeldItem() == null)
    ///     && block.onBlockActivated(world, x, y, z, player, side, f, f1, f2)) flag = true;
    /// if (!flag && stack != null && stack.getItem() instanceof ItemBlock
    ///     && !itemBlock.func_150936_a(world, x, y, z, side, player, stack)) return false;
    /// addToSendQueue(new C08PacketPlayerBlockPlacement(x, y, z, side, getCurrentItem(),
    ///                                                  f, f1, f2));
    /// if (flag) return true; else if (stack == null) return false;
    /// else return stack.tryPlaceItemIntoWorld(player, world, x, y, z, side, f, f1, f2);
    /// ```
    ///
    /// [`Self::on_player_right_click`]'s shape without the world border and spectator gates,
    /// with 1.7.10's float hit offsets (`PlayerControllerMP.java:299-301`), block activations,
    /// `ItemBlock` gate and item
    /// callbacks. `canPlayerEdit` passes for any held stack in 1.7.10
    /// ([`interact::v5::can_player_edit`]), so every item callback gets `allow_edit = true`.
    fn on_player_right_click_v5(
        &mut self,
        hit: interact::BlockHit,
        held: Option<&mc_protocol::ItemStack>,
    ) -> bool {
        let slot = self.held_slot();
        let cursor = interact::v5::cursor(&hit);
        let Some(allow_edit) = self
            .session
            .as_ref()
            .map(|session| session.state().local.abilities.allow_edit)
        else {
            return false;
        };
        if let Some(session) = &mut self.session {
            session.player_action(|out, packets| out.sync_held_item(slot, packets));
        }
        let sneaking = self
            .session
            .as_ref()
            .and_then(Session::predictor)
            .is_some_and(|predictor| predictor.state.sneaking);
        let consumed = if !sneaking || held.is_none() {
            self.activate_block_v5(hit, held, allow_edit)
        } else {
            false
        };

        if !consumed {
            if let Some(stack) = held {
                if interact::v5::placed_block(stack).is_some() {
                    let blocking = self.placement_blockers();
                    let Some(session) = &self.session else {
                        return false;
                    };
                    if !interact::v5::can_place_block_on_side(
                        &session.state().world,
                        hit.pos,
                        hit.face,
                        stack,
                        &blocking,
                    ) {
                        return false;
                    }
                }
            }
        }

        // java17: `player.inventory.getCurrentItem()`, read after the block's activation.
        let stack = self.held_item().cloned();
        if let Some(session) = &mut self.session {
            session.player_action(move |out, packets| {
                out.place_block(hit.pos, hit.face, cursor, stack, slot, packets);
            });
        }
        if consumed {
            return true;
        }
        let Some(stack) = held else {
            return false;
        };
        self.item_use_on_block_v5(hit, stack, cursor, slot)
    }

    /// java17: `Block.onBlockActivated` on the client, as `onPlayerRightClick` calls it. The
    /// 1.8.9 activation chain with 1.7.10's differences (java17 citations by block):
    ///
    /// * command block: consumed in every game mode, the editor opens
    ///   (`BlockCommandBlock.java:64-74`; java18 creative only, `CommandBlockLogic.java:237-252`);
    /// * sign and daylight detector: no override, not consumed (`Block.java:819-822`);
    /// * repeater and comparator: no edit-permission test (`BlockRedstoneRepeater.java:22-29`,
    ///   `BlockRedstoneComparator.java:132-143`);
    /// * TNT: flint and steel only (`BlockTNT.java:87-100`);
    /// * jukebox: consumed when it holds a record, with no client-side change
    ///   (`BlockJukebox.java:32-43`).
    fn activate_block_v5(
        &mut self,
        hit: interact::BlockHit,
        held: Option<&mc_protocol::ItemStack>,
        allow_edit: bool,
    ) -> bool {
        let Some(block) = self
            .session
            .as_ref()
            .map(|session| session.state().world.block_at(hit.pos))
        else {
            return false;
        };
        let player_yaw = self
            .session
            .as_ref()
            .and_then(Session::predictor)
            .map_or(0.0, |predictor| predictor.state.yaw);
        match block.block().0 {
            137 => {
                let state = self
                    .session
                    .as_mut()
                    .and_then(|session| session.state_mut().world.tile_entity_or_create(hit.pos))
                    .and_then(|tile| tile.command_block().cloned());
                if let Some(state) = state {
                    self.open_command_block_screen(hit.pos, state);
                }
                return true;
            }
            63 | 68 | 151 => return false,
            46 => {
                if held.is_some_and(|stack| stack.item_id == 259) {
                    self.activate_tnt_v5(hit.pos, held);
                    return true;
                }
                return false;
            }
            84 => return block.meta() != 0,
            _ => {}
        }
        if let Some(activation) = self.session.as_ref().and_then(|session| {
            interact::activate_remote_block(&session.state().world, hit.pos, allow_edit)
        }) {
            return activation.consumed;
        }
        if let Some(activation) = self
            .session
            .as_mut()
            .and_then(|session| interact::activate_button(&mut session.state_mut().world, hit.pos))
        {
            return activation.consumed;
        }
        if self.activate_cake(hit.pos) {
            return true;
        }
        if let Some(activation) = self
            .session
            .as_mut()
            .and_then(|session| session.state_mut().world.activate_door(hit.pos))
        {
            if let Some(effect) = activation.effect {
                self.play_aux_sfx(effect.id, effect.pos, effect.data);
            }
            return activation.consumed;
        }
        if let Some(activation) = self.session.as_mut().and_then(|session| {
            session
                .state_mut()
                .world
                .activate_fence_gate(hit.pos, player_yaw)
        }) {
            if let Some(effect) = activation.effect {
                self.play_aux_sfx(effect.id, effect.pos, effect.data);
            }
            return activation.consumed;
        }
        if let Some(activation) = self
            .session
            .as_mut()
            .and_then(|session| session.state_mut().world.activate_trapdoor(hit.pos))
        {
            if let Some(effect) = activation.effect {
                self.play_aux_sfx(effect.id, effect.pos, effect.data);
            }
            return activation.consumed;
        }
        if let Some(activation) = self.session.as_mut().and_then(|session| {
            interact::activate_repeater(&mut session.state_mut().world, hit.pos, true)
        }) {
            return activation.consumed;
        }
        if let Some(activation) = self.session.as_mut().and_then(|session| {
            interact::activate_comparator(&mut session.state_mut().world, hit.pos, true)
        }) {
            return activation.consumed;
        }
        if let Some(activation) = self.session.as_mut().and_then(|session| {
            interact::activate_flower_pot(&mut session.state_mut().world, hit.pos, held)
        }) {
            if activation.consumed && !self.is_creative() {
                self.decrement_selected_stack_v5(held);
            }
            return activation.consumed;
        }
        if let Some(activation) = self
            .session
            .as_ref()
            .and_then(|session| interact::activate_dragon_egg(&session.state().world, hit.pos))
        {
            if activation.consumed {
                self.activate_dragon_egg_particles(hit.pos);
            }
            return activation.consumed;
        }
        if matches!(block.block().0, 73 | 74) {
            // java17: `BlockRedstoneOre.onBlockActivated` lights the ore and spawns its
            // particles, then returns `super` (false) (`BlockRedstoneOre.java:47-51`).
            let seed = (self.tick as i64)
                .wrapping_mul(31)
                .wrapping_add(i64::from(hit.pos.x).wrapping_mul(73_428_767))
                .wrapping_add(i64::from(hit.pos.y).wrapping_mul(912_271))
                .wrapping_add(i64::from(hit.pos.z).wrapping_mul(43_829));
            let mut random = mc_math::JavaRandom::new(seed);
            let particles = self.activate_local_redstone_ore(hit.pos, &mut random);
            for (particle, position, velocity) in particles {
                self.spawn_effect_particle(particle, position, velocity, &[]);
            }
        }
        false
    }

    /// java17: `BlockTNT.onBlockActivated` with flint and steel (`BlockTNT.java:87-100`): the
    /// block goes to air and, outside creative, the flint and steel takes one damage, before the
    /// targeted C08 is built.
    fn activate_tnt_v5(&mut self, pos: mc_math::BlockPos, held: Option<&mc_protocol::ItemStack>) {
        if let Some(session) = &mut self.session {
            session
                .state_mut()
                .world
                .set_block(pos, mc_data::BlockState::AIR);
        }
        if self.is_creative() {
            return;
        }
        let inventory_slot = HOTBAR_SLOT_OFFSET + usize::from(self.held_slot());
        if let Some(mut stack) = self
            .player_container
            .slot_stack(inventory_slot)
            .cloned()
            .or_else(|| held.cloned())
        {
            self.damage_item_stack(&mut stack, 1);
            // Java keeps the object through the following C08 even at zero size.
            self.player_container
                .set_slot_contents_in_place(inventory_slot, Some(stack));
            self.refresh_hotbar();
            self.mirror_creative_player_inventory();
        }
    }

    /// The selected survival stack loses one item (a placed block, a used seed, a flower put in
    /// a pot), as Java's `--stackSize` on the held object; an empty stack leaves the slot.
    fn decrement_selected_stack_v5(&mut self, held: Option<&mc_protocol::ItemStack>) {
        let inventory_slot = HOTBAR_SLOT_OFFSET + usize::from(self.held_slot());
        let Some(mut current) = self
            .player_container
            .slot_stack(inventory_slot)
            .cloned()
            .or_else(|| held.cloned())
        else {
            return;
        };
        current.count -= 1;
        let replacement = (current.count > 0).then_some(current);
        self.player_container
            .set_slot_contents_in_place(inventory_slot, replacement.clone());
        if let Ui::Creative(screen) = &mut self.ui {
            screen.player.set_slot_contents(inventory_slot, replacement);
        }
        self.refresh_hotbar();
    }

    /// java17: `ItemStack.tryPlaceItemIntoWorld` -> `Item.onItemUse` on the client, after the
    /// targeted C08: whether it returned true (the swing, and no fallback C08) and its local
    /// prediction. The 1.8.9 item callbacks are reused where 1.7.10 agrees, with
    /// `allow_edit = true` ([`interact::v5::can_player_edit`]) and these 1.7.10 rules:
    ///
    /// * `ItemBlock` places into the cell [`interact::v5::item_block_target`] chooses, refuses
    ///   a solid block at `y == 255`, and places exactly when 1.7.10's `canPlaceEntityOnSide`
    ///   passes (`ItemBlock.java:42-92`, [`interact::v5::item_block_use`]);
    /// * `ItemDoor` places one cell above a top face only ([`interact::v5::door_item_use`]);
    /// * `ItemReed` returns true even when nothing can be placed (`ItemReed.java:70-87`);
    /// * `ItemRedstone` places into a snow layer or the empty next cell, and returns true once
    ///   it gets that far (`ItemRedstone.java:19-70`);
    /// * `ItemSkull` returns true for any click but a bottom face on a solid block
    ///   (`ItemSkull.java:36-117`).
    fn item_use_on_block_v5(
        &mut self,
        hit: interact::BlockHit,
        stack: &mc_protocol::ItemStack,
        cursor: DVec3,
        slot: u8,
    ) -> bool {
        let allow_edit = interact::v5::can_player_edit(
            Some(stack),
            self.session
                .as_ref()
                .is_some_and(|session| session.state().local.abilities.allow_edit),
        );
        let yaw = self
            .session
            .as_ref()
            .and_then(Session::predictor)
            .map_or(0.0, |predictor| f64::from(predictor.state.yaw));
        let mut hoe_use_sound = None;
        let item_block = interact::v5::placed_block(stack).is_some();
        let reed = interact::item_reed_block(stack).is_some();
        let special_result = if stack.item_id == interact::LEAD_ITEM_ID {
            self.session
                .as_ref()
                .map(|session| interact::lead_item_use(&session.state().world, hit.pos, stack))
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if matches!(stack.item_id, 290..=294) {
            let result = self
                .session
                .as_ref()
                .map(|session| {
                    interact::hoe_item_use(
                        &session.state().world,
                        hit.pos,
                        hit.face,
                        stack,
                        allow_edit,
                    )
                })
                .unwrap_or(interact::HoeUseResult::Rejected);
            hoe_use_sound = interact::hoe_item_use_sound(result);
            match result {
                interact::HoeUseResult::NotApplicable => interact::BlockUseResult::NotApplicable,
                interact::HoeUseResult::Rejected => interact::BlockUseResult::Rejected,
                interact::HoeUseResult::Consumed { .. } => interact::BlockUseResult::Consumed {
                    placement: None,
                    decrement_stack: false,
                },
            }
        } else if stack.item_id == 259 {
            self.session
                .as_ref()
                .map(|session| {
                    interact::flint_and_steel_item_use(
                        &session.state().world,
                        hit.pos,
                        hit.face,
                        stack,
                        allow_edit,
                    )
                })
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if stack.item_id == 351 {
            self.session
                .as_ref()
                .map(|session| {
                    interact::dye_item_use(
                        &session.state().world,
                        hit.pos,
                        hit.face,
                        stack,
                        allow_edit,
                    )
                })
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if matches!(stack.item_id, 321 | 389) {
            let hanging_entities = self.hanging_entity_boxes();
            self.session
                .as_ref()
                .map(|session| {
                    interact::hanging_item_use(
                        &session.state().world,
                        hit.pos,
                        hit.face,
                        stack,
                        allow_edit,
                        &hanging_entities,
                    )
                })
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if matches!(stack.item_id, 328 | 342 | 343 | 407 | 408) {
            self.session
                .as_ref()
                .map(|session| interact::minecart_item_use(&session.state().world, hit.pos, stack))
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if stack.item_id == 331 {
            self.session
                .as_ref()
                .map(|session| redstone_item_use_v5(&session.state().world, hit.pos, hit.face))
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if matches!(stack.item_id, 295 | 361 | 362 | 372 | 391 | 392) {
            self.session
                .as_ref()
                .map(|session| {
                    interact::seeds_item_use(
                        &session.state().world,
                        hit.pos,
                        hit.face,
                        stack,
                        allow_edit,
                    )
                })
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if matches!(stack.item_id, 324 | 330) {
            self.session
                .as_ref()
                .map(|session| {
                    interact::v5::door_item_use(
                        &session.state().world,
                        hit.pos,
                        hit.face,
                        yaw,
                        stack,
                    )
                })
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if matches!(stack.item_id, 44 | 78 | 126) {
            let blocking = self.placement_blockers();
            self.session
                .as_ref()
                .map(|session| {
                    let world = &session.state().world;
                    match interact::slab_item_use(
                        world, hit.pos, hit.face, stack, allow_edit, &blocking,
                    ) {
                        interact::BlockUseResult::NotApplicable => interact::snow_item_use(
                            world, hit.pos, hit.face, stack, allow_edit, &blocking,
                        ),
                        result => result,
                    }
                })
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if reed || item_block {
            let blocking = self.placement_blockers();
            self.session
                .as_ref()
                .and_then(|session| {
                    let world = &session.state().world;
                    let predictor = session.predictor()?;
                    let (target, side) = interact::v5::item_block_target(world, hit.pos, hit.face);
                    // java17: `else if (y == 255 && block.getMaterial().isSolid()) return false;`
                    if item_block
                        && target.y == 255
                        && interact::v5::placed_block(stack)
                            .is_some_and(|block| block.default_state().props().material.is_solid())
                    {
                        return Some(interact::BlockUseResult::Rejected);
                    }
                    // The 1.8.9 callback steps past the clicked block unless 1.8.9's
                    // `isReplaceable` keeps it. The two rules differ only for a double plant,
                    // which 1.8.9 replaces and 1.7.10 steps past, so the callback gets 1.7.10's
                    // cell there (and the forced side up on a one-layer snow layer).
                    let pos = if world.block_at(hit.pos).block().0 == 175 {
                        target
                    } else {
                        hit.pos
                    };
                    let result = interact::item_block_delegate_use_with_pose(
                        world,
                        pos,
                        side,
                        cursor.y,
                        yaw,
                        stack,
                        allow_edit,
                        &blocking,
                        predictor.state.pos,
                        predictor.state.eye_pos().y,
                    );
                    // 1.7.10's `canPlaceEntityOnSide` decides whether the block is placed.
                    let result = if item_block {
                        interact::v5::item_block_use(
                            world, hit.pos, hit.face, cursor.y, yaw, stack, &blocking, result,
                        )
                    } else {
                        result
                    };
                    Some(match result {
                        // java17: `ItemReed.onItemUse` returns true whether or not it placed.
                        interact::BlockUseResult::Rejected if reed && !item_block => {
                            interact::BlockUseResult::Consumed {
                                placement: None,
                                decrement_stack: false,
                            }
                        }
                        result => result,
                    })
                })
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else if matches!(stack.item_id, 2256..=2267) {
            self.session
                .as_ref()
                .map(|session| interact::record_item_use(&session.state().world, hit.pos, stack))
                .unwrap_or(interact::BlockUseResult::NotApplicable)
        } else {
            interact::BlockUseResult::NotApplicable
        };

        let callback_result = match special_result {
            interact::BlockUseResult::Delegate { pos } => {
                let blocking = self.placement_blockers();
                self.session
                    .as_ref()
                    .map(|session| {
                        interact::item_block_delegate_use_with_pose(
                            &session.state().world,
                            pos,
                            hit.face,
                            cursor.y,
                            yaw,
                            stack,
                            allow_edit,
                            &blocking,
                            session
                                .predictor()
                                .map_or(DVec3::ZERO, |predictor| predictor.state.pos),
                            session
                                .predictor()
                                .map_or(0.0, |predictor| predictor.state.eye_pos().y),
                        )
                    })
                    .unwrap_or(interact::BlockUseResult::NotApplicable)
            }
            result => result,
        };

        let placements: Vec<interact::BlockUsePlacement>;
        let decrement;
        match callback_result {
            interact::BlockUseResult::Rejected => return false,
            interact::BlockUseResult::NotApplicable | interact::BlockUseResult::Delegate { .. } => {
                return self.item_use_without_placement_v5(hit, stack);
            }
            interact::BlockUseResult::Consumed {
                placement,
                decrement_stack,
            } => {
                placements = placement.into_iter().collect();
                decrement = decrement_stack;
            }
            interact::BlockUseResult::ConsumedPair {
                placements: pair,
                decrement_stack,
            } => {
                placements = pair.to_vec();
                decrement = decrement_stack;
            }
            interact::BlockUseResult::ConsumedMany {
                placements: many,
                decrement_stack,
            } => {
                placements = many;
                decrement = decrement_stack;
            }
        }
        // Client `World.playSoundEffect` calls reach the empty sink (the server's S29 is the
        // audible copy), but the flint's pitch operand still draws from its Random.
        if let Some(sound) = hoe_use_sound.take() {
            Self::render_global_play_sound(sound);
        }
        if stack.item_id == 259 {
            let target = placements
                .first()
                .map_or(hit.pos.relative(hit.face), |placement| placement.pos);
            let pitch = self.item_use_rng.next_f32() * 0.4 + 0.8;
            Self::render_global_play_sound(mc_audio::PlayRequest::at(
                mc_assets::ResourceLocation::minecraft("fire.ignite"),
                1.0,
                pitch,
                DVec3::new(
                    f64::from(target.x) + 0.5,
                    f64::from(target.y) + 0.5,
                    f64::from(target.z) + 0.5,
                ),
            ));
        }
        // No client neighbour reaction: `setBlock` notifies only on the server.
        if let Some(session) = &mut self.session {
            for placement in &placements {
                session
                    .state_mut()
                    .world
                    .set_block(placement.pos, placement.state);
            }
        }
        // `PlayerControllerMP` restores the creative copy after `onItemUse`.
        if decrement && !self.is_creative() {
            self.decrement_selected_stack_v5(Some(stack));
        }
        if stack.item_id == 259 && !self.is_creative() {
            let inventory_slot = HOTBAR_SLOT_OFFSET + usize::from(slot);
            if let Some(mut current) = self
                .player_container
                .slot_stack(inventory_slot)
                .cloned()
                .or_else(|| Some(stack.clone()))
            {
                self.damage_item_stack(&mut current, 1);
                let replacement = (current.count > 0).then_some(current);
                self.player_container
                    .set_slot_contents_in_place(inventory_slot, replacement.clone());
                if let Ui::Creative(screen) = &mut self.ui {
                    screen.player.set_slot_contents(inventory_slot, replacement);
                }
                self.refresh_hotbar();
            }
        }
        true
    }

    /// The `onItemUse` return of the 1.7.10 items whose client callback changes nothing.
    fn item_use_without_placement_v5(
        &self,
        hit: interact::BlockHit,
        stack: &mc_protocol::ItemStack,
    ) -> bool {
        let Some(session) = self.session.as_ref() else {
            return false;
        };
        let world = &session.state().world;
        if interact::v5::placed_block(stack).is_some() {
            // Only a slab or snow item whose 1.8.9 fall-through found nothing to model gets
            // here (every other `ItemBlock` is decided by `interact::v5::item_block_use`); its
            // `onItemUse` returns true once the pre-click placement test, which passed, holds.
            true
        } else if stack.item_id == interact::ENDER_EYE_ITEM_ID {
            interact::ender_eye_item_use(world, hit.pos, hit.face, stack, true)
        } else if stack.item_id == interact::FIREWORK_ITEM_ID {
            // java17: `ItemFirework.onItemUse` spawns only on the server and returns false
            // (`ItemFirework.java:16-33`).
            false
        } else if stack.item_id == interact::SIGN_ITEM_ID {
            interact::sign_item_use(world, hit.pos, hit.face, stack, true).is_some()
        } else if stack.item_id == interact::SKULL_ITEM_ID {
            // java17: `ItemSkull.onItemUse`: `if (side == 0) return false; if
            // (!world.getBlock(x, y, z).getMaterial().isSolid()) return false;` then true on
            // the client (`ItemSkull.java:36-117`).
            hit.face != mc_math::Facing::Down && world.block_at(hit.pos).props().material.is_solid()
        } else {
            interact::has_on_item_use_override(stack)
        }
    }
}

/// java17: `ItemRedstone.onItemUse` (`ItemRedstone.java:19-70`).
///
/// ```text
/// if (world.getBlock(x, y, z) != Blocks.snow_layer) {
///     <step one cell along side>
///     if (!world.isAirBlock(x, y, z)) return false;
/// }
/// if (!player.canPlayerEdit(x, y, z, side, stack)) return false;
/// if (Blocks.redstone_wire.canPlaceBlockAt(world, x, y, z)) {
///     --stackSize; setBlock(redstone_wire);
/// }
/// return true;
/// ```
///
/// 1.8.9 places into a replaceable cell and returns false when the wire cannot stay.
fn redstone_item_use_v5(
    world: &mc_world::World,
    hit_pos: mc_math::BlockPos,
    side: mc_math::Facing,
) -> interact::BlockUseResult {
    let target = if world.block_at(hit_pos).block().0 == 78 {
        hit_pos
    } else {
        let target = hit_pos.relative(side);
        if !world.block_at(target).props().is_air() {
            return interact::BlockUseResult::Rejected;
        }
        target
    };
    let wire = mc_data::BlockId(55);
    if interact::v5::can_place_block_on_side_v5(world, wire, target, mc_math::Facing::Up) {
        interact::BlockUseResult::Consumed {
            placement: Some(interact::BlockUsePlacement {
                pos: target,
                state: wire.default_state(),
            }),
            decrement_stack: true,
        }
    } else {
        interact::BlockUseResult::Consumed {
            placement: None,
            decrement_stack: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{ToGame, ToNet};
    use mc_protocol::play::C2SPlay;
    use mc_version::GameVersion;

    /// One packet the client queued, in either form.
    #[derive(Debug)]
    enum Sent {
        V47(C2SPlay),
        V5(C2SPlay5),
    }

    fn drained(to_net: &mut tokio::sync::mpsc::UnboundedReceiver<ToNet>) -> Vec<Sent> {
        let mut out = Vec::new();
        while let Ok(message) = to_net.try_recv() {
            match message {
                ToNet::Send(packet) => out.push(Sent::V47(packet)),
                ToNet::SendV5(packet) => out.push(Sent::V5(packet)),
                ToNet::Disconnect => {}
            }
        }
        out
    }

    /// The 1.7.10 C07s among `sent`, as `(status, x, y, z, face)`.
    fn diggings(sent: &[Sent]) -> Vec<(u8, i32, u8, i32, u8)> {
        sent.iter()
            .filter_map(|sent| match sent {
                Sent::V5(C2SPlay5::PlayerDigging(dig)) => {
                    Some((dig.status, dig.x, dig.y, dig.z, dig.face))
                }
                _ => None,
            })
            .collect()
    }

    fn swings(sent: &[Sent]) -> usize {
        sent.iter()
            .filter(|sent| matches!(sent, Sent::V47(C2SPlay::Animation(_))))
            .count()
    }

    fn placements(sent: &[Sent]) -> usize {
        sent.iter()
            .filter(|sent| matches!(sent, Sent::V47(C2SPlay::PlayerBlockPlacement(_))))
            .count()
    }

    fn stack_named(name: &str, count: i32) -> mc_protocol::ItemStack {
        let id = mc_data::item_by_name(name).expect("item").0;
        mc_protocol::ItemStack {
            item_id: i16::try_from(id).expect("id fits"),
            count,
            damage: 0,
            nbt: None,
        }
    }

    /// A real client (skipped without the vanilla jar) whose session plays `version`, standing at
    /// the origin facing +Z with a stone block at (3, 65, 4) under its crosshair, as the 1.8.9
    /// driver tests set it up.
    fn client_aimed_at_stone(
        version: GameVersion,
        held: Option<mc_protocol::ItemStack>,
    ) -> Option<(
        Client,
        tempfile::TempDir,
        tokio::sync::mpsc::UnboundedReceiver<ToNet>,
    )> {
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
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let pos = mc_math::BlockPos::new(3, 65, 4);
        let mut world = mc_render::terrain::fixture::blank_world(1);
        mc_render::terrain::fixture::place(&mut world, pos, "minecraft:stone");
        let (handle, to_game, mut to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::with_version(handle, JoinSettings::default(), version);
        to_game
            .try_send(ToGame::Packet(mc_protocol::play::S2CPlay::JoinGame(
                mc_protocol::play::s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                },
            )))
            .expect("channel");
        session.poll(16);
        session.tick(&MoveInput::default());
        if let Some(predictor) = session.predictor_mut() {
            predictor.state.set_position(DVec3::ZERO);
            predictor.state.prev_pos = DVec3::ZERO;
            predictor.state.yaw = 0.0;
            predictor.state.pitch = 0.0;
        }
        session.state_mut().world = world;
        client.session = Some(session);
        client.ui = Ui::InGame;
        client.player_hud.selected_slot = 0;
        if let Some(held) = held {
            client
                .player_container
                .set_slot_contents(HOTBAR_SLOT_OFFSET, Some(held));
            client.refresh_hotbar();
        }
        client.hit = HitResult::Block(interact::BlockHit {
            pos,
            face: mc_math::Facing::Up,
            hit_vec: DVec3::new(3.5, 66.0, 4.5),
        });
        let _ = drained(&mut to_net);
        Some((client, dir, to_net))
    }

    /// java17 `Minecraft.clickMouse` (`:1342-1384`): a MISS sets no lockout, so every press of
    /// the tick swings; java18 (`:1554-1559`) swallows the second.
    #[test]
    fn clicks_at_nothing_keep_swinging_in_1_7_10() {
        for (version, expected) in [(GameVersion::V1_7_10, 2), (GameVersion::V1_8_9, 1)] {
            let Some((mut client, _dir, mut to_net)) = client_aimed_at_stone(version, None) else {
                return;
            };
            client.hit = HitResult::Miss;
            client.click_mouse();
            client.click_mouse();
            let sent = drained(&mut to_net);
            assert_eq!(swings(&sent), expected, "{version:?}: {sent:?}");
        }
    }

    /// java17 `Minecraft.sendClickBlockToController` (`:1309-1340`) has no `isUsingItem` gate:
    /// with a sword raised, holding attack on a block starts the dig (C07 START, in its 1.7.10
    /// form) and swings every tick. java18 does nothing (`:1503`).
    #[test]
    fn digging_and_swinging_go_on_while_blocking_in_1_7_10() {
        let sword = stack_named("diamond_sword", 1);
        for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
            let Some((mut client, _dir, mut to_net)) =
                client_aimed_at_stone(version, Some(sword.clone()))
            else {
                return;
            };
            client
                .controller
                .set_item_in_use(sword.clone(), interact::BOW_MAX_USE_DURATION);
            client.send_click_block_to_controller(true);
            client.send_click_block_to_controller(true);
            let sent = drained(&mut to_net);
            if version.is_1_7_10() {
                assert_eq!(diggings(&sent), vec![(0, 3, 65, 4, 1)], "{sent:?}");
                assert_eq!(swings(&sent), 2, "a swing on each tick: {sent:?}");
            } else {
                assert!(sent.is_empty(), "negative control, 1.8.9: {sent:?}");
            }
        }
    }

    /// java17 `PlayerControllerMP.resetBlockRemoving` (`:183-193`): the ABORT after a released
    /// dig names the block with face -1 (0xFF).
    #[test]
    fn a_released_dig_aborts_with_face_minus_one() {
        let Some((mut client, _dir, mut to_net)) =
            client_aimed_at_stone(GameVersion::V1_7_10, None)
        else {
            return;
        };
        client.send_click_block_to_controller(true);
        client.send_click_block_to_controller(false);
        client.send_click_block_to_controller(false);
        let sent = drained(&mut to_net);
        assert_eq!(
            diggings(&sent),
            vec![(0, 3, 65, 4, 1), (1, 3, 65, 4, 0xFF)],
            "one ABORT, only while digging: {sent:?}"
        );
        assert!(
            sent.iter()
                .all(|sent| !matches!(sent, Sent::V47(C2SPlay::PlayerDigging(_)))),
            "no 1.8.9 C07 on a 1.7.10 connection: {sent:?}"
        );
    }

    /// java17 `PlayerControllerMP.onStoppedUsingItem` (`:416-421`): `C07(5, 0, 0, 0, 255)`.
    /// java18 sends face `DOWN` (`:576-581`), as a 1.8.9 C07.
    #[test]
    fn releasing_an_item_sends_face_255() {
        let bow = stack_named("bow", 1);
        for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
            let Some((mut client, _dir, mut to_net)) =
                client_aimed_at_stone(version, Some(bow.clone()))
            else {
                return;
            };
            client
                .controller
                .set_item_in_use(bow.clone(), interact::BOW_MAX_USE_DURATION);
            client.release_item_use();
            assert!(!client.controller.is_using_item());
            let sent = drained(&mut to_net);
            if version.is_1_7_10() {
                assert_eq!(diggings(&sent), vec![(5, 0, 0, 0, 255)], "{sent:?}");
                assert!(no_v47_digging(&sent), "{sent:?}");
            } else {
                assert!(
                    diggings(&sent).is_empty(),
                    "negative control, 1.8.9: {sent:?}"
                );
                assert!(
                    sent.iter().any(|sent| matches!(
                        sent,
                        Sent::V47(C2SPlay::PlayerDigging(c2s::PlayerDigging {
                            status: c2s::PlayerDiggingStatus::ReleaseUseItem,
                            face: mc_math::Facing::Down,
                            ..
                        }))
                    )),
                    "negative control, 1.8.9: {sent:?}"
                );
            }
        }
    }

    /// java17 `Minecraft.rightClickMouse` (`:1386-1449`) has no `getIsHittingBlock` gate: a right
    /// click mid-dig places. java18 ignores it (`:1572`).
    #[test]
    fn a_right_click_places_while_digging_in_1_7_10() {
        let stone = stack_named("stone", 64);
        for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
            let Some((mut client, _dir, mut to_net)) =
                client_aimed_at_stone(version, Some(stone.clone()))
            else {
                return;
            };
            client.send_click_block_to_controller(true);
            assert!(
                client.controller.hitting_block().is_some(),
                "{version:?} is digging"
            );
            let _ = drained(&mut to_net);
            client.right_click_mouse();
            let sent = drained(&mut to_net);
            let expected = usize::from(version.is_1_7_10());
            assert_eq!(placements(&sent), expected, "{version:?}: {sent:?}");
        }
    }

    /// The tick pick of a 1.7.10 session aims with the current yaw, not last tick's head yaw.
    #[test]
    fn the_1_7_10_pick_aims_with_the_current_yaw() {
        for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
            let Some((mut client, _dir, _to_net)) = client_aimed_at_stone(version, None) else {
                return;
            };
            // The player faces the stone at (3, 65, 4) this frame (yaw 0, +Z) with its eye on
            // the block's lower edge, while last tick's head yaw still points the other way.
            if let Some(session) = client.session.as_mut() {
                if let Some(predictor) = session.predictor_mut() {
                    predictor.state.set_position(DVec3::new(3.5, 63.38, 1.5));
                    predictor.state.yaw = 0.0;
                    predictor.state.pitch = 0.0;
                }
                session.state_mut().local.head_yaw = 180.0;
            }
            client.update_hit_result();
            let on_stone = matches!(client.hit, HitResult::Block(hit) if hit.pos.z == 4);
            assert_eq!(
                on_stone,
                version.is_1_7_10(),
                "{version:?}: {:?}",
                client.hit
            );
        }
    }

    /// java17 `Minecraft.clickMouse` (`:1348-1356`): a null `objectMouseOver` costs the 10-tick
    /// lockout outside creative, so the second click of the tick does not swing; a MISS (the
    /// control) costs nothing.
    #[test]
    fn a_click_at_a_null_pick_locks_the_attack_out_in_1_7_10() {
        let Some((mut client, _dir, mut to_net)) =
            client_aimed_at_stone(GameVersion::V1_7_10, None)
        else {
            return;
        };
        for (null, expected) in [(true, 1), (false, 2)] {
            client.controller.release_attack();
            client.hit = HitResult::Miss;
            client.pick_v5.null = null;
            client.click_mouse();
            client.click_mouse();
            let sent = drained(&mut to_net);
            assert_eq!(swings(&sent), expected, "null {null}: {sent:?}");
        }
    }

    /// A diamond sword with Knockback I.
    fn knockback_sword() -> mc_protocol::ItemStack {
        let mut stack = stack_named("diamond_sword", 1);
        let mut enchantment = mc_nbt::NbtCompound::new();
        enchantment.insert("id", mc_nbt::Nbt::Short(19));
        enchantment.insert("lvl", mc_nbt::Nbt::Short(1));
        let mut tag = mc_nbt::NbtCompound::new();
        tag.insert(
            "ench",
            mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Compound(enchantment)]),
        );
        stack.nbt = Some(tag);
        stack
    }

    /// The pick's record of a remote player, network id 77, that the tracker no longer holds.
    fn picked_remote_player() -> PickedEntityV5 {
        PickedEntityV5 {
            network_id: 77,
            kind: None,
            remote_player: true,
            living: true,
            dragon_part: false,
            facts: Some(EntityFactsV5 {
                kind: None,
                generation: None,
                facts: interact::EntityInteractionFacts::default(),
                leashed_to_local: false,
                command_minecart: None,
            }),
        }
    }

    /// java17: the tick picks before its packets (`Minecraft.java:1564-1569`), so a remote player
    /// an S13 removed after the pick is still the `Entity` Java attacks. Its client-side
    /// `attackEntityFrom` succeeds, so a Knockback hit damps the motion by 0.6 and ends the
    /// sprint (`EntityPlayer.java:1052-1169`). Without the pick's record (the control) nothing is
    /// known about id 77 and only the C02 goes out.
    #[test]
    fn an_attack_on_a_player_removed_after_the_pick_still_knocks_back() {
        for picked in [true, false] {
            let Some((mut client, _dir, mut to_net)) =
                client_aimed_at_stone(GameVersion::V1_7_10, Some(knockback_sword()))
            else {
                return;
            };
            client.hit = HitResult::Entity(interact::EntityHit {
                network_id: 77,
                hit_vec: DVec3::new(0.5, 1.0, 2.0),
            });
            client.pick_v5.entity = picked.then(picked_remote_player);
            if let Some(predictor) = client.session.as_mut().and_then(Session::predictor_mut) {
                predictor.state.vel = DVec3::new(1.0, 0.0, 1.0);
            }
            client.click_mouse();
            let sent = drained(&mut to_net);
            let attacks = sent
                .iter()
                .filter(|sent| {
                    matches!(
                        sent,
                        Sent::V47(C2SPlay::UseEntity(c2s::UseEntity {
                            entity_id: 77,
                            action: c2s::UseEntityAction::Attack,
                        }))
                    )
                })
                .count();
            assert_eq!(attacks, 1, "picked {picked}: {sent:?}");
            let vel_x = client
                .session
                .as_ref()
                .and_then(Session::predictor)
                .map(|predictor| predictor.state.vel.x);
            assert_eq!(
                vel_x,
                Some(if picked { 0.6 } else { 1.0 }),
                "picked {picked}"
            );
        }
    }

    /// java17 `rightClickMouse` (`:1386-1449`): `interactWith` on a remote player is false, so a
    /// held item still goes out as the fallback C08, also for a player an S13 removed after the
    /// pick. Without the pick's record (the control) the click stays consumed.
    #[test]
    fn a_right_click_on_a_player_removed_after_the_pick_falls_back_to_c08() {
        for picked in [true, false] {
            let Some((mut client, _dir, mut to_net)) =
                client_aimed_at_stone(GameVersion::V1_7_10, Some(stack_named("stick", 1)))
            else {
                return;
            };
            client.hit = HitResult::Entity(interact::EntityHit {
                network_id: 77,
                hit_vec: DVec3::new(0.5, 1.0, 2.0),
            });
            client.pick_v5.entity = picked.then(picked_remote_player);
            client.right_click_mouse();
            let sent = drained(&mut to_net);
            assert_eq!(
                placements(&sent),
                usize::from(picked),
                "picked {picked}: {sent:?}"
            );
        }
    }

    /// java17 `Minecraft.middleClickMouse` (`:2259-2377`) has no creative Ctrl branch: a Ctrl
    /// pick of a player skull is the plain skull of its type, without the tile's NBT, in the
    /// hotbar and in the C10. java18 copies the tile's NBT (`:2526-2529`, `:2618-2627`).
    #[test]
    fn a_creative_ctrl_pick_copies_no_tile_nbt_in_1_7_10() {
        use winit::keyboard::KeyCode;

        for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
            let Some((mut client, _dir, mut to_net)) = client_aimed_at_stone(version, None) else {
                return;
            };
            let pos = mc_math::BlockPos::new(3, 65, 4);
            {
                let session = client.session.as_mut().expect("session");
                session.state_mut().local.game_mode = mc_protocol::play::s2c::GameMode::Creative;
                session.state_mut().local.abilities.creative_mode = true;
                session.state_mut().world.set_block(
                    pos,
                    mc_data::BlockState::new(mc_data::block_by_name("skull").expect("skull"), 0),
                );
                session.state_mut().world.set_tile_entity(
                    pos,
                    mc_world::TileEntity {
                        kind: mc_world::TileKind::Skull,
                        data: mc_world::TileData::Skull(mc_world::SkullState {
                            skull_type: 3,
                            rotation: 2,
                            owner_name: Some("Notch".to_owned()),
                            owner_profile: None,
                            skin_url: None,
                        }),
                    },
                );
            }
            client
                .keyboard
                .push(KeyCode::ControlLeft, true, false, None);
            let _ = drained(&mut to_net);
            client.middle_click_mouse();
            let picked = client
                .player_container
                .slot_stack(HOTBAR_SLOT_OFFSET)
                .cloned();
            assert!(
                picked
                    .as_ref()
                    .is_some_and(|stack| stack.item_id == 397 && stack.damage == 3),
                "{version:?}: {picked:?}"
            );
            let with_nbt = !version.is_1_7_10();
            assert_eq!(
                picked.as_ref().is_some_and(|stack| stack.nbt.is_some()),
                with_nbt,
                "{version:?}: {picked:?}"
            );
            let sent = drained(&mut to_net);
            let c10: Vec<_> = sent
                .iter()
                .filter_map(|sent| match sent {
                    Sent::V47(C2SPlay::CreativeInventoryAction(action)) => Some(action),
                    _ => None,
                })
                .collect();
            assert_eq!(c10.len(), 1, "{version:?}: {sent:?}");
            assert_eq!(c10[0].slot_id, 36, "{version:?}");
            assert_eq!(
                c10[0]
                    .stack
                    .as_ref()
                    .is_some_and(|stack| stack.nbt.is_some()),
                with_nbt,
                "{version:?}: {sent:?}"
            );
        }
    }

    /// Whether `sent` holds no 1.8.9 C07: `C2SPlay5::from_v47` refuses every one, so on a 1.7.10
    /// connection it would never reach the server.
    fn no_v47_digging(sent: &[Sent]) -> bool {
        sent.iter()
            .all(|sent| !matches!(sent, Sent::V47(C2SPlay::PlayerDigging(_))))
    }

    /// java17: Escape (like E, T and `/`) opens its screen inside the keyboard loop, and the same
    /// `runTick` still reaches `sendClickBlockToController(false)` (`Minecraft.java:1710-1875`,
    /// `:1930`), whose `resetBlockRemoving` sends `C07(1, x, y, z, -1)`
    /// (`PlayerControllerMP.java:183-193`). The 1.7.10 twin of the driver's
    /// `raw_escape_modal_aborts_once_clears_dig_and_cannot_later_stop`.
    #[test]
    fn a_screen_opened_mid_dig_aborts_with_face_minus_one() {
        use winit::keyboard::KeyCode;

        let Some((mut client, _dir, mut to_net)) =
            client_aimed_at_stone(GameVersion::V1_7_10, None)
        else {
            return;
        };
        client.send_click_block_to_controller(true);
        let _ = drained(&mut to_net);
        assert!(client.controller.hitting_block().is_some());

        client.handle_raw_key(KeyCode::Escape, true, false, None);
        client.handle_screen_input(&mut InputState::new(), 854, 480);
        assert!(matches!(client.ui, Ui::Pause(_)));
        assert!(client.controller.hitting_block().is_none());
        let sent = drained(&mut to_net);
        assert_eq!(
            diggings(&sent),
            vec![(1, 3, 65, 4, 0xFF)],
            "one 1.7.10 ABORT: {sent:?}"
        );
        assert!(
            no_v47_digging(&sent),
            "no 1.8.9 C07 on a 1.7.10 connection: {sent:?}"
        );

        client.show_screen(None);
        let _ = drained(&mut to_net);
        client.send_click_block_to_controller(false);
        let sent = drained(&mut to_net);
        assert!(
            sent.is_empty(),
            "the cleared dig sends nothing more: {sent:?}"
        );
    }

    /// Focus loss resets the dig at the shared focus seam (`note_focus_changed_at`); in 1.7.10
    /// that reset is `resetBlockRemoving`'s `C07(1, x, y, z, -1)`
    /// (`PlayerControllerMP.java:183-193`), and regaining focus installs `leftClickCounter =
    /// 10000` (`Minecraft.setIngameFocus`, `Minecraft.java:1272-1284`). The 1.7.10 twin of the
    /// driver's `focus_loss_aborts_once_and_regain_reinstalls_damage_lockout`.
    #[test]
    fn focus_loss_mid_dig_aborts_with_face_minus_one() {
        let Some((mut client, _dir, mut to_net)) =
            client_aimed_at_stone(GameVersion::V1_7_10, None)
        else {
            return;
        };
        client.send_click_block_to_controller(true);
        let _ = drained(&mut to_net);
        assert!(client.controller.hitting_block().is_some());

        <Client as mc_render::ShellApp>::on_focus_changed(&mut client, false);
        assert!(client.controller.hitting_block().is_none());
        let sent = drained(&mut to_net);
        assert_eq!(
            diggings(&sent),
            vec![(1, 3, 65, 4, 0xFF)],
            "one 1.7.10 ABORT: {sent:?}"
        );
        assert!(
            no_v47_digging(&sent),
            "no 1.8.9 C07 on a 1.7.10 connection: {sent:?}"
        );

        client.controller.release_attack();
        assert!(client.controller.may_damage_block());
        <Client as mc_render::ShellApp>::on_focus_changed(&mut client, true);
        assert!(!client.controller.may_damage_block());
        client.send_click_block_to_controller(false);
        let sent = drained(&mut to_net);
        assert!(sent.is_empty(), "nothing after the lockout: {sent:?}");
    }
}
