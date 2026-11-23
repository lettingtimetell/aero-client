//! The client's capabilities-into-physics seam.
//!
//! A second `impl Client` block, so the capabilities-into-physics seam reads as one file instead
//! of as scattered arms of [`super::driver`]: the `AbilitiesChanged` applier, the
//! potion/velocity/item-use movement inputs, and the per-tick signals `PhysicsState` hands back.
//!
//! # What lives where
//!
//! A11's verdict was that the *arithmetic* in `mc-physics` is right and the **client never feeds
//! it what the server sent**. Closing that splits cleanly in two, and the split is by ownership
//! rather than by convenience:
//!
//! * **Inputs** — capabilities, the game mode, health, the food level, potion effects, the
//!   `generic.movementSpeed` attribute and an incoming `S12PacketEntityVelocity` — go in through
//!   [`crate::Session`], which owns both the `ClientState` they arrive on and the
//!   `mc_physics::Predictor` they drive. See `Session::push_server_movement_state`.
//! * **Outputs** — the footstep, the swim splash, the fall sound, the predicted fall damage and
//!   the portal timer come back out of `PhysicsState` and need a sound engine, an overlay and a
//!   HUD. Those belong to [`Client`], and that half is this module. The cactus contact bit is
//!   intentionally not listed: it is retained only by the parity trace because the client-side
//!   `attackEntityFrom` path is a no-op, so there is no client damage/UI event to consume here.
//!
//! The one input that cannot go through `Session` is the Depth Strider level, because it is a
//! function of the *armour inventory*, which `Client` owns and `Session` does not; it is computed
//! here and pushed down.

use mc_audio::PlayRequest;
use mc_physics::state::{StepSound, WaterEntry};

use super::driver::Client;

/// java: `EntityPlayerSP.onLivingUpdate`'s one-shot `PositionedSoundRecord.create` call.
const PORTAL_TRIGGER_SOUND: &str = "portal.trigger";

/// Advance the local portal ramp and return the entry sound, if this tick is a new contact.
///
/// Vanilla tests `timeInPortal == 0.0F` before applying the `0.0125F` increment. Keeping the
/// transition and ramp in one helper makes the edge latch explicit: held contact cannot replay
/// the sound, while a decay back to zero permits the next entry to trigger it again.
fn advance_portal_signal(
    time_in_portal: &mut f32,
    prev_time_in_portal: &mut f32,
    in_portal: bool,
    confusion_duration: Option<i32>,
    rng: &mut mc_math::JavaRandom,
) -> Option<PlayRequest> {
    *prev_time_in_portal = *time_in_portal;

    if in_portal {
        let trigger = (*time_in_portal == 0.0).then(|| {
            PlayRequest::ambient_with_pitch(
                mc_assets::ResourceLocation::minecraft(PORTAL_TRIGGER_SOUND),
                rng.next_f32() * 0.4 + 0.8,
            )
        });
        *time_in_portal = (*time_in_portal + 0.0125).min(1.0);
        trigger
    } else if confusion_duration.is_some_and(|duration| duration > 60) {
        // java: `EntityPlayerSP.onLivingUpdate` — Potion.confusion's client-only visual ramp.
        // The strict `> 60` guard is intentional: at three seconds or less vanilla lets the
        // normal `-0.05F` decay run instead.
        *time_in_portal += 0.006666667;
        if *time_in_portal > 1.0 {
            *time_in_portal = 1.0;
        }
        None
    } else {
        *time_in_portal = (*time_in_portal - 0.05).max(0.0);
        None
    }
}

/// java: `Enchantment.depthStrider` — `new EnchantmentWaterWalker(8, "depth_strider", 2)`, so the
/// `effectId` stored in an `ench` tag is `8`.
const DEPTH_STRIDER_ID: i16 = 8;

/// Window `0`'s armour slots. java: `ContainerPlayer`'s four `SlotArmor`s, indices `5..=8`, which
/// mirror `InventoryPlayer.armorInventory` — the array `EntityPlayer.getInventory()` returns and
/// `EnchantmentHelper.getMaxEnchantmentLevel` scans.
const ARMOR_SLOTS: std::ops::RangeInclusive<usize> = 5..=8;

/// java: `EntityPlayer.getFallSoundString` (`entity/player/EntityPlayer.java:1953-1956`), the
/// override `EntityLivingBase.fall` reaches for the local player. The `EntityLivingBase` default
/// (`game.neutral.hurt.fall.*`) is the friendly-creature category, so using it put the local
/// player's own fall grunt on the wrong volume slider.
fn fall_sound_for(damage: u32) -> &'static str {
    if damage > 4 {
        "game.player.hurt.fall.big"
    } else {
        "game.player.hurt.fall.small"
    }
}

/// java: `EntityPlayer.getSwimSound` (`entity/player/EntityPlayer.java:423`).
const SWIM_SOUND: &str = "game.player.swim";

/// java: `EntityPlayer.getSplashSound` (`entity/player/EntityPlayer.java:428`).
const SPLASH_SOUND: &str = WaterEntry::PLAYER_SOUND_NAME;

/// One fully expanded `Entity.resetHeight` signal, ready for client-owned sinks.
struct WaterEntryFanout {
    sound: PlayRequest,
    particles: Vec<(i32, mc_math::DVec3, mc_math::DVec3)>,
}

/// Expand physics' deterministic water-entry recipe with the local player's persistent RNG.
///
/// Java draws the pitch first, then five values per loop: bubble X/Z/Y velocity followed by
/// splash X/Z. Bubble and splash positions are therefore independent even though they share the
/// same centre and loop count.
fn water_entry_fanout(entry: WaterEntry, rng: &mut mc_math::JavaRandom) -> WaterEntryFanout {
    let pitch =
        entry.sound_pitch_base + (rng.next_f32() - rng.next_f32()) * entry.sound_pitch_jitter;
    let mut particles = Vec::with_capacity(entry.particle_count as usize * 2);

    for _ in 0..entry.particle_count {
        let bubble_pos = entry.particle_origin
            + mc_math::DVec3::new(
                f64::from((rng.next_f32() * 2.0 - 1.0) * entry.particle_width),
                0.0,
                f64::from((rng.next_f32() * 2.0 - 1.0) * entry.particle_width),
            );
        let mut bubble_velocity = entry.bubble_velocity;
        bubble_velocity.y -= f64::from(rng.next_f32() * entry.bubble_y_random_subtract);
        particles.push((
            mc_render::particle::packet::WATER_BUBBLE,
            bubble_pos,
            bubble_velocity,
        ));

        let splash_pos = entry.particle_origin
            + mc_math::DVec3::new(
                f64::from((rng.next_f32() * 2.0 - 1.0) * entry.particle_width),
                0.0,
                f64::from((rng.next_f32() * 2.0 - 1.0) * entry.particle_width),
            );
        particles.push((
            mc_render::particle::packet::WATER_SPLASH,
            splash_pos,
            entry.splash_velocity,
        ));
    }

    WaterEntryFanout {
        sound: PlayRequest {
            // This fanout is consumed only by the local EntityPlayerSP path.  Remote tracker
            // water entries use WaterEntry::SOUND_NAME in their own expansion boundary.
            event: mc_assets::ResourceLocation::minecraft(SPLASH_SOUND),
            volume: entry.sound_volume,
            pitch,
            position: Some(entry.particle_origin),
            attenuation: mc_audio::AttenuationType::Linear,
            repeat: false,
            repeat_delay_ticks: 0,
        },
        particles,
    }
}

/// The first level of `enchantment_id` in one stack's `ench` list.
///
/// Java's `EnchantmentHelper.getEnchantmentLevel` returns as soon as it finds a matching entry.
/// `NBTTagCompound.getShort` supplies zero for an absent or non-numeric field, so a matching
/// entry with no usable `lvl` still wins with level zero; later duplicates are not considered.
pub(super) fn enchantment_level(stack: &mc_protocol::item::ItemStack, enchantment_id: i16) -> i16 {
    let Some(list) = stack.nbt.as_ref().and_then(|nbt| nbt.get_list("ench")) else {
        return 0;
    };

    for entry in list {
        // `NBTTagList.getCompoundTagAt` turns a non-compound list entry into an empty compound.
        // Keep that fallback here too: it matters for id zero (Protection), where the empty
        // compound itself is the first matching entry and supplies level zero.
        let compound = entry.as_compound();
        if compound.and_then(|tag| tag.get_i16("id")).unwrap_or(0) == enchantment_id {
            return compound.and_then(|tag| tag.get_i16("lvl")).unwrap_or(0);
        }
    }
    0
}

/// The largest Depth Strider level on any armour piece in `stacks`.
///
/// java: `EnchantmentHelper.getDepthStriderModifier(entity)` →
/// `getMaxEnchantmentLevel(Enchantment.depthStrider.effectId, entity.getInventory())`, whose body
/// is a plain max over `getEnchantmentLevel(id, stack)`. That in turn reads the stack's `ench`
/// list of `{ id: short, lvl: short }` compounds — the same NBT a server sends on the wire, so no
/// item registry lookup is involved and this works for a modded boot too.
fn depth_strider_level<'a>(stacks: impl Iterator<Item = &'a mc_protocol::item::ItemStack>) -> f32 {
    let mut best = 0i16;
    for stack in stacks {
        best = best.max(enchantment_level(stack, DEPTH_STRIDER_ID));
    }
    f32::from(best)
}

impl Client {
    /// Push the current armour inventory's Depth Strider level into the physics input which the
    /// imminent [`crate::Session::tick`] consumes.
    ///
    /// java: `EntityLivingBase.moveEntityWithHeading` calls
    /// `EnchantmentHelper.getDepthStriderModifier(this)` from inside the current water-movement
    /// tick. Inventory packets and local container clicks have already converged before this
    /// boundary, so sampling here prevents equip and unequip from taking effect one tick late.
    pub(super) fn push_armor_depth_strider(&mut self) {
        let depth_strider = depth_strider_level(
            ARMOR_SLOTS
                .clone()
                .filter_map(|slot| self.player_container.slot_stack(slot)),
        );
        if let Some(session) = self.session.as_mut() {
            session.set_armor_depth_strider(depth_strider);
        }
    }

    /// GAP `A11-01` (P0). java: `NetHandlerPlayClient.handlePlayerAbilities` (`:1668`) writes
    /// straight into `thePlayer.capabilities` and `PlayerControllerMP`.
    ///
    /// `sync::dispatch` has folded this into `ClientState::local.abilities` since S14 and the
    /// only reader in the whole tree was the air-supply HUD's `invulnerable` check, so a server
    /// that set `isFlying` — which every creative server and most hub servers do on join — got a
    /// client that applied `−0.08` gravity, fell, and rubber-banded against the corrections
    /// forever. [`crate::Session`] applies the physics half immediately before every tick;
    /// this pushes it down the moment the packet lands as well, so a frame rendered between the
    /// packet and the next tick already agrees with it (the FOV modifier reads
    /// `capabilities.isFlying`, and `EntityRenderer.updateFovModifierHand` runs per tick).
    pub(super) fn apply_abilities(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.push_server_movement_state();
        }
    }

    /// java: `EntityPlayerSP.onLivingUpdate:732-780`, the portal block at the top of the local
    /// player's tick (`BlockPortal.onEntityCollidedWithBlock` is A11-21; H2 made the timing
    /// vanilla's). The caller runs it only on a tick `EntityPlayerSP.onUpdate` runs, before the
    /// session's player tick.
    ///
    /// * `inPortal` is the flag the **previous** tick's `doBlockCollisions` set: `moveEntity`
    ///   runs after this block, so contact ramps, plays `portal.trigger` and closes screens at
    ///   the start of the next tick. (It used to run after the tick on that tick's own contact,
    ///   one tick early, and also on ticks the player did not run, which froze nothing.)
    /// * The three-way ramp: contact climbs by `0.0125`, long-lived confusion by `0.006666667`,
    ///   and all other ticks fall by `0.05`; each branch clamps to `0..=1`, and
    ///   `GuiIngame.renderFrame` interpolates `prevTimeInPortal -> timeInPortal` by `partialTicks`.
    /// * Contact closes a screen that does not pause the game, with `displayGuiScreen(null)` and
    ///   so without a C0D: [`Client::close_non_pausing_screen_for_portal`].
    pub(super) fn tick_portal_prologue(&mut self) {
        let in_portal = self
            .session
            .as_ref()
            .and_then(|session| session.predictor())
            .is_some_and(|predictor| predictor.state.in_portal);
        if in_portal {
            self.close_non_pausing_screen_for_portal();
        }
        let confusion_duration = self.confusion_duration();
        if let Some(request) = advance_portal_signal(
            &mut self.time_in_portal,
            &mut self.prev_time_in_portal,
            in_portal,
            confusion_duration,
            &mut self.portal_rng,
        ) {
            self.play_sound(request);
        }
    }

    /// Everything `mc_physics::PhysicsState` produced this tick that needs a sound engine or the
    /// HUD. Called from `Client::tick` after `Session::tick`; movement inputs such as Depth
    /// Strider are pushed separately before the session tick, and the portal flag is consumed by
    /// the next tick's [`Client::tick_portal_prologue`].
    ///
    /// Each of these was a computed-and-discarded value before Wave 1:
    ///
    /// * **A11-15** `predicted_fall_damage` was bit-exact and read by nothing but the parity
    ///   harness, so landing was silent until `S06PacketUpdateHealth` arrived a round trip later.
    /// * **A11-13** `distanceWalked*`/`playStepSound` did not exist at all, so walking was silent
    ///   despite `mc-data` carrying `step_sound` for every block.
    pub(super) fn consume_physics_signals(&mut self) {
        let Some(session) = self.session.as_mut() else {
            return;
        };

        let Some(state) = session.predictor().map(|predictor| predictor.state) else {
            return;
        };

        let feet = state.pos;

        // java: `Entity.resetHeight` — entry sound first, then one bubble and one splash for
        // every loop iteration. Physics owns the air-to-water edge and deterministic recipe;
        // this client boundary owns RNG expansion and the audio/particle sinks.
        if let Some(entry) = state.water_entry {
            let fanout = water_entry_fanout(entry, &mut self.water_entry_rng);
            self.play_sound(fanout.sound);
            for (particle, pos, velocity) in fanout.particles {
                self.spawn_effect_particle(particle, pos, velocity, &[]);
            }
        }

        // java: `Entity.moveEntity:880` — the swim sound fires *before* the step sound, at a
        // volume derived from motion. Its pitch is `1.0F + (rand - rand) * 0.4F`, which is RNG
        // vanilla draws from the entity's own `Random`; this client has no per-entity RNG stream
        // for the local player, so the pitch is left at `1.0` and the volume — the part that is
        // a function of state, and the part the differential fixture pins — is exact.
        if let Some(volume) = state.swim_sound_volume {
            self.play_sound(PlayRequest {
                event: mc_assets::ResourceLocation::minecraft(SWIM_SOUND),
                volume,
                pitch: 1.0,
                position: Some(feet),
                attenuation: mc_audio::AttenuationType::Linear,
                repeat: false,
                repeat_delay_ticks: 0,
            });
        }

        // java: `Entity.playStepSound` (`entity/Entity.java:986`). The snow-layer substitution and
        // the "a liquid with nothing on top is silent" rule are already applied by `mc-physics`;
        // `StepSound::sound_name` carries `Block.SoundType.getStepSound()`'s one override.
        if let Some(step) = state.step_sound {
            self.play_step_sound(step, feet, 1.0, 1.0);
        }

        // java: `EntityLivingBase.fall` (`:1151`) — the fall grunt, then the ground block's own
        // step sound at `volume * 0.5F, pitch * 0.75F`. Both are inside `if (i > 0)`, i.e. only
        // when the fall actually hurt, which is why a two-block drop is silent.
        if let Some(damage) = state.predicted_fall_damage {
            self.play_sound(PlayRequest {
                event: mc_assets::ResourceLocation::minecraft(fall_sound_for(damage)),
                volume: 1.0,
                pitch: 1.0,
                position: Some(feet),
                attenuation: mc_audio::AttenuationType::Linear,
                repeat: false,
                repeat_delay_ticks: 0,
            });
            let ground = mc_math::BlockPos::new(
                mc_math::floor_f64(feet.x),
                mc_math::floor_f64(feet.y - 0.200_000_002_980_232_24),
                mc_math::floor_f64(feet.z),
            );
            if let Some(session) = self.session.as_ref() {
                let block = session.state().world.block_at(ground);
                if mc_data::block_props(block).material != mc_data::Material::Air {
                    self.play_step_sound(
                        StepSound {
                            pos: ground,
                            state: block,
                        },
                        feet,
                        0.5,
                        0.75,
                    );
                }
            }
        }
    }

    /// One `playStepSound`-shaped call. `volume_scale`/`pitch_scale` are `1.0`/`1.0` for a
    /// footstep and `0.5`/`0.75` for the block thump `EntityLivingBase.fall` adds on landing.
    fn play_step_sound(&mut self, step: StepSound, at: mc_math::DVec3, volume: f32, pitch: f32) {
        self.play_sound(PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(step.sound_name()),
            volume: step.volume() * volume,
            pitch: step.pitch() * pitch,
            position: Some(at),
            attenuation: mc_audio::AttenuationType::Linear,
            repeat: false,
            repeat_delay_ticks: 0,
        });
    }

    /// For the entity-layer author. java: `EntityLivingBase.limbSwing` /
    /// `limbSwingAmount` / `prevLimbSwingAmount`, the last twelve lines of
    /// `moveEntityWithHeading` — which `mc_render::entity::anim::AnimParams` has documented as
    /// its input since S18 and which nothing produced.
    ///
    /// Returns `(limb_swing, limb_swing_amount, prev_limb_swing_amount)` for the **local**
    /// player, or `None` before a world exists. The renderer wants
    /// `prev + (amount - prev) * partial_ticks` for the amount and
    /// `limb_swing - amount * (1 - partial_ticks)` for the phase, exactly as
    /// `RendererLivingEntity.doRender` builds them.
    #[must_use]
    pub fn local_limb_swing(&self) -> Option<(f32, f32, f32)> {
        let state = self.session()?.predictor()?.state;
        Some((
            state.limb_swing,
            state.limb_swing_amount,
            state.prev_limb_swing_amount,
        ))
    }

    /// GAP `A11-13`'s other half: the view-bobbing input. java:
    /// `EntityRenderer.setupViewBobbing` (`:390`) reads
    /// `prevDistanceWalkedModified + (distanceWalkedModified - prevDistanceWalkedModified) * partialTicks`
    /// and swings the camera by it.
    ///
    /// Returns `(prev_distance_walked, distance_walked)`. No bobbing is drawn yet — that is a
    /// camera-matrix change in `mc-render` — but the two numbers it needs now exist and are
    /// asserted bit-exact against the game by every fixture in `crates/mc-physics/tests/parity.rs`.
    #[must_use]
    pub fn view_bob_distance(&self) -> Option<(f32, f32)> {
        let state = self.session()?.predictor()?.state;
        Some((
            state.prev_distance_walked_modified,
            state.distance_walked_modified,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_nbt::{Nbt, NbtCompound};

    fn enchanted(id: i16, level: i16) -> mc_protocol::item::ItemStack {
        enchanted_entries([(id, Some(level))])
    }

    fn enchanted_entries(
        entries: impl IntoIterator<Item = (i16, Option<i16>)>,
    ) -> mc_protocol::item::ItemStack {
        let entries = entries
            .into_iter()
            .map(|(id, level)| {
                let mut entry = NbtCompound::new();
                entry.insert("id", Nbt::Short(id));
                if let Some(level) = level {
                    entry.insert("lvl", Nbt::Short(level));
                }
                Nbt::Compound(entry)
            })
            .collect();
        let mut nbt = NbtCompound::new();
        nbt.insert("ench", Nbt::list(entries));
        mc_protocol::item::ItemStack {
            item_id: 313,
            count: 1,
            damage: 0,
            nbt: Some(nbt),
        }
    }

    fn plain() -> mc_protocol::item::ItemStack {
        mc_protocol::item::ItemStack {
            item_id: 313,
            count: 1,
            damage: 0,
            nbt: None,
        }
    }

    /// java: `getMaxEnchantmentLevel` is a max over the whole armour array, and an unenchanted or
    /// differently-enchanted piece contributes nothing.
    #[test]
    fn depth_strider_takes_the_highest_level_on_any_armour_piece() {
        assert_eq!(depth_strider_level([].iter()), 0.0);
        assert_eq!(depth_strider_level([plain()].iter()), 0.0);
        assert_eq!(depth_strider_level([enchanted(8, 3)].iter()), 3.0);
        // Protection (id 0) is not Depth Strider.
        assert_eq!(depth_strider_level([enchanted(0, 4)].iter()), 0.0);
        assert_eq!(
            depth_strider_level([enchanted(8, 1), plain(), enchanted(8, 2)].iter()),
            2.0,
            "the maximum wins, not the first or the last"
        );
    }

    /// java: `EntityLivingBase.getFallSoundString` — the split is at `> 4`, not `>= 4`.
    /// `getEnchantmentLevel` returns on the first matching list element. The max is only across
    /// armour slots, never across duplicate entries in one stack.
    #[test]
    fn depth_strider_uses_the_first_duplicate_in_each_enchantment_list() {
        let low_then_high = enchanted_entries([(8, Some(1)), (8, Some(3))]);
        let high_then_low = enchanted_entries([(8, Some(3)), (8, Some(1))]);
        let ordinary = enchanted(8, 2);

        assert_eq!(depth_strider_level([low_then_high.clone()].iter()), 1.0);
        assert_eq!(depth_strider_level([high_then_low].iter()), 3.0);
        assert_eq!(
            depth_strider_level([low_then_high, ordinary].iter()),
            2.0,
            "the outer armour-array maximum remains intact"
        );
    }

    /// Java's `NBTTagCompound.getShort` returns zero for an absent level. That malformed first
    /// duplicate must therefore stop the scan rather than expose a later valid duplicate.
    #[test]
    fn malformed_first_matching_enchantment_is_zero_and_stops_the_scan() {
        let missing_level_then_high = enchanted_entries([(8, None), (8, Some(3))]);
        assert_eq!(depth_strider_level([missing_level_then_high].iter()), 0.0);
    }

    #[test]
    fn client_tick_pushes_depth_strider_before_session_physics() {
        let source = include_str!("driver.rs");
        let (_, tick) = source
            .split_once("pub fn tick(&mut self, input: &MoveInput) {")
            .expect("the concrete client tick remains present");
        let (tick, _) = tick
            .split_once("fn upload_terrain(")
            .expect("the tick ends before frame-only terrain work");

        let push = tick
            .find("self.push_armor_depth_strider();")
            .expect("the armour-derived movement input must be refreshed each client tick");
        let physics = tick
            .find("session.tick_with_held(input, held.as_ref());")
            .expect("the session physics boundary must remain live");
        let consume = tick
            .find("self.consume_physics_signals();")
            .expect("post-physics sound and overlay consumers must remain live");

        assert_eq!(
            tick.matches("self.push_armor_depth_strider();").count(),
            1,
            "one armour scan supplies exactly one physics tick"
        );
        assert!(
            push < physics && physics < consume,
            "equip/unequip must affect the imminent tick, then its produced signals are consumed"
        );
    }

    #[test]
    fn water_entry_fanout_emits_one_sound_and_thirteen_ordered_pairs() {
        let entry = WaterEntry {
            particle_origin: mc_math::DVec3::new(4.0, 65.0, -3.0),
            sound_volume: 0.7,
            particle_count: 13,
            particle_width: 0.6,
            bubble_velocity: mc_math::DVec3::new(0.2, 0.4, -0.1),
            bubble_y_random_subtract: 0.2,
            splash_velocity: mc_math::DVec3::new(0.2, 0.4, -0.1),
            sound_pitch_base: 1.0,
            sound_pitch_jitter: 0.4,
        };
        let mut rng = mc_math::JavaRandom::new(0x5741544552);
        let fanout = water_entry_fanout(entry, &mut rng);

        assert_eq!(
            fanout.sound.event,
            mc_assets::ResourceLocation::minecraft(SPLASH_SOUND)
        );
        assert_eq!(fanout.sound.volume, 0.7);
        assert_eq!(fanout.sound.position, Some(entry.particle_origin));
        assert_eq!(fanout.sound.attenuation, mc_audio::AttenuationType::Linear);
        assert_eq!(fanout.particles.len(), 26);

        for pair in fanout.particles.chunks_exact(2) {
            assert_eq!(pair[0].0, mc_render::particle::packet::WATER_BUBBLE);
            assert_eq!(pair[1].0, mc_render::particle::packet::WATER_SPLASH);
            assert_eq!(pair[1].2, entry.splash_velocity);
            assert!(pair[0].2.y <= entry.bubble_velocity.y);
            assert!(
                pair[0].2.y >= entry.bubble_velocity.y - f64::from(entry.bubble_y_random_subtract)
            );
            for (_, pos, _) in pair {
                assert_eq!(pos.y, entry.particle_origin.y);
                assert!((pos.x - entry.particle_origin.x).abs() <= 0.6);
                assert!((pos.z - entry.particle_origin.z).abs() <= 0.6);
            }
        }
    }

    #[test]
    fn local_player_water_sounds_use_player_overrides() {
        // EntityPlayer overrides both inherited Entity sound hooks in vanilla 1.8.9.  Keep the
        // ordinary WaterEntry constant available for remote tracker expansion, but ensure the
        // local ability fanout cannot silently regress to the neutral family.
        assert_eq!(SWIM_SOUND, "game.player.swim");
        assert_eq!(SPLASH_SOUND, "game.player.swim.splash");
        assert_eq!(WaterEntry::SOUND_NAME, "game.neutral.swim.splash");
        assert_ne!(SWIM_SOUND, "game.neutral.swim");
        assert_ne!(SPLASH_SOUND, WaterEntry::SOUND_NAME);
    }

    #[test]
    fn consume_physics_signals_keeps_the_live_water_entry_bridge() {
        let source = include_str!("abilities.rs");
        let (_, consumer) = source
            .split_once("pub(super) fn consume_physics_signals(&mut self) {")
            .expect("the physics-signal consumer remains live");
        let (consumer, _) = consumer
            .split_once("pub fn local_limb_swing")
            .expect("the consumer ends before the public render-state helpers");
        assert_eq!(consumer.matches("water_entry_fanout(entry,").count(), 1);
        assert!(consumer.contains("self.play_sound(fanout.sound);"));
        assert!(consumer.contains("self.spawn_effect_particle(particle, pos, velocity, &[]);"));
    }

    /// `getCompoundTagAt` presents a wrong-type list entry as an empty compound. Its zero id is
    /// consequently a first match when looking up Protection (id zero).
    #[test]
    fn non_compound_enchantment_entry_is_an_empty_compound_for_id_zero() {
        let mut valid_protection = NbtCompound::new();
        valid_protection.insert("id", Nbt::Short(0));
        valid_protection.insert("lvl", Nbt::Short(4));
        let mut nbt = NbtCompound::new();
        nbt.insert(
            "ench",
            Nbt::list(vec![Nbt::Int(1), Nbt::Compound(valid_protection)]),
        );
        let stack = mc_protocol::item::ItemStack {
            nbt: Some(nbt),
            ..plain()
        };
        assert_eq!(enchantment_level(&stack, 0), 0);
    }

    /// java: `EntityLivingBase.getFallSoundString` — the split is at `> 4`, not `>= 4`.
    #[test]
    fn the_fall_sound_switches_to_the_big_variant_above_four_damage() {
        // java: `EntityPlayer.getFallSoundString` overrides the `game.neutral` default.
        assert_eq!(fall_sound_for(1), "game.player.hurt.fall.small");
        assert_eq!(fall_sound_for(4), "game.player.hurt.fall.small");
        assert_eq!(fall_sound_for(5), "game.player.hurt.fall.big");
    }

    #[test]
    fn portal_trigger_is_one_shot_for_stay_and_reentry_after_decay() {
        let mut rng = mc_math::JavaRandom::new(0x504F5254414C);
        let mut time = 0.0;
        let mut previous = 0.0;

        let entry = advance_portal_signal(&mut time, &mut previous, true, None, &mut rng)
            .expect("the first portal contact must trigger sound");
        assert_eq!(previous, 0.0);
        assert_eq!(time, 0.0125);
        assert_eq!(
            entry.event,
            mc_assets::ResourceLocation::minecraft(PORTAL_TRIGGER_SOUND)
        );
        assert_eq!(entry.volume, 0.25);
        assert_eq!(entry.attenuation, mc_audio::AttenuationType::None);
        assert!(entry.position.is_none());
        assert!(!entry.repeat);
        assert!((0.8..1.2).contains(&entry.pitch));

        let stay = advance_portal_signal(&mut time, &mut previous, true, None, &mut rng);
        assert!(
            stay.is_none(),
            "held portal contact must not replay each tick"
        );
        assert_eq!(previous, 0.0125);
        assert_eq!(time, 0.025);

        let no_contact = advance_portal_signal(&mut time, &mut previous, false, None, &mut rng);
        assert!(no_contact.is_none());
        assert_eq!(previous, 0.025);
        assert_eq!(time, 0.0);

        let reentry = advance_portal_signal(&mut time, &mut previous, true, None, &mut rng)
            .expect("contact after a full decay must trigger again");
        assert_eq!(previous, 0.0);
        assert_eq!(time, 0.0125);
        assert_eq!(
            reentry.event,
            mc_assets::ResourceLocation::minecraft(PORTAL_TRIGGER_SOUND)
        );
    }

    #[test]
    fn portal_trigger_entry_edge_catches_post_increment_and_no_contact_mutants() {
        let mut rng = mc_math::JavaRandom::new(7);
        let mut previous = 0.0;
        let mut time = 0.0;

        // A post-increment `timeInPortal == 0` check would miss this edge entirely.
        assert!(advance_portal_signal(&mut time, &mut previous, true, None, &mut rng).is_some());

        let mut no_contact_previous = 0.0;
        let mut no_contact_time = 0.0;
        assert!(advance_portal_signal(
            &mut no_contact_time,
            &mut no_contact_previous,
            false,
            None,
            &mut rng,
        )
        .is_none());
        assert_eq!(no_contact_time, 0.0);
        assert_eq!(no_contact_previous, 0.0);
    }

    #[test]
    fn portal_nausea_ramp_has_the_strict_duration_gate_and_portal_precedence() {
        let mut rng = mc_math::JavaRandom::new(7);

        // A contact tick wins over nausea, including an active-but-zero-duration map entry.
        let mut portal_time = 0.0;
        let mut portal_previous = 0.0;
        let contact = advance_portal_signal(
            &mut portal_time,
            &mut portal_previous,
            true,
            Some(0),
            &mut rng,
        );
        assert!(contact.is_some());
        assert_eq!(portal_previous, 0.0);
        assert_eq!(portal_time, 0.0125);

        // `duration == 60` is the three-second cutoff: it decays instead of taking the nausea
        // branch. This catches a tempting `>= 60` mutant.
        let mut cutoff_time = 0.5;
        let mut cutoff_previous = 0.0;
        assert!(advance_portal_signal(
            &mut cutoff_time,
            &mut cutoff_previous,
            false,
            Some(60),
            &mut rng,
        )
        .is_none());
        assert_eq!(cutoff_previous, 0.5);
        assert_eq!(cutoff_time, 0.45);

        let mut nausea_time = 0.5;
        let mut nausea_previous = 0.0;
        assert!(advance_portal_signal(
            &mut nausea_time,
            &mut nausea_previous,
            false,
            Some(61),
            &mut rng,
        )
        .is_none());
        assert_eq!(nausea_previous, 0.5);
        assert_eq!(nausea_time, 0.50666666);

        // The long-lived nausea branch clamps at one, rather than overflowing the overlay input.
        let mut capped_time = 0.999;
        let mut capped_previous = 0.0;
        advance_portal_signal(
            &mut capped_time,
            &mut capped_previous,
            false,
            Some(61),
            &mut rng,
        );
        assert_eq!(capped_time, 1.0);
    }
}
