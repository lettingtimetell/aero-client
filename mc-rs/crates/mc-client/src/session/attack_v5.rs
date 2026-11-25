//! 1.7.10 support: `PlayerControllerMP.attackEntity` on an entity the tick's packets removed
//! after the tick's pick.
//!
//! 1.7.10 picks before `updateController` applies the tick's packets (java17
//! `client/Minecraft.java:1564-1569`), so an S13 can remove the picked entity before the click.
//! Java still holds the `Entity` object and runs `attackTargetEntityWithCurrentItem` on it
//! (java17 `multiplayer/PlayerControllerMP.java:373-378`); the tracker no longer has the entity,
//! so [`Session::attack_entity_with_knockback`] cannot tell what it was. The click passes what
//! the pick saw (`app/interact_v5.rs`). 1.8.9 applies the tick's packets before its pick and
//! never gets here.

use super::Session;
use crate::app::interact;

impl Session {
    /// The C02 ATTACK and the local half of `attackTargetEntityWithCurrentItem` for a removed
    /// target, returning whether `ItemStack.hitEntity` ran (the app owns the stack).
    ///
    /// java17 (`entity/player/EntityPlayer.java:1052-1169`): with `var2 > 0.0F || var4 > 0.0F`,
    /// `attackEntityFrom` on the client succeeds for a remote player (`EntityOtherPlayerMP`) and
    /// for the non-living targets [`interact::attack_cancels_sprint`] names, whether or not the
    /// entity is dead; then `if (var3 > 0)` damps the motion by 0.6 and ends the sprint, and a
    /// living target gets `hitEntity`. `knockback_level` is the Knockback level the target
    /// counts (none for a non-living one). The target's own impulse and the hit particles
    /// belong to an entity that no longer exists here and are not drawn.
    pub(crate) fn attack_removed_entity_v5(
        &mut self,
        entity_id: i32,
        kind: Option<mc_data::EntityType>,
        remote_player: bool,
        slot: u8,
        knockback_level: i32,
        held: Option<&mc_protocol::ItemStack>,
    ) -> bool {
        self.player_action(|out, packets| out.attack_entity(entity_id, slot, packets));
        // `var4` is the creature modifier, taken against a living target only; a remote
        // player's creature attribute is `UNDEFINED`.
        let creature = if remote_player {
            interact::undefined_creature_damage_modifier(held)
        } else {
            0.0
        };
        if !(self.local_attack_damage() > 0.0 || creature > 0.0) {
            return false;
        }
        if remote_player || interact::attack_cancels_sprint(kind) {
            self.cancel_sprint_after_attack(knockback_level);
        }
        remote_player
    }
}
