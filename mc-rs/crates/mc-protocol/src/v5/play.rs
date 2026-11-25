//! The protocol-5 Play dispatch enums: [`S2CPlay5`] (65 clientbound packets, `0x00`-`0x40`) and
//! [`C2SPlay5`] (24 serverbound, `0x00`-`0x17`).
//!
//! Ids are the 1.8.9 ones: `EnumConnectionState.PLAY` registers the same classes at the same
//! numbers in both versions, and 1.8.9 only appended clientbound `0x41`-`0x49` and serverbound
//! `0x18`-`0x19`, which are unknown ids here (`IOException("Bad packet id")` in 1.7.10).
//!
//! java17: `network/EnumConnectionState.java:115-207`

use super::c2s::{self, NotInProtocol5, V5Ctx};
use super::chunk;
use super::s2c;
use crate::packet::{Packet, State};
use crate::play::C2SPlay;
use crate::{PacketBuffer, ProtoError};

crate::play::dispatch_enum! {
    /// Every protocol-5 clientbound Play packet, id `0x00`-`0x40`.
    S2CPlay5 {
        KeepAlive(s2c::KeepAlive),
        JoinGame(s2c::JoinGame),
        Chat(s2c::Chat),
        TimeUpdate(s2c::TimeUpdate),
        EntityEquipment(s2c::EntityEquipment),
        SpawnPosition(s2c::SpawnPosition),
        UpdateHealth(s2c::UpdateHealth),
        Respawn(s2c::Respawn),
        PlayerPosLook(s2c::PlayerPosLook),
        HeldItemChange(s2c::HeldItemChange),
        UseBed(s2c::UseBed),
        Animation(s2c::Animation),
        SpawnPlayer(s2c::SpawnPlayer),
        CollectItem(s2c::CollectItem),
        SpawnObject(s2c::SpawnObject),
        SpawnMob(s2c::SpawnMob),
        SpawnPainting(s2c::SpawnPainting),
        SpawnExperienceOrb(s2c::SpawnExperienceOrb),
        EntityVelocity(s2c::EntityVelocity),
        DestroyEntities(s2c::DestroyEntities),
        Entity(s2c::Entity),
        EntityRelMove(s2c::EntityRelMove),
        EntityLook(s2c::EntityLook),
        EntityLookAndRelMove(s2c::EntityLookAndRelMove),
        EntityTeleport(s2c::EntityTeleport),
        EntityHeadLook(s2c::EntityHeadLook),
        EntityStatus(s2c::EntityStatus),
        EntityAttach(s2c::EntityAttach),
        EntityMetadata(s2c::EntityMetadata),
        EntityEffect(s2c::EntityEffect),
        RemoveEntityEffect(s2c::RemoveEntityEffect),
        SetExperience(s2c::SetExperience),
        EntityProperties(s2c::EntityProperties),
        ChunkData(chunk::ChunkData),
        MultiBlockChange(s2c::MultiBlockChange),
        BlockChange(s2c::BlockChange),
        BlockAction(s2c::BlockAction),
        BlockBreakAnimation(s2c::BlockBreakAnimation),
        MapChunkBulk(chunk::MapChunkBulk),
        Explosion(s2c::Explosion),
        Effect(s2c::Effect),
        SoundEffect(s2c::SoundEffect),
        Particles(s2c::Particles),
        ChangeGameState(s2c::ChangeGameState),
        SpawnGlobalEntity(s2c::SpawnGlobalEntity),
        OpenWindow(s2c::OpenWindow),
        CloseWindow(s2c::CloseWindow),
        SetSlot(s2c::SetSlot),
        WindowItems(s2c::WindowItems),
        WindowProperty(s2c::WindowProperty),
        ConfirmTransaction(s2c::ConfirmTransaction),
        UpdateSign(s2c::UpdateSign),
        Maps(s2c::Maps),
        UpdateTileEntity(s2c::UpdateTileEntity),
        SignEditorOpen(s2c::SignEditorOpen),
        Statistics(s2c::Statistics),
        PlayerListItem(s2c::PlayerListItem),
        PlayerAbilities(s2c::PlayerAbilities),
        TabComplete(s2c::TabComplete),
        ScoreboardObjective(s2c::ScoreboardObjective),
        UpdateScore(s2c::UpdateScore),
        DisplayScoreboard(s2c::DisplayScoreboard),
        Teams(s2c::Teams),
        PluginMessage(s2c::PluginMessage),
        Disconnect(s2c::Disconnect),
    }
}

crate::play::dispatch_enum! {
    /// Every protocol-5 serverbound Play packet, id `0x00`-`0x17`.
    C2SPlay5 {
        KeepAlive(c2s::KeepAlive),
        ChatMessage(c2s::ChatMessage),
        UseEntity(c2s::UseEntity),
        Player(c2s::Player),
        PlayerPosition(c2s::PlayerPosition),
        PlayerLook(c2s::PlayerLook),
        PlayerPositionAndLook(c2s::PlayerPositionAndLook),
        PlayerDigging(c2s::PlayerDigging),
        PlayerBlockPlacement(c2s::PlayerBlockPlacement),
        HeldItemChange(c2s::HeldItemChange),
        Animation(c2s::Animation),
        EntityAction(c2s::EntityAction),
        SteerVehicle(c2s::SteerVehicle),
        CloseWindow(c2s::CloseWindow),
        ClickWindow(c2s::ClickWindow),
        ConfirmTransaction(c2s::ConfirmTransaction),
        CreativeInventoryAction(c2s::CreativeInventoryAction),
        EnchantItem(c2s::EnchantItem),
        UpdateSign(c2s::UpdateSign),
        PlayerAbilities(c2s::PlayerAbilities),
        TabComplete(c2s::TabComplete),
        ClientSettings(c2s::ClientSettings),
        ClientStatus(c2s::ClientStatus),
        PluginMessage(c2s::PluginMessage),
    }
}

impl S2CPlay5 {
    /// java17: `Packet.hasPriority()`. `NetworkManager.channelRead0` handles these on the network
    /// thread at once instead of queueing them for the tick: S00 and S40 (S05 overrides the
    /// method too, but returns `false`).
    #[must_use]
    pub fn has_priority(&self) -> bool {
        matches!(self, S2CPlay5::KeepAlive(_) | S2CPlay5::Disconnect(_))
    }
}

impl C2SPlay5 {
    /// The protocol-5 packet that means exactly what a 1.8.9 one does, for call sites shared by
    /// both versions. Packets whose protocol-5 form needs data the 1.8.9 struct lacks, or whose
    /// bytes would mean something else, are refused rather than approximated:
    ///
    /// | 1.8.9 | here |
    /// |---|---|
    /// | C00, C01, C03, C05, C09, C0D, C0E, C0F, C10, C11, C13, C16 | same fields (an `int` keep-alive id, byte window ids, gzip item NBT) |
    /// | C02 Interact, Attack | `int` id, byte action; INTERACT_AT refused |
    /// | C08 | `y` as the byte the server reads (-1..=255 only) |
    /// | C0A | `ctx.entity_id`, type 1 |
    /// | C0B | action ordinal + 1 |
    /// | C0C | flags 1/2 as the jump/sneak booleans; other bits refused |
    /// | C14 | without a target block only |
    /// | C04, C06 (stance), C07 (face sentinels), C12 (plain lines), C15 (difficulty, cape), C17 (payload formats), C18, C19 | refused |
    ///
    /// # Errors
    ///
    /// [`NotInProtocol5`] naming the packet and the reason.
    pub fn from_v47(packet: &C2SPlay, ctx: &V5Ctx) -> Result<C2SPlay5, NotInProtocol5> {
        c2s::from_v47(packet, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2c_ids_are_0x00_to_0x40() {
        let mut ids = S2CPlay5::all_ids();
        assert_eq!(ids.len(), 65);
        ids.sort_unstable();
        assert_eq!(ids, (0x00..=0x40).collect::<Vec<i32>>());
    }

    #[test]
    fn c2s_ids_are_0x00_to_0x17() {
        let mut ids = C2SPlay5::all_ids();
        assert_eq!(ids.len(), 24);
        ids.sort_unstable();
        assert_eq!(ids, (0x00..=0x17).collect::<Vec<i32>>());
    }

    #[test]
    fn ids_appended_by_1_8_9_are_unknown() {
        for id in 0x41..=0x49 {
            assert!(matches!(
                S2CPlay5::read(id, &mut PacketBuffer::new()),
                Err(ProtoError::UnknownPacketId { id: got, .. }) if got == id
            ));
        }
        for id in [0x18, 0x19] {
            assert!(C2SPlay5::read(id, &mut PacketBuffer::new()).is_err());
        }
    }

    #[test]
    fn every_shared_c2s_variant_is_mapped_or_refused() {
        // Exhaustive by construction (from_v47 matches every C2SPlay variant); this pins the
        // refusals that 1.7.10 send sites must cover themselves.
        let ctx = V5Ctx { entity_id: 1 };
        let refused = |p: C2SPlay| C2SPlay5::from_v47(&p, &ctx).is_err();
        assert!(refused(C2SPlay::ClientSettings(
            crate::play::c2s::ClientSettings {
                locale: "en_US".into(),
                view_distance: 8,
                chat_visibility: crate::play::c2s::ChatVisibility::Full,
                chat_colors: true,
                displayed_skin_parts: 0x7F,
            }
        )));
        assert!(!refused(C2SPlay::KeepAlive(crate::play::c2s::KeepAlive {
            key: -1
        })));
    }

    #[test]
    fn priority_is_keep_alive_and_disconnect_only() {
        assert!(S2CPlay5::KeepAlive(s2c::KeepAlive { id: 1 }).has_priority());
        assert!(S2CPlay5::Disconnect(s2c::Disconnect {
            reason: "{}".into()
        })
        .has_priority());
        assert!(!S2CPlay5::SpawnPosition(s2c::SpawnPosition { x: 0, y: 0, z: 0 }).has_priority());
    }
}
