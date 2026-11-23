//! Play-state packets: **74** clientbound (`S2CPlay`), **26** serverbound (`C2SPlay`).
//!
//! IDs are `EnumConnectionState`'s actual per-direction registration order (see the crate-level
//! doc for why this is 74/26, not the "77/27" the class file names suggest). Two of
//! them — `0x21 ChunkData` and `0x26 MapChunkBulk` — live in [`crate::chunk`] rather than
//! [`s2c`], since their codec shares a size-computation helper the other packets don't need;
//! [`S2CPlay`] still carries them as ordinary variants.
//!
//! java: `network/play/**`, `EnumConnectionState.PLAY`

pub mod c2s;
pub mod s2c;

use crate::packet::{Packet, State};
use crate::{PacketBuffer, ProtoError};

/// Declare a dispatch enum over a set of packet types that share one (state, direction) id
/// namespace: the enum itself, `id()`, `read(id, buf)`, and `write(&self, buf)`. Each packet's
/// own type already carries its full documentation, so variants are `#[allow(missing_docs)]`
/// (inherited by the macro-generated ones) rather than repeating it.
macro_rules! dispatch_enum {
    ($(#[$meta:meta])* $enum_name:ident { $($variant:ident($ty:path)),+ $(,)? }) => {
        $(#[$meta])*
        #[allow(missing_docs)]
        #[derive(Debug, Clone, PartialEq)]
        pub enum $enum_name {
            $($variant($ty),)+
        }

        impl $enum_name {
            /// This packet's id within its (state, direction) namespace.
            #[must_use]
            pub fn id(&self) -> i32 {
                match self {
                    $($enum_name::$variant(_) => <$ty as Packet>::ID,)+
                }
            }

            /// Decode by wire id, dispatching to the matching packet type's [`Packet::read`].
            /// The frame codec ([`crate::codec`]) and the id itself are read by the caller;
            /// `buf` holds only this packet's body.
            pub fn read(id: i32, buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
                match id {
                    $(<$ty as Packet>::ID => Ok($enum_name::$variant(<$ty as Packet>::read(buf)?)),)+
                    other => Err(ProtoError::UnknownPacketId { state: State::Play, id: other }),
                }
            }

            /// Encode this packet's body (not its id or a length prefix — see [`Packet::write`]).
            pub fn write(&self, buf: &mut PacketBuffer) {
                match self {
                    $($enum_name::$variant(p) => p.write(buf),)+
                }
            }

            /// Every id this enum dispatches, in declaration order. Test-only: exists so the
            /// "every id is unique and contiguous" check doesn't have to hand-maintain a second
            /// copy of the id list that could silently drift from the variants above.
            #[cfg(test)]
            fn all_ids() -> Vec<i32> {
                vec![$(<$ty as Packet>::ID,)+]
            }
        }
    };
}

dispatch_enum! {
    /// Every clientbound Play packet, id `0x00`-`0x49`.
    S2CPlay {
        KeepAlive(s2c::KeepAlive),
        JoinGame(s2c::JoinGame),
        Chat(s2c::Chat),
        TimeUpdate(s2c::TimeUpdate),
        EntityEquipment(s2c::EntityEquipment),
        SpawnPosition(s2c::SpawnPosition),
        UpdateHealth(s2c::UpdateHealth),
        Respawn(s2c::Respawn),
        PlayerPositionLook(s2c::PlayerPositionLook),
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
        ChunkData(crate::chunk::ChunkData),
        MultiBlockChange(s2c::MultiBlockChange),
        BlockChange(s2c::BlockChange),
        BlockAction(s2c::BlockAction),
        BlockBreakAnimation(s2c::BlockBreakAnimation),
        MapChunkBulk(crate::chunk::MapChunkBulk),
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
        ServerDifficulty(s2c::ServerDifficulty),
        CombatEvent(s2c::CombatEvent),
        Camera(s2c::Camera),
        WorldBorder(s2c::WorldBorder),
        Title(s2c::Title),
        SetCompression(s2c::SetCompression),
        PlayerListHeaderFooter(s2c::PlayerListHeaderFooter),
        ResourcePackSend(s2c::ResourcePackSend),
        UpdateEntityNBT(s2c::UpdateEntityNBT),
    }
}

dispatch_enum! {
    /// Every serverbound Play packet, id `0x00`-`0x19`.
    C2SPlay {
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
        Spectate(c2s::Spectate),
        ResourcePackStatus(c2s::ResourcePackStatus),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2c_ids_are_unique_and_span_0x00_to_0x49() {
        let mut ids = S2CPlay::all_ids();
        assert_eq!(ids.len(), 74, "expected 74 clientbound Play packets");
        ids.sort_unstable();
        let expected: Vec<i32> = (0x00..=0x49).collect();
        assert_eq!(ids, expected, "ids must be unique and exactly 0x00..=0x49");
    }

    #[test]
    fn c2s_ids_are_unique_and_span_0x00_to_0x19() {
        let mut ids = C2SPlay::all_ids();
        assert_eq!(ids.len(), 26, "expected 26 serverbound Play packets");
        ids.sort_unstable();
        let expected: Vec<i32> = (0x00..=0x19).collect();
        assert_eq!(ids, expected, "ids must be unique and exactly 0x00..=0x19");
    }

    #[test]
    fn every_s2c_id_round_trips_through_the_dispatch_enum() {
        // One minimal, valid instance per variant, constructed the same way each packet's own
        // unit test does, then run through `S2CPlay::write` -> `S2CPlay::read` to prove the
        // enum's `id()`/`read`/`write` wiring (not just each packet's own codec) is correct.
        let samples: Vec<S2CPlay> = vec![
            S2CPlay::KeepAlive(s2c::KeepAlive { id: 1 }),
            S2CPlay::JoinGame(s2c::JoinGame {
                entity_id: 1,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 0,
                max_players: 8,
                level_type: "default".into(),
                reduced_debug_info: false,
            }),
            S2CPlay::ChunkData(crate::chunk::ChunkData {
                chunk_x: 0,
                chunk_z: 0,
                ground_up_continuous: true,
                primary_bit_mask: 0,
                data: vec![0; 256],
            }),
            S2CPlay::MapChunkBulk(crate::chunk::MapChunkBulk {
                has_sky_light: true,
                columns: vec![],
            }),
            S2CPlay::Disconnect(s2c::Disconnect {
                reason: "{}".into(),
            }),
        ];
        for sample in samples {
            let mut buf = PacketBuffer::new();
            sample.write(&mut buf);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let decoded = S2CPlay::read(sample.id(), &mut read_buf).unwrap();
            assert_eq!(decoded, sample);
            assert!(read_buf.is_empty());
        }
    }

    #[test]
    fn c2s_ids_round_trip_through_the_dispatch_enum() {
        let samples: Vec<C2SPlay> = vec![
            C2SPlay::KeepAlive(c2s::KeepAlive { key: 1 }),
            C2SPlay::ChatMessage(c2s::ChatMessage {
                message: "hi".into(),
            }),
        ];
        for sample in samples {
            let mut buf = PacketBuffer::new();
            sample.write(&mut buf);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let decoded = C2SPlay::read(sample.id(), &mut read_buf).unwrap();
            assert_eq!(decoded, sample);
        }
    }

    #[test]
    fn unknown_s2c_id_errors_not_panics() {
        let mut buf = PacketBuffer::new();
        assert!(matches!(
            S2CPlay::read(0x7F, &mut buf),
            Err(ProtoError::UnknownPacketId {
                state: State::Play,
                id: 0x7F
            })
        ));
    }

    #[test]
    fn unknown_c2s_id_errors_not_panics() {
        let mut buf = PacketBuffer::new();
        assert!(matches!(
            C2SPlay::read(0x7F, &mut buf),
            Err(ProtoError::UnknownPacketId {
                state: State::Play,
                id: 0x7F
            })
        ));
    }
}
