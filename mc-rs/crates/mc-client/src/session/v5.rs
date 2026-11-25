//! The 1.7.10 half of [`Session`]: the per-tick inbound drain and the 1.7.10 packets the
//! session itself owns.
//!
//! # When 1.7.10 applies what the server sends
//!
//! 1.7.10 handles only `S00 KeepAlive` and `S40 Disconnect` as they arrive (the network task
//! does, `net/task_v5.rs`). Every other packet waits in `NetworkManager.receivedPacketsQueue`
//! until `processReceivedPackets` (java17 `network/NetworkManager.java:188-215`) handles at most
//! [`PACKETS_PER_DRAIN`] of them, once per tick: from `PlayerControllerMP.updateController`
//! once a world exists (`client/multiplayer/PlayerControllerMP.java:254-270`, called from
//! `runTick`'s "gameMode" section right after the tick's pick, `client/Minecraft.java:1563-1570`),
//! before that from `GuiConnecting.updateScreen` (`client/multiplayer/GuiConnecting.java:104-117`,
//! the screen-update step of `runTick`, `Minecraft.java:1618-1640`). A frame without a tick
//! applies nothing.
//!
//! Here: [`Session::poll_next`] (the frame poll) keeps a 1.7.10 connection's play messages, in
//! order, in `inbound_v5`; the connection's lifecycle messages stay in the frame poll. The
//! client's tick first takes what arrived since that poll ([`Session::pulls_v5_in_tick`]), then
//! calls [`Session::apply_next_v5`] up to [`PACKETS_PER_DRAIN`] times at those two points
//! ([`Session::drains_v5_in_world`], [`Session::drains_v5_before_world`]), reacting to each
//! message's events before the next, like the frame poll. Once the connection has ended nothing
//! more is applied: `updateController` checks `isChannelOpen()` before draining
//! (`PlayerControllerMP.java:258-269`), so packets that were still queued are dropped.
//!
//! # A bound vanilla does not have
//!
//! java17's queue is an unbounded `ConcurrentLinkedQueue`, drained at most 1001 packets a tick
//! (20,020 a second), so a server sending faster than that makes it grow without end. This
//! queue holds at most [`INBOUND_V5_MAX_MESSAGES`] messages and [`INBOUND_V5_MAX_BYTES`] bytes
//! (see [`queued_bytes`]); past either the client closes the connection with a reason that says
//! so. Both are far above anything a vanilla server sends, so only a hostile one can tell.

use std::collections::VecDeque;

use mc_physics::prediction::PositionCorrection;
use mc_physics::v5::OutboundV5;
use mc_protocol::v5::{s2c, S2CPlay5};

use super::{Phase, Session};
use crate::net::{DisconnectReason, ToGame};
use crate::sync::ClientEvent;

/// java17: `NetworkManager.processReceivedPackets` handles at most this many queued packets per
/// call: `for (int var2 = 1000; !queue.isEmpty() && var2 >= 0; --var2)`
/// (`network/NetworkManager.java:205`).
pub(crate) const PACKETS_PER_DRAIN: usize = 1001;

/// The most 1.7.10 play messages a session holds for its tick drains: 64 ticks of 1.7.10's
/// largest drain (3.2 s at 20 ticks a second). A vanilla join, the largest burst a vanilla
/// server sends, is a few thousand packets and drains within a few ticks.
pub(crate) const INBOUND_V5_MAX_MESSAGES: usize = 64 * PACKETS_PER_DRAIN;

/// The most bytes those messages may hold ([`queued_bytes`]): 512 MiB. Chunk columns dominate.
/// The largest vanilla view (view distance 15, 31 x 31 columns) of full-height columns converts
/// to about 190 MB (961 x 196,864 bytes), and it arrives over many ticks.
pub(crate) const INBOUND_V5_MAX_BYTES: usize = 512 << 20;

/// What a queued message counts towards [`INBOUND_V5_MAX_BYTES`].
const MESSAGE_BYTES: usize = 256;

/// The bytes a queued 1.7.10 message stands for: a fixed amount per message, plus the payloads
/// whose size the protocol does not bound by much: chunk data (by its converted 1.8.9 size once
/// converted, its inflated and compressed size before), plugin message, map and block-change
/// data, and the raw bytes kept for the replay recorder. Every other packet is small or bounded
/// by the decoder's own limits, and the message count bounds how many there are.
#[must_use]
pub(crate) fn queued_bytes(message: &ToGame) -> usize {
    MESSAGE_BYTES
        + match message {
            ToGame::PacketV5(packet) => packet_bytes(packet),
            ToGame::RawPacketV5(tapped) => packet_bytes(&tapped.0) + tapped.1.len(),
            ToGame::ChunkPacketV5(_, bytes) => *bytes,
            ToGame::RawChunkPacketV5(_, bytes, raw) => bytes + raw.len(),
            _ => 0,
        }
}

fn packet_bytes(packet: &S2CPlay5) -> usize {
    match packet {
        S2CPlay5::ChunkData(chunk) => chunk.compressed.len() + chunk.data.len(),
        S2CPlay5::MapChunkBulk(bulk) => {
            bulk.compressed.len()
                + bulk
                    .columns
                    .iter()
                    .map(|column| column.data.len())
                    .sum::<usize>()
        }
        S2CPlay5::PluginMessage(message) => message.data.len(),
        S2CPlay5::Maps(map) => map.data.len(),
        S2CPlay5::MultiBlockChange(change) => change.data.as_ref().map_or(0, Vec::len),
        _ => 0,
    }
}

/// A 1.7.10 session's received-packet queue (java17 `NetworkManager.receivedPacketsQueue`), with
/// the bound described in the module docs.
#[derive(Debug, Default)]
pub(crate) struct InboundV5 {
    /// The messages, in arrival order, each with what it counts towards `bytes`.
    queue: VecDeque<(ToGame, usize)>,
    /// What the queued messages count, in all.
    bytes: usize,
}

impl InboundV5 {
    /// Queue `message`, or hand it back (`Some`) when it would exceed a bound.
    fn push(&mut self, message: ToGame) -> Option<ToGame> {
        let bytes = queued_bytes(&message);
        if self.queue.len() >= INBOUND_V5_MAX_MESSAGES
            || self.bytes.saturating_add(bytes) > INBOUND_V5_MAX_BYTES
        {
            return Some(message);
        }
        self.bytes += bytes;
        self.queue.push_back((message, bytes));
        None
    }

    fn pop(&mut self) -> Option<ToGame> {
        let (message, bytes) = self.queue.pop_front()?;
        self.bytes -= bytes;
        Some(message)
    }

    fn clear(&mut self) {
        self.queue.clear();
        self.bytes = 0;
    }

    /// How many messages wait.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.queue.len()
    }

    /// Queue a message in a test, as [`Session::poll_next`] would.
    #[cfg(test)]
    pub(crate) fn push_back(&mut self, message: ToGame) {
        assert!(
            self.push(message).is_none(),
            "a test message fits the bound"
        );
    }
}

impl Session {
    /// [`Session::poll_next`]'s arm for a 1.7.10 play message: keep it for the tick. A 1.8.9
    /// session never gets one; if it ever did it would wait forever, so it is dropped with an
    /// error. Past the queue's bound the connection is closed (`false`, as for any end).
    pub(super) fn queue_v5(&mut self, message: ToGame, events: &mut Vec<ClientEvent>) -> bool {
        if !self.version.is_1_7_10() {
            tracing::error!(?message, "a 1.7.10 message on a 1.8.9 session; dropped");
            return true;
        }
        if self.inbound_v5.push(message).is_none() {
            return true;
        }
        let waiting = self.inbound_v5.len();
        let bytes = self.inbound_v5.bytes;
        tracing::error!(
            waiting,
            bytes,
            "the server sent more than a 1.7.10 client can apply; disconnecting"
        );
        let reason = DisconnectReason::Kicked {
            reason_json: serde_json::json!({
                "translate": "disconnect.genericReason",
                "with": [format!(
                    "The server sent more than the client can apply ({waiting} packets, \
                     {} MiB waiting)",
                    bytes >> 20
                )],
            })
            .to_string(),
        }
        .with_play_origin();
        self.inbound_v5.clear();
        let _ = self.handle.disconnect();
        self.phase = Phase::Disconnected(reason.clone());
        events.push(ClientEvent::Disconnected(reason));
        false
    }

    /// Whether the tick takes the messages that arrived since the frame poll before it drains: a
    /// live 1.7.10 connection, whose packets Netty queues the moment they arrive (a replay's clock
    /// moves once a frame, and a watched replay is fed by its own player).
    #[must_use]
    pub(crate) fn pulls_v5_in_tick(&self) -> bool {
        self.version.is_1_7_10()
            && !self.handle.is_driven()
            && self.handle.replay_source().is_none()
    }

    /// Whether the tick's in-world drain runs: a 1.7.10 session with a world. java17:
    /// `runTick` calls `updateController` only `if (!isGamePaused && theWorld != null)`, and a
    /// multiplayer game never pauses (`Minecraft.java:1020`).
    #[must_use]
    pub(crate) fn drains_v5_in_world(&self) -> bool {
        self.version.is_1_7_10() && matches!(self.phase, Phase::Playing)
    }

    /// Whether the tick's connecting-screen drain runs: a 1.7.10 session that has logged in but
    /// has no world yet (`GuiConnecting.updateScreen`).
    #[must_use]
    pub(crate) fn drains_v5_before_world(&self) -> bool {
        self.version.is_1_7_10() && matches!(self.phase, Phase::LoggingIn)
    }

    /// Apply the next queued 1.7.10 play message and append its events to `events`, after
    /// reacting to them as [`Session::poll_next`] does. `false` when nothing waits, or once the
    /// connection has ended (the queue is then dropped).
    pub(crate) fn apply_next_v5(&mut self, events: &mut Vec<ClientEvent>) -> bool {
        if !self.phase.is_live() {
            self.inbound_v5.clear();
            return false;
        }
        let Some(message) = self.inbound_v5.pop() else {
            return false;
        };
        match message {
            ToGame::PacketV5(packet) => self.apply_packet_v5(*packet, events),
            // The same application, and the bytes kept for the replay recorder.
            ToGame::RawPacketV5(tapped) => {
                let (packet, raw) = *tapped;
                self.tapped_raw = Some(raw);
                self.apply_packet_v5(packet, events);
            }
            // A chunk packet already in its 1.8.9 form: the 1.8.9 path, whose handler 1.7.10
            // shares (java17 `NetHandlerPlayClient.handleChunkData` / `handleMapChunkBulk`).
            ToGame::ChunkPacketV5(job, _) => self.apply_chunk_packet(&job, events),
            ToGame::RawChunkPacketV5(job, _, raw) => {
                self.tapped_raw = Some(raw);
                self.apply_chunk_packet(&job, events);
            }
            // `poll_next` queues only the four messages above.
            other => tracing::error!(?other, "not a 1.7.10 play message; dropped"),
        }
        true
    }

    /// Apply one protocol-5 packet and react to its events: [`crate::sync::dispatch_v5`] for
    /// the world, then the replies and local-player effects [`Session::react_to`] performs for
    /// those events, as `apply_packet` does for 1.8.9.
    fn apply_packet_v5(&mut self, packet: S2CPlay5, events: &mut Vec<ClientEvent>) {
        self.sync_local_entity_packet_shadow();
        if let S2CPlay5::PlayerPosLook(look) = &packet {
            self.player_pos_look_v5(look, events);
            return;
        }
        // S01 and S07 build the player on the shared path; it becomes the 1.7.10 player before
        // the next packet (often an S08 in the same drain) touches it.
        let builds_player = matches!(packet, S2CPlay5::JoinGame(_) | S2CPlay5::Respawn(_));
        let before = events.len();
        let game_mode_before_packet = self.state.local.game_mode;
        match crate::sync::dispatch_v5::apply_v5(&mut self.state, packet, events) {
            Ok(()) => self.react_to(&events[..], before, game_mode_before_packet),
            Err(error) => {
                tracing::error!(%error, "clientbound packet handler failed; continuing connection");
            }
        }
        if builds_player {
            self.ensure_predictor_v5();
        }
    }

    /// java17: `NetHandlerPlayClient.handlePlayerPosLook` (`client/network/
    /// NetHandlerPlayClient.java:545-566`): no relative flags; the player is placed with
    /// its eye at the packet's y, motion zeroed; the reply is a C06 with the box bottom, the eye
    /// as stance and the packet's own yaw, pitch and on-ground flag; the first one after a
    /// (re)spawn closes the terrain screen.
    ///
    /// Before the player exists vanilla throws on the null player before doing anything, so
    /// nothing happens here either.
    fn player_pos_look_v5(&mut self, look: &s2c::PlayerPosLook, events: &mut Vec<ClientEvent>) {
        self.ensure_predictor_v5();
        let Some(predictor) = self.predictor.as_mut() else {
            tracing::debug!("S08 before the player exists; ignored");
            return;
        };
        let Some(reply) = predictor.apply_correction_v5(
            look.x,
            look.y,
            look.z,
            look.yaw,
            look.pitch,
            look.on_ground,
        ) else {
            tracing::error!("S08 for a player without the 1.7.10 fields; ignored");
            return;
        };
        self.send_v5(crate::outbound::v5::movement_packet(
            OutboundV5::PositionLook {
                x: reply.x,
                y: reply.y,
                stance: reply.stance,
                z: reply.z,
                yaw: reply.yaw,
                pitch: reply.pitch,
                on_ground: reply.on_ground,
            },
        ));
        self.sync_local_entity_packet_shadow();
        // For the client (the camera follows a server-set rotation): an absolute correction to
        // where the player now is. Pushed after the reply, so no 1.8.9 C06 is ever built for it.
        events.push(ClientEvent::PositionCorrection(
            PositionCorrection::from_packet(reply.x, reply.y, reply.z, look.yaw, look.pitch, 0),
        ));
        if !self.state.local.done_loading_terrain {
            self.state.local.done_loading_terrain = true;
            self.terrain_download_ticks = None;
            events.push(ClientEvent::TerrainDownloaded);
        }
    }
}

#[cfg(test)]
mod tests {
    use mc_protocol::chunk as c47;
    use mc_protocol::play::S2CPlay;
    use mc_version::GameVersion;

    use super::*;
    use crate::net::{NetHandle, ToNet};
    use crate::session::JoinSettings;
    use crate::sync::prebuild::PrebuildingChunkPacket;

    fn time_v5() -> ToGame {
        ToGame::PacketV5(Box::new(S2CPlay5::TimeUpdate(s2c::TimeUpdate {
            total_world_time: 1,
            world_time: 1,
        })))
    }

    /// A converted one-section column, counted as `bytes`.
    fn chunk_v5(bytes: usize) -> ToGame {
        let data = vec![0; mc_world::expected_chunk_data_len(1, true, true)];
        let packet = S2CPlay::ChunkData(c47::ChunkData {
            chunk_x: 0,
            chunk_z: 0,
            ground_up_continuous: true,
            primary_bit_mask: 1,
            data,
        });
        let job = PrebuildingChunkPacket::spawn(packet, true).expect("a background thread");
        ToGame::ChunkPacketV5(job, bytes)
    }

    /// The queue holds at most its message bound, and at most its byte bound with chunk columns
    /// counted by their converted size; popping gives the room back.
    #[test]
    fn the_queue_is_bounded_by_messages_and_bytes() {
        let mut queue = InboundV5::default();
        for _ in 0..INBOUND_V5_MAX_MESSAGES {
            assert!(queue.push(time_v5()).is_none());
        }
        assert!(queue.push(time_v5()).is_some(), "one message too many");
        assert!(queue.pop().is_some());
        assert!(queue.push(time_v5()).is_none(), "room again");
        queue.clear();
        assert_eq!((queue.len(), queue.bytes), (0, 0));

        let half = INBOUND_V5_MAX_BYTES / 2;
        assert!(queue.push(chunk_v5(half)).is_none());
        assert!(queue.push(chunk_v5(half)).is_some(), "over the byte bound");
        assert_eq!(queue.bytes, half + MESSAGE_BYTES);
        assert!(queue.pop().is_some());
        assert_eq!(queue.bytes, 0);
    }

    /// Past the bound the session closes the connection with a reason saying why, and keeps
    /// nothing; a 1.8.9 session drops a stray 1.7.10 message instead of keeping it forever.
    #[test]
    fn an_overflowing_queue_closes_the_connection() {
        let (handle, to_game, mut to_net) = NetHandle::new_pair();
        let mut session =
            Session::with_version(handle, JoinSettings::default(), GameVersion::V1_7_10);
        for _ in 0..INBOUND_V5_MAX_MESSAGES {
            session.inbound_v5.push_back(time_v5());
        }
        to_game.try_send(time_v5()).expect("room");
        let mut events = Vec::new();
        assert!(!session.poll_next(&mut events), "the session ended");
        let Phase::Disconnected(reason) = session.phase() else {
            panic!("still live: {:?}", session.phase());
        };
        assert!(reason
            .to_component_json()
            .contains("more than the client can apply"));
        assert!(matches!(events.as_slice(), [ClientEvent::Disconnected(_)]));
        assert_eq!(session.inbound_v5.len(), 0);
        assert!(matches!(to_net.try_recv(), Ok(ToNet::Disconnect)));

        let (handle, to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        to_game.try_send(time_v5()).expect("room");
        assert!(session.poll_next(&mut Vec::new()));
        assert_eq!(session.inbound_v5.len(), 0, "dropped on a 1.8.9 session");
    }
}
