//! Protocol-5 (1.7.10) captures for [`super::ReplaySource`].
//!
//! A capture of a 1.7.10 connection says so in its header (protocol 5,
//! `CaptureSink::to_file_for_protocol`). Its frames decode with protocol 5's packets and are
//! dropped by the same rules as a 1.8.9 capture's (see the module docs of `net::replay`): only
//! clientbound frames, `LoginSuccess` for the login, no keep-alive (answered on the network side)
//! and no disconnect. Each kept packet carries the sky flag of the dimension the stream was in,
//! tracked from `JoinGame`/`Respawn` as the network task tracks it, for its chunk columns.

use mc_parity::capture::{Capture, Direction};
use mc_protocol::connection::LoginOutcome;
use mc_protocol::login::LoginSuccess;
use mc_protocol::v5::S2CPlay5;
use mc_protocol::{Packet, PacketBuffer, ProtoError, State};

use super::{Queue, Recorded, ReplaySummary, Timed};
use crate::error::ClientError;
use crate::sync::state::dimension_has_sky_light;

/// Decode a protocol-5 capture's frames into `packets` and `login`, counting into `summary`.
///
/// # Errors
///
/// [`ProtoError::TrailingBytes`] for a frame longer than its packet, as for a 1.8.9 capture.
pub(super) fn decode_frames(
    capture: &Capture,
    keep_bytes: bool,
    summary: &mut ReplaySummary,
    packets: &mut Queue,
    login: &mut Option<LoginOutcome>,
) -> Result<(), ClientError> {
    let mut sky = true;
    for frame in capture.frames() {
        if frame.direction != Direction::Clientbound {
            continue;
        }
        let (Some(id), Some(body)) = (frame.packet_id(), frame.body()) else {
            summary.decode_failures += 1;
            continue;
        };
        let mut buf = PacketBuffer::from_slice(body);
        match frame.state {
            State::Login if id == LoginSuccess::ID => {
                let Ok(success) = LoginSuccess::read(&mut buf) else {
                    summary.decode_failures += 1;
                    continue;
                };
                ensure_consumed(id, &buf)?;
                *login = Some(LoginOutcome {
                    uuid: success.uuid,
                    username: success.username,
                    compression_threshold: None,
                    encrypted: false,
                });
            }
            State::Play => {
                let Ok(packet) = S2CPlay5::read(id, &mut buf) else {
                    summary.decode_failures += 1;
                    continue;
                };
                ensure_consumed(id, &buf)?;
                match &packet {
                    S2CPlay5::KeepAlive(_) => continue,
                    S2CPlay5::Disconnect(_) => {
                        summary.disconnects_dropped += 1;
                        continue;
                    }
                    S2CPlay5::JoinGame(join) => {
                        sky = dimension_has_sky_light(i32::from(join.dimension));
                    }
                    S2CPlay5::Respawn(respawn) => sky = dimension_has_sky_light(respawn.dimension),
                    S2CPlay5::ChunkData(chunk) => {
                        summary.chunk_data += 1;
                        summary.sections += chunk.primary_bit_mask.count_ones() as usize;
                    }
                    S2CPlay5::MapChunkBulk(bulk) => {
                        summary.chunk_bulk += 1;
                        summary.sections += bulk
                            .columns
                            .iter()
                            .map(|column| column.primary_bit_mask.count_ones() as usize)
                            .sum::<usize>();
                    }
                    _ => {}
                }
                packets.push_back(Timed {
                    micros: frame.micros,
                    packet: Recorded::V5(Box::new(packet), sky),
                    raw: keep_bytes.then(|| Box::from(&frame.payload[..])),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn ensure_consumed(packet_id: i32, buf: &PacketBuffer) -> Result<(), ClientError> {
    if buf.is_empty() {
        Ok(())
    } else {
        Err(ProtoError::TrailingBytes {
            packet_id,
            remaining: buf.remaining(),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use mc_parity::capture::{CaptureMeta, Recorder};
    use mc_protocol::v5::chunk::MapChunkBulk;
    use mc_protocol::v5::s2c;

    use super::super::ReplaySource;
    use super::*;
    use crate::net::ToGame;

    fn payload(packet: &S2CPlay5) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    /// A protocol-5 capture replays as 1.7.10 messages: the login from `LoginSuccess`, packets
    /// as [`ToGame::PacketV5`], a full chunk column converted and unpacking
    /// ([`ToGame::ChunkPacketV5`]), keep-alives and the kick dropped.
    #[test]
    fn a_protocol_5_capture_replays_as_1_7_10_messages() {
        let mut rec = Recorder::new(CaptureMeta {
            protocol: 5,
            address: "test".into(),
            started_at_unix_millis: 0,
        });
        let success = LoginSuccess {
            uuid: uuid::Uuid::from_u128(9),
            username: "Replayed".into(),
        };
        let mut login = PacketBuffer::new();
        login.write_varint(LoginSuccess::ID);
        success.write(&mut login);
        rec.record_at(Direction::Clientbound, State::Login, 0, login.as_slice());
        let mut raw = vec![1u8; 4096];
        raw.extend(std::iter::repeat_n(0u8, 3 * 2048 + 256));
        let packets = [
            S2CPlay5::JoinGame(s2c::JoinGame {
                entity_id: 3,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 1,
                max_players: 8,
                level_type: "default".into(),
            }),
            S2CPlay5::KeepAlive(s2c::KeepAlive { id: 1 }),
            S2CPlay5::MapChunkBulk(
                MapChunkBulk::from_raw(true, &[(0, 0, 1, 0, &raw)]).expect("bulk"),
            ),
            S2CPlay5::Chat(s2c::Chat {
                component: r#"{"text":"hi"}"#.into(),
            }),
            S2CPlay5::Disconnect(s2c::Disconnect {
                reason: r#"{"text":"bye"}"#.into(),
            }),
        ];
        for (micros, packet) in (10..).zip(&packets) {
            rec.record_at(
                Direction::Clientbound,
                State::Play,
                micros,
                &payload(packet),
            );
        }
        let capture = rec.finish();

        let mut source = ReplaySource::from_capture(&capture, "x".into(), 1000).expect("source");
        let summary = source.summary().clone();
        assert_eq!(summary.decode_failures, 0);
        assert_eq!(summary.replayed, 3);
        assert_eq!((summary.chunk_bulk, summary.sections), (1, 1));
        assert_eq!(summary.disconnects_dropped, 1);
        let Some(ToGame::Connected(outcome)) = source.next_message() else {
            panic!("the login comes first");
        };
        assert_eq!(outcome.username, "Replayed");
        source.advance();
        let mut messages = Vec::new();
        while let Some(message) = source.next_message() {
            messages.push(message);
        }
        assert!(matches!(
            messages.as_slice(),
            [
                ToGame::PacketV5(join),
                ToGame::ChunkPacketV5(..),
                ToGame::PacketV5(chat),
            ] if matches!(**join, S2CPlay5::JoinGame(_)) && matches!(**chat, S2CPlay5::Chat(_))
        ));
    }
}
