//! A synthetic protocol-47 session, built from `mc-protocol`'s own encoders.
//!
//! # Read this before trusting it
//!
//! **This is not a recording of a real server.** There is no network and no Minecraft server of
//! any kind in this environment, so no genuine capture exists to commit. What the session below
//! proves is that the container, the stream framing, the state machine and the decode/re-encode
//! loop all work on a realistically shaped session — a handshake, a login, a world join,
//! movement, chat, a block change and a kick, in the order and the states a real session puts
//! them in. What it cannot prove is that a real server's bytes decode, because every byte here
//! was produced by the very encoders replay then checks.
//!
//! That circularity is why [`crate::replay`]'s `byte_identical` count is only *half* the
//! evidence. The other half already exists elsewhere and does not share code with this file:
//! `mc-protocol`'s `tests/golden.rs` hand-derives wire bytes from the protocol-47 specification
//! for the packets that gate every connection, independently of this crate's encoder. Together
//! they say "the encoder matches the spec" and "the decoder is the encoder's exact inverse".
//! Only a real capture closes the remaining gap, and when one arrives it drops straight into
//! [`Capture::parse`](crate::Capture::parse) with nothing here to change.

use mc_protocol::play::{c2s, s2c, C2SPlay, S2CPlay};
use mc_protocol::{handshake, login, State};

use crate::capture::{frame_bytes, Capture, CaptureMeta, Direction, Recorder};
use crate::replay::Decoded;

/// The address a synthetic capture records itself against. Deliberately in the reserved
/// `.invalid` TLD so no reader can mistake it for a server that was really contacted.
pub const SYNTHETIC_ADDRESS: &str = "synthetic.invalid:25565";

/// The joining player's name in the synthetic session.
pub const SYNTHETIC_USERNAME: &str = "ParityBot";

/// The joining player's UUID, dashed. Every nibble group differs enough that a byte-order slip
/// in the dashed-string encoding shows up immediately in a diff.
pub const SYNTHETIC_UUID: &str = "01234567-89ab-4cde-8f01-020304050607";

/// One scripted step of the synthetic session.
struct Step {
    direction: Direction,
    state: State,
    micros: u64,
    packet: Decoded,
}

fn script() -> Vec<Step> {
    let s2c_play = |p: S2CPlay| Decoded::PlayClientbound(Box::new(p));
    let c2s_play = |p: C2SPlay| Decoded::PlayServerbound(Box::new(p));

    let mut steps = vec![
        Step {
            direction: Direction::Serverbound,
            state: State::Handshaking,
            micros: 0,
            packet: Decoded::Handshake(handshake::Handshake {
                protocol_version: 47,
                server_address: "synthetic.invalid".into(),
                server_port: 25565,
                next_state: handshake::NextState::Login,
            }),
        },
        Step {
            direction: Direction::Serverbound,
            state: State::Login,
            micros: 1_000,
            packet: Decoded::LoginStart(login::LoginStart {
                username: "ParityBot".into(),
            }),
        },
        Step {
            direction: Direction::Clientbound,
            state: State::Login,
            micros: 40_000,
            packet: Decoded::SetCompression(login::SetCompression { threshold: 256 }),
        },
        Step {
            direction: Direction::Clientbound,
            state: State::Login,
            micros: 41_000,
            packet: Decoded::LoginSuccess(login_success()),
        },
        Step {
            direction: Direction::Clientbound,
            state: State::Play,
            micros: 45_000,
            packet: s2c_play(S2CPlay::JoinGame(s2c::JoinGame {
                entity_id: 12_345,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 2,
                max_players: 20,
                level_type: "default".into(),
                reduced_debug_info: false,
            })),
        },
        Step {
            direction: Direction::Serverbound,
            state: State::Play,
            micros: 46_000,
            packet: c2s_play(C2SPlay::ClientSettings(c2s::ClientSettings {
                locale: "en_US".into(),
                view_distance: 12,
                chat_visibility: c2s::ChatVisibility::Full,
                chat_colors: true,
                displayed_skin_parts: 0x7F,
            })),
        },
        Step {
            direction: Direction::Clientbound,
            state: State::Play,
            micros: 50_000,
            packet: s2c_play(S2CPlay::TimeUpdate(s2c::TimeUpdate {
                total_world_time: 1_234_567,
                world_time: 6_000,
            })),
        },
        Step {
            direction: Direction::Clientbound,
            state: State::Play,
            micros: 51_000,
            // Negative coordinates on purpose: hazard H5's sign-extended 26/12/26 packing only
            // misbehaves away from the origin, so a capture that never leaves the positive
            // octant cannot catch it.
            packet: s2c_play(S2CPlay::BlockChange(s2c::BlockChange {
                position: mc_math::BlockPos::new(-1_204, 71, -33_554_000),
                block_state_id: 1 << 4,
            })),
        },
        Step {
            direction: Direction::Clientbound,
            state: State::Play,
            micros: 52_000,
            packet: s2c_play(S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                x: -1_204.5,
                y: 71.0,
                z: -33_554_000.5,
                yaw: 179.75,
                pitch: -12.5,
                flags: 0,
            })),
        },
    ];

    // Twenty ticks of movement plus the keep-alive exchange, which is the cadence half of
    // hazard H11: the shape of a real session is mostly this, and a replay that only ever sees
    // one of each packet type is not a replay of anything.
    let mut micros = 100_000u64;
    for tick in 0..20i32 {
        steps.push(Step {
            direction: Direction::Serverbound,
            state: State::Play,
            micros,
            packet: c2s_play(C2SPlay::PlayerPositionAndLook(c2s::PlayerPositionAndLook {
                x: -1_204.5 + f64::from(tick) * 0.215_66,
                y: 71.0,
                z: -33_554_000.5,
                yaw: 179.75,
                pitch: -12.5,
                on_ground: true,
            })),
        });
        micros += 50_000;
        if tick % 5 == 0 {
            steps.push(Step {
                direction: Direction::Clientbound,
                state: State::Play,
                micros,
                packet: s2c_play(S2CPlay::KeepAlive(s2c::KeepAlive { id: 1_000 + tick })),
            });
            steps.push(Step {
                direction: Direction::Serverbound,
                state: State::Play,
                micros: micros + 500,
                packet: c2s_play(C2SPlay::KeepAlive(c2s::KeepAlive { key: 1_000 + tick })),
            });
        }
    }

    steps.push(Step {
        direction: Direction::Serverbound,
        state: State::Play,
        micros,
        packet: c2s_play(C2SPlay::ChatMessage(c2s::ChatMessage {
            message: "hello from the parity harness".into(),
        })),
    });
    steps.push(Step {
        direction: Direction::Clientbound,
        state: State::Play,
        micros: micros + 20_000,
        packet: s2c_play(S2CPlay::Chat(s2c::Chat {
            component: r#"{"text":"<ParityBot> hello from the parity harness"}"#.into(),
            position: 0,
        })),
    });
    steps.push(Step {
        direction: Direction::Clientbound,
        state: State::Play,
        micros: micros + 40_000,
        packet: s2c_play(S2CPlay::Disconnect(s2c::Disconnect {
            reason: r#"{"translate":"multiplayer.disconnect.generic"}"#.into(),
        })),
    });
    steps
}

/// Build the whole scripted session as a [`Capture`].
///
/// Deterministic: same frames, same timestamps, same bytes, every call.
#[must_use]
pub fn session() -> Capture {
    let mut rec = Recorder::new(CaptureMeta {
        protocol: 47,
        address: SYNTHETIC_ADDRESS.into(),
        started_at_unix_millis: 1_700_000_000_000,
    });
    for step in script() {
        rec.record_at(
            step.direction,
            step.state,
            step.micros,
            &step.packet.encode(),
        );
    }
    rec.finish()
}

/// The same session as raw, length-prefixed **stream** bytes, split by direction, along with
/// the state each direction was in when each frame was written.
///
/// This is what a live recorder would see coming off a socket, and it is what
/// [`StreamRecorder`](crate::StreamRecorder) is tested against: the transitions
/// `Handshaking → Login → Play` are driven exactly as the real client drives them.
#[must_use]
pub fn session_stream() -> Vec<(Direction, State, Vec<u8>)> {
    script()
        .into_iter()
        .map(|s| (s.direction, s.state, frame_bytes(&s.packet.encode())))
        .collect()
}

/// Build the session's `LoginSuccess`. `uuid` is now
/// a named dependency, so this names `Uuid` directly instead of decoding a constant wire body.
fn login_success() -> login::LoginSuccess {
    login::LoginSuccess {
        uuid: uuid::Uuid::parse_str(SYNTHETIC_UUID).expect("a constant, valid UUID literal"),
        username: SYNTHETIC_USERNAME.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::Replayer;

    #[test]
    fn the_synthetic_session_replays_clean() {
        let capture = session();
        let report = Replayer::new(&capture).run();
        println!("{}", report.summary());
        assert!(report.is_clean(), "{report:#?}");
        assert!(
            capture.len() >= 40,
            "the session should be long enough to be a session, got {}",
            capture.len()
        );
    }

    #[test]
    fn the_synthetic_session_is_byte_for_byte_reproducible() {
        let a = session().encode().expect("encode a");
        let b = session().encode().expect("encode b");
        assert_eq!(a, b, "session() must not depend on a clock or a hash order");
    }

    #[test]
    fn the_session_visits_all_three_states_in_both_directions() {
        let capture = session();
        let mut states: Vec<State> = capture.frames().iter().map(|f| f.state).collect();
        states.dedup();
        assert_eq!(
            states,
            vec![State::Handshaking, State::Login, State::Play],
            "states must advance once, in order, and never go back"
        );
        assert!(capture.frames_in(Direction::Clientbound).count() > 5);
        assert!(capture.frames_in(Direction::Serverbound).count() > 5);
    }

    #[test]
    fn the_synthetic_login_success_round_trips() {
        let p = login_success();
        assert_eq!(p.username, SYNTHETIC_USERNAME);
        assert_eq!(p.uuid.to_string(), SYNTHETIC_UUID);
    }

    #[test]
    fn the_stream_form_and_the_capture_form_carry_the_same_payloads() {
        let capture = session();
        let stream = session_stream();
        assert_eq!(stream.len(), capture.len());
        for (frame, (direction, state, bytes)) in capture.frames().iter().zip(&stream) {
            assert_eq!(frame.direction, *direction);
            assert_eq!(frame.state, *state);
            let (len, used) = crate::capture::read_varint(bytes).expect("length prefix");
            assert_eq!(len as usize, frame.payload.len());
            assert_eq!(&bytes[used..], &frame.payload[..]);
        }
    }
}
