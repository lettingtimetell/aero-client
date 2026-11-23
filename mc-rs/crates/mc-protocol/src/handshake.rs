//! The `Handshaking` state: exactly one packet, sent immediately after TCP connect.
//!
//! java: `network/handshake/client/C00Handshake.java`, `EnumConnectionState.HANDSHAKING`

use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError, State};

/// The state id carried by a [`Handshake`]. Vanilla decodes the full
/// `EnumConnectionState.getById` range (`-1..=2`), even though its handshake handler accepts
/// only [`Status`](NextState::Status) and [`Login`](NextState::Login) and rejects the other two
/// later as invalid intentions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextState {
    /// Handshaking (`-1`); decoded by vanilla, then rejected by the handshake handler.
    Handshaking,
    /// Play (`0`); decoded by vanilla, then rejected by the handshake handler.
    Play,
    /// Server-list ping.
    Status,
    /// Proceed to authentication.
    Login,
}

impl NextState {
    fn from_id(id: i32) -> Result<NextState, ProtoError> {
        match id {
            -1 => Ok(NextState::Handshaking),
            0 => Ok(NextState::Play),
            1 => Ok(NextState::Status),
            2 => Ok(NextState::Login),
            other => Err(ProtoError::InvalidEnum {
                what: "handshake next-state",
                value: other,
            }),
        }
    }

    fn to_id(self) -> i32 {
        match self {
            NextState::Handshaking => -1,
            NextState::Play => 0,
            NextState::Status => 1,
            NextState::Login => 2,
        }
    }
}

/// `0x00 Handshake` (serverbound only — this state has no clientbound packets).
/// java: `C00Handshake`
#[derive(Debug, Clone, PartialEq)]
pub struct Handshake {
    /// The protocol version the client speaks. `47` for 1.8.9; sent as-is even when it won't
    /// match the server's, so the server can reply with its own "outdated client/server" kick.
    pub protocol_version: i32,
    /// The address the client connected to, as typed by the user (not necessarily resolved).
    pub server_address: String,
    /// The port the client connected to.
    pub server_port: u16,
    /// Which state to switch to next.
    pub next_state: NextState,
}

impl Packet for Handshake {
    const STATE: State = State::Handshaking;
    const ID: i32 = 0x00;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let protocol_version = buf.read_varint()?;
        let server_address = buf.read_string(255)?;
        let server_port = buf.read_u16()?;
        let next_state = NextState::from_id(buf.read_varint()?)?;
        Ok(Handshake {
            protocol_version,
            server_address,
            server_port,
            next_state,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.protocol_version);
        buf.write_string(&self.server_address);
        buf.write_u16(self.server_port);
        buf.write_varint(self.next_state.to_id());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let pkt = Handshake {
            protocol_version: 47,
            server_address: "play.example.com".into(),
            server_port: 25565,
            next_state: NextState::Login,
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(Handshake::read(&mut read_buf).unwrap(), pkt);
    }

    #[test]
    fn bad_next_state_errors() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(47);
        buf.write_string("h");
        buf.write_u16(25565);
        buf.write_varint(99);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(Handshake::read(&mut read_buf).is_err());
    }

    #[test]
    fn decodes_full_vanilla_connection_state_id_range() {
        for (id, expected) in [
            (-1, NextState::Handshaking),
            (0, NextState::Play),
            (1, NextState::Status),
            (2, NextState::Login),
        ] {
            let mut buf = PacketBuffer::new();
            buf.write_varint(47);
            buf.write_string("h");
            buf.write_u16(25565);
            buf.write_varint(id);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            assert_eq!(Handshake::read(&mut read_buf).unwrap().next_state, expected);
        }
    }

    #[tokio::test]
    async fn oversized_address_aborts_the_handshake_before_any_wire_bytes() {
        use tokio::io::AsyncReadExt;

        let (client, mut peer) = tokio::io::duplex(1 << 16);
        let mut connection = crate::connection::Connection::new(client);
        let packet = Handshake {
            protocol_version: 47,
            server_address: "x".repeat(32_768),
            server_port: 25_565,
            next_state: NextState::Login,
        };

        assert!(matches!(
            connection.send(&packet).await,
            Err(ProtoError::StringTooLong {
                len: 32_768,
                max: 32_767
            })
        ));
        drop(connection);

        let mut emitted = Vec::new();
        peer.read_to_end(&mut emitted).await.unwrap();
        assert!(
            emitted.is_empty(),
            "vanilla raises EncoderException; a truncated/partial handshake must not escape"
        );
    }
}
