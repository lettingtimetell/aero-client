//! The `Status` state: server-list ping. Two packets each direction, each direction's ids
//! independent (`0x00`/`0x01` on both sides — see [`crate::State`]'s doc).
//!
//! java: `network/status/**`

use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError, State};

/// `0x00 Request` (serverbound): "send me your status." No fields.
/// java: `C00PacketServerQuery`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Request;

impl Packet for Request {
    const STATE: State = State::Status;
    const ID: i32 = 0x00;

    fn read(_buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Request)
    }

    fn write(&self, _buf: &mut PacketBuffer) {}
}

/// `0x00 Response` (clientbound): the MOTD/player-count/version JSON blob, still serialized —
/// parsing it into a typed `ServerStatusResponse` is a UI concern (S16's server-list screen),
/// not this crate's.
/// java: `S00PacketServerInfo`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// The raw JSON response body.
    pub json: String,
}

impl Packet for Response {
    const STATE: State = State::Status;
    const ID: i32 = 0x00;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Response {
            json: buf.read_string(32767)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.json);
    }
}

/// `0x01 Ping` (serverbound): an opaque round-trip token, echoed back verbatim by [`Pong`].
/// java: `C01PacketPing`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ping {
    /// Client-chosen payload (vanilla uses the local clock, but any `i64` round-trips fine).
    pub payload: i64,
}

impl Packet for Ping {
    const STATE: State = State::Status;
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Ping {
            payload: buf.read_i64()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i64(self.payload);
    }
}

/// `0x01 Pong` (clientbound): echoes [`Ping::payload`].
/// java: `S01PacketPong`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pong {
    /// The value the client sent in [`Ping::payload`].
    pub payload: i64,
}

impl Packet for Pong {
    const STATE: State = State::Status;
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Pong {
            payload: buf.read_i64()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i64(self.payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_has_no_body() {
        let mut buf = PacketBuffer::new();
        Request.write(&mut buf);
        assert!(buf.is_empty());
        let mut read_buf = PacketBuffer::from_slice(&[]);
        assert_eq!(Request::read(&mut read_buf).unwrap(), Request);
    }

    #[test]
    fn response_round_trips() {
        let pkt = Response {
            json: r#"{"version":{"name":"1.8.9","protocol":47}}"#.into(),
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(Response::read(&mut read_buf).unwrap(), pkt);
    }

    #[test]
    fn ping_pong_round_trip() {
        let ping = Ping {
            payload: 0x1122_3344_5566_7788,
        };
        let mut buf = PacketBuffer::new();
        ping.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(Ping::read(&mut read_buf).unwrap(), ping);

        let pong = Pong {
            payload: ping.payload,
        };
        let mut buf = PacketBuffer::new();
        pong.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(Pong::read(&mut read_buf).unwrap(), pong);
    }
}
