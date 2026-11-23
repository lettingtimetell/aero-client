//! The [`Packet`] trait and the [`State`] state machine.
//!
//! java: `network/Packet.java`, `network/EnumConnectionState.java`

use crate::{PacketBuffer, ProtoError};

/// A connection state. Each state has its own independent packet-ID namespace per direction,
/// exactly as `EnumConnectionState` assigns IDs by per-state, per-direction registration order.
///
/// java: `EnumConnectionState` (`HANDSHAKING(-1)`, `PLAY(0)`, `STATUS(1)`, `LOGIN(2)` — the
/// numbers are the state's own protocol-level id, unrelated to packet ids)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    /// Just after TCP connect, before the client declares whether it wants `Status` or `Login`.
    Handshaking,
    /// Server-list ping: `Request`/`Response`, `Ping`/`Pong`.
    Status,
    /// Encryption, compression negotiation, login success.
    Login,
    /// The state a joined client spends the game in.
    Play,
}

/// A single packet type: a wire id plus a symmetric codec against [`PacketBuffer`].
///
/// `write` does not return a `Result`: every ordinary field a well-formed client constructs is
/// representable on the wire. [`PacketBuffer`] nevertheless records invalid outgoing NBT as a
/// deferred encoder error, which production send paths surface before framing. Java's other
/// write-side failure (`writeString` throwing on a >32767-byte string) is deferred by
/// [`PacketBuffer::write_string`] and surfaced by [`PacketBuffer::finish_write`] before framing.
pub trait Packet: Sized {
    /// The connection-state namespace that registers this packet type.
    ///
    /// `EnumConnectionState` keeps one class-to-state map, independent of packet direction.
    /// Its outbound `NetworkManager.dispatchPacket` selects the map entry from the packet class
    /// and changes the channel state before it encodes the packet.  Keeping that registration
    /// next to the Rust packet type avoids a separate, fallible runtime lookup.  Most packet
    /// types are Play packets; the three earlier protocol modules override this default.
    const STATE: State = State::Play;

    /// This packet's id within its (state, direction) namespace.
    const ID: i32;

    /// Decode this packet's body. `buf` holds exactly this packet's payload (the frame codec
    /// has already stripped the length prefix and packet id), but a malicious or truncated
    /// server message can still run out of bytes mid-field — every error path here is a
    /// [`ProtoError`], never a panic.
    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError>;

    /// Encode this packet's body into `buf`. Does not write the length prefix or packet id;
    /// the frame codec ([`crate::codec`]) and the enum dispatcher ([`crate::play`]) do that.
    fn write(&self, buf: &mut PacketBuffer);
}
