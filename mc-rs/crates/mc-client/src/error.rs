//! The one `thiserror` enum for `mc-client`.
//!
//! `anyhow` is permitted in this crate, but only at the binary's top level; everything in
//! `net`, `sync`, `outbound` and `session` returns [`ClientError`].

use thiserror::Error;

/// Errors produced by the connection state machine and world sync.
///
/// Every variant reachable from decoded server bytes exists so that no code path in this crate
/// has to `unwrap()` on untrusted input (§11). A hostile or merely buggy server produces one of
/// these; it never panics the client.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientError {
    /// A framing, decryption, decompression or packet-decoding failure from `mc-protocol`.
    #[error("mc-client: protocol error: {0}")]
    Protocol(#[from] mc_protocol::ProtoError),

    /// Chunk payload application failed (a length that did not match the declared bitmask).
    #[error("mc-client: world error: {0}")]
    World(#[from] mc_world::WorldError),

    /// The server kicked us in the `Play` state. `reason` is chat-component JSON, exactly as
    /// `mc_protocol::ProtoError::Disconnected` carries it for the `Login` state.
    /// java: `S40PacketDisconnect` → `NetHandlerPlayClient.handleDisconnect`
    #[error("mc-client: disconnected by the server: {reason}")]
    Disconnected {
        /// The kick reason, as chat-component JSON.
        reason: String,
    },

    /// The network task has exited, so nothing can be sent or received any more.
    #[error("mc-client: the connection task is no longer running")]
    NotConnected,

    /// The status response was not the JSON object the status protocol specifies.
    #[error("mc-client: malformed server status JSON: {0}")]
    StatusJson(String),

    /// The status response carried a `favicon` this client could not turn into pixels.
    /// java: `OldServerPinger` logs "Invalid server icon (unknown format)" and drops it.
    #[error("mc-client: invalid server favicon: {0}")]
    Favicon(String),

    /// A chat component could not be parsed by `mc-ui`.
    #[error("mc-client: chat component error: {0}")]
    Chat(#[from] mc_ui::UiError),

    /// A caller requested strict vanilla registry validation for an `S20PacketEntityProperties`
    /// name. The live packet path deliberately retains unknown names for extension consumers,
    /// but this public variant remains available to strict/offline validators.
    #[error("mc-client: unknown entity attribute: {0}")]
    UnknownEntityAttribute(String),

    /// `S20PacketEntityProperties` resolved an existing non-living entity. Vanilla's handler
    /// throws before touching any snapshot; keeping this distinct from an unknown attribute makes
    /// the packet boundary observable without discarding the connection.
    #[error("mc-client: entity properties target is not living: {entity_id}")]
    EntityPropertiesForNonLiving {
        /// The packet's entity id.
        entity_id: i32,
    },

    /// A socket-level failure that never became a `ProtoError`.
    #[error("mc-client: I/O error: {0}")]
    Io(#[from] std::io::Error),
}
