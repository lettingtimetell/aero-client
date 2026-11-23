//! The channel protocol between the tokio connection task and the synchronous game loop.
//!
//! Two enums and one handle. [`ToGame`] travels network → game over a **bounded** channel;
//! [`ToNet`] travels game → network over an **unbounded** one. See the crate-level docs for why
//! the capacities differ.

use mc_protocol::connection::LoginOutcome;
use mc_protocol::play::{C2SPlay, S2CPlay};
use tokio::sync::mpsc;

use crate::error::ClientError;

/// How many decoded packets may sit directly in the inbound queue before the network task spills
/// further packets into its ordered overflow FIFO.
///
/// Sized generously enough that a normal join burst (`MapChunkBulk` plus a few hundred entity
/// spawns) normally stays on the direct path. java: vanilla has no equivalent limit — the main
/// thread's scheduled-task queue is unbounded. `play_loop` reproduces that semantic when this
/// bounded handoff fills so Netty-thread work such as KeepAlive cannot be back-pressured.
pub const INBOUND_CAPACITY: usize = 4096;

/// A message from the connection task to the game loop.
#[derive(Debug)]
#[non_exhaustive]
pub enum ToGame {
    /// The TCP socket is open and the login handshake is about to start. Always the first
    /// message for a live connection; replay handles begin at [`ToGame::Connected`] because a
    /// capture has no in-progress socket phase.
    SocketConnected,
    /// The login sequence finished; the connection is in `Play`. The first message carrying
    /// session state.
    Connected(LoginOutcome),
    /// One decoded clientbound Play packet, ready to be applied to the world.
    Packet(S2CPlay),
    /// A full-column chunk packet (`S21` with `ground_up_continuous` and a non-zero bitmask, or
    /// an `S26`) whose columns a background thread is unpacking. Applied exactly where the same
    /// [`ToGame::Packet`] would have been; see [`crate::sync::prebuild`] (#15 / NET-1).
    ChunkPacket(std::sync::Arc<crate::sync::prebuild::PrebuildingChunkPacket>),
    /// The connection ended, for any reason. Always the last message.
    Disconnected(DisconnectReason),
    /// A [`ToGame::Packet`] plus the exact bytes it was decoded from (packet
    /// id varint then body, decrypted and decompressed). Sent instead of `Packet` only while the
    /// replay recorder's tap is on ([`NetHandle::set_raw_tap`]); the session applies the packet
    /// exactly as a `Packet` and keeps the bytes for the recorder
    /// ([`crate::session::Session::take_tapped_raw`]).
    ///
    /// Boxed so that `ToGame` stays exactly as large as before the tap existed: every message
    /// on the channel pays for its largest variant, tap on or off.
    RawPacket(Box<(S2CPlay, Box<[u8]>)>),
    /// A [`ToGame::ChunkPacket`] plus its exact bytes; see
    /// [`ToGame::RawPacket`].
    RawChunkPacket(
        std::sync::Arc<crate::sync::prebuild::PrebuildingChunkPacket>,
        Box<[u8]>,
    ),
}

impl ToGame {
    /// Attach a packet's exact bytes (the replay recorder's tap) to a decoded
    /// message. Anything but a packet, or no bytes, passes through unchanged.
    #[must_use]
    pub(crate) fn with_raw(self, raw: Option<Box<[u8]>>) -> ToGame {
        match (self, raw) {
            (ToGame::Packet(packet), Some(raw)) => ToGame::RawPacket(Box::new((packet, raw))),
            (ToGame::ChunkPacket(job), Some(raw)) => ToGame::RawChunkPacket(job, raw),
            (message, _) => message,
        }
    }
}

/// Which vanilla network handler owned the disconnect.
///
/// The UI cannot infer this from its current screen: a `Play` connection can end before
/// `JoinGame`, while `GuiConnecting` is still visible. Keep this finite and explicit at the
/// network boundary so the origin survives the Session and client-state event bridges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DisconnectOrigin {
    /// `NetHandlerLoginClient` / `GuiConnecting` owns the failure.
    Login,
    /// `NetHandlerPlayClient` owns the failure, including a pre-`JoinGame` Play EOF/kick.
    Play,
}

/// Why the connection ended. Surfaced to the UI (S16's disconnect screen) verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DisconnectReason {
    /// The server sent `S40PacketDisconnect` (Play) or `S00PacketDisconnect` (Login). The
    /// payload is chat-component JSON; render it with `mc_ui::font::component::ChatComponent`.
    Kicked {
        /// The kick reason, as chat-component JSON.
        reason_json: String,
    },
    /// The transport failed, or the server sent something undecodable. `detail` remains
    /// available to logs, but is never rendered: Rust/OS error prose is not a stable,
    /// localizable disconnect component.
    ConnectionLost {
        /// A human-readable description of the failure.
        detail: String,
    },
    /// The client asked to leave ([`ToNet::Disconnect`]).
    ClosedByClient,
    /// The reason plus the protocol handler that produced it.
    ///
    /// This is a wrapper rather than a field on each legacy reason variant so existing reason
    /// constructors remain source-compatible for replay and headless callers.
    WithOrigin {
        /// The underlying server/transport reason.
        reason: Box<DisconnectReason>,
        /// The owning vanilla protocol handler.
        origin: DisconnectOrigin,
    },
}

impl DisconnectReason {
    /// Attach the protocol handler origin to this reason.
    #[must_use]
    pub fn with_origin(self, origin: DisconnectOrigin) -> Self {
        DisconnectReason::WithOrigin {
            reason: Box::new(self),
            origin,
        }
    }

    /// Mark this reason as a login/connecting failure.
    #[must_use]
    pub fn with_login_origin(self) -> Self {
        self.with_origin(DisconnectOrigin::Login)
    }

    /// Mark this reason as a Play disconnect.
    #[must_use]
    pub fn with_play_origin(self) -> Self {
        self.with_origin(DisconnectOrigin::Play)
    }

    /// Return the owning protocol handler. Legacy unwrapped reasons are Play reasons, matching
    /// the historical `play_loop` return contract.
    #[must_use]
    pub fn origin(&self) -> DisconnectOrigin {
        match self {
            DisconnectReason::WithOrigin { origin, .. } => *origin,
            _ => DisconnectOrigin::Play,
        }
    }

    /// The vanilla disconnect-screen title key for this protocol origin.
    #[must_use]
    pub fn title_key(&self) -> &'static str {
        match self.origin() {
            DisconnectOrigin::Login => "connect.failed",
            DisconnectOrigin::Play => "disconnect.lost",
        }
    }

    /// The reason as chat-component JSON, so the UI has one uniform thing to render.
    ///
    /// A transport failure has no server-provided component, so one is synthesised with
    /// vanilla's `disconnect.genericReason` translation rather than exposing Rust or OS prose.
    #[must_use]
    pub fn to_component_json(&self) -> String {
        match self {
            DisconnectReason::Kicked { reason_json } => reason_json.clone(),
            DisconnectReason::ConnectionLost { .. } => serde_json::json!({
                "translate": "disconnect.genericReason",
                "with": ["Internal Exception: java.io.IOException"],
            })
            .to_string(),
            DisconnectReason::ClosedByClient => {
                serde_json::json!({ "translate": "disconnect.endOfStream" }).to_string()
            }
            DisconnectReason::WithOrigin { reason, .. } => reason.to_component_json(),
        }
    }
}

/// A command from the game loop to the connection task.
#[derive(Debug)]
#[non_exhaustive]
pub enum ToNet {
    /// Serialise and send one serverbound Play packet.
    Send(C2SPlay),
    /// Close the connection cleanly. The task answers with
    /// [`DisconnectReason::ClosedByClient`] and exits.
    Disconnect,
}

/// Where a handle's inbound messages come from.
///
/// Two producers satisfy the same contract from the game loop's side. [`Inbound::Live`] is the
/// tokio connection task on a real socket; [`Inbound::Replay`] is a `.mccap` file read straight
/// off disk on the game thread. The game loop cannot tell them apart and
/// does not need to: everything above [`NetHandle`] sees the same [`ToGame`] stream either way.
#[derive(Debug)]
enum Inbound {
    /// The tokio connection task's end of the bounded channel.
    Live {
        /// Handlers vanilla executes on the Netty event loop, but whose effects need the
        /// single-thread-owned Rust game state. Once the connection task has decoded the direct
        /// handler, this lane bypasses the scheduled-packet queue so the next game-loop poll sees
        /// it before every deferred packet still waiting.
        immediate: mpsc::UnboundedReceiver<ToGame>,
        /// Every `PacketThreadUtil.checkThreadAndEnqueue`-style packet.
        scheduled: mpsc::Receiver<ToGame>,
    },
    /// A capture, plus the outbound receiver kept alive so that queueing a packet still
    /// succeeds — a replay has nowhere to send, but the game loop must not start reporting
    /// [`ClientError::NotConnected`] and taking a different code path than a live session does.
    Replay {
        source: Box<crate::net::replay::ReplaySource>,
        sink: mpsc::UnboundedReceiver<ToNet>,
    },
    /// An Aero replay being watched. The playback engine
    /// ([`NetHandle::push_driven`]) decides which recorded packet the session sees next and when;
    /// there is no socket and no task, and whatever the game queues for the server is dropped
    /// on the next [`NetHandle::advance`].
    Driven {
        queue: std::collections::VecDeque<ToGame>,
        sink: mpsc::UnboundedReceiver<ToNet>,
        /// How many outbound messages the game produced and this handle dropped.
        dropped: u64,
    },
}

/// The game loop's end of the two channels.
///
/// Every method here is non-blocking and safe to call from a thread that is not inside a tokio
/// runtime — which is the entire point, since the frame must never await.
#[derive(Debug)]
pub struct NetHandle {
    inbound: Inbound,
    outbound: mpsc::UnboundedSender<ToNet>,
    closed: bool,
    /// While set, packets bound for the game carry their exact bytes
    /// ([`ToGame::RawPacket`]). Shared with the connection task (or read by a capture source);
    /// one relaxed load per packet while it is clear.
    raw_tap: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl NetHandle {
    /// Pair a handle with the halves the connection task keeps.
    #[cfg(test)]
    pub(crate) fn new_pair() -> (
        NetHandle,
        mpsc::Sender<ToGame>,
        mpsc::UnboundedReceiver<ToNet>,
    ) {
        let (handle, to_game, _immediate, to_net) = Self::new_live_pair();
        (handle, to_game, to_net)
    }

    /// Like [`Self::new_pair`], but also returns the task-only Netty-bypass lane.
    ///
    /// Only the connection task receives `immediate`: callers that model a game-thread packet
    /// injection must use the ordinary scheduled sender returned by [`Self::new_pair`].
    pub(crate) fn new_live_pair() -> (
        NetHandle,
        mpsc::Sender<ToGame>,
        mpsc::UnboundedSender<ToGame>,
        mpsc::UnboundedReceiver<ToNet>,
    ) {
        let (to_game_tx, to_game_rx) = mpsc::channel(INBOUND_CAPACITY);
        let (immediate_tx, immediate_rx) = mpsc::unbounded_channel();
        let (to_net_tx, to_net_rx) = mpsc::unbounded_channel();
        (
            NetHandle {
                inbound: Inbound::Live {
                    immediate: immediate_rx,
                    scheduled: to_game_rx,
                },
                outbound: to_net_tx,
                closed: false,
                raw_tap: std::sync::Arc::default(),
            },
            to_game_tx,
            immediate_tx,
            to_net_rx,
        )
    }

    /// A handle driven by a capture file rather than a socket.
    ///
    /// There is no task and no runtime: [`NetHandle::try_recv`] reads the next decoded packet
    /// out of the source, and [`NetHandle::advance`] is what makes the source's virtual clock
    /// move. See [`crate::net::replay`] for why the cadence is counted in polls.
    #[must_use]
    pub fn replay(source: crate::net::replay::ReplaySource) -> NetHandle {
        let (to_net_tx, to_net_rx) = mpsc::unbounded_channel();
        NetHandle {
            inbound: Inbound::Replay {
                source: Box::new(source),
                sink: to_net_rx,
            },
            outbound: to_net_tx,
            closed: false,
            raw_tap: std::sync::Arc::default(),
        }
    }

    /// Turn the replay recorder's tap on or off: while on, every packet the
    /// game receives carries its exact bytes ([`ToGame::RawPacket`]). Takes effect from the next
    /// packet the connection task decodes (packets already queued arrive without bytes).
    pub fn set_raw_tap(&self, on: bool) {
        self.raw_tap.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the tap is on.
    #[must_use]
    pub fn raw_tap(&self) -> bool {
        self.raw_tap.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The tap flag, for the connection task this handle belongs to.
    pub(crate) fn raw_tap_flag(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        std::sync::Arc::clone(&self.raw_tap)
    }

    /// The capture driving this handle, if it is a replay rather than a live connection.
    #[must_use]
    pub fn replay_source(&self) -> Option<&crate::net::replay::ReplaySource> {
        match &self.inbound {
            Inbound::Live { .. } | Inbound::Driven { .. } => None,
            Inbound::Replay { source, .. } => Some(source),
        }
    }

    /// A handle for watching an Aero replay. Nothing reaches the
    /// game until [`Self::push_driven`] queues it, and nothing the game sends leaves the process.
    #[must_use]
    pub fn driven() -> NetHandle {
        let (to_net_tx, to_net_rx) = mpsc::unbounded_channel();
        NetHandle {
            inbound: Inbound::Driven {
                queue: std::collections::VecDeque::new(),
                sink: to_net_rx,
                dropped: 0,
            },
            outbound: to_net_tx,
            closed: false,
            raw_tap: std::sync::Arc::default(),
        }
    }

    /// Queue one message for the next [`Self::try_recv`] of a [`Self::driven`] handle. Returns
    /// `false` (and drops the message) for any other kind of handle.
    pub fn push_driven(&mut self, message: ToGame) -> bool {
        match &mut self.inbound {
            Inbound::Driven { queue, .. } => {
                queue.push_back(message);
                true
            }
            Inbound::Live { .. } | Inbound::Replay { .. } => false,
        }
    }

    /// Whether this is a [`Self::driven`] handle.
    #[must_use]
    pub fn is_driven(&self) -> bool {
        matches!(self.inbound, Inbound::Driven { .. })
    }

    /// For a [`Self::driven`] handle: how many outbound messages the game produced (and this
    /// handle dropped, since a watched replay has no server). `None` for other handles.
    #[must_use]
    pub fn driven_dropped(&self) -> Option<u64> {
        match &self.inbound {
            Inbound::Driven { dropped, .. } => Some(*dropped),
            Inbound::Live { .. } | Inbound::Replay { .. } => None,
        }
    }

    /// Let one poll's worth of time pass.
    ///
    /// A no-op for a live connection, whose cadence is the socket's. For a replay it advances
    /// the virtual clock by one fixed step and discards whatever the game loop queued for a
    /// server that is not there — bounded, because an unbounded queue nobody drains is a leak
    /// over a 30 000-frame run.
    ///
    /// Called once per [`Session::poll`](crate::session::Session::poll), which the game loop
    /// calls once per frame. That is the whole of the replay's clock.
    pub fn advance(&mut self) {
        match &mut self.inbound {
            Inbound::Replay { source, sink } => {
                source.advance();
                while sink.try_recv().is_ok() {}
            }
            // A watched replay has no server: drop what the game queued for one.
            Inbound::Driven { sink, dropped, .. } => {
                while sink.try_recv().is_ok() {
                    *dropped += 1;
                }
            }
            Inbound::Live { .. } => {}
        }
    }

    /// Take the next inbound message, or `None` if none is ready **or** the task has exited.
    ///
    /// Never blocks. Distinguishing "nothing yet" from "gone" is what [`NetHandle::is_closed`]
    /// is for; in practice the task always sends a [`ToGame::Disconnected`] before exiting, so
    /// the caller learns the reason through the queue rather than through a closed channel.
    pub fn try_recv(&mut self) -> Option<ToGame> {
        let inbound = match &mut self.inbound {
            // The two direct Netty handlers must not wait behind a join burst once decoded. On
            // Java they mutate the client in `channelRead0` instead of entering the scheduled-task
            // queue; this is the closest safe representation while `ClientState` remains
            // deliberately non-Send and game-thread-owned.
            Inbound::Live {
                immediate,
                scheduled,
            } => match immediate.try_recv() {
                Ok(message) => Ok(message),
                Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                    scheduled.try_recv()
                }
            },
            // A spent capture reports `Empty`, never `Disconnected`: the session stays in
            // `Playing` with its world resident so the rest of the run keeps measuring the
            // scene the file built. See `net::replay`'s module header.
            // A capture that kept its frames' bytes attaches them while the
            // recorder's tap is on, exactly like the connection task does.
            Inbound::Replay { source, .. } => source
                .next_message_tapped(self.raw_tap.load(std::sync::atomic::Ordering::Relaxed))
                .ok_or(mpsc::error::TryRecvError::Empty),
            Inbound::Driven { queue, .. } => {
                queue.pop_front().ok_or(mpsc::error::TryRecvError::Empty)
            }
        };
        match inbound {
            Ok(msg) => {
                if matches!(msg, ToGame::Disconnected(_)) {
                    self.closed = true;
                }
                Some(msg)
            }
            Err(mpsc::error::TryRecvError::Empty) => None,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                self.closed = true;
                None
            }
        }
    }

    /// Queue one packet for the connection task.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotConnected`] once the task has exited. Note that this is *not* an
    /// acknowledgement of delivery: the packet is queued, not yet on the wire.
    pub fn send(&self, packet: C2SPlay) -> Result<(), ClientError> {
        self.outbound
            .send(ToNet::Send(packet))
            .map_err(|_| ClientError::NotConnected)
    }

    /// Ask the connection task to close the connection.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotConnected`] if the task has already exited.
    pub fn disconnect(&self) -> Result<(), ClientError> {
        self.outbound
            .send(ToNet::Disconnect)
            .map_err(|_| ClientError::NotConnected)
    }

    /// Whether the connection task has finished.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed || self.outbound.is_closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnect_reason_always_renders_as_component_json() {
        let kicked = DisconnectReason::Kicked {
            reason_json: r#"{"translate":"multiplayer.disconnect.kicked"}"#.into(),
        };
        assert_eq!(
            kicked.to_component_json(),
            r#"{"translate":"multiplayer.disconnect.kicked"}"#
        );

        // The diagnostic remains in the reason for logging, but never reaches UI JSON.
        let lost = DisconnectReason::ConnectionLost {
            detail: r#"bad "packet" \ id"#.into(),
        };
        let value: serde_json::Value =
            serde_json::from_str(&lost.to_component_json()).expect("valid JSON");
        assert_eq!(value["translate"], "disconnect.genericReason");
        assert_eq!(
            value["with"],
            serde_json::json!(["Internal Exception: java.io.IOException"])
        );
        assert!(!lost.to_component_json().contains("packet"));
    }

    #[test]
    fn disconnect_origin_survives_reason_rendering_and_selects_the_handler_title() {
        let reason = DisconnectReason::ConnectionLost {
            detail: "pre-JoinGame EOF".into(),
        }
        .with_play_origin();
        assert_eq!(reason.origin(), DisconnectOrigin::Play);
        assert_eq!(reason.title_key(), "disconnect.lost");
        assert_eq!(
            reason.to_component_json(),
            DisconnectReason::ConnectionLost {
                detail: "pre-JoinGame EOF".into(),
            }
            .to_component_json()
        );

        let login = DisconnectReason::ConnectionLost {
            detail: "resolver failed".into(),
        }
        .with_login_origin();
        assert_eq!(login.origin(), DisconnectOrigin::Login);
        assert_eq!(login.title_key(), "connect.failed");
    }

    #[test]
    fn a_handle_with_no_task_reports_not_connected() {
        let (handle, to_game, to_net) = NetHandle::new_pair();
        drop(to_game);
        drop(to_net);
        assert!(handle.is_closed());
        assert!(matches!(
            handle.send(C2SPlay::KeepAlive(mc_protocol::play::c2s::KeepAlive {
                key: 1
            })),
            Err(ClientError::NotConnected)
        ));
    }

    #[test]
    fn try_recv_is_non_blocking_and_reports_empty_as_none() {
        let (mut handle, to_game, _to_net) = NetHandle::new_pair();
        assert!(handle.try_recv().is_none());
        to_game
            .try_send(ToGame::Disconnected(DisconnectReason::ClosedByClient))
            .expect("queue has room");
        assert!(matches!(handle.try_recv(), Some(ToGame::Disconnected(_))));
        assert!(handle.is_closed());
    }
}
