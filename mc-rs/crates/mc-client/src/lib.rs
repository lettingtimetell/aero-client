#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! The client: connection state machine, world sync, and the game-state lifecycle.
//!
//! Its modules:
//!
//! | Module | What |
//! |---|---|
//! | [`net`], [`sync`], [`outbound`], [`session`], [`error`] | connection state machine & world sync |
//! | [`app`], `main.rs` | integration: end-to-end join & play |
//!
//! `anyhow` is permitted in this crate and nowhere else in the workspace; library
//! code here still prefers a typed error, with `anyhow` reserved for the binary's top level.
//!
//! # Where the async/sync seam falls, and why
//!
//! By design, `tokio` is for the network only and the game loop is synchronous. That leaves
//! one real design decision: **which side of the channel does packet decoding happen on?**
//!
//! This crate decodes on the **network** side and applies on the **game** side:
//!
//! ```text
//!   tokio task                     channels                  synchronous game loop
//!  ┌────────────────────────┐                              ┌──────────────────────────┐
//!  │ socket → decrypt →     │  ToGame  (bounded mpsc)      │ Session::poll()          │
//!  │ deframe → decompress → ├─────────────────────────────►│  → sync::apply()         │
//!  │ S2CPlay::read          │                              │  → mc_world::World       │
//!  │                        │  ToNet  (unbounded mpsc)     │  → hecs ECS              │
//!  │ writer.send_payload    │◄─────────────────────────────┤ Session::send()          │
//!  └────────────────────────┘                              └──────────────────────────┘
//! ```
//!
//! * **Decoding belongs to the network task** because it is pure, it needs no game state, and
//!   it is the only place that can *see* a framing error early enough to kill the connection
//!   before the byte stream is unrecoverable.
//! * **Application belongs to the game loop** because [`mc_world::World`] is owned there.
//!   The arena-index rule forbids `Rc<RefCell<…>>` in the world/entity graph, and the natural
//!   alternative — an `Arc<Mutex<World>>` shared with the network task — would put a lock on
//!   the mesher's hot path and make the frame block on the socket after all. The world is never
//!   shared; only decoded packets cross the boundary.
//! * The **direct inbound handoff is bounded**, while overflow remains ordered in a network-task
//!   FIFO. This mirrors vanilla's `scheduledTasks`: a stalled frame cannot stop the network task
//!   from reading and handling KeepAlive packets. Like vanilla, a completely wedged game loop can
//!   grow that backlog until it resumes.
//! * The **outbound channel is unbounded**, which is what makes "never block the frame"
//!   literally true: [`tokio::sync::mpsc::UnboundedSender::send`] is synchronous, non-blocking,
//!   and legal to call from a thread with no runtime.
//!
//! Five vanilla handlers skip `PacketThreadUtil.checkThreadAndEnqueue`. Three never reach the
//! game loop because the connection task can perform their effects directly; header/footer and
//! resource-pack use a Netty-bypass lane that is drained before scheduled packets, while their
//! UI/world effects remain safely game-thread-owned. See [`net::task`] for the per-packet
//! justification.
//!
//! # Layout
//!
//! | Module | Contents |
//! |---|---|
//! | [`error`] | [`ClientError`], the one `thiserror` enum for this crate (§11) |
//! | [`net`] | The tokio connection task, the channel protocol, and the server-list ping |
//! | [`sync`] | [`sync::ClientState`] and the dispatch for all 74 S2C Play packets |
//! | [`outbound`] | Movement cadence (hazard H11), digging, placing, held-item sync |
//! | [`session`] | [`session::Session`]: the game-loop-facing state machine |
//! | [`window_icon`] | The Aero Client logo as the window's icon (title bar, taskbar, Alt+Tab) |
//! | [`packaged`] | The packaged start: the portable package's launcher, in the game |

pub mod app;
pub mod error;
pub mod net;
pub mod outbound;
pub mod packaged;
pub mod session;
pub mod sync;
pub mod window_icon;

pub use app::{Client, ClientConfig, CrashReport, GameState};
pub use error::ClientError;
pub use net::{
    ConnectOptions, Credentials, DisconnectReason, NetHandle, NetRuntime, ServerPing, ToGame, ToNet,
};
pub use outbound::OutboundState;
pub use session::{Phase, Session};
pub use sync::{ClientEvent, ClientState};
