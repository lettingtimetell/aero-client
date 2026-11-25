//! World sync: applying decoded server packets to the client's own state.
//!
//! Everything here is **synchronous and `tokio`-free**. It runs on the game thread, owns its
//! state outright, and can be driven from a test with no runtime at all — which is why the
//! dispatch is unit-testable packet by packet rather than only through a socket.

pub mod dispatch;
pub mod dispatch_v5;
pub mod entities;
pub mod env;
pub mod event;
#[cfg(test)]
pub(crate) mod fuzz;
pub mod players;
pub mod players_v5;
pub mod prebuild;
pub mod scoreboard;
pub mod state;
pub mod watched;

pub use watched::{
    ArmorStandPose, ArmorStandState, DamageWobble, DyeColour, EntityFlags, HorseState, Rotation3,
    SkinParts, Watched,
};

pub use dispatch::apply;
pub use entities::{EntityTracker, RemotePlayer, TrackedEntity, TrackedKind};
pub use env::{BorderState, Environment};
pub use event::{ChatPosition, ClientEvent, ScoreboardEvent, WindowEvent};
pub use players::{PlayerInfo, PlayerList};
pub use scoreboard::{ScoreObjective, Scoreboard, Team};
pub use state::{Abilities, ClientState, LocalPlayer};
