//! The game-state machine: `menu ↔ connecting ↔ playing ↔ disconnected`.
//!
//! java: vanilla has no such type. `Minecraft` tracks the same four states implicitly across
//! three unrelated fields — `theWorld != null`, `currentScreen instanceof GuiConnecting`, and
//! `myNetworkManager` — and every transition is a hand-written combination of
//! `displayGuiScreen`, `loadWorld` and `NetHandlerPlayClient.cleanup` scattered over the file.
//! Making it one enum is the point: a transition that vanilla can get half-done (a world with no
//! connection, a connecting screen over a live world) is unrepresentable here.
//!
//! # Layering, and why this is not [`crate::session::Phase`]
//!
//! [`crate::session::Phase`] is *one connection's* progress and belongs to S14. This is
//! the *client's* state, which includes having no connection at all — the main menu — and
//! outlives any single session. [`StateMachine::on_phase`] is the one-way bridge: the session
//! reports, the client decides.

use crate::net::DisconnectReason;
use crate::session::Phase;

/// Where the client is.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GameState {
    /// No connection. java: `theWorld == null` with `GuiMainMenu` (or any menu) showing.
    Menu,
    /// A selected address is being parsed and resolved before a transport exists.
    Resolving {
        /// Original address text, before parsing, IDN conversion, or SRV lookup.
        address: String,
    },
    /// A connection is in flight: TCP connect, handshake, login, and the join burst up to the
    /// first `S08PacketPlayerPosLook`. java: `GuiConnecting` then `GuiDownloadTerrain`.
    Connecting {
        /// The server being connected to.
        host: String,
        /// Its port.
        port: u16,
    },
    /// In the world. java: `theWorld != null && thePlayer != null`.
    Playing,
    /// The connection ended and the player has not yet gone back to the menu.
    /// java: `GuiDisconnected`.
    Disconnected {
        /// Why, for the screen to render.
        reason: DisconnectReason,
    },
}

impl GameState {
    /// Whether a world exists and should be ticked, meshed and drawn.
    #[must_use]
    pub fn is_in_world(&self) -> bool {
        matches!(self, GameState::Playing)
    }

    /// Whether a [`Session`](crate::Session) is expected to exist in this state.
    ///
    /// A disconnect reason is carried by the transition/event and the teardown effects drop the
    /// session before [`GameState::Disconnected`] is presented, so the disconnected screen is a
    /// screen-only state rather than a retained world.
    #[must_use]
    pub fn has_session(&self) -> bool {
        matches!(self, GameState::Connecting { .. } | GameState::Playing)
    }

    /// A short, stable name for logs, the F3 overlay and crash reports.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            GameState::Menu => "menu",
            GameState::Resolving { .. } => "resolving",
            GameState::Connecting { .. } => "connecting",
            GameState::Playing => "playing",
            GameState::Disconnected { .. } => "disconnected",
        }
    }
}

/// Something that happened, which the machine turns into a state change plus effects.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Transition {
    /// Begin resolving a saved/direct-connect address without opening a transport yet.
    BeginResolve {
        /// Original address text shown by `GuiConnecting`.
        address: String,
    },
    /// Resolution produced the concrete endpoint the transport must open.
    Resolved {
        /// Resolved host name or address.
        host: String,
        /// Resolved socket port.
        port: u16,
    },
    /// The player picked a server. java: `GuiMultiplayer.connectToServer` → `new GuiConnecting`.
    Connect {
        /// Host name or address.
        host: String,
        /// Port.
        port: u16,
    },
    /// `S01PacketJoinGame` landed. java: `handleJoinGame`.
    Joined,
    /// The connection ended, for any reason. java: `GuiDisconnected` from either
    /// `NetHandlerLoginClient.onDisconnect` or `NetHandlerPlayClient.onDisconnect`.
    Ended(DisconnectReason),
    /// The player cancelled a connection in progress. java: `GuiConnecting.actionPerformed`'s
    /// cancel button, which calls `cancelResourceDownload` and returns to the previous screen.
    Cancel,
    /// The player left the world deliberately. java: `GuiIngameMenu`'s "Disconnect" button →
    /// `theWorld.sendQuittingDisconnectingPacket()` then `loadWorld(null)`.
    LeaveWorld,
    /// The player dismissed the disconnect screen. java: `GuiDisconnected`'s "Back to title
    /// screen".
    BackToMenu,
    /// The player quit. java: `Minecraft.shutdown()` sets `running = false`.
    Quit,
}

/// What the host must actually do as a result of a transition.
///
/// Returned rather than performed, so the machine itself owns no sockets, no GPU and no files
/// and can be exhaustively tested in microseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Effect {
    /// Spawn a connection task. java: `new GuiConnecting`'s background thread.
    OpenConnection {
        /// Host name or address.
        host: String,
        /// Port.
        port: u16,
    },
    /// Ask the connection to close. java: `NetworkManager.closeChannel`.
    CloseConnection,
    /// Throw the world, the ECS and every chunk mesh away. java: `loadWorld(null)`.
    UnloadWorld,
    /// Leave the run loop. java: `Minecraft.running = false`.
    Quit,
}

/// A transition that does not exist. Reporting it rather than panicking matters: several of
/// these are reachable from *server* behaviour (a `Disconnect` before login finishes, a second
/// `JoinGame`) and a client must not die because a server was odd.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("mc-client: {transition:?} is not legal in state {state}")]
pub struct IllegalTransition {
    /// The state the machine was in.
    pub state: &'static str,
    /// What was attempted.
    pub transition: Transition,
}

/// The state machine itself.
#[derive(Debug, Clone)]
pub struct StateMachine {
    state: GameState,
    running: bool,
    joins: u64,
    disconnects: u64,
}

impl Default for StateMachine {
    fn default() -> Self {
        StateMachine::new()
    }
}

impl StateMachine {
    /// A client sitting on the main menu. java: `Minecraft.startGame:577`.
    #[must_use]
    pub fn new() -> StateMachine {
        StateMachine {
            state: GameState::Menu,
            running: true,
            joins: 0,
            disconnects: 0,
        }
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &GameState {
        &self.state
    }

    /// Whether the run loop should keep going. java: `Minecraft.running`.
    #[must_use]
    pub fn running(&self) -> bool {
        self.running
    }

    /// How many times a world has been joined. Used by the leak check to prove the cycles
    /// actually happened.
    #[must_use]
    pub fn joins(&self) -> u64 {
        self.joins
    }

    /// How many connections have ended.
    #[must_use]
    pub fn disconnects(&self) -> u64 {
        self.disconnects
    }

    /// Apply a transition.
    ///
    /// # Errors
    ///
    /// [`IllegalTransition`] when the transition does not exist from the current state. The
    /// machine is left untouched, so a caller can log and carry on.
    pub fn apply(&mut self, transition: Transition) -> Result<Vec<Effect>, IllegalTransition> {
        use GameState as S;
        use Transition as T;

        let effects = match (&self.state, &transition) {
            (S::Menu, T::BeginResolve { address }) => {
                self.state = S::Resolving {
                    address: address.clone(),
                };
                Vec::new()
            }
            (S::Disconnected { .. }, T::BeginResolve { address }) => {
                self.state = S::Resolving {
                    address: address.clone(),
                };
                vec![Effect::UnloadWorld]
            }
            (S::Resolving { .. }, T::Resolved { host, port }) => {
                self.state = S::Connecting {
                    host: host.clone(),
                    port: *port,
                };
                vec![Effect::OpenConnection {
                    host: host.clone(),
                    port: *port,
                }]
            }
            // ---- menu → connecting ----
            (S::Menu, T::Connect { host, port }) => {
                let effects = vec![Effect::OpenConnection {
                    host: host.clone(),
                    port: *port,
                }];
                self.state = S::Connecting {
                    host: host.clone(),
                    port: *port,
                };
                effects
            }
            // java: `GuiDisconnected` offers only "Back to title screen", but the multiplayer
            // screen behind it can be reached in one click, so a reconnect from the disconnected
            // state is a real path. It must close the dead session first.
            (S::Disconnected { .. }, T::Connect { host, port }) => {
                let effects = vec![
                    Effect::UnloadWorld,
                    Effect::OpenConnection {
                        host: host.clone(),
                        port: *port,
                    },
                ];
                self.state = S::Connecting {
                    host: host.clone(),
                    port: *port,
                };
                effects
            }

            // ---- connecting → playing ----
            (S::Connecting { .. }, T::Joined) => {
                self.joins += 1;
                self.state = S::Playing;
                Vec::new()
            }

            // ---- anything live → disconnected ----
            (S::Resolving { .. }, T::Ended(reason)) => {
                self.disconnects += 1;
                self.state = S::Disconnected {
                    reason: reason.clone(),
                };
                vec![Effect::UnloadWorld]
            }
            (S::Connecting { .. } | S::Playing, T::Ended(reason)) => {
                self.disconnects += 1;
                self.state = S::Disconnected {
                    reason: reason.clone(),
                };
                // java: NetHandlerPlayClient.onDisconnect -> Minecraft.loadWorld(null) runs
                // before GuiDisconnected is displayed. The host applies these in order: ask the
                // transport to close, then drop the session/world and its client-owned mirrors.
                vec![Effect::CloseConnection, Effect::UnloadWorld]
            }

            // ---- deliberate departures ----
            (S::Resolving { .. }, T::Cancel) => {
                self.state = S::Menu;
                vec![Effect::UnloadWorld]
            }
            (S::Connecting { .. }, T::Cancel) => {
                self.state = S::Menu;
                vec![Effect::CloseConnection, Effect::UnloadWorld]
            }
            (S::Playing, T::LeaveWorld) => {
                self.disconnects += 1;
                self.state = S::Menu;
                vec![Effect::CloseConnection, Effect::UnloadWorld]
            }
            (S::Disconnected { .. }, T::BackToMenu) => {
                self.state = S::Menu;
                vec![Effect::UnloadWorld]
            }

            // ---- quit, from anywhere ----
            (state, T::Quit) => {
                self.running = false;
                let mut effects = Vec::new();
                if state.has_session() {
                    effects.push(Effect::CloseConnection);
                    effects.push(Effect::UnloadWorld);
                }
                effects.push(Effect::Quit);
                effects
            }

            (state, _) => {
                return Err(IllegalTransition {
                    state: state.name(),
                    transition,
                })
            }
        };
        tracing::debug!(state = self.state.name(), ?effects, "state transition");
        Ok(effects)
    }

    /// Fold a session's [`Phase`] into the machine, ignoring anything that is not news.
    ///
    /// This is the bridge S14's session crosses each frame. It is deliberately *idempotent*:
    /// `Phase` is a level, not an edge, so the same phase arriving twice must not count two
    /// joins or two disconnects.
    pub fn on_phase(&mut self, phase: &Phase) -> Vec<Effect> {
        let transition = match (&self.state, phase) {
            (GameState::Connecting { .. }, Phase::Playing) => Transition::Joined,
            (GameState::Connecting { .. } | GameState::Playing, Phase::Disconnected(reason)) => {
                Transition::Ended(reason.clone())
            }
            _ => return Vec::new(),
        };
        // Both arms above are legal by construction, so a failure here is a logic bug rather
        // than something a server can cause; log it and change nothing.
        self.apply(transition).unwrap_or_else(|error| {
            tracing::error!(%error, "phase bridge produced an illegal transition");
            Vec::new()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connect(machine: &mut StateMachine) -> Vec<Effect> {
        machine
            .apply(Transition::Connect {
                host: "mc.example.com".to_owned(),
                port: 25_565,
            })
            .expect("connecting from the menu is legal")
    }

    #[test]
    fn the_happy_path_walks_menu_connecting_playing_menu() {
        let mut machine = StateMachine::new();
        assert_eq!(*machine.state(), GameState::Menu);
        assert!(machine.running());

        assert_eq!(
            connect(&mut machine),
            vec![Effect::OpenConnection {
                host: "mc.example.com".to_owned(),
                port: 25_565
            }]
        );
        assert!(matches!(machine.state(), GameState::Connecting { .. }));
        assert!(!machine.state().is_in_world());
        assert!(machine.state().has_session());

        assert_eq!(machine.apply(Transition::Joined).unwrap(), vec![]);
        assert_eq!(*machine.state(), GameState::Playing);
        assert!(machine.state().is_in_world());
        assert_eq!(machine.joins(), 1);

        assert_eq!(
            machine.apply(Transition::LeaveWorld).unwrap(),
            vec![Effect::CloseConnection, Effect::UnloadWorld]
        );
        assert_eq!(*machine.state(), GameState::Menu);
        assert!(!machine.state().has_session());
        assert_eq!(machine.disconnects(), 1);
    }

    #[test]
    fn a_kick_lands_in_disconnected_and_back_to_menu_unloads_the_world() {
        let mut machine = StateMachine::new();
        connect(&mut machine);
        machine.apply(Transition::Joined).unwrap();

        let reason = DisconnectReason::Kicked {
            reason_json: r#"{"text":"bye"}"#.to_owned(),
        };
        assert_eq!(
            machine.apply(Transition::Ended(reason.clone())).unwrap(),
            vec![Effect::CloseConnection, Effect::UnloadWorld]
        );
        assert_eq!(*machine.state(), GameState::Disconnected { reason });
        assert!(
            !machine.state().has_session(),
            "the disconnect screen must not retain the dead session/world"
        );
        assert!(!machine.state().is_in_world());

        assert_eq!(
            machine.apply(Transition::BackToMenu).unwrap(),
            vec![Effect::UnloadWorld]
        );
        assert_eq!(*machine.state(), GameState::Menu);
    }

    /// A login that fails before `JoinGame` never reaches `Playing` — the disconnect must still
    /// be accepted, and it must not count as a join.
    #[test]
    fn a_connection_that_fails_before_join_goes_straight_to_disconnected() {
        let mut machine = StateMachine::new();
        connect(&mut machine);
        machine
            .apply(Transition::Ended(DisconnectReason::ConnectionLost {
                detail: "refused".to_owned(),
            }))
            .unwrap();
        assert!(matches!(machine.state(), GameState::Disconnected { .. }));
        assert_eq!(machine.joins(), 0);
        assert_eq!(machine.disconnects(), 1);
    }

    #[test]
    fn phase_bridge_preserves_play_origin_while_still_connecting() {
        let mut machine = StateMachine::new();
        connect(&mut machine);
        let reason = DisconnectReason::ConnectionLost {
            detail: "EOF before JoinGame".to_owned(),
        }
        .with_play_origin();
        let title = reason.title_key();
        let phase = Phase::Disconnected(reason.clone());

        machine.on_phase(&phase);
        assert_eq!(*machine.state(), GameState::Disconnected { reason });
        assert_eq!(title, "disconnect.lost");
    }

    #[test]
    fn cancelling_a_connection_returns_to_the_menu_and_closes_the_socket() {
        let mut machine = StateMachine::new();
        connect(&mut machine);
        assert_eq!(
            machine.apply(Transition::Cancel).unwrap(),
            vec![Effect::CloseConnection, Effect::UnloadWorld]
        );
        assert_eq!(*machine.state(), GameState::Menu);
    }

    #[test]
    fn reconnecting_from_the_disconnect_screen_unloads_first() {
        let mut machine = StateMachine::new();
        connect(&mut machine);
        machine
            .apply(Transition::Ended(DisconnectReason::ClosedByClient))
            .unwrap();
        let effects = machine
            .apply(Transition::Connect {
                host: "other".to_owned(),
                port: 1234,
            })
            .unwrap();
        assert_eq!(
            effects,
            vec![
                Effect::UnloadWorld,
                Effect::OpenConnection {
                    host: "other".to_owned(),
                    port: 1234
                }
            ],
            "the dead world must go before the new one arrives"
        );
    }

    /// Every state must accept `Quit`, and only the states that own a session may close one.
    #[test]
    fn quit_is_legal_from_every_state_and_closes_only_what_exists() {
        let mut machine = StateMachine::new();
        assert_eq!(
            machine.apply(Transition::Quit).unwrap(),
            vec![Effect::Quit],
            "quitting from the menu has nothing to close"
        );
        assert!(!machine.running());

        for build in [
            (|m: &mut StateMachine| {
                connect(m);
            }) as fn(&mut StateMachine),
            |m: &mut StateMachine| {
                connect(m);
                m.apply(Transition::Joined).unwrap();
            },
            |m: &mut StateMachine| {
                connect(m);
                m.apply(Transition::Ended(DisconnectReason::ClosedByClient))
                    .unwrap();
            },
        ] {
            let mut machine = StateMachine::new();
            build(&mut machine);
            assert_eq!(
                machine.apply(Transition::Quit).unwrap(),
                if machine.state().name() == "disconnected" {
                    vec![Effect::Quit]
                } else {
                    vec![Effect::CloseConnection, Effect::UnloadWorld, Effect::Quit]
                },
                "state {} must tear its session down",
                machine.state().name()
            );
            assert!(!machine.running());
        }
    }

    /// Illegal transitions must be reported, not panic, and must leave the machine untouched —
    /// several of these are reachable from a badly-behaved *server*.
    #[test]
    fn illegal_transitions_are_rejected_without_changing_state() {
        let cases: Vec<(&str, StateMachine, Transition)> = vec![
            ("menu", StateMachine::new(), Transition::Joined),
            (
                "menu",
                StateMachine::new(),
                Transition::Ended(DisconnectReason::ClosedByClient),
            ),
            ("menu", StateMachine::new(), Transition::LeaveWorld),
            ("menu", StateMachine::new(), Transition::Cancel),
            ("menu", StateMachine::new(), Transition::BackToMenu),
            (
                "playing",
                {
                    let mut m = StateMachine::new();
                    connect(&mut m);
                    m.apply(Transition::Joined).unwrap();
                    m
                },
                // A second JoinGame on the same connection: vanilla would rebuild the world.
                Transition::Joined,
            ),
            (
                "playing",
                {
                    let mut m = StateMachine::new();
                    connect(&mut m);
                    m.apply(Transition::Joined).unwrap();
                    m
                },
                Transition::Connect {
                    host: "x".to_owned(),
                    port: 1,
                },
            ),
        ];

        for (expected_state, mut machine, transition) in cases {
            let before = machine.state().clone();
            let error = machine
                .apply(transition.clone())
                .expect_err("must be rejected");
            assert_eq!(error.state, expected_state);
            assert_eq!(error.transition, transition);
            assert_eq!(*machine.state(), before, "state must be untouched");
            assert!(machine.running());
        }
    }

    /// The bridge from S14: `Phase` is a level, so the same phase twice must not double-count.
    #[test]
    fn the_phase_bridge_is_idempotent() {
        let mut machine = StateMachine::new();
        connect(&mut machine);

        assert!(machine.on_phase(&Phase::Connecting).is_empty());
        assert!(machine.on_phase(&Phase::LoggingIn).is_empty());
        assert!(matches!(machine.state(), GameState::Connecting { .. }));

        machine.on_phase(&Phase::Playing);
        assert_eq!(*machine.state(), GameState::Playing);
        assert_eq!(machine.joins(), 1);
        machine.on_phase(&Phase::Playing);
        machine.on_phase(&Phase::Playing);
        assert_eq!(machine.joins(), 1, "the same phase must not re-join");

        let reason = DisconnectReason::ClosedByClient;
        machine.on_phase(&Phase::Disconnected(reason.clone()));
        machine.on_phase(&Phase::Disconnected(reason));
        assert_eq!(machine.disconnects(), 1, "the same phase must not re-count");
        assert!(matches!(machine.state(), GameState::Disconnected { .. }));
    }

    /// Ten full connect/disconnect cycles must leave the machine exactly where it started, with
    /// the counters as the only difference. This is the state-machine half of the leak check.
    #[test]
    fn repeated_cycles_return_to_the_menu_every_time() {
        let mut machine = StateMachine::new();
        for cycle in 0..10 {
            connect(&mut machine);
            machine.apply(Transition::Joined).unwrap();
            assert_eq!(*machine.state(), GameState::Playing);
            machine
                .apply(Transition::Ended(DisconnectReason::ClosedByClient))
                .unwrap();
            machine.apply(Transition::BackToMenu).unwrap();
            assert_eq!(*machine.state(), GameState::Menu, "cycle {cycle}");
        }
        assert_eq!(machine.joins(), 10);
        assert_eq!(machine.disconnects(), 10);
        assert!(machine.running());
    }
}
