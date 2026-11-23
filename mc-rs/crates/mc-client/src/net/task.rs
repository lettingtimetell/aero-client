//! The tokio connection task: connect, log in, then pump the full-duplex `Play` stream.
//!
//! java: `network/NetworkManager.java` (the Netty pipeline and its two queues) plus the handful
//! of `NetHandlerPlayClient` handlers that run on the Netty thread.
//!
//! # What stays on this side of the channel
//!
//! Vanilla routes almost every `Play` packet to the main thread through
//! `PacketThreadUtil.checkThreadAndEnqueue`. Exactly five handlers do **not**. Three have effects
//! that the connection task can perform outright, while the other two use its Netty-bypass lane:
//!
//! | Packet | java | Why it cannot wait for a frame |
//! |---|---|---|
//! | `0x00 KeepAlive` | `handleKeepAlive` (`NetHandlerPlayClient.java:1663`) | The server kicks after 30 s of silence. A hitching frame must not become a kick. |
//! | `0x46 SetCompression` | `handleSetCompressionLevel` (`:1592`) | It changes how the **next frame off the socket** is decoded. Deferring it corrupts the stream permanently. |
//! | `0x40 Disconnect` | `handleDisconnect` (`:785`) | It closes the channel; nothing after it is meaningful. |
//! | `0x47 PlayerListHeaderFooter` | `handlePlayerListHeaderFooter` (`:1600`) | Bypasses scheduled packets; state remains game-thread-owned. |
//! | `0x48 ResourcePackSend` | `handleResourcePack` (`:1701`) | Bypasses scheduled packets so the status/prompt path is not backlog-delayed. |
//!
//! Everything else is decoded here and applied by [`crate::sync`] on the game thread. If its
//! bounded channel fills, this task preserves overflow in an ordered FIFO while continuing to
//! read the socket, matching vanilla's unbounded scheduled-task queue without making frame work
//! unbounded.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::time::Duration;
use std::{pin::Pin, task::Context, task::Poll};

use mc_protocol::connection::{
    handshake, login_offline, login_with_auth_policy, Connection, ConnectionReader,
    ConnectionWriter, Direction as LoginDirection, LoginAuthPolicy,
};
use mc_protocol::handshake::NextState;
use mc_protocol::play::{c2s, s2c, C2SPlay, S2CPlay};
use mc_protocol::{PacketBuffer, ProtoError};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::mpsc::{Sender, UnboundedReceiver};

use crate::error::ClientError;
use crate::net::capture::CaptureSink;
use crate::net::channel::{DisconnectOrigin, DisconnectReason, ToGame, ToNet};

/// Timestamp captured at the transport boundary, before framing or decode.
type ActivityInstant = tokio::time::Instant;

/// How the client should authenticate to this server.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Credentials {
    /// Online mode: complete the key exchange and register the join with the Mojang session
    /// server. Boxed because `mc_auth::Session` carries an access token and dwarfs the other
    /// variant.
    Online(Box<mc_auth::Session>),
    /// Offline (`online-mode=false`) mode, or a dev build: send this name and never contact the
    /// session server. An online-mode server will kick us at `hasJoined`, which is the honest
    /// failure — see `mc_protocol::connection::login_offline`.
    Offline {
        /// The name to log in with.
        username: String,
    },
}

impl Credentials {
    /// The name that will appear in `C00PacketLoginStart`.
    #[must_use]
    pub fn username(&self) -> &str {
        match self {
            Credentials::Online(session) => &session.username,
            Credentials::Offline { username } => username,
        }
    }
}

/// Everything the connection task needs to reach `Play`.
#[derive(Debug, Clone)]
pub struct ConnectOptions {
    /// Host as the user typed it. Echoed verbatim in the handshake, because some proxies route
    /// on it. java: `C00Handshake.ip`
    pub host: String,
    /// TCP port, and the value echoed in the handshake.
    pub port: u16,
    /// How to authenticate.
    pub credentials: Credentials,
    /// Session-server failure policy selected from vanilla's `ServerData.isOnLAN()` bit.
    /// Saved/direct servers remain strict; only discovered LAN entries opt into tolerance.
    pub auth_policy: LoginAuthPolicy,
    /// Optional Java SOCKS5 proxy for the session-server join request.
    pub proxy: Option<mc_auth::SocksProxy>,
    /// Where to write a `.mccap` recording of this session, if anywhere.
    ///
    /// `None` — the default from both constructors — falls back to the `MC_CAPTURE` environment
    /// variable, which is how a user turns recording on. The field exists so a *test* can ask
    /// for a capture at a known path without mutating process-global environment state.
    /// See [`crate::net::capture`].
    pub capture: Option<std::path::PathBuf>,
    /// Where the session-server join goes instead of Mojang's, for tests only.
    ///
    /// `None` (both constructors) is `sessionserver.mojang.com`. Nothing in the client sets it:
    /// no option, flag or environment variable reaches it, so the access token can only ever go
    /// to Mojang. Tests point it at a fake session server on loopback through
    /// [`ConnectOptions::with_session_join_url`].
    pub session_join_url: Option<String>,
}

impl ConnectOptions {
    /// Offline-mode options for `host:port`.
    #[must_use]
    pub fn offline(host: impl Into<String>, port: u16, username: impl Into<String>) -> Self {
        ConnectOptions {
            host: host.into(),
            port,
            credentials: Credentials::Offline {
                username: username.into(),
            },
            auth_policy: LoginAuthPolicy::Strict,
            proxy: None,
            capture: None,
            session_join_url: None,
        }
    }

    /// Online-mode options for `host:port`, joining as `session`.
    #[must_use]
    pub fn online(host: impl Into<String>, port: u16, session: mc_auth::Session) -> Self {
        ConnectOptions {
            host: host.into(),
            port,
            credentials: Credentials::Online(Box::new(session)),
            auth_policy: LoginAuthPolicy::Strict,
            proxy: None,
            capture: None,
            session_join_url: None,
        }
    }

    /// Select vanilla's LAN-only tolerance for session-server authentication failures.
    #[must_use]
    pub fn with_lan_auth_tolerance(mut self, lan: bool) -> Self {
        self.auth_policy = if lan {
            LoginAuthPolicy::AllowLanAuthFailure
        } else {
            LoginAuthPolicy::Strict
        };
        self
    }

    /// Route session-server HTTP(S) requests through the launcher's Java-compatible SOCKS5
    /// proxy. The game socket itself remains direct.
    #[must_use]
    pub fn with_proxy(mut self, proxy: Option<mc_auth::SocksProxy>) -> Self {
        self.proxy = proxy;
        self
    }

    /// Record this session to `path`, overriding the `MC_CAPTURE` environment variable.
    #[must_use]
    pub fn capture_to(mut self, path: impl Into<std::path::PathBuf>) -> ConnectOptions {
        self.capture = Some(path.into());
        self
    }

    /// Send the session-server join to `url` instead of Mojang (tests only; see
    /// [`ConnectOptions::session_join_url`]).
    #[doc(hidden)]
    #[must_use]
    pub fn with_session_join_url(mut self, url: impl Into<String>) -> ConnectOptions {
        self.session_join_url = Some(url.into());
        self
    }

    /// The session-server client for this connection's join: the launcher's SOCKS proxy if
    /// any, Mojang's endpoints unless a test redirected the join.
    fn session_client(&self) -> Result<mc_auth::AuthClient, mc_auth::AuthError> {
        let client = mc_auth::AuthClient::new_with_proxy(
            mc_auth::AuthConfig::default(),
            self.proxy.clone(),
        )?;
        Ok(match &self.session_join_url {
            Some(url) => client.with_endpoints(mc_auth::Endpoints {
                session_join: url.clone(),
                ..mc_auth::Endpoints::default()
            }),
            None => client,
        })
    }
}

/// Connect, log in, and pump packets until either side hangs up.
///
/// This never returns an error: every failure is reported to the game loop as a
/// [`ToGame::Disconnected`], because a disconnect is a normal outcome the UI has to render
/// rather than something a caller could handle. The task is spawned and forgotten.
pub(crate) async fn run(
    opts: ConnectOptions,
    to_game: Sender<ToGame>,
    immediate_to_game: tokio::sync::mpsc::UnboundedSender<ToGame>,
    from_game: UnboundedReceiver<ToNet>,
    raw_tap: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    // Plan §13's S22 -> S21 row. Off unless the caller asked or `MC_CAPTURE` names a file;
    // see `net::capture`.
    let address = format!("{}:{}", opts.host, opts.port);
    let mut capture = match &opts.capture {
        Some(path) => CaptureSink::to_file(path.clone(), &address),
        None => CaptureSink::from_env(&address),
    };
    // The replay recorder's tap.
    capture.set_raw_tap(raw_tap);
    // java: GuiConnecting.connect logs "Connecting to <ip>, <port>" at INFO. Without it, a
    // failed login in a player's log had nothing to say which server it was.
    tracing::info!(host = %opts.host, port = opts.port, "connecting");
    let (reason, origin) = match connect_and_login(&opts, &to_game, &mut capture).await {
        Ok((conn, inbound_activity)) => {
            // The one place the client transitions state itself: `login` has just returned, so
            // the connection is in `Play` and every frame from here on is a Play frame.
            capture.set_state(mc_protocol::State::Play);
            let (reader, writer) = conn.split();
            let reason = play_loop(
                reader,
                writer,
                &to_game,
                &immediate_to_game,
                from_game,
                inbound_activity,
                READ_TIMEOUT,
                &mut capture,
            )
            .await;
            (reason, DisconnectOrigin::Play)
        }
        Err(reason) => (reason, DisconnectOrigin::Login),
    };
    capture.finish();
    // Best-effort: if the game loop has already dropped the handle there is nobody to tell.
    let _ = to_game
        .send(ToGame::Disconnected(reason.with_origin(origin)))
        .await;
}

/// TCP connect → `ToGame::SocketConnected` → handshake(next=Login) → login →
/// `ToGame::Connected`.
///
/// `connection::login`'s new observer hook is wired to `capture`
/// here, the same sink `play_loop` feeds `Play`-state frames into. `capture.set_state(Login)`
/// happens the instant `handshake` returns — the same place `mc_protocol::connection::login`
/// itself transitions the *connection's* state, mirroring how `run` already sets `Play` the
/// instant `login` returns. `handshake`'s own single `Handshake` frame is not captured: it has
/// no observer hook (only `login` does — that is where hazards H3/H4 actually live), so the
/// capture's `Handshaking` state currently has zero frames in
/// it. Nothing here changes behaviour when recording is off: `CaptureSink::record_login` on a
/// disabled sink is the same one-`Option`-check no-op `record` already is.
async fn connect_and_login(
    opts: &ConnectOptions,
    to_game: &Sender<ToGame>,
    capture: &mut CaptureSink,
) -> Result<
    (
        Connection<ReadActivity<tokio::net::TcpStream>>,
        tokio::sync::mpsc::UnboundedReceiver<ActivityInstant>,
    ),
    DisconnectReason,
> {
    connect_and_login_with_timeouts(opts, to_game, capture, READ_TIMEOUT).await
}

/// The production connection sequence with an injectable timeout for deterministic tests.
///
/// Netty has two separate clocks here: its bootstrap's 30-second connect timeout, then the
/// pipeline's `ReadTimeoutHandler(30)`. The latter resets on every inbound socket read, so wrapping the
/// entire login future in one `timeout` would reject a slow but active login that vanilla keeps.
async fn connect_and_login_with_timeouts(
    opts: &ConnectOptions,
    to_game: &Sender<ToGame>,
    capture: &mut CaptureSink,
    timeout_budget: Duration,
) -> Result<
    (
        Connection<ReadActivity<tokio::net::TcpStream>>,
        tokio::sync::mpsc::UnboundedReceiver<ActivityInstant>,
    ),
    DisconnectReason,
> {
    let connected = match tokio::time::timeout(timeout_budget, open_connection(opts)).await {
        Ok(Ok(connected)) => connected,
        Ok(Err(error)) => return Err(open_failure_reason(&error, opts)),
        Err(_) => return Err(timeout_reason()),
    };
    // java: GuiConnecting switches from `connect.connecting` to `connect.authorizing` as soon
    // as `createNetworkManagerAndConnect` returns. Publish that exact socket-open boundary,
    // before the handshake or Login Start is written; login success is much too late.
    to_game
        .send(ToGame::SocketConnected)
        .await
        .map_err(|_| disconnect_reason_for(&ClientError::NotConnected))?;

    let (activity_tx, mut activity_rx) = tokio::sync::mpsc::unbounded_channel();
    // No protocol operation has touched `connected` yet, so rebuilding it around this transport
    // preserves the pristine Handshaking state while letting us observe each successful socket
    // read. This is earlier than framing, exactly where Netty's head-of-pipeline timeout sits.
    let mut conn = Connection::new(ReadActivity::new(connected.into_inner(), activity_tx));
    let login_sequence = async {
        handshake(&mut conn, &opts.host, opts.port, NextState::Login).await?;
        capture.set_state(mc_protocol::State::Login);

        let mut observe = |direction: LoginDirection, payload: &[u8]| {
            capture.record_login(direction, payload);
        };
        let outcome = match &opts.credentials {
            Credentials::Online(session) => {
                // `mc_protocol::connection::login_online_with_auth_policy_and_proxy`, with the
                // session client built here so a test can point the join at a fake server.
                // Built lazily, like that function: only a server that asks for encryption
                // needs it.
                let session: &mc_auth::Session = session;
                let join = |server_hash: String| async move {
                    opts.session_client()?
                        .join_server(session, &server_hash)
                        .await
                        .map_err(ProtoError::from)
                };
                login_with_auth_policy(
                    &mut conn,
                    &session.username,
                    join,
                    Some(&mut observe),
                    opts.auth_policy,
                )
                .await?
            }
            Credentials::Offline { username } => {
                login_offline(&mut conn, username, Some(&mut observe)).await?
            }
        };
        Ok::<_, ClientError>(outcome)
    };
    let outcome =
        match await_with_read_timeout(login_sequence, &mut activity_rx, timeout_budget).await {
            Ok(result) => result.map_err(|err| login_failure_reason(&err, opts))?,
            Err(ReadTimedOut) => return Err(timeout_reason()),
        };
    to_game
        .send(ToGame::Connected(outcome))
        .await
        .map_err(|_| disconnect_reason_for(&ClientError::NotConnected))?;
    Ok((conn, activity_rx))
}

#[derive(Debug)]
enum OpenError {
    UnknownHost,
    Connect {
        error: ProtoError,
        resolved: SocketAddr,
    },
}

/// Resolve exactly once, then connect to the first IPv4 address like `InetAddress.getByName`
/// with the vanilla launcher's `java.net.preferIPv4Stack=true` setting.
/// Resolution remains inside the outer connection timeout, preserving the bounded behavior of
/// this task while retaining enough context to normalize `GuiConnecting`'s error component.
async fn open_connection(
    opts: &ConnectOptions,
) -> Result<Connection<tokio::net::TcpStream>, OpenError> {
    let addresses = tokio::net::lookup_host((opts.host.as_str(), opts.port))
        .await
        .map_err(|_| OpenError::UnknownHost)?;
    let resolved = super::first_ipv4_address(addresses).ok_or(OpenError::UnknownHost)?;
    Connection::connect(resolved)
        .await
        .map_err(|error| OpenError::Connect { error, resolved })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReadTimedOut;

fn deadline_after_activity(
    activity_at: ActivityInstant,
    timeout_budget: Duration,
) -> ActivityInstant {
    activity_at + timeout_budget
}

/// Transport adapter that reports every successful inbound read before framing or decode.
/// `ReadTimeoutHandler` is the first inbound Netty handler and observes the same boundary.
#[derive(Debug)]
struct ReadActivity<S> {
    inner: S,
    activity: tokio::sync::mpsc::UnboundedSender<ActivityInstant>,
}

impl<S> ReadActivity<S> {
    fn new(inner: S, activity: tokio::sync::mpsc::UnboundedSender<ActivityInstant>) -> Self {
        Self { inner, activity }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for ReadActivity<S> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let filled_before = buf.filled().len();
        let result = Pin::new(&mut self.inner).poll_read(cx, buf);
        if matches!(result, Poll::Ready(Ok(()))) && buf.filled().len() > filled_before {
            // Unbounded and synchronous: observing activity must never backpressure the decoder
            // whose deadline it refreshes. A dropped receiver just means login has completed.
            let _ = self.activity.send(ActivityInstant::now());
        }
        result
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for ReadActivity<S> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// Await one protocol phase while reproducing Netty's inbound-activity deadline.
async fn await_with_read_timeout<F>(
    future: F,
    inbound_activity: &mut tokio::sync::mpsc::UnboundedReceiver<ActivityInstant>,
    timeout_budget: Duration,
) -> Result<F::Output, ReadTimedOut>
where
    F: std::future::Future,
{
    tokio::pin!(future);
    let deadline = tokio::time::sleep(timeout_budget);
    tokio::pin!(deadline);

    loop {
        tokio::select! {
            // If completion and the deadline become ready together, accept the completed
            // protocol operation. Netty likewise delivers a read before its timeout task when
            // the event loop observes both in the same turn.
            biased;
            output = &mut future => return Ok(output),
            Some(activity_at) = inbound_activity.recv() => {
                // The signal can sit behind a completed frame or another signal. Use the
                // transport timestamp, not this loop's processing time, or stale activity can
                // keep a dead phase alive indefinitely under a decode burst.
                let mut latest_activity = activity_at;
                while let Ok(next_activity) = inbound_activity.try_recv() {
                    latest_activity = latest_activity.max(next_activity);
                }
                deadline
                    .as_mut()
                    .reset(deadline_after_activity(latest_activity, timeout_budget));
            }
            () = &mut deadline => return Err(ReadTimedOut),
        }
    }
}

/// Map a failure on the way in to the reason the UI will show.
///
/// A `Login`-state kick already carries the server's own component JSON, so it must not be
/// flattened into a `ConnectionLost` string — S16 renders the two differently.
fn disconnect_reason_for(err: &ClientError) -> DisconnectReason {
    // Keep Rust's complete error chain in the log. It is invaluable when diagnosing a bad
    // server, but it is neither localizable nor part of vanilla's disconnect-screen contract.
    tracing::debug!(error = %err, "connection task failure");
    match err {
        ClientError::Protocol(ProtoError::Disconnected { reason }) => {
            server_component_reason(reason)
        }
        ClientError::Disconnected { reason } => server_component_reason(reason),
        // NetworkManager.channelInactive always uses this translation, including an EOF in
        // the middle of a frame. Do not substitute our ProtoError prose here.
        ClientError::Protocol(ProtoError::ConnectionClosed) => {
            translated_reason("disconnect.endOfStream")
        }
        // Netty's ReadTimeoutHandler takes this branch before exceptionCaught. Socket timeout
        // errors can reach the same boundary through an outbound write, so normalize those too.
        ClientError::Protocol(ProtoError::Io(io)) | ClientError::Io(io)
            if io.kind() == std::io::ErrorKind::TimedOut =>
        {
            timeout_reason()
        }
        ClientError::Protocol(ProtoError::Auth(auth)) => auth_failure_reason(auth),
        // A repeated non-negative Login S03: the same cast as a repeated play S46.
        ClientError::Protocol(ProtoError::CompressionReconfigured) => {
            compression_reconfiguration_reason()
        }
        ClientError::Protocol(protocol) => {
            generic_exception_reason(protocol_exception_class(protocol))
        }
        ClientError::Io(_) => generic_exception_reason("java.io.IOException"),
        // World/UI errors cannot be emitted by a vanilla decoder, but a live task still needs a
        // valid, translated terminal component if one escapes from a future handler.
        _ => generic_exception_reason("java.lang.RuntimeException"),
    }
}

/// The Java-side exception family that reaches NetworkManager.exceptionCaught.
///
/// Keeping this deliberately coarse is important: ProtoError::Display contains Rust crate
/// names, field labels and often untrusted bytes. Netty wraps packet-decoder failures in a
/// DecoderException; a transport error reaches the pipeline as an IOException.
fn protocol_exception_class(error: &ProtoError) -> &'static str {
    match error {
        ProtoError::Io(_) => "java.io.IOException",
        _ => "io.netty.handler.codec.DecoderException",
    }
}

/// Validate a server-authored disconnect component at the decode boundary.
///
/// Vanilla's `PacketBuffer.readChatComponent` calls Gson before dispatch. A malformed component
/// therefore follows `NetworkManager.exceptionCaught`, rather than reaching the disconnect screen
/// as literal JSON. `ChatComponent::from_json` already mirrors Gson's lenient spellings, so valid
/// non-strict components remain byte-for-byte server-authored reasons.
fn server_component_reason(reason_json: &str) -> DisconnectReason {
    match mc_ui::font::component::ChatComponent::from_json_nullable(reason_json) {
        Ok(Some(_)) => DisconnectReason::Kicked {
            reason_json: reason_json.to_owned(),
        },
        // java: Gson resolves a root `null` (or an empty document) to a null component. Both
        // S40 and the Login S00 hand it to `NetworkManager.closeChannel`, which stores the null
        // `terminationReason`; `checkDisconnected` then substitutes
        // `new ChatComponentText("Disconnected")` (`NetworkManager.java:490-505`).
        Ok(None) => component_reason(serde_json::json!({ "text": "Disconnected" })),
        Err(error) => malformed_component_reason(&error),
    }
}

/// Validate one field that vanilla reads with `PacketBuffer.readChatComponent`.
///
/// `IChatComponent.Serializer.jsonToComponent` is `GSON.fromJson`, which *returns* null (it does
/// not throw) for a root JSON `null`, an empty document and an empty component array. The
/// decoder therefore never fails on those spellings; whether a null component is harmless,
/// ignored or fatal is decided later by each handler. Only genuinely malformed JSON is a decoder
/// exception here.
fn validate_component(json: &str) -> Result<(), mc_ui::UiError> {
    mc_ui::font::component::ChatComponent::from_json_nullable(json).map(|_| ())
}

/// Whether Gson's component adapter turns `json` into a null `IChatComponent`.
fn is_gson_null_component(json: &str) -> bool {
    matches!(
        mc_ui::font::component::ChatComponent::from_json_nullable(json),
        Ok(None)
    )
}

/// The component `NetworkManager.exceptionCaught` builds for an unchecked exception thrown by a
/// Netty-thread handler. `Throwable.toString()` of a message-less `NullPointerException` is just
/// its class name.
fn null_pointer_reason() -> DisconnectReason {
    generic_exception_reason("java.lang.NullPointerException")
}

fn malformed_component_reason(error: &mc_ui::UiError) -> DisconnectReason {
    // Preserve the detailed parser failure in the log, but only the Java-visible exception
    // family belongs in the translation argument. PacketBuffer.readChatComponent throws a Gson
    // JsonSyntaxException before the packet is dispatched.
    tracing::debug!(error = %error, "malformed server chat component");
    generic_exception_reason("com.google.gson.JsonSyntaxException")
}

/// Validate every Play field read through vanilla's `PacketBuffer.readChatComponent` before the
/// packet can enter the game-thread queue. Keeping this inventory beside the dispatch boundary
/// prevents individual UI consumers from accidentally swallowing decoder errors.
/// Queue a tab-list update's `textures` signatures on the background verifier, so the game thread
/// finds the answer already remembered when a row first needs it.
///
/// java: `SkinManager.loadProfileTextures` does this on `THREAD_POOL`, never on the game thread.
/// This is only a head start: nothing waits for it. It used to run at most 2 packets at once on
/// the blocking pool and silently drop the rest, which on a join (one S38 per player) left most
/// rows to an RSA check inside a frame (HITCH-1). Every entry is now queued, still bounded:
/// see `sync::players::precheck_texture_signature`.
fn precheck_texture_signatures(packet: &S2CPlay) {
    let S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(entries)) = packet else {
        return;
    };
    for entry in entries {
        let Some(property) = entry
            .properties
            .iter()
            .find(|property| property.name == "textures")
        else {
            continue;
        };
        if let Some(signature) = &property.signature {
            crate::sync::players::precheck_texture_signature(
                property.value.clone(),
                signature.clone(),
            );
        }
    }
}

fn validate_play_components(packet: &S2CPlay) -> Result<(), mc_ui::UiError> {
    match packet {
        S2CPlay::Chat(chat) => validate_component(&chat.component),
        S2CPlay::OpenWindow(open) => validate_component(&open.window_title),
        S2CPlay::UpdateSign(update) => {
            for line in &update.lines {
                // Gson may return null for a blank sign component. Vanilla stores it and
                // TileEntitySignRenderer skips that row; it is not malformed JSON.
                validate_component(line)?;
            }
            Ok(())
        }
        S2CPlay::PlayerListItem(update) => {
            match update {
                s2c::PlayerListItem::AddPlayer(entries) => {
                    for component in entries
                        .iter()
                        .filter_map(|entry| entry.display_name.as_deref())
                    {
                        validate_component(component)?;
                    }
                }
                s2c::PlayerListItem::UpdateDisplayName(entries) => {
                    for component in entries
                        .iter()
                        .filter_map(|entry| entry.display_name.as_deref())
                    {
                        validate_component(component)?;
                    }
                }
                s2c::PlayerListItem::UpdateGameMode(_)
                | s2c::PlayerListItem::UpdateLatency(_)
                | s2c::PlayerListItem::RemovePlayer(_) => {}
            }
            Ok(())
        }
        S2CPlay::Disconnect(disconnect) => validate_component(&disconnect.reason),
        S2CPlay::Title(s2c::Title::SetTitle { text } | s2c::Title::SetSubtitle { text }) => {
            validate_component(text)
        }
        S2CPlay::PlayerListHeaderFooter(header_footer) => {
            validate_component(&header_footer.header)?;
            validate_component(&header_footer.footer)
        }
        _ => Ok(()),
    }
}

/// `GuiConnecting` catches failures that happen before a `NetworkManager` exists itself, so its
/// `disconnect.genericReason` argument does not carry NetworkManager's `Internal Exception:`
/// prefix. Once the socket is open, [`disconnect_reason_for`] models the pipeline exception path.
fn open_failure_reason(error: &OpenError, opts: &ConnectOptions) -> DisconnectReason {
    match error {
        OpenError::UnknownHost => generic_reason("Unknown host".to_owned()),
        OpenError::Connect { error, resolved } => {
            let detail = match error {
                // Java's `Exception.toString()` does not contain mc-rs's crate wrapper. The OS
                // message is the closest portable analogue; normalize its address separately.
                ProtoError::Io(io) => io.to_string(),
                other => other.to_string(),
            };
            generic_reason(strip_connect_address(&detail, opts, *resolved))
        }
    }
}

fn strip_connect_address(detail: &str, opts: &ConnectOptions, resolved: SocketAddr) -> String {
    let ip = resolved.ip();
    let port = resolved.port();
    let mut candidates = [
        // Rust socket errors commonly use one of these two.
        resolved.to_string(),
        format!("{}:{}", opts.host, opts.port),
        // `InetAddress.toString()` is `host/address` (or `/address` without a host), to which
        // GuiConnecting appends `:port` before removing it.
        format!("{}/{ip}:{port}", opts.host),
        format!("/{ip}:{port}"),
        // Some OS adapters omit IPv6 brackets in prose even though `SocketAddr::to_string` does
        // not. Keeping this candidate makes normalization platform-independent.
        format!("{ip}:{port}"),
    ];
    // Remove the compound Java spelling before its shorter IP:port suffix; otherwise the first
    // replacement would leave the hostname behind.
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.len()));
    candidates
        .iter()
        .filter(|candidate| !candidate.is_empty())
        .fold(detail.to_owned(), |text, candidate| {
            text.replace(candidate, "")
        })
}

fn auth_failure_reason(auth: &mc_auth::AuthError) -> DisconnectReason {
    auth_failure_reason_for(auth, None)
}

/// The reason for a failed login, with what only the login knows: the server, and what the
/// launcher's token says about itself. A failed session-server join is also logged at WARN,
/// since the screen is all the player sees and it used to reach no log at INFO or above.
fn login_failure_reason(err: &ClientError, opts: &ConnectOptions) -> DisconnectReason {
    let (ClientError::Protocol(ProtoError::Auth(auth)), Credentials::Online(session)) =
        (err, &opts.credentials)
    else {
        return disconnect_reason_for(err);
    };
    let token = TokenCheck::of(session, mc_auth::unix_now());
    tracing::warn!(
        host = %opts.host,
        port = opts.port,
        error = %error_chain(auth),
        token_validity = %token,
        "could not log in: the session server join failed"
    );
    auth_failure_reason_for(auth, token.problem(&session.username))
}

/// What a failed session-server join shows on the disconnect screen.
///
/// java: `NetHandlerLoginClient.handleEncryptionRequest` maps authlib's exceptions:
/// `AuthenticationUnavailableException` to `loginFailedInfo.serversUnavailable`,
/// `InvalidCredentialsException` to `loginFailedInfo.invalidSession`, and any other
/// `AuthenticationException` to its own message, the session server's `errorMessage`
/// ("Failed to login: <message>"). `mc_auth::AuthClient::join_server` reads the answer the
/// way authlib does, so each variant here is one of those exceptions.
///
/// This used to turn every answer except 401/403 into one
/// opaque line, "Authentication server rejected the request", and log the real one at DEBUG.
/// Where this differs from vanilla it is only to be more truthful on a failure:
/// * when the token's own claims show it has expired, or was issued for another account,
///   that is the reason given (vanilla cannot know; its advice, "restart your game", does not
///   refresh a launcher's token);
/// * a refusal without a message names the error type (vanilla prints "null");
/// * an unreadable answer adds its HTTP status, and a 429 says to wait (vanilla calls both
///   "down for maintenance").
fn auth_failure_reason_for(
    auth: &mc_auth::AuthError,
    token: Option<TokenProblem>,
) -> DisconnectReason {
    use mc_auth::AuthError;
    // The session server answered and said no: if the token's own claims explain why, say that.
    // (A request that never completed said nothing about the token.)
    let answered = !matches!(
        auth,
        AuthError::Network { .. } | AuthError::HttpClientInit(_)
    );
    if let (true, Some(problem)) = (answered, token) {
        return login_failed_text(&problem.screen_text());
    }
    match auth {
        AuthError::Network { .. } | AuthError::HttpClientInit(_) => {
            login_failed_nested("disconnect.loginFailedInfo.serversUnavailable")
        }
        AuthError::TokenRejected => {
            login_failed_nested("disconnect.loginFailedInfo.invalidSession")
        }
        AuthError::JoinRefused { error, message, .. } => match message {
            Some(message) => login_failed_text(message),
            None => login_failed_text(refusal_without_message(error)),
        },
        AuthError::JoinUnreadable { status: 429, .. } => login_failed_text(
            "Too many logins in a short time (HTTP 429). Wait a minute, then try again.",
        ),
        AuthError::JoinUnreadable { status, .. } => component_reason(serde_json::json!({
            "translate": "disconnect.loginFailedInfo",
            "with": [{
                "translate": "disconnect.loginFailedInfo.serversUnavailable",
                "extra": [{ "text": format!(" (HTTP {status})") }],
            }],
        })),
        // Not produced by the join; the error enum is non-exhaustive.
        other => login_failed_text(&format!(
            "Authentication failed ({})",
            error_kind_label(other)
        )),
    }
}

/// The text for a refusal that came without an `errorMessage`: the two error types Mojang is
/// known to send for an account-level block, in words; any other, by its name.
fn refusal_without_message(error: &str) -> &str {
    match error {
        "InsufficientPrivilegesException" => {
            "Multiplayer is turned off for this account (Microsoft account or family settings)."
        }
        "UserBannedException" => "This account is banned from online multiplayer.",
        other => other,
    }
}

/// A short, token-free label for an unexpected authentication error, for the screen.
fn error_kind_label(error: &mc_auth::AuthError) -> &'static str {
    match error {
        mc_auth::AuthError::Http { .. } => "unexpected answer",
        mc_auth::AuthError::MalformedResponse { .. } => "unreadable answer",
        _ => "unexpected error",
    }
}

/// "Failed to login: <text>", with `text` as a literal argument, exactly as vanilla passes an
/// `AuthenticationException`'s message.
fn login_failed_text(text: &str) -> DisconnectReason {
    component_reason(serde_json::json!({
        "translate": "disconnect.loginFailedInfo",
        "with": [text],
    }))
}

/// An error and its sources on one line: reqwest's own message names only the URL, and the
/// cause (DNS, TLS, timeout) is in its sources. Never contains the access token: `AuthError`
/// does not carry it and reqwest's errors describe the request, not its body.
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// Why the launcher's token cannot work, when its own claims say so.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenProblem {
    /// Its `exp` claim is in the past.
    Expired,
    /// It names a different Minecraft profile than the one being logged in as.
    OtherAccount {
        /// The name the client logs in with.
        username: String,
    },
}

impl TokenProblem {
    fn screen_text(&self) -> String {
        match self {
            TokenProblem::Expired => "Your Minecraft login has expired (logins last about a \
                                      day). Close the game and start it again from your \
                                      launcher to refresh it."
                .to_owned(),
            TokenProblem::OtherAccount { username } => format!(
                "The login your launcher passed is for a different Minecraft account than \
                 {username}. Sign in to {username} again in your launcher, then restart the \
                 game."
            ),
        }
    }
}

/// How long a token that close to expiry is still worth a warning at start-up.
const TOKEN_EXPIRY_WARNING_SECS: i64 = 30 * 60;

/// After a refused join, a token this close to its expiry is treated as expired: the PC's clock
/// may run a little behind Mojang's.
const TOKEN_CLOCK_SKEW_SECS: i64 = 2 * 60;

/// What the launcher's token says about itself, for the log and the failure reason. Holds no
/// part of the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenCheck {
    /// Seconds until the token expires; negative once it has. `None` without an `exp` claim.
    expires_in: Option<i64>,
    /// The token names a different Minecraft profile than the session's.
    other_account: bool,
}

impl TokenCheck {
    fn of(session: &mc_auth::Session, now: u64) -> Self {
        let facts = session.token_facts();
        TokenCheck {
            expires_in: facts.expires_at.map(|expires_at| {
                i64::try_from(expires_at)
                    .unwrap_or(i64::MAX)
                    .saturating_sub(i64::try_from(now).unwrap_or(i64::MAX))
            }),
            other_account: facts.issued_for_another_profile(session.uuid),
        }
    }

    /// The problem to report after a failed join. Expiry first: it is certain from the claim,
    /// and refreshing fixes it. A token within [`TOKEN_CLOCK_SKEW_SECS`] of its expiry counts
    /// as expired here, since this PC's clock may be a little behind Mojang's.
    fn problem(&self, username: &str) -> Option<TokenProblem> {
        if self
            .expires_in
            .is_some_and(|left| left <= TOKEN_CLOCK_SKEW_SECS)
        {
            Some(TokenProblem::Expired)
        } else if self.other_account {
            Some(TokenProblem::OtherAccount {
                username: username.to_owned(),
            })
        } else {
            None
        }
    }
}

impl std::fmt::Display for TokenCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.expires_in {
            None => f.write_str("no expiry in the token")?,
            Some(left) if left > 0 => write!(f, "valid for {}", human_duration(left))?,
            Some(left) => write!(f, "expired {} ago", human_duration(left.saturating_neg()))?,
        }
        if self.other_account {
            f.write_str(", issued for a different Minecraft account")?;
        }
        Ok(())
    }
}

/// `3h 05m`, `12m`, `40s`.
fn human_duration(secs: i64) -> String {
    let secs = secs.max(0);
    if secs >= 3600 {
        format!("{}h {:02}m", secs / 3600, secs % 3600 / 60)
    } else if secs >= 60 {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

/// Log how long the launcher's access token stays valid.
///
/// A launcher hands over the token it has. It lasts about a day from when the launcher fetched
/// it, nothing refreshes it mid-session, and once it expires every online-mode server refuses
/// the login. Saying so at start-up makes that failure obvious in the log. Never logs the
/// token, only what it says about its own expiry.
pub fn log_login_token(session: &mc_auth::Session) {
    let check = TokenCheck::of(session, mc_auth::unix_now());
    match check.expires_in {
        None if !check.other_account => {
            tracing::debug!("the login token carries no expiry");
        }
        Some(left) if left > TOKEN_EXPIRY_WARNING_SECS && !check.other_account => {
            tracing::info!(token_validity = %check, "login token");
        }
        _ if check.other_account => tracing::warn!(
            token_validity = %check,
            "login token: it is for a different Minecraft account than the one logging in, so \
             online-mode servers will refuse the login"
        ),
        _ => tracing::warn!(
            token_validity = %check,
            "login token: online-mode servers will refuse the login once it has expired, until \
             the launcher refreshes it (restart the game from the launcher)"
        ),
    }
}

fn login_failed_nested(inner_key: &str) -> DisconnectReason {
    component_reason(serde_json::json!({
        "translate": "disconnect.loginFailedInfo",
        "with": [{ "translate": inner_key }],
    }))
}

fn generic_exception_reason(exception_class: &str) -> DisconnectReason {
    generic_reason(format!("Internal Exception: {exception_class}"))
}

fn generic_reason(detail: String) -> DisconnectReason {
    component_reason(serde_json::json!({
        "translate": "disconnect.genericReason",
        "with": [detail],
    }))
}

fn translated_reason(key: &str) -> DisconnectReason {
    component_reason(serde_json::json!({ "translate": key }))
}

fn component_reason(component: serde_json::Value) -> DisconnectReason {
    DisconnectReason::Kicked {
        reason_json: component.to_string(),
    }
}

/// The exact component produced by `NetworkManager.exceptionCaught(TimeoutException)`.
fn timeout_reason() -> DisconnectReason {
    component_reason(serde_json::json!({ "translate": "disconnect.timeout" }))
}

/// Vanilla 1.8.9's positive compression reconfiguration path updates the decoder and then casts
/// that decoder to the unrelated encoder class. The resulting exception closes the channel.
///
/// The shipped 1.8.9 client is obfuscated: `NettyCompressionDecoder` is class `ei` and
/// `NettyCompressionEncoder` is `ej` (the deobfuscation mapping), and Java 8's
/// `ClassCastException` message is "X cannot be cast to Y". The readable names only appear in a
/// deobfuscated development run, never on a player's disconnect screen.
fn compression_reconfiguration_reason() -> DisconnectReason {
    generic_exception_reason("java.lang.ClassCastException: ei cannot be cast to ej")
}

/// The full-duplex `Play` pump.
///
/// Split out from [`run`] and generic over the transport so tests can drive it over
/// `tokio::io::duplex` as well as a real socket.
///
/// # Cancel-safety
///
/// `read_payload` is documented cancel-safe by S07 and is therefore legal as a `select!` arm.
/// `send_payload` is documented **not** cancel-safe, so it appears only inside a branch *body*:
/// `tokio::select!` drops the other futures once a branch is chosen and then runs that branch's
/// body to completion, so the write is never abandoned mid-frame. Abandoning one would leave a
/// partial frame on the wire and desynchronise the CFB8 stream permanently (hazard H4).
// These are deliberately separate: each channel belongs to a different side of the
// connection-task boundary, and bundling them would obscure the ordering guarantees below.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn play_loop<R, W>(
    reader: ConnectionReader<R>,
    writer: ConnectionWriter<W>,
    to_game: &Sender<ToGame>,
    immediate_to_game: &tokio::sync::mpsc::UnboundedSender<ToGame>,
    from_game: UnboundedReceiver<ToNet>,
    inbound_activity: tokio::sync::mpsc::UnboundedReceiver<ActivityInstant>,
    timeout_budget: Duration,
    capture: &mut CaptureSink,
) -> DisconnectReason
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut pending_to_game = VecDeque::new();
    let reason = pump_play(
        reader,
        writer,
        to_game,
        immediate_to_game,
        from_game,
        inbound_activity,
        timeout_budget,
        capture,
        &mut pending_to_game,
    )
    .await;
    // java: a packet Netty already handed to
    // `Minecraft.addScheduledTask` still runs in the next scheduled-task drain, which precedes
    // the tick whose `checkDisconnected` reports the close. The spill FIFO is the overflow of
    // that same queue, so after a server or transport close it reaches the game ahead of the
    // terminal `ToGame::Disconnected` that `run` sends next, with that send's own wait for
    // capacity. A client-initiated close (or a game loop that is gone, which ends the pump the
    // same way) has nobody draining for it: awaiting capacity there would never finish.
    if reason != DisconnectReason::ClosedByClient {
        for message in pending_to_game {
            if to_game.send(message).await.is_err() {
                break;
            }
        }
    }
    reason
}

/// [`play_loop`]'s pump. `pending_to_game` is the scheduled-packet spill FIFO, owned by the
/// caller so that what is still in it when the pump ends can be delivered in order.
#[allow(clippy::too_many_arguments)]
async fn pump_play<R, W>(
    mut reader: ConnectionReader<R>,
    mut writer: ConnectionWriter<W>,
    to_game: &Sender<ToGame>,
    immediate_to_game: &tokio::sync::mpsc::UnboundedSender<ToGame>,
    mut from_game: UnboundedReceiver<ToNet>,
    mut inbound_activity: tokio::sync::mpsc::UnboundedReceiver<ActivityInstant>,
    timeout_budget: Duration,
    capture: &mut CaptureSink,
    pending_to_game: &mut VecDeque<ToGame>,
) -> DisconnectReason
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    // The deadline advances only when the transport adapter observes inbound bytes, before
    // framing or decode. A busy outbound stream cannot reset it, while a slowly arriving frame
    // stays alive exactly like Netty's head-of-pipeline ReadTimeoutHandler.
    let mut read_deadline = deadline_after_activity(ActivityInstant::now(), timeout_budget);
    // The sky-light payload layout an S21 is unpacked with ahead of the game thread: the last
    // JoinGame/Respawn dimension in stream order (an overworld until one arrives).
    let mut chunk_sky = true;

    loop {
        tokio::select! {
            biased;

            // A raw read and its newly completed frame can become ready together. Observe the
            // byte first so activity at the deadline wins, matching Netty's event-loop order.
            Some(activity_at) = inbound_activity.recv() => {
                // Activity may be queued while frame decode or a game-thread handoff runs. The
                // deadline belongs to the socket read that produced the signal, not to the later
                // turn that happens to observe it.
                let mut latest_activity = activity_at;
                while let Ok(next_activity) = inbound_activity.try_recv() {
                    latest_activity = latest_activity.max(next_activity);
                }
                read_deadline = deadline_after_activity(latest_activity, timeout_budget);
            }

            () = tokio::time::sleep_until(read_deadline) => {
                // java: NetworkManager installs `new ReadTimeoutHandler(30)`, whose exception
                // path kicks with `disconnect.timeout`.
                return timeout_reason();
            }

            // After timeout bookkeeping, prefer draining the outbound queue: movement packets
            // are cadence-sensitive (hazard H11) and must not be starved by a chunk burst.
            command = from_game.recv() => match command {
                Some(ToNet::Send(packet)) => {
                    if let Err(err) = send(&mut writer, &packet, capture).await {
                        return disconnect_reason_for(&err);
                    }
                }
                Some(ToNet::Disconnect) | None => return DisconnectReason::ClosedByClient,
            },

            // Vanilla's scheduledTasks queue is unbounded, so a stalled game thread cannot stop
            // the Netty thread from reaching KeepAlive. Preserve the bounded handoff for frame
            // pacing, but spill in order and flush the oldest packet as soon as capacity returns.
            permit = to_game.reserve(), if !pending_to_game.is_empty() => match permit {
                Ok(permit) => permit.send(
                    pending_to_game
                        .pop_front()
                        .expect("reserve arm is guarded by a non-empty FIFO"),
                ),
                Err(_) => return DisconnectReason::ClosedByClient,
            },

            read = reader.read_payload() => {
                let mut payload = match read {
                    Ok(payload) => payload,
                    Err(err) => return disconnect_reason_for(&ClientError::from(err)),
                };
                // Before `decode`, which advances the read cursor: `PacketBuffer::as_slice` is
                // the *unread* tail, so recording afterwards would lose the packet id varint.
                // These bytes are already decrypted and decompressed — see `net::capture`.
                capture.record(mc_parity::Direction::Clientbound, payload.as_slice());
                // The same bytes for the replay recorder, copied only while
                // its tap is on (one relaxed load otherwise); they travel with the decoded
                // packet so the game thread records them when it applies it.
                let raw = capture.tapped(payload.as_slice());
                let packet = match decode(&mut payload) {
                    Ok(packet) => packet,
                    Err(err) => return disconnect_reason_for(&err),
                };
                if let Err(error) = validate_play_components(&packet) {
                    return malformed_component_reason(&error);
                }
                precheck_texture_signatures(&packet);

                match packet {
                    // --- the three direct handlers vanilla keeps on the Netty thread ---
                    S2CPlay::KeepAlive(keep_alive) => {
                        // java: NetHandlerPlayClient.handleKeepAlive — echo the id straight back.
                        let echo = C2SPlay::KeepAlive(c2s::KeepAlive { key: keep_alive.id });
                        if let Err(err) = send(&mut writer, &echo, capture).await {
                            return disconnect_reason_for(&err);
                        }
                    }
                    S2CPlay::SetCompression(set) => {
                        // java: handleSetCompressionLevel. Must take effect before the next
                        // frame is deframed, so it can never be deferred to the game thread.
                        if set.threshold >= 0 && reader.compression_threshold().is_some() {
                            // The decoder update occurs before vanilla's bad decoder-to-encoder
                            // cast throws. The channel closes immediately, so the writer keeps
                            // its old threshold until both halves are dropped.
                            reader.set_compression_threshold(set.threshold);
                            return compression_reconfiguration_reason();
                        }
                        reader.set_compression_threshold(set.threshold);
                        writer.set_compression_threshold(set.threshold);
                    }
                    S2CPlay::Disconnect(s2c::Disconnect { reason }) => {
                        // A Gson-null reason becomes vanilla's "Disconnected" text.
                        return server_component_reason(&reason);
                    }

                    // --- Netty handlers whose Rust state effects remain game-thread-owned ---
                    //
                    // Java executes these two bodies directly in `channelRead0`, before any
                    // PacketThreadUtil queue. `ClientState` intentionally cannot cross this
                    // thread boundary, so preserve that ordering with a dedicated unbounded
                    // lane rather than putting a mutex around world/UI state. Once this handler
                    // has run, `NetHandle` drains this lane before its bounded scheduled queue on
                    // every game-loop poll. Neither Java nor Rust can order a game-loop poll
                    // against a later socket packet whose Netty handler has not run yet.
                    direct @ (S2CPlay::PlayerListHeaderFooter(_) | S2CPlay::ResourcePackSend(_)) => {
                        // java: handlePlayerListHeaderFooter (`:1600-1604`) runs right here on
                        // the Netty thread and calls `getFormattedText()` on both components
                        // with no null check. A Gson-null header or footer therefore throws a
                        // NullPointerException into `exceptionCaught`, which closes the channel.
                        if let S2CPlay::PlayerListHeaderFooter(header_footer) = &direct {
                            if is_gson_null_component(&header_footer.header)
                                || is_gson_null_component(&header_footer.footer)
                            {
                                return null_pointer_reason();
                            }
                        }
                        if immediate_to_game
                            .send(ToGame::Packet(direct).with_raw(raw))
                            .is_err()
                        {
                            return DisconnectReason::ClosedByClient;
                        }
                    }

                    // --- everything else is the game loop's ---
                    other => {
                        // Full-column chunk packets start unpacking on a background thread now
                        // and still enter the queue here, in this exact position (#15 / NET-1).
                        // `chunk_sky` follows JoinGame/Respawn in stream order; the game thread
                        // re-checks it against the world before installing an S21.
                        match &other {
                            S2CPlay::JoinGame(join) => {
                                chunk_sky = crate::sync::state::dimension_has_sky_light(
                                    i32::from(join.dimension),
                                );
                            }
                            S2CPlay::Respawn(respawn) => {
                                chunk_sky =
                                    crate::sync::state::dimension_has_sky_light(respawn.dimension);
                            }
                            _ => {}
                        }
                        let message =
                            match crate::sync::prebuild::PrebuildingChunkPacket::spawn(
                                other, chunk_sky,
                            ) {
                                Ok(job) => ToGame::ChunkPacket(job),
                                Err(other) => ToGame::Packet(other),
                            }
                            .with_raw(raw);
                        if enqueue_scheduled(to_game, pending_to_game, message).is_err() {
                            return DisconnectReason::ClosedByClient;
                        }
                    }
                }
            }

        }
    }
}

/// The game loop's end of the scheduled-packet handoff has gone away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GameLoopGone;

/// Hand one scheduled packet to the game loop, keeping stream order.
///
/// java: `Minecraft.scheduledTasks` is one FIFO. Once a packet has
/// spilled into `pending` (the bounded channel was full), every later packet queues behind it
/// even if the game loop has made room since: a `try_send` that found room would put the newest
/// packet ahead of every spilled one. The `reserve` arm of [`play_loop`] moves `pending` into
/// the channel, oldest first.
fn enqueue_scheduled(
    to_game: &Sender<ToGame>,
    pending: &mut VecDeque<ToGame>,
    message: ToGame,
) -> Result<(), GameLoopGone> {
    if !pending.is_empty() {
        pending.push_back(message);
        return Ok(());
    }
    match to_game.try_send(message) {
        Ok(()) => Ok(()),
        Err(tokio::sync::mpsc::error::TrySendError::Full(message)) => {
            pending.push_back(message);
            Ok(())
        }
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => Err(GameLoopGone),
    }
}

/// Serialise one packet — id VarInt then body — and write it.
///
/// The payload is recorded here rather than at the two call sites so that every serverbound
/// frame is captured, including the keep-alive echo this task answers on its own.
async fn send<W: AsyncWrite + Unpin>(
    writer: &mut ConnectionWriter<W>,
    packet: &C2SPlay,
    capture: &mut CaptureSink,
) -> Result<(), ClientError> {
    let mut payload = PacketBuffer::new();
    payload.write_varint(packet.id());
    packet.write(&mut payload);
    payload.finish_write()?;
    capture.record(mc_parity::Direction::Serverbound, payload.as_slice());
    writer.send_payload(payload.as_slice()).await?;
    Ok(())
}

/// Split the id off a payload and decode it.
///
/// An unknown id is fatal, matching vanilla: `EnumConnectionState.getPacket` returns `null` and
/// `NettyPacketDecoder` throws "Bad packet id", which closes the channel. Silently skipping the
/// packet would be worse than closing — the *body* has already been framed off, so we would be
/// guessing about a server we demonstrably do not understand.
fn decode(payload: &mut PacketBuffer) -> Result<S2CPlay, ClientError> {
    let id = payload.read_varint()?;
    let packet = S2CPlay::read(id, payload)?;
    if !payload.is_empty() {
        return Err(ProtoError::TrailingBytes {
            packet_id: id,
            remaining: payload.remaining(),
        }
        .into());
    }
    Ok(packet)
}

/// How long the server may go silent before we consider the link dead.
/// java: `NetworkManager` installs a `ReadTimeoutHandler(30)`.
pub const READ_TIMEOUT: Duration = Duration::from_secs(30);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::channel::{NetHandle, INBOUND_CAPACITY};
    use mc_protocol::packet::Packet;
    use mc_protocol::play::s2c;

    #[test]
    fn connect_options_enable_auth_tolerance_only_for_lan_entries() {
        let strict = ConnectOptions::offline("saved.example", 25_565, "Notch");
        assert_eq!(strict.auth_policy, LoginAuthPolicy::Strict);

        let lan = strict.clone().with_lan_auth_tolerance(true);
        assert_eq!(lan.auth_policy, LoginAuthPolicy::AllowLanAuthFailure);
        assert_eq!(
            lan.with_lan_auth_tolerance(false).auth_policy,
            LoginAuthPolicy::Strict,
            "a later saved/direct selection must not inherit a previous LAN exception"
        );
    }

    /// Wire two `Connection`s back to back over an in-memory duplex and hand the client half's
    /// split ends to [`play_loop`], returning the handle plus the far side.
    fn rig() -> (
        NetHandle,
        Connection<tokio::io::DuplexStream>,
        tokio::task::JoinHandle<DisconnectReason>,
    ) {
        rig_with_compression(None)
    }

    fn rig_with_compression(
        threshold: Option<i32>,
    ) -> (
        NetHandle,
        Connection<tokio::io::DuplexStream>,
        tokio::task::JoinHandle<DisconnectReason>,
    ) {
        let (a, b) = tokio::io::duplex(1 << 20);
        let (handle, to_game, immediate_to_game, from_game) = NetHandle::new_live_pair();
        let (activity_tx, activity_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut client = Connection::new(ReadActivity::new(a, activity_tx));
        let mut server = Connection::new(b);
        if let Some(threshold) = threshold {
            client.set_compression_threshold(threshold);
            server.set_compression_threshold(threshold);
        }
        let (reader, writer) = client.split();
        let task = tokio::spawn(async move {
            play_loop(
                reader,
                writer,
                &to_game,
                &immediate_to_game,
                from_game,
                activity_rx,
                READ_TIMEOUT,
                &mut CaptureSink::disabled(),
            )
            .await
        });
        (handle, server, task)
    }

    async fn expect_packet(handle: &mut NetHandle) -> S2CPlay {
        for _ in 0..1000 {
            if let Some(ToGame::Packet(packet)) = handle.try_recv() {
                return packet;
            }
            tokio::task::yield_now().await;
        }
        panic!("no packet arrived");
    }

    #[tokio::test]
    async fn localhost_ipv4_server_accepts_compressed_login() {
        use mc_protocol::handshake::Handshake;
        use mc_protocol::login::{LoginStart, LoginSuccess, SetCompression};

        // Windows resolves localhost to ::1 before 127.0.0.1. Exercise the real resolver,
        // connection task, and login exchange against a listener accepting IPv4 only.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("IPv4 loopback listener");
        let port = listener.local_addr().expect("listener address").port();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept localhost client");
            let mut conn = Connection::new(socket);
            let (id, mut body) = conn.read_packet().await.expect("handshake");
            assert_eq!(id, Handshake::ID);
            let handshake = Handshake::read(&mut body).expect("decode handshake");
            assert_eq!(handshake.protocol_version, 47);
            assert_eq!(handshake.server_address, "localhost");
            assert_eq!(handshake.server_port, port);
            assert_eq!(handshake.next_state, NextState::Login);
            conn.set_state(mc_protocol::State::Login).unwrap();
            let (id, mut body) = conn.read_packet().await.expect("login start");
            assert_eq!(id, LoginStart::ID);
            assert_eq!(LoginStart::read(&mut body).unwrap().username, "LocalTest");
            conn.send(&SetCompression { threshold: 256 })
                .await
                .expect("compression negotiation");
            conn.set_compression_threshold(256);
            conn.send(&LoginSuccess {
                uuid: uuid::Uuid::from_u128(1),
                username: "LocalTest".to_owned(),
            })
            .await
            .expect("compressed login success");
        });
        let (mut handle, to_game, _from_game) = NetHandle::new_pair();
        let opts = ConnectOptions::offline("localhost", port, "LocalTest");
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            connect_and_login(&opts, &to_game, &mut CaptureSink::disabled()),
        )
        .await;
        if !matches!(result, Ok(Ok(_))) {
            server.abort();
        }
        let (conn, _) = result
            .expect("localhost login remains bounded")
            .expect("localhost selects the available IPv4 address");
        assert_eq!(conn.state(), mc_protocol::State::Play);
        assert_eq!(conn.compression_threshold(), Some(256));
        assert!(matches!(handle.try_recv(), Some(ToGame::SocketConnected)));
        assert!(matches!(handle.try_recv(), Some(ToGame::Connected(outcome))
            if outcome.username == "LocalTest" && outcome.uuid == uuid::Uuid::from_u128(1)));
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn socket_open_is_reported_while_login_is_still_blocked() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let port = listener.local_addr().expect("listener address").port();
        let (mut handle, to_game, _from_game) = NetHandle::new_pair();
        let opts = ConnectOptions::offline("127.0.0.1", port, "PhaseTest");

        let connect = tokio::spawn(async move {
            connect_and_login(&opts, &to_game, &mut CaptureSink::disabled()).await
        });
        let (socket, _) = listener.accept().await.expect("accept TCP connection");

        // The peer deliberately never answers Login Start. The phase signal must therefore be
        // observable independently of login success; moving the send below `login_offline`
        // makes this timeout.
        let message = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Some(message) = handle.try_recv() {
                    break message;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("socket-open phase before login completes");
        assert!(matches!(message, ToGame::SocketConnected));

        drop(socket);
        connect.abort();
    }

    #[tokio::test]
    async fn silent_login_times_out_with_vanillas_translated_reason() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let port = listener.local_addr().expect("listener address").port();
        let (mut handle, to_game, _from_game) = NetHandle::new_pair();
        let opts = ConnectOptions::offline("127.0.0.1", port, "TimeoutTest");

        let connect = tokio::spawn(async move {
            connect_and_login_with_timeouts(
                &opts,
                &to_game,
                &mut CaptureSink::disabled(),
                Duration::from_millis(25),
            )
            .await
        });
        let (socket, _) = listener.accept().await.expect("accept TCP connection");

        let first = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Some(message) = handle.try_recv() {
                    break message;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("socket phase");
        assert!(matches!(first, ToGame::SocketConnected));

        let reason = tokio::time::timeout(Duration::from_secs(1), connect)
            .await
            .expect("bounded login wait")
            .expect("connection task")
            .expect_err("silent login must time out");
        assert_eq!(reason, timeout_reason());
        assert_eq!(
            reason.to_component_json(),
            r#"{"translate":"disconnect.timeout"}"#
        );
        drop(socket);
    }

    #[tokio::test]
    async fn partial_login_frame_bytes_reset_the_read_deadline() {
        use tokio::io::AsyncWriteExt;

        let (activity_tx, mut activity_rx) = tokio::sync::mpsc::unbounded_channel();
        let (transport, mut peer) = tokio::io::duplex(16);
        let mut connection = Connection::new(ReadActivity::new(transport, activity_tx));
        let sender = tokio::spawn(async move {
            // Frame length first, then the two-byte payload. Neither write alone completes a
            // frame, so a packet-level observer cannot make this test pass.
            tokio::time::sleep(Duration::from_millis(120)).await;
            peer.write_all(&[2]).await.expect("length prefix");
            tokio::time::sleep(Duration::from_millis(120)).await;
            peer.write_all(&[0, 0]).await.expect("payload");
        });

        // A fixed 200 ms cap expires between the writes. The byte read at 120 ms refreshes the
        // deadline to 320 ms, so the complete frame at 240 ms must succeed like Netty's
        // head-of-pipeline ReadTimeoutHandler does.
        let packet = await_with_read_timeout(
            connection.read_payload(),
            &mut activity_rx,
            Duration::from_millis(200),
        )
        .await
        .expect("partial-frame bytes keep the read alive")
        .expect("decode frame");
        assert_eq!(packet.as_slice(), &[0, 0]);
        sender.await.expect("sender task");
    }

    #[tokio::test]
    async fn stale_read_activity_does_not_extend_a_phase_deadline() {
        let (activity_tx, mut activity_rx) = tokio::sync::mpsc::unbounded_channel();
        // This signal represents bytes read a full minute ago. If the timeout handler refreshes
        // from the receive time instead of the transport timestamp, the pending phase survives
        // for its full five-second budget and the outer 250 ms control times out.
        activity_tx
            .send(ActivityInstant::now() - Duration::from_secs(60))
            .expect("activity receiver is alive");

        let result = tokio::time::timeout(
            Duration::from_millis(250),
            await_with_read_timeout(
                std::future::pending::<()>(),
                &mut activity_rx,
                Duration::from_secs(5),
            ),
        )
        .await
        .expect("stale activity must not postpone the phase timeout");
        assert_eq!(result, Err(ReadTimedOut));
    }

    #[tokio::test]
    async fn stale_read_activity_does_not_extend_the_play_deadline() {
        let (client, _peer) = tokio::io::duplex(64);
        let (_handle, to_game, immediate_to_game, from_game) = NetHandle::new_live_pair();
        let (activity_tx, activity_rx) = tokio::sync::mpsc::unbounded_channel();
        activity_tx
            .send(ActivityInstant::now() - Duration::from_secs(60))
            .expect("activity receiver is alive");
        let (reader, writer) = Connection::new(client).split();

        let reason = tokio::time::timeout(
            Duration::from_millis(250),
            play_loop(
                reader,
                writer,
                &to_game,
                &immediate_to_game,
                from_game,
                activity_rx,
                Duration::from_secs(5),
                &mut CaptureSink::disabled(),
            ),
        )
        .await
        .expect("stale activity must not postpone the play timeout");
        assert_eq!(reason, timeout_reason());
    }

    fn component_value(reason: &DisconnectReason) -> serde_json::Value {
        serde_json::from_str(&reason.to_component_json()).expect("component JSON")
    }

    #[test]
    fn login_disconnect_errors_use_vanilla_components_without_rust_prose() {
        let closed = disconnect_reason_for(&ClientError::Protocol(ProtoError::ConnectionClosed));
        assert_eq!(
            component_value(&closed),
            serde_json::json!({ "translate": "disconnect.endOfStream" })
        );

        let malformed = disconnect_reason_for(&ClientError::Protocol(ProtoError::VarIntTooBig));
        assert_eq!(
            component_value(&malformed),
            serde_json::json!({
                "translate": "disconnect.genericReason",
                "with": ["Internal Exception: io.netty.handler.codec.DecoderException"],
            })
        );

        let timed_out = disconnect_reason_for(&ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "Rust-only timeout detail",
        )));
        assert_eq!(
            component_value(&timed_out),
            serde_json::json!({ "translate": "disconnect.timeout" })
        );

        let io = disconnect_reason_for(&ClientError::Io(std::io::Error::other(
            "Rust-only socket detail",
        )));
        assert_eq!(
            component_value(&io),
            serde_json::json!({
                "translate": "disconnect.genericReason",
                "with": ["Internal Exception: java.io.IOException"],
            })
        );

        let unknown = disconnect_reason_for(&ClientError::StatusJson(
            "Rust-only JSON parser detail".to_owned(),
        ));
        assert_eq!(
            component_value(&unknown),
            serde_json::json!({
                "translate": "disconnect.genericReason",
                "with": ["Internal Exception: java.lang.RuntimeException"],
            })
        );

        let invalid_session = disconnect_reason_for(&ClientError::Protocol(ProtoError::Auth(
            mc_auth::AuthError::TokenRejected,
        )));
        assert_eq!(
            component_value(&invalid_session),
            serde_json::json!({
                "translate": "disconnect.loginFailedInfo",
                "with": [{ "translate": "disconnect.loginFailedInfo.invalidSession" }],
            })
        );

        // Any other Yggdrasil error is vanilla's plain
        // AuthenticationException, whose message is the argument (it used to be one opaque
        // "Authentication server rejected the request" line).
        let refused = disconnect_reason_for(&ClientError::Protocol(ProtoError::Auth(
            mc_auth::AuthError::JoinRefused {
                status: 403,
                error: "InsufficientPrivilegesException".into(),
                message: Some("Multiplayer is disabled".into()),
            },
        )));
        assert_eq!(
            component_value(&refused),
            serde_json::json!({
                "translate": "disconnect.loginFailedInfo",
                "with": ["Multiplayer is disabled"],
            })
        );

        // An unreadable answer is vanilla's AuthenticationUnavailableException, plus its status.
        let unreadable = disconnect_reason_for(&ClientError::Protocol(ProtoError::Auth(
            mc_auth::AuthError::JoinUnreadable {
                status: 503,
                detail: "an empty body".into(),
            },
        )));
        assert_eq!(
            component_value(&unreadable),
            serde_json::json!({
                "translate": "disconnect.loginFailedInfo",
                "with": [{
                    "translate": "disconnect.loginFailedInfo.serversUnavailable",
                    "extra": [{ "text": " (HTTP 503)" }],
                }],
            })
        );

        // `AuthError::Network` contains reqwest's opaque error type, which downstream crates
        // cannot construct without adding a test-only dependency. Exercise the exact component
        // constructor shared by that match arm instead.
        let unavailable = login_failed_nested("disconnect.loginFailedInfo.serversUnavailable");
        assert_eq!(
            component_value(&unavailable),
            serde_json::json!({
                "translate": "disconnect.loginFailedInfo",
                "with": [{ "translate": "disconnect.loginFailedInfo.serversUnavailable" }],
            })
        );
    }

    /// A JWT-shaped token with `claims` as its payload (unsigned: nothing verifies it).
    fn fake_jwt(claims: &serde_json::Value) -> String {
        use base64::Engine as _;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        format!(
            "{}.{}.c2lnbmF0dXJl",
            engine.encode(br#"{"alg":"HS256"}"#),
            engine.encode(claims.to_string())
        )
    }

    /// The text a player reads for each join failure,
    /// rendered through the real component code with vanilla's en_US strings.
    #[test]
    fn join_failures_read_like_vanilla_on_the_disconnect_screen() {
        use mc_auth::AuthError;
        let mut lang = mc_assets::Language::new("en_US");
        lang.merge_lang_str(concat!(
            "disconnect.loginFailedInfo=Failed to login: %s\n",
            "disconnect.loginFailedInfo.serversUnavailable=The authentication servers are ",
            "currently down for maintenance.\n",
            "disconnect.loginFailedInfo.invalidSession=Invalid session (Try restarting your game)\n",
        ));
        let screen = |auth: &AuthError, token: Option<TokenProblem>| {
            let json = auth_failure_reason_for(auth, token).to_component_json();
            mc_ui::font::component::ChatComponent::from_json(&json)
                .expect("component")
                .unformatted_text(&lang)
        };
        let refused = |error: &str, message: Option<&str>| AuthError::JoinRefused {
            status: 403,
            error: error.into(),
            message: message.map(Into::into),
        };
        let unreadable = |status| AuthError::JoinUnreadable {
            status,
            detail: String::new(),
        };
        let expired = || Some(TokenProblem::Expired);
        let cases = [
            (
                AuthError::TokenRejected,
                None,
                "Failed to login: Invalid session (Try restarting your game)",
            ),
            (
                refused(
                    "InsufficientPrivilegesException",
                    Some("Multiplayer is disabled"),
                ),
                None,
                "Failed to login: Multiplayer is disabled",
            ),
            (
                refused("UserBannedException", None),
                None,
                "Failed to login: This account is banned from online multiplayer.",
            ),
            (
                refused("SomethingNewException", None),
                None,
                "Failed to login: SomethingNewException",
            ),
            (
                unreadable(502),
                None,
                "Failed to login: The authentication servers are currently down for \
                 maintenance. (HTTP 502)",
            ),
            (
                unreadable(429),
                None,
                "Failed to login: Too many logins in a short time (HTTP 429). Wait a minute, \
                 then try again.",
            ),
            // The token's own claims explain any answer from the session server.
            (
                AuthError::TokenRejected,
                expired(),
                "Failed to login: Your Minecraft login has expired (logins last about a day). \
                 Close the game and start it again from your launcher to refresh it.",
            ),
            (
                unreadable(500),
                expired(),
                "Failed to login: Your Minecraft login has expired (logins last about a day). \
                 Close the game and start it again from your launcher to refresh it.",
            ),
            (
                refused("ForbiddenOperationException", None),
                Some(TokenProblem::OtherAccount {
                    username: "Steve".into(),
                }),
                "Failed to login: The login your launcher passed is for a different Minecraft \
                 account than Steve. Sign in to Steve again in your launcher, then restart the \
                 game.",
            ),
        ];
        for (auth, token, expected) in cases {
            let label = format!("{auth:?} {token:?}");
            assert_eq!(screen(&auth, token), expected, "{label}");
        }
    }

    #[test]
    fn token_checks_explain_a_dead_token_without_showing_it() {
        let me = uuid::Uuid::from_u128(7);
        let session =
            |claims: serde_json::Value| mc_auth::Session::new(me, "Steve", fake_jwt(&claims));
        let now = 1_800_000_000u64;
        let valid = TokenCheck::of(
            &session(serde_json::json!({ "exp": now + 3 * 3600 + 300 })),
            now,
        );
        assert_eq!(valid.to_string(), "valid for 3h 05m");
        assert_eq!(valid.problem("Steve"), None);

        let expired = TokenCheck::of(&session(serde_json::json!({ "exp": now - 12 * 60 })), now);
        assert_eq!(expired.to_string(), "expired 12m ago");
        assert_eq!(expired.problem("Steve"), Some(TokenProblem::Expired));

        // Within the clock-skew allowance of expiry, a refusal is explained as expiry too.
        let nearly = TokenCheck::of(&session(serde_json::json!({ "exp": now + 90 })), now);
        assert_eq!(nearly.to_string(), "valid for 1m");
        assert_eq!(nearly.problem("Steve"), Some(TokenProblem::Expired));
        let later = TokenCheck::of(&session(serde_json::json!({ "exp": now + 180 })), now);
        assert_eq!(later.problem("Steve"), None);

        let other = TokenCheck::of(
            &session(serde_json::json!({
                "exp": now + 7200,
                "profiles": { "mc": uuid::Uuid::from_u128(8).to_string() },
            })),
            now,
        );
        assert_eq!(
            other.to_string(),
            "valid for 2h 00m, issued for a different Minecraft account"
        );
        assert_eq!(
            other.problem("Steve"),
            Some(TokenProblem::OtherAccount {
                username: "Steve".into()
            })
        );

        let opaque = TokenCheck::of(&mc_auth::Session::new(me, "Steve", "opaque-token"), now);
        assert_eq!(opaque.to_string(), "no expiry in the token");
        assert_eq!(opaque.problem("Steve"), None);
        assert_eq!(human_duration(40), "40s");
    }

    /// The start-up line says how long the launcher's token lasts, at the right level, and
    /// never contains the token.
    #[test]
    fn the_login_token_log_never_contains_the_token() {
        #[derive(Clone, Default)]
        struct Buffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Buffer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().expect("buffer").extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let me = uuid::Uuid::from_u128(7);
        let now = mc_auth::unix_now();
        let other = uuid::Uuid::from_u128(8).to_string();
        let cases = [
            (
                fake_jwt(&serde_json::json!({ "exp": now + 5 * 3600 + 30 })),
                "INFO",
                "valid for 5h 00m",
            ),
            (
                fake_jwt(&serde_json::json!({ "exp": now + 630 })),
                "WARN",
                "valid for 10m",
            ),
            (
                fake_jwt(&serde_json::json!({ "exp": now - 3600 })),
                "WARN",
                "expired 1h 00m ago",
            ),
            (
                fake_jwt(&serde_json::json!({ "exp": now + 7200, "profiles": { "mc": other } })),
                "WARN",
                "different Minecraft account",
            ),
            (
                "opaque-launcher-token".to_owned(),
                "DEBUG",
                "carries no expiry",
            ),
        ];
        for (token, level, words) in cases {
            let buffer = Buffer::default();
            let writer = buffer.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_writer(move || writer.clone())
                .with_max_level(tracing::Level::DEBUG)
                .with_ansi(false)
                .finish();
            let session = mc_auth::Session::new(me, "Steve", token.clone());
            tracing::subscriber::with_default(subscriber, || log_login_token(&session));
            let text = String::from_utf8(buffer.0.lock().expect("buffer").clone()).expect("utf8");
            assert!(text.contains(level), "{words}: {text}");
            assert!(text.contains(words), "{words}: {text}");
            assert!(!text.contains(&token), "the token leaked: {text}");
            assert!(
                !text.contains("c2lnbmF0dXJl"),
                "the signature leaked: {text}"
            );
        }
    }

    /// A throwaway 1024-bit RSA public key (SPKI DER) for the fake online-mode server. Its
    /// private half was never kept: these tests only need the client to encrypt to it.
    fn test_server_key_der() -> Vec<u8> {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(concat!(
                "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDEL+giQEzMHibQ6c/1K/721lwwZsTv8eXpeaOIo1Rm",
                "z2hXAxCri+b3EVjg0F+vh71yTaOkIAhQOgP3WHzsKp5RDS0nnCyCxO98a3KhUKqlRnOnF8Kywhtz8p6q",
                "7Kd/oEsN9g3r1KFM0gLbhTSSRKy5b4qVR/oliwcGLU4kKkS1bwIDAQAB",
            ))
            .expect("test key")
    }

    /// A fake session server on loopback: answers each request with the next canned
    /// `(status, body)` (the last one repeats) and keeps every request body it received.
    async fn fake_session_server(
        answers: Vec<(u16, &'static str)>,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fake session server");
        let url = format!(
            "http://{}/session/minecraft/join",
            listener.local_addr().expect("address")
        );
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            let mut answers = answers.into_iter();
            let mut last = None;
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut request = Vec::new();
                let mut chunk = [0u8; 4096];
                let mut body_start = None;
                let mut length = 0usize;
                loop {
                    if let Some(start) = body_start {
                        if request.len() >= start + length {
                            break;
                        }
                    }
                    let read = socket.read(&mut chunk).await.unwrap_or(0);
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&chunk[..read]);
                    if body_start.is_none() {
                        if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&request[..end]).to_lowercase();
                            length = head
                                .lines()
                                .find_map(|line| line.strip_prefix("content-length:"))
                                .and_then(|value| value.trim().parse().ok())
                                .unwrap_or(0);
                            body_start = Some(end + 4);
                        }
                    }
                }
                let body = body_start
                    .map(|start| String::from_utf8_lossy(&request[start..]).into_owned())
                    .unwrap_or_default();
                log.lock().expect("log").push(body);
                let (status, text) = answers.next().or(last).expect("an answer");
                last = Some((status, text));
                let response = format!(
                    "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{text}",
                    text.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (url, seen)
    }

    /// A fake online-mode game server: reads the handshake and Login Start, asks for
    /// encryption, then reports the id of the client's next packet (`Some(0x01)` for the
    /// Encryption Response), or `None` if the client hung up instead.
    async fn fake_online_server() -> (u16, tokio::task::JoinHandle<Option<i32>>) {
        use mc_protocol::handshake::Handshake;
        use mc_protocol::login::{EncryptionRequest, LoginStart};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fake game server");
        let port = listener.local_addr().expect("address").port();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept");
            let mut conn = Connection::new(socket);
            let (id, _) = conn.read_packet().await.expect("handshake");
            assert_eq!(id, Handshake::ID);
            conn.set_state(mc_protocol::State::Login)
                .expect("login state");
            let (id, _) = conn.read_packet().await.expect("login start");
            assert_eq!(id, LoginStart::ID);
            conn.send(&EncryptionRequest {
                server_id: String::new(),
                public_key_der: test_server_key_der(),
                verify_token: vec![1, 2, 3, 4],
            })
            .await
            .expect("encryption request");
            conn.read_packet().await.ok().map(|(id, _)| id)
        });
        (port, server)
    }

    /// The whole online login over loopback, with a fake
    /// session server standing in for Mojang. The bug report, "Failed to login:
    /// Authentication server rejected the request", came from this path; each case checks
    /// what reaches the disconnect screen, what the session server received, and whether the
    /// game server got the Encryption Response (only after a successful join, as in vanilla).
    #[tokio::test]
    async fn online_login_reports_the_session_servers_reason_end_to_end() {
        let me = uuid::Uuid::from_u128(0x2a);
        let now = mc_auth::unix_now();
        let valid = fake_jwt(&serde_json::json!({ "exp": now + 7200 }));
        let expired = fake_jwt(&serde_json::json!({ "exp": now - 600 }));
        let refused = r#"{"error":"InsufficientPrivilegesException","errorMessage":"Multiplayer is disabled"}"#;
        let forbidden = r#"{"error":"ForbiddenOperationException","errorMessage":"Invalid token"}"#;
        let html = "<html><head><title>500 Internal Server Error</title></head></html>";
        let expiry_text = TokenProblem::Expired.screen_text();
        // (what, token, session-server answers, reason on screen, session-server requests)
        type Case = (
            &'static str,
            String,
            Vec<(u16, &'static str)>,
            serde_json::Value,
            usize,
        );
        let cases: Vec<Case> = vec![
            (
                "refusal with a message: vanilla shows the message",
                valid.clone(),
                vec![(403, refused)],
                serde_json::json!({
                    "translate": "disconnect.loginFailedInfo",
                    "with": ["Multiplayer is disabled"],
                }),
                1,
            ),
            (
                "expired token: say so",
                expired,
                vec![(403, forbidden)],
                serde_json::json!({
                    "translate": "disconnect.loginFailedInfo",
                    "with": [expiry_text],
                }),
                1,
            ),
            (
                "a 500 page twice: retried once, then vanilla's servers-unavailable line",
                valid.clone(),
                vec![(500, html)],
                serde_json::json!({
                    "translate": "disconnect.loginFailedInfo",
                    "with": [{
                        "translate": "disconnect.loginFailedInfo.serversUnavailable",
                        "extra": [{ "text": " (HTTP 500)" }],
                    }],
                }),
                2,
            ),
        ];
        for (label, token, answers, expected, requests) in cases {
            let (url, seen) = fake_session_server(answers).await;
            let (port, server) = fake_online_server().await;
            let session = mc_auth::Session::new(me, "Steve", token.clone());
            let opts =
                ConnectOptions::online("127.0.0.1", port, session).with_session_join_url(url);
            let (_handle, to_game, _from_game) = NetHandle::new_pair();
            let reason = connect_and_login_with_timeouts(
                &opts,
                &to_game,
                &mut CaptureSink::disabled(),
                Duration::from_secs(10),
            )
            .await
            .map(|_| ())
            .expect_err(label);
            assert_eq!(component_value(&reason), expected, "{label}");
            assert!(
                !reason.to_component_json().contains(&token),
                "{label}: the token must never reach the screen"
            );
            assert_eq!(
                server.await.expect("game server"),
                None,
                "{label}: no Encryption Response after a failed join"
            );
            let seen = seen.lock().expect("requests").clone();
            assert_eq!(seen.len(), requests, "{label}: session server requests");
            let body: serde_json::Value = serde_json::from_str(&seen[0]).expect("join body");
            assert_eq!(body["accessToken"], token.as_str(), "{label}");
            assert_eq!(body["selectedProfile"], me.simple().to_string(), "{label}");
        }

        // A 503 then a 204: the retry succeeds and the login carries on to the Encryption
        // Response, exactly as after a first-time success.
        let (url, seen) = fake_session_server(vec![(503, ""), (204, "")]).await;
        let (port, server) = fake_online_server().await;
        let session = mc_auth::Session::new(me, "Steve", valid);
        let opts = ConnectOptions::online("127.0.0.1", port, session).with_session_join_url(url);
        let (_handle, to_game, _from_game) = NetHandle::new_pair();
        let mut capture = CaptureSink::disabled();
        let login =
            connect_and_login_with_timeouts(&opts, &to_game, &mut capture, Duration::from_secs(10));
        let (result, sent) = tokio::join!(login, server);
        assert_eq!(
            sent.expect("game server"),
            Some(0x01),
            "Encryption Response sent"
        );
        // The fake server then hangs up instead of finishing the login.
        assert!(result.is_err());
        assert_eq!(seen.lock().expect("requests").len(), 2);
    }

    #[test]
    fn server_disconnect_components_validate_at_the_network_boundary() {
        // Gson is lenient here. Validation must not normalize or reserialize a valid server reason.
        let lenient = "{text:'bye'}";
        assert_eq!(
            server_component_reason(lenient),
            DisconnectReason::Kicked {
                reason_json: lenient.into(),
            }
        );

        let malformed = disconnect_reason_for(&ClientError::Protocol(ProtoError::Disconnected {
            reason: "{".into(),
        }));
        let component = component_value(&malformed);
        assert_eq!(component["translate"], "disconnect.genericReason");
        let detail = component["with"][0].as_str().expect("generic detail");
        assert_eq!(
            detail,
            "Internal Exception: com.google.gson.JsonSyntaxException"
        );
        assert_ne!(component, serde_json::json!({ "text": "{" }));
    }

    #[test]
    fn every_play_read_chat_field_is_validated_before_dispatch() {
        let invalid = "{".to_owned();
        let valid = r#"{"text":"ok"}"#.to_owned();
        let uuid = uuid::Uuid::nil();
        let packets = vec![
            (
                "chat",
                S2CPlay::Chat(s2c::Chat {
                    component: invalid.clone(),
                    position: 0,
                }),
            ),
            (
                "open-window title",
                S2CPlay::OpenWindow(s2c::OpenWindow {
                    window_id: 1,
                    inventory_type: "minecraft:chest".into(),
                    window_title: invalid.clone(),
                    slot_count: 9,
                    entity_id: None,
                }),
            ),
            (
                "sign line",
                S2CPlay::UpdateSign(s2c::UpdateSign {
                    position: Default::default(),
                    lines: [valid.clone(), valid.clone(), invalid.clone(), valid.clone()],
                }),
            ),
            (
                "add-player display name",
                S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![
                    s2c::AddPlayerEntry {
                        uuid,
                        username: "Notch".into(),
                        properties: Vec::new(),
                        game_mode: s2c::GameMode::Survival,
                        ping: 0,
                        display_name: Some(invalid.clone()),
                    },
                ])),
            ),
            (
                "updated display name",
                S2CPlay::PlayerListItem(s2c::PlayerListItem::UpdateDisplayName(vec![
                    s2c::UpdateDisplayNameEntry {
                        uuid,
                        display_name: Some(invalid.clone()),
                    },
                ])),
            ),
            (
                "disconnect reason",
                S2CPlay::Disconnect(s2c::Disconnect {
                    reason: invalid.clone(),
                }),
            ),
            (
                "title",
                S2CPlay::Title(s2c::Title::SetTitle {
                    text: invalid.clone(),
                }),
            ),
            (
                "subtitle",
                S2CPlay::Title(s2c::Title::SetSubtitle {
                    text: invalid.clone(),
                }),
            ),
            (
                "tab header",
                S2CPlay::PlayerListHeaderFooter(s2c::PlayerListHeaderFooter {
                    header: invalid.clone(),
                    footer: valid.clone(),
                }),
            ),
            (
                "tab footer",
                S2CPlay::PlayerListHeaderFooter(s2c::PlayerListHeaderFooter {
                    header: valid,
                    footer: invalid,
                }),
            ),
        ];

        for (field, packet) in packets {
            assert!(
                validate_play_components(&packet).is_err(),
                "malformed {field} reached dispatch"
            );
        }
    }

    /// `PacketBuffer.readChatComponent` is `GSON.fromJson`, which returns
    /// null (it does not throw) for a root `null`, an empty document and `[]`. None of those is a
    /// decoder exception, so none may close the connection at this boundary. The former strict
    /// validation disconnected on every one of them.
    #[test]
    fn gson_null_components_are_not_decoder_errors() {
        let uuid = uuid::Uuid::nil();
        for null_json in ["null", "", "[]", " null ", "[[]]"] {
            let null_json = null_json.to_owned();
            assert!(
                mc_ui::font::component::ChatComponent::from_json(&null_json).is_err(),
                "control: the former non-nullable validator rejected {null_json:?}"
            );
            let entry = s2c::AddPlayerEntry {
                uuid,
                username: "Notch".into(),
                properties: Vec::new(),
                game_mode: s2c::GameMode::Survival,
                ping: 0,
                display_name: Some(null_json.clone()),
            };
            let packets = vec![
                S2CPlay::Chat(s2c::Chat {
                    component: null_json.clone(),
                    position: 0,
                }),
                S2CPlay::Chat(s2c::Chat {
                    component: null_json.clone(),
                    position: 2,
                }),
                S2CPlay::OpenWindow(s2c::OpenWindow {
                    window_id: 1,
                    inventory_type: "minecraft:chest".into(),
                    window_title: null_json.clone(),
                    slot_count: 27,
                    entity_id: None,
                }),
                S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![entry])),
                S2CPlay::PlayerListItem(s2c::PlayerListItem::UpdateDisplayName(vec![
                    s2c::UpdateDisplayNameEntry {
                        uuid,
                        display_name: Some(null_json.clone()),
                    },
                ])),
                S2CPlay::Disconnect(s2c::Disconnect {
                    reason: null_json.clone(),
                }),
                S2CPlay::Title(s2c::Title::SetTitle {
                    text: null_json.clone(),
                }),
                S2CPlay::Title(s2c::Title::SetSubtitle {
                    text: null_json.clone(),
                }),
                S2CPlay::PlayerListHeaderFooter(s2c::PlayerListHeaderFooter {
                    header: null_json.clone(),
                    footer: null_json.clone(),
                }),
            ];
            for packet in packets {
                assert!(
                    validate_play_components(&packet).is_ok(),
                    "{null_json:?} is a Gson null, not a decoder error: {packet:?}"
                );
            }
        }
    }

    /// java: S40 and the Login S00 pass a Gson-null reason to `closeChannel`; with a null
    /// `terminationReason`, `checkDisconnected` shows `new ChatComponentText("Disconnected")`.
    #[test]
    fn a_gson_null_disconnect_reason_is_vanillas_disconnected_text() {
        for null_json in ["null", "", "[]"] {
            let disconnected = serde_json::json!({ "text": "Disconnected" });
            assert_eq!(
                component_value(&server_component_reason(null_json)),
                disconnected,
                "play {null_json:?}"
            );
            let login = disconnect_reason_for(&ClientError::Protocol(ProtoError::Disconnected {
                reason: null_json.into(),
            }));
            assert_eq!(component_value(&login), disconnected, "login {null_json:?}");
        }
    }

    /// End to end through the real play loop: a Gson-null S02 (chat or action bar) and S45 are
    /// scheduled for the game thread and the connection carries on, exactly as vanilla's
    /// decoder does; a following KeepAlive is still answered. A Gson-null S40 then closes with
    /// vanilla's "Disconnected". The former validation closed the channel at the first S02 with
    /// `Internal Exception: com.google.gson.JsonSyntaxException`, so no echo ever came back.
    #[tokio::test]
    async fn gson_null_chat_and_title_keep_the_connection_and_reach_the_game_thread() {
        let (mut handle, mut server, task) = rig();
        server
            .send(&s2c::Chat {
                component: "null".into(),
                position: 2,
            })
            .await
            .expect("send null action bar");
        server
            .send(&s2c::Title::SetSubtitle {
                text: String::new(),
            })
            .await
            .expect("send empty-document subtitle");
        server
            .send(&s2c::KeepAlive { id: 0x4e55 })
            .await
            .expect("send barrier");
        let (id, mut body) = server
            .read_packet()
            .await
            .expect("echo after null components");
        assert_eq!(id, c2s::KeepAlive::ID);
        assert_eq!(c2s::KeepAlive::read(&mut body).unwrap().key, 0x4e55);

        match expect_packet(&mut handle).await {
            S2CPlay::Chat(chat) => {
                assert_eq!((chat.component.as_str(), chat.position), ("null", 2));
            }
            other => panic!("expected the null S02 on the scheduled queue, got {other:?}"),
        }
        match expect_packet(&mut handle).await {
            S2CPlay::Title(title) => assert_eq!(
                title,
                s2c::Title::SetSubtitle {
                    text: String::new()
                }
            ),
            other => panic!("expected the null S45 on the scheduled queue, got {other:?}"),
        }

        server
            .send(&s2c::Disconnect {
                reason: "null".into(),
            })
            .await
            .expect("send null kick");
        assert_eq!(
            component_value(&task.await.unwrap()),
            serde_json::json!({ "text": "Disconnected" })
        );
    }

    /// java: handlePlayerListHeaderFooter runs on the Netty thread and calls
    /// `getFormattedText()` on both components without a null check. A Gson-null header or footer
    /// is a NullPointerException in `channelRead0`, so `exceptionCaught` closes the channel with
    /// that exception's name, and nothing reaches the game thread.
    #[tokio::test]
    async fn a_gson_null_tab_header_or_footer_throws_on_the_netty_thread() {
        for (header, footer) in [("null", r#"{"text":"f"}"#), (r#"{"text":"h"}"#, "[]")] {
            let (mut handle, mut server, task) = rig();
            server
                .send(&s2c::PlayerListHeaderFooter {
                    header: header.into(),
                    footer: footer.into(),
                })
                .await
                .expect("send header/footer");
            assert_eq!(
                component_value(&task.await.unwrap()),
                serde_json::json!({
                    "translate": "disconnect.genericReason",
                    "with": ["Internal Exception: java.lang.NullPointerException"],
                }),
                "header {header:?}, footer {footer:?}"
            );
            assert!(
                handle.try_recv().is_none(),
                "the throwing Netty handler must not forward the packet"
            );
        }
    }

    #[test]
    fn pre_channel_failures_hide_resolved_addresses_and_name_unknown_hosts() {
        let opts = ConnectOptions::offline("example.test", 25565, "ConnectTest");
        let unknown = open_failure_reason(&OpenError::UnknownHost, &opts);
        assert_eq!(
            component_value(&unknown),
            serde_json::json!({
                "translate": "disconnect.genericReason",
                "with": ["Unknown host"],
            })
        );

        let resolved: SocketAddr = "203.0.113.7:25565".parse().expect("socket address");
        let refused = OpenError::Connect {
            error: ProtoError::Io(std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                format!("connection refused at example.test/203.0.113.7:25565 ({resolved})"),
            )),
            resolved,
        };
        let component = component_value(&open_failure_reason(&refused, &opts));
        assert_eq!(component["translate"], "disconnect.genericReason");
        let detail = component["with"][0].as_str().expect("generic detail");
        assert!(!detail.contains("example.test"), "{detail}");
        assert!(!detail.contains("203.0.113.7"), "{detail}");
        assert!(!detail.contains("25565"), "{detail}");
        assert!(detail.contains("connection refused"), "{detail}");
    }

    #[tokio::test]
    async fn keep_alive_is_answered_by_the_task_without_the_game_loop() {
        let (mut handle, mut server, task) = rig();
        server
            .send(&s2c::KeepAlive { id: 0x4321 })
            .await
            .expect("send");

        let (id, mut body) = server.read_packet().await.expect("echo");
        assert_eq!(id, c2s::KeepAlive::ID);
        assert_eq!(c2s::KeepAlive::read(&mut body).unwrap().key, 0x4321);

        // The game loop was never polled, and the keep-alive must not appear in its queue.
        assert!(handle.try_recv().is_none());
        handle.disconnect().expect("disconnect");
        assert_eq!(task.await.unwrap(), DisconnectReason::ClosedByClient);
    }

    /// `S47` and `S48` are two of vanilla's five handlers which deliberately do not call
    /// `PacketThreadUtil.checkThreadAndEnqueue`. The safe Rust representation is an immediate
    /// lane: it cannot mutate `ClientState` on the Tokio worker, but after its handler has run it
    /// must bypass every scheduled packet still backed up for the game thread.
    #[tokio::test]
    async fn header_footer_and_resource_pack_bypass_a_full_scheduled_queue() {
        let (mut handle, mut server, task) = rig();
        let burst_len = INBOUND_CAPACITY + 3;
        for index in 0..burst_len {
            server
                .send(&s2c::Chat {
                    component: format!(r#"{{"text":"scheduled-{index}"}}"#),
                    position: 0,
                })
                .await
                .expect("send scheduled packet");
        }
        server
            .send(&s2c::PlayerListHeaderFooter {
                header: r#"{"text":"netty-header"}"#.to_owned(),
                footer: r#"{"text":"netty-footer"}"#.to_owned(),
            })
            .await
            .expect("send direct header/footer handler");
        server
            .send(&s2c::ResourcePackSend {
                url: "https://example.invalid/pack.zip".to_owned(),
                hash: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            })
            .await
            .expect("send direct resource-pack handler");

        // A completed socket write does not mean the peer task has decoded that frame. Java has
        // the same concurrency boundary: its game thread may drain an older scheduled task before
        // the Netty event loop reaches a later S47/S48. A following direct KeepAlive and its echo
        // form an exact processing barrier because play_loop decodes clientbound frames in order.
        server
            .send(&s2c::KeepAlive { id: 0x47_48 })
            .await
            .expect("send processing barrier");
        let (id, mut body) = server.read_packet().await.expect("read barrier echo");
        assert_eq!(id, c2s::KeepAlive::ID);
        assert_eq!(
            c2s::KeepAlive::read(&mut body)
                .expect("decode barrier echo")
                .key,
            0x47_48
        );

        assert!(matches!(
            expect_packet(&mut handle).await,
            S2CPlay::PlayerListHeaderFooter(_)
        ));
        assert!(matches!(
            expect_packet(&mut handle).await,
            S2CPlay::ResourcePackSend(_)
        ));

        // Negative control: Chat is `PacketThreadUtil`-scheduled and stays behind the direct
        // lane even though it was decoded first. Making every packet use the immediate sender
        // would make this assertion fail.
        match expect_packet(&mut handle).await {
            S2CPlay::Chat(chat) => assert_eq!(chat.component, r#"{"text":"scheduled-0"}"#),
            other => panic!("ordinary packet unexpectedly used the Netty-bypass lane: {other:?}"),
        }

        handle.disconnect().expect("disconnect");
        assert_eq!(task.await.unwrap(), DisconnectReason::ClosedByClient);
    }

    #[tokio::test]
    async fn partial_play_frame_bytes_reset_the_read_deadline() {
        use tokio::io::AsyncWriteExt;

        let (client, mut peer) = tokio::io::duplex(64);
        let (handle, to_game, immediate_to_game, from_game) = NetHandle::new_live_pair();
        let (activity_tx, activity_rx) = tokio::sync::mpsc::unbounded_channel();
        let (reader, writer) = Connection::new(ReadActivity::new(client, activity_tx)).split();
        let task = tokio::spawn(async move {
            play_loop(
                reader,
                writer,
                &to_game,
                &immediate_to_game,
                from_game,
                activity_rx,
                Duration::from_millis(200),
                &mut CaptureSink::disabled(),
            )
            .await
        });

        // A complete clientbound KeepAlive frame is [length=2, id=0, key=42]. Each socket write
        // arrives before 200 ms of silence, but the packet itself is not decodable until 240 ms.
        // A packet-boundary timer expires; Netty's raw-read timer and this implementation do not.
        tokio::time::sleep(Duration::from_millis(120)).await;
        peer.write_all(&[2]).await.expect("length prefix");
        tokio::time::sleep(Duration::from_millis(120)).await;
        peer.write_all(&[0, 42]).await.expect("packet payload");

        let mut server = Connection::new(peer);
        let (id, mut body) = tokio::time::timeout(Duration::from_secs(1), server.read_packet())
            .await
            .expect("slow active frame must survive")
            .expect("keep-alive echo");
        assert_eq!(id, c2s::KeepAlive::ID);
        assert_eq!(c2s::KeepAlive::read(&mut body).unwrap().key, 42);

        handle.disconnect().expect("disconnect");
        assert_eq!(task.await.unwrap(), DisconnectReason::ClosedByClient);
    }

    #[tokio::test]
    async fn full_game_queue_cannot_block_a_following_keep_alive_echo() {
        let (mut handle, mut server, task) = rig();

        // Fill the bounded game handoff and put one more ordinary packet in the spill FIFO.
        // Never polling `handle` is essential: the pre-fix task parks on packet 4097 and cannot
        // read the KeepAlive that follows it.
        let burst_len = INBOUND_CAPACITY + 3;
        for index in 0..burst_len {
            server
                .send(&s2c::Chat {
                    component: index.to_string(),
                    position: 0,
                })
                .await
                .expect("send join-burst packet");
        }
        server
            .send(&s2c::KeepAlive { id: 0x4a11 })
            .await
            .expect("send keep-alive after full queue");

        let (id, mut body) = tokio::time::timeout(Duration::from_secs(1), server.read_packet())
            .await
            .expect("keep-alive echo cannot wait for a game-loop poll")
            .expect("read echo");
        assert_eq!(id, c2s::KeepAlive::ID);
        assert_eq!(c2s::KeepAlive::read(&mut body).unwrap().key, 0x4a11);

        // The overflow path is only allowed to change storage, never dispatch order.
        for expected in 0..burst_len {
            match expect_packet(&mut handle).await {
                S2CPlay::Chat(chat) => assert_eq!(chat.component, expected.to_string()),
                other => panic!("expected Chat, got {other:?}"),
            }
        }

        handle.disconnect().expect("disconnect");
        assert_eq!(task.await.unwrap(), DisconnectReason::ClosedByClient);
    }

    /// java: everything Netty scheduled before an S40 still runs on the
    /// main thread before `checkDisconnected` shows the kick. A kick that arrives while the
    /// bounded handoff is full must therefore still deliver every spilled packet, in order; the
    /// spill FIFO used to be dropped with the pump, losing them.
    #[tokio::test]
    async fn a_kick_behind_a_full_game_queue_still_delivers_every_spilled_packet() {
        let (mut handle, mut server, mut task) = rig();
        let burst_len = INBOUND_CAPACITY + 3;
        for index in 0..burst_len {
            server
                .send(&s2c::Chat {
                    component: index.to_string(),
                    position: 0,
                })
                .await
                .expect("send scheduled packet");
        }
        server
            .send(&s2c::Disconnect {
                reason: r#"{"text":"bye"}"#.into(),
            })
            .await
            .expect("send kick");

        // Nothing is drained yet, so the last three chats are certainly in the spill FIFO when
        // the kick ends the pump. The pump's end drops both client halves: the server reading
        // EOF is the exact moment, with no timing guess.
        let eof = tokio::time::timeout(Duration::from_secs(10), server.read_packet())
            .await
            .expect("the kick ends the pump");
        assert!(eof.is_err(), "the client sends nothing after the kick");

        let mut received = Vec::new();
        let reason = loop {
            while let Some(message) = handle.try_recv() {
                match message {
                    ToGame::Packet(S2CPlay::Chat(chat)) => received.push(chat.component),
                    other => panic!("unexpected {other:?}"),
                }
            }
            tokio::select! {
                result = &mut task => break result.expect("play task"),
                () = tokio::task::yield_now() => {}
            }
        };
        while let Some(message) = handle.try_recv() {
            if let ToGame::Packet(S2CPlay::Chat(chat)) = message {
                received.push(chat.component);
            }
        }
        assert_eq!(
            reason,
            DisconnectReason::Kicked {
                reason_json: r#"{"text":"bye"}"#.into()
            }
        );
        let expected: Vec<String> = (0..burst_len).map(|index| index.to_string()).collect();
        assert_eq!(
            received, expected,
            "every scheduled packet, in order, before the kick"
        );
    }

    /// A client-initiated close has nobody draining the game queue, so
    /// the pump must end even with a full channel and spilled packets left behind. A flush that
    /// awaited capacity here deadlocked `header_footer_and_resource_pack_bypass_a_full_scheduled_
    /// queue`, which disconnects after draining only part of a full queue.
    #[tokio::test]
    async fn a_client_close_never_waits_on_a_full_game_queue() {
        let (handle, mut server, task) = rig();
        for index in 0..INBOUND_CAPACITY + 3 {
            server
                .send(&s2c::Chat {
                    component: index.to_string(),
                    position: 0,
                })
                .await
                .expect("send scheduled packet");
        }
        // The KeepAlive echo is a barrier: every chat before it has been read (and the last
        // three spilled) once it comes back.
        server
            .send(&s2c::KeepAlive { id: 0x0c10 })
            .await
            .expect("send barrier");
        let (id, _) = server.read_packet().await.expect("barrier echo");
        assert_eq!(id, c2s::KeepAlive::ID);

        handle.disconnect().expect("disconnect");
        let reason = tokio::time::timeout(Duration::from_secs(10), task)
            .await
            .expect("a client close must not wait for the game loop to make room")
            .expect("play task");
        assert_eq!(reason, DisconnectReason::ClosedByClient);
    }

    /// Once a packet has spilled, a later one queues behind it even when
    /// the game loop has made room again. The read arm used to `try_send` unconditionally, so the
    /// newest packet took the freed slot ahead of every spilled one.
    #[test]
    fn a_packet_read_after_a_spill_never_overtakes_the_spilled_ones() {
        let keep_alive = |id: i32| ToGame::Packet(S2CPlay::KeepAlive(s2c::KeepAlive { id }));
        let id_of = |message: ToGame| match message {
            ToGame::Packet(S2CPlay::KeepAlive(keep_alive)) => keep_alive.id,
            other => panic!("unexpected {other:?}"),
        };
        let (to_game, mut game) = tokio::sync::mpsc::channel(1);
        let mut pending = VecDeque::new();
        enqueue_scheduled(&to_game, &mut pending, keep_alive(1)).expect("open");
        enqueue_scheduled(&to_game, &mut pending, keep_alive(2)).expect("open");
        assert_eq!(pending.len(), 1, "the second packet spills");

        // The game loop takes the first packet, so the channel has room again.
        assert_eq!(id_of(game.try_recv().expect("first")), 1);
        enqueue_scheduled(&to_game, &mut pending, keep_alive(3)).expect("open");
        assert!(
            game.try_recv().is_err(),
            "nothing may enter the channel ahead of a spill"
        );
        let order: Vec<i32> = pending.into_iter().map(id_of).collect();
        assert_eq!(order, [2, 3]);

        // A closed game loop is reported, not swallowed.
        drop(game);
        let mut empty = VecDeque::new();
        assert_eq!(
            enqueue_scheduled(&to_game, &mut empty, keep_alive(4)),
            Err(GameLoopGone)
        );
    }

    /// java: `MessageDeserializer` requires `readPacketData` to consume the complete framed
    /// body. A round-trip test cannot see this because its encoder never appends hostile bytes.
    #[test]
    fn a_play_packet_with_trailing_bytes_is_rejected() {
        // packet id 0 (S00 KeepAlive), key 1, then one byte that belongs to no field.
        let mut payload = PacketBuffer::from_slice(&[0x00, 0x01, 0x7f]);
        let err = decode(&mut payload).expect_err("trailing byte must be fatal");
        assert!(matches!(
            err,
            ClientError::Protocol(ProtoError::TrailingBytes {
                packet_id: 0,
                remaining: 1
            })
        ));
    }

    #[tokio::test]
    async fn set_compression_is_applied_before_the_next_frame_is_read() {
        let (mut handle, mut server, task) = rig();
        server
            .send(&s2c::SetCompression { threshold: 16 })
            .await
            .expect("send");
        server.set_compression_threshold(16);

        // Long enough to actually compress; if the threshold had been deferred to the game
        // thread this frame would deframe into garbage.
        let long = format!(r#"{{"text":"{}"}}"#, "y".repeat(400));
        server
            .send(&s2c::Chat {
                component: long.clone(),
                position: 0,
            })
            .await
            .expect("send");

        match expect_packet(&mut handle).await {
            S2CPlay::Chat(chat) => assert_eq!(chat.component, long),
            other => panic!("expected Chat, got {other:?}"),
        }
        handle.disconnect().expect("disconnect");
        let _ = task.await;
    }

    #[tokio::test]
    async fn positive_compression_reconfiguration_disconnects_like_vanilla() {
        let (_handle, mut server, task) = rig_with_compression(Some(16));
        server
            .send(&s2c::SetCompression { threshold: 32 })
            .await
            .expect("send reconfiguration under the old compression layer");

        let reason = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("vanilla's ClassCastException path must end the play loop")
            .expect("play task");
        assert_eq!(reason, compression_reconfiguration_reason());
    }

    /// The disconnect text a player of the shipped (obfuscated)
    /// client reads, and the Login-state S03 twin of the play-state S46 cast.
    #[test]
    fn compression_reconfiguration_names_the_obfuscated_classes_like_the_shipped_client() {
        assert_eq!(
            component_value(&compression_reconfiguration_reason()),
            serde_json::json!({
                "translate": "disconnect.genericReason",
                "with": [
                    "Internal Exception: java.lang.ClassCastException: ei cannot be cast to ej"
                ],
            })
        );
        assert_eq!(
            disconnect_reason_for(&ClientError::Protocol(ProtoError::CompressionReconfigured)),
            compression_reconfiguration_reason()
        );
    }

    #[tokio::test]
    async fn negative_compression_reconfiguration_still_disables_the_layer() {
        let (mut handle, mut server, task) = rig_with_compression(Some(16));
        server
            .send(&s2c::SetCompression { threshold: -1 })
            .await
            .expect("send disable under the old compression layer");
        server.set_compression_threshold(-1);
        server
            .send(&s2c::Chat {
                component: r#"{"text":"after disable"}"#.into(),
                position: 0,
            })
            .await
            .expect("send uncompressed packet after disable");

        match expect_packet(&mut handle).await {
            S2CPlay::Chat(chat) => assert_eq!(chat.component, r#"{"text":"after disable"}"#),
            other => panic!("expected Chat, got {other:?}"),
        }
        handle.disconnect().expect("disconnect");
        assert_eq!(task.await.unwrap(), DisconnectReason::ClosedByClient);
    }

    /// `S40PacketDisconnect` must end the loop with the server's reason JSON intact.
    ///
    /// [`play_loop`] *returns* the reason; publishing it as a [`ToGame::Disconnected`] is
    /// [`run`]'s job, and that half is covered end to end by `tests/play_loopback.rs`.
    #[tokio::test]
    async fn play_state_disconnect_surfaces_the_reason_json() {
        let (mut handle, mut server, task) = rig();
        let reason = r#"{"translate":"multiplayer.disconnect.idling"}"#;
        server
            .send(&s2c::Disconnect {
                reason: reason.to_owned(),
            })
            .await
            .expect("send");
        assert_eq!(
            task.await.unwrap(),
            DisconnectReason::Kicked {
                reason_json: reason.to_owned()
            }
        );
        // And the kick must not have been forwarded to the game loop as an ordinary packet.
        assert!(handle.try_recv().is_none());
    }

    #[tokio::test]
    async fn a_null_sign_line_reaches_game_without_closing_connection() {
        let (mut handle, mut server, task) = rig_with_compression(Some(256));
        // A real server sent this S33 shape during login. Vanilla's sign renderer skips the
        // null fourth component; rejecting it at the network boundary disconnects the player.
        let lines = [
            "\"Have fun\"".to_owned(),
            "\"at\"".to_owned(),
            "\"play.test  !\"".to_owned(),
            "null".to_owned(),
        ];
        server
            .send(&s2c::UpdateSign {
                position: Default::default(),
                lines: lines.clone(),
            })
            .await
            .expect("send sign");
        let packet = expect_packet(&mut handle).await;
        assert!(matches!(packet, S2CPlay::UpdateSign(sign) if sign.lines == lines));

        // A subsequent packet must still make the round trip after the nullable sign.
        server
            .send(&s2c::KeepAlive { id: 42 })
            .await
            .expect("send keepalive");
        let (id, mut body) = tokio::time::timeout(Duration::from_secs(1), server.read_packet())
            .await
            .expect("connection stays live")
            .expect("keepalive reply");
        assert_eq!(id, c2s::KeepAlive::ID);
        assert_eq!(c2s::KeepAlive::read(&mut body).unwrap().key, 42);
        handle.disconnect().expect("close test connection");
        assert_eq!(task.await.unwrap(), DisconnectReason::ClosedByClient);
    }

    #[tokio::test]
    async fn malformed_play_disconnect_uses_the_stable_decoder_exception_path() {
        let (_handle, mut server, task) = rig();
        server
            .send(&s2c::Disconnect { reason: "{".into() })
            .await
            .expect("send malformed component");

        let component = component_value(&task.await.unwrap());
        assert_eq!(component["translate"], "disconnect.genericReason");
        assert_eq!(
            component["with"][0].as_str().expect("generic detail"),
            "Internal Exception: com.google.gson.JsonSyntaxException"
        );
    }

    #[tokio::test]
    async fn an_unknown_packet_id_closes_the_connection_rather_than_being_skipped() {
        let (_handle, mut server, task) = rig();
        // 0x7F is not a legal clientbound Play id (the space is 0x00..=0x49).
        server.send_payload(&[0x7F, 0x00]).await.expect("send");
        let reason = task.await.unwrap();
        let component = component_value(&reason);
        assert_eq!(component["translate"], "disconnect.genericReason");
        assert_eq!(
            component["with"][0].as_str().expect("generic detail"),
            "Internal Exception: io.netty.handler.codec.DecoderException"
        );
    }

    #[tokio::test]
    async fn a_dropped_socket_reports_vanillas_end_of_stream_component() {
        let (_handle, server, task) = rig();
        drop(server);
        assert_eq!(
            component_value(&task.await.unwrap()),
            serde_json::json!({ "translate": "disconnect.endOfStream" })
        );
    }

    #[tokio::test]
    async fn outbound_packets_reach_the_wire_in_order() {
        let (handle, mut server, task) = rig();
        for key in 0..5 {
            handle
                .send(C2SPlay::KeepAlive(c2s::KeepAlive { key }))
                .expect("queue");
        }
        for key in 0..5 {
            let (id, mut body) = server.read_packet().await.expect("packet");
            assert_eq!(id, c2s::KeepAlive::ID);
            assert_eq!(c2s::KeepAlive::read(&mut body).unwrap().key, key);
        }
        handle.disconnect().expect("disconnect");
        let _ = task.await;
    }

    /// The replay recorder's tap: while it is
    /// on, every packet the game receives carries the exact bytes it was decoded from (packet id
    /// then body, after decompression), on the scheduled lane, the immediate lane and the chunk
    /// prebuild path; while it is off, packets arrive exactly as before.
    #[tokio::test]
    async fn the_raw_tap_carries_each_packets_exact_bytes_only_while_on() {
        let (a, b) = tokio::io::duplex(1 << 20);
        let (mut handle, to_game, immediate_to_game, from_game) = NetHandle::new_live_pair();
        let (activity_tx, activity_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut client = Connection::new(ReadActivity::new(a, activity_tx));
        let mut server = Connection::new(b);
        // Compressed, so the bytes are the ones after decompression.
        client.set_compression_threshold(64);
        server.set_compression_threshold(64);
        let (reader, writer) = client.split();
        let mut capture = CaptureSink::disabled();
        capture.set_raw_tap(handle.raw_tap_flag());
        let task = tokio::spawn(async move {
            play_loop(
                reader,
                writer,
                &to_game,
                &immediate_to_game,
                from_game,
                activity_rx,
                READ_TIMEOUT,
                &mut capture,
            )
            .await
        });
        async fn next(handle: &mut NetHandle) -> ToGame {
            for _ in 0..10_000 {
                if let Some(message) = handle.try_recv() {
                    return message;
                }
                tokio::task::yield_now().await;
            }
            panic!("no message arrived");
        }
        fn bytes(packet: &S2CPlay) -> Vec<u8> {
            let mut buf = PacketBuffer::new();
            buf.write_varint(packet.id());
            packet.write(&mut buf);
            buf.as_slice().to_vec()
        }
        let chat = |text: &str| s2c::Chat {
            component: format!(r#"{{"text":"{text}"}}"#),
            position: 0,
        };

        // Off: a plain packet, nothing copied.
        server.send(&chat("before")).await.expect("send");
        assert!(matches!(
            next(&mut handle).await,
            ToGame::Packet(S2CPlay::Chat(_))
        ));

        handle.set_raw_tap(true);
        let header = s2c::PlayerListHeaderFooter {
            header: r#"{"text":"Aero"}"#.into(),
            footer: r#"{"text":"tap"}"#.into(),
        };
        let mask = 0b1u16;
        let chunk = mc_protocol::chunk::ChunkData {
            chunk_x: 3,
            chunk_z: -2,
            ground_up_continuous: true,
            primary_bit_mask: mask,
            data: (0..mc_world::expected_chunk_data_len(mask, true, true))
                .map(|i| (i % 7) as u8)
                .collect(),
        };
        let long_chat = chat(&"x".repeat(300));
        server.send(&long_chat).await.expect("send");
        server.send(&header).await.expect("send");
        server.send(&chunk).await.expect("send");
        let expected = [
            bytes(&S2CPlay::Chat(long_chat)),
            bytes(&S2CPlay::PlayerListHeaderFooter(header)),
            bytes(&S2CPlay::ChunkData(chunk)),
        ];
        let mut got = Vec::new();
        let mut chunk_path = false;
        while got.len() < expected.len() {
            match next(&mut handle).await {
                ToGame::RawPacket(tapped) => {
                    let (packet, raw) = *tapped;
                    assert_eq!(bytes(&packet), &raw[..], "the bytes are the packet's own");
                    got.push(raw.to_vec());
                }
                ToGame::RawChunkPacket(job, raw) => {
                    chunk_path = true;
                    let (packet, _) = job.take().expect("the chunk packet");
                    assert_eq!(bytes(&packet), &raw[..]);
                    got.push(raw.to_vec());
                }
                other => panic!("expected a tapped packet, got {other:?}"),
            }
        }
        // The immediate lane may overtake the scheduled one; compare as sets of exact bytes.
        let mut sorted_got = got.clone();
        sorted_got.sort();
        let mut sorted_expected = expected.to_vec();
        sorted_expected.sort();
        assert_eq!(sorted_got, sorted_expected);
        assert!(
            chunk_path,
            "a full chunk takes the prebuild path and keeps its bytes"
        );

        // Off again: plain packets.
        handle.set_raw_tap(false);
        server.send(&chat("after")).await.expect("send");
        assert!(matches!(
            next(&mut handle).await,
            ToGame::Packet(S2CPlay::Chat(_))
        ));
        handle.disconnect().expect("disconnect");
        let _ = task.await;
    }
}
