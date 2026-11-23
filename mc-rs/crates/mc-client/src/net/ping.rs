//! Server-list ping: the status protocol, its JSON schema, and the favicon.
//!
//! java: `client/network/OldServerPinger.java` (the ping itself and how each JSON field is
//! consumed), `network/ServerStatusResponse.java` (the schema and its Gson serialisers),
//! `client/multiplayer/ServerData.java` (where the results are stored).
//!
//! `mc_protocol::connection::ping_server` already does the handshake/`Request`/`Response`/
//! `Ping`/`Pong` exchange and hands back the raw JSON plus a measured round trip. Everything in
//! this module is the layer above that: turning the JSON into something the server-list screen
//! can draw.

use std::time::{Duration, Instant};
use std::{future::Future, net::SocketAddr, process::Command};

use base64::Engine as _;
use mc_ui::font::component::ChatComponent;
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::error::ClientError;

/// The `data:` prefix vanilla insists on before a favicon's base64 payload.
/// java: `OldServerPinger` — anything else logs "Invalid server icon (unknown format)".
const FAVICON_PREFIX: &str = "data:image/png;base64,";

/// Vanilla only accepts a 64×64 icon. java: `ServerListEntryNormal` checks both dimensions and
/// silently drops anything else.
pub const FAVICON_SIZE: u32 = 64;

/// Minecraft's default server port.
pub const DEFAULT_SERVER_PORT: i32 = 25_565;

/// Wall-clock budget for one saved-row ping, including address resolution, the modern status
/// exchange, and the legacy compatibility fallback. Java's status channel has a thirty-second
/// idle `ReadTimeoutHandler`; the outer budget also prevents a resolver/connect attempt or a
/// fallback socket from surviving forever after the row has stopped being visible.
pub const PING_TIMEOUT: Duration = Duration::from_secs(30);

/// The result of Java 1.8.9's `ServerAddress.fromString`, before its optional DNS SRV lookup.
///
/// `port` deliberately remains an `i32`: Java accepts syntactically valid values such as `-1`
/// and `70000` here, then `InetSocketAddress` rejects them. Narrowing during parsing would instead
/// turn them into the default port and connect to a server the player did not request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedServerAddress {
    /// Host text after Java's bracket and colon rules, before IDN ASCII conversion or SRV.
    pub host: String,
    /// Parsed Java `int` port, including values outside the socket port range.
    pub port: i32,
    /// Whether vanilla calls `_minecraft._tcp.<host>` lookup for this result.
    ///
    /// This is true whenever the parsed/fallback port is exactly 25565, including an explicitly
    /// typed `:25565` and a malformed port string that fell back to the default.
    pub srv_eligible: bool,
}

impl ParsedServerAddress {
    /// Return the port a Rust socket API can accept, preserving Java's later failure for values
    /// outside `0..=65535`.
    ///
    /// The error is the original Java-width value, suitable for a connection-failure message.
    pub fn socket_port(&self) -> Result<u16, i32> {
        u16::try_from(self.port).map_err(|_| self.port)
    }
}

/// The only structural failure in Java's parser: an input made solely of colons is reduced by
/// `String.split(":")` to an empty array and the subsequent `astring[0]` access throws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerAddressParseError;

impl std::fmt::Display for ServerAddressParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Java ServerAddress split produced no host slot")
    }
}

impl std::error::Error for ServerAddressParseError {}

/// Parse a saved/direct-connect address exactly like Java 1.8.9's
/// `ServerAddress.fromString`, stopping immediately before `getServerAddress` performs DNS SRV.
///
/// The caller must perform SRV lookup when [`ParsedServerAddress::srv_eligible`] is true, then
/// apply IDN-to-ASCII to the chosen host. If SRV is absent/fails, use `host` and `socket_port()`.
/// A failed `socket_port()` is a connection failure, not permission to substitute 25565.
///
/// Java accepts `null` by returning `null`; Rust callers already hold a saved `String`, so this
/// API models the non-null branch only.
pub fn parse_server_address(address: &str) -> Result<ParsedServerAddress, ServerAddressParseError> {
    // Java String.split(regex) drops *all* trailing empty elements, except that splitting the
    // empty string itself returns one empty element. This distinction makes `""` a blank host
    // but `":"` the parser's otherwise-surprising structural exception.
    let mut parts: Vec<&str> = if address.is_empty() {
        vec![""]
    } else {
        let mut split: Vec<&str> = address.split(':').collect();
        while split.last() == Some(&"") {
            split.pop();
        }
        split
    };

    if let Some(rest) = address.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            let host = &rest[..end];
            let suffix = java_trim(&rest[end + 1..]);
            parts = if let Some(port) = suffix.strip_prefix(':') {
                vec![host, port]
            } else {
                vec![host]
            };
        }
    }

    if parts.len() > 2 {
        parts = vec![address];
    }
    let host = parts.first().ok_or(ServerAddressParseError)?.to_string();
    let port = parts
        .get(1)
        .and_then(|value| java_parse_i32(value))
        .unwrap_or(DEFAULT_SERVER_PORT);
    Ok(ParsedServerAddress {
        host,
        port,
        srv_eligible: port == DEFAULT_SERVER_PORT,
    })
}

/// Java 8 `String.trim`: remove code units no greater than U+0020, not all Unicode whitespace.
fn java_trim(value: &str) -> &str {
    value.trim_matches(|character| character <= '\u{20}')
}

/// Java 8's `Integer.parseInt(value.trim())` for radix 10.
///
/// `Integer.parseInt` delegates each UTF-16 code unit to `Character.digit`, so decimal digits are
/// not limited to ASCII (`host:１２３` and `host:١٢٣` are both valid), while a supplementary-plane
/// digit is rejected because its surrogate halves are not digits. Keeping the accumulator signed
/// in spirit (with an explicit positive magnitude bound) also preserves Java's distinction between
/// `Integer.MIN_VALUE` and an overflowing value that falls back to 25565 in `ServerAddress`.
fn java_parse_i32(value: &str) -> Option<i32> {
    let trimmed = java_trim(value);
    let mut units = trimmed.encode_utf16();
    let negative = match units.next()? {
        45 => true,
        43 => false,
        _ => {
            // The first code unit was a digit candidate. Re-run the iterator from the beginning;
            // this keeps the sign handling explicit without treating a non-ASCII sign specially.
            units = trimmed.encode_utf16();
            false
        }
    };
    let mut magnitude = 0_i64;
    let mut saw_digit = false;
    for unit in units {
        let digit = java_character_digit(unit)?;
        saw_digit = true;
        magnitude = magnitude.checked_mul(10)?.checked_add(i64::from(digit))?;
        let limit = i64::from(i32::MAX) + if negative { 1 } else { 0 };
        if magnitude > limit {
            return None;
        }
    }
    if !saw_digit {
        return None;
    }
    if negative {
        if magnitude == i64::from(i32::MAX) + 1 {
            Some(i32::MIN)
        } else {
            Some(-(magnitude as i32))
        }
    } else {
        Some(magnitude as i32)
    }
}

/// The BMP decimal-digit ranges accepted by Java 8's `Character.digit(char, 10)`.
///
/// `Integer.parseInt` walks UTF-16 `char`s, so this deliberately does not include supplementary
/// Unicode digit blocks: their surrogate halves are what `parseInt` sees and both return `-1`.
fn java_character_digit(unit: u16) -> Option<u32> {
    const RANGES: &[(u16, u16)] = &[
        (0x0030, 0x0039),
        (0x0660, 0x0669),
        (0x06f0, 0x06f9),
        (0x07c0, 0x07c9),
        (0x0966, 0x096f),
        (0x09e6, 0x09ef),
        (0x0a66, 0x0a6f),
        (0x0ae6, 0x0aef),
        (0x0b66, 0x0b6f),
        (0x0be6, 0x0bef),
        (0x0c66, 0x0c6f),
        (0x0ce6, 0x0cef),
        (0x0d66, 0x0d6f),
        (0x0e50, 0x0e59),
        (0x0ed0, 0x0ed9),
        (0x0f20, 0x0f29),
        (0x1040, 0x1049),
        (0x1090, 0x1099),
        (0x17e0, 0x17e9),
        (0x1810, 0x1819),
        (0x1946, 0x194f),
        (0x19d0, 0x19d9),
        (0x1a80, 0x1a89),
        (0x1a90, 0x1a99),
        (0x1b50, 0x1b59),
        (0x1bb0, 0x1bb9),
        (0x1c40, 0x1c49),
        (0x1c50, 0x1c59),
        (0xa620, 0xa629),
        (0xa8d0, 0xa8d9),
        (0xa900, 0xa909),
        (0xa9d0, 0xa9d9),
        (0xaa50, 0xaa59),
        (0xabf0, 0xabf9),
        (0xff10, 0xff19),
    ];
    RANGES.iter().find_map(|&(start, end)| {
        if (start..=end).contains(&unit) {
            Some(u32::from(unit - start))
        } else {
            None
        }
    })
}

/// A decoded server icon, ready to upload as a texture.
#[derive(Clone, PartialEq, Eq)]
pub struct Favicon {
    /// Always [`FAVICON_SIZE`].
    pub width: u32,
    /// Always [`FAVICON_SIZE`].
    pub height: u32,
    /// Straight `RGBA8` rows, top to bottom: `width * height * 4` bytes.
    pub rgba: Vec<u8>,
}

impl std::fmt::Debug for Favicon {
    /// 16 KiB of pixels has no business in a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Favicon")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.rgba.len())
            .finish()
    }
}

/// One entry of the status response's player sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerSample {
    /// The player's name.
    pub name: String,
    /// The player's UUID, as the server rendered it. Vanilla validates this with Java 8's
    /// unusually-lenient `UUID.fromString` before retaining the profile.
    pub id: String,
}

/// Everything the server-list screen learns from one ping.
/// java: the fields `OldServerPinger` writes into `ServerData`.
#[derive(Debug, Clone)]
pub struct ServerPing {
    /// The MOTD, as a parsed chat component. java: `ServerData.serverMOTD`, which vanilla
    /// stores pre-flattened; keeping the tree lets S16 render colours properly.
    pub motd: ChatComponent,
    /// The version name the server advertises (`"1.8.9"`, `"Lobby 1.8.9"`, `"Requires MC 1.8"`…).
    /// java: `ServerData.gameVersion`, defaulted to `"Old"` when the object is missing.
    pub version_name: String,
    /// The protocol number. java: `ServerData.version`, defaulted to `0` when missing.
    pub protocol: i32,
    /// Players currently online, or `None` if the server omitted the `players` object —
    /// vanilla renders that case as `???`.
    pub online_players: Option<i32>,
    /// The advertised player cap.
    pub max_players: Option<i32>,
    /// The (usually truncated, usually shuffled) player sample.
    pub sample: Vec<PlayerSample>,
    /// The decoded icon, if the server sent a valid one.
    pub favicon: Option<Favicon>,
    /// The favicon payload without the `data:image/png;base64,` prefix, when that prefix was
    /// present. Keeping the original payload lets the UI persist exactly what vanilla stores,
    /// while [`Self::favicon`] carries the already-validated RGBA pixels.
    pub favicon_base64: Option<String>,
    /// Whether the status JSON contained a primitive `favicon` field. A present value with an
    /// unsupported prefix must keep the previous icon; an absent field clears it.
    pub favicon_present: bool,
    /// Round-trip time measured across the `Ping`/`Pong` exchange.
    /// java: `ServerData.pingToServer`
    pub latency: Duration,
    /// Transport-side send instant retained until the client thread receives this result.
    ///
    /// [`Self::sample_latency_on_client_tick`] is deliberately its only consumer:
    /// `GuiMultiplayer.updateScreen` processes the queued status Pong once per 20 Hz game tick,
    /// so the visible number is receipt-time latency rather than Tokio's decode-time round trip.
    ping_sent_at: Option<Instant>,
    /// The response verbatim, for the F3 screen and for debugging odd servers.
    pub raw_json: String,
}

impl ServerPing {
    /// Vanilla's `"<online>/<max>"` population string, with the `§7`/`§8` colouring it applies.
    /// java: `OldServerPinger.handleServerInfo` builds exactly this, and `"§8???"` when the
    /// `players` object is absent.
    #[must_use]
    pub fn population_info(&self) -> String {
        match (self.online_players, self.max_players) {
            (Some(online), Some(max)) => format!("\u{a7}7{online}\u{a7}8/\u{a7}7{max}"),
            _ => "\u{a7}8???".to_owned(),
        }
    }

    /// Commit the displayed latency at the client-thread receipt boundary.
    ///
    /// java: `GuiMultiplayer.updateScreen` -> `OldServerPinger.pingPendingNetworks` ->
    /// `handlePong`. This is intentionally *not* a synthetic 50 ms rounding rule: an actual
    /// delayed client tick contributes its real delay, just as vanilla does. Calling it more
    /// than once is harmless; only the first receipt consumes the pending sample.
    pub fn sample_latency_on_client_tick(&mut self, tick_now: Instant) {
        if let Some(sent_at) = self.ping_sent_at.take() {
            self.latency = tick_now.saturating_duration_since(sent_at);
        }
    }
}

/// Ping `host:port` and parse the result.
///
/// # Errors
///
/// [`ClientError::Protocol`] for any connect/handshake/status failure, or
/// [`ClientError::StatusJson`] if the response is not a JSON object. An unusable *favicon* is
/// not an error — it is dropped with a warning, matching vanilla.
pub async fn ping(host: &str, port: u16) -> Result<ServerPing, ClientError> {
    let lookup = match tokio::net::lookup_host((host, port)).await {
        Ok(addresses) => Ok(super::first_ipv4_address(addresses)),
        Err(error) => Err(error),
    };
    ping_with_lookup_result(host, port, lookup).await
}

async fn ping_with_lookup_result(
    host: &str,
    port: u16,
    lookup: std::io::Result<Option<SocketAddr>>,
) -> Result<ServerPing, ClientError> {
    ping_with_lookup_result_timeout(host, port, lookup, None).await
}

async fn ping_with_lookup_result_timeout(
    host: &str,
    port: u16,
    lookup: std::io::Result<Option<SocketAddr>>,
    idle_timeout: Option<Duration>,
) -> Result<ServerPing, ClientError> {
    let resolved = match lookup {
        Ok(Some(resolved)) => resolved,
        Ok(None) => return Err(unknown_host_error(host, None)),
        Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {
            return Err(ClientError::Io(error));
        }
        Err(error) => return Err(unknown_host_error(host, Some(&error))),
    };

    let modern = async {
        // Java resolves with InetAddress.getByName once under preferIPv4Stack, connects that
        // first IPv4 address, and keeps ServerAddress.getIP() as the C00Handshake host. Keeping
        // those values separate also prevents a second DNS failure from escaping as ProtoError::Io.
        let mut conn = match idle_timeout {
            Some(timeout_budget) => tokio::time::timeout(
                timeout_budget,
                mc_protocol::connection::Connection::connect(resolved),
            )
            .await
            .map_err(|_| ping_timeout_error(timeout_budget))??,
            None => mc_protocol::connection::Connection::connect(resolved).await?,
        };
        let handshake = mc_protocol::connection::handshake(
            &mut conn,
            host,
            port,
            mc_protocol::handshake::NextState::Status,
        );
        match idle_timeout {
            Some(timeout_budget) => tokio::time::timeout(timeout_budget, handshake)
                .await
                .map_err(|_| ping_timeout_error(timeout_budget))??,
            None => handshake.await?,
        }
        // `status_with_read_timeout` owns an idle (not total) deadline. A server that keeps
        // fragmenting a frame remains valid, while the post-Response Pong leg still retains its
        // committed JSON when its own idle window expires.
        let status = match idle_timeout {
            Some(timeout_budget) => {
                mc_protocol::connection::status_with_read_timeout(&mut conn, timeout_budget).await?
            }
            None => mc_protocol::connection::status(&mut conn).await?,
        };
        parse_status_with_ping_sent_at(&status.json, status.latency, status.ping_sent_at)
    }
    .await;

    match modern {
        Ok(ping) => Ok(ping),
        Err(modern_error) => {
            let legacy = legacy_ping(host, port);
            let legacy_result = match idle_timeout {
                Some(timeout_budget) => tokio::time::timeout(timeout_budget, legacy)
                    .await
                    .unwrap_or_else(|_| Err(ping_timeout_error(timeout_budget))),
                None => legacy.await,
            };
            match legacy_result {
                Ok(ping) => Ok(ping),
                Err(_) => Err(modern_error),
            }
        }
    }
}

fn ping_timeout_error(timeout_budget: Duration) -> ClientError {
    ClientError::Io(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        format!("server ping timed out after {timeout_budget:?}"),
    ))
}

fn unknown_host_error(host: &str, source: Option<&std::io::Error>) -> ClientError {
    let detail = source.map_or_else(
        || format!("no addresses were returned for {host}"),
        |error| format!("cannot resolve {host}: {error}"),
    );
    ClientError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, detail))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LegacyStatus {
    version_name: String,
    motd: String,
    online: i32,
    max: i32,
}

/// Vanilla's pre-1.7 `tryCompatibilityPing` fallback. It runs only after the modern exchange
/// failed before committing a status response; a failed fallback leaves the original modern
/// error as the row's failure reason.
async fn legacy_ping(host: &str, port: u16) -> Result<ServerPing, ClientError> {
    let mut stream = TcpStream::connect((host, port)).await?;
    // Java catches and ignores platforms that reject TCP_NODELAY.
    let _ = stream.set_nodelay(true);
    legacy_ping_stream(&mut stream, host, port).await
}

async fn legacy_ping_stream<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    host: &str,
    port: u16,
) -> Result<ServerPing, ClientError> {
    stream.write_all(&legacy_request(host, port)).await?;
    stream.flush().await?;

    if stream.read_u8().await? != 0xFF {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "legacy ping response did not start with 0xFF",
        )));
    }
    let utf16_units = stream.read_i16().await?;
    if utf16_units < 0 {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "legacy ping response has a negative string length",
        )));
    }
    let mut encoded = vec![0u8; utf16_units as usize * 2];
    stream.read_exact(&mut encoded).await?;
    let units: Vec<u16> = encoded
        .chunks_exact(2)
        .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
        .collect();
    let response = String::from_utf16_lossy(&units);
    let legacy = parse_legacy_response(&response).ok_or_else(|| {
        ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "legacy ping response is not the six-field 1.6 format",
        ))
    })?;

    // `raw_json` retains a useful diagnostic/status snapshot for the legacy result. The protocol
    // number is deliberately -1: OldServerPinger ignores the legacy response's protocol field
    // and assigns `ServerData.version = -1` unconditionally.
    let raw_json = serde_json::json!({
        "version": { "name": legacy.version_name.clone(), "protocol": -1 },
        "players": { "online": legacy.online, "max": legacy.max },
        "description": legacy.motd.clone(),
    })
    .to_string();
    Ok(ServerPing {
        motd: ChatComponent::text(legacy.motd),
        version_name: legacy.version_name,
        protocol: -1,
        online_players: Some(legacy.online),
        max_players: Some(legacy.max),
        sample: Vec::new(),
        favicon: None,
        favicon_base64: None,
        favicon_present: false,
        latency: Duration::MAX,
        ping_sent_at: None,
        raw_json,
    })
}

fn legacy_request(host: &str, port: u16) -> Vec<u8> {
    let channel: Vec<u16> = "MC|PingHost".encode_utf16().collect();
    let host: Vec<u16> = host.encode_utf16().collect();
    let mut out = Vec::with_capacity(3 + 2 + channel.len() * 2 + 3 + host.len() * 2 + 4);
    out.extend_from_slice(&[0xFE, 0x01, 0xFA]);
    write_legacy_utf16(&mut out, &channel);
    out.extend_from_slice(&(7usize.wrapping_add(host.len() * 2) as u16).to_be_bytes());
    out.push(127);
    write_legacy_utf16(&mut out, &host);
    out.extend_from_slice(&i32::from(port).to_be_bytes());
    out
}

fn write_legacy_utf16(out: &mut Vec<u8>, units: &[u16]) {
    out.extend_from_slice(&(units.len() as u16).to_be_bytes());
    for unit in units {
        out.extend_from_slice(&unit.to_be_bytes());
    }
}

fn parse_legacy_response(response: &str) -> Option<LegacyStatus> {
    let mut fields = response.splitn(6, '\0');
    if fields.next()? != "\u{a7}1" {
        return None;
    }
    let _reported_protocol = fields.next()?.parse::<i32>().unwrap_or(0);
    let version_name = fields.next()?.to_owned();
    let motd = fields.next()?.to_owned();
    let online = fields.next()?.parse::<i32>().unwrap_or(-1);
    let max = fields.next()?.parse::<i32>().unwrap_or(-1);
    Some(LegacyStatus {
        version_name,
        motd,
        online,
        max,
    })
}

/// Parse, optionally SRV-resolve, and ping one saved/direct-connect address.
///
/// This is the live server-list entry point corresponding to Java's
/// `OldServerPinger.ping(ServerData)`: it uses the same [`parse_server_address`] result as the
/// connect path should, attempts `_minecraft._tcp` SRV only when vanilla does, and preserves an
/// out-of-range signed port as a failed ping rather than silently substituting 25565.
///
/// # Errors
///
/// Returns [`ClientError::Io`] for Java's colon-only parser exception or a port outside the socket
/// range. SRV lookup failure is not an error; Java catches it and pings the original endpoint.
/// Status-protocol and JSON errors are forwarded from [`ping`].
pub async fn ping_address(address: &str) -> Result<ServerPing, ClientError> {
    let (host, port) = resolve_server_address(address).await?;
    ping(&host, port).await
}

/// Run one saved-row ping under its client-side lifetime budget.
///
/// The protocol layer deliberately retains a decoded status response when only its follow-up
/// Pong fails. This wrapper bounds resolution/connect and the compatibility retry while leaving
/// the modern status exchange's read deadline idle-based, so a fragmented response or a retained
/// post-Response row is not cut off by a fixed total wall-clock timer. A row Refresh or screen
/// close can still abort the whole task instead of leaving a detached read behind.
pub(crate) async fn ping_with_timeout(
    address: &str,
    timeout_budget: Duration,
) -> Result<ServerPing, ClientError> {
    let (host, port) =
        match tokio::time::timeout(timeout_budget, resolve_server_address(address)).await {
            Ok(result) => result?,
            Err(_) => return Err(ping_timeout_error(timeout_budget)),
        };
    let lookup = match tokio::time::timeout(
        timeout_budget,
        tokio::net::lookup_host((host.as_str(), port)),
    )
    .await
    {
        Ok(Ok(addresses)) => Ok(super::first_ipv4_address(addresses)),
        Ok(Err(error)) => Err(error),
        Err(_) => return Err(ping_timeout_error(timeout_budget)),
    };
    ping_with_lookup_result_timeout(&host, port, lookup, Some(timeout_budget)).await
}

/// Parse and asynchronously resolve one saved/direct-connect server address exactly as Java's
/// `ServerAddress.fromString` + `getServerAddress` path does.
///
/// The returned host is the SRV target when an eligible lookup succeeds, otherwise the parsed
/// original host. An SRV failure is deliberately not an error. A structurally invalid address or
/// a signed Java port outside the socket's `u16` range is an error and must stop the connection.
///
/// # Errors
///
/// [`ClientError::Io`] with [`std::io::ErrorKind::InvalidInput`] for Java's colon-only parser
/// exception or a port outside `0..=65535`.
pub async fn resolve_server_address(address: &str) -> Result<(String, u16), ClientError> {
    resolve_server_address_with(
        address,
        |host| async move { resolve_minecraft_srv(&host).await },
    )
    .await
}

#[cfg(test)]
async fn ping_address_with<R, RFut, P, PFut>(
    address: &str,
    mut resolver: R,
    mut pinger: P,
) -> Result<ServerPing, ClientError>
where
    R: FnMut(String) -> RFut,
    RFut: Future<Output = Option<(String, u16)>>,
    P: FnMut(String, u16) -> PFut,
    PFut: Future<Output = Result<ServerPing, ClientError>>,
{
    let (host, socket_port) = resolve_server_address_with(address, &mut resolver).await?;
    pinger(host, socket_port).await
}

async fn resolve_server_address_with<R, RFut>(
    address: &str,
    mut resolver: R,
) -> Result<(String, u16), ClientError>
where
    R: FnMut(String) -> RFut,
    RFut: Future<Output = Option<(String, u16)>>,
{
    let parsed = parse_server_address(address).map_err(|error| {
        ClientError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
    })?;
    let mut host = parsed.host;
    let mut port = parsed.port;
    if parsed.srv_eligible {
        if let Some((srv_host, srv_port)) = resolver(host.clone()).await {
            host = srv_host;
            port = i32::from(srv_port);
        }
    }
    // java: `ServerAddress.fromString` performs SRV first; `getIP()` converts whichever host won.
    // This ordering matters for Unicode SRV targets and preserves a target's trailing root dot.
    host = mc_ui::screens::multiplayer::server_data::java_idn_to_ascii(&host).map_err(|error| {
        ClientError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
    })?;
    let socket_port = u16::try_from(port).map_err(|_| {
        ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("server port {port} is outside 0..=65535"),
        ))
    })?;
    Ok((host, socket_port))
}

/// Resolve Minecraft's SRV record through the operating system's DNS client.
///
/// Java 1.8.9 catches every JNDI/DNS failure and keeps the original host and port. Spawning the
/// platform `nslookup` utility provides the same system-resolver boundary without adding a second
/// DNS stack or hard-coding a public resolver. Missing utility, timeout/process failure, NXDOMAIN,
/// and an unparseable response therefore all return `None` and take Java's fallback path.
async fn resolve_minecraft_srv(host: &str) -> Option<(String, u16)> {
    let query = format!("_minecraft._tcp.{host}");
    let lookup = tokio::task::spawn_blocking(move || {
        let output = Command::new("nslookup")
            .arg("-type=SRV")
            .arg(&query)
            .output()
            .ok()?;
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        parse_nslookup_srv(&text)
    });
    tokio::time::timeout(Duration::from_secs(3), lookup)
        .await
        .ok()?
        .ok()
        .flatten()
}

fn parse_nslookup_srv(output: &str) -> Option<(String, u16)> {
    let lines: Vec<&str> = output.lines().collect();

    // BIND-style nslookup prints the RFC record on one line:
    // `_minecraft._tcp.example service = 0 5 25570 target.example.`
    for line in &lines {
        let lower = line.to_ascii_lowercase();
        if let Some(index) = lower.find("service =") {
            let values: Vec<&str> = line[index + "service =".len()..]
                .split_whitespace()
                .collect();
            if values.len() >= 4 {
                let port = values[2].parse::<u16>().ok()?;
                return Some((values[3].to_owned(), port));
            }
        }
    }

    // Windows nslookup prints one labelled value per line. Use value order rather than localized
    // label text: priority, weight, port, target.
    let mut numeric_values = Vec::with_capacity(3);
    for line in &lines {
        let Some((_, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        if numeric_values.len() < 3 {
            if let Ok(number) = value.parse::<u16>() {
                numeric_values.push(number);
                continue;
            }
        }
        if numeric_values.len() == 3 && !value.is_empty() {
            return Some((value.to_owned(), numeric_values[2]));
        }
    }
    None
}

/// Parse a status response body.
///
/// Split out from [`ping`] so the schema is testable without a socket, and so S16 can re-parse a
/// cached response.
///
/// # Errors
///
/// [`ClientError::StatusJson`] if the body is not a JSON object, if a present nested object lacks
/// one of Gson's required fields, or if `description` is not a component `mc-ui` can parse.
pub fn parse_status(json: &str, latency: Duration) -> Result<ServerPing, ClientError> {
    parse_status_with_ping_sent_at(json, latency, None)
}

/// Constructs a transport-status result with a retained Pong send instant for the client-driver
/// receipt-boundary regression.  Production callers deliberately cannot manufacture this
/// transport state: only the socket status path supplies it.
#[cfg(test)]
pub(crate) fn parse_status_for_client_tick_test(
    json: &str,
    latency: Duration,
    ping_sent_at: Instant,
) -> Result<ServerPing, ClientError> {
    parse_status_with_ping_sent_at(json, latency, Some(ping_sent_at))
}

/// The live status path retains its Ping send instant until the game thread receives it. Pure
/// JSON callers deliberately use [`parse_status`] and cannot manufacture that transport state.
fn parse_status_with_ping_sent_at(
    json: &str,
    latency: Duration,
    ping_sent_at: Option<Instant>,
) -> Result<ServerPing, ClientError> {
    // java: `S00PacketServerInfo` reads with `Gson.fromJson`, which is
    // always lenient. A strict parse listed a server whose status has, say, a raw newline in
    // its MOTD or a comment as unreachable, while vanilla lists it. Valid JSON parses as before.
    let (root, relaxed) = match serde_json::from_str::<Value>(json) {
        Ok(root) => (root, false),
        Err(strict) => match mc_ui::font::component::gson_json_value(json) {
            Ok(root) => (root, true),
            Err(_) => return Err(ClientError::StatusJson(strict.to_string())),
        },
    };
    // The lenient rewrite writes every bare token as a string (all a chat component needs),
    // but Gson's `peekNumber` reads a bare number as a number, which `JsonUtils.getInt`
    // requires. So in a relaxed document, a string holding exactly a JSON number counts.
    let gson_i32 = |value: &Value| {
        gson_i32(value).or_else(|| {
            let text = value.as_str().filter(|_| relaxed)?;
            serde_json::from_str::<serde_json::Number>(text)
                .ok()
                .and_then(|number| gson_i32(&Value::Number(number)))
        })
    };
    let object = root
        .as_object()
        .ok_or_else(|| ClientError::StatusJson("the response is not a JSON object".to_owned()))?;

    // java: ServerStatusResponse.Serializer.deserialize — every field is optional, and vanilla
    // leaves absent top-level fields null. A present nested object remains strict: failures from
    // its adapter close the status connection instead of producing a partially populated row.
    let motd = match object.get("description") {
        Some(Value::Null) => ChatComponent::text(""),
        Some(Value::Array(items)) if items.is_empty() => ChatComponent::text(""),
        Some(value) => ChatComponent::from_value(value, 0)?,
        None => ChatComponent::text(""),
    };

    let (version_name, protocol) = match object.get("version") {
        // java: "Old" / 0 when the object is absent — a pre-1.7 server that answered the legacy
        // ping, which is the only way this happens in practice. A JSON
        // null member is the same: `context.deserialize` of it is null (`TreeTypeAdapter`).
        None | Some(Value::Null) => ("Old".to_owned(), 0),
        Some(value) => {
            let version = value.as_object().ok_or_else(|| {
                ClientError::StatusJson("version is not a JSON object".to_owned())
            })?;
            let name = version.get("name").and_then(gson_string).ok_or_else(|| {
                ClientError::StatusJson("version.name is not primitive".to_owned())
            })?;
            let protocol = version.get("protocol").and_then(gson_i32).ok_or_else(|| {
                ClientError::StatusJson("version.protocol is not a number".to_owned())
            })?;
            (name, protocol)
        }
    };

    let (online_players, max_players, sample) = match object.get("players") {
        // A JSON null member is an absent one, as for `version`.
        None | Some(Value::Null) => (None, None, Vec::new()),
        Some(value) => {
            let players = value.as_object().ok_or_else(|| {
                ClientError::StatusJson("players is not a JSON object".to_owned())
            })?;
            let online = players.get("online").and_then(gson_i32).ok_or_else(|| {
                ClientError::StatusJson("players.online is not a number".to_owned())
            })?;
            let max = players
                .get("max")
                .and_then(gson_i32)
                .ok_or_else(|| ClientError::StatusJson("players.max is not a number".to_owned()))?;
            let mut sample = Vec::new();
            // JsonUtils.isJsonArray ignores a wrong-typed optional `sample`, but once it is an
            // array every entry and both profile fields are required.
            if let Some(entries) = players.get("sample").and_then(Value::as_array) {
                sample.reserve(entries.len());
                for value in entries {
                    let entry = value.as_object().ok_or_else(|| {
                        ClientError::StatusJson(
                            "a players.sample entry is not an object".to_owned(),
                        )
                    })?;
                    let id = entry.get("id").and_then(gson_string).ok_or_else(|| {
                        ClientError::StatusJson("players.sample[].id is not primitive".to_owned())
                    })?;
                    // `UUID.fromString` exactly as Java 8 runs it
                    // (lenient part widths, no 36-character cap): the shared login parser.
                    if mc_auth::java_uuid_from_string(&id).is_none() {
                        return Err(ClientError::StatusJson(
                            "players.sample[].id is not a Java UUID".to_owned(),
                        ));
                    }
                    let name = entry.get("name").and_then(gson_string).ok_or_else(|| {
                        ClientError::StatusJson("players.sample[].name is not primitive".to_owned())
                    })?;
                    sample.push(PlayerSample { name, id });
                }
            }
            (Some(online), Some(max), sample)
        }
    };

    let (favicon, favicon_base64, favicon_present) = match object.get("favicon") {
        Some(value) => {
            let raw = gson_string(value)
                .ok_or_else(|| ClientError::StatusJson("favicon is not primitive".to_owned()))?;
            let payload = raw.strip_prefix(FAVICON_PREFIX).map(str::to_owned);
            let favicon = match parse_favicon(&raw) {
                Ok(icon) => Some(icon),
                Err(err) => {
                    tracing::warn!(%err, "dropping an unusable server favicon");
                    None
                }
            };
            (favicon, payload, true)
        }
        None => (None, None, false),
    };

    Ok(ServerPing {
        motd,
        version_name,
        protocol,
        online_players,
        max_players,
        sample,
        favicon,
        favicon_base64,
        favicon_present,
        latency,
        ping_sent_at,
        raw_json: json.to_owned(),
    })
}

/// Gson 2.2.4's `JsonPrimitive.getAsInt` coercion for JSON numbers.
fn gson_i32(value: &Value) -> Option<i32> {
    let text = value.as_number()?.to_string();
    text.parse::<i32>()
        .ok()
        .or_else(|| text.parse::<i64>().ok().map(|number| number as i32))
        .or_else(|| text.parse::<u64>().ok().map(|number| number as i32))
        .or_else(|| decimal_to_wrapping_i32(&text))
}

/// `JsonUtils.getString` accepts every JSON primitive, despite its name.
fn gson_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// `BigDecimal.intValue`: truncate toward zero and retain the low 32 integer bits.
fn decimal_to_wrapping_i32(text: &str) -> Option<i32> {
    let (negative, unsigned) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(index) => (
            &unsigned[..index],
            unsigned[index + 1..].parse::<i64>().ok()?,
        ),
        None => (unsigned, 0),
    };
    let mut digits = Vec::with_capacity(mantissa.len());
    let mut fractional_digits = 0_i64;
    let mut after_decimal = false;
    for byte in mantissa.bytes() {
        match byte {
            b'0'..=b'9' => {
                digits.push(byte - b'0');
                if after_decimal {
                    fractional_digits += 1;
                }
            }
            b'.' if !after_decimal => after_decimal = true,
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }

    // BigDecimal stores `scale = fractional_digits - exponent` as a signed Java int and throws
    // `NumberFormatException: Exponent overflow` when that derived scale does not fit.
    let scale = fractional_digits.checked_sub(exponent)?;
    i32::try_from(scale).ok()?;
    let integer_digits = i64::try_from(digits.len()).ok()?.checked_sub(scale)?;
    if integer_digits <= 0 {
        return Some(0);
    }
    let kept = usize::try_from(integer_digits)
        .unwrap_or(usize::MAX)
        .min(digits.len());
    let mut low = 0_u32;
    for &digit in &digits[..kept] {
        low = low.wrapping_mul(10).wrapping_add(u32::from(digit));
    }
    let trailing_zeroes = integer_digits.saturating_sub(i64::try_from(kept).ok()?);
    if trailing_zeroes >= 32 {
        low = 0;
    } else {
        for _ in 0..trailing_zeroes {
            low = low.wrapping_mul(10);
        }
    }
    Some(if negative {
        low.wrapping_neg() as i32
    } else {
        low as i32
    })
}

/// Decode a `favicon` field into RGBA pixels.
///
/// # Errors
///
/// [`ClientError::Favicon`] if the prefix is wrong, the base64 does not decode, the payload is
/// not a PNG, or the image is not [`FAVICON_SIZE`]² — the three checks vanilla splits between
/// `OldServerPinger` (prefix) and `ServerListEntryNormal` (dimensions).
pub fn parse_favicon(raw: &str) -> Result<Favicon, ClientError> {
    let encoded = raw
        .strip_prefix(FAVICON_PREFIX)
        .ok_or_else(|| ClientError::Favicon("unknown format (not a data:image/png URI)".into()))?;
    // **Line breaks inside the base64 must be stripped, not merely trimmed.**
    //
    // java: `OldServerPinger` decodes with Apache Commons' `Base64.decodeBase64`, which silently
    // discards every character outside the base64 alphabet — newlines included. Java's own
    // encoders wrap at 76 columns by default, so a wrapped favicon is not an edge case, it is
    // what real servers send: the 1.8.8 server this was found against emits
    // `...suXdm\n7vS2O7Mzs32XXcou...`, and `base64`'s `STANDARD` engine rejects the interior
    // `\n` outright. `.trim()` only removed the ends, so every wrapped favicon was being dropped
    // with a warning and the server list drew a blank icon.
    //
    // Found by a run against a live server, which is exactly the class
    // of bug no amount of self-authored fixture data can surface.
    let compact: String = encoded
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&compact)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&compact))
        .map_err(|e| ClientError::Favicon(format!("base64: {e}")))?;
    // The decoder must refuse a size other than 64x64 from the PNG
    // header, before it allocates the surface. A few hundred bytes of PNG can claim
    // 11000x11000, which `load_from_memory_with_format` allocates (up to the image crate's
    // 512 MiB default) for every server pinged, only for the size check below to drop it.
    // Vanilla rejects any other size as well (`ServerListEntryNormal.prepareServerIcon`).
    let mut reader =
        image::ImageReader::with_format(std::io::Cursor::new(&bytes), image::ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(FAVICON_SIZE);
    limits.max_image_height = Some(FAVICON_SIZE);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|e| ClientError::Favicon(format!("png: {e}")))?
        .to_rgba8();
    if image.width() != FAVICON_SIZE || image.height() != FAVICON_SIZE {
        return Err(ClientError::Favicon(format!(
            "expected {FAVICON_SIZE}x{FAVICON_SIZE}, got {}x{}",
            image.width(),
            image.height()
        )));
    }
    Ok(Favicon {
        width: image.width(),
        height: image.height(),
        rgba: image.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_assets::Language;

    fn lang() -> Language {
        Language::new("en_US")
    }

    async fn assert_localhost_ipv4_modern_status(timed: bool) {
        use mc_protocol::Packet;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("IPv4 loopback listener");
        let port = listener.local_addr().expect("listener address").port();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept localhost ping");
            let mut conn = mc_protocol::connection::Connection::new(socket);
            let (id, mut body) = conn.read_packet().await.expect("modern handshake");
            assert_eq!(id, mc_protocol::handshake::Handshake::ID);
            let handshake = mc_protocol::handshake::Handshake::read(&mut body).unwrap();
            assert_eq!(handshake.protocol_version, 47);
            assert_eq!(handshake.server_address, "localhost");
            assert_eq!(handshake.server_port, port);
            assert_eq!(
                handshake.next_state,
                mc_protocol::handshake::NextState::Status
            );
            conn.set_state(mc_protocol::State::Status).unwrap();
            let (id, _) = conn.read_packet().await.expect("status request");
            assert_eq!(id, mc_protocol::status::Request::ID);
            conn.send(&mc_protocol::status::Response {
                json: r#"{"description":"local IPv4","version":{"name":"1.8.8","protocol":47},"players":{"online":1,"max":20}}"#.to_owned(),
            })
            .await
            .expect("status response");
            let (id, mut body) = conn.read_packet().await.expect("ping");
            assert_eq!(id, mc_protocol::status::Ping::ID);
            let ping = mc_protocol::status::Ping::read(&mut body).unwrap();
            conn.send(&mc_protocol::status::Pong {
                payload: ping.payload,
            })
            .await
            .expect("pong");
        });

        let result = tokio::time::timeout(Duration::from_secs(2), async {
            if timed {
                ping_with_timeout(&format!("localhost:{port}"), Duration::from_secs(1)).await
            } else {
                ping("localhost", port).await
            }
        })
        .await;
        if !matches!(result, Ok(Ok(_))) {
            server.abort();
        }
        let status = result
            .expect("localhost ping remains bounded")
            .expect("localhost selects IPv4 for modern status without a legacy retry");
        assert_eq!(status.protocol, 47);
        assert_eq!(status.version_name, "1.8.8");
        assert_eq!(status.online_players, Some(1));
        assert_eq!(status.motd.unformatted_text(&lang()), "local IPv4");
        assert!(status.latency < Duration::MAX, "modern Pong was received");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn localhost_ipv4_server_accepts_modern_ping() {
        assert_localhost_ipv4_modern_status(false).await;
    }

    #[tokio::test]
    async fn localhost_ipv4_server_accepts_timed_modern_ping() {
        assert_localhost_ipv4_modern_status(true).await;
    }

    #[test]
    fn client_tick_receipt_adds_the_rest_of_a_sub_tick_to_a_decoded_pong() {
        // A 12 ms RTT decodes on the network worker, but a client tick that began just before
        // that decode does not dispatch it until 50 ms after send. Vanilla records 50, not 12.
        let sent_at = Instant::now();
        let mut ping =
            parse_status_with_ping_sent_at("{}", Duration::from_millis(12), Some(sent_at))
                .expect("minimal status");

        ping.sample_latency_on_client_tick(sent_at + Duration::from_millis(50));
        assert_eq!(ping.latency, Duration::from_millis(50));
    }

    #[test]
    fn client_tick_receipt_keeps_an_exact_tick_boundary() {
        let sent_at = Instant::now();
        let mut ping =
            parse_status_with_ping_sent_at("{}", Duration::from_millis(49), Some(sent_at))
                .expect("minimal status");

        ping.sample_latency_on_client_tick(sent_at + Duration::from_millis(100));
        assert_eq!(ping.latency, Duration::from_millis(100));
    }

    #[test]
    fn client_tick_receipt_keeps_multi_tick_rtt_and_is_one_shot() {
        let sent_at = Instant::now();
        let mut ping =
            parse_status_with_ping_sent_at("{}", Duration::from_millis(127), Some(sent_at))
                .expect("minimal status");

        ping.sample_latency_on_client_tick(sent_at + Duration::from_millis(150));
        assert_eq!(ping.latency, Duration::from_millis(150));
        // A stale drain must not keep inflating the already-committed row.
        ping.sample_latency_on_client_tick(sent_at + Duration::from_millis(200));
        assert_eq!(ping.latency, Duration::from_millis(150));
    }

    /// A real 64x64 PNG, built here rather than committed, so the test proves the *decode* path
    /// rather than a fixture's provenance.
    fn favicon_data_uri(size: u32) -> String {
        let mut buffer = std::io::Cursor::new(Vec::new());
        let image = image::RgbaImage::from_fn(size, size, |x, y| {
            image::Rgba([x as u8, y as u8, 0x40, 0xFF])
        });
        image
            .write_to(&mut buffer, image::ImageFormat::Png)
            .expect("encode png");
        format!(
            "{FAVICON_PREFIX}{}",
            base64::engine::general_purpose::STANDARD.encode(buffer.into_inner())
        )
    }

    #[test]
    fn server_address_parser_matches_java_colon_and_bracket_oracle() {
        // Oracle table generated from an instrumented transcription of ServerAddress.fromString
        // lines 38-77, stopping where it would call getServerAddress. These cases cover every
        // branch in the parser, including Java String.split's trailing-empty behavior.
        for (input, host, port, srv_eligible) in [
            ("", "", 25_565, true),
            ("example.test", "example.test", 25_565, true),
            ("example.test:25565", "example.test", 25_565, true),
            ("example.test:not-a-port", "example.test", 25_565, true),
            ("example.test:+25565", "example.test", 25_565, true),
            ("example.test:65535", "example.test", 65_535, false),
            ("[::1]", "::1", 25_565, true),
            ("[::1]:25566", "::1", 25_566, false),
            ("[::1]ignored", "::1", 25_565, true),
            ("::1", "::1", 25_565, true),
            ("a:b:c", "a:b:c", 25_565, true),
            ("host:", "host", 25_565, true),
            ("host::", "host", 25_565, true),
            ("a:b:", "a", 25_565, true),
            (":1", "", 1, false),
        ] {
            let parsed = parse_server_address(input).expect(input);
            assert_eq!(parsed.host, host, "host for {input:?}");
            assert_eq!(parsed.port, port, "port for {input:?}");
            assert_eq!(parsed.srv_eligible, srv_eligible, "SRV for {input:?}");
        }

        for input in [":", "::", ":::"] {
            assert_eq!(
                parse_server_address(input),
                Err(ServerAddressParseError),
                "Java's astring[0] access throws for {input:?}"
            );
        }
    }

    #[test]
    fn out_of_socket_range_java_ports_fail_instead_of_becoming_the_default() {
        for (input, java_port) in [("host:-1", -1), ("host:70000", 70_000)] {
            let parsed = parse_server_address(input).expect("Java parses a signed int");
            assert_eq!(parsed.port, java_port);
            assert!(!parsed.srv_eligible);
            assert_eq!(parsed.socket_port(), Err(java_port));
            assert_ne!(
                parsed.port, DEFAULT_SERVER_PORT,
                "negative control: an out-of-u16 Java int must not silently default"
            );
        }
        assert_eq!(
            parse_server_address("host:65535").unwrap().socket_port(),
            Ok(65_535)
        );
        assert_eq!(
            parse_server_address("host:65536").unwrap().socket_port(),
            Err(65_536)
        );
    }

    #[test]
    fn server_address_ports_use_java_ascii_trim_not_unicode_trim() {
        let ascii_control = parse_server_address("host:\u{1f}47\u{20}").unwrap();
        assert_eq!(ascii_control.port, 47);
        let non_breaking_space = parse_server_address("host:\u{a0}47\u{a0}").unwrap();
        assert_eq!(
            non_breaking_space.port, DEFAULT_SERVER_PORT,
            "Java String.trim leaves U+00A0, so Integer.parseInt fails and defaults"
        );
        assert!(non_breaking_space.srv_eligible);
    }

    #[test]
    fn server_address_ports_accept_java_character_digit_values() {
        // Integer.parseInt delegates to Character.digit rather than an ASCII-only parser. The
        // supplementary-plane mathematical digit is a negative control: Java iterates UTF-16
        // code units, so its surrogate pair is not a digit and the port falls back to 25565.
        assert_eq!(
            parse_server_address("host:\u{0661}\u{0662}").unwrap().port,
            12
        );
        assert_eq!(
            parse_server_address("host:\u{ff11}\u{ff12}").unwrap().port,
            12
        );
        assert_eq!(parse_server_address("host:+\u{0661}").unwrap().port, 1);
        assert_eq!(
            parse_server_address("host:\u{0c70}").unwrap().port,
            DEFAULT_SERVER_PORT,
            "the gap between Java's Telugu and Telugu Supplement digit blocks is not a digit"
        );
        assert_eq!(
            parse_server_address("host:\u{1d7d9}").unwrap().port,
            DEFAULT_SERVER_PORT
        );
    }

    #[test]
    fn legacy_ping_request_matches_old_server_pingers_exact_bytes() {
        // Java OldServerPinger.tryCompatibilityPing writes these fields through Netty's
        // big-endian ByteBuf. This literal is an independent oracle for `a:25565`, including
        // the UTF-16-unit counts and the inner `7 + 2 * host.length()` payload length.
        let expected = vec![
            0xFE, 0x01, 0xFA, 0x00, 0x0B, 0x00, b'M', 0x00, b'C', 0x00, b'|', 0x00, b'P', 0x00,
            b'i', 0x00, b'n', 0x00, b'g', 0x00, b'H', 0x00, b'o', 0x00, b's', 0x00, b't', 0x00,
            0x09, 0x7F, 0x00, 0x01, 0x00, b'a', 0x00, 0x00, 0x63, 0xDD,
        ];
        assert_eq!(legacy_request("a", 25_565), expected);

        let astral = legacy_request("\u{1f600}", 25_565);
        assert_eq!(&astral[27..30], &[0x00, 0x0B, 0x7F]);
        assert_eq!(&astral[30..36], &[0x00, 0x02, 0xD8, 0x3D, 0xDE, 0x00]);
    }

    #[test]
    fn legacy_ping_response_uses_six_field_split_and_java_integer_defaults() {
        assert_eq!(
            parse_legacy_response("\u{a7}1\0not-a-number\x001.6.4\0Legacy MOTD\x0012\x0020"),
            Some(LegacyStatus {
                version_name: "1.6.4".to_owned(),
                motd: "Legacy MOTD".to_owned(),
                online: 12,
                max: 20,
            })
        );
        assert_eq!(
            parse_legacy_response("\u{a7}1\x00127\0Beta\0Old\0many\x0099999999999"),
            Some(LegacyStatus {
                version_name: "Beta".to_owned(),
                motd: "Old".to_owned(),
                online: -1,
                max: -1,
            })
        );
        assert!(parse_legacy_response("wrong\x00127\0Beta\0Old\x001\x002").is_none());
        assert!(parse_legacy_response("\u{a7}1\x00127\0too-short").is_none());
    }

    #[tokio::test]
    async fn resolver_failures_are_top_level_not_found_without_a_legacy_attempt() {
        for lookup in [
            Err(std::io::Error::other("simulated getaddrinfo failure")),
            Ok(None),
        ] {
            let result = tokio::time::timeout(
                Duration::from_millis(100),
                ping_with_lookup_result("unresolvable.invalid", 25_565, lookup),
            )
            .await
            .expect("UnknownHostException does not start Java's legacy socket");
            assert!(matches!(
                result,
                Err(ClientError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound
            ));
        }
    }

    #[tokio::test]
    async fn resolved_socket_keeps_the_server_address_in_the_status_handshake() {
        use mc_protocol::Packet;

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let resolved = listener.local_addr().unwrap();
        let (handshake_host, handshake_port) = resolve_server_address_with("typed.example", |_| {
            std::future::ready(Some(("FOO.Example".to_owned(), resolved.port())))
        })
        .await
        .unwrap();
        assert_eq!(
            (handshake_host.as_str(), handshake_port),
            ("FOO.Example", resolved.port())
        );
        let task = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut server = mc_protocol::connection::Connection::new(socket);
            let (id, mut body) = server.read_packet().await.unwrap();
            assert_eq!(id, mc_protocol::handshake::Handshake::ID);
            let handshake = mc_protocol::handshake::Handshake::read(&mut body).unwrap();
            assert_eq!(handshake.server_address, "FOO.Example");
            assert_eq!(handshake.server_port, resolved.port());
            server
                .set_state(mc_protocol::packet::State::Status)
                .unwrap();

            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, mc_protocol::status::Request::ID);
            server
                .send(&mc_protocol::status::Response {
                    json: "{}".to_owned(),
                })
                .await
                .unwrap();
            let (id, mut body) = server.read_packet().await.unwrap();
            assert_eq!(id, mc_protocol::status::Ping::ID);
            let ping = mc_protocol::status::Ping::read(&mut body).unwrap();
            server
                .send(&mc_protocol::status::Pong {
                    payload: ping.payload,
                })
                .await
                .unwrap();
        });

        let result = ping_with_lookup_result(&handshake_host, handshake_port, Ok(Some(resolved)))
            .await
            .unwrap();
        assert_eq!(result.protocol, 0);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn failed_modern_status_falls_back_to_the_live_legacy_exchange() {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let expected_request = legacy_request("127.0.0.1", port);

        let task = tokio::spawn(async move {
            let (modern, _) = listener.accept().await.unwrap();
            drop(modern);

            let (mut legacy, _) = listener.accept().await.unwrap();
            let mut request = vec![0; expected_request.len()];
            legacy.read_exact(&mut request).await.unwrap();
            assert_eq!(request, expected_request);

            let text = "\u{a7}1\x00127\x001.6.4\0Legacy MOTD\x0012\x0020";
            let units: Vec<u16> = text.encode_utf16().collect();
            let mut response = vec![0xFF];
            response.extend_from_slice(&(units.len() as i16).to_be_bytes());
            for unit in units {
                response.extend_from_slice(&unit.to_be_bytes());
            }
            legacy.write_all(&response).await.unwrap();
        });

        let result = ping("127.0.0.1", port)
            .await
            .expect("legacy response should replace the failed modern attempt");
        assert_eq!(result.version_name, "1.6.4");
        assert_eq!(result.protocol, -1);
        assert_eq!(result.motd.unformatted_text(&lang()), "Legacy MOTD");
        assert_eq!(
            (result.online_players, result.max_players),
            (Some(12), Some(20))
        );
        assert_eq!(result.latency, Duration::MAX);
        assert_eq!(result.latency.as_millis() as i64, -1);

        let live_parser = mc_ui::screens::multiplayer::StatusResponse::parse(&result.raw_json)
            .expect("the live driver reparses this bridge payload");
        assert_eq!(live_parser.version_name, "1.6.4");
        assert_eq!(live_parser.protocol, -1);
        assert_eq!((live_parser.online, live_parser.max), (Some(12), Some(20)));
        assert_eq!(live_parser.motd, "Legacy MOTD\u{a7}r");
        task.await.unwrap();
    }

    fn endpoint_marker(host: &str, port: u16) -> ServerPing {
        let mut ping = parse_status("{}", Duration::ZERO).expect("minimal status");
        ping.raw_json = format!("{host}|{port}");
        ping
    }

    #[tokio::test]
    async fn live_ping_address_uses_mocked_srv_then_pings_the_resolved_endpoint() {
        let resolver_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let pinger_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let resolver_observer = resolver_calls.clone();
        let pinger_observer = pinger_calls.clone();

        let ping = ping_address_with(
            "example.test:not-a-port",
            move |host| {
                resolver_observer.lock().unwrap().push(host);
                std::future::ready(Some(("srv.target.test".to_owned(), 25_570)))
            },
            move |host, port| {
                pinger_observer.lock().unwrap().push((host.clone(), port));
                std::future::ready(Ok(endpoint_marker(&host, port)))
            },
        )
        .await
        .expect("mock ping");

        assert_eq!(&*resolver_calls.lock().unwrap(), &["example.test"]);
        assert_eq!(
            &*pinger_calls.lock().unwrap(),
            &[("srv.target.test".to_owned(), 25_570)]
        );
        assert_eq!(ping.raw_json, "srv.target.test|25570");
    }

    #[tokio::test]
    async fn reusable_server_address_resolver_returns_the_socket_endpoint() {
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observer = calls.clone();
        let endpoint = resolve_server_address_with("example.test:25565", move |host| {
            observer.lock().unwrap().push(host);
            std::future::ready(Some(("srv.target.test.".to_owned(), 25_570)))
        })
        .await
        .unwrap();
        assert_eq!(endpoint, ("srv.target.test.".to_owned(), 25_570));
        assert_eq!(&*calls.lock().unwrap(), &["example.test"]);

        let invalid = resolve_server_address_with("example.test:70000", |_| async {
            panic!("an explicitly non-default port cannot issue SRV")
        })
        .await;
        assert!(matches!(
            invalid,
            Err(ClientError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidInput
        ));
    }

    #[tokio::test]
    async fn resolver_applies_java_idn_after_srv_selection_on_every_live_consumer() {
        let direct = resolve_server_address_with("b\u{fc}cher.example:25566", |_| async {
            panic!("non-default ports do not query SRV")
        })
        .await
        .unwrap();
        assert_eq!(direct, ("xn--bcher-kva.example".to_owned(), 25_566));

        let srv = resolve_server_address_with("typed.example", |_| async {
            Some(("ma\u{f1}ana.example.".to_owned(), 25_570))
        })
        .await
        .unwrap();
        assert_eq!(srv, ("xn--maana-pta.example.".to_owned(), 25_570));

        // The actual live resolver must preserve an ASCII-only label's case just as Java's
        // `IDN.toASCII` does; this is a wire-visible negative control for a whole-domain UTS 46
        // call, which lowercases the host before the handshake.
        let case_preserved = resolve_server_address_with("FOO.Example:25566", |_| async {
            panic!("non-default ports do not query SRV")
        })
        .await
        .unwrap();
        assert_eq!(case_preserved, ("FOO.Example".to_owned(), 25_566));

        let invalid = resolve_server_address_with("a..b:25566", |_| async {
            panic!("non-default ports do not query SRV")
        })
        .await;
        assert!(matches!(
            invalid,
            Err(ClientError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidInput
        ));
    }

    #[tokio::test]
    async fn live_ping_address_skips_srv_for_nondefault_and_falls_back_on_lookup_failure() {
        let direct = ping_address_with(
            "example.test:25566",
            |_| async { panic!("a non-default port is not SRV eligible") },
            |host, port| async move { Ok(endpoint_marker(&host, port)) },
        )
        .await
        .expect("direct ping");
        assert_eq!(direct.raw_json, "example.test|25566");

        let fallback = ping_address_with(
            "example.test",
            |_| std::future::ready(None),
            |host, port| async move { Ok(endpoint_marker(&host, port)) },
        )
        .await
        .expect("failed SRV falls back like Java");
        assert_eq!(fallback.raw_json, "example.test|25565");
    }

    #[tokio::test]
    async fn live_ping_address_rejects_signed_out_of_range_ports_before_dns_or_socket() {
        for address in ["example.test:-1", "example.test:70000"] {
            let result = ping_address_with(
                address,
                |_| async { panic!("invalid non-default ports are not SRV eligible") },
                |_, _| async { panic!("invalid Java-width port must not reach the socket") },
            )
            .await;
            assert!(matches!(
                result,
                Err(ClientError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidInput
            ));
        }
    }

    #[tokio::test]
    async fn ping_budget_keeps_a_committed_response_when_pong_is_idle() {
        use mc_protocol::Packet;

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("loopback listener");
        let port = listener.local_addr().expect("listener address").port();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("modern ping connection");
            let mut server = mc_protocol::connection::Connection::new(socket);
            let (id, mut body) = server.read_packet().await.expect("handshake");
            assert_eq!(id, mc_protocol::handshake::Handshake::ID);
            let handshake = mc_protocol::handshake::Handshake::read(&mut body).unwrap();
            assert_eq!(
                handshake.next_state,
                mc_protocol::handshake::NextState::Status
            );
            server
                .set_state(mc_protocol::packet::State::Status)
                .unwrap();
            let (id, _) = server.read_packet().await.expect("status request");
            assert_eq!(id, mc_protocol::status::Request::ID);
            server
                .send(&mc_protocol::status::Response {
                    json: r#"{"description":"kept","version":{"name":"1.8.9","protocol":47}}"#
                        .to_owned(),
                })
                .await
                .expect("status response");
            let (id, _) = server.read_packet().await.expect("status ping");
            assert_eq!(id, mc_protocol::status::Ping::ID);
            std::future::pending::<()>().await;
        });

        let result = tokio::time::timeout(
            Duration::from_secs(1),
            ping_with_timeout(&format!("127.0.0.1:{port}"), Duration::from_millis(40)),
        )
        .await
        .expect("the test itself must remain bounded")
        .expect("a decoded Response is a successful best-effort ping");
        assert_eq!(result.motd.unformatted_text(&lang()), "kept");
        assert_eq!(result.latency, Duration::MAX);
        assert!(result.ping_sent_at.is_none());
        server.abort();
    }

    #[tokio::test]
    async fn loopback_black_hole_respects_the_outer_ping_budget() {
        use tokio::io::AsyncReadExt as _;

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("loopback listener");
        let port = listener.local_addr().expect("listener address").port();
        let server = tokio::spawn(async move {
            // Keep both the modern attempt and a possible compatibility retry open. Neither
            // sends a response: the client-side budget, not a peer EOF, must terminate the row.
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.expect("ping connection");
                let mut first_byte = [0_u8; 1];
                let _ = socket.read(&mut first_byte).await;
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });

        let result =
            ping_with_timeout(&format!("127.0.0.1:{port}"), Duration::from_millis(40)).await;
        assert!(
            matches!(
                &result,
                Err(ClientError::Protocol(mc_protocol::ProtoError::Io(error)))
                    if error.kind() == std::io::ErrorKind::TimedOut
            ) || matches!(
                &result,
                Err(ClientError::Io(error)) if error.kind() == std::io::ErrorKind::TimedOut
            )
        );
        server.abort();
    }

    #[test]
    fn nslookup_srv_parser_accepts_windows_and_bind_presentations() {
        let windows = r#"
_minecraft._tcp.example.test SRV service location:
          priority       = 1
          weight         = 5
          port           = 25570
          svr hostname   = target.example.test
"#;
        assert_eq!(
            parse_nslookup_srv(windows),
            Some(("target.example.test".to_owned(), 25_570))
        );

        let bind = "_minecraft._tcp.example.test service = 0 5 25571 target.example.test.\n";
        assert_eq!(
            parse_nslookup_srv(bind),
            Some(("target.example.test.".to_owned(), 25_571))
        );
        assert_eq!(parse_nslookup_srv("*** NXDOMAIN"), None);
    }

    #[test]
    fn srv_parser_uses_jndis_first_attribute_value_without_rfc_reselection() {
        // ServerAddress.java calls Attribute.get() and splits that one value. It does not sort by
        // priority or perform RFC 2782 weighted selection itself; the DNS/JNDI order is binding.
        let answers = concat!(
            "_minecraft._tcp.example service = 20 0 25570 first.example.\n",
            "_minecraft._tcp.example service = 0 100 25571 tempting.example.\n",
        );
        assert_eq!(
            parse_nslookup_srv(answers),
            Some(("first.example.".to_owned(), 25_570))
        );

        let malformed_first = concat!(
            "_minecraft._tcp.example service = 0 5 nope broken.example.\n",
            "_minecraft._tcp.example service = 0 5 25571 later.example.\n",
        );
        assert_eq!(
            parse_nslookup_srv(malformed_first),
            None,
            "Java's catch(Throwable) falls back; it does not scan for a later usable value"
        );
    }

    /// **Regression test for a bug a real server found.**
    ///
    /// Java's base64 encoders wrap at 76 columns, so a real favicon arrives with embedded
    /// newlines. Vanilla's Apache Commons decoder ignores them; `base64`'s `STANDARD` engine
    /// refuses them. Before S21c this dropped the icon of every server that wrapped — including
    /// the 1.8.8 server the live-server test runs against, whose response carries
    /// `...suXdm\n7vS2O7Mzs32XXcou...`.
    ///
    /// The two forms are asserted to produce **byte-identical** pixels, not merely to both
    /// succeed, so a decoder that silently dropped the wrapped payload's tail would still fail.
    #[test]
    fn a_line_wrapped_favicon_decodes_the_way_vanillas_decoder_does() {
        let unwrapped = favicon_data_uri(FAVICON_SIZE);
        let payload = unwrapped
            .strip_prefix(FAVICON_PREFIX)
            .expect("the helper builds a data URI");

        // Exactly what a Java MIME encoder emits: a CRLF or LF every 76 characters.
        for line_ending in ["\n", "\r\n"] {
            let wrapped: String = payload
                .as_bytes()
                .chunks(76)
                .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
                .collect::<Vec<_>>()
                .join(line_ending);
            assert!(
                wrapped.len() > payload.len(),
                "the test data must actually be wrapped"
            );

            let plain = parse_favicon(&unwrapped).expect("the unwrapped form must decode");
            let folded = parse_favicon(&format!("{FAVICON_PREFIX}{wrapped}"))
                .expect("a line-wrapped favicon is what real servers send");
            assert_eq!(
                folded.rgba, plain.rgba,
                "wrapping must not change a single pixel ({line_ending:?})"
            );
            assert_eq!((folded.width, folded.height), (FAVICON_SIZE, FAVICON_SIZE));
        }
    }

    #[test]
    fn an_unpadded_favicon_decodes_the_way_nettys_decoder_does() {
        let padded = favicon_data_uri(FAVICON_SIZE);
        let unpadded = padded.trim_end_matches('=');
        assert_ne!(
            padded, unpadded,
            "negative control: the generated payload must actually have padding"
        );
        assert_eq!(
            parse_favicon(unpadded).expect("Netty accepts missing terminal padding"),
            parse_favicon(&padded).expect("control icon parses")
        );
    }

    #[test]
    fn a_full_vanilla_shaped_response_parses() {
        let json = format!(
            r#"{{"version":{{"name":"1.8.9","protocol":47}},
                 "players":{{"max":20,"online":2,"sample":[
                    {{"name":"Notch","id":"069a79f4-44e9-4726-a5be-fca90e38aaf5"}},
                    {{"name":"jeb_","id":"853c80ef-3c37-49fd-aa49-938b674adae6"}}]}},
                 "description":{{"text":"A Minecraft Server","color":"gold"}},
                 "favicon":"{}"}}"#,
            favicon_data_uri(FAVICON_SIZE)
        );
        let ping = parse_status(&json, Duration::from_millis(42)).expect("parse");
        assert_eq!(ping.version_name, "1.8.9");
        assert_eq!(ping.protocol, 47);
        assert_eq!(ping.online_players, Some(2));
        assert_eq!(ping.max_players, Some(20));
        assert_eq!(ping.sample.len(), 2);
        assert_eq!(ping.sample[0].name, "Notch");
        assert_eq!(ping.motd.unformatted_text(&lang()), "A Minecraft Server");
        // The colour must survive as a style, which is the whole reason the tree is kept.
        assert!(ping.motd.formatted_text(&lang()).contains('\u{a7}'));
        assert_eq!(ping.latency, Duration::from_millis(42));
        assert_eq!(ping.population_info(), "\u{a7}72\u{a7}8/\u{a7}720");
        let icon = ping.favicon.expect("favicon");
        assert_eq!((icon.width, icon.height), (64, 64));
        assert_eq!(icon.rgba.len(), 64 * 64 * 4);
        assert!(ping.favicon_present);
        assert!(ping.favicon_base64.is_some());
    }

    /// The status protocol allows a bare string description, and plenty of servers send one.
    #[test]
    fn a_plain_string_description_is_a_legal_component() {
        let ping = parse_status(r#"{"description":"Hello §aWorld"}"#, Duration::ZERO).unwrap();
        assert_eq!(ping.motd.unformatted_text(&lang()), "Hello \u{a7}aWorld");
    }

    /// java: `ServerStatusResponse.Serializer` leaves every field null when absent, and
    /// `OldServerPinger` substitutes `"Old"`/`0`/`???`.
    #[test]
    fn missing_objects_fall_back_the_way_vanilla_does() {
        let ping = parse_status("{}", Duration::ZERO).unwrap();
        assert_eq!(ping.version_name, "Old");
        assert_eq!(ping.protocol, 0);
        assert_eq!(ping.online_players, None);
        assert_eq!(ping.population_info(), "\u{a7}8???");
        assert!(ping.favicon.is_none());
        assert_eq!(ping.motd.unformatted_text(&lang()), "");
    }

    #[test]
    fn a_non_object_response_is_an_error_not_a_panic() {
        assert!(matches!(
            parse_status("[]", Duration::ZERO),
            Err(ClientError::StatusJson(_))
        ));
        assert!(matches!(
            parse_status("not json at all", Duration::ZERO),
            Err(ClientError::StatusJson(_))
        ));
    }

    #[test]
    fn favicons_are_validated_the_way_vanilla_validates_them() {
        // Wrong prefix — java logs "Invalid server icon (unknown format)".
        assert!(matches!(
            parse_favicon("data:image/jpeg;base64,AAAA"),
            Err(ClientError::Favicon(_))
        ));
        // Not base64.
        assert!(matches!(
            parse_favicon(&format!("{FAVICON_PREFIX}!!!!")),
            Err(ClientError::Favicon(_))
        ));
        // Valid PNG, wrong size.
        assert!(matches!(
            parse_favicon(&favicon_data_uri(32)),
            Err(ClientError::Favicon(_))
        ));
        // Valid.
        assert!(parse_favicon(&favicon_data_uri(64)).is_ok());
    }

    /// A bad favicon must not lose the rest of the response — vanilla drops the icon and keeps
    /// the MOTD, which is what a server-list row needs most.
    #[test]
    fn a_bad_favicon_does_not_fail_the_whole_ping() {
        let ping = parse_status(
            r#"{"description":{"text":"hi"},"favicon":"garbage"}"#,
            Duration::ZERO,
        )
        .expect("parse");
        assert!(ping.favicon.is_none());
        assert!(ping.favicon_present);
        assert!(ping.favicon_base64.is_none());
        assert_eq!(ping.motd.unformatted_text(&lang()), "hi");
    }

    #[test]
    fn gson_numeric_coercion_truncates_and_wraps_instead_of_zeroing_or_saturating() {
        let ping = parse_status(
            r#"{"version":{"name":"x","protocol":99999999999},"players":{"online":2.9,"max":-99999999999}}"#,
            Duration::ZERO,
        )
        .expect("Gson accepts every numeric primitive here");
        assert_eq!(ping.protocol, 99_999_999_999_i64 as i32);
        assert_eq!(ping.online_players, Some(2));
        assert_eq!(ping.max_players, Some(-99_999_999_999_i64 as i32));
        assert_ne!(
            ping.protocol,
            i32::MAX,
            "negative control: clamping an out-of-range protocol must fail"
        );

        assert_eq!(gson_i32(&serde_json::from_str("1e100").unwrap()), Some(0));
        assert_eq!(gson_i32(&serde_json::from_str("-2.9").unwrap()), Some(-2));
    }

    #[test]
    fn gson_numeric_coercion_keeps_extreme_fraction_lexemes_exact() {
        // Oracle: bundled Gson 2.2.4 JsonParser + JsonPrimitive.getAsInt. Every value here loses
        // relevant integer bits through an f64 round trip, so this also guards the workspace's
        // serde_json `arbitrary_precision` feature.
        for (lexeme, expected) in [
            ("9007199254740993.5", 1),
            ("-9007199254740993.5", -1),
            ("4294967297.999999999999", 1),
            ("1.5e32", i32::MIN),
            ("1.0e2147483649", 0),
        ] {
            let json = format!(
                r#"{{"players":{{"online":{lexeme},"max":0}},"version":{{"name":"x","protocol":47}}}}"#
            );
            let ping =
                parse_status(&json, Duration::ZERO).expect("Gson-compatible numeric primitive");
            assert_eq!(ping.online_players, Some(expected), "lexeme {lexeme}");
        }

        for lexeme in ["1e2147483649", "1e-2147483648"] {
            let json = format!(r#"{{"players":{{"online":{lexeme},"max":0}}}}"#);
            assert!(
                matches!(
                    parse_status(&json, Duration::ZERO),
                    Err(ClientError::StatusJson(_))
                ),
                "Gson reports exponent overflow for {lexeme}"
            );
        }
    }

    #[test]
    fn null_and_empty_array_descriptions_are_empty_components() {
        for description in ["null", "[]"] {
            let json =
                format!(r#"{{"description":{description},"players":{{"online":1,"max":2}}}}"#);
            let ping = parse_status(&json, Duration::ZERO)
                .expect("Gson maps null and an empty component array to null");
            assert_eq!(ping.motd.unformatted_text(&lang()), "");
            assert_eq!(ping.online_players, Some(1));
        }
    }

    #[test]
    fn malformed_present_status_fields_fail_the_whole_ping_like_gson() {
        for json in [
            r#"{"players":{}}"#,
            r#"{"players":"x"}"#,
            r#"{"players":{"online":1,"max":"20"}}"#,
            r#"{"version":{"name":"x","protocol":"47"}}"#,
            r#"{"favicon":{}}"#,
            r#"{"players":{"online":1,"max":2,"sample":[7]}}"#,
            r#"{"players":{"online":1,"max":2,"sample":[{"name":"x","id":"not-a-uuid"}]}}"#,
            r#"{"players":{"online":1,"max":2,"sample":[{"id":"0-0-0-0-0"}]}}"#,
        ] {
            assert!(
                matches!(
                    parse_status(json, Duration::ZERO),
                    Err(ClientError::StatusJson(_))
                ),
                "accepted {json}"
            );
        }

        let ping = parse_status(
            r#"{"players":{"online":1,"max":2,"sample":"not an array"}}"#,
            Duration::ZERO,
        )
        .expect("JsonUtils.isJsonArray ignores a wrong-typed optional sample");
        assert!(ping.sample.is_empty());

        let ping = parse_status(
            r#"{"players":{"online":1,"max":2,"sample":[{"id":"0-0-0-0-0","name":"short"}]}}"#,
            Duration::ZERO,
        )
        .expect("Java 8 UUID.fromString accepts shortened groups");
        assert_eq!(ping.sample[0].id, "0-0-0-0-0");
    }

    #[test]
    fn jsonutils_get_string_coerces_primitive_profile_and_status_fields() {
        let ping = parse_status(
            r#"{"version":{"name":7,"protocol":47},"players":{"online":1,"max":2,"sample":[{"id":"0-0-0-0-0","name":true}]},"favicon":12}"#,
            Duration::ZERO,
        )
        .expect("JsonUtils.getString accepts number and boolean primitives");
        assert_eq!(ping.version_name, "7");
        assert_eq!(ping.sample[0].name, "true");
        assert!(
            ping.favicon.is_none(),
            "numeric favicon becomes an unsupported-prefix string"
        );
    }
}
