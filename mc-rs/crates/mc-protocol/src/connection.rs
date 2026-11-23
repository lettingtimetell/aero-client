//! The connection itself: framing, encryption, compression, and the state machine that gets a
//! client from TCP connect to [`State::Play`].
//!
//! java: `network/NetworkManager.java`, `client/network/NetHandlerLoginClient.java`,
//! `client/multiplayer/OldServerPinger.java`
//!
//! # Layer order — the thing that is easy to get backwards
//!
//! Vanilla builds this as a Netty pipeline. After `enableEncryption` and `setCompressionTreshold`
//! have both run, `NetworkManager`'s pipeline reads (head → tail):
//!
//! ```text
//! timeout → decrypt → splitter → decompress → decoder → encrypt → prepender → compress → encoder
//! ```
//!
//! Netty runs inbound handlers head→tail and outbound handlers tail→head, so the two directions
//! are:
//!
//! ```text
//! inbound   socket → decrypt(whole byte stream) → strip length prefix → decompress → packet
//! outbound  packet → compress → add length prefix → encrypt(whole byte stream) → socket
//! ```
//!
//! Two consequences, both load-bearing:
//!
//! * **Compression is inside encryption.** The length prefix is compressed data's *outer*
//!   wrapper but encryption's *inner* content — encryption covers the prefix, compression does
//!   not. `enableEncryption` inserts `decrypt` before `splitter` and `encrypt` before
//!   `prepender`, which is precisely this ordering.
//! * **Encryption is a property of the byte stream, not of a packet.** It is applied to whatever
//!   bytes happen to be moving, which is why [`crypt::EncryptStream`](crate::crypt::EncryptStream)
//!   is created once and lives as long as the connection (hazard H4).
//!
//! [`Connection`] reproduces exactly this: each socket read is decrypted the moment it arrives,
//! so the framing buffer always holds plaintext, and [`Connection::send_payload`] compresses,
//! frames, then encrypts, in that order.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bytes::BytesMut;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{lookup_host, TcpStream, ToSocketAddrs};
use tokio_util::codec::{Decoder, Encoder};
use uuid::Uuid;

use crate::codec::PacketFrameCodec;
use crate::compress::CompressionCodec;
use crate::crypt::{begin_encryption, DecryptStream, EncryptStream, SharedSecret};
use crate::handshake::{Handshake, NextState};
use crate::login::{Disconnect, EncryptionRequest, LoginStart, LoginSuccess, SetCompression};
use crate::packet::{Packet, State};
use crate::status::{Ping, Pong, Request, Response};
use crate::{PacketBuffer, ProtoError};

/// The protocol version this client speaks: **47**, Minecraft 1.8.9.
pub const PROTOCOL_VERSION: i32 = 47;

/// How many bytes to pull off the socket per read.
const READ_CHUNK: usize = 8192;

/// Vanilla installs Netty's `ReadTimeoutHandler(30)` on status connections. The deadline is an
/// idle deadline: every successful transport read starts a fresh thirty-second interval.
const STATUS_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// A server that keeps sending login packets without finishing is stalling us; give up rather
/// than spin forever on a hostile peer.
///
/// Match the vanilla launcher's `java.net.preferIPv4Stack=true` before opening a socket.
///
/// Java installs that property before `InetAddress.getByName` resolves the server address, so an
/// IPv6-only result is not a fallback candidate. Keep the filtering separate from the transport
/// call: it makes the address-family contract explicit and gives it a mutation-sensitive oracle.
fn ipv4_only(addrs: impl IntoIterator<Item = SocketAddr>) -> Vec<SocketAddr> {
    addrs.into_iter().filter(SocketAddr::is_ipv4).collect()
}

/// Is `from -> to` a transition `EnumConnectionState` actually permits?
///
/// java: the client only ever moves `HANDSHAKING → STATUS` (`C00Handshake` with next-state 1),
/// `HANDSHAKING → LOGIN` (next-state 2), and `LOGIN → PLAY`
/// (`NetHandlerLoginClient.handleLoginSuccess`). There is no path back.
fn transition_is_legal(from: State, to: State) -> bool {
    matches!(
        (from, to),
        (State::Handshaking, State::Status)
            | (State::Handshaking, State::Login)
            | (State::Login, State::Play)
    )
}

/// Compress, frame, then encrypt — the outbound half of the layer order, in one place so that
/// [`Connection`] and [`ConnectionWriter`] cannot drift apart on it.
fn seal(
    compression: &mut CompressionCodec,
    framer: &mut PacketFrameCodec,
    encrypt: Option<&mut EncryptStream>,
    payload: &[u8],
) -> Result<BytesMut, ProtoError> {
    let mut compressed = BytesMut::with_capacity(payload.len() + 8);
    compression.encode(payload, &mut compressed)?;

    let mut framed = BytesMut::with_capacity(compressed.len() + 5);
    framer.encode(&compressed[..], &mut framed)?;

    if let Some(cipher) = encrypt {
        cipher.apply(&mut framed);
    }
    Ok(framed)
}

/// Decrypt a freshly-read socket chunk and append it to the (plaintext) framing buffer.
fn absorb(decrypt: Option<&mut DecryptStream>, read_buf: &mut BytesMut, chunk: &mut [u8]) {
    if let Some(cipher) = decrypt {
        cipher.apply(chunk);
    }
    read_buf.extend_from_slice(chunk);
}

/// Take one whole packet payload out of the framing buffer, if there is one.
fn unseal(
    compression: &mut CompressionCodec,
    framer: &mut PacketFrameCodec,
    read_buf: &mut BytesMut,
) -> Result<Option<PacketBuffer>, ProtoError> {
    loop {
        match framer.decode(read_buf)? {
            // Both vanilla's compressed and uncompressed decoder paths ignore an empty frame.
            // Keep draining already-buffered frames rather than surfacing an id-less packet.
            Some(frame) if frame.is_empty() => continue,
            Some(frame) => {
                let payload = compression.decode(&frame)?;
                // NettyCompressionDecoder emits an empty ByteBuf for a marker-only compressed
                // body (`[0]`), and MessageDeserializer drops it before attempting a packet id.
                // Preserve that boundary behavior while leaving all decode errors observable.
                if payload.is_empty() {
                    continue;
                }
                return Ok(Some(PacketBuffer::from_bytes(payload)));
            }
            None => return Ok(None),
        }
    }
}

/// One client connection: the socket plus the three transforms that sit on it.
///
/// Generic over the transport so tests can drive it over `tokio::io::duplex` or a loopback
/// [`TcpStream`] without changing a line of the protocol code.
///
/// Reading and writing both take `&mut self`, which is right for the strictly request/response
/// handshake, status and login sequences. Once in [`State::Play`] traffic is full duplex; call
/// [`Connection::split`] then, and drive the two halves from separate tasks.
pub struct Connection<S> {
    stream: S,
    /// Plaintext bytes received but not yet framed. Decryption happens on the way *in* to this
    /// buffer, so its contents are always post-decryption, pre-framing.
    read_buf: BytesMut,
    scratch: Vec<u8>,
    framer: PacketFrameCodec,
    encrypt: Option<EncryptStream>,
    decrypt: Option<DecryptStream>,
    compression: CompressionCodec,
    state: State,
}

impl<S> std::fmt::Debug for Connection<S> {
    /// Deliberately hand-written: the read scratch buffer is 8 KiB of noise and the cipher
    /// state is key-derived, so neither belongs in a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field("state", &self.state)
            .field("encrypted", &self.encrypt.is_some())
            .field("compression", &self.compression.threshold())
            .field("buffered", &self.read_buf.len())
            .finish_non_exhaustive()
    }
}

impl Connection<TcpStream> {
    /// Open a TCP connection to a Minecraft server. The connection starts in
    /// [`State::Handshaking`] with no encryption and no compression, exactly as vanilla's does.
    ///
    /// Nagle's algorithm is disabled: the protocol is latency-sensitive and made of many small
    /// writes, and vanilla sets `ChannelOption.TCP_NODELAY` on its bootstrap for the same reason.
    /// Address resolution is likewise IPv4-only, matching the launcher's
    /// `java.net.preferIPv4Stack=true` setting.
    ///
    /// # Errors
    ///
    /// [`ProtoError::Io`] if the connection cannot be established.
    pub async fn connect(addr: impl ToSocketAddrs) -> Result<Connection<TcpStream>, ProtoError> {
        let addrs = ipv4_only(lookup_host(addr).await?);
        let stream = TcpStream::connect(addrs.as_slice()).await?;
        stream.set_nodelay(true)?;
        Ok(Connection::new(stream))
    }
}

impl<S> Connection<S> {
    /// Wrap an already-established transport.
    pub fn new(stream: S) -> Connection<S> {
        Connection {
            stream,
            read_buf: BytesMut::with_capacity(READ_CHUNK),
            scratch: vec![0u8; READ_CHUNK],
            framer: PacketFrameCodec::new(),
            encrypt: None,
            decrypt: None,
            compression: CompressionCodec::disabled(),
            state: State::Handshaking,
        }
    }

    /// The connection state packet ids are currently interpreted in.
    #[must_use]
    pub fn state(&self) -> State {
        self.state
    }

    /// Whether AES-CFB8 is active in both directions.
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.encrypt.is_some()
    }

    /// The negotiated compression threshold, or `None` if the layer is not installed.
    #[must_use]
    pub fn compression_threshold(&self) -> Option<usize> {
        self.compression.threshold()
    }

    /// Move to the next connection state.
    ///
    /// # Errors
    ///
    /// [`ProtoError::IllegalStateTransition`] for anything vanilla's own state machine cannot do
    /// — the point being that a server which sends, say, `LoginSuccess` while we are still in
    /// `Status` gets a typed error rather than silently reinterpreting every subsequent packet id.
    pub fn set_state(&mut self, to: State) -> Result<(), ProtoError> {
        if !transition_is_legal(self.state, to) {
            return Err(ProtoError::IllegalStateTransition {
                from: self.state,
                to,
            });
        }
        self.state = to;
        Ok(())
    }

    /// Switch both directions to AES-CFB8 under `secret`.
    ///
    /// Call this only after `C01PacketEncryptionResponse` has been flushed — that packet is the
    /// last plaintext byte in either direction. java: the write-completion listener in
    /// `NetHandlerLoginClient.handleEncryptionRequest`.
    ///
    /// # Errors
    ///
    /// [`ProtoError::AlreadyEncrypted`] if encryption is already on. Re-keying mid-connection is
    /// not part of the protocol, and doing it by accident is hazard H4's failure mode.
    pub fn enable_encryption(&mut self, secret: &SharedSecret) -> Result<(), ProtoError> {
        if self.encrypt.is_some() {
            return Err(ProtoError::AlreadyEncrypted);
        }
        self.encrypt = Some(EncryptStream::new(secret));
        self.decrypt = Some(DecryptStream::new(secret));
        Ok(())
    }

    /// Apply a `SetCompression` threshold; a negative value removes the layer.
    /// java: `NetworkManager.setCompressionTreshold`
    pub fn set_compression_threshold(&mut self, threshold: i32) {
        self.compression.set_threshold(threshold);
    }

    /// The underlying transport, for callers that need to close or inspect it.
    pub fn get_ref(&self) -> &S {
        &self.stream
    }

    /// Give the transport back.
    pub fn into_inner(self) -> S {
        self.stream
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin> Connection<S> {
    /// Send one typed packet in the current state.
    ///
    /// # Errors
    ///
    /// [`ProtoError::NbtEncode`] for invalid outgoing NBT, [`ProtoError::Io`] on a transport
    /// failure, or [`ProtoError::Zlib`] if compression fails.
    pub async fn send<P: Packet>(&mut self, packet: &P) -> Result<(), ProtoError> {
        // `NetworkManager.dispatchPacket` selects the state from the outgoing packet's class,
        // then changes the channel's state before its encoder sees the packet.  This is why a
        // C00 handshake can be followed immediately by a LoginStart/Status Request: the latter
        // packet moves the Netty channel into its registered namespace.  Do this before the
        // encode too; vanilla has already changed the channel even if the eventual write fails.
        self.state = P::STATE;
        let mut payload = PacketBuffer::new();
        payload.write_varint(P::ID);
        packet.write(&mut payload);
        payload.finish_write()?;
        self.send_payload(payload.as_slice()).await
    }

    /// Send an already-serialised payload (`packet id` VarInt followed by the body) through the
    /// compress → frame → encrypt pipeline.
    ///
    /// # Errors
    ///
    /// [`ProtoError::FrameTooLong`] if the framed result exceeds the codec's ceiling,
    /// [`ProtoError::Zlib`] if compression fails, [`ProtoError::Io`] on a write failure.
    pub async fn send_payload(&mut self, payload: &[u8]) -> Result<(), ProtoError> {
        let framed = seal(
            &mut self.compression,
            &mut self.framer,
            self.encrypt.as_mut(),
            payload,
        )?;
        self.stream.write_all(&framed).await?;
        self.stream.flush().await?;
        Ok(())
    }

    /// Read one whole packet payload: `packet id` VarInt followed by the body, decrypted and
    /// decompressed.
    ///
    /// # Errors
    ///
    /// [`ProtoError::ConnectionClosed`] if the peer hangs up mid-frame, plus any framing or
    /// decompression error.
    pub async fn read_payload(&mut self) -> Result<PacketBuffer, ProtoError> {
        loop {
            if let Some(payload) =
                unseal(&mut self.compression, &mut self.framer, &mut self.read_buf)?
            {
                return Ok(payload);
            }
            self.fill().await?;
        }
    }

    /// Read one packet and split its id off the front.
    ///
    /// # Errors
    ///
    /// As [`Connection::read_payload`], plus [`ProtoError::Eof`] for a payload with no id.
    pub async fn read_packet(&mut self) -> Result<(i32, PacketBuffer), ProtoError> {
        let mut payload = self.read_payload().await?;
        let id = payload.read_varint()?;
        Ok((id, payload))
    }

    /// Pull more bytes off the transport, decrypting them on the way in so that the framing
    /// buffer only ever holds plaintext.
    ///
    /// This is where hazard H4 is actually satisfied: one [`DecryptStream`] handles every read
    /// for the life of the connection, regardless of how TCP chose to split the data.
    async fn fill(&mut self) -> Result<(), ProtoError> {
        let n = self.stream.read(&mut self.scratch).await?;
        if n == 0 {
            return Err(ProtoError::ConnectionClosed);
        }
        absorb(
            self.decrypt.as_mut(),
            &mut self.read_buf,
            &mut self.scratch[..n],
        );
        Ok(())
    }

    /// Split into halves that can be driven from two tasks at once.
    ///
    /// Play-state traffic is full duplex — S14 has to keep reading the socket while the game
    /// loop pushes movement packets at it — and a single `&mut Connection` cannot do both. The
    /// split is clean because the two CFB8 streams are already independent: the writer takes the
    /// encryptor, the reader takes the decryptor and whatever bytes are still buffered. The
    /// compression threshold is copied into both halves, mirroring vanilla's two separate
    /// `NettyCompression{En,De}coder` handlers.
    ///
    /// Split *after* the login sequence has finished: [`login`] needs both directions, and the
    /// connection state ([`Connection::state`]) does not survive the split — by then it is
    /// always [`State::Play`].
    #[must_use]
    pub fn split(
        self,
    ) -> (
        ConnectionReader<tokio::io::ReadHalf<S>>,
        ConnectionWriter<tokio::io::WriteHalf<S>>,
    ) {
        let (read_half, write_half) = tokio::io::split(self.stream);
        let threshold = self.compression.threshold();
        let mut read_compression = CompressionCodec::disabled();
        let mut write_compression = CompressionCodec::disabled();
        if let Some(t) = threshold {
            read_compression = CompressionCodec::with_threshold(t);
            write_compression = CompressionCodec::with_threshold(t);
        }
        (
            ConnectionReader {
                stream: read_half,
                read_buf: self.read_buf,
                scratch: self.scratch,
                framer: self.framer,
                decrypt: self.decrypt,
                compression: read_compression,
            },
            ConnectionWriter {
                stream: write_half,
                framer: self.framer,
                encrypt: self.encrypt,
                compression: write_compression,
            },
        )
    }
}

/// The inbound half of a split [`Connection`]. See [`Connection::split`].
pub struct ConnectionReader<R> {
    stream: R,
    read_buf: BytesMut,
    scratch: Vec<u8>,
    framer: PacketFrameCodec,
    decrypt: Option<DecryptStream>,
    compression: CompressionCodec,
}

impl<R> std::fmt::Debug for ConnectionReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionReader")
            .field("encrypted", &self.decrypt.is_some())
            .field("compression", &self.compression.threshold())
            .field("buffered", &self.read_buf.len())
            .finish_non_exhaustive()
    }
}

impl<R> ConnectionReader<R> {
    /// Apply a `SetCompression` threshold to the inbound direction.
    pub fn set_compression_threshold(&mut self, threshold: i32) {
        self.compression.set_threshold(threshold);
    }

    /// The active inbound compression threshold, if any.
    #[must_use]
    pub fn compression_threshold(&self) -> Option<usize> {
        self.compression.threshold()
    }
}

impl<R: AsyncRead + Unpin> ConnectionReader<R> {
    /// Read one whole packet payload. See [`Connection::read_payload`].
    ///
    /// This is cancel-safe: the only await point is the socket read, and any bytes it delivers
    /// are absorbed into the framing buffer before the next one. A `tokio::select!` arm that
    /// drops this future loses nothing.
    ///
    /// # Errors
    ///
    /// As [`Connection::read_payload`].
    pub async fn read_payload(&mut self) -> Result<PacketBuffer, ProtoError> {
        loop {
            if let Some(payload) =
                unseal(&mut self.compression, &mut self.framer, &mut self.read_buf)?
            {
                return Ok(payload);
            }
            let n = self.stream.read(&mut self.scratch).await?;
            if n == 0 {
                return Err(ProtoError::ConnectionClosed);
            }
            absorb(
                self.decrypt.as_mut(),
                &mut self.read_buf,
                &mut self.scratch[..n],
            );
        }
    }

    /// Read one packet and split its id off the front. See [`Connection::read_packet`].
    ///
    /// # Errors
    ///
    /// As [`Connection::read_packet`].
    pub async fn read_packet(&mut self) -> Result<(i32, PacketBuffer), ProtoError> {
        let mut payload = self.read_payload().await?;
        let id = payload.read_varint()?;
        Ok((id, payload))
    }
}

/// The outbound half of a split [`Connection`]. See [`Connection::split`].
pub struct ConnectionWriter<W> {
    stream: W,
    framer: PacketFrameCodec,
    encrypt: Option<EncryptStream>,
    compression: CompressionCodec,
}

impl<W> std::fmt::Debug for ConnectionWriter<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionWriter")
            .field("encrypted", &self.encrypt.is_some())
            .field("compression", &self.compression.threshold())
            .finish_non_exhaustive()
    }
}

impl<W> ConnectionWriter<W> {
    /// Apply a `SetCompression` threshold to the outbound direction.
    pub fn set_compression_threshold(&mut self, threshold: i32) {
        self.compression.set_threshold(threshold);
    }

    /// The active outbound compression threshold, if any.
    #[must_use]
    pub fn compression_threshold(&self) -> Option<usize> {
        self.compression.threshold()
    }
}

impl<W: AsyncWrite + Unpin> ConnectionWriter<W> {
    /// Send one typed packet. See [`Connection::send`].
    ///
    /// # Errors
    ///
    /// As [`Connection::send`].
    pub async fn send<P: Packet>(&mut self, packet: &P) -> Result<(), ProtoError> {
        let mut payload = PacketBuffer::new();
        payload.write_varint(P::ID);
        packet.write(&mut payload);
        payload.finish_write()?;
        self.send_payload(payload.as_slice()).await
    }

    /// Send an already-serialised payload. See [`Connection::send_payload`].
    ///
    /// Unlike [`ConnectionReader::read_payload`] this is **not** cancel-safe: dropping the
    /// future mid-write can leave a partial frame on the wire, which desynchronises the CFB8
    /// stream permanently. Drive it to completion.
    ///
    /// # Errors
    ///
    /// As [`Connection::send_payload`].
    pub async fn send_payload(&mut self, payload: &[u8]) -> Result<(), ProtoError> {
        let framed = seal(
            &mut self.compression,
            &mut self.framer,
            self.encrypt.as_mut(),
            payload,
        )?;
        self.stream.write_all(&framed).await?;
        self.stream.flush().await?;
        Ok(())
    }
}

/// A server-list ping result. java: `OldServerPinger`
#[derive(Debug, Clone)]
pub struct ServerStatus {
    /// The raw status JSON (version, players, description, favicon). Parsing it is S14/S16's
    /// job — this crate deliberately does not model the schema, exactly as it leaves chat
    /// components as JSON strings elsewhere.
    pub json: String,
    /// Round-trip time measured across the `Ping`/`Pong` exchange. [`Duration::MAX`] is vanilla's
    /// `pingToServer = -1` sentinel: a valid status response was received, but the best-effort
    /// Ping/Pong leg did not complete.
    pub latency: Duration,
    /// Monotonic instant at which the client sent the status `Ping`, when its `Pong` completed.
    ///
    /// `OldServerPinger` does not commit `pingToServer` on Netty's decode thread. Its game
    /// thread calls `pingPendingNetworks()` once per client tick and samples the clock there.
    /// A UI that needs that observable cadence must therefore re-sample this instant when it
    /// folds the completed result into its row, rather than displaying [`latency`](Self::latency)
    /// measured on this transport task. `None` is the same post-response Ping/Pong failure that
    /// gives [`latency`](Self::latency) the `Duration::MAX` sentinel.
    pub ping_sent_at: Option<Instant>,
}

/// Which way a frame crossed the wire, for [`login`]'s optional observer hook.
///
/// Duplicated here rather than reused from `mc_parity::capture::Direction`, which has the exact
/// same two variants: `mc-parity` depends on this crate for [`Connection`] and the packet types,
/// so the reverse dependency would be a cycle. A caller wiring the two together (see
/// `mc-client`'s `net::capture::CaptureSink::record_login`) matches variant-for-variant, not by
/// conversion logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Server -> client.
    Clientbound,
    /// Client -> server.
    Serverbound,
}

/// The type of [`login`]'s optional observer hook, named so its three nested layers (`Option`, a
/// mutable reference, and a `+ Send` trait object — `Send` because `login` ends up inside a
/// future `mc-client` spawns onto a multi-threaded executor) read as one thing at each call site
/// instead of tripping clippy's complexity lint every time.
pub type LoginObserver<'a> = Option<&'a mut (dyn FnMut(Direction, &[u8]) + Send + 'a)>;

/// What the client learned by completing the login sequence.
#[derive(Debug, Clone)]
pub struct LoginOutcome {
    /// The authoritative UUID the server assigned. For an online-mode server this is the real
    /// Mojang UUID; offline-mode servers derive it from the name.
    pub uuid: Uuid,
    /// The authoritative username. May differ in case from what was sent.
    pub username: String,
    /// The compression threshold the server negotiated, if it sent `SetCompression`.
    pub compression_threshold: Option<usize>,
    /// Whether the connection ended up encrypted. False for an offline-mode server, which never
    /// sends `EncryptionRequest`.
    pub encrypted: bool,
}

/// Policy for failures returned by the session-server join callback during login.
///
/// java: `NetHandlerLoginClient.handleEncryptionRequest` catches every
/// `AuthenticationException` and continues only when `ServerData.isOnLAN()` is true.  The
/// protocol layer cannot infer that server-list fact from a generic transport, so the ordinary
/// [`login`] and [`login_online`] entry points remain strict; a client that owns the server-data
/// context must opt into [`LoginAuthPolicy::AllowLanAuthFailure`] through the policy-aware entry
/// point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoginAuthPolicy {
    /// Abort on every session-server join failure (the normal remote-server behavior).
    Strict,
    /// Ignore only [`ProtoError::Auth`] from the join callback and continue the encryption
    /// exchange, matching vanilla's LAN tolerance. Transport/protocol errors remain fatal.
    AllowLanAuthFailure,
}

impl LoginAuthPolicy {
    /// Whether `error` is the LAN-tolerated session-server failure class.
    #[must_use]
    fn tolerates_join_error(self, error: &ProtoError) -> bool {
        matches!(
            (self, error),
            (Self::AllowLanAuthFailure, ProtoError::Auth(_))
        )
    }
}

/// Send the handshake and move to `next`.
///
/// `address`/`port` are echoed back to the server as the client typed them, not as they
/// resolved; some proxies route on this. java: `C00Handshake`
///
/// # Errors
///
/// [`ProtoError::IllegalStateTransition`] if the connection is not in [`State::Handshaking`],
/// plus any write error.
pub async fn handshake<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    address: &str,
    port: u16,
    next: NextState,
) -> Result<(), ProtoError> {
    let target = match next {
        NextState::Handshaking => State::Handshaking,
        NextState::Play => State::Play,
        NextState::Status => State::Status,
        NextState::Login => State::Login,
    };
    if !transition_is_legal(conn.state(), target) {
        return Err(ProtoError::IllegalStateTransition {
            from: conn.state(),
            to: target,
        });
    }
    conn.send(&Handshake {
        protocol_version: PROTOCOL_VERSION,
        server_address: address.to_owned(),
        server_port: port,
        next_state: next,
    })
    .await?;
    conn.set_state(target)
}

/// Run the status protocol on an already-handshaked connection: `Request` → `Response` →
/// `Ping` → `Pong`.
///
/// # Errors
///
/// [`ProtoError::UnexpectedPacket`] if the server answers with something else, plus any I/O or
/// framing error.
pub async fn status<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
) -> Result<ServerStatus, ProtoError> {
    status_with_read_timeout(conn, STATUS_READ_TIMEOUT).await
}

/// Run the status exchange with an injectable idle-read timeout.
///
/// The timeout is applied independently to the `Response` and `Pong` reads. Once `Response` has
/// decoded, a later timeout/EOF/malformed Pong still returns `Ok(ServerStatus)` with the raw JSON
/// and [`Duration::MAX`] latency, matching `OldServerPinger`'s already-committed row state.
pub async fn status_with_read_timeout<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    read_timeout: Duration,
) -> Result<ServerStatus, ProtoError> {
    if conn.state() != State::Status {
        return Err(ProtoError::IllegalStateTransition {
            from: conn.state(),
            to: State::Status,
        });
    }
    conn.send(&Request).await?;
    let (id, mut body) = read_status_packet(conn, read_timeout).await?;
    if id != Response::ID {
        return Err(ProtoError::UnexpectedPacket {
            state: State::Status,
            expected: "status Response (0x00)",
            got: id,
        });
    }
    let response = Response::read(&mut body)?;
    ensure_packet_consumed(id, &body)?;

    // java: OldServerPinger sends `Minecraft.getSystemTime()` and measures the round trip
    // itself. The value is opaque to the server, which must echo it verbatim.
    let pong_latency: Result<(Instant, Duration), ProtoError> = async {
        let payload = minecraft_system_time();
        let sent_at = Instant::now();
        conn.send(&Ping { payload }).await?;
        let (id, mut body) = read_status_packet(conn, read_timeout).await?;
        if id != Pong::ID {
            return Err(ProtoError::UnexpectedPacket {
                state: State::Status,
                expected: "status Pong (0x01)",
                got: id,
            });
        }
        let _pong = Pong::read(&mut body)?;
        ensure_packet_consumed(id, &body)?;
        // Vanilla's OldServerPinger.handlePong ignores the packet's echoed value and measures only
        // the interval between two Minecraft.getSystemTime() calls.
        Ok((sent_at, sent_at.elapsed()))
    }
    .await;

    // java: handleServerInfo commits every visible field before sending Ping and sets
    // `field_147403_d = true`. Any later disconnect/timeout/decode failure is ignored by
    // onDisconnect, leaving the response intact and `pingToServer == -1`.
    Ok(ServerStatus {
        json: response.json,
        latency: pong_latency
            .as_ref()
            .map_or(Duration::MAX, |(_, latency)| *latency),
        ping_sent_at: pong_latency.ok().map(|(sent_at, _)| sent_at),
    })
}

/// Read one status packet with Netty `ReadTimeoutHandler` semantics. Timing each transport read,
/// rather than the whole packet, matters for a fragmented response that continues making
/// progress: inbound bytes reset vanilla's idle deadline even when the frame is not complete yet.
async fn read_status_packet<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    read_timeout: Duration,
) -> Result<(i32, PacketBuffer), ProtoError> {
    let mut payload = loop {
        if let Some(payload) = unseal(&mut conn.compression, &mut conn.framer, &mut conn.read_buf)?
        {
            break payload;
        }

        match tokio::time::timeout(read_timeout, conn.fill()).await {
            Ok(result) => result?,
            Err(_) => {
                return Err(ProtoError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "status connection read timed out",
                )))
            }
        }
    };
    let id = payload.read_varint()?;
    Ok((id, payload))
}

/// A monotonic millisecond counter matching LWJGL's `Sys.getTime() * 1000 /
/// Sys.getTimerResolution()`. Its origin is intentionally arbitrary: the status payload is
/// opaque and vanilla uses a second call to the same clock, rather than the echoed value, for
/// its latency measurement.
fn minecraft_system_time() -> i64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let elapsed = ORIGIN.get_or_init(Instant::now).elapsed().as_millis();
    i64::try_from(elapsed).unwrap_or(i64::MAX)
}

fn ensure_packet_consumed(packet_id: i32, body: &PacketBuffer) -> Result<(), ProtoError> {
    if body.is_empty() {
        Ok(())
    } else {
        Err(ProtoError::TrailingBytes {
            packet_id,
            remaining: body.remaining(),
        })
    }
}

/// Connect, handshake, and ping a server in one call — everything the server-list screen needs.
///
/// # Errors
///
/// Any error from [`Connection::connect`], [`handshake`], or [`status`].
pub async fn ping_server(host: &str, port: u16) -> Result<ServerStatus, ProtoError> {
    let mut conn = Connection::connect((host, port)).await?;
    handshake(&mut conn, host, port, NextState::Status).await?;
    status(&mut conn).await
}

/// Drive the login sequence to completion.
///
/// The connection must already be in [`State::Login`] (see [`handshake`]). On success the
/// connection is left in [`State::Play`], with encryption and compression configured as the
/// server asked.
///
/// `join` is invoked with the computed server hash (hazard H3 — signed two's-complement hex)
/// **before** `C01PacketEncryptionResponse` goes out, matching
/// `NetHandlerLoginClient.handleEncryptionRequest`: the server calls `hasJoined` the instant it
/// receives the response, so registering the join afterwards is a race we would usually lose.
/// It is a parameter rather than a hard-wired `mc_auth` call so that offline-mode connections
/// and tests can supply a no-op; [`login_online`] is the wiring for the real thing.
///
/// `observe`, if given, is called once for every frame this function puts on the wire or takes
/// off it, in wire order, both directions — `LoginStart`, `EncryptionRequest`, the encryption
/// response, `SetCompression`, `LoginSuccess`, and a kicking `Disconnect` if one arrives. Each
/// call gets the frame at the same level [`Connection::send`]/[`Connection::read_payload`]
/// already operate at: a VarInt packet id followed by the packet body, decrypted and
/// decompressed for an inbound frame, not yet compressed or encrypted for an outbound one. That
/// is deliberately the same level `mc-client`'s `net::capture::CaptureSink` already records
/// `Play`-state traffic at, so a caller can feed both into the
/// same recorder and get one capture spanning `Handshaking`/`Login`/`Play` rather than one that
/// begins at `Play`.
///
/// `observe: None` is the path every existing caller took before this parameter existed, and
/// costs one pointer-null check per frame: a pure addition, not a behaviour change. The trait
/// object is `+ Send` because `login` itself ends up inside a future a caller may spawn onto a
/// multi-threaded executor (`mc-client`'s connection task does exactly this) — a bound on the
/// hook, not a runtime cost, since a closure over ordinary owned/borrowed data is `Send` for
/// free.
///
/// # Errors
///
/// [`ProtoError::Disconnected`] carrying the server's reason JSON if we are kicked,
/// [`ProtoError::UnexpectedPacket`] for a packet id that has no meaning in `Login`, plus
/// anything `join` returns.
pub async fn login<S, F, Fut>(
    conn: &mut Connection<S>,
    username: &str,
    join: F,
    observe: LoginObserver<'_>,
) -> Result<LoginOutcome, ProtoError>
where
    S: AsyncRead + AsyncWrite + Unpin,
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = Result<(), ProtoError>>,
{
    login_with_auth_policy(conn, username, join, observe, LoginAuthPolicy::Strict).await
}

/// Drive login with an explicit session-server join failure policy.
///
/// This is the policy-bearing form of [`login`].  It exists so a caller that owns vanilla's
/// server-list `isOnLAN` bit can reproduce the LAN exception without weakening ordinary remote
/// login or swallowing transport/protocol errors.
pub async fn login_with_auth_policy<S, F, Fut>(
    conn: &mut Connection<S>,
    username: &str,
    join: F,
    mut observe: LoginObserver<'_>,
    auth_policy: LoginAuthPolicy,
) -> Result<LoginOutcome, ProtoError>
where
    S: AsyncRead + AsyncWrite + Unpin,
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = Result<(), ProtoError>>,
{
    if conn.state() != State::Login {
        return Err(ProtoError::IllegalStateTransition {
            from: conn.state(),
            to: State::Login,
        });
    }
    send_observed(
        conn,
        &LoginStart {
            username: username.to_owned(),
        },
        &mut observe,
    )
    .await?;

    let mut join = Some(join);
    loop {
        // Not `read_packet`: that splits the id off before we get a look at the buffer, and
        // `PacketBuffer::as_slice` after the split would be missing the id varint the observer
        // needs. Read the whole payload, hand it to `observe`, *then* split the id off locally.
        let mut payload = conn.read_payload().await?;
        if let Some(cb) = observe.as_deref_mut() {
            cb(Direction::Clientbound, payload.as_slice());
        }
        let id = payload.read_varint()?;
        let mut body = payload;
        match id {
            Disconnect::ID => {
                let packet = Disconnect::read(&mut body)?;
                ensure_packet_consumed(id, &body)?;
                return Err(ProtoError::Disconnected {
                    reason: packet.reason,
                });
            }
            EncryptionRequest::ID => {
                let request = EncryptionRequest::read(&mut body)?;
                ensure_packet_consumed(id, &body)?;
                let join = join.take().ok_or(ProtoError::AlreadyEncrypted)?;
                let exchange = begin_encryption(&request, SharedSecret::generate())?;
                if let Err(error) = join(exchange.server_hash).await {
                    if !auth_policy.tolerates_join_error(&error) {
                        return Err(error);
                    }
                }
                send_observed(conn, &exchange.response, &mut observe).await?;
                // Strictly after the response is on the wire — see hazard H4.
                conn.enable_encryption(&exchange.secret)?;
            }
            SetCompression::ID => {
                let packet = SetCompression::read(&mut body)?;
                ensure_packet_consumed(id, &body)?;
                // java: NetHandlerLoginClient.handleEnableCompression -> setCompressionTreshold.
                // With both handlers already installed, a non-negative threshold reaches the bad
                // decoder-to-encoder cast; the channel closes before anything else is read.
                if packet.threshold >= 0 && conn.compression_threshold().is_some() {
                    return Err(ProtoError::CompressionReconfigured);
                }
                conn.set_compression_threshold(packet.threshold);
            }
            LoginSuccess::ID => {
                let packet = LoginSuccess::read(&mut body)?;
                ensure_packet_consumed(id, &body)?;
                conn.set_state(State::Play)?;
                return Ok(LoginOutcome {
                    uuid: packet.uuid,
                    username: packet.username,
                    compression_threshold: conn.compression_threshold(),
                    encrypted: conn.is_encrypted(),
                });
            }
            other => {
                return Err(ProtoError::UnexpectedPacket {
                    state: State::Login,
                    expected: "a Login-state packet (0x00-0x03)",
                    got: other,
                })
            }
        }
    }
}

/// Serialise one packet and send it, first handing the wire bytes (VarInt id + body) to
/// [`login`]'s optional observer. Kept separate from [`Connection::send`] — which every other
/// caller in this crate uses — so that function's hot path gains neither the branch nor the
/// extra [`PacketBuffer`] construction this needs.
async fn send_observed<S, P>(
    conn: &mut Connection<S>,
    packet: &P,
    observe: &mut LoginObserver<'_>,
) -> Result<(), ProtoError>
where
    S: AsyncRead + AsyncWrite + Unpin,
    P: Packet,
{
    let mut payload = PacketBuffer::new();
    payload.write_varint(P::ID);
    packet.write(&mut payload);
    payload.finish_write()?;
    if let Some(cb) = observe.as_deref_mut() {
        cb(Direction::Serverbound, payload.as_slice());
    }
    conn.send_payload(payload.as_slice()).await
}

/// Log in against an online-mode server, registering the join with the Mojang session server.
///
/// This is the [`login`] wiring for `mc-auth`: [`mc_auth::server_hash`] is computed inside
/// [`crate::crypt::begin_encryption`] and handed to [`mc_auth::join_server`] here.
///
/// `observe` is threaded straight through to [`login`]; see its docs.
///
/// # Errors
///
/// As [`login`], plus [`ProtoError::Auth`] wrapping the session-server failure — a
/// [`mc_auth::AuthError::TokenRejected`] here is the caller's cue to refresh and retry once.
pub async fn login_online<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    session: &mc_auth::Session,
    observe: LoginObserver<'_>,
) -> Result<LoginOutcome, ProtoError> {
    login_online_with_auth_policy(conn, session, observe, LoginAuthPolicy::Strict).await
}

/// Online-mode login with an explicit LAN session-server failure policy.
///
/// The caller must pass [`LoginAuthPolicy::AllowLanAuthFailure`] only when its server-list entry
/// is marked LAN, matching vanilla's `ServerData.isOnLAN()` guard.  All other errors retain the
/// normal strict behavior.
pub async fn login_online_with_auth_policy<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    session: &mc_auth::Session,
    observe: LoginObserver<'_>,
    auth_policy: LoginAuthPolicy,
) -> Result<LoginOutcome, ProtoError> {
    login_online_with_auth_policy_and_proxy(conn, session, observe, auth_policy, None).await
}

/// Online-mode login with LAN policy and an optional Java-compatible SOCKS5 session proxy.
///
/// The proxy is used only for the Mojang session-server join request; the Minecraft game socket
/// remains on its ordinary direct transport, matching vanilla's `NetworkManager` construction.
pub async fn login_online_with_auth_policy_and_proxy<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    session: &mc_auth::Session,
    observe: LoginObserver<'_>,
    auth_policy: LoginAuthPolicy,
    proxy: Option<&mc_auth::SocksProxy>,
) -> Result<LoginOutcome, ProtoError> {
    let username = session.username.clone();
    login_with_auth_policy(
        conn,
        &username,
        move |server_hash| async move {
            mc_auth::join_server_with_proxy(session, &server_hash, proxy)
                .await
                .map_err(ProtoError::from)
        },
        observe,
        auth_policy,
    )
    .await
}

/// Log in without contacting the session server.
///
/// Correct for offline-mode (`online-mode=false`) servers, which never send
/// `EncryptionRequest` at all. If such a server *does* ask for encryption, the key exchange
/// still happens — the secret and verify token are real — we simply never register the join, so
/// an online-mode server will reject us at `hasJoined`. That is the honest behaviour: the
/// failure surfaces as the server's own kick message rather than as a silent hang.
///
/// `observe` is threaded straight through to [`login`]; see its docs.
///
/// # Errors
///
/// As [`login`].
pub async fn login_offline<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    username: &str,
    observe: LoginObserver<'_>,
) -> Result<LoginOutcome, ProtoError> {
    login(conn, username, |_server_hash| async { Ok(()) }, observe).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::login::EncryptionResponse;
    use rand::rngs::OsRng;
    use rsa::pkcs8::EncodePublicKey;
    use rsa::{Pkcs1v15Encrypt, RsaPrivateKey};

    struct InvalidNbtPacket;

    impl Packet for InvalidNbtPacket {
        const ID: i32 = 0;

        fn read(_buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
            Ok(Self)
        }

        fn write(&self, buf: &mut PacketBuffer) {
            let mut compound = mc_nbt::NbtCompound::new();
            compound.insert("x".repeat(65_536), mc_nbt::Nbt::Byte(1));
            buf.write_nbt_compound_opt(Some(&compound));
        }
    }

    #[tokio::test]
    async fn a_packet_with_unencodable_nbt_fails_the_send_without_panicking() {
        let (stream, _peer) = tokio::io::duplex(64);
        let mut conn = Connection::new(stream);
        assert!(matches!(
            conn.send(&InvalidNbtPacket).await,
            Err(ProtoError::NbtEncode(_))
        ));
    }

    /// `NetworkManager.dispatchPacket` changes the channel's state from the outgoing packet
    /// class.  In particular, vanilla queues C00Handshake and C00PacketLoginStart back-to-back:
    /// the latter is what selects LOGIN for its own encoder, not an out-of-band state setter.
    #[tokio::test]
    async fn outgoing_packet_selects_its_registered_state_before_encoding() {
        let (client_stream, server_stream) = tokio::io::duplex(1024);
        let mut client = Connection::new(client_stream);
        let mut server = Connection::new(server_stream);

        client
            .send(&Handshake {
                protocol_version: PROTOCOL_VERSION,
                server_address: "example.test".to_owned(),
                server_port: 25565,
                next_state: NextState::Login,
            })
            .await
            .unwrap();
        assert_eq!(client.state(), State::Handshaking);
        assert_eq!(server.read_packet().await.unwrap().0, Handshake::ID);

        client
            .send(&LoginStart {
                username: "Notch".to_owned(),
            })
            .await
            .unwrap();
        assert_eq!(client.state(), State::Login);
        let (id, mut body) = server.read_packet().await.unwrap();
        assert_eq!(id, LoginStart::ID);
        assert_eq!(LoginStart::read(&mut body).unwrap().username, "Notch");
    }

    #[test]
    fn only_vanillas_transitions_are_legal() {
        assert!(transition_is_legal(State::Handshaking, State::Status));
        assert!(transition_is_legal(State::Handshaking, State::Login));
        assert!(transition_is_legal(State::Login, State::Play));

        assert!(!transition_is_legal(State::Handshaking, State::Play));
        assert!(!transition_is_legal(State::Status, State::Login));
        assert!(!transition_is_legal(State::Status, State::Play));
        assert!(!transition_is_legal(State::Play, State::Login));
        assert!(!transition_is_legal(State::Login, State::Status));
        for state in [State::Handshaking, State::Status, State::Login, State::Play] {
            assert!(!transition_is_legal(state, state), "{state:?} -> itself");
        }
    }

    #[test]
    fn connection_address_selection_matches_java_ipv4_stack() {
        let ipv4 = SocketAddr::from(([127, 0, 0, 1], 25565));
        let ipv6 = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], 25565));

        assert_eq!(ipv4_only([ipv6, ipv4]), vec![ipv4]);
        assert!(ipv4_only([ipv6]).is_empty());
    }

    /// java: `NetHandlerLoginClient` tolerates every `AuthenticationException` only inside the
    /// `ServerData.isOnLAN()` branch; a socket/protocol error is never converted into that LAN
    /// exception and must remain fatal even when the caller selected the LAN policy.
    #[test]
    fn lan_auth_policy_only_tolerates_session_server_failures() {
        let auth = ProtoError::Auth(mc_auth::AuthError::TokenRejected);
        let http = ProtoError::Auth(mc_auth::AuthError::JoinRefused {
            status: 403,
            error: "InsufficientPrivilegesException".to_owned(),
            message: Some("Multiplayer is disabled".to_owned()),
        });
        let unreadable = ProtoError::Auth(mc_auth::AuthError::JoinUnreadable {
            status: 503,
            detail: "an empty body".to_owned(),
        });
        let transport = ProtoError::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "socket reset",
        ));

        assert!(!LoginAuthPolicy::Strict.tolerates_join_error(&auth));
        assert!(LoginAuthPolicy::AllowLanAuthFailure.tolerates_join_error(&auth));
        assert!(LoginAuthPolicy::AllowLanAuthFailure.tolerates_join_error(&http));
        assert!(LoginAuthPolicy::AllowLanAuthFailure.tolerates_join_error(&unreadable));
        assert!(!LoginAuthPolicy::AllowLanAuthFailure.tolerates_join_error(&transport));
    }

    /// java: the LAN branch logs an `AuthenticationException` and still queues the encryption
    /// response. This wire-level control proves the tolerated join error does not skip C01 or
    /// leave the connection in Login once the server answers with LoginSuccess.
    #[tokio::test]
    async fn lan_auth_failure_continues_the_encryption_exchange() {
        let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("test RSA keygen");
        let public_key_der = private
            .to_public_key()
            .to_public_key_der()
            .expect("SPKI DER")
            .as_bytes()
            .to_vec();
        let (client_stream, server_stream) = tokio::io::duplex(1 << 16);
        let mut client = Connection::new(client_stream);
        let mut server = Connection::new(server_stream);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let server_task = tokio::spawn(async move {
            let (id, mut body) = server.read_packet().await.unwrap();
            assert_eq!(id, LoginStart::ID);
            assert_eq!(LoginStart::read(&mut body).unwrap().username, "Notch");
            server
                .send(&EncryptionRequest {
                    server_id: String::new(),
                    public_key_der,
                    verify_token: vec![0xCA, 0xFE, 0xBA, 0xBE],
                })
                .await
                .unwrap();

            let (id, mut body) = server.read_packet().await.unwrap();
            assert_eq!(id, EncryptionResponse::ID);
            let response = EncryptionResponse::read(&mut body).unwrap();
            let secret = private
                .decrypt(Pkcs1v15Encrypt, &response.shared_secret_encrypted)
                .unwrap();
            let secret: [u8; 16] = secret.try_into().expect("AES-128 secret");
            server
                .enable_encryption(&SharedSecret::from_bytes(secret))
                .unwrap();
            server
                .send(&LoginSuccess {
                    uuid: Uuid::from_u128(0x1234),
                    username: "Notch".to_owned(),
                })
                .await
                .unwrap();
        });

        let outcome = login_with_auth_policy(
            &mut client,
            "Notch",
            |_hash| async { Err(ProtoError::Auth(mc_auth::AuthError::TokenRejected)) },
            None,
            LoginAuthPolicy::AllowLanAuthFailure,
        )
        .await
        .expect("LAN auth failure must not abort login");
        assert_eq!(client.state(), State::Play);
        assert!(outcome.encrypted);
        server_task.await.unwrap();
    }

    #[test]
    fn set_state_rejects_illegal_moves() {
        let mut conn = Connection::new(tokio::io::duplex(64).0);
        assert!(matches!(
            conn.set_state(State::Play),
            Err(ProtoError::IllegalStateTransition {
                from: State::Handshaking,
                to: State::Play
            })
        ));
        conn.set_state(State::Login).unwrap();
        assert_eq!(conn.state(), State::Login);
        conn.set_state(State::Play).unwrap();
        assert_eq!(conn.state(), State::Play);
    }

    #[test]
    fn encryption_cannot_be_enabled_twice() {
        let mut conn = Connection::new(tokio::io::duplex(64).0);
        let secret = SharedSecret::generate();
        assert!(!conn.is_encrypted());
        conn.enable_encryption(&secret).unwrap();
        assert!(conn.is_encrypted());
        assert!(matches!(
            conn.enable_encryption(&secret),
            Err(ProtoError::AlreadyEncrypted)
        ));
    }

    #[test]
    fn compression_threshold_tracks_set_compression() {
        let mut conn = Connection::new(tokio::io::duplex(64).0);
        assert_eq!(conn.compression_threshold(), None);
        conn.set_compression_threshold(256);
        assert_eq!(conn.compression_threshold(), Some(256));
        conn.set_compression_threshold(-1);
        assert_eq!(conn.compression_threshold(), None);
    }

    /// Two `Connection`s wired back to back must agree, in every combination of the two
    /// optional layers. This is the layering contract in miniature.
    #[tokio::test]
    async fn back_to_back_round_trips_in_every_layer_combination() {
        for encrypted in [false, true] {
            for threshold in [None, Some(0i32), Some(4), Some(1024)] {
                let (a, b) = tokio::io::duplex(1 << 20);
                let mut client = Connection::new(a);
                let mut server = Connection::new(b);
                if encrypted {
                    let secret = SharedSecret::generate();
                    client.enable_encryption(&secret).unwrap();
                    server.enable_encryption(&secret).unwrap();
                }
                if let Some(t) = threshold {
                    client.set_compression_threshold(t);
                    server.set_compression_threshold(t);
                }

                // Several packets in a row, spanning the threshold in both directions: the
                // second and later packets are where a re-initialised CFB8 stream would break.
                for n in [1usize, 3, 100, 2, 5000, 7] {
                    let payload: Vec<u8> = (0..n).map(|i| (i % 253) as u8).collect();
                    client.send_payload(&payload).await.unwrap();
                    let got = server.read_payload().await.unwrap();
                    assert_eq!(
                        got.as_slice(),
                        &payload[..],
                        "encrypted={encrypted} threshold={threshold:?} len={n}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn zero_length_frames_are_skipped_with_and_without_compression() {
        for threshold in [None, Some(4)] {
            let (stream, mut peer) = tokio::io::duplex(64);
            let mut connection = Connection::new(stream);
            let wire: &[u8] = if let Some(threshold) = threshold {
                connection.set_compression_threshold(threshold);
                // Two zero-length outer frames, then `[raw marker 0, packet byte 42]`.
                &[0, 0, 2, 0, 42]
            } else {
                // Two zero-length outer frames, then one uncompressed packet byte.
                &[0, 0, 1, 42]
            };
            peer.write_all(wire).await.unwrap();

            assert_eq!(connection.read_payload().await.unwrap().as_slice(), &[42]);
        }
    }

    #[tokio::test]
    async fn marker_only_compressed_frames_are_skipped_like_java() {
        let (stream, mut peer) = tokio::io::duplex(64);
        let mut connection = Connection::new(stream);
        connection.set_compression_threshold(4);

        // Outer frame 1 contains only the raw/uncompressed marker `[0]`; the following frame is
        // a normal below-threshold payload. Java's MessageDeserializer silently drops the first.
        peer.write_all(&[1, 0, 2, 0, 42]).await.unwrap();

        assert_eq!(connection.read_payload().await.unwrap().as_slice(), &[42]);
    }

    #[tokio::test]
    async fn split_reader_skips_marker_only_compressed_frames() {
        let (stream, mut peer) = tokio::io::duplex(64);
        let mut connection = Connection::new(stream);
        connection.set_compression_threshold(4);
        let (mut reader, _writer) = connection.split();

        peer.write_all(&[1, 0, 2, 0, 42]).await.unwrap();

        assert_eq!(reader.read_payload().await.unwrap().as_slice(), &[42]);
    }

    #[tokio::test]
    async fn split_reader_also_skips_zero_length_frames() {
        let (stream, mut peer) = tokio::io::duplex(64);
        let connection = Connection::new(stream);
        let (mut reader, _writer) = connection.split();
        peer.write_all(&[0, 1, 42]).await.unwrap();

        assert_eq!(reader.read_payload().await.unwrap().as_slice(), &[42]);
    }

    /// The split halves must carry the mid-flight CFB8 state across the split, and must be
    /// usable from two tasks at once — the shape S14's connection task needs.
    #[tokio::test]
    async fn split_halves_carry_the_cipher_state_and_run_concurrently() {
        let (a, b) = tokio::io::duplex(1 << 20);
        let secret = SharedSecret::generate();
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.enable_encryption(&secret).unwrap();
        server.enable_encryption(&secret).unwrap();
        client.set_compression_threshold(16);
        server.set_compression_threshold(16);

        // Get both streams off byte zero before splitting, so a split that reset them would fail.
        client.send_payload(&[0x00, 0x01]).await.unwrap();
        assert_eq!(server.read_payload().await.unwrap().as_slice(), &[0, 1]);
        server.send_payload(&[0x02, 0x03]).await.unwrap();
        assert_eq!(client.read_payload().await.unwrap().as_slice(), &[2, 3]);

        let (mut reader, mut writer) = client.split();
        assert_eq!(reader.compression_threshold(), Some(16));
        assert_eq!(writer.compression_threshold(), Some(16));

        let payloads: Vec<Vec<u8>> = [1usize, 100, 2, 4000]
            .iter()
            .map(|&n| (0..n).map(|i| (i % 251) as u8).collect())
            .collect();

        let server_task = tokio::spawn(async move {
            for _ in 0..4 {
                let echo = server.read_payload().await.unwrap();
                server.send_payload(echo.as_slice()).await.unwrap();
            }
        });
        let to_send = payloads.clone();
        let write_task = tokio::spawn(async move {
            for payload in &to_send {
                writer.send_payload(payload).await.unwrap();
            }
        });

        for payload in &payloads {
            assert_eq!(
                reader.read_payload().await.unwrap().as_slice(),
                &payload[..]
            );
        }
        write_task.await.unwrap();
        server_task.await.unwrap();
    }

    /// A typed packet through the full pipeline, both directions, with both layers on.
    #[tokio::test]
    async fn typed_packets_survive_the_full_pipeline() {
        let (a, b) = tokio::io::duplex(1 << 16);
        let secret = SharedSecret::generate();
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.enable_encryption(&secret).unwrap();
        server.enable_encryption(&secret).unwrap();
        client.set_compression_threshold(8);
        server.set_compression_threshold(8);

        let sent = LoginSuccess {
            uuid: Uuid::from_u128(0x1234_5678_9abc_def0_1234_5678_9abc_def0),
            username: "Notch".to_owned(),
        };
        client.send(&sent).await.unwrap();
        let (id, mut body) = server.read_packet().await.unwrap();
        assert_eq!(id, LoginSuccess::ID);
        assert_eq!(LoginSuccess::read(&mut body).unwrap(), sent);
    }

    /// A peer that hangs up mid-frame must produce a typed error, not a hang or a panic.
    #[tokio::test]
    async fn closed_transport_is_a_typed_error() {
        let (a, b) = tokio::io::duplex(64);
        drop(b);
        let mut conn = Connection::new(a);
        assert!(matches!(
            conn.read_payload().await,
            Err(ProtoError::ConnectionClosed)
        ));
    }

    #[tokio::test]
    async fn handshake_rejects_a_bad_starting_state() {
        let (a, _b) = tokio::io::duplex(1024);
        let mut conn = Connection::new(a);
        conn.set_state(State::Login).unwrap();
        assert!(matches!(
            handshake(&mut conn, "example.com", 25565, NextState::Status).await,
            Err(ProtoError::IllegalStateTransition { .. })
        ));
    }

    #[tokio::test]
    async fn login_requires_the_login_state() {
        let (a, _b) = tokio::io::duplex(1024);
        let mut conn = Connection::new(a);
        assert!(matches!(
            login_offline(&mut conn, "Notch", None).await,
            Err(ProtoError::IllegalStateTransition { .. })
        ));
    }

    /// The disconnect reason must reach the caller intact — S14 shows it on the kick screen.
    #[tokio::test]
    async fn login_surfaces_the_disconnect_reason() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let reason = r#"{"translate":"multiplayer.disconnect.outdated_client"}"#;
        let task = tokio::spawn(async move {
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, LoginStart::ID);
            server
                .send(&Disconnect {
                    reason: reason.to_owned(),
                })
                .await
                .unwrap();
        });

        let err = login_offline(&mut client, "Notch", None).await.unwrap_err();
        match err {
            ProtoError::Disconnected { reason: got } => assert_eq!(got, reason),
            other => panic!("expected Disconnected, got {other:?}"),
        }
        task.await.unwrap();
    }

    #[tokio::test]
    async fn login_rejects_a_play_packet_id() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();

        let task = tokio::spawn(async move {
            let _ = server.read_packet().await.unwrap();
            // 0x40 is a legal *Play* id and has no meaning in Login.
            server.send_payload(&[0x40, 0x00]).await.unwrap();
        });
        assert!(matches!(
            login_offline(&mut client, "Notch", None).await,
            Err(ProtoError::UnexpectedPacket {
                state: State::Login,
                got: 0x40,
                ..
            })
        ));
        task.await.unwrap();
    }

    #[tokio::test]
    async fn login_has_no_artificial_packet_count_limit() {
        let (a, b) = tokio::io::duplex(1 << 16);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let uuid = Uuid::from_u128(0x1234);
        let task = tokio::spawn(async move {
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, LoginStart::ID);

            // Many Login packets, each one legal for vanilla: alternating install/remove never
            // reaches setCompressionTreshold's casting update branch (a repeated non-negative
            // threshold does; see the test after this one).
            for _ in 0..33 {
                for threshold in [256, -1] {
                    server.send(&SetCompression { threshold }).await.unwrap();
                    server.set_compression_threshold(threshold);
                }
            }
            server
                .send(&SetCompression { threshold: 256 })
                .await
                .unwrap();
            server.set_compression_threshold(256);
            server
                .send(&LoginSuccess {
                    uuid,
                    username: "Notch".to_owned(),
                })
                .await
                .unwrap();
        });

        let outcome = login_offline(&mut client, "Notch", None).await.unwrap();
        assert_eq!(outcome.uuid, uuid);
        assert_eq!(outcome.compression_threshold, Some(256));
        task.await.unwrap();
    }

    #[tokio::test]
    async fn login_rejects_trailing_bytes_before_dispatching_success() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let task = tokio::spawn(async move {
            let _ = server.read_packet().await.unwrap();
            let mut payload = PacketBuffer::new();
            payload.write_varint(LoginSuccess::ID);
            LoginSuccess {
                uuid: Uuid::from_u128(0x1234),
                username: "Notch".to_owned(),
            }
            .write(&mut payload);
            payload.write_u8(0xAA);
            server.send_payload(payload.as_slice()).await.unwrap();
        });

        assert!(matches!(
            login_offline(&mut client, "Notch", None).await,
            Err(ProtoError::TrailingBytes {
                packet_id: LoginSuccess::ID,
                remaining: 1
            })
        ));
        assert_eq!(client.state(), State::Login);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn login_rejects_trailing_bytes_before_enabling_compression() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let task = tokio::spawn(async move {
            let _ = server.read_packet().await.unwrap();
            let mut payload = PacketBuffer::new();
            payload.write_varint(SetCompression::ID);
            SetCompression { threshold: 256 }.write(&mut payload);
            payload.write_u8(0xAA);
            server.send_payload(payload.as_slice()).await.unwrap();
        });

        assert!(matches!(
            login_offline(&mut client, "Notch", None).await,
            Err(ProtoError::TrailingBytes {
                packet_id: SetCompression::ID,
                remaining: 1
            })
        ));
        assert_eq!(client.state(), State::Login);
        assert_eq!(client.compression_threshold(), None);
        task.await.unwrap();
    }

    /// java: a second non-negative Login S03 reaches
    /// `NetworkManager.setCompressionTreshold`'s decoder-to-encoder cast (`:467-469`), and the
    /// `ClassCastException` closes the channel. Rust used to re-apply the threshold and log in.
    #[tokio::test]
    async fn login_closes_on_a_second_nonnegative_set_compression_like_javas_cast() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let task = tokio::spawn(async move {
            let _ = server.read_packet().await.unwrap();
            server
                .send(&SetCompression { threshold: 256 })
                .await
                .unwrap();
            server.set_compression_threshold(256);
            server
                .send(&SetCompression { threshold: 512 })
                .await
                .unwrap();
            server.set_compression_threshold(512);
            // A vanilla client never gets this far; an old Rust client would accept it.
            let _ = server
                .send(&LoginSuccess {
                    uuid: Uuid::nil(),
                    username: "Notch".into(),
                })
                .await;
        });

        assert!(matches!(
            login_offline(&mut client, "Notch", None).await,
            Err(ProtoError::CompressionReconfigured)
        ));
        assert_eq!(client.state(), State::Login);
        task.await.unwrap();
    }

    /// A negative threshold removes both handlers (`:476-487`), so a later non-negative one is a
    /// fresh install, not the casting update branch.
    #[tokio::test]
    async fn login_reinstalls_compression_after_a_negative_threshold_removed_it() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let task = tokio::spawn(async move {
            let _ = server.read_packet().await.unwrap();
            for threshold in [256, -1, 64] {
                server.send(&SetCompression { threshold }).await.unwrap();
                server.set_compression_threshold(threshold);
            }
            server
                .send(&LoginSuccess {
                    uuid: Uuid::nil(),
                    username: "Notch".into(),
                })
                .await
                .unwrap();
        });

        let outcome = login_offline(&mut client, "Notch", None)
            .await
            .expect("remove-then-install is not the casting branch");
        assert_eq!(outcome.compression_threshold, Some(64));
        assert_eq!(client.state(), State::Play);
        task.await.unwrap();
    }

    /// `login`'s `observe` hook must see every frame the sequence
    /// puts on the wire, in order, tagged with the right [`Direction`] — proven here for the
    /// offline (no encryption, no compression) shape, which is fast enough to be a unit test
    /// rather than a loopback one. `mc-protocol/tests/login_loopback.rs` proves the same claim
    /// for the online-mode-shaped sequence, over real TCP with real RSA and a real AES-CFB8
    /// stream, which is where hazards H3/H4 actually live.
    #[tokio::test]
    async fn login_rejects_trailing_bytes_in_a_disconnect_body() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let task = tokio::spawn(async move {
            let _ = server.read_packet().await.unwrap();
            let mut payload = PacketBuffer::new();
            payload.write_varint(Disconnect::ID);
            Disconnect {
                reason: r#"{"text":"kicked"}"#.to_owned(),
            }
            .write(&mut payload);
            payload.write_u8(0xAA);
            server.send_payload(payload.as_slice()).await.unwrap();
        });

        assert!(matches!(
            login_offline(&mut client, "Notch", None).await,
            Err(ProtoError::TrailingBytes {
                packet_id: Disconnect::ID,
                remaining: 1
            })
        ));
        assert_eq!(client.state(), State::Login);
        task.await.unwrap();
    }

    /// `login`'s `observe` hook must see every frame the sequence
    /// puts on the wire, in order, tagged with the right [`Direction`] — proven here for the
    /// offline (no encryption, no compression) shape, which is fast enough to be a unit test
    /// rather than a loopback one. `mc-protocol/tests/login_loopback.rs` proves the same claim
    /// for the online-mode-shaped sequence, over real TCP with real RSA and a real AES-CFB8
    /// stream, which is where hazards H3/H4 actually live.
    #[tokio::test]
    async fn login_offline_reports_start_and_success_to_the_observer() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Login).unwrap();
        server.set_state(State::Login).unwrap();

        let uuid = Uuid::from_u128(0xdead_beef_dead_beef_dead_beef_dead_beef);
        let task = tokio::spawn(async move {
            let (id, mut body) = server.read_packet().await.unwrap();
            assert_eq!(id, LoginStart::ID);
            assert_eq!(LoginStart::read(&mut body).unwrap().username, "Notch");
            server
                .send(&LoginSuccess {
                    uuid,
                    username: "Notch".to_owned(),
                })
                .await
                .unwrap();
        });

        // Every frame the hook reports, decoded just enough to check direction and id — proving
        // the bytes are real packet frames (VarInt id + body), not some other shape.
        let mut frames: Vec<(Direction, i32)> = Vec::new();
        let mut observe = |direction: Direction, bytes: &[u8]| {
            let mut buf = PacketBuffer::from_slice(bytes);
            frames.push((direction, buf.read_varint().unwrap()));
        };

        let outcome = login_offline(&mut client, "Notch", Some(&mut observe))
            .await
            .unwrap();
        assert_eq!(outcome.uuid, uuid);

        assert_eq!(
            frames,
            vec![
                (Direction::Serverbound, LoginStart::ID),
                (Direction::Clientbound, LoginSuccess::ID),
            ],
            "the hook must see exactly LoginStart out then LoginSuccess back, in that order"
        );
        task.await.unwrap();
    }

    #[tokio::test]
    async fn status_round_trips_against_a_scripted_server() {
        let (a, b) = tokio::io::duplex(1 << 16);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        let json = r#"{"version":{"name":"1.8.9","protocol":47}}"#;

        let task = tokio::spawn(async move {
            let (id, mut body) = server.read_packet().await.unwrap();
            assert_eq!(id, Handshake::ID);
            let hs = Handshake::read(&mut body).unwrap();
            assert_eq!(hs.protocol_version, PROTOCOL_VERSION);
            assert_eq!(hs.next_state, NextState::Status);
            server.set_state(State::Status).unwrap();

            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Request::ID);
            server
                .send(&Response {
                    json: json.to_owned(),
                })
                .await
                .unwrap();

            let (id, mut body) = server.read_packet().await.unwrap();
            assert_eq!(id, Ping::ID);
            let ping = Ping::read(&mut body).unwrap();
            assert!(
                ping.payload < 86_400_000,
                "status ping must use a monotonic uptime counter, not Unix-epoch milliseconds"
            );
            server
                .send(&Pong {
                    payload: ping.payload,
                })
                .await
                .unwrap();
        });

        handshake(&mut client, "example.com", 25565, NextState::Status)
            .await
            .unwrap();
        let got = status(&mut client).await.unwrap();
        assert_eq!(got.json, json);
        assert!(
            got.ping_sent_at.is_some(),
            "a completed Pong retains its send instant for the client-tick receipt boundary"
        );
        task.await.unwrap();
    }

    #[tokio::test]
    async fn status_ignores_a_mismatched_pong_payload_like_vanilla() {
        let (a, b) = tokio::io::duplex(1 << 16);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);

        let task = tokio::spawn(async move {
            let _ = server.read_packet().await.unwrap();
            server.set_state(State::Status).unwrap();
            let _ = server.read_packet().await.unwrap();
            server
                .send(&Response {
                    json: "{}".to_owned(),
                })
                .await
                .unwrap();
            let (id, mut body) = server.read_packet().await.unwrap();
            assert_eq!(id, Ping::ID);
            let ping = Ping::read(&mut body).unwrap();
            server
                .send(&Pong {
                    payload: ping.payload.wrapping_add(1),
                })
                .await
                .unwrap();
        });

        handshake(&mut client, "example.com", 25565, NextState::Status)
            .await
            .unwrap();
        let result = status(&mut client).await.unwrap();
        assert_eq!(result.json, "{}");
        task.await.unwrap();
    }

    #[tokio::test]
    async fn status_times_out_while_waiting_for_the_response() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Status).unwrap();
        server.set_state(State::Status).unwrap();

        let task = tokio::spawn(async move {
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Request::ID);
            std::future::pending::<()>().await;
        });

        let result = tokio::time::timeout(
            Duration::from_millis(400),
            status_with_read_timeout(&mut client, Duration::from_millis(40)),
        )
        .await
        .expect("the status idle deadline must resolve the stalled read");
        assert!(matches!(
            result,
            Err(ProtoError::Io(error)) if error.kind() == std::io::ErrorKind::TimedOut
        ));
        task.abort();
    }

    #[tokio::test]
    async fn status_preserves_the_response_when_the_pong_times_out() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Status).unwrap();
        server.set_state(State::Status).unwrap();

        let task = tokio::spawn(async move {
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Request::ID);
            server
                .send(&Response {
                    json: "{}".to_owned(),
                })
                .await
                .unwrap();
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Ping::ID);
            std::future::pending::<()>().await;
        });

        let result = tokio::time::timeout(
            Duration::from_millis(400),
            status_with_read_timeout(&mut client, Duration::from_millis(40)),
        )
        .await
        .expect("the status idle deadline must resolve the stalled pong read");
        let result = result.expect("a post-response timeout is best-effort in vanilla");
        assert_eq!(result.json, "{}");
        assert_eq!(result.latency, Duration::MAX);
        assert_eq!(result.latency.as_millis() as i64, -1);
        assert!(result.ping_sent_at.is_none());
        task.abort();
    }

    #[tokio::test]
    async fn status_preserves_the_response_when_the_peer_closes_before_pong() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Status).unwrap();
        server.set_state(State::Status).unwrap();

        let task = tokio::spawn(async move {
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Request::ID);
            server
                .send(&Response {
                    json: r#"{"description":"kept"}"#.to_owned(),
                })
                .await
                .unwrap();
        });

        let result = status_with_read_timeout(&mut client, Duration::from_millis(100))
            .await
            .expect("a disconnect after Response must not discard the row data");
        assert_eq!(result.json, r#"{"description":"kept"}"#);
        assert_eq!(result.latency, Duration::MAX);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn status_preserves_the_first_response_when_a_second_response_replaces_pong() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Status).unwrap();
        server.set_state(State::Status).unwrap();

        let task = tokio::spawn(async move {
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Request::ID);
            server
                .send(&Response {
                    json: r#"{"description":"first"}"#.to_owned(),
                })
                .await
                .unwrap();
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Ping::ID);
            server
                .send(&Response {
                    json: r#"{"description":"second"}"#.to_owned(),
                })
                .await
                .unwrap();
        });

        let result = status_with_read_timeout(&mut client, Duration::from_millis(100))
            .await
            .expect("vanilla keeps the committed first response");
        assert_eq!(result.json, r#"{"description":"first"}"#);
        assert_eq!(result.latency, Duration::MAX);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn status_preserves_the_response_when_the_pong_body_is_truncated() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);
        client.set_state(State::Status).unwrap();
        server.set_state(State::Status).unwrap();

        let task = tokio::spawn(async move {
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Request::ID);
            server
                .send(&Response {
                    json: "{}".to_owned(),
                })
                .await
                .unwrap();
            let (id, _) = server.read_packet().await.unwrap();
            assert_eq!(id, Ping::ID);
            server.send_payload(&[Pong::ID as u8]).await.unwrap();
        });

        let result = status_with_read_timeout(&mut client, Duration::from_millis(100))
            .await
            .expect("a malformed Pong arrives after vanilla committed the response");
        assert_eq!(result.json, "{}");
        assert_eq!(result.latency, Duration::MAX);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn status_idle_timeout_resets_when_fragmented_bytes_keep_arriving() {
        let (a, mut b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);

        let mut payload = PacketBuffer::new();
        payload.write_varint(Response::ID);
        Response {
            json: "{}".to_owned(),
        }
        .write(&mut payload);
        let mut wire = BytesMut::new();
        PacketFrameCodec::new()
            .encode(payload.as_slice(), &mut wire)
            .unwrap();
        assert!(wire.len() > 2);

        let task = tokio::spawn(async move {
            b.write_all(&wire[..1]).await.unwrap();
            tokio::time::sleep(Duration::from_millis(55)).await;
            b.write_all(&wire[1..2]).await.unwrap();
            tokio::time::sleep(Duration::from_millis(55)).await;
            b.write_all(&wire[2..]).await.unwrap();
        });

        let started = Instant::now();
        let (id, mut body) = tokio::time::timeout(
            Duration::from_millis(500),
            read_status_packet(&mut client, Duration::from_millis(80)),
        )
        .await
        .expect("fragmented response should complete")
        .expect("each fragment arrives before the idle deadline");
        assert!(
            started.elapsed() >= Duration::from_millis(100),
            "fixture must take longer than one whole-packet deadline"
        );
        assert_eq!(id, Response::ID);
        assert_eq!(Response::read(&mut body).unwrap().json, "{}");
        task.await.unwrap();
    }

    #[tokio::test]
    async fn status_rejects_trailing_bytes_in_response_body() {
        let (a, b) = tokio::io::duplex(1 << 14);
        let mut client = Connection::new(a);
        let mut server = Connection::new(b);

        let task = tokio::spawn(async move {
            let _ = server.read_packet().await.unwrap();
            server.set_state(State::Status).unwrap();
            let _ = server.read_packet().await.unwrap();
            let mut payload = PacketBuffer::new();
            payload.write_varint(Response::ID);
            Response {
                json: "{}".to_owned(),
            }
            .write(&mut payload);
            payload.write_u8(0xAA);
            server.send_payload(payload.as_slice()).await.unwrap();
        });

        handshake(&mut client, "example.com", 25565, NextState::Status)
            .await
            .unwrap();
        assert!(matches!(
            status(&mut client).await,
            Err(ProtoError::TrailingBytes {
                packet_id: Response::ID,
                remaining: 1
            })
        ));
        task.await.unwrap();
    }
}
