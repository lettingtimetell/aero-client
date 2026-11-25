//! The protocol-5 (1.7.10) login sequence.
//!
//! Run it on a [`Connection`] that [`crate::connection::handshake_for`] has moved to
//! [`State::Login`] with `GameVersion::V1_7_10`. It differs from
//! [`crate::connection::login_with_auth_policy`] in exactly what protocol 5 differs in:
//!
//! * the login state registers only `S00 Disconnect`, `S01 EncryptionRequest` and
//!   `S02 LoginSuccess` (`EnumConnectionState.LOGIN`); there is no `0x03 SetCompression`, so
//!   `0x03`, like any other id, is "Bad packet id" in 1.7.10 and
//!   [`ProtoError::UnexpectedPacket`] here;
//! * the encryption request and response carry `i16`-prefixed byte arrays
//!   ([`crate::v5::login`]).
//!
//! Everything else is reused: `C00PacketLoginStart`, `S00PacketDisconnect` and
//! `S02PacketLoginSuccess` are byte-identical to 1.8.9's; [`crate::crypt::begin_encryption`]
//! does the key exchange; the session-server join runs before the response is sent and the
//! ciphers start after it is on the wire (`NetHandlerLoginClient.handleEncryptionRequest`, whose
//! LAN tolerance is the same as 1.8.9's); and the observer sees every frame as in [`login`].
//!
//! [`login`]: crate::connection::login
//!
//! java17: `client/network/NetHandlerLoginClient.java`, `network/EnumConnectionState.java`

use std::future::Future;

use tokio::io::{AsyncRead, AsyncWrite};

use crate::connection::{Connection, Direction, LoginAuthPolicy, LoginObserver, LoginOutcome};
use crate::crypt::{begin_encryption, SharedSecret};
use crate::login::{Disconnect, LoginStart, LoginSuccess};
use crate::packet::{Packet, State};
use crate::v5::login::{EncryptionRequest, EncryptionResponse};
use crate::{PacketBuffer, ProtoError};

/// Drive the protocol-5 login to completion.
///
/// The connection must be in [`State::Login`]; on success it is in [`State::Play`], encrypted if
/// the server asked for it, and never compressed. `join`, `observe` and `auth_policy` mean what
/// they mean for [`crate::connection::login_with_auth_policy`].
///
/// # Errors
///
/// [`ProtoError::Disconnected`] with the server's reason JSON if it kicks us,
/// [`ProtoError::UnexpectedPacket`] for an id protocol 5 does not register in the login state
/// (including `0x03`), [`ProtoError::TrailingBytes`] for a packet longer than its fields,
/// [`ProtoError::AlreadyEncrypted`] for a second encryption request, plus whatever `join`, the
/// key exchange or the transport report.
pub async fn login_v5_with_auth_policy<S, F, Fut>(
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
        let mut payload = conn.read_payload().await?;
        if let Some(cb) = observe.as_deref_mut() {
            cb(Direction::Clientbound, payload.as_slice());
        }
        let id = payload.read_varint()?;
        let mut body = payload;
        match id {
            Disconnect::ID => {
                let packet = Disconnect::read(&mut body)?;
                ensure_consumed(id, &body)?;
                return Err(ProtoError::Disconnected {
                    reason: packet.reason,
                });
            }
            EncryptionRequest::ID => {
                let request = EncryptionRequest::read(&mut body)?;
                ensure_consumed(id, &body)?;
                let join = join.take().ok_or(ProtoError::AlreadyEncrypted)?;
                let exchange = begin_encryption(&request.into(), SharedSecret::generate())?;
                if let Err(error) = join(exchange.server_hash).await {
                    let lan_tolerated = matches!(
                        (auth_policy, &error),
                        (LoginAuthPolicy::AllowLanAuthFailure, ProtoError::Auth(_))
                    );
                    if !lan_tolerated {
                        return Err(error);
                    }
                }
                let response = EncryptionResponse::from(exchange.response);
                send_observed(conn, &response, &mut observe).await?;
                // After the response is on the wire: the JDK enables the ciphers in the send's
                // completion listener.
                conn.enable_encryption(&exchange.secret)?;
            }
            LoginSuccess::ID => {
                let packet = LoginSuccess::read(&mut body)?;
                ensure_consumed(id, &body)?;
                conn.set_state(State::Play)?;
                return Ok(LoginOutcome {
                    uuid: packet.uuid,
                    username: packet.username,
                    compression_threshold: None,
                    encrypted: conn.is_encrypted(),
                });
            }
            other => {
                return Err(ProtoError::UnexpectedPacket {
                    state: State::Login,
                    expected: "a protocol-5 Login packet (0x00-0x02)",
                    got: other,
                })
            }
        }
    }
}

/// The protocol-5 login without a session-server join, for offline-mode servers (which never
/// ask for encryption). As [`crate::connection::login_offline`].
///
/// # Errors
///
/// As [`login_v5_with_auth_policy`].
pub async fn login_v5_offline<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    username: &str,
    observe: LoginObserver<'_>,
) -> Result<LoginOutcome, ProtoError> {
    login_v5_with_auth_policy(
        conn,
        username,
        |_server_hash| async { Ok(()) },
        observe,
        LoginAuthPolicy::Strict,
    )
    .await
}

fn ensure_consumed(packet_id: i32, body: &PacketBuffer) -> Result<(), ProtoError> {
    if body.is_empty() {
        Ok(())
    } else {
        Err(ProtoError::TrailingBytes {
            packet_id,
            remaining: body.remaining(),
        })
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{duplex, AsyncReadExt, AsyncWriteExt, DuplexStream};

    /// One unencrypted frame: VarInt length, then the payload.
    async fn write_frame(stream: &mut DuplexStream, payload: &[u8]) {
        let mut frame = PacketBuffer::new();
        frame.write_varint(payload.len() as i32);
        frame.write_bytes(payload);
        stream.write_all(frame.as_slice()).await.unwrap();
    }

    async fn read_frame(stream: &mut DuplexStream) -> Vec<u8> {
        let mut len = 0usize;
        for shift in (0..21).step_by(7) {
            let byte = stream.read_u8().await.unwrap();
            len |= usize::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
        }
        let mut payload = vec![0; len];
        stream.read_exact(&mut payload).await.unwrap();
        payload
    }

    fn packet_bytes<P: Packet>(packet: &P) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(P::ID);
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    fn login_conn(stream: DuplexStream) -> Connection<DuplexStream> {
        let mut conn = Connection::new(stream);
        conn.set_state(State::Login).unwrap();
        conn
    }

    #[tokio::test]
    async fn offline_login_succeeds_without_compression() {
        let (client, mut server) = duplex(4096);
        let mut conn = login_conn(client);
        let script = tokio::spawn(async move {
            let start = read_frame(&mut server).await;
            assert_eq!(
                start,
                packet_bytes(&LoginStart {
                    username: "Steve".into()
                })
            );
            let success = LoginSuccess {
                uuid: uuid::Uuid::from_u128(7),
                username: "Steve".into(),
            };
            write_frame(&mut server, &packet_bytes(&success)).await;
            server
        });
        let mut seen = Vec::new();
        let mut observe =
            |direction: Direction, bytes: &[u8]| seen.push((direction, bytes.to_vec()));
        let outcome = login_v5_offline(&mut conn, "Steve", Some(&mut observe))
            .await
            .unwrap();
        assert_eq!(outcome.username, "Steve");
        assert_eq!(outcome.compression_threshold, None);
        assert!(!outcome.encrypted);
        assert_eq!(conn.state(), State::Play);
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].0, Direction::Serverbound);
        assert_eq!(seen[1].0, Direction::Clientbound);
        drop(script.await.unwrap());
    }

    #[tokio::test]
    async fn set_compression_is_not_a_protocol_5_login_packet() {
        let (client, mut server) = duplex(4096);
        let mut conn = login_conn(client);
        let script = tokio::spawn(async move {
            read_frame(&mut server).await;
            // 1.8.9's SetCompression(256): id 3, VarInt 256.
            write_frame(&mut server, &[0x03, 0x80, 0x02]).await;
            server
        });
        let error = login_v5_offline(&mut conn, "Steve", None)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            ProtoError::UnexpectedPacket {
                state: State::Login,
                got: 3,
                ..
            }
        ));
        drop(script.await.unwrap());
    }

    #[tokio::test]
    async fn disconnect_reports_the_reason() {
        let (client, mut server) = duplex(4096);
        let mut conn = login_conn(client);
        let script = tokio::spawn(async move {
            read_frame(&mut server).await;
            let kick = Disconnect {
                reason: "{\"text\":\"Outdated client!\"}".into(),
            };
            write_frame(&mut server, &packet_bytes(&kick)).await;
            server
        });
        let error = login_v5_offline(&mut conn, "Steve", None)
            .await
            .unwrap_err();
        assert!(
            matches!(error, ProtoError::Disconnected { reason } if reason.contains("Outdated"))
        );
        drop(script.await.unwrap());
    }

    #[tokio::test]
    async fn a_1_8_9_encryption_request_is_misread_not_accepted() {
        // A VarInt-prefixed 1.8.9 request read as protocol 5: the "i16 length" spans the
        // VarInt and the first key byte, so the body cannot be consumed exactly.
        let (client, mut server) = duplex(4096);
        let mut conn = login_conn(client);
        let script = tokio::spawn(async move {
            read_frame(&mut server).await;
            let request = crate::login::EncryptionRequest {
                server_id: String::new(),
                public_key_der: vec![0x30; 162],
                verify_token: vec![1, 2, 3, 4],
            };
            write_frame(&mut server, &packet_bytes(&request)).await;
            server
        });
        assert!(login_v5_offline(&mut conn, "Steve", None).await.is_err());
        drop(script.await.unwrap());
    }

    #[tokio::test]
    async fn login_requires_the_login_state() {
        let (client, _server) = duplex(64);
        let mut conn = Connection::new(client);
        assert!(matches!(
            login_v5_offline(&mut conn, "Steve", None).await,
            Err(ProtoError::IllegalStateTransition { .. })
        ));
    }
}
