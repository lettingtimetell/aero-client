//! Step 6 — the join-time session-server call.
//!
//! S07 calls this in the middle of the encryption handshake: after generating the 16-byte AES
//! key and computing [`server_hash`](crate::server_hash), and *before* sending
//! `C01PacketEncryptionResponse`. The server then asks `hasJoined` for the same hash; if the
//! client never made this call, or made it with a different hash, the server disconnects with
//! "Failed to verify username".
//!
//! // java: NetHandlerLoginClient.handleEncryptionRequest → MinecraftSessionService.joinServer
//!
//! # How the answer is read
//!
//! Vanilla decides what a join answer means from its **body**, not its status code
//! (authlib 1.5.21, `HttpAuthenticationService.performPostRequest` and
//! `YggdrasilAuthenticationService.makeRequest`, read from the bundled jar):
//!
//! * a status of 400 or more hands over the error page; an empty one throws instead;
//! * Gson reads the text: an empty document or `null` is success;
//! * a non-blank `error` field fails the join: `ForbiddenOperationException` (or a
//!   `UserMigratedException` cause) is `InvalidCredentialsException`, shown as "Invalid
//!   session"; any other error is a plain `AuthenticationException` whose `errorMessage` is
//!   shown verbatim ("Failed to login: <message>");
//! * a response without `error` is success, whatever its status;
//! * text Gson cannot read as that object is `AuthenticationUnavailableException`, shown as
//!   "The authentication servers are currently down for maintenance."
//!
//! [`classify_join_response`] is that procedure. Before this, mc-rs decided by status alone:
//! every 401/403 became "Invalid session" (including a ban or multiplayer being switched off),
//! every other failure became one opaque line, and the real answer went only to a DEBUG log.
//!
//! Two deliberate differences from vanilla, both on failure paths only:
//! * HTTP 401 with an unreadable body is an invalid session, not "servers down": 401 is
//!   unambiguous about the credentials.
//! * A 429 or a 5xx that came back quickly is retried once after a short pause (the server's
//!   `Retry-After` when it is at most 3 seconds). The game server sees nothing but a slightly
//!   later encryption response; vanilla would have given up.
//!
//! Every failed attempt is logged at WARN with the status, Mojang's error type and message,
//! and the response's content type, never the access token.

use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::client::AuthClient;
use crate::error::{snippet, AuthError};
use crate::session::Session;

const EP_JOIN: &str = "session server join";

/// Statuses worth one quick retry: rate limiting and the gateway/availability failures.
const TRANSIENT_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

/// The longest `Retry-After` honoured. A server asking for more gets no retry: its own answer
/// goes to the screen instead, well inside the game server's 30-second login timeout.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(3);

/// A first attempt that took longer than this is not retried, so the whole join stays well
/// inside the game server's login timeout.
const RETRY_ONLY_WITHIN: Duration = Duration::from_secs(5);

/// How authlib 1.5.21 reads one join response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JoinVerdict {
    /// Gson read null, or an object without an `error`: vanilla carries on with the login.
    Accepted,
    /// authlib's `InvalidCredentialsException`. `error` is Mojang's error type, `None` for the
    /// unreadable-401 case.
    InvalidSession {
        /// Mojang's `error` field.
        error: Option<String>,
    },
    /// authlib's plain `AuthenticationException`, whose message vanilla shows.
    Refused {
        /// Mojang's `error` field.
        error: String,
        /// Mojang's `errorMessage` field.
        message: Option<String>,
    },
    /// authlib's `AuthenticationUnavailableException`: nothing it could read came back.
    Unreadable {
        /// What came back instead, in a few words.
        what: &'static str,
    },
}

/// Classify one join response exactly as authlib 1.5.21 would (see the module docs).
pub(crate) fn classify_join_response(status: u16, body: &str) -> JoinVerdict {
    // java: performPostRequest returns the error page of a >= 400 response, or rethrows its
    // IOException when there is none. Gson reads an empty (or blank) document as null, which
    // makeRequest returns as success.
    if body.trim().is_empty() {
        return if status >= 400 {
            unreadable(status, "an empty body")
        } else {
            JoinVerdict::Accepted
        };
    }
    let fields = match serde_json::from_str::<Value>(body) {
        Ok(Value::Null) => return JoinVerdict::Accepted,
        Ok(Value::Object(fields)) => fields,
        // java: Gson's reflective adapter wants BEGIN_OBJECT -> JsonSyntaxException.
        Ok(_) => return unreadable(status, "JSON that is not an object"),
        Err(_) => return unreadable(status, "a body that is not JSON"),
    };
    let (Some(error), Some(message), Some(cause)) = (
        text_field(&fields, "error"),
        text_field(&fields, "errorMessage"),
        text_field(&fields, "cause"),
    ) else {
        return unreadable(status, "a Yggdrasil field that is not text");
    };
    // java: StringUtils.isNotBlank(result.getError()). Without an error, makeRequest returns
    // the response and joinServer succeeds, whatever the HTTP status was.
    let Some(error) = error.filter(|error| !error.trim().is_empty()) else {
        return JoinVerdict::Accepted;
    };
    if cause.as_deref() == Some("UserMigratedException") || error == "ForbiddenOperationException" {
        return JoinVerdict::InvalidSession { error: Some(error) };
    }
    JoinVerdict::Refused { error, message }
}

/// The unreadable verdict, except that an HTTP 401 is an invalid session (module docs).
fn unreadable(status: u16, what: &'static str) -> JoinVerdict {
    if status == 401 {
        JoinVerdict::InvalidSession { error: None }
    } else {
        JoinVerdict::Unreadable { what }
    }
}

/// One field read the way Gson's `String` adapter reads it: text as is, numbers and booleans as
/// their text, missing or `null` as null. The outer `None` is an object or array, which Gson
/// rejects with a `JsonSyntaxException`.
fn text_field(fields: &Map<String, Value>, name: &str) -> Option<Option<String>> {
    match fields.get(name) {
        None | Some(Value::Null) => Some(None),
        Some(Value::String(text)) => Some(Some(text.clone())),
        Some(Value::Number(number)) => Some(Some(number.to_string())),
        Some(Value::Bool(flag)) => Some(Some(flag.to_string())),
        Some(Value::Array(_) | Value::Object(_)) => None,
    }
}

/// Server-supplied text made safe for a log line and a disconnect screen: the access token
/// removed should a response ever echo it, control characters replaced, length capped.
fn clean_server_text(text: &str, token: &str) -> String {
    // Real tokens are hundreds of characters; a short test token must not eat ordinary words.
    let text = if token.len() >= 8 {
        text.replace(token, "<redacted>")
    } else {
        text.to_owned()
    };
    let printable: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    snippet(&printable)
}

/// The `<title>` of an HTML error page, cleaned: enough to tell a CDN's block page from a
/// gateway error in the log, without logging the page itself.
fn html_title(body: &str, token: &str) -> Option<String> {
    let lower = body.to_ascii_lowercase();
    let start = lower.find("<title>")? + "<title>".len();
    let end = start + lower[start..].find("</title>")?;
    let title = clean_server_text(body.get(start..end)?, token);
    let title = title.trim();
    (!title.is_empty()).then(|| title.chars().take(100).collect())
}

/// The headers that help diagnose a failed join, read before the body consumes the response.
#[derive(Debug, Default)]
struct ResponseFacts {
    content_type: String,
    server: String,
    retry_after: Option<Duration>,
}

impl ResponseFacts {
    fn read(headers: &reqwest::header::HeaderMap) -> Self {
        let text = |name: reqwest::header::HeaderName| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(|value| value.chars().filter(|c| !c.is_control()).take(80).collect())
                .unwrap_or_default()
        };
        // Only the delay-seconds form; an HTTP-date is treated as absent.
        let retry_after = headers
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<u64>().ok())
            .map(Duration::from_secs);
        ResponseFacts {
            content_type: text(reqwest::header::CONTENT_TYPE),
            server: text(reqwest::header::SERVER),
            retry_after,
        }
    }
}

/// One failed join request: the error, and what the retry decision needs.
#[derive(Debug)]
struct JoinFailure {
    error: AuthError,
    /// A 429 or 5xx: worth one quick retry.
    transient: bool,
    /// The server's `Retry-After`, when it sent the seconds form.
    retry_after: Option<Duration>,
}

impl JoinFailure {
    fn final_error(error: AuthError) -> Self {
        JoinFailure {
            error,
            transient: false,
            retry_after: None,
        }
    }
}

impl AuthClient {
    /// Tell the session server that this player is about to join a server with `server_hash`.
    ///
    /// Success is HTTP 204 with an empty body. The answer is read the way vanilla reads it (see
    /// the module docs): [`AuthError::TokenRejected`] for an invalid or expired token,
    /// [`AuthError::JoinRefused`] for any other Yggdrasil error (its message is what vanilla
    /// shows), [`AuthError::JoinUnreadable`] when nothing readable came back, and
    /// [`AuthError::Network`] when the request never completed. A 429 or 5xx that came back
    /// quickly is retried once.
    ///
    /// Requires no Azure application registration: the Minecraft access token is the only
    /// credential involved.
    pub async fn join_server(&self, session: &Session, server_hash: &str) -> Result<(), AuthError> {
        let started = Instant::now();
        let failure = match self.join_once(session, server_hash).await {
            Ok(()) => return Ok(()),
            Err(failure) => failure,
        };
        let Some(delay) = self.join_retry_delay_for(&failure, started.elapsed()) else {
            return Err(failure.error);
        };
        tracing::warn!(
            error = %failure.error,
            delay_ms = delay.as_millis() as u64,
            "session server join failed with a temporary error; retrying once"
        );
        tokio::time::sleep(delay).await;
        self.join_once(session, server_hash)
            .await
            .map_err(|failure| failure.error)
    }

    /// The pause before the single retry, or `None` when `failure` is not worth one.
    fn join_retry_delay_for(&self, failure: &JoinFailure, elapsed: Duration) -> Option<Duration> {
        if !failure.transient || elapsed > RETRY_ONLY_WITHIN {
            return None;
        }
        match failure.retry_after {
            Some(wait) if wait > MAX_RETRY_AFTER => None,
            Some(wait) => Some(wait),
            None => Some(self.join_retry_delay),
        }
    }

    /// One join request, read the way authlib 1.5.21 reads it.
    async fn join_once(&self, session: &Session, server_hash: &str) -> Result<(), JoinFailure> {
        let body = json!({
            "accessToken": session.expose(),
            "selectedProfile": session.uuid_undashed(),
            "serverId": server_hash,
        });

        let started = Instant::now();
        let network = |source| {
            JoinFailure::final_error(AuthError::Network {
                endpoint: EP_JOIN,
                source,
            })
        };
        let response = self
            .http
            .post(&self.endpoints.session_join)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&body)
            .send()
            .await
            .map_err(network)?;
        let status = response.status().as_u16();
        let facts = ResponseFacts::read(response.headers());
        let text = response.text().await.map_err(network)?;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let token = session.expose();

        let transient = TRANSIENT_STATUSES.contains(&status);
        let error = match classify_join_response(status, &text) {
            JoinVerdict::Accepted => {
                if (200..300).contains(&status) {
                    tracing::debug!(player = %session.username, "session server join accepted");
                } else {
                    // java: no `error` field means success, whatever the status. Say so loudly:
                    // if the join did not really register, the server will refuse the login.
                    tracing::warn!(
                        status,
                        content_type = %facts.content_type,
                        elapsed_ms,
                        "session server answered the join with an error status but no error; \
                         continuing like vanilla"
                    );
                }
                return Ok(());
            }
            JoinVerdict::InvalidSession { error } => {
                let error = error.map(|error| clean_server_text(&error, token));
                tracing::warn!(
                    status,
                    error = error.as_deref().unwrap_or("(none)"),
                    content_type = %facts.content_type,
                    server = %facts.server,
                    elapsed_ms,
                    "session server rejected the join: the access token is invalid or expired"
                );
                AuthError::TokenRejected
            }
            JoinVerdict::Refused { error, message } => {
                let error = clean_server_text(&error, token);
                let message = message
                    .map(|message| clean_server_text(&message, token))
                    .filter(|message| !message.trim().is_empty());
                tracing::warn!(
                    status,
                    error = %error,
                    message = message.as_deref().unwrap_or("(none)"),
                    retry_after_s = facts.retry_after.map(|wait| wait.as_secs()),
                    server = %facts.server,
                    elapsed_ms,
                    "session server refused the join"
                );
                AuthError::JoinRefused {
                    status,
                    error,
                    message,
                }
            }
            JoinVerdict::Unreadable { what } => {
                let content_type = if facts.content_type.is_empty() {
                    "no content type"
                } else {
                    facts.content_type.as_str()
                };
                let detail = format!("{what} ({content_type}, {} bytes)", text.len());
                tracing::warn!(
                    status,
                    detail = %detail,
                    page_title = html_title(&text, token).as_deref().unwrap_or("(none)"),
                    retry_after_s = facts.retry_after.map(|wait| wait.as_secs()),
                    server = %facts.server,
                    elapsed_ms,
                    "session server's answer to the join could not be read"
                );
                AuthError::JoinUnreadable { status, detail }
            }
        };
        Err(JoinFailure {
            transient: transient && !matches!(error, AuthError::TokenRejected),
            retry_after: facts.retry_after,
            error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(error: &str, message: Option<&str>) -> JoinVerdict {
        JoinVerdict::Refused {
            error: error.to_owned(),
            message: message.map(str::to_owned),
        }
    }

    /// Every branch of authlib 1.5.21's `makeRequest`, plus the two documented differences.
    #[test]
    fn join_answers_are_read_like_authlib() {
        let cases: &[(u16, &str, JoinVerdict)] = &[
            // Success: the documented empty 204, and anything else Gson reads as null.
            (204, "", JoinVerdict::Accepted),
            (200, "  \n", JoinVerdict::Accepted),
            (200, "null", JoinVerdict::Accepted),
            // A response without `error` is success whatever the status (vanilla).
            (200, r#"{"id":"x"}"#, JoinVerdict::Accepted),
            (
                503,
                r#"{"path":"/session/minecraft/join"}"#,
                JoinVerdict::Accepted,
            ),
            (
                400,
                r#"{"error":"  ","errorMessage":"blank error"}"#,
                JoinVerdict::Accepted,
            ),
            // InvalidCredentialsException.
            (
                403,
                r#"{"error":"ForbiddenOperationException","errorMessage":"Invalid token"}"#,
                JoinVerdict::InvalidSession {
                    error: Some("ForbiddenOperationException".into()),
                },
            ),
            (
                200,
                r#"{"error":"ForbiddenOperationException"}"#,
                JoinVerdict::InvalidSession {
                    error: Some("ForbiddenOperationException".into()),
                },
            ),
            (
                403,
                r#"{"error":"Whatever","cause":"UserMigratedException"}"#,
                JoinVerdict::InvalidSession {
                    error: Some("Whatever".into()),
                },
            ),
            // Plain AuthenticationException: the message is what vanilla shows.
            (
                403,
                r#"{"error":"InsufficientPrivilegesException","errorMessage":"Multiplayer is disabled"}"#,
                refused(
                    "InsufficientPrivilegesException",
                    Some("Multiplayer is disabled"),
                ),
            ),
            (
                403,
                r#"{"error":"UserBannedException","path":"/session/minecraft/join"}"#,
                refused("UserBannedException", None),
            ),
            (
                429,
                r#"{"error":"TooManyRequestsException","errorMessage":"Slow down"}"#,
                refused("TooManyRequestsException", Some("Slow down")),
            ),
            (
                400,
                r#"{"error":7,"errorMessage":true}"#,
                refused("7", Some("true")),
            ),
            // AuthenticationUnavailableException.
            (
                503,
                "",
                JoinVerdict::Unreadable {
                    what: "an empty body",
                },
            ),
            (
                403,
                "<html><title>Request blocked</title></html>",
                JoinVerdict::Unreadable {
                    what: "a body that is not JSON",
                },
            ),
            (
                200,
                "<html>captive portal</html>",
                JoinVerdict::Unreadable {
                    what: "a body that is not JSON",
                },
            ),
            (
                500,
                "[1,2]",
                JoinVerdict::Unreadable {
                    what: "JSON that is not an object",
                },
            ),
            (
                400,
                r#"{"error":{"nested":true}}"#,
                JoinVerdict::Unreadable {
                    what: "a Yggdrasil field that is not text",
                },
            ),
            // The documented difference: an unreadable 401 is an invalid session.
            (401, "", JoinVerdict::InvalidSession { error: None }),
            (
                401,
                "Unauthorized",
                JoinVerdict::InvalidSession { error: None },
            ),
        ];
        for (status, body, expected) in cases {
            assert_eq!(
                &classify_join_response(*status, body),
                expected,
                "HTTP {status} with body {body:?}"
            );
        }
    }

    #[test]
    fn server_text_never_carries_the_token_or_control_characters() {
        let token = "eyJhbGciOiJIUzI1NiJ9.payload.signature";
        let cleaned = clean_server_text(&format!("bad token {token}\n\u{1b}[31m"), token);
        assert!(!cleaned.contains(token), "{cleaned}");
        assert!(cleaned.contains("<redacted>"), "{cleaned}");
        assert!(!cleaned.chars().any(char::is_control), "{cleaned:?}");
        // A short token (tests only) must not redact ordinary words.
        assert_eq!(clean_server_text("token x", "x"), "token x");
        assert_eq!(
            clean_server_text(&"é".repeat(400), token).chars().count(),
            clean_server_text(&"é".repeat(300), token).chars().count(),
            "long messages are capped"
        );
    }

    #[test]
    fn html_titles_are_extracted_without_the_page() {
        assert_eq!(
            html_title(
                "<HTML><HEAD><TITLE>ERROR: The request could not be satisfied</TITLE>",
                "token-is-long-enough"
            )
            .as_deref(),
            Some("ERROR: The request could not be satisfied")
        );
        assert_eq!(html_title("<title> </title>", "token-is-long-enough"), None);
        assert_eq!(html_title("no title here", "token-is-long-enough"), None);
    }
}
