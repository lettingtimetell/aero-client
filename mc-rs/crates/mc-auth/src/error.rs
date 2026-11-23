//! Typed errors for `mc-auth`.
//!
//! The variants are deliberately specific where a user can act on the difference — "you have
//! no Xbox profile", "this is a child account", "this account does not own the game" are three
//! different screens in the UI, not one generic "login failed".

use std::path::PathBuf;

use thiserror::Error;

/// The environment variable that supplies the Azure application (client) ID.
pub const CLIENT_ID_ENV: &str = "MC_RS_MSA_CLIENT_ID";

/// Everything that can go wrong while authenticating or joining a server.
///
/// `Display` never contains an access token, a refresh token, or a device code. HTTP error
/// bodies are included truncated, because Microsoft's and Mojang's failure responses carry the
/// diagnostic detail and never carry credentials.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AuthError {
    /// No Azure application (client) ID is configured, so the OAuth2 flow cannot start.
    ///
    /// See the crate-level docs for the app-registration steps. This is not an error the code
    /// can recover from — it needs a one-time action from the user.
    #[error(
        "no Microsoft application (client) ID is configured: set ${env} or put \
         {{\"client_id\": \"<guid>\"}} in the mc-auth config file (see the mc-auth crate docs \
         for the Azure app registration steps)",
        env = CLIENT_ID_ENV
    )]
    MissingClientId,

    /// This platform has no discoverable per-user configuration directory.
    #[error("could not determine a per-user configuration directory on this platform")]
    NoConfigDir,

    /// A configuration file exists but is not valid JSON.
    #[error("configuration file {path} is not valid JSON")]
    Config {
        /// The offending file.
        path: PathBuf,
        /// The underlying parse failure.
        #[source]
        source: serde_json::Error,
    },

    /// A filesystem operation on the config directory or token cache failed.
    #[error("i/o error on {path}")]
    Io {
        /// The path that was being read or written.
        path: PathBuf,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// The HTTP client itself could not be constructed (missing TLS backend, bad proxy config).
    #[error("could not build the HTTP client")]
    HttpClientInit(#[source] reqwest::Error),

    /// The request never completed: DNS, TCP, TLS, timeout, or a broken response body.
    #[error("network error talking to {endpoint}")]
    Network {
        /// A human-readable name for the endpoint, e.g. `"XSTS authorize"`.
        endpoint: &'static str,
        /// The underlying transport failure.
        #[source]
        source: reqwest::Error,
    },

    /// The server answered with an unexpected status and a body we could not interpret.
    #[error("{endpoint} returned HTTP {status}: {body}")]
    Http {
        /// A human-readable name for the endpoint.
        endpoint: &'static str,
        /// The HTTP status code.
        status: u16,
        /// The response body, truncated.
        body: String,
    },

    /// The server answered with the expected status but a body that did not match the schema.
    #[error("{endpoint} returned a response this client could not parse: {detail}")]
    MalformedResponse {
        /// A human-readable name for the endpoint.
        endpoint: &'static str,
        /// What was wrong with it.
        detail: String,
    },

    /// The Microsoft identity platform returned an OAuth2 error that is not one of the
    /// specifically-handled device-code polling states.
    #[error("Microsoft rejected the request ({error}): {description}")]
    Oauth {
        /// The machine-readable `error` field, e.g. `invalid_client`.
        error: String,
        /// The human-readable `error_description` field, if present.
        description: String,
    },

    /// The user actively declined the sign-in request in the browser.
    #[error("the user declined the sign-in request")]
    Declined,

    /// The device code expired before the user finished signing in.
    #[error("the sign-in code expired before it was used; start the login again")]
    DeviceCodeExpired,

    /// `XErr 2148916233` — the Microsoft account has never been used with Xbox Live.
    #[error("this Microsoft account has no Xbox profile; create one at https://signup.live.com/signup?markets=280&lic=1 and sign in again")]
    NoXboxAccount,

    /// `XErr 2148916227` — the account is banned from Xbox Live.
    #[error("this account is banned from Xbox Live")]
    XboxBanned,

    /// `XErr 2148916235` — Xbox Live is unavailable in the account's country/region.
    #[error("Xbox Live is not available in this account's region")]
    XboxRegionUnavailable,

    /// `XErr 2148916236` / `2148916237` — the account needs adult verification (South Korea).
    #[error("this account needs adult verification before it can use Xbox Live")]
    AdultVerificationRequired,

    /// `XErr 2148916238` — a child account that is not in a Microsoft family group.
    ///
    /// This one is specific to third-party Azure applications: Mojang's own launcher is
    /// allowlisted and does not hit it, so it is a common surprise for reimplementations.
    #[error("this is a child account and must be added to a Microsoft family group first{}",
            match .redirect { Some(u) => format!(" ({u})"), None => String::new() })]
    ChildAccount {
        /// The `Redirect` URL Microsoft supplied, if any.
        redirect: Option<String>,
    },

    /// XSTS refused the token with an `XErr` code we do not map to a friendlier variant.
    #[error("Xbox Live refused the token (XErr {xerr}): {message}")]
    Xsts {
        /// The raw `XErr` code.
        xerr: u64,
        /// The `Message` field, usually empty in practice.
        message: String,
    },

    /// The account authenticated but does not own Minecraft: Java Edition.
    ///
    /// Xbox Game Pass counts as ownership, but only once the profile has been created in the
    /// official launcher at least one time.
    #[error("this account does not own Minecraft: Java Edition (or has never created a profile)")]
    NoEntitlement,

    /// Minecraft services refused the Xbox identity at `login_with_xbox` with a `403`.
    ///
    /// Two unrelated causes share this status code: an Azure application ID that Mojang has not
    /// approved for the Minecraft Services API, and an account with no Java Edition entitlement.
    /// They are indistinguishable from the status alone, so the response body is carried here —
    /// it is the only thing that tells them apart, and collapsing this into
    /// [`AuthError::NoEntitlement`] misreports an unapproved client ID as a missing purchase.
    #[error(
        "Minecraft services refused the sign-in (HTTP 403) — either the Azure client ID is not \
         approved by Mojang, or the account does not own Java Edition. Response body: {body}"
    )]
    ServicesForbidden {
        /// The raw `login_with_xbox` response body, truncated to a readable snippet.
        body: String,
    },

    /// A Minecraft access token was rejected as expired or revoked.
    ///
    /// For the server join this is authlib 1.5.21's `InvalidCredentialsException`: a
    /// `ForbiddenOperationException` error (or a `UserMigratedException` cause), or an HTTP 401.
    /// Vanilla shows "Failed to login: Invalid session (Try restarting your game)".
    #[error("the Minecraft access token was rejected; sign in again")]
    TokenRejected,

    /// The session server refused the join with a reason of its own.
    ///
    /// This is authlib 1.5.21's plain `AuthenticationException`: a Yggdrasil error response
    /// whose `error` is not `ForbiddenOperationException` (that one is
    /// [`AuthError::TokenRejected`]). Vanilla 1.8.9 shows `errorMessage` on the disconnect
    /// screen as "Failed to login: <message>". Typical `error` values are
    /// `InsufficientPrivilegesException` (multiplayer turned off for the account) and
    /// `UserBannedException`.
    ///
    /// Both strings come from the server, cleaned of control characters, truncated, and with
    /// the access token removed should a response ever echo it.
    #[error(
        "the session server refused the join (HTTP {status}, {error}): {}",
        .message.as_deref().unwrap_or("no message")
    )]
    JoinRefused {
        /// The HTTP status code.
        status: u16,
        /// Mojang's machine-readable error type, e.g. `InsufficientPrivilegesException`.
        error: String,
        /// Mojang's human-readable `errorMessage`, if the response had one.
        message: Option<String>,
    },

    /// The session server's answer to the join was not a Yggdrasil response this client (or
    /// vanilla) can read: an error status with an empty body, an HTML error page, or malformed
    /// JSON.
    ///
    /// authlib 1.5.21 turns every such answer into `AuthenticationUnavailableException`
    /// ("Cannot contact authentication server"), which vanilla shows as "The authentication
    /// servers are currently down for maintenance."
    #[error("the session server's answer to the join could not be read (HTTP {status}): {detail}")]
    JoinUnreadable {
        /// The HTTP status code.
        status: u16,
        /// What came back instead, without any of its content (content type and length).
        detail: String,
    },

    /// The on-disk token cache could not be decrypted or decoded.
    ///
    /// Callers should treat this as "no cached credentials" and fall back to an interactive
    /// login rather than failing outright.
    #[error("the cached credentials could not be decrypted; they will be discarded")]
    CacheCorrupt,
}

impl AuthError {
    /// Whether Minecraft services refused the sign-in because the Azure application (client) ID
    /// is not approved by Mojang: a `403` from `login_with_xbox` whose body says "Invalid app
    /// registration" (Mojang's wording, with a link to `aka.ms/AppRegInfo`). The other `403`
    /// cause, an account without Java Edition, does not match.
    #[must_use]
    pub fn is_unapproved_app(&self) -> bool {
        match self {
            AuthError::ServicesForbidden { body } => {
                let body = body.to_ascii_lowercase();
                body.contains("invalid app registration") || body.contains("appreginfo")
            }
            _ => false,
        }
    }

    /// Whether signing in again is the fix: the Minecraft token was rejected, the saved sign-in
    /// cannot be opened on this PC, or Microsoft refused the refresh token (expired, revoked,
    /// issued to another application, or the account needs to approve something).
    #[must_use]
    pub fn needs_new_sign_in(&self) -> bool {
        match self {
            AuthError::TokenRejected | AuthError::CacheCorrupt => true,
            AuthError::Oauth { error, .. } => matches!(
                error.as_str(),
                "invalid_grant" | "interaction_required" | "consent_required" | "login_required"
            ),
            _ => false,
        }
    }
}

/// Truncate a response body so an error message stays readable in a log line.
///
/// Registered secrets are removed *before* the cut: the log writer only finds
/// whole tokens, so a token cut in half here would otherwise keep its first half.
pub(crate) fn snippet(body: &str) -> String {
    const LIMIT: usize = 256;
    let body = mc_core::secrets::scrub(body);
    let trimmed = body.trim();
    if trimmed.len() <= LIMIT {
        return trimmed.to_owned();
    }
    // Cut on a char boundary.
    let mut end = LIMIT;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &trimmed[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_truncates_on_char_boundaries() {
        let long = "é".repeat(400);
        let s = snippet(&long);
        assert!(s.ends_with('…'));
        assert!(s.len() <= 256 + '…'.len_utf8());
    }

    #[test]
    fn snippet_leaves_short_bodies_alone() {
        assert_eq!(snippet("  hello  "), "hello");
    }

    /// A body that echoes a token straddling the 256-byte cut keeps no part of
    /// it: the token is replaced before the body is cut.
    #[test]
    fn snippet_removes_a_registered_token_before_cutting() {
        let token = format!(
            "eyJhbGciOiJIUzI1NiJ9.{}.c2ln",
            "fake-snippet-token-".repeat(20)
        );
        mc_core::secrets::register(&token);
        let body = format!("{} {token} tail", "x".repeat(200));
        let cut = snippet(&body);
        assert!(!cut.contains("eyJhbGciOiJIUzI1NiJ9"), "{cut}");
        assert!(!cut.contains("fake-snippet-token"), "{cut}");
        assert!(cut.contains("<redacted> tail"), "{cut}");
    }

    #[test]
    fn mojangs_invalid_app_registration_answer_is_told_apart_from_ownership() {
        let unapproved = AuthError::ServicesForbidden {
            body: snippet(
                r#"{"path":"/authentication/login_with_xbox","errorMessage":"Invalid app registration, see https://aka.ms/AppRegInfo for more information"}"#,
            ),
        };
        assert!(unapproved.is_unapproved_app());
        let ownership = AuthError::ServicesForbidden {
            body: r#"{"path":"/authentication/login_with_xbox","error":"FORBIDDEN"}"#.into(),
        };
        assert!(!ownership.is_unapproved_app());
        assert!(!AuthError::NoEntitlement.is_unapproved_app());
    }

    #[test]
    fn only_a_dead_or_unreadable_sign_in_asks_for_a_new_one() {
        let oauth = |error: &str| AuthError::Oauth {
            error: error.into(),
            description: String::new(),
        };
        assert!(AuthError::TokenRejected.needs_new_sign_in());
        assert!(AuthError::CacheCorrupt.needs_new_sign_in());
        assert!(oauth("invalid_grant").needs_new_sign_in());
        assert!(oauth("interaction_required").needs_new_sign_in());
        assert!(!oauth("invalid_client").needs_new_sign_in());
        assert!(!oauth("temporarily_unavailable").needs_new_sign_in());
        assert!(!AuthError::MissingClientId.needs_new_sign_in());
        assert!(!AuthError::ServicesForbidden {
            body: String::new()
        }
        .needs_new_sign_in());
    }

    #[test]
    fn child_account_message_includes_redirect() {
        let e = AuthError::ChildAccount {
            redirect: Some("https://start.ui.xboxlive.com/AddChildToFamily".to_owned()),
        };
        assert!(e.to_string().contains("AddChildToFamily"));
        let e = AuthError::ChildAccount { redirect: None };
        assert!(e.to_string().ends_with("family group first"));
    }
}
