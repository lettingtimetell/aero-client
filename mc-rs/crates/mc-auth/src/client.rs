//! The HTTP client and the endpoint table every step of the flow shares.

use std::time::Duration;

use crate::config::{AuthConfig, SocksProxy, SCOPE, TENANT};
use crate::error::{snippet, AuthError};

/// User agent sent on every request. Mojang and Microsoft both accept anything, but a
/// recognisable one makes their side of a support ticket tractable.
const USER_AGENT: &str = concat!("mc-rs/", env!("CARGO_PKG_VERSION"), " (mc-auth)");

/// Default request timeout. The device-code *poll* is a normal short request; the long wait
/// happens between polls, not inside one.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Every URL the flow touches, in one place so tests can point them at a mock server.
///
/// [`Endpoints::default`] is the real service. Overriding these is how the wiremock-based
/// integration tests exercise the whole chain without contacting Microsoft.
#[derive(Debug, Clone)]
pub struct Endpoints {
    /// `POST` — device authorization request (RFC 8628 §3.1).
    pub msa_device_code: String,
    /// `POST` — token endpoint, used for both device-code polling and refresh.
    pub msa_token: String,
    /// `POST` — Xbox Live user authentication.
    pub xbl_authenticate: String,
    /// `POST` — XSTS authorization for the Minecraft services relying party.
    pub xsts_authorize: String,
    /// `POST` — exchange an XSTS token for a Minecraft access token.
    pub mc_login_with_xbox: String,
    /// `GET` — the entitlement store, used for an explicit ownership check.
    pub mc_entitlements: String,
    /// `GET` — the Minecraft profile (UUID, name, skins, capes).
    pub mc_profile: String,
    /// `POST` — the session-server join call made during the encryption handshake.
    pub session_join: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Self {
            msa_device_code: format!(
                "https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/devicecode"
            ),
            msa_token: format!("https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"),
            xbl_authenticate: "https://user.auth.xboxlive.com/user/authenticate".to_owned(),
            xsts_authorize: "https://xsts.auth.xboxlive.com/xsts/authorize".to_owned(),
            mc_login_with_xbox: "https://api.minecraftservices.com/authentication/login_with_xbox"
                .to_owned(),
            mc_entitlements: "https://api.minecraftservices.com/entitlements/mcstore".to_owned(),
            mc_profile: "https://api.minecraftservices.com/minecraft/profile".to_owned(),
            session_join: "https://sessionserver.mojang.com/session/minecraft/join".to_owned(),
        }
    }
}

impl Endpoints {
    /// Point every endpoint at `base` (a mock server root), preserving the real paths.
    ///
    /// Only useful for tests; the paths are the ones the mocks register against.
    #[must_use]
    pub fn rooted_at(base: &str) -> Self {
        let base = base.trim_end_matches('/');
        Self {
            msa_device_code: format!("{base}/oauth2/v2.0/devicecode"),
            msa_token: format!("{base}/oauth2/v2.0/token"),
            xbl_authenticate: format!("{base}/user/authenticate"),
            xsts_authorize: format!("{base}/xsts/authorize"),
            mc_login_with_xbox: format!("{base}/authentication/login_with_xbox"),
            mc_entitlements: format!("{base}/entitlements/mcstore"),
            mc_profile: format!("{base}/minecraft/profile"),
            session_join: format!("{base}/session/minecraft/join"),
        }
    }
}

/// Knobs for the device-code polling loop.
///
/// Defaults follow RFC 8628: never poll faster than the server's `interval`, and add five
/// seconds on every `slow_down`. Tests shrink both so a full polling scenario runs in
/// milliseconds.
#[derive(Debug, Clone, Copy)]
pub struct PollTuning {
    /// Lower bound on the wait between polls, regardless of what the server asked for.
    pub min_interval: Duration,
    /// How much to add to the interval on a `slow_down` response.
    pub slow_down_increment: Duration,
}

impl Default for PollTuning {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_secs(1),
            slow_down_increment: Duration::from_secs(5),
        }
    }
}

/// The authenticating HTTP client.
///
/// One instance can serve the whole application: it is cheap to clone (the inner
/// `reqwest::Client` is an `Arc`) and holds no per-login state.
#[derive(Debug, Clone)]
pub struct AuthClient {
    pub(crate) http: reqwest::Client,
    pub(crate) endpoints: Endpoints,
    pub(crate) config: AuthConfig,
    pub(crate) poll: PollTuning,
    /// The pause before the session-server join's single retry of a transient failure, when
    /// the server sent no `Retry-After` (see `join.rs`).
    pub(crate) join_retry_delay: Duration,
}

/// The default pause before a transient join failure is retried.
const JOIN_RETRY_DELAY: Duration = Duration::from_secs(1);

impl AuthClient {
    /// Build a client from configuration discovered in the environment and config file.
    ///
    /// A missing client ID is *not* an error here — it surfaces when a login is actually
    /// attempted, so that [`join_server`](Self::join_server) and
    /// [`server_hash`](crate::server_hash) work on an unconfigured machine.
    pub fn discover() -> Result<Self, AuthError> {
        Self::new(AuthConfig::load()?)
    }

    /// Build a client from an explicit configuration.
    pub fn new(config: AuthConfig) -> Result<Self, AuthError> {
        Self::new_with_proxy(config, None)
    }

    /// Build a client with Java's optional SOCKS5 proxy transport.
    pub fn new_with_proxy(
        config: AuthConfig,
        proxy: Option<SocksProxy>,
    ) -> Result<Self, AuthError> {
        let mut builder = reqwest::Client::builder();
        if let Some(proxy) = proxy {
            builder = builder.proxy(
                reqwest::Proxy::all(proxy.to_reqwest_url()).map_err(AuthError::HttpClientInit)?,
            );
        }
        let http = builder
            .user_agent(USER_AGENT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(AuthError::HttpClientInit)?;
        Ok(Self {
            http,
            endpoints: Endpoints::default(),
            config,
            poll: PollTuning::default(),
            join_retry_delay: JOIN_RETRY_DELAY,
        })
    }

    /// Replace the endpoint table (tests, or a future proxy/enterprise deployment).
    #[must_use]
    pub fn with_endpoints(mut self, endpoints: Endpoints) -> Self {
        self.endpoints = endpoints;
        self
    }

    /// Replace the pause before a transient join failure is retried (tests shorten it).
    #[must_use]
    pub fn with_join_retry_delay(mut self, delay: Duration) -> Self {
        self.join_retry_delay = delay;
        self
    }

    /// Replace the polling tuning.
    #[must_use]
    pub fn with_poll_tuning(mut self, poll: PollTuning) -> Self {
        self.poll = poll;
        self
    }

    /// The OAuth2 scopes this client requests.
    #[must_use]
    pub fn scope(&self) -> &'static str {
        SCOPE
    }

    /// The configured Azure application (client) ID, or [`AuthError::MissingClientId`].
    pub fn client_id(&self) -> Result<&str, AuthError> {
        self.config.require_client_id()
    }
}

/// Read a response body as text, mapping transport failures onto `endpoint`.
pub(crate) async fn body_text(
    endpoint: &'static str,
    response: reqwest::Response,
) -> Result<(u16, String), AuthError> {
    let status = response.status().as_u16();
    let text = response
        .text()
        .await
        .map_err(|source| AuthError::Network { endpoint, source })?;
    Ok((status, text))
}

/// Parse JSON from a body, turning a schema mismatch into a typed error rather than a panic.
pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(
    endpoint: &'static str,
    body: &str,
) -> Result<T, AuthError> {
    serde_json::from_str(body).map_err(|e| AuthError::MalformedResponse {
        endpoint,
        detail: e.to_string(),
    })
}

/// Build the generic "unexpected status" error for an endpoint.
pub(crate) fn http_error(endpoint: &'static str, status: u16, body: &str) -> AuthError {
    AuthError::Http {
        endpoint,
        status,
        body: snippet(body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SocksProxy;

    #[test]
    fn default_endpoints_are_the_documented_ones() {
        let e = Endpoints::default();
        assert_eq!(
            e.msa_device_code,
            "https://login.microsoftonline.com/consumers/oauth2/v2.0/devicecode"
        );
        assert_eq!(
            e.msa_token,
            "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
        );
        assert_eq!(
            e.xbl_authenticate,
            "https://user.auth.xboxlive.com/user/authenticate"
        );
        assert_eq!(
            e.xsts_authorize,
            "https://xsts.auth.xboxlive.com/xsts/authorize"
        );
        assert_eq!(
            e.mc_login_with_xbox,
            "https://api.minecraftservices.com/authentication/login_with_xbox"
        );
        assert_eq!(
            e.mc_profile,
            "https://api.minecraftservices.com/minecraft/profile"
        );
        assert_eq!(
            e.session_join,
            "https://sessionserver.mojang.com/session/minecraft/join"
        );
        // Everything must be HTTPS: these carry bearer tokens.
        for url in [
            &e.msa_device_code,
            &e.msa_token,
            &e.xbl_authenticate,
            &e.xsts_authorize,
            &e.mc_login_with_xbox,
            &e.mc_entitlements,
            &e.mc_profile,
            &e.session_join,
        ] {
            assert!(url.starts_with("https://"), "{url} is not HTTPS");
        }
    }

    #[test]
    fn rooted_endpoints_keep_the_real_paths() {
        let e = Endpoints::rooted_at("http://127.0.0.1:1234/");
        assert_eq!(e.mc_profile, "http://127.0.0.1:1234/minecraft/profile");
        assert_eq!(e.xsts_authorize, "http://127.0.0.1:1234/xsts/authorize");
    }

    #[test]
    fn scope_requests_offline_access() {
        // Without offline_access there is no refresh token and no silent re-login.
        assert!(SCOPE.contains("offline_access"));
        assert!(SCOPE.contains("XboxLive.signin"));
    }

    #[test]
    fn unconfigured_client_reports_missing_client_id() {
        let c = AuthClient::new(AuthConfig::default()).expect("build");
        assert!(matches!(c.client_id(), Err(AuthError::MissingClientId)));
    }

    #[test]
    fn proxy_aware_client_accepts_socks5h_transport() {
        let proxy = SocksProxy::new("proxy.example", 1080, None, None).expect("proxy");
        AuthClient::new_with_proxy(AuthConfig::default(), Some(proxy)).expect("build");
    }
}
