//! Where the Azure application (client) ID and the token cache live.
//!
//! Mojang's own client ID is licensed to Mojang, not to us (hazard H13), so it is deliberately
//! absent from this source tree. The ID is loaded at runtime from the environment or from a
//! small JSON file, and its absence is a first-class, actionable error
//! ([`AuthError::MissingClientId`]).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AuthError, CLIENT_ID_ENV};

/// Environment variable overriding the whole config directory (used by tests and portable
/// installs).
pub const CONFIG_DIR_ENV: &str = "MC_RS_CONFIG_DIR";

/// File name of the JSON config inside the config directory.
pub const CONFIG_FILE_NAME: &str = "auth.json";

/// OAuth2 scopes required for the Minecraft chain.
///
/// `XboxLive.signin` is what Xbox Live's RPS ticket exchange needs; `offline_access` is what
/// makes Microsoft issue a refresh token, without which there is no silent re-login.
pub const SCOPE: &str = "XboxLive.signin offline_access";

/// The AAD tenant to authenticate against.
///
/// It must be `consumers`: `XboxLive.signin` is not grantable through `common` or
/// `organizations`, and using the wrong tenant fails late, at the Xbox Live step, with a
/// confusing error.
pub const TENANT: &str = "consumers";

/// A SOCKS5 proxy used for authentication and session-service HTTP(S) requests.
///
/// Java's launcher creates a `Proxy(Type.SOCKS, ...)`, which routes the destination lookup
/// through the proxy as well as the TCP stream.  `socks5h` is the reqwest equivalent, so host
/// names do not leak to the local resolver.  Credentials are deliberately redacted from
/// [`Debug`] output because launchers commonly pass `proxyPass` on the command line.
#[derive(Clone, PartialEq, Eq)]
pub struct SocksProxy {
    host: String,
    port: u16,
    username: Option<String>,
    password: Option<String>,
}

impl SocksProxy {
    /// Construct a SOCKS5 proxy from Java's `proxyHost`, `proxyPort`, `proxyUser`, and
    /// `proxyPass` options.  Empty credentials are treated as absent, matching Java's
    /// `isNullOrEmpty` check before it installs its authenticator.
    pub fn new(
        host: impl Into<String>,
        port: u16,
        username: Option<String>,
        password: Option<String>,
    ) -> Result<Self, &'static str> {
        let host = host.into();
        if host.trim().is_empty() {
            return Err("proxy host must not be empty");
        }
        Ok(Self {
            host,
            port,
            username: username.filter(|value| !value.is_empty()),
            password: password.filter(|value| !value.is_empty()),
        })
    }

    /// The proxy host, retained separately for diagnostics and tests.
    #[must_use]
    pub fn host(&self) -> &str {
        &self.host
    }

    /// The proxy port.
    #[must_use]
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// The optional username.  The password intentionally has no accessor.
    #[must_use]
    pub fn username(&self) -> Option<&str> {
        self.username.as_deref()
    }

    /// Render a reqwest URL.  `socks5h` preserves Java SOCKS semantics by delegating DNS to the
    /// proxy.  This method is crate-public API because both the auth and resource-pack clients
    /// must build exactly the same proxy transport.
    #[must_use]
    pub fn to_reqwest_url(&self) -> String {
        let credentials = match (&self.username, &self.password) {
            (Some(username), Some(password)) => {
                format!(
                    "{}:{}@",
                    encode_proxy_component(username),
                    encode_proxy_component(password)
                )
            }
            _ => String::new(),
        };
        let host = if self.host.contains(':') && !self.host.starts_with('[') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("socks5h://{credentials}{host}:{}", self.port)
    }
}

impl std::fmt::Debug for SocksProxy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SocksProxy")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}

fn encode_proxy_component(value: &str) -> String {
    value
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                vec![byte as char]
            }
            other => format!("%{other:02X}").chars().collect(),
        })
        .collect()
}

/// Runtime configuration for [`AuthClient`](crate::AuthClient).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    /// The Azure application (client) ID — a GUID.
    ///
    /// `None` is legal: [`join_server`](crate::join_server) and
    /// [`server_hash`](crate::server_hash) do not need it. Only the login flows do.
    pub client_id: Option<String>,
}

/// The Azure application (client) ID compiled into this build: the `AERO_MSA_CLIENT_ID`
/// environment variable of the `cargo build` that compiled this crate. `None` (or blank) in a
/// build made without it; the ID is never written in the source.
pub const BUILT_IN_CLIENT_ID: Option<&str> = option_env!("AERO_MSA_CLIENT_ID");

/// [`BUILT_IN_CLIENT_ID`], trimmed, when it is not blank.
#[must_use]
pub fn built_in_client_id() -> Option<&'static str> {
    BUILT_IN_CLIENT_ID
        .map(str::trim)
        .filter(|id| !id.is_empty())
}

/// Where the client ID in use came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientIdSource {
    /// The `MC_RS_MSA_CLIENT_ID` environment variable: a development override.
    Environment,
    /// `client_id` in the config directory's `auth.json`: a development override.
    ConfigFile,
    /// Compiled in at build time ([`BUILT_IN_CLIENT_ID`]).
    BuiltIn,
}

impl AuthConfig {
    /// Load configuration: the environment first, then the config file, then the ID compiled
    /// into this build ([`BUILT_IN_CLIENT_ID`]).
    ///
    /// Never fails just because nothing is configured — it returns an empty config and lets the
    /// login flow raise [`AuthError::MissingClientId`] at the point of use.
    pub fn load() -> Result<Self, AuthError> {
        Self::load_with_source().map(|(config, _)| config)
    }

    /// [`AuthConfig::load`], and where its client ID came from (`None` when there is none).
    pub fn load_with_source() -> Result<(Self, Option<ClientIdSource>), AuthError> {
        if let Ok(id) = std::env::var(CLIENT_ID_ENV) {
            let id = id.trim().to_owned();
            if !id.is_empty() {
                let config = Self {
                    client_id: Some(id),
                };
                return Ok((config, Some(ClientIdSource::Environment)));
            }
        }
        let from_file = match config_dir() {
            Ok(dir) => Self::load_from(&dir.join(CONFIG_FILE_NAME))?,
            // No config dir and no env var simply means "not configured".
            Err(AuthError::NoConfigDir) => Self::default(),
            Err(e) => return Err(e),
        };
        if from_file.require_client_id().is_ok() {
            return Ok((from_file, Some(ClientIdSource::ConfigFile)));
        }
        Ok(Self::with_built_in(from_file))
    }

    /// `config` with the built-in client ID filled in when it has none of its own.
    fn with_built_in(config: Self) -> (Self, Option<ClientIdSource>) {
        match built_in_client_id() {
            Some(id) => (
                Self {
                    client_id: Some(id.to_owned()),
                },
                Some(ClientIdSource::BuiltIn),
            ),
            None => (config, None),
        }
    }

    /// Load configuration from a specific JSON file. A missing file yields an empty config.
    pub fn load_from(path: &Path) -> Result<Self, AuthError> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(source) => {
                return Err(AuthError::Io {
                    path: path.to_owned(),
                    source,
                })
            }
        };
        serde_json::from_str(&text).map_err(|source| AuthError::Config {
            path: path.to_owned(),
            source,
        })
    }

    /// The client ID, or [`AuthError::MissingClientId`] with an actionable message.
    pub fn require_client_id(&self) -> Result<&str, AuthError> {
        self.client_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .ok_or(AuthError::MissingClientId)
    }
}

/// The per-user configuration directory for the client.
///
/// Honours `MC_RS_CONFIG_DIR` if set, otherwise the platform convention (the `directories`
/// crate's project config folder: `%APPDATA%\mc-rs\config` on Windows, `~/.config/mc-rs` on
/// Linux).
pub fn config_dir() -> Result<PathBuf, AuthError> {
    if let Ok(dir) = std::env::var(CONFIG_DIR_ENV) {
        if !dir.trim().is_empty() {
            return Ok(PathBuf::from(dir));
        }
    }
    directories::ProjectDirs::from("", "", "mc-rs")
        .map(|d| d.config_dir().to_path_buf())
        .ok_or(AuthError::NoConfigDir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_an_empty_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = AuthConfig::load_from(&dir.path().join("nope.json")).expect("load");
        assert!(cfg.client_id.is_none());
        assert!(matches!(
            cfg.require_client_id(),
            Err(AuthError::MissingClientId)
        ));
    }

    #[test]
    fn reads_client_id_from_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(CONFIG_FILE_NAME);
        std::fs::write(
            &path,
            br#"{"client_id":"00000000-1111-2222-3333-444444444444"}"#,
        )
        .expect("write");
        let cfg = AuthConfig::load_from(&path).expect("load");
        assert_eq!(
            cfg.require_client_id().expect("id"),
            "00000000-1111-2222-3333-444444444444"
        );
    }

    #[test]
    fn unknown_keys_are_tolerated() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(CONFIG_FILE_NAME);
        std::fs::write(&path, br#"{"client_id":"x","future_field":42}"#).expect("write");
        assert_eq!(
            AuthConfig::load_from(&path)
                .expect("load")
                .client_id
                .as_deref(),
            Some("x")
        );
    }

    #[test]
    fn malformed_json_is_reported_with_its_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(CONFIG_FILE_NAME);
        std::fs::write(&path, b"{not json").expect("write");
        let err = AuthConfig::load_from(&path).expect_err("should fail");
        assert!(matches!(err, AuthError::Config { .. }));
        assert!(err.to_string().contains(CONFIG_FILE_NAME));
    }

    #[test]
    fn blank_client_id_counts_as_absent() {
        let cfg = AuthConfig {
            client_id: Some("   ".to_owned()),
        };
        assert!(matches!(
            cfg.require_client_id(),
            Err(AuthError::MissingClientId)
        ));
    }

    #[test]
    fn the_built_in_id_fills_in_only_when_nothing_else_is_set() {
        let (config, source) = AuthConfig::with_built_in(AuthConfig::default());
        match built_in_client_id() {
            // A build made with AERO_MSA_CLIENT_ID.
            Some(id) => {
                assert_eq!(config.client_id.as_deref(), Some(id));
                assert_eq!(source, Some(ClientIdSource::BuiltIn));
            }
            // A build made without it (tests normally are): nothing is invented.
            None => {
                assert!(config.client_id.is_none());
                assert_eq!(source, None);
            }
        }
        // An ID of its own always wins over the built-in one.
        let own = AuthConfig {
            client_id: Some("00000000-1111-2222-3333-444444444444".to_owned()),
        };
        assert!(own.require_client_id().is_ok());
    }

    #[test]
    fn missing_client_id_message_names_the_env_var() {
        assert!(AuthError::MissingClientId
            .to_string()
            .contains(CLIENT_ID_ENV));
    }

    #[test]
    fn socks_proxy_uses_remote_dns_and_redacts_password() {
        let proxy = SocksProxy::new(
            "proxy.example",
            1080,
            Some("user:@".to_owned()),
            Some("secret!".to_owned()),
        )
        .expect("proxy");
        assert_eq!(
            proxy.to_reqwest_url(),
            "socks5h://user%3A%40:secret%21@proxy.example:1080"
        );
        assert!(!format!("{proxy:?}").contains("secret"));
    }
}
