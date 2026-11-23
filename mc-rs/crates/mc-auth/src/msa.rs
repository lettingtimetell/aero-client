//! Step 1 — the Microsoft OAuth2 device authorization grant (RFC 8628).
//!
//! Device code is the right grant for a game client: no embedded browser, no loopback HTTP
//! server, no client secret, and it works over SSH or on a machine with no browser at all. The
//! cost is that the user has to type a short code on another device.

use std::fmt;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::client::{body_text, http_error, parse_json, AuthClient};
use crate::error::AuthError;
use crate::session::Redacted;

const EP_DEVICE_CODE: &str = "Microsoft device authorization";
const EP_TOKEN: &str = "Microsoft token";

/// What the UI must show the user to complete a device-code login.
///
/// S16's login screen renders `verification_uri` and `user_code` prominently — `message` is
/// Microsoft's own localisable sentence and is a reasonable fallback, but a hand-built screen
/// with a copy button and a QR code for the URI is much better.
#[derive(Debug, Clone)]
pub struct DeviceCodePrompt {
    /// The short code the user types, e.g. `"A1B2C3D4"`.
    pub user_code: String,
    /// The URL the user opens, normally `https://microsoft.com/link`.
    pub verification_uri: String,
    /// Microsoft's human-readable instruction sentence.
    pub message: String,
    /// How long the code stays valid (typically 15 minutes).
    pub expires_in: Duration,
    /// How often Microsoft wants to be polled (typically every 5 seconds).
    pub interval: Duration,
}

/// A device-code login in progress.
///
/// Holds the secret `device_code` alongside the [`DeviceCodePrompt`]; `Debug` redacts it,
/// because it is a bearer credential for the pending sign-in.
#[derive(Clone)]
pub struct DeviceCodeSession {
    /// What to show the user.
    pub prompt: DeviceCodePrompt,
    pub(crate) device_code: String,
}

impl fmt::Debug for DeviceCodeSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceCodeSession")
            .field("prompt", &self.prompt)
            .field("device_code", &Redacted)
            .finish()
    }
}

/// Microsoft access + refresh tokens. Both fields are secret; `Debug` redacts them.
#[derive(Clone)]
pub struct MsaTokens {
    /// The MSA access token, consumed immediately by the Xbox Live step.
    pub access_token: String,
    /// The MSA refresh token, cached so the next start needs no user interaction.
    ///
    /// Absent only if the authorization server ignored `offline_access`.
    pub refresh_token: Option<String>,
}

impl fmt::Debug for MsaTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MsaTokens")
            .field("access_token", &Redacted)
            .field("refresh_token", &Redacted)
            .finish()
    }
}

#[derive(Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    message: Option<String>,
    #[serde(default = "default_expires_in")]
    expires_in: u64,
    #[serde(default = "default_interval")]
    interval: u64,
}

fn default_expires_in() -> u64 {
    900
}

/// The longest a device code is taken to live and the longest wait between
/// polls, whatever the server says (Microsoft's are 15 minutes and 5 seconds). A nonsense value
/// would otherwise overflow the poll loop's deadline (`Instant + Duration` panics) or park the
/// sign-in screen for days.
const MAX_DEVICE_CODE_LIFETIME_SECS: u64 = 60 * 60;
const MAX_POLL_INTERVAL_SECS: u64 = 60;

/// The server's `expires_in` and `interval`, clamped (see the constants above).
fn device_code_timing(expires_in: u64, interval: u64) -> (Duration, Duration) {
    (
        Duration::from_secs(expires_in.min(MAX_DEVICE_CODE_LIFETIME_SECS)),
        Duration::from_secs(interval.min(MAX_POLL_INTERVAL_SECS)),
    )
}

fn default_interval() -> u64 {
    5
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
}

#[derive(Deserialize)]
struct OauthErrorResponse {
    error: String,
    #[serde(default)]
    error_description: Option<String>,
}

impl AuthClient {
    /// Begin a device-code login: ask Microsoft for a code pair.
    ///
    /// The user has `prompt.expires_in` to complete the sign-in, so only call this once the
    /// user has actually asked to log in.
    pub async fn device_code_start(&self) -> Result<DeviceCodeSession, AuthError> {
        let client_id = self.client_id()?;
        let response = self
            .http
            .post(&self.endpoints.msa_device_code)
            .form(&[("client_id", client_id), ("scope", self.scope())])
            .send()
            .await
            .map_err(|source| AuthError::Network {
                endpoint: EP_DEVICE_CODE,
                source,
            })?;

        let (status, body) = body_text(EP_DEVICE_CODE, response).await?;
        if !(200..300).contains(&status) {
            return Err(oauth_error(EP_DEVICE_CODE, status, &body));
        }
        let parsed: DeviceCodeResponse = parse_json(EP_DEVICE_CODE, &body)?;
        // The device code is a bearer credential for the pending sign-in.
        mc_core::secrets::register(&parsed.device_code);
        let (expires_in, interval) = device_code_timing(parsed.expires_in, parsed.interval);
        Ok(DeviceCodeSession {
            prompt: DeviceCodePrompt {
                user_code: parsed.user_code,
                message: parsed.message.unwrap_or_else(|| {
                    format!(
                        "Open {} and enter the code to sign in.",
                        parsed.verification_uri
                    )
                }),
                verification_uri: parsed.verification_uri,
                expires_in,
                interval,
            },
            device_code: parsed.device_code,
        })
    }

    /// Poll the token endpoint until the user finishes signing in, or the code dies.
    ///
    /// Handles the four polling states the protocol defines:
    /// `authorization_pending` (keep waiting), `slow_down` (back off, then keep waiting),
    /// `expired_token` ([`AuthError::DeviceCodeExpired`]) and
    /// `authorization_declined`/`access_denied` ([`AuthError::Declined`]).
    pub async fn device_code_poll(
        &self,
        session: &DeviceCodeSession,
    ) -> Result<MsaTokens, AuthError> {
        let client_id = self.client_id()?.to_owned();
        // Never a panic, even for a prompt built with a nonsense lifetime.
        let now = Instant::now();
        let deadline = now
            .checked_add(session.prompt.expires_in)
            .unwrap_or_else(|| now + Duration::from_secs(MAX_DEVICE_CODE_LIFETIME_SECS));
        let mut interval = session.prompt.interval.max(self.poll.min_interval);

        loop {
            tokio::time::sleep(interval).await;

            if Instant::now() >= deadline {
                return Err(AuthError::DeviceCodeExpired);
            }

            let response = self
                .http
                .post(&self.endpoints.msa_token)
                .form(&[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("client_id", client_id.as_str()),
                    ("device_code", session.device_code.as_str()),
                ])
                .send()
                .await
                .map_err(|source| AuthError::Network {
                    endpoint: EP_TOKEN,
                    source,
                })?;

            let (status, body) = body_text(EP_TOKEN, response).await?;
            if (200..300).contains(&status) {
                let parsed: TokenResponse = parse_json(EP_TOKEN, &body)?;
                return Ok(MsaTokens::registered(
                    parsed.access_token,
                    parsed.refresh_token,
                ));
            }

            // Anything non-2xx is either a polling state or a real failure.
            let code = serde_json::from_str::<OauthErrorResponse>(&body)
                .map(|e| e.error)
                .unwrap_or_default();
            match code.as_str() {
                "authorization_pending" => {
                    tracing::trace!("device code still pending");
                }
                "slow_down" => {
                    interval += self.poll.slow_down_increment;
                    tracing::debug!(?interval, "device code polling backed off");
                }
                "expired_token" | "code_expired" => return Err(AuthError::DeviceCodeExpired),
                "authorization_declined" | "access_denied" => return Err(AuthError::Declined),
                _ => return Err(oauth_error(EP_TOKEN, status, &body)),
            }
        }
    }

    /// Exchange a refresh token for a fresh MSA access token (and usually a new refresh token).
    ///
    /// Microsoft rotates refresh tokens, so the returned one must replace the stored one.
    pub async fn msa_refresh(&self, refresh_token: &str) -> Result<MsaTokens, AuthError> {
        let client_id = self.client_id()?;
        let response = self
            .http
            .post(&self.endpoints.msa_token)
            .form(&[
                ("grant_type", "refresh_token"),
                ("client_id", client_id),
                ("refresh_token", refresh_token),
                ("scope", self.scope()),
            ])
            .send()
            .await
            .map_err(|source| AuthError::Network {
                endpoint: EP_TOKEN,
                source,
            })?;

        let (status, body) = body_text(EP_TOKEN, response).await?;
        if !(200..300).contains(&status) {
            return Err(oauth_error(EP_TOKEN, status, &body));
        }
        let parsed: TokenResponse = parse_json(EP_TOKEN, &body)?;
        Ok(MsaTokens::registered(
            parsed.access_token,
            // If Microsoft declines to rotate, keep using the one we have.
            parsed
                .refresh_token
                .or_else(|| Some(refresh_token.to_owned())),
        ))
    }
}

impl MsaTokens {
    /// Tokens just received, registered with `mc_core::secrets` so logs and
    /// crash reports never show them.
    fn registered(access_token: String, refresh_token: Option<String>) -> Self {
        mc_core::secrets::register(&access_token);
        if let Some(refresh_token) = &refresh_token {
            mc_core::secrets::register(refresh_token);
        }
        MsaTokens {
            access_token,
            refresh_token,
        }
    }
}

/// Turn an OAuth2 error body into a typed error, falling back to the raw HTTP status.
fn oauth_error(endpoint: &'static str, status: u16, body: &str) -> AuthError {
    match serde_json::from_str::<OauthErrorResponse>(body) {
        Ok(e) => AuthError::Oauth {
            description: e.error_description.unwrap_or_default(),
            error: e.error,
        },
        Err(_) => http_error(endpoint, status, body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_code_session_debug_redacts_the_code() {
        let s = DeviceCodeSession {
            prompt: DeviceCodePrompt {
                user_code: "A1B2C3D4".to_owned(),
                verification_uri: "https://microsoft.com/link".to_owned(),
                message: "go here".to_owned(),
                expires_in: Duration::from_secs(900),
                interval: Duration::from_secs(5),
            },
            device_code: "the-secret-device-code".to_owned(),
        };
        let rendered = format!("{s:?}");
        assert!(!rendered.contains("the-secret-device-code"));
        assert!(rendered.contains("A1B2C3D4"));
    }

    #[test]
    fn msa_tokens_debug_redacts_both() {
        let t = MsaTokens {
            access_token: "access-secret".to_owned(),
            refresh_token: Some("refresh-secret".to_owned()),
        };
        let rendered = format!("{t:?}");
        assert!(!rendered.contains("access-secret"));
        assert!(!rendered.contains("refresh-secret"));
    }

    #[test]
    fn oauth_error_prefers_the_typed_body() {
        let e = oauth_error(
            EP_TOKEN,
            400,
            r#"{"error":"invalid_client","error_description":"AADSTS7000218: bad app"}"#,
        );
        match e {
            AuthError::Oauth { error, description } => {
                assert_eq!(error, "invalid_client");
                assert!(description.contains("AADSTS7000218"));
            }
            other => panic!("expected Oauth, got {other:?}"),
        }
    }

    #[test]
    fn oauth_error_falls_back_to_http_for_non_json() {
        let e = oauth_error(EP_TOKEN, 503, "<html>gateway timeout</html>");
        assert!(matches!(e, AuthError::Http { status: 503, .. }));
    }

    /// Microsoft's values pass through; nonsense ones are clamped instead of
    /// overflowing the poll loop's deadline.
    #[test]
    fn device_code_timing_is_clamped() {
        assert_eq!(
            device_code_timing(900, 5),
            (Duration::from_secs(900), Duration::from_secs(5))
        );
        assert_eq!(
            device_code_timing(u64::MAX, u64::MAX),
            (Duration::from_secs(3600), Duration::from_secs(60))
        );
        // The clamped lifetime can be added to any `Instant` a process sees.
        let (lifetime, _) = device_code_timing(u64::MAX, 0);
        assert!(Instant::now().checked_add(lifetime).is_some());
    }

    #[test]
    fn device_code_response_defaults_are_sane() {
        let parsed: DeviceCodeResponse = serde_json::from_str(
            r#"{"device_code":"d","user_code":"u","verification_uri":"https://microsoft.com/link"}"#,
        )
        .expect("parse");
        assert_eq!(parsed.expires_in, 900);
        assert_eq!(parsed.interval, 5);
        assert!(parsed.message.is_none());
    }
}
