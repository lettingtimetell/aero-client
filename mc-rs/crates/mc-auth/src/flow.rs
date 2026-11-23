//! The orchestration: the six documented steps stitched into two entry points.

use std::time::Duration;

use crate::cache::TokenCache;
use crate::client::AuthClient;
use crate::error::AuthError;
use crate::msa::{DeviceCodePrompt, MsaTokens};
use crate::session::{Authenticated, Credentials, Session};

/// How much slack to leave before a cached token's expiry before refreshing it.
///
/// A token that expires mid-handshake produces a confusing "Failed to verify username" on the
/// server side, so the client refreshes well ahead of the deadline.
pub const REFRESH_SKEW: Duration = Duration::from_secs(5 * 60);

impl AuthClient {
    /// The full interactive login: device code, Xbox Live, XSTS, Minecraft, profile.
    ///
    /// `on_prompt` is called exactly once, as soon as Microsoft issues the code, and must not
    /// block: it is invoked from inside the async task. In the real client it pushes the code
    /// onto the login screen (S16); in a headless tool it prints it.
    pub async fn login_device_code(
        &self,
        on_prompt: impl Fn(&DeviceCodePrompt),
    ) -> Result<Authenticated, AuthError> {
        let session = self.device_code_start().await?;
        // Never the user code: while it is live (about 15 minutes) anyone who
        // reads it in a shared log can finish this pending sign-in with an account of their own.
        // It goes only to the sign-in screen, through `on_prompt`.
        tracing::info!(
            uri = %session.prompt.verification_uri,
            "waiting for the user to complete the Microsoft sign-in"
        );
        on_prompt(&session.prompt);
        let msa = self.device_code_poll(&session).await?;
        self.complete(msa).await
    }

    /// Silent re-login from a stored Microsoft refresh token.
    pub async fn refresh(&self, refresh_token: &str) -> Result<Authenticated, AuthError> {
        let msa = self.msa_refresh(refresh_token).await?;
        self.complete(msa).await
    }

    /// Steps 2–5: everything downstream of holding a valid MSA access token.
    async fn complete(&self, msa: MsaTokens) -> Result<Authenticated, AuthError> {
        let xbl = self.xbox_authenticate(&msa.access_token).await?;
        let xsts = self.xsts_authorize(&xbl).await?;
        let mc = self.login_with_xbox(&xsts).await?;
        let profile = self.profile(&mc.access_token).await?;

        let session = Session::new(profile.uuid, &profile.username, &mc.access_token);
        let credentials = Credentials {
            msa_refresh_token: msa.refresh_token.unwrap_or_default(),
            mc_access_token: mc.access_token,
            expires_at: mc.expires_at,
            uuid: profile.uuid,
            username: profile.username.clone(),
        };
        tracing::info!(player = %profile.username, uuid = %profile.uuid, "signed in");
        Ok(Authenticated {
            session,
            profile,
            credentials,
        })
    }

    /// Startup path: reuse the cache if possible, refresh it silently if stale, and only fall
    /// back to an interactive device-code login if neither works.
    ///
    /// Whatever route it takes, the resulting credentials are written back to `cache`, so the
    /// next start is silent. A cache that cannot be read is logged and discarded rather than
    /// failing the login — a corrupt file must not lock the user out of the game.
    pub async fn login(
        &self,
        cache: &TokenCache,
        on_prompt: impl Fn(&DeviceCodePrompt),
    ) -> Result<Authenticated, AuthError> {
        match cache.load() {
            Ok(Some(creds)) => {
                if creds.is_fresh(REFRESH_SKEW) {
                    tracing::debug!("reusing the cached Minecraft access token");
                    // The cached token is still good, but the profile is worth re-reading: the
                    // username may have changed, and this validates the token in one request.
                    match self.profile(&creds.mc_access_token).await {
                        Ok(profile) => {
                            let session = Session::new(
                                profile.uuid,
                                &profile.username,
                                &creds.mc_access_token,
                            );
                            let credentials = Credentials {
                                username: profile.username.clone(),
                                uuid: profile.uuid,
                                ..creds
                            };
                            cache.save(&credentials).unwrap_or_else(warn_cache_write);
                            return Ok(Authenticated {
                                session,
                                profile,
                                credentials,
                            });
                        }
                        Err(AuthError::TokenRejected) => {
                            tracing::debug!("cached token was rejected; refreshing");
                        }
                        Err(e) => return Err(e),
                    }
                }
                if !creds.msa_refresh_token.is_empty() {
                    match self.refresh(&creds.msa_refresh_token).await {
                        Ok(auth) => {
                            cache
                                .save(&auth.credentials)
                                .unwrap_or_else(warn_cache_write);
                            return Ok(auth);
                        }
                        Err(e @ (AuthError::Network { .. } | AuthError::Http { .. })) => {
                            // A transient outage must not silently drop the user into an
                            // interactive login they cannot complete offline.
                            return Err(e);
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "silent refresh failed; falling back to device code");
                        }
                    }
                }
            }
            Ok(None) => tracing::debug!("no cached credentials"),
            Err(e) => tracing::warn!(error = %e, "discarding unusable cached credentials"),
        }

        let auth = self.login_device_code(on_prompt).await?;
        cache
            .save(&auth.credentials)
            .unwrap_or_else(warn_cache_write);
        Ok(auth)
    }
}

fn warn_cache_write(e: AuthError) {
    // Failing to persist is annoying (the next start is interactive) but not fatal.
    tracing::warn!(error = %e, "could not write the token cache");
}
