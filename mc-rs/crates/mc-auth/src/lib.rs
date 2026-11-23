// `deny`, not `forbid`: the one exception is the Windows DPAPI binding (`dpapi.rs`, three
// hand-declared Win32 functions), allowed on that module alone.
#![deny(unsafe_code)]
#![warn(missing_docs)]

//! Microsoft-account authentication and the session-server join call.
//!
//! It is a reimplementation from public
//! documentation rather than a port. The Java client used `authlib` and Mojang's Yggdrasil
//! service; Yggdrasil is retired (hazard H13) and every account is a Microsoft account now, so
//! none of the original code applies. `util/Session.java` contributes only the *shape* of what
//! the rest of the client needs: a UUID, a username, and an access token.
//!
//! # The flow
//!
//! ```text
//! 1. MSA device code   login.microsoftonline.com/consumers/oauth2/v2.0/{devicecode,token}
//! 2. Xbox Live         user.auth.xboxlive.com/user/authenticate
//! 3. XSTS              xsts.auth.xboxlive.com/xsts/authorize
//! 4. Minecraft token   api.minecraftservices.com/authentication/login_with_xbox
//! 5. Profile           api.minecraftservices.com/minecraft/profile
//! 6. Join (per server) sessionserver.mojang.com/session/minecraft/join
//! ```
//!
//! Steps 1–5 happen once at startup ([`login_device_code`], or silently via
//! [`AuthClient::login`]). Step 6 happens during every encryption handshake, driven by S07.
//!
//! # You must register your own Azure application
//!
//! Mojang's client ID is licensed to Mojang. Using it in a third-party client is not permitted,
//! so there is deliberately no client ID in this source tree and the crate refuses to start a
//! login without one ([`AuthError::MissingClientId`]).
//!
//! To register one, once:
//!
//! 1. Sign in to the [Microsoft Entra admin center](https://entra.microsoft.com) →
//!    **App registrations** → **New registration**.
//! 2. Give it any name. Under *Supported account types* pick
//!    **Personal Microsoft accounts only** — Xbox Live sign-in is only grantable to the
//!    `consumers` tenant, which is the tenant this crate authenticates against.
//! 3. Leave the redirect URI empty; the device-code grant does not use one.
//! 4. After creating it, open **Authentication** and set
//!    *Allow public client flows* → **Yes**. Without this the token endpoint rejects the
//!    device-code grant with `invalid_client`.
//! 5. Do **not** create a client secret. This is a public client.
//! 6. Copy the **Application (client) ID** (a GUID) from the Overview page.
//! 7. New applications also need Mojang's approval to call the Minecraft API — submit the
//!    client ID through Microsoft's Minecraft-launcher review form (<https://aka.ms/mce-reviewappid>).
//!    Until that is granted, steps 1–3 succeed and step 4 fails.
//!
//! Then provide the ID, one of three ways (the first that is set wins):
//!
//! ```text
//! MC_RS_MSA_CLIENT_ID=<guid>                        # environment variable at run time
//! <config dir>/auth.json  →  { "client_id": "<guid>" }
//! AERO_MSA_CLIENT_ID=<guid> cargo build ...         # compiled in (config::BUILT_IN_CLIENT_ID)
//! ```
//!
//! The first two are development overrides; a release build compiles the ID in.
//!
//! `<config dir>` is the platform's per-user config folder for `mc-rs` (`%APPDATA%\mc-rs\config`
//! on Windows, `~/.config/mc-rs` on Linux), overridable with `MC_RS_CONFIG_DIR`. The accounts
//! added in the client live there too ([`accounts`]).
//!
//! [`server_hash`] and [`join_server`] need none of this — only the login flows do.
//!
//! # Secrets
//!
//! Access tokens, refresh tokens, device codes and Xbox user hashes are never logged and are
//! redacted in every `Debug` implementation in this crate ([`Session`], [`Credentials`],
//! [`MsaTokens`], [`XboxToken`], [`McToken`], [`DeviceCodeSession`]). The on-disk cache is
//! encrypted; read [`cache`] for its threat model, which is narrower than "encrypted" suggests.
//!
//! # Example
//!
//! ```no_run
//! # async fn demo() -> Result<(), mc_auth::AuthError> {
//! let client = mc_auth::AuthClient::discover()?;
//! let cache = mc_auth::TokenCache::discover()?;
//!
//! // Silent if a cached token or refresh token still works; interactive otherwise.
//! let auth = client
//!     .login(&cache, |prompt| {
//!         println!("Go to {} and enter {}", prompt.verification_uri, prompt.user_code);
//!     })
//!     .await?;
//!
//! println!("signed in as {}", auth.session.username);
//! # Ok(())
//! # }
//! ```

pub mod accounts;
pub mod cache;
mod client;
pub mod config;
#[cfg(windows)]
#[allow(unsafe_code)]
mod dpapi;
mod error;
mod flow;
mod game_profile;
mod hash;
mod join;
mod msa;
pub mod offline;
pub mod seal;
mod services;
mod session;
mod xbox;

pub use accounts::{
    AccountList, AccountSecrets, AccountStore, ActiveAccount, JoinSession, StoredAccount,
};
pub use cache::TokenCache;
pub use client::{AuthClient, Endpoints, PollTuning};
pub use config::{AuthConfig, ClientIdSource, SocksProxy};
pub use error::{AuthError, CLIENT_ID_ENV};
pub use flow::REFRESH_SKEW;
pub use game_profile::{
    java_uuid_from_string, normalize_legacy_skull_owner, read_game_profile_from_nbt,
    write_game_profile_preserving, write_game_profile_to_nbt, GameProfile, GameProfileProperty,
};
pub use hash::{remembered_textures_verdict, server_hash, signed_hex, verify_textures_signature};
pub use msa::{DeviceCodePrompt, DeviceCodeSession, MsaTokens};
pub use services::McToken;
pub use session::{
    unix_now, Authenticated, Cape, Credentials, Profile, Session, Skin, SkinVariant, TokenFacts,
};
pub use xbox::{XboxToken, MINECRAFT_RELYING_PARTY};

/// Log in interactively with the device-code flow, using the ambient configuration.
///
/// `cb` is invoked once with the code and URL to show the user, then the call blocks (async)
/// until the user finishes signing in, declines, or the code expires.
///
/// This is the frozen §7 entry point. [`AuthClient::login`] is usually the better choice in the
/// real client: it reuses the token cache and only falls back to this when it has to.
///
/// # Errors
///
/// [`AuthError::MissingClientId`] if no Azure application is configured;
/// [`AuthError::Declined`] or [`AuthError::DeviceCodeExpired`] if the user does not complete the
/// sign-in; [`AuthError::NoXboxAccount`], [`AuthError::ChildAccount`] and friends for the Xbox
/// Live failures; [`AuthError::NoEntitlement`] if the account does not own Java Edition.
pub async fn login_device_code(cb: impl Fn(&DeviceCodePrompt)) -> Result<Session, AuthError> {
    login_device_code_with_proxy(cb, None).await
}

/// Run the device-code flow with an optional Java-compatible SOCKS5 proxy for every Microsoft,
/// Xbox, Minecraft-services, and profile request.
pub async fn login_device_code_with_proxy(
    cb: impl Fn(&DeviceCodePrompt),
    proxy: Option<&SocksProxy>,
) -> Result<Session, AuthError> {
    Ok(
        AuthClient::new_with_proxy(AuthConfig::load()?, proxy.cloned())?
            .login_device_code(cb)
            .await?
            .session,
    )
}

/// Exchange a stored Microsoft refresh token for a fresh session, without user interaction.
///
/// Note that Microsoft rotates refresh tokens: the one used here may no longer be valid
/// afterwards. [`AuthClient::refresh`] returns the replacement (inside
/// [`Authenticated::credentials`]) and [`AuthClient::login`] persists it for you.
///
/// # Errors
///
/// [`AuthError::MissingClientId`], or [`AuthError::Oauth`] if the refresh token has expired or
/// been revoked and the user must sign in again.
pub async fn refresh(refresh_token: &str) -> Result<Session, AuthError> {
    refresh_with_proxy(refresh_token, None).await
}

/// Exchange a refresh token through an optional Java-compatible SOCKS5 proxy.
pub async fn refresh_with_proxy(
    refresh_token: &str,
    proxy: Option<&SocksProxy>,
) -> Result<Session, AuthError> {
    Ok(
        AuthClient::new_with_proxy(AuthConfig::load()?, proxy.cloned())?
            .refresh(refresh_token)
            .await?
            .session,
    )
}

/// Register a pending server join with the session server (step 6).
///
/// Call this after computing [`server_hash`] and before sending
/// `C01PacketEncryptionResponse`; the server verifies it via `hasJoined` immediately
/// afterwards.
///
/// # Errors
///
/// [`AuthError::TokenRejected`] if the access token has expired or is invalid — the caller
/// should refresh and retry once — [`AuthError::JoinRefused`] when the session server gave
/// another reason (vanilla shows its message), [`AuthError::JoinUnreadable`] when its answer
/// could not be read, or [`AuthError::Network`] when the request never completed.
pub async fn join_server(s: &Session, server_hash: &str) -> Result<(), AuthError> {
    join_server_with_proxy(s, server_hash, None).await
}

/// Register a pending server join through an optional Java-compatible SOCKS5 proxy.
///
/// The no-proxy [`join_server`] entry point remains frozen for callers that do not have launcher
/// transport settings.  `mc-client` uses this seam so the same proxy reaches every session-server
/// request in the encryption handshake.
pub async fn join_server_with_proxy(
    s: &Session,
    server_hash: &str,
    proxy: Option<&SocksProxy>,
) -> Result<(), AuthError> {
    AuthClient::new_with_proxy(AuthConfig::default(), proxy.cloned())?
        .join_server(s, server_hash)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The §7 frozen contract, checked by the compiler rather than by reading.
    #[test]
    fn frozen_api_signatures_are_intact() {
        fn assert_send<T: Send>(_: T) {}

        let session = Session::new(uuid::Uuid::nil(), String::new(), String::new());
        let _: String = server_hash("id", &[0u8; 16], &[0u8; 162]);
        assert_send(login_device_code(|_p: &DeviceCodePrompt| {}));
        assert_send(refresh("token"));
        assert_send(join_server(&session, "hash"));
    }
}
