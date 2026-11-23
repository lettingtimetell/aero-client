//! Steps 4 and 5 — Minecraft services: the access token, the entitlement, and the profile.

use std::fmt;

use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::client::{body_text, http_error, parse_json, AuthClient};
use crate::error::{snippet, AuthError};
use crate::session::{expires_at, Cape, Profile, Redacted, Skin};
use crate::xbox::XboxToken;

const EP_LOGIN: &str = "Minecraft login_with_xbox";
const EP_ENTITLEMENTS: &str = "Minecraft entitlements";
const EP_PROFILE: &str = "Minecraft profile";

/// The Minecraft services access token and its expiry.
#[derive(Clone)]
pub struct McToken {
    /// The bearer token used for the profile call and for joining servers. **Secret.**
    pub access_token: String,
    /// Unix timestamp (seconds) at which the token stops working.
    pub expires_at: u64,
}

impl fmt::Debug for McToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McToken")
            .field("access_token", &Redacted)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

#[derive(Deserialize)]
struct LoginResponse {
    access_token: String,
    #[serde(default = "default_expires_in")]
    expires_in: u64,
}

fn default_expires_in() -> u64 {
    86_400
}

#[derive(Deserialize)]
struct ProfileResponse {
    id: String,
    name: String,
    #[serde(default)]
    skins: Vec<Skin>,
    #[serde(default)]
    capes: Vec<Cape>,
}

#[derive(Deserialize)]
struct EntitlementsResponse {
    #[serde(default)]
    items: Vec<EntitlementItem>,
}

#[derive(Deserialize)]
struct EntitlementItem {
    name: String,
}

impl AuthClient {
    /// Step 4 — exchange an XSTS token for a Minecraft services access token.
    ///
    /// A `403` here is reported as [`AuthError::ServicesForbidden`], carrying the response body.
    /// It deliberately is *not* [`AuthError::NoEntitlement`]: since Mojang began gating the
    /// Minecraft Services API on approved Azure application IDs, an unapproved client ID fails
    /// with exactly the same status as an account that does not own the game, and only the body
    /// distinguishes them.
    pub async fn login_with_xbox(&self, xsts: &XboxToken) -> Result<McToken, AuthError> {
        let body = json!({ "identityToken": xsts.identity_token() });
        let response = self
            .http
            .post(&self.endpoints.mc_login_with_xbox)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|source| AuthError::Network {
                endpoint: EP_LOGIN,
                source,
            })?;

        let (status, text) = body_text(EP_LOGIN, response).await?;
        if !(200..300).contains(&status) {
            // 403 means the Xbox identity is fine but Minecraft services will not issue a token
            // for it. That is either an unapproved Azure client ID or a missing Java Edition
            // entitlement, so keep the body rather than guessing which.
            if status == 403 {
                return Err(AuthError::ServicesForbidden {
                    body: snippet(&text),
                });
            }
            return Err(http_error(EP_LOGIN, status, &text));
        }
        let parsed: LoginResponse = parse_json(EP_LOGIN, &text)?;
        // Logs and crash reports never show it.
        mc_core::secrets::register(&parsed.access_token);
        Ok(McToken {
            access_token: parsed.access_token,
            expires_at: expires_at(parsed.expires_in),
        })
    }

    /// Step 5 — fetch the profile: UUID, username, skins and capes.
    ///
    /// A `404` from this endpoint is the canonical "this account does not own Minecraft: Java
    /// Edition" signal — the account authenticates fine, it just has no Java profile — so it is
    /// mapped to [`AuthError::NoEntitlement`] rather than a generic HTTP error. A `401` means
    /// the token has expired or been revoked ([`AuthError::TokenRejected`]).
    pub async fn profile(&self, access_token: &str) -> Result<Profile, AuthError> {
        let response = self
            .http
            .get(&self.endpoints.mc_profile)
            .bearer_auth(access_token)
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .await
            .map_err(|source| AuthError::Network {
                endpoint: EP_PROFILE,
                source,
            })?;

        let (status, text) = body_text(EP_PROFILE, response).await?;
        match status {
            200..=299 => {}
            401 | 403 => return Err(AuthError::TokenRejected),
            404 => return Err(AuthError::NoEntitlement),
            _ => return Err(http_error(EP_PROFILE, status, &text)),
        }

        let parsed: ProfileResponse = parse_json(EP_PROFILE, &text)?;
        let uuid = Uuid::parse_str(&parsed.id).map_err(|e| AuthError::MalformedResponse {
            endpoint: EP_PROFILE,
            detail: format!("profile id {:?} is not a UUID: {e}", parsed.id),
        })?;
        Ok(Profile {
            uuid,
            username: parsed.name,
            skins: parsed.skins,
            capes: parsed.capes,
        })
    }

    /// Explicitly check the entitlement store for Java Edition ownership.
    ///
    /// Not needed by the login path — [`profile`](Self::profile) already fails with
    /// [`AuthError::NoEntitlement`] for accounts that do not own the game — but useful for a
    /// clearer diagnostic on the login screen, and for Game Pass accounts that own the game but
    /// have never created a profile.
    pub async fn has_java_entitlement(&self, access_token: &str) -> Result<bool, AuthError> {
        let response = self
            .http
            .get(&self.endpoints.mc_entitlements)
            .bearer_auth(access_token)
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .await
            .map_err(|source| AuthError::Network {
                endpoint: EP_ENTITLEMENTS,
                source,
            })?;

        let (status, text) = body_text(EP_ENTITLEMENTS, response).await?;
        match status {
            200..=299 => {}
            401 | 403 => return Err(AuthError::TokenRejected),
            _ => return Err(http_error(EP_ENTITLEMENTS, status, &text)),
        }
        let parsed: EntitlementsResponse = parse_json(EP_ENTITLEMENTS, &text)?;
        Ok(parsed.items.iter().any(|i| is_java_entitlement(&i.name)))
    }
}

/// Whether an entitlement item name grants Java Edition.
///
/// `product_minecraft`/`game_minecraft` are the purchased forms; the two Game Pass products
/// also grant it. Bedrock, Dungeons and Legends entries appear in the same list and must not
/// count.
fn is_java_entitlement(name: &str) -> bool {
    matches!(
        name,
        "product_minecraft"
            | "game_minecraft"
            | "product_game_pass_pc"
            | "product_game_pass_ultimate"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SkinVariant;

    #[test]
    fn parses_the_documented_profile_body() {
        let body = r#"{
            "id":"986dec87b7ec47ff89ff033fdb95c4b5",
            "name":"HowDoesAuthWork",
            "skins":[{"id":"6a6e65e5-76dd-4c3c-a625-162924514568","state":"ACTIVE",
                      "url":"http://textures.minecraft.net/texture/1a4a","variant":"CLASSIC",
                      "alias":"STEVE"}],
            "capes":[{"id":"5af20372-79e0-4e1f-80f8-6bd8e3135995","state":"ACTIVE",
                      "url":"http://textures.minecraft.net/texture/2340","alias":"Migrator"}]
        }"#;
        let parsed: ProfileResponse = serde_json::from_str(body).expect("parse");
        assert_eq!(
            Uuid::parse_str(&parsed.id).expect("uuid").to_string(),
            "986dec87-b7ec-47ff-89ff-033fdb95c4b5"
        );
        assert_eq!(parsed.skins[0].variant, SkinVariant::Classic);
        assert_eq!(parsed.capes[0].alias.as_deref(), Some("Migrator"));
    }

    #[test]
    fn profile_without_skins_or_capes_still_parses() {
        let parsed: ProfileResponse =
            serde_json::from_str(r#"{"id":"986dec87b7ec47ff89ff033fdb95c4b5","name":"x"}"#)
                .expect("parse");
        assert!(parsed.skins.is_empty());
        assert!(parsed.capes.is_empty());
    }

    #[test]
    fn java_entitlements_are_distinguished_from_the_other_games() {
        assert!(is_java_entitlement("product_minecraft"));
        assert!(is_java_entitlement("game_minecraft"));
        assert!(is_java_entitlement("product_game_pass_pc"));
        assert!(is_java_entitlement("product_game_pass_ultimate"));
        assert!(!is_java_entitlement("product_minecraft_bedrock"));
        assert!(!is_java_entitlement("game_minecraft_bedrock"));
        assert!(!is_java_entitlement("product_dungeons"));
        assert!(!is_java_entitlement("product_legends"));
    }

    #[test]
    fn login_response_defaults_to_a_day() {
        let parsed: LoginResponse =
            serde_json::from_str(r#"{"access_token":"t","token_type":"Bearer"}"#).expect("parse");
        assert_eq!(parsed.expires_in, 86_400);
    }

    #[test]
    fn mc_token_debug_redacts() {
        let t = McToken {
            access_token: "access-secret".to_owned(),
            expires_at: 42,
        };
        assert!(!format!("{t:?}").contains("access-secret"));
    }
}
