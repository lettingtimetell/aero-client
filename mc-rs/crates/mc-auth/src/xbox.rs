//! Steps 2 and 3 — Xbox Live user authentication, then XSTS authorization.
//!
//! The MSA access token is not usable against Minecraft services directly. It is exchanged for
//! an Xbox Live user token, which is exchanged for an XSTS token scoped to the
//! `rp://api.minecraftservices.com/` relying party. The XSTS step is where the interesting
//! user-facing failures live: no Xbox profile, banned, unsupported region, and — the one that
//! only bites third-party applications — child accounts outside a Microsoft family group.

use std::fmt;

use serde::Deserialize;
use serde_json::json;

use crate::client::{body_text, http_error, parse_json, AuthClient};
use crate::error::AuthError;
use crate::session::Redacted;

const EP_XBL: &str = "Xbox Live authenticate";
const EP_XSTS: &str = "XSTS authorize";

/// The relying party that scopes the XSTS token to Minecraft services.
///
/// The trailing slash matters — `rp://api.minecraftservices.com` without it is rejected.
pub const MINECRAFT_RELYING_PARTY: &str = "rp://api.minecraftservices.com/";

/// An Xbox Live or XSTS token plus the user hash that must accompany it.
#[derive(Clone)]
pub struct XboxToken {
    /// The JWT. **Secret.**
    pub token: String,
    /// The `uhs` display claim — half of the `XBL3.0 x=<uhs>;<token>` identity header.
    pub user_hash: String,
}

impl fmt::Debug for XboxToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XboxToken")
            .field("token", &Redacted)
            .field("user_hash", &Redacted)
            .finish()
    }
}

impl XboxToken {
    /// The `identityToken` value Minecraft services expects.
    #[must_use]
    pub fn identity_token(&self) -> String {
        format!("XBL3.0 x={};{}", self.user_hash, self.token)
    }
}

#[derive(Deserialize)]
struct XboxTokenResponse {
    #[serde(rename = "Token")]
    token: String,
    #[serde(rename = "DisplayClaims")]
    display_claims: DisplayClaims,
}

#[derive(Deserialize)]
struct DisplayClaims {
    xui: Vec<Xui>,
}

#[derive(Deserialize)]
struct Xui {
    uhs: String,
}

#[derive(Deserialize)]
struct XstsErrorResponse {
    #[serde(rename = "XErr")]
    xerr: u64,
    #[serde(rename = "Message")]
    #[serde(default)]
    message: Option<String>,
    #[serde(rename = "Redirect")]
    #[serde(default)]
    redirect: Option<String>,
}

impl AuthClient {
    /// Step 2 — exchange an MSA access token for an Xbox Live user token.
    pub async fn xbox_authenticate(&self, msa_access_token: &str) -> Result<XboxToken, AuthError> {
        // The `d=` prefix marks the RPS ticket as coming from a v2.0 (MSA) access token.
        let body = json!({
            "Properties": {
                "AuthMethod": "RPS",
                "SiteName": "user.auth.xboxlive.com",
                "RpsTicket": format!("d={msa_access_token}"),
            },
            "RelyingParty": "http://auth.xboxlive.com",
            "TokenType": "JWT",
        });

        let response = self
            .http
            .post(&self.endpoints.xbl_authenticate)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|source| AuthError::Network {
                endpoint: EP_XBL,
                source,
            })?;

        let (status, text) = body_text(EP_XBL, response).await?;
        if !(200..300).contains(&status) {
            return Err(http_error(EP_XBL, status, &text));
        }
        into_token(EP_XBL, &text)
    }

    /// Step 3 — exchange an Xbox Live token for an XSTS token for Minecraft services.
    ///
    /// Maps the documented `XErr` codes onto typed errors; see [`AuthError`].
    pub async fn xsts_authorize(&self, xbl: &XboxToken) -> Result<XboxToken, AuthError> {
        let body = json!({
            "Properties": {
                "SandboxId": "RETAIL",
                "UserTokens": [xbl.token],
            },
            "RelyingParty": MINECRAFT_RELYING_PARTY,
            "TokenType": "JWT",
        });

        let response = self
            .http
            .post(&self.endpoints.xsts_authorize)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|source| AuthError::Network {
                endpoint: EP_XSTS,
                source,
            })?;

        let (status, text) = body_text(EP_XSTS, response).await?;
        if !(200..300).contains(&status) {
            return Err(xsts_error(status, &text));
        }
        into_token(EP_XSTS, &text)
    }
}

fn into_token(endpoint: &'static str, body: &str) -> Result<XboxToken, AuthError> {
    let parsed: XboxTokenResponse = parse_json(endpoint, body)?;
    let user_hash = parsed
        .display_claims
        .xui
        .into_iter()
        .next()
        .map(|x| x.uhs)
        .ok_or(AuthError::MalformedResponse {
            endpoint,
            detail: "DisplayClaims.xui was empty, so there is no user hash".to_owned(),
        })?;
    // Logs and crash reports never show it.
    mc_core::secrets::register(&parsed.token);
    Ok(XboxToken {
        token: parsed.token,
        user_hash,
    })
}

/// Map an XSTS failure body onto a typed error.
///
/// Exposed to the crate (and unit-tested directly) because these are the errors real users hit
/// and the mapping is the part worth getting right.
pub(crate) fn xsts_error(status: u16, body: &str) -> AuthError {
    let Ok(parsed) = serde_json::from_str::<XstsErrorResponse>(body) else {
        return http_error(EP_XSTS, status, body);
    };
    let message = parsed.message.unwrap_or_default();
    match parsed.xerr {
        2_148_916_227 => AuthError::XboxBanned,
        2_148_916_233 => AuthError::NoXboxAccount,
        2_148_916_235 => AuthError::XboxRegionUnavailable,
        2_148_916_236 | 2_148_916_237 => AuthError::AdultVerificationRequired,
        2_148_916_238 => AuthError::ChildAccount {
            redirect: parsed.redirect,
        },
        xerr => AuthError::Xsts { xerr, message },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Predicate over a mapped error, so the table-driven tests stay readable.
    type Check = fn(&AuthError) -> bool;

    #[test]
    fn identity_token_has_the_documented_shape() {
        let t = XboxToken {
            token: "jwt-value".to_owned(),
            user_hash: "1234567890".to_owned(),
        };
        assert_eq!(t.identity_token(), "XBL3.0 x=1234567890;jwt-value");
    }

    #[test]
    fn xbox_token_debug_redacts_everything_sensitive() {
        let t = XboxToken {
            token: "jwt-value".to_owned(),
            user_hash: "1234567890".to_owned(),
        };
        let rendered = format!("{t:?}");
        assert!(!rendered.contains("jwt-value"));
        assert!(!rendered.contains("1234567890"));
    }

    #[test]
    fn parses_the_documented_success_body() {
        let body = r#"{
            "IssueInstant":"2020-12-07T19:52:08.4463796Z",
            "NotAfter":"2020-12-21T19:52:08.4463796Z",
            "Token":"token",
            "DisplayClaims":{"xui":[{"uhs":"userhash"}]}
        }"#;
        let t = into_token(EP_XBL, body).expect("parse");
        assert_eq!(t.token, "token");
        assert_eq!(t.user_hash, "userhash");
    }

    #[test]
    fn empty_display_claims_is_a_typed_error_not_a_panic() {
        let body = r#"{"Token":"t","DisplayClaims":{"xui":[]}}"#;
        assert!(matches!(
            into_token(EP_XBL, body),
            Err(AuthError::MalformedResponse { .. })
        ));
    }

    #[test]
    fn maps_no_xbox_account() {
        let e = xsts_error(
            401,
            r#"{"Identity":"0","XErr":2148916233,"Message":"","Redirect":""}"#,
        );
        assert!(matches!(e, AuthError::NoXboxAccount));
    }

    #[test]
    fn maps_child_account_with_its_redirect() {
        let body = r#"{"Identity":"0","XErr":2148916238,"Message":"",
                       "Redirect":"https://start.ui.xboxlive.com/AddChildToFamily"}"#;
        match xsts_error(401, body) {
            AuthError::ChildAccount { redirect } => {
                assert_eq!(
                    redirect.as_deref(),
                    Some("https://start.ui.xboxlive.com/AddChildToFamily")
                );
            }
            other => panic!("expected ChildAccount, got {other:?}"),
        }
    }

    #[test]
    fn maps_the_rest_of_the_documented_codes() {
        let cases: [(u64, Check); 5] = [
            (2_148_916_227, |e| matches!(e, AuthError::XboxBanned)),
            (2_148_916_235, |e| {
                matches!(e, AuthError::XboxRegionUnavailable)
            }),
            (2_148_916_236, |e| {
                matches!(e, AuthError::AdultVerificationRequired)
            }),
            (2_148_916_237, |e| {
                matches!(e, AuthError::AdultVerificationRequired)
            }),
            (2_148_916_262, |e| {
                matches!(
                    e,
                    AuthError::Xsts {
                        xerr: 2_148_916_262,
                        ..
                    }
                )
            }),
        ];
        for (xerr, check) in cases {
            let e = xsts_error(401, &format!(r#"{{"XErr":{xerr},"Message":"m"}}"#));
            assert!(check(&e), "XErr {xerr} mapped to {e:?}");
        }
    }

    #[test]
    fn non_json_xsts_failures_stay_http_errors() {
        let e = xsts_error(500, "internal server error");
        assert!(matches!(e, AuthError::Http { status: 500, .. }));
    }
}
