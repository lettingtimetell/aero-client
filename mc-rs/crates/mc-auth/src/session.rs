//! The session and profile types the rest of the client consumes.
//!
//! // java: util/Session.java — same three fields (uuid, username, token); the `legacy`/`mojang`
//! // session type is gone because every account is a Microsoft account now (hazard H13).

use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

/// A logged-in player: everything the network layer needs to join a server.
///
/// `Debug` is implemented by hand so that a stray `dbg!` or `tracing::debug!("{session:?}")`
/// cannot leak the access token into a log file or a crash report. The token field itself is
/// private for the same reason: a plain field read looks like any other field access, so nothing
/// stops it from being logged, formatted, or copied into a wider struct by accident. Reading it
/// goes through [`Session::expose`] instead — a name that is conspicuous at every call site,
/// mirroring `mc-client`'s `AccessToken::expose`.
#[derive(Clone, PartialEq, Eq)]
pub struct Session {
    /// The player's Minecraft UUID.
    pub uuid: Uuid,
    /// The player's current Minecraft username.
    pub username: String,
    /// The Minecraft services access token. **Secret.** Read via [`Session::expose`], never
    /// directly.
    access_token: String,
}

impl Session {
    /// Construct a session from its three parts.
    ///
    /// The token is registered with `mc_core::secrets`, so logs and crash reports never show it
    /// even if it is formatted by mistake.
    #[must_use]
    pub fn new(uuid: Uuid, username: impl Into<String>, access_token: impl Into<String>) -> Self {
        let access_token = access_token.into();
        mc_core::secrets::register(&access_token);
        Self {
            uuid,
            username: username.into(),
            access_token,
        }
    }

    /// The UUID in the dashless form the session server and the protocol expect.
    ///
    /// // java: Session.getPlayerID() — vanilla stores the dashless form as a String.
    #[must_use]
    pub fn uuid_undashed(&self) -> String {
        let mut buf = Uuid::encode_buffer();
        self.uuid.simple().encode_lower(&mut buf).to_owned()
    }

    /// The access token itself.
    ///
    /// Named to be conspicuous at the call site: every use is a point where the secret leaves
    /// this struct, and there should never be many of them.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.access_token
    }

    /// What the access token says about itself, read locally without verifying anything.
    ///
    /// A Minecraft access token is a JWT. Its `exp` claim is the moment Mojang stops accepting
    /// it (about a day after the launcher fetched it), and current tokens name the Minecraft
    /// profile they were issued for. Only those two claims are read, nothing returned could
    /// stand in for the token, and an opaque (non-JWT) token yields [`TokenFacts::default`].
    #[must_use]
    pub fn token_facts(&self) -> TokenFacts {
        TokenFacts::read(&self.access_token)
    }
}

/// What a JWT access token says about itself; see [`Session::token_facts`].
///
/// Used to explain a failed server join truthfully ("your login expired", "this login belongs
/// to another account") and to warn in the log before that happens. A launcher passes the token
/// it has; it may be hours from expiry or minutes, and nothing refreshes it mid-session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenFacts {
    /// Unix seconds after which Mojang stops accepting the token (the `exp` claim).
    pub expires_at: Option<u64>,
    /// The Minecraft profile the token was issued for (`profiles.mc`, or a `pfd` entry of
    /// type `mc`), when the token says.
    pub profile: Option<Uuid>,
}

impl TokenFacts {
    fn read(token: &str) -> Self {
        use base64::Engine as _;
        use serde_json::Value;

        let mut parts = token.split('.');
        let (Some(_header), Some(payload), Some(_signature), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Self::default();
        };
        let Ok(bytes) =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload.trim_end_matches('='))
        else {
            return Self::default();
        };
        let Ok(claims) = serde_json::from_slice::<Value>(&bytes) else {
            return Self::default();
        };
        let expires_at = claims.get("exp").and_then(|exp| {
            exp.as_u64().or_else(|| {
                exp.as_f64()
                    .filter(|exp| exp.is_finite() && *exp >= 0.0)
                    .map(|exp| exp as u64)
            })
        });
        let uuid = |value: &Value| value.as_str().and_then(|id| Uuid::parse_str(id).ok());
        let profile = claims.pointer("/profiles/mc").and_then(uuid).or_else(|| {
            claims
                .get("pfd")?
                .as_array()?
                .iter()
                .find(|entry| entry.get("type").and_then(Value::as_str) == Some("mc"))?
                .get("id")
                .and_then(uuid)
        });
        TokenFacts {
            expires_at,
            profile,
        }
    }

    /// Whether the token had expired by `now` (unix seconds). An unknown expiry never has.
    #[must_use]
    pub fn expired_by(&self, now: u64) -> bool {
        self.expires_at.is_some_and(|expires_at| expires_at <= now)
    }

    /// Whether the token says it was issued for a profile other than `uuid`. A token that does
    /// not name its profile never is.
    #[must_use]
    pub fn issued_for_another_profile(&self, uuid: Uuid) -> bool {
        self.profile.is_some_and(|profile| profile != uuid)
    }
}

/// The current time in unix seconds, for comparing with [`TokenFacts::expires_at`].
#[must_use]
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("uuid", &self.uuid)
            .field("username", &self.username)
            .field("access_token", &Redacted)
            .finish()
    }
}

/// A `Debug` placeholder that prints `<redacted>` instead of a secret.
pub(crate) struct Redacted;

impl fmt::Debug for Redacted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// Which of the two player-model shapes a skin uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SkinVariant {
    /// The 4-pixel-wide-arm model (Steve).
    Classic,
    /// The 3-pixel-wide-arm model (Alex).
    Slim,
}

impl<'de> Deserialize<'de> for SkinVariant {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Unknown values degrade to CLASSIC rather than failing the whole login: a new variant
        // appearing server-side must not lock the user out.
        let s = String::deserialize(d)?;
        Ok(if s.eq_ignore_ascii_case("slim") {
            Self::Slim
        } else {
            Self::Classic
        })
    }
}

/// One skin entry from the Minecraft services profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skin {
    /// Mojang's identifier for this skin.
    pub id: String,
    /// `"ACTIVE"` for the skin currently worn.
    pub state: String,
    /// Download URL on `textures.minecraft.net`.
    pub url: String,
    /// Arm-width model.
    pub variant: SkinVariant,
    /// Mojang's name for a default skin, e.g. `"STEVE"`. Absent for uploaded skins.
    #[serde(default)]
    pub alias: Option<String>,
}

/// One cape entry from the Minecraft services profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cape {
    /// Mojang's identifier for this cape.
    pub id: String,
    /// `"ACTIVE"` for the cape currently worn, `"INACTIVE"` for owned-but-hidden.
    pub state: String,
    /// Download URL on `textures.minecraft.net`.
    pub url: String,
    /// Mojang's name for the cape, e.g. `"Migrator"`.
    #[serde(default)]
    pub alias: Option<String>,
}

/// The full Minecraft services profile.
///
/// [`Session`] is the trimmed-down view the network layer needs; this is what a skin renderer
/// (S12) or the profile screen (S16) wants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    /// The player's Minecraft UUID.
    pub uuid: Uuid,
    /// The player's current username.
    pub username: String,
    /// Every skin on the account, worn or not.
    pub skins: Vec<Skin>,
    /// Every cape unlocked on the account, worn or not.
    pub capes: Vec<Cape>,
}

impl Profile {
    /// The currently-worn skin, if any.
    #[must_use]
    pub fn active_skin(&self) -> Option<&Skin> {
        self.skins
            .iter()
            .find(|s| s.state.eq_ignore_ascii_case("ACTIVE"))
    }

    /// The currently-worn cape, if any.
    #[must_use]
    pub fn active_cape(&self) -> Option<&Cape> {
        self.capes
            .iter()
            .find(|c| c.state.eq_ignore_ascii_case("ACTIVE"))
    }
}

/// Long-lived credentials, i.e. everything that has to survive a restart.
///
/// Written to disk encrypted by [`TokenCache`](crate::TokenCache). Both token fields are
/// secret and are redacted in `Debug`.
#[derive(Clone, Serialize, Deserialize)]
pub struct Credentials {
    /// The Microsoft refresh token — the thing that makes silent re-login possible. **Secret.**
    pub msa_refresh_token: String,
    /// The current Minecraft services access token. **Secret.**
    pub mc_access_token: String,
    /// Unix timestamp (seconds) at which `mc_access_token` stops being valid.
    pub expires_at: u64,
    /// The player's UUID, cached so a valid token needs no profile round-trip on startup.
    pub uuid: Uuid,
    /// The player's username at the time of the last login.
    pub username: String,
}

impl Credentials {
    /// Whether the Minecraft access token is still usable, allowing for `skew` of clock drift
    /// and request latency.
    #[must_use]
    pub fn is_fresh(&self, skew: Duration) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.expires_at > now.saturating_add(skew.as_secs())
    }

    /// The session these credentials describe, without contacting the network.
    #[must_use]
    pub fn session(&self) -> Session {
        Session::new(
            self.uuid,
            self.username.clone(),
            self.mc_access_token.clone(),
        )
    }
}

impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credentials")
            .field("msa_refresh_token", &Redacted)
            .field("mc_access_token", &Redacted)
            .field("expires_at", &self.expires_at)
            .field("uuid", &self.uuid)
            .field("username", &self.username)
            .finish()
    }
}

/// A completed login: the session, the full profile, and the credentials to cache.
#[derive(Debug, Clone)]
pub struct Authenticated {
    /// What the network layer needs.
    pub session: Session,
    /// What the UI and skin renderer want.
    pub profile: Profile,
    /// What should be written to the token cache.
    pub credentials: Credentials,
}

/// Convert a unix-epoch offset in seconds into an absolute expiry timestamp.
pub(crate) fn expires_at(expires_in_secs: u64) -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .saturating_add(expires_in_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Session {
        Session::new(
            Uuid::parse_str("986dec87-b7ec-47ff-89ff-033fdb95c4b5").expect("uuid"),
            "HowDoesAuthWork",
            "super-secret-token-value",
        )
    }

    #[test]
    fn debug_redacts_the_access_token() {
        let rendered = format!("{:?}", sample());
        assert!(!rendered.contains("super-secret-token-value"));
        assert!(rendered.contains("<redacted>"));
        assert!(rendered.contains("HowDoesAuthWork"));
    }

    /// A session's token is registered on construction, so it is scrubbed
    /// from any text that leaves the process even when it is formatted directly.
    #[test]
    fn a_new_session_registers_its_token_for_scrubbing() {
        let token = "eyJhbGciOiJIUzI1NiJ9.fake-session-registration.c2lnbmF0dXJl";
        let line = format!("joining with {token}");
        let session = with_token(token.to_owned());
        assert_eq!(
            session.expose(),
            token,
            "the session itself keeps the real token"
        );
        assert_eq!(mc_core::secrets::scrub(&line), "joining with <redacted>");
    }

    #[test]
    fn credentials_debug_redacts_both_tokens() {
        let creds = Credentials {
            msa_refresh_token: "refresh-secret".to_owned(),
            mc_access_token: "access-secret".to_owned(),
            expires_at: 0,
            uuid: Uuid::nil(),
            username: "x".to_owned(),
        };
        let rendered = format!("{creds:?}");
        assert!(!rendered.contains("refresh-secret"));
        assert!(!rendered.contains("access-secret"));
    }

    #[test]
    fn undashed_uuid_matches_the_protocol_form() {
        assert_eq!(sample().uuid_undashed(), "986dec87b7ec47ff89ff033fdb95c4b5");
    }

    /// A fake JWT with `claims` as its payload (unsigned: nothing verifies it).
    fn jwt(claims: &serde_json::Value, padded: bool) -> String {
        use base64::Engine as _;
        let engine = if padded {
            base64::engine::general_purpose::URL_SAFE
        } else {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
        };
        format!(
            "{}.{}.c2lnbmF0dXJl",
            engine.encode(br#"{"alg":"HS256"}"#),
            engine.encode(claims.to_string())
        )
    }

    fn with_token(token: String) -> Session {
        Session::new(sample().uuid, "HowDoesAuthWork", token)
    }

    #[test]
    fn token_facts_read_the_expiry_and_the_profile() {
        let me = sample().uuid;
        let claims = serde_json::json!({
            "xuid": "1234",
            "exp": 1_900_000_000u64,
            "profiles": { "mc": me.to_string() },
        });
        for padded in [false, true] {
            let facts = with_token(jwt(&claims, padded)).token_facts();
            assert_eq!(facts.expires_at, Some(1_900_000_000));
            assert_eq!(facts.profile, Some(me));
            assert!(!facts.expired_by(1_899_999_999));
            assert!(facts.expired_by(1_900_000_000));
            assert!(!facts.issued_for_another_profile(me));
            assert!(facts.issued_for_another_profile(Uuid::nil()));
        }
        // The `pfd` spelling, with an undashed id.
        let pfd = serde_json::json!({
            "exp": 5,
            "pfd": [{ "type": "xbox", "id": "x" }, { "type": "mc", "id": me.simple().to_string() }],
        });
        let facts = with_token(jwt(&pfd, false)).token_facts();
        assert_eq!(facts.profile, Some(me));
        assert_eq!(facts.expires_at, Some(5));
    }

    #[test]
    fn opaque_or_odd_tokens_say_nothing() {
        for token in [
            String::new(),
            "not-a-jwt".to_owned(),
            "a.b".to_owned(),
            "a.b.c.d".to_owned(),
            "a.!!!.c".to_owned(),
            format!("a.{}.c", "bm90IGpzb24"),
        ] {
            let facts = with_token(token.clone()).token_facts();
            assert_eq!(facts, TokenFacts::default(), "{token:?}");
            assert!(!facts.expired_by(u64::MAX));
            assert!(!facts.issued_for_another_profile(Uuid::nil()));
        }
        // Claims that are there but not usable are ignored one by one.
        let odd = serde_json::json!({ "exp": "soon", "profiles": { "mc": "not-a-uuid" } });
        assert_eq!(
            with_token(jwt(&odd, false)).token_facts(),
            TokenFacts::default()
        );
    }

    #[test]
    fn freshness_accounts_for_skew() {
        let mut creds = Credentials {
            msa_refresh_token: String::new(),
            mc_access_token: String::new(),
            expires_at: expires_at(600),
            uuid: Uuid::nil(),
            username: String::new(),
        };
        assert!(creds.is_fresh(Duration::from_secs(60)));
        assert!(!creds.is_fresh(Duration::from_secs(3600)));
        creds.expires_at = 0;
        assert!(!creds.is_fresh(Duration::ZERO));
    }

    #[test]
    fn skin_variant_deserialises_leniently() {
        let s: Skin = serde_json::from_str(
            r#"{"id":"a","state":"ACTIVE","url":"http://x","variant":"SLIM"}"#,
        )
        .expect("parse");
        assert_eq!(s.variant, SkinVariant::Slim);
        let s: Skin = serde_json::from_str(
            r#"{"id":"a","state":"ACTIVE","url":"http://x","variant":"SOMETHING_NEW"}"#,
        )
        .expect("parse");
        assert_eq!(s.variant, SkinVariant::Classic);
    }
}
