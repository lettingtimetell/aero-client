//! End-to-end tests for the Microsoft → Xbox Live → XSTS → Minecraft chain, against a mock
//! HTTP server.
//!
//! These pin the *shapes* documented on the Microsoft Authentication Scheme page: the exact
//! request bodies each endpoint receives, the polling states of the device-code grant, and the
//! mapping from XSTS `XErr` codes and Minecraft-services statuses onto typed errors. They
//! cannot prove the real services behave as documented — see the S03 handoff note.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use mc_auth::{
    AuthClient, AuthConfig, AuthError, Credentials, Endpoints, PollTuning, Session, TokenCache,
};
use serde_json::json;
use uuid::Uuid;
use wiremock::matchers::{body_partial_json, body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Predicate over a mapped error, so the table-driven tests stay readable.
type Check = fn(&AuthError) -> bool;

const CLIENT_ID: &str = "00000000-1111-2222-3333-444444444444";
const PROFILE_ID: &str = "986dec87b7ec47ff89ff033fdb95c4b5";
const PROFILE_UUID: &str = "986dec87-b7ec-47ff-89ff-033fdb95c4b5";

fn client(server: &MockServer) -> AuthClient {
    AuthClient::new(AuthConfig {
        client_id: Some(CLIENT_ID.to_owned()),
    })
    .expect("build client")
    .with_endpoints(Endpoints::rooted_at(&server.uri()))
    .with_poll_tuning(PollTuning {
        min_interval: Duration::from_millis(1),
        slow_down_increment: Duration::from_millis(1),
    })
}

fn device_code_body() -> serde_json::Value {
    json!({
        "device_code": "DEVICE-CODE-SECRET",
        "user_code": "A1B2C3D4",
        "verification_uri": "https://microsoft.com/link",
        "message": "To sign in, use a web browser to open https://microsoft.com/link",
        "expires_in": 900,
        "interval": 0,
    })
}

async fn mount_device_code(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/devicecode"))
        .and(body_string_contains(CLIENT_ID))
        .and(body_string_contains("XboxLive.signin"))
        .and(body_string_contains("offline_access"))
        .respond_with(ResponseTemplate::new(200).set_body_json(device_code_body()))
        .mount(server)
        .await;
}

async fn mount_token_success(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "token_type": "Bearer",
            "scope": "XboxLive.signin",
            "expires_in": 3599,
            "access_token": "MSA-ACCESS",
            "refresh_token": "MSA-REFRESH",
        })))
        .mount(server)
        .await;
}

/// XBL, XSTS, login_with_xbox and profile, all succeeding, with their bodies asserted.
async fn mount_happy_chain(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/user/authenticate"))
        .and(body_partial_json(json!({
            "Properties": {
                "AuthMethod": "RPS",
                "SiteName": "user.auth.xboxlive.com",
                "RpsTicket": "d=MSA-ACCESS",
            },
            "RelyingParty": "http://auth.xboxlive.com",
            "TokenType": "JWT",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "IssueInstant": "2020-12-07T19:52:08.4463796Z",
            "NotAfter": "2020-12-21T19:52:08.4463796Z",
            "Token": "XBL-TOKEN",
            "DisplayClaims": { "xui": [ { "uhs": "USERHASH" } ] },
        })))
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/xsts/authorize"))
        .and(body_partial_json(json!({
            "Properties": { "SandboxId": "RETAIL", "UserTokens": ["XBL-TOKEN"] },
            "RelyingParty": "rp://api.minecraftservices.com/",
            "TokenType": "JWT",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "Token": "XSTS-TOKEN",
            "DisplayClaims": { "xui": [ { "uhs": "USERHASH" } ] },
        })))
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/authentication/login_with_xbox"))
        .and(body_partial_json(json!({
            "identityToken": "XBL3.0 x=USERHASH;XSTS-TOKEN",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "username": "some-uuid",
            "roles": [],
            "access_token": "MC-ACCESS",
            "token_type": "Bearer",
            "expires_in": 86400,
        })))
        .mount(server)
        .await;

    Mock::given(method("GET"))
        .and(path("/minecraft/profile"))
        .and(header("authorization", "Bearer MC-ACCESS"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": PROFILE_ID,
            "name": "HowDoesAuthWork",
            "skins": [{
                "id": "6a6e65e5-76dd-4c3c-a625-162924514568",
                "state": "ACTIVE",
                "url": "http://textures.minecraft.net/texture/1a4a",
                "variant": "SLIM",
                "alias": "STEVE",
            }],
            "capes": [{
                "id": "5af20372-79e0-4e1f-80f8-6bd8e3135995",
                "state": "ACTIVE",
                "url": "http://textures.minecraft.net/texture/2340",
                "alias": "Migrator",
            }],
        })))
        .mount(server)
        .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn full_device_code_login() {
    let server = MockServer::start().await;
    mount_device_code(&server).await;
    mount_token_success(&server).await;
    mount_happy_chain(&server).await;

    let seen = Arc::new(AtomicUsize::new(0));
    let seen2 = Arc::clone(&seen);
    let auth = client(&server)
        .login_device_code(move |p| {
            assert_eq!(p.user_code, "A1B2C3D4");
            assert_eq!(p.verification_uri, "https://microsoft.com/link");
            assert_eq!(p.expires_in, Duration::from_secs(900));
            seen2.fetch_add(1, Ordering::SeqCst);
        })
        .await
        .expect("login");

    assert_eq!(
        seen.load(Ordering::SeqCst),
        1,
        "prompt callback runs exactly once"
    );
    assert_eq!(auth.session.username, "HowDoesAuthWork");
    assert_eq!(auth.session.uuid, Uuid::parse_str(PROFILE_UUID).unwrap());
    assert_eq!(auth.session.expose(), "MC-ACCESS");
    assert_eq!(auth.credentials.msa_refresh_token, "MSA-REFRESH");
    assert_eq!(auth.credentials.mc_access_token, "MC-ACCESS");
    assert!(auth.credentials.is_fresh(Duration::from_secs(3600)));
    assert_eq!(
        auth.profile.active_skin().expect("skin").variant,
        mc_auth::SkinVariant::Slim
    );
    assert_eq!(
        auth.profile.active_cape().expect("cape").alias.as_deref(),
        Some("Migrator")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn polling_survives_authorization_pending_and_slow_down() {
    let server = MockServer::start().await;
    mount_device_code(&server).await;
    mount_happy_chain(&server).await;

    // Priority 1 first, then 2, then the default-priority success. `up_to_n_times` retires each
    // mock once it has been used, so this plays a scripted sequence.
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": "authorization_pending",
            "error_description": "AADSTS70016: pending",
        })))
        .up_to_n_times(2)
        .with_priority(1)
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": "slow_down",
            "error_description": "polling too fast",
        })))
        .up_to_n_times(1)
        .with_priority(2)
        .expect(1)
        .mount(&server)
        .await;
    mount_token_success(&server).await;

    let auth = client(&server)
        .login_device_code(|_| {})
        .await
        .expect("login should succeed after backing off");
    assert_eq!(auth.session.username, "HowDoesAuthWork");
    // wiremock verifies the `.expect(..)` counts when the server drops.
}

#[tokio::test(flavor = "multi_thread")]
async fn polling_maps_terminal_states() {
    for (body, check) in [
        (
            json!({"error": "expired_token", "error_description": "expired"}),
            (|e: &AuthError| matches!(e, AuthError::DeviceCodeExpired)) as Check,
        ),
        (
            json!({"error": "authorization_declined", "error_description": "declined"}),
            |e| matches!(e, AuthError::Declined),
        ),
        (
            json!({"error": "access_denied", "error_description": "denied"}),
            |e| matches!(e, AuthError::Declined),
        ),
        (
            json!({"error": "invalid_client", "error_description": "AADSTS7000218"}),
            |e| matches!(e, AuthError::Oauth { .. }),
        ),
    ] {
        let server = MockServer::start().await;
        mount_device_code(&server).await;
        Mock::given(method("POST"))
            .and(path("/oauth2/v2.0/token"))
            .respond_with(ResponseTemplate::new(400).set_body_json(body.clone()))
            .mount(&server)
            .await;

        let err = client(&server)
            .login_device_code(|_| {})
            .await
            .expect_err("should fail");
        assert!(check(&err), "{body} mapped to {err:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn device_code_deadline_is_enforced() {
    let server = MockServer::start().await;
    // expires_in = 0 means the code is dead before the first poll can succeed.
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/devicecode"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "device_code": "D",
            "user_code": "U",
            "verification_uri": "https://microsoft.com/link",
            "expires_in": 0,
            "interval": 0,
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": "authorization_pending",
        })))
        .mount(&server)
        .await;

    let err = client(&server)
        .login_device_code(|_| {})
        .await
        .expect_err("should time out");
    assert!(matches!(err, AuthError::DeviceCodeExpired), "{err:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn xsts_errors_map_to_typed_variants() {
    let cases: [(u64, Check); 4] = [
        (2_148_916_233, |e| matches!(e, AuthError::NoXboxAccount)),
        (2_148_916_238, |e| {
            matches!(e, AuthError::ChildAccount { .. })
        }),
        (2_148_916_227, |e| matches!(e, AuthError::XboxBanned)),
        (2_148_916_235, |e| {
            matches!(e, AuthError::XboxRegionUnavailable)
        }),
    ];

    for (xerr, check) in cases {
        let server = MockServer::start().await;
        mount_device_code(&server).await;
        mount_token_success(&server).await;
        Mock::given(method("POST"))
            .and(path("/user/authenticate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "Token": "XBL-TOKEN",
                "DisplayClaims": { "xui": [ { "uhs": "USERHASH" } ] },
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/xsts/authorize"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "Identity": "0",
                "XErr": xerr,
                "Message": "",
                "Redirect": "https://start.ui.xboxlive.com/AddChildToFamily",
            })))
            .mount(&server)
            .await;

        let err = client(&server)
            .login_device_code(|_| {})
            .await
            .expect_err("should fail");
        assert!(check(&err), "XErr {xerr} mapped to {err:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn profile_404_means_no_entitlement() {
    let server = MockServer::start().await;
    mount_device_code(&server).await;
    mount_token_success(&server).await;
    mount_happy_chain(&server).await;
    // Override the profile response: mounted later wins at equal priority is not guaranteed, so
    // use an explicit higher priority.
    Mock::given(method("GET"))
        .and(path("/minecraft/profile"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "path": "/minecraft/profile",
            "errorType": "NOT_FOUND",
            "error": "NOT_FOUND",
        })))
        .with_priority(1)
        .mount(&server)
        .await;

    let err = client(&server)
        .login_device_code(|_| {})
        .await
        .expect_err("should fail");
    assert!(matches!(err, AuthError::NoEntitlement), "{err:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn login_with_xbox_403_keeps_the_body_instead_of_guessing() {
    let server = MockServer::start().await;
    mount_device_code(&server).await;
    mount_token_success(&server).await;
    mount_happy_chain(&server).await;
    Mock::given(method("POST"))
        .and(path("/authentication/login_with_xbox"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({ "error": "FORBIDDEN" })))
        .with_priority(1)
        .mount(&server)
        .await;

    let err = client(&server)
        .login_device_code(|_| {})
        .await
        .expect_err("should fail");
    // Not NoEntitlement: an unapproved Azure client ID returns this same 403, and only the body
    // separates that from an account that really does not own the game.
    let AuthError::ServicesForbidden { body } = &err else {
        panic!("expected ServicesForbidden, got {err:?}");
    };
    assert!(body.contains("FORBIDDEN"), "{body:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn entitlement_check_distinguishes_owners() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/entitlements/mcstore"))
        .and(header("authorization", "Bearer MC-ACCESS"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [
                {"name": "product_minecraft", "signature": "jwt"},
                {"name": "game_minecraft", "signature": "jwt"},
            ],
            "signature": "jwt",
            "keyId": "1",
        })))
        .mount(&server)
        .await;
    assert!(client(&server)
        .has_java_entitlement("MC-ACCESS")
        .await
        .expect("check"));

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/entitlements/mcstore"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [{"name": "product_minecraft_bedrock", "signature": "jwt"}],
            "signature": "jwt",
            "keyId": "1",
        })))
        .mount(&server)
        .await;
    assert!(!client(&server)
        .has_java_entitlement("MC-ACCESS")
        .await
        .expect("check"));
}

#[tokio::test(flavor = "multi_thread")]
async fn refresh_uses_the_refresh_token_grant() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .and(body_string_contains("refresh_token=OLD-REFRESH"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "token_type": "Bearer",
            "expires_in": 3599,
            "access_token": "MSA-ACCESS",
            "refresh_token": "NEW-REFRESH",
        })))
        .expect(1)
        .mount(&server)
        .await;
    mount_happy_chain(&server).await;

    let auth = client(&server)
        .refresh("OLD-REFRESH")
        .await
        .expect("refresh");
    assert_eq!(auth.session.expose(), "MC-ACCESS");
    assert_eq!(
        auth.credentials.msa_refresh_token, "NEW-REFRESH",
        "the rotated refresh token must replace the old one"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn refresh_keeps_the_old_token_when_microsoft_does_not_rotate() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "token_type": "Bearer",
            "expires_in": 3599,
            "access_token": "MSA-ACCESS",
        })))
        .mount(&server)
        .await;
    mount_happy_chain(&server).await;

    let auth = client(&server)
        .refresh("OLD-REFRESH")
        .await
        .expect("refresh");
    assert_eq!(auth.credentials.msa_refresh_token, "OLD-REFRESH");
}

#[tokio::test(flavor = "multi_thread")]
async fn join_server_sends_the_documented_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/session/minecraft/join"))
        .and(body_partial_json(json!({
            "accessToken": "MC-ACCESS",
            "selectedProfile": PROFILE_ID,
            "serverId": "-7c9d5b0044c130109a5d7b5fb5c317c02b4e28c1",
        })))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&server)
        .await;

    let session = Session::new(
        Uuid::parse_str(PROFILE_UUID).unwrap(),
        "HowDoesAuthWork",
        "MC-ACCESS",
    );
    client(&server)
        .join_server(&session, &mc_auth::server_hash("jeb_", &[], &[]))
        .await
        .expect("join");
}

/// The join answer is read from its body, the way authlib
/// 1.5.21 reads it, over a real loopback HTTP server. Before this, every 401/403 was "invalid
/// session" and everything else one opaque error.
#[tokio::test(flavor = "multi_thread")]
async fn join_server_reads_the_answer_like_vanilla() {
    let invalid: Check = |e| matches!(e, AuthError::TokenRejected);
    let cases: Vec<(u16, &str, Check)> = vec![
        (
            403,
            r#"{"error":"ForbiddenOperationException","errorMessage":"Invalid token"}"#,
            invalid,
        ),
        (401, "", invalid),
        (
            403,
            r#"{"error":"InsufficientPrivilegesException","errorMessage":"Multiplayer is disabled"}"#,
            |e| {
                matches!(e, AuthError::JoinRefused { status: 403, error, message }
                    if error == "InsufficientPrivilegesException"
                        && message.as_deref() == Some("Multiplayer is disabled"))
            },
        ),
        (
            403,
            r#"{"error":"UserBannedException","path":"/session/minecraft/join"}"#,
            |e| {
                matches!(e, AuthError::JoinRefused { status: 403, error, message: None }
                    if error == "UserBannedException")
            },
        ),
        (
            400,
            r#"{"error":"IllegalArgumentException","errorMessage":"Invalid UUID string"}"#,
            |e| matches!(e, AuthError::JoinRefused { status: 400, .. }),
        ),
        (
            403,
            "<html><head><title>403 ERROR</title></head><body>Request blocked.</body></html>",
            |e| {
                matches!(e, AuthError::JoinUnreadable { status: 403, detail }
                    if detail.contains("not JSON") && !detail.contains("blocked"))
            },
        ),
        (404, "nope", |e| {
            matches!(e, AuthError::JoinUnreadable { status: 404, .. })
        }),
    ];
    for (status, body, check) in cases {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/session/minecraft/join"))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .expect(1)
            .mount(&server)
            .await;

        let session = Session::new(Uuid::nil(), "x", "MC-ACCESS");
        let err = client(&server)
            .join_server(&session, "hash")
            .await
            .expect_err("should fail");
        assert!(check(&err), "HTTP {status} {body:?} mapped to {err:?}");
    }
}

/// java: a Yggdrasil response without `error` is success whatever the status, so vanilla
/// carries on with the login (the server's own `hasJoined` check then decides).
#[tokio::test(flavor = "multi_thread")]
async fn join_server_accepts_an_answer_without_an_error_like_vanilla() {
    for (status, body) in [
        (204u16, ""),
        (200, "{}"),
        (202, "null"),
        (400, r#"{"path":"/x"}"#),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/session/minecraft/join"))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .expect(1)
            .mount(&server)
            .await;
        let session = Session::new(Uuid::nil(), "x", "MC-ACCESS");
        client(&server)
            .join_server(&session, "hash")
            .await
            .unwrap_or_else(|e| panic!("HTTP {status} {body:?} must be accepted, got {e:?}"));
    }
}

/// A quick 429 or 5xx is retried exactly once; anything else, or a long `Retry-After`, is not.
#[tokio::test(flavor = "multi_thread")]
async fn join_server_retries_a_temporary_failure_once() {
    let session = Session::new(Uuid::nil(), "x", "MC-ACCESS");

    // 503 then 204: one retry, success.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/session/minecraft/join"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .expect(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/session/minecraft/join"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .with_priority(2)
        .mount(&server)
        .await;
    client(&server)
        .with_join_retry_delay(Duration::from_millis(5))
        .join_server(&session, "hash")
        .await
        .expect("the retry succeeds");
    server.verify().await;

    // 429 twice: two requests, then the second answer is the error.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/session/minecraft/join"))
        .respond_with(ResponseTemplate::new(429).set_body_string(
            r#"{"error":"TooManyRequestsException","errorMessage":"Too many requests"}"#,
        ))
        .expect(2)
        .mount(&server)
        .await;
    let err = client(&server)
        .with_join_retry_delay(Duration::from_millis(5))
        .join_server(&session, "hash")
        .await
        .expect_err("still rate limited");
    assert!(
        matches!(err, AuthError::JoinRefused { status: 429, .. }),
        "{err:?}"
    );
    server.verify().await;

    // A long Retry-After, a refusal that is not temporary, and an invalid token: one request.
    for template in [
        ResponseTemplate::new(503).insert_header("Retry-After", "30"),
        ResponseTemplate::new(403).set_body_string(
            r#"{"error":"InsufficientPrivilegesException","errorMessage":"Multiplayer is disabled"}"#,
        ),
        ResponseTemplate::new(403)
            .set_body_string(r#"{"error":"ForbiddenOperationException","errorMessage":"x"}"#),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/session/minecraft/join"))
            .respond_with(template)
            .expect(1)
            .mount(&server)
            .await;
        client(&server)
            .with_join_retry_delay(Duration::from_millis(5))
            .join_server(&session, "hash")
            .await
            .expect_err("not retried");
        server.verify().await;
    }
}

/// The access token must never reach an error message, even when a response echoes it.
#[tokio::test(flavor = "multi_thread")]
async fn join_errors_never_carry_the_access_token() {
    const TOKEN: &str = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjF9.secret-signature-value";
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/session/minecraft/join"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "error": format!("BadTokenException {TOKEN}"),
            "errorMessage": format!("token {TOKEN} is not welcome"),
        })))
        .mount(&server)
        .await;
    let session = Session::new(Uuid::nil(), "x", TOKEN);
    let err = client(&server)
        .join_server(&session, "hash")
        .await
        .expect_err("refused");
    for rendered in [err.to_string(), format!("{err:?}")] {
        assert!(!rendered.contains("secret-signature-value"), "{rendered}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn login_reuses_a_fresh_cached_token_without_contacting_microsoft() {
    let server = MockServer::start().await;
    mount_happy_chain(&server).await;
    // Any call to Microsoft would be a bug: assert zero hits.
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/devicecode"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().expect("tempdir");
    let cache = TokenCache::at(dir.path());
    cache
        .save(&Credentials {
            msa_refresh_token: "MSA-REFRESH".to_owned(),
            mc_access_token: "MC-ACCESS".to_owned(),
            expires_at: unix_now() + 86_400,
            uuid: Uuid::parse_str(PROFILE_UUID).unwrap(),
            username: "StaleName".to_owned(),
        })
        .expect("seed cache");

    let auth = client(&server)
        .login(&cache, |_| panic!("must not prompt"))
        .await
        .expect("silent login");
    assert_eq!(auth.session.expose(), "MC-ACCESS");
    // The cached username was stale; the profile call is what corrects it.
    assert_eq!(auth.session.username, "HowDoesAuthWork");
    assert_eq!(
        cache.load().expect("load").expect("some").username,
        "HowDoesAuthWork"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn login_silently_refreshes_an_expired_cached_token() {
    let server = MockServer::start().await;
    mount_happy_chain(&server).await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "token_type": "Bearer",
            "expires_in": 3599,
            "access_token": "MSA-ACCESS",
            "refresh_token": "ROTATED",
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/devicecode"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().expect("tempdir");
    let cache = TokenCache::at(dir.path());
    cache
        .save(&Credentials {
            msa_refresh_token: "OLD-REFRESH".to_owned(),
            mc_access_token: "STALE".to_owned(),
            expires_at: 1, // long expired
            uuid: Uuid::nil(),
            username: String::new(),
        })
        .expect("seed cache");

    let auth = client(&server)
        .login(&cache, |_| panic!("must not prompt"))
        .await
        .expect("silent refresh");
    assert_eq!(auth.session.expose(), "MC-ACCESS");

    let stored = cache.load().expect("load").expect("some");
    assert_eq!(stored.msa_refresh_token, "ROTATED");
    assert_eq!(stored.uuid, Uuid::parse_str(PROFILE_UUID).unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn login_falls_back_to_device_code_when_the_refresh_token_is_dead() {
    let server = MockServer::start().await;
    mount_device_code(&server).await;
    mount_happy_chain(&server).await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": "invalid_grant",
            "error_description": "AADSTS70000: refresh token expired",
        })))
        .with_priority(1)
        .mount(&server)
        .await;
    mount_token_success(&server).await;

    let dir = tempfile::tempdir().expect("tempdir");
    let cache = TokenCache::at(dir.path());
    cache
        .save(&Credentials {
            msa_refresh_token: "DEAD".to_owned(),
            mc_access_token: "STALE".to_owned(),
            expires_at: 1,
            uuid: Uuid::nil(),
            username: String::new(),
        })
        .expect("seed cache");

    let prompted = Arc::new(AtomicUsize::new(0));
    let p2 = Arc::clone(&prompted);
    let auth = client(&server)
        .login(&cache, move |_| {
            p2.fetch_add(1, Ordering::SeqCst);
        })
        .await
        .expect("interactive fallback");
    assert_eq!(prompted.load(Ordering::SeqCst), 1);
    assert_eq!(auth.credentials.msa_refresh_token, "MSA-REFRESH");
}

#[tokio::test(flavor = "multi_thread")]
async fn login_does_not_prompt_when_the_network_is_simply_down() {
    // A refresh that fails for transport reasons must surface as an error, not drop the user
    // into an interactive login they cannot complete without connectivity.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(503).set_body_string("upstream unavailable"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().expect("tempdir");
    let cache = TokenCache::at(dir.path());
    cache
        .save(&Credentials {
            msa_refresh_token: "REFRESH".to_owned(),
            mc_access_token: "STALE".to_owned(),
            expires_at: 1,
            uuid: Uuid::nil(),
            username: String::new(),
        })
        .expect("seed cache");

    let err = client(&server)
        .login(&cache, |_| panic!("must not prompt"))
        .await
        .expect_err("should surface the outage");
    assert!(
        matches!(err, AuthError::Http { status: 503, .. }),
        "{err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_corrupt_cache_does_not_lock_the_user_out() {
    let server = MockServer::start().await;
    mount_device_code(&server).await;
    mount_token_success(&server).await;
    mount_happy_chain(&server).await;

    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("session.bin"), b"garbage").expect("write");
    let cache = TokenCache::at(dir.path());

    let auth = client(&server)
        .login(&cache, |_| {})
        .await
        .expect("should fall back to an interactive login");
    assert_eq!(auth.session.username, "HowDoesAuthWork");
    // ...and the good credentials replace the garbage.
    assert_eq!(
        cache.load().expect("load").expect("some").mc_access_token,
        "MC-ACCESS"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn login_without_a_client_id_fails_before_any_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let client = AuthClient::new(AuthConfig::default())
        .expect("build")
        .with_endpoints(Endpoints::rooted_at(&server.uri()));
    let err = client
        .login_device_code(|_| panic!("must not prompt"))
        .await
        .expect_err("should refuse");
    assert!(matches!(err, AuthError::MissingClientId), "{err:?}");
    assert!(err.to_string().contains(mc_auth::CLIENT_ID_ENV));
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_responses_are_typed_errors_not_panics() {
    let server = MockServer::start().await;
    mount_device_code(&server).await;
    mount_token_success(&server).await;
    Mock::given(method("POST"))
        .and(path("/user/authenticate"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html>not json</html>"))
        .mount(&server)
        .await;

    let err = client(&server)
        .login_device_code(|_| {})
        .await
        .expect_err("should fail");
    assert!(
        matches!(err, AuthError::MalformedResponse { .. }),
        "{err:?}"
    );
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs()
}
