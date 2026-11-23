//! The account store against a local mock of the Microsoft, Xbox and Minecraft endpoints: adding
//! an account with the device-code flow, refreshing before a join, "sign in again", and Mojang's
//! 403 for an unapproved application. No real service is contacted.

use std::time::Duration;

use mc_auth::{
    AccountStore, ActiveAccount, AuthClient, AuthConfig, AuthError, Endpoints, PollTuning,
};
use serde_json::json;
use uuid::Uuid;
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CLIENT_ID: &str = "00000000-1111-2222-3333-444444444444";
const PROFILE_ID: &str = "986dec87b7ec47ff89ff033fdb95c4b5";

fn client(server: &MockServer) -> AuthClient {
    AuthClient::new(AuthConfig {
        client_id: Some(CLIENT_ID.to_owned()),
    })
    .expect("client")
    .with_endpoints(Endpoints::rooted_at(&server.uri()))
    .with_poll_tuning(PollTuning {
        min_interval: Duration::from_millis(1),
        slow_down_increment: Duration::from_millis(1),
    })
}

fn profile_uuid() -> Uuid {
    Uuid::parse_str(PROFILE_ID).expect("uuid")
}

/// Device code, token (any grant), Xbox, XSTS, Minecraft login and profile, all succeeding.
/// `access` is the Minecraft token the mock hands out; `refresh` the Microsoft refresh token.
async fn mount_chain(server: &MockServer, access: &str, refresh: &str, expires_in: u64) {
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/devicecode"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "device_code": "fake-device-code-0123456789",
            "user_code": "ABCD1234",
            "verification_uri": "https://example.invalid/link",
            "expires_in": 900,
            "interval": 0,
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "fake-msa-access-0123456789",
            "refresh_token": refresh,
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/user/authenticate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "Token": "fake-xbl-token-0123456789",
            "DisplayClaims": { "xui": [ { "uhs": "USERHASH" } ] },
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/xsts/authorize"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "Token": "fake-xsts-token-0123456789",
            "DisplayClaims": { "xui": [ { "uhs": "USERHASH" } ] },
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/authentication/login_with_xbox"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": access,
            "expires_in": expires_in,
        })))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/minecraft/profile"))
        .and(header("authorization", format!("Bearer {access}").as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": PROFILE_ID,
            "name": "MockPlayer",
            "skins": [{
                "id": "skin", "state": "ACTIVE", "variant": "CLASSIC",
                "url": "http://textures.minecraft.net/texture/mock",
            }],
        })))
        .mount(server)
        .await;
}

/// Add the mock account with the device-code flow, its Minecraft token valid for `expires_in`.
async fn add_account(store: &AccountStore, expires_in: u64) {
    let server = MockServer::start().await;
    mount_chain(
        &server,
        "fake-mc-access-first-0123456789",
        "fake-refresh-first-0123456789",
        expires_in,
    )
    .await;
    let auth = client(&server)
        .login_device_code(|prompt| assert_eq!(prompt.user_code, "ABCD1234"))
        .await
        .expect("sign-in");
    let added = store.add(&auth).expect("add");
    assert_eq!(added.username, "MockPlayer");
    assert_eq!(
        added.skin_url.as_deref(),
        Some("http://textures.minecraft.net/texture/mock")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_account_added_with_the_device_code_flow_is_stored_sealed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    add_account(&store, 86_400).await;
    let list = store.load().expect("load");
    assert_eq!(list.accounts.len(), 1);
    let raw = std::fs::read_to_string(store.file_path()).expect("read");
    assert!(!raw.contains("fake-mc-access-first"));
    assert!(!raw.contains("fake-refresh-first"));
    let secrets = store.unseal(&list.accounts[0]).expect("unseal");
    assert_eq!(secrets.refresh_token, "fake-refresh-first-0123456789");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fresh_token_joins_without_any_request() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    add_account(&store, 86_400).await;

    // A server with no routes: any request would get a 404 and fail the join.
    let silent = MockServer::start().await;
    let join = store
        .session_for_join(&client(&silent), profile_uuid())
        .await
        .expect("join session");
    assert!(!join.refreshed);
    assert_eq!(join.session.expose(), "fake-mc-access-first-0123456789");
    assert!(silent
        .received_requests()
        .await
        .expect("recorded")
        .is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expiring_token_is_refreshed_and_the_rotated_token_is_saved_first() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    // Expires within the five-minute skew: a join must refresh.
    add_account(&store, 60).await;
    let account = store.load().expect("load").accounts[0].clone();
    assert!(account.needs_refresh(mc_auth::unix_now()));
    store
        .set_active(Some(ActiveAccount::Stored { uuid: account.uuid }))
        .expect("active");

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .and(body_string_contains(
            "refresh_token=fake-refresh-first-0123456789",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "fake-msa-access-0123456789",
            "refresh_token": "fake-refresh-rotated-0123456789",
        })))
        .expect(1)
        .mount(&server)
        .await;
    mount_chain(
        &server,
        "fake-mc-access-second-0123456789",
        "unused",
        86_400,
    )
    .await;

    let join = store
        .session_for_join(&client(&server), account.uuid)
        .await
        .expect("refreshed");
    assert!(join.refreshed);
    assert_eq!(join.session.expose(), "fake-mc-access-second-0123456789");
    assert!(!join.account.needs_refresh(mc_auth::unix_now()));

    // On disk before the session was handed out: the rotated refresh token and the new token.
    let list = store.load().expect("load");
    assert_eq!(list.accounts.len(), 1, "updated in place");
    assert_eq!(
        list.active,
        Some(ActiveAccount::Stored { uuid: account.uuid }),
        "the choice survives a refresh"
    );
    let secrets = store.unseal(&list.accounts[0]).expect("unseal");
    assert_eq!(secrets.refresh_token, "fake-refresh-rotated-0123456789");
    assert_eq!(secrets.access_token, "fake-mc-access-second-0123456789");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dead_refresh_token_asks_for_a_new_sign_in_and_changes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    add_account(&store, 60).await;
    let before = std::fs::read(store.file_path()).expect("read");

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth2/v2.0/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": "invalid_grant",
            "error_description": "AADSTS70008: The refresh token has expired.",
        })))
        .mount(&server)
        .await;
    let error = store
        .session_for_join(&client(&server), profile_uuid())
        .await
        .expect_err("dead refresh token");
    assert!(error.needs_new_sign_in(), "{error}");
    assert_eq!(std::fs::read(store.file_path()).expect("read"), before);
    assert!(!error.to_string().contains("fake-refresh-first"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_network_failure_is_not_a_sign_in_problem() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    add_account(&store, 60).await;
    // A port nothing listens on.
    let unreachable = AuthClient::new(AuthConfig {
        client_id: Some(CLIENT_ID.to_owned()),
    })
    .expect("client")
    .with_endpoints(Endpoints::rooted_at("http://127.0.0.1:9"));
    let error = store
        .session_for_join(&unreachable, profile_uuid())
        .await
        .expect_err("offline");
    assert!(matches!(error, AuthError::Network { .. }), "{error:?}");
    assert!(!error.needs_new_sign_in());
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_client_id_an_expired_account_cannot_refresh() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    add_account(&store, 60).await;
    let server = MockServer::start().await;
    let no_id = AuthClient::new(AuthConfig::default())
        .expect("client")
        .with_endpoints(Endpoints::rooted_at(&server.uri()));
    let error = store
        .session_for_join(&no_id, profile_uuid())
        .await
        .expect_err("no client id");
    assert!(matches!(error, AuthError::MissingClientId), "{error:?}");
    assert!(server
        .received_requests()
        .await
        .expect("recorded")
        .is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn mojangs_403_for_an_unapproved_app_is_recognised() {
    let server = MockServer::start().await;
    mount_chain(&server, "unused-access-0123456789", "unused", 86_400).await;
    // Higher priority than the chain's success answer.
    Mock::given(method("POST"))
        .and(path("/authentication/login_with_xbox"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "path": "/authentication/login_with_xbox",
            "errorMessage": "Invalid app registration, see https://aka.ms/AppRegInfo for more information",
        })))
        .with_priority(1)
        .mount(&server)
        .await;
    let error = client(&server)
        .login_device_code(|_| {})
        .await
        .expect_err("unapproved app");
    assert!(error.is_unapproved_app(), "{error}");
    assert!(!error.needs_new_sign_in());
}
