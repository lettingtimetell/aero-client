//! Tokens that enter through the account store are registered with
//! `mc_core::secrets`, so the log writer, crash reports and panic output never show them: a token
//! stored by a sign-in is scrubbed at once, and a token only unsealed from disk (a later start of
//! the client, here a child process that never saw it) is scrubbed as soon as it is opened.
//!
//! Each half has its control: the refresh token is shown unscrubbed before the store touched it
//! (in this process, and in the child before it unseals).

use std::path::PathBuf;
use std::process::Command;

use mc_auth::accounts::AccountStore;
use mc_auth::{Authenticated, Credentials, Profile, Session};
use uuid::Uuid;

/// Set (to the store's folder) only in the child process.
const CHILD_ENV: &str = "MC_W21_SA_STORE_CHILD";

/// Obviously fake tokens. The refresh token never goes through `Session::new`, so only the store
/// can have registered it.
const REFRESH: &str = "fake-store-refresh-token-0123456789-abcdefghijklmnop";
const ACCESS: &str = "eyJhbGciOiJIUzI1NiJ9.ZmFrZS1zdG9yZS1hY2Nlc3MtdG9rZW4.c2lnbmF0dXJl";

fn signed_in() -> Authenticated {
    let uuid = Uuid::parse_str("986dec87-b7ec-47ff-89ff-033fdb95c4b5").expect("uuid");
    Authenticated {
        session: Session::new(uuid, "StoreTest", ACCESS),
        profile: Profile {
            uuid,
            username: "StoreTest".to_owned(),
            skins: Vec::new(),
            capes: Vec::new(),
        },
        credentials: Credentials {
            msa_refresh_token: REFRESH.to_owned(),
            mc_access_token: ACCESS.to_owned(),
            expires_at: 4_000_000_000,
            uuid,
            username: "StoreTest".to_owned(),
        },
    }
}

/// `"redacted"` when `REFRESH` is scrubbed in this process, `"plain"` when it is not.
fn refresh_state() -> &'static str {
    if mc_core::secrets::scrub(REFRESH) == REFRESH {
        "plain"
    } else {
        "redacted"
    }
}

#[test]
fn tokens_stored_or_unsealed_by_the_account_store_are_scrubbed() {
    if std::env::var_os(CHILD_ENV).is_some() {
        // Inside the child this test has nothing to do; `child_opens_the_store` runs instead.
        return;
    }
    assert_eq!(
        refresh_state(),
        "plain",
        "control: nothing registered it yet"
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    store.add(&signed_in()).expect("add the account");
    assert_eq!(
        mc_core::secrets::scrub(&format!("refresh {REFRESH} access {ACCESS}")),
        "refresh <redacted> access <redacted>"
    );

    // A later start of the client: a process that has only the store on disk.
    let output = Command::new(std::env::current_exe().expect("this test binary"))
        .args([
            "child_opens_the_store",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, dir.path())
        .output()
        .expect("run the child process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the child failed\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("store-child: before=plain after=redacted"),
        "{stdout}"
    );
}

/// The child process: opens the store a sign-in left on disk.
#[test]
fn child_opens_the_store() {
    let Some(dir) = std::env::var_os(CHILD_ENV) else {
        return;
    };
    let before = refresh_state();
    let store = AccountStore::at(PathBuf::from(dir));
    let list = store.load().expect("load the list");
    let account = list.accounts.first().expect("the stored account");
    let secrets = store.unseal(account).expect("unseal");
    assert_eq!(secrets.refresh_token, REFRESH, "the store kept the token");
    println!("store-child: before={before} after={}", refresh_state());
}
