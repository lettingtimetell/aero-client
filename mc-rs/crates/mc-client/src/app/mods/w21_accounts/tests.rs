//! The client half of the accounts: which account starts in use, switching, removing, the
//! refresh before a join and its failures, the mod menu's button, and (run explicitly) the
//! pictures of the title corner and the Accounts screen. No real Microsoft, Xbox or Minecraft
//! endpoint is contacted: stores live in temporary folders, and every sign-in or refresh task is
//! pointed at a local mock (or a local port nothing listens on) with a test client ID.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mc_auth::{
    AccountStore, ActiveAccount, Authenticated, Credentials, JoinSession, Profile, Session,
};
use mc_mods::ModEvent;
use mc_ui::screens::accounts::{AccountKey, AccountsAction};
use uuid::Uuid;

use super::super::super::{Client, Ui};
use super::{startup_with_store, AccMsg, AuthOverride};
use crate::app::config::{AccessToken, ClientConfig, LoginMode};

/// An obviously fake application ID for the mock (never a real one).
const TEST_CLIENT_ID: &str = "00000000-1111-2222-3333-444444444444";
const LAUNCHER_UUID: u128 = 0x1111_2222_3333_4444_5555_6666_7777_8888;
const LAUNCHER_TOKEN: &str = "fake-launcher-access-token-0123456789";

fn uuid(n: u8) -> Uuid {
    Uuid::from_bytes([n; 16])
}

fn launcher_login() -> LoginMode {
    LoginMode::Launcher {
        username: "FromLauncher".to_owned(),
        uuid: Uuid::from_u128(LAUNCHER_UUID),
        access_token: AccessToken::new(LAUNCHER_TOKEN),
    }
}

fn signed_in(n: u8, name: &str, expires_at: u64) -> Authenticated {
    let access = format!("fake-mc-access-{name}-0123456789");
    Authenticated {
        session: Session::new(uuid(n), name, access.clone()),
        profile: Profile {
            uuid: uuid(n),
            username: name.to_owned(),
            skins: Vec::new(),
            capes: Vec::new(),
        },
        credentials: Credentials {
            msa_refresh_token: format!("fake-refresh-{name}-0123456789"),
            mc_access_token: access,
            expires_at,
            uuid: uuid(n),
            username: name.to_owned(),
        },
    }
}

/// A day from now.
fn later() -> u64 {
    mc_auth::unix_now() + 86_400
}

fn config(login: LoginMode) -> ClientConfig {
    ClientConfig {
        login,
        ..ClientConfig::default()
    }
}

#[test]
fn with_nothing_chosen_the_launchers_session_is_used() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    store.add(&signed_in(1, "Alpha", later())).expect("add");
    let config = config(launcher_login());
    let launcher = config.login.launcher_session();
    let (state, auth) = startup_with_store(&config, launcher, Some(store));
    assert_eq!(state.active, Some(ActiveAccount::Launcher));
    assert_eq!(auth.expect("session").expose(), LAUNCHER_TOKEN);
    assert_eq!(state.badge.name, "FromLauncher");
    assert_eq!(state.badge.caption, "From the launcher");
    assert_eq!(state.rows().len(), 2, "the launcher's session, then Alpha");
}

#[test]
fn a_saved_choice_wins_and_only_it_is_unsealed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    store.add(&signed_in(1, "Alpha", later())).expect("add");
    store.add(&signed_in(2, "Beta", later())).expect("add");
    store
        .set_active(Some(ActiveAccount::Stored { uuid: uuid(2) }))
        .expect("choose");
    // Damage Alpha's sealed value: starting must not need it.
    damage_sealed(&store, 0);
    let config = config(launcher_login());
    let launcher = config.login.launcher_session();
    let (state, auth) = startup_with_store(&config, launcher, Some(store));
    assert_eq!(state.active, Some(ActiveAccount::Stored { uuid: uuid(2) }));
    let auth = auth.expect("Beta's session");
    assert_eq!(auth.username, "Beta");
    assert_eq!(auth.expose(), "fake-mc-access-Beta-0123456789");
    assert!(state.needs_sign_in.is_empty(), "Alpha was never unsealed");
    assert_eq!(state.badge.name, "Beta");
    assert_eq!(state.badge.caption, "Microsoft account");
}

#[test]
fn a_chosen_account_that_will_not_open_asks_for_a_new_sign_in() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    store.add(&signed_in(1, "Alpha", later())).expect("add");
    store
        .set_active(Some(ActiveAccount::Stored { uuid: uuid(1) }))
        .expect("choose");
    damage_sealed(&store, 0);
    let config = config(launcher_login());
    let launcher = config.login.launcher_session();
    let (state, auth) = startup_with_store(&config, launcher, Some(store));
    assert!(
        auth.is_none(),
        "never the launcher's session in its place, silently"
    );
    assert!(state.needs_sign_in.contains(&uuid(1)));
    assert_eq!(state.badge.caption, "Sign in again");
    assert!(state.badge.attention);
    assert_eq!(
        state.rows()[1].status,
        mc_ui::screens::accounts::RowStatus::NeedsSignIn
    );
}

#[test]
fn without_a_launcher_the_most_recently_used_account_starts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    store.add(&signed_in(1, "Alpha", later())).expect("add");
    store.add(&signed_in(2, "Beta", later())).expect("add");
    store
        .set_active(Some(ActiveAccount::Stored { uuid: uuid(1) }))
        .expect("use Alpha once");
    store
        .set_active(Some(ActiveAccount::Launcher))
        .expect("then the launcher");
    // Started without a launcher session: the launcher choice cannot be honoured.
    let (state, auth) = startup_with_store(&config(LoginMode::Microsoft), None, Some(store));
    assert_eq!(state.active, Some(ActiveAccount::Stored { uuid: uuid(1) }));
    assert_eq!(auth.expect("Alpha").username, "Alpha");
    // And with no accounts at all: nobody, and the corner says so.
    let empty = tempfile::tempdir().expect("tempdir");
    let (state, auth) = startup_with_store(
        &config(LoginMode::Microsoft),
        None,
        Some(AccountStore::at(empty.path())),
    );
    assert!(auth.is_none());
    assert_eq!(state.badge.name, "Not signed in");
    assert!(state.badge.head.is_none());
}

#[test]
fn offline_runs_have_no_accounts() {
    let config = config(LoginMode::Offline("Dev".to_owned()));
    assert!(!super::uses_accounts(&config));
    let (state, auth) = startup_with_store(&config, None, None);
    assert!(auth.is_none());
    assert_eq!(state.badge.name, "Dev");
    assert_eq!(state.badge.caption, "Offline mode");
}

/// Replace the sealed value of the `index`-th account with something that will not open.
fn damage_sealed(store: &AccountStore, index: usize) {
    let path = store.file_path();
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
    json["accounts"][index]["sealed"] = serde_json::Value::String("AAAA".into());
    std::fs::write(&path, serde_json::to_vec(&json).expect("json")).expect("write");
}

// ---- with a real client (needs the vanilla jar) ----

/// A client started by "the launcher", with an account store in `store_dir` holding `accounts`.
/// Sign-in and refresh tasks point at a local port nothing listens on.
fn client(
    login: LoginMode,
    store_dir: &Path,
    accounts: &[Authenticated],
) -> Option<(Client, tempfile::TempDir)> {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = ClientConfig {
        game_dir: dir.path().to_path_buf(),
        login,
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
        eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
        return None;
    }
    let store = AccountStore::at(store_dir);
    for account in accounts {
        store.add(account).expect("add");
    }
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let launcher = client.config.login.launcher_session();
    let (mut acc, auth) = startup_with_store(&client.config, launcher, Some(store));
    acc.overrides = Some(AuthOverride {
        endpoints: mc_auth::Endpoints::rooted_at(&format!("http://127.0.0.1:{}", free_port())),
        client_id: Some(TEST_CLIENT_ID.to_owned()),
    });
    client.acc = acc;
    client.auth = auth;
    Some((client, dir))
}

/// A local port nothing listens on.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.local_addr().expect("addr").port()
}

fn screen_status(client: &Client) -> Option<String> {
    match &client.ui {
        Ui::Accounts(screen) => screen.status().map(|(text, _)| text.to_owned()),
        _ => None,
    }
}

fn disconnect_text(client: &Client) -> Option<String> {
    match &client.ui {
        Ui::Disconnected(screen) => Some(screen.message.unformatted_text(&client.assets.language)),
        _ => None,
    }
}

#[test]
fn switching_changes_the_session_everywhere_and_is_remembered() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(
        launcher_login(),
        store_dir.path(),
        &[signed_in(1, "Alpha", later())],
    ) else {
        return;
    };
    assert_eq!(client.auth.as_ref().expect("auth").expose(), LAUNCHER_TOKEN);
    client.handle_action(mc_ui::screens::Action::Accounts(AccountsAction::Open));
    assert!(matches!(client.ui, Ui::Accounts(_)));

    client.acc_action(AccountsAction::Use(AccountKey::Stored(uuid(1).as_u128())));
    let auth = client.auth.as_ref().expect("auth");
    assert_eq!(auth.username, "Alpha");
    assert_eq!(auth.uuid, uuid(1));
    assert_eq!(auth.expose(), "fake-mc-access-Alpha-0123456789");
    assert_eq!(client.acc.badge.name, "Alpha");
    assert_eq!(
        screen_status(&client).as_deref(),
        Some("You're playing as Alpha now.")
    );
    // Remembered for the next start.
    let store = AccountStore::at(store_dir.path());
    assert_eq!(
        store.load().expect("load").active,
        Some(ActiveAccount::Stored { uuid: uuid(1) })
    );
    // And back to the launcher's session.
    client.acc_action(AccountsAction::Use(AccountKey::Launcher));
    assert_eq!(client.auth.as_ref().expect("auth").expose(), LAUNCHER_TOKEN);
    assert_eq!(
        store.load().expect("load").active,
        Some(ActiveAccount::Launcher)
    );
    // Done: back to the title screen.
    client.acc_action(AccountsAction::Done);
    assert!(matches!(client.ui, Ui::MainMenu(_)));
}

#[test]
fn removing_the_account_in_use_deletes_it_and_falls_back_to_the_launcher() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(
        launcher_login(),
        store_dir.path(),
        &[signed_in(1, "Alpha", later())],
    ) else {
        return;
    };
    client.acc_open();
    client.acc_action(AccountsAction::Use(AccountKey::Stored(uuid(1).as_u128())));
    client.acc_action(AccountsAction::Remove(AccountKey::Stored(
        uuid(1).as_u128(),
    )));
    assert_eq!(client.auth.as_ref().expect("auth").expose(), LAUNCHER_TOKEN);
    assert_eq!(client.acc.active, Some(ActiveAccount::Launcher));
    assert_eq!(client.acc.badge.name, "FromLauncher");
    let raw = std::fs::read_to_string(store_dir.path().join("accounts.json")).expect("read");
    assert!(
        !raw.contains("Alpha"),
        "its entry and sealed sign-in are gone"
    );
    assert!(screen_status(&client)
        .expect("status")
        .contains("Its saved sign-in was deleted"));
    // The launcher's session itself cannot be removed.
    client.acc_action(AccountsAction::Remove(AccountKey::Launcher));
    assert_eq!(client.auth.as_ref().expect("auth").expose(), LAUNCHER_TOKEN);
}

#[test]
fn a_fresh_token_joins_at_once_and_an_expiring_one_waits_for_its_refresh() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(
        LoginMode::Microsoft,
        store_dir.path(),
        &[signed_in(1, "Alpha", mc_auth::unix_now() + 60)],
    ) else {
        return;
    };
    assert_eq!(
        client.acc.active,
        Some(ActiveAccount::Stored { uuid: uuid(1) })
    );
    let port = free_port();
    client.connect("127.0.0.1", port);
    // Expires within the five-minute skew: the join waits, nothing connects yet.
    assert!(
        client.session.is_none(),
        "no join with a token about to expire"
    );
    assert!(matches!(client.ui, Ui::Connecting(_)));
    let generation = client.acc.join_pending.expect("a refresh is pending");

    // The refresh comes back (as `session_for_join` would: the store updated first).
    let refreshed = signed_in(1, "Alpha", later());
    AccountStore::at(store_dir.path())
        .add(&refreshed)
        .expect("store the refreshed sign-in");
    let account = AccountStore::at(store_dir.path())
        .load()
        .expect("load")
        .accounts[0]
        .clone();
    client.acc_message(AccMsg::Joined {
        generation,
        uuid: uuid(1),
        host: "127.0.0.1".to_owned(),
        port,
        result: Box::new(Ok(JoinSession {
            session: Session::new(uuid(1), "Alpha", "fake-refreshed-access-0123456789"),
            account,
            refreshed: true,
        })),
    });
    assert_eq!(
        client.auth.as_ref().expect("auth").expose(),
        "fake-refreshed-access-0123456789"
    );
    assert!(client.acc.join_pending.is_none());
    assert!(
        client.session.is_some(),
        "the held join went ahead with the new token"
    );
    client.disconnect();

    // Now fresh: the next join goes straight ahead.
    client.connect("127.0.0.1", free_port());
    assert!(client.session.is_some());
    assert!(client.acc.join_pending.is_none());
    client.disconnect();
}

#[test]
fn a_failed_refresh_never_joins_and_says_to_sign_in_again() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(
        LoginMode::Microsoft,
        store_dir.path(),
        &[signed_in(1, "Alpha", 1)],
    ) else {
        return;
    };
    let port = free_port();
    client.connect("127.0.0.1", port);
    let generation = client.acc.join_pending.expect("a refresh is pending");
    client.acc_message(AccMsg::Joined {
        generation,
        uuid: uuid(1),
        host: "127.0.0.1".to_owned(),
        port,
        result: Box::new(Err(mc_auth::AuthError::Oauth {
            error: "invalid_grant".to_owned(),
            description: "AADSTS70008: expired".to_owned(),
        })),
    });
    assert!(client.session.is_none(), "never joined with the dead token");
    let text = disconnect_text(&client).expect("the disconnect screen");
    assert!(text.contains("Sign in again"), "{text}");
    assert!(client.acc.needs_sign_in.contains(&uuid(1)));
    assert_eq!(client.acc.badge.caption, "Sign in again");
    // Back to the title screen; the next attempt is refused at once, with the same advice and
    // no refresh started.
    client.handle_action(mc_ui::screens::Action::Back);
    assert!(
        disconnect_text(&client).is_none(),
        "left the disconnect screen"
    );
    client.connect("127.0.0.1", free_port());
    assert!(client.session.is_none());
    assert!(client.acc.join_pending.is_none());
    assert!(disconnect_text(&client)
        .expect("again")
        .contains("Sign in again"));
}

#[test]
fn a_cancelled_join_is_not_resumed_by_its_late_refresh() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(
        LoginMode::Microsoft,
        store_dir.path(),
        &[signed_in(1, "Alpha", 1)],
    ) else {
        return;
    };
    let port = free_port();
    client.connect("127.0.0.1", port);
    let generation = client.acc.join_pending.expect("pending");
    client.handle_action(mc_ui::screens::Action::CancelConnect);
    assert!(!matches!(client.ui, Ui::Connecting(_)));
    let refreshed = signed_in(1, "Alpha", later());
    AccountStore::at(store_dir.path())
        .add(&refreshed)
        .expect("store");
    let account = AccountStore::at(store_dir.path())
        .load()
        .expect("load")
        .accounts[0]
        .clone();
    client.acc_message(AccMsg::Joined {
        generation,
        uuid: uuid(1),
        host: "127.0.0.1".to_owned(),
        port,
        result: Box::new(Ok(JoinSession {
            session: Session::new(uuid(1), "Alpha", "fake-late-access-0123456789"),
            account,
            refreshed: true,
        })),
    });
    assert!(
        client.session.is_none(),
        "the player left; nothing reconnects behind them"
    );
    assert_eq!(
        client.auth.as_ref().expect("kept").expose(),
        "fake-late-access-0123456789",
        "the fresh token is kept for the next join"
    );
}

#[test]
fn with_nobody_signed_in_a_join_explains_where_to_add_an_account() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(LoginMode::Microsoft, store_dir.path(), &[]) else {
        return;
    };
    assert!(client.auth.is_none());
    client.connect("127.0.0.1", free_port());
    assert!(client.session.is_none());
    let text = disconnect_text(&client).expect("the disconnect screen");
    assert!(text.contains("not signed in"), "{text}");
}

#[test]
fn the_mod_menus_button_opens_the_accounts_screen() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(launcher_login(), store_dir.path(), &[]) else {
        return;
    };
    client.mods.events.push(ModEvent::Button {
        mod_id: "aero_title",
        option_id: "accounts",
    });
    client.acc_frame();
    assert!(matches!(client.ui, Ui::Accounts(_)));
    assert!(client.mods.events.is_empty(), "the event is taken");
    // Escape leaves it again.
    client.acc_back();
    assert!(matches!(client.ui, Ui::MainMenu(_)));
}

#[test]
fn a_finished_sign_in_is_stored_sealed_and_used() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(launcher_login(), store_dir.path(), &[]) else {
        return;
    };
    client.acc_open();
    // Pretend a sign-in was started (its task points at a dead local port) and finished.
    client.acc_action(AccountsAction::Add);
    let generation = client.acc.add_task.as_ref().expect("a sign-in task").0;
    if let Ui::Accounts(screen) = &client.ui {
        assert!(screen.is_signing_in());
    }
    client.acc_message(AccMsg::Added {
        generation,
        result: Box::new(Ok(signed_in(7, "Gamma", later()))),
    });
    assert!(client.acc.add_task.is_none());
    let auth = client.auth.as_ref().expect("auth");
    assert_eq!(
        auth.username, "Gamma",
        "no world loaded: the new account is used"
    );
    let raw = std::fs::read_to_string(store_dir.path().join("accounts.json")).expect("read");
    assert!(raw.contains("Gamma"));
    assert!(
        !raw.contains("fake-refresh-Gamma"),
        "the refresh token is sealed"
    );
    assert!(
        !raw.contains("fake-mc-access-Gamma"),
        "the access token is sealed"
    );
    match &client.ui {
        Ui::Accounts(screen) => {
            assert!(!screen.is_signing_in());
            assert_eq!(
                screen.selected_key(),
                Some(AccountKey::Stored(uuid(7).as_u128()))
            );
        }
        _ => panic!("still on the Accounts screen"),
    }
    // A sign-in that fails shows why, on the sign-in page.
    client.acc_action(AccountsAction::Add);
    let generation = client.acc.add_task.as_ref().expect("task").0;
    client.acc_message(AccMsg::Added {
        generation,
        result: Box::new(Err(mc_auth::AuthError::ServicesForbidden {
            body: "Invalid app registration, see https://aka.ms/AppRegInfo".to_owned(),
        })),
    });
    match &client.ui {
        Ui::Accounts(screen) => {
            let page = screen.sign_in().expect("the sign-in page shows the error");
            let lines = page.body_lines(&client.assets.language);
            assert!(lines[0].contains("hasn't approved"), "{lines:?}");
        }
        _ => panic!("still on the Accounts screen"),
    }
    // Cancel goes back to the list.
    client.acc_back();
    match &client.ui {
        Ui::Accounts(screen) => assert!(!screen.is_signing_in()),
        _ => panic!("still on the Accounts screen"),
    }
}

#[test]
fn the_late_answer_of_a_cancelled_sign_in_is_ignored() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(launcher_login(), store_dir.path(), &[]) else {
        return;
    };
    client.acc_open();
    client.acc_action(AccountsAction::Add);
    let generation = client.acc.add_task.as_ref().expect("task").0;
    client.acc_action(AccountsAction::CancelAdd);
    client.acc_message(AccMsg::Added {
        generation,
        result: Box::new(Ok(signed_in(7, "Gamma", later()))),
    });
    assert_eq!(client.auth.as_ref().expect("auth").expose(), LAUNCHER_TOKEN);
    assert!(AccountStore::at(store_dir.path())
        .load()
        .expect("load")
        .accounts
        .is_empty());
}

#[test]
fn the_title_corner_shows_the_account_and_opens_the_screen() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(launcher_login(), store_dir.path(), &[]) else {
        return;
    };
    assert!(client.acc.badge.head.is_some());
    // A press and a release on the corner (its rectangle from a drawn frame).
    let screen = (1920.0, 1080.0);
    client.acc.corner = {
        // Draw once so the corner is "shown" at its place.
        let mut corner = mc_ui::screens::accounts::corner::AccountCorner::new();
        let mut text = mc_ui::screens::modmenu::text::MenuText::minecraft_only();
        let mut title = mc_ui::screens::aero_title::font::TitleFont::embedded();
        let labels = mc_ui::screens::aero_title::AeroButton::ALL.map(|b| b.english().to_owned());
        let now = std::time::Instant::now();
        let mut env = mc_ui::screens::aero_title::AeroTitleEnv {
            text: &mut text,
            title_font: &mut title,
            mc_font: &mut client.font,
            labels: &labels,
            version: "",
            game_version: mc_version::GameVersion::V1_8_9,
            copyright: "",
            screen,
            now,
            time_s: 0.0,
        };
        let mut list = mc_ui::widget::draw::DrawList::new();
        let mut glass = mc_ui::screens::modmenu::glass::GlassFrame::default();
        corner.build(
            &client.acc.badge,
            &mut env,
            now - Duration::from_secs(2),
            (-1.0, -1.0),
            &mut list,
            None,
            &mut glass,
        );
        assert_eq!(glass.rects.len(), 1);
        corner
    };
    let (x, y) = client.acc.corner.rect().center();
    assert!(
        x > screen.0 * 0.5 && y < screen.1 * 0.25,
        "top right: {x}, {y}"
    );
    assert!(client.acc_title_mouse(x, y, 0, true));
    assert!(client.acc_title_mouse(x, y, 0, false));
    assert!(matches!(client.ui, Ui::Accounts(_)));
    // Elsewhere: not taken.
    assert!(!client.acc_title_mouse(5.0, 5.0, 0, true));

    // With the Aero Title Screen mod off the title screen is
    // vanilla's: the corner's old place (it keeps its last rectangle) takes no click.
    client.show_main_menu();
    assert!(
        client.mods.settings.set_enabled("aero_title", false),
        "it was on"
    );
    assert!(
        !client.acc_title_mouse(x, y, 0, true),
        "no corner on vanilla's title screen"
    );
    assert!(!client.acc_title_mouse(x, y, 0, false));
    assert!(
        matches!(client.ui, Ui::MainMenu(_)),
        "the Accounts screen did not open"
    );
    // Back on: the corner works again.
    assert!(client.mods.settings.set_enabled("aero_title", true));
    assert!(client.acc_title_mouse(x, y, 0, true));
}

/// Pictures of the title corner and the Accounts screen through the real client (`Client::new`
/// on the real assets, `install`, `frame`, a headless GPU), at 1080p and 1440p:
///
/// `MC_ACC_PICTURES_DIR=<dir> cargo test -p mc-client --lib account_pictures -- --ignored`
#[test]
#[ignore = "writes the account PNGs; run explicitly with MC_ACC_PICTURES_DIR"]
fn account_pictures() {
    use mc_render::{Gpu, GpuOptions, InputState, PhaseList};

    let Some(out) = std::env::var_os("MC_ACC_PICTURES_DIR").map(std::path::PathBuf::from) else {
        eprintln!("skip: set MC_ACC_PICTURES_DIR to write the pictures");
        return;
    };
    std::fs::create_dir_all(&out).expect("pictures dir");
    let launcher = LoginMode::Launcher {
        username: "SkyRunner".to_owned(),
        uuid: Uuid::from_u128(LAUNCHER_UUID),
        access_token: AccessToken::new(LAUNCHER_TOKEN),
    };
    let accounts = [
        signed_in(1, "AeroPlayer", later()),
        signed_in(2, "Builder42", later()),
    ];
    for (size, tag) in [((1920u32, 1080u32), "1080p"), ((2560, 1440), "1440p")] {
        let mut gpu = match Gpu::headless(
            size.0,
            size.1,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: no headless GPU ({error})");
                return;
            }
        };
        let store_dir = tempfile::tempdir().expect("tempdir");
        let Some((mut client, _dir)) = client(launcher.clone(), store_dir.path(), &accounts) else {
            return;
        };
        // AeroPlayer in use; Builder42's saved sign-in has expired.
        client.acc_action(AccountsAction::Use(AccountKey::Stored(uuid(1).as_u128())));
        client.acc.needs_sign_in.insert(uuid(2));
        client.show_main_menu();
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, size.0, size.1);
        let mut camera = crate::app::driver::menu_camera();
        camera.set_viewport(size.0, size.1);
        // The title screen at a settled moment: 12 s into the scene (the greenhouse's camera
        // drift), the pop-in long over; the hover eases on real time, hence the pauses.
        let settle = |client: &mut Client| {
            client
                .aero
                .set_live_clocks(Duration::from_secs(12), Duration::from_secs(11));
        };
        let mut shoot = |client: &mut Client, frames: u32, pause_ms: u64, name: &str| {
            for _ in 0..frames {
                client.frame(&gpu, &mut camera, 0.0);
                settle(client);
                std::thread::sleep(Duration::from_millis(pause_ms));
            }
            client.frame(&gpu, &mut camera, 0.0);
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &camera, 0.0);
            let pixels = gpu.read_pixels().expect("readback");
            let path = out.join(name);
            image::save_buffer(
                &path,
                &pixels,
                size.0,
                size.1,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
            eprintln!("wrote {}", path.display());
        };
        // The title screen, its intro and pop-in over.
        shoot(&mut client, 4, 50, &format!("acc-title-{tag}.png"));
        // Hovering the corner.
        let (x, y) = client.acc.corner.rect().center();
        let mut input = InputState::new();
        input.on_cursor_moved(f64::from(x), f64::from(y));
        client.handle_screen_input(&mut input, size.0, size.1);
        shoot(&mut client, 6, 60, &format!("acc-title-hover-{tag}.png"));
        // The Accounts screen, opened by clicking the corner.
        assert!(client.acc_title_mouse(x, y, 0, true));
        assert!(client.acc_title_mouse(x, y, 0, false));
        assert!(matches!(client.ui, Ui::Accounts(_)));
        shoot(&mut client, 3, 30, &format!("acc-screen-{tag}.png"));
        if tag != "1080p" {
            continue;
        }
        // The sign-in page with a code (no task: the page is fed as the task would).
        if let Ui::Accounts(screen) = &mut client.ui {
            screen.begin_sign_in();
            screen.sign_in_prompt(mc_auth::DeviceCodePrompt {
                user_code: "ABCD-1234".to_owned(),
                verification_uri: "https://www.microsoft.com/link".to_owned(),
                message: "To sign in, use a web browser to open the page \
                          https://www.microsoft.com/link and enter the code ABCD-1234 to \
                          authenticate."
                    .to_owned(),
                expires_in: Duration::from_secs(900),
                interval: Duration::from_secs(5),
            });
        }
        shoot(&mut client, 3, 30, "acc-signin-1080p.png");
        // Mojang's 403 for an application it has not approved yet.
        let lang = client.assets.language.clone();
        if let Ui::Accounts(screen) = &mut client.ui {
            screen.sign_in_failed(
                &mc_auth::AuthError::ServicesForbidden {
                    body: r#"{"path":"/authentication/login_with_xbox","errorMessage":"Invalid app registration, see https://aka.ms/AppRegInfo for more information"}"#.to_owned(),
                },
                &lang,
            );
        }
        shoot(&mut client, 3, 30, "acc-unapproved-1080p.png");
        // Removing asks first.
        client.acc_back();
        if let Ui::Accounts(screen) = &mut client.ui {
            screen.select(AccountKey::Stored(uuid(2).as_u128()));
        }
        shoot(&mut client, 2, 30, "acc-screen-needs-sign-in-1080p.png");
    }
    // Nobody signed in: the corner asks for an account.
    let mut gpu = Gpu::headless(
        1920,
        1080,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    )
    .expect("gpu");
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(LoginMode::Microsoft, store_dir.path(), &[]) else {
        return;
    };
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, 1920, 1080);
    let mut camera = crate::app::driver::menu_camera();
    camera.set_viewport(1920, 1080);
    for _ in 0..4 {
        client.frame(&gpu, &mut camera, 0.0);
        client
            .aero
            .set_live_clocks(Duration::from_secs(12), Duration::from_secs(11));
        std::thread::sleep(Duration::from_millis(50));
    }
    client.frame(&gpu, &mut camera, 0.0);
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(&gpu, target.view(), &camera, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let path = out.join("acc-title-not-signed-in-1080p.png");
    image::save_buffer(&path, &pixels, 1920, 1080, image::ExtendedColorType::Rgba8)
        .expect("write the PNG");
    eprintln!("wrote {}", path.display());
}

#[test]
fn the_head_only_ever_asks_mojangs_texture_host_for_the_accounts_own_skin() {
    const SKIN: &str = "http://textures.minecraft.net/texture/\
                        1a4af718455d4aab528e7a61f86fa25e6a369d1768dcb13f7df319a713eb810b";
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AccountStore::at(dir.path());
    let mut auth = signed_in(1, "Alpha", later());
    auth.profile.skins.push(mc_auth::Skin {
        id: "skin".to_owned(),
        state: "ACTIVE".to_owned(),
        url: SKIN.to_owned(),
        variant: mc_auth::SkinVariant::Slim,
        alias: None,
    });
    store.add(&auth).expect("add");
    store
        .set_active(Some(ActiveAccount::Stored { uuid: uuid(1) }))
        .expect("choose");
    let (state, _) = startup_with_store(&config(LoginMode::Microsoft), None, Some(store.clone()));
    match &state.head_skin {
        super::HeadSkin::Pending { url, slim, .. } => {
            assert_eq!(url, SKIN);
            assert!(*slim);
        }
        other => panic!("the account's own skin is asked for: {other:?}"),
    }
    // Until it arrives, the default skin.
    assert!(state.head_upload().is_none());
    assert!(state.badge.head.is_some());

    // The list is a file the user can edit: a URL on another host is never fetched.
    let path = store.file_path();
    let text = std::fs::read_to_string(&path).expect("read");
    std::fs::write(
        &path,
        text.replace(SKIN, "https://example.invalid/skin.png"),
    )
    .expect("write");
    let (state, _) = startup_with_store(&config(LoginMode::Microsoft), None, Some(store));
    assert!(matches!(state.head_skin, super::HeadSkin::Default));
}

#[test]
fn opened_from_options_it_goes_back_to_options_with_the_same_parent() {
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(launcher_login(), store_dir.path(), &[]) else {
        return;
    };
    client.handle_action(mc_ui::screens::Action::OpenOptions);
    assert!(matches!(client.ui, Ui::Options(_)));
    let parents = client.options_parents.len();
    client.handle_action(mc_ui::screens::Action::Accounts(AccountsAction::Open));
    assert!(matches!(client.ui, Ui::Accounts(_)));
    // Escape: back to Options, its parent not pushed a second time.
    client.acc_back();
    assert!(matches!(client.ui, Ui::Options(_)));
    assert_eq!(client.options_parents.len(), parents);
    // Done on Options then leaves to the title screen as before.
    client.handle_action(mc_ui::screens::Action::Back);
    assert!(matches!(client.ui, Ui::MainMenu(_)));
}

// ---- end to end: the real sign-in and refresh tasks against a local mock ----

/// The mock's Minecraft profile.
const PROFILE_HEX: &str = "986dec87b7ec47ff89ff033fdb95c4b5";

/// A tiny HTTP server on 127.0.0.1 standing in for the Microsoft, Xbox and Minecraft endpoints:
/// every path gets canned JSON and is recorded. It serves at most 32 connections.
struct MockChain {
    url: String,
    paths: Arc<Mutex<Vec<String>>>,
}

impl MockChain {
    fn start(access: &'static str, refresh: &'static str) -> MockChain {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        let paths = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&paths);
        std::thread::spawn(move || {
            for stream in listener.incoming().take(32).flatten() {
                let _ = serve(stream, access, refresh, &seen);
            }
        });
        MockChain { url, paths }
    }

    fn paths(&self) -> Vec<String> {
        self.paths.lock().expect("lock").clone()
    }

    fn overrides(&self) -> AuthOverride {
        AuthOverride {
            endpoints: mc_auth::Endpoints::rooted_at(&self.url),
            client_id: Some(TEST_CLIENT_ID.to_owned()),
        }
    }
}

fn serve(
    stream: std::net::TcpStream,
    access: &str,
    refresh: &str,
    seen: &Mutex<Vec<String>>,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request = String::new();
    reader.read_line(&mut request)?;
    let path = request.split_whitespace().nth(1).unwrap_or("").to_owned();
    let mut length = 0usize;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    seen.lock().expect("lock").push(path.clone());
    let uhs = serde_json::json!({ "xui": [ { "uhs": "USERHASH" } ] });
    let answer = match path.as_str() {
        "/oauth2/v2.0/devicecode" => serde_json::json!({
            "device_code": "fake-device-code-0123456789",
            "user_code": "ABCD1234",
            "verification_uri": "https://example.invalid/link",
            "expires_in": 900,
            "interval": 0,
        }),
        "/oauth2/v2.0/token" => serde_json::json!({
            "access_token": "fake-msa-access-0123456789",
            "refresh_token": refresh,
        }),
        "/user/authenticate" => {
            serde_json::json!({ "Token": "fake-xbl-token-0123456789", "DisplayClaims": uhs })
        }
        "/xsts/authorize" => {
            serde_json::json!({ "Token": "fake-xsts-token-0123456789", "DisplayClaims": uhs })
        }
        "/authentication/login_with_xbox" => {
            serde_json::json!({ "access_token": access, "expires_in": 86_400 })
        }
        "/minecraft/profile" => serde_json::json!({ "id": PROFILE_HEX, "name": "MockPlayer" }),
        _ => serde_json::json!({}),
    }
    .to_string();
    let mut stream = stream;
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n",
        answer.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(answer.as_bytes())?;
    stream.flush()
}

fn signed_in_as(id: Uuid, name: &str, expires_at: u64) -> Authenticated {
    let mut auth = signed_in(0, name, expires_at);
    auth.profile.uuid = id;
    auth.credentials.uuid = id;
    auth.session = Session::new(id, name, auth.credentials.mc_access_token.clone());
    auth
}

#[test]
fn adding_an_account_end_to_end_shows_the_code_then_stores_and_uses_it() {
    let mock = MockChain::start(
        "fake-mc-access-mock-0123456789",
        "fake-refresh-mock-0123456789",
    );
    let store_dir = tempfile::tempdir().expect("tempdir");
    let Some((mut client, _dir)) = client(launcher_login(), store_dir.path(), &[]) else {
        return;
    };
    client.acc.overrides = Some(mock.overrides());
    client.acc_open();
    client.acc_action(AccountsAction::Add);
    let mut saw_code = false;
    let deadline = Instant::now() + Duration::from_secs(20);
    while client.acc.add_task.is_some() && Instant::now() < deadline {
        client.acc_frame();
        if let Ui::Accounts(screen) = &client.ui {
            saw_code |= screen
                .sign_in()
                .and_then(|page| page.prompt.as_ref())
                .is_some_and(|prompt| prompt.user_code == "ABCD1234");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(client.acc.add_task.is_none(), "the sign-in finished");
    assert!(
        saw_code,
        "the code was on the page while Microsoft waited for the player"
    );
    let profile = Uuid::parse_str(PROFILE_HEX).expect("uuid");
    let auth = client.auth.as_ref().expect("auth");
    assert_eq!(auth.username, "MockPlayer");
    assert_eq!(auth.uuid, profile);
    assert_eq!(auth.expose(), "fake-mc-access-mock-0123456789");
    let store = AccountStore::at(store_dir.path());
    let list = store.load().expect("load");
    assert_eq!(list.active, Some(ActiveAccount::Stored { uuid: profile }));
    let secrets = store.unseal(&list.accounts[0]).expect("unseal");
    assert_eq!(secrets.refresh_token, "fake-refresh-mock-0123456789");
    assert_eq!(
        mock.paths(),
        [
            "/oauth2/v2.0/devicecode",
            "/oauth2/v2.0/token",
            "/user/authenticate",
            "/xsts/authorize",
            "/authentication/login_with_xbox",
            "/minecraft/profile",
        ]
    );
}

#[test]
fn an_expiring_account_is_refreshed_end_to_end_before_the_join() {
    let mock = MockChain::start(
        "fake-mc-access-renewed-0123456789",
        "fake-refresh-rotated-0123456789",
    );
    let store_dir = tempfile::tempdir().expect("tempdir");
    let profile = Uuid::parse_str(PROFILE_HEX).expect("uuid");
    let expiring = signed_in_as(profile, "MockPlayer", mc_auth::unix_now() + 60);
    let Some((mut client, _dir)) = client(LoginMode::Microsoft, store_dir.path(), &[expiring])
    else {
        return;
    };
    client.acc.overrides = Some(mock.overrides());
    assert_eq!(
        client.acc.active,
        Some(ActiveAccount::Stored { uuid: profile })
    );
    client.connect("127.0.0.1", free_port());
    assert!(client.session.is_none(), "the join waits for the refresh");
    let deadline = Instant::now() + Duration::from_secs(20);
    while client.acc.join_pending.is_some() && Instant::now() < deadline {
        client.acc_frame();
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(client.acc.join_pending.is_none(), "the refresh finished");
    assert!(client.session.is_some(), "then the join went ahead");
    assert_eq!(
        client.auth.as_ref().expect("auth").expose(),
        "fake-mc-access-renewed-0123456789"
    );
    let store = AccountStore::at(store_dir.path());
    let list = store.load().expect("load");
    assert_eq!(list.accounts.len(), 1, "updated in place");
    assert_eq!(
        store
            .unseal(&list.accounts[0])
            .expect("unseal")
            .refresh_token,
        "fake-refresh-rotated-0123456789",
        "the rotated refresh token is the one stored"
    );
    assert_eq!(
        mock.paths(),
        [
            "/oauth2/v2.0/token",
            "/user/authenticate",
            "/xsts/authorize",
            "/authentication/login_with_xbox",
            "/minecraft/profile",
        ]
    );
    client.disconnect();
}
