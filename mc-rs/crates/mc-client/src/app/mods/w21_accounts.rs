//! Accounts: the account corner on the Aero title screen, the
//! Accounts screen, adding an account with the in-client Microsoft sign-in, switching, and
//! refreshing an added account's Minecraft token before a join.
//!
//! The pieces live elsewhere: the store and its sealing in `mc_auth::accounts` (the per-user
//! config folder, never the game folder; Windows DPAPI), the screens in
//! `mc_ui::screens::accounts`. This file is the glue: what the client knows about the accounts,
//! the background tasks, and the hooks.
//!
//! # Which account is in use
//! At start: the one chosen last time ([`mc_auth::ActiveAccount`] in the store) when it is still
//! there; otherwise the launcher's session (`--accessToken`), otherwise the added account used
//! most recently. Only that account is unsealed. The session it gives is `Client::auth`, which
//! every join, the resource-pack request headers and everything else that names the player
//! already read; switching replaces it, only while no world is loaded.
//!
//! # Joins
//! An added account whose Minecraft token expires within `mc_auth::REFRESH_SKEW` is refreshed
//! (on the network runtime, behind the usual connecting screen) before the join goes ahead.
//! When that fails, or the account's saved sign-in cannot be opened on this PC, the join is not
//! attempted: the disconnect screen says to sign in again (or why else it failed).
//!
//! # Hooks (one line each)
//! * `driver.rs`: the `Ui::Accounts` variant and its name, draw, click, wheel and Escape arms;
//!   `Action::Accounts` in `handle_action`; [`startup`] in `Client::new` and the `acc` field;
//!   [`Client::acc_before_join`] at the top of `open_connection`; [`Client::acc_frame`] each
//!   frame.
//! * `aero_title.rs`: [`title_corner`] and [`Client::acc_title_head`] in `aero_title_frame`,
//!   [`Client::acc_title_mouse`] in `aero_title_mouse`.
//!
//! # Cost
//! Off the title screen and the Accounts screen: [`Client::acc_frame`] is two `Option` tests, an
//! enum test and a length test per frame (plus, for at most 30 s after the account changes, one
//! skin-cache lookup while its head waits for its skin). Nothing allocates, locks, hashes or
//! formats per frame anywhere else.

use std::collections::HashSet;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use mc_auth::accounts::loggable;
use mc_auth::{
    AccountStore, ActiveAccount, AuthClient, AuthConfig, AuthError, Authenticated,
    DeviceCodePrompt, JoinSession, Session, StoredAccount,
};
use mc_mods::ModEvent;
use mc_render::entity::skin::SkinModel;
use mc_ui::screens::accounts::corner::{AccountCorner, CornerAccount};
use mc_ui::screens::accounts::{
    AccountKey, AccountRow, AccountsAction, AccountsScreen, RowStatus, Tone,
};
use mc_ui::screens::aero_title::AeroTitleEnv;
use mc_ui::screens::connect::ConnectingScreen;
use mc_ui::screens::modmenu::glass::GlassFrame;
use mc_ui::widget::draw::{DrawList, UiTexture};
use uuid::Uuid;

use super::super::{Client, Shared, Ui};
use crate::app::config::{ClientConfig, LoginMode};
use crate::app::state::Transition;
use crate::net::DisconnectReason;

/// The `aero_title` mod's "Accounts" button (`mc_mods::mods::aero_title::ACCOUNTS`).
const MENU_MOD: &str = "aero_title";
/// Its option id.
const MENU_BUTTON: &str = "accounts";
/// How long the head waits for the account's own skin (from the skin cache on disk, or Mojang's
/// texture server) before it settles for the default skin.
const HEAD_WAIT: Duration = Duration::from_secs(30);

/// The head of the account in use: its own skin once the skin cache has it, else the default.
#[derive(Debug, Clone)]
enum HeadSkin {
    /// No skin of its own known: the default skin (Steve or Alex by UUID).
    Default,
    /// Asked the skin cache (which only fetches from Mojang's texture host).
    Pending {
        key: String,
        url: String,
        slim: bool,
        until: Instant,
    },
    /// Downloaded: the UI texture and its 64x64 RGBA pixels.
    Ready {
        url: String,
        texture: UiTexture,
        rgba: Arc<[u8]>,
    },
}

/// What the background tasks send back.
enum AccMsg {
    /// Microsoft issued the sign-in code.
    Prompt {
        generation: u64,
        prompt: DeviceCodePrompt,
    },
    /// The sign-in finished.
    Added {
        generation: u64,
        result: Box<Result<Authenticated, AuthError>>,
    },
    /// The refresh before a join finished.
    Joined {
        generation: u64,
        uuid: Uuid,
        host: String,
        port: u16,
        result: Box<Result<JoinSession, AuthError>>,
    },
}

/// The client's account state (`Client::acc`).
pub(crate) struct AccountsState {
    /// `None` where there is no per-user folder, in offline and replay runs, and in tests unless
    /// one sets a store.
    store: Option<AccountStore>,
    /// The added accounts' public details (nothing unsealed).
    accounts: Vec<StoredAccount>,
    /// The launcher's session: name and UUID (its token stays in the config).
    launcher: Option<(String, Uuid)>,
    /// The offline (developer) name, in offline mode.
    offline: Option<(String, Uuid)>,
    /// The account in use.
    active: Option<ActiveAccount>,
    /// Added accounts whose saved sign-in expired or cannot be opened here.
    needs_sign_in: HashSet<Uuid>,
    tx: mpsc::Sender<AccMsg>,
    rx: mpsc::Receiver<AccMsg>,
    generation: u64,
    /// The sign-in in progress.
    add_task: Option<(u64, tokio::task::JoinHandle<()>)>,
    /// The refresh a held join waits for.
    join_pending: Option<u64>,
    /// The title screen's corner and what it shows.
    corner: AccountCorner,
    badge: CornerAccount,
    /// The head of the account in use.
    head_skin: HeadSkin,
    /// The Accounts screen was opened from the Options screen (closing it goes back there).
    from_options: bool,
    /// Tests: point the sign-in and refresh at a mock server.
    overrides: Option<AuthOverride>,
}

impl Default for AccountsState {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        let mut state = AccountsState {
            store: None,
            accounts: Vec::new(),
            launcher: None,
            offline: None,
            active: None,
            needs_sign_in: HashSet::new(),
            tx,
            rx,
            generation: 0,
            add_task: None,
            join_pending: None,
            corner: AccountCorner::new(),
            badge: CornerAccount {
                name: String::new(),
                caption: String::new(),
                head: None,
                attention: false,
            },
            head_skin: HeadSkin::Default,
            from_options: false,
            overrides: None,
        };
        state.refresh_badge();
        state
    }
}

impl std::fmt::Debug for AccountsState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccountsState")
            .field("accounts", &self.accounts.len())
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

/// `Client::new`: the account state, and the session to start with (`auth` is the launcher's or
/// the offline one, from the command line).
pub(crate) fn startup(
    config: &ClientConfig,
    auth: Option<Session>,
) -> (AccountsState, Option<Session>) {
    let store = if uses_accounts(config) && !cfg!(test) {
        match AccountStore::discover() {
            Ok(store) => Some(store),
            Err(error) => {
                tracing::warn!(error = %loggable(&error),"no per-user folder: accounts cannot be added");
                None
            }
        }
    } else {
        None
    };
    startup_with_store(config, auth, store)
}

/// Whether this run has accounts at all: not the developer's offline mode, a replay or a
/// diagnostic run.
fn uses_accounts(config: &ClientConfig) -> bool {
    !matches!(config.login, LoginMode::Offline(_))
        && config.replay.is_none()
        && config.diagnostic_simulation.is_none()
}

/// [`startup`] with a given store (tests use a temporary folder).
pub(crate) fn startup_with_store(
    config: &ClientConfig,
    auth: Option<Session>,
    store: Option<AccountStore>,
) -> (AccountsState, Option<Session>) {
    let mut state = AccountsState {
        store,
        ..AccountsState::default()
    };
    match &config.login {
        LoginMode::Launcher { username, uuid, .. } => {
            state.launcher = Some((username.clone(), *uuid));
        }
        LoginMode::Offline(name) => {
            state.offline = Some((name.clone(), auth.as_ref().map_or(Uuid::nil(), |s| s.uuid)));
        }
        LoginMode::Microsoft => {}
    }
    let saved = state.reload();
    let auth = match state.choose(saved) {
        Some(ActiveAccount::Stored { uuid }) => {
            state.active = Some(ActiveAccount::Stored { uuid });
            match state.open(uuid) {
                Ok(session) => Some(session),
                Err(error) => {
                    tracing::warn!(error = %loggable(&error),"the chosen account needs a new sign-in");
                    state.needs_sign_in.insert(uuid);
                    None
                }
            }
        }
        Some(ActiveAccount::Launcher) => {
            state.active = Some(ActiveAccount::Launcher);
            auth
        }
        None => auth,
    };
    state.refresh_badge();
    tracing::info!(
        accounts = state.accounts.len(),
        launcher = state.launcher.is_some(),
        in_use = ?state.active.map(|active| matches!(active, ActiveAccount::Launcher)),
        "accounts loaded"
    );
    (state, auth)
}

impl AccountsState {
    /// Re-read the list; returns the saved choice. A list that cannot be read is an empty one.
    fn reload(&mut self) -> Option<ActiveAccount> {
        let Some(store) = &self.store else {
            return None;
        };
        match store.load() {
            Ok(list) => {
                self.accounts = list.accounts;
                list.active
            }
            Err(error) => {
                tracing::warn!(error = %loggable(&error),"the account list could not be read");
                self.accounts.clear();
                None
            }
        }
    }

    /// The saved choice when it can still be honoured, otherwise the default: the launcher's
    /// session, else the added account used most recently.
    fn choose(&self, saved: Option<ActiveAccount>) -> Option<ActiveAccount> {
        match saved {
            Some(ActiveAccount::Stored { uuid }) if self.account(uuid).is_some() => saved,
            Some(ActiveAccount::Launcher) if self.launcher.is_some() => saved,
            _ if self.launcher.is_some() => Some(ActiveAccount::Launcher),
            _ => self
                .accounts
                .iter()
                .max_by_key(|account| (account.last_used_at, account.added_at))
                .map(|account| ActiveAccount::Stored { uuid: account.uuid }),
        }
    }

    fn account(&self, uuid: Uuid) -> Option<&StoredAccount> {
        self.accounts.iter().find(|account| account.uuid == uuid)
    }

    /// Unseal an added account's session (only ever the one being used).
    fn open(&self, uuid: Uuid) -> Result<Session, AuthError> {
        let store = self.store.as_ref().ok_or(AuthError::CacheCorrupt)?;
        let account = self.account(uuid).ok_or(AuthError::TokenRejected)?;
        store.session(account)
    }

    fn next_generation(&mut self) -> u64 {
        self.generation += 1;
        self.generation
    }

    fn active_key(&self) -> Option<AccountKey> {
        self.active.map(|active| match active {
            ActiveAccount::Launcher => AccountKey::Launcher,
            ActiveAccount::Stored { uuid } => AccountKey::Stored(uuid.as_u128()),
        })
    }

    fn name_of(&self, key: AccountKey) -> String {
        match key {
            AccountKey::Launcher => self.launcher.as_ref().map(|(name, _)| name.clone()),
            AccountKey::Stored(raw) => self
                .account(Uuid::from_u128(raw))
                .map(|account| account.username.clone()),
        }
        .unwrap_or_default()
    }

    /// The Accounts screen's rows: the launcher's session first, then the added accounts. The
    /// account in use shows its own skin once it is downloaded.
    fn rows(&self) -> Vec<AccountRow> {
        let in_use = self.active_uuid();
        let head = |uuid: Uuid| match &self.head_skin {
            HeadSkin::Ready { texture, .. } if Some(uuid) == in_use => texture.clone(),
            _ => default_head(uuid),
        };
        let launcher = self.launcher.iter().map(|(name, uuid)| AccountRow {
            key: AccountKey::Launcher,
            name: name.clone(),
            head: head(*uuid),
            head_sheet_height: 64.0,
            status: RowStatus::Ready,
        });
        let added = self.accounts.iter().map(|account| AccountRow {
            key: AccountKey::Stored(account.uuid.as_u128()),
            name: account.username.clone(),
            head: head(account.uuid),
            head_sheet_height: 64.0,
            status: if self.needs_sign_in.contains(&account.uuid) {
                RowStatus::NeedsSignIn
            } else {
                RowStatus::Ready
            },
        });
        launcher.chain(added).collect()
    }

    /// What the title screen's corner shows, from the account in use.
    fn refresh_badge(&mut self) {
        let (name, caption, uuid, attention) = match self.active {
            Some(ActiveAccount::Stored { uuid }) => {
                let name = self
                    .account(uuid)
                    .map_or_else(String::new, |account| account.username.clone());
                if self.needs_sign_in.contains(&uuid) {
                    (name, "Sign in again", Some(uuid), true)
                } else {
                    (name, "Microsoft account", Some(uuid), false)
                }
            }
            Some(ActiveAccount::Launcher) => match &self.launcher {
                Some((name, uuid)) => (name.clone(), "From the launcher", Some(*uuid), false),
                None => ("Not signed in".to_owned(), "Add an account", None, true),
            },
            None => match &self.offline {
                Some((name, uuid)) => (name.clone(), "Offline mode", Some(*uuid), false),
                None => match &self.launcher {
                    Some((name, uuid)) => (name.clone(), "From the launcher", Some(*uuid), false),
                    None => ("Not signed in".to_owned(), "Add an account", None, true),
                },
            },
        };
        self.update_head_skin();
        let head = match &self.head_skin {
            HeadSkin::Ready { texture, .. } => Some((texture.clone(), 64.0)),
            _ => uuid.map(|uuid| (default_head(uuid), 64.0)),
        };
        self.badge = CornerAccount {
            name,
            caption: caption.to_owned(),
            head,
            attention,
        };
    }

    /// The UUID of the account in use (the launcher's, an added one's, or the offline one).
    fn active_uuid(&self) -> Option<Uuid> {
        match self.active {
            Some(ActiveAccount::Stored { uuid }) => Some(uuid),
            Some(ActiveAccount::Launcher) => self.launcher.as_ref().map(|(_, uuid)| *uuid),
            None => self
                .launcher
                .as_ref()
                .or(self.offline.as_ref())
                .map(|(_, uuid)| *uuid),
        }
    }

    /// The skin the account in use wears, as of its last sign-in or refresh: its own entry, or
    /// (for the launcher's session) an added account with the same UUID. Only a URL on Mojang's
    /// texture host counts: the list is a file the user can edit.
    fn active_skin_url(&self) -> Option<(String, bool)> {
        let uuid = self.active_uuid()?;
        let account = self.account(uuid)?;
        let url = account.skin_url.as_deref()?;
        mc_render::entity::skin::is_mojang_texture_url(url).then(|| (url.to_owned(), account.slim))
    }

    /// Ask for the account in use's own skin when it changed (the frame hook polls for it).
    fn update_head_skin(&mut self) {
        let wanted = self.active_skin_url();
        let current = match &self.head_skin {
            HeadSkin::Default => None,
            HeadSkin::Pending { url, .. } | HeadSkin::Ready { url, .. } => Some(url.as_str()),
        };
        if wanted.as_ref().map(|(url, _)| url.as_str()) == current {
            return;
        }
        self.head_skin = match wanted {
            Some((url, slim)) => match mc_render::entity::skin::skin_key_for_texture_url(&url) {
                Some(key) => HeadSkin::Pending {
                    key,
                    url,
                    slim,
                    until: Instant::now() + HEAD_WAIT,
                },
                None => HeadSkin::Default,
            },
            None => HeadSkin::Default,
        };
    }

    /// The downloaded head texture and its pixels, for the UI renderer's upload (the Accounts
    /// screen's rows).
    pub(crate) fn head_upload(&self) -> Option<(&UiTexture, &Arc<[u8]>)> {
        match &self.head_skin {
            HeadSkin::Ready { texture, rgba, .. } => Some((texture, rgba)),
            _ => None,
        }
    }

    /// Whether this build can sign in with Microsoft (an application ID from the build, the
    /// environment or the config file).
    fn sign_in_available(&self) -> bool {
        match &self.overrides {
            Some(overrides) => overrides.client_id.is_some(),
            None => matches!(AuthConfig::load_with_source(), Ok((_, Some(_)))),
        }
    }
}

/// The default skin for a UUID: java `DefaultPlayerSkin.getDefaultSkin` (Steve or Alex).
fn default_head(uuid: Uuid) -> UiTexture {
    UiTexture::gui(SkinModel::default_for_uuid(uuid.as_u128()).default_texture())
}

/// Draw the account corner over the title screen this frame (after
/// the title screen's own `build_layers`, into the same lists and glass frame).
pub(crate) fn title_corner(
    state: &mut AccountsState,
    env: &mut AeroTitleEnv<'_>,
    opened_at: Instant,
    cursor: (f32, f32),
    list: &mut DrawList,
    crisp: Option<&mut DrawList>,
    glass: &mut GlassFrame,
) {
    state
        .corner
        .build(&state.badge, env, opened_at, cursor, list, crisp, glass);
}

impl Client {
    /// The corner's head texture in the title screen's phases (a map
    /// lookup per phase per title-screen frame once it is there).
    pub(crate) fn acc_title_head<P>(&self, gpu: &mc_render::Gpu, phases: [&Shared<P>; 2])
    where
        P: TitleRenderer,
    {
        let Some((key, _)) = &self.acc.badge.head else {
            return;
        };
        let own = self
            .acc
            .head_upload()
            .filter(|(texture, _)| *texture == key);
        for phase in phases {
            phase.with(|phase| {
                let renderer = phase.title_renderer();
                match own {
                    // The account's own skin: its pixels, uploaded once per phase.
                    Some((texture, rgba)) => {
                        if !renderer.has_texture(texture) {
                            renderer.upload(gpu, texture.clone(), rgba, 64, 64);
                        }
                    }
                    // The default skin, from the vanilla resources.
                    None => {
                        let _ = renderer.ensure_texture(gpu, &self.assets.resources, key);
                    }
                }
            });
        }
    }

    /// A mouse edge on the title screen at `(x, y)` (physical
    /// pixels). Returns whether the corner took it; a completed click opens the Accounts screen.
    pub(crate) fn acc_title_mouse(&mut self, x: f32, y: f32, button: i32, pressed: bool) -> bool {
        // Only while the Aero title screen is what shows: the corner
        // keeps the rectangle it last drew, so with the Aero Title Screen mod turned off
        // (vanilla's title screen) or under the loading screen, its old place took clicks and
        // opened the Accounts screen from an empty spot of vanilla's menu.
        if !self.aero_title_showing() {
            return false;
        }
        let Ok(button) = u8::try_from(button) else {
            return false;
        };
        if pressed {
            return self.acc.corner.mouse_pressed(x, y, button);
        }
        if self.acc.corner.mouse_released(x, y, button) {
            self.play_button_press();
            self.acc_open();
            return true;
        }
        false
    }

    /// Each frame. Finished background work, the head's skin while it is
    /// on its way, and the mod menu's "Accounts" button. Two `Option` tests, an enum test and a
    /// length test when there is nothing to do.
    pub(crate) fn acc_frame(&mut self) {
        if self.acc.add_task.is_some() || self.acc.join_pending.is_some() {
            while let Ok(message) = self.acc.rx.try_recv() {
                self.acc_message(message);
            }
        }
        if matches!(self.acc.head_skin, HeadSkin::Pending { .. }) {
            self.acc_poll_head();
        }
        if !self.mods.events.is_empty() {
            let before = self.mods.events.len();
            self.mods.events.retain(|event| {
                !matches!(
                    event,
                    ModEvent::Button {
                        mod_id: MENU_MOD,
                        option_id: MENU_BUTTON,
                    }
                )
            });
            if self.mods.events.len() != before {
                self.acc_open();
            }
        }
    }

    /// The account in use's own skin from the skin cache (its disk cache, or Mojang's texture
    /// host only), while the head waits for it: one map lookup and one channel poll a frame, for
    /// at most [`HEAD_WAIT`].
    fn acc_poll_head(&mut self) {
        let HeadSkin::Pending {
            key,
            url,
            slim,
            until,
        } = &self.acc.head_skin
        else {
            return;
        };
        if Instant::now() >= *until {
            self.acc.head_skin = HeadSkin::Default;
            return;
        }
        let (key, url) = (key.clone(), url.clone());
        let model = if *slim {
            SkinModel::Slim
        } else {
            SkinModel::Classic
        };
        self.appearance.spawn_worker(self.runtime.handle());
        self.appearance.skins.poll();
        let Some(skin) = self.appearance.skins.get(&key, &url, model) else {
            return;
        };
        let rgba = skin.rgba.clone();
        self.acc.head_skin = HeadSkin::Ready {
            url,
            texture: Client::ui_skin_texture_key(&key),
            rgba,
        };
        self.acc.refresh_badge();
        self.acc_refresh_screen();
    }

    /// What the Accounts screen and the corner ask for.
    pub(crate) fn acc_action(&mut self, action: AccountsAction) {
        match action {
            AccountsAction::Open => self.acc_open(),
            AccountsAction::Add => self.acc_add(),
            AccountsAction::CancelAdd => {
                if let Some((_, task)) = self.acc.add_task.take() {
                    task.abort();
                }
            }
            AccountsAction::Use(key) => self.acc_use(key),
            AccountsAction::Remove(key) => self.acc_remove(key),
            AccountsAction::Done => self.acc_close(),
        }
    }

    /// Escape on the Accounts screen.
    pub(crate) fn acc_back(&mut self) {
        let action = match &mut self.ui {
            Ui::Accounts(screen) => screen.back(),
            _ => return,
        };
        if let mc_ui::screens::Action::Accounts(action) = action {
            self.acc_action(action);
        }
    }

    /// Open the Accounts screen over whatever is up (the mod menu closes first).
    pub(crate) fn acc_open(&mut self) {
        if matches!(self.ui, Ui::ModMenu(_)) {
            self.close_mod_menu(Instant::now());
        }
        self.acc.from_options = matches!(self.ui, Ui::Options(_));
        let _ = self.acc.reload();
        let mut screen = AccountsScreen::new(self.acc.rows(), self.acc.active_key());
        screen.set_world_loaded(self.session.is_some());
        screen.set_sign_in_available(self.acc.sign_in_available());
        self.ui = Ui::Accounts(Box::new(screen));
    }

    /// Close the Accounts screen: back to Options when it was opened there, otherwise to the
    /// game in a world, or the title screen.
    fn acc_close(&mut self) {
        if let Some((_, task)) = self.acc.add_task.take() {
            task.abort();
        }
        if std::mem::take(&mut self.acc.from_options) {
            // The Options screen again, as `Action::OpenOptions` builds it, but without pushing a
            // second parent: the one pushed when Options first opened is still on the stack.
            let in_world = self.session.is_some();
            let difficulty = self.current_server_difficulty();
            self.ui = Ui::Options(Box::new(
                mc_ui::screens::options::OptionsScreen::new_with_difficulty(in_world, difficulty),
            ));
        } else if self.session.is_some() {
            self.show_screen(None);
        } else {
            self.show_main_menu();
        }
    }

    fn acc_screen(&mut self) -> Option<&mut AccountsScreen> {
        match &mut self.ui {
            Ui::Accounts(screen) => Some(screen),
            _ => None,
        }
    }

    /// Put fresh rows on the screen (after an add, a removal or a switch).
    fn acc_refresh_screen(&mut self) {
        let rows = self.acc.rows();
        let active = self.acc.active_key();
        if let Some(screen) = self.acc_screen() {
            screen.set_rows(rows, active);
        }
    }

    /// Start the Microsoft sign-in for a new account, on the network runtime.
    fn acc_add(&mut self) {
        if !self.acc.sign_in_available() || self.acc.store.is_none() {
            let text = mc_ui::screens::accounts::not_set_up_text(&self.assets.language);
            if let Some(screen) = self.acc_screen() {
                screen.set_sign_in_available(false);
                screen.set_status(text, Tone::Notice);
            }
            return;
        }
        if let Some((_, task)) = self.acc.add_task.take() {
            task.abort();
        }
        let generation = self.acc.next_generation();
        let tx = self.acc.tx.clone();
        let proxy = self.config.proxy.clone();
        let overrides = self.acc.overrides.clone();
        let task = self.runtime.handle().spawn(async move {
            let prompt_tx = tx.clone();
            let result = async {
                let client = auth_client(proxy, overrides)?;
                client
                    .login_device_code(move |prompt| {
                        let _ = prompt_tx.send(AccMsg::Prompt {
                            generation,
                            prompt: prompt.clone(),
                        });
                    })
                    .await
            }
            .await;
            let _ = tx.send(AccMsg::Added {
                generation,
                result: Box::new(result),
            });
        });
        self.acc.add_task = Some((generation, task));
        if let Some(screen) = self.acc_screen() {
            screen.begin_sign_in();
        }
    }

    fn acc_message(&mut self, message: AccMsg) {
        match message {
            AccMsg::Prompt { generation, prompt } => {
                if self.acc.add_task.as_ref().map(|(g, _)| *g) == Some(generation) {
                    if let Some(screen) = self.acc_screen() {
                        screen.sign_in_prompt(prompt);
                    }
                }
            }
            AccMsg::Added { generation, result } => {
                if self.acc.add_task.as_ref().map(|(g, _)| *g) != Some(generation) {
                    return;
                }
                self.acc.add_task = None;
                self.acc_added(*result);
            }
            AccMsg::Joined {
                generation,
                uuid,
                host,
                port,
                result,
            } => self.acc_joined(generation, uuid, &host, port, *result),
        }
    }

    /// The sign-in finished: store the account, and use it when no world is loaded.
    fn acc_added(&mut self, result: Result<Authenticated, AuthError>) {
        let auth = match result {
            Ok(auth) => auth,
            Err(error) => {
                tracing::warn!(error = %loggable(&error),"adding an account failed");
                let lang = &self.assets.language;
                if let Ui::Accounts(screen) = &mut self.ui {
                    screen.sign_in_failed(&error, lang);
                }
                return;
            }
        };
        let Some(store) = self.acc.store.clone() else {
            return;
        };
        let account = match store.add(&auth) {
            Ok(account) => account,
            Err(error) => {
                tracing::warn!(error = %loggable(&error),"the new account could not be saved");
                if let Some(screen) = self.acc_screen() {
                    screen.sign_in_finished(AccountKey::Launcher);
                    screen.set_status("Couldn't save the account on this PC.", Tone::Problem);
                }
                return;
            }
        };
        let uuid = account.uuid;
        let name = account.username.clone();
        self.acc.needs_sign_in.remove(&uuid);
        let _ = self.acc.reload();
        let switched = self.session.is_none()
            && self
                .acc_switch(ActiveAccount::Stored { uuid }, Some(auth.session))
                .is_ok();
        self.acc_refresh_screen();
        let status = if switched {
            format!("Added {name}. You're playing as {name} now.")
        } else {
            format!("Added {name}. Leave the server to switch to it.")
        };
        if let Some(screen) = self.acc_screen() {
            screen.sign_in_finished(AccountKey::Stored(uuid.as_u128()));
            screen.set_status(status, Tone::Done);
        }
    }

    /// Make `choice` the account in use: `Client::auth`, the corner, the saved choice. `opened`
    /// is the session when the caller already has it (a sign-in that just finished).
    fn acc_switch(
        &mut self,
        choice: ActiveAccount,
        opened: Option<Session>,
    ) -> Result<(), AuthError> {
        let session = match (choice, opened) {
            (_, Some(session)) => session,
            (ActiveAccount::Launcher, None) => self
                .config
                .login
                .launcher_session()
                .ok_or(AuthError::TokenRejected)?,
            (ActiveAccount::Stored { uuid }, None) => match self.acc.open(uuid) {
                Ok(session) => session,
                Err(error) => {
                    self.acc.needs_sign_in.insert(uuid);
                    return Err(error);
                }
            },
        };
        self.auth = Some(session);
        self.acc.active = Some(choice);
        self.acc.refresh_badge();
        if let Some(store) = &self.acc.store {
            if let Err(error) = store.set_active(Some(choice)) {
                tracing::warn!(error = %loggable(&error),"the account choice could not be saved");
            }
        }
        Ok(())
    }

    /// The Use button: switch, only with no world loaded.
    fn acc_use(&mut self, key: AccountKey) {
        if self.session.is_some() {
            return;
        }
        let name = self.acc.name_of(key);
        let choice = match key {
            AccountKey::Launcher => ActiveAccount::Launcher,
            AccountKey::Stored(raw) => ActiveAccount::Stored {
                uuid: Uuid::from_u128(raw),
            },
        };
        let result = self.acc_switch(choice, None);
        self.acc_refresh_screen();
        let (message, tone) = match result {
            Ok(()) => (format!("You're playing as {name} now."), Tone::Done),
            Err(error) => {
                tracing::warn!(error = %loggable(&error),"switching accounts failed");
                (format!("Sign in again to use {name}."), Tone::Notice)
            }
        };
        if let Some(screen) = self.acc_screen() {
            screen.set_status(message, tone);
        }
    }

    /// The Remove button (confirmed): delete the account and its saved sign-in.
    fn acc_remove(&mut self, key: AccountKey) {
        let AccountKey::Stored(raw) = key else {
            return;
        };
        let uuid = Uuid::from_u128(raw);
        let in_use = self.acc.active == Some(ActiveAccount::Stored { uuid });
        if in_use && self.session.is_some() {
            return;
        }
        let name = self.acc.name_of(key);
        let Some(store) = self.acc.store.clone() else {
            return;
        };
        if let Err(error) = store.remove(uuid) {
            tracing::warn!(error = %loggable(&error),"the account could not be removed");
            if let Some(screen) = self.acc_screen() {
                screen.set_status("Couldn't remove the account.", Tone::Problem);
            }
            return;
        }
        let _ = self.acc.reload();
        self.acc.needs_sign_in.remove(&uuid);
        if in_use {
            // The launcher's session when there is one, otherwise nobody.
            self.auth = self.config.login.launcher_session();
            self.acc.active = self.acc.launcher.as_ref().map(|_| ActiveAccount::Launcher);
            self.acc.refresh_badge();
        }
        self.acc_refresh_screen();
        if let Some(screen) = self.acc_screen() {
            screen.set_status(
                format!("Removed {name}. Its saved sign-in was deleted from this PC."),
                Tone::Done,
            );
        }
    }

    /// From `open_connection`, before the login is chosen. Returns `true`
    /// when the join is held (a refresh first) or refused (the account needs a new sign-in, or
    /// nobody is signed in); `false` lets it go ahead as before.
    pub(crate) fn acc_before_join(&mut self, host: &str, port: u16) -> bool {
        if matches!(self.config.login, LoginMode::Offline(_)) {
            return false;
        }
        let Some(ActiveAccount::Stored { uuid }) = self.acc.active else {
            if self.auth.is_none() {
                self.acc_fail_join(
                    "You're not signed in. Add your Microsoft account in Options > \
                     Accounts... (or the account corner at the top right of the title screen).",
                );
                return true;
            }
            return false;
        };
        let Some(account) = self.acc.account(uuid).cloned() else {
            return false;
        };
        if self.acc.needs_sign_in.contains(&uuid) || self.auth.is_none() {
            self.acc_fail_join(&format!(
                "The saved sign-in for {} has expired. Sign in again in Accounts to keep \
                 playing.",
                account.username
            ));
            return true;
        }
        if !account.needs_refresh(mc_auth::unix_now()) {
            return false;
        }
        let Some(store) = self.acc.store.clone() else {
            return false;
        };
        let generation = self.acc.next_generation();
        self.acc.join_pending = Some(generation);
        let tx = self.acc.tx.clone();
        let proxy = self.config.proxy.clone();
        let overrides = self.acc.overrides.clone();
        let (host_owned, port_owned) = (host.to_owned(), port);
        self.runtime.handle().spawn(async move {
            let result = async {
                let client = auth_client(proxy, overrides)?;
                store.session_for_join(&client, uuid).await
            }
            .await;
            let _ = tx.send(AccMsg::Joined {
                generation,
                uuid,
                host: host_owned,
                port: port_owned,
                result: Box::new(result),
            });
        });
        tracing::info!("refreshing the sign-in before joining");
        self.ui = Ui::Connecting(Box::new(ConnectingScreen::with_parent(
            format!("{host}:{port}"),
            self.connection_parent,
        )));
        true
    }

    /// The refresh before a join finished: keep the new session, then join (if the player is
    /// still waiting for this join) or say why not.
    fn acc_joined(
        &mut self,
        generation: u64,
        uuid: Uuid,
        host: &str,
        port: u16,
        result: Result<JoinSession, AuthError>,
    ) {
        let current = self.acc.join_pending == Some(generation);
        if current {
            self.acc.join_pending = None;
        }
        let waiting = current && matches!(self.ui, Ui::Connecting(_)) && self.session.is_none();
        let name = self
            .acc
            .account(uuid)
            .map_or_else(String::new, |account| account.username.clone());
        match result {
            Ok(join) => {
                if self.acc.active == Some(ActiveAccount::Stored { uuid }) {
                    self.auth = Some(join.session);
                }
                let _ = self.acc.reload();
                self.acc.refresh_badge();
                if waiting {
                    self.open_connection(host, port);
                }
            }
            Err(error) => {
                tracing::warn!(error = %loggable(&error),"refreshing the sign-in before a join failed");
                let message = if error.needs_new_sign_in() {
                    self.acc.needs_sign_in.insert(uuid);
                    self.acc.refresh_badge();
                    format!(
                        "The saved sign-in for {name} has expired. Sign in again in Accounts \
                         to keep playing."
                    )
                } else if error.is_unapproved_app() {
                    "Mojang hasn't approved this build's Microsoft sign-in yet (Invalid app \
                     registration). Start Aero Client from its launcher to play."
                        .to_owned()
                } else if matches!(error, AuthError::MissingClientId) {
                    "This build of Aero Client can't renew Microsoft sign-ins. Start it from \
                     its launcher to play."
                        .to_owned()
                } else {
                    format!(
                        "Couldn't renew the sign-in for {name}. Check your connection and try \
                         again."
                    )
                };
                if waiting {
                    self.acc_fail_join(&message);
                }
            }
        }
    }

    /// End the connection attempt with `message` on the disconnect screen, as a failed login.
    fn acc_fail_join(&mut self, message: &str) {
        let reason = DisconnectReason::Kicked {
            reason_json: serde_json::json!({ "text": message }).to_string(),
        }
        .with_login_origin();
        let _ = self.machine.apply(Transition::Ended(reason.clone()));
        self.show_disconnected(reason.title_key(), &reason);
    }
}

/// The phase type the title screen's head texture goes into (`AeroTitlePhase`).
pub(crate) trait TitleRenderer {
    /// Its UI renderer.
    fn title_renderer(&mut self) -> &mut mc_ui::widget::UiRenderer;
}

impl TitleRenderer for mc_ui::screens::aero_title::phase::AeroTitlePhase {
    fn title_renderer(&mut self) -> &mut mc_ui::widget::UiRenderer {
        self.renderer()
    }
}

/// Tests only: where the sign-in and refresh go instead of the real services (a local mock) and
/// the client ID they use, so a test never reads the user's own sign-in configuration either.
#[derive(Debug, Clone)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct AuthOverride {
    /// The mock server's endpoints.
    pub(crate) endpoints: mc_auth::Endpoints,
    /// The client ID (`None`: as a build without one).
    pub(crate) client_id: Option<String>,
}

/// The client that signs in and refreshes: the build's (or the development override's)
/// application ID and the launcher's proxy; in tests, a local mock.
fn auth_client(
    proxy: Option<mc_auth::SocksProxy>,
    overrides: Option<AuthOverride>,
) -> Result<AuthClient, AuthError> {
    let Some(overrides) = overrides else {
        return AuthClient::new_with_proxy(AuthConfig::load()?, proxy);
    };
    let config = AuthConfig {
        client_id: overrides.client_id,
    };
    Ok(AuthClient::new_with_proxy(config, proxy)?
        .with_endpoints(overrides.endpoints)
        .with_poll_tuning(mc_auth::PollTuning {
            // Long enough for a test to see the code on the page before the sign-in finishes.
            min_interval: Duration::from_millis(300),
            slow_down_increment: Duration::from_millis(10),
        }))
}

#[cfg(test)]
mod tests;
