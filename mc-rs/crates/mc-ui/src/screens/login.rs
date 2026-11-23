//! The Microsoft sign-in screen: renders `mc-auth`'s device-code prompt and its typed errors.
//!
//! There is no Java original. Vanilla 1.8.9 authenticated through Yggdrasil at launcher time
//! and the client never showed a login screen at all — hazard **H13** and §7 S03 record that
//! Yggdrasil is retired and only the Microsoft flow works. So this screen is a *design*, laid
//! out in vanilla's idiom (dirt background, centred title, 200×20 buttons on the same grid the
//! other menus use) rather than a port.
//!
//! # The flow it drives
//!
//! `mc_auth::login_device_code` takes a callback that fires once, with a
//! [`DeviceCodePrompt`] holding the code, the URL and the polling interval; the future then
//! polls Microsoft until the user finishes or the code expires. The screen is the callback's
//! display half and a state machine over the outcome:
//!
//! ```text
//!  Idle ──(sign in)──> Requesting ──(prompt)──> AwaitingUser ──> Success
//!    ^                     │                        │
//!    └─────(retry)─────────┴───────> Failed <───────┘
//! ```
//!
//! The async half lives in `mc-client` (§11: `tokio` is for the network only, and the game loop
//! is synchronous), so the screen is driven entirely by [`LoginScreen::set_prompt`],
//! [`LoginScreen::succeed`] and [`LoginScreen::fail`] — which is what makes the whole thing
//! testable without touching Microsoft.

use mc_assets::Language;
use mc_auth::{AuthError, DeviceCodePrompt};

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_dirt_background, LABEL_COLOR, TITLE_COLOR};
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::keys::{self, Clipboard};
use crate::widget::{draw_centered_string, Button};

/// Translate an mc-rs extension key, falling back to readable English.
///
/// The stock 1.8.9 language table cannot contain these keys because its client had no in-game
/// Microsoft sign-in flow. Resource packs may still provide translations for them; the fallback
/// only prevents an unmodified install from displaying raw keys such as `auth.signIn`.
fn extension_text(lang: &Language, key: &str, fallback: &str) -> String {
    if lang.has_key(key) {
        lang.tr(key).to_owned()
    } else {
        fallback.to_owned()
    }
}

fn extension_format(lang: &Language, key: &str, fallback: &str, args: &[&str]) -> String {
    let template = if lang.has_key(key) {
        lang.tr(key)
    } else {
        fallback
    };
    mc_assets::format_template(template, args)
}

/// Where the login flow has got to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoginState {
    /// Nothing has been attempted.
    Idle,
    /// Waiting for Microsoft to hand back a device code.
    Requesting,
    /// Showing the code; polling for the user to finish in their browser.
    AwaitingUser,
    /// Signed in as this username.
    Success(String),
    /// The flow failed. The string is the already-rendered message.
    Failed(String),
}

/// Turn an [`AuthError`] into the two lines the screen shows.
///
/// The first is what went wrong; the second is what the player can do about it. Every typed
/// variant §7 S03 lists gets its own advice — that is the whole point of them being typed.
#[must_use]
pub fn describe_error(err: &AuthError, lang: &Language) -> (String, String) {
    // Two failures a player cannot fix get plain words instead of the error's
    // technical text: Mojang's 403 for an application it has not approved yet ("Invalid app
    // registration"), and a build made without a Microsoft application ID.
    if err.is_unapproved_app() {
        return (
            extension_text(
                lang,
                "auth.error.unapprovedApp",
                "Mojang hasn't approved this build's Microsoft sign-in yet (Invalid app \
                 registration).",
            ),
            extension_text(
                lang,
                "auth.error.unapprovedApp.help",
                "Until it is approved, start Aero Client from its launcher to play with your \
                 account.",
            ),
        );
    }
    if matches!(err, AuthError::MissingClientId) {
        return (
            extension_text(
                lang,
                "auth.error.clientId",
                "Microsoft sign-in isn't set up in this build of Aero Client.",
            ),
            extension_text(
                lang,
                "auth.error.clientId.help",
                "Start Aero Client from its launcher to play with your account.",
            ),
        );
    }
    let (key, fallback) = match err {
        AuthError::NoXboxAccount => (
            "auth.error.noXboxAccount.help",
            "Create an Xbox profile, then try again.",
        ),
        AuthError::ChildAccount { .. } => (
            "auth.error.childAccount.help",
            "Add this account to a Microsoft family group, then try again.",
        ),
        AuthError::NoEntitlement => (
            "auth.error.noEntitlement.help",
            "Use an account that owns Minecraft: Java Edition.",
        ),
        AuthError::ServicesForbidden { .. } => (
            "auth.error.servicesForbidden.help",
            "Check the Azure client ID approval and the account's game ownership.",
        ),
        AuthError::Network { .. } | AuthError::HttpClientInit(_) => (
            "auth.error.network.help",
            "Check your network connection, then try again.",
        ),
        AuthError::DeviceCodeExpired => (
            "auth.error.expired.help",
            "Start sign-in again to get a new code.",
        ),
        AuthError::Declined => (
            "auth.error.declined.help",
            "Start sign-in again when you are ready.",
        ),
        AuthError::TokenRejected | AuthError::CacheCorrupt => (
            "auth.error.signInAgain.help",
            "Sign in again to refresh your session.",
        ),
        AuthError::MissingClientId => (
            "auth.error.clientId.help",
            "Configure a Microsoft application client ID, then try again.",
        ),
        _ => (
            "auth.error.generic.help",
            "Try again. Check the log for more details if the problem continues.",
        ),
    };
    (err.to_string(), extension_text(lang, key, fallback))
}

/// The sign-in screen.
#[derive(Debug)]
pub struct LoginScreen {
    /// Where the flow has got to.
    pub state: LoginState,
    /// The prompt, once Microsoft has issued one.
    pub prompt: Option<DeviceCodePrompt>,
    /// Advice text for the current failure, if any.
    pub advice: String,
    buttons: Vec<Button>,
}

/// Button ids.
mod ids {
    pub const SIGN_IN: i32 = 0;
    pub const COPY_CODE: i32 = 1;
    pub const OPEN_BROWSER: i32 = 2;
    pub const CANCEL: i32 = 3;
}

impl Default for LoginScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl LoginScreen {
    /// A fresh sign-in screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: LoginState::Idle,
            prompt: None,
            advice: String::new(),
            buttons: Vec::new(),
        }
    }

    /// Lay out.
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        self.buttons = vec![
            Button::new(
                ids::SIGN_IN,
                w / 2 - 100,
                h / 4 + 96,
                extension_text(lang, "auth.signIn", "Sign in with Microsoft"),
            ),
            Button::sized(
                ids::COPY_CODE,
                w / 2 - 100,
                h / 4 + 120,
                98,
                20,
                extension_text(lang, "auth.copyCode", "Copy code"),
            ),
            Button::sized(
                ids::OPEN_BROWSER,
                w / 2 + 2,
                h / 4 + 120,
                98,
                20,
                extension_text(lang, "auth.openPage", "Open page"),
            ),
            Button::new(
                ids::CANCEL,
                w / 2 - 100,
                h / 4 + 144,
                lang.tr("gui.cancel").to_owned(),
            ),
        ];
        self.refresh_enabled();
    }

    fn refresh_enabled(&mut self) {
        let has_usable_prompt = self.state == LoginState::AwaitingUser && self.prompt.is_some();
        for b in &mut self.buttons {
            b.enabled = match b.id {
                ids::SIGN_IN => matches!(self.state, LoginState::Idle | LoginState::Failed(_)),
                ids::COPY_CODE | ids::OPEN_BROWSER => has_usable_prompt,
                _ => true,
            };
        }
    }

    /// Microsoft issued a device code.
    pub fn set_prompt(&mut self, prompt: DeviceCodePrompt) {
        self.prompt = Some(prompt);
        self.state = LoginState::AwaitingUser;
        self.advice.clear();
        self.refresh_enabled();
    }

    /// The flow finished.
    pub fn succeed(&mut self, username: impl Into<String>) {
        self.state = LoginState::Success(username.into());
        self.prompt = None;
        self.refresh_enabled();
    }

    /// The flow failed.
    pub fn fail(&mut self, err: &AuthError, lang: &Language) {
        let (message, advice) = describe_error(err, lang);
        self.state = LoginState::Failed(message);
        self.advice = advice;
        self.prompt = None;
        self.refresh_enabled();
    }

    /// Each button's rectangle and whether it is enabled (tests of screens that host this one).
    #[cfg(test)]
    pub(crate) fn buttons_for_tests(&self) -> Vec<(crate::widget::draw::Rect, bool)> {
        self.buttons.iter().map(|b| (b.rect, b.enabled)).collect()
    }

    /// Whether the screen is waiting on the network.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        matches!(
            self.state,
            LoginState::Requesting | LoginState::AwaitingUser
        )
    }

    /// Handle a click.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        clipboard: &mut dyn Clipboard,
    ) -> Action {
        if button != 0 {
            return Action::None;
        }
        let hit = self
            .buttons
            .iter()
            .find(|b| b.mouse_pressed(x, y))
            .map(|b| b.id);
        match hit {
            Some(ids::SIGN_IN) => {
                self.state = LoginState::Requesting;
                self.advice.clear();
                self.refresh_enabled();
                Action::BeginLogin
            }
            Some(ids::COPY_CODE) => match &self.prompt {
                Some(p) => {
                    clipboard.set(&p.user_code);
                    Action::CopyToClipboard(p.user_code.clone())
                }
                None => Action::None,
            },
            Some(ids::OPEN_BROWSER) => match &self.prompt {
                Some(p) => Action::OpenUrl(p.verification_uri.clone()),
                None => Action::None,
            },
            Some(ids::CANCEL) => Action::Back,
            _ => Action::None,
        }
    }

    /// Handle a key press.
    pub fn key_typed(&mut self, key: i32) -> Action {
        if key == keys::ESCAPE {
            Action::Back
        } else {
            Action::None
        }
    }

    /// The lines the middle of the screen shows for the current state.
    #[must_use]
    pub fn body_lines(&self, lang: &Language) -> Vec<String> {
        match &self.state {
            LoginState::Idle => vec![extension_text(
                lang,
                "auth.idle",
                "Sign in to play on online-mode servers.",
            )],
            LoginState::Requesting => vec![extension_text(
                lang,
                "auth.requesting",
                "Requesting a sign-in code...",
            )],
            LoginState::AwaitingUser => match &self.prompt {
                Some(p) => vec![
                    p.message.clone(),
                    p.verification_uri.clone(),
                    format!("\u{a7}e\u{a7}l{}", p.user_code),
                    extension_format(
                        lang,
                        "auth.expiresIn",
                        "Code expires in %s minutes",
                        &[&format!("{}", p.expires_in.as_secs() / 60)],
                    ),
                ],
                None => vec![extension_text(
                    lang,
                    "auth.requesting",
                    "Requesting a sign-in code...",
                )],
            },
            LoginState::Success(name) => vec![extension_format(
                lang,
                "auth.success",
                "Signed in as %s",
                &[name],
            )],
            LoginState::Failed(message) => {
                let mut lines = vec![format!("\u{a7}c{message}")];
                if !self.advice.is_empty() {
                    lines.push(self.advice.clone());
                }
                lines
            }
        }
    }

    /// Draw.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        draw_dirt_background(d, ctx.width, ctx.height, 0.0);
        draw_centered_string(
            d,
            font,
            &extension_text(lang, "auth.title", "Microsoft sign-in"),
            ctx.width / 2,
            20,
            TITLE_COLOR,
        );
        let mut y = ctx.height / 4;
        for line in self.body_lines(lang) {
            for wrapped in font.wrap_to_lines(&line, ctx.width - 40) {
                draw_centered_string(d, font, &wrapped, ctx.width / 2, y, LABEL_COLOR);
                y += font.font_height() + 2;
            }
        }
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screens::clipboard::SystemClipboard;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;
    use std::cell::RefCell;
    use std::time::Duration;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn lang() -> Language {
        let mut l = Language::new("en_US");
        for k in [
            "auth.title",
            "auth.signIn",
            "auth.copyCode",
            "auth.openPage",
            "auth.idle",
            "auth.requesting",
            "gui.cancel",
            "auth.error.noXboxAccount.help",
            "auth.error.childAccount.help",
            "auth.error.noEntitlement.help",
            "auth.error.servicesForbidden.help",
            "auth.error.network.help",
            "auth.error.generic.help",
            "auth.error.expired.help",
            "auth.error.declined.help",
            "auth.error.signInAgain.help",
            "auth.error.clientId.help",
        ] {
            l.insert(k, k);
        }
        l.insert("auth.expiresIn", "expires in %s minutes");
        l.insert("auth.success", "Signed in as %s");
        l
    }

    fn prompt() -> DeviceCodePrompt {
        DeviceCodePrompt {
            user_code: "A1B2C3D4".into(),
            verification_uri: "https://microsoft.com/link".into(),
            message: "Go to the page and enter the code".into(),
            expires_in: Duration::from_secs(900),
            interval: Duration::from_secs(5),
        }
    }

    fn screen() -> LoginScreen {
        let mut s = LoginScreen::new();
        s.init(&ctx(), &lang());
        s
    }

    fn click(s: &mut LoginScreen, id: i32, clip: &mut dyn Clipboard) -> Action {
        let b = s.buttons.iter().find(|b| b.id == id).unwrap();
        let (x, y) = (b.rect.x + 1, b.rect.y + 1);
        s.mouse_clicked(x, y, 0, clip)
    }

    #[test]
    fn the_flow_walks_idle_to_requesting_to_awaiting_to_success() {
        let mut s = screen();
        let mut clip = crate::screens::clipboard::NullClipboard;
        assert_eq!(s.state, LoginState::Idle);
        assert!(!s.is_busy());
        assert_eq!(click(&mut s, ids::SIGN_IN, &mut clip), Action::BeginLogin);
        assert_eq!(s.state, LoginState::Requesting);
        assert!(s.is_busy());
        s.set_prompt(prompt());
        assert_eq!(s.state, LoginState::AwaitingUser);
        s.succeed("Notch");
        assert_eq!(s.state, LoginState::Success("Notch".into()));
        assert!(!s.is_busy());
        assert!(s.prompt.is_none());
    }

    #[test]
    fn sign_in_is_disabled_while_busy_and_the_code_buttons_only_work_with_a_prompt() {
        let mut s = screen();
        let mut clip = crate::screens::clipboard::NullClipboard;
        let enabled =
            |s: &LoginScreen, id: i32| s.buttons.iter().find(|b| b.id == id).unwrap().enabled;
        assert!(enabled(&s, ids::SIGN_IN));
        assert!(!enabled(&s, ids::COPY_CODE));
        assert!(!enabled(&s, ids::OPEN_BROWSER));
        click(&mut s, ids::SIGN_IN, &mut clip);
        assert!(!enabled(&s, ids::SIGN_IN));
        s.set_prompt(prompt());
        assert!(enabled(&s, ids::COPY_CODE) && enabled(&s, ids::OPEN_BROWSER));
        assert!(!enabled(&s, ids::SIGN_IN));
        s.succeed("Alex");
        assert!(
            !enabled(&s, ids::SIGN_IN),
            "a completed flow must not start a second overlapping login"
        );
    }

    #[test]
    fn awaiting_state_without_a_prompt_does_not_offer_dead_actions() {
        let mut s = LoginScreen::new();
        s.state = LoginState::AwaitingUser;
        s.init(&ctx(), &lang());
        assert!(s.prompt.is_none());
        assert!(s
            .buttons
            .iter()
            .filter(|button| matches!(button.id, ids::COPY_CODE | ids::OPEN_BROWSER))
            .all(|button| !button.enabled));
    }

    #[test]
    fn copy_puts_the_user_code_on_the_real_clipboard() {
        let mut s = screen();
        s.set_prompt(prompt());
        let backing = RefCell::new(String::new());
        let mut clip = SystemClipboard::new(
            || Some(backing.borrow().clone()),
            |t| {
                *backing.borrow_mut() = t.to_owned();
                true
            },
        );
        assert_eq!(
            click(&mut s, ids::COPY_CODE, &mut clip),
            Action::CopyToClipboard("A1B2C3D4".into())
        );
        assert_eq!(*backing.borrow(), "A1B2C3D4");
    }

    #[test]
    fn open_page_hands_back_microsofts_url() {
        let mut s = screen();
        s.set_prompt(prompt());
        let mut clip = crate::screens::clipboard::NullClipboard;
        assert_eq!(
            click(&mut s, ids::OPEN_BROWSER, &mut clip),
            Action::OpenUrl("https://microsoft.com/link".into())
        );
    }

    #[test]
    fn the_prompt_body_shows_the_code_the_url_and_the_expiry() {
        let mut s = screen();
        s.set_prompt(prompt());
        let lines = s.body_lines(&lang());
        assert_eq!(lines[0], "Go to the page and enter the code");
        assert_eq!(lines[1], "https://microsoft.com/link");
        assert!(lines[2].ends_with("A1B2C3D4"));
        assert!(lines[2].starts_with('\u{a7}'), "the code is highlighted");
        assert_eq!(lines[3], "expires in 15 minutes");
    }

    #[test]
    fn stock_language_without_extension_keys_uses_readable_fallbacks() {
        // 1.8.9's bundled en_US.lang predates Microsoft's device-code flow. An empty table is
        // therefore a useful negative control: every auth.* lookup would leak its raw key if the
        // fallback branch were removed or inverted.
        let mut l = Language::new("en_US");
        l.insert("gui.cancel", "Cancel");
        let mut s = LoginScreen::new();
        s.init(&ctx(), &l);

        let button_text = |id| {
            s.buttons
                .iter()
                .find(|button| button.id == id)
                .unwrap()
                .text
                .clone()
        };
        assert_eq!(button_text(ids::SIGN_IN), "Sign in with Microsoft");
        assert_eq!(button_text(ids::COPY_CODE), "Copy code");
        assert_eq!(button_text(ids::OPEN_BROWSER), "Open page");
        assert_eq!(
            s.body_lines(&l),
            ["Sign in to play on online-mode servers."]
        );

        s.set_prompt(prompt());
        assert_eq!(s.body_lines(&l)[3], "Code expires in 15 minutes");
        s.succeed("Notch");
        assert_eq!(s.body_lines(&l), ["Signed in as Notch"]);

        s.fail(&AuthError::NoEntitlement, &l);
        let lines = s.body_lines(&l);
        assert_eq!(
            lines[1],
            "Use an account that owns Minecraft: Java Edition."
        );
        assert!(
            s.buttons
                .iter()
                .all(|button| !button.text.starts_with("auth.")),
            "no extension translation key should be visible"
        );
    }

    #[test]
    fn resource_pack_can_override_extension_fallbacks() {
        let mut l = Language::new("xx_TEST");
        l.insert("auth.signIn", "Custom sign in");
        l.insert("auth.success", "Welcome, %s!");
        l.insert("auth.error.noEntitlement.help", "Custom ownership advice");
        let mut s = LoginScreen::new();
        s.init(&ctx(), &l);
        assert_eq!(
            s.buttons
                .iter()
                .find(|button| button.id == ids::SIGN_IN)
                .unwrap()
                .text,
            "Custom sign in"
        );
        s.succeed("Alex");
        assert_eq!(s.body_lines(&l), ["Welcome, Alex!"]);
        s.fail(&AuthError::NoEntitlement, &l);
        assert_eq!(s.body_lines(&l)[1], "Custom ownership advice");
    }

    /// Every typed variant §7 S03 promised gets its own advice line rather than a generic one.
    #[test]
    fn each_typed_auth_error_gets_its_own_advice() {
        let l = lang();
        let cases = [
            (AuthError::NoXboxAccount, "auth.error.noXboxAccount.help"),
            (
                AuthError::ChildAccount { redirect: None },
                "auth.error.childAccount.help",
            ),
            (AuthError::NoEntitlement, "auth.error.noEntitlement.help"),
            (
                AuthError::ServicesForbidden {
                    body: "{\"error\":\"FORBIDDEN\"}".to_owned(),
                },
                "auth.error.servicesForbidden.help",
            ),
            (AuthError::DeviceCodeExpired, "auth.error.expired.help"),
            (AuthError::Declined, "auth.error.declined.help"),
            (AuthError::TokenRejected, "auth.error.signInAgain.help"),
            (AuthError::MissingClientId, "auth.error.clientId.help"),
            (AuthError::XboxBanned, "auth.error.generic.help"),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for (err, expected) in cases {
            let (message, advice) = describe_error(&err, &l);
            assert_eq!(advice, expected, "wrong advice for {err:?}");
            assert!(!message.is_empty(), "{err:?} rendered an empty message");
            seen.insert(advice);
        }
        assert!(seen.len() >= 7, "the advice lines are not all distinct");
    }

    #[test]
    fn a_failure_shows_the_message_in_red_with_its_advice_underneath() {
        let mut s = screen();
        s.fail(&AuthError::NoEntitlement, &lang());
        assert!(matches!(s.state, LoginState::Failed(_)));
        let lines = s.body_lines(&lang());
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("\u{a7}c"));
        assert!(lines[0].contains("does not own Minecraft"));
        assert_eq!(lines[1], "auth.error.noEntitlement.help");
        // And sign-in becomes available again so the player can retry.
        assert!(
            s.buttons
                .iter()
                .find(|b| b.id == ids::SIGN_IN)
                .unwrap()
                .enabled
        );
    }

    #[test]
    fn escape_and_cancel_both_leave() {
        let mut s = screen();
        let mut clip = crate::screens::clipboard::NullClipboard;
        assert_eq!(s.key_typed(keys::ESCAPE), Action::Back);
        assert_eq!(s.key_typed(keys::A), Action::None);
        assert_eq!(click(&mut s, ids::CANCEL, &mut clip), Action::Back);
    }

    #[test]
    fn drawing_every_state_produces_geometry_and_never_panics() {
        let mut font = uniform_font();
        let l = lang();
        for state in [
            LoginState::Idle,
            LoginState::Requesting,
            LoginState::AwaitingUser,
            LoginState::Success("Notch".into()),
            LoginState::Failed("boom".into()),
        ] {
            let mut s = screen();
            s.state = state;
            if s.state == LoginState::AwaitingUser {
                s.prompt = Some(prompt());
            }
            let mut d = DrawList::new();
            s.draw(&mut d, &mut font, &ctx(), (0, 0), &l);
            assert!(!d.is_empty());
        }
    }
}
