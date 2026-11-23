//! Getting on and off a server: direct connect, add/edit server, connecting, disconnected.
//!
//! java: `client/gui/GuiScreenServerList.java`, `GuiScreenAddServer.java`,
//! `client/multiplayer/GuiConnecting.java`, `client/gui/GuiDisconnected.java`
//!
//! All four are the same shape — a title, some text, one or two buttons — and all four are
//! `Screen` implementations rather than plain structs, because none of them needs anything the
//! trait does not already hand them.

use mc_assets::Language;

use crate::font::component::ChatComponent;
use crate::font::FontRenderer;
use crate::screen_stack::{Screen, ScreenAction, ScreenCtx};
use crate::screens::common::{draw_world_background, LABEL_COLOR, TITLE_COLOR};
use crate::screens::multiplayer::server_data::{java_idn_to_ascii, ResourceMode, ServerEntry};
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::keys::{self, Clipboard, Modifiers};
use crate::widget::{draw_centered_string, draw_string, Button, TextField};

/// Vanilla's cap on a typed server address. java: `setMaxStringLength(128)`
pub const MAX_ADDRESS_LENGTH: i32 = 128;
/// Vanilla's cap on a typed server name. java: `GuiScreenAddServer` — `setMaxStringLength(32)`
pub const MAX_NAME_LENGTH: i32 = 32;

/// The direct-connect screen: one address box, Join and Cancel.
// java: GuiScreenServerList.java:8
#[derive(Debug)]
pub struct DirectConnectScreen {
    /// The address box.
    pub address: TextField,
    /// The entry being filled in.
    pub entry: ServerEntry,
    buttons: Vec<Button>,
}

impl DirectConnectScreen {
    /// A screen pre-filled with `last_server` (vanilla's `gameSettings.lastServer`).
    #[must_use]
    pub fn new(entry: ServerEntry, last_server: &str, font: &FontRenderer) -> Self {
        let mut address = TextField::new(2, 0, 116, 200, 20);
        address.set_max_length(MAX_ADDRESS_LENGTH);
        address.set_focused(true);
        address.set_text(last_server, font);
        Self {
            address,
            entry,
            buttons: Vec::new(),
        }
    }

    /// Lay out. java: `initGui`
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        self.address.rect.x = w / 2 - 100;
        self.address.rect.y = 116;
        self.buttons = vec![
            Button::new(
                0,
                w / 2 - 100,
                h / 4 + 96 + 12,
                lang.tr("selectServer.select").to_owned(),
            ),
            Button::new(
                1,
                w / 2 - 100,
                h / 4 + 120 + 12,
                lang.tr("gui.cancel").to_owned(),
            ),
        ];
        self.refresh_enabled();
    }

    /// Vanilla's enable rule for the Join button: a non-empty address that splits on `:` into
    /// at least one piece.
    ///
    /// The second half is vacuous — `"x".split(":")` is always length ≥ 1 in Java — so the
    /// condition reduces to "not empty". Reproduced as written rather than simplified, because
    /// the *empty* string is the one case where Java's `split` returns a length-1 array
    /// containing `""` and the length check still passes; only the `length() > 0` test stops
    /// it.
    // java: GuiScreenServerList.java:42
    pub fn refresh_enabled(&mut self) {
        let ok = !self.address.text().is_empty();
        if let Some(b) = self.buttons.first_mut() {
            b.enabled = ok;
        }
    }

    /// Whether Join is currently clickable.
    #[must_use]
    pub fn can_join(&self) -> bool {
        self.buttons.first().is_some_and(|b| b.enabled)
    }

    /// Handle a key press.
    pub fn key_typed(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        clipboard: &mut dyn Clipboard,
        font: &FontRenderer,
    ) -> Action {
        if self.address.key_typed(typed, key, mods, clipboard, font) {
            self.refresh_enabled();
            return Action::None;
        }
        match key {
            keys::ENTER | keys::NUMPAD_ENTER => self.join(),
            keys::ESCAPE => Action::Back,
            _ => Action::None,
        }
    }

    /// Handle a click.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32, font: &FontRenderer) -> Action {
        self.address.mouse_clicked(x, y, button, font);
        for b in &self.buttons {
            if b.mouse_pressed(x, y) {
                return match b.id {
                    0 => self.join(),
                    _ => Action::Back,
                };
            }
        }
        Action::None
    }

    fn join(&mut self) -> Action {
        if !self.can_join() {
            return Action::None;
        }
        self.entry.address = self.address.text();
        Action::Connect(Box::new(self.entry.clone()))
    }

    /// Draw.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
        in_world: bool,
    ) {
        draw_world_background(d, ctx.width, ctx.height, in_world);
        draw_centered_string(
            d,
            font,
            lang.tr("selectServer.direct"),
            ctx.width / 2,
            20,
            TITLE_COLOR,
        );
        draw_string(
            d,
            font,
            lang.tr("addServer.enterIp"),
            ctx.width / 2 - 100,
            100,
            LABEL_COLOR,
        );
        self.address.draw(d, font);
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }
}

/// The add / edit server screen: a name box, an address box, a resource-pack cycle button.
// java: GuiScreenAddServer.java
#[derive(Debug)]
pub struct AddServerScreen {
    /// The entry being built.
    pub entry: ServerEntry,
    /// Name box.
    pub name: TextField,
    /// Address box.
    pub address: TextField,
    /// Whether this is an edit of an existing entry rather than a new one.
    pub editing: bool,
    buttons: Vec<Button>,
}

impl AddServerScreen {
    /// A screen over `entry`.
    #[must_use]
    pub fn new(entry: ServerEntry, editing: bool, font: &FontRenderer) -> Self {
        let mut name = TextField::new(0, 0, 66, 200, 20);
        name.set_max_length(MAX_NAME_LENGTH);
        name.set_focused(true);
        name.set_text(&entry.name, font);
        let mut address = TextField::new(1, 0, 106, 200, 20);
        address.set_max_length(MAX_ADDRESS_LENGTH);
        address.set_text(&entry.address, font);
        // Java installs this predicate after preloading `serverIP`, so an invalid saved address
        // remains visible but every subsequent edit must pass IDN.toASCII on the host portion.
        // `split(":")[0]` is intentional: the port (and any further colon-separated text) is
        // outside this GUI validator's scope and must not trigger DNS or SRV resolution.
        // java: GuiScreenAddServer.java:17-47,81
        address.set_validator(|candidate| {
            if candidate.is_empty() {
                return true;
            }
            let host = candidate.split(':').next().unwrap_or_default();
            java_idn_to_ascii(host).is_ok()
        });
        Self {
            entry,
            name,
            address,
            editing,
            buttons: Vec::new(),
        }
    }

    /// Lay out.
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        self.name.rect.x = w / 2 - 100;
        self.address.rect.x = w / 2 - 100;
        self.buttons = vec![
            Button::new(2, w / 2 - 100, h / 4 + 72, self.resource_label(lang)),
            Button::new(
                0,
                w / 2 - 100,
                h / 4 + 96 + 18,
                lang.tr("addServer.add").to_owned(),
            ),
            Button::new(
                1,
                w / 2 - 100,
                h / 4 + 120 + 18,
                lang.tr("gui.cancel").to_owned(),
            ),
        ];
        self.refresh_enabled();
    }

    fn resource_label(&self, lang: &Language) -> String {
        format!(
            "{}: {}",
            lang.tr("addServer.resourcePack"),
            lang.tr(self.entry.resource_mode.translation_key())
        )
    }

    /// Vanilla's enable rule: both boxes non-empty.
    // java: GuiScreenAddServer.java:70
    pub fn refresh_enabled(&mut self) {
        let ok = !self.address.text().is_empty() && !self.name.text().is_empty();
        if let Some(b) = self.buttons.iter_mut().find(|b| b.id == 0) {
            b.enabled = ok;
        }
    }

    /// Handle a key press.
    pub fn key_typed(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        clipboard: &mut dyn Clipboard,
        font: &FontRenderer,
    ) -> Action {
        // Tab moves between the two boxes, which is vanilla's only keyboard navigation.
        if key == keys::TAB {
            let to_name = !self.name.is_focused();
            self.name.set_focused(to_name);
            self.address.set_focused(!to_name);
            return Action::None;
        }
        let handled = self.name.key_typed(typed, key, mods, clipboard, font)
            | self.address.key_typed(typed, key, mods, clipboard, font);
        if handled {
            self.refresh_enabled();
            return Action::None;
        }
        match key {
            keys::ENTER | keys::NUMPAD_ENTER => self.accept(),
            _ => Action::None,
        }
    }

    /// Handle a click.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        font: &FontRenderer,
        lang: &Language,
    ) -> Action {
        self.name.mouse_clicked(x, y, button, font);
        self.address.mouse_clicked(x, y, button, font);
        let mut cycled = false;
        let mut result = Action::None;
        for b in &self.buttons {
            if b.mouse_pressed(x, y) {
                match b.id {
                    0 => result = self.accept(),
                    1 => result = Action::Back,
                    _ => cycled = true,
                }
                break;
            }
        }
        if cycled {
            self.entry.resource_mode = self.entry.resource_mode.next();
            let label = self.resource_label(lang);
            if let Some(b) = self.buttons.iter_mut().find(|b| b.id == 2) {
                b.text = label;
            }
        }
        result
    }

    /// The finished entry, if both boxes are filled in.
    fn accept(&mut self) -> Action {
        if self.name.text().is_empty() || self.address.text().is_empty() {
            return Action::None;
        }
        self.entry.name = self.name.text();
        self.entry.address = self.address.text();
        Action::AcceptServer(Box::new(self.entry.clone()))
    }

    /// The current resource-pack policy.
    #[must_use]
    pub fn resource_mode(&self) -> ResourceMode {
        self.entry.resource_mode
    }

    /// Draw.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
        in_world: bool,
    ) {
        draw_world_background(d, ctx.width, ctx.height, in_world);
        // GuiScreenAddServer reuses this title for both Add and Edit. In en_US the value is
        // "Edit Server Info"; `selectServer.edit` is only the list button's one-word label.
        draw_centered_string(
            d,
            font,
            lang.tr("addServer.title"),
            ctx.width / 2,
            17,
            TITLE_COLOR,
        );
        draw_string(
            d,
            font,
            lang.tr("addServer.enterName"),
            ctx.width / 2 - 100,
            53,
            LABEL_COLOR,
        );
        draw_string(
            d,
            font,
            lang.tr("addServer.enterIp"),
            ctx.width / 2 - 100,
            94,
            LABEL_COLOR,
        );
        self.name.draw(d, font);
        self.address.draw(d, font);
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }
}

/// How far a connection attempt has got.
// java: GuiConnecting.java:159 — `networkManager == null ? "connect.connecting" : "connect.authorizing"`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectPhase {
    /// Resolving the host and opening the socket.
    #[default]
    Connecting,
    /// Socket open, login handshake in flight.
    Authorizing,
}

impl ConnectPhase {
    /// The translation key vanilla shows for this phase.
    #[must_use]
    pub const fn translation_key(self) -> &'static str {
        match self {
            ConnectPhase::Connecting => "connect.connecting",
            ConnectPhase::Authorizing => "connect.authorizing",
        }
    }
}

/// The menu that opened a connection attempt. java: `GuiConnecting.previousGuiScreen`.
///
/// The driver owns the actual screen values, so the connecting/disconnected screens retain this
/// small discriminator rather than a recursive `Ui` parent. Vanilla's network-server path only
/// needs these two parents: command-line/direct startup comes from the main menu, while the
/// multiplayer list supplies the server-list parent for both saved and direct joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionParent {
    /// `GuiMainMenu` (including a command-line `--server` startup).
    #[default]
    MainMenu,
    /// `GuiMultiplayer`.
    Multiplayer,
}

/// The "Connecting to the server…" screen.
#[derive(Debug, Default)]
pub struct ConnectingScreen {
    /// Where the handshake has got to.
    pub phase: ConnectPhase,
    /// The address being connected to, for the F3 overlay and logs.
    pub address: String,
    /// The screen restored by Cancel or a failed connection.
    parent: ConnectionParent,
    buttons: Vec<Button>,
}

impl ConnectingScreen {
    /// A screen for an in-flight connection.
    #[must_use]
    pub fn new(address: impl Into<String>) -> Self {
        Self::with_parent(address, ConnectionParent::MainMenu)
    }

    /// A screen for an in-flight connection with its vanilla opener retained.
    #[must_use]
    pub fn with_parent(address: impl Into<String>, parent: ConnectionParent) -> Self {
        Self {
            phase: ConnectPhase::Connecting,
            address: address.into(),
            parent,
            buttons: Vec::new(),
        }
    }

    /// The screen that vanilla restores after Cancel or a failed connection.
    #[must_use]
    pub const fn parent(&self) -> ConnectionParent {
        self.parent
    }

    /// Lay out. java: one Cancel button at `height / 4 + 120 + 12`.
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        self.buttons = vec![Button::new(
            0,
            ctx.width / 2 - 100,
            ctx.height / 4 + 120 + 12,
            lang.tr("gui.cancel").to_owned(),
        )];
    }

    /// Handle a click.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        if button == 0 && self.buttons.iter().any(|b| b.mouse_pressed(x, y)) {
            Action::CancelConnect
        } else {
            Action::None
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
        draw_world_background(d, ctx.width, ctx.height, false);
        draw_centered_string(
            d,
            font,
            lang.tr(self.phase.translation_key()),
            ctx.width / 2,
            ctx.height / 2 - 50,
            TITLE_COLOR,
        );
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }
}

impl Screen for ConnectingScreen {
    fn name(&self) -> &'static str {
        "connecting"
    }
    // java: GuiConnecting.keyTyped is empty — Escape does *not* cancel a connection.
    fn key_typed(&mut self, _typed: Option<char>, _key: i32, _mods: Modifiers) -> ScreenAction {
        ScreenAction::None
    }
}

/// The disconnect screen: a reason line and a wrapped server message.
// java: GuiDisconnected.java:8
#[derive(Debug)]
pub struct DisconnectedScreen {
    /// The already-translated reason ("Failed to connect to the server").
    pub reason: String,
    /// The server's message, as a chat component.
    pub message: ChatComponent,
    /// The screen restored by the "Back to title screen" button.
    parent: ConnectionParent,
    formatted_message: String,
    lines: Vec<String>,
    text_height: i32,
    buttons: Vec<Button>,
}

impl DisconnectedScreen {
    /// A screen for a disconnect.
    #[must_use]
    pub fn new(reason: impl Into<String>, message: ChatComponent) -> Self {
        Self::with_parent(reason, message, ConnectionParent::MainMenu)
    }

    /// A disconnect screen retaining the vanilla opener/parent chain.
    #[must_use]
    pub fn with_parent(
        reason: impl Into<String>,
        message: ChatComponent,
        parent: ConnectionParent,
    ) -> Self {
        Self {
            reason: reason.into(),
            message,
            parent,
            formatted_message: String::new(),
            lines: Vec::new(),
            text_height: 0,
            buttons: Vec::new(),
        }
    }

    /// The screen that vanilla returns to after the disconnect button.
    #[must_use]
    pub const fn parent(&self) -> ConnectionParent {
        self.parent
    }

    /// Lay out: wrap the message to `width - 50` and put the button below it.
    // java: GuiDisconnected.java:35
    pub fn init(&mut self, ctx: &ScreenCtx, font: &FontRenderer, lang: &Language) {
        self.formatted_message = self.message.formatted_text(lang);
        self.buttons = vec![Button::new(0, 0, 0, lang.tr("gui.toMenu").to_owned())];
        self.relayout(ctx, font);
    }

    /// Recompute the viewport-dependent part of `initGui`.
    ///
    /// Java calls `initGui` again whenever the window is resized.  The driver may retain this
    /// screen across a resize, so keep the wrap width, vertical stack, and button hitbox tied to
    /// the live [`ScreenCtx`] even when no language reload occurred.
    fn relayout(&mut self, ctx: &ScreenCtx, font: &FontRenderer) {
        self.lines = font.wrap_to_lines(&self.formatted_message, ctx.width - 50);
        self.text_height = self.lines.len() as i32 * font.font_height();
        if let Some(button) = self.buttons.first_mut() {
            button.rect.x = ctx.width / 2 - 100;
            button.rect.y = ctx.height / 2 + self.text_height / 2 + font.font_height();
        }
    }

    /// The wrapped message lines.
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Handle a click.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        if button == 0 && self.buttons.iter().any(|b| b.mouse_pressed(x, y)) {
            Action::Back
        } else {
            Action::None
        }
    }

    /// Draw.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
    ) {
        self.relayout(ctx, font);
        draw_world_background(d, ctx.width, ctx.height, false);
        draw_centered_string(
            d,
            font,
            &self.reason.clone(),
            ctx.width / 2,
            ctx.height / 2 - self.text_height / 2 - font.font_height() * 2,
            11_184_810,
        );
        let mut y = ctx.height / 2 - self.text_height / 2;
        for line in self.lines.clone() {
            draw_centered_string(d, font, &line, ctx.width / 2, y, TITLE_COLOR);
            y += font.font_height();
        }
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }
}

impl Screen for DisconnectedScreen {
    fn name(&self) -> &'static str {
        "disconnected"
    }
    fn opaque(&self) -> bool {
        true
    }
    // java: GuiDisconnected.keyTyped is empty — you must click the button.
    fn key_typed(&mut self, _typed: Option<char>, _key: i32, _mods: Modifiers) -> ScreenAction {
        ScreenAction::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screens::clipboard::NullClipboard;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn lang() -> Language {
        let mut l = Language::new("en_US");
        for k in [
            "selectServer.select",
            "selectServer.direct",
            "selectServer.edit",
            "gui.cancel",
            "gui.toMenu",
            "addServer.enterIp",
            "addServer.enterName",
            "addServer.title",
            "addServer.add",
            "addServer.resourcePack",
            "addServer.resourcePack.prompt",
            "addServer.resourcePack.enabled",
            "addServer.resourcePack.disabled",
            "connect.connecting",
            "connect.authorizing",
        ] {
            l.insert(k, k);
        }
        l
    }

    #[test]
    fn direct_connect_disables_join_until_something_is_typed() {
        let font = uniform_font();
        let mut s = DirectConnectScreen::new(ServerEntry::new("x", ""), "", &font);
        s.init(&ctx(), &lang());
        assert!(!s.can_join());
        let mut clip = NullClipboard;
        s.key_typed(Some('h'), 0, Modifiers::NONE, &mut clip, &font);
        assert!(s.can_join());
        assert_eq!(s.address.text(), "h");
    }

    #[test]
    fn direct_connect_prefills_the_last_server_and_returns_it_on_join() {
        let font = uniform_font();
        let mut s = DirectConnectScreen::new(
            ServerEntry::new("Minecraft Server", ""),
            "a.example:25566",
            &font,
        );
        s.init(&ctx(), &lang());
        assert!(s.can_join());
        let mut clip = NullClipboard;
        match s.key_typed(None, keys::ENTER, Modifiers::NONE, &mut clip, &font) {
            Action::Connect(e) => {
                assert_eq!(e.address, "a.example:25566");
                assert_eq!(e.host_and_port(), ("a.example", 25566));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn direct_connect_puts_its_box_and_buttons_where_vanilla_does() {
        let font = uniform_font();
        let mut s = DirectConnectScreen::new(ServerEntry::new("x", ""), "", &font);
        s.init(&ctx(), &lang());
        assert_eq!((s.address.rect.x, s.address.rect.y), (854 / 2 - 100, 116));
        assert_eq!(s.address.rect.width, 200);
        assert_eq!(s.buttons[0].rect.y, 480 / 4 + 96 + 12);
        assert_eq!(s.buttons[1].rect.y, 480 / 4 + 120 + 12);
    }

    #[test]
    fn add_server_needs_both_boxes_and_cycles_the_resource_mode() {
        let font = uniform_font();
        let l = lang();
        let mut s = AddServerScreen::new(ServerEntry::new("", ""), false, &font);
        s.init(&ctx(), &l);
        assert_eq!(s.accept(), Action::None, "both boxes empty");
        s.name.set_text("Home", &font);
        s.address.set_text("127.0.0.1", &font);
        s.refresh_enabled();
        match s.accept() {
            Action::AcceptServer(e) => {
                assert_eq!(e.name, "Home");
                assert_eq!(e.address, "127.0.0.1");
            }
            other => panic!("{other:?}"),
        }
        // The cycle button is at height/4 + 72.
        let b = s.buttons.iter().find(|b| b.id == 2).unwrap();
        let (bx, by) = (b.rect.x + 5, b.rect.y + 5);
        assert_eq!(s.resource_mode(), ResourceMode::Prompt);
        s.mouse_clicked(bx, by, 0, &font, &l);
        assert_eq!(s.resource_mode(), ResourceMode::Enabled);
        s.mouse_clicked(bx, by, 0, &font, &l);
        assert_eq!(s.resource_mode(), ResourceMode::Disabled);
        s.mouse_clicked(bx, by, 0, &font, &l);
        assert_eq!(s.resource_mode(), ResourceMode::Prompt);
    }

    #[test]
    fn add_server_tab_moves_between_the_two_boxes() {
        let font = uniform_font();
        let mut s = AddServerScreen::new(ServerEntry::new("n", "a"), true, &font);
        s.init(&ctx(), &lang());
        assert!(s.name.is_focused() && !s.address.is_focused());
        let mut clip = NullClipboard;
        s.key_typed(None, keys::TAB, Modifiers::NONE, &mut clip, &font);
        assert!(!s.name.is_focused() && s.address.is_focused());
        s.key_typed(None, keys::TAB, Modifiers::NONE, &mut clip, &font);
        assert!(s.name.is_focused());
    }

    #[test]
    fn add_server_address_validator_matches_java_idn_host_split() {
        let font = uniform_font();
        let mut direct = DirectConnectScreen::new(ServerEntry::new("n", ""), "", &font);
        direct.address.set_text("a..b", &font);
        assert_eq!(
            direct.address.text(),
            "a..b",
            "GuiScreenServerList leaves Direct Connect unrestricted"
        );

        let saved_invalid = AddServerScreen::new(ServerEntry::new("n", "a..b"), true, &font);
        assert_eq!(
            saved_invalid.address.text(),
            "a..b",
            "the validator is installed after Java's initial setText"
        );
        let mut s = AddServerScreen::new(ServerEntry::new("n", "FOO.Example"), true, &font);

        // Java's ASCII-only IDN label path preserves case, and the validator must not rewrite it.
        assert_eq!(s.address.text(), "FOO.Example");
        s.address.set_text("FOO.Example", &font);
        assert_eq!(s.address.text(), "FOO.Example");
        s.address.set_text("bücher.Example", &font);
        assert_eq!(s.address.text(), "bücher.Example");

        // A malformed/empty interior label is rejected, while the empty whole field is accepted.
        s.address.set_text("a..b", &font);
        assert_eq!(s.address.text(), "bücher.Example");
        s.address.set_text("", &font);
        assert!(s.address.is_empty());

        // Java checks only split(":")[0]: an empty host and arbitrary port text both pass.
        s.address.set_text(":not-a-port", &font);
        assert_eq!(s.address.text(), ":not-a-port");
        s.address.set_text("bücher.Example:not-a-port", &font);
        assert_eq!(s.address.text(), "bücher.Example:not-a-port");
    }

    #[test]
    fn add_server_address_rejection_preserves_edit_cursor_and_text() {
        let font = uniform_font();
        let mut s = AddServerScreen::new(ServerEntry::new("n", "valid.example"), true, &font);
        s.address.set_focused(true);
        s.address.set_cursor_position(5, &font);
        let before = (
            s.address.text(),
            s.address.cursor_position(),
            s.address.selection_end(),
        );

        // Inserting a second dot makes an invalid host; TextField's validator path must reject
        // the candidate atomically, just as GuiTextField.writeText does.
        s.address.write_text(".", &font);
        assert_eq!(
            (
                s.address.text(),
                s.address.cursor_position(),
                s.address.selection_end()
            ),
            before
        );
    }

    #[test]
    fn add_server_ignores_escape_and_add_edit_share_the_same_title() {
        let font = uniform_font();
        let l = lang();
        let entry = ServerEntry::new("n", "a");
        let mut add = AddServerScreen::new(entry.clone(), false, &font);
        let mut edit = AddServerScreen::new(entry, true, &font);
        add.init(&ctx(), &l);
        edit.init(&ctx(), &l);
        let mut clip = NullClipboard;
        assert_eq!(
            add.key_typed(None, keys::ESCAPE, Modifiers::NONE, &mut clip, &font),
            Action::None
        );
        let mut add_draw = DrawList::new();
        let mut edit_draw = DrawList::new();
        let mut add_font = uniform_font();
        let mut edit_font = uniform_font();
        add.draw(&mut add_draw, &mut add_font, &ctx(), (0, 0), &l, false);
        edit.draw(&mut edit_draw, &mut edit_font, &ctx(), (0, 0), &l, false);
        assert_eq!(add_draw, edit_draw);
    }

    #[test]
    fn editing_an_entry_keeps_vanillas_stale_ping_state() {
        use crate::screens::multiplayer::server_data::PingState;
        let font = uniform_font();
        let mut entry = ServerEntry::new("n", "a");
        entry.state = PingState::Ok;
        entry.ping_ms = 12;
        let mut s = AddServerScreen::new(entry, true, &font);
        s.init(&ctx(), &lang());
        match s.accept() {
            Action::AcceptServer(e) => {
                assert_eq!(e.state, PingState::Ok);
                assert_eq!(e.ping_ms, 12);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_connecting_screen_names_its_phase_and_ignores_escape() {
        let mut s = ConnectingScreen::new("host");
        s.init(&ctx(), &lang());
        assert_eq!(s.phase.translation_key(), "connect.connecting");
        s.phase = ConnectPhase::Authorizing;
        assert_eq!(s.phase.translation_key(), "connect.authorizing");
        assert_eq!(
            s.key_typed(None, keys::ESCAPE, Modifiers::NONE),
            ScreenAction::None,
            "vanilla's GuiConnecting.keyTyped is empty"
        );
        let b = &s.buttons[0];
        assert_eq!(
            s.mouse_clicked(b.rect.x + 1, b.rect.y + 1, 0),
            Action::CancelConnect
        );
    }

    #[test]
    fn the_disconnect_screen_wraps_the_message_and_places_the_button_under_it() {
        let font = uniform_font();
        let mut s =
            DisconnectedScreen::new("Failed to connect", ChatComponent::text("word ".repeat(80)));
        s.init(&ctx(), &font, &lang());
        assert!(s.lines().len() > 1, "the message must wrap");
        let expected =
            480 / 2 + (s.lines().len() as i32 * font.font_height()) / 2 + font.font_height();
        assert_eq!(s.buttons[0].rect.y, expected);
        assert_eq!(
            s.mouse_clicked(s.buttons[0].rect.x + 1, s.buttons[0].rect.y + 1, 0),
            Action::Back
        );
        assert_eq!(
            s.key_typed(None, keys::ESCAPE, Modifiers::NONE),
            ScreenAction::None
        );
        assert!(s.opaque(), "there is no world left to draw behind it");
    }

    #[test]
    fn drawing_a_resized_disconnect_screen_rewraps_and_moves_its_hitbox() {
        let mut font = uniform_font();
        let mut s =
            DisconnectedScreen::new("Connection Lost", ChatComponent::text("word ".repeat(80)));
        s.init(&ctx(), &font, &lang());
        let initial_line_count = s.lines().len();
        assert_eq!(s.buttons[0].rect.x, 854 / 2 - 100);

        let resized = ScreenCtx::new(ScaledResolution::new(320, 240, 1, false), 0.0);
        s.draw(&mut DrawList::new(), &mut font, &resized, (0, 0));

        assert!(
            s.lines().len() > initial_line_count,
            "a narrower viewport must wrap the same component onto more lines"
        );
        assert_eq!(s.buttons[0].rect.x, 320 / 2 - 100);
        assert_eq!(
            s.buttons[0].rect.y,
            240 / 2 + (s.lines().len() as i32 * font.font_height()) / 2 + font.font_height()
        );
        assert_eq!(
            s.mouse_clicked(s.buttons[0].rect.x + 1, s.buttons[0].rect.y + 1, 0),
            Action::Back,
            "the live hitbox must move with the rendered button"
        );
    }

    #[test]
    fn a_translated_disconnect_component_is_wrapped_after_translation() {
        let font = uniform_font();
        let mut l = lang();
        l.insert("gui.toMenu", "Back to title screen");
        l.insert("disconnect.timeout", "Timed out");
        let mut s = DisconnectedScreen::new(
            "Connection Lost",
            ChatComponent::translate("disconnect.timeout", vec![]),
        );
        s.init(&ctx(), &font, &l);
        assert_eq!(s.lines(), ["Timed out§r"]);
    }
}
