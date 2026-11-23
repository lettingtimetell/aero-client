//! The spectator hotbar and its two-press command menu.
//!
//! java: `client/gui/GuiSpectator.java` and `client/gui/spectator/**`
//!
//! This module deliberately owns only presentation and menu mutation. The application supplies
//! the current player/team views and consumes [`SpectatorAction::Spectate`], keeping C18 packet
//! construction out of `mc-ui`.

use std::cmp::Ordering;

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::widget::{DrawList, UiTexture};

/// `GuiSpectator.field_175269_a`.
pub const SPECTATOR_WIDGETS_TEXTURE: &str = "textures/gui/spectator_widgets.png";
/// `GuiSpectator.field_175267_f`.
pub const WIDGETS_TEXTURE: &str = "textures/gui/widgets.png";

const ROOT_CAPTION: &str = "Press a key to select a command, and again to use it.";
const PLAYER_CAPTION: &str = "Select a player to teleport to";
const TEAM_CAPTION: &str = "Select a team to teleport to";

/// An icon drawn in one of the spectator bar's 16-pixel cells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectatorIcon {
    /// A 16x16 region of `spectator_widgets.png`.
    Widget {
        /// Source x in the 256x256 texture.
        u: i32,
        /// Source y in the 256x256 texture.
        v: i32,
    },
    /// The face and hat layers of a 64x64 player skin.
    PlayerHead {
        /// The dynamically uploaded skin texture.
        texture: UiTexture,
        /// Optional 14x14 team-colour swatch behind the head.
        team_color: Option<u32>,
    },
}

/// A menu command. Commands are activated by choosing the same slot twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectatorCommand {
    /// Replace the current view with a submenu.
    Open(Box<SpectatorView>),
    /// Emit a C18 spectate request for these UUID bytes.
    Spectate([u8; 16]),
}

/// One selectable object in a [`SpectatorView`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpectatorEntry {
    /// Text drawn above the bar while this object is selected.
    pub name: String,
    /// Artwork drawn in its bar cell.
    pub icon: SpectatorIcon,
    /// Vanilla dims disabled entries to 25 percent and will not activate them.
    pub enabled: bool,
    /// Mutation or application action performed by the second selection.
    pub command: SpectatorCommand,
}

impl SpectatorEntry {
    /// A player head that asks the application to emit C18 when activated.
    #[must_use]
    pub fn player(name: impl Into<String>, uuid: [u8; 16], texture: UiTexture) -> Self {
        Self {
            name: name.into(),
            icon: SpectatorIcon::PlayerHead {
                texture,
                team_color: None,
            },
            enabled: true,
            command: SpectatorCommand::Spectate(uuid),
        }
    }

    /// The root menu's "Teleport to player" command.
    #[must_use]
    pub fn players(entries: Vec<SpectatorEntry>) -> Self {
        Self {
            name: "Teleport to player".into(),
            icon: SpectatorIcon::Widget { u: 0, v: 0 },
            enabled: !entries.is_empty(),
            command: SpectatorCommand::Open(Box::new(SpectatorView::players(entries))),
        }
    }

    /// The root menu's "Teleport to team member" command.
    #[must_use]
    pub fn teams(entries: Vec<SpectatorEntry>) -> Self {
        let enabled = entries.iter().any(|entry| entry.enabled);
        Self {
            name: "Teleport to team member".into(),
            icon: SpectatorIcon::Widget { u: 16, v: 0 },
            enabled,
            command: SpectatorCommand::Open(Box::new(SpectatorView::teams(entries))),
        }
    }

    /// A team object that opens its non-spectator member list.
    #[must_use]
    pub fn team(
        name: impl Into<String>,
        members: Vec<SpectatorEntry>,
        texture: UiTexture,
        team_color: Option<u32>,
    ) -> Self {
        Self {
            name: name.into(),
            icon: SpectatorIcon::PlayerHead {
                texture,
                team_color,
            },
            enabled: !members.is_empty(),
            command: SpectatorCommand::Open(Box::new(SpectatorView::players(members))),
        }
    }
}

/// One spectator-menu view and the caption shown when none of its entries is selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpectatorView {
    caption: String,
    entries: Vec<SpectatorEntry>,
}

impl SpectatorView {
    /// Build an arbitrary view. This is also the seam for server/plugin-specific categories.
    #[must_use]
    pub fn new(caption: impl Into<String>, entries: Vec<SpectatorEntry>) -> Self {
        Self {
            caption: caption.into(),
            entries,
        }
    }

    /// Build vanilla's player-selection view.
    #[must_use]
    pub fn players(mut entries: Vec<SpectatorEntry>) -> Self {
        // Java UUID.compareTo compares the signed most-significant long, then the signed least.
        entries.sort_by(|a, b| match (&a.command, &b.command) {
            (SpectatorCommand::Spectate(a), SpectatorCommand::Spectate(b)) => java_uuid_cmp(a, b),
            _ => Ordering::Equal,
        });
        Self::new(PLAYER_CAPTION, entries)
    }

    /// Build vanilla's team-selection view.
    #[must_use]
    pub fn teams(entries: Vec<SpectatorEntry>) -> Self {
        Self::new(TEAM_CAPTION, entries)
    }

    /// The fallback caption for this view.
    #[must_use]
    pub fn caption(&self) -> &str {
        &self.caption
    }

    /// Objects in this view, before page controls are inserted.
    #[must_use]
    pub fn entries(&self) -> &[SpectatorEntry] {
        &self.entries
    }
}

/// An application-facing effect produced by the menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpectatorAction {
    /// No external action was produced.
    #[default]
    None,
    /// Send `C18PacketSpectate` for these UUID bytes.
    Spectate([u8; 16]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell<'a> {
    Empty,
    Entry(&'a SpectatorEntry),
    Previous,
    Next(bool),
    Close,
}

impl Cell<'_> {
    fn enabled(self) -> bool {
        match self {
            Cell::Entry(entry) => entry.enabled,
            Cell::Previous | Cell::Close => true,
            Cell::Next(enabled) => enabled,
            Cell::Empty => false,
        }
    }

    fn empty(self) -> bool {
        matches!(self, Cell::Empty)
    }

    fn name(self) -> Option<&'static str> {
        match self {
            Cell::Previous => Some("Previous Page"),
            Cell::Next(_) => Some("Next Page"),
            Cell::Close => Some("Close menu"),
            _ => None,
        }
    }
}

/// The stateful spectator bar.
///
/// It is not a modal screen: vanilla draws this over the world and keeps mouse-look active.
#[derive(Debug, Clone)]
pub struct SpectatorHud {
    root: SpectatorView,
    view: SpectatorView,
    selected: i32,
    page: i32,
    last_interaction_ms: i64,
    open: bool,
}

impl SpectatorHud {
    /// Build vanilla's root view from the already-filtered player and team objects.
    #[must_use]
    pub fn new(players: Vec<SpectatorEntry>, teams: Vec<SpectatorEntry>) -> Self {
        let root = SpectatorView::new(
            ROOT_CAPTION,
            vec![
                SpectatorEntry::players(players),
                SpectatorEntry::teams(teams),
            ],
        );
        Self {
            root: root.clone(),
            view: root,
            selected: -1,
            page: 0,
            last_interaction_ms: 0,
            open: false,
        }
    }

    /// Whether a menu currently exists. An elapsed menu closes lazily during [`draw`](Self::draw).
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Close the bar and clear its fade clock, as vanilla's menu recipient does.
    pub fn dismiss(&mut self) {
        self.close();
    }

    /// The selected hotbar index, or `-1` when no object is selected.
    #[must_use]
    pub fn selected_slot(&self) -> i32 {
        self.selected
    }

    /// Current page index. Page zero has seven content cells; later pages have six.
    #[must_use]
    pub fn page(&self) -> i32 {
        self.page
    }

    /// Current view caption.
    #[must_use]
    pub fn caption(&self) -> &str {
        self.view.caption()
    }

    /// Replace the root data, for example after a player-list update.
    ///
    /// Vanilla reconstructs category objects whenever the menu is opened; this explicit seam lets
    /// the driver do the same without letting the UI borrow client state.
    pub fn replace_entries(&mut self, players: Vec<SpectatorEntry>, teams: Vec<SpectatorEntry>) {
        let was_root = self.view.caption == self.root.caption;
        self.root = SpectatorView::new(
            ROOT_CAPTION,
            vec![
                SpectatorEntry::players(players),
                SpectatorEntry::teams(teams),
            ],
        );
        if !self.open {
            self.view = self.root.clone();
            self.selected = -1;
            self.page = 0;
        } else if was_root {
            // The driver refreshes player skins/data while the overlay is visible. Replacing the
            // root must not turn the first hotbar press back into an unselected menu before the
            // second press arrives; Java's menu object keeps its selected slot/page until the
            // recipient closes it. The new root data is still adopted so removals cannot leave
            // an activatable stale object.
            self.view = self.root.clone();
        }
    }

    /// Route a hotbar key. First use opens the menu; later uses select/activate that slot.
    pub fn select_slot(&mut self, slot: i32, now_ms: i64) -> SpectatorAction {
        self.last_interaction_ms = now_ms;
        if !self.open {
            self.open_root();
            return SpectatorAction::None;
        }
        self.select_or_activate(slot)
    }

    /// Route middle click: open the menu, or activate the currently selected object.
    pub fn activate_selected(&mut self, now_ms: i64) -> SpectatorAction {
        self.last_interaction_ms = now_ms;
        if !self.open {
            self.open_root();
            return SpectatorAction::None;
        }
        if self.selected == -1 {
            SpectatorAction::None
        } else {
            self.select_or_activate(self.selected)
        }
    }

    /// Move selection through enabled, non-empty cells. `direction` must be `-1` or `1`.
    pub fn scroll(&mut self, direction: i32, now_ms: i64) {
        if !self.open || !matches!(direction, -1 | 1) {
            return;
        }
        let mut slot = self.selected + direction;
        while (0..=8).contains(&slot) {
            let cell = self.cell(slot);
            if !cell.empty() && cell.enabled() {
                self.selected = slot;
                self.last_interaction_ms = now_ms;
                return;
            }
            slot += direction;
        }
    }

    /// Fade value used for both alpha and the bar's slide-down transform.
    ///
    /// It remains 1 for three seconds, then linearly reaches 0 at five seconds.
    #[must_use]
    pub fn fade(&self, now_ms: i64) -> f32 {
        if !self.open {
            return 0.0;
        }
        ((self.last_interaction_ms - now_ms + 5_000) as f32 / 2_000.0).clamp(0.0, 1.0)
    }

    /// Selected entry name, or the view caption if selection is empty.
    #[must_use]
    pub fn selected_text(&self) -> Option<&str> {
        if !self.open {
            return None;
        }
        match self.cell(self.selected) {
            Cell::Entry(entry) => Some(&entry.name),
            control => control.name().or(Some(self.view.caption())),
        }
    }

    /// Draw the bar and caption. This also performs vanilla's lazy five-second close.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        hotbar_labels: &[String; 9],
        now_ms: i64,
    ) {
        if !self.open {
            return;
        }
        let fade = self.fade(now_ms);
        if fade <= 0.0 {
            self.close();
            return;
        }
        let alpha = (fade * 255.0) as u32;
        let tint = alpha << 24 | 0x00FF_FFFF;
        let center = ctx.width / 2;
        let y = ctx.height as f32 - 22.0 * fade;
        let widgets = UiTexture::gui(WIDGETS_TEXTURE);
        draw_texture(
            d,
            &widgets,
            (center - 91) as f32,
            y,
            0.0,
            0.0,
            182.0,
            22.0,
            256.0,
            256.0,
            tint,
        );
        if self.selected >= 0 {
            draw_texture(
                d,
                &widgets,
                (center - 92 + self.selected * 20) as f32,
                y - 1.0,
                0.0,
                22.0,
                24.0,
                22.0,
                256.0,
                256.0,
                tint,
            );
        }

        for slot in 0..9 {
            let x = center - 90 + slot * 20 + 2;
            self.draw_cell(d, self.cell(slot), x as f32, y + 3.0, alpha);
            let cell = self.cell(slot);
            if alpha > 3 && cell.enabled() && !cell.empty() {
                let label = &hotbar_labels[slot as usize];
                let label_x = x + 17 - font.string_width(label);
                font.draw_string_with_shadow(d, label, label_x as f32, y + 9.0, tint as i32);
            }
        }

        if alpha > 3 {
            if let Some(text) = self.selected_text() {
                let x = (ctx.width - font.string_width(text)) / 2;
                font.draw_string_with_shadow(
                    d,
                    text,
                    x as f32,
                    (ctx.height - 35) as f32,
                    tint as i32,
                );
            }
        }
    }

    fn open_root(&mut self) {
        self.view = self.root.clone();
        self.selected = -1;
        self.page = 0;
        self.open = true;
    }

    fn close(&mut self) {
        self.open = false;
        self.last_interaction_ms = 0;
    }

    fn cell(&self, slot: i32) -> Cell<'_> {
        if !(0..=8).contains(&slot) {
            return Cell::Empty;
        }
        let index = slot + self.page * 6;
        if self.page > 0 && slot == 0 {
            Cell::Previous
        } else if slot == 7 {
            Cell::Next(index < self.view.entries.len() as i32)
        } else if slot == 8 {
            Cell::Close
        } else {
            self.view
                .entries
                .get(index as usize)
                .map(Cell::Entry)
                .unwrap_or(Cell::Empty)
        }
    }

    fn select_or_activate(&mut self, slot: i32) -> SpectatorAction {
        let cell = self.cell(slot);
        if cell.empty() {
            return SpectatorAction::None;
        }
        if self.selected != slot || !cell.enabled() {
            self.selected = slot;
            return SpectatorAction::None;
        }

        match cell {
            Cell::Entry(entry) => match &entry.command {
                SpectatorCommand::Open(view) => {
                    self.view = (**view).clone();
                    self.selected = -1;
                    self.page = 0;
                    SpectatorAction::None
                }
                SpectatorCommand::Spectate(uuid) => SpectatorAction::Spectate(*uuid),
            },
            Cell::Previous => {
                // The source decompile says assignment. The 1.8.9 bytecode is iadd + putfield.
                self.page -= 1;
                // Java leaves the activated navigation control selected. This means a repeated
                // previous-page key immediately pages again, while ordinary entry selection
                // still follows the normal two-press path.
                self.selected = 0;
                SpectatorAction::None
            }
            Cell::Next(true) => {
                self.page += 1;
                // Keep slot seven selected just like SpectatorMenu.MoveMenuObject. A held/repeated
                // next-page input therefore advances again without an intervening reselection.
                self.selected = 7;
                SpectatorAction::None
            }
            Cell::Close => {
                self.close();
                SpectatorAction::None
            }
            Cell::Next(false) | Cell::Empty => SpectatorAction::None,
        }
    }

    fn draw_cell(&self, d: &mut DrawList, cell: Cell<'_>, x: f32, y: f32, alpha: u32) {
        if cell.empty() {
            return;
        }
        let brightness = if cell.enabled() { 1.0 } else { 0.25 };
        let channel = (brightness * 255.0) as u32;
        let tint = alpha << 24 | channel << 16 | channel << 8 | channel;
        let spectator = UiTexture::gui(SPECTATOR_WIDGETS_TEXTURE);
        match cell {
            Cell::Previous => draw_texture(
                d, &spectator, x, y, 144.0, 0.0, 16.0, 16.0, 256.0, 256.0, tint,
            ),
            Cell::Next(_) => draw_texture(
                d, &spectator, x, y, 160.0, 0.0, 16.0, 16.0, 256.0, 256.0, tint,
            ),
            Cell::Close => draw_texture(
                d, &spectator, x, y, 128.0, 0.0, 16.0, 16.0, 256.0, 256.0, tint,
            ),
            Cell::Entry(entry) => match &entry.icon {
                SpectatorIcon::Widget { u, v } => draw_texture(
                    d, &spectator, x, y, *u as f32, *v as f32, 16.0, 16.0, 256.0, 256.0, tint,
                ),
                SpectatorIcon::PlayerHead {
                    texture,
                    team_color,
                } => {
                    if let Some(rgb) = team_color {
                        let r = (((rgb >> 16) & 255) as f32 * brightness) as u32;
                        let g = (((rgb >> 8) & 255) as f32 * brightness) as u32;
                        let b = ((rgb & 255) as f32 * brightness) as u32;
                        d.quad_rect(
                            &UiTexture::Solid,
                            (x + 1.0, y + 1.0, x + 15.0, y + 15.0),
                            (0.0, 0.0, 1.0, 1.0),
                            alpha << 24 | r << 16 | g << 8 | b,
                        );
                    }
                    draw_skin_layer(d, texture, x + 2.0, y + 2.0, 8.0, tint);
                    draw_skin_layer(d, texture, x + 2.0, y + 2.0, 40.0, tint);
                }
            },
            Cell::Empty => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_texture(
    d: &mut DrawList,
    texture: &UiTexture,
    x: f32,
    y: f32,
    u: f32,
    v: f32,
    width: f32,
    height: f32,
    texture_width: f32,
    texture_height: f32,
    tint: u32,
) {
    d.quad_rect(
        texture,
        (x, y, x + width, y + height),
        (
            u / texture_width,
            v / texture_height,
            (u + width) / texture_width,
            (v + height) / texture_height,
        ),
        tint,
    );
}

fn draw_skin_layer(d: &mut DrawList, texture: &UiTexture, x: f32, y: f32, u: f32, tint: u32) {
    d.quad_rect(
        texture,
        (x, y, x + 12.0, y + 12.0),
        (u / 64.0, 8.0 / 64.0, (u + 8.0) / 64.0, 16.0 / 64.0),
        tint,
    );
}

/// Compare UUIDs like Java 8's `UUID.compareTo`, with signed 64-bit halves.
#[must_use]
pub fn java_uuid_cmp(a: &[u8; 16], b: &[u8; 16]) -> Ordering {
    let a_most = i64::from_be_bytes(a[..8].try_into().expect("eight-byte UUID half"));
    let b_most = i64::from_be_bytes(b[..8].try_into().expect("eight-byte UUID half"));
    a_most.cmp(&b_most).then_with(|| {
        let a_least = i64::from_be_bytes(a[8..].try_into().expect("eight-byte UUID half"));
        let b_least = i64::from_be_bytes(b[8..].try_into().expect("eight-byte UUID half"));
        a_least.cmp(&b_least)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn player(n: u8) -> SpectatorEntry {
        SpectatorEntry::player(
            format!("player {n}"),
            [n; 16],
            UiTexture::gui("skins/test.png"),
        )
    }

    fn labels() -> [String; 9] {
        std::array::from_fn(|i| (i + 1).to_string())
    }

    #[test]
    fn hotbar_key_opens_then_selects_then_activates() {
        let uuid = [7; 16];
        let mut hud = SpectatorHud::new(vec![player(7)], Vec::new());
        assert_eq!(hud.select_slot(0, 100), SpectatorAction::None);
        assert!(hud.is_open());
        assert_eq!(hud.selected_slot(), -1);
        assert_eq!(hud.select_slot(0, 110), SpectatorAction::None);
        assert_eq!(hud.selected_slot(), 0);
        // Root slot zero opens the player view on the second press.
        assert_eq!(hud.select_slot(0, 120), SpectatorAction::None);
        assert_eq!(hud.caption(), PLAYER_CAPTION);
        assert_eq!(hud.select_slot(0, 130), SpectatorAction::None);
        assert_eq!(hud.select_slot(0, 140), SpectatorAction::Spectate(uuid));
    }

    #[test]
    fn refreshing_an_open_root_preserves_the_two_press_selection() {
        let uuid = [7; 16];
        let mut hud = SpectatorHud::new(vec![player(7)], Vec::new());
        assert_eq!(hud.select_slot(0, 100), SpectatorAction::None);
        assert_eq!(hud.select_slot(0, 110), SpectatorAction::None);
        assert_eq!(hud.selected_slot(), 0);

        // `Client::build_ui` and the input path both replace the live entry payload. This used
        // to reset `selected` to -1, making a root category impossible to activate reliably.
        hud.replace_entries(vec![player(7)], Vec::new());
        assert!(hud.is_open());
        assert_eq!(hud.selected_slot(), 0);
        assert_eq!(hud.page(), 0);
        assert_eq!(hud.select_slot(0, 120), SpectatorAction::None);
        assert_eq!(hud.caption(), PLAYER_CAPTION);

        assert_eq!(hud.select_slot(0, 130), SpectatorAction::None);
        assert_eq!(hud.select_slot(0, 140), SpectatorAction::Spectate(uuid));
    }

    #[test]
    fn disabled_object_can_be_selected_but_not_activated() {
        let mut hud = SpectatorHud::new(Vec::new(), Vec::new());
        hud.select_slot(0, 0);
        hud.select_slot(0, 1);
        assert_eq!(hud.selected_slot(), 0);
        assert_eq!(hud.select_slot(0, 2), SpectatorAction::None);
        assert_eq!(hud.caption(), ROOT_CAPTION);
    }

    #[test]
    fn page_controls_use_addition_and_preserve_vanillas_seven_then_six_layout() {
        let mut hud = SpectatorHud::new((0..20).map(player).collect(), Vec::new());
        hud.select_slot(0, 0);
        hud.select_slot(0, 1);
        hud.select_slot(0, 2); // open player view
        hud.select_slot(7, 3);
        hud.select_slot(7, 4);
        assert_eq!(hud.page(), 1);
        assert_eq!(hud.selected_slot(), 7);
        hud.select_slot(7, 5);
        // Java retains the next-page control as selected, so a repeated activation advances
        // again without requiring a separate reselection press.
        assert_eq!(hud.page(), 2, "bytecode is += 1, not decompiler's = 1");
        assert_eq!(hud.selected_slot(), 7);
        hud.select_slot(0, 7);
        hud.select_slot(0, 8);
        assert_eq!(hud.page(), 1);
        assert_eq!(hud.selected_slot(), 0);
        hud.select_slot(0, 9);
        assert_eq!(
            hud.page(),
            0,
            "repeated previous activation should page immediately"
        );
        assert_eq!(hud.selected_slot(), 0);
    }

    #[test]
    fn page_controls_stop_at_last_page_and_entries_keep_two_press_activation() {
        let uuid = [13; 16];
        let mut hud = SpectatorHud::new((0..18).map(player).collect(), Vec::new());
        hud.select_slot(0, 0);
        hud.select_slot(0, 1);
        hud.select_slot(0, 2); // open player view

        // There are seven entries on page zero and six on each later page: page two is full,
        // with no enabled next control beyond it.
        hud.select_slot(7, 3);
        hud.select_slot(7, 4);
        hud.select_slot(7, 5);
        hud.select_slot(7, 6);
        assert_eq!(hud.page(), 2);
        hud.select_slot(7, 7);
        assert_eq!(hud.page(), 2);

        // A normal object on the page still uses the ordinary select-then-activate behavior.
        hud.select_slot(1, 8);
        assert_eq!(hud.selected_slot(), 1);
        assert_eq!(hud.select_slot(1, 9), SpectatorAction::Spectate(uuid));
    }

    #[test]
    fn partial_last_page_keeps_empty_cells_and_disabled_next_stable() {
        let mut hud = SpectatorHud::new((0..8).map(player).collect(), Vec::new());
        hud.select_slot(0, 0);
        hud.select_slot(0, 1);
        hud.select_slot(0, 2); // open player view

        // Page zero has seven entries. Page one has one entry at slot one, empty slots two through
        // six, and the Java disabled next-page sentinel at slot seven.
        hud.select_slot(7, 3);
        hud.select_slot(7, 4);
        assert_eq!(hud.page(), 1);
        assert_eq!(hud.selected_slot(), 7);
        assert_eq!(hud.selected_text(), Some("Next Page"));

        hud.select_slot(2, 5); // Java's empty sentinel does not change selection.
        assert_eq!(hud.selected_slot(), 7);
        hud.select_slot(7, 6); // Disabled next remains disabled and cannot advance.
        assert_eq!(hud.page(), 1);
        assert_eq!(hud.selected_slot(), 7);
    }

    #[test]
    fn previous_navigation_returns_to_zero_without_underflow_and_reopen_resets_root() {
        let mut hud = SpectatorHud::new((0..8).map(player).collect(), Vec::new());
        hud.select_slot(0, 0);
        hud.select_slot(0, 1);
        hud.select_slot(0, 2); // open player view
        hud.select_slot(7, 3);
        hud.select_slot(7, 4); // page one
        hud.select_slot(0, 5);
        hud.select_slot(0, 6); // page zero, previous control is no longer present
        assert_eq!(hud.page(), 0);
        assert_eq!(hud.selected_slot(), 0);

        // A third press now targets page-zero entry zero, exactly as Java's slot mapping does;
        // it must not decrement the page below zero.
        hud.select_slot(0, 7);
        assert_eq!(hud.page(), 0);

        // Close is two-press activated, and a fresh open reconstructs the root view/page state.
        hud.select_slot(8, 8);
        hud.select_slot(8, 9);
        assert!(!hud.is_open());
        hud.select_slot(4, 10);
        assert!(hud.is_open());
        assert_eq!(hud.page(), 0);
        assert_eq!(hud.selected_slot(), -1);
        assert_eq!(hud.caption(), ROOT_CAPTION);
    }

    #[test]
    fn invalid_slots_and_scroll_directions_are_noops() {
        let mut hud = SpectatorHud::new(vec![player(1)], Vec::new());
        hud.select_slot(-1, 0);
        assert!(hud.is_open());
        assert_eq!(hud.selected_slot(), -1);
        hud.select_slot(9, 1);
        hud.scroll(0, 2);
        hud.scroll(2, 3);
        assert_eq!(hud.selected_slot(), -1);
        assert_eq!(hud.page(), 0);
    }

    #[test]
    fn scroll_skips_empty_and_disabled_cells() {
        let mut hud = SpectatorHud::new(Vec::new(), Vec::new());
        hud.select_slot(0, 10);
        hud.scroll(1, 20);
        assert_eq!(
            hud.selected_slot(),
            8,
            "close is the only enabled root cell"
        );
    }

    #[test]
    fn fade_is_full_for_three_seconds_then_linear_and_closes_at_five() {
        let mut hud = SpectatorHud::new(Vec::new(), Vec::new());
        hud.select_slot(0, 1_000);
        assert_eq!(hud.fade(4_000), 1.0);
        assert_eq!(hud.fade(5_000), 0.5);
        let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0);
        hud.draw(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx,
            &labels(),
            6_000,
        );
        assert!(!hud.is_open());
    }

    #[test]
    fn bar_geometry_slides_from_screen_bottom_to_vanilla_y() {
        let mut hud = SpectatorHud::new(Vec::new(), Vec::new());
        hud.select_slot(0, 1_000);
        let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0);
        let mut draw = DrawList::new();
        hud.draw(&mut draw, &mut uniform_font(), &ctx, &labels(), 5_000);
        let vertices = draw.vertices();
        assert_eq!(vertices[0].pos, [336.0, 469.0]);
        assert_eq!(vertices[1].pos, [336.0, 491.0]);
    }

    #[test]
    fn slide_keeps_vanillas_fractional_y_instead_of_snapping_to_pixels() {
        let mut hud = SpectatorHud::new(Vec::new(), Vec::new());
        hud.select_slot(0, 1_000);
        let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0);
        let mut draw = DrawList::new();
        hud.draw(&mut draw, &mut uniform_font(), &ctx, &labels(), 4_501);
        let y = draw.vertices()[0].pos[1];
        assert!((y - 463.511).abs() < 0.001, "fractional bar y was {y}");
    }

    #[test]
    fn java_uuid_order_treats_high_half_as_signed() {
        let positive = [0x7f; 16];
        let negative = [0x80; 16];
        assert_eq!(java_uuid_cmp(&negative, &positive), Ordering::Less);
    }
}
