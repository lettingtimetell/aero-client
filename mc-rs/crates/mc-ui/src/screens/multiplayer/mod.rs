//! The multiplayer server list: `servers.dat`, ping display, add / edit / delete, direct connect.
//!
//! java: `client/gui/GuiMultiplayer.java`, `ServerSelectionList.java`,
//! `ServerListEntryNormal.java`, `GuiScreenAddServer.java`, `GuiScreenServerList.java`,
//! `client/multiplayer/{ServerList,ServerData}.java`
//!
//! | Module | What it holds |
//! |---|---|
//! | [`server_data`] | [`ServerEntry`], [`ServerList`], and the `servers.dat` NBT codec |
//! | [`status`] | the status-ping JSON → MOTD / players / version / favicon |
//! | [`row`] | one list row's layout, hit-testing and drawing |
//!
//! # Where the network is
//!
//! Nowhere. [`ping_server`](mc_protocol::connection::ping_server) is `async` and this crate's
//! game loop is synchronous (§11), so the screen exposes [`MultiplayerScreen::take_ping_queue`]
//! — the addresses that want pinging — and [`MultiplayerScreen::apply_ping`] /
//! [`MultiplayerScreen::fail_ping`] for the results. `mc-client` owns the tokio task. That also
//! means every state transition here is testable without a socket.

pub mod row;
pub mod server_data;
pub mod status;

use mc_assets::Language;
use mc_version::GameVersion;

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_dirt_background, TITLE_COLOR};
use crate::screens::container::tooltip_box::draw as draw_hovering_text;
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::{Button, ScrollList, ScrollbarX};

pub use server_data::{ResourceMode, ServerEntry, ServerList};
pub use status::{StatusResponse, FAVICON_SIDE};

/// The protocol version this client speaks. The protocol-47 rule. A list shown for 1.7.10 compares with
/// that version's instead ([`MultiplayerScreen::set_client_version`]).
pub const CLIENT_PROTOCOL: i32 = 47;

/// `ServerListEntryNormal` submits ping work to a five-thread executor. The app owns the actual
/// tokio tasks, but this outbound queue is the last UI-owned boundary before those tasks exist.
// java: ServerListEntryNormal.java:28
const MAX_CONCURRENT_PINGS: usize = 5;

/// The most LAN games one discovery session lists.
///
/// Any device on the local network can multicast LAN beacons, and the duplicate key is the
/// sender's address plus the beacon's free-text `[AD]` part, so one device can announce as many
/// "games" as it likes. Vanilla's `LanServerList` keeps them all and scans the whole list for
/// every beacon: a flood grows memory and the game thread's work per beacon without bound while
/// Multiplayer is open. Real networks show a handful.
pub const MAX_LAN_SERVERS: usize = 64;

/// One validated server icon cached as RGBA8 for the renderer's dynamic-texture upload seam.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerIcon {
    /// The server address used by vanilla's synthetic `servers/<address>/icon` texture key.
    pub address: String,
    /// Exactly 64x64 tightly-packed RGBA8 pixels.
    pub rgba: Vec<u8>,
}

/// Button ids, matching vanilla's so the layout table reads the same as the Java.
mod ids {
    pub const CANCEL: i32 = 0;
    pub const JOIN: i32 = 1;
    pub const DELETE: i32 = 2;
    pub const ADD: i32 = 3;
    pub const DIRECT: i32 = 4;
    pub const EDIT: i32 = 7;
    pub const REFRESH: i32 = 8;
}

/// Identity of one slot in vanilla's saved -> scanning -> detected-LAN ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowId {
    Saved(usize),
    Scan,
    Lan(usize),
}

/// The multiplayer screen.
#[derive(Debug)]
pub struct MultiplayerScreen {
    /// The saved list. Public so the app can persist it after an edit.
    pub servers: ServerList,
    /// Discovered servers live only for this screen instance and are never serialised.
    lan_servers: Vec<ServerEntry>,
    list: ScrollList,
    buttons: Vec<Button>,
    selected: i32,
    hovering: Option<String>,
    last_click_ms: u64,
    ping_queue: Vec<usize>,
    /// Indices handed to the app but not yet completed. Keeping this here makes repeated app
    /// polls unable to fan out more than vanilla's five status-pinger threads.
    active_pings: Vec<usize>,
    /// Generation of the status-ping requests represented by this screen.
    ///
    /// Refresh and screen replacement invalidate the previous generation. The app copies this
    /// value into each async result so a late task cannot write into a newly refreshed row that
    /// happens to reuse the same index.
    ping_generation: u64,
    /// Validated icon pixels. The app uploads these under [`row::favicon_location`].
    icons: Vec<ServerIcon>,
    /// Whether the app must replace the renderer's same-address texture with these pixels.
    ///
    /// The synthetic resource key is stable for one server address, so checking only whether a
    /// renderer texture exists would leave a replaced favicon visually stale. This edge-triggered
    /// latch lets the app upload once when the decoded cache changes, while resource reloads can
    /// still use their ordinary missing-texture path.
    icons_dirty: bool,
    servers_dirty: bool,
    touchscreen: bool,
    hide_server_address: bool,
}

impl MultiplayerScreen {
    /// A screen over an already-loaded list.
    #[must_use]
    pub fn new(mut servers: ServerList) -> Self {
        let mut list = ScrollList::new(0, 0, 32, 0, row::SLOT_HEIGHT);
        list.list_width = row::LIST_WIDTH;
        list.scrollbar_placement = ScrollbarX::Centered { offset: 30 };
        // java: ServerSelectionList.getSize always includes the unclickable LAN scan row,
        // even before (or without) any discovered LAN servers.
        list.size = servers.len() + 1;
        let mut icons = Vec::new();
        let mut servers_dirty = false;
        for entry in servers.entries_mut() {
            let Some(encoded) = entry.icon_base64.as_deref() else {
                continue;
            };
            if let Some(rgba) = status::decode_favicon(encoded) {
                icons.push(ServerIcon {
                    address: entry.address.clone(),
                    rgba,
                });
            } else {
                // ServerListEntryNormal.prepareServerIcon nulls an invalid stored payload and
                // immediately saves the list.
                entry.icon_base64 = None;
                servers_dirty = true;
            }
        }
        Self {
            servers,
            lan_servers: Vec::new(),
            list,
            buttons: Vec::new(),
            selected: -1,
            hovering: None,
            last_click_ms: 0,
            ping_queue: Vec::new(),
            active_pings: Vec::new(),
            ping_generation: 0,
            icons_dirty: !icons.is_empty(),
            icons,
            servers_dirty,
            touchscreen: false,
            hide_server_address: false,
        }
    }

    fn row_count(&self) -> usize {
        self.servers.len() + 1 + self.lan_servers.len()
    }

    fn row_id(&self, slot: usize) -> Option<RowId> {
        let saved = self.servers.len();
        if slot < saved {
            return Some(RowId::Saved(slot));
        }
        if slot == saved {
            return Some(RowId::Scan);
        }
        let lan = slot.checked_sub(saved + 1)?;
        (lan < self.lan_servers.len()).then_some(RowId::Lan(lan))
    }

    fn entry_for_row(&self, row: RowId) -> Option<&ServerEntry> {
        match row {
            RowId::Saved(index) => self.servers.get(index),
            RowId::Lan(index) => self.lan_servers.get(index),
            RowId::Scan => None,
        }
    }

    fn selected_row(&self) -> Option<RowId> {
        self.selected().and_then(|slot| self.row_id(slot))
    }

    fn adjacent_selectable_slot(&self, slot: usize, up: bool) -> Option<usize> {
        if up {
            for candidate in (0..slot).rev() {
                if !matches!(self.row_id(candidate), Some(RowId::Scan) | None) {
                    return Some(candidate);
                }
            }
        } else {
            for candidate in slot.saturating_add(1)..self.row_count() {
                if !matches!(self.row_id(candidate), Some(RowId::Scan) | None) {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Add a newly detected LAN server unless its exact vanilla address key is already present.
    ///
    /// Java refreshes only the duplicate row's private `timeLastSeen`; that value is not rendered
    /// and entries are not expired while the screen remains open, so a duplicate is intentionally
    /// a no-op here and preserves its original order and MOTD.
    pub fn upsert_lan(&mut self, motd: String, address: String) -> bool {
        // A full list takes nothing more, so the scan below stays short.
        if self.lan_servers.len() >= MAX_LAN_SERVERS
            || self
                .lan_servers
                .iter()
                .any(|entry| entry.address == address)
        {
            return false;
        }
        let mut entry = ServerEntry::new("", address);
        entry.motd = motd;
        entry.lan = true;
        self.lan_servers.push(entry);
        self.list.size = self.row_count();
        true
    }

    /// Forget detections when Refresh replaces vanilla's multiplayer screen/listener.
    pub fn clear_lan_servers(&mut self) {
        let selected_lan = matches!(self.selected_row(), Some(RowId::Lan(_)));
        self.lan_servers.clear();
        self.list.size = self.row_count();
        if selected_lan {
            self.select(-1);
        }
    }

    /// The transient detected rows in discovery order.
    #[must_use]
    pub fn lan_servers(&self) -> &[ServerEntry] {
        &self.lan_servers
    }

    /// The generation attached to requests currently owned by this screen.
    #[must_use]
    pub fn ping_generation(&self) -> u64 {
        self.ping_generation
    }

    /// Use an app-owned generation when a screen is opened after another screen was closed.
    ///
    /// This is intentionally separate from [`Self::new`] so UI-only fixtures can keep their
    /// zero-based default while the live driver can allocate a process-wide monotonic token.
    pub fn set_ping_generation(&mut self, generation: u64) {
        self.ping_generation = generation;
    }

    /// Show the list to a client that plays `version` (1.8.9 unless this is called): each saved
    /// row is compared with `version`'s protocol, and a row that was pinged by another version
    /// (the list keeps its rows' ping state between visits) forgets that ping and is pinged again
    /// with `version`'s protocol ([`ServerEntry::set_client_version`]). Call it when the screen
    /// opens, before the first ping.
    pub fn set_client_version(&mut self, version: GameVersion) {
        for entry in self.servers.entries_mut() {
            entry.set_client_version(version);
        }
    }

    /// Invalidate queued results while keeping this screen instance alive (for example behind a
    /// confirmation dialog).
    pub fn invalidate_ping_generation(&mut self) {
        self.ping_generation = self.ping_generation.wrapping_add(1);
    }

    /// Cancel queued/in-flight UI requests without accepting any result from them.
    ///
    /// The tokio task itself is owned by `mc-client`; this only releases the UI's executor slots
    /// and returns transient rows to the lazy-ping state. The driver aborts the matching task
    /// handles and advances its generation separately.
    pub fn cancel_ping_requests(&mut self) {
        self.ping_queue.clear();
        self.active_pings.clear();
        for entry in self.servers.entries_mut() {
            if matches!(
                entry.state,
                server_data::PingState::Pinging | server_data::PingState::AwaitingStatus
            ) {
                entry.state = server_data::PingState::Unqueried;
                entry.motd.clear();
                entry.population.clear();
                entry.player_list.clear();
                entry.ping_ms = 0;
            }
        }
    }

    /// Invalidate all pending work for a Refresh action and queue the newly visible rows.
    pub fn refresh_pings(&mut self) {
        self.invalidate_ping_generation();
        self.ping_queue.clear();
        self.active_pings.clear();
        // java: GuiMultiplayer.refreshServerList constructs a fresh ServerList and
        // ServerSelectionList.  The saved NBT contains no ping/runtime fields, so a refresh
        // must not let the previous visit's status, population, sample list, or version label
        // leak into the new selector while the next status request is pending.
        for entry in self.servers.entries_mut() {
            entry.forget_ping();
        }
        self.lan_servers.clear();
        self.select(-1);
        self.rebuild_selector();
        self.queue_visible_pings();
    }

    /// Rebuild the selector state after vanilla replaces its entry-wrapper list.
    ///
    /// Rust rows borrow the screen's [`ServerEntry`] values directly rather than wrapping each
    /// one in a `ServerListEntryNormal`, but the replacement is still observable: the selector
    /// starts at scroll zero, has no selected row, and contains the saved rows plus the LAN scan
    /// row.  Keep the current geometry so this also works before the next layout pass.
    fn rebuild_selector(&mut self) {
        let width = self.list.width;
        let height = self.list.height;
        let top = self.list.top;
        let bottom = self.list.bottom;
        let slot_height = self.list.slot_height;
        let left = self.list.left;
        let right = self.list.right;
        let header_padding = self.list.header_padding;
        let has_header = self.list.has_header;
        let show_selection_box = self.list.show_selection_box;
        let list_width = self.list.list_width;
        let scrollbar_placement = self.list.scrollbar_placement;
        let enabled = self.list.enabled;
        let mut list = ScrollList::new(width, height, top, bottom, slot_height);
        list.left = left;
        list.right = right;
        list.header_padding = header_padding;
        list.has_header = has_header;
        list.show_selection_box = show_selection_box;
        list.list_width = list_width;
        list.scrollbar_placement = scrollbar_placement;
        list.enabled = enabled;
        list.size = self.row_count();
        list.selected = self.selected;
        self.list = list;
    }

    /// Refresh the selector's entry count and selected index after a saved-row replacement.
    ///
    /// `ServerSelectionList.func_148195_a` rebuilds its wrapper rows after every reorder.  The
    /// Rust entries remain in-place, but keeping this explicit makes the same lifecycle boundary
    /// visible and prevents a future LAN/list-size change from leaving stale selector metadata.
    fn repopulate_selector(&mut self) {
        self.list.size = self.row_count();
        self.list.selected = self.selected;
    }

    /// Lay the buttons and list out for a screen size. java: `createButtons`
    // java: GuiMultiplayer.java:92
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        self.list.set_dimensions(w, h, 32, h - 64);
        self.list.size = self.row_count();
        let tr = |k: &str| lang.tr(k).to_owned();
        self.buttons = vec![
            Button::sized(
                ids::EDIT,
                w / 2 - 154,
                h - 28,
                70,
                20,
                tr("selectServer.edit"),
            ),
            Button::sized(
                ids::DELETE,
                w / 2 - 74,
                h - 28,
                70,
                20,
                tr("selectServer.delete"),
            ),
            Button::sized(
                ids::JOIN,
                w / 2 - 154,
                h - 52,
                100,
                20,
                tr("selectServer.select"),
            ),
            Button::sized(
                ids::DIRECT,
                w / 2 - 50,
                h - 52,
                100,
                20,
                tr("selectServer.direct"),
            ),
            Button::sized(
                ids::ADD,
                w / 2 + 4 + 50,
                h - 52,
                100,
                20,
                tr("selectServer.add"),
            ),
            Button::sized(
                ids::REFRESH,
                w / 2 + 4,
                h - 28,
                70,
                20,
                tr("selectServer.refresh"),
            ),
            Button::sized(
                ids::CANCEL,
                w / 2 + 4 + 76,
                h - 28,
                75,
                20,
                tr("gui.cancel"),
            ),
        ];
        self.select(self.selected);
        self.queue_visible_pings();
    }

    /// Queue status pings for the server rows that this frame can actually draw.
    // java: ServerListEntryNormal.java:50 — `drawEntry`'s lazy `field_78841_f` gate.  The LAN
    // scan slot is part of `visible_slots`, but has no saved-server entry and is skipped here.
    pub fn queue_visible_pings(&mut self) {
        for i in self.list.visible_slots() {
            let Some(e) = self.servers.get_mut(i) else {
                continue;
            };
            if e.state == server_data::PingState::Unqueried {
                e.state = server_data::PingState::Pinging;
                e.motd = String::new();
                e.population = String::new();
                e.player_list.clear();
                e.ping_ms = 0;
                self.ping_queue.push(i);
            }
        }
    }

    /// Take the pending ping requests. The app resolves these off-thread.
    ///
    /// Handing a request off is the UI-side boundary corresponding to
    /// `OldServerPinger.ping` installing its status handler. From this point vanilla writes the
    /// literal MOTD `Pinging...` and exposes the `pingToServer == -1` sprite, rather than keeping
    /// the initial animated `-2` sprite through the entire status exchange.
    // java: OldServerPinger.java:57-59; ServerListEntryNormal.java:103-151
    pub fn take_ping_queue(&mut self) -> Vec<(usize, String)> {
        let room = MAX_CONCURRENT_PINGS.saturating_sub(self.active_pings.len());
        let mut out = Vec::with_capacity(room);
        while out.len() < room && !self.ping_queue.is_empty() {
            let index = self.ping_queue.remove(0);
            let Some(e) = self.servers.get_mut(index) else {
                continue;
            };
            // A refresh or another UI transition can make an already-staged item obsolete
            // before the app polls this queue. It must not consume a scarce executor slot.
            if e.state != server_data::PingState::Pinging {
                continue;
            }
            e.state = server_data::PingState::AwaitingStatus;
            e.motd = "Pinging...".to_owned();
            e.ping_ms = -1;
            self.active_pings.push(index);
            out.push((index, e.address.clone()));
        }
        out
    }

    /// Fold a successful ping back into the list.
    pub fn apply_ping(
        &mut self,
        index: usize,
        status: &StatusResponse,
        latency_ms: i64,
        lang: &Language,
    ) -> bool {
        self.finish_ping(index);
        let Some(e) = self.servers.get_mut(index) else {
            return false;
        };
        let icon_changed = status.apply_to(e, latency_ms, lang);
        let address = e.address.clone();
        let icon_rgba = if status.favicon_base64.is_some() {
            // The live transport result carries the already-validated 64x64 pixels. Raw UI
            // fixtures leave this field empty and transparently take `decode_favicon` instead.
            status.decoded_favicon()
        } else if status.favicon_present {
            // An unsupported prefix is vanilla's keep-old-icon branch.
            e.icon_base64.as_deref().and_then(status::decode_favicon)
        } else {
            // An absent field clears both the persisted payload and the address cache.
            None
        };
        let icon = icon_rgba.map(|rgba| ServerIcon {
            address: address.clone(),
            rgba,
        });
        let previous_icon = self
            .icons
            .iter()
            .find(|cached| cached.address == address)
            .map(|cached| cached.rgba.clone());
        let icon_cache_changed =
            previous_icon.as_deref() != icon.as_ref().map(|cached| cached.rgba.as_slice());
        self.icons.retain(|cached| cached.address != address);
        if let Some(icon) = icon {
            self.icons.push(icon);
        }
        if icon_cache_changed {
            self.icons_dirty = true;
        }
        self.servers_dirty |= icon_changed;
        // A result opens one executor permit.  Vanilla reaches the next row on a later draw;
        // this queues that row now, but the driver still starts it only after the UI boundary.
        self.queue_visible_pings();
        icon_changed
    }

    /// Validated server icons that must be resident in the UI renderer.
    #[must_use]
    pub fn icons(&self) -> &[ServerIcon] {
        &self.icons
    }

    /// Take the favicon-upload edge. The renderer may have a texture under the same address,
    /// therefore a changed decode must force one replacement upload rather than only probing
    /// `has_texture`.
    pub fn take_icon_upload_dirty(&mut self) -> bool {
        std::mem::take(&mut self.icons_dirty)
    }

    /// Take whether favicon validation changed the persisted server list since the last call.
    pub fn take_servers_dirty(&mut self) -> bool {
        std::mem::take(&mut self.servers_dirty)
    }

    /// Apply the current Touchscreen Mode option to row overlay visibility.
    pub fn set_touchscreen(&mut self, touchscreen: bool) {
        self.touchscreen = touchscreen;
    }

    /// Apply `hideServerAddress` to detected-LAN rows.
    pub fn set_hide_server_address(&mut self, hide: bool) {
        self.hide_server_address = hide;
    }

    /// Mark a ping as having failed.
    pub fn fail_ping(&mut self, index: usize, unresolved_host: bool) {
        self.finish_ping(index);
        if let Some(e) = self.servers.get_mut(index) {
            status::mark_failed(e, unresolved_host);
        }
        self.queue_visible_pings();
    }

    /// Return an executor slot when the app reports a status-ping result. The app may receive a
    /// stale result after a row was removed, so this deliberately runs before the row lookup.
    fn finish_ping(&mut self, index: usize) {
        self.active_pings.retain(|active| *active != index);
    }

    /// The selected row, or `None`.
    #[must_use]
    pub fn selected(&self) -> Option<usize> {
        usize::try_from(self.selected).ok()
    }

    /// Select a row (`-1` clears), updating which buttons are enabled.
    // java: GuiMultiplayer.java:402 (selectServer)
    pub fn select(&mut self, index: i32) {
        let row = usize::try_from(index)
            .ok()
            .and_then(|slot| self.row_id(slot));
        let valid = matches!(row, Some(RowId::Saved(_) | RowId::Lan(_)));
        let saved = matches!(row, Some(RowId::Saved(_)));
        self.selected = if valid { index } else { -1 };
        self.list.selected = self.selected;
        for b in &mut self.buttons {
            if b.id == ids::JOIN {
                b.enabled = valid;
            } else if matches!(b.id, ids::EDIT | ids::DELETE) {
                b.enabled = saved;
            }
        }
    }

    /// Whether row `index` may move up / down.
    // java: GuiMultiplayer.java:455 (func_175392_a / func_175394_b)
    #[must_use]
    pub fn can_move(&self, index: usize) -> (bool, bool) {
        match self.row_id(index) {
            Some(RowId::Saved(saved)) => (saved > 0, saved + 1 < self.servers.len()),
            _ => (false, false),
        }
    }

    /// Move a row, returning whether the list changed. `to_end` is vanilla's shift-click, which
    /// jumps the entry to the top or bottom instead of one step.
    // java: GuiMultiplayer.java:465 (func_175391_a / func_175393_b)
    pub fn move_row(&mut self, index: usize, up: bool, to_end: bool) -> bool {
        let Some(RowId::Saved(index)) = self.row_id(index) else {
            return false;
        };
        let target = if up {
            if index == 0 {
                return false;
            }
            if to_end {
                0
            } else {
                index - 1
            }
        } else {
            if index + 1 >= self.servers.len() {
                return false;
            }
            if to_end {
                self.servers.len() - 1
            } else {
                index + 1
            }
        };
        if !self.servers.swap(index, target) {
            return false;
        }
        // Replies come back by row index, so a reply still on its way
        // for either swapped row would land on the other one (another server's MOTD, players and
        // icon, and one row left at "Pinging..." for good). Vanilla's ping task holds the row
        // itself. When a swapped row is waiting, restart the waiting rows as Refresh does: the
        // replies already asked for are dropped by generation, rows that have answered keep it.
        let waiting =
            |row: usize| self.active_pings.contains(&row) || self.ping_queue.contains(&row);
        if waiting(index) || waiting(target) {
            self.cancel_ping_requests();
            self.invalidate_ping_generation();
            self.queue_visible_pings();
        }
        if self.selected == index as i32 {
            self.select(target as i32);
        }
        self.repopulate_selector();
        true
    }

    /// Delete the selected row.
    pub fn delete_selected(&mut self) -> Option<ServerEntry> {
        let RowId::Saved(i) = self.selected_row()? else {
            return None;
        };
        let removed = self.servers.remove(i);
        if let Some(address) = removed.as_ref().map(|entry| entry.address.as_str()) {
            // The renderer key is address-scoped, so retain a cached favicon while another row
            // still points at that same key. Otherwise the removed row would leave a stale icon
            // eligible for a later row with no favicon of its own.
            let still_used = self
                .servers
                .entries()
                .iter()
                .any(|entry| entry.address == address);
            if !still_used {
                let before = self.icons.len();
                self.icons.retain(|icon| icon.address != address);
                if self.icons.len() != before {
                    self.icons_dirty = true;
                }
            }
        }
        self.list.size = self.row_count();
        self.select(-1);
        removed
    }

    /// Append a new entry and re-select nothing, as vanilla's add flow does.
    pub fn add(&mut self, entry: ServerEntry) {
        self.servers.add(entry);
        self.list.size = self.row_count();
        self.select(-1);
    }

    /// Replace the selected entry with an edited copy.
    pub fn replace_selected(&mut self, entry: ServerEntry) -> bool {
        match self.selected_row() {
            Some(RowId::Saved(i)) => self.servers.set(i, entry),
            None => false,
            _ => false,
        }
    }

    /// Handle a click.
    // java: GuiMultiplayer.java:435 (mouseClicked) + ServerListEntryNormal.mousePressed
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32, now_ms: u64) -> Action {
        if button != 0 {
            return Action::None;
        }
        if let Some(id) = self
            .buttons
            .iter()
            .find(|b| b.mouse_pressed(x, y))
            .map(|b| b.id)
        {
            return self.activate(id);
        }
        if self.list.mouse_pressed(x, y) {
            return Action::None;
        }
        if let Some(index) = self.slot_index_at(x, y) {
            let rect = self.list.slot_rect(index);
            if matches!(self.row_id(index), Some(RowId::Lan(_))) {
                let double = self.selected() == Some(index)
                    && now_ms.saturating_sub(self.last_click_ms) < 250;
                self.select(index as i32);
                self.last_click_ms = now_ms;
                return if double {
                    self.connect_selected()
                } else {
                    Action::None
                };
            }
            let (up, down) = self.can_move(index);
            let hit = row::hit_test(x - rect.x, y - rect.y, up, down);
            return match hit {
                row::RowHit::Join => {
                    self.select(index as i32);
                    self.connect_selected()
                }
                row::RowHit::MoveUp => {
                    self.move_row(index, true, false);
                    Action::SaveServers
                }
                row::RowHit::MoveDown => {
                    self.move_row(index, false, false);
                    Action::SaveServers
                }
                row::RowHit::Body => {
                    let double = self.selected() == Some(index)
                        && now_ms.saturating_sub(self.last_click_ms) < 250;
                    self.select(index as i32);
                    self.last_click_ms = now_ms;
                    if double {
                        self.connect_selected()
                    } else {
                        Action::None
                    }
                }
            };
        }
        Action::None
    }

    /// Vanilla's `getSlotIndexFromScreenCoords`, with `ServerSelectionList`'s wider scrollbar
    /// offset.
    ///
    /// `ScrollList::slot_index_at` hard-codes `GuiSlot`'s `width / 2 + 124`; the server list
    /// overrides it to `+ 154` and widens the content to 305, so using the widget's own hit
    /// test would make the rightmost 28 pixel columns of every row unclickable.
    // java: GuiSlot.java:149 + ServerSelectionList.java:91
    #[must_use]
    pub fn slot_index_at(&self, x: i32, y: i32) -> Option<usize> {
        let (l, r) = self.list.content_bounds();
        let scrollbar_x = self.list.width / 2 + row::SCROLLBAR_OFFSET;
        let k = y - self.list.top + self.list.scroll_pixels() - 4;
        let slot = k / row::SLOT_HEIGHT;
        (x < scrollbar_x
            && x >= l
            && x <= r
            && k >= 0
            && slot >= 0
            && self
                .row_id(slot as usize)
                .is_some_and(|row| !matches!(row, RowId::Scan)))
        .then_some(slot as usize)
    }

    fn connect_selected(&self) -> Action {
        match self.selected_row().and_then(|row| self.entry_for_row(row)) {
            Some(e) => Action::Connect(Box::new(e.clone())),
            None => Action::None,
        }
    }

    fn activate(&mut self, id: i32) -> Action {
        match id {
            ids::JOIN => self.connect_selected(),
            ids::CANCEL => Action::Back,
            ids::DIRECT => Action::OpenDirectConnect,
            ids::ADD => Action::OpenAddServer,
            ids::EDIT => match self.selected_row() {
                Some(RowId::Saved(i)) => match self.servers.get(i) {
                    Some(e) => Action::OpenEditServer(Box::new(e.clone())),
                    None => Action::None,
                },
                _ => Action::None,
            },
            ids::DELETE => match self.selected_row() {
                Some(RowId::Saved(i)) => match self.servers.get(i) {
                    Some(e) => Action::ConfirmDeleteServer(e.name.clone()),
                    None => Action::None,
                },
                _ => Action::None,
            },
            ids::REFRESH => {
                self.refresh_pings();
                Action::None
            }
            _ => Action::None,
        }
    }

    /// Handle a key press.
    // java: GuiMultiplayer.java:266 (keyTyped)
    pub fn key_typed(&mut self, key: i32, shift: bool) -> Action {
        use crate::widget::keys;
        match key {
            // F5 refreshes.
            63 => self.activate(ids::REFRESH),
            keys::ESCAPE => Action::Back,
            keys::UP | keys::DOWN => {
                let Some(i) = self.selected() else {
                    return Action::None;
                };
                let up = key == keys::UP;
                if shift {
                    if self.move_row(i, up, false) {
                        // java: GuiMultiplayer.keyTyped scrolls exactly one slot after the
                        // swap/reselect and then repopulates ServerSelectionList.
                        self.list.scroll_by(if up {
                            -row::SLOT_HEIGHT
                        } else {
                            row::SLOT_HEIGHT
                        });
                        self.repopulate_selector();
                        return Action::SaveServers;
                    }
                } else {
                    if let Some(next) = self.adjacent_selectable_slot(i, up) {
                        self.select(next as i32);
                        self.list.scroll_by(if up {
                            -row::SLOT_HEIGHT
                        } else {
                            row::SLOT_HEIGHT
                        });
                    } else {
                        // java: GuiMultiplayer.keyTyped clears the selector when an arrow
                        // would move past either edge of the saved list (rather than leaving
                        // the prior row selected).
                        self.select(-1);
                    }
                }
                Action::None
            }
            keys::ENTER | keys::NUMPAD_ENTER => self.connect_selected(),
            _ => Action::None,
        }
    }

    /// Wheel and drag pass-through.
    ///
    /// `GuiMultiplayer.handleMouseInput` delegates to `GuiSlot.handleMouseInput`, whose wheel
    /// branch is active only while the cursor is inside the list viewport. Keep the coordinates
    /// here so the live driver cannot scroll the server list from a button or the title area.
    pub fn mouse_wheel(&mut self, x: i32, y: i32, delta: f32) {
        self.list.mouse_wheel_at(x, y, delta);
        self.queue_visible_pings();
    }
    /// Continue a scrollbar drag.
    pub fn mouse_dragged(&mut self, y: i32) {
        self.list.mouse_dragged(y);
        self.queue_visible_pings();
    }
    /// End a scrollbar drag.
    pub fn mouse_released(&mut self) {
        self.list.mouse_released();
    }

    /// Draw the whole screen.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
        now_ms: u64,
    ) {
        self.hovering = None;
        draw_dirt_background(d, ctx.width, ctx.height, 0.0);
        self.list.size = self.row_count();
        self.list.draw_background(d);
        for index in self.list.visible_slots() {
            let rect = self.list.slot_rect(index);
            match self.row_id(index) {
                Some(RowId::Scan) => row::draw_lan_scan_row(
                    d,
                    font,
                    lang.tr("lanServer.scanning"),
                    ctx.width,
                    rect.y,
                    now_ms,
                ),
                Some(RowId::Lan(lan)) => {
                    let Some(entry) = self.lan_servers.get(lan) else {
                        continue;
                    };
                    if self.selected() == Some(index) {
                        self.list.draw_selection_box(d, index);
                    }
                    row::draw_lan_server_row(
                        d,
                        font,
                        lang.tr("lanServer.title"),
                        &entry.motd,
                        &entry.address,
                        lang.tr("selectServer.hiddenAddress"),
                        self.hide_server_address,
                        rect.x,
                        rect.y,
                    );
                }
                Some(RowId::Saved(saved)) => {
                    let Some(entry) = self.servers.get(saved) else {
                        continue;
                    };
                    if self.selected() == Some(index) {
                        self.list.draw_selection_box(d, index);
                    }
                    let (up, down) = self.can_move(index);
                    let have_icon = self.icons.iter().any(|icon| icon.address == entry.address);
                    // GuiListExtended names this drawEntry argument `isSelected`, but computes it
                    // from getSlotIndexFromScreenCoords(mouse): hover, not persistent selection.
                    let hovered = self.slot_index_at(mouse.0, mouse.1) == Some(index);
                    if let Some(text) = row::draw_row(
                        d,
                        font,
                        entry,
                        saved,
                        rect.x,
                        rect.y,
                        row::LIST_WIDTH,
                        mouse,
                        self.touchscreen || hovered,
                        up,
                        down,
                        now_ms,
                        have_icon,
                    ) {
                        self.hovering = Some(text);
                    }
                }
                None => {}
            }
        }
        self.list.draw_overlay(d);
        crate::widget::draw_centered_string(
            d,
            font,
            lang.tr("multiplayer.title"),
            ctx.width / 2,
            20,
            TITLE_COLOR,
        );
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
        if let Some(text) = self.hovering.clone() {
            let lines: Vec<String> = text.split('\n').map(str::to_owned).collect();
            draw_hovering_text(d, font, &lines, mouse.0, mouse.1, ctx.width, ctx.height);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;
    use base64::Engine;
    use std::io::Cursor;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn lang() -> Language {
        let mut l = Language::new("en_US");
        for k in [
            "selectServer.edit",
            "selectServer.delete",
            "selectServer.select",
            "selectServer.direct",
            "selectServer.add",
            "selectServer.refresh",
            "gui.cancel",
            "multiplayer.title",
            "lanServer.scanning",
        ] {
            l.insert(k, k);
        }
        l
    }

    fn screen(n: usize) -> MultiplayerScreen {
        let mut list = ServerList::new();
        for i in 0..n {
            list.add(ServerEntry::new(format!("s{i}"), format!("host{i}")));
        }
        let mut s = MultiplayerScreen::new(list);
        s.init(&ctx(), &lang());
        s
    }

    fn favicon_base64() -> String {
        favicon_base64_with([9, 8, 7, 255])
    }

    fn favicon_base64_with(color: [u8; 4]) -> String {
        let image = image::RgbaImage::from_pixel(64, 64, image::Rgba(color));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode PNG");
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn the_buttons_land_on_vanillas_exact_coordinates() {
        let s = screen(0);
        let by = |id: i32| s.buttons.iter().find(|b| b.id == id).expect("button");
        let (w, h) = (854, 480);
        assert_eq!(
            (by(ids::EDIT).rect.x, by(ids::EDIT).rect.y),
            (w / 2 - 154, h - 28)
        );
        assert_eq!(by(ids::EDIT).rect.width, 70);
        assert_eq!(
            (by(ids::DELETE).rect.x, by(ids::DELETE).rect.y),
            (w / 2 - 74, h - 28)
        );
        assert_eq!(
            (by(ids::JOIN).rect.x, by(ids::JOIN).rect.y),
            (w / 2 - 154, h - 52)
        );
        assert_eq!(by(ids::JOIN).rect.width, 100);
        assert_eq!(by(ids::DIRECT).rect.x, w / 2 - 50);
        assert_eq!(by(ids::ADD).rect.x, w / 2 + 54);
        assert_eq!(by(ids::REFRESH).rect.x, w / 2 + 4);
        assert_eq!(
            (by(ids::CANCEL).rect.x, by(ids::CANCEL).rect.width),
            (w / 2 + 80, 75)
        );
    }

    #[test]
    fn join_edit_and_delete_are_disabled_until_something_is_selected() {
        let mut s = screen(3);
        let enabled =
            |s: &MultiplayerScreen, id: i32| s.buttons.iter().find(|b| b.id == id).unwrap().enabled;
        assert!(!enabled(&s, ids::JOIN));
        assert!(!enabled(&s, ids::EDIT));
        assert!(!enabled(&s, ids::DELETE));
        assert!(enabled(&s, ids::ADD), "add is always available");
        s.select(1);
        assert!(enabled(&s, ids::JOIN) && enabled(&s, ids::EDIT) && enabled(&s, ids::DELETE));
        s.select(99);
        assert_eq!(
            s.selected(),
            None,
            "an out-of-range index clears the selection"
        );
        assert!(!enabled(&s, ids::JOIN));
    }

    #[test]
    fn every_visible_entry_is_queued_for_a_ping_exactly_once() {
        let mut s = screen(3);
        let q = s.take_ping_queue();
        assert_eq!(q.len(), 3);
        assert_eq!(q[0], (0, "host0".to_owned()));
        assert!(s.take_ping_queue().is_empty(), "not queued twice");
        s.queue_visible_pings();
        assert!(
            s.take_ping_queue().is_empty(),
            "still pinging, not requeued"
        );
    }

    /// `ServerListEntryNormal` has a five-thread executor. Calling the app-facing drain again
    /// while those five tasks have not reported a result must not make a sixth connection
    /// possible, even if the user scrolls the UI between the polls.
    #[test]
    fn the_ping_queue_never_hands_the_app_more_than_five_live_entries() {
        let mut s = screen(40);
        let first = s.take_ping_queue();
        assert_eq!(
            first.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4],
            "the initial FIFO batch is exactly the executor width"
        );
        assert_eq!(s.active_pings.len(), MAX_CONCURRENT_PINGS);
        assert!(
            s.take_ping_queue().is_empty(),
            "five live pings fill every slot"
        );

        // Scroll well beyond the initial seven-ish rows. This changes the viewport, but cannot
        // bypass the UI-owned task cap before a real async completion frees a slot.
        s.mouse_wheel(427, 100, -100.0);
        assert!(
            s.take_ping_queue().is_empty(),
            "scrolling cannot create a sixth live ping"
        );

        s.fail_ping(2, false);
        assert_eq!(s.active_pings.len(), MAX_CONCURRENT_PINGS - 1);
        assert_eq!(
            s.take_ping_queue(),
            [(5, "host5".to_owned())],
            "one completion opens exactly one FIFO slot"
        );
        assert_eq!(s.active_pings.len(), MAX_CONCURRENT_PINGS);
    }

    #[test]
    fn server_list_wheel_is_viewport_gated_and_clamps_at_the_next_draw() {
        let mut s = screen(40);
        let step = row::SLOT_HEIGHT / 2;
        let max = s.list.max_scroll();
        assert!(max > 0, "the fixture must have scrollable server rows");

        // GuiSlot's bounds are inclusive, but a wheel over the title/fade or side gutter is
        // ignored.
        s.mouse_wheel(427, s.list.top - 1, -1.0);
        s.mouse_wheel(s.list.right + 1, 100, -1.0);
        assert_eq!(
            s.list.amount_scrolled(),
            0.0,
            "outside-viewport wheel input must not move the server list"
        );
        s.mouse_wheel(427, s.list.top, -1.0);
        assert_eq!(
            s.list.amount_scrolled(),
            step as f32,
            "the viewport's top boundary is part of GuiSlot's hit region"
        );

        // Wheel input intentionally leaves the transient overscroll for GuiSlot.drawScreen's
        // bindAmountScrolled step; this catches an eager clamp in either seam.
        for _ in 0..=(max / step + 1) {
            s.mouse_wheel(427, 100, -1.0);
        }
        assert!(
            s.list.amount_scrolled() > max as f32,
            "wheel overscroll must remain observable until drawing"
        );
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, &ctx(), (0, 0), &lang(), 0);
        assert_eq!(
            s.list.amount_scrolled(),
            max as f32,
            "drawing clamps the deferred wheel overscroll to the vanilla maximum"
        );
    }

    #[test]
    fn beginning_a_reping_clears_the_previous_row_state() {
        let mut s = screen(1);
        let _ = s.take_ping_queue();
        let entry = s.servers.get_mut(0).unwrap();
        entry.state = server_data::PingState::Unqueried;
        entry.motd = "stale MOTD".to_owned();
        entry.population = "1/20".to_owned();
        entry.player_list = "stale player".to_owned();
        s.queue_visible_pings();
        let entry = s.servers.get(0).unwrap();
        assert_eq!(entry.state, server_data::PingState::Pinging);
        assert!(entry.motd.is_empty());
        assert!(entry.population.is_empty());
        assert!(entry.player_list.is_empty());
    }

    #[test]
    fn only_rows_in_the_viewport_are_staged_for_ping() {
        let mut s = screen(40);
        let initially_visible = s.list.visible_slots();
        assert!(initially_visible.iter().all(|&index| {
            s.servers.get(index).unwrap().state == server_data::PingState::Pinging
        }));
        assert_eq!(
            s.servers.get(39).unwrap().state,
            server_data::PingState::Unqueried,
            "opening the screen must not touch rows outside GuiSlot's viewport"
        );

        s.mouse_wheel(427, 100, -1_000.0);
        let scrolled_visible = s.list.visible_slots();
        assert!(scrolled_visible.iter().all(|&index| {
            s.servers.get(index).unwrap().state == server_data::PingState::Pinging
        }));
        assert!(
            scrolled_visible
                .iter()
                .any(|index| !initially_visible.contains(index)),
            "the scroll changes which rows Java's drawEntry can stage"
        );
    }

    /// Java's initial `pingToServer = -2` state has the animated icon. Once the task begins,
    /// `OldServerPinger.ping` changes that sentinel to `-1` and supplies this literal MOTD before
    /// its first status reply. Keeping the two stages separate is what makes the result observable
    /// and stops a stale population line from surviving into the second one.
    #[test]
    fn taking_a_ping_request_enters_vanillas_pinging_motd_phase() {
        let mut s = screen(1);
        assert_eq!(
            s.servers.get(0).unwrap().state,
            server_data::PingState::Pinging
        );
        assert!(s.servers.get(0).unwrap().motd.is_empty());

        assert_eq!(s.take_ping_queue(), [(0, "host0".to_owned())]);
        let entry = s.servers.get(0).unwrap();
        assert_eq!(entry.state, server_data::PingState::AwaitingStatus);
        assert_eq!(entry.motd, "Pinging...");
        assert_eq!(entry.ping_ms, -1);
    }

    #[test]
    fn refresh_requeues_everything() {
        let mut s = screen(2);
        let _ = s.take_ping_queue();
        let before = s.ping_generation();
        s.activate(ids::REFRESH);
        assert_ne!(s.ping_generation(), before);
        assert_eq!(s.take_ping_queue().len(), 2);
    }

    #[test]
    fn refresh_rebuilds_the_selector_and_drops_stale_status_fields() {
        let mut s = screen(40);
        s.select(3);
        s.mouse_wheel(427, 100, -1.0);
        assert!(s.list.amount_scrolled() > 0.0);
        assert!(s.upsert_lan("LAN".to_owned(), "192.0.2.9:25565".to_owned()));

        let entry = s.servers.get_mut(0).expect("saved row");
        entry.state = server_data::PingState::Ok;
        entry.motd = "stale MOTD".to_owned();
        entry.population = "3/20".to_owned();
        entry.player_list = "stale player".to_owned();
        entry.ping_ms = 123;
        entry.protocol = 999;
        entry.game_version = "stale version".to_owned();

        s.activate(ids::REFRESH);

        assert_eq!(s.selected(), None, "fresh Java screen has no selected row");
        assert_eq!(s.list.selected, -1);
        assert_eq!(s.list.amount_scrolled(), 0.0);
        assert!(
            s.lan_servers().is_empty(),
            "refresh drops old LAN detections"
        );
        assert_eq!(s.list.size, s.servers.len() + 1);
        let entry = s.servers.get(0).expect("saved row");
        assert_eq!(entry.state, server_data::PingState::Pinging);
        assert!(entry.motd.is_empty());
        assert!(entry.population.is_empty());
        assert!(entry.player_list.is_empty());
        assert_eq!(entry.ping_ms, 0);
        assert_eq!(entry.protocol, CLIENT_PROTOCOL);
        assert_eq!(entry.game_version, "1.8.9");
    }

    #[test]
    fn canceling_ping_requests_releases_slots_and_returns_rows_to_lazy_state() {
        let mut s = screen(2);
        assert_eq!(s.take_ping_queue().len(), 2);
        let before = s.ping_generation();
        s.cancel_ping_requests();
        assert!(s.active_pings.is_empty());
        assert!(s
            .servers
            .entries()
            .iter()
            .all(|entry| entry.state == server_data::PingState::Unqueried));
        s.invalidate_ping_generation();
        assert_ne!(s.ping_generation(), before);
        s.queue_visible_pings();
        assert_eq!(s.take_ping_queue().len(), 2);
    }

    #[test]
    fn retained_status_without_pong_keeps_fields_but_uses_no_connection_sentinel() {
        let mut s = screen(1);
        let status = StatusResponse::parse(
            r#"{"description":"still here","version":{"name":"1.8.9","protocol":47},"players":{"online":3,"max":9}}"#,
        )
        .expect("status response");
        s.apply_ping(0, &status, -1, &lang());
        let entry = s.servers.get(0).expect("row");
        assert_eq!(entry.state, server_data::PingState::Ok);
        assert_eq!(entry.ping_ms, -1);
        assert_eq!(entry.motd, "still here\u{a7}r");
        assert_eq!(row::ping_tooltip(entry), "(no connection)");
    }

    #[test]
    fn a_ping_result_lands_on_the_right_row() {
        let mut s = screen(2);
        let status = StatusResponse::parse(
            r#"{"players":{"online":3,"max":9},"version":{"name":"1.8.9","protocol":47}}"#,
        )
        .expect("parse");
        s.apply_ping(1, &status, 55, &lang());
        assert_eq!(s.servers.get(1).unwrap().ping_ms, 55);
        assert_eq!(s.servers.get(0).unwrap().ping_ms, 0);
        s.fail_ping(0, false);
        assert_eq!(
            s.servers.get(0).unwrap().state,
            server_data::PingState::Failed
        );
        // Out-of-range indices are ignored rather than panicking: the reply may arrive after
        // the row it belongs to was deleted.
        s.apply_ping(99, &status, 1, &lang());
        s.fail_ping(99, false);
    }

    #[test]
    fn a_valid_ping_icon_is_cached_as_64_by_64_rgba_for_upload() {
        let mut s = screen(1);
        let status = StatusResponse {
            favicon_base64: Some(favicon_base64()),
            ..StatusResponse::default()
        };
        assert!(s.apply_ping(0, &status, 55, &lang()));
        assert_eq!(s.icons().len(), 1);
        assert_eq!(s.icons()[0].address, "host0");
        assert_eq!(s.icons()[0].rgba.len(), 64 * 64 * 4);
        assert!(s.take_servers_dirty());
        assert!(!s.take_servers_dirty(), "the dirty latch is edge-triggered");
    }

    #[test]
    fn replacing_or_clearing_a_favicon_invalidates_the_live_texture_edge() {
        let mut s = screen(1);
        let first = StatusResponse {
            favicon_base64: Some(favicon_base64()),
            ..StatusResponse::default()
        };
        assert!(s.apply_ping(0, &first, 55, &lang()));
        assert!(s.take_icon_upload_dirty(), "the first icon needs an upload");
        assert!(!s.take_icon_upload_dirty(), "the upload edge is one-shot");
        assert_eq!(s.icons()[0].rgba[0..4], [9, 8, 7, 255]);

        let replacement = StatusResponse {
            favicon_base64: Some(favicon_base64_with([1, 2, 3, 255])),
            ..StatusResponse::default()
        };
        assert!(s.apply_ping(0, &replacement, 56, &lang()));
        assert!(
            s.take_icon_upload_dirty(),
            "a same-address replacement must force the renderer upload"
        );
        assert_eq!(s.icons()[0].rgba[0..4], [1, 2, 3, 255]);

        assert!(
            s.apply_ping(0, &StatusResponse::default(), 57, &lang()),
            "an absent favicon clears the persisted payload"
        );
        assert!(
            s.icons().is_empty(),
            "the row must stop requesting the old key"
        );
        assert!(
            s.take_icon_upload_dirty(),
            "clearing the cache also invalidates the upload edge"
        );
    }

    #[test]
    fn deleting_the_last_row_for_an_address_clears_its_cached_favicon() {
        let mut s = screen(1);
        let status = StatusResponse {
            favicon_base64: Some(favicon_base64()),
            ..StatusResponse::default()
        };
        s.apply_ping(0, &status, 55, &lang());
        let _ = s.take_icon_upload_dirty();
        s.select(0);
        assert!(s.delete_selected().is_some());
        assert!(
            s.icons().is_empty(),
            "deleted addresses cannot keep stale icons"
        );
        assert!(s.take_icon_upload_dirty());
    }

    #[test]
    fn stored_icons_are_validated_and_cached_before_the_first_ping() {
        let mut list = ServerList::new();
        let mut valid = ServerEntry::new("valid", "valid-host");
        valid.icon_base64 = Some(favicon_base64());
        list.add(valid);
        let mut invalid = ServerEntry::new("invalid", "invalid-host");
        invalid.icon_base64 = Some("not a PNG".to_owned());
        list.add(invalid);

        let mut s = MultiplayerScreen::new(list);
        assert_eq!(s.icons().len(), 1);
        assert_eq!(s.icons()[0].address, "valid-host");
        assert!(s.servers.get(0).unwrap().icon_base64.is_some());
        assert!(s.servers.get(1).unwrap().icon_base64.is_none());
        assert!(
            s.take_servers_dirty(),
            "invalid stored icon must be re-saved as null"
        );
    }

    #[test]
    fn moving_a_row_follows_the_selection() {
        let mut s = screen(4);
        s.select(2);
        assert!(s.move_row(2, true, false));
        assert_eq!(s.selected(), Some(1));
        assert_eq!(s.servers.get(1).unwrap().name, "s2");
        assert_eq!(s.servers.get(2).unwrap().name, "s1");
        // Shift-move jumps to the end.
        assert!(s.move_row(1, false, true));
        assert_eq!(s.selected(), Some(3));
        assert_eq!(s.servers.get(3).unwrap().name, "s2");
    }

    #[test]
    fn the_first_and_last_rows_refuse_to_move_further() {
        let mut s = screen(3);
        assert!(!s.move_row(0, true, false));
        assert!(!s.move_row(2, false, false));
        assert_eq!(s.can_move(0), (false, true));
        assert_eq!(s.can_move(2), (true, false));
        assert_eq!(s.can_move(1), (true, true));
    }

    #[test]
    fn delete_add_and_edit_keep_the_list_size_in_sync() {
        let mut s = screen(2);
        s.select(0);
        assert_eq!(s.delete_selected().unwrap().name, "s0");
        assert_eq!(s.servers.len(), 1);
        assert_eq!(s.selected(), None);
        s.add(ServerEntry::new("new", "n"));
        assert_eq!(s.servers.len(), 2);
        s.select(1);
        assert!(s.replace_selected(ServerEntry::new("edited", "e")));
        assert_eq!(s.servers.get(1).unwrap().name, "edited");
        s.select(-1);
        assert!(!s.replace_selected(ServerEntry::new("x", "y")));
    }

    #[test]
    fn the_buttons_produce_the_right_actions() {
        let mut s = screen(2);
        assert_eq!(s.activate(ids::CANCEL), Action::Back);
        assert_eq!(s.activate(ids::DIRECT), Action::OpenDirectConnect);
        assert_eq!(s.activate(ids::ADD), Action::OpenAddServer);
        assert_eq!(s.activate(ids::JOIN), Action::None, "nothing selected");
        s.select(1);
        match s.activate(ids::JOIN) {
            Action::Connect(e) => assert_eq!(e.address, "host1"),
            other => panic!("{other:?}"),
        }
        match s.activate(ids::EDIT) {
            Action::OpenEditServer(e) => assert_eq!(e.name, "s1"),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            s.activate(ids::DELETE),
            Action::ConfirmDeleteServer("s1".to_owned())
        );
    }

    #[test]
    fn a_double_click_on_a_row_joins_but_a_slow_second_click_does_not() {
        let mut s = screen(2);
        // Row 0 occupies y 36..68 with the list top at 32 and no scroll.
        let rect = s.list.slot_rect(0);
        let (x, y) = (rect.x + 100, rect.y + 5);
        assert_eq!(s.mouse_clicked(x, y, 0, 1_000), Action::None);
        assert_eq!(s.selected(), Some(0));
        assert!(matches!(
            s.mouse_clicked(x, y, 0, 1_100),
            Action::Connect(_)
        ));
        assert_eq!(s.mouse_clicked(x, y, 0, 5_000), Action::None, "too slow");
    }

    #[test]
    fn clicking_the_right_half_of_the_favicon_joins_immediately() {
        let mut s = screen(2);
        let rect = s.list.slot_rect(1);
        assert!(matches!(
            s.mouse_clicked(rect.x + 24, rect.y + 5, 0, 0),
            Action::Connect(_)
        ));
        assert_eq!(s.selected(), Some(1));
    }

    #[test]
    fn clicking_the_reorder_arrows_moves_the_row_and_asks_for_a_save() {
        let mut s = screen(3);
        let rect = s.list.slot_rect(1);
        assert_eq!(
            s.mouse_clicked(rect.x + 5, rect.y + 5, 0, 0),
            Action::SaveServers
        );
        assert_eq!(s.servers.get(0).unwrap().name, "s1");
    }

    #[test]
    fn the_rightmost_columns_of_a_row_are_still_clickable() {
        // The regression this guards: `ScrollList::slot_index_at` uses GuiSlot's narrower
        // scrollbar x, which would make the last 28 columns dead.
        let s = screen(2);
        let (_, r) = s.list.content_bounds();
        let y = s.list.slot_rect(0).y + 5;
        assert_eq!(
            s.slot_index_at(r, y),
            Some(0),
            "the right edge is inclusive"
        );
        assert_eq!(
            s.list.slot_index_at(r, y),
            Some(0),
            "the server list applies its +30 scrollbar override to hit-testing too"
        );
    }

    #[test]
    fn the_permanent_lan_scan_row_extends_the_list_but_cannot_be_selected() {
        let mut servers = ServerList::new();
        servers.add(ServerEntry::new("s0", "host0"));
        servers.add(ServerEntry::new("s1", "host1"));
        let mut s = MultiplayerScreen::new(servers);
        assert_eq!(
            s.list.size, 3,
            "the constructor includes the LAN scan row before first layout"
        );
        s.init(&ctx(), &lang());
        assert_eq!(s.list.size, 3, "two servers plus the LAN scan row");
        let rect = s.list.slot_rect(2);
        assert_eq!(
            s.slot_index_at(rect.x + 10, rect.y + 5),
            None,
            "ServerListEntryLanScan.mousePressed always returns false"
        );
    }

    /// One device flooding beacons with a new `[AD]` text each time cannot
    /// grow the list (or the per-beacon duplicate scan) past `MAX_LAN_SERVERS`.
    #[test]
    fn a_lan_beacon_flood_lists_at_most_the_cap() {
        let mut s = screen(1);
        s.init(&ctx(), &lang());
        let added = (0..10_000)
            .filter(|n| s.upsert_lan(format!("Game {n}"), format!("192.0.2.7:{n}")))
            .count();
        assert_eq!(added, MAX_LAN_SERVERS);
        assert_eq!(s.lan_servers().len(), MAX_LAN_SERVERS);
        assert_eq!(s.lan_servers()[0].address, "192.0.2.7:0");
        assert_eq!(
            s.list.size,
            1 + 1 + MAX_LAN_SERVERS,
            "saved + scan + LAN rows"
        );
        // Refresh starts a new discovery session with an empty list.
        s.clear_lan_servers();
        assert!(s.upsert_lan("Back".to_owned(), "192.0.2.8:25565".to_owned()));
    }

    #[test]
    fn detected_lan_rows_follow_the_scan_dedupe_and_remain_connect_only() {
        let mut s = screen(2);
        s.init(&ctx(), &lang());
        let saved_before = s.servers.to_nbt();

        assert!(s.upsert_lan("First MOTD".to_owned(), "192.0.2.7:25565".to_owned()));
        assert!(!s.upsert_lan("Replacement MOTD".to_owned(), "192.0.2.7:25565".to_owned()));
        assert!(s.upsert_lan("Second".to_owned(), "192.0.2.8:25566".to_owned()));
        assert_eq!(s.lan_servers().len(), 2);
        assert_eq!(s.lan_servers()[0].motd, "First MOTD");
        assert_eq!(s.list.size, 5, "saved + scan + detected LAN rows");
        assert_eq!(s.row_id(0), Some(RowId::Saved(0)));
        assert_eq!(s.row_id(1), Some(RowId::Saved(1)));
        assert_eq!(s.row_id(2), Some(RowId::Scan));
        assert_eq!(s.row_id(3), Some(RowId::Lan(0)));
        assert_eq!(s.row_id(4), Some(RowId::Lan(1)));

        let scan = s.list.slot_rect(2);
        assert_eq!(s.slot_index_at(scan.x + 10, scan.y + 5), None);
        let lan = s.list.slot_rect(3);
        assert_eq!(
            s.mouse_clicked(lan.x + 100, lan.y + 5, 0, 1_000),
            Action::None
        );
        assert_eq!(s.selected(), Some(3));
        assert_eq!(s.can_move(3), (false, false));
        assert_eq!(s.activate(ids::EDIT), Action::None);
        assert_eq!(s.activate(ids::DELETE), Action::None);
        match s.activate(ids::JOIN) {
            Action::Connect(entry) => {
                assert!(entry.lan);
                assert_eq!(entry.address, "192.0.2.7:25565");
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            s.mouse_clicked(lan.x + 100, lan.y + 5, 0, 1_100),
            Action::Connect(_)
        ));
        assert_eq!(s.servers.to_nbt(), saved_before);
        assert!(!s.take_servers_dirty());
    }

    #[test]
    fn server_selection_uses_its_wider_scrollbar_override() {
        let s = screen(20);
        let bar = s.list.scrollbar().expect("enough rows to scroll");
        assert_eq!(bar.x, s.list.width / 2 + row::SCROLLBAR_OFFSET);
        assert_eq!(bar.x, 854 / 2 + 154);
    }

    #[test]
    fn shift_arrow_reorders_and_plain_arrow_moves_the_selection() {
        use crate::widget::keys;
        let mut s = screen(3);
        s.select(1);
        assert_eq!(s.key_typed(keys::UP, false), Action::None);
        assert_eq!(s.selected(), Some(0));
        s.select(1);
        assert_eq!(s.key_typed(keys::UP, true), Action::SaveServers);
        assert_eq!(s.servers.get(0).unwrap().name, "s1");
        assert_eq!(s.key_typed(keys::ESCAPE, false), Action::Back);
        s.select(0);
        assert!(matches!(
            s.key_typed(keys::ENTER, false),
            Action::Connect(_)
        ));
    }

    #[test]
    fn shift_arrow_swaps_reselects_scrolls_one_slot_and_repopulates() {
        use crate::widget::keys;
        let mut s = screen(40);
        s.select(5);
        s.list.scroll_by(100);
        let before = s.list.amount_scrolled();

        assert_eq!(s.key_typed(keys::UP, true), Action::SaveServers);
        assert_eq!(s.selected(), Some(4));
        assert_eq!(s.list.selected, 4);
        assert_eq!(s.list.amount_scrolled(), before - row::SLOT_HEIGHT as f32);
        assert_eq!(s.servers.get(4).unwrap().name, "s5");
        assert_eq!(s.servers.get(5).unwrap().name, "s4");

        assert_eq!(s.key_typed(keys::DOWN, true), Action::SaveServers);
        assert_eq!(s.selected(), Some(5));
        assert_eq!(s.list.selected, 5);
        assert_eq!(s.list.amount_scrolled(), before);
        assert_eq!(s.servers.get(4).unwrap().name, "s4");
        assert_eq!(s.servers.get(5).unwrap().name, "s5");
    }

    /// A reply is matched to its row by the row's index. Moving rows while replies are on their
    /// way must still put each server's own reply on its own row (vanilla's ping task holds the
    /// row itself), and no row may be left waiting forever.
    #[test]
    fn moving_rows_while_pings_are_pending_keeps_each_reply_on_its_own_row() {
        use crate::widget::keys;
        let mut s = screen(4);
        let reply = |address: &str| {
            StatusResponse::parse(&format!(
                r#"{{"description":"hello from {address}","version":{{"name":"1.8.9","protocol":47}},"players":{{"online":1,"max":9}}}}"#
            ))
            .expect("status")
        };
        // The live driver's rule: a reply counts only while the generation it was asked under is
        // still the screen's.
        let ask = |s: &mut MultiplayerScreen| {
            let generation = s.ping_generation();
            s.take_ping_queue()
                .into_iter()
                .map(|(index, address)| (generation, index, address))
                .collect::<Vec<_>>()
        };
        let deliver = |s: &mut MultiplayerScreen,
                       (generation, index, address): (u64, usize, String)| {
            if s.ping_generation() == generation {
                s.apply_ping(index, &reply(&address), 20, &lang());
            }
        };
        let mut pending = ask(&mut s);
        assert_eq!(pending.len(), 4);
        // One server answers before the player moves anything.
        let early = pending.remove(2);
        deliver(&mut s, early);
        // The top row goes to the bottom with Shift+Down while three replies are on their way.
        s.select(0);
        for _ in 0..3 {
            assert_eq!(s.key_typed(keys::DOWN, true), Action::SaveServers);
        }
        assert_eq!(s.servers.get(3).unwrap().address, "host0");
        // Every reply arrives, then whatever the screen asks for after them, until it is quiet.
        while !pending.is_empty() {
            for request in std::mem::take(&mut pending) {
                deliver(&mut s, request);
            }
            pending = ask(&mut s);
        }
        for entry in s.servers.entries() {
            assert_eq!(
                (entry.state, entry.motd.as_str()),
                (
                    server_data::PingState::Ok,
                    format!("hello from {}\u{a7}r", entry.address).as_str()
                ),
                "row {}",
                entry.name
            );
        }
    }

    #[test]
    fn plain_arrow_edges_clear_without_reordering() {
        use crate::widget::keys;
        let mut s = screen(3);
        s.select(0);
        assert_eq!(s.key_typed(keys::UP, false), Action::None);
        assert_eq!(s.selected(), None);
        assert_eq!(s.servers.get(0).unwrap().name, "s0");

        s.select(2);
        assert_eq!(s.key_typed(keys::DOWN, false), Action::None);
        assert_eq!(s.selected(), None);
        assert_eq!(s.servers.get(2).unwrap().name, "s2");
    }

    #[test]
    fn drawing_produces_a_background_a_title_and_one_row_per_server() {
        let mut s = screen(2);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, &ctx(), (0, 0), &lang(), 0);
        assert!(!d.is_empty());
        // Both favicons fall back to the placeholder until a ping delivers one.
        let placeholders = d
            .batches()
            .iter()
            .filter(|b| {
                matches!(&b.texture,
                crate::widget::UiTexture::Resource(r)
                    if r.path == crate::screens::common::UNKNOWN_SERVER_TEXTURE)
            })
            .count();
        assert_eq!(placeholders, 2);

        let first_icon_x = s.list.slot_rect(0).x as f32;
        let icon_batch = d
            .batches()
            .iter()
            .find(|b| {
                matches!(&b.texture,
                crate::widget::UiTexture::Resource(r)
                    if r.path == crate::screens::common::UNKNOWN_SERVER_TEXTURE)
            })
            .expect("placeholder batch");
        let min_x = d.vertices()[icon_batch.range.clone()]
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::INFINITY, f32::min);
        assert_eq!(min_x, first_icon_x, "the screen must not add a second +2");
    }

    #[test]
    fn drawing_a_selected_row_emits_vanillas_selection_outline() {
        let mut s = screen(2);
        s.select(0);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, &ctx(), (0, 0), &lang(), 0);
        let rect = s.list.slot_rect(0);
        let (left, _) = s.list.content_bounds();
        let gray = crate::widget::draw::unpack_argb(0xFF80_8080);
        assert!(d
            .vertices()
            .iter()
            .any(|v| v.pos == [left as f32, (rect.y - 2) as f32] && v.color == gray));
    }

    #[test]
    fn touchscreen_mode_shows_overlay_controls_on_unselected_rows() {
        let mut s = screen(2);
        s.set_touchscreen(true);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, &ctx(), (0, 0), &lang(), 0);
        let overlay_vertices: usize = d
            .batches()
            .iter()
            .filter(|b| {
                matches!(&b.texture,
                    crate::widget::UiTexture::Resource(r)
                        if r.path == crate::screens::common::SERVER_SELECTION_TEXTURE)
            })
            .map(|b| b.range.len())
            .sum();
        assert!(overlay_vertices > 0);
    }

    #[test]
    fn overlay_controls_follow_hover_not_the_persistent_selection() {
        let mut s = screen(2);
        s.select(0);
        let hovered = s.list.slot_rect(1);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(
            &mut d,
            &mut font,
            &ctx(),
            (hovered.x + 100, hovered.y + 5),
            &lang(),
            0,
        );
        let overlay_positions: Vec<f32> = d
            .batches()
            .iter()
            .filter(|b| {
                matches!(&b.texture,
                    crate::widget::UiTexture::Resource(r)
                        if r.path == crate::screens::common::SERVER_SELECTION_TEXTURE)
            })
            .flat_map(|b| d.vertices()[b.range.clone()].iter().map(|v| v.pos[1]))
            .collect();
        assert!(!overlay_positions.is_empty());
        assert!(
            overlay_positions.iter().all(|&y| y >= hovered.y as f32),
            "the selected-but-not-hovered first row must not receive overlay art"
        );
    }

    // ---- the version the list is shown for ----------------------------------------------------

    fn status_of(protocol: i32, name: &str) -> StatusResponse {
        StatusResponse::parse(&format!(
            r#"{{"players":{{"online":1,"max":9}},"version":{{"name":"{name}","protocol":{protocol}}}}}"#
        ))
        .expect("status response")
    }

    /// java17 `ServerListEntryNormal.drawEntry` compares with 5 where java18 compares with 47.
    #[test]
    fn a_list_shown_for_1_7_10_compares_servers_with_protocol_5() {
        let mut s = screen(3);
        s.set_client_version(GameVersion::V1_7_10);
        for entry in s.servers.entries() {
            // java17 `ServerData.<init>`'s defaults.
            assert_eq!((entry.protocol, entry.game_version.as_str()), (5, "1.7.10"));
            assert!(
                !entry.version_mismatch(),
                "not out of date before a ping answers"
            );
        }
        s.apply_ping(0, &status_of(5, "1.7.10"), 20, &lang());
        s.apply_ping(1, &status_of(47, "1.8.9"), 20, &lang());
        s.apply_ping(2, &status_of(4, "1.7.2"), 20, &lang());
        let entry = |index: usize| s.servers.get(index).expect("row");
        assert!(!entry(0).version_mismatch());
        assert_eq!(row::ping_tooltip(entry(1)), "Client out of date!");
        assert_eq!(row::ping_tooltip(entry(2)), "Server out of date!");

        // Negative control: the same answers on a 1.8.9 list.
        let mut s = screen(2);
        s.apply_ping(0, &status_of(5, "1.7.10"), 20, &lang());
        s.apply_ping(1, &status_of(47, "1.8.9"), 20, &lang());
        let entry = |index: usize| s.servers.get(index).expect("row");
        assert_eq!(row::ping_tooltip(entry(0)), "Server out of date!");
        assert!(!entry(1).version_mismatch());
    }

    #[test]
    fn a_list_shown_for_another_version_pings_its_rows_again() {
        let mut s = screen(2);
        let _ = s.take_ping_queue();
        s.apply_ping(0, &status_of(47, "1.8.9"), 20, &lang());
        s.fail_ping(1, false);
        // The rows keep their ping state between visits: shown for 1.8.9 again, nothing changes.
        let before = s.servers.clone();
        s.set_client_version(GameVersion::V1_8_9);
        assert_eq!(s.servers, before, "1.8.9 rows stay exactly as they were");
        // Shown for 1.7.10: both forget what their 1.8.9 pings said and are pinged again.
        s.set_client_version(GameVersion::V1_7_10);
        for entry in s.servers.entries() {
            assert_eq!(entry.state, server_data::PingState::Unqueried);
            assert!(entry.motd.is_empty() && entry.population.is_empty());
            assert_eq!(entry.client_version, GameVersion::V1_7_10);
        }
        s.queue_visible_pings();
        assert_eq!(s.take_ping_queue().len(), 2);
    }

    #[test]
    fn a_refresh_resets_rows_to_the_versions_own_defaults() {
        let mut s = screen(1);
        s.set_client_version(GameVersion::V1_7_10);
        s.apply_ping(0, &status_of(47, "1.8.9"), 20, &lang());
        assert!(s.servers.get(0).expect("row").version_mismatch());
        s.activate(ids::REFRESH);
        let entry = s.servers.get(0).expect("row");
        assert_eq!((entry.protocol, entry.game_version.as_str()), (5, "1.7.10"));
        assert!(!entry.version_mismatch());
    }
}
