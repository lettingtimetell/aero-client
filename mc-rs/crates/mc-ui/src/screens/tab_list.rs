//! The Tab player-list overlay.
//!
//! java: `client/gui/GuiPlayerTabOverlay.java` (399 lines), driven from `GuiIngame.java:348`
//! while `gameSettings.keyBindPlayerList.isKeyDown()`.
//!
//! # What is a port and what is not
//!
//! Ported exactly: [`sort_rows`] (`PlayerComparator`), the column split at more than 20 rows,
//! the `min(columns * (head + name + score + 13), width - 50) / columns` width formula, the
//! header/footer wrap-and-widen pass, both `Integer.MIN_VALUE` backdrops, the `553648127` row
//! plate, the spectator italic and its `-1862270977` tint, the five-bar ping sprite from
//! `icons.png` (`u = 0`, `v = 176 + bars * 8`, 10×8), and the objective score column.
//!
//! The HEARTS branch keeps Java's per-player previous-score/displayed-health state in the overlay
//! rather than `NetworkPlayerInfo`, which is the equivalent lifetime in this client. Whether a
//! head is hatted or upside down is supplied explicitly on [`TabRow`] because Java reads those two
//! flags from the loaded world entity rather than from its network player-info row.

use std::collections::HashMap;

use crate::font::FontRenderer;
use crate::widget::draw::{DrawList, UiTexture};
use crate::widget::Transform;

// Minecraft 1.7.10's player list: a child module, for this overlay's state and helpers.
#[path = "tab_list_v5.rs"]
mod v5;
pub use v5::{TabCellV5, TabGridV5};

/// Numeric ping is secondary information, smaller than the player name.
const PING_TEXT_SCALE: f32 = 0.75;

/// The default skin, used for a row whose real skin has not been fetched.
/// java: `NetworkPlayerInfo.getLocationSkin` falls back to `DefaultPlayerSkin.getDefaultSkin`.
pub const DEFAULT_SKIN_TEXTURE: &str = "textures/entity/steve.png";

/// Vanilla's cap on how many rows the overlay will draw. java: `Math.min(list.size(), 80)`
pub const MAX_ROWS: usize = 80;

/// Vanilla's per-column row cap before it splits into another column.
/// java: `for (j4 = 1; i4 > 20; i4 = (l3 + j4 - 1) / j4) ++j4;`
pub const MAX_ROWS_PER_COLUMN: usize = 20;

/// The translucent white plate behind every row. java: `553648127`
const ROW_PLATE: u32 = 553_648_127;
/// The backdrop behind the whole block, the header and the footer. java: `Integer.MIN_VALUE`
const BACKDROP: u32 = 0x8000_0000;
/// java: `drawStringWithShadow(s1, ..., -1)`
const NAME_COLOR: i32 = -1;
/// java: the spectator branch's `-1862270977`
const SPECTATOR_COLOR: i32 = -1_862_270_977;

/// How an objective's score is drawn beside each name.
/// java: `IScoreObjectiveCriteria.EnumRenderType`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabRenderType {
    /// java: `INTEGER` — a yellow number, right-aligned.
    Integer,
    /// java: `HEARTS`.
    Hearts,
}

/// The display-slot-0 objective, when the server set one.
/// java: `scoreboard.getObjectiveInDisplaySlot(0)`, passed into `renderPlayerlist`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabObjective {
    /// java: `ScoreObjective.getDisplayName`. Unused by the layout, kept because a caller that
    /// has an objective always has this and A15-02's sidebar wants the same struct.
    pub display_name: String,
    /// java: `ScoreObjective.getRenderType`.
    pub render_type: TabRenderType,
}

/// One row of the overlay. java: `NetworkPlayerInfo`, narrowed to what the overlay reads.
#[derive(Debug, Clone, PartialEq)]
pub struct TabRow {
    /// The account name. **The sort key**, and the scoreboard lookup key — not the display name.
    /// java: `networkPlayerInfo.getGameProfile().getName()`
    pub username: String,
    /// What is actually drawn: the display-name override if the server sent one, otherwise the
    /// team-formatted username. java: `GuiPlayerTabOverlay.getPlayerName`
    pub display: String,
    /// The registered name of this player's team, or `""` for none. java:
    /// `scoreplayerteam.getRegisteredName()`, with the comparator's own `: ""` for `null`.
    pub team: String,
    /// java: `getGameType() == SPECTATOR`
    pub spectator: bool,
    /// java: `getResponseTime()`. Negative means "unknown", which draws the crossed-out bar.
    pub ping_ms: i32,
    /// This player's score under the display-slot-0 objective, when there is one.
    pub score: Option<i32>,
    /// The skin sheet to take the 8×8 face out of, or `None` for [`DEFAULT_SKIN_TEXTURE`].
    pub skin: Option<UiTexture>,
    /// Whether the loaded player entity is wearing its hat layer.
    pub hat: bool,
    /// Dinnerbone and Grumm are drawn upside down while their cape model part is enabled.
    pub upside_down: bool,
}

impl TabRow {
    /// A row with nothing but a name — the shape a client with no scoreboard produces.
    #[must_use]
    pub fn new(username: impl Into<String>, ping_ms: i32) -> Self {
        let username = username.into();
        Self {
            display: username.clone(),
            username,
            team: String::new(),
            spectator: false,
            ping_ms,
            score: None,
            skin: None,
            hat: false,
            upside_down: false,
        }
    }
}

/// Sort rows the way `GuiPlayerTabOverlay.PlayerComparator` does.
///
/// `ComparisonChain.start().compareTrueFirst(notSpectator).compare(team).compare(name)`:
/// spectators last, then by the *registered team name* (`""` when the player has no team, which
/// sorts before every real team name), then by account name. The name comparison is Java's
/// `String.compareTo`, i.e. UTF-16 code-unit order. This is intentionally not `Ord for str`:
/// Rust compares UTF-8 bytes, which puts a BMP private-use character before an astral character,
/// while Java compares the astral character's high surrogate first. Team names and offline-mode
/// profile names are server-provided strings, so the otherwise unusual ordering is reachable.
// java: GuiPlayerTabOverlay.java:392
pub fn sort_rows(rows: &mut [TabRow]) {
    rows.sort_by(|a, b| {
        a.spectator
            .cmp(&b.spectator)
            .then_with(|| compare_java_string(&a.team, &b.team))
            .then_with(|| compare_java_string(&a.username, &b.username))
    });
}

/// Java 8's `String.compareTo`, which compares UTF-16 `char` values rather than Unicode scalar
/// values or UTF-8 bytes. `GuiPlayerTabOverlay.PlayerComparator` delegates both its registered
/// team and profile-name comparisons to that method.
fn compare_java_string(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// How many rows go in each column, given the total. java: the `j4`/`i4` loop.
///
/// Returns `(rows_per_column, columns)`. Vanilla increments the column count until no column
/// holds more than 20, using integer ceiling division at each step.
// java: GuiPlayerTabOverlay.java:94
#[must_use]
pub fn column_split(total: usize) -> (usize, usize) {
    let mut columns = 1usize;
    let mut per_column = total;
    while per_column > MAX_ROWS_PER_COLUMN {
        columns += 1;
        // java: `(l3 + j4 - 1) / j4`, i.e. an integer ceiling division.
        per_column = total.div_ceil(columns);
    }
    (per_column, columns)
}

/// Which of the five ping sprites a latency picks. java: `drawPing`'s ladder.
///
/// `0` is the full five-bar sprite and `5` is the crossed-out "no connection" one, which is what
/// a **negative** response time selects — servers send `-1` before they have measured a player.
// java: GuiPlayerTabOverlay.java:244
#[must_use]
pub fn ping_sprite_row(ping_ms: i32) -> i32 {
    if ping_ms < 0 {
        5
    } else if ping_ms < 150 {
        0
    } else if ping_ms < 300 {
        1
    } else if ping_ms < 600 {
        2
    } else if ping_ms < 1000 {
        3
    } else {
        4
    }
}

// The `tab_list` mod.

/// How each row shows its latency. [`TabPing::Bars`] is vanilla's five-bar icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TabPing {
    /// Vanilla's `icons.png` bars.
    #[default]
    Bars,
    /// The latency in milliseconds, coloured by the same thresholds vanilla's bars use.
    Numbers,
    /// Nothing; the column shrinks by the icon's width.
    Hidden,
}

/// The `tab_list` mod's changes to the overlay. [`TabListStyle::VANILLA`] (the default) draws
/// exactly what vanilla draws: the mod-off path never computes anything different.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TabListStyle {
    /// Draw the server's header.
    pub header: bool,
    /// Draw the server's footer.
    pub footer: bool,
    /// The latency column.
    pub ping: TabPing,
    /// Multiplier on the alpha of the dark backdrop and the row plates, `0.0..=1.0`.
    pub background: f32,
    /// ARGB colour of the outer/header/footer panels.
    pub background_color: u32,
    /// ARGB colour behind each player row.
    pub row_color: u32,
}

impl TabListStyle {
    /// Vanilla's overlay.
    pub const VANILLA: TabListStyle = TabListStyle {
        header: true,
        footer: true,
        ping: TabPing::Bars,
        background: 1.0,
        background_color: BACKDROP,
        row_color: ROW_PLATE,
    };
}

impl Default for TabListStyle {
    fn default() -> Self {
        TabListStyle::VANILLA
    }
}

/// The colour a latency is written in under [`TabPing::Numbers`]: vanilla's bar thresholds
/// (`ping_sprite_row`) mapped onto the chat palette, grey for "not measured yet".
#[must_use]
pub fn ping_number_color(ping_ms: i32) -> i32 {
    match ping_sprite_row(ping_ms) {
        0 => 0x55FF55, // green: under 150 ms (five bars)
        1 => 0xFFFF55, // yellow: under 300 ms
        2 => 0xFFAA00, // gold: under 600 ms
        3 => 0xFF5555, // red: under 1000 ms
        4 => 0xAA0000, // dark red: a second or more
        _ => 0xAAAAAA, // grey: negative = unknown
    }
}

/// The text a latency is written as under [`TabPing::Numbers`], into `buf` (no allocation).
fn ping_number_text(buf: &mut [u8; 24], ping_ms: i32) -> &str {
    if ping_ms < 0 {
        "?"
    } else {
        let len = prefixed_number(buf, "", ping_ms).len();
        buf[len..len + 2].copy_from_slice(b"ms");
        std::str::from_utf8(&buf[..len + 2]).expect("ASCII ping and ms suffix")
    }
}

/// `color`'s alpha scaled by `factor`; `color` itself, bit for bit, at `1.0`.
fn scaled_alpha(color: u32, factor: f32) -> u32 {
    if factor >= 1.0 {
        return color;
    }
    let alpha = ((color >> 24) as f32 * factor.max(0.0)).round() as u32;
    (alpha.min(255) << 24) | (color & 0x00FF_FFFF)
}

/// The overlay's own state: whether it is up, and since when.
///
/// java: `GuiPlayerTabOverlay`'s `isBeingRendered` and `lastTimeOpened`, which exist only so the
/// HEARTS column can tell "the list was just opened" from "the score changed while it was open".
#[derive(Debug, Clone, Default)]
pub struct TabListOverlay {
    /// The `tab_list` mod's changes, set by the client when its settings
    /// change. [`TabListStyle::VANILLA`] by default.
    pub style: TabListStyle,
    /// java: `lastTimeOpened` — `Minecraft.getSystemTime()` at the moment it went up.
    pub last_time_opened: u64,
    /// java: `isBeingRendered`
    pub being_rendered: bool,
    /// Most recent system time supplied by [`Self::update`].
    now_ms: u64,
    /// `GuiIngame.getUpdateCounter()`, used by the three-tick HEARTS flash cadence.
    update_counter: u64,
    /// Java stores these values on `NetworkPlayerInfo`; this renderer owns their Rust equivalent.
    hearts: HashMap<String, HeartAnimation>,
    /// The caller's revision of the rows it passes to the next [`Self::draw`]
    /// (see [`Self::set_measure_revision`]); taken by that draw. `None` measures, as before.
    measure_revision: Option<u64>,
    /// The measurements of the last drawn frame. Recomputed in place (its
    /// buffers keep their capacity) unless [`Self::measure_revision`] and every other input
    /// match, in which case they are reused as they are.
    measured: TabMeasured,
    /// Draw the rows element by element across all rows (see
    /// [`Self::set_pass_order`]); `None` follows `MC_NW_F3_PASSES` (on unless `=0`).
    pass_order: Option<bool>,
    /// The 1.7.10 list's cells, rebuilt only when their inputs change (`tab_list_v5.rs`).
    v5: v5::TabCacheV5,
}

/// The parts of one Tab row `TabListOverlay::draw` can draw separately.
const PART_PLATE: u8 = 1;
const PART_HEAD: u8 = 2;
const PART_NAME: u8 = 4;
const PART_SCORE: u8 = 8;
const PART_PING: u8 = 16;
const PART_ALL: u8 = PART_PLATE | PART_HEAD | PART_NAME | PART_SCORE | PART_PING;

/// Everything [`TabListOverlay::draw`] measures before it draws. Each value is a
/// pure function of the rows (identified by the caller's revision), the header and footer text,
/// the screen width, the style, the objective's type and the font's layout generation.
#[derive(Debug, Clone, Default)]
struct TabMeasured {
    /// `None` until measured with a caller revision: never reused.
    key: Option<(u64, u64, i32)>,
    style: TabListStyle,
    objective: Option<TabRenderType>,
    header: Option<String>,
    footer: Option<String>,
    rows: usize,
    name_width: i32,
    score_width: i32,
    ping_text_width: i32,
    /// `font.string_width` of each row's latency text (`TabPing::Numbers` only).
    ping_widths: Vec<i32>,
    /// Each spectator row's italic name (`"§o" + display`), empty for the others.
    italic: Vec<String>,
    header_lines: Vec<String>,
    header_widths: Vec<i32>,
    footer_lines: Vec<String>,
    footer_widths: Vec<i32>,
}

impl TabMeasured {
    #[allow(clippy::too_many_arguments)]
    fn matches(
        &self,
        revision: Option<u64>,
        font_generation: u64,
        width: i32,
        style: TabListStyle,
        objective: Option<TabRenderType>,
        header: Option<&str>,
        footer: Option<&str>,
        rows: usize,
    ) -> bool {
        revision.is_some_and(|revision| self.key == Some((revision, font_generation, width)))
            && self.style == style
            && self.objective == objective
            && self.header.as_deref() == header
            && self.footer.as_deref() == footer
            && self.rows == rows
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct HeartAnimation {
    last_score: i32,
    displayed_health: i32,
    last_update_ms: u64,
    flash_until_tick: u64,
    last_opened: u64,
}

impl TabListOverlay {
    /// A closed overlay.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Tell the overlay whether it will be drawn this frame.
    ///
    /// java: `updatePlayerList` — called **unconditionally** every frame from
    /// `GuiIngame.renderGameOverlay`, with `false` when the key is up. The rising edge is what
    /// stamps `lastTimeOpened`.
    // java: GuiPlayerTabOverlay.java:57
    pub fn update(&mut self, will_be_rendered: bool, now_ms: u64) {
        if will_be_rendered && !self.being_rendered {
            self.last_time_opened = now_ms;
        }
        self.being_rendered = will_be_rendered;
        self.now_ms = now_ms;
        // Compatibility for callers that do not own the game tick. The client uses
        // `update_with_counter` so tick stalls and wall-clock stalls remain distinct like Java.
        self.update_counter = now_ms / 50;
    }

    /// The revision of the rows the **next** [`Self::draw`] call receives (it
    /// applies to that one draw; set it before every draw). A caller that rebuilds its rows only
    /// when their inputs change passes a value that changes exactly then (and never repeats for
    /// other rows); the overlay then measures names, scores, latencies and the header and footer
    /// wraps once per revision instead of every frame. The header and footer text, width, style,
    /// objective and font are compared as well. Without one, the draw measures, as before.
    pub fn set_measure_revision(&mut self, revision: Option<u64>) {
        self.measure_revision = revision;
    }

    /// `Some(true)` draws the rows' plates, then heads, names, scores and
    /// latencies across all rows (fewer UI batches, the same pixels whenever the list fits its
    /// columns; an overflowing list always keeps the row order); `Some(false)` row by row;
    /// `None` (the default) follows `MC_NW_F3_PASSES`.
    pub fn set_pass_order(&mut self, passes: Option<bool>) {
        self.pass_order = passes;
    }

    /// Update the overlay with Java's exact 20 Hz `GuiIngame.updateCounter` value.
    pub fn update_with_counter(
        &mut self,
        will_be_rendered: bool,
        now_ms: u64,
        update_counter: u64,
    ) {
        self.update(will_be_rendered, now_ms);
        self.update_counter = update_counter;
    }

    /// Draw the overlay across a `width`-wide screen.
    ///
    /// `rows` must already be sorted — call [`sort_rows`] — and is truncated to [`MAX_ROWS`] the
    /// way vanilla's `subList` does. `show_heads` is vanilla's `flag`
    /// (`isIntegratedServerRunning() || getIsencrypted()`), which decides both whether a 9-pixel
    /// head column is reserved *and* whether the face is blitted.
    // java: GuiPlayerTabOverlay.java:70 (renderPlayerlist)
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        width: i32,
        rows: &[TabRow],
        header: Option<&str>,
        footer: Option<&str>,
        objective: Option<&TabObjective>,
        show_heads: bool,
    ) {
        let icons = UiTexture::icons();
        let font_height = font.font_height();

        // java: NetHandlerPlayClient.handlePlayerListHeaderFooter stores `null` when the
        // component's `getFormattedText()` is exactly empty. The sync model cannot perform that
        // test faithfully because it has no access to the client's loaded language table; its
        // empty fallback table can turn a translation that resolves to `""` into the literal
        // translation key. The caller formats with the live table before reaching this renderer,
        // so this is the first point where the exact predicate can be applied.
        //
        // Do not trim or test rendered width: `"\u{a7}r"` has no visible glyphs but has non-zero
        // Java string length, so vanilla keeps it and allocates a header/footer line.
        let style = self.style;
        let header = header.filter(|text| !text.is_empty() && style.header);
        let footer = footer.filter(|text| !text.is_empty() && style.footer);
        let backdrop = scaled_alpha(style.background_color, style.background);
        let row_plate = scaled_alpha(style.row_color, style.background);

        // Everything measured before drawing is a pure function of the inputs
        // `TabMeasured::matches` compares; with a caller revision it is reused while they match
        // (`MC_NW_F3_TAB=0` in the client never sets one: measured every frame, as before). The
        // revision applies to this draw only, so a caller that does not set one never inherits
        // another caller's.
        let measure_revision = self.measure_revision.take();
        let objective_kind = objective.map(|o| o.render_type);
        let font_generation = font.layout_generation();
        let mut measured = std::mem::take(&mut self.measured);
        if !measured.matches(
            measure_revision,
            font_generation,
            width,
            style,
            objective_kind,
            header,
            footer,
            rows.len(),
        ) {
            measured.key = measure_revision.map(|revision| (revision, font_generation, width));
            measured.style = style;
            measured.objective = objective_kind;
            measured.rows = rows.len();
            // Only a reusable measurement remembers the text it was taken for.
            let cached = measured.key.is_some();
            for (slot, text) in [
                (&mut measured.header, header),
                (&mut measured.footer, footer),
            ] {
                match text.filter(|_| cached) {
                    Some(text) => {
                        let copy = slot.get_or_insert_with(String::new);
                        copy.clear();
                        copy.push_str(text);
                    }
                    None => *slot = None,
                }
            }
            // java: the first pass measures the widest name (`i`) and the widest " <score>" (`j`).
            let mut name_width = 0;
            let mut score_width = 0;
            // X3: the widest latency text, only measured when the mod writes numbers.
            let mut ping_text_width = 0;
            measured.ping_widths.clear();
            measured.italic.resize(rows.len(), String::new());
            for (row, italic) in rows.iter().zip(measured.italic.iter_mut()) {
                name_width = name_width.max(font.string_width(&row.display));
                if let Some(objective) = objective {
                    if objective.render_type != TabRenderType::Hearts {
                        let mut buf = [0u8; 24];
                        let text = prefixed_number(&mut buf, " ", row.score.unwrap_or(0));
                        score_width = score_width.max(font.string_width(text));
                    }
                }
                if style.ping == TabPing::Numbers {
                    let mut buf = [0u8; 24];
                    let text = ping_number_text(&mut buf, row.ping_ms);
                    let text_width = font.string_width(text);
                    ping_text_width =
                        ping_text_width.max((text_width as f32 * PING_TEXT_SCALE).ceil() as i32);
                    measured.ping_widths.push(text_width);
                }
                // java: `EnumChatFormatting.ITALIC + s1` — the code is prefixed, not replaced.
                italic.clear();
                if row.spectator {
                    italic.push_str("\u{a7}o");
                    italic.push_str(&row.display);
                }
            }
            measured.name_width = name_width;
            measured.score_width = score_width;
            measured.ping_text_width = ping_text_width;
            for (text, lines, widths) in [
                (
                    header,
                    &mut measured.header_lines,
                    &mut measured.header_widths,
                ),
                (
                    footer,
                    &mut measured.footer_lines,
                    &mut measured.footer_widths,
                ),
            ] {
                lines.clear();
                widths.clear();
                if let Some(text) = text {
                    for line in font.wrap_to_lines(text, width - 50) {
                        widths.push(font.string_width(&line));
                        lines.push(line);
                    }
                }
            }
        }
        let (name_width, score_width, ping_text_width) = (
            measured.name_width,
            measured.score_width,
            measured.ping_text_width,
        );
        // java: the `+ 13` in the column width is a 1-pixel gap after the name, a 1-pixel gap
        // before the 10-pixel ping icon and a 1-pixel right margin. Numbers replace the icon's
        // 10 pixels with the widest number; hidden pings drop the icon and its gap.
        let ping_reserve = match style.ping {
            TabPing::Bars => 13,
            TabPing::Numbers => 3 + ping_text_width,
            TabPing::Hidden => 2,
        };

        // Java truncates a sorted *view* of the player-info map. The per-player health fields
        // remain alive on every NetworkPlayerInfo, including rows currently beyond the 80-row
        // render cap, so keep the complete slice for the equivalent retention pass below.
        let all_rows = rows;
        let rows = &all_rows[..all_rows.len().min(MAX_ROWS)];
        let total = rows.len();
        let (per_column, columns) = column_split(total);
        // java: `l` — the width reserved right of the name for the objective column.
        let extra = match objective {
            Some(o) if o.render_type == TabRenderType::Hearts => 90,
            Some(_) => score_width,
            None => 0,
        };
        let head = i32::from(show_heads) * 9;
        let column_width = (columns as i32 * (head + name_width + extra + ping_reserve))
            .min(width - 50)
            / columns as i32;
        let left = width / 2 - (column_width * columns as i32 + (columns as i32 - 1) * 5) / 2;
        let mut y = 10;
        // java: `l1` starts at the block width and is widened by any header/footer line.
        let mut block_width = column_width * columns as i32 + (columns as i32 - 1) * 5;

        // The header's and footer's wrapped lines and their widths were measured above (empty
        // without a header/footer).
        for &line_width in measured.header_widths.iter().chain(&measured.footer_widths) {
            block_width = block_width.max(line_width);
        }

        if header.is_some() {
            let lines = &measured.header_lines;
            d.rect(
                width / 2 - block_width / 2 - 1,
                y - 1,
                width / 2 + block_width / 2 + 1,
                y + lines.len() as i32 * font_height,
                backdrop,
            );
            for (line, &line_width) in lines.iter().zip(&measured.header_widths) {
                font.draw_string_with_shadow(
                    d,
                    line,
                    (width / 2 - line_width / 2) as f32,
                    y as f32,
                    NAME_COLOR,
                );
                y += font_height;
            }
            y += 1;
        }

        d.rect(
            width / 2 - block_width / 2 - 1,
            y - 1,
            width / 2 + block_width / 2 + 1,
            y + per_column as i32 * 9,
            backdrop,
        );

        // One row's elements, those in `parts` (`ROW_*` bits): its plate, head, name, score and
        // latency, in vanilla's order when all are asked for at once.
        let (last_time_opened, now_ms, update_counter) =
            (self.last_time_opened, self.now_ms, self.update_counter);
        let draw_row = |d: &mut DrawList,
                        font: &mut FontRenderer,
                        hearts: &mut HashMap<String, HeartAnimation>,
                        index: usize,
                        row: &TabRow,
                        parts: u8| {
            let column = index / per_column.max(1);
            let slot = index % per_column.max(1);
            let mut x = left + column as i32 * column_width + column as i32 * 5;
            let row_y = y + slot as i32 * 9;
            if parts & PART_PLATE != 0 {
                d.rect(x, row_y, x + column_width, row_y + 8, row_plate);
            }

            let name_x_of_ping = x;
            if show_heads {
                if parts & PART_HEAD != 0 {
                    // java: `drawScaledCustomSizeModalRect(j2, k2, 8, 8, 8, 8, 8, 8, 64, 64)` —
                    // the face at (8,8) of a 64×64 skin, then the hat overlay at (40,8).
                    // Borrow the row's key rather than cloning it (two `String`s per row per
                    // frame).
                    let default_skin;
                    let skin = match &row.skin {
                        Some(skin) => skin,
                        None => {
                            default_skin = UiTexture::gui(DEFAULT_SKIN_TEXTURE);
                            &default_skin
                        }
                    };
                    draw_head_layer(d, skin, x, row_y, 8.0, row.upside_down);
                    if row.hat {
                        draw_head_layer(d, skin, x, row_y, 40.0, row.upside_down);
                    }
                }
                x += 9;
            }

            if parts & PART_NAME != 0 {
                if row.spectator {
                    // java: `EnumChatFormatting.ITALIC + s1`, built when measuring
                    // (`measured.italic`).
                    let italic = &measured.italic[index];
                    font.draw_string_with_shadow(
                        d,
                        italic,
                        x as f32,
                        row_y as f32,
                        SPECTATOR_COLOR,
                    );
                } else {
                    font.draw_string_with_shadow(
                        d,
                        &row.display,
                        x as f32,
                        row_y as f32,
                        NAME_COLOR,
                    );
                }
            }

            // java: the objective column is skipped for spectators.
            if parts & PART_SCORE != 0 {
                if let (Some(objective), false) = (objective, row.spectator) {
                    let score_left = x + name_width + 1;
                    let score_right = score_left + extra;
                    if score_right - score_left > 5 {
                        let heart_state = (objective.render_type == TabRenderType::Hearts)
                            .then(|| hearts.entry(row.username.clone()).or_default());
                        draw_score(
                            d,
                            font,
                            objective,
                            row.score.unwrap_or(0),
                            row_y,
                            score_left,
                            score_right,
                            heart_state,
                            last_time_opened,
                            now_ms,
                            update_counter,
                        );
                    }
                }
            }

            // java: `drawPing(i1, j2 - (flag ? 9 : 0), k2, ...)` — the ping is placed against the
            // *row's* left edge, i.e. before the head shifted `j2` along.
            if parts & PART_PING == 0 {
                return;
            }
            match style.ping {
                TabPing::Bars => d.blit(
                    icons,
                    name_x_of_ping + column_width - 11,
                    row_y,
                    0,
                    176 + ping_sprite_row(row.ping_ms) * 8,
                    10,
                    8,
                ),
                // X3: right-aligned against the same 1-pixel margin as the icon.
                TabPing::Numbers => {
                    let mut buf = [0u8; 24];
                    let text = ping_number_text(&mut buf, row.ping_ms);
                    // `font.string_width(text)`, measured with the rows.
                    let text_x = (name_x_of_ping + column_width - 1) as f32
                        - measured.ping_widths[index] as f32 * PING_TEXT_SCALE;
                    let text_y = row_y as f32 + (8.0 - 8.0 * PING_TEXT_SCALE) * 0.5;
                    d.with_transform(
                        Transform::translate(text_x, text_y).then_scale(PING_TEXT_SCALE),
                        |d| {
                            font.draw_string_with_shadow(
                                d,
                                text,
                                0.0,
                                0.0,
                                ping_number_color(row.ping_ms),
                            );
                        },
                    );
                }
                TabPing::Hidden => {}
            }
        };
        // When every row's elements fit their column (the column width is not
        // capped by the screen), no element overlaps another row, column or element of its row
        // except its own plate: the name ends (shadow included) where the score column starts, the
        // score where the latency starts, rows are 9 px apart for 8 px plates and 9 px text, and
        // columns 5 px apart. Drawing all plates, then all heads, names, scores and latencies is
        // then the same picture with ~5 batches instead of ~2-4 per row (name/latency texture
        // switches). A capped (overflowing) list keeps vanilla's row order. `MC_NW_F3_PASSES=0`
        // (or `set_pass_order(Some(false))`) turns it off.
        let fits = columns as i32 * (head + name_width + extra + ping_reserve) <= width - 50;
        let passes = fits
            && self
                .pass_order
                .unwrap_or_else(crate::screens::hud::nw_f3_passes_enabled);
        if passes {
            for parts in [PART_PLATE, PART_HEAD, PART_NAME, PART_SCORE, PART_PING] {
                for (index, row) in rows.iter().enumerate() {
                    draw_row(d, font, &mut self.hearts, index, row, parts);
                }
            }
        } else {
            for (index, row) in rows.iter().enumerate() {
                draw_row(d, font, &mut self.hearts, index, row, PART_ALL);
            }
        }

        if footer.is_some() {
            let lines = &measured.footer_lines;
            let mut y = y + per_column as i32 * 9 + 1;
            d.rect(
                width / 2 - block_width / 2 - 1,
                y - 1,
                width / 2 + block_width / 2 + 1,
                y + lines.len() as i32 * font_height,
                backdrop,
            );
            for (line, &line_width) in lines.iter().zip(&measured.footer_widths) {
                font.draw_string_with_shadow(
                    d,
                    line,
                    (width / 2 - line_width / 2) as f32,
                    y as f32,
                    NAME_COLOR,
                );
                y += font_height;
            }
        }

        // Java owns this state on `NetworkPlayerInfo`, whose removal drops it immediately.
        self.hearts
            .retain(|username, _| all_rows.iter().any(|row| row.username == *username));
        self.measured = measured;
    }
}

/// `prefix` followed by the decimal `value`, written into `buf`: the Tab list measures and draws
/// one score string per row per frame, and this keeps that off the heap. Same bytes as
/// `format!("{prefix}{value}")`.
fn prefixed_number<'a>(buf: &'a mut [u8; 24], prefix: &str, value: i32) -> &'a str {
    use std::io::Write as _;
    let len = {
        let mut cursor = std::io::Cursor::new(&mut buf[..]);
        write!(cursor, "{prefix}{value}").expect("a short prefix and any i32 fit in 24 bytes");
        cursor.position() as usize
    };
    std::str::from_utf8(&buf[..len]).expect("UTF-8 prefix followed by ASCII digits")
}

/// Draw one 8×8 face/hat layer. Java flips Dinnerbone/Grumm by giving the sampled source a
/// negative height; reversing the V coordinates is the equivalent operation for our quad API.
fn draw_head_layer(d: &mut DrawList, skin: &UiTexture, x: i32, y: i32, u: f32, upside_down: bool) {
    let (v0, v1) = if upside_down {
        (16.0 / 64.0, 8.0 / 64.0)
    } else {
        (8.0 / 64.0, 16.0 / 64.0)
    };
    d.quad_rect(
        skin,
        (x as f32, y as f32, (x + 8) as f32, (y + 8) as f32),
        (u / 64.0, v0, (u + 8.0) / 64.0, v1),
        0xFFFF_FFFF,
    );
}

/// The objective column beside one name.
// java: GuiPlayerTabOverlay.java:274 (drawScoreboardValues)
// Keep the Java helper's explicit layout/state arguments; the call sites intentionally pass the
// live heart animation and timing values independently.
#[allow(clippy::too_many_arguments)]
fn draw_score(
    d: &mut DrawList,
    font: &mut FontRenderer,
    objective: &TabObjective,
    score: i32,
    y: i32,
    left: i32,
    right: i32,
    heart_state: Option<&mut HeartAnimation>,
    last_time_opened: u64,
    now_ms: u64,
    update_counter: u64,
) {
    match objective.render_type {
        TabRenderType::Integer => {
            // java: `EnumChatFormatting.YELLOW + "" + i`, right-aligned against `p_175247_5_`.
            let mut buf = [0u8; 24];
            let text = prefixed_number(&mut buf, "\u{a7}e", score);
            let x = right - font.string_width(text);
            font.draw_string_with_shadow(d, text, x as f32, y as f32, 0x00FF_FFFF);
        }
        TabRenderType::Hearts => {
            let state = heart_state.expect("HEARTS score must carry animation state");
            let tick = update_counter;

            if last_time_opened == state.last_opened {
                if score < state.last_score {
                    state.last_update_ms = now_ms;
                    state.flash_until_tick = tick + 20;
                } else if score > state.last_score {
                    state.last_update_ms = now_ms;
                    state.flash_until_tick = tick + 10;
                }
            }

            if now_ms.saturating_sub(state.last_update_ms) > 1_000
                || last_time_opened != state.last_opened
            {
                state.last_score = score;
                state.displayed_health = score;
                state.last_update_ms = now_ms;
            }
            state.last_opened = last_time_opened;
            state.last_score = score;

            let visible_hearts = ((score.max(state.displayed_health) as f32) / 2.0).ceil() as i32;
            let slots = (score / 2).max(state.displayed_health / 2).max(10);
            let flash =
                state.flash_until_tick > tick && ((state.flash_until_tick - tick) / 3) % 2 == 1;
            if visible_hearts <= 0 {
                return;
            }

            let spacing = ((right - left - 4) as f32 / slots as f32).min(9.0);
            if spacing > 3.0 {
                let icons = UiTexture::icons();
                for index in visible_hearts..slots {
                    draw_icon_f32(
                        d,
                        icons,
                        left as f32 + index as f32 * spacing,
                        y as f32,
                        if flash { 25 } else { 16 },
                    );
                }
                for index in 0..visible_hearts {
                    let x = left as f32 + index as f32 * spacing;
                    draw_icon_f32(d, icons, x, y as f32, if flash { 25 } else { 16 });
                    if flash {
                        if index * 2 + 1 < state.displayed_health {
                            draw_icon_f32(d, icons, x, y as f32, 70);
                        } else if index * 2 + 1 == state.displayed_health {
                            draw_icon_f32(d, icons, x, y as f32, 79);
                        }
                    }
                    if index * 2 + 1 < score {
                        draw_icon_f32(d, icons, x, y as f32, if index >= 10 { 160 } else { 52 });
                    } else if index * 2 + 1 == score {
                        draw_icon_f32(d, icons, x, y as f32, if index >= 10 { 169 } else { 61 });
                    }
                }
            } else {
                let ratio = (score as f32 / 20.0).clamp(0.0, 1.0);
                let color =
                    (((1.0 - ratio) * 255.0) as i32) << 16 | (((ratio * 255.0) as i32) << 8);
                let mut text = health_text(score);
                if right - font.string_width(&format!("{text}hp")) >= left {
                    text.push_str("hp");
                }
                let x = (right + left) / 2 - font.string_width(&text) / 2;
                font.draw_string_with_shadow(d, &text, x as f32, y as f32, color);
            }
        }
    }
}

fn draw_icon_f32(d: &mut DrawList, icons: &UiTexture, x: f32, y: f32, u: i32) {
    d.quad_rect(
        icons,
        (x, y, x + 9.0, y + 9.0),
        (u as f32 / 256.0, 0.0, (u + 9) as f32 / 256.0, 9.0 / 256.0),
        0xFFFF_FFFF,
    );
}

fn health_text(score: i32) -> String {
    java_float_text(score as f32 / 2.0)
}

/// Spell a finite `float` the way Java's `Float.toString` does. Rust and Java both begin with a
/// shortest round-tripping decimal, but Java switches to scientific notation below `1e-3` and at
/// `1e7`, always retains a decimal point, and uses an uppercase `E` without a plus sign.
fn java_float_text(value: f32) -> String {
    let raw = value.to_string();
    let (sign, raw) = raw
        .strip_prefix('-')
        .map_or(("", raw.as_str()), |unsigned| ("-", unsigned));
    let (mantissa, explicit_exponent) = raw
        .split_once(['e', 'E'])
        .map_or((raw, 0), |(digits, exponent)| {
            (digits, exponent.parse::<i32>().expect("f32 exponent"))
        });
    let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let mut digits: String = mantissa.chars().filter(|&ch| ch != '.').collect();
    let leading_zeroes = digits.bytes().take_while(|&byte| byte == b'0').count();
    let decimal_position = point + explicit_exponent - leading_zeroes as i32;
    digits.drain(..leading_zeroes);

    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }

    let scientific_exponent = decimal_position - 1;
    if (-3..7).contains(&scientific_exponent) {
        if decimal_position <= 0 {
            format!(
                "{sign}0.{}{digits}",
                "0".repeat((-decimal_position) as usize)
            )
        } else {
            let decimal_position = decimal_position as usize;
            if decimal_position >= digits.len() {
                format!(
                    "{sign}{digits}{}.0",
                    "0".repeat(decimal_position - digits.len())
                )
            } else {
                format!(
                    "{sign}{}.{}",
                    &digits[..decimal_position],
                    &digits[decimal_position..]
                )
            }
        }
    } else {
        let fraction = if digits.len() == 1 { "0" } else { &digits[1..] };
        format!("{sign}{}.{fraction}E{scientific_exponent}", &digits[..1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::draw::ICONS_TEXTURE;
    use crate::widget::test_support::uniform_font;

    #[test]
    fn prefixed_number_matches_format_for_both_score_prefixes() {
        let mut buf = [0u8; 24];
        for prefix in [" ", "\u{a7}e"] {
            for value in [0, 7, -7, 1_000, i32::MAX, i32::MIN] {
                assert_eq!(
                    prefixed_number(&mut buf, prefix, value),
                    format!("{prefix}{value}")
                );
            }
        }
    }

    /// `DrawList` is non-indexed: [`DrawList::quad`] pushes two triangles, i.e. six vertices, and
    /// [`DrawBatch::range`] is a range over *vertices*. Counting quads therefore divides by six —
    /// and it has to be done per batch, because adjacent same-texture draws merge into one.
    const QUAD_VERTICES: usize = 6;

    fn row(name: &str, ping: i32) -> TabRow {
        TabRow::new(name, ping)
    }

    /// java: `compareTrueFirst(notSpectator)` then team then name.
    #[test]
    fn spectators_sink_and_the_rest_sort_by_team_then_name() {
        let mut rows = vec![
            row("zed", 0),
            {
                let mut r = row("aaron", 0);
                r.spectator = true;
                r
            },
            {
                let mut r = row("bob", 0);
                r.team = "red".into();
                r
            },
            row("alice", 0),
        ];
        sort_rows(&mut rows);
        let names: Vec<&str> = rows.iter().map(|r| r.username.as_str()).collect();
        assert_eq!(
            names,
            ["alice", "zed", "bob", "aaron"],
            "no team sorts before \"red\"; the spectator is last regardless of name"
        );
    }

    /// Java's `String.compareTo` compares UTF-16 code units. Rust's UTF-8 byte order gets the
    /// astral-vs-BMP boundary backwards, so this negative control must fail if either comparator
    /// in `PlayerComparator` regresses to `str::cmp`.
    #[test]
    fn tab_sort_uses_java_utf16_order_for_server_team_and_profile_names() {
        use std::cmp::Ordering;

        let astral = "\u{10000}";
        let bmp_private_use = "\u{e000}";
        assert_eq!(
            compare_java_string(astral, bmp_private_use),
            Ordering::Less,
            "Java compares the astral high surrogate (D800) before U+E000"
        );
        assert_eq!(
            compare_java_string(bmp_private_use, astral),
            Ordering::Greater,
            "the reverse comparison is the biting sign control"
        );

        let mut names = vec![row(bmp_private_use, 0), row(astral, 0)];
        sort_rows(&mut names);
        assert_eq!(
            names
                .iter()
                .map(|row| row.username.as_str())
                .collect::<Vec<_>>(),
            vec![astral, bmp_private_use]
        );

        let mut teams = vec![
            {
                let mut row = row("bmp", 0);
                row.team = bmp_private_use.to_owned();
                row
            },
            {
                let mut row = row("astral", 0);
                row.team = astral.to_owned();
                row
            },
        ];
        sort_rows(&mut teams);
        assert_eq!(
            teams
                .iter()
                .map(|row| row.team.as_str())
                .collect::<Vec<_>>(),
            vec![astral, bmp_private_use]
        );
    }

    /// java: the `for (j4 = 1; i4 > 20; i4 = (l3 + j4 - 1) / j4) ++j4;` loop, which is what makes
    /// a 21-player server suddenly show two columns of 11.
    #[test]
    fn the_list_splits_into_columns_above_twenty_rows() {
        assert_eq!(column_split(0), (0, 1));
        assert_eq!(column_split(1), (1, 1));
        assert_eq!(column_split(20), (20, 1));
        assert_eq!(column_split(21), (11, 2));
        assert_eq!(column_split(40), (20, 2));
        assert_eq!(column_split(41), (14, 3));
        assert_eq!(column_split(80), (20, 4));
        // Every split really does hold everything, and no column exceeds the cap.
        for total in 0..=MAX_ROWS {
            let (per, cols) = column_split(total);
            assert!(per <= MAX_ROWS_PER_COLUMN, "{total}");
            assert!(per * cols >= total, "{total}: {per} x {cols}");
        }
    }

    /// java: `drawPing`'s ladder, including the negative "not measured yet" case that servers
    /// send for a player who has just joined.
    #[test]
    fn the_ping_ladder_matches_vanillas_thresholds() {
        assert_eq!(ping_sprite_row(-1), 5);
        assert_eq!(ping_sprite_row(0), 0);
        assert_eq!(ping_sprite_row(149), 0);
        assert_eq!(ping_sprite_row(150), 1);
        assert_eq!(ping_sprite_row(299), 1);
        assert_eq!(ping_sprite_row(300), 2);
        assert_eq!(ping_sprite_row(599), 2);
        assert_eq!(ping_sprite_row(600), 3);
        assert_eq!(ping_sprite_row(999), 3);
        assert_eq!(ping_sprite_row(1000), 4);
    }

    /// java: `updatePlayerList` stamps `lastTimeOpened` only on the rising edge — holding Tab
    /// must not restamp it every frame, which is exactly what the HEARTS column relies on.
    #[test]
    fn the_open_time_is_stamped_once_per_press() {
        let mut overlay = TabListOverlay::new();
        overlay.update(true, 1_000);
        assert!(overlay.being_rendered);
        assert_eq!(overlay.last_time_opened, 1_000);
        overlay.update(true, 2_000);
        assert_eq!(overlay.last_time_opened, 1_000, "held, not re-opened");
        overlay.update(false, 3_000);
        assert!(!overlay.being_rendered);
        overlay.update(true, 4_000);
        assert_eq!(overlay.last_time_opened, 4_000);
    }

    /// Java's `updatePlayerList` only stamps the overlay's open time. HEARTS fields live on
    /// `NetworkPlayerInfo` and are reset by `drawScoreboardValues`, not by the rising edge. The
    /// UI must therefore leave its per-row cache untouched here; clearing it would erase a flash
    /// on a same-millisecond reopen, and the client driver already replaces this overlay at the
    /// ordinary session-unload boundary.
    #[test]
    fn opening_the_overlay_does_not_clear_per_row_health_state_before_draw() {
        let mut overlay = TabListOverlay::new();
        overlay.hearts.insert(
            "same-name".into(),
            HeartAnimation {
                displayed_health: 20,
                last_opened: 10,
                ..HeartAnimation::default()
            },
        );

        overlay.update(true, 1_000);
        assert_eq!(
            overlay
                .hearts
                .get("same-name")
                .map(|state| state.displayed_health),
            Some(20),
            "updatePlayerList only stamps lastTimeOpened; drawScoreboardValues owns row reset"
        );
        assert_eq!(overlay.last_time_opened, 1_000);

        overlay.update(true, 2_000);
        assert_eq!(
            overlay
                .hearts
                .get("same-name")
                .map(|state| state.displayed_health),
            Some(20),
            "holding Tab must preserve the in-progress row state"
        );
    }

    /// The whole overlay must actually emit geometry: a backdrop, one plate per row, and one
    /// ping sprite per row from `icons.png`.
    #[test]
    fn drawing_emits_a_plate_and_a_ping_sprite_for_every_row() {
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let mut overlay = TabListOverlay::new();
        let rows: Vec<TabRow> = (0..3).map(|i| row(&format!("p{i}"), i * 200)).collect();
        overlay.draw(&mut d, &mut font, 854, &rows, None, None, None, false);

        let icon_quads: usize = d
            .batches()
            .iter()
            .filter(|b| matches!(&b.texture, UiTexture::Resource(r) if r.path == ICONS_TEXTURE))
            .map(|b| b.range.len() / QUAD_VERTICES)
            .sum();
        assert_eq!(icon_quads, 3, "one ping sprite per row");

        let solid_quads: usize = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Solid))
            .map(|b| b.range.len() / QUAD_VERTICES)
            .sum();
        assert_eq!(solid_quads, 4, "one backdrop plus one plate per row");
    }

    /// The header and footer widen the backdrop and are drawn centred, which is the whole reason
    /// servers use them for branding.
    #[test]
    fn a_header_and_footer_are_wrapped_drawn_and_widen_the_backdrop() {
        let mut font = uniform_font();
        let rows = vec![row("a", 0)];

        let mut narrow = DrawList::new();
        TabListOverlay::new().draw(&mut narrow, &mut font, 854, &rows, None, None, None, false);
        let mut wide = DrawList::new();
        TabListOverlay::new().draw(
            &mut wide,
            &mut font,
            854,
            &rows,
            Some("a very long header line indeed that is wider than one short row"),
            Some("footer"),
            None,
            false,
        );

        let solid = |d: &DrawList| -> usize {
            d.batches()
                .iter()
                .filter(|b| matches!(b.texture, UiTexture::Solid))
                .map(|b| b.range.len() / QUAD_VERTICES)
                .sum()
        };
        assert_eq!(solid(&narrow), 2, "backdrop + one row plate");
        assert_eq!(
            solid(&wide),
            4,
            "header backdrop + block backdrop + row plate + footer backdrop"
        );
        // The glyph batches must have grown: the header and footer text is really drawn.
        let glyphs = |d: &DrawList| -> usize {
            d.batches()
                .iter()
                .filter(|b| matches!(b.texture, UiTexture::Font(_)))
                .map(|b| b.range.len() / QUAD_VERTICES)
                .sum()
        };
        assert!(glyphs(&wide) > glyphs(&narrow));
    }

    /// `handlePlayerListHeaderFooter` turns a component whose formatted text has Java length zero
    /// into `null`. A formatting-only string is deliberately different: `"§r".length() == 2`,
    /// so vanilla keeps it and still reserves a line even though it emits no glyph quad.
    #[test]
    fn exactly_empty_header_and_footer_are_absent_but_formatting_only_is_present() {
        let rows = vec![row("a", 0)];
        let solid_quads = |header: Option<&str>, footer: Option<&str>| {
            let mut font = uniform_font();
            let mut d = DrawList::new();
            TabListOverlay::new().draw(&mut d, &mut font, 854, &rows, header, footer, None, false);
            d.batches()
                .iter()
                .filter(|batch| matches!(batch.texture, UiTexture::Solid))
                .map(|batch| batch.range.len() / QUAD_VERTICES)
                .sum::<usize>()
        };

        assert_eq!(solid_quads(None, None), 2, "main backdrop + row plate");
        assert_eq!(
            solid_quads(Some(""), Some("")),
            2,
            "empty packet components must not add backdrops or shift the rows"
        );
        assert_eq!(
            solid_quads(Some("\u{a7}r"), None),
            3,
            "non-empty formatting codes survive Java's length gate"
        );
    }

    /// The objective column is skipped for spectators and right-aligns a yellow number for
    /// everybody else. java: the `networkplayerinfo1.getGameType() != SPECTATOR` guard at `:208`.
    #[test]
    fn the_objective_column_is_drawn_for_players_and_skipped_for_spectators() {
        let mut font = uniform_font();
        let objective = TabObjective {
            display_name: "kills".into(),
            render_type: TabRenderType::Integer,
        };
        let mut glyphs = |rows: &[TabRow]| -> usize {
            let mut d = DrawList::new();
            TabListOverlay::new().draw(
                &mut d,
                &mut font,
                854,
                rows,
                None,
                None,
                Some(&objective),
                false,
            );
            d.batches()
                .iter()
                .filter(|b| matches!(b.texture, UiTexture::Font(_)))
                .map(|b| b.range.len() / QUAD_VERTICES)
                .sum()
        };
        let mut player = row("bob", 0);
        player.score = Some(1234);
        let mut spectator = player.clone();
        spectator.spectator = true;
        assert!(
            glyphs(std::slice::from_ref(&player)) > glyphs(std::slice::from_ref(&spectator)),
            "the spectator row draws no score"
        );
    }

    /// Vanilla truncates to 80 rows with `subList` before it draws anything.
    #[test]
    fn no_more_than_eighty_rows_are_drawn() {
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let rows: Vec<TabRow> = (0..120).map(|i| row(&format!("p{i}"), 0)).collect();
        TabListOverlay::new().draw(&mut d, &mut font, 854, &rows, None, None, None, false);
        let icon_quads: usize = d
            .batches()
            .iter()
            .filter(|b| matches!(&b.texture, UiTexture::Resource(r) if r.path == ICONS_TEXTURE))
            .map(|b| b.range.len() / QUAD_VERTICES)
            .sum();
        assert_eq!(icon_quads, MAX_ROWS);
    }

    /// Java concatenates a `float`, whose textual form keeps the decimal point for whole halves.
    /// The odd-score cases are the negative control: they already agreed before this fix.
    #[test]
    fn narrow_health_text_keeps_java_float_spelling() {
        assert_eq!(health_text(0), "0.0");
        assert_eq!(health_text(1), "0.5");
        assert_eq!(health_text(7), "3.5");
        assert_eq!(health_text(20), "10.0");
        assert_eq!(health_text(40), "20.0");
        assert_eq!(health_text(19_999_998), "9999999.0");
        assert_eq!(health_text(20_000_000), "1.0E7");
        assert_eq!(health_text(i32::MAX), "1.0737418E9");
        assert_eq!(health_text(i32::MIN), "-1.0737418E9");
    }

    /// The 80-row cap is a rendering sub-list, not a lifetime boundary for NetworkPlayerInfo's
    /// health animation fields. A hidden row may sort back into view later and must retain them.
    #[test]
    fn health_state_survives_beyond_the_render_cap_until_the_player_is_removed() {
        let mut overlay = TabListOverlay::new();
        overlay.hearts.insert(
            "p080".into(),
            HeartAnimation {
                displayed_health: 20,
                ..HeartAnimation::default()
            },
        );
        let mut rows: Vec<TabRow> = (0..=MAX_ROWS)
            .map(|index| row(&format!("p{index:03}"), 0))
            .collect();
        let mut font = uniform_font();

        overlay.draw(
            &mut DrawList::new(),
            &mut font,
            854,
            &rows,
            None,
            None,
            None,
            false,
        );
        assert_eq!(
            overlay
                .hearts
                .get("p080")
                .map(|state| state.displayed_health),
            Some(20),
            "row 81 still exists even though the render sub-list omits it"
        );

        rows.pop();
        overlay.draw(
            &mut DrawList::new(),
            &mut font,
            854,
            &rows,
            None,
            None,
            None,
            false,
        );
        assert!(
            !overlay.hearts.contains_key("p080"),
            "removing the player-info row is the actual lifetime boundary"
        );
    }

    #[test]
    fn hearts_render_as_icon_layers_and_zero_health_draws_nothing() {
        let objective = TabObjective {
            display_name: "health".into(),
            render_type: TabRenderType::Hearts,
        };
        let mut font = uniform_font();
        let mut state = HeartAnimation::default();
        let mut hearts = DrawList::new();
        draw_score(
            &mut hearts,
            &mut font,
            &objective,
            20,
            0,
            0,
            90,
            Some(&mut state),
            1_000,
            1_000,
            20,
        );
        assert_eq!(
            hearts.vertices().len() / QUAD_VERTICES,
            20,
            "ten containers plus ten full-heart layers"
        );
        assert!(hearts.batches().iter().all(
            |batch| matches!(&batch.texture, UiTexture::Resource(r) if r.path == ICONS_TEXTURE)
        ));

        let mut dead = DrawList::new();
        let mut dead_state = HeartAnimation::default();
        draw_score(
            &mut dead,
            &mut font,
            &objective,
            0,
            0,
            0,
            90,
            Some(&mut dead_state),
            1_000,
            1_000,
            20,
        );
        assert!(dead.is_empty(), "Java's `j > 0` guard draws no 0hp text");
    }

    #[test]
    fn damage_retains_the_previous_health_and_enters_the_flash_phase() {
        let objective = TabObjective {
            display_name: "health".into(),
            render_type: TabRenderType::Hearts,
        };
        let mut font = uniform_font();
        let mut state = HeartAnimation::default();
        draw_score(
            &mut DrawList::new(),
            &mut font,
            &objective,
            20,
            0,
            0,
            90,
            Some(&mut state),
            1_000,
            1_000,
            20,
        );
        draw_score(
            &mut DrawList::new(),
            &mut font,
            &objective,
            10,
            0,
            0,
            90,
            Some(&mut state),
            1_000,
            1_050,
            21,
        );
        let mut flashing = DrawList::new();
        draw_score(
            &mut flashing,
            &mut font,
            &objective,
            10,
            0,
            0,
            90,
            Some(&mut state),
            1_000,
            1_200,
            24,
        );
        assert_eq!(state.displayed_health, 20);
        assert_eq!(
            flashing.vertices().len() / QUAD_VERTICES,
            25,
            "ten containers + ten previous-health ghosts + five live hearts"
        );
    }

    #[test]
    fn head_hat_gate_and_upside_down_uvs_are_observable() {
        let skin = UiTexture::gui(DEFAULT_SKIN_TEXTURE);
        let mut normal = DrawList::new();
        draw_head_layer(&mut normal, &skin, 0, 0, 8.0, false);
        let mut flipped = DrawList::new();
        draw_head_layer(&mut flipped, &skin, 0, 0, 8.0, true);
        assert_eq!(normal.vertices()[0].uv[1], 8.0 / 64.0);
        assert_eq!(flipped.vertices()[0].uv[1], 16.0 / 64.0);
        assert_eq!(flipped.vertices()[1].uv[1], 8.0 / 64.0);

        let skin_quads = |hat: bool| {
            let mut font = uniform_font();
            let mut list = DrawList::new();
            let mut player = row("Dinnerbone", 0);
            player.hat = hat;
            TabListOverlay::new().draw(
                &mut list,
                &mut font,
                854,
                &[player],
                None,
                None,
                None,
                true,
            );
            list.batches()
                .iter()
                .filter(|batch| {
                    matches!(&batch.texture, UiTexture::Resource(r) if r.path == DEFAULT_SKIN_TEXTURE)
                })
                .map(|batch| batch.range.len() / QUAD_VERTICES)
                .sum::<usize>()
        };
        assert_eq!(skin_quads(false), 1, "the face does not imply a hat layer");
        assert_eq!(skin_quads(true), 2, "the explicit HAT model bit enables it");
    }

    /// java: `GuiPlayerTabOverlay` reserves and draws the head column only when its encrypted
    /// connection flag is true. A row skin must therefore be completely absent when callers pass
    /// `show_heads = false`, even if the row has a hat bit and custom pixels ready.
    #[test]
    fn head_column_and_custom_face_are_gated_together() {
        let mut player = row("alex", 0);
        player.skin = Some(UiTexture::gui("textures/entity/alex.png"));
        player.hat = true;
        let face_quads = |show_heads: bool| {
            let mut font = uniform_font();
            let mut list = DrawList::new();
            TabListOverlay::new().draw(
                &mut list,
                &mut font,
                854,
                &[player.clone()],
                None,
                None,
                None,
                show_heads,
            );
            list.batches()
                .iter()
                .filter(|batch| {
                    matches!(&batch.texture, UiTexture::Resource(r) if r.path == "textures/entity/alex.png")
                })
                .map(|batch| batch.range.len() / QUAD_VERTICES)
                .sum::<usize>()
        };
        assert_eq!(
            face_quads(false),
            0,
            "unencrypted lists do not draw a face or hat"
        );
        assert_eq!(
            face_quads(true),
            2,
            "encrypted lists draw face plus the enabled hat"
        );
    }

    // ---- X3: the `tab_list` mod's style ------------------------------------------------------

    fn styled(style: TabListStyle, rows: &[TabRow], header: Option<&str>) -> DrawList {
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let mut overlay = TabListOverlay::new();
        overlay.style = style;
        overlay.draw(
            &mut d,
            &mut font,
            854,
            rows,
            header,
            Some("footer"),
            None,
            false,
        );
        d
    }

    fn quads(d: &DrawList, pick: impl Fn(&UiTexture) -> bool) -> usize {
        d.batches()
            .iter()
            .filter(|b| pick(&b.texture))
            .map(|b| b.range.len() / QUAD_VERTICES)
            .sum()
    }

    /// The mod-off path: an overlay whose style is `VANILLA` (the default) emits exactly what an
    /// explicitly vanilla style does, and the default really is vanilla.
    #[test]
    fn the_default_style_is_vanilla() {
        assert_eq!(TabListOverlay::new().style, TabListStyle::VANILLA);
        assert_eq!(scaled_alpha(BACKDROP, 1.0), BACKDROP);
        assert_eq!(scaled_alpha(ROW_PLATE, 1.0), ROW_PLATE);
        let rows: Vec<TabRow> = (0..5).map(|i| row(&format!("p{i}"), i * 250 - 1)).collect();
        let vanilla = styled(TabListStyle::VANILLA, &rows, Some("header"));
        let mut font = uniform_font();
        let mut untouched = DrawList::new();
        TabListOverlay::new().draw(
            &mut untouched,
            &mut font,
            854,
            &rows,
            Some("header"),
            Some("footer"),
            None,
            false,
        );
        assert_eq!(vanilla, untouched);
    }

    #[test]
    fn header_and_footer_can_be_hidden() {
        let rows = vec![row("a", 0)];
        let both = styled(TabListStyle::VANILLA, &rows, Some("header"));
        let neither = styled(
            TabListStyle {
                header: false,
                footer: false,
                ..TabListStyle::VANILLA
            },
            &rows,
            Some("header"),
        );
        assert_eq!(quads(&both, |t| matches!(t, UiTexture::Solid)), 4);
        assert_eq!(
            quads(&neither, |t| matches!(t, UiTexture::Solid)),
            2,
            "only the block backdrop and the row plate"
        );
    }

    #[test]
    fn numbers_replace_the_bars_and_hidden_draws_neither() {
        let rows: Vec<TabRow> = [-1, 42, 1_234].iter().map(|&p| row("ab", p)).collect();
        let icons = |t: &UiTexture| matches!(t, UiTexture::Resource(r) if r.path == ICONS_TEXTURE);
        let glyphs = |t: &UiTexture| matches!(t, UiTexture::Font(_));
        let bars = styled(TabListStyle::VANILLA, &rows, None);
        let numbers = styled(
            TabListStyle {
                ping: TabPing::Numbers,
                ..TabListStyle::VANILLA
            },
            &rows,
            None,
        );
        let hidden = styled(
            TabListStyle {
                ping: TabPing::Hidden,
                ..TabListStyle::VANILLA
            },
            &rows,
            None,
        );
        assert_eq!(quads(&bars, icons), 3);
        assert_eq!(quads(&numbers, icons), 0);
        assert_eq!(quads(&hidden, icons), 0);
        // "?", "42ms" and "1234ms" (with shadows) are extra glyphs over the names.
        assert_eq!(
            quads(&numbers, glyphs) - quads(&hidden, glyphs),
            2 * (1 + 4 + 6)
        );
        assert_eq!(quads(&hidden, glyphs), quads(&bars, glyphs));
    }

    /// The column grows by the widest number instead of the icon, and shrinks without one.
    #[test]
    fn the_column_width_follows_the_ping_mode() {
        let rows = vec![row("abc", 1_234)];
        let plate_width = |style: TabListStyle| {
            let d = styled(style, &rows, None);
            // The second Solid quad is the row plate (after the block backdrop); its vertices
            // span the column width.
            let solid: Vec<_> = d
                .batches()
                .iter()
                .filter(|b| matches!(b.texture, UiTexture::Solid))
                .flat_map(|b| d.vertices()[b.range.clone()].iter().copied())
                .collect();
            let plate = &solid[QUAD_VERTICES..2 * QUAD_VERTICES];
            let xs = plate.iter().map(|v| v.pos[0]);
            let (lo, hi) = xs.fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
            (hi - lo) as i32
        };
        let font = uniform_font();
        let name = font.string_width("abc");
        assert_eq!(plate_width(TabListStyle::VANILLA), name + 13);
        assert_eq!(
            plate_width(TabListStyle {
                ping: TabPing::Numbers,
                ..TabListStyle::VANILLA
            }),
            name + 3 + (font.string_width("1234ms") as f32 * PING_TEXT_SCALE).ceil() as i32
        );
        assert_eq!(
            plate_width(TabListStyle {
                ping: TabPing::Hidden,
                ..TabListStyle::VANILLA
            }),
            name + 2
        );
    }

    #[test]
    fn ping_colours_follow_vanillas_bar_thresholds() {
        assert_eq!(ping_number_color(-1), 0xAAAAAA);
        assert_eq!(ping_number_color(0), 0x55FF55);
        assert_eq!(ping_number_color(149), 0x55FF55);
        assert_eq!(ping_number_color(150), 0xFFFF55);
        assert_eq!(ping_number_color(300), 0xFFAA00);
        assert_eq!(ping_number_color(600), 0xFF5555);
        assert_eq!(ping_number_color(1000), 0xAA0000);
        let mut buf = [0u8; 24];
        assert_eq!(ping_number_text(&mut buf, -1), "?");
        assert_eq!(ping_number_text(&mut buf, 57), "57ms");
    }

    #[test]
    fn the_background_opacity_scales_both_plates() {
        let rows = vec![row("a", 0)];
        let faint = styled(
            TabListStyle {
                background: 0.5,
                ..TabListStyle::VANILLA
            },
            &rows,
            None,
        );
        let alphas: Vec<f32> = faint
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Solid))
            .flat_map(|b| faint.vertices()[b.range.clone()].iter().map(|v| v.color[3]))
            .collect();
        let backdrop = (0x80 as f32 * 0.5).round() / 255.0;
        let plate = (0x20 as f32 * 0.5).round() / 255.0;
        assert!(alphas.iter().any(|a| (a - backdrop).abs() < 1e-3));
        assert!(alphas.iter().any(|a| (a - plate).abs() < 1e-3));
        assert_eq!(scaled_alpha(BACKDROP, 0.0) >> 24, 0);
    }
    #[test]
    fn panel_and_row_colours_are_independent_and_ping_glyphs_are_smaller() {
        let rows = [row("Player", 42)];
        let d = styled(
            TabListStyle {
                background_color: 0xC010_2030,
                row_color: 0x7040_5060,
                ping: TabPing::Numbers,
                ..TabListStyle::VANILLA
            },
            &rows,
            None,
        );
        let solids: Vec<_> = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Solid))
            .flat_map(|b| d.vertices()[b.range.clone()].iter())
            .collect();
        assert!(solids
            .iter()
            .any(|v| (v.color[0] - 16.0 / 255.0).abs() < 1e-6
                && (v.color[3] - 192.0 / 255.0).abs() < 1e-6));
        assert!(solids
            .iter()
            .any(|v| (v.color[0] - 64.0 / 255.0).abs() < 1e-6
                && (v.color[3] - 112.0 / 255.0).abs() < 1e-6));
        let glyphs: Vec<_> = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Font(_)))
            .flat_map(|b| d.vertices()[b.range.clone()].chunks_exact(QUAD_VERTICES))
            .collect();
        let height = |quad: &[crate::widget::draw::UiVertex]| {
            let (lo, hi) = quad
                .iter()
                .map(|v| v.pos[1])
                .fold((f32::MAX, f32::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
            hi - lo
        };
        assert!(glyphs
            .iter()
            .any(|q| (height(q) - 7.99 * PING_TEXT_SCALE).abs() < 1e-4));
        assert!(glyphs.iter().any(|q| (height(q) - 7.99).abs() < 1e-4));
    }

    /// A scene that exercises every measured value: spectators (italic names), an
    /// integer objective (score column), latency numbers (per-row widths), a header that wraps and
    /// a footer.
    fn f3_scene() -> (Vec<TabRow>, TabObjective) {
        let mut rows: Vec<TabRow> = (0..23)
            .map(|i| {
                let mut row = row(&format!("Player_{i:02}"), i * 37 - 5);
                row.display = format!("\u{a7}{}Player_{i:02}", (b'a' + (i % 6) as u8) as char);
                row.score = Some(i * 3 - 4);
                row.spectator = i % 5 == 0;
                row
            })
            .collect();
        sort_rows(&mut rows);
        let objective = TabObjective {
            display_name: "Kills".to_owned(),
            render_type: TabRenderType::Integer,
        };
        (rows, objective)
    }

    fn f3_draw(
        overlay: &mut TabListOverlay,
        font: &mut crate::font::FontRenderer,
        rows: &[TabRow],
        objective: &TabObjective,
        header: &str,
    ) -> DrawList {
        let mut d = DrawList::new();
        overlay.draw(
            &mut d,
            font,
            854,
            rows,
            Some(header),
            Some("\u{a7}bfooter line"),
            Some(objective),
            true,
        );
        d
    }

    const F3_HEADER: &str = "a header long enough to wrap onto a second line of the overlay \
                             because it is far wider than the screen allows";

    /// With a measure revision, the second and later frames reuse the
    /// measurements, and every frame is exactly the frame the per-frame path draws.
    #[test]
    fn measured_frames_equal_the_per_frame_measurement() {
        let mut font = uniform_font();
        let (rows, objective) = f3_scene();
        for ping in [TabPing::Numbers, TabPing::Bars] {
            let style = TabListStyle {
                ping,
                ..TabListStyle::VANILLA
            };
            let mut uncached = TabListOverlay::new();
            uncached.style = style;
            let expected = f3_draw(&mut uncached, &mut font, &rows, &objective, F3_HEADER);
            let mut cached = TabListOverlay::new();
            cached.style = style;
            for frame in 0..3 {
                cached.set_measure_revision(Some(7));
                let got = f3_draw(&mut cached, &mut font, &rows, &objective, F3_HEADER);
                assert!(
                    got == expected,
                    "{ping:?} frame {frame} differs from the uncached draw"
                );
            }
            // A changed header is re-measured even under the same revision.
            let expected = f3_draw(&mut uncached, &mut font, &rows, &objective, "short");
            cached.set_measure_revision(Some(7));
            let got = f3_draw(&mut cached, &mut font, &rows, &objective, "short");
            assert!(
                got == expected,
                "{ping:?}: a new header must be re-measured"
            );
        }
    }

    /// Negative control: the reuse is real. Under an unchanged revision, rows the
    /// caller changed anyway are drawn with the old measurements (the caller's contract is to
    /// move the revision whenever its rows change), so the frame differs from a fresh measure;
    /// a new revision measures again.
    #[test]
    fn a_kept_revision_reuses_the_measurements_and_a_new_one_remeasures() {
        let mut font = uniform_font();
        let (rows, objective) = f3_scene();
        let mut wider = rows.clone();
        wider[3].display.push_str(" with a much longer name");
        let mut fresh = TabListOverlay::new();
        let expected = f3_draw(&mut fresh, &mut font, &wider, &objective, F3_HEADER);
        let mut cached = TabListOverlay::new();
        cached.set_measure_revision(Some(1));
        f3_draw(&mut cached, &mut font, &rows, &objective, F3_HEADER);
        cached.set_measure_revision(Some(1));
        let stale = f3_draw(&mut cached, &mut font, &wider, &objective, F3_HEADER);
        assert!(stale != expected, "the measurements were not reused");
        cached.set_measure_revision(Some(2));
        let remeasured = f3_draw(&mut cached, &mut font, &wider, &objective, F3_HEADER);
        assert!(remeasured == expected, "a new revision must measure again");
        // The revision is spent by its draw: a draw without one measures, never reuses.
        cached.set_measure_revision(Some(2));
        f3_draw(&mut cached, &mut font, &rows, &objective, F3_HEADER);
        let fresh_again = f3_draw(&mut cached, &mut font, &wider, &objective, F3_HEADER);
        assert!(
            fresh_again == expected,
            "a draw without a revision must not inherit the previous one"
        );
    }
}
