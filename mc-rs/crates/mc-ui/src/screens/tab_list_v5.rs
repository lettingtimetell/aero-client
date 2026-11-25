//! Minecraft 1.7.10's player list. A child of `tab_list`, for its helpers and the overlay's state.
//!
//! java17 `GuiIngame.renderGameOverlay` (`client/gui/GuiIngame.java:388-483`) draws a grid with
//! one cell per slot of the server's maximum player count (`S01PacketJoinGame`): columns are
//! added until no column holds more than 20 rows, each column is `300 / columns` wide (at most
//! 150), and cells fill **row by row** across the columns. The players fill the first cells in
//! the order their `S38PacketPlayerListItem` added them (no sorting), each as its team-formatted
//! name, the list objective's score in yellow when there is room, and the ping bars. No heads,
//! header or footer: 1.8.9's `GuiPlayerTabOverlay` sorts, sizes the grid by the players and
//! draws all three.
//!
//! The `tab_list` mod's style applies as in 1.8.9: the backdrop and cell colours with their
//! alpha factor, and the latency as bars, numbers or nothing.

use super::{
    ping_number_color, ping_number_text, ping_sprite_row, scaled_alpha, TabListOverlay, TabPing,
    PING_TEXT_SCALE,
};
use crate::font::FontRenderer;
use crate::widget::draw::{DrawList, UiTexture};
use crate::widget::Transform;

/// One player's cell: what java17 draws for one `GuiPlayerInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabCellV5 {
    /// `ScorePlayerTeam.formatPlayerName(team, name)`.
    pub name: String,
    /// The list objective's score as drawn (`§e` and the points), when there is a list objective
    /// and room for it ([`TabGridV5::has_score_room`]).
    pub score: Option<String>,
    /// `GuiPlayerInfo.responseTime`, in milliseconds.
    pub ping_ms: i32,
}

/// The grid java17 sizes from the maximum player count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabGridV5 {
    /// Number of cells: the server's maximum player count.
    pub cells: i32,
    /// Columns.
    pub columns: i32,
    /// Rows per column.
    pub rows: i32,
    /// One column's width in pixels.
    pub column_width: i32,
}

impl TabGridV5 {
    /// java17 `GuiIngame.renderGameOverlay:395-405`.
    #[must_use]
    pub fn for_max_players(max_players: u8) -> TabGridV5 {
        let cells = i32::from(max_players);
        let mut rows = cells;
        let mut columns = 1;
        while rows > 20 {
            columns += 1;
            rows = (cells + columns - 1) / columns;
        }
        TabGridV5 {
            cells,
            columns,
            rows,
            column_width: (300 / columns).min(150),
        }
    }

    /// Whether a name `name_width` pixels wide leaves room for the score: java17 draws it only
    /// when `(x + width - 12 - 5) - (x + name_width + 5) > 5`.
    #[must_use]
    pub const fn has_score_room(&self, name_width: i32) -> bool {
        (self.column_width - 12 - 5) - (name_width + 5) > 5
    }
}

/// The 1.7.10 list's cells and the inputs they were built from.
#[derive(Debug, Clone, Default)]
pub(super) struct TabCacheV5 {
    key: Option<(u64, u64, u8)>,
    cells: Vec<TabCellV5>,
}

impl TabListOverlay {
    /// Keep the 1.7.10 cells for the inputs named by `key` (the caller's player-list and
    /// scoreboard revisions, and the maximum player count): `build` runs only when `key` changed
    /// since the last call, so an unchanged list costs no formatting.
    pub fn refresh_v5(&mut self, key: (u64, u64, u8), build: impl FnOnce() -> Vec<TabCellV5>) {
        if self.v5.key != Some(key) {
            self.v5.cells = build();
            self.v5.key = Some(key);
        }
    }

    /// The cells [`Self::refresh_v5`] last built.
    #[must_use]
    pub fn cells_v5(&self) -> &[TabCellV5] {
        &self.v5.cells
    }

    /// Forget the cached cells (a new connection).
    pub fn clear_v5(&mut self) {
        self.v5 = TabCacheV5::default();
    }

    /// Draw 1.7.10's player list for a screen `width` GUI pixels wide: the grid for
    /// `max_players`, the cells [`Self::refresh_v5`] built in its first slots. java17
    /// `GuiIngame.renderGameOverlay:388-483`.
    pub fn draw_v5(&self, d: &mut DrawList, font: &mut FontRenderer, width: i32, max_players: u8) {
        let cells = &self.v5.cells;
        let style = self.style;
        let grid = TabGridV5::for_max_players(max_players);
        let left = (width - grid.columns * grid.column_width) / 2;
        let top = 10;
        d.rect(
            left - 1,
            top - 1,
            left + grid.column_width * grid.columns,
            top + 9 * grid.rows,
            scaled_alpha(style.background_color, style.background),
        );
        let plate = scaled_alpha(style.row_color, style.background);
        let icons = UiTexture::icons();
        for index in 0..grid.cells {
            let x = left + index % grid.columns * grid.column_width;
            let y = top + index / grid.columns * 9;
            d.rect(x, y, x + grid.column_width - 1, y + 8, plate);
            let Some(cell) = cells.get(index as usize) else {
                continue;
            };
            font.draw_string_with_shadow(d, &cell.name, x as f32, y as f32, 0x00FF_FFFF);
            if let Some(score) = &cell.score {
                let right = x + grid.column_width - 12 - 5;
                let score_x = right - font.string_width(score);
                font.draw_string_with_shadow(d, score, score_x as f32, y as f32, 0x00FF_FFFF);
            }
            match style.ping {
                TabPing::Bars => d.blit(
                    icons,
                    x + grid.column_width - 12,
                    y,
                    0,
                    176 + ping_sprite_row(cell.ping_ms) * 8,
                    10,
                    8,
                ),
                TabPing::Numbers => {
                    let mut buf = [0u8; 24];
                    let text = ping_number_text(&mut buf, cell.ping_ms);
                    let text_x = (x + grid.column_width - 1) as f32
                        - font.string_width(text) as f32 * PING_TEXT_SCALE;
                    let text_y = y as f32 + (8.0 - 8.0 * PING_TEXT_SCALE) * 0.5;
                    d.with_transform(
                        Transform::translate(text_x, text_y).then_scale(PING_TEXT_SCALE),
                        |d| {
                            font.draw_string_with_shadow(
                                d,
                                text,
                                0.0,
                                0.0,
                                ping_number_color(cell.ping_ms),
                            );
                        },
                    );
                }
                TabPing::Hidden => {}
            }
        }
    }
}

#[cfg(test)]
#[path = "tab_list_v5_tests.rs"]
mod tests;
