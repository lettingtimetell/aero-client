//! Minecraft 1.7.10's book screen. A child of `book`, for the screen's private state.
//!
//! - **Page fit**: java17 `GuiScreenBook.pageInsertIntoCurrent`
//!   (`client/gui/GuiScreenBook.java:346-355`) keeps an insertion while the page (with its caret)
//!   wraps to at most 118 pixels of height at the 118-pixel edit width: 13 lines, where 1.8.9
//!   allows 128 (14 lines).
//! - **Plain pages**: a 1.7.10 book's pages are plain strings. A signed book is drawn as its raw
//!   page text (`drawSplitString`, `:424`) and has no click events, so nothing on a page can run
//!   a command or open a link; 1.8.9 reads each page as a JSON component. Nor is there a
//!   validity check: 1.8.9 shows "* Invalid book tag *" for a book its `ItemEditableBook`
//!   rules reject (no title or author, a title over 32 characters), 1.7.10 the page.
//!
//! `MC|BSign` keeps the typed pages on the wire; that conversion is the send site's.

use mc_version::GameVersion;

use super::{BookScreen, PAGE_EDIT_WIDTH};
use crate::font::FontRenderer;

/// The tallest page 1.7.10's editor accepts, caret included, in pixels.
pub const MAX_PAGE_HEIGHT_V5: i32 = 118;

impl BookScreen {
    /// Edit and show the book with `version`'s rules. Screens start with 1.8.9's.
    pub fn set_game_version(&mut self, version: GameVersion) {
        self.version = version;
    }
}

/// java17 `pageInsertIntoCurrent`'s height test for the page text plus its caret.
pub(super) fn page_fits_v5(font: &FontRenderer, page_with_caret: &str) -> bool {
    font.split_string_height(page_with_caret, PAGE_EDIT_WIDTH) <= MAX_PAGE_HEIGHT_V5
}

#[cfg(test)]
#[path = "book_v5_tests.rs"]
mod tests;
