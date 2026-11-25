//! Written/editable book screen.
//!
//! java: `client/gui/GuiScreenBook.java`
//!
//! The application owns networking. This screen mutates the same cloned [`ItemStack`] that it
//! renders and queues [`BookAction`] values at the exact points where vanilla closes the GUI or
//! sends `MC|BEdit` / `MC|BSign`.

use mc_assets::{format_template, Language};
use mc_nbt::{Nbt, NbtCompound, TagType};
use mc_protocol::item::ItemStack;
use mc_version::GameVersion;

use crate::font::component::{ChatComponent, ClickAction, ResolvedStyle, Span};
use crate::font::FontRenderer;
use crate::screen_stack::{Screen, ScreenCtx};
use crate::widget::draw::{DrawList, UiTexture};
use crate::widget::keys::{self, Clipboard, Modifiers};
use crate::widget::{draw_centered_string, draw_string, Button, Rect};

// Minecraft 1.7.10's page fit and plain pages: a child module, for the screen's private state.
#[path = "book_v5.rs"]
mod v5;
pub use v5::MAX_PAGE_HEIGHT_V5;

/// `textures/gui/book.png`, the 256x256 sheet used by the page and arrow sprites.
pub const BOOK_TEXTURE: &str = "textures/gui/book.png";
/// Maximum number of editable pages. java: `addNewPage`.
pub const MAX_PAGES: usize = 50;
/// The strict Java `s1.length() < 256` limit leaves at most 255 UTF-16 code units.
pub const MAX_PAGE_UTF16: usize = 255;
/// Maximum rendered page height, including the blinking cursor.
pub const MAX_PAGE_HEIGHT: i32 = 128;
/// Maximum title length in Java UTF-16 code units.
pub const MAX_TITLE_UTF16: usize = 16;

const BOOK_WIDTH: i32 = 192;
const BOOK_HEIGHT: i32 = 192;
const PAGE_LEFT: i32 = 36;
// Java's GuiScreenBook uses j = 2, then draws page text at j + 16 + 16 = 34.
// Keep the hit-test origin on that same pixel: the texture itself starts at y = 2,
// but the first glyph row starts two pixels below the old 32px approximation.
const PAGE_TOP: i32 = 34;
const PAGE_WIDTH: i32 = 116;
const PAGE_EDIT_WIDTH: i32 = 118;

mod ids {
    pub const DONE: i32 = 0;
    pub const NEXT: i32 = 1;
    pub const PREVIOUS: i32 = 2;
    pub const SIGN: i32 = 3;
    pub const CANCEL: i32 = 4;
    pub const FINALIZE: i32 = 5;
}

/// An intent emitted by [`BookScreen`].
#[derive(Debug, Clone, PartialEq)]
pub enum BookAction {
    /// Close the screen without saving. Escape and `run_command` use this path.
    Close,
    /// Send a fully-mutated stack as a `C17PacketCustomPayload` payload.
    Send {
        /// Either `MC|BEdit` or `MC|BSign`.
        channel: &'static str,
        /// The stack serialized directly into the plugin-message payload.
        stack: ItemStack,
    },
    /// Run a command from a written-book component.
    RunCommand(String),
    /// Seed the chat input from a written-book component.
    SuggestCommand(String),
    /// Ask the host to run its normal URL confirmation flow.
    OpenUrl(String),
}

#[derive(Debug, Clone)]
struct WrappedSpan {
    text: String,
    style: ResolvedStyle,
}

/// A Java `String`'s UTF-16 code units plus the Rust-facing display view.
///
/// `GuiScreenBook` edits page/title strings with `String.substring(0, length - 1)`, so a
/// backspace immediately after a supplementary character deliberately leaves one unpaired
/// surrogate in the Java value. Rust `String` cannot hold that intermediate value; retain the
/// units here and expose `from_utf16_lossy` only where a Rust API or renderer requires `&str`.
#[derive(Debug, Clone, Default)]
struct RawUtf16 {
    units: Vec<u16>,
    display: String,
}

impl RawUtf16 {
    fn from_str(value: &str) -> Self {
        Self::from_units(value.encode_utf16().collect())
    }

    fn from_units(units: Vec<u16>) -> Self {
        let display = String::from_utf16_lossy(&units);
        Self { units, display }
    }

    fn as_str(&self) -> &str {
        &self.display
    }

    fn len(&self) -> usize {
        self.units.len()
    }

    fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    fn push_str(&mut self, value: &str) {
        self.units.extend(value.encode_utf16());
        self.refresh_display();
    }

    /// Append at most `max_units` UTF-16 units, matching Java's per-`char` title gate.
    ///
    /// Java receives supplementary input as two `char` events. If only one unit remains,
    /// the high surrogate is accepted and the low surrogate is rejected by the next event;
    /// retaining that transient unit is important for the exact 16-unit title boundary.
    fn push_char_up_to(&mut self, value: char, max_units: usize) -> bool {
        let mut units = [0; 2];
        let encoded = value.encode_utf16(&mut units);
        let room = max_units.saturating_sub(self.len());
        if room == 0 {
            return false;
        }
        self.units.extend(encoded.iter().copied().take(room));
        self.refresh_display();
        true
    }

    fn pop(&mut self) -> Option<u16> {
        let unit = self.units.pop();
        if unit.is_some() {
            self.refresh_display();
        }
        unit
    }

    fn to_string_lossy(&self) -> String {
        self.display.clone()
    }

    /// Match Java 8 `String.trim()`, which removes only UTF-16 units at or below U+0020.
    fn trim_java(&self) -> String {
        let Some(start) = self.units.iter().position(|unit| *unit > 0x20) else {
            return String::new();
        };
        let end = self
            .units
            .iter()
            .rposition(|unit| *unit > 0x20)
            .expect("trim start guarantees a non-control unit")
            + 1;
        String::from_utf16_lossy(&self.units[start..end])
    }

    fn refresh_display(&mut self) {
        self.display = String::from_utf16_lossy(&self.units);
    }
}

/// Pure state and rendering model for `GuiScreenBook`.
#[derive(Debug, Clone)]
pub struct BookScreen {
    book: ItemStack,
    unsigned: bool,
    modified: bool,
    signing: bool,
    update_count: i32,
    total_pages: usize,
    current_page: usize,
    pages: Option<Vec<RawUtf16>>,
    title: RawUtf16,
    author: String,
    buttons: Vec<Button>,
    next_rect: Rect,
    previous_rect: Rect,
    width: i32,
    actions: Vec<BookAction>,
    /// Whose page rules the screen follows (`book_v5.rs`); 1.8.9 unless set.
    version: GameVersion,
}

impl BookScreen {
    /// Construct the screen from the live held stack and player name.
    ///
    /// `unsigned` is true only for item 386 opened locally. Item 387 is opened read-only by
    /// inbound `MC|BOpen`.
    #[must_use]
    pub fn new(book: ItemStack, author: impl Into<String>, unsigned: bool) -> Self {
        // `getTagList("pages", 8)` returns an empty list when the key/type is wrong, but only
        // when a compound exists. A missing compound leaves Java's `bookPages` null.
        let mut pages = book.nbt.as_ref().map(|tag| {
            let Some(Nbt::List(values)) = tag.get("pages") else {
                return Vec::new();
            };
            if values
                .first()
                .is_some_and(|value| value.tag_type() != TagType::String)
            {
                return Vec::new();
            }
            values
                .iter()
                .filter_map(|value| value.as_str().map(RawUtf16::from_str))
                .collect::<Vec<_>>()
        });
        if pages.is_none() && unsigned {
            pages = Some(vec![RawUtf16::default()]);
        }
        let total_pages = pages.as_ref().map_or(1, Vec::len).max(1);
        Self {
            book,
            unsigned,
            modified: false,
            signing: false,
            update_count: 0,
            total_pages,
            current_page: 0,
            pages,
            title: RawUtf16::default(),
            author: author.into(),
            buttons: Vec::new(),
            next_rect: Rect::new(0, 0, 23, 13),
            previous_rect: Rect::new(0, 0, 23, 13),
            width: 0,
            actions: Vec::new(),
            version: GameVersion::V1_8_9,
        }
    }

    /// Whether this is an editable book-and-quill screen.
    #[must_use]
    pub const fn is_unsigned(&self) -> bool {
        self.unsigned
    }

    /// Whether page/title state has changed since the screen opened.
    #[must_use]
    pub const fn is_modified(&self) -> bool {
        self.modified
    }

    /// Whether the title/finalization view is open.
    #[must_use]
    pub const fn is_signing(&self) -> bool {
        self.signing
    }

    /// Zero-based current page.
    #[must_use]
    pub const fn current_page_index(&self) -> usize {
        self.current_page
    }

    /// Number shown in the page indicator, clamped to at least one.
    #[must_use]
    pub const fn total_pages(&self) -> usize {
        self.total_pages
    }

    /// Current page text, or the empty string for Java's null/out-of-range path.
    #[must_use]
    pub fn current_page(&self) -> &str {
        self.pages
            .as_ref()
            .and_then(|pages| pages.get(self.current_page))
            .map_or("", RawUtf16::as_str)
    }

    /// Current untrimmed title field.
    #[must_use]
    pub fn title(&self) -> &str {
        self.title.as_str()
    }

    /// The screen's locally-mutated held stack.
    #[must_use]
    pub const fn book(&self) -> &ItemStack {
        &self.book
    }

    /// Drain queued close/network/component actions in occurrence order.
    pub fn take_actions(&mut self) -> Vec<BookAction> {
        std::mem::take(&mut self.actions)
    }

    /// Recreate vanilla's buttons and page arrows for the current resolution/language.
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        self.width = ctx.width;
        let y = 4 + BOOK_HEIGHT;
        self.buttons.clear();
        if self.unsigned {
            self.buttons.push(Button::sized(
                ids::SIGN,
                ctx.width / 2 - 100,
                y,
                98,
                20,
                lang.tr("book.signButton"),
            ));
            self.buttons.push(Button::sized(
                ids::DONE,
                ctx.width / 2 + 2,
                y,
                98,
                20,
                lang.tr("gui.done"),
            ));
            self.buttons.push(Button::sized(
                ids::FINALIZE,
                ctx.width / 2 - 100,
                y,
                98,
                20,
                lang.tr("book.finalizeButton"),
            ));
            self.buttons.push(Button::sized(
                ids::CANCEL,
                ctx.width / 2 + 2,
                y,
                98,
                20,
                lang.tr("gui.cancel"),
            ));
        } else {
            self.buttons.push(Button::new(
                ids::DONE,
                ctx.width / 2 - 100,
                y,
                lang.tr("gui.done"),
            ));
        }
        let left = (ctx.width - BOOK_WIDTH) / 2;
        self.next_rect = Rect::new(left + 120, 156, 23, 13);
        self.previous_rect = Rect::new(left + 38, 156, 23, 13);
        self.update_buttons();
    }

    fn update_buttons(&mut self) {
        for button in &mut self.buttons {
            match button.id {
                ids::DONE => button.visible = !self.unsigned || !self.signing,
                ids::SIGN => button.visible = !self.signing,
                ids::CANCEL => button.visible = self.signing,
                ids::FINALIZE => {
                    button.visible = self.signing;
                    button.enabled = !self.title.trim_java().is_empty();
                }
                _ => {}
            }
        }
    }

    /// Whether the next-page arrow is currently visible.
    #[must_use]
    pub fn next_visible(&self) -> bool {
        !self.signing && (self.current_page < self.total_pages - 1 || self.unsigned)
    }

    /// Whether the previous-page arrow is currently visible.
    #[must_use]
    pub fn previous_visible(&self) -> bool {
        !self.signing && self.current_page > 0
    }

    fn add_new_page(&mut self) {
        if let Some(pages) = self.pages.as_mut().filter(|pages| pages.len() < MAX_PAGES) {
            pages.push(RawUtf16::default());
            self.total_pages += 1;
            self.modified = true;
        }
    }

    fn set_current_page(&mut self, page: RawUtf16) {
        if let Some(slot) = self
            .pages
            .as_mut()
            .and_then(|pages| pages.get_mut(self.current_page))
        {
            *slot = page;
            self.modified = true;
        }
    }

    fn insert_into_page(&mut self, inserted: &str, font: &FontRenderer) {
        let Some(mut next) = self
            .pages
            .as_ref()
            .and_then(|pages| pages.get(self.current_page))
            .cloned()
        else {
            return;
        };
        next.push_str(inserted);
        let mut measured = next.as_str().to_owned();
        measured.push_str("\u{a7}0_");
        // 1.7.10's pages hold 13 lines, not 14 (`book_v5.rs`).
        if self.version.is_1_7_10() && !v5::page_fits_v5(font, &measured) {
            return;
        }
        if next.len() <= MAX_PAGE_UTF16
            && font.split_string_height(&measured, PAGE_EDIT_WIDTH) <= MAX_PAGE_HEIGHT
        {
            self.set_current_page(next);
        }
    }

    fn backspace_page(&mut self) {
        let Some(mut page) = self
            .pages
            .as_ref()
            .and_then(|pages| pages.get(self.current_page))
            .cloned()
        else {
            return;
        };
        if page.pop().is_some() {
            self.set_current_page(page);
        }
    }

    fn key_in_page(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        clipboard: &dyn Clipboard,
        font: &FontRenderer,
    ) {
        if mods.is_ctrl_combo(key, keys::V) {
            // Unlike GuiTextField, GuiScreenBook sends the clipboard directly to
            // pageInsertIntoCurrent.  Its length and rendered-height checks are the only paste
            // gate; `ChatAllowedCharacters` applies exclusively to one typed `char` at a time.
            // In particular, a pasted section sign remains part of the page and is serialized
            // in the following MC|BEdit/MC|BSign payload.
            self.insert_into_page(&clipboard.get(), font);
        } else if key == keys::BACKSPACE {
            self.backspace_page();
        } else if key == keys::ENTER || key == keys::NUMPAD_ENTER {
            self.insert_into_page("\n", font);
        } else if let Some(c) = typed.filter(|c| keys::is_allowed_character(*c)) {
            self.insert_into_page(&c.to_string(), font);
        }
    }

    fn key_in_title(&mut self, typed: Option<char>, key: i32) {
        if key == keys::BACKSPACE {
            self.title.pop();
            self.update_buttons();
        } else if key == keys::ENTER || key == keys::NUMPAD_ENTER {
            if !self.title.is_empty() {
                self.queue_send(true);
                self.actions.push(BookAction::Close);
            }
        } else if let Some(c) = typed.filter(|c| keys::is_allowed_character(*c)) {
            if self.title.push_char_up_to(c, MAX_TITLE_UTF16) {
                self.modified = true;
                self.update_buttons();
            }
        }
    }

    /// Handle one keyboard event, including clipboard paste and the no-save Escape path.
    pub fn key_typed(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        clipboard: &dyn Clipboard,
        font: &FontRenderer,
    ) {
        if key == keys::ESCAPE {
            self.actions.push(BookAction::Close);
        }
        if self.unsigned {
            if self.signing {
                self.key_in_title(typed, key);
            } else {
                self.key_in_page(typed, key, mods, clipboard, font);
            }
        }
    }

    fn activate(&mut self, id: i32) {
        match id {
            ids::DONE => {
                self.actions.push(BookAction::Close);
                self.queue_send(false);
            }
            ids::SIGN if self.unsigned => self.signing = true,
            ids::NEXT => {
                if self.current_page < self.total_pages - 1 {
                    self.current_page += 1;
                } else if self.unsigned {
                    self.add_new_page();
                    if self.current_page < self.total_pages - 1 {
                        self.current_page += 1;
                    }
                }
            }
            ids::PREVIOUS if self.current_page > 0 => self.current_page -= 1,
            ids::FINALIZE if self.signing && !self.title.trim_java().is_empty() => {
                self.queue_send(true);
                self.actions.push(BookAction::Close);
            }
            ids::CANCEL if self.signing => self.signing = false,
            _ => {}
        }
        self.update_buttons();
    }

    fn queue_send(&mut self, publish: bool) {
        if !self.unsigned || !self.modified {
            return;
        }
        let Some(mut pages) = self.pages.clone() else {
            return;
        };
        while pages.len() > 1 && pages.last().is_some_and(RawUtf16::is_empty) {
            pages.pop();
        }
        let tag = self.book.nbt.get_or_insert_with(NbtCompound::new);
        tag.insert(
            "pages",
            Nbt::list(
                pages
                    .iter()
                    // `Nbt::String` is a Rust `String`; a lone Java surrogate therefore becomes
                    // U+FFFD at this unavoidable serialization boundary.
                    .map(RawUtf16::to_string_lossy)
                    .map(Nbt::String)
                    .collect(),
            ),
        );
        let channel = if publish {
            tag.insert("author", Nbt::String(self.author.clone()));
            tag.insert("title", Nbt::String(self.title.trim_java()));
            tag.insert(
                "pages",
                Nbt::list(
                    pages
                        .iter()
                        .map(|page| Nbt::String(gson_string_literal(page.as_str())))
                        .collect(),
                ),
            );
            self.book.item_id = 387;
            "MC|BSign"
        } else {
            "MC|BEdit"
        };
        self.pages = Some(pages);
        self.total_pages = self.pages.as_ref().map_or(1, Vec::len).max(1);
        self.current_page = self.current_page.min(self.total_pages - 1);
        self.actions.push(BookAction::Send {
            channel,
            stack: self.book.clone(),
        });
    }

    /// Handle a mouse press. Component clicks take precedence over buttons, like vanilla.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        font: &FontRenderer,
        lang: &Language,
    ) {
        if button != 0 {
            return;
        }
        if let Some(style) = self.style_at(x, y, font, lang) {
            if self.handle_component_click(&style) {
                return;
            }
        }
        if self.next_visible() && self.next_rect.contains(x, y) {
            self.activate(ids::NEXT);
            return;
        }
        if self.previous_visible() && self.previous_rect.contains(x, y) {
            self.activate(ids::PREVIOUS);
            return;
        }
        if let Some(id) = self
            .buttons
            .iter()
            .find(|candidate| candidate.mouse_pressed(x, y))
            .map(|button| button.id)
        {
            self.activate(id);
        }
    }

    fn handle_component_click(&mut self, style: &ResolvedStyle) -> bool {
        let Some(event) = style.click_event.as_ref() else {
            return false;
        };
        match event.action {
            ClickAction::ChangePage => {
                let Ok(page) = event.value.parse::<usize>() else {
                    return false;
                };
                let Some(index) = page.checked_sub(1) else {
                    return false;
                };
                if index < self.total_pages && index != self.current_page {
                    self.current_page = index;
                    true
                } else {
                    false
                }
            }
            ClickAction::RunCommand => {
                self.actions
                    .push(BookAction::RunCommand(event.value.clone()));
                self.actions.push(BookAction::Close);
                true
            }
            ClickAction::SuggestCommand => {
                self.actions
                    .push(BookAction::SuggestCommand(event.value.clone()));
                true
            }
            ClickAction::OpenUrl => {
                self.actions.push(BookAction::OpenUrl(event.value.clone()));
                true
            }
            // `CopyImage`is client-only and never parsed from a
            // book's JSON, like `OpenFile`.
            ClickAction::OpenFile | ClickAction::CopyImage | ClickAction::TwitchUserInfo => false,
        }
    }

    fn parsed_lines(&self, font: &FontRenderer, lang: &Language) -> Option<Vec<Vec<WrappedSpan>>> {
        // A 1.7.10 page is plain text, drawn raw and never clickable (`book_v5.rs`).
        if self.version.is_1_7_10() {
            return None;
        }
        if self.unsigned || !valid_written_book(&self.book) {
            return None;
        }
        let component = ChatComponent::from_json(self.current_page()).ok()?;
        Some(split_component(&component, PAGE_WIDTH, font, lang))
    }

    fn style_at(
        &self,
        x: i32,
        y: i32,
        font: &FontRenderer,
        lang: &Language,
    ) -> Option<ResolvedStyle> {
        let lines = self.parsed_lines(font, lang)?;
        let local_x = x - (self.width - BOOK_WIDTH) / 2 - PAGE_LEFT;
        let local_y = y - PAGE_TOP;
        if !(0..=PAGE_WIDTH).contains(&local_x) || local_y < 0 {
            return None;
        }
        let visible = (MAX_PAGE_HEIGHT / font.font_height()) as usize;
        let row = (local_y / font.font_height()) as usize;
        if row >= lines.len().min(visible)
            || local_y >= font.font_height() * visible as i32 + visible as i32
        {
            return None;
        }
        let mut width = 0;
        for span in &lines[row] {
            width += font.string_width(&span.text);
            if width > local_x {
                return Some(span.style.clone());
            }
        }
        None
    }

    /// Draw the book page, text, navigation arrows, and current button set.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        let texture = UiTexture::gui(BOOK_TEXTURE);
        let left = (ctx.width - BOOK_WIDTH) / 2;
        d.blit(&texture, left, 2, 0, 0, BOOK_WIDTH, BOOK_HEIGHT);
        if self.signing {
            let cursor = if self.update_count / 6 % 2 == 0 {
                "\u{a7}0_"
            } else {
                "\u{a7}7_"
            };
            let shown = format!("{}{cursor}", self.title.as_str());
            draw_centered_string(
                d,
                font,
                lang.tr("book.editTitle"),
                left + PAGE_LEFT + PAGE_WIDTH / 2,
                34,
                0,
            );
            draw_centered_string(d, font, &shown, left + 94, 50, 0);
            let by = format_template(lang.tr("book.byAuthor"), &[&self.author]);
            draw_centered_string(d, font, &format!("\u{a7}8{by}"), left + 94, 60, 0);
            draw_wrapped(
                d,
                font,
                lang.tr("book.finalizeWarning"),
                left + PAGE_LEFT,
                82,
                PAGE_WIDTH,
                0,
            );
        } else {
            let one = (self.current_page + 1).to_string();
            let total = self.total_pages.to_string();
            let indicator = format_template(lang.tr("book.pageIndicator"), &[&one, &total]);
            draw_string(
                d,
                font,
                &indicator,
                left + BOOK_WIDTH - 44 - font.string_width(&indicator),
                18,
                0,
            );
            if self.unsigned {
                let cursor = if font.bidi_flag() {
                    "_"
                } else if self.update_count / 6 % 2 == 0 {
                    "\u{a7}0_"
                } else {
                    "\u{a7}7_"
                };
                draw_wrapped(
                    d,
                    font,
                    &format!("{}{cursor}", self.current_page()),
                    left + PAGE_LEFT,
                    PAGE_TOP,
                    PAGE_WIDTH,
                    0,
                );
            } else if !self.version.is_1_7_10() && !valid_written_book(&self.book) {
                // 1.7.10 has no validity check: its raw page is drawn below (`book_v5.rs`).
                draw_string(
                    d,
                    font,
                    "\u{a7}4* Invalid book tag *",
                    left + PAGE_LEFT,
                    PAGE_TOP,
                    0,
                );
            } else if let Some(lines) = self.parsed_lines(font, lang) {
                for (row, line) in lines
                    .iter()
                    .take((MAX_PAGE_HEIGHT / font.font_height()) as usize)
                    .enumerate()
                {
                    let mut text = String::new();
                    for span in line {
                        text.push_str(&span.text);
                    }
                    draw_string(
                        d,
                        font,
                        &text,
                        left + PAGE_LEFT,
                        PAGE_TOP + row as i32 * font.font_height(),
                        0,
                    );
                }
            } else {
                draw_wrapped(
                    d,
                    font,
                    self.current_page(),
                    left + PAGE_LEFT,
                    PAGE_TOP,
                    PAGE_WIDTH,
                    0,
                );
            }
        }
        if self.next_visible() {
            draw_arrow(
                d,
                &texture,
                self.next_rect,
                true,
                self.next_rect.contains(mouse.0, mouse.1),
            );
        }
        if self.previous_visible() {
            draw_arrow(
                d,
                &texture,
                self.previous_rect,
                false,
                self.previous_rect.contains(mouse.0, mouse.1),
            );
        }
        for button in &mut self.buttons {
            button.draw(d, font, mouse.0, mouse.1);
        }
    }
}

impl Screen for BookScreen {
    fn name(&self) -> &'static str {
        "book"
    }

    fn tick(&mut self) {
        self.update_count = self.update_count.wrapping_add(1);
    }
}

fn draw_arrow(d: &mut DrawList, texture: &UiTexture, rect: Rect, next: bool, hovered: bool) {
    let u = if hovered { 23 } else { 0 };
    let v = if next { 192 } else { 205 };
    d.blit(texture, rect.x, rect.y, u, v, 23, 13);
}

fn draw_wrapped(
    d: &mut DrawList,
    font: &mut FontRenderer,
    text: &str,
    x: i32,
    mut y: i32,
    width: i32,
    color: i32,
) {
    for line in font.wrap_to_lines(text, width) {
        draw_string(d, font, &line, x, y, color);
        y += font.font_height();
    }
}

fn valid_written_book(stack: &ItemStack) -> bool {
    let Some(tag) = stack.nbt.as_ref() else {
        return false;
    };
    let Some(Nbt::List(pages)) = tag.get("pages") else {
        return false;
    };
    // `getTagList("pages", 8)` substitutes an empty list when the actual list's element type
    // is not TAG_String. The validation loop therefore accepts that malformed-but-empty view.
    if pages
        .first()
        .is_none_or(|page| page.tag_type() == TagType::String)
        && pages.iter().any(|page| {
            page.as_str()
                .is_some_and(|text| text.encode_utf16().count() > 32_767)
        })
    {
        return false;
    }
    tag.get_str("title")
        .is_some_and(|title| title.encode_utf16().count() <= 32)
        && tag.contains_type("author", TagType::String)
}

/// Gson 2.2.4's JSON string literal, including its default HTML-safe escapes.
fn gson_string_literal(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '=' => out.push_str("\\u003d"),
            '\'' => out.push_str("\\u0027"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if c <= '\u{001f}' => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn split_component(
    component: &ChatComponent,
    wrap_width: i32,
    font: &FontRenderer,
    lang: &Language,
) -> Vec<Vec<WrappedSpan>> {
    let mut pending = component.spans(lang);
    let mut out = Vec::new();
    let mut line = Vec::new();
    let mut line_width = 0;
    let mut index = 0;
    while index < pending.len() {
        let source = pending[index].clone();
        let mut raw = source.text;
        let mut line_break = false;
        if let Some(newline) = raw.find('\n') {
            let remainder = raw[newline + 1..].to_owned();
            raw.truncate(newline + 1);
            pending.insert(
                index + 1,
                Span {
                    text: remainder,
                    style: source.style.clone(),
                },
            );
            line_break = true;
        }
        let rendered = format!("{}{}", source.style.formatting_code(), raw);
        let without_newline = rendered.strip_suffix('\n').unwrap_or(&rendered);
        let mut piece = without_newline.to_owned();
        let mut piece_width = font.string_width(&piece);
        if line_width + piece_width > wrap_width {
            piece = font.trim_to_width(&rendered, wrap_width - line_width);
            let cut = piece.len();
            let mut remainder = (cut < rendered.len()).then(|| rendered[cut..].to_owned());
            if let Some(rest) = remainder.as_mut().filter(|rest| !rest.is_empty()) {
                if let Some(space) = piece.rfind(' ') {
                    if font.string_width(&rendered[..space]) > 0 {
                        piece = rendered[..space].to_owned();
                        // The book passes `p_178908_3_ = true`, so the separator is discarded
                        // from the continuation. Chat passes false and deliberately retains it.
                        *rest = rendered[space + 1..].to_owned();
                    }
                } else if line_width > 0 && !rendered.contains(' ') {
                    piece.clear();
                    *rest = rendered.clone();
                }
                pending.insert(
                    index + 1,
                    Span {
                        text: rest.clone(),
                        style: source.style.clone(),
                    },
                );
            }
            piece_width = font.string_width(&piece);
            line_break = true;
        }
        if line_width + piece_width <= wrap_width {
            line_width += piece_width;
            line.push(WrappedSpan {
                text: piece,
                style: source.style,
            });
        } else {
            line_break = true;
        }
        if line_break {
            out.push(std::mem::take(&mut line));
            line_width = 0;
        }
        index += 1;
    }
    out.push(line);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::{MemoryClipboard, ScaledResolution};

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn writable() -> BookScreen {
        BookScreen::new(
            ItemStack {
                item_id: 386,
                count: 1,
                damage: 0,
                nbt: None,
            },
            "Player",
            true,
        )
    }

    fn key(screen: &mut BookScreen, c: char, font: &FontRenderer) {
        screen.key_typed(
            Some(c),
            0,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            font,
        );
    }

    #[test]
    fn constructor_reproduces_the_null_vs_empty_pages_distinction() {
        let empty = writable();
        assert_eq!(empty.current_page(), "");
        assert_eq!(empty.total_pages(), 1);

        let mut stack = empty.book.clone();
        stack.nbt = Some(NbtCompound::new());
        let mut broken = BookScreen::new(stack, "Player", true);
        key(&mut broken, 'x', &uniform_font());
        assert_eq!(
            broken.current_page(),
            "",
            "a compound with no string-list cannot be edited"
        );
        assert!(!broken.is_modified());
    }

    #[test]
    fn next_adds_pages_only_to_the_fifty_page_cap() {
        let mut screen = writable();
        for _ in 0..80 {
            screen.activate(ids::NEXT);
        }
        assert_eq!(screen.total_pages(), MAX_PAGES);
        assert_eq!(screen.current_page_index(), MAX_PAGES - 1);
        assert!(screen.is_modified());
    }

    #[test]
    fn insertion_obeys_both_the_utf16_and_render_height_gates() {
        let font = uniform_font();
        let mut screen = writable();
        for _ in 0..255 {
            key(&mut screen, 'a', &font);
        }
        assert_eq!(screen.current_page().len(), 255);
        let before = screen.current_page().to_owned();
        screen.insert_into_page("z", &font);
        assert_eq!(screen.current_page(), before);

        let mut tall = writable();
        tall.insert_into_page(&"\n".repeat(20), &font);
        assert_eq!(
            tall.current_page(),
            "",
            "the 128px height gate rejects an otherwise-short insertion"
        );

        // Bold advances are six pixels in this fixture, spaces four, and the reset cursor five:
        // 18*6 + 4 + 5 = 117. It fits Java's 118px edit gate but not the 116px draw width.
        let boundary = format!("{}\u{a7}l{} ", "\n".repeat(13), "a".repeat(18));
        let measured = format!("{boundary}\u{a7}0_");
        assert!(
            font.split_string_height(&measured, PAGE_EDIT_WIDTH) <= MAX_PAGE_HEIGHT,
            "the Java edit width admits the boundary fixture"
        );
        assert!(
            font.split_string_height(&measured, PAGE_WIDTH) > MAX_PAGE_HEIGHT,
            "negative control: the 116px draw width rejects the same fixture"
        );
        let mut exact_width = writable();
        exact_width.insert_into_page(&boundary, &font);
        assert_eq!(exact_width.current_page(), boundary);
    }

    #[test]
    fn page_and_title_caps_count_utf16_units_for_non_bmp_input() {
        let font = uniform_font();
        let mut page = writable();
        let boundary = format!("{}😀", "a".repeat(253));
        page.insert_into_page(&boundary, &font);
        assert_eq!(page.current_page().encode_utf16().count(), MAX_PAGE_UTF16);
        let before = page.current_page().to_owned();
        page.insert_into_page("x", &font);
        assert_eq!(page.current_page(), before);

        let mut title = writable();
        title.activate(ids::SIGN);
        for _ in 0..8 {
            key(&mut title, '😀', &font);
        }
        assert_eq!(title.title().encode_utf16().count(), MAX_TITLE_UTF16);
        key(&mut title, '😀', &font);
        assert_eq!(title.title().encode_utf16().count(), MAX_TITLE_UTF16);
    }

    #[test]
    fn title_supplementary_input_stops_at_the_last_java_utf16_unit() {
        let font = uniform_font();
        let mut title = writable();
        title.activate(ids::SIGN);
        for _ in 0..7 {
            key(&mut title, '😀', &font);
        }
        key(&mut title, 'a', &font);
        assert_eq!(title.title().encode_utf16().count(), MAX_TITLE_UTF16 - 1);

        // A Java keyTyped call receives one surrogate char at a time. The first half fits;
        // its low-surrogate partner is rejected by the now-full 16-unit title gate.
        key(&mut title, '😀', &font);
        assert_eq!(title.title().encode_utf16().count(), MAX_TITLE_UTF16);
        assert!(title.title().ends_with('�'));
        key(&mut title, 'x', &font);
        assert_eq!(title.title().encode_utf16().count(), MAX_TITLE_UTF16);
    }

    #[test]
    fn page_backspace_removes_one_utf16_unit_and_renders_the_transient_half() {
        let font = uniform_font();
        let mut screen = writable();
        key(&mut screen, 'A', &font);
        key(&mut screen, '😀', &font);
        assert_eq!(screen.current_page(), "A😀");

        screen.key_typed(
            None,
            keys::BACKSPACE,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            &font,
        );
        assert_eq!(screen.current_page(), "A�");

        // The Java value contains one high surrogate here. The renderer receives the only safe
        // Rust representation, U+FFFD, and must still be able to draw the page.
        screen.init(&ctx(), &Language::new("en_US"));
        let mut draw_font = uniform_font();
        screen.draw(
            &mut DrawList::new(),
            &mut draw_font,
            &ctx(),
            (0, 0),
            &Language::new("en_US"),
        );

        screen.key_typed(
            None,
            keys::BACKSPACE,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            &font,
        );
        assert_eq!(screen.current_page(), "A");
    }

    #[test]
    fn page_backspace_sends_the_lossy_half_surrogate_at_the_nbt_boundary() {
        let font = uniform_font();
        let mut screen = writable();
        key(&mut screen, 'A', &font);
        key(&mut screen, '😀', &font);
        screen.key_typed(
            None,
            keys::BACKSPACE,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            &font,
        );
        screen.activate(ids::DONE);

        let actions = screen.take_actions();
        let BookAction::Send { channel, stack } = &actions[1] else {
            panic!("done sends the edited stack")
        };
        assert_eq!(*channel, "MC|BEdit");
        let pages = stack.nbt.as_ref().unwrap().get_list("pages").unwrap();
        assert_eq!(pages, [Nbt::String("A�".into())]);
    }

    #[test]
    fn title_backspace_removes_one_utf16_unit_and_signing_serializes_the_half() {
        let font = uniform_font();
        let mut screen = writable();
        screen.activate(ids::SIGN);
        key(&mut screen, 'A', &font);
        key(&mut screen, '😀', &font);
        assert_eq!(screen.title(), "A😀");

        screen.key_typed(
            None,
            keys::BACKSPACE,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            &font,
        );
        assert_eq!(screen.title(), "A�");

        screen.init(&ctx(), &Language::new("en_US"));
        let mut draw_font = uniform_font();
        screen.draw(
            &mut DrawList::new(),
            &mut draw_font,
            &ctx(),
            (0, 0),
            &Language::new("en_US"),
        );

        screen.key_typed(
            None,
            keys::BACKSPACE,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            &font,
        );
        assert_eq!(screen.title(), "A");

        // Rebuild the Java sequence so the outgoing title contains the transient lone unit.
        let mut outgoing = writable();
        key(&mut outgoing, 'A', &font);
        key(&mut outgoing, '😀', &font);
        outgoing.key_typed(
            None,
            keys::BACKSPACE,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            &font,
        );
        outgoing.activate(ids::SIGN);
        key(&mut outgoing, 'A', &font);
        key(&mut outgoing, '😀', &font);
        outgoing.key_typed(
            None,
            keys::BACKSPACE,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            &font,
        );
        outgoing.activate(ids::FINALIZE);
        let actions = outgoing.take_actions();
        let BookAction::Send { channel, stack } = &actions[0] else {
            panic!("finalize sends the signed stack")
        };
        assert_eq!(*channel, "MC|BSign");
        assert_eq!(stack.nbt.as_ref().unwrap().get_str("title"), Some("A�"));
        assert_eq!(
            stack.nbt.as_ref().unwrap().get_list("pages").unwrap(),
            [Nbt::String("\"A�\"".into())]
        );
    }

    #[test]
    fn paste_bypasses_chat_allowed_characters_but_typing_does_not() {
        let font = uniform_font();
        let mut screen = writable();
        let mut clipboard = MemoryClipboard::default();
        clipboard.set("a\u{a7}b\t");

        screen.key_typed(None, keys::V, Modifiers::CTRL, &clipboard, &font);
        assert_eq!(
            screen.current_page(),
            "a\u{a7}b\t",
            "GuiScreenBook forwards GuiScreen.getClipboardString directly to pageInsertIntoCurrent"
        );

        key(&mut screen, '\u{a7}', &font);
        assert_eq!(
            screen.current_page(),
            "a\u{a7}b\t",
            "negative control: ChatAllowedCharacters still rejects a physically typed section sign"
        );
    }

    #[test]
    fn done_sends_only_after_a_mutation_and_trims_trailing_blank_pages() {
        let font = uniform_font();
        let mut untouched = writable();
        untouched.activate(ids::DONE);
        assert_eq!(untouched.take_actions(), vec![BookAction::Close]);

        let mut edited = writable();
        key(&mut edited, 'x', &font);
        edited.activate(ids::NEXT);
        edited.activate(ids::DONE);
        let actions = edited.take_actions();
        assert!(matches!(actions[0], BookAction::Close));
        let BookAction::Send { channel, stack } = &actions[1] else {
            panic!("send")
        };
        assert_eq!(*channel, "MC|BEdit");
        assert_eq!(stack.item_id, 386);
        let pages = stack.nbt.as_ref().unwrap().get_list("pages").unwrap();
        assert_eq!(pages, [Nbt::String("x".into())]);
    }

    #[test]
    fn signing_mutates_id_author_title_and_uses_gsons_html_safe_literal() {
        let font = uniform_font();
        let mut screen = writable();
        for c in "a < b & c's".chars() {
            key(&mut screen, c, &font);
        }
        screen.activate(ids::SIGN);
        for c in "  title  ".chars() {
            screen.key_typed(
                Some(c),
                0,
                Modifiers::NONE,
                &MemoryClipboard::default(),
                &font,
            );
        }
        screen.activate(ids::FINALIZE);
        let actions = screen.take_actions();
        let BookAction::Send { channel, stack } = &actions[0] else {
            panic!("send")
        };
        assert_eq!(*channel, "MC|BSign");
        assert_eq!(stack.item_id, 387);
        let tag = stack.nbt.as_ref().unwrap();
        assert_eq!(tag.get_str("author"), Some("Player"));
        assert_eq!(tag.get_str("title"), Some("title"));
        assert_eq!(
            tag.get_list("pages").unwrap()[0].as_str(),
            Some("\"a \\u003c b \\u0026 c\\u0027s\"")
        );
        assert!(matches!(actions[1], BookAction::Close));
        assert_eq!(
            serde_json::json!("a < b & c's").to_string(),
            "\"a < b & c's\"",
            "negative control reaches Gson's divergent HTML characters"
        );
    }

    #[test]
    fn signing_uses_java_trim_not_unicode_trim_for_title_controls_and_nbt() {
        let font = uniform_font();
        let mut screen = writable();
        screen.activate(ids::SIGN);
        key(&mut screen, '\u{2003}', &font);
        screen.init(&ctx(), &Language::new("en_US"));

        assert!(screen
            .buttons
            .iter()
            .find(|button| button.id == ids::FINALIZE)
            .is_some_and(|button| button.enabled));

        screen.activate(ids::FINALIZE);
        let actions = screen.take_actions();
        let BookAction::Send { channel, stack } = &actions[0] else {
            panic!("finalize sends a title containing Java-trimmable-only whitespace")
        };
        assert_eq!(*channel, "MC|BSign");
        assert_eq!(
            stack.nbt.as_ref().unwrap().get_str("title"),
            Some("\u{2003}")
        );
        assert!(matches!(actions[1], BookAction::Close));
    }

    #[test]
    fn escape_never_saves_even_when_modified() {
        let font = uniform_font();
        let mut screen = writable();
        key(&mut screen, 'x', &font);
        screen.key_typed(
            None,
            keys::ESCAPE,
            Modifiers::NONE,
            &MemoryClipboard::default(),
            &font,
        );
        assert_eq!(screen.take_actions(), vec![BookAction::Close]);
    }

    #[test]
    fn buttons_and_arrows_use_vanillas_geometry_and_visibility() {
        let mut screen = writable();
        screen.init(&ctx(), &Language::new("en_US"));
        assert_eq!(
            screen.next_rect,
            Rect::new((854 - 192) / 2 + 120, 156, 23, 13)
        );
        assert!(screen.next_visible());
        assert!(!screen.previous_visible());
        screen.activate(ids::NEXT);
        assert!(screen.previous_visible());
        screen.activate(ids::SIGN);
        assert!(!screen.next_visible());
        assert!(!screen.previous_visible());
    }

    #[test]
    fn change_page_click_is_one_based_and_run_command_closes_the_book() {
        let mut screen = writable();
        screen.add_new_page();
        let page = ResolvedStyle {
            click_event: Some(crate::font::component::ClickEvent {
                action: ClickAction::ChangePage,
                value: "2".into(),
            }),
            ..ResolvedStyle::default()
        };
        assert!(screen.handle_component_click(&page));
        assert_eq!(screen.current_page_index(), 1);
        assert!(
            !screen.handle_component_click(&page),
            "same-page click is not consumed"
        );

        let run = ResolvedStyle {
            click_event: Some(crate::font::component::ClickEvent {
                action: ClickAction::RunCommand,
                value: "/say hi".into(),
            }),
            ..ResolvedStyle::default()
        };
        assert!(screen.handle_component_click(&run));
        assert_eq!(
            screen.take_actions(),
            vec![BookAction::RunCommand("/say hi".into()), BookAction::Close]
        );
    }

    #[test]
    fn written_page_hit_testing_preserves_component_click_actions() {
        let mut tag = NbtCompound::new();
        tag.insert(
            "pages",
            Nbt::list(vec![Nbt::String(
                r#"{"text":"click","clickEvent":{"action":"run_command","value":"/go"}}"#.into(),
            )]),
        );
        tag.insert("title", Nbt::String("T".into()));
        tag.insert("author", Nbt::String("A".into()));
        let mut screen = BookScreen::new(
            ItemStack {
                item_id: 387,
                count: 1,
                damage: 0,
                nbt: Some(tag),
            },
            "Player",
            false,
        );
        let font = uniform_font();
        let lang = Language::new("en_US");
        screen.init(&ctx(), &lang);
        screen.mouse_clicked(
            (854 - 192) / 2 + PAGE_LEFT + 1,
            PAGE_TOP + 1,
            0,
            &font,
            &lang,
        );
        assert_eq!(
            screen.take_actions(),
            vec![BookAction::RunCommand("/go".into()), BookAction::Close]
        );

        // A click just beyond the glyphs is a real negative control for the x-accumulation.
        screen.mouse_clicked(
            (854 - 192) / 2 + PAGE_LEFT + 80,
            PAGE_TOP + 1,
            0,
            &font,
            &lang,
        );
        assert!(screen.take_actions().is_empty());
    }

    #[test]
    fn page_text_origin_matches_java_two_pixel_offset() {
        let mut tag = NbtCompound::new();
        tag.insert(
            "pages",
            Nbt::list(vec![Nbt::String(
                r#"{"text":"click","clickEvent":{"action":"run_command","value":"/go"}}"#.into(),
            )]),
        );
        tag.insert("title", Nbt::String("T".into()));
        tag.insert("author", Nbt::String("A".into()));
        let mut screen = BookScreen::new(
            ItemStack {
                item_id: 387,
                count: 1,
                damage: 0,
                nbt: Some(tag),
            },
            "Player",
            false,
        );
        let font = uniform_font();
        let lang = Language::new("en_US");
        screen.init(&ctx(), &lang);
        let x = (854 - 192) / 2 + PAGE_LEFT + 1;
        assert!(
            screen.style_at(x, 33, &font, &lang).is_none(),
            "Java hit testing rejects the row immediately above y = j + 16 + 16"
        );
        assert!(
            screen.style_at(x, 34, &font, &lang).is_some(),
            "Java page text and hit testing begin at y = 34"
        );
    }

    #[test]
    fn wrong_typed_pages_list_matches_get_tag_lists_empty_substitute() {
        let mut tag = NbtCompound::new();
        tag.insert("pages", Nbt::list(vec![Nbt::Int(42)]));
        tag.insert("title", Nbt::String("T".into()));
        tag.insert("author", Nbt::String("A".into()));
        let stack = ItemStack {
            item_id: 387,
            count: 1,
            damage: 0,
            nbt: Some(tag),
        };
        assert!(valid_written_book(&stack));
        let screen = BookScreen::new(stack, "Player", false);
        assert_eq!(screen.current_page(), "");
        assert_eq!(screen.total_pages(), 1);
    }

    #[test]
    fn renderer_emits_the_book_texture_and_blinking_cursor_changes_every_six_ticks() {
        let mut screen = writable();
        let mut font = uniform_font();
        let lang = Language::new("en_US");
        screen.init(&ctx(), &lang);
        let mut first = DrawList::new();
        screen.draw(&mut first, &mut font, &ctx(), (0, 0), &lang);
        assert!(first
            .batches()
            .iter()
            .any(|batch| batch.texture == UiTexture::gui(BOOK_TEXTURE)));
        for _ in 0..6 {
            Screen::tick(&mut screen);
        }
        let mut second = DrawList::new();
        screen.draw(&mut second, &mut font, &ctx(), (0, 0), &lang);
        assert_ne!(
            first, second,
            "the cursor color branch must affect emitted glyph vertices"
        );
    }
}
