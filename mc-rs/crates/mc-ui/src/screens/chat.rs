//! The chat log and the chat input screen.
//!
//! java: `client/gui/GuiNewChat.java`, `GuiChat.java`, `ChatLine.java`,
//! `GuiUtilRenderComponents.java`
//!
//! [`ChatLog`] is the persistent side — it survives the input screen opening and closing, holds
//! the 100-line scrollback, the wrapped display lines and the sent-message history.
//! [`ChatScreen`] is the transient side: one text field, tab completion, and up/down recall.
//!
//! # The two line lists
//!
//! Vanilla keeps *two* lists. `chatLines` holds the messages as they arrived; `drawnChatLines`
//! holds them **wrapped to the current chat width and inserted at index 0**, newest first. A
//! width or scale change calls `refreshChat`, which rebuilds the second from the first. Keeping
//! both is what makes re-wrapping lossless — wrapping the already-wrapped lines would compound
//! the line breaks.
//!
//! # History length
//!
//! Vanilla caps both lists at 100. The cap is [`ChatLog::set_max_lines`] (the Chat mod's
//! "History length", 100 to 5000, default 1000 in Aero; [`MAX_LINES`] until the client sets it).
//! Both lists are `VecDeque`s, newest at the front: a new line is an O(1) `push_front` where
//! Java's `ArrayList.add(0, ..)` (and the old `Vec::insert(0, ..)`) moved every older line, and a
//! re-wrap builds the newest lines first and stops when the cap is full. The draw and the hit
//! tests touch only the visible rows, so a long history costs nothing per frame.

use std::collections::VecDeque;
use std::ops::Range;

use mc_assets::Language;
use mc_data::{item_by_name, item_props, ItemId};
use mc_nbt::{parse_snbt_compound, Nbt, NbtCompound};
use mc_protocol::item::ItemStack;

use crate::font::component::{ChatComponent, ClickAction, HoverAction, ResolvedStyle, Span};
use crate::font::{strip_formatting, FontRenderer};
use crate::screen_stack::ScreenCtx;
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::keys::{self, Clipboard, Modifiers};
use crate::widget::{Rect, TextField, Transform};

// The `chat` mod (stacking, timestamps, backdrop, shadow, raise,
// copy). A child module so it can reach `ChatLog`'s private fields; vanilla while off.
mod m2;
pub use m2::ChatMods;
// Minecraft 1.7.10's chat clicks: a child module, for the log's lines and the input's layout.
#[path = "chat_v5.rs"]
mod v5;

/// Vanilla's scrollback cap, for both lists: the log's cap until [`ChatLog::set_max_lines`].
// java: GuiNewChat.java:162
pub const MAX_LINES: usize = 100;

/// The chat input's character limit. java: `setMaxStringLength(100)`
pub const MAX_INPUT_LENGTH: i32 = 100;

/// The chat input's limit in Aero: 1.11's `GuiChat` value, which multi-version servers
/// accept. Always on (it was once the "256 Character Chat" option).
pub const LONG_INPUT_LENGTH: i32 = 256;

/// Java's `String.trim()`: strips every leading and trailing char at or below `U+0020`, and
/// nothing else. `GuiChat`'s Enter sends `inputField.getText().trim()`; Rust's `str::trim` also
/// strips Unicode spaces such as `U+00A0` and `U+3000`, which vanilla sends.
#[must_use]
pub fn java_trim(text: &str) -> &str {
    text.trim_matches(|c: char| c <= ' ')
}

/// How many ticks a message stays visible with chat closed.
// java: GuiNewChat.java:61 — `if (j1 < 200 || flag)`
pub const FADE_TICKS: i32 = 200;

/// `calculateChatboxWidth`: `scale * (320 - 40) + 40`, floored.
// java: GuiNewChat.java:361
#[must_use]
pub fn chatbox_width(scale: f32) -> i32 {
    mc_math::floor_f32(scale * 280.0 + 40.0)
}

/// `calculateChatboxHeight`: `scale * (180 - 20) + 20`, floored.
// java: GuiNewChat.java:368
#[must_use]
pub fn chatbox_height(scale: f32) -> i32 {
    mc_math::floor_f32(scale * 160.0 + 20.0)
}

/// One wrapped display line.
// java: ChatLine.java
#[derive(Debug, Clone, PartialEq)]
pub struct ChatLine {
    /// The tick this line's message arrived, for the fade-out.
    pub updated_at: i32,
    /// The rendered text, formatting codes included.
    pub text: String,
    /// The deletion id. Non-zero ids can be replaced in place by a later message.
    pub id: i32,
    /// The wrapped component fragments, retaining the style events used by chat hit testing.
    spans: Vec<LineSpan>,
}

/// The host-facing result of resolving a chat component's hover event.
///
/// All supported actions resolve into the final tooltip lines immediately.  In particular,
/// `SHOW_ITEM` and the advanced-tooltip `SHOW_ENTITY` parse their stringified NBT here, exactly
/// where vanilla's `GuiScreen.handleComponentHover` consumes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatHoverContent {
    /// Already-formatted tooltip lines (`SHOW_TEXT`). Empty lines, including a trailing one, are
    /// retained because Guava's unspecialised newline splitter retains them too.
    Text(Vec<String>),
    /// Final `GuiScreen.handleComponentHover` lines for a valid statistic or achievement.
    Achievement(Vec<String>),
}

/// One `ChatComponentText` child produced by `GuiUtilRenderComponents.splitText`.
#[derive(Debug, Clone, PartialEq)]
struct LineSpan {
    /// The child's literal value. This includes the source style's formatting prefix, just like
    /// vanilla's `s4`, while [`Self::style`] retains the shallow-copied component style.
    text: String,
    style: ResolvedStyle,
}

/// Chat geometry derived from the settings — pulled out so the layout can be tested without a
/// `GameSettings`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatLayout {
    /// `chatWidth`, 0..1.
    pub width_scale: f32,
    /// `chatHeightFocused`, 0..1.
    pub height_focused: f32,
    /// `chatHeightUnfocused`, 0..1.
    pub height_unfocused: f32,
    /// `chatScale`, 0..1.
    pub scale: f32,
    /// `chatOpacity`, 0..1.
    pub opacity: f32,
    /// `chatColours`: whether legacy formatting codes embedded in component text survive.
    pub colors: bool,
    /// `chatVisibility`: 0 full, 1 system only, 2 hidden.
    pub visibility: i32,
}

impl Default for ChatLayout {
    fn default() -> Self {
        Self {
            width_scale: 1.0,
            height_focused: 1.0,
            height_unfocused: 0.443_661_96,
            scale: 1.0,
            opacity: 1.0,
            colors: true,
            visibility: 0,
        }
    }
}

impl ChatLayout {
    /// Read the seven chat fields out of the settings.
    #[must_use]
    pub fn from_settings(s: &mc_render::settings::GameSettings) -> Self {
        Self {
            width_scale: s.chat_width,
            height_focused: s.chat_height_focused,
            height_unfocused: s.chat_height_unfocused,
            scale: s.chat_scale,
            opacity: s.chat_opacity,
            colors: s.chat_colors,
            visibility: s.chat_visibility,
        }
    }

    /// The chat box's width in GUI pixels.
    #[must_use]
    pub fn width(&self) -> i32 {
        chatbox_width(self.width_scale)
    }

    /// The chat box's height, which differs depending on whether the input is open.
    #[must_use]
    pub fn height(&self, open: bool) -> i32 {
        chatbox_height(if open {
            self.height_focused
        } else {
            self.height_unfocused
        })
    }

    /// How many lines fit. java: `getChatHeight() / 9`
    #[must_use]
    pub fn line_count(&self, open: bool) -> i32 {
        self.height(open) / 9
    }

    /// The width messages are wrapped to: the box width divided by the chat scale, floored.
    // java: GuiNewChat.java:147
    #[must_use]
    pub fn wrap_width(&self) -> i32 {
        mc_math::floor_f32(self.width() as f32 / self.scale)
    }
}

/// Port of `GuiUtilRenderComponents.splitText` for the chat call site (`keepFormatting = false`).
///
/// Unlike `FontRenderer::wrap_to_lines`, this walks component spans separately, retains their
/// styles, and keeps the leading space when a continuation backs up to the last word boundary.
fn split_component(
    component: &ChatComponent,
    wrap_width: i32,
    font: &FontRenderer,
    lang: &Language,
    chat_colors: bool,
) -> Vec<Vec<LineSpan>> {
    let mut pending = component.spans(lang);
    let mut out = Vec::new();
    let mut line = Vec::new();
    let mut line_width = 0;
    let mut index = 0;

    while index < pending.len() {
        let source = pending[index].clone();
        let mut raw = source.text;
        let mut line_break = false;

        // java: split only the first newline and put its remainder immediately after this span.
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

        let mut rendered = source.style.formatting_code();
        rendered.push_str(&raw);
        if !chat_colors {
            rendered = strip_formatting(&rendered);
        }
        let without_newline = rendered.strip_suffix('\n').unwrap_or(&rendered);
        let mut piece = without_newline.to_owned();
        let mut piece_width = font.string_width(&piece);
        // One glyph placed alone on a line too narrow for it.
        let mut forced = false;

        if line_width + piece_width > wrap_width {
            piece = font.trim_to_width(&rendered, wrap_width - line_width);
            let cut_units = piece.encode_utf16().count();
            let rendered_units: Vec<u16> = rendered.encode_utf16().collect();
            let mut remainder = (cut_units < rendered_units.len())
                .then(|| String::from_utf16_lossy(&rendered_units[cut_units..]));

            // Not one glyph fits on an empty line: the wrap width is
            // below a glyph's (a negative or NaN `chatScale` or `chatWidth` in options.txt).
            // Vanilla then queues the same text again forever and runs out of memory; here the
            // first glyph goes on the line by itself instead, so every width ends. A line that
            // holds at least one glyph takes vanilla's path below unchanged.
            if line_width == 0 && strip_formatting(&piece).is_empty() {
                if let Some(rest) = remainder.as_mut().filter(|rest| !rest.is_empty()) {
                    let cut = first_glyph_end(rest);
                    piece.push_str(&rest[..cut]);
                    rest.replace_range(..cut, "");
                    forced = true;
                }
            }

            if let Some(rest) = remainder.as_mut().filter(|rest| !rest.is_empty()) {
                if let Some(space) = piece.rfind(' ') {
                    if font.string_width(&rendered[..space]) > 0 {
                        piece = rendered[..space].to_owned();
                        *rest = rendered[space..].to_owned();
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

        if forced || line_width + piece_width <= wrap_width {
            line_width += piece_width;
            line.push(LineSpan {
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

    // java always appends the accumulator, including an empty final component.
    out.push(line);
    out
}

/// Where `text`'s first visible glyph ends (in bytes), the formatting
/// codes in front of it (`§` and the character after it) included; all of `text` if it has no
/// glyph.
fn first_glyph_end(text: &str) -> usize {
    let mut chars = text.char_indices();
    while let Some((_, c)) = chars.next() {
        if c == '\u{a7}' {
            chars.next();
            continue;
        }
        return chars.next().map_or(text.len(), |(end, _)| end);
    }
    text.len()
}

/// `ChatComponentStyle.getFormattedText` for the empty line root plus its split children.
fn formatted_line_text(spans: &[LineSpan]) -> String {
    let mut text = String::from("\u{a7}r");
    for span in spans {
        text.push_str(&span.style.formatting_code());
        text.push_str(&span.text);
        text.push_str("\u{a7}r");
    }
    text
}

/// The text `TileEntitySignRenderer` draws for one sign row.
///
/// java: `TileEntitySignRenderer.renderTileEntityAt:95-96` —
/// `GuiUtilRenderComponents.splitText(row, width, fontrenderer, false, true)`, then the first
/// line's `getFormattedText()` (`""` only for an empty list, which `splitText` never returns).
/// This is the chat call site's `split_component` with the fifth argument `true`, so
/// `func_178909_a` keeps `§` codes whatever `chatColours` says.
///
/// `splitText` walks the row's root **and every sibling**: its work list is
/// `Lists.newArrayList(Iterable)` over the component (the `Iterable` overload is what the jar's
/// bytecode calls, `avu.a` offset 20), and `ChatComponentStyle.iterator` yields the root and then
/// its deep-copied `extra` tree. Many servers send every sign row as an empty root with the
/// text in `extra` (`{"extra":["text"],"text":""}`), so a root-only reading draws those signs
/// blank.
#[must_use]
pub fn sign_row_text(
    component: &ChatComponent,
    width: i32,
    font: &FontRenderer,
    lang: &Language,
) -> String {
    let lines = split_component(component, width, font, lang, true);
    lines
        .first()
        .map_or_else(String::new, |line| formatted_line_text(line))
}

/// The persistent chat log.
#[derive(Debug, Clone)]
pub struct ChatLog {
    /// Messages as received, newest first.
    received: VecDeque<(i32, ChatComponent, i32)>,
    /// Wrapped display lines, newest first.
    drawn: VecDeque<ChatLine>,
    /// Messages the player sent, oldest first, for up/down recall.
    sent: Vec<String>,
    scroll: usize,
    scrolled: bool,
    /// The `chat` mod's options; the default is vanilla.
    mods: ChatMods,
    /// The last id-0 message, for stacking repeats.
    stack: m2::StackState,
    /// The cap on both lists: vanilla's [`MAX_LINES`] until [`ChatLog::set_max_lines`].
    max_lines: usize,
    /// The Unix time (milliseconds) the `chat` mod's timestamps show
    /// instead of the wall clock ([`ChatLog::set_stamp_clock`]).
    stamp_clock: Option<i64>,
}

impl Default for ChatLog {
    fn default() -> Self {
        Self {
            received: VecDeque::new(),
            drawn: VecDeque::new(),
            sent: Vec::new(),
            scroll: 0,
            scrolled: false,
            mods: ChatMods::default(),
            stack: m2::StackState::default(),
            max_lines: MAX_LINES,
            stamp_clock: None,
        }
    }
}

/// The local `(hour, minute)` a timestamp shows for a Unix time in
/// milliseconds ([`ChatLog::set_stamp_clock`]).
#[must_use]
pub fn stamp_clock_at(unix_ms: i64) -> (u32, u32) {
    m2::local_clock_at(unix_ms)
}

impl ChatLog {
    /// An empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The wrapped display lines, newest first.
    #[must_use]
    pub fn lines(&self) -> &VecDeque<ChatLine> {
        &self.drawn
    }

    /// The cap on the received messages and on the display lines (vanilla 100).
    #[must_use]
    pub fn max_lines(&self) -> usize {
        self.max_lines
    }

    /// Stamp the next messages (the `chat` mod's timestamps) with this
    /// Unix time in milliseconds instead of the wall clock; `None` puts the wall clock back. A
    /// replay's chat is stamped with the recording's clock, so a line shows when it came in, not
    /// when it is watched, and every viewing, seek and export of a moment reads the same.
    pub fn set_stamp_clock(&mut self, unix_ms: Option<i64>) {
        self.stamp_clock = unix_ms;
    }

    /// Set the history length. A smaller cap drops the oldest
    /// messages and lines at once; a larger one keeps everything and lets the log grow. The
    /// newest message, an open stack and the reader's scroll stay valid either way.
    pub fn set_max_lines(&mut self, lines: usize) {
        let lines = lines.max(1);
        if lines == self.max_lines {
            return;
        }
        self.max_lines = lines;
        let cut = self.received.len() > lines || self.drawn.len() > lines;
        self.received.truncate(lines);
        self.drawn.truncate(lines);
        if cut {
            // The oldest lines went; a reader scrolled up to them starts again at the bottom, as
            // after vanilla's `refreshChat`.
            self.reset_scroll();
        }
    }

    /// How many messages have been received (before wrapping).
    #[must_use]
    pub fn received_count(&self) -> usize {
        self.received.len()
    }

    /// The player's sent-message history, oldest first.
    #[must_use]
    pub fn sent_messages(&self) -> &[String] {
        &self.sent
    }

    /// Current scroll position, in lines from the bottom.
    #[must_use]
    pub fn scroll_pos(&self) -> usize {
        self.scroll
    }

    /// Whether the log has been scrolled away from the bottom.
    #[must_use]
    pub fn is_scrolled(&self) -> bool {
        self.scrolled
    }

    /// Drop everything. java: `clearChatMessages`
    pub fn clear(&mut self) {
        self.received.clear();
        self.drawn.clear();
        self.sent.clear();
        self.reset_scroll();
        self.stack.valid = false;
    }

    /// Record a message the player sent, skipping an immediate duplicate.
    // java: GuiNewChat.java:200
    pub fn add_sent(&mut self, message: &str) {
        if self.sent.last().map(String::as_str) != Some(message) {
            self.sent.push(message.to_owned());
        }
    }

    /// Add a received message.
    ///
    /// `id` is `chatLineId`: non-zero replaces any existing line with the same id, which is how
    /// the tab-completion hint line updates in place.
    // java: GuiNewChat.java:140 (setChatLine)
    #[allow(clippy::too_many_arguments)]
    pub fn add_message(
        &mut self,
        component: ChatComponent,
        id: i32,
        tick: i32,
        layout: &ChatLayout,
        font: &FontRenderer,
        lang: &Language,
        chat_open: bool,
    ) {
        // The `chat` mod's stacking/timestamps (id 0 only). Off: one test, and the
        // stack is closed so a later repeat cannot reach past this message.
        if id == 0 && self.mods.rewrites_messages() {
            let clock = if self.mods.timestamps != 0 {
                self.stamp_clock
                    .map_or_else(m2::local_clock, m2::local_clock_at)
            } else {
                (0, 0)
            };
            self.add_message_modded(component, tick, layout, font, lang, chat_open, clock);
            return;
        }
        self.stack.valid = false;
        if id != 0 {
            self.delete_by_id(id);
        }
        self.push_drawn(&component, id, tick, layout, font, lang, chat_open);
        self.received.push_front((tick, component, id));
        self.received.truncate(self.max_lines);
    }

    /// Wrap and insert one message's display lines; returns how many it produced.
    #[allow(clippy::too_many_arguments)]
    fn push_drawn(
        &mut self,
        component: &ChatComponent,
        id: i32,
        tick: i32,
        layout: &ChatLayout,
        font: &FontRenderer,
        lang: &Language,
        chat_open: bool,
    ) -> usize {
        let mut count = 0;
        for spans in split_component(component, layout.wrap_width(), font, lang, layout.colors) {
            count += 1;
            // java: while the chat is open and scrolled, a new line pushes the view up by one
            // so the reader keeps looking at the same message.
            if chat_open && self.scroll > 0 {
                self.scrolled = true;
                self.scroll_by(1, layout, chat_open);
            }
            let text = formatted_line_text(&spans);
            self.drawn.push_front(ChatLine {
                updated_at: tick,
                text,
                id,
                spans,
            });
        }
        self.drawn.truncate(self.max_lines);
        count
    }

    /// Re-wrap every message after a width, scale or language change.
    ///
    /// java walks the messages oldest first, inserting each one's lines at index 0 and trimming
    /// to the cap, so what survives is the newest `max_lines` display lines. This builds the same
    /// list from the other end: newest message first, each one's lines bottom row first, pushed
    /// at the back, stopping as soon as the cap is full. Linear in the lines kept, where the
    /// front insertion was quadratic.
    // java: GuiNewChat.java:178 (refreshChat)
    pub fn refresh(&mut self, layout: &ChatLayout, font: &FontRenderer, lang: &Language) {
        self.drawn.clear();
        self.reset_scroll();
        let wrap_width = layout.wrap_width();
        let mut newest_lines = 0;
        for (i, (tick, component, id)) in self.received.iter().enumerate() {
            let lines = split_component(component, wrap_width, font, lang, layout.colors);
            if i == 0 {
                newest_lines = lines.len();
            }
            for spans in lines.into_iter().rev() {
                if self.drawn.len() == self.max_lines {
                    break;
                }
                let text = formatted_line_text(&spans);
                self.drawn.push_back(ChatLine {
                    updated_at: *tick,
                    text,
                    id: *id,
                    spans,
                });
            }
            if self.drawn.len() == self.max_lines {
                break;
            }
        }
        // The newest message (a stack, if one is open) was wrapped first.
        self.stack.lines = newest_lines;
    }

    /// Remove every line with this id.
    // java: GuiNewChat.java:315 (deleteChatLine)
    pub fn delete_by_id(&mut self, id: i32) {
        self.stack.valid = false;
        self.drawn.retain(|l| l.id != id);
        // java: the *received* list only drops the first match, then breaks.
        if let Some(i) = self.received.iter().position(|(_, _, l)| *l == id) {
            self.received.remove(i);
        }
    }

    /// Reset the scroll. java: `resetScroll`
    pub fn reset_scroll(&mut self) {
        self.scroll = 0;
        self.scrolled = false;
    }

    /// Scroll by `amount` lines, clamped.
    // java: GuiNewChat.java:222
    pub fn scroll_by(&mut self, amount: i32, layout: &ChatLayout, chat_open: bool) {
        let visible = layout.line_count(chat_open);
        let mut pos = self.scroll as i32 + amount;
        let max = self.drawn.len() as i32 - visible;
        if pos > max {
            pos = max;
        }
        if pos <= 0 {
            pos = 0;
            self.scrolled = false;
        }
        self.scroll = pos as usize;
    }

    /// The lines that are on screen right now, top-most first in draw order (index 0 is the
    /// bottom row, as vanilla draws upwards).
    #[must_use]
    pub fn visible_lines(&self, layout: &ChatLayout, chat_open: bool) -> Vec<&ChatLine> {
        self.drawn
            .range(self.visible_range(layout, chat_open))
            .collect()
    }

    /// The indices into [`ChatLog::lines`] of the rows on screen: the scroll position and the
    /// rows that fit below it. The draw and the bounds use it directly, without a per-frame
    /// list, so a long history costs nothing per frame.
    fn visible_range(&self, layout: &ChatLayout, chat_open: bool) -> Range<usize> {
        let n = layout.line_count(chat_open).max(0) as usize;
        let start = self.scroll.min(self.drawn.len());
        start..(start + n).min(self.drawn.len())
    }

    /// A line's alpha at `tick`, in 0..=255, or `None` when it has faded out.
    ///
    /// Vanilla's curve: `((1 - age/200) * 10)` clamped to 0..1, **squared**, times 255 — so a
    /// message is fully opaque for its first 20 ticks and then falls off quadratically.
    // java: GuiNewChat.java:63
    #[must_use]
    pub fn line_alpha(
        &self,
        line: &ChatLine,
        tick: i32,
        layout: &ChatLayout,
        chat_open: bool,
    ) -> Option<i32> {
        let age = tick - line.updated_at;
        if age >= FADE_TICKS && !chat_open {
            return None;
        }
        let alpha = if chat_open {
            255.0
        } else {
            let d = (1.0 - f64::from(age) / 200.0) * 10.0;
            let d = d.clamp(0.0, 1.0);
            255.0 * d * d
        };
        let alpha = (alpha as i32 as f32 * (layout.opacity * 0.9 + 0.1)) as i32;
        (alpha > 3).then_some(alpha)
    }

    /// Draw the chat box.
    ///
    /// The anchor is the composition of two translates: `GuiIngame` does
    /// `translate(0, height - 48)` before calling `drawChat`, and `GuiNewChat` then does
    /// `translate(2, 20)` — so the bottom row's backdrop ends at `height - 28`, six pixels
    /// above the hotbar, and rows walk *upwards* from there.
    ///
    /// The anchor and `chatScale` are applied through `DrawList`'s transform stack, so backdrop,
    /// nine-pixel row pitch, glyphs and the open-chat scrollbar all scale together.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        tick: i32,
        layout: &ChatLayout,
        chat_open: bool,
    ) {
        mc_core::profile_scope!("hud.chat");
        if layout.visibility == 2 {
            return;
        }
        let box_width = mc_math::ceil_f32(layout.width() as f32 / layout.scale);
        let visible = self.visible_range(layout, chat_open);
        // `raise`, the backdrop factor and the shadow are the `chat` mod's; their
        // defaults (0, 1.0, true) are exactly vanilla's numbers.
        d.push_transform(
            Transform::translate(2.0, (ctx.height - 28 - self.mods.raise) as f32)
                .then_scale(layout.scale),
        );
        for (row, line) in self.drawn.range(visible.clone()).enumerate() {
            let Some(alpha) = self.line_alpha(line, tick, layout, chat_open) else {
                continue;
            };
            let y = -(row as i32) * 9;
            let backdrop = self.mods.backdrop_alpha(alpha);
            if backdrop != 0 {
                d.rect(0, y - 9, box_width + 4, y, backdrop << 24);
            }
            font.draw_string(
                d,
                &line.text,
                0.0,
                (y - 8) as f32,
                0x00FF_FFFF | ((alpha as u32) << 24) as i32,
                self.mods.shadow,
            );
        }

        // java: GuiNewChat.java:93-109. With chat open every visited line passes the fade test,
        // so `j` is exactly this visible slice's length (and is counted before alpha > 3).
        if chat_open && !self.drawn.is_empty() {
            let total = self.drawn.len() as i32;
            let shown = visible.len() as i32;
            let pitch = font.font_height();
            let full_height = total * pitch + total;
            let shown_height = shown * pitch + shown;
            let scroll_top = self.scroll as i32 * shown_height / total;
            let thumb_height = shown_height * shown_height / full_height;
            if full_height != shown_height {
                let alpha = if scroll_top > 0 { 170 } else { 96 };
                let fill = if self.scrolled { 13_382_451 } else { 3_355_562 };
                d.rect(
                    -3,
                    -scroll_top,
                    -1,
                    -scroll_top - thumb_height,
                    fill | ((alpha as u32) << 24),
                );
                d.rect(
                    -1,
                    -scroll_top,
                    -2,
                    -scroll_top - thumb_height,
                    13_421_772 | ((alpha as u32) << 24),
                );
            }
        }
        d.pop_transform();
    }

    /// The line under a GUI-space cursor position, if the chat is open.
    ///
    /// Vanilla works in *window* pixels here and divides by the scale factor itself; this takes
    /// GUI pixels, because [`ScaledResolution::cursor_to_gui`](crate::widget::ScaledResolution)
    /// has already done that conversion.
    // java: GuiNewChat.java:245 (getChatComponent)
    #[must_use]
    pub fn line_at(
        &self,
        gui_x: i32,
        gui_y: i32,
        ctx: &ScreenCtx,
        layout: &ChatLayout,
        chat_open: bool,
    ) -> Option<&ChatLine> {
        if !chat_open {
            return None;
        }
        let x = mc_math::floor_f32((gui_x - 3) as f32 / layout.scale);
        // Vanilla's `mouseY / i - 27` is measured from the *bottom* of the window, because LWJGL
        // reports the cursor with y growing upwards. In GUI space y grows downwards, so the
        // equivalent is the distance up from the chat anchor.
        let from_bottom = ctx.height - gui_y;
        let y = mc_math::floor_f32((from_bottom - 27 - self.mods.raise) as f32 / layout.scale);
        if x < 0 || y < 0 {
            return None;
        }
        self.line_at_chat_coords(x, y, layout, chat_open)
    }

    /// The line under a raw LWJGL-space cursor, using vanilla's integer scale-factor arithmetic.
    ///
    /// `mouse_x`/`mouse_y` are physical window pixels with the origin at the **bottom left**.
    /// This distinction matters when a window dimension is not divisible by the GUI scale:
    /// converting to the ceil-rounded scaled surface first loses information and can move the
    /// hit by one GUI pixel. This is the drop-in geometry for `GuiNewChat.getChatComponent`.
    // java: GuiNewChat.java:245 (getChatComponent)
    #[must_use]
    pub fn line_at_window(
        &self,
        mouse_x: i32,
        mouse_y: i32,
        ctx: &ScreenCtx,
        layout: &ChatLayout,
        chat_open: bool,
    ) -> Option<&ChatLine> {
        if !chat_open {
            return None;
        }
        let factor = ctx.resolution.scale_factor().max(1);
        // Java integer division truncates toward zero, as Rust's signed integer division does.
        let x = mc_math::floor_f32((mouse_x / factor - 3) as f32 / layout.scale);
        let y = mc_math::floor_f32((mouse_y / factor - 27 - self.mods.raise) as f32 / layout.scale);
        self.line_at_chat_coords(x, y, layout, chat_open)
    }

    fn line_at_chat_coords(
        &self,
        x: i32,
        y: i32,
        layout: &ChatLayout,
        chat_open: bool,
    ) -> Option<&ChatLine> {
        if x < 0 || y < 0 {
            return None;
        }
        let visible = layout.line_count(chat_open).min(self.drawn.len() as i32);
        if x > layout.wrap_width() || y >= 9 * visible + visible {
            return None;
        }
        let index = (y / 9) as usize + self.scroll;
        self.drawn.get(index)
    }

    /// The resolved component style under a GUI-space cursor position.
    ///
    /// This is the per-child width walk from `GuiNewChat.getChatComponent`. Returning the style
    /// (rather than only the line) keeps click, hover, and insertion events available to the host.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn style_at(
        &self,
        gui_x: i32,
        gui_y: i32,
        ctx: &ScreenCtx,
        layout: &ChatLayout,
        chat_open: bool,
        font: &FontRenderer,
    ) -> Option<&ResolvedStyle> {
        let line = self.line_at(gui_x, gui_y, ctx, layout, chat_open)?;
        let x = mc_math::floor_f32((gui_x - 3) as f32 / layout.scale);
        Self::style_in_line(line, x, font)
    }

    /// The resolved style under a raw LWJGL-space cursor.
    ///
    /// This is the exact `getChatComponent` input contract. Prefer it over [`Self::style_at`]
    /// whenever raw cursor pixels are available.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn style_at_window(
        &self,
        mouse_x: i32,
        mouse_y: i32,
        ctx: &ScreenCtx,
        layout: &ChatLayout,
        chat_open: bool,
        font: &FontRenderer,
    ) -> Option<&ResolvedStyle> {
        let line = self.line_at_window(mouse_x, mouse_y, ctx, layout, chat_open)?;
        let factor = ctx.resolution.scale_factor().max(1);
        let x = mc_math::floor_f32((mouse_x / factor - 3) as f32 / layout.scale);
        Self::style_in_line(line, x, font)
    }

    fn style_in_line<'a>(
        line: &'a ChatLine,
        x: i32,
        font: &FontRenderer,
    ) -> Option<&'a ResolvedStyle> {
        let mut right = 0;
        for span in &line.spans {
            right += font.string_width(&span.text);
            if right > x {
                return Some(&span.style);
            }
        }
        None
    }

    /// Resolve the hover payload under a scaled, top-left-origin GUI cursor.
    ///
    /// This is the hover counterpart to [`Self::style_at`]. Hosts that already retain GUI-space
    /// cursor coordinates must use it directly: reconstructing raw LWJGL pixels for
    /// [`Self::hover_at_window`] loses information whenever a physical dimension is not divisible
    /// by the GUI scale factor.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn hover_at(
        &self,
        gui_x: i32,
        gui_y: i32,
        ctx: &ScreenCtx,
        layout: &ChatLayout,
        chat_open: bool,
        advanced_tooltips: bool,
        inventory_key_name: &str,
        font: &FontRenderer,
        lang: &Language,
    ) -> Option<ChatHoverContent> {
        let style = self.style_at(gui_x, gui_y, ctx, layout, chat_open, font)?;
        hover_content(style, advanced_tooltips, inventory_key_name, font, lang)
    }

    /// Resolve the hover payload under a raw LWJGL-space cursor.
    ///
    /// Vanilla suppresses `SHOW_ENTITY` completely unless advanced item tooltips are enabled.
    /// The remaining payload interpretation is represented by [`ChatHoverContent`] so the host
    /// can use the shared item/stat registries without duplicating hit-test logic.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn hover_at_window(
        &self,
        mouse_x: i32,
        mouse_y: i32,
        ctx: &ScreenCtx,
        layout: &ChatLayout,
        chat_open: bool,
        advanced_tooltips: bool,
        inventory_key_name: &str,
        font: &FontRenderer,
        lang: &Language,
    ) -> Option<ChatHoverContent> {
        let style = self.style_at_window(mouse_x, mouse_y, ctx, layout, chat_open, font)?;
        hover_content(style, advanced_tooltips, inventory_key_name, font, lang)
    }

    /// The whole box's rectangle, for hit tests and the F3 overlay.
    #[must_use]
    pub fn bounds(&self, ctx: &ScreenCtx, layout: &ChatLayout, chat_open: bool) -> Rect {
        let rows = self.visible_range(layout, chat_open).len() as i32;
        let h = rows * 9;
        Rect::new(
            2,
            ctx.height - 28 - self.mods.raise - h,
            layout.width() + 4,
            h,
        )
    }
}

/// The chat input screen.
// java: GuiChat.java:18
#[derive(Debug)]
pub struct ChatScreen {
    /// The input field.
    pub input: TextField,
    /// Where the up/down history cursor is. Equal to `sent.len()` means "the live buffer".
    history_cursor: usize,
    history_buffer: String,
    /// Completion candidates from the server's last `S3APacketTabComplete`.
    completions: Vec<String>,
    completion_index: usize,
    /// Whether a completion cycle is in progress.
    completing: bool,
    /// Whether a completion request is outstanding.
    waiting: bool,
    /// Whether the current server response took Java's candidate-list hint branch.
    ///
    /// A response that extends the input with a longer common prefix does not print the
    /// id-1 candidate line. The driver consumes this producer-side bit after routing the
    /// response, rather than inferring the branch from the filtered candidate count.
    completion_hint: bool,
    // `new` has no screen width, so its zero-width TextField computes the wrong visible window.
    // Recompute it once `init` has installed the real dimensions and a font reaches us.
    needs_input_relayout: bool,
}

impl ChatScreen {
    /// A chat screen pre-filled with `default_text` (`/` for the command key).
    #[must_use]
    pub fn new(default_text: &str, sent_count: usize, font: &FontRenderer) -> Self {
        Self::with_input_limit(default_text, sent_count, font, MAX_INPUT_LENGTH)
    }

    /// [`Self::new`] with the input capped at `limit` UTF-16 units instead of vanilla's 100
    /// ([`LONG_INPUT_LENGTH`] in Aero, always). The cap is set before the pre-fill, so
    /// a long suggested command keeps up to `limit` units, as vanilla keeps up to 100.
    #[must_use]
    pub fn with_input_limit(
        default_text: &str,
        sent_count: usize,
        font: &FontRenderer,
        limit: i32,
    ) -> Self {
        // java: `new GuiTextField(0, font, 4, height - 12, width - 4, 12)` with no background.
        let mut input = TextField::new(0, 4, 0, 0, 12);
        input.set_max_length(limit);
        input.background = false;
        input.can_lose_focus = false;
        input.set_focused(true);
        input.set_text(default_text, font);
        Self {
            input,
            history_cursor: sent_count,
            history_buffer: String::new(),
            completions: Vec::new(),
            completion_index: 0,
            completing: false,
            waiting: false,
            completion_hint: false,
            needs_input_relayout: true,
        }
    }

    /// Lay out.
    pub fn init(&mut self, ctx: &ScreenCtx) {
        let rect = Rect::new(4, ctx.height - 12, ctx.width - 4, 12);
        if self.input.rect != rect {
            self.input.rect = rect;
            // `build_ui` calls this every frame, while vanilla only runs `initGui` on an open or
            // resized screen. Keep the relayout marker tied to an actual geometry change so a
            // live selection is not collapsed merely because a frame was drawn.
            self.needs_input_relayout = true;
        }
    }

    /// Advance the caret blink.
    pub fn tick(&mut self) {
        self.input.tick();
    }

    /// Handle a key press.
    #[allow(clippy::too_many_arguments)]
    pub fn key_typed(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        clipboard: &mut dyn Clipboard,
        font: &FontRenderer,
        log: &mut ChatLog,
        layout: &ChatLayout,
    ) -> Action {
        self.waiting = false;
        // The hint describes this key only: a cycling Tab sets it again.
        self.completion_hint = false;
        if key == keys::TAB {
            return self.request_or_cycle_completion(font);
        }
        // java: any key other than Tab abandons the completion cycle.
        self.completing = false;

        match key {
            keys::ESCAPE => Action::Back,
            keys::ENTER | keys::NUMPAD_ENTER => {
                let text = java_trim(&self.input.text()).to_owned();
                if text.is_empty() {
                    return Action::Back;
                }
                log.add_sent(&text);
                Action::SendChat(text)
            }
            keys::UP => {
                self.recall(-1, log, font);
                Action::None
            }
            keys::DOWN => {
                self.recall(1, log, font);
                Action::None
            }
            // Page Up / Page Down scroll the log by a screenful less one line.
            201 => {
                log.scroll_by(layout.line_count(true) - 1, layout, true);
                Action::None
            }
            209 => {
                log.scroll_by(-(layout.line_count(true) - 1), layout, true);
                Action::None
            }
            _ => {
                self.input.key_typed(typed, key, mods, clipboard, font);
                Action::None
            }
        }
    }

    /// Vanilla's wheel handling: one notch is seven lines, or one with Shift held.
    // java: GuiChat.java:143
    pub fn mouse_wheel(&mut self, delta: f32, shift: bool, log: &mut ChatLog, layout: &ChatLayout) {
        if delta == 0.0 {
            return;
        }
        let mut n = if delta > 0.0 { 1 } else { -1 };
        if !shift {
            n *= 7;
        }
        log.scroll_by(n, layout, true);
    }

    /// Forward a mouse press to the chat input so a left click places the caret.
    // java: GuiChat.java:184
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32, font: &FontRenderer) -> Action {
        // GuiTextField only reads the font for a left click inside its bounds. Avoiding a
        // relayout for outside/right clicks preserves Java's selection anchor (and matters when
        // the first click arrives before the initial draw has consumed the init marker).
        if button == 0 && self.input.rect.contains(x, y) {
            self.ensure_input_layout(font);
        }
        self.input.mouse_clicked(x, y, button, font);
        Action::None
    }

    /// Apply vanilla's `GuiScreen.handleComponentClick` semantics to an already hit-tested
    /// component style.
    ///
    /// `Some(action)` means the component consumed the click. `Some(Action::None)` is meaningful:
    /// vanilla consumes `CHANGE_PAGE` even on a non-book screen after logging that it cannot
    /// handle it. `None` means the input field should still receive the click. Shift-insertion
    /// deliberately returns `None` after modifying the input, matching `GuiChat.mouseClicked`.
    #[must_use]
    pub fn component_clicked(
        &mut self,
        style: &ResolvedStyle,
        shift: bool,
        chat_links: bool,
        font: &FontRenderer,
    ) -> Option<Action> {
        if shift {
            if let Some(insertion) = &style.insertion {
                self.ensure_input_layout(font);
                self.input.write_text(insertion, font);
            }
            return None;
        }

        let click = style.click_event.as_ref()?;
        if click.action == ClickAction::OpenUrl && !chat_links {
            return None;
        }
        Some(match click.action {
            ClickAction::OpenUrl => Action::OpenUrl(click.value.clone()),
            ClickAction::RunCommand => Action::RunCommand(click.value.clone()),
            ClickAction::SuggestCommand => Action::SuggestCommand(click.value.clone()),
            // These actions cannot survive server-component parsing in 1.8.9. Keeping the arms
            // explicit makes this method correct for locally-created components too: vanilla
            // consumes every recognised click event even when the current screen cannot act.
            // The client acts on its own `OpenFile` (the F2 chat line) and `CopyImage` ([COPY])
            // links before this method is reached (`x2_chat_click` in mc-client).
            ClickAction::OpenFile
            | ClickAction::CopyImage
            | ClickAction::TwitchUserInfo
            | ClickAction::ChangePage => Action::None,
        })
    }

    fn ensure_input_layout(&mut self, font: &FontRenderer) {
        if self.needs_input_relayout {
            let cursor = self.input.cursor_position();
            self.input.set_cursor_position(cursor, font);
            self.needs_input_relayout = false;
        }
    }

    /// The word before the caret, which is what gets completed.
    // java: GuiTextField.func_146197_a(-1, cursor, false)
    #[must_use]
    pub fn completion_prefix(&self) -> String {
        let text = self.input.text();
        let cursor = self.input.cursor_position();
        let start = self.input.word_from_pos(-1, cursor, false);
        utf16_range(&text, start, cursor)
    }

    fn request_or_cycle_completion(&mut self, font: &FontRenderer) -> Action {
        if self.completing && !self.completions.is_empty() {
            self.apply_completion(font);
            // java: `GuiChat.autocompletePlayerNames:232-247` prints the
            // candidate line (`printChatMessageWithOptionalDeletion(..., 1)`) on every cycling
            // Tab press while more than one name was found, not only when S3A starts the cycle.
            self.completion_hint = self.completions.len() > 1;
            return Action::None;
        }
        let head = utf16_prefix(&self.input.text(), self.input.cursor_position());
        if head.is_empty() {
            return Action::None;
        }
        self.completions.clear();
        self.completion_index = 0;
        self.waiting = true;
        Action::RequestTabComplete(head)
    }

    /// Feed the server's completion candidates in.
    ///
    /// Vanilla first inserts the longest **common prefix** of all candidates; only when that
    /// adds nothing does it start cycling through them one at a time.
    // java: GuiChat.java:315 (onAutocompleteResponse)
    pub fn on_completions(&mut self, candidates: &[String], font: &FontRenderer) {
        self.completion_hint = false;
        if !self.waiting {
            return;
        }
        self.completing = false;
        // java computes StringUtils.getCommonPrefix over the packet's original array, while the
        // cycling cache below deliberately drops empty entries. Those are observably different
        // for a mixed `["name", ""]` reply: the empty entry suppresses prefix insertion, then
        // the one non-empty entry begins a local cycle.
        let common_units = common_prefix_units(candidates);
        self.completions = candidates
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        if self.completions.is_empty() {
            return;
        }
        let current = self.completion_prefix();
        let current_units = current.encode_utf16().collect::<Vec<_>>();
        if !common_units.is_empty() && !java_utf16_equals_ignore_case(&current_units, &common_units)
        {
            self.replace_prefix_units(&common_units, font);
        } else {
            self.completing = true;
            self.apply_completion(font);
            self.completion_hint = self.completions.len() > 1;
        }
    }

    fn apply_completion(&mut self, font: &FontRenderer) {
        if self.completions.is_empty() {
            return;
        }
        if self.completion_index >= self.completions.len() {
            self.completion_index = 0;
        }
        let candidate = self.completions[self.completion_index].clone();
        self.completion_index += 1;
        self.replace_prefix(&candidate, font);
        self.completing = true;
    }

    fn replace_prefix(&mut self, replacement: &str, font: &FontRenderer) {
        // TextField positions are Java UTF-16 units, not Rust scalar values. In particular, an
        // astral character before the caret occupies two units while still being one `char`.
        // Keep the deletion count in the same coordinate system as the field so the suffix
        // remains untouched when a completion is applied in the middle of a message.
        let cursor = self.input.cursor_position();
        let start = self.input.word_from_pos(-1, cursor, false);
        let removed = cursor - start;
        if removed > 0 {
            self.input.delete_from_cursor(-removed, font);
        }
        self.input.write_text(replacement, font);
    }

    fn replace_prefix_units(&mut self, replacement: &[u16], font: &FontRenderer) {
        let cursor = self.input.cursor_position();
        let start = self.input.word_from_pos(-1, cursor, false);
        let removed = cursor - start;
        if removed > 0 {
            self.input.delete_from_cursor(-removed, font);
        }
        self.input.write_utf16_units(replacement, font);
    }

    /// All current completion candidates, for the hint line.
    #[must_use]
    pub fn completions(&self) -> &[String] {
        &self.completions
    }

    /// Whether the most recent response, or the most recent key, printed Java's id-1 candidate
    /// hint.
    ///
    /// This is intentionally a one-event producer flag. The live driver reads it immediately
    /// after [`Self::on_completions`] and after a Tab reaches [`Self::key_typed`]; each of those
    /// resets it before choosing its own branch, and only the cycling branch (with more than one
    /// candidate) sets it, as `GuiChat.autocompletePlayerNames` prints.
    #[must_use]
    pub fn completion_hint(&self) -> bool {
        self.completion_hint
    }

    /// Up/down history recall.
    // java: GuiChat.java:272 (getSentHistory)
    pub fn recall(&mut self, delta: i32, log: &ChatLog, font: &FontRenderer) {
        let n = log.sent_messages().len();
        let target = (self.history_cursor as i32 + delta).clamp(0, n as i32) as usize;
        if target == self.history_cursor {
            return;
        }
        if target == n {
            self.history_cursor = n;
            let buffer = self.history_buffer.clone();
            self.input.set_text(&buffer, font);
        } else {
            if self.history_cursor == n {
                self.history_buffer = self.input.text();
            }
            let text = log.sent_messages()[target].clone();
            self.input.set_text(&text, font);
            self.history_cursor = target;
        }
    }

    /// Draw the input box. The log itself is drawn by [`ChatLog::draw`].
    // java: GuiChat.java:301
    pub fn draw(&mut self, d: &mut DrawList, font: &mut FontRenderer, ctx: &ScreenCtx) {
        self.ensure_input_layout(font);
        d.rect(
            2,
            ctx.height - 14,
            ctx.width - 2,
            ctx.height - 2,
            0x8000_0000,
        );
        self.input.draw(d, font);
    }
}

/// Interpret one resolved hover style using `GuiScreen.handleComponentHover`'s action and gate.
#[must_use]
pub fn hover_content(
    style: &ResolvedStyle,
    advanced_tooltips: bool,
    inventory_key_name: &str,
    font: &FontRenderer,
    lang: &Language,
) -> Option<ChatHoverContent> {
    let hover = style.hover_event.as_ref()?;
    let value = hover.value.as_ref();
    match hover.action {
        HoverAction::ShowText => Some(ChatHoverContent::Text(
            value
                .formatted_text(lang)
                .split('\n')
                .map(str::to_owned)
                .collect(),
        )),
        HoverAction::ShowItem => Some(resolve_item_hover(
            &value.unformatted_text(lang),
            advanced_tooltips,
            lang,
        )),
        HoverAction::ShowEntity if advanced_tooltips => {
            Some(resolve_entity_hover(&value.unformatted_text(lang)))
        }
        HoverAction::ShowEntity => None,
        HoverAction::ShowAchievement => Some(resolve_stat_hover(
            &value.unformatted_text(lang),
            inventory_key_name,
            font,
            lang,
        )),
    }
}

/// Java: `GuiScreen.handleComponentHover`'s `SHOW_ITEM` branch.  The chat-component value is
/// Mojangson, not JSON: malformed input and a compound that cannot make an `ItemStack` both take
/// the same red fallback rather than silently producing no hover tooltip.
fn resolve_item_hover(value: &str, advanced_tooltips: bool, lang: &Language) -> ChatHoverContent {
    let Some(stack) = parse_snbt_compound(value)
        .ok()
        .and_then(item_stack_from_nbt)
    else {
        return invalid_hover("Item");
    };
    let lines =
        crate::screens::container::tooltip::build_localized(&stack, advanced_tooltips, lang);
    ChatHoverContent::Text(
        crate::screens::container::tooltip_box::recolour_tooltip_lines(
            lines,
            crate::screens::container::tooltip::rarity(&stack),
        ),
    )
}

/// Java: `ItemStack.loadItemStackFromNBT`.  String `id`s go through the item registry while a
/// numeric tag follows `NBTTagCompound.getShort`; either missing/unknown item makes Java return
/// `null`, which the caller renders as `Invalid Item!`.
fn item_stack_from_nbt(nbt: NbtCompound) -> Option<ItemStack> {
    let item_id = match nbt.get("id") {
        Some(Nbt::String(name)) => item_by_name(name),
        _ => {
            let id = nbt.get_i16("id")?;
            (id > 0)
                .then_some(ItemId(id as u16))
                .filter(|id| item_props(*id).registered)
        }
    }?;

    Some(ItemStack {
        item_id: item_id.0 as i16,
        // `NBTTagCompound.getByte("Count")` is a signed byte, widened by the ItemStack
        // constructor into Java's `int stackSize`.
        count: i32::from(nbt.get_i8("Count").unwrap_or(0)),
        // `readFromNBT` routes through `setItemDamage`, which clamps negative metadata before
        // `GuiScreen.handleComponentHover` asks the stack for its tooltip.
        damage: nbt.get_i16("Damage").unwrap_or(0).max(0),
        nbt: nbt.get_compound("tag").cloned(),
    })
}

/// Java: `GuiScreen.handleComponentHover`'s advanced-only `SHOW_ENTITY` branch.  A valid
/// compound is intentionally not registry-validated: vanilla simply reads the three display
/// fields from it after parsing.  The type line also carries `EntityList.getIDFromString`, whose
/// intentionally surprising unknown-name fallback is `90` (the Pig id) in 1.8.9.
fn resolve_entity_hover(value: &str) -> ChatHoverContent {
    let Ok(nbt) = parse_snbt_compound(value) else {
        return invalid_hover("Entity");
    };
    let mut lines = vec![nbt.get_str("name").unwrap_or_default().to_owned()];
    if let Some(entity_type) = nbt.get_str("type") {
        let entity_id =
            mc_data::entity_by_name(entity_type).map_or(90, |entity| i32::from(entity.net_id()));
        lines.push(format!("Type: {entity_type} ({entity_id})"));
    }
    lines.push(nbt.get_str("id").unwrap_or_default().to_owned());
    ChatHoverContent::Text(lines)
}

fn invalid_hover(kind: &str) -> ChatHoverContent {
    ChatHoverContent::Text(vec![format!("\u{a7}cInvalid {kind}!")])
}

fn resolve_stat_hover(
    id: &str,
    inventory_key_name: &str,
    font: &FontRenderer,
    lang: &Language,
) -> ChatHoverContent {
    let Some(stat) = mc_data::stat_by_id(id) else {
        return ChatHoverContent::Text(vec!["\u{a7}cInvalid statistic/achievement!".to_owned()]);
    };

    let achievement = mc_data::achievement_by_stat_id(id);
    let name_color = match achievement {
        Some(achievement) if achievement.is_special() => '5',
        Some(_) => 'a',
        None => '7',
    };
    let mut lines = vec![
        format!("\u{a7}{name_color}{}\u{a7}r", lang.tr(stat.translation_key)),
        format!(
            "\u{a7}o{}\u{a7}r",
            lang.tr(if achievement.is_some() {
                "stats.tooltip.type.achievement"
            } else {
                "stats.tooltip.type.statistic"
            })
        ),
    ];

    if let Some(achievement) = achievement {
        let info = achievement.info();
        let description_key = format!("{}.desc", info.stat_id);
        let description = if info.stat_id == "achievement.openInventory" {
            lang.format(&description_key, &[inventory_key_name])
        } else {
            lang.tr(&description_key).to_owned()
        };
        lines.extend(font.wrap_to_lines(&description, 150));
    }

    ChatHoverContent::Achievement(lines)
}

/// The longest common prefix of a set of strings, case-sensitively.
// java: org.apache.commons.lang3.StringUtils.getCommonPrefix
#[must_use]
pub fn common_prefix(items: &[String]) -> String {
    String::from_utf16_lossy(&common_prefix_units(items))
}

fn common_prefix_units(items: &[String]) -> Vec<u16> {
    let Some(first) = items.first() else {
        return Vec::new();
    };
    let first_units = first.encode_utf16().collect::<Vec<_>>();
    let mut len = first_units.len();
    for other in &items[1..] {
        let other_units = other.encode_utf16().collect::<Vec<_>>();
        let shared = first_units
            .iter()
            .zip(other_units.iter())
            .take_while(|(a, b)| a == b)
            .count();
        len = len.min(shared);
    }
    first_units[..len].to_vec()
}

/// Java `String.equalsIgnoreCase` over UTF-16 units.
///
/// `String#regionMatches(true, ...)` first compares each `char` directly, then compares the
/// single-`char` upper-case forms, and finally the lower-case forms of those upper-case values.
/// Java's `char` overload never expands one unit into several, so mappings such as `ß -> SS`
/// retain the original unit here as well.
fn java_utf16_equals_ignore_case(left: &[u16], right: &[u16]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(&left, &right)| {
            if left == right {
                return true;
            }
            let left_upper = simple_utf16_upper(left);
            let right_upper = simple_utf16_upper(right);
            left_upper == right_upper
                || simple_utf16_lower(left_upper) == simple_utf16_lower(right_upper)
        })
}

fn simple_utf16_upper(unit: u16) -> u16 {
    let Some(character) = char::from_u32(u32::from(unit)) else {
        return unit;
    };
    let mut mapped = character.to_uppercase();
    let Some(first) = mapped.next() else {
        return unit;
    };
    if mapped.next().is_some() || first.len_utf16() != 1 {
        return unit;
    }
    first as u16
}

fn simple_utf16_lower(unit: u16) -> u16 {
    let Some(character) = char::from_u32(u32::from(unit)) else {
        return unit;
    };
    let mut mapped = character.to_lowercase();
    let Some(first) = mapped.next() else {
        return unit;
    };
    if mapped.next().is_some() || first.len_utf16() != 1 {
        return unit;
    }
    first as u16
}

/// Take a prefix using Java's UTF-16 cursor units while returning valid Rust text.
///
/// `TextField` deliberately preserves raw UTF-16, so a caret can be positioned between the two
/// units of an astral character. Rust strings cannot contain that half-surrogate directly; the
/// lossy conversion below gives it a stable replacement scalar while still consuming exactly
/// the requested number of Java units. This keeps chat completion's prefix/suffix boundary safe
/// without changing the widget's raw editing semantics.
fn utf16_prefix(text: &str, units: i32) -> String {
    let requested = units.max(0) as usize;
    if requested == 0 {
        return String::new();
    }

    let mut consumed = 0;
    let mut output = String::new();
    for character in text.chars() {
        let encoded = character.len_utf16();
        if consumed + encoded <= requested {
            output.push(character);
            consumed += encoded;
            continue;
        }

        // The cursor cuts this scalar in half. Preserve only the requested UTF-16 units and
        // convert the resulting lone surrogate safely instead of slicing a Rust string.
        let mut buffer = [0; 2];
        let partial = character
            .encode_utf16(&mut buffer)
            .iter()
            .copied()
            .take(requested - consumed)
            .collect::<Vec<_>>();
        output.push_str(&String::from_utf16_lossy(&partial));
        break;
    }
    output
}

/// Take a UTF-16-unit range from a Rust string without slicing at a byte offset.
///
/// The bounds are Java `String` indices. A range may end in the middle of an astral scalar;
/// `from_utf16_lossy` preserves the unit count while exposing that unavoidable lone surrogate as
/// U+FFFD to Rust callers.
fn utf16_range(text: &str, start: i32, end: i32) -> String {
    let start = start.max(0) as usize;
    let count = end.saturating_sub(start as i32).max(0) as usize;
    let units = text
        .encode_utf16()
        .skip(start)
        .take(count)
        .collect::<Vec<_>>();
    String::from_utf16_lossy(&units)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::component::{ChatStyle, ClickAction, ClickEvent, HoverAction, HoverEvent};
    use crate::font::TextFormat;
    use crate::screens::clipboard::NullClipboard;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn lang() -> Language {
        Language::new("en_US")
    }

    fn log_with(n: usize, font: &FontRenderer) -> (ChatLog, ChatLayout) {
        let layout = ChatLayout::default();
        let mut log = ChatLog::new();
        for i in 0..n {
            log.add_message(
                ChatComponent::text(format!("line {i}")),
                0,
                i as i32,
                &layout,
                font,
                &lang(),
                false,
            );
        }
        (log, layout)
    }

    #[test]
    fn clicking_the_chat_input_places_the_caret_at_the_clicked_glyph() {
        let font = uniform_font();
        let mut screen = ChatScreen::new("abcdef", 0, &font);
        screen.init(&ctx());
        assert_eq!(
            screen.input.cursor_position(),
            6,
            "new fields start at the end"
        );

        // The synthetic font advances five pixels per ASCII glyph. Eleven pixels into the
        // background-free field therefore lands after two glyphs.
        assert_eq!(
            screen.mouse_clicked(4 + 11, 480 - 11, 0, &font),
            Action::None
        );
        assert_eq!(screen.input.cursor_position(), 2);

        // GuiTextField only responds to button zero; this guards the button predicate too.
        screen.mouse_clicked(4 + 26, 480 - 11, 1, &font);
        assert_eq!(screen.input.cursor_position(), 2);
    }

    #[test]
    fn chat_click_uses_each_java_pixel_boundary_and_preserves_selection_outside() {
        let font = uniform_font();
        let mut screen = ChatScreen::new("abcdef", 0, &font);
        screen.init(&ctx());
        let rect = screen.input.rect;

        // GuiTextField.trimStringToWidth chooses the longest prefix that fits, so the exact
        // integer boundary belongs to the character on its right and the pixel before it stays
        // on the left. The synthetic font is five pixels per ASCII glyph.
        for (offset, expected) in [(0, 0), (4, 0), (5, 1), (9, 1), (10, 2), (15, 3)] {
            screen.mouse_clicked(rect.x + offset, rect.y + 1, 0, &font);
            assert_eq!(
                screen.input.cursor_position(),
                expected,
                "x offset {offset} must land on Java boundary {expected}"
            );
        }

        // Ctrl+A leaves an active selection. Re-running init at the same size is what the live
        // renderer does every frame; it must not collapse that selection before an outside click.
        screen.input.set_cursor_position(6, &font);
        screen.input.set_selection_pos(0, &font);
        assert_eq!(screen.input.selected_text(), "abcdef");
        screen.init(&ctx());
        screen.mouse_clicked(rect.x, rect.y - 1, 0, &font);
        assert_eq!(
            screen.input.cursor_position(),
            6,
            "outside input does not move caret"
        );
        assert_eq!(
            screen.input.selection_end(),
            0,
            "outside input preserves selection"
        );

        // Java's button predicate is exact: a right click inside the field neither moves the
        // caret nor clears the selection.
        screen.mouse_clicked(rect.x + 2, rect.y + 1, 1, &font);
        assert_eq!(screen.input.cursor_position(), 6);
        assert_eq!(screen.input.selection_end(), 0);

        // A left click inside replaces the selection with the nearest measured boundary.
        screen.mouse_clicked(rect.x + 10, rect.y + 1, 0, &font);
        assert_eq!(screen.input.cursor_position(), 2);
        assert_eq!(screen.input.selection_end(), 2);
        assert!(screen.input.selected_text().is_empty());
    }

    #[test]
    fn chat_click_uses_the_scaled_screen_input_rect() {
        let font = uniform_font();
        let resolution = ScaledResolution::new(1280, 601, 2, false);
        let scaled = ScreenCtx::new(resolution, 0.0);
        let mut screen = ChatScreen::new("abcdef", 0, &font);
        screen.init(&scaled);

        // The physical 1280x601 window becomes a ceil-rounded 640x301 GUI. The input starts at
        // y=301-12 and retains the same unscaled five-pixel glyph advances as Java.
        assert_eq!(screen.input.rect, Rect::new(4, 289, 636, 12));
        screen.mouse_clicked(4 + 5, 289 + 1, 0, &font);
        assert_eq!(screen.input.cursor_position(), 1);
        assert_eq!(screen.input.selection_end(), 1);
    }

    /// Hand-derived from `scale * (320 - 40) + 40` and `scale * (180 - 20) + 20`.
    #[test]
    fn the_chat_box_dimensions_match_vanillas_formulas() {
        assert_eq!(chatbox_width(1.0), 320);
        assert_eq!(chatbox_width(0.0), 40);
        assert_eq!(chatbox_width(0.5), 180);
        assert_eq!(chatbox_height(1.0), 180);
        assert_eq!(chatbox_height(0.0), 20);
        // The default unfocused height, 0.44366196, gives 90 px = 10 lines.
        assert_eq!(chatbox_height(0.443_661_96), 90);
        let l = ChatLayout::default();
        assert_eq!(l.line_count(true), 20);
        assert_eq!(l.line_count(false), 10);
    }

    #[test]
    fn messages_are_stored_newest_first_and_capped_at_a_hundred() {
        let font = uniform_font();
        let (log, _) = log_with(120, &font);
        assert_eq!(log.received_count(), MAX_LINES);
        assert_eq!(log.lines().len(), MAX_LINES);
        assert_eq!(log.lines()[0].text, "§rline 119§r", "newest first");
    }

    // ---- The history length -----------------------------

    /// A busy minigame server's mix: rank chat, coin lines, long lines that wrap, blank spacers,
    /// kills.
    fn minigame_line(i: usize) -> ChatComponent {
        match i % 5 {
            0 => ChatComponent::text(format!(
                "\u{a7}7[\u{a7}a{i}\u{a7}7] \u{a7}bPlayer{i}\u{a7}f: gg wp"
            )),
            1 => ChatComponent::text(format!("\u{a7}6+{i} coins! (Win)")),
            2 => ChatComponent::text("word ".repeat(40 + i % 50)),
            3 => ChatComponent::text(""),
            _ => ChatComponent::text(format!("\u{a7}e{i} was killed by Bot{i}.")),
        }
    }

    #[test]
    fn the_history_length_caps_both_lists_and_a_smaller_one_drops_the_oldest() {
        let font = uniform_font();
        let layout = ChatLayout::default();
        let mut log = ChatLog::new();
        assert_eq!(
            log.max_lines(),
            MAX_LINES,
            "vanilla until the client sets it"
        );
        log.set_max_lines(1000);
        for i in 0..1500 {
            log.add_message(
                ChatComponent::text(format!("line {i}")),
                0,
                i,
                &layout,
                &font,
                &lang(),
                false,
            );
        }
        assert_eq!((log.received_count(), log.lines().len()), (1000, 1000));
        assert_eq!(log.lines()[0].text, "§rline 1499§r", "newest first");
        assert_eq!(log.lines()[999].text, "§rline 500§r");
        // Scrolled all the way up, then shortened to vanilla's 100: the oldest lines go and the
        // view starts again at the bottom.
        log.scroll_by(5000, &layout, true);
        assert_eq!(log.scroll_pos(), 1000 - layout.line_count(true) as usize);
        log.set_max_lines(MAX_LINES);
        assert_eq!((log.received_count(), log.lines().len()), (100, 100));
        assert_eq!(log.lines()[99].text, "§rline 1400§r");
        assert_eq!(log.scroll_pos(), 0);
        assert_eq!(log.visible_lines(&layout, true)[0].text, "§rline 1499§r");
        // Longer again: nothing comes back, nothing is cut.
        log.set_max_lines(5000);
        assert_eq!(log.lines().len(), 100);
    }

    /// The re-wrap builds exactly the list vanilla's `refreshChat` builds (oldest first, each line
    /// inserted at index 0, trimmed to the cap): the same lines, ids and ticks in the same order,
    /// including when the cap cuts the oldest message in two, and the stack keeps the newest
    /// message's line count.
    #[test]
    fn a_rewrap_builds_vanillas_list_from_the_newest_end() {
        let font = uniform_font();
        let lang = lang();
        for cap in [MAX_LINES, 1000] {
            let mut log = ChatLog::new();
            log.set_max_lines(cap);
            let wide = ChatLayout::default();
            for i in 0..cap + 300 {
                log.add_message(minigame_line(i), 0, i as i32, &wide, &font, &lang, false);
            }
            let narrow = ChatLayout {
                width_scale: 0.4,
                ..ChatLayout::default()
            };
            log.refresh(&narrow, &font, &lang);
            let mut expected: Vec<ChatLine> = Vec::new();
            for (tick, component, id) in log.received.iter().rev() {
                for spans in
                    split_component(component, narrow.wrap_width(), &font, &lang, narrow.colors)
                {
                    let text = formatted_line_text(&spans);
                    expected.insert(
                        0,
                        ChatLine {
                            updated_at: *tick,
                            text,
                            id: *id,
                            spans,
                        },
                    );
                }
                expected.truncate(cap);
            }
            assert_eq!(log.lines().len(), cap, "full");
            assert!(log.lines().iter().eq(expected.iter()), "cap {cap}");
            let newest = &log.received[0].1;
            let newest_lines =
                split_component(newest, narrow.wrap_width(), &font, &lang, narrow.colors).len();
            assert_eq!(log.stack.lines, newest_lines);
        }
    }

    /// Only the visible rows are drawn: with chat closed, a 5000-line history draws exactly what
    /// a log holding just its newest lines draws; open, only the scrollbar's place differs.
    #[test]
    fn a_long_history_draws_what_a_short_one_does() {
        let mut font = uniform_font();
        let layout = ChatLayout::default();
        let mut long = ChatLog::new();
        long.set_max_lines(5000);
        let mut short = ChatLog::new();
        for i in 0..6000 {
            long.add_message(minigame_line(i), 0, 5990, &layout, &font, &lang(), false);
        }
        for i in 5900..6000 {
            short.add_message(minigame_line(i), 0, 5990, &layout, &font, &lang(), false);
        }
        assert_eq!(long.lines().len(), 5000);
        let draw = |log: &ChatLog, open: bool, font: &mut FontRenderer| {
            let mut d = DrawList::new();
            log.draw(&mut d, font, &ctx(), 6000, &layout, open);
            d
        };
        assert_eq!(
            draw(&long, false, &mut font),
            draw(&short, false, &mut font)
        );
        assert_eq!(
            draw(&long, true, &mut font).vertices().len(),
            draw(&short, true, &mut font).vertices().len()
        );
    }

    /// Adding lines and re-wrapping stay cheap at the 5000-line maximum (timings provisional:
    /// the test profile is unoptimised). The old
    /// front-inserting list is timed beside it for comparison.
    #[test]
    fn a_full_five_thousand_line_history_adds_and_rewraps_fast() {
        use std::time::Instant;
        let font = uniform_font();
        let lang = lang();
        let layout = ChatLayout::default();
        let mut log = ChatLog::new();
        log.set_max_lines(5000);
        for i in 0..5000 {
            log.add_message(minigame_line(i), 0, 0, &layout, &font, &lang, false);
        }
        let burst = Instant::now();
        for i in 0..100 {
            log.add_message(minigame_line(i), 0, 0, &layout, &font, &lang, false);
        }
        let per_message = burst.elapsed().as_secs_f64() * 1000.0 / 100.0;
        let narrow = ChatLayout {
            width_scale: 0.6,
            ..ChatLayout::default()
        };
        let rewrap = Instant::now();
        log.refresh(&narrow, &font, &lang);
        let rewrap_ms = rewrap.elapsed().as_secs_f64() * 1000.0;
        // The old shape: a Vec with every line inserted at index 0.
        let old = Instant::now();
        let mut drawn: Vec<ChatLine> = Vec::new();
        for (tick, component, id) in log.received.iter().rev() {
            for spans in split_component(component, narrow.wrap_width(), &font, &lang, true) {
                let text = formatted_line_text(&spans);
                drawn.insert(
                    0,
                    ChatLine {
                        updated_at: *tick,
                        text,
                        id: *id,
                        spans,
                    },
                );
            }
            drawn.truncate(5000);
        }
        let old_ms = old.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "B1 at 5000 lines (test profile, provisional): {per_message:.3} ms per message, \
             re-wrap {rewrap_ms:.1} ms (old front-insert re-wrap {old_ms:.1} ms)"
        );
        assert_eq!(log.lines().len(), 5000);
        assert!(per_message < 5.0, "adding a message: {per_message:.3} ms");
        assert!(rewrap_ms < 1000.0, "re-wrap: {rewrap_ms:.1} ms");
    }

    #[test]
    fn a_long_message_becomes_several_display_lines_but_one_received_line() {
        let font = uniform_font();
        let layout = ChatLayout::default();
        let mut log = ChatLog::new();
        // The synthetic font is 5 px per glyph, so 320/5 = 64 glyphs per line.
        log.add_message(
            ChatComponent::text("word ".repeat(60)),
            0,
            0,
            &layout,
            &font,
            &lang(),
            false,
        );
        assert_eq!(log.received_count(), 1);
        assert!(log.lines().len() > 1, "it wrapped");
    }

    #[test]
    fn refreshing_rewraps_from_the_received_list_not_the_drawn_one() {
        let font = uniform_font();
        let mut layout = ChatLayout::default();
        let mut log = ChatLog::new();
        let text = "word ".repeat(60);
        log.add_message(
            ChatComponent::text(&text),
            0,
            0,
            &layout,
            &font,
            &lang(),
            false,
        );
        let wide = log.lines().len();

        layout.width_scale = 0.25; // a much narrower box
        log.refresh(&layout, &font, &lang());
        let narrow = log.lines().len();
        assert!(
            narrow > wide,
            "narrower box, more lines: {narrow} vs {wide}"
        );

        // And back again — this is the lossless half. Re-wrapping already-wrapped lines would
        // leave the extra breaks in place and never recover the original count.
        layout.width_scale = 1.0;
        log.refresh(&layout, &font, &lang());
        assert_eq!(log.lines().len(), wide);
        assert_eq!(log.received_count(), 1);
    }

    #[test]
    fn a_non_zero_id_replaces_the_previous_line_with_that_id() {
        let font = uniform_font();
        let layout = ChatLayout::default();
        let mut log = ChatLog::new();
        log.add_message(
            ChatComponent::text("first"),
            1,
            0,
            &layout,
            &font,
            &lang(),
            false,
        );
        log.add_message(
            ChatComponent::text("second"),
            1,
            1,
            &layout,
            &font,
            &lang(),
            false,
        );
        assert_eq!(log.lines().len(), 1, "the old id-1 line was removed");
        assert_eq!(log.lines()[0].text, "§rsecond§r");
        // An id of 0 never replaces anything.
        log.add_message(
            ChatComponent::text("a"),
            0,
            2,
            &layout,
            &font,
            &lang(),
            false,
        );
        log.add_message(
            ChatComponent::text("b"),
            0,
            3,
            &layout,
            &font,
            &lang(),
            false,
        );
        assert_eq!(log.lines().len(), 3);
    }

    #[test]
    fn scrolling_clamps_at_both_ends() {
        let font = uniform_font();
        let (mut log, layout) = log_with(30, &font);
        assert_eq!(log.scroll_pos(), 0);
        log.scroll_by(-5, &layout, false);
        assert_eq!(log.scroll_pos(), 0, "cannot scroll below the newest line");
        log.scroll_by(5, &layout, false);
        assert_eq!(log.scroll_pos(), 5);
        log.scroll_by(1000, &layout, false);
        // 30 lines, 10 visible → the oldest reachable scroll is 20.
        assert_eq!(log.scroll_pos(), 20);
        assert!(
            !log.is_scrolled(),
            "the flag is only set by an incoming message"
        );
        log.reset_scroll();
        assert_eq!(log.scroll_pos(), 0);
    }

    #[test]
    fn a_message_arriving_while_scrolled_keeps_the_view_on_the_same_line() {
        let font = uniform_font();
        let (mut log, layout) = log_with(30, &font);
        log.scroll_by(5, &layout, true);
        let anchored = log.visible_lines(&layout, true)[0].text.clone();
        log.add_message(
            ChatComponent::text("new"),
            0,
            99,
            &layout,
            &font,
            &lang(),
            true,
        );
        assert_eq!(log.scroll_pos(), 6, "the scroll advanced with the insert");
        assert_eq!(log.visible_lines(&layout, true)[0].text, anchored);
        assert!(log.is_scrolled());
    }

    /// Vanilla's fade: opaque for 20 ticks, quadratic decay to nothing at 200.
    #[test]
    fn line_alpha_follows_vanillas_quadratic_fade() {
        let font = uniform_font();
        let (log, layout) = log_with(1, &font);
        let line = &log.lines()[0];
        // Age 0: d = 10 → clamped to 1 → 255.
        assert_eq!(log.line_alpha(line, 0, &layout, false), Some(255));
        // Age 18: (1 - 0.09) * 10 = 9.1 → clamped to 1 → still 255.
        assert_eq!(log.line_alpha(line, 18, &layout, false), Some(255));
        // Age 100: d = (1 - 0.5) * 10 = 5, clamped to 1 → still 255. The curve is flat for
        // the first 180 ticks; only the last tenth of the window fades.
        assert_eq!(log.line_alpha(line, 100, &layout, false), Some(255));
        // 180 ticks: `1.0 - 180.0/200.0` is 0.09999999999999998 in f64, so the clamped
        // factor is a hair under 1 and `(int)(255 * d * d)` truncates to 254 — in Java too.
        assert_eq!(log.line_alpha(line, 180, &layout, false), Some(254));
        // Age 190: d = 0.5, squared = 0.25 → 63.
        assert_eq!(log.line_alpha(line, 190, &layout, false), Some(63));
        // Age 199: d = 0.05, squared = 0.0025 → 0, which is below the >3 threshold.
        assert_eq!(log.line_alpha(line, 199, &layout, false), None);
        assert_eq!(log.line_alpha(line, 500, &layout, false), None, "faded out");
        // With the chat open every line is fully opaque however old.
        assert_eq!(log.line_alpha(line, 5000, &layout, true), Some(255));
        // Opacity scales it: 0 gives 10% of 255.
        let dim = ChatLayout {
            opacity: 0.0,
            ..layout
        };
        assert_eq!(log.line_alpha(line, 0, &dim, true), Some(25));
    }

    #[test]
    fn hidden_chat_draws_nothing_at_all() {
        let mut font = uniform_font();
        let (log, mut layout) = log_with(5, &font);
        layout.visibility = 2;
        let mut d = DrawList::new();
        log.draw(&mut d, &mut font, &ctx(), 0, &layout, false);
        assert!(d.is_empty());
        layout.visibility = 0;
        log.draw(&mut d, &mut font, &ctx(), 0, &layout, false);
        assert!(!d.is_empty());
    }

    #[test]
    fn the_bottom_row_sits_at_height_minus_twenty_eight() {
        let mut font = uniform_font();
        let (log, layout) = log_with(3, &font);
        let mut d = DrawList::new();
        let c = ctx();
        log.draw(&mut d, &mut font, &c, 0, &layout, false);
        // Only the backdrops, not the glyph quads: a glyph's box overhangs its row by 8 px.
        let ys: Vec<f32> = d
            .batches()
            .iter()
            .filter(|b| b.texture == crate::widget::UiTexture::Solid)
            .flat_map(|b| d.vertices()[b.range.clone()].iter())
            .map(|v| v.pos[1])
            .collect();
        let max = ys.iter().cloned().fold(f32::MIN, f32::max);
        assert_eq!(max, (c.height - 28) as f32);
        // Three rows step up by 9 each, so the top backdrop starts 27 px higher.
        let min = ys.iter().cloned().fold(f32::MAX, f32::min);
        assert_eq!(min, (c.height - 28 - 27) as f32);
    }

    #[test]
    fn chat_scale_shrinks_the_backdrop_and_row_pitch_about_vanillas_anchor() {
        let mut font = uniform_font();
        let (log, mut layout) = log_with(3, &font);
        layout.scale = 0.5;
        let mut d = DrawList::new();
        let c = ctx();
        log.draw(&mut d, &mut font, &c, 0, &layout, false);
        let points = d
            .batches()
            .iter()
            .filter(|batch| batch.texture == crate::widget::UiTexture::Solid)
            .flat_map(|batch| d.vertices()[batch.range.clone()].iter())
            .map(|vertex| vertex.pos)
            .collect::<Vec<_>>();
        let min_x = points.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        let max_x = points.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        let min_y = points.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let max_y = points.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        assert_eq!((min_x, max_x), (2.0, 324.0));
        assert_eq!(max_y, (c.height - 28) as f32, "the anchor does not scale");
        assert_eq!(min_y, (c.height - 28) as f32 - 27.0 * 0.5);
    }

    #[test]
    fn chat_scale_shrinks_glyph_geometry_about_the_same_anchor() {
        fn font_bounds(d: &DrawList) -> (f32, f32, f32, f32) {
            let points = d
                .batches()
                .iter()
                .filter(|batch| matches!(batch.texture, crate::widget::UiTexture::Font(_)))
                .flat_map(|batch| d.vertices()[batch.range.clone()].iter())
                .map(|vertex| vertex.pos)
                .collect::<Vec<_>>();
            assert!(!points.is_empty());
            (
                points.iter().map(|p| p[0]).fold(f32::MAX, f32::min),
                points.iter().map(|p| p[0]).fold(f32::MIN, f32::max),
                points.iter().map(|p| p[1]).fold(f32::MAX, f32::min),
                points.iter().map(|p| p[1]).fold(f32::MIN, f32::max),
            )
        }

        let mut font = uniform_font();
        let (log, mut layout) = log_with(1, &font);
        let c = ctx();

        let mut full = DrawList::new();
        log.draw(&mut full, &mut font, &c, 0, &layout, false);
        let full_bounds = font_bounds(&full);

        layout.scale = 0.5;
        let mut half = DrawList::new();
        log.draw(&mut half, &mut font, &c, 0, &layout, false);
        let half_bounds = font_bounds(&half);

        let anchor = (c.height - 28) as f32;
        assert_eq!(half_bounds.0, full_bounds.0, "the x anchor is not scaled");
        assert!(
            ((half_bounds.1 - half_bounds.0) * 2.0 - (full_bounds.1 - full_bounds.0)).abs()
                < 1.0e-4
        );
        assert!(((half_bounds.2 - anchor) * 2.0 - (full_bounds.2 - anchor)).abs() < 1.0e-4);
        assert!(((half_bounds.3 - anchor) * 2.0 - (full_bounds.3 - anchor)).abs() < 1.0e-4);
    }

    #[test]
    fn open_chat_draws_vanillas_two_part_scrollbar_and_scrolled_colour() {
        let mut font = uniform_font();
        let (mut log, layout) = log_with(30, &font);
        log.scroll = 5;
        log.scrolled = true;
        let mut d = DrawList::new();
        log.draw(&mut d, &mut font, &ctx(), 0, &layout, true);

        let mut bars = Vec::new();
        for batch in d
            .batches()
            .iter()
            .filter(|batch| batch.texture == crate::widget::UiTexture::Solid)
        {
            for quad in d.vertices()[batch.range.clone()].chunks_exact(6) {
                let min_x = quad.iter().map(|v| v.pos[0]).fold(f32::MAX, f32::min);
                let max_x = quad.iter().map(|v| v.pos[0]).fold(f32::MIN, f32::max);
                if max_x <= 1.0 {
                    bars.push((min_x, max_x, quad));
                }
            }
        }
        assert_eq!(bars.len(), 2, "the fill and its one-pixel highlight");
        assert_eq!((bars[0].0, bars[0].1), (-1.0, 1.0));
        assert_eq!((bars[1].0, bars[1].1), (0.0, 1.0));
        let min_y = bars
            .iter()
            .flat_map(|(_, _, quad)| quad.iter().map(|v| v.pos[1]))
            .fold(f32::MAX, f32::min);
        let max_y = bars
            .iter()
            .flat_map(|(_, _, quad)| quad.iter().map(|v| v.pos[1]))
            .fold(f32::MIN, f32::max);
        // k=30, j=20, pitch=9: j3=5*200/30=33 and k1=200*200/300=133.
        assert_eq!((min_y, max_y), (286.0, 419.0));
        let fill = bars[0].2[0].color;
        assert!(fill[0] > fill[2], "isScrolled uses vanilla's red fill");
        assert!((fill[3] - 170.0 / 255.0).abs() < 1.0e-6);
    }

    #[test]
    fn open_chat_scrollbar_uses_idle_alpha_when_at_the_bottom() {
        let mut font = uniform_font();
        let (log, layout) = log_with(30, &font);
        let mut d = DrawList::new();
        log.draw(&mut d, &mut font, &ctx(), 0, &layout, true);

        let bars = d
            .batches()
            .iter()
            .filter(|batch| batch.texture == crate::widget::UiTexture::Solid)
            .flat_map(|batch| d.vertices()[batch.range.clone()].chunks_exact(6))
            .filter(|quad| quad.iter().map(|v| v.pos[0]).fold(f32::MIN, f32::max) <= 1.0)
            .collect::<Vec<_>>();
        assert_eq!(
            bars.len(),
            2,
            "the fill and one-pixel highlight remain distinct"
        );
        assert!((bars[0][0].color[3] - 96.0 / 255.0).abs() < 1.0e-6);
        assert!(
            bars[0][0].color[0] < bars[0][0].color[2],
            "the unscrolled fill uses vanilla's grey colour"
        );
    }

    #[test]
    fn chat_scrollbar_is_hidden_when_closed_or_when_every_line_fits() {
        fn scrollbar_quads(d: &DrawList) -> usize {
            d.batches()
                .iter()
                .filter(|batch| batch.texture == crate::widget::UiTexture::Solid)
                .flat_map(|batch| d.vertices()[batch.range.clone()].chunks_exact(6))
                .filter(|quad| quad.iter().map(|v| v.pos[0]).fold(f32::MIN, f32::max) <= 1.0)
                .count()
        }

        let mut font = uniform_font();
        let fitting = ChatLayout::default().line_count(true) as usize;
        let (log, layout) = log_with(fitting, &font);
        let mut d = DrawList::new();
        log.draw(&mut d, &mut font, &ctx(), 0, &layout, true);
        assert_eq!(
            scrollbar_quads(&d),
            0,
            "no bar when the visible window fits"
        );

        let (log, layout) = log_with(30, &font);
        let mut d = DrawList::new();
        log.draw(&mut d, &mut font, &ctx(), 0, &layout, false);
        assert_eq!(
            scrollbar_quads(&d),
            0,
            "the bar is only drawn for open chat"
        );
    }

    #[test]
    fn chat_layout_carries_the_chat_colours_setting() {
        let mut settings = mc_render::settings::GameSettings {
            chat_colors: false,
            ..Default::default()
        };
        let layout = ChatLayout::from_settings(&settings);
        assert!(!layout.colors);

        settings.chat_colors = true;
        assert!(ChatLayout::from_settings(&settings).colors);
    }

    #[test]
    fn sent_history_skips_immediate_duplicates() {
        let mut log = ChatLog::new();
        log.add_sent("hello");
        log.add_sent("hello");
        log.add_sent("world");
        log.add_sent("hello");
        assert_eq!(log.sent_messages(), ["hello", "world", "hello"]);
    }

    #[test]
    fn up_and_down_walk_the_history_and_restore_the_live_buffer() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        log.add_sent("one");
        log.add_sent("two");
        let mut s = ChatScreen::new("", log.sent_messages().len(), &font);
        s.input.set_text("draft", &font);

        s.recall(-1, &log, &font);
        assert_eq!(s.input.text(), "two");
        s.recall(-1, &log, &font);
        assert_eq!(s.input.text(), "one");
        s.recall(-1, &log, &font);
        assert_eq!(s.input.text(), "one", "clamped at the oldest");
        s.recall(1, &log, &font);
        assert_eq!(s.input.text(), "two");
        s.recall(1, &log, &font);
        assert_eq!(s.input.text(), "draft", "the live buffer comes back");
    }

    #[test]
    fn enter_sends_the_trimmed_message_and_records_it() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut s = ChatScreen::new("", 0, &font);
        s.input.set_text("  hi there  ", &font);
        let mut clip = NullClipboard;
        let action = s.key_typed(
            None,
            keys::ENTER,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(action, Action::SendChat("hi there".to_owned()));
        assert_eq!(log.sent_messages(), ["hi there"]);

        // An empty box just closes.
        s.input.set_text("   ", &font);
        let action = s.key_typed(
            None,
            keys::ENTER,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(action, Action::Back);
    }

    /// Enter trims like Java's `String.trim()` (chars at or
    /// below U+0020 only), so a trailing no-break or ideographic space is sent as vanilla sends it.
    #[test]
    fn enter_trims_like_java_and_keeps_unicode_spaces() {
        assert_eq!(java_trim("  hi  "), "hi");
        assert_eq!(java_trim("\u{3000}hi\u{a0}"), "\u{3000}hi\u{a0}");
        assert_eq!(java_trim("   "), "");
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut s = ChatScreen::new("", 0, &font);
        s.input.set_text(" gg\u{3000} ", &font);
        let action = s.key_typed(
            None,
            keys::ENTER,
            Modifiers::NONE,
            &mut NullClipboard,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(action, Action::SendChat("gg\u{3000}".to_owned()));
    }

    /// 256 Character Chat: the input holds exactly `limit` UTF-16
    /// units, vanilla's 100 by default; the cap applies before the pre-fill.
    #[test]
    fn the_chat_input_holds_one_hundred_or_the_long_limit() {
        let font = uniform_font();
        let long = "x".repeat(300);
        let mut vanilla = ChatScreen::new("", 0, &font);
        vanilla.input.write_text(&long, &font);
        assert_eq!(vanilla.input.text().len(), MAX_INPUT_LENGTH as usize);

        let mut wide = ChatScreen::with_input_limit("", 0, &font, LONG_INPUT_LENGTH);
        wide.input.write_text(&long, &font);
        assert_eq!(wide.input.text().len(), 256);

        // A suggested command longer than 100 keeps its tail with the long limit.
        let suggested = format!("/msg {}", "y".repeat(150));
        let wide = ChatScreen::with_input_limit(&suggested, 0, &font, LONG_INPUT_LENGTH);
        assert_eq!(wide.input.text(), suggested);
        let vanilla = ChatScreen::new(&suggested, 0, &font);
        assert_eq!(vanilla.input.text(), suggested[..100]);
    }

    #[test]
    fn tab_asks_the_server_for_completions_for_the_text_before_the_caret() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut s = ChatScreen::new("/tp No", 0, &font);
        let mut clip = NullClipboard;
        let action = s.key_typed(
            None,
            keys::TAB,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(action, Action::RequestTabComplete("/tp No".to_owned()));
        assert_eq!(s.completion_prefix(), "No");
        // An empty box asks for nothing.
        let mut empty = ChatScreen::new("", 0, &font);
        assert_eq!(
            empty.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout
            ),
            Action::None
        );
    }

    #[test]
    fn tab_completion_uses_utf16_units_after_an_emoji_and_preserves_the_suffix() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut screen = ChatScreen::new("😀 No!", 0, &font);

        // The emoji occupies two Java units. The caret is after "😀 No" (five units), before
        // the exclamation mark; a scalar-counting implementation incorrectly includes the '!'.
        screen.input.set_cursor_position(5, &font);
        assert_eq!(screen.completion_prefix(), "No");
        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("😀 No".to_owned())
        );

        screen.on_completions(&["Nobody".to_owned()], &font);
        assert_eq!(screen.input.text(), "😀 Nobody!");
        assert_eq!(screen.input.cursor_position(), 9);
    }

    #[test]
    fn tab_completion_replaces_a_non_bmp_word_in_place_and_preserves_the_suffix() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut screen = ChatScreen::new("say 😀!", 0, &font);

        // `say ` is four UTF-16 units and 😀 is two. The completion is requested from the
        // middle of the message, so deleting one Rust scalar here would leave the old emoji's
        // trailing surrogate in front of the replacement.
        screen.input.set_cursor_position(6, &font);
        assert_eq!(screen.completion_prefix(), "😀");
        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("say 😀".to_owned())
        );

        screen.on_completions(&["😀smile".to_owned()], &font);
        assert_eq!(screen.input.text(), "say 😀smile!");
        assert_eq!(screen.input.cursor_position(), 11);
    }

    #[test]
    fn tab_completion_replaces_an_ascii_word_in_place_and_preserves_the_suffix() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut screen = ChatScreen::new("say he!", 0, &font);

        screen.input.set_cursor_position(6, &font);
        assert_eq!(screen.completion_prefix(), "he");
        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("say he".to_owned())
        );

        screen.on_completions(&["hello".to_owned()], &font);
        assert_eq!(screen.input.text(), "say hello!");
        assert_eq!(screen.input.cursor_position(), 9);
    }

    #[test]
    fn tab_completion_at_a_split_surrogate_is_safe_and_keeps_the_remaining_unit() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut screen = ChatScreen::new("😀", 0, &font);

        // Java permits the caret between an astral character's surrogate halves. Completion
        // must not panic or accidentally consume the trailing half when replacing the prefix.
        screen.input.set_cursor_position(1, &font);
        assert_eq!(screen.completion_prefix(), "�");
        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("�".to_owned())
        );

        screen.on_completions(&["x".to_owned()], &font);
        assert_eq!(screen.input.text(), "x�");
        assert_eq!(screen.input.cursor_position(), 1);
    }

    #[test]
    fn common_prefix_uses_java_utf16_code_units() {
        let split = vec!["😀alpha".to_owned(), "😁beta".to_owned()];
        assert_eq!(common_prefix(&split), "�");

        let whole = vec!["😀alpha".to_owned(), "😀beta".to_owned()];
        assert_eq!(common_prefix(&whole), "😀");
    }

    #[test]
    fn unicode_case_equal_common_prefix_starts_candidate_cycle_like_java() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut screen = ChatScreen::new("say é", 0, &font);

        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("say é".to_owned())
        );
        screen.on_completions(&["Éclair".to_owned(), "Étoile".to_owned()], &font);

        // Java `equalsIgnoreCase` treats the existing `é` as equal to common prefix `É`, so it
        // skips prefix insertion and immediately cycles to the first complete candidate. The old
        // ASCII-only comparison stopped at a visible one-character `É` instead.
        assert_eq!(screen.input.text(), "say Éclair");
        assert!(screen.completion_hint());
    }

    #[test]
    fn split_surrogate_common_prefix_keeps_java_prefix_branch() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut screen = ChatScreen::new("say x!", 0, &font);
        screen.input.set_cursor_position(5, &font);
        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("say x".to_owned())
        );

        screen.on_completions(&["😀alpha".to_owned(), "😁beta".to_owned()], &font);

        // Java inserts the shared high-surrogate unit and does not start candidate cycling. The
        // public Rust string exposes that unavoidable lone surrogate as U+FFFD, while the raw
        // TextField storage retains its one-unit cursor boundary.
        assert_eq!(screen.input.text(), "say �!");
        assert_eq!(screen.input.cursor_position(), 5);
        assert!(!screen.completion_hint());
    }

    #[test]
    fn the_common_prefix_is_inserted_before_any_cycling_starts() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut s = ChatScreen::new("/tp No", 0, &font);
        let mut clip = NullClipboard;
        s.key_typed(
            None,
            keys::TAB,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        s.on_completions(
            &["Notch".to_owned(), "Nobody".to_owned(), "Nomad".to_owned()],
            &font,
        );
        // "No" is *already* the common prefix, so vanilla skips the insert and goes straight
        // to cycling — `!s1.equalsIgnoreCase(s2)` is false.
        assert_eq!(s.input.text(), "/tp Notch");
        assert!(
            s.completion_hint(),
            "cycling a multi-match response prints id 1"
        );

        let mut s = ChatScreen::new("/tp N", 0, &font);
        s.key_typed(
            None,
            keys::TAB,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        s.on_completions(&["Notch".to_owned(), "Nobody".to_owned()], &font);
        assert_eq!(s.input.text(), "/tp No", "extended to the common prefix");
        assert!(
            !s.completion_hint(),
            "inserting the longer common prefix suppresses id 1"
        );
    }

    #[test]
    fn a_second_tab_cycles_through_the_candidates() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut s = ChatScreen::new("/tp No", 0, &font);
        s.key_typed(
            None,
            keys::TAB,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        s.on_completions(&["Notch".to_owned(), "Nobody".to_owned()], &font);
        // "No" is already the common prefix, so vanilla starts cycling immediately.
        assert_eq!(s.input.text(), "/tp Notch");
        s.key_typed(
            None,
            keys::TAB,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(s.input.text(), "/tp Nobody");
        s.key_typed(
            None,
            keys::TAB,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(s.input.text(), "/tp Notch", "it wraps");
    }

    /// java: `GuiChat.autocompletePlayerNames:232-247` prints the id-1
    /// candidate line on every cycling Tab while more than one name was found. The flag is set
    /// by each cycling Tab and cleared by the next key, so the host prints once per press.
    #[test]
    fn every_cycling_tab_flags_the_candidate_hint() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut s = ChatScreen::new("/tp No", 0, &font);
        let mut press = |s: &mut ChatScreen, typed: Option<char>, key: i32| {
            s.key_typed(
                typed,
                key,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            )
        };
        press(&mut s, None, keys::TAB);
        assert!(!s.completion_hint(), "the request Tab prints nothing");
        s.on_completions(&["Notch".to_owned(), "Nobody".to_owned()], &font);
        assert!(s.completion_hint());
        press(&mut s, None, keys::TAB);
        assert_eq!(s.input.text(), "/tp Nobody");
        assert!(s.completion_hint(), "the cycling Tab prints the line again");
        press(&mut s, Some('x'), 45);
        assert!(!s.completion_hint(), "any other key clears it");
    }

    #[test]
    fn a_single_candidate_completes_the_whole_word() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut s = ChatScreen::new("/gamem", 0, &font);
        s.key_typed(
            None,
            keys::TAB,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        s.on_completions(&["/gamemode".to_owned()], &font);
        assert_eq!(s.input.text(), "/gamemode");
        assert!(
            !s.completion_hint(),
            "a single candidate has no candidate list"
        );
    }

    #[test]
    fn no_common_prefix_cycles_and_exposes_the_filtered_hint_candidates() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut screen = ChatScreen::new("/tp X", 0, &font);
        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("/tp X".into())
        );

        screen.on_completions(&["Notch".into(), "Nobody".into(), String::new()], &font);
        assert_eq!(screen.input.text(), "/tp Notch");
        assert_eq!(screen.completions(), &["Notch", "Nobody"]);
        assert!(screen.completion_hint());
    }

    #[test]
    fn unicode_common_prefix_uses_utf16_caret_and_suppresses_the_hint() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut screen = ChatScreen::new("\u{1f600} \u{00e9}!", 0, &font);

        // The caret is after the one-unit `\u{00e9}` and before the suffix. The astral glyph
        // before it still occupies two Java units.
        screen.input.set_cursor_position(4, &font);
        assert_eq!(screen.completion_prefix(), "\u{00e9}");
        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("\u{1f600} \u{00e9}".into())
        );

        screen.on_completions(&["\u{00e9}clair".into(), "\u{00e9}corce".into()], &font);
        assert_eq!(screen.input.text(), "\u{1f600} \u{00e9}c!");
        assert!(!screen.completion_hint());
    }

    #[test]
    fn completions_that_arrive_unrequested_are_ignored() {
        let font = uniform_font();
        let mut s = ChatScreen::new("/x", 0, &font);
        s.on_completions(&["nope".to_owned()], &font);
        assert_eq!(s.input.text(), "/x");
        assert!(s.completions().is_empty());
    }

    #[test]
    fn an_empty_packet_candidate_suppresses_the_common_prefix_but_not_local_cycling() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clip = NullClipboard;
        let mut s = ChatScreen::new("/tp N", 0, &font);
        assert_eq!(
            s.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::RequestTabComplete("/tp N".into())
        );
        s.on_completions(&["Notch".into(), String::new()], &font);
        assert_eq!(s.input.text(), "/tp Notch");

        // The raw-array common prefix is empty, so vanilla enters its local-cycle branch. A
        // filtered-prefix implementation reaches the same visible text once, but sends a second
        // C14 here instead; this assertion discriminates the packet-rate state.
        assert_eq!(
            s.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clip,
                &font,
                &mut log,
                &layout,
            ),
            Action::None
        );
    }

    #[test]
    fn the_common_prefix_helper_matches_the_apache_semantics() {
        assert_eq!(common_prefix(&[]), "");
        assert_eq!(common_prefix(&["abc".into()]), "abc");
        assert_eq!(common_prefix(&["abc".into(), "abd".into()]), "ab");
        assert_eq!(common_prefix(&["abc".into(), "xyz".into()]), "");
        assert_eq!(
            common_prefix(&["Notch".into(), "Nobody".into(), "No".into()]),
            "No"
        );
    }

    #[test]
    fn page_up_and_page_down_scroll_by_a_screenful_less_one() {
        let font = uniform_font();
        let (mut log, layout) = log_with(60, &font);
        let mut s = ChatScreen::new("", 0, &font);
        let mut clip = NullClipboard;
        s.key_typed(
            None,
            201,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(log.scroll_pos(), (layout.line_count(true) - 1) as usize);
        s.key_typed(
            None,
            209,
            Modifiers::NONE,
            &mut clip,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(log.scroll_pos(), 0);
    }

    #[test]
    fn the_wheel_moves_seven_lines_or_one_with_shift() {
        let font = uniform_font();
        let (mut log, layout) = log_with(60, &font);
        let mut s = ChatScreen::new("", 0, &font);
        s.mouse_wheel(1.0, false, &mut log, &layout);
        assert_eq!(log.scroll_pos(), 7);
        s.mouse_wheel(1.0, true, &mut log, &layout);
        assert_eq!(log.scroll_pos(), 8);
        s.mouse_wheel(-1.0, false, &mut log, &layout);
        assert_eq!(log.scroll_pos(), 1);
        s.mouse_wheel(0.0, false, &mut log, &layout);
        assert_eq!(log.scroll_pos(), 1);
    }

    #[test]
    fn the_input_box_spans_the_screen_bottom() {
        let font = uniform_font();
        let mut s = ChatScreen::new("", 0, &font);
        let c = ctx();
        s.init(&c);
        assert_eq!(s.input.rect, Rect::new(4, c.height - 12, c.width - 4, 12));
    }

    #[test]
    fn a_click_on_the_chat_area_finds_the_line_under_it() {
        let font = uniform_font();
        let (log, layout) = log_with(5, &font);
        let c = ctx();
        // Row 0's backdrop is `height - 37 .. height - 28`; vanilla's `k = fromBottom - 27`
        // then indexes by nine, so anything 27..35 above the bottom is the newest line.
        let hit = log.line_at(10, c.height - 30, &c, &layout, true);
        assert!(hit.is_some(), "a click just above the input hits a line");
        assert_eq!(
            hit.unwrap().text,
            "§rline 4§r",
            "the newest line is at the bottom"
        );
        // Nothing when the chat is closed, and nothing above the box.
        assert!(log.line_at(10, c.height - 30, &c, &layout, false).is_none());
        assert!(log.line_at(10, 5, &c, &layout, true).is_none());
        assert!(log.line_at(-5, c.height - 30, &c, &layout, true).is_none());
        // Three rows up is "line 2", which is what makes this a real index and not a constant.
        assert_eq!(
            log.line_at(10, c.height - 30 - 18, &c, &layout, true)
                .map(|l| l.text.as_str()),
            Some("§rline 2§r")
        );
    }

    /// Wrapping a hostile message stays bounded. A component that
    /// spends the whole flatten budget (`MAX_FLATTEN_WORK`) is about 200,000 one-letter runs; the
    /// widest legitimate shape (one 32767-character packet of short siblings) is 8,000.
    #[test]
    fn wrapping_the_largest_flattened_components_stays_fast() {
        let font = uniform_font();
        let mut tower = r#"{"text":"a"}"#.to_owned();
        for _ in 0..30 {
            tower = format!(r#"{{"translate":"%1$s%1$s","with":[{tower}]}}"#);
        }
        let siblings = format!("[\"\"{}]", ",\"a\"".repeat(8_000));
        for (name, json) in [("budget-sized tower", tower), ("8000 siblings", siblings)] {
            let component = ChatComponent::from_json(&json).expect("valid JSON");
            let spans = component.spans(&lang()).len();
            let started = std::time::Instant::now();
            let lines = split_component(&component, 320, &font, &lang(), true);
            let elapsed = started.elapsed();
            println!("{name}: {spans} spans, {} lines, {elapsed:?}", lines.len());
            assert!(
                elapsed < std::time::Duration::from_secs(5),
                "{name}: {elapsed:?}"
            );
        }
    }

    /// A wrap width below one glyph (a negative or NaN `chatScale` or
    /// `chatWidth` in options.txt) used to queue the same text again forever: the client's memory
    /// grew by gigabytes a minute in game. Now every width ends, one glyph per line, and no
    /// glyph is lost. Run on a thread so a regression fails here instead of hanging the suite.
    #[test]
    fn a_wrap_width_below_one_glyph_still_ends_with_every_glyph() {
        let cases: Vec<(i32, &str)> = [-320, -1, 0, 1, 3]
            .into_iter()
            .flat_map(|width| {
                ["Hello world", "\u{a7}cred \u{a7}lbold", "a", "", "  "]
                    .into_iter()
                    .map(move |text| (width, text))
            })
            .collect();
        let (done, results) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let font = uniform_font();
            for (width, text) in cases {
                let lines =
                    split_component(&ChatComponent::text(text), width, &font, &lang(), true);
                let visible: String = lines
                    .iter()
                    .flatten()
                    .map(|span| strip_formatting(&span.text))
                    .collect();
                let _ = done.send((width, text, lines.len(), visible));
            }
        });
        for _ in 0..25 {
            let (width, text, count, visible) = results
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("wrapping must end for every width");
            // At most one line per glyph, plus vanilla's empty lines around an empty piece.
            let glyphs = strip_formatting(text).chars().count();
            assert_eq!(visible, strip_formatting(text), "width {width}: {text:?}");
            assert!(
                count <= 2 * glyphs + 2,
                "width {width}: {text:?}: {count} lines"
            );
        }
    }

    #[test]
    fn component_split_keeps_vanillas_leading_space_on_a_continuation() {
        let font = uniform_font();
        let lines = split_component(&ChatComponent::text("aaaa bbbb"), 40, &font, &lang(), true);
        let literal = lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        assert_eq!(literal, ["aaaa", " bbbb"]);

        // Negative control: FontRenderer.listFormattedStringToWidth is deliberately a different
        // algorithm and drops this separator. This proves the fixture reaches the divergent path.
        assert_eq!(font.wrap_to_lines("aaaa bbbb", 40), ["aaaa", "bbbb"]);
    }

    #[test]
    fn component_split_preserves_sibling_boundaries_and_styles() {
        let font = uniform_font();
        let first = ChatComponent::text("aaaa").with_style(ChatStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/first".into(),
            }),
            ..ChatStyle::default()
        });
        let second = ChatComponent::text("bbbbbbbbbb").with_style(ChatStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/second".into(),
            }),
            ..ChatStyle::default()
        });
        let lines = split_component(&first.with_extra(second), 40, &font, &lang(), true);
        let literal = lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        assert_eq!(literal, ["aaaa", "bbbbbbbb", "bb"]);
        assert_eq!(
            lines[0][0]
                .style
                .click_event
                .as_ref()
                .map(|event| event.value.as_str()),
            Some("/first")
        );
        assert_eq!(
            lines[1][0]
                .style
                .click_event
                .as_ref()
                .map(|event| event.value.as_str()),
            Some("/second")
        );
    }

    #[test]
    fn colors_off_strips_embedded_codes_but_keeps_json_style() {
        let font = uniform_font();
        let component = ChatComponent::text("§aembedded").with_style(ChatStyle {
            color: Some(TextFormat::Red),
            ..ChatStyle::default()
        });
        let mut layout = ChatLayout {
            colors: false,
            ..ChatLayout::default()
        };
        let mut log = ChatLog::new();
        log.add_message(component.clone(), 0, 0, &layout, &font, &lang(), false);
        assert_eq!(log.lines()[0].text, "§r§cembedded§r");

        layout.colors = true;
        log.clear();
        log.add_message(component, 0, 0, &layout, &font, &lang(), false);
        assert_eq!(log.lines()[0].text, "§r§c§c§aembedded§r");
    }

    #[test]
    fn styled_hit_test_returns_the_exact_component_under_x() {
        let font = uniform_font();
        let first = ChatComponent::text("left").with_style(ChatStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/left".into(),
            }),
            ..ChatStyle::default()
        });
        let second = ChatComponent::text("right").with_style(ChatStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::SuggestCommand,
                value: "/right".into(),
            }),
            ..ChatStyle::default()
        });
        let layout = ChatLayout::default();
        let mut log = ChatLog::new();
        log.add_message(
            first.with_extra(second),
            0,
            0,
            &layout,
            &font,
            &lang(),
            true,
        );

        let left = log
            .style_at(3 + 5, ctx().height - 30, &ctx(), &layout, true, &font)
            .and_then(|style| style.click_event.as_ref());
        assert_eq!(left.map(|event| event.value.as_str()), Some("/left"));
        let right = log
            .style_at(3 + 25, ctx().height - 30, &ctx(), &layout, true, &font)
            .and_then(|style| style.click_event.as_ref());
        assert_eq!(right.map(|event| event.value.as_str()), Some("/right"));
        assert!(log
            .style_at(3 + 100, ctx().height - 30, &ctx(), &layout, true, &font)
            .is_none());
    }

    #[test]
    fn gui_space_hover_lookup_resolves_the_exact_styled_span_without_raw_reconstruction() {
        let font = uniform_font();
        let first = ChatComponent::text("left").with_style(ChatStyle {
            hover_event: Some(HoverEvent {
                action: HoverAction::ShowText,
                value: Box::new(ChatComponent::text("left tip")),
            }),
            ..ChatStyle::default()
        });
        let second = ChatComponent::text("right").with_style(ChatStyle {
            hover_event: Some(HoverEvent {
                action: HoverAction::ShowText,
                value: Box::new(ChatComponent::text("right tip")),
            }),
            ..ChatStyle::default()
        });
        let c = ScreenCtx::new(ScaledResolution::new(1280, 601, 2, false), 0.0);
        let layout = ChatLayout::default();
        let mut log = ChatLog::new();
        log.add_message(
            first.with_extra(second),
            0,
            0,
            &layout,
            &font,
            &lang(),
            true,
        );

        assert_eq!(
            log.hover_at(
                3 + 5,
                c.height - 30,
                &c,
                &layout,
                true,
                false,
                "E",
                &font,
                &lang(),
            ),
            Some(ChatHoverContent::Text(vec!["left tip\u{a7}r".into()]))
        );
        assert_eq!(
            log.hover_at(
                3 + 25,
                c.height - 30,
                &c,
                &layout,
                true,
                false,
                "E",
                &font,
                &lang(),
            ),
            Some(ChatHoverContent::Text(vec!["right tip\u{a7}r".into()]))
        );
        assert!(log
            .hover_at(
                3 + 100,
                c.height - 30,
                &c,
                &layout,
                true,
                false,
                "E",
                &font,
                &lang(),
            )
            .is_none());
    }

    #[test]
    fn raw_window_hit_test_uses_integer_scale_before_span_width() {
        let font = uniform_font();
        let first = ChatComponent::text("aa").with_style(ChatStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/left".into(),
            }),
            ..ChatStyle::default()
        });
        let second = ChatComponent::text("bb").with_style(ChatStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/right".into(),
            }),
            ..ChatStyle::default()
        });
        // Height 601 is deliberately not divisible by the selected factor 2. This is the band
        // where converting through ceil-rounded GUI coordinates is not equivalent to Java's
        // raw `mouseY / scaleFactor` arithmetic.
        let c = ScreenCtx::new(ScaledResolution::new(1280, 601, 2, false), 0.0);
        assert_eq!(c.resolution.scale_factor(), 2);
        let layout = ChatLayout::default();
        let mut log = ChatLog::new();
        log.add_message(
            first.with_extra(second),
            0,
            0,
            &layout,
            &font,
            &lang(),
            true,
        );

        // At scale two, raw x=25 gives j=(25/2)-3=9, the final pixel in the first 10px span.
        // x=26 gives j=10; vanilla's strict `runningWidth > j` moves to the second span.
        let event = |x| {
            log.style_at_window(x, 54, &c, &layout, true, &font)
                .and_then(|style| style.click_event.as_ref())
                .map(|event| event.value.as_str())
        };
        assert_eq!(event(25), Some("/left"));
        assert_eq!(event(26), Some("/right"));
        assert!(log
            .style_at_window(25, 53, &c, &layout, true, &font)
            .is_none());
    }

    #[test]
    fn component_clicks_preserve_shift_insertion_and_consumption_rules() {
        let font = uniform_font();
        let mut screen = ChatScreen::new(">", 0, &font);
        let insertion = ResolvedStyle {
            insertion: Some("name".into()),
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/must_not_run".into(),
            }),
            ..ResolvedStyle::default()
        };
        assert_eq!(
            screen.component_clicked(&insertion, true, true, &font),
            None
        );
        assert_eq!(screen.input.text(), ">name");

        let run = ResolvedStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/help".into(),
            }),
            ..ResolvedStyle::default()
        };
        assert_eq!(
            screen.component_clicked(&run, false, true, &font),
            Some(Action::RunCommand("/help".into()))
        );

        let page = ResolvedStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::ChangePage,
                value: "2".into(),
            }),
            ..ResolvedStyle::default()
        };
        assert_eq!(
            screen.component_clicked(&page, false, true, &font),
            Some(Action::None),
            "a non-book screen still consumes a recognised change_page click"
        );

        let url = ResolvedStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::OpenUrl,
                value: "https://example.invalid".into(),
            }),
            ..ResolvedStyle::default()
        };
        assert_eq!(
            screen.component_clicked(&url, false, false, &font),
            None,
            "chatLinks=false leaves the click unconsumed"
        );
    }

    #[test]
    fn hover_payloads_keep_formatted_text_and_advanced_entity_gate() {
        let font = uniform_font();
        let style = |action, value: &str| ResolvedStyle {
            hover_event: Some(HoverEvent {
                action,
                value: Box::new(ChatComponent::text(value)),
            }),
            ..ResolvedStyle::default()
        };

        assert_eq!(
            hover_content(&ResolvedStyle::default(), false, "E", &font, &lang()),
            None,
            "a span without a hover event is not a tooltip"
        );
        assert_eq!(
            hover_content(
                &style(HoverAction::ShowText, "one\ntwo\n"),
                false,
                "E",
                &font,
                &lang(),
            ),
            Some(ChatHoverContent::Text(vec![
                "one".into(),
                "two".into(),
                "\u{00a7}r".into(),
            ]))
        );
        assert_eq!(
            hover_content(
                &style(
                    HoverAction::ShowItem,
                    r#"{id:"minecraft:stone",Count:1b,Damage:0s,tag:{display:{Name:"Hover Stone"}}}"#,
                ),
                false,
                "E",
                &font,
                &lang(),
            ),
            Some(ChatHoverContent::Text(vec![
                "\u{a7}f\u{a7}oHover Stone\u{a7}r".into(),
            ])),
            "the live SHOW_ITEM consumer parses Mojangson, loads the stack, and reaches the tooltip renderer"
        );
        assert_eq!(
            hover_content(
                &style(HoverAction::ShowEntity, "{id:x}"),
                false,
                "E",
                &font,
                &lang(),
            ),
            None
        );
        assert_eq!(
            hover_content(
                &style(HoverAction::ShowEntity, "{id:x}"),
                true,
                "E",
                &font,
                &lang(),
            ),
            Some(ChatHoverContent::Text(vec!["".into(), "x".into()]))
        );

        // Java's `hasKey("type", 8)` gate ignores a numeric type tag rather than coercing it to
        // text. The remaining name/id lines are still valid tooltip output.
        assert_eq!(
            hover_content(
                &style(HoverAction::ShowEntity, r#"{name:"N",type:1,id:"x"}"#),
                true,
                "E",
                &font,
                &lang(),
            ),
            Some(ChatHoverContent::Text(vec!["N".into(), "x".into()]))
        );
    }

    #[test]
    fn item_and_entity_hover_snbt_failures_draw_their_vanilla_red_fallbacks() {
        let font = uniform_font();
        let style = |action, value: &str| ResolvedStyle {
            hover_event: Some(HoverEvent {
                action,
                value: Box::new(ChatComponent::text(value)),
            }),
            ..ResolvedStyle::default()
        };

        assert_eq!(
            hover_content(
                &style(HoverAction::ShowItem, "{id:1s"),
                false,
                "E",
                &font,
                &lang(),
            ),
            Some(ChatHoverContent::Text(vec!["\u{a7}cInvalid Item!".into()])),
            "a malformed item payload must not disappear"
        );
        assert_eq!(
            hover_content(
                &style(HoverAction::ShowEntity, "{name:Steve"),
                true,
                "E",
                &font,
                &lang(),
            ),
            Some(ChatHoverContent::Text(
                vec!["\u{a7}cInvalid Entity!".into()]
            )),
            "a malformed entity payload must not disappear"
        );
    }

    #[test]
    fn show_item_loader_matches_itemstack_nbt_id_and_damage_rules() {
        let tagged = parse_snbt_compound(
            r#"{id:"minecraft:diamond_sword",Count:-2b,Damage:-7s,tag:{display:{Name:"Negative Damage"}}}"#,
        )
        .expect("valid Mojangson");
        let stack = item_stack_from_nbt(tagged).expect("registered string id");
        assert_eq!(stack.item_id, 276, "registry-name id");
        assert_eq!(
            stack.count, -2,
            "NBTTagCompound.getByte sign-extends into stackSize"
        );
        assert_eq!(
            stack.damage, 0,
            "ItemStack.setItemDamage clamps negative metadata"
        );
        assert_eq!(
            stack
                .nbt
                .as_ref()
                .and_then(|tag| tag.get_compound("display"))
                .and_then(|display| display.get_str("Name")),
            Some("Negative Damage"),
            "the optional tag compound survives loadItemStackFromNBT"
        );

        let legacy =
            parse_snbt_compound(r#"{id:276s,Count:1b,Damage:0s}"#).expect("valid Mojangson");
        assert_eq!(
            item_stack_from_nbt(legacy).map(|stack| stack.item_id),
            Some(276),
            "a non-string id follows Item.getItemById(getShort(\"id\"))"
        );
        for invalid in [r#"{id:"minecraft:not_an_item"}"#, r#"{id:0s}"#] {
            assert!(
                item_stack_from_nbt(parse_snbt_compound(invalid).expect("valid Mojangson"))
                    .is_none(),
                "unknown Item registry rows make loadItemStackFromNBT return null: {invalid}"
            );
        }
    }

    #[test]
    fn a_valid_entity_hover_keeps_vanillas_name_type_id_and_entitylist_number() {
        let font = uniform_font();
        let style = ResolvedStyle {
            hover_event: Some(HoverEvent {
                action: HoverAction::ShowEntity,
                value: Box::new(ChatComponent::text(
                    r#"{name:"Steve",type:"minecraft:player",id:"00000000-0000-0000-0000-000000000000"}"#,
                )),
            }),
            ..ResolvedStyle::default()
        };
        assert_eq!(
            hover_content(&style, true, "E", &font, &lang()),
            Some(ChatHoverContent::Text(vec![
                "Steve".into(),
                // `minecraft:player` is not an EntityList registration in 1.8.9, so Java's
                // `EntityList.getIDFromString` returns its historical Pig-id fallback, 90.
                "Type: minecraft:player (90)".into(),
                "00000000-0000-0000-0000-000000000000".into(),
            ]))
        );

        let known = ResolvedStyle {
            hover_event: Some(HoverEvent {
                action: HoverAction::ShowEntity,
                value: Box::new(ChatComponent::text(
                    r#"{name:"Zombie",type:"Zombie",id:"z"}"#,
                )),
            }),
            ..ResolvedStyle::default()
        };
        assert_eq!(
            hover_content(&known, true, "E", &font, &lang()),
            Some(ChatHoverContent::Text(vec![
                "Zombie".into(),
                "Type: Zombie (54)".into(),
                "z".into(),
            ])),
            "known EntityList names use their registered numeric ids"
        );
    }

    #[test]
    fn show_achievement_resolves_stats_achievements_descriptions_and_invalid_ids() {
        let font = uniform_font();
        let mut language = Language::new("oracle");
        language.merge_lang_str(
            "stat.leaveGame=Games Quit\n\
             achievement.mineWood=Getting Wood\n\
             achievement.mineWood.desc=Punch a tree\n\
             achievement.overpowered=Overpowered\n\
             achievement.overpowered.desc=Eat the apple\n\
             achievement.openInventory=Taking Inventory\n\
             achievement.openInventory.desc=Press '%1$s' now\n\
             stats.tooltip.type.statistic=Statistic\n\
             stats.tooltip.type.achievement=Achievement\n",
        );
        let style = |id: &str| ResolvedStyle {
            hover_event: Some(HoverEvent {
                action: HoverAction::ShowAchievement,
                value: Box::new(ChatComponent::text(id)),
            }),
            ..ResolvedStyle::default()
        };
        let resolve = |id| hover_content(&style(id), false, "Button 3", &font, &language);

        assert_eq!(
            resolve("stat.leaveGame"),
            Some(ChatHoverContent::Achievement(vec![
                "\u{a7}7Games Quit\u{a7}r".into(),
                "\u{a7}oStatistic\u{a7}r".into(),
            ])),
            "ordinary stats are grey and have no description"
        );
        assert_eq!(
            resolve("achievement.mineWood"),
            Some(ChatHoverContent::Achievement(vec![
                "\u{a7}aGetting Wood\u{a7}r".into(),
                "\u{a7}oAchievement\u{a7}r".into(),
                "Punch a tree".into(),
            ])),
            "ordinary achievements are green and include the wrapped description"
        );
        assert_eq!(
            resolve("achievement.overpowered"),
            Some(ChatHoverContent::Achievement(vec![
                "\u{a7}5Overpowered\u{a7}r".into(),
                "\u{a7}oAchievement\u{a7}r".into(),
                "Eat the apple".into(),
            ])),
            "special achievements use dark purple"
        );
        assert_eq!(
            resolve("achievement.openInventory"),
            Some(ChatHoverContent::Achievement(vec![
                "\u{a7}aTaking Inventory\u{a7}r".into(),
                "\u{a7}oAchievement\u{a7}r".into(),
                "Press 'Button 3' now".into(),
            ])),
            "the one IStatStringFormat description receives the live inventory binding"
        );
        assert_eq!(
            resolve("achievement.definitelyMissing"),
            Some(ChatHoverContent::Text(vec![
                "\u{a7}cInvalid statistic/achievement!".into(),
            ]))
        );
    }
}
