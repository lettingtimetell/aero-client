//! The `chat` mod's half of the chat log: repeated messages
//! stacked into one line with an `(xN)` counter, arrival timestamps, the backdrop's opacity, the
//! text shadow, raising the box above the health bar, and right-click copy.
//!
//! [`ChatMods::default`] is vanilla, and it is what the log holds while the mod is off: adding a
//! message then costs one `bool`/`u8` test more than vanilla, and the draw and the hit tests use
//! vanilla's numbers. Only id-0 messages (server chat and the client's own notices) are stacked
//! or stamped. A line with a non-zero id is replaced in place by vanilla's own rule and ends a
//! stack.
//!
//! # Stacking
//!
//! Consecutive only: a message whose formatted text equals
//! the previous message's, and is not blank, replaces it. Its wrapped lines are the first
//! `lines` entries of `drawn` (newest first) and it is `received[0]`, so both are dropped and the
//! new message is added with a grey ` (xN)` sibling. Blank or whitespace-only lines, which servers
//! use as spacers, are never stacked. Nothing here runs per frame.
//!
//! # Timestamps
//!
//! A grey `[h:mm] ` (12-hour, no AM/PM) or `[HH:mm] ` sibling in front of the
//! message, stamped on arrival and kept in `received`, so a re-wrap keeps the original time. The
//! stack comparison uses the text without the stamp, so a stacked line shows the latest time.

use mc_assets::Language;

use crate::font::component::{ChatComponent, ChatStyle};
use crate::font::{strip_formatting, FontRenderer, TextFormat};
use crate::screen_stack::ScreenCtx;

use super::{ChatLayout, ChatLog};

/// The `chat` mod's options, as the log applies them. The default is vanilla.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatMods {
    /// Stack a message identical to the one before it into one line with a counter.
    pub stack: bool,
    /// Stamp new messages with the local time: 0 off, 1 12-hour, 2 24-hour.
    pub timestamps: u8,
    /// Multiplier on the line backdrop's alpha, `0.0..=1.0` (vanilla 1.0).
    pub background: f32,
    /// Draw chat text with the font's drop shadow (vanilla: yes).
    pub shadow: bool,
    /// GUI pixels the chat box is lifted by (vanilla 0, the mod's option [`ChatMods::RAISE`]).
    pub raise: i32,
}

impl Default for ChatMods {
    fn default() -> Self {
        ChatMods {
            stack: false,
            timestamps: 0,
            background: 1.0,
            shadow: true,
            raise: 0,
        }
    }
}

impl ChatMods {
    /// How far "Raise above health bar" lifts the chat: 12 pixels, as in vanilla 1.12 (clear of
    /// the hearts and the armour row).
    pub const RAISE: i32 = 12;

    /// Whether new id-0 messages take the stacking/timestamp path.
    pub(super) fn rewrites_messages(&self) -> bool {
        self.stack || self.timestamps != 0
    }

    /// The backdrop alpha behind a line whose text alpha is `alpha`: vanilla's `alpha / 2`,
    /// scaled by [`ChatMods::background`]. Exactly vanilla's value at 1.0.
    pub(super) fn backdrop_alpha(&self, alpha: i32) -> u32 {
        let half = (alpha / 2).max(0) as u32;
        if self.background >= 1.0 {
            half
        } else {
            (half as f32 * self.background.max(0.0)).round() as u32
        }
    }
}

/// The last id-0 message, for stacking.
#[derive(Debug, Clone, Default)]
pub(super) struct StackState {
    /// The fields below describe `received[0]` and the first `lines` drawn lines.
    pub(super) valid: bool,
    /// Its formatted text without the stamp or the counter.
    pub(super) text: String,
    /// How many times it arrived in a row.
    pub(super) count: u32,
    /// How many display lines it wrapped into.
    pub(super) lines: usize,
}

/// The local wall clock as `(hour, minute)`, for timestamps.
pub(super) fn local_clock() -> (u32, u32) {
    use chrono::Timelike as _;
    let now = chrono::Local::now();
    (now.hour(), now.minute())
}

/// The local `(hour, minute)` of a Unix time in milliseconds (a
/// replayed message's own time); the wall clock for a time the calendar cannot place.
pub(super) fn local_clock_at(unix_ms: i64) -> (u32, u32) {
    use chrono::{TimeZone as _, Timelike as _};
    chrono::Local
        .timestamp_millis_opt(unix_ms)
        .earliest()
        .map_or_else(local_clock, |at| (at.hour(), at.minute()))
}

/// `[h:mm] ` for the 12-hour mode (`0`/`12` both show as `12`), `[HH:mm] ` otherwise.
fn timestamp(mode: u8, (hour, minute): (u32, u32)) -> String {
    if mode == 1 {
        let hour = match hour % 12 {
            0 => 12,
            hour => hour,
        };
        format!("[{hour}:{minute:02}] ")
    } else {
        format!("[{hour:02}:{minute:02}] ")
    }
}

/// Whether a formatted message has nothing but formatting codes and whitespace.
fn is_blank(formatted: &str) -> bool {
    strip_formatting(formatted).trim().is_empty()
}

/// `component` with a grey stamp in front and, from the second repeat on, a grey ` (xN)` after
/// it, under an empty root (the shape servers send most messages in, `{"text":"","extra":[..]}`),
/// so the original's own style still inherits from the default and is drawn unchanged.
fn decorate(component: ChatComponent, stamp: Option<String>, count: u32) -> ChatComponent {
    if stamp.is_none() && count < 2 {
        return component;
    }
    let gray = || ChatStyle {
        color: Some(TextFormat::Gray),
        ..ChatStyle::default()
    };
    let mut root = ChatComponent::text("");
    if let Some(stamp) = stamp {
        root.extra
            .push(ChatComponent::text(stamp).with_style(gray()));
    }
    root.extra.push(component);
    if count >= 2 {
        root.extra
            .push(ChatComponent::text(format!(" (x{count})")).with_style(gray()));
    }
    root
}

/// The copied text of one display line: no stamp in front, no counter behind.
fn strip_decorations(text: &str) -> &str {
    let mut text = text;
    if let Some(rest) = text.strip_prefix('[') {
        if let Some(end) = rest.find("] ") {
            if is_clock(&rest[..end]) {
                text = &rest[end + 2..];
            }
        }
    }
    if let Some(head) = text.strip_suffix(')') {
        if let Some(open) = head.rfind(" (x") {
            let digits = &head[open + 3..];
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                text = &head[..open];
            }
        }
    }
    text
}

/// `h:mm` or `hh:mm`.
fn is_clock(text: &str) -> bool {
    let Some((hours, minutes)) = text.split_once(':') else {
        return false;
    };
    (1..=2).contains(&hours.len())
        && hours.bytes().all(|b| b.is_ascii_digit())
        && minutes.len() == 2
        && minutes.bytes().all(|b| b.is_ascii_digit())
}

impl ChatLog {
    /// Apply the `chat` mod's options. [`ChatMods::default`] restores vanilla. Messages already
    /// in the log keep their stamps and counters; the look options apply from the next draw.
    pub fn set_mods(&mut self, mods: ChatMods) {
        self.mods = mods;
    }

    /// The options in force.
    #[must_use]
    pub fn mods(&self) -> ChatMods {
        self.mods
    }

    /// [`ChatLog::add_message`] for an id-0 message while stacking or timestamps are on.
    /// `clock` is the local `(hour, minute)` the stamp shows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn add_message_modded(
        &mut self,
        component: ChatComponent,
        tick: i32,
        layout: &ChatLayout,
        font: &FontRenderer,
        lang: &Language,
        chat_open: bool,
        clock: (u32, u32),
    ) {
        let text = self.mods.stack.then(|| component.formatted_text(lang));
        let stackable = text.as_deref().is_some_and(|text| !is_blank(text));
        let repeat =
            stackable && self.stack.valid && text.as_deref() == Some(self.stack.text.as_str());
        let count = if repeat {
            self.stack.count.saturating_add(1)
        } else {
            1
        };
        if repeat {
            self.remove_last_message(layout, chat_open);
        }
        let stamp = (self.mods.timestamps != 0).then(|| timestamp(self.mods.timestamps, clock));
        let component = decorate(component, stamp, count);
        let lines = self.push_drawn(&component, 0, tick, layout, font, lang, chat_open);
        self.received.push_front((tick, component, 0));
        self.received.truncate(self.max_lines);
        self.stack = match text {
            Some(text) if stackable => StackState {
                valid: true,
                text,
                count,
                lines,
            },
            _ => StackState::default(),
        };
    }

    /// Drop the stacked message's display lines and its `received` entry, keeping a scrolled-up
    /// reader on the same lines.
    fn remove_last_message(&mut self, layout: &ChatLayout, chat_open: bool) {
        let lines = self.stack.lines.min(self.drawn.len());
        self.drawn.drain(..lines);
        self.received.pop_front();
        if chat_open && self.scroll > 0 {
            self.scroll_by(-(lines as i32), layout, chat_open);
        }
    }

    /// What a right-click copies (`chat` mod): the display line under a GUI-space cursor, with
    /// formatting codes, a timestamp and a stack counter removed and surrounding whitespace
    /// trimmed. `None` off the lines or on a line with no text.
    #[must_use]
    pub fn copy_text_at(
        &self,
        gui_x: i32,
        gui_y: i32,
        ctx: &ScreenCtx,
        layout: &ChatLayout,
    ) -> Option<String> {
        let line = self.line_at(gui_x, gui_y, ctx, layout, true)?;
        let plain = strip_formatting(&line.text);
        let text = strip_decorations(&plain).trim();
        (!text.is_empty()).then(|| text.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::{DrawList, ScaledResolution, UiTexture};

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn lang() -> Language {
        Language::new("en_US")
    }

    fn modded(stack: bool, timestamps: u8) -> ChatLog {
        let mut log = ChatLog::new();
        log.set_mods(ChatMods {
            stack,
            timestamps,
            ..ChatMods::default()
        });
        log
    }

    /// Add `text` the way the client does (`add_message`), or straight through the modded path
    /// with a fixed clock when timestamps are on.
    fn add(log: &mut ChatLog, text: &str, id: i32, font: &FontRenderer) {
        let layout = ChatLayout::default();
        if id == 0 && log.mods.timestamps != 0 {
            log.add_message_modded(
                ChatComponent::text(text),
                0,
                &layout,
                font,
                &lang(),
                false,
                (13, 7),
            );
        } else {
            log.add_message(
                ChatComponent::text(text),
                id,
                0,
                &layout,
                font,
                &lang(),
                false,
            );
        }
    }

    fn plain_lines(log: &ChatLog) -> Vec<String> {
        log.lines()
            .iter()
            .map(|line| strip_formatting(&line.text))
            .collect()
    }

    #[test]
    fn the_default_is_vanilla_and_does_not_rewrite() {
        let mods = ChatMods::default();
        assert!(!mods.rewrites_messages());
        assert_eq!((mods.background, mods.shadow, mods.raise), (1.0, true, 0));
        for alpha in [4, 5, 100, 254, 255] {
            assert_eq!(mods.backdrop_alpha(alpha), (alpha / 2) as u32);
        }
        // The plain log does not stack: identical lines stay separate.
        let font = uniform_font();
        let mut log = ChatLog::new();
        add(&mut log, "+1 coins", 0, &font);
        add(&mut log, "+1 coins", 0, &font);
        assert_eq!(plain_lines(&log), ["+1 coins", "+1 coins"]);
        assert_eq!(log.received_count(), 2);
    }

    #[test]
    fn consecutive_repeats_stack_into_one_line_with_a_counter() {
        let font = uniform_font();
        let mut log = modded(true, 0);
        add(&mut log, "+1 coins", 0, &font);
        add(&mut log, "+1 coins", 0, &font);
        add(&mut log, "+1 coins", 0, &font);
        assert_eq!(plain_lines(&log), ["+1 coins (x3)"]);
        assert_eq!(log.received_count(), 1);
        // A different line ends the run; the same text after it starts a new one.
        add(&mut log, "Kill!", 0, &font);
        add(&mut log, "+1 coins", 0, &font);
        assert_eq!(plain_lines(&log), ["+1 coins", "Kill!", "+1 coins (x3)"]);
    }

    #[test]
    fn blank_spacers_and_non_zero_ids_never_stack() {
        let font = uniform_font();
        let mut log = modded(true, 0);
        add(&mut log, "", 0, &font);
        add(&mut log, "", 0, &font);
        add(&mut log, "  ", 0, &font);
        assert_eq!(log.lines().len(), 3);
        add(&mut log, "hint", 1, &font);
        add(&mut log, "hint", 1, &font);
        assert_eq!(
            plain_lines(&log)[0],
            "hint",
            "id 1 replaces itself, vanilla"
        );
        // An id-1 line between two equal messages breaks the run.
        add(&mut log, "same", 0, &font);
        add(&mut log, "tab", 1, &font);
        add(&mut log, "same", 0, &font);
        assert_eq!(plain_lines(&log)[..3], ["same", "tab", "same"]);
    }

    #[test]
    fn a_wrapped_message_stacks_all_its_lines() {
        let font = uniform_font();
        let mut log = modded(true, 0);
        // 70 glyphs of 5 pixels: wraps into two lines at the default 320-pixel width.
        let long = "x".repeat(70);
        add(&mut log, "before", 0, &font);
        add(&mut log, &long, 0, &font);
        assert_eq!(log.lines().len(), 3);
        add(&mut log, &long, 0, &font);
        assert_eq!(
            log.lines().len(),
            3,
            "the first copy's two lines were replaced"
        );
        let lines = plain_lines(&log);
        assert!(lines[0].ends_with(" (x2)"), "{lines:?}");
        assert_eq!(lines[2], "before");
        assert_eq!(log.received_count(), 2);
    }

    #[test]
    fn a_rewrap_keeps_the_run_and_its_line_count() {
        let font = uniform_font();
        let lang = lang();
        let mut log = modded(true, 0);
        let long = "y".repeat(70);
        add(&mut log, &long, 0, &font);
        add(&mut log, &long, 0, &font);
        // Narrower chat: the stacked message now wraps into more lines.
        let narrow = ChatLayout {
            width_scale: 0.2,
            ..ChatLayout::default()
        };
        log.refresh(&narrow, &font, &lang);
        let wrapped = log.lines().len();
        assert!(wrapped > 2);
        log.add_message_modded(
            ChatComponent::text(long.clone()),
            1,
            &narrow,
            &font,
            &lang,
            false,
            (0, 0),
        );
        assert_eq!(log.received_count(), 1);
        assert_eq!(
            log.lines().len(),
            wrapped,
            "every re-wrapped line was replaced"
        );
        assert!(plain_lines(&log)[0].ends_with(" (x3)"));
    }

    #[test]
    fn a_stacked_repeat_keeps_a_scrolled_up_reader_on_the_same_lines() {
        let font = uniform_font();
        let lang = lang();
        let layout = ChatLayout::default();
        let mut log = modded(true, 0);
        let add_open = |log: &mut ChatLog, text: String| {
            log.add_message(ChatComponent::text(text), 0, 0, &layout, &font, &lang, true);
        };
        for i in 0..40 {
            add_open(&mut log, format!("line {i}"));
        }
        add_open(&mut log, "spam".to_owned());
        log.scroll_by(5, &layout, true);
        assert_eq!(log.scroll_pos(), 5);
        let shown = |log: &ChatLog| -> Vec<String> {
            log.visible_lines(&layout, true)
                .iter()
                .map(|line| strip_formatting(&line.text))
                .collect()
        };
        let before = shown(&log);
        add_open(&mut log, "spam".to_owned());
        assert_eq!(log.scroll_pos(), 5, "one line out, one line in");
        assert_eq!(shown(&log), before, "the reader still sees the same lines");
        assert_eq!(strip_formatting(&log.lines()[0].text), "spam (x2)");
    }

    #[test]
    fn timestamps_use_the_arrival_clock_in_both_modes() {
        assert_eq!(timestamp(1, (0, 5)), "[12:05] ");
        assert_eq!(timestamp(1, (12, 30)), "[12:30] ");
        assert_eq!(timestamp(1, (13, 7)), "[1:07] ");
        assert_eq!(timestamp(2, (9, 3)), "[09:03] ");
        assert_eq!(timestamp(2, (23, 59)), "[23:59] ");
        let font = uniform_font();
        let mut log = modded(false, 1);
        add(&mut log, "hello", 0, &font);
        add(&mut log, "hello", 0, &font);
        assert_eq!(plain_lines(&log), ["[1:07] hello", "[1:07] hello"]);
        // With stacking too, the stamp is not part of the comparison.
        let mut log = modded(true, 2);
        add(&mut log, "hello", 0, &font);
        add(&mut log, "hello", 0, &font);
        assert_eq!(plain_lines(&log), ["[13:07] hello (x2)"]);
        // The stamp and counter are grey; the message keeps its own (default) colour.
        let text = &log.lines()[0].text;
        assert!(text.contains("\u{a7}7[13:07] "), "{text:?}");
        assert!(text.contains("\u{a7}7 (x2)"), "{text:?}");
    }

    /// With a stamp clock set (a replay: the recording's clock), a
    /// message added the client's way shows that clock's time, whatever the time is now; `None`
    /// is the wall clock again.
    #[test]
    fn a_stamp_clock_replaces_the_wall_clock() {
        let font = uniform_font();
        let layout = ChatLayout::default();
        let mut log = modded(false, 2);
        let add_now = |log: &mut ChatLog, text: &str| {
            log.add_message(
                ChatComponent::text(text),
                0,
                0,
                &layout,
                &font,
                &lang(),
                false,
            );
        };
        // Two recorded moments 11 h 23 min apart, so their stamps differ in any time zone.
        let (early, late) = (1_794_621_923_186_i64, 1_794_621_923_186_i64 + 41_000_000);
        let stamp = |unix_ms: i64| {
            let (hour, minute) = local_clock_at(unix_ms);
            format!("[{hour:02}:{minute:02}]")
        };
        assert_ne!(stamp(early), stamp(late));
        log.set_stamp_clock(Some(early));
        add_now(&mut log, "first");
        log.set_stamp_clock(Some(late));
        add_now(&mut log, "second");
        assert_eq!(
            plain_lines(&log),
            [
                format!("{} second", stamp(late)),
                format!("{} first", stamp(early))
            ]
        );
        // Back on the wall clock.
        log.set_stamp_clock(None);
        let before = local_clock();
        add_now(&mut log, "live");
        let after = local_clock();
        let line = &plain_lines(&log)[0];
        let at = |(hour, minute): (u32, u32)| format!("[{hour:02}:{minute:02}] live");
        assert!(line == &at(before) || line == &at(after), "{line}");
    }

    #[test]
    fn copy_strips_codes_stamp_and_counter() {
        assert_eq!(strip_decorations("[1:07] hello (x2)"), "hello");
        assert_eq!(
            strip_decorations("[13:07] [MVP+] Bob: hi"),
            "[MVP+] Bob: hi"
        );
        assert_eq!(
            strip_decorations("[MVP+] Bob: hi (x)"),
            "[MVP+] Bob: hi (x)"
        );
        assert_eq!(strip_decorations("plain"), "plain");
        let font = uniform_font();
        let mut log = modded(true, 1);
        let layout = ChatLayout::default();
        log.add_message_modded(
            ChatComponent::text("\u{a7}aGreen \u{a7}lbold"),
            0,
            &layout,
            &font,
            &lang(),
            false,
            (9, 41),
        );
        log.add_message_modded(
            ChatComponent::text("\u{a7}aGreen \u{a7}lbold"),
            0,
            &layout,
            &font,
            &lang(),
            false,
            (9, 42),
        );
        let c = ctx();
        // The bottom line's row is GUI y 480 - 28 - 9 .. 480 - 28.
        assert_eq!(
            log.copy_text_at(10, 480 - 30, &c, &layout).as_deref(),
            Some("Green bold")
        );
        assert_eq!(
            log.copy_text_at(10, 480 - 60, &c, &layout),
            None,
            "above the line"
        );
    }

    #[test]
    fn raise_moves_the_draw_and_every_hit_test_together() {
        let font = uniform_font();
        let mut log = ChatLog::new();
        add(&mut log, "bottom", 0, &font);
        let layout = ChatLayout::default();
        let c = ctx();
        // Vanilla: the bottom row spans GUI y 443..452 (from-bottom 28..37).
        assert!(log.line_at(10, 450, &c, &layout, true).is_some());
        assert!(log.line_at(10, 440, &c, &layout, true).is_none());
        let vanilla_bounds = log.bounds(&c, &layout, true);
        log.set_mods(ChatMods {
            raise: ChatMods::RAISE,
            ..ChatMods::default()
        });
        assert!(log.line_at(10, 450, &c, &layout, true).is_none());
        assert!(log.line_at(10, 438, &c, &layout, true).is_some());
        assert_eq!(
            log.bounds(&c, &layout, true).y,
            vanilla_bounds.y - ChatMods::RAISE
        );
        // The window-pixel variant (origin bottom-left, scale factor 1) moves the same way.
        assert!(log
            .line_at_window(10, 480 - 450, &c, &layout, true)
            .is_none());
        assert!(log
            .line_at_window(10, 480 - 438, &c, &layout, true)
            .is_some());
        // And the drawn backdrop moved up by the same 12 pixels.
        let mut d = DrawList::new();
        let mut font = font;
        log.draw(&mut d, &mut font, &c, 0, &layout, true);
        let backdrop_bottom = d
            .batches()
            .iter()
            .filter(|b| b.texture == UiTexture::Solid)
            .flat_map(|b| d.vertices()[b.range.clone()].iter().map(|v| v.pos[1]))
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(backdrop_bottom, (480 - 28 - ChatMods::RAISE) as f32);
    }

    #[test]
    fn background_and_shadow_reach_the_draw() {
        let font = uniform_font();
        let layout = ChatLayout::default();
        let c = ctx();
        let draw = |log: &ChatLog| {
            let mut font = uniform_font();
            let mut d = DrawList::new();
            log.draw(&mut d, &mut font, &c, 0, &layout, true);
            d
        };
        let mut log = ChatLog::new();
        add(&mut log, "hello", 0, &font);
        let vanilla = draw(&log);
        let solid_alphas = |d: &DrawList| -> Vec<f32> {
            d.batches()
                .iter()
                .filter(|b| b.texture == UiTexture::Solid)
                .flat_map(|b| d.vertices()[b.range.clone()].iter().map(|v| v.color[3]))
                .collect()
        };
        // Chat open: alpha 255, backdrop 127.
        assert!(solid_alphas(&vanilla)
            .iter()
            .all(|&a| (a - 127.0 / 255.0).abs() < 1e-6));
        log.set_mods(ChatMods {
            background: 0.0,
            ..ChatMods::default()
        });
        assert!(solid_alphas(&draw(&log)).is_empty(), "0% draws no backdrop");
        log.set_mods(ChatMods {
            background: 0.5,
            ..ChatMods::default()
        });
        assert!(solid_alphas(&draw(&log))
            .iter()
            .all(|&a| (a - 64.0 / 255.0).abs() < 1e-6));
        log.set_mods(ChatMods {
            shadow: false,
            ..ChatMods::default()
        });
        assert!(
            draw(&log).vertices().len() < vanilla.vertices().len(),
            "no shadow pass"
        );
        // Back to the defaults: byte-identical to the vanilla draw.
        log.set_mods(ChatMods::default());
        assert_eq!(draw(&log), vanilla);
    }
}
