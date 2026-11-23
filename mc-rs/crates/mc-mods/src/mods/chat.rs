//! Chat: stack repeated messages, timestamps, background, shadow, raise, right-click copy.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Vanilla's own chat sliders (scale, width,
//! height, text opacity, visibility) stay in vanilla's Chat Settings and are not duplicated.
//! "Stack repeated messages" defaults to **on**. Every other
//! default is vanilla's look, so enabling the mod changes only stacking (and right-click,
//! which vanilla's chat ignores).
//!
//! * The chat input always takes 256 characters, by design without an option. The old
//!   "256 Character Chat" toggle (`long_chat`) is gone; a settings file that
//!   still has it loads unchanged (the key is kept as an unknown option, without a warning).
//! * [`KEEP_CHAT`] and [`HISTORY_LENGTH`] apply **whether or not the Chat module is on**: chat
//!   and the sent-message history survive leaving a server, server switches and the title screen
//!   for the whole client run, and the scrollback holds 1000 lines. Vanilla is Keep off with 100
//!   lines. The client half is `mc-client/src/app/mods/x2_chat.rs`.
//!
//! [`SYMBOL_SHADOWS`] ("symbols in the chat dont render with a
//! background shadow like text"). Vanilla 1.8.9 draws the shadow of a symbol (a glyph
//! `ascii.png` does not have: hearts, stars, check marks, arrows) exactly under the symbol, so
//! none shows. On, it sits one pixel down and right like a letter's, in every text with a
//! shadow. It applies **whether or not the Chat module is on**; default on by design, off is
//! vanilla. The client half is `mc-client/src/app/mods/w18_fa.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// "Keep chat between servers": the option id.
pub const KEEP_CHAT: &str = "keep_chat";
/// "History length", in display lines: the option id.
pub const HISTORY_LENGTH: &str = "history_length";
/// The smallest history length: vanilla's 100 (`GuiNewChat`'s cap).
pub const HISTORY_MIN: f32 = 100.0;
/// The largest history length.
pub const HISTORY_MAX: f32 = 5000.0;
/// "Symbol shadows": the option id.
pub const SYMBOL_SHADOWS: &str = "symbol_shadows";

static OPTIONS: [OptionDescriptor; 9] = [
    OptionDescriptor {
        id: "stack",
        label: "Stack repeated messages",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "A message identical to the one before it becomes one line with (x2).",
    },
    OptionDescriptor {
        id: "timestamps",
        label: "Timestamps",
        kind: OptionKind::Choice {
            choices: &["Off", "12-hour", "24-hour"],
        },
        default: OptionValue::Choice(0),
        description: "Applies to new messages.",
    },
    OptionDescriptor {
        id: "background_opacity",
        label: "Background opacity",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "100% is vanilla's backdrop.",
    },
    OptionDescriptor {
        id: "text_shadow",
        label: "Text shadow",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "raise",
        label: "Raise above health bar",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Moves chat up 12 pixels.",
    },
    OptionDescriptor {
        id: "copy",
        label: "Right-click to copy",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "With chat open, right-click a line to copy its text.",
    },
    OptionDescriptor {
        id: KEEP_CHAT,
        label: "Keep chat between servers",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Keeps chat and the messages you sent when you leave a server or go to the \
                      title screen, until you quit the game. Works even with Chat off. Off: the \
                      title screen clears chat, as in vanilla.",
    },
    OptionDescriptor {
        id: HISTORY_LENGTH,
        label: "History length",
        kind: OptionKind::Slider {
            min: HISTORY_MIN,
            max: HISTORY_MAX,
            step: 100.0,
            unit: " lines",
            decimals: 0,
        },
        default: OptionValue::Float(1000.0),
        description: "How many chat lines you can scroll back through. Works even with Chat \
                      off. 100 is vanilla.",
    },
    OptionDescriptor {
        id: SYMBOL_SHADOWS,
        label: "Symbol shadows",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Hearts, stars, check marks and other symbols get the same drop shadow as \
                      letters, in chat and everywhere else text has one. Works even with Chat \
                      off. Off is vanilla: a symbol's shadow is hidden under it.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "chat",
    name: "Chat",
    category: Category::Interface,
    description: "Stack repeats, timestamps, background, raise and copy. Chat history is kept \
                  between servers either way (see History).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menu_rules::{applies_while_off, groups_for, settings_layout};

    /// "Symbol shadows" is on by default, works with the Chat mod
    /// off (the settings window must not say it waits for the mod), and has its own heading.
    #[test]
    fn symbol_shadows_is_on_by_default_and_applies_with_chat_off() {
        let option = DESCRIPTOR.option(SYMBOL_SHADOWS).expect("the option");
        assert_eq!(option.default, OptionValue::Bool(true));
        assert!(matches!(option.kind, OptionKind::Toggle));
        assert!(
            !DESCRIPTOR.default_enabled,
            "Chat itself stays off by default"
        );
        assert!(applies_while_off(DESCRIPTOR.id, SYMBOL_SHADOWS));
        let group = groups_for(DESCRIPTOR.id)
            .find(|group| group.options.contains(&SYMBOL_SHADOWS))
            .expect("a heading");
        assert_eq!(group.title, "All text");
        let index = DESCRIPTOR.option_index(SYMBOL_SHADOWS).expect("index");
        assert!(settings_layout(&DESCRIPTOR).options().any(|i| i == index));
    }
}
