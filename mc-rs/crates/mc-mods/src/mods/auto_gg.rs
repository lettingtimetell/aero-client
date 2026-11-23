//! Auto GG: an optional match-end chat reply, with an editable message and delay.
//!
//! The client recognizes server match-end messages and titles and sends the user's
//! message, which defaults to `gg`. Actual sending follows the normal chat input rules.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 2] = [
    OptionDescriptor {
        id: "message",
        label: "Message",
        kind: OptionKind::Text {
            max_len: crate::mods::chat_macros::TEXT_MAX,
        },
        default: OptionValue::Text(std::borrow::Cow::Borrowed("gg")),
        description: "Your chat message or /command. Defaults to gg.",
    },
    OptionDescriptor {
        id: "delay_ms",
        label: "Send delay",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 3000.0,
            step: 100.0,
            unit: " ms",
            decimals: 0,
        },
        default: OptionValue::Float(500.0),
        description: "How long to wait after recognizing the end of a match.",
    },
];

/// The catalog entry. Existing installations leave automatic chat replies disabled.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "auto_gg",
    name: "Auto GG",
    category: Category::Interface,
    description: "Sends your message when a match ends, recognized from the server's end-of-match \
                  messages or titles.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
